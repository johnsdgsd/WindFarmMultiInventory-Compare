"""PPO 环境：tau 先于 q 的新时序，自包含状态转移与审计。

每期（t=1..T）时序::

    s_t=(I,n,o1,o2)
      -> 选主动调拨 tau_t（净量，Σtau=0）
      -> 选订货 q_t（0..Qmax）
      -> 抽样 Theta_t（训练时在线抽样；评估可用基线 scene）
      -> 规则被动调拨 y_t（先本地，再按 tc_route/priority 全局贪心）
      -> 转移
           I_{t+1} = I_t + tau_t - Σ_dest y[dest,r,t] + o1_t
           o1_{t+1}=o2_t, o2_{t+1}=q_t
           n_{t+1,0}=Y+N_{0,0}; n_{t+1,j}=Σ_i N_{i,j}

约束（动作投影后逐项断言，不静默违反）::

    -I <= tau <= Imax - I - Σo          # 含期初容量；Σtau=0
    tau_r <= Σ_{m!=r} I_m               # 由 Σtau=0 与 -I<=tau 自动满足
    0 <= q <= min(Qmax, Imax - (I+tau) - Σo)
    I_out + o1_next + o2_next <= Imax   # 由 cap_init 自动满足

被动调拨可用库存按新时序为 ``I+tau+o1``，需求 ``D=W_in+F_new``。
成本口径与主线一致（万元）::

    C q + h/2 (I_t+I_{t+1}) + Cs W_in
      + atc/2 Σ|tau| + Σ tc_route*y

奖励 = -reward_scale * 阶段成本；env 同时返回未缩放的实际成本。
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math
from dataclasses import dataclass

import numpy as np

TERM_NAMES = ("采购", "持有", "停机", "主动调拨", "被动调拨")


# ---------------------------------------------------------------- 转移矩阵
def poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def transition_matrix(lam: float, G: int, p_digits: int | None = 3) -> np.ndarray:
    """G×(G+1) 上三角转移矩阵；与主模型 / rolling_horizon.scenarios 同口径。"""
    lam = float(lam)
    if lam < 0:
        raise ValueError(f"lambda 必须非负，实际 {lam}")
    p = np.array([poisson_pmf(lam, k) for k in range(G + 1)], dtype=float)
    P = np.zeros((G, G + 1), dtype=float)
    for i in range(G):
        off = []
        for j in range(i + 1, G + 1):
            if j < G:
                off.append(p[j - i])
            else:
                off.append(1.0 - float(p[: G - i].sum()))
        off = np.asarray(off, dtype=float)
        if p_digits is not None:
            off = np.round(off, int(p_digits))
            s = float(off.sum())
            if s > 1.0 + 1e-12:
                off = off / s
        P[i, i + 1:] = off
        P[i, i] = 1.0 - float(off.sum())
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    if np.any(rs <= 0):
        raise ValueError("转移矩阵出现空行")
    return P / rs


def _cov_factor(beta_cov, z_bar, z_r) -> float:
    return float(np.exp(np.dot(beta_cov, np.asarray(z_r) - np.asarray(z_bar))))


class ThetaSampler:
    """在线 Theta 抽样器；复制主模型 problem_data.sample_pool 的单期数学。

    不建立 N×T 优化池：训练每步独立抽一个 Theta；同一 seed 的调用序列可复现。
    """

    def __init__(self, prob, seed: int | None = None):
        self.prob = prob
        self.rng = np.random.default_rng(None if seed is None else int(seed))
        cov = prob.cov
        self.mu_T = float(cov["mu_T"])
        self.sigma_T = float(cov["sigma_T"])
        self.lam_cov = float(cov["lambda"])
        self.alpha_L = float(cov["alpha_L"])
        self.beta_L = float(cov["beta_L"])
        self.lambda0 = float(prob.lambda0)
        self.beta_cov = np.asarray(prob.beta_cov, dtype=float)
        self.z_bar = np.asarray(prob.z_bar, dtype=float)
        self.dirichlet_s = float(prob.dirichlet_s)
        self.p_digits = int(prob.p_digits)
        self.G = int(prob.G)
        self.N_R = int(prob.N_R)

    def sample(self) -> np.ndarray:
        rng = self.rng
        N_R, G = self.N_R, self.G
        z = np.empty((3, N_R), dtype=float)
        z[0] = rng.normal(self.mu_T, self.sigma_T, size=N_R)
        z[1] = rng.poisson(self.lam_cov, size=N_R)
        z[2] = 100.0 * rng.beta(self.alpha_L, self.beta_L, size=N_R)
        theta = np.zeros((N_R, G, G + 1), dtype=float)
        for r in range(N_R):
            f = _cov_factor(self.beta_cov, self.z_bar, z[:, r])
            P = transition_matrix(self.lambda0 * f, G, self.p_digits)
            for i in range(G):
                cols = [j for j in range(i, G + 1) if P[i, j] > 0.0]
                if not cols:
                    raise ValueError(f"P 第 {i} 行无正支撑")
                if len(cols) == 1:
                    vals = np.array([1.0])
                else:
                    alpha = np.array([self.dirichlet_s * P[i, j]
                                      for j in cols], dtype=float)
                    vals = rng.dirichlet(alpha)
                theta[r, i, cols] = vals
        rs = theta.sum(axis=2)
        if np.max(np.abs(rs - 1.0)) > 1e-12:
            raise ValueError("Theta 行和偏离 1")
        if np.any(theta < 0):
            raise ValueError("Theta 出现负值")
        for i in range(G):
            if i and np.any(theta[:, i, :i] != 0.0):
                raise ValueError("Theta 下三角非零")
        return theta


# ---------------------------------------------------------------- 动作投影
def project_box_sum(raw, lo, hi, target: float = 0.0,
                    iters: int = 80) -> tuple[np.ndarray, float]:
    """把 raw 投影到 {lo<=x<=hi, Σx=target}（欧氏投影）。

    用对偶变量 lambda 的单调性做二分：x(lambda)=clip(raw-lambda,lo,hi)。
    """
    raw = np.asarray(raw, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if not (raw.size == lo.size == hi.size):
        raise ValueError("project_box_sum 维度不一致")
    tol = 1e-12
    if np.any(lo > hi + tol):
        raise ValueError(f"投影箱不可行: min(hi-lo)={np.min(hi - lo):.3e}")
    s_lo = float(lo.sum())
    s_hi = float(hi.sum())
    if s_lo > target + 1e-9 or s_hi < target - 1e-9:
        raise ValueError(
            f"投影和不可行: Σlo={s_lo:.6f}, Σhi={s_hi:.6f}, target={target}")

    # f(lam) 单调不增；a 端 f>=target，b 端 f<=target
    a = float(np.min(raw - hi))
    b = float(np.max(raw - lo))
    fa = float(np.clip(raw - a, lo, hi).sum())
    fb = float(np.clip(raw - b, lo, hi).sum())
    if fa < target - 1e-9 or fb > target + 1e-9:
        # 浮点边界兜底：把区间向外扩
        a -= 1.0 + abs(a)
        b += 1.0 + abs(b)
        fa = float(np.clip(raw - a, lo, hi).sum())
        fb = float(np.clip(raw - b, lo, hi).sum())
    if fa < target - 1e-9 or fb > target + 1e-9:
        raise ValueError(f"投影二分区间失败: f(a)={fa}, f(b)={fb}, target={target}")
    for _ in range(int(iters)):
        mid = 0.5 * (a + b)
        f = float(np.clip(raw - mid, lo, hi).sum())
        if f > target:
            a = mid
        else:
            b = mid
    lam = 0.5 * (a + b)
    x = np.clip(raw - lam, lo, hi)

    # 浮点残差修正：按可调裕度比例分配，最多几轮
    for _ in range(8):
        resid = target - float(x.sum())
        if abs(resid) <= 1e-13:
            break
        slack = (hi - x) if resid > 0 else (x - lo)
        total = float(slack.sum())
        if total <= 1e-15:
            break
        x = np.clip(x + resid * slack / total, lo, hi)
    return x, float(np.max(np.abs(x - raw)))


def priority_fulfillment(avail, demand, edges) -> tuple[np.ndarray, np.ndarray]:
    """先本地满足，再按 (tc_route, dest, source) 升序全局贪心跨仓调拨。

    y[dest, source]；edges 为预排序的 [(cost, dest, source), ...]。
    """
    avail = np.asarray(avail, dtype=float).reshape(-1)
    demand = np.asarray(demand, dtype=float).reshape(-1)
    n = avail.size
    if demand.size != n:
        raise ValueError("avail/demand 维度不一致")
    y = np.zeros((n, n), dtype=float)
    a = avail.copy()
    rem = demand.copy()
    diag = np.minimum(rem, a)
    idx = np.arange(n)
    y[idx, idx] = diag
    a -= diag
    rem -= diag
    for _, r, m in edges:
        if rem[r] <= 1e-12 or a[m] <= 1e-12:
            continue
        take = min(rem[r], a[m])
        y[r, m] += take
        a[m] -= take
        rem[r] -= take
    return y, rem


def round_repair_state(n_raw, N, digits: int, repair_tol: float) -> np.ndarray:
    """单状态 n 的 round + 微小 Σn 超额修正；与主模型同语义。"""
    nr = np.round(np.asarray(n_raw, dtype=float), int(digits)).copy()
    N = np.asarray(N, dtype=float)
    excess = nr.sum(axis=1) - N
    if np.any(excess > float(repair_tol) + 1e-12):
        r = int(np.argmax(excess))
        raise ValueError(
            f"仓库 {r + 1} 期末 Σn 超机队 {excess[r]:.3e} "
            f"> repair_tol={repair_tol:g}")
    for r in range(nr.shape[0]):
        for _ in range(nr.shape[1]):
            if excess[r] <= 1e-12:
                break
            j = int(np.argmax(nr[r]))
            take = min(nr[r, j], excess[r])
            nr[r, j] -= take
            excess[r] -= take
        if excess[r] > 1e-9:
            raise ValueError(f"仓库 {r + 1} 的 Σn 超额无法扣除")
    if float(nr.min()) < -1e-9:
        raise ValueError("round_repair 后出现负的 n 分量")
    return nr


# ---------------------------------------------------------------- 环境
@dataclass
class StepInfo:
    cost: float
    terms: np.ndarray
    tau: np.ndarray
    q: np.ndarray
    y: np.ndarray
    I_next: np.ndarray
    o_next: np.ndarray
    n_next: np.ndarray
    F_new: np.ndarray
    W_in: np.ndarray
    W_out: np.ndarray
    D: np.ndarray
    theta: np.ndarray
    projection: dict
    cap_init_max: float
    cap_next_max: float
    episode_cost: float | None = None
    episode_return: float | None = None


class TauFirstEnv:
    """单状态 PPO 环境（新时序 tau -> q -> Theta -> y）。

    动作 10 维：前 5 维乘 Imax 得 tau 的原始量，后 5 维乘 Qmax 得 q 的原始量；
    env 内部投影后才是实际动作。投影量写入 info["projection"]。
    """

    def __init__(self, prob, *, reward_scale: float = 1e-3,
                 terminal_q_zero: bool = True,
                 seed: int | None = None, theta_path=None):
        self.prob = prob
        self.T = int(prob.T)
        self.N_R = int(prob.N_R)
        self.G = int(prob.G)
        self.L = int(prob.L)
        if self.L != 3:
            raise ValueError(f"本环境按 L=3（o1/o2 两层）实现，实际 L={self.L}")
        self.K = self.L - 1
        self.N = np.asarray(prob.N, dtype=float)
        self.Qmax = np.asarray(prob.Qmax, dtype=float)
        self.Imax = np.asarray(prob.Imax, dtype=float)
        self.C = float(prob.C)
        self.h = float(prob.h)
        self.Cs = float(prob.Cs)
        self.atc = float(getattr(prob, "atc", 0.0))
        self.tc_route = np.asarray(prob.tc_route, dtype=float)
        # 被动调拨优先级列表：prob.priority[r+1] 是 1-based，转成 0-based。
        if not hasattr(prob, "priority"):
            raise ValueError("problem data 缺少 priority 表")
        self.priority = {
            r: [int(m) - 1 for m in prob.priority[r + 1]]
            for r in range(self.N_R)
        }
        for r in range(self.N_R):
            if sorted(self.priority[r]) != list(range(self.N_R)):
                raise ValueError(
                    f"priority[{r + 1}] 不是 0..{self.N_R - 1} 的排列: "
                    f"{self.priority[r]}")
        self.n0 = np.asarray(prob.n0, dtype=float)
        self.I0 = np.asarray(prob.I0, dtype=float)
        self.o0 = np.asarray(prob.o0, dtype=float)
        if self.n0.shape != (self.N_R, self.G):
            raise ValueError(f"n0 shape {self.n0.shape}")
        if self.o0.shape != (self.K, self.N_R):
            raise ValueError(f"o0 shape {self.o0.shape}")
        self.digits = int(prob.state_round_digits)
        self.repair_tol = float(prob.state_repair_tol)
        self.fleet_tol = float(getattr(prob, "fleet_tol", 0.0))
        self.reward_scale = float(reward_scale)
        self.terminal_q_zero = bool(terminal_q_zero)
        self.sampler = ThetaSampler(prob, seed=seed) if theta_path is None \
            else None
        self.edges = sorted(
            (float(self.tc_route[r, m]), r, m)
            for r in range(self.N_R) for m in range(self.N_R) if r != m)
        # 全局贪心按 (tc_route, dest, source) 排序；对每个目的地，source
        # 的先后必须与 prob.priority 列表一致（本地 self 已在前面单独处理）。
        for r in range(self.N_R):
            got = [m for _, rr, m in self.edges if rr == r]
            expected = [m for m in self.priority[r] if m != r]
            if got != expected:
                raise ValueError(
                    f"仓库 {r + 1} 的跨仓顺序 {got} 与 priority "
                    f"{expected} 不一致；请检查 tc_route 是否由 priority 生成")
        # 状态 + 归一化时间 t/(T-1)；PPO 需要显式时间特征
        self.obs_dim = self.N_R + self.N_R * self.G + 2 * self.N_R + 1
        self.act_dim = 2 * self.N_R
        if self.obs_dim != 36:
            raise ValueError(f"T60 口径下 obs_dim 应为 36，实际 {self.obs_dim}")
        self.theta_path = None
        self._theta_path_input = theta_path
        self.reset(theta_path=theta_path)

    # ------------------------------------------------------------- 状态
    def reset(self, theta_path=None) -> np.ndarray:
        self.I = self.I0.copy()
        self.o1 = self.o0[0].copy()
        self.o2 = self.o0[1].copy()
        self.n = self.n0.copy()
        self.t = 0
        self.episode_cost = 0.0
        self.episode_return = 0.0
        if theta_path is None:
            theta_path = self._theta_path_input
        if theta_path is not None:
            th = np.asarray(theta_path, dtype=float)
            if th.shape != (self.T, self.N_R, self.G, self.G + 1):
                raise ValueError(
                    f"theta_path shape {th.shape} != "
                    f"{(self.T, self.N_R, self.G, self.G + 1)}")
            self.theta_path = th
        else:
            self.theta_path = None
        return self.observation()

    def set_state(self, I, o=None, n=None, *, o1=None, o2=None,
                  t: int = 0) -> np.ndarray:
        """测试/诊断用：直接设置状态（不检查 episode 预算）。"""
        self.I = np.asarray(I, dtype=float).reshape(self.N_R).copy()
        if o is not None:
            o = np.asarray(o, dtype=float)
            if o.shape != (self.K, self.N_R):
                raise ValueError(f"o shape {o.shape}")
            self.o1 = o[0].copy()
            self.o2 = o[1].copy()
        if o1 is not None:
            self.o1 = np.asarray(o1, dtype=float).reshape(self.N_R).copy()
        if o2 is not None:
            self.o2 = np.asarray(o2, dtype=float).reshape(self.N_R).copy()
        if n is not None:
            n = np.asarray(n, dtype=float)
            if n.shape != (self.N_R, self.G):
                raise ValueError(f"n shape {n.shape}")
            self.n = n.copy()
        self.t = int(t)
        self.episode_cost = 0.0
        self.episode_return = 0.0
        return self.observation()

    def observation(self) -> np.ndarray:
        time_frac = self.t / max(1, self.T - 1)
        obs = np.concatenate([
            self.I / self.Imax,
            (self.n / self.N[:, None]).reshape(-1),
            self.o1 / self.Qmax,
            self.o2 / self.Qmax,
            np.asarray([time_frac], dtype=float),
        ]).astype(np.float32)
        if obs.shape != (self.obs_dim,):
            raise RuntimeError(f"obs shape {obs.shape}")
        return obs

    # ------------------------------------------------------------- 单步
    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        if self.t >= self.T:
            raise RuntimeError("episode 已结束，请 reset")
        action = np.asarray(action, dtype=float).reshape(self.act_dim)
        I = self.I.copy()
        o1 = self.o1.copy()
        o2 = self.o2.copy()
        n = self.n.copy()
        o_sum = o1 + o2

        # ---- 状态导出（与 Theta 无关的部分）
        W_in_raw = self.N - n.sum(axis=1)
        if float(W_in_raw.min()) < -1e-6:
            raise ValueError(f"运行台数超过机队: W_in={W_in_raw}")
        W_in = np.maximum(W_in_raw, 0.0)

        # ---- 动作投影：tau 先，q 后
        tau_raw = action[:self.N_R] * self.Imax
        q_raw = action[self.N_R:] * self.Qmax
        hi_tau = self.Imax - I - o_sum
        if float(hi_tau.min()) < -1e-7:
            raise ValueError(
                f"状态在期初容量下不可行: I+Σo-Imax={I + o_sum - self.Imax}")
        hi_tau = np.maximum(hi_tau, 0.0)
        tau, tau_max_dev = project_box_sum(tau_raw, -I, hi_tau, target=0.0)
        src_viol = float(np.maximum(tau - (I.sum() - I), 0.0).max())
        q_upper = np.minimum(self.Qmax, self.Imax - (I + tau) - o_sum)
        if float(q_upper.min()) < -1e-7:
            raise ValueError(f"q 容量上界为负: {q_upper}")
        q_upper = np.maximum(q_upper, 0.0)
        q = np.clip(q_raw, 0.0, q_upper)
        terminal_zero = False
        if self.terminal_q_zero and self.t == self.T - 1:
            q = np.zeros_like(q)
            terminal_zero = True
        q_max_dev = float(np.max(np.abs(q - q_raw)))
        tau_l1 = float(np.abs(tau - tau_raw).sum())
        q_l1 = float(np.abs(q - q_raw).sum())

        # ---- 观测 Theta_t（tau/q 之后；策略未使用该信息）
        theta = (self.theta_path[self.t] if self.theta_path is not None
                 else self.sampler.sample())
        N_counts = n[:, :, None] * theta
        F_new = N_counts[:, :, self.G].sum(axis=1)
        D = W_in + F_new

        # ---- 被动调拨：可用 I+tau+o1；先本地再跨仓
        avail = I + tau + o1
        y, unmet = priority_fulfillment(avail, D, self.edges)
        source_out = y.sum(axis=0)
        dest_in = y.sum(axis=1)

        # ---- 状态转移
        I_next = I + tau - source_out + o1
        o1_next = o2.copy()
        o2_next = q.copy()
        n_raw = np.zeros((self.N_R, self.G), dtype=float)
        n_raw[:, 0] = dest_in + N_counts[:, 0, 0]
        for j in range(1, self.G):
            n_raw[:, j] = N_counts[:, :, j].sum(axis=1)
        n_next = round_repair_state(
            n_raw, self.N, self.digits, self.repair_tol)
        W_out = self.N - n_next.sum(axis=1)

        # ---- 成本（万元）
        proc = self.C * float(q.sum())
        hold = self.h / 2.0 * float(I.sum() + I_next.sum())
        down = self.Cs * float(W_in_raw.sum())
        active = self.atc / 2.0 * float(np.abs(tau).sum())
        passive = float((self.tc_route * y).sum())
        terms = np.array([proc, hold, down, active, passive], dtype=float)
        cost = float(terms.sum())
        reward = -self.reward_scale * cost

        # ---- 硬检查（不静默）
        neg = 1e-7
        if float(I_next.min()) < -neg:
            raise ValueError(f"I_next 出现负库存: {I_next}")
        if float((I_next - self.Imax).max()) > neg:
            raise ValueError(f"I_next 超 Imax: {I_next - self.Imax}")
        if float((I_next + o1_next + o2_next - self.Imax).max()) > neg:
            raise ValueError(
                f"下一期状态容量越界: "
                f"{I_next + o1_next + o2_next - self.Imax}")
        if abs(float(tau.sum())) > 1e-8:
            raise ValueError(f"Σtau 未归零: {tau.sum():.3e}")
        if float(src_viol) > 1e-9:
            raise ValueError(f"tau_src 违约: {src_viol:.3e}")
        if float(q.min()) < -1e-9 or float((q - self.Qmax).max()) > 1e-9:
            raise ValueError(f"q 越界: {q}")
        cap_init_max = float(np.max(I + tau + o_sum + q - self.Imax))
        cap_next_max = float(np.max(I_next + o1_next + o2_next - self.Imax))
        if cap_init_max > neg or cap_next_max > neg:
            raise ValueError(
                f"容量违约: cap_init={cap_init_max:.3e}, "
                f"cap_next={cap_next_max:.3e}")
        if float(y.min()) < -1e-12:
            raise ValueError("y 出现负值")
        if float(np.maximum(dest_in - D, 0.0).max()) > 1e-9:
            raise ValueError("被动调拨超过需求")
        if float(np.maximum(source_out - avail, 0.0).max()) > 1e-9:
            raise ValueError("被动调拨超过可用库存")

        # ---- 提交状态
        self.I = I_next
        self.o1 = o1_next
        self.o2 = o2_next
        self.n = n_next
        self.t += 1
        self.episode_cost += cost
        self.episode_return += reward
        done = self.t >= self.T

        projection = {
            "tau_raw": tau_raw,
            "tau": tau,
            "q_raw": q_raw,
            "q": q,
            "q_upper": q_upper,
            "tau_l1": tau_l1,
            "tau_max": tau_max_dev,
            "tau_raw_sum": float(tau_raw.sum()),
            "q_l1": q_l1,
            "q_max": q_max_dev,
            "terminal_q_zero": bool(terminal_zero),
            "src_violation": float(src_viol),
        }
        info = StepInfo(
            cost=cost, terms=terms, tau=tau, q=q, y=y,
            I_next=I_next, o_next=np.stack([o1_next, o2_next]),
            n_next=n_next, F_new=F_new, W_in=W_in_raw, W_out=W_out,
            D=D, theta=theta, projection=projection,
            cap_init_max=cap_init_max, cap_next_max=cap_next_max,
            episode_cost=self.episode_cost if done else None,
            episode_return=self.episode_return if done else None,
        )
        return self.observation(), reward, done, self._info_dict(info, leftover=unmet)

    def _info_dict(self, info: StepInfo, leftover: np.ndarray) -> dict:
        return {
            "cost": info.cost,
            "terms": info.terms,
            "tau": info.tau,
            "q": info.q,
            "y": info.y,
            "I_next": info.I_next,
            "o_next": info.o_next,
            "n_next": info.n_next,
            "F_new": info.F_new,
            "W_in": info.W_in,
            "W_out": info.W_out,
            "D": info.D,
            "theta": info.theta,
            "projection": info.projection,
            "cap_init_max": info.cap_init_max,
            "cap_next_max": info.cap_next_max,
            "unmet": leftover,
            "episode_cost": info.episode_cost,
            "episode_return": info.episode_return,
            "t": self.t,
        }


# ---------------------------------------------------------------- 审计
def _theta_of(prob, scene_idx: int) -> np.ndarray:
    return np.asarray(prob.pool[int(scene_idx)][1], dtype=float)


def audit_rollout(prob, arrays: dict, *, tol: float = 1e-6,
                  state_tol: float = 1e-4,
                  reward_scale: float = 1e-3) -> tuple[dict, dict]:
    """逐 (路径,期) 核对新时序状态转移、容量、动作约束和被动优先级。"""
    need = ("I_in", "o_in", "n_in", "q", "tau", "y", "I_out", "o_next",
            "n_out", "F", "W_in", "W_out", "cost", "terms", "reward",
            "scene")
    miss = [k for k in need if k not in arrays]
    if miss:
        raise KeyError(f"审计数组缺字段: {miss}")
    NR, G, K = int(prob.N_R), int(prob.G), int(prob.L) - 1
    P, T = arrays["cost"].shape
    edges = sorted(
        (float(prob.tc_route[r, m]), r, m)
        for r in range(NR) for m in range(NR) if r != m)
    Nv = np.asarray(prob.N, dtype=float)
    Imax = np.asarray(prob.Imax, dtype=float)
    Qmax = np.asarray(prob.Qmax, dtype=float)
    C, h, Cs = float(prob.C), float(prob.h), float(prob.Cs)
    atc = float(getattr(prob, "atc", 0.0))
    tc_route = np.asarray(prob.tc_route, dtype=float)
    audit = {c: 0.0 for c in (
        "pas", "priority", "local_first", "demand", "y_neg", "fleet", "dem",
        "theta_row", "结构零", "N链接", "F_new", "grade0", "grade", "W守恒",
        "send_def", "inv", "pipe1", "pipe2", "tau_bal", "tau_lo", "tau_hi",
        "tau_src", "q_lo", "q_hi", "cap_init", "cap_next", "界.下",
        "界.上", "链.I", "链.o1", "链.o2", "链.n", "terms_sum",
        "reward_identity")}
    for p in range(P):
        for t in range(T):
            I = np.asarray(arrays["I_in"][p, t], dtype=float)
            o = np.asarray(arrays["o_in"][p, t], dtype=float)
            nin = np.asarray(arrays["n_in"][p, t], dtype=float)
            q = np.asarray(arrays["q"][p, t], dtype=float)
            tau = np.asarray(arrays["tau"][p, t], dtype=float)
            y = np.asarray(arrays["y"][p, t], dtype=float)
            Iout = np.asarray(arrays["I_out"][p, t], dtype=float)
            on = np.asarray(arrays["o_next"][p, t], dtype=float)
            nout = np.asarray(arrays["n_out"][p, t], dtype=float)
            F = np.asarray(arrays["F"][p, t], dtype=float)
            Win = np.asarray(arrays["W_in"][p, t], dtype=float)
            Wout = np.asarray(arrays["W_out"][p, t], dtype=float)
            terms = np.asarray(arrays["terms"][p, t], dtype=float)
            cost = float(arrays["cost"][p, t])
            reward = float(arrays["reward"][p, t])
            th = _theta_of(prob, int(arrays["scene"][p, t]))

            Ncnt = nin[:, :, None] * th
            F_calc = Ncnt[:, :, G].sum(axis=1)
            W_calc = Nv - nin.sum(axis=1)
            D = np.maximum(W_calc, 0.0) + F_calc
            avail = I + tau + o[0]
            y_exp, _ = priority_fulfillment(avail, D, edges)
            source_out = y.sum(axis=0)
            dest_in = y.sum(axis=1)
            local_exp = np.minimum(D, avail)
            diag = np.diag(y)
            Y = dest_in
            good = Ncnt[:, :, :G].sum(axis=(1, 2))

            zero_dev = 0.0
            for i in range(G):
                if i:
                    zero_dev = max(zero_dev,
                                   float(np.max(np.abs(th[:, i, :i]))))
            row_dev = float(np.max(np.abs(Ncnt.sum(axis=2) - nin)))
            grade_dev = max(
                (float(np.max(np.abs(nout[:, j] - Ncnt[:, :, j].sum(axis=1))))
                 for j in range(1, G)), default=0.0)
            grade0_dev = float(np.max(np.abs(
                nout[:, 0] - Y - Ncnt[:, 0, 0])))
            d = {
                "pas": float(np.maximum(source_out - avail, 0).max()),
                "priority": float(np.max(np.abs(y - y_exp))),
                "local_first": float(np.max(np.abs(diag - local_exp))),
                "demand": float(np.maximum(Y - D, 0).max()),
                "y_neg": float(max(0.0, -y.min())),
                "fleet": float(np.maximum(nin.sum(axis=1)
                                          - (Nv + prob.fleet_tol), 0).max()),
                "dem": float(np.maximum(Y + good
                                        - (Nv + prob.fleet_tol), 0).max()),
                "theta_row": float(np.max(np.abs(th.sum(axis=2) - 1.0))),
                "结构零": zero_dev,
                "N链接": row_dev,
                "F_new": float(np.max(np.abs(F - F_calc))),
                "grade0": grade0_dev,
                "grade": grade_dev,
                "W守恒": float(np.max(np.abs(Wout - (Win + F_calc - Y)))),
                "send_def": float(np.max(np.abs(
                    Iout - (I + tau - source_out + o[0])))),
                "inv": float(np.max(np.abs(Iout - (I + tau - source_out
                                                   + o[0])))),
                "pipe1": float(np.max(np.abs(on[0] - o[1]))),
                "pipe2": float(np.max(np.abs(on[1] - q))),
                "tau_bal": float(abs(tau.sum())),
                "tau_lo": float(np.maximum(-(I + tau), 0).max()),
                "tau_hi": float(np.maximum(I + tau - Imax, 0).max()),
                "tau_src": float(np.maximum(
                    tau - (I.sum() - I), 0).max()),
                "q_lo": float(max(0.0, -q.min())),
                "q_hi": float(max(0.0, (q - Qmax).max())),
                "cap_init": float(np.maximum(
                    I + tau + o.sum(axis=0) + q - Imax, 0).max()),
                "cap_next": float(np.maximum(
                    Iout + on.sum(axis=0) - Imax, 0).max()),
                "界.下": float(max(0.0, -min(I.min(), o.min(), nin.min()))),
                "界.上": float(max(
                    np.maximum(I - Imax, 0).max(),
                    np.maximum(o - Qmax[None, :], 0).max(),
                    np.maximum(nin - Nv[:, None], 0).max())),
                "terms_sum": float(abs(terms.sum() - cost)),
                "reward_identity": float(abs(reward + reward_scale * cost)),
            }
            if t < T - 1:
                d["链.I"] = float(np.max(np.abs(
                    Iout - arrays["I_in"][p, t + 1])))
                d["链.o1"] = float(np.max(np.abs(
                    on - arrays["o_in"][p, t + 1])))
                d["链.o2"] = d["链.o1"]
                d["链.n"] = float(np.max(np.abs(
                    nout - arrays["n_in"][p, t + 1])))
            else:
                for k in ("链.I", "链.o1", "链.o2", "链.n"):
                    d[k] = 0.0
            for c, v in d.items():
                audit[c] = max(audit[c], float(v))
    state_checks = {"grade0", "grade", "W守恒"}
    bad = {}
    for c, v in audit.items():
        lim = max(tol, state_tol) if c in state_checks else tol
        if v > lim:
            bad[c] = v
    return audit, bad
