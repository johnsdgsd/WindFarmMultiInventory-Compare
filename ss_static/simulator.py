from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass, field

import numpy as np

TERM_NAMES = ("采购", "持有", "停机", "主动调拨", "被动调拨")


@dataclass
class StepResult:
    q_out: np.ndarray
    y: np.ndarray
    I_out: np.ndarray
    S_end: np.ndarray
    n_out: np.ndarray
    F_new: np.ndarray
    W_in: np.ndarray          # 未 clamp，与模型成本口径一致
    W_out: np.ndarray
    terms: np.ndarray         # (P,5)
    cost: np.ndarray          # (P,)
    N_counts: np.ndarray
    tau: np.ndarray           # 净主动调拨（正=净流入）
    taup: np.ndarray
    taum: np.ndarray


@dataclass
class SimulationResult:
    arrays: dict
    totals: np.ndarray
    term_totals: np.ndarray      # (P,5)：即使 record=False 也累计
    q0: np.ndarray
    T: int
    P: int
    policy_params: dict
    meta: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(np.mean(self.totals))

    @property
    def std(self) -> float:
        return float(np.std(self.totals, ddof=1)) if self.P > 1 else 0.0

    @property
    def se(self) -> float:
        return self.std / np.sqrt(self.P) if self.P > 0 else float("nan")

    @property
    def ci95(self) -> float:
        return 1.959963984540054 * self.se


def generate_scene_paths(n_paths: int, seed: int, chunk_size: int,
                         n_scenes: int, T: int) -> np.ndarray:
    """与 simulate.py 相同口径：块种子 = seed + k，每块独立 rng。"""
    if n_paths < 1 or chunk_size < 1 or n_scenes < 1 or T < 1:
        raise ValueError("n_paths/chunk_size/n_scenes/T 必须为正")
    out = np.empty((n_paths, T), dtype=np.int16)
    done = 0
    blk = 0
    while done < n_paths:
        n = min(chunk_size, n_paths - done)
        rng = np.random.default_rng(int(seed) + blk)
        for i in range(n):
            out[done + i] = rng.integers(0, n_scenes, size=T)
        done += n
        blk += 1
    return out


def round_repair_vec(n_raw: np.ndarray, N: np.ndarray, digits: int,
                     repair_tol: float) -> np.ndarray:
    """向量化版 round_repair_n：round 后只修正 ≤ repair_tol 的 Σn 超额。

    与 code/subproblem.round_repair_n 的语义一致：超额从最大分量开始扣，
    不缩放；超过 repair_tol 直接报错。超额只有 1e-4 量级，最多 G 次即可扣完。
    """
    nr = np.round(np.asarray(n_raw, dtype=float), int(digits)).copy()
    N = np.asarray(N, dtype=float)
    excess = nr.sum(axis=2) - N[None, :]
    if np.any(excess > float(repair_tol) + 1e-12):
        r, p = np.unravel_index(np.argmax(excess), excess.shape)
        raise ValueError(
            f"路径 {p} 仓库 {r + 1} 期末 Σn 超机队 "
            f"{excess[r, p]:.3e} > repair_tol={repair_tol:g}")
    P, N_R, G = nr.shape
    for r in range(N_R):
        for _ in range(G):
            rows = np.nonzero(excess[:, r] > 1e-12)[0]
            if rows.size == 0:
                break
            j = np.argmax(nr[rows, r, :], axis=1)
            take = np.minimum(nr[rows, r, j], excess[rows, r])
            nr[rows, r, j] -= take
            excess[rows, r] -= take
        if np.any(excess[:, r] > 1e-9):
            p = int(np.argmax(excess[:, r]))
            raise ValueError(
                f"路径 {p} 仓库 {r + 1} 的 Σn 超额无法从非负分量扣除")
    if float(nr.min()) < -1e-9:
        raise ValueError("round_repair 后出现负的 n 分量")
    return nr


class Simulator:
    """纯 NumPy 闭环规则仿真器；不建 COPT 模型。"""

    def __init__(self, prob):
        self.prob = prob
        self.T = int(prob.T)
        self.N_R = int(prob.N_R)
        self.G = int(prob.G)
        self.L = int(prob.L)
        self.N = np.asarray(prob.N, dtype=float)
        self.Qmax = np.asarray(prob.Qmax, dtype=float)
        self.Imax = np.asarray(prob.Imax, dtype=float)
        self.C = float(prob.C)
        self.h = float(prob.h)
        self.Cs = float(prob.Cs)
        self.atc = float(getattr(prob, "atc", 0.0))
        self.tc_route = np.asarray(prob.tc_route, dtype=float)
        self.n0 = np.asarray(prob.n0, dtype=float)
        self.I0 = np.asarray(prob.I0, dtype=float)
        self.o0 = np.asarray(prob.o0, dtype=float)
        self.digits = int(prob.state_round_digits)
        self.repair_tol = float(prob.state_repair_tol)
        self.theta_pool = np.stack(
            [np.asarray(th, dtype=float) for _, th in prob.pool], axis=0)
        self.layout = prob.layout
        self.ub = self.layout.upper(prob)

    # ------------------------------------------------------------------ 单期
    def step(self, t: int, I, o, n, theta, policy) -> StepResult:
        I = np.asarray(I, dtype=float)
        o = np.asarray(o, dtype=float)
        n = np.asarray(n, dtype=float)
        theta = np.asarray(theta, dtype=float)
        P = I.shape[0]
        if I.shape != (P, self.N_R):
            raise ValueError(f"I shape {I.shape}")
        if o.shape != (P, self.L - 1, self.N_R):
            raise ValueError(f"o shape {o.shape}")
        if n.shape != (P, self.N_R, self.G):
            raise ValueError(f"n shape {n.shape}")
        if theta.shape != (P, self.N_R, self.G, self.G + 1):
            raise ValueError(f"theta shape {theta.shape}")

        N_counts = n[:, :, :, None] * theta
        F_new = N_counts[:, :, :, self.G].sum(axis=2)
        W_in = self.N[None, :] - n.sum(axis=2)
        W_in_pos = np.maximum(W_in, 0.0)

        q_out = np.asarray(policy.order(t, I, o, n), dtype=float)
        if q_out.shape != (P, self.N_R):
            raise ValueError(f"policy.order 返回 shape {q_out.shape}")
        # 超容量/上界直接截断，保证 I + Σo + q <= Imax 且 q <= Qmax。
        q_out = np.clip(q_out, 0.0, self.Qmax[None, :])
        q_out = np.minimum(
            q_out, np.maximum(self.Imax[None, :]
                              - I - o.sum(axis=1), 0.0))
        if np.any(q_out > self.Qmax[None, :] + 1e-9):
            raise ValueError("policy.order 返回的 q 越界 [0,Qmax]")

        D = W_in_pos + F_new
        y = np.asarray(policy.fulfill(t, I, D), dtype=float)
        if y.shape != (P, self.N_R, self.N_R):
            raise ValueError(f"policy.fulfill 返回 shape {y.shape}")
        if np.any(y < -1e-9):
            raise ValueError("policy.fulfill 返回负的 y")

        Y = y.sum(axis=2)
        S_end = I - y.sum(axis=1) + o[:, 0, :]

        n_raw = np.zeros((P, self.N_R, self.G), dtype=float)
        n_raw[:, :, 0] = Y + N_counts[:, :, 0, 0]
        for j in range(1, self.G):
            n_raw[:, :, j] = N_counts[:, :, :, j].sum(axis=2)
        n_out = round_repair_vec(n_raw, self.N, self.digits, self.repair_tol)
        W_out = self.N[None, :] - n_out.sum(axis=2)

        # 下一期在途：q_t 在到货后进入最后一层；没有 q_in。
        o_next = np.empty_like(o)
        if self.L > 2:
            o_next[:, :self.L - 2, :] = o[:, 1:, :]
        o_next[:, self.L - 2, :] = q_out

        tau = np.asarray(
            policy.active_transfer(
                t, S_end, W_out, n_out, o_next, theta),
            dtype=float)
        if tau.shape != (P, self.N_R):
            raise ValueError(
                f"policy.active_transfer 返回 shape {tau.shape}")
        if np.any(np.abs(tau.sum(axis=1)) > 1e-7):
            raise ValueError("主动调拨 tau 不满足 Σtau=0")
        I_out = S_end + tau
        if np.any(I_out < -1e-9) or np.any(
                I_out > self.Imax[None, :] + 1e-9):
            raise ValueError("主动调拨后 I_out 越界")
        taup = np.maximum(tau, 0.0)
        taum = np.maximum(-tau, 0.0)

        # 与 docs/s,S.md 一致：q_t 在当期计采购费；末期 q_T 不计费。
        proc = (self.C * q_out.sum(axis=1) if t < self.T
                else np.zeros(P, dtype=float))
        hold = self.h / 2.0 * (I.sum(axis=1) + I_out.sum(axis=1))
        down = self.Cs * W_in.sum(axis=1)
        active = self.atc / 2.0 * (taup + taum).sum(axis=1)
        passive = (self.tc_route[None, :, :] * y).sum(axis=(1, 2))
        terms = np.stack([proc, hold, down, active, passive], axis=1)
        cost = terms.sum(axis=1)

        return StepResult(
            q_out=q_out, y=y, I_out=I_out, S_end=S_end, n_out=n_out,
            F_new=F_new, W_in=W_in, W_out=W_out, terms=terms, cost=cost,
            N_counts=N_counts, tau=tau, taup=taup, taum=taum)

    # ------------------------------------------------------------------ 仿真
    def simulate(self, policy, scene_paths, record: bool = False) -> SimulationResult:
        scene_paths = np.asarray(scene_paths)
        if scene_paths.ndim != 2 or scene_paths.shape[1] != self.T:
            raise ValueError(f"scene_paths shape 必须是 (P,{self.T})，实际 {scene_paths.shape}")
        if not np.issubdtype(scene_paths.dtype, np.integer):
            raise TypeError("scene_paths 必须是整数场景号")
        if scene_paths.min(initial=0) < 0 or scene_paths.max(initial=-1) >= len(self.theta_pool):
            raise ValueError("scene_paths 含越界场景号")
        P = scene_paths.shape[0]
        if P < 1:
            raise ValueError("至少 1 条路径")
        if int(policy.T) != self.T or int(policy.N_R) != self.N_R:
            raise ValueError("policy 与 prob 的 T/N_R 不一致")

        I = np.tile(self.I0[None, :], (P, 1))
        o = np.tile(self.o0[None, :, :], (P, 1, 1))
        n = np.tile(self.n0[None, :, :], (P, 1, 1))
        # s,S 没有虚拟阶段 q0；第一期由 t=1 的决策直接订货。
        q0 = np.zeros((P, self.N_R), dtype=float)

        arrays: dict = {}
        if record:
            arrays = {
                "x_in": np.zeros((P, self.T, self.layout.dim), dtype=float),
                "q": np.zeros((P, self.T, self.N_R), dtype=float),
                "taup": np.zeros((P, self.T, self.N_R), dtype=float),
                "taum": np.zeros((P, self.T, self.N_R), dtype=float),
                "S_end": np.zeros((P, self.T, self.N_R), dtype=float),
                "y": np.zeros((P, self.T, self.N_R, self.N_R), dtype=float),
                "n_out": np.zeros((P, self.T, self.N_R, self.G), dtype=float),
                "F": np.zeros((P, self.T, self.N_R), dtype=float),
                "W_in": np.zeros((P, self.T, self.N_R), dtype=float),
                "W_out": np.zeros((P, self.T, self.N_R), dtype=float),
                "cost": np.zeros((P, self.T), dtype=float),
                "terms": np.zeros((P, self.T, 5), dtype=float),
                "theta": np.zeros((P, self.T), dtype=float),
                "scene": np.asarray(scene_paths, dtype=np.int16).copy(),
            }

        totals = np.zeros(P, dtype=float)
        term_totals = np.zeros((P, 5), dtype=float)
        for t in range(1, self.T + 1):
            theta = self.theta_pool[scene_paths[:, t - 1]]
            st = self.step(t, I, o, n, theta, policy)
            totals += st.cost
            term_totals += st.terms
            if record:
                k = t - 1
                x = np.zeros((P, self.layout.dim), dtype=float)
                x[:, self.layout.off_I:self.layout.off_I + self.layout.n_I] = I
                x[:, self.layout.off_o:self.layout.off_o + self.layout.n_o] = \
                    o.reshape(P, -1)
                x[:, self.layout.off_n:self.layout.off_n + self.layout.n_n] = \
                    n.reshape(P, -1)
                # 没有 q_in；主布局的 q 槽留 0（审计时另做映射）。
                x[:, self.layout.off_q:self.layout.off_q + self.layout.n_q] = 0.0
                arrays["x_in"][:, k, :] = x
                arrays["q"][:, k, :] = st.q_out
                arrays["taup"][:, k, :] = st.taup
                arrays["taum"][:, k, :] = st.taum
                arrays["S_end"][:, k, :] = st.S_end
                arrays["y"][:, k, :, :] = st.y
                arrays["n_out"][:, k, :, :] = st.n_out
                arrays["F"][:, k, :] = st.F_new
                arrays["W_in"][:, k, :] = st.W_in
                arrays["W_out"][:, k, :] = st.W_out
                arrays["cost"][:, k] = st.cost
                arrays["terms"][:, k, :] = st.terms
                arrays["scene"][:, k] = np.asarray(scene_paths[:, k], dtype=np.int16)

            # 状态转移：o_out[l]=o_in[l+1]；最后一层 = 本期 q_t。
            o_out = np.empty_like(o)
            if self.L > 2:
                o_out[:, :self.L - 2, :] = o[:, 1:, :]
            o_out[:, self.L - 2, :] = st.q_out
            I = st.I_out
            o = o_out
            n = st.n_out

        if record:
            arrays["totals"] = totals.copy()
        return SimulationResult(
            arrays=arrays, totals=totals, term_totals=term_totals,
            q0=q0, T=self.T, P=P,
            policy_params=policy.params(),
            meta={"scene_paths_hash": scene_hash(scene_paths)})

    # ------------------------------------------------------- 单状态（测试用）
    def step_from_state(self, policy, t: int, x, scene_idx: int) -> StepResult:
        x = np.asarray(x, dtype=float).ravel()
        Ly = self.layout
        I = Ly.I(x)[None, :]
        o = Ly.o(x)[None, :, :]
        n = Ly.n(x)[None, :, :]
        theta = self.theta_pool[int(scene_idx)][None, :, :, :]
        return self.step(t, I, o, n, theta, policy)


def scene_hash(scene_paths: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(scene_paths).tobytes()).hexdigest()


def out_state_from_step(layout, x_in, st: StepResult, p: int = 0) -> np.ndarray:
    """s,S 期末状态：o_out 最后一层是本期 q_t；没有 q_in。"""
    xo = np.empty(layout.dim)
    xo[:layout.n_I] = st.I_out[p]
    o = np.empty((layout.L - 1, layout.N_R))
    for l in range(layout.L - 2):
        o[l] = layout.o(x_in)[l + 1]
    o[layout.L - 2] = st.q_out[p]
    xo[layout.off_o:layout.off_o + layout.n_o] = o.ravel()
    xo[layout.off_n:layout.off_n + layout.n_n] = st.n_out[p].ravel()
    xo[layout.off_q:layout.off_q + layout.n_q] = 0.0
    return xo


def audit_result(prob, res: SimulationResult, *, tol: float = 1e-6,
                 state_tol: float = 1e-4, max_points: int | None = None):
    """逐 (路径,期) 调用模型自己的 check_stage；返回 (audit, bad)。"""
    from code.sim_analysis import (CHECKS, STATE_NUMERIC_CHECKS, chain_dev,
                                   check_stage, out_state)

    if not res.arrays:
        raise ValueError("audit 需要 record=True 的仿真结果")
    Ly = prob.layout
    ub = Ly.upper(prob)
    T = res.T
    P = res.P
    audit = {c: 0.0 for c in CHECKS}
    audit["q.下"] = 0.0
    audit["q.上"] = 0.0
    n = 0
    for p in range(P):
        for t in range(T):
            if max_points is not None and n >= max_points:
                break
            q_t = np.asarray(res.arrays["q"][p, t], dtype=float)
            # 审计映射：s,S 无 q_in；把本期 q_t 放到审计副本的 q 槽，
            # 使主模型 out_state/check_stage 正好生成下一期在途 [o2,q_t]。
            x = np.asarray(res.arrays["x_in"][p, t], dtype=float).copy()
            x[Ly.off_q:Ly.off_q + Ly.n_q] = q_t
            rec = {
                # out_state 的 q 槽是下一期 q_in；新时序没有 q_in，置 0。
                "q": np.zeros_like(q_t),
                "taup": res.arrays["taup"][p, t],
                "taum": res.arrays["taum"][p, t],
                "S_end": res.arrays["S_end"][p, t],
                "y": res.arrays["y"][p, t],
                "n_out": res.arrays["n_out"][p, t],
                "F_new": res.arrays["F"][p, t],
                "W_in": res.arrays["W_in"][p, t],
                "W_out": res.arrays["W_out"][p, t],
                "scene": int(res.arrays["scene"][p, t]),
            }
            rec["I_out"] = (np.asarray(rec["S_end"], float)
                            + np.asarray(rec["taup"], float)
                            - np.asarray(rec["taum"], float))
            xo = out_state(Ly, x, rec)
            a = check_stage(
                prob, Ly, x, xo, rec["q"], rec["taup"], rec["taum"],
                rec["S_end"], rec["y"], rec["scene"], ub, F_new=rec["F_new"])
            # 当前期订货容量：I + Σo + q_t <= Imax
            I_in = Ly.I(x)
            o_in = Ly.o(x)
            a["cap"] = float(max(
                a["cap"],
                np.maximum(I_in + o_in.sum(axis=0) + q_t
                           - np.asarray(prob.Imax), 0.0).max()))
            if t < T - 1:
                a["链.状态"] = chain_dev(Ly, x, res.arrays["x_in"][p, t + 1], rec)
            a["q.下"] = float(max(0.0, -q_t.min()))
            a["q.上"] = float(max(0.0, (q_t - np.asarray(prob.Qmax)).max()))
            for c, v in a.items():
                audit[c] = max(audit[c], float(v))
            n += 1
        if max_points is not None and n >= max_points:
            break
    bad = {}
    for c, v in audit.items():
        lim = max(tol, state_tol) if c in STATE_NUMERIC_CHECKS else tol
        if v > lim:
            bad[c] = v
    return audit, bad


def cost_terms_max_dev(res: SimulationResult) -> float:
    if not res.arrays:
        return float("nan")
    return float(np.max(np.abs(res.arrays["terms"].sum(axis=2) - res.arrays["cost"])))
