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
    W_in: np.ndarray
    W_out: np.ndarray
    o_next: np.ndarray        # (P,L-1,N_R)
    terms: np.ndarray
    cost: np.ndarray
    N_counts: np.ndarray
    tau: np.ndarray
    taup: np.ndarray
    taum: np.ndarray


@dataclass
class SimulationResult:
    arrays: dict
    totals: np.ndarray
    term_totals: np.ndarray
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


def initial_o(prob) -> np.ndarray:
    """初始在途状态 o_1..o_{L-1}，直接取 problem_data.o0。"""
    return np.asarray(prob.o0, dtype=float).reshape(prob.L - 1, prob.N_R)


class Simulator:
    """纯 NumPy 闭环仿真；状态 I,n,o_1..o_{L-1}。"""

    def __init__(self, prob):
        self.prob = prob
        self.T = int(prob.T)
        self.N_R = int(prob.N_R)
        self.G = int(prob.G)
        self.L = int(prob.L)
        self.K = self.L - 1
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
        self.o0 = initial_o(prob)
        self.digits = int(prob.state_round_digits)
        self.repair_tol = float(prob.state_repair_tol)
        self.theta_pool = np.stack(
            [np.asarray(th, dtype=float) for _, th in prob.pool], axis=0)
        self.layout = prob.layout
        self.ub = self.layout.upper(prob)

    def step(self, t: int, I, o, n, theta, policy) -> StepResult:
        I = np.asarray(I, dtype=float)
        o = np.asarray(o, dtype=float)
        n = np.asarray(n, dtype=float)
        theta = np.asarray(theta, dtype=float)
        P = I.shape[0]
        if I.shape != (P, self.N_R):
            raise ValueError(f"I shape {I.shape}")
        if o.shape != (P, self.K, self.N_R):
            raise ValueError(f"o shape {o.shape} != {(P,self.K,self.N_R)}")
        if n.shape != (P, self.N_R, self.G):
            raise ValueError(f"n shape {n.shape}")
        if theta.shape != (P, self.N_R, self.G, self.G + 1):
            raise ValueError(f"theta shape {theta.shape}")

        N_counts = n[:, :, :, None] * theta
        F_new = N_counts[:, :, :, self.G].sum(axis=2)
        W_in = self.N[None, :] - n.sum(axis=2)
        if np.any(W_in < -1e-6):
            raise ValueError("当前状态运行台数超过机队")
        W_in_pos = np.maximum(W_in, 0.0)

        q_out, tau = policy.order(t, I, o, n)
        q_out = np.asarray(q_out, dtype=float)
        tau = np.asarray(tau, dtype=float)
        if q_out.shape != (P, self.N_R):
            raise ValueError(f"policy.order 返回 q shape {q_out.shape}")
        if tau.shape != (P, self.N_R):
            raise ValueError(f"policy.order 返回 tau shape {tau.shape}")
        if np.any(q_out < -1e-8) or np.any(q_out > self.Qmax[None, :] + 1e-8):
            raise ValueError("policy.order 返回的 q 越界 [0,Qmax]")
        if np.any(np.abs(tau.sum(axis=1)) > 1e-6):
            raise ValueError("第一阶段 tau 不满足 Σtau=0")
        cap_period = I + tau + o.sum(axis=1) + q_out - self.Imax[None, :]
        if np.any(cap_period > 1e-6):
            p0, r0 = np.unravel_index(np.argmax(cap_period), cap_period.shape)
            raise ValueError(
                f"t={t} 期初容量越界：path={p0} 仓={r0} "
                f"I+tau+Σo+q="
                f"{I[p0,r0]+tau[p0,r0]+o[p0,:,r0].sum()+q_out[p0,r0]:.6f} "
                f"> Imax={self.Imax[r0]:.6f}")

        D = W_in_pos + F_new
        y = np.asarray(policy.fulfill(t, I, D), dtype=float)
        if y.shape != (P, self.N_R, self.N_R):
            raise ValueError(f"policy.fulfill 返回 shape {y.shape}")
        if np.any(y < -1e-8):
            raise ValueError("policy.fulfill 返回负的 y")

        # y[p,r,m]：目的地 r、来源 m；对 r 的出流是 sum_s y[s,r]
        outflow = y.sum(axis=1)
        Y = y.sum(axis=2)
        S_end = I + tau - outflow + o[:, 0, :]

        n_raw = np.zeros((P, self.N_R, self.G), dtype=float)
        n_raw[:, :, 0] = Y + N_counts[:, :, 0, 0]
        for j in range(1, self.G):
            n_raw[:, :, j] = N_counts[:, :, :, j].sum(axis=2)
        n_out = round_repair_vec(n_raw, self.N, self.digits, self.repair_tol)
        W_out = self.N[None, :] - n_out.sum(axis=2)

        # o_{1,t+1}=o_{2,t}, ..., o_{L-1,t+1}=q_t
        o_next = np.empty((P, self.K, self.N_R), dtype=float)
        if self.K > 1:
            o_next[:, :self.K - 1, :] = o[:, 1:, :]
        o_next[:, self.K - 1, :] = q_out

        I_out = S_end
        if np.any(I_out < -1e-6) or np.any(
                I_out > self.Imax[None, :] + 1e-6):
            raise ValueError("tau 后库存 I_out 越界")
        cap_next = I_out + o_next.sum(axis=1) - self.Imax[None, :]
        if np.any(cap_next > 1e-6):
            p0, r0 = np.unravel_index(np.argmax(cap_next), cap_next.shape)
            raise ValueError(
                f"t={t} 下一期状态容量越界：path={p0} 仓={r0} "
                f"I_out+Σo_next={I_out[p0,r0]+o_next[p0,:,r0].sum():.6f} "
                f"> Imax={self.Imax[r0]:.6f}")
        taup = np.maximum(tau, 0.0)
        taum = np.maximum(-tau, 0.0)

        proc = self.C * q_out.sum(axis=1)
        hold = self.h / 2.0 * (I.sum(axis=1) + I_out.sum(axis=1))
        down = self.Cs * W_in.sum(axis=1)
        active = self.atc / 2.0 * (taup + taum).sum(axis=1)
        passive = (self.tc_route[None, :, :] * y).sum(axis=(1, 2))
        terms = np.stack([proc, hold, down, active, passive], axis=1)
        cost = terms.sum(axis=1)

        return StepResult(
            q_out=q_out, y=y, I_out=I_out, S_end=S_end, n_out=n_out,
            F_new=F_new, W_in=W_in, W_out=W_out, o_next=o_next,
            terms=terms, cost=cost,
            N_counts=N_counts, tau=tau, taup=taup, taum=taum)

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
        q0 = np.zeros((P, self.N_R), dtype=float)

        arrays: dict = {}
        if record:
            arrays = {
                "x_in": np.zeros((P, self.T, self.layout.dim), dtype=float),
                "I_in": np.zeros((P, self.T, self.N_R), dtype=float),
                "o_in": np.zeros((P, self.T, self.K, self.N_R), dtype=float),
                "n_in": np.zeros((P, self.T, self.N_R, self.G), dtype=float),
                "q": np.zeros((P, self.T, self.N_R), dtype=float),
                "taup": np.zeros((P, self.T, self.N_R), dtype=float),
                "taum": np.zeros((P, self.T, self.N_R), dtype=float),
                "S_end": np.zeros((P, self.T, self.N_R), dtype=float),
                "I_out": np.zeros((P, self.T, self.N_R), dtype=float),
                "o_next": np.zeros((P, self.T, self.K, self.N_R), dtype=float),
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
                x[:, self.layout.off_q:self.layout.off_q + self.layout.n_q] = 0.0
                arrays["x_in"][:, k, :] = x
                arrays["I_in"][:, k, :] = I
                arrays["o_in"][:, k, :, :] = o
                arrays["n_in"][:, k, :, :] = n
                arrays["q"][:, k, :] = st.q_out
                arrays["taup"][:, k, :] = st.taup
                arrays["taum"][:, k, :] = st.taum
                arrays["S_end"][:, k, :] = st.S_end
                arrays["I_out"][:, k, :] = st.I_out
                arrays["o_next"][:, k, :, :] = st.o_next
                arrays["y"][:, k, :, :] = st.y
                arrays["n_out"][:, k, :, :] = st.n_out
                arrays["F"][:, k, :] = st.F_new
                arrays["W_in"][:, k, :] = st.W_in
                arrays["W_out"][:, k, :] = st.W_out
                arrays["cost"][:, k] = st.cost
                arrays["terms"][:, k, :] = st.terms
                arrays["scene"][:, k] = np.asarray(scene_paths[:, k], dtype=np.int16)

            I = st.I_out
            o = st.o_next
            n = st.n_out

        if record:
            arrays["totals"] = totals.copy()
        return SimulationResult(
            arrays=arrays, totals=totals, term_totals=term_totals,
            q0=q0, T=self.T, P=P,
            policy_params=policy.params(),
            meta={"scene_paths_hash": scene_hash(scene_paths)})


def scene_hash(scene_paths: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(scene_paths).tobytes()).hexdigest()


def audit_result(prob, res: SimulationResult, *, tol: float = 1e-6,
                 state_tol: float = 1e-4, max_points: int | None = None):
    """逐 (路径,期) 核对状态递推、容量和决策约束。"""
    if not res.arrays:
        raise ValueError("audit 需要 record=True 的仿真结果")
    NR, G, K = prob.N_R, prob.G, prob.L - 1
    T, P = res.T, res.P
    a = res.arrays
    audit = {c: 0.0 for c in (
        "pas", "fleet", "dem", "grade0", "grade", "N链接", "结构零",
        "F_new", "W守恒", "send_def", "tau_bal", "tau_lo", "tau_hi",
        "tau_src", "inv", "pipe", "pipe_last", "cap_init", "cap_next",
        "界.下", "界.上", "q.下", "q.上", "链.状态")}
    n = 0
    for p in range(P):
        for t in range(T):
            if max_points is not None and n >= max_points:
                break
            I = np.asarray(a["I_in"][p, t], float)
            o = np.asarray(a["o_in"][p, t], float)
            nin = np.asarray(a["n_in"][p, t], float)
            q = np.asarray(a["q"][p, t], float)
            taup = np.asarray(a["taup"][p, t], float)
            taum = np.asarray(a["taum"][p, t], float)
            tau = taup - taum
            S = np.asarray(a["S_end"][p, t], float)
            Iout = np.asarray(a["I_out"][p, t], float)
            on = np.asarray(a["o_next"][p, t], float)
            y = np.asarray(a["y"][p, t], float)
            nout = np.asarray(a["n_out"][p, t], float)
            F = np.asarray(a["F"][p, t], float)
            W_in = np.asarray(a["W_in"][p, t], float)
            W_out = np.asarray(a["W_out"][p, t], float)
            si = int(a["scene"][p, t])
            th = np.asarray(prob.pool[si][1], float)
            Nv = th * nin[:, :, None]
            F_calc = Nv[:, :, G].sum(axis=1)
            outflow = y.sum(axis=0)
            Y = y.sum(axis=1)
            zero_dev = 0.0
            for i in range(G):
                if i:
                    zero_dev = max(zero_dev, float(np.max(np.abs(th[:, i, :i]))))
            row_dev = float(np.max(np.abs(Nv.sum(axis=2) - nin)))
            grade_dev = max(
                (float(np.max(np.abs(nout[:, j] - Nv[:, :, j].sum(axis=1))))
                 for j in range(1, G)), default=0.0)
            grade0_dev = float(np.max(np.abs(
                nout[:, 0] - Y - Nv[:, 0, 0])))
            good = Nv[:, :, :G].sum(axis=(1, 2))
            capN = prob.N + prob.fleet_tol
            d = {
                "pas": float(np.maximum(outflow - (I + tau), 0).max()),
                "fleet": float(np.maximum(nin.sum(axis=1) - capN, 0).max()),
                "dem": float(np.maximum(Y + good - capN, 0).max()),
                "grade0": grade0_dev,
                "grade": grade_dev,
                "N链接": row_dev,
                "结构零": zero_dev,
                "F_new": float(np.max(np.abs(F - F_calc))),
                "W守恒": float(np.max(np.abs(W_out - (W_in + F_calc - Y)))),
                "send_def": float(np.max(np.abs(
                    S - (I + tau - outflow + o[0])))),
                "tau_bal": float(abs(tau.sum())),
                "tau_lo": float(np.maximum(-(I + tau), 0).max()),
                "tau_hi": float(np.maximum(I + tau - prob.Imax, 0).max()),
                "tau_src": float(np.maximum(
                    tau - (I.sum() - I), 0).max()),
                "inv": float(np.max(np.abs(Iout - S))),
                "pipe": float(np.max(np.abs(on[:K - 1] - o[1:]))) if K > 1 else 0.0,
                "pipe_last": float(np.max(np.abs(on[K - 1] - q))),
                "cap_init": float(np.maximum(
                    I + tau + o.sum(axis=0) + q - prob.Imax, 0).max()),
                "cap_next": float(np.maximum(Iout + on.sum(axis=0)
                                             - prob.Imax, 0).max()),
                "界.下": float(max(0.0, -min(I.min(), o.min(), nin.min()))),
                "界.上": float(max(
                    np.maximum(I - prob.Imax, 0).max(),
                    np.maximum(o - np.asarray(prob.Qmax)[None, :], 0).max(),
                    np.maximum(nin - np.asarray(prob.N)[:, None], 0).max())),
                "q.下": float(max(0.0, -q.min())),
                "q.上": float(max(0.0, (q - np.asarray(prob.Qmax)).max())),
            }
            if t < T - 1:
                next_I = np.asarray(a["I_in"][p, t + 1], float)
                next_o = np.asarray(a["o_in"][p, t + 1], float)
                next_n = np.asarray(a["n_in"][p, t + 1], float)
                d["链.状态"] = float(max(
                    np.max(np.abs(Iout - next_I)),
                    np.max(np.abs(on - next_o)),
                    np.max(np.abs(nout - next_n))))
            for c, v in d.items():
                audit[c] = max(audit[c], float(v))
            n += 1
        if max_points is not None and n >= max_points:
            break
    bad = {}
    for c, v in audit.items():
        lim = max(tol, state_tol) if c in ("grade0", "grade", "W守恒") else tol
        if v > lim:
            bad[c] = v
    return audit, bad


def cost_terms_max_dev(res: SimulationResult) -> float:
    if not res.arrays:
        return float("nan")
    return float(np.max(np.abs(res.arrays["terms"].sum(axis=2) - res.arrays["cost"])))
