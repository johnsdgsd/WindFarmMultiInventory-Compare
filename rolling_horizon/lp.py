"""滚动时域的两阶段随机 LP 与当期追索 LP。

口径（新 docs/滚动时域.md + 主模型约束）：
- 状态 s_t=(I,n,o_1..o_{L-1})，W=N-sum(n) 由状态导出；
- 每期只有一个订货 q_t；采购成本 C q_t 按当期计；
- 只有 t 期 q_t 无情景索引；tau>t 的 q_tau 是追索变量；
- 状态转移 I_{t+1}=S_out+tau, o_{l,t+1}=o_{l+1,t}, o_{L,t+1}=q_t,
  n_{t+1,0}=N_00+Y, n_{t+1,j}=sum_i N_ij；
- 容量按主模型期末侧：I_{t+1}+sum_l o_{l,t+1} <= Imax；
- 不使用 q_in/q_out 或 in/out 两套变量。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


class LPError(RuntimeError):
    """LP 求解失败（含不可行/超时）。"""

    def __init__(self, message: str, **info):
        super().__init__(message)
        self.info = info


@dataclass
class LPStats:
    name: str
    t: int
    window: int
    n_scen: int
    n_vars: int
    n_eq: int
    n_le: int
    build_s: float
    solve_s: float
    objective: float
    constant: float = 0.0
    solver: str = "copt"


@dataclass
class State:
    """滚动时域状态：I(N_R), o(L-1,N_R), n(N_R,G)；W 由 n 导出。

    o[0] 本期期末到货并加入 I_{t+1}；o[1] 下期期末到货；依此类推。
    """

    I: np.ndarray
    o: np.ndarray
    n: np.ndarray

    def validate(self, N_R: int, G: int, L: int):
        if L < 2:
            raise ValueError(f"L 必须 >=2，实际 {L}")
        self.I = np.asarray(self.I, dtype=float).reshape(N_R)
        self.o = np.asarray(self.o, dtype=float).reshape(L - 1, N_R)
        self.n = np.asarray(self.n, dtype=float).reshape(N_R, G)
        neg_tol = 1e-6
        for name, arr in (("I", self.I), ("o", self.o), ("n", self.n)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"状态 {name} 含 NaN/Inf")
            if np.any(arr < -neg_tol):
                raise ValueError(f"状态 {name} 含负值: {arr.min():.3e}")
        # 吸收 LP 的 FeasTol 量级负残差，避免把 -1e-8 传播到下一期。
        self.I = np.maximum(self.I, 0.0)
        self.o = np.maximum(self.o, 0.0)
        self.n = np.maximum(self.n, 0.0)


@dataclass
class TwoStageResult:
    q_first: np.ndarray
    tau_first: np.ndarray
    objective: float
    stats: LPStats


@dataclass
class RecourseResult:
    y: np.ndarray
    tau: np.ndarray
    taup: np.ndarray
    taum: np.ndarray
    S_end: np.ndarray
    I_out: np.ndarray
    objective: float
    stats: LPStats


# --------------------------------------------------------------- 稀疏 LP 外壳
_COPT_ENV = None
_COPT_LOCK = threading.Lock()


def _copt():
    """惰性创建进程内唯一 COPT 环境；loadMatrix 不写文件。"""
    global _COPT_ENV
    import coptpy as cp
    from coptpy import COPT
    if _COPT_ENV is None:
        with _COPT_LOCK:
            if _COPT_ENV is None:
                _COPT_ENV = cp.Envr()
    return cp, COPT


class _SparseLP:
    def __init__(self, name: str, time_limit: float | None = None,
                 solver: str = "copt", threads: int = 1,
                 presolve: int = -1):
        self.name = name
        self.time_limit = time_limit
        self.solver = str(solver)
        self.threads = int(threads)
        self.presolve = int(presolve)
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.c: list[float] = []
        self.names: list[str] = []
        self.eq_rows: list[dict[int, float]] = []
        self.eq_rhs: list[float] = []
        self.le_rows: list[dict[int, float]] = []
        self.le_rhs: list[float] = []
        self.eq_names: list[str] = []
        self.le_names: list[str] = []

    def var(self, name: str, lb: float = 0.0, ub: float = np.inf,
            cost: float = 0.0) -> int:
        i = len(self.c)
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.c.append(float(cost))
        self.names.append(name)
        return i

    def _clean(self, terms: dict[int, float]) -> dict[int, float]:
        return {int(k): float(v) for k, v in terms.items()
                if abs(float(v)) > 1e-15}

    def eq(self, terms: dict[int, float], rhs: float = 0.0,
           name: str = "") -> None:
        t = self._clean(terms)
        if not t:
            if abs(float(rhs)) > 1e-9:
                raise LPError(
                    f"{self.name}: 空等式但 rhs={rhs}", name=name, rhs=rhs)
            return
        self.eq_rows.append(t)
        self.eq_rhs.append(float(rhs))
        self.eq_names.append(name)

    def le(self, terms: dict[int, float], rhs: float,
           name: str = "") -> None:
        t = self._clean(terms)
        if not t:
            if float(rhs) < -1e-9:
                raise LPError(
                    f"{self.name}: 空不等式但 rhs={rhs}", name=name, rhs=rhs)
            return
        self.le_rows.append(t)
        self.le_rhs.append(float(rhs))
        self.le_names.append(name)

    def _matrix(self):
        """把 eq + le 行拼成一个 CSC 矩阵，并给出 lhs/rhs。"""
        from scipy.sparse import coo_matrix
        n = len(self.c)
        rows_eq = len(self.eq_rows)
        rows_le = len(self.le_rows)
        m = rows_eq + rows_le
        data, ri, ci = [], [], []
        lhs = np.empty(m, dtype=float)
        rhs = np.empty(m, dtype=float)
        for i, row in enumerate(self.eq_rows):
            for j, v in row.items():
                data.append(v)
                ri.append(i)
                ci.append(j)
            lhs[i] = self.eq_rhs[i]
            rhs[i] = self.eq_rhs[i]
        for k, row in enumerate(self.le_rows):
            i = rows_eq + k
            for j, v in row.items():
                data.append(v)
                ri.append(i)
                ci.append(j)
            lhs[i] = -1e30
            rhs[i] = self.le_rhs[k]
        A = coo_matrix((data, (ri, ci)), shape=(m, n)).tocsc()
        return A, lhs, rhs

    def solve(self):
        n = len(self.c)
        n_eq = len(self.eq_rows)
        n_le = len(self.le_rows)
        t0 = time.perf_counter()
        if self.solver == "copt":
            cp, COPT = _copt()
            A, lhs, rhs = self._matrix()
            lb = [float(v) for v in self.lb]
            ub = [COPT.INFINITY if not np.isfinite(v) else float(v)
                  for v in self.ub]
            # COPT auto presolve occasionally reports NUMERICAL (status=5)
            # on degenerate two-stage LPs; retrying the same matrix with
            # Presolve=0 solves them in <1 s (see probe_numeric.py).
            trials = [int(self.presolve)]
            if int(self.presolve) != 0:
                trials.append(0)
            statuses = []
            x = None
            fun = float("nan")
            for k, presolve in enumerate(trials):
                m = _COPT_ENV.createModel()
                try:
                    m.setParam(COPT.Param.Logging, 0)
                    m.setParam(COPT.Param.Threads, int(self.threads))
                    if self.time_limit is not None:
                        m.setParam(COPT.Param.TimeLimit,
                                   float(self.time_limit))
                    if presolve >= 0:
                        m.setParam(COPT.Param.Presolve, int(presolve))
                    m.loadMatrix(np.asarray(self.c, dtype=float), A,
                                 lhs, rhs, lb, ub)
                    m.setObjSense(COPT.MINIMIZE)
                    m.solveLP()
                    status = m.status
                    statuses.append(int(status))
                    if (status == COPT.OPTIMAL
                            and int(m.getAttr(COPT.Attr.HasLpSol)) == 1):
                        x = np.asarray(m.getValues(), dtype=float)
                        fun = float(m.getAttr(COPT.Attr.LpObjVal))
                        break
                finally:
                    try:
                        m.clear()
                    except Exception:
                        pass
            solve_s = time.perf_counter() - t0
            if x is None:
                raise LPError(
                    f"{self.name}: COPT 失败 status={statuses} "
                    f"(presolve trials={trials})",
                    n_vars=n, n_eq=n_eq, n_le=n_le, solve_s=solve_s,
                    presolve=trials, statuses=statuses)
            if x.size != n:
                raise LPError(
                    f"{self.name}: COPT 返回值数 {x.size} != {n}",
                    n_vars=n, n_eq=n_eq, n_le=n_le, solve_s=solve_s)
            return x, fun, n, n_eq, n_le, solve_s

        if self.solver == "highs":
            from scipy.optimize import linprog
            A, lhs, rhs = self._matrix()
            bounds = [(lb, None if not np.isfinite(ub) else ub)
                      for lb, ub in zip(self.lb, self.ub)]
            eq = slice(0, n_eq)
            le = slice(n_eq, n_eq + n_le)
            last = None
            for method in ("highs", "highs-ipm", "highs-ds"):
                res = linprog(
                    np.asarray(self.c, dtype=float),
                    A_ub=A[le, :] if n_le else None,
                    b_ub=rhs[le] if n_le else None,
                    A_eq=A[eq, :] if n_eq else None,
                    b_eq=rhs[eq] if n_eq else None,
                    bounds=bounds, method=method,
                    options={"presolve": True, "disp": False})
                last = res
                if res.success and res.x is not None:
                    solve_s = time.perf_counter() - t0
                    return (np.asarray(res.x, dtype=float), float(res.fun),
                            n, n_eq, n_le, solve_s)
            solve_s = time.perf_counter() - t0
            raise LPError(
                f"{self.name}: HiGHS 三种方法均失败 status={last.status}: "
                f"{last.message}", n_vars=n, n_eq=n_eq,
                n_le=n_le, solve_s=solve_s)
        raise ValueError(f"未知 solver: {self.solver}")


# --------------------------------------------------------------- 工具函数
def _add(row: dict[int, float], expr, coef: float = 1.0) -> float:
    """表达式 expr=(idx|None, const)；把 coef*expr 加入 row，返回常数贡献。"""
    idx, const = expr
    c = float(coef) * float(const)
    if idx is not None:
        row[int(idx)] = row.get(int(idx), 0.0) + float(coef)
    return c


def _add_cost(lp: _SparseLP, expr, coef: float, weight: float):
    idx, const = expr
    if idx is None:
        return float(coef) * float(weight) * float(const)
    lp.c[int(idx)] += float(coef) * float(weight)
    return 0.0


def _check_theta(theta, n_scen, window, N_R, G):
    th = np.asarray(theta, dtype=float)
    expected = (n_scen, window, N_R, G, G + 1)
    if th.shape != expected:
        raise ValueError(f"theta_window shape {th.shape} != {expected}")
    tol = 1e-6
    if not np.all(np.isfinite(th)):
        raise ValueError("theta 含 NaN 或 Inf")
    if np.any(th < -tol):
        raise ValueError("theta 存在负转移比例")
    for i in range(G):
        row_sum = th[:, :, :, i, i:].sum(axis=-1)
        if not np.allclose(row_sum, 1.0, atol=tol, rtol=0.0):
            raise ValueError(f"theta 第 {i} 行转移比例之和不为 1")
        if i > 0 and np.any(np.abs(th[:, :, :, i, :i]) > tol):
            raise ValueError(f"theta 第 {i} 行包含结构性零")
    return th


# --------------------------------------------------------------- 两阶段 LP
class TwoStageLP:
    """给定 t 期初状态 + 优化池切片，求解两阶段随机 LP。"""

    def __init__(self, prob, state: State, t: int, theta_window,
                 window_len: int | None = None,
                 time_limit: float | None = None,
                 solver: str = "copt", threads: int = 1,
                 presolve: int = -1):
        self.prob = prob
        self.t = int(t)
        self.T = int(prob.T)
        self.N_R = int(prob.N_R)
        self.G = int(prob.G)
        self.L = int(prob.L)
        self.K = self.L - 1
        if not (1 <= self.t <= self.T):
            raise ValueError(f"滚动 LP 只支持 1<=t<=T；t={self.t}")
        if window_len is None:
            self.t_end = self.T
        else:
            w = int(window_len)
            if w < 1:
                raise ValueError(f"window_len 必须 >=1，实际 {w}")
            self.t_end = min(self.T, self.t + w - 1)
        self.window = self.t_end - self.t + 1
        self.Nsc = int(np.asarray(theta_window).shape[0])
        self.theta = _check_theta(
            theta_window, self.Nsc, self.window, self.N_R, self.G)
        state.validate(self.N_R, self.G, self.L)
        self.state = state
        self.N = np.asarray(prob.N, dtype=float)
        self.Qmax = np.asarray(prob.Qmax, dtype=float)
        self.Imax = np.asarray(prob.Imax, dtype=float)
        if np.any(self.state.I > self.Imax + 1e-6):
            raise ValueError(f"I 超 Imax: {self.state.I}")
        if np.any(self.state.n.sum(axis=1) > self.N + 1e-6):
            raise ValueError("状态运行台数超过机队")
        self.C = float(prob.C)
        self.h = float(prob.h)
        self.Cs = float(prob.Cs)
        self.atc = float(getattr(prob, "atc", 0.0))
        self.tc = np.asarray(prob.tc_route, dtype=float)
        self.fleet_tol = float(getattr(prob, "fleet_tol", 0.0))
        self.time_limit = time_limit
        self.solver = str(solver)
        self.threads = int(threads)
        self.presolve = int(presolve)
        self.lp: _SparseLP | None = None
        self.qfirst: list[int] = []
        self.taufp: list[int] = []
        self.taufm: list[int] = []
        self.state_vars: dict = {}
        self.dec: dict = {}
        self.constant = 0.0
        self.stats: LPStats | None = None
        self.x: np.ndarray | None = None
        self.q_first: np.ndarray | None = None
        self.tau_first: np.ndarray | None = None
        self.build_s = 0.0

    # ---- 状态/决策表达式：返回 (变量下标|None, 常数) ----
    def _I(self, s, tau, r):
        if tau == self.t:
            return (None, float(self.state.I[r]))
        return (self.state_vars[(s, tau)]["I"][r], 0.0)

    def _o(self, s, tau, r, l):
        if tau == self.t:
            return (None, float(self.state.o[l, r]))
        return (self.state_vars[(s, tau)]["o"][l][r], 0.0)

    def _n(self, s, tau, r, i):
        if tau == self.t:
            return (None, float(self.state.n[r, i]))
        return (self.state_vars[(s, tau)]["n"][r][i], 0.0)

    def _q(self, s, tau, r):
        if tau == self.t:
            return (self.qfirst[r], 0.0)
        return (self.dec[(s, tau)]["q"][r], 0.0)

    # ---- 建模型 ----
    def build(self):
        t0 = time.perf_counter()
        N_R, G, L, K = self.N_R, self.G, self.L, self.K
        lp = _SparseLP(f"two_stage_t{self.t}", self.time_limit,
                       solver=self.solver, threads=self.threads,
                       presolve=self.presolve)
        self.lp = lp

        # 第一阶段：tau_t（先调拨）和 q_t（再订货），均全情景共享
        # tau 的紧上界是约束的推论：流入 <= Imax-I 且 <= 其他仓可用库存；
        # 流出 <= I。收紧边界可减少 tau+ / tau- 的共同偏移和数值退化。
        sum_I = float(self.state.I.sum())
        ubp = [max(0.0, min(float(self.Imax[r] - self.state.I[r]),
                            sum_I - float(self.state.I[r])))
               for r in range(N_R)]
        ubm = [max(0.0, float(self.state.I[r])) for r in range(N_R)]
        self.taufp = [
            lp.var(f"taufp[{r}]", lb=0.0, ub=float(ubp[r]))
            for r in range(N_R)]
        self.taufm = [
            lp.var(f"taufm[{r}]", lb=0.0, ub=float(ubm[r]))
            for r in range(N_R)]
        self.qfirst = [
            lp.var(f"qfirst[{r}]", lb=0.0,
                   ub=0.0 if self.t == self.t_end else float(self.Qmax[r]))
            for r in range(N_R)]

        # 第一阶段 tau 的确定性约束
        for r in range(N_R):
            lp.le({self.taufm[r]: 1.0, self.taufp[r]: -1.0},
                  float(self.state.I[r]), name=f"tau_lo_f[{r}]")
            lp.le({self.taufp[r]: 1.0, self.taufm[r]: -1.0},
                  float(self.Imax[r] - self.state.I[r]),
                  name=f"tau_hi_f[{r}]")
            rhs_src = float(self.state.I.sum() - self.state.I[r])
            lp.le({self.taufp[r]: 1.0, self.taufm[r]: -1.0},
                  rhs_src, name=f"tau_src_f[{r}]")
        row = {}
        for r in range(N_R):
            row[self.taufp[r]] = row.get(self.taufp[r], 0.0) + 1.0
            row[self.taufm[r]] = row.get(self.taufm[r], 0.0) - 1.0
        lp.eq(row, 0.0, name="tau_bal_f")

        # 期初容量： (I_t + tau_t) + sum_l o_{l,t} + q_t <= Imax
        for r in range(N_R):
            base = float(self.state.I[r] + self.state.o[:, r].sum())
            rhs = float(self.Imax[r]) - base
            lp.le({self.qfirst[r]: 1.0, self.taufp[r]: 1.0,
                   self.taufm[r]: -1.0},
                  rhs, name=f"cap_init[{r}]")

        # 状态变量：t+1..T+1；t 期状态已知
        for s in range(self.Nsc):
            for tau in range(self.t + 1, self.t_end + 2):
                sv = {
                    "I": [lp.var(f"I[{s},{tau},{r}]", 0.0,
                                 float(self.Imax[r]))
                          for r in range(N_R)],
                    "o": [[lp.var(f"o[{s},{tau},{l},{r}]", 0.0,
                                  float(self.Qmax[r]))
                           for r in range(N_R)] for l in range(K)],
                    "n": [[lp.var(f"n[{s},{tau},{r},{i}]", 0.0,
                                  float(self.N[r]))
                           for i in range(G)] for r in range(N_R)],
                }
                self.state_vars[(s, tau)] = sv

        # 当期决策：t..T
        for s in range(self.Nsc):
            for tau in range(self.t, self.t_end + 1):
                d = {
                    "q": (None if tau == self.t else
                          [lp.var(f"q[{s},{tau},{r}]", 0.0,
                                  0.0 if tau == self.t_end
                                  else float(self.Qmax[r]))
                           for r in range(N_R)]),
                    "y": [[lp.var(f"y[{s},{tau},{r},{m}]", 0.0,
                                  float(min(self.N[r], self.Imax[m])))
                           for m in range(N_R)] for r in range(N_R)],
                    "taup": (self.taufp if tau == self.t else
                             [lp.var(f"taup[{s},{tau},{r}]", 0.0,
                                     float(min(self.Imax.sum()
                                               + self.Qmax.sum(),
                                               self.Imax[r])))
                              for r in range(N_R)]),
                    "taum": (self.taufm if tau == self.t else
                             [lp.var(f"taum[{s},{tau},{r}]", 0.0,
                                     float(self.Imax[r] + self.Qmax[r]))
                              for r in range(N_R)]),
                    "S_end": [lp.var(f"send[{s},{tau},{r}]", 0.0,
                                     float(self.Imax[r] + self.Qmax[r]))
                              for r in range(N_R)],
                    "N": [[[None] * (G + 1) for _ in range(G)]
                          for _ in range(N_R)],
                }
                for r in range(N_R):
                    for i in range(G):
                        for j in range(i, G + 1):
                            d["N"][r][i][j] = lp.var(
                                f"N[{s},{tau},{r},{i},{j}]", 0.0,
                                float(self.N[r]))
                self.dec[(s, tau)] = d

        # 约束与目标
        weight = 1.0 / float(self.Nsc)
        const_total = 0.0
        for s in range(self.Nsc):
            for tau in range(self.t, self.t_end + 1):
                d = self.dec[(s, tau)]
                w = tau - self.t
                th = self.theta[s, w]
                nxt = self.state_vars[(s, tau + 1)]
                for r in range(N_R):
                    # N = Theta n_tau
                    for i in range(G):
                        for j in range(i, G + 1):
                            row = {}
                            c = _add(row, (d["N"][r][i][j], 0.0))
                            c += _add(row, self._n(s, tau, r, i),
                                      -float(th[r, i, j]))
                            lp.eq(row, -c,
                                  name=f"Ndef[{s},{tau},{r},{i},{j}]")
                    # n_{t+1,0} = N_00 + Y
                    row = {}
                    c = _add(row, (nxt["n"][r][0], 0.0))
                    c += _add(row, (d["N"][r][0][0], 0.0), -1.0)
                    for m in range(N_R):
                        c += _add(row, (d["y"][r][m], 0.0), -1.0)
                    lp.eq(row, -c, name=f"grade0[{s},{tau},{r}]")
                    # n_{t+1,j} = sum_i N_ij
                    for j in range(1, G):
                        row = {}
                        c = _add(row, (nxt["n"][r][j], 0.0))
                        for i in range(j + 1):
                            c += _add(row, (d["N"][r][i][j], 0.0), -1.0)
                        lp.eq(row, -c, name=f"grade[{s},{tau},{r},{j}]")
                    # S_end = I_tau + tau - sum_s y + o_1
                    row = {}
                    c = _add(row, (d["S_end"][r], 0.0))
                    c += _add(row, self._I(s, tau, r), -1.0)
                    c += _add(row, (d["taup"][r], 0.0), -1.0)
                    c += _add(row, (d["taum"][r], 0.0), 1.0)
                    for ss in range(N_R):
                        c += _add(row, (d["y"][ss][r], 0.0), 1.0)
                    c += _add(row, self._o(s, tau, r, 0), -1.0)
                    lp.eq(row, -c, name=f"send[{s},{tau},{r}]")
                    # I_{t+1} = S_end（tau 已进入 S_end）
                    row = {}
                    c = _add(row, (nxt["I"][r], 0.0))
                    c += _add(row, (d["S_end"][r], 0.0), -1.0)
                    lp.eq(row, -c, name=f"inv[{s},{tau},{r}]")
                    # 在途状态转移：o_next[l]=o_tau[l+1]，末层=q_tau
                    for l in range(K - 1):
                        row = {}
                        c = _add(row, (nxt["o"][l][r], 0.0))
                        c += _add(row, self._o(s, tau, r, l + 1), -1.0)
                        lp.eq(row, -c,
                              name=f"pipe[{s},{tau},{l},{r}]")
                    row = {}
                    c = _add(row, (nxt["o"][K - 1][r], 0.0))
                    c += _add(row, self._q(s, tau, r), -1.0)
                    lp.eq(row, -c, name=f"pipe_last[{s},{tau},{r}]")
                    # 被动源容量
                    row = {}
                    c = 0.0
                    for ss in range(N_R):
                        c += _add(row, (d["y"][ss][r], 0.0))
                    c += _add(row, self._I(s, tau, r), -1.0)
                    c += _add(row, (d["taup"][r], 0.0), -1.0)
                    c += _add(row, (d["taum"][r], 0.0), 1.0)
                    lp.le(row, -c, name=f"pas[{s},{tau},{r}]")
                    # 需求/机队
                    row = {}
                    c = 0.0
                    for m in range(N_R):
                        c += _add(row, (d["y"][r][m], 0.0))
                    for i in range(G):
                        for j in range(i, G):
                            c += _add(row, (d["N"][r][i][j], 0.0))
                    lp.le(row, float(self.N[r] + self.fleet_tol) - c,
                          name=f"dem[{s},{tau},{r}]")
                    row = {}
                    c = 0.0
                    for i in range(G):
                        c += _add(row, self._n(s, tau, r, i))
                    lp.le(row, float(self.N[r] + self.fleet_tol) - c,
                          name=f"fleet[{s},{tau},{r}]")
                    # tau：t 期已加确定性版本；未来期用 tau 前库存 I_tau
                    if tau > self.t:
                        row = {}
                        c = _add(row, self._I(s, tau, r), 1.0)
                        c += _add(row, (d["taup"][r], 0.0), 1.0)
                        c += _add(row, (d["taum"][r], 0.0), -1.0)
                        lp.le(row, float(self.Imax[r]) - c,
                              name=f"tau_hi[{s},{tau},{r}]")
                        row = {}
                        c = _add(row, self._I(s, tau, r), -1.0)
                        c += _add(row, (d["taup"][r], 0.0), -1.0)
                        c += _add(row, (d["taum"][r], 0.0))
                        lp.le(row, -c, name=f"tau_lo[{s},{tau},{r}]")
                        row = {}
                        c = _add(row, (d["taup"][r], 0.0))
                        c += _add(row, (d["taum"][r], 0.0), -1.0)
                        for m in range(N_R):
                            if m != r:
                                c += _add(row, self._I(s, tau, m), -1.0)
                        lp.le(row, -c, name=f"tau_src[{s},{tau},{r}]")
                    # 未来期的本期容量：I_tau + tau + Σo_tau + q_tau <= Imax
                    if tau > self.t:
                        row = {}
                        c = _add(row, self._I(s, tau, r), 1.0)
                        c += _add(row, (d["taup"][r], 0.0), 1.0)
                        c += _add(row, (d["taum"][r], 0.0), -1.0)
                        for l in range(K):
                            c += _add(row, self._o(s, tau, r, l), 1.0)
                        c += _add(row, self._q(s, tau, r), 1.0)
                        lp.le(row, float(self.Imax[r]) - c,
                              name=f"cap[{s},{tau},{r}]")
                    # 转移后的状态容量：I_{tau+1}+Σo_{tau+1} <= Imax
                    row = {}
                    c = _add(row, (nxt["I"][r], 0.0), 1.0)
                    for l in range(K):
                        c += _add(row, (nxt["o"][l][r], 0.0), 1.0)
                    lp.le(row, float(self.Imax[r]) - c,
                          name=f"cap_next[{s},{tau},{r}]")
                    # 成本：采购 C q_tau（当期）
                    const_total += _add_cost(
                        lp, self._q(s, tau, r), self.C, weight)
                    # 持有 h/2 (I_tau + I_{t+1})
                    const_total += _add_cost(
                        lp, self._I(s, tau, r), self.h / 2.0, weight)
                    const_total += _add_cost(
                        lp, (nxt["I"][r], 0.0), self.h / 2.0, weight)
                    # 停机 Cs W_tau = Cs (N_r - sum_i n_tau)
                    const_total += (
                        self.Cs * float(self.N[r]) * weight)
                    for i in range(G):
                        const_total += _add_cost(
                            lp, self._n(s, tau, r, i), -self.Cs, weight)
                    lp.c[d["taup"][r]] += self.atc / 2.0 * weight
                    lp.c[d["taum"][r]] += self.atc / 2.0 * weight
                    for m in range(N_R):
                        lp.c[d["y"][r][m]] += float(
                            self.tc[r, m]) * weight
                # tau 守恒：t 期已加确定性版本
                if tau > self.t:
                    row = {}
                    c = 0.0
                    for r in range(N_R):
                        c += _add(row, (d["taup"][r], 0.0))
                        c += _add(row, (d["taum"][r], 0.0), -1.0)
                    lp.eq(row, -c, name=f"tau_bal[{s},{tau}]")

        self.constant = float(const_total)
        self.build_s = time.perf_counter() - t0
        return self

    def solve(self) -> TwoStageResult:
        if self.lp is None:
            self.build()
        x, fun, n, neq, nle, solve_s = self.lp.solve()
        q = np.asarray([x[i] for i in self.qfirst], dtype=float)
        q = np.maximum(q, 0.0)
        tau = np.asarray([x[self.taufp[r]] - x[self.taufm[r]]
                          for r in range(self.N_R)], dtype=float)
        if np.any(q > np.asarray(self.prob.Qmax, dtype=float) + 1e-6):
            raise LPError(f"q_first 越上界: {q.tolist()}")
        # cap_init 必须是 tau 后口径：I + tau + Σo + q <= Imax
        viol = (self.state.I + tau + self.state.o.sum(axis=0) + q
                - np.asarray(self.prob.Imax))
        if np.any(viol > 1e-6):
            raise LPError(
                f"t={self.t}: q_first/tau_first 违反 cap_init, "
                f"viol={viol.tolist()}, I={self.state.I.tolist()}, "
                f"tau={tau.tolist()}, "
                f"sum_o={self.state.o.sum(axis=0).tolist()}, "
                f"q={q.tolist()}, "
                f"Imax={np.asarray(self.prob.Imax).tolist()}")
        self.x = x
        self.q_first = q
        self.tau_first = tau
        self.stats = LPStats(
            name=self.lp.name, t=self.t, window=self.window,
            n_scen=self.Nsc, n_vars=n, n_eq=neq, n_le=nle,
            build_s=float(self.build_s), solve_s=float(solve_s),
            objective=float(fun + self.constant),
            constant=float(self.constant), solver=self.solver)
        return TwoStageResult(
            q_first=q, tau_first=tau,
            objective=float(fun + self.constant), stats=self.stats)


# --------------------------------------------------------------- 追索 LP
class RecourseLP:
    """真实 Theta_t 观测后的当期确定性追索 LP（y/tau）。

    约束按主模型当期版本：pas、dem、send_def、I_out、tau_*，
    并限制真实执行后的下一期状态 I_out+Σo_next<=Imax。
    目标含 -Cs*Y（下一期待修减少的收益）和 h/2*I_out；仿真成本仍按即时五项累加。
    """

    def __init__(self, prob, state: State, q_out, tau_in, F_new,
                 time_limit: float | None = None,
                 solver: str = "copt", threads: int = 1,
                 presolve: int = -1):
        self.prob = prob
        self.N_R = int(prob.N_R)
        self.G = int(prob.G)
        self.L = int(prob.L)
        self.K = self.L - 1
        state.validate(self.N_R, self.G, self.L)
        self.state = state
        self.N = np.asarray(prob.N, dtype=float)
        self.Imax = np.asarray(prob.Imax, dtype=float)
        self.Qmax = np.asarray(prob.Qmax, dtype=float)
        self.Cs = float(prob.Cs)
        self.h = float(prob.h)
        self.atc = float(getattr(prob, "atc", 0.0))
        self.tc = np.asarray(prob.tc_route, dtype=float)
        self.q_out = np.asarray(q_out, dtype=float).reshape(self.N_R)
        self.tau_in = np.asarray(tau_in, dtype=float).reshape(self.N_R)
        self.F_new = np.asarray(F_new, dtype=float).reshape(self.N_R)
        if np.any(self.q_out < -1e-9) or np.any(
                self.q_out > self.Qmax + 1e-9):
            raise ValueError("q_out 越界 [0,Qmax]")
        if np.any(self.state.I + self.tau_in < -1e-6):
            raise ValueError("tau_in 使库存为负")
        if not np.all(np.isfinite(self.F_new)) or np.any(self.F_new < -1e-9):
            raise ValueError(f"F_new 非法: {self.F_new}")
        self.W_in = self.N - self.state.n.sum(axis=1)
        if np.any(self.W_in < -1e-6):
            raise LPError(
                f"当前状态运行机组超过机队: W_in={self.W_in.tolist()}")
        self.W_in = np.maximum(self.W_in, 0.0)
        self.D = self.W_in + self.F_new
        # 下一期在途：o_{t+1}=[o_{2..L-1}, q_t]，共 L-1 层
        self.o_next = np.empty((self.K, self.N_R), dtype=float)
        self.o_next[:self.K - 1, :] = self.state.o[1:, :]
        self.o_next[self.K - 1, :] = self.q_out
        self.time_limit = time_limit
        self.solver = str(solver)
        self.threads = int(threads)
        self.presolve = int(presolve)
        self.lp: _SparseLP | None = None
        self.stats: LPStats | None = None
        self.build_s = 0.0

    def build(self):
        t0 = time.perf_counter()
        N_R = self.N_R
        lp = _SparseLP("recourse", self.time_limit,
                       solver=self.solver, threads=self.threads,
                       presolve=self.presolve)
        self.lp = lp
        y = [[lp.var(f"y[{r},{m}]", 0.0,
                     float(min(self.N[r], self.Imax[m])))
              for m in range(N_R)] for r in range(N_R)]
        S_end = [lp.var(f"send[{r}]", 0.0,
                        float(self.Imax[r] + self.Qmax[r]))
                 for r in range(N_R)]
        I_out = [lp.var(f"Iout[{r}]", 0.0, float(self.Imax[r]))
                 for r in range(N_R)]
        self.y = y
        self.S_end = S_end
        self.I_out = I_out

        for r in range(N_R):
            # 被动源容量：I + tau_in 是可调拨的库存
            row = {}
            c = 0.0
            for ss in range(N_R):
                c += _add(row, (y[ss][r], 0.0))
            lp.le(row, float(self.state.I[r] + self.tau_in[r]) - c,
                  name=f"pas[{r}]")
            # 需求上限 Y <= W_in + F_new
            row = {}
            c = 0.0
            for m in range(N_R):
                c += _add(row, (y[r][m], 0.0))
            lp.le(row, float(self.D[r]) - c, name=f"dem[{r}]")
            # S_end = I + tau_in - sum_s y + o_1
            row = {}
            c = _add(row, (S_end[r], 0.0))
            c += -float(self.state.I[r] + self.tau_in[r])
            for ss in range(N_R):
                c += _add(row, (y[ss][r], 0.0))
            c += -float(self.state.o[0, r])
            lp.eq(row, -c, name=f"send[{r}]")
            # I_out = S_end
            row = {}
            c = _add(row, (I_out[r], 0.0))
            c += _add(row, (S_end[r], 0.0), -1.0)
            lp.eq(row, -c, name=f"inv[{r}]")
            # 真实执行后的下一期状态容量
            rhs = float(self.Imax[r] - self.o_next[:, r].sum())
            if rhs < -1e-8:
                raise LPError(
                    f"仓库 {r} 下一期在途超过容量: "
                    f"sum(o_next)={self.o_next[:, r].sum():.6f}, "
                    f"Imax={self.Imax[r]:.6f}")
            lp.le({I_out[r]: 1.0}, rhs, name=f"cap_end[{r}]")
            # 目标（tau 已在第一阶段计价）
            lp.c[I_out[r]] += self.h / 2.0
            for m in range(N_R):
                lp.c[y[r][m]] += (float(self.tc[r, m]) - self.Cs)
        self.constant = 0.0
        self.build_s = time.perf_counter() - t0
        return self

    def solve(self) -> RecourseResult:
        if self.lp is None:
            self.build()
        x, fun, n, neq, nle, solve_s = self.lp.solve()
        yv = np.asarray([[x[self.y[r][m]] for m in range(self.N_R)]
                         for r in range(self.N_R)], dtype=float)
        tau = self.tau_in.copy()
        taup = np.maximum(tau, 0.0)
        taum = np.maximum(-tau, 0.0)
        S_end = np.asarray([x[self.S_end[r]] for r in range(self.N_R)])
        I_out = np.asarray([x[self.I_out[r]] for r in range(self.N_R)])
        self.stats = LPStats(
            name="recourse", t=0, window=1, n_scen=1,
            n_vars=n, n_eq=neq, n_le=nle,
            build_s=float(self.build_s), solve_s=float(solve_s),
            objective=float(fun), constant=0.0, solver=self.solver)
        return RecourseResult(
            y=yv, tau=tau, taup=taup, taum=taum,
            S_end=S_end, I_out=I_out,
            objective=float(fun), stats=self.stats)
