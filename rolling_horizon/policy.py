"""滚动时域策略：每期用固定优化池切片求解两阶段 LP，只执行当期 q_t。

状态 s_t=(I,n,o_1..o_{L-1})；没有 q_in/q_out。
order(t):      只用 t 期初状态 + 优化池；t=0 返回零，t=T 返回 0。
fulfill(t):    真实 Theta_t 观测后，解当期追索 LP 得 y/tau。
active_transfer: 返回追索 LP 的 tau。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from rolling_horizon.lp import (LPError, RecourseLP, State, TwoStageLP)


class RollingHorizonPolicy:
    def __init__(self, prob, options: dict, pool: np.ndarray,
                 seed_opt: int | None = None, pool_hash: str | None = None,
                 log=None):
        self.prob = prob
        self.T = int(prob.T)
        self.N_R = int(prob.N_R)
        self.G = int(prob.G)
        self.L = int(prob.L)
        self.K = self.L - 1
        self.options = dict(options)
        self.pool = np.asarray(pool, dtype=float)
        if self.pool.ndim != 5:
            raise ValueError(f"pool shape 必须是 5 维，实际 {self.pool.shape}")
        if self.pool.shape[1] != self.T or self.pool.shape[2] != self.N_R:
            raise ValueError(
                f"pool shape {self.pool.shape} 与 T/N_R 不符")
        self.Nsc = int(self.pool.shape[0])
        self.seed_opt = seed_opt
        self.pool_hash = pool_hash
        self.solver = str(self.options.get("solver", "copt"))
        self.threads = int(self.options.get("threads", 1))
        self.workers = max(1, int(self.options.get("workers", 1)))
        tl = self.options.get("time_limit", None)
        self.time_limit = (None if tl is None or float(tl) <= 0.0
                           else float(tl))
        self.verbose = bool(self.options.get("verbose", False))
        self.log_every = max(1, int(self.options.get("log_every", 1)))
        self._log = log
        self._t0 = None
        self._total_lp_solves = 0
        self.active = str(self.options.get("active", "lp"))
        if self.active != "lp":
            raise ValueError("滚动时域只实现 active='lp'（LP 优化 tau）")

        self._t = None
        self._I = None
        self._o = None
        self._n = None
        self._qout = None
        self._tau_first = None
        self._tau = None
        self._y = None
        self.lp_stats: list[dict] = []
        self.lp_failures: list[dict] = []

    # ------------------------------------------------------------- 日志
    def _log_line(self, msg: str):
        if self._log is not None:
            self._log(msg)
        elif self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------- 订货
    def _solve_order(self, p, st: State, theta_win):
        lp = TwoStageLP(
            self.prob, st, self._t, theta_win,
            time_limit=self.time_limit,
            solver=self.solver, threads=self.threads)
        res = lp.solve()
        return p, res

    def order(self, t: int, I, o, n):
        I = np.asarray(I, dtype=float)
        o = np.asarray(o, dtype=float)
        n = np.asarray(n, dtype=float)
        P = I.shape[0]
        if I.shape != (P, self.N_R):
            raise ValueError(f"I shape {I.shape}")
        if o.shape != (P, self.K, self.N_R):
            raise ValueError(f"o shape {o.shape} != {(P,self.K,self.N_R)}")
        if n.shape != (P, self.N_R, self.G):
            raise ValueError(f"n shape {n.shape}")
        t = int(t)
        if t == 0:
            z = np.zeros((P, self.N_R), dtype=float)
            return z, z.copy()

        self._t = t
        qout = np.zeros((P, self.N_R), dtype=float)
        tauout = np.zeros((P, self.N_R), dtype=float)
        theta_win = self.pool[:, t - 1:, :, :, :]
        states = [State(I[p], o[p], n[p]) for p in range(P)]
        results = {}
        failures = []

        def run_one(p):
            try:
                return self._solve_order(p, states[p], theta_win)
            except Exception as exc:  # noqa: BLE001 - 记录后继续收集
                return p, exc

        if self.workers > 1 and P > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(run_one, p) for p in range(P)]
                for fut in as_completed(futs):
                    p, res = fut.result()
                    if isinstance(res, Exception):
                        failures.append((p, res))
                    else:
                        results[p] = res
        else:
            for p in range(P):
                p2, res = run_one(p)
                if isinstance(res, Exception):
                    failures.append((p2, res))
                else:
                    results[p2] = res

        if failures:
            p, exc = failures[0]
            self.lp_failures.append({
                "path": int(p), "t": t, "error": str(exc),
                "info": dict(getattr(exc, "info", {})),
            })
            raise exc

        for p in range(P):
            res = results[p]
            qout[p] = np.clip(
                res.q_first, 0.0,
                np.asarray(self.prob.Qmax, dtype=float))
            tauout[p] = np.asarray(res.tau_first, dtype=float)
            stt = res.stats
            self.lp_stats.append({
                "path": int(p), "t": t, "window": stt.window,
                "n_scen": stt.n_scen, "n_vars": stt.n_vars,
                "n_eq": stt.n_eq, "n_le": stt.n_le,
                "build_s": stt.build_s, "solve_s": stt.solve_s,
                "objective": stt.objective,
                "solver": stt.solver, "status": "ok",
            })
            self._total_lp_solves += 1
        self._I = I.copy()
        self._o = o.copy()
        self._n = n.copy()
        self._qout = qout.copy()
        self._tau_first = tauout.copy()
        self._tau = None
        self._y = None
        if self._t0 is None:
            self._t0 = time.perf_counter()
        if self.verbose or self._log is not None:
            if (t % self.log_every == 0) or t == self.T:
                elapsed = time.perf_counter() - self._t0
                eta = (elapsed / max(t, 1)) * max(self.T - t, 0)
                self._log_line(
                    f"[rolling] t={t:03d}/{self.T} paths={P} "
                    f"lp_calls={self._total_lp_solves} "
                    f"elapsed={elapsed:.1f}s eta~{eta:.0f}s")
        return qout, tauout

    # ------------------------------------------------------------- 追索
    def _solve_recourse(self, p, I, D):
        W_in = np.asarray(self.prob.N, dtype=float) - \
            self._n[p].sum(axis=1)
        F_new = D[p] - np.maximum(W_in, 0.0)
        st = State(I[p], self._o[p], self._n[p])
        lp = RecourseLP(
            self.prob, st, self._qout[p], self._tau_first[p], F_new,
            time_limit=self.time_limit,
            solver=self.solver, threads=self.threads)
        return p, lp.solve()

    def fulfill(self, t: int, I, D):
        if self._t != int(t):
            raise RuntimeError("fulfill 必须在同一期的 order 之后调用")
        I = np.asarray(I, dtype=float)
        D = np.asarray(D, dtype=float)
        P = I.shape[0]
        if D.shape != (P, self.N_R):
            raise ValueError(f"D shape {D.shape}")
        y = np.zeros((P, self.N_R, self.N_R), dtype=float)
        tau = np.zeros((P, self.N_R), dtype=float)
        results = {}
        failures = []

        def run_one(p):
            try:
                return self._solve_recourse(p, I, D)
            except Exception as exc:  # noqa: BLE001
                return p, exc

        if self.workers > 1 and P > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(run_one, p) for p in range(P)]
                for fut in as_completed(futs):
                    p, rr = fut.result()
                    if isinstance(rr, Exception):
                        failures.append((p, rr))
                    else:
                        results[p] = rr
        else:
            for p in range(P):
                p2, rr = run_one(p)
                if isinstance(rr, Exception):
                    failures.append((p2, rr))
                else:
                    results[p2] = rr
        if failures:
            p, exc = failures[0]
            self.lp_failures.append({
                "path": int(p), "t": int(t), "error": str(exc),
                "info": dict(getattr(exc, "info", {})),
            })
            raise exc

        for p in range(P):
            rr = results[p]
            y[p] = rr.y
            tau_p = np.asarray(rr.tau, dtype=float).copy()
            tau_p -= tau_p.mean()
            tau[p] = tau_p
            stt = rr.stats
            self._total_lp_solves += 1
            self.lp_stats.append({
                "path": int(p), "t": int(t), "window": 1, "n_scen": 1,
                "n_vars": stt.n_vars, "n_eq": stt.n_eq,
                "n_le": stt.n_le, "build_s": stt.build_s,
                "solve_s": stt.solve_s, "objective": stt.objective,
                "solver": stt.solver, "status": "recourse",
            })
        self._y = y
        self._tau = tau
        return y

    def active_transfer(self, t, S_end, W_out, n_out, o_next, q_out, theta):
        if self._tau is None:
            raise RuntimeError("active_transfer 必须在 fulfill 之后调用")
        if self._tau.shape != np.asarray(S_end).shape:
            raise RuntimeError(
                f"tau shape {self._tau.shape} 与 S_end "
                f"{np.asarray(S_end).shape} 不一致")
        return self._tau.copy()

    # ------------------------------------------------------------- 元数据
    def params(self) -> dict:
        lp = self.lp_stats
        two = [r for r in lp if r["status"] == "ok"]
        max_vars = max((r["n_vars"] for r in two), default=0)
        max_row = max((r["n_eq"] + r["n_le"] for r in two), default=0)
        return {
            "kind": "rolling_horizon",
            "T": self.T, "N_R": self.N_R, "G": self.G, "L": self.L,
            "n_scenarios": self.Nsc,
            "seed_opt": self.seed_opt,
            "pool_hash": self.pool_hash,
            "pool_shape": list(self.pool.shape),
            "state": "I,n,o1..o(L-1)",
            "purchase_timing": "current_q",
            "capacity_side": "period_and_next",
            "solver": self.solver,
            "threads": self.threads,
            "workers": self.workers,
            "time_limit": self.time_limit,
            "window": "full_remaining",
            "terminal": "zero",
            "active": self.active,
            "n_lp_calls": int(self._total_lp_solves),
            "n_two_stage_calls": len(two),
            "n_recourse_calls": len(lp) - len(two),
            "max_two_stage_vars": int(max_vars),
            "max_two_stage_rows": int(max_row),
            "two_stage_build_s_total": float(sum(
                r["build_s"] for r in two)),
            "two_stage_solve_s_total": float(sum(
                r["solve_s"] for r in two)),
            "recourse_solve_s_total": float(sum(
                r["solve_s"] for r in lp if r["status"] == "recourse")),
            "lp_failures": len(self.lp_failures),
        }
