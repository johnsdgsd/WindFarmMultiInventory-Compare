"""Probe a known COPT status=5 two-stage LP and compare solve strategies.

Rebuilds the exact state from an archived simulation npz (I_in/o_in/n_in)
and solves the same LP with COPT/HiGHS settings.

Usage:
  "$P" -B rolling_horizon/probe_numeric.py \
    --run results/rolling_horizon/tau_W20_M50_rand_20261029_170958 \
    --path 32 --t 26 --window-len 20
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from rolling_horizon.lp import State, TwoStageLP, _SparseLP  # noqa: E402
from rolling_horizon.scenarios import make_opt_pool  # noqa: E402

DEFAULT_CFG = ROOT / "config" / "compare_T60_base204105.toml"


def _load_state(run_dir: Path, p: int, t: int):
    z = np.load(run_dir / "sim_50paths.npz", allow_pickle=False)
    I = np.asarray(z["I_in"][p, t - 1], dtype=float)
    o = np.asarray(z["o_in"][p, t - 1], dtype=float)
    n = np.asarray(z["n_in"][p, t - 1], dtype=float)
    return State(I, o, n)


def _copt_solve(sp: _SparseLP, *, threads: int, time_limit: float,
                presolve: int = -1, lpmethod: int = -1,
                scaling: int = -1) -> dict:
    import rolling_horizon.lp as lpm
    cp, COPT = lpm._copt()
    m = lpm._COPT_ENV.createModel()
    try:
        m.setParam(COPT.Param.Logging, 0)
        m.setParam(COPT.Param.Threads, int(threads))
        if time_limit > 0:
            m.setParam(COPT.Param.TimeLimit, float(time_limit))
        if presolve >= 0:
            m.setParam(COPT.Param.Presolve, int(presolve))
        if lpmethod >= 0:
            m.setParam(COPT.Param.LpMethod, int(lpmethod))
        if scaling >= 0:
            m.setParam(COPT.Param.Scaling, int(scaling))
        A, lhs, rhs = sp._matrix()
        lb = [float(v) for v in sp.lb]
        ub = [COPT.INFINITY if not np.isfinite(v) else float(v)
              for v in sp.ub]
        m.loadMatrix(np.asarray(sp.c, dtype=float), A, lhs, rhs, lb, ub)
        m.setObjSense(COPT.MINIMIZE)
        t0 = time.perf_counter()
        m.solveLP()
        status = int(m.status)
        obj = (float(m.getAttr(COPT.Attr.LpObjVal))
               if status == COPT.OPTIMAL else float("nan"))
        return {"solver": "copt", "threads": threads,
                "presolve": presolve, "lpmethod": lpmethod,
                "status": status, "objective": obj,
                "wall_s": time.perf_counter() - t0}
    finally:
        try:
            m.clear()
        except Exception:
            pass


def _highs_solve(sp: _SparseLP, *, method: str,
                 time_limit: float) -> dict:
    from scipy.optimize import linprog
    A, lhs, rhs = sp._matrix()
    n_eq = len(sp.eq_rows)
    n_le = len(sp.le_rows)
    eq = slice(0, n_eq)
    le = slice(n_eq, n_eq + n_le)
    bounds = [(lb, None if not np.isfinite(ub) else ub)
              for lb, ub in zip(sp.lb, sp.ub)]
    opts = {"presolve": True, "disp": False,
            "time_limit": float(time_limit)}
    t0 = time.perf_counter()
    res = linprog(np.asarray(sp.c, dtype=float),
                  A_ub=A[le, :] if n_le else None,
                  b_ub=rhs[le] if n_le else None,
                  A_eq=A[eq, :] if n_eq else None,
                  b_eq=rhs[eq] if n_eq else None,
                  bounds=bounds, method=method, options=opts)
    return {"solver": f"highs-{method}", "status": int(res.status),
            "message": str(res.message), "objective": float(res.fun)
            if res.success else float("nan"),
            "wall_s": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--run", required=True)
    ap.add_argument("--path", type=int, required=True)
    ap.add_argument("--t", type=int, required=True)
    ap.add_argument("--window-len", type=int, default=20)
    ap.add_argument("--n-scenarios", type=int, default=10)
    ap.add_argument("--seed-opt", type=int, default=2027)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--modes", default="copt1,copt3,copt1-p0,copt1-p2,"
                                       "highs-ds,highs-ipm,highs")
    args = ap.parse_args()
    cfg = load_compare_config(Path(args.config))
    prob = load_problem(cfg)
    opt = make_opt_pool(prob, args.n_scenarios, args.seed_opt)
    state = _load_state(Path(args.run), args.path, args.t)
    t_end = min(prob.T, args.t + args.window_len - 1)
    theta = opt["pool"][:, args.t - 1:t_end, :, :, :]
    lp = TwoStageLP(prob, state, args.t, theta,
                    window_len=args.window_len, solver="copt", threads=1)
    lp.build()
    sp = lp.lp
    print(f"path={args.path} t={args.t} window={lp.window} "
          f"vars={len(sp.c)} eq={len(sp.eq_rows)} le={len(sp.le_rows)}",
          flush=True)
    for mode in args.modes.split(","):
        if mode == "api":
            t0 = time.perf_counter()
            res = lp.solve()
            print(json.dumps({
                "solver": "api-presolve-retry",
                "status": 1,
                "objective": float(res.objective),
                "solve_s": float(time.perf_counter() - t0),
                "stats_solve_s": float(res.stats.solve_s),
            }, ensure_ascii=False), flush=True)
            continue
        if mode.startswith("copt"):
            parts = mode.split("-")
            cfg_s = parts[0]
            threads = int(cfg_s[4:])
            presolve = -1
            for part in parts[1:]:
                if part.startswith("p"):
                    presolve = int(part[1:])
            out = _copt_solve(sp, threads=threads,
                              time_limit=args.time_limit,
                              presolve=presolve)
        else:
            out = _highs_solve(sp, method=mode, time_limit=args.time_limit)
        print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
