"""Benchmark two-stage LP throughput for rolling-horizon parallelism.

Times a fixed batch of t=1 two-stage LPs (full window) for several
(workers, threads, solver) settings.  The estimate for a full 60-period
run applies the average/t=1 ratio observed on the W=20/M=50 archive.

Usage:
  "$P" -B rolling_horizon/bench_parallel.py --n-paths 48
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from rolling_horizon.policy import RollingHorizonPolicy  # noqa: E402
from rolling_horizon.scenarios import make_opt_pool  # noqa: E402

DEFAULT_CFG = ROOT / "config" / "compare_T60_base204105.toml"
# Mean of W=20/M=50 over all 60 periods divided by its t=1 mean.
AVG_OVER_T1 = 2.19 / 1.84


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--n-paths", type=int, default=48)
    ap.add_argument("--n-scenarios", type=int, default=10)
    ap.add_argument("--seed-opt", type=int, default=2027)
    ap.add_argument("--window-len", type=int, default=20)
    ap.add_argument("--configs", default="copt:8:3,copt:8:1,copt:14:1,"
                                         "copt:20:1,copt:24:1,highs:8:1")
    ap.add_argument("--target-paths", type=int, default=3000)
    ap.add_argument("--periods", type=int, default=60)
    args = ap.parse_args()

    cfg = load_compare_config(Path(args.config))
    prob = load_problem(cfg)
    opt = make_opt_pool(prob, args.n_scenarios, args.seed_opt)
    P = int(args.n_paths)
    I = np.tile(np.asarray(prob.I0, dtype=float)[None, :], (P, 1))
    o = np.tile(np.asarray(prob.o0, dtype=float).reshape(
        prob.L - 1, prob.N_R)[None, :, :], (P, 1, 1))
    n = np.tile(np.asarray(prob.n0, dtype=float)[None, :, :], (P, 1, 1))

    n_two = int(args.target_paths) * int(args.periods)
    print(f"paths={P} N={args.n_scenarios} W={args.window_len} "
          f"target={n_two} two-stage LPs", flush=True)
    for spec in args.configs.split(","):
        solver, workers, threads = spec.split(":")
        workers, threads = int(workers), int(threads)
        opts = {
            "solver": solver, "fallback_solver": "highs",
            "threads": threads, "workers": workers,
            "window_len": int(args.window_len), "time_limit": 0.0,
            "verbose": False, "log_every": 1, "active": "lp",
        }
        pol = RollingHorizonPolicy(prob, opts, opt["pool"],
                                   seed_opt=args.seed_opt,
                                   pool_hash=opt["pool_hash"])
        # Warm up COPT/env and BLAS; not timed.
        pol.order(1, I[:2], o[:2], n[:2])
        pol.lp_stats.clear()
        t0 = time.perf_counter()
        pol.order(1, I, o, n)
        wall = time.perf_counter() - t0
        two = [r for r in pol.lp_stats if r["status"] == "ok"]
        mean_lp = wall / max(len(two), 1)
        est_s = wall / max(P, 1) * n_two * AVG_OVER_T1
        b = float(np.mean([r["build_s"] for r in two]))
        s = float(np.mean([r["solve_s"] for r in two]))
        print(f"{solver:5s} workers={workers:2d} threads={threads} "
              f"wall={wall:7.2f}s lp={len(two)} "
              f"per_lp={mean_lp:6.3f}s build={b:5.3f}s "
              f"solve={s:6.3f}s est_full={est_s / 3600:6.2f}h",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
