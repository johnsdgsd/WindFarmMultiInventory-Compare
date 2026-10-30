"""Compare COPT Presolve=auto vs Presolve=0 on archived W=20 states.

Builds each LP once and solves it twice (auto, 0).  Used to verify that
disabling presolve removes status=5 without changing objectives.

Usage:
  "$P" -B rolling_horizon/verify_presolve.py \
    --run results/rolling_horizon/tau_W20_M50_rand_20261029_170958
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from rolling_horizon.lp import State, TwoStageLP  # noqa: E402
from rolling_horizon.probe_numeric import _copt_solve  # noqa: E402
from rolling_horizon.scenarios import make_opt_pool  # noqa: E402

DEFAULT_CFG = ROOT / "config" / "compare_T60_base204105.toml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--run", required=True)
    ap.add_argument("--window-len", type=int, default=20)
    ap.add_argument("--n-scenarios", type=int, default=10)
    ap.add_argument("--seed-opt", type=int, default=2027)
    ap.add_argument("--paths", default="0,7,13,21,32,41,49")
    ap.add_argument("--ts", default="1,11,21,31,41,51,56,60")
    ap.add_argument("--time-limit", type=float, default=120.0)
    args = ap.parse_args()
    cfg = load_compare_config(Path(args.config))
    prob = load_problem(cfg)
    opt = make_opt_pool(prob, args.n_scenarios, args.seed_opt)
    z = np.load(Path(args.run) / "sim_50paths.npz", allow_pickle=False)
    paths = [int(x) for x in args.paths.split(",")]
    ts = [int(x) for x in args.ts.split(",")]
    rows = []
    for p in paths:
        for t in ts:
            if t > prob.T:
                continue
            I = np.asarray(z["I_in"][p, t - 1], dtype=float)
            o = np.asarray(z["o_in"][p, t - 1], dtype=float)
            n = np.asarray(z["n_in"][p, t - 1], dtype=float)
            t_end = min(prob.T, t + args.window_len - 1)
            theta = opt["pool"][:, t - 1:t_end, :, :, :]
            lp = TwoStageLP(prob, State(I, o, n), t, theta,
                            window_len=args.window_len,
                            solver="copt", threads=1)
            lp.build()
            a = _copt_solve(lp.lp, threads=1,
                            time_limit=args.time_limit, presolve=-1)
            b = _copt_solve(lp.lp, threads=1,
                            time_limit=args.time_limit, presolve=0)
            rel = (abs(a["objective"] - b["objective"])
                   / max(abs(b["objective"]), 1.0)
                   if a["status"] == 1 and b["status"] == 1 else float("nan"))
            rows.append((p, t, a, b, rel))
            print(f"p={p:2d} t={t:2d} auto={a['status']} "
                  f"{a['wall_s']:.3f}s   p0={b['status']} "
                  f"{b['wall_s']:.3f}s  rel={rel:.2e}", flush=True)
    ok = [r for r in rows if r[2]["status"] == 1 and r[3]["status"] == 1]
    print("\nsummary")
    print(f"  n={len(rows)} auto_ok={sum(r[2]['status'] == 1 for r in rows)} "
          f"p0_ok={sum(r[3]['status'] == 1 for r in rows)}")
    print(f"  max_rel_diff={max((r[4] for r in ok), default=float('nan')):.3e}")
    print(f"  median_auto={statistics.median(r[2]['wall_s'] for r in rows):.3f}s "
          f"median_p0={statistics.median(r[3]['wall_s'] for r in rows):.3f}s")
    print(f"  mean_auto={statistics.mean(r[2]['wall_s'] for r in rows):.3f}s "
          f"mean_p0={statistics.mean(r[3]['wall_s'] for r in rows):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
