from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from common.config import load_compare_config, load_policy_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ss_static.evaluation import (SSEvaluator, SSParams,  # noqa: E402
                                  evaluate_cost)
from ss_static.simulator import generate_scene_paths  # noqa: E402

sys.stdout.reconfigure(errors="replace")

PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def main():
    global PASS
    cfg = load_compare_config()
    prob = load_problem(cfg)
    pcfg = load_policy_config(PROJECT)
    opts = dict(pcfg["policy"])
    paths = generate_scene_paths(6, 123, 1000, len(prob.pool), prob.T)
    ev = SSEvaluator(prob, opts)
    theta = np.concatenate([np.full(prob.N_R, 0.5),
                            np.full(prob.N_R, 1.0)])

    r = ev.evaluate(theta, paths, record=True)
    check(r.path_costs.shape == (6,), f"path_costs shape {r.path_costs.shape}")
    check(r.P == 6 and r.T == prob.T, f"P/T {r.P}/{r.T}")
    check(abs(r.cost - float(np.mean(r.path_costs))) < 1e-12, "cost != mean")
    check(r.term_totals.shape == (6, 5), f"term_totals {r.term_totals.shape}")
    check(float(np.max(np.abs(r.term_totals.sum(axis=1) - r.path_costs))) < 1e-9,
          "五项和 != path cost")
    check(r.step_terms.shape == (6, prob.T, 5),
          f"step_terms {r.step_terms.shape}")
    check(r.behavior is not None and len(r.behavior) >= 10, "behavior 缺失")
    check(r.arrays is not None and "x_in" in r.arrays, "arrays 缺失")

    expected_inv = float(np.mean(
        np.asarray(r.arrays["x_in"])[:, :, :prob.N_R].sum(axis=2)))
    check(abs(r.behavior["mean_inventory"] - expected_inv) < 1e-9,
          "mean_inventory 不是五仓合计")

    r2 = ev.evaluate(theta, paths, record=True)
    check(abs(r.cost - r2.cost) < 1e-12, "评价不确定")
    check(abs(ev.evaluate_cost(theta, paths) - r.cost) < 1e-12,
          "evaluate_cost != evaluate().cost")
    check(abs(evaluate_cost(prob, opts, theta, paths) - r.cost) < 1e-12,
          "纯函数 evaluate_cost 不一致")

    r3 = ev.evaluate(theta, paths, record=False)
    check(r3.step_terms is None and r3.arrays is None and r3.behavior is None,
          "record=False 不应返回明细")
    check(r3.term_totals.shape == (6, 5) and
          float(np.max(np.abs(r3.term_totals.sum(axis=1) - r3.path_costs))) < 1e-9,
          "record=False 的 term_totals 不对")

    sp = SSParams.from_theta(theta, prob.N_R)
    check(np.allclose(sp.theta, theta), "SSParams round-trip")

    bad = [
        (np.concatenate([np.full(prob.N_R, 2.0), np.full(prob.N_R, 1.0)]),
         "s>S 应被拒绝"),
        (np.concatenate([np.full(prob.N_R, 0.0),
                         np.asarray(prob.Imax) + 1.0]), "S>Imax 应被拒绝"),
        (np.zeros(2 * prob.N_R + 1), "长度错误应被拒绝"),
        (np.full(2 * prob.N_R, np.nan), "NaN 应被拒绝"),
    ]
    for t, msg in bad:
        try:
            ev.check_theta(t)
        except ValueError:
            PASS += 1
        else:
            raise AssertionError(msg)

    print(f"_t_evaluation: {PASS} checks passed")
    print(f"  cost={r.cost:.4f}  mean_inventory={r.behavior['mean_inventory']:.4f}")


if __name__ == "__main__":
    main()
