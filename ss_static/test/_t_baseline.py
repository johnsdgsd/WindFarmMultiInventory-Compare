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

from common.baseline import (load_baseline, paired_stats,  # noqa: E402
                             term_stats)
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ss_static.metrics import baseline_behavior  # noqa: E402

sys.stdout.reconfigure(errors="replace")


def main():
    cfg = load_compare_config()
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)
    if base.P != cfg.n_paths:
        raise AssertionError("路径数不一致")
    expected_mean = {
        (2000, 60): 825.4215340961761,      # 旧 T60 基线
        (3000, 60): 35833.32280094246,     # base_T60_20261026_204105（3000 路径）
        (1000, 240): 159380.87234937897,   # T240
    }.get((base.P, int(prob.T)))
    if expected_mean is not None and abs(base.mean - expected_mean) > 1e-6:
        raise AssertionError(
            f"基线均值 {base.mean} 与已知基线 {expected_mean} 不符")
    if not base.problem_data_match:
        raise AssertionError("基线 problem_data.json 与共享配置不匹配")

    ps = paired_stats(base.totals, base.totals)
    if abs(ps["diff_mean"]) > 1e-12 or ps["win_rate_policy_lower"] != 0.0:
        raise AssertionError(f"自配对统计异常: {ps}")
    ts = term_stats(base.terms, base.terms)
    if any(abs(z["diff_mean"]) > 1e-12 for z in ts):
        raise AssertionError("自配对五项成本差不为 0")
    beh = baseline_behavior(prob, base)
    print("_t_baseline: all checks passed")
    print(f"  mean={base.mean:.4f} std={base.std:.4f} ci95=±{base.ci95:.4f}")
    print(f"  failures={beh['total_failures']:.4f} "
          f"replacements={beh['total_replacements']:.4f} "
          f"final_backlog={beh['final_backlog']:.4f}")
    print(f"  orders_placed={beh['total_orders_placed']:.4f} "
          f"charged={beh['total_orders_charged']:.4f} "
          f"q0/路径={base.q0(prob.layout).sum() / base.P:.4f}")


if __name__ == "__main__":
    main()
