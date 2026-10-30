from __future__ import annotations

import sys

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parent
for p in (str(ROOT), str(PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse  # noqa: E402

from ss_static.calibrate import main  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="标定时间无关 (s,S) 基准策略")
    ap.add_argument("--tag", default=None, help="结果目录前缀")
    ap.add_argument("--quick", action="store_true",
                    help="60/60 路径、坐标下降、小预算的 smoke 运行")
    ap.add_argument("--integer", action="store_true",
                    help="只搜整数 s,S（敏感性；默认按 MD 连续搜索）")
    ap.add_argument("--algo", default=None,
                    help="算法：逗号分隔，或 all；缺省读 policy.toml。"
                         "可选 coordinate,random,nelder-mead,powell,"
                         "differential-evolution,dual-annealing,genetic,"
                         "simulated-annealing")
    ap.add_argument("--archive-root", default=None,
                    help="归档根目录；缺省 <项目>/results")
    ap.add_argument("--train-paths", type=int, default=None)
    ap.add_argument("--val-paths", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="优化器随机种子；缺省读 policy.toml")
    ap.add_argument("--budget-unique", type=int, default=None,
                    help="各算法约同一真实评估预算（算法对比用；0=用各自默认）")
    ap.add_argument("--policy-config", default=None,
                    help="替代策略 TOML，如 ss_static/config/policy_active.toml")
    ap.add_argument("--active", action="store_true",
                    help="开启主动调拨（active_transfer=true）")
    ap.add_argument("--active-threshold-init", type=float, default=None,
                    help="每仓保留/目标水平 h 的初值；h 本身是优化参数")
    ap.add_argument("--active-fraction", type=float, default=None,
                    help="每次调拨比例 alpha（默认 1.0）")
    ap.add_argument("--active-rule", default=None,
                    help="主动调拨规则（threshold_balance）")
    ap.add_argument("--active-stop-offset", type=int, default=None,
                    help="末期不调拨的期数（默认 2）")
    ap.add_argument("--active-priority", default=None,
                    help="deficit 或 deficit_risk")
    args = ap.parse_args()
    active_options = {
        "active_transfer": True if args.active else None,
        "active_threshold_init": args.active_threshold_init,
        "active_fraction": args.active_fraction,
        "active_rule": args.active_rule,
        "active_stop_offset": args.active_stop_offset,
        "active_priority": args.active_priority,
    }
    active_options = {k: v for k, v in active_options.items() if v is not None}
    main(PROJECT, tag=args.tag, quick=args.quick,
         integer=True if args.integer else None,
         algorithms=args.algo, archive_root=args.archive_root,
         train_paths=args.train_paths, val_paths=args.val_paths,
         seed=args.seed, budget_unique=args.budget_unique,
         policy_config=args.policy_config,
         active_options=active_options or None)
