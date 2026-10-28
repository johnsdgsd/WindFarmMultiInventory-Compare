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

from rolling_horizon.evaluate import evaluate  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="滚动时域（全窗口两阶段 SAA）样本外评估与 SDDP 配对")
    ap.add_argument("--config", default=None,
                    help="共享配置；缺省 COMPARE_CONFIG 或 "
                         "config/compare_T60_base204105.toml")
    ap.add_argument("--policy-config", default=None,
                    help="滚动策略配置；缺省 rolling_horizon/config/policy.toml")
    ap.add_argument("--tag", default=None, help="结果目录前缀")
    ap.add_argument("--n-scenarios", type=int, default=None,
                    help="优化池情景数 N（缺省取策略配置）")
    ap.add_argument("--real-paths", type=int, default=None,
                    help="样本外真实路径数 M（缺省取策略配置）")
    ap.add_argument("--seed-opt", type=int, default=None,
                    help="优化池种子（缺省取策略配置）")
    ap.add_argument("--threads", type=int, default=None,
                    help="COPT 单模型线程（缺省取策略配置）")
    ap.add_argument("--workers", type=int, default=None,
                    help="按真实路径并行的线程数（缺省取策略配置）")
    ap.add_argument("--log-every", type=int, default=None,
                    help="每多少期写一次进度日志（缺省取策略配置）")
    ap.add_argument("--solver", default=None, choices=["copt", "highs"],
                    help="LP 求解器（缺省 copt）")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="单次 LP 时间限制（秒；0 表示不限）")
    ap.add_argument("--archive-root", default=None,
                    help="归档根目录；缺省 <项目>/results")
    ap.add_argument("--quick", action="store_true",
                    help="N=smoke_n、M=smoke_m 的快速 smoke；目录带 _quick")
    ap.add_argument("--skip-audit", action="store_true",
                    help="跳过逐点审计（仅 smoke）")
    args = ap.parse_args()

    evaluate(
        ROOT,
        config_path=args.config,
        policy_config=args.policy_config,
        tag=args.tag,
        n_scenarios=args.n_scenarios,
        real_paths=args.real_paths,
        seed_opt=args.seed_opt,
        archive_root=args.archive_root,
        quick=args.quick,
        threads=args.threads,
        workers=args.workers,
        log_every=args.log_every,
        time_limit=args.time_limit,
        solver=args.solver,
        skip_audit=args.skip_audit,
    )
