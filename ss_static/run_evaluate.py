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

from ss_static.evaluate import main  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="评估时间无关 (s,S) 并配对 SDDP 基线")
    ap.add_argument("--policy", required=True,
                    help="policy.json 路径（必填；避免误取 test/smoke 或其他 T 的策略）")
    ap.add_argument("--tag", default=None, help="结果目录前缀")
    ap.add_argument("--archive-root", default=None,
                    help="归档根目录；缺省 <项目>/results")
    ap.add_argument("--policy-config", default=None,
                    help="归档用的策略 TOML（如 active 配置）；缺省 policy.toml")
    ap.add_argument("--skip-audit", action="store_true",
                    help="跳过逐点约束审计（只用于快速试跑）")
    args = ap.parse_args()
    main(PROJECT, Path(args.policy), tag=args.tag,
         archive_root=args.archive_root, skip_audit=args.skip_audit,
         policy_config=args.policy_config)
