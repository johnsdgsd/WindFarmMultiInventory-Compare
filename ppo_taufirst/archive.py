"""PPO 子项目的归档小工具（运行目录 + append-only index.csv）。"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

INDEX_FIELDS = [
    "run_id", "kind", "status", "T", "steps", "seed", "M",
    "wall_s", "mean_episode_cost", "policy_mean", "baseline_mean",
    "diff_mean", "rel_diff_pct", "audit_passed", "policy", "created_at",
]


def make_run_dir(tag: str, archive_root: str | Path = "results") -> Path:
    root = Path(archive_root)
    if not root.is_absolute():
        root = PROJECT / root
    base = root / "ppo_taufirst"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"{tag}_{ts}"
    i = 1
    while d.exists():
        d = base / f"{tag}_{ts}_{i}"
        i += 1
    d.mkdir()
    return d


def append_index(run_dir: str | Path, row: dict) -> Path:
    """在 <archive>/ppo_taufirst/index.csv 追加一行；不覆盖历史。"""
    run_dir = Path(run_dir)
    path = run_dir.parent / "index.csv"
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if not exists:
            w.writeheader()
        clean = {k: row.get(k, "") for k in INDEX_FIELDS}
        w.writerow(clean)
    return path
