from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from common.report import (copy_configs, provenance, write_json,
                           write_text)

INDEX_FIELDS = [
    "run_id", "kind", "status", "created_at", "config", "T", "n_paths",
    "algorithm", "selected_start", "train_cost", "val_cost", "test_cost",
    "baseline_mean", "diff_mean", "rel_pct", "win_rate", "audit_ok",
    "policy_json", "summary_md",
]


def _scipy_version() -> str | None:
    try:
        import scipy
        return scipy.__version__
    except Exception:
        return None


@dataclass
class RunArchive:
    dir: Path
    run_id: str
    kind: str
    tag: str
    created_at: str
    config: str
    manifest: dict = field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_json(self, name: str, obj) -> Path:
        p = self.path(name)
        write_json(p, obj)
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        write_text(p, text)
        return p

    def write_npz(self, name: str, **arrays) -> Path:
        p = self.path(name)
        if p.exists():
            raise FileExistsError(f"拒绝覆盖已有 npz: {p}")
        np.savez_compressed(p, **arrays)
        return p

    def copy_configs(self, cfg, project_dir, policy_path=None) -> None:
        copy_configs(self.dir, cfg, project_dir, policy_path=policy_path)

    def write_provenance(self, cfg, prob, policy_params,
                         baseline=None) -> Path:
        return self.write_json(
            "provenance.json",
            provenance(cfg, prob, policy_params, baseline=baseline))

    def finalize(self, status: str = "completed",
                 extra: dict | None = None,
                 index_row: dict | None = None) -> None:
        payload = dict(self.manifest)
        payload.update({
            "run_id": self.run_id,
            "kind": self.kind,
            "tag": self.tag,
            "created_at": self.created_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": _scipy_version(),
            "platform": platform.platform(),
        })
        if extra:
            payload.update(extra)
        self.write_json("run.json", payload)
        if index_row is not None:
            row = dict(index_row)
            row.setdefault("run_id", self.run_id)
            row.setdefault("kind", self.kind)
            row.setdefault("status", status)
            row.setdefault("created_at", self.created_at)
            append_index(self.dir.parent, row)


def output_tag(tag, pcfg: dict, T: int) -> str:
    """默认 tag 若带 T60/T240 字样，自动替换为当前 T，避免 T240 写成 T60。"""
    if tag is not None:
        return str(tag)
    out = str(pcfg["output"]["tag"])
    if f"T{int(T)}" in out:
        return out
    if re.search(r"T\d+", out):
        return re.sub(r"T\d+", f"T{int(T)}", out)
    return f"{out}_T{int(T)}"


def _make_run_dir(results_root, tag: str, kind: str) -> Path:
    """<results_root>/ss_static/<tag>_<kind>_<timestamp>；拒绝覆盖。"""
    base = Path(results_root) / "ss_static"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"{tag}_{kind}_{ts}"
    i = 1
    while d.exists():
        d = base / f"{tag}_{kind}_{ts}_{i}"
        i += 1
    d.mkdir()
    return d


def create_run(cfg, project_dir, kind: str, tag: str,
               archive_root=None, *, manifest: dict | None = None) -> RunArchive:
    root = (Path(archive_root) if archive_root is not None
            else Path(cfg.root) / "results")
    d = _make_run_dir(root, tag, kind)
    run = RunArchive(
        dir=d, run_id=d.name, kind=str(kind), tag=str(tag),
        created_at=datetime.now().isoformat(timespec="seconds"),
        config=str(cfg.config_path),
        manifest=dict(manifest or {}),
    )
    run.write_json("run.json", {
        "run_id": run.run_id, "kind": run.kind, "tag": run.tag,
        "created_at": run.created_at, "status": "running",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": _scipy_version(), "platform": platform.platform(),
        **(manifest or {}),
    })
    return run


def append_index(project_results_dir: Path, row: dict) -> Path:
    """追加一行到 <results>/ss_static/index.csv；绝不覆盖旧行。"""
    base = Path(project_results_dir)
    base.mkdir(parents=True, exist_ok=True)
    p = base / "index.csv"
    is_new = not p.is_file() or p.stat().st_size == 0
    clean = {k: row.get(k, "") for k in INDEX_FIELDS}
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(clean)
    return p


def index_path(archive_root, cfg) -> Path:
    root = Path(archive_root) if archive_root is not None \
        else Path(cfg.root) / "results"
    return root / "ss_static" / "index.csv"
