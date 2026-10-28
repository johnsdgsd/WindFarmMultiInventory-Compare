from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from common.baseline import sha256_file


def make_result_dir(root: str | Path, project: str, tag: str) -> Path:
    base = Path(root) / "results" / project
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"{tag}_{ts}"
    i = 1
    while d.exists():
        d = base / f"{tag}_{ts}_{i}"
        i += 1
    d.mkdir()
    return d


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"无法序列化 {type(o)}")


def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def copy_configs(result_dir: str | Path, cfg, project_dir: str | Path,
                 policy_path: str | Path | None = None) -> None:
    """结果目录自包含：共享配置、策略配置、模型配置副本。"""
    d = Path(result_dir) / "config"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.config_path, d / "compare.toml")
    pp = Path(policy_path) if policy_path is not None else \
        Path(project_dir) / "config" / "policy.toml"
    if pp.is_file():
        shutil.copy2(pp, d / "policy.toml")
    if Path(cfg.problem_toml).is_file():
        shutil.copy2(cfg.problem_toml, d / "problem.toml")
    if Path(cfg.solver_toml).is_file():
        shutil.copy2(cfg.solver_toml, d / "solver.toml")
    if Path(cfg.problem_data).is_file():
        shutil.copy2(cfg.problem_data, d / "problem_data.json")


def provenance(cfg, prob, policy_params: dict, baseline=None) -> dict:
    from common.model import verify_model
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "compare_config": str(cfg.config_path),
        "compare_config_hash": sha256_file(cfg.config_path),
        "problem_data": str(cfg.problem_data),
        "problem_data_hash": sha256_file(cfg.problem_data)
        if Path(cfg.problem_data).is_file() else None,
        "problem_toml": str(cfg.problem_toml),
        "problem_toml_hash": sha256_file(cfg.problem_toml)
        if Path(cfg.problem_toml).is_file() else None,
        "solver_toml": str(cfg.solver_toml),
        "solver_toml_hash": sha256_file(cfg.solver_toml),
        "model": verify_model(cfg),
        "policy": policy_params,
        "problem": {
            "T": int(prob.T), "G": int(prob.G), "N_R": int(prob.N_R),
            "L": int(prob.L), "state_dim": int(prob.state_dim),
            "n_scenes": int(len(prob.pool)),
        },
    }
    if baseline is not None:
        out["baseline"] = {
            "name": baseline.name,
            "result_dir": str(baseline.result_dir),
            "npz": str(baseline.npz_path),
            "npz_hash": baseline.npz_hash,
            "scene_hash": baseline.scene_hash,
            "problem_data_hash": baseline.problem_data_hash,
            "model_problem_data_hash": baseline.model_problem_data_hash,
            "problem_data_match": baseline.problem_data_match,
            "n_paths": baseline.P,
            "mean": baseline.mean,
            "std": baseline.std,
            "ci95": baseline.ci95,
        }
    return out


def save_sim_npz(result_dir: str | Path, res, n_paths: int) -> Path:
    d = Path(result_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"sim_{n_paths}paths.npz"
    if p.exists():
        raise FileExistsError(f"拒绝覆盖已有 npz: {p}")
    np.savez_compressed(p, **res.arrays)
    return p


def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(v):
        return "nan"
    return f"{v:.{nd}f}"
