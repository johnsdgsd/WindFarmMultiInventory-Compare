from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompareConfig:
    root: Path
    config_path: Path
    model_root: Path
    model_commit: str
    problem_data: Path
    problem_toml: Path
    solver_toml: Path
    baseline_name: str
    baseline_dir: Path
    baseline_npz: Path
    baseline_allow_mismatch: bool
    n_paths: int
    seed: int
    chunk_size: int
    tol: float
    state_tol: float
    train_seed: int
    val_seed: int
    train_paths: int
    val_paths: int
    calib_chunk_size: int


def _table(d: dict, key: str, section: str) -> dict:
    if key not in d:
        raise KeyError(f"共享配置缺少 [{section}] 节")
    v = d[key]
    if not isinstance(v, dict):
        raise TypeError(f"[{section}] 必须是表")
    return v


def _req(d: dict, key: str, section: str):
    if key not in d:
        raise KeyError(f"共享配置缺少 [{section}] {key}")
    return d[key]


def _positive_int(v, name: str) -> int:
    iv = int(v)
    if iv <= 0:
        raise ValueError(f"{name} 必须为正整数，实际 {v}")
    return iv


def _nonnegative_int(v, name: str) -> int:
    iv = int(v)
    if iv < 0:
        raise ValueError(f"{name} 必须为非负整数，实际 {v}")
    return iv


def _nonnegative_float(v, name: str) -> float:
    fv = float(v)
    if fv < 0:
        raise ValueError(f"{name} 必须非负，实际 {v}")
    return fv


def load_compare_config(path: str | Path | None = None) -> CompareConfig:
    root = Path(__file__).resolve().parents[1]
    chosen = path if path is not None else os.environ.get("COMPARE_CONFIG")
    p = Path(chosen) if chosen is not None else root / "config" / "compare.toml"
    if not p.is_absolute():
        p = root / p
    if not p.is_file():
        raise FileNotFoundError(f"共享配置不存在: {p}")
    data = tomllib.loads(p.read_text(encoding="utf-8"))

    model = _table(data, "model", "model")
    baseline = _table(data, "baseline", "baseline")
    evaluation = _table(data, "evaluation", "evaluation")
    calibration = _table(data, "calibration", "calibration")

    model_root = (root / str(_req(model, "root", "model"))).resolve()
    if not model_root.is_dir():
        raise FileNotFoundError(f"model.root 不存在: {model_root}")

    baseline_dir = model_root / str(_req(baseline, "result_dir", "baseline"))
    baseline_npz = baseline_dir / str(_req(baseline, "npz", "baseline"))
    problem_data = model_root / str(_req(model, "problem_data", "model"))
    problem_toml = model_root / str(_req(model, "problem_toml", "model"))
    solver_toml = model_root / str(_req(model, "solver_toml", "model"))
    if not solver_toml.is_file():
        raise FileNotFoundError(f"solver_toml 不存在: {solver_toml}")
    if not problem_toml.is_file() and not problem_data.is_file():
        raise FileNotFoundError(
            f"problem_data.json 与 problem.toml 都不存在: {problem_data} / {problem_toml}")
    if not baseline_npz.is_file():
        raise FileNotFoundError(f"baseline npz 不存在: {baseline_npz}")

    allow_mismatch = bool(_req(baseline, "allow_model_mismatch", "baseline"))
    n_paths = _positive_int(_req(evaluation, "n_paths", "evaluation"), "n_paths")
    seed = int(_req(evaluation, "seed", "evaluation"))
    chunk_size = _positive_int(_req(evaluation, "chunk_size", "evaluation"), "chunk_size")
    tol = _nonnegative_float(_req(evaluation, "tol", "evaluation"), "tol")
    state_tol = _nonnegative_float(_req(evaluation, "state_tol", "evaluation"), "state_tol")

    return CompareConfig(
        root=root,
        config_path=p,
        model_root=model_root,
        model_commit=str(_req(model, "commit", "model")),
        problem_data=problem_data,
        problem_toml=problem_toml,
        solver_toml=solver_toml,
        baseline_name=str(_req(baseline, "name", "baseline")),
        baseline_dir=baseline_dir,
        baseline_npz=baseline_npz,
        baseline_allow_mismatch=allow_mismatch,
        n_paths=n_paths,
        seed=seed,
        chunk_size=chunk_size,
        tol=tol,
        state_tol=state_tol,
        train_seed=int(_req(calibration, "train_seed", "calibration")),
        val_seed=int(_req(calibration, "val_seed", "calibration")),
        train_paths=_positive_int(_req(calibration, "train_paths", "calibration"),
                                  "train_paths"),
        val_paths=_positive_int(_req(calibration, "val_paths", "calibration"),
                                "val_paths"),
        calib_chunk_size=_positive_int(_req(calibration, "chunk_size", "calibration"),
                                       "calibration.chunk_size"),
    )


def load_policy_config(project_dir: str | Path,
                       path: str | Path | None = None) -> dict:
    """读取子项目自己的 config/policy.toml；缺键直接报错，不静默兜底。

    path 非空时作为替代策略配置（如 policy_active.toml）；
    相对路径按项目根目录解析。
    """
    project_dir = Path(project_dir).resolve()
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = project_dir.parent / p
    else:
        p = project_dir / "config" / "policy.toml"
    if not p.is_file():
        raise FileNotFoundError(f"策略配置不存在: {p}")
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    for key in ("policy", "search", "output"):
        if key not in data or not isinstance(data[key], dict):
            raise KeyError(f"策略配置缺少 [{key}] 节: {p}")
    return data


def project_root(project_dir: str | Path) -> Path:
    """子项目目录的上一级就是对比实验根目录。"""
    return Path(project_dir).resolve().parent
