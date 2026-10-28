from __future__ import annotations

import sys

# 只读导入模型项目：禁止在模型仓库里生成/更新 .pyc
sys.dont_write_bytecode = True

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from common.config import CompareConfig

MODEL_CODE_FILES = (
    "code/config.py",
    "code/problem_data.py",
    "code/state_layout.py",
    "code/subproblem.py",
    "code/sim_analysis.py",
)

_MODEL: SimpleNamespace | None = None


def bootstrap(cfg: CompareConfig) -> SimpleNamespace:
    """把模型项目根放到 sys.path 最前，导入 code.*；只读。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    root = Path(cfg.model_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"模型项目根不存在: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import code.config as cc
    import code.problem_data as pd
    import code.sim_analysis as sa
    import code.state_layout as sl
    import code.subproblem as sp

    _MODEL = SimpleNamespace(cc=cc, pd=pd, sa=sa, sl=sl, sp=sp)
    return _MODEL


def load_problem(cfg: CompareConfig):
    """优先读基线结果里的 problem_data.json（精确场景池/参数）。"""
    m = bootstrap(cfg)
    if Path(cfg.problem_data).is_file():
        data = json.loads(Path(cfg.problem_data).read_text(encoding="utf-8"))
        return m.pd.from_json(data)
    pcfg = m.cc.load_problem_config(cfg.problem_toml)
    return m.pd.DefineProblem(pcfg)


def load_solver(cfg: CompareConfig):
    m = bootstrap(cfg)
    return m.cc.load_solver_config(cfg.solver_toml)


def code_hash(cfg: CompareConfig) -> str:
    h = hashlib.sha256()
    for rel in MODEL_CODE_FILES:
        p = Path(cfg.model_root) / rel
        h.update(rel.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def git_commit(cfg: CompareConfig) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(cfg.model_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def verify_model(cfg: CompareConfig) -> dict:
    """返回可落盘的模型 provenance；commit 不一致只警告，不阻断。"""
    commit = git_commit(cfg)
    dirty = None
    try:
        r = subprocess.run(
            ["git", "-C", str(cfg.model_root), "status", "--porcelain",
             "--", "code", "utils"],
            capture_output=True, text=True, timeout=10, check=False)
        if r.returncode == 0:
            dirty = bool(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    if commit is not None and not (
            commit.startswith(cfg.model_commit) or
            cfg.model_commit.startswith(commit)):
        print(f"[警告] 模型 HEAD={commit} ≠ 配置 commit={cfg.model_commit}")
    if dirty:
        print("[警告] 模型 code/ 或 utils/ 有未提交改动，结果可能不可复现")
    return {
        "model_root": str(cfg.model_root),
        "model_commit_configured": cfg.model_commit,
        "model_commit_actual": commit,
        "model_code_dirty": dirty,
        "model_code_hash": code_hash(cfg),
    }
