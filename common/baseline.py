from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.config import CompareConfig


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@dataclass
class BaselineData:
    name: str
    result_dir: Path
    npz_path: Path
    scene: np.ndarray
    totals: np.ndarray
    terms: np.ndarray
    cost: np.ndarray
    x_in: np.ndarray
    q: np.ndarray
    taup: np.ndarray
    taum: np.ndarray
    S_end: np.ndarray
    y: np.ndarray
    n_out: np.ndarray
    F: np.ndarray
    W_in: np.ndarray
    W_out: np.ndarray
    problem_data_hash: str | None
    model_problem_data_hash: str | None
    problem_data_match: bool | None
    npz_hash: str
    scene_hash: str
    meta: dict

    @property
    def P(self) -> int:
        return int(self.totals.shape[0])

    @property
    def mean(self) -> float:
        return float(np.mean(self.totals))

    @property
    def std(self) -> float:
        return float(np.std(self.totals, ddof=1))

    @property
    def se(self) -> float:
        return self.std / np.sqrt(self.P)

    @property
    def ci95(self) -> float:
        return 1.959963984540054 * self.se

    def q0(self, layout) -> np.ndarray:
        """首期订货在 x_in[:,0] 的 q 块，不在 q 数组里。"""
        return np.asarray(self.x_in[:, 0, layout.q_slice()], dtype=float)


def load_baseline(cfg: CompareConfig, prob) -> BaselineData:
    npz_path = Path(cfg.baseline_npz)
    if not npz_path.is_file():
        raise FileNotFoundError(f"基线 npz 不存在: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        keys = set(z.files)
        need = {"scene", "totals", "terms", "cost", "x_in", "q", "taup",
                "taum", "S_end", "y", "n_out", "F", "W_in", "W_out"}
        miss = need - keys
        if miss:
            raise KeyError(f"基线 npz 缺字段: {sorted(miss)}")
        data = {k: np.asarray(z[k]) for k in need}

    scene = data["scene"]
    P, T = scene.shape
    if P != cfg.n_paths:
        raise ValueError(
            f"基线路径数 {P} ≠ evaluation.n_paths {cfg.n_paths}；"
            f"要换基线请同步改 config/compare.toml")
    if T != int(prob.T):
        raise ValueError(f"基线 T={T} ≠ 模型 T={prob.T}")
    if data["x_in"].shape != (P, T, prob.state_dim):
        raise ValueError(f"基线 x_in shape {data['x_in'].shape} ≠ "
                         f"{(P, T, prob.state_dim)}")
    if data["q"].shape != (P, T, prob.N_R):
        raise ValueError(f"基线 q shape {data['q'].shape}")
    if data["terms"].shape != (P, T, 5):
        raise ValueError(f"基线 terms shape {data['terms'].shape}")
    if data["totals"].shape != (P,):
        raise ValueError(f"基线 totals shape {data['totals'].shape}")
    n_scenes = len(prob.pool)
    if scene.min() < 0 or scene.max() >= n_scenes:
        raise ValueError(f"基线 scene 越界 [0,{n_scenes})")

    base_pd_path = Path(cfg.baseline_dir) / "problem_data.json"
    base_pd_hash = sha256_file(base_pd_path) if base_pd_path.is_file() else None
    model_pd_hash = sha256_file(cfg.problem_data) if Path(cfg.problem_data).is_file() else None
    match = None
    if base_pd_hash is not None and model_pd_hash is not None:
        match = (base_pd_hash == model_pd_hash)
        if not match and not cfg.baseline_allow_mismatch:
            raise RuntimeError(
                "基线 problem_data.json 与共享配置的 problem_data 不同："
                f"\n  baseline={base_pd_path} ({base_pd_hash[:12]})"
                f"\n  model={cfg.problem_data} ({model_pd_hash[:12]})"
                "\n模型参数变了但基线未重训，配对无效；"
                "确需比较请把 baseline.allow_model_mismatch 设为 true。")
    elif base_pd_hash is None:
        print(f"[警告] 基线目录没有 problem_data.json，无法校验场景池: {base_pd_path}")

    meta = {}
    meta_path = npz_path.with_suffix(".json")
    if meta_path.is_file():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return BaselineData(
        name=cfg.baseline_name,
        result_dir=Path(cfg.baseline_dir),
        npz_path=npz_path,
        scene=scene,
        totals=data["totals"],
        terms=data["terms"],
        cost=data["cost"],
        x_in=data["x_in"],
        q=data["q"],
        taup=data["taup"],
        taum=data["taum"],
        S_end=data["S_end"],
        y=data["y"],
        n_out=data["n_out"],
        F=data["F"],
        W_in=data["W_in"],
        W_out=data["W_out"],
        problem_data_hash=base_pd_hash,
        model_problem_data_hash=model_pd_hash,
        problem_data_match=match,
        npz_hash=sha256_file(npz_path),
        scene_hash=sha256_array(scene),
        meta=meta,
    )


def paired_stats(policy_totals, baseline_totals) -> dict:
    a = np.asarray(policy_totals, dtype=float)
    b = np.asarray(baseline_totals, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"配对数组形状不一致: {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else float("nan")
    t = mean / se if se > 0 else float("nan")
    ci = 1.959963984540054 * se
    return {
        "n": int(n),
        "policy_mean": float(np.mean(a)),
        "baseline_mean": float(np.mean(b)),
        "diff_mean": mean,
        "diff_std": std,
        "diff_se": se,
        "t": t,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "win_rate_policy_lower": float(np.mean(d < 0)),
        "rel_diff_pct": mean / float(np.mean(b)) * 100.0 if np.mean(b) else float("nan"),
    }


def term_stats(policy_terms, baseline_terms, names=("采购", "持有", "停机",
                                                    "主动调拨", "被动调拨")) -> list[dict]:
    """按路径汇总五项成本，再做配对差。"""
    a = np.asarray(policy_terms, dtype=float)
    b = np.asarray(baseline_terms, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 5:
        raise ValueError(f"terms shape 不匹配: {a.shape} vs {b.shape}")
    out = []
    for k, name in enumerate(names):
        at = a[:, :, k].sum(axis=1)
        bt = b[:, :, k].sum(axis=1)
        d = at - bt
        se = float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        out.append({
            "term": name,
            "policy_mean": float(np.mean(at)),
            "baseline_mean": float(np.mean(bt)),
            "diff_mean": float(np.mean(d)),
            "diff_se": se,
            "t": float(np.mean(d) / se) if se > 0 else float("nan"),
            "ci95_low": float(np.mean(d) - 1.959963984540054 * se),
            "ci95_high": float(np.mean(d) + 1.959963984540054 * se),
        })
    return out


