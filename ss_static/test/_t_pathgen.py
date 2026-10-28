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

from common.baseline import sha256_array  # noqa: E402
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ss_static.simulator import generate_scene_paths  # noqa: E402

sys.stdout.reconfigure(errors="replace")


def main():
    cfg = load_compare_config()
    prob = load_problem(cfg)
    ns, T = len(prob.pool), prob.T
    got = generate_scene_paths(7, 777, 3, ns, T)
    exp = []
    for blk, n in ((0, 3), (1, 3), (2, 1)):
        rng = np.random.default_rng(777 + blk)
        for _ in range(n):
            exp.append(rng.integers(0, ns, size=T))
    exp = np.asarray(exp)
    if not np.array_equal(got, exp):
        raise AssertionError("scene path 生成与 simulate.py 分块逻辑不一致")
    got2 = generate_scene_paths(7, 777, 3, ns, T)
    if not np.array_equal(got, got2):
        raise AssertionError("路径生成不可复现")
    if got.dtype != np.int16:
        raise AssertionError(f"dtype={got.dtype}")

    with np.load(cfg.baseline_npz) as z:
        scene = np.asarray(z["scene"])
    if scene.shape != (cfg.n_paths, T):
        raise AssertionError(f"基线 scene shape {scene.shape}")
    if scene.min() < 0 or scene.max() >= ns:
        raise AssertionError("基线 scene 越界")
    print("_t_pathgen: all checks passed")
    print(f"  generated {got.shape}, seed=777, chunk=3, hash={sha256_array(got)[:16]}")
    print(f"  baseline scene {scene.shape}, hash={sha256_array(scene)[:16]}")


if __name__ == "__main__":
    main()
