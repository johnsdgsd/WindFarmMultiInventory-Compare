from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.baseline import sha256_array  # noqa: E402
from common.config import load_compare_config  # noqa: E402
from common.model import (bootstrap, load_problem, load_solver,  # noqa: E402
                          verify_model)
from ss_static.simulator import generate_scene_paths  # noqa: E402

sys.stdout.reconfigure(errors="replace")


def main():
    cfg = load_compare_config()
    m = bootstrap(cfg)
    prob = load_problem(cfg)
    scfg = load_solver(cfg)
    if (prob.T, prob.G, prob.N_R, prob.L) != (60, 4, 5, 3):
        raise AssertionError(f"T60 维度不符: {prob.T},{prob.G},{prob.N_R},{prob.L}")
    if prob.state_dim != 40 or len(prob.pool) != 300:
        raise AssertionError(f"state_dim/pool: {prob.state_dim}/{len(prob.pool)}")

    pcfg = m.cc.load_problem_config(cfg.problem_toml)
    prob2 = m.pd.DefineProblem(pcfg)
    for name in ("T", "G", "N_R", "L"):
        if getattr(prob, name) != getattr(prob2, name):
            raise AssertionError(f"{name} 不一致")
    for name in ("N", "Qmax", "Imax", "n0", "I0", "o0", "tc_route"):
        if not np.allclose(getattr(prob, name), getattr(prob2, name)):
            raise AssertionError(f"{name} 不一致")
    for name in ("C", "h", "Cs", "atc", "tc", "lambda0", "dirichlet_s"):
        if not np.isclose(getattr(prob, name), getattr(prob2, name)):
            raise AssertionError(f"{name} 不一致")
    h1 = sha256_array(np.stack([th for _, th in prob.pool]))
    h2 = sha256_array(np.stack([th for _, th in prob2.pool]))
    if h1 != h2:
        raise AssertionError("problem_data.json 与 config/problem.toml 的场景池不一致")

    paths = generate_scene_paths(5, 123, 1000, len(prob.pool), prob.T)
    if paths.shape != (5, prob.T):
        raise AssertionError("路径 shape 异常")
    v = verify_model(cfg)
    if not v["model_code_hash"]:
        raise AssertionError("模型 code hash 为空")
    print("_t_common: all checks passed")
    print(f"  model_root={cfg.model_root}")
    print(f"  T={prob.T} G={prob.G} N_R={prob.N_R} L={prob.L} "
          f"dim={prob.state_dim} pool={len(prob.pool)}")
    print(f"  solver n_processes={scfg.n_processes} "
          f"n_forward_paths={scfg.n_forward_paths} seed={scfg.seed}")
    print(f"  model_commit={v['model_commit_actual']} "
          f"code_hash={v['model_code_hash'][:16]}")
    print(f"  problem_data_hash={sha256_array(np.stack([th for _, th in prob.pool]))[:16]}")


if __name__ == "__main__":
    main()
