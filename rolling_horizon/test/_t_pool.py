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

from common.baseline import load_baseline  # noqa: E402
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from rolling_horizon.scenarios import (make_opt_pool, pool_slice,  # noqa: E402
                                       real_paths_from_baseline,
                                       sample_pool, transition_matrix)

sys.stdout.reconfigure(errors="replace")
PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def main():
    cfg = load_compare_config("config/compare_T60_base204105.toml")
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)

    # 1) 转移矩阵与主模型逐位一致
    n = 0
    for lam in (0.0, 0.5, 2.0, 5.0):
        a = transition_matrix(lam, prob.G, prob.p_digits)
        b = prob.P_ref if lam == prob.lambda0 else None
        if b is not None:
            check(np.allclose(a, b, atol=1e-14), "P_ref 不一致")
        check(a.shape == (prob.G, prob.G + 1), "P shape")
        check(np.allclose(a.sum(axis=1), 1.0, atol=1e-12), "P 行和")
        check(np.all(a >= 0), "P 负值")
        n += 1
    check(n == 4, "transition 数量")

    # 2) 本地 sample_pool 与主模型 sample_pool 同 RNG 同结果
    from code.problem_data import sample_pool as main_sample_pool
    a = sample_pool(prob.lambda0, prob.beta_cov, prob.z_bar, prob.cov,
                    prob.N_R, prob.G, prob.dirichlet_s, prob.p_digits,
                    3, 12345)
    b = main_sample_pool(prob.lambda0, prob.beta_cov, prob.z_bar, prob.cov,
                         prob.N_R, prob.G, prob.dirichlet_s, prob.p_digits,
                         3, 12345)
    for (za, ta), (zb, tb) in zip(a, b):
        check(np.allclose(za, zb, atol=0, rtol=0), "z 不一致")
        check(np.allclose(ta, tb, atol=0, rtol=0), "Theta 不一致")

    # 3) 优化池 shape/可复现/不同种子不同
    p1 = make_opt_pool(prob, 3, 2027)
    p2 = make_opt_pool(prob, 3, 2027)
    p3 = make_opt_pool(prob, 3, 2028)
    check(p1["pool"].shape == (3, prob.T, prob.N_R, prob.G, prob.G + 1),
          "pool shape")
    check(p1["pool_hash"] == p2["pool_hash"], "同种子 pool hash 不同")
    check(p1["pool_hash"] != p3["pool_hash"], "不同种子 pool hash 相同")
    check(np.allclose(p1["pool"], p2["pool"], atol=0, rtol=0),
          "同种子 pool 不同")

    # 4) 切片 CRN：t=2 的后半段 == t=3 的完整切片（同路径编号）
    s2 = pool_slice(p1["pool"], 2)
    s3 = pool_slice(p1["pool"], 3)
    check(s2.shape[1] == prob.T - 1, "slice 2 长度")
    check(s3.shape[1] == prob.T - 2, "slice 3 长度")
    check(np.allclose(s2[:, 1:, :, :, :], s3, atol=0, rtol=0),
          "CRN 切片不一致")
    check(np.allclose(p1["pool"][:, 1:, :, :, :], s2, atol=0, rtol=0),
          "t=2 切片")

    # 5) 真实路径子集
    rp = real_paths_from_baseline(base, 7)
    check(rp["scene"].shape == (7, prob.T), "real scene shape")
    check(np.allclose(rp["scene"], base.scene[:7]), "real scene 不是基线前 7 条")
    check(np.allclose(rp["totals"], base.totals[:7]), "real totals")
    check(np.allclose(rp["terms"], base.terms[:7]), "real terms")
    check(rp["M"] == 7, "real M")
    check(len(rp["scene_hash"]) == 64, "scene hash")
    check(abs(rp["baseline_full_mean"] - base.mean) < 1e-12,
          "baseline full mean")

    print(f"_t_pool: {PASS} checks passed")
    print(f"  pool shape={p1['pool'].shape} hash={p1['pool_hash'][:12]}…")
    print(f"  real subset hash={rp['scene_hash'][:12]}…")


if __name__ == "__main__":
    main()
