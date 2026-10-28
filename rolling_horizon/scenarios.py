"""滚动时域基准的场景源。

优化池（定义 2）：固定 seed_opt，一次性生成 N×T 个独立 Theta 矩阵；
t 期使用全部 N 条路径的 [t,T] 切片（CRN），不重新抽样。

真实路径（定义 7）：本实现直接取 SDDP 基线 npz 的 scene 前 M 条，保证与
SDDP 在同一组真实路径上配对；该 300 场景池与优化池种子独立，不使用未来信息。

这里复制了主模型 `code/problem_data.py` 的转移矩阵/Dirichlet 抽样数学，
只读、不修改主项目。
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


# ---------------------------------------------------------------- 转移矩阵
def poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def transition_matrix(lam: float, G: int, p_digits: int | None = 3) -> np.ndarray:
    """G×(G+1) 上三角转移矩阵，与主模型完全同口径。"""
    lam = float(lam)
    if lam < 0:
        raise ValueError(f"lambda 必须非负，实际 {lam}")
    p = np.array([poisson_pmf(lam, k) for k in range(G + 1)], dtype=float)
    P = np.zeros((G, G + 1), dtype=float)
    for i in range(G):
        off = []
        for j in range(i + 1, G + 1):
            if j < G:
                off.append(p[j - i])
            else:
                off.append(1.0 - float(p[: G - i].sum()))
        off = np.asarray(off, dtype=float)
        if p_digits is not None:
            off = np.round(off, int(p_digits))
            s = float(off.sum())
            if s > 1.0 + 1e-12:
                off = off / s
        P[i, i + 1:] = off
        P[i, i] = 1.0 - float(off.sum())
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    if np.any(rs <= 0):
        raise ValueError("转移矩阵出现空行")
    return P / rs


def _cov_factor(beta_cov, z_bar, z_r) -> float:
    return float(np.exp(np.dot(beta_cov, np.asarray(z_r) - np.asarray(z_bar))))


# ---------------------------------------------------------------- 场景池
def sample_pool(lambda0: float, beta_cov, z_bar, cov, N_R: int, G: int,
                dirichlet_s: float, p_digits: int, pool_size: int,
                seed: int) -> list:
    """(z, Theta) 场景池；与 `code/problem_data.py::sample_pool` 相同。

    每期/每仓/每行独立；Theta 为 (N_R,G,G+1)，只在上三角正支撑抽样。
    """
    rng = np.random.default_rng(int(seed))
    pool = []
    for _ in range(int(pool_size)):
        z = np.empty((3, N_R))
        z[0] = rng.normal(cov["mu_T"], cov["sigma_T"], size=N_R)
        z[1] = rng.poisson(cov["lambda"], size=N_R)
        z[2] = 100.0 * rng.beta(cov["alpha_L"], cov["beta_L"], size=N_R)
        theta = np.zeros((N_R, G, G + 1))
        for r in range(N_R):
            f = _cov_factor(beta_cov, z_bar, z[:, r])
            P = transition_matrix(lambda0 * f, G, p_digits)
            for i in range(G):
                cols = [j for j in range(i, G + 1) if P[i, j] > 0.0]
                if not cols:
                    raise ValueError(f"P 第 {i} 行无正支撑")
                alpha = np.array([dirichlet_s * P[i, j] for j in cols])
                if len(cols) == 1:
                    vals = np.array([1.0])
                else:
                    vals = rng.dirichlet(alpha)
                theta[r, i, cols] = vals
        row_sum = theta.sum(axis=2)
        if np.max(np.abs(row_sum - 1.0)) > 1e-12:
            raise ValueError("Theta 行和偏离 1")
        if np.any(theta[:, :, :] < 0):
            raise ValueError("Theta 出现负值")
        pool.append((z, theta))
    return pool


def make_opt_pool(prob, n_scenarios: int, seed_opt: int) -> dict:
    """生成并 reshape 固定优化池 P，shape=(N,T,N_R,G,G+1)。

    池中 N×T 个矩阵两两独立；路径编号 omega 固定；t 期使用 pool[:,t-1:]。
    """
    N = int(n_scenarios)
    if N < 1:
        raise ValueError("n_scenarios 必须为正")
    T = int(prob.T)
    scenes = sample_pool(
        prob.lambda0, prob.beta_cov, prob.z_bar, prob.cov,
        prob.N_R, prob.G, prob.dirichlet_s, prob.p_digits,
        N * T, int(seed_opt))
    theta = np.stack([th for _, th in scenes], axis=0)
    pool = theta.reshape(N, T, prob.N_R, prob.G, prob.G + 1)
    return {
        "pool": pool,
        "n_scenarios": N,
        "T": T,
        "seed_opt": int(seed_opt),
        "pool_hash": sha256_array(pool),
        "shape": tuple(int(v) for v in pool.shape),
        "z": [z for z, _ in scenes],
    }


def pool_slice(pool: np.ndarray, t: int) -> np.ndarray:
    """t 期窗口切片，返回 (N, T-t+1, N_R, G, G+1)。"""
    t = int(t)
    if t < 1 or t > pool.shape[1]:
        raise ValueError(f"t={t} 越界 [1,{pool.shape[1]}]")
    return pool[:, t - 1:, :, :, :]


def real_paths_from_baseline(base, m_paths: int) -> dict:
    """取基线 npz 的前 M 条真实路径，用于配对评估。

    同时返回对应的基线 totals/terms 子集；scene 不重新生成。
    """
    M = int(m_paths)
    if M < 1:
        raise ValueError("real_paths 必须为正")
    if M > base.P:
        raise ValueError(f"real_paths={M} > 基线路径数 {base.P}")
    scene = np.ascontiguousarray(base.scene[:M])
    return {
        "scene": scene,
        "totals": np.asarray(base.totals[:M], dtype=float).copy(),
        "terms": np.asarray(base.terms[:M], dtype=float).copy(),
        "M": M,
        "indices": np.arange(M, dtype=int),
        "scene_hash": sha256_array(scene),
        "baseline_full_mean": float(base.mean),
        "baseline_full_P": int(base.P),
    }


def sha256_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
