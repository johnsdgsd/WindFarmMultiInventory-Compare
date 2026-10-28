from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from statistics import NormalDist

import numpy as np

from ss_static.optimize import InitialPoint
from ss_static.simulator import Simulator, round_repair_vec

START_NAMES = ("mean", "one_sigma", "two_z")


def infinite_demand(prob, paths: np.ndarray) -> np.ndarray:
    """docs/s,S.md §8：库存无限、需求即时满足的基线，估计 d_r。"""
    sim = Simulator(prob)
    P, T = np.asarray(paths).shape
    n = np.tile(np.asarray(prob.n0, float)[None, :, :], (P, 1, 1))
    sumF = np.zeros((P, prob.N_R), dtype=float)
    for t in range(1, T + 1):
        theta = sim.theta_pool[paths[:, t - 1]]
        Nc = n[:, :, :, None] * theta
        F = Nc[:, :, :, sim.G].sum(axis=2)
        sumF += F
        n_raw = np.zeros((P, prob.N_R, sim.G), dtype=float)
        n_raw[:, :, 0] = F + Nc[:, :, 0, 0]
        for j in range(1, sim.G):
            n_raw[:, :, j] = Nc[:, :, :, j].sum(axis=2)
        n = round_repair_vec(n_raw, sim.N, sim.digits, sim.repair_tol)
    return sumF.mean(axis=0) / T


def moments(prob, d: np.ndarray) -> dict:
    """MD §8 的 mu/sigma/beta/z。"""
    L = int(prob.L)
    d = np.asarray(d, dtype=float)
    mu = L * d
    sigma = np.sqrt(np.maximum(L * d, 0.0))
    beta = float(prob.Cs) / (float(prob.Cs) + float(prob.h))
    beta = float(np.clip(beta, 1e-9, 1.0 - 1e-9))
    z = NormalDist().inv_cdf(beta)
    return {"L": L, "d": d, "mu": mu, "sigma": sigma,
            "beta": beta, "z": float(z)}


def analytic_starts(prob, d: np.ndarray, opts: dict, *,
                    active: bool = False,
                    active_target: float = 5.0,
                    active_threshold_init: float = 5.0) -> list[InitialPoint]:
    """docs/s,S.md §8：报童分位数初值、多起点与搜索半径。

    active=True 时，θ 末尾追加每仓主动调拨门限 h 的初值；
    h 本身是优化参数，初值默认等于 active_target。
    """
    Imax = np.asarray(prob.Imax, dtype=float)
    m = moments(prob, d)
    mu, sigma, z = m["mu"], m["sigma"], m["z"]
    integer = bool(opts.get("integer_params", False))
    q_r = float(opts.get("q_r", 1.0))
    names = [str(v) for v in opts.get("starts", list(START_NAMES))]
    unknown = set(names) - set(START_NAMES)
    if unknown:
        raise ValueError(f"未知 starts: {sorted(unknown)}；只支持 {START_NAMES}")

    base = [
        ("mean", mu + z * sigma),
        ("one_sigma", mu + 1.0 * sigma),
        ("two_z", mu + 2.0 * z * sigma),
    ]
    out: list[InitialPoint] = []
    seen = set()
    for name, S0 in base:
        if name not in names:
            continue
        S0c = np.clip(np.asarray(S0, float), 0.0, Imax)
        if integer:
            variants = [(f"{name}_floor", np.floor(S0c)),
                        (f"{name}_ceil", np.ceil(S0c))]
        else:
            variants = [(name, S0c)]
        for vname, S0v in variants:
            S0v = np.clip(np.asarray(S0v, float), 0.0, Imax)
            q = max(1.0, float(round(q_r))) if integer else q_r
            s0 = np.maximum(0.0, S0v - q)
            s0 = np.minimum(s0, S0v)
            rho_s = np.maximum(0.0, np.minimum(S0v, 3.0 * sigma))
            rho_S = np.maximum(0.0, np.minimum(Imax - s0, 5.0 * sigma))
            if integer:
                rho_s = np.maximum(1.0, rho_s)
                rho_S = np.maximum(1.0, rho_S)
            key = (tuple(np.round(s0, 9)), tuple(np.round(S0v, 9)))
            if key in seen:
                continue
            seen.add(key)
            theta = np.concatenate([s0, S0v])
            radius = np.concatenate([rho_s, rho_S])
            meta = {"mu": mu.tolist(), "sigma": sigma.tolist(),
                    "z": z, "beta": m["beta"]}
            if active:
                h0 = np.full(int(prob.N_R), float(active_threshold_init))
                h0 = np.clip(h0, float(active_target), Imax)
                theta = np.concatenate([theta, h0])
                radius = np.concatenate(
                    [radius, np.maximum(0.0, Imax - float(active_target))])
                meta["active_target"] = float(active_target)
                meta["active_threshold_init"] = float(active_threshold_init)
            out.append(InitialPoint(
                name=vname, theta=theta, radius=radius, meta=meta))
    if not out:
        raise RuntimeError("没有生成任何解析初值")
    return out
