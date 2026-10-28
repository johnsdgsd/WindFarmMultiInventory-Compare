from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np


def _paired_stats(policy_totals, baseline_totals) -> dict:
    """逐路径配对统计；与 common.baseline.paired_stats 同口径。"""
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
        "rel_diff_pct": mean / float(np.mean(b)) * 100.0
        if np.mean(b) else float("nan"),
    }


def _fill_rate(total_rep: np.ndarray, F: np.ndarray) -> np.ndarray:
    """逐路径替换率，统一截断到 [0,1]；F=0 时记为 1。"""
    F = np.asarray(F, dtype=float)
    total_rep = np.asarray(total_rep, dtype=float)
    return np.where(F > 1e-12,
                    np.minimum(total_rep / np.maximum(F, 1e-12), 1.0),
                    1.0)


def _behavior_from_arrays(N_R: int, arrays: dict, q0: np.ndarray,
                          P: int) -> dict:
    a = arrays
    q_out = np.asarray(a["q"], dtype=float)
    total_placed = q0.sum(axis=1) + q_out.sum(axis=(1, 2))
    total_charged = q0.sum(axis=1) + q_out[:, :-1, :].sum(axis=(1, 2))
    y = np.asarray(a["y"], dtype=float)
    diag = np.einsum("ptii->pt", y)
    total_rep = y.sum(axis=(1, 2, 3))
    local_rep = diag.sum(axis=1)
    cross_rep = total_rep - local_rep
    F = np.asarray(a["F"], dtype=float).sum(axis=(1, 2))
    W_in = np.asarray(a["W_in"], dtype=float)
    W_out = np.asarray(a["W_out"], dtype=float)
    x_in = np.asarray(a["x_in"], dtype=float)
    taup = np.asarray(a.get("taup", np.zeros_like(q_out)), dtype=float)
    taum = np.asarray(a.get("taum", np.zeros_like(q_out)), dtype=float)
    active = (taup + taum).sum(axis=(1, 2)) / 2.0

    return {
        "total_failures": float(np.mean(F)),
        "total_replacements": float(np.mean(total_rep)),
        "fill_rate": float(np.mean(_fill_rate(total_rep, F))),
        "final_backlog": float(np.mean(W_out[:, -1, :].sum(axis=1))),
        "mean_backlog": float(np.mean(W_in.sum(axis=2))),
        # ★ 五仓合计，和持有成本口径一致（不是每仓平均）
        "mean_inventory": float(np.mean(x_in[:, :, :N_R].sum(axis=2))),
        "total_orders_placed": float(np.mean(total_placed)),
        "total_orders_charged": float(np.mean(total_charged)),
        "q0_total": float(q0.sum() / max(P, 1)),
        "order_frequency": float(np.mean(q_out > 1e-9)),
        "local_replacements": float(np.mean(local_rep)),
        "cross_replacements": float(np.mean(cross_rep)),
        "active_transfer_volume": float(np.mean(active)),
    }


def behavior_stats(prob, res) -> dict:
    """策略行为统计；要求 res.arrays 存在（record=True）。"""
    if not getattr(res, "arrays", None):
        raise ValueError("behavior_stats 需要 record=True 的仿真结果")
    q0 = np.asarray(res.arrays["x_in"][:, 0, prob.layout.q_slice()],
                    dtype=float)
    return _behavior_from_arrays(prob.N_R, res.arrays, q0, res.P)


def baseline_behavior(prob, base) -> dict:
    """基线行为统计；mean_inventory 同样为五仓合计。"""
    arrays = {
        "x_in": base.x_in,
        "q": base.q,
        "taup": base.taup,
        "taum": base.taum,
        "y": base.y,
        "F": base.F,
        "W_in": base.W_in,
        "W_out": base.W_out,
    }
    return _behavior_from_arrays(prob.N_R, arrays,
                                 base.q0(prob.layout), base.P)


def time_profile(policy_terms: np.ndarray,
                 baseline_terms: np.ndarray) -> dict:
    """逐期/十分位成本差与累计差。

    policy_terms / baseline_terms: (P,T,5)，通常是仿真记录里的 terms。
    """
    a = np.asarray(policy_terms, dtype=float).sum(axis=2)
    b = np.asarray(baseline_terms, dtype=float).sum(axis=2)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"terms shape 不匹配: {a.shape} vs {b.shape}")
    T = a.shape[1]
    p = a.mean(axis=0)
    q = b.mean(axis=0)
    d = p - q
    cum = np.cumsum(d)
    bins = np.array_split(np.arange(T), 10)
    deciles = []
    for k, ix in enumerate(bins):
        if ix.size == 0:
            continue
        deciles.append({
            "decile": k + 1,
            "t_first": int(ix[0]) + 1,
            "t_last": int(ix[-1]) + 1,
            "n_periods": int(ix.size),
            "policy_cost": float(p[ix].sum()),
            "baseline_cost": float(q[ix].sum()),
            "diff": float(d[ix].sum()),
            "policy_mean": float(p[ix].mean()),
            "baseline_mean": float(q[ix].mean()),
        })
    k = max(1, T // 10)
    return {
        "period": list(range(1, T + 1)),
        "policy": p.tolist(),
        "baseline": q.tolist(),
        "diff": d.tolist(),
        "cum_diff": cum.tolist(),
        "deciles": deciles,
        "first_10pct_diff": float(d[:k].sum()),
        "last_10pct_diff": float(d[-k:].sum()),
        "middle_diff": float(d[k:-k].sum()) if T > 2 * k else 0.0,
        "total_diff": float(d.sum()),
        "worst_decile": max(deciles, key=lambda z: z["diff"]) if deciles else None,
        "best_decile": min(deciles, key=lambda z: z["diff"]) if deciles else None,
    }


def _terminal_components(prob, res, base):
    a = res.arrays
    I_p = (np.asarray(a["S_end"])[:, -1, :]
           + np.asarray(a["taup"])[:, -1, :]
           - np.asarray(a["taum"])[:, -1, :]).sum(axis=1)
    W_p = np.asarray(a["W_out"])[:, -1, :].sum(axis=1)
    I_b = (np.asarray(base.S_end)[:, -1, :]
           + np.asarray(base.taup)[:, -1, :]
           - np.asarray(base.taum)[:, -1, :]).sum(axis=1)
    W_b = np.asarray(base.W_out)[:, -1, :].sum(axis=1)
    return I_p, W_p, I_b, W_b


def terminal_sensitivity(prob, res, base) -> list[dict]:
    """期末库存回收/期末待修罚值的配对敏感性。

    \tilde J = J - c_salv * Σ_r I_out[r,T] + c_term * Σ_r W_out[r,T]
    仅作为敏感性分析；主目标仍是无期末值的 J。
    """
    if not getattr(res, "arrays", None):
        raise ValueError("terminal_sensitivity 需要 record=True 的仿真结果")
    I_p, W_p, I_b, W_b = _terminal_components(prob, res, base)
    C = float(prob.C)
    Cs = float(prob.Cs)
    grid = [
        (0.0, 0.0, "无期末值"),
        (0.0, Cs, "库存回收=0，待修罚=Cs"),
        (0.0, C, "库存回收=0，待修罚=C"),
        (C, 0.0, "库存回收=C，待修罚=0"),
        (C, Cs, "库存回收=C，待修罚=Cs"),
        (C, C, "库存回收=C，待修罚=C"),
    ]
    out = []
    for salv, term, label in grid:
        p = (res.path_costs - salv * I_p + term * W_p)
        b = (base.totals - salv * I_b + term * W_b)
        ps = _paired_stats(p, b)
        out.append({
            "label": label,
            "salvage": float(salv),
            "backlog_penalty": float(term),
            "policy_mean": float(np.mean(p)),
            "baseline_mean": float(np.mean(b)),
            "diff_mean": ps["diff_mean"],
            "ci95_low": ps["ci95_low"],
            "ci95_high": ps["ci95_high"],
            "t": ps["t"],
            "rel_diff_pct": ps["rel_diff_pct"],
            "win_rate_policy_lower": ps["win_rate_policy_lower"],
        })
    return out
