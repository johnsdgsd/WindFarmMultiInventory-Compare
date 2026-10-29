"""PPO tau-first 的样本外评估、配对统计、审计与归档。

评估口径：
- 真实路径：基线 npz 的 scene 前 M 条（默认 M=50，与 rolling 正式评估一致）；
- Theta：直接取 prob.pool[scene]，不重新采样；
- 初始状态：problem_data 的 I0/o0/n0；没有 q0 状态；
- 策略：state-only 确定性动作（Gaussian mean），动作经 env 投影；
- 成本：五项成本直接累加，单位万元；奖励只用于 PPO 学习（常数缩放）。

用法::

    P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
    "$P" -B ppo_taufirst/evaluate.py \
        --policy results/ppo_taufirst/<run>/policy.pt --m-paths 50 \
        --tag ppo_taufirst_T60_eval
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

from common.baseline import (load_baseline, paired_stats, sha256_array,  # noqa: E402
                             sha256_file, term_stats)
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from common.report import (copy_configs, fmt, provenance,  # noqa: E402
                           save_sim_npz, write_json, write_text)
from ppo_taufirst.archive import append_index, make_run_dir  # noqa: E402
from ppo_taufirst.env import (TERM_NAMES, TauFirstEnv,  # noqa: E402
                              audit_rollout)
from ppo_taufirst.train import (DEFAULT_PPO_CONFIG, load_ppo_config,  # noqa: E402
                                resolve_compare_config)


# ---------------------------------------------------------------- 行为统计
def _fill_rate(total_rep, F):
    F = np.asarray(F, dtype=float)
    total_rep = np.asarray(total_rep, dtype=float)
    return np.where(F > 1e-12,
                    np.minimum(total_rep / np.maximum(F, 1e-12), 1.0), 1.0)


def behavior_from_ppo(prob, arrays: dict) -> dict:
    q = np.asarray(arrays["q"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    F = np.asarray(arrays["F"], dtype=float)
    W_in = np.asarray(arrays["W_in"], dtype=float)
    W_out = np.asarray(arrays["W_out"], dtype=float)
    I_in = np.asarray(arrays["I_in"], dtype=float)
    I_out = np.asarray(arrays["I_out"], dtype=float)
    tau = np.asarray(arrays["tau"], dtype=float)
    total_rep = y.sum(axis=(1, 2, 3))
    F_sum = F.sum(axis=(1, 2))
    diag = np.einsum("ptii->pt", y)
    local = diag.sum(axis=1)
    return {
        "mean_inventory": float(I_in.sum(axis=2).mean()),
        "mean_backlog": float(W_in.sum(axis=2).mean()),
        "final_backlog": float(W_out[:, -1, :].sum(axis=1).mean()),
        "final_inventory": float(I_out[:, -1, :].sum(axis=1).mean()),
        "total_failures": float(F_sum.mean()),
        "total_replacements": float(total_rep.mean()),
        "fill_rate": float(_fill_rate(total_rep, F_sum).mean()),
        "total_orders_placed": float(q.sum(axis=(1, 2)).mean()),
        "total_orders_charged": float(q[:, :-1, :].sum(axis=(1, 2)).mean()),
        "q0_total": 0.0,
        "order_frequency": float((q > 1e-9).mean()),
        "local_replacements": float(local.mean()),
        "cross_replacements": float(total_rep.mean() - local.mean()),
        "active_transfer_volume": float(
            0.5 * np.abs(tau).sum(axis=(1, 2)).mean()),
        "passive_transfer_volume": float(total_rep.mean()),
        "mean_q": float(q.sum(axis=(1, 2)).mean() / max(q.shape[1], 1)),
    }


def behavior_from_base(prob, base, m: int) -> dict:
    M = int(m)
    layout = prob.layout
    I_in = np.asarray(
        base.x_in[:M, :, layout.off_I:layout.off_I + layout.n_I],
        dtype=float)
    q = np.asarray(base.q[:M], dtype=float)
    y = np.asarray(base.y[:M], dtype=float)
    F = np.asarray(base.F[:M], dtype=float)
    W_in = np.asarray(base.W_in[:M], dtype=float)
    W_out = np.asarray(base.W_out[:M], dtype=float)
    taup = np.asarray(base.taup[:M], dtype=float)
    taum = np.asarray(base.taum[:M], dtype=float)
    q0 = np.asarray(base.q0(layout)[:M], dtype=float)
    total_rep = y.sum(axis=(1, 2, 3))
    F_sum = F.sum(axis=(1, 2))
    diag = np.einsum("ptii->pt", y)
    local = diag.sum(axis=1)
    I_out_T = (np.asarray(base.S_end[:M, -1], dtype=float)
               + (taup - taum)[:, -1, :]).sum(axis=1)
    return {
        "mean_inventory": float(I_in.sum(axis=2).mean()),
        "mean_backlog": float(W_in.sum(axis=2).mean()),
        "final_backlog": float(W_out[:, -1, :].sum(axis=1).mean()),
        "final_inventory": float(I_out_T.mean()),
        "total_failures": float(F_sum.mean()),
        "total_replacements": float(total_rep.mean()),
        "fill_rate": float(_fill_rate(total_rep, F_sum).mean()),
        "total_orders_placed": float(q0.sum(axis=1).mean()
                                     + q.sum(axis=(1, 2)).mean()),
        "total_orders_charged": float(q0.sum(axis=1).mean()
                                      + q[:, :-1, :].sum(axis=(1, 2)).mean()),
        "q0_total": float(q0.sum(axis=1).mean()),
        "order_frequency": float((q > 1e-9).mean()),
        "local_replacements": float(local.mean()),
        "cross_replacements": float(total_rep.mean() - local.mean()),
        "active_transfer_volume": float(
            0.5 * (taup + taum).sum(axis=(1, 2)).mean()),
        "passive_transfer_volume": float(total_rep.mean()),
        "mean_q": float((q0.sum(axis=1).mean()
                         + q.sum(axis=(1, 2)).mean()) / max(q.shape[1], 1)),
    }


def time_profile(policy_terms, baseline_terms) -> dict:
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
        "middle_diff": float(d[k:-k].sum()) if T > 2 * k else 0.0,
        "last_10pct_diff": float(d[-k:].sum()),
        "total_diff": float(d.sum()),
        "worst_decile": max(deciles, key=lambda z: z["diff"])
        if deciles else None,
        "best_decile": min(deciles, key=lambda z: z["diff"])
        if deciles else None,
    }


def terminal_sensitivity(prob, arrays: dict, base, m: int,
                         totals: np.ndarray) -> list[dict]:
    M = int(m)
    I_p = np.asarray(arrays["I_out"][:, -1], dtype=float).sum(axis=1)
    W_p = np.asarray(arrays["W_out"][:, -1], dtype=float).sum(axis=1)
    taup = np.asarray(base.taup[:M], dtype=float)
    taum = np.asarray(base.taum[:M], dtype=float)
    I_b = (np.asarray(base.S_end[:M, -1], dtype=float)
           + (taup - taum)[:, -1, :]).sum(axis=1)
    W_b = np.asarray(base.W_out[:M, -1], dtype=float).sum(axis=1)
    b0 = np.asarray(base.totals[:M], dtype=float)
    C, Cs = float(prob.C), float(prob.Cs)
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
        p = totals - salv * I_p + term * W_p
        b = b0 - salv * I_b + term * W_b
        ps = paired_stats(p, b)
        out.append({
            "label": label, "salvage": float(salv),
            "backlog_penalty": float(term),
            "policy_mean": ps["policy_mean"],
            "baseline_mean": ps["baseline_mean"],
            "diff_mean": ps["diff_mean"],
            "ci95_low": ps["ci95_low"], "ci95_high": ps["ci95_high"],
            "t": ps["t"], "rel_diff_pct": ps["rel_diff_pct"],
            "win_rate_policy_lower": ps["win_rate_policy_lower"],
        })
    return out


# ---------------------------------------------------------------- 滚动对照
def load_rolling(path: str | Path, m: int, base_scene: np.ndarray) -> dict | None:
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("sim_*paths.npz"))
        if not cands:
            return {"error": f"目录内没有 sim_*paths.npz: {p}"}
        npz = cands[-1]
    else:
        npz = p
    if not npz.is_file():
        return {"error": f"rolling npz 不存在: {npz}"}
    with np.load(npz, allow_pickle=False) as z:
        need = {"totals", "terms", "scene"}
        if not need.issubset(set(z.files)):
            return {"error": f"rolling npz 缺字段: {sorted(need - set(z.files))}"}
        totals = np.asarray(z["totals"], dtype=float)
        terms = np.asarray(z["terms"], dtype=float)
        scene = np.asarray(z["scene"])
    out = {
        "npz": str(npz),
        "M": int(totals.size),
        "mean": float(totals.mean()),
        "terms_mean": np.asarray(terms, dtype=float).sum(axis=1).mean(axis=0)
        .tolist(),
        "scene_hash": sha256_array(scene),
        "scene_match": (scene.shape == base_scene.shape
                        and np.array_equal(scene, base_scene)),
    }
    if totals.size != int(m) or not out["scene_match"]:
        out["paired_with_ppo"] = None
        out["note"] = ("rolling 路径数或 scene 与 PPO 评估子集不一致，"
                       "只报告均值，不做配对。")
        return out
    out["totals"] = totals
    out["terms"] = terms
    return out


# ---------------------------------------------------------------- 报告
def _pct(x):
    try:
        return f"{float(x):+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _table(headers, rows, aligns=None) -> str:
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def _summary_md(ctx: dict) -> str:
    ps = ctx["paired"]
    meta = ctx["meta"]
    terms = ctx["terms"]
    beh = ctx["behavior"]
    tp = ctx["time"]
    audit = ctx["audit"]
    bad = ctx["bad"]
    proj = ctx["projection"]
    lines = [
        "# PPO tau-first — 样本外评估与 SDDP 配对",
        "",
        f"- 结果目录：`{ctx['run_dir']}`",
        f"- 创建时间：{ctx['created_at']}",
        f"- 共享配置：`{ctx['compare_config']}`；PPO 配置：`{ctx['ppo_config']}`",
        f"- 模型：T={ctx['T']}，N_R={ctx['N_R']}，G={ctx['G']}，L={ctx['L']}",
        f"- 策略：`{ctx['policy_src']}`（训练 {meta.get('train_steps')} steps，"
        f"seed={meta.get('train_seed')}，墙钟 {fmt(meta.get('train_wall_s'), 1)} s）",
        (f"- 训练收敛曲线：`{ctx['train_curve']}`"
         if ctx.get("train_curve") else "- 训练收敛曲线：未找到/未生成"),
        "",
        "## 1. 协议",
        "",
        "- 新时序：`tau_t -> q_t -> Theta_t -> y_t`；策略在 tau/q 时只看到状态，"
        "不看本期 Theta_t。",
        "- 状态：`s_t=(I,n,o1,o2)`，观测 36 维："
        "`I/Imax, n/N, o1/Qmax, o2/Qmax, t/(T-1)`；"
        "没有 q0 状态。",
        "- 动作：10 维 Gaussian；前 5 维 ×Imax 得 tau 原始量，后 5 维 ×Qmax 得 q "
        "原始量；env 先投影 tau（箱约束 + Σtau=0），再投影 q。",
        "- 被动调拨：先本地满足，再按 `prob.priority` / `tc_route` 的优先级"
        "（例如 1:[1,2,3,4,5]，2:[2,1,3,4,5]）做 `(tc,dest,source)` 全局贪心；"
        "可用库存 = `I+tau+o1`；需求 = `W_in+F_new`。",
        f"- PPO 学习奖励 = -reward_scale × 阶段成本；"
        f"reward_scale={ctx['reward_scale']:g}；报告成本不缩放。",
        "- 训练 Theta：在线 Dirichlet/Lambda 抽样（每步独立），不建 N×T 池。",
        f"- 评估 Theta：基线 npz 的 scene 前 M={ctx['M']} 条，"
        f"scene_hash=`{ctx['scene_hash']}`；基线全量均值 {fmt(ctx['base_full_mean'])}，"
        f"子集均值 {fmt(ctx['base_subset_mean'])}。",
        f"- 末期订货：{'q_T=0（与 rolling/SDDP 口径一致）' if ctx['terminal_q_zero'] else '允许 q_T>0'}。",
        "- 信息集：state-only 策略；SDDP 的价值函数/t 期随机情景与 rolling 的"
        "两阶段 SAA 使用更丰富的信息，**不能把本结果当作同等信息集下的比较**。",
        "",
        "## 2. 主结果（同一 M 条真实路径配对）",
        "",
        _table(
            ["指标", "值"],
            [
                ["PPO 均值（万元/路径）", fmt(ps["policy_mean"])],
                ["基线子集均值", fmt(ps["baseline_mean"])],
                ["配对差（PPO−基线）", fmt(ps["diff_mean"])],
                ["95% CI", f"[{fmt(ps['ci95_low'])}, {fmt(ps['ci95_high'])}]"],
                ["t", fmt(ps["t"], 3)],
                ["胜率（PPO 更低）", f"{100 * ps['win_rate_policy_lower']:.2f}%"],
                ["相对差", _pct(ps["rel_diff_pct"])],
                ["M", ps["n"]],
            ],
            ["---", "---:"],
        ),
        "",
        "## 3. 五项成本",
        "",
        _table(
            ["成本项", "PPO", "基线子集", "配对差", "t", "95% CI"],
            [
                [t["term"], fmt(t["policy_mean"]), fmt(t["baseline_mean"]),
                 fmt(t["diff_mean"]), fmt(t["t"], 2),
                 f"[{fmt(t['ci95_low'])}, {fmt(t['ci95_high'])}]"]
                for t in terms
            ],
        ),
        "",
        "## 4. 行为",
        "",
        _table(
            ["指标", "PPO", "基线子集"],
            [[k, fmt(beh["ppo"][k]), fmt(beh["baseline"][k])]
             for k in ("mean_inventory", "mean_backlog", "final_backlog",
                       "final_inventory", "total_failures",
                       "total_replacements", "fill_rate",
                       "total_orders_placed", "total_orders_charged",
                       "order_frequency", "local_replacements",
                       "cross_replacements", "active_transfer_volume",
                       "passive_transfer_volume")],
        ),
        "",
        "## 5. 时间剖面",
        "",
        f"- 首 10% 期差：{fmt(tp['first_10pct_diff'])}；"
        f"中间段：{fmt(tp['middle_diff'])}；末 10% 期差："
        f"{fmt(tp['last_10pct_diff'])}；合计：{fmt(tp['total_diff'])}。",
        f"- 最差十分位：第 {tp['worst_decile']['decile']} 段"
        f"（t={tp['worst_decile']['t_first']}..{tp['worst_decile']['t_last']}，"
        f"差 {fmt(tp['worst_decile']['diff'])}）。" if tp["worst_decile"] else "",
        "",
        _table(
            ["十分位", "t 范围", "PPO", "基线", "差"],
            [[d["decile"], f"{d['t_first']}..{d['t_last']}",
              fmt(d["policy_cost"]), fmt(d["baseline_cost"]),
              fmt(d["diff"])] for d in tp["deciles"]],
        ),
        "",
        "## 6. 审计",
        "",
        f"- 结论：{'通过' if ctx['audit_passed'] else '未通过'}；"
        f"违约项 {len(bad)} 个。",
        f"- `terms` 与 `cost` 最大偏差：{ctx['terms_dev']:.3e}；"
        f"`reward + reward_scale*cost` 最大偏差："
        f"{audit['reward_identity']:.3e}。",
        "",
        _table(
            ["核对项", "最大偏差"],
            [[k, f"{v:.3e}"] for k, v in audit.items()],
        ),
        "",
        "## 7. 动作投影诊断",
        "",
        _table(
            ["指标", "均值", "最大"],
            [
                ["‖tau−tau_raw‖₁", fmt(proj["tau_l1_mean"], 3),
                 fmt(proj["tau_l1_max"], 3)],
                ["‖tau−tau_raw‖∞", fmt(proj["tau_max_mean"], 3),
                 fmt(proj["tau_max_max"], 3)],
                ["‖q−q_raw‖₁", fmt(proj["q_l1_mean"], 3),
                 fmt(proj["q_l1_max"], 3)],
                ["‖q−q_raw‖∞", fmt(proj["q_max_mean"], 3),
                 fmt(proj["q_max_max"], 3)],
                ["q_upper 均值", fmt(proj["q_upper_mean"], 3),
                 fmt(proj["q_upper_max"], 3)],
                ["q_upper 最小", fmt(proj["q_upper_min"], 3), "—"],
            ],
        ),
        f"- 末期 q_T 被强制为 0 的步数比例：{100 * proj['terminal_zero_frac']:.2f}%。",
        "",
    ]
    lines += ["## 8. 滚动时域对照", ""]
    if ctx.get("rolling"):
        r = ctx["rolling"]
        if r.get("error"):
            lines.append(f"- {r['error']}")
        else:
            lines.append(f"- rolling npz：`{r['npz']}`；M={r['M']}；"
                         f"均值 {fmt(r['mean'])}；scene 一致：{r['scene_match']}。")
            if r.get("paired_with_ppo"):
                rp = r["paired_with_ppo"]
                lines += [
                    f"- PPO − rolling：{fmt(rp['diff_mean'])}"
                    f"（相对 {_pct(rp['rel_diff_pct'])}，t={fmt(rp['t'], 2)}，"
                    f"胜率 {100 * rp['win_rate_policy_lower']:.2f}%）。",
                    f"- rolling − 基线子集：{fmt(r['paired_vs_base']['diff_mean'])}"
                    f"（相对 {_pct(r['paired_vs_base']['rel_diff_pct'])}）。",
                ]
            else:
                lines.append(f"- {r.get('note', '未做配对')}")
    else:
        lines.append("- 未提供 `--rolling-dir`，未做滚动对照。")
    lines.append("")
    lines += [
        "## 9. 结论与限制",
        "",
        f"- 本评估是 M={ctx['M']} 条基线路径上的配对结果；CI/t 只对当前子集成立，"
        "不能外推为全量 3000 条路径的结论。",
        "- PPO 是 state-only 策略；SDDP 使用价值函数与阶段随机信息，rolling 使用"
        "两阶段 SAA。信息集不同，差距不能全部归因于算法优劣。",
        "- PPO 的被动调拨 y 按 prob.priority/tc_route 的优先级规则（本地优先 + "
        "跨仓全局贪心），不是动作；基线/rolling 的 y 由 LP 优化（成本仍用 "
        "tc_route，但不强制优先级顺序）。策略类不同，差距含履约规则差异。",
        "- 观测不含 t，PPO 是**时不变（stationary）** state-only 策略，无法表示"
        "末期 ramp-down；有限视界两端（尤其末 10% 期）的差距含此结构限制。",
        "- PPO 没有 q0 状态；基线有初始订单/在途，配对差含首期初始条件差异。",
        "- 动作投影会把 Gaussian 原始动作改成可行动作；投影量见第 7 节，"
        "不能把原始动作当作实际执行量。",
        "- 本模块只做单环境 CPU PPO；正式结论需要 200k steps 训练与多次种子复现。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO tau-first evaluation")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--config", default=None,
                    help="PPO 配置 TOML（默认 ppo_taufirst/config.toml）")
    ap.add_argument("--compare-config", default=None)
    ap.add_argument("--m-paths", type=int, default=None)
    ap.add_argument("--tag", default="ppo_taufirst_eval")
    ap.add_argument("--archive-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rolling-dir", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow-terminal-q", action="store_true",
                    help="覆盖策略 meta，末期允许 q_T>0")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ppo_cfg_path = Path(args.config) if args.config else DEFAULT_PPO_CONFIG
    if not ppo_cfg_path.is_absolute():
        ppo_cfg_path = PROJECT / ppo_cfg_path
    ppo_data = load_ppo_config(ppo_cfg_path)
    compare_path = resolve_compare_config(args.compare_config, ppo_data)
    cfg = load_compare_config(compare_path)
    prob = load_problem(cfg)

    # torch 会导入标准库 code 模块，必须在 load_problem 之后再导入
    from ppo_taufirst.ppo import deterministic_actions, load_policy
    import torch
    torch.set_num_threads(max(1, int(args.threads)))

    policy_src = Path(args.policy)
    if not policy_src.is_absolute():
        policy_src = PROJECT / policy_src
    if not policy_src.is_file():
        raise FileNotFoundError(f"策略文件不存在: {policy_src}")

    model, meta = load_policy(policy_src, device="cpu")
    if meta.get("kind") != "ppo_taufirst":
        raise ValueError(f"不是 ppo_taufirst 策略: {meta.get('kind')}")
    for k, v in (("T", int(prob.T)), ("N_R", int(prob.N_R)),
                 ("G", int(prob.G)), ("L", int(prob.L))):
        if int(meta.get(k, -1)) != v:
            raise ValueError(f"策略 meta {k}={meta.get(k)} 与模型 {v} 不一致")
    if int(meta.get("obs_dim", -1)) != 36 or int(meta.get("act_dim", -1)) != 10:
        raise ValueError(f"策略维度异常: obs_dim={meta.get('obs_dim')}, "
                         f"act_dim={meta.get('act_dim')}")
    pd_hash = sha256_file(cfg.problem_data) if Path(cfg.problem_data).is_file() \
        else None
    if meta.get("problem_data_hash") and pd_hash and \
            meta["problem_data_hash"] != pd_hash:
        raise ValueError(
            "策略训练的 problem_data 与当前比较配置不一致，配对无效：\n"
            f"  policy={meta['problem_data_hash'][:16]}\n"
            f"  config={pd_hash[:16]}")
    if not meta.get("problem_data_hash"):
        print("[警告] 策略 meta 没有 problem_data_hash，无法校验问题数据")

    reward_scale = float(meta.get("reward_scale", 1e-3))
    terminal_q_zero = bool(meta.get("terminal_q_zero", True))
    if args.allow_terminal_q:
        terminal_q_zero = False

    base = load_baseline(cfg, prob)
    M = int(args.m_paths if args.m_paths is not None
            else ppo_data["evaluation"]["m_paths"])
    if M < 1 or M > base.P:
        raise ValueError(f"m_paths={M} 必须在 [1,{base.P}]")
    scene = np.ascontiguousarray(base.scene[:M])
    theta_paths = np.stack([
        np.stack([np.asarray(prob.pool[int(s)][1], dtype=float)
                  for s in scene[p]], axis=0)
        for p in range(M)
    ], axis=0)
    envs = [
        TauFirstEnv(prob, theta_path=theta_paths[p],
                    reward_scale=reward_scale,
                    terminal_q_zero=terminal_q_zero)
        for p in range(M)
    ]

    T, NR, G = int(prob.T), int(prob.N_R), int(prob.G)
    arrays = {
        "I_in": np.zeros((M, T, NR)),
        "o_in": np.zeros((M, T, 2, NR)),
        "n_in": np.zeros((M, T, NR, G)),
        "q": np.zeros((M, T, NR)),
        "tau": np.zeros((M, T, NR)),
        "taup": np.zeros((M, T, NR)),
        "taum": np.zeros((M, T, NR)),
        "y": np.zeros((M, T, NR, NR)),
        "I_out": np.zeros((M, T, NR)),
        "o_next": np.zeros((M, T, 2, NR)),
        "n_out": np.zeros((M, T, NR, G)),
        "F": np.zeros((M, T, NR)),
        "W_in": np.zeros((M, T, NR)),
        "W_out": np.zeros((M, T, NR)),
        "D": np.zeros((M, T, NR)),
        "cost": np.zeros((M, T)),
        "terms": np.zeros((M, T, 5)),
        "reward": np.zeros((M, T)),
        "action": np.zeros((M, T, 10)),
        "tau_raw": np.zeros((M, T, NR)),
        "q_raw": np.zeros((M, T, NR)),
        "q_upper": np.zeros((M, T, NR)),
        "tau_l1": np.zeros((M, T)),
        "tau_max": np.zeros((M, T)),
        "q_l1": np.zeros((M, T)),
        "q_max": np.zeros((M, T)),
        "terminal_q_zero": np.zeros((M, T), dtype=bool),
        "scene": np.asarray(scene, dtype=np.int16).copy(),
        "theta": np.asarray(scene, dtype=np.int16).copy(),
    }
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    for t in range(T):
        obs_batch = np.stack([e.observation() for e in envs], axis=0)
        actions = deterministic_actions(model, obs_batch)
        for p, e in enumerate(envs):
            I0, o0, n0 = e.I.copy(), np.stack([e.o1, e.o2]), e.n.copy()
            _, reward, done, info = e.step(actions[p])
            arrays["I_in"][p, t] = I0
            arrays["o_in"][p, t] = o0
            arrays["n_in"][p, t] = n0
            arrays["q"][p, t] = info["q"]
            arrays["tau"][p, t] = info["tau"]
            arrays["taup"][p, t] = np.maximum(info["tau"], 0.0)
            arrays["taum"][p, t] = np.maximum(-info["tau"], 0.0)
            arrays["y"][p, t] = info["y"]
            arrays["I_out"][p, t] = info["I_next"]
            arrays["o_next"][p, t] = info["o_next"]
            arrays["n_out"][p, t] = info["n_next"]
            arrays["F"][p, t] = info["F_new"]
            arrays["W_in"][p, t] = info["W_in"]
            arrays["W_out"][p, t] = info["W_out"]
            arrays["D"][p, t] = info["D"]
            arrays["cost"][p, t] = info["cost"]
            arrays["terms"][p, t] = info["terms"]
            arrays["reward"][p, t] = reward
            arrays["action"][p, t] = actions[p]
            arrays["tau_raw"][p, t] = info["projection"]["tau_raw"]
            arrays["q_raw"][p, t] = info["projection"]["q_raw"]
            arrays["q_upper"][p, t] = info["projection"]["q_upper"]
            arrays["tau_l1"][p, t] = info["projection"]["tau_l1"]
            arrays["tau_max"][p, t] = info["projection"]["tau_max"]
            arrays["q_l1"][p, t] = info["projection"]["q_l1"]
            arrays["q_max"][p, t] = info["projection"]["q_max"]
            arrays["terminal_q_zero"][p, t] = \
                info["projection"]["terminal_q_zero"]
            if t == T - 1 and not done:
                raise RuntimeError(f"path {p} 未在末期结束")
    wall_sim = time.perf_counter() - t0

    totals = arrays["cost"].sum(axis=1)
    term_totals = arrays["terms"].sum(axis=1)
    arrays["totals"] = totals.copy()
    arrays["term_totals"] = term_totals.copy()

    audit, bad = audit_rollout(
        prob, arrays, tol=float(cfg.tol), state_tol=float(cfg.state_tol),
        reward_scale=reward_scale)
    audit_passed = len(bad) == 0
    terms_dev = float(np.max(np.abs(
        arrays["terms"].sum(axis=2) - arrays["cost"])))

    ps = paired_stats(totals, base.totals[:M])
    terms_stat = term_stats(arrays["terms"], base.terms[:M], TERM_NAMES)
    beh_ppo = behavior_from_ppo(prob, arrays)
    beh_base = behavior_from_base(prob, base, M)
    tp = time_profile(arrays["terms"], base.terms[:M])
    tsens = terminal_sensitivity(prob, arrays, base, M, totals)

    proj = {
        "tau_l1_mean": float(arrays["tau_l1"].mean()),
        "tau_l1_max": float(arrays["tau_l1"].max()),
        "tau_max_mean": float(arrays["tau_max"].mean()),
        "tau_max_max": float(arrays["tau_max"].max()),
        "q_l1_mean": float(arrays["q_l1"].mean()),
        "q_l1_max": float(arrays["q_l1"].max()),
        "q_max_mean": float(arrays["q_max"].mean()),
        "q_max_max": float(arrays["q_max"].max()),
        "q_upper_mean": float(arrays["q_upper"].mean()),
        "q_upper_min": float(arrays["q_upper"].min()),
        "q_upper_max": float(arrays["q_upper"].max()),
        "terminal_zero_frac": float(arrays["terminal_q_zero"].mean()),
        "tau_raw_sum_max_abs": float(np.abs(
            arrays["tau_raw"].sum(axis=2)).max()),
    }

    rolling = None
    if args.rolling_dir:
        rolling = load_rolling(args.rolling_dir, M, scene)
        if rolling and not rolling.get("error") and rolling.get("totals") is not None:
            rolling["paired_with_ppo"] = paired_stats(
                totals, rolling["totals"])
            rolling["paired_vs_base"] = paired_stats(
                rolling["totals"], base.totals[:M])
            rolling["terms"] = rolling["terms"].tolist() \
                if isinstance(rolling.get("terms"), np.ndarray) else rolling.get("terms")
            rolling["totals"] = rolling["totals"].tolist()

    # ---- 归档
    if args.out:
        run_dir = Path(args.out)
        if not run_dir.is_absolute():
            run_dir = PROJECT / run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"结果目录非空，拒绝覆盖: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(
            args.tag, args.archive_root or ppo_data["run"].get(
                "archive_root", "results"))
    created_at = datetime.now().isoformat(timespec="seconds")
    copy_configs(run_dir, cfg, PROJECT, ppo_cfg_path)
    shutil.copy2(policy_src, run_dir / "policy.pt")
    # 训练收敛曲线随评估结果一起归档（复制或从 train_log.csv 生成）
    train_dir = policy_src.parent
    curve_dst = run_dir / "train_curve.png"
    train_curve = None
    curve_src = train_dir / "train_curve.png"
    if curve_src.is_file():
        shutil.copy2(curve_src, curve_dst)
        train_curve = str(curve_dst)
    elif (train_dir / "train_log.csv").is_file():
        # 独立子进程：torch 与 matplotlib 的 OpenMP runtime 不能同进程混用
        cmd = [
            sys.executable, "-B",
            str(Path(__file__).resolve().parent / "plot_train.py"),
            "--run", str(train_dir), "--out", str(curve_dst),
            "--baseline-mean", f"{float(base.totals[:M].mean()):.6f}",
        ]
        try:
            res = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True,
                                 text=True, timeout=120, check=False)
            if res.returncode != 0:
                print(f"[警告] 收敛曲线生成失败: "
                      f"{res.stderr.strip()[:500]}")
            elif curve_dst.is_file():
                train_curve = str(curve_dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 收敛曲线生成失败: {exc}")
    save_sim_npz(run_dir, SimpleNamespace(arrays=arrays), M)
    write_json(run_dir / "sim_meta.json", {
        "scene_hash": sha256_array(scene),
        "scene_shape": list(scene.shape),
        "M": M,
        "baseline_full_P": int(base.P),
        "baseline_full_mean": base.mean,
        "baseline_subset_mean": float(base.totals[:M].mean()),
        "policy_src": str(policy_src),
        "policy_meta": meta,
        "terminal_q_zero": terminal_q_zero,
        "reward_scale": reward_scale,
    })
    write_json(run_dir / "audit.json", {
        "audit": audit, "bad": bad, "passed": audit_passed,
        "terms_vs_stage_cost_max_dev": terms_dev,
        "reward_scale": reward_scale,
    })
    comparison = {
        "meta": meta,
        "compare_config": str(cfg.config_path),
        "policy_src": str(policy_src),
        "M": M,
        "scene_hash": sha256_array(scene),
        "baseline_full_mean": base.mean,
        "baseline_subset_mean": float(base.totals[:M].mean()),
        "paired": ps,
        "terms": terms_stat,
        "behavior": {"ppo": beh_ppo, "baseline": beh_base},
        "time": tp,
        "terminal_sensitivity": tsens,
        "audit": {"passed": audit_passed, "max": audit, "bad": bad,
                  "terms_vs_stage_cost_max_dev": terms_dev},
        "projection": proj,
        "rolling": rolling,
        "train_curve": train_curve,
        "wall_sim_s": float(wall_sim),
    }
    write_json(run_dir / "comparison.json", comparison)
    write_json(run_dir / "behavior.json", comparison["behavior"])
    write_json(run_dir / "time_profile.json", tp)
    write_json(run_dir / "terminal_sensitivity.json", tsens)
    with (run_dir / "time_profile.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "ppo_cost", "baseline_cost", "diff", "cum_diff"])
        for i in range(T):
            w.writerow([i + 1, f"{tp['policy'][i]:.6f}",
                        f"{tp['baseline'][i]:.6f}", f"{tp['diff'][i]:.6f}",
                        f"{tp['cum_diff'][i]:.6f}"])
    with (run_dir / "time_profile_deciles.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tp["deciles"][0].keys()))
        w.writeheader()
        w.writerows(tp["deciles"])

    ctx = {
        "run_dir": str(run_dir), "created_at": created_at,
        "compare_config": str(cfg.config_path),
        "ppo_config": str(ppo_cfg_path),
        "T": T, "N_R": NR, "G": G, "L": int(prob.L),
        "policy_src": str(policy_src), "meta": meta,
        "M": M, "scene_hash": sha256_array(scene),
        "base_full_mean": base.mean,
        "base_subset_mean": float(base.totals[:M].mean()),
        "terminal_q_zero": terminal_q_zero,
        "reward_scale": reward_scale,
        "paired": ps, "terms": terms_stat,
        "behavior": {"ppo": beh_ppo, "baseline": beh_base},
        "time": tp, "audit": audit, "bad": bad,
        "audit_passed": audit_passed, "terms_dev": terms_dev,
        "projection": proj, "rolling": rolling,
        "train_curve": train_curve,
    }
    write_text(run_dir / "summary.md", _summary_md(ctx))
    conclusion = [
        "# 结论 — PPO tau-first（新时序）",
        "",
        f"- 结果目录：`{run_dir}`",
        f"- PPO state-only，M={M}，训练 {meta.get('train_steps')} steps；"
        f"末期 q_T={'0' if terminal_q_zero else '允许正'}。",
        f"- PPO 均值 **{ps['policy_mean']:.4f}** vs 基线子集 "
        f"**{ps['baseline_mean']:.4f}**；配对差 **{ps['diff_mean']:+.4f}**"
        f"（{ps['rel_diff_pct']:+.2f}%），t={ps['t']:.4f}，"
        f"胜率 {100 * ps['win_rate_policy_lower']:.2f}%。",
        f"- 五项差（PPO−基线）：" + "、".join(
            f"{t['term']} {t['diff_mean']:+.2f}" for t in terms_stat) + "。",
        f"- 行为：平均库存 {beh_ppo['mean_inventory']:.4f} vs "
        f"{beh_base['mean_inventory']:.4f}；履约率 "
        f"{beh_ppo['fill_rate']:.4f} vs {beh_base['fill_rate']:.4f}；"
        f"主动调拨量 {beh_ppo['active_transfer_volume']:.4f} vs "
        f"{beh_base['active_transfer_volume']:.4f}。",
        f"- 审计：{'通过' if audit_passed else '未通过（' + str(list(bad)) + '）'}；"
        f"仿真墙钟 {wall_sim:.1f}s。",
        "- 信息集：PPO 是 state-only；SDDP/rolling 使用更丰富信息，"
        "差距不能全部归因于算法。M=50 的 CI 只对当前子集成立。",
        "",
    ]
    write_text(run_dir / "conclusion.md", "\n".join(conclusion))

    run = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "config": str(cfg.config_path),
        "ppo_config": str(ppo_cfg_path),
        "policy_src": str(policy_src),
        "policy": str(run_dir / "policy.pt"),
        "policy_meta": meta,
        "T": T, "N_R": NR, "G": G, "L": int(prob.L),
        "M": M,
        "baseline": base.name,
        "baseline_full_P": int(base.P),
        "baseline_full_mean": base.mean,
        "baseline_subset_mean": float(base.totals[:M].mean()),
        "scene_hash": sha256_array(scene),
        "policy_mean": ps["policy_mean"],
        "diff_mean": ps["diff_mean"],
        "rel_diff_pct": ps["rel_diff_pct"],
        "audit_passed": audit_passed,
        "wall_sim_s": float(wall_sim),
        "terminal_q_zero": terminal_q_zero,
        "reward_scale": reward_scale,
        "train_curve": train_curve,
    }
    write_json(run_dir / "run.json", run)
    write_json(run_dir / "provenance.json",
               provenance(cfg, prob, meta, baseline=base))
    try:
        append_index(run_dir, {
            "run_id": run_dir.name,
            "kind": "eval",
            "status": "completed",
            "T": T,
            "M": M,
            "seed": meta.get("train_seed", ""),
            "wall_s": f"{wall_sim:.2f}",
            "policy_mean": f"{ps['policy_mean']:.4f}",
            "baseline_mean": f"{ps['baseline_mean']:.4f}",
            "diff_mean": f"{ps['diff_mean']:.4f}",
            "rel_diff_pct": f"{ps['rel_diff_pct']:.4f}",
            "audit_passed": audit_passed,
            "policy": str(run_dir / "policy.pt"),
            "created_at": created_at,
        })
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] index.csv 写入失败: {exc}")

    if not args.quiet:
        print(f"[eval] result_dir={run_dir}")
        print(f"[eval] PPO {ps['policy_mean']:.4f} vs baseline subset "
              f"{ps['baseline_mean']:.4f}; diff {ps['diff_mean']:+.4f} "
              f"({ps['rel_diff_pct']:+.2f}%), t={ps['t']:.3f}, "
              f"win={100 * ps['win_rate_policy_lower']:.2f}%")
        print(f"[eval] audit={'PASS' if audit_passed else 'FAIL'} "
              f"bad={bad}; sim_wall={wall_sim:.1f}s")
    return run_dir


if __name__ == "__main__":
    main()
