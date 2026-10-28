"""滚动时域基准的样本外评估、配对、审计与归档。"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import csv
import json
import os
import time as _time
import tomllib
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common.baseline import (load_baseline, paired_stats, term_stats)
from common.config import load_compare_config
from common.model import load_problem
from common.report import (copy_configs, fmt, make_result_dir, provenance,
                           save_sim_npz, write_json, write_text)
from rolling_horizon.metrics import (baseline_behavior, behavior_stats,
                                    terminal_sensitivity, time_profile)
from rolling_horizon.policy import RollingHorizonPolicy
from rolling_horizon.scenarios import (make_opt_pool,
                                       real_paths_from_baseline)
from rolling_horizon.simulator import (TERM_NAMES, audit_result,
                                      cost_terms_max_dev)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "config" / "compare_T60_base204105.toml"


# ---------------------------------------------------------------- 配置
def load_rolling_config(project: Path, path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else Path(project) / "config" / "policy.toml"
    if not p.is_absolute():
        p = Path(project) / p
    if not p.is_file():
        raise FileNotFoundError(f"滚动策略配置不存在: {p}")
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    for key in ("policy", "output"):
        if key not in data or not isinstance(data[key], dict):
            raise KeyError(f"滚动策略配置缺少 [{key}]: {p}")
    return data


def _subset_base(base, m: int):
    """基线子集包装，供 behavior/terminal 函数复用。"""
    m = int(m)
    return SimpleNamespace(
        P=m,
        totals=np.asarray(base.totals[:m], dtype=float),
        terms=np.asarray(base.terms[:m], dtype=float),
        x_in=np.asarray(base.x_in[:m], dtype=float),
        q=np.asarray(base.q[:m], dtype=float),
        taup=np.asarray(base.taup[:m], dtype=float),
        taum=np.asarray(base.taum[:m], dtype=float),
        S_end=np.asarray(base.S_end[:m], dtype=float),
        y=np.asarray(base.y[:m], dtype=float),
        F=np.asarray(base.F[:m], dtype=float),
        W_in=np.asarray(base.W_in[:m], dtype=float),
        W_out=np.asarray(base.W_out[:m], dtype=float),
        q0=lambda layout: np.asarray(base.q0(layout)[:m], dtype=float),
    )


# ---------------------------------------------------------------- 小工具
def _pct(x) -> str:
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


def _audit_limits(cfg):
    return float(cfg.tol), float(cfg.state_tol)


def _lp_summary(policy: RollingHorizonPolicy) -> dict:
    rows = policy.lp_stats
    two = [r for r in rows if r["status"] == "ok"]
    rec = [r for r in rows if r["status"] == "recourse"]
    t1 = [r for r in two if r["t"] == 1]
    by_t = {}
    for r in two:
        d = by_t.setdefault(r["t"], {"n": 0, "vars": 0, "rows": 0,
                                     "build_s": 0.0, "solve_s": 0.0,
                                     "obj": 0.0})
        d["n"] += 1
        d["vars"] += r["n_vars"]
        d["rows"] += r["n_eq"] + r["n_le"]
        d["build_s"] += r["build_s"]
        d["solve_s"] += r["solve_s"]
        d["obj"] += r["objective"]
    for d in by_t.values():
        d["mean_vars"] = d["vars"] / max(d["n"], 1)
        d["mean_rows"] = d["rows"] / max(d["n"], 1)
        d["mean_objective"] = d["obj"] / max(d["n"], 1)
    return {
        "n_two_stage": len(two),
        "n_recourse": len(rec),
        "n_failures": len(policy.lp_failures),
        "max_vars": int(max((r["n_vars"] for r in two), default=0)),
        "max_rows": int(max((r["n_eq"] + r["n_le"] for r in two), default=0)),
        "t1_n_vars": int(t1[0]["n_vars"]) if t1 else 0,
        "t1_n_eq": int(t1[0]["n_eq"]) if t1 else 0,
        "t1_n_le": int(t1[0]["n_le"]) if t1 else 0,
        "t1_build_s": float(t1[0]["build_s"]) if t1 else 0.0,
        "t1_solve_s": float(t1[0]["solve_s"]) if t1 else 0.0,
        "t1_n": len(t1),
        "t1_mean_solve_s": float(np.mean([r["solve_s"] for r in t1])) if t1 else 0.0,
        "t1_objective_mean": float(np.mean([r["objective"] for r in t1])) if t1 else float("nan"),
        "t1_objective_first": float(t1[0]["objective"]) if t1 else float("nan"),
        "two_stage_build_s_total": float(sum(r["build_s"] for r in two)),
        "two_stage_solve_s_total": float(sum(r["solve_s"] for r in two)),
        "recourse_solve_s_total": float(sum(r["solve_s"] for r in rec)),
        "mean_two_stage_solve_s": float(np.mean([r["solve_s"] for r in two])) if two else 0.0,
        "by_t": [{"t": int(t), **v} for t, v in sorted(by_t.items())],
    }


# ---------------------------------------------------------------- 报告
def _summary_md(ctx: dict) -> str:
    ps = ctx["paired"]
    base_full = ctx["base_full_mean"]
    base_sub = ctx["base_subset_mean"]
    q0_mean = ctx["base_q0_mean"]
    lp = ctx["lp"]
    saa = ctx["saa"]
    tp = ctx["time"]
    audit = ctx["audit"]
    bad = ctx["bad"]
    terms = ctx["terms"]
    pb = ctx["policy_behavior"]
    bb = ctx["baseline_behavior"]
    ts = ctx["terminal"]
    meta = ctx["base_meta"]

    lines = [
        "# 滚动时域（全窗口两阶段 SAA）— 样本外评估与 SDDP 配对",
        "",
        f"- 结果目录：`{ctx['result_dir']}`",
        f"- 创建时间：{ctx['created_at']}",
        f"- 配置：`{ctx['compare_config']}`；策略配置：`{ctx['policy_config']}`",
        f"- 模型：T={ctx['T']}，N_R={ctx['N_R']}，G={ctx['G']}，L={ctx['L']}",
        f"- 基线：`{ctx['baseline_name']}`；基线全量均值 {fmt(base_full)}，"
        f"路径数 {ctx['base_P']}；本评估子集 M={ctx['M']}，子集均值 {fmt(base_sub)}",
        "",
        "## 1. 协议",
        "",
        f"- 优化池：N={ctx['N']} 条独立 Theta 路径，seed_opt={ctx['seed_opt']}，"
        f"pool_hash=`{ctx['pool_hash']}`，shape={ctx['pool_shape']}。",
        f"- 窗口：W=T-t+1（剩余全部时域）；终端 V_{{T+1}}=0，无终端近似。",
        "- 状态：`I,n,o1..o(L-1)`；W=N−Σn 由状态导出，不存储。",
        "- 每期只有一个订货 q_t；purchase_timing=`current_q`，"
        "采购成本按当期 C q_t；没有 q_in/q_out。",
        f"- 第一阶段只有 q_t；其余 q/y/tau/S_end/I/n/N/o 均为情景追索。",
        f"- t 期 Theta_t 作为随机变量；真实 Theta_t 只在订货后由仿真观测，"
        f"不进入 order。",
        f"- 真实路径：取基线 npz 的 scene 前 M={ctx['M']} 条，"
        f"subset_hash=`{ctx['scene_hash']}`；与优化池种子独立；不重新采样。",
        f"- 初始 I0/o0/n0 取 problem_data；滚动没有独立的 q0 状态。"
        f"基线 q_in 均值 {fmt(q0_mean)}，配对差包含初始在途/订单差异。",
        "- 状态转移：`I_{t+1}=S_end+tau`，`o_{l,t+1}=o_{l+1,t}`，"
        "`o_{L-1,t+1}=q_t`；容量为每期 "
        "`I_t+Σo_t+q_t<=Imax` 和转移后 `I_{t+1}+Σo_{t+1}<=Imax`。",
        "- 主动调拨由两个 LP 直接优化 tau（不是 ss_static 的门限规则）。",
        f"- solver={ctx['solver']}，threads={ctx['threads']}，"
        f"workers={ctx['workers']}；"
        f"基线自身 LB/UB/gap：{meta.get('lb')}/{meta.get('ub')}/{meta.get('gap_pct')}"
        f"（LB 不是基线策略成本）。",
        "",
        "## 2. 主结果（样本外，逐路径配对）",
        "",
        _table(
            ["指标", "滚动", "基线子集", "配对差", "相对差", "95% CI", "t", "胜率"],
            [[
                "均值（万元/路径）", fmt(ps["policy_mean"]),
                fmt(ps["baseline_mean"]), f"{ps['diff_mean']:+.4f}",
                _pct(ps["rel_diff_pct"]),
                f"[{ps['ci95_low']:+.4f}, {ps['ci95_high']:+.4f}]",
                fmt(ps["t"]), f"{100 * ps['win_rate_policy_lower']:.2f}%",
            ]]),
        "",
        f"- 滚动均值 {fmt(ps['policy_mean'])}，基线子集均值 {fmt(ps['baseline_mean'])}，"
        f"配对差 {ps['diff_mean']:+.4f}（{_pct(ps['rel_diff_pct'])}），"
        f"t={fmt(ps['t'])}，胜率 {100 * ps['win_rate_policy_lower']:.2f}%。",
        f"- 基线全量均值 {fmt(base_full)}（{ctx['base_P']} 路径）；子集 M={ctx['M']}。"
        "相对差按子集基线计算。",
        "",
        "## 3. 五项成本（逐路径汇总后配对）",
        "",
        _table(
            ["成本项", "滚动", "基线子集", "差", "t", "95% CI"],
            [[t["term"], fmt(t["policy_mean"]), fmt(t["baseline_mean"]),
              f"{t['diff_mean']:+.4f}", fmt(t["t"]),
              f"[{t['ci95_low']:+.4f}, {t['ci95_high']:+.4f}]"]
             for t in terms]),
        "",
        "## 4. 行为（Σ仓；库存/待修单位为台/件）",
        "",
        _table(
            ["指标", "滚动", "基线子集"],
            [
                ["平均库存", fmt(pb["mean_inventory"]), fmt(bb["mean_inventory"])],
                ["履约率", fmt(pb["fill_rate"]), fmt(bb["fill_rate"])],
                ["平均待修", fmt(pb["mean_backlog"]), fmt(bb["mean_backlog"])],
                ["期末待修", fmt(pb["final_backlog"]), fmt(bb["final_backlog"])],
                ["主动调拨量", fmt(pb["active_transfer_volume"]),
                 fmt(bb["active_transfer_volume"])],
                ["总订货量", fmt(pb["total_orders_placed"]),
                 fmt(bb["total_orders_placed"])],
                ["订货频率", f"{100 * pb['order_frequency']:.2f}%",
                 f"{100 * bb['order_frequency']:.2f}%"],
                ["跨仓替换", fmt(pb["cross_replacements"]),
                 fmt(bb["cross_replacements"])],
                ["本地替换", fmt(pb["local_replacements"]),
                 fmt(bb["local_replacements"])],
            ]),
        "",
        "## 5. 时间剖面（逐期总成本差）",
        "",
        f"- 首 10% 期：{tp['first_10pct_diff']:+.4f}；中间段："
        f"{tp['middle_diff']:+.4f}；末 10% 期：{tp['last_10pct_diff']:+.4f}。",
        f"- 最差十分位：{tp['worst_decile']['decile']}（t="
        f"{tp['worst_decile']['t_first']}–{tp['worst_decile']['t_last']}，"
        f"{tp['worst_decile']['diff']:+.4f}）；"
        f"最好十分位：{tp['best_decile']['decile']}（t="
        f"{tp['best_decile']['t_first']}–{tp['best_decile']['t_last']}，"
        f"{tp['best_decile']['diff']:+.4f}）。",
        "",
        _table(
            ["十分位", "期", "滚动", "基线", "差"],
            [[f"{d['decile']}", f"{d['t_first']}–{d['t_last']}",
              fmt(d["policy_cost"]), fmt(d["baseline_cost"]),
              f"{d['diff']:+.4f}"] for d in tp["deciles"]]),
        "",
        "## 6. 期末值敏感性（仅诊断；主目标无期末值）",
        "",
        _table(
            ["口径", "滚动", "基线", "差", "t", "相对差"],
            [[r["label"], fmt(r["policy_mean"]), fmt(r["baseline_mean"]),
              f"{r['diff_mean']:+.4f}", fmt(r["t"]),
              _pct(r["rel_diff_pct"])] for r in ts]),
        "",
        "## 7. 约束审计",
        "",
        f"- 审计结果：{'**通过**' if not bad else '**失败**'}"
        f"（tol={ctx['tol']:g}, state_tol={ctx['state_tol']:g}）。",
        f"- 最大检查值：{json.dumps(audit, ensure_ascii=False)}",
        f"- 五项和与总成本最大偏差：{fmt(ctx['cost_identity_dev'])}。",
        f"- 审计范围：{ctx['M']} 路径 × {ctx['T']} 期。",
        "",
        "## 8. LP 规模、求解时间与 SAA",
        "",
        f"- 两阶段 LP 次数：{lp['n_two_stage']}；追索 LP 次数：{lp['n_recourse']}；"
        f"失败：{lp['n_failures']}。",
        f"- t=1 最大 LP：变量 {lp['t1_n_vars']}，等式 {lp['t1_n_eq']}，"
        f"不等式 {lp['t1_n_le']}；首条 build {lp['t1_build_s']:.3f} s，"
        f"solve {lp['t1_solve_s']:.3f} s；t=1 平均 solve "
        f"{lp['t1_mean_solve_s']:.3f} s。",
        f"- 两阶段 LP：build 合计 {lp['two_stage_build_s_total']:.1f} s，"
        f"solve 合计 {lp['two_stage_solve_s_total']:.1f} s；"
        f"追索 solve 合计 {lp['recourse_solve_s_total']:.1f} s。",
        f"- 整个仿真墙钟：{ctx['wall_s']:.1f} s（{ctx['M']} 路径，"
        f"{ctx['workers']} 个 path-worker）；进度日志 `progress.log`。",
        "- 注：LP 目标值是**样本内 SAA 值**，只用于产生 q_t，**不作为绩效**；"
        f"t=1 的 SAA 目标均值 {fmt(saa['t1_objective_mean'])}，"
        f"首条 {fmt(saa['t1_objective_first'])}；样本外均值 {fmt(ps['policy_mean'])}。"
        "两者口径不完全相同（SAA 用固定池的期望，真实执行逐期重优化），"
        "差值只能作为 SAA 过拟合/模型差异的诊断。",
        "",
        "## 9. 结论（按实际数字）",
        "",
        ctx["conclusion_text"],
        "",
        "## 10. 限制",
        "",
        f"- N={ctx['N']} 是文档建议 N=50 的缩减；M={ctx['M']} 是基线 "
        f"{ctx['base_P']} 条测试路径的子集。扩大 N/M 会改变结果，"
        "本报告只对当前规模负责。",
        "- 滚动没有独立 q0 状态（初始 o_{L-1}=0）；基线自身 q_in/q0 非零，"
        "配对差包含初始在途/订单差异；不能把全部差异解释成策略优劣。",
        "- 真实路径来自基线 scene 子集（与 SDDP 同路径），不是独立 seed_real 重采样；"
        "与优化池独立、无未来信息，但不是文档字面的 seed_real。",
        "- 开环时变 (s_t,S_t) 基准尚未实现，无法给出 C_roll − C_open 的反馈价值归因；"
        "本报告只给 C_SDDP − C_roll 的配对差。",
        "- 采购按当期 Cq 计，与主 SDDP 按 q_in 计费的时点不同；"
        "q0/qT 边界差异必须单独看，不能把逐期剖面差异全部归因于策略。",
        "- 未额外加入安全界/fallback；第一阶段 q_t 只有 Qmax 和两阶段 LP 的约束。"
        "若真实 Θ 下追索 LP 不可行，应如实报告，而不是偷偷加约束。",
        "- 不把 SDDP 的 LB 当作基线策略成本；基线自身有 gap。",
        "",
    ]
    return "\n".join(lines)


def _comparison_md(ctx: dict) -> str:
    ps = ctx["paired"]
    terms = ctx["terms"]
    lp = ctx["lp"]
    return "\n".join([
        "# 滚动时域 vs SDDP 基线 — 对比摘要",
        "",
        _table(
            ["方法", "均值", "配对差", "相对差", "t", "胜率", "审计", "LP 次数", "墙钟"],
            [["滚动时域（全窗口 SAA）", fmt(ps["policy_mean"]),
              f"{ps['diff_mean']:+.4f}", _pct(ps["rel_diff_pct"]),
              fmt(ps["t"]), f"{100 * ps['win_rate_policy_lower']:.2f}%",
              "通过" if not ctx["bad"] else "失败", lp["n_two_stage"] + lp["n_recourse"],
              f"{ctx['wall_s']:.1f}s"],
             ["SDDP 基线（子集）", fmt(ps["baseline_mean"]), "—", "—", "—",
              "—", "—", "—", "—"]]),
        "",
        _table(
            ["成本项", "滚动", "基线子集", "差", "t"],
            [[t["term"], fmt(t["policy_mean"]), fmt(t["baseline_mean"]),
              f"{t['diff_mean']:+.4f}", fmt(t["t"])] for t in terms]),
        "",
        f"- N={ctx['N']}，M={ctx['M']}，seed_opt={ctx['seed_opt']}，"
        f"workers={ctx['workers']}；真实路径 subset_hash=`{ctx['scene_hash']}`。",
        f"- LP：t=1 最大 {lp['t1_n_vars']} 变量 / {lp['t1_n_eq'] + lp['t1_n_le']} 行；"
        f"两阶段 solve 合计 {lp['two_stage_solve_s_total']:.1f}s；"
        f"仿真墙钟 {ctx['wall_s']:.1f}s。",
        "- 基线自身 LB/UB/gap 仍有效；不要把 LB 当基线策略成本。",
        "- 口径：状态 I,n,o1..o(L-1)；每期单一 q；采购按当期 Cq；"
        "期初+转移后容量；追索另有 cap_end；详见 result 目录的 summary.md。",
        "",
        ctx["conclusion_text"],
        "",
    ])


def _conclusion_md(ctx: dict) -> str:
    ps = ctx["paired"]
    terms = ctx["terms"]
    tmap = {t["term"]: t for t in terms}
    return "\n".join([
        "# 结论 — 滚动时域（全窗口两阶段 SAA）",
        "",
        f"- 结果目录：`{ctx['result_dir']}`",
        f"- N={ctx['N']}，M={ctx['M']}，seed_opt={ctx['seed_opt']}，"
        f"workers={ctx['workers']}；"
        f"基线子集 M={ctx['M']}（全量均值 {fmt(ctx['base_full_mean'])}）。",
        f"- 滚动样本外均值 **{fmt(ps['policy_mean'])}** vs 基线子集 "
        f"**{fmt(ps['baseline_mean'])}**；配对差 **{ps['diff_mean']:+.4f}**"
        f"（{_pct(ps['rel_diff_pct'])}），t={fmt(ps['t'])}，"
        f"胜率 {100 * ps['win_rate_policy_lower']:.2f}%。",
        f"- 五项差（滚动−基线）：采购 {tmap['采购']['diff_mean']:+.2f}、"
        f"持有 {tmap['持有']['diff_mean']:+.2f}、停机 {tmap['停机']['diff_mean']:+.2f}、"
        f"主动 {tmap['主动调拨']['diff_mean']:+.2f}、被动 {tmap['被动调拨']['diff_mean']:+.2f}。",
        f"- 行为：平均库存 {fmt(ctx['policy_behavior']['mean_inventory'])} vs "
        f"{fmt(ctx['baseline_behavior']['mean_inventory'])}；"
        f"履约率 {fmt(ctx['policy_behavior']['fill_rate'])} vs "
        f"{fmt(ctx['baseline_behavior']['fill_rate'])}；"
        f"主动调拨 {fmt(ctx['policy_behavior']['active_transfer_volume'])} vs "
        f"{fmt(ctx['baseline_behavior']['active_transfer_volume'])}。",
        f"- 审计：{'通过' if not ctx['bad'] else '失败'}；"
        f"t=1 LP {ctx['lp']['t1_n_vars']} 变量，"
        f"仿真墙钟 {ctx['wall_s']:.1f}s。",
        f"- 滚动没有独立 q0 状态；N={ctx['N']}/M={ctx['M']} 是当前规模；"
        "结论只对当前规模与口径成立。",
        "",
        ctx["conclusion_text"],
        "",
    ])


# ---------------------------------------------------------------- 主评估
def evaluate(
    project: str | Path,
    *,
    config_path: str | Path | None = None,
    policy_config: str | Path | None = None,
    tag: str | None = None,
    n_scenarios: int | None = None,
    real_paths: int | None = None,
    seed_opt: int | None = None,
    archive_root: str | Path | None = None,
    quick: bool = False,
    threads: int | None = None,
    workers: int | None = None,
    log_every: int | None = None,
    time_limit: float | None = None,
    solver: str | None = None,
    skip_audit: bool = False,
) -> Path:
    project = Path(project).resolve()
    pcfg_path = (Path(policy_config) if policy_config is not None
                 else project / "rolling_horizon" / "config" / "policy.toml")
    pcfg = load_rolling_config(project, pcfg_path)
    pol_cfg = pcfg["policy"]
    out_cfg = pcfg["output"]

    if quick:
        N = int(pol_cfg.get("smoke_n", 2))
        M = int(pol_cfg.get("smoke_m", 2))
    else:
        N = int(n_scenarios if n_scenarios is not None
                else pol_cfg["n_scenarios"])
        M = int(real_paths if real_paths is not None
                else pol_cfg["real_paths"])
    seed = int(seed_opt if seed_opt is not None
               else pol_cfg["seed_opt"])
    threads = int(threads if threads is not None
                  else pol_cfg.get("threads", 1))
    workers = int(workers if workers is not None
                  else pol_cfg.get("workers", 1))
    log_every = int(log_every if log_every is not None
                    else pol_cfg.get("log_every", 1))
    time_limit = (float(time_limit) if time_limit is not None
                  else float(pol_cfg.get("time_limit", 0.0)))
    solver = str(solver if solver is not None
                 else pol_cfg.get("solver", "copt"))

    env_cfg = os.environ.get("COMPARE_CONFIG")
    cfg_path = (Path(config_path) if config_path is not None
                else Path(env_cfg) if env_cfg
                else Path(DEFAULT_CONFIG))
    if not cfg_path.is_absolute():
        cfg_path = project / cfg_path
    cfg = load_compare_config(cfg_path)
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)
    if M > base.P:
        raise ValueError(f"M={M} > 基线路径数 {base.P}")
    rp = real_paths_from_baseline(base, M)
    opt = make_opt_pool(prob, N, seed)

    # 结果目录在仿真前创建；失败/中断也保留日志与 run.json
    out_tag = tag if tag is not None else out_cfg["tag"]
    if quick:
        out_tag = f"{out_tag}_quick"
    result_dir = Path(make_result_dir(
        archive_root if archive_root is not None else project,
        "rolling_horizon", out_tag))
    copy_configs(result_dir, cfg, project, policy_path=pcfg_path)
    run_log = open(result_dir / "run.log", "a", encoding="utf-8")
    progress_log = open(result_dir / "progress.log", "a", encoding="utf-8")

    def _write_log(fh, prefix: str, msg: str):
        line = f"{datetime.now().isoformat(timespec='seconds')} {prefix} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    def log(msg: str):
        _write_log(run_log, "[rolling]", msg)

    def plog(msg: str):
        _write_log(progress_log, "", msg)
        run_log.write(f"{datetime.now().isoformat(timespec='seconds')} "
                      f"{msg}\n")
        run_log.flush()

    write_json(result_dir / "run.json", {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "tag": result_dir.name,
        "quick": bool(quick),
        "config": str(cfg.config_path),
        "policy_config": str(pcfg_path),
        "N": int(N), "M": int(M), "seed_opt": int(seed),
        "solver": solver, "threads": int(threads),
        "workers": int(workers),
    })
    log(f"开始滚动仿真：N={N} M={M} seed_opt={seed} "
        f"workers={workers} threads={threads} solver={solver}")

    opts = dict(pol_cfg)
    opts.update({
        "solver": solver, "threads": threads, "workers": workers,
        "log_every": log_every, "verbose": True,
        "time_limit": time_limit,
    })
    policy = RollingHorizonPolicy(
        prob, opts, opt["pool"], seed_opt=seed,
        pool_hash=opt["pool_hash"], log=plog)

    from rolling_horizon.simulator import Simulator
    sim = Simulator(prob)
    t_start = datetime.now()
    t0 = _time.perf_counter()
    try:
        res = sim.simulate(policy, rp["scene"], record=True)
        wall_s = _time.perf_counter() - t0
        t_end = datetime.now()
        if skip_audit:
            audit, bad = {}, {}
        else:
            audit, bad = audit_result(
                prob, res, tol=cfg.tol, state_tol=cfg.state_tol)
    except Exception as exc:  # noqa: BLE001 - 失败也要归档
        wall_s = _time.perf_counter() - t0
        msg = traceback.format_exc()
        write_text(result_dir / "error.txt", msg)
        write_json(result_dir / "run.json", {
            "status": "failed",
            "started_at": t_start.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "wall_s": float(wall_s),
            "result_dir": str(result_dir),
            "tag": result_dir.name,
            "quick": bool(quick),
            "config": str(cfg.config_path),
            "policy_config": str(pcfg_path),
            "N": int(N), "M": int(M), "seed_opt": int(seed),
            "solver": solver, "threads": int(threads),
            "workers": int(workers),
            "error": str(exc),
            "lp_failures": list(policy.lp_failures),
            "lp": _lp_summary(policy),
        })
        log(f"FAILED: {exc}")
        run_log.close()
        progress_log.close()
        raise
    cost_dev = cost_terms_max_dev(res)

    ps = paired_stats(res.totals, rp["totals"])
    terms = term_stats(res.arrays["terms"], rp["terms"], names=TERM_NAMES)
    pb = behavior_stats(prob, res)
    base_sub = _subset_base(base, M)
    bb = baseline_behavior(prob, base_sub)
    tp = time_profile(res.arrays["terms"], rp["terms"])
    ts = terminal_sensitivity(prob, res, base_sub)
    lp = _lp_summary(policy)
    saa = {
        "t1_objective_mean": lp["t1_objective_mean"],
        "t1_objective_first": lp["t1_objective_first"],
        "out_of_sample_mean": ps["policy_mean"],
        "gap_insample_minus_oos": (
            lp["t1_objective_mean"] - ps["policy_mean"]),
        "note": "SAA 目标是样本内期望，只用于产生 q；不作为绩效。",
    }

    base_full_mean = float(base.mean)
    base_sub_mean = float(np.mean(rp["totals"]))
    base_q0 = np.asarray(base.q0(prob.layout), dtype=float)
    base_q0_mean = float(np.mean(base_q0))
    scene_hash = rp["scene_hash"]
    pool_hash = opt["pool_hash"]

    # 结论文字由实际符号决定，不预设主次
    if ps["diff_mean"] < 0:
        verdict = (
            f"滚动基准的样本外均值比基线子集低 {-ps['diff_mean']:.4f} "
            f"万元/路径（{-ps['rel_diff_pct']:.2f}%），"
            f"配对 t={ps['t']:.2f}，胜率 {100 * ps['win_rate_policy_lower']:.2f}%。"
            "在 M 条真实路径上滚动更优。")
    else:
        verdict = (
            f"滚动基准的样本外均值比基线子集高 {ps['diff_mean']:.4f} "
            f"万元/路径（{ps['rel_diff_pct']:.2f}%），"
            f"配对 t={ps['t']:.2f}，胜率 {100 * ps['win_rate_policy_lower']:.2f}%。"
            "在 M 条真实路径上滚动不优于基线。")
    conclusion_text = "\n".join([
        verdict,
        "",
        "这是同一组真实路径、同一模型/审计口径下的配对比较；"
        "基线自身有 SDDP gap，滚动优化池的 SAA 目标不是绩效。"
        f"N={N}、M={M}、workers={workers} 等限制见上；"
        "不要外推为 N=50/M=500 的结论。",
        "口径：状态 I,n,o1..o(L-1)；每期单一 q，采购按当期 Cq；"
        "约束按主模型每期/转移后容量；无安全界/fallback。",
    ])

    ctx = {
        "result_dir": str(result_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "compare_config": str(cfg.config_path),
        "policy_config": str(pcfg_path),
        "T": int(prob.T), "N_R": int(prob.N_R),
        "G": int(prob.G), "L": int(prob.L),
        "baseline_name": base.name,
        "base_P": int(base.P),
        "base_full_mean": base_full_mean,
        "base_subset_mean": base_sub_mean,
        "base_q0_mean": base_q0_mean,
        "base_meta": base.meta,
        "M": int(M), "N": int(N), "seed_opt": int(seed),
        "pool_hash": pool_hash, "pool_shape": list(opt["pool"].shape),
        "scene_hash": scene_hash,
        "solver": solver, "threads": int(threads),
        "workers": int(workers), "log_every": int(log_every),
        "tol": float(cfg.tol), "state_tol": float(cfg.state_tol),
        "paired": ps, "terms": terms,
        "policy_behavior": pb, "baseline_behavior": bb,
        "time": tp, "terminal": ts,
        "lp": lp, "saa": saa,
        "audit": audit, "bad": bad,
        "cost_identity_dev": cost_dev,
        "wall_s": float(wall_s),
        "conclusion_text": conclusion_text,
    }

    # ---- 归档文件 ----
    write_json(result_dir / "policy.json", {
        "policy": policy.params(),
        "options": opts,
        "protocol": {
            "n_scenarios": int(N), "real_paths": int(M),
            "seed_opt": int(seed),
            "pool_hash": pool_hash, "pool_shape": list(opt["pool"].shape),
            "scene_hash": scene_hash,
            "state": "I,n,o1..o(L-1)",
            "purchase_timing": "current_q",
            "window": "full_remaining",
            "terminal": "zero",
            "capacity_side": "period_and_next",
            "solver": solver, "threads": int(threads),
            "workers": int(workers),
        },
    })
    write_json(result_dir / "comparison.json", {
        "headline": {
            "policy_mean": ps["policy_mean"],
            "baseline_subset_mean": ps["baseline_mean"],
            "diff_mean": ps["diff_mean"],
            "rel_diff_pct": ps["rel_diff_pct"],
            "t": ps["t"],
            "ci95": [ps["ci95_low"], ps["ci95_high"]],
            "win_rate": ps["win_rate_policy_lower"],
            "audit_pass": not bad,
        },
        "protocol": {
            "N": int(N), "M": int(M), "seed_opt": int(seed),
            "pool_hash": pool_hash, "scene_hash": scene_hash,
            "state": "I,n,o1..o(L-1)",
            "purchase_timing": "current_q",
            "window": "full_remaining", "terminal": "zero",
            "capacity_side": "period_and_next",
            "solver": solver, "threads": int(threads),
            "workers": int(workers),
            "baseline_full_mean": base_full_mean,
            "baseline_full_P": int(base.P),
            "baseline_subset_mean": base_sub_mean,
            "base_meta": base.meta,
        },
        "paired": ps, "terms": terms,
        "behavior": {"policy": pb, "baseline": bb},
        "time_profile": tp, "terminal_sensitivity": ts,
        "audit": {"checks": audit, "bad": bad,
                  "cost_identity_dev": cost_dev,
                  "tol": float(cfg.tol), "state_tol": float(cfg.state_tol)},
        "lp": lp, "saa": saa, "wall_s": float(wall_s),
        "created_at": ctx["created_at"],
    })
    write_json(result_dir / "audit.json", {
        "checks": audit, "bad": bad,
        "tol": float(cfg.tol), "state_tol": float(cfg.state_tol),
        "cost_identity_dev": cost_dev,
    })
    write_json(result_dir / "provenance.json", {
        **provenance(cfg, prob, policy.params(), baseline=base),
        "rolling": {
            "N": int(N), "M": int(M), "seed_opt": int(seed),
            "pool_hash": pool_hash, "pool_shape": list(opt["pool"].shape),
            "scene_hash": scene_hash,
            "baseline_subset_mean": base_sub_mean,
            "state": "I,n,o1..o(L-1)",
            "purchase_timing": "current_q",
            "solver": solver, "threads": int(threads),
            "capacity_side": "period_and_next",
            "workers": int(workers),
            "policy_config": str(pcfg_path),
            "config_path": str(cfg.config_path),
        },
    })
    write_json(result_dir / "sim_meta.json", {
        "n_paths": int(M), "T": int(prob.T),
        "scene_hash": scene_hash,
        "policy": policy.params(),
        "baseline_subset_mean": base_sub_mean,
        "baseline_full_mean": base_full_mean,
        "audit_pass": not bad,
        "paired": ps,
        "lp": lp,
        "wall_s": float(wall_s),
    })
    if not skip_audit:
        save_sim_npz(result_dir, res, M)
    with open(result_dir / "lp_stats.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "t", "window", "n_scen", "n_vars", "n_eq",
                    "n_le", "build_s", "solve_s", "objective", "solver",
                    "status"])
        for r in policy.lp_stats:
            w.writerow([r["path"], r["t"], r["window"], r["n_scen"],
                        r["n_vars"], r["n_eq"], r["n_le"],
                        f"{r['build_s']:.6f}", f"{r['solve_s']:.6f}",
                        f"{r['objective']:.6f}", r["solver"],
                        r["status"]])
    summary_text = _summary_md(ctx)
    write_text(result_dir / "summary.md", summary_text)
    write_text(result_dir / "comparison.md", _comparison_md(ctx))
    write_text(result_dir / "conclusion.md", _conclusion_md(ctx))
    write_json(result_dir / "run.json", {
        "status": "completed",
        "started_at": t_start.isoformat(timespec="seconds"),
        "finished_at": t_end.isoformat(timespec="seconds"),
        "wall_s": float(wall_s),
        "result_dir": str(result_dir),
        "tag": result_dir.name,
        "quick": bool(quick),
        "config": str(cfg.config_path),
        "policy_config": str(pcfg_path),
        "N": int(N), "M": int(M), "seed_opt": int(seed),
        "pool_hash": pool_hash, "scene_hash": scene_hash,
        "state": "I,n,o1..o(L-1)",
        "purchase_timing": "current_q",
        "capacity_side": "period_and_next",
        "solver": solver, "threads": int(threads),
        "workers": int(workers),
        "policy_mean": ps["policy_mean"],
        "baseline_subset_mean": base_sub_mean,
        "baseline_full_mean": base_full_mean,
        "diff_mean": ps["diff_mean"],
        "rel_diff_pct": ps["rel_diff_pct"],
        "t": ps["t"],
        "win_rate": ps["win_rate_policy_lower"],
        "audit_pass": not bad,
        "lp": lp,
    })

    # append-only index
    index = result_dir.parent / "index.csv"
    new = not index.is_file()
    with open(index, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["created_at", "tag", "N", "M", "seed_opt",
                        "policy_mean", "baseline_subset_mean",
                        "baseline_full_mean", "diff_mean", "rel_diff_pct",
                        "t", "win_rate", "audit_pass", "solver",
                        "wall_s", "result_dir"])
        w.writerow([ctx["created_at"], result_dir.name, N, M, seed,
                    f"{ps['policy_mean']:.6f}", f"{base_sub_mean:.6f}",
                    f"{base_full_mean:.6f}", f"{ps['diff_mean']:.6f}",
                    f"{ps['rel_diff_pct']:.6f}", f"{ps['t']:.6f}",
                    f"{ps['win_rate_policy_lower']:.6f}",
                    int(not bool(bad)), solver, f"{wall_s:.3f}",
                    str(result_dir)])

    print(f"[rolling] 结果目录: {result_dir}")
    print(f"[rolling] 滚动均值 {ps['policy_mean']:.4f} vs 基线子集 "
          f"{base_sub_mean:.4f}；配对差 {ps['diff_mean']:+.4f} "
          f"({ps['rel_diff_pct']:+.2f}%)；t={ps['t']:.2f}；"
          f"审计 {'通过' if not bad else '失败'}")
    print(f"[rolling] N={N} M={M} seed_opt={seed} workers={workers}；"
          f"t=1 LP {lp['t1_n_vars']} 变量；墙钟 {wall_s:.1f}s")
    log(f"完成：均值 {ps['policy_mean']:.4f}，配对差 "
        f"{ps['diff_mean']:+.4f}，审计 {'通过' if not bad else '失败'}，"
        f"墙钟 {wall_s:.1f}s")
    run_log.close()
    progress_log.close()
    return result_dir
