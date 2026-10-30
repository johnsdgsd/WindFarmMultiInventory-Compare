from __future__ import annotations

import sys

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from common.baseline import (load_baseline, paired_stats, sha256_file,
                             term_stats)
from common.config import load_compare_config, load_policy_config
from common.model import load_problem
from ss_static.archive import create_run, output_tag
from ss_static.evaluation import SSEvaluator, SSParams
from ss_static.metrics import (baseline_behavior, behavior_stats,
                               terminal_sensitivity, time_profile)
from ss_static.policy import SSStaticPolicy
from ss_static.simulator import (TERM_NAMES, audit_result,
                                 cost_terms_max_dev)


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _pct(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):+.2f}%"
    except (TypeError, ValueError):
        return str(x)


def _table(headers: list[str], rows: list[list], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(aligns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def load_policy(prob, policy_path: str | Path, cfg=None):
    data = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    pol = SSStaticPolicy.from_dict(prob, data)
    meta = data.get("problem", {})
    if meta:
        if int(meta.get("T", prob.T)) != prob.T:
            raise ValueError(
                f"policy T={meta.get('T')} ≠ 当前模型 T={prob.T}；"
                "不能用 T60 的策略评估 T240")
        if int(meta.get("N_R", prob.N_R)) != prob.N_R:
            raise ValueError("policy N_R 与当前模型不一致")
        if "Imax" in meta and not np.allclose(meta["Imax"], prob.Imax):
            raise ValueError("policy Imax 与当前模型不一致")
        if cfg is not None and meta.get("problem_data_hash"):
            h = sha256_file(cfg.problem_data)
            if h != meta["problem_data_hash"]:
                raise ValueError(
                    "policy problem_data_hash 与当前 problem_data.json 不一致；"
                    "策略与模型不匹配")
    else:
        print("[警告] policy.json 没有 problem 元数据；无法校验 T/Imax")
    return pol, data


def _conclusion(ps: dict, tp: dict, terms_terminal: list[dict],
                active: bool = False) -> str:
    if ps["ci95_low"] > 0:
        sig = (f"在本测试路径上，静态 (s,S) **显著差于** SDDP 基线"
               f"（95% CI 不跨 0，t={ps['t']:.2f}）。")
    elif ps["ci95_high"] < 0:
        sig = (f"在本测试路径上，静态 (s,S) **显著优于** SDDP 基线"
               f"（95% CI 不跨 0，t={ps['t']:.2f}）。")
    else:
        sig = (f"在本测试路径上，静态 (s,S) 与 SDDP 基线的差异"
               f"**不显著**（95% CI 跨 0，t={ps['t']:.2f}）。")
    lines = [
        sig,
        f"配对差（策略−基线）= **{ps['diff_mean']:+.4f} 万元/路径**，"
        f"相对基线 {_pct(ps['rel_diff_pct'])}；"
        f"策略更优路径占比 {ps['win_rate_policy_lower']:.2%}。",
    ]
    total = float(tp["total_diff"])
    if abs(total) > 1e-12 and tp.get("worst_decile"):
        w = tp["worst_decile"]
        lines.append(
            f"差距最集中的十分位是第 {w['decile']} 段"
            f"（第 {w['t_first']}–{w['t_last']} 期，差 {w['diff']:+.4f} 万元/路径）；"
            f"首 10% 期差 {tp['first_10pct_diff']:+.4f}，"
            f"末 10% 期差 {tp['last_10pct_diff']:+.4f}，"
            f"中间段差 {tp['middle_diff']:+.4f}。")
        if tp["middle_diff"] < 0 < total:
            lines.append(
                "中间段策略反而更便宜，说明总差距不能简单解释为"
                "“静态订货全程更差”；需结合时间剖面和期末值敏感性阅读。")
    if terms_terminal:
        rels = [z["rel_diff_pct"] for z in terms_terminal]
        lines.append(
            f"期末库存回收/待修罚值的敏感性：调整后相对差范围为 "
            f"{_pct(min(rels))} 到 {_pct(max(rels))}；"
            "主结果仍以无期末值的模型目标为准。")
    lines += [
        "",
        "**解释边界（必须与结论一起读）：**",
        "- 这是同一批测试路径上的两个**可执行策略**配对；SDDP 基线自身仍有 gap，"
        "LB 是模型下界、不是基线策略成本。",
        ("- 静态规则有主动调拨（规则型 threshold + 每仓门限 h 优化），但仍是规则策略；"
         if active else "- 静态规则无主动调拨；")
        + "履约用固定成本优先级规则、订货只看 IP（把健康等级聚合成 W）；"
        "SDDP 策略使用更丰富的状态与阶段信息。差异不是单纯的“订货规则”差异。",
        "- 静态规则无 stop offset，末期仍按 (s,S) 补货；期末库存/待修不计价值。"
        "期末值敏感性只用于说明这一边界，不改变主目标。",
        "- 测试 `scene` 来自基线同一 300 场景池，属于路径级配对，不是换池外推。",
        "- 成本单位是万元；库存/待修的单位是台或件，不是金额。",
    ]
    return "\n".join(lines)


def _summary_md(prob, cfg, base, pol, policy_data, res, ps, ts_list,
                beh_pol, beh_base, tp, term_list, audit, bad, dev,
                run_id: str, detailed: bool = False) -> str:
    cal = policy_data.get("calibration", {})
    baseline_lb = base.meta.get("lb")
    baseline_gap = base.meta.get("gap_pct")
    pol_options = getattr(pol, "options", {}) or {}
    mode = 1 if pol_options.get("active_transfer", False) else 0
    active_txt = ""
    if mode:
        active_txt = (
            f"；rule={pol_options.get('active_rule')}"
            f"，alpha={pol_options.get('active_fraction')}"
            f"，stop_offset={pol_options.get('active_stop_offset')}"
            f"；每仓保留/目标水平 h 是优化参数")
    h_txt = ""
    if getattr(pol, "h", None) is not None:
        h_txt = f"，h={np.round(pol.h, 4).tolist()}"
    lines = [
        f"# 非时变 (s,S) 对比验证总结（T={prob.T}，"
        f"{base.P} 条测试路径）",
        "",
        "## 1. 运行信息与口径",
        "",
        f"- run_id：`{run_id}`",
        f"- 配置：`{cfg.config_path}`",
        f"- 模型：T={prob.T}，G={prob.G}，N_R={prob.N_R}，L={prob.L}，"
        f"state_dim={prob.state_dim}",
        f"- 基线：`{base.name}`；目录 `{base.result_dir}`；"
        f"测试路径 `{Path(base.npz_path).name}`（{base.P} 条）",
        f"- 测试场景 hash：`{base.scene_hash[:16]}…`",
        f"- 策略：时间无关 (s,S)，mode_tau={mode}{active_txt}；"
        f"算法 `{cal.get('selected_algorithm', '—')}`，"
        f"起点 `{cal.get('selected_start', '—')}`",
        f"- 参数：s={np.round(pol.s, 4).tolist()}，"
        f"S={np.round(pol.S, 4).tolist()}{h_txt}",
        f"- 成本单位：万元；行为量：台/件。",
        "",
        "## 2. 验证协议",
        "",
        f"- 标定：`{cal.get('train_paths', '—')}` 条训练路径"
        f"（seed={cal.get('train_seed', '—')}），"
        f"`{cal.get('val_paths', '—')}` 条验证路径"
        f"（seed={cal.get('val_seed', '—')}）；测试路径不参与搜索/选择。",
        f"- 测试：直接使用基线 npz 的 `scene` 数组，逐路径配对；"
        f"不重新生成测试路径。",
        f"- 基线 LB={_fmt(baseline_lb)}，基线仿真均值={_fmt(base.mean)}，"
        f"基线 gap={_fmt(baseline_gap, 3)}%。"
        f"LB 是模型下界，不是基线策略的最优成本。",
        "",
        "## 3. 主结果",
        "",
        _table(
            ["方法", "均值（万元/路径）", "标准差", "95% CI"],
            [["SDDP 基线", _fmt(base.mean), _fmt(base.std), f"±{_fmt(base.ci95)}"],
             ["时不变 (s,S)", _fmt(res.cost), _fmt(res.std), f"±{_fmt(res.ci95)}"]],
            ["---", "---:", "---:", "---:"]),
        "",
        f"- 配对差（策略−基线）：**{ps['diff_mean']:+.4f}** 万元/路径；"
        f"SE {ps['diff_se']:.4f}；95% CI "
        f"[{ps['ci95_low']:+.4f}, {ps['ci95_high']:+.4f}]；"
        f"t={ps['t']:+.2f}；胜率 {ps['win_rate_policy_lower']:.2%}；"
        f"相对差 {_pct(ps['rel_diff_pct'])}。",
        "",
        "## 4. 成本结构（每路径 T 期合计的均值，万元）",
        "",
        _table(
            ["成本项", "基线", "策略", "差值（策略−基线）", "95% CI"],
            [[z["term"], _fmt(z["baseline_mean"]), _fmt(z["policy_mean"]),
              f"{z['diff_mean']:+.4f}",
              f"[{z['ci95_low']:+.4f}, {z['ci95_high']:+.4f}]"]
             for z in ts_list],
            ["---", "---:", "---:", "---:", "---:"]),
        "",
        "## 5. 行为诊断（库存/待修均为 **Σ仓合计**）",
        "",
    ]
    keys = [
        ("total_failures", "总失效（台/路径）"),
        ("total_replacements", "总替换（台/路径）"),
        ("fill_rate", "替换率（替换/失效）"),
        ("final_backlog", "期末待修（台，Σ仓）"),
        ("mean_backlog", "平均期初待修（台/期，Σ仓）"),
        ("mean_inventory", "平均期初库存（台/期，Σ仓）"),
        ("total_orders_placed", "下单量（含 q0，件/路径）"),
        ("total_orders_charged", "计费下单量（件/路径）"),
        ("order_frequency", "有订单的 (路径,期) 比例"),
        ("local_replacements", "本地替换（件/路径）"),
        ("cross_replacements", "跨仓替换（件/路径）"),
        ("active_transfer_volume", "主动调拨量（件/路径）"),
    ]
    lines += [
        _table(["指标", "基线", "策略"],
               [[label, _fmt(beh_base.get(k)), _fmt(beh_pol.get(k))]
                for k, label in keys], ["---", "---:", "---:"]),
        "",
        "## 6. 时间剖面（成本差按十分位）",
        "",
        _table(
            ["十分位", "期范围", "期数", "基线（万元/路径）", "策略", "差值"],
            [[z["decile"], f"{z['t_first']}–{z['t_last']}", z["n_periods"],
              _fmt(z["baseline_cost"]), _fmt(z["policy_cost"]),
              f"{z['diff']:+.4f}"] for z in tp["deciles"]],
            ["---:", "---", "---:", "---:", "---:", "---:"]),
        "",
        f"- 首 10% 期差 {tp['first_10pct_diff']:+.4f}；"
        f"末 10% 期差 {tp['last_10pct_diff']:+.4f}；"
        f"中间段差 {tp['middle_diff']:+.4f}；总计 {tp['total_diff']:+.4f}。",
    ]
    if tp.get("worst_decile"):
        w = tp["worst_decile"]
        lines.append(
            f"- 最差十分位：第 {w['decile']} 段（{w['t_first']}–{w['t_last']} 期，"
            f"{w['diff']:+.4f}）。")
    if tp.get("best_decile"):
        b = tp["best_decile"]
        lines.append(
            f"- 最好十分位：第 {b['decile']} 段（{b['t_first']}–{b['t_last']} 期，"
            f"{b['diff']:+.4f}）。")
    lines += [
        "",
        "> 时间剖面用于识别有限视界/失效率爬坡的影响；主目标仍是全 T 期总成本。",
        "",
        "## 7. 期末值敏感性（只作敏感性，不改主目标）",
        "",
        _table(
            ["期末值设定", "基线", "策略", "差值", "相对差"],
            [[z["label"], _fmt(z["baseline_mean"]), _fmt(z["policy_mean"]),
              f"{z['diff_mean']:+.4f}", _pct(z["rel_diff_pct"])]
             for z in term_list],
            ["---", "---:", "---:", "---:", "---:"]),
        "",
        "## 8. 约束审计",
        "",
    ]
    if bad:
        lines.append("- **策略审计有违规**：")
        for k, v in sorted(bad.items()):
            lines.append(f"  - `{k}`：{v:.6g}")
    else:
        lines.append("- 策略逐 (路径,期) 审计：全部通过。")
    lines.append(
        f"- 五项和 vs 仿真器 stage cost 的最大偏差：{dev:.3e}"
        "（这是仿真器内部恒等式；真正的 LP 对拍在 "
        "`ss_static/test/_t_lp_equiv.py`）。")
    lines.append(
        f"- 基线 sidecar 审计记录：`audit_violations={base.meta.get('audit_violations', '未记录')}`，"
        f"`terms_vs_stage_cost_max_dev={base.meta.get('terms_vs_stage_cost_max_dev', '未记录')}`；"
        "本次未重跑基线审计。")
    lines += [
        "",
        "## 9. 结论与限制",
        "",
        _conclusion(ps, tp, term_list, active=(mode == 1)),
        "",
        "## 10. 数据文件",
        "",
        f"- `sim_{base.P}paths.npz` / `.json`（与主项目 `simulate.py` 同名同形状）",
        "- `comparison.json` / `comparison.md`（详细表）",
        "- `time_profile.csv` / `time_profile_deciles.csv`",
        "- `terminal_sensitivity.json`",
        "- `summary.md`（本文件）",
    ]
    if detailed:
        lines += [
            "",
            "## 附录：逐期平均成本（万元/路径）",
            "",
            "| 期 | 基线 | 策略 | 差值 | 累计差 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for i in range(prob.T):
            lines.append(
                f"| {i + 1} | {tp['baseline'][i]:.6f} | "
                f"{tp['policy'][i]:.6f} | {tp['diff'][i]:+.6f} | "
                f"{tp['cum_diff'][i]:+.6f} |")
    return "\n".join(lines) + "\n"


def _sim_report_txt(prob, base, res, ps, ts_list, audit, bad, dev,
                    wall_s: float, audit_wall_s: float, npz_name: str) -> str:
    term_mean = res.term_totals.mean(axis=0)
    st = res.step_terms.mean(axis=0).sum(axis=1)
    lines = [
        "=" * 72,
        "非时变 (s,S) 规则策略蒙特卡洛仿真报告",
        "=" * 72,
        f"问题规模      : T={prob.T} G={prob.G} N_R={prob.N_R} L={prob.L} "
        f"状态维 {prob.state_dim}",
        f"测试路径      : {base.P}（基线 `{base.name}` 的 scene 数组，配对）",
        f"策略          : ss_static；s={np.round(res.meta['s'], 4).tolist()}  "
        f"S={np.round(res.meta['S'], 4).tolist()}",
        f"成本单位      : 万元",
        "",
        "— 总成本（每路径 T 期合计，万元）—",
        f"基线均值      : {base.mean:.4f}",
        f"策略均值      : {res.cost:.4f}",
        f"配对差        : {ps['diff_mean']:+.4f}  "
        f"(SE {ps['diff_se']:.4f}, t={ps['t']:+.2f}, "
        f"{_pct(ps['rel_diff_pct'])})",
        f"仿真用时      : {wall_s:.1f} s   审计用时 {audit_wall_s:.1f} s",
        "",
        "— 成本结构（每路径 T 期合计的均值，万元）—",
    ]
    for i, name in enumerate(TERM_NAMES):
        z = ts_list[i]
        lines.append(
            f"  {name:<8}{term_mean[i]:>14,.4f}  "
            f"基线 {z['baseline_mean']:>14,.4f}  "
            f"差 {z['diff_mean']:>+12,.4f}")
    lines += ["", "— 约束审计 —"]
    if bad:
        lines.append("  ★ 有违规：")
        for k, v in sorted(bad.items()):
            lines.append(f"    {k:<12} 最大违约量 {v:,.6g}")
    else:
        lines.append("  策略审计全部通过 ✓")
    lines.append(
        f"  五项和 vs stage cost 最大偏差 {dev:.3e}"
        "（仿真器内部恒等式；LP 对拍见 _t_lp_equiv.py）")
    lines += [
        "",
        "— 逐期平均成本（第 1..T 期，万元/路径）—",
        "  " + " ".join(f"{v:.4g}" for v in st),
        "",
        f"数据文件      : {npz_name}",
        "",
    ]
    return "\n".join(lines)


def run_evaluation(project_dir: str | Path, policy_path: str | Path, *,
                   tag: str | None = None, archive_root=None,
                   skip_audit: bool = False, policy_config=None) -> Path:
    project_dir = Path(project_dir).resolve()
    policy_path = Path(policy_path).resolve()
    cfg = load_compare_config()
    pcfg = load_policy_config(project_dir)
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)
    pol, policy_data = load_policy(prob, policy_path, cfg=cfg)
    evaluator = SSEvaluator(prob, pol.options)
    if pol.h is not None:
        theta = np.concatenate([pol.s, pol.S, pol.h])
    else:
        theta = np.concatenate([pol.s, pol.S])

    run = create_run(
        cfg, project_dir, "eval", output_tag(tag, pcfg, prob.T),
        archive_root,
        manifest={
            "T": int(prob.T), "G": int(prob.G), "N_R": int(prob.N_R),
            "L": int(prob.L), "n_paths": int(base.P),
            "baseline": base.name, "baseline_dir": str(base.result_dir),
            "baseline_npz": str(base.npz_path),
            "scene_hash": base.scene_hash,
            "policy_path": str(policy_path),
            "policy_config": (str(policy_config) if policy_config is not None
                              else str(project_dir / "config" / "policy.toml")),
            "problem_data": str(cfg.problem_data),
            "problem_data_hash": sha256_file(cfg.problem_data)
            if Path(cfg.problem_data).is_file() else None,
            "skip_audit": bool(skip_audit),
        })
    try:
        run.copy_configs(cfg, project_dir, policy_path=policy_config)
        shutil.copy2(policy_path, run.path("policy.json"))
        calib_src = policy_path.parent
        copies = ("calibration.md", "calibration_log.csv",
                  "algorithm_summary.csv", "algorithm_starts.csv",
                  "optimization_history.csv", "paths_train.npz",
                  "paths_val.npz", "provenance.json")
        if any((calib_src / name).is_file() for name in copies):
            cdir = run.path("calibration")
            cdir.mkdir(exist_ok=True)
            for name in copies:
                src = calib_src / name
                if src.is_file():
                    shutil.copy2(src, cdir / name)

        t0 = datetime.now()
        res = evaluator.evaluate(theta, base.scene, record=True)
        wall_s = (datetime.now() - t0).total_seconds()
        dev = cost_terms_max_dev(res)
        if skip_audit:
            audit, bad, audit_wall_s = {}, {}, 0.0
        else:
            ta = datetime.now()
            audit, bad = audit_result(
                prob, res, tol=cfg.tol, state_tol=cfg.state_tol)
            audit_wall_s = (datetime.now() - ta).total_seconds()

        ps = paired_stats(res.path_costs, base.totals)
        ts_list = term_stats(res.step_terms, base.terms, TERM_NAMES)
        beh_pol = behavior_stats(prob, res)
        beh_base = baseline_behavior(prob, base)
        tp = time_profile(res.step_terms, base.terms)
        term_list = terminal_sensitivity(prob, res, base)

        run.write_npz(f"sim_{base.P}paths.npz", **res.arrays)
        # 逐期/十分位 CSV
        with open(run.path("time_profile.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["period", "policy", "baseline", "diff", "cum_diff"])
            for i in range(prob.T):
                w.writerow([i + 1, tp["policy"][i], tp["baseline"][i],
                            tp["diff"][i], tp["cum_diff"][i]])
        with open(run.path("time_profile_deciles.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["decile", "t_first", "t_last", "n_periods",
                        "policy_cost", "baseline_cost", "diff", "cum_diff"])
            cum = 0.0
            for z in tp["deciles"]:
                cum += z["diff"]
                w.writerow([z["decile"], z["t_first"], z["t_last"],
                            z["n_periods"], z["policy_cost"],
                            z["baseline_cost"], z["diff"], cum])
        run.write_json("terminal_sensitivity.json", term_list)

        prov = run.write_provenance(cfg, prob, pol.params(), baseline=base)
        prov = json.loads(Path(prov).read_text(encoding="utf-8"))

        meta = {
            "dir": run.run_id, "kind": "policy",
            "policy_kind": pol.policy_kind, "model": "markov_grade",
            "schema": 3, "n_paths": int(base.P), "T": int(prob.T),
            "G": int(prob.G), "N_R": int(prob.N_R), "L": int(prob.L),
            "state_dim": int(prob.state_dim),
            "lambda0": float(prob.lambda0),
            "dirichlet_s": float(prob.dirichlet_s),
            "p_digits": int(prob.p_digits),
            "fleet_tol": float(prob.fleet_tol),
            "state_round_digits": int(prob.state_round_digits),
            "state_repair_tol": float(prob.state_repair_tol),
            "terms": list(TERM_NAMES), "policy": pol.params(),
            "scene_source": "baseline npz", "scene_hash": base.scene_hash,
            "seed": int(cfg.seed), "chunk_size": int(cfg.chunk_size),
            "baseline": {
                "name": base.name, "result_dir": str(base.result_dir),
                "npz": str(base.npz_path), "mean": base.mean,
                "std": base.std, "ci95": base.ci95,
                "lb": base.meta.get("lb"), "gap_pct": base.meta.get("gap_pct"),
            },
            "policy_mean": float(res.cost), "policy_std": float(res.std),
            "policy_ci95": float(res.ci95),
            "paired": ps,
            "wall_s": float(wall_s), "audit_wall_s": float(audit_wall_s),
            "n_solves": 0,
            "cost_coef": {"C": prob.C, "h": prob.h, "Cs": prob.Cs,
                          "atc": prob.atc, "tc": prob.tc},
            "cost_unit": "万元",
            "terms_vs_stage_cost_max_dev": float(dev),
            "audit_violations": bad,
            "q0_total_mean": float(beh_pol["q0_total"]),
        }
        run.write_json(f"sim_{base.P}paths.json", meta)
        run.write_text(
            f"sim_report_{datetime.now():%Y%m%d_%H%M%S}.txt",
            _sim_report_txt(prob, base, res, ps, ts_list, audit, bad, dev,
                            wall_s, audit_wall_s, f"sim_{base.P}paths.npz"))

        comp = {
            "policy": {**pol.params(), "theta": theta.tolist()},
            "policy_meta": {
                "selected_algorithm": policy_data.get("calibration", {}).get(
                    "selected_algorithm"),
                "selected_start": policy_data.get("calibration", {}).get(
                    "selected_start"),
                "train_cost": policy_data.get("calibration", {}).get(
                    "train_cost"),
                "val_cost": policy_data.get("calibration", {}).get(
                    "val_cost"),
            },
            "baseline": {
                "name": base.name, "result_dir": str(base.result_dir),
                "npz": str(base.npz_path), "mean": base.mean,
                "std": base.std, "ci95": base.ci95,
                "lb": base.meta.get("lb"), "gap_pct": base.meta.get("gap_pct"),
                "problem_data_match": base.problem_data_match,
                "scene_hash": base.scene_hash,
            },
            "meta": {
                "T": int(prob.T), "n_paths": int(base.P),
                "cost_unit": "万元", "config": str(cfg.config_path),
                "baseline_lb": base.meta.get("lb"),
                "baseline_gap_pct": base.meta.get("gap_pct"),
            },
            "paired": ps, "terms": ts_list,
            "behavior_policy": beh_pol, "behavior_baseline": beh_base,
            "time_profile": tp, "terminal_sensitivity": term_list,
            "audit": {
                "violations": bad,
                "max": {k: float(v) for k, v in audit.items()},
                "skipped": bool(skip_audit),
                "baseline_sidecar_violations": base.meta.get(
                    "audit_violations"),
                "baseline_sidecar_dev": base.meta.get(
                    "terms_vs_stage_cost_max_dev"),
            },
            "terms_vs_stage_cost_max_dev": float(dev),
            "provenance": prov,
        }
        run.write_json("comparison.json", comp)
        run.write_text("comparison.md", _summary_md(
            prob, cfg, base, pol, policy_data, res, ps, ts_list,
            beh_pol, beh_base, tp, term_list, audit, bad, dev,
            run.run_id, detailed=True))
        run.write_text("summary.md", _summary_md(
            prob, cfg, base, pol, policy_data, res, ps, ts_list,
            beh_pol, beh_base, tp, term_list, audit, bad, dev,
            run.run_id, detailed=False))

        run.finalize(
            "completed",
            extra={
                "policy_mean": float(res.cost),
                "baseline_mean": float(base.mean),
                "paired_diff_mean": float(ps["diff_mean"]),
                "rel_diff_pct": float(ps["rel_diff_pct"]),
                "win_rate_policy_lower": float(ps["win_rate_policy_lower"]),
                "audit_ok": not bool(bad),
                "summary": "summary.md",
            },
            index_row={
                "run_id": run.run_id, "kind": "eval", "status": "completed",
                "created_at": run.created_at, "config": cfg.config_path.name,
                "T": int(prob.T), "n_paths": int(base.P),
                "algorithm": policy_data.get("calibration", {}).get(
                    "selected_algorithm"),
                "selected_start": policy_data.get("calibration", {}).get(
                    "selected_start"),
                "train_cost": policy_data.get("calibration", {}).get(
                    "train_cost"),
                "val_cost": policy_data.get("calibration", {}).get(
                    "val_cost"),
                "test_cost": float(res.cost),
                "baseline_mean": float(base.mean),
                "diff_mean": float(ps["diff_mean"]),
                "rel_pct": float(ps["rel_diff_pct"]),
                "win_rate": float(ps["win_rate_policy_lower"]),
                "audit_ok": int(not bool(bad)),
                "policy_json": f"{run.run_id}/policy.json",
                "summary_md": f"{run.run_id}/summary.md",
            })
        print(f"评估完成: {run.dir}", flush=True)
        print(f"策略均值 {res.cost:.4f} vs 基线 {base.mean:.4f}；"
              f"配对差 {ps['diff_mean']:+.4f} (t={ps['t']:+.2f})", flush=True)
        if bad:
            print(f"[警告] 约束审计有违规: {bad}", flush=True)
        return run.dir
    except Exception as e:
        run.finalize(
            "failed", extra={"error": repr(e)},
            index_row={
                "run_id": run.run_id, "kind": "eval", "status": "failed",
                "created_at": run.created_at, "config": cfg.config_path.name,
                "T": int(prob.T), "n_paths": int(base.P),
                "policy_json": f"{run.run_id}/policy.json",
                "summary_md": f"{run.run_id}/summary.md",
            })
        raise


def main(project_dir: str | Path | None = None,
         policy_path: str | Path | None = None, *,
         tag: str | None = None, archive_root=None,
         skip_audit: bool = False, policy_config=None) -> Path:
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent
    if policy_path is None:
        raise SystemExit("需要 --policy <calibration_dir>/policy.json")
    return run_evaluation(project_dir, policy_path, tag=tag,
                          archive_root=archive_root,
                          skip_audit=skip_audit,
                          policy_config=policy_config)
