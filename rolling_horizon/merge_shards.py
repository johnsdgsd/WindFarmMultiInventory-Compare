"""Merge contiguous rolling-horizon shards into one full-test report.

The shards must cover the baseline test paths exactly once, in order
(0..M-1).  Each shard is a completed `run_evaluate.py` archive with
`sim_<m>paths.npz`; the merged directory reuses the same metrics code.

Usage:
  "$P" -B rolling_horizon/merge_shards.py \
    --dirs "results/.../shard0_ts,results/.../shard1_ts,..." \
    --tag tau_W20_M3000_full_merged
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.baseline import load_baseline, paired_stats, term_stats  # noqa: E402
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from common.report import (copy_configs, fmt, make_result_dir, write_json,
                           write_text)  # noqa: E402
from rolling_horizon.metrics import (baseline_behavior, behavior_stats,
                                     terminal_sensitivity, time_profile)  # noqa: E402
from rolling_horizon.simulator import TERM_NAMES, cost_terms_max_dev  # noqa: E402

DEFAULT_CFG = ROOT / "config" / "compare_T60_base204105.toml"


def _table(headers, rows, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(aligns) + "|"]
    out.extend("| " + " | ".join(str(v) for v in row) + " |"
               for row in rows)
    return "\n".join(out)


def _read_shard(d: Path):
    d = Path(d)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "completed":
        raise ValueError(f"分片未完成 status={run.get('status')}: {d}")
    ck = json.loads((d / "checkpoint.json").read_text(encoding="utf-8"))
    idx = np.asarray(ck["path_indices"], dtype=int)
    npzs = sorted(d.glob("sim_*paths.npz"))
    if len(npzs) != 1:
        raise ValueError(f"分片应有且仅有一个 sim_*paths.npz: {d}")
    z = np.load(npzs[0], allow_pickle=False)
    arrays = {k: np.asarray(z[k]) for k in z.files}
    z.close()
    audit = json.loads((d / "audit.json").read_text(encoding="utf-8"))
    lp_path = d / "lp_stats.csv"
    lp_counts = None
    if lp_path.is_file():
        n_two = n_rec = 0
        build_s = two_solve = rec_solve = 0.0
        with open(lp_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["status"] == "ok":
                    n_two += 1
                    build_s += float(row["build_s"])
                    two_solve += float(row["solve_s"])
                elif row["status"] == "recourse":
                    n_rec += 1
                    rec_solve += float(row["solve_s"])
        lp_counts = {
            "n_two_stage": n_two, "n_recourse": n_rec,
            "two_stage_build_s_total": build_s,
            "two_stage_solve_s_total": two_solve,
            "recourse_solve_s_total": rec_solve,
        }
    return {
        "dir": d, "run": run, "indices": idx, "arrays": arrays,
        "audit": audit, "lp_csv": lp_path, "lp_counts": lp_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--policy-config", default=None)
    ap.add_argument("--dirs", required=True,
                    help="逗号分隔的分片结果目录，按路径顺序给出")
    ap.add_argument("--tag", default="tau_W20_M3000_full_merged")
    ap.add_argument("--archive-root", default=None)
    ap.add_argument("--no-sim-npz", action="store_true")
    args = ap.parse_args()

    cfg = load_compare_config(Path(args.config))
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)
    dirs = [Path(x) for x in args.dirs.split(",") if x.strip()]
    shards = [_read_shard(d) for d in dirs]
    for s in shards[1:]:
        for k in ("N", "seed_opt", "window_len", "pool_hash"):
            if s["run"].get(k) != shards[0]["run"].get(k):
                raise ValueError(
                    f"分片 {k} 不一致: {s['dir']} "
                    f"{s['run'].get(k)} vs {shards[0]['run'].get(k)}")
        if int(s["run"].get("M", -1)) != int(s["indices"].size):
            raise ValueError(f"分片 M 与路径数不一致: {s['dir']}")

    indices = np.concatenate([s["indices"] for s in shards])
    if not np.array_equal(indices, np.arange(base.P, dtype=int)):
        raise ValueError(
            "分片路径没有恰好覆盖 0..P-1；"
            f"got {indices.min()}..{indices.max()}, n={indices.size}, P={base.P}")
    M = int(base.P)
    # shard order must be increasing path order
    order = np.argsort([int(s["indices"].min()) for s in shards])
    shards = [shards[i] for i in order]

    # A resumed shard's sim npz may carry checkpoint-only extras
    # (I/o/n/term_totals); use the common array schema and verify the
    # metrics fields are present everywhere.
    keys = set.intersection(*(set(s["arrays"]) for s in shards))
    required = {
        "scene", "totals", "terms", "cost", "x_in", "I_in", "o_in",
        "n_in", "q", "taup", "taum", "S_end", "I_out", "o_next", "y",
        "n_out", "F", "W_in", "W_out", "theta",
    }
    missing = required - keys
    if missing:
        raise ValueError(f"合并所需数组缺失: {sorted(missing)}")
    arrays = {k: np.concatenate([s["arrays"][k] for s in shards], axis=0)
              for k in sorted(keys)}
    totals = np.asarray(arrays["totals"], dtype=float)
    terms = np.asarray(arrays["terms"], dtype=float)
    if totals.size != M or terms.shape[0] != M:
        raise ValueError(f"合并后路径数 {totals.size}/{terms.shape[0]} != {M}")
    if not np.array_equal(arrays["scene"], base.scene):
        raise ValueError("合并后的 scene 与基线 scene 不一致（顺序/内容）")

    res = SimpleNamespace(
        arrays=arrays, totals=totals,
        term_totals=terms.sum(axis=1),
        q0=np.zeros((M, prob.N_R), dtype=float),
        T=int(prob.T), P=M, policy_params={}, meta={},
        mean=float(np.mean(totals)),
    )
    ps = paired_stats(totals, base.totals)
    tstats = term_stats(terms, base.terms, names=TERM_NAMES)
    pb = behavior_stats(prob, res)
    bb = baseline_behavior(prob, base)
    tp = time_profile(terms, base.terms)
    ts = terminal_sensitivity(prob, res, base)
    cost_dev = cost_terms_max_dev(res)

    # audit: max over shards; fail if any shard flagged a check
    checks = {}
    bad = {}
    for s in shards:
        for k, v in s["audit"].get("checks", {}).items():
            checks[k] = max(float(v), float(checks.get(k, -np.inf)))
        bad.update(s["audit"].get("bad", {}))
    tol = float(shards[0]["audit"].get("tol", cfg.tol))
    state_tol = float(shards[0]["audit"].get("state_tol", cfg.state_tol))

    # LP stats: sum counts/times; max sizes from the first shard
    lp0 = shards[0]["run"].get("lp", {})
    lp = dict(lp0)
    for k in ("n_two_stage", "n_recourse", "two_stage_build_s_total",
              "two_stage_solve_s_total", "recourse_solve_s_total"):
        vals = []
        for s in shards:
            c = s["lp_counts"] or s["run"].get("lp", {})
            vals.append(c.get(k, 0))
        lp[k] = int(sum(vals)) if k.startswith("n_") else float(sum(vals))
    lp["n_failures"] = int(sum(s["run"].get("lp", {}).get("n_failures", 0)
                                for s in shards))
    lp["by_t"] = []
    lp["by_t_note"] = "分片并行；逐 t 的 LP 统计见各分片 lp_stats.csv"
    wall_s = float(max(s["run"].get("wall_s", 0.0) for s in shards))
    wall_sum = float(sum(s["run"].get("wall_s", 0.0) for s in shards))
    pool_hash = shards[0]["run"].get("pool_hash")
    seed_opt = int(shards[0]["run"].get("seed_opt", 2027))
    N = int(shards[0]["run"].get("N", 10))
    window_len = int(shards[0]["run"].get("window_len", 20))
    t1_obj = lp0.get("t1_objective_mean", float("nan"))
    saa = {
        "t1_objective_mean": float(t1_obj),
        "out_of_sample_mean": float(ps["policy_mean"]),
        "note": "分片共享优化池；t=1 目标取自分片 0，仅作 SAA 诊断。",
    }

    if ps["diff_mean"] < 0:
        verdict = (
            f"全量 {M} 条测试路径配对：滚动均值 {ps['policy_mean']:.4f} "
            f"比 SDDP 基线 {ps['baseline_mean']:.4f} 低 "
            f"{-ps['diff_mean']:.4f} 万元/路径"
            f"（{ps['rel_diff_pct']:.2f}%），t={ps['t']:.2f}，"
            f"胜率 {100 * ps['win_rate_policy_lower']:.2f}%。")
    else:
        verdict = (
            f"全量 {M} 条测试路径配对：滚动均值 {ps['policy_mean']:.4f} "
            f"比 SDDP 基线 {ps['baseline_mean']:.4f} 高 "
            f"{ps['diff_mean']:.4f} 万元/路径"
            f"（{ps['rel_diff_pct']:.2f}%），t={ps['t']:.2f}，"
            f"胜率 {100 * ps['win_rate_policy_lower']:.2f}%。")
    conclusion_text = "\n".join([
        verdict,
        "",
        "这是 3000 条基线测试路径的完整配对评估；分片只是并行实现，"
        "每片使用同一优化池、同一策略和同一 W，合并后逐路径配对。",
        "N=10、W=20、M=3000；分片数为 "
        f"{len(shards)}，每片路径数见上表。审计按分片逐 (路径,期) 检查，"
        "合并后取各检查的最大值。",
        "不要把这个结果与 M=50 随机子集或 PPO 的独立回合混为同一配对协议。",
    ])

    tag = args.tag
    out = Path(make_result_dir(
        args.archive_root if args.archive_root is not None else ROOT,
        "rolling_horizon", tag))
    policy_path = (Path(args.policy_config) if args.policy_config is not None
                   else ROOT / "rolling_horizon" / "config" / "policy.toml")
    copy_configs(out, cfg, ROOT, policy_path=policy_path)

    created = datetime.now().isoformat(timespec="seconds")
    ctx = {
        "created_at": created, "M": M, "N": N, "seed_opt": seed_opt,
        "window_len": window_len, "ps": ps, "terms": tstats,
        "pb": pb, "bb": bb, "tp": tp, "ts": ts, "lp": lp,
        "audit": checks, "bad": bad, "cost_dev": cost_dev,
        "wall_s": wall_s, "wall_sum": wall_sum, "shards": shards,
        "pool_hash": pool_hash, "conclusion_text": conclusion_text,
        "cost_identity_dev": cost_dev,
    }

    rows = [[s["dir"].name, s["indices"].min(), s["indices"].max(),
             fmt(np.mean(s["arrays"]["totals"])),
             fmt(np.mean(base.totals[s["indices"]])),
             f"{np.mean(s['arrays']['totals']) - np.mean(base.totals[s['indices']]):+.4f}",
             f"{s['run'].get('wall_s', 0.0):.1f}"]
            for s in shards]
    lines = [
        "# 滚动时域 — 全量 3000 路径分片合并评估",
        "",
        f"- 结果目录：`{out}`",
        f"- 创建时间：{created}",
        f"- 基线：{base.name}，全量均值 {fmt(base.mean)}，路径数 {base.P}",
        f"- 合并协议：分片并行；每片 N={N}、W={window_len}、"
        f"seed_opt={seed_opt}；pool_hash=`{pool_hash}`",
        "- 路径覆盖：0..2999 恰好一次，按原顺序拼接；不重采样、不重新生成测试路径。",
        "",
        "## 1. 分片",
        "",
        _table(["分片目录", "路径起", "路径止", "滚动均值", "基线均值",
                "差值", "墙钟(s)"], rows),
        f"- 墙钟：各片并行，最长 {wall_s:.1f} s；各片墙钟之和 "
        f"{wall_sum:.1f} s。",
        "",
        "## 2. 主结果（逐路径配对）",
        "",
        _table(["指标", "滚动", "SDDP 基线", "配对差", "相对差", "95% CI",
                "t", "胜率"],
               [["均值（万元/路径）", fmt(ps["policy_mean"]),
                 fmt(ps["baseline_mean"]), f"{ps['diff_mean']:+.4f}",
                 f"{ps['rel_diff_pct']:+.2f}%",
                 f"[{ps['ci95_low']:+.4f}, {ps['ci95_high']:+.4f}]",
                 fmt(ps["t"]),
                 f"{100 * ps['win_rate_policy_lower']:.2f}%"]]),
        "",
        "## 3. 五项成本",
        "",
        _table(["成本项", "滚动", "基线", "差", "t", "95% CI"],
               [[t["term"], fmt(t["policy_mean"]), fmt(t["baseline_mean"]),
                 f"{t['diff_mean']:+.4f}", fmt(t["t"]),
                 f"[{t['ci95_low']:+.4f}, {t['ci95_high']:+.4f}]"]
                for t in tstats]),
        "",
        "## 4. 行为（Σ仓）",
        "",
        _table(["指标", "滚动", "基线"],
               [["平均库存", fmt(pb["mean_inventory"]),
                 fmt(bb["mean_inventory"])],
                ["履约率", fmt(pb["fill_rate"]), fmt(bb["fill_rate"])],
                ["平均待修", fmt(pb["mean_backlog"]),
                 fmt(bb["mean_backlog"])],
                ["期末待修", fmt(pb["final_backlog"]),
                 fmt(bb["final_backlog"])],
                ["主动调拨量", fmt(pb["active_transfer_volume"]),
                 fmt(bb["active_transfer_volume"])],
                ["总订货量", fmt(pb["total_orders_placed"]),
                 fmt(bb["total_orders_placed"])]]),
        "",
        "## 5. 时间剖面",
        "",
        f"- 首 10% 期 {tp['first_10pct_diff']:+.4f}；中间段 "
        f"{tp['middle_diff']:+.4f}；末 10% 期 "
        f"{tp['last_10pct_diff']:+.4f}。",
        f"- 最差十分位 {tp['worst_decile']['decile']} "
        f"(t={tp['worst_decile']['t_first']}-{tp['worst_decile']['t_last']}, "
        f"{tp['worst_decile']['diff']:+.4f})；最好十分位 "
        f"{tp['best_decile']['decile']} ({tp['best_decile']['diff']:+.4f})。",
        "",
        "## 6. 审计与计算",
        "",
        f"- 审计：{'通过' if not bad else '失败'}（tol={tol:g}, "
        f"state_tol={state_tol:g}）；五项和与总成本最大偏差 "
        f"{cost_dev:.3e}。",
        f"- 两阶段 LP {lp['n_two_stage']} 次，追索 LP "
        f"{lp['n_recourse']} 次，失败 {lp['n_failures']} 次；"
        f"两阶段 solve 合计 {lp['two_stage_solve_s_total']:.1f} s。",
        f"- 每片 workers/threads 与 checkpoint 见各自 run.json；"
        "分片并行，故总墙钟取最长分片。",
        "",
        "## 7. 结论",
        "",
        conclusion_text,
        "",
    ]
    summary = "\n".join(lines)

    write_text(out / "summary.md", summary)
    write_text(out / "conclusion.md", "# 结论\n\n" + conclusion_text + "\n")
    write_text(out / "comparison.md", summary)
    write_json(out / "comparison.json", {
        "headline": {k: ps[k] for k in (
            "policy_mean", "baseline_mean", "diff_mean", "rel_diff_pct",
            "t", "win_rate_policy_lower")},
        "ci95": [ps["ci95_low"], ps["ci95_high"]],
        "terms": tstats, "behavior": {"policy": pb, "baseline": bb},
        "time_profile": tp, "terminal_sensitivity": ts,
        "audit": {"checks": checks, "bad": bad,
                  "cost_identity_dev": cost_dev,
                  "tol": tol, "state_tol": state_tol},
        "lp": lp, "saa": saa, "wall_s": wall_s, "wall_sum": wall_sum,
        "shards": [{"dir": str(s["dir"]),
                    "i0": int(s["indices"].min()),
                    "i1": int(s["indices"].max()),
                    "wall_s": s["run"].get("wall_s")} for s in shards],
        "protocol": {"N": N, "M": M, "seed_opt": seed_opt,
                     "pool_hash": pool_hash, "window_len": window_len,
                     "path_coverage": "0..P-1 exactly once"},
        "created_at": created,
    })
    write_json(out / "audit.json", {
        "checks": checks, "bad": bad,
        "cost_identity_dev": cost_dev,
        "tol": tol, "state_tol": state_tol,
    })
    write_json(out / "run.json", {
        "status": "completed",
        "created_at": created,
        "tag": out.name,
        "config": str(cfg.config_path),
        "N": N, "M": M, "seed_opt": seed_opt,
        "window_len": window_len, "pool_hash": pool_hash,
        "path_sample": "first", "path_offset": 0,
        "shard_count": len(shards),
        "shards": [str(s["dir"]) for s in shards],
        "policy_mean": ps["policy_mean"],
        "baseline_full_mean": float(base.mean),
        "diff_mean": ps["diff_mean"],
        "rel_diff_pct": ps["rel_diff_pct"],
        "t": ps["t"], "win_rate": ps["win_rate_policy_lower"],
        "audit_pass": not bool(bad),
        "wall_s": wall_s, "wall_sum": wall_sum, "lp": lp,
    })
    # policy/provenance copied from the first shard, with merge metadata
    for name in ("policy.json", "provenance.json"):
        src = shards[0]["dir"] / name
        if src.is_file():
            obj = json.loads(src.read_text(encoding="utf-8"))
            if name == "policy.json":
                obj.setdefault("protocol", {})["M"] = M
                obj["protocol"]["real_paths"] = M
                obj["protocol"]["n_scenarios"] = N
                obj["protocol"]["merged_shards"] = len(shards)
                obj["protocol"]["path_coverage"] = "0..P-1 exactly once"
            else:
                obj.setdefault("rolling", {})["M"] = M
                obj["rolling"]["merged_shards"] = len(shards)
            write_json(out / name, obj)
    # concatenate LP stats
    out_lp = out / "lp_stats.csv"
    with open(out_lp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "t", "window", "n_scen", "n_vars", "n_eq",
                    "n_le", "build_s", "solve_s", "objective", "solver",
                    "status"])
        for s in shards:
            if not s["lp_csv"].is_file():
                continue
            off = int(s["indices"].min())
            with open(s["lp_csv"], newline="", encoding="utf-8") as fh:
                rd = csv.reader(fh)
                next(rd, None)
                for row in rd:
                    row[0] = str(int(row[0]) + off)
                    w.writerow(row)
    if not args.no_sim_npz:
        np.savez_compressed(out / f"sim_{M}paths.npz", **arrays)

    # append-only merged index
    index = out.parent / "index.csv"
    new = not index.is_file()
    with open(index, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["created_at", "tag", "N", "M", "seed_opt",
                        "policy_mean", "baseline_subset_mean",
                        "baseline_full_mean", "diff_mean", "rel_diff_pct",
                        "t", "win_rate", "audit_pass", "solver",
                        "wall_s", "result_dir"])
        w.writerow([created, out.name, N, M, seed_opt,
                    f"{ps['policy_mean']:.6f}", f"{ps['baseline_mean']:.6f}",
                    f"{float(base.mean):.6f}", f"{ps['diff_mean']:.6f}",
                    f"{ps['rel_diff_pct']:.6f}", f"{ps['t']:.6f}",
                    f"{ps['win_rate_policy_lower']:.6f}",
                    int(not bool(bad)), "copt-merged",
                    f"{wall_s:.3f}", str(out)])

    print(f"合并结果目录: {out}")
    print(f"滚动 {ps['policy_mean']:.4f} vs 基线 {ps['baseline_mean']:.4f}；"
          f"配对差 {ps['diff_mean']:+.4f} ({ps['rel_diff_pct']:+.2f}%)；"
          f"t={ps['t']:.2f}；审计 {'通过' if not bad else '失败'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
