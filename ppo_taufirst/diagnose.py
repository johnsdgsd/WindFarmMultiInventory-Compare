"""环境 sanity check：在 TauFirstEnv 中跑几个简单 base-stock 规则。

目的：区分“PPO 欠训练/策略差”与“环境/成本实现有 bug”。
若 S=Imax 这类简单规则在本环境中能接近基线，则环境口径基本可信，
PPO 的大 gap 主要来自策略本身（欠训练、state-only、固定被动规则）。

用法::

    P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
    "$P" -B ppo_taufirst/diagnose.py --m-paths 50
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
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

from common.baseline import (load_baseline, paired_stats,  # noqa: E402
                             sha256_array, term_stats)
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from common.report import write_json, write_text  # noqa: E402
from ppo_taufirst.archive import append_index, make_run_dir  # noqa: E402
from ppo_taufirst.env import (TERM_NAMES, TauFirstEnv,  # noqa: E402
                              audit_rollout)
from ppo_taufirst.evaluate import (behavior_from_base,  # noqa: E402
                                   behavior_from_ppo)
from ppo_taufirst.train import (DEFAULT_PPO_CONFIG, load_ppo_config,  # noqa: E402
                                resolve_compare_config)

S_NV = np.array([8.86, 6.75, 11.40, 8.49, 8.24])


def _rollout(prob, S, theta_paths, m: int):
    """固定 base-stock 目标 S（每仓）的 rollout；tau=0。"""
    T, NR, G = int(prob.T), int(prob.N_R), int(prob.G)
    envs = [TauFirstEnv(prob, theta_path=theta_paths[p],
                        terminal_q_zero=True) for p in range(m)]
    arrays = {
        "I_in": np.zeros((m, T, NR)), "o_in": np.zeros((m, T, 2, NR)),
        "n_in": np.zeros((m, T, NR, G)), "q": np.zeros((m, T, NR)),
        "tau": np.zeros((m, T, NR)), "taup": np.zeros((m, T, NR)),
        "taum": np.zeros((m, T, NR)), "y": np.zeros((m, T, NR, NR)),
        "I_out": np.zeros((m, T, NR)), "o_next": np.zeros((m, T, 2, NR)),
        "n_out": np.zeros((m, T, NR, G)), "F": np.zeros((m, T, NR)),
        "W_in": np.zeros((m, T, NR)), "W_out": np.zeros((m, T, NR)),
        "cost": np.zeros((m, T)), "terms": np.zeros((m, T, 5)),
        "reward": np.zeros((m, T)),
    }
    S = np.asarray(S, dtype=float).reshape(NR)
    for t in range(T):
        for p, e in enumerate(envs):
            I0, o0, n0 = e.I.copy(), np.stack([e.o1, e.o2]), e.n.copy()
            W = e.N - e.n.sum(axis=1)
            pos = e.I + e.o1 + e.o2 - W
            qub = np.minimum(e.Qmax, e.Imax - e.I - e.o1 - e.o2)
            q = np.clip(S - pos, 0.0, np.maximum(qub, 0.0))
            action = np.concatenate([np.zeros(e.N_R), q / e.Qmax])
            _, reward, done, info = e.step(action)
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
            arrays["cost"][p, t] = info["cost"]
            arrays["terms"][p, t] = info["terms"]
            arrays["reward"][p, t] = reward
            if t == T - 1 and not done:
                raise RuntimeError("episode 未在末期结束")
    arrays["totals"] = arrays["cost"].sum(axis=1)
    arrays["term_totals"] = arrays["terms"].sum(axis=1)
    return envs, arrays


def _rollout_with_scene(prob, S, theta_paths, scene, m):
    """与 _rollout 相同，但把 scene 一并记录，便于 audit。"""
    envs, arrays = _rollout(prob, S, theta_paths, m)
    arrays["scene"] = np.asarray(scene, dtype=np.int16).copy()
    return arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description="base-stock sanity check")
    ap.add_argument("--config", default=None,
                    help="PPO 配置 TOML（用于默认 compare_config / M）")
    ap.add_argument("--compare-config", default=None)
    ap.add_argument("--m-paths", type=int, default=None)
    ap.add_argument("--tag", default="ppo_basestock_diagnose")
    ap.add_argument("--archive-root", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ppo_cfg_path = Path(args.config) if args.config else DEFAULT_PPO_CONFIG
    if not ppo_cfg_path.is_absolute():
        ppo_cfg_path = PROJECT / ppo_cfg_path
    ppo_data = load_ppo_config(ppo_cfg_path)
    compare_path = resolve_compare_config(args.compare_config, ppo_data)
    cfg = load_compare_config(compare_path)
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)
    M = int(args.m_paths if args.m_paths is not None
            else ppo_data["evaluation"]["m_paths"])
    if M < 1 or M > base.P:
        raise ValueError(f"m_paths={M} 必须在 [1,{base.P}]")
    scene = np.ascontiguousarray(base.scene[:M])
    theta_paths = np.stack([
        np.stack([np.asarray(prob.pool[int(s)][1], dtype=float)
                  for s in scene[p]], axis=0) for p in range(M)], axis=0)

    policies = [
        ("S_zero", np.zeros(prob.N_R)),
        ("S_Qmax", np.asarray(prob.Qmax, dtype=float)),
        ("S_half_Imax", 0.5 * np.asarray(prob.Imax, dtype=float)),
        ("S_NV", S_NV.copy()),
        ("S_Imax", np.asarray(prob.Imax, dtype=float)),
    ]
    rows = []
    details = []
    for name, S in policies:
        arrays = _rollout_with_scene(prob, S, theta_paths, scene, M)
        audit, bad = audit_rollout(
            prob, arrays, tol=float(cfg.tol), state_tol=float(cfg.state_tol),
            reward_scale=1e-3)
        ps = paired_stats(arrays["totals"], base.totals[:M])
        ts = term_stats(arrays["terms"], base.terms[:M], TERM_NAMES)
        beh = behavior_from_ppo(prob, arrays)
        rows.append({
            "policy": name, "S": S.tolist(), "paired": ps, "terms": ts,
            "behavior": beh, "audit_passed": len(bad) == 0, "bad": bad,
        })
        details.append({
            "policy": name, "S": S.tolist(),
            "mean": ps["policy_mean"], "diff": ps["diff_mean"],
            "rel": ps["rel_diff_pct"], "t": ps["t"],
            "terms": {z["term"]: z["diff_mean"] for z in ts},
            "mean_inventory": beh["mean_inventory"],
            "fill_rate": beh["fill_rate"],
            "audit_passed": len(bad) == 0,
        })
        if not args.quiet:
            print(f"{name:12s} S={np.round(S, 2).tolist()} "
                  f"mean={ps['policy_mean']:.2f} "
                  f"diff={ps['diff_mean']:+.2f} "
                  f"({ps['rel_diff_pct']:+.2f}%) "
                  f"t={ps['t']:.2f} "
                  f"I={beh['mean_inventory']:.2f} "
                  f"fill={beh['fill_rate']:.4f} "
                  f"audit={'PASS' if len(bad) == 0 else 'FAIL'}")

    beh_base = behavior_from_base(prob, base, M)
    run_dir = make_run_dir(
        args.tag, args.archive_root or ppo_data["run"].get(
            "archive_root", "results"))
    created_at = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "diagnose.json", {
        "created_at": created_at, "M": M,
        "scene_hash": sha256_array(scene),
        "baseline_subset_mean": float(base.totals[:M].mean()),
        "baseline_behavior": beh_base,
        "policies": rows,
    })
    with (run_dir / "basestock.csv").open("w", newline="",
                                          encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "S", "mean", "diff", "rel_pct", "t",
                    "proc_diff", "hold_diff", "down_diff", "active_diff",
                    "passive_diff", "mean_inventory", "fill_rate",
                    "audit_passed"])
        for r in rows:
            S = r["S"]
            td = {z["term"]: z["diff_mean"] for z in r["terms"]}
            w.writerow([r["policy"],
                        "/".join(f"{x:g}" for x in S),
                        f"{r['paired']['policy_mean']:.4f}",
                        f"{r['paired']['diff_mean']:.4f}",
                        f"{r['paired']['rel_diff_pct']:.4f}",
                        f"{r['paired']['t']:.4f}",
                        f"{td['采购']:.4f}", f"{td['持有']:.4f}",
                        f"{td['停机']:.4f}", f"{td['主动调拨']:.4f}",
                        f"{td['被动调拨']:.4f}",
                        f"{r['behavior']['mean_inventory']:.4f}",
                        f"{r['behavior']['fill_rate']:.4f}",
                        r["audit_passed"]])
    lines = [
        "# Base-stock sanity check（TauFirstEnv）",
        "",
        f"- 结果目录：`{run_dir}`",
        f"- M={M}，基线子集均值 {base.totals[:M].mean():.4f} 万元/路径。",
        "- 规则：`q = clip(S - (I+o1+o2-W_in), 0, min(Qmax, Imax-I-Σo))`，"
        "`tau=0`；被动调拨按 `prob.priority`/`tc_route` 的优先级规则（本地优先 + 跨仓全局贪心）。",
        "- 目的：如果简单规则能接近基线，说明环境/成本/约束口径基本可信；"
        "否则先修环境，再谈 PPO。",
        "",
        "| 策略 | S | 均值 | 配对差 | 相对差 | t | 平均库存 | 履约率 | 审计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {'/'.join(f'{x:g}' for x in r['S'])} | "
            f"{r['paired']['policy_mean']:.2f} | "
            f"{r['paired']['diff_mean']:+.2f} | "
            f"{r['paired']['rel_diff_pct']:+.2f}% | "
            f"{r['paired']['t']:.2f} | "
            f"{r['behavior']['mean_inventory']:.2f} | "
            f"{r['behavior']['fill_rate']:.4f} | "
            f"{'PASS' if r['audit_passed'] else 'FAIL'} |")
    lines += [
        "",
        f"基线子集行为：平均库存 {beh_base['mean_inventory']:.2f}，"
        f"履约率 {beh_base['fill_rate']:.4f}，主动调拨 "
        f"{beh_base['active_transfer_volume']:.2f}。",
        "",
        "结论：若 `S_Imax` 接近/优于基线，则 env 的转移、成本和约束实现"
        "可作为 PPO 的可靠底座；PPO 的大 gap 主要来自欠训练/策略类。",
        "",
    ]
    write_text(run_dir / "summary.md", "\n".join(lines))
    append_index(run_dir, {
        "run_id": run_dir.name, "kind": "diagnose", "status": "completed",
        "T": int(prob.T), "M": M, "created_at": created_at,
        "policy": str(run_dir),
    })
    if not args.quiet:
        print(f"[diagnose] result_dir={run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
