"""训练结束后做几千个回合的独立仿真，报告总成本均值和 95% 置信区间。

- 每回合从 problem_data 的 I0/o0/n0 开始；
- Theta 在线独立抽样（不是固定 300 场景池）；
- 策略用 Gaussian 均值（deterministic）；
- 同时给出基线 3000 条测试路径的均值和 CI 作为参照。

用法::

    P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
    "$P" -B ppo_taufirst/simulate.py \
        --policy results/ppo_taufirst/<train_run>/policy.pt --episodes 3000
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
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

from common.baseline import sha256_file  # noqa: E402
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from common.report import (copy_configs, provenance, write_json,  # noqa: E402
                           write_text)
from ppo_taufirst.env import TauFirstEnv  # noqa: E402
from ppo_taufirst.train import (DEFAULT_PPO_CONFIG, load_ppo_config,  # noqa: E402
                                resolve_compare_config)


def _ci95(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else float("nan")
    ci = 1.959963984540054 * se
    return mean, std, se, ci


def main(argv=None):
    ap = argparse.ArgumentParser(description="Monte Carlo episode simulation")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=None,
                    help="仿真回合数；缺省取 [evaluation].episodes（默认 3000）")
    ap.add_argument("--seed", type=int, default=None,
                    help="Theta 抽样种子；缺省取 [evaluation].sim_seed")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--config", default=None,
                    help="PPO 配置 TOML（默认 ppo_taufirst/config.toml）")
    ap.add_argument("--compare-config", default=None)
    ap.add_argument("--tag", default="ppo_episode_sim")
    ap.add_argument("--archive-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ppo_cfg_path = Path(args.config) if args.config else DEFAULT_PPO_CONFIG
    if not ppo_cfg_path.is_absolute():
        ppo_cfg_path = PROJECT / ppo_cfg_path
    ppo_data = load_ppo_config(ppo_cfg_path)
    compare_path = resolve_compare_config(args.compare_config, ppo_data)
    cfg = load_compare_config(compare_path)
    prob = load_problem(cfg)

    policy_src = Path(args.policy)
    if not policy_src.is_absolute():
        policy_src = PROJECT / policy_src
    if not policy_src.is_file():
        raise FileNotFoundError(f"策略文件不存在: {policy_src}")

    # torch 必须在 load_problem 之后导入（标准库 code 冲突）
    from ppo_taufirst.ppo import deterministic_actions, load_policy
    import torch
    torch.set_num_threads(max(1, int(args.threads)))

    model, meta = load_policy(policy_src, device="cpu")
    if meta.get("kind") != "ppo_taufirst":
        raise ValueError(f"不是 ppo_taufirst 策略: {meta.get('kind')}")
    for k, v in (("T", int(prob.T)), ("N_R", int(prob.N_R)),
                 ("G", int(prob.G)), ("L", int(prob.L))):
        if int(meta.get(k, -1)) != v:
            raise ValueError(f"策略 meta {k}={meta.get(k)} 与模型 {v} 不一致")
    if int(meta.get("obs_dim", -1)) != 35 or int(meta.get("act_dim", -1)) != 10:
        raise ValueError("策略 obs_dim/act_dim 异常")
    pd_hash = sha256_file(cfg.problem_data) if Path(cfg.problem_data).is_file() \
        else None
    if meta.get("problem_data_hash") and pd_hash and \
            meta["problem_data_hash"] != pd_hash:
        raise ValueError("策略训练的 problem_data 与当前配置不一致")

    ev = ppo_data.get("evaluation", {})
    n_episodes = int(args.episodes if args.episodes is not None
                     else ev.get("episodes", 3000))
    seed = int(args.seed if args.seed is not None
               else ev.get("sim_seed", 888))
    if n_episodes <= 0:
        raise ValueError("episodes 必须为正")
    terminal_q_zero = bool(meta.get("terminal_q_zero", True))
    reward_scale = float(meta.get("reward_scale", 1e-3))

    env = TauFirstEnv(prob, reward_scale=reward_scale,
                      terminal_q_zero=terminal_q_zero, seed=seed)
    T = int(prob.T)
    costs = np.zeros(n_episodes, dtype=float)
    terms = np.zeros((n_episodes, 5), dtype=float)
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        env.reset()
        for _ in range(T):
            act = deterministic_actions(model, env.observation()[None, :])[0]
            _, _, done, info = env.step(act)
            if done:
                break
        if not done:
            raise RuntimeError(f"episode {ep} 未在 T 期结束")
        costs[ep] = float(info["episode_cost"])
        terms[ep] = np.asarray(info["terms"], dtype=float)
        if (ep + 1) % 500 == 0 and not args.quiet:
            print(f"[sim] {ep + 1}/{n_episodes} "
                  f"mean={costs[:ep + 1].mean():.2f} "
                  f"wall={time.perf_counter() - t0:.1f}s")
    wall = time.perf_counter() - t0

    mean, std, se, ci = _ci95(costs)
    # 基线 3000 条测试路径（独立样本，作为参照）
    with np.load(cfg.baseline_npz, allow_pickle=False) as z:
        base_totals = np.asarray(z["totals"], dtype=float)
    b_mean, b_std, b_se, b_ci = _ci95(base_totals)
    diff = mean - b_mean
    diff_se = float(np.sqrt(se ** 2 + b_se ** 2))
    diff_ci = 1.959963984540054 * diff_se
    rel = diff / b_mean * 100.0 if b_mean else float("nan")

    if args.out:
        run_dir = Path(args.out)
        if not run_dir.is_absolute():
            run_dir = PROJECT / run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"结果目录非空，拒绝覆盖: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 仿真结果放在训练结果目录里
        parent = policy_src.parent
        run_dir = parent / "simulation"
        i = 2
        while run_dir.exists():
            run_dir = parent / f"simulation_{i}"
            i += 1
        run_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now().isoformat(timespec="seconds")
    shutil.copy2(policy_src, run_dir / "policy.pt")
    copy_configs(run_dir, cfg, PROJECT, ppo_cfg_path)
    # 训练收敛图随仿真结果归档
    train_dir = policy_src.parent
    curve_dst = run_dir / "train_curve.png"
    train_curve = None
    curve_src = train_dir / "train_curve.png"
    if curve_src.is_file():
        shutil.copy2(curve_src, curve_dst)
        train_curve = str(curve_dst)
    elif (train_dir / "episode_log.csv").is_file():
        cmd = [
            sys.executable, "-B",
            str(Path(__file__).resolve().parent / "plot_train.py"),
            "--run", str(train_dir), "--out", str(curve_dst),
            "--baseline-mean", f"{b_mean:.6f}",
        ]
        try:
            res = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True,
                                 text=True, timeout=120, check=False)
            if res.returncode == 0 and curve_dst.is_file():
                train_curve = str(curve_dst)
            else:
                print(f"[警告] 收敛图生成失败: {res.stderr.strip()[:300]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 收敛图生成失败: {exc}")
    with (run_dir / "episode_costs.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "total_cost"])
        for i, c in enumerate(costs, start=1):
            w.writerow([i, f"{c:.6f}"])
    result = {
        "policy": str(policy_src),
        "episodes": int(n_episodes),
        "seed": seed,
        "deterministic": True,
        "T": T,
        "ppo": {
            "mean": mean, "std": std, "se": se,
            "ci95_low": mean - ci, "ci95_high": mean + ci,
        },
        "baseline": {
            "n": int(base_totals.size),
            "mean": b_mean, "std": b_std, "se": b_se,
            "ci95_low": b_mean - b_ci, "ci95_high": b_mean + b_ci,
            "npz": str(cfg.baseline_npz),
        },
        "diff": {
            "mean": diff, "se": diff_se,
            "ci95_low": diff - diff_ci, "ci95_high": diff + diff_ci,
            "rel_pct": rel,
        },
        "train_curve": train_curve,
        "wall_s": float(wall),
        "created_at": created_at,
        "policy_meta": meta,
    }
    write_json(run_dir / "episode_eval.json", result)
    write_text(run_dir / "summary.md", "\n".join([
        "# PPO 回合仿真（均值 + 95% CI）",
        "",
        f"- 策略：`{policy_src}`",
        f"- 回合数：{n_episodes}；Theta seed={seed}；确定性动作；T={T}",
        f"- PPO 均值：**{mean:.4f}** 万元/路径；"
        f"95% CI [{mean - ci:.4f}, {mean + ci:.4f}]；"
        f"std={std:.4f}，SE={se:.4f}",
        f"- 基线（{base_totals.size} 条）：{b_mean:.4f}；"
        f"95% CI [{b_mean - b_ci:.4f}, {b_mean + b_ci:.4f}]",
        f"- PPO−基线：**{diff:+.4f}（{rel:+.2f}%）**；"
        f"95% CI [{diff - diff_ci:+.4f}, {diff + diff_ci:+.4f}]",
        f"- 训练收敛图：`{train_curve}`"
        if train_curve else "- 训练收敛图：未找到/未生成",
        f"- 仿真墙钟：{wall:.1f}s",
        "",
        "逐回合总成本见 `episode_costs.csv`。",
        "",
    ]))
    run = {
        "status": "completed",
        "created_at": created_at,
        "policy": str(policy_src),
        "episodes": int(n_episodes),
        "seed": seed,
        "ppo_mean": mean,
        "ppo_ci95_low": mean - ci,
        "ppo_ci95_high": mean + ci,
        "baseline_mean": b_mean,
        "diff_mean": diff,
        "rel_diff_pct": rel,
        "train_curve": train_curve,
        "wall_s": float(wall),
    }
    write_json(run_dir / "run.json", run)
    write_json(run_dir / "provenance.json",
               provenance(cfg, prob, meta))
    if not args.quiet:
        print(f"[sim] result_dir={run_dir}")
        print(f"[sim] PPO mean={mean:.4f} CI=[{mean - ci:.4f}, "
              f"{mean + ci:.4f}] | baseline={b_mean:.4f} | "
              f"diff={diff:+.4f} ({rel:+.2f}%) wall={wall:.1f}s")
    return run_dir


if __name__ == "__main__":
    main()
