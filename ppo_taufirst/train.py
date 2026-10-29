"""训练 ppo_taufirst/PPO（新时序 tau_t -> q_t -> Theta_t -> y_t）。

一轮完整回合 = 一次迭代；只需设置回合数。

示例::

    P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
    "$P" -B ppo_taufirst/train.py --episodes 3000 --seed 2027 \
        --tag ppo_taufirst_T60_204105 --threads 4
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
import subprocess
import time
import tomllib
import traceback
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
from ppo_taufirst.archive import append_index, make_run_dir  # noqa: E402
from ppo_taufirst.env import TauFirstEnv  # noqa: E402

DEFAULT_PPO_CONFIG = Path(__file__).resolve().parent / "config.toml"


def load_ppo_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_PPO_CONFIG
    if not p.is_absolute():
        p = PROJECT / p
    if not p.is_file():
        raise FileNotFoundError(f"PPO 配置不存在: {p}")
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    for sec in ("run", "ppo", "evaluation"):
        if sec not in data or not isinstance(data[sec], dict):
            raise KeyError(f"PPO 配置缺少 [{sec}]: {p}")
    return data


def resolve_compare_config(cli: str | None, ppo_data: dict) -> str | None:
    if cli:
        return cli
    env = os.environ.get("COMPARE_CONFIG")
    if env:
        return env
    return str(ppo_data["run"]["compare_config"])


def _write_csv(path: Path, history: list[dict]) -> None:
    if not history:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in history:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, "") for k in keys})


def write_train_log(path: Path, history: list[dict]) -> None:
    _write_csv(path, history)


def write_episode_log(path: Path, history: list[dict]) -> None:
    _write_csv(path, history)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO tau-first training")
    ap.add_argument("--episodes", type=int, default=None,
                    help="训练回合数（一轮完整回合 = 一次迭代）；"
                         "缺省取 config [run].episodes")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--config", type=str, default=None,
                    help="PPO 配置 TOML（默认 ppo_taufirst/config.toml）")
    ap.add_argument("--compare-config", type=str, default=None,
                    help="共享 compare TOML；缺省读 COMPARE_CONFIG 或 PPO 配置")
    ap.add_argument("--archive-root", type=str, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="显式结果目录（必须不存在或为空）")
    ap.add_argument("--reward-scale", type=float, default=None)
    ap.add_argument("--allow-terminal-q", action="store_true",
                    help="末期仍允许 q_T>0（默认 q_T=0，与 rolling/SDDP 口径一致）")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    ppo_cfg_path = Path(args.config) if args.config else DEFAULT_PPO_CONFIG
    if not ppo_cfg_path.is_absolute():
        ppo_cfg_path = PROJECT / ppo_cfg_path
    ppo_data = load_ppo_config(ppo_cfg_path)

    compare_path = resolve_compare_config(args.compare_config, ppo_data)
    cfg = load_compare_config(compare_path)
    prob = load_problem(cfg)

    run_cfg = ppo_data["run"]
    seed = int(args.seed if args.seed is not None else run_cfg["seed"])
    threads = int(args.threads if args.threads is not None
                  else run_cfg.get("threads", 4))
    reward_scale = float(args.reward_scale if args.reward_scale is not None
                         else run_cfg.get("reward_scale", 1e-3))
    terminal_q_zero = bool(run_cfg.get("terminal_q_zero", True))
    if args.allow_terminal_q:
        terminal_q_zero = False

    episodes = (int(args.episodes) if args.episodes is not None
                else int(run_cfg.get("episodes", 3000)))
    if episodes <= 0:
        raise ValueError("训练回合数必须为正")
    total_steps = episodes * int(prob.T)

    # torch 会导入标准库 code 模块，必须在 load_problem 之后再导入
    from ppo_taufirst.ppo import PPOConfig, PPOTrainer, save_policy
    import torch
    torch.set_num_threads(max(1, threads))

    env = TauFirstEnv(
        prob, reward_scale=reward_scale, terminal_q_zero=terminal_q_zero,
        seed=seed)
    ppo_cfg = PPOConfig.from_dict(ppo_data["ppo"])
    tag = args.tag or str(run_cfg.get("tag", "ppo_taufirst_T60"))

    if args.out:
        run_dir = Path(args.out)
        if not run_dir.is_absolute():
            run_dir = PROJECT / run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"结果目录非空，拒绝覆盖: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(
            tag, args.archive_root or run_cfg.get("archive_root", "results"))

    started_at = datetime.now().isoformat(timespec="seconds")
    run = {
        "status": "running",
        "started_at": started_at,
        "tag": tag,
        "config": str(cfg.config_path),
        "ppo_config": str(ppo_cfg_path),
        "T": int(prob.T), "N_R": int(prob.N_R), "G": int(prob.G),
        "L": int(prob.L),
        "episodes": int(episodes),
        "steps": int(total_steps),
        "seed": seed,
        "reward_scale": reward_scale,
        "terminal_q_zero": terminal_q_zero,
        "torch_threads": int(threads),
    }
    write_json(run_dir / "run.json", run)
    copy_configs(run_dir, cfg, PROJECT, ppo_cfg_path)
    if not args.quiet:
        print(f"[train] result_dir={run_dir}")
        print(f"[train] compare={cfg.config_path}")
        print(f"[train] problem T={prob.T} N_R={prob.N_R} G={prob.G} "
              f"L={prob.L}; episodes={episodes} "
              f"(env_steps={total_steps}) seed={seed}")

    def _log(msg):
        print(msg, flush=True)

    t0 = time.perf_counter()
    trainer = PPOTrainer(
        env, ppo_cfg, seed=seed, log=None if args.quiet else _log)
    try:
        summary = trainer.train(total_steps, log_every=args.log_every)
    except Exception:
        run.update({"status": "failed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "wall_s": time.perf_counter() - t0,
                    "traceback": traceback.format_exc()})
        write_json(run_dir / "run.json", run)
        raise
    wall = time.perf_counter() - t0

    created_at = datetime.now().isoformat(timespec="seconds")
    meta = {
        "kind": "ppo_taufirst",
        "obs_dim": int(env.obs_dim),
        "act_dim": int(env.act_dim),
        "hidden": list(ppo_cfg.hidden),
        "log_std_init": float(ppo_cfg.log_std_init),
        "T": int(prob.T), "N_R": int(prob.N_R), "G": int(prob.G),
        "L": int(prob.L),
        "problem_data": str(cfg.problem_data),
        "problem_data_hash": sha256_file(cfg.problem_data)
        if Path(cfg.problem_data).is_file() else None,
        "compare_config": str(cfg.config_path),
        "state": "I,n,o1,o2,t",
        "timing": "tau_t -> q_t -> Theta_t -> y_t",
        "action": "10d raw; tau=raw[:5]*Imax; q=raw[5:]*Qmax; env projects",
        "obs_norm": {
            "Imax": np.asarray(prob.Imax, dtype=float).tolist(),
            "N": np.asarray(prob.N, dtype=float).tolist(),
            "Qmax": np.asarray(prob.Qmax, dtype=float).tolist(),
        },
        "action_scale": {
            "tau": np.asarray(prob.Imax, dtype=float).tolist(),
            "q": np.asarray(prob.Qmax, dtype=float).tolist(),
        },
        "theta": "online Dirichlet/Lambda sampler (no N×T pool)",
        "reward_scale": reward_scale,
        "terminal_q_zero": terminal_q_zero,
        "train_steps": int(total_steps),
        "train_seed": seed,
        "train_wall_s": float(wall),
        "created_at": created_at,
        "ppo": ppo_cfg.to_dict(),
    }
    policy_path = run_dir / "policy.pt"
    save_policy(policy_path, trainer.model, meta)
    write_json(run_dir / "policy_meta.json", meta)
    write_json(run_dir / "train_summary.json", {
        "summary": summary,
        "meta": meta,
        "history_rows": len(trainer.history),
    })
    write_train_log(run_dir / "train_log.csv", trainer.history)
    write_episode_log(run_dir / "episode_log.csv", trainer.episode_history)
    # 每个回合总成本的纯文本日志
    with (run_dir / "train_episodes.log").open("w", encoding="utf-8") as f:
        for row in trainer.episode_history:
            f.write(f"episode={row['episode']} steps={row['steps']} "
                    f"total_cost={row['total_cost']:.6f}\n")
    # 收敛曲线放到结果目录（独立子进程，避免 torch + matplotlib 的
    # OpenMP runtime 冲突）
    curve_path = run_dir / "train_curve.png"
    base_mean = None
    try:
        with np.load(cfg.baseline_npz, allow_pickle=False) as z:
            base_mean = float(np.mean(np.asarray(z["totals"])))
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 读取基线均值失败，收敛图不画基线线: {exc}")
    cmd = [
        sys.executable, "-B",
        str(Path(__file__).resolve().parent / "plot_train.py"),
        "--run", str(run_dir), "--out", str(curve_path),
    ]
    if base_mean is not None:
        cmd += ["--baseline-mean", f"{base_mean:.6f}"]
    try:
        res = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True,
                             text=True, timeout=120, check=False)
        if res.returncode != 0:
            print(f"[警告] 收敛图生成失败: {res.stderr.strip()[:500]}")
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 收敛图生成失败: {exc}")
    run.update({
        "status": "completed",
        "finished_at": created_at,
        "wall_s": float(wall),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "obs_dim": int(env.obs_dim),
        "act_dim": int(env.act_dim),
        "mean_episode_cost_last10": summary["mean_episode_cost_last10"],
        "episodes": summary["episodes"],
        "episode_log": str(run_dir / "episode_log.csv"),
        "train_episodes_log": str(run_dir / "train_episodes.log"),
        "train_curve": str(curve_path) if curve_path.is_file() else None,
    })
    write_json(run_dir / "run.json", run)
    prov = provenance(cfg, prob, {
        "kind": "ppo_taufirst",
        "meta": meta,
        "ppo": ppo_cfg.to_dict(),
    })
    write_json(run_dir / "provenance.json", prov)

    try:
        append_index(run_dir, {
            "run_id": run_dir.name,
            "kind": "train",
            "status": "completed",
            "T": int(prob.T),
            "steps": int(total_steps),
            "seed": seed,
            "wall_s": f"{wall:.2f}",
            "mean_episode_cost": f"{summary['mean_episode_cost_last10']:.4f}",
            "policy": str(policy_path),
            "created_at": created_at,
        })
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] index.csv 写入失败: {exc}")

    write_text(run_dir / "README.txt",
               "PPO tau-first 训练结果。\n"
               f"policy.pt: {policy_path}\n"
               f"train_log.csv: 每次 update 一行\n"
               f"episode_log.csv: 每个 episode 一行（cost/raw_reward/scaled_reward）\n"
               f"train_curve.png: 收敛曲线（若 matplotlib 可用）\n"
               f"train_summary.json / run.json / provenance.json\n"
               "评估见 ppo_taufirst/evaluate.py。\n")

    if not args.quiet:
        print(f"[train] completed wall={wall:.1f}s "
              f"episodes={summary['episodes']} "
              f"mean_ep_cost_last10={summary['mean_episode_cost_last10']:.2f}")
        print(f"[train] policy={policy_path}")
    return run_dir


if __name__ == "__main__":
    main()
