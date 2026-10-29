"""画每个回合的总成本随回合数下降的曲线（train_curve.png）。

- 读取训练目录的 episode_log.csv（episode, total_cost, ...）；
- 原始点 + EMA 平滑；
- 可选画基线均值虚线。

用法::

    P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
    "$P" -B ppo_taufirst/plot_train.py \
        --run results/ppo_taufirst/<train_run> --baseline-mean 35833.3228
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(path: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.items():
            if v in ("", None):
                continue
            try:
                out.setdefault(k, []).append(float(v))
            except ValueError:
                pass
    return out


def _ema(values, alpha: float = 0.2) -> list[float]:
    out: list[float] = []
    s = None
    for v in values:
        if v is None or not (v == v):
            out.append(float("nan"))
            continue
        s = float(v) if s is None else alpha * float(v) + (1 - alpha) * s
        out.append(s)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="plot total cost per episode")
    ap.add_argument("--run", required=True,
                    help="训练结果目录或 episode_log.csv")
    ap.add_argument("--out", default=None,
                    help="输出 PNG；默认 <run>/train_curve.png")
    ap.add_argument("--baseline-mean", type=float, default=None,
                    help="基线均值虚线（万元/路径）")
    ap.add_argument("--smooth-alpha", type=float, default=0.2,
                    help="EMA 平滑系数（0<alpha<=1）")
    args = ap.parse_args(argv)
    p = Path(args.run)
    if not p.is_absolute():
        p = PROJECT / p
    if p.is_dir():
        ep_path = p / "episode_log.csv"
        default_out = p / "train_curve.png"
        title = p.name
    else:
        ep_path = p
        default_out = p.with_name("train_curve.png")
        title = p.parent.name
    if not ep_path.is_file():
        raise FileNotFoundError(f"episode_log.csv 不存在: {ep_path}")
    d = _read(ep_path)
    episodes = d.get("episode", [])
    cost = d.get("total_cost", d.get("cost", []))
    if not cost:
        raise ValueError(f"{ep_path} 没有 total_cost/cost 列")
    if not episodes:
        episodes = list(range(1, len(cost) + 1))
    n = min(len(episodes), len(cost))
    episodes, cost = episodes[:n], cost[:n]
    alpha = min(1.0, max(0.01, float(args.smooth_alpha)))
    out = Path(args.out) if args.out else default_out
    if not out.is_absolute():
        out = PROJECT / out

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.2), constrained_layout=True)
    ax.plot(episodes, cost, "-", color="tab:blue", lw=0.7, alpha=0.25,
            label="total cost per episode")
    ax.plot(episodes, _ema(cost, alpha), "-", color="tab:blue", lw=2.2,
            label=f"EMA(α={alpha:g})")
    if args.baseline_mean is not None:
        ax.axhline(args.baseline_mean, color="tab:red", ls="--", lw=1.3,
                   label=f"baseline mean ({args.baseline_mean:.1f})")
    ax.set_title(f"Total cost per episode — {title} (episodes={n})")
    ax.set_xlabel("episode")
    ax.set_ylabel("total cost (10k CNY / episode)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    k = max(1, n // 10)
    ax.text(0.01, 0.98,
            f"first 10% mean: {sum(cost[:k]) / k:.1f}\n"
            f"last 10% mean:  {sum(cost[-k:]) / k:.1f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[plot] {out}")
    return out


if __name__ == "__main__":
    main()
