"""画每个回合的总成本随回合数下降的曲线（train_curve.png）。

- 只读取并绘制 episode_log.csv 的 total_cost（原始未缩放成本）；
- 浅色细线为逐回合原始成本，深色粗线为 MA(10) 或 EMA；
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


def _moving_average(values, window: int) -> list[float]:
    """尾部滑动平均；前 window-1 个点用已有数据的均值。"""
    out: list[float] = []
    window = max(1, int(window))
    for i, v in enumerate(values):
        if v is None or not (v == v):
            out.append(float("nan"))
            continue
        vals = [float(x) for x in values[max(0, i - window + 1):i + 1]
                if x is not None and (x == x)]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
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
                    help="EMA 平滑系数（0<alpha<=1；ma-window=0 时使用）")
    ap.add_argument("--ma-window", type=int, default=10,
                    help="尾部滑动平均窗口，默认 10；0 表示改用 EMA")
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
            label="total cost per episode (raw, unscaled)")
    if args.ma_window > 0:
        smooth = _moving_average(cost, args.ma_window)
        smooth_label = f"MA(window={args.ma_window})"
    else:
        smooth = _ema(cost, alpha)
        smooth_label = f"EMA(α={alpha:g})"
    ax.plot(episodes, smooth, "-", color="tab:blue", lw=2.2,
            label=smooth_label)
    if args.baseline_mean is not None:
        ax.axhline(args.baseline_mean, color="tab:red", ls="--", lw=1.3,
                   label=f"baseline mean ({args.baseline_mean:.1f})")
    ax.set_title(f"Total cost per episode — {title} (episodes={n})")
    ax.set_xlabel("episode")
    ax.set_ylabel("total cost (10k CNY / episode)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[plot] {out}")
    return out


if __name__ == "__main__":
    main()
