from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import csv  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from common.baseline import load_baseline  # noqa: E402
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ss_static.calibrate import calibrate  # noqa: E402
from ss_static.evaluate import _summary_md  # noqa: E402
from ss_static.metrics import baseline_behavior  # noqa: E402
from ss_static.simulator import TERM_NAMES  # noqa: E402

sys.stdout.reconfigure(errors="replace")

PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def _fake_t240_summary() -> str:
    prob = SimpleNamespace(T=240, G=4, N_R=5, L=3, state_dim=40)
    cfg = SimpleNamespace(config_path=Path("config/compare_T240.toml"))
    base = SimpleNamespace(
        name="SDDP_T240_377r", result_dir=Path("baseline_T240"),
        npz_path=Path("sim_1000paths.npz"), P=1000,
        mean=159380.8723, std=71692.2152, ci95=4443.4479,
        scene_hash="b9189b76cb701e1b" + "0" * 48,
        meta={"lb": 112307.4432, "gap_pct": 29.535,
              "audit_violations": {},
              "terms_vs_stage_cost_max_dev": 0.0})
    pol = SimpleNamespace(
        s=np.array([12.0, 8.0, 17.5318, 11.9736, 8.0]),
        S=np.array([12.0, 8.0, 20.0, 14.3835, 8.0]))
    policy_data = {"calibration": {
        "selected_algorithm": "coordinate", "selected_start": "two_z",
        "train_paths": 1000, "val_paths": 1000,
        "train_seed": 777, "val_seed": 888,
        "train_cost": 187588.50, "val_cost": 190791.96,
    }}
    res = SimpleNamespace(cost=188154.7413, std=82980.1401, ci95=5143.0679)
    ps = {"diff_mean": 28773.8690, "diff_se": 658.7963, "t": 43.68,
          "ci95_low": 27482.6520, "ci95_high": 30065.0860,
          "win_rate_policy_lower": 0.056, "rel_diff_pct": 18.054, "n": 1000}
    ts_list = [{"term": n, "baseline_mean": 100.0, "policy_mean": 110.0,
                "diff_mean": 10.0, "ci95_low": 1.0, "ci95_high": 19.0}
               for n in TERM_NAMES]
    beh = {"total_failures": 1415.97, "total_replacements": 1245.54,
           "fill_rate": 0.8856, "final_backlog": 170.43,
           "mean_backlog": 18.14, "mean_inventory": 14.16,
           "total_orders_placed": 1245.77, "total_orders_charged": 1245.77,
           "order_frequency": 0.48, "local_replacements": 941.27,
           "cross_replacements": 304.26, "active_transfer_volume": 447.48}
    tp = {
        "deciles": [{"decile": 1, "t_first": 1, "t_last": 24,
                     "n_periods": 24, "policy_cost": 4091.2,
                     "baseline_cost": 631.9, "diff": 3459.3}],
        "first_10pct_diff": 3459.3, "last_10pct_diff": 31141.5,
        "middle_diff": -7355.4, "total_diff": 28773.87,
        "worst_decile": {"decile": 10, "t_first": 217, "t_last": 240,
                         "diff": 23029.1},
        "best_decile": {"decile": 7, "t_first": 145, "t_last": 168,
                        "diff": -3109.0},
    }
    term_list = [{"label": "无期末值", "baseline_mean": 159380.87,
                  "policy_mean": 188154.74, "diff_mean": 28773.87,
                  "rel_diff_pct": 18.05}]
    return _summary_md(prob, cfg, base, pol, policy_data, res, ps, ts_list,
                       beh, beh, tp, term_list, {}, {}, 0.0, "fake_T240")


def main():
    global PASS
    text = _fake_t240_summary()
    check("T=240" in text, "T240 报告未显示 T=240")
    check("1000" in text, "T240 报告未显示 1000 条路径")
    check("112307.44" in text, "T240 报告未显示 T240 的 LB")
    check("T=60" not in text, "T240 报告泄漏 T=60")
    check("712.9522" not in text, "T240 报告泄漏 T60 LB")
    check("2000 条" not in text, "T240 报告泄漏 2000 条路径")

    cfg = load_compare_config()
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)
    beh = baseline_behavior(prob, base)
    expected = float(np.mean(base.x_in[:, :, :prob.N_R].sum(axis=2)))
    check(abs(beh["mean_inventory"] - expected) < 1e-9,
          "基线 mean_inventory 不是五仓合计")
    known_inv = {
        (2000, 60): 0.9417050626635332,
        (3000, 60): 24.083876922715188,
        (1000, 240): 14.15815212836061,
    }.get((base.P, int(prob.T)))
    if known_inv is not None:
        check(abs(beh["mean_inventory"] - known_inv) < 1e-6,
              f"基线库存应为 {known_inv}，实际 {beh['mean_inventory']}")

    with tempfile.TemporaryDirectory() as td:
        d1 = calibrate("ss_static", tag="arch", quick=True,
                       archive_root=td, train_paths=6, val_paths=6)
        d2 = calibrate("ss_static", tag="arch", quick=True,
                       archive_root=td, train_paths=6, val_paths=6)
        check(d1 != d2 and Path(d1).is_dir() and Path(d2).is_dir(),
              "两次运行目录未分开")
        for d in (d1, d2):
            for name in ("run.json", "policy.json", "calibration.md",
                         "algorithm_summary.csv", "optimization_history.csv",
                         "paths_train.npz", "paths_val.npz",
                         "provenance.json"):
                check((Path(d) / name).is_file(), f"{d} 缺 {name}")
            check((Path(d) / "config" / "compare.toml").is_file(),
                  f"{d} 缺 config/compare.toml")
            run = json.loads((Path(d) / "run.json").read_text(encoding="utf-8"))
            check(run["status"] == "completed", f"{d} run.json 状态不对")
            check(int(run["T"]) == prob.T, f"{d} run.json T 不对")
            pol = json.loads((Path(d) / "policy.json").read_text(encoding="utf-8"))
            check(pol["problem"]["T"] == prob.T, f"{d} policy T 不对")
            check(np.all(np.asarray(pol["s"]) <= np.asarray(pol["S"]) + 1e-9),
                  "policy s>S")
        ip = Path(td) / "ss_static" / "index.csv"
        check(ip.is_file(), "index.csv 未生成")
        rows = list(csv.DictReader(ip.open(encoding="utf-8")))
        check(len(rows) == 2, f"index 行数 {len(rows)} != 2")
        check(all(r["kind"] == "calib" and r["status"] == "completed"
                  for r in rows), "index 行状态不对")

    print(f"_t_archive: {PASS} checks passed")


if __name__ == "__main__":
    main()
