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

import numpy as np  # noqa: E402

from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ss_static.calibrate import calibrate  # noqa: E402
from ss_static.policy import SSStaticPolicy  # noqa: E402

sys.stdout.reconfigure(errors="replace")


def main():
    cfg = load_compare_config()
    prob = load_problem(cfg)
    with tempfile.TemporaryDirectory() as td:
        d = calibrate(PROJECT, tag="test_calib", quick=True,
                      archive_root=td, train_paths=6, val_paths=6)
        pol_path = Path(d) / "policy.json"
        if not pol_path.is_file():
            raise AssertionError("没有生成 policy.json")
        data = json.loads(pol_path.read_text(encoding="utf-8"))
        pol = SSStaticPolicy.from_dict(prob, data)
        s = np.asarray(pol.s, float)
        S = np.asarray(pol.S, float)
        if np.any(s < -1e-9) or np.any(s > S + 1e-9):
            raise AssertionError(f"s/S 越界: s={s}, S={S}")
        if np.any(S > prob.Imax + 1e-9):
            raise AssertionError(f"S 超过 Imax: S={S}")
        cal = data.get("calibration", {})
        if cal.get("train_paths") != 6 or cal.get("val_paths") != 6:
            raise AssertionError(f"quick 路径数异常: {cal.get('train_paths')}/"
                                 f"{cal.get('val_paths')}")
        if cal.get("mode_tau") != 0:
            raise AssertionError("主基准必须是 mode_tau=0（无主动调拨）")
        if data["options"].get("cross_cost_rule") != "all":
            raise AssertionError("被动调拨应为全配对，无阈值")
        if cal.get("selected_algorithm") != "coordinate":
            raise AssertionError(f"quick 应选 coordinate: {cal.get('selected_algorithm')}")
        if data.get("problem", {}).get("T") != prob.T:
            raise AssertionError("policy problem T 不一致")
        if not np.isfinite(cal.get("train_cost", np.nan)) or \
                not np.isfinite(cal.get("val_cost", np.nan)):
            raise AssertionError("train/val cost 非有限")
        for name in ("calibration.md", "algorithm_summary.csv",
                     "algorithm_starts.csv", "optimization_history.csv",
                     "paths_train.npz", "paths_val.npz", "run.json",
                     "provenance.json"):
            if not (Path(d) / name).is_file():
                raise AssertionError(f"缺少 {name}")
        ip = Path(td) / "ss_static" / "index.csv"
        rows = list(csv.DictReader(ip.open(encoding="utf-8")))
        if len(rows) != 1 or rows[0]["status"] != "completed":
            raise AssertionError(f"index 异常: {rows}")
        print("_t_calibrate: all checks passed")
        print(f"  result={d}")
        print(f"  s={np.round(s, 4).tolist()} S={np.round(S, 4).tolist()}")
        print(f"  train={cal['train_cost']:.4f} val={cal['val_cost']:.4f} "
              f"algo={cal.get('selected_algorithm')}")


if __name__ == "__main__":
    main()
