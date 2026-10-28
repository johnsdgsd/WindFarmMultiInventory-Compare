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
from rolling_horizon.evaluate import evaluate  # noqa: E402

sys.stdout.reconfigure(errors="replace")
PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def main():
    global PASS
    with tempfile.TemporaryDirectory(prefix="roll_eval_") as td:
        d = evaluate(
            ROOT, config_path="config/compare_T60_base204105.toml",
            policy_config="rolling_horizon/config/policy.toml",
            tag="t_eval", archive_root=td, quick=True,
            threads=1, workers=1, skip_audit=False)
        d = Path(d)
        check(d.is_dir(), "result dir")
        for name in ("summary.md", "comparison.md", "conclusion.md",
                     "comparison.json", "policy.json", "provenance.json",
                     "run.json", "sim_meta.json", "audit.json",
                     "lp_stats.csv", "sim_2paths.npz",
                     "run.log", "progress.log"):
            check((d / name).is_file(), f"缺少 {name}")
        check((d / "config" / "compare.toml").is_file(), "config/compare")
        check((d / "config" / "policy.toml").is_file(), "config/policy")
        check((d / "config" / "problem_data.json").is_file(),
              "config/problem_data")

        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        check(run["status"] == "completed", "run status")
        check(run["quick"] is True, "quick flag")
        check(run["N"] == 2 and run["M"] == 2, "quick N/M")
        check(run["audit_pass"] is True, "audit")
        check(run["diff_mean"] == run["diff_mean"], "diff nan")

        comp = json.loads((d / "comparison.json").read_text(
            encoding="utf-8"))
        check(comp["headline"]["audit_pass"] is True, "comparison audit")
        check(comp["protocol"]["N"] == 2 and comp["protocol"]["M"] == 2,
              "comparison protocol")
        check(comp["protocol"]["state"] == "I,n,o1..o(L-1)",
              "comparison state")
        check(comp["protocol"]["purchase_timing"] == "current_q",
              "comparison purchase_timing")
        check(comp["protocol"]["capacity_side"] == "period_and_next",
              "comparison capacity")
        check(comp["protocol"]["workers"] == 1, "comparison workers")
        check(len(comp["terms"]) == 5, "terms")
        check(comp["lp"]["t1_n_vars"] > 0, "lp t1 vars")
        check(comp["lp"]["n_two_stage"] == 2 * 60, "lp two-stage count")
        check(comp["lp"]["n_recourse"] == 2 * 60, "lp recourse count")
        check(comp["saa"]["note"], "saa note")
        check("behavior" in comp and "policy" in comp["behavior"],
              "behavior")

        pol = json.loads((d / "policy.json").read_text(encoding="utf-8"))
        check(pol["protocol"]["state"] == "I,n,o1..o(L-1)", "policy state")
        check(pol["protocol"]["purchase_timing"] == "current_q",
              "policy purchase_timing")
        check("initial_q0" not in pol["protocol"], "policy 不应有 q0")
        check(pol["protocol"]["window"] == "full_remaining", "policy window")
        check(pol["policy"]["n_scenarios"] == 2, "policy N")
        check(pol["policy"]["n_two_stage_calls"] == 120, "policy two-stage")
        check(pol["policy"]["n_recourse_calls"] == 120, "policy recourse")

        prob = load_problem(load_compare_config(
            "config/compare_T60_base204105.toml"))
        with np.load(d / "sim_2paths.npz") as z:
            check(z["q"].shape == (2, 60, 5), "sim q shape")
            check(z["scene"].shape == (2, 60), "sim scene shape")
            check(np.allclose(
                z["x_in"][:, 0, prob.layout.q_slice()], 0.0), "sim q0")
            check(np.allclose(z["q"][:, -1, :], 0.0), "sim q_T")

        with open(d / "lp_stats.csv", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        check(rows[0][0] == "path", "lp_stats header")
        check(len(rows) == 1 + 120 + 120, "lp_stats rows")

        with open(d.parent / "index.csv", encoding="utf-8") as f:
            idx = list(csv.reader(f))
        check(len(idx) == 2, "index rows")
        check(idx[0][0] == "created_at", "index header")

        txt = (d / "summary.md").read_text(encoding="utf-8")
        for s in ("滚动时域", "审计", "限制", "优化池", "LP", "I,n,o1..o(L-1)",
                  "current_q"):
            check(s in txt, f"summary 缺 {s}")
        check("N=10" not in txt or "N=10 是" not in txt,
              "summary 仍硬编码 N=10")

        prov = json.loads((d / "provenance.json").read_text(
            encoding="utf-8"))
        check("rolling" in prov, "provenance rolling")
        check(prov["rolling"]["N"] == 2, "provenance N")
        check("q_in_recurrence" not in prov["rolling"], "provenance q_in")
        check(prov["rolling"]["state"] == "I,n,o1..o(L-1)", "provenance state")
        check(prov["rolling"]["purchase_timing"] == "current_q",
              "provenance purchase_timing")
        check(prov["rolling"]["capacity_side"] == "period_and_next", "provenance cap")

    print(f"_t_evaluate: {PASS} checks passed")


if __name__ == "__main__":
    main()
