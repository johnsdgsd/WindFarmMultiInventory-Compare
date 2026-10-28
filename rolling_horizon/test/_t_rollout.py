from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from common.baseline import load_baseline  # noqa: E402
from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from rolling_horizon.policy import RollingHorizonPolicy  # noqa: E402
from rolling_horizon.scenarios import make_opt_pool  # noqa: E402
from rolling_horizon.simulator import (Simulator, audit_result,  # noqa: E402
                                      cost_terms_max_dev)

sys.stdout.reconfigure(errors="replace")
PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def main():
    cfg = load_compare_config("config/compare_T60_base204105.toml")
    prob = load_problem(cfg)
    base = load_baseline(cfg, prob)
    opt = make_opt_pool(prob, 2, 2027)
    opts = {"solver": "copt", "threads": 1, "workers": 1,
            "log_every": 1, "time_limit": 0.0, "active": "lp"}

    def run():
        pol = RollingHorizonPolicy(
            prob, opts, opt["pool"], seed_opt=2027,
            pool_hash=opt["pool_hash"])
        sim = Simulator(prob)
        res = sim.simulate(pol, base.scene[:1], record=True)
        return pol, sim, res

    pol, sim, res = run()

    # 1) 基本结果与 q0
    check(res.P == 1 and res.T == prob.T, "result shape")
    check(np.allclose(res.q0, 0.0), "q0 非零")
    check(np.allclose(res.arrays["x_in"][:, 0, prob.layout.q_slice()], 0.0),
          "x_in 首期 q0 非零")
    check(np.allclose(res.arrays["q"][:, -1, :], 0.0), "q_T 非零")
    check(np.all(res.arrays["q"] >= -1e-8), "q 负值")
    check(np.all(res.arrays["q"]
                 <= np.asarray(prob.Qmax)[None, None, :] + 1e-8),
          "q 越上界")

    # 2) 状态/主动调拨恒等式
    tau = res.arrays["taup"] - res.arrays["taum"]
    check(np.allclose(res.arrays["taup"], np.maximum(tau, 0.0)),
          "taup != max(tau,0)")
    check(np.allclose(res.arrays["taum"], np.maximum(-tau, 0.0)),
          "taum != max(-tau,0)")
    check(np.allclose(tau.sum(axis=2), 0.0, atol=1e-8), "tau 不守恒")
    iout = res.arrays["S_end"] + tau
    check(np.allclose(res.arrays["x_in"][:, 1:, :prob.N_R],
                      iout[:, :-1, :], atol=cfg.state_tol),
          "I_out 与下一期 I 不一致")
    active = res.arrays["terms"][:, :, 3].sum(axis=1)
    expect = prob.atc / 2.0 * (
        res.arrays["taup"] + res.arrays["taum"]).sum(axis=(1, 2))
    check(np.allclose(active, expect), "active cost 恒等式")
    check(np.allclose(res.arrays["terms"].sum(axis=2), res.arrays["cost"]),
          "五项和 != 总成本")
    check(cost_terms_max_dev(res) < 1e-9, "cost_terms_max_dev")

    # 3) 审计
    audit, bad = audit_result(prob, res, tol=cfg.tol,
                              state_tol=cfg.state_tol)
    check(not bad, f"审计违规 {bad}")
    check("cap" not in audit, "不应再审计旧 cap")
    check(float(audit["cap_init"]) < 1e-8, "cap_init 审计")
    check(float(audit["cap_next"]) < 1e-8, "cap_next 审计")
    check(float(audit["q.上"]) < 1e-8, "q 上界审计")

    # 4) 策略统计与接口
    params = pol.params()
    check(params["kind"] == "rolling_horizon", "kind")
    check(params["n_scenarios"] == 2, "N")
    check(params["n_two_stage_calls"] == prob.T, "two-stage 次数")
    check(params["n_recourse_calls"] == prob.T, "recourse 次数")
    check(params["n_lp_calls"] == 2 * prob.T, "LP 总次数")
    check(params["lp_failures"] == 0, "LP 失败")
    check(params["state"] == "I,n,o1..o(L-1)", "state")
    check(params["purchase_timing"] == "current_q", "purchase_timing")
    check(params["capacity_side"] == "period_and_next", "capacity_side")
    check("q_in_recurrence" not in params, "不应再有 q_in")
    check(params["workers"] == 1, "workers")
    check(params["max_two_stage_vars"] > 0, "max vars")
    check(params["two_stage_solve_s_total"] >= 0, "solve total")
    check(pol.active_transfer(
        prob.T, res.arrays["S_end"][:, -1], res.arrays["W_out"][:, -1],
        res.arrays["n_out"][:, -1], None, res.arrays["q"][:, -1],
        None).shape == (1, prob.N_R), "active_transfer shape")

    # 5) 同一池/路径可复现
    pol2, sim2, res2 = run()
    check(np.allclose(res.totals, res2.totals, atol=1e-8), "结果不可复现")
    check(np.allclose(res.arrays["q"], res2.arrays["q"], atol=1e-8),
          "q 不可复现")
    check(np.allclose(res.arrays["y"], res2.arrays["y"], atol=1e-8),
          "y 不可复现")

    active_volume = float(
        (res.arrays["taup"] + res.arrays["taum"]).sum() / 2.0)
    print(f"_t_rollout: {PASS} checks passed")
    print(f"  total={res.mean:.4f}  two_stage={params['n_two_stage_calls']}  "
          f"vars={params['max_two_stage_vars']}  active={active_volume:.3f}")


if __name__ == "__main__":
    main()
