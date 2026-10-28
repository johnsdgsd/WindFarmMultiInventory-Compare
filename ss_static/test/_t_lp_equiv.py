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
from coptpy import COPT  # noqa: E402

from common.config import load_compare_config, load_policy_config  # noqa: E402
from common.model import bootstrap, load_problem, load_solver  # noqa: E402
from ss_static.simulator import Simulator, out_state_from_step  # noqa: E402
from ss_static.policy import SSStaticPolicy  # noqa: E402

sys.stdout.reconfigure(errors="replace")

PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def main():
    cfg = load_compare_config()
    prob = load_problem(cfg)
    m = bootstrap(cfg)
    scfg = load_solver(cfg)
    pcfg = load_policy_config(PROJECT)
    base = np.load(cfg.baseline_npz)
    x_in = base["x_in"]
    scenes = base["scene"]
    Ly = prob.layout
    ub = Ly.upper(prob)

    opts = dict(pcfg["policy"])
    pol = SSStaticPolicy(prob, [1.0] * prob.N_R, [3.0] * prob.N_R, opts)
    sim = Simulator(prob)

    t_list = [1, 3, 5, 10, 20, 30, 40, 50, 55, 57, 60]
    n_positive_q = 0
    for k, t in enumerate(t_list):
        p = (37 * t + 11) % x_in.shape[0]
        x = np.asarray(x_in[p, t - 1], dtype=float)
        si = int(scenes[p, t - 1])
        st = sim.step_from_state(pol, t, x, si)
        if np.any(st.q_out > 1e-9):
            n_positive_q += 1

        sub = m.sp.make_stage_subproblem(t, prob, scfg)
        try:
            sub.assemble(None)
            sub.set_point(x, prob.pool[si])
            for r in range(prob.N_R):
                v = sub.x_out[Ly.ix_q(r)]
                qv = float(st.q_out[0, r])
                sub.model.setInfo(COPT.Info.LB, [v], qv)
                sub.model.setInfo(COPT.Info.UB, [v], qv)
                for mm in range(prob.N_R):
                    yv = float(st.y[0, r, mm])
                    w = sub.y[r][mm]
                    sub.model.setInfo(COPT.Info.LB, [w], yv)
                    sub.model.setInfo(COPT.Info.UB, [w], yv)
            for v in sub.taup + sub.taum:
                sub.model.setInfo(COPT.Info.LB, [v], 0.0)
                sub.model.setInfo(COPT.Info.UB, [v], 0.0)
            sub.solve_lp()
            dec = sub.decision_values()
            xo = sub.out_state_vector()

            check(abs(sub.stage_cost() - float(st.terms.sum())) <= 1e-6,
                  f"t={t}: LP cost {sub.stage_cost()} vs {st.terms.sum()}")
            check(np.allclose(np.asarray(dec["q"], float), st.q_out[0],
                              atol=1e-6), f"t={t}: q 不一致")
            check(np.allclose(np.asarray(dec["y"], float), st.y[0], atol=1e-6),
                  f"t={t}: y 不一致")
            check(np.allclose(np.asarray(dec["S_end"], float), st.S_end[0],
                              atol=1e-6), f"t={t}: S_end 不一致")
            check(np.allclose(np.asarray(dec["n_out"], float), st.n_out[0],
                              atol=max(1e-6, prob.state_repair_tol)),
                  f"t={t}: n_out 不一致")
            xo2 = out_state_from_step(Ly, x, st, 0)
            check(np.max(np.abs(xo - xo2)) <= max(1e-6, prob.state_repair_tol),
                  f"t={t}: 下一状态不一致")

            a = m.sa.check_stage(
                prob, Ly, x, xo, dec["q"], dec["taup"], dec["taum"],
                dec["S_end"], dec["y"], si, ub, F_new=dec["F_new"])
            for key, val in a.items():
                lim = max(1e-6, prob.state_repair_tol) \
                    if key in m.sa.STATE_NUMERIC_CHECKS else 1e-6
                check(float(val) <= lim,
                      f"t={t}: check_stage {key}={val} > {lim}")
        finally:
            sub.free()

    check(n_positive_q >= 1, "采样点里没有正订单，测试未覆盖订货路径")

    # 虚拟首阶段：固定策略 q0，LP 应得到同一 q 和初值状态
    x0 = prob.initial_state()
    I = np.asarray(prob.I0, float)[None, :]
    o = np.asarray(prob.o0, float)[None, :, :]
    n = np.asarray(prob.n0, float)[None, :, :]
    qz = np.zeros((1, prob.N_R))
    q0 = pol.order(0, I, o, n, qz)
    sub0 = m.sp.make_stage_subproblem(0, prob, scfg)
    try:
        sub0.assemble(None)
        sub0.set_state(x0)
        for r in range(prob.N_R):
            v = sub0.x_out[Ly.ix_q(r)]
            qv = float(q0[0, r])
            sub0.model.setInfo(COPT.Info.LB, [v], qv)
            sub0.model.setInfo(COPT.Info.UB, [v], qv)
        sub0.solve_lp()
        xo0 = sub0.out_state_vector()
        check(np.allclose(xo0[Ly.q_slice()], q0[0], atol=1e-6),
              f"stage0 q {xo0[Ly.q_slice()]} vs {q0[0]}")
        check(np.allclose(xo0[:Ly.off_q], x0[:Ly.off_q], atol=1e-6),
              "stage0 非 q 分量应等于初值")
        check(np.allclose(q0, np.minimum(pol.S, prob.Qmax)[None, :]),
              f"stage0 q0={q0} 应为 min(S,Qmax)")
    finally:
        sub0.free()

    # ---------------------------------------------------------------- active mode 1
    # 构造一个明确有主动调拨的状态：仓 1 库存 10，其余 0；全部等级 0，需求很小。
    pcfg_a = load_policy_config(PROJECT, "ss_static/config/policy_active.toml")
    aopts = dict(pcfg_a["policy"])
    imax = np.asarray(prob.Imax, float)
    pol_a = SSStaticPolicy(prob, imax, imax, aopts,
                           h=np.full(prob.N_R, 5.0))
    sim_a = Simulator(prob)
    xa = np.asarray(prob.initial_state(), dtype=float)
    xa[Ly.ix_I(0)] = 10.0
    nb = xa[Ly.off_n:Ly.off_n + Ly.n_n].reshape(prob.N_R, prob.G)
    nb[:, 0] = prob.N
    nb[:, 1:] = 0.0
    t_a = 1
    si = 0
    st_a = sim_a.step_from_state(pol_a, t_a, xa, si)
    check(float(np.max(np.abs(st_a.taup) + np.abs(st_a.taum))) > 1e-9,
          "active LP 测试没有非零 tau")

    sub = m.sp.make_stage_subproblem(t_a, prob, scfg)
    try:
        sub.assemble(None)
        sub.set_point(xa, prob.pool[si])
        for r in range(prob.N_R):
            v = sub.x_out[Ly.ix_q(r)]
            qv = float(st_a.q_out[0, r])
            sub.model.setInfo(COPT.Info.LB, [v], qv)
            sub.model.setInfo(COPT.Info.UB, [v], qv)
            for mm in range(prob.N_R):
                w = sub.y[r][mm]
                yv = float(st_a.y[0, r, mm])
                sub.model.setInfo(COPT.Info.LB, [w], yv)
                sub.model.setInfo(COPT.Info.UB, [w], yv)
            sub.model.setInfo(COPT.Info.LB, [sub.taup[r]],
                              float(st_a.taup[0, r]))
            sub.model.setInfo(COPT.Info.UB, [sub.taup[r]],
                              float(st_a.taup[0, r]))
            sub.model.setInfo(COPT.Info.LB, [sub.taum[r]],
                              float(st_a.taum[0, r]))
            sub.model.setInfo(COPT.Info.UB, [sub.taum[r]],
                              float(st_a.taum[0, r]))
        sub.solve_lp()
        dec = sub.decision_values()
        xo = sub.out_state_vector()
        check(abs(sub.stage_cost() - float(st_a.terms.sum())) <= 1e-6,
              f"active LP cost {sub.stage_cost()} vs {st_a.terms.sum()}")
        check(np.allclose(np.asarray(dec["q"], float), st_a.q_out[0],
                          atol=1e-6), "active q 不一致")
        check(np.allclose(np.asarray(dec["y"], float), st_a.y[0],
                          atol=1e-6), "active y 不一致")
        check(np.allclose(np.asarray(dec["S_end"], float), st_a.S_end[0],
                          atol=1e-6), "active S_end 不一致")
        check(np.allclose(np.asarray(dec["taup"], float), st_a.taup[0],
                          atol=1e-6), "active taup 不一致")
        check(np.allclose(np.asarray(dec["taum"], float), st_a.taum[0],
                          atol=1e-6), "active taum 不一致")
        check(np.allclose(np.asarray(dec["n_out"], float), st_a.n_out[0],
                          atol=max(1e-6, prob.state_repair_tol)),
              "active n_out 不一致")
        xo2 = out_state_from_step(Ly, xa, st_a, 0)
        check(np.max(np.abs(xo - xo2)) <= max(1e-6, prob.state_repair_tol),
              "active 下一状态不一致")
        active_cost = float(st_a.terms[0, 3])
        expected = float(prob.atc / 2.0
                         * (st_a.taup[0] + st_a.taum[0]).sum())
        check(abs(active_cost - expected) <= 1e-9,
              f"active cost {active_cost} vs {expected}")
        a = m.sa.check_stage(
            prob, Ly, xa, xo, dec["q"], dec["taup"], dec["taum"],
            dec["S_end"], dec["y"], si, ub, F_new=dec["F_new"])
        for key, val in a.items():
            lim = max(1e-6, prob.state_repair_tol) \
                if key in m.sa.STATE_NUMERIC_CHECKS else 1e-6
            check(float(val) <= lim,
                  f"active check_stage {key}={val} > {lim}")
    finally:
        sub.free()

    print(f"_t_lp_equiv: {PASS} checks passed "
          f"({n_positive_q}/{len(t_list)} 个采样点有正订单)")


if __name__ == "__main__":
    main()
