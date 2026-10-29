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

from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from rolling_horizon.lp import RecourseLP, State, TwoStageLP  # noqa: E402
from rolling_horizon.scenarios import make_opt_pool  # noqa: E402
from rolling_horizon.simulator import initial_o  # noqa: E402

sys.stdout.reconfigure(errors="replace")
PASS = 0
TOL = 2e-6


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def val(expr, x):
    idx, c = expr
    return (float(x[idx]) if idx is not None else 0.0) + float(c)


def main():
    global PASS
    cfg = load_compare_config("config/compare_T60_base204105.toml")
    prob = load_problem(cfg)
    opt = make_opt_pool(prob, 2, 2027)
    K, G = prob.L - 1, prob.G
    o0 = initial_o(prob)
    state = State(prob.I0, o0, prob.n0)
    state.validate(prob.N_R, prob.G, prob.L)

    lp = TwoStageLP(prob, state, 1, opt["pool"], solver="copt", threads=1)
    res = lp.solve()
    x = lp.x
    NR = prob.N_R

    check(res.q_first.shape == (NR,), "q_first shape")
    check(res.tau_first.shape == (NR,), "tau_first shape")
    check(np.all(res.q_first >= -1e-8), "q_first 负值")
    check(np.all(res.q_first <= np.asarray(prob.Qmax) + 1e-6),
          "q_first 越上界")
    check(len(lp.qfirst) == NR, "qfirst 数量")
    check(len(lp.taufp) == NR and len(lp.taufm) == NR, "tauf 数量")
    check(lp.dec[(0, 1)]["q"] is None, "t 期 q 应使用 qfirst")
    check(lp.dec[(0, prob.T)]["q"] is not None, "T 期 q 应为追索变量")
    check(lp.stats.n_vars > 0 and lp.stats.n_eq > 0 and lp.stats.n_le > 0,
          "LP 统计")
    check(lp.stats.window == prob.T, "window")
    check(lp.stats.solver == "copt", "solver")
    init_cap = state.I + res.tau_first + state.o.sum(axis=0) + res.q_first
    check(np.all(init_cap <= np.asarray(prob.Imax) + TOL),
          f"期初容量 qfirst/tau 越界: {init_cap}")

    for s in range(2):
        for r in range(NR):
            check(abs(val(lp._q(s, 1, r), x) - res.q_first[r]) < 1e-8,
                  "t 期 q != qfirst")

    for s in range(2):
        for tau in range(1, prob.T + 1):
            d = lp.dec[(s, tau)]
            nx = lp.state_vars[(s, tau + 1)]
            th = lp.theta[s, tau - 1]
            I_tau = np.array([val(lp._I(s, tau, r), x) for r in range(NR)])
            o_tau = np.array([[val(lp._o(s, tau, r, l), x)
                               for r in range(NR)] for l in range(K)])
            n_tau = np.array([[val(lp._n(s, tau, r, i), x)
                               for i in range(G)] for r in range(NR)])
            I_next = np.array([x[nx["I"][r]] for r in range(NR)])
            o_next = np.array([[x[nx["o"][l][r]] for r in range(NR)]
                               for l in range(K)])
            n_next = np.array([[x[nx["n"][r][i]] for i in range(G)]
                               for r in range(NR)])
            q = np.array([val(lp._q(s, tau, r), x) for r in range(NR)])
            yv = np.array([[x[d["y"][r][m]] for m in range(NR)]
                           for r in range(NR)])
            taup = np.array([x[d["taup"][r]] for r in range(NR)])
            taum = np.array([x[d["taum"][r]] for r in range(NR)])
            tau_net = taup - taum
            Sn = np.array([x[d["S_end"][r]] for r in range(NR)])
            Nv = np.zeros((NR, G, G + 1))
            for r in range(NR):
                for i in range(G):
                    for j in range(i, G + 1):
                        Nv[r, i, j] = x[d["N"][r][i][j]]
            outflow = yv.sum(axis=0)
            Y = yv.sum(axis=1)

            check(np.allclose(Nv, th * n_tau[:, :, None], atol=TOL),
                  f"N=Theta n t={tau}")
            check(np.allclose(n_next[:, 0], Nv[:, 0, 0] + Y, atol=TOL),
                  f"grade0 t={tau}")
            for j in range(1, G):
                check(np.allclose(
                    n_next[:, j], Nv[:, :j + 1, j].sum(axis=1), atol=TOL),
                    f"grade{j} t={tau}")
            check(np.allclose(Sn, I_tau + tau_net - outflow + o_tau[0],
                              atol=TOL), f"S_end t={tau}")
            check(np.allclose(I_next, Sn, atol=TOL),
                  f"I_out t={tau}")
            if K > 1:
                check(np.allclose(o_next[:K - 1], o_tau[1:], atol=TOL),
                      f"pipe t={tau}")
            check(np.allclose(o_next[K - 1], q, atol=TOL),
                  f"pipe_last t={tau}")
            check(np.all(I_tau + tau_net >= -TOL), f"tau_lo t={tau}")
            check(np.all(I_tau + tau_net
                         <= np.asarray(prob.Imax) + TOL),
                  f"tau_hi t={tau}")
            check(np.all(tau_net <= np.array(
                [I_tau[[m for m in range(NR) if m != r]].sum()
                 for r in range(NR)]) + TOL), f"tau_src t={tau}")
            check(abs(tau_net.sum()) < TOL, f"tau_bal t={tau}")
            check(np.all(outflow <= I_tau + tau_net + TOL),
                  f"pas t={tau}")
            check(np.all(Y + Nv[:, :, :G].sum(axis=(1, 2))
                         <= np.asarray(prob.N) + prob.fleet_tol + TOL),
                  f"dem t={tau}")
            check(np.all(n_tau.sum(axis=1)
                         <= np.asarray(prob.N) + prob.fleet_tol + TOL),
                  f"fleet t={tau}")
            if tau > 1:
                check(np.all(I_tau + tau_net + o_tau.sum(axis=0) + q
                             <= np.asarray(prob.Imax) + TOL),
                      f"cap t={tau}")
            check(np.all(I_next + o_next.sum(axis=0)
                         <= np.asarray(prob.Imax) + TOL),
                  f"cap_next t={tau}")
            check(np.all(q >= -TOL), f"q 下 t={tau}")
            check(np.all(q <= np.asarray(prob.Qmax) + TOL),
                  f"q 上 t={tau}")
            check(np.all(I_next >= -TOL), f"I_next 下 t={tau}")
            check(np.all(I_next <= np.asarray(prob.Imax) + TOL),
                  f"I_next 上 t={tau}")
            check(np.all(o_next >= -TOL), f"o_next 下 t={tau}")
            check(np.all(o_next <= np.asarray(prob.Qmax)[None, :] + TOL),
                  f"o_next 上 t={tau}")
            check(np.all(n_next >= -TOL), f"n_next 下 t={tau}")
            check(np.all(n_next <= np.asarray(prob.N)[:, None] + TOL),
                  f"n_next 上 t={tau}")

    # 目标复算
    obj = 0.0
    for s in range(2):
        for tau in range(1, prob.T + 1):
            d = lp.dec[(s, tau)]
            nx = lp.state_vars[(s, tau + 1)]
            for r in range(NR):
                obj += prob.C * val(lp._q(s, tau, r), x)
                obj += prob.h / 2.0 * (
                    val(lp._I(s, tau, r), x) + x[nx["I"][r]])
                W_in = prob.N[r] - sum(
                    val(lp._n(s, tau, r, i), x) for i in range(G))
                obj += prob.Cs * W_in
                obj += prob.atc / 2.0 * (
                    x[d["taup"][r]] + x[d["taum"][r]])
                for m in range(NR):
                    obj += prob.tc_route[r, m] * x[d["y"][r][m]]
    obj /= 2.0
    check(abs(obj - lp.stats.objective) <= 1e-4 * max(1.0, abs(obj)),
          f"objective 复算 {obj} vs {lp.stats.objective}")

    # t=T-1 与 t=T
    lp2 = TwoStageLP(prob, state, prob.T - 1,
                     opt["pool"][:, prob.T - 2:, :, :, :],
                     solver="copt", threads=1)
    r2 = lp2.solve()
    for s in range(2):
        for r in range(NR):
            check(abs(val(lp2._q(s, prob.T - 1, r), lp2.x)
                      - r2.q_first[r]) < 1e-8, "t=T-1 qfirst")
        check(lp2.dec[(s, prob.T)]["q"] is not None,
              "t=T-1 的 q_T 应为追索变量")
        check(np.allclose(lp2.dec[(s, prob.T)]["q"],
                          lp2.dec[(s, prob.T)]["q"]), "q_T 变量")
    check(r2.stats.window == 2, "t=T-1 window")

    lp3 = TwoStageLP(prob, state, prob.T,
                     opt["pool"][:, -1:, :, :, :],
                     solver="copt", threads=1)
    r3 = lp3.solve()
    check(r3.stats.window == 1, "t=T window")
    check(np.allclose(r3.q_first, 0.0, atol=1e-9), "t=T qfirst=0")

    # 追索 LP：当期约束 + cap_end + h/2 I_out
    th = opt["pool"][0, 0]
    Nv = state.n[:, :, None] * th
    F = Nv[:, :, prob.G].sum(axis=1)
    q = res.q_first
    tau_in = res.tau_first
    rec = RecourseLP(prob, state, q, tau_in, F,
                     solver="copt", threads=1)
    rr = rec.solve()
    check(rr.y.shape == (NR, NR), "recourse y shape")
    check(np.allclose(rr.tau, tau_in, atol=TOL), "recourse tau=tau_in")
    check(abs(rr.tau.sum()) < TOL, "recourse tau_bal")
    check(np.all(state.I + rr.tau >= -TOL), "recourse tau_lo")
    check(np.all(state.I + rr.tau <= np.asarray(prob.Imax) + TOL),
          "recourse tau_hi")
    outflow = rr.y.sum(axis=0)
    check(np.all(outflow <= state.I + tau_in + TOL), "recourse pas")
    check(np.all(rr.y.sum(axis=1)
                 <= prob.N - state.n.sum(axis=1) + F + TOL),
          "recourse dem")
    o_next = np.empty((K, NR))
    if K > 1:
        o_next[:K - 1] = state.o[1:, :]
    o_next[K - 1] = q
    check(np.all(rr.I_out + o_next.sum(axis=0)
                 <= np.asarray(prob.Imax) + TOL), "recourse cap_end")
    check(np.allclose(rr.I_out, rr.S_end, atol=TOL),
          "recourse I_out")
    check(np.allclose(rr.S_end,
                      state.I + tau_in - outflow + state.o[0],
                      atol=TOL), "recourse S_end")

    # 满库存状态：t 期 qfirst 为 0
    o_full = np.zeros_like(o0)
    state_full = State(np.asarray(prob.Imax, dtype=float), o_full, prob.n0)
    state_full.validate(prob.N_R, prob.G, prob.L)
    lp4 = TwoStageLP(prob, state_full, 1, opt["pool"],
                     solver="copt", threads=1)
    r4 = lp4.solve()
    check(np.allclose(r4.q_first, 0.0, atol=1e-6),
          f"满库存 qfirst={r4.q_first}")

    print(f"_t_lp: {PASS} checks passed")
    print(f"  q_first={np.round(res.q_first, 4).tolist()} "
          f"obj={res.objective:.2f}  vars={lp.stats.n_vars} "
          f"eq={lp.stats.n_eq} le={lp.stats.n_le}")


if __name__ == "__main__":
    main()
