"""s,S 新时序自检：无 q_in / 无虚拟 q0；q_t 进入下期在途最后一层。

替换旧的 `_t_lp_equiv.py`。旧测试把 s,S 对齐到主模型的 q_in/q_out
两套变量；新口径明确没有 q_in，因此主模型 LP 对拍不再适用。
"""
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

from common.config import load_compare_config, load_policy_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ss_static.policy import SSStaticPolicy  # noqa: E402
from ss_static.simulator import (Simulator, audit_result,  # noqa: E402
                                 cost_terms_max_dev)

sys.stdout.reconfigure(errors="replace")
PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def main():
    global PASS
    cfg = load_compare_config()
    prob = load_problem(cfg)
    pcfg = load_policy_config(PROJECT)
    base = np.load(cfg.baseline_npz)
    scene = np.asarray(base["scene"][:20], dtype=np.int16)
    pol = SSStaticPolicy(prob, [0.0] * prob.N_R, prob.Imax.astype(float),
                         dict(pcfg["policy"]))
    res = Simulator(prob).simulate(pol, scene, record=True)
    a = res.arrays
    Ly = prob.layout
    T, P, K, L = res.T, res.P, prob.N_R, prob.L

    # 1) 无 q_in：x_in 的 q 槽必须全 0
    check(float(np.max(np.abs(a["x_in"][:, :, Ly.q_slice()]))) < 1e-12,
          "x_in q 槽不是 0（仍存在 q_in）")
    def I_of(x):
        return np.asarray(x[..., Ly.off_I:Ly.off_I + Ly.n_I], float)

    def o_of(x):
        return np.asarray(x[..., Ly.off_o:Ly.off_o + Ly.n_o],
                          float).reshape(x.shape[:-1] + (L - 1, K))

    def n_of(x):
        return np.asarray(x[..., Ly.off_n:Ly.off_n + Ly.n_n],
                          float).reshape(x.shape[:-1] + (K, prob.G))

    # 2) 无虚拟 q0：I_in[0]/o_in[0] 直接是初始状态
    I0 = np.tile(np.asarray(prob.I0, float)[None, :], (P, 1))
    o0 = np.tile(np.asarray(prob.o0, float)[None, :, :], (P, 1, 1))
    check(np.allclose(I_of(a["x_in"][:, 0, :]), I0),
          "I_in[0] 不是初始 I")
    check(np.allclose(o_of(a["x_in"][:, 0, :]), o0),
          "o_in[0] 不是初始 o")

    q = np.asarray(a["q"], float)
    check(np.all(q >= -1e-9), "q<0")
    check(np.all(q <= np.asarray(prob.Qmax)[None, None, :] + 1e-9),
          "q 超过 Qmax")

    for t in range(T):
        x = a["x_in"][:, t, :]
        I = I_of(x)
        o = o_of(x)
        n = n_of(x)
        y = a["y"][:, t]
        S = a["S_end"][:, t]
        tau = a["taup"][:, t] - a["taum"][:, t]
        I_out = S + tau
        o_next = np.empty_like(o)
        if L > 2:
            o_next[:, :L - 2, :] = o[:, 1:, :]
        o_next[:, L - 2, :] = q[:, t]

        check(np.allclose(S, I - y.sum(axis=1) + o[:, 0, :],
                          atol=1e-8), f"t={t+1}: S_end 递推")
        check(np.allclose(I_out, a["S_end"][:, t] + tau, atol=1e-8),
              f"t={t+1}: I_out")
        # 当前期订货容量：I + Σo + q_t <= Imax
        check(np.all(I + o.sum(axis=1) + q[:, t]
                     <= np.asarray(prob.Imax)[None, :] + 1e-6),
              f"t={t+1}: q 超过当前容量")
        # 下一期状态容量：I_out + Σo_next <= Imax
        check(np.all(I_out + o_next.sum(axis=1)
                     <= np.asarray(prob.Imax)[None, :] + 1e-6),
              f"t={t+1}: 下一期容量越界")
        if t < T - 1:
            x_next = a["x_in"][:, t + 1, :]
            check(np.allclose(I_of(x_next), I_out, atol=1e-8),
                  f"t={t+1}: I 链")
            check(np.allclose(o_of(x_next), o_next, atol=1e-8),
                  f"t={t+1}: o 链（q_t 应进入最后一层）")
            check(np.allclose(n_of(x_next), a["n_out"][:, t], atol=1e-6),
                  f"t={t+1}: n 链")

    # 3) 采购成本：q_t 只对 t<T 计费，q_T 不计
    expected = prob.C * q[:, :T - 1, :].sum(axis=(1, 2))
    actual = a["terms"][:, :, 0].sum(axis=1)
    check(np.max(np.abs(expected - actual)) < 1e-6, "采购成本恒等式")
    check(abs(cost_terms_max_dev(res)) < 1e-9, "五项和 != 总成本")

    # 4) 审计（新时序映射）
    audit, bad = audit_result(prob, res, tol=cfg.tol, state_tol=cfg.state_tol)
    check(not bad, f"新时序审计违规 {bad}")

    print(f"_t_timing: {PASS} checks passed")
    print(f"  20 路径均值: {res.mean:.4f}")
    print(f"  q_T max: {np.abs(q[:, -1, :]).max():.3e}；"
          f"审计最大违约 {max(audit.values()):.3e}")


if __name__ == "__main__":
    main()
