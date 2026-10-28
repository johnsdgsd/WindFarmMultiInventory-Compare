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
from ss_static.simulator import (Simulator, audit_result,  # noqa: E402
                                 cost_terms_max_dev)
from ss_static.policy import SSStaticPolicy  # noqa: E402

sys.stdout.reconfigure(errors="replace")


def main():
    cfg = load_compare_config()
    prob = load_problem(cfg)
    pcfg = load_policy_config(PROJECT)
    base = np.load(cfg.baseline_npz)
    scene = np.asarray(base["scene"][:20], dtype=np.int16)
    pol = SSStaticPolicy(prob, [0.0] * prob.N_R, [1.0] * prob.N_R,
                         dict(pcfg["policy"]))
    res = Simulator(prob).simulate(pol, scene, record=True)
    audit, bad = audit_result(prob, res, tol=cfg.tol, state_tol=cfg.state_tol)
    if bad:
        raise AssertionError(f"审计违规: {bad}")
    dev = cost_terms_max_dev(res)
    if dev > 1e-6:
        raise AssertionError(f"五项和 vs 总成本偏差 {dev}")

    Ly = prob.layout
    q0 = np.asarray(res.arrays["x_in"][:, 0, Ly.q_slice()], float)
    q_out = np.asarray(res.arrays["q"], float)
    expected = prob.C * (q0.sum(axis=1) + q_out[:, :-1, :].sum(axis=(1, 2)))
    actual = np.asarray(res.arrays["terms"][:, :, 0], float).sum(axis=1)
    proc_dev = float(np.max(np.abs(expected - actual)))
    if proc_dev > 1e-6:
        raise AssertionError(f"采购成本恒等式偏差 {proc_dev}")
    if float(np.max(np.abs(np.asarray(res.arrays["taup"])))) > 1e-12:
        raise AssertionError("taup 不为 0")
    if float(np.max(np.abs(np.asarray(res.arrays["taum"])))) > 1e-12:
        raise AssertionError("taum 不为 0")
    total_dev = float(np.max(np.abs(
        res.totals - np.asarray(res.arrays["terms"]).sum(axis=(1, 2)))))
    if total_dev > 1e-6:
        raise AssertionError(f"totals vs terms 偏差 {total_dev}")

    print("_t_audit: all checks passed")
    print(f"  审计项最大违约: {max(audit.values()):.3e}")
    print(f"  五项和偏差: {dev:.3e}；采购恒等式偏差: {proc_dev:.3e}")
    print(f"  20 路径均值: {res.mean:.4f}；q0 合计/路径: "
          f"{q0.sum() / res.P:.4f}；q_T max: {np.abs(q_out[:, -1, :]).max():.3e}")


if __name__ == "__main__":
    main()
