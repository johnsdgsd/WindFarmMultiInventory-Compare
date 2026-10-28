from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import json  # noqa: E402
import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from common.config import load_compare_config, load_policy_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ss_static.active import ThresholdBalanceRule  # noqa: E402
from ss_static.calibrate import calibrate  # noqa: E402
from ss_static.evaluation import SSEvaluator  # noqa: E402
from ss_static.optimize import ParamSpace, run_algorithm  # noqa: E402
from ss_static.policy import SSStaticPolicy  # noqa: E402
from ss_static.simulator import generate_scene_paths, audit_result  # noqa: E402

sys.stdout.reconfigure(errors="replace")

PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def _fake_prob(K=5, G=4, L=3, T=60, Imax=None):
    Imax = np.asarray([12.0, 5.0, 20.0, 15.0, 8.0]) if Imax is None \
        else np.asarray(Imax, float)
    th = np.zeros((K, G, G + 1))
    return SimpleNamespace(
        N_R=K, G=G, L=L, T=T, Imax=Imax,
        pool=[(None, th.copy())], atc=5.0,
    )


def _rule(alpha=1.0, stop=0, target=5.0):
    prob = _fake_prob()
    opts = {
        "active_transfer": True, "active_rule": "threshold_balance",
        "active_target": target, "active_threshold_min": target,
        "active_fraction": alpha, "active_stop_offset": stop,
        "active_priority": "deficit", "active_forecast": "pool_mean",
    }
    return prob, ThresholdBalanceRule(prob, opts)


def _state(row, P=1):
    X = np.asarray(row, float).reshape(P, -1)
    K = X.shape[1]
    G = 4
    L = 3
    return (X, np.zeros_like(X), np.zeros((P, K, G)),
            np.zeros((P, L - 1, K)), np.zeros_like(X))


def main():
    global PASS
    # 1) 规则：目标 5，源/目标、净额、守恒
    prob, rule = _rule()
    X, W, n, o, q = _state([10, 0, 0, 0, 0])
    tau = rule.decide(1, X, W, n, o, q, None, np.full(5, 5.0))
    check(np.allclose(tau, [[-5, 5, 0, 0, 0]]),
          f"基本调拨 tau={tau}")
    check(abs(float(tau.sum())) < 1e-12, "tau 不守恒")
    check(np.all(X + tau >= -1e-9), "调拨后负库存")

    # 2) 门限：X=5 不触发；X>h 才调出
    X, W, n, o, q = _state([5, 0, 0, 0, 0])
    tau = rule.decide(1, X, W, n, o, q, None, np.full(5, 5.0))
    check(float(np.abs(tau).sum()) < 1e-12, "X=5 不应调出")
    h6 = np.asarray([6.0, 5.0, 5.0, 5.0, 5.0])
    X, W, n, o, q = _state([5.5, 0, 0, 0, 0])
    tau = rule.decide(1, X, W, n, o, q, None, h6)
    check(float(np.abs(tau).sum()) < 1e-12, "X<h 不应调出")
    X, W, n, o, q = _state([6.5, 0, 0, 0, 0])
    tau = rule.decide(1, X, W, n, o, q, None, h6)
    check(abs(float(tau[0, 0] + 1.5)) < 1e-12
          and abs(float(tau[0, 1] - 1.5)) < 1e-12,
          f"h=6 的调出量不对: {tau}")

    # 3) h=Imax：该仓关闭
    X, W, n, o, q = _state([10, 0, 0, 0, 0])
    tau = rule.decide(1, X, W, n, o, q, None,
                      np.asarray(prob.Imax, float))
    check(float(np.abs(tau).sum()) < 1e-12, "h=Imax 应关闭")

    # 4) alpha=0.5
    _, rule_half = _rule(alpha=0.5)
    X, W, n, o, q = _state([10, 0, 0, 0, 0])
    tau = rule_half.decide(1, X, W, n, o, q, None, np.full(5, 5.0))
    check(np.allclose(tau, [[-2.5, 2.5, 0, 0, 0]]),
          f"alpha=0.5 tau={tau}")

    # 5) 多单位/部分缺口：只有仓 2 缺 0.1，源 5
    X, W, n, o, q = _state([10, 4.9, 5, 5, 5])
    tau = rule.decide(1, X, W, n, o, q, None, np.full(5, 5.0))
    check(abs(float(tau[0, 0] + 0.1)) < 1e-9
          and abs(float(tau[0, 1] - 0.1)) < 1e-9,
          f"部分缺口 tau={tau}")

    # 6) 硬容量：目标仓 Imax=5、已有 3、下一期在途 1.5、订单 0.3，
    #    剩余容量只有 0.2，不能把 2 全部调进去
    p2 = _fake_prob(Imax=[12, 5, 20, 15, 8])
    opts2 = {
        "active_transfer": True, "active_rule": "threshold_balance",
        "active_target": 5.0, "active_threshold_min": 5.0,
        "active_fraction": 1.0, "active_stop_offset": 0,
        "active_priority": "deficit", "active_forecast": "pool_mean",
    }
    r2 = ThresholdBalanceRule(p2, opts2)
    X, W, n, o, q = _state([10, 3, 5, 5, 5])
    o = o.copy()
    o[0, 0, 1] = 1.5     # o_next 的 destination 1
    q = q.copy()
    q[0, 1] = 0.3
    tau = r2.decide(1, X, W, n, o, q, None, np.full(5, 5.0))
    # cap = 5 - 3 - 1.5 - 0.3 = 0.2
    check(abs(float(tau[0, 1]) - 0.2) < 1e-9,
          f"目标容量未生效 tau={tau}")

    # 7) 末期：最后 2 期不调
    _, rule_stop = _rule(stop=2)
    X, W, n, o, q = _state([10, 0, 0, 0, 0])
    for t in (59, 60):
        tau = rule_stop.decide(t, X, W, n, o, q, None, np.full(5, 5.0))
        check(float(np.abs(tau).sum()) < 1e-12, f"t={t} 应停调")

    # 8) 实际模型：active 评估、审计、I_out 恒等式、末期零 tau
    # active 的真实模型测试固定用新 T60 204105（该问题有足够库存触发调拨）；
    # 不受外部 COMPARE_CONFIG 影响。
    cfg = load_compare_config("config/compare_T60_base204105.toml")
    prob_real = load_problem(cfg)
    pcfg = load_policy_config(PROJECT, "ss_static/config/policy_active.toml")
    opts = dict(pcfg["policy"])
    ev = SSEvaluator(prob_real, opts)
    paths = generate_scene_paths(200, cfg.train_seed,
                                 cfg.calib_chunk_size, len(prob_real.pool),
                                 prob_real.T)
    imax = np.asarray(prob_real.Imax, float)
    theta = np.concatenate([imax, imax, np.full(prob_real.N_R, 5.0)])
    r = ev.evaluate(theta, paths, record=True)
    check(r.behavior["active_transfer_volume"] > 0.0,
          "active 评估没有任何调拨，测试会空跑")
    # I_out = S_end + tau，应与下一期记录的 I_in 一致（t=1..T-1）
    iout = (r.arrays["S_end"][:, :-1, :]
            + r.arrays["taup"][:, :-1, :]
            - r.arrays["taum"][:, :-1, :])
    check(np.allclose(r.arrays["x_in"][:, 1:, :prob_real.N_R], iout,
                      atol=cfg.state_tol), "I_out 与下一期 I 不一致")
    check(float(np.max(r.arrays["taup"][:, -1]
                       + r.arrays["taum"][:, -1])) < 1e-12,
          "末期 tau 非零")
    audit, bad = audit_result(prob_real, r, tol=cfg.tol,
                              state_tol=cfg.state_tol)
    check(not bad, f"active 审计违规 {bad}")
    active_cost = float(r.step_terms[:, :, 3].sum(axis=1).mean())
    expected_cost = float(
        prob_real.atc / 2.0
        * (r.arrays["taup"] + r.arrays["taum"]).sum(axis=(1, 2)).mean())
    check(abs(active_cost - expected_cost) < 1e-9,
          "active cost 恒等式不成立")
    check(abs(float(np.max(np.abs(
        r.step_terms.sum(axis=2).sum(axis=1) - r.path_costs)))) < 1e-9,
        "五项和 != 总成本")

    # 9) active 与 no-active：q/y/n/W 相同，仅 I_out/持有/active 变
    o0 = dict(pcfg["policy"])
    o0["active_transfer"] = False
    ev0 = SSEvaluator(prob_real, o0)
    theta0 = np.concatenate([imax, imax])
    r0 = ev0.evaluate(theta0, paths, record=True)
    # 首期开口状态相同：active 不应改变当期的 q/y/n/W；之后状态会分叉
    for key in ("q", "y", "n_out", "F", "W_in", "W_out", "S_end"):
        check(np.allclose(r.arrays[key][:, 0], r0.arrays[key][:, 0],
                          atol=1e-9),
              f"active 不应改变首期 {key}")
    check(np.allclose(r0.arrays["taup"], 0.0)
          and np.allclose(r0.arrays["taum"], 0.0), "no-active tau 非零")

    # 10) policy JSON round-trip h
    pol = SSStaticPolicy(prob_real, imax, imax, opts,
                         h=np.full(prob_real.N_R, 5.0))
    data = pol.params()
    pol2 = SSStaticPolicy.from_dict(prob_real, {
        **data, "problem": {"T": prob_real.T}})
    check(np.allclose(pol2.h, pol.h), "policy h round-trip")
    check(pol2.active_enabled, "policy active_enabled")

    # 11) 3K 参数空间
    space = ParamSpace(imax, active=True, active_target=5.0,
                       active_min=5.0)
    check(space.dim == 3 * prob_real.N_R, f"active dim={space.dim}")
    th = space.clip(np.concatenate([imax, imax,
                                    np.full(prob_real.N_R, 5.0)]))
    check(np.all(th[2 * prob_real.N_R:] >= 5.0 - 1e-9), "h 下界")
    rt = space.from_u(space.to_u(th))
    check(float(np.max(np.abs(rt - th))) < 1e-6, "active u round-trip")
    samp = space.sample(np.random.default_rng(1), 10)
    check(samp.shape == (10, 3 * prob_real.N_R), "active sample shape")

    # 12) no-active 回归（仅在 base 204105 配置下）
    if "base204105" in cfg.config_path.name:
        pol_path = (ROOT / "results" / "ss_static"
                    / "ss_static_T60_204105_calib_20261027_130616"
                    / "policy.json")
        if pol_path.is_file():
            data = json.loads(pol_path.read_text(encoding="utf-8"))
            p0 = SSStaticPolicy.from_dict(prob_real, data)
            ev0 = SSEvaluator(prob_real, p0.options)
            th0 = np.concatenate([p0.s, p0.S])
            from common.baseline import load_baseline
            base = load_baseline(cfg, prob_real)
            r0 = ev0.evaluate(th0, base.scene, record=False)
            check(abs(r0.cost - 41394.9455) < 1e-3,
                  f"no-active 回归 {r0.cost} != 41394.9455")

    # 13) active quick 标定：mode/h/归档
    with tempfile.TemporaryDirectory() as td:
        d = calibrate(PROJECT, tag="active_test", quick=True,
                      archive_root=td,
                      policy_config="ss_static/config/policy_active.toml")
        data = json.loads((Path(d) / "policy.json").read_text(
            encoding="utf-8"))
        check(data["calibration"]["mode_tau"] == 1, "active mode_tau != 1")
        check(data["calibration"]["active_transfer"] is True,
              "active_transfer 未记录")
        h_opt = np.asarray(data.get("active_thresholds", []), float)
        check(h_opt.size == prob_real.N_R
              and np.all(h_opt >= 5.0 - 1e-9)
              and np.all(h_opt <= imax + 1e-9), f"active h={h_opt}")
        run = json.loads((Path(d) / "run.json").read_text(encoding="utf-8"))
        check(run["status"] == "completed", "active run 未完成")
        check((Path(d) / "calibration.md").is_file(), "active calibration.md")

    print(f"_t_active: {PASS} checks passed")
    print(f"  active volume (200 paths)={r.behavior['active_transfer_volume']:.3f} "
          f"cost={r.cost:.2f}")


if __name__ == "__main__":
    main()
