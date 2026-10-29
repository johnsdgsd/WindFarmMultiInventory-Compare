"""环境自检：新时序状态转移、容量、被动优先级、奖励恒等式、审计。"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.config import load_compare_config  # noqa: E402
from common.model import load_problem  # noqa: E402
from ppo_taufirst.env import (TERM_NAMES, TauFirstEnv, ThetaSampler,  # noqa: E402
                              audit_rollout, priority_fulfillment)


def _cfg_path():
    env = os.environ.get("COMPARE_CONFIG")
    if env:
        return env
    return ROOT / "config" / "compare_T60_base204105.toml"


def _no_failure_theta(prob):
    th = np.zeros((prob.N_R, prob.G, prob.G + 1), dtype=float)
    for i in range(prob.G):
        th[:, i, i] = 1.0
    return th


def _rollout_zero(prob, scene_idx: int = 0):
    T, NR, G = int(prob.T), int(prob.N_R), int(prob.G)
    th = np.asarray(prob.pool[int(scene_idx)][1], dtype=float)
    env = TauFirstEnv(prob, theta_path=np.stack([th] * T, axis=0))
    env.reset()
    keys = ("I_in", "o_in", "n_in", "q", "tau", "taup", "taum", "y",
            "I_out", "o_next", "n_out", "F", "W_in", "W_out", "cost",
            "terms", "reward")
    shapes = {
        "I_in": (1, T, NR), "o_in": (1, T, 2, NR),
        "n_in": (1, T, NR, G), "q": (1, T, NR), "tau": (1, T, NR),
        "taup": (1, T, NR), "taum": (1, T, NR),
        "y": (1, T, NR, NR), "I_out": (1, T, NR),
        "o_next": (1, T, 2, NR), "n_out": (1, T, NR, G),
        "F": (1, T, NR), "W_in": (1, T, NR), "W_out": (1, T, NR),
        "cost": (1, T), "terms": (1, T, 5), "reward": (1, T),
    }
    a = {k: np.zeros(shapes[k]) for k in keys}
    a["scene"] = np.full((1, T), int(scene_idx), dtype=np.int16)
    for t in range(T):
        I0, o0, n0 = env.I.copy(), np.stack([env.o1, env.o2]), env.n.copy()
        _, reward, done, info = env.step(np.zeros(env.act_dim))
        a["I_in"][0, t] = I0
        a["o_in"][0, t] = o0
        a["n_in"][0, t] = n0
        a["q"][0, t] = info["q"]
        a["tau"][0, t] = info["tau"]
        a["taup"][0, t] = np.maximum(info["tau"], 0.0)
        a["taum"][0, t] = np.maximum(-info["tau"], 0.0)
        a["y"][0, t] = info["y"]
        a["I_out"][0, t] = info["I_next"]
        a["o_next"][0, t] = info["o_next"]
        a["n_out"][0, t] = info["n_next"]
        a["F"][0, t] = info["F_new"]
        a["W_in"][0, t] = info["W_in"]
        a["W_out"][0, t] = info["W_out"]
        a["cost"][0, t] = info["cost"]
        a["terms"][0, t] = info["terms"]
        a["reward"][0, t] = reward
        if t < T - 1 and done:
            raise AssertionError("episode 提前结束")
    if not done:
        raise AssertionError("episode 未在 T 结束")
    return a


def main():
    cfg = load_compare_config(_cfg_path())
    prob = load_problem(cfg)
    if (prob.T, prob.G, prob.N_R, prob.L) != (60, 4, 5, 3):
        raise AssertionError("维度不是 T60/G4/N5/L3")

    # 1) obs 维度与归一化
    env = TauFirstEnv(prob, seed=123)
    # 被动调拨优先级：与 problem_data.priority 完全一致
    for r in range(prob.N_R):
        expected = [int(m) - 1 for m in prob.priority[r + 1]]
        if env.priority[r] != expected:
            raise AssertionError(
                f"priority[{r + 1}]={env.priority[r]} != {expected}")
        got = [m for _, rr, m in env.edges if rr == r]
        want = [m for m in expected if m != r]
        if got != want:
            raise AssertionError(
                f"仓库 {r + 1} 跨仓顺序 {got} != priority {want}")
    if env.priority[0] != [0, 1, 2, 3, 4]:
        raise AssertionError(f"仓库1 priority={env.priority[0]}")
    if env.priority[1] != [1, 0, 2, 3, 4]:
        raise AssertionError(f"仓库2 priority={env.priority[1]}")
    obs = env.reset()
    if obs.shape != (35,):
        raise AssertionError(f"obs shape {obs.shape}")
    if not np.allclose(obs[:5], prob.I0 / prob.Imax):
        raise AssertionError("I/Imax 归一化不符")
    if not np.allclose(obs[5:25], (prob.n0 / prob.N[:, None]).ravel()):
        raise AssertionError("n/N 归一化不符")
    if not np.allclose(obs[25:30], prob.o0[0] / prob.Qmax):
        raise AssertionError("o1/Qmax 归一化不符")
    if not np.allclose(obs[30:35], prob.o0[1] / prob.Qmax):
        raise AssertionError("o2/Qmax 归一化不符")

    # 2) 初始状态零动作：tau=0，q=0，cost=0（初始无库存、无停机）
    obs2, r, done, info = env.step(np.zeros(env.act_dim))
    if not np.allclose(info["tau"], 0.0):
        raise AssertionError(f"零动作 tau≠0: {info['tau']}")
    if not np.allclose(info["q"], 0.0):
        raise AssertionError(f"零动作 q≠0: {info['q']}")
    if abs(info["cost"]) > 1e-9:
        raise AssertionError(f"初始零动作成本应为 0，实际 {info['cost']}")
    if not np.allclose(r, -env.reward_scale * info["cost"]):
        raise AssertionError("reward 缩放不符")
    if not np.allclose(info["terms"].sum(), info["cost"]):
        raise AssertionError("terms 与 cost 不一致")
    if not np.allclose(info["I_next"], 0.0):
        raise AssertionError(f"初始 I_next≠0: {info['I_next']}")
    if done:
        raise AssertionError("第 1 期不应结束")

    # 3) 极值动作投影：tau 归零/有界，q 有界
    for scale in (10.0, -10.0, 1e6):
        env.reset()
        _, _, _, inf = env.step(np.full(env.act_dim, scale))
        tau = inf["tau"]
        q = inf["q"]
        if abs(float(tau.sum())) > 1e-8:
            raise AssertionError(f"Σtau={tau.sum()}")
        if np.any(prob.I0 + tau < -1e-9):
            raise AssertionError("I+tau<0")
        if np.any(prob.I0 + tau > prob.Imax + 1e-9):
            raise AssertionError("I+tau>Imax")
        if np.any(q < -1e-9) or np.any(q > prob.Qmax + 1e-9):
            raise AssertionError(f"q 越界: {q}")
        cap = (prob.I0 + tau + prob.o0.sum(axis=0) + q - prob.Imax).max()
        if cap > 1e-8:
            raise AssertionError(f"cap_init 违约 {cap}")
        if not inf["projection"]["tau_l1"] >= 0:
            raise AssertionError("投影诊断缺失")

    # 3b) 末期 q_T=0 约定（动作 q 原始量被强制投影到 0）
    env.reset()
    env.set_state(prob.I0, o=prob.o0, n=prob.n0, t=prob.T - 1)
    _, _, done_T, inf_T = env.step(np.full(env.act_dim, 5.0))
    if not done_T:
        raise AssertionError("末期 step 后应 done")
    if np.max(np.abs(inf_T["q"])) > 1e-12:
        raise AssertionError(f"q_T 未强制为 0: {inf_T['q']}")
    if not inf_T["projection"]["terminal_q_zero"]:
        raise AssertionError("terminal_q_zero 标记缺失")

    # 4) 容量边界：I 已顶到 Imax - Σo
    env.reset()
    o = np.zeros((2, prob.N_R))
    env.set_state(prob.Imax.copy(), o=o, n=prob.n0)
    _, _, _, inf = env.step(np.full(env.act_dim, 5.0))
    if not np.allclose(inf["tau"], 0.0, atol=1e-9):
        raise AssertionError(f"容量满时 tau 应为 0: {inf['tau']}")
    if not np.allclose(inf["q"], 0.0, atol=1e-9):
        raise AssertionError(f"容量满时 q 应为 0: {inf['q']}")

    # 5) 被动调拨优先级：本地先，跨仓按 tc_route 升序
    th = _no_failure_theta(prob)
    env2 = TauFirstEnv(prob, theta_path=np.stack([th] * prob.T, axis=0))
    env2.reset()
    W = np.array([8.0, 0.0, 0.0, 0.0, 0.0])
    n = np.zeros((prob.N_R, prob.G))
    n[:, prob.G - 1] = prob.N - W
    I = np.array([0.0, 0.0, 3.0, 3.0, 3.0])
    env2.set_state(I, o=np.zeros((2, prob.N_R)), n=n)
    _, _, _, inf2 = env2.step(np.zeros(env2.act_dim))
    y = inf2["y"]
    expected = np.zeros_like(y)
    expected[0, 2] = 3.0
    expected[0, 3] = 3.0
    expected[0, 4] = 2.0
    if not np.allclose(y, expected):
        raise AssertionError(f"优先级调拨不符:\n{y}\nexpected=\n{expected}")
    if not np.allclose(inf2["D"], W):
        raise AssertionError(f"D={inf2['D']} != {W}")
    if not np.allclose(inf2["I_next"], np.array([0.0, 0.0, 0.0, 0.0, 1.0])):
        raise AssertionError(f"I_next={inf2['I_next']}")
    terms = inf2["terms"]
    exp_terms = np.array([
        0.0,
        prob.h / 2.0 * (I.sum() + inf2["I_next"].sum()),
        prob.Cs * W.sum(),
        0.0,
        (prob.tc_route * expected).sum(),
    ])
    if not np.allclose(terms, exp_terms):
        raise AssertionError(f"terms={terms}, expected={exp_terms}")
    if abs(inf2["cost"] - exp_terms.sum()) > 1e-9:
        raise AssertionError("成本恒等式失败")

    # 6) ThetaSampler 与 rolling_horizon.scenarios.sample_pool 同种子逐位一致
    from rolling_horizon.scenarios import sample_pool
    ref = sample_pool(
        prob.lambda0, prob.beta_cov, prob.z_bar, prob.cov,
        prob.N_R, prob.G, prob.dirichlet_s, prob.p_digits, 1, 123)[0][1]
    ours = ThetaSampler(prob, seed=123).sample()
    if not np.allclose(ref, ours, atol=0.0, rtol=0.0):
        raise AssertionError("ThetaSampler 与 sample_pool 不一致")
    s1 = ThetaSampler(prob, seed=7)
    s2 = ThetaSampler(prob, seed=7)
    for _ in range(3):
        if not np.allclose(s1.sample(), s2.sample(), atol=0.0, rtol=0.0):
            raise AssertionError("ThetaSampler 同种子不可复现")
    th2 = s1.sample()
    if th2.shape != (prob.N_R, prob.G, prob.G + 1):
        raise AssertionError("Theta shape 异常")
    if np.max(np.abs(th2.sum(axis=2) - 1.0)) > 1e-12:
        raise AssertionError("Theta 行和不为 1")
    if np.any(th2 < 0):
        raise AssertionError("Theta 负值")

    # 7) 真实 scene 的零动作 rollout + 审计
    arrays = _rollout_zero(prob, scene_idx=0)
    audit, bad = audit_rollout(
        prob, arrays, tol=float(cfg.tol), state_tol=float(cfg.state_tol),
        reward_scale=1e-3)
    if bad:
        raise AssertionError(f"审计失败: {bad}\n{audit}")
    if audit["terms_sum"] > 1e-12 or audit["reward_identity"] > 1e-12:
        raise AssertionError("成本/奖励恒等式不精确")

    # 8) 篡改检测
    bad_y = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in arrays.items()}
    bad_y["y"][0, 0, 0, 1] += 1.0
    _, bad2 = audit_rollout(prob, bad_y, tol=float(cfg.tol),
                            state_tol=float(cfg.state_tol), reward_scale=1e-3)
    if "priority" not in bad2 and "pas" not in bad2:
        raise AssertionError(f"篡改 y 未被审计发现: {bad2}")
    bad_q = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in arrays.items()}
    bad_q["q"][0, 0, 0] += 100.0
    _, bad3 = audit_rollout(prob, bad_q, tol=float(cfg.tol),
                            state_tol=float(cfg.state_tol), reward_scale=1e-3)
    if not any(k in bad3 for k in ("q_hi", "cap_init")):
        raise AssertionError(f"篡改 q 未被审计发现: {bad3}")
    bad_r = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in arrays.items()}
    bad_r["reward"][0, 0] += 1.0
    _, bad4 = audit_rollout(prob, bad_r, tol=float(cfg.tol),
                            state_tol=float(cfg.state_tol), reward_scale=1e-3)
    if "reward_identity" not in bad4:
        raise AssertionError(f"篡改 reward 未被审计发现: {bad4}")

    print("_t_env: all checks passed")
    print(f"  T={prob.T} N_R={prob.N_R} G={prob.G} L={prob.L} "
          f"obs_dim={env.obs_dim} act_dim={env.act_dim}")
    print(f"  audit max terms_sum={audit['terms_sum']:.3e} "
          f"reward_identity={audit['reward_identity']:.3e} "
          f"priority={audit['priority']:.3e}")
    print(f"  terms={tuple(TERM_NAMES)}")


if __name__ == "__main__":
    main()
