"""PPO 自检：前向/更新/GAE/保存加载/种子复现（不依赖真实模型）。"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ppo_taufirst.ppo import (ActorCritic, PPOConfig, PPOTrainer,  # noqa: E402
                              deterministic_actions, load_policy,
                              save_policy)


class ToyEnv:
    obs_dim = 4
    act_dim = 2
    reward_scale = 0.1

    def __init__(self, horizon: int = 5):
        self.T = int(horizon)
        self.t = 0

    def reset(self):
        self.t = 0
        self.episode_cost = 0.0
        self.episode_return = 0.0
        return np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(self.act_dim)
        cost = float(np.sum((a - 0.5) ** 2)) + 0.1
        reward = -self.reward_scale * cost
        self.t += 1
        self.episode_cost += cost
        self.episode_return += reward
        done = self.t >= self.T
        info = {
            "cost": cost,
            "episode_cost": self.episode_cost if done else None,
            "episode_return": self.episode_return if done else None,
        }
        obs = np.full(self.obs_dim, self.t / self.T, dtype=np.float32)
        return obs, reward, done, info


def _cfg():
    return PPOConfig(
        hidden=(16, 16), gamma=0.99, gae_lambda=0.95, clip=0.2,
        lr=1e-3, entropy_coef=0.01, value_coef=0.5,
        max_grad_norm=0.5, rollout_steps=64, minibatch=16, epochs=2,
        log_std_init=-0.5, normalize_adv=True)


def main():
    torch.manual_seed(0)
    obs = torch.randn(3, 4)
    model = ActorCritic(4, 2, (16, 16), -0.5)
    action, logp, value, entropy = model.get_action(obs)
    if tuple(action.shape) != (3, 2):
        raise AssertionError(f"action shape {action.shape}")
    if tuple(logp.shape) != (3,) or tuple(value.shape) != (3,) \
            or tuple(entropy.shape) != (3,):
        raise AssertionError("logp/value/entropy shape 异常")
    lp2, ent2, val2 = model.evaluate_actions(obs, action)
    if not torch.allclose(logp, lp2):
        raise AssertionError(f"logp 重算不一致: {logp} vs {lp2}")
    if not torch.allclose(value, val2):
        raise AssertionError("value 重算不一致")
    if not torch.all(torch.isfinite(logp)) or not torch.all(torch.isfinite(val2)):
        raise AssertionError("logp/value 含非有限值")

    # 训练 smoke：有限 loss、参数确实更新
    env = ToyEnv()
    trainer = PPOTrainer(env, _cfg(), seed=11)
    before = {k: v.detach().clone()
              for k, v in trainer.model.state_dict().items()}
    summary = trainer.train(256)
    if summary["steps_done"] != 256:
        raise AssertionError(f"steps_done={summary['steps_done']}")
    if not trainer.history:
        raise AssertionError("没有 training history")
    for row in trainer.history:
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                    "clip_frac", "grad_norm"):
            if not np.isfinite(row.get(key, np.nan)):
                raise AssertionError(f"history {key} 非有限: {row}")
    changed = any(not torch.allclose(before[k], v)
                  for k, v in trainer.model.state_dict().items())
    if not changed:
        raise AssertionError("训练后参数没有变化")

    # 种子复现：两个 trainer 同样 seed，前 8 步动作一致
    t1 = PPOTrainer(ToyEnv(), _cfg(), seed=99)
    t2 = PPOTrainer(ToyEnv(), _cfg(), seed=99)
    r1 = t1.collect(8)
    r2 = t2.collect(8)
    if not np.allclose(r1.actions, r2.actions, atol=0.0, rtol=0.0):
        raise AssertionError("同 seed 的采样动作不一致")

    # 保存/加载
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "policy.pt"
        meta = {"kind": "ppo_taufirst", "note": "toy"}
        save_policy(path, trainer.model, meta)
        loaded, meta2 = load_policy(path)
        if meta2.get("note") != "toy":
            raise AssertionError("meta 未保存")
        obs_b = np.random.randn(5, 4).astype(np.float32)
        a1 = deterministic_actions(trainer.model, obs_b)
        a2 = deterministic_actions(loaded, obs_b)
        if not np.allclose(a1, a2, atol=0.0, rtol=0.0):
            raise AssertionError("保存/加载后确定性动作不一致")

    print("_t_ppo: all checks passed")
    print(f"  updates={summary['updates']} episodes={summary['episodes']} "
          f"wall={summary['wall_s']:.2f}s "
          f"last_ep_cost={summary['mean_episode_cost_last10']:.3f}")
    print(f"  final pi_loss={trainer.history[-1]['policy_loss']:.4f} "
          f"v_loss={trainer.history[-1]['value_loss']:.4f} "
          f"kl={trainer.history[-1]['approx_kl']:.5f}")


if __name__ == "__main__":
    main()
