"""自包含 PyTorch PPO（actor-critic、GAE、clip、entropy），不依赖 SB3。

设计：
- 2 层 MLP actor / critic，Gaussian 对角策略，state-independent log_std；
- 动作由 env 投影；PPO 只对原始 Gaussian 动作算 log-prob；
- CPU torch；种子可复现；policy.pt 保存/加载。
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math
import time
from dataclasses import dataclass, asdict, fields
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    hidden: tuple[int, int] = (128, 128)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    minibatch: int = 256
    epochs: int = 4
    log_std_init: float = -0.5
    normalize_adv: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if "hidden" in kw:
            kw["hidden"] = tuple(int(x) for x in kw["hidden"])
        return cls(**kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int,
         out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        lin = nn.Linear(last, int(h))
        nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = int(h)
    out = nn.Linear(last, out_dim)
    nn.init.orthogonal_(out.weight, gain=float(out_gain))
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """2 层 MLP 的 Gaussian actor + 标量 critic。"""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: tuple[int, int] = (128, 128),
                 log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.actor = _mlp(self.obs_dim, self.hidden, self.act_dim, 0.01)
        self.critic = _mlp(self.obs_dim, self.hidden, 1, 1.0)
        self.log_std = nn.Parameter(
            torch.full((self.act_dim,), float(log_std_init)))

    def _dist(self, obs):
        mean = self.actor(obs)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action(self, obs, deterministic: bool = False, generator=None):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        dist = self._dist(obs)
        if deterministic:
            action = dist.mean
        else:
            noise = torch.randn(
                dist.mean.shape, generator=generator,
                dtype=dist.mean.dtype, device=dist.mean.device)
            action = dist.mean + dist.stddev * noise
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return action, logp, value, entropy

    def evaluate_actions(self, obs, actions):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        dist = self._dist(obs)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(obs)
        return logp, entropy, value


@dataclass
class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_costs: list[float]
    episode_returns: list[float]


class PPOTrainer:
    """单环境 PPO；env 需提供 reset/step、obs_dim、act_dim。"""

    def __init__(self, env, config: PPOConfig, *, seed: int = 0,
                 device: str = "cpu", log=None):
        self.env = env
        self.config = config
        self.device = torch.device(device)
        self.seed = int(seed)
        self._log = log
        torch.manual_seed(self.seed)
        self.model = ActorCritic(
            env.obs_dim, env.act_dim, config.hidden,
            config.log_std_init).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(config.lr), eps=1e-5)
        self.action_gen = torch.Generator().manual_seed(self.seed + 2)
        self.perm_gen = torch.Generator().manual_seed(self.seed + 1)
        self.obs = env.reset()
        self.history: list[dict] = []
        self.episode_history: list[dict] = []
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_costs_all: list[float] = []
        self.t0 = time.perf_counter()

    # ------------------------------------------------------------- 日志
    def _log_line(self, msg: str):
        if self._log is not None:
            self._log(msg)

    def _obs_tensor(self, obs):
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------- 采样
    def collect(self, n_steps: int) -> Rollout:
        cfg = self.config
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = \
            [], [], [], [], [], []
        ep_costs: list[float] = []
        ep_returns: list[float] = []
        obs = self.obs
        last_done = True
        for _ in range(int(n_steps)):
            obs_t = self._obs_tensor(obs)
            with torch.no_grad():
                action, logp, value, _ = self.model.get_action(
                    obs_t, generator=self.action_gen)
            act_np = action.detach().cpu().numpy().astype(np.float64)
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(act_np)
            logp_buf.append(float(logp.item()))
            val_buf.append(float(value.item()))
            next_obs, reward, done, info = self.env.step(act_np)
            rew_buf.append(float(reward))
            done_buf.append(bool(done))
            last_done = bool(done)
            if done:
                ep_cost = float(info.get("episode_cost", np.nan))
                ep_ret = float(info.get("episode_return", np.nan))
                ep_costs.append(ep_cost)
                ep_returns.append(ep_ret)
                self.episode_costs_all.append(ep_cost)
                self.total_episodes += 1
                rscale = float(getattr(self.env, "reward_scale", 1.0))
                self.episode_history.append({
                    "episode": int(self.total_episodes),
                    "steps": int(self.total_steps + 1),
                    "total_cost": ep_cost,
                    "cost": ep_cost,
                    "raw_reward": (-ep_cost
                                   if np.isfinite(ep_cost) else float("nan")),
                    "scaled_reward": (-rscale * ep_cost
                                      if np.isfinite(ep_cost)
                                      else float("nan")),
                })
                self.obs = self.env.reset()
                obs = self.obs
            else:
                self.obs = next_obs
                obs = next_obs
            self.total_steps += 1

        obs_arr = np.stack(obs_buf, axis=0)
        act_arr = np.stack(act_buf, axis=0)
        logp_arr = np.asarray(logp_buf, dtype=np.float64)
        rew_arr = np.asarray(rew_buf, dtype=np.float64)
        val_arr = np.asarray(val_buf, dtype=np.float64)
        done_arr = np.asarray(done_buf, dtype=np.float64)

        # GAE：done 时 bootstrap=0；rollout 截断时用 V(s_{t+1}) bootstrap
        with torch.no_grad():
            if last_done:
                last_val = 0.0
            else:
                last_val = float(self.model.value(
                    self._obs_tensor(self.obs)).item())
        adv = np.zeros_like(rew_arr)
        gae = 0.0
        vals_ext = np.append(val_arr, last_val)
        for t in range(len(rew_arr) - 1, -1, -1):
            nonterminal = 1.0 - done_arr[t]
            delta = rew_arr[t] + cfg.gamma * vals_ext[t + 1] * nonterminal \
                - vals_ext[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + val_arr
        return Rollout(
            obs=obs_arr, actions=act_arr, logp=logp_arr,
            rewards=rew_arr, values=val_arr, dones=done_arr,
            advantages=adv, returns=returns,
            episode_costs=ep_costs, episode_returns=ep_returns)

    # ------------------------------------------------------------- 更新
    def update(self, roll: Rollout) -> dict:
        cfg = self.config
        n = len(roll.rewards)
        if n < 2:
            return {"n": int(n), "skipped": 1}
        obs = torch.as_tensor(roll.obs, dtype=torch.float32,
                              device=self.device)
        actions = torch.as_tensor(roll.actions, dtype=torch.float32,
                                  device=self.device)
        old_logp = torch.as_tensor(roll.logp, dtype=torch.float32,
                                   device=self.device)
        adv = torch.as_tensor(roll.advantages, dtype=torch.float32,
                              device=self.device)
        returns = torch.as_tensor(roll.returns, dtype=torch.float32,
                                  device=self.device)
        if cfg.normalize_adv and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        mb = max(1, min(int(cfg.minibatch), n))
        stats = {
            "n": int(n), "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0,
            "grad_norm": 0.0, "updates": 0,
        }
        for _ in range(int(cfg.epochs)):
            perm = torch.randperm(n, generator=self.perm_gen).to(self.device)
            for start in range(0, n, mb):
                idx = perm[start:start + mb]
                new_logp, entropy, values = self.model.evaluate_actions(
                    obs[idx], actions[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1.0 - cfg.clip,
                                 1.0 + cfg.clip) * adv[idx]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = 0.5 * ((values - returns[idx]) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_mean)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_logp.detach() - old_logp[idx]
                    approx_kl = float(
                        ((torch.exp(log_ratio) - 1.0) - log_ratio).mean())
                    clip_frac = float(
                        ((ratio.detach() - 1.0).abs() > cfg.clip)
                        .float().mean())
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["grad_norm"] += float(grad_norm)
                stats["updates"] += 1
        k = max(1, stats["updates"])
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                    "clip_frac", "grad_norm"):
            stats[key] /= k
        with torch.no_grad():
            var_y = float(np.var(roll.returns))
            var_e = float(np.var(roll.returns - roll.values))
            stats["explained_var"] = 1.0 - var_e / var_y if var_y > 0 else 0.0
        # 实际优化的总损失（各 minibatch 系数固定，等价于分别取均值）
        stats["total_loss"] = (
            stats["policy_loss"]
            + cfg.value_coef * stats["value_loss"]
            - cfg.entropy_coef * stats["entropy"])
        return stats

    # ------------------------------------------------------------- 训练
    def train(self, total_steps: int, *, log_every: int = 1) -> dict:
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps 必须为正")
        start_steps = self.total_steps
        update_idx = 0
        while self.total_steps < total_steps:
            n = min(int(self.config.rollout_steps),
                    total_steps - self.total_steps)
            roll = self.collect(n)
            stats = self.update(roll)
            update_idx += 1
            rscale = float(getattr(self.env, "reward_scale", 1.0))
            last_cost = (float(roll.episode_costs[-1])
                         if roll.episode_costs else float("nan"))
            mean_cost = (float(np.mean(roll.episode_costs))
                         if roll.episode_costs else float("nan"))
            hist = {
                "update": update_idx,
                "steps": self.total_steps,
                "episodes": self.total_episodes,
                "rollout_steps": int(n),
                "wall_s": time.perf_counter() - self.t0,
                "mean_step_cost": float(-roll.rewards.mean() / rscale),
                "mean_step_reward": float(roll.rewards.mean()),
                "last_episode_cost": last_cost,
                "last_episode_reward": (-rscale * last_cost
                                        if np.isfinite(last_cost)
                                        else float("nan")),
                "mean_episode_cost": mean_cost,
                "mean_episode_reward": (-rscale * mean_cost
                                        if np.isfinite(mean_cost)
                                        else float("nan")),
                **stats,
            }
            self.history.append(hist)
            if log_every and update_idx % int(log_every) == 0:
                self._log_line(
                    f"[train] update={update_idx} "
                    f"episodes={self.total_episodes}/"
                    f"{total_steps // int(getattr(self.env, 'T', 60))} "
                    f"ep_cost={hist['mean_episode_cost']:.1f} "
                    f"policy_loss={stats['policy_loss']:.5f} "
                    f"value_loss={stats['value_loss']:.2f} "
                    f"wall={hist['wall_s']:.1f}s")
        wall = time.perf_counter() - self.t0
        recent = self.episode_costs_all[-10:]
        return {
            "requested_steps": total_steps,
            "steps_done": int(self.total_steps - start_steps),
            "total_steps": int(self.total_steps),
            "updates": int(update_idx),
            "episodes": int(self.total_episodes),
            "episodes_logged": int(len(self.episode_history)),
            "wall_s": float(wall),
            "fps": float((self.total_steps - start_steps) / max(wall, 1e-9)),
            "mean_episode_cost_all": (float(np.mean(self.episode_costs_all))
                                      if self.episode_costs_all else float("nan")),
            "mean_episode_cost_last10": (float(np.mean(recent))
                                         if recent else float("nan")),
            "mean_episode_reward_last10": (
                float(-self.env.reward_scale * np.mean(recent))
                if recent else float("nan")),
            "last_update": self.history[-1] if self.history else {},
        }


# ---------------------------------------------------------------- 存取
def save_policy(path, model: ActorCritic, meta: dict) -> None:
    payload = {
        "kind": "ppo_taufirst",
        "state_dict": model.state_dict(),
        "obs_dim": int(model.obs_dim),
        "act_dim": int(model.act_dim),
        "hidden": list(model.hidden),
        "log_std_init": float(meta.get("log_std_init", -0.5)),
        "meta": dict(meta),
    }
    torch.save(payload, path)


def load_policy(path, *, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("kind", "ppo_taufirst") != "ppo_taufirst":
        raise ValueError(f"不是 ppo_taufirst 策略: {ckpt.get('kind')}")
    model = ActorCritic(
        int(ckpt["obs_dim"]), int(ckpt["act_dim"]),
        tuple(int(h) for h in ckpt["hidden"]),
        float(ckpt.get("log_std_init", -0.5)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, dict(ckpt.get("meta", {}))


def deterministic_actions(model: ActorCritic, obs_batch) -> np.ndarray:
    with torch.no_grad():
        action, _, _, _ = model.get_action(obs_batch, deterministic=True)
    return action.detach().cpu().numpy().astype(np.float64)
