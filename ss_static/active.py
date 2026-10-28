from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np


class ActiveTransferRule:
    """主动调拨规则的基类。

    decide(...) 返回净 tau（模型约定 tau = taup - taum，正=净流入）。
    规则不修改当期 q_out / 被动 y / n_out / W_out；只改变 I_out 和主动成本。
    """

    name = "none"

    def __init__(self, prob, options: dict):
        self.prob = prob
        self.options = dict(options)
        self.K = int(prob.N_R)
        self.T = int(prob.T)
        self.G = int(prob.G)
        self.Imax = np.asarray(prob.Imax, dtype=float)
        self.atc = float(getattr(prob, "atc", 0.0))

    def decide(self, t: int, X, W_out, n_out, o_next, q_out, theta_t, h):
        raise NotImplementedError


class NoActiveRule(ActiveTransferRule):
    name = "none"

    def decide(self, t, X, W_out, n_out, o_next, q_out, theta_t, h):
        return np.zeros_like(np.asarray(X, dtype=float))


class ThresholdBalanceRule(ActiveTransferRule):
    """每仓门限 h_r + 目标库存 T 的主动调拨。

    被动履约后 X=S_end：
      - X_r > h_r 的仓为调出仓，可调出 X_r - T（向 T 靠拢）；
      - X_m < T 的仓为调入仓，缺口 T - X_m；
      - 贪心选择最大盈余/最大缺口；同分时调入仓优先高 W_out+E[F_next]，
        调出仓优先低 W_out+E[F_next]；
      - 目标剩余容量扣除下一期在途 o_next 和本期订单 q_out；
      - 多源对一个目标时维护剩余容量/来源约束台账；
      - 默认末期最后 2 期不调（active_stop_offset=2）。
    """

    name = "threshold_balance"

    def __init__(self, prob, options: dict):
        super().__init__(prob, options)
        self.target = float(options.get("active_target", 5.0))
        self.alpha = float(options.get("active_fraction", 1.0))
        self.stop_offset = int(options.get("active_stop_offset", 2))
        if self.target < 0:
            raise ValueError("active_target 必须 ≥0")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("active_fraction 必须在 (0,1]")
        if self.stop_offset < 0:
            raise ValueError("active_stop_offset 必须 ≥0")
        self.priority = str(options.get("active_priority", "deficit_risk"))
        self.forecast = str(options.get("active_forecast", "pool_mean"))
        if self.priority not in ("deficit", "deficit_risk"):
            raise ValueError(f"未知 active_priority={self.priority!r}")
        if self.forecast not in ("pool_mean", "current"):
            raise ValueError(f"未知 active_forecast={self.forecast!r}")
        # 300 场景池的平均转移矩阵：state-only 的下一期失效预测
        ths = [np.asarray(th, dtype=float) for _, th in prob.pool]
        self.theta_bar = np.mean(np.stack(ths, axis=0), axis=0)  # (K,G,G+1)

    def _expected_failures(self, n_out, theta_t):
        n_out = np.asarray(n_out, dtype=float)  # (P,K,G)
        if self.forecast == "current" and theta_t is not None:
            th = np.asarray(theta_t, dtype=float)  # (P,K,G,G+1)
            return (n_out * th[:, :, :, self.G]).sum(axis=2)
        return (n_out * self.theta_bar[None, :, :, self.G]).sum(axis=2)

    def decide(self, t, X, W_out, n_out, o_next, q_out, theta_t, h):
        X0 = np.asarray(X, dtype=float)
        if X0.ndim != 2 or X0.shape[1] != self.K:
            raise ValueError(f"X shape {X0.shape}")
        P = X0.shape[0]
        if t <= 0 or (self.stop_offset > 0
                      and t >= self.T - self.stop_offset + 1):
            return np.zeros_like(X0)
        if h is None:
            raise ValueError("active_transfer=True 但 policy 没有 h（门限）")
        h = np.asarray(h, dtype=float).reshape(-1)
        if h.size != self.K:
            raise ValueError(f"active threshold h 长度 {h.size} ≠ {self.K}")
        if np.any(h < self.target - 1e-9):
            raise ValueError(f"h 必须 ≥ active_target={self.target}: {h}")
        if np.any(h > self.Imax + 1e-9):
            raise ValueError(f"h 超过 Imax: {h}")

        W_out = np.asarray(W_out, dtype=float)
        o_next = np.asarray(o_next, dtype=float)
        q_out = np.asarray(q_out, dtype=float)
        Xw = X0.copy()
        tau = np.zeros_like(X0)
        risk = W_out + self._expected_failures(n_out, theta_t)
        if self.priority == "deficit":
            risk = np.zeros_like(risk)
        rows = np.arange(P)
        r_idx = np.broadcast_to(np.arange(self.K)[None, :], (P, self.K))
        sum_other = X0.sum(axis=1, keepdims=True) - X0
        # alpha 是每个源仓本期的总调出比例；避免多轮贪心把 alpha 抵消掉
        E0 = np.where(X0 > h[None, :],
                      np.maximum(X0 - self.target, 0.0), 0.0)
        source_budget = self.alpha * E0

        # 最多 2K 轮；每轮至少耗尽一个盈余/缺口/容量，通常远少于 2K
        for _ in range(max(1, 2 * self.K)):
            cap = (self.Imax[None, :] - Xw
                   - o_next.sum(axis=1) - q_out)
            cap = np.maximum(cap, 0.0)
            E = np.where(Xw > h[None, :],
                         np.maximum(Xw - self.target, 0.0), 0.0)
            D = np.maximum(self.target - Xw, 0.0)
            D_eff = np.where(cap > 0.0, D, 0.0)
            if not np.any(E > 1e-12) or not np.any(D_eff > 1e-12):
                break

            # 主键：E/D；次键：risk；再次：仓号
            src = np.lexsort((r_idx, risk, -E), axis=1)[:, 0]
            dst = np.lexsort((r_idx, -risk, -D_eff), axis=1)[:, 0]
            e = E[rows, src]
            d = D_eff[rows, dst]
            cap_d = cap[rows, dst]
            src_cap = sum_other[rows, dst]
            budget = source_budget[rows, src]
            active = ((e > 1e-12) & (d > 1e-12)
                      & (cap_d > 1e-12) & (src_cap > 1e-12)
                      & (budget > 1e-12) & (src != dst))
            if not np.any(active):
                break
            amount = np.minimum.reduce([e, d, cap_d, src_cap, budget])
            amount = np.where(active, amount, 0.0)
            # 更新净 tau 与工作库存
            np.add.at(tau, (rows, src), -amount)
            np.add.at(tau, (rows, dst), amount)
            Xw[rows, src] -= amount
            Xw[rows, dst] += amount
            sum_other[rows, dst] -= amount
            source_budget[rows, src] -= amount
        return tau


_RULES = {
    "none": NoActiveRule,
    "threshold_balance": ThresholdBalanceRule,
}


def make_active_rule(prob, options: dict) -> ActiveTransferRule:
    if not bool(options.get("active_transfer", False)):
        return NoActiveRule(prob, options)
    name = str(options.get("active_rule", "threshold_balance"))
    if name not in _RULES:
        raise ValueError(
            f"未知 active_rule={name!r}；可选 {sorted(_RULES)}")
    return _RULES[name](prob, options)
