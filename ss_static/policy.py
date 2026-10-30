from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np

from ss_static.active import make_active_rule


class PolicyBase:
    """规则策略基类：订货由子类实现；履约默认“本地优先 + 优先级被动调拨”。

    本文件属于 ss_static 方法自包含目录；不依赖 common.policy。

    约定：所有向量按 P 条路径向量化；y 的轴为 [destination, source]，
    与 code/sim_analysis.py 的 pas/send_def 口径一致。
    """

    policy_kind = "base"

    def __init__(self, prob, options: dict):
        self.prob = prob
        self.options = dict(options)
        self.T = int(prob.T)
        self.N_R = int(prob.N_R)
        self.G = int(prob.G)
        self.L = int(prob.L)
        self.N = np.asarray(prob.N, dtype=float)
        self.Qmax = np.asarray(prob.Qmax, dtype=float)
        self.Imax = np.asarray(prob.Imax, dtype=float)
        self.Cs = float(prob.Cs)
        self.atc = float(getattr(prob, "atc", 0.0))
        self.tc_route = np.asarray(prob.tc_route, dtype=float)
        self.priority = {
            r: [int(m) - 1 for m in prob.priority[r + 1]]
            for r in range(self.N_R)
        }

        self.position = self._choice("position", ("net", "gross"))
        self.passive_transshipment = bool(self._opt("passive_transshipment"))
        self.cross_cost_rule = self._choice(
            "cross_cost_rule", ("horizon", "fixed", "all"))
        self.max_passive_cost = float(self._opt("max_passive_cost"))
        # active 机制可插拔；旧 policy JSON 没有这些键时默认 mode 0
        self.active_enabled = bool(self.options.get("active_transfer", False))
        self.active_rule_name = str(self.options.get(
            "active_rule", "threshold_balance"))
        self.active_rule = make_active_rule(prob, self.options)
        self.fill = str(self._opt("fill"))
        if self.fill != "local_then_priority":
            raise ValueError(
                f"未实现的履约规则 fill={self.fill!r}；只支持 local_then_priority")

    def _opt(self, key: str):
        if key not in self.options:
            raise KeyError(f"策略选项缺少 {key}")
        return self.options[key]

    def _choice(self, key: str, choices: tuple[str, ...]) -> str:
        v = self._opt(key)
        if v not in choices:
            raise ValueError(f"策略选项 {key}={v!r} 不在 {choices}")
        return str(v)

    def order(self, t: int, I, o, n):
        raise NotImplementedError

    def fulfill(self, t: int, I, D):
        """两阶段履约：先全仓本地替换，再按优先级做跨仓被动调拨。

        本地阶段必须先对所有仓做完，避免前面的跨仓请求抢走后面仓库的本地库存。
        只使用期初库存 I，不含本期到货 o[0]。
        """
        I = np.asarray(I, dtype=float)
        D = np.asarray(D, dtype=float)
        P = I.shape[0]
        y = np.zeros((P, self.N_R, self.N_R), dtype=float)
        avail = I.copy()
        rem = D.copy()

        # 阶段 1：所有仓先用本地库存替换自己的待修/新失效
        take = np.minimum(rem, avail)
        idx = np.arange(self.N_R)
        y[:, idx, idx] = take
        avail -= take
        rem -= take

        # 阶段 2：跨仓被动调拨（只使用期初库存 I_in；本期到货不可用）
        if self.passive_transshipment:
            if self.cross_cost_rule == "horizon":
                # 替换在本期发生，减少的是下一期起的停机；剩余视界价值
                limit = self.Cs * max(0, self.T - t)
            elif self.cross_cost_rule == "fixed":
                limit = self.max_passive_cost
            else:
                limit = float("inf")
            allow = limit > 0.0
            if allow:
                # docs/s,S.md 第 3 节：所有跨仓配对按 (tc,r,m) 升序全局贪心
                edges = []
                for r in range(self.N_R):
                    for m in range(self.N_R):
                        if m == r:
                            continue
                        c = float(self.tc_route[r, m])
                        if c <= limit + 1e-12:
                            edges.append((c, r, m))
                edges.sort()
                for _, r, m in edges:
                    take = np.minimum(rem[:, r], avail[:, m])
                    y[:, r, m] += take
                    avail[:, m] -= take
                    rem[:, r] -= take
        return y

    def active_transfer(self, t, X, W_out, n_out, o_next, theta):
        """默认无主动调拨；active 子类用 active_rule 覆盖。"""
        return np.zeros_like(np.asarray(X, dtype=float))

    def params(self) -> dict:
        return {"kind": self.policy_kind, "options": dict(self.options)}


class SSStaticPolicy(PolicyBase):
    """非时变 (s,S) 基准，按 docs/s,S.md 实现（mode 0：无主动调拨）。

    订货只看 x_t（I、在途、q_in、W_in），不看本期 Θ；履约看本期 Θ。
    """

    policy_kind = "ss_static"

    def __init__(self, prob, s, S, options: dict, h=None):
        super().__init__(prob, options)
        self.s = np.asarray(s, dtype=float).reshape(-1)
        self.S = np.asarray(S, dtype=float).reshape(-1)
        if self.s.size != self.N_R or self.S.size != self.N_R:
            raise ValueError(f"s/S 长度必须为 N_R={self.N_R}")
        if not np.all(np.isfinite(self.s)) or not np.all(np.isfinite(self.S)):
            raise ValueError("s/S 必须有限")
        if np.any(self.s < -1e-9):
            raise ValueError("docs/s,S.md 要求 0≤s")
        if np.any(self.S < -1e-9):
            raise ValueError("S 必须非负")
        if np.any(self.S > self.Imax + 1e-9):
            raise ValueError(f"S 超过 Imax: {self.S} vs {self.Imax}")
        if np.any(self.s > self.S + 1e-9):
            raise ValueError(f"要求 s≤S，实际 s={self.s}, S={self.S}")
        self.h = None
        if self.active_enabled:
            if h is None:
                raise ValueError("active_transfer=True 时必须提供每仓门限 h")
            h = np.asarray(h, dtype=float).reshape(-1)
            if h.size != self.N_R:
                raise ValueError(f"h 长度必须为 N_R={self.N_R}")
            if not np.all(np.isfinite(h)):
                raise ValueError("h 必须有限")
            if np.any(h < -1e-9):
                raise ValueError(f"h 必须 ≥0: {h}")
            if np.any(h > self.Imax + 1e-9):
                raise ValueError(f"h 超过 Imax: {h}")
            self.h = h

    def order(self, t: int, I, o, n):
        I = np.asarray(I, dtype=float)
        o = np.asarray(o, dtype=float)
        n = np.asarray(n, dtype=float)
        P = I.shape[0]

        # s,S 没有 q_in：IP 只含在手库存 + 在途，不含当期决策 q_t。
        gross = I + o.sum(axis=1)
        W = np.maximum(self.N[None, :] - n.sum(axis=2), 0.0)
        pos = gross - W if self.position == "net" else gross
        R = self.Imax[None, :] - gross

        need = self.S[None, :] - pos
        raw = np.maximum(0.0, np.minimum(need, R))
        # q 同时受模型变量上界 Qmax 和库存容量 R 约束；越界直接截断。
        raw = np.minimum(raw, self.Qmax[None, :])
        q_new = np.where(pos < self.s[None, :], raw, 0.0)
        q_new = np.clip(q_new, 0.0, self.Qmax[None, :])
        if q_new.shape != (P, self.N_R):
            raise RuntimeError(f"q_new shape {q_new.shape}")
        return q_new

    def active_transfer(self, t, X, W_out, n_out, o_next, theta):
        if not self.active_enabled:
            return np.zeros_like(np.asarray(X, dtype=float))
        return self.active_rule.decide(
            t, X, W_out, n_out, o_next, theta, self.h)

    def params(self) -> dict:
        out = {
            "kind": self.policy_kind,
            "s": self.s.tolist(),
            "S": self.S.tolist(),
            "options": dict(self.options),
        }
        if self.h is not None:
            out["active_thresholds"] = self.h.tolist()
        return out

    def to_dict(self) -> dict:
        return self.params()

    @classmethod
    def from_dict(cls, prob, data: dict) -> "SSStaticPolicy":
        if data.get("kind", "ss_static") != "ss_static":
            raise ValueError(f"不是 ss_static 策略: {data.get('kind')}")
        h = data.get("active_thresholds")
        if data["options"].get("active_transfer", False) and h is None:
            raise ValueError("active policy.json 缺少 active_thresholds")
        return cls(prob, data["s"], data["S"], data["options"], h=h)
