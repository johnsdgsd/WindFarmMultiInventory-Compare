from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass, field

import numpy as np

from ss_static.metrics import behavior_stats
from ss_static.policy import SSStaticPolicy
from ss_static.simulator import TERM_NAMES, Simulator, scene_hash


@dataclass(frozen=True)
class SSParams:
    """(s,S) 参数；active 时额外带每仓主动调拨门限 h。

    no-active: theta = concat([s, S])           (2K)
    active:    theta = concat([s, S, h])        (3K)
    """
    s: np.ndarray
    S: np.ndarray
    h: np.ndarray | None = None

    def __post_init__(self):
        s = np.asarray(self.s, dtype=float).reshape(-1)
        S = np.asarray(self.S, dtype=float).reshape(-1)
        if s.size != S.size:
            raise ValueError(f"s/S 长度不一致: {s.size} vs {S.size}")
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "S", S)
        if self.h is not None:
            h = np.asarray(self.h, dtype=float).reshape(-1)
            if h.size != s.size:
                raise ValueError(f"h 长度 {h.size} != K={s.size}")
            object.__setattr__(self, "h", h)

    @property
    def K(self) -> int:
        return int(self.s.size)

    @property
    def theta(self) -> np.ndarray:
        if self.h is None:
            return np.concatenate([self.s, self.S])
        return np.concatenate([self.s, self.S, self.h])

    @classmethod
    def from_theta(cls, theta, K: int, active: bool = False) -> "SSParams":
        t = np.asarray(theta, dtype=float).reshape(-1)
        need = (3 if active else 2) * int(K)
        if t.size != need:
            raise ValueError(f"theta 长度 {t.size} ≠ {need}")
        K = int(K)
        if active:
            return cls(t[:K], t[K:2 * K], t[2 * K:])
        return cls(t[:K], t[K:])

    def as_dict(self) -> dict:
        out = {"s": self.s.tolist(), "S": self.S.tolist()}
        if self.h is not None:
            out["active_thresholds"] = self.h.tolist()
        return out


@dataclass
class EvaluationResult:
    """评价函数输出：成本 + 五项明细 + 可选审计/行为数据。"""
    cost: float
    path_costs: np.ndarray
    term_totals: np.ndarray
    step_terms: np.ndarray | None = None
    behavior: dict | None = None
    arrays: dict | None = None
    meta: dict = field(default_factory=dict)

    @property
    def P(self) -> int:
        return int(self.path_costs.size)

    @property
    def T(self) -> int:
        return int(self.meta["T"])

    @property
    def mean(self) -> float:
        return float(self.cost)

    @property
    def std(self) -> float:
        return float(np.std(self.path_costs, ddof=1)) if self.P > 1 else 0.0

    @property
    def se(self) -> float:
        return self.std / np.sqrt(self.P) if self.P > 0 else float("nan")

    @property
    def ci95(self) -> float:
        return 1.959963984540054 * self.se


class SSEvaluator:
    """非时变 (s,S) 的方法自包含评价函数。

    输入：theta = [s_1..s_K, S_1..S_K]，场景路径 paths=(P,T)。
    输出：EvaluationResult.cost = 该参数在 paths 上的平均总成本。
    """

    def __init__(self, prob, options: dict):
        self.prob = prob
        self.options = dict(options)
        self.sim = Simulator(prob)
        self.N_R = int(prob.N_R)
        self.T = int(prob.T)
        self.Imax = np.asarray(prob.Imax, dtype=float)
        self.term_names = TERM_NAMES
        self.active = bool(self.options.get("active_transfer", False))

    def check_theta(self, theta) -> SSParams:
        t = np.asarray(theta, dtype=float).reshape(-1)
        need = (3 if self.active else 2) * self.N_R
        if t.size != need:
            raise ValueError(f"theta 长度 {t.size} ≠ {need}")
        if not np.all(np.isfinite(t)):
            raise ValueError("theta 含 NaN/Inf")
        s, S = t[:self.N_R], t[self.N_R:2 * self.N_R]
        if np.any(s < -1e-9):
            raise ValueError(f"s 必须 ≥0: {s}")
        if np.any(S > self.Imax + 1e-9):
            raise ValueError(f"S 超过 Imax: {S} vs {self.Imax}")
        if np.any(s > S + 1e-9):
            raise ValueError(f"要求 s≤S，实际 s={s}, S={S}")
        h = None
        if self.active:
            h = t[2 * self.N_R:]
            if np.any(h < -1e-9):
                raise ValueError(f"active threshold h 必须 ≥0: {h}")
            if np.any(h > self.Imax + 1e-9):
                raise ValueError(f"active threshold h 超过 Imax: {h}")
        return SSParams(s, S, h)

    def evaluate(self, theta, paths, *, record: bool = False) -> EvaluationResult:
        params = self.check_theta(theta)
        paths = np.asarray(paths)
        if paths.ndim != 2 or paths.shape[1] != self.T:
            raise ValueError(
                f"paths shape 必须是 (P,{self.T})，实际 {paths.shape}")
        pol = SSStaticPolicy(self.prob, params.s, params.S, self.options,
                             h=params.h)
        res = self.sim.simulate(pol, paths, record=record)
        step_terms = (np.asarray(res.arrays["terms"], dtype=float)
                      if record else None)
        behavior = behavior_stats(self.prob, res) if record else None
        return EvaluationResult(
            cost=float(np.mean(res.totals)),
            path_costs=np.asarray(res.totals, dtype=float),
            term_totals=np.asarray(res.term_totals, dtype=float),
            step_terms=step_terms,
            behavior=behavior,
            arrays=res.arrays if record else None,
            meta={
                "theta": params.theta.tolist(),
                "s": params.s.tolist(),
                "S": params.S.tolist(),
                "active_thresholds": (params.h.tolist()
                                      if params.h is not None else None),
                "n_paths": int(paths.shape[0]),
                "T": self.T,
                "path_hash": scene_hash(paths),
                "policy_params": pol.params(),
                "record": bool(record),
            },
        )

    def evaluate_cost(self, theta, paths) -> float:
        return self.evaluate(theta, paths, record=False).cost


def evaluate(prob, options: dict, theta, paths, *, record: bool = False):
    """纯函数入口；返回 EvaluationResult。"""
    return SSEvaluator(prob, options).evaluate(theta, paths, record=record)


def evaluate_cost(prob, options: dict, theta, paths) -> float:
    """纯函数入口；输入参数，输出路径平均成本。"""
    return SSEvaluator(prob, options).evaluate_cost(theta, paths)
