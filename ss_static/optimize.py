from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import time
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy import optimize as sp_optimize
except Exception:  # pragma: no cover
    sp_optimize = None


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------
@dataclass
class InitialPoint:
    name: str
    theta: np.ndarray
    radius: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class OptimizeResult:
    algorithm: str
    theta: np.ndarray
    cost: float
    n_calls: int
    n_unique: int
    wall_s: float
    history: list[dict] = field(default_factory=list)
    start_results: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 参数空间
# ---------------------------------------------------------------------------
@dataclass
class ParamSpace:
    """0 ≤ s_r ≤ S_r ≤ Imax_r 的参数空间。

    坐标下降/随机搜索在物理空间工作；Nelder-Mead/Powell/DE/SA/GA 在
    sigmoid 变换后的无约束 u 空间工作，解码一定满足 s≤S 和上下界。
    """
    Imax: np.ndarray
    integer: bool = False
    u_bound: float = 12.0
    snap_frac: float = 1e-5
    active: bool = False
    active_min: float | None = None

    def __post_init__(self):
        self.Imax = np.asarray(self.Imax, dtype=float).reshape(-1)
        if self.Imax.size < 1 or np.any(self.Imax <= 0):
            raise ValueError(f"Imax 非法: {self.Imax}")
        self.K = int(self.Imax.size)
        self.active_min = (0.0 if self.active_min is None
                           else float(self.active_min))
        if self.active_min < 0:
            raise ValueError("active_min 必须 ≥0")
        if self.active and np.any(self.Imax < self.active_min - 1e-12):
            raise ValueError(
                f"Imax 必须 ≥ active_min={self.active_min}: {self.Imax}")

    @property
    def dim(self) -> int:
        return (3 if self.active else 2) * self.K

    # -------------------------------------------------- 物理空间
    def clip_many(self, thetas) -> np.ndarray:
        t = np.asarray(thetas, dtype=float)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        if t.ndim != 2 or t.shape[1] != self.dim:
            raise ValueError(f"theta shape 必须是 (n,{self.dim})，实际 {t.shape}")
        s = np.clip(t[:, :self.K], 0.0, self.Imax[None, :])
        S = np.clip(t[:, self.K:2 * self.K], s, self.Imax[None, :])
        if self.integer:
            s = np.clip(np.rint(s), 0.0, self.Imax[None, :])
            S = np.clip(np.rint(S), s, self.Imax[None, :])
        tol = self.snap_frac * self.Imax[None, :]
        s = np.where(s < tol, 0.0, s)
        s = np.where(s > self.Imax[None, :] - tol, self.Imax[None, :], s)
        S = np.where(S < tol, 0.0, S)
        S = np.clip(S, s, self.Imax[None, :])
        S = np.where(S > self.Imax[None, :] - tol, self.Imax[None, :], S)
        if self.active:
            h = np.clip(t[:, 2 * self.K:],
                        self.active_min, self.Imax[None, :])
            tol_h = self.snap_frac * np.maximum(
                self.Imax - self.active_min, 1.0)
            h = np.where(h < self.active_min + tol_h, self.active_min, h)
            h = np.where(h > self.Imax - tol_h, self.Imax, h)
            return np.concatenate([s, S, h], axis=1)
        return np.concatenate([s, S], axis=1)

    def clip(self, theta) -> np.ndarray:
        return self.clip_many(np.asarray(theta, dtype=float).reshape(1, -1))[0]

    def sample(self, rng, n: int, unit=None) -> np.ndarray:
        """物理空间采样：s~U(0,Imax)，宽度~U(0,Imax-s)，保证 s≤S。"""
        n = int(n)
        if n < 1:
            raise ValueError("n 必须 ≥1")
        if unit is None:
            unit = rng.random((n, self.dim))
        u = np.asarray(unit, dtype=float)
        if u.shape != (n, self.dim):
            raise ValueError(f"unit shape {u.shape} ≠ {(n, self.dim)}")
        s = self.Imax[None, :] * u[:, :self.K]
        S = s + (self.Imax[None, :] - s) * u[:, self.K:2 * self.K]
        if self.active:
            h = (self.active_min
                 + (self.Imax[None, :] - self.active_min)
                 * u[:, 2 * self.K:])
            return self.clip_many(np.concatenate([s, S, h], axis=1))
        return self.clip_many(np.concatenate([s, S], axis=1))

    def sample_lhs(self, rng, n: int) -> np.ndarray:
        """拉丁超立方采样；scipy 不可用时退化为均匀采样。"""
        n = int(n)
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(
                d=self.dim, seed=int(rng.integers(0, 2**31 - 1)))
            unit = sampler.random(n)
        except Exception:
            unit = rng.random((n, self.dim))
        return self.sample(rng, n, unit=unit)

    # -------------------------------------------------- u 空间
    def sample_u(self, rng, n: int) -> np.ndarray:
        return rng.uniform(-self.u_bound, self.u_bound,
                           size=(int(n), self.dim))

    def bounds_u(self) -> list[tuple[float, float]]:
        return [(-self.u_bound, self.u_bound)] * self.dim

    @staticmethod
    def _sigmoid(x):
        return 0.5 * (1.0 + np.tanh(0.5 * x))

    def from_u(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != self.dim:
            raise ValueError(f"u 长度 {u.size} ≠ {self.dim}")
        uc = np.clip(u, -self.u_bound, self.u_bound)
        sig = self._sigmoid(uc)
        s = self.Imax * sig[:self.K]
        S = s + (self.Imax - s) * sig[self.K:2 * self.K]
        if self.active:
            h = (self.active_min
                 + (self.Imax - self.active_min) * sig[2 * self.K:])
            return self.clip(np.concatenate([s, S, h]))
        return self.clip(np.concatenate([s, S]))

    def to_u(self, theta) -> np.ndarray:
        t = self.clip(theta)
        s, S = t[:self.K], t[self.K:2 * self.K]
        eps = 1e-9
        p = np.clip(s / self.Imax, eps, 1.0 - eps)
        u1 = np.log(p / (1.0 - p))
        w = np.clip((S - s) / np.maximum(self.Imax - s, eps), eps, 1.0 - eps)
        u2 = np.log(w / (1.0 - w))
        out = [np.clip(u1, -self.u_bound, self.u_bound),
               np.clip(u2, -self.u_bound, self.u_bound)]
        if self.active:
            h = t[2 * self.K:]
            p_h = np.clip(
                (h - self.active_min)
                / np.maximum(self.Imax - self.active_min, eps),
                eps, 1.0 - eps)
            u3 = np.log(p_h / (1.0 - p_h))
            out.append(np.clip(u3, -self.u_bound, self.u_bound))
        return np.concatenate(out)


# ---------------------------------------------------------------------------
# 带缓存的评价函数
# ---------------------------------------------------------------------------
class CachedObjective:
    """θ -> cost 的确定性缓存包装。

    n_calls 是调用次数（含缓存命中）；n_unique 是真实仿真次数。
    算法比较时以 n_unique 为准。
    """

    def __init__(self, fn, space: ParamSpace, *, keep_history: bool = True):
        self.fn = fn
        self.space = space
        self.keep_history = bool(keep_history)
        self.cache: dict = {}
        self.n_calls = 0
        self.history: list[dict] = []

    def __call__(self, theta) -> float:
        self.n_calls += 1
        t = self.space.clip(theta)
        if self.space.integer:
            key = tuple(int(x) for x in np.rint(t).tolist())
        else:
            key = tuple(float(x) for x in np.round(t, 9).tolist())
        val = self.cache.get(key)
        if val is None:
            val = float(self.fn(t))
            if not np.isfinite(val):
                raise ValueError(f"评价函数返回非有限值: {val}")
            self.cache[key] = val
            if self.keep_history:
                self.history.append({"theta": t.tolist(), "cost": val})
        return val

    @property
    def n_unique(self) -> int:
        return len(self.cache)


# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, dict] = {
    "coordinate": {
        "n_grid": 20, "max_rounds": 3, "eps": 1e-3, "eta": 1e-4,
        "rho_min": 1e-4,
    },
    "random": {
        "n_samples": 1000, "n_top": 10, "n_refine": 300,
        "sigma0": 0.25, "sigma_min": 1e-3,
    },
    "nelder-mead": {
        "max_evals": 1500, "xatol": 1e-3, "fatol": 1e-3,
    },
    "powell": {
        "max_evals": 1500, "xtol": 1e-3, "ftol": 1e-4,
    },
    "differential-evolution": {
        "maxiter": 40, "popsize": 8, "tol": 1e-4, "polish": False,
        "restarts": 1,
    },
    "dual-annealing": {
        "maxiter": 300, "maxfun": 3000, "no_local_search": True,
        "restarts": 1,
    },
    "genetic": {
        "population": 40, "generations": 40, "crossover_prob": 0.8,
        "mutation_prob": 0.2, "mutation_sigma": 0.5, "alpha": 0.5,
        "elite": 2, "max_evals": 4000,
    },
    "simulated-annealing": {
        "n_iters": 2000, "T0_frac": 0.05, "alpha": 0.995, "sigma": 0.5,
        "n_chains": 3,
    },
}

ALGORITHM_ORDER = (
    "coordinate", "random", "nelder-mead", "powell",
    "differential-evolution", "dual-annealing", "genetic",
    "simulated-annealing",
)


def available_algorithms() -> list[str]:
    return list(ALGORITHM_ORDER)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _norm_starts(starts, space: ParamSpace) -> list[InitialPoint]:
    if isinstance(starts, InitialPoint):
        starts = [starts]
    if starts is None:
        starts = []
    if isinstance(starts, np.ndarray):
        if starts.ndim == 1:
            starts = [starts]
        elif starts.ndim == 2:
            starts = [row for row in starts]
        else:
            raise ValueError(f"starts ndarray 维度非法: {starts.shape}")
    out: list[InitialPoint] = []
    for i, st in enumerate(starts):
        if isinstance(st, InitialPoint):
            theta, name = st.theta, st.name
            radius, meta = st.radius, dict(st.meta)
        else:
            theta, name, radius, meta = st, f"start{i}", None, {}
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if space.active and theta.size == 2 * space.K:
            theta = np.concatenate(
                [theta, np.full(space.K, space.active_min)])
        theta = space.clip(theta)
        if radius is not None:
            radius = np.asarray(radius, dtype=float).reshape(-1)
            if space.active and radius.size == 2 * space.K:
                radius = np.concatenate(
                    [radius, space.Imax - space.active_min])
            if radius.size != space.dim:
                raise ValueError(
                    f"radius 长度 {radius.size} ≠ {space.dim}")
            if space.integer:
                radius = np.maximum(radius, 1.0)
        out.append(InitialPoint(name, theta, radius, meta))
    if not out:
        theta0 = np.concatenate([0.25 * space.Imax, 0.5 * space.Imax])
        if space.active:
            theta0 = np.concatenate(
                [theta0, np.full(space.K, space.active_min)])
        out = [InitialPoint("default", space.clip(theta0))]
    return out


def _default_radius(space: ParamSpace,
                    radius: np.ndarray | None) -> np.ndarray:
    if radius is not None:
        r = np.asarray(radius, dtype=float).reshape(-1)
        if space.active and r.size == 2 * space.K:
            r = np.concatenate([r, space.Imax - space.active_min])
        return r
    r = np.concatenate([0.25 * space.Imax, 0.5 * space.Imax])
    if space.active:
        r = np.concatenate([r, space.Imax - space.active_min])
    return np.maximum(r, 1.0) if space.integer else r


def _coord_bounds(space: ParamSpace, theta: np.ndarray, j: int):
    if j < space.K:
        return 0.0, float(theta[space.K + j])
    if j < 2 * space.K:
        r = j - space.K
        return float(theta[r]), float(space.Imax[r])
    r = j - 2 * space.K
    return float(space.active_min), float(space.Imax[r])


def _grid(center: float, radius: float, n_grid: int,
          lo: float, hi: float) -> list[float]:
    if hi < lo - 1e-12:
        lo, hi = min(lo, hi), max(lo, hi)
    center = float(np.clip(center, lo, hi))
    k = max(2, int(n_grid))
    if radius <= 0:
        return [center]
    vals = center + float(radius) * (
        2.0 * np.arange(k, dtype=float) / (k - 1) - 1.0)
    vals = np.clip(vals, lo, hi)
    vals = np.unique(np.concatenate([vals, [center]]))
    return [float(v) for v in vals]


def _grid_int(center: float, radius: float,
              lo: float, hi: float) -> list[int]:
    if hi < lo - 1e-12:
        lo, hi = min(lo, hi), max(lo, hi)
    a = int(np.ceil(max(float(lo), float(center) - float(radius))))
    b = int(np.floor(min(float(hi), float(center) + float(radius))))
    vals = set(range(a, b + 1))
    c = int(np.clip(round(float(center)), float(lo), float(hi)))
    vals.add(c)
    return sorted(vals)


def _cfg(name: str, config: dict | None) -> dict:
    out = dict(DEFAULTS[name])
    if config:
        out.update(dict(config))
    return out


# ---------------------------------------------------------------------------
# 坐标下降（docs/s,S.md §9）
# ---------------------------------------------------------------------------
def _run_coordinate(obj, space, starts, rng, cfg) -> tuple:
    theta_star = None
    cost_star = float("inf")
    start_results = []
    n_grid = int(cfg["n_grid"])
    max_rounds = int(cfg["max_rounds"])
    eps = float(cfg["eps"])
    eta = float(cfg["eta"])
    rho_min = float(cfg["rho_min"])

    for st in starts:
        theta = space.clip(st.theta)
        radius = _default_radius(space, st.radius)
        cur = obj(theta)
        hist = []
        for rd in range(max_rounds):
            old_theta = theta.copy()
            old_cost = cur
            for j in range(space.dim):
                lo, hi = _coord_bounds(space, theta, j)
                vals = (_grid_int(theta[j], radius[j], lo, hi)
                        if (space.integer and j < 2 * space.K)
                        else _grid(theta[j], radius[j], n_grid, lo, hi))
                best_v, best_val, best_dist = theta[j], float("inf"), float("inf")
                for v in vals:
                    cand = theta.copy()
                    cand[j] = v
                    cand = space.clip(cand)
                    val = obj(cand)
                    dist = abs(cand[j] - theta[j])
                    if (val < best_val - 1e-12
                            or (abs(val - best_val) <= 1e-12
                                and dist < best_dist)):
                        best_v, best_val, best_dist = cand[j], val, dist
                theta[j] = best_v
                cur = best_val
            delta = float(np.max(np.abs(theta - old_theta)))
            rel = abs(cur - old_cost) / max(abs(old_cost), 1e-12)
            hist.append({"round": rd + 1, "cost": float(cur),
                         "delta_inf": delta, "rel_improve": rel})
            if delta < eps or rel < eta:
                break
            if rd == max_rounds - 1:
                break
            radius = np.maximum(radius / 2.0, rho_min)
        start_results.append({
            "name": st.name, "theta": theta.tolist(), "cost": float(cur),
            "history": hist,
        })
        if cur < cost_star:
            cost_star, theta_star = cur, theta.copy()
    return theta_star, cost_star, start_results, {
        "n_grid": n_grid, "max_rounds": max_rounds, "eps": eps, "eta": eta,
        "rho_min": rho_min,
    }


# ---------------------------------------------------------------------------
# 随机搜索 / LHS + 局部精修
# ---------------------------------------------------------------------------
def _run_random(obj, space, starts, rng, cfg) -> tuple:
    n_samples = int(cfg["n_samples"])
    n_top = max(1, min(int(cfg["n_top"]), n_samples))
    n_refine = int(cfg["n_refine"])
    sigma0 = float(cfg["sigma0"])
    sigma_min = float(cfg["sigma_min"])

    start_results = []
    for st in starts:
        start_results.append({
            "name": st.name, "theta": st.theta.tolist(),
            "cost": float(obj(st.theta)),
        })

    samples = (space.sample_lhs(rng, n_samples)
               if n_samples > 0 else np.empty((0, space.dim)))
    cand = [st.theta for st in starts] + [row for row in samples]
    costs = np.asarray([obj(t) for t in cand], dtype=float)
    order = np.argsort(costs)
    top = [cand[i].copy() for i in order[:n_top]]
    top_costs = [float(costs[i]) for i in order[:n_top]]
    best_i = int(order[0])
    theta_star = cand[best_i].copy()
    cost_star = float(costs[best_i])

    scale_vec = np.concatenate([space.Imax, space.Imax])
    for k in range(n_refine):
        frac = k / max(n_refine - 1, 1)
        scale = sigma0 * (sigma_min / sigma0) ** frac
        i = k % n_top
        noise = rng.normal(0.0, 1.0, size=space.dim) * scale * scale_vec
        cand_t = space.clip(top[i] + noise)
        val = obj(cand_t)
        if val < top_costs[i] - 1e-12:
            top[i] = cand_t
            top_costs[i] = val
            if val < cost_star:
                cost_star, theta_star = val, cand_t.copy()
    start_results.append({
        "name": "random_best", "theta": theta_star.tolist(),
        "cost": float(cost_star), "n_samples": n_samples,
        "n_refine": n_refine,
    })
    return theta_star, cost_star, start_results, {
        "n_samples": n_samples, "n_top": n_top, "n_refine": n_refine,
        "sigma0": sigma0, "sigma_min": sigma_min,
    }


# ---------------------------------------------------------------------------
# SciPy 局部算法
# ---------------------------------------------------------------------------
def _run_scipy_local(method, obj, space, starts, rng, cfg) -> tuple:
    if sp_optimize is None:
        raise ImportError("需要 scipy 才能使用 Nelder-Mead/Powell")
    max_evals = int(cfg["max_evals"])
    theta_star, cost_star = None, float("inf")
    start_results = []

    def f(u):
        return obj(space.from_u(u))

    for st in starts:
        x0 = space.to_u(st.theta)
        try:
            if method == "nelder-mead":
                res = sp_optimize.minimize(
                    f, x0, method="Nelder-Mead", bounds=space.bounds_u(),
                    options={"maxfev": max_evals,
                             "xatol": float(cfg["xatol"]),
                             "fatol": float(cfg["fatol"]),
                             "disp": False})
            elif method == "powell":
                res = sp_optimize.minimize(
                    f, x0, method="Powell", bounds=space.bounds_u(),
                    options={"maxfev": max_evals,
                             "xtol": float(cfg["xtol"]),
                             "ftol": float(cfg["ftol"]),
                             "disp": False})
            else:  # pragma: no cover
                raise ValueError(method)
        except TypeError:
            # 老版本 scipy 的 Nelder-Mead 不接受 bounds
            res = sp_optimize.minimize(
                f, x0, method="Nelder-Mead" if method == "nelder-mead"
                else "Powell",
                options={"maxfev": max_evals, "disp": False})
        theta = space.from_u(res.x)
        cost = obj(theta)
        start_results.append({
            "name": st.name, "theta": theta.tolist(), "cost": float(cost),
            "success": bool(getattr(res, "success", False)),
            "message": str(getattr(res, "message", "")),
            "nfev": int(getattr(res, "nfev", -1)),
        })
        if cost < cost_star:
            cost_star, theta_star = cost, theta.copy()
    return theta_star, cost_star, start_results, {
        "method": method, "max_evals": max_evals,
    }


def _run_nelder_mead(obj, space, starts, rng, cfg):
    return _run_scipy_local("nelder-mead", obj, space, starts, rng, cfg)


def _run_powell(obj, space, starts, rng, cfg):
    return _run_scipy_local("powell", obj, space, starts, rng, cfg)


# ---------------------------------------------------------------------------
# 差分进化
# ---------------------------------------------------------------------------
def _run_differential_evolution(obj, space, starts, rng, cfg) -> tuple:
    if sp_optimize is None:
        raise ImportError("需要 scipy 才能使用差分进化")

    def f(u):
        return obj(space.from_u(u))

    theta_star, cost_star = None, float("inf")
    start_results = []
    restarts = max(1, int(cfg["restarts"]))
    for k in range(restarts):
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            res = sp_optimize.differential_evolution(
                f, space.bounds_u(),
                maxiter=int(cfg["maxiter"]), popsize=int(cfg["popsize"]),
                tol=float(cfg["tol"]), seed=seed,
                polish=bool(cfg.get("polish", False)),
                init="sobol", workers=1)
        except Exception:
            res = sp_optimize.differential_evolution(
                f, space.bounds_u(),
                maxiter=int(cfg["maxiter"]), popsize=int(cfg["popsize"]),
                tol=float(cfg["tol"]), seed=seed,
                polish=bool(cfg.get("polish", False)), workers=1)
        theta = space.from_u(res.x)
        cost = obj(theta)
        start_results.append({
            "name": f"de_restart_{k + 1}", "theta": theta.tolist(),
            "cost": float(cost), "success": bool(getattr(res, "success", False)),
            "nfev": int(getattr(res, "nfev", -1)), "seed": seed,
        })
        if cost < cost_star:
            cost_star, theta_star = cost, theta.copy()
    for st in starts:
        cost = obj(st.theta)
        start_results.append({"name": st.name, "theta": st.theta.tolist(),
                              "cost": float(cost)})
        if cost < cost_star:
            cost_star, theta_star = cost, st.theta.copy()
    return theta_star, cost_star, start_results, {
        "maxiter": int(cfg["maxiter"]), "popsize": int(cfg["popsize"]),
        "restarts": restarts,
    }


# ---------------------------------------------------------------------------
# 对偶退火
# ---------------------------------------------------------------------------
def _run_dual_annealing(obj, space, starts, rng, cfg) -> tuple:
    if sp_optimize is None:
        raise ImportError("需要 scipy 才能使用对偶退火")

    def f(u):
        return obj(space.from_u(u))

    theta_star, cost_star = None, float("inf")
    start_results = []
    restarts = max(1, int(cfg["restarts"]))
    for k in range(restarts):
        seed = int(rng.integers(0, 2**31 - 1))
        res = sp_optimize.dual_annealing(
            f, space.bounds_u(),
            maxiter=int(cfg["maxiter"]), maxfun=int(cfg["maxfun"]),
            seed=seed,
            no_local_search=bool(cfg.get("no_local_search", True)))
        theta = space.from_u(res.x)
        cost = obj(theta)
        start_results.append({
            "name": f"da_restart_{k + 1}", "theta": theta.tolist(),
            "cost": float(cost), "success": bool(getattr(res, "success", False)),
            "nfev": int(getattr(res, "nfev", -1)), "seed": seed,
        })
        if cost < cost_star:
            cost_star, theta_star = cost, theta.copy()
    for st in starts:
        cost = obj(st.theta)
        start_results.append({"name": st.name, "theta": st.theta.tolist(),
                              "cost": float(cost)})
        if cost < cost_star:
            cost_star, theta_star = cost, st.theta.copy()
    return theta_star, cost_star, start_results, {
        "maxiter": int(cfg["maxiter"]), "maxfun": int(cfg["maxfun"]),
        "restarts": restarts,
    }


# ---------------------------------------------------------------------------
# 遗传算法（实数编码）
# ---------------------------------------------------------------------------
def _run_genetic(obj, space, starts, rng, cfg) -> tuple:
    pop_size = int(cfg["population"])
    generations = int(cfg["generations"])
    elite = max(1, min(int(cfg["elite"]), pop_size))
    if pop_size < 2:
        raise ValueError("population 必须 ≥2")
    U = float(space.u_bound)
    dim = space.dim
    max_evals = int(cfg.get("max_evals", 0))

    def f(u):
        return obj(space.from_u(u))

    X = rng.uniform(-U, U, size=(pop_size, dim))
    for i, st in enumerate(starts[:pop_size]):
        X[i] = space.to_u(st.theta)
    costs = np.asarray([f(x) for x in X], dtype=float)

    best_i = int(np.argmin(costs))
    theta_star = space.from_u(X[best_i])
    cost_star = float(costs[best_i])
    start_results = [{"name": st.name, "theta": st.theta.tolist(),
                      "cost": float(obj(st.theta))} for st in starts]

    def tournament():
        i, j = rng.integers(0, pop_size, size=2)
        return X[i] if costs[i] <= costs[j] else X[j]

    for gen in range(generations):
        order = np.argsort(costs)
        newX = [X[i].copy() for i in order[:elite]]
        newC = [float(costs[i]) for i in order[:elite]]
        while len(newX) < pop_size:
            p1, p2 = tournament(), tournament()
            if rng.random() < float(cfg["crossover_prob"]):
                lo = np.minimum(p1, p2)
                hi = np.maximum(p1, p2)
                d = hi - lo
                a = float(cfg["alpha"])
                child = rng.uniform(lo - a * d, hi + a * d)
            else:
                child = p1.copy()
            mask = rng.random(dim) < float(cfg["mutation_prob"])
            if np.any(mask):
                child[mask] += rng.normal(
                    0.0, float(cfg["mutation_sigma"]), size=int(mask.sum()))
            child = np.clip(child, -U, U)
            newX.append(child)
            newC.append(float(f(child)))
        X = np.vstack(newX)
        costs = np.asarray(newC)
        i = int(np.argmin(costs))
        if costs[i] < cost_star:
            cost_star = float(costs[i])
            theta_star = space.from_u(X[i])
        if max_evals > 0 and obj.n_unique >= max_evals:
            break

    return theta_star, cost_star, start_results, {
        "population": pop_size, "generations": generations,
        "elite": elite,
    }


# ---------------------------------------------------------------------------
# 模拟退火（经典 Metropolis）
# ---------------------------------------------------------------------------
def _run_simulated_annealing(obj, space, starts, rng, cfg) -> tuple:
    n_iters = int(cfg["n_iters"])
    n_chains = max(1, int(cfg["n_chains"]))
    T0_frac = float(cfg["T0_frac"])
    alpha = float(cfg["alpha"])
    sigma = float(cfg["sigma"])
    U = float(space.u_bound)
    dim = space.dim

    def f(u):
        return obj(space.from_u(u))

    chains = list(starts)
    while len(chains) < n_chains:
        base = chains[len(chains) % len(starts)].theta
        noise = rng.normal(0.0, 1.0, size=space.dim) * 0.1 * \
            np.concatenate([space.Imax, space.Imax])
        chains.append(InitialPoint(f"sa_{len(chains) + 1}",
                                   space.clip(base + noise)))

    per_chain = max(1, n_iters // n_chains)
    theta_star, cost_star = None, float("inf")
    start_results = []
    for st in chains[:n_chains]:
        x = space.to_u(st.theta)
        fx = f(x)
        best_x, best_fx = x.copy(), float(fx)
        T = T0_frac * max(abs(fx), 1.0)
        for _ in range(per_chain):
            x_new = np.clip(x + rng.normal(0.0, sigma, size=dim), -U, U)
            f_new = f(x_new)
            if f_new < fx or rng.random() < np.exp(
                    -(f_new - fx) / max(T, 1e-12)):
                x, fx = x_new, f_new
            if fx < best_fx:
                best_x, best_fx = x.copy(), float(fx)
            T *= alpha
        theta = space.from_u(best_x)
        cost = obj(theta)
        start_results.append({
            "name": st.name, "theta": theta.tolist(), "cost": float(cost),
            "final_temperature": float(T),
        })
        if cost < cost_star:
            cost_star, theta_star = cost, theta.copy()
    return theta_star, cost_star, start_results, {
        "n_iters": n_iters, "n_chains": n_chains, "T0_frac": T0_frac,
        "alpha": alpha, "sigma": sigma,
    }


# ---------------------------------------------------------------------------
# 注册表与统一入口
# ---------------------------------------------------------------------------
_RUNNERS = {
    "coordinate": _run_coordinate,
    "random": _run_random,
    "nelder-mead": _run_nelder_mead,
    "powell": _run_powell,
    "differential-evolution": _run_differential_evolution,
    "dual-annealing": _run_dual_annealing,
    "genetic": _run_genetic,
    "simulated-annealing": _run_simulated_annealing,
}


def run_algorithm(name: str, objective, space: ParamSpace, starts, *,
                  seed: int = 0, config: dict | None = None) -> OptimizeResult:
    """运行一个算法；objective(theta)->float，返回统一结果。"""
    name = str(name)
    if name not in _RUNNERS:
        raise ValueError(
            f"未知算法 {name!r}；可选: {', '.join(available_algorithms())}")
    starts_n = _norm_starts(starts, space)
    cfg = _cfg(name, config)
    rng = np.random.default_rng(int(seed))
    cache = CachedObjective(objective, space)
    t0 = time.time()
    theta, cost, start_results, meta = _RUNNERS[name](
        cache, space, starts_n, rng, cfg)
    if theta is None:
        theta = starts_n[0].theta
        cost = cache(theta)
    theta = space.clip(theta)
    cost = float(cache(theta))
    return OptimizeResult(
        algorithm=name, theta=theta, cost=cost,
        n_calls=cache.n_calls, n_unique=cache.n_unique,
        wall_s=time.time() - t0,
        history=cache.history, start_results=start_results, meta=meta,
    )
