from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ss_static.optimize import (InitialPoint, ParamSpace,  # noqa: E402
                                available_algorithms, run_algorithm)

sys.stdout.reconfigure(errors="replace")

PASS = 0


def check(cond, msg):
    global PASS
    if not cond:
        raise AssertionError(msg)
    PASS += 1


def main():
    global PASS
    Imax = np.array([2.0, 3.0])
    space = ParamSpace(Imax)
    target = np.array([0.5, 0.8, 1.2, 2.0])
    start = np.array([0.0, 0.0, 2.0, 3.0])

    def f(theta):
        t = np.asarray(theta, float)
        return float(np.sum((t - target) ** 2))

    start_cost = f(start)

    # 参数空间
    rng = np.random.default_rng(3)
    raw = rng.uniform(-1.0, 4.0, size=(50, 4))
    clipped = space.clip_many(raw)
    check(np.all(clipped[:, :2] >= -1e-12), "clip s<0")
    check(np.all(clipped[:, :2] <= clipped[:, 2:] + 1e-12), "clip s>S")
    check(np.all(clipped[:, 2:] <= Imax[None, :] + 1e-12), "clip S>Imax")
    rt = space.from_u(space.to_u(target))
    check(float(np.max(np.abs(rt - target))) < 1e-6,
          f"u round-trip {rt} vs {target}")
    check(np.allclose(space.clip(np.array([0.0, 0.0, 1e-9, 1e-9])), 0.0),
          "S 近 0 未吸附")
    check(np.allclose(space.clip(np.array([2.0, 3.0, 2.0, 3.0])),
                      [2.0, 3.0, 2.0, 3.0]), "上界吸附")
    sampled = space.sample(rng, 20)
    check(sampled.shape == (20, 4), f"sample shape {sampled.shape}")
    check(np.all(sampled[:, :2] <= sampled[:, 2:] + 1e-12), "sample s>S")
    check(np.all(sampled[:, 2:] <= Imax[None, :] + 1e-12), "sample S>Imax")
    sp_int = ParamSpace(Imax, integer=True)
    ti = sp_int.clip(np.array([0.4, 0.6, 0.8, 2.4]))
    check(np.allclose(ti, np.rint(ti)), f"integer clip {ti}")
    check(np.all(ti[:2] <= ti[2:] + 1e-12), f"integer clip s>S {ti}")

    starts = [InitialPoint("start", start, radius=np.array([2.0, 2.0, 3.0, 3.0]))]
    cfgs = {
        "coordinate": {"n_grid": 5, "max_rounds": 1},
        "random": {"n_samples": 10, "n_top": 3, "n_refine": 5},
        "nelder-mead": {"max_evals": 80},
        "powell": {"max_evals": 80},
        "differential-evolution": {"maxiter": 2, "popsize": 3, "restarts": 1},
        "dual-annealing": {"maxiter": 5, "maxfun": 80, "restarts": 1},
        "genetic": {"population": 8, "generations": 2, "elite": 2,
                    "max_evals": 80},
        "simulated-annealing": {"n_iters": 50, "n_chains": 2},
    }
    check(set(cfgs) == set(available_algorithms()), "算法注册表不一致")
    results = {}
    for name in available_algorithms():
        r = run_algorithm(name, f, space, starts, seed=7, config=cfgs[name])
        results[name] = r
        check(r.theta.shape == (4,), f"{name} theta shape")
        check(np.allclose(space.clip(r.theta), r.theta), f"{name} theta 不可行")
        check(np.all(r.theta[:2] <= r.theta[2:] + 1e-12), f"{name} s>S")
        check(np.all(r.theta[2:] <= Imax + 1e-12), f"{name} S>Imax")
        check(r.cost <= start_cost + 1e-9,
              f"{name} 比起点更差: {r.cost} > {start_cost}")
        check(r.n_unique >= 1 and len(r.history) == r.n_unique,
              f"{name} history/n_unique 不一致")
    for name in ("coordinate", "nelder-mead", "powell"):
        check(results[name].cost < start_cost - 1e-6,
              f"{name} 未在二次目标上改进")

    # 确定性
    a = run_algorithm("random", f, space, starts, seed=11,
                      config={"n_samples": 10, "n_top": 3, "n_refine": 5})
    b = run_algorithm("random", f, space, starts, seed=11,
                      config={"n_samples": 10, "n_top": 3, "n_refine": 5})
    check(abs(a.cost - b.cost) < 1e-12 and np.allclose(a.theta, b.theta),
          "random 不可复现")
    check(a.n_unique <= 1 + 10 + 5, f"random 预算超限 {a.n_unique}")

    print(f"_t_optimize: {PASS} checks passed")
    print("  " + "  ".join(f"{k}={v.cost:.3g}" for k, v in results.items()))


if __name__ == "__main__":
    main()
