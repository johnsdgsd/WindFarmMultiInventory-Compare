from __future__ import annotations

import sys

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import csv
import json
from pathlib import Path

import numpy as np

from common.baseline import sha256_array, sha256_file
from common.config import load_compare_config, load_policy_config
from common.model import load_problem
from ss_static.archive import create_run, output_tag
from ss_static.evaluation import SSEvaluator, SSParams
from ss_static.optimize import (ParamSpace, available_algorithms,
                                run_algorithm)
from ss_static.policy import SSStaticPolicy
from ss_static.simulator import generate_scene_paths
from ss_static.starts import (START_NAMES, analytic_starts, infinite_demand,
                              moments)


def normalize_algorithms(algorithms) -> list[str]:
    if algorithms is None:
        return ["coordinate"]
    if isinstance(algorithms, str):
        parts = [a.strip() for a in algorithms.split(",") if a.strip()]
    else:
        parts = [str(a).strip() for a in algorithms if str(a).strip()]
    if not parts:
        raise ValueError("至少选择一个算法")
    if "all" in parts:
        return available_algorithms()
    unknown = set(parts) - set(available_algorithms())
    if unknown:
        raise ValueError(
            f"未知算法 {sorted(unknown)}；可选 {available_algorithms()}")
    return parts


def _validate_search(search: dict, prob) -> dict:
    opts = dict(search or {})
    algorithms = normalize_algorithms(opts.get("algorithms", ["coordinate"]))
    integer = bool(opts.get("integer_params", False))
    seed = int(opts.get("seed", 2026))
    q_r = float(opts.get("q_r", 1.0))
    if q_r <= 0:
        raise ValueError("q_r 必须 >0")
    starts = [str(v) for v in opts.get("starts", list(START_NAMES))]
    unknown = set(starts) - set(START_NAMES)
    if unknown:
        raise ValueError(f"未知 starts: {sorted(unknown)}")
    budget_unique = int(opts.get("budget_unique", 0) or 0)
    if budget_unique < 0:
        raise ValueError("budget_unique 必须 ≥0")
    out = {
        "algorithms": algorithms,
        "integer_params": integer,
        "seed": seed,
        "q_r": q_r,
        "starts": starts,
        "budget_unique": budget_unique,
    }
    for name in available_algorithms():
        sub = opts.get(name, {})
        if sub is None:
            sub = {}
        if not isinstance(sub, dict):
            raise TypeError(f"[search.{name}] 必须是表")
        out[name] = dict(sub)
    return out


def _apply_budget(search: dict, budget: int, n_starts: int, prob,
                  active: bool = False) -> None:
    """把各算法的配置调到约同一真实评估预算（算法对比用）。

    预算是对目标的近似；实际 n_unique 以归档为准。坐标下降按网格上界缩放，
    其余算法按各自的结构换算。active 时维度为 3K。
    """
    b = int(budget)
    if b <= 0:
        return
    dim = (3 if active else 2) * int(prob.N_R)
    ns = max(1, int(n_starts))

    c = dict(search["random"])
    c["n_samples"] = max(10, b // 2)
    c["n_refine"] = max(0, b - c["n_samples"])
    search["random"] = c

    for name in ("nelder-mead", "powell"):
        c = dict(search[name])
        c["max_evals"] = max(10, b // ns)
        search[name] = c

    c = dict(search["differential-evolution"])
    pop = max(4, int(c.get("popsize", 8)) * dim)
    c["maxiter"] = max(1, b // pop)
    search["differential-evolution"] = c

    c = dict(search["dual-annealing"])
    c["maxfun"] = max(dim + 5, b)
    c["maxiter"] = max(10, b // 10)
    search["dual-annealing"] = c

    c = dict(search["genetic"])
    pop = max(2, int(c.get("population", 40)))
    c["generations"] = max(1, b // pop - 1)
    c["max_evals"] = b
    search["genetic"] = c

    c = dict(search["simulated-annealing"])
    c["n_iters"] = max(10, b)
    search["simulated-annealing"] = c

    c = dict(search["coordinate"])
    rounds = max(1, int(c.get("max_rounds", 3)))
    ng = int(c.get("n_grid", 20))
    est = ng * dim * rounds * ns
    if est > b:
        c["n_grid"] = max(2, b // max(1, dim * rounds * ns))
    search["coordinate"] = c


def _calibration_md(prob, cfg, run_id, paths_train, paths_val, d, m,
                    starts, candidates, chosen, search, quick,
                    code_versions: dict, *, active: bool = False,
                    options: dict | None = None) -> str:
    options = dict(options or {})
    mode = 1 if active else 0
    active_txt = ""
    if active:
        active_txt = (
            f"；active_rule={options.get('active_rule')}"
            f"，target={options.get('active_target')}"
            f"，alpha={options.get('active_fraction')}"
            f"，stop_offset={options.get('active_stop_offset')}"
            f"；每仓门限 h 是优化参数")
    h_header = " h |" if active else ""
    h_sep = "---|" if active else ""
    lines = [
        f"# 非时变 (s,S) 标定报告",
        "",
        f"- run_id：`{run_id}`",
        f"- 配置：`{cfg.config_path}`",
        f"- 问题：T={prob.T}，G={prob.G}，N_R={prob.N_R}，L={prob.L}",
        f"- 训练路径：{len(paths_train)} 条（seed={cfg.train_seed}，hash={sha256_array(paths_train)[:16]}…）",
        f"- 验证路径：{len(paths_val)} 条（seed={cfg.val_seed}，hash={sha256_array(paths_val)[:16]}…）",
        f"- 模式：mode_tau={mode}{active_txt}；被动调拨为全配对严格成本优先级",
        f"- 搜索算法：{', '.join(search['algorithms'])}；quick={bool(quick)}；"
        f"integer={bool(search['integer_params'])}；seed={search['seed']}；"
        f"budget_unique={search.get('budget_unique', 0)}",
        f"- 成本单位：万元（见主项目 `config/problem.toml` 的 [cost]）。",
        "",
        "## 解析初值（docs/s,S.md §8）",
        "",
        f"- d={np.round(d, 6).tolist()}",
        f"- mu=L·d={np.round(m['mu'], 4).tolist()}",
        f"- sigma=sqrt(L·d)={np.round(m['sigma'], 4).tolist()}",
        f"- beta=Cs/(Cs+h)={m['beta']:.4f}，z=Phi^-1(beta)={m['z']:.4f}",
        f"- 初值：{[z.name for z in starts]}",
        "",
        "## 算法/起点结果（训练选参数，验证选最终策略）",
        "",
        f"| 算法 | 起点 | s | S |{h_header} 训练成本 | 验证成本 | n_unique | 用时 s |",
        f"|---|---|---|---|{h_sep}---:|---:|---:|---:|",
    ]
    for c in candidates:
        h_cell = (f" {np.round(c.get('h'), 4).tolist()} |"
                  if active and c.get("h") is not None else "")
        lines.append(
            f"| {c['algorithm']} | {c['start']} | "
            f"{np.round(c['s'], 4).tolist()} | {np.round(c['S'], 4).tolist()} |"
            f"{h_cell} "
            f"{c['train_cost']:.4f} | {c['val_cost']:.4f} | "
            f"{c['n_unique']} | {c['wall_s']:.2f} |")
    lines += [
        "",
        "## 选择结果",
        "",
        f"- 选中算法：`{chosen['algorithm']}`；起点：`{chosen['start']}`",
        f"- 选中 s={np.round(chosen['s'], 4).tolist()}，"
        f"S={np.round(chosen['S'], 4).tolist()}",
    ]
    if active and chosen.get("h") is not None:
        lines.append(
            f"- 选中每仓门限 h={np.round(chosen['h'], 4).tolist()}"
            f"（目标 T={options.get('active_target')}）")
    lines += [
        f"- 训练成本：{chosen['train_cost']:.4f}；验证成本：{chosen['val_cost']:.4f}",
        f"- 验证最优：{candidates[0]['val_cost']:.4f}；"
        f"选择规则：严格验证最优；仅数值并列（≤1e-9）时选 ΣS 更小者。",
        "",
        "## 说明",
        "",
        "- 评价函数在 `ss_static/evaluation.py`；仿真器/履约/策略都在 `ss_static/`。",
        "- 各算法/起点明细见 `algorithm_starts.csv`；唯一评价历史见 `optimization_history.csv`。",
        "- 训练/验证路径不参与测试；测试只在 `evaluate.py` 使用基线 npz 的 `scene`。",
        "- q0 在 t=1 计采购费；q_T 不计采购费；q≤Qmax；无 stop offset（docs/s,S.md 口径）。",
        "- active 时 θ=[s,S,h]；每仓门限 h 与 s,S 一起优化，目标 T 固定；"
        "末期默认最后 2 期不主动调拨。",
        "- 成本单位是万元；行为量（库存/待修）是台/件，不是金额。",
        f"- 版本：Python {code_versions.get('python')}，NumPy {code_versions.get('numpy')}，"
        f"SciPy {code_versions.get('scipy')}",
        "",
    ]
    return "\n".join(lines)


def calibrate(project_dir: str | Path, *, tag: str | None = None,
              quick: bool = False, integer: bool | None = None,
              algorithms=None, archive_root=None,
              train_paths: int | None = None,
              val_paths: int | None = None,
              seed: int | None = None,
              budget_unique: int | None = None,
              policy_config=None, active: bool | None = None,
              active_options: dict | None = None) -> Path:
    project_dir = Path(project_dir).resolve()
    cfg = load_compare_config()
    pcfg = load_policy_config(project_dir, policy_config)
    prob = load_problem(cfg)
    options = dict(pcfg["policy"])
    if active is not None:
        options["active_transfer"] = bool(active)
    if active_options:
        for k, v in dict(active_options).items():
            if v is not None:
                options[k] = v
    active_flag = bool(options.get("active_transfer", False))
    active_target = float(options.get("active_target", 5.0))
    active_min = float(options.get("active_threshold_min", active_target))
    active_h_init = float(options.get("active_threshold_init", active_target))
    if active_min < active_target - 1e-12:
        raise ValueError("active_threshold_min 必须 ≥ active_target")
    search = _validate_search(pcfg["search"], prob)
    if integer is not None:
        search["integer_params"] = bool(integer)
    if algorithms is not None:
        search["algorithms"] = normalize_algorithms(algorithms)
    if seed is not None:
        search["seed"] = int(seed)
    if budget_unique is not None:
        if int(budget_unique) < 0:
            raise ValueError("budget_unique 必须 ≥0")
        search["budget_unique"] = int(budget_unique)

    n_train = int(train_paths if train_paths is not None else cfg.train_paths)
    n_val = int(val_paths if val_paths is not None else cfg.val_paths)
    if quick:
        n_train = min(n_train, 60)
        n_val = min(n_val, 60)
        search["algorithms"] = ["coordinate"]
        sub = dict(search["coordinate"])
        sub.update({"n_grid": 5, "max_rounds": 1})
        search["coordinate"] = sub
    if n_train < 1 or n_val < 1:
        raise ValueError("train_paths/val_paths 必须 ≥1")

    out_tag = output_tag(tag, pcfg, prob.T)

    print(f"加载模型: T={prob.T} N_R={prob.N_R} G={prob.G} L={prob.L} "
          f"pool={len(prob.pool)}", flush=True)
    paths_train = generate_scene_paths(
        n_train, cfg.train_seed, cfg.calib_chunk_size, len(prob.pool), prob.T)
    paths_val = generate_scene_paths(
        n_val, cfg.val_seed, cfg.calib_chunk_size, len(prob.pool), prob.T)
    print(f"标定路径 {paths_train.shape} seed={cfg.train_seed}；"
          f"验证路径 {paths_val.shape} seed={cfg.val_seed}", flush=True)

    d = infinite_demand(prob, paths_train)
    m = moments(prob, d)
    starts = analytic_starts(
        prob, d, search, active=active_flag,
        active_target=active_target, active_threshold_init=active_h_init)
    if search["budget_unique"] > 0:
        _apply_budget(search, search["budget_unique"], len(starts), prob,
                      active=active_flag)
    evaluator = SSEvaluator(prob, options)
    space = ParamSpace(
        prob.Imax, integer=search["integer_params"],
        active=active_flag, active_target=active_target,
        active_min=active_min)

    print(f"无限库存基线 d={np.round(d, 6).tolist()} "
          f"mu={np.round(m['mu'], 4).tolist()} "
          f"sigma={np.round(m['sigma'], 4).tolist()} "
          f"beta={m['beta']:.4f} z={m['z']:.4f}", flush=True)
    print(f"起点: {[(z.name, np.round(z.theta, 4).tolist()) for z in starts]}",
          flush=True)

    manifest = {
        "T": int(prob.T), "G": int(prob.G), "N_R": int(prob.N_R),
        "L": int(prob.L), "n_train": n_train, "n_val": n_val,
        "train_seed": int(cfg.train_seed), "val_seed": int(cfg.val_seed),
        "algorithms": search["algorithms"],
        "integer_params": bool(search["integer_params"]),
        "budget_unique": int(search.get("budget_unique", 0)),
        "quick": bool(quick),
        "active_transfer": active_flag,
        "mode_tau": 1 if active_flag else 0,
        "active_options": {k: v for k, v in options.items()
                           if k.startswith("active_") or k == "mode_tau"},
        "policy_config": (str(policy_config) if policy_config is not None
                          else str(project_dir / "config" / "policy.toml")),
        "problem_data": str(cfg.problem_data),
        "problem_data_hash": sha256_file(cfg.problem_data)
        if Path(cfg.problem_data).is_file() else None,
        "config": str(cfg.config_path),
        "policy_config": str(project_dir / "config" / "policy.toml"),
    }
    archive = create_run(cfg, project_dir, "calib", out_tag, archive_root,
                         manifest=manifest)
    try:
        archive.copy_configs(cfg, project_dir, policy_path=policy_config)
        archive.write_npz("paths_train.npz", scene=paths_train)
        archive.write_npz("paths_val.npz", scene=paths_val)

        def objective(theta):
            return evaluator.evaluate_cost(theta, paths_train)

        candidates = []
        raw_results = []
        start_rows = []
        for algo in search["algorithms"]:
            print(f"运行算法 {algo} ...", flush=True)
            res = run_algorithm(
                algo, objective, space, starts,
                seed=search["seed"], config=search.get(algo, {}))
            # 每个起点的最终 θ + 算法整体最优 θ，都在验证集上重评
            points = []
            for sr in res.start_results:
                if "theta" not in sr:
                    continue
                points.append((str(sr.get("name", "start")),
                               np.asarray(sr["theta"], dtype=float),
                               float(sr.get("cost", np.nan))))
            theta_best = np.asarray(res.theta, dtype=float)
            if not any(np.allclose(theta_best, t, atol=1e-9, rtol=0.0)
                       for _, t, _ in points):
                points.append((f"{algo}_best", theta_best, float(res.cost)))
            seen = set()
            uniq = []
            for name, theta, train_cost in points:
                theta = space.clip(theta)
                key = tuple(np.round(theta, 9).tolist())
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((name, theta, train_cost))

            best_for_algo = None
            for name, theta, train_cost in uniq:
                val = evaluator.evaluate(theta, paths_val)
                ss = SSParams.from_theta(theta, prob.N_R, active_flag)
                row = {
                    "algorithm": algo, "start": name,
                    "s": ss.s.tolist(), "S": ss.S.tolist(),
                    "h": (ss.h.tolist() if ss.h is not None else None),
                    "theta": theta.tolist(),
                    "train_cost": (float(train_cost)
                                   if np.isfinite(train_cost) else float("nan")),
                    "val_cost": float(val.cost),
                    "n_calls": int(res.n_calls),
                    "n_unique": int(res.n_unique),
                    "wall_s": float(res.wall_s),
                    "meta": dict(res.meta),
                }
                start_rows.append(row)
                if (best_for_algo is None
                        or row["val_cost"] < best_for_algo["val_cost"] - 1e-12
                        or (abs(row["val_cost"]
                                - best_for_algo["val_cost"]) <= 1e-12
                            and sum(row["S"]) < sum(best_for_algo["S"]))):
                    best_for_algo = row
            candidates.append(best_for_algo)
            raw_results.append((algo, res))
            print(f"  {algo}: n_unique={res.n_unique} wall={res.wall_s:.1f}s "
                  f"best_val={best_for_algo['val_cost']:.4f} "
                  f"start={best_for_algo['start']} "
                  f"s={np.round(best_for_algo['s'], 4).tolist()} "
                  f"S={np.round(best_for_algo['S'], 4).tolist()}", flush=True)

        # 严格验证最优；仅数值并列时选 ΣS 更小者
        candidates.sort(key=lambda z: (z["val_cost"], sum(z["S"])))
        best = candidates[0]
        chosen = best

        ss = SSParams.from_theta(
            np.asarray(chosen["theta"], float), prob.N_R, active_flag)
        final_policy = SSStaticPolicy(prob, ss.s, ss.S, options, h=ss.h)
        policy_doc = {
            **final_policy.params(),
            "theta": ss.theta.tolist(),
            "problem": {
                "T": int(prob.T), "G": int(prob.G),
                "N_R": int(prob.N_R), "L": int(prob.L),
                "Imax": np.asarray(prob.Imax, float).tolist(),
                "problem_data_hash": manifest["problem_data_hash"],
            },
            "calibration": {
                "method": "docs/s,S.md",
                "mode_tau": 1 if active_flag else 0,
                "active_transfer": active_flag,
                "active_options": {
                    k: v for k, v in options.items()
                    if k.startswith("active_") or k == "mode_tau"},
                "quick": bool(quick),
                "train_paths": n_train, "val_paths": n_val,
                "train_seed": int(cfg.train_seed),
                "val_seed": int(cfg.val_seed),
                "train_scene_hash": sha256_array(paths_train),
                "val_scene_hash": sha256_array(paths_val),
                "d": d.tolist(),
                "mu": m["mu"].tolist(), "sigma": m["sigma"].tolist(),
                "beta": float(m["beta"]), "z": float(m["z"]),
                "search": search,
                "algorithms": search["algorithms"],
                "candidates": candidates,
                "selected_algorithm": chosen["algorithm"],
                "selected_start": chosen["start"],
                "train_cost": float(chosen["train_cost"]),
                "val_cost": float(chosen["val_cost"]),
                "val_best": float(best["val_cost"]),
                "selection_rule": "strict validation minimum; exact numeric tie prefer smaller total S",
            },
        }
        archive.write_json("policy.json", policy_doc)
        archive.write_provenance(cfg, prob, final_policy.params())

        # 算法汇总 CSV
        with open(archive.path("algorithm_summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "algorithm", "start", "s", "S", "h", "train_cost",
                "val_cost", "n_calls", "n_unique", "wall_s", "selected"])
            w.writeheader()
            for c in candidates:
                w.writerow({
                    "algorithm": c["algorithm"], "start": c["start"],
                    "s": json.dumps(np.round(c["s"], 9).tolist()),
                    "S": json.dumps(np.round(c["S"], 9).tolist()),
                    "h": (json.dumps(np.round(c["h"], 9).tolist())
                          if c.get("h") is not None else ""),
                    "train_cost": c["train_cost"], "val_cost": c["val_cost"],
                    "n_calls": c["n_calls"], "n_unique": c["n_unique"],
                    "wall_s": c["wall_s"],
                    "selected": int(c["algorithm"] == chosen["algorithm"]
                                    and c["theta"] == chosen["theta"]),
                })
        # 起点汇总 CSV：每个算法/起点的最终 θ 与验证成本
        with open(archive.path("algorithm_starts.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "algorithm", "start", "theta", "h", "train_cost", "val_cost",
                "n_unique", "wall_s", "selected"])
            w.writeheader()
            for r in start_rows:
                w.writerow({
                    "algorithm": r["algorithm"], "start": r["start"],
                    "theta": json.dumps(np.round(
                        np.asarray(r["theta"], float), 9).tolist()),
                    "h": (json.dumps(np.round(r["h"], 9).tolist())
                          if r.get("h") is not None else ""),
                    "train_cost": r["train_cost"],
                    "val_cost": r["val_cost"],
                    "n_unique": r["n_unique"], "wall_s": r["wall_s"],
                    "selected": int(r["algorithm"] == chosen["algorithm"]
                                    and r["theta"] == chosen["theta"]),
                })
        # 唯一评价历史 CSV
        with open(archive.path("optimization_history.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["algorithm", "eval_id",
                                              "theta", "cost"])
            w.writeheader()
            for algo, res in raw_results:
                for i, h in enumerate(res.history):
                    w.writerow({
                        "algorithm": algo, "eval_id": i,
                        "theta": json.dumps(np.round(
                            np.asarray(h["theta"], float), 9).tolist()),
                        "cost": h["cost"],
                    })

        versions = {"python": sys.version.split()[0],
                    "numpy": np.__version__}
        try:
            import scipy
            versions["scipy"] = scipy.__version__
        except Exception:
            versions["scipy"] = None
        archive.write_text("calibration.md", _calibration_md(
            prob, cfg, archive.run_id, paths_train, paths_val, d, m,
            starts, candidates, chosen, search, quick, versions,
            active=active_flag, options=options))
        archive.finalize(
            "completed",
            extra={
                "mode_tau": 1 if active_flag else 0,
                "active_transfer": active_flag,
                "active_options": {k: v for k, v in options.items()
                                   if k.startswith("active_") or k == "mode_tau"},
                "selected_algorithm": chosen["algorithm"],
                "selected_start": chosen["start"],
                "train_cost": float(chosen["train_cost"]),
                "val_cost": float(chosen["val_cost"]),
                "val_best": float(best["val_cost"]),
                "policy": final_policy.params(),
            },
            index_row={
                "run_id": archive.run_id, "kind": "calib", "status": "completed",
                "created_at": archive.created_at,
                "config": cfg.config_path.name, "T": int(prob.T),
                "n_paths": n_train, "algorithm": chosen["algorithm"],
                "selected_start": chosen["start"],
                "train_cost": float(chosen["train_cost"]),
                "val_cost": float(chosen["val_cost"]),
                "policy_json": f"{archive.run_id}/policy.json",
                "summary_md": f"{archive.run_id}/calibration.md",
            })
        print(f"标定完成: {archive.dir}", flush=True)
        return archive.dir
    except Exception as e:
        archive.finalize(
            "failed", extra={"error": repr(e)},
            index_row={
                "run_id": archive.run_id, "kind": "calib", "status": "failed",
                "created_at": archive.created_at, "config": cfg.config_path.name,
                "T": int(prob.T), "n_paths": n_train,
                "policy_json": f"{archive.run_id}/policy.json",
                "summary_md": f"{archive.run_id}/calibration.md",
            })
        raise


def main(project_dir: str | Path | None = None, *, tag: str | None = None,
         quick: bool = False, integer: bool | None = None,
         algorithms=None, archive_root=None, train_paths=None,
         val_paths=None, seed=None, budget_unique=None,
         policy_config=None, active=None, active_options=None) -> Path:
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent
    return calibrate(
        project_dir, tag=tag, quick=quick, integer=integer,
        algorithms=algorithms, archive_root=archive_root,
        train_paths=train_paths, val_paths=val_paths, seed=seed,
        budget_unique=budget_unique, policy_config=policy_config,
        active=active, active_options=active_options)
