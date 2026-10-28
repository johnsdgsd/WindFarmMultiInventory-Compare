# CLAUDE.md（对比实验项目交接）

> 本轮按用户要求重构：每个验证方法**自包含**评价函数、仿真器、策略、优化器、
> 归档和报告；`common/` 只保留模型加载、配置、基线读取、通用配对统计和文件工具。
> 旧 `results/ss_static/` 已按用户要求清空；不再沿用旧结论。

## 1. 项目目标

把对照方法与主项目的 SDDP 基线放在**同一模型、同一场景池、同一测试路径**上比较，
量化 SDDP 相对简单策略的收益，并给论文提供可复现、可归档的实验结果。

主项目（只读）：

```
D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python
```

当前已实现**非时变 (s,S)**（`ss_static/`）和**滚动时域（全窗口两阶段 SAA）**
（`rolling_horizon/`）；`ss_timevarying/` 仍是骨架。每个方法各自维护
simulator/policy/evaluation，不共享 common 仿真器。

## 2. 目录与职责

```
config/compare.toml           T60 共享配置
config/compare_T240.toml      T240 共享配置
common/config.py              读 TOML；COMPARE_CONFIG
common/model.py               只读导入主模型；problem_data.json；provenance
common/baseline.py            基线 npz 加载；paired_stats/term_stats
common/report.py              结果目录/配置副本/provenance/文件工具
ss_static/
  evaluation.py               ★ 评价函数 θ -> 成本/明细
  simulator.py                纯 NumPy 状态/履约/成本/审计（方法自包含）
  policy.py                   PolicyBase + SSStaticPolicy
  metrics.py                  行为、时间剖面、期末值敏感性
  starts.py                   无限库存基线、解析初值、半径
  optimize.py                 ParamSpace + CachedObjective + 8 种算法
  calibrate.py                训练/验证、算法调度、选择、归档
  evaluate.py                 测试评估、配对统计、summary.md
  archive.py                  run 目录、run.json、index.csv
  config/policy.toml          策略 + 算法预算
  run_calibrate.py            CLI --algo/--tag/--quick/--integer/--archive-root
  run_evaluate.py             CLI --policy（必填）/--tag/--skip-audit
  test/                       自包含测试
docs/s,S.md                   用户提供的权威策略口径（保留）
docs/滚动时域.md              用户提供的权威滚动时域模型/协议（保留）
docs/method-ss-static.md      ★ 评价函数数学、算法、验证、归档
docs/method-rolling-horizon.md ★ 滚动时域实现、口径勘误、验证
test/_t_common.py             共享模型/配置自检
results/                      归档（只新增）
ss_timevarying/               方法 2 骨架
rolling_horizon/              ★ 方法 3：滚动时域（已实现）
  scenarios.py                固定优化池 N×T；基线路径子集
  lp.py                       COPT loadMatrix 两阶段 LP/追索 LP
  policy.py                   order/fulfill/active_transfer；单一 q；并行
  simulator.py/metrics.py     复用 ss_static 迁入的仿真与行为统计
  evaluate.py/run_evaluate.py 样本外评估、配对、审计、LP 统计、归档
  config/policy.toml          N=10/M=50/seed_opt=2027/threads=4/workers=4
  test/                       _t_pool/_t_lp/_t_rollout/_t_evaluate
```

`ss_static` 现在是 Python 包（有 `__init__.py`），模块用
`from ss_static.xxx import ...`；入口脚本把项目根和 `ss_static/` 放进 `sys.path`。

## 3. 环境与命令

```
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
```

所有命令在项目根运行，并加 `-B`（不写主项目 `.pyc`）。

### 3.1 测试

```bash
"$P" -B test/_t_common.py
"$P" -B ss_static/test/_t_policy.py
"$P" -B ss_static/test/_t_evaluation.py
"$P" -B ss_static/test/_t_optimize.py
"$P" -B ss_static/test/_t_archive.py
"$P" -B ss_static/test/_t_audit.py
"$P" -B ss_static/test/_t_lp_equiv.py
"$P" -B ss_static/test/_t_pathgen.py
"$P" -B ss_static/test/_t_baseline.py
"$P" -B ss_static/test/_t_calibrate.py
"$P" -B rolling_horizon/test/_t_pool.py
"$P" -B rolling_horizon/test/_t_lp.py
"$P" -B rolling_horizon/test/_t_rollout.py
"$P" -B rolling_horizon/test/_t_evaluate.py
```

`_t_lp_equiv.py` 需要 COPT，固定 q/y 放进主模型 LP，逐项对拍 290 项；
`_t_evaluation`/`_t_optimize`/`_t_archive` 是新增的方法自包含测试；
`_t_archive` 用临时目录，不污染 `results/`。

### 3.2 T60

```bash
"$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T60
"$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T60_calib>/policy.json --tag ss_static_T60
```

### 3.3 T240

```bash
COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T240

COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T240_calib>/policy.json --tag ss_static_T240
```

`COMPARE_CONFIG` 支持相对项目根的路径；不设时用 `config/compare.toml`。

### 3.4 算法对比/快速试跑

```bash
# 8 种算法在 T60 上对比（固定路径/约同一真实评估预算）
"$P" -B ss_static/run_calibrate.py --algo all \
  --train-paths 500 --val-paths 500 --budget-unique 1500 \
  --tag ss_static_T60_algo_compare

# smoke
"$P" -B ss_static/run_calibrate.py --quick --tag ss_static_T60_quick

# 整数敏感性
"$P" -B ss_static/run_calibrate.py --algo coordinate --integer --tag ss_static_T60_int

# 测试/实验临时归档
"$P" -B ss_static/run_calibrate.py --quick --archive-root /tmp/ss_test
```

### 3.5 滚动时域

```bash
# 测试
"$P" -B rolling_horizon/test/_t_pool.py
"$P" -B rolling_horizon/test/_t_lp.py
"$P" -B rolling_horizon/test/_t_rollout.py
"$P" -B rolling_horizon/test/_t_evaluate.py

# smoke（N=2/M=2，目录带 _quick）
"$P" -B rolling_horizon/run_evaluate.py --quick

# T60 正式（N=10/M=50，COPT；当前后台运行的 tag）
"$P" -B rolling_horizon/run_evaluate.py \
  --n-scenarios 10 --real-paths 50 \
  --workers 4 --threads 4 --tag rolling_T60_204105_final

# 指定其他共享配置（如 T240；N/M 需自行控制）
"$P" -B rolling_horizon/run_evaluate.py \
  --config config/compare_T240.toml \
  --n-scenarios 5 --real-paths 10 --tag rolling_T240
```

`--solver highs` 可切 HiGHS；`--archive-root` 可改归档根；
`--quick` 只作 smoke，不是正式结论。

`--algo` 可选：`coordinate,random,nelder-mead,powell,differential-evolution,
dual-annealing,genetic,simulated-annealing`，或 `all`。
`run_evaluate.py --policy` 必填；不再自动取“最新”策略，避免混 T。

## 4. 评价函数与优化

### 4.1 评价函数

`ss_static/evaluation.py`：

```python
cost = evaluate_cost(prob, options, theta, paths)      # 纯函数
ev = SSEvaluator(prob, options)
r = ev.evaluate(theta, paths, record=True)             # 明细
```

- `theta=[s_1..s_K, S_1..S_K]`，`paths=(P,T)` 场景号；
- 目标：路径平均总成本 \(E(\theta;\mathcal P)\)；
- 仿真口径：虚拟 q0 → 期初状态 → 订货 → 本期 Θ/失效 → 被动调拨 → 成本 → 转移；
- q0 在 t=1 计采购费；q_T 不计；q≤Qmax；无 stop offset；mode 0 无主动调拨；
- `record=False` 也累计 `(P,5)` 五项合计，避免优化时保存大数组；
- 仿真/履约/策略全部在 `ss_static/`，不依赖 common 的策略逻辑。

### 4.2 参数空间与算法

`ss_static/optimize.py`：

- `ParamSpace`：\(0\le s\le S\le I^{\max}\)；`clip` 修复；整数模式取整；
- sigmoid 变换：\(s=I^{\max}\sigma(u)\)，\(S=s+(I^{\max}-s)\sigma(v)\)，
  \(u,v\in[-12,12]\)，解码后吸附边界；
- `CachedObjective`：物理 θ 四舍五入 9 位缓存；`n_calls/n_unique`；
- 算法：`coordinate`、`random`、`nelder-mead`、`powell`、
  `differential-evolution`、`dual-annealing`、`genetic`、`simulated-annealing`；
- 每个算法独立缓存和随机种子；多起点/多 restart；预算见
  `ss_static/config/policy.toml` 的 `[search.*]`；
- 算法比较以 `n_unique` 和实际墙钟为准；SciPy 的 `maxfev/maxiter` 是近似上限。

## 5. 模型/基线/测试口径（不能混）

### 5.1 基线

| | T60 | T240 |
|---|---|---|
| 结果目录 | `results/base_T60_20261025_131043` | `results/base_20261024_153043` |
| 测试路径 | `sim_2000paths.npz`，2000×60 | `sim_1000paths.npz`，1000×240 |
| LB | 712.9522 | 112,307.44 |
| 基线仿真均值 | 825.4215 | 159,380.87 |
| gap | 13.6257% | 29.535% |
| 场景池 | 300 条 | 300 条（逐位相同） |

- 测试路径直接取基线 npz 的 `scene` 数组；**不要重新生成**，不要混 T。
- `problem_data.json` 是模型/场景池的事实源；`config/problem.toml` 只作留痕。
- 基线 LB 是模型下界，不是基线策略成本；不要把“策略成本−LB”当作策略 gap。

### 5.2 训练/验证/测试

| 用途 | 路径 | 是否参与选择 |
|---|---|---|
| 训练 | seed 777，`train_paths` 条 | 是（优化目标） |
| 验证 | seed 888，`val_paths` 条 | 是（选最终策略） |
| 测试 | 基线 npz 的 `scene` | **否** |

- 多起点/多算法在验证集上比较；测试只跑一次；
- 选择规则：严格验证成本最低；仅数值并列（≤1e-9）时选 ΣS 更小者，并在
  `policy.json`/`calibration.md` 写明；每个算法/起点的最终 θ 都在验证集上重评；
- 测试路径不参与搜索、起点选择或调参。

## 6. 结果与归档

```
results/ss_static/
  index.csv                         # append-only: run_id,kind,status,T,n_paths,
                                    # algorithm,selected_start,train/val/test,
                                    # baseline_mean,diff,rel,win,audit,paths
  <tag>_calib_<timestamp>/
    run.json, policy.json, calibration.md,
    algorithm_summary.csv, algorithm_starts.csv, optimization_history.csv,
    paths_train.npz, paths_val.npz, provenance.json, config/
  <tag>_eval_<timestamp>/
    run.json, policy.json, sim_<P>paths.npz/.json, sim_report_*.txt,
    comparison.json/md, summary.md,
    time_profile.csv, time_profile_deciles.csv, terminal_sensitivity.json,
    provenance.json, config/, calibration/
```

- 每次运行新时间戳目录，禁止覆盖；失败运行也写 `run.json` 的 failed 状态。
- `summary.md` = 对比验证总结；`calibration.md` = 标定/算法对比总结。
- `summary.md` 必须包含：T/P/基线/策略/算法、训练/验证/测试协议、配对差/CI/t/胜率、
  五项成本、行为（Σ仓）、时间剖面、期末值敏感性、审计、结论与限制。
- 旧 `results/ss_static/` 已按用户要求删除；不要引用旧结论。

## 7. 已验证的关键口径

- 成本单位是**万元**；库存/待修/失效/替换是台/件。
- 行为 `mean_inventory` 统一为**五仓合计**（旧基线的每仓口径是 5 倍误差）；
  用持有成本恒等式核对：\(hT(\bar I_{pol}-\bar I_{base})\approx\) 持有成本差。
- 五项成本与主项目 LP 同口径：`_t_lp_equiv.py` 290 项通过。
- 审计逐 (路径,期) `check_stage` + \(q\in[0,Q^{max}]\)；报告里
  `terms_vs_stage_cost_max_dev` 只是仿真器内部恒等式，不是 LP 对拍。
- 静态规则 state-only；SDDP 的 \(q^{out}\) 是阶段决策、使用更丰富信息；
  论文中必须写明信息集/策略类差异。
- 无 stop offset、\(q_T\) 不计采购费、mode 0 无主动调拨：MD + 模型口径，
  不要只在策略层改。
- \(q\le Q^{max}\) 是模型变量上界；去掉需要放宽模型并重训 SDDP 基线。

## 8. 当前结果（2026-10-26）

### 8.1 T60 主验证（坐标下降，2000/2000 训练/验证）

- 标定：`results/ss_static/ss_static_T60_calib_20261026_181316`
- 评估：`results/ss_static/ss_static_T60_eval_20261026_181640`
- 选中：`coordinate` / `one_sigma`；s=[0.7692, 0.5209, 0.7808, 0.4341, 0.507]，
  S=[0.8811, 0.5209, 0.7808, 0.4341, 0.507]
- 训练 1154.0274，验证 1184.2502；`n_unique=1016`，标定墙钟约 136 s
- 测试：策略 1171.5302 vs 基线 825.4215（万元/路径）；配对差 **+346.1087**
  （SE 4.8278，95% CI [+336.6463, +355.5710]，t=71.69，胜率 0.60%，
  相对差 **+41.93%**）；审计通过
- 五项差：采购 +211.2919，持有 +84.3916，停机 +18.1440，主动调拨 -14.8853，
  被动调拨 +47.1665
- 行为：平均期初库存 0.9417（基线）vs 2.3385（策略），Σ仓；
  替换率 0.8131 vs 0.9310；期末待修 2.8135 vs 1.2627
- 时间剖面：首 10% 期差 +165.1955，末 10% 期差 +232.7352，
  中间段差 -51.8220；最差十分位第 10 段。**差距不是全程均匀**，而是首/末段+
  无主动调拨/规则履约的混合；中间段静态反而更便宜。
- 期末值敏感性：调整后相对差 +21.97% 到 +41.93%；主目标仍是无期末值口径。

### 8.2 T240 主验证（坐标下降，1000/1000 训练/验证）

- 标定：`results/ss_static/ss_static_T240_calib_20261026_181533`
- 评估：`results/ss_static/ss_static_T240_eval_20261026_182140`
- 选中：`coordinate` / `two_z`；s=[12, 8, 17.5318, 11.9736, 8]，
  S=[12, 8, 20, 14.3835, 8]
- 训练 187588.5039，验证 190791.9615；`n_unique=1065`，标定墙钟约 255 s
- 测试：策略 188154.7413 vs 基线 159380.8723；配对差 **+28773.8690**
  （SE 658.7963，95% CI [+27482.6520, +30065.0860]，t=43.68，胜率 5.60%，
  相对差 **+18.05%**）；审计通过
- 五项差：采购 +749.9845，持有 +5345.9970，停机 +21783.8381，
  主动调拨 -2237.3858，被动调拨 +3131.4353
- 行为：平均期初库存 14.1582（基线）vs 36.4092（策略，Σ仓）；
  替换率 0.8856 vs 0.8736；期末待修 170.4301 vs 189.6591；
  主动调拨量 447.4772（基线）vs 0（策略）
- 解释：T240 的静态策略**不是总量不足，而是空间错配**；没有主动调拨，
  堆到 Imax 仍会缺货（3 个仓 S=Imax）。停机差 21.8k 是最大项；
  实施 mode 1 主动调拨是下一步最可能缩小差距的改动。

### 8.3 T60 算法对比与 NM 确认

主对比：`--algo all --train-paths 500 --val-paths 500 --budget-unique 1500`，
run `results/ss_static/ss_static_T60_algo_compare_calib_20261026_182233`。

| 算法 | 验证成本 | n_unique | 墙钟 s |
|---|---:|---:|---:|
| nelder-mead（选中） | 1138.93 | 1380 | 45.2 |
| powell | 1142.65 | 1499 | 48.6 |
| coordinate | 1145.37 | 824 | 27.0 |
| random | 1145.65 | 1503 | 49.2 |
| simulated-annealing | 1146.13 | 1503 | 55.8 |
| dual-annealing | 1151.33 | 1502 | 48.8 |
| differential-evolution | 1166.15 | 2435 | 79.7 |
| genetic | 1193.97 | 1378 | 45.2 |

- 选中 NM 策略测试：1161.9423 vs 基线 825.4215，配对差 +336.5207（+40.77%），
  审计通过；比坐标下降主结果（1171.5302）低 9.5879（配对 t=-19.53，
  CI [-10.5501,-8.6257]，胜率 69.7%）。
- 全量 NM 确认（2000/2000，budget 1500）：
  `ss_static_T60_nm_full_calib_20261026_184102` / `..._eval_20261026_184452`；
  验证 1175.5963，测试 1162.8456（+40.88%），比坐标下降低 8.6845
  （配对 t=-16.13）。NM 500 策略比 NM 全量低 0.9034（t=3.29），
  但它是 500 路径标定，正式全量口径用 NM 全量。
- 结论：搜索算法会影响 (s,S) 的表现，NM 明显优于坐标下降；但最好的 (s,S)
  仍比 SDDP 基线差约 40.8%。主对比仍按预注册的 coordinate 结果报告，
  NM 作为算法对比/改进项。

### 8.4 结论口径

- 两个 T 上，当前静态 (s,S) 都**显著差于** SDDP 基线（同一测试路径配对）。
- T60 的 +41.9% 被有限视界两端显著放大；写作时必须同时给出时间剖面
  和期末值敏感性，不能写成“稳态下静态订货差 42%”。
- T240 的 +18.1% 更稳健（期末值调整后仍约 +17–18%）；核心缺口是
  无主动调拨与空间错配。
- 新基线 204105（混合 n0）：no-active +15.52%，active +13.65%；
  active vs no-active -1.62%（t=-55.35，胜率 89.0%）；两者都正式报告，
  不预设主次。
- 基线自身有 gap（旧 T60 13.63%，T240 29.54%，新 T60 11.889%）；
  LB 不是基线策略成本。
- 跨运行总结：`results/ss_static/SUMMARY_20261027_164929.md`（含 active）。

### 8.5 新 T60 基线 `base_T60_20261026_204105`（混合初始健康分布）

配置文件：`config/compare_T60_base204105.toml`。

- 与旧 T60 的区别：**只有 `n0` 变成混合初始健康分布**（约 42%/37%/16%/5% 在等级
  0/1/2/3），`N`、`Qmax`、`Imax`、成本系数、300 场景池和 Θ 哈希都与旧 T60 相同。
  因此旧 policy 的 `problem_data_hash` 不匹配，必须重新标定。
- 基线测试集用最新的 3000 路径 sim（`sim_3000paths.npz`）：
  LB=31573.2123，mean=35833.3228，gap=11.889%。
  目录里还有较早的 2000 路径 sim（mean=36663.8448，gap=14.678%），本对比不用它。
- 标定：`results/ss_static/ss_static_T60_204105_calib_20261027_130616`；
  选中 `coordinate` / `two_z`；**s=S=[12, 8, 20, 15, 8]，全部顶到 Imax（边界解）**；
  train 41432.1628，val 41454.8334，n_unique=865。
- 评估：`results/ss_static/ss_static_T60_204105_eval_20261027_131049`；
  41394.9455 vs 35833.3228；配对差 **+5561.6227**
  （95% CI [+5454.8368, +5668.4085]，t=102.08，胜率 1.37%，
  相对差 **+15.52%**）；审计通过。
- **结论文件**：评估目录 `.../conclusion.md`；标定目录
  `ss_static_T60_204105_calib_20261027_130616/conclusion.md`。
- 五项差：采购 +3025.37，持有 -95.90，停机 +814.08，主动调拨 -531.84，
  被动调拨 +2349.91。
- 行为：平均期初库存 24.08（基线）vs 22.35（策略，Σ仓）；替换率 0.9036 vs 0.9532；
  期末待修 51.19 vs 26.23；计费下单 454.52 vs 515.03；主动调拨 106.37 vs 0。
- 时间剖面：首 10% 期 +234.43，末 10% 期 +3287.02，中间段 +2040.18；
  最差十分位第 10 段。期末值敏感性 +9.20% 到 +15.52%。
- 解释：高失效率 + 混合初始健康把静态最优推到 Imax 边界，`s=S=Imax` 是
  “尽量满仓”策略；它服务率更高、期末待修更低，但采购和被动调拨更贵，
  仍显著差于 SDDP。**这是比旧 T60 更有利于静态策略的实例，但结论不变。**
- 该基线使用主项目 HEAD `4ec38f9f...`；新配置已记录。关键测试
  `_t_common`/`_t_audit`/`_t_pathgen`/`_t_lp_equiv`（290 项）在新配置下通过；
  `_t_baseline`/`_t_archive` 已改成按 (P,T) 识别已知基线。

### 8.6 为什么 s=S=Imax：边界解诊断

诊断文件：
`results/ss_static/ss_static_T60_204105_eval_20261027_131049/boundary_diagnostic.md`
和同名 `.json`（500 条验证路径，seed 888；不是 3000 条测试集）。

- **单仓切片**：固定其他仓在 Imax，单独变化某仓的 `S`（取 `s=S`）或 `s`
  （固定 `S=Imax`）；5 个仓的曲线都在 **Imax 处取得最小值**。这不是搜索越界，
  是目标在箱约束内单调到边界。
- **为什么**：`Cs/h=20`，newsvendor 临界比 `β=0.952`；新 `n0` 混合分布使
  `d≈1.2–2.3/仓·月`，远高于旧 T60 的 `≈0.04`；多期、初始待修和 3 期提前期
  进一步把目标推到最大。单期 newsvendor 初值
  `S0=[8.86, 6.75, 11.40, 8.49, 8.24]` 明显低于 Imax，说明边界解不是单期报童解。
- **Imax 是不是太小**：部分是。角点的实现平均库存只有 **22.59（Σ仓）**，
  而 `ΣImax=63`，说明库存并不是“放不下”；真正限制补货的是
  **Qmax（每期 2–3 件）和 3 期提前期**。松弛诊断（修改 Imax/Qmax，
  **不是原模型下的可行策略**）：
  - Imax ×1.5：成本约 **-10.2%**
  - Qmax ×2：约 **-6.3%**
  - 两者 ×1.5：约 **-12.3%**
  - Imax ×2 后收益饱和（约 -10.4%），说明 Imax 确实偏紧但边际收益递减。
- **参数判断**：旧 Imax/Qmax 是为低失效率（`d≈0.04`）标定的；新 `n0` 混合后
  `d≈1.2–2.3`，同一套 Imax/Qmax 明显更紧。若这是有意的“成熟机队/混合健康”
  场景，建议按失效率重标 Imax/Qmax 并**重训 SDDP 基线**；不要只为让 s,S
  好看而单方面放宽边界。即使放宽，静态策略仍无主动调拨、规则履约、state-only，
  结构差异仍在；本次 +15.52% 的差距不能全归因于 Imax。

### 8.7 主动调拨（mode 1，每仓门限 h 优化）

- 机制：`ss_static/active.py::ThresholdBalanceRule`，可插拔；目标 T=5 固定，
  **每仓门限 h 是优化参数**，与 s,S 一起搜索（active 3K，no-active 2K）；
  `active_stop_offset=2`（最后 2 期不调）；下一期失效用 300 场景池平均预测。
- 配置：`ss_static/config/policy_active.toml`；no-active 与 active 都正式报告，
  不预设主次。
- 标定：`results/ss_static/ss_static_T60_204105_active_calib_20261027_163659`
  - `coordinate` / `two_z`；s=S=[12,8,20,15,8]=Imax；
  - h=[9.5132, 5.0, 10.7237, 5.0, 5.0]（仓 1/3 高门限、少调出；仓 2/4/5 取 5）；
  - train 40751.0852，val 40808.2530，n_unique=1428，墙钟约 419 s。
- 评估：`results/ss_static/ss_static_T60_204105_active_eval_20261027_164616`
  - **40723.1142 vs 基线 35833.3228**；配对差 **+4889.7914**（+13.65%）；
    95% CI [+4792.8141, +4986.7688]；t=98.83；胜率 1.67%；审计通过。
- **active vs no-active（同一 3000 测试路径）**：
  **-671.8313 万元/路径（-1.623%）**；95% CI [-695.6203, -648.0422]；
  t=-55.35；胜率 89.03%。主动调拨显著改善 s,S。
- 五项差（active−基线）：采购 +3130.36，持有 -39.41，停机 +94.21，
  主动 -259.42，被动 +1964.05；相对 no-active，停机 -719.87、被动 -385.86，
  但采购 +104.99、主动 +272.42，净 -671.83。
- 行为：平均库存 24.08（基线）vs 23.29（active）；替换率 0.9036 vs 0.9560；
  期末待修 51.19 vs 24.75；主动调拨量 106.37 vs 54.48。
- 时间：首 10% 期 +238.14，中间段 +1531.93，末 10% 期 +3119.72；
  期末值敏感性 +7.17% 到 +13.65%。
- 结论：主动调拨 + h 优化显著改善 s,S（−1.62%），但仍是规则型/state-only，
  **仍比 SDDP 基线差 +13.65%**；不能把主动调拨等同于基线的完整价值函数。
- 结论文件：active eval/calib 各有 `conclusion.md`；no-active 的
  `..._eval_20261027_131049/conclusion.md` 第 10 节也写了 active 对照。
- 跨运行总结：`results/ss_static/SUMMARY_20261027_164929.md`。

### 滚动时域（全窗口两阶段 SAA）

按 `docs/滚动时域.md` 实现，代码在 `rolling_horizon/`：

- 每期用当前状态 + 固定优化池（N×T，seed_opt）的 \([t,T]\) 切片解两阶段 LP，
  只执行 \(q_t\)；真实 \(\Theta_t\) 后解追索 LP 得 \(y,\tau\)；窗口
  \(W=T-t+1\)，\(V_{T+1}=0\)。
- 状态 `I,n,o1..o(L-1)`；初始 `I0/o0/n0` 取 `problem_data`，无额外虚拟 q0；
  每期单一 \(q_t\)，采购按当期 \(Cq_t\)；无 q_in/q_out。
- 容量：每期 `I_t+Σo_t+q_t≤Imax`（`cap_init`/`cap`）和转移后
  `I_{t+1}+Σo_{t+1}≤Imax`（`cap_next`）；追索 LP 有 `cap_end`。
- 追索目标含 `-C_sY` 与 `h/2 I_out`；终端 \(q_T=0\)；允许 \(t=T\) 的两阶段 LP。
- 真实路径取基线 scene 前 M 条（不是独立 seed_real）；优化池与基线 300 场景池
  独立；优化 LP 目标只是样本内 SAA，不作为绩效。
- 采购按当期 `Cq_t`，主 SDDP 按到货 `q_in` 计费；逐期成本剖面须保留时点差异说明。
- 已实现 `workers` 路径级并行、`threads` COPT 内部线程、`progress.log`/`run.log`、
  失败 run 归档；不加安全界/fallback。
- **正式运行进行中**（tag `rolling_T60_204105_final`），**尚无有效正式结果**；
  旧 `capfix`/`simpletest`/`quick` 结果不能作为最终口径；数字以归档目录的
  `summary.md`/`conclusion.md`/`comparison.json` 为准。
- 测试：`rolling_horizon/test/_t_pool.py`、`_t_lp.py`（L-1 状态/两层容量/目标复算/
  t=T q=0/追索 cap_end）、`_t_rollout.py`、`_t_evaluate.py`；现有
  `_t_lp_equiv.py` 仍通过。
- 限制：N=10/M=50 是缩减规模；真实路径不是独立 seed_real；开环时变
  (s_t,S_t) 未实现，不能给 \(C^{roll}-C^{open}\) 归因。

## 9. 不要做

- 不要修改主项目 `code/`、`utils/`、`results/`；只读导入。
- 不要重新采样 300 场景池；不要重新生成测试路径。
- 不要用测试路径做标定/选择；不要把 LB 当基线策略成本。
- 不要覆盖 `results/`；不要删除新结果；失败/中间结果保留。
- 不要在策略层偷偷去掉 Qmax 或加 stop offset；要改就改模型/文档并重跑。
- 不要把 T60 的策略/报告用于 T240（policy.json 有 T/Imax/problem_data_hash 校验）。
- 不要修改 `docs/s,S.md` 的权威口径；实现说明写在 `docs/method-ss-static.md`。

## 10. 下一步

已完成：T60/T240 坐标下降主验证、T60 多算法对比、跨运行 SUMMARY、
新 T60 204105 的 no-active 与 active（mode 1，每仓 h 优化）两臂。

1. （可选）frozen active：固定 no-active 的 s,S，只优化 h，隔离“机制 + h”
   与“s,S 重新优化”的混杂；只用训练/验证选择。
2. （可选）h/target 敏感性：只在训练/验证路径上做，不用 3000 测试调参。
3. T240 active：T240 基线主动调拨 447.48 件/路径，可能比 T60 更值得做；
   需要在 T240 problem data 上重新标定 s,S+h。
4. 实现 `ss_timevarying`（K 段 s,S + lead-time-aware stop offset；自包含模拟器/评价）。
5. 滚动时域：`rolling_T60_204105_final` 完成后把结果回填到 §8 滚动段；
   N=50/100 的规模扩展；T240（N/M 需按 LP 规模评估）；开环时变基准用于归因；
   采购成本时点与 SDDP/s,S 的统一。
6. 论文中 no-active/active/滚动都正式报告；active 仍是规则型/state-only，
   滚动仍受 N/M 与 SAA 近似限制，不能等同于 SDDP 的完整价值函数。
