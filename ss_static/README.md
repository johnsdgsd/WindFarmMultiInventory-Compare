# 非时变 (s,S)（方法自包含）

按 `docs/s,S.md` 实现通用 10 参数 \((s_r,S_r)\)，主基准无主动调拨
（`mode_tau=0`）。本目录**自包含**评价函数、仿真器、履约、策略、优化器、
归档和报告；`common/` 只提供模型加载、配置、基线读取和通用配对统计。

另有可插拔主动调拨变体（mode 1，`active.py::ThresholdBalanceRule`）：
**每仓保留/目标水平 h 是优化参数**（active 3K 维，no-active 2K）。
\(X_r>h_r\) 时最多调出 \(X_r-h_r\)，\(X_m<h_m\) 时补到 \(h_m\)；
没有单独的目标 T。
no-active 与 active 两个 arm 都正式报告。

## 文件

```
evaluation.py   ★ 评价函数：θ -> 成本/明细
simulator.py    纯 NumPy 状态转移、履约、成本、审计（由 common 迁入）
policy.py       PolicyBase（被动调拨）+ SSStaticPolicy（订货）
active.py       可插拔主动调拨规则（mode 1，目标5 + 每仓h优化）
metrics.py      行为统计、时间剖面、期末值敏感性
starts.py       无限库存基线、解析初值、搜索半径
optimize.py     ParamSpace、CachedObjective、8 种算法、注册表
calibrate.py    训练/验证路径、算法调度、候选选择、归档
evaluate.py     测试集评估、配对统计、summary.md
archive.py      时间戳目录、run.json、index.csv、summary 写入
config/policy.toml
config/policy_active.toml
run_calibrate.py / run_evaluate.py
test/
```

## 评价函数

```python
from ss_static.evaluation import SSEvaluator, evaluate_cost

ev = SSEvaluator(prob, options)
# no-active: theta = [s_1..s_K, S_1..S_K]（2K）
# active:    theta = [s_1..s_K, S_1..S_K, h_1..h_K]（3K）
cost = ev.evaluate_cost(theta, paths)
res = ev.evaluate(theta, paths, record=True)   # 明细：五项、行为、arrays
```

- `paths` 是 `(P,T)` 的整数场景号；
- 单次评价 = 在该路径集合上的平均总成本 \(E(\theta;\mathcal P)\)；
- `record=False` 时也返回 `(P,5)` 五项合计，但不保存逐期大数组；
- 目标口径见 `docs/method-ss-static.md` 第 6 节。

## 算法

`ss_static/optimize.py` 注册表：

| 名称 | 方法 |
|---|---|
| `coordinate` | 坐标下降（网格、多起点，MD §9） |
| `random` | LHS/均匀采样 + 局部高斯精修 |
| `nelder-mead` | SciPy Nelder-Mead（u 空间） |
| `powell` | SciPy Powell（u 空间 + bounds） |
| `differential-evolution` | SciPy DE（u 空间） |
| `dual-annealing` | SciPy dual annealing（u 空间） |
| `genetic` | 实数编码 GA（锦标赛、BLX-α、变异、精英） |
| `simulated-annealing` | 经典 Metropolis 退火 |

所有算法共用 `0≤s≤S≤Imax` 参数空间、缓存、多起点和确定性种子；
预算与默认参数见 `config/policy.toml` 的 `[search.*]`。

## 主动调拨（可选 mode 1）

`active.py::ThresholdBalanceRule`：

- **每仓保留/目标水平 h 是优化参数**，与 s,S 一起搜索（3K）。
- 被动履约后 \(X=S^{end}\)：\(X_r>h_r\) 的仓最多调出 \(X_r-h_r\)；
  \(X_m<h_m\) 的仓补到 \(h_m\)；按盈余/缺口贪心，扣容量/来源/守恒约束；
  每仓每期调出比例 α。
- h 是源仓保留下限/触发线；h=Imax 等价于该仓不作为调出源。
- 最后 2 期默认不调；下一期失效用 300 场景池平均预测。
- 配置 `config/policy_active.toml`；no-active 与 active 都正式报告。
- 新一轮 active 标定统一用遗传算法（`--algo genetic`）。

```bash
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_calibrate.py \
  --policy-config ss_static/config/policy_active.toml \
  --algo coordinate --tag ss_static_T60_204105_active
COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<active_calib>/policy.json \
  --policy-config ss_static/config/policy_active.toml \
  --tag ss_static_T60_204105_active
```

## 运行

```bash
cd "D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-对比实验"
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"

# 测试
"$P" -B ss_static/test/_t_policy.py
"$P" -B ss_static/test/_t_active.py
"$P" -B ss_static/test/_t_evaluation.py
"$P" -B ss_static/test/_t_optimize.py
"$P" -B ss_static/test/_t_archive.py
"$P" -B ss_static/test/_t_audit.py
"$P" -B ss_static/test/_t_timing.py
"$P" -B ss_static/test/_t_pathgen.py
"$P" -B ss_static/test/_t_baseline.py
"$P" -B ss_static/test/_t_calibrate.py

# T60
"$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T60
"$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T60_calib>/policy.json --tag ss_static_T60

# T240
COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T240
COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T240_calib>/policy.json --tag ss_static_T240

# 算法对比（T60，固定路径/预算）
"$P" -B ss_static/run_calibrate.py --algo all \
  --train-paths 500 --val-paths 500 --budget-unique 1500 \
  --tag ss_static_T60_algo_compare
```

`--archive-root` 可指定归档根（缺省 `<项目>/results`）；
`--quick` 只做 60/60 路径的 coordinate smoke；`--integer` 为整数敏感性。
`run_evaluate.py --policy` 必填，避免误取 test/smoke 或其他 T 的策略。

## 归档

```
results/ss_static/
  index.csv
  <tag>_calib_<timestamp>/
    run.json, policy.json, calibration.md, conclusion.md,
    algorithm_summary.csv, algorithm_starts.csv, optimization_history.csv,
    paths_train.npz, paths_val.npz, provenance.json, config/
  <tag>_eval_<timestamp>/
    run.json, policy.json, sim_<P>paths.npz/.json, sim_report_*.txt,
    comparison.json/md, summary.md, conclusion.md, time_profile.csv,
    time_profile_deciles.csv, terminal_sensitivity.json, provenance.json,
    config/, calibration/
```

`summary.md` 是对比验证总结；`calibration.md` 是标定/算法对比总结；
`index.csv` 是 append-only 跨运行索引。每次运行新建目录，不覆盖。

## 当前结果（2026-10-26）

- T60 标定 `results/ss_static/ss_static_T60_calib_20261026_181316`；
  评估 `results/ss_static/ss_static_T60_eval_20261026_181640`；
  策略 1171.5302 vs 基线 825.4215 万元/路径，配对差 +346.1087（+41.93%），审计通过。
- T240 标定 `results/ss_static/ss_static_T240_calib_20261026_181533`；
  评估 `results/ss_static/ss_static_T240_eval_20261026_182140`；
  策略 188154.7413 vs 基线 159380.8723，配对差 +28773.8690（+18.05%），审计通过。
- T60 算法对比（500/500，budget 1500）：NM 最优（验证 1138.93），
  测试 1161.9423（+40.77%）；全量 NM 确认测试 1162.8456（+40.88%），
  均显著优于坐标下降主结果（1171.5302），但仍远差于 SDDP。
- 新 T60 基线 `base_T60_20261026_204105`（混合 n0，3000 测试路径）：
  **s,S 新时序（无 q_in/q0，2 层在途）**；GA，s=S=Imax；
  no-active 标定 `..._noactive_ga_noqin_calib_20261030_110033`，
  评估 `..._noactive_ga_noqin_eval_20261030_110621`；
  **37950.9220 vs 35833.3228，+2117.5992（+5.91%），t=36.23，审计通过**。
  旧 q_in 口径 41394.9455（+15.52%）仅作历史。
- **active（mode 1，h 同时为源仓下限和调入目标，GA，新时序）**：
  calib `..._active_ga_hboth_calib_20261030_112535`，
  eval `..._active_ga_hboth_eval_20261030_113357`；
  s=[12.00,7.84,20.00,14.99,7.98]，S=[12,8,20,14.99,8.00]，
  h=[6.278,4.355,11.079,6.095,2.818]；
  **37338.0637 vs 基线 35833.3228（+1504.7409，+4.20%，t=23.66，审计通过）**；
  active vs no-active **-612.8583（-1.6149%，t=-39.16，胜率 76.50%）**。
  旧 q_in active（+13.36%/+13.65%）仅作历史。
- 跨运行索引：`results/ss_static/index.csv`；跨运行总结：
  `results/ss_static/SUMMARY_20261027_164929.md`；每 run 的
  `summary.md`/`calibration.md`/`conclusion.md` 是最终口径来源。

## 口径与限制

- 策略严格按 `docs/s,S.md`：严格 \(IP<s\)、`R` 截断、另加模型 \(Q^{\max}\) 截断、
  无 stop offset、\(q_T\) 不计采购费。
- mode 0 无主动调拨；被动调拨为本地优先 + 全配对按 `tc` 升序贪心。
- 成本单位万元；库存/待修是台/件。
- 测试路径直接取基线 npz 的 `scene`；训练/验证 seed 777/888；
  测试不参与搜索/选择。
- 静态规则 state-only；SDDP 的阶段决策使用更丰富信息。差异不是单纯的订货规则。
- 基线 LB 是模型下界，不是基线策略成本；不要把它当作静态策略的下界。
