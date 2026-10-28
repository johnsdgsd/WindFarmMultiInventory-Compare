# 风场多仓库备件库存 — 方法对比实验

本目录把对照方法与主项目的 SDDP 基线放在**同一模型、同一场景池、同一测试路径**上比较。
主项目只读：

```
D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python
```

当前已实现 **非时变 (s,S)**（`ss_static/`）和 **滚动时域（全窗口两阶段 SAA）**
（`rolling_horizon/`）；`ss_timevarying/` 仍是骨架。按本轮约定，每个方法
**各自维护** simulator / policy / evaluation，不共享 `common/` 的仿真器。

## 目录

```
config/compare.toml         T60 共享配置（模型、基线、评估、训练/验证路径）
config/compare_T240.toml    T240 共享配置
common/
  config.py                 读 TOML；COMPARE_CONFIG 环境变量
  model.py                  只读导入主模型；problem_data.json；provenance
  baseline.py               基线 npz 加载；配对统计；五项成本统计
  report.py                 结果目录/配置副本/provenance/文件工具
ss_static/                  ★ 方法 1：时间无关 (s,S)，自包含
  evaluation.py             评价函数 θ -> 成本/明细
  simulator.py              纯 NumPy 状态/成本/审计（从 common 迁入）
  policy.py                 PolicyBase + SSStaticPolicy
  active.py                 可插拔主动调拨规则（mode 1，目标5 + 每仓h优化）
  metrics.py                行为统计、时间剖面、期末值敏感性
  starts.py                 无限库存基线、解析初值
  optimize.py               8 种优化算法 + 参数空间 + 缓存
  calibrate.py              标定、算法调度、归档
  evaluate.py               测试集评估、summary.md
  archive.py                时间戳目录、run.json、index.csv
  config/policy.toml        主基准 no-active 配置 + 算法预算
  config/policy_active.toml active 变体（目标5、每仓h优化）
  test/                     方法自包含测试（含 _t_active.py）
rolling_horizon/            ★ 方法 2：滚动时域（全窗口两阶段 SAA），自包含
  scenarios.py              固定优化池 N×T；基线真实路径子集
  lp.py                     COPT loadMatrix 稀疏两阶段 LP/追索 LP
  policy.py                 order/fulfill/active_transfer；单一 q；workers 并行
  simulator.py              纯 NumPy 闭环仿真/成本/审计
  metrics.py                行为/时间剖面/期末值敏感性
  evaluate.py               样本外评估、配对、审计、LP 统计、归档
  run_evaluate.py
  config/policy.toml
  test/                     池/LP/rollout/评估测试
docs/s,S.md                 用户提供的权威 (s,S) 模型口径
docs/滚动时域.md            用户提供的权威滚动时域模型/协议
docs/method-ss-static.md    ★ (s,S) 评价函数数学、算法、验证协议、归档
docs/method-rolling-horizon.md ★ 滚动时域实现、口径勘误、验证
test/_t_common.py           共享模型/配置自检
results/                    标定/评估归档（只新增）
```

## 快速开始

```bash
cd "D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-对比实验"
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"

# 全部测试
"$P" -B test/_t_common.py
"$P" -B ss_static/test/_t_policy.py
"$P" -B ss_static/test/_t_active.py
"$P" -B ss_static/test/_t_evaluation.py
"$P" -B ss_static/test/_t_optimize.py
"$P" -B ss_static/test/_t_archive.py
"$P" -B ss_static/test/_t_audit.py
"$P" -B ss_static/test/_t_lp_equiv.py
"$P" -B ss_static/test/_t_pathgen.py
"$P" -B ss_static/test/_t_baseline.py
"$P" -B ss_static/test/_t_calibrate.py

# T60 主验证
"$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T60
"$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T60_calib>/policy.json --tag ss_static_T60

# T240 主验证
COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T240
COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T240_calib>/policy.json --tag ss_static_T240

# T60 多算法对比（500/500 路径；各算法约同一真实评估预算）
"$P" -B ss_static/run_calibrate.py --algo all \
  --train-paths 500 --val-paths 500 --budget-unique 1500 \
  --tag ss_static_T60_algo_compare

# 滚动时域测试 / smoke / 正式
"$P" -B rolling_horizon/test/_t_pool.py
"$P" -B rolling_horizon/test/_t_lp.py
"$P" -B rolling_horizon/test/_t_rollout.py
"$P" -B rolling_horizon/test/_t_evaluate.py
"$P" -B rolling_horizon/run_evaluate.py --quick
"$P" -B rolling_horizon/run_evaluate.py \
  --n-scenarios 10 --real-paths 50 \
  --tag rolling_T60_204105_final --workers 4 --threads 4
```

`--algo` 支持 `coordinate,random,nelder-mead,powell,differential-evolution,
dual-annealing,genetic,simulated-annealing` 或 `all`。其他常用参数：
`--quick`（60/60 路径 smoke）、`--integer`、`--archive-root`、`--seed`、
`--train-paths`、`--val-paths`。

## 当前主结果（2026-10-26）

| 场景 | 标定 run | 评估 run | 策略均值 | 基线均值 | 配对差 | 相对差 | 胜率 | 审计 |
|---|---|---|---:|---:|---:|---:|---:|---|
| T60 | `ss_static_T60_calib_20261026_181316` | `ss_static_T60_eval_20261026_181640` | 1171.5302 | 825.4215 | +346.1087 | +41.93% | 0.60% | 通过 |
| T240 | `ss_static_T240_calib_20261026_181533` | `ss_static_T240_eval_20261026_182140` | 188154.7413 | 159380.8723 | +28773.8690 | +18.05% | 5.60% | 通过 |

- 均值单位万元/路径；同一基线 `scene` 逐路径配对；明细见各 run 的 `summary.md`。
- T60：差距集中在首/末段（首 10% 期 +165.2，末 10% 期 +232.7，中间段 −51.8），
  期末值调整后相对差 +21.97% 到 +41.93%；不能写成“稳态全程差 42%”。
- T240：差距主要是无主动调拨/空间错配（停机 +21783.8，主动调拨 −2237.4）；
  静态平均库存 36.41 vs 基线 14.16（Σ仓），仍缺货。
- 基线自身 gap：T60 13.63%，T240 29.54%；LB 不是基线策略成本。
- T60 算法对比（500/500，`budget-unique=1500`）：NM 最优（验证 1138.93），
  测试 1161.9423（+40.77%），比坐标下降主结果低 9.5879（配对 t=-19.53）。
  全量 NM 确认（2000/2000）：测试 1162.8456（+40.88%），比坐标下降低 8.6845
  （配对 t=-16.13）。搜索算法会影响 (s,S) 的表现，但最好仍比 SDDP 差约 40.8%。
- 跨运行总结（含新基线与 active）：`results/ss_static/SUMMARY_20261027_164929.md`。

### 新 T60 基线 `base_T60_20261026_204105`

该基线的 `problem_data.json` 只有 `n0` 不同（混合初始健康分布，约 42%/37%/16%/5%
在等级 0/1/2/3），其余 `N`/`Qmax`/`Imax`/成本/300 场景池/Θ 哈希与旧 T60 相同；
旧 policy 不能复用，需重新标定。配置见 `config/compare_T60_base204105.toml`，
测试集用最新的 3000 路径 sim。

| 方法 | 策略均值 | 基线均值 | 配对差 | 相对差 | 胜率 | 审计 |
|---|---:|---:|---:|---:|---:|---|
| s,S（s=S=Imax，边界解） | 41394.9455 | 35833.3228 | +5561.6227 | +15.52% | 1.37% | 通过 |

- 基线 LB=31573.2123，gap=11.889%；目录里较早的 2000 路径 sim 是
  mean=36663.8448、gap=14.678%，本对比用最新的 3000 路径 sim。
- 标定 `results/ss_static/ss_static_T60_204105_calib_20261027_130616`；
  评估 `results/ss_static/ss_static_T60_204105_eval_20261027_131049`。
- 结论文件：评估目录下的 `conclusion.md`（标定目录也有同名文件），
  内含主结果、成本/行为、边界解诊断和参数评估。
- 高失效率 + 混合初始健康把静态最优推到 Imax：`s=S=Imax` 是“尽量满仓”策略；
  服务率更高（0.953 vs 0.904）、期末待修更低（26.23 vs 51.19），但采购
  +3025.37、被动调拨 +2349.91，仍显著差于 SDDP。
- 这是比旧 T60（+41.9%）更有利于静态策略的实例，但“静态 (s,S) 不占优”的结论不变。
- 边界解诊断（`..._eval.../boundary_diagnostic.md`）：单仓 S/s 曲线都在 Imax 取最小；
  角点实现库存仅 22.35（Σ仓，ΣImax=63），Qmax/提前期是实际补货瓶颈。
  松弛 Imax×1.5 约 -10.2%、Qmax×2 约 -6.3%（仅诊断，非原模型可行）；
  Imax 对新混合 n0 偏紧，但不是 +15.5% 差距的全部原因。

### 主动调拨（mode 1，每仓门限 h 优化）

`ss_static/active.py` 的 `ThresholdBalanceRule`：目标库存 T=5 固定，
**每仓门限 h 是优化参数**，与 s,S 一起搜索（active 3K 维，no-active 2K）；
被动履约后按门限/缺口贪心调拨，满足容量/来源/守恒约束，末期最后 2 期不调。
配置：`ss_static/config/policy_active.toml`。

| 方法 | 策略均值 | 基线均值 | 配对差 | 相对差 | 胜率 | 审计 |
|---|---:|---:|---:|---:|---:|---|
| no-active | 41394.9455 | 35833.3228 | +5561.6227 | +15.52% | 1.37% | 通过 |
| active（h 优化） | **40723.1142** | 35833.3228 | **+4889.7914** | **+13.65%** | 1.67% | 通过 |

- active 标定：`results/ss_static/ss_static_T60_204105_active_calib_20261027_163659`
  - s=S=[12,8,20,15,8]=Imax；h=[9.5132, 5.0, 10.7237, 5.0, 5.0]；
  - train 40751.0852，val 40808.2530，n_unique=1428，墙钟约 419 s。
- active 评估：`results/ss_static/ss_static_T60_204105_active_eval_20261027_164616`
  - 40723.1142 vs 35833.3228；配对差 +4889.7914（+13.65%），
    t=98.83，胜率 1.67%，审计通过。
- **active vs no-active（同一 3000 测试路径）**：
  **-671.8313 万元/路径（-1.623%）**，t=-55.35，胜率 89.03%。
  主动调拨显著改善 s,S，但仍是规则型/state-only，仍比 SDDP 基线差 +13.65%。
- 五项差（active−基线）：采购 +3130.36、持有 -39.41、停机 +94.21、
  主动 -259.42、被动 +1964.05；active 主动调拨量 54.48 vs 基线 106.37。
- no-active 与 active 都正式报告，不预设主次；结论文件在各自结果目录的
  `conclusion.md`。

### 滚动时域（全窗口两阶段 SAA）

`rolling_horizon/` 按 `docs/滚动时域.md` 实现：每期用当前状态 + 固定优化池
\(N\times T\) 切片解两阶段随机 LP，只执行 \(q_t\)；真实 \(\Theta_t\) 观测后
解追索 LP 得 \(y,\tau\)，逐期递推。窗口 \(W=T-t+1\)，\(V_{T+1}=0\)。
优化池 N 条路径、seed_opt 固定；真实路径取基线 scene 前 M 条，与 SDDP 配对；
优化 LP 目标只用于产生 q，绩效用样本外均值。

- 状态：`I,n,o1..o(L-1)`；初始 `I0/o0/n0` 取 `problem_data`，没有额外虚拟 q0。
- 每期单一 \(q_t\)，采购按当期 \(Cq_t\)；无 q_in/q_out。主 SDDP 按到货
  \(q^{in}\) 计费，两者采购成本时点不同，逐期剖面对比须保留说明。
- 容量：每期 \(I_t+\Sigma o_t+q_t\le Imax\)（`cap_init`/`cap`）；转移后
  \(I_{t+1}+\Sigma o_{t+1}\le Imax\)（`cap_next`）；追索另有 `cap_end`。
- 允许 \(t=T\) 的两阶段 LP；无终端价值，\(q_T=0\)；主动调拨由 LP 优化。
- `workers` 是路径级线程并行，`threads` 是 COPT 单模型线程；
  `progress.log`/`run.log` 记录进度；失败 run 也保留 `run.json`/`error.txt`。
- 真实路径取基线 `scene` 前 M 条，与 SDDP 配对，不是独立 `seed_real`。
- **正式运行进行中**：T60 新基线、N=10、M=50、seed_opt=2027、COPT，
  tag `rolling_T60_204105_final`；尚无有效正式结果，旧 `capfix`/测试/smoke
  运行不作为论文结果。结果目录
  `results/rolling_horizon/<tag>_<timestamp>/`，数字以
  `summary.md`/`conclusion.md`/`comparison.json` 为准。详细口径见
  `docs/method-rolling-horizon.md`。

## 评价函数与算法

- 评价函数在 `ss_static/evaluation.py`，纯函数入口：
  `evaluate_cost(prob, options, theta, paths)`；
  no-active 的 `theta=[s_1..s_K,S_1..S_K]`（2K），
  active 的 `theta=[s,S,h_1..h_K]`（3K）；
  返回该路径集合上的平均总成本。`SSEvaluator.evaluate(..., record=True)`
  额外返回 `path_costs`、`term_totals`、`step_terms`、行为和记录数组。
- 参数空间 \(0\le s\le S\le I^{\max}\)。坐标下降/随机搜索在物理空间；
  NM/Powell/DE/SA/GA 用 sigmoid 变换到无约束 \(u\) 空间，解码一定可行。
- 算法预算与默认参数在 `ss_static/config/policy.toml`；每个算法独立缓存、
  独立随机种子，报告 `n_calls/n_unique/wall_s`。
- 详细公式、伪代码、验证协议、归档格式见 **`docs/method-ss-static.md`**。

## 模型与基线口径

- 模型参数/场景池优先从基线结果里的 `problem_data.json` 读取；
  T60/T240 的 300 场景池逐位相同，只有 T 不同。
- T60 基线：`results/base_T60_20261025_131043`，`sim_2000paths.npz`（2000×60），
  LB 712.9522、仿真均值 825.4215、gap 13.6257%。
- T240 基线：`results/base_20261024_153043`，`sim_1000paths.npz`（1000×240），
  LB 112,307.44、仿真均值 159,380.87、gap 29.535%。
- 测试路径直接取基线 npz 的 `scene`，逐路径配对；不重新生成。
- 训练/验证路径 seed 777/888，同一 300 场景池；测试不参与搜索/选择。
- 成本单位是**万元**；库存/待修是台/件。
- \(q\le Q^{\max}\)、无 stop offset、\(q_T\) 不计采购费、mode 0 无主动调拨，
  都是 `docs/s,S.md` + 当前模型的明确口径；不要只在策略层偷偷改。
- 静态规则是 state-only；SDDP 的 \(q^{out}\) 是阶段 LP 决策，使用更丰富信息。
  这是信息集/策略类差异，报告必须写明。
- 基线 LB 是模型下界，不是基线策略成本；不要把“策略成本 − LB”当作策略 gap。

## 归档与总结

每次标定/评估新建时间戳目录，禁止覆盖：

```
results/ss_static/index.csv                         # append-only 跨运行索引
results/ss_static/<tag>_calib_<timestamp>/          # 标定 + 算法对比
results/ss_static/<tag>_eval_<timestamp>/           # 测试评估 + summary.md
```

标定目录含 `policy.json`、`calibration.md`、`algorithm_summary.csv`、
`algorithm_starts.csv`、`optimization_history.csv`、`paths_train/val.npz`、
`provenance.json`、`config/`。

评估目录含 `sim_<P>paths.npz/.json`、`sim_report_*.txt`、`comparison.json/md`、
`summary.md`（对比验证总结）、`time_profile.csv`、`time_profile_deciles.csv`、
`terminal_sensitivity.json`、`provenance.json`、`config/`、`calibration/`。

`summary.md` 的事件顺序：运行信息 → 验证协议 → 主结果/配对统计 → 五项成本 →
行为诊断（Σ仓）→ 时间剖面 → 期末值敏感性 → 约束审计 → 结论与限制。
`calibration.md` 记录解析初值、算法/起点表、选择规则和选中策略。

## 只读与复现规则

- 不修改主项目的 `code/`、`utils/`、`results/`；只读导入，所有命令加 `-B`。
- 不重新采样 300 场景池；不重新生成测试路径；不用测试调参。
- 不覆盖 `results/` 目录；每次运行新目录；旧结果已按用户要求清空，新结果重新开始。
- 模型参数变了必须重训 SDDP 基线；旧 cuts 不兼容。
