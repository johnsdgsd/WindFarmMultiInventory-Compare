# CLAUDE.md（对比实验项目交接）

> 本轮按用户要求重构：每个验证方法**自包含**评价函数、仿真器、策略、优化器、
> 归档和报告；`common/` 只保留模型加载、配置、基线读取、通用配对统计和文件工具。
> 旧 `results/ss_static/` 已按用户要求清空；不再沿用旧结论。
>
> **2026-10-29 更新：最新交接以第 0 节为准；后面的 s,S 细节保留作历史。**

## 0. 最新交接（2026-10-29）

### 0.1 当前分支与结果

- `main`：旧时序滚动 + s,S，已推送到
  `https://github.com/johnsdgsd/WindFarmMultiInventory-Compare.git`。
- `tau-before-q`：新时序（先主动调拨 τ，再订货 q）实验分支，当前主要工作分支。
- 关键正式结果（T60 新基线 `base_T60_20261026_204105`）：

| 方法 | 均值（万元/路径） | 基线口径 | 相对差 |
|---|---:|---|---:|
| SDDP 基线全量 | 35,833.3228 | 3000 路径 | — |
| s,S no-active（新时序） | 37,950.9220 | 全量基线 | +5.91% |
| s,S active（新时序，GA，h 源仓下限+调入目标） | 37,338.0637 | 全量基线 | +4.20% |
| 旧时序滚动 final2（历史） | 39,891.1219 | 同 50 子集 | +12.52% |
| 新时序 taufirst3（历史，前50） | 34,307.1189 | 同 50 子集 | -3.23% |
| **新时序 W20 随机50** | **32,052.1364** | **同 50 子集 32,556.3100** | **-1.55%** |
| **新时序 W20 随机100** | **34,426.5611** | **同 100 子集 35,108.0177** | **-1.94%** |
| **新时序 W20 M=3000 全量** | **34,769.9266** | **全量基线 35,833.3228** | **-2.97%** |
| PPO（无时间特征，旧） | 37,285.9472 | 3000 独立回合 | +4.05% |
| **PPO（含 t，10000ep）** | **36,645.8471** | 5000 独立回合 | **+2.27%** |

- W20/30/60 随机 50 对照：-1.55%/-1.59%/-1.58%，W≥20 饱和，W20 最快。
- taufirst3 用前 50 条（-3.23%），与随机 50 子集基线 32,556 不能混表；
  旧时序 final2 只作历史对照。
- M=100：`results/rolling_horizon/tau_W20_M100_rand_20261029_192726/`；
  配对差 -681.4566（-1.94%），t=-3.34，胜率 61%，审计通过。
  该 run 从 t=35 checkpoint 续跑，`wall_s` 与 `lp_stats.csv` 只覆盖后半段；
  成本/配对结果完整，但不能当完整 LP 统计。
- **M=3000 全量已完成（2026-10-30 07:41）**：34,769.9266 vs 35,833.3228，
  配对差 **-1,063.3962（-2.97%）**，95% CI [-1,183.9738,-942.8186]，
  t=-17.29，胜率 62%，审计通过。
  结果目录 `results/rolling_horizon/tau_W20_M3000_full_merged_20261030_074140/`。
  shard0/2/3/4 在 00:18 卡死后，07:04–07:41 按 checkpoint 串行续跑；
  正常分片墙钟约 3 h，实际总墙钟约 10.3 h。成本/配对结果完整，
  但长跑数值稳定性仍需改进。
- PPO 含时间特征报告：
  `results/ppo_taufirst/ppo_taufirst_tobs_10000ep_20261029_132639/PPO_结果报告_时间特征_10000ep.md`。

### 0.2 要做的

1. **滚动时域以新时序分支为准**：`tau-before-q` 的
   `s_t -> tau_t -> q_t -> Theta_t -> y_t`；状态 `I,n,o1,o2`；τ/q 第一阶段，
   y 追索；`I_{t+1}=I_t+tau_t-Σy+o1`，`I_out=S_end`。
2. **滚动评估**：全量 M=3000 已完成为主结果；随机 50/100 子集只作稳健性，
   不再把旧时序 `final2` 或前 50 子集 `taufirst3` 当最终结果。
   全量结果与 s,S/active 同为 3000 路径逐路径配对。
3. **PPO 模块**：`ppo_taufirst/` 自包含 PyTorch PPO；观测含
   `t/(T-1)`（36 维），训练 10,000 回合，评估 5,000 独立回合；
   PPO 主评估**不用 50 子集**。
4. **结果目录规则**：一个 run 一个时间戳目录，训练文件与
   `simulation/`/评估文件放在同一目录内；不要拆成两个结果目录。
5. **图像规则**：只画原始未缩放的成本曲线；浅色为逐回合原始值，
   深色为平滑趋势（默认 MA(10)，可 MA(20)）；不画负奖励/缩放奖励图。
6. **数值健壮性**：COPT status=5 时自动 HiGHS（highs/highs-ipm/highs-ds）；
   记录 fallback_calls；若两套求解器都失败，记录 path/t，不改模型约束。
7. **文档与 git**：新时序结果、PPO 报告、限制写入对应结果目录和 docs；
   新代码在 `tau-before-q` 分支本地 commit；push 前确认 remote/branch。
8. **剩余分析**：PPO 多 seed / warm-start 仅作为敏感性；不要用测试集调参。

### 0.3 不要做的

- 不修改主项目 `code/`、`utils/`、`results/`；只读导入。
- 不重新采样 300 场景池；不重新生成测试路径；不用测试路径调参。
- 不覆盖已有 `results/`；失败/中间结果保留；一个 run 的训练和仿真放同一目录。
- 不在策略层偷偷去掉 Qmax、加 stop offset 或隐藏的安全界；要改就改模型并重跑。
- 不把 `tau-before-q` 的结果和旧时序 `main` 结果混在同一张表里当作同一策略。
- 不把 PPO/rolling/SDDP 当成同一信息集：PPO 是 state-only（含 t），
  rolling 是两阶段 SAA，SDDP 使用 cuts/阶段信息。
- PPO 的被动调拨是固定优先级规则；rolling/SDDP 的 y 是 LP 优化，不能混为一谈。
- 不把缩放奖励当作绩效，也不画负奖励图；绩效用实际成本（万元/路径）。
- 不对 PPO 使用 50 条基线子集作为主评估；PPO 用独立 5000 回合。
- 不擅自 push 到 remote；尤其是 `main` 和只读主项目。

### 0.4 目前收获

- **新时序 τ→q 显著优于旧时序 q→τ**：
  旧时序 +12.52%（同 50 子集），新时序随机 50 约 -1.55%；
  主动调拨量从 0 变为约 120，停机成本下降。
- **全量 M=3000 滚动低于 SDDP 基线**：34,769.9266 vs 35,833.3228，
  **-2.97%**（t=-17.29，胜率 62%，审计通过）。这是当前最强对照结果；
  N=10/W=20，且基线自身 gap=11.889%，不能写成“优于最优”。
- **PPO 加入时间特征后改善**：36,645.85 vs 旧版 37,285.95；
  但仍比基线高 +2.27%，95% CI [+320.14,+1,304.91]，尚未追平。
- **PPO 仍高于新时序 active s,S**（37,338），但 5000 独立回合不是配对，
  不能直接和 rolling 的 3000 路径配对差比较。
- **数值问题可处理**：COPT status=5 用 Presolve=0 重试（秒级），
  不再依赖 HiGHS 的 ~300 s IPM fallback；M=100/续跑/测试均通过。
- **长跑稳定性**：6 分片中有 4 片在 00:18 后无进度但持续占 CPU；
  杀掉后按 checkpoint 串行续跑完成。正常分片约 3 h，实际总墙钟约 10.3 h。
- **结果目录/图像规范已明确**：训练+仿真同目录；成本曲线只画 raw + MA。

### 0.5 经验教训

1. **决策时序会改变可行域**：τ 放在 q 前，必须先满足
   `I+τ∈[0,Imax]`、τ 来源和 `(I+τ)+Σo+q≤Imax`；q 后的追索容量仍然保留。
2. **第一阶段容量约束不能少**：缺少期初/每期容量会导致 q_t 在真实 Θ 下
   使追索不可行；`cap_init`、未来期 `cap`、`cap_next`、`cap_end` 各司其职。
3. **数值容差不能过严**：`-1e-8` 的 I/y 负残差是 COPT FeasTol，不是模型错；
   统一用 `1e-6` 并只对残差截零，不改变约束。
4. **COPT status=5 优先用 Presolve=0 重试**：退化两阶段 LP 在默认
   presolve 下偶发数值失败；同一矩阵 `Presolve=0` 通常 <1 s 解出，
   目标值与正常 presolve 一致（实测 24 个 LP 最大相对差 5e-12）。
   HiGHS `highs-ipm` fallback 实测约 300 s，不能作为第一选择；
   `lp.py` 已自动 auto→Presolve=0 重试。不要因 status=5 改模型约束。
5. **PPO 必须有时间特征**：state-only、无 t 无法表达末期；加入 `t/(T-1)`
   后均值改善约 1.7%，但仍然是有限信息集。
6. **平滑只用于可视化**：EMA α=0.2 有效记忆约 5；现在主图用 MA(10)，
   MA20 更平滑；平滑不参与训练。
7. **奖励=成本的负数**：报告只画原始成本曲线；`reward_scale=0.001` 只用于
   PPO 梯度，不影响报告成本。
8. **PPO 的固定优先级 y 与 LP y 不同**：比较时必须单列策略类差异；
   PPO 5000 独立回合 vs rolling 50 子集不是同一配对协议。
9. **50 路径与 3000 路径要分开写**：滚动用同 50 路径配对；PPO 用 5000
   独立回合；跨方法比较只能作方向性参考。
10. **结果目录不要拆**：训练、曲线、`simulation/summary.md` 放同一 run 目录，
    否则交接和复现会混乱。
11. **滚动公平比较必须同路径配对**：随机 M 条子集的基线均值不等于全量
    均值（随机 50 条基线 32556 vs 全量 35833）；报告要写“同路径配对差”，
    全量只作换算参考。W=20/30/60 在随机 50 条上分别为 -1.55%/-1.59%/
    -1.58%，W≥20 饱和，W20 最快；W10 数值失败已停。
12. **Windows 断点续跑**：`np.load(checkpoint.npz)` 的 NpzFile 必须显式
    `close()`，否则第一次 `os.replace` 覆盖 checkpoint 会报 WinError 5。
13. **lp_stats 要增量归档**：重启后内存里的 LP 统计会丢；每个 checkpoint
    追加写 `lp_stats.csv`，resume 时载入，否则报告只剩后半段。
14. **M=3000 全量已完成**：rolling -2.97%（t=-17.29，胜率 62%，审计通过），
    可与 s,S/active 同表；子集结果只作稳健性。仍不能写成“策略优于最优”，
    因为 N=10/W=20 且基线自身 gap=11.889%。
15. **长跑会卡死，不等于模型失败**：4 个分片在 00:18 后停在 t=49/51，
    无 FAILED、无 fallback、CPU 持续占用；杀掉后按 checkpoint 串行续跑
    （`--time-limit 120`）全部正常完成。多进程 COPT 长跑需监控进度，
    不要只进程存活就认为在推进。
16. **合并要容忍续跑 npz 的额外键**：resume 的 sim npz 可能多出
    I/o/n/term_totals；`merge_shards.py` 用公共键交集并校验必需字段，
    不要假设所有分片 npz 键完全一致。
15. **单进程 GIL 限制**：即使 24 workers/1 thread，单 Python 进程的 LP
    构建仍受 GIL；M=3000 约 27–33 h。实测 6 个独立进程×2 workers×2 threads
    聚合 ~13.5 LP/s，估算 3.7 h。分片只是并行实现，策略不变。
16. **分片合并必须校验路径覆盖**：`merge_shards.py` 要求 6 片恰好覆盖
    0..2999 且按原顺序拼接；合并后重新做逐路径配对/审计。smoke 已验证
    4 片与 6 片在相同 N/W 下得出一致的 3000 路径均值。



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
"$P" -B ss_static/test/_t_timing.py
"$P" -B ss_static/test/_t_pathgen.py
"$P" -B ss_static/test/_t_baseline.py
"$P" -B ss_static/test/_t_calibrate.py
"$P" -B rolling_horizon/test/_t_pool.py
"$P" -B rolling_horizon/test/_t_lp.py
"$P" -B rolling_horizon/test/_t_rollout.py
"$P" -B rolling_horizon/test/_t_evaluate.py
```

`_t_timing.py` 检查 s,S 新时序（无 q_in/q0、q_t 进入在途、容量/成本/审计）；
旧 `_t_lp_equiv.py` 因 s,S 与主模型 q_in 口径不再一致而退役；
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
- 仿真口径：期初状态 → (s,S) 订货 q_t → 本期 Θ/失效 → 被动调拨 →
  在途滚动/到货 → q_t 进入在途 → 主动调拨 → 成本 → 转移；
  **没有虚拟 q0/q_in**；
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
- s,S 五项成本与自身状态/时序一致：`_t_timing.py` 通过；
  旧主 LP 对拍 `_t_lp_equiv.py` 因 q_in 口径变更退役。
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
- 新基线 204105（混合 n0，**s,S 新时序：无 q_in/q0**）：
  no-active **+5.91%**，active **+4.20%**；
  active vs no-active **-1.61%**（t=-39.16，胜率 76.50%）；
  两者都正式报告，不预设主次。旧 q_in 的 no-active +15.52%、active +13.36%/
  +13.65% 只作历史对照。
- 基线自身有 gap（旧 T60 13.63%，T240 29.54%，新 T60 11.889%）；
  LB 不是基线策略成本。
- 跨运行总结：`results/ss_static/SUMMARY_20261027_164929.md`（含 active）。

### 8.5 新 T60 基线 `base_T60_20261026_204105`（混合初始健康分布）

> 以下 41,394.9455/+15.52% 等具体 s,S 数值是旧 q_in 口径；
> 新时序 no-active 为 37,950.9220/+5.91%，见 §0.1/§8.7。

配置文件：`config/compare_T60_base204105.toml`。

- 与旧 T60 的区别：**只有 `n0` 变成混合初始健康分布**（约 42%/37%/16%/5% 在等级
  0/1/2/3），`N`、`Qmax`、`Imax`、成本系数、300 场景池和 Θ 哈希都与旧 T60 相同。
  因此旧 policy 的 `problem_data_hash` 不匹配，必须重新标定。
- 基线测试集用最新的 3000 路径 sim（`sim_3000paths.npz`）：
  LB=31573.2123，mean=35833.3228，gap=11.889%。
  目录里还有较早的 2000 路径 sim（mean=36663.8448，gap=14.678%），本对比不用它。
- **新时序标定/评估（2026-10-30，无 q_in/q0）**：
  `..._noactive_ga_noqin_calib_20261030_110033` /
  `..._noactive_ga_noqin_eval_20261030_110621`；
  GA，s=S=Imax，train 37939.0513，val 38052.8820，n_unique=1420；
  测试 **37950.9220** vs 35833.3228，配对差 **+2117.5992（+5.91%）**，
  t=36.23，胜率 10.17%，审计通过。
- 旧 q_in 口径（历史）：coordinate，s=S=Imax，测试 41394.9455
  （+15.52%）；只作历史对照。
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
  `_t_common`/`_t_audit`/`_t_pathgen`/`_t_timing` 在新配置下通过；
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
  结构差异仍在；旧 q_in 口径的 +15.52% 不能全归因于 Imax。

### 8.7 主动调拨（mode 1，h 为源仓保留下限 + GA）

- 机制：`ss_static/active.py::ThresholdBalanceRule`，可插拔；每仓
  **h 同时是源仓保留下限和调入目标**（`X>h` 时最多调出 `X-h`，
  `X<h` 时补到 `h`）；没有单独 T；与 s,S 一起搜索（active 3K，no-active 2K）；
  `active_threshold_min=0`，`active_stop_offset=2`；下一期失效用 300 场景池平均预测。
- 配置：`ss_static/config/policy_active.toml`；no-active 与 active 都正式报告；
  **新标定统一用 genetic**（population=40/generations=40，约 1540 次唯一评估）。
- 新 active 标定（**s,S 新时序，无 q_in/q0；h 同时为源仓下限和调入目标**）：
  `..._active_ga_hboth_calib_20261030_112535`
  - genetic；s=[12.00,7.84,20.00,14.99,7.98]，
    S=[12,8,20,14.99,8.00]；
  - h=[6.2784, 4.3547, 11.0788, 6.0950, 2.8185]；
  - train 37314.5841，val 37461.2067，n_unique=1544，墙钟 480.2 s。
- 新 active 评估：`..._active_ga_hboth_eval_20261030_113357`
  - **37338.0637 vs 基线 35833.3228**；配对差 **+1504.7409（+4.20%）**；
    95% CI [+1380.1000, +1629.3818]；t=23.66；胜率 13.97%；审计通过。
- **active vs no-active（同一 3000 路径，二者都用 GA）**：
  **-612.8583 万元/路径（-1.6149%）**；95% CI [-643.5315,-582.1852]；
  t=-39.16；胜率 76.50%。
- 五项差（active−基线）：采购 +3499.38，持有 +328.24，停机 -3871.27，
  主动 -107.45，被动 +1655.85；相对 no-active，停机 -856.46、被动 -519.67，
  但采购 +202.10、持有 +136.79、主动 +424.39，净 -612.86。
- 行为：平均库存 29.37（active）；替换率 0.9756；期末待修 14.00；
  主动调拨量 84.88。
- 时间：首 10% 期 +553.90，中间段 -727.92，末 10% 期 +1678.76；
  期末值敏感性 -3.87% 到 +4.20%。
- 旧 q_in active（+13.36%/+13.65%）只作历史对照，不作主结果。
- 结论：主动调拨 + h 双目标 + GA 显著改善 s,S（−1.61%），但仍是
  规则型/state-only，**仍比 SDDP 基线差 +4.20%**；不能等同于完整价值函数。
- 跨运行总结：`results/ss_static/SUMMARY_20261027_164929.md`（旧口径）。

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
- **正式结果**（2026-10-28，tag `rolling_T60_204105_final2`）：
  N=10/M=50、seed_opt=2027、workers=6/threads=4；
  滚动 39891.1219 vs 同 50 路径基线 35451.4783，配对差 +4439.6437
  （+12.52%），t=8.40，胜率 8.00%，审计通过；相对全量基线 35833.3228
  为 +11.32%，低于当前 active s,S 的 +13.36%。旧 `capfix`/`simpletest`/`quick`
  结果不能作为最终口径；数字以归档目录
  `results/rolling_horizon/rolling_T60_204105_final2_20261028_223241/` 为准。
- 测试：`rolling_horizon/test/_t_pool.py`、`_t_lp.py`（L-1 状态/两层容量/目标复算/
  t=T q=0/追索 cap_end）、`_t_rollout.py`、`_t_evaluate.py`；现有
  `_t_timing.py` 通过（s,S 无 q_in 的新时序）。
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

## 10. 下一步（以第 0 节为准）

1. 在 `tau-before-q` 分支整理并提交新时序滚动结果、PPO 时间特征报告和成本曲线；
   决定是否把该分支 push 到 GitHub。
2. 若继续优化，优先做 PPO 多 seed / warm-start 敏感性，以及 N=20/50 的滚动扩展；
   不要用 3000 测试路径调参。
3. 论文主表使用：
   - s,S no-active / active（全量 3000 路径）；
   - 旧时序滚动 final2（说明已被新时序取代）；
   - 新时序滚动 taufirst3（主结果，同 50 路径配对 + 全量基线换算）；
   - PPO（5000 独立回合，含 t；不与 rolling 做假配对）。
4. 统一写清采购成本时点（滚动按当期 Cq，SDDP 按 q_in）、q0/qT、信息集、
   PPO 固定优先级 y 与 LP y 的差异。
5. 结果目录保持“一个 run 一个目录，训练+仿真同目录”的规则；图像只画原始成本 +
   平滑趋势。
