# CLAUDE.md — 风场多仓库备件库存对比实验交接

> 交接日期：2026-10-30  
> 适用项目：`D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-对比实验`
> 主项目（只读）：`D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-python`

## 0. 先看这里（TL;DR）

### 0.1 当前 git 状态

- 分支：`window-fair-resume`
- remote：`https://github.com/johnsdgsd/WindFarmMultiInventory-Compare.git`
- 已推送的最新提交：
  - `4adf71a` — 文档更新到最新 s,S/active/M3000 结果
  - `d321ebc` — active 的 h 改为“源仓下限 + 调入目标”，删除 T
  - `92889ad` — s,S 去掉 q_in/虚拟 q0
- 工作区只多一个未跟踪的二进制文件：`对比试验汇报.docx`。代码与 md 文档均已提交。
- **不要擅自 push**；push 前确认 remote/branch。

### 0.2 当前正式结果（全部 T60 新基线 `base_T60_20261026_204105`，3000 测试路径）

| 方法 | 均值（万元/路径） | 基线口径 | 配对差 | 相对差 | t | 胜率 | 审计 |
|---|---:|---|---:|---:|---:|---:|---|
| SDDP 基线 | **35,833.3228** | 3000 | — | — | — | — | LB=31,573.2123，gap=11.889% |
| s,S no-active（新时序，GA） | 37,950.9220 | 35,833.3228 | +2,117.5992 | +5.91% | 36.23 | 10.17% | 通过 |
| s,S active（新时序，GA，h 双目标） | **37,338.0637** | 35,833.3228 | +1,504.7409 | +4.20% | 23.66 | 13.97% | 通过 |
| 滚动 W20 M=3000（τ→q，全量） | **34,769.9266** | 35,833.3228 | -1,063.3962 | **-2.97%** | -17.29 | 62.00% | 通过 |
| PPO 含 t，10000ep | 36,645.8471 | 35,833.3228 | +812.5243 | +2.27% | — | — | 5000 独立回合，非配对 |

- active vs no-active（同 3000 路径，二者 GA）：**-612.8583（-1.6149%）**，
  95% CI [-643.5315,-582.1852]，t=-39.16，胜率 76.50%。
- 当前最强结果：**滚动 M=3000 的 -2.97%**，全量 3000 路径配对，审计通过。
- 但 rolling 是 N=10/W=20 的缩减 SAA，基线自身 gap=11.889%，
  不能写成“策略优于最优”。

### 0.3 三条必须记住的口径

1. **s,S 没有 q_in，也没有虚拟 q0**。状态只有 \(I,o_1,o_2,n\)；
   \[
   IP=I+o_1+o_2-W,
   \]
   \(q_t\) 是当期决策，订单在被动履约/到货后进入下一期最后一层在途
   \(o_{2,t+1}=q_t\)。
2. **active 没有独立 T**。每仓 \(h_r\) 同时是源仓下限和调入目标：
   \(X_r>h_r\) 时最多调出 \(X_r-h_r\)；\(X_m<h_m\) 时补到 \(h_m\)。
3. **旧 q_in/h-target 结果只作历史**：
   - 旧 s,S no-active：41,394.9455（+15.52%）
   - 旧 active：40,723.1142（+13.65%）
   - 中间版 active（h 源仓下限 + T=5）：37,416.4504（+4.42%）
   这些都不能与当前结果混表。

---

## 1. 项目目标与范围

把 SDDP 基线与简单策略放在**同一模型、同一场景池、同一测试路径**上比较，
量化简单策略相对 SDDP 的差距，并给论文提供可复现、可归档结果。

当前实现：

- `ss_static/`：非时变 (s,S)，含 active 规则与 GA/多种优化器。
- `rolling_horizon/`：τ→q 新时序、两阶段滚动 SAA、分片/checkpoint/合并。
- `ppo_taufirst/`：自包含 PyTorch PPO（新时序）。
- `ss_timevarying/`：骨架，未实现。
- `common/`：只读模型加载、配置、基线、配对统计、归档工具。
- `docs/`：方法说明与用户提供的权威口径。
- `results/`：归档，只新增，不覆盖。

主项目 `code/`、`utils/`、`results/` **只读**。

## 2. 环境与常用命令

```bash
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
# 始终在项目根运行；加 -B，避免往主项目写 .pyc
```

T60 共享配置：

```bash
COMPARE_CONFIG=config/compare_T60_base204105.toml
```

### 2.1 测试

```bash
"$P" -B test/_t_common.py
"$P" -B ss_static/test/_t_policy.py
"$P" -B ss_static/test/_t_timing.py      # s,S 新时序，425 checks
"$P" -B ss_static/test/_t_audit.py
"$P" -B ss_static/test/_t_active.py      # 38 checks
"$P" -B ss_static/test/_t_optimize.py
"$P" -B ss_static/test/_t_evaluation.py
"$P" -B ss_static/test/_t_calibrate.py
"$P" -B ss_static/test/_t_archive.py
"$P" -B ss_static/test/_t_pathgen.py
"$P" -B ss_static/test/_t_baseline.py
"$P" -B rolling_horizon/test/_t_pool.py
"$P" -B rolling_horizon/test/_t_lp.py
"$P" -B rolling_horizon/test/_t_rollout.py
"$P" -B rolling_horizon/test/_t_evaluate.py
```

说明：

- `_t_lp_equiv.py` 已删除。它把 s,S 与主模型 q_in/q_out 两套变量逐项对拍；
  新 s,S 没有 q_in，该对拍不再成立。
- `_t_timing.py` 检查：无 q_in/q0、q_t 进入在途、当前/下一期容量、
  采购成本、审计。不要重新引入 q_in 去“修”这个测试。
- `_t_archive` 用临时目录，不污染 `results/`。

### 2.2 s,S 标定/评估（统一 GA）

no-active：

```bash
COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_calibrate.py \
  --algo genetic --train-paths 2000 --val-paths 2000 \
  --tag ss_static_T60_204105_noactive_ga_noqin

COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<calib>/policy.json \
  --tag ss_static_T60_204105_noactive_ga_noqin
```

active：

```bash
COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_calibrate.py \
  --policy-config ss_static/config/policy_active.toml \
  --algo genetic --train-paths 2000 --val-paths 2000 \
  --tag ss_static_T60_204105_active_ga_hboth

COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<calib>/policy.json \
  --policy-config ss_static/config/policy_active.toml \
  --tag ss_static_T60_204105_active_ga_hboth
```

GA 预算（`ss_static/config/*.toml`）：

- population=40，generations=40，elite=2，BLX-α=0.5，
  crossover_prob=0.8，mutation_prob=0.2，mutation_sigma=0.5；
- max_evals=4000 只是安全上限，40 代实际约 1560 次评估，不会触发；
- **每次 GA 适应度评估使用 2000 条训练路径 ×60 期**；
- 验证 2000 条只在候选终点重评；测试 3000 条不参与选择；
- 不要用测试集调参。

### 2.3 滚动时域 M=3000

单分片命令示例（正式全量由 wrapper 启动 6 片）：

```bash
"$P" -B -u rolling_horizon/run_evaluate.py \
  --n-scenarios 10 --real-paths 500 --seed-opt 2027 \
  --path-sample first --path-offset 0 --window-len 20 \
  --workers 2 --threads 2 --checkpoint-every 1 \
  --skip-index --tag tau_W20_M3000_shard0
```

全量分片 + 自动续跑 + 合并：

```bash
TOTAL=3000 SHARDS=6 N_SCENARIOS=10 WINDOW=20 \
WORKERS=2 THREADS=2 CHECKPOINT_EVERY=1 \
TAG_PREFIX=tau_W20_M3000_shard \
MERGED_TAG=tau_W20_M3000_full_merged \
TARGET="2026-10-30 00:00:00" \
bash rolling_horizon/run_m3000_shards.sh
```

- 六个分片路径为 0–499、500–999、…、2500–2999；
- `merge_shards.py` 校验 0..2999 恰好覆盖、N/W/pool_hash 一致；
- 结果目录：`results/rolling_horizon/tau_W20_M3000_full_merged_20261030_074140/`；
- 本次 4 片在 00:18 卡死，07:04–07:41 用
  `rolling_horizon/resume_m3000_stuck.sh` 串行续跑完成；
  正常分片约 3 h，实际总墙钟约 10.3 h。结果成本/配对完整。

### 2.4 PPO

```bash
"$P" -B ppo_taufirst/train.py --episodes 10000 --seed 2027 --threads 4 \
  --tag ppo_taufirst_tobs_10000ep
"$P" -B ppo_taufirst/simulate.py \
  --policy results/ppo_taufirst/<run>/policy.pt \
  --episodes 5000 --seed 888 --threads 4
```

- 观测 36 维：I/Imax 5 + n/N 20 + o1/Qmax 5 + o2/Qmax 5 + t/(T-1) 1；
- 动作 10 维：τ 5 + q 5；
- 被动 y 是固定优先级规则，不是 LP；
- reward = -成本 × reward_scale；报告只用实际成本；
- PPO 5000 独立回合不能与 rolling 3000 路径做假配对。

## 3. 模型/基线/测试口径

### 3.1 系统参数（T60 新基线）

来源：主项目
`results/base_T60_20261026_204105/problem_data.json` 与
`config/problem.toml`。

| 参数 | 值 |
|---|---|
| T | 60 |
| N_R | 5 |
| G | 4（等级 0–3；4=失效） |
| L | 3（当前 s,S 实际在途状态 2 层） |
| N | [3680, 2730, 3780, 3320, 2520] |
| Imax | [12, 8, 20, 15, 8] |
| Qmax | [3, 2, 3, 3, 2] |
| C / h / Cs / atc | 50 / 1 / 20 / 5 万元 |
| tc | 路线矩阵 0/10/20/30/40 |
| 初始 | I0=0，o0=0，n0 混合健康分布 |
| 场景池 | 300 条 (z,Θ)，seed=2026 |
| 测试路径 | 3000 条 `sim_3000paths.npz` |

基线：

- mean **35,833.3228**，std 11,417.7152，LB **31,573.2123**，gap **11.889%**；
- 训练 231 轮 / 7,503 s；仿真 401.9 s；
- **LB 不是基线策略成本**；所有相对差都是“策略对策略”。

### 3.2 训练/验证/测试协议

| 用途 | 路径 | 参与选择 |
|---|---|---|
| 训练 | seed 777，2000 条 | 是（GA fitness） |
| 验证 | seed 888，2000 条 | 是（选策略） |
| 测试 | 基线 npz 的 `scene`，3000 条 | **否** |

- s,S/active 的测试使用同 3000 条基线路径逐路径配对；
- rolling M=3000 也是同一组 3000 条路径；
- PPO 是 5000 独立回合，非配对；
- M=50/M=100 滚动子集只作稳健性，不能与全量直接相减。

## 4. 方法实现

### 4.1 SDDP 基线

- 完整状态 \((I,n,o,q^{in})\)、300 场景 cuts、阶段 LP；
- 作为参照，不是最优解；LB/gap 见上；
- 不要用 LB 替代基线仿真均值。

### 4.2 s,S（无主动调拨）

**状态**：

\[
x_t=(I_t,o_{1,t},o_{2,t},n_t),\qquad
W_t=N-\sum_i n_{i,t}.
\]

**订货**：

\[
IP_t=I_t+o_{1,t}+o_{2,t}-W_t,
\]

\[
q_t=\mathbf 1\{IP_t<s\}\max\{0,\min(S-IP_t,\ I^{max}-(I+o_1+o_2),\ Q^{max})\}.
\]

- q 越界直接截断，保证 \(I+o_1+o_2+q_t\le I^{max}\)、\(q_t\le Q^{max}\)；
- 时序：决定 \(q_t\) → 用 \(I_t\) 履约/被动调拨 → \(o_1\) 到货、在途滚动 →
  \(q_t\) 进入下一期 \(o_2\)；主动调拨（若有）在到货后的 \(S^{end}\) 上执行；
- 没有 q_in/q0；\(q_T\) 仍按旧 docs 口径**不计采购费**（如果用户要 q_T=0
  或计费，需要另外确认/修改）。

**实现文件**：

- `ss_static/simulator.py`：状态/时序/成本/审计；
- `ss_static/policy.py`：`SSStaticPolicy.order(t,I,o,n)`；
- `ss_static/evaluation.py`：评价函数；
- `ss_static/optimize.py`：GA 等；
- `ss_static/calibrate.py`：GA 标定/归档。

**当前结果**：

- no-active 37,950.9220（+5.91%，t=36.23，胜率 10.17%，审计通过）；
- 标定/评估目录见 §0/§6。

### 4.3 active s,S

规则 `ss_static/active.py::ThresholdBalanceRule`：

- \(X_r>S^{end}\) 为调拨前库存；
- 源仓：\(X_r>h_r\) 时最多调出 \(X_r-h_r\)；
- 调入仓：\(X_m<h_m\) 时补到 \(h_m\)；
- **没有独立 T**；h 同时承担两个角色；
- 贪心匹配最大盈余源/最大缺口目标；扣下一期在途容量、
  来源约束和累计预算；
- `alpha=1.0`，末期最后 2 期不调；
- 下一期失效用 300 场景池平均预测；
- 状态仍是 state-only 规则，不是完整值函数。

参数/结果：

- s=[12.00,7.84,20.00,14.99,7.98]；
- S=[12,8,20,14.99,8.00]；
- h=[6.2784,4.3547,11.0788,6.0950,2.8185]；
- train 37,314.5841；val 37,461.2067；n_unique=1544；墙钟 480.2 s；
- 测试 37,338.0637（+4.20%，t=23.66，胜率 13.97%，审计通过）；
- active vs no-active -612.8583（-1.6149%，t=-39.16，胜率 76.50%）。

五项差（active−基线）：采购 +3,499.38、持有 +328.24、
停机 -3,871.27、主动 -107.45、被动 +1,655.85。
相对 no-active：采购 +202.10、持有 +136.79、停机 -856.46、
主动 +424.39、被动 -519.67。

### 4.4 滚动时域（新时序 τ→q）

时序：

\[
s_t \to \tau_t \to q_t \to \Theta_t \to y_t \to s_{t+1}.
\]

- τ_t、q_t 第一阶段（无情景索引）；
- y_t 追索（真实 Θ 后）；
- 状态 `I,n,o1,o2`；
- \(S^{end}=I+\tau-\sum y+o_1\)，\(I_{t+1}=S^{end}\)；
- 每期容量 `I+τ+Σo+q≤Imax`；转移后 `I_{t+1}+Σo_{t+1}≤Imax`；
- 追索 cap_end；q_T=0；窗口末端 V=0；
- 采购按当期 Cq_t（SDDP 按 q_in，时点不同）。

正式全量：

- N=10，W=20，seed_opt=2027，M=3000，6 个 500 路径分片；
- 滚动 34,769.9266 vs 35,833.3228，差 -1,063.3962（-2.97%），
  t=-17.29，胜率 62%，审计通过；
- 五项差：采购 -392.91、持有 +0.92（不显著）、停机 -630.73、
  主动 +83.99、被动 -124.66；
- 时间：首 10% +245.08、中间段 -1,759.83、末 10% +451.35；
- 180,000 两阶段 LP + 180,000 追索 LP，0 失败；
- 正常分片墙钟约 10,922.8 s；实际总墙钟约 10.3 h（卡死/续跑）；
- 结果目录 `results/rolling_horizon/tau_W20_M3000_full_merged_20261030_074140/`.

历史滚动：

- 旧时序 final2：39891.1219（+12.52%，同 50 子集），只作历史；
- 前 50 子集 taufirst3：34,307.1189（-3.23%），只作历史；
- 随机 50/100 子集：-1.55%/-1.94%，只作稳健性。

### 4.5 PPO

- 自包含 `ppo_taufirst/`；观测含 `t/(T-1)`，共 36 维；
- actor/critic 2×128 MLP，约 43,925 参数；
- 10,000 episodes / 600,000 steps；5000 独立评估回合；
- 结果 36,645.8471，相对基线 +812.5243（+2.27%），
  95% CI [+320.14,+1,304.91]；
- 旧无时间特征 37,285.9472（+4.05%）只作历史；
- state-only、规则被动 y、单 seed；没有五项成本/逐路径审计归档；
- 不能与 rolling 做配对 t 检验。

## 5. 关键数值/工程经验

1. **COPT status=5**：退化两阶段 LP 的默认 presolve 偶发数值失败。
   同一矩阵用 `Presolve=0` 重试通常秒级解出；`lp.py` 已自动
   auto→Presolve=0。HiGHS `highs-ipm` 曾耗时约 300 s，不能作第一 fallback。
2. **断点续跑**：Windows 下 `np.load(checkpoint.npz)` 必须显式 `close()`，
   否则 `os.replace` 覆盖 checkpoint 报 WinError 5。
3. **lp_stats 增量归档**：每个 checkpoint 追加 `lp_stats.csv`，resume 时载入；
   否则重启后只剩后半段统计。
4. **长跑卡死**：4 个滚动分片在 00:18 后无进度但持续占 CPU，
   无 FAILED/无 fallback；停止后按 checkpoint 串行续跑成功。
   多进程 COPT 长跑必须监控 progress，不要只看进程存活。
5. **分片合并**：resume 的 sim npz 可能多出 I/o/n/term_totals；
   `merge_shards.py` 用公共键交集并校验必需字段；路径必须恰好覆盖 0..2999。
6. **公平比较**：随机 50/100 子集的基线均值不同于全量；
   报告必须写同路径配对差，不能把子集均值与全量直接相减。
7. **采购时点**：rolling/s,S 按当期 Cq；SDDP 按 q_in；
   逐期成本剖面不能直接对齐。
8. **s,S 的 L 层数**：当前显式在途 `o` 只有 2 层，q_t 在下一期进入 o₂，
   再下一期成为 o₁/到货。若用户要求“t 决策、t+3 到货”，需要 3 层在途，
   并重跑 s,S/active；这是当前最需要确认的建模口径。
9. **q_T**：当前按旧 docs 不计采购费，但 q_T 仍由规则产生；
   如果用户要求 q_T=0 或计费，必须改代码并重跑。
10. **GA 预算**：40 代、约 1560 次唯一评估；max_evals=4000 实际不约束。
   每次评估 2000 训练路径；验证只在候选终点。

## 6. 结果与文档索引

### 6.1 当前应优先阅读

- 简明当前报告：`实验结果汇报.md`
  - 系统参数、SDDP、s,S/active、rolling、PPO、统一对比；
  - 算法表只保留 genetic；active 无 T。
- 扩展报告：`对比实验结果总结_20261029.md`
  - §0.1/§0.1.2/§7/§10 是当前结果；
  - §3/§4 的旧 q_in/h-target 数值保留为历史，节首有提示；
  - 不要引用其中 41,394.9455、40,723.1142、37,416.4504 等旧值作当前结论。
- 方法文档：
  - `docs/s,S.md`（s,S 新时序/无 q_in/q0）
  - `docs/method-ss-static.md`
  - `docs/滚动时域.md`
  - `docs/method-rolling-horizon.md`
- 结果目录：
  - s,S no-active：`results/ss_static/ss_static_T60_204105_noactive_ga_noqin_calib_20261030_110033/`
    与 `..._noactive_ga_noqin_eval_20261030_110621/`
  - active：`results/ss_static/ss_static_T60_204105_active_ga_hboth_calib_20261030_112535/`
    与 `..._active_ga_hboth_eval_20261030_113357/`
  - rolling M3000：`results/rolling_horizon/tau_W20_M3000_full_merged_20261030_074140/`
  - PPO：`results/ppo_taufirst/ppo_taufirst_tobs_10000ep_20261029_132639/`

### 6.2 归档规则

- 一个 run 一个时间戳目录；训练/仿真/报告同目录；
- 不覆盖 `results/`；失败/中间结果保留；
- s,S 的 `summary.md`/`calibration.md`/`conclusion.md` 是最终口径；
- rolling 的 `summary.md`/`comparison.json`/`audit.json` 是最终口径；
- PPO 报告在训练目录内。

## 7. 不要做

- 不修改主项目 `code/`、`utils/`、`results/`；只读导入。
- 不重新采样 300 场景池；不重新生成测试路径；不用测试路径调参/选择。
- 不覆盖已有 `results/`；不删除失败/中间结果。
- 不把旧 q_in/h-target 的 s,S/active 数值与当前结果混表。
- 不把 rolling 的 50/100 子集与全量 3000 直接相减。
- 不把 PPO 5000 独立回合与 rolling 3000 配对路径做假配对。
- 不把 LB 当基线策略成本；不写“策略优于最优”。
- 不在策略层偷偷改 Qmax、加 stop offset 或隐藏安全界；要改就改模型并重跑。
- 不把 T60 策略/报告用于 T240（policy 有 T/Imax/hash 校验）。
- 不擅自 push；尤其是 `main` 和只读主项目。

## 8. 待确认/下一步

1. **确认 s,S 到货时序**：当前 2 层在途，q_t 在 t+2 期末到货；
   如果用户要求 L=3 表示 t+3 到货，需要增加一层并重跑 no-active/active。
2. **确认 q_T**：当前 q_T 不计采购费但仍按规则产生；是否需要 q_T=0/计费。
3. **active 结果仍 +4.20%**：若要继续缩小，优先做：
   - h 与 s/S 的 frozen-active 消融（固定 no-active 的 s,S，只优化 h）；
   - 多 seed / GA 多起点；
   - 明确 state-only 与 SDDP cuts 的信息集差异。
4. **rolling**：N=10/W=20 的缩减 SAA；可做 N=50/多 seed，
   但不要用 3000 测试集调参。
5. **PPO**：多 seed/warm-start/口径对齐（被动可用 I+τ vs I+τ+o1）；
   当前没有五项/审计归档，不能当“审计通过”。
6. **长篇扩展报告**：`对比实验结果总结_20261029.md` 的旧 §3/§4
   仍是 q_in/h-target 历史；若要求完全无旧数值，应继续清理，而不是把旧值当当前。

## 9. 一句话交接

当前正式口径是：
**s,S 无 q_in/q0、active h 双目标（无 T）、GA；rolling 全量 M=3000 为 -2.97%；
s,S no-active +5.91%、active +4.20%；PPO +2.27%（独立回合）。
旧 q_in/h-target 结果只作历史，不要混表。**
