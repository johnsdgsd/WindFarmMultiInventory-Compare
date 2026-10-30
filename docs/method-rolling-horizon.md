# 滚动时域（全窗口两阶段 SAA）基准 — 实现与评估

> 权威模型：`docs/滚动时域.md`（已按当前代码口径同步）。
> 本文记录实现口径、代码落点、验证、运行与结果；当前代码为
> “状态 \(I,n,o_1..o_{L-1}\)、单一 q、当期采购成本”。

## 1. 方法定位

在 \(t\) 期初已知

\[
s_t=(I_{r,t},n_{r,i,t},o_{1,r,t},\dots,o_{L-1,r,t}),
\]

窗口取剩余全部时域 \(\mathcal H_t=\{t,\dots,T\}\)，\(W=T-t+1\)。
对 \(N\) 条固定优化池路径解两阶段随机 LP：

- 第一阶段只有 \(q_{r,t}\)（无情景索引）；
- 其余 \(q_{\tau>t},y,\tau^\pm,S^{end},I,n,N,o\) 全部是情景追索变量；
- 真实 \(\Theta_t\) 在 \(q_t\) 之后观测，实际执行时解当期追索 LP 得 \(y_t,\tau_t\)；
- 采购成本按**当期** \(q_t\)：\(C q_{r,t}\)（不再有 q_in/q_out）。

目标：

\[
\min\frac1N\sum_{\omega}\sum_{\tau=t}^{T}\sum_r
\Big[Cq_{r,\tau}
+h\frac{I_{r,\tau}+I_{r,\tau+1}}2
+C_sW_{r,\tau}
+\frac{atc}2(\tau^+_{r,\tau}+\tau^-_{r,\tau})
+\sum_m tc_{r,m}y_{r,m,\tau}\Big],
\]

其中 \(W_{r,\tau}=N_r-\sum_i n_{r,i,\tau}\) 由状态导出，不存储。
终端 \(V_{T+1}=0\)，无终端近似。

## 2. 状态转移与约束

### 2.1 单一时间线

代码中每期只有一套状态变量，不建 q_in/q_out 或 in/out 两套变量：

\[
I_{r,t+1}=S^{end}_{r,t}+\tau_{r,t},\qquad
S^{end}_{r,t}=I_{r,t}-\sum_s y_{s,r,t}+o_{1,r,t},
\]

\[
o_{l,r,t+1}=o_{l+1,r,t}\ (l=1,\dots,L-2),\qquad
o_{L-1,r,t+1}=q_{r,t},
\]

\[
n_{r,0,t+1}=N_{r,0,0,t}+Y_{r,t},\qquad
n_{r,j,t+1}=\sum_iN_{r,i,j,t}\ (j\ge1).
\]

### 2.2 期初容量（确定性，第一阶段）

对 \(t\) 期加一次、与情景无关：

\[
I_{r,t}+\sum_{l=1}^{L-1}o_{l,r,t}+q_{r,t}\le I^{max}_r.
\]

这是修复 path=18 不可行的关键：\(q_t\) 是第一阶段变量，必须由
**当前已知状态**约束，不能依赖未来 \(y,\tau\) 在期末降库存；
完整可行性还要靠 `cap_next` 与追索 `cap_end` 一起保证。
代码位置：`rolling_horizon/lp.py::TwoStageLP.build`，约束名 `cap_init[r]`；
对每个未来情景期 \(\tau>t\) 还加同名 `cap[...]`：
\(I_{r,\tau}+\sum_{l=1}^{L-1}o_{l,r,\tau}+q_{r,\tau}\le I^{max}_r\)。

### 2.3 转移后的状态容量

对每个情景、每期 \(\tau\)：

\[
I_{r,\tau+1}+\sum_{l=1}^{L-1}o_{l,r,\tau+1}\le I^{max}_r,
\]

代码约束名 `cap_next[...]`。终端 \(T+1\) 也由它兜底。

### 2.4 其他约束

等级转移 \(N=\Theta n\)、\(F\)、被动源、需求/机队、\(\tau\) 守恒/上下界/来源、
\(I^{out}=S^{end}+\tau\)、\(q\in[0,Q^{max}]\)、状态上下界。
滚动模型的容量由 `cap_init`、未来期 `cap` 和 `cap_next` 组成；
追索 LP 另有 `cap_end`。不加入安全界、fallback 或额外松弛。

- \(t=T\) 仍求解两阶段 LP（窗口 \(W=1\)），第一阶段 \(q_T\) 的上界直接设为 0；
  \(t<T\) 时，追索变量 \(q_T\) 的上界也为 0（无终端价值）。
- 两阶段 LP 建模前校验状态：\(I\le I^{max}\)、\(\sum_i n_{r,i}\le N_r\)、
  \(I,o,n\ge0\)、\(\Theta\) 形状/行和/结构零；非法直接报错，不静默截断。
- 仿真审计是滚动方法自包含的：逐 (路径,期) 核对 `cap_init`、
  `cap_next`、在途递推、`S_end`/`I_out`、\(\tau\) 上下界/来源/守恒、
  等级转移、需求/机队和状态链；不依赖 ss_static 的审计器。

## 3. 追索 LP

给定 \(q_t\)、真实 \(\Theta_t\) 下的 \(F^{new}\)，解

\[
\min\sum_r\Big[\frac{atc}2(\tau^+_r+\tau^-_r)
+\sum_m tc_{r,m}y_{r,m}-C_sY_r+\frac h2 I^{out}_r\Big],
\]

约束为当期 `pas`、`dem`、`send_def`、`I_out`、`tau_*`，
以及 `cap_end`：\(I^{out}_r+\sum_l o^{next}_{l,r}\le Imax_r\)。
其中 \(o^{next}\) 的最后一层就是本期 \(q_t\)（
\(o^{next}_{L-1}=q_t\)），所以追索可行性**依赖**第一阶段订货；
两阶段 LP 里 \(q_t\) 与情景追索联合求解，不另加状态型安全上界。
\(-C_sY_r\) 保留，\(hI^{out}_r/2\) 是本期持有成本的一步前瞻。
若追索仍不可行，记录 path/t 并报错，不做策略层修补。

## 4. 场景与评估协议

- 优化池：\(N\times T\) 个独立 \(\Theta\)，固定 `seed_opt`；每期取
  \([t,T]\) 切片，CRN，无二次抽样。
- 真实路径：取基线 npz 的 `scene` 前 \(M\) 条，与 SDDP 同路径配对；
  与优化池独立；不重新采样、不用测试调参。
- 样本外绩效：

\[
\bar C^{roll}=\frac1M\sum_m C^{total}_m,\qquad
D_m=C^{total}_m-C^{SDDP}_m.
\]

- 优化 LP 目标是样本内 SAA，只用于产生 \(q_t\)，不作为绩效。
- 文档 Part 2 的独立 `seed_real` **未被采用**；正式配对路径仍是基线 `scene`
  前 \(M\) 条。这是已知限制，不是重新采样的独立真实路径。

## 5. 并行、日志与归档

- `workers`：按真实路径并行求解两阶段/追索 LP；
  `threads`：每个 COPT 模型内部线程。
- 每期写 `[rolling] t=... paths=... lp_calls=... elapsed=... eta=...`；
  同时写 `progress.log`/`run.log`。
- 结果目录在仿真前创建；失败也保留 `run.json`(failed)、`error.txt`、日志、
  配置副本；不覆盖旧结果。
- 输出：`policy.json`、`comparison.json/md`、`summary.md`、`conclusion.md`、
  `audit.json`、`lp_stats.csv`、`sim_<M>paths.npz`、`provenance.json`。

## 6. 代码落点

```text
rolling_horizon/
  scenarios.py   优化池、滑动切片、基线真实路径子集
  lp.py          State(I,o(L-1),n)、TwoStageLP、RecourseLP、COPT/HiGHS
  policy.py      order/fulfill/active_transfer；workers 并行；进度日志
  simulator.py   状态转移、成本、自包含审计
  evaluate.py    评估、配对、归档、失败 run
  run_evaluate.py CLI（含 --path-offset/--resume/--skip-index）
  run_m3000_shards.sh  6 分片全量运行 + 自动续跑 + 合并
  resume_m3000_stuck.sh 长跑卡死后的 checkpoint 串行续跑
  merge_shards.py      校验路径覆盖、拼接 sim npz、合并报告
  bench_parallel.py    workers/threads/进程级吞吐基准
  probe_numeric.py     status=5 单 LP 复现与 solver 对比
  verify_presolve.py   正常 LP 上 Presolve=0 与 auto 的目标一致性
  config/policy.toml
  test/_t_pool.py, _t_lp.py, _t_rollout.py, _t_evaluate.py
```

## 7. 验证

```bash
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
"$P" -B rolling_horizon/test/_t_pool.py
"$P" -B rolling_horizon/test/_t_lp.py
"$P" -B rolling_horizon/test/_t_rollout.py
"$P" -B rolling_horizon/test/_t_evaluate.py
"$P" -B ss_static/test/_t_lp_equiv.py
```

`_t_lp` 覆盖：单一 q、`cap_init`/未来期 `cap`/`cap_next`、状态转移、
等级/在途/需求/机队/tau、目标复算、t=T 的 q=0、追索 `cap_end`。
`_t_rollout` 覆盖：初始 o、q 记录、审计、可复现。
`_t_evaluate` 覆盖：quick 评估、日志、归档、index。

## 8. 运行

```bash
# quick smoke（N=2/M=2）
"$P" -B rolling_horizon/run_evaluate.py --quick --workers 2 --threads 1

# 正式（当前后台运行的 tag）
"$P" -B -u rolling_horizon/run_evaluate.py \
  --n-scenarios 10 --real-paths 50 --seed-opt 2027 \
  --workers 4 --threads 4 --tag rolling_T60_204105_final

# 更大 N（耗时显著增加）
"$P" -B -u rolling_horizon/run_evaluate.py \
  --n-scenarios 50 --real-paths 50 --seed-opt 2027 \
  --workers 4 --threads 4 --tag rolling_T60_204105_N50
```

## 9. 当前结果

正式运行（2026-10-28）：

```text
results/rolling_horizon/rolling_T60_204105_final2_20261028_223241/
N=10, M=50, seed_opt=2027, workers=6, threads=4
```

- 滚动均值 **39891.1219** vs 同 50 路径基线子集 **35451.4783**；
  配对差 **+4439.6437**（**+12.52%**），t=8.40，胜率 8.00%，审计通过。
- 换成全量 3000 路径基线 35833.3228 计算，滚动相对差为 **+11.32%**，
  低于主动调拨 s,S 的 **+13.65%**，也低于同 50 路径 active s,S 的约 +14.85%。
- 五项差（滚动−基线子集）：采购 -699.99、持有 +0.79、停机 +5523.66、
  主动调拨 -509.05、被动调拨 +124.24。
- 行为：履约率 0.8837 vs 0.9166；平均待修 12.28 vs 7.67；
  期末待修 60.22 vs 44.11；主动调拨 0 vs 101.81。
- 时间：首 10% +284.60，中间段 +2486.47，末 10% +1668.57；
  最好十分位第 3 段（-363.86），最差第 10 段（+1668.57）。
- LP：两阶段 3000 次、追索 3000 次，0 失败；t=1 89,955 变量；
  仿真墙钟 3810.8 s（6 个 path-worker）。

**结论**：在当前修正后的容量口径下，滚动样本外 gap 已低于主动调拨 s,S；
但仍高于基线约 11–13%，主要剩余缺口是停机/待修（+5523.66），
主动调拨仍为 0，且 N=10 的 SAA 与主 SDDP 的信息集/策略类不同。
更早的 `capfix`、`simpletest`、`quick` 运行只作历史/ smoke，不作论文结果。

### 9.1 已修复的关键问题

- 旧实现只有期末容量、没有期初容量；旧正式运行曾在 path=18/t=7 追索不可行。
- 本轮加入 `cap_init`、未来期 `cap`、`cap_next` 和追索 `cap_end`。
- path=39/t=22 曾因 `I=-9.3e-08` 被过严的状态校验拒绝；已把负值容差统一到
  `1e-6`，仅吸收 LP 的 FeasTol 残差，不改模型约束。
- 数值修复后正式运行完成，审计通过。

### 9.2 限制与可继续方向

- N=10 是缩减 SAA；N=50–100 的计算代价需单独评估。
- 采购成本按当期 \(q\)，与主 SDDP 按 q_in 计费的时点不同；q0/qT 边界需单列。
- 真实路径为基线 scene 子集，不是独立 `seed_real`。
- 旧时序（`final2`）已废弃；新时序全量 M=3000 为 -2.97%
  （t=-17.29，胜率 62%，审计通过），见 §9.3/§9.5。
  后续优化优先分析 N、W、多 seed 和长跑数值稳定性；不得擅自删除容量约束。
- 论文中统一 SDDP、滚动、s,S 的成本时点与终端处理。

### 9.3 新时序（τ→q）正式结果（2026-10-29）

新时序：`s_t -> τ_t -> q_t -> Θ_t -> y_t -> s_{t+1}`；状态
`I,n,o1,o2`（L=3, K=2），τ/q 第一阶段、y 追索；
`S_end=I+τ-Σy+o1`，`I_{t+1}=S_end`；每期 q 只出现一次，采购按当期 `Cq`。

**历史子集结果**（前 50 条基线路径）：

```text
results/rolling_horizon/rolling_T60_204105_taufirst3_20261029_122520/
N=10, M=50, seed_opt=2027
滚动 34307.1189 vs 子集 35451.4783
配对差 -1144.3594（-3.23%），t=-2.19，胜率 66%，审计通过
```

该结果使用**前 50 条**路径，子集基线 35451.4783 与随机 50 条子集的
32556.3100 不同，不能与 W 对照表混表；它只说明新时序相对旧时序
（旧时序 +12.52%）大幅改善，主动调拨从 0 变为约 120、停机成本下降。

**随机 50 条公平对照**（`--path-sample random --path-seed 2027`，
同一组随机路径同时评估滚动与基线）：

| W | 滚动均值 | 同路径基线 | 配对差 | 相对差 | t | 胜率 | 墙钟 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 32052.1364 | 32556.3100 | -504.17 | -1.55% | -2.22 | 52% | 1783 s |
| 30 | 32038.9216 | 32556.3100 | -517.39 | -1.59% | -2.28 | 54% | 2067 s |
| 60 | 32040.5728 | 32556.3100 | -515.74 | -1.58% | -2.28 | 54% | 3454 s |

- 两两比较：W20−W30 `+13.21`（t=1.86）、W20−W60 `+11.56`（t=1.57）、
  W30−W60 `-1.65`（t=-1.32）；W≥20 基本饱和，W20 最快。
- 随机子集不能直接与全量基线相减；随机 50 条配对约 -1.55%，
  不能写成“相对全量 -10.6%”。全量 M=3000 的正式结果是
  34769.9266 vs 35833.3228，即 **-2.97%**（见 §9.5）。
- W=10 在 t=7/60 出现大量 COPT status=5、预计约 6 h，已停止，不作正式结果。
- N=10/M=50 仍是缩减规模；M=100 随机公平实验已完成：
  `results/rolling_horizon/tau_W20_M100_rand_20261029_192726/`
  滚动 34426.5611 vs 同 100 子集 35108.0177，配对差 -681.4566
  （-1.94%），t=-3.34，胜率 61%，审计通过。该 run 从 t=35 checkpoint
  续跑，`wall_s`/`lp_stats.csv` 只覆盖后半段，不能当完整 LP 统计。
- **M=3000 全量已完成（2026-10-30 07:41）**：
  `results/rolling_horizon/tau_W20_M3000_full_merged_20261030_074140/`
  滚动 **34769.9266** vs 全量基线 **35833.3228**，配对差
  **-1063.3962（-2.97%）**，95% CI [-1183.9738, -942.8186]，
  t=-17.29，胜率 62.00%，审计通过。五项差：采购 -392.91、
  持有 +0.92（不显著）、停机 -630.73、主动 +83.99、被动 -124.66。
  时间剖面：首 10% +245.08、中间段 -1759.83、末 10% +451.35；
  6 种期末值口径下 -2.86% 到 -1.75%，均显著。
  这是 rolling 的主结果，与 s,S/active 同为 3000 路径逐路径配对；
  但仍受 N=10、W=20、窗口末端 V=0 限制，不能写成“优于最优”。

### 9.5 全量分片与合并（2026-10-29）

单进程 COPT 路径并行受 Python/GIL 限制：8 workers×3 threads 的
M=3000 估算约 27–33 h。实测多个独立进程可以绕开 GIL：

| 并行方式 | 24 个 t=1 LP/进程的墙钟 | 聚合吞吐 | M=3000 估算 |
|---|---:|---:|---:|
| 1 进程×8 workers×3 threads | 35.45 s/64 LP | ~1.8 LP/s | ~27 h |
| 4 进程×3 workers×2 threads | 10.7 s/24 LP/进程 | ~9.0 LP/s | ~5.6 h |
| **6 进程×2 workers×2 threads** | **10.7 s/24 LP/进程** | **~13.5 LP/s** | **~3.7 h** |

实现：`run_evaluate.py --path-offset` 取连续路径块；
`run_m3000_shards.sh` 启动 6 片、失败自动 `--resume`；
`merge_shards.py` 校验六片恰好覆盖 0..2999、按原顺序拼接
`sim_*.npz`，重新做逐路径配对、五项成本、行为、时间剖面、
期末值敏感性和合并审计。分片只是并行实现，不改变每条路径的状态转移
或策略；合并后的成本与单进程结果应一致（smoke 已验证）。

**实际运行与卡死（2026-10-30）**：

- 21:21:44 启动；shard1/5 约 10,920 s 正常完成；
- shard0/2/3/4 在 00:18 后停在 t=49/51：进程存活、CPU 持续占用，
  无 FAILED、无 `切 highs`，属于求解层卡死；
- 停止卡死进程后，`resume_m3000_stuck.sh` 按 checkpoint **串行**续跑
  （每片 `--time-limit 120`、`timeout 3600`），07:04–07:41 完成；
- 合并后 180,000 次两阶段 + 180,000 次追索 LP，0 失败；
- 正常分片墙钟约 3 h；实际总墙钟约 10.3 h（含卡死和串行续跑）。
  `merge_shards.py` 的 `wall_s` 取最长正常分片，不是实际总墙钟。
- 续跑分片的 `sim_*.npz` 会多出 `I/o/n/term_totals` 等 checkpoint 字段；
  合并按公共键交集并校验必需数组，不能假设所有 npz 键完全一致。

### 9.4 COPT 数值与断点（2026-10-29）

- **status=5 的原因**：退化两阶段 LP 在 COPT 默认 Presolve 下偶发数值失败；
  同一矩阵改用 `Presolve=0` 后 0.15–0.7 s 解出，目标值与正常 presolve
  在 24 个正常 LP 上最大相对差 `5.36e-12`。
- 已在 `_SparseLP.solve` 加入自动重试：先按 `copt_presolve`
  （默认 -1=auto）求解，非最优时用 `Presolve=0` 重解；正常 LP 不增加开销。
  这比原 HiGHS fallback 快得多：已知失败 LP 的 `highs-ipm` 实测约 300 s。
- 断点续跑：`--resume` 从 `checkpoint.npz/json` 恢复；Windows 下必须先
  `close()` `np.load` 返回的 NpzFile，否则 `os.replace` 报 WinError 5。
- `lp_stats.csv` 在每个 checkpoint 增量追加；续跑时载入已有行，避免
  只保留后半段 LP 统计。
- 仍未在策略层加安全界、stop offset 或删除 Qmax；数值处理只针对求解器/容差。
