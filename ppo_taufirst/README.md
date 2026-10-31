# PPO tau-first（新时序 `tau_t -> q_t -> Theta_t -> y_t`）

自包含 PPO 对比模块：环境、PPO、训练、总成本曲线、回合仿真均值和置信区间。
`common/` 只用于模型加载、共享配置、基线读取和文件工具。

> **当前协议**：actor 初始输出为 0，从零开始训练；训练日志逐回合记录总成本；
> 训练后用 3000 个独立回合仿真，报告总成本均值和 95% CI。

## 1. 新时序与状态

每期：

```
s_t=(I,n,o1,o2)
  -> 选主动调拨 tau_t（净量，Σtau=0）
  -> 选订货 q_t（0..Qmax）
  -> 抽样 Theta_t
  -> 规则被动调拨 y_t（本地优先 + 跨仓优先级）
  -> 转移
```

- 状态：`I(5), n(5×4), o1(5), o2(5)`；没有 q0 状态。
- 观测 36 维：`[I/Imax, n/N, o1/Qmax, o2/Qmax, t/(T-1)]`。
- 转移：
  `I_{t+1}=I_t+tau_t-Σ_dest y[dest,r,t]+o1_t`，
  `o1_{t+1}=o2_t`，`o2_{t+1}=q_t`，
  `n_{t+1,0}=Y+N_{0,0}`，`n_{t+1,j}=Σ_i N_{i,j}`。
- 动作 10 维：前 5 维 ×Imax 得 tau，后 5 维 ×Qmax 得 q；env 内部投影：
  - `-I <= tau <= Imax-I-Σo`，`Σtau=0`；
  - `0 <= q <= min(Qmax, Imax-(I+tau)-Σo)`。
  投影量写入 `info["projection"]`，评估时保存；不静默违反约束。

## 2. 被动调拨优先级

`prob.priority` 显式保存（0-based；例如仓 1 `[1,2,3,4,5]`，仓 2
`[2,1,3,4,5]`）。规则：

1. 所有仓先本地满足；
2. 剩余需求按 `tc_route`（由 `prob.priority` 生成）的
   `(tc, dest, source)` 升序全局贪心。

构造 `TauFirstEnv` 时校验每个目的地的 source 顺序与 `prob.priority` 一致；
`test/_t_env.py` 也检查。被动调拨可用库存按新时序为 `I+tau+o1`，
需求 `D=W_in+F_new`。

## 3. 奖励与 Theta

- 阶段成本：`Cq + h/2(I_t+I_{t+1}) + Cs W_in + atc/2 Σ|tau| + Σ tc_route*y`
  （万元）。
- PPO 奖励 = `-reward_scale × 阶段成本`（默认 `reward_scale=1e-3`）；
  报告的成本和曲线不做缩放。
- 训练 Theta 每步独立在线抽样（复制主模型 Poisson/Dirichlet 数学），
  不建 N×T 池；`ThetaSampler` 与 `rolling_horizon.scenarios.sample_pool`
  同种子逐位一致（测试）。

## 4. PPO

- 2 层 MLP actor / critic（128×128），对角 Gaussian，GAE，clip，entropy；
- CPU torch；同一 seed 可复现；`policy.pt` 保存/加载；
- **从零开始**：actor/critic 标准初始化，无手工 q/tau 偏置；
- `episode_log.csv` 每个回合一行：`episode, steps, total_cost, raw_reward,
  scaled_reward`；`train_episodes.log` 每个回合一行纯文本
  `episode=... total_cost=...`；`train_log.csv` 每次 update 一行。
- 训练日志里的量（控制台只打印前三个 + wall）：
  - `policy_loss`（旧日志的 `pi`）：PPO 的 negative clipped surrogate，
    可以为负，不要求降到 0；
  - `value_loss`（旧 `v`）：`0.5 × MSE(V(s), GAE return)`，
    单位是缩放后奖励的平方；
  - `approx_kl`（旧 `kl`）：只做诊断，**不参与更新**；PPO 用 ratio clip，
    不需要算 KL；
  - `clip_frac`（旧 `clip`）：ratio 被 clip 的比例，诊断量；
  - `wall`：墙钟秒数。
  `approx_kl`/`clip_frac` 只保留在 `train_log.csv`，控制台不再打印。

## 5. 命令

```bash
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"
```

### 5.1 测试

```bash
"$P" -B ppo_taufirst/test/_t_env.py
"$P" -B ppo_taufirst/test/_t_ppo.py
```

### 5.2 训练（从零；只需设置回合数）

```bash
"$P" -B ppo_taufirst/train.py --episodes 10000 --seed 2027 \
  --tag ppo_taufirst_T60_204105 --threads 4
```

输出：`policy.pt`、`train_log.csv`、`episode_log.csv`、
`train_episodes.log`、`train_curve.png`（每个回合总成本 vs 回合数）、
`run.json` 等。

### 5.3 总成本曲线

训练结束会自动生成；也可单独重画：

```bash
"$P" -B ppo_taufirst/plot_train.py \
  --run results/ppo_taufirst/<train_run> --baseline-mean 35833.3228
```

### 5.4 训练后回合仿真（5000 回合，均值和 95% CI）

```bash
"$P" -B ppo_taufirst/simulate.py \
  --policy results/ppo_taufirst/<train_run>/policy.pt \
  --episodes 5000 --seed 888 --threads 4
```

输出到 `<train_run>/simulation/`：`episode_costs.csv`、`summary.md`、
`episode_eval.json`、`train_curve.png`（从训练目录复制/生成）。

### 5.5 可选：配对评估 / 环境 sanity check

```bash
# 与基线 scene 前 50 条配对，含五项成本、行为、审计
"$P" -B ppo_taufirst/evaluate.py \
  --policy results/ppo_taufirst/<train_run>/policy.pt \
  --m-paths 50 --tag ppo_eval --threads 4

# base-stock 规则 sanity check（可选）
"$P" -B ppo_taufirst/diagnose.py --m-paths 50
```

## 6. 当前从零训练结果（时间特征 + 10000 回合）

- 训练：`results/ppo_taufirst/ppo_taufirst_tobs_10000ep_20261029_132639`
  - 10,000 episodes（600,000 env steps），墙钟约 652 s（CPU，4 线程）。
  - 观测含 `t/(T-1)`，`obs_dim=36`。
  - 最后 10% 回合平均成本 **38,525.31**。
  - `episode_log.csv` / `train_episodes.log`：每个回合的总成本。
- 回合仿真：`.../simulation`
  - 5,000 个独立回合（Theta seed=888，确定性动作，T=60），墙钟约 333 s。
  - **PPO 均值 36,645.85 万元/路径；95% CI [36,371.05, 36,920.64]**；
    std=9,913.89，SE=140.20。
  - 基线 3,000 条测试路径均值 35,833.32；差 **+812.52（+2.27%）**，
    95% CI [+320.14, +1,304.91]。
- 结论：加入时间特征后，PPO 从旧版约 37.29k 降到 36.65k；
  仍比 SDDP 基线高约 2.27%，95% CI 不含 0。单 seed、5000 回合独立仿真。
- **3000 条基线路径配对评估（2026-10-30 补，不重训）**：
  直接加载同一 `policy.pt`，在基线同一 3000 条路径上做确定性评估，
  Θ 取基线 `scene`；结果 **36,341.9921** vs 基线 35,833.3228，
  **+508.6693（+1.42%）**，95% CI [+372.95, +644.39]，t=7.35，
  胜率 23.70%，审计通过。五项差：采购 +2,718.99、持有 −191.57、
  停机 −4,172.36、主动 +414.54、被动 +1,739.07。
  归档：`results/ppo_taufirst/ppo_taufirst_tobs_10000ep_eval3000_20261030_115423/`。
  这是同一策略的另一种评估协议，不是策略改进；5000 独立回合（+2.27%）
  与 3000 配对（+1.42%）不能混用。
- 旧版无时间特征 8000 回合结果（37,285.95，+4.05%）保留作历史对照，
  不再作为当前主结果。

## 7. 文件

`env.py`、`ppo.py`、`train.py`、`plot_train.py`、`simulate.py`、
`evaluate.py`（可选配对/审计）、`diagnose.py`（可选 sanity check）、
`archive.py`、`config.toml`、`test/_t_env.py`、`test/_t_ppo.py`。

## 8. 限制与待确认

1. **末期 q_T=0**：默认与 rolling/SDDP 一致；`--allow-terminal-q` 可改。
2. **被动调拨可用库存**：按新时序用 `I+tau+o1`；主模型 `pas` 只用
   `I+tau`，若要逐约束一致需确认。
3. **优先级规则**：当前是全局 `(tc,dest,source)` 贪心（与
   `ss_static`/`docs/s,S.md` 同口径）；每个目的地的 source 顺序与
   `prob.priority` 一致。若你要逐目的地顺序处理，需要改。
4. **信息集**：PPO 是 state-only，观测包含 `t/(T-1)` 但不含未来 Theta、
   价值函数或历史；SDDP/rolling 使用更丰富信息；被动 y 是规则而非 LP 优化。
5. **动作投影**：Gaussian 原始动作经 env 投影，投影会扭曲梯度；
   投影量可查 `sim_*npz`/`comparison.json`。
6. **单 seed / 5000 回合**：当前是单次训练 + 单次仿真；论文建议多 seed。
