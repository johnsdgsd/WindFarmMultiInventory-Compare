# 滚动时域（全窗口两阶段 SAA）基准

按 `docs/滚动时域.md` 实现：每期以当前状态 + 固定优化池切片解两阶段随机 LP，
只执行当期 \(q_t\)；观测真实 \(\Theta_t\) 后解追索 LP 得 \(y,\tau\)。
窗口为剩余全部时域，\(V_{T+1}=0\)。

## 口径

- 状态：`I, n, o1..o(L-1)`；\(W=N-\sum n\) 由状态导出。
- 初始 `I0/o0/n0` 取 `problem_data`；没有额外的虚拟 q0 状态。
- **没有 q_in/q_out**；每期只有一个 \(q_t\)，采购成本按当期 \(Cq_t\)。
- 只有 \(q_t\) 是第一阶段；\(\tau>t\) 的 \(q\) 及全部状态/决策为追索。
- 状态转移：
  \(I_{t+1}=I_t-\sum_m y_{m,r,t}+o_{1,t}+\tau\)，
  \(o_{l,t+1}=o_{l+1,t}\)（\(l<L-1\)），\(o_{L-1,t+1}=q_t\)，
  \(n_{t+1,0}=N_{00}+Y\)，\(n_{t+1,j}=\sum_iN_{ij}\)。
- 每期容量：\(I_t+\sum_{l=1}^{L-1}o_{l,t}+q_t\le Imax\)。
- 转移后容量：\(I_{t+1}+\sum_l o_{l,t+1}\le Imax\)（`cap_next`）。
- 追索 LP：`cap_end` 限制真实下一期状态容量；目标含 \(-C_sY\) 和 \(hI^{out}/2\)。
- 允许 \(t=T\) 的 LP；无终端价值时 \(q_T=0\)。
- 真实路径取基线 `scene` 前 M 条，与 SDDP 配对；优化池 seed 独立；
  不是独立 `seed_real` 重采样。
- 采购按订货当期 `C q_t`，与主 SDDP 按到货量 `q_in` 计费的时点不同；
  逐期成本剖面比较必须保留这一说明。
- 不加安全界/fallback。

## 状态

正式运行正在后台进行：tag `rolling_T60_204105_final`，N=10、M=50、
seed_opt=2027、COPT、workers=4。结果尚未产出；完成前不填任何绩效数字。
更早的 `capfix`/`simpletest`/`quick` 运行口径或规模不符，均不作为论文结果。
结果以归档目录的 `summary.md`/`conclusion.md`/`comparison.json` 为准。

## 运行

```bash
cd "D:/WYJ/我的文件/论文/小论文/风场多仓库备件库存管理案例-对比实验"
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"

# 测试
"$P" -B rolling_horizon/test/_t_pool.py
"$P" -B rolling_horizon/test/_t_lp.py
"$P" -B rolling_horizon/test/_t_rollout.py
"$P" -B rolling_horizon/test/_t_evaluate.py

# quick smoke
"$P" -B rolling_horizon/run_evaluate.py --quick --workers 2 --threads 1

# 正式（N=10/M=50；当前后台运行的 tag）
"$P" -B -u rolling_horizon/run_evaluate.py \
  --n-scenarios 10 --real-paths 50 --seed-opt 2027 \
  --workers 4 --threads 4 --tag rolling_T60_204105_final
```

`--workers` 按真实路径并行；`threads` 是 COPT 单模型线程。
结果目录 `results/rolling_horizon/<tag>_<timestamp>/`，含 `summary.md`、
`comparison.md`、`conclusion.md`、`progress.log`、`run.log`、`lp_stats.csv`、
`audit.json`、`run.json`。失败运行也保留 `run.json`(failed)/`error.txt`。

详细推导与口径见 `docs/method-rolling-horizon.md`。
