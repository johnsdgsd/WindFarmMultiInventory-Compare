下面给出**完整的非时变 \((s,S)\) 基准库存策略模型**，主策略为通用 10 参数，不预设 \(s_r=S_r\)；\(s_r=S_r\) 只作为优化结果的特例自然出现。被动调拨严格按成本优先级；主动调拨给出两种可选规则，主基准默认关闭，变体启用固定平衡规则。

---

# 非时变 \((s,S)\) 基准库存策略：完整数学模型

## 1. 策略参数

对每个仓库 \(r=1,\dots,N_R\)，参数为

\[
\theta=\{(s_r,S_r)\}_{r=1}^{N_R}\in\mathbb{R}^{2N_R},
\qquad
0\le s_r\le S_r\le I^{\max}_r.
\]

共 \(2N_R\) 个参数。主策略是通用 \((s,S)\)：当 \(IP_{r,t}<s_r\) 时补到 \(S_r\)；\(s_r<S_r\) 表示批量补货，\(s_r=S_r\) 是 order-up-to 特例。**不预设 \(s_r=S_r\)**，是否退化由优化结果决定。

另有两个固定的调拨模式开关：

- \(\text{mode}_\tau=0\)：无主动调拨，\(\tau_{r,t}=0\)（主基准）；
- \(\text{mode}_\tau=1\)：启用主动调拨平衡规则（变体，规则见第 4 节，不增加优化参数）。

---

## 2. 订货决策

阶段初观测 \(I_{r,t},q^{in}_{r,t},o_{l,r,t},n_{r,i,t}\)，导出

\[
W_{r,t}=N_r-\sum_{i=0}^{G-1}n_{r,i,t},
\]

\[
IP_{r,t}=I_{r,t}+q^{in}_{r,t}+\sum_{l=1}^{L-1}o_{l,r,t}-W_{r,t},
\]

容量余量

\[
R_{r,t}=I^{\max}_r-\left(I_{r,t}+q^{in}_{r,t}+\sum_{l=1}^{L-1}o_{l,r,t}\right).
\]

订货规则

\[
q^{out}_{r,t}
=
\mathbf{1}_{\{IP_{r,t}<s_r\}}
\cdot
\max\left\{0,\ \min\left\{S_r-IP_{r,t},\ R_{r,t}\right\}\right\}.
\]

容量约束由 \(R_{r,t}\) 截断自动满足，不需要单独约束 \(q^{out}\le Q^{\max}_r\)。

---

## 3. 被动调拨：严格成本优先级

被动调拨 \(y_{r,m,t}\) 表示“仓库 \(m\) 的期初库存满足仓库 \(r\) 的需求量”。规则固定为**本地优先 + 跨仓配对按 \(tc_{r,m}\) 升序全局贪心**。

### 3.1 需求实现

随机转移后，本期新失效

\[
F^{new}_{r,t}=\sum_{i=0}^{G-1}N_{r,i,G,t},
\]

本期需求

\[
D_{r,t}=W_{r,t}+F^{new}_{r,t}.
\]

### 3.2 本地优先

\[
y_{r,r,t}=\min\{I_{r,t},D_{r,t}\},
\qquad
\bar D_{r,t}=D_{r,t}-y_{r,r,t},
\qquad
\bar I_{r,t}=I_{r,t}-y_{r,r,t}.
\]

### 3.3 跨仓贪心

对所有跨仓配对 \((r,m)\)，\(r\ne m\)，按 \((tc_{r,m},r,m)\) 升序排序，记排序为

\[
(r_1,m_1),(r_2,m_2),\dots,(r_K,m_K),
\qquad K=N_R(N_R-1).
\]

依次执行

\[
y_{r_k,m_k,t}=\min\left\{\bar D_{r_k,t},\ \bar I_{m_k,t}\right\},
\]

\[
\bar D_{r_k,t}\leftarrow\bar D_{r_k,t}-y_{r_k,m_k,t},
\qquad
\bar I_{m_k,t}\leftarrow\bar I_{m_k,t}-y_{r_k,m_k,t}.
\]

直到所有需求满足或所有库存耗尽。

### 3.4 性质

\[
\sum_{m=1}^{N_R}y_{r,m,t}\le D_{r,t},
\qquad
\sum_{r=1}^{N_R}y_{r,m,t}\le I_{m,t}.
\]

实际替换量

\[
Y_{r,t}=\sum_{m=1}^{N_R}y_{r,m,t},
\]

待修递推

\[
W_{r,t+1}=W_{r,t}+F^{new}_{r,t}-Y_{r,t}\ge0.
\]

被动调拨只使用**期初在手库存 \(I_{m,t}\)**，不含本期到货 \(o_{1,r,t}\)。规则不是单期运输问题的最优解，但固定、可复现、严格遵循成本优先级。

---

## 4. 主动调拨

被动调拨后，计算含本期到货的剩余库存

\[
S^{end}_{r,t}=I_{r,t}-\sum_{s=1}^{N_R}y_{s,r,t}+o_{1,r,t}.
\]

主动调拨 \(\tau_{r,t}\) 本期决策、下一期生效，满足

\[
\sum_{r=1}^{N_R}\tau_{r,t}=0,
\qquad
S^{end}_{r,t}+\tau_{r,t}\ge0,
\qquad
S^{end}_{r,t}+\tau_{r,t}\le I^{\max}_r,
\qquad
\tau_{r,t}\le\sum_{m\ne r}S^{end}_{m,t}.
\]

库存递推

\[
I_{r,t+1}=S^{end}_{r,t}+\tau_{r,t}.
\]

### 4.1 模式 0：无主动调拨（主基准）

\[
\tau_{r,t}=0,\qquad \tau^\pm_{r,t}=0.
\]

主基准 BS-\((s,S)\) 默认此模式。10 个参数只控制订货，调拨只用第 3 节的被动规则。

### 4.2 模式 1：固定平衡规则（变体，不增加优化参数）

被动调拨后，定义各仓相对目标库存的偏差

\[
\Delta_{r,t}=S^{end}_{r,t}-S_r.
\]

超额仓集合与不足仓集合

\[
\mathcal{E}_t=\{r:\Delta_{r,t}>\delta\},
\qquad
\mathcal{S}_t=\{r:\Delta_{r,t}<-\delta\},
\]

其中 \(\delta\ge0\) 是死区，固定取 \(\delta=0.5d_r\) 或 \(\delta=1\)（不参与优化）。

对每一对 \((r,m)\in\mathcal{E}_t\times\mathcal{S}_t\)，按主动调拨路线成本 \(atc_{r,m}\) 升序排序（若 \(atc\) 为统一值，则按 \(r,m\) 编号排序），依次分配

\[
\tau_{r,m,t}
=
\alpha\cdot
\min\left\{
\Delta_{r,t}-\delta,\;
-\Delta_{m,t}-\delta,\;
S^{end}_{r,t}-\delta,\;
I^{\max}_m-S^{end}_{m,t}
\right\}^+,
\]

其中 \(\alpha\in(0,1]\) 是保守系数，固定取 \(\alpha=0.5\)，避免下一期需求随机导致反向缺货。每次分配后更新

\[
\Delta_{r,t}\leftarrow\Delta_{r,t}-\tau_{r,m,t},
\qquad
\Delta_{m,t}\leftarrow\Delta_{m,t}+\tau_{r,m,t}.
\]

净调拨量

\[
\tau_{r,t}=\sum_{m\ne r}\tau_{r,m,t}-\sum_{m\ne r}\tau_{m,r,t}.
\]

正负部分

\[
\tau^+_{r,t}=\max\{\tau_{r,t},0\},
\qquad
\tau^-_{r,t}=\max\{-\tau_{r,t},0\}.
\]

该规则把超额库存按成本从低到高调向不足仓，调拨量受双方偏差、库存非负和容量上限约束。\(\alpha,\delta\) 固定，不进入优化，因此参数仍为 \(2N_R\) 个。

---

## 5. 仿真器状态递推

给定随机计划 \(\xi\)，每期每仓每源等级 \(i=0,\dots,G-1\)：

\[
\lambda_{r,t}=\lambda_0\exp\left(\sum_k\beta_k(z_{k,r,t}-\bar z_k)\right),
\]

\[
P_{i,j}(\lambda_{r,t})=
\begin{cases}
\Pr(D=j-i), & i\le j<G,\\
\Pr(D\ge G-i), & j=G,\\
0, & j<i,
\end{cases}
\]

\[
(\Theta_{r,i,i,t},\dots,\Theta_{r,i,G,t})
\sim
\mathrm{Dirichlet}\bigl(sP_{i,i}(\lambda_{r,t}),\dots,sP_{i,G}(\lambda_{r,t})\bigr),
\]

只对 \(P_{i,j}>0\) 的分量构造。转移量

\[
N_{r,i,j,t}=n_{r,i,t}\Theta_{r,i,j,t},\quad j=i,\dots,G;
\qquad
N_{r,i,j,t}=0,\quad j<i.
\]

新失效与需求

\[
F^{new}_{r,t}=\sum_{i=0}^{G-1}N_{r,i,G,t},
\qquad
D_{r,t}=W_{r,t}+F^{new}_{r,t}.
\]

被动调拨 \(y\) 按第 3 节，主动调拨 \(\tau\) 按第 4 节。状态递推：

\[
W_{r,t+1}=W_{r,t}+F^{new}_{r,t}-Y_{r,t},
\]

\[
n_{r,j,t+1}=\sum_{i=0}^{G-1}N_{r,i,j,t}+Y_{r,t}\mathbf{1}_{\{j=0\}},
\qquad j=0,\dots,G-1,
\]

\[
S^{end}_{r,t}=I_{r,t}-\sum_{s=1}^{N_R}y_{s,r,t}+o_{1,r,t},
\qquad
I_{r,t+1}=S^{end}_{r,t}+\tau_{r,t},
\]

\[
o_{l,r,t+1}=o_{l+1,r,t},\ l=1,\dots,L-2,
\qquad
o_{L-1,r,t+1}=q^{in}_{r,t},
\qquad
q^{in}_{r,t+1}=q^{out}_{r,t}.
\]

初始条件

\[
I_{r,1}=0,\quad o_{l,r,1}=0,\quad q^{in}_{r,1}=q^{out}_{r,0},\quad
n_{r,0,1}=N_r,\quad n_{r,i,1}=0\ (i\ge1),\quad W_{r,1}=0.
\]

---

## 6. 单路径成本

\[
J(\theta;\xi)
=
\sum_{t=1}^{T}\sum_{r=1}^{N_R}
\left[
C\,q^{in}_{r,t}
+h\,\frac{I_{r,t}+I_{r,t+1}}2
+C_s\,W_{r,t}
+\frac{atc}{2}\left(\tau^+_{r,t}+\tau^-_{r,t}\right)
+\sum_{m=1}^{N_R}tc_{r,m}\,y_{r,m,t}
\right].
\]

模式 0 下 \(\tau^\pm=0\)，主动调拨项自动消失。

---

## 7. SAA 优化问题

固定训练随机计划 \(\xi^{(1)},\dots,\xi^{(K)}\)，

\[
\hat J_K(\theta)=\frac1K\sum_{k=1}^K J(\theta;\xi^{(k)}).
\]

优化问题

\[
\min_{\theta}\ \hat J_K(\theta)
\quad\text{s.t.}\quad
0\le s_r\le S_r\le I^{\max}_r,\quad r=1,\dots,N_R.
\]

随机计划固定后，\(\hat J_K(\theta)\) 是 \(\theta\) 的确定性函数。系统约束由决策规则和状态递推自动满足，优化只需箱约束。

---

## 8. 解析初值与搜索尺度

用“库存无限、需求即时满足”的基线仿真估计

\[
d_r=\frac1T\mathbb{E}\left[\sum_{t=1}^T F^{new}_{r,t}\right],
\qquad
\mu_r=Ld_r,
\qquad
\sigma_r=\sqrt{Ld_r}.
\]

报童临界比与分位数

\[
\beta_r=\frac{C_s}{C_s+h},
\qquad
z_r=\Phi^{-1}(\beta_r),
\]

初值

\[
S_r^{(0)}=\min\left\{I^{\max}_r,\ \mu_r+z_r\sigma_r\right\},
\qquad
s_r^{(0)}=\max\left\{0,\ S_r^{(0)}-Q_r\right\},
\]

其中 \(Q_r\) 取 \(1\) 或 \(d_r\)。**主策略保留 \(s_r^{(0)}<S_r^{(0)}\)，不强制相等。**

搜索半径

\[
\rho_{s_r}^{(0)}=\min\left\{S_r^{(0)},\ 3\sigma_r\right\},
\qquad
\rho_{S_r}^{(0)}=\min\left\{I^{\max}_r-s_r^{(0)},\ 5\sigma_r\right\}.
\]

多起点可选：

\[
S_r^{(0)}\in
\left\{
\mu_r+z_r\sigma_r,\ 
\mu_r+1.0\sigma_r,\ 
\min\{I^{\max}_r,\ \mu_r+2z_r\sigma_r\}
\right\}.
\]

---

## 9. 坐标下降算法

**输入**：初值 \(\theta^{(0)}\)，训练路径数 \(K_{\text{train}}\)，随机计划 \(\xi^{(1..K_{\text{train}})}\)，最大轮数 \(R_{\max}\)，网格点数 \(n_{\text{grid}}\)，停止阈值 \(\varepsilon,\eta\)。

**Step 0** 构造场景池与固定训练随机计划；算 \(d_r,\mu_r,\sigma_r,z_r\)；得 \(\theta^{(0)}\) 与 \(\rho_j^{(0)}\)；令 \(q=0\)。

**Step 1** 记 \(\theta_{\text{old}}=\theta^{(q)}\)。

**Step 2** 对坐标 \(j=1,\dots,2N_R\)（奇数 \(j=2r-1\) 为 \(s_r\)，偶数 \(j=2r\) 为 \(S_r\)）：

1. 确定箱约束：
   - \(j=2r-1\)：\(L_j=0,\ U_j=S_r^{(q)}\)（若 \(r\) 已更新则用新值）；
   - \(j=2r\)：\(L_j=s_r^{(q+1)},\ U_j=I^{\max}_r\)。
2. 构造一维网格：
   \[
   \mathcal{G}_j^{(q)}
   =
   \left\{
   \theta_j^{(q)}+\rho_j^{(q)}\left(\frac{2k}{n_{\text{grid}}-1}-1\right)
   :k=0,\dots,n_{\text{grid}}-1
   \right\}
   \cap[L_j,U_j].
   \]
3. 对每个 \(v\in\mathcal{G}_j^{(q)}\)，构造临时参数 \(\theta^{(q,j,v)}\)，计算
   \[
   \hat J_K(\theta^{(q,j,v)})
   =
   \frac1K\sum_{k=1}^{K_{\text{train}}}J(\theta^{(q,j,v)};\xi^{(k)}).
   \]
4. 取
   \[
   \theta_j^{(q+1)}
   =
   \arg\min_{v\in\mathcal{G}_j^{(q)}}\hat J_K(\theta^{(q,j,v)}),
   \]
   并列时取离 \(\theta_j^{(q)}\) 最近者。

**Step 3** 轮末检查：

1. \(\|\theta^{(q+1)}-\theta^{(q)}\|_\infty<\varepsilon\)，停；
2. \(|\hat J_K(\theta^{(q+1)})-\hat J_K(\theta^{(q)})|/\hat J_K(\theta^{(q)})<\eta\)，停；
3. \(q+1=R_{\max}\)，停；
4. 否则 \(q\leftarrow q+1\)，\(\rho_j^{(q)}=\max\{\rho_j^{(q-1)}/2,\rho_{\min}\}\)，返回 Step 1。

**Step 4（可选精搜）** 在 \(\theta^*\) 附近用更密网格（\(n_{\text{grid}}=10\sim15\)，\(K_{\text{train}}=50\)）重复一轮。

**Step 5（测试验证）** 用独立测试计划 \(\xi'^{(1..K_{\text{test}})}\) 报告

\[
\hat J_{\text{test}}(\theta^{**})
=
\frac1{K_{\text{test}}}\sum_{k=1}^{K_{\text{test}}}J(\theta^{**};\xi'^{(k)})
\]

及 \(95\%\) 置信区间和成本分解。

---

## 10. 主策略与变体

| 策略 | 参数 | 被动调拨 | 主动调拨 |
|---|---|---|---|
| **BS-\((s,S)\)**（主基准） | \(2N_R\) 个，通用 \((s,S)\) | 第 3 节严格优先级 | \(\tau=0\) |
| BS-\((s,S)\)-bal | 同上，不增加参数 | 第 3 节 | 第 4.2 节固定平衡规则，\(\alpha=0.5,\delta\) 固定 |

主基准不预设 \(s_r=S_r\)。若优化结果出现 \(s_r=S_r\)，说明在该成本参数和容量下批量补货无收益；若出现 \(s_r<S_r\)，说明触发式批量补货有收益。两种都是模型的合法输出，报告中应如实给出每个仓库的优化结果。

---

## 11. 实现注记

1. **随机计划固定**：\(\hat J_K(\theta)\) 对给定 \(\theta\) 是确定值；同一参数重复评估结果相同。抽样误差只体现在训练集是总体样本，用测试集和置信区间回答。
2. **被动调拨的排序**：每次评估前预计算配对排序，所有参数共用，避免重复排序。
3. **主动调拨模式 1 的守恒**：分配 \(\tau_{r,m,t}\) 后计算净 \(\tau_{r,t}\)，再检查 \(\sum_r\tau_{r,t}=0\)、\(S^{end}_{r,t}+\tau_{r,t}\in[0,I^{\max}_r]\)；若违反则截断并重新分配剩余量。
4. **约束满足**：\(s_r\le S_r\) 和 \(S_r\le I^{\max}_r\) 在网格构造时满足，轮末再检查一次。
5. **计算量**：每轮 \(2N_R n_{\text{grid}}\) 次评估，每次 \(K_{\text{train}}T\) 期仿真；\(N_R=5,n_{\text{grid}}=20,R_{\max}=3\) 约 600 次评估。