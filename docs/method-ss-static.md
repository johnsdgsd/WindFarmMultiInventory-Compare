# 非时变 (s,S) 对比验证：评价函数、算法与验证协议

> 本文件是 `ss_static/` 的方法说明。权威策略口径仍是同目录 **`docs/s,S.md`**（用户提供）。
> 本文件补充：评价函数的数学定义、参数空间、可选优化算法、训练/验证/测试协议、
> 行为指标、时间剖面、期末值敏感性、约束审计与归档格式。

## 1. 范围、记号与文件边界

### 1.1 索引与符号

| 符号 | 含义 |
|---|---|
| \(r,m=1,\dots,N_R\) | 仓库编号；本算例 \(N_R=5\) |
| \(t=1,\dots,T\) | 月度周期；T60 的 \(T=60\)，T240 的 \(T=240\) |
| \(i,j=0,\dots,G-1\) | 健康等级；本算例 \(G=4\) |
| \(l=1,\dots,L-1\) | 在途层；本算例 \(L=3\)，代码中 `o` 数组下标为 \(0,\dots,L-2\) |
| \(N_r\) | 仓库 \(r\) 的机队规模（台） |
| \(I_r^{\max}\) | 仓库 \(r\) 的库存容量上限（件） |
| \(Q_r^{\max}\) | 仓库 \(r\) 的单期最大订货量（件） |
| \(C\) | 采购成本（万元/件） |
| \(h\) | 持有成本（万元/件·月） |
| \(C_s\) | 停机成本（万元/台·月） |
| \(atc\) | 主动调拨成本（万元/件）；主基准 mode 0 为 0 |
| \(tc_{r,m}\) | 从仓库 \(m\) 向仓库 \(r\) 被动调拨的单位成本（万元/件） |
| \(L\) | 订货提前期（月）；本算例 \(L=3\) |

**金额单位是万元**（主项目 `config/problem.toml` 的 `[cost]` 节）；
库存、待修、失效、替换等是台/件，不是金额。

### 1.2 方法自包含边界

本轮按“评价函数写在自己的文件夹内，不公共”实现：

- `ss_static/simulator.py`：纯 NumPy 状态转移、履约、成本、审计；
- `ss_static/policy.py`：`PolicyBase`（被动调拨）+ `SSStaticPolicy`（订货）；
- `ss_static/active.py`：可插拔主动调拨规则（mode 1，`ThresholdBalanceRule`）；
- `ss_static/evaluation.py`：**评价函数** \(\theta\mapsto\) 成本/明细；
- `ss_static/metrics.py`：行为统计、时间剖面、期末值敏感性；
- `ss_static/starts.py`：无限库存基线、解析初值、搜索半径；
- `ss_static/optimize.py`：参数空间、缓存、8 种优化算法；
- `ss_static/calibrate.py` / `evaluate.py` / `archive.py`：标定、评估、归档；
- `common/` 只保留模型加载、配置、基线读取、通用配对统计和通用文件工具。

不把任何 (s,S) 的评价/策略逻辑放在 `common/`。`docs/s,S.md` 保留为策略口径；
本文件是实现与验证说明。

## 2. 状态、时序与初始条件

### 2.1 状态

阶段 \(t\) 的期初状态为

\[
x_t=\bigl(I_t,\;o_t,\;n_t,\;q^{in}_t\bigr),
\]

其中 \(I_{r,t}\) 为期初在手库存，\(o_{l,r,t}\) 为第 \(l\) 层在途，
\(n_{r,i,t}\) 为等级 \(i\) 的运行台数，\(q^{in}_{r,t}\) 为上期订货、本期计费的到货/在途。

待修（停机）台数定义为

\[
W_{r,t}=N_r-\sum_{i=0}^{G-1}n_{r,i,t}.
\]

数值上 \(W_{r,t}\) 可能因等级修复出现 \(10^{-4}\) 量级的负值；决策中使用
\(W^+_{r,t}=\max(W_{r,t},0)\)，成本仍使用模型口径的 \(W_{r,t}\)（正常修复后 \(W\ge0\)）。

### 2.2 初始条件与虚拟阶段 0

\[
I_{r,1}=0,\quad o_{l,r,1}=0,\quad
n_{r,0,1}=N_r,\quad n_{r,i,1}=0\ (i\ge1),\quad W_{r,1}=0.
\]

在 \(t=0\) 的虚拟阶段，只有订货 \(q^0\) 是自由决策，不计当期成本：

\[
q^0_r=\mathbf 1\{IP_{r,0}<s_r\}\,
\max\{0,\min(S_r-IP_{r,0},\,R_{r,0},\,Q^{\max}_r)\},
\]

其中 \(I_{r,0}=o_{r,0}=n_{r,0}=q^{in}_{r,0}=0\)，故 \(IP_{r,0}=0\)、\(R_{r,0}=I^{\max}_r\)。
因此 \(s_r>0\) 时 \(q^0_r=\min(S_r,Q^{\max}_r)\)，否则为 0。
\(q^0\) 进入 \(q^{in}_{r,1}\)，在 \(t=1\) **计采购费**。

### 2.3 决策时序

1. 观测 \(x_t\)；
2. 按 (s,S) 规则选择 \(q^{out}_t\)（**只看状态，不看本期 \(\Theta_t\)**）；
3. 实现 \(\Theta_t\)，计算新失效 \(F^{new}_t\) 与需求 \(D_t\)；
4. 选择被动调拨 \(y_t\)（此时需求已知）；mode 0 的主动调拨 \(\tau_t=0\)；
5. 计算本期成本；
6. 状态转移到 \(x_{t+1}\)。

> 口径说明：SDDP 基线把 \(q^{out}_t\) 作为阶段 LP 的决策变量，和 \(\Theta_t\) 在同一个
> 阶段问题里；静态规则是 state-only。这是“规则策略 vs 全状态/阶段决策”的信息集差异，
> 论文中必须与“无主动调拨、规则履约”一起写明。

## 3. 订货规则

定义

\[
G_{r,t}=I_{r,t}+q^{in}_{r,t}+\sum_{l=1}^{L-1}o_{l,r,t},
\qquad
R_{r,t}=I^{\max}_r-G_{r,t},
\qquad
IP_{r,t}=G_{r,t}-W^+_{r,t}.
\]

\[
\boxed{
q^{out}_{r,t}
=
\mathbf 1\{IP_{r,t}<s_r\}\,
\max\bigl\{0,\;\min\{S_r-IP_{r,t},\;R_{r,t},\;Q^{\max}_r\}\bigr\}
}
\]

- 触发是**严格** \(IP<s_r\)，不是 \(\le\)；
- 参数空间 \(0\le s_r\le S_r\le I^{\max}_r\)；
- \(Q^{\max}_r\) 截断来自模型的 `q_out` 变量上界（`code/state_layout.py::upper`）。
  `docs/s,S.md` 认为 \(R\) 截断足够，但本实例 \(R\) 可大于 \(Q^{\max}\)，所以实现保留
  \(Q^{\max}\)；去掉它需要放宽模型变量界并重训 SDDP 基线，不能只在策略层去掉；
- 主策略**没有 stop offset**：所有 \(t\) 都按同一 \((s,S)\) 订货；\(q_T\) 不计采购费，
  但仍按规则执行。这是 MD 口径，不是 bug。

## 4. 需求、被动调拨与状态转移

### 4.1 新失效与需求

\[
N_{r,i,j,t}=n_{r,i,t}\Theta_{r,i,j,t},
\qquad
F^{new}_{r,t}=\sum_{i=0}^{G-1}N_{r,i,G,t},
\qquad
D_{r,t}=W^+_{r,t}+F^{new}_{r,t}.
\]

### 4.2 被动调拨：本地优先 + 全局贪心

本地阶段：

\[
y_{r,r,t}=\min\{I_{r,t},D_{r,t}\},\quad
\bar D_{r,t}=D_{r,t}-y_{r,r,t},\quad
\bar I_{r,t}=I_{r,t}-y_{r,r,t}.
\]

跨仓阶段：把所有 \(r\ne m\) 的配对按 \((tc_{r,m},r,m)\) 升序排序，依次执行

\[
y_{r,m,t}=\min\{\bar D_{r,t},\bar I_{m,t}\},\quad
\bar D_{r,t}\leftarrow\bar D_{r,t}-y_{r,m,t},\quad
\bar I_{m,t}\leftarrow\bar I_{m,t}-y_{r,m,t}.
\]

只使用期初库存 \(I_{r,t}\)，不含本期到货 \(o_{1,r,t}\)。替换量

\[
Y_{r,t}=\sum_{m=1}^{N_R}y_{r,m,t}.
\]

### 4.3 mode 0：无主动调拨

\[
\tau_{r,t}=0,\qquad \tau^+_{r,t}=\tau^-_{r,t}=0,
\qquad
S^{end}_{r,t}=I_{r,t}-\sum_{s=1}^{N_R}y_{s,r,t}+o_{1,r,t},
\qquad
I_{r,t+1}=S^{end}_{r,t}.
\]

### 4.4 等级与在途递推

\[
n_{r,0,t+1}=Y_{r,t}+N_{r,0,0,t},
\qquad
n_{r,j,t+1}=\sum_{i=0}^{j}N_{r,i,j,t}\quad(j\ge1),
\]

\[
W_{r,t+1}=W_{r,t}+F^{new}_{r,t}-Y_{r,t},
\]

\[
o_{l,r,t+1}=o_{l+1,r,t}\ (l=1,\dots,L-2),
\qquad
o_{L-1,r,t+1}=q^{in}_{r,t},
\qquad
q^{in}_{r,t+1}=q^{out}_{r,t}.
\]

等级转移后按模型的 `round_repair` 做 6 位小数取整和 \(10^{-4}\) 容差修复：
只从最大分量扣除不超过 `repair_tol` 的 \(\sum n\) 超额，不做缩放。

## 5. 单路径成本与计费口径

\[
J(\theta;\xi)
=
\sum_{t=1}^{T}\sum_{r=1}^{N_R}
\left[
C\,q^{in}_{r,t}
+\frac h2\bigl(I_{r,t}+I_{r,t+1}\bigr)
+C_s\,W_{r,t}
+\sum_{m=1}^{N_R}tc_{r,m}\,y_{r,m,t}
\right].
\]

mode 0 下主动调拨项为 0。计费口径：

- \(q^{in}_{r,1}=q^0_r\)，\(q^{in}_{r,t}=q^{out}_{r,t-1}\)（\(t\ge2\)）；
- 因此采购成本为
  \[
  C\sum_{r=1}^{N_R}\left(q^0_r+\sum_{t=1}^{T-1}q^{out}_{r,t}\right);
  \]
  \(q_T\) 不计采购费；
- 期末库存 \(I_{r,T+1}\) 进入第 \(T\) 期持有成本（\(\frac h2(I_T+I_{T+1})\)）；
- 期末待修 \(W_{r,T+1}\) **不计**停机成本；
- 主目标没有 terminal value / salvage；期末值只在评估中做敏感性分析。

**金额单位是万元；库存/待修/失效/替换的单位是台或件。**

## 6. 评价函数与 SAA 目标

### 6.1 评价函数

固定参数 \(\theta=(s,S)\) 和场景路径集合
\(\mathcal P=\{\xi^{(1)},\dots,\xi^{(P)}\}\)，定义

\[
\boxed{
E(\theta;\mathcal P)
=
\frac1{|\mathcal P|}\sum_{\xi\in\mathcal P}J(\theta;\xi)
}
\]

其中 \(J(\theta;\xi)\) 由第 5 节的闭环仿真给出。实现接口：

```python
# ss_static/evaluation.py
class SSEvaluator:
    def __init__(self, prob, options: dict): ...
    def evaluate(self, theta, paths, *, record=False) -> EvaluationResult: ...
    def evaluate_cost(self, theta, paths) -> float: ...

def evaluate_cost(prob, options, theta, paths) -> float: ...
```

- 输入：no-active 时 `theta = [s_1..s_K, S_1..S_K]`（2K）；
  active 时 `theta = [s_1..s_K, S_1..S_K, h_1..h_K]`（3K）；
  `paths` 为 `(P,T)` 的整数场景号；
- 输出：`EvaluationResult.cost = E(theta; paths)`，以及
  `path_costs (P,)`、`term_totals (P,5)`；`record=True` 时还有
  `step_terms (P,T,5)`、`arrays`、`behavior`；
- 评价函数不建 COPT 模型，不读基线 npz；它只依赖模型 `prob`、策略选项和路径。
- 计算量：\(O(P\,T\,N_R\,G^2)\) 的状态/成本运算；单次 2000×60 仿真约 0.1–0.5 s。

### 6.2 SAA 目标

训练固定路径 \(\xi^{(1)},\dots,\xi^{(K)}\)：

\[
\hat J_K(\theta)=E(\theta;\{\xi^{(k)}\}_{k=1}^K).
\]

优化问题：

\[
\min_{\theta\in\Theta}\ \hat J_K(\theta),
\qquad
\Theta=\{\theta=(s,S):0\le s_r\le S_r\le I^{\max}_r\}.
\]

## 7. 参数空间与无约束变换

### 7.1 物理空间

\[
\Theta=\{\theta:0\le s_r\le S_r\le I^{\max}_r,\ r=1,\dots,N_R\}.
\]

`clip(theta)`：

\[
s_r\leftarrow \min\{\max(s_r,0),\,S_r\},\qquad
S_r\leftarrow \min\{\max(S_r,s_r),\,I^{\max}_r\}.
\]

整数模式在 `clip` 内先四舍五入再修复 \(s\le S\le I^{\max}\)。

### 7.2 sigmoid 变换（给 NM/Powell/DE/SA/GA）

对 \(u_r,v_r\in[-12,12]\)：

\[
s_r=I^{\max}_r\sigma(u_r),
\qquad
S_r=s_r+\bigl(I^{\max}_r-s_r\bigr)\sigma(v_r),
\qquad
\sigma(x)=\frac1{1+e^{-x}}.
\]

该变换把 \(\mathbb R^{2N_R}\) 映射到 \(\Theta\)，解码后必然 \(s_r\le S_r\) 且不越界。
解码后若 \(s_r\) 或 \(S_r\) 距 \(0\)/\(I^{\max}_r\) 小于 \(10^{-5}I^{\max}_r\)，
吸附到边界；整数模式再四舍五入。逆变换：

\[
u_r=\operatorname{logit}\!\left(\frac{s_r}{I^{\max}_r}\right),
\qquad
v_r=\operatorname{logit}\!\left(\frac{S_r-s_r}{I^{\max}_r-s_r}\right),
\]

边界处用 \(10^{-9}\) 截断。

### 7.3 随机采样

物理空间的均匀采样：

\[
s_r\sim U(0,I^{\max}_r),\qquad
\Delta_r\sim U(0,I^{\max}_r-s_r),\qquad
S_r=s_r+\Delta_r.
\]

随机搜索使用拉丁超立方（SciPy `qmc.LatinHypercube`）生成
\((u^{(1)}_r,u^{(2)}_r)\)，再按上式映射；scipy 不可用时退化为
`rng.random`。

## 8. 优化算法

所有算法共用：

- 同一个确定性评价函数 `objective(theta) -> float`；
- `CachedObjective`：按物理 \(\theta\) 四舍五入到 9 位缓存；整数模式按整数缓存；
  缓存只在固定训练路径集合内使用，每个算法独立缓存（不会混路径集）；
  `n_calls` 为调用次数（含缓存命中），`n_unique` 为真实仿真次数；
- 每个算法独立缓存、独立随机种子；算法比较以 `n_unique` 和实际墙钟为准；
- 多起点：解析初值 `mean/one_sigma/two_z`（见第 9 节）；
- 返回值：`OptimizeResult(algorithm, theta, cost, n_calls, n_unique,
  wall_s, history, start_results, meta)`。

### 8.1 坐标下降（`coordinate`）

对每个起点 \(\theta^{(0)}\) 和每一轮 \(q\)：

1. 对坐标 \(j=1,\dots,2N_R\)（\(j\le N_R\) 为 \(s_j\)，否则为 \(S_{j-N_R}\)）：
   \[
   L_j=\begin{cases}0,&j\le N_R\\ s_{j-N_R},&j>N_R\end{cases},
   \qquad
   U_j=\begin{cases}S_j,&j\le N_R\\ I^{\max}_{j-N_R},&j>N_R\end{cases};
   \]
2. 一维网格
   \[
   \mathcal G_j=
   \left\{\theta_j+\rho_j\left(\frac{2k}{n_{\text{grid}}-1}-1\right):
   k=0,\dots,n_{\text{grid}}-1\right\}\cap[L_j,U_j],
   \]
   并加入当前中心；
3. 更新
   \[
   \theta_j^{(q+1)}=
   \arg\min_{v\in\mathcal G_j}\hat J_K(\theta^{(q,j,v)}),
   \]
   并列时取离当前值最近者；
4. 轮末：若 \(\|\theta^{(q+1)}-\theta^{(q)}\|_\infty<\epsilon\)
   或相对改进 \(< \eta\)，停止；否则 \(\rho_j\leftarrow\max(\rho_j/2,\rho_{\min})\)。

整数模式下 \(\mathcal G_j\) 为 \([L_j,U_j]\) 内的整数点，半径至少 1。
这是 `docs/s,S.md` §9 的 MD 口径。

### 8.2 随机搜索 / LHS + 局部精修（`random`）

1. 用 LHS 在物理空间采样 \(n_{\text{samples}}\) 个点，连同解析初值一起评价；
2. 取最好的 \(n_{\text{top}}\) 个点；
3. 对 \(k=0,\dots,n_{\text{refine}}-1\)，从 top 点中轮换取基點，做
   \[
   \theta'=\operatorname{clip}\bigl(\theta+\sigma_k\odot
   [I^{\max},I^{\max}]\odot z\bigr),\quad z\sim\mathcal N(0,I),
   \]
   \[
   \sigma_k=\sigma_0\left(\frac{\sigma_{\min}}{\sigma_0}\right)^{k/(n_{\text{refine}}-1)};
   \]
4. 若改进则替换对应 top 点，并更新全局最优。

### 8.3 Nelder-Mead / Powell（`nelder-mead`, `powell`）

在 \(u\) 空间上用 SciPy：

\[
\min_{u\in[-12,12]^{2N_R}}\ \hat J_K(\operatorname{decode}(u)).
\]

- Nelder-Mead：`method="Nelder-Mead"`, `maxfev`、`xatol`、`fatol`；
- Powell：`method="Powell"`, `bounds`、`maxfev`、`xtol`、`ftol`；
- 每个解析初值一条链，取最好。

### 8.4 差分进化（`differential-evolution`）

\[
\min_{u\in[-12,12]^{2N_R}}\ \hat J_K(\operatorname{decode}(u))
\]

用 `scipy.optimize.differential_evolution`：`popsize`、`maxiter`、`tol`、
`seed`、`init="sobol"`、`polish=false`；可多次 restart。
实际评估数约为 \(O(\text{popsize}\cdot 2N_R\cdot \text{maxiter})\)。

### 8.5 对偶退火（`dual-annealing`）

`scipy.optimize.dual_annealing`，`maxiter`、`maxfun`、`seed`、
`no_local_search=true`；可多次 restart。

### 8.6 遗传算法（`genetic`）

实数编码 GA（在 \(u\) 空间）：

- 初始种群：`starts` 的 \(u\) + 随机 \(U(-12,12)\)；
- 选择：锦标赛；
- 交叉：BLX-\(\alpha\)，
  \[
  c_j\sim U\bigl(\min(p_{1j},p_{2j})-\alpha|p_{1j}-p_{2j}|,\;
  \max(p_{1j},p_{2j})+\alpha|p_{1j}-p_{2j}|\bigr);
  \]
- 变异：以概率 \(p_m\) 加 \(\mathcal N(0,\sigma_m^2)\)；
- 精英保留 `elite` 个；迭代 `generations` 代；
- 预算由 `population × (generations+1)` 控制。

### 8.7 模拟退火（`simulated-annealing`）

从解析初值出发（多链）：

\[
T_0=\tau_0\max(|f(x_0)|,1),\qquad T_{k+1}=\alpha T_k,
\]

\[
x'=\operatorname{clip}\bigl(x+\sigma z\bigr),\quad z\sim\mathcal N(0,I),
\]

\[
x\leftarrow\begin{cases}
x',&f(x')\le f(x)\ \text{或}\ U(0,1)<e^{-(f(x')-f(x))/T_k},\\
x,&\text{否则}.
\end{cases}
\]

记录每条链的最优；链数 `n_chains`，总迭代 `n_iters` 在各链间平分。

### 8.8 算法默认预算

见 `ss_static/config/policy.toml` 的 `[search.*]`。首次运行建议：

- 主结果：`coordinate`（T60/T240 全量训练/验证路径）；
- 算法对比：T60 上用较小训练/验证路径和固定预算；
- 所有算法的 `n_unique`、`wall_s`、`train/val` 成本都写入归档，不能只报“最好结果”。
- 算法对比时可用 `--budget-unique B`：`calibrate` 会把各算法调到约同一真实评估预算
  （坐标下降缩放 `n_grid`；随机搜索均分采样/精修；NM/Powell 按起点均分；
  DE 按 `popsize·2N_R` 换算 `maxiter`；dual annealing 设 `maxfun`；
  GA 按 `population` 换算 `generations`；SA 设 `n_iters`）。
  实际 `n_unique` 以归档为准，不在不同预算下直接比较算法优劣。

## 9. 解析初值与搜索半径

用“库存无限、需求即时满足”的仿真估计各仓月失效率 \(d_r\)：

\[
\mu_r=L d_r,\qquad \sigma_r=\sqrt{L d_r},
\qquad
\beta=\frac{C_s}{C_s+h},\qquad
z=\Phi^{-1}(\beta).
\]

初值：

\[
S_r^{(0)}=\min\{I^{\max}_r,\ \mu_r+z\sigma_r\},
\qquad
s_r^{(0)}=\max\{0,\ S_r^{(0)}-Q_r\},
\]

其中 \(Q_r\) 默认取 1，可配置为 \(d_r\)。多起点还使用
\(\mu_r+\sigma_r\) 与 \(\min\{I^{\max}_r,\mu_r+2z\sigma_r\}\)。
搜索半径：

\[
\rho_{s_r}^{(0)}=\min\{S_r^{(0)},3\sigma_r\},
\qquad
\rho_{S_r}^{(0)}=\min\{I^{\max}_r-s_r^{(0)},5\sigma_r\}.
\]

整数模式下初值取 floor/ceil 两个变体，半径至少 1。

## 10. 验证协议

### 10.1 训练/验证/测试

| 用途 | 路径 | 是否参与选择 |
|---|---|---|
| 训练 | seed 777，`train_paths` 条，同一 300 场景池 | 是（优化目标） |
| 验证 | seed 888，`val_paths` 条，同一 300 场景池 | 是（选最终策略） |
| 测试 | 基线 npz 的 `scene` 数组 | **否** |

- 测试路径不重新生成；T60 为 2000×60，T240 为 1000×240；
- 多起点/多算法在验证集上比较；测试只在 `evaluate.py` 跑一次；
- 每个算法的每个起点最终 θ 都在验证集上重评；先在每个算法内部选验证成本最低的起点，
  再在所有算法候选上取严格最小；仅数值并列（\(\le 10^{-9}\)）时选
  \(\sum_r S_r\) 更小者（更简洁/更少库存），报告中显式写明。

### 10.2 配对比较

对同一测试路径 \(k=1,\dots,P\)：

\[
\Delta_k=J(\theta;\xi^{(k)})-J(\theta_{\text{base}};\xi^{(k)}),
\]

\[
\bar\Delta=\frac1P\sum_k\Delta_k,\qquad
SE=\frac{s_\Delta}{\sqrt P},\qquad
t=\frac{\bar\Delta}{SE},
\]

\[
\text{95\% CI}=\bar\Delta\pm1.95996\,SE,\qquad
\text{win rate}=\frac1P\sum_k\mathbf 1\{\Delta_k<0\},
\qquad
\text{rel}=\frac{\bar\Delta}{\bar J_{\text{base}}}.
\]

- \(\Delta>0\) 表示策略比基线贵；
- 不能用 SDDP 的 LB 当作基线策略成本；LB 只是模型下界；
- 成本单位万元。

### 10.3 五项成本

每路径每期五项：采购、持有、停机、主动调拨、被动调拨。逐路径汇总后在
策略/基线之间配对；报告给出每项均值、差值、95% CI。

### 10.4 行为指标

| 指标 | 定义 |
|---|---|
| 总失效 | \(\frac1P\sum_p\sum_{t,r}F^{new}_{r,t,p}\) |
| 总替换 | \(\frac1P\sum_p\sum_{t,r}Y_{r,t,p}\) |
| 替换率 | \(\frac1P\sum_p\min\{1,\sum Y/\sum F\}\) |
| 期末待修 | \(\frac1P\sum_p\sum_r W_{r,T+1,p}\)（Σ仓） |
| 平均期初待修 | \(\frac1{PT}\sum_{p,t}\sum_r W_{r,t,p}\)（Σ仓） |
| 平均期初库存 | \(\frac1{PT}\sum_{p,t}\sum_r I_{r,t,p}\)（**Σ仓**；不含在途和 \(q\)） |
| 下单量 | \(q^0+\sum_t q^{out}_t\)（含免费 \(q_T\)） |
| 计费下单量 | \(q^0+\sum_{t=1}^{T-1}q^{out}_t\) |
| 订单频率 | \(\frac1{PTN_R}\sum_{p,t,r}\mathbf 1\{q^{out}_{r,t,p}>0\}\) |
| 本地/跨仓替换 | \(y_{r,r}\) / \(y_{r,m},r\ne m\) |
| 主动调拨量 | \(\frac12\sum_{t,r}(\tau^++\tau^-)\) |

> 旧报告曾把基线的“平均库存”按每仓统计、策略按五仓合计，导致 5 倍夸大；
> 本版统一为 **Σ仓合计**，并用持有成本恒等式核对：
> \(hT(\bar I_{\text{policy}}-\bar I_{\text{base}})\) 应约等于持有成本差。

### 10.5 时间剖面

逐期平均成本

\[
\bar c_t^{pol}=\frac1P\sum_p c^{pol}_{t,p},\qquad
\bar c_t^{base}=\frac1P\sum_p c^{base}_{t,p},
\]

\[
\Delta_t=\bar c_t^{pol}-\bar c_t^{base},\qquad
\text{cum}_t=\sum_{s=1}^t\Delta_s.
\]

按十分位汇总 \(\sum_{t\in\text{decile}}\Delta_t\)，并报告首 10%、末 10%、
中间段的差。该剖面用于识别有限视界、失效率爬坡、期末停订等影响；
主目标仍是全 T 期总成本。

### 10.6 期末值敏感性

主目标没有期末价值。为说明边界，定义

\[
\tilde J(\theta;\xi)
=
J(\theta;\xi)
-c_{\text{salv}}\sum_{r=1}^{N_R}I^{out}_{r,T+1}
+c_{\text{term}}\sum_{r=1}^{N_R}W^{out}_{r,T+1},
\]

在
\[
(c_{\text{salv}},c_{\text{term}})\in
\{(0,0),(0,C_s),(0,C),(C,0),(C,C_s),(C,C)\}
\]
上做配对比较。它**只作敏感性**，不改变主结果。

### 10.7 约束审计

逐 (路径,期) 调用主模型 `check_stage`，并额外检查 \(q\in[0,Q^{\max}]\)。
审计项包括：`pas`、`fleet`、`dem`、`grade0`、`grade`、`N链接`、`结构零`、
`F_new`、`W守恒`、`send_def`、`tau_*`、`inv`、`pipe*`、`cap`、`q.下`、`q.上`、
`链.状态`。容差：`tol=1e-6`、`state_tol=1e-4`。
需求满足、采购恒等式、五项和、状态链、LP 对拍由测试覆盖：

- `ss_static/test/_t_audit.py`：审计、采购恒等式、五项和；
- `ss_static/test/_t_lp_equiv.py`：固定 q/y 放进主模型 LP，逐项对拍
  `stage_cost`、状态、五项成本（290 项）；
- `ss_static/test/_t_evaluation.py`：评价函数、record/非 record、确定性；
- `ss_static/test/_t_optimize.py`：8 种算法、可行性、确定性、预算；
- `ss_static/test/_t_archive.py`：归档、索引、T240 报告元数据、库存单位。

> 报告里的 `terms_vs_stage_cost_max_dev` 是仿真器内部
> \(\sum_k terms_k=cost\) 的恒等式，恒为 0；真正的 LP 对拍是
> `_t_lp_equiv.py`，不要把前者当成后者。

## 11. 归档与对比验证总结

每次标定/评估写入新目录，禁止覆盖：

```
results/ss_static/
  index.csv                       # append-only 跨运行索引
  <tag>_calib_<timestamp>/
    run.json                      # 状态/配置/版本/T/算法/预算
    policy.json                   # s,S + options + problem 元数据 + calibration
    calibration.md                # 标定/算法对比总结
    algorithm_summary.csv         # 每个算法一行
    algorithm_starts.csv          # 每个算法/起点一行
    optimization_history.csv      # 唯一评价历史
    paths_train.npz / paths_val.npz
    provenance.json
    config/                       # compare/policy/problem/solver/problem_data 副本
  <tag>_eval_<timestamp>/
    run.json
    policy.json
    sim_<P>paths.npz / .json      # 与主项目 simulate.py 同名同形状
    sim_report_<timestamp>.txt
    comparison.json / comparison.md
    summary.md                    # 对比验证总结
    time_profile.csv / time_profile_deciles.csv
    terminal_sensitivity.json
    provenance.json
    config/
    calibration/                  # 标定材料副本
```

`index.csv` 列：

```
run_id,kind,status,created_at,config,T,n_paths,algorithm,selected_start,
train_cost,val_cost,test_cost,baseline_mean,diff_mean,rel_pct,win_rate,
audit_ok,policy_json,summary_md
```

`summary.md` 固定包含：运行信息、验证协议、主结果与配对统计、五项成本、
行为诊断（Σ仓）、时间剖面、期末值敏感性、约束审计、结论与限制。
`calibration.md` 固定包含：解析初值、算法/起点表、选择结果、口径说明。

所有运行结束后可生成跨运行总结（读取 `index.csv`），汇总 T60/T240 主结果
与 T60 算法对比；该文件只新增，不覆盖。

## 12. 限制与解释边界

1. **规则 vs 优化**：静态 (s,S) 无主动调拨、履约用固定成本优先级规则、
   订货只看 IP（把健康等级聚合成 W）；SDDP 使用完整状态和阶段决策。
   差异不是单纯的“订货规则”差异。
2. **信息集**：SDDP 的 \(q^{out}\) 是阶段 LP 决策，和 \(\Theta_t\) 同阶段；
   静态规则是 state-only。论文中必须写明。
3. **有限视界**：无 stop offset，末期仍按 (s,S) 补货；期末库存/待修不计价值。
   时间剖面和期末值敏感性用于说明该边界。
4. **Qmax**：\(q\le Q^{\max}\) 是模型变量上界；去掉需要放宽模型并重训基线。
5. **场景池**：训练/验证/测试都用同一 300 场景池；属于路径级配对，
   不是换池外推。
6. **基线成本**：比较的是两个可执行策略在同一路径上的成本；
   不是“策略成本 vs LB”。基线自身有 gap。
7. **成本单位**：万元；库存/待修是台/件。
8. **测试不调参**：测试路径不参与搜索、起点选择或算法选择。

## 13. 复现命令

```bash
cd "D:\WYJ\我的文件\论文\小论文\风场多仓库备件库存管理案例-对比实验"
P="D:/WYJ/Anaconda/Anaconda3/envs/MyRL/python.exe"

# 测试
"$P" -B ss_static/test/_t_policy.py
"$P" -B ss_static/test/_t_evaluation.py
"$P" -B ss_static/test/_t_optimize.py
"$P" -B ss_static/test/_t_archive.py
"$P" -B ss_static/test/_t_audit.py
"$P" -B ss_static/test/_t_lp_equiv.py
"$P" -B ss_static/test/_t_pathgen.py
"$P" -B ss_static/test/_t_baseline.py
"$P" -B ss_static/test/_t_calibrate.py

# T60 主结果（坐标下降）
"$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T60
"$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T60_calib>/policy.json --tag ss_static_T60

# T240 主结果
COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_calibrate.py --algo coordinate --tag ss_static_T240
COMPARE_CONFIG=config/compare_T240.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/<T240_calib>/policy.json --tag ss_static_T240

# T60 多算法对比（示例：500/500 路径，固定预算）
"$P" -B ss_static/run_calibrate.py --algo all \
  --train-paths 500 --val-paths 500 --budget-unique 1500 \
  --tag ss_static_T60_algo_compare

# 整数敏感性 / smoke
"$P" -B ss_static/run_calibrate.py --algo coordinate --integer --tag ss_static_T60_int
"$P" -B ss_static/run_calibrate.py --quick --tag ss_static_T60_quick
```

## 14. 主动调拨（mode 1，可选变体）

### 14.1 定位

主基准 mode 0 没有主动调拨。本轮新增一个**可插拔**的主动调拨规则
`ThresholdBalanceRule`，并把**每仓门限 \(h_r\) 作为优化参数**，
与 \((s_r,S_r)\) 一起搜索。no-active 仍是 2K 维度，active 是 3K 维度：

\[
\theta=[s_1,\dots,s_K,S_1,\dots,S_K,h_1,\dots,h_K],
\qquad
h_r\in[T,I^{\max}_r].
\]

`active_transfer=false` 时 \(h\) 不参与；旧 policy JSON 没有 active 键时
默认 mode 0。

### 14.2 规则

目标库存 \(T=5\)（每仓固定，可配置）。被动履约后 \(X_r=S^{end}_r\)：

- **调出仓**：\(X_r>h_r\)，可向目标 \(T\) 靠拢，盈余 \(E_r=X_r-T\)；
- **调入仓**：\(X_m<T\)，缺口 \(D_m=T-X_m\)；
- 贪心：调出仓按 \(E_r\) 降序，调入仓按 \(D_m\) 降序；同分时调入仓优先
  \(W_{out}+\hat F^{next}\) 大者，调出仓优先小者；
  \[
  \hat F^{next}_r=\sum_i n^{out}_{r,i}\bar\Theta_{r,i,G},
  \]
  \(\bar\Theta\) 是 300 场景池平均（state-only，不用未来信息）；
- 每次调拨量（\(\alpha\) 为每仓每期总调出比例）：
  \[
  \Delta=\min\{E_r,\ D_m,\ \text{源剩余预算},\ \text{目标剩余容量}\},
  \]
  \[
  \text{目标剩余容量}=I^{\max}_m-X_m-\sum_l o^{out}_{l,m}-q^{out}_m;
  \]
  多源对一个目标时维护剩余容量/来源约束台账；
- 更新净 \(\tau\)，重复直到无盈余/缺口。

硬约束：

\[
\sum_r\tau_r=0,\qquad
0\le X_r+\tau_r\le I^{\max}_r,\qquad
\tau_r\le\sum_{m\ne r}S^{end}_m\ \text{(正净流入)}.
\]

符号：模型约定 \(\tau=taup-taum\)（正=净流入），
\[
I_{out}=S_{end}+\tau,
\qquad
\text{active cost}=\frac{atc}{2}\sum_r(\tau^+_r+\tau^-_r).
\]

时间边界：虚拟阶段 0 不调；默认最后 2 期不调
（`active_stop_offset=2`）；末期若执行必须满足 `cap`，默认关闭。
\(\tau\) 连续，不取整。

### 14.3 为什么优化每仓门限

现有 s,S mode-0 的 \(S^{end}\) 每仓差异很大（仓 3/4 有盈余，仓 1/2/5 有缺口）。
固定一个全局门限只是手工折中；把 \(h_r\) 作为优化参数，每仓可以自己选择
“何时调出、保留多少”，并与 \(S_r\) 交互：

- \(h_r=T=5\)：只要 \(X_r>5\) 就调出（激进）；
- \(h_r=I^{\max}_r\)：该仓永不调出（等价于该仓关闭主动调拨）；
- 中间值：按仓库自己的门限触发。

### 14.4 配置与运行

`ss_static/config/policy_active.toml`（完整 active 配置）：

```toml
[policy]
active_transfer = true
active_rule = "threshold_balance"
active_target = 5.0
active_threshold_init = 5.0   # h 初值；h 本身是优化参数
active_threshold_min = 5.0    # h 下界 = T
active_fraction = 1.0        # alpha
active_stop_offset = 2
active_priority = "deficit_risk"
active_forecast = "pool_mean"
```

```bash
COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_calibrate.py \
  --policy-config ss_static/config/policy_active.toml \
  --algo coordinate --tag ss_static_T60_204105_active

COMPARE_CONFIG=config/compare_T60_base204105.toml \
  "$P" -B ss_static/run_evaluate.py \
  --policy results/ss_static/ss_static_T60_204105_active_calib_<时间戳>/policy.json \
  --policy-config ss_static/config/policy_active.toml \
  --tag ss_static_T60_204105_active
```

`--policy-config` 会把 active TOML 复制进结果目录，避免“active policy +
no-active config”矛盾。评估报告从 policy JSON 读取 active options；
`summary.md` 的 mode_tau 动态显示 1/0，参数行列出每仓 \(h\)。

### 14.5 归档/消融

- no-active arm 与 active arm 都正式报告；不预设主次；
- 同一个 3000 条测试路径上做 active vs no-active 配对；
- 可选 frozen active：固定 no-active 的 \(s,S\)、只优化 \(h\)，
  隔离“机制 + h 优化”与“\(s,S\) 重新优化”的混杂；
- 门限初值/目标/α 的敏感性只在训练/验证路径上做，不用测试集调参；
- active 可能增加采购（调出后又补货），必须分五项成本报告；
- 末期默认停调，避免纯成本。
