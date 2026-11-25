# Rosetta Flex ddG 端到端异构加速设计文档（Ascend 950PR）

# 需求背景（required）

## 需求来源

本设计文档对应 CANN 社区 2026 年 11 月任务 **「生命科学 flexddg ΔΔG 计算加速（24 核 CPU + 950）」**（平台 `taskId`: `4373d1139dfe4ba7b448a8ef5cc3596b`），目标基于 **24 核 CPU（`taskset -c 0-23` 严格限核）+ 单卡昇腾 Ascend 950PR**，在不改变官方协议语义与 `talaris2014` 力场精度的前提下，实现 Rosetta Flex ddG 端到端异构加速，使 `BD-623`（PDB `7E7Y`）全部 50 个抗原单点突变（50×35 条轨迹 × 35,000 步 backrub）端到端墙钟耗时达到 48 核 CPU 开源基线（`55,653.4 s`）的 **1/10 及以内**（即 `<= 5,565.3 s`，加速比 `>= 10.0×`），并满足大效应/小效应分层统计精度验收标准。

## 背景介绍

### Rosetta Flex ddG 协议与开源基线现状

Rosetta Flex ddG（Barlow et al., *J. Phys. Chem. B* 2018）是预测蛋白质-蛋白质结合界面突变自由能变化（ΔΔG）的主流蒙特卡洛系综协议，官方协议定义于 `Kortemme-Lab/flex_ddG_tutorial` 的 `ddG-backrub.xml`，底层基于 `RosettaCommons/rosetta`（`2026.29` / `80a0635615`）实现。

随任务测试包（`test_cases/bd623_reference.csv`）提供了 2×Xeon Platinum 8360Y（48 进程满载）实测基线数据，共覆盖 `BD-623.pdb`（420 残基：H 链 124 残基、L 链 107 残基、A 链 189 残基；3,217 个 `ATOM`，0 个 `HETATM`）抗原 `A` 链上的 50 个单点突变（分布于 21 个抗原界面位点，含 15 个大效应样本 `|ddG_mean_ref| > 1.0 REU`、35 个小效应样本 `|ddG_mean_ref| <= 1.0 REU`，以及 4 个几何敏感的脯氨酸突变 `N440P / N481P / N501P / F486P`）：

| 基线统计指标 | 48 核 CPU 实测数值 | 说明 |
| --- | ---: | --- |
| 总任务规模 | 50 突变 × 35 轨迹 × 35,000 步 | 共 1,750 条轨迹、61,250,000 步 Backrub MC |
| 50 突变累计 CPU 核时 | **2,671,363.3 s（742.05 CPU·h）** | 48 进程并行跑完 1,750 条轨迹 |
| 单突变累计核时均值 / 范围 | 53,427 s（14.84 CPU·h）/ 50,618 ~ 60,127 s | 单轨迹平均耗时约 1,526 s（~25.4 分钟） |
| **48 核满载端到端墙钟基线** | **55,653.4 s（15.46 h / 927.6 min）** | 均摊 ~18.55 分钟/突变 |

### 计算热点与性能瓶颈剖析

结合 `ddG-backrub.xml` 与 `RosettaCommons/rosetta` 源码（`BackrubProtocol.cc`、`BackrubMover.cc`、`SidechainMover.cc`、`MultiCoolAnnealer.cc`、`ddGMover.cc`、`InterfaceDdGMover.cc`、`AtomTreeMinimizer.cc`、`Minimizer.cc`）剖析，开源 CPU 实现存在四大瓶颈：

1. **MC 采样海量重复与标量开销（主热点，约占单轨迹 75%–85%）**：
   - 每条轨迹执行 `35,000` 步蒙特卡洛（`mc_kt = 1.2`，75% `BackrubMover` 主链旋转 + 25% `SidechainMover` 局部侧链 χ 采样）；每次仅扰动 `neighbor_shell`（`BD-623` 中 50 突变对应 `neighbor_shell` 规模仅 **9–29 个残基，均值 18.7 残基**）内的局部片段，但 Rosetta CPU 版每次需更新 `AtomTree` 坐标、维护 `EnergyGraph`/`MinimizationGraph` 并逐原子对查表计算 `Etable`（`fa_atr`/`fa_rep`/`fa_sol`）、`Coulomb`（`fa_elec`）、`HBond` 及内能项（`mm_bend`、`fa_dun`、`rama`、`p_aa_pp`），标量分支多且无法跨轨迹向量化。
2. **WT/MUT 后处理最小化与重复全结构打分（热点，约占 15%–20%）**：
   - 每条轨迹终点需执行 WT `repack`（`UseMultiCoolAnnealer states=6`）+ 全蛋白约束 `L-BFGS` 最小化，随后恢复 Backrub 终点构象再执行 MUT `mutate` packing + 全蛋白约束 `L-BFGS` 最小化。单次最小化在 `420` 残基上做 `lbfgs_armijo_nonmonotone`（`max_iter=5000`, `tol=1e-6`），其中超过 90% 的时间花在非键能量与笛卡尔力梯度计算上。
3. **1,750 次初始约束最小化冗余**：
   - `ddG-backrub.xml` 的 `PROTOCOLS` 段在 `backrub` 前对野生型 `BD-623.pdb` 施加 `CA-CA < 9.0 Å` 谐振子距离约束（`coord_dev=0.5`）并执行一次全蛋白最小化；该步骤在原版脚本中随每条轨迹重复执行 1,750 次，而其输入仅依赖初始 WT 结构。
4. **Python 多进程与 SQLite I/O 开销**：
   - 1,750 个独立 `ddG.db3` / `struct.db3` 写入以及 PyRosetta 进程间冗余 Pose 克隆、Resfile 文本解析产生额外 I/O 与内存开销。

---

# 需求分析（required）

## 需求描述

基于 **24 核 CPU（`taskset -c 0-23`）+ 单卡 Ascend 950PR（CANN >= 9.0.0）**，设计并实现 Rosetta Flex ddG 端到端异构加速模块 `rosetta_flex_ddg`：

1. 提供统一端到端入口 `run_flexddg_npu.py`，参数规范与异常语义严格对齐任务书 §2.3–§2.4；
2. 完整等价覆盖 `ddG-backrub.xml` 全流程，严守**三条不可修改红线**：
   - 力场保持 **`talaris2014`**（等价 `-corrections::restore_talaris_behavior true`），不替换为 REF2015 或删减能量项；
   - 采样与收敛预算不调低：保持 `nstruct = 35`、`backrub_trials = 35000`、8 Å 邻域（`Neighborhood distance="8.0"` + `PrimarySequenceNeighborhood lower=1 upper=1`）、`max_minimization_iter = 5000`、`abs_score_convergence_thresh = 1.0`；
   - 不删减 `ddG-backrub.xml` 任一核心阶段（初始约束最小化、Backrub/Sidechain MC、WT repack + 最小化、MUT mutate + 最小化、四态接口打分与 ZEMu GAM 重加权）。
3. 端到端输出 22 列 `results.csv` 与 `run_summary.json`，通过 `test_cases/verify_bd623.py` 全部格式、精度与性能校验。

## 需求拆解

| 需求项 | 验收标准与约束 |
| --- | --- |
| **输入校验与负向边界** | 校验 `--selected`（6 列、`wt_aa` 与 PDB 残基精确一致）、`--pdb-dir`、`--xml`、`--move-chain`、`--nstruct`（正式 35，`<=0` 报错）、`--trials`（正式 35000，`<=0` 报错）、`--cpu-cores`（`1..24`，`>24` 或 `<=0` 报错）、`--device-id`、`--seed`；任意异常返回非零退出码 |
| **输出格式与自洽性** | `[50, 22]` 标准 CSV，无 NaN/Inf，`nstruct_done==35`，`backrub_trials==35000`，`ddG_all(REU)` 含 35 个 `;` 分隔有限值，`abs(ddG_residual(REU)) <= 0.01`；`run_summary.json` 含 `wall_time_s`, `mutations_completed`, `cpu_cores_limit`, `npu_device`, `cann_version`, `seed`, `git_commit`, `command` |
| **精度验收（BD-623）** | 大效应组（15 个，`abs(ddG_mean_ref) > 1.0`）中位相对误差 `< 5%`；小效应组（35 个，`abs(ddG_mean_ref) <= 1.0`）全部样本 `abs(ddG_mean - ddG_mean_ref) < 1.0 * std_ref`（即 `max(sigma) < 1.0`） |
| **端到端性能验收** | `taskset -c 0-23` 绑定 24 核 CPU + 单卡 Ascend 950PR，`wall_time_s <= 5,565.3 s`（加速比 `>= 10.0×`）；计时覆盖从命令启动到 `results.csv` 落盘的完整墙钟时间 |
| **内存检查** | 任务书 §3.4 明确标注“不涉及” |

---

# 详细设计（required）

## 算法与参考

### 1. 协议五阶段数学与物理建模

严格对齐 `ddG-backrub.xml` 与 Rosetta 源码实现：

1. **Stage 0：邻域构造、CA 约束与初始最小化**
   - **邻域选择**：对突变位点 $r_{\text{mut}}$，取其代表原子（`GLY` 为 `CA`，其余为 `CB`）8.0 Å 球内残基集合 `bubble`，再沿同链主链序列前后扩展 1 个残基（`PrimarySequenceNeighborhood lower=1 upper=1`）得到 `neighbor_shell`（记 $|S|$，在 BD-623 中 $|S| \in [9, 29]$）。
   - **CA 距离约束**：对所有蛋白质残基对 $(i, j)$ 满足 $j < i$ 且初始距离 $d_{ij}^0 = \|\mathbf{x}_{i,\text{CA}}^0 - \mathbf{x}_{j,\text{CA}}^0\|_2 < 9.0\ \text{Å}$，添加谐振子约束 $E_{\text{cst}}(d_{ij}) = \left(\dfrac{d_{ij} - d_{ij}^0}{\sigma}\right)^2$，其中 $\sigma = \text{coord\_dev} = 0.5\ \text{Å}$，权重 $w_{\text{cst}} = 1.0$。
   - **初始最小化**：在 `fa_talaris2014_cst`（`talaris2014` + `atom_pair_constraint=1.0`, `fa_max_dis=9.0`）下以 `lbfgs_armijo_nonmonotone`（`M=10`, `PAST=3`, `tol=1e-6`, `max_iter=5000`）优化全蛋白主链（$\phi, \psi, \omega$）与侧链（$\chi$）二面角直至外循环 $|\Delta E| \le 1.0$。

2. **Stage 1：Backrub + Sidechain MC 采样（$nstruct=35,\ ntrials=35000,\ kT=1.2$）**
   - 每步随机以概率 $p_{\text{sc}} = 0.25$ 执行 `SidechainMover`（在 `neighbor_shell` 非 Pro 残基中随机选一位点，以 `prob_uniform=0.1` 均匀采样 rotamer、90% 基于 Dunbrack 库局部采样 $\chi$，按 `dunbrack_accept` 预筛后更新坐标）；以概率 $1 - p_{\text{sc}} = 0.75$ 执行 `BackrubMover`：
     - 从 `neighbor_shell` 连续残基片段的 `CA` 原子中构建所有长度为 `3..34` 主链原子的合法段（排除 Proline 的 `N/CA` 端点）；
     - 随机择一段 `(start_CA, end_CA)`，依据 Betancourt (2005) 解析旋转方程计算在两端键角热力学容许区间（$\Delta E_{\text{bend}} \le -0.6\ln 0.05$）与最大角位移 `max_angle_disp`（4 原子段 40°、7 原子段 20°、斜率 `-1/3°/atom`）交集内的旋转角 $\tau$，解析更新段内原子坐标与连接键角/二面角，并按 `branchopt`（`MMBondAngle`）重调支链原子（`C`, `O`, `H`, `CB`）几何位置；
     - 在 `talaris2014 + mm_bend` 下仅对受扰动残基相关局部原子对计算 $\Delta E = E_{\text{new}} - E_{\text{old}}$，按 Metropolis 准则 $\min(1, \exp(-\Delta E / 1.2))$ 接收或回滚（`recover_low=false`）。

3. **Stage 2：WT / MUT 后处理（MultiCool Packer + 约束 L-BFGS）**
   - 对每条轨迹终点构象 `backrubpdb`：
     - **WT 分支**：在 `neighbor_shell` 上仅允许 `RestrictToRepacking`（其余残基 `PreventRepacking`，开启 `-ex1 -ex2 -extrachi_cutoff 0`）构建 Dunbrack 2010 旋转异构体库与 `OneBody + TwoBody Interaction Graph`，运行 `MultiCoolAnnealer`（`states=6`：第一阶段 $T=10 \to 0.2$ 交替恒温采样与 quench，保留能量最低 6 个网络状态；第二阶段从每个历史状态 $T=0.25 \to 0.05$ 再冷却并 quench，取全局最优构象），随后施加 `CA < 9.0 Å` 约束并运行 `lbfgs_armijo_nonmonotone` 收敛至 `abs_score_convergence_threshold = 1.0`，记为 `wt_bound_pose`。
     - **MUT 分支**：从 `backrubpdb` 恢复，将突变位点设为 `PIKAA mut_aa`（其余 `neighbor_shell` 残基仅 repack），同样运行 6-state `MultiCoolAnnealer` 与约束 `L-BFGS` 最小化，得到 `mut_bound_pose`。

4. **Stage 3：四态接口打分与 ZEMu GAM 重加权**
   - 通过刚体平移（`chainstomove = A` 相对抗体 `H/L` 链沿 X 轴平移 `1000.0 Å`）构造解离态 `unbound_wt` 与 `unbound_mut`。
   - 对四态分别按 `fa_talaris2014` 打分，分子内项精确抵消，保留 7 个分子间分量与原始总和：
     $$\Delta\Delta G_{\text{term}} = (E_{\text{bound\_mut},\text{term}} + E_{\text{unbound\_wt},\text{term}}) - (E_{\text{bound\_wt},\text{term}} + E_{\text{unbound\_mut},\text{term}})$$
     $$\Delta\Delta G_{\text{gam}} = \sum_{t \in \mathcal{T}_7} \left(-e^{a_t} + \frac{2e^{a_t}}{1 + \exp(-\Delta\Delta G_t \cdot e^{b_t})}\right)$$
     其中 $\mathcal{T}_7 = \{\text{fa\_atr}, \text{fa\_rep}, \text{fa\_sol}, \text{fa\_elec}, \text{hbond\_sc}, \text{hbond\_bb\_sc}, \text{hbond\_lr\_bb}\}$ 系数 $(a_t, b_t)$ 严格采用官方 `analyze_flex_ddG.py` / `flexddg_lib.py` 标定参数。

---

## 整体异构架构设计

### 1. CPU（24 核）+ Ascend 950PR（单卡）异构分工

采用 **“Host 24 核控制面/拓扑构建 + NPU 常驻张量计算面 + 双缓冲异步流水”** 架构：

```text
+-----------------------------------------------------------------------------+
| Host 24-Core CPU (taskset -c 0-23, --cpu-cores 24)                          |
|  1. 输入校验 / PDB & XML 解析 / wt_aa 校验 / 50 突变分组（21 位点共享拓扑） |
|  2. talaris2014 力场参数表导出与序列化（Etable / Coulomb / HBond / Dunbrack）|
|  3. 确定性 Stage 0 WT 初始最小化（跨 50 突变验证后共享 WT 起点）             |
|  4. 24 核线程池调度：Batch 组装、L-BFGS 双循环方向更新、结果归约与 CSV 落盘   |
+-------------------------------+-----------------------------+---------------+
                                | H2D (拓扑/邻域/Batch坐标)   ^ D2H (能量/梯度/坐标)
                                v                             |
+-----------------------------------------------------------------------------+
| Ascend 950PR (Device 0, HBM 常驻参数表 + 双缓冲 Stream 流水线)               |
|  [Stream A / B 并发]                                                         |
|   Kernel 1: Backrub/Sidechain MC 驻留内核（35 轨迹批处理，局部 ΔE + PRNG）   |
|   Kernel 2: Rotamer 1-Body/2-Body IG 构建 + MultiCool 6-State 并行退火       |
|   Kernel 3: talaris2014 批量打分与梯度内核（Etable + Elec + HBond + Cst）    |
|   Kernel 4: 内坐标 AtomTree FK/Grad 变换与 L-BFGS 线搜索批量评估             |
+-----------------------------------------------------------------------------+
```

### 2. 跨突变共享与批次调度策略（24 核均衡）

- **21 位点邻域共享优化**：`BD-623` 50 个突变分布在 21 个抗原位点上，同一位点的多个突变（如 `F486 -> G/H/I/N/P/S` 共 6 个突变）共享完全相同的 `neighbor_shell`、Backrub 段列表与初始 WT 约束最小化起点。虽然每个突变仍严格独立执行 `nstruct=35` 条独立随机轨迹（`seed` 按突变独立递增，严禁跨突变共用同一批采样轨迹以保证统计独立性），但**静态邻域索引、CA 约束边表、Dunbrack 查表掩码与初始最小化 Pose 可零开销复用**，节省大量前处理时间。
- **双缓冲多 Stream 流水线**：24 核 CPU 线程划分为 3 组职责：
  - `Worker Pool`（16 线程）：并行处理 Batch 数据打包、Host 侧 L-BFGS $m=10$ 历史双循环方向计算、四态结果校验与写盘；
  - `NPU Stream Dispatchers`（4 线程）：绑定 4 条独立 ACL Stream 交替下发 NPU 核函数（每批 1 个或 2 个突变的 `35` 条轨迹），重叠 H2D/Compute/D2H；
  - `IO/Verify`（主线程 + 异步落盘）：实时统计每突变耗时与残差校验 `|ddG_residual| <= 0.01`。

---

## Host 与 Kernel 设计

### 1. NPU 内存布局与常驻数据结构

所有高频参数在启动时一次性 H2D 常驻于 Ascend 950PR HBM（总占用 `< 256 MB`）：

- **力场只读查表区（`talaris2014`）**：
  - `etable_table`: `[N_atype, N_atype, N_bins, 4]`（`fa_atr`, `fa_rep`, `fa_sol` 分段线性/三次插值系数，6.0 Å 截断，`bins_per_A2 = 20` -> 720 bins）；
  - `hbond_poly`: HBond 多项式与 `FadeInterval` 分段表（`talaris2014` sp2/sp3 几何参数与 `hb_fade_energy`）；
  - `dunbrack_rot_db`: 20 种标准氨基酸的 $(\phi, \psi)$ 网格与 `-ex1 -ex2` 旋转异构体均值/标准差表；
  - `rama_table` / `paa_pp_table` / `mm_bend_params`。
- **轨迹状态张量（SOA 排布，按 `B=35` 对齐 32B）**：
  - `coords`: `[B, N_atoms, 3]`（FP32/FP64 混合：L-BFGS 阶段使用 FP64 或双精度累计以防线搜索停滞，MC 阶段坐标与能量差使用 FP32/FP64 校准）；
  - `dofs`: `[B, N_dof]`（主链 $\phi,\psi,\omega,\theta$ 与侧链 $\chi$）；
  - `nbr_list`: 6.0 Å / 9.0 Å 压缩邻接表 `Tile-Compact Verlet`（`[B, N_res, MAX_NBR]`），仅在原子位移超过 `skin = 1.0 Å` 时重建。

### 2. 核心 NPU Kernel 设计（Ascend C / Triton-Ascend）

#### Kernel 1：NPU 驻留式 Backrub + Sidechain MC 内核（`backrub_mc_kernel`）
- **并行粒度**：每 AI Core 负责 1 条或一组轨迹（`B = 35` 或 `70` 并行，按 950PR Core 数量动态分核，核内 Vector 单元并行计算局部移动原子对与邻域环境的非键相互作用）。
- **增量能量评估 $\Delta E$**：
  - `BackrubMover` 每次仅旋转 `segment` 内 $K$ 个主链及附属侧链原子（典型 6–30 个原子）；`SidechainMover` 每次仅旋转 1 个残基的侧链原子（$\le 14$ 个原子）。
  - 计算旧构象下「移动原子 $\leftrightarrow$ 移动原子 + 静态 6 Å 邻居原子」的 `fa_atr`, `fa_rep`, `fa_sol`, `fa_elec`, `hbond_*` 及局部 `mm_bend`/`fa_dun`/`rama`/`p_aa_pp` 能量 $E_{\text{old}}^{\text{local}}$；在 UB 内完成旋转矩阵应用并计算新构象能量 $E_{\text{new}}^{\text{local}}$，精确得到 $\Delta E = E_{\text{new}}^{\text{local}} - E_{\text{old}}^{\text{local}}$，由每轨迹独立 Philox/SplitMix64 PRNG 完成 Metropolis 判据。
  - **关键点**：`35,000` 步全在 NPU 驻留循环内执行（仅在 `stride=35000` 时写回一次终点坐标），**完全消除 35,000 次 Host-Device 同步与内核发射开销**！

#### Kernel 2：Two-Body Interaction Graph 构建与 MultiCool 6-State 并行退火（`packer_ig_multicool_kernel`）
- 对 `neighbor_shell`（平均 19 个可变残基，`-ex1 -ex2` 下每残基平均 ~40–180 个 rotamer，合计约 `800–2500` 个 rotamer）：
  - **1-Body & 2-Body 预计算**：利用 Cube/Vector 并行生成所有可变残基 rotamer 的笛卡尔坐标，批量评估 rotamer 与静态骨架的 1-Body 能量 $E_1(i, r)$ 及相邻可变残基对的 2-Body 能量 $E_2(i, r_1; j, r_2)$，存入片上/HBM 对称稀疏块表。
  - **6-State MultiCool 退火**：严格复现 `MultiCoolAnnealer.cc` 逻辑：外循环 $T=10 \to 0.2$ 交替恒温采样与 Quench，维护 `top_to_keep = 6` 的去重最低能量网络状态与 `hamming` 过滤，第二轮从 `top_6` 出发 $T=0.25 \to 0.05$ 精炼与 Quench。单条轨迹的退火在 NPU 单核/多核上只需毫秒级完成。

#### Kernel 3：talaris2014 能量与梯度联合算子（`talaris2014_energy_grad_kernel`）
- 服务于全蛋白约束 `L-BFGS` 最小化与四态接口打分：
  - 将 `AtomPair`（`fa_atr`, `fa_rep`, `fa_sol`, `fa_intra_rep`, `fa_elec`）、`HBond`（`hbond_sr_bb`, `hbond_lr_bb`, `hbond_bb_sc`, `hbond_sc` 及导数）以及 `AtomPairConstraint`（`CA-CA < 9.0 Å`）融合为两级流水算子：
    1. **Pairwise Force Kernel**：按 32 原子 Tile 分块遍历邻接表，在 UB 中同时输出标量能量分量与各原子笛卡尔受力 $\mathbf{F}_a = -\nabla_{\mathbf{x}_a} E$（以及用于链式法则的力矩项 $\mathbf{x}_a \times \mathbf{F}_a$）；
    2. **Internal-Coord & L-BFGS Batch Step**：将笛卡尔力与力矩沿蛋白质 `AtomTree` 投影到主链/侧链二面角梯度 $g_k = \frac{\partial E}{\partial \theta_k}$（结合 `rama`, `omega`, `fa_dun`, `p_aa_pp`, `pro_close`, `dslf_fa13`, `yhh_planarity` 解析导数），按 Batch $B=35$ 并行推进 Armijo 非单调线搜索（`PAST=3`）！

### 3. `talaris2014` 数值一致性保障机制

1. **离线参数提取与单点对拍门禁**：利用开源 Rosetta (`2026.29` / `talaris2014`) 在 `BD-623.pdb` 上导出全套权重、参考能量、Etable 样条表、Dunbrack 旋转异构体表与 HBond 参数；构造单残基/双残基/全复合物单点打分与梯度对拍单测，要求 NPU 算子相对 Rosetta 标杆的各能量分项绝对误差 `< 1e-4 REU`、梯度相对误差 `< 1e-4`。
2. **四态自洽性约束**：对每条轨迹的 `bound_wt`, `unbound_wt`, `bound_mut`, `unbound_mut` 四态采用同一套打分内核与精度规范，保证所有单链内能项在双差时精确抵消，`|ddG_residual(REU)| <= 0.01`（目标 `~0.000`）。
3. **几何敏感位点专项校验**：重点对 `N440P / N481P / N501P / F486P` 四个脯氨酸突变（涉及 `pro_close` 环闭合势及 Backrub 段 `N/CA` 排除规则）与 Rosetta 逐项对拍，避免 Proline 引入的系统偏差。

---

## 硬件支持与兼容性

| 项目 | 规格与要求 |
| --- | --- |
| 目标 NPU | **Ascend 950PR（单卡）**（实测型号 `Ascend950PR_9579` / `A310-50-C00MM304A1`：`28` Cube Core / `56` Vector Core，Vector UB `253,952 B`，L2 `128 MiB`，HBM `128 GiB`） |
| 目标 CPU | 24 核（实测宿主机 `2 × AMD EPYC 9355 32-Core`，运行时严格绑定 `taskset -c 0-23`，`--cpu-cores 24`） |
| 软件栈 | `openEuler 24.03 (LTS-SP4)`、`CANN 9.2.1`（Driver `25.7.rc1.10`，Firmware `9.0.0.110.220`）、Python `3.12.16`、`g++ 12.3.1`、`bisheng clang 15.0.5`、`cmake 3.31.12` |
| 兼容性与隔离 | 不修改系统级 CANN/驱动；本模块独立封装于 `rosetta_flex_ddg/`，远端工作目录严格隔离于 `/workspace/flexddg-950/` |

---

# 可维可测分析（required）

## 自测与性能验证方案

1. **分层单元测试（纯功能与单点精度）**：
   - **输入校验与负向边界测试**：缺失 PDB、缺失 CSV 必选列、`wt_aa` 与 PDB 残基不匹配、`--cpu-cores > 24` 或 `<= 0`、`--nstruct <= 0`、`--trials <= 0`、`--move-chain` 不存在，均须明确报错并返回非零退出码；
   - **能量与梯度单点对拍**：与开源 Rosetta (`talaris2014`) 逐项比对 `fa_atr`, `fa_rep`, `fa_sol`, `fa_elec`, `hbond_*` 及 `CA` 约束梯度；
   - **ZEMu GAM 与四态汇总单测**：对已知输出与 `bd623_reference.csv` 抽样校验 `ddG_residual` 与 GAM 公式。
2. **1/10 典型测例调优策略**：
   - 任务书全量性能集为 `N = 50` 个突变，按工作流选取 `ceil(50 / 10) = 5` 个代表性突变作为固定调优指标集：
     1. `A:Y369C`（48 核基线最慢位点，`60,126.7 s`，小效应 `ddG_ref = -0.022`）；
     2. `A:F486P`（大效应 + 脯氨酸突变 + 最大 `neighbor_shell=29`，`ddG_ref = +8.892`）；
     3. `A:G476R`（大效应带电突变 `G->R`，`ddG_ref = +4.336`）；
     4. `A:N440P`（小效应脯氨酸突变 + 小邻域 `neighbor_shell=11`，`ddG_ref = -0.196`）；
     5. `A:T478L`（负值增强结合效应，`ddG_ref = -0.609`）。
   - 每轮算子优化后先在 5 个代表突变（各 `35` 轨迹 × `35,000` 步）上验证精度与单突变均摊耗时（目标单突变平均 `<= 110 s`，50 突变推算 `<= 5,500 s`），全部指标用例达标后再跑全量 50 突变。
3. **全量端到端验收**：
   - 使用 `taskset -c 0-23 python3 run_flexddg_npu.py ...` 完整运行 50 突变并记录端到端 `wall_time_s`；
   - 运行 `python3 test_cases/verify_bd623.py --pred results.csv --ref test_cases/bd623_reference.csv --summary run_summary.json --json-out verify_report.json`，核对全部 50/50 完整性、大效应中位相对误差 `< 5%`、小效应 `max(sigma) < 1.0` 及 `wall_time_s <= 5,565.3 s`；
   - 内存检查按任务书 §3.4 标注“不涉及”。

## 兼容性与风险分析

| 风险点 | 影响 | 缓解措施 |
| --- | --- | --- |
| Backrub/Packer 与 Rosetta `talaris2014` 细微参数差异导致大/小效应偏移 | 精度不达标（尤其小效应 `max_sigma < 1.0` 对系统偏差敏感） | 全力场表直接从 Rosetta `2026.29` (`talaris2014`) 导出，并在单点打分、Proline 采样、`MultiCoolAnnealer` 与 `L-BFGS` 逐阶段与 PyRosetta 对拍 |
| `L-BFGS` 线搜索或连续内坐标变换在 NPU 侧收敛较慢 | 拖慢 Stage 2 端到端耗时 | 采用 `B=35` 跨轨迹批处理评估能量与梯度，结合 `CA` 约束初值复用，压缩单突变 L-BFGS 总耗时 |
| 24 核 CPU 绑核约束下 Host 调度与数据组装成为瓶颈 | 无法打满 Ascend 950PR | 将坐标系更新、邻接表维护与 Backrub MC 循环全部常驻 NPU，Host 仅负责按突变批次调度与最终 CSV/JSON 落盘 |

---

# 评审问题与变更（待合入前持续更新）

- 待确认问题：远端 24 核 CPU + Ascend 950PR 算力环境（CANN 9.x、PyRosetta/Rosetta 预装情况）待 OMP 下发后补充实测版本并据此确定最终算子后端（Ascend C / Triton-Ascend）细节与实测 profiling 拆分。
