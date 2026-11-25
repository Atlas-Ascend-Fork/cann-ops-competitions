# 需求背景

## 需求来源

本需求来源于 2026 年 11 月 CANN 社区任务《Rosetta Flex ddG 端到端加速开发（Ascend 950PR）》，面向**整模块端到端异构加速**场景，要求在**严格绑定 24 核 CPU（`taskset -c 0-23`）+ 单卡昇腾 Ascend 950PR** 平台上，基于 Ascend C / CANN 软件栈与 C++/Python 异构调度，实现 Rosetta Flex ddG 端到端加速模块并完成设计、开发与验证全流程。

## 背景介绍

### Rosetta Flex ddG 协议功能现状分析

Rosetta Flex ddG 是蛋白质-蛋白质结合界面突变自由能变化（$\Delta\Delta G$）预测的经典蒙特卡洛系综协议，对标开源基线为 `RosettaCommons/rosetta`（`2026.29` / `80a0635615`）与 `Kortemme-Lab/flex_ddG_tutorial` 官方协议 `ddG-backrub.xml`。其计算流程由四大核心阶段组成：

1. **结构准备与初始最小化**：加载全原子 PDB 与突变清单，校验野生型残基 `wt_aa` 一致性，基于 `cb8`（8 Å）邻域构建 `bubble` 及前后相邻残基 `neighbor_shell`，施加 CA 距离约束（`coord_dev=0.5`, `max_distance=9`）后执行 `fa_talaris2014_cst` 约束 L-BFGS 最小化（`lbfgs_armijo_nonmonotone`, `tolerance=1e-6`, `max_iter=5000`, `abs_score_convergence_threshold=1.0`）。
2. **Backrub / Sidechain 蒙特卡洛采样**：每突变独立运行 `nstruct = 35` 条轨迹，每条轨迹在 `neighbor_shell` 上执行 `backrub_trials = 35,000` 步采样（`mc_kt = 1.2`, `-ex1 -ex2`, `extrachi_cutoff=0`）。
3. **WT / MUT 后处理**：对每条轨迹终点构象分别执行野生型 `repack`（`UseMultiCoolAnnealer states=6`）+ 约束 L-BFGS 最小化，以及突变体 `mutate` packing + 约束 L-BFGS 最小化。
4. **四态接口打分与重加权**：按 `chainstomove = A` 计算 `bound`/`unbound` 四态能量双差、7 项分子间能量分量及 ZEMu GAM 重加权结果。

在纯 CPU 上该协议计算开销极大：以抗体-抗原复合物 `BD-623`（PDB `7E7Y`，420 残基）的 50 个抗原侧单点突变（共 1,750 条独立轨迹）为例，开源 Rosetta 在 48 核 CPU 上累计消耗 **742.05 CPU·h（2,671,363.3 s）**，满载墙钟耗时达 **55,653.4 s（15.46 小时）**。其中，大量细粒度 Backrub/Sidechain MC 步进与重复初始最小化/串行化 I/O 构成了主要性能瓶颈，非常适合通过 CPU/NPU 异构分工与大批量设备驻留计算实现数量级加速。

# 需求分析

## 需求描述

在 **24 核 CPU + 单卡 Ascend 950PR** 上实现 Rosetta Flex ddG 端到端异构加速模块，提供统一命令行入口 `run_flexddg_npu.py`（参数序列对齐基线脚本 `06b_ddG_flex_antigen.py`），在严格遵循官方 `ddG-backrub.xml` 语义与 `talaris2014` 力场精度的前提下，完成 `BD-623` 全部 50 个突变的端到端计算，输出符合规范的 22 列 `results.csv` 与 `run_summary.json`。

```bash
taskset -c 0-23 python3 run_flexddg_npu.py \
  --selected <mutations_csv> --pdb-dir <pdb_dir> --xml <protocol_xml> \
  --out <results_csv> --summary <run_summary_json> --work <work_dir> \
  --move-chain A --nstruct 35 --trials 35000 --cpu-cores 24 --device-id 0 --seed 111111
```

## 需求拆解

1. **协议语义与三条红线**：严格保持 `talaris2014` 力场（等价 `-corrections::restore_talaris_behavior true`）；保持 `nstruct=35`、`backrub_trials=35000`、8 Å `cb8` 邻域、`max_iter=5000`、`abs_score_convergence_threshold=1.0`；完整保留初始最小化、Backrub MC、WT/MUT 重排与最小化、四态打分全部阶段。
2. **输入校验与异常处理**：启动前校验 CSV/PDB/XML 存在性与完整性、位点及 `wt_aa` 匹配性、`--move-chain` 合法性以及 `--nstruct`/`--trials`/`--cpu-cores`/`--device-id`/`--seed`/`--work` 边界，非法场景立即返回非零退出码。
3. **统计精度达标**：满足大效应组中位相对误差 `< 5%`、小效应组全部样本 `< 1.0 * std_ref`、以及 50/50 完整性与 `abs(ddG_residual) <= 0.01` 自洽要求。
4. **端到端性能达标**：24 核 CPU + 单卡 Ascend 950PR 完整墙钟耗时 `wall_time_s <= 5,565.3 s`（相对 48 核 CPU 基线加速比 `>= 10.0×`）。
5. **工程可复现性**：支持 `--seed` 控制随机序列，不入库大体积 `.db3` 临时轨迹，遵循 `rosetta_flex_ddg/{README.md,build.sh,env/,pipeline/,kernels/,tests/}` 仓内目录规范。

# 详细设计

## 算子分析

### 数学公式

1. **Backrub / Sidechain 蒙特卡洛 Metropolis-Hastings 判据**：在每步构象提议 $x \to x'$ 中，在 `talaris2014`（加 `mm_bend` 键角弯曲势）下计算局域增量能量 $\Delta E = E(x') - E(x)$，以温度参数 $k_B T = 1.2\ \text{REU}$ 判定接受概率：
   $$P_{\text{accept}}(x \to x') = \min\left(1,\ \exp\left(-\frac{\Delta E}{k_B T}\right)\right)$$
2. **四态结合自由能双差**：对单条轨迹终点分别完成 WT 与 MUT 的重排及约束 L-BFGS 最小化后，按 `chainstomove = A` 分离复合物得到 `bound` 与 `unbound` 态，以 `fa_talaris2014` 计算双差：
   $$\Delta\Delta G = \left(E_{\text{bound}}^{\text{mut}} + E_{\text{unbound}}^{\text{wt}}\right) - \left(E_{\text{bound}}^{\text{wt}} + E_{\text{unbound}}^{\text{mut}}\right)$$
3. **分子间分项与自洽残差**：记 7 项分子间能量分量集合 $\mathcal{T} = \{\text{fa\_atr}, \text{fa\_rep}, \text{fa\_sol}, \text{fa\_elec}, \text{hbond\_sc}, \text{hbond\_bb\_sc}, \text{hbond\_lr\_bb}\}$，对 35 条轨迹求算术均值 $\overline{\Delta\Delta G}$、样本标准差（`ddof=1`）及分项均值 $\overline{\Delta\Delta G}_t$，校验自洽残差：
   $$\text{ddG\_residual} = \overline{\Delta\Delta G} - \sum_{t \in \mathcal{T}} \overline{\Delta\Delta G}_t,\qquad \lvert \text{ddG\_residual} \rvert \le 0.01\ \text{REU}$$
4. **ZEMu GAM 重加权公式**：对每条轨迹的 7 项分子间双差应用广义加性模型非线性重加权并统计 35 轨迹均值与标准差：
   $$\text{gam}(x, t) = -e^{a_t} + \frac{2\,e^{a_t}}{1 + \exp(-x \cdot e^{b_t})},\qquad \Delta\Delta G_{\text{gam}} = \sum_{t \in \mathcal{T}} \text{gam}(\Delta\Delta G_t, t)$$

### 参数原型与支持数据类型

| 参数名 | 方向/属性 | 描述 | 类型/格式 | 维度(shape) | 值域与异常行为 |
| --- | --- | --- | --- | --- | --- |
| `--selected` | 输入 | 突变清单 CSV（6 列：`pdb_id,chain,pdb_position,icode,wt_aa,mut_aa`） | file(CSV) | `[N, 6]`（`N=50`） | 缺失、空表、缺列、位点缺失或 `wt_aa` 不匹配返回非零 |
| `--pdb-dir` | 输入 | 全原子 PDB 目录（无 `HETATM`，链 `A`/`H`/`L`） | dir(PDB) | `BD-623`: 420 残基 | 目录或对应 PDB 缺失返回非零 |
| `--xml` | 输入 | 官方协议文件 `ddG-backrub.xml` | file(XML) | 单文件 | 缺失或解析失败返回非零 |
| `--move-chain` | 属性 | 解离态移开链 ID（对应 `%%chainstomove%%`） | string | 标量 | 非空且存在于 PDB（本任务 `'A'`），否则返回非零 |
| `--nstruct` | 属性 | 每突变独立轨迹数 | int32 | 标量 | 正整数（验收固定 `35`），`<= 0` 返回非零 |
| `--trials` | 属性 | 每轨迹 Backrub 采样步数 | int32 | 标量 | 正整数（验收固定 `35000`），`<= 0` 返回非零 |
| `--cpu-cores` | 属性 | 主机侧绑定 CPU 核数 | int32 | 标量 | `1 <= cores <= 24`（验收固定 `24`），越界返回非零 |
| `--device-id` | 属性 | Ascend 950PR 逻辑设备号 | int32 | 标量 | `>= 0` 有效设备号，越界或不可用返回非零 |
| `--seed` | 属性 | 随机数起始种子 | int64 | 标量 | 默认 `111111`，非法格式返回非零 |
| `--work` | 输出(中间) | 中间状态与缓存目录 | dir | - | 不可创建或不可写返回非零 |
| `--out` | 输出 | 汇总结果 `results.csv`（22 列，含 `ddG_mean`/`ddG_std`/`ddG_all`/7 分项/`ddG_residual`/`ddG_gam_*`/`nstruct_done`/`backrub_trials`/`cpu_time`） | file(CSV) | `[50, 22]` | 有限浮点，无 NaN/Inf，`|ddG_residual| <= 0.01` |
| `--summary` | 输出 | 运行摘要 `run_summary.json`（含 `wall_time_s`, `mutations_completed`, `cpu_cores_limit`, `npu_device`, `cann_version`, `seed`, `git_commit`, `command`） | file(JSON) | JSON Object | `wall_time_s > 0` 且 `mutations_completed == 50` |

设备侧内部张量数据类型：原子坐标、力场查找表（Lennard-Jones、LK 溶剂化、库仑静电、氢键多项式、Dunbrack 旋转异构体与 `mm_bend` 键角参数）采用 `float32`/`float64` 混合精度表示；残基索引、邻域表与拓扑连接表采用 `int32`/`uint16`；轨迹独立伪随机数生成器状态采用 `uint64`。

### 支持形状与规模

- 复合物结构支持常规蛋白-蛋白界面体系，本任务 `BD-623` 含 420 残基（链 `A` 为抗原，链 `H`/`L` 为抗体）；
- 局域活动区为突变位点 `cb8`（8 Å）`bubble` 及序列前后延伸 1 位构成的 `neighbor_shell`（典型覆盖数十个残基），其外围背景原子在 MC 阶段保持刚性静止；
- 端到端整批作业规模为 `50 突变 × 35 轨迹 = 1,750` 条独立轨迹，每条轨迹执行 `35,000` 步采样（总计 `6.125 × 10^7` 步 MC 评估）。

## 算子实现

### 实现方案

总体架构遵循「**共享前端去重 + NPU 大批量设备驻留 MC 采样 + CPU/NPU 异步流水后处理**」的异构分工设计，针对原版 CPU 协议的三大瓶颈逐层突破：

```text
[Host 24-Core CPU (taskset -c 0-23)]                     [Ascend 950PR NPU (Ascend C SIMT)]
  1. PDB/CSV/XML 严格校验 + cb8 neighbor_shell 构建
  2. 共享 WT 初始约束 L-BFGS 最小化（50×35 次去重为 1 次）
  3. 导出 talaris2014 力场表与 -ex1/-ex2 rotamer 库  ──► 4. 1,750 条轨迹批量驻留 GM/片上缓冲
                                                            5. SIMT 并行 Backrub/Sidechain 提议
                                                            6. 邻域截断增量 talaris2014 + mm_bend 评估
                                                            7. Metropolis-Hastings 步进 35,000 步
  8. 异步接收已完成轨迹终点构象  ◄─────────────────────────┘
  9. 24 核并行 WT repack + L-BFGS / MUT mutate + L-BFGS
 10. 四态 bound/unbound fa_talaris2014 打分 + ZEMu GAM
 11. 自洽校验并落盘 results.csv (22列) & run_summary.json
```

#### host侧设计

##### 1. 输入校验与共享初始化策略

- **前置严格校验**：在进入任何耗时计算前完成全部文件存在性、CSV 必选列、氨基酸单字母码、PDB 残基编号/插入码与 `wt_aa` 一致性、`--move-chain` 链存在性以及全部命令行数值边界校验，确保负向用例快速失败并返回非零退出码。
- **共享 WT 初始最小化去重**：在官方协议中，同一复合物对全部 50 个突变 × 35 条轨迹所执行的初始约束 L-BFGS 最小化均从相同的野生型构象与相同 CA 距离约束出发，结果严格一致。Host 侧将其合并执行一次并缓存最小化后的野生型构象与坐标，消除 1,749 次重复最小化计算，同时不改变任何轨迹进入 Backrub 阶段的初始物理状态。
- **邻域与力场表导出**：针对 50 个突变分别构造严格等价于 `Neighborhood(distance=8.0)` + `PrimarySequenceNeighborhood(lower=1, upper=1)` 的 `neighbor_shell` 残基集，将背景刚性原子与活动原子重排为紧凑索引，并一次性提取 `talaris2014` 力场参数表、氢键参数与 `-ex1 -ex2`（`extrachi_cutoff=0`）rotamer 库，经单次 H2D 拷贝驻留至 NPU 设备内存。

##### 2. 分核与 24 核负载均衡策略

- **NPU 侧轨迹级并发分核**：1,750 条轨迹（50 突变 × 35 轨迹）彼此完全独立、无跨轨迹数据依赖。Host 通过运行时接口动态查询 Ascend 950PR 可用 AI Core（Vector Core）数量，将整批轨迹按批次切分并均匀映射到多核 SIMT 执行队列；不同突变因 `neighbor_shell` 大小存在差异，按活动残基数降序（LPT）动态派发批次，避免长尾突变拖慢整体收尾。
- **24 核 CPU 约束与流水线重叠**：严格在 `taskset -c 0-23` 限核环境下运行，进程/线程池上限钉死为 `--cpu-cores`（24），杜绝子进程与底层线程库超核争抢。采用**分批异步流水线**调度：当 NPU 完成第一批轨迹的 35,000 步 MC 采样并将终点构象 D2H 回传后，24 核 CPU 立即并行消费该批轨迹的 WT/MUT 后处理（`MultiCoolAnnealer states=6` 重排、约束 L-BFGS 最小化与四态打分），与此同时 NPU 继续执行后续批次的 MC 采样，实现 CPU 与 NPU 计算高度重叠，并消除原版逐轨迹写盘 `.db3` 的 SQLite 锁竞争开销。

##### 3. 数据分块、内存优化与路径分派策略（kernel 直调，无 TilingKey）

- **设备侧驻留缓冲与零细粒度同步**：将单条轨迹完整 35,000 步循环（构象提议 → 增量打分 → MH 判据 → 坐标提交/回滚）整体下沉至 NPU 设备侧驻留执行，彻底消除每步 kernel launch 与 host-device 同步开销；静态只读力场参数表在全局只读缓冲跨轨迹共享，轨迹私有坐标与活动能量缓存按紧凑结构体排布以提升访存合并度与片上局部性。
- **路径分派策略（kernel 直调，无 TilingKey）**：本模块采用 CANN/Ascend C Kernel 直调模式，不走框架 TilingKey 二进制匹配，而是由 Host 将批次轨迹数、活动原子/残基表偏移、`neighbor_shell` 维度、步数 `trials`、温度 `mc_kt` 与随机种子偏移等标量打包为 tiling 控制块随启动参数下发；Kernel 读取控制块动态分派各轨迹元数据与尾批边界，一套通用 Kernel 覆盖全部 50 个突变（含 `N440P`/`N481P`/`N501P`/`F486P` 等脯氨酸几何受限位点）。

#### kernel侧设计

1. **初始化与数据搬入（Init / CopyIn）**：每个 SIMT 执行单元从控制块定位所属轨迹及对应突变的 `neighbor_shell` 描述符，加载初始最小化坐标、活动残基拓扑、局域截断邻接表及独立 RNG 状态（按 `(seed, mutation_id, struct_id)` 派生，确保轨迹间统计独立且同种子可复现）。
2. **局域构象提议（Backrub & Sidechain Move）**：在设备侧循环内按官方协议比例交替提议：① **Backrub 骨架微动**：在 `neighbor_shell` 内选取 2~3 残基片段，围绕首尾 $C_\alpha$ 轴执行 Rodrigues 刚体旋转并联动肽键平面微调，对脯氨酸（PRO）环状键角遵循几何约束跳过非法骨架扭转；② **Sidechain 侧链采样**：基于导出的 `-ex1 -ex2` rotamer 库在 `neighbor_shell` 内执行局域侧链旋转异构体提议。
3. **邻域截断裁剪增量能量评估（Compute）**：单步提议仅改变极少数活动残基的原子坐标，背景区域与未动活动残基间能量恒定不变。Kernel 仅对受影响原子及其非键截断半径内的邻域原子对计算 `talaris2014` 核心短程/静电/溶剂化/氢键能量变化及 `mm_bend` 键角项增量 $\Delta E$，在保持与全量打分严格数学等价的同时，将单步评估复杂度由全蛋白 $O(N^2)$ 降至局域 $O(k)$。
4. **`talaris2014` 数值一致性保障**：在能量多项式切换区间、软核排斥截断、`fa_elec` 距离介电平滑以及累加归约环节，关键能量累加与指数接受判据采用高精度累加与防溢出保护，避免 `float32` 累加漂移引入系统性能量偏移；同时，每条轨迹结束后的 WT/MUT 后处理（`repack`/`mutate`、`minimize` 及最终四态 `fa_talaris2014` 接口打分）严格遵循官方 `talaris2014` 全部能量项定义，确保 7 项分子间分量之和与总双差满足 $\lvert \text{ddG\_residual} \rvert \le 0.01\ \text{REU}$。
5. **结果搬出（CopyOut）与后处理汇合**：35,000 步完成后，Kernel 仅将每条轨迹的最终活动区原子坐标批量写回 GM，经异步流 D2H 回传至 Host，由 24 核 CPU 流水线完成 WT/MUT 后处理、四态打分、ZEMu GAM 重加权与 `results.csv`/`run_summary.json` 落盘。

## 支持硬件

| 硬件/平台组件 | 配置要求 | 涉及勾选 |
| --- | --- | --- |
| 昇腾 Ascend 950PR（单卡） | NPU 侧 Ascend C SIMT 核函数与设备驻留缓冲执行 | √ |
| Host CPU（24 核） | 验收通过 `taskset -c 0-23` 或容器 `--cpuset-cpus=0-23` 严格限核 | √ |
| 软件栈 | CANN 9.0.0 及以上，配套对应驱动、固件与 Python/C++ 运行环境 | √ |

## 算子约束限制

- **三条协议红线**：力场严格锁定 `talaris2014`（不得切换 REF2015 或删减能量项）；正式验收参数固定 `nstruct=35`、`backrub_trials=35000`、`cb8`（8 Å 邻域）、`max_iter=5000`、`abs_score_convergence_threshold=1.0`；不得跳过初始最小化、缩短采样步数或删减 WT/MUT 任一侧重排与最小化。
- **硬件与资源边界**：仅支持单卡 Ascend 950PR + 24 核 CPU，禁止超核运行或跨机并行；`--cpu-cores` 超过 24 或 `<= 0` 直接拒绝执行。
- **随机性与确定性**：Flex ddG 为蒙特卡洛随机系综协议，不要求不同运行间逐比特一致，以统计精度标准验收；支持 `--seed` 控制随机序列以便排查与复现。
- **输入输出约束**：输入 PDB 须为无 `HETATM` 的全原子蛋白结构，突变位点野生型须与 PDB 一致；中间轨迹不强制落盘 `.db3` 文件，也不入库至代码仓。

# 可维可测分析

## 精度标准/性能标准

### 1. 分层统计精度标准

以随包 48 核开源 Rosetta 实测结果 `test_cases/bd623_reference.csv`（每突变 35 轨迹）为参考真值，通过 `test_cases/verify_bd623.py` 自动验收：

| 样本分组 | 筛选条件（BD-623 样本数） | 误差指标定义 | 验收通过判据 |
| --- | --- | --- | --- |
| **大效应样本** | $\lvert \overline{\Delta\Delta G}_{\text{ref}, i} \rvert > 1.0\ \text{REU}$（15 个） | $\text{rel\_err}_i = \lvert \overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref}, i} \rvert / \lvert \overline{\Delta\Delta G}_{\text{ref}, i} \rvert$ | 中位数 $\text{median}(\text{rel\_err}) < 5\%$ |
| **小效应样本** | $\lvert \overline{\Delta\Delta G}_{\text{ref}, i} \rvert \le 1.0\ \text{REU}$（35 个） | $\sigma_i = \lvert \overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref}, i} \rvert / \text{std}_{\text{ref}, i}$ | 全部样本满足 $\max(\sigma) < 1.0$ |
| **完整性与自洽性** | 全部 50 个突变（含 4 个脯氨酸位点） | 完成度、轨迹数、步数、残差、有限值 | 50/50 完成；`nstruct_done=35`；`backrub_trials=35000`；`ddG_all` 含 35 有限值；$\lvert \text{ddG\_residual} \rvert \le 0.01$ |

验收脚本同时输出大效应逐位点相对误差、小效应逐位点 $\sigma_i$ 及全量 50 突变的 Pearson 相关系数 $r$，用于核查是否存在系统性偏移。

### 2. 48 核基线与 10× 端到端性能门槛

计时口径为从命令行启动读取输入至全部 50 突变（1,750 轨迹）完成且 `results.csv` 落盘的完整端到端墙钟时间 `wall_time_s`（含输入解析、参数导出、算子加载、采样、后处理与落盘；不含一次性环境安装与编译）：

| 评测对象与配置 | 评测 Case | 端到端墙钟时间 | 加速比要求 |
| --- | --- | --- | --- |
| 开源 Rosetta 2026.29（48 核 CPU 满载参考基线） | BD-623 全部 50 突变（50×35×35,000） | `55,653.4 s`（927.6 min / 742.05 CPU·h） | 1.0×（标杆） |
| **本方案（24 核 `taskset -c 0-23` + 单卡 Ascend 950PR）** | BD-623 全部 50 突变（50×35×35,000） | **`wall_time_s <= 5,565.3 s`**（≤ 92.8 min） | **≥ 10.0×** |

### 3. 负向边界与关键功能用例覆盖

| 用例类别 | 测试场景 | 预期行为 |
| --- | --- | --- |
| 正向全量回归 | `bd623_mutations.csv` 50 突变（含 15 大效应、35 小效应及 `N440P`/`N481P`/`N501P`/`F486P` 脯氨酸位点） | `verify_bd623.py` 完整性、精度、性能全部 PASS |
| 能量一致性单测 | 局域增量 `talaris2014` + `mm_bend` 核函数 vs 官方全量打分对照 | 单步增量误差控制在浮点容差内，无系统偏移 |
| 输入文件负向 | 缺失 PDB、缺失 CSV 或缺必选列、空突变表、缺失或损坏 XML | 启动阶段明确报错并返回非零退出码 |
| 语义与属性负向 | `wt_aa` 与 PDB 实际残基不匹配、非法 `--move-chain`、`--cpu-cores > 24` 或 `<= 0`、`--nstruct <= 0`、`--trials <= 0`、无效 `--device-id`、不可写 `--work` | 拒绝进入计算，打印明确错误原因并返回非零退出码 |

## 兼容性分析

1. **接口与脚本兼容**：命令行入口 `run_flexddg_npu.py` 参数与开源标杆 `06b_ddG_flex_antigen.py` 逐项对齐（仅将 `--nproc` 替换为 `--cpu-cores`，并新增 `--device-id`、`--seed`、`--summary`）；输出 `results.csv`（22 列）与 `run_summary.json` 严格兼容官方自动化验收脚本 `test_cases/verify_bd623.py`。
2. **目录与工程兼容**：代码工程按任务书第 5 章约定的 `rosetta_flex_ddg/{README.md,build.sh,env/,pipeline/,kernels/,tests/}` 结构组织，支持一键构建与端到端复现，不包含本地绝对路径或临时编译/轨迹产物。
3. **软件栈兼容**：基于 CANN 9.0.0 及以上版本与 Ascend C 开发，运行时动态读取 CANN 版本、NPU 设备信息与代码提交号写入 `run_summary.json`，便于不同版本环境回归复核。
