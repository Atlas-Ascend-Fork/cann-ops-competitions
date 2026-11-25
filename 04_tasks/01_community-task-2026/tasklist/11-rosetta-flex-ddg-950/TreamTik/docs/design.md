# Rosetta Flex ddG 端到端异构加速设计文档

# 需求背景（required）

## 需求来源

对应 11 月社区任务「生命科学 Rosetta Flex ddG ΔΔG 计算加速（24 核 CPU + Ascend 950PR）」。算法与协议基线对标开源 Rosetta（`RosettaCommons/rosetta`，参考版本 `2026.29` / commit `80a0635615`）与 Kortemme 实验室开源协议 `flex_ddG_tutorial`（官方 `ddG-backrub.xml`）。

## 背景介绍

Rosetta Flex ddG 是预测蛋白质-蛋白质结合界面单点突变自由能变化（ΔΔG）的主流蒙特卡洛（MC）系综协议。该协议对每个突变独立生成多条主链柔性（Backrub）与侧链重排（Sidechain）蒙特卡洛采样轨迹，并对每条轨迹终点分别执行野生型（WT）与突变体（MUT）的侧链重排、约束最小化及结合/解离四态打分。

在纯 CPU 环境下该协议计算耗时极高：根据任务书提供的 48 核 CPU 开源基线（2×Xeon Platinum 8360Y，48 进程），对抗体-抗原复合物 `BD-623`（来自 PDB `7E7Y`，420 残基）的 50 个抗原侧单点突变执行完整协议（每突变 35 条轨迹 × 35,000 步 Backrub，共 1,750 条轨迹），累计消耗 742.05 CPU·h（2,671,363.3 s），48 核满载端到端墙钟耗时为 55,653.4 s（15.46 小时）。

本设计面向**24 核 Host CPU + 单卡昇腾 Ascend 950PR** 异构平台，在严格保持官方协议语义与 `talaris2014` 力场精度的前提下，实现整模块端到端异构加速，使 50 个突变完整端到端墙钟耗时降至 48 核开源基线的 1/10 及以内（即 `wall_time_s <= 5,565.3 s`，加速比 `>= 10.0×`）。

# 需求分析（required）

## 需求描述

交付独立端到端异构加速模块 `rosetta_flex_ddg`，在主机侧严格绑定 24 核 CPU（`taskset -c 0-23` 或容器 `--cpuset-cpus=0-23`）与单卡 Ascend 950PR 上，通过统一命令行入口 `run_flexddg_npu.py` 完成输入解析与校验、结构准备、Backrub/Sidechain MC 采样、WT/MUT 后处理、四态接口打分、ZEMu GAM 重加权以及结果与摘要落盘。

## 需求拆解

### 三条不可修改红线

1. **力场保持 `talaris2014`**：等价开启 `-corrections::restore_talaris_behavior true`，不得替换为 REF2015，不得删减能量项。
2. **采样与收敛预算不得调低**：固定保持 `nstruct = 35`、`backrub_trials = 35000`、`cb8` 8.0 Å 邻域（`Neighborhood distance="8.0"`）、`max_minimization_iter = 5000`、`abs_score_convergence_thresh = 1.0`。
3. **完整等价覆盖 `ddG-backrub.xml` 核心阶段**：不得跳过初始约束 L-BFGS 最小化、缩短采样步数或删减 WT/MUT 任一侧的重排（`repack` / `mutate`）与后处理约束最小化。

### 命令行接口定义

入口脚本 `run_flexddg_npu.py` 对齐基线脚本 `test_cases/baseline/06b_ddG_flex_antigen.py` 的参数语义，调用形式如下：

```bash
taskset -c 0-23 python3 run_flexddg_npu.py \
  --selected   <mutations_csv> \
  --pdb-dir    <pdb_dir> \
  --xml        <protocol_xml> \
  --out        <results_csv> \
  --summary    <run_summary_json> \
  --work       <work_dir> \
  --move-chain <chain_id> \
  --nstruct    <35> \
  --trials     <35000> \
  --cpu-cores  <24> \
  --device-id  <0> \
  --seed       <int>
```

### 输入/输出与属性参数规范

| 参数名 | 方向/属性 | 描述 | 类型/格式 | 维度(shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- |
| `--selected` | 输入 | 待计算突变清单 CSV（UTF-8 带表头，6 列：`pdb_id,chain,pdb_position,icode,wt_aa,mut_aa`） | file(CSV) | `[N, 6]`，本任务 `N=50` | 合法单字母氨基酸码，位点须存在于对应 PDB | 文件不存在、缺列、空表、位点不存在或 PDB 残基与 `wt_aa` 不一致时拒绝计算并返回非零退出码 |
| `--pdb-dir` | 输入 | 全原子 PDB 结构目录（标准 ATOM 记录，无 `HETATM`，链 `A`=抗原、`H/L`=抗体） | dir(PDB) | `BD-623.pdb` 含 420 残基 | 有效三维原子坐标 | 目录或 `pdb_id` 对应文件缺失时立即报错并返回非零退出码 |
| `--xml` | 输入 | 官方 Flex ddG 协议 XML 文件路径 | file(XML) | 单文件 | 官方 `ddG-backrub.xml` | 文件不存在或 XML 解析失败返回非零退出码 |
| `--move-chain` | 属性 | 解离态（`unbound`）沿界面移开的链 ID，对应 `%%chainstomove%%` | string | 标量 | 非空链 ID，本任务为 `'A'` | 链 ID 在 PDB 中不存在返回非零退出码 |
| `--nstruct` | 属性 | 每突变独立轨迹数 | int | 标量 | 正整数，正式验收固定 `35` | `<= 0` 返回非零退出码；`< 35` 不通过正式验收 |
| `--trials` | 属性 | 每条轨迹的 Backrub 采样步数 | int | 标量 | 正整数，正式验收固定 `35000` | `<= 0` 返回非零退出码；`< 35000` 不通过正式验收 |
| `--cpu-cores` | 属性 | 主机侧绑定的 CPU 核数上限 | int | 标量 | 正整数，正式验收固定 `24` | `<= 0` 或 `> 24` 返回非零退出码；非 `24` 不通过正式验收 |
| `--device-id` | 属性 | 使用的 Ascend 950PR 逻辑卡号 | int | 标量 | `>= 0`（有效 NPU 设备号） | 设备不可用或越界返回非零退出码 |
| `--seed` | 属性 | 随机种子起始值 | int | 标量 | 32/64 位整数（默认 `111111`） | 非法整数格式返回非零退出码 |
| `--work` | 中间输出 | 中间状态、临时缓存或轨迹工作目录 | dir | - | 可写目录路径 | 目录不可创建或不可写返回非零退出码 |
| `--out` | 输出 | 端到端汇总结果 CSV（`results.csv`） | file(CSV) | `[N, 22]`，本任务 `[50, 22]` | 有限浮点数，无 NaN/Inf，`abs(ddG_residual) <= 0.01` | 任意突变未完成、缺列、含 NaN/Inf 或残差超限返回非零退出码 |
| `--summary` | 输出 | 端到端运行摘要 JSON（`run_summary.json`） | file(JSON) | JSON Object | 必含 `wall_time_s`, `mutations_completed`, `cpu_cores_limit`, `npu_device`, `cann_version`, `seed`, `git_commit`, `command` | 写入失败或字段缺失返回非零退出码 |

### 主输出 `results.csv` 22 列字段规范

| 列序号 | 列名 | 类型 | 含义与校验要求 |
| --- | --- | --- | --- |
| 1–6 | `pdb_id,chain,pdb_position,icode,wt_aa,mut_aa` | string/int | 与输入突变清单逐行精确匹配 |
| 7 | `ddG_mean(REU)` | float | 35 条轨迹原始 $\Delta\Delta G$ 算术均值（REU），`< 0` 表示结合增强 |
| 8 | `ddG_std(REU)` | float | 35 条轨迹原始 $\Delta\Delta G$ 样本标准差（`ddof=1`） |
| 9 | `ddG_all(REU)` | string | 35 条轨迹原始 $\Delta\Delta G$ 值，分号 `;` 分隔（35 个有限浮点数） |
| 10–16 | `ddG_fa_atr(REU)`, `ddG_fa_rep(REU)`, `ddG_fa_sol(REU)`, `ddG_fa_elec(REU)`, `ddG_hbond_sc(REU)`, `ddG_hbond_bb_sc(REU)`, `ddG_hbond_lr_bb(REU)` | float | 7 个分子间能量项双差均值 |
| 17 | `ddG_residual(REU)` | float | `ddG_mean(REU) - sum(7 个分子间项)`，须满足 `abs(ddG_residual) <= 0.01` |
| 18–19 | `ddG_gam_mean(kcal/mol)`, `ddG_gam_std(kcal/mol)` | float | ZEMu GAM 重加权均值与样本标准差（`ddof=1`） |
| 20 | `nstruct_done` | int | 完成且收敛的轨迹数，验收须严格等于 `35` |
| 21 | `backrub_trials` | int | 每条轨迹实际执行的 Backrub 步数，验收须严格等于 `35000` |
| 22 | `cpu_time(s)` | float | 该突变累计执行耗时（秒，非负有限值） |

# 详细设计（required）

## 算子与算法分析

### 协议阶段与数学公式

完整协议按官方 `ddG-backrub.xml` 定义依次包含四个阶段：

1. **结构准备、邻域构造与初始约束最小化**：
   - 加载全原子 PDB，校验 `(chain, pdb_position, icode)` 对应残基与 `wt_aa` 一致。
   - 以突变位点为中心，按 `Neighborhood distance="8.0"`（`cb8` 判据，GLY 使用 `CA` 原子、其余残基使用 `CB` 原子）构造 `bubble`，再沿同一链序列方向前后各扩展 1 个残基（`PrimarySequenceNeighborhood lower="1" upper="1"`）得到活跃残基集合 `neighbor_shell`。
   - 对全复合物施加 CA 原子对距离谐振约束（`coord_dev=0.5`, `max_distance=9.0`, `CA_only=1`, `bound_width=0.0`，`fa_talaris2014_cst` 中 `atom_pair_constraint` 权重 `1.0`、`fa_max_dis=9.0`），执行 L-BFGS 最小化（`lbfgs_armijo_nonmonotone`, `bb=1`, `chi=1`, `tolerance=1e-6`, `max_iter=5000`, `abs_score_convergence_threshold=1.0`），完成后清除约束。
2. **Backrub / Sidechain 蒙特卡洛采样**：
   - 从初始最小化构象出发，每突变独立运行 $N_{\text{struct}} = 35$ 条轨迹，每条执行 $T = 35,000$ 步蒙特卡洛迭代（`mc_kt = 1.2`，`recover_low = 0`，`trajectory_stride = 35000`）。
   - 仅以 `neighbor_shell` 作为 Backrub 支点选择集与侧链重排集合，启用 `-ex1 -ex2` 且 `extrachi_cutoff = 0`。每步按协议概率提出局部主链 Backrub 旋转（围绕 $C_\alpha^{i-1}\text{–}C_\alpha^{i+1}$ 轴及伴随肽平面微调）或侧链旋转异构体（chi）扰动，评估 `talaris2014` 局部能量差 $\Delta E = E_{\text{trial}} - E_{\text{curr}}$，依 Metropolis 准则决定接受与否：
     $$P_{\text{acc}} = \min\left(1,\; \exp\left(-\frac{\Delta E}{k_B T}\right)\right),\quad k_B T = 1.2\ \text{REU}$$
3. **WT / MUT 后处理（Repack + Mutate + 约束最小化）**：
   - 保存每条轨迹终点构象 `backrubpdb`。
   - **WT 分支**：仅对 `neighbor_shell` 执行野生型侧链重排（`RestrictToRepackingRLT`，`UseMultiCoolAnnealer states=6`，其余残基 `PreventRepackingRLT`），施加 CA 距离约束并执行相同参数配置的 L-BFGS 最小化，清除约束后保存 `wt_bound_pose`。
   - **MUT 分支**：恢复 `backrubpdb`，对突变位点执行 `PIKAA mut_aa` 突变并对 `neighbor_shell` 执行 `mutate` packing（`UseMultiCoolAnnealer states=6`），再执行相同的 CA 约束 L-BFGS 最小化并清除约束，得到 `mut_bound_pose`。
4. **四态接口双差与 ZEMu GAM 重加权**：
   - 对每条轨迹 $k \in \{1,\dots,35\}$，在 `fa_talaris2014` 下评估结合态（`bound_wt`, `bound_mut`）以及沿界面移开 `--move-chain`（链 `A`）后的刚体解离态（`unbound_wt`, `unbound_mut`）能量，计算四态双差：
     $$\Delta\Delta G^{(k)} = \left(E_{\text{bound,mut}}^{(k)} + E_{\text{unbound,wt}}^{(k)}\right) - \left(E_{\text{bound,wt}}^{(k)} + E_{\text{unbound,mut}}^{(k)}\right)$$
   - 因刚体解离不改变单链内部构象，仅依赖单链自身的能量项（如 `fa_dun`, `rama`, `p_aa_pp`, `omega`, `pro_close`, `fa_intra_rep`, `hbond_sr_bb`, `ref` 等）在四态双差中精确抵消，非零分子间能量项集合为：
     $$\mathcal{T}_7 = \{\text{fa\_atr},\; \text{fa\_rep},\; \text{fa\_sol},\; \text{fa\_elec},\; \text{hbond\_sc},\; \text{hbond\_bb\_sc},\; \text{hbond\_lr\_bb}\}$$
   - 残差自洽性校验：
     $$\text{ddG\_residual} = \overline{\Delta\Delta G} - \sum_{t \in \mathcal{T}_7} \overline{\Delta\Delta G}_t,\quad \lvert\text{ddG\_residual}\rvert \le 0.01$$
   - ZEMu GAM 重加权（逐项过缩放 sigmoid 后求和，单位 kcal/mol）：
     $$\text{gam}(x, t) = -\exp(a_t) + \frac{2\exp(a_t)}{1 + \exp\left(-x\exp(b_t)\right)},\qquad \Delta\Delta G_{\text{gam}}^{(k)} = \sum_{t \in \mathcal{T}_7} \text{gam}\!\left(\Delta\Delta G_t^{(k)},\; t\right)$$
     官方标定参数 $(a_t, b_t)$ 如下表（与 `flexddg_lib.py` 一致）：

| 能量项 $t$ | $a_t$ | $b_t$ |
| --- | ---: | ---: |
| `fa_atr` | `2.313` | `-1.649` |
| `fa_rep` | `1.659` | `-0.836` |
| `fa_sol` | `6.940` | `-6.722` |
| `fa_elec` | `0.697` | `-0.122` |
| `hbond_sc` | `1.902` | `-1.999` |
| `hbond_bb_sc` | `0.063` | `0.452` |
| `hbond_lr_bb` | `2.738` | `-1.179` |

   - 汇总统计：对每突变 $i$ 的 35 条轨迹输出均值 $\overline{\Delta\Delta G}_i = \frac{1}{35}\sum_{k=1}^{35} \Delta\Delta G_i^{(k)}$ 与样本标准差 $\text{std}_i = \sqrt{\frac{1}{34}\sum_{k=1}^{35}(\Delta\Delta G_i^{(k)} - \overline{\Delta\Delta G}_i)^2}$（`ddG_gam` 同理）。

### 热点剖析与结构局部性特征

对评测对象 `BD-623.pdb`（PDB `7E7Y`）与 `bd623_mutations.csv` 进行静态拓扑与计算量剖析，关键特征如下：

- **复合物规模与拓扑**：总计 420 个残基（重链 `H` 124 残基、轻链 `L` 107 残基、抗原链 `A` 189 残基），3,217 个重原子（加氢全原子约 6.3k 原子），包含 6 对天然二硫键（`H22-H95`、`L22-L90`、`A336-A361`、`A379-A432`、`A391-A525`、`A480-A488`，其中 `A480-A488` 位于界面热点环区）。
- **任务并发度与位点共享**：50 个抗原侧突变分布在 21 个独立位点上，对应 $50 \times 35 = 1,750$ 条完全独立的采样轨迹。同一位点的多个突变（如 `A:486` 含 6 个突变、`A:487` 含 4 个突变）共享完全一致的 `cb8` 邻域、`neighbor_shell` 拓扑以及初始约束最小化构象。
- **极强的空间局部性**：在 21 个位点上，`cb8` 8.0 Å 邻域仅含 **5–14 个残基**（均值 10.05），扩展后的 `neighbor_shell` 仅含 **9–29 个残基**（均值 17.71），仅占全复合物 420 残基的 **2.1%–6.9%**。在 35,000 步 Backrub/Sidechain MC 及后续 `repack`/`mutate` 期间，其余 391–411 个残基坐标完全固定为静态刚体背景。
- **核心计算热点**：单条轨迹的 35,000 步 Backrub/Sidechain MC 占据绝大部分计算时间（核心操作为高频局部坐标更新与 `talaris2014` 局部增量打分，每步仅涉及 1–3 个扰动残基与周围 6.0 Å 截断壳层内的原子对相互作用）；其次为 WT/MUT 侧的 MultiCool 侧链重排（单/双体旋转异构体能量表构建与模拟退火）以及约束 L-BFGS 最小化（高频能量与梯度评估）。此外，清单包含 4 个突变入脯氨酸位点（`N440P`、`N481P`、`N501P`、`F486P`）与 3 个野生型脯氨酸突变位点（`P479D/L/V`），对五元环闭合几何与 `pro_close` 势能项数值精度敏感。

## 异构架构与算子实现

### 目标硬件资源映射

目标硬件为 **24 核 Host CPU（`taskset -c 0-23`）+ 单卡 Ascend 950PR（`Ascend950PR_9579`）**：

- **Ascend 950PR 片上与显存资源**：28 Cube Core / 56 Vector Core，每核 248 KB Unified Buffer（UB），128 MB 共享 L2 Cache，128 GB HBM；CANN 9.2.1，支持 Ascend C 与纯 SIMT（`dav-c310-vec`）编程模型。
- Flex ddG 的 MC 采样与局部能量评估以标量/短向量几何变换、分支判断（距离截断、氢键几何判据、Metropolis 接受）与查表插值为主，契合 Ascend 950PR **56 个 Vector Core + SIMT 执行模式 + 大容量 UB/L2** 的微架构特性。

### Host CPU / NPU 异构分工与数据流

```text
[Host 24-Core CPU (taskset -c 0-23)]                       [Device Ascend 950PR (56 Vector Cores)]
==================================================         ==================================================
1. 输入解析与前置严格校验                                   (预加载只读力场常量表至 HBM / L2 Cache)
   - CSV 6 列、PDB 存在性、chain/icode/wt_aa 匹配
   - 参数值域与 24 核上限校验
        |
        v
2. 按 21 个独立位点去重预处理（24 核并行）
   - 全原子拓扑建图、6 对二硫键 CYD 与终端补丁
   - cb8 (8Å) bubble 与 neighbor_shell (9~29 res)
   - 初始 CA 约束 L-BFGS 最小化
   - 提取活跃残基 + 6Å 截断背景接触壳层（压缩紧凑表）
   - 构建 Dunbrack (-ex1 -ex2, cutoff=0) 旋转异构体库
        |
        +--- 异步 H2D (ACL Stream A/B 双缓冲) -----------> 3. 批量跨轨迹 Backrub / Sidechain MC 内核
        |                                                     - 56 Vector Cores + SIMT 跨轨迹并发
        |                                                     - UB 驻留活跃壳层坐标与局部能量累加器
        |                                                     - L2 共享静态背景接触壳层与 talaris2014 表
        |                                                     - 完整执行 35,000 步 MC，零步级 H2D/D2H
        |                                                          |
        |                                                          v
        |                                                  4. WT / MUT 后处理加速内核
        |                                                     - 并行构建活跃区单/双体旋转异构体能表
        |                                                     - 批量 MultiCool (states=6) repack & mutate
        |                                                     - 批量 CA 约束 L-BFGS 局部能量/梯度与迭代
        |                                                          |
        |<-- 异步 D2H (仅回传终态坐标与四态能量项) --------+
        v
5. Host 汇总与自洽校验（24 核异步重叠）
   - 四态接口双差计算与 7 项分子间项拆解
   - abs(ddG_residual) <= 0.01 与有限值检查
   - ZEMu GAM 重加权与 35 轨迹均值/标准差
   - 原子落盘 22 列 results.csv 与 run_summary.json
```

### Host 侧设计

1. **前置校验与快速失败**：启动后第一时间校验 `--selected`、`--pdb-dir`、`--xml`、`--move-chain`、`--nstruct`、`--trials`、`--cpu-cores`、`--device-id`、`--seed`、`--work`、`--out`、`--summary` 全部约束；在解析 PDB 时按 `(chain, pdb_position, icode)` 精确匹配每个突变位点并核对 `wt_aa`，任何违规立即输出清晰错误信息并以非零退出码退出。
2. **21 位点去重预处理**：将 50 个突变按 `(pdb_id, chain, pdb_position, icode)` 归并为 21 个独立位点，在 24 核线程池上并行完成 PDB 全原子加氢、`CYD` 二硫键配对、终端补丁、`cb8`/`neighbor_shell` 划分及初始 CA 约束 L-BFGS 最小化；随后按每个突变构建对应的活跃区结构与 `talaris2014` 紧凑参数包，消除重复的初始最小化与参数导出开销。
3. **分桶打包与 LPT 负载均衡调度**：
   - 将 1,750 条轨迹按 `neighbor_shell` 残基数和旋转异构体数量分桶打包为大小均匀的 Device Batch（例如按 56 个 Vector Core 的整数倍组织批内轨迹数，使每个批内各轨迹的活跃残基数相近，抑制同一核心/SIMT 线程束内的长尾等待）。
   - 调度队列采用最长处理时间优先（LPT）策略，将活跃壳层较大、计算量较高的位点优先派发，小位点用于尾部填缝。
4. **24 核约束与双 Stream 流水线**：
   - 主机侧线程池/进程池严格受 `--cpu-cores`（上限 24）与 `taskset -c 0-23` 管控，不拉起超核后台线程。
   - 使用两条异步 ACL Stream 交替提交 H2D、NPU Kernel 与 D2H，使下一批输入准备与上一批结果校验/落盘与 NPU 计算时间重叠。

### Kernel 侧设计（Ascend 950PR）

#### 1. 分核与并行映射策略

- **一级并行（核间）**：以「独立轨迹」作为基本任务单元均匀映射到 Ascend 950PR 的 56 个 Vector Core。单条轨迹的 35,000 步 MC 存在严格的步间马尔可夫串行依赖，但 1,750 条轨迹之间完全无数据依赖；通过跨突变、跨轨迹混合成批，56 个核心可维持满核负载均衡。
- **二级并行（核内 SIMT）**：在 `dav-c310-vec` SIMT 模式下，每个 Vector Core 内按线程束（warp/lane）同时推进多条同桶轨迹（或针对大型 `neighbor_shell` 由多 lane 协同展开同一轨迹的局部原子对打分并做 warp 级归约），同时将多条轨迹的只读力场参数访问合并至 L2 缓存命中。

#### 2. 片上 UB（248 KB）与 L2（128 MB）存储布局与数据搬运

- **活跃壳层常驻 UB**：单条轨迹的 `neighbor_shell` 仅 9–29 个残基（约 140–450 全原子）。采用 Structure-of-Arrays（SoA）且 32 字节对齐的紧凑结构存放活跃原子坐标 `(x, y, z)`、原子类型索引、部分电荷、残基偏移表与 Backrub 支点索引。单轨迹活跃状态与试探坐标双缓冲约占 15–35 KB，可在 248 KB UB 内安全驻留并预留旋转矩阵、氢键矢量与局部能量累加器的片上临时空间。
- **6.0 Å 截断背景壳层驻留 L2**：`talaris2014` 绝大多数两体短程项（`fa_atr`, `fa_rep`, `fa_sol`, `fa_elec`, 氢键）的交互截断半径为 5.5–6.0 Å（`fa_max_dis = 6.0`，约束最小化时 `fa_max_dis = 9.0`）。在 Host 预处理阶段预先筛选距 `neighbor_shell` 任意重原子在安全截断半径（如 8.0 Å，覆盖 Backrub 小幅主链位移裕量）内的静态刚体背景残基，压缩为连续只读表存入 HBM 并常驻 128 MB L2 Cache，同位点 35 条轨迹共享访问；远离界面的约 300+ 个背景残基完全退出 MC 内循环。
- **静态背景自能预扣除**：背景-背景原子对能量在 MC 与 Repack 阶段为常量，仅在阶段切换或全结构最终打分时计入一次，MC 步内仅计算「活跃-活跃」与「活跃-背景壳层」增量能量差 $\Delta E$。
- **零步级 GM 往返**：35,000 步 MC 演化（含几何提案、增量打分、Metropolis 接受/回滚）全部在设备端连续完成，仅在终态（步数 `35000`）写回活跃残基最终坐标、能量分量及统计计数，大幅压缩 HBM 与 PCIe 带宽开销。

#### 3. 局部 `talaris2014` 增量打分与后处理内核

- **增量能量内核**：当某步扰动仅涉及局部 1–3 个残基（Backrub 三残基段或单残基侧链 rotamer）时，内核通过预构建的残基级空间邻接表（结合 Cβ 包围球快速剔除）仅遍历与受扰残基距离在截断半径内的邻居原子对，逐项评估 `talaris2014` 全部分量：
  - 两体非键项：`fa_atr`、`fa_rep`、`fa_sol`（Lazaridis-Karplus）、`fa_elec`（距离相关介电静电）、`hbond_sr_bb`、`hbond_lr_bb`、`hbond_bb_sc`、`hbond_sc`（`talaris2014` SP2 氢键多项式）、`dslf_fa13`；
  - 一体与局部扭转项：`fa_intra_rep`、`fa_dun`（Dunbrack 2010 旋转异构体势能）、`rama`、`omega`、`p_aa_pp`、`pro_close`、`yhh_planarity`、`ref`。
- **后处理重排与最小化内核**：
  - **Interaction Graph 与 MultiCool Repack/Mutate**：对 `neighbor_shell` 的 `-ex1 -ex2` 旋转异构体集合在 NPU 上并行计算单/双体交互能表，再以 6 态 MultiCool 退火策略（`states=6`）批量求解 WT `repack` 与 MUT `mutate` 最优侧链构象；
  - **约束 L-BFGS 最小化**：在活跃壳层自由度及全复合物 CA 距离约束（`coord_dev=0.5`, `max_distance=9.0`, `fa_max_dis=9.0`）下，利用 NPU 批量评估能量与解析梯度（`lbfgs_armijo_nonmonotone`, `tolerance=1e-6`, `max_iter=5000`, `abs_score_convergence_threshold=1.0`），确保与官方最小化收敛判据一致。
- **跨轨迹独立 RNG**：采用计数器型伪随机数生成器（如 Philox-4×32-10 或独立流 SplitMix64/PCG），按 `(seed, mutation_id, trajectory_id, step)` 确定性派生各轨迹独立随机流，消除线程间共享状态竞争并支持同 `--seed` 复现。

### `talaris2014` 数值一致性与特殊位点保障

1. **参数源头精确一致**：力场权重、原子类型 LJ/LK 表（Etable）、氢键多项式系数、Dunbrack 2010 旋转异构体库、`rama`/`p_aa_pp` 样条系数及氨基酸参考能 `ref` 均直接从开源 Rosetta `2026.29`（开启 `-corrections::restore_talaris_behavior true`）导出并固化校验哈希，不进行手工简化或近似拟合。
2. **混合精度数值保护**：三维坐标、Rodrigues 旋转与键角/二面角重建、距离计算及力场能量累加全程禁用 FP16 截断；MC 局部增量打分与坐标更新至少采用 FP32，其中能量总和归约、L-BFGS 目标函数/梯度内积及最终四态接口打分采用 FP64（或补偿累加）以防步间误差累积与线搜索过早终止。
3. **二硫键 `CYD` 与终端补丁**：严格识别全部 6 对二硫键（含位于界面热点区的 `A480-A488`），保持 `CYD` 拓扑参数与 `dslf_fa13` 势能项，禁止将交联半胱氨酸误作自由 `CYS` 处理；正确加载三条链 N/C 端全原子补丁（`NtermProteinFull` / `CtermProteinFull`）的原子类型与电荷。
4. **脯氨酸环闭合与 `pro_close`**：针对 4 个突变入脯氨酸位点（`N440P`、`N481P`、`N501P`、`F486P`）及 3 个野生型脯氨酸位点（`P479D/L/V`），精确实现脯氨酸五元环（`N-CA-CB-CG-CD`）内坐标重建与环闭合谐振势能 `pro_close`（及其对主/侧链二面角的解析梯度），防止最小化阶段环几何畸变。
5. **四态自洽与逐项对账**：在最终四态打分阶段验证 `bound_wt / unbound_wt / bound_mut / unbound_mut` 全部 7 个分子间项与总双差的一致性，强制执行 `abs(ddG_residual) <= 0.01`，任意轨迹或突变越界即报错阻断。

## 支持硬件

| 硬件组件 | 规格与约束 | 涉及勾选 |
| --- | --- | --- |
| NPU | Ascend 950PR（单卡，28 Cube Core / 56 Vector Core，248 KB UB/Core，128 MB L2，128 GB HBM） | √ |
| Host CPU | 固定绑定 24 核（`taskset -c 0-23` 或 `--cpuset-cpus=0-23`） | √ |

## 约束限制

1. 输入 PDB 须为不含 `HETATM` 的标准全原子蛋白质复合物结构，突变清单中的 `(chain, pdb_position, icode, wt_aa)` 须与 PDB 精确一致。
2. 正式验收全程固定 24 核 CPU 绑定与单卡 Ascend 950PR，不得以超核或跨机并行规避。
3. 本算法为蒙特卡洛随机系综，不要求不同运行或不同硬件架构间逐比特（bit-wise）一致，以分层统计精度与残差自洽标准验收；提供 `--seed` 保障同环境配置可复现。
4. 单条轨迹中间临时产物不纳入 Git 仓库，仅输出汇总 `results.csv`、`run_summary.json` 及验收报告。

# 可维可测分析

## 精度标准/性能标准

| 验收维度 | 验收指标与判据 | 标准来源 |
| --- | --- | --- |
| 完整性与格式 | 50/50 突变全部完成；22 列字段完整；无 NaN/Inf；`nstruct_done == 35`；`backrub_trials == 35000`；`ddG_all(REU)` 含 35 个有限值；`abs(ddG_residual) <= 0.01` | 任务书 §2.4.2、§3.2，`verify_bd623.py` |
| 精度（大效应组） | $|\overline{\Delta\Delta G}_{\text{ref}, i}| > 1.0\ \text{REU}$（15 个突变）：相对误差 $\text{rel\_err}_i = \frac{|\overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref}, i}|}{|\overline{\Delta\Delta G}_{\text{ref}, i}|}$ 的**中位数 `< 5%`（`< 0.05`）** | 任务书 §3.2，`verify_bd623.py` |
| 精度（小效应组） | $|\overline{\Delta\Delta G}_{\text{ref}, i}| \le 1.0\ \text{REU}$（35 个突变）：**全部 35 个样本**均满足 $\sigma_i = \frac{|\overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref}, i}|}{\text{std}_{\text{ref}, i}} < 1.0$（即 $\max(\sigma) < 1.0$） | 任务书 §3.2，`verify_bd623.py` |
| 精度（辅助诊断） | 同步输出全量 50 突变 Pearson 相关系数 $r$、逐突变 $\sigma_i$ 与相对误差，排查系统偏移 | 任务书 §3.2，`verify_bd623.py` |
| 端到端性能 | 24 核 CPU（`taskset -c 0-23`）+ 单卡 Ascend 950PR 运行 BD-623 全部 50 突变，完整墙钟耗时 `wall_time_s <= 5,565.3 s`（相对 48 核基线 `55,653.4 s` 加速比 `>= 10.0×`） | 任务书 §3.3，`verify_bd623.py` |

## 验证与测试设计

1. **力场与算子单元测试**：
   - 在相同输入构象上逐项对比自研 NPU/CPU `talaris2014` 评估内核与开源 Rosetta `2026.29` 的全部 17 项能量分量（覆盖普通界面位点、二硫键 `A480-A488` 邻域、以及 4 个脯氨酸突变 `N440P/N481P/N501P/F486P`），验证单构象逐项能量误差在可控阈值内且解析梯度与有限差分一致；
   - 验证 Backrub 主链旋转、侧链旋转异构体坐标重建、CA 距离约束与四态双差/ZEMu GAM 汇总逻辑的数学正确性。
2. **端到端全量正向验收**：
   - 严格通过 `taskset -c 0-23` 执行 `run_flexddg_npu.py` 完成 `bd623_mutations.csv` 全部 50 个突变（1,750 条轨迹），计时覆盖从命令行启动、输入解析、参数准备/算子加载、MC 采样、后处理到 `results.csv` 与 `run_summary.json` 落盘的完整端到端墙钟时间；
   - 调用随包 `test_cases/verify_bd623.py` 对输出执行一键完整性、分层精度与性能校验，并生成 `verify_report.json`。
3. **负向与边界用例测试**：
   - 覆盖缺失 PDB 文件、缺失 CSV 必选列、空 CSV、位点在 PDB 中不存在、`wt_aa` 与 PDB 实际残基不匹配、非法 `--move-chain`、`--cpu-cores <= 0` 或 `> 24`、`--nstruct <= 0`、`--trials <= 0`、越界 `--device-id`、非法 `--seed`、不可写 `--work`/`--out` 路径，确认均能明确报错并返回非零退出码。
4. **可观测性与 Profiling**：
   - `run_summary.json` 固定输出硬件型号（CPU 型号与核数绑定、Ascend 950PR）、CANN 与驱动版本、代码 commit、`seed`、完整执行命令及分阶段耗时（预处理、MC 采样、WT/MUT 后处理、四态打分与落盘），支持性能归因与复现审计。

## 兼容性分析

- **工程目录规范**：交付工程按任务书第 5 章组织为独立模块 `rosetta_flex_ddg/`，包含 `README.md`、`build.sh`、`env/`、`pipeline/`（含 `run_flexddg_npu.py`）、`kernels/`（Ascend C / SIMT 算子及原生扩展）与 `tests/`，不修改系统全局环境，不将 `.db3` 等大体积中间轨迹文件纳入代码仓。
- **软件栈依赖**：依赖 CANN 9.0.0 及以上（适配开发与验收环境 CANN 9.2.1）及对应驱动固件、C++17 编译器与 Python 3 标准运行环境；不引入非开源商业依赖。
