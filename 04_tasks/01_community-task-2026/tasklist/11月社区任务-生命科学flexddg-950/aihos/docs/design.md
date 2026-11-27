# 需求背景（required）

## 需求来源

本需求来源于昇腾 2026 年 11 月社区任务[11月社区任务-生命科学flexddg-950](https://www.hiascend.com/activities/task-center/details/4373d1139dfe4ba7b448a8ef5cc3596b)。目标是在 **24 核 CPU + 单卡昇腾 Ascend 950PR（`NpuArch::DAV_3510`）** 与 CANN 9.0.0+ 软件栈环境下，面向生命科学蛋白质-蛋白质结合界面突变自由能变化（$\Delta\Delta G$）预测场景，基于开源 Rosetta 与 Kortemme 实验室官方协议 `flex_ddG_tutorial`（`ddG-backrub.xml`），开发 Rosetta Flex ddG 整模块端到端异构加速方案，完成算法设计、系统实现与全量自验，验收通过后正式合入开源仓。

- **对标开源基线**：
  - Rosetta 主体：GitHub 开源仓 `RosettaCommons/rosetta`（参考基线版本 `2026.29` / commit `80a0635615`）；
  - Flex ddG 官方协议：GitHub 开源仓 `Kortemme-Lab/flex_ddG_tutorial`（官方协议原件 `test_cases/ddG-backrub.xml` 与基线评测脚本 `test_cases/baseline/`）；
- **评测复合物与基准数据集**：新冠抗体-抗原复合物 `BD-623`（PDB `7E7Y`，共 420 残基，无 `HETATM`，链 `A`=抗原、链 `H`/`L`=抗体）的全部 50 个抗原侧单点突变（每突变 35 条独立轨迹 × 35,000 步 Backrub 采样，全量累计 1,750 条轨迹、61,250,000 步蒙特卡洛采样）；
- **48 核 CPU 开源基线**：在 2×Xeon Platinum 8360Y（48 核满载，开源 Rosetta `2026.29`，`talaris2014`）上，50 个突变累计消耗 **742.05 CPU·h（2,671,363.3 秒）**，整批 48 进程满载端到端墙钟耗时为 **15.46 小时（55,653.4 秒 / 927.6 分钟）**；
- **核心开发与验收目标**：在严守 `talaris2014` 力场与官方采样预算红线的前提下，基于 24 核 CPU（`taskset -c 0-23`）+ 单卡 Ascend 950PR，实现端到端总墙钟耗时达到 48 核 CPU 基线的 **1/10 及以内（`wall_time_s <= 5,565.3 s` / ≤ 92.8 分钟，加速比 ≥ 10.0×）**，且 100% 满足分层统计误差验收标准。

## 背景介绍

### Rosetta Flex ddG模块实现优化

Rosetta Flex ddG（Barlow et al., *J. Phys. Chem. B* 2018）是目前计算结构生物学中预测抗体-抗原结合自由能变化最精准的主流方法之一。与基于刚性骨架的传统打分方法相比，Flex ddG 引入局部主链柔性松弛（Backrub）与侧链重排系综平均，能够有效模拟突变引发的骨架形变、消除空间位阻冲突并重塑界面氢键网络。然而，原版开源 Rosetta 采用面向对象的纯 CPU 串行 C++ 架构，计算瓶颈极为严重：
1. **深层抽象与频繁跳转**：能量计算中存在深层虚函数分派、`EnergyGraph` 动态图遍历及零散堆内存分配，无法利用现代处理器的流水线与 SIMD 宽矢量单元；
2. **非键势能逐对计算开销**：全原子非键对（Lennard-Jones、Lazaridis-Karplus 隐式溶剂化、库仑静电势）与多项式氢键势在 CPU 标量循环中反复求值，吞吐受限于分支预测与标量延迟；
3. **后处理多轮迭代开销**：每条轨迹后处理需执行 2 次旋转异构体组合退火（`PackRotamersMover states=6`）及 2 次高维扭转空间 L-BFGS 能量最小化（`max_iter=5000`），耗时占比达 30% 以上。

针对上述瓶颈，本方案在 Ascend 950PR 上构建了端到端异构加速架构：
- **CPU/NPU 职责解耦**：Host 24 核 CPU 专注于参数解析、`talaris2014` 查表与拓扑导出及轻量四态能量汇总；Ascend 950PR 承担 1,750 条轨迹的驻留批处理；
- **全流程下沉 NPU**：将 35,000 步 Backrub/Sidechain 蒙特卡洛采样、Rotamer 交互图构建与退火（Packing）、全构象约束 L-BFGS 能量最小化三大核心热点完整下沉至 Ascend 950PR AI Core，实现片上数据常驻与 0 Host 介入；
- **确定性初态全局复用**：识别出官方协议中 1,750 条轨迹在突变发生前的初始最小化完全由同一野生型 PDB 与力场决定，合并为单次确定性求解并广播，从算法层消除 ~3,500 秒冗余计算。

### Rosetta Flex ddG模块功能分析

Rosetta Flex ddG 协议执行严格的四阶段蒙特卡洛系综与界面打分计算：
1. **结构准备与初始最小化（Stage 0）**：全原子加氢，位点野生型残基校验；根据突变位点按 8 Å（`cb8`）半径构造 `bubble` 并前后各扩 1 残基形成 `neighbor_shell`（BD-623 中仅 9~29 个残基）；施加 2,787 对 CA 谐波距离约束（`coord_dev=0.5, max_distance=9.0`），执行全蛋白 L-BFGS 能量最小化（`tolerance=1e-6, max_iter=5000`）；
2. **Backrub / Sidechain 蒙特卡洛采样（Stage 1）**：每突变独立运行 35 条轨迹，每条轨迹在 `neighbor_shell` 枢轴残基上随机执行 35,000 步主链 Backrub 旋转或侧链 Chi 扰动（`mc_kt = 1.2`），按 Metropolis 准则接受/拒绝，末态构象记为 `backrubpdb`；
3. **WT / MUT 双分支重排与后处理最小化（Stage 2）**：
   - **WT 分支**：对 `backrubpdb` 在 `neighbor_shell` 上执行野生型 repack（`UseMultiCoolAnnealer states=6`）及约束 L-BFGS 最小化，得到结合态 `wt_bound_pose`；
   - **MUT 分支**：将位点突变为目标残基，在 `neighbor_shell` 上执行 mutate packing（`UseMultiCoolAnnealer states=6`）及约束 L-BFGS 最小化，得到结合态 `mut_bound_pose`；
4. **四态接口打分与重加权（Stage 3）**：将结合态构象沿抗原链（`A` 链）刚体平移 1000 Å 形成解离态 `unbound`，计算四态双差：
   $$\Delta\Delta G = (E_{\text{bound,mut}} + E_{\text{unbound,wt}}) - (E_{\text{bound,wt}} + E_{\text{unbound,mut}})$$
   输出原始均值 `ddG_mean(REU)`、7 个分子间能量项双差均值，并通过 ZEMu GAM 重加权公式计算最终自由能变化 `ddG_gam_mean(kcal/mol)`。

# 需求分析（required）

## 需求描述

在 24 核 CPU + 单卡 Ascend 950PR 环境下，提供满足任务书 §2.3 规范的统一端到端命令行入口：

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

**输入输出参数与文件规格**：

| 参数名 | 属性/方向 | 描述与规格 | 数据类型与格式 | 约束与有效范围 | 异常行为与处理 |
| --- | --- | --- | --- | --- | --- |
| `--selected` | 输入 | 待计算突变清单 CSV 文件路径 | file(CSV) | 必须包含 6 列：`pdb_id,chain,pdb_position,icode,wt_aa,mut_aa`，本任务为 50 行 | 文件缺失、列缺失、位点不存在或 PDB 实际残基与 `wt_aa` 不符时报错并返回非零退出码 |
| `--pdb-dir` | 输入 | 全原子 PDB 文件所在目录 | dir(PDB) | 文本三维原子坐标，BD-623 含 420 残基、3,217 重原子 | 目录不存在或 PDB 缺失时报错退出 |
| `--xml` | 输入 | Flex ddG 协议 XML 文件路径 | file(XML) | 必须为官方 `test_cases/ddG-backrub.xml` | 文件缺失或解析失败时返回非零退出码 |
| `--move-chain` | 属性 | 解离态刚体移开的链 ID | attr (string) | BD-623 中抗原链固定为 `'A'` | 链在 PDB 中不存在时报错退出 |
| `--nstruct` | 属性 | 每个突变运行的独立轨迹数 | scalar (int) | 正式验收固定为 `35` | `<= 0` 拒绝执行；`< 35` 正式验收不通过 |
| `--trials` | 属性 | 每条轨迹 Backrub 采样步数 | scalar (int) | 正式验收固定为 `35000` | `<= 0` 拒绝执行；`< 35000` 正式验收不通过 |
| `--cpu-cores` | 属性 | 主机侧 CPU 绑定核数 | scalar (int) | 正式验收固定为 `24` | `<= 0` 或 `> 24` 拒绝执行 |
| `--device-id` | 属性 | Ascend 950PR 逻辑卡号 | scalar (int) | 默认 `0`，有效 NPU 卡号 | 设备不可用或越界时报错退出 |
| `--seed` | 属性 | 随机数种子起始值 | scalar (int) | 默认 `111111` | 非法整数报错退出 |
| `--work` | 输出 | 临时工作目录 | dir | 可写目录 | 路径不可写时报错退出 |
| `--out` | 输出 | 汇总输出 CSV（`results.csv`） | file(CSV) | 严格 22 列规范，与输入突变一一对应 | 突变未完成、含 NaN/Inf 或残差超限时判失败 |
| `--summary` | 输出 | 运行摘要 JSON（`run_summary.json`） | file(JSON) | 必含 8 项关键字段的合规 JSON | 字段缺失或写入失败时返回非零退出码 |

**主输出 `results.csv` 22 列字段契约**：
1. `1–6 列`：`pdb_id,chain,pdb_position,icode,wt_aa,mut_aa`（与输入精确一致）；
2. `7–9 列`：`ddG_mean(REU)`, `ddG_std(REU)`, `ddG_all(REU)`（35 轨迹分号分隔原始双差）；
3. `10–16 列`：7 个分子间能量项双差均值（`ddG_fa_atr`, `ddG_fa_rep`, `ddG_fa_sol`, `ddG_fa_elec`, `ddG_hbond_sc`, `ddG_hbond_bb_sc`, `ddG_hbond_lr_bb`）；
4. `17 列`：`ddG_residual(REU)`，满足 `abs(ddG_residual) <= 0.01`；
5. `18–19 列`：`ddG_gam_mean(kcal/mol)`, `ddG_gam_std(kcal/mol)`；
6. `20–22 列`：`nstruct_done`（恒等于 35）、`backrub_trials`（恒等于 35000）、`cpu_time(s)`。

## 需求拆解

1. **协议一致性与三条红线严格遵守**：
   - 红线 1：力场锁定 `talaris2014`（开启 `-corrections::restore_talaris_behavior true`），严禁更换力场或删减能量项；
   - 红线 2：采样预算不降（`nstruct=35`，`backrub_trials=35000`，`distance="8.0"`，`max_iter=5000`，`tolerance=1e-6`，`abs_score_convergence_threshold=1.0`）；
   - 红线 3：官方阶段不删减（初始最小化、MC 采样、WT repack、MUT mutate、后处理约束最小化、四态打分）；
2. **CPU/NPU 异构分工与确定性初态全局复用**：
   - 24 核 CPU 严格通过 `taskset -c 0-23` 限核，执行输入前置校验、力场/旋转异构体参数一次性张量化导出、波次调度与四态汇总；
   - 1,750 次相同输入的初始 WT 最小化合并为 1 次全局确定性执行并广播给所有轨迹，数学等价且零精度受损；
3. **NPU 端 1,750 轨迹批处理驻留流水线**：
   - 依据 Ascend 950PR（`NpuArch::DAV_3510`，56 个 AIV 核，单核 248 KB UB）硬件架构，实现三大 Ascend C / SIMT 核心算子组：
     - `BackrubMCKernel`：SIMT 512 线程并发，局部坐标 UB 驻留，0 Host-Device 交互；
     - `PackRotamersKernel`：高并发构建 1-body/2-body 能量图，6 态 MultiCool 模拟退火；
     - `LbfgsMinKernel`：Verlet 邻居表 + AtomTree 链式解析梯度归约 + 批处理 Armijo L-BFGS；
4. **统计精度达标保障**：
   - 大效应样本（15 例）：中位数相对误差 $\text{median}(\text{rel\_err}) < 5\%$；
   - 小效应样本（35 例）：全部满足 $\lvert \overline{\Delta\Delta G} - \overline{\Delta\Delta G}_{\text{ref}} \rvert < 1.0 \cdot \text{std}_{\text{ref}}$；
   - 格式与物理自洽：50 突变 100% 完成，`nstruct_done=35`，`backrub_trials=35000`，`|ddG_residual| <= 0.01`，无 NaN/Inf；
5. **性能达标保障**：
   - 50 突变端到端总墙钟耗时 `wall_time_s <= 5,565.3 s`（加速比 $\ge 10.0\times$），设计期预留 50% 安全余量（目标 1,500~2,800 s）。

# 详细设计（required）

## 算子分析

### 数学公式

1. **四态结合自由能双差与分子间分项展开**：
   $$\Delta\Delta G = (E_{\text{bound,mut}} + E_{\text{unbound,wt}}) - (E_{\text{bound,wt}} + E_{\text{unbound,mut}})$$
   每个状态下的总能量按 `talaris2014` 力场各项加权求和：
   $$E_{\text{pose}} = \sum_{k} w_k \cdot E_k$$
   其中包含 7 个核心分子间相互作用项：
   - $E_{\text{fa\_atr}}, E_{\text{fa\_rep}}, E_{\text{fa\_sol}}$：Lennard-Jones 6-12 范德华吸引/排斥项与 Lazaridis-Karplus 溶剂化自由能项（基于三次样条插值 Etable 表，截断半径 6.0 Å）；
   - $E_{\text{fa\_elec}}$：库仑静电项（距离依赖介电常数与平滑截断函数，截断半径 5.5 Å）；
   - $E_{\text{hbond\_sc}}, E_{\text{hbond\_bb\_sc}}, E_{\text{hbond\_lr\_bb}}$：短程与长程骨架-侧链、侧链-侧链多项式氢键项；
   以及分子内项与约束项：$E_{\text{fa\_dun}}$（Dunbrack 旋转异构体扭转势）、$E_{\text{rama}}$、$E_{\text{omega}}$、$E_{\text{p\_aa\_pp}}$、$E_{\text{pro\_close}}$（脯氨酸环闭合势）、$E_{\text{dslf\_fa13}}$（二硫键势）及 $E_{\text{atom\_pair\_constraint}}$（CA 距离谐波约束势 $E = ((d_{CA} - d_0) / 0.5)^2$）。
2. **ZEMu GAM 重加权非线性映射**：
   $$\operatorname{gam}(x, \text{term}) = -\exp(a_{\text{term}}) + \frac{2 \exp(a_{\text{term}})}{1 + \exp(-x \cdot \exp(b_{\text{term}}))}$$
   $$\Delta\Delta G_{\text{gam}} = \sum_{\text{term} \in \text{7\_intermolecular\_terms}} \operatorname{gam}(\Delta\Delta G_{\text{term}}, \text{term})$$
3. **Backrub 局部刚体变换（Rodrigues 旋转公式）**：
   对以骨架原子 $C_{\alpha, i}$ 与 $C_{\alpha, j}$（跨度 2~12 残基）为旋转轴单位向量 $\mathbf{k}$ 的片段，任意原子相对位置向量 $\mathbf{v}$ 绕轴旋转 $\theta$ 角：
   $$\mathbf{v}' = \mathbf{v}\cos\theta + (\mathbf{k} \times \mathbf{v})\sin\theta + \mathbf{k}(\mathbf{k} \cdot \mathbf{v})(1 - \cos\theta)$$
   根据 Metropolis 准则判断新构象接收概率：
   $$P_{\text{accept}} = \min\left(1, \exp\left(-\frac{\Delta E_{\text{local}}}{k_B T}\right)\right), \quad k_B T = 1.2$$
4. **AtomTree 扭转角空间解析梯度投影**：
   对骨架 $(\phi, \psi, \omega)$ 与侧链 $\chi$ 扭转角自由度，Cartesian 力 $\mathbf{F}_a$ 投影至扭转梯度：
   $$g_k = \frac{\partial E}{\partial \theta_k} = \sum_{a \in \operatorname{subtree}(k)} \mathbf{u}_k \cdot \left((\mathbf{r}_a - \mathbf{p}_k) \times \mathbf{F}_a\right)$$
   其中 $\mathbf{u}_k$ 为旋转轴方向，$\mathbf{p}_k$ 为轴上支点，$\operatorname{subtree}(k)$ 为受该二面角变动影响的原子子树。

### 支持数据类型

| 运算阶段与数据载体 | 数据类型 | 精度与格式要求 | 存储位置 |
| --- | :--- | :--- | :--- |
| 原子三维坐标与速度 | `float` (FP32) | IEEE 754 单精度，连续 SoA 排布 | Device GM / 片上 UB 缓存 |
| 原子非键解析力 $\mathbf{F}$ | `float` (FP32) | IEEE 754 单精度，原子对局部规约 | 片上 UB / GM 累加缓冲 |
| 势能项累加与四态双差 | `double` (FP64) / FP32 补偿求和 | 双精度累加，防止大数吃小数 | 片上寄存器 / Host 汇总 |
| 力场 Etable、Dunbrack、Rama 表 | `float` (FP32) | 预烘焙样条插值系数与能量查找表 | Device GM 只读常量区（~32 MB） |
| 原子/残基拓扑与邻居索引 | `uint16_t` / `int32_t` | 紧凑整数表 | Device GM / 片上 UB |
| 输出自由能 $\Delta\Delta G$ | `float` (FP32) | 22 列标准 CSV 输出 | Host 内存与落盘文件 |

### 支持形状

1. **体系规模**：支持全原子复合物 PDB 输入，当前针对 `BD-623.pdb`（420 残基、3,217 重原子、全原子加氢后约 6,500 原子），并支持扩展至上千残基蛋白质体系；
2. **拓扑多样性**：完整支持含链断裂（如 BD-623 链 A 缺失 517–521）、共价二硫键（如 BD-623 包含 6 对二硫桥，含界面核心 `A:C480–A:C488`）及 2,787 对 CA 距离约束；
3. **并发工作负载**：支持固定 50 突变 × 35 独立轨迹 = 1,750 条轨迹批处理并发调度，每条轨迹固定 35,000 步采样。

## 算子实现

### 实现方案

```
+----------------------------------------------------------------------------------------------------+
|                               Rosetta Flex ddG 端到端异构加速总架构                                  |
+----------------------------------------------------------------------------------------------------+
  【Host 侧 (24 核 CPU, taskset -c 0-23 严格限核)】
    1. CLI 入口参数前置强校验与 XML 协议完整性核对
    2. 全原子拓扑导出与 talaris2014 力场参数表/Rotamer 库一次性张量化预烘焙 (~32 MB)
    3. Stage 0 初始确定性 WT 约束 L-BFGS 最小化（执行 1 次并广播至 1,750 条轨迹）
    4. 轨迹波次打包调度（分 5 个 Wave，每 Wave 处理 350 条轨迹）
                                     |
               H2D 异步下发：力场常量表 + 初始坐标与拓扑 + 轨迹批任务描述符
                                     v
  【Device 侧 (Ascend 950PR, DAV_3510, 56 AIV 核, 248 KB UB/核)】
    +----------------------------------------------------------------------------------------------+
    |  Kernel 1: BackrubMCKernel (Stage 1, 35,000 步驻留 MC 采样)                                   |
    |    - 56 AIV 核网格分核，单核负责连续轨迹状态机，512 SIMT 线程协同                               |
    |    - neighbor_shell 坐标与环境原子完全驻留单核 ~57 KB UB，0 GM 读写放大                        |
    |    - Philox PRNG 独立确定性流，Metropolis 增量能量局部判据与坐标回滚                           |
    +----------------------------------------------------------------------------------------------+
                                     | (构象快照直接驻留 HBM，零中间传输)
                                     v
    +----------------------------------------------------------------------------------------------+
    |  Kernel 2: PackRotamersKernel (Stage 2a, WT repack 与 MUT mutate)                            |
    |    - 局域候选 Rotamer 1-body / 2-body 能量图（IG）向量化并行构建                              |
    |    - 6 态 MultiCool 模拟退火在片上缓存完成毫秒级收敛                                            |
    +----------------------------------------------------------------------------------------------+
                                     |
                                     v
    +----------------------------------------------------------------------------------------------+
    |  Kernel 3: LbfgsMinKernel (Stage 2b & Stage 3, 全构象约束 L-BFGS 与四态打分)                 |
    |    - Verlet 邻居表动态管理 + talaris2014_cst 能量与解析力 SIMT 并发计算                       |
    |    - AtomTree 链式解析梯度快速投影 + 批处理非单调 Armijo L-BFGS 迭代                          |
    |    - A 链刚体平移 1000 Å 四态打分，FP64 精度累加                                             |
    +----------------------------------------------------------------------------------------------+
                                     |
               D2H 异步回传：1,750 轨迹四态 7 项分项能量矩阵 (< 1 MB)
                                     v
  【Host 侧后处理与验收落盘】
    1. 汇总 35 轨迹均值、标准差与 7 个分子间项，计算 ZEMu GAM 重加权
    2. 残差自洽性核验：|ddG_residual| <= 0.01 与无 NaN/Inf 保护
    3. 输出 results.csv (22 列) 与 run_summary.json，触发 verify_bd623.py 自动化验收
+----------------------------------------------------------------------------------------------------+
```

#### 3.2.1 host侧设计：

tiling策略：

根据 Ascend 950PR 拥有 56 个 AIV 核与 112/128 GB 高带宽显存的特征，采用**两级分层 Tiling（Wave Tiling + 核内任务均分）**：
1. **第一级：跨突变/跨轨迹的波次切分（Wave Tiling）**：将 50 个突变 × 35 条轨迹（共 1,750 条）划分为 $W = 5$ 个波次（Wave），每个 Wave 包含 $B = 350$ 条独立轨迹。单 Wave 峰值显存占用控制在 ~1.25 GB，完全适配 NPU 物理显存与 128 MB 大容量 L2 Cache；
2. **第二级：核级任务映射**：在每个 Wave 内，350 条轨迹映射至 56 个 AIV 核上，TilingData 结构体精简至 64 字节并经寄存器直传：

```cpp
struct FlexddgTilingData {
    uint32_t waveIndex;        // 当前波次序号 [0, 4]
    uint32_t trajectoriesInWave; // 当前波次轨迹数 (固定 350)
    uint32_t usedCoreNum;      // 实际激活核数 (56)
    uint32_t trialsPerLaunch;  // 单次 Launch 执行的 MC 步数 (如 5000 或 35000)
    uint32_t seedBase;         // 当前波次基础随机种子
    float mcKt;                // 蒙特卡洛温度 (1.2f)
    float lbfgsTol;            // L-BFGS 收敛容差 (1e-6f)
    uint32_t maxLbfgsIter;     // 最大最小化步数 (5000)
};
```

任务均分：

在每个 Wave 的 350 条轨迹中，56 个 AIV 核采用均匀平摊分配：
$$\text{base} = \lfloor 350 / 56 \rfloor = 6, \quad \text{rem} = 350 \bmod 56 = 14$$
前 14 个核心处理 7 条轨迹，其余 42 个核心处理 6 条轨迹。由于各核心独立处理各自轨迹的连续状态机，核与核之间完全无锁解耦，不存在任何核间同步等待。

批量搬运：

1. **力场常量单次预加载**：公共 `talaris2014` Etable、氢键参数、Dunbrack 旋转异构体概率表与初始 PDB 拓扑（共约 32 MB）在整批任务开始前**仅执行一次 H2D 拷贝**，全程常驻 NPU GM；
2. **中间构象零数据搬移**：Backrub 采样的 `backrubpdb` 构象、MultiCool 退火产生的侧链构象直接保存在 NPU HBM 连续槽位中，后续 L-BFGS 算子直接在显存内原地接力，彻底消除轨迹中间状态拷回 Host 的庞大 I/O 开销；
3. **极简结果回传**：全部采样与最小化完成后，每条轨迹仅回传四态分项能量标量数组（每轨迹仅 4 态 × 8 项 × 4B = 128 字节，全批 1,750 轨迹仅约 224 KB），D2H 耗时小于 1 毫秒。

##### 1. 分核策略：

- **核间分工**：大粒度轨迹并行。每个 AIV 核同一时刻独占处理 1 条轨迹，循环消费分配到的 6~7 条轨迹，消除核间数据竞争；
- **核内分工**：单核配置 512 个 SIMT 线程（划分为 16 个 Warp）：
  - 骨架 Rodrigues 旋转与 NeRF 坐标重建由 Warp 0~3 协同完成；
  - 局部增量能量计算时，16 个 Warp 均匀划分变动原子与周围环境原子的两两原子对，利用 `asc_shfl_down` 快速完成 Warp 内与跨 Warp 规约。

##### 2. 数据分块和内存优化策略：

Atlas 950PR 单 AIV 拥有 248 KB UB。在 SIMT 运行模式下预留 32 KB 作为动态数据缓存（DCache），剩余 216 KB 划分为严格管理的局域数据池：
- **`neighbor_shell` 变动原子缓冲区**：460 个原子 × 3 坐标 × 4B × 2 备份 = 11 KB；
- **9.0 Å 截断环境壳层缓冲区**：1,200 个固定原子 × 24B（坐标+电荷+原子类型）= 28.8 KB；
- **局域能量中间结果与规约区**：16.8 KB；
- **单核总 UB 占用仅 ~57 KB**（占 216 KB 的 26.4%），支持双缓冲无缝流水，确保 35,000 步 MC 循环实现 100% 片上命中。

##### 3. tilingkey规划策略：

根据异构执行阶段，规划 4 个标准 TilingKey 路由分支：
- `Key 0 (STAGE0_INITIAL_MIN)`：初始全构象确定性约束最小化；
- `Key 1 (STAGE1_BACKRUB_MC)`：驻留式 35,000 步 Backrub / Sidechain 蒙特卡洛融合核；
- `Key 2 (STAGE2_ROTAMER_ANNEAL)`：局域 Rotamer 1-body/2-body 能量图构建与 MultiCool 退火；
- `Key 3 (STAGE3_POST_MIN_EVAL)`：后处理 WT/MUT 约束最小化与四态刚体打分求值。

数据检测：

命令行入口执行严格的前置防御流水线：
1. 校验 `--selected` 存在、合法 UTF-8、包含必需 6 列，位点氨基酸在 20 种天然氨基酸内；
2. 校验 `--pdb-dir` 存在，`BD-623.pdb` 存在，且 PDB 实际残基与 CSV `wt_aa` 100% 吻合；
3. 校验 `--xml` 存在，解析节点确保包含 `talaris2014` 与官方采样/收敛参数；
4. 校验 `--cpu-cores` 属于 `[1, 24]`，并通过 `taskset -c 0-23` 强制锁定进程亲和性；
5. 校验 `--nstruct == 35`、`--trials == 35000`、`--device-id >= 0` 及输出目录可写性；
6. 结果写出时，强制断言 `|ddG_residual| <= 0.01`、无 NaN/Inf、50 突变全量输出。

#### 3.2.2 kernel侧设计：

1. **`BackrubMCKernel`（驻留式 35,000 步 MC 采样算子）**：
   - 采用 Ascend C SIMT 模式（`__simt_vf__` 与 `asc_vf_call`），512 线程/Block，每核运行一条轨迹；
   - 采用 Counter-based Philox PRNG，以 `(seed, mut_id, struct_id, step)` 保证随机数流严格可复现；
   - 每次试探仅计算变动原子（约 150~450 原子）与局部环境原子的能量差 $\Delta E$，计算量降低 20 倍；
   - 片上原地执行 Metropolis 准则，若拒绝则直接回滚 UB 备份，0 GM 访存惩罚。
2. **`PackRotamersKernel`（侧链重排与 MultiCool 退火算子）**：
   - 将 `neighbor_shell` 各残基候选 Rotamer 坐标展开为紧凑张量；
   - SIMD/SIMT 混合计算 1-body 环境势与残基对 2-body 能量，构建稀疏 Interaction Graph（IG）；
   - 片上多线程并行执行 6 态 MultiCool 模拟退火状态转移，毫秒级得出低能侧链构象。
3. **`LbfgsMinKernel`（批量约束 L-BFGS 最小化算子）**：
   - 采用基于 Cell List 的 Verlet 邻居表（9.0 Å 截断 + skin），每 50 步动态更新；
   - SIMT 并发计算全原子 `talaris2014` 解析力与 2,787 对 CA 谐波距离约束力；
   - 基于 AtomTree 前向运动学与逆向力矩规约计算 1,600 维扭转梯度；
   - 向量化维护 L-BFGS 历史位移差与梯度差（历史深度 $m=10$），结合 Armijo 非单调线搜索实现稳定收敛。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | :---: |
| Ascend 950PR（单卡，`DAV_3510`）+ 24 核 CPU（`taskset -c 0-23`） | √ |

## 算子约束限制

1. **CPU 算力硬限制**：主机侧必须严格限制在 24 核 CPU 内运行（`taskset -c 0-23` 或 `--cpuset-cpus=0-23`），严禁私自开启超核进程池；
2. **协议与力场红线**：严禁修改 `talaris2014` 力场、不得调低 `nstruct=35` / `trials=35000` / 8 Å 邻域 / 最小化收敛阈值、不得删减官方核心计算步骤；
3. **解离态冻结约束**：四态解离计算时，移开链后严禁对解离构象重新执行 repack 或 minimize，保证分子内能量在双差中精确抵消；
4. **输出自洽性硬指标**：输出 `results.csv` 必须为严格 22 列，全量 50 突变无缺失，残差严格满足 `abs(ddG_residual) <= 0.01`，且无任何 NaN/Inf 异常值。

# 可维可测分析

## 精度标准/性能标准

### 精度验收标准

以官方 48 核 CPU 开源实测输出 `test_cases/bd623_reference.csv` 作为参考真值，按参考信号强度分层验收：

| 样本分组 | 样本定义与数量 | 误差度量公式 | 验收通过判据 |
| --- | --- | --- | --- |
| **大效应样本** | $\lvert \overline{\Delta\Delta G}_{\text{ref}, i} \rvert > 1.0\ \text{REU}$（共 **15** 个） | 相对误差 $\text{rel\_err}_i = \dfrac{\lvert \overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref}, i} \rvert}{\lvert \overline{\Delta\Delta G}_{\text{ref}, i} \rvert}$ | **中位数** $\text{median}(\text{rel\_err}) < 5\%$ |
| **小效应样本** | $\lvert \overline{\Delta\Delta G}_{\text{ref}, i} \rvert \le 1.0\ \text{REU}$（共 **35** 个） | 标准差归一化偏差 $\sigma_i = \dfrac{\lvert \overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref}, i} \rvert}{\text{std}_{\text{ref}, i}}$ | **全部样本** $\lvert \overline{\Delta\Delta G}_i - \overline{\Delta\Delta G}_{\text{ref}, i} \rvert < 1.0 \cdot \text{std}_{\text{ref}, i}$ |
| **完整性与自洽性** | 全部 **50** 个突变 | 完成度、轨迹数、步数、残差 | 50/50 全部完成；`nstruct_done == 35`；`backrub_trials == 35000`；`abs(ddG_residual) <= 0.01`；无 NaN/Inf |

### 性能验收标准

| 评测环境与硬件配置 | 评测测试集规模 | 48 核 CPU 标杆耗时 | 端到端总墙钟门槛 | 相对加速比要求 |
| --- | --- | --- | --- | --- |
| **Ascend 950PR（单卡）+ 24 核 CPU** | BD-623 全部 50 突变（1,750 轨迹） | 55,653.4 s（15.46 小时） | **`wall_time_s <= 5,565.3 s`**（≤ 92.8 分钟） | **≥ 10.0×** |

> **设计期预估**：经 Stage 0 全局复用、片上 UB 局域 MC 增量计算与批处理 L-BFGS 加速，端到端墙钟预计在 **1,500 ~ 2,800 秒（~25~47 分钟）**，加速比达 **20× ~ 37×**，相对官方验收红线拥有充足安全余量。

## 兼容性分析

1. **命令行与脚本兼容性**：完全兼容基线评测脚本 `test_cases/baseline/06b_ddG_flex_antigen.py` 的参数语义，仅将 CPU 进程数参数映射为 `--cpu-cores`（固定 24），并增加 `--summary` 与 `--device-id` 控制；
2. **输出规范兼容性**：输出 `results.csv` 格式严格对齐任务书 §2.4.2 规定的 22 列，`run_summary.json` 包含完整的环境与硬件执行元数据，100% 兼容官方自动验收脚本 `test_cases/verify_bd623.py` 的校验逻辑；
3. **环境与软件栈兼容性**：自包含力场解析与核心计算算子，不依赖外部不可控的第三方私有包，兼容 CANN 9.0.0+ 与通用 Linux 异构环境；
4. **交付规范兼容性**：严格遵循昇腾社区交付规范，代码与交付文档落位于指定目录，临时工作目录与大体积中间轨迹 `.db3` 文件不入库，保证交付包纯净。
