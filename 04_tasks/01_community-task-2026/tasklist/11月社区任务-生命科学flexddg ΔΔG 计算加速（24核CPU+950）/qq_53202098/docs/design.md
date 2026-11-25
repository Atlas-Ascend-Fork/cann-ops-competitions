# 需求背景（required）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 11月社区任务-生命科学flexddg ΔΔG 计算加速（24核CPU+950） |
| 提交账号 | qq_53202098 |
| 适配硬件 | Ascend 950PR（单卡）+ 24 核 CPU |
| CANN 版本 | ≥ 9.0.0（开发/自验环境按实机记录） |
| 标杆基线 | RosettaCommons/rosetta `2026.29` / `80a0635615` + Kortemme-Lab/flex_ddG_tutorial 官方 `ddG-backrub.xml` |
| 评测 Case | BD-623（PDB `7E7Y`，420 残基）全部 50 个抗原侧单点突变 |
| 文档状态 | 设计文档提交 |

## 背景介绍

### Rosetta Flex ddG 端到端加速

Rosetta Flex ddG 是蛋白质-蛋白质结合界面突变自由能变化（ΔΔG）预测的主流蒙特卡洛系综协议（Barlow et al., *J. Phys. Chem. B* 2018）。协议基于开源 Rosetta 与 Kortemme 实验室 `flex_ddG_tutorial` 官方协议文件 `ddG-backrub.xml`，核心阶段包括：

1. 结构准备与初始约束最小化（`cb8` / 8 Å bubble + `neighbor_shell`，CA 距离约束 + L-BFGS）
2. Backrub / Sidechain 蒙特卡洛采样（每突变 `nstruct=35`，每轨迹 `backrub_trials=35000`，`mc_kt=1.2`）
3. WT / MUT 后处理：repack（`UseMultiCoolAnnealer states=6`）+ 约束最小化，以及 mutate packing + 约束最小化
4. 四态接口打分与 ZEMu GAM 重加权，输出 `ddG_mean(REU)`、7 个分子间能量项与 `ddG_gam_mean(kcal/mol)`

### 开源 CPU 基线现状

在 48 核 CPU（2×Xeon Platinum 8360Y）上，对 BD-623 全部 50 突变（共 1,750 条轨迹）开源 Flex ddG 实测：

| 统计项 | 基线数值 |
| --- | ---: |
| 累计核时 | 2,671,363.3 s（742.05 CPU·h） |
| 50 突变整批 48 核墙钟 | **55,653.4 s（15.46 h）** |

纯 CPU 端到端代价过高，难以支撑高通量界面突变扫描。

### 本任务目标

面向**整模块端到端异构加速**（非单一 aclnn / torch 算子），在 **24 核 CPU + 单卡 Ascend 950PR** 上完成与官方协议语义等价的端到端实现，并以 BD-623 全部 50 突变实测结果验收精度与性能。

# 需求分析（required）

## 需求描述

基于 Ascend C / Triton Ascend / CANN 软件栈结合 C++/Python 异构调度，开发 Rosetta Flex ddG 端到端加速模块 `rosetta_flex_ddg`，在不改变官方协议语义与 `talaris2014` 力场精度前提下，完成 BD-623 全部 50 突变端到端计算，并通过随包 `verify_bd623.py` 验收。

## 需求拆解

1. **协议等价覆盖**：完整覆盖 `ddG-backrub.xml` 核心阶段（初始最小化、Backrub/Sidechain MC、WT/MUT 后处理、四态打分与 GAM 重加权），不得删减阶段或调低采样/收敛预算。
2. **力场红线**：保持 `talaris2014`（等价 `-corrections::restore_talaris_behavior true`），不得替换为 REF2015 或删减能量项。
3. **统一入口**：提供 `run_flexddg_npu.py`（或等价入口），参数对齐基线 `06b_ddG_flex_antigen.py`，并将 `--nproc` 改为 `--cpu-cores`，新增 `--device-id` / `--seed` / `--summary`。
4. **输出规范**：写出 22 列 `results.csv` 与含 `wall_time_s` 等字段的 `run_summary.json`；`nstruct_done==35`、`backrub_trials==35000`、`abs(ddG_residual)<=0.01`。
5. **精度验收**（相对 `bd623_reference.csv`）：
   - 大效应（`|ref_mean| > 1.0 REU`，15 个）：相对误差中位数 `< 5%`
   - 小效应（`|ref_mean| ≤ 1.0 REU`，35 个）：全部满足 `|pred-ref| < 1.0 * std_ref`
6. **性能验收**：`taskset -c 0-23` 严格限核，端到端 `wall_time_s ≤ 5,565.3 s`（相对 48 核基线加速比 ≥ 10.0×）。
7. **资源约束**：全程主机侧固定绑定 24 核，不得超核或跨机并行规避；单卡 950PR。

# 详细设计（required）

## 模块分析

### 四态 ΔΔG 与 GAM

解离链按 `--move-chain`（BD-623 为抗原链 `A`）：

```text
ΔΔG = (E_bound_mut + E_unbound_wt) - (E_bound_wt + E_unbound_mut)
```

对 7 个分子间项（`fa_atr, fa_rep, fa_sol, fa_elec, hbond_sc, hbond_bb_sc, hbond_lr_bb`）做 ZEMu GAM：

```text
gam(x, term) = -exp(a) + 2 * exp(a) / (1 + exp(-x * exp(b)))
ddG_gam = sum_term gam(ddG_term, term)
```

### 协议常量（不可下调）

| 项 | 值 |
| --- | --- |
| 力场 | `talaris2014` |
| bubble / Neighborhood | `cb8` / 8.0 Å |
| `nstruct` | 35 |
| `backrub_trials` | 35000 |
| `mc_kt` | 1.2 |
| packer extras | `-ex1 -ex2`，`extrachi_cutoff=0` |
| 最小化 | `lbfgs_armijo_nonmonotone`，`tolerance=1e-6`，`max_iter=5000`，`abs_score_convergence_threshold=1.0` |
| CA 约束 | `coord_dev=0.5`，`max_distance=9` |

### 输入 / 输出

- 输入：`bd623_mutations.csv`（6 列）、`PDB/BD-623.pdb`、官方 `ddG-backrub.xml`
- 输出：`results.csv`（22 列，与输入一一对应）、`run_summary.json`（含 `wall_time_s, mutations_completed, cpu_cores_limit, npu_device, cann_version, seed, git_commit, command`）

## 实现方案

### 整体异构架构

采用 **Host CPU（C++/Python 调度）+ NPU（Ascend C / Triton Ascend / CANN Kernel）** 混合工程：

| 层级 | 职责 |
| --- | --- |
| Host 调度 | 参数校验、PDB/突变装载、`taskset` 24 核绑定、轨迹调度、结果汇总与落盘 |
| Host 协议控制 | 邻域构造、突变位点校验、轨迹级状态机、随机种子（`--seed`） |
| NPU 热点下沉 | Backrub/Sidechain 相关能量评估、成对非键/氢键项、可向量化的打分与批量化最小化内核 |
| 异步流水 | 跨突变/跨轨迹批量化；Host 准备下一轨迹数据与 NPU 计算重叠，降低端到端墙钟 |

热点剖析预期（相对开源 CPU 基线）：能量评估与 Backrub MC 步内打分占比最高，其次为 L-BFGS 约束最小化与 repack annealer；设计上将可向量化、可批处理的能量与局部更新优先下沉 950PR AI Core，Host 保留协议语义与不可向量化控制流。

### Host 侧设计

1. **入口**：`pipeline/run_flexddg_npu.py`，参数见任务书 §2.3 / §2.4；非法输入（缺 PDB、`wt_aa` 不匹配、`--cpu-cores>24`、`--nstruct<=0` 等）返回非零退出码。
2. **限核**：启动即校验 `cpu_cores==24`，并以 `taskset -c 0-23` / cpuset 保证真实生效。
3. **轨迹调度**：50 突变 × 35 轨迹；在 24 核约束下做有界并发（不得超核），优先跨突变批量化喂给 NPU 驻留流水线。
4. **数值一致性保障**：能量项集合与 `talaris2014` 对齐；对 7 项双差做残差自洽检查 `abs(ddG_mean - sum(7_terms)) <= 0.01`；支持 `--seed` 复现排查（不要求跨跑 bit-exact）。
5. **摘要计时**：从读入到 `results.csv` 落盘计入 `wall_time_s`（含参数准备/算子加载/采样/后处理/落盘；不含一次性环境安装与源码编译）。

### Kernel / NPU 侧设计

1. 将高频成对相互作用（LJ/静电/溶剂/氢键相关可向量化路径）与局部坐标更新批处理到 AI Core。
2. Tiling：按残基邻域 / 原子对块均分到多核；UB 内 CopyIn → Compute → CopyOut；必要时 double buffer 隐藏搬运。
3. 最小化与 annealer：能下沉的梯度/能量批量内核走 NPU，Host 负责线搜索与收敛判定，保证 `max_iter` 与 `abs_score_convergence_threshold` 语义不降低。
4. 不做“缩短 trials / 降 nstruct / 缩邻域”的假加速；加速来自异构并行与算子优化。

### 目录与交付形态（合入目标结构）

```text
rosetta_flex_ddg/
├── README.md
├── build.sh
├── env/
├── pipeline/          # run_flexddg_npu.py 等
├── kernels/           # NPU 算子与原生扩展
└── tests/             # 单测与端到端自验
```

私仓验收目录另按模板提供 `task_submission/`（自验证说明、精度/性能报告与日志）；大体积 `.db3` 不入库。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（单卡）+ 24 核 CPU | √ |

## 约束限制

1. 不得更换 `talaris2014`、不得调低 `nstruct/backrub_trials/8Å/max_iter/convergence`、不得删减官方 XML 核心阶段。
2. 验收全程 `taskset -c 0-23`（或等价 cpuset），禁止后台超核。
3. 蒙特卡洛系综以统计精度标准验收，不以逐比特确定性为门禁。
4. 合入代码不得包含本地绝对路径、临时轨迹与编译产物。

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 完整性 | 50/50 完成；`nstruct_done==35`；`backrub_trials==35000`；`ddG_all` 35 个有限值；`abs(ddG_residual)<=0.01`；无 NaN/Inf | 任务书 §3.2 |
| 大效应精度 | 15 个大效应位点相对误差中位数 `< 5%` | 任务书 §3.2 / `verify_bd623.py` |
| 小效应精度 | 35 个小效应位点全部 `\|pred-ref\| < 1.0 * std_ref` | 任务书 §3.2 / `verify_bd623.py` |
| 性能 | `wall_time_s <= 5565.3`（≥10.0× vs 48 核 55653.4 s） | 任务书 §3.3 |
| 负向用例 | 缺 PDB/缺列/`wt_aa` 不匹配/`cpu-cores>24`/`nstruct<=0` 等非零退出 | 任务书 §3.5 |

自验命令：

```bash
taskset -c 0-23 python3 pipeline/run_flexddg_npu.py \
  --selected test_cases/bd623_mutations.csv \
  --pdb-dir test_cases/PDB \
  --xml test_cases/ddG-backrub.xml \
  --out ./out_bd623/results.csv \
  --summary ./out_bd623/run_summary.json \
  --work /tmp/flexddg_npu_work \
  --move-chain A --nstruct 35 --trials 35000 \
  --cpu-cores 24 --device-id 0 --seed 111111

python3 test_cases/verify_bd623.py \
  --pred ./out_bd623/results.csv \
  --ref test_cases/bd623_reference.csv \
  --summary ./out_bd623/run_summary.json \
  --json-out ./out_bd623/verify_report.json
```

## 兼容性分析

新整模块端到端加速交付，不替代既有 TBE 图算子；接口以任务书 CLI 为准，与基线脚本协议参数对齐。
