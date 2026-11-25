# 需求背景（required）

## 需求来源

本设计对应 2026 年 11 月 CANN 社区任务“生命科学 Flex ddG ΔΔG 计算加速（24 核 CPU + 950）”。正式验收对象为 Ascend 950PR 单卡与 24 核 CPU，任务基线为 Rosetta `2026.29`（`80a0635615`）及 Kortemme-Lab `flex_ddG_tutorial` 提供的 `ddG-backrub.xml`。

## 背景介绍

Rosetta Flex ddG 通过 Backrub 与 Sidechain 蒙特卡洛系综估计蛋白质结合界面单点突变的自由能变化。BD-623 包含 50 个抗原侧单点突变；每个突变运行 35 条轨迹，每条轨迹执行 35,000 次 Backrub 采样。开源基线在 48 核 CPU 上完成全部计算耗时 55,653.4 秒，累计 2,671,363.3 CPU 秒。

计算热点集中在三个部分：

1. 每步构象提议后的局部 `talaris2014` 能量差评估；
2. MultiCool 侧链重排和约束最小化中的成对能量与梯度计算；
3. 35 条轨迹之间的调度、状态复制和结果汇总。

第一部分和第二部分具有显著的跨轨迹并行性。将局部原子对、能量项、梯度与转子候选打包为张量，适合下沉到 Ascend 950PR；输入校验、协议状态机、不可完全向量化的 L-BFGS 控制流及结果落盘保留在 Host。

## 目标与边界

- 在 BD-623 全部 50 个突变上保持 `talaris2014` 力场及官方协议语义。
- 在 24 核 CPU、单张 Ascend 950PR 上达到端到端 `wall_time_s <= 5565.3 s`。
- 35 条轨迹以统计精度验收，不要求跨运行逐比特一致。
- 驱动、固件、CANN、代码 commit、硬件型号与完整运行命令写入自验报告。

# 需求分析（required）

## 需求描述

交付一个完整的 Rosetta Flex ddG 异构加速模块，而不是单一算子。模块读取 PDB、突变 CSV 和官方 XML，将 50 个突变的准备、采样、重排、最小化、四态打分及 GAM 重加权串联为端到端流水线。

统一入口接受以下参数：

```text
--selected --pdb-dir --xml --out --summary --work --move-chain
--nstruct --trials --cpu-cores --device-id --seed
```

正式运行固定 `--nstruct 35`、`--trials 35000`、`--cpu-cores 24`、`--move-chain A`。外层使用 `taskset -c 0-23` 限核，进程启动后再次检查可用 CPU 集合。

## 输入与输出契约

输入 CSV 必须包含 `pdb_id,chain,pdb_position,icode,wt_aa,mut_aa` 六列。入口校验文件、表头、残基编号、插入编号、PDB 中的野生型残基和移动链。任一条件不满足时立即以非零状态退出。

`results.csv` 固定输出 22 列，与输入行一一对应。每行的 `ddG_all(REU)` 包含 35 个有限轨迹值，`nstruct_done=35`，`backrub_trials=35000`，七个分子间项之和与 `ddG_mean(REU)` 的残差满足：

$$ \left|\text{ddG\_residual}\right| \leq 0.01 $$

`run_summary.json` 至少包含 `wall_time_s`、`mutations_completed`、`cpu_cores_limit`、`npu_device`、`cann_version`、`seed`、`git_commit` 和 `command`，并记录解析、参数导出、采样、后处理、打分和落盘阶段耗时。

## 协议与资源约束

| 项目 | 固定条件 |
| --- | --- |
| 力场 | `talaris2014`，开启 `-corrections::restore_talaris_behavior true` |
| 邻域 | `cb8`，CB 距离 8.0 Å，并加入两侧各一个序列邻残基 |
| 轨迹预算 | `nstruct=35`，`backrub_trials=35000` |
| 采样参数 | `mc_kt=1.2`，`-ex1 -ex2`，`extrachi_cutoff=0` |
| 初始最小化 | CA 距离约束、`coord_dev=0.5`、`max_distance=9` |
| 最小化器 | `lbfgs_armijo_nonmonotone`，`tolerance=1e-6`，`max_iter=5000` |
| 收敛判定 | `abs_score_convergence_threshold=1.0` |
| 后处理 | WT/MUT 均执行 MultiCool `states=6`、约束最小化和四态接口打分 |
| 主机资源 | 24 核，外层 `taskset -c 0-23` |
| 设备资源 | Ascend 950PR 单卡 |

# 详细设计（required）

## 模块分析

### 四态接口双差

解离链由 `--move-chain` 指定，BD-623 使用抗原链 A。第 \(k\) 条轨迹的原始双差为：

$$ \Delta\Delta G^{(k)} =
\left(E_{\mathrm{bound,mut}}^{(k)}+E_{\mathrm{unbound,wt}}^{(k)}\right)
-\left(E_{\mathrm{bound,wt}}^{(k)}+E_{\mathrm{unbound,mut}}^{(k)}\right) $$

七个分子间能量项为 `fa_atr`、`fa_rep`、`fa_sol`、`fa_elec`、`hbond_sc`、`hbond_bb_sc` 和 `hbond_lr_bb`。四态相减后，只依赖单链构象的内部项相互抵消。

### ZEMu GAM 重加权

对能量项 \(t\) 和该项的拟合参数 \((a_t,b_t)\)，有：

$$ \operatorname{gam}(x,t)=-e^{a_t}+\frac{2e^{a_t}}{1+\exp(-x e^{b_t})} $$

$$ \Delta\Delta G_{\mathrm{gam}}^{(k)}=
\sum_{t\in\mathcal{T}_7}\operatorname{gam}\left(\Delta\Delta G_t^{(k)},t\right) $$

最终输出 35 条轨迹原始值的算术均值、样本标准差和 GAM 均值、样本标准差。

### 阶段映射

| 阶段 | 官方 XML 对应 | Host 职责 | 950PR 职责 |
| --- | --- | --- | --- |
| 结构准备 | `addcst`、`neighbor_shell_storer`、初始 `minimize` | PDB/XML 解析、残基校验、邻域构造、约束与控制流 | 批量距离、能量和梯度计算 |
| Backrub/Sidechain MC | `BackrubProtocol` | 随机数、提议类型与协议状态机、收敛及异常处理 | 转子候选能量、局部能量差、批量 Metropolis 所需规约量 |
| WT 后处理 | `repack`、`addcst`、`minimize` | MultiCool 状态控制、L-BFGS 线搜索与收敛判断 | 侧链组合能量、成对能量和解析梯度 |
| MUT 后处理 | `mutate`、`addcst`、`minimize` | 突变 resfile、脯氨酸闭环语义、L-BFGS 控制流 | 突变体侧链组合能量、成对能量和梯度 |
| 接口打分 | `InterfaceDdGMover`、`ReportToDB` | 四态组织、逐项双差、GAM、汇总与落盘 | 边界原子对和七个能量项的批量求值 |

## 算子实现

### 总体架构

Host 与 NPU 使用生产者-消费者流水线：

```text
Host 24 核
  ├─ 输入校验、邻域与转子参数准备
  ├─ 轨迹状态机、随机种子与负载均衡
  ├─ L-BFGS 线搜索、收敛判定
  └─ 四态双差、GAM、CSV/JSON 落盘
          │ H2D: SoA 坐标、原子类型、参数、邻域 CSR
          ▼
Ascend 950PR
  ├─ Backrub/Sidechain 批量能量与梯度
  ├─ MultiCool 侧链组合能量
  ├─ 七个分子间能量项
  └─ 批量归约与候选排序
          │ D2H: 候选能量、梯度、批次结果
          ▲
          └──────── 下一批准备与当前批后处理重叠
```

参数准备、设备计算和 Host 后处理采用双缓冲。NPU 执行第 \(i\) 批采样时，Host 准备第 \(i+1\) 批数据并处理第 \(i-1\) 批结果。Host 工作池上限与 `--cpu-cores` 一致，计算线程统一设置单线程 BLAS/OpenMP，避免后台扩核。

### Host 侧设计

1. 以一次 PDB 解析建立残基、链、原子类型、拓扑和序列邻接索引。
2. 按突变位点缓存 CB 8 Å 邻域和静态环境；每个突变仍独立生成 35 条轨迹和 35,000 步采样。
3. 为每个 `(mutation, trajectory)` 派生独立随机流，起始值由 `--seed`、突变键和轨迹编号确定。
4. L-BFGS 在 Host 执行线搜索和收敛判断，将批量能量、梯度请求发送 NPU；`max_iter=5000` 与 `abs_score_convergence_threshold=1.0` 不作局部替代。
5. NPU 输出先写入轨迹级中间摘要；全部 35 条完成后计算均值、标准差、逐项双差和 GAM。
6. 逐行完成即刷新，最终原子替换 `results.csv` 和 `run_summary.json`，避免中断时留下半行。

### 950PR 数据排布与 Kernel

局部坐标采用 `[B, 3, N]` 的 SoA 排布，\(B\) 为并行轨迹或候选数量，\(N\) 为邻域原子数。残基索引、原子类型、供受体类型和链编号打包为整数元数据；邻域使用 CSR 形式传入，避免每次能量求值扫描全复合物。

Kernel 规划如下：

| Kernel | 计算内容 | 主要输出 |
| --- | --- | --- |
| `pair_energy` | 原子对距离、平滑截断、LJ、静电、隐式溶剂和氢键几何项 | 七个能量项 |
| `pair_grad` | 七个能量项对坐标的解析梯度 | 局部梯度 |
| `rotamer_energy` | 单残基与成对转子的组合能量 | 转子能量表 |
| `mc_delta_reduce` | 当前/候选能量的分块归约 | Metropolis 输入量 |
| `four_state_reduce` | bound/unbound、WT/MUT 分批累加 | 四态七项能量 |

每个 Kernel 按原子对连续块切分，使用双缓冲搬运坐标和参数。一次装载后处理多个轨迹块，尾块通过 mask 保持无效元素不参与归约。邻域原子表只在输入变化时重建，轨迹采样期间复用。

### 协议数值一致性

1. 原子类型、LJ 参数、溶剂参数、介电参数、氢键分类参数和 `talaris2014.wts` 从指定 Rosetta 版本导出，作为构建资产随模块版本管理。
2. 单构象、单 Backrub 步和单 Sidechain 步分别与 Rosetta CPU 路径比较七个能量项和总能量差分。
3. 所有边界项按四态分别计算后再相减，避免先做局部近似导致内部项不抵消。
4. 输出前重新计算 `ddG_residual`；超过 0.01 时把该次运行标记失败并保留现场日志。
5. 4 个脯氨酸突变及大效应位点增加独立回归测试，覆盖闭环、侧链冲突和约束最小化。

### 性能设计

- 跨突变把局部能量评估组合为批次，减少小算子启动与 Host/Device 往返。
- 以残基邻域为单位重排原子，提高同一批次内局部邻接的连续性。
- NPU 执行 35,000 步采样期间保持参数常驻；Host 同时完成前后批的状态准备与后处理。
- 把轨迹计数、步数和能量项作为 `run_summary.json` 中的完整性字段，运行结束自动断言 `1750` 条轨迹和 `61,250,000` 次 Backrub 步均已调度。
- 正式计时从入口读取输入开始，到 `results.csv` 落盘结束，包含参数准备、算子加载、采样、后处理、打分和写盘。

### 工程结构

```text
rosetta_flex_ddg/
├── README.md
├── build.sh
├── env/
├── pipeline/
│   └── run_flexddg_npu.py
├── kernels/
├── tests/
└── task_submission/
```

`task_submission/` 按模板提供自验证步骤说明、精度报告、精度日志、性能报告和性能日志。`.db3`、编译产物和临时工作目录不进入 Git。

## 支持硬件

| 芯片版本 | 支持 |
| --- | --- |
| Ascend 950PR 单卡 + 24 核 Host CPU | √ |

运行环境要求 CANN 9.0.0 及以上及配套驱动、固件。

## 算子约束限制

1. 正式验收固定 `nstruct=35`、`backrub_trials=35000`、CB 8 Å 邻域、最小化上限 5000 和收敛阈值 1.0。
2. 正式验收固定 `talaris2014`，仅支持任务规定的 PDB、CSV 和 XML 输入契约。
3. Host 可见并行度不超过 24；跨机并行、超核进程池与后台线程不受支持。
4. Monte Carlo 轨迹不保证跨运行一致；相同 seed 的目标是提供可重复排查路径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 完整性 | 50/50 突变完成；每行 35 条轨迹；`nstruct_done=35`；`backrub_trials=35000`；无 NaN/Inf；残差绝对值不超过 0.01 | 任务书 §2.4.2、§3.2 |
| 大效应精度 | 15 个 \(\left|\Delta\Delta G_{\mathrm{ref}}\right|>1.0\) 样本的相对误差中位数小于 0.05 | 任务书 §3.2 |
| 小效应精度 | 35 个 \(\left|\Delta\Delta G_{\mathrm{ref}}\right|\leq1.0\) 样本全部满足 \(\left|\Delta\Delta G-\Delta\Delta G_{\mathrm{ref}}\right|<\mathrm{std}_{\mathrm{ref}}\) | 任务书 §3.2 |
| 性能 | 24 核 CPU 与 950PR 单卡端到端运行时间不超过 5565.3 秒 | 任务书 §3.3 |
| 负向边界 | 缺失 PDB/CSV/XML、缺列、野生型不匹配、链不存在、非法 `nstruct/trials/cpu-cores/device-id` 均返回非零 | 任务书 §2.4.1、§3.5 |
| 交付 | 五项模板材料齐全；私仓、分支、commit 可访问；验收说明填写完整 commit 链接 | 任务书 §4、§5 |

固定自验命令：

```bash
taskset -c 0-23 python3 pipeline/run_flexddg_npu.py \
  --selected test_cases/bd623_mutations.csv \
  --pdb-dir test_cases/PDB \
  --xml test_cases/ddG-backrub.xml \
  --out out_bd623/results.csv \
  --summary out_bd623/run_summary.json \
  --work work_bd623 \
  --move-chain A \
  --nstruct 35 \
  --trials 35000 \
  --cpu-cores 24 \
  --device-id 0 \
  --seed 111111

python3 test_cases/verify_bd623.py \
  --pred out_bd623/results.csv \
  --ref test_cases/bd623_reference.csv \
  --summary out_bd623/run_summary.json \
  --json-out out_bd623/verify_report.json
```

## 兼容性分析

模块与基线脚本的输入语义一致，只将 `--nproc` 对齐为 `--cpu-cores`，并增加 `--device-id`、`--seed` 和 `--summary`。输出字段、顺序与官方验收脚本一致。该任务是新增端到端模块，不修改已有算子接口。
