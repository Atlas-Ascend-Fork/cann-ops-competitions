# 需求背景

## 需求来源

2026 年 11 月 CANN 社区任务 `aclblasCgemm3m`（Atlas A2/A3）要求使用 Ascend C，在 `ops-blas` 的句柄式 BLAS 框架下实现单精度复数三乘法通用矩阵乘法。接口对标 `cublasCgemm3m`，验收以任务书及配套测试数据为准。 [昇腾任务页面](https://www.hiascend.com/activities/task-center/details/cae74cb900484862a2b43b24c0f7d9d3) · [官方任务书及自验包](https://www.hiascend.com/p/resource/202611/21b52bc81a1f43b0b5f59e0f337e9665.zip) · [任务讨论](https://gitcode.com/cann/ops-blas/discussions/93) · [cuBLAS 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm3m)

## 背景介绍

基线为 `ops-blas` 主分支提交 `55ef791b77f3677b45da3468b0a8b17941424f84`。`include/cann_ops_blas.h` 已声明 `aclblasCgemm3m`。`blas/gemm/arch35/` 提供 Ascend 950PR 实现，`test/gemm/cgemm3m/arch35/` 提供对应测试。Atlas A2/A3 使用 arch22，可参考现有 `blas/gemm/arch22/cgemm_host.cpp`、`cgemm_kernel.cpp` 的复数解交织、共轭、输出缩放及尾块处理，以及 `gemm_host.cpp`、`gemm_kernel.cpp` 的 FP32 Cube 分块。 [ops-blas 基线](https://gitcode.com/cann/ops-blas/commit/55ef791b77f3677b45da3468b0a8b17941424f84)

# 需求分析

## 需求描述

计算 `C = alpha * op(A) * op(B) + beta * C`。A、B、C 以交织的 complex64 元素按列主序存放在 Device。`alpha`、`beta` 指向 Host 的 `aclblasComplex`（实部和虚部各为 float32）。`op` 支持 N、T、C，依次表示原矩阵、转置、共轭转置。C 原地输出，调用方保证 C 与 A、B 的存储互不重叠。算子通过 handle 绑定的 stream 异步下发。

| 对象 | 逻辑形状 | 实际列主序布局及前导维下界 |
| --- | --- | --- |
| `op(A)` | `m × k` | N 时 `lda × k`，`lda ≥ max(1,m)`。T/C 时 `lda × m`，`lda ≥ max(1,k)` |
| `op(B)` | `k × n` | N 时 `ldb × n`，`ldb ≥ max(1,k)`。T/C 时 `ldb × k`，`ldb ≥ max(1,n)` |
| `C` | `m × n` | `ldc × n`，`ldc ≥ max(1,m)`，仅写 `row < m` 的元素 |

## 需求拆解

1. 保持公开接口 ABI，覆盖九组转置、带填充的前导维和结构参数错误，正确处理空输出、零 K 及复数零标量。
2. 在 arch22 上完成三次实数矩阵乘法、复数缩放与回写，并根据规模和片上容量选择执行路径。
3. 集成现有构建及 ATK/GTest 测试，覆盖官方 1,000 条精度用例与 200 条性能用例，记录真机数据与可复现命令。

# 详细设计

## 算子分析

设 `op(A) = Ar + i*Ai`、`op(B) = Br + i*Bi`，其中实虚平面已含 N/T/C 的坐标映射及共轭符号。逻辑下标为 `(i,p)` 和 `(p,j)` 时，A 在 N 下的元素下标为 `i + p*lda`，在 T/C 下为 `p + i*lda`。B 在 N 下的元素下标为 `p + j*ldb`，在 T/C 下为 `j + p*ldb`。C 的下标为 `i + j*ldc`。

直接分阶段路径采用 `P0 = Ar*Br`、`P1 = Ai*Bi`、`P2 = (Ar+Ai)*(Br+Bi)`，乘积实部为 `P0-P1`、虚部为 `P2-P0-P1`。融合候选采用 `P0 = (Ar+Ai)*Br`、`P1 = Ar*(Bi-Br)`、`P2 = Ai*(Br+Bi)`，乘积实部为 `P0-P2`、虚部为 `P0+P1`。这里每项 `*` 表示实数矩阵乘法。两种分解的舍入路径各自验证。最后对乘积和 C 旧值做复数乘标量，并按逻辑范围写回 C。

### 对标算子与测量流程

```text
官方用例 → 核对转置、m/n/k、lda/ldb/ldc 和复数 alpha/beta
├─ 非法参数 → 校验 Ascend 接口返回状态并留档
└─ 合法参数 → 准备列主序 A/B/C 及 Host 标量
   ├─ GPU：cuBLAS cublasCgemm3m
   │  → 按 N/T/C 取 op(A)、op(B) → Gauss 3M → alpha/beta 更新 C
   │  → 以 GPU kernel 耗时形成对应参数的基线 gpu_ms
   ├─ CPU：Netlib cblas_cgemm 或冻结 golden → 精度参考 C
   └─ NPU：GTest 调用 aclblasCgemm3m
      ├─ 精度 → 同步并读回 C → 对照参考 C 的实部和虚部
      └─ 性能 → 预热 5 次，记录后续 10 个成功调用
         ├─ msprof op → 每次调用汇总必经 kernel 的 Task Duration
         │  → 取均值并换算为 ms → 同一参数键判定 NPU ms ≤ 1.25 × gpu_ms
         └─ aclrtEvent → 连续下发事件均值 avg_us，用于交叉验证
```

精度 golden 采用 Netlib BLAS 3.10.0 或任务书允许的 float64 累加后收窄至 complex64，并在报告中记录口径。性能对比使用附件中对应参数的 GPU kernel 基线，并在预置的 Device 缓冲上测量 NPU kernel 耗时。

## 算子实现

### Host 侧设计

Host 依次校验 handle、转置枚举、m/n/k、lda/ldb/ldc 及适用的指针。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法转置返回 `ACLBLAS_STATUS_INVALID_ENUM`，负维度、非法前导维或必需空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。结构参数有效后按下表调度。

| 条件 | 指针要求及动作 |
| --- | --- |
| `m == 0` 或 `n == 0` | alpha、beta 必须有效，A/B/C 可空，直接返回成功，不下发 |
| `k == 0` 或 `alpha == 0+0i` | A/B 可空，C 必须有效。`beta == 1+0i` 直接返回成功，`beta == 0+0i` 下发逻辑区域清零，其余 beta 下发复数缩放 |
| 普通 3M | A/B/C 均有效。在绑定 stream 下发计算。`beta == 0+0i` 走仅写 C 的后处理分支 |

Host 将标量按值传入 kernel，使用 64 位检查矩阵范围、对齐和工作区偏移，检查乘加溢出后才转换为字节数。按设备查询 AIC/AIV 核数和可用片上空间，按形状、输入步长、可用块数及工作区容量决定分块。工作区沿用 handle 管理，预先计算并划分对齐的互不重叠区域。用户提供的空间不足时选择满足容量且已验证的路径，否则返回对应分配失败状态。需要委托 SGEMM 时，将其限制在当前调用的独立 scratch 切片内。各阶段按同一 stream 顺序下发，分配失败、下发失败沿用仓内状态码，检测到下发失败即停止排队后续阶段。测试核查连续调用、工作区增长和不同 handle 并行的生命周期。

### Ascend C 执行流程

```text
入口 → Host 校验 handle、转置、维度、前导维和标量指针
├─ 校验失败 → 返回对应错误状态
└─ 校验通过
   ├─ m == 0 或 n == 0 → SUCCESS，无 kernel 下发
   └─ 非空输出 → 校验 C
      ├─ k == 0 或 alpha == 0 → 按 beta 处理 C
      │  ├─ beta == 1 → SUCCESS，无 kernel 下发
      │  ├─ beta == 0 → AIV 清零逻辑 C
      │  └─ 其余 beta → AIV 复数缩放逻辑 C
      └─ 普通 3M → 校验 A/B → 分块、工作区检查及路径选择
         ├─ 分阶段基线 → AIV 解交织、共轭、填零和派生平面
         │  → AIC 三次 FP32 实数 GEMM → AIV 组合、缩放和交织回写
         ├─ 短规模候选 → 单次 AIV 完成三组点积及回写
         └─ 通用及长 K 候选 → AIV 面板打包 → AIC 分块累加及可选归约
            → AIV 组合、缩放和交织回写
```

空输出时 A/B/C 均可为空。按 beta 处理 C 时 A/B 可为空，且不访问 A/B。`beta == 0` 时写回不读旧 C，各写回路径都保留 `ldc` 填充区。Host 在绑定的 stream 上按序下发，失败则返回错误并停止后续下发，成功入队即返回 `SUCCESS`。调用方同步 stream 后读取 C。

cuBLAS 与 Ascend C 路径均以 Gauss 3M 计算矩阵乘积，再应用复数 alpha/beta。cuBLAS 由 GPU 库调度，Ascend C 分阶段基线明确将打包和后处理交给 AIV，将实数矩阵乘交给 AIC。文中两条优化候选的分派条件由真机试验确定。

### Kernel 侧设计

1. **分阶段基线**。AIV 融合解交织、共轭符号、边界填零及派生平面的生成。复用 arch22 FP32 实数 GEMM 计算 P0/P1/P2，AIV 读结果做组合、缩放和交织写回。以 N/T 原布局或可直接用于 Cube 的打包布局为起点，使用窄 K 块验证结果。现有 arch22 `Cgemm` 委托 `aclblasSgemmEx` 执行 FP32 实数乘积，并将 K 块限制到 1,024。对中小 M/N、K 超过 1,024 的形状首先复测 1,024/1,025/1,536 边界，并在完整链路验证后再采用更长 K。 [关联问题](https://gitcode.com/cann/ops-blas/issues/1077)
2. **短规模路径候选**。以单次 AIV 下发完成三组短点积和复数回写。对 K 短且输出稀疏的形状，按输出列或输出元素分配任务，连续区间向量化，按每个核的逻辑输出范围处理尾部。对 `1³` 至 `64³` 及短 K 矩形用精度与耗时对照确定切换区间。单核处理的行片段、向量寄存器数量与 UB 空间共同限制每块尺寸。
3. **通用及长 K 路径候选**。AIV 以整列或 K 面板打包，将原布局与共轭映射合并进一次搬运。AIC 按 `(tileM,tileN)` 分发输出块，K 面板在块内连续累加，复用经验证的 FP32 Cube 分块。优先试验在同一次 AIC 下发中计算三组乘积，并将同一输出块的结果写为相邻工作区面板，由 AIV 一次读回和交织写出。再评估混合 AIC/AIV kernel 的合法同步、片上中间结果驻留及搬运计算双缓冲。若片上容量限制同时保留三组乘积，沿用有界 GM 面板，仅在 AIV 回写后复用。对于输出块不足以占满 Cube 的长 K 矩形，比较板内顺序累加与分核 K 后归约，并将归约时间和工作区计入整体指标。

片上空间按所选 tile 的输入、输出、对齐尾块及缓冲份数逐项计算。整列连续区使用批量搬运，尾段搬运与写回以实际逻辑长度为限，保证无需接触 C 的前导维填充区。输入解交织过程保持 FP32，Cube 显式禁用 HF32 作为精度基线。若大标量或抵消用例超出限值，先对照两种分解、K 分段与加和顺序定位误差，再验证 FP32 补偿组合或局部重算。将额外工作纳入同一正式性能计时。Inf/NaN 用例按 golden 分类检查位置与类型。

### 性能预算与试验

根据官方 `gpu_baseline.csv` 计算典型用例的预算。`6mnk` 仅估算三次实数 GEMM 的浮点运算量，表中吞吐按总预算折算，实际计算还要给打包、归约和后处理留出时间。

| 转置 | m/n/k | GPU 基线（µs） | NPU 总 kernel 预算（µs） | 最低等效吞吐（TFLOP/s） |
| --- | --- | ---: | ---: | ---: |
| NN | 256/256/256 | 26.010 | 32.512 | 3.10 |
| NN | 1024/1024/1024 | 458.752 | 573.440 | 11.24 |
| NC | 2048/2048/2048 | 3388.928 | 4236.160 | 12.17 |
| CN | 4096/4096/4096 | 22039.347 | 27549.184 | 14.97 |

- 最小 `1³` 的预算为 `9.856 µs`。对 `1³` 至 `64³` 先比较空下发、单次 AIV 和分阶段基线，再验证将中间结果保留在单核的收益。
- `4096³` 若完整物化 A/B 各三个实数平面和三个乘积平面，工作区为 `9 × 4096² × 4 = 576 MiB`，另有实数 GEMM scratch。比较完整打包与面板打包的 GM 流量、峰值工作区、AIC/AIV 时间和阶段间空隙。测试覆盖 `4095³`、长 K 瘦矩阵、宽矩阵、转置和非零 beta。大方阵测量三次独立 FP32 SGEMM 的总 kernel 时间，先判定计算部分能否落在总预算内，再评估打包、AIV 后处理及 K 分片的余量。`4096³` 对应三次同等成本的调用均摊上限约 `9.18 ms`，实际可分配给每次 GEMM 的时间更少。
- 在 910B3 上对比候选路径。优先记录用例名、转置和标量、下发次数、每阶段耗时、峰值工作区和精度。以可重复的实测结果选择阈值与路径，阈值仅依赖运行时形状、布局、标量模式及硬件资源。表格为官方 GPU 基线推导的预算，各 NPU 路径和阈值待真机实测。

## 支持硬件

| 产品 | 实现目标与验证计划 |
| --- | --- |
| Atlas A2（含 Atlas 800T A2 / 910B3） | arch22。使用 CANN 9.1.0，在 910B3 运行精度和正式性能自验 |
| Atlas A3（含 Atlas 800T A3） | arch22。按设备对应 `ascend910_93*` SoC 构建。有设备时复测精度和异步行为，任务书允许精度自验覆盖其中一种款型 |

## 算子约束限制

每次调用处理一组矩阵，以 `lda/ldb/ldc` 描述的列主序及填充为布局合同。调用方负责矩阵的有效存储、不同操作数间的读写范围及数据生存期，并在读取 C 或回收其存储前同步 stream。运行时参数和片上资源决定分块及归约顺序。

# 可维可测分析

## 精度标准与性能标准

| 项目 | 验收及证据 |
| --- | --- |
| 精度 | 使用任务书允许的 Netlib BLAS 3.10.0 `cblas_cgemm` 或按 float64 累加后收窄至 complex64 的冻结 golden，记录所采用的口径。对 C 各分量检查 `abs(actual-golden) ≤ atol + rtol*abs(golden)`，FP32 的 `atol=rtol=2^-13`，实部与虚部分别要求匹配率至少 99%，最大绝对误差满足 `max(0.01, 32 ULP)`。非有限值按任务书规则单独判断。 [混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 正式性能 | Atlas 800T A2（910B3）预热 5 次，`msprof op` 采集后续 10 个成功调用，对每次调用汇总必经 kernel 的 `Task Duration`，按同一参数键与 GPU kernel-only `gpu_ms` 对照。每个用例的平均 NPU kernel 耗时须 `≤ gpu_ms × 1.25`，汇总记录每项及平均性能倍率，保留异常样本说明和 profiler 原始记录。 |
| 端到端交叉验证 | 同样预热 5 次，`aclrtEvent` 包围预置 Device 缓冲上的 10 次下发，输出 `[PERF] case=<name> ... avg_us=<value>`。事件计时与正式 kernel 数据分别留档。 |

测试将复用 `test/gemm/cgemm3m/` 的参数、golden 和校验结构，在 `arch22/` 增加 ATK/GTest 入口并注册 `test/gemm/cgemm3m/CMakeLists.txt`。官方 CSV 覆盖九组转置、均匀/高斯输入、`4095³/4096³`、大复数标量、边界与 200 个性能参数键。配套 `gpu_baseline.csv` 的 200 条 `gpu_ms` 均为正值，转置、尺寸、前导维和复数标量组合与性能用例一一对应。自验另检查非法枚举与负维度、零维复合条件、零 beta 与 NaN 旧 C、零 alpha 与空或 NaN A/B、共轭与非共轭的不同结果、`+1/+8/+64` 填充哨兵、K 分块边界、重复调用与异步执行。用例包按任务书保留 `cases.json`、`perf/`、冻结 `inputs/`、`golden/`、`facts.json` 和可复现说明。

### 官方用例的一致性核对

当前附件中有 60 条性能用例预期 `SUCCESS`，但前导维低于任务书 §2.4 的下界，其中 35 条涉及 lda、30 条涉及 ldb、5 条同时涉及两者。`TC_PF_0005` 在 NN、`m/n/k=4096/64/64` 时给出 `lda=64`，下界为 4,096。`TC_PF_0007` 在 NN、`m/n/k=64/64/4096` 时给出 `ldb=64`，下界也为 4,096。附件中的 GPU 耗时按这组原始参数登记。

Host 按任务书校验并对这些参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。保留 200 条原始参数及校验记录，先用 140 条合法用例评价可比较的性能。其余 60 条已在任务讨论中反馈，等待维护者确认用例口径。若采用修正后的前导维，需要使用按修正后完整参数重新采集的 GPU 标杆，才能计入正式性能结果。精度用例 `TC_ED_411` 的 `alpha=0`、A/B 为 NaN、`beta=2`，按任务书只读取并缩放 C。 [任务讨论](https://gitcode.com/cann/ops-blas/discussions/93)

构建入口为 `bash build.sh --ops=cgemm3m --soc=ascend910b3`。集成时核对选择算子能找到 arch22 的 `blas/gemm/arch22/` 实现及 `test/gemm/cgemm3m/arch22/` 测试，并验证按 SoC 选取的 target 和执行目录。自验报告记录 CANN、驱动、设备、源代码 commit、命令、测试结果及性能原始数据，验收材料同步到 `task_submission`。 [交付流程](https://www.hiascend.com/developer/activities/cann-community-task#cann-task-all)

## 兼容性分析

新增 arch22 的 `aclblasCgemm3m` 定义，沿用 `include/cann_ops_blas.h` 的现有签名及项目的状态码和 stream 模型。注册 arch22 测试时保留 arch35 测试和开发套件门控，复测 `aclblasCgemm`、`aclblasSgemm` 的原有入口。更新 `blas/gemm/README.md` 和 API 支持表，按产品说明接口的 arch22/arch35 可用性。
