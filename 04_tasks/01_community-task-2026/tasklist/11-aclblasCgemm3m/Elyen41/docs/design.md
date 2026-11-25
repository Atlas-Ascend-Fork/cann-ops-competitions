# 需求背景（required）

## 需求来源

11 月 CANN 社区任务《aclblasCgemm3m 算子开发（A2A3）》要求在 Atlas A2/A3 上实现单精度复数三乘法通用矩阵乘。算子以 Ascend C 实现，基于 ops-blas 的句柄式接口运行，功能对标 cuBLAS `cublasCgemm3m`，CPU 精度基准采用 Netlib BLAS 3.10.0 的 `cblas_cgemm`。验收指标取自任务书第 3 节。 [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html#cublascgemm3m) · [Netlib CGEMM](https://www.netlib.org/blas/cgemm.f) · [任务发布页](https://www.hiascend.com/activities/task-center/details/cae74cb900484862a2b43b24c0f7d9d3) · [任务讨论](https://gitcode.com/cann/ops-blas/discussions/93)

## 背景介绍

`ops-blas` 的公共头文件已声明 `aclblasCgemm3m`，arch35 已支持 Ascend 950PR。本任务补齐 arch22 实现和测试。arch22 的 `aclblasCgemm` 提供复数拆包和回写参考，`aclblasSgemm` 提供 FP32 Cube 计算和尾块搬运参考。3M 用三次实数矩阵乘得到一次复数矩阵积，性能收益取决于额外拆包、加减、暂存和回写的成本。 [ops-blas](https://gitcode.com/cann/ops-blas) · [arch22 GEMM 实现](https://gitcode.com/cann/ops-blas/tree/master/blas/gemm/arch22) · [arch35 参考实现](https://gitcode.com/cann/ops-blas/blob/master/blas/gemm/arch35/cgemm3m_host.cpp)

# 需求分析（required）

## 需求描述

在 `aclblasSetStream` 绑定的 stream 上异步计算 `C = alpha * op(A) * op(B) + beta * C`，原地更新 C。A、B、C 为 Device 上的列主序 COMPLEX64 矩阵，alpha、beta 为 Host 上的 COMPLEX64 标量。A、B 各支持 N、T、C 三种操作，C 模式对输入共轭转置。入口采用 `include/cann_ops_blas.h` 中已有的声明。 [公共接口](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h)

## 需求拆解

1. 在 Host 校验句柄、转置枚举、尺寸、前导维和条件性指针，处理零维快返、置零和仅缩放。
2. 实现带 padding 的列主序索引、复数拆包、三次 FP32 实数矩阵乘、复数重组和 C 原地回写。
3. 按 M/N/K 形状、核数和片上空间选路。通过分块、数据复用和流水减少 GM 往返，覆盖小矩阵、矩形、大方阵与非对齐尾块。
4. 在 `blas/gemm/arch22/` 接入实现，在 `test/gemm/cgemm3m/arch22/` 接入测试，在父级 CMake 注册 arch22 目标并在 `blas/gemm/README.md` 补充产品支持。用 GTest/ATK、Netlib golden 和 `msprof op` 分别验证功能、精度和性能。

# 详细设计（required）

## 算子分析

### 数学公式

令 `A′ = op(A)`、`B′ = op(B)`。下表的 Ar、Ai、Br、Bi 均表示应用转置及共轭后的实部和虚部。记 `D = A′ · B′ = Dr + i·Di`。

| 路径 | P0 | P1 | P2 | Dr | Di |
| --- | --- | --- | --- | --- | --- |
| 非融合 | Ar·Br | Ai·Bi | (Ar+Ai)·(Br+Bi) | P0−P1 | P2−P0−P1 |
| 融合 | (Ar+Ai)·Br | Ar·(Bi−Br) | Ai·(Br+Bi) | P0−P2 | P0+P1 |

设 `alpha = ar + i·ai`、`beta = br + i·bi`、旧 `C = cr + i·ci`。回写的实部为 `(ar*Dr - ai*Di) + (br*cr - bi*ci)`，虚部为 `(ar*Di + ai*Dr) + (br*ci + bi*cr)`。beta 为零时直接省去整个旧 C 项。alpha 为零或 K 为零时只计算 beta 项。两条 3M 路径的预加减、点积与重组舍入不同，按路径验证。

### 支持数据类型

公开类型为 `aclblasComplex`，实部、虚部各为 float32，按公共头文件的布局访问。拆包、预加减、三路点积累加、重组及缩放均使用 FP32。Cube 路径沿用 arch22 SGEMM 关闭 HF32 的配置。 [公共类型](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas_common.h) · [arch22 SGEMM Kernel](https://gitcode.com/cann/ops-blas/blob/master/blas/gemm/arch22/gemm_kernel.cpp)

### 支持形状

`op(A)` 为 m×k，`op(B)` 为 k×n，C 为 m×n，m/n/k 为非负 `int`。以下物理形状和偏移以复数元素为单位。i、j、p 分别为 M、N、K 方向的零起始逻辑下标。

| 矩阵与模式 | 物理形状 | 前导维下限 | 逻辑元素的物理偏移 |
| --- | --- | --- | --- |
| A，N | lda×k | `max(1,m)` | `i + p*lda` |
| A，T/C | lda×m | `max(1,k)` | `p + i*lda` |
| B，N | ldb×n | `max(1,k)` | `p + j*ldb` |
| B，T/C | ldb×k | `max(1,n)` | `j + p*ldb` |
| C | ldc×n | `max(1,m)` | `i + j*ldc` |

T 和 C 的物理下标相同。C 模式在拆包后、形成 3M 线性组合前对虚部取反。只读矩阵的逻辑元素，只写 C 的逻辑 m×n 区域。

## 算子实现

### 实现方案

采用可独立验证的非融合路径提供完整功能，再以形状和设备能力选择融合路径。两条路径共享布局计算与复数回写逻辑。缩放使用独立的 Vector 路径。

#### 3.2.1 host侧设计

公共入口的参数及顺序如下。

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

| 顺序 | 校验与分流 | 结果 |
| --- | --- | --- |
| 1 | handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | transa/transb 超出 N、T、C 枚举 | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | m/n/k 为负，lda/ldb/ldc 低于对应下限，或 alpha/beta 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | m、n 均为正且 C 为空。另当 m、n、k 均为正且 alpha 非零时，任一 A/B 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | 合法调用且 m 或 n 为零 | 返回成功，无 kernel，无 A/B/C 访存 |
| 6 | 合法调用且 k 或 alpha 为零 | beta 为 1 时直接返回，beta 为 0 时写零，其余执行复数缩放。均不读 A/B |
| 7 | 其余合法调用 | 规划 workspace 和 tiling，在绑定的 stream 上下发。beta 为 0 的路径跳过 C 读入 |

alpha、beta 按值写入下发参数。访问跨度、任务数、对齐和 workspace 段偏移使用经溢出检查的 64 位计算，转换成设备描述字段前检查可表示范围。入口按 ops-blas 规范映射资源与下发错误。参数校验覆盖结构参数和指针的空值条件，Device 缓冲的有效性由调用方保证。

##### 1. 分核策略

按输出 Tile `B_M × B_N` 建立 `ceil(m/B_M) × ceil(n/B_N)` 网格，M 方向优先编号。每个核按实际 block 数步进领取 Tile，活动核数取相应 AIC/AIV 资源与有效任务数的较小者。每个输出 Tile 由唯一任务完成回写，首版优先让每个输出 Tile 按递增顺序累加完整 K，避免跨核归约和 C 写冲突。若长 K 窄矩阵实测受限，再评估 split-K。采用时须将部分积放在独立 workspace，汇总后仅一次应用 alpha 和 beta，并重新验精度。

##### 2. 数据分块和内存优化策略

从设备查询有效 AIC/AIV 数量及相关片上内存容量，按 CANN 9.1.0 的 arch22 数据搬运及 Matmul 约束枚举合法 `B_M, B_N, B_K`。对每个候选计算对齐后每个缓冲的尺寸及生命周期。AIV 的整块输入、拆包、组合、尾块和回写临时区计入 UB，A/B staging 计入 L1 和 L0A/L0B，Cube 累加器计入 L0C。只在各区余量均满足时启用双缓冲，按有效行列和 K 尾长处理余块。首轮非融合调优以现有 arch22 SGEMM 的 `64×256×32` Cube 分块作为候选，融合另选更小的 M/N 分块以预留三路累加空间。复用行主序内部 Matmul 接口时将列主序输出视作 `Cᵀ`，交换两个已规范化的操作数并核对内部 stride。

非融合路径按一批输出 Tile 规划窗口 `M_w × N_w` 和 K 面板 `K_w`，复用同一 A 或 B 面板。令 (A_p,B_p,R_p) 分别为对齐后的单路 A、B 面板和窗口输出平面的 FP32 元素数。GM workspace 预算包括 `4*(3*A_p + 3*B_p + 3*R_p)` 字节，以及 Matmul scratch、地址对齐和必要的 K 分片累加区。全 K 面板可容纳时优先一次打包。容量不足时缩小窗口或 K 面板。对分片 K，第一片初始化三路累计平面，随后将三路部分积分别累加，最后一片结束后回写 C。每次完整规划后再下发，循环复用切片由同一 stream 的先后顺序保护。

workspace 使用句柄管理，遵循用户 workspace 容量及默认 workspace 所有权规则。子阶段得到互不重叠的临时区，按仓内约定保障异步任务使用期间的生命周期。空间不足时尝试更小分块，合法下界仍不足时返回库的资源错误。正常执行路径不在 Host 同步 stream。

##### 3. tilingkey规划策略

| 路径键 | 启用条件 | 下发形式 |
| --- | --- | --- |
| `SCALE` | 无乘积且 beta 非 1 | 启动 AIV。置零与一般复数缩放分支分别处理 |
| `THREE_MM` | 常规回退。任意合法布局、形状及尾块均可规划 | AIV 打包、AIC 三乘、AIV 合成，必要时按 K 面板循环 |
| `FUSED_3M` | M/N/K 和布局适合片上复用，容量、搬运与同步前提通过校验，且该形状族已通过真机验证 | 混合核按 Tile 流水处理三路点积和合成 |

tiling 数据包含 m/n/k、三项前导维、转置与共轭位、alpha/beta、输出 Tile 与 K 面板尺寸、全块及尾块数、活动块数、workspace 段偏移和 beta 读标志。按形状、布局、硬件资源选路，候选排布和启用阈值通过实测确定。

#### 3.2.2 kernel侧设计

**初始化与搬入。** 用 block 索引映射逻辑 Tile，并计算实际 m/n/k 尾长。对物理连续区按列批量读入交错复数，到 UB 后拆成 FP32 平面。对跨列的转置区域先按物理列成块搬运，再在局部重排。按 N/T/C 解码通用索引，内部补齐区写零。以有效长度处理 GM 非对齐首尾，通过有界搬运或标量末段处理避免读写 padding。

**非融合计算。** AIV 为 A 和 B 分别打包实部、虚部及两者之和。AIC 复用 arch22 FP32 Matmul 的搬运与累加配置，从三对实平面得到 `P0, P1, P2`。完整 K 归约后，AIV 读三路结果，按非融合公式得到 D。高复用形状按输出窗口重用已打包面板，三路共享同一套地址与尾长计算。跨 kernel 的 Pack、Matmul、Combine 依赖由同一 stream 保证。

**融合计算。** AIV 从一个 K 面板派生 Ar+Ai、Bi−Br、Br+Bi 等线性组合，AIC 按融合公式更新同一输出 Tile 的三路 FP32 累加状态。按对齐后的三路 L0C 驻留区估算预算，L0A/L0B staging 随三路计算复用。AIV/AIC 按配对块使用有界 GM staging，AIV 将打包输入写入 staging，AIC 将点积 Tile 经 Fixpipe 写入 staging 后由 AIV 读入 UB 合成。用 arch22 混合核的就绪和释放信号保护每个缓冲槽，事件与搬运流水匹配后再允许覆盖。若设备组合无法维持三路累加、尾块回写或低成本交接，则 Host 选用非融合路径。融合路径以复用有限的 Tile 暂存区，减少整窗中间数据的物化和 kernel 下发为目标。

**数值风险与纠正。** 融合的 `P0−P2`、`P0+P1` 和非融合的 `P2−P0−P1` 分别监测强消去及非有限中间值。调试与验证分别统计两条路径重组前后量级，例如融合实部用 `abs(P0) + abs(P2)` 与 `abs(Dr)` 比较。采用融合纠正时，对风险元素在回写前用原始操作数重算对应 FP32 实部差积、虚部交叉积与 K 点积。以任务书精度验证消去判据和漏判情况，记录纠正率与耗时。非融合路径独立验证其消去误差和大值风险，必要时复用原始操作数重算逻辑。包含 Inf/NaN 时按目标参考的分量传播结果验证，并单独检查未读分支。

**合成与回写。** 以完成 K 归约的 D 为输入，使用 FP32 施加 alpha 与 beta。beta 非零时先保留所需旧 C 的两分量，再覆写同一元素。beta 为零时走无 C 搬入的分支。将实虚交错写回有效区域，保持每列 padding 的哨兵。仅缩放路径对 C 的 m×n 区域独立分核，置零分支直接生成零。

## 支持硬件

| 产品 | 架构与支持范围 | 验证安排 |
| --- | --- | --- |
| Atlas A2 系列，含 Atlas 800T A2 (910B3) | arch22，Ascend C，CANN 9.1.0 | 910B3 采集正式性能数据，可同时完成精度自验 |
| Atlas A3 系列，含 Atlas 800T A3 | arch22，Ascend C，CANN 9.1.0 | 使用目标机的 `ascend910_93*` SoC 名构建。精度自验可选择 A3 |

A2 构建入口为 `bash build.sh --ops=cgemm3m --soc=ascend910b3`。A3 替换为实际产品的 SoC 参数。运行时以设备资源和 arch22 API 约束配置分块及搬运，融合能力通过目标环境验证后启用。 [ops-blas 构建入口](https://gitcode.com/cann/ops-blas/blob/master/build.sh)

## 算子约束限制

| 项目 | 处理 |
| --- | --- |
| 布局与类型 | 单矩阵调用，COMPLEX64 列主序。lda/ldb/ldc 描述各列步幅和 padding，不支持任意步长视图或广播 |
| 地址与原地 | A/B/C 为 Device 指针，alpha/beta 为 Host 指针。C 写入区与 A/B 可读区不得重叠。输出即 C |
| 空输入与维度 | 支持 m/n 为零的快返和 k 为零的缩放，边界指针按 Host 校验表处理。尺寸作为运行时参数参与 tiling |
| 精度与归约 | 3M 舍入、尾块和实际归约顺序按精度阈值验收，无逐位一致要求 |
| 异步 | 用句柄绑定的 stream，下发成功表示请求已提交。调用方需维持 Device 缓冲有效直至执行完成，并在 Host 读回前同步该 stream |
| 内存 | 任务书无独立内存验收阈值。实现记录峰值 workspace 和可选路径，测试包的 Host 内存预算用于测试装配 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 判定与采集 | 标准来源 |
| --- | --- | --- |
| 精度 | Netlib BLAS 3.10.0 `cblas_cgemm` 作为参考，实部、虚部分别比对。令 `atol = rtol = 2^-13`，满足 `abs(actual-golden) <= atol + rtol*abs(golden)` 计入匹配。用例须同时满足 matched ratio ≥ 0.99 与 `max_abs_error <= max(1e-2, 32 * ULP(g_low))`。`g_low` 为最大有限误差点对应的 FP32 golden | 任务书 §3.2、生态算子精度标准 |
| 性能正式判据 | Atlas 800T A2 (910B3) 用 `msprof op` 采集 Task Duration。预热 5 次，随后取连续 10 次成功调用的均值，按同键比较 `NPU_us / 1000 <= 1.25 * gpu_ms`。多阶段路径按一次调用关联并累计各 kernel 的 Task Duration，报告各阶段和总耗时 | 任务书 §3.3、`gpu_baseline.csv` |
| 性能交叉验证 | 预置 Device 缓冲，预热 5 次，再用 `aclrtEvent` 计时连续 10 次，输出 `[PERF] case=<case_name> ... avg_us=<数值>`。排除 H2D/D2H、造数与 golden 比对 | 任务书 §3.3、`verify_performance.py` |
| 内存 | 记录设备 workspace 与自验装配峰值，用于复现及定位，按任务书 §3.4 无单独门限 | 任务书 §3.4 |

若使用等价的 FP64 累加后收窄至 COMPLEX64 的 golden，报告须标识生成路径及与直接 cblas FP32 参考的差异。CSV 的 MERE/MARE 列供测试框架配置，生态精度判定采用上述 FP32 固定门限。性能基线按转置、m/n/k、lda/ldb/ldc 及两组复标量完整匹配。记录基线版本、异常样本和缺失基线。附件中 200 组耗时已填入且键与用例一一对应。对新增带 padding 的性能组合需先采集同键 GPU 基线。 [生态算子精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)

## 测试方案与定位

- 用任务附件 `cgemm3m_test.csv` 驱动 1000 条精度与 200 条性能用例，覆盖九种转置组合、紧凑和带填充布局、非对齐与 4095/4096 大尺寸、复数标量、纯实/纯虚、均匀与高斯分布。补测任务规定但附件覆盖不足的前导维增量 +1、+8、+64，以及 Cube 分块与尾块边界。
- 用 C 含 NaN 且 beta 为零证明计算不依赖旧值，另用 padding 哨兵和访存检查验证写入范围与不读分支。用 k 为零、alpha 为零叠加空指针及 A/B 含 NaN 验证仅缩放，同时覆盖 beta 为 0、1 和一般复数。用同形状非对称复数矩阵区分 T 与 C。
- 单独测试空 handle、非法枚举、负维度、非法前导维、空标量、条件性空矩阵和快返前校验顺序。用形状和内存边界测试 checked arithmetic 与 workspace 切分。
- 在测试构建中分别覆盖两条 3M 路径，增加大幅值、强消去、NaN/Inf、K 尾块及 workspace 回退用例。日志记录版本、SoC、用例参数、路径键、tile、workspace 大小、失败分量和误差。用 profiler 核对快返无 kernel、缩放只启动 Vector 任务，并分解打包、Cube 和回写的实际开销。

## 兼容性分析

保持现有 BLAS 接口签名、状态码、标量位置、C 原地语义和 stream 模型。新实现由 arch22 构建选中，arch35 的 950PR 实现和测试继续按其原有架构注册。arch22 按本任务要求以 `ACLBLAS_STATUS_INVALID_ENUM` 报告非法转置枚举，其余非法数值按 `ACLBLAS_STATUS_INVALID_VALUE` 报告。cuBLAS 参考用其默认 Host pointer mode，转置、尺寸、前导维和复数输入按同一参数组合构造。

本次提交为设计阶段，算子实现、路径阈值和 Atlas 真机精度及性能均待后续验证。
