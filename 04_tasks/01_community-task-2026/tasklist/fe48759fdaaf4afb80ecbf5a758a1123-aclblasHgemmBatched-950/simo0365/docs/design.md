# aclblasHgemmBatched 算子设计

本文按[社区任务官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)编写。

| 项目 | 信息 |
| --- | --- |
| 任务名称 | 10月社区任务-aclblasHgemmBatched算子开发（950） |
| 任务链接 | [社区任务详情](https://www.hiascend.com/activities/task-center/details/fe48759fdaaf4afb80ecbf5a758a1123?menu=tasks) |
| 任务 ID | `fe48759fdaaf4afb80ecbf5a758a1123` |
| GitCode 账号 | `simo0365` |
| 个人代码仓 | [simo0365/ops-blas](https://gitcode.com/simo0365/ops-blas) |
| 开发分支 | `feat/hgemm-batched-950` |
| 实现版本 | [dad1aefe0d615efa8cca00566a3a522b6af62a9e](https://gitcode.com/simo0365/ops-blas/commit/dad1aefe0d615efa8cca00566a3a522b6af62a9e) |

详细测试数据和原始日志见配套自测报告。

## 1. 需求背景（required）

### 1.1 需求来源

社区任务要求在 Ascend 950PR 上使用 Ascend C 实现半精度实数批量矩阵乘 `aclblasHgemmBatched`，补齐 ops-blas 的 Level-3 GEMM 能力。接口参数及数学语义对齐 cuBLAS `cublasHgemmBatched`，软件基线为 CANN 9.1.0。

### 1.2 背景介绍

任务基线公共头文件尚无该接口，需要新增统一的 `aclblasHgemmBatched` 声明，并复用 `blas/gemm/arch35/` 的 GEMM batch kernel 直调框架。调用方通过 handle 绑定 stream，传入设备端矩阵指针数组，单次调用完成多个独立 GEMM。

矩阵和 alpha/beta 标量均使用 FLOAT16 存储。每批矩阵按列主序保存，支持独立地址及 leading dimension；不同批次不要求连续，也不能由第一批地址加固定 stride 推导。FLOAT16 为实数，因此共轭转置 C 与转置 T 等价。

## 2. 需求分析（required）

### 2.1 需求描述

- 支持 N/T/C 的九种组合、非方阵、非对齐尾块以及合法的 `lda/ldb/ldc`。
- `Aarray/Barray/Carray` 及其中的矩阵地址位于 Device；`alpha/beta` 为 Host 标量指针。
- `m/n/k/batchCount` 为运行时参数。`m=0`、`n=0` 或 `batchCount=0` 在参数校验后成功返回；`k=0` 仅执行 C 的 beta 缩放。
- 复用公共 handle、状态码、stream 和 workspace，通过 Ascend C kernel 直调异步执行。
- 覆盖任务随附 1,000 条精度用例和 200 条性能用例，按任务规定精度阈值、单例性能倍率及平均性能倍率验证。

### 2.2 需求拆解

| 开发项 | 实现内容 |
| --- | --- |
| 公共接口 | 在 `include/cann_ops_blas_common.h` 增加 `aclblasHalf` binary16 存储类型，在 `include/cann_ops_blas.h` 增加统一 API 声明 |
| Host | 参数校验、half 标量解码、路径选择、分核 tiling 和 workspace 容量管理 |
| 通用 GEMM | 复用同族 `gemm_batched_cube_impl.h` 半精度流水线，使用 FP32 累加 |
| 标量融合 | `alpha=1,beta=0` 直接输出 half；一般标量先保留 FP32 乘积，再完成缩放和 half 舍入 |
| 小矩阵与退化分支 | 小矩阵 SIMT 或设备内合并 Cube 路径；`k=0/alpha=0` 使用 scale 路径 |
| 自测 | CSV 驱动 GTest、Netlib FP32 golden、逐批精度检查、padding/guard 检查及独立性能采集 |

## 3. 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

对批次 `b ∈ [0,batchCount)`：

```text
C_b = alpha × op(A_b) × op(B_b) + beta × C_b
C_b(r,c) = alpha × Σ[p=0..k-1] op(A_b)(r,p) × op(B_b)(p,c)
           + beta × C_b(r,c)
```

其中 `r ∈ [0,m)`，`c ∈ [0,n)`，`op(X)` 为 X 或 X 的转置。`k=0` 时只计算 `beta × C_b`；`beta=0` 时不读取旧 C，直接写入乘积结果或零。

#### 3.1.2 接口定义与支持数据类型

```cpp
aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const* Aarray, int lda,
    const aclblasHalf* const* Barray, int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const* Carray, int ldc,
    int batchCount);
```

| 参数或计算阶段 | 数据类型 | 内存位置及说明 |
| --- | --- | --- |
| A/B/C 矩阵 | IEEE 754 binary16（FLOAT16） | Device，C 原地输出 |
| alpha/beta | `const aclblasHalf*`，指向 FLOAT16 标量 | Host，不能为空 |
| Aarray/Barray/Carray | Device 指针数组 | 每个元素为对应批次矩阵的 Device 地址 |
| m/n/k、lda/ldb/ldc、batchCount | `int` | Host 参数 |
| 乘积累加、标量运算及临时矩阵 | FLOAT32 | 计算完成后舍入为 FLOAT16 |

`aclblasHalf` 是 `uint16_t` 的 binary16 位存储别名。Host 按符号位、指数和尾数解码为 FP32，覆盖次正规数、Inf 和 NaN；例如 half 数值 1.0 的编码为 `0x3c00`，不能用整数值 1 代替。

#### 3.1.3 支持形状与布局

| 矩阵 | N 时物理行 × 列 | T/C 时物理行 × 列 | leading dimension 下限 |
| --- | --- | --- | --- |
| A | m × k | k × m | `max(1, transa=N ? m : k)` |
| B | k × n | n × k | `max(1, transb=N ? k : n)` |
| C | m × n | 不适用 | `max(1,m)` |

`m/n/k/batchCount ≥ 0`，不要求编译期固定 shape。任务测试覆盖单维达到 16384、batchCount 达到 65536 的边界；实际可执行规模受设备内存和地址容量限制。列主序元素 `C(row,col)` 地址为 `Cbase + col*ldc + row`，leading dimension 中的 padding 不属于逻辑输出，保持原值。不支持广播或超出 leading dimension 语义的任意 stride。

### 3.2 算子实现

#### 3.2.1 实现方案

Host 仅校验顶层指针和参数；设备代码读取各批指针槽，不进行 D2H 指针表复制，也不在 Host 逐批调用 GEMM。所有计算通过 handle stream 发射。

| 条件 | 执行路径 | 临时矩阵 |
| --- | --- | --- |
| `k=0` 或 `alpha=0`，且 `beta=1` | 成功返回，C 保持原值 | 无 |
| `k=0` 或 `alpha=0`，其余 beta | AIV scale；`beta=0` 直接置零 | 无 |
| 紧凑 NN 16³/32³，`alpha=1,beta=0`，高批次且可整组 | 设备内合并小矩阵，Cube 只输出各批对角结果 | 无 |
| m/n/k 均不大于 16 的其他场景 | fused SIMT，FP32 累加并融合标量缩放 | 无 |
| 其他形状，`alpha=1,beta=0` | Cube FP32 累加，Fixpipe 输出 half | 无 |
| 其他形状及标量 | Cube 输出 FP32 workspace，AIV epilogue 输出 half | 有 |

公开列主序接口通过 `Cᵀ = op(B)ᵀ × op(A)ᵀ` 映射到同族行主序 Cube 模板，交换 m/n、A/B leading dimension 和转置标记，无须显式转置输入输出。`HgemmBatchedTilingData` 同时保存公开 m/n/ldc 与变换后的 Cube tiling，epilogue 使用公开坐标。

#### 3.2.2 Host 侧设计

**参数校验。** 依次检查 handle、转置枚举、非负维度及标量非空、leading dimension、必要的顶层数组，随后处理 no-op。错误返回分别为 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`、`ACLBLAS_STATUS_INVALID_ENUM` 或 `ACLBLAS_STATUS_INVALID_VALUE`。`batchCount>0` 时 Carray 必须非空；Aarray/Barray 在 `batchCount>0 && k>0` 时必须非空。`batchCount=0` 不检查数组；`m=0/n=0` 不省略此前的参数校验。

**分核策略。** 常规路径联合划分 batch 与 M/N 空间，taskId 按实际核数执行 grid-stride 循环。Cube/Vector 核数由平台运行时查询。并行度充足时，一个 Cube task 对应一个输出 tile；分块边界按 16 元素单位均衡，减少细碎尾块。大矩阵按 batch 聚集相邻空间任务以复用 L2，较小 tile 网格优先轮换 batch。

**数据分块和内存规划。** 输出 tile 使用 128×128、256×128 或 256×256，L0 baseK=64，L1 K chunk=256。M/N 均至少 256，且 batch 已覆盖 Cube 核或完整空间网格可提供至少两波工作时优先采用 256×256；否则根据并行度选用 256×128 或 128×128。网格不足或索引乘积有溢出风险时，退回基于核数与 batchCount 最大公约数的粗粒度分块，并使用 128×128 双缓冲路径。

一般标量路径的 FP32 workspace 按列主序保存，`tempRowStride=ceil(m/16)*16`，每批占用 `tempRowStride*n*sizeof(float)` 字节。复用 handle 的有效 workspace，容量不足一批时由现有管理函数扩容，多批按容量分块。`batchStart` 指向原设备指针表起点，当前分块临时区从零批开始。Cube 与 epilogue 在同一 stream 顺序执行，后续分块复用临时区，无额外 Host 同步。

**路径与 tiling 参数。** 本实现采用 kernel 直调，不注册图算子的 tilingkey。Host 根据尺寸和标量选择 scale/small/compute/epilogue 入口；`HgemmBatchedTilingData` 携带尺寸、转置标志、分块、`balancedSplit`、`batchStart`、标量和临时行跨度。任务数超出索引容量时分批发射。

#### 3.2.3 Kernel 侧设计

Cube 复用 `GbResolveTaskCoordinates/Bounds`、`GbProcessMNTiles<1>`、`GbProcessKL1Loop<1>` 及半精度 trait。设备读取 `uint64_t` 指针槽获得当前批次的 A/B/C 地址，执行以下流水线：

```text
GM half A/B → tensor_api Copy → L1 NZ/ZN
            → CopyL12L0A/B  → L0A/L0B half
            → Mmad FP32    → L0C float
            → Fixpipe      → C half（alpha=1,beta=0）
                          或 workspace float（一般标量）
workspace float + 可选旧 C half → AIV FP32 alpha/beta 运算 → C half
```

L1 与 L0A/B 使用 ping-pong。128×128×256 分块的 A/B 合计占 128 KiB，256×256×256 时占 256 KiB，均位于单个 L1 bank 内。L0A/B 每份 256×64 half 为 32 KiB，分别置于各自 64 KiB 存储的两个半区。较小输出使用两个 L0C bank；256×256 float 输出占满 256 KiB L0C，固定使用起始地址 0，由 `FIX_M0/M_FIX0` 保护计算与搬出，Host 保证每任务只处理一个输出 tile。尾块由既有 tensor_api Copy/Mmad 路径处理。

AIV epilogue 每 tile 最多处理 4096 个元素，使用双缓冲输入/输出队列及 FP32 计算区，设计分配上限约 96 KiB。旧 C 先提升为 float，再计算 `alpha*temp + beta*C`，最后按 ties-to-even 舍入 half。`beta=0` 不读取旧 C，scale 的对应分支直接生成零。DataCopyPad 处理非 32 字节对齐尾块；`ldc=m` 且临时区无列 padding 时展平批次，否则按列切块，避免写入 padding。

小矩阵 SIMT 按输出元素分配工作项，在 K 方向 FP32 累加并融合标量缩放。对 NN、leading dimension 等于尺寸、`alpha=1,beta=0`、batchCount 至少 128 的紧凑 16³/32³，分别每八批或四批在 L1 纵向拼接左矩阵、横向拼接右矩阵。一次 128×128×K 计算产生跨批交叉乘积，仅将对角上的各批结果搬回其独立 C 地址。16³ 要求批次数为 8 的倍数，32³ 要求为 4 的倍数，其余使用通用路径。该路径用少量额外 Cube 运算换取较少的逐批任务设置，无 GM 临时矩阵。

### 3.3 支持硬件

| 支持的芯片版本 | 支持情况 | 软件及编译基线 |
| --- | --- | --- |
| Ascend 950PR | 支持，已在目标设备自测 | CANN 9.1.0；arch35 / DAV_3510；`--npu-arch=dav-3510` |
| 其他芯片版本 | 本次未验证 | 无对应验收证据 |

实现沿用仓库 ASC/CXX 构建框架和 tensor_api 兼容处理，构建方式如下：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --soc=ascend950 --ops=hgemmBatched
```

公共接口位于 `include/cann_ops_blas.h`，Host/Kernel 实现位于 `blas/gemm/arch35/`。测试构建和运行步骤见 `test/gemm/hgemmBatched/README.md`。

### 3.4 算子约束限制

- alpha/beta 必须指向有效 Host half 标量；指针数组及有效矩阵位于 Device。
- 各批 C 的逻辑输出应互不重叠；不保证跨批输入输出重叠时的计算顺序。
- 仅支持列主序及 leading dimension 语义，不支持广播、稀疏布局或任意额外 stride。
- 使用 FP32 累加后舍入为 half，按混合容差判定，不要求 bit-exact。
- 容量和地址计算使用 64 位中间量，Host 拒绝工作区字节数溢出；实际规模受设备内存限制。

**k=0 材料差异的处理。** 原始 `TC_ED_138` 设置 `m=n=8,k=0,batchCount=2,beta=0`，Aarray/Barray 顶层指针为空，期望 SUCCESS；任务书 §2.5/§3.5 的文字则要求正 batch 时三个数组均非空。本实现按随附测试的退化语义处理：k=0 允许 Aarray/Barray 为空，不引用 A/B，仍要求正 batch 的 Carray 非空及 alpha/beta 非空。原 CSV 输入和 SUCCESS 期望保持不变。

## 4. 可维可测分析

### 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 全部转置组合、独立批次指针、leading dimension、标量分支、no-op、k=0 和异常参数行为 | 任务书 §2、§3.5 及随附 CSV；k=0 差异见 §3.4 |
| 精度标准 | FP32 Netlib `cblas_sgemm` 逐批 golden，再就近舍入 half；每批有限值 matched ratio ≥0.99，同时满足绝对误差上限；非有限值单独验证 | 任务书 §3.2、[生态混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 每例 `GPU_us/NPU_us ≥0.4`；全部 200 例倍率算术平均值 ≥0.5，两项同时满足 | 任务书 §3.3、随附 `gpu_baseline.csv` 和性能校验脚本 |
| 采样标准 | `msprof op` task duration，5 次 warmup 后连续 10 次有效样本取均值；SUCCESS 调用计入样本 | 任务书 §3.3 |

有限值逐元素条件为 `abs(actual-golden) ≤ 2^-10 + 2^-10*abs(golden)`，每批 matched ratio 至少 0.99，并要求各有限值误差不超过 `max(0.1,32*ULP(golden))`。NaN/NaN、同号 Inf/Inf、NaN/Inf 按任务规则等效；有限值与非有限值配对失败。非有限元素不参与有限值统计，另记录总数及通过数，要求全部满足特殊值规则。

GPU 基线 `gpu_ms` 乘 1000 转为微秒；平均倍率为全部性能用例 `GPU_us/NPU_us` 的算术平均值。

### 4.2 测试覆盖与复现

测试使用仓库 C++ GTest 框架直接调用公共 API。保留原始 CSV 的 1,000 条精度及 200 条性能用例，以及 200 行 GPU 基线，按原 case 名称报告结果。golden 使用 Netlib BLAS 3.10.0（测试环境包版本 `3.10.0-2ubuntu1`），逐批生成并检查全部 `m*n` 逻辑元素。Device 指针数组指向独立、带间隔和 padding 的矩阵，验证逻辑输出以及分配 guard。

覆盖 N/T/C 九种组合、非方阵与尾块、标量特殊值、leading dimension、非法参数、零维和 k=0。补充 m/n/k 各达到 16384 的低内存边界、batchCount=65536、half 转换，以及有效设备指针数组内 A/B 元素为空的 k=0 用例。精度运行共 1006 个 GTest（1,000 个原始用例、5 个补充用例、1 个转换检查）。

原始随机模式与 `HGB_DISTRIBUTION=mixed` 分开运行、报告。混合模式使用可复现的均匀 `[-5,5]` / 正态分布（μ∈[-5,5]、σ∈[0.1,2]）；仅随机扩展类 `TC_EX` 的 alpha/beta 按各 50% 分布重新生成，固定和边界标量保持 CSV 原值。golden 使用实际 half 转换后的标量。混合模式属于追加分布覆盖，不将变更后的标量称为原始 CSV 参数。

性能路径不计算 CPU golden。大规模性能输入使用固定 8 条工作线程及独立、确定性的种子填充，全部填充和上传在计时前完成；精度模式的输入生成流程不变。ACL event 仅用于初步诊断，GTest 总运行时间不作为 kernel 耗时。

### 4.3 性能采集与当前证据

正式性能应由 `msprof op` 获取 task duration；一次调用存在 Cube 和 epilogue 多任务时需归并该调用的任务耗时，保留有效样本、均值及每例倍率。记录设备、驱动、CANN、编译选项和测试负载。

本次环境的 `msprof op` BasicInfo 采集报 `StartFFTSTask Profiling channel start failed`，缺少 `op_basic_info/DeviceProf1`，未获得有效正式样本。工具可能返回零退出码，因此采集结果还需检查有效任务数和非零时长。该正式证据仍待补齐。

现有普通 msprof `op_summary` 数据由 `profile_tasks.py` 解析，筛选 `hgemm_batched_cube` / `hgemm_batched_small`，要求 200 个原始性能用例按顺序对应 3,000 个任务，每例前 5 次预热、后 10 次取均值。这一任务与调用的一一对应仅适用于本批 `alpha=1,beta=0` 的原始性能用例。结果明确标记 `NORMAL_MSPROF_TASK_DURATION`，可作为性能诊断，不能替代要求的 `MSPROF_OP_TASK_DURATION` 正式验收证据。

### 4.4 兼容性分析

新增接口复用公共 `aclblasHandle_t`、状态码和 stream/workspace 管理，不定义 950 私有平行 API。实现复用同族 GEMM 模板，Sgemm/Cgemm 等现有调用入口保持原有接口。新增测试通过仓库 GTest 机制注册，同族算子回归用于检查共享模板变更的影响。当前硬件支持结论限定于已测试的 Ascend 950PR。
