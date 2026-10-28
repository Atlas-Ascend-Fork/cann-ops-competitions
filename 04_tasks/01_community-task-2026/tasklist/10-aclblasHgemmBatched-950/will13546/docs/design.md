# 需求背景

## 需求来源

CANN 社区任务 2026 年 10 月任务：在 Ascend 950PR 上使用 Ascend C 开发半精度批量矩阵乘算子 `aclblasHgemmBatched`，补齐 ops-blas 仓 Level-3 GEMM 族的 FLOAT16 批量能力，验收通过后合入 `cann/ops-blas` 主仓。

对标接口为 cuBLAS `cublasHgemmBatched`，参数序列一一对应。

## 背景介绍

### 算子功能

对每个 batch i，原地计算：

```
C_i = alpha * op(A_i) * op(B_i) + beta * C_i ，i ∈ [0, batchCount)
```

其中 `op(X)` 由 transa/transb 决定（N 不转置，T 转置，C 共轭转置；FLOAT16 为实数，T 与 C 等价）。A_i、B_i、C_i 均为 FLOAT16 矩阵，按列主序（Column-Major）存储；`Aarray`、`Barray`、`Carray` 是指向 batchCount 个矩阵首地址的设备侧指针数组，各矩阵前导维分别为 lda、ldb、ldc。alpha、beta 为 FLOAT16 标量，以 Host 指针传入，整批共用。

### ops-blas 仓现状分析

ops-blas 主仓的批量 GEMM 族已具备较完整的实现基础，本算子选择复用而非重写：

| 组件 | 位置 | 现状 |
| --- | --- | --- |
| FP32 批量接口 | `aclblasSgemmBatched`（`blas/gemm_batched/arch35/gemm_batched_host.cpp`） | 已实现并合入 |
| 复数批量接口 | `aclblasCgemmBatched` | 已实现并合入，走 4M 分解复用 FP32 cube 内核 |
| cube 批量内核 | `blas/gemm_batched/arch35/gemm_batched_cube_impl.h`、`gemm_batched_kernel.cpp` | **已按 dtype 模板化**，`GbDtypeTraits<0/1/2>` 分别对应 FP32 / FP16 / BF16 |
| alpha/beta 后处理内核 | 同上 | 已有 `gemm_batched_alpha_beta_kernel_fp16` 实例化 |
| k=0 / alpha=0 的 early-exit 内核 | 同上 | 已有 FP16 实例化 |
| 归一化 tiling 结构 | `gemm_batched_tiling_data.h` | `dtypeCase` 字段已预留 `GEMM_BATCHED_DTYPE_FP16 = 1` |

也就是说，**内核侧三套 kernel 的 FP16 路径已经存在并可运行**，`GEMM_BATCHED_FP16_C0 = 16`、`GEMM_BATCHED_FP16_L0_BASE_K = 64` 等 FP16 常量也已定义。

### 缺口分析

现有实现中 FP16 能力尚未对外暴露，缺口集中在 host 侧与接口层：

1. 公共头 `include/cann_ops_blas.h` 只有 `aclblasSgemmBatched` / `aclblasCgemmBatched`，没有 `aclblasHgemmBatched` 声明；
2. 公共头与 `cann_ops_blas_common.h` 中**没有定义 FLOAT16 标量类型** `aclblasHalf`（只有 `aclblasComplex` / `aclblasDoubleComplex`）；
3. host 侧 tiling 计算 `CalcGemmTiling` 把 `dtypeCase` 硬编码为 `GEMM_BATCHED_DTYPE_FP32`，临时工作区 `PrepareTempWorkspace` 按 `sizeof(float)` 计算容量；
4. 参数校验函数 `ValidateSgemmBatchedParams` 与 FP32 标量类型绑定；
5. 缺少 FLOAT16 的测试工程与用例。

因此本算子的设计原则是：**复用已验证的批量 GEMM 内核，把 host 侧按 dtype 参数化，新增 FLOAT16 的对外入口与测试**，不新写矩阵乘算法。

# 需求分析

## 需求描述

新增句柄式 BLAS 接口：

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

通过 handle 绑定的 stream 直调 NPU kernel，不引入 950PR 私有平行接口，声明与实现与其他产品线共用。

## 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 新增 `aclblasHalf` 类型定义与 `aclblasHgemmBatched` 公共声明 | 头文件审查、接口可被外部工程 include 后编译 |
| R2 | 实现 host 入口：参数校验、no-op 短路、tiling、kernel 下发 | 负向用例返回码逐条匹配 |
| R3 | 复用 FP16 cube / alpha-beta / early-exit 三套内核，不新写算法 | 代码审查 + 精度用例 |
| R4 | 支持 transa/transb ∈ {N,T,C} 全 9 种组合、任意 m/n/k/batchCount、lda/ldb/ldc padding | 1200 条 CSV 用例 |
| R5 | 边界语义：`m=0`/`n=0`/`batchCount=0` no-op；`k=0` 逐 batch 做 `C_i = beta*C_i` | TC_ED 边界用例 |
| R6 | 精度达生态算子开源标准 FLOAT16 档 | `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-1 或 32×ULP` |
| R7 | 性能：单 case ≥ 0.4× H100，全部 case 平均 ≥ 0.5× H100 | `verify_performance.py` 双口径判定 |
| R8 | 提供 1000 条精度 + 200 条性能 CSV 用例与可复现的测试工程 | 交付件 + 自测报告 |

# 详细设计

## 算子分析

### 数学公式

对 i ∈ [0, batchCount)、r ∈ [0, m)、c ∈ [0, n)：

```
C_i[r][c] = alpha × Σ_{p=0}^{k-1} op(A_i)[r][p] × op(B_i)[p][c] + beta × C_i[r][c]
```

转置语义：

- `transa = N`：A_i 为 m×k，按列主序 `A_i[r][p]` 位于 `r + p*lda`
- `transa ∈ {T, C}`：A_i 为 k×m，`op(A_i)[r][p] = A_i[p][r]` 位于 `p + r*lda`
- B 同理；C 恒为 m×n，`C_i[r][c]` 位于 `r + c*ldc`

### 支持数据类型

| 角色 | 类型 | 说明 |
| --- | --- | --- |
| A_i、B_i、C_i | FLOAT16 | IEEE 754 binary16 |
| alpha、beta | FLOAT16（Host 指针） | 整批共用同一标量 |
| 累加 | FLOAT32 | 内核内部累加与 golden 口径一致 |

本算子不设 dtype 轴，全部固定 FLOAT16。

### 支持形状与参数约束

| 参数 | 约束 | 违反时返回 |
| --- | --- | --- |
| handle | 非空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| transa / transb | ∈ {N, T, C} | `ACLBLAS_STATUS_INVALID_ENUM` |
| m, n, k | ≥ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| batchCount | ≥ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha / beta | 非空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| Aarray / Barray / Carray | batchCount > 0 时非空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | ≥ max(1, A_i 物理行数) | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldb | ≥ max(1, B_i 物理行数) | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldc | ≥ max(1, m) | `ACLBLAS_STATUS_INVALID_VALUE` |

A_i、B_i 的物理行数由转置决定：N 时分别为 m、k；T/C 时分别为 k、n。

## 算子实现

### 总体方案

采用「内核复用 + host 参数化」的实现路径：

```
aclblasHgemmBatched (host, 新增)
  ├─ 参数校验（FLOAT16 版）
  ├─ no-op 短路（m=0 / n=0 / batchCount=0）
  ├─ early-exit 判定（k=0 或 alpha=0 → C_i = beta*C_i）
  ├─ tiling 计算（dtypeCase = FP16，FP16 常量）
  ├─ workspace 准备（指针数组 D2D 拷贝 + 临时 C 区）
  └─ kernel 下发（复用 gemm_batched 的三个 launcher）
        ├─ gemm_batched_gemm_kernel_do      (dtypeCase=FP16 → LaunchGemmFp16)
        ├─ gemm_batched_alpha_beta_kernel_do (FP16 实例化)
        └─ gemm_batched_early_exit_kernel_do (FP16 实例化)
```

### 文件与目录

按仓内同族实现的惯例组织（与 `aclblasSgemmBatched` 同目录）：

| 文件 | 职责 | 类型 |
| --- | --- | --- |
| `include/cann_ops_blas_common.h` | 新增 `aclblasHalf` 类型定义 | 修改 |
| `include/cann_ops_blas.h` | 新增 `aclblasHgemmBatched` 声明 | 修改 |
| `blas/gemm_batched/arch35/gemm_batched_host.cpp` | host 侧校验与 tiling 按 dtype 参数化，新增 FLOAT16 入口 | 修改 |
| `blas/gemm_batched/README.md` | 补充 `aclblasHgemmBatched` 章节 | 修改 |
| `test/gemm_batched/hgemm_batched/hgemm_batched_param.h` | CSV 参数结构 `HgemmBatchedParam` | 新增 |
| `test/gemm_batched/hgemm_batched/hgemm_batched_golden.h` | FLOAT32 golden、FP16 舍入与判定 | 新增 |
| `test/gemm_batched/hgemm_batched/README.md` | 复现步骤 | 新增 |
| `test/gemm_batched/hgemm_batched/CMakeLists.txt` | 测试目标注册 | 新增 |
| `test/gemm_batched/hgemm_batched/arch35/hgemm_batched_test.cpp` | GTest 用例 | 新增 |
| `test/gemm_batched/hgemm_batched/arch35/hgemm_batched_npu_wrapper.h` | 设备缓冲与指针数组封装 | 新增 |
| `test/gemm_batched/hgemm_batched/arch35/hgemm_batched_test.csv` | 1200 条用例 | 新增 |
| `test/gemm_batched/hgemm_batched/arch35/gpu_baseline.csv` | 200 条性能基线 | 新增 |

> 落位说明：任务书 §5 写的是 `blas/gemm/arch35/` 与 `test/gemm/hgemmBatched/arch35/`，但仓内同族实现（`aclblasSgemmBatched`）位于 `blas/gemm_batched/arch35/` 与 `test/gemm_batched/sgemm_batched/`。本实现与既有实现同目录，避免跨编译单元暴露内核 launcher，也避免复制一份 tiling 与校验逻辑。

### host 侧设计

#### 1. 类型定义

`aclblasHalf` 需要同时满足「Host 侧读取标量」与「与设备侧 half 二进制兼容」两个条件。方案：

```c
/*! \brief Half-precision floating point type (IEEE 754 binary16). */
typedef _Float16 aclblasHalf;
```

理由：`_Float16` 在 ASC 编译器（Clang 系）与 GCC/ARM 上均为原生类型，`sizeof == 2`，与 Ascend C 的 `half` 二进制一致，Host 与 Device 可直接按 `void*` 传递。

备选方案：若目标工具链不接受 `_Float16`，改为 `typedef unsigned short aclblasHalf;`，在 host 侧用「FP32 计算、手工做 round-to-nearest-even 转换」的辅助函数读写标量。两种方案的对外 ABI 相同（2 字节），不影响接口签名。

#### 2. 参数校验

校验顺序严格按任务书 §2.5 执行，与 `aclblasSgemmBatched` 对齐：

1. `handle` 判空 → `HANDLE_IS_NULLPTR`
2. `alpha`、`beta` 判空 → `INVALID_VALUE`（先于一切短路）
3. `m/n/k/batchCount < 0` → `INVALID_VALUE`
4. `transa/transb` 非法 → `INVALID_ENUM`
5. `lda/ldb/ldc` 小于最小前导维 → `INVALID_VALUE`
6. `batchCount > 0` 时 `Aarray/Barray/Carray` 判空 → `INVALID_VALUE`

注意顺序上的两个易错点：

- 数组指针判空**后于** m/n/batchCount 非法判定，但对 `batchCount > 0` 一律要求非空——即 `m = 0 && batchCount > 0` 时数组为空仍返回 `INVALID_VALUE`；
- `batchCount = 0` 时不校验数组指针，直接返回成功。

#### 3. 执行路径与短路

| 条件 | 行为 | 返回 |
| --- | --- | --- |
| `m == 0` 或 `n == 0` 或 `batchCount == 0` | 不访问 A/B/C，直接返回 | `SUCCESS` |
| `k == 0` 或 `alpha == 0` 且 `beta != 0` | 逐 batch 执行 `C_i = beta * C_i` | `SUCCESS` |
| `k == 0` 或 `alpha == 0` 且 `beta == 0` | 逐 batch 置零 C_i | `SUCCESS` |
| 其余 | 主路径：cube GEMM + alpha/beta 后处理 | `SUCCESS` |

`k = 0` 时不引用 A/B，符合任务书 §2.5。

#### 4. tiling 设计

复用 `GemmBatchedGemmTilingData`，FLOAT16 取值如下：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `dtypeCase` | `GEMM_BATCHED_DTYPE_FP16` (1) | 决定内核实例化 |
| `tileM` / `tileN` | 128 / 128 | 与 FP32 共用默认块 |
| `tileKChunk` | 128 | K 向分块长度 |
| L0 base K | `GEMM_BATCHED_FP16_L0_BASE_K = 64` | 由 `sizeof(DType)` 在模板内自动选择 |
| C0 | `GEMM_BATCHED_FP16_C0 = 16` | 分形对齐单位 |
| `mBlocks` / `nBlocks` | `CalcMnBlockSplit(m, n, batchCount, cubeCoreNum)` | 按任务总量选择 M/N 切分 |
| `singleCoreM` / `singleCoreN` | 单核覆盖范围，按 16 对齐后截断到 m/n | 分形对齐 |
| `totalTasks` | `batchCount × mBlocks × nBlocks` | 溢出检查，超过 UINT32_MAX 返回 `INVALID_VALUE` |
| `usedAicCoreNum` | `min(cubeCoreNum, totalTasks)` | 不空转 |

L1 使用双 bank 布局（arch35 的 L1 分为 2 个 256KB 独立 bank）：Ping 在 offset 0、Pong 在 `TOTAL_L1_SIZE/2`，每个 bank 内 A、B 连续排布（`[Ping A][Ping B] | [Pong A][Pong B]`），避免 MTE1 搬运时的 bank 冲突。

#### 5. 分核策略

- 以「任务」为调度单位，一个任务 = 一个 batch 内的一块 `singleCoreM × singleCoreN` 输出；
- 总任务数 `totalTasks = batchCount × mBlocks × nBlocks`，各核按 `balancedSplit` 均分，余数分给前若干个核；
- 核数取 `min(AIC 核数, totalTasks)`，避免小用例空转；
- 小 batch × 大矩阵时 N/M 块多、大 batch × 小矩阵时 batch 维多，两者都能填满核。

#### 6. 工作区

主路径需要两块设备侧临时空间：

1. **指针数组区**：`Aarray`/`Barray`/`Carray` 三组指针数组需按 64B 对齐拷入 workspace，供 kernel 按 batch 索引；
2. **临时 C 区**：cube kernel 先写未缩放结果，再由 alpha/beta 后处理内核与旧 C 合并。

**临时 C 区固定为 FLOAT32**，不随输入 dtype 变化：`GemmBatchedAlphaBetaOp` 内部以 `__gm__ float*` 解释该区域，用 FP32 完成 `alpha*T + beta*C` 后再一次性舍入到 FLOAT16 写回。这样整个链路只有「写回 C」一次舍入，与 golden「FP32 计算 + 写回舍入」的口径完全一致。因此现有工作区容量公式 `batchCount × n × tempRowStride × sizeof(float)` 对 FLOAT16 同样适用，无需按 dtype 改写；FP16 相对 FP32 节省的只是指针数组之外的用户矩阵本身。

### kernel 侧设计

内核沿用 `blas/gemm_batched/arch35/` 的三段式结构，FP16 走同一套模板实例化。

#### Kernel 1：cube 批量 GEMM（AIC）

- 按 `totalTasks` 下标解出 `(batchIdx, mBlockIdx, nBlockIdx)`，由指针数组取本 batch 的 A/B/C 首地址；
- 通过 `MakeGmLayout` 选择访存布局：
  - 非转置：`DNExtLayoutPtn(ld, cols)`，元素 `(i,j)` 位于 `j*ld+i`（列主序读）；
  - 转置：`NDExtLayoutPtn(rows, ld)`，元素 `(i,j)` 位于 `i*ld+j`；
- 以 16×16 分形把 A、B 搬入 L1/L0，K 向按 `tileKChunk` 切块累加；
- FP16 的 C0=16（FP32 为 8），L0 base K=64，由 `GbDtypeTraits<1>` 静态选择；
- 结果写入临时 C 区，不带 alpha/beta。

#### Kernel 2：alpha/beta 后处理（AIV）

- 按 `batchCount × n` 的列任务切分，AIV 多核并行；
- 读临时结果 `T_i`、旧 `C_i`，计算 `alpha*T_i + beta*C_i` 写回原地；
- 标量由 host 以 FP32 传入 tiling（`GemmBatchedAlphaBetaTilingData.alpha/beta` 为 `float`），kernel 内做 FP32 运算后舍入回 FP16；
- `beta == 0` 时跳过旧 C 读取，因此「beta=0 且 Carray 为 NaN」的用例不会被 NaN 污染。

#### Kernel 3：early-exit（AIV）

- 处理 `k == 0` 或 `alpha == 0`：逐 batch 计算 `C_i = beta * C_i`；
- `beta == 0` 时直接置零；
- 不访问 A/B，满足任务书「k=0 不引用 A/B」的要求。

#### 精度设计

- 内核算术全程 FP32 累加（FP16 输入提升到 FP32），仅在写回 C 时按 IEEE 754 就近舍入（round-to-nearest-even）转 FP16；
- golden 同样以 `cblas_sgemm` 逐 batch 计算 FP32 结果再舍入为 FP16，两侧舍入口径一致；
- 因此误差来源只有累加顺序差异，不引入额外的 dtype 转换偏差。

#### 性能设计

- 计算主体走 cube 单元；FLOAT16 实例使用 `GEMM_BATCHED_FP16_C0 = 16` 与 `GEMM_BATCHED_FP16_L0_BASE_K = 64` 的分形与切块常量（FP32 为 8 与 16），单次 K 向装载覆盖更长的累加链；
- L1 双 bank ping/pong 掩盖 MTE1 延迟；
- 大 batch 小矩阵时以 batch 维填满任务队列，小 batch 大矩阵时以 M/N 分块填满；
- 后处理阶段只做一遍线性读改写，不重复搬运矩阵主体。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 本期不涉及 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 本期不涉及 |

前置条件：依赖 CANN asc-devkit ≥ 9.1（与 `aclblasSgemmBatched` 相同）。

## 算子约束限制

1. 不支持 batch 间 stride 非连续访问，各 batch 矩阵按列主序连续存储，仅支持 lda/ldb/ldc 前导维语义；
2. 不支持广播：各 batch 的 A_i/B_i/C_i 为独立矩阵；
3. m/n/k/batchCount 为运行时入参，编译期 shape 未知（dynamic shape），不要求编译期特化；
4. 确定性计算不作要求；
5. `beta = 0` 时 C_i 不必是有效输入（旧值不参与计算）。

# 可维可测分析

## 测试设计

### 用例构成

`hgemm_batched_test.csv` 共 1200 条：1000 条精度 + 200 条性能。

| 类别 | 前缀 | 条数 | 覆盖点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 18 | transa×transb 全 9 组合 × 小尺寸 4³/8³，batchCount=2 |
| 尺寸扫描 | TC_MN | 47 | 1→2048 方阵 + 12 组非方阵矩形 |
| 标量 | TC_AB | 10 | FLOAT16 可精确表示的 alpha/beta 组合 |
| 前导维 | TC_LD | 15 | lda/ldb/ldc 同时 padding 与单维 padding |
| 填充值域 | TC_FL | 18 | A/B/C 分别取 6 种填充 token |
| batch 覆盖 | TC_CV | 21 | batchCount ∈ {1,2,3,4,5,8,16,32,64,128,256,512,1024} |
| 边界负向 | TC_ED | 37 | no-op、k=0 缩放、指针为空、非法枚举/前导维/负维度、空 handle、65504 边界 |
| 扩展精度 | TC_EX | 834 | 尺寸池 × 转置 × 标量 × batchCount 确定性采样 |
| 性能 | TC_PF | 200 | 4 条任务书典型 case + 小尺寸大 batch + 转置组合 + 8192² 等 |

### golden 口径

逐 batch 以 FLOAT32 调用 `cblas_sgemm`（Netlib 实现）计算，再按 IEEE 754 就近舍入转 FLOAT16，对每个 batch 的 C_i（m×n）全矩阵逐元素比对。FP16 输入先提升到 FP32 参与累加。

### 判定

- 有限值：`|actual - golden| ≤ atol + rtol × |golden|`，`atol = rtol = 2⁻¹⁰`；
- 非有限值：双方同为 NaN、同为同号 Inf、或一方 NaN 一方 Inf 视为一致；一方有限另一方非有限判失败；
- 用例通过：有限值 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-1 或 32×ULP`，且非有限值元素全部一致。

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT16 档：rtol = atol = 2⁻¹⁰，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-1 或 32×ULP | 生态算子开源精度标准 mixed_tolerance_standard |
| 性能标准 | 单 case `GPU 标杆 / NPU 实测 ≥ 0.4`；全部性能 case 倍率算术平均 ≥ 0.5 | 任务书 §3.3（L3 算子双口径） |

性能采集：msprof op 模式取单算子 task duration，warmup 5 次后连续采样 >10 次取平均。

## 兼容性分析

1. 新增接口，不修改任何既有函数签名与语义，`aclblasSgemmBatched` / `aclblasCgemmBatched` 的行为不变；
2. `aclblasHalf` 为新引入的公共类型，与既有 `aclblasComplex` 并列，不影响已有头文件使用者；
3. 复用 `gemm_batched` 内核时，FP32/BF16 实例化路径保持零改动，仅 host 侧新增 FP16 分支；
4. 测试工程按 `SOC_ARCH_DIRS` 选择架构目录，与既有算子测试的注册方式一致。
