# aclblasSgemmBatched / aclblasHgemmBatched 算子设计文档（Atlas A2/A3）


# 需求背景（required）

## 需求来源

- 任务书：`aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md`。
- 目标仓 ops-blas。目标硬件 Atlas A2/A3 系列（arch22）。性能验收覆盖任务书规定的 A2 款型，自验证覆盖一种款型。
- 对标接口：S 为 `cublasSgemmBatched`，H 为 `cublasHgemmBatched`。
- 依据优先级：任务书 > `include/cann_ops_blas.h` 与 `include/cann_ops_blas_common.h` > 仓内 arch35 的 S/H host 与 arch22 的 `aclblasSgemm` > cuBLAS 公开文档与 Netlib `sgemm.f`。
- 符号：任务书写两个声明已在 `include/cann_ops_blas.h`，行号分别为 606 和 616。当前头文件与该位置、该签名一致。`aclblasHalf` 为 `uint16_t`。本任务沿用这两条公开 ABI，不新增符号，也不定义 arch22 私有接口。

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta, float* const Carray[], int ldc, int batchCount);

aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha,
    const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb,
    const aclblasHalf* beta, aclblasHalf* const Carray[], int ldc, int batchCount);
```

## 背景介绍

两个接口计算一批共享转置、维度、前导维和标量的独立 GEMM：

```text
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  i = 0 .. batchCount-1
```

矩阵为列主序。S 的 alpha、beta、A、B、C 是 FLOAT32。H 的对应量是 FLOAT16。二者都是实数算子，`ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 相同，不做共轭。Aarray、Barray、Carray 是 Device 上的指针数组。alpha、beta 指向 Host 内存。只写每个 C[i] 的逻辑 m×n，不改 ldc 的 padding。

当前基线：

- `blas/gemm_batched/arch35/gemm_batched_host.cpp` 已实现 `aclblasSgemmBatched`，文件说明面向 arch35，kernel 使用 tensor_api。
- `blas/gemm_batched/arch35/hgemmBatched_host.cpp` 已实现 `aclblasHgemmBatched`。Cube 把 FLOAT16 输入累加到 FP32。`alpha == 1` 且 `beta == 0` 时直接收窄写入 C；否则先写 FP32 临时矩阵，再由向量 epilogue 做标量并收窄回 FLOAT16。标量经 `HalfToFloat` 再与 0、1 比较。
- `blas/gemm_batched/README.md` 将 S 与 H 的 Atlas A2、Atlas A3 标为不支持。H 的 950DT 也标为不支持。
- `blas/gemm_batched/arch22/` 已有 `aclblasCgemmBatched`。那是复数四实数乘加，不是本任务的 S/H。
- `blas/gemm/arch22/gemm_host.cpp` 已有 arch22 `aclblasSgemm`：按转置选择 Cube kernel，向量侧做打包、累加和 epilogue。仓内 arch22 没有 half GEMM kernel。

arch35 的 tensor_api 与其 tiling 头里的 L1 双 bank 布局属于 DAV_3510。本任务在 arch22 上另写 S/H，不编译那些 arch35 kernel 源文件，也不把复数 batched 的四平面拆分用于实数 S/H。

arch35 的公共入口已有本任务需要的控制流：`m = 0`、`n = 0` 或 `batchCount = 0` 直接成功返回；`k = 0` 或 `alpha = 0` 走 `C = beta * C`，且 `beta == 1` 时不再启动 kernel（`GemmBatchedLaunchEarlyExit`）。非法 trans 返回 `ACLBLAS_STATUS_INVALID_ENUM`。与任务书不同的一点是：`batchCount > 0` 时 arch35 在 quick return 之前就要求 Aarray、Barray、Carray 都非空，因此 `m = 0` 或 `n = 0` 时空数组、以及 `k = 0` 或 `alpha = 0` 时的空 A/B，都会先返回 `ACLBLAS_STATUS_INVALID_VALUE`。任务书允许这两种空指针。arch22 按任务书实现，不回改 arch35。

本任务不涉及 TBE、OPP、Python 或 ATen。

| 项 | 当前基线 | 本任务 |
| --- | --- | --- |
| 接口 | `cann_ops_blas.h` 已有 S 与 H 声明 | 沿用声明，为 arch22 提供定义 |
| 硬件 | README：S 在 950 支持，A2/A3 不支持；H 在 950PR 支持，A2/A3 与 950DT 不支持 | A2/A3 增加 S 与 H。950 源码与 950 支持行保持不动 |
| 功能 | arch35 已有 FP32 与 FP16 指针数组批量 GEMM。空数组规则严于任务书 | arch22 拟按任务书语义实现 S 与 H |
| 测试 | `test/gemm_batched/sgemm_batched/` 只有 arch35 源，CMake 注册 `sgemm_batched_test`。参数结构只有 FLOAT32 | 拟新增 arch22 测试。S/H 分文件提供用例，由同一 GTest 按算子分派 |

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22，dav-2201）上用 Ascend C 实现已声明的 `aclblasSgemmBatched` 与 `aclblasHgemmBatched`。handle 绑定 stream，kernel 异步执行。每个 batch 是列主序 GEMM，batch 之间共享 trans、m、n、k、lda、ldb、ldc、alpha、beta，矩阵地址互相独立。S 为 FLOAT32，H 为 FLOAT16。`OP_C` 与 `OP_T` 等价。`m = 0`、`n = 0` 或 `batchCount = 0` 直接成功返回，此时三个指针数组可以为空。`k = 0` 或 `alpha = 0` 只做 `C = beta * C`，此时 Aarray、Barray 可以为空；`m > 0` 且 `n > 0` 时 Carray 必须非空。`beta = 0` 不读 C 的旧值。`beta = 1` 且不做乘加时不启动 kernel。本提交范围不含 arch35 kernel 改动、复数 GEMM、跨距批量接口、广播、动态 shape 推导和 Python 入口。

## 需求拆解

1. 接口和 ABI：链接已有的两个符号。返回 `aclblasStatus_t`。
2. 功能、dtype、shape、format：S 为 FLOAT32，H 为 FLOAT16。op(A) 为 m×k，op(B) 为 k×n，C 为 m×n，列主序 ND。前导维下界见「算子约束限制」。
3. Host 校验和错误行为：非法 trans 返回 `ACLBLAS_STATUS_INVALID_ENUM`。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。阶数、batchCount、前导维和结构空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。不扫描 Device 指针数组的元素，不检查 C[i] 重叠。
4. Kernel、Tiling、分核：乘加走 arch22 Cube，标量与写回走向量 epilogue。Host 按 dtype 与转置选择 kernel 函数。不使用 TilingKey。
5. 边界与 quick return：`m = 0`、`n = 0`、`batchCount = 0`，以及 `k = 0` 或 `alpha = 0` 且 `beta = 1`，不启动 kernel。其余 `C = beta * C` 只写逻辑 m×n。
6. 精度和性能验证：S 按 FLOAT32 档、H 按 FLOAT16 档做混合容差。正式性能判据是 msprof 的 kernel task duration，不超过任务随附 GPU 标杆的 1.25 倍。NPU 耗时待测。
7. 兼容与回归：不修改 arch35 源码，不修改 arch22 的 `aclblasCgemmBatched`。同一次构建只编入当前 `SOC_ARCH_DIRS`。
8. 上层入口：验收入口就是这两个 aclblas 函数。不涉及 aclnn、TBE 或 PyTorch。

# 详细设计（required）

## 算子分析

### 语义

对每个 batch i，在 n > 0、m > 0、k > 0 且 alpha ≠ 0 时：

```text
op(X) = X           当 trans = ACLBLAS_OP_N
op(X) = X 的转置     当 trans = ACLBLAS_OP_T 或 ACLBLAS_OP_C
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
```

`OP_C` 不做共轭。H 的乘加在 FP32 累加域中进行，写回时收窄到 FLOAT16。K 方向的分块顺序不要求逐位确定，验收使用任务书的容差，不用 bit-exact。

`beta = 0` 时不读 C 的旧值，逻辑 m×n 按 `alpha * op(A) * op(B)` 写出。`k = 0` 或 `alpha = 0` 时不读 A、B：`beta = 0` 把逻辑区域写成 0，`beta = 1` 保持 C 不变，其他 beta 做 `C = beta * C`。ldc 上超出 m 的 padding 不写。

NaN 与 Inf 按 IEEE 乘加传播。任务书把它们列为测试输入，不把某一种 NaN/Inf 互相替代规定为算子语义。

### 存储、workspace、异步

| 参数 | 内存 | dtype | 布局 |
| --- | --- | --- | --- |
| handle | Host | `aclblasHandle_t` | 标量，携带 stream |
| transa、transb | Host | `aclblasOperation_t` | N / T / C |
| m、n、k、lda、ldb、ldc、batchCount | Host | int | 标量 |
| alpha、beta | Host | S=FLOAT32，H=FLOAT16 | 标量指针 |
| Aarray、Barray | Device | 指针数组，元素为 S 或 H | 列主序，带 lda、ldb |
| Carray | Device，原地写回 | 指针数组，元素为 S 或 H | 列主序，逻辑 m×n，物理行距 ldc |

物理行数：transa 为 N 时 A 的行数是 m，否则是 k；transb 为 N 时 B 的行数是 k，否则是 n。C 的行数是 m，列数是 n。

workspace 只在需要 FP32 临时积或向量 epilogue 时使用。拟走库属 workspace：`EnsureDefaultWorkspace` 与 `GetEffectiveWorkspace`，与 arch22 `aclblasSgemm` 相同。算子内不另做 `aclrtMalloc`。长度在实现时按 tile、batch 和元素宽度计算。不足时返回该辅助函数已有的失败码。`alpha == 1` 且 `beta == 0` 的直接写回，以及 `beta == 1` 的纯退化，不依赖这块临时积。

kernel 挂在 `handle->stream` 上。接口返回前不同步。调用方读回 C 之前自行同步。

## 算子实现

arch22 目录里已有复数 batched 源文件，没有 S/H 批量源文件。下面 Host 契约按任务书写。调度层次参考 arch35 的 S/H host，kernel 体使用 arch22 已有的 Cube 与向量阶段，不把 arch35 的 tensor_api 源文件拿来编译。

### 总体架构

```text
aclblasSgemmBatched / aclblasHgemmBatched
  └─ Host：校验 → quick return，或 C = beta * C，或提交批量 GEMM
       ├─ m = 0 或 n = 0 或 batchCount = 0：直接成功返回
       ├─ k = 0 或 alpha = 0：beta = 1 时直接成功返回，否则只缩放逻辑 C
       └─ 其余：按转置选择 Cube 乘加，再按 alpha、beta 决定是否做向量 epilogue
```

没有 Python / ATen 层。不使用 TilingKey。分支是 Host 上的 dtype、转置和标量判断。

### Host

拟新增 `blas/gemm_batched/arch22/` 下的 S、H host 文件，不写入 `cgemm_batched_host.cpp`。校验顺序按任务书，两种 dtype 相同：

| 顺序 | 条件 | 行为 |
| --- | --- | --- |
| 1 | handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | transa 或 transb 不是 N、T、C | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | m、n 或 k 小于 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | lda、ldb、ldc 低于下界 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | batchCount 小于 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | alpha 或 beta 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | 指针数组本体 | 见下段 |
| 8 | 结构合法后的 quick return | `m = 0` 或 `n = 0` 或 `batchCount = 0` 时返回 `ACLBLAS_STATUS_SUCCESS`，不启动 kernel |

第 7 步：`batchCount = 0`，或 `m = 0`，或 `n = 0` 时，不要求 Aarray、Barray、Carray 非空。`k = 0` 或 alpha 为 0，且 `m > 0`、`n > 0` 时，Aarray、Barray 可以为空，Carray 必须非空。其余需要读 A、B 的路径上，三个数组本体都必须非空。不检查数组元素是否为空，也不检查 C[i] 是否重叠。

通过之后：`OP_C` 与 `OP_T` 都记为转置。H 的 alpha、beta 先用 `HalfToFloat` 再与 0、1 比较，与 arch35 host 相同。`k = 0` 或 alpha 为 0 时调用与 `GemmBatchedLaunchEarlyExit` 相同的退化：beta 为 1 则返回成功；beta 为 0 则把逻辑区域写成 0；其他 beta 只缩放逻辑 m×n。退化 kernel 失败时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。`GetAicCoreCount` 或需要向量核时的 `GetAivCoreCount` 得到 0，返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。

arch35 现有校验把 alpha、beta 放在 trans 之前，并且 `batchCount > 0` 时一律拒绝空数组。arch22 不沿用这两处，以免与任务书的错误码和空指针用例不一致。

提交 kernel 后返回 `ACLBLAS_STATUS_SUCCESS`，不调用 `aclrtSynchronizeStream`。

### Kernel

拟在同一 arch22 目录新增 S/H kernel，不改 `cgemm_batched_kernel.cpp`。

S 的乘加复用 arch22 `aclblasSgemm` 已有的阶段：Cube `Matmul` 按 NN、NT、TN、TT 四个入口计算列主序乘积，向量侧在需要时把结果按 alpha、beta 写回逻辑 m×n。Host 按转置标志选择入口。batch 的 A、B、C 地址来自 Device 指针数组，Host 不把这三份数组读回。一个 batch 的矩阵大于单次 tile 时，按 arch22 `aclblasSgemm` 的方式在 M、N 上分块再下发；batch 之间没有数据依赖，可以按核组切 batch，也可以在一个 batch 内切块。具体切分留到实现时按 Cube 核数和 192KB UB 预算确定，不把 arch35 的 `tileM = 128` 或 L1 双 bank 写成 arch22 参数。

直接写回与 epilogue：`alpha == 1` 且 `beta == 0` 时，Cube 的结果写入 C 的逻辑区域，不读 C 的旧值。其他标量先得到不含 beta 的积，再由向量 epilogue 完成 `alpha * product + beta * C`。`beta == 0` 的 epilogue 不读 C。padding 列不在写出范围内。

H 在 arch22 源码中还没有 half GEMM。拟新增 FP16 输入、FP32 累加的 Cube 路径，写回与 arch35 `hgemmBatched_host.cpp` 的两条出口一致：alpha、beta 分别为 1 和 0 时，把 FP32 累加结果收窄后直接写入 C；否则写入 FP32 临时矩阵，再由向量 epilogue 做标量并收窄到 FLOAT16。不编译 arch35 的 `gemm_batched_gemm_kernel_halfout_do`。

这些 kernel 用 Init 与计算、写出阶段。连续矩阵乘不是单次 CopyIn、Compute、CopyOut 就结束：Cube 侧按 K 分块取 A、B，向量 epilogue 再按列主序把逻辑块写回。队列深度和 L1 是否双缓冲在实现时按 dav-2201 的片上容量决定。代码未打开双缓冲之前，本文不写搬运与计算重叠。

kernel 内不重复 Host 已做的枚举、空指针和前导维检查，也不扫描 C[i] 是否重叠。

### 目录与构建

```text
ops-blas/
├── include/cann_ops_blas.h                         # 已有两条声明，不改签名
├── blas/gemm_batched/README.md                     # A2/A3 支持行改为支持，并写明本任务的空指针规则
├── blas/gemm_batched/arch22/
│   ├── cgemm_batched_*                             # 已有，保持不动
│   ├── sgemm_batched_host.cpp                      # 待新建
│   ├── hgemm_batched_host.cpp                      # 待新建
│   └── sgemm/hgemm 的 kernel 与 tiling 头          # 待新建，不并入复数 kernel
├── blas/gemm_batched/arch35/                       # 保持不动
└── test/gemm_batched/sgemm_batched/
    ├── CMakeLists.txt                              # 已注册 sgemm_batched_test
    ├── sgemm_batched_param.h                       # 当前只有 FLOAT32，拟扩展到能分派 H
    ├── arch35/                                     # 保持不动
    └── arch22/                                     # 待新建
        └── sgemm_batched_test.cpp
```

`blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 收集 `*/arch22/*.cpp`。A2/A3 只编入 arch22，950 只编入 arch35，两套定义不会同时链接。arch22 根下没有与新文件同名的非 arch 源文件。

`test/gemm_batched/sgemm_batched/CMakeLists.txt` 现在无条件注册 `sgemm_batched_test`。GTest 发现逻辑只收集当前 `SOC_ARCH_DIRS` 下的 `*_test.cpp`。拟把 arch22 测试源放在 `arch22/`，目标名仍是 `sgemm_batched_test`。

## 支持硬件

| 芯片 | 本任务 |
| --- | --- |
| Atlas A2 训练 / 推理系列 | 拟支持。性能验收覆盖任务书规定的 A2 款型 |
| Atlas A3 训练 / 推理系列 | 拟支持，与 A2 共用 arch22 源。自验证覆盖一种款型，另一种待测 |
| Ascend 950PR / Ascend 950DT | 已有 arch35。本提交不改该实现，也不计入 A2/A3 验收 |

## 算子约束限制

- S 只支持 FLOAT32，H 只支持 FLOAT16。alpha、beta 与 A、B、C 同 dtype，且 alpha、beta 在 Host。
- 非法 trans 返回 `ACLBLAS_STATUS_INVALID_ENUM`。这与 cuBLAS 把非法 trans 报成 invalid value 不同，是 ops-blas 的既有约定。
- m、n、k、batchCount 小于 0，或前导维低于下界，或 alpha、beta 为空，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- `m = 0`、`n = 0` 或 `batchCount = 0`：成功返回，不读三个指针数组。
- `k = 0` 或 alpha 为 0：不读 A、B。`m > 0` 且 `n > 0` 时 Carray 必须非空。beta 为 1 时不写 C。
- `beta = 0` 不读 C 的旧值。只写逻辑 m×n。
- 不支持广播、动态 shape 推导、任意非连续布局、视图返回。lda、ldb、ldc 是列主序前导维，不是第二套 stride 语义。
- 调用方保证数组元素指向当前 device 上的有效内存，C[i] 互不重叠，且不与任意 A[j]、B[j] 重叠。Host 不检查这些条件。
- 同一输入两次执行不要求逐位相同。
- 本提交范围不含 arch35 行为变更、复数 batched、跨距批量，以及未测量的性能数字。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 逐元素 `|actual - golden| ≤ atol + rtol * |golden|` 记为 matched。用例同时满足 matched ratio ≥ 0.99，且最大绝对误差不超过该档上限时通过。S 为 FLOAT32：rtol = atol = 2^-13，上限为 1e-2 或 32 ULP。H 为 FLOAT16：rtol = atol = 2^-10，上限为 1e-1 或 32 ULP。S 的 golden 按 Netlib `sgemm`，测试工程可以用 float64 累加后收窄到 float32，也可以直接调用 cblas 的 float32 参考。H 没有 Netlib half 参考，按 float64 累加、收窄到 float32，再按就近舍入收到 FLOAT16 | 任务书 §3.2；任务书所给 opbase mixed tolerance 文档 |
| 性能标准 | 正式判据是 msprof op 的 kernel task duration（us）。warmup 5 次不计入，之后 10 次成功样本取平均。平均耗时 ≤ 任务随附 `gpu_baseline.csv` 中 `gpu_ms` 的 1.25 倍。S、H 各 100 行，列含转置、m、n、k、batchCount、alpha、beta 和三个前导维。GTest 的 `[PERF] avg_us` 含下发开销，只作交叉验证。NPU 耗时待测 | 任务书 §3.3；两份 `gpu_baseline.csv` |

内存：任务书 §3.4 为不涉及。验收模板仍有内存报告，设计阶段不填占用数字。

随任务 README 把 GTest `avg_us` 写成与 `gpu_ms` 比对的脚本口径。任务书以 msprof 为正式判据。验收按任务书。自测报告并列两列。

## 测试设计

用例分两份 CSV，各 600 行，没有 `op` 列。任务书要求同一 GTest 按算子分派；拟在 arch22 测试里分别加载这两份文件，或在加载时标上算子，而不是假定 CSV 已有 `op` 列。仓内 `sgemm_batched_param.h` 目前只解析 FLOAT32。H 的分派待测试侧扩展。

当前文件统计：

| 文件 | 行数 | 前缀 |
| --- | --- | --- |
| `sgemm_batched_test.csv` | 600，其中精度 500、`TC_PF` 100 | `TC_L0` 18、`TC_MN` 61、`TC_AB` 22、`TC_LD` 15、`TC_FL` 21、`TC_CV` 24、`TC_ED` 64、`TC_EX` 275、`TC_PF` 100 |
| `hgemm_batched_test.csv` | 600，其中精度 500、`TC_PF` 100 | `TC_L0` 18、`TC_MN` 61、`TC_AB` 22、`TC_LD` 15、`TC_FL` 27、`TC_CV` 24、`TC_ED` 70、`TC_EX` 263、`TC_PF` 100 |

两份文件的期望状态码都是成功 574、`ACLBLAS_STATUS_INVALID_VALUE` 21、`ACLBLAS_STATUS_INVALID_ENUM` 4、`ACLBLAS_STATUS_HANDLE_IS_NULLPTR` 1。S 的 `mere_threshold` 为 2^-13，H 为 2^-10，与任务书分档一致。同目录 README 里的若干前缀条数与 CSV 不同，以 CSV 为准。

| 维度 | 用例里已有的内容 | arch22 拟执行 |
| --- | --- | --- |
| 正向 | 九种转置、多种 m/n/k、padding、`RANDOM_NORM_5_5` 与 `GAUSS_NORM_0_1` | 调对应接口，按混合容差比对逻辑 m×n |
| 边界 | m、n、k、batchCount 为 0；alpha 为 0；beta 为 0 或 1 | 按 quick return 和 `C = beta * C` 判定。不把未写的 padding 算进误差 |
| 负向 | 非法 trans、负维度、前导维不足、空 handle、空 alpha 或空数组本体 | 只比对状态码 |
| 一致性 | S 对 cblas，H 对 float64 累加后再收窄 | 逐 batch 比对 |
| 特殊值 | 填充与标量列含 0、1 和 NaN 相关描述。H 另有次正规与大值用例 | 按 IEEE 传播后做容差比对 |
| 回归 | arch35 测试源不改 | A2/A3 构建不链入 arch35 的 S/H 定义；950 构建不链入本次新增的 arch22 定义 |
| 性能 | 每算子 100 条 `TC_PF`，`gpu_ms` 列已有值 | msprof 正式判据待测；GTest `avg_us` 作交叉验证 |
| 空指针 quick return | 任务书允许 `m = 0` 或 `n = 0` 时三个数组为空 | arch22 期望成功。这与 arch35 当前「batchCount > 0 即拒绝空数组」不同，用例期望以任务书为准 |

精度和性能都待在 A2 或 A3 上实测。

## 兼容性分析

- 新增：arch22 的 S/H host 与 kernel，以及 arch22 GTest。头文件符号已存在，不新增 ABI。
- 增量架构：只增加 arch22 文件，并让测试发现逻辑在 A2/A3 上编到新的测试源。`SOC_ARCH_DIRS` 与 arch35 互斥。
- dtype / shape：S 仍是 FLOAT32，H 仍是 FLOAT16。m、n、k、batchCount 和前导维都是运行时标量。本提交不扩展 dtype，也不做广播。
- 并行贡献：与 950 共用 `include/cann_ops_blas.h` 的同一对声明。不修改 `blas/gemm_batched/arch35/`，不修改 arch22 的 `aclblasCgemmBatched`。README 只把 A2/A3 改为支持，并分开写本任务的空指针规则；950 支持行保持原样。
- 不涉及 TBE 插件、Python 包和旧私有 API。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 把 arch35 的 tensor_api kernel 直接放进 arch22 | arch22 使用本仓已有的 Cube `Matmul` 与向量 epilogue。H 的 FP16 入口单独新增，不编译 arch35 的 halfout 源文件 |
| 沿用 arch35「batchCount > 0 就拒绝空数组」，与任务书的 m、n 为 0 以及 alpha 为 0 的用例冲突 | Host 按任务书第 7 步放行对应空指针。arch35 源码不改 |
| 校验顺序与 arch35 不同，非法 trans 同时伴随空 alpha 时错误码不一致 | arch22 先判 trans，返回 `ACLBLAS_STATUS_INVALID_ENUM`。负向用例按这个顺序写期望 |
| H 的收窄时机与 golden 的「先到 float32 再到 FP16」不一致 | 累加保持 FP32，只在写 C 时收窄。自测报告写明与 golden 收窄链的差异 |
| 写出范围含 ldc padding，或 beta 为 0 时仍读到 C 里的 NaN | epilogue 与置零只覆盖逻辑 m×n。beta 为 0 的路径不读 C |
| 性能门限尚未测量 | 先完成容差比对，再按 msprof 采集。超门限时再改 tile。本文不把双缓冲写成已有优化 |
| 库属 workspace 小于临时积 | 返回 `EnsureDefaultWorkspace` 的失败码。直接写回和 beta 为 1 的退化不申请这块临时积 |
