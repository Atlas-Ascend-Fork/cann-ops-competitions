# aclblasCsyrk3mEx 算子设计文档（Ascend 950PR）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 11月社区任务-aclblasCsyrk3mEx算子开发（950） |
| 任务 taskId | `3d67a5f992894ea8b43e472342c122e6` |
| 任务链接 | [昇腾社区任务中心](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6) |
| 任务讨论帖 | [cann/ops-blas#88](https://gitcode.com/cann/ops-blas/discussions/88) |
| 目标代码仓 | [cann/ops-blas](https://gitcode.com/cann/ops-blas)（master） |
| 实现目录 | `blas/syrk/arch35/` |
| 测试目录 | `test/syrk/csyrk3m_ex/arch35/` |
| 适配硬件 | Ascend 950PR（arch35 / dav-3510） |
| CANN 版本 | CANN 9.1.0 及以上 |
| 设计文档提交路径 | `04_tasks/01_community-task-2026/tasklist/11-aclblasCsyrk3mEx-950/qq_45622250/docs/design.md` |
| 文档状态 | 设计评审阶段；本文不宣称代码、精度或性能已完成验收 |

# 需求背景（required）

## 需求来源

本算子来自“11月社区任务-aclblasCsyrk3mEx算子开发（950）”（taskId：`3d67a5f992894ea8b43e472342c122e6`）。任务要求基于昇腾 950PR（`arch35` / `dav-3510`）硬件与 [ops-blas](https://gitcode.com/cann/ops-blas) 开源仓的句柄式 BLAS kernel 直调工程模式，使用 Ascend C 开发单精度复数对称矩阵秩-k 更新（3M 扩展版）算子 `aclblasCsyrk3mEx`，对标 cuBLAS `cublasCsyrk3mEx` 接口（语义参考 Netlib `csyrk`），完成设计、开发、测试全流程工作，验收通过后合入 `cann/ops-blas` 开源仓。

## 背景介绍

### 算子定位

`aclblasCsyrk3mEx` 是 BLAS Level-3 复数对称矩阵秩-k 更新的 3M 扩展接口，核心计算公式为：

```text
C = alpha * op(A) * op(A)^T + beta * C
```

其中 `A`、`C` 均为列主序（column-major）存储矩阵：

- `trans = ACLBLAS_OP_N`：`op(A) = A`，逻辑尺寸 `n x k`，前导维 `lda >= max(1, n)`；
- `trans = ACLBLAS_OP_T`：`op(A) = A^T`，`A` 物理尺寸 `k x n`，前导维 `lda >= max(1, k)`；
- `trans = ACLBLAS_OP_C`：本算子为**对称（Symmetric）**语义而非 Hermitian 语义，`OP_C` 按 `OP_T` 等价处理，**不做共轭**；
- `C` 为 `n x n` 复数对称矩阵（`C = C^T`），前导维 `ldc >= max(1, n)`；仅 `uplo` 指定的上三角（`ACLBLAS_UPPER`，`i <= j`）或下三角（`ACLBLAS_LOWER`，`i >= j`）被读写，另一三角及 `[n, ldc)` 行填充区保持原值不变；
- **对角元素虚部保留**：对称语义下 `C(i, i)` 的虚部为有效数据，计算后原样保留，**不置 0**（这是与 Hermitian 算子 `cherk` 的本质区别之一）；
- **扩展输入类型支持**：`Atype` 支持 `ACL_COMPLEX64`（交错 8 字节 `(float real, float imag)`）与 `ACL_INT8`（complex-int8，交错 2 字节 `(int8_t real, int8_t imag)`，对标 cuBLAS `CUDA_C_8I`），`Ctype` 固定为 `ACL_COMPLEX64`；`alpha`、`beta` 为单精度复数标量 `aclblasComplex`（同时支持 Host 与 Device 指针）；
- **对称专用 3M 算法**：相比直接复数矩阵乘（4 次实数 GEMM，4M），3M 算法将复数对称乘积 `X * X^T` 分解为 **3 次实数对称矩阵乘**，在理论计算量与中间暂存带宽上均比 4M 降低 25%；且任务书明确要求**三次实数矩阵乘必须在同一次 kernel launch 内完成**，不得用三次独立 GEMM 调用拼接。

### 仓内现状（基于 ops-blas master 核实）

| 项目 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| 公共接口声明 | `include/cann_ops_blas.h`（L799-801）仅有 `aclblasCsyrk`，尚无 `aclblasCsyrk3mEx` 或 `aclblasCsyrkEx` 声明 | 须在 `include/cann_ops_blas.h` 按任务书 §2.3 新增 `aclblasCsyrk3mEx` 公共原型，禁止定义私有平行接口 |
| `blas/syrk/arch35/` | 已有非 Ex 的 `aclblasCsyrk`（4M 管线：`csyrk_host.cpp`、`csyrk_kernel.cpp`、`csyrk_cube_kernel.cpp`、`csyrk_gemm_fused.h`、`csyrk_direct_kernel.cpp`）与 `aclblasSsyrk` | 可参考参数校验、`CheckPtrLocation` 标量探测、uplo 严格三角写回、对角虚部保留与 `tensor_api` Cube 融合范式；但本任务须实现 3M 三乘积融合路径与 `ACL_INT8` 精确提升 |
| `blas/gemm/arch35/` | 已有 `cgemm3m_host.cpp`、`cgemm3m_rounding_kernel.cpp`、`cgemm_ex_host.cpp`、`cgemm_ex_cube3_kernel.cpp` 等 | 可参考 3M 数值舍入防护与单发射三乘积 Cube 融合结构 |
| 同批独立任务边界 | `aclblasCsyrkEx` 与本任务 `aclblasCsyrk3mEx` 均为 1125 新增的独立接口 | 两者**彼此不得依赖**；本任务严禁调用尚未合入的 `aclblasCsyrkEx`，也不得拼接多次 GEMM |
| 构建与源码过滤 | `blas/CMakeLists.txt` 自动收集 `*/arch35/*.cpp`；单算子 `--ops=csyrk3m_ex` 时按 `^csyrk3m_ex_` 文件名前缀自动保留匹配源码 | 新建文件统一以 `csyrk3m_ex_` 为前缀放入 `blas/syrk/arch35/`，即可被 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 自动收集 |
| 测试目录发现 | `test/syrk/` 下当前仅有 `csyrk/`、`ssyrk/`；`test/CMakeLists.txt` 自动按家族子目录发现 `test/syrk/csyrk3m_ex/` 并检查 `arch35/csyrk3m_ex_test.cpp` | 新建 `test/syrk/csyrk3m_ex/` 及 `CMakeLists.txt`、`arch35/` 测试源码，无需修改顶层 `test/CMakeLists.txt` 即可注册构建目标 |
| 算子文档 | `blas/syrk/README.md` 当前仅包含 `aclblasSsyrk` 与 `aclblasCsyrk` 章节 | 交付时在 `blas/syrk/README.md` 补充 `aclblasCsyrk3mEx` 章节与产品支持表 |

### 参考实现与工程骨架

| 参考文件 | 用途 |
| --- | --- |
| `blas/syrk/arch35/csyrk_host.cpp` | 参数校验、Host/Device 复数标量指针探测（`CheckPtrLocation`）、64 位溢出保护的 workspace 计算、小尺寸直算与大尺寸管线分派 |
| `blas/syrk/arch35/csyrk_cube_kernel.cpp` 与 `csyrk_gemm_fused.h` | `DAV_3510` 上基于 `AscendC::Te`（`tensor_api`）的 `BM=BN=64` 单 owner 块三角跳过、蛇形遍历、双 L0C 累加器 ping-pong 及 FP32 / HF32x3 补偿驱动 |
| `blas/syrk/arch35/csyrk_kernel.cpp` | AIV 侧连续块读取 + `DeInterleave` 预处理、严格 `uplo` 三角写回、以及大 `k` 对角 8 行批量直接重算（`ComputeDiagBatch8`） |
| `blas/gemm/arch35/cgemm3m_host.cpp`、`cgemm3m_rounding_kernel.cpp`、`cgemm_ex_cube3_kernel.cpp` | 3M 舍入误差分析与单发射三乘积 Cube 调度参考 |
| `blas/common/helper/syrk_gemm_arch35.h`、`aclblas_handle_internal.h`、`host_utils.h`、`syrk_host_utils.h` | `GetAicCoreCount()` / `GetAivCoreCount()`、handle 默认 32 MiB / 上限 2 GiB workspace 管理（`EnsureDefaultWorkspace` / `GetEffectiveWorkspace`）与公共常量 |
| `test/syrk/csyrk/` | ATK GTest CSV 驱动测试结构、实/虚部分离 mixed tolerance 校验与非 `uplo` 三角精确写保护断言 |

### 口径裁决

1. **错误码区分（ops-blas 约定）**：
   - `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - `uplo` 或 `trans` 非法枚举值返回 `ACLBLAS_STATUS_INVALID_ENUM`（任务书 §2.3 明确说明与 cuBLAS `INVALID_VALUE` 属有意差异）；
   - `Atype` / `Ctype` 不在支持集 `{(ACL_COMPLEX64, ACL_COMPLEX64), (ACL_INT8, ACL_COMPLEX64)}` 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；
   - `n < 0`、`k < 0`、`lda`/`ldc` 低于下界、指针为空、或 Host/Device 标量混用返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. **对称 3M 分解（严禁套用 Hermitian）**：固定计算 `X * X^T`（`P0 = Xr * Xr^T`、`P1 = Xi * Xi^T`、`P2 = (Xr + Xi) * (Xr + Xi)^T`，实部 `P0 - P1`、虚部 `P2 - P0 - P1`）；`OP_C` 映射为 `OP_T` 不共轭；对角虚部保留不置 0。
3. **单次 Kernel Launch 完成三次实数乘**：3M 的 `P0`、`P1`、`P2` 在同一个 AIC Cube kernel（`csyrk3m_ex_cube3m_kernel`）内针对每个被选中的三角输出 tile 融合计算完成，绝不在 Host 侧发起三次独立 GEMM 调用拼接。
4. **Host / Device 标量指针双支持**：通过 `CheckPtrLocation(alpha/beta)` 自动识别标量指针位置，支持全 Host 或全 Device 标量指针；混用 Host 与 Device 指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。标量数值按值写入 tiling 下发给所有计算路径（包括小形状单发射路径），避免 Host 标量误走低效回退。
5. **退化与边界语义**：
   - `n == 0`：完成结构校验后合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，允许 `A`、`C` 为空，不启动 kernel；
   - `k == 0` 或 `alpha == (0, 0)`：不引用 `A`（`A` 可为空或含 NaN）；若 `beta == (1, 0)` 则 Host 侧 quick return，否则仅对 `C` 的 `uplo` 三角执行 `C = beta * C`（零 workspace）；
   - `beta == (0, 0)`：不读取 `C` 旧值（`C` 旧值含 NaN/Inf 时不污染 `uplo` 三角输出）；
   - 非 `uplo` 三角与 `[n, ldc)` 行填充区严禁写入，保持调用前原值。
6. **异步执行语义**：计算 kernel 异步提交至 `handle` 绑定的 stream，接口返回成功不代表 Device 计算完成，由调用方在读回 `C` 前同步 stream（仅当 `alpha`/`beta` 为 Device 指针需 D2H 读取标量，或库 workspace 扩容时，按 `ops-blas` 既有 handle 契约在发射前同步 stream）。

### 设计阶段范围

本 PR 仅提交设计文档 `docs/design.md`，不包含 `ops-blas` 算子实现代码、测试代码与自测报告。算子代码、测试工程与实机验收报告在设计评审通过并获取 Ascend 950PR 开发环境后另行提交。文中列出的 `gpu_ms` 数值均来自任务书附带基线文件 `test_cases/gpu_baseline.csv`，作为验收目标参考，**不是** NPU 实测数据。

# 需求分析（required）

## 需求描述

在 `include/cann_ops_blas.h` 新增公开接口声明，并在 `blas/syrk/arch35/` 基于 Ascend C 实现 `aclblasCsyrk3mEx`：

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

参数序列与 cuBLAS `cublasCsyrk3mEx`（13 个参数）逐位一一对应，无需额外参数映射层。实现需支持 `ACLBLAS_UPPER` / `ACLBLAS_LOWER`、`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C`（`OP_C` 等价于 `OP_T`）、`(Atype=ACL_COMPLEX64, Ctype=ACL_COMPLEX64)` 与 `(Atype=ACL_INT8, Ctype=ACL_COMPLEX64)` 两种数据类型组合、紧凑与带 padding 的 `lda`/`ldc`、Host 与 Device 标量指针，并通过 handle 绑定的 stream 异步执行。

## 需求拆解

| 编号 | 需求项 | 详细要求 |
| --- | --- | --- |
| FR1 | 公共头文件声明 | 在 `include/cann_ops_blas.h` 新增 `aclblasCsyrk3mEx` 声明，与其他产品线共用，禁止定义私有平行接口 |
| FR2 | 对称秩-k 数学语义 | 计算 `C = alpha * op(A) * op(A)^T + beta * C`，列主序；`OP_C` 按 `OP_T` 处理不共轭；对角元素虚部原样保留不置 0 |
| FR3 | 3M 单发射融合 | 采用对称专用 3M 分解 `P0 = Xr*Xr^T`、`P1 = Xi*Xi^T`、`P2 = (Xr+Xi)*(Xr+Xi)^T`，`Re = P0 - P1`、`Im = P2 - P0 - P1`；三次实数乘在同一次 kernel launch 内完成 |
| FR4 | 混合输入类型 | 支持 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)`；`ACL_INT8` 按 `(int8 real, int8 imag)` 交错 2 字节存储并无损精确提升为 FP32；其他组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| FR5 | 三角写保护 | 仅 `uplo` 指定三角（含对角）被读写，另一三角及 `ldc > n` 填充区不得改写 |
| FR6 | 参数校验与顺序 | 按任务书 §2.5 校验顺序逐项校验 `handle`、`uplo/trans`、`n/k`、`lda/ldc`、`alpha/beta`、`Atype/Ctype`、`A/C` 指针并返回规定状态码 |
| FR7 | 边界与退化语义 | `n=0` 合法 no-op；`k=0` 或 `alpha=(0,0)` 不访问 `A`，`beta=(1,0)` quick return、`beta != (1,0)` 仅做 `C=beta*C`；`beta=(0,0)` 不读 `C` 旧值 |
| FR8 | 标量指针与异步执行 | 同时支持 Host / Device `alpha/beta` 指针；kernel 在 handle 绑定 stream 上异步下发 |
| FR9 | 精度验收标准 | 以 CPU CBLAS `cblas_csyrk`（float64 域可用 `cblas_zsyrk`）为 golden，实/虚部分别按 FLOAT32 混合容差判定：`rtol = atol = 2^-13`、`matched_ratio >= 0.99`、`max_abs_error <= 1e-2` 或 `32 * ULP`；通过 1000 条精度用例（含 4 条 `TC_3M_*` 专项） |
| FR10 | 性能验收标准 | Ascend 950PR 上以 `msprof op` 采集 kernel `Task Duration(us)`（warmup 5 + 有效采样 10 取平均），与 `gpu_baseline.csv` 的 `gpu_ms` 做 kernel-to-kernel 对比：单 case 加速比 ≥ 0.4（耗时 ≤ `gpu_ms / 0.4`）且 200 条均值 ≥ 0.5；另补充 `Atype=ACL_INT8` 性能采集并与 `ACL_COMPLEX64` 基线对比 |
| FR11 | 工程与交付 | 算子实现位于 `blas/syrk/arch35/`、测试位于 `test/syrk/csyrk3m_ex/arch35/`、支持 `bash build.sh --ops=csyrk3m_ex --soc=ascend950`、更新 `blas/syrk/README.md` |

## 公开接口与范围

本任务规划涉及的代码改动范围如下：

- **公共头**：`include/cann_ops_blas.h` 新增 `aclblasCsyrk3mEx` 函数声明；
- **算子实现（`blas/syrk/arch35/`）**：新增 `csyrk3m_ex_host.cpp`、`csyrk3m_ex_tiling_data.h`、`csyrk3m_ex_kernel.h`、`csyrk3m_ex_kernel.cpp`、`csyrk3m_ex_cube_kernel.cpp`、`csyrk3m_ex_gemm_fused.h`、`csyrk3m_ex_small_kernel.cpp`；
- **构建门控**：按仓内 `cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS` 列表规范，为 `csyrk3m_ex` 添加 `asc-devkit >= 9.1` 版本门控（与 `ssyrk`、`csyrk` 一致）；
- **测试工程（`test/syrk/csyrk3m_ex/`）**：新增 `CMakeLists.txt`、`csyrk3m_ex_param.h`、`csyrk3m_ex_golden.h`、`arch35/csyrk3m_ex_test.cpp`、`arch35/csyrk3m_ex_npu_wrapper.h`、`arch35/csyrk3m_ex_test.csv`；
- **算子文档**：在 `blas/syrk/README.md` 新增 `aclblasCsyrk3mEx` 章节（接口原型、参数说明、约束说明、产品支持情况与调用示例）。

不修改：已有非 Ex 的 `aclblasCsyrk` / `aclblasSsyrk` 实现与测试、`blas/gemm/` 实现、公共 handle 内部结构（直接复用既有默认/用户 workspace 接口）；且不依赖同批任务 `aclblasCsyrkEx`。

## 任务书 §2.5 约束逐项核对

| §2.5 约束项 | 本设计结论 | 落地位置 |
| --- | --- | --- |
| 参数合法性与校验层级 | Host 入口严格按 §2.4/§2.5 顺序逐参数校验并返回规定状态码；`Atype=ACL_INT8` 按交错 `(int8 real, int8 imag)` 解释；不对大尺寸 Device 张量做元素级扫描，矩阵对称性由调用方保证 | `csyrk3m_ex_host.cpp::ValidateCsyrk3mExParams` |
| 非连续 Tensor 支持 | 通过 `lda`/`ldc` 支持列主序带行填充布局；不引入 AutoContiguous / ViewCopy；Phase 0 去除 `lda` 空洞，Phase 2 严格跳过 `ldc` 空洞不改写 | `csyrk3m_ex_kernel.cpp` / `csyrk3m_ex_small_kernel.cpp` |
| broadcast 规则 | 不支持广播，`A` 与 `C` 为独立矩阵操作数 | — |
| dynamic shape 要求 | 不要求图模式动态 shape；`n`、`k`、`lda`、`ldc` 为运行时入参，Host 实时计算 tiling 并按值下发 | `csyrk3m_ex_host.cpp` tiling 计算函数 |
| 原地与视图语义 | `C` 为原地覆写，不返回视图；仅 `uplo` 指定三角被读写；`beta == (0,0)` 时不读取 `C` 旧值；调用方保证 `C` 可写区与 `A` 可读区不重叠 | Phase 2 combine / small kernel 写回逻辑 |
| 确定性计算要求 | 不要求跨路径/跨次逐位一致；同一形状按确定性规则分派路径、无核间原子竞争，按 §3.2 精度阈值验收 | Host 形状路由 + 固定 tile 归属映射 |
| 空 Tensor 处理 | `n == 0` 校验通过后直接返回 `ACLBLAS_STATUS_SUCCESS`，不引用 `A`/`C`、不启动 kernel；`k == 0` 时不引用 `A`，`beta == (1,0)` quick return，`beta != (1,0)` 仅对 `uplo` 三角做 `beta` 缩放 | `csyrk3m_ex_host.cpp` quick return 与 `betaOnly` 路径 |
| 0 维 Tensor 处理 | 不涉及 0 维 Tensor 入参；矩阵经指针 + 维度/前导维描述，复数标量经指针传入，Host 与 Device 两种指针位置均支持 | `ReadComplexScalars`（`CheckPtrLocation`） |
| 异步执行 | 经 `aclblasSetStream` 绑定 stream 异步发射计算 kernel；Host 读回结果前须由调用方同步该 stream | `LaunchCsyrk3mExKernel` |

# 详细设计（required）

## 算子分析

### 数学公式与对称 3M 分解

以 0-based 索引表示列主序元素 `M(i, j) = M[i + j * ld]`，计算区域为 `uplo` 指定的三角（`UPPER`：`0 <= i <= j < n`；`LOWER`：`0 <= j <= i < n`）：

```text
trans = N:    op(A)(i, t) = A[i + t * lda],  i in [0, n), t in [0, k)
trans = T/C:  op(A)(i, t) = A[t + i * lda],  i in [0, n), t in [0, k)   (OP_C 不共轭)

C_new(i, j) = alpha * Sum_{t=0..k-1}( op(A)(i, t) * op(A)(j, t) ) + beta * C_old(i, j)
```

令 `X = op(A) = Xr + i * Xi`（`Xr, Xi` 为 `n x k` 实数 FP32 矩阵；当 `Atype == ACL_INT8` 时，每个交错 `(int8_t re, int8_t im)` 先无损精确提升为 FP32）。展开对称乘积 `X * X^T`：

```text
X * X^T = (Xr + i * Xi) * (Xr^T + i * Xi^T)
        = (Xr * Xr^T - Xi * Xi^T) + i * (Xr * Xi^T + Xi * Xr^T)
```

利用对称交叉项恒等式 `(Xr + Xi) * (Xr + Xi)^T = Xr * Xr^T + Xi * Xi^T + Xr * Xi^T + Xi * Xr^T`，引入 **3M 实数矩阵和** `Xs = Xr + Xi`，定义三次实数对称乘积：

```text
P0 = Xr * Xr^T
P1 = Xi * Xi^T
P2 = Xs * Xs^T = (Xr + Xi) * (Xr + Xi)^T
```

从而得到复数对称秩-k 乘积的实部与虚部：

```text
Cr = Re(X * X^T) = P0 - P1
Ci = Im(X * X^T) = P2 - P0 - P1
```

随后按复数乘法完成 `alpha` / `beta` 合并（其中 `alpha = (ar, ai)`、`beta = (br, bi)`、`C_old = (c0r, c0i)`）：

```text
out_re = (ar * Cr - ai * Ci) + (br * c0r - bi * c0i)
out_im = (ar * Ci + ai * Cr) + (br * c0i + bi * c0r)
```

**对称 3M 与 Hermitian 分解的本质区分**：

| 维度 | 对称 `csyrk3mEx`（本算子，`X * X^T`） | Hermitian `cherk`（严禁套用，`X * X^H`） |
| --- | --- | --- |
| 转置共轭 | `OP_C` 等价于 `OP_T`，**不共轭** | `OP_C` 对虚部取负共轭 |
| 乘积实部 | `P0 - P1 = Xr * Xr^T - Xi * Xi^T` | `Xr * Xr^T + Xi * Xi^T` |
| 乘积虚部 | `P2 - P0 - P1 = Xr * Xi^T + Xi * Xr^T`（对称） | `Xi * Xr^T - Xr * Xi^T`（反对称） |
| 对角元素虚部 | `2 * Sum(Xr * Xi)`，**有效数据、原样保留** | 恒为 0，强制置 0 |
| `alpha` / `beta` 类型 | 复数 `aclblasComplex` | 实数 `float` |

**计算量与访存收益分析**：
- 相比仓内 4M `aclblasCsyrk`（4 次实数 GEMM：`Q0, Q1, Q2, Q3`），3M 将 Cube MMAD 次数与中间 `temp` 缓冲从 4 份降至 3 份，直接减少 **25%** 矩阵乘算力与 **25%** 中间结果 GM 读写；
- 结合 AIC 侧 `triangleMode` 三角 tile 跳过，进一步省去近 **50%** 非 `uplo` 区域的 Cube 计算与写回；综合算力需求仅为朴素全矩阵 4M 的 **37.5%**。

### 支持数据类型与内存布局

| 参数 | `aclDataType` / C 类型 | 物理存储与编码 | 处理方式 |
| --- | --- | --- | --- |
| `A`（FP32 复数） | `ACL_COMPLEX64` | 交错 8 字节：`[float re, float im]`，列主序步长 `lda`（单位：复元素） | AIV 按列连续读取后 `DeInterleave` 得 `Xr`、`Xi`，并计算 `Xs = Xr + Xi` |
| `A`（INT8 复数） | `ACL_INT8` | 交错 2 字节：`[int8_t re, int8_t im]`（对应 cuBLAS `CUDA_C_8I`），列主序步长 `lda`（单位：复元素） | AIV 按列连续读取 2B 交错数据，经 `int8 -> half -> float` 无损精确提升为 FP32，拆分 `Xr`、`Xi` 并计算 `Xs = Xr + Xi` |
| `C` | `ACL_COMPLEX64` | 交错 8 字节：`[float re, float im]`，列主序步长 `ldc`（单位：复元素） | 原地覆写，仅读写 `uplo` 三角 |
| `alpha`、`beta` | `const aclblasComplex*` | `{float real; float imag;}`，Host 或 Device 指针 | `CheckPtrLocation` 自动识别，按值写入 tiling |
| 不支持组合 | 除 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)` 外的任意组合（如 `ACL_FLOAT`） | — | 返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |

**`ACL_INT8` 在 Ascend 950PR Cube 上的硬件精度特性（关键优化）**：
- `int8_t` 取值范围 `[-128, 127]`，二进制尾数最多需要 **7 bit**；
- 3M 求和矩阵 `Xs = Xr + Xi` 的每个元素取值范围 `[-256, 254]`，二进制尾数最多需要 **8 bit**；
- Ascend 950PR（`dav-3510`）Cube 的 HF32 模式具有 **10 bit 显式尾数**（11 bit 有效精度）与 FP32 累加器；
- 因此 `Atype == ACL_INT8` 时，`Xr`、`Xi`、`Xs` 三个 FP32 矩阵在进入 HF32 Cube 时**零截断误差（100% 位精确表示）**！无需 HF32x3 残差拆分与三次补偿乘加，单次 HF32 MMAD 即可在获得 HF32 高吞吐的同时保持与 FP32 累加完全一致的输入精度。

### 支持形状与特殊语义

| 场景 | 行为规范 |
| --- | --- |
| `n = 0`（合法 `k >= 0`） | 结构参数全部合法后返回 `ACLBLAS_STATUS_SUCCESS`，不访问 `A`/`C`，不发射 kernel |
| `k = 0` 且 `beta = (1, 0)` | 结构参数合法后 Host 直接 quick return，`C` 不变，不访问 `A` |
| `k = 0` 且 `beta != (1, 0)` | 不访问 `A`（允许 `A == nullptr`），启动轻量 `betaOnly` AIV kernel 仅更新 `uplo` 三角为 `beta * C`（`beta = (0,0)` 直接写 0 不读 `C_old`） |
| `alpha = (0, 0)` 且 `beta = (1, 0)` | Host 直接 quick return，不访问 `A`（即使 `A` 为 `nullptr` 或含 `NaN`），`C` 不变 |
| `alpha = (0, 0)` 且 `beta != (1, 0)` | 不访问 `A`（即使 `A` 为 `nullptr` 或含 `NaN`），走 `betaOnly` 路径执行 `C = beta * C` |
| `beta = (0, 0)` 且 `C_old` 含 `NaN/Inf` | 写回阶段严禁读取 `C_old`，确保 `uplo` 三角输出不受旧值污染 |
| 对角元素虚部非零 | `C(i, i)` 按完整复数公式计算并写回实部和虚部，绝不清零虚部 |
| `lda > aRows` / `ldc > n` | `A` 仅读有效 `[0, aRows)` 行；`C` 仅写 `uplo` 三角，另一三角及 `[n, ldc)` 填充区原位保持不变 |
| 3M 抵消与极值输入（`TC_3M_*` / `TC_FL_*`） | 通过对角直接重算与小矩阵异常值/极值安全路径防止 3M 减法灾难性抵消及 `Inf - Inf = NaN` 偏离 |

## 算子实现

### 文件与工程改动

| 文件路径 | 改动类型 | 职责说明 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 修改 | 新增 `aclblasCsyrk3mEx` 公共 C 接口声明（按 §2.3 原型） |
| `blas/syrk/arch35/csyrk3m_ex_tiling_data.h` | 新增 | Host 与 Kernel 共享的 Tiling 数据结构与 DAV_3510 硬件分块常量 |
| `blas/syrk/arch35/csyrk3m_ex_kernel.h` | 新增 | 各阶段 kernel launcher（`*_do`）对外声明 |
| `blas/syrk/arch35/csyrk3m_ex_host.cpp` | 新增 | `aclblasCsyrk3mEx` API 入口、参数校验、标量探测、workspace 分配、路径路由与 tiling 计算 |
| `blas/syrk/arch35/csyrk3m_ex_small_kernel.cpp` | 新增 | 小尺寸（`n <= 64`）单发射 SIMT 3M kernel（同时支持 `ACL_COMPLEX64` 与 `ACL_INT8`） |
| `blas/syrk/arch35/csyrk3m_ex_gemm_fused.h` | 新增 | `tensor_api` 融合 3M Cube 驱动模板（支持 FP32、INT8 单遍 HF32 与 COMPLEX64 补偿 HF32x3 三种内核模式） |
| `blas/syrk/arch35/csyrk3m_ex_cube_kernel.cpp` | 新增 | Phase 1 AIC 单发射融合 3M Cube kernel 入口与 launcher |
| `blas/syrk/arch35/csyrk3m_ex_kernel.cpp` | 新增 | Phase 0 AIV 预处理 kernel（deinterleave / int8 提升 / `Xs = Xr+Xi` / 融合 HF32 残差）与 Phase 2 AIV 合并/缩放 kernel（含 `betaOnly` 与 8 行批量对角重算） |
| `cmake/asc_devkit_version.cmake` | 修改 | 在 `TENSOR_API_OPS` 中登记 `CSYRK3M_EX`，要求 `asc-devkit >= 9.1`（与 `CSYRK`/`SSYRK` 一致） |
| `blas/syrk/README.md` | 修改 | 补充 `aclblasCsyrk3mEx` 算子说明、约束、产品支持情况与调用示例 |
| `test/syrk/csyrk3m_ex/CMakeLists.txt` | 新增 | 注册 `csyrk3m_ex_test` GTest 目标并配置 `-O2`（加速 CPU golden） |
| `test/syrk/csyrk3m_ex/csyrk3m_ex_param.h` | 新增 | 解析 20 列 CSV（含 `Atype`/`Ctype`、复数标量、`NULLPTR`/`null` 及新分布 token） |
| `test/syrk/csyrk3m_ex/csyrk3m_ex_golden.h` | 新增 | 调用 CBLAS `cblas_csyrk`（可选 `cblas_zsyrk`）与 `int8` 无损提升生成 CPU golden |
| `test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_npu_wrapper.h` | 新增 | 封装 Device 缓冲管理（含 `ACL_INT8` 2B 交错与 `ACL_COMPLEX64` 8B 交错）、非 `uplo` 哨兵保护与 `aclrtEvent` 计时 |
| `test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_test.cpp` | 新增 | CSV 驱动 GTest、`NullHandle` 及 Host/Device 标量 `TEST_F`、精度与 `[PERF] avg_us` 打印 |
| `test/syrk/csyrk3m_ex/arch35/csyrk3m_ex_test.csv` | 新增 | 任务书 1200 条用例（1000 精度 + 200 性能） |

文件组织严格与同目录 `blas/syrk/arch35/csyrk_*` 保持同构，所有新增实现文件均以 `csyrk3m_ex_` 命名，无冗余文件。

### Host（参数校验、路径分派与 Tiling）设计

#### 参数校验顺序

严格按照任务书 §2.4 与 §2.5 的校验顺序执行：

```text
1.  handle == nullptr                                             -> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2.  uplo 不在 {ACLBLAS_UPPER, ACLBLAS_LOWER}                      -> ACLBLAS_STATUS_INVALID_ENUM
3.  trans 不在 {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}         -> ACLBLAS_STATUS_INVALID_ENUM
4.  n < 0 或 k < 0                                                -> ACLBLAS_STATUS_INVALID_VALUE
5.  trans == OP_N 时 lda < max(1, n)；否则 lda < max(1, k)        -> ACLBLAS_STATUS_INVALID_VALUE
6.  ldc < max(1, n)                                               -> ACLBLAS_STATUS_INVALID_VALUE
7.  alpha == nullptr 或 beta == nullptr                           -> ACLBLAS_STATUS_INVALID_VALUE
8.  Ctype != ACL_COMPLEX64 或 Atype 不在 {ACL_COMPLEX64, ACL_INT8} -> ACLBLAS_STATUS_NOT_SUPPORTED
9.  n == 0（结构合法）                                            -> ACLBLAS_STATUS_SUCCESS（quick return）
10. C == nullptr（此时 n > 0）                                    -> ACLBLAS_STATUS_INVALID_VALUE
11. CheckPtrLocation(alpha/beta) 识别 Host/Device 位置并读入标量：
    - 混用 Host 与 Device 指针                                    -> ACLBLAS_STATUS_INVALID_VALUE
    - 均为 Device 指针时经 handle stream 异步 D2H + 单次同步读入
12. 令 isAlphaZero = (alpha.re == 0 && alpha.im == 0)，isKZero = (k == 0)：
    - 若 (!isKZero && !isAlphaZero) 且 A == nullptr               -> ACLBLAS_STATUS_INVALID_VALUE
    - 若 (isKZero || isAlphaZero) 且 beta == (1, 0)               -> ACLBLAS_STATUS_SUCCESS（quick return）
    - 若 (isKZero || isAlphaZero) 且 beta != (1, 0)               -> 分派至 BetaOnly 路径（零 workspace，不访问 A）
```

> 说明：步骤 12 将 `A == nullptr` 校验置于标量读取之后，使得 `k > 0` 且 `alpha != (0,0)` 时空 `A` 返回 `ACLBLAS_STATUS_INVALID_VALUE`（满足 §2.4 与 `null_A` 负向用例），同时 `k == 0` 或 `alpha == (0,0)` 时允许 `A` 为空且不引用 `A`（满足 §2.5 约束）。校验通过后将 `ACLBLAS_OP_C` 归一化为 `ACLBLAS_OP_T`（不共轭）。

#### 多路径分派策略

| 路径 | 触发条件 | 发射 Kernel 数 | Workspace 占用 | 设计目标 |
| --- | --- | --- | --- | --- |
| QuickReturn | `n == 0`，或 `(k == 0 或 alpha == (0,0))` 且 `beta == (1,0)` | 0 | 0 B | 零开销快速返回 |
| BetaOnly | `(k == 0 或 alpha == (0,0))` 且 `beta != (1,0)` | 1（AIV combine `betaOnly`） | 0 B | 仅缩放/清零 `uplo` 三角，大 `n` 不受 workspace 限制 |
| SpecialFallback | `Atype == ACL_COMPLEX64` 且 `n * k <= 128 * 128`，扫描检测到 `alpha/beta/A` 含 `Inf/NaN`、`abs(v) > 1e19` 或 subnormal | 0（Host 精确 Netlib 参考回写） | 0 B | 避免极值溢出与 3M `Inf - Inf = NaN` 偏离；尺寸门控使性能热路径零 D2H 同步 |
| SmallSimt3m | `n <= 64`（任意 `k > 0`，`ACL_COMPLEX64` 与 `ACL_INT8` 均支持） | 1（AIV SIMT） | 0 B | 消除 3 阶段 launch 开销，满足 `n=1..64`（GPU 基线 2.5~5.1 us）低延迟目标，并规避小 `n` 大 `k` 的 Cube 舍入放大 |
| Large3m-INT8 | `n > 64` 且 `Atype == ACL_INT8` | 3（Prep → Cube3M[HF32] → Combine） | `3*arBytes + 3*tempBytes` | `int8` 与 `Xr+Xi` 在 HF32 10-bit 尾数下零截断，单遍 HF32 满速执行且精度无损 |
| Large3m-C64-FP32 | `64 < n <= 256` 且 `k <= 512` 且 `Atype == ACL_COMPLEX64` | 3（Prep → Cube3M[FP32] → Combine） | `3*arBytes + 3*tempBytes` | 中小规模使用原生 FP32 Cube，省去残差生成与额外 2 次补偿 MMAD |
| Large3m-C64-HF32x3 | `(n > 256 或 k > 512)` 且 `Atype == ACL_COMPLEX64` | 3（Prep[融合残差] → Cube3M[HF32x3] → Combine） | `6*arBytes + 3*tempBytes` | 大规模发挥 HF32 Cube 高吞吐，通过 `xh*yh + xl*yh + xh*yl` 恢复约 21-bit 输入精度，将 Prep 与残差融合为单次发射 |

> 性能调优预留：实机验收阶段将在 Ascend 950PR 上对比 `Large3m-C64-FP32` 与 `Large3m-C64-HF32x3` 在 `n = 256..4096` 各档位的实测 `msprof` 耗时与精度余量，微调切换阈值；对 `64 < n <= 512` 中等形状（如 `n=512, k=512` 基线 `45.3 us`），保留 AIC+AIV 融合（将 Phase 1 与 Phase 2 合为单 Mix kernel）的演进接口以进一步压缩发射间隙。

#### Workspace 规划与 64 位溢出保护

沿用 `ops-blas` 句柄式 workspace 管理（默认 32 MiB，动态倍增上限 `ACLBLAS_MAX_WORKSPACE_SIZE = 2 GiB`，用户经 `aclblasSetWorkspace` 注入时遵循不重分配契约）。定义 `arRows = isTransN ? n : k`、`arCols = isTransN ? k : n`、`arLdc = CeilAlign(arRows, 8)`、`tempLdc = CeilAlign(n, 8)`：

```text
arMatBytes    = arLdc * arCols * 4B          (单个 FP32 实平面，32B 对齐)
tempSlabBytes = tempLdc * n * 4B             (单个 FP32 乘积矩阵 P0/P1/P2，32B 对齐)

平面数 numPlanes = (useHf32x3 ? 6 : 3)       // [Xr, Xi, Xs] 或 [Xr, Xi, Xs, XrLow, XiLow, XsLow]
totalBytes       = AlignUp32(numPlanes * arMatBytes + 3 * tempSlabBytes)
```

所有乘法与加法均经 64 位溢出守卫函数 `Csyrk3mMulU64` / `Csyrk3mAddU64` 计算；若溢出或 `totalBytes > ACLBLAS_MAX_WORKSPACE_SIZE`，记录错误日志并返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

最大测试规模（`n = 4096, k = 4096`，任务书 512 MB 输入预算上限）下的 workspace 需求：

| 路径 | `arMatBytes` | `tempSlabBytes` | 总 Workspace 占用 | 与 2 GiB 上限关系 |
| --- | --- | --- | --- | --- |
| `Large3m-INT8` / `Large3m-C64-FP32` | 64 MiB × 3 = 192 MiB | 64 MiB × 3 = 192 MiB | **384 MiB** | 远低于 2 GiB（占用 18.8%） |
| `Large3m-C64-HF32x3` | 64 MiB × 6 = 384 MiB | 64 MiB × 3 = 192 MiB | **576 MiB** | 远低于 2 GiB（占用 28.1%） |

#### Tiling 结构与按值下发

所有 Tiling 结构通过值传递直接进入 kernel 参数，不申请独立 tiling GM，不做 H2D 拷贝：

```cpp
// 1. 小形状单发射 SIMT 3M Tiling (n <= 64)
struct Csyrk3mExSmallTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldc;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    uint8_t isTransN;     // 1 = OP_N, 0 = OP_T/OP_C
    uint8_t isUpper;      // 1 = UPPER, 0 = LOWER
    uint8_t isInt8A;      // 1 = ACL_INT8, 0 = ACL_COMPLEX64
    uint8_t isBetaZero;   // 1 = beta == (0,0), 跳过 C_old 读取
};

// 2. Phase 0: 预处理 Tiling (AIV-only)
struct Csyrk3mExPrepTilingData {
    uint32_t aRows;       // isTransN ? n : k
    uint32_t aCols;       // isTransN ? k : n
    uint32_t lda;         // A 列主序步长（单位：复元素）
    uint32_t outLdc;      // 32B 对齐的平面前导维 CeilAlign(aRows, 8)
    uint32_t colsPerCore; // 每 AIV 核处理的列数
    uint8_t isInt8A;      // 1 = ACL_INT8 (2B 交错), 0 = ACL_COMPLEX64 (8B 交错)
    uint8_t genHf32Resid; // 1 = 同步生成 XrLow/XiLow/XsLow 残差平面
};

// 3. Phase 1: 单发射融合 3M Cube Tiling (AIC-only)
//    单次 launch 内计算 P0 = Xr*Xr^T, P1 = Xi*Xi^T, P2 = Xs*Xs^T，输出至 temp = [P0 | P1 | P2]
struct Csyrk3mExCubeTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t arLdc;       // Xr/Xi/Xs 前导维（32B 对齐）
    uint32_t tempLdc;     // P0/P1/P2 前导维 CeilAlign(n, 8)
    uint32_t usedCoreNum; // 实际启动的 AIC 核数
    uint8_t isTransN;     // 1 = X*X^T, 0 = X^T*X
    uint8_t triangleMode; // 1 = 仅计算上三角 tile, 2 = 仅计算下三角 tile
    uint8_t cubeMode;     // 0 = 原生 FP32, 1 = 单遍 HF32 (INT8 精确), 2 = HF32x3 补偿 (C64 大尺寸)
};

// 4. Phase 2: 3M 合并/缩放/对角精度保护 Tiling (AIV-only)
struct Csyrk3mExCombineTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t arLdc;
    uint32_t tempLdc;
    uint32_t ldc;
    uint32_t rowsPerCore;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    uint8_t isTransN;
    uint8_t uploMode;      // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint8_t betaOnly;      // 1 = alpha==(0,0) 或 k==0，仅做 beta*C
    uint8_t isBetaZero;    // 1 = 不读取 C_old
    uint8_t isFastCfg;     // 1 = alpha==(1,0) 且 beta==(0,0) 快速路径
    uint8_t useDirectDiag; // 1 = 从 Xr/Xi 批量直接重算对角线，消除 P0-P1 抵消
};
```

### Kernel 设计（Ascend 950PR / dav-3510）

#### 总体执行管线

```text
aclblasCsyrk3mEx (Host)
 ├─ 参数校验 (handle -> uplo/trans -> n/k -> lda/ldc -> alpha/beta -> Atype/Ctype -> A/C)
 ├─ [n == 0] 或 [(k == 0 / alpha == 0) && beta == (1,0)] ──> Quick Return (0 launch)
 ├─ [k == 0 / alpha == 0] && beta != (1,0) ────────────────> Phase 2 AIV (betaOnly, 0 ws)
 ├─ [n <= 64] ─────────────────────────────────────────────> SmallSimt3m AIV (单发射, 0 ws)
 └─ [n > 64] ──> 3 阶段 3M 融合管线 (handle workspace):
      ├─ Phase 0 (AIV): csyrk3m_ex_prep_kernel
      │    A (COMPLEX64 或 INT8 交错) ──> Xr, Xi, Xs(=Xr+Xi) [+ 可选 XrLow, XiLow, XsLow]
      ├─ Phase 1 (AIC, 单次 kernel launch): csyrk3m_ex_cube3m_kernel
      │    仅遍历 uplo 三角 64x64 tile；单核单 tile 内同步计算 P0/P1/P2 三实积
      │    temp = [ P0 = Xr*Xr^T | P1 = Xi*Xi^T | P2 = Xs*Xs^T ]
      └─ Phase 2 (AIV): csyrk3m_ex_combine_kernel
           Cr = P0 - P1, Ci = P2 - P0 - P1 [+ 可选 8 行批量对角重算]
           C_uplo = alpha * (Cr + i*Ci) + beta * C_old (beta==0 不读 C_old；仅写 uplo 三角)
```

#### Path 1：小尺寸单发射 SIMT 3M Kernel（`n <= 64`）

- **核模型**：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，使用 `asc_vf_call` 发射 SIMT 线程束；按 `uplo` 三角有效元素数 `totalTri = n * (n + 1) / 2` 计算 block 数 `blocks = min(aivCoreNum, max(1, ceil(totalTri / 64)))`，保证 `n = 1..64` 低延迟发射且不启动多余空核。
- **三角索引映射**：每个线程根据全局线程 ID 映射到唯一的 `(i, j)`（`UPPER` 满足 `i <= j`，`LOWER` 满足 `i >= j`），绝不触碰另一三角与 `ldc` 行填充。
- **单元素 3M / 补偿累加**：
  - 按 `isInt8A` 分支读取 `A`：`isInt8A == 1` 时从 `int8_t*` 读取交错 2 字节 `(re, im)` 并转为 `float`；`isInt8A == 0` 时读取 `float` 实/虚部对；
  - 遍历 `t = 0..k-1`，对非对角元素（`i != j`）按对称 3M 累加三路部分和 `(p0, p1, p2)`（`p0 += xri * xrj`、`p1 += xii * xij`、`p2 += (xri + xii) * (xrj + xij)`，最后 `cr = p0 - p1`、`ci = p2 - p0 - p1`）；对大 `k` 或对角元素（`i == j`）使用逐项差分 `sum((xri * xri - xii * xii), 2 * xri * xii)` 与 Kahan 补偿累加，从根源消除小 `n` 大 `k`（如 `n=4..64, k=2048`）及 `TC_3M_*` 的抵消误差；
  - Epilogue：`isBetaZero == 1` 时直接写入 `alpha * (cr + i*ci)`，不读取 `C_old`；`isBetaZero == 0` 时读取 `C_old` 做复数乘加并原地写回；对角元素虚部原样保留。

#### Path 2 Phase 0：AIV 预处理 Kernel（`csyrk3m_ex_prep_kernel`）

- **核模型与分核**：`KERNEL_TYPE_AIV_ONLY`，按物理列 `aCols` 均分给 `usedAivCores = min(aCols, GetAivCoreCount())` 个 AIV 核，每核负责连续列段 `[colStart, colEnd)`，列内 GM 访问天然连续合并。
- **`ACL_COMPLEX64` 路径**：
  - 采用双缓冲软件流水（`ab0` / `ab1` 双槽 + 显式 `MTE2_V` / `V_MTE2` / `V_MTE3` / `MTE3_V` 事件），以 `DataCopyPad<float, PaddingMode::Compact>` 一次连续搬入整列（或 `DEINT_BLOCK = 4096` 分段）交错复数；
  - 向量 `DeInterleave(xrUb, xiUb, abUb, 2 * cnt)` 拆分实/虚部，紧接着在 UB 内执行向量 `Add(xsUb, xrUb, xiUb, cnt)` 直接生成 `Xs = Xr + Xi`，省去对 `Xr/Xi` 的二次 GM 读写；
  - 写出 `dXr`、`dXi`、`dXs` 至 workspace（按 `outLdc = CeilAlign(aRows, 8)` 对齐，自动消除 `lda > aRows` 的列间空洞）；
  - 当 `genHf32Resid == 1`（大型 `COMPLEX64` 走 `HF32x3` 时），在同一 kernel 内紧接计算 `x_low = x - HF32(x)`（`round10` 最近偶数舍入残差，Inf/NaN 残差置 0）并写出 `dXrLow`、`dXiLow`、`dXsLow`，相比仓内 `csyrk` 减少 1 次 kernel 发射和 1 轮 `Xr/Xi` GM 读回。
- **`ACL_INT8` 路径**：
  - 源矩阵 `A` 每元素仅 2 字节（`[int8_t re, int8_t im]`），GM 读带宽需求仅为 `COMPLEX64` 的 **1/4**；
  - 按 `INT8_BLOCK = 4096` 个复元素（`8192` 字节）连续读入 UB，利用 Ascend C 向量 `Cast<half, int8_t>` → `Cast<float, half>` 两步完成**零精度损失**的类型提升（所有 `[-128, 127]` 整数在 FP16 与 FP32 中均精确可表示），随后 `DeInterleave` 得 `Xr`、`Xi` 并计算 `Xs = Xr + Xi` 写出至 `dXr`、`dXi`、`dXs`。

#### Path 2 Phase 1：AIC 单发射融合 3M Cube Kernel（`csyrk3m_ex_cube3m_kernel`）

本阶段严格满足任务书 §2.1“三次实数矩阵乘必须在同一次 kernel launch 内完成”的核心约束，基于 DAV_3510 的 `AscendC::Te`（`tensor_api`）构建单发射融合 3M 驱动：

1. **硬件 HF32 状态显式锁定**：kernel 入口显式调用 `AscendC::SetHF32Mode((tiling.cubeMode != 0) ? HF32Mode::ENABLE : HF32Mode::DISABLE)`，防止前序 kernel 残留的全局 `PIPE_S` HF32 状态污染原生 FP32 路径。
2. **对称转置布局零拷贝映射**：
   - `isTransN == 1`（`X * X^T`）：左矩阵使用 `DNExt(arLdc, k)`，右矩阵使用 `NDExt(k, arLdc)`，硬件 `CopyGM2L1` 在搬运时自动完成转置；
   - `isTransN == 0`（`X^T * X`，含 `OP_T` 与 `OP_C`）：左矩阵使用 `NDExt(n, arLdc)`，右矩阵使用 `DNExt(arLdc, n)`；
   - 输出 `P0, P1, P2` 均采用 `DNExt(n, tempLdc)` 列主序布局，与 Phase 2 读取无缝衔接。
3. **三角跳过 + 蛇形顺序 + 保留索引均分**：
   - 将 `n x n` 输出划分为 `64 x 64` 基础 tile 网格（`divM = divN = ceil(n / 64)`），奇数块行反向遍历（蛇形顺序）提升 L2 面板复用率；
   - 根据 `triangleMode` 跳过严格位于非 `uplo` 区域的 tile（`UPPER` 跳过 `mStart > nStart + 63`；`LOWER` 跳过 `nStart > mStart + 63`），将有效 tile 数缩减至约 `divM * (divM + 1) / 2`；
   - 仅对保留的三角 tile 递增 `kept` 计数，并按 `(kept % totalCores) == coreIdx` 均匀分配给全部 AIC 核，避免按全局索引取模造成的三角负载倾斜。
4. **单 Owner Tile 融合三积与双 L0C Ping-Pong**：
   - 每个 `64 x 64` 输出 tile 需要 **3 个 FP32 L0C 累加器**（`accP0, accP1, accP2`），每个占 `64 * 64 * 4B = 16 KB`，一组共 `48 KB`（相比 4M 的 `64 KB` 节省 25%）；
   - 开辟 2 组 L0C 累加器（`Set 0` 偏移 `0 KB` 占 `48 KB`，`Set 1` 按 `64 KB` 对齐偏移占 `48 KB`，有效占用 `96 KB`、地址跨度 `112 KB`，远小于 DAV_3510 `256 KB` L0C 容量），相邻输出 tile 通过 `FIX_M(0/1)` / `M_FIX(0/1)` 交替 ping-pong，实现当前 tile 的 `Mmad` 与上一 tile 3 次 `Fixpipe` 写回的流水重叠。
5. **三种 Cube 计算模式与片上 L1/L0 预算**：

| `cubeMode` | 适用场景 | K-chunk 块长 | 每 L1 缓冲内容与大小 | L1 环形缓冲数与总量 | 每 K-chunk MMAD 次数 | 精度特征 |
| --- | --- | --- | --- | --- | --- | --- |
| `0`：原生 FP32 | `ACL_COMPLEX64` 中等尺寸（`64 < n <= 256` 且 `k <= 512`） | `CHUNK_K = 64`（或 `128`） | 6 个面板（M 侧 `Xr/Xi/Xs` + N 侧 `Xr/Xi/Xs`）× `16 KB` = `96 KB` | 4 缓冲 = **384 KB** ≤ 512 KB | **3 次** FP32 MMAD（`P0, P1, P2`） | 全 23-bit 尾数输入 + FP32 累加 |
| `1`：单遍 HF32 | `ACL_INT8` 全部 `n > 64` 形状 | `CHUNK_K = 64`（或 `128`） | 同上 6 个面板 = `96 KB` | 4 缓冲 = **384 KB** ≤ 512 KB | **3 次** HF32 MMAD（`P0, P1, P2`） | `int8`（≤7-bit）与 `Xr+Xi`（≤8-bit）在 HF32 10-bit 尾数下 **零舍入误差** |
| `2`：HF32x3 补偿 | `ACL_COMPLEX64` 大尺寸（`n > 256` 或 `k > 512`） | `CHUNK_K = 64` | 12 个面板（M/N 两侧各 3 高位 + 3 残差）× `16 KB` = `192 KB` | 2 缓冲双缓冲 = **384 KB** ≤ 512 KB | **9 次** HF32 MMAD（每积 3 次：`xl*yh + xh*yl + xh*yh`） | ~21-bit 等效输入精度，比 4M HF32x3（12 次 MMAD）少 **25%** |

在 L0A/L0B 侧（单块 `64 x 64` FP32 为 `16 KB`）：
- `cubeMode = 0/1`：每次搬入 3 个 M 面板（`48 KB`）与 3 个 N 面板（`48 KB`），或按 `BASE_K = 32`（每面板 `8 KB`，3 面板仅 `24 KB`）实现 L0A/L0B 双槽 ping-pong（总 `48 KB`），隐藏 `L1 -> L0` 搬运延迟；
- `cubeMode = 2`：按 `BASE_K = 32` 将单个 64-K chunk 拆为两拍 `32-K` 子块（6 个面板 × `8 KB` = `48 KB` 放入 L0A/L0B），对每个乘积 `c ∈ {0, 1, 2}` 依次发射 3 条有序 MMAD：`xl * yh`（首 K 块置 `initAcc=true`）、`xh * yl`、`xh * yh`（末 K 块置 `unitFlag=3`），最后 `Fixpipe` 将 3 个 L0C 累加器排入 `dTemp` 的 `[P0 | P1 | P2]` 对应位置。

#### Path 2 Phase 2：AIV 3M 合并/缩放与严格三角写回 Kernel（`csyrk3m_ex_combine_kernel`）

- **核模型与分核**：`KERNEL_TYPE_AIV_ONLY`，对 `n <= 640` 采用按行带分核（`usedAiv = min(ceil(n / 16), aivCoreNum)`），对 `n > 640` 采用 2D 三角 `64 x 64` tile 轮转分核（满核利用全部 AIV 核），确保负载均衡。
- **3M 实/虚部重构**：从 `dTemp` 读取当前 `uplo` tile 的 `P0`、`P1`、`P2` 三块 FP32 数据至 UB：
  ```cpp
  Sub(crUb, p0Ub, p1Ub, cnt);      // Cr = P0 - P1
  Sub(ciUb, p2Ub, p0Ub, cnt);      // tmp = P2 - P0
  Sub(ciUb, ciUb, p1Ub, cnt);      // Ci = P2 - P0 - P1
  ```
- **快速配置（`isFastCfg == 1`：`alpha == (1,0)` 且 `beta == (0,0)`）**：
  - 覆盖全部 200 条 `TC_PF_*` 性能用例；
  - 跳过 `C_old` 读取与复数标量乘法，直接 `Interleave(outUb, crUb, ciUb)` 写回 `C`，采用双缓冲重叠 `MTE2` 读 `P0/P1/P2`、`V` 合并交错与 `MTE3` 写 `C`。
- **通用配置与 `betaOnly` 模式**：
  - `isBetaZero == 1`：不读取 `C_old`，仅执行 `alpha * (Cr + i*Ci)`（`betaOnly` 时直接写 0）；
  - `isBetaZero == 0`：按 2D `DataCopyPad` 仅读 `uplo` 区域 `C_old`、`DeInterleave` 拆分后完成复数乘加并 `Interleave` 写回；
  - **严格 `uplo` 三角与 padding 写保护**：
    - 完整落在 `uplo` 内的非对角 `64 x 64`（或 `64 x 32`）tile 使用 2D `DataCopyPad` 一次写回，步长跳过 `ldc - rows`；
    - 跨越主对角线的边界 tile 逐行/按 8 行带计算精确有效列区间（`UPPER` 写 `[i, jEnd)`，`LOWER` 写 `[jStart, i]`），仅向有效区间发起 `DataCopyPad` 写出，保证非 `uplo` 三角及 `[n, ldc)` 行填充零污染；
    - 对角元素 `C(i, i)` 的实部与虚部同其余元素一并写出，**绝不清零虚部**。
- **大 `k` 对角精度保护（`useDirectDiag`）**：
  - 原因：对角位置 `P0(i, i) = Sum(Xr^2) >= 0`、`P1(i, i) = Sum(Xi^2) >= 0`，两项大正平方和相减 `P0 - P1` 在 `k` 较大（如 `k > 512`）或正负交替输入（`TC_3M_*`）下会产生严重有效位抵消，导致单元素误差超过 `max_abs_error <= 1e-2`；
  - 策略：当 `useDirectDiag == 1`（`k > 512` 或 `n <= 128 && k > 128`）时，仅针对 `n` 个对角元素（总计算量 `O(n * k)`，远小于 `O(n^2 * k)` 矩阵乘），复用 `csyrk_kernel.cpp` 已验证的 8 行批量连续读 + `Deint8Rows`（`trans=N`）或连续列读（`trans=T/C`）在 AIV 上直接按 `Sum(Xr^2 - Xi^2)` 与 `Sum(2 * Xr * Xi)` 累加得到高精度 `(diagRe, diagIm)`，在写回对角条带时覆盖 `(Cr(i,i), Ci(i,i))`，既消除 3M 对角抵消又完整保留对角虚部。

#### 片上 UB 内存预算（AIV 阶段）

按 Ascend 950PR 单 AIV UB 容量做保守预算（控制单核 UB 总占用 ≤ 192 KB 留足管理余量）：

| 阶段 / 配置 | 主要 UB 缓冲 | 典型尺寸 | 小计 |
| --- | --- | --- | --- |
| Phase 0 Prep（`COMPLEX64`） | 双缓冲交错输入 `2 * 2 * 4096 * 4B` + `Xr/Xi/Xs` 3 路输出 `3 * 4096 * 4B` | `64 KB + 48 KB` | **112 KB** |
| Phase 0 Prep（`INT8`） | `int8` 输入 `8 KB` + `half` 缓冲 `16 KB` + `float` 交错 `32 KB` + `Xr/Xi/Xs` `48 KB` | `8 + 16 + 32 + 48 KB` | **104 KB** |
| Phase 2 Combine（Fast `64x64` 双缓冲） | `P0/P1/P2` 双缓冲 `3 * 2 * 16 KB` + `out` 交错缓冲 `32 KB` + 对角暂存 `24 KB` | `96 + 32 + 24 KB` | **152 KB** |
| Phase 2 Combine（Full `64x32` 通用） | `P0/P1/P2` `24 KB` + `C_old` `16 KB` + `re/im/t1/t2/outRe/outIm` `48 KB` + `out` `16 KB` + 对角暂存 `24 KB` | `24 + 16 + 48 + 16 + 24 KB` | **128 KB** |

#### 3M 数值稳定性与特殊值处理策略

1. **对角 `P0 - P1` 抵消**：通过 `useDirectDiag` 批量直接重算消除（仅 `O(n*k)` 开销，实测占整体耗时极小）。
2. **非对角 `P2 - P0 - P1` 抵消与 `TC_3M_*` 专项**：
   - `TC_3M_001`（`n=128, k=128, RANDOM_ALTER`）、`TC_3M_002`（`n=128, k=64, RANDOM_EXTREME`）、`TC_3M_003`（`n=64, k=64, INT8_EXTREME`）、`TC_3M_004`（`n=128, k=64, RANDOM_DIAG_IMAG`）：
   - `n <= 64` 直接走单发射补偿 SIMT 路径；
   - `64 < n <= 256 && k <= 512` 走原生 FP32 Cube（全 23-bit 尾数，不引入 HF32 10-bit 量化误差）；
   - `n * k <= 128 * 128` 触发 `SpecialFallback` 安全扫描（仅在含 `Inf/NaN`、`|v| > 1e19` 或 subnormal 时切 Host Netlib 参考，避免 `P0 - P1` 出现 `Inf - Inf = NaN`；常规有限输入零回退）。
3. **`ACL_INT8` 极值路径**：`int8` 极值 `{-128, 127}` 乘积为精确整数，单项最大 `128^2 = 16384`，在 `k <= 1024` 内累加值不超过 `2^24`（FP32 精确整数范围），且单遍 HF32 无任何输入尾数截断，精度天然稳定。

### 资源与调度

| 维度 | 设计方案 |
| --- | --- |
| 核资源模型 | 小形状及 `betaOnly` 纯 AIV；常规大形状采用 AIV（Prep）→ AIC（单发射融合 3M Cube）→ AIV（Combine）顺序流水，同 stream 硬件保序 |
| 核间同步 | 各阶段内严格按列段 / 三角 tile 划分唯一 owner，**无核间原子操作、无 `SyncAll` 跨核屏障** |
| Workspace 管理 | 复用 handle 默认 32 MiB workspace（按需倍增至 2 GiB）；每次调用稳态零 `aclrtMalloc`；QuickReturn / BetaOnly / SmallSimt3m 路径零 workspace |
| 并发与隔离 | 不同 handle 独立管理各自 stream 与 workspace；不在静态全局存储输入或中间数据，支持多 stream 并发 |
| 异步提交 | 计算 kernel 全部提交至 `handle->stream`，不阻塞调用线程 |

### 支持硬件

| 芯片版本 | 架构目录 | 编译目标 / 宏门控 | 支持状态 |
| --- | --- | --- | --- |
| Ascend 950PR | `arch35`（`dav-3510`） | `--soc=ascend950`，`ASC_DEVKIT_GE_9_1`（CANN >= 9.1.0） | 目标开发与自验支持设备 |
| Ascend 950DT | `arch35`（`dav-3510`） | 同 `arch35` 二进制 | 架构兼容（自验按任务书 §3.1 仅需覆盖 950PR） |
| Atlas A2 / A3（`arch22`） | — | 不涉及 | 本任务不涉及 |

### 算子约束限制

| 约束项 | 限制说明与处理策略 |
| --- | --- |
| 支持数据类型 | 仅支持 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)`；其余返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 枚举值范围 | `uplo` 仅允许 `ACLBLAS_UPPER` / `ACLBLAS_LOWER`，`trans` 仅允许 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C`；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 维度与前导维 | `n >= 0`、`k >= 0`；`OP_N` 要求 `lda >= max(1, n)`，`OP_T/OP_C` 要求 `lda >= max(1, k)`；`ldc >= max(1, n)`；越界返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 指针约束 | `handle == nullptr` 返回 `HANDLE_IS_NULLPTR`；`alpha/beta == nullptr`、`n > 0` 时 `C == nullptr`、`n > 0 && k > 0 && alpha != (0,0)` 时 `A == nullptr`、或 Host/Device 标量混用均返回 `INVALID_VALUE` |
| Workspace 上限 | 大尺寸 3M 管线依赖 handle workspace，总需求超过 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）或超出用户自设 workspace 时返回 `ACLBLAS_STATUS_ALLOC_FAILED` |
| 内存重叠 | `C` 为原地更新，调用方须保证 `C` 的 `uplo` 可写区域与 `A` 的可读区域不重叠 |
| 3M 逐位一致性 | 3M 算法中间 FP32 舍入路径与直接复数乘（`cblas_csyrk` / `aclblasCsyrk`）不同，不保证逐位一致，按 §3.2 容差阈值判定 |
| 广播 / 0 维 / 动态 shape | 不涉及 |
| 工具链版本 | 依赖 `tensor_api` 与 DAV_3510 指令，要求 CANN 9.1.0 及以上（`asc-devkit >= 9.1`） |

# 可维可测分析

## 精度标准与验证

| 校验项 | 验收口径 |
| --- | --- |
| CPU Golden | Netlib BLAS 3.10.0 / CBLAS 非 3M 参考实现 `cblas_csyrk`（列主序 `CblasColMajor`；float64 精度域可用 `cblas_zsyrk` 后收窄至 complex64）；`Atype=ACL_INT8` 时先将每个 `(int8 re, int8 im)` 无损提升为 float/double 再调用 |
| 比对范围与分量 | 仅比对 `uplo` 指定三角（含主对角线），将 `COMPLEX64` 拆分为**实部（FLOAT32）与虚部（FLOAT32）分别判定** |
| 对角虚部检验 | 对称语义下对角元素虚部按普通有效分量参与比对，绝不按 Hermitian 清零；由 `diag_imag_kept`、`3m_diag_imag_keep` 专项用例验证 |
| 写保护检验 | 非 `uplo` 三角及 `ldc > n` 行填充区在调用前写入哨兵值，调用后执行 **EXACT 逐位比对**，确保未被污染 |
| 混合容差阈值（FLOAT32 分量） | `rtol = 2^-13`（约 `1.2207e-4`）、`atol = 2^-13`（约 `1.2207e-4`）、`required_matched_ratio = 0.99`、`max_abs_error_limit = 1e-2`（或 `32 * ULP`） |
| 逐元素判定条件 | 满足 `abs(actual - golden) <= atol + rtol * abs(golden)` 计入 matched；须同时满足 `matched_ratio >= 0.99` 且最大绝对误差不超过上限 |
| 随机类算子判定 | 不涉及：本算子为确定性浮点乘加运算，不含随机数生成；任务书 §3.5 的均匀分布与正态分布仅用于构造测试输入数据 |
| 精度用例总量 | **1000 条**（`Atype` 约 50% `ACL_COMPLEX64` + 50% `ACL_INT8`；`COMPLEX64` 输入按 50% 均匀 `RANDOM_NORM_5_5` + 50% 正态 `RANDOM_GAUSS_0_1` 独立采样） |

**1000 条精度用例分类覆盖（对照 `test_cases/README.md` 与 `gen_csv.py`）**：

| 类别前缀 | 条数 | 覆盖要点 |
| --- | --- | --- |
| `TC_L0` | 12 | 6 组 `uplo × trans` 全组合 × 小尺寸 `(4, 8)` |
| `TC_SQ` | 204 | 34 种尺寸（1 到 4096，含 2 的幂及 ±1 非对齐值）× 6 组组合 |
| `TC_AB` | 48 | 8 组复数 `alpha/beta` 组合 × 6 组枚举组合 |
| `TC_WS` / `TC_TH` | 24 | 宽矩阵 `k >> n`（`trans=N`）与高窄矩阵 `n >> k`（`trans=T`） |
| `TC_LD` | 27 | `lda`/`ldc` 紧凑 + padding（`+1/+8/+64`）与写保护验证 |
| `TC_FL` | 12 | `COMPLEX64` 均匀 / 全零 / 交替 / 极端 / `Inf` / `NaN` 填充（`A`、`C` 分别覆盖） |
| `TC_CV` | 48 | 中等尺寸（10 ~ 400）× 6 组枚举组合 |
| `TC_ED` | 38 | `n=0`、`k=0`（含 `beta=(1,0)` 与 `beta != (1,0)`）、`alpha=(0,0)` + `A=NaN`、`beta=(0,0)` + `C=NaN`、对角虚部保留、`±1e4` 标量、`int8` 边界/全零、空指针、非法枚举（`INVALID_ENUM`）、不支持 dtype（`NOT_SUPPORTED`）、非法 `lda`/`ldc`（下界/0/负数）与负维度 |
| `TC_3M` | 4 | 3M 专项：交替/极值填充下的实虚部抵消、`int8` 极值、对角虚部保留 |
| `TC_EX` | 583 | 尺寸池 × 枚举 × 标量 × padding 的确定性扩展采样 |

另在 `csyrk3m_ex_test.cpp` 中通过独立 `TEST_F` 补充 CSV 无法表达的场景：`NullHandle`（期望 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）及 Host / Device 两种标量指针模式验证（含混用返回 `ACLBLAS_STATUS_INVALID_VALUE`）。

## 性能标准与验证

| 验收项 | 验收口径 |
| --- | --- |
| 验收设备 | Ascend 950PR（`arch35`） |
| 正式采集方式（任务书 §3.3.1） | 使用 `msprof op` 采集 GTest 性能用例的 `OPPROF_*/OpBasicInfo.csv`，统计单次 `aclblasCsyrk3mEx` 调用涉及的 NPU kernel `Task Duration(us)` 之和：丢弃前 5 次 warmup，取后续 10 次有效样本算术平均（剔除非 SUCCESS 样本并在报告说明），除以 1000 转为 ms |
| 正式验收判据（kernel-to-kernel） | NPU kernel 平均耗时与 `gpu_baseline.csv` 的 GPU kernel-only 基线 `gpu_ms` 对比：1）单 case 加速比 `gpu_ms / npu_ms >= 0.4`（即 `npu_ms <= gpu_ms / 0.4`）；2）全部 200 条可比性能 case 加速比算术平均 `>= 0.5`；两项须同时满足 |
| 辅助快速回归（§3.3.2，非正式判据） | GTest 在预置 Device 缓冲上 warmup 5 次 + `aclrtEvent` 夹住 10 次下发取平均，按 `[PERF] case=<case_name> ... avg_us=<数值>` 格式输出，供 `verify_performance.py` 快速回归；**严禁**用端到端 `avg_us` 替代 `msprof op` kernel 耗时作为正式验收结论 |
| `Atype=ACL_INT8` 性能补充（§3.3.6） | 随包 200 条 `TC_PF` 全部为 `ACL_COMPLEX64`（其中 2 条与典型 case 同参）；开发者在 950PR 上额外补充采集典型规模（如 512/1024/2048/4096）的 `Atype=ACL_INT8` `msprof` 性能数据，并沿用同形状 `ACL_COMPLEX64` 的 `gpu_ms` 基线对比写入自测报告 |

**代表性性能锚点与目标预算（引自 `test_cases/gpu_baseline.csv`，非实测数据）**：

| Baseline ID | `uplo` | `trans` | `n` | `k` | `gpu_ms`（GPU kernel-only） | 单 case 达标上限（`gpu_ms / 0.4`） | 均值参考线（`gpu_ms / 0.5`） | 关键优化抓手 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `csyrk3m_ex-perf-007` | UPPER | N | 1 | 1 | 0.002477 ms | ≤ 0.006192 ms | ≤ 0.004954 ms | `SmallSimt3m` 单发射（1 核、零 workspace） |
| `csyrk3m_ex-base-000` | UPPER | N | 512 | 512 | 0.045306 ms | ≤ 0.113265 ms | ≤ 0.090612 ms | 3M 融合 Cube + Fast Combine（可演进 Mix 融合降 launch 开销） |
| `csyrk3m_ex-base-004` | UPPER | C | 1024 | 256 | 0.063767 ms | ≤ 0.159417 ms | ≤ 0.127534 ms | `OP_C` 零共轭复用 `OP_T` + 三角跳过省 50% tile |
| `csyrk3m_ex-base-001` | UPPER | N | 1024 | 1024 | 0.254358 ms | ≤ 0.635894 ms | ≤ 0.508715 ms | 单发射 3M Cube + 双 L0C ping-pong 隐藏 Fixpipe |
| `csyrk3m_ex-base-003` | LOWER | T | 2048 | 2048 | 1.270021 ms | ≤ 3.175053 ms | ≤ 2.540042 ms | 3M 比 4M 少 25% MMAD + 融合残差省 1 次发射 |
| `csyrk3m_ex-base-002` | UPPER | N | 2048 | 2048 | 1.417319 ms | ≤ 3.543296 ms | ≤ 2.834637 ms | 蛇形保留索引均分满占 AIC 核 + 8 行批量对角重算 |
| `csyrk3m_ex-perf-091` | UPPER | N | 4096 | 4096 | 8.636193 ms | ≤ 21.590483 ms | ≤ 17.272386 ms | 大矩阵 Compute-bound，3M HF32x3 满吞吐发挥 |

## 测试工程与覆盖

测试工程新建于 `test/syrk/csyrk3m_ex/`，与 `ops-blas` ATK 测试框架集成：

- **`CMakeLists.txt`**：调用 `ops_blas_add_gtest_tests(${OPS_BLAS} csyrk3m_ex_test)`，为 CPU golden 开启 `-O2` 编译选项（参考 `test/syrk/csyrk/CMakeLists.txt` 避免 `-O0` 导致大尺寸 golden 耗时过长）；
- **`csyrk3m_ex_param.h`**：解析全部 20 列 CSV 字段（含 `Atype`、`Ctype` 枚举解析、`alpha_real/imag`、`beta_real/imag`、`null` 标量空指针与 `NULLPTR` 矩阵空指针），并扩展支持任务书要求的填充 token：`RANDOM_GAUSS_0_1`（N(0,1) 正态分布）、`RANDOM_INT8`（实/虚部独立均匀 `[-128, 127]`）、`VALUE_INT8_EXTREME`（`{-128, 127}` 极值）、`RANDOM_DIAG_IMAG`（对角线虚部显式填非零值）；
- **`csyrk3m_ex_golden.h`**：按 `Atype` 准备输入（`ACL_INT8` 精确提升为 float/double），调用 CBLAS `cblas_csyrk`（或 `cblas_zsyrk` 收窄）生成 CPU 真值；
- **`arch35/csyrk3m_ex_npu_wrapper.h` 与 `arch35/csyrk3m_ex_test.cpp`**：
  - 负向用例不按异常 `lda/ldc` 预分配非法尺寸内存，以最小占位缓冲或 `nullptr` 调用接口验证预期错误码；
  - 精度用例执行非 `uplo` 三角与 `ldc` 填充区哨兵写保护检查 + 实/虚部分离 `applyMixedTolerance(..., ACL_FLOAT, ...)` 校验；
  - 性能用例（`TC_PF_*`）预分配并复用 Device 缓冲，先 warmup 5 次再用 `aclrtEvent` 测量连续 10 次下发均值并按 `[PERF] case=<case_name> ... avg_us=<数值>` 打印。

### 自验步骤（代码验收阶段执行）

```bash
# 1. 编译算子库与 csyrk3m_ex 测试可执行文件
bash build.sh --ops=csyrk3m_ex --soc=ascend950

# 2. 全量 1000 条精度自验（安装任务包 CSV 并运行 GTest，排除 TC_PF_*）
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./csyrk3m_ex_test.csv

# 3. 性能快速交叉回归（解析 [PERF] avg_us 与 gpu_baseline.csv 比对，非正式判据）
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend950

# 4. 正式性能采集（msprof op kernel-to-kernel 口径，按任务书 §3.3.1 统计 Task Duration）
msprof op --application="./build/test/syrk/csyrk3m_ex/csyrk3m_ex_test --gtest_filter=*TC_PF*" --output=./prof_csyrk3m_ex
```

## 兼容性分析

| 维度 | 结论 |
| --- | --- |
| 公共 API 与 ABI | 在 `include/cann_ops_blas.h` 末尾新增 `aclblasCsyrk3mEx` 声明，不改动任何既有函数签名、枚举值或结构体布局，向后完全兼容 |
| 与 `aclblasCsyrk` 共存 | `blas/syrk/arch35/` 下独立前缀 `csyrk3m_ex_*` 源文件，内部符号与 `csyrk_*` 完全隔离，互不影响 |
| 与 `aclblasCsyrkEx` 解耦 | 两者作为 1125 独立接口互不依赖、互不调用 |
| 多架构与工具链 | 仅在 `arch35`（`ascend950*`）下编译，且经 `TENSOR_API_OPS` 门控要求 `asc-devkit >= 9.1`，不影响 `arch22`/`arch20` 及低版本工具链构建 |
| 文档与测试 | `blas/syrk/README.md` 增量追加章节；`test/syrk/csyrk3m_ex/` 为独立测试目标，不影响既有 `csyrk`/`ssyrk` 测试套件 |

## 已知限制与规避

| 风险 / 限制 | 潜在影响 | 规避与应对策略 |
| --- | --- | --- |
| 3M 减法抵消（对角 `P0 - P1`、非对角 `P2 - P0 - P1`） | 大 `k` 或交替/极值输入可能触发 `max_abs_error` 超限 | 大 `k` 开启 8 行批量直接对角重算；`n <= 64` 走补偿 SIMT；中等形状走原生 FP32 Cube；小矩阵极值/Inf/NaN 走安全参考回退 |
| 3 阶段发射开销（中等形状如 `512 x 512`） | 若 Prep + Cube + Combine 三次 kernel 间隙过大，可能压缩 `0.4x` 性能余量 | 将 HF32 残差融合进 Prep（省 1 次发射）；全 `TC_PF` 走 Fast Combine 路径；实机若中等形状 launch 开销显著，将 Phase 1 与 Phase 2 融合为单 AIC+AIV Mix kernel |
| 大尺寸 `COMPLEX64` 吞吐与精度权衡 | 单遍 HF32 精度不足，纯 FP32 Cube 吞吐可能不及 HF32x3 | 同时设计原生 FP32 与 HF32x3 补偿两条 Cube 驱动，实机按 `msprof` 实测拐点择优切换 |
| 设备标量指针 D2H 同步开销 | `alpha/beta` 在 Device 内存时需 D2H 读取判断 `alpha==0`/`beta==0` | 仅 Device 标量且 `n > 0` 时发起一次合并 D2H + 同步；Host 标量零同步（与 `gpu_baseline.csv` Host pointer mode 采集条件一致） |
| 用户自定义小 workspace | `n > 64` 且用户通过 `aclblasSetWorkspace` 设置过小缓冲时无法分配中间矩阵 | 按 handle 契约返回 `ACLBLAS_STATUS_ALLOC_FAILED` 并日志提示所需字节数；`n == 0`、`n <= 64`、`betaOnly` 均零 workspace 不受影响 |
| 性能测试包不含 `INT8` 行 | `verify_performance.py` 默认仅统计 200 条 `COMPLEX64` 用例 | 按任务书 §3.3.6 单独补充 `Atype=ACL_INT8` `msprof op` 性能采集并与 `ACL_COMPLEX64` `gpu_ms` 基线对比写入自测报告 |

## 交付物对照（任务书 §4）

| 交付件 | 本设计对应安排 |
| --- | --- |
| 1. 算子设计文档 | 本 PR 提交至 `cann/cann-ops-competitions`（仅 1 个新增文件 `docs/design.md`） |
| 2. 自测用例及测试代码 | 代码验收阶段提交 `test/syrk/csyrk3m_ex/` 完整 ATK 测试工程及任务包 `test_cases/`（1000 精度 + 200 性能） |
| 3. 自测报告 | 代码验收阶段按官方模板填写，含实/虚部分离精度统计、`msprof op` 逐 case kernel 耗时（含 `ACL_INT8` 补充）及截图 |
| 4. 待验收代码地址 | 个人 `ops-blas` 仓库分支（邀请 `Ascend-CANN` 为开发者），含 `include/cann_ops_blas.h`、`blas/syrk/arch35/`、`test/syrk/csyrk3m_ex/arch35/` 及 `blas/syrk/README.md` |
| 5. `task_submission` 目录 | 验收阶段按模板在开发者私仓根目录提交自验证步骤说明、精度/性能/内存自验证报告及完整日志 |

## AI 辅助披露

本文档撰写过程中使用了 AI 编程助手辅助梳理任务书、`ops-blas` 开源仓源码与 Ascend 950PR（`arch35`）架构资料，提交者已逐节复核；文中涉及的公共头文件现状、目录结构、CMake 规则与基线数据均以 `ops-blas` 当前 `master` 源码及随包 `test_cases/` 文件核对为准。

## 修订记录

| 版本 | 日期 | 说明 |
| --- | --- | --- |
| v1.0 | 2026-11-25 | 初稿：完成需求背景、需求分析、对称 3M 单发射架构与 Host/Kernel 详细设计、可维可测与 §2.5 约束逐项核对 |

---

*参考：cuBLAS `cublasCsyrk3mEx`、Netlib `csyrk.f`、`ops-blas` `include/cann_ops_blas.h`、`blas/syrk/arch35/csyrk_*`、`blas/gemm/arch35/cgemm3m_*`、`test/syrk/csyrk/`，以及任务书 `aclblasCsyrk3mEx_Atlas950PR_task_doc.md` 与 `test_cases/README.md`。*
