# 需求背景

## 需求来源

本设计对应 2026 年 11 月社区任务《11月社区任务-aclblasSgemmBatched_HgemmBatched算子开发（A2A3）》。任务要求在 Atlas A2/A3 系列产品上，基于 [ops-blas](https://gitcode.com/cann/ops-blas) 工程和 Ascend C，补齐单精度实数批量矩阵乘 `aclblasSgemmBatched` 与半精度实数批量矩阵乘 `aclblasHgemmBatched`。

设计文档按 [2026 社区任务设计模板](../../../../resources/design_template.md) 和社区流程 #39 组织。当前 11 月任务列表尚未给出本任务编号，因此目录使用任务书完整名称，贡献者目录使用 GitCode 账号 `m0_65133328`。

## 背景介绍

两个接口分别对齐 cuBLAS `cublasSgemmBatched` 与 `cublasHgemmBatched`，针对每个 batch 独立计算：

\[
C_i = \alpha \operatorname{op}(A_i)\operatorname{op}(B_i) + \beta C_i,
\quad i=0,1,\ldots,\text{batchCount}-1.
\]

其中 \(A_i\)、\(B_i\)、\(C_i\) 由 Device 指针数组给出，矩阵采用列主序和前导维访问。ops-blas 已有 `arch35` 的 S/H 实现，也已有 `arch22` 复数批量 GEMM 的 Matmul 工程实践，但 `arch22` 尚无实数 S/H 批量 GEMM。

# 需求分析

## 算子功能

`op(A)` 为 \(m \times k\)，`op(B)` 为 \(k \times n\)，\(C\) 为 \(m \times n\)。支持 `N`、`T`、`C` 三种转置枚举，实数算子的 `C` 与 `T` 等价。

输入输出语义如下：

| 参数 | 类型 | 语义 |
| --- | --- | --- |
| `transa/transb` | `aclblasOperation_t` | 分别选择 `op(A)`、`op(B)` 的转置方式 |
| `m/n/k` | `int` | 统一的问题维数 |
| `alpha/beta` | Host 标量指针 | S 为 FP32，H 为 FP16 |
| `Aarray/Barray` | Device 指针数组 | 每个 batch 指向独立的 A/B 矩阵 |
| `lda/ldb/ldc` | `int` | 列主序前导维 |
| `Carray` | Device 指针数组 | 原地输出，每个 batch 指向独立的 C 矩阵 |
| `batchCount` | `int` | batch 数 |

边界行为：

- `m=0`、`n=0` 或 `batchCount=0` 返回 `ACLBLAS_STATUS_SUCCESS`，不启动 kernel。
- `k=0` 或 `alpha=0` 时仅执行 \(C=\beta C\)，不读取 A/B。
- `beta=0` 时不读取 C 的旧值。
- `beta=1` 且 `k=0` 或 `alpha=0` 时保持 C 不变。
- 仅写入每个 \(C_i\) 的 \(m \times n\) 逻辑区域，`ldc` padding 保持不变。

## 公开接口与范围

接口声明已存在，本任务不新增或修改公共 API：

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount);

aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const Carray[], int ldc,
    int batchCount);
```

Host 按以下顺序校验：handle、`transa/transb`、`m/n/k`、`batchCount`、`lda/ldb/ldc`、`alpha/beta`、顶层指针数组。非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余非法结构参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

`batchCount>0` 且 `m,n>0` 时要求 `Carray` 非空；`k>0` 且 `alpha!=0` 时要求 `Aarray`、`Barray` 非空。上述退化路径不扫描 Device 指针数组元素。

# 详细设计

## 算子分析

公开接口为列主序。内核使用行主序 Matmul 计算转置后的结果：

\[
C_i^\top = \operatorname{op}(B_i)^\top
           \operatorname{op}(A_i)^\top.
\]

内部 GEMM 的维数为 `M=n`、`N=m`、`K=k`。转置枚举只决定 Matmul 对操作数原始存储方向的解释，不产生额外的完整矩阵转置。

批量矩阵由 Device 指针数组逐 batch 解引用。各 batch 的行列尺寸、转置方式、前导维和标量相同；地址和数据相互独立。

## 算子实现

实现位于 `blas/gemm_batched/arch22/`，复用现有 ops-blas handle/stream、workspace 和 Matmul 工程设施。

### Host侧设计

Host 入口分为四步：

1. 完成结构参数校验和 quick return 判定。
2. 计算内部 tiling、工作区布局和输出 tile 数。
3. 根据前导维选择直接读取或先执行 AIV compact pack。
4. 在 `handle` 绑定的 stream 上依次启动 pack、Cube GEMM 和 epilogue。

GEMM 输出 tile 为 \(128 \times 128\)。任务空间由 batch 和输出 tile 共同组成，Host 将任务数限制到当前 AIC 核数。若 `lda` 或 `ldb` 与物理行数不一致，则由 AIV 将每个 batch 的列连续数据压紧到独立 workspace plane；紧凑布局直接使用调用方 Device 矩阵地址。

当 `alpha=1`、`beta=0` 且 `ldc=m` 时，Cube 结果直接写入 C。其他标量或带 padding 的 C 先写入 FP32 workspace，再由 epilogue 写回，保证仅修改逻辑区域。

工作区偏移按 512 Byte 对齐；A、B plane 和 FP32 输出 plane 均按 batch 独立编址。Host 使用宽整数计算元素数和字节数，并在进入分配器前检查溢出。分配失败返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

### Kernel侧设计

**Pack kernel（AIV）**

- 从 Aarray/Barray 读取当前 batch 的 Device 地址。
- 按列执行 `DataCopyPad`，将带 `lda/ldb` padding 的列复制到紧密 plane。
- 每个 AIV 核处理不同的 batch/column job，plane 之间无写重叠。
- 紧凑输入跳过该 kernel。

**GEMM kernel（AIC）**

- AIC 任务按 batch、内部 M tile、内部 N tile 展开。
- 从 raw operand 或 packed plane 构造 `GlobalTensor`，按 `transb` 选择左侧 Matmul 方向，按 `transa` 选择右侧 Matmul 方向。
- S 使用 FP32 Matmul；H 使用 FP16 输入、FP32 累加、FP32 临时输出。
- `SetOrgShape` 同时描述内部 M/N/K 和输出行跨度；`SetSingleShape` 描述当前 tile。
- 不同 AIC 核以 block-stride 方式遍历任务，覆盖全部 batch 和输出 tile。

**Epilogue kernel（AIV）**

- 以 C 的列作为连续 job，读取 FP32 临时结果。
- 计算 \(\alpha T + \beta C_{\text{old}}\)。
- H 将结果按 IEEE round-to-nearest-even 转为 FP16。
- `beta=0` 时不读取 C 旧值，避免传播旧 C 中的 NaN。
- 写回仅覆盖 \(m\) 个逻辑行，保留 `ldc` padding。

**Early-exit kernel（AIV）**

`k=0` 或 `alpha=0` 时启动独立 scale kernel，按 batch 执行 \(C=\beta C\)。`beta=0` 写零但不读 C；`beta=1` 在 Host 直接返回。

### 资源与调度

- Cube Matmul 静态 tiling 的最大单次 K 为 8192；任务书正式用例最大 K 为 4096。
- C、A、B workspace 以 512 Byte 为边界；epilogue 使用 2048 个元素的 UB 分块。
- AIV 负责 pack、scale 和 epilogue，AIC 负责 GEMM，各阶段按同一 stream 串行依赖。
- 工作区优先使用 `EnsureDefaultWorkspace`；handle 已有足够空间时直接复用。
- batchCount 较大且矩阵为 \(1 \times 1 \times 1\) 时，任务空间仍保持在单次 GEMM launch 内，不按 batch 发起多次 Host launch。

## 支持硬件

支持 Atlas A2/A3 系列产品，代码目录为 `arch22`。自验证硬件为 Atlas 800T A2（Ascend 910B3），软件基线为 CANN 9.1.0。

## 算子约束限制

- 数据类型固定为 S=FLOAT32、H=FLOAT16，不做隐式混合类型转换。
- `alpha/beta` 为 Host 指针；Aarray、Barray、Carray 及其元素为 Device 内存。
- 列主序前导维下限分别为：
  - `transa=N`：`lda >= max(1,m)`；否则 `lda >= max(1,k)`。
  - `transb=N`：`ldb >= max(1,k)`；否则 `ldb >= max(1,n)`。
  - `ldc >= max(1,m)`。
- 调用方保证各 batch 的 Device 指针元素有效，并保证 C[i] 之间以及 C[i] 与任意 A[j]/B[j] 不重叠。
- 不支持 batch 间广播、任意非连续 Tensor 和动态 shape 图模式；m/n/k 等均在运行时确定。

# 可维可测分析

## 精度标准与验证

测试位于 `test/gemm_batched/sgemm_batched/arch22/` 与 `test/gemm/hgemmBatched/arch22/`。CSV 驱动 GTest 覆盖 N/T/C 全组合、零维与负向参数、前导维 padding、batch 扫描、alpha/beta 特殊值、Inf/NaN、FP16 次正规数与接近上限值。

精度判定使用 ops-blas 混合容差标准：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FLOAT32 | \(2^{-13}\) | \(2^{-13}\) | 0.99 | `1e-2` 或 32 ULP |
| FLOAT16 | \(2^{-10}\) | \(2^{-10}\) | 0.99 | `1e-1` 或 32 ULP |

逐元素满足：

\[
|\text{actual}-\text{golden}|
\le \text{atol}+\text{rtol}\,|\text{golden}|.
\]

S golden 使用 Netlib `cblas_sgemm` 逐 batch 计算。H 将 FP16 输入展开为 FP32，使用 FP32/float64 累加后先收窄到 FP32、再按 IEEE round-to-nearest-even 收窄到 FP16。负向用例仅校验返回码；正向用例校验完整逻辑区域及 `ldc` padding 保持。

正式自验数量按任务书 §3.5：

- 精度全量 1000 条：S 500 条、H 500 条。
- 性能子集 200 条：S 100 条、H 100 条。

## 性能标准与验证

性能设备为 Atlas 800T A2（Ascend 910B3）。正式判据使用 `msprof op` 采集单次 kernel Task Duration：

- 预热 5 次，不计样本。
- 连续采集 10 次有效样本，取算术平均。
- 单 case NPU kernel 平均耗时不超过 `gpu_ms / 0.8`。
- 全部可比 case 的平均倍率不低于 0.8。

GTest `[PERF] avg_us` 按任务书采集连续 10 次事件计时，仅作交叉验证，不作为正式性能判据。精度测试与性能测试分开执行、分开判定。

性能报告同时保留逐 case 参数、GPU baseline、NPU kernel 时间、倍率、warmup/样本数量、msprof 原始输出和采集命令。

## 兼容性分析

- 公共声明已存在，保持签名、ABI、状态码和调用顺序兼容。
- 新增实现只落在 `blas/gemm_batched/arch22/`，不创建产品线私有平行 API。
- `arch35` 的 S/H 行为保持原状；`arch22` 的 `aclblasCgemmBatched` 保持现有实现。
- 更新 `blas/gemm_batched/README.md`，将 Atlas A2/A3 的 S/H 支持状态改为支持，并补充列主序、前导维、quick return 和 batch 语义。
- 测试代码按 `arch22` 独立注册，构建入口为：

```bash
bash build.sh --ops=sgemm_batched,hgemm_batched --soc=ascend910b3
```

本设计 PR 只新增本文件；代码、测试和自验材料在设计评审通过后提交。
