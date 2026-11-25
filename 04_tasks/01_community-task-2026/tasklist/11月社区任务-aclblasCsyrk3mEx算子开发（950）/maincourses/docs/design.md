# 需求背景

## 需求来源

11 月 CANN 社区任务要求在昇腾 950PR 的 `ops-blas` 工程中新增 `aclblasCsyrk3mEx`，用 Ascend C 实现扩展复数对称秩-k 更新。接口功能对标 NVIDIA cuBLAS `cublasCsyrk3mEx`，CPU 数学参考采用 Netlib BLAS 3.10.0 的 `cblas_csyrk`。[cuBLAS 接口](https://docs.nvidia.com/cuda/archive/13.2.1/cublas/index.html#cublascsyrk3mex) · [Netlib CSYRK](https://www.netlib.org/blas/csyrk.f)

## 背景介绍

源码调研以 `cann/ops-blas` 提交 `55ef791b77f3677b45da3468b0a8b17941424f84` 为基线。arch35 的 `aclblasCsyrk` 已提供三角分块、紧凑与带 padding 的前导维支持，以及 AIV 直算和 AIC 分块计算范例。它的 Cube 路径生成四份实数乘积，本任务复用其中可独立使用的三角调度和搬运方式。`aclblasCgemm3m` 提供共享 GEMM 工程接入和舍入风险筛选参考。新增接口采用自己的 Host、分派和 kernel 路径，并保持既有接口的 ABI。[非 Ex SYRK](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/syrk/arch35) · [3M GEMM 调研基线](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/arch35)

# 需求分析

## 需求描述

在 handle 绑定的 stream 上异步执行 `C ← αXXᵀ + βC`，其中 `X = op(A)`。A、C 均为列主序，C 原地更新 `uplo` 指定的上三角或下三角，包含有效的对角虚部。A 接受 `ACL_COMPLEX64` 和 `ACL_INT8` 两种存储，C 固定为 `ACL_COMPLEX64`。

## 需求拆解

| 要求 | 任务书依据 | 实现及验证 |
| --- | --- | --- |
| 公共签名、dtype 和状态码 | §2.2–2.5 | 新增公开声明及 arch35 实现，验证非法 handle、枚举、尺寸、前导维、类型及必要指针 |
| 对称语义及原地更新 | §2.1、§2.4 | 覆盖两种三角、N/T/C、带 padding 列主序、对角虚部和写边界哨兵 |
| 三次实乘融合 | §2.1、§2.2 | 三个实乘在一次计算 kernel launch 内完成，按形状验证实际下发序列 |
| 退化及访问屏蔽 | §2.1、§2.5 | 覆盖空矩阵、空秩、零 alpha、零 beta、A/C 空指针及 NaN 防读用例 |
| 精度和性能 | §3.1–3.5 | 950PR 真机与 Netlib golden，分别记录逐用例精度及 msprof kernel 计时 |
| 内存 | §3.4 | 无独立验收阈值，验证 workspace 边界及复用正确性 |

# 详细设计

## 算子分析

设 `X = R + iI`、`S = R + I`，每个三角块计算

- `P0 = RRᵀ`，`P1 = IIᵀ`，`P2 = SSᵀ`。
- `Re(XXᵀ) = P0 − P1`，`Im(XXᵀ) = (P2 − P0) − P1`。
- 将实虚部按复数算术乘以 `alpha`，再与 `beta · C` 相加。

| 参数 | 存储及约束 |
| --- | --- |
| `uplo` | `ACLBLAS_UPPER` 访问 `i≤j`，`ACLBLAS_LOWER` 访问 `i≥j` |
| `trans` | N 时 `X(i,t)=A[i+t·lda]`，T/C 时 `X(i,t)=A[t+i·lda]`。T/C 均使用普通转置 |
| `n, k` | 非负整数，运行时分块。覆盖紧凑、小尺寸、非对齐及宽窄矩阵 |
| `lda, ldc` | N 时 `lda≥max(1,n)`，T/C 时 `lda≥max(1,k)`，`ldc≥max(1,n)`。前导维以复元素为单位 |
| `A, Atype` | `ACL_COMPLEX64` 每元素 8 字节。`ACL_INT8` 每元素 2 字节，布局为有符号 `(real, imag)`，分别精确提升到 FP32 |
| `C, Ctype` | `ACL_COMPLEX64`，元素地址为 `C[i+j·ldc]`。参与更新的 C 与被引用的 A 不重叠 |
| `alpha, beta` | 复数 FP32 标量，支持 Host 或 Device 指针 |

## 算子实现

### Host 侧设计

在 `include/cann_ops_blas.h` 添加声明，使用与现有 BLAS 接口相同的 C 链接方式。

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

1. 依任务书顺序校验 handle、uplo/trans、n/k、lda/ldc、标量指针、类型和结构依赖的 A/C 指针。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法尺寸、前导维或必要指针返回 `ACLBLAS_STATUS_INVALID_VALUE`，其余 dtype 组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。尺寸乘积、元素地址和 workspace 偏移用 64 位算术，Host 检查计算尺寸的溢出后再下发。
2. `n=0` 时完成结构校验后直接返回成功，A/C 可为空。`n>0` 时 C 必须有效。`k=0` 或 `alpha=0` 时 A 可为空，`beta=1` 时直接返回或由 Device 标量路径在 kernel 内退出，其他 beta 下执行三角缩放。`beta=0` 时直接构造结果，旧 C 可含 NaN。
3. Host 标量在下发时复制进 tiling。Device 标量随绑定 stream 交给 kernel 读取，以遵循先前生产标量的操作顺序。`A=nullptr`、`n,k>0` 时为返回准确的同步参数状态，Host 需要先知道 Device alpha 是否为零，此分支只读回一个复数并按同一 stream 顺序等待。Host 只检查地址及元数据，张量数据在 Device 上处理。
4. Tiling 携带尺寸、前导维、dtype、枚举、标量来源、活跃核数、分块和 workspace 偏移。根据 `n/k`、布局、核数和片上容量选择缩放、AIV 直算或 Cube 分块路径。核心实现放在 `blas/syrk/arch35/`，GTest 通过仓内 CMake 注册于 `test/syrk/csyrk3m_ex/arch35/`，并在 `blas/syrk/README.md` 记录公开 API 与支持型号。

### Kernel 侧设计

**缩放和小问题。** 缩放仅触发 AIV 三角读写。小规模或窄 K 采用 AIV SIMT，按列三角区间展开，使相邻线程写入相邻 C 元素，每个输出由一个线程或线程组独占。一个 kernel 内累积三个实数内积，使用分段或补偿归约减小长 K 的抵消误差。划分三角区间时排除对角外另一侧和列尾 padding。

**大问题。** 使用 arch35 Tensor API 将整个三角计算分配为输出 tile。初始调优分块为 `64×64`，只调度与目标三角相交的 tile，动态限制活跃 AIC 数。每个输出 tile 由一个 AIC 负责完整 K 累积，AIV 对其输出执行复数 epilogue。小 n 且 K 很长时，优先重新评估 AIV 路径或缩小 C tile 提高并行度。实际分派阈值根据真机精度和耗时确定。

1. 从 A 的物理列进行合并搬运。N 按 `lda` 取每个 K 面板的行切片，T/C 按 `lda` 取每列连续的 K 切片。AIV 解交错并生成 R、I、S 三平面，可先实现一次面板预处理，再评估把双缓冲搬运并入 MIX kernel。临时平面对齐并在合法末尾内填零。
2. AIC 为当前 tile 初始化三个 FP32 L0C 累加器。每个 K 分块从 L1 取相应输入到 L0A/L0B，在同一次计算 launch 内依次执行 R、I、S 三个实乘，并在 K 循环内更新相应累加器。`64×64` 的三个 FP32 累加器占 48 KiB，双组累加占 96 KiB，尾块裁剪按当前 Tensor API 的对齐要求处理。
3. 大问题优先采用 MIX 输出流水。AIC 将三块结果发送到所属 AIV 的缓冲，发布就绪标记后由 AIV 取数，消费完毕回传可复用标记。用 AIC/AIV 成对的交接和两槽轮转重叠数据搬运。实现阶段根据 CANN 实际可用的 Fixpipe 目的位置选择片上缓冲或每核 GM 暂存。主计算采用同一 tile 所有权，三次实数矩阵乘始终在同一 launch 中。
4. AIV 计算 `P0−P1`、`(P2−P0)−P1` 及复数 alpha/beta，按列主序写入 C。非对角 tile 按有效矩形写回，对角 tile 按列裁剪有效行范围。尾块与不满足向量写对齐的窄区间使用不跨越边界的写回路径。`beta=0` 省略旧 C 加载。

**精度控制。** 默认以 FP32 路径为正确性基线。对 INT8 输入，保留 `R+I` 的完整有符号范围 `[−256,254]`。当 K 较长时，三路分块累加并关注超过 FP32 整数精确域的中间和。COMPLEX64 重点验证 `P0−P1`、`P2−P0−P1` 的近零抵消、极大 alpha 和长 K。对高风险项，评估对 `R+I` 舍入残差和分块归约误差的补偿，或在同次计算中按数值条件补算局部结果。NaN/Inf 专项按参考的对应分量传播行为验证。任何精度降档或稀疏修正均须通过边界及全量回归，分派依据为形状、布局、dtype 和数值条件。

**Workspace 与并发。** 以现有 handle 的 workspace 管理接口分配并划分临时区域，尊重用户设置的容量和所有权。预处理三平面上界为 `3 × 对齐后物理行数 × 物理列数 × sizeof(float)`。采用每核 GM 暂存时，双槽三面板上界为 `2 × 活跃 AIC 数 × 64 × 64 × 3 × sizeof(float)`，另计对齐与同步控制区。直算及缩放不需要整矩阵 scratch。复用以同一 stream 顺序为依据，扩容或切换 stream 时遵守 handle 的在途资源保护规则。分配或计算提交失败沿用仓内状态码，Host 校验失败不发起 kernel。

## 支持硬件

| 硬件 | 编译目标 | 软件 |
| --- | --- | --- |
| 昇腾 950PR | `ascend950` / arch35 | 任务书要求 CANN ≥ 9.1.0，计划在 CANN 9.2.1 验证 |

## 算子约束限制

- A 与 C 按列主序和前导维访问，不提供广播、批处理或任意步长张量接口。
- 仅目标三角参与计算与写回。另一半三角和前导维 padding 保留调用前的内容。输出中的对角虚部与其他复数元素使用相同算术。
- 数值验收以任务书容差为准。调用方在 Host 消费输出之前同步 handle 所绑定的 stream。

# 可维可测分析

## 精度标准与性能标准

| 项目 | 方法与通过条件 |
| --- | --- |
| 环境 | 使用 950PR 和指定 CANN，记录编译器、BLAS、设备、绑定 stream、构建类型和每轮固定代码版本 |
| 精度 | 运行 1000 条精度 CSV，用 Netlib BLAS 3.10.0 `cblas_csyrk` 生成 golden。OP_C 映射为 CBLAS 普通转置，INT8 先精确提升。目标三角的实部、虚部分别按 `atol=rtol=2⁻¹³`、匹配率 ≥ 99% 及最大误差限制 `1e−2 或 32×ULP` 判断 |
| 正负向与写保护 | 覆盖 4 条 `TC_3M`、两种 dtype、N/T/C、上下三角、紧凑和带 padding、零尺寸/零 alpha/零 beta、对角虚部、空 handle 和 Host/Device 标量。另半三角与 padding 用逐元素哨兵校验 |
| 正式性能 | 逐条运行 200 条 `TC_PF`。以 `msprof op` 提取 `OpBasicInfo.csv` 中的计算 kernel `Task Duration(us)`，剔除前 5 次预热，统计随后 10 次有效样本均值。分别记录辅助阶段和调用端到端耗时。以 `gpu_ms / (NPU_us / 1000)` 计算倍率，逐条 ≥ 0.4，整体算术平均 ≥ 0.5 |
| 补充性能 | 用同参数 COMPLEX64 的 GPU 基线对照 INT8 路径，单列补充结果。GTest `[PERF] avg_us` 及原验证脚本用于快速回归 |
| 内存和生命周期 | 任务书无独立内存阈值。补测用户 workspace 不足、扩容、同 stream 连续调用及原地写保护 |

建议先跑基础及退化用例，随后检查 `TC_3M`、大 K、带 padding 和两类输入。待精度稳定后再跑完整 200 条性能用例。示例构建入口为 `bash build.sh --ops=csyrk3m_ex --soc=ascend950`，全量自验使用任务附件脚本和归档的原始 GPU 基线。

**附件核对。** 现有 200 条性能用例及基线均为 COMPLEX64，INT8 结果按任务书补采。GPU 基线中 512×512 和 1024×1024 各有两条同参且耗时不同的记录，正式统计保留用例与基线 ID 对应关系，避免按参数字典覆盖。精度 CSV 包含 MERE/MARE 配置，报告同时记录原脚本判定和任务书混合容差判定。当前附件提供生成脚本及 CSV，交付所需的冻结输入和 golden 另按任务书约定产生或核对。

## 兼容性分析

公开头文件新增一个声明。arch35 下新增对应实现和测试，既有 `aclblasCsyrk`、其他 arch35 BLAS 实现以及原有产品线接口保持现有行为。该设计的分核和性能阈值需在真机上验证。
