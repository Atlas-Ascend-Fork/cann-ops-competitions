# aclblasCsyrk3mEx 算子设计文档

## 一、需求背景

### 1.1 需求来源

11 月 CANN 社区任务要求在 Ascend 950PR（arch35）上，为 ops-blas 增加句柄式接口 aclblasCsyrk3mEx。算子计算复数对称矩阵秩-k 更新，并支持 complex64 与交错复数 int8 输入。

### 1.2 算子实现参考

#### 1.2.1 TBE 参考与实现依据

本任务对标的是 BLAS 句柄接口 cublasCsyrk3mEx。CANN 9.2.1 的 OPP 中没有该接口对应的 TBE kernel、算子信息条目或专用原型，因此没有可准确引用的同名 TBE 源码。TBE 基线流程图按任务书给出的 BLAS 数学语义和 Netlib CBLAS 参考流程绘制，未把 cuBLAS 的内部实现描述成已知 TBE 逻辑。

实现参考来自仓内 CSYRK 参数校验和三角更新、CGEMM3M 的复数三乘分解，以及任务书指定的 Netlib CSYRK 参考。

| 参考项 | 用途 |
| --- | --- |
| Netlib BLAS 3.10.0 的 CSYRK 与 CBLAS CSYRK | 数学语义和精度 golden |
| ops-blas/blas/syrk/arch35/csyrk_host.cpp、csyrk_kernel.cpp | BLAS 句柄校验、三角更新与退化路径 |
| ops-blas/blas/gemm/arch35/cgemm_host.cpp、cgemm3m_rounding_kernel.cpp | complex 3M 分解、AIV/Cube 数据流与长 K 精度处理参考 |

[Netlib CSYRK](https://www.netlib.org/blas/csyrk.f) · [ops-blas CSYRK](https://gitcode.com/cann/ops-blas/tree/master/blas/syrk/arch35) · [ops-blas CGEMM3M](https://gitcode.com/cann/ops-blas/tree/master/blas/gemm/arch35)

#### 1.2.2 算子现状分析

##### 1.2.2.1 数据类型和数据格式

| 张量 | 支持类型 | 内存格式与布局 |
| --- | --- | --- |
| A | ACL_COMPLEX64 | 每个复数由两个 FP32 组成，列主序交错存储 real、imag |
| A | ACL_INT8 | 每个复数由两个 int8 组成，列主序交错存储 real、imag |
| C | ACL_COMPLEX64 | 每个复数由两个 FP32 组成，列主序交错存储 real、imag |
| alpha、beta | aclblasComplex | Host 或 Device 复数标量，real 与 imag 均为 FP32 |

A 的逻辑尺寸在 trans=N 时为 n×k，在 trans=T/C 时为 k×n。前导维度分别满足 lda≥max(1,n) 或 lda≥max(1,k)，且 ldc≥max(1,n)。C 为对称矩阵而非 Hermitian 矩阵，uplo 指定的三角才会被读写。

##### 1.2.2.2 参考实现流程

接口语义为 C = alpha·op(A)·op(A)^T + beta·C。OP_C 在本算子中与 OP_T 等价，不做共轭。Netlib CBLAS 的 CSYRK 逐列处理指定三角，生成复数乘加结果后再应用 alpha、beta。对角线虚部属于有效输出。

任务书的 GPU 性能基线来自 cuBLAS cublasCsyrk3mEx。精度参考使用 Netlib CBLAS，INT8 的实部和虚部分别精确提升。有限值使用 cblas_zsyrk 并将结果收窄到 FP32。可能出现 FP32 溢出或含非有限参与值时，使用 cblas_csyrk 保留对应的运算语义。

~~~mermaid
flowchart TD
    A["检查 BLAS 句柄、枚举、尺寸、dtype 和指针"] --> B{"n 是否为零"}
    B -->|是| C["成功返回，不读写 A、C"]
    B -->|否| D{"秩-k 项是否有效"}
    D -->|否| E{"beta 是否为单位复数"}
    E -->|是| F["不启动 kernel，C 保持不变"]
    E -->|否| G["对 uplo 指定三角计算 beta·C"]
    D -->|是| H["按列主序读取 op(A)"]
    H --> I["按标准复数乘加累积 op(A)·op(A)^T"]
    I --> J["应用 alpha 与 beta"]
    G --> K["只写回 uplo 指定三角"]
    J --> K
~~~

##### 1.2.2.3 参考实现流程图说明

当前 CANN OPP 未提供同名 TBE 实现。上图给出任务书的 CBLAS 数学参考流程，并用于与 Ascend C 实现逐项对照。Netlib 源码规定了对称语义、转置寻址、三角更新和 alpha/beta 行为。

### 1.3 功能定义

1. 计算 C = alpha·op(A)·op(A)^T + beta·C。
2. trans=N 时 op(A)=A，trans=T 或 trans=C 时 op(A)=A^T。trans=C 不做共轭。
3. 仅更新 uplo 指定的上三角或下三角，另一三角保持原始位型。
4. 对角线复数按普通复数计算，虚部不清零。
5. n=0 为合法 no-op。k=0 或 alpha=(0,0) 时不引用 A，只按 beta 处理 C。
6. 支持 (Atype,Ctype)=(ACL_COMPLEX64,ACL_COMPLEX64) 和 (ACL_INT8,ACL_COMPLEX64)。

## 二、需求分析

### 2.1 外部组件依赖

| 组件 | 用途 |
| --- | --- |
| CANN ACL Runtime | 句柄 stream、Device 内存访问和 kernel 下发 |
| ops-blas | BLAS 状态码、workspace、AIC/AIV 核数查询及 CGEMM3M Cube kernel |
| Netlib CBLAS | 仅用于测试 golden，不参与算子运行 |

### 2.2 内部适配模块

接口实现位于 blas/syrk/arch35。Host 使用 handle 内的 stream 与 workspace，复用 common/helper 中的指针位置判断、尺寸溢出检查和 workspace 管理。长矩阵路径复用 gemm/arch35 中的 complex deinterleave 与 triangular products kernel。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

~~~cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
~~~

#### 2.3.2 参数约束

| 参数 | 合法值或约束 | 错误处理 |
| --- | --- | --- |
| handle | 有效 ops-blas 句柄 | 空指针返回 HANDLE_IS_NULLPTR |
| uplo | ACLBLAS_UPPER 或 ACLBLAS_LOWER | 非法枚举返回 INVALID_ENUM |
| trans | ACLBLAS_OP_N、ACLBLAS_OP_T、ACLBLAS_OP_C | 非法枚举返回 INVALID_ENUM |
| n、k | 非负整数 | 负值返回 INVALID_VALUE |
| lda | trans=N 时至少 max(1,n)，否则至少 max(1,k) | 不满足时返回 INVALID_VALUE |
| ldc | 至少 max(1,n) | 不满足时返回 INVALID_VALUE |
| Atype、Ctype | Atype 为 ACL_COMPLEX64 或 ACL_INT8，Ctype 固定 ACL_COMPLEX64 | 不支持组合返回 NOT_SUPPORTED |
| alpha、beta | 非空复数标量，二者同处 Host 或同处 Device | 空指针或位置不一致返回 INVALID_VALUE |
| A、C | 秩-k 项有效时 A 非空；n>0 时 C 非空 | 不合法指针返回 INVALID_VALUE |

本接口不支持 A 与 C 的内存重叠。A 的逻辑维度和实际存储由 trans、lda 描述。校验完成后才执行 n=0、k=0 和 alpha=0 快速路径。

## 三、需求详细设计

### 3.1 计算模型

令 X=op(A)=Xr+iXi。三次实数矩阵乘为：

$$
P_0=X_rX_r^T,\
P_1=X_iX_i^T,\
P_2=(X_r+X_i)(X_r+X_i)^T.
$$

乘积实部为 P0−P1，虚部为 P2−P0−P1。Host 将这三项提交给一次 CGEMM3M triangular-products kernel 下发。AIV 合并 K 分片，应用复数 alpha 与 beta，并写回所选三角。

### 3.2 Host 分核、分块与工作区

#### 3.2.1 校验和快速路径

Host 先校验 handle、uplo、trans、n/k、lda/ldc、dtype 与指针。alpha 和 beta 可以同时位于 Host 或 Device。Device 标量通过绑定 stream 异步拷贝回 Host 后进行路径判定。

n=0 不下发 kernel。若 k=0 或 alpha=(0,0)，且 beta=(1,0)，直接返回。其他退化情形只启动 AIV 三角缩放，不读取 A。beta=(0,0) 时 kernel 不读取旧 C。

#### 3.2.2 分核和 tiling

n≤16 且 k≤128 时采用单个 AIV kernel。有效乘积在 k≥8 时由四个 warp lane 协作归约 K，较短的 K 由单 lane 计算。其他有效乘积由 Cube 计算。对称秩-k 更新的两侧读取同一个 A，因此可以共用实、虚、和三个 FP32 面板。

- Cube tile 尺寸 B 在 n≤64 时为 32，65≤n≤256 时为 64，n>256 时为 128。
- tileCount=ceil(n/B)，三角 tile 数 T=tileCount·(tileCount+1)/2。
- precisionParts=max(1,ceil(k/512))。
- parallelParts=max(1,ceil(aicCores/(3·T))).
- 若 T<aicCores，或 1024≤k≤4096，则 kPartitions=precisionParts，以 K 分片扩展并行度并缩短单次累加长度；其他情况 kPartitions=parallelParts。
- 每个 K 分片按 16 对齐，kSpan=ceil(ceil(k/16)/kPartitions)·16。
- Cube block 数为 min(aicCores,3·T·kPartitions)。AIV block 数不超过可用 AIV 核数。

#### 3.2.3 workspace 与内存布局

输入面板行数 outRows=align16(inputRows)，其中 inputRows 在 trans=N 时为 n，否则为 k。inputCols 在 trans=N 时为 k，否则为 n。tempLdc=align16(n)。

单个输入面板字节数为 align32(outRows·inputCols·4)。三个面板存储 Ar、Ai、Ar+Ai，两侧 Cube 操作数共享这些面板。每个乘积平面字节数为 kPartitions·tempLdc·align16(n)·4。

总 workspace 字节数为：

$$
W=3\cdot align32(outRows\cdot inputCols\cdot 4)
 +3\cdot kPartitions\cdot tempLdc\cdot align16(n)\cdot 4.
$$

Host 对乘法、加法和字节对齐做 64 位溢出检查。workspace 起始地址按 32 字节对齐，并校验不超过库的最大 workspace 限制。

#### 3.2.4 tilingKey 规划

本接口不通过 TBE 算子 tilingKey 分派。Host 直接组装 `Csyrk3mExEpilogueTilingData` 与 `GemmTilingData`，并按运行时尺寸、dtype 和标量选择小矩阵 AIV、退化 AIV 或 Cube 三乘积路径。Kernel 的编译架构由 ops-blas 的 arch35 构建目标决定。K 分片数由三角 tile 数、AIC 核数与 K 长度计算，不依赖测试用例编号或输入随机种子。

### 3.3 Kernel 数据流

#### 3.3.1 Kernel 实现

1. 小矩阵路径在 AIV 上处理 uplo 指定的输出元素，直接做三个实数内积并合并。
2. ACL_COMPLEX64 的 Cube 路径调用 CGEMM3M deinterleave，将实、虚、和分量写入三个面板。
3. ACL_INT8 的 Cube 路径在 pack kernel 中逐分量提升为 FP32，并生成相同布局的三个面板。
4. Cube kernel 按三角 tile 和 K 分片一次完成 P0、P1、P2 三个实数乘积。
5. AIV combine kernel 累加各 K 分片，按 3M 公式重构复数乘积，应用 alpha、beta，并只写回 uplo 指定三角。
6. 当 k>128 或 |Re(α)|+|Im(α)|>1 时，合并 kernel 直接归约对角实部 r²−i² 与虚部 2ri，缓解 3M 相减抵消及缩放放大的舍入误差。COMPLEX64 用 warp shuffle 合并 K 子序列，N 布局优先合并连续行访问，T/C 布局优先合并连续 K 访问。INT8 对角线累加使用 64 位整数，转回 FP32 后应用 alpha、beta。重构出非有限值时，按复数乘加顺序重算对应元素。

~~~mermaid
flowchart TD
    A["Host 参数校验、标量解析和 workspace 绑定"] --> B{"秩-k 项有效"}
    B -->|否| C["AIV 对 uplo 三角计算 beta·C"]
    B -->|是| D{"n≤16 且 k≤128"}
    D -->|是| E["AIV 小矩阵路径"]
    D -->|否| F["AIV deinterleave 或 int8 pack"]
    F --> G["一次 Cube kernel 计算 P0、P1、P2"]
    G --> H["AIV 归并 K 分片"]
    E --> I["重构复数乘积并处理精度补偿"]
    H --> I
    I --> J["应用 alpha、beta"]
    J --> K["仅写回 uplo 指定三角"]
~~~

#### 3.3.2 三乘流程与参考流程的差异

| 处理项 | CBLAS 参考 | Ascend C |
| --- | --- | --- |
| 复数乘法 | 对每个复数乘积直接执行复数乘加 | 拆成 P0、P1、P2 三个实数矩阵乘 |
| 并行维度 | CPU 按列或行迭代 | Cube 按目标三角 tile 和 K 分片并行 |
| 数据准备 | 直接读取列主序复数矩阵 | AIV 将交错复数拆成三个共享的 FP32 面板 |
| 数值误差 | CBLAS 复数乘加 | 通过 K 分片缩短 Cube 累加，长 K 对角线在 AIV 上并行归约 |
| 写回 | CBLAS 仅更新所选三角 | AIV 只更新同一三角并保留另一三角与 padding |

~~~mermaid
flowchart LR
    A["CBLAS CSYRK 输入 A"] --> B["直接复数乘加"]
    B --> C["alpha·乘积 + beta·C"]
    C --> D["写 uplo 三角"]

    E["Ascend C 输入 A"] --> F["AIV 生成实、虚、和面板"]
    F --> G["Cube 三个 real GEMM"]
    G --> H["AIV 合并分片和精度补偿"]
    H --> I["alpha、beta 后写 uplo 三角"]
~~~

### 3.4 支持硬件与限制

目标硬件为 Ascend 950PR，arch35。最低软件要求为 CANN 9.1.0。本文验证环境为 DevEnv_622833，设备 Ascend950PR，CANN 9.2.1。

A 仅支持 ACL_COMPLEX64 与 ACL_INT8，C 固定 ACL_COMPLEX64。OP_C 按普通转置解释。算子不保证对称输入，也不执行元素扫描。仅 uplo 指定三角有输出定义，调用方不得依赖另一三角被更新。

## 四、特性交叉分析

| 维度 | 覆盖范围 |
| --- | --- |
| uplo | UPPER、LOWER |
| trans | N、T、C |
| Atype | ACL_COMPLEX64、ACL_INT8 |
| 尺寸 | 空矩阵、极小矩阵、方阵、长 K、非方形 n×k |
| 前导维 | 紧凑布局、带 padding 的 lda/ldc |
| scalar | 复数 alpha/beta、零乘积、beta=0、beta=1 |
| 数值 | uniform、normal、int8 随机和极值、对角虚部非零、3M 相消 |
| 输出完整性 | 逐实虚分量比较，并逐位检查另一三角和 ldc padding |

任务包 CSV 共 1200 条，其中 1000 条精度用例和 200 条性能用例。性能用例均为 ACL_COMPLEX64，覆盖上/下三角、N/T/C 和多组方阵与非方形尺寸。

## 五、可维可测

### 5.1 精度标准与验证方法

有限值的 golden 采用任务书允许的 Netlib cblas_zsyrk，结果收窄至 FP32。若参与值非有限，或输入和标量幅值推导的 FP32 累加上界达到最大有限值，则采用 cblas_csyrk。ACL_INT8 输入先将实、虚分量精确提升。实部和虚部分别满足：

$$
|actual-golden|\le 2^{-13}+2^{-13}|golden|.
$$

每个 case 的 matched_ratio 至少为 0.99，最大绝对误差不超过 max(1e-2,32·ULP)，ULP 取自最大绝对误差点的 FP32 golden 值。另一三角及 ldc padding 按位保持不变。任务要求的 1000 条精度用例结果将在最终提交中记录。

### 5.2 性能标准与验证方法

950PR 上按 kernel-to-kernel 口径验证。每条性能用例先预热 5 次，再用 msprof 采集 10 个有效的 kernel Task Duration 样本并取算术平均。一次调用若包含多个 kernel，合计各阶段耗时。逐条比值为 GPU kernel-only gpu_ms 除以 NPU kernel 耗时。单条 NPU 耗时须不大于 gpu_ms/0.4，200 条比值的算术平均须不小于 0.5。另采集 INT8 性能，并使用相同形状的 COMPLEX64 GPU 基线作补充对照。

GTest ACL event 输出仅作为端到端趋势检查，不替代 msprof 的正式 kernel 耗时。

### 5.3 实测结果

最终提交前填入相同代码提交下的构建、1000 条精度集、200 条性能集、三次稳定性采样与逐 case 性能统计。报告注明设备、CANN 版本、提交 SHA 和测试时间。

