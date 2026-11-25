# aclblasSgemvStridedBatched 类 5 算子设计文档

> **任务编号**：11 月社区任务-gemvStridedBatched 类 5 算子开发（A2/A3）<br>
> **团队名称**：`<待填写>`<br>
> **适配硬件**：Atlas A2 系列（性能设备 Atlas 800T A2 / 910B3）、Atlas A3 系列，统一走 `arch22` 路径<br>
> **CANN 版本**：CANN 9.1.0<br>
> **开发语言**：Ascend C（handle 直调 AIV Vector kernel）<br>
> **目标仓库**：<https://gitcode.com/cann/ops-blas><br>
> **实现目录**：`blas/gemv_strided_batched/arch22/`；测试目录：`test/gemv_strided_batched/arch22/`

# 一、需求背景（required）

## 1.1 需求来源

本设计依据社区任务书 `aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md` 编写，要求在昇腾 Atlas A2/A3（`arch22`）上使用 Ascend C、基于 ops-blas 开源仓的句柄式 BLAS 工程模式，实现跨步批量矩阵-向量乘算子族共 5 个 API：

- `aclblasSgemvStridedBatched`（S：FP32/FP32/FP32）
- `aclblasHSHgemvStridedBatched`（HSH：FP16/FP16/FP16）
- `aclblasHSSgemvStridedBatched`（HSS：FP16/FP16/FP32）
- `aclblasTSTgemvStridedBatched`（TST：BF16/BF16/BF16）
- `aclblasTSSgemvStridedBatched`（TSS：BF16/BF16/FP32）

对标 cuBLAS `cublasSgemvStridedBatched` 及 `cublasHSH/HSS/TST/TSSgemvStridedBatched`，数学语义参考 Netlib `sgemv`。5 个公开声明均已存在于 `include/cann_ops_blas.h`（约 L296-L337），本任务无需新增声明。

## 1.2 背景介绍

### 1.2.1 功能与数学语义

对 `i = 0, 1, ..., batchCount-1`：

```text
y + i*stridey = alpha * op(A + i*strideA) * (x + i*stridex) + beta * (y + i*stridey)
```

- 第 i 个 batch 的 A/x/y 基址分别为 `A + i*strideA`、`x + i*stridex`、`y + i*stridey`，偏移以元素为单位；
- 列主序：`A(i,j)` 位于 `A[i + j*lda]`；`trans=N` 时 `op(A)=A` 为 `m×n`，否则 `op(A)=A^T` 为 `n×m`；
- 实数算子中 `OP_C ≡ OP_T`（不共轭）；
- `trans=N` 时 x 长 `n`、y 长 `m`；否则 x 长 `m`、y 长 `n`；
- 5 个变体仅在 A/x/y 的存储 dtype 与输出 dtype 上不同，alpha/beta 恒为 FLOAT32 Host 标量；半精度/bf16 变体在 FP32 中完成乘加后再按输出 dtype 收窄。

> **物理布局（重要，按任务书 §9 修订记录）**：A **恒为 `lda×n`**，`lda ≥ max(1,m)`，**与 trans 无关**；trans 只作用于 `op(A)`，不改变 A 的物理存储形状。相邻 batch 的 A 最小间距为 `strideA ≥ lda×n`。这与 cuBLAS / Netlib `sgemv` 及仓内 golden 的判定一致。

### 1.2.2 接口与参数边界

```cpp
// S（其余变体参数序列相同，仅指针 dtype 不同）
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

变体 dtype：

| 变体 | A | x | y | alpha / beta |
| --- | --- | --- | --- | --- |
| S | FP32 | FP32 | FP32 | FP32 |
| HSH | FP16 | FP16 | FP16 | FP32 |
| HSS | FP16 | FP16 | FP32 | FP32 |
| TST | BF16 | BF16 | BF16 | FP32 |
| TSS | BF16 | BF16 | FP32 | FP32 |

> HSH/HSS/TST/TSS 的 A/x/y 以 `uint16_t*` 承载 FP16/BF16 位模式，kernel 按变体解释；alpha/beta 始终为 `const float*`。

参数约束与错误行为：

| 参数 | 约束与异常行为 |
| --- | --- |
| `handle` | 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | 仅 N/T/C；否则 `ACLBLAS_STATUS_INVALID_ENUM`（与 cuBLAS 的 `INVALID_VALUE` 为有意差异） |
| `m/n` | `≥0`；负值 `INVALID_VALUE`；`m==0` 或 `n==0` 为合法 quick return |
| `alpha/beta` | 非空；否则 `INVALID_VALUE`；取值不限制（含 0/Inf/NaN） |
| `lda` | `≥max(1,m)`，与 trans 无关；否则 `INVALID_VALUE` |
| `incx/incy` | `≠0` 且 `≠INT_MIN`，可正可负；否则 `INVALID_VALUE` |
| `strideA/stridex/stridey` | `>0`，且相邻 batch 区域不重叠；否则 `INVALID_VALUE` |
| `batchCount` | `≥0`；负值 `INVALID_VALUE`；0 为合法 no-op |
| `A` | `batchCount>0 && m>0 && n>0` 时非空；否则 `INVALID_VALUE` |
| `x/y` | `batchCount>0` 且逻辑长度 >0 时非空；否则 `INVALID_VALUE` |

边界语义：

- `batchCount==0`、`m==0`、`n==0`：校验结构参数后直接返回 `SUCCESS`，不引用 A/x/y、不启动 kernel；
- `alpha==0`：只做 `y=beta*y`，不读 A/x；按 §2.4 指针仍须非空（与仓内 golden 校验一致）；
- `beta==0`：不读 y 旧值，直接写 `alpha*sum`；
- y 为原地覆写，只写各 batch 的逻辑 `O` 个元素，按 `incy` 精确落位，不触碰空洞；
- 接口异步执行，通过 handle 绑定 stream，读取结果前调用方须同步 stream。

### 1.2.3 现有实现与本次范围

仓内 `blas/gemv_strided_batched/arch35/` 已实现 5 个变体，可作为 stride 偏移、变体分派、N/T 数据流与缩放逻辑的语义参考；但 arch35 使用 Ascend 950PR 的 SIMT（`__simt_vf__`）编程模型，**不能直接用于 A2/A3**。本任务在 `arch22` 新建 AIV Vector 实现。当前 `arch22` 目录不存在，需按仓内规范新建 `gemv_strided_batched_host.cpp`、`gemv_strided_batched_kernel.cpp`、`gemv_strided_batched_kernel.h`、`gemv_strided_batched_tiling_data.h`。

测试侧 `test/gemv_strided_batched/` 已有通用 GTest、参数结构、golden 与 CSV（arch35 版），本任务复用其参数/golden，新增 arch22 测试目标并在 `test/gemv_strided_batched/CMakeLists.txt` 注册。

本次范围：

1. 5 个 Host 入口与 arch22 AIV kernel；
2. N 模式（列累加）与 T/C 模式（按输出点积）两条数据流；
3. FP32 计算 + FP16/BF16 收窄、负步长、非单位步长、stride 偏移、padding；
4. quick return、`alpha=0` 缩放路径、`beta=0` 不读旧 y；
5. 完整参数校验（含 stride 重叠与跨度溢出）；
6. 测试工程与 README 支持说明。

## 1.3 测试资料与覆盖

测试由 `test_cases/` 下 CSV 驱动，5 个变体各 1000 条精度（合计 5000）、各 200 条性能（合计 1000）。覆盖 m/n 尺寸池（含 0/1/2/…/4095/4096）、N/T/C、alpha/beta 特殊值与随机值、lda padding（+1/+8/+64）、incx/incy ∈ {±1,±2,±3,7,8,16,31,64,127}、batchCount ∈ {0,1,2,3,8,16,64,256}、stride 紧凑/带间隙/非法/重叠、均匀/高斯/Inf/NaN、`beta=0` 配 NaN 旧值、`alpha=0` 配 NaN x、batchCount=0、m=0/n=0 等。golden 由 Netlib `cblas_sgemv` 逐 batch 生成，实部即全体（实数），按输出 y dtype 比对。

# 二、需求分析（required）

## 2.1 需求描述

在 A2/A3 `arch22` 上使用 Ascend C 实现 5 个跨步批量 GEMV 接口，支持 `trans=N/T/C`、列主序、`lda≥max(1,m)`、正负非零 `incx/incy`、显式 batch stride、运行时尺寸与标量，原地更新 y，句柄 stream 异步执行，且：

1. 数学与 cuBLAS/Netlib 一致；
2. FP16/BF16 变体 FP32 计算后正确收窄；
3. 完整参数校验与错误码；
4. quick return / 缩放 / `beta=0` 语义正确；
5. 精度、性能满足任务书 §3.2/§3.3。

## 2.2 需求拆解与覆盖矩阵

| 编号 | 要求（来源） | 设计落点 | 计划验证 |
| --- | --- | --- | --- |
| R01 | 5 个公共接口、变体 dtype（§2.1/§2.3） | Host 模板 + kernel dtype 分派 | 5 变体全量精度 |
| R02 | 列主序、A 恒 lda×n、lda≥max(1,m)（§2.1/§2.4/§9） | 地址映射、Host 校验 | LD/矩形/转置用例 |
| R03 | N/T/C，实数 C≡T（§2.1） | N 列累加、T/C 点积两条数据流 | 9 组合对照 golden |
| R04 | 正负/非单位 incx/incy（§2.4/§2.5） | 有符号步长映射、gather/scatter | INC 用例 |
| R05 | strideA/stridex/stridey>0 且不重叠（§2.4/§2.5） | Host 最小间距与跨度检查 | 非法/重叠负向 |
| R06 | batch/m/n 零值 quick return、alpha=0、beta=0（§2.1/§2.5） | Host 快返 + 缩放 kernel | ED/特殊值 |
| R07 | FP32 计算 + 收窄（§2.1） | Cast 上/下变换、RINT 收窄 | HSH/HSS/TST/TSS |
| R08 | 异步 stream、原地 y、无广播/无非连续 Tensor 语义外支持（§2.5） | launch 与所有权 | 生命周期/连续调用 |
| R09 | 精度/性能口径（§3.2/§3.3） | 精度与性能设计 | 5000 精度/1000 性能 |
| R10 | CANN9.1.0、A2/A3、arch22（§3.1） | 支持硬件 | 910B3 编译与实测 |

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式与地址映射

零基列主序：`A[i,j]=A[i+j*lda]`，`0≤i<m, 0≤j<n`。

定义有符号步长映射（基址取低地址端）：

```text
p(t, L, s) = t*s                    , s > 0
p(t, L, s) = (L-1-t)*(-s)           , s < 0
```

令 `K = (trans==N) ? n : m`（归约/点积长度），`O = (trans==N) ? m : n`（输出长度）。对第 b 个 batch：

```text
N  模式： y_b[p(r,O,incy)] = alpha * Σ_{j=0}^{K-1} A_b[r + j*lda]      * x_b[p(j,K,incx)] + beta * y_old_b[p(r,O,incy)]
T/C模式： y_b[p(r,O,incy)] = alpha * Σ_{i=0}^{K-1} A_b[i + r*lda]      * x_b[p(i,K,incx)] + beta * y_old_b[p(r,O,incy)]
```

其中 `A_b = A + b*strideA`、`x_b = x + b*stridex`、`y_b = y + b*stridey`。所有地址乘积提升到 `int64`/`uint64` 计算并做溢出检查，`lda*n`、`i*lda`、`b*stride` 不得在 `int32` 中相乘。

物理跨度与最小间距：

```text
span(L, s) = (L>0) ? 1 + (L-1)*|s| : 0
A 每 batch 占用 = lda*n            -> 要求 strideA >= lda*n
x 每 batch 占用 = span(K, incx)     -> 要求 stridex >= span(K, incx)
y 每 batch 占用 = span(O, incy)     -> 要求 stridey >= span(O, incy)
```

上述最小间距只在 `batchCount>1 && m>0 && n>0` 的实际计算路径上检查；`batchCount≤1` 时不存在相邻 batch，只要求 stride>0。这样既拦截真实重叠负向，又不会把合法 quick return 判为非法。

`beta==0`（含 -0）不加载旧 y；`alpha==0` 不读 A/x，仅 `y=beta*y`。不启用会消除 Inf/NaN 语义的 fast-math；不跳过零 x 对应项。

### 3.1.2 支持数据类型

| 变体 | A/x 计算类型 | 累加/标量 | 输出收窄 |
| --- | --- | --- | --- |
| S | FP32 | FP32 | —（写 FP32） |
| HSH | FP16 → FP32 | FP32 | FP32 → FP16 |
| HSS | FP16 → FP32 | FP32 | —（写 FP32） |
| TST | BF16 → FP32 | FP32 | FP32 → BF16 |
| TSS | BF16 → FP32 | FP32 | —（写 FP32） |

半精度/bf16 输入在 AIV 内 `Cast` 到 FP32 后计算；输出用舍入到最近偶数（RINT）收窄，与 CPU `static_cast` 参考一致。alpha/beta 始终为 FP32。

### 3.1.3 支持形状与布局

- A 物理 `lda×n`（`lda≥max(1,m)`），列主序；`trans` 只改变 `op(A)`；
- x/y 按 `incx/incy` 步长访问，支持负步长（基址低地址端，逻辑下标 0 对应物理末端）；
- batch 间由 stride 偏移，允许带间隙；
- 不支持广播、不支持超出 lda/inc/stride 语义的任意非连续布局；
- m/n/batchCount 任意合法非负 int32；通过 tile 循环处理，不把测试最大尺寸当接口上限。

## 3.2 算子实现

### 3.2.1 实现方案：整体架构与文件划分

调用链：

```text
aclblasCreate -> aclblasSetStream
             -> aclblasSgemvStridedBatched / HSH / HSS / TST / TSS (handle, ..., stream)
             -> 调用方同步 stream
```

全部计算在 Device AIV 完成，Host 只做校验、tiling 与 launch，无 Host 逐元素计算、无其他后端 fallback。

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| Host 入口 | `blas/gemv_strided_batched/arch22/gemv_strided_batched_host.cpp` | 5 个 `extern "C"` 入口、模板化校验/tiling/launch |
| Kernel | `.../arch22/gemv_strided_batched_kernel.cpp` | N 列累加、T/C 点积、缩放、可选 split-K 归约 |
| Kernel 头 | `.../arch22/gemv_strided_batched_kernel.h` | kernel `_do` 声明，Host/Device 共用 |
| Tiling 数据 | `.../arch22/gemv_strided_batched_tiling_data.h` | Host/Device 共享 POD |
| 测试 | `test/gemv_strided_batched/arch22/` | 复用通用 GTest/param/golden；新增 arch22 目标 |
| 文档 | `blas/gemv_strided_batched/README.md` | 补 A2/A3 支持说明 |

复用现有 `test/gemv_strided_batched/gemv_strided_batched_{param,golden}.h` 与 CSV 加载，避免重复实现 golden。

### 3.2.2 host侧设计

#### 1）参数校验顺序

严格按任务书 §2.5：

1. `handle` 空 → `HANDLE_IS_NULLPTR`；
2. `trans∉{N,T,C}` → `INVALID_ENUM`；
3. `m<0 || n<0` → `INVALID_VALUE`；
4. `lda<max(1,m)` → `INVALID_VALUE`；
5. `incx==0 || incy==0 || incx==INT_MIN || incy==INT_MIN` → `INVALID_VALUE`；
6. `strideA<=0 || stridex<=0 || stridey<=0` → `INVALID_VALUE`；
7. 最小间距/重叠检查（`batchCount>1 && m>0 && n>0` 时）：`strideA<lda*n || stridex<span(K,incx) || stridey<span(O,incy)` → `INVALID_VALUE`；零维/单 batch 只保留第 6 步的正值检查，避免把合法 no-op 误判为重叠；
8. `batchCount<0` → `INVALID_VALUE`；
9. `alpha==nullptr || beta==nullptr` → `INVALID_VALUE`；
10. 条件指针：`batchCount>0 && m>0 && n>0` 时 `A/x/y` 非空，否则 `INVALID_VALUE`；
11. quick return：`batchCount==0 || m==0 || n==0` → `SUCCESS`，不启动 kernel。

> 说明：第 11 步位于结构校验之后，因此即使 quick return，`lda/inc/stride/alpha/beta` 仍须合法；`A/x/y` 因第 10 步条件化，在零维时允许为空。`alpha==0` 仍要求 A/x 非空（与 §2.4 及仓内 golden 一致），只是运行时不读取。

#### 2）quick return 与缩放路径

- `batchCount/m/n` 任一为 0：直接成功，不发射 kernel；
- `alpha==0`：发射缩放 kernel，`beta==0` 写零、`beta==1` 直接返回、其余 `y=beta*y`，不读 A/x；
- `beta==0`：正常路径不加载旧 y。

#### 3）dtype 分派与变体入口

Host 用 `<T_IN, T_OUT>` 模板 + `dtype` 标签统一 5 个入口：

| dtype | 入口 | T_IN | T_OUT |
| --- | --- | --- | --- |
| S | SgemvStridedBatched | float | float |
| HSH | HSH | half | half |
| HSS | HSS | half | float |
| TST | TST | bfloat16_t | bfloat16_t |
| TSS | TSS | bfloat16_t | float |

接口层 `uint16_t*` 在 kernel 入口按 dtype 重解释为 `half`/`bfloat16_t`；不改变公共 ABI。

#### 4）tiling 与分核

tiling 结构（Host/Device 共用）包含：`dtype`、`transMode(0=N,1=T/C)`、`m/n/lda`、`alpha/beta`、`incx/incy`、`strideA/stridex/stridey`、`outSize(O)`、`dotSize(K)`、`outTile(R)`、`dotTile(Kt)`、`usedCoreNum`、`batchCount`、`taskCount`、`xStart/yStart` 标志、UB 子缓冲大小、`splitCount`、`workspacePitch`。字段用足够宽类型，stride 用 int64。

分核策略：任务空间 `(batch, outputTile)`，`outTilesPerBatch=ceil(O/R)`，`taskCount=batchCount*outTilesPerBatch`，`usedCoreNum=min(taskCount, AivCoreNum)`；各核按 `task=blockIdx+q*blockDim` 轮转，**每个输出元素唯一所有者，不使用原子写**。这样：

- batch 大时天然按 batch 并行；
- batch 小时按输出 tile 并行；
- 极端“输出短、归约长”时，可选 split-K（见下）避免核闲置。

`outTile R` 与 `dotTile Kt` 由 UB 预算二分确定，非固定测试常量。

#### 5）UB 预算

按 `GetCoreMemSize(UB)` 实测可用量（A2/A3 典型约 192 KiB）计算，预留双缓冲与临时区：

```text
N 模式 : A列块 R*Kt*sizeof(T_IN) + 累加 R*4 + x Kt*4 + 临时 R*4
T/C模式: A列块 R*Kt*sizeof(T_IN) + 累加 R*4 + x Kt*4 + 归约临时
```

以两份 A 缓冲（双缓冲）+ 累加 + 临时做预算，超过则缩小 `R/Kt`；容量不足时返回明确内部错误，不越界运行。

#### 6）溢出与地址安全

`lda*n`、`(batchCount-1)*stride`、`i*lda`、`(L-1)*|inc|` 等先提升到 64 位并检查；`stride` 跨度检查失败返回 `INVALID_VALUE`。Host 不扫描 Device 数据。

#### 7）split-K（可选，容量受限）

仅当 `batchCount*outTilesPerBatch < usedCoreNum` 且 `K` 较大、handle 现有 workspace 容量足够时启用：将 K 均分为 S 段半开区间 `[floor(s*K/S), floor((s+1)*K/S))`，各核写独占 workspace 部分和，再由 reduce kernel 按 `alpha/beta` 合并并写 y。workspace 需求 `W=4*S*AlignUp(O,8)` 字节（单 batch 复用）；容量不足则退回单核完整归约，保证合法调用可执行。split-K 为性能优化，首版以正确性优先。

#### 8）launch 与异步

kernel 在 `h->stream` 上异步下发，tiling 按值传递；Host 返回 SUCCESS 仅表示提交成功。workspace 由 handle 持有，生命周期覆盖 kernel 完成；不在接口内同步或释放。

### 3.2.3 kernel侧设计

Kernel 为 AIV-only，入口 `gemv_strided_batched_kernel_do(A, x, y, tiling, numBlocks, stream)`，按 `dtype` 分发到模板实例，按 `transMode` 分发到 N/T-C 数据流。核心循环：

```text
for task = blockIdx; task < taskCount; task += blockDim:
    b = task / outTilesPerBatch; tile = task % outTilesPerBatch
    outBegin = tile*R; outCnt = min(R, O - outBegin)
    A_b = A + b*strideA; x_b = x + b*stridex; y_b = y + b*stridey
    若 trans==N : NMode(...)  否则 : TMode(...)
```

#### 1）N 模式（列累加）

- 负责 `R` 个连续输出行 `[outBegin, outBegin+outCnt)`，初始化 `R` 个 FP32 累加器；
- 遍历 K 列（按 `dotTile` 分块）：每列取 A 的 R 个连续元素 `A_b[outBegin + j*lda .. ]` 搬入 UB（列内连续，列间步长 lda），取标量 `x_b[p(j,K,incx)]`（incx==1 时可预先把 x 紧凑搬入 UB 并按下标索引；非 1/负步长时按映射 gather 或逐标量读取）；
- `Muls(tmp, acol, xj)` + `Add(acc, acc, tmp)`（或 `Axpy`）累加；
- K 循环结束后 `out = alpha*acc + beta*y_old`（beta==0 不读 y），按 `incy`（含负步长）写回；FP16/BF16 输出先 Cast 再写。

列内连续、x 标量复用、R 行累加器常驻，适合大 m 场景。

#### 2）T/C 模式（按输出点积）

- 每个输出 `r∈[outBegin, outBegin+outCnt)` 对应 A 的第 r 列（`op(A)` 的第 r 行），元素 `A_b[i + r*lda]`（i=0..K-1）在内存中连续；
- 对每个 r：把 A 列按 `dotTile` 分段搬入 UB，与 `x_b[p(i,K,incx)]` 对应元素相乘，FP32 分段累加后 `ReduceSum`（或 `Mul`+逐级归约）得到点积；
- x 在同一 tile 的 R 个输出间复用；incx==1 时直接向量对齐，非 1/负步长时先 gather 到紧凑 UB；
- 结果做 `alpha/beta` epilogue，按 `incy` 写回。

C 与 T 共用此路径，不做实数共轭。

#### 3）dtype/变体处理

- 输入 `uint16_t` 视 dtype 解释为 `half`/`bfloat16_t`；`Cast` 到 FP32（上转无精度损失）参与乘加；
- 输出 HSH/TST 用 RINT 从 FP32 收窄到 FP16/BF16 再写 `uint16_t`；HSS/TSS 直接写 FP32；
- 同一模板通过 `if constexpr` 选择输入类型与收窄分支，避免 5 套重复 kernel。

#### 4）负步长与非单位步长

统一用 `p(t,L,s)` 计算逻辑→物理偏移；基址取低地址端（负步长时逻辑 0 在物理末端）。`inc==1` 走连续快速路径；`inc==-1` 以反向偏移走连续路径；其他步长走 gather/scatter（分段 `DataCopyPad` 或标量），并受物理 span 上界保护，禁止为少量元素一次搬入整段物理 span。

#### 5）缩放 kernel

`alpha==0`：逐 batch 逐输出处理 y：`beta==0` 写零、`beta==1` 跳过、其余 FP32 乘 beta 后按输出 dtype 写回；完全不访问 A/x。`O` 与 `batchCount` 作为任务空间多核并行。

#### 6）对齐、尾部与流水

- GM/Local 按 32B/256B 对齐要求；非整 tile 的 R/Kt 用有效长度与 mask，`DataCopyPad` 精确搬运；
- A 列双缓冲：`CopyIn→Compute→CopyOut` 分别建立事件/队列依赖，至少验证两轮外层迭代；
- 索引用 64 位，字节量用经溢出检查的 `size_t`；尾部补零区不得引入改变特殊值传播的额外乘法；
- 输出只写逻辑 O 个元素，incy 空洞与相邻 batch 间隙保持原值。

#### 7）workspace 与异步生命周期

常规路径不使用 workspace；split-K 借用 handle 现有 workspace（`GetEffectiveWorkspace`/`GetEffectiveWorkspaceSize`），容量/对齐不足自动退回常规路径。workspace 生命周期覆盖部分和 kernel 与 reduce kernel，不在接口内扩容/释放/同步；同一 handle+stream 顺序复用，调用方负责跨 stream 并发隔离。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练/推理系列产品（含 Atlas 800T A2 / 910B3） | √ |
| Atlas A3 训练/推理系列产品 | √ |

统一位于 `blas/gemv_strided_batched/arch22/`；性能验收设备按任务书为 Atlas 800T A2 (910B3)，精度建议覆盖 910B3 或 A3。

## 3.4 算子约束限制

1. 仅支持任务书声明的 5 个变体与列主序，不支持其他 dtype/广播/越界 stride；
2. A 恒 `lda×n`、`lda≥max(1,m)`，与 trans 无关；`trans` 只作用于 `op(A)`；
3. 要求 `strideA/stridex/stridey>0` 且相邻 batch 不重叠（按 `lda*n` 与向量物理 span 检查）；
4. `incx/incy` 非零、非 `INT_MIN`，支持负步长；y 仅写逻辑元素，保留空洞；
5. `batchCount=0`/`m=0`/`n=0` 合法 no-op；`alpha=0` 只缩放 y；`beta=0` 不读旧 y；
6. 异步执行，调用方同步 stream；不要求逐位确定性；
7. 半精度/bf16 变体 FP32 计算后收窄，精度按 §四对应 dtype 阈值判定；
8. 不定义私有平行 API，不改动公共声明与其他架构实现。

# 四、可维可测分析

## 4.1 精度标准与验证

- golden：对每个 batch 以 `op(A+i*strideA)`、`x+i*stridex`、`y+i*stridey` 独立调用 Netlib `cblas_sgemv`；H 变体输入先上转 FP32，计算后按输出 dtype 收窄；S/HSS/TSS 按 FLOAT32 判定，HSH 按 FLOAT16、TST 按 BFLOAT16 判定。可另用 float64 累加作诊断，差异在自测报告说明。
- 阈值（生态算子开源精度标准）：`matched_ratio≥0.99`；逐元素 `|actual−golden| ≤ atol + rtol·|golden|`；`rtol=atol=2^-13`（FP32）、`2^-10`（FP16）、`2^-7`（BF16）；`max_abs_error ≤` 对应 `1e-2`（FP32）/`1e-1`（FP16/BF16）或 32×ULP。
- 边界：`beta=0` golden 不读旧 y；`alpha=0` golden 只做 `beta*y`；`batch=0/m=0/n=0` 不写 y；负步长/非单位步长按逻辑顺序比对；padding 与 batch 间隙哨兵写保护；Inf/NaN 按同类匹配；5 变体分别统计。
- 负向：`handle` 空、非法 trans、负维度、lda 不足、零/INT_MIN 步长、stride≤0/重叠、batch<0、alpha/beta 空、条件指针空，逐项校验状态码。

## 4.2 性能标准与验证

- 设备：Atlas 800T A2 (910B3)；
- 正式判据：`msprof op` 采集单算子 kernel Task Duration（us），warmup 5 次不计、随后 ≥10 次有效样本取算术平均；要求单 case 与整体平均均 `gpu_ms/npu_ms ≥ 0.8`（NPU 平均耗时 ≤ `gpu_ms`×1.25），`gpu_ms` 为 `gpu_baseline.csv` 中对应变体 cuBLAS 接口的 kernel-only 耗时；
- 交叉验证：GTest 用 `aclrtEvent` 夹住连续 10 次下发打印 `[PERF] case=<name> ... avg_us=<数值>`，`verify_performance.py` 与 `gpu_ms` 比对，仅作回归，不作正式判据；
- 性能子集按变体 × 规模档分层，保留 alpha/beta 判定键，含约 12% 非默认标量；
- 报告保留命令、CANN 版本、profiling 输出、逐次计时；异常样本不计入并说明。

## 4.3 兼容性分析

- 5 个公共声明已存在，复用 `aclblasHandle_t`/枚举/状态码，不新增私有接口；
- 仅新增 `arch22` 实现，不影响 arch35 SIMT 实现；公共测试/golden 变更限定作用域并回归 arch35；
- 复用 handle/stream/workspace 机制，保持异步提交与错误码语义；
- `uint16_t` 仅承载 FP16/BF16 位模式，不改变公共 ABI；
- 测试与 README 随实现一并提交，符合 ops-blas 贡献指南，不含冗余文件。

# 五、参考资料

1. 任务书 `aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md`；
2. 设计文档模板：<https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>；
3. cuBLAS `cublas<t>gemvStridedBatched`：<https://docs.nvidia.com/cuda/cublas/index.html>；
4. Netlib BLAS `sgemv`：<https://www.netlib.org/blas/sgemv.f>；
5. ops-blas：<https://gitcode.com/cann/ops-blas>；
6. 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>。
