# 需求背景（required）

## 需求来源

CANN 社区任务 2026 任务书《aclblasSgemv 算子开发（Atlas A2/A3）》。目标是在 ops-blas 仓 Atlas A2/A3（`arch22`，DAV_2201）上补齐 BLAS-2 单精度矩阵-向量乘能力，接口对标 cuBLAS `cublasSgemv`（CUBLAS Library §2.6.2）；参数语义、返回码与快速返回口径以 cuBLAS 官方文档、Netlib `sgemv.f` 与仓内 950PR 既有实现（`blas/gemv/arch35/sgemv_host.cpp`）为准。

## 背景介绍

### aclblasSgemv 算子实现

| 组成 | 路径 | 说明 |
|---|---|---|
| 接口声明 | `include/cann_ops_blas.h` L468-470 | 与 950PR 共用既有声明，本任务**不修改**，不新增产品私有平行接口 |
| Host 实现 | `blas/gemv/arch22/sgemv_host.cpp` | 参数校验、快速返回、标量解析、tiling 计算、kernel 下发 |
| Kernel 实现 | `blas/gemv/arch22/sgemv_kernel.cpp` | arch22 设备侧**唯一编译单元**，含 trans=N / trans=T/C 两条数据通路与 `sgemv_kernel_do` 下发封装 |
| Host/Kernel 共享 tiling | `blas/gemv/arch22/sgemv_tiling_data.h` | 纯 POD，按值随 kernel 入参下发（arch22 不经 GM 传 tiling） |
| 下发入口声明 | `blas/gemv/arch22/sgemv_kernel.h` | `sgemv_kernel_do(...)`，Host 不 include 设备头 |
| 测试工程 | `test/gemv/sgemv/arch22/`（新建） | CSV 驱动 GTest、NPU wrapper、任务口径精度判定器 |
| Golden | `test/gemv/sgemv/sgemv_golden.h`（复用，不改动） | `cblas_sgemv`（OpenBLAS float32） |
| 构建接入 | `blas/CMakeLists.txt`（未修改） | `arch22/*.cpp` 由 CMake 按 `SOC_ARCH_DIRS` 自动收集，`--soc=ascend910b` 时只编译 arch22 |

### aclblasSgemv 算子实现现状分析

- 仓内 `blas/gemv/arch35/sgemv_*` 面向 Ascend 950PR/950DT（DAV_3510），依赖 `asc_simt.h` / Reg API，**不可用于 arch22**。其 `ValidateSgemvParams` 与入口顺序是本任务参数校验的语义锚点：已包含 `incx != 0` / `incx != INT32_MIN`（L44/L46）与 `incy != 0` / `incy != INT32_MIN`（L47/L49）校验，但**缺少跨度溢出守卫**（`(逻辑长度−1)×|inc|` 超 int32 域）；A2/A3 实现按任务书 §2.5 补齐该守卫。
- 同仓 arch22 兄弟算子 `cgemv`（同目录 `blas/gemv/arch22/`，host + 两个自包含 kernel TU）与 `sgbmv`（`blas/gbmv/arch22/`，经 `#include` 合并为单 TU）给出 Vector 直调工程范式。`blas/CMakeLists.txt` L65 明确 arch22 设备代码不支持跨 TU 链接，故本算子采用单一设备编译单元。
- `blas/gemv/README.md` 当前产品支持矩阵标注 A2/A3 不支持 `aclblasSgemv`；本任务合入后需同步更新（交付件 4 的算子 README 要求）。

### aclblasSgemv 算子功能分析

功能：列主序 `m×n` 矩阵 A 与向量 x 的乘加，原地更新 y：

```text
y[j] = alpha * Σ(i=1..k) op(A)[j,i] * x[i] + beta * y[j]     (j = 1..len(y))
op(A) = A   (trans = ACLBLAS_OP_N)
      = Aᵀ  (trans = ACLBLAS_OP_T)
      = Aᴴ  (trans = ACLBLAS_OP_C，实数矩阵等价于 Aᵀ)
```

| 参数名 | 输入/输出 | 参数含义 | 数据类型 | 内存/形状约束 |
|---|---|---|---|---|
| handle | 输入 | 库上下文句柄，内部携带 stream | `aclblasHandle_t` | Host；`nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| trans | 输入 | 矩阵操作类型 | `aclblasOperation_t` | Host；取值 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` |
| m | 输入 | A 的行数 | int32 | Host；`m ≥ 0`，`m=0` 在校验通过后快速返回 |
| n | 输入 | A 的列数 | int32 | Host；`n ≥ 0`，`n=0` 在校验通过后快速返回 |
| alpha | 输入 | 标量 α 指针（值不校验） | FLOAT32 | Host；`nullptr` → `INVALID_VALUE` |
| A | 输入 | 列主序系数矩阵，物理 `lda×n`，前 `m×n` 有效 | FLOAT32 | Device；`m>0 && n>0` 时 `nullptr` → `INVALID_VALUE` |
| lda | 输入 | 列主维（相邻列间距） | int32 | Host；`lda ≥ max(1,m)` |
| x | 输入 | 输入向量，逻辑长度 `trans=N ? n : m` | FLOAT32 | Device；物理长度 `1+(逻辑长度−1)×|incx|`；逻辑长度>0 时 `nullptr` → `INVALID_VALUE` |
| incx | 输入 | x 相邻元素步长 | int32 | Host；非 0、非 `INT32_MIN`；支持正负 |
| beta | 输入 | 标量 β 指针（值不校验；`β=0` 不读 y 旧值） | FLOAT32 | Host；`nullptr` → `INVALID_VALUE` |
| y | 输入/输出 | 原地输出向量，逻辑长度 `trans=N ? m : n` | FLOAT32 | Device；物理长度 `1+(逻辑长度−1)×|incy|`；逻辑长度>0 时 `nullptr` → `INVALID_VALUE` |
| incy | 输入 | y 相邻元素步长 | int32 | Host；非 0、非 `INT32_MIN`；支持正负 |

- 输入：handle、trans、m、n、alpha、A、lda、x、incx、beta、y、incy；输出：y（原地覆写）。
- 支持数据类型：仅 FLOAT32（A、x、y、alpha、beta 全部单精度实数）。
- 支持形状：二维矩阵 + 一维向量；无操作数间广播；不要求图模式动态 shape。

---

# 需求分析（required）

## 需求描述

实现 `aclblasSgemv`，语义对标 cuBLAS `cublasSgemv`。接口原型（与 `include/cann_ops_blas.h` L468-470 逐字一致）：

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);
```

- 维度映射：trans=N 时 `k=n、len(x)=n、len(y)=m`；trans=T/C 时 `k=m、len(x)=m、len(y)=n`。
- A 为列主序，有效元素 `A[i,j]`（0-based）位于偏移 `i + j*lda`，`lda ≥ max(1,m)`；`lda>m` 时列间存在空洞。
- x/y 支持任意非零 `incx/incy`（正负均可），物理长度 `1+(逻辑长度−1)×|inc|`；负步长按 Netlib 语义自物理 span 尾端起算逻辑首元素。
- y 原地覆写：`beta==0` 不读 y 旧值；实现侧 `alpha==0` **不短路**（仍按 `0·op(A)·x + beta·y` 计算）；元素值与标量值不校验，NaN/Inf 按 IEEE 规则自然传播。
- 参数校验通过后 `m==0 || n==0` 合法快速返回（不修改 y、不下发 kernel）。
- 计算经 handle 绑定 stream 异步下发，返回成功仅表示 Host 已提交，不保证完成。

## 需求拆解

| 项目 | 实现要求 |
|---|---|
| 接口与校验 | 复用公共声明；按任务书 §2.5 校验 handle、trans、m/n、lda、incx/incy（含 0、`INT32_MIN` 与跨度溢出）、alpha/beta、A、x/y 空指针，返回码逐项对齐 |
| 快速返回 | 全部参数校验通过后 `m==0 ‖ n==0` 返回 SUCCESS，不访问 A/x/y、不下发 kernel |
| trans 三态 | N 与 T/C 两条数据通路；实数域下 C 归一化为 T |
| 非连续布局 | 矩阵按 `lda` 跨列访问（列间空洞）；向量按 `incx/incy` 步长访问，正负步长均生效 |
| 原地更新 | y 输入旧值与输出复用同一内存；`beta==0` 不读旧值；A、x 只读 |
| 标量语义 | `alpha==0` 不短路；元素/标量值不校验，Inf/NaN 按 IEEE 传播 |
| 异步执行 | 经 handle 绑定 stream 异步下发，正常路径不做主机同步 |
| dynamic shape | 不要求；tiling 按运行时 m/n/lda/inc 计算，单一 kernel 入口适配任意合法规模 |
| 确定性 | 不要求跨次一致；单次执行内部按固定分核与固定列序累加 |
| 自验 | arch22 新建 GTest 工程；精度全量 1000 条 + 性能子集 200 条，并补齐 `GAUSS_NORM_5_2` 正态分布用例；判定 golden 为 `cblas_sgemv` float32 |

---

# 详细设计（required）

## 算子分析

### 数学公式

```text
y[j] = alpha * Σ(i=1..k) op(A)[j,i] * x[i] + beta * y[j]     (j = 1..len(y))
```

- 元素定位（1-based 公式、0-based 实现）：`A[i,j]` 位于列主序数组偏移 `(i−1) + (j−1)×lda`。
- trans=N：`op(A)=A`（m×n），`k=n`，`len(y)=m`；trans=T/C：`op(A)=Aᵀ`（n×m），`k=m`，`len(y)=n`。
- `beta=0` 时无 `+ beta·y[j]` 项读取；`alpha=0` 时实现仍完成归约（不短路）。

### 支持数据类型

| 数据 | 类型 | 说明 |
|---|---|---|
| A、x、y | FLOAT32 | 单精度实数，ND 布局，全值域（含 Inf/NaN） |
| alpha、beta | FLOAT32 | Host 指针（实现兼容 device 指针，见“算子约束限制”） |
| m、n、lda、incx、incy | int32 | Host 入参 |
| trans | `aclblasOperation_t` | `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` |

不使用 HF32/Cube 近似路径，全程 FP32 标量与向量运算。

### 支持形状

| 对象 | 形状约定 |
|---|---|
| A | 列主序 `lda×n`，前 `m×n` 为有效矩阵；`lda ≥ max(1,m)`，`lda>m` 时列间有空洞 |
| x | 一维，逻辑长度 `trans=N ? n : m`，物理长度 `1+(逻辑长度−1)×|incx|` |
| y | 一维（原地输入兼输出），逻辑长度 `trans=N ? m : n`，物理长度 `1+(逻辑长度−1)×|incy|` |
| 边界 | `m=0` 或 `n=0` 合法快速返回；不涉及广播；不要求图模式动态 shape |

## 算子实现

### 实现方案

采用 kernel 直调方式：Host 完成参数校验、标量解析与整数 tiling 计算，经 `sgemv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, tiling)` 下发单一 AIV-only kernel；设备侧按 `tiling.trans` 选择两条数据通路，按输出维分组分核，各核独立完成自己负责的输出元素。trans 决定输出维、归约维与 A 的取数方向，是路径划分的唯一依据：

| trans | op(A) | 归约维 k | 逻辑输出维 | A 元素访问 | 计算形态 |
|---|---|---|---|---|---|
| `ACLBLAS_OP_N` | A (m×n) | n | m | `A[i + j*lda]`，固定 i、沿 j 跨列 | 每行独立点积；列主序下同一列的 m 个元素连续 |
| `ACLBLAS_OP_T` | Aᵀ (n×m) | m | n | `A[i + j*lda]`，固定 j、沿 i 列内连续 | 每输出列独立点积；列内连续，适合整列搬运后归约 |
| `ACLBLAS_OP_C` | Aᴴ ≡ Aᵀ | m | n | 同 T | 与 T 共用通路 |

#### host侧设计

##### 1. tiling策略

tiling 为纯 POD `SgemvTilingData`（`sgemv_tiling_data.h` L23-35），按值随 kernel 入参下发（arch22 不经 GM 传 tiling），全部字段由运行时入参计算；Host 侧无动态内存申请、无 workspace。

| 字段 | 类型 | 含义 |
|---|---|---|
| `m` / `n` / `lda` | uint32 | 矩阵逻辑维度与列主维 |
| `trans` | uint32 | 0=N；1=T/C（实数域下二者等价） |
| `useCoreNum` | uint32 | 实际下发核数 |
| `outDim` | uint32 | 输出逻辑长度（N 取 m，否则 n） |
| `yUnit` | uint32 | 输出分核粒度（逻辑元素数） |
| `alpha` / `beta` | float | 标量值（Host 解析后按值携带） |
| `incx` / `incy` | int32 | 步长（已校验非 0、非 `INT32_MIN`） |

标量解析（`sgemv_host.cpp` L134-143）经 `ReadScalarFromHostOrDevice` 先判定指针位置：Host 指针直接解引用；Device 指针经 `aclrtMemcpyAsync` D2H + `aclrtSynchronizeStream` 读回。任务规格将 alpha/beta 定义为 Host 内存，device 分支属仓内共享能力的兼容扩展，不用于性能路径评估。

##### 2. 分核策略

```text
outDim    = trans==N ? m : n
yUnit     = 8 / gcd(|incy|, 8)                    // 每个分组的物理步长恒为 32B 整数倍
groups    = ceil(outDim / yUnit)
useCoreNum = min(GetAivCoreCount(), groups)       // 至少 1
```

- 优先满核；核间按连续整组均分，余组落在前几个核。
- `yUnit` 选取使组边界对应的物理步长 `yUnit×|incy|` 恒为 8 个 float（32 B）的整数倍：arch22 上多核 DMA 写同一 32B sector 会丢写，整组分配可保证不同核的写区间互斥。
- `GetAivCoreCount()` 经 `PlatformAscendCManager::GetCoreNumAiv()` 动态获取，不硬编码；返回 0 时报 `ACLBLAS_STATUS_EXECUTION_FAILED`（`sgemv_host.cpp` L127-131）。
- 只切分**输出维**，不做归约维切分：不存在跨核累加、workspace、原子操作或 `SyncAll`。

##### 3. 数据分块和内存优化策略

单核内按 UB 切分，全部缓冲区为编译期固定偏移（32B 对齐），总预算 100928 B < arch22 UB 192 KiB：

| 区 | 字节偏移 | 大小 | 用途 |
|---|---|---|---|
| A | 0 | 65536 | N：A 列块 64×256 float；T：A tile 2048 float |
| X | 65536 | 16384 | N：x 列块；T：x tile 2048 float / 串行分支 512×8 float |
| PROD | 81920 | 8192 | T：a×x 乘积 2048 float |
| YTILE | 90112 | 1024 | N：y tile 256 float |
| YSPREAD | 91136 | 8192 | 写回前 32B/元素展开区 256×8 float |
| YACC | 99328 | 512 | T：列结果 128 float |
| ACC | 99840 | 64 | T：16 个累加器 |
| XBCAST | 99904 | 1024 | N 路径广播预留（当前标量融合累加方案未使用） |
| **合计** | — | **100928 B ≈ 98.6 KiB** | < arch22 UB 192 KiB |

- 9 条 `static_assert`（`sgemv_kernel.cpp` L76-84）在编译期锁定各段边界，布局调整不可静默越界。
- 分块参数：N 路径行 tile 256（8 的倍数，保证 `blockLen` 32B 对齐）、列块 64；T 路径整块 2048 元素、串行分支 512 元素、每批写回 128 列。
- 批量搬运：A 列块用一次 `DataCopyPad(blockCount=cnt, blockLen=rows*4, srcStride=(lda−rows)*4)` 完成“多列 + 跨列空洞”搬运；x 单位步长一次连续搬入；y 写回按 `incy` 选择连续 DMA 或 32B/元素展开后跨步 DMA。
- 无 `aclrtMalloc`、无 workspace、无 double buffer（单缓冲 + 屏障），降低 UB 压力与同步复杂度。

##### 4. tilingkey规划策略

- **不设 TilingKey**：arch22 直调按值传 tiling，无 GM tiling 结构，也无需 `TilingKey` 机制。
- 唯一分支维度为 `trans`：Host 归一化为 `trans=0`（N）或 `trans=1`（T/C），设备侧在 kernel 入口按该字段进入 `SgemvNoTrans` / `SgemvTrans`；实数域 C 与 T 共用通路。
- 若后续新增数据通路（如归约维切分）再引入独立分支标识，避免当前无收益的 key 组合。

#### kernel侧设计

##### 1. 入口与区间划分

```text
blk >= useCoreNum 或 yUnit==0 -> 直接返回
groups  = ceil(outDim / yUnit)
perCore = ceil(groups / useCoreNum)
g0 = blk*perCore; g1 = min(groups, g0+perCore)
start = g0*yUnit;  end = min(outDim, g1*yUnit)
trans==0 -> SgemvNoTrans(start, end) ; 否则 -> SgemvTrans(start, end)
```

入口见 `sgemv_kernel.cpp` L332-366，声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，纯 Vector 任务，不使用 Cube/SIMT/Reg API。每核获得连续的整组输出区间，核间无写重叠、无同步。

##### 2. trans = N 路径（`SgemvNoTrans`）

以“行 tile × 列块”组织，每行维护一个 UB 累加器：

| 步骤 | 实现 |
|---|---|
| 行 tile | 每次 256 行；`rows = min(256, endRow−rs)`，UB 目的按 `aStride=AlignUpSimd(rows)` 对齐 |
| y 初始化 | `beta==0`：`Duplicate(yTile, 0)`（**不读** y）；否则逐行标量读 `yG.GetValue(PhysIdx(...))` 并做一次 `beta*old` |
| 列块 | 每次 64 列，`cnt = min(64, n−js)` |
| α·x | `incx==1`：一次连续 DMA 搬 `cnt` 个 x，再 `Muls(alpha)`；否则逐元素标量读 `alpha*x_j` |
| A 列块 | `DataCopyPad(aG[rs + js*lda], {blockCount=cnt, blockLen=rows*4, srcStride=(lda−rows)*4})`，列主序下每列 `rows` 个 float 连续 |
| 累加 | 逐列按 `y_i += A[i,j] · (alpha·x_j)`（j 升序 AXPY 形式）做**标量融合乘加** |
| 写回 | `StoreToY`：`incy==1` 单次连续 DMA；否则按物理升序展开到 `ySpread` 后跨步 DMA |

设计要点：单行累加器等价于 OpenBLAS `gemv_n` 的逐列 AXPY 顺序；标量乘加在 arch22 上编译为融合乘加，与 golden 的单次舍入语义一致（见“精度设计”）。

##### 3. trans = T/C 路径（`SgemvTrans`）

按输出列分核，每列独立点积；每批处理 128 列，结果先在 UB 聚成 `yAcc` 再整体写回。

**（a）单位步长 `incx==1`（`ColumnDotUnitStride`）**

```text
acc[0..16) = 0
按 2048 元素分块：A 列连续搬入 + x 连续搬入
  对块内每 32 元素：acc += prod[b..b+16); acc += prod[b+16..b+32)
逐 lane 归约：s[l] = ((acc[l]+acc[4+l])+acc[8+l])+acc[12+l]，l=0..3
m%32 尾部：按 4 元素组累加进 s[0..3]；sum = (s0+s2)+(s1+s3)
m%4   尾部：逐元素标量累加
```

该结构镜像 golden（OpenBLAS 0.3.25 tsv110 单线程 BLAS2 gemv）的累加结构，避免长归约误差偏置。

**（b）非单位步长 `incx!=1`（`ColumnDotStrided`）**

按 i 升序**单累加器串行**：A 列连续搬入；x 以 `blockCount=cnt、blockLen=4B、srcStride=(|incx|−1)*4` 跨步搬入 UB（槽位按 8 float 布局）；负步长从分块物理最小端起搬、按槽位反序取用。该分支严格遵循 golden 的串行累加次序，是精度兜底而非性能路径。

**（c）标量合并**

```text
yOld = (beta==0) ? 0 : beta * yG.GetValue(PhysIdx(j, outDim, incy))   // beta==0 不读 y
yAcc[c] = alpha * sum + yOld                                          // alpha==0 不短路
```

##### 4. 步长与写回

- `PhysIdx(logical, count, inc)` 统一处理正负步长：`inc>0` 取 `logical*inc`，否则取 `(count−1−logical)*|inc|`。
- `StoreToY` 在 `incy!=1` 时把 UB 结果按物理地址升序展开到 `ySpread`（每元素占 8 float 槽位），再以 `blockCount=count、blockLen=4B、dstStride=(|incy|−1)*4` 一次 DMA 写回；负步长通过展开时的逻辑逆序实现。
- 写回一律经 MTE3 `DataCopyPad`，不使用标量写 GM（避免多核并发写同一 cache line 丢写）。

##### 5. 尾块处理

| 尾块来源 | 处理 |
|---|---|
| `n % 64`（N 路径归约维） | `cnt=min(64, n−js)`，DMA 只搬 `cnt` 列，无需补零 |
| `rows % 256`（N 路径输出 tile） | `rows=min(256,…)`；`blockLen=rows*4` 为字节粒度，UB 按 `AlignUpSimd(rows)` 布局 |
| `outDim % yUnit`（输出分核） | `end=min(outDim, g1*yUnit)`；`StoreToY` 只写 `end−start` 个元素 |
| `m % 32` / `m % 4`（T 路径单位步长归约） | 标量尾部：4 元素组累加进 `s[0..3]`，剩余逐元素累加（镜像 golden） |
| `m % 512`（T 路径跨步归约） | `cnt=min(512, m−is)`，逐元素串行累加 |
| `count % 128`（T 路径列批） | `count=min(128, endCol−cs)` |

##### 6. 同步与流水

- 同一核内使用 `PIPE_BARRIER(ALL)` 保护“DMA 覆写 UB ↔ 标量/向量流水读取”与“写回前结果就绪”，均为核内流水同步。
- 无跨核同步：不使用 `SyncAll`、`CrossCoreSetFlag/WaitFlag`、`SetAtomicAdd`，也不使用 handle workspace。
- 单设备编译单元：`sgemv_kernel.cpp` 内含 kernel 入口、`sgemv_kernel_do` 与两条路径的全部 `inline` 辅助函数，满足 `blas/CMakeLists.txt` L65 所述 arch22 跨 TU 链接限制。

##### 7. 精度设计

- **判定 golden**：`cblas_sgemv`（OpenBLAS float32，列主序），经仓内 `aclblasSgemv_cpu` 调用，与 `sgemv_golden.h` 一致；float64 收窄参考仅作 `[PREC64]` 诊断，不参与判定（依据任务书 §3.2/§3.5/§7-4）。
- **判定口径**：FLOAT32 `atol = rtol = 2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ max(1e-2, 32×ULP(g_low))`，非有限值按等效规则单独判定；实现位置 `sgemv_precision_arch22.h::EvaluateSgemvTask`。
- **累加顺序保证**：内核按 golden 的归约语义组织——N 路径逐列 AXPY（j 升序）、T 路径单位步长 16 累加器/32 元素分块 + 固定树形归约、T 路径跨步按 i 升序单累加器。该顺序属于精度约束，不随性能调参改动。
- **舍入语义（关键决策）**：golden 的 `gemv_n` 使用融合乘加（单次舍入）；arch22 向量乘加（`MulAddDst`/`FusedMulAdd`）实测均为两次舍入，长归约用例会累积到 70+ ULP 超出 32 ULP 门限；标量乘加在 arch22 被编译为融合乘加，与 golden 逐位一致。故 N 路径逐列采用**标量融合乘加**累加（`sgemv_kernel.cpp` L196-208）。开发期位级验证：`std::fma` 与 golden **4/4 位级相等**，非融合乘加在长归约用例累积至 **71 ULP**，T 路径长链 `max_abs_error=0`。
- **原地/标量语义**：`beta==0` 不读 y；`alpha==0` 不短路；8 条 `alpha=0` 且 A/x 含 Inf/NaN/EXTREME 的冻结用例按已冻结的判定规则处理（实际为 NaN 或与 golden 满足混合容差均通过），不放宽任何有限值误差。

---

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（arch22，DAV_2201） | √（本机 Ascend 910B3 实测） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（arch22，构建参数 `--soc=ascend910_93`） | √（共用同一 arch22 源码；本机未实测，最终版按 A3 构建/抽查结论补充） |
| Ascend 950PR / Ascend 950DT（arch35） | 既有实现，本次不改动 |

---

## 算子约束限制

- 仅支持 FLOAT32、列主序与任务约定的 lda/inc 布局；无广播、无 dynamic shape 要求、不要求确定性。
- A、x 只读，y 原地覆写；调用方须保证操作数与 y 的存储生命周期覆盖 stream 完成时点。
- 标量 alpha/beta 规格定义为 Host 指针；实现经 `ReadScalarFromHostOrDevice` 兼容 device 指针，该兼容分支会引入 D2H + `aclrtSynchronizeStream`，不属于性能评估路径。
- 跨度溢出守卫在元素域以 int64 计算 `(len−1)×|inc| ≤ INT32_MAX`，x/y 各查一次（`sgemv_host.cpp` L54-61、L99-104）；`DataCopyPad` 的 `srcStride/dstStride` 为 uint32 字节域，理论上 `(|inc|−1)×4` 或 `(lda−rows)×4` 可能溢出。按当前用例集复算（`test/gemv/sgemv/arch22/sgemv_test.csv`，sha256 `223d5b66…`，2169 数据行）：**SUCCESS 用例** `|incx|`、`|incy|` 最大均为 3，`lda` 最大 2097152（86 条 SUCCESS 用例 `lda>8192`，如 `TC_EX_0514/0686/0738` 等 10 条 `lda=2097152`）；**`TC_PF` 性能子集** `incx=incy=1`、`lda` 最大 1048576。对应字节域步长上界为 `(2097152−1)×4 ≈ 8.4 MB` 与 `(3−1)×4 = 8 B`，均远小于 uint32 上限，当前用例集不触发字节域溢出。`|inc|` 接近 1.07e9 或 `lda` 接近 uint32 上限的极端参数不在用例集内，属已知边界，后续可补字节域守卫或降级标量路径。

**已知限制与规避（设计分析，非实测结论）**

| 编号 | 限制 | 影响 | 规避/后续方向 |
|---|---|---|---|
| L1 | 分核只切输出维：`outDim` 较小时（如 N 4×1048576，`outDim=4`、`yUnit=8` → 仅 1 个输出组）仅 1 个核获得全部输出 | 极端矩形用例并行度不足 | 精度不受影响；性能迭代时评估归约维切分 + 单写者/atomic 方案（需专门验证 `beta·y` 只计一次） |
| L2 | N 路径 `incx != 1` 时逐元素标量读 x | 跨步大 shape 吞吐下降 | `TC_PF` 子集均为 `incx=1`；后续可建 offset 表 + `Gather` 落连续 UB |
| L3 | T 路径 `incx != 1` 时单累加器串行 | 大归约维跨步用例为标量级吞吐 | 精度兜底路径；`TC_PF` 子集不含该形态 |
| L4 | `beta != 0` 时逐元素标量读 y | 增加标量 GM 访问 | `TC_PF` 子集全部 `beta=0`，正常路径不读 y；非性能用例可能受影响 |
| L5 | N 路径改为标量融合累加 | 逐元素吞吐低于纯向量累加（精度优先） | 已在开发期以位级证据确定；性能优化须保持融合舍入语义 |
| L6 | alpha/beta 为 device 指针时 D2H + 同步 | 破坏异步、增加固定开销 | 规格为 Host 指针，正常路径不触发 |
| L7 | `DataCopyPad` stride 字节域极端值溢出 | 极端参数下 stride 计算不正确 | 当前用例集不触发（上界见前条）；建议补字节域守卫或降级标量路径 |
| L8 | A3 未在本机实测 | A3 结论为推演 | arch22 同源；最终版补 A3 构建/抽查结论 |

**任务书 §2.5 约束逐项核对结论**

| 约束项 | 任务书要求 | 实现位置（文件 : 函数） | 核对结论 |
|---|---|---|---|
| 参数合法性与校验层级 | handle→`HANDLE_IS_NULLPTR`；其余非法→`INVALID_VALUE`；**跨度溢出守卫**由 A2/A3 补齐；校验通过后 m=0/n=0 合法快速返回；`alpha=0` 不短路；元素/标量值不校验 | `sgemv_host.cpp` : `aclblasSgemv`（L110-176）、`ValidateSgemvParams`（L63-106）、`StridedSpanOverflow`（L54-61）；`sgemv_kernel.cpp` : N 路径融合累加 L196-208、T 路径标量合并 L320-322 | **满足**。校验顺序：① handle 空 → `HANDLE_IS_NULLPTR`；② trans∉{N,T,C}；③ m<0；④ n<0；⑤ lda<max(1,m)；⑥ incx==0；⑦ incx==INT32_MIN；⑧ incy==0；⑨ incy==INT32_MIN；⑩ alpha 空；⑪ beta 空；⑫ m>0&&n>0 时 A 空；⑬ xDim>0 时 x 空；⑭ yDim>0 时 y 空；⑮ x 跨度溢出；⑯ y 跨度溢出 → ②–⑯ 均 `INVALID_VALUE`。跨度溢出用 int64 计算、x/y 各查一次（arch35 缺该项，本实现已补齐）；快速返回位于全部校验之后（L123-125）；alpha=0 全程参与计算 |
| 非连续 Tensor 支持 | 列主序按 lda 跨列访问（列间空洞）；向量按 incx/incy 步长访问；正负步长生效（负步长自物理 span 尾端起算）；inc 为 0/INT32_MIN 报错 | `sgemv_kernel.cpp` : A 列块 `srcStride=(lda−rows)*4` L189-193、T 路径跨步搬入 L280-289、`PhysIdx` L87-93、`StoreToY` L121-144；`sgemv_host.cpp` L77-82 | **满足**（功能语义）。跨步形态吞吐限制见 L2/L3 |
| broadcast 规则 | 不涉及（无操作数间广播） | 无广播相关代码；维度由 m/n/trans 唯一确定 | **不适用**（与任务书一致） |
| dynamic shape 要求 | 不要求（静态 shape 满足） | `sgemv_host.cpp` L145-165 按运行时 m/n/lda/inc 生成 tiling；`sgemv_kernel.cpp` L361-365 单一入口按 tiling 分支 | **满足**。无图模式动态 shape 依赖；任意合法规模一次适配 |
| 原地与视图语义 | y 原地覆写；beta 读旧值、`beta=0` 不读；A/x 只读；不返回视图 | `sgemv_kernel.cpp` : N 路径 `beta==0` 走 `Duplicate` L164-166、T 路径 `beta==0` 时 `yOld=0` L321、`StoreToY` 只写 y L121-144 | **满足**。A/x 入参为 `const float*`，设备侧仅构造只读访问 |
| 确定性计算要求 | 不要求 | 无跨核归约/atomic/SyncAll；每核固定区间、固定列序、固定累加树 | **满足**（要求为“不要求”；实现单次执行内部确定） |
| 空 Tensor 处理 | m=0 或 n=0：不访问 A/x/y、不启动 kernel、返回成功 | `sgemv_host.cpp` L123-125（校验后快速返回，先于核数查询与下发） | **满足** |
| 0 维 Tensor 处理 | 不涉及 | 无 0 维输入；标量语义由 m/n 与 Host 标量承担 | **不适用**（与任务书一致） |
| 异步执行 | 经 handle 绑定 stream 异步下发，正常路径不主机同步 | `sgemv_host.cpp` L171-173（`sgemv_kernel_do` → `<<<numBlocks, nullptr, h->stream>>>`）；正常路径无 `aclrtSynchronizeStream`/无 malloc | **满足**（规格内 Host 标量为前提）；device 标量兼容分支会同步 stream，属规格外扩展，见 L6 |

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
|---|---|---|
| 精度标准 | FLOAT32：`atol = rtol = 2^-13`；`matched_ratio ≥ 0.99`；`max_abs_error ≤ max(1e-2, 32×ULP(g_low))` | 任务书 §3.2；生态算子开源精度标准 mixed_tolerance |
| 精度判定公式 | 逐元素 `|actual − golden| ≤ atol + rtol×|golden|`；同时满足 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max_abs_error_limit` 才判该用例通过 | 任务书 §3.2 |
| Inf/NaN 判定 | golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 另一方 ±Inf 视为等效；一方有限另一方非有限、或异号 Inf 判不通过；非有限值不计入 matched_ratio 与 max_abs_error，但须全部按等效规则通过 | 任务书 §3.2 |
| 精度 golden | `cblas_sgemv`（OpenBLAS float32，列主序），与仓内 `sgemv_golden.h` 一致；float64 收窄参考仅诊断 | 任务书 §3.2/§3.5/§7-4 |
| 性能标准 | 平均单次耗时不高于 GPU 标杆耗时的 1.25 倍（即 ≥0.8× GPU 基线）；200 条 `TC_PF` 性能用例全部 PASS、无 NO_REF | 任务书 §3.3 |
| 性能采样与判定数据来源 | 预分配 device buffer，H2D 一次；warmup 5 次不计时；随后在一个 `aclrtEvent` 区间内连续下发 N 次（`m×n ≤ 1048576` 取 50，否则 20，均 ≥10），`aclrtSynchronizeStream` 后 `avg_us = elapsed/N`；H2D/D2H/golden 不计入计时；GTest 打印 `[PERF] case=<name> ... avg_us=<数值>`，判定只用该 `avg_us` | 任务书 §3.3；已冻结性能采样协议 |
| 性能交叉验证 | 任务书 §3.3 规定的 msprof op `Task Duration` 仅作交叉验证，不参与判定；两者口径不同（`aclrtEvent` 含 Host 下发与核间调度，msprof 仅统计 kernel task），报告同时记录偏差 | 任务书 §3.3；已冻结性能采样协议 |
| 性能基线换算 | 判定阈值 = `gpu_ms/0.8×1000` µs；性能匹配键 `(trans,m,n,alpha,beta,lda,incx,incy)` 联合匹配 | 任务书用例包 `gpu_baseline.csv`、`verify_performance.py` |

- 测试工程：`test/gemv/sgemv/arch22/`，CSV 驱动 GTest（`sgemv_test.csv` 2169 条数据行 + `NullHandle`），golden 与阈值复用仓内既有资产；期望非 SUCCESS 的负向用例以单元素 dummy 占位、只断言返回码，规避 `INT32_MIN` 与溢出乘积的分配 UB；性能用例同样执行一次精度判定。
- 用例构成：原始包 1200 行（1000 精度 + 200 性能）逐字节保留在前，新增 968 条 `TC_GS_*`（`GAUSS_NORM_5_2` 正态镜像）与 1 条 `TC_ED_199`；用例名唯一，期望码 SUCCESS 2143 / INVALID_VALUE 26。
- **开发同步期实测状态**（最终版按定稿实测修订）：
  - 构建：官方入口 `build.sh --soc=ascend910b --ops=sgemv` 通过，构建输出 `SOC_VERSION=ascend910b, NPU_ARCH=dav-2201` 与 `Built target sgemv_test`（构建日志开发期留存）。
  - 全量回归：`Running 2170 tests from 2 test suites` 且 `[ PASSED ] 2170 tests`（2169 条 CSV 用例 + `NullHandle`），精度用例全部通过 GTest 断言（回归日志开发期留存）。
  - 全量性能：同一回归记录包含 200 条 `[PERF] case=... avg_us=...` 原始采样；**尚未执行 `verify_performance.py` 的 GPU 基线逐例判定与 msprof 交叉验证，性能验收结论待回填（最终验收后修订）**。
  - msprof `Task Duration` 交叉验证（4096² N/T/C 锚点）：**待回填（最终验收后修订）**。

## 兼容性分析

- **接口**：复用 `include/cann_ops_blas.h` 既有 `aclblasSgemv` 声明（L468-470），不改签名、不改 ABI、不新增平行接口；与 950PR/950DT 共用。
- **源码边界**：仅新增 `blas/gemv/arch22/sgemv_{host,kernel}.cpp`、`sgemv_kernel.h`、`sgemv_tiling_data.h` 与 `test/gemv/sgemv/arch22/`；不修改 `blas/gemv/arch35/`、`test/gemv/sgemv/arch35/`、`include/cann_ops_blas.h` 与任务书用例包。
- **构建系统**：根 `CMakeLists.txt`/`blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 自动收集 arch22 源码（`ascend910b*`/`ascend910_93*` → arch22，`ascend950*` → arch35）；`test/gemv/sgemv/CMakeLists.txt` 当前为无条件 `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)`，配合 arch22 源发现即可编译，**本任务无需改动 CMake**（与任务书 §2.2“如需要则补齐”的差异以实际仓内容为准）。
- **产品支持矩阵**：`blas/gemv/README.md` 的 A2/A3 支持状态需随本任务更新（交付件 4 的算子 README 要求）。
- **可维护性**：Host 只做校验/分核/tiling/下发，设备侧单 TU 内含入口 + 两条路径 + 写回工具；tiling 为纯 POD 且 host/kernel 共享同一头文件；UB 预算由 `static_assert` 编译期校验；校验集中在一个函数内按序短路，新增校验项只需插入对应位置并补一条负向用例；无 workspace、无原子、无跨核屏障，降低死锁与丢写类问题排查面。
