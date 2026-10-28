# aclblasChpr 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasChpr` |
| 任务名称 | 10月社区任务-aclblasChpr 算子开发（950） |
| 目标硬件 | Ascend 950PR（arch35） |
| 软件版本 | CANN 9.1.0 |
| 开发方式 | Ascend C Kernel 直调（handle 绑定 stream） |
| 数据类型 | COMPLEX64（`aclblasComplex`，实部/虚部均为 FLOAT32）；`alpha` 为 FLOAT32 **实数** |
| 对标接口 | cuBLAS `cublasChpr` |
| 精度 golden | cblas（Netlib BLAS 3.10.0）`cblas_chpr` 单标杆 |
| 合入仓及目录 | `cann/ops-blas`：实现 `blas/hpr/arch35/`，测试 `test/hpr/chpr/arch35/` |
| 接口声明 | 复用 `include/cann_ops_blas.h:406-408` 既有声明，**不新增、不重复定义、无 950PR 私有平行 API** |
| 文档状态 | 设计稿，待评审。§7.1/§7.2 的实测结果字段标注为「待自测回填」，未填入任何未实测数值 |

---

# 1. 需求背景（required）

## 1.1 需求来源

本需求来自昇腾 CANN 社区任务《10月社区任务-aclblasChpr 算子开发（950）》。任务要求在 Ascend 950PR（arch35）上使用 Ascend C 补齐单精度复数 Hermitian 压缩存储秩-1 更新接口 `aclblasChpr` 的 arch35 实现，完成 Host 参数处理、Device Kernel、测试工程、算子 README 与自测材料，验收通过后合入 `cann/ops-blas`。

本任务为**补齐 Blas 仓能力**：`aclblasChpr` 的 API 声明与 arch22 实现均已存在，缺的是 arch35（950PR）实现。因此本任务的边界非常明确——**不碰 API，只补 arch35**。

## 1.2 背景介绍

### 1.2.1 算子简介

`aclblasChpr` 属于 BLAS Level 2，对以 **packed（压缩）格式**存储的 Hermitian 复数矩阵执行原地秩-1 更新：

```text
A := alpha * x * conjg(x^T) + A
```

- `A` 为 n×n Hermitian 复数矩阵，以长度 `n(n+1)/2` 的一维复数数组 `ap` 压缩存储，**无前导维 lda**；
- `x` 为 n 元素 complex64 向量，步长 `incx`（支持负步长）；
- `alpha` 为 **FLOAT32 实数**标量（非复数）。

`alpha` 必须为实数，是 Hermitian 性质的数学要求而非接口简化：秩-1 更新量 `alpha·x·x^H` 满足 `(alpha·x·x^H)^H = conj(alpha)·x·x^H`，仅当 `alpha` 为实数时该更新量才是 Hermitian 的，`A` 才能在更新后仍为 Hermitian。这一点与同族复数对称算子 `aclblasCsyr`（`alpha` 为复数）形成对照，是本算子接口上最易被误解的一处，也是 §2.3.2 中与 `Csyr`/`Cher` 的差异点来源。

Hermitian 的直接推论：对角元素 `A(j,j)` 必为实数。其更新量为 `alpha·x_j·conj(x_j) = alpha·|x_j|²`，恒为实数。

### 1.2.2 标杆算子现状分析

#### 1.2.2.1 标杆支持的数据类型与数据格式

| 标杆 | 接口 | alpha | x / ap | 存储格式 | 三角引用 |
| --- | --- | --- | --- | --- | --- |
| cuBLAS | `cublasChpr` | `const float*`（实数，HOST/DEVICE pointer mode） | `cuComplex` | packed，长度 n(n+1)/2，列主序 | 仅 `uplo` 指定三角 |
| Netlib BLAS | `chpr.f` | `REAL ALPHA` | `COMPLEX` | 同上 | 同上 |

两者参数序列、语义完全一致，本算子与之逐参数对应。

#### 1.2.2.2 标杆实现描述与关键语义

Netlib `chpr.f`（UPPER 分支）的核心循环：

```fortran
DO 20 J = 1,N
    TEMP = ALPHA*CONJG(X(JX))
    DO 10 I = 1,J-1
        AP(K) = AP(K) + X(IX)*TEMP          ! 非对角
10  CONTINUE
    AP(KK+J-1) = REAL(AP(KK+J-1)) + REAL(X(JX)*TEMP)   ! 对角
20 CONTINUE
```

从中提取出三条必须在实现中精确复刻的语义：

1. **标量先并入列项**：`TEMP = alpha*conjg(x_j)` 先算，再与 `x_i` 做复数乘。这决定了浮点运算的结合顺序，直接影响与 golden 的逐位接近程度（见 §3.2.1、§7.1）。
2. **对角实部**：`REAL(AP(diag)) + REAL(x_j*TEMP)` = `REAL(AP(diag)) + alpha*|x_j|²`。
3. **对角虚部被丢弃，而非被累加**：`REAL(AP(kk+j-1))` 取的是**输入旧值的实部**，输入的虚部被直接舍弃，输出虚部为 0。这不是「加了个 0」——若输入对角虚部为 NaN/Inf，正确结果仍是 **0**，而非 NaN。这是本算子最容易写错的一处，也是 §3.4.5 单独设计的原因。

#### 1.2.2.3 标杆实现流程

```text
quick return: n==0 或 alpha==0  ──► return
        │
        ▼
确定 x 起始物理下标 KX（incx<0 时 KX = 1-(n-1)*incx，即反向遍历）
        │
        ▼
for j = 0 .. n-1:
    TEMP ← alpha * conj(x_j)
    ├─ 非对角元素 (i≠j)：ap(i,j) += x_i * TEMP
    └─ 对角元素 (i=j)  ：ap(j,j) ← Re(ap(j,j)) + alpha*|x_j|²，虚部置 0
```

### 1.2.3 ops-blas 仓内现状

| 位置 | 现状 | 本任务动作 |
| --- | --- | --- |
| `include/cann_ops_blas.h:406-408` | `aclblasChpr` 声明**已存在** | 直接复用，不改动 |
| `blas/hpr/arch22/` | `chpr_host.cpp` / `chpr_kernel.{cpp,h}` / `chpr_tiling_data.h` 已实现（A2/A3） | 作为复数语义参考，不改动 |
| `blas/hpr/arch35/` | **不存在** | **本任务新增** |
| `blas/spr/arch35/` | `sspr_host.cpp` / `sspr_kernel.cpp` / `sspr_tiling_data.h` 已实现（实数 packed 秩-1，SIMT 范式） | 作为 arch35 工程范式与分核参考 |
| `blas/hpr/README.md` | 产品支持表标注「Ascend 950PR / Ascend 950DT：**不支持**」 | **本任务改为「支持（arch35）」**，并补齐参数说明/约束/示例 |
| `test/hpr/chpr/` | `chpr_param.h` / `chpr_golden.h` / `CMakeLists.txt` / `arch22/` 已存在 | 新增 `arch35/`，复用公共 param/golden（需处理列名映射，见 §7.3.2） |

本任务的技术定位因此十分清晰：**`blas/hpr/arch22`（复数 Hermitian 语义）× `blas/spr/arch35`（arch35 SIMT 工程范式）的交集**。两个"父本"均在仓内可直接对照，这既降低了语义风险，也意味着评审会以这两者的规范为基准。

---

# 2. 需求分析（required）

## 2.1 外部组件依赖

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| CANN 9.1.0（ASC 编译器、`kernel_operator.h`、`simt_api/asc_simt.h`） | arch35 SIMT kernel 编译 | `blas/spr/arch35` 已验证该工具链可用 |
| `acl` 运行时（`aclrtMalloc` / `aclrtMemcpyAsync` / stream） | Device 内存与异步下发 | — |
| cblas（Netlib BLAS 3.10.0） | 精度 golden `cblas_chpr` | 随测试工程提供 |
| GTest | 测试框架 | 仓内 `test/frame` 已封装 `ops_blas_add_gtest_tests` |

不引入任何新的三方依赖。

## 2.2 内部适配模块

| 模块 | 复用方式 |
| --- | --- |
| `common/helper/aclblas_handle_internal.h` | 从 handle 取 stream（`h->stream`） |
| `common/helper/kernel_constant.h` | `SIMT_MIN_THREAD_NUM = 128`、`SIMT_MAX_THREAD_NUM = 2048` |
| `common/helper/host_utils.h` | `CeilDiv` / `CeilAlign` / `GetAivCoreCount` / `CheckPtrLocation` / `ReadHostScalarFromDevice` |
| `cann_ops_blas_common.h` | `aclblasComplex`、`aclblasFillMode_t`、`aclblasStatus_t`、`CHECK_RET` |
| `blas/CMakeLists.txt` | **无需改动**：按 `SOC_ARCH_DIRS` 对 `blas/*/arch35/*.cpp` 做 `GLOB_RECURSE` 自动收集（见 §3.2.2） |

## 2.3 需求模块设计

### 2.3.1 算子原型与功能

```cpp
aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha,
    const aclblasComplex* x, int incx,
    aclblasComplex* ap);
```

逐参数与 `cublasChpr` 对应，与仓内实数对应算子 `aclblasSspr` 逐参数对应（`x`/`ap` 类型由 `float` 扩展为 `aclblasComplex`），无需额外映射说明。

### 2.3.2 约束与异常行为

校验顺序固定如下（任务书 §2.5）：

| 序号 | 条件 | 返回值 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（**直接返回，不再校验 uplo/incx/alpha/x/ap**） |
| 4 | `uplo ∉ {ACLBLAS_UPPER, ACLBLAS_LOWER}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `incx == 0` 或 `incx == INT_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `alpha == nullptr` 或 `x == nullptr` 或 `ap == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `alpha == 0.0f` | `ACLBLAS_STATUS_SUCCESS`（no-op，`ap` 不变，`x` 内容不被读取） |

两处必须精确遵守的次序，也是测试用例直接覆盖的点：

- **`n == 0` 的优先级最高**：`n=0` 叠加非法 uplo / `incx=0` / `incx=INT_MIN` / 空指针，一律期望 `SUCCESS`（CSV 用例 `n0_*`）。
- **`alpha == 0` 的优先级最低**：空指针与非法步长的校验**先于** `alpha=0` 的 quick return，因此 `n>0` 且 `alpha=0` 叠加空指针/零步长，仍期望 `INVALID_VALUE`（CSV 用例 `alpha0_*`）。

> 关于 uplo 非法的返回码：仓内 `blas/spr/README.md` 的参数表写作 `ACLBLAS_STATUS_INVALID_ENUM`，但 `blas/spr/arch35/sspr_host.cpp` 与 `blas/hpr/arch22/chpr_host.cpp` 的实现均返回 `ACLBLAS_STATUS_INVALID_VALUE`，任务书 §2.4/§2.5 亦明确为 `INVALID_VALUE`。本实现以**任务书与既有实现为准**返回 `INVALID_VALUE`，并在 `blas/hpr/README.md` 中按实际行为书写，不沿用 spr README 的该处笔误。

与同族算子的差异（避免混用）：

| 算子 | 公式 | alpha 类型 | 存储 | 对角特殊处理 |
| --- | --- | --- | --- | --- |
| `aclblasChpr`（本算子） | `A += alpha·x·x^H` | **实数** float | packed | **虚部置 0** |
| `aclblasCsyr` | `A += alpha·x·x^T` | 复数 | 全存储（含 lda） | 无 |
| `aclblasSspr` | `A += alpha·x·x^T` | 实数 float | packed | 无（实数矩阵） |

### 2.3.3 数学与内存边界

**packed 索引映射**（0-based，列主序，列内无间隙）：

| uplo | 有效范围 | 索引公式 | 列基址 `colBase(j)` | 列内偏移 |
| --- | --- | --- | --- | --- |
| `ACLBLAS_UPPER` | `i ≤ j` | `ap[i + j(j+1)/2]` | `j(j+1)/2` | `i` |
| `ACLBLAS_LOWER` | `i ≥ j` | `ap[i + (2n-j-1)j/2]` | `j(2n-j+1)/2` | `i - j` |

LOWER 的两种写法等价，代入 `i=j` 均得 `j(2n-j+1)/2`，与仓内 `SsprColumnBase`/`ChprColumnBase` 的 `col*(2n-col+1)/2` 一致。本实现统一采用 `colBase(j) + 列内偏移` 形式，因为它直接暴露了**列内 stride = 1** 这一关键性质（见 §3.4.2）。

**内存边界**：

- `ap` 元素数 `P = n(n+1)/2`，字节数 `8P`。`n=4096` 时 `P = 8,390,656`，`ap` 占 67.1 MB。
- `x` 物理元素数 `1 + (n-1)·|incx|`，字节数 `8·(1+(n-1)·|incx|)`。
- **索引一律用 `uint64_t`**：`n=65536` 时 `P ≈ 2.15e9` 已溢出 `int32`/`uint32`。虽超出本批次性能规模，但 `n` 是运行时入参，不设上限，故索引类型必须按 64 位设计。仓内 `ChprColumnBase` 同样返回 `uint64_t`。
- **负步长物理下标**：逻辑元素 `i`（0-based）取自物理位置 `(n-1-i)·|incx|`。该式与 Netlib 的 `KX = 1-(n-1)*INCX` 逐元素递增 `INCX` 等价（以 `incx=-2, n=4` 验证：Netlib 物理 1-based 下标序列 7,5,3,1，对应 0-based 6,4,2,0，与 `(3-i)·2` 一致）。

### 2.3.4 quick return 与标量位置

- `n == 0`：`P = 0`，无元素可更新，直接返回 `SUCCESS`，**不下发 kernel**。
- `alpha == 0`：更新量恒为 0，`ap` 保持不变。注意此时**对角虚部不被置 0**——因为 Netlib/cblas 在 `alpha==0` 时同样直接 `RETURN`，golden 侧亦不改写对角虚部，两侧口径一致。
- **alpha 指针位置**：任务书 §2.4 规定 `alpha` 为 Host 内存（与 cuBLAS 默认 pointer mode = HOST 一致）。但仓内 arch35 已按 issue #510 统一支持 Host/Device 两种 `alpha` 指针：`blas/spr/arch35/sspr_host.cpp` 先 `CheckPtrLocation` 判定，Device 指针经 handle stream 做 D2H 读回后再解引用，**不可在 Host 直接解引用 Device 指针**（否则崩溃或读到错误标量）。本实现沿用该范式，在满足任务书要求的同时与仓内 arch35 约定保持一致；arch22 的 `chpr_host.cpp` 直接 `*alpha` 的写法不作为参照。

### 2.3.5 需求拆解

1. **Host 实现**：`blas/hpr/arch35/chpr_host.cpp` —— 参数校验（§2.3.2 固定序）、alpha 指针位置判定、quick return、tiling 计算、kernel 下发。
2. **Kernel 实现**：`blas/hpr/arch35/chpr_kernel.cpp` —— SIMT 模型，平坦均分 + 行向合并访问主路径、UB-x 缓存、对角虚部置 0、任意 incx（含负步长）支持。
3. **Tiling 结构**：`blas/hpr/arch35/chpr_tiling_data.h`。
4. **测试工程**：`test/hpr/chpr/arch35/` —— CSV 驱动 GTest，golden 为 `cblas_chpr`，覆盖随任务提供的 1200 条用例。
5. **文档**：更新 `blas/hpr/README.md` 产品支持表与参数说明；输出自测报告。

---

# 3. 详细设计（required）

## 3.1 调用方式

Ascend C Kernel 直调，不走单算子 aclnn 两段式接口：

```text
aclblasCreate(&handle)
   └─ aclblasSetStream(handle, stream)        # handle 绑定 stream
        └─ aclblasChpr(handle, uplo, n, &alpha, dX, incx, dAp)
             ├─ Host: 校验 → alpha 定位/读取 → quick return → tiling
             └─ chpr_kernel<<<blocks, nullptr, handle->stream>>>(x, ap, tiling)   # 异步
        └─ aclrtSynchronizeStream(stream)      # 读回 Device 结果前必须同步
```

**单次 kernel 发射，无 workspace，无中间 Device 缓冲**（任务书 §3.4 内存不涉及，本实现实际额外 Device 内存开销为 0）。`ap` 原地更新。

## 3.2 需求总体设计

### 3.2.1 数学与数据表示

设 `x_i = (ar, ai)`、`x_j = (br, bi)`。按 §1.2.2.2 的 Netlib 结合顺序，先把实标量并入列项：

```text
axr = alpha * br
axi = alpha * bi                 # 即 TEMP = alpha*conj(x_j) = (axr, -axi)
```

则 `ap(i,j) += x_i * TEMP` 展开为：

```text
ap(i,j).real += ar*axr + ai*axi
ap(i,j).imag += ai*axr - ar*axi
```

**与 Netlib 逐项同构的验证**：Netlib 算 `x_i*TEMP`，其中 `TEMP = (alpha*br, -alpha*bi)`，
`(ar + i·ai)(axr - i·axi)` → 实部 `ar*axr + ai*axi`、虚部 `ai*axr - ar*axi`，与上式**逐项一致**（含乘加结合顺序）。这是本设计有意为之：运算图与 golden 同构，误差仅来自末位舍入，使精度风险从「算法误差」降为「ULP 级舍入」（见 §7.1）。

**对角元素（i = j，即 ar=br、ai=bi）**：

```text
real += ar*axr + ai*axi = alpha*(br² + bi²) = alpha*|x_j|²      # 与 Netlib REAL(x_j*TEMP) 同式
imag += ai*axr - ar*axi = alpha*(bi*br - br*bi) = 0             # 代数恒为 0
```

虚部的两项大小完全相等、符号相反，浮点下精确抵消，增量恒为 `0`。但**这不足以保证输出虚部为 0**：Netlib 取的是 `REAL(AP(diag))`，即输入旧虚部被**丢弃**。若输入对角虚部为非 0（CSV 的 Inf/NaN 填充用例会构造该情形），`+= 0` 会把旧值原样留下，而 golden 是 0。故实现必须**显式写 0**，而非依赖增量为 0：

```text
ap(j,j).real = ap(j,j).real + alpha*|x_j|²
ap(j,j).imag = 0.0f                            # 显式赋值，覆盖任意输入旧值（含 Inf/NaN）
```

arch22 实现正是这样做的（UPPER 路径在搬入时用 pad 把对角虚部位置置 0；LOWER 路径 `apLocal.SetValue(1U, 0.0f)`）。arch35 的 SIMT 标量模型下可直接赋值，比 arch22 更简洁。

**运算/访存强度**：每个 packed 元素 4 次乘 + 2 次加（6 flop），GM 访存 16 B（`ap` 读 8 B + 写 8 B，`x` 命中 UB/Cache）。强度 ≈ **0.375 flop/B**，远低于硬件乘加/带宽比，故本算子**确定为访存带宽受限（memory-bound）**。这一结论决定了优化方向：**一切以 `ap` 的流式访存效率为中心**，计算侧无需向量化指令堆叠。

### 3.2.2 工程使能与文件布局

仓内 `blas/CMakeLists.txt` 对架构目录做自动收集：

```cmake
foreach(arch_dir ${SOC_ARCH_DIRS})            # ascend950 → arch35
    file(GLOB_RECURSE arch_dir_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*/${arch_dir}/*.cpp)
    list(APPEND ARCH_SRC_FILES ${arch_dir_srcs})
endforeach()
```

因此**新增 `blas/hpr/arch35/*.cpp` 即被自动纳入编译，无需修改任何 CMakeLists**；`SOC_VERSION=ascend910b*` 时该目录自动不参与编译，与 arch22 实现互不干扰（符号不冲突）。

新增/改动文件清单（严格控制数量，避免任务书 §5「文件个数明显不符合正常数量」「多出无用代码/测试/编译文件」的反例）：

| 文件 | 性质 | 说明 |
| --- | --- | --- |
| `blas/hpr/arch35/chpr_host.cpp` | 新增 | Host 校验、tiling、下发 |
| `blas/hpr/arch35/chpr_kernel.cpp` | 新增 | SIMT kernel 与入口 `chpr_kernel_do` |
| `blas/hpr/arch35/chpr_tiling_data.h` | 新增 | `ChprTilingData` 与调优常量 |
| `blas/hpr/README.md` | 改动 | 产品支持表 950PR 改为「支持（arch35）」，补参数/约束/示例 |
| `test/hpr/chpr/arch35/chpr_test.cpp` | 新增 | GTest 用例体 |
| `test/hpr/chpr/arch35/chpr_npu_wrapper.h` | 新增 | NPU 调用封装 |
| `test/hpr/chpr/arch35/chpr_test.csv` | 新增 | 1200 条用例 |
| `test/hpr/chpr/chpr_param.h` | 改动 | 列名兼容 `x_fill`/`ap_fill`（见 §7.3.2） |

合计新增 6 个文件、改动 2 个文件，与 `blas/spr/arch35`（3 个实现文件）的体量一致。**不新增** kernel 头文件（arch35 范式中 kernel 入口在 host 侧就地声明，`sspr_host.cpp` 即如此，无 `sspr_kernel.h`），**不新增** 独立 CMakeLists。

## 3.3 Host 侧设计

### 3.3.1 参数校验流程

```text
aclblasChpr(handle, uplo, n, alpha, x, incx, ap)
  │
  ├─ handle == nullptr ?                    ──► HANDLE_IS_NULLPTR
  ├─ n < 0 ?                                ──► INVALID_VALUE
  ├─ n == 0 ?                               ──► SUCCESS            (不再校验其余参数)
  │
  ├─ ValidateChprParams(uplo, incx, alpha, x, ap)
  │     ├─ uplo ∉ {UPPER,LOWER} ?           ──► INVALID_VALUE
  │     ├─ incx == 0 || incx == INT_MIN ?   ──► INVALID_VALUE
  │     └─ alpha/x/ap == nullptr ?          ──► INVALID_VALUE
  │
  ├─ CheckPtrLocation(alpha) → Host ? *alpha : ReadHostScalarFromDevice(...)   # issue #510
  ├─ alphaVal == 0.0f ?                     ──► SUCCESS            (ap 不变)
  │
  ├─ GetAivCoreCount() == 0 ?               ──► EXECUTION_FAILED
  ├─ tiling ← CalChprTilingData(...)
  └─ chpr_kernel_do(x, ap, tiling, blocks, handle->stream)  ──► SUCCESS （异步）
```

`incx == INT_MIN` 单独排除的原因：负步长实现需要 `-incx`，而 `-INT_MIN` 在 32 位补码下溢出未定义。该约束与同任务族的 `chbmv`/`chpmv` 不同（后者允许 `INT_MIN`），任务书 §2.4 对本算子明确要求拒绝，CSV 亦提供 `incx_intmin`（期望 `INVALID_VALUE`）与 `n0_incx_intmin`（期望 `SUCCESS`）两条用例交叉验证校验次序。

> 实现次序说明：`sspr_host.cpp` 的 `ValidateSsprParams` 先判 `alpha != nullptr` 再判 `uplo`。由于第 4~6 项全部返回同一状态码 `INVALID_VALUE`，其内部先后顺序**对返回值不可观测**（仅影响日志文本）。本实现按任务书 §2.5 列出的顺序（uplo → incx → 指针）书写，以便与任务书逐行对照。

### 3.3.2 分核与线程策略

本算子访存受限且**无归约、无跨核依赖**（每个 packed 元素恰被一个 (i,j) 对更新，列与列之间 `ap` 区间互不重叠），因此分核目标只有两个：**负载均衡**与**合并访存**。

总工作量 `P = n(n+1)/2` 个 packed 元素。关键观察：

- **按列平均分配会严重倾斜**。UPPER 第 `j` 列有 `j+1` 个元素，列长线性递增。若按**连续列块**分给 `B` 个核（`blas/spr/arch35` 的 UB 路径即如此），末块工作量 / 平均工作量 → `2 - 1/B ≈ 2`，等于把有效吞吐砍掉近一半。
- **按平坦元素数均分则天然均衡**：给每个核分配等量的 packed 元素。

本实现采用**平坦均分（flat partition）**：核 `b` 负责平坦区间 `[P·b/B, P·(b+1)/B)`，各核元素数相差至多 1，**与 uplo 无关、无倾斜**。

| 量 | 取值 | 说明 |
| --- | --- | --- |
| `B`（blocks） | `min(CeilDiv(P, MIN_ELEMS_PER_BLOCK), aivCoreNum)` | 小规模时自动收敛到少量核，避免启核开销超过收益 |
| `numThreads` | `min(CeilAlign(·, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)` | 即 128 对齐、上限 2048，与 `sspr_host.cpp` 一致 |
| `MIN_ELEMS_PER_BLOCK` | 初值 `SIMT_MIN_THREAD_NUM`（128），待 profiling 校准 | 保证每核至少有一个满线程束的工作量 |

区间起始列的还原在 Device 侧一次性完成（每核一次，非每元素），用闭式解加整数校正（§3.4.2），避免把 `B` 长度的数组塞进按值传递的 tiling 结构。

### 3.3.3 Tiling 数据

```cpp
struct ChprTilingData {
    uint32_t numThreads;   // 每 block 线程数（128 对齐，≤2048）
    uint32_t numBlocks;    // 实际使用的 block 数 B
    uint32_t n;            // 矩阵阶数
    uint32_t uplo;         // ACLBLAS_UPPER(121) / ACLBLAS_LOWER(122)
    float    alpha;        // 已从指针解引用（Host 或 D2H 读回）的实标量
    int64_t  incx;         // x 步长，可为负
    uint64_t packedTotal;  // P = n(n+1)/2，uint64 防溢出
};
```

字段与 `SsprTilingData` 同构，仅多 `numBlocks` 与 `packedTotal` 两项（平坦均分所需），不含任何冗余字段。

### 3.3.4 TilingKey / 路由规划策略

arch35 SIMT 模型下，路由通过**模板实参 + kernel 内分支**实现，不使用 TilingKey 机制（与 `sspr_kernel.cpp` 一致：`asc_vf_call<SsprGmImpl<true,true>>` 形式按 uplo/incx 符号实例化）。

编译期分支（模板参数，消除内循环中的条件判断）：

| 模板参数 | 取值 | 作用 |
| --- | --- | --- |
| `UPLO_IS_UPPER` | `true` / `false` | 列基址与行范围公式 |
| `INCX_UNIT` | `true` / `false` | `true` 时 `x` 下标退化为 `i`，可做连续向量访问 |
| `INCX_POSITIVE` | `true` / `false` | 负步长的反向映射 `(n-1-i)·|incx|` |

运行期分支（kernel 内，每核一次）：是否启用 UB-x 缓存（§3.4.3）。模板组合数为 `2×2×2 = 8`（`INCX_UNIT=true` 时 `INCX_POSITIVE` 必为 `true`，实际 6 个实例），代码膨胀可控。

## 3.4 Kernel 侧设计

### 3.4.1 路由总览

```text
chpr_kernel(x, ap, tiling)                   # KERNEL_TYPE_AIV_ONLY
  │
  ├─ blockIdx → 平坦区间 [pBegin, pEnd)  = [P·b/B, P·(b+1)/B)
  ├─ pBegin == pEnd ? return                 # 空核直接退出
  ├─ 还原起始列 jStart（闭式解 + 整数校正，§3.4.2）
  │
  ├─ incx==1 且 n ≤ UB_X_COMPLEX ?
  │      ├─ 是 ──► Path B：x 全量缓存入 UB，再走行向合并主循环
  │      └─ 否 ──► Path A：x 自 GM 读（含任意/负步长）
  │
  └─ 主循环（两条路径共用）：逐列推进，块内线程覆盖该列行区间
```

### 3.4.2 主路径：平坦均分 + 行向合并访问

**平坦区间 → (列, 行) 的还原**。核 `b` 的起始平坦下标 `pBegin`，需求出其所在列 `j` 与行 `i`：

- UPPER：`p = j(j+1)/2 + i`，`0 ≤ i ≤ j` ⟹ `j = ⌊(√(8p+1) − 1)/2⌋`，`i = p − j(j+1)/2`
- LOWER：`p = j(2n−j+1)/2 + (i−j)`，`j ≤ i < n` ⟹ `j = ⌊(2n+1 − √((2n+1)² − 8p))/2⌋`，`i = j + (p − colBase(j))`

**浮点精度陷阱与处理**：`n=4096` 时 `8p+1 ≈ 6.7e7`，已超 FLOAT32 的 24 位有效位（≈1.6e7），单精度 `sqrt` 会给出错位的 `j`。因此：用 `double` 求 `sqrt`，再做**整数校正**收敛到精确解：

```text
while (colBase(j+1) <= p) ++j;        # 至多执行 1~2 次
while (colBase(j)    >  p) --j;
```

校正后 `j` 一定精确，且与 `sqrt` 的舍入方向无关。该还原**每核仅执行一次**（非每元素），开销可忽略。

> 备选：由 Host 预先二分算出各核起始列，经 tiling 传入。但这需要在按值传递的 tiling 结构中放入长度为 `B` 的数组，与仓内 tiling 结构的简洁约定不符，故选择 Device 侧闭式解 + 校正。

**块内线程映射（合并访存的关键）**。确定起始列后，块沿列方向推进；对每一列 `j`，本块需覆盖的行区间是「该列完整行范围」与「本块平坦区间」的交集：

```text
for (j = jStart; p < pEnd; ++j):
    rowBegin = (UPPER ? 0 : j)                       # 该列完整行范围
    rowEnd   = (UPPER ? j+1 : n)
    [rb, re) = 本块平坦区间在该列内的行子区间            # 首/末列可能是部分列
    (axr, axi) = (alpha*x_j.real, alpha*x_j.imag)      # 列级常量，提到行循环外
    colBase  = (UPPER ? j*(j+1)/2 : j*(2n-j+1)/2)
    for (i = rb + threadIdx.x; i < re; i += blockDim.x):     # 线程沿行方向展开
        (ar, ai) = x_i
        q = colBase + (UPPER ? i : i - j)
        if (i == j):                                   # 对角：显式置虚部 0
            ap[q].real = ap[q].real + ar*axr + ai*axi
            ap[q].imag = 0.0f
        else:
            ap[q].real = ap[q].real + ar*axr + ai*axi
            ap[q].imag = ap[q].imag + ai*axr - ar*axi
```

这个映射同时满足三个性质，这是选它而非「每线程一整列」的原因：

| 性质 | 原因 |
| --- | --- |
| **`ap` 合并访问** | 同一 warp 内相邻线程的 `i` 连续，`q = colBase + i(−j)` 连续 ⟹ 读改写落在连续 cache line，这是 memory-bound 算子的首要指标 |
| **`x_i` 连续访问** | 线程 `i` 连续 ⟹ `incx=1` 时 `x_i` 连续；`incx≠1` 时为固定 stride |
| **`x_j` 广播** | 列内 `j` 不变，整块共享一个标量对 `(axr, axi)`，提到行循环外，每列只读一次 |
| **负载均衡** | 各核平坦元素数相等（§3.3.2） |
| **无原子、无屏障** | 每个 `q` 恰被一个线程写；列间、核间无依赖 |

对比「每线程负责一整列」（`sspr_kernel.cpp` 的 GM 路径范式）：那种映射下同一 warp 的线程写入 `n` 个不同列的 `ap`，地址彼此相距 `O(n)`，**不合并**；本算子 `ap` 元素 8 B，不合并会显著放大实际访存事务数。对于纯流式、带宽受限的本算子，行向映射是更合适的选择。

### 3.4.3 UB-x 缓存（Path B）

`x_i` 的逻辑读取次数等于 `P`——与 `ap` 的元素数相同。若全部走 GM，`x` 侧流量与 `ap` 侧同量级（尽管实际会大量命中 DCache）。把 `x` 整体驻留 UB 可使其与 `ap` 的流式访问**互不争用 cache**：

```text
__ubuf__ float xUb[2 * UB_X_COMPLEX];              # 按 (real, imag) 交错存放
for (k = threadIdx.x; k < 2*n; k += blockDim.x)    # 块内协同搬入，incx==1 时连续
    xUb[k] = reinterpret_cast<__gm__ float*>(x)[k];
asc_syncthreads();
```

容量与阈值：

| n | `x` 占 UB | 说明 |
| --- | --- | --- |
| 4096（性能用例上限） | 32 KB | 全部性能用例均可驻留 |
| 8192 | 64 KB | `UB_X_COMPLEX` 初值取 8192 |

`blas/spr/arch35/sspr_tiling_data.h` 记录了一条实测经验：`UB_X_FLOATS` 取 32 KB 优于 48/64 KB，因为「32KB leaves 216KB DCache for AP streaming」——UB 占用过大会挤占 `ap` 流式访问可用的 DCache。本实现沿用该结论的量级（初值 32 KB 对应 n ≤ 4096 的复数，正好覆盖全部性能用例），并把 `UB_X_COMPLEX` 与启用阈值 `UB_THRESHOLD` 作为**待 profiling 校准的常量**，在源码注释中记录校准依据（与 `sspr_tiling_data.h` 的注释风格一致）。

`n` 超阈值或 `incx ≠ 1` 时走 Path A（`x` 自 GM 读）。两条路径共用 §3.4.2 的主循环，仅 `x` 的取数方式不同，由模板参数在编译期选定，无运行期分支开销。

### 3.4.4 任意步长与负步长

逻辑下标 `i` → 物理下标：

```text
INCX_UNIT     : i
INCX_POSITIVE : i * incx
否则（incx<0） : (n - 1 - i) * (-incx)
```

全部用 `int64_t` 中间量计算（`incx` 为 `int64_t`，`n` 最大 `uint32`，乘积可超 32 位）。`incx == INT_MIN` 已在 Host 侧拒绝，故 `-incx` 安全。

### 3.4.5 对角虚部置 0

每列恰有一个对角元素，平坦下标为：

- UPPER：`colBase(j) + j`（列内**末**元素）
- LOWER：`colBase(j)`（列内**首**元素）

在 §3.4.2 的行循环中以 `i == j` 判定命中。实现要点（对应 §3.2.1 的分析）：

1. **显式赋 `0.0f`**，不写成 `+= 0`。输入对角虚部为 Inf/NaN 时，golden（`cblas_chpr`）输出 0，`+= 0` 会得到 Inf/NaN 而判错。
2. **实部仍走通用公式** `ar*axr + ai*axi`，在 `i==j` 时代数等于 `alpha*|x_j|²`，且与 Netlib 的 `REAL(x_j*TEMP)` 表达式同构（§3.2.1），逐位一致性最好。
3. `alpha == 0` 时 Host 已 quick return，**不进入 kernel**，对角虚部保持输入旧值——与 cblas 在 `alpha==0` 时直接 `RETURN` 的行为一致（§2.3.4）。

该分支位于内循环，但 `i == j` 对整个 warp 几乎总是 false，分支预测友好；且本算子访存受限，该标量判断不在关键路径上。

### 3.4.6 小规模低延迟路径

性能用例含 `n = 1, 2, 4, 8, 16` 等极小规模，其 GPU 标杆仅 3.0~4.1 us，折算预算 7.6~10.1 us（§7.2），**完全由固定开销主导**，与带宽无关。相应设计：

- `B = min(CeilDiv(P, MIN_ELEMS_PER_BLOCK), aivCoreNum)` 使 `n` 小时 `B = 1`，只启一个 block；
- Host 侧 tiling 计算仅为若干整数运算，无查询类 API、无内存分配；
- 全程**单次 kernel 发射**，无 workspace 申请、无额外 stream 同步、无 D2H（`alpha` 为 Host 指针时不触发 §2.3.4 的 D2H 读回）；
- `n = 0` 不下发 kernel。

### 3.4.7 实现流程图

```text
┌─────────────────────────── Host: aclblasChpr ───────────────────────────┐
│ 校验（固定序）→ alpha 定位/读取 → quick return(n=0 / alpha=0)            │
│ tiling: P=n(n+1)/2, B=min(CeilDiv(P,MIN_ELEMS),aiv), numThreads=128 对齐 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ chpr_kernel<<<B, nullptr, stream>>>
┌───────────────────────────────▼─────────────────────────────────────────┐
│ Device: block b                                                          │
│  [pBegin,pEnd) = [P·b/B, P·(b+1)/B)        ← 平坦均分，零倾斜            │
│  jStart ← 闭式解(double sqrt) + 整数校正    ← 每核一次                   │
│                                                                          │
│  if (incx==1 && n ≤ UB_X_COMPLEX):  x → UB（块内协同搬入）+ syncthreads  │
│                                                                          │
│  for j = jStart; p < pEnd; ++j:                                          │
│      (axr,axi) = alpha * x_j                 ← 列级常量，广播            │
│      colBase   = UPPER ? j(j+1)/2 : j(2n-j+1)/2                          │
│      for i = rb + threadIdx.x; i < re; i += blockDim.x:   ← 行向合并     │
│          q = colBase + (UPPER ? i : i-j)                                 │
│          ap[q].real += ar*axr + ai*axi                                   │
│          ap[q].imag  = (i==j) ? 0.0f : ap[q].imag + ai*axr - ar*axi      │
└──────────────────────────────────────────────────────────────────────────┘
          无原子操作 · 无跨核屏障 · 无 workspace · ap 原地更新
```

## 3.5 与标杆实现的差异及原因

| 项 | Netlib / cuBLAS | 本实现 | 原因 |
| --- | --- | --- | --- |
| 遍历顺序 | 单线程逐列、列内逐行 | 平坦均分到核，列内行向多线程并行 | 发挥 AIV 多核与 SIMT 并行；因无归约，顺序改变**不影响每个元素的运算式** |
| 每元素运算式 | `x_i * (alpha*conj(x_j))` | 同式同序（§3.2.1 已逐项验证） | 保持与 golden 的运算图同构，误差压到 ULP 级 |
| 对角处理 | `REAL(AP(diag)) + REAL(x_j*TEMP)` | 实部同式；虚部**显式写 0** | 等价复刻「丢弃输入虚部」语义，含 Inf/NaN 输入 |
| `alpha` 指针 | cuBLAS 支持 HOST/DEVICE pointer mode | 支持 Host/Device（`CheckPtrLocation`） | 对齐 cuBLAS 能力与仓内 arch35 约定（issue #510） |
| 累加顺序敏感性 | — | 无 | 秩-1 更新每个输出元素只被写一次，**不存在累加次序问题**，结果与并行度无关（确定性，见 §3.6） |

## 3.6 正确性与并发保证

| 风险 | 保证机制 |
| --- | --- |
| 写冲突 | packed 布局下每个下标 `q` 唯一对应一个 `(i,j)`，由唯一线程写入 ⟹ **无需原子操作** |
| 跨核依赖 | 平坦区间互不重叠，列间无依赖 ⟹ **无需跨核屏障** |
| UB 搬入竞争 | Path B 的协同搬入后加 `asc_syncthreads()`，确保全块可见后再进入计算 |
| 读写同一缓冲 | `ap` 原地读改写，但同一地址仅一个线程读写，无 RAW/WAW 竞争 |
| 结果确定性 | 每个输出元素单次写入、运算式固定 ⟹ **同输入逐位可复现**，与 `B`/`numThreads` 取值无关（这也使 §7.3 的多次复跑可用于判定稳定性） |
| 索引溢出 | 平坦下标、列基址一律 `uint64_t`；物理下标用 `int64_t` |
| `-incx` 溢出 | Host 侧拒绝 `incx == INT_MIN` |
| FMA 收缩导致与 golden 偏离 | `ar*axr + ai*axi` 可能被编译器收缩为 FMA，改变舍入。仓内已有先例按需固定（`CMakeLists.txt` 对 `blas/gerc/arch35/cgerc_kernel.cpp` 设 `-ffp-contract=off`）。本实现默认不强制关闭（FMA 通常更准），若自测发现个别用例 `max_abs_error` 逼近门限，再按该先例固定编译选项，并在自测报告中记录 |

## 3.7 备选方案与取舍

| 方案 | 负载均衡 | `ap` 合并访存 | 结论 |
| --- | --- | --- | --- |
| 连续列块分核（`sspr` arch35 UB 路径范式） | 差（末块 ≈2× 平均） | 好 | **不采用**：倾斜直接折损近半吞吐 |
| 列 grid-stride（`sspr` arch35 GM 路径范式） | 好（各核采样到均匀列长） | **差**（warp 内跨 `n` 个列，地址相距 `O(n)`） | **不采用**：memory-bound 算子下合并访存优先 |
| 每元素独立求 `(i,j)`（平坦 grid-stride） | 最优 | 好 | **不采用**：每元素一次 `sqrt`，且 FLOAT32 精度不足需 `double`，开销进入内循环 |
| **平坦均分 + 列内行向映射（本设计）** | **最优** | **好** | **采用**：`sqrt` 每核一次，兼得均衡与合并 |

---

# 4. 支持硬件（required）

| 支持的芯片版本 | 是否勾选 | 架构目录 |
| --- | --- | --- |
| Ascend 950PR | √ | `arch35` |
| Ascend 950DT | —（本任务不验收，代码路径同 arch35，天然共用） | `arch35` |
| Atlas A3 训练/推理系列 | √（**已有**，本任务不改动） | `arch22` |
| Atlas A2 训练/推理系列 | √（**已有**，本任务不改动） | `arch22` |

`blas/hpr/README.md` 现标注「Ascend 950PR / Ascend 950DT：不支持」，本任务交付时同步改为「支持（arch35）」。接口声明位于 `include/cann_ops_blas.h`，各产品线共用，**无产品私有平行接口**。

---

# 5. 算子约束限制（required）

| 约束项 | 内容 |
| --- | --- |
| 数据类型 | `x`/`ap` 为 COMPLEX64（`aclblasComplex`，实部/虚部 FLOAT32）；`alpha` 为 FLOAT32 **实数** |
| 维度 | `n ≥ 0`；`ap` 长度 `n(n+1)/2`；`x` 物理长度 ≥ `1+(n-1)·|incx|`。**调用方须保证 `ap` 实际分配不小于 `n(n+1)/2`**（packed 无前导维参数，实现无法校验长度） |
| 步长 | `incx ≠ 0` 且 `incx ≠ INT_MIN`，支持负步长（Netlib 反向遍历语义） |
| 三角引用 | 仅 `uplo` 指定三角被引用/更新；另一半不存于 `ap`，由共轭对称隐含，**不引用、不更新** |
| 对角 | 虚部假定为 0；更新后**置 0**（`alpha=0` quick return 时不改写） |
| 非连续 Tensor | 不涉及（步长语义仅由 `incx` 描述） |
| broadcast | 不涉及（`x` 与 `ap` 为独立操作数） |
| dynamic shape | 不要求，`n`/`incx` 为运行时入参 |
| 原地与视图 | `ap` 原地更新，不返回视图 |
| 确定性 | 不要求；但本实现**天然确定性**（§3.6） |
| 空 Tensor / 0 维 | `n=0` 合法 no-op 且不再校验其余参数；`n>0 且 alpha=0` 先完成校验再 no-op |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须 `aclrtSynchronizeStream` |
| 额外内存 | 无 workspace，无中间 Device 缓冲（任务书 §3.4 内存不涉及） |

---

# 6. 特性交叉分析（required）

| 特性 | 是否涉及 | 说明 |
| --- | --- | --- |
| 确定性计算 | 涉及（天然满足） | 每输出元素单次写入，无累加次序问题 |
| 空 Tensor | 涉及 | `n=0` quick return |
| 原地（in-place） | 涉及 | `ap` 原地读改写，唯一线程归属，无竞争 |
| 负步长 | 涉及 | Netlib 反向遍历语义 |
| Inf/NaN 传播 | 涉及 | 非对角按 IEEE 自然传播；**对角虚部强制为 0**（§3.4.5），与 cblas 一致 |
| dynamic shape | 不涉及 | `n` 为运行时入参，无编译期 shape |
| broadcast | 不涉及 | 无广播语义 |
| 混合精度 | 不涉及 | 全程 FLOAT32 分量，无类型提升/降级 |
| 多卡/分布式 | 不涉及 | 单 Device 单 stream |
| workspace | 不涉及 | 无额外 Device 内存 |
| 量化 | 不涉及 | — |

---

# 7. 可维可测分析

## 7.1 精度标准

**比对口径**（任务书 §3.2）：golden 由 cblas（Netlib `chpr`）单标杆生成，对输出 `ap` 的 `n(n+1)/2` 个复数元素**全量**验证，**实部、虚部分别**按 FLOAT32 标准判定。

| 项 | 取值 |
| --- | --- |
| rtol | `2⁻¹³ ≈ 1.22e-4` |
| atol | `2⁻¹³ ≈ 1.22e-4` |
| 逐元素通过 | `|actual − golden| ≤ atol + rtol × |golden|` |
| required_matched_ratio | `≥ 0.99` |
| max_abs_error_limit | `≤ 1e-2` 或 `32 × ULP` |

**Inf/NaN 判定**：不适用上述判定式。golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 另一方 ±Inf，视为等效通过；一方有限、另一方非有限则不通过。非有限元素不计入 `matched_ratio` 与 `max_abs_error` 统计，单独记录总数与通过数。

**比对前置口径**：golden 与实现输出**均须满足对角虚部为 0**（`cblas_chpr` 本身即置 0；本实现由 §3.4.5 保证）。

**精度可达性分析**：本算子每个输出元素仅被写一次，**不存在多项累加**，误差来源仅为 4 次乘 + 2 次加的末位舍入；且 §3.2.1 已逐项验证本实现的运算式与结合顺序**与 Netlib 同构**。因此预期逐元素误差在 **1~2 ULP** 量级，相对 `rtol = 1.22e-4` 有数个数量级余量。真正需要在自测中重点确认的不是「精度够不够」，而是三类**语义**正确性：

1. 对角虚部在 Inf/NaN 输入下是否严格为 0（CSV 的 `TC_FL` 填充用例）；
2. 负步长与非单位步长的下标映射（`TC_INC` 用例）；
3. `uplo` 两种 packed 索引公式的边界（`n=1`、非对齐 `n`：33/65/127/1025）。

> 实测结果（逐条 PASS/FAIL、`matched_ratio`、`max_abs_error`、非有限元素统计、实部/虚部截图）：**待自测回填**。

**阈值列说明**：CSV 的 `mere_threshold = 2⁻¹³`、`mare_multiplier = 10.0` 为 ops-blas 测试工程的 MERE/MARE 策略配置，**不参与**生态精度标准比对；标准比对按 dtype 固定阈值执行，不随用例列变化。两套口径均需在自测报告中给出。

## 7.2 性能标准与预算分析

**判据**（任务书 §3.3）：平均单次耗时 ≤ GPU 标杆的 **2.5 倍**（即达到标杆 **0.4 倍**以上性能）。标杆为 GPU H100（驱动 550.54、CUDA 12.4），`gpu_baseline.csv` 共 200 条。

**采样方式**：`msprof op` 模式采集单算子 task duration（us）。`msprof op` 自带 5 次 warmup（不计入样本），warmup 后连续执行 10 次作为 10 个有效样本，取算术平均。仅算子返回 `SUCCESS` 方计入有效样本，异常样本不计入并须在自测报告中说明。

```bash
msprof op --application="./build/test/hpr/chpr/chpr_test --gtest_filter=*TC_PF_001*" --output=./prof_chpr
# 在 OPPROF_*/OpBasicInfo.csv 中按 kernel 名统计 Task Duration(us)
```

> `--gtest_filter` 须收敛到**单条**用例，否则多条用例的 kernel 会混叠，无法按条归因。

**预算分析**（由 `gpu_baseline.csv` 推导；`ap` 流量 = `n(n+1)/2 × 8 B × 2`，读 + 写）：

| 用例 | uplo | n | GPU (us) | 预算 (us) | `ap` (MB) | 流量 (MB) | **所需带宽 (GB/s)** | GPU 实际带宽 (GB/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chpr-base-000 | UPPER | 512 | 11.909 | 29.77 | 1.05 | 2.10 | 70.6 | 176.4 |
| chpr-base-001 | LOWER | 1024 | 30.999 | 77.50 | 4.20 | 8.40 | 108.3 | 270.9 |
| chpr-base-002 | UPPER | 2048 | 96.491 | 241.23 | 16.79 | 33.57 | 139.2 | 347.9 |
| chpr-base-003 | LOWER | 4096 | 570.764 | 1426.91 | 67.13 | 134.25 | 94.1 | 235.2 |
| chpr-perf-089（最紧） | UPPER | 2562 | 143.981 | 359.95 | 26.27 | 52.53 | **145.9** | 364.8 |
| chpr-perf-004（最小） | UPPER | 1 | 3.046 | **7.62** | ~0 | ~0 | — | — |

两点结论直接决定优化重心：

1. **大规模（带宽受限）**：全部 200 条用例中，所需持续带宽的**最大值仅约 146 GB/s**（n≈2560）。该值远低于 Ascend 950PR 的 HBM 能力，余量充足。同时注意 H100 在本算子上的实测带宽也仅 176~365 GB/s，远未达其峰值——说明 `cublasChpr` 本身并非带宽最优实现，标杆并不苛刻。**只要 §3.4.2 的合并访存与 §3.3.2 的零倾斜分核落地，大规模用例应有可观余量。**
2. **小规模（延迟受限）**：`n=1` 的预算仅 **7.62 us**，`n ≤ 16` 的预算在 7.6~10.1 us。这些用例与带宽无关，**完全由 kernel 下发 + task 固定开销决定**，是本算子性能上**真正的风险点**。§3.4.6 的低延迟路径（单 block、无 workspace、无额外同步、Host 侧零查询）即针对此设计。

> 实测数据（200 条逐条 Avg time、倍率、余量、msprof 截图）：**待自测回填**。若小规模用例余量不足，备选手段为：进一步下调 `B` 使极小 `n` 仅启 1 block 且线程数取 `SIMT_MIN_THREAD_NUM`、精简 Host 路径上的日志调用（`OP_LOGI`/`OP_LOGD` 在 Release 下的实际开销需实测确认）。

## 7.3 测试覆盖

### 7.3.1 用例构成

随任务提供 `chpr_test.csv` 共 **1200 条**（`gen_csv.py` 固定种子 `20261028` 可复现，同种子两次运行输出逐字节一致）：

| 类别 | 前缀 | 条数 | 覆盖内容 |
| --- | --- | --- | --- |
| L0 基础 | `TC_L0` | 4 | uplo × 小尺寸（4, 8） |
| L1 尺寸 | `TC_SQ` | 46 | 23 种尺寸（1→2048，含奇数/边界/非对齐 33/65/127/1025）× uplo |
| L2 标量 | `TC_AB` | 16 | 8 组 alpha（1/−1/0/0.5/2/1000/0.25/−0.5）× uplo |
| L4 步长 | `TC_INC` | 12 | `incx ∈ ±1/±2/±3` × uplo |
| L5 填充 | `TC_FL` | 12 | 均匀随机 / 全零 / 交替 / 极端值 / **Inf** / **NaN**（x、ap 分别） |
| L5b 覆盖 | `TC_CV` | 16 | 中等尺寸（10~400）× uplo |
| L6 边界 | `TC_ED` | 25 | `n=0` quick return（含叠加非法 uplo / `incx=0` / `INT_MIN` / 空指针）、`alpha=0` no-op（含叠加空指针与零步长）、空指针、非法枚举、负维度、零步长与 `INT_MIN` |
| EX 扩展 | `TC_EX` | 869 | 尺寸池 × uplo × alpha × 步长确定性采样 |
| PF 性能 | `TC_PF` | 200 | 4 条任务书典型 case + 小尺寸 + 阶数对数扫描至 4096（均为 `incx=1` 连续访存） |

精度用例 1000 条（自动排除 `TC_PF_` 前缀），性能用例 200 条。负向用例的期望状态码以任务书 §2.4/§2.5 为准。

**`handle = nullptr` 用例无法由 CSV 列表达**（列序按 §2.3 去掉 handle），由测试工程在参数装配层固定传入有效句柄；该项以接口单测单独覆盖，期望 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

### 7.3.2 测试工程改造点（必须处理）

| 问题 | 现状 | 处理 |
| --- | --- | --- |
| **列名不匹配** | 仓内 `test/hpr/chpr/chpr_param.h` 的 `ReadMap` 用裸列名 `"x"` / `"ap"`；新 CSV 为 `x_fill` / `ap_fill` | 将 `ReadMap` 键同步为 `x_fill`/`ap_fill`，并保留裸列名回退以兼容 `arch22/chpr_test.csv`（否则 arch22 既有用例会全部读到默认填充而静默失效） |
| 输入分布 | CSV 当前为均匀分布 `RANDOM_NORM_5_5`（[−5,5]） | 任务书 §3.5 要求均匀/正态各 50%；`gen_csv.py` 已预留 `--dist mixed` 开关，需测试工程侧支持该填充模式 |
| golden | `test/hpr/chpr/chpr_golden.h` 已有 | 复用；确认其 `alpha=0` 与对角虚部口径与 §2.3.4/§3.4.5 一致 |

### 7.3.3 复现步骤

```bash
# 编译（SOC=ascend950 → 自动纳入 blas/hpr/arch35）
bash build.sh --soc=ascend950 --ops=hpr

# 精度（自动排除 TC_PF_ 性能用例）
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./chpr_test.csv

# 性能（TC_PF 用例 + gpu_baseline.csv 关联比对）
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600

# 单条 kernel 级耗时
msprof op --application="./build/test/hpr/chpr/chpr_test --gtest_filter=*TC_PF_001*" --output=./prof_chpr
```

`test/hpr/chpr/arch35/README.md`（或测试目录 README）需写明上述步骤，保证验收人可复现。

## 7.4 兼容性与维护性

| 项 | 说明 |
| --- | --- |
| API 兼容 | **不改动** `include/cann_ops_blas.h`，复用既有声明；无产品私有平行 API |
| arch22 兼容 | `arch35` 与 `arch22` 按 `SOC_ARCH_DIRS` 互斥编译，符号不冲突；本任务不改动 arch22 代码 |
| 跨 SOC 编译 | 需验证 `--soc=ascend910b*`（仅 arch22 参与）与 `--soc=ascend950`（仅 arch35 参与）均编译通过 |
| 新算子 | 对 950PR 为新增实现，无历史行为需兼容 |
| 可维护性 | 文件数与 `blas/spr/arch35` 对齐（3 个实现文件）；调优常量集中在 `chpr_tiling_data.h` 并在注释中记录校准依据（沿用 `sspr_tiling_data.h` 风格） |
| 代码规范 | 遵循仓内 `.clang-format` 与 `CONTRIBUTING.md`；`.pre-commit-config.yaml` 本地校验 |

## 7.5 风险与应对

| 风险 | 等级 | 应对 |
| --- | --- | --- |
| **小规模用例（n ≤ 16）延迟预算仅 7.6~10 us** | **高** | §3.4.6 低延迟路径：单 block、无 workspace、无额外同步、Host 零查询；若余量不足则精简 Host 日志、下调线程数（§7.2） |
| 对角虚部在 Inf/NaN 输入下未严格置 0 | 中 | §3.4.5 显式赋 0（非 `+= 0`）；`TC_FL` 用例专项验证 |
| 平坦区间起始列还原因 FLOAT32 `sqrt` 精度不足而错位 | 中 | 用 `double` 求解 + 整数校正循环（§3.4.2），结果与舍入方向无关 |
| 测试工程列名 `x_fill`/`ap_fill` 不匹配导致**静默**读到默认填充 | 中 | §7.3.2 同步 `ReadMap` 键并保留回退；改造后须确认 arch22 既有用例仍通过 |
| UB 容量/阈值取值不当挤占 `ap` 流式 DCache | 中 | 沿用 `sspr` 的 32 KB 量级结论为初值，`UB_X_COMPLEX`/`UB_THRESHOLD` 作为可调常量由 profiling 校准 |
| FMA 收缩使个别元素偏离 golden | 低 | §3.6：必要时按 `cgerc` 先例对 kernel 设 `-ffp-contract=off`，并在报告中记录 |
| `n` 很大时 packed 索引溢出 32 位 | 低 | 全程 `uint64_t`/`int64_t`（§2.3.3） |
| 交付文件冗余触发任务书 §5 反例 | 低 | §3.2.2 已固定文件清单：新增 6、改动 2 |

---

# 8. 交付与验收

| 序号 | 交付件 | 内容与状态 |
| --- | --- | --- |
| 1 | 算子设计文档 | 本文档。按 cann-competitions 仓规范提交至 `04_tasks/01_community-task-2026/tasklist/{任务编号}-aclblasChpr-950/shiyueqingchen/docs/design.md`，经评审后合入 |
| 2 | 自测用例及测试代码 | `test/hpr/chpr/arch35/`（含 `chpr_test.csv` 1200 条）+ README 复现步骤；精度 case 与性能 case 分别列出（§7.3） |
| 3 | 自测报告 | 按官方模板输出，含用例参数、精度比对结果及**实部/虚部分别**的截图、性能数据及截图 |
| 4 | 待验收代码地址 | 个人代码仓链接、分支、算子目录；需在个人仓邀请账号 `Ascend-CANN` 为开发者；`blas/hpr/README.md` 产品支持表标注 Ascend 950PR：支持 |

**PR 去向**（测试通过后）：实现 → `cann/ops-blas` 的 `blas/hpr/arch35/`；测试 → `test/hpr/chpr/arch35/`。设计文档 → `cann/cann-competitions`。

> 提交前待确认事项：
> - 提交目录中的**任务编号**需与社区任务 IT 系统中的认领记录一致（`cann-competitions` 仓当前 `tasklist/` 下尚无本任务目录，`docs/` 亦只到 `202608`，故编号无法从仓内推断，本文档暂留占位）；团队目录名用 `shiyueqingchen`；
> - `Ascend-CANN` 开发者邀请须在个人仓网页端由本人操作。

---

# 9. 参考资料

1. Netlib BLAS `chpr` 参考实现：https://www.netlib.org/blas/chpr.f
2. cuBLAS `cublasChpr`：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hpr
3. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
   - `include/cann_ops_blas.h:406-408`（`aclblasChpr` 声明）
   - `blas/hpr/arch22/`（复数 Hermitian packed 秩-1，A2/A3 实现）
   - `blas/spr/arch35/`（实数 packed 秩-1，arch35 SIMT 范式与调优注释）
   - `blas/CMakeLists.txt`（架构目录自动收集机制）
   - `CONTRIBUTING.md`（贡献指南）
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
6. Ascend C 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html
7. 设计文档模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
8. 社区任务流程及注意事项：https://gitcode.com/org/cann/discussions/39
