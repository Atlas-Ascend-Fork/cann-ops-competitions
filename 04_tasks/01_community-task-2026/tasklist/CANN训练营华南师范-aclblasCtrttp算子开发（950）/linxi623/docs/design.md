# aclblasCtrttp 算子设计文档（Atlas 950PR）

# 1. 需求背景（required）

## 1.1 需求来源

需求来自任务附件 `aclblasCtrttp_Atlas950PR_task_doc.md`，附件同时提供 1200 条 CSV 用例（`ctrttp_test.csv`）、用例生成脚本、H100 性能基线（`gpu_baseline.csv`）与精度/性能自验脚本。实现目标是在 ops-blas 开源仓 `blas/trttp/arch35/` 中补齐单精度复数三角压缩接口 `aclblasCtrttp`，与同目录实数版本 `aclblasStrttp` 同族共存。

接口语义对齐 cuBLAS `cublasCtrttp`，数学参考为 Netlib LAPACK 3.12.0 `ctrttp.f`。

## 1.2 背景介绍

Trttp（Triangular matrix, Regular storage To Triangular matrix, Packed format）属于 LAPACK 的存储格式转换例程：把列主序 `lda × n` 三角矩阵压缩成 `n(n+1)/2` 个元素的 packed 数组。它常用于把稠密三角矩阵交给只接受 packed 存储的例程（如 `ctptrs`、`ctpttr` 的逆过程），也是 LAPACK 兼容层的基础能力。

本算子只做数据搬运：不改变任何元素数值，不做算术运算，因此精度判据是**按位相等**（含 Inf/NaN 载荷），性能瓶颈完全在访存：需要把 `n(n+1)/2` 个 complex64（8 B）从源矩阵按列取出，写入 packed 数组的对应位置。

## 1.3 同族实现现状分析

ops-blas 基线提交 `9565ef4`（master）已有 `blas/trttp/arch35/strttp_{host,kernel}.cpp` 与 `test/trttp/strttp/`，提供实数单精度 `aclblasStrttp`。复数版本缺失，且实数版本有三处不能直接照搬：

| 差异 | 本设计处理 |
| --- | --- |
| 元素宽度 | complex64 为 8 B（实部/虚部各 float32），按 float32 对（2 个 float）搬运，偏移全部乘 2 |
| 校验顺序 | 实数版本在 `n == 0` 早返回之后才校验 `lda`，导致 `n = 0 && lda = 0` 返回 SUCCESS；本算子按任务书要求把 `lda >= max(1, n)` 前置，`n = 0 && lda = 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 负载均衡 | 实数版本按列数均分核，LOWER 时首列 n 个元素、末列 1 个元素，负载相差 n 倍；本算子按元素数均分 |
| 偏移位宽 | packed 偏移在 n = 8192 时达 33 558 528，实数版本用 32 位中间量；本算子全程 64 位 |

# 2. 需求分析（required）

## 2.1 需求描述

把列主序复数三角矩阵 A 的 uplo 指定三角压缩为 packed 存储：

```
UPPER: AP[j*(j+1)/2 + i]          = A(i,j)，0 <= i <= j < n
LOWER: AP[j*n - j*(j-1)/2 + (i-j)] = A(i,j)，0 <= j <= i < n
```

- 仅读取 uplo 指定三角；A 的另一三角与 `lda` padding 行不读。
- AP 写满前 `n(n+1)/2` 个元素，尾部（若调用方分配更大）不得改写。
- A 与 AP 不得重叠（调用方契约）。
- 异步执行：依赖 `aclblasSetStream` 绑定的 stream，读回前须同步。

## 2.2 公共接口

声明放入 `include/cann_ops_blas.h`，与其他产品线共用，不定义 950PR 私有平行接口：

```cpp
aclblasStatus_t aclblasCtrttp(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* A, const int lda,
    aclblasComplex* AP);
```

`aclblasComplex` 取自 `include/cann_ops_blas_common.h`（`float real; float imag;`，与 complex64 布局一致）。

## 2.3 参数、形状与内存契约

| 参数 | 方向 | 类型 | 约束 | 异常行为 |
| --- | --- | --- | --- | --- |
| handle | 输入 | `aclblasHandle_t` | 有效句柄（携带 stream） | `nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 输入 | `aclblasFillMode_t` | `ACLBLAS_UPPER`(121) / `ACLBLAS_LOWER`(122) | 其它值 → `ACLBLAS_STATUS_INVALID_VALUE` |
| n | 输入 | `const int` | `n >= 0` | `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE` |
| A | 输入 | `const aclblasComplex*` | 列主序 `lda × n`，仅读 uplo 三角 | `n > 0` 且 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | 输入 | `const int` | `lda >= max(1, n)` | 否则 → `ACLBLAS_STATUS_INVALID_VALUE` |
| AP | 输出 | `aclblasComplex*` | 长度 `n(n+1)/2`，与 A 不重叠 | `n > 0` 且 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |

## 2.4 功能与异常行为拆解

1. 两种 uplo 的 packed 起点不同，不能复用同一套偏移：UPPER 第 j 列从 `j(j+1)/2` 开始，LOWER 第 j 列从 `jn - j(j-1)/2` 开始。
2. `n = 0 && lda >= 1` 为合法 no-op：返回 SUCCESS，不触发 kernel，不访问 A/AP。
3. `n = 0 && lda = 0` 必须返回 `ACLBLAS_STATUS_INVALID_VALUE`（前导维校验先于早返回）。
4. 非法 uplo 返回 `ACLBLAS_STATUS_INVALID_VALUE`，不使用 `ACLBLAS_STATUS_INVALID_ENUM`（与 cuBLAS 13.5 一致）。
5. 纯搬运：AP 与 A 对应元素按位相等，NaN 载荷、Inf 符号均需保留。

## 2.5 校验顺序与边界裁定

固定顺序（与任务书 §2.5/§7.5 一致）：

```
handle 判空 -> n < 0 -> uplo 合法性 -> lda < max(1, n) -> n == 0 早返回 SUCCESS -> A 判空 -> AP 判空
```

该顺序被三条负向用例锁定：`neg_n_invalid_uplo`（n<0 先于 uplo）、`n0_lda0`（lda 先于 n==0 早返回）、`n0_null_a_ap`（n=0 时 A/AP 为空仍合法）。

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```
UPPER: for j in [0, n): for i in [0, j]:   AP[j*(j+1)/2 + i]           = A[i + j*lda]
LOWER: for j in [0, n): for i in [j, n):   AP[j*n - j*(j-1)/2 + (i-j)] = A[i + j*lda]
```

两式均为列主序：A 的第 j 列在内存中连续（步长 1），列间步长 `lda`；AP 中第 j 列的 packed 段同样连续，且相邻列的段在 AP 中首尾相接。

### 3.1.2 支持数据类型

| A | AP | 计算 |
| --- | --- | --- |
| COMPLEX64（`aclblasComplex`） | COMPLEX64 | 无算术，按位搬运 |

### 3.1.3 支持形状与复杂度

- `n ∈ [0, 8192]`（packed 长度上限 33 558 528 个元素，约 256 MiB）。
- `lda ∈ [max(1,n), 2*max(1,n)]`（任务用例覆盖 `max(1,n)`、`+1`、`+3`、`+8`、`2n`）。
- 时间复杂度 O(n²) 次元素搬运，空间复杂度 O(1) 额外设备内存（不使用 workspace）。

## 3.2 算子实现

### 3.2.1 总体方案

host 侧完成参数校验、tiling 计算与 kernel 直调（`ctrttp_kernel_do` → `ctrttp_kernel<<<useCoreNum, nullptr, stream>>>`），kernel 侧以 AIV 单核为粒度搬运列片段，不做任何数值运算。整个算子无 workspace、无临时 GM 缓冲。

### 3.2.2 Host 侧设计

`blas/trttp/arch35/ctrttp_host.cpp`：

1. `ValidateParams`：按 §2.5 顺序校验，返回 `aclblasStatus_t`。
2. `CalTilingData`：`useCoreNum = min(GetAivCoreCount(), n)`（n > 0），并预计算 `totalElems = n(n+1)/2`（64 位）写入 tiling，供 kernel 做负载均衡。
3. `ExecuteKernel`：把 A/AP 指针与 tiling 传给 `ctrttp_kernel_do`，block 数取 `useCoreNum`，stream 取自 handle。

tiling 结构（`ctrttp_tiling_data.h`）：

```cpp
struct CtrttpTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;        // 0 = LOWER, 1 = UPPER
    uint32_t useCoreNum;
    uint64_t totalElems;  // n*(n+1)/2
};
```

### 3.2.3 Kernel 侧设计

`blas/trttp/arch35/ctrttp_kernel.cpp`，单 AIV 类 `CtrttpAIV`：

1. **按元素数分核**。第 b 个核负责 packed 元素区间 `[total*b/cores, total*(b+1)/cores)`，用前缀工作量闭式解反查列号：
   - UPPER：`prefix(j) = j(j+1)/2`
   - LOWER：`prefix(j) = j*n - j(j-1)/2`
   二分查找得到起止列，避免实数版本"按列数均分"在 LOWER 下首列 n 个元素、末列 1 个元素造成的 n 倍负载差。
2. **列片段搬运**。每列按 `(srcBase, dstBase, lenFloats)` 切成不超过 32 KiB（8192 个 float）的 chunk，`DataCopyPad` GM→UB→GM；complex64 以 2 个 float 为单位，因此 `lenFloats = 2 * 列长`，`srcBase/dstBase` 均为 2 倍元素偏移。
3. **7 级流水**。arch35（DAV_3510）UB 预算 248 KiB，先按仓内 arch35 kernel 惯例预留 20 KiB 给 TPipe 元数据，余 233472 B 均分为 7 块 32 KiB 缓冲（7 × 32 KiB = 224 KiB；8 块 256 KiB 会超预算 8192 B）。7 组 `MTE2_MTE3`/`MTE3_MTE2` 事件配 `LoadChunk(slot+k)` 与 `StoreChunk(slot)` 重叠，尾部统一 drain，保证 kernel 退出前所有 store 完成。`static_assert` 在编译期校验 `深度 × 块大小 + 预留 ≤ UB_SIZE`。
4. **64 位偏移**。`SourceOffset`/`PackedOffset` 返回 `uint64_t`，`n = 8192` 时 packed 偏移 33 558 528 不溢出。
5. **尾块处理**。chunk 长度非 32 B 对齐时用 `DataCopyPadExtParams` 补齐，只写有效字节，不污染 AP 后续元素。

### 3.2.4 代码组织与构建接入

```
blas/trttp/arch35/
├── strttp_host.cpp / strttp_kernel.cpp / strttp_tiling_data.h   # 既有实数版本
├── ctrttp_host.cpp / ctrttp_kernel.cpp / ctrttp_tiling_data.h   # 本算子
test/trttp/ctrttp/
├── CMakeLists.txt                 # ops_blas_add_gtest_tests(... ctrttp_test)
├── ctrttp_param.h                 # CSV 列解析（缺列即抛异常）
├── ctrttp_golden.h                # LAPACK 3.12 ctrttp_ golden
└── arch35/
    ├── ctrttp_npu_wrapper.h       # host<->device 搬运封装
    ├── ctrttp_test.cpp            # CSV 驱动 GTest + 2 条手写用例
    └── ctrttp_test.csv            # 1200 条任务用例
```

`blas/CMakeLists.txt` 按 `arch35` 目录自动收集源文件，`test/CMakeLists.txt` 按 `test/trttp/ctrttp` 家族路径自动定位，无需额外注册（已在 CANN 9.1.0 + ascend950 下验证）。

## 3.3 支持硬件

- Ascend 950PR / Ascend 950DT（arch35）：支持
- Atlas A2 / A3（arch22）：不支持（无 arch22 实现）

## 3.4 算子约束限制

- A 与 AP 不得重叠；AP 必须是独立输出。
- 不解析任意 stride，A 仅按 `lda` 前导维访问。
- 不要求 dynamic shape；n 为运行时入参。
- 同一设备、同一输入、同一 tiling 下输出逐元素一致（确定性）。

# 4. 可维可测分析

## 4.1 精度标准

纯搬运算子，判据为 `rtol = 0`、`atol = 0`、`required_matched_ratio = 1.0`、`max_abs_error = 0`，即 AP 前 `n(n+1)/2` 个元素与 golden **按位相等**；同时校验 packed 前缀之外的哨兵尾部未被改写。测试工程对整块 AP 缓冲做 `memcmp`，比逐元素浮点比较更严格（NaN 载荷差异也会被抓到）。

golden 使用 LAPACK 3.12.0 的 `ctrttp_`（Ubuntu 22.04 `liblapack3 3.10.0-2ubuntu1` 提供，Fortran ABI 为交错 re/im float），声明见 `test/trttp/ctrttp/ctrttp_golden.h`；CBLAS 不提供 `cblas_ctrttp`。

## 4.2 测试设计

### 4.2.1 用例覆盖

任务包 `ctrttp_test.csv` 共 1200 条（1000 条精度 + 200 条性能）：

| 类别 | 前缀 | 条数 | 覆盖点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 4 | uplo × n ∈ {1,4} |
| L1 尺寸 | TC_SQ | 88 | 44 种尺寸（1→4096，含质数/2 的幂 ±1/非对齐） |
| L2 packed 偏移 | TC_PK | 8 | n ∈ {1,2,3,5} 逐元素核对偏移公式 |
| L4 前导维 | TC_LD | 24 | padding {+1,+3,+8,+n} |
| L5 填充 | TC_FL | 20 | 7 种输入填充 + 3 种哨兵填充 |
| L5b 覆盖 | TC_CV | 13 | 中等尺寸 + n=8192 内存核对 |
| L6 边界 | TC_ED | 23 | no-op、n=0&lda=0、负 n、非法 uplo、非法 lda、空指针、Inf/NaN |
| EX 扩展 | TC_EX | 820 | 尺寸池 × uplo × lda × 填充确定性采样 |
| PF 性能 | TC_PF | 200 | 4 条 §3.4 用例 + 小尺寸 + 对数扫描 |

补充两条手写 `TEST_F`：空 handle（期望 `HANDLE_IS_NULLPTR`）、A/AP 重叠（契约外，仅验证不崩溃）。

### 4.2.2 数据准备与独立参考

CSV 每行由 `test/frame/fill.h` 的填充器生成 A（`RANDOM_NORM_5_5` 均匀 U(-5,5) 与 `GAUSS_NORM_5_2` 截断正态各 50%），AP 预填哨兵；`makeBlasArray(size*2, ...)` 以 float 数组承载复数。golden 由 LAPACK `ctrttp_` 生成，与实现完全独立。

## 4.3 性能标准与测量方案

判据（任务书 §3.3）：每个性能 case 的 `gpu_us / npu_us >= 0.4`，且全部 case 的比值算术平均 `>= 0.5`。

测量方式：

1. 主口径：`bench/ctrttp_bench.cpp`，`aclrtEvent` 记录连续 20 次下发（warmup 5 次）的设备总耗时取平均，等价于单算子 task duration（不含 host 准备与 golden）。
2. 保守上界：GTest `[ OK ] ... (N ms)` 端到端耗时（含 host 搬运与 golden）。
3. `msprof op` 对直调 kernel 无法解析算子名（`Analyzing kernel data failed ... name is: 0`），故自验以 aclrtEvent 为准，报告中同时给出端到端上界。

实测（Ascend 950PR / CANN 9.1.0，单卡，n = 1024/2048/4096/8192）：

| n | uplo | NPU 实测 (us) | H100 基线 (us) | 比值 |
| --- | --- | --- | --- | --- |
| 1024 | LOWER | 10.90 | 17.68 | 1.62 |
| 2048 | UPPER | 21.00 | 22.87 | 1.09 |
| 4096 | LOWER | 86.77 | 112.89 | 1.30 |
| 8192 | UPPER | 445.93 | 351.00 | 0.79 |

n = 8192 时 packed 搬运量 256 MiB 读 + 256 MiB 写，按 445.93 us 计有效带宽约 1.2 TB/s。
全量 200 条 `TC_PF` 用例（n = 1 … 8192，两种 uplo，lda = n）全部达标：
最小比值 0.79（n = 8192 UPPER），比值算术平均 2.46（判据：单 case ≥ 0.4、均值 ≥ 0.5）。

## 4.4 内存、可维性与兼容性

- 额外内存：kernel 只用 UB（7 × 32 KiB = 224 KiB，满足 248 KiB 预算减去 20 KiB 元数据预留），无 workspace、无 GM 临时缓冲；n = 8192 时 A 512 MiB + AP 256 MiB 均由调用方提供。
- 可维性：host 校验、tiling、kernel 各自独立函数；分核公式与偏移公式均为闭式，注释给出列主序推导。
- 兼容性：接口声明放公共头文件，其他产品线可共用；实数 `aclblasStrttp` 行为不变（本次未改动其校验顺序，差异在 README 与设计文档中说明）。
- CANN 9.1.0 兼容：`test/frame/csv_loader.h` 的 `ACL_HIFLOAT4` 映射加 `OPS_BLAS_HAS_ACL_HIFLOAT4` 守卫（CANN 9.2 起才有该枚举），否则 9.1.0 下所有 GTest 目标无法编译。
