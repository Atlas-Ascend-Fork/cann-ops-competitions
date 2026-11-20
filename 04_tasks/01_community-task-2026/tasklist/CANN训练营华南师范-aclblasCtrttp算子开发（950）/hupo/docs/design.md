# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务：aclblasCtrttp 算子开发（Atlas 950PR）。任务书见 `aclblasCtrttp_Atlas950PR_task_doc.md`。

任务目标是补齐 ops-blas 仓 BLAS 扩展接口能力：`blas/trttp/arch35` 目录已有实数版本 `aclblasStrttp`，复数版本 `aclblasCtrttp` 缺失。

## 背景介绍

### 算子定位

`ctrttp` 是 LAPACK 例程（不是 BLAS/CBLAS 接口，CBLAS 不提供 `cblas_ctrttp`），功能是把常规存储的三角矩阵压缩为 packed 存储。cuBLAS 自 13.x 起提供对应的 `cublasCtrttp`，本算子的核心功能、参数语义与之逐项对齐。

数学定义（A 为 lda×n 列主序 complex64 矩阵，AP 为长度 n(n+1)/2 的 packed 数组，0 ≤ i, j < n）：

```
LOWER：AP[j·n − j·(j−1)/2 + (i − j)] = A(i,j)，j ≤ i
UPPER：AP[j·(j+1)/2 + i]         = A(i,j)，i ≤ j
```

两种 uplo 的列起点不同（LOWER 起点随 j 递减、UPPER 起点随 j 递增），不能复用同一套偏移。

### 仓内现状分析

`blas/trttp/arch35` 现有三个文件：`strttp_host.cpp`、`strttp_kernel.cpp`、`strttp_tiling_data.h`。S 版已实现同构的实数压缩，可直接作为工程结构与风格基准，但存在两处不能沿用的行为：

| 项 | aclblasStrttp 现状 | aclblasCtrttp 要求 |
| --- | --- | --- |
| `n == 0 && lda == 0` | 返回 SUCCESS（前导维校验在早返回之后） | 返回 `ACLBLAS_STATUS_INVALID_VALUE`（前导维校验在早返回之前） |
| 校验顺序 | handle → n<0 → uplo → n==0 早返回 → A/AP → lda | handle → n<0 → uplo → lda → n==0 早返回 → A/AP |
| 偏移计算 | 32 位（`uint32_t`） | 64 位（`uint64_t`），n=8192 时 packed 长度 33 558 528 |

接口层面，`include/cann_ops_blas.h` 目前只有 `aclblasStrttp` 声明，需新增 `aclblasCtrttp`，与其他产品线共用同一 API，不定义 950PR 私有平行接口。

### 接口定义

```cpp
aclblasStatus_t aclblasCtrttp(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* A, const int lda,
    aclblasComplex* AP);
```

`aclblasComplex` 取自 `include/cann_ops_blas_common.h`，实部/虚部各 float32，元素 8 字节。

# 需求分析（required）

## 需求描述

在 Ascend 950PR（arch35）上，使用 Ascend C 编程语言、kernel 直调方式，基于 ops-blas 开源仓实现 `aclblasCtrttp` 句柄式 BLAS 接口：通过 handle 绑定 stream 直调 NPU kernel，把列主序 lda×n 复数矩阵 A 的 uplo 三角压缩为 packed 数组 AP。

## 需求拆解

1. 功能：按列从 A 读取 uplo 指定三角，顺序写入 AP 前 n(n+1)/2 个位置；LOWER/UPPER 两套偏移分别实现；只读指定三角，A 的另一三角与 lda padding 行不读；A 与 AP 不得重叠。
2. 数据类型：A、AP 均为 complex64；本算子只做数据搬运，不改变数值。
3. 参数校验：handle → n<0 → uplo 枚举 → lda ≥ max(1,n) → n==0 早返回 → A/AP 判空；非法 uplo/n<0/lda 不足统一返回 `ACLBLAS_STATUS_INVALID_VALUE`（不用 `INVALID_ENUM`）；空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`n=0 && lda=0` 必须返回 `INVALID_VALUE`。
4. 边界：n=0 且 lda ≥ 1 为合法 no-op，不写 AP。
5. 精度：纯搬运，按位相等（re/im 分别判定），rtol = atol = 0。
6. 性能：单 case 平均耗时 ≤ 2.5× GPU 标杆（H100 + cuBLAS 13.3），全部性能 case 平均倍率 ≥ 0.5×。
7. 工程：代码落 `blas/trttp/arch35/`，测试落 `test/trttp/ctrttp/arch35/`，接口声明进 `include/cann_ops_blas.h`，README 产品支持表标注 Ascend 950PR 支持。

# 详细设计（required）

## 算子分析

### 数学公式

```
AP = pack(A)
LOWER：AP[j·n − j·(j−1)/2 + (i − j)] = A(i,j)，j ≤ i
UPPER：AP[j·(j+1)/2 + i]         = A(i,j)，i ≤ j
```

列 j 在 packed 数组中的起点与长度：

| uplo | packed 起点 | 列长 | 源起始行 |
| --- | --- | --- | --- |
| LOWER | `j·n − j·(j−1)/2` | `n − j` | `j` |
| UPPER | `j·(j+1)/2` | `j + 1` | `0` |

关键性质：在单列内部，packed 位置随行号单调递增，因此「一列的一段」在源端和目的端**都是连续区间**，可以用一次连续搬运完成，不需要 gather。

### 支持数据类型

complex64（`aclblasComplex`，re/im 各 float32），设备侧按两个 float 处理，保证 4 字节可寻址。

### 支持形状

A：`[lda, n]` 列主序，`lda ≥ max(1, n)`；AP：`[n(n+1)/2]` 一维 packed。不涉及广播、stride、dynamic shape，n 为运行时入参。

## 算子实现

### 实现方案

新增三个文件，与实数版本同目录同风格：

| 文件 | 职责 |
| --- | --- |
| `blas/trttp/arch35/ctrttp_tiling_data.h` | tiling 结构（64 位 n/lda/packedLen + uplo/useCoreNum） |
| `blas/trttp/arch35/ctrttp_host.cpp` | 参数校验、tiling 计算、kernel 下发 |
| `blas/trttp/arch35/ctrttp_kernel.cpp` | AIV kernel（首版）：按 packed 区间分核，逐列连续搬运 |
| `blas/trttp/arch35/ctrttp_kernel_column_rr.cpp` | AIV kernel（列轮转快速路径）：按列轮转分核，`n >= useCoreNum` 时启用 |

`blas/CMakeLists.txt` 原本按 `archXX` 目录递归 glob 收集 `*.cpp`。新增的列轮转 kernel 与首版 kernel 是两个独立编译单元，为免递归 glob 重复收集、并确保只对 arch35 生效，构建里对该文件显式去重后单独登记。

#### 3.2.1 host 侧设计

**校验顺序**（这是与 S 版最重要的行为差异）：

```cpp
if (handle == nullptr)            return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
if (n < 0)                        return ACLBLAS_STATUS_INVALID_VALUE;
if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) return ACLBLAS_STATUS_INVALID_VALUE;
if (lda < std::max(1, n))         return ACLBLAS_STATUS_INVALID_VALUE;   // 覆盖 n==0 && lda==0
if (n == 0)                       return ACLBLAS_STATUS_SUCCESS;         // 合法 no-op，不写 AP
if (A == nullptr || AP == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;  // 仅 n>0 时判空
```

把 `lda` 校验放在 `n == 0` 早返回之前，`n=0, lda=0` 就自然落到 `INVALID_VALUE`，不需要额外分支。

**tiling 计算**：

```
packedLen = (uint64)n * (n+1) / 2
useCoreNum = min(AIV 核数, packedLen)     // packedLen ≥ 1，故至少 1 核
```

核数上限取 `min(核数, packedLen)`：每核至少分到 1 个 packed 元素的自然下界，避免下发空核。

**kernel 下发与路由**：stream 来自 handle。`n >= useCoreNum` 时下发列轮转快速路径 `ctrttp_kernel_column_rr`，否则下发首版 `ctrttp_kernel`；两条路径的 packed 映射完全一致，路由只改变分核方式，不改变输出。

#### 3.2.2 kernel 侧设计

采用 `Init` + `Process` 两段式，单 AIV 核逻辑：

**1. 分核策略（首版）：按 packed 输出区间切分，不做列切分**

S 版按「连续列」分核。对 LOWER 而言列长从 n 递减到 1，前几列极重、后几百列极轻：n=8192、48 核时，最重的核要搬约 1.6M 元素，而均衡值只有约 0.7M，负载差 2 倍以上，直接损失并行度。

本算子改为把 packed 输出区间 `[0, packedLen)` 均分为 `useCoreNum` 段，每核负责一段连续区间：

```
base = packedLen / useCoreNum；remain = packedLen % useCoreNum
packedStart = blockIdx * base + min(blockIdx, remain)
packedCount = base + (blockIdx < remain ? 1 : 0)
```

每核从自己的 `packedStart` 反解出所属列 `FindCol`，然后逐列推进，直到 `packedStart + packedCount`。列粒度上的负载误差不超过 1 列（n=8192 时约占单核工作量的 0.02%），实际是平的。

**1b. 列轮转快速路径（上板实测后新增）**

首版在大 n 上达标，但 LOWER 的列长从 n 递减到 1，按连续 packed 区间分核后，`n < useCoreNum` 的规模会出现空核，`n` 略大于核数的规模又会出现列长不均。快速路径改为**列轮转**：每核负责列 `blockIdx`、`blockIdx + useCoreNum`、…，各核连续搬运段数最多相差 1，LOWER/UPPER 都自然均匀：

```
for (col = blockIdx; col < n; col += useCoreNum) { CopyRun(...) }
```

目标 packed 偏移沿用与首版相同的列映射，因此两条路径输出逐位一致；`n < useCoreNum` 时轮转会产生空核，故由 host 路由回首版 kernel。

**2. 三角反解：整数二分，不引入浮点**

`FindCol(p)` 求满足 `colStart(j) ≤ p < colStart(j+1)` 的列 j，`colStart` 单调递增，直接二分：

```
lo = 0, hi = n-1
while (lo < hi) { mid = lo + (hi-lo+1)/2; if (colStart(mid) <= p) lo = mid; else hi = mid-1; }
```

O(log n) 次标量比较（n=8192 时 13 次），每核只做一次。用公式反解（解二次方程）需要 double sqrt 并做浮点修正，在 AIV 标量单元上既慢又有精度负担，因此不用。

**3. 单核搬运流程**

对列 j 取它与本核区间的交集 `[runStart, runEnd)`：

```
runLen   = runEnd − runStart                 // 该段搬多少个 complex
srcRow   = (runStart − colStart(j)) + (uplo==0 ? j : 0)
srcComplex = j * lda + srcRow
dstComplex = runStart
```

然后以「两个 float 一个 complex」为单位，按 `MAX_CHUNK_FLOATS = 16384`（64 KB UB）分段做 GM→UB→GM：

```
DataCopyPad(ub, aGM_[srcComplex*2 + done], cp, pad)     // GM → UB
SetFlag<MTE2_MTE3> / WaitFlag<MTE2_MTE3>
DataCopyPad(apGM_[dstComplex*2 + done], ub, cp)         // UB → GM，只写 chunk 字节
SetFlag<MTE3_MTE2> / WaitFlag<MTE3_MTE2>
```

`cp = DataCopyExtParams{1, chunkBytes, 0, 0, 0}`，非 32B 整数倍的尾段用 `DataCopyPadExtParams` 的右 padding 补齐 UB；搬出时只写 `chunkBytes`，padding 不落到 GM，因此不污染数据。这正是同族 S 版验证过的搬运方式。

**4. 内存与 64 位偏移**

- 全程流式搬运，不申请 workspace，不做整矩阵中转，峰值额外内存只有 64 KB UB。
- `n`、`lda`、`packedLen`、列起点、源偏移全部用 `uint64_t` 计算；n=8192 时 packed 长度 33 558 528 元素（256 MiB），lda 方向的源偏移最大到 8192×8192=67 108 864 complex（512 MiB），32 位中间量有溢出风险，必须 64 位。

**5. 关键设计取舍**

| 取舍 | 选择 | 理由 |
| --- | --- | --- |
| 分核维度 | packed 输出区间 | 负载平；列切分在 LOWER 下严重倾斜 |
| 反解方式 | 整数二分 | 无浮点、无精度修正、可验证 |
| 数据单位 | complex 拆两个 float | 4 字节对齐、按位搬运不变数值、复用 DataCopyPad |
| UB | 单缓冲 64 KB，串行 MTE2/MTE3 | 首版对齐仓内既有模式，风险最低；见性能章节 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |
| Atlas A3 训练/推理系列产品 | 不涉及（无 arch35 实现） |
| Atlas A2 训练/推理系列产品 | 不涉及（无 arch35 实现） |

## 算子约束限制

- n ≥ 0；`n=0 && lda ≥ 1` 为合法 no-op，不读写任何内存。
- `lda ≥ max(1, n)`；不满足返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- uplo 仅支持 `ACLBLAS_UPPER`(121)/`ACLBLAS_LOWER`(122)，其他值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- A 与 AP 不得重叠，重叠时行为未定义（由调用方保证）。
- 不解析任意 stride，不支持广播、不支持原地/视图语义。
- 确定性：同一设备、同一输入、同一 tiling 下多次调用输出逐元素一致（无原子、无并发写同一地址）。
- 异步执行：依赖 `aclblasSetStream` 绑定的 stream，读回结果前需同步。

# 可维可测分析

## 上板实测结果（Ascend 950PR / arch35，CANN 9.1.0）

| 项 | 结果 |
| --- | --- |
| 构建 | `bash build.sh --soc=ascend950 --ops=ctrttp`，零补丁通过（当前 master 已定义 `ACL_HIFLOAT4`，无需 CXXFLAGS 补丁） |
| 功能 | 1004/1004 通过（1000 条 CSV + 4 条专项），0 失败，AP 尾部哨兵改写 0 |
| 性能采集 | msprof op 逐例采集，200/200 有效 |
| 性能达标 | 200/200 `ratio >= 0.4`，均值 1.5514，最小 0.6741，最大 6.7911 |
| 相对首版 | 有效提升 193 / 抖动 7 / 有效退化 0，有效 case 几何加速 4.2094x |

统计口径为 `legacy_jitter_v1`：`|Δ|` 落在回退抖动阈值内的 case 记为 `jitter_no_effect`，不参与有效指标。

## 已完成的自验证（设备无关部分）

设备不可用期间先完成了三层离线验证，均可复现，且都明确标注了边界：

| 层 | 内容 | 结果 |
| --- | --- | --- |
| 算法模型 | Python 仿真分核/偏移/三角反解，穷举+随机组合 | 2616 组组合按位一致 |
| 真实代码 | CPU shim 顶替 AscendC API，直接编译执行本算子 host/kernel，含 `chunk=6` 变体强制多轮分块与非 32B 对齐拷贝 | 两个变体各 3063 项检查 0 失败 |
| 第三方契约 | 与 LAPACK 3.12.0 `ctrttp_` 交叉比对；用仓内 CSV 加载器核验 1200 行用例的期望状态码 | 838 项 0 失败；1200 行 0 不一致 |

离线验证覆盖不到、必须上板确认的：`DataCopyPad` 的地址对齐约束与参数语义、POD tiling 传值与 kernel 启动的编译器接受度、多核并发与 MTE2/MTE3 同步、精度实测与 `msprof` 性能采集。

其中地址对齐一项已取得旁证：同仓 `blas/dot/arch35/cdotc_kernel.cpp` 以 8 字节对齐的 GM 地址、非 32B 整数倍的长度搬运 complex64，该 kernel 在 arch35 上随仓 CI 通过，说明平台容忍此类偏移。最终仍以精度用例（含 n=1/2/3/5 与多档 padding lda）在设备上的结果为准。

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | AP 前 n(n+1)/2 个元素与 golden 逐元素按位相等（re/im 分别判定，rtol=atol=0），AP 尾部哨兵不被改写；Inf/NaN 同样按位搬运 | 任务书 §3.2；生态算子开源精度标准中更严格的搬运类口径 |
| 性能标准 | 单 case 平均单次耗时 ≤ 2.5× GPU 标杆（H100 + cuBLAS 13.3，`gpu_baseline.csv`），全部性能 case 平均倍率 ≥ 0.5× | 任务书 §3.3 |

CSV 精度 golden 使用 **LAPACK 3.12.0 `ctrttp_`**（CBLAS 无 `cblas_ctrttp`），测试工程通过 `test/utils/cblas_compat.h` 与参考 LAPACK 链接；显式 NaN payload、±Inf、±0 和重复调用确定性用手工位模式输入，按 §2.1 公式独立构造 CPU 期望，不依赖共享浮点 verifier。

测试工程沿用仓内 GTest + CSV 框架：

| 文件 | 作用 |
| --- | --- |
| `test/trttp/ctrttp/ctrttp_param.h` | CSV 列读取（uplo/n/a/lda/ap 及公共列），缺列或空值抛异常 |
| `test/trttp/ctrttp/ctrttp_golden.h` | LAPACK `ctrttp_` golden，校验顺序与设备侧一致 |
| `test/trttp/ctrttp/arch35/ctrttp_npu_wrapper.h` | host↔device 拷贝、同步、AP 哨兵尾部回拷 |
| `test/trttp/ctrttp/arch35/ctrttp_test.cpp` | `NullHandle`、显式位模式、A/AP 重叠记录、重复调用确定性 + CSV 驱动用例，逐位比对 + 尾部哨兵 `memcmp` |
| `test/trttp/ctrttp/arch35/ctrttp_test.csv` | 随任务提供的 1200 条用例 |

用例覆盖：两种 uplo × n 扫描（0/1/质数/2 的幂 ±1/非对齐/8192）× lda（紧凑/+1/+3/2n）× 均匀与正态混合输入 × AP 哨兵填充 × Inf/NaN × 负向（负 n、非法 uplo、lda 不足、`n=0,lda=0`、空指针）。packed 偏移在 n=1,2,3,5 上逐元素核对。另用手工 `TEST_F` 覆盖不同 NaN payload、±Inf、±0、A/AP 同指针（输出未定义，仅记录当前返回与同步行为）和重复调用确定性；所有 CSV 比对直接检查 float 位模式，避免共享 verifier 跳过 NaN 对。

规模与内存：n=8192、lda=8192 时 A 为 512 MiB、AP 为 256 MiB；测试进程 host 侧另有等量的比对缓冲，需 ≥ 2 GiB 可用内存。

## 性能设计说明

首版 kernel 为「单缓冲 + 串行 MTE2/MTE3」，正确性风险最低。上板实测后确认，瓶颈不在单核流水深度，而在分核方式：LOWER 的列长从 n 递减到 1，按连续列或连续 packed 区间分核，都会让部分核承担明显更重的搬运量。

实测与迭代（Ascend 950PR，56 个 AIV 核，msprof op，5 次预热 + 10 次采样，1650MHz 定频）：

1. 首版：多数 case 的 `GPU/NPU ratio` 低于 0.4，未达标；
2. 按 packed 输出区间均衡分核：大 n 明显改善，但小 n 出现空核与列长不均；
3. 最终采用「列轮转」快速路径（见 3.2.2 节 1b），`n < useCoreNum` 的极小规模保留首版 kernel。

最终结果：200/200 例 `ratio >= 0.4`，ratio 均值 1.5514（最小 0.6741，最大 6.7911）；相对首版实现有效提升 193 例、抖动 7 例、有效退化 0 例，有效 case 几何加速 4.2094x。

未采用的对策：双缓冲 ping-pong 与 tiling 核数收敛经评估收益低于分核改造（搬运粒度为整列连续区间，UB 已按 64KB 分段），因此没有引入，避免在无收益场景增加 kernel 复杂度。

## 兼容性分析

新增接口，不改动已有 `aclblasStrttp` 的行为与签名；`include/cann_ops_blas.h` 仅追加声明，不影响其他产品线的编译与链接。构建脚本仅对新增的列轮转 kernel 做了显式去重登记，对既有算子零影响。