# aclblasCtrttp 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | aclblasCtrttp（复数单精度三角压缩，complex64 / `aclblasComplex`） |
| 对标接口 | cuBLAS `cublasCtrttp`（CUDA 13.x，参数序列一一对应） |
| 目标硬件 | Atlas 950PR（arch35） |
| 软件栈 | CANN 9.1.0，Ascend C kernel 直调模式 |
| 代码路径 | 实现 `blas/trttp/arch35/`；测试 `test/trttp/ctrttp/arch35/` |
| 接口声明 | `include/cann_ops_blas.h`（新增，全产品线共用，禁止私有平行接口） |
| 上游仓库 | ops-blas（https://gitcode.com/cann/ops-blas ） |
| 定位 | 补齐 `blas/trttp/arch35` 下仅有实数版本 `aclblasStrttp` 的复数能力缺口 |

---

## 1 算子概述

`aclblasCtrttp` 把列主序 `lda × n` 三角矩阵 `A` 的 `uplo` 指定三角，按列顺序压缩（pack）到长度为 `n(n+1)/2` 的一维数组 `AP` 中，对应 LAPACK 的 `CTRTTP` 例程语义。算子是**纯数据搬运**算子：不做任何浮点运算、不做类型转换、不做数值重排以外的变换，因此输出必须与输入逐位（bit-exact）相同。

### 1.1 功能定义

记 `A(i, j)` 为矩阵第 `i` 行、第 `j` 列元素（`0 ≤ i, j < n`），列主序下其线性偏移为 `i + j·lda`；`AP` 按列依次存放指定三角的全部元素，`0 ≤ k < n(n+1)/2`。

- `uplo == ACLBLAS_LOWER`（下三角，取 `j ≤ i`）：

  ```text
  AP[j·n − j·(j−1)/2 + (i − j)] = A(i, j),   j ≤ i
  S_lower(j) = j·n − j·(j−1)/2 = j·(2n − j + 1)/2
  列 j 长度 = n − j
  ```

- `uplo == ACLBLAS_UPPER`（上三角，取 `i ≤ j`）：

  ```text
  AP[j·(j+1)/2 + i] = A(i, j),   i ≤ j
  S_upper(j) = j·(j+1)/2
  列 j 长度 = j + 1
  ```

其中 `S_uplo(j)` 为第 `j` 列在 `AP` 中的起始偏移，`S_uplo(0) = 0`，`S_uplo(n) = n(n+1)/2`。两套偏移的列起点与列长度均不同，**不可复用同一套偏移公式**。

### 1.2 关键行为

1. **只读指定三角**：`LOWER` 仅读 `j ≤ i` 的元素，`UPPER` 仅读 `i ≤ j` 的元素；`A` 的另一三角元素与 `lda` padding 行（`n ≤ i < lda`）**一律不读取**，其取值不影响 `AP`。
2. **只写 packed 区**：仅写 `AP[0, n(n+1)/2)`；尾部哨兵区（若存在）不得被改写。
3. **64 位偏移**：`n` 最大支持 8192，此时 `n(n+1)/2 = 33 558 528`，`j·n` 最大约 `6.7×10^7`，所有列起始偏移、元素偏移、`AP` 长度均必须使用 `int64_t` 计算，禁止使用 `int`/`int32_t` 中间量（避免未来更大 `n` 与 `lda` 组合下的溢出）。
4. **无重叠**：调用方须保证 `A` 与 `AP` 内存不重叠；重叠属未定义行为，算子不做检测。
5. **复数按位搬运**：`aclblasComplex` 由 `include/cann_ops_blas_common.h` 定义（实部、虚部各 `float32`，共 8 字节），算子在字节层面搬运，`Inf`/`NaN`（含 NaN 载荷位、符号位）原样保留。

### 1.3 典型示例（n = 4）

| 列 j | LOWER 起始 `S(j)` | LOWER 区间 | UPPER 起始 `S(j)` | UPPER 区间 |
| --- | --- | --- | --- | --- |
| 0 | 0 | `AP[0..3]` ← `A(0..3, 0)` | 0 | `AP[0]` ← `A(0, 0)` |
| 1 | 4 | `AP[4..6]` ← `A(1..3, 1)` | 1 | `AP[1..2]` ← `A(0..1, 1)` |
| 2 | 7 | `AP[7..8]` ← `A(2..3, 2)` | 3 | `AP[3..5]` ← `A(0..2, 2)` |
| 3 | 9 | `AP[9]` ← `A(3, 3)` | 6 | `AP[6..9]` ← `A(0..3, 3)` |

两种 `uplo` 的 `AP` 长度均为 `10 = 4·5/2`。

---

## 2 接口说明

### 2.1 接口声明

在 `include/cann_ops_blas.h` 中新增声明，与仓内同族实数接口 `aclblasStrttp` 同构，导出宏沿用该头文件既有约定（如 `ACLBLASAPI`）：

```cpp
aclblasStatus_t aclblasCtrttp(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* A, const int lda,
    aclblasComplex* AP);
```

> 该接口为跨产品线公共 API，实现体放在 `blas/trttp/arch35/`，但**不得**在 arch35 目录内定义任何 950PR 私有的平行接口或重载；其他产品线的实现只需复用同一声明。

### 2.2 参数说明

| 参数 | 输入/输出 | 类型 | 说明 | 约束 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t`（Host） | ops-blas 上下文句柄，内部携带 `stream` | 不得为 `nullptr`；为 `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 输入（属性） | `aclblasFillMode_t` | `ACLBLAS_LOWER`：下三角（`j ≤ i`）；`ACLBLAS_UPPER`：上三角（`i ≤ j`） | 仅接受上述两个枚举；其他取值返回 `ACLBLAS_STATUS_INVALID_VALUE`（**不是** `INVALID_ENUM`） |
| `n` | 输入 | `const int` | 矩阵阶数 | `n ≥ 0`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n = 0 && lda ≥ 1` 为合法 no-op |
| `A` | 输入 | `const aclblasComplex*`（Device） | `lda × n` 列主序复数矩阵，只读 | `n > 0` 时不得为 `nullptr`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`；仅引用 `uplo` 指定三角 |
| `lda` | 输入 | `const int` | `A` 的前导维度（列主序，单位：元素） | `lda ≥ max(1, n)`；否则返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `AP` | 输出（独立输出） | `aclblasComplex*`（Device） | packed 三角数组，长度 `n(n+1)/2`，按列压缩 | `n > 0` 时不得为 `nullptr`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`；不与 `A` 重叠 |

### 2.3 返回值

| 返回码 | 触发条件 |
| --- | --- |
| `ACLBLAS_STATUS_SUCCESS` | 正常执行（异步下发成功）或合法 no-op（`n = 0`、`lda ≥ 1`） |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | `handle == nullptr` |
| `ACLBLAS_STATUS_INVALID_VALUE` | `n < 0`；`uplo` 非法；`lda < max(1, n)`（含 `n = 0 && lda = 0`）；`n > 0` 且 `A`/`AP` 为 `nullptr` |

状态码语义与 `include/cann_ops_blas_common.h` 一致。**本接口不返回** `ACLBLAS_STATUS_INVALID_ENUM`。

### 2.4 异步语义

接口仅做参数校验、tiling 计算与 kernel 下发，随后立即返回 `SUCCESS`；实际搬运由 `handle` 绑定的 `stream` 异步执行。调用方须在读取 `AP` 前同步该 stream（`aclblasStreamSynchronize` 或等价操作），`aclblasSetStream` 更换 stream 后行为随之改变。

---

## 3 输入输出描述

### 3.1 张量与内存布局

| 名称 | 方向 | dtype | 排布 | 逻辑形状 | 元素数 | 字节数 |
| --- | --- | --- | --- | --- | --- | --- |
| `A` | 输入 | `COMPLEX64`（实部/虚部各 `FLOAT32`） | ND，列主序（`i + j·lda`） | `[lda, n]` | `lda·n` | `8·lda·n` |
| `AP` | 输出 | `COMPLEX64` | ND，连续一维 | `[n(n+1)/2]` | `n(n+1)/2` | `8·n(n+1)/2` |

要点：

- 仅 `A` 的 `lda × n` 有效区域内、且属于 `uplo` 指定三角的元素被读取；`A` 中 `j·lda` 列内 `i ≥ n` 的 padding 行不参与。
- `AP` 依次存放第 0 列到第 `n−1` 列的三角段，中间无空洞、无对齐填充（元素级连续）。
- 不涉及 stride、广播、动态 shape、视图或原地写；`AP` 为独立输出，不复用 `A` 的显存。

### 3.2 规模与内存占用（`lda = n` 时为最小占用）

| `n` | `n(n+1)/2` | `AP` 字节数 | `A` 最小字节数 |
| --- | --- | --- | --- |
| 1 | 1 | 8 B | 8 B |
| 3 | 6 | 48 B | 72 B |
| 1 024 | 524 800 | 4.0 MiB | 8.0 MiB |
| 4 096 | 8 390 656 | 64.0 MiB | 128.0 MiB |
| 8 192 | 33 558 528 | 256.0 MiB | 512.0 MiB |

`lda` 取 `2·max(1, n)` 时 `A` 占用翻倍（`n = 8192` 时为 1 GiB）。算子不需要额外 GM workspace，仅使用 UB。

---

## 4 算子算法原理

### 4.1 映射的本质：按列的两次连续区间拷贝

对固定列 `j`，`AP` 中该列占用一个**连续的**区间 `[S_uplo(j), S_uplo(j) + L_uplo(j))`；`A` 中该列的三角段同样连续（列主序下同列元素相邻）：

| uplo | `S(j)` | `L(j)` | `A` 起始元素偏移 | `AP` 写入偏移 |
| --- | --- | --- | --- | --- |
| LOWER | `j·(2n − j + 1)/2` | `n − j` | `j·lda + j` | `S(j)` |
| UPPER | `j·(j+1)/2` | `j + 1` | `j·lda + 0` | `S(j)` |

因此整个算子可分解为 `n` 个「一维连续段从 `A` 搬到 `AP`」的子任务，无需逐元素地址计算：

```text
for j in [0, n):
    src = A + (j·lda + (uplo == LOWER ? j : 0))
    dst = AP + S_uplo(j)
    copy L_uplo(j) 个 complex64 元素： dst[0:L] = src[0:L]
```

两式中的 `/2` 均可整除：`j` 为偶数时 `j·(2n−j+1)` 含偶因子 `j`；`j` 为奇数时 `2n−j+1` 为偶数；`j·(j+1)` 恒为两个连续整数之积。实现上仍用 `int64_t` 正向乘法、再整除，避免先除法带来的截断风险。

### 4.2 反解列号（kernel 侧按输出区间分核时使用）

若按 `AP` 的一维区间 `[p0, p1)` 给核分片，需要由 `p` 反解列号 `j`，即求满足 `S(j) ≤ p < S(j+1)` 的 `j`：

- `LOWER`：`S(j)` 单调递增，用 `int64_t` 二分（`0 ≤ j < n`，最多 13 次迭代，`n = 8192`）；
- `UPPER`：同理二分；亦可由 `j = floor((sqrt(8p+1) − 1)/2)` 估算后 ±1 修正，但为避免浮点在超大 `n` 下的边界误差，**统一采用整数二分**。

`p` 与 `S(j)` 均为 `int64_t`，`S(n) = n(n+1)/2` 作为二分上界哨兵。

### 4.3 位精确性论证

算子对每个元素执行 `dst[k] = src[k]` 的 8 字节原样拷贝：不经过任何浮点算术单元、不做舍入、不做大小端/寄存器重排，因此：

- 有限值与 LAPACK/CPU 参考实现逐位相同；
- `±Inf`、`NaN`（quiet/signaling、任意载荷位、符号位）逐位保留；
- 结果是确定性的：同一输入、同一 tiling、同一设备多次执行结果完全一致。

---

## 5 host 侧设计

### 5.1 代码与文件布局

```text
include/
└── cann_ops_blas.h                     # 新增 aclblasCtrttp 声明（全产品线共用）
blas/trttp/arch35/
├── aclblas_ctrttp.h                    # host 入口声明、tiling 结构体
├── aclblas_ctrttp.cpp                  # 参数校验 + tiling + kernel 下发（含 Stream 绑定）
├── aclblas_ctrttp_tiling.h             # TilingData 定义与 tiling 计算
├── aclblas_ctrttp_kernel.cpp           # Ascend C kernel 入口（含 LOWER/UPPER 两个核函数）
└── CMakeLists.txt（或沿用父目录构建脚本） # 与 aclblasStrttp 同构的构建接入
test/trttp/ctrttp/arch35/
├── test_aclblas_ctrttp.cpp             # GTest 用例（CSV 驱动）
├── test_aclblas_ctrttp_precision.csv   # 精度用例
├── test_aclblas_ctrttp_performance.csv # 性能用例
├── gen_csv.py                          # 用例生成脚本（含 golden 生成说明）
└── README.md                           # 复现步骤
```

### 5.2 主流程

```text
aclblasCtrttp(handle, uplo, n, A, lda, AP)
  1. if handle == nullptr                     -> HANDLE_IS_NULLPTR
  2. if n < 0                                 -> INVALID_VALUE
  3. if uplo ∉ {LOWER, UPPER}                 -> INVALID_VALUE
  4. if lda < max(1, n)                       -> INVALID_VALUE    // 覆盖 n=0 && lda=0
  5. if n == 0                                -> SUCCESS（quick return，不下发 kernel，不触碰 AP）
  6. if A == nullptr || AP == nullptr         -> INVALID_VALUE
  7. T = (int64)n * (n + 1) / 2
     blockNum = min(aiCoreNum, T)
     base, rem = T / blockNum, T % blockNum
     tileElems = 依据 UB 大小与 double buffer 计算（见 §7）
  8. 组装 TilingData{n, lda, blockNum, tileElems, base, rem, isUpper}
  9. stream = GetStreamFromHandle(handle)
 10. aclblasCtrttpKernel<<<blockNum, nullptr, stream>>>(A, AP, tiling)
     // 或使用 Ascend C kernel 直调接口按仓内规范下发
 11. return SUCCESS
```

实现要点：

- 校验顺序必须严格按上表；`n == 0` 的早返回位于 `A/AP` 判空之前，因此 no-op 路径不会解引用任何指针。
- `max(1, n)` 用 `int minLda = (n > 1) ? n : 1;` 表达，`n = 1` 与 `n = 0` 均要求 `lda ≥ 1`。
- 所有 `int64_t` 计算在 host 侧完成，kernel 侧只接收已校验的 `n/lda` 与分核参数，避免在 kernel 内重复做除法。
- 不申请 GM workspace；tiling 结构体通过常量/参数区传入（与仓内 `aclblasStrttp` 保持一致）。
- 错误路径不产生任何 device 访问，保证负向用例下 `A`/`AP` 可为非法指针。

### 5.3 tiling 结构体

```cpp
struct CtrttpTilingData {
    int64_t n;          // 矩阵阶数（已校验 >= 1）
    int64_t lda;        // 前导维（元素）
    int64_t total;      // n * (n + 1) / 2
    int64_t base;       // total / blockNum
    int64_t rem;        // total % blockNum（前 rem 个核多处理 1 个元素）
    int64_t tileElems;  // 单次 GM<->UB 搬运的最大元素数
    int32_t blockNum;   // 实际启用核数
    int32_t isUpper;    // 0 = LOWER，1 = UPPER（同时作为 tilingKey 分支依据）
};
```

---

## 6 kernel 侧设计

### 6.1 总体结构

采用 Ascend C kernel 直调（`__global__ __aicore__`）模式，流程为 `Init → Process → 结束`，`Process` 内按「MTE2 搬入 → MTE3 搬出」流水，UB 内不做计算：

```text
              MTE2 (GM->UB)                 MTE3 (UB->GM)
A(GM) ───────────────────▶ UB ping/pong ───────────────────▶ AP(GM)
      DataCopyPad(精确长度)                 DataCopyPad(精确长度)
```

- 使用 `TPipe` + `TQue<TPosition::VECIN, 2>` / `TQue<TPosition::VECOUT, 2>` 形成 2 级 double buffer，使下一段的搬入与当前段的搬出重叠；对纯拷贝场景也可用两块 `TBuf` 手动 `SetFlag/WaitFlag`，但统一采用 `TQue` 与仓内风格一致。
- 计算单元（Vector）不参与；每段处理的地址与长度由标量 `int64_t` 运算得到。
- 不使用 `Atomic`、不使用 `DataCopy` 的 padding 链路做越界读写。

### 6.2 核内主循环（按 packed 区间分核）

```cpp
__aicore__ inline void Process() {
    const int64_t T = tiling.total;
    int64_t cnt  = tiling.base + (GetBlockIdx() < tiling.rem ? 1 : 0);
    int64_t p0   = GetBlockIdx() * tiling.base
                 + (GetBlockIdx() < tiling.rem ? GetBlockIdx() : tiling.rem);
    int64_t p1   = p0 + cnt;
    if (cnt == 0) return;                     // T < blockNum 时多余核直接退出

    int64_t j = FindCol(p0);                  // §4.2 整数二分反解列号
    while (p0 < p1) {
        const int64_t segStart = (tiling.isUpper) ? j * (j + 1) / 2
                                                  : j * (2 * tiling.n - j + 1) / 2;
        const int64_t segLen   = (tiling.isUpper) ? (j + 1) : (tiling.n - j);
        const int64_t begin    = Max(p0, segStart);
        const int64_t end      = Min(p1, segStart + segLen);

        const int64_t srcOff = (tiling.isUpper) ? (j * tiling.lda + (begin - segStart))
                                                : (j * tiling.lda + j + (begin - segStart));
        const int64_t dstOff = begin;         // AP 偏移即 packed 下标
        CopySegment(srcOff, dstOff, end - begin);

        p0 = end; ++j;
    }
}
```

`CopySegment` 按 `tileElems` 切块，单块源/目的均为连续区间：

```cpp
__aicore__ inline void CopySegment(int64_t srcOff, int64_t dstOff, int64_t elems) {
    while (elems > 0) {
        int64_t len = Min(elems, tiling.tileElems);
        LocalTensor<aclblasComplex> in  = inQue.AllocTensor<aclblasComplex>();
        LocalTensor<aclblasComplex> out = outQue.AllocTensor<aclblasComplex>();
        CopyIn(in, srcOff, len);          // MTE2，精确 len 个元素
        inQue.EnQue(in);
        LocalTensor<aclblasComplex> r = inQue.DeQue<aclblasComplex>();
        CopyOut(out, r, dstOff, len);     // MTE3，精确 len 个元素
        outQue.EnQue(out);
        outQue.DeQue<aclblasComplex>();
        outQue.FreeTensor(out);
        srcOff += len; dstOff += len; elems -= len;
    }
}
```

> 也可直接 `DataCopyPad(ubIn, gmIn[srcOff], params)` / `DataCopyPad(gmOut[dstOff], ubOut, params)`，其中 `blockLen = len * 8` 字节，`blockCount = 1`，不读取 `blockLen` 之外的数据。

### 6.3 tilingKey 规划

| tilingKey | `isUpper` | 说明 |
| --- | --- | --- |
| 0 | 0 | LOWER 核函数（列起点 `j·lda + j`，列长 `n − j`） |
| 1 | 1 | UPPER 核函数（列起点 `j·lda`，列长 `j + 1`） |

按 `uplo` 分成两个编译期分支，热循环内不再出现 `uplo` 判断，避免每段都做分支与重复乘法；`n`、`lda` 仍为运行时常量，不生成 `n` 相关的模板组合，控制编译产物规模（与 `aclblasStrttp` 一致）。

### 6.4 边界与小规模处理

- `n = 1`：`T = 1`，`blockNum = 1`，单核单段 1 元素，走标量兜底或 1 个 block 的 `DataCopyPad`。
- `n` 很小（`T < 32`，如 `n = 1, 2, 3, 5`）：`blockNum = min(coreNum, T)`，避免空核；不满足 32 B 整块的部分由 `DataCopyPad` 的 tail 处理路径完成（只搬 `len` 个元素对应字节）。
- 末尾段（`end` 落在列的中间甚至只 1 个元素）与整列段共用同一 `CopySegment`，无特殊分支。

---

## 7 tiling 策略

### 7.1 分核策略：按 packed 元素均分

1. `T = n(n+1)/2`（`int64_t`）。
2. `blockNum = min(aiCoreNum, T)`：`n = 0` 不下发；`n` 很小时不启用空闲核。
3. `base = T / blockNum`，`rem = T % blockNum`；前 `rem` 个核各多处理 1 个元素。核 `b` 的区间为：

   ```text
   p0 = b·base + min(b, rem)
   len = base + (b < rem ? 1 : 0)
   ```

4. 该策略按**字节数**天然均衡（每核搬运量差不超过 1 个元素 = 8 B）；由于 `AP` 目标区间连续，核间不写同一 cache line（相邻核边界处最多共享 1 个 32 B 块导致轻微写合并，无正确性影响）。

### 7.2 UB 切分

- 单个 `aclblasComplex` 为 8 B，32 B 块 = 4 个元素；`tileElems` 取 4 的整数倍，保证除尾段外每块按块对齐。
- double buffer 占用：`4 × tileElems × 8 B`（in 两片 + out 两片）。
  - `tileElems = 2048` → 64 KiB；`tileElems = 4096` → 128 KiB。
- `tileElems` 计算：`tileElems = min(4096, floor(UB可用字节 / (4 × 8)) )`，再向下取 4 的倍数，且不小于 4。UB 大小通过平台信息接口获取，默认按 arch35 实际 UB 容量取上限，保证单核常驻 4 片缓冲。
- 单段长度最大为 `n = 8192` 元素（64 KiB），因此单列最多被切成 2 段（`tileElems = 4096` 时），切块开销可忽略。

### 7.3 只对 host 做一次除法

`base/rem`、`tileElems`、`total` 均在 host 算好放进 `TilingData`；kernel 内每个核只做常数次加减乘与 13 次二分，列内循环只做 `+ len` 推进，`int64_t` 开销可忽略（`n = 8192` 时单核约 205 列）。

---

## 8 内存访问策略

### 8.1 读：仅触及指定三角

| 访问 | 范围 | 字节数 |
| --- | --- | --- |
| `A` 读 | LOWER：列 `j` 的 `[j·lda + j, j·lda + n)`；UPPER：列 `j` 的 `[j·lda, j·lda + j + 1)` | 每列 `L(j)·8` B |
| `A` 总读 | 恰好 `T` 个元素 | `8T` B |
| `A` 未读 | 非指定三角元素、`i ≥ n` 的 `lda` padding 行 | 0 B |

实现上每个 `CopySegment` 的 `srcOff/len` 由三角内边界推导，**不做 32 B 向上取整读**，从而严格满足「另一三角与 padding 行不读」。若平台要求 GM 基址 32 B 对齐，则对段首不足 1 个块的 0～3 个前导元素使用标量 `GetValue`（仅读三角内元素）搬入 UB，再对剩余按块对齐的主体使用 `DataCopyPad`，不引入任何越界读。

### 8.2 写：恰好 packed 长度

- `AP` 写区间为 `[0, T)`，总写 `8T` B；不写尾部哨兵区、不做 32 B 尾部覆盖写。
- 若某段尾部不足 1 块，使用 `DataCopyPad` 的 tail 语义（`blockLen` 精确字节数）或标量 `SetValue` 精确写 `len` 个元素，**禁止按块补齐写**，否则会篡改 `AP[T]` 之后的哨兵值。

### 8.3 流水与带宽

- `DataCopyPad` 单块最大 32 KiB（`tileElems = 4096`），MTE2/MTE3 以块为单位流水，double buffer 使读与写重叠。
- 总搬运字节 `16T`（读 `8T` + 写 `8T`）；`n = 8192` 时为 512 MiB。性能预期由 HBM 带宽主导，标量地址计算不构成瓶颈。
- 不使用 L2 缓存提示、不需要 `DataCacheClean` 之类的额外维护操作（纯搬运，无跨核共享中间结果）。

### 8.4 地址计算防溢出

```cpp
int64_t srcOff = static_cast<int64_t>(j) * tiling.lda + colBase; // j*lda 先转 int64
int64_t dstOff = static_cast<int64_t>(j) * (tiling.n - j) ...;   // 见 §4.1，全 int64
```

`lda` 与 `n` 在 host 已为 `int`，进入 kernel 即提升为 `int64_t`；`j·lda` 最大 `8192×16384 = 1.34×10^8`，在 `int64` 下安全。

---

## 9 校验逻辑

### 9.1 严格顺序

| 步骤 | 条件 | 返回 | 说明 |
| --- | --- | --- | --- |
| ① | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 最优先，保证句柄不可用时不做任何其他判断 |
| ② | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` | 先判 `n`，再判 `uplo`、`lda` |
| ③ | `uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER` | `ACLBLAS_STATUS_INVALID_VALUE` | **不得**返回 `INVALID_ENUM`（对齐 cuBLAS 13.5 `cublasCtrttp`） |
| ④ | `lda < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` | `max(1,n)`：`n ≤ 1` 时为 1，因此 `n = 0, lda = 0` 在此被拒 |
| ⑤ | `n == 0`（且 `lda ≥ 1`） | `ACLBLAS_STATUS_SUCCESS` | quick return：不下发 kernel、不读写 `A`/`AP` |
| ⑥ | `n > 0 && (A == nullptr \|\| AP == nullptr)` | `ACLBLAS_STATUS_INVALID_VALUE` | 指针校验只对 `n > 0` 生效 |

### 9.2 旁路说明

- 不校验 `A`/`AP` 重叠（未定义行为，与 cuBLAS 行为一致），文档与 README 明确提示调用方保证不重叠。
- 不扫描 tensor 元素值（`Inf`/`NaN` 合法），不做 `n > 8192` 的显式拒绝：超出 8192 的组合不在支持/验收范围，README 标注为不支持（若后续要支持更大 `n`，现有 `int64` 设计可直接扩展，无需改偏移逻辑）。
- 不校验设备指针可访问性、不做隐式 stream 同步。

### 9.3 负向用例矩阵

| `handle` | `uplo` | `n` | `lda` | `A`/`AP` | 期望 |
| --- | --- | --- | --- | --- | --- |
| `nullptr` | 任意 | 任意 | 任意 | 任意 | `HANDLE_IS_NULLPTR` |
| 有效 | LOWER/UPPER | −1 | ≥1 | 有效 | `INVALID_VALUE` |
| 有效 | 非法枚举 | 4 | 4 | 有效 | `INVALID_VALUE`（非 `INVALID_ENUM`） |
| 有效 | LOWER | 4 | 3 | 有效 | `INVALID_VALUE` |
| 有效 | LOWER | 0 | 0 | 任意 | `INVALID_VALUE` |
| 有效 | LOWER | 0 | 1 | `nullptr` 亦可 | `SUCCESS`（AP 不被写） |
| 有效 | LOWER | 4 | 4 | `A = nullptr` | `INVALID_VALUE` |
| 有效 | LOWER | 4 | 4 | `AP = nullptr` | `INVALID_VALUE` |
| 有效 | LOWER | 1 | 1 | 有效 | `SUCCESS` |

---

## 10 精度方案

### 10.1 参考实现（golden）

- 使用 **LAPACK 3.12.0 的 `CTRTTP`**（`ctrttp_(uplo, n, A, lda, AP, info)`）生成参考输出；CBLAS 不提供 `cblas_ctrttp`，不得以此为由跳过比对。
- golden 与测试工程一同提交（构建方式、LAPACK 路径与版本写入 README 与自测报告）；或按 §1.1 的列主序公式实现独立的 CPU 参考（逐元素 for 循环，`int64` 偏移）。
- 输入 `A` 与 golden 均以「列主序 `lda × n` + packed `AP`」生成，保证与算子入参布局一致。

### 10.2 判定标准

| 项 | 取值 |
| --- | --- |
| 比对范围 | 仅 `AP[0, n(n+1)/2)`；`A` 非指定三角与 padding 行不参与 |
| 判定粒度 | 实部 `float32`、虚部 `float32` 分别逐元素比较 |
| `rtol` | 0 |
| `atol` | 0 |
| `required_matched_ratio` | 1.0 |
| `max_abs_error_limit` | 0 |
| 等价判据 | `memcmp(AP_actual, AP_golden, 8T) == 0`（按位相等） |

逐元素公式化描述为 `|actual − golden| ≤ atol + rtol·|golden|` 在 `atol = rtol = 0` 下退化为严格相等；同时要求 `matched_ratio = 1.0` 且 `max_abs_error = 0`。

### 10.3 特殊值

- `Inf`：符号必须一致（`+Inf`/`-Inf` 不可互换）；
- `NaN`：要求载荷位与符号位完全一致（按位比较天然满足）；
- 若实现中误引入浮点运算，`NaN` 与 `Inf` 会立刻在按位比较中暴露。

### 10.4 哨兵与边界校验

- 调用前把 `AP` 整块（`n(n+1)/2` 元素 + 尾部 `S` 个哨兵元素，`S` 取 8～32）填为固定哨兵图案；
- 调用并同步后检查：前 `T` 个元素 == golden，`[T, T+S)` 哨兵不变；
- 另用小 `n`（1、2、3、5）做**索引穷举**：逐元素打印 `AP[k] ← A(i,j)` 映射，人工核对 §1.1 公式（LOWER 与 UPPER 各一套）。

---

## 11 性能方案

### 11.1 目标

| 项 | 指标 |
| --- | --- |
| 硬件 | Atlas 950PR（arch35） |
| 单 case | 平均单次耗时 ≤ H100（sm_90）cuBLAS 基线的 **2.5 倍**（即 ≥ 0.4× 标杆性能） |
| 整体 | 全部性能 case 的 `GPU耗时 / NPU耗时` 算术平均 **≥ 0.5** |
| 采样 | 5 次 warmup（不计入）+ ≥ 10 次有效采样取算术平均，单位 us |
| 采集 | `msprof op` 采集单算子 task duration；返回非 `SUCCESS` 的样本剔除并说明 |
| 基线环境 | H100 80GB，CUDA 13.2 / cuBLAS 13.3 / 驱动 ≥ 580.65，参考 `test_cases/gpu_baseline.csv` |

### 11.2 性能用例

| `n` | `lda` | `uplo` | 备注 |
| --- | --- | --- | --- |
| 64 / 256 / 1024 | `max(1,n)`、`2·max(1,n)` | LOWER/UPPER | 小中规模，考察下发与标量开销 |
| 4096 | `max(1,n)`、`max(1,n)+1` | LOWER/UPPER | 混合对齐 |
| 8192 | `n`、`n+3`、`2n` | LOWER/UPPER | 最大规模，带宽主导 |

### 11.3 优化手段与预期

1. **零计算**：只有 MTE2/MTE3，Vector 空闲；单核按 packed 元素均分，负载差 ≤ 1 元素。
2. **大块搬运**：`tileElems = 4096`（32 KiB）块，double buffer 隐藏读写切换；单列 64 KiB 最多 2 块。
3. **少标量开销**：分段边界由 host 预计算，kernel 内每列仅常数次 `int64` 运算 + 13 次二分（仅首元素）。
4. **精确长度写**：不按 32 B 补齐写，避免额外写放大（补写比例对 `n = 8192` 可忽略，但对小 `n` 与尾部影响明显）。
5. **可扩展**：若实测离目标有差距，可将一个核的一次 `DataCopyPad` 提升为多条 MTE 队列批量下发，或按列预取增加 in-queue 深度；极端情况下允许同一核处理多个连续列区间以提升单次搬入长度（`tileElems` 上限受 UB 限制）。

理论下界：读 `8T` B + 写 `8T` B；`n = 8192` 共 512 MiB。算子耗时应接近「2 × packed 字节数 / 有效 HBM 带宽」，与 cuBLAS 的 gather-scatter 实现同量级。

---

## 12 约束限制

| 约束项 | 内容 |
| --- | --- |
| 数据类型 | 仅 `aclblasComplex`（complex64，实/虚各 `float32`）；无 half/bf16/double 版本 |
| 规模 | `n ∈ [0, 8192]`；`n > 8192` 不在支持与验收范围（偏移逻辑本身支持更大 `n`） |
| 前导维 | `lda ≥ max(1, n)`，`lda` 单位是元素；`lda > n` 时第 `n..lda−1` 行不读 |
| 内存重叠 | `A` 与 `AP` 不得重叠，重叠为未定义行为，算子不检测、不报错 |
| 布局 | `A` 必须列主序；不支持行主序、任意 stride、子矩阵视图 |
| 语义 | 非原地（`AP` 独立输出）；不返回视图；不涉及 broadcast、动态 shape、batching |
| 确定性 | 同一设备、同一输入、同一 tiling 多次调用输出逐位一致 |
| 异步 | 结果在 `handle` 绑定的 stream 上异步可见；调用方负责同步后再读 `AP` |
| 设备 | 仅 Ascend 950PR（arch35）构建路径；接口声明跨产品线共用，实现不私有化 |
| 校验 | 非法 `uplo` 返回 `INVALID_VALUE`（非 `INVALID_ENUM`）；顺序固定为 handle → n → uplo → lda → n==0 早返回 → 指针 |
| 空输入 | `n = 0, lda ≥ 1` 为合法 no-op（`SUCCESS`，不写 `AP`）；`n = 0, lda = 0` 返回 `INVALID_VALUE` |
| workspace | 不需要 GM workspace；仅 UB 约 64～128 KiB |
| 依赖 | CANN 9.1.0；不依赖 LAPACK/CBLAS 运行时（golden 仅在测试工程中使用） |

---

## 13 测试方案

### 13.1 框架与位置

- 测试代码：`test/trttp/ctrttp/arch35/`，采用 ops-blas 仓 test 目录的 CSV + C++ GTest 框架，加载 CSV 调用 `aclblasCtrttp`，与 golden 比对。
- 每个用例流程：生成/读入 `A`（含 padding 行填充特殊值以验证「不读 padding」）→ `AP` 填哨兵 → 调接口 → `aclblasStreamSynchronize` → 精度比对 + 哨兵检查 → 记录日志。
- CSV 列建议：`TC_ID, dtype, uplo, n, lda, dist, a, ap, expect_status, golden_file`；性能用例单独一个 CSV（`TC_PF_*`），固定均匀分布。

### 13.2 功能与精度覆盖

| 覆盖点 | 取值 | 期望 |
| --- | --- | --- |
| uplo | `ACLBLAS_LOWER`、`ACLBLAS_UPPER` | 与 LAPACK `ctrttp_` 按位一致 |
| `n = 0` | `lda = 1`、`lda = 8` | `SUCCESS`，`AP` 一个字节都不变 |
| `n = 1` | `lda = 1`、`lda = 2`、`lda = 4` | 单元素正确 |
| 小质数 | `n = 3, 5, 7, 13, 31` | 索引穷举核对 |
| 2 的幂 / ±1 | `n = 64, 127, 128, 129, 1024, 2047, 4096, 8191` | 正确 |
| 最大规模 | `n = 8192`，`lda = 8192 / 8193 / 16384` | 正确，`int64` 偏移无溢出 |
| lda padding | `lda = max(1,n)`、`+1`、`+3`、`2·max(1,n)` | 正确；padding 行填 `NaN`/特殊图案，验证未被读入 |
| 元素特殊值 | 实/虚部分别包含 `±Inf`、`NaN`、`±0`、次正规数 | 按位一致 |
| 非三角区域污染 | `A` 非指定三角填互异的 `NaN` 图案 | `AP` 不受影响 |
| 哨兵 | `AP` 尾部追加固定值 | 不被改写 |
| 边界穷举 | `n = 1, 2, 3, 5`，两种 `uplo` 全部元素 | 逐元素索引与公式一致 |
| 随机分布 | 均匀 `U(-5,5)` 50%、正态（`GAUSS_NORM_5_2`）50%，实/虚独立采样 | 按位一致 |

### 13.3 负向与健壮性用例

- `handle = nullptr` → `HANDLE_IS_NULLPTR`；
- `n = -1`（含 `lda` 合法）→ `INVALID_VALUE`；
- `uplo` 取非法值（如 0xFF、`ACLBLAS_UPPER + 1`）→ `INVALID_VALUE`；
- `lda = max(1,n) − 1`（含 `n = 0, lda = 0`）→ `INVALID_VALUE`；
- `n > 0` 且 `A = nullptr` / `AP = nullptr` → `INVALID_VALUE`；
- `n = 0, lda = 0` → `INVALID_VALUE`（重点回归，不得沿袭实数版先返回 `SUCCESS` 的行为）；
- `A` 与 `AP` 重叠：作为健壮性观察用例，不判定结果正确性，仅确认无越界/崩溃，并在报告中说明为未定义行为；
- 校验顺序用例：如 `n = -1` 且 `uplo` 非法，期望先命中 `n < 0`；`handle = nullptr` 且其余全非法，期望先命中 `HANDLE_IS_NULLPTR`。

### 13.4 性能用例

- 规模与 `uplo`/`lda` 组合见 §11.2；每个 case：5 次 warmup + ≥10 次有效采样，`msprof op` 采集 task duration，取平均；
- 记录 NPU 实测、GPU 基线（`gpu_baseline.csv`）、倍率 `GPU/NPU`；
- 汇总生成各 case 单次耗时与平均倍率，输出性能自测报告与截图。

### 13.5 交付与复现

- 测试目录 README 写明：环境（CANN 9.1.0、950PR）、LAPACK 3.12.0 golden 的构建方式、CSV 生成（`gen_csv.py`）、编译命令、执行命令（含 `--gtest_filter` 示例）、精度/性能日志位置；
- 自测报告记录：用例总数、精度通过率（要求 100% 按位一致）、性能表与截图、异常样本剔除说明；
- 提交前在 `task_submission/` 目录按模板准备：`1 自验证步骤说明.md`、精度/性能/内存报告与日志（内存项按“不涉及”说明）。

---

## 附录 A 与实数版本 `aclblasStrttp` 的差异清单

| 项 | `aclblasStrttp` | `aclblasCtrttp`（本设计） |
| --- | --- | --- |
| dtype | `float32`（4 B） | `aclblasComplex`（8 B） |
| 单块元素数 | 32 B / 4 B = 8 | 32 B / 8 B = 4 |
| 偏移计算 | 可用 `int32` | 必须 `int64_t` |
| `n = 0, lda = 0` | 实数版先返回 `SUCCESS`（不复用） | 返回 `INVALID_VALUE`（step ④ 拦截） |
| 精度 | 按位相等 | 实部/虚部分别按位相等，`Inf`/`NaN` 载荷保留 |
| 其余（接口形态、校验顺序、tiling、分核、测试框架） | — | 与实数版同构，直接复用工程骨架 |
