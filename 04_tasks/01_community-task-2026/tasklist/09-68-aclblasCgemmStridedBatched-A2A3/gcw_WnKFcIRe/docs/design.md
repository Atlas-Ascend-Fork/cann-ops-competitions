# aclblasCgemmStridedBatched 算子设计文档

本文档描述 `aclblasCgemmStridedBatched`（complex64 跨步批量矩阵乘法）在 Atlas A2/A3（arch22 /
DAV_2201）上的设计、算法、精度与验证方案，并给出实测数据与已知限制。

---

## 1. 算子概述

### 1.1 数学定义

对每个 batch 独立执行：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
A_i = A + i * strideA
B_i = B + i * strideB
C_i = C + i * strideC
i = 0, 1, ..., batchCount - 1
```

- `op(A_i)` 为 `m x k`，`op(B_i)` 为 `k x n`，`C_i` 为 `m x n`，全部列主序、complex64；
- stride 单位为 **complex 元素个数**而非字节；所有 batch 共用同一组
  `m/n/k/alpha/beta/transa/transb`；
- `op(X)` 支持 N/T/C，`ACLBLAS_OP_C` 为共轭转置（复数下与 `ACLBLAS_OP_T` 必须区分）；
- `strideA = 0` / `strideB = 0` 表示所有 batch 广播复用同一输入矩阵（本仓扩展语义）；
- 不支持负 stride；`strideC = 0 && batchCount > 1` 语义未定义。

### 1.2 接口

```cpp
aclblasStatus_t aclblasCgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    int64_t strideA, const aclblasComplex* B, int ldb, int64_t strideB,
    const aclblasComplex* beta, aclblasComplex* C, int ldc, int64_t strideC, int batchCount);
```

`alpha`/`beta` 为 Host 指针，A/B/C 为 Device 指针，任务异步提交到 handle 绑定的 stream。

### 1.3 返回值约定（校验顺序即接口契约）

| 条件 | 返回值 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `transa/transb` 非法 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `m/n/k/batchCount < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1, transa==N ? m : k)`、`ldb < max(1, transb==N ? k : n)`、`ldc < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha == nullptr` 或 `beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `m == 0 \|\| n == 0 \|\| batchCount == 0` | 直接 `SUCCESS`，不访问任何矩阵指针 |
| `k > 0 且 alpha != 0` 时 `A/B == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 输出 shape 非空时 `C == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

A/B 只在 alpha 项真正被求值（`k > 0 && alpha != 0`）时才要求非空；C 在输出 shape 非空时总是被引用。

---

## 2. 硬件约束与关键设计前提

以下事实与仓内已合入的 arch22 复数 BLAS-3 算子（csymm / csyrk，见
`blas/common/helper/complex_blas3_arch22.h`）共用，均在开发期用独立探针核在本工具链
（bisheng + CANN 9.0，dav-2201）上逐一验证，构成硬前提：

| # | 事实 | 对设计的影响 |
| --- | --- | --- |
| 1 | Cube 单元只有实数乘加，无复数指令 | 复数 GEMM 必须分解为实数 GEMM；分解方式直接决定精度（见 4.1） |
| 2 | `DataCopyPad` 连续读/写正确，但"跨步取元素"类读法（多块 + `srcStride`、逐元素、`GlobalTensor::GetValue` 批量）在本工具链上不可用或不可靠 | Phase 0 的所有搬入/搬出都按"一条存储列 = 一段连续 run"组织，只依赖连续 burst |
| 3 | 连续读 + `GatherMask` 按 256B repeat 拆分实/虚平面正确 | 平面拆分以整 repeat 为最小单位 |
| 4 | Matmul 内部按行主序组织操作数，用户矩阵是列主序 | 通过计算 `C^T = op(B)^T * op(A)^T` 把"列主序的一列"映射为"行主序的一行"，不做全矩阵转置（见 3.2） |
| 5 | Matmul 静态 tiling 约束：单核 `singleK = 8192` 槽，tile 128x128，baseK 64 | K 方向按 launch 分块（见 5.1），temp 行宽 128 对齐 |
| 6 | UB 向量操作数必须 32B 对齐；GM 侧 `DataCopyPad` 无对齐要求 | workspace 段全部 512B 对齐；temp 行宽 `tempLdc = align128(m)`（float 单位）保证每行起点对齐 |
| 7 | aicore 函数内整型与浮点互转受限 | 长度/偏移全整型计算，浮点标量经 tiling 参数传入 |

---

## 3. 总体架构

### 3.1 执行路径

```text
aclblasCgemmStridedBatched
   |
   +-- 参数校验 / quick return（1.3 的顺序）
   |
   +-- m=0 || n=0 || batchCount=0  -> no-op
   |
   +-- k=0 || alpha=0              -> C_i = beta * C_i（ScaleAllBatches）
   |       |-- beta = 0 : 逐 batch 异步 memset 清零（ldc > m 时逐列）
   |       |-- beta = 1 : no-op
   |       `-- 其他     : 复用 combine kernel，skipAlphaTerm=1
   |
   `-- 常规路径（两种调度形态，见下）
             Phase 0 (AIV) split/run : 复数操作数 -> K 交错实数槽位缓冲
             Phase 1 (AIC) gemm x2   : L * R1 = Re(C^T)，L * R2 = Im(C^T)
             Phase 2 (AIV) combine   : C = alpha*(Pr + i*Pi)*2^8 + beta*C_old
```

常规路径有两种调度形态，由 workspace 预算决定：

- **串行形态**（batchCount == 1，或展平后 workspace 超过 1 GiB 预算）：逐 batch 顺序提交到 handle
  stream，batch 间复用同一份 workspace。Host 侧不做任何同步：第 `i` 个 batch 的 combine 与第 `i+1`
  个 batch 的 pack 在同一条 stream 上顺序提交，stream 语义自动串行化。
- **展平形态**（batchCount > 1 且展平后 workspace <= 1 GiB）：每个 workspace 段按 batchCount 份
  分配，每个阶段只发**一次** launch，kernel 内部把 (batch, 工作项) 展平成一维 job 循环
  （split/run/combine 按"列 x batch"，gemm 按"tile x batch"）。小 shape 大 batch 场景同时消除
  两类开销：每 batch 4~5 次 launch 的 Host 提交开销，和单 batch tile 数过少导致的 AIC 利用率
  不足。数值与串行形态完全一致：每个 (batch, tile) job 仍在单个 block 内串行走完自己的 K 分块链，
  分块边界和累加次序不变。

### 3.2 列主序映射：计算 C^T

用户矩阵是列主序，Matmul 按行主序组织操作数。本设计计算

```text
C^T = op(B)^T * op(A)^T    （n x m 行主序 temp）
```

使得 **C 的一列（列主序下连续）恰好是 temp 的一行（行主序下连续）**。Phase 2 的 combine 因此可以
按"一整列连续读 temp、连续写 C"组织，不做任何转置搬运。GEMM 自身的维度记号随之交换：
`gemm.m = 逻辑 n`，`gemm.n = 逻辑 m`，`gemm.k = 2k`（槽位数）。

### 3.3 workspace 布局（字节，均 512B 对齐）

```text
tempPr : n x tempLdc 个 float   Re(C^T) 乘积，tempLdc = align128(m)
tempPi : n x tempLdc 个 float   Im(C^T) 乘积
packA0 : 2k x m 个 float        右操作数 R1 槽位缓冲（op(A)=C 时用作 R2）
packA1 : 2k x m 个 float        右操作数 R2 槽位缓冲（仅 op(A)=N 时分配）
packB  : 2k x n 个 float        左操作数槽位缓冲，或 (op(A)=C, op(B)=N) 角落的 B 缩放拷贝
```

串行形态下尺寸只依赖算子维度，在 batch 间复用；展平形态下每个段按 batchCount 份分配（段内
stride 见 tiling 的 `*BatchStride` 字段，单位 float），展平总量上限 1 GiB。分配前做 64 位溢出
检查，失败返回 `ALLOC_FAILED`。零拷贝快路径命中时对应缓冲不分配（见 4.3）。

### 3.4 并行度

| 阶段 | 核类型 | blockDim（串行 / 展平） |
| --- | --- | --- |
| split / run（Phase 0） | AIV | `min(aivCoreNum, 列数)` / `min(aivCoreNum, 列数 x batch)` |
| gemm（Phase 1） | AIC | `min(mBlocks*nBlocks, aicCoreNum)` / `min(mBlocks*nBlocks x batch, aicCoreNum)`，tile 128x128 |
| combine（Phase 2） | AIV | `min(aivCoreNum, n)` / `min(aivCoreNum, n x batch)` |

### 3.5 文件清单（arch22）

| 文件 | 职责 |
| --- | --- |
| `blas/gemm_strided_batched/arch22/cgemm_strided_batched_host.cpp` | 校验、路径选择、workspace 规划、tiling 构造、逐 batch launch |
| `blas/gemm_strided_batched/arch22/cgemm_strided_batched_kernel.cpp` | split / run / gemm / combine 四个 kernel |
| `blas/gemm_strided_batched/arch22/cgemm_strided_batched_kernel.h` | kernel launcher 声明（host/kernel 两个 TU 共用） |
| `blas/gemm_strided_batched/arch22/cgemm_strided_batched_tiling_data.h` | tiling 结构（按值传入 kernel，两侧字节布局必须一致） |
| `test/gemm_strided_batched/cgemm_strided_batched/` | CSV 驱动测试 + golden（见第 7 节） |

---

## 4. 复数乘法的实数化分解

### 4.1 K 交错槽位：把减法放进同一条 Cube 累加链

记 `op(A)^T = Ar2 + i*Ai2`、`op(B)^T = Br2 + i*Bi2`（均为实矩阵），则

```text
Re(C^T) = Br2*Ar2 - Bi2*Ai2
Im(C^T) = Br2*Ai2 + Bi2*Ar2
```

若把 `ac - bd` 拆成两次独立实数 GEMM 再相减，两个 O(k) 量级的结果各自在 FP32 中舍入后再相减
得到 O(sqrt(k)) 量级结果时，相对误差会被放大约一个数量级。本设计改为 **K 交错槽位配对**：
归约维的槽位 `2q` 承载操作数第 `q` 列的实部平面，槽位 `2q+1` 承载虚部平面，两个实数 GEMM 的
归约维均为 `2k`：

```text
L (n x 2k)   : L[j][2q] = Br2(j,q)，L[j][2q+1] = Bi2(j,q)
R1 (2k x m)  : R1[2q][i] = Ar2(q,i)，R1[2q+1][i] = -Ai2(q,i)
R2 (2k x m)  : R2[2q][i] = Ai2(q,i)，R2[2q+1][i] =  Ar2(q,i)
```

于是 `L*R1` 的第 `q` 对槽位把 `Br2*Ar2` 与 `-Bi2*Ai2` 放进**同一条 Cube 累加链**（L0C 中 FP32
累加），相消项在累加器内部相遇，而不是两个独立舍入的和相减。实部精度由此回到正常 FP32 累加
水平。虚部 `L*R2` 的两项本来就是相加，同样受益。该结构与仓内 csymm/csyrk 一致，共享
`complex_blas3_arch22.h` 的 `BuildInterleaveIndex` / `GatherMask` 拆分等机制。

### 4.2 Phase 0 的两个打包 kernel

复数元素在 GM 中就是 `re, im` 相邻的两个 float，因此**一条存储列本身就是一对交错槽位**，
打包只是按转置组合做符号/交换/拆分：

- **split（`cgemm_sb_split_slots_kernel_do`）**：把列主序复数矩阵拆成 K 交错实数槽位缓冲。
  存储列 `c` 喂给相邻两个槽位行：`2c / 2c+1 = re / ±im`（dst1），可选再发一份
  `im / re`（dst2，`dual=1`，仅 A 侧需要 R1、R2 两份时用）。每次读、每次写都是一条连续 burst。
- **run（`cgemm_sb_run_kernel_do`）**：按整条存储列做变换，输出行 `i` 是第 `i` 列导出的交错 run：
  dst1 为 `[re, ±im]`（普通交错，R1^T），dst2 为 `[±im, re]`（交换交错，R2^T）。`negImR1=0` 时
  dst1 退化为纯缩放拷贝，供 `(op(A)=C, op(B)=N)` 角落缩放 B 使用。

### 4.3 各转置组合的槽位来源与零拷贝快路径

| op(A) | R1 来源 | R2 来源 | op(B) | L 来源 |
| --- | --- | --- | --- | --- |
| N | split 出的 `[Ar2, -Ai2]` | split 出的 `[Ai2, Ar2]` | N | **raw B**（行步长 `2*ldb`），不打包 |
| T | run 出的 `[re, -im]` | run 出的 `[im, re]` | T | split 出的 `[re, im]` 槽位 |
| C | **raw A**（行步长 `2*lda`） | run 出的 `[-im, re]` | C | split 出的 `[re, -im]` 槽位 |

两条零拷贝快路径：

- **op(B)=N**：存储的 B 列本身就是 `[re, im]` 交错槽位对，直接把调用方的 B 以行步长 `2*ldb`
  交给 GEMM 当左操作数，完全不做打包；
- **op(A)=C**：共轭恰好抵消槽位取负，R1 就是 raw A（行步长 `2*lda`），只打包 R2。

注意一个由此引入的 tiling 细节：op(A)=C 时实部 GEMM 的右操作数行步长是 `2*lda` 而虚部 GEMM
的 R2 是紧凑打包的 `2k`，两者不同，因此 host 为虚部 GEMM 准备独立 tiling 副本
`gemmR2.ldB = kSlots`（否则 `lda > k` 时 R2 会按错误步长寻址，这是开发期 41 个 op(A)=C 用例
失败的根因）。kernel 侧 `TRANS_BOTH` 分支的 `orgKb` 同样必须使用 `ldB` 而非 `k`。

### 4.4 范围缩放（2^-8）

每个 GEMM 的**恰好一侧**操作数在 Phase 0 乘以 `2^-8`（`CBLAS3_RANGE_SCALE_DOWN`），Phase 2 的
combine 乘回 `2^8`。这是 2 的幂，缩放本身无舍入误差，作用是把 Cube 长链累加的中间量级压到低
动态区间，避免极端填充下累加器溢出或吞没小项。位置规则：op(A) 为 N/T 时缩放打包的 A 侧；
op(A)=C 时 R1 是 raw A 不能改动，缩放改由 B 侧携带（op(B)=N 时 B 侧做一次缩放拷贝，其余走
split 时顺带缩放）。

---

## 5. GEMM kernel 与 K 分块

### 5.1 单次 launch 的 K 上限

GEMM kernel 的编译期 `singleK = 8192` 槽（`complex_blas3_arch22.h`），但 host 实际按
`kChunk = min(singleK, 2048)` 槽分块提交：每个 chunk 一次 launch，首块 `enAtomic=0` 覆盖写
temp，后续块 `enAtomic=1` 原子累加。chunk 在同一条 stream 上顺序提交，原子累加次序确定，结果
可复现。

分块的原因是**串行累加链的 FP32 误差随链长线性增长**：每个 chunk 边界结束一条 Cube 串行
累加链，部分和以一次原子加并入 temp。验收集中最坏 shape（k=4096，即 8192 槽）单链实测
maxAbsErr = 1.0010e-2，恰好擦过验收硬门限 1e-2（对 fp64 累加 golden 测得）；把链四等分后
误差降到 3~4e-3，余量约 3 倍。

### 5.2 transMode

`CgemmSbGemmTilingData.transMode` 标记两侧操作数是否以转置形式存储（K x extent 而非
extent x K），kernel 据此设置 Matmul 的 `SetTensorA/SetTensorB` 转置标志与 orgShape：
op(B)=N 时左侧非转置、否则转置；op(A)=N 时右侧非转置、否则转置（含 `TRANS_BOTH` 分支
`orgKb = ldB` 的修正，见 4.3）。

---

## 6. 边界路径

| 场景 | 行为 |
| --- | --- |
| 零维（m/n/batchCount 任一 0） | quick return，不访问任何指针 |
| `k=0` 或 `alpha=0`，`beta=0` | 逐 batch 异步 memset 清零逻辑区（`ldc > m` 时逐列清，避免触碰 padding 之外的内存） |
| `k=0` 或 `alpha=0`，`beta=1` | no-op |
| `k=0` 或 `alpha=0`，一般 beta | 复用 combine kernel，`skipAlphaTerm=1`，temp 指针悬空不引用 |
| 一般 beta | combine 内完成 `alpha*(Pr+i*Pi)*2^8 + beta*C_old`；`beta=0` 时不读旧 C |
| `strideA/strideB = 0` | 自然复用 batch 0 基址，无特判 |

---

## 7. 测试与精度验证

### 7.1 验收标准

`test/frame/verify.h` 的 `MixedToleranceStrategy`：COMPLEX64 按 FLOAT32 判定，
rtol = atol = 2^-13，matched_ratio >= 0.99，且任一元素的绝对误差不得超过
`max(1e-2, 32*ULP)`（硬门限，单个元素超限即 FAIL）。

### 7.2 golden 的精度要求

串行 fp32 cblas golden 在 k >= 3072、正负 5 均匀填充下自身误差即超过 1e-2 门限（此时测得的
"算子误差"其实是 golden 的误差）。golden 因此实现为 **fp32 分块 cgemm（K chunk = 256）+
fp64 累加**（`test/gemm_strided_batched/cgemm_strided_batched/cgemm_strided_batched_golden.h`）。
分块偏移按列主序的转置方向取值：A 侧为 `isTransA ? k0 : k0*lda`，B 侧为
`isTransB ? k0*ldb : k0`（写反会越界）。

### 7.3 harness 的负例构造

`TC_ED_147/148/149`（handle_null / null_alpha / null_beta）要求真正传空指针。CsvDriven 驱动
按用例 description 特判：`handle_null` 传 `nullptr` handle，`alphaNull/betaNull` 传
`nullptr` 标量指针，否则这三个用例永远走不到目标分支。

### 7.4 用例与实测结果

CSV 共 1200 例，覆盖 SQ（方阵扫描）、EX（极端 shape/转置组合/ld padding）、ED（错误注入）、
PF（性能 shape，最大 batch=4、m=n=k=4096）。在 910B4-1（CANN 9.0.0）上全量
**1200/1200 通过**（总时长约 2.5 h，golden 的 CPU fp64 参考是瓶颈）。最坏点实测：

| 用例 | shape | maxAbsErr（real / imag） | 门限 |
| --- | --- | --- | --- |
| TC_PF_1005 | 4096x4096x4096 | 3.30e-3 / 3.69e-3 | 1e-2 |
| TC_PF_1025 | 2048x2048x4096 | 3.42e-3 / 2.93e-3 | 1e-2 |

---

## 8. 性能设计与实测

### 8.1 任务书门槛与实测结果

任务书性能门槛（Atlas 800T A2 / 910B3，msprof op 口径）。下表实测自 910B4-1（CANN 9.0.0），
每次调用 warmup 5 次后取 20 次设备同步平均：

| case | shape（transa/transb/stride） | 门槛（us） | 优化前（us） | 展平后（us） | 结论 |
| --- | --- | --- | --- | --- | --- |
| 1 | 256^3 x 8，NN，紧凑 | 153.7 | 237 | **98** | 通过（63%） |
| 2 | 512^3 x 32，NN，紧凑 | 2757 | 1824 | **1532** | 通过（56%） |
| 3 | 1024^3 x 16，NT，紧凑 | 10061 | 4743 | **4215** | 通过（42%） |
| 4 | 2048^3 x 8，TN，padding | 36073 | 22927 | **22454** | 通过（62%） |
| 5 | 4096^3 x 4，NN，padding | 143326 | 51022 | 51097（串行形态） | 通过（36%） |

case 1 优化前的瓶颈是 Host 提交开销：每 batch 4 次 launch x 8 batch = 32 次，每次约 6~7us，
合计约 200us，远超实际计算。展平形态把 launch 数降到每次调用 4 次（split + gemm x2 + combine），
并把 GEMM grid 从 4 tile 提升到 32 job，AIC 利用率同步改善。case 5 超出展平预算保持串行形态，
本身余量充足。

### 8.2 性能设计要点

- 批间/段内 workspace 复用或一次性分配，不随调用次数增长；
- pack 按"列 x batch"并行，每列一次连续读 + 连续写；
- 两次 AIC GEMM 复用同一 kernel 族与同一份静态 tiling，无 per-shape 重编译；
- combine 按"列 x batch"并行，写回是一次连续 burst；
- 零拷贝快路径（op(B)=N、op(A)=C）在小 shape 下直接省掉整段 pack。

---

## 9. 演进与已知限制

- 本实现（C^T 映射 + K 交错槽位 + 零拷贝快路径）取代了早期 plain+partner 行打包方案，后者在
  通用 ld padding 与 op(A)=C 组合下存在寻址缺陷，且无法利用零拷贝路径。
- K 分块固定 2048 槽是按验收集最坏 shape 标定的保守值；更细的链长-误差模型留作后续优化。
- 展平形态受 1 GiB workspace 预算限制，超大 shape x 大 batch 仍走串行形态（launch 开销相对
  计算量可忽略）。
- 负 stride、`strideC=0 && batchCount>1` 不支持或语义未定义（见 1.1）。
