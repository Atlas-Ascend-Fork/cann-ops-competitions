# 【社区任务】aclblasStrsmBatched 算子设计文档（A2/A3）

# 需求背景（required）

## 需求来源

本需求来自 9 月社区任务《aclblasStrsmBatched 算子开发（A2/A3）》。任务要求在 Atlas 800I A2 / Atlas 800I A3（arch22）上基于 Ascend C，于 `cann/ops-blas` 开源仓实现单精度实数批量三角求解接口 `aclblasStrsmBatched`，完成设计、开发、精度/性能自验，并以 PR 合入 `blas/trsmbatched/arch22/` 与 `test/trsmbatched/strsmbatched/arch22/`。

- **上游仓**：https://gitcode.com/cann/ops-blas
- **设计文档仓**：https://gitcode.com/cann/cann-ops-competitions
- **对标接口**：cuBLAS `cublasStrsmBatched`
- **Golden**：cblas（Netlib BLAS）`strsm` 逐 batch
- **验收环境**：CANN 9.1.0；精度建议 910B3 或 A3；性能建议 Atlas 800T A2（910B3）

## 背景介绍

### aclblasStrsmBatched 算子功能

对一批相互独立、尺寸一致的三角线性系统做批量求解（Batched Triangular Solve）。所有矩阵列主序存储。每个 batch `i`（`i = 0 .. batchCount-1`）独立求解，结果写入独立输出 `C[i]`（离席，不覆盖 `B[i]`）：

```text
side = LEFT :  op(A[i]) * X[i] = alpha * B[i] ,  C[i] = X[i]
side = RIGHT:  X[i] * op(A[i]) = alpha * B[i] ,  C[i] = X[i]
```

`op(A)` 由 `trans` 决定：`ACLBLAS_OP_N` → `A`，`ACLBLAS_OP_T` → `Aᵀ`。本接口为实数算子，**不支持** `ACLBLAS_OP_C`（非法枚举，返回 `ACLBLAS_STATUS_INVALID_VALUE`，与 cuBLAS 实数档"OP_C 等价 OP_T"刻意收紧）。

### 与仓内 arch35 实现的关系

主仓已有 `blas/trsmbatched/arch35/` 的 950 实现（SIMT panel 回代 + AIC-only GEMM 更新的分块方案），公共声明目前为 cuBLAS 风格的原地覆写（解 `X` 写回 `B`，无 `Carray/ldc`），且 `trans` 接受 `OP_C`：

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* const A[], int lda,
    float* const B[], int ldb,
    int batchCount);
```

本任务书将语义扩展为 **B 只读、C 独立输出**，收紧 `trans`（禁止 `OP_C`），并把 `batchCount=0` 定义为合法 no-op。A2/A3 实现必须按任务书签名落地；分块求解的调度思想（panel 回代 + GEMM 更新）可以复用，但需要按 A2（910B3，24 AIC / 48 AIV、L0A/L0B=64KB、L0C=128KB）与 batch 维度重新设计分核与 tiling。

### 任务书接口（A2/A3 真值）

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const float* alpha,
    const float* const* Aarray, int lda,
    const float* const* Barray, int ldb,
    float* const* Carray, int ldc,
    int batchCount);
```

| 参数 | 输入/输出 | 说明 | 约束 / 异常 |
| --- | --- | --- | --- |
| handle | 输入 | ops-blas 句柄，携带 stream | nullptr → `HANDLE_IS_NULLPTR` |
| side | 输入 | LEFT：A 为 m×m；RIGHT：A 为 n×n | 非法枚举 → `INVALID_VALUE` |
| uplo | 输入 | UPPER / LOWER，只引用对应三角 | 非法枚举 → `INVALID_VALUE` |
| trans | 输入 | N / T（实数，禁止 C） | 非法枚举（含 OP_C）→ `INVALID_VALUE` |
| diag | 输入 | NON_UNIT 用实际对角；UNIT 对角视为 1，不读对角存储 | 非法枚举 → `INVALID_VALUE` |
| m, n | 输入 | C 的行/列 | <0 → `INVALID_VALUE`；=0 合法 no-op |
| alpha | 输入 | Host FLOAT32 标量指针（较 cuBLAS 收紧，仅 Host） | nullptr → `INVALID_VALUE` |
| Aarray | 输入 | Device 三角矩阵指针数组 | nullptr 或元素 nullptr → `INVALID_VALUE`（alpha=0 时不引用 A） |
| lda | 输入 | A 前导维 | LEFT：`lda ≥ max(1,m)`；RIGHT：`lda ≥ max(1,n)` |
| Barray | 输入 | Device 右端矩阵指针数组，只读 | nullptr 或元素 nullptr → `INVALID_VALUE`（alpha=0 时不引用 B） |
| ldb | 输入 | B 前导维 | `ldb ≥ max(1,m)` |
| Carray | 输出 | Device 结果矩阵指针数组，独立输出 | nullptr 或元素 nullptr → `INVALID_VALUE` |
| ldc | 输入 | C 前导维 | `ldc ≥ max(1,m)` |
| batchCount | 输入 | 批量个数 | <0 → `INVALID_VALUE`；=0 合法 no-op |

返回值与 `include/cann_ops_blas_common.h` 一致。

# 需求分析（required）

## 需求描述

在 ops-blas 的 arch22 路径上实现 `aclblasStrsmBatched`：句柄式直调 NPU kernel，覆盖 LEFT/RIGHT × UPPER/LOWER × N/T × UNIT/NON_UNIT 全枚举组合，支持运行时 `m/n/batchCount` 与 lda/ldb/ldc padding，离席写出 C，满足 FLOAT32 混合容差精度与任务书 5 条性能 case（达标 630.7us ~ 14361us）。

## 需求拆解

1. 公共头文件按任务书提供带 `Carray/ldc` 的 A2/A3 签名；arch35 原地签名保留，二者以函数重载共存（参数个数不同，C++ 重载无歧义），不破坏 950。
2. Host 完成全部合法性校验与 no-op / `alpha=0` 快路径，kernel 不做业务校验。
3. Kernel 按 batch 维度并行分核，batch 间独立：尺寸、指针、结果互不混淆；`C` 与 `A`/`B` 不允许内存重叠。
4. `ACLBLAS_DIAG_UNIT` 跳过对角存储，对角乘子视为 1。
5. 路径分级：小矩阵走 AIV SIMT 三角回代；大矩阵走分块 TRSM（AIV panel 回代 + AIC Cube GEMM 更新），以覆盖 2048/4096 性能 case。
6. RIGHT 方向复用 LEFT 求解核（设备侧转置 RHS 的等价变换），避免第二套四枚举×三角方向实现。
7. 测试落入 `test/trsmbatched/strsmbatched/arch22/`，CSV 驱动 + GTest，golden 为 cblas `strsm` 逐 batch，负向用例对齐任务书错误码。
8. README 产品支持表标注 Atlas A2/A3（含 800I A2/A3）：支持。

# 详细设计（required）

## 算子分析

### 数学公式

令 `op(A)` 为 `A` 或 `Aᵀ`，对每个 batch `i` 独立求解三角系统：

- LEFT + NON_UNIT：对 `B[i]` 逐列做前代（LOWER）或回代（UPPER），乘子取 `A[i]` 实际对角；`trans=T` 时按对偶三角访问（`op(A)=Aᵀ` 的 UPPER 等价于 `A` 的 LOWER 列方向访问）
- LEFT + UNIT：乘子恒为 1，不读取 `A[i]` 对角元
- RIGHT：等价变换 `X·op(A)=αB ⟺ op(A)ᵀ·Xᵀ=α·Bᵀ`，即把 `B` 转置后做 LEFT 求解再转置回来；`op(A)ᵀ` 的三角方向与 `trans` 取反对应
- `alpha=0`：不读 `A[i]`、`B[i]`，`C[i]` 的有效 `m×n` 区域（按 ldc）置零

列主序地址：`A(row,col) = A + row + col*lda`（B/C 用 ldb/ldc）。

### 支持数据类型

仅 FLOAT32。alpha 仅 Host 指针，kernel 入参按值传递。

### 支持形状

- `m ≥ 0`、`n ≥ 0`、`batchCount ≥ 0`（运行时 Host 标量，无 dynamic shape 约束）
- LEFT：`A[i]` 为 `lda × m` 物理存储中的 `m×m` 三角；RIGHT：`lda × n` 中的 `n×n`
- `B[i]`、`C[i]` 为 `ldb/ldc × n` 中的 `m×n`，允许前导维 padding
- 无 broadcast；不支持超出前导维语义的非连续访问

## 算子实现

### 工程落点

```text
ops-blas/
├── include/cann_ops_blas.h                 # 增加带 Carray/ldc 的 A2/A3 重载声明
├── blas/trsmbatched/
│   ├── README.md                           # 产品支持表补 Atlas A2/A3：支持
│   └── arch22/
│       ├── strsmbatched_host.cpp           # 校验/快路径/batch 分核/调度
│       ├── strsmbatched_kernel.cpp         # SIMT panel + transpose/scale/zero + gemm/axpy
│       ├── strsmbatched_kernel.h
│       └── strsmbatched_tiling_data.h
└── test/trsmbatched/strsmbatched/
    ├── strsmbatched_golden.h               # cblas strsm 逐 batch（复用）
    ├── strsmbatched_param.h                # 增补 carray/ldc 列
    └── arch22/
        ├── strsmbatched_npu_wrapper.h      # 调任务书签名（含指针数组 H2D 处理）
        ├── strsmbatched_test.cpp           # GTest 用例
        └── strsmbatched_test.csv           # 任务书配套 1200 条 + 缺口自补
```

开发在个人 fork 的 `feat/aclblas-strsmbatched-arch22` 分支进行；合入目录与任务书第 5 节一致。

### Host 侧设计

1. **参数校验顺序**（先 handle 后枚举后维度，与任务书异常列对齐）：
   - `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
   - `side/uplo/trans/diag` 非法（trans 仅允许 N/T）→ `INVALID_VALUE`
   - `m < 0 || n < 0 || batchCount < 0` → `INVALID_VALUE`
   - `alpha == nullptr` → `INVALID_VALUE`
   - 前导维：LEFT 时 `lda ≥ max(1,m)`，RIGHT 时 `lda ≥ max(1,n)`；`ldb ≥ max(1,m)`；`ldc ≥ max(1,m)`
   - `Aarray/Barray/Carray == nullptr` → `INVALID_VALUE`（`alpha≠0` 时还校验 Aarray；batch 内元素指针的 nullptr 在 Device 端 kernel 逐 batch 判空上报，避免 Host 侧 D2H 同步破坏异步语义——见第 4 点）
2. **快路径（不 launch 求解 kernel）**：
   - `m==0 || n==0 || batchCount==0`：合法 no-op，返回 SUCCESS
   - `alpha==0`：专用 zero kernel 按 batch 将 C 的有效区域置零，不读 A/B
3. **指针数组处理**：`Aarray/Barray/Carray` 是 Device 上的 `float*` 数组。kernel 侧直接按 batch 下标解引用 Device 指针数组（`__gm__ float* const*`），不拷回 Host——保证异步语义并消除 `batchCount×8B` 的 D2H 同步开销（相比 arch35 单 batch 循环 launch 的 Host 侧指针回读，这是 batched 接口的正确形态）。
4. **异步**：全部 kernel 下发到 `handle` 绑定的 stream，Host 不主动同步。
5. **batch 分核（与 arch35 单批循环的本质区别）**：求解 kernel 一次 launch 覆盖全部 batch：
   - `unitCount = batchCount × unitsPerBatch`（unitsPerBatch 为 batch 内并行单元数，见路径设计）
   - `usedCores = min(aivCoreNum, unitCount)`，核 `c` 处理 `u = c, c+usedCores, c+2·usedCores, ...`
   - `u → (batchIdx = u / unitsPerBatch, unitInBatch = u % unitsPerBatch)`，每 batch 的 `Aarray[batchIdx]/Barray[batchIdx]/Carray[batchIdx]` 独立取基址
   - batch 小而矩阵大时 unitsPerBatch 增大（按 RHS 列/panel 切），batch 大而矩阵小时 unitsPerBatch=1，总是打满 48 AIV

6. **路径选择（tilingKey / host 分发）**：

| 条件 | 路径 | 覆盖 |
| --- | --- | --- |
| `max(m,n) ≤ 64` 或整题无法填满 Cube 最小块 | AIV SIMT 三角回代（一次到位） | 精度用例绝大多数、case 1 小批 |
| LEFT 且 `m > 64` | 分块 TRSM：AIV panel 回代 + AIC GEMM 更新 | case 2/4 |
| RIGHT 且 `n > 64` | 设备转置 RHS → LEFT 分块 → 转置回（arch35 同款变换，batch 并行化） | case 3/5 |

### Kernel 侧设计

**kernel 家族**（沿用 arch35 已验证的 kernel 拆分，按 batch 扩展）：

| kernel | 核型 | 职责 |
| --- | --- | --- |
| `strsmbatched_zero_kernel` | AIV | alpha=0 时逐 batch 置零 C |
| `strsmbatched_scale_kernel` | AIV | `alpha≠1` 时 `B→C` 拷贝并融合 `×alpha` |
| `strsmbatched_copy_kernel` | AIV | `alpha=1` 时 `B→C` 纯拷贝（离席语义的第一步） |
| `strsmbatched_panel_kernel` | AIV | SIMT 三角回代：对角 panel 或整题（小矩阵） |
| `strsmbatched_gemm_kernel` | AIC | 分块更新 `C_tail -= A_panel × C_panel` |
| `strsmbatched_transpose_kernel` | AIV | RIGHT 路径的 RHS 转置（batch 并行） |

**执行序列（LEFT 分块为例）**：

```text
copy/scale:  C[i] = alpha * B[i]                (一次，batch 并行)
for panel k = 0 .. m step nb:                   (host 循环 launch，核间无跨级依赖)
    panel:   AIV 对 C[:, k:k+nb] 做三角回代      (对角 panel A[k:k+nb, k:k+nb])
    update:  AIC GEMM C[:, k+nb:] -= A[k+nb:, k:k+nb] × C[:, k:k+nb]
```

panel 级间是串行依赖链，级内（batch × RHS 列）完全并行——这是 TRSM 多核化的核心矛盾，设计上用 **batch 并行填补级间空隙**：`unitsPerBatch = ceil(nRHS / panelCols) × batch 内 panel 数`，使 5 条性能 case 的级内并行度（case1: 64 batch × 2 列块、case2: 128 × …）远超 48 核。

**panel kernel（SIMT 回代，AIV）**：

- 每核处理一个 (batch, RHS 列区间)；64 线程/核协作
- `A` 的 nb×nb 对角块一次性搬入 UB（nb≤128，64KB fp32），全核共享只读
- 列主序下按列连续读 `A`（stride=lda），`C` 列按 ldb 访问；`trans=T` 用对偶三角索引（`A(col,row)`），不物化转置
- UPPER/LOWER × N/T 四个组合模板特化（`template <bool UPPER, bool TRANS>`），消除运行时分支
- UNIT：跳过对角 load，乘子恒 1.0f

**gemm kernel（AIC-only，与 arch35 同构）**：

- `KERNEL_TYPE_AIC_ONLY`，直接用 Ascend C Cube 分块接口做 `tail -= A_panel × C_panel`（M=尾行数，N=剩余 RHS，K=nb）
- tileM/tileN 由 host 按"`tiles ≥ aicCoreNum` 下取最大 tile"选择（16/32/64/128 候选），A2 的 24 AIC 打满
- 尾块不足 16 对齐时由 kernel 内边界处理，不回退标量

**RIGHT 路径**：`transpose(B→Bt)` → LEFT 求解（`trans` 取反，`uplo` 不变，作用于 `Bt`）→ `transpose(Bt→C)`。三个 kernel 均 batch 并行；workspace 需 `batchCount × n × m × 4B`（host 检查 handle workspace 容量）。

**多 batch 一致性**：所有 kernel 以 `batchIdx` 为最高维索引，从 Device 指针数组取各 batch 基址；batch 间无任何数据依赖，禁止任何跨 batch 累加或共享中间量。

### 存储与对齐

- 外部契约始终列主序 `row + col*ld`；kernel 内 UB 块按 32B 对齐搬入
- lda/ldb/ldc padding 区域不读不写（C 仅写 m×n 有效区）
- GEMM 输入按 fractal 对齐（FP32_C0=8）在 host 侧对 K 分块（tileKChunk 8/32/64/128），L1 双缓冲占用经 `2×(tileM×Kc + Kc×tileN)×4B ≤ 0.9×L1` 约束收敛

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（arch22） | √ |
| Atlas 800I/T A3（arch22） | √ |
| Ascend 950（arch35） | 已有仓内实现，不在本任务修改范围 |

自验证覆盖其中一种款型即可；性能以 910B3 为准。

## 算子约束限制

- 不支持 broadcast、不支持超出 lda/ldb/ldc 的非连续访问
- 不要求确定性逐位一致（浮点累加顺序不保证）
- `C` 与 `A` 或 `B` 不允许内存重叠
- alpha 仅 Host；A/B/C 及指针数组仅 Device
- 读回结果前由调用方同步 stream
- `trans=ACLBLAS_OP_C` 非法；`m/n/batchCount=0` 为合法 no-op

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | FLOAT32：rtol=atol=`2^-13`（1.22e-4），matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2（或 32 ULP，大数规约场景可放宽至 2 ULP）；逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|` | 任务书 §3.2 / 生态混合容差标准 |
| Golden | cblas `strsm` 逐 batch，对输出 C 全矩阵比对 | 任务书 §3.2 |
| 性能 | 910B3，warmup 后有效采样 >10 次取平均（msprof op 自带 5 次 warmup），Avg time 不高于下表 | 任务书 §3.3 |

| case | batchCount | m | n | side | uplo | trans | diag | 达标耗时（us） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 256 | 256 | LEFT | LOWER | N | NON_UNIT | 630.7 |
| 2 | 128 | 384 | 512 | LEFT | UPPER | T | NON_UNIT | 3062 |
| 3 | 32 | 1024 | 1024 | RIGHT | LOWER | N | UNIT | 5134 |
| 4 | 8 | 2048 | 2048 | LEFT | UPPER | T | NON_UNIT | 8680 |
| 5 | 2 | 4096 | 4096 | RIGHT | UPPER | N | UNIT | 14361 |

**性能可行性要点**：

- case 1（64×256×256 LEFT）：m=256 分 4 个 panel（nb=64），级内并行度 = 64 batch × 4 panel = 256 单元 ≥ 48 核，GEMM 更新量小，SIMT panel 占主导——batch 并行设计正好化解 TRSM 的级间串行瓶颈
- case 3/5（RIGHT 大矩阵）：转置变换后按 LEFT 分块，Cube 更新承担主要 FLOPs（case 5 约 137 GFLOP，910B3 fp32 cube 有效 ~10 TFLOPS 量级 → ~14ms 量级），与 14361us 标线同量级，需要 panel/GEMM 的 tile 选择按 A2 实测微调
- case 4（m=n=2048, batch=8）：panel 数 16×8 batch=128 单元，AIV panel 回代与 AIC GEMM 更新交替，级间空隙由 8 batch × 多 RHS 列块填补

## 测试计划

- 用例来源：任务配套 `test_cases/strsmbatched_test.csv`（1000 精度 + 200 性能，含与任务书 5 条性能 case 逐参数匹配的 base-000~004）+ 缺口自补（零维、空指针数组、batch 内元素 nullptr、非法枚举含 OP_C、非法前导维、alpha∈{0,1,-1}、Inf/NaN、lda/ldb/ldc padding、UNIT 跳对角验证）
- 框架：ops-blas test 目录 CSV → C++ GTest → `aclblasStrsmBatched`（A2/A3 签名）
- golden：cblas `strsm` 逐 batch 生成，全矩阵混合容差比对
- 负向对齐任务书错误码：`OP_C` 判非法（与 cuBLAS 实数档的差异点）、`batchCount<0` 判非法、`batchCount=0` 判成功
- 性能：msprof op，warmup 后有效采样 >10 次取平均

## 兼容性分析

- 新增 arch22 实现与测试目录，不删除、不修改 arch35
- 公共头文件以重载形式增加 `Carray/ldc` 签名（参数个数不同，无二义），950 原地接口保持可编译
- `test/trsmbatched/strsmbatched/strsmbatched_param.h` 增补 `carray/ldc` 列解析，arch35 既有 CSV 不受影响（新增列有默认值语义时向后兼容；如需拆分则 arch35 测试目录保持原样）
- README 产品支持表：Atlas A2/A3 改为"支持"，Ascend 950 保持原状
- 个人仓将邀请 `Ascend-CANN` 作为开发者，供验收拉取
