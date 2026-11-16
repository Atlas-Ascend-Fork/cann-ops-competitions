# aclblasSgemv 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 `04_tasks/01_community-task-2026/resources/design_template.md` 六段结构（cann-ops-competitions）。
> 数据来源：任务书《aclblasSgemv算子开发 任务书》逐条溯源 + 对 ops-blas master@9565ef4 现场核验
> （`blas/gemv/`（arch22 含 cgemv 双 kernel）、`blas/gemv/arch35/sgemv_host.cpp`、`include/cann_ops_blas.h`）。
> 文中性能数值为验收目标；设备实测数据在交付件中回填，未测数字均已显式标注。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasSgemv 算子开发（A2A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，目标平台为 Atlas 800T A2（Ascend 910B3，arch22 / dav-2201），CANN 9.1.0。以 Ascend C kernel 直调补齐 Blas 仓 Atlas A2/A3 系列产品的 BLAS-2 单精度实数矩阵-向量乘能力缺口。

## 背景介绍

接口对齐 cuBLAS `cublasSgemv`（[CUBLAS_Library.pdf](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf) §2.6.2），数值 Golden 依据 Netlib BLAS 参考实现 [`sgemv.f`](https://netlib.org/blas/sgemv.f)。公共声明已存在于 `include/cann_ops_blas.h:477`，本任务**不新增、不修改公共头文件**，只补齐 arch22 实现与测试工程。

仓内现状与可复用锚点：

| 锚点 | 位置 | 本任务的用法 |
|---|---|---|
| arch35 单精度实现（SIMT） | `blas/gemv/arch35/sgemv_host.cpp` | 参数校验顺序与返回码口径的行为基准（L31-107） |
| arch22 复数同族实现 | `blas/gemv/arch22/cgemv_host.cpp`、`cgemv_no_trans_kernel.cpp`、`cgemv_do_trans_kernel.cpp` | arch22 向量流水工程形态、转置访存分块先例 |
| arch22 UB 转置已认证先例 | `blas/gbmv/arch22/cgbmv_kernel.cpp:611`（`TransDataTo5HD`）、`:615`（`Gather` 回退） | T/C 路径 UB 内转置 |
| arch22 标量位置探测 + 缓存 | `blas/symv/arch22/chemv_host.cpp:67`（`ResolveChemvScalarLocations`）、`:123`（`GetCachedAivCoreCount`） | host 侧标量读取与核数缓存 |
| 跨度守卫同源公式 | `blas/common/helper/gemv_batched_span_check.h`（`GemvBatchedSpanExceedsInt32`） | §2.4 新增 int32 跨度守卫 |
| 精度/性能测试夹具 | `test/symv/chemv/arch22/chemv_test.cpp:292-321` | `kWarmup=5`/`kSamples=10`、`aclrtEvent`、`[PERF] avg_us` 输出 |
| 设计文档模板 | `04_tasks/01_community-task-2026/resources/design_template.md`（本仓）；仓内写法参考 `docs/zh/develop/aclblasCher2_design.md` | 本文结构 |

**arch35 实现不可移植**：arch35 sgemv 走 SIMT（`__simt_vf__` / `asc_vf_call` / `blockIdx` / `threadIdx`），arch22 无 SIMT 向量函数接口，必须改用 `TPipe` / `TQue` / `GlobalTensor` / `DataCopyPad` / `Muls` / `Add` / `GetBlockIdx()` 的向量流水形态。arch22（910B3）UB 硬容量为 192 KiB（`blas/common/arch/hardware.h:27`）；已合入先例给出两档预算——`blas/axpy/arch22/caxpy_kernel.cpp:30` 用满 192 KiB 并对每个 UB layout 加**编译期 `static_assert`**，`blas/ger/arch22/cgeru_kernel.cpp:61` 取 `UB_BUDGET_FAST = 176 KiB` 留出 per-core scratch 余量。本设计取 **176 KiB 为保守上限**并沿用 caxpy 的 `static_assert` 纪律。向量寄存器 64 个 fp32 lane（256 B）、最小可寻址 UB 块 8 个 fp32（32 B），与 `cgeru_kernel.cpp:57-58` 的 `VREG_FLOATS`/`ALIGN_FLOATS_32B` 一致。

# 需求分析（required）

## 需求描述

计算 `y ← alpha * op(A) * x + beta * y`。A 为列主序 `m×n` 单精度实数矩阵，`A[i,j]` 位于 `i + j*lda`；`op(A)` 由 trans 决定（N：A；T：Aᵀ；C：实数域等价 Aᵀ）。y 原地更新。

公共接口唯一声明于 `include/cann_ops_blas.h:477`（任务书 §2.3 引用的 L461-463 为旧行号，实际声明在 L477-479）：

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);
```

`alpha`/`beta` 按任务书 §2.4 标注为 **Host 内存**标量指针，A/x/y 为 Device 内存。实现侧仍按仓内同族口径兼容 device 标量指针（见 3.2.1）。异步经 handle 绑定的 stream 下发，API 内不同步设备。

> 任务书 §3.1 写作 `cblas_sgemm`、"逐 batch 生成"，为模板复用残留；本算子为单矩阵 BLAS-2，Golden 取 `cblas_sgemv`，无 batch 语义。

## 需求拆解

| 项目 | 实现要求 |
|---|---|
| 结构参数 | 有效 handle；trans ∈ {N,T,C}；m,n ≥ 0；lda ≥ max(1,m)；incx/incy ≠ 0 且 ≠ INT32_MIN；alpha/beta 非空 |
| 跨度守卫（**arch35 缺失，本任务新增**） | `(xDim−1)*|incx|`、`(yDim−1)*|incy|`、`(n−1)*lda+m` 超 int32 域 → `INVALID_VALUE`；host 侧 int64 计算（issue #866/#860 口径） |
| quick return | **全部校验通过后** m==0 或 n==0 → `SUCCESS`，不访问 A/x/y、不启动 kernel、y 逐字节不变 |
| alpha==0 短路 | `y = beta*y`，不读 A/x；beta==0 时不读 y 旧值直接覆写；alpha==0 且 beta==1 时 y 不变 |
| beta==0 | 不读 y 输入值，可直接覆写（性能子集 200 条全部 beta=0，走此路径） |
| 非连续布局 | lda>m 的列间空洞；incx/incy 正负步长；负步长按 Netlib 物理基址约定从物理跨度尾端反向取元素 |
| 特殊值 | 元素值与标量值一律不校验，Inf/NaN 按 IEEE 自然传播 |
| 确定性 | 不要求跨运行一致，但**单次执行内确定**：无原子累加、无跨核归约、每个输出元素唯一写者 |
| 异步 | 使用 handle 当前 stream；返回 SUCCESS 仅表示 Host 已提交 |
| 自验 | 精度 1000 条 + 性能 200 条；开发者自行补充 GAUSS_NORM_5_2 正态分布用例并报告方法/数量/覆盖 |

负向用例与 quick return 用例的缓冲区按任务书 §3.5 约定：**不得**按 `(逻辑长度−1)*|inc|` 预分配，使用 1 元素 dummy 或 NULLPTR，避免非法跨度触发真实分配。

# 详细设计（required）

## 算子分析

### 数学公式

Golden 取 Netlib `sgemv.f` 的 **FLOAT32 逐元素运算顺序**，本实现逐位复现该顺序（依据见「精度标准」节的实测互斥性证据）。

**trans = N**（`y ← alpha*A*x + beta*y`）：

```
先整体一遍：y[i] ← round(beta * y[i])          // beta==0 时直接置 0，不读 y
若 alpha == 0 则返回
for j = 0 .. n-1:
    temp = round(alpha * x[j])                 // 每列只算一次，对该列所有行复用
    for i = 0 .. m-1:
        y[i] = round(y[i] + round(temp * A[i,j]))
```

**trans = T / C**（`y ← alpha*Aᵀ*x + beta*y`）：

```
先整体一遍：y[j] ← round(beta * y[j])
若 alpha == 0 则返回
for j = 0 .. n-1:
    temp = 0
    for i = 0 .. m-1:
        temp = round(temp + round(A[i,j] * x[i]))
    y[j] = round(y[j] + round(alpha * temp))
```

关键约束：**乘法与加法必须分离为两条 FLOAT32 指令**（先 `Muls`/`Mul` 舍入，再 `Add` 舍入），不得使用 `vmla`、`Axpy` 等融合乘加。融合会改变中间舍入，实测在大 alpha 下突破逐元素硬限（见「精度标准」节）。

负步长的物理基址按 Netlib：`inc > 0` 时基址为 0，`inc < 0` 时基址为 `(逻辑长度−1)*|inc|`，第 k 个逻辑元素位于 `base + k*inc`。全部地址计算用 int64。

### 支持数据类型

A、x、y、alpha、beta 统一 FLOAT32。中间乘加全程 FLOAT32，**不引入 FLOAT64 补偿、不引入 double-single**：实测补偿求和同样偏离 Netlib 顺序（4096²/alpha=1e10 下 676 个元素超硬限），且 arch22 向量流水无原生 fp64。

### 支持形状

A 物理跨度 `lda*n` 个 float，元素地址 `A[i + j*lda]`；x 物理长度 `1+(xDim−1)*|incx|`，y 物理长度 `1+(yDim−1)*|incy|`，其中 `xDim = (trans==N) ? n : m`、`yDim = (trans==N) ? m : n`。原始 C ABI 无法推断真实分配长度，调用方保证存储有效；host 侧按上式做 int32 跨度守卫。m、n 支持 0、1 及一般尺寸，最大实测规模 1048576×4 与 4096×4096。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

新增 `blas/gemv/arch22/sgemv_host.cpp`，校验链与 arch35 `ValidateSgemvParams` 逐条对齐并补齐跨度守卫：

1. `handle == nullptr` → `HANDLE_IS_NULLPTR`（**先于一切参数校验**）。
2. 依序校验 `trans → m → n → lda → incx(≠0,≠INT32_MIN) → incy(同) → alpha≠null → beta≠null → (m>0&&n>0 时 A≠null) → (xDim>0 时 x≠null) → (yDim>0 时 y≠null)`，任一失败返回 `INVALID_VALUE` 并 `OP_LOGE`。非法 trans 返回 `INVALID_VALUE`（**不是** `INVALID_ENUM`，与 arch35 及任务书 §2.4 一致）。
3. **新增跨度守卫**：以 int64 计算 `spanX=(xDim−1)*|incx|`、`spanY=(yDim−1)*|incy|`、`spanA=(n−1)*lda+m`，任一 `> INT32_MAX` → `INVALID_VALUE`。公式与 `blas/common/helper/gemv_batched_span_check.h` 的 `GemvBatchedSpanExceedsInt32` 同源（本算子取 `elemFactor=1`、`batchCount=1`），避免 CodeCheck 重复代码告警。`INT32_MIN` 已在第 2 步拒绝，故此处 `abs` 无溢出风险。
4. **校验全部通过后**才判 `m==0 || n==0` → 直接 `SUCCESS`，不取核数、不读标量、不启动 kernel。
5. 读标量：复用 `chemv_host.cpp:67` 的**位置探测 + handle 缓存**形态（`ResolveChemvScalarLocations`），按 `(alpha 指针, beta 指针)` 命中缓存则跳过 `aclrtPointerGetAttributes`；未命中才探测，host 指针直接解引用，device 指针走 `aclrtMemcpyAsync` D2H。为此在 `blas/common/helper/aclblas_handle_internal.h` 追加 `sgemv_scalar_location_valid / _alpha_ptr / _beta_ptr / _alpha_device / _beta_device` 五个字段（与既有 `cher2k_scalar_*` 同构，additive，不改公共头）。
   - **性能理由**：`aclrtPointerGetAttributes` 是 runtime 调用，200 条性能用例中 9 条预算 <12 µs（最小 9.86 µs），每次调用两次探测会吃掉预算。缓存命中后稳态为 0 次 runtime 调用。缓存内容只是"指针在 host 还是 device"这一指针自身属性，与算子无关，按指针身份键控是安全的。
   - 该形态同时满足 PR949 的阻塞检视意见（`*alpha` 无条件解引用对 device 指针会主机访问设备地址，issue #736），与 `cgemv_host.cpp:111/115`、`ssymv_host.cpp:109/113`、`sger_host.cpp:122`、`ssyr2_host.cpp:59` 一致。
6. `alpha == 0` → 选择 SCALE 路径（只算 `y = beta*y`，不读 A/x）；`beta == 0` 时在 tiling 置 `skipYRead` 标志，kernel 不读 y 旧值。
7. 取 AIV 核数：复用 `GetCachedAivCoreCount`（`chemv_host.cpp:123`，缓存于 `handle->cgemm_aiv_cores`），避免每次调用查询平台。
8. 按 trans 与规模计算 tiling（见下），**一次非空调用只发射一个 kernel**。无 CPU fallback、无 Device 结果读回、无 per-call `aclrtMalloc`/`aclrtMemcpy`/`aclrtSynchronizeStream`、无额外 workspace（区别于 `ssymv` 的两阶段双 launch：本算子靠输出维切分即可保证顺序，不需要原子累加，故单 launch 足够）。

Tiling 按值传递（`blas/gemv/arch22/sgemv_tiling_data.h`）：`m, n, lda, trans, alpha, beta, incx(int64), incy(int64), useCoreNum, outPerCore, outChunk, colTile, rowTile, skipYRead, mode`。host 与 kernel 以该结构为唯一契约。

**路径选择只依据布局、规模与平台信息**，不依据数据值（alpha==0 除外，它是任务书 §2.4 明定的语义短路）：

| 条件 | 路径 | 切分维 |
|---|---|---|
| `alpha == 0` | SCALE | 输出维 |
| `trans == N` | NO_TRANS | y 的行（m） |
| `trans == T/C` | DO_TRANS | y 的列（n） |

#### 3.2.2 kernel侧设计

三个 kernel 入口共用 `sgemv_kernel.h` 的 UB 别名图与同步封装，均为**输出维切分 + 单写者**，A 全程只读一遍，无原子、无跨核归约、无 GM workspace。

**（a）NO_TRANS 路径**（`sgemv_no_trans_kernel.cpp`）

核 c 拥有 y 的逻辑行区间 `[r0, r0+R)`，`acc[R]` 常驻 UB：

1. 初始化 `acc`：`beta != 0` 时按 incy 把 `y[r0..r0+R)` 搬入 UB 并 `Muls(acc, yOld, beta, R)`；`beta == 0` 时 `Duplicate(acc, 0, R)`，**不读 y**。
2. 外层按列面板推进 `j0`，步长 K：
   - 一次 `DataCopyPad` 把 `A[r0:r0+R, j0:j0+K]` 搬入 UB，`blockCount=K`、`blockLen=R*4`、`srcStride=(lda−R)*4`、`dstStride=0`，落地为 **[k][i]** 连续面板（每列 R 个 float 连续，列间共 lda 跨度 → GM 侧合并访存）。`lda ≥ max(1,m) ≥ R` 保证 `srcStride ≥ 0`。
   - 同批搬入 `x[j0:j0+K]`（incx≠1 时按物理基址逐元素标量填 UB），`Muls(temp, xUb, alpha, K)` 得到 K 个列标量。
   - `SyncPipe<HardEvent::V_S>()` 后逐 k 取 `temp.GetValue(k)` 为标量，对 `i` 向量化：
     `Muls(prod, aPanel[k*R], temp_k, R)` → `Add(acc, acc, prod, R)`。
3. 全部列面板处理完后把 `acc` 一次写回 `y[r0..r0+R)`（incy=1 走 `DataCopyPad`，否则按物理基址标量/跨步写）。

**顺序等价性证明**：Netlib 是 j 外层、i 内层；本路径对固定面板按 k（即 j）递增、对 i 向量化。每个输出元素 `acc[i]` 只被自己那一列的乘加更新，列的处理顺序在面板间与面板内都是 j 升序，故逐元素累加序列与 Netlib 完全一致 → 逐位相同。

**行分块**：`R = min(CeilAlign(CeilDiv(m, useCoreNum), 8), R_CAP=8192)`，`rowTiles = CeilDiv(outPerCore, R)`。`useCoreNum` 由 `GetAivCoreCount()` 运行时查询（`host_utils.h:75`）并缓存，不写死核数。`m ≤ useCoreNum*R_CAP` 时 `rowTiles=1`；`m=1048576` 时若 `useCoreNum=48` 则 `outPerCore=21846`、`rowTiles=3`，A 按行分块重读 3 遍。这些超大 m 用例（TC_EX_0514/0712/0738/0970 等）均为精度用例、无性能预算，重读可接受。

**（b）DO_TRANS 路径**（`sgemv_do_trans_kernel.cpp`，T 与 C 共用）

核 c 拥有 y 的逻辑列区间 `[j0, j0+C)`，`acc[C]` 常驻 UB，初值 0：

1. 外层按行面板推进 `i0`，步长 R：
   - 一次 `DataCopyPad` 把 `A[i0:i0+R, j0:j0+C]` 搬入 UB staged，`blockCount=C`、`blockLen=R*4`、`srcStride=(lda−R)*4`、`dstStride=0`，落地为 **[j][i]**（每列 R 个 float 连续 → GM 侧合并访存）。
   - **UB 内转置 [j][i] → [i][j]**：首版用 `Gather(dst, src, idxU32, offset, count)` + 一次性预置索引表（`idx[i*C+j] = j*R+i`，由 iota 向量 `Muls` 出 `base[j]=j*R` 后逐行 `Adds` 得到，整个 kernel 只建一次、所有行面板复用）。选它是因为下标算式可在本地用 Python 逐位镜像验证，不依赖硬件转置指令的布局契约。`cgbmv_kernel.cpp:611` 的 `TransDataTo5HD<float>` 16×16 硬件转置（`:615` 为其 `Gather` 回退）作为**后续实测优化项**：其 `TransDataTo5HDParams` 的 repeat/stride 语义与复数双平面布局绑定，须在真机先证明逐位等价再替换，不先验采纳。
   - 转置结果写入独立缓冲 `aT`（`Gather` 的 src/dst 不得重叠）。
   - 同批搬入 `x[i0:i0+R]`（incx≠1 时标量填 UB，填后 `SetFlag/WaitFlag<S_V>` 再供向量读）。
   - **按 i 升序**逐行向量化：`Muls(prod, aT[i*C], x[i0+i], C)` → `Add(acc, acc, prod, C)`。
2. 收尾：`Muls(t1, yOld, beta, C)`、`Muls(t2, acc, alpha, C)`、`Add(yNew, t1, t2, C)`，一次写回 y。`beta==0` 时 `t1` 直接置 0 且不读 y。

**顺序等价性证明**：Netlib 对每个 j 沿 i 升序串行累加 `temp`；本路径对固定列段按行面板 i0 升序、面板内 i 升序，`acc[j]` 的累加序列即 i=0,1,…,m−1，与 Netlib 一致。收尾的 `round(beta*y) + round(alpha*temp)` 两次独立舍入亦与 Netlib `Y(J) = Y(J) + ALPHA*TEMP` 逐位一致。

**不使用 `ReduceSum`/`RepeatReduceSum`/`BlockReduceSum` 做归约**：树形归约改变求和顺序，实测 4096²/alpha=1e10 下 683 个元素超硬限。也不复用 `cgemv_do_trans_kernel.cpp:219` 的 `vmla` + 64  lane 分块归约形态——该形态等价于 chunked 求和，实测 700 个元素超硬限。

**列分块**：`C = min(CeilAlign(CeilDiv(n, useCoreNum), 16), C_CAP=128)`，`colTiles = CeilDiv(outPerCore, C)`。C 对齐 16 以便后续切换到 `TransDataTo5HD` 的 16×16 子块；`R` 取 64（GM 侧每列块 256 B，兼顾 DMA 效率与转置子块数）。`C_CAP=128` 由 UB 反推：staged/aT/idx 三个 `C*R` 平面常驻，`3*C*R*4 = 96 KiB`。R 与 C 是上机后的第一批调优旋钮（增大 R 改善 GM 合并粒度，但按 `3*C*R*4` 挤占 UB）。

**（c）SCALE 路径**（`sgemv_scale_kernel.cpp`）

`alpha == 0`：按输出维切分，`beta==0` → `Duplicate(y, 0)`；`beta==1` → 不启动 kernel（y 不变，host 直接返回 SUCCESS）；其余 → 搬入 y、`Muls(y, y, beta)`、写回。全程不访问 A/x。

**UB 预算**（arch22 硬容量 192 KiB，本设计保守上限 176 KiB = 180224 B）：

| 路径 | 缓冲 | 典型取值 | 字节 |
|---|---|---|---|
| NO_TRANS | `aPanel[K*R]` + `acc[R]` + `prod[R]` + `xUb/temp[K]` | R=1024, K=16 | 72 KiB |
| NO_TRANS | 同上 | R=8192, K=2（超大 m） | 128 KiB |
| DO_TRANS | `staged[C*R]` + `aT[C*R]` + `idx[C*R]`(uint32) + `acc[C]` + `prod[C]` + `xUb[R]` + `yOld/yNew[C]` | C=128, R=64 | ≈98 KiB |
| DO_TRANS | 同上 | C=96, R=64（n=4096、48 核时） | ≈74 KiB |
| SCALE | `yStage[outChunk]` | outChunk=8192 | 32 KiB |

`K = clamp(floor(64 KiB / (4*R)), 1, 16)`，使 NO_TRANS 恒满足 `(K+2)*R*4 ≤ 128 KiB`。全设计最大占用 128 KiB（NO_TRANS 的 R=8192/K=2 档），对 176 KiB 上限留 ≥48 KiB 余量给 `TQue` 与对齐 padding。每个 UB layout 旁加编译期 `static_assert(... <= SGEMV_UB_BUDGET_BYTES)`（照 `caxpy_kernel.cpp:60/350` 形态），不靠人工 review 保证不超配——arch22 UB 超配会崩核并输出全零。

**同步纪律**（PR949 两条阻塞/建议检视意见的直接整改，先例 `blas/tbsv/arch22/stbsv_kernel.cpp:257-261`、`blas/tpsv/arch22/ctpsv_kernel.cpp:997-998`，两处均已逐行核对）：

- 标量填 UB（incx≠1 的 xTile、非连续 y 的标量写）之后、向量读之前：`SetFlag/WaitFlag<HardEvent::S_V>`。
- 标量覆盖上一拍被向量读过的 UB 之前：`SyncPipe<HardEvent::V_S>()`（WAR 保护）。
- DMA 落地后向量消费前：`SyncPipe<HardEvent::MTE2_V>()`；向量写回后 DMA 读前：`SyncPipe<HardEvent::V_MTE3>()`。
- 每个向量算子之间按需 `PipeBarrier<PIPE_V>()`；UB 别名图保证每片独占缓冲，早退分支不得让某类形状跳过同步。

### 工程组织

| 位置 | 内容 |
|---|---|
| `include/cann_ops_blas.h` | **不改动**（声明已在 L477） |
| `blas/gemv/arch22/sgemv_host.cpp` | 参数合同、跨度守卫、标量缓存读取、路径选择、单 launch 异步调度 |
| `blas/gemv/arch22/sgemv_tiling_data.h` | host↔kernel 唯一契约与 UB/分块常量 |
| `blas/gemv/arch22/sgemv_kernel.h` | UB 别名图、同步封装、三路径公共工具 |
| `blas/gemv/arch22/sgemv_no_trans_kernel.cpp` | trans=N |
| `blas/gemv/arch22/sgemv_do_trans_kernel.cpp` | trans=T/C（含 UB 转置） |
| `blas/gemv/arch22/sgemv_scale_kernel.cpp` | alpha==0 |
| `blas/common/helper/aclblas_handle_internal.h` | 追加 5 个 `sgemv_scalar_*` 缓存字段（additive） |
| `test/gemv/sgemv/arch22/` | CSV、GPU 基线、GTest、NPU wrapper、README |
| `test/gemv/sgemv/CMakeLists.txt` | 补 arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)`，else 分支 `ops_blas_add_tests`（照 `test/symv/chemv/CMakeLists.txt` 形态；当前文件为无条件注册，A2/A3 构建会退回非 GTest runner） |
| `blas/gemv/README.md` | aclblasSgemv 的 Atlas A2/A3 支持状态由「不支持」改为「支持」 |

构建复用仓内按文件名前缀与架构筛选机制：`blas/CMakeLists.txt` 的通用规则 `_src_name MATCHES "^${_test_name}_"` 已能在 `--ops=sgemv` 时保留新增源文件，**无需改动算子侧 CMake**。验证命令 `bash build.sh --soc=ascend910b --ops=sgemv`，测试二进制 `build/test/gemv/sgemv/sgemv_test`。

正式自验日志、报告与截图交付至个人私仓 `task_submission/`，不放入上游源码目录。

## 支持硬件

| 芯片版本 | 支持情况 |
|---|---|
| Atlas 800T A2（Ascend 910B3，arch22） | 支持 |
| Atlas A3（arch22） | 支持 |
| Ascend 950PR（arch35） | 本任务不涉及，既有 arch35 实现不改动 |

## 算子约束限制

仅支持任务书约定的 FLOAT32、列主序与 incx/incy/lda 布局；无广播、无 batch、无 dynamic shape 图模式。标量值与元素值不校验。调用方保证操作数存储有效及异步生命周期（读输出前须 `aclrtSynchronizeStream`）。跨度守卫只覆盖 int32 寻址域，不代表真实分配长度校验。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 判定 | 来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_sgemv(CblasColMajor, …)`，FLOAT32 | 任务书 §3.1/§3.5；CI `.gitcode/scripts/pre_smoke.sh:44` 安装 `libblas-dev` |
| 普通精度 | 实部 atol=rtol=2⁻¹³=0.0001220703125，`matched_ratio ≥ 0.99` | 任务书 §3.2；`test/frame/verify.h` MixedToleranceStrategy |
| 逐元素硬限 | `max_abs ≤ max(1e-2, 32*ULP(|golden|))`，**任一元素超限即整例 FAIL**（与 ratio 无关） | 同上 |
| 非有限值 | NaN/NaN 等效；同号 Inf 等效；有限 vs 非有限不等效（严格 `SameNonFinite`） | 任务书 §3.2 |
| 完整性 | y 原地覆写；A/x 只读哨兵逐字节检查；未引用区域不变 | 任务书 §2.5/§3.5 |
| 性能 | 200 条逐例 NPU 平均单次耗时 ≤ `gpu_ms / 0.8`（≥0.8× A100） | 任务书 §3.3；`test_cases/gpu_baseline.csv` |
| 采样 | 每例 5 次预热 + 10 个有效样本取算术平均；`aclrtEvent` 计时并输出 `[PERF] … avg_us=`，另用 msprof op task duration 交叉核对 | 任务书 §3.3；`chemv_test.cpp:292-321` |
| 内存 | 任务书 §3.4 标注「不涉及」；仍按模板交付 `4.1/4.2` 两项并注明口径 | 任务书 §3.4 |

### Golden 口径冲突的披露与实测裁定

任务书内部存在**互相排斥**的两处 Golden 定义，本设计不静默放宽判据，而是披露并给出实测证据：

- §3.1（L87）与 §3.5（L122）：Netlib BLAS 3.10.0 / `cblas_sgemv` **float32 参考实现**，与仓内 `test/gemv/sgemv/sgemv_golden.h` 一致。
- §3.2（L91）：「输入转 **float64** 真值计算，按输出 dtype（FLOAT32）收窄后落盘」。

本地以 `probe_golden.py` 对两种 Golden 同时打分（判据逐字复现 `MixedToleranceStrategy`，含有限值掩码与逐元素硬限），结论：

| 形状 / alpha / beta | netlib_fp32 vs fp64_narrow 的互斥性 |
|---|---|
| 4096×4096, alpha=1e10, beta=0.5, N | **690** 个元素超硬限、10 个超容差、ratio=0.99756 → 两种 Golden 不可能同时通过 |
| 4096×4096, alpha=1e10, beta=2.5, T | **662** 个元素超硬限、5 个超容差、ratio=0.99878 |
| 2048×2048, alpha=1e10, beta=0.5, T | **223** 个元素超硬限、2 个超容差、ratio=0.99902 |
| 4096×4096, alpha=1, beta=0.5, T（全部 200 条性能用例的 alpha） | 0 个超限，两种 Golden 均通过 |

即：**只有大 alpha 精度用例能区分两种口径，性能用例不能**。本设计取 §3.1/§3.5 的 **Netlib fp32** 口径，理由有三：①CI 实际安装 `libblas-dev` 并以 `cblas_sgemv` 生成 golden，PreSmoke_A2 在真机跑 114 条 sgemv 用例；②仓内 `sgemv_golden.h` 就是 fp32 `cblas_sgemv`；③§3.2 的 fp64 措辞与 §3.1/§3.5 及 CI 事实冲突，属孤例。该冲突与上述实测数字将同时写入自验证报告，不通过放宽硬限或引入 fp64 fallback 来"消除"失败。

### 累加顺序与融合乘加的实测裁定

同一探针对候选实现打分（vs netlib_fp32，4096×4096 / alpha=1e10 / beta=0.5 / trans=N）：

| 候选 | 超硬限元素数 | 判定 |
|---|---|---|
| **Netlib 逐元素顺序（本设计）** | **0**，ratio=1.00000 | PASS |
| 列外层 + 向量化（等价形态） | 0，ratio=1.00000 | PASS |
| 树形/成对归约（`ReduceSum` 类） | 683 | FAIL |
| 64-lane 分块归约（`cgemv` 现形态） | 700 | FAIL |
| double-single 补偿求和 | 676 | FAIL |

`probe_fma.py` 进一步分离「乘法+加法」与「融合乘加」：

| 形状 / alpha | 分离 Mul+Add | 融合 FMA |
|---|---|---|
| 512², 1e10, N | 0 超限 PASS | **26** 超限 FAIL |
| 512², 1e10, T | 0 超限 PASS | **19** 超限 FAIL |
| 1024² / 2048² / 4096², 1e10, N | 0 超限 PASS | 40 / 97 / **246** 超限 FAIL |
| 4096², 1e10, T | 0 超限 PASS | **219** 超限 FAIL |
| 1048576×4, 1e10, T（细长） | 0 超限 PASS | **1** 超限 FAIL |
| 任意形状, alpha=1 | 0 超限 PASS | 0 超限 PASS（但 512² 下 433/512、4096² 下 3729/4096 个元素与分离版逐位不同） |

结论：**融合乘加在 512² 就已破坏硬限，细长用例也不能豁免**；分离 `Muls`/`Mul` + `Add` 在全部被测形状与 alpha 下逐位一致。故三个 kernel 一律禁用 `vmla`/`Axpy`，即使 alpha=1 时融合版也能通过。

### 下标算式与分块边界的本地镜像验证

`Gather` 转置的下标算式、DMA 落地布局、尾块 padding 与累加顺序已用 `probe_do_trans_mirror.py` 在本地做 **fp32 逐指令镜像**（NO_TRANS 与 DO_TRANS 两条路径共用同一夹具），不依赖硬件转置指令的布局契约：

- **镜像范围**：`CreateVecIndex → Muls(R*4) → 逐行 Adds(i*4)` 建表、`DataCopyPad` 的 `[j][i]` 落地与 `rightPad`、`Gather` 按字节偏移取数、i 升序 `Muls`+`Add` 分离舍入、收尾 `round(beta*y) + round(alpha*temp)` 两次独立舍入、按核比例切分的输出维分区。
- **覆盖 17 例**：1×1 / 4×4 / 64×64 / 65×63(lda=70) / 129×257 / 512² / 100×100 / 33×47(lda=40) / 17×5 / 5×17 / 200×300(lda=256) / 9×9；行尾块与列尾块、`lda>m` 空洞、`incx/incy ∈ {1,2,3,−2,−3}`、`beta ∈ {0,1,0.5,2.5,−1}`、`alpha ∈ {1,1e10,−2.5,0.5,1.2}`、`cores ∈ {8,16,48}`、`n<cores` 的空闲核、以及 `m=9 < R` 使 `srcStride` 必须按 `activeRows` 计算的角落。
- **自带断言**：每个输出元素**恰好被写一次**（writer 计数逐元素校验）；三片 UB 平面以 **NaN 起盘**，任何误消费未写入 lane 都会显式暴露；结果按 `uint32` 视图与 Netlib fp32 顺序**逐位**比对，而非只比容差。
- **结果：17/17 逐位一致。**
- **负对照（证明镜像有判别力，不是恒真）**：①索引表换成 identity（等于取消转置）→ NaN 守卫触发；②`Adds` 步长误写 `i*8` → 越界；③row0 偏移 `+4`（`j*R+i` → `j*R+i+1`）→ 0/63 逐位一致。还原正确表后回到 63/63。

**明确未做**：仍未在 NPU 上编译或运行任何 kernel。该镜像只验证**下标算式、分块边界、累加顺序与舍入**，不验证 `DataCopyPad`/`Gather` 指令的真实布局契约、同步是否充分、以及吞吐是否达标——这三项只能上机判定，`Gather` 转置的吞吐与 `TransDataTo5HD` 的替换收益列为首批实测项。

### 用例分布与补充计划

`test_cases/sgemv_test.csv` 共 1200 行（精度 1000 + 性能 200），列名为 `case_name`（非 `case_id`）。

- **性能 200 条**：alpha 全为 1、beta 全为 0、incx=incy=1、lda=max(1,m)；trans N=80/T=62/C=58。规模从 1×1 到 4097²。NPU 预算最小 **9.86 µs**（4×4 N）、10.11（8×8）、10.62（2×2）、10.75（1×1）、11.65（16×16）、11.90（32/64/128）；最大 76.80 µs（4060² N）；4096² 为 N=65.79 / T=64.26 / C=65.28 µs；中位数 25.09 µs。**9 条 <12 µs、28 条 <15 µs**，故 host 必须无 per-call runtime 调用（标量位置缓存 + 核数缓存 + 单 launch）。
- **精度 1000 条**：alpha 直方图 1:283、1e-30:115、0.5:105、**1e+10:104**、−2.5:99、0:96、2.5:95、−1:91、1.2:4、1.5:3、1.3:3、inf:1、nan:1；trans N=431/T=295/C=272/非法999=2。其中 **51 条 |alpha|≥1e9 且 max(m,n)≥512** 是精度关键例（含 0345/0387/0454/0510/0867/0914 等 4096² 与 lda>m 空洞例），**均无性能预算**——精度集与性能集不相交，可分别优化。
- **正态分布补充**：按任务书 §3.5 由开发者自行补充 `GAUSS_NORM_5_2`（均值 5、标准差 2）用例，覆盖 x/y/A 三类输入。计划以原 1000 条精度用例的参数为骨架配对生成正态版本，另加 32 条随机 alpha（实部/虚部不适用，取独立采样的正态 alpha）均匀/正态各 16 条；固定 alpha 边界场景（0、1e-30、1e10、inf、nan）保留原值不替换。补充方法、条数与覆盖维度写入自验证报告，**原始 CSV 的 1200 行与 200 条性能基线一字不改**。

### 测试工程要点

- C++ 装载与 Python 校验分别固定断言原始 CSV 共 1200 行、精度 1000 行、性能 200 行、`case_name` 唯一；Python 额外拒绝字段缺失/多余，防止 CSV 与测试集合一同缩水。
- NPU wrapper 必须在 `std::abs` 之前短路 `INT32_MIN`；非法 trans 期望 `INVALID_VALUE`；负向用例与 n=0/inc≤0 快速返回用例使用 1 元素 dummy 或 NULLPTR，不按跨度预分配。
- 守卫哨兵（照 `test/symv/chemv/arch22/chemv_npu_wrapper.h` 的 `CHEMV_GUARD_ELEMENTS=16` 形态）检查 A/x 只读与 y 未引用区域逐字节不变；**验覆盖前先以 NaN 起盘**，避免复用旧缓冲造成假绿。
- 非连续夹具逐 stride 轴枚举（incx、incy、lda 各自独立），不得只做 packed-slice 一臂。
- 补 `NullHandle` 的 `TEST_F`；`[PERF] avg_us` 缺失即判 FAIL，不得以 GTest 整例墙钟耗时替代设备耗时。
- 保留仓内既有回归（4 字节对齐、正负大步长、特殊值、分派边界），并跑 `run_memcheck.py` 式独立进程检查（若适用）。旧验收报告不替代新增验证结果。

## 兼容性分析

不改动公共头文件、不改动 arch35 实现、不改动 `cgemv`/`ssymv`/`chemv` 等既有算子。仅在 `aclblas_handle_internal.h` 追加 5 个缓存字段（additive，不影响既有布局语义）。`test/gemv/sgemv/CMakeLists.txt` 由无条件注册改为 arch 分支，arch35 行为保持不变。其他产品线共用同一公共声明，其实现按各自任务独立接入；本任务只声明 Atlas A2/A3 支持。

## 交付与复现

按官方模板（[task_submission.zip](https://cann-op.obs.cn-north-4.myhuaweicloud.com/task_submission.zip)）交付：设计文档（本文件，随 PR 合入）、`test/gemv/sgemv/arch22/` 测试代码与 README、三个本地口径探针脚本（`probe_golden.py` / `probe_fma.py` / `probe_do_trans_mirror.py`，随自验证步骤说明一并交付，供评审核验本文各表数字）、自验证步骤说明、精度/性能/内存报告与日志、原始 msprof 证据与截图。私仓根目录 `task_submission/` 恰好包含 7 项：`1 自验证步骤说明.md`、`2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`、`3.1 性能自验证报告.xlsx`、`3.2 性能自验证日志.log`、`4.1 内存自验证报告.xlsx`、`4.2 内存自验证日志.log`；私仓邀请 `Ascend-CANN` 为开发者，并在【说明】填写验收 commit 链接。

复现路径：`bash build.sh --soc=ascend910b --ops=sgemv` → `build/test/gemv/sgemv/sgemv_test`（精度全量 + 性能 200 条）→ `verify_accuracy.py` / `verify_performance.py` 解析日志出报告。日志须带平台头（芯片型号、CANN 版本、驱动版本），交付前以 md5 对账「测量源码 = 交付源码」。

参考：[Netlib sgemv.f](https://netlib.org/blas/sgemv.f)、[cuBLAS §2.6.2](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)、[ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)、[CANN 社区任务](https://www.hiascend.com/developer/activities/cann-community-task#cann-task-all)。本设计中的性能数值为验收目标，实际结果另以完成后的真机报告提交。
