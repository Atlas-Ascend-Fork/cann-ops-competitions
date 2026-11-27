# aclblasSgemvStridedBatched 类 5 个算子设计（Atlas A2/A3，arch22）

团队 `gcw_EzaQN38n` · 2026-11 月社区任务 · 设计送审稿

## 1. 需求背景

同一流上的一组等形矩阵-向量乘，在设备内存中按固定间距排列。给出第 0 个 batch 的 A/x/y 基址与相邻 batch 的元素间距，一次下发整组 GEMV。任务书要求在 ops-blas 工程内用 Ascend C 实现 5 个句柄式接口（S / HSH / HSS / TST / TSS），目标芯片 Atlas A2/A3（arch22，DAV_2201），CANN 9.1.0，与 cuBLAS `cublas<t>gemvStridedBatched` 逐参数对齐；精度 golden 为 CPU 侧 Netlib `cblas_sgemv` 逐 batch 生成；性能以 A100 cuBLAS kernel-only 耗时为标杆（NPU ≤ gpu_ms / 0.8）。

### 1.1 工程起点（上游 master 已含 arch22 骨架）

上游 `cann/ops-blas` master 已合入 `blas/gemv_strided_batched/arch22/`（PR !1066，`41f26c96`，2026-11-23）与发射状态回传修复（PR !1338，`14014195`），`test/gemv_strided_batched/arch22/` 亦已有通用 GTest（126 条仓内格式用例）。本任务在该起点上**补齐任务书契约并补足性能路径**，不另起平行实现。本设计对照的上游提交：`b0193c1a`（2026-11-26）。

| 起点 | 现状 | 本任务改动 |
| --- | --- | --- |
| [arch22 Host](https://gitcode.com/cann/ops-blas/blob/master/blas/gemv_strided_batched/arch22/gemv_strided_batched_host.cpp) | 5 入口共用 `GemvStridedBatchedImpl`；m/n/bc 为 0 时在完整校验前 quick return；仅检查 stride > 0，无 batch 重叠检测；按 batch 均分核 | 重排校验顺序对齐任务书 §2.5；增加 A/x/y 批间占用（重叠）检测；tiling 改为「batch × 输出块」二维划分 |
| [arch22 Kernel](https://gitcode.com/cann/ops-blas/blob/master/blas/gemv_strided_batched/arch22/gemv_strided_batched_kernel.cpp) | 每个 batch 整体交给一个核；N 模式逐列 `DataCopyPad` + 标量取 x；T 模式逐列 `ReduceSum` + `GetValue`；无流水 | 保留接口/tiling 结构风格与 5 变体分派；重写核内数据通路：列组 2D 搬运、K-slab、双缓冲、向量指令 repeat 跨列复用，消除逐列标量同步 |
| [arch22 测试](https://gitcode.com/cann/ops-blas/tree/master/test/gemv_strided_batched/arch22) | 单一 `gemv_strided_batched_test.csv`（`dtype` 列区分 5 变体），无 `[PERF]` 输出 | 直接加载任务包 5 个变体 CSV（按文件名推 dtype）；`alpha/beta=null` 空指针语义；TC_PF 用例按 warmup 5 + 10 次 `aclrtEvent` 计时并打印 `[PERF] case=... avg_us=...` |
| [arch35 同族实现](https://gitcode.com/cann/ops-blas/tree/master/blas/gemv_strided_batched/arch35) | 950 口径，含 SIMT/RegBase 路径 | 只用于核对语义（stride 偏移、变体分派、缩放分支），不复用其 SDK 不兼容代码 |

## 2. 需求分析

### 2.1 接口与数据类型

5 个入口共用参数序列 `(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)`，返回 `aclblasStatus_t`。声明已存在于 `include/cann_ops_blas.h`，本任务不改声明、不定义私有平行接口。A/x/y 为 Device 指针，alpha/beta 为指向有限 FP32 的 Host 指针，16 位数据以 `uint16_t` 承载位模式。

| 入口 | cuBLAS 对应 | A / x | 乘加与缩放 | y（输出 dtype，决定容差档位） |
| --- | --- | --- | --- | --- |
| `aclblasSgemvStridedBatched` | `cublasSgemvStridedBatched` | FP32 | FP32 | FP32 |
| `aclblasHSHgemvStridedBatched` | `cublasHSHgemvStridedBatched` | FP16 | FP32 | FP16 |
| `aclblasHSSgemvStridedBatched` | `cublasHSSgemvStridedBatched` | FP16 | FP32 | FP32 |
| `aclblasTSTgemvStridedBatched` | `cublasTSTgemvStridedBatched` | BF16 | FP32 | BF16 |
| `aclblasTSSgemvStridedBatched` | `cublasTSSgemvStridedBatched` | BF16 | FP32 | FP32 |

### 2.2 数学定义与寻址

对批次 `b ∈ [0, B)`：`y_b = alpha · op(A_b) · x_b + beta · y_b`。A 恒按列主序存储、物理形状 **恒为 `lda × n`**（与 trans 无关），`A_b(k, j)` 地址为 `A + b·strideA + j·lda + k`，`lda ≥ max(1, m)`。

| trans | x 逻辑长度 Lx | y 逻辑长度 Ly | 第 j 个输出 |
| --- | ---: | ---: | --- |
| N | n | m | `alpha · Σ_k A_b(j,k)·x_b(k) + beta·y_b(j)` |
| T / C | m | n | `alpha · Σ_k A_b(k,j)·x_b(k) + beta·y_b(j)`；实数算子 C ≡ T |

对 `v ∈ {x, y}`，步长 `s = incv`，逻辑长度 L：第 i 个逻辑元素批内偏移为 `(s<0 ? (L-1)·|s| : 0) + i·s`，非空向量占用区间长 `span_v = 1 + (L-1)·|s|`。三个 stride 以各自 dtype 的元素数计。

**算例**：`m=2, n=3, lda=4, trans=T, incx=-2, incy=2`，三列有效值 `(1,2)、(3,4)、(5,6)`，x 物理 `[10,*,20]`（逻辑 `[20,10]`），`alpha=1, beta=0` → y 物理槽位 0/2/4 分别为 40/100/160，槽位 1/3 保持原值；`strideA=13, stridex=4, stridey=6` 时第二批结果写入槽位 6/8/10。

### 2.3 最小交付边界（紧扣任务书，不做发散）

| 项 | 本任务交付 | 明确不做 |
| --- | --- | --- |
| 实现 | `blas/gemv_strided_batched/arch22/` 5 变体 Host + Kernel | 复数变体、arch35 改动、任务书外的 dtype/layout/参数组合 |
| 测试 | `test/gemv_strided_batched/arch22/` 加载任务包 5×1200 条（精度 1000 + 性能 200 / 变体）并输出 `[PERF]` | 自造额外测试体系、修改 golden 判据 |
| 文档 | `blas/gemv_strided_batched/README.md` 补充 A2/A3 支持说明（上游已含，核对即可） | 其他算子章节、公共文档重构 |
| 内存 | 任务书 §3.4 不涉及 | — |

## 3. 详细设计

### 3.1 Host 校验、标量分支与流

按任务书 §2.5 顺序逐参数校验，首个失败即返回；全部通过后才允许 quick return 与发射。stride/lda/inc 均以 `int64_t` 计算占用，避免 `lda × n` 溢出。

| 顺序 | 检查 | 返回 |
| ---: | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `trans ∉ {N, T, C}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | `m < 0 ‖ n < 0 ‖ batchCount < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `lda < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `incx == 0 ‖ incx == INT_MIN ‖ incy == 0 ‖ incy == INT_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `strideA ≤ 0 ‖ stridex ≤ 0 ‖ stridey ≤ 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `alpha == nullptr ‖ beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `m == 0 ‖ n == 0 ‖ batchCount == 0` → quick return | `ACLBLAS_STATUS_SUCCESS`（不引用 A/x/y、不发射） |
| 9 | `A/x/y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | 批间重叠：`strideA < lda·n ‖ stridex < span_x ‖ stridey < span_y` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 11 | tiling 与发射；`gemv_strided_batched_kernel_do` 返回非 ACL_SUCCESS | `ACLBLAS_STATUS_EXECUTION_FAILED` |

标量语义：`alpha = 0` 时核内跳过 A/x 读取（x/A 含 NaN 不传播）；`beta = 0` 时以 0 初始化累加器，不读 y 旧值。kernel 在 `handle->stream` 上异步执行。

### 3.2 Tiling：「batch × 输出块」二维划分

记 `outLen = (N ? m : n)`，`usedCores = GetAivCoreCount()`（910B3 实测 50 AIV 核，开发时以运行时值为准）。目标：**任何 batchCount 下都用满核**，且单核输出块尽量大以利带宽。

```
splits   = clamp(ceil(usedCores / batchCount), 1, outLen)         // 每个 batch 的输出切块数
outChunk = alignUp(ceil(outLen / splits), partitionAlign)         // partitionAlign = 32 元素（128 B）
splits   = ceil(outLen / outChunk)
workItems = batchCount × splits                                       // 核内按 blockIdx 领取 (batch, chunk)
```

batchCount=1、m=n=4096 时 splits=50 → 每核约 82 行；batchCount ≥ 50 时 splits=1 退化为按 batch 分核（与上游一致）。tiling 结构沿用 `GemvStridedBatchedTilingData`，字段调整为 `outChunk / splits / usedCoreNum / workItems` 等二维描述。

### 3.3 Kernel 数据通路（AIV，`KERNEL_TYPE_AIV_ONLY`）

UB 预算 192 KB，所有 tile 缓冲以 `TBuf<VECCALC>` 分配；半精度/bf16 先 `Cast` 到 FP32 再乘加，输出按 y dtype `Cast(RINT)` 收窄。每个工作项 = 一个 `(batch, 输出区间 [j0, j0+jc))`，按下述两种通路处理，全程无逐列标量同步，仅在写回前做一次 `V_S` 同步。

**N 模式（`outLen = m`，对 k∈[0,n) 的列做 axpy 累加）**

- 以 `kr` 列为一个 K-slab（`kr` 自适应：使 `kr × jcPitch × sizeof(T)` 落在 32–64 KB），一次 **2D `DataCopyPad`**：`blockCount = kr, blockLen = jc·sizeof(T), srcStride = (lda − jc)·sizeof(T)`，把 slab 搬成 UB 内 `[kr][jcPitch]` 布局（`jcPitch = alignUp(jc, 64)`，便于 64 元素 mask 的 repeat 寻址）。
- x 预处理：工作项开头一次性把 x 的 n 个逻辑元素按 incx 取到 UB（incx=±1 连续拷贝；|incx|>1 且跨度 ≤ WIN_MAX 时窗口搬运后标量重排；否则逐元素），并计算 `xs = alpha·x`。
- 累加（沿用仓内 `blas/gemv/arch22` `AccumulateFastNBatch` 已验证的范式，不逐列发射）：用 `Brcb` 把本 slab 的 `xs[k]` 广播成 `[kr][jcPitch]` 布局的 `work`（每个 k 占一个 32 B 块，按行块 repeat），一条 `Mul(tile32, tile32, work)` 完成整 slab 乘法，再按列做树形归约 `Add(tile, tile, tile[half·jcPitch])`（`half = kr/2 … 1`，`kr` 取 2 的幂），最后 `Add(acc, acc, tile[0..jc))`。每个 slab 仅 `jcPitch/8` 条 `Brcb` + 1 条 `Mul` + `log2(kr)` 条 `Add`，与列数成对数关系，彻底消除上游逐列 `GetValue` 标量同步。
- 双缓冲：slab A 计算 / slab B 搬运以 `MTE2_V / V_MTE2` 事件交替，隐藏 MTE2 延迟。
- `beta ≠ 0` 时先 `LoadStridedVec(y)` → `Cast` → `Muls(beta)` 作为 acc 初值；最后 `CastFromFloat` → `StoreStridedVec(y)`（incy=±1 连续写，其他步长逐元素写）。

**T/C 模式（`outLen = n`，对每列做点积）**

- 以 `cg` 列为一组、`kr` 行为一个 K-slab（`kr = min(m, 256)`），一次 2D `DataCopyPad`：`blockCount = cg, blockLen = kr·sizeof(T), srcStride = (lda − kr)·sizeof(T)`，UB 内布局 `[cg][krPitch]`（`krPitch = alignUp(kr, 64)`）；`cg = clamp(128KB / (krPitch·(sizeof(T)+4)), 1, 64)`，小 m 时一组可含 64 列，单条 2D 搬运即 64 个小列。
- 累加：对 slab 内每个 64 元素列块 b：一条 `Mul(tile32, tile32, x[k0 + b·64], mask, repeat=cg, src1RepStride=0)`，再一条 `WholeReduceSum(part, tile32, repStride=krPitch/8, repeat=cg)`，再 `Add(acc, acc, part, cg)`。每组每 slab 共 `3·ceil(kr/64)` 条向量指令，与列数无关；跨 slab、跨列组用同一 `acc[cg]` 累加，m=3 时 64 列只需 3 条指令。
- 写回：`acc = alpha·acc + beta·y`（beta≠0 时先取 y），`CastFromFloat` 后按 incy 写出 cg 个输出。

**步长与特殊值**：负 incx/incy 的起点 `(L−1)·|s|` 在 host 侧 int64 计算并入 tiling；`beta=0` 不读 y，`alpha=0` 直接走 `y = beta·y` 通路；FP16/BF16 Inf/NaN 按 IEEE 在 FP32 乘加中自然传播，与 golden（float64 累加后收窄）在容差框架内比较（NaN/Inf 按 verify.h 既有口径同值同号判匹配）。

### 3.4 测试工程

- `test/gemv_strided_batched/arch22/` 新增任务包 5 个 `{s,hsh,hss,tst,tss}gemv_strided_batched_test.csv` 与对应 `gpu_baseline.csv`（原仓内 `gemv_strided_batched_test.csv` 保留）；`GemvStridedBatchedParam` 兼容任务包列名（`aarray_fill/xarray_fill/yarray_fill` → `A/x/y`）、记录 `alpha/beta` 为 `null` 的空指针标记，按 CSV 文件名前缀推断 `dtype`；用例名含 `TC_PF` 前缀判定性能用例。
- `CsvDriven`：负向用例只断言状态码；正向用例 NPU 执行 → `aclblasGemvStridedBatched_cpu`（cblas FP32 逐 batch）→ `applyMixedTolerance`（按输出 dtype：FP32 2⁻¹³、FP16 2⁻¹⁰、BF16 2⁻⁷，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2/1e-1 或 32 ULP）。
- `Perf`：仅 `TC_PF_*`，Device 缓冲一次性预置后，在 handle 的 stream 上 warmup 5 次（不计时）+ `aclrtEvent` 夹住连续 10 次下发取算术平均，打印 `[PERF] case=<case_name> warmup=5 samples=10 avg_us=<数值>`；`ACLBLAS_GEMV_SB_MSOPROF_SAMPLES` 设定时 warmup=0、samples=N，供 `msprof op` 逐次采集 kernel Task Duration。
- `test/gemv_strided_batched/CMakeLists.txt` 维持 `gemv_strided_batched_test` 单目标（5 变体共用），并让 `build.sh --ops=sgemv_strided_batched|hsh_gemv_strided_batched|…` 可直接构建该目标（任务包 `verify_accuracy.py` / `verify_performance.py` 以此 ops 名调用 build.sh）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B3，实测设备） | √ |
| Atlas A3 训练/推理系列产品 | √ |
| Atlas A2 训练/推理系列产品 | √ |

## 算子约束限制

- A/x/y 为 Device 指针，调用方保证各 batch 区域互不重叠且 y 可写区域与 A/x 不别名；`strideA ≥ lda·n`、`stridex ≥ span_x`、`stridey ≥ span_y`，违者返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- 复数变体与 950（arch35）路径不在本任务范围。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 5 变体 × 1000 条精度用例全部通过；逐 batch 逐元素 `|actual−golden| ≤ atol + rtol·|golden|`，按输出 dtype 取 rtol/atol（FP32 2⁻¹³、FP16 2⁻¹⁰、BF16 2⁻⁷），`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2 / 1e-1 或 32 ULP`；负向用例状态码与任务书 §2.4 一致 | 任务书 §3.2；生态算子开源精度标准 mixed_tolerance_standard.md |
| 性能标准 | 5 变体 × 200 条性能用例：`msprof op` 采集 kernel Task Duration(us)，warmup 5 次不计入、连续 10 次取算术平均，单 case NPU ≤ gpu_ms / 0.8（正式判据）；GTest `[PERF] avg_us` 端到端事件计时仅作交叉验证 | 任务书 §3.3；`test_cases/*/gpu_baseline.csv`（A100 cuBLAS kernel-only） |
| 内存标准 | 任务书 §3.4 不涉及 | 任务书 §3.4 |

验证以 `check_validation.py gtest/performance` 核对完整用例身份、退出码与有效样本数；精度、性能各自独立一轮完整执行，结果不跨轮拼接。

## 兼容性分析

新增 arch22 路径的内核与校验逻辑，公开接口签名与 `include/cann_ops_blas.h` 既有声明一致，arch35 实现不受影响；对上游已合入的 arch22 行为变化仅为：非法参数校验更严格（按任务书口径）、kernel 性能路径替换，功能语义不变。
