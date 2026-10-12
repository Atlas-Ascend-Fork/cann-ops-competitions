# aclblasCtrsv 算子设计文档（Ascend 950PR，arch35）

> 依据《aclblasCtrsv 950 算子开发任务书》编写，按官方设计模板组织，四个必需一级章节齐全。
> 交付目标仓：https://gitcode.com/cann/ops-blas （`blas/trsv/arch35/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 9 月社区任务「aclblasCtrsv 算子开发(950)」。在 `ops-blas` 仓为 Ascend 950PR（`arch35`）实现单精度复数（COMPLEX64）三角求解算子 `aclblasCtrsv`：解 `op(A)·x = b`，A 为 n×n 列主序全存储三角矩阵（lda 前导维），右端 b 初始存放于 x，解原地写回 x。

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasCtrsv_Atlas950PR_task_doc.md` | 功能、接口、硬件、精度/性能门槛与交付要求 |
| `test_cases/`（ctrsv_test.csv、gpu_baseline.csv、gen_csv.py、verify_accuracy.py、verify_performance.py、README） | 用例运行方式、golden、性能统计口径 |
| ops-blas 仓 master | 工程框架、头文件、同族实现与测试组织 |

## 1.2 背景介绍

### 算子语义：全存储三角的串行求解

`ctrsv` 为 BLAS Level-2 单右端三角求解（unpack、带 lda）。A 列主序存储于 `lda×n` 数组：`A(i,j) = A[i + j·lda]`，**同一列内元素物理连续**。仅 `uplo` 指定三角被引用；`diag = UNIT` 时对角不读取、视为 1；`OP_C` 为共轭转置（复数下 T/C 不等价）。与 packed 版 ctpsv 的差异：无打包偏移换算，但列间存在 lda 跨步、且另一三角与 padding 区不得访问。

求解为逐列（右视）/逐行（左视）循环携带依赖，单右端项并行度受限，属依赖链/延迟敏感型算子（性能反算见 §3.2.5）。

### ops-blas 现状分析

基于 ops-blas master 实测核对：

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | **仅有实数版 `aclblasStrsv` 声明，无 `aclblasCtrsv`** | 需**新增声明**（签名对齐 `aclblasStrsv` 的参数次序与 int 维度，见 §3.2.1），供其他产品线共用 |
| `blas/trsv/arch35/` | 已有实数版 strsv 全套（`strsv_host.cpp / strsv_kernel.cpp / strsv_tiling_data.h`，SIMT 编程模型：`simt_api/asc_simt.h`、`__simt_callee__`、`threadIdx.x` 跨步循环 + `__ubuf__` 树归约） | **直接工程模板**：Host 校验/分发 + TilingData + SIMT kernel 骨架整体沿用 |
| `blas/trsv/arch22/` | 实数版 strsv（标准 Ascend C） | 不涉及 |
| `test/trsv/strsv/` | 家族级 `trsv_golden.h / trsv_param.h` 及 arch22/arch35 测试 | 测试落点 `test/trsv/ctrsv/arch35/`，CSV 列映射对齐 `trsv_param.h` ReadMap（uplo/trans/diag/n/A/lda/x/incx） |

strsv arch35 的关键先例（源码实测）：`StrsvTilingData { n, uplo, trans, diag, incx, lda, numThreads }`；行点积路径 `StrsvSimtProcessRow`（`ComputePartial` GM 直读 / `DotPartial` UB 命中两形态）+ `StrsvSimtReduce`（`__ubuf__ partialSums` 树形归约）。复数版在其上扩展实虚分离计算与共轭路径。

### 功能缺口

**ctrsv 复数版整体缺失**（仓内无任何 ctrsv 实现）。相对 strsv 的差异即全部工作量：

1. 数据宽度 4B → 8B（complex64 实虚交错），UB 中 x 常驻容量与 numThreads 预算重估；
2. 标量乘加 → 复数乘减（4 乘 2 加/元素）与复数对角除法；
3. `OP_C` 共轭路径（实数版 T/C 共用的做法不适用，必须拆分）；
4. 头文件新增声明（公共 API 面，需与既有命名/参数风格严格一致）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/trsv/arch35/` 沿用 SIMT kernel 直调框架实现 `aclblasCtrsv`，并在 `include/cann_ops_blas.h` 新增公共声明。能力达到：COMPLEX64 × {UPPER, LOWER} × {OP_N, OP_T, OP_C} × {NON_UNIT, UNIT} 全 12 组枚举，`lda ≥ max(1,n)`（含 padding 场景），`incx` 非零（含负），`n = 0` 合法 no-op，n 覆盖精度段 [1, 2048] 与性能段 [1, 4096]；精度满足 experimental 标准（实/虚按 FLOAT32：rtol = 2⁻¹⁰，atol = 2⁻¹⁶，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP）；5 条锚点性能 case 达标（NPU ≤ GPU(H100) 实测 / 0.4，warmup 后有效采样 > 50 次）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | COMPLEX64，解 `op(A)·x = b`，原地写回 | 任务书 §2.1 | 实虚分离 SIMT 计算，x UB 常驻 |
| R02 | 12 组枚举全组合 | 任务书 §2.1 | Host 统一分发 + kernel 双形态（N 型列 AXPY / T·C 型行点积 + 共轭） |
| R03 | 列主序 + lda ≥ max(1,n)，仅引用指定三角，不访问 padding/另一三角 | 任务书 §2.1/§2.5 | 列地址 `base = j·lda`，访问界严格按 [0,n) 行区间裁剪 |
| R04 | `incx ≠ 0` 含负步长 | 任务书 §2.1 | 装载/回写按 `XPhys(i)` gather/scatter，kernel 主循环恒连续 |
| R05 | `n=0` no-op；非法参数/枚举/前导维返回 INVALID_VALUE；handle 空 HANDLE_IS_NULLPTR | 任务书 §2.4 | Host 校验层（§3.2.1） |
| R06 | UNIT 对角不读取 A 对角 | 任务书 §2.1 | 对角分支跳过除法且不触碰 `A[i+i·lda]` |
| R07 | 头文件新增声明，禁私有平行接口 | 任务书 §2.1/§2.3 | §3.2.1 签名与放置位置 |
| R08 | 精度：experimental 标准 + 测试工程 MERE/MARE 双口径 | 任务书 §3.2 | §4.1，Netlib ctrsv golden |
| R09 | 性能：5 锚点 ≤ gpu_ms/0.4 | 任务书 §3.3 | §3.2.5 反算 + §3.2.4 分块并行设计 |
| R10 | 异步执行 | 任务书 §2.5 | `ctrsv_kernel_do(..., stream)`，内部无隐式同步 |
| R11 | 工程合规 | 任务书 §5 | §3.2.0 文件布局，不动 strsv 既有文件 |
| R12 | 可测可复现 | 任务书 §3.5 | CSV 驱动 GTest + 固定种子 + verify_performance.py |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

设 `A(i,j) = A[i + j·lda]`（列主序），求解递推（0 基）：

**LOWER + OP_N（前代，右视列扫描）**：对 k = 0..n−1

```text
xk = UNIT ? xk : xk / A(k,k)
x[i] -= A(i,k) · xk    （i = k+1 .. n-1；A(i,k) 为列 k 的连续段）
```

UPPER + OP_N 为回代方向（k = n−1..0，更新 i = 0..k−1）。

**LOWER + OP_T / OP_C（点积型回代）**：对 i = n−1..0

```text
x[i] = ( b[i] − Σ_{j>i} ô(j,i) · x[j] ) / ô(i,i)，  ô = A（OP_T）或 conj(A)（OP_C）
```

求和项为列 i 上 `j ∈ (i, n)` 的连续段。UPPER + OP_T/OP_C 为前代方向点积（列 j 上段 `∈ [0, j)`）。四种方向与 ctpsv 同构，区别仅在列基地址（`j·lda`）与长度（不再由打包公式决定）。

**复数基础运算**（实虚分离）：

```text
CMul(a,u)：Re = ar·ur − ai·ui；Im = ar·ui + ai·ur            （4 mul + 2 add）
CDiv(x,a)：d = ar² + ai²；Re = (xr·ar + xi·ai)/d；Im = (xi·ar − xr·ai)/d
```

OP_C 共轭：对取用列段虚部取负（SIMT 线程按元素处理，无 lane 对齐负担）。

### 3.1.2 支持数据类型

仅 COMPLEX64（`aclblasComplex`，实/虚各 FP32）。

### 3.1.3 支持形状

`n ≥ 0`；A 数组维度 `lda×n`，`lda ≥ max(1,n)`；x 逻辑 n、物理 `1+(n−1)·|incx|`。UB 预算下 n 上限 8192（§3.2.2，超限 `NOT_SUPPORTED`；任务用例域 ≤ 4096，测试侧另有单用例 host ≤ 512MB 预算约束 n ≤ 4096，与 API 无关）。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内同族，不动 strsv 既有文件）

```text
include/cann_ops_blas.h          # 新增 aclblasCtrsv 声明（紧邻 aclblasStrsv）
blas/trsv/arch35/
├── ctrsv_host.cpp               # 接口校验 + tiling 组装 + kernel 启动
├── ctrsv_kernel.h               # ctrsv_kernel_do 声明
├── ctrsv_kernel.cpp             # SIMT kernel（N 型 / T·C 型双路径 + 分块大 n 路径）
└── ctrsv_tiling_data.h          # CtrsvTilingData
test/trsv/ctrsv/arch35/
├── ctrsv_npu_wrapper.h          # stream 绑定与同步封装
├── ctrsv_test.cpp               # CSV 驱动 GTest（NullHandle 直测）
└── ctrsv_test.csv               # 任务包 1200 条用例
```

### 3.2.1 Host 侧设计

**公共 API 新增**（与 `aclblasStrsv` 逐参数对应）：

```cpp
aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, const aclblasComplex* A, int lda,
    aclblasComplex* x, int incx);
```

校验层（kernel 启动前完成，顺序即返回优先级）：

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | handle == nullptr | `HANDLE_IS_NULLPTR`（n=0 亦先判） |
| 2 | n < 0 | `INVALID_VALUE` |
| 3 | uplo / trans / diag ∉ 合法枚举 | `INVALID_VALUE` |
| 4 | n > 0 且（A == nullptr 或 x == nullptr） | `INVALID_VALUE` |
| 5 | lda < max(1, n) | `INVALID_VALUE` |
| 6 | incx == 0 | `INVALID_VALUE` |
| 7 | n > N_MAX | `NOT_SUPPORTED` |
| 8 | n == 0 | `SUCCESS`（不发射 kernel；依据 Netlib `IF (N.EQ.0) RETURN`） |

TilingData（对齐 `StrsvTilingData` 结构）：

```cpp
struct CtrsvTilingData {
    uint32_t n;
    uint32_t uplo, trans, diag;
    int32_t  incx;
    int32_t  lda;
    uint32_t numThreads;  // SIMT 线程数（0 = 标量小 n 路径）
    uint32_t blockSize;   // 大 n 分块面板宽（§3.2.4，0 = 不分块）
};
```

启动：`ctrsv_kernel_do(tiling, handle->stream)`，SIMT 调用 `asc_vf_call<CtrsvSimtMain>(dim3{numThreads}, ...)`，绑定 handle 携带 stream，无隐式同步。

### 3.2.2 Kernel 侧设计（本任务技术主体）

**总体：x 常驻 `__ubuf__`、行/列双形态、SIMT 线程协同 + 大 n 分块加速。** 沿用 strsv arch35 的 SIMT 骨架向 complex64 扩展。

**UB 预算与 n 上限**：x 工作区 `8n` B + 列/行段搬运缓冲与归约暂存。950 单核 UB 约束下取 `N_MAX = 8192`（x 占 64KB；任务域 n ≤ 4096 → 32KB，余量 2×）。

**主流程**：

```text
Phase0  StrsvSimtLoadX 同构：按 XPhys(i) = (incx>0 ? i : n-1-i)·|incx| 协同 gather GM→UB
Phase1  求解主循环（形态由 uplo×trans 决定）：
          N 型：右视列扫描 —— 列 k 对角归一（CDiv）后，余段更新为复数 AXPY，
                线程按 i 跨步分担（threadIdx.x stride blockDim.x）
          T·C 型：左视行点积 —— 行 i 的已解前缀段上每线程累部分积
                （OP_C 对虚部取负），__ubuf__ partialSums 树形归约后做 CDiv
Phase2  StoreX：协同 scatter UB→GM（原地覆写）
```

**复数点积的线程归约**（T·C 型核心，扩展 `StrsvSimtReduce`）：

```text
每线程 j 跨步累加：dRe += ar·xr − ai·xi；dIm += ar·xi + ai·xr
partialRe/partialIm[threadIdx.x] → 树形两路归约（stride 减半直至 1）
```

### 3.2.3 调用与执行流程图

```text
Host                                            Device (SIMT block × numThreads)
────                                            ─────────────────────────────
aclblasCtrsv(...)
  ├─ 校验表 §3.2.1 ─→ 非法即返
  ├─ n==0 ─→ SUCCESS（不发射）
  ├─ CtrsvTilingData 组装（numThreads/blockSize 由 n 分档）
  └─ ctrsv_kernel_do(t, stream) ─────────────→  CtrsvSimtMain
                                                   ├─ LoadX（gather by incx）
                                                   ├─ n ≤ 小阈值：标量路径
                                                   ├─ N 型：列扫描 AXPY（列基址 j·lda）
                                                   ├─ T·C 型：行点积 + conj + 树归约
                                                   │    （n > blockThreshold：分块面板，§3.2.4）
                                                   └─ StoreX（scatter by incx）
```

### 3.2.4 大 n 分块面板（性能路径）

朴素行点积/列扫描在 n 增大时受单发射依赖链限制（每步一个归约/一次对角除）。n 超过阈值（取 1024，tiling 可调）时启用**分块两阶段**：

```text
将 [0,n) 划分为宽度 B 的面板（B 默认 512）：
  阶段 A（顺序）：解当前面板的对角三角块（面板内仍按 N/T·C 形态，SIMT 线程并行）
  阶段 B（并行）：用刚解出的面板解向量更新其后未解段
        x_tail -= A(尾段, 面板) · x_panel   —— 复数 GEMV，按列线程块并行度高
依赖链长度由 n 降为 n/B，阶段 B 的访存/计算占主导且可充分并行
```

面板更新为纯连续读（列主序下面板各列连续、跨列 lda 步进），是主要带宽消耗段，与 §3.2.5 反算一致。

### 3.2.5 性能设计反算（R09）

锚点最大 case：n = 4096，有效三角读 `4096×4097/2 × 8B ≈ 67.1MB`，达标线 1177.84µs ⇒ 有效带宽 ≥ **57GB/s**；GPU（H100）实测 471µs ⇒ 142GB/s，同为依赖链/发射受限。950 侧：阶段 B 面板 GEMV 连续读 + SIMT 多线程/向量装载可支撑数十 GB/s 量级的流式读，依赖链经分块缩短 n/B 倍后不再支配耗时；小 n case（146µs@n=512）由固定开销主导，单 kernel 单发射结构最小化该开销。达标结论：结构上无反例，最终以 `verify_performance.py` + msprof 实测为准（§4.2）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

（`include/cann_ops_blas.h` 声明为公共 API，其他产品线（如 arch22）后续可另行实现共用。）

## 3.4 算子约束限制

- n ∈ [0, 8192]，超限返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（UB 常驻约束；任务用例域 n ≤ 4096）；
- 不做奇异性/近奇异性检测（对齐 cuBLAS/Netlib），NON_UNIT 时调用方保证对角非零（测试侧 `boost = max(5,n)` 填充策略）；
- 仅引用 uplo 指定三角，另一三角与 lda padding 区不访问；A 只读且与 x 不允许内存重叠；
- 不支持超出 lda/incx 语义的非连续访问；无广播、无 dynamic shape、无确定性计算承诺；UNIT 对角时不读取 A 对角位置。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实/虚按 FLOAT32：`|actual−golden| ≤ 2⁻¹⁶ + 2⁻¹⁰·|golden|`，matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP；测试工程双口径另计 MERE < 2⁻¹³、MARE ≤ 10× 离群倍率 | 任务书 §3.2；opbase experimental_standard |
| 性能标准 | 5 锚点 Avg time ≤ gpu_ms/0.4：146.33 / 272.53 / 526.22 / 1177.84 / 1030.61 µs（n=512/1024/2048/4096/4096-UNIT）；warmup 后有效采样 > 50 次均值 | 任务书 §3.3；gpu_baseline.csv（H100 基线） |

## 4.2 测试方案

- **用例**：任务包 1200 条（1000 精度 + 200 TC_PF；12 组枚举全组合 × 尺寸扫描 1→2048/4096 × lda 紧凑/_PADDING(n+4/n+8) × incx ±1/±2/±3 × 填充/Inf/NaN × 边界负向），固定种子可复现；性能段均为连续访存（incx=1、lda 紧凑）；
- **golden**：cblas（Netlib `ctrsv`）单标杆，实/虚分别比对；填充 A 后对被引用对角加符号保持偏移 `boost = max(5,n)`（沿用 `test/trsv/strsv/arch35/strsv_test.cpp` 做法），golden 用同一偏移后矩阵回代；UNIT 用例 golden 不读对角；
- **执行**：`verify_accuracy.py --repo <ops-blas> --soc ascend950 --csv ctrsv_test.csv`（精度，逐条 PASS/FAIL）；`verify_performance.py --repo <ops-blas> --soc ascend950`（性能，与 gpu_baseline.csv 逐条关联，倍率 0.4，基线缺失记 NO_REF 仅采集）；精确 kernel 耗时辅以 msprof；
- **覆盖核查**：L0 12 组全枚举、L1 尺寸扫描、L2 对角语义（UNIT 全零对照）、L4 前导维 padding 与步长（含负）、L5 填充/Inf/NaN、L6 边界负向（零维/空指针/非法枚举/非法前导维/负维度/零步长）全覆盖，缺失场景按任务书 §3.5 补充；handle nullptr 以非 CSV 的 GTest 直测（NullHandle）覆盖（仓内惯例）。

## 4.3 兼容性分析

- **公共头文件**：仅新增 `aclblasCtrsv` 声明，不改动既有任何声明（含 `aclblasStrsv`），其他产品线可后续共用；
- **仓内文件**：不动 `blas/trsv/arch22|arch35` 既有 strsv 文件与 `test/trsv/strsv/`；构建按 arch 目录 glob 自动纳入新文件；
- **行为兼容**：新算子，无存量调用方，不涉及行为变更。

# 5. 参考链接

1. 任务书与测试包：`aclblasCtrsv_Atlas950PR_task_doc.md`、`test_cases/README.md`；
2. ops-blas 仓：https://gitcode.com/cann/ops-blas （`blas/trsv/arch35/strsv_*` 工程先例、`include/cann_ops_blas_common.h`、`test/trsv/strsv/trsv_param.h`）；
3. Netlib BLAS ctrsv：https://www.netlib.org/blas/ctrsv.f ；
4. cuBLAS cublasCtrsv：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsv ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md 。
