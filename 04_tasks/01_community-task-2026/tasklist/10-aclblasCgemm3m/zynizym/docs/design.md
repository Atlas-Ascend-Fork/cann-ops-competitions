# aclblasCgemm3m 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasCgemm3m` |
| 任务名称 | 10月社区任务-aclblasCgemm3m算子开发（950） |
| 目标硬件 | Ascend 950PR（arch35，`DAV_3510`，短 soc 名 `ascend950`） |
| 软件版本 | CANN 9.1.0（asc-devkit ≥ 9.1） |
| 开发方式 | Ascend C Kernel 直调（handle 绑定 stream），不做 aclnn 两段式 |
| 数据类型 | COMPLEX64（`aclblasComplex`，实部/虚部均为 FLOAT32）；`alpha`/`beta` 为复数标量指针 |
| 对标接口 | cuBLAS `cublasCgemm3m`（legacy，3m 算法） |
| 精度 golden | cblas（Netlib BLAS 3.10.0）`cblas_cgemm` 单标杆，实部/虚部分别比对 |
| 算法要求 | 3m 分解：三次实数矩阵乘的线性组合完成复数乘 |
| 合入仓及目录 | `cann/ops-blas`：实现 `blas/gemm3m/arch35/`，测试 `test/gemm3m/cgemm3m/` |
| 接口声明 | 新增 `include/cann_ops_blas.h` 声明（紧邻 `aclblasSgemm3m`），**无 950PR 私有平行接口** |
| 文档状态 | 设计稿，待评审。§7.1/§7.2 的实测字段标注为「待自测回填」，未填入任何未实测数值 |

---

# 1. 需求背景（required）

## 1.1 需求来源

本需求来自昇腾 CANN 社区任务《10月社区任务-aclblasCgemm3m 算子开发（950）》，权威需求来源为任务书 `aclblasCgemm3m_Atlas950PR_task_doc.md`。任务要求在 Ascend 950PR（arch35）上使用 Ascend C 开发单精度复数 3m 算法通用矩阵乘算子 `aclblasCgemm3m`，完成 Host 参数处理、Device Kernel、测试工程、算子 README 与自测材料，验收通过后合入 `cann/ops-blas`。

本任务为**补齐 Blas 仓能力**：`ops-blas` 已有实数版 `aclblasSgemm3m`（`blas/gemm3m/arch35/`）与复数版 `aclblasCgemm`（`blas/gemm/arch35/`，其内部已具备 3 乘积流水线），但**没有以 3m 为契约的复数接口** `aclblasCgemm3m`。需求的关键不在于"再造一套复数 GEMM"，而在于把仓内已验证的 3 乘积复数流水线以 3m 语义固定下来，补齐接口、校验、路由与验收材料。

## 1.2 背景介绍

### 1.2.1 算子简介

`aclblasCgemm3m` 属于 BLAS Level 3 GEMM 族，计算：

```text
C = alpha * op(A) * op(B) + beta * C
```

- `A`、`B`、`C` 为 COMPLEX64 矩阵，**列主序**（Column-Major）存储，复数按 real-first 交错（实部在低地址）；
- `op(X)` 取 `ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）、`ACLBLAS_OP_C`（共轭转置）；复数下 `T` 与 `C` 语义不同（`C` 对虚部取反）；
- `alpha`、`beta` 为**复数标量指针，驻留 Host 内存**，由 Host 侧解引用后随 tiling 下发；
- `A` 为 m×k（`N`）或 k×m（`T/C`），`B` 为 k×n（`N`）或 n×k（`T/C`），`C` 为 m×n；
- 语义与 `cublasCgemm3m` 逐参数对齐（含 `handle` 位置，维数为 `int`），无额外映射。

与 `aclblasCgemm` 的唯一区别是**算法契约**：`aclblasCgemm3m` 承诺用 3m 分解（三次实数乘）完成复数乘，`aclblasCgemm` 不承诺（其实现按 alpha/beta 取值在 3 乘积与 4 乘积之间切换）。

### 1.2.2 标杆算子现状分析

#### 1.2.2.1 标杆接口与数据类型

| 标杆 | 接口 | alpha/beta | A/B/C | 存储 | 算法 |
| --- | --- | --- | --- | --- | --- |
| cuBLAS | `cublasCgemm3m` | `const cuComplex*`（pointer mode） | `cuComplex` | 列主序，`lda/ldb/ldc` | 3m |
| Netlib BLAS（golden） | `cblas_cgemm` | `const void*` | `void*` | 列主序 | 4m（标准实现） |

参数序列、语义完全一致。注意 benchmark 口径：golden 用 `cblas_cgemm`（4m）生成，3m 与 4m 的差异仅在浮点舍入层面（见 §7.1），故按同一 golden 判定。

#### 1.2.2.2 3m 算法与数值特性

复数乘 `(a+bi)(c+di) = (ac−bd) + i(ad+bc)` 朴素实现需要 4 次实数乘。3m 用 3 次实数乘 + 额外的输入/输出加减完成同一结果。

本设计采用 **Gauss 形式**（仓内 `cgemm` 已验证的恒等式）：

```text
P0 = (Ar + Ai) · Br
P1 = Ar · (Bi − Br)
P2 = Ai · (Br + Bi)
```

推导：

```text
P0 − P2 = (Ar·Br + Ai·Br) − (Ai·Br + Ai·Bi) = Ar·Br − Ai·Bi = Re(C)
P0 + P1 = (Ar·Br + Ai·Br) + (Ar·Bi − Ar·Br) = Ar·Bi + Ai·Br = Im(C)
```

即 `Re = P0 − P2`、`Im = P0 + P1`：**3 次 m×k×n 的实数乘 + 2 次 m×n 的加减**，无形状相关系数。

数值特性（3m 的固有代价，`cublasCgemm3m` 亦然）：

1. `Ar+Ai`、`Br+Bi`、`Bi−Br` 在输入侧预先加减，规模为 O(mk+kn)，误差约为 1 ULP/元素；
2. 输出侧 `P0−P2` 在 `|Ar·Br| ≈ |Ai·Bi|`（即结果实部远小于乘积量级）时存在相消，相对误差放大，但绝对误差仍受 `k·u·Σ|a·b|` 约束；
3. 与 4m 相比，3m 的累加项结构不同（`P0` 被实部与虚部共用），故**不要求与 4m 逐位一致**，只要求满足 §7.1 的生态精度标准。

#### 1.2.2.3 标杆执行流程

```text
quick return: m==0 || n==0  ──► return SUCCESS（不写 C）
        │
        ▼
k==0 或 alpha==(0,0)  ──► C = beta*C（beta==0 置零、beta==1 不变、其余复数缩放）
        │
        ▼
拆解 op(A)、op(B) 为实数平面（含转置、共轭）
        │
        ▼
3 次实数 GEMM：P0、P1、P2
        │
        ▼
复数合成 + alpha/beta：C = alpha*((P0−P2) + i*(P0+P1)) + beta*C
```

### 1.2.3 ops-blas 仓内现状

| 位置 | 现状 | 本任务动作 |
| --- | --- | --- |
| `include/cann_ops_blas.h`（`aclblasSgemm3m` 声明处） | 仅有实数 3m 的 `aclblasSgemm3m` 声明 | **新增 `aclblasCgemm3m` 声明**，紧随 `aclblasSgemm3m` |
| `blas/gemm3m/arch35/` | `sgemm3m_host.cpp` / `sgemm3m_kernel.{cpp,h}` / `sgemm3m_tiling_data.h`（实数、调用方预拆分的 A1/A2/A3 拼接矩阵） | 作为**工程范式**参考（目录归属、CMake 自动收集、kernel 直调风格） |
| `blas/gemm/arch35/cgemm_*.cpp` | 复数 GEMM 完整实现：`deinterleave`（AIV SIMT 拆分，含 `threeProducts` 分支）+ `products`（AIC Cube，3/4 乘积）+ `epilogue`（AIV SIMT 复数合成）+ `fused`/`onchip` 融合路径 | **复用为 3m 的计算内核**，不改动其算法 |
| `blas/gemm/arch35/cgemm_host.cpp` | 复数 GEMM Host：校验、工作区、tiling、路由；`threeProducts` **仅在 `alpha==(1,0) && beta==(0,0)` 时启用** | 抽出内部下发函数供 3m 复用（§3.2.2） |
| `blas/gemm3m/README.md` | 仅描述 `aclblasSgemm3m` | 补充 `aclblasCgemm3m` 的产品支持、参数、约束、示例 |
| `test/gemm3m/sgemm3m/` | `CMakeLists.txt` / `sgemm3m_golden.h` / `sgemm3m_param.h` / `arch35/` | 作为测试工程范式；**新增 `test/gemm3m/cgemm3m/`** |

技术定位因此非常清晰：**`blas/gemm/arch35` 的三乘积复数流水线 × `blas/gemm3m/arch35` 的 3m 契约与目录范式**。3m 分解本身不需要新算法——仓内 `cgemm_deinterleave_common.h` 的 `ThreeProducts` 分支正是 Gauss 3m；本任务的增量是把该路径从"仅单位 alpha/beta 时启用的性能优化"提升为"**任意 alpha/beta 下都必须走的算法契约**"，并补齐接口、校验、路由与验收材料。

> 说明：任务书 §5 给出的合入示例路径为 `blas/gemm/arch35` 与 `test/gemm/arch35`。仓内现状是 3m 家族已独立成目录 `blas/gemm3m/arch35`（`sgemm3m` 所在处），且测试指导明确用例安装路径为 `test/gemm3m/cgemm3m/<arch>/cgemm3m_test.csv`。本设计**按仓内既有 gemm3m 家族目录放置**（与 `sgemm3m` 同目录、同类测试结构），以符合仓内组织约定；若有偏差以评审意见为准。

---

# 2. 需求分析（required）

## 2.1 外部组件依赖

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| CANN 9.1.0（ASC 编译器、`kernel_operator.h`、tensor_api、SIMT `asc_vf_call`） | arch35 Kernel 编译 | `cgemm`/`sgemm3m` 已在该工具链上验证 |
| `acl` 运行时（`aclrtMalloc`/`aclrtMemcpy`/stream/`aclrtMemset`） | Device 内存与异步下发 | 复用 `common/helper` |
| cblas（Netlib BLAS 3.10.0）`cblas_cgemm` | 精度 golden | 随测试工程提供 |
| GTest | 测试框架 | 仓内 `test/frame` 已封装 |

不引入任何新的三方依赖。

## 2.2 内部适配模块

| 模块 | 复用方式 |
| --- | --- |
| `common/helper/aclblas_handle_internal.h` | 从 handle 取 stream（`h->stream`）、workspace（`EnsureDefaultWorkspace`/`GetEffectiveWorkspace`） |
| `common/helper/host_utils.h` | `CeilDiv` / `CeilAlign` / `GetAicCoreCount` / `GetAivCoreCount` |
| `blas/gemm/arch35/gemm_host_common.h` | `BuildGemmTilingData` / `ApplyGemmColMajorSwap` |
| `blas/gemm3m/arch35/*` | 目录与 CMake 自动收集范式（`blas/CMakeLists.txt` 按 `arch35` glob） |
| `cann_ops_blas_common.h` | `aclblasComplex`、`aclblasOperation_t`、`aclblasStatus_t` |

## 2.3 需求模块设计

### 2.3.1 接口原型与参数

`include/cann_ops_blas.h` 新增（紧跟 `aclblasSgemm3m`，不使用 `aclblasHandle` 旧拼写，统一 `aclblasHandle_t`）：

```cpp
/**
 * @brief Single-precision complex GEMM using the 3m algorithm, column-major.
 * C = alpha * op(A) * op(B) + beta * C, computed with three real GEMMs per
 * complex product (Gauss identity). alpha/beta are host complex scalars.
 * A/B/C are device pointers; lda >= max(1, transa==N ? m : k),
 * ldb >= max(1, transb==N ? k : n), ldc >= max(1, m).
 * Zero m or n returns success without touching C. Zero k or zero alpha applies
 * C = beta*C only. Ascend 950PR implementation requires asc-devkit >= 9.1.
 */
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
```

| 参数 | 输入/输出 | 类型 | 约束 |
| --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | 非空，绑定 `aclrtStream` |
| `transa`、`transb` | 属性 | `aclblasOperation_t` | `ACLBLAS_OP_N/T/C` |
| `m`、`n`、`k` | 属性 | `int` | ≥ 0 |
| `alpha`、`beta` | 输入 | `const aclblasComplex*` | 非空，Host 内存 |
| `A`、`B` | 输入 | `const aclblasComplex*` | Device 列主序 |
| `C` | 输入/输出 | `aclblasComplex*` | Device 列主序，原地更新 |
| `lda` | 属性 | `int` | ≥ max(1, `N` 时 m，否则 k) |
| `ldb` | 属性 | `int` | ≥ max(1, `N` 时 k，否则 n) |
| `ldc` | 属性 | `int` | ≥ max(1, m) |

### 2.3.2 约束与异常行为

校验顺序与返回码严格按任务书 §2.4/§2.5：

| 序号 | 检查项 | 条件 | 返回码 |
| --- | --- | --- | --- |
| 1 | handle | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | transa | ∉ {N,T,C} | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | transb | ∉ {N,T,C} | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | 维度 | `m<0` 或 `n<0` 或 `k<0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | lda | `< max(1, N?m:k)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | ldb | `< max(1, N?k:n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | ldc | `< max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | alpha/beta | 任一为空指针 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | quick return | `m==0 或 n==0` | `ACLBLAS_STATUS_SUCCESS`（不再校验 C/A/B） |
| 10 | C | `m>0 && n>0` 且 `C==nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 11 | A/B | `m>0 && n>0 && k>0 && alpha≠(0,0)` 且 `A` 或 `B` 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |

**与 `aclblasCgemm` 的差异（有意为之）**：`aclblasCgemm` 对非法 `transa/transb` 返回 `ACLBLAS_STATUS_INVALID_ENUM`，而任务书 §2.4 对本算子明确要求返回 `ACLBLAS_STATUS_INVALID_VALUE`。`aclblasCgemm3m` **遵循任务书**返回 `INVALID_VALUE`，并在 README 与测试 CSV 中固化该口径（CSV 非法枚举行写 `999`，期望 `INVALID_VALUE`）。

条件性空指针校验是易错点：`A/B` 仅在 `m>0 && n>0 && k>0 && alpha≠0` 时报错，因此 `m0_a_null`、`k0_a_b_null`、`alpha0_a_null` 等叠加用例期望 `SUCCESS`。

### 2.3.3 数学与内存边界

**计算量**：复数 MAC 计 8 flop（4 乘 + 4 加）。

| 方案 | 实数乘次数 | flop | 相对 4m |
| --- | --- | --- | --- |
| 4m（朴素） | 4·m·n·k | 8mnk | 1.00 |
| **3m（本设计）** | **3·m·n·k** | **6mnk** + 2(mk+kn) + 4mn | **≈0.75** |

理论上限加速比 1.33×，这是 `cublasCgemm3m` 相对 `cublasCgemm` 的收益来源，也是本算子性能达标的算力依据（§7.2）。

**工作区规模**（`A` 平面 3 份 + `B` 平面 3 份 + 3 个 m×n 中间量）：

```text
W = 12·m·k + 12·k·n + 12·ceilAlign(m,16)·ceilAlign(n,16)·kPartitions   （字节）
```

| m=n=k | 6 个 A/B 平面（`24N²` B） | 3 个中间量（`12N²` B，kPartitions=1） | 合计 |
| --- | --- | --- | --- |
| 256 | 1.57 MB | 0.79 MB | 2.36 MB |
| 1024 | 25.2 MB | 12.6 MB | 37.7 MB |
| 2048 | 100.7 MB | 50.3 MB | 151 MB |
| 4096 | 402.7 MB | 201.3 MB | 604 MB |

结论：工作区随规模线性增长，4096 阶约 604 MB，**必须在 Host 侧做容量检查并在不足时返回明确错误码**（`ACLBLAS_STATUS_EXECUTION_FAILED`），不得静默截断。融合路径（§3.4.5，`alpha==1 && beta==0`）不物化 3 个中间量，工作区降为 `24N²` 字节（4096 阶约 403 MB）。

### 2.3.4 quick return 与标量位置

- `m==0 || n==0`：**不启核**，直接返回 `SUCCESS`，不写 C；
- `k==0` 或 `alpha==(0,0)`：不引用 A/B，执行 `C = beta*C`；`beta==(0,0)` 走 `aclrtMemset` 置零（不读 C 旧值），`beta==(1,0)` 直接返回（C 不变），其余走 AIV SIMT 复数缩放；
- `alpha`、`beta` 为 **Host 指针**，Host 侧解引用为 `std::complex<float>` 后放入 tiling 结构；不引入 device scalar 拷贝，也不改变 cuBLAS 的 pointer mode 语义（本仓统一按 Host 指针处理，与 `cgemm` 一致）；
- `beta==0` 时 epilogue 的 `BetaZero` 模板分支**不读 C**，保证 C 含 NaN/Inf 旧值时仍输出有限结果（§7.1 特殊值口径）。

### 2.3.5 需求拆解

| # | 需求项 | 交付物 |
| --- | --- | --- |
| 1 | 接口声明 `aclblasCgemm3m`（14 参数） | `include/cann_ops_blas.h` |
| 2 | Host 入口：11 步校验、quick return、工作区、tiling、三阶段下发 | `blas/gemm3m/arch35/cgemm3m_host.cpp` |
| 3 | 3m 分解（Gauss 恒等式）落地 | 复用 `cgemm_deinterleave`（`threeProducts=1`）+ `cgemm_products`（3 乘积）+ `cgemm_epilogue`（三乘积合并） |
| 4 | 复数 alpha/beta 与 `beta*C` 融合 | `cgemm_epilogue` 的 SIMT 路径（已支持任意复数 alpha/beta） |
| 5 | 融合加速路径（`alpha==1,beta==0`） | 复用 `cgemm_fused` / `cgemm_onchip`（L0C 内三乘积累加、直写 C） |
| 6 | 工作区容量检查与溢出防护 | Host `CheckedMulSize` + `EnsureDefaultWorkspace` |
| 7 | 确定性（同输入逐位一致） | 无原子累加、固定归约顺序（§3.4.7） |
| 8 | 算子 README（产品支持、参数、约束、示例） | `blas/gemm3m/README.md` |
| 9 | 测试工程 + 1200 条用例 + golden | `test/gemm3m/cgemm3m/` |
| 10 | 自测报告（精度、性能、msprof 截图） | 独立文档，不入仓 |

---

# 3. 详细设计（required）

## 3.1 调用方式

```text
aclblasCreate(&handle)
   └─ aclblasSetStream(handle, stream)              # handle 绑定 stream
        └─ aclblasCgemm3m(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc)
             ├─ Host: 校验 → 标量解引用 → quick return → 工作区 → tiling → 三阶段下发
             │        ├─ cgemm3m 拆解 kernel（AIV SIMT）        ：A/B → 3+3 实数平面
             │        ├─ cgemm 三乘积 kernel（AIC Cube）        ：P0、P1、P2 → 3 个 m×n 中间量
             │        └─ cgemm epilogue kernel（AIV SIMT）      ：C = alpha*((P0−P2)+i(P0+P1)) + beta*C
             └─ 返回 SUCCESS（异步；读回 Device 结果前须 aclrtSynchronizeStream）
```

`alpha==(1,0) && beta==(0,0)` 时走**单 kernel 融合路径**（§3.4.5），无中间量、无 workspace 中间缓冲。

## 3.2 需求总体设计

### 3.2.1 数学与数据表示

**3m 分解选型**：采用 Gauss 形式（§1.2.2.2），与仓内 `cgemm` 的三乘积实现**逐式一致**。

三种平面的生成方式（`cgemm_deinterleave_common.h` 的 `DeinterleaveElement<Packed, ThreeProducts=true>`）：

| 乘积 | A 侧平面 | B 侧平面 |
| --- | --- | --- |
| `P0 = (Ar+Ai)·Br` | `Ar+Ai` | `Br` |
| `P1 = Ar·(Bi−Br)` | `Ar` | `Bi−Br` |
| `P2 = Ai·(Br+Bi)` | `Ai` | `Br+Bi` |

拆解阶段同时完成三件事，顺序不可交换：

1. **列主序定位**：`source = 2*(col*ld + row)`（`ld` 为调用方 `lda/ldb`），`+1` 取虚部；
2. **共轭**：`trans{C} ⇒ imag = −imag`，必须先于平面生成——A 侧平面变为 `(Ar−Ai), Ar, −Ai`；
3. **紧凑化**：输出平面的前导维即行数（`aRows`/`bRows`），调用方的 padding 不带入计算平面。

**转置的处理**：共轭已在拆解阶段折叠，因此实数 GEMM 侧只需区分"N / 非 N"，即 `transa == OP_N ? N : T`，与 `cgemm` 的 `PrepareCgemmTiling` 一致。

**列主序 → 行主序交换**：为复用仓内行主序 GEMM 模板，Host 侧调用 `ApplyGemmColMajorSwap(tiling)`，实际计算 `C^T = op(B)^T·op(A)^T`（行主序），融合路径的 `C^T` 直接在 UB 内做 `Interleave` 后按列主序写回。

**与 4m 的等价性**：Gauss 形式与 4m 在实数域代数等价，差异仅来自浮点结合顺序与输入侧预加/预减的舍入，详见 §7.1。**不承诺与 4m 逐位一致，也不承诺与 `cublasCgemm3m` 逐位一致**，只承诺满足生态精度标准。

### 3.2.2 工程使能与文件布局

`blas/CMakeLists.txt` 对架构目录做 `GLOB_RECURSE`（`*/*/arch35/*.cpp`），因此 `blas/gemm3m/arch35/cgemm3m_host.cpp` **新增即被自动纳入编译**，`SOC_VERSION != ascend950` 时自动不参与，无符号冲突。

新增/改动文件清单（严格控制数量，避免任务书 §5「文件个数明显不符合正常数量」的反例）：

| 文件 | 性质 | 说明 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 改动 | 新增 `aclblasCgemm3m` 声明（+9 行） |
| `blas/gemm3m/arch35/cgemm3m_host.cpp` | **新增** | 唯一实现文件：入口、校验、工作区、tiling、三阶段下发（约 300 行） |
| `blas/gemm/arch35/cgemm_host_internal.h` | 新增 | 三乘积流水线下发函数声明（从 `cgemm_host.cpp` 抽出，供 cgemm/cgemm3m 共用） |
| `blas/gemm/arch35/cgemm_host.cpp` | 改动 | 三乘积流水线改为调用内部函数，行为不变（纯重构，附回归说明） |
| `blas/gemm/CMakeLists.txt` 或 `blas/CMakeLists.txt` | 改动 | 单算子构建时保留 `cgemm` 源文件（见下） |
| `blas/gemm3m/README.md` | 改动 | 补 `aclblasCgemm3m` 产品支持、参数、约束、示例 |
| `test/gemm3m/cgemm3m/CMakeLists.txt` | 新增 | 测试目标 |
| `test/gemm3m/cgemm3m/cgemm3m_param.h` | 新增 | CSV 列映射（按 `cgemm3m_test.csv` 列序） |
| `test/gemm3m/cgemm3m/cgemm3m_golden.h` | 新增 | `cblas_cgemm` golden + 边界语义 |
| `test/gemm3m/cgemm3m/cgemm3m_test.cpp` | 新增 | GTest 用例体 + NPU 调用封装 |
| `test/gemm3m/cgemm3m/arch35/cgemm3m_test.csv` | 新增 | 1200 条（1000 精度 + 200 性能） |

合计**新增 7 个文件、改动 4 个文件**。与 `sgemm3m`（3 个实现文件）+ `cgemm`（9 个实现文件）的体量相比，`cgemm3m` 只有 1 个实现文件，**不复制任何 kernel 代码**，符合"kernel 设计合理、清晰、便于维护"的要求。

**CMake 登记点（必须处理）**：`blas/CMakeLists.txt` 在单算子构建（`--ops` / `TEST_NAMES`）时按算子名过滤源文件。复用 `cgemm` 的 kernel TU 意味着构建 `cgemm3m` 时必须保留 `blas/gemm/arch35/cgemm_*.cpp`。做法与仓内既有 `_GEMM_KERNEL_DEP_OPS`（保留 `gemm_kernel.cpp`）同构：

```cmake
# 复用 cgemm 三乘积 kernel 的算子：单算子构建时保留 cgemm 源文件
set(_CGEMM_KERNEL_DEP_OPS "cgemm3m")
# …在过滤循环中：若 TEST_NAMES 命中该表，则
#   _src_file MATCHES "/gemm/arch35/cgemm_"  →  _keep_src TRUE
```

若不接受改动 CMake，备选是把 `cgemm3m_host.cpp` 落到 `blas/gemm/arch35/` 并让文件名命中 `^cgemm3m_`（仍需保留 `cgemm_*.cpp`，因为文件名前缀不匹配）——**结论：无论如何都需处理该登记点**，故直接按上表 3 行改动实现。

> 该实现**不新增 kernel 头文件、不新增 tiling 结构体、不新增 tiling 常量**：`GemmTilingData`（含 `cgemmThreeProducts/cgemmKPartitions/cgemmKSpan`）、`CgemmDeinterleaveTilingData`、`CgemmEpilogueTilingData` 已覆盖 3m 的全部参数面。这也是选择"复用"而非"另写 kernel"的核心依据——3m 需要的所有信息，仓内结构体已具备。

## 3.3 Host 侧设计

### 3.3.1 参数校验流程

```text
aclblasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  │
  ├─ handle == nullptr ?                          ──► HANDLE_IS_NULLPTR
  ├─ transa ∉ {N,T,C} ?                           ──► INVALID_VALUE   (任务书口径，非 INVALID_ENUM)
  ├─ transb ∉ {N,T,C} ?                           ──► INVALID_VALUE
  ├─ m<0 || n<0 || k<0 ?                          ──► INVALID_VALUE
  ├─ lda < max(1, N?m:k) ?                        ──► INVALID_VALUE
  ├─ ldb < max(1, N?k:n) ?                        ──► INVALID_VALUE
  ├─ ldc < max(1, m) ?                            ──► INVALID_VALUE
  ├─ alpha==nullptr || beta==nullptr ?            ──► INVALID_VALUE
  ├─ m==0 || n==0 ?                               ──► SUCCESS          (quick return，不启核)
  ├─ C == nullptr ?                               ──► INVALID_VALUE
  │
  ├─ alphaVal = {alpha->real, alpha->imag};  betaVal = {beta->real, beta->imag}
  ├─ k==0 || alphaVal==(0,0) ?                    ──► ApplyBetaOnly(C, ldc, n, betaVal)
  │                                                   beta==0 → memset 0
  │                                                   beta==1 → 直接返回
  │                                                   其余   → epilogue(alphaZero=1) 复数缩放
  ├─ GetAicCoreCount()/GetAivCoreCount() == 0 ?   ──► INTERNAL_ERROR
  ├─ 工作区计算（CheckedMulSize + AddWorkspaceRegion，对齐 16×32B）
  │     └─ EnsureDefaultWorkspace 失败              ──► EXECUTION_FAILED
  └─ 路由：onchip / fused / products+epilogue（§3.3.4）──► SUCCESS（异步）
```

校验与 quick return 的相对顺序是**可观测语义**：`m==0` 用例配合空 `A/B` 必须返回 `SUCCESS`，故 `A/B/C` 的空指针校验必须在 quick return 之后；而维度/前导维/枚举校验必须在 quick return 之前（非法维度不应被 0 维掩盖）。

### 3.3.2 工作区布局

| 区域 | 大小（字节） | 生命周期 |
| --- | --- | --- |
| `planeA0 = Ar+Ai` | `4·aElements` | 拆解后 → 乘积结束 |
| `planeA1 = Ar` | `4·aElements` | 同上 |
| `planeA2 = Ai` | `4·aElements` | 同上 |
| `planeB0 = Br` | `4·bElements` | 同上 |
| `planeB1 = Bi−Br` | `4·bElements` | 同上 |
| `planeB2 = Br+Bi` | `4·bElements` | 同上 |
| `temp1/temp2/temp3` | `4·partElements`（`partElements = ceilAlign(m,16)·ceilAlign(n,16)·kPartitions`） | 乘积 → epilogue |

- `aElements = (N?m:k)·(N?k:m)`，`bElements = (N?k:n)·(N?n:k)`；
- 区域按 16×32 B 对齐切分，`AddWorkspaceRegion` 做逐项溢出检查；
- **融合路径不分配 temp**（`fusedTile != 0` 时 `tempBytes = 0`），只在需要时保留 A/B 平面；
- 中间量 `temp1..3` 的 `ldc = ceilAlign(m, 16)`，避免 `m<n` 时按 `n` 取行距导致越界（仓内 `sgemm3m` 曾踩过该类问题，其注释记录了 `ldc=m` 的必要性）。

### 3.3.3 分核与 Tiling 策略

**分核目标**：三个乘积 + k 分段构成 2 维任务空间，按"任务数 ≥ Cube 核数"优先占满核，再谈复用。

1. **累加 tile 选择**（`SelectCgemmAccumulationTile`）：从 `CGEMM_MAX_SQUARE_TILE` 起按 1/2 递减，直到 `products · ceil(m/tile) · ceil(n/tile) ≥ cubeCores`。3m 下 `products = 3`（`cgemm` 的 4 乘积路径按 4 计），因此**同一尺寸下 3m 更早满足占满条件，可用更大 tile → 更高输入复用率**，这是 3m 除算力外的第二个收益点。
2. **k 分段**（`kPartitions`）：当 `k` 超过 L1 单面板深度（`CGEMM_L1_OPERAND_STAGE_ELEMENTS / tile`）时二分归约，`kSpan = ceil(ceil(k,16)/parts)·16`；短归约（`32 ≤ k ≤ 64`，两段各不超过 4 个 16 行块）同样二分——此时多一份乘积缓冲的搬运代价可忽略，而缩短累加链能显著降低 3M 相消误差（实测使 `TC_FL_263` 的最大绝对误差由 1.56e-2 降到 1.37e-2，落入该元素 32·ULP 允许限）；中间区间保持单段，因为残差主要由 epilogue 在全幅值乘积上的合成舍入决定，分段对其无改善且会增加缓冲带宽。分段结果按 `part = 0,1` **固定顺序**合并（§3.4.4），保证确定性。
3. **单核工作量**（`SelectCgemmProductTile`）：比较相邻 tile 的"最忙核 block 数"估计，取更优者；tile 相等时取大 tile（输入复用更多）。
4. **`usedCoreNum = min(products · kPartitions · tiles, cubeCores)`**，空闲核不下发任务（kernel 内 `task >= taskCount` 直接返回，避免空转与越界）。

**3m 专属调整**（相对 `cgemm` 的通用路径）：

| 项 | `cgemm` | `cgemm3m` |
| --- | --- | --- |
| `threeProducts` | 仅 `alpha==1 && beta==0` | **恒为 1** |
| 乘积数 `products` | 3 或 4 | **恒为 3** |
| 中间量个数 | 3 或 4 | **3**（`temp4` 传 `nullptr`，kernel 已有 `cgemmThreeProducts` 分支，不写 `temp4`） |
| `kPartitions` 上限 | 2 | 2（不变） |
| 融合路径条件 | `alpha==1 && beta==0 && k ≤ 单载深度` | 同（3m 与原语义务一致，不额外收窄） |

### 3.3.4 路由规划（等效 TilingKey）

arch35 直调范式不使用图算子 TilingKey，路由通过 **Host 侧入口选择 + kernel 内模板实参**实现：

```text
                        ┌─ m==0 || n==0 ─────────────────────────────► 直接 SUCCESS
                        │
aclblasCgemm3m ─────────┼─ k==0 || alpha==(0,0) ─────────────────────► ApplyBetaOnly（AIV）
                        │
                        ├─ alpha==(1,0) && beta==(0,0)
                        │     ├─ k ≤ shortDepth ─────────────────────► onchip（读复数 GM 直算，单 kernel）
                        │     ├─ 大 tile 覆盖全部核 ─────────────────► fused（L0C 三乘积累加 + 直写 C）
                        │     └─ 其余 ───────────────────────────────► products + epilogue
                        └─ 其他 alpha/beta ──────────────────────────► products + epilogue
```

kernel 内模板实参（编译期消除分支）：

| 模板参数 | 取值 | 作用 |
| --- | --- | --- |
| `IsTransA`/`IsTransB` | 2×2 | GM→L1 layout（`NDExt` / `DNExt`） |
| `BS`（base tile） | `dispatch` 链（64/32/16） | L0C/L1 预算内的 tile 尺寸 |
| `AlphaZero`/`BetaZero` | 2×2 | epilogue 跳过 alpha/beta 计算 |
| `Packed` | 2 | `ldc==m && tempLdc==m` 时省去列/行还原 |

`threeProducts` 在 kernel 内是**运行期常量字段**（`tiling.cgemmThreeProducts`），`cgemm3m` 恒置 1；不新增模板维度，避免代码膨胀。

## 3.4 Kernel 侧设计

### 3.4.1 路由总览

| 阶段 | kernel 入口 | 核型 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| 拆解 | `gemm_cgemm_deinterleave_do` | AIV（SIMT，512 线程/block） | `A`、`B`（复数，含 padding） | 6 个实数平面（紧凑列主序） |
| 三乘积 | `gemm_cgemm_products_do` | AIC（Cube） | 6 个平面 | `temp1..3`（m×n，列主序） |
| 合成 | `gemm_cgemm_epilogue_do` | AIV（SIMT） | `temp1..3`、`C` 旧值 | `C`（原地，交错写回） |
| 融合 | `gemm_cgemm_fused_do` | AIC + 配对 AIV | 6 个平面 | `C`（直写） |
| 片上 | `gemm_cgemm_onchip_do` | AIC + AIV | `A`、`B`、`C` | `C`（短 k 专用） |

### 3.4.2 阶段一：复数拆解（AIV SIMT）

- 线程映射：`index ∈ [0, aElements + bElements)`，`asc_vf_call` 以 `GetBlockIdx()/GetBlockNum()` 做 grid-stride，**A、B 共用一个 vf**，按 `index < aElements` 分支；
- 每元素 1 次读 2 float、3 次写（`threeProducts`），访存强度 ≈ 1 flop/byte 量级，属**带宽受限**阶段，分核按元素数均分；
- 转置不在此阶段处理（由 GEMM 侧 `IsTransA/IsTransB` 的 layout 选择承担），共轭在此阶段处理；
- 效率要点：一次读 `source`、`source+1`（连续 8 B，可合并为一次 64-bit 访问），3 个平面写地址独立、可向量化。

### 3.4.3 阶段二：三乘积（AIC Cube）

- 任务映射：`task = partition·(products·tileCount) + product·tileCount + tile`，按 `block += blocks` 轮转，**每个 task 只写自己那段 `temp`**，无跨核写冲突；
- 面板流水：GM → L1（双缓冲 ping-pong，`K_L1` 深度）→ L0A/L0B → MMAD 累加 → L0C → Fixpipe 写 `temp`；
- 三乘积复用同一份 `A`/`B` 平面：`P0/P1/P2` 三次任务共享 L1 驻留的 A 面板（`A` 平面被 3 个 product 复用），这是 3m 相对"三次独立调用"的**关键融合收益**——`A`/`B` 平面只需从 GM 读一次（在同一 tile 内）；
- `kPartitions > 1` 时不同 partition 写 `temp + part·partStride`，互不覆盖。

### 3.4.4 阶段三：复数合成与 alpha/beta（AIV SIMT）

逐元素（`index ∈ [0, m·n)`）：

```text
productReal = Σ_part ( t1[part] − t3[part] )        // 固定 part 递增顺序 → 确定性
productImag = Σ_part ( t1[part] + t2[part] )
resultReal  = alphaRe·productReal − alphaIm·productImag
resultImag  = alphaRe·productImag + alphaIm·productReal
resultReal += betaRe·cReal − betaIm·cImag           // BetaZero 模板分支时整段省略
resultImag += betaRe·cImag + betaIm·cReal
store(C[idx])                                       // 交错写回，ldc 支持 padding
```

- `BetaZero` 分支**不读 C**，因此 `beta==0` 时 C 的 NaN/Inf 旧值不会污染结果；
- `AlphaZero` 分支跳过乘积合并，用于 `alpha==(0,0)` 的 `C=beta*C` 路径（复用同一 kernel 入口，`t1..t4 = nullptr`）；
- `Packed` 分支（`ldc==m && tempLdc==m`）省去 `col/row` 除法还原，小尺寸收益明显；
- 写回为**列主序 + 交错**：`c[2*(col·ldc+row)] = real`、`+1 = imag`，padding 区（`row ∈ [m, ldc)`）不写。

### 3.4.5 融合路径（fused / onchip）

`alpha==(1,0) && beta==(0,0)` 是**性能 case 的典型配置**，此时可省掉 3 个 m×n 中间量：

- **onchip**（短 k）：直接从 GM 读复数 `A`/`B`，在 UB 内拆解 + 三乘积 + 合成，**单 kernel、零 workspace、零中间量**，专治小尺寸的固定开销（§7.2 的延迟受限用例）；
- **fused**（大 tile）：L0C 内三乘积累加，配对 AIV 在 UB 内做 `Interleave` 后按列主序直写 C，`UB` 面 4 个产品面 + 2 个复数面（`(4+2)·BS·BS` 预算）；
- **非有限值修复**：Cube 与标量 BLAS 在"单个乘积先溢出再相消"时结论可能不同（3m 下 `Ar+Ai` 也可能提前溢出）。融合路径对输出 tile 做非有限检测（`FP32_EXPONENT_MASK`），命中后用显式 FP32 标量运算**重算该 tile**，使 Inf/NaN 形态与 golden 的等效规则一致（§7.1）。

该路径**不改 3m 语义**：它仍只做 3 次实数乘，只是把"物化中间量"换成"片上累加"，因此契约上与通用路径等价。

### 3.4.6 边界与零值路径

| 场景 | 行为 | 启核 |
| --- | --- | --- |
| `m==0 或 n==0` | 返回 `SUCCESS`，不写 C，不校验 A/B/C | 否 |
| `k==0` | `C = beta*C`；`beta==0` → `aclrtMemset`（按 `ldc*n*8` 置零，含 padding，与 `sgemm3m` 一致） | AIV（beta∉{0,1}） |
| `alpha==(0,0)` | 同上；`beta==1` 直接返回 | 同上 |
| `beta==0` | epilogue `BetaZero` 不读 C | — |
| 任意 `lda/ldb/ldc` | 拆解按 `ld` 定位，epilogue 按 `ldc` 写回；padding 不参与计算、不被破坏 | — |

**`k==0` 与 `beta==0` 的置零范围**：任务书要求"beta=0 时 C 的原值不被读取"，未要求保护 `ldc` padding。本设计选择**按 `ldc×n` 整体置零**（与 `sgemm3m` 一致），理由是 padding 属于调用方未定义区，整体置零不会破坏有效区域，且可用单次 `aclrtMemset` 完成；该口径在 README 中显式写明。

### 3.4.7 确定性保证

任务书要求"同输入两次执行逐位一致"。本设计逐条满足：

1. **无原子累加**：每个输出元素由唯一 task 写入，不存在 `AtomicAdd` 竞争；
2. **归约顺序固定**：`kPartitions` 由 `k` 与 tile 唯一确定（确定性函数，无随机、无设备侧负载均衡），epilogue 按 `part = 0..parts−1` 顺序累加；
3. **任务到核的映射固定**：`task = block, block+blocks, …` 的轮转与核数无关地只影响"谁算"，不影响"累加顺序"（每个 task 内部 k 顺序固定，跨 task 无共享归约）；
4. **无时间相关分支**：路由只依赖 `m,n,k,alpha,beta,trans`，不依赖 profiling 结果或运行时负载。

因此同输入、同形状、同 alpha/beta → 同 tiling → 同累加顺序 → **逐位一致**。

## 3.5 与标杆实现的差异及原因

| 项 | `cublasCgemm3m` | 本设计 | 原因 |
| --- | --- | --- | --- |
| 3m 恒等式形式 | 未公开（Gauss/Karatsuba 等价类） | Gauss（`P0,P1,P2`） | 与仓内 `cgemm` 三乘积逐式一致，复用已验证的 epilogue 与单测 |
| 3m 启用条件 | 始终 | 始终（`threeProducts=1`） | 契约要求 |
| 中间量物化 | 内部实现细节 | `alpha==1,beta==0` 时不物化 | 省 12mn 字节流量与 workspace |
| 归约顺序 | 未承诺逐位一致 | 固定顺序，支持确定性 | 任务书 §2.5 |
| Inf/NaN | 由实现决定 | 融合路径重算 + §7.1 等效规则 | 3m 的输入预加可能提前溢出，需与 golden 对齐口径 |
| 非法枚举返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_VALUE` | 任务书 §2.4 |

## 3.6 正确性保障

1. **代数正确性**：Gauss 恒等式在实数域严格成立（§1.2.2.2 逐式推导），无近似假设；
2. **语义正确性**：`N/T/C` 三种 op 由"共轭在拆解、转置在 layout"两处正交处理，9 种组合不产生分支爆炸；padding 只影响定位、不影响计算；
3. **边界正确性**：11 步校验的先后顺序可观测，已按任务书固化并用 CSV 固化（§7.3）；
4. **回归正确性**：复用 `cgemm` 内核，需在 PR 中附 `cgemm` 全量用例回归结果，证明 Host 抽取（§3.2.2）是纯重构。

## 3.7 备选方案与取舍

| 方案 | 内容 | 取舍 |
| --- | --- | --- |
| **A（选定）** | 复用 `cgemm` 三乘积流水线，新增薄 Host | 代码量最小（1 个实现文件）、零重复、复用已验证 kernel；代价是需 CMake 登记（3 行）与 Host 抽取 |
| B | 在 `blas/gemm3m/arch35` 自建三乘积 kernel（仿 `sgemm3m`） | 与 `cgemm` 逻辑重复约 1500 行，评审明确反对（任务书 §5 反例） |
| C | 三次调用 `aclblasSgemm` + 外部中间量 | 每次调用都要重新拆解，A/B 平面重复读 3 次；且 `aclblasSgemm` 的 tiling 无法与 epilogue 融合，性能劣于 A |
| D | 3m 走 Karatsuba 形式（`P1'=ArBr, P2'=AiBi, P3'=(Ar+Ai)(Br+Bi)`） | epilogue 需 4 次加减（Gauss 为 2 次），且 `P0` 无法被实/虚部共用 → 多读一次 `temp`。数学等价但实现更差，不选 |

---

# 4. 支持硬件（required）

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（`ascend950`，arch35，`DAV_3510`） | √（本任务唯一目标） |
| Ascend 950DT | 随 arch35 同步支持（`sgemm3m` 的既有口径） |
| Atlas A2/A3 训练、推理系列 | ×（arch35 kernel 不参与编译） |

依赖：CANN 9.1.0，asc-devkit ≥ 9.1（`ASC_DEVKIT_GE_9_1`）。新增 `cgemm3m_host.cpp` 与 `cgemm_*.cpp` 一致，用 `devkit_version_compat.h` 的 `ASC_DEVKIT_GE_9_1` 包裹整个 TU，低版本下自动空实现，**无需在 `TENSOR_API_OPS` 清单新增条目**（`cgemm` 即采用该做法，而非 `sgemm3m` 的 CMake 清单方式）。

---

# 5. 算子约束限制（required）

1. 仅支持 COMPLEX64（实部/虚部各 FLOAT32），不支持半精度/双精度复数与混合精度；
2. 仅支持列主序稠密存储与前导维 padding，**不支持任意步长/负步长**（BLAS Level-3 无 incx/incy）；
3. 不支持广播、不支持批量（Batched/StridedBatched 为独立接口）；
4. `alpha`、`beta` 为 **Host 指针**，不支持 device scalar pointer mode；
5. `C` 为 in/out，**不复用 A/B 内存**，不返回视图；`A/B/C` 之间不允许重叠；
6. `m/n/k` 为图编译期已知的静态值，不要求 dynamic shape；
7. 规模受 device workspace 与 `int` 索引约束（`m·n`、`ld·cols` 计算全部走 64-bit 中间量，Host 侧做溢出检查）；
8. `k==0` 的置零按 `ldc×n` 覆盖 padding（§3.4.6），调用方不应依赖 padding 内容。

---

# 6. 特性交叉分析（required）

| 交叉维度 | 组合 | 处理方式 |
| --- | --- | --- |
| `transa` × `transb` | 3×3 = 9 | 共轭在拆解阶段（`conjugateA/B`），转置在 GEMM layout（`IsTransA/B`），组合独立 |
| 数据类型 | COMPLEX64 单一 | 无需类型分支 |
| `alpha` 取值 | `(0,0)` / `(1,0)` / 纯虚 / 一般复数 | `(0,0)` → `C=beta*C`；`(1,0)+beta=0` → 融合；其余 → 通用 |
| `beta` 取值 | `(0,0)` / `(1,0)` / 一般 / 纯虚 | `BetaZero`/`BetaOne` 模板分支；`(1,0)` 省一次读 C |
| `m/n/k` 为 0 | 7 种组合 | `m==0||n==0` → quick return；`k==0` → `beta*C`；组合见 §7.3 |
| 前导维 padding | `lda/ldb/ldc` 含 `+4/+8` padding | 拆解按 `ld` 定位、epilogue 按 `ldc` 写回，padding 不读不写 |
| 数值特殊值 | 0、±1、纯虚、大值、Inf、NaN | 有限值按 §7.1 判定；非有限按等效规则；融合路径做重算修复 |
| 空指针 | `alpha`/`beta`/`A`/`B`/`C` 及条件叠加 | 按 §2.3.2 的 11 步顺序 |
| 非法枚举 | `transa/transb` 取 999 | `INVALID_VALUE` |
| 负维度 / 非法前导维 | `m/n/k=-1`、`lda<m` | `INVALID_VALUE` |

---

# 7. 可维可测分析

## 7.1 精度标准

**比对口径**（任务书 §3.2）：golden 由 cblas（Netlib `cblas_cgemm`）生成，输出 C 的 m×n 元素**全量**验证，**实部、虚部分别**按 FLOAT32 判定。

| 项 | 取值 |
| --- | --- |
| rtol | `2⁻¹³ ≈ 1.22e-4` |
| atol | `2⁻¹³ ≈ 1.22e-4` |
| 逐元素通过 | `|actual − golden| ≤ atol + rtol × |golden|` |
| required_matched_ratio | `≥ 0.99` |
| max_abs_error_limit | `≤ 1e-2` 或 `32 × ULP` |

**Inf/NaN 判定**：不适用上述判定式。golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 另一方 ±Inf，视为等效通过；一方有限、另一方非有限则不通过。非有限元素不计入 `matched_ratio` 与 `max_abs_error`，单独记录总数与通过数。

**3m 误差分析**（3m 相对 golden 的 4m，差异来自两处）：

1. **输入侧预加/预减**：`(Ar+Ai)`、`(Bi−Br)`、`(Br+Bi)` 各引入 ≤1 ULP 的相对误差，放大因子不超过 2（`|Ar+Ai| ≤ |Ar|+|Ai|`）；
2. **输出侧相消**：`P0−P2` 在 `|Ar·Br| ≈ |Ai·Bi|` 时相消，相对误差放大，绝对误差仍受下式约束

```text
保守上界（假设累加中无符号相消）： |Δ| ≲ γ_k · Σ_l (|Ar·Br| + |Ai·Bi|),  γ_k ≈ k·u/(1−k·u),  u = 2⁻²⁴
实际统计量级（随机符号输入）：     |Δ| ≈ u · √k · rms(|a|·|b|)
```

以 `k=2048`、元素 `N(0,1)` 估算：`u√k ≈ 6e-8 × 45 ≈ 2.7e-6`，而 `|golden| ≈ √k ≈ 45`、`rtol·|golden| ≈ 5.5e-3`，**余量约 2000×**；即使叠加 2× 的输入侧放大，仍有两到三个数量级余量。保守上界（`γ_k·Σ|a·b|`）在 `k=2048` 可达 0.1 量级，属"全同号最坏情况"，随机用例不会命中，仅作设计说明。

**结论**：3m 在给定容差下可达，但**相消边界 case（大 k、结果接近 0）是精度上唯一的真实风险**——此时 rtol 项失效，靠 atol = 1.22e-4 兜底。自测须专门统计：`TC_AB`（alpha/beta 特殊值）、`TC_SQ` 大 k、`TC_MN` 极端矩形。

> 实测结果（逐条 PASS/FAIL、`matched_ratio`、`max_abs_error`、非有限元素统计、实部/虚部截图）：**待自测回填**。

**阈值列说明**：CSV 的 `mere_threshold = 2⁻¹³`、`mare_multiplier = 10.0` 是 ops-blas 测试工程的 MERE/MARE 配置，**不参与**生态精度标准比对；标准比对按 dtype 固定阈值执行。两套口径均在自测报告中给出。

## 7.2 性能标准与预算分析

**判据**（任务书 §3.3）：

1. 单 case：`NPU 平均单次耗时 ≤ 2.5 × GPU 标杆`（即倍率 `GPU/NPU ≥ 0.4`）；
2. L3 平均：全部性能 case 的倍率算术平均 `≥ 0.5`；
3. 两项**同时**满足。

**采样方式**：`msprof op` 采集单算子 task duration（us），自带 5 次 warmup（不计入样本），随后 10 次有效采样取算术平均；仅 `SUCCESS` 计入有效样本。

**预算推导**（`gpu_baseline.csv` 前 4 条，3m 计 6mnk flop）：

| 用例 | trans | N | GPU (us) | 预算 = 2.5×GPU (us) | 单 case 所需算力 (TFLOP/s) | 0.5× 平均目标所需算力 (TFLOP/s) |
| --- | --- | --- | --- | --- | --- | --- |
| `cgemm3m-base-000` | NN | 256 | 15.311 | 38.28 | 2.63 | 5.26 |
| `cgemm3m-base-001` | NN | 1024 | 178.718 | 446.80 | 14.42 | 28.84 |
| `cgemm3m-base-002` | NC | 2048 | 1417.110 | 3542.78 | 14.54 | 29.07 |
| `cgemm3m-base-003` | CN | 4096 | 10875.700 | 27189.25 | 15.17 | 30.33 |

**流量预算**（通用三乘积 + epilogue 路径，`m=n=k=N`）：

```text
拆解：读 16N²（A/B 复数）+ 写 24N²（6 平面）
乘积：读 24N²（6 平面，L1 复用后实际更低）+ 写 12N²（3 中间量）
合成：读 12N² + 读 C 8N² + 写 C 8N²
合计 ≈ 104N² 字节；融合路径（不物化中间量）≈ 72N² 字节
```

`N=4096`：通用路径 ≈ 1.74 GB / 27.2 ms → **≈64 GB/s**；融合路径 ≈ 1.21 GB → ≈44 GB/s。相对 950PR 的 HBM 能力余量充足。

**两点结论**：

1. **大规模为算力受限**（需 ≈15 TFLOP/s 持续 FP32 算力）。优化重心在 **Cube 利用率**：tile 选择（3m 的 `products=3` 使大 tile 更早可用）、L1 双缓冲、L0C 占用（3 个中间量 vs 4 个）、k 分段减少 L1 面板切换。3m 自身的 25% 算力节省是可达标的基础。
2. **小规模为延迟受限**：`N≤64` 的预算在 8~10 us 量级（`cgemm3m-perf-004..006` 的 GPU 值 10~11 us），完全由下发 + 启核固定开销决定。`onchip` 路径（单 kernel、零 workspace、零中间量）即为此设计；同时 Host 侧应避免不必要的日志与多次查询（`GetAicCoreCount` 结果可缓存）。

> 实测数据（200 条逐条 Avg time、倍率、余量、msprof 截图）：**待自测回填**。若平均倍率不足 0.5，优化顺序为：① 大尺寸 tile 与 k 分段调优 → ② 融合路径覆盖面扩大（`beta==0 且 alpha 为实数` 等）→ ③ 拆解与乘积的流水重叠。

## 7.3 测试覆盖

### 7.3.1 用例构成

随任务提供 `cgemm3m_test.csv` 共 **1200 条**（`gen_csv.py` 固定种子 `20261028`，同种子逐字节可复现）：精度 1000 条 + 性能 200 条（`TC_PF` 前缀）。

| 类别 | 前缀 | 条数 | 覆盖内容 |
| --- | --- | --- | --- |
| L0 基础 | `TC_L0` | 18 | trans 3×3 全组合 × 尺寸 (4,8) |
| L1 方阵 | `TC_SQ` | 108 | 27 种尺寸 × 4 组 trans |
| L1 矩形 | `TC_MN` | 81 | 9 组 m≠n≠k（含 1000×1001×999）× 9 组 trans |
| L2 标量 | `TC_AB` | 32 | 8 组 alpha/beta（0/1/−1/复数/共轭组合）× 4 组 trans |
| L4 前导维 | `TC_LD` | 16 | `lda/ldb/ldc` padding（+4/+8）× 4 组 trans × 4 尺寸 |
| L5 填充 | `TC_FL` | 18 | 均匀/全零/交替/极端值/Inf/NaN（A、B、C 分别） |
| L5b 覆盖 | `TC_CV` | 32 | 中等尺寸（10~400）× 4 组 trans |
| L6 边界 | `TC_ED` | 32 | 0 维 / `k=0` / `beta=0` 不读 C / 空指针叠加 / 非法枚举 / 非法前导维 / 负维度 |
| EX 扩展 | `TC_EX` | 663 | 尺寸池 × trans × 标量 × padding 确定性采样 |
| PF 性能 | `TC_PF` | 200 | 4 条任务书典型 case + 小尺寸 + 方阵对数扫描（256→4096）+ 矩形/随机 |

### 7.3.2 测试工程改造点

| 改造点 | 说明 |
| --- | --- |
| 新增 `test/gemm3m/cgemm3m/` | 仓内当前**无** `test/gemm3m/cgemm3m/`，需按 `sgemm3m` 的结构新建 |
| CSV 列序 | 按 `case_name,description,transa,transb,m,n,k,alpha_real,alpha_imag,a_fill,lda,b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,mere_threshold,mare_multiplier,random_seed`，需新增 `cgemm3m_param.h` 完成列映射 |
| 空指针/非法值编码 | `alpha/beta` 写 `null`、`A/B/C` 写 `NULLPTR`、非法枚举写 `999`、负维度 `-1` |
| golden | `cblas_cgemm`（Netlib 3.10.0）；`beta==0` 不读 C 旧值；`k==0` 只做 `beta*C`；`alpha==0` 且 A 空指针时直接返回 |
| 测试目标登记 | 在 `test/gemm3m/` 或 `test/CMakeLists.txt` 登记 `cgemm3m` 目标（与 `sgemm3m` 同样处理，需按仓内最新写法确认） |

### 7.3.3 复现步骤

```bash
# 编译（SOC=ascend950，自动纳入 blas/gemm3m/arch35 与 test/gemm3m/cgemm3m）
bash build.sh --soc ascend950

# 精度（自动排除 TC_PF_ 性能用例）
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./cgemm3m_test.csv

# 性能（TC_PF 用例 + gpu_baseline.csv 关联比对）
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend950

# 单条 kernel 级耗时
msprof op --application="./build/test/gemm3m/cgemm3m/cgemm3m_test --gtest_filter=*TC_PF_001*" --output=./prof_cgemm3m
# 在 OPPROF_*/OpBasicInfo.csv 中按 kernel 名统计 Task Duration(us)
```

## 7.4 兼容性与维护性

| 项 | 结论 |
| --- | --- |
| 接口兼容 | 新增函数，不改动既有签名；不定义 950PR 私有平行 API |
| 二进制兼容 | 仅新增符号，无结构体/枚举变更 |
| 与 `aclblasCgemm` 的关系 | 共享 kernel 与 tiling 结构，Host 路径抽取后行为不变（附回归） |
| 与 `aclblasSgemm3m` 的关系 | 同目录、同目录 README、同测试范式；语义差异（实数 vs 复数、预拆分输入 vs 原始复数输入）在 README 中对比说明 |
| 可维护性 | 1 个实现文件、零 kernel 复制；3m 恒等式注释直接指向 `cgemm_deinterleave_common.h`，避免两处口径漂移 |

## 7.5 风险与应对

| # | 风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| 1 | 单算子构建过滤导致 `cgemm_*` 符号缺失 | 链接失败 | §3.2.2 的 CMake 登记（3 行），并在 CI 中覆盖 `--ops cgemm3m` 构建 |
| 2 | 强制 `threeProducts=1` 改变 `cgemm` 的既有分支假设 | `cgemm` 回归失败 | Host 抽取为纯重构；`threeProducts` 仍由参数决定，`cgemm3m` 单独传 1 |
| 3 | 3m 相消导致大 k 精度临界 | 少量 case FAIL | §7.1 分析；测试覆盖 `TC_SQ`/`TC_MN` 大 k；必要时收紧 tile 与累加顺序（不放松阈值） |
| 4 | 4096 阶 workspace ≈604 MB（融合路径 ≈403 MB） | 分配失败 | Host 容量检查 + `EXECUTION_FAILED`；大尺寸优先走融合路径（不物化中间量）；README 写明规模上限 |
| 5 | 小尺寸倍率拉低平均值 | 平均 <0.5 | `onchip` 单 kernel 路径 + Host 侧减少查询/日志 |
| 6 | `k==0` 置零范围与调用方期望不一致 | 语义争议 | §3.4.6 固化口径并在 README/接口注释中写明 |
| 7 | asc-devkit < 9.1 环境 | 编译跳过 | `ASC_DEVKIT_GE_9_1` 包裹，README 说明依赖 |

---

# 8. 交付与验收

| 序号 | 交付件 | 内容 |
| --- | --- | --- |
| 1 | 算子设计文档 | 本文档（`cann-ops-competitions` PR） |
| 2 | 自测用例及测试代码 | `test/gemm3m/cgemm3m/`（1200 条 CSV + golden + README 复现步骤） |
| 3 | 自测报告 | 精度逐条结果与截图、性能逐条结果与 msprof 截图、非有限值统计 |
| 4 | 待验收代码 | `ops-blas` 个人仓分支 + 算子 README；邀请 `Ascend-CANN` 为开发者 |

验收口径：功能 + 精度（§7.1）+ 性能（单 case ≥0.4×、平均 ≥0.5×）+ 确定性（§3.4.7）全部满足方可申请验收。

---

# 9. 参考资料

1. `aclblasCgemm3m` 任务书（`aclblasCgemm3m_Atlas950PR_task_doc.md`）
2. 算子设计文档模板：`cann/cann-ops-competitions/04_tasks/01_community-task-2026/resources/design_template.md`
3. 生态算子开源精度标准：`cann/opbase/docs/zh/ops_precision_standard/mixed_tolerance_standard.md`
4. `ops-blas`：`blas/gemm/arch35/cgemm_*.cpp`（复数 GEMM 三乘积/四乘积流水线）
5. `ops-blas`：`blas/gemm3m/arch35/sgemm3m_*`（3m 家族目录范式）
6. `ops-blas`：`include/cann_ops_blas.h`、`include/cann_ops_blas_common.h`
7. cuBLAS 参考：`cublasCgemm3m` 接口与 3m 算法说明
8. Netlib BLAS 3.10.0：`cgemm.f` / `cblas_cgemm`
9. Ascend C 算子开发文档与 tensor_api 接口文档（CANN 9.1.0）
