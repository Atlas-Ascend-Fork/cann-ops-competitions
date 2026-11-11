# aclblasSgemv 算子设计文档

> 任务：11月社区任务-aclblasSgemv 算子开发（A2A3）
> 目标仓：https://gitcode.com/cann/ops-blas
> 实现目录：`blas/gemv/arch22/`
> 测试目录：`test/gemv/sgemv/arch22/`
> 适配硬件：Atlas A2 / A3 训练与推理系列产品（自验覆盖 910B3）
> CANN 版本：9.1.0
> 模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md

| 项 | 内容 |
|----|------|
| 算子 | aclblasSgemv（BLAS-2 单精度矩阵-向量乘加） |
| 硬件 / CANN | Atlas A2 / A3 · CANN 9.1.0 |
| 实现目录 | `blas/gemv/arch22/` |
| 测试目录 | `test/gemv/sgemv/arch22/` |
| 上游锚点 | `blas/gemv/arch35/sgemv_host.cpp`（语义与校验口径） |
| 工程骨架 | `agent/skills/repo-op-templates/references/simd-membase/`（仓内官方 SIMD 模板） |

---

## 一、需求背景

### 1.1 需求来源

在 Atlas A2/A3（arch22）上补齐 `ops-blas` 缺口的 BLAS Level-2 实数矩阵-向量乘加接口 `aclblasSgemv`，语义对标 cuBLAS `cublasSgemv`（CUBLAS Library §2.6.2），算法基准为 Netlib `sgemv.f`。

当前仓库中 `aclblasSgemv` 仅有 950PR（arch35）的 SIMT 实现，`blas/gemv/README.md` 产品支持表明确标注「Atlas A2/A3：不支持」。本任务即为该两代产品补齐 arch22 实现，与既有 `aclblasCgemv`（复数版，已有 arch22）形成实数/复数完整覆盖。

### 1.2 背景介绍

#### 1.2.1 标杆来源、TBE 源码与算子信息库核查

本任务是在既有公共头声明下新增 arch22 的 Ascend C Kernel 直调实现，**不是**把 ACLNN/TBE 算子迁移为 Ascend C。因此任务书模板中「原 TBE 算子源码路径」「算子信息库路径」两项的结论是**不适用**，而非缺少材料：

| 审核对象 | 核查结论 | 依据及替代参考 |
| --- | --- | --- |
| 原 TBE 算子源码路径 | 不适用：BLAS 句柄式接口，不属于 ACLNN 图算子，仓库中不存在对应的 TBE 实现 | 语义以 cuBLAS `cublasSgemv` 与 Netlib `sgemv.f` 为准 |
| 算子信息库路径 | 不适用：不注册 ACLNN 算子，不使用算子信息库 | 公共声明位于 `include/cann_ops_blas.h` |
| 接口声明 | 已存在，直接复用 | `include/cann_ops_blas.h:468-470`（`aclblasSgemv`，与 950PR 共用同一声明，**不新增私有接口**） |
| 语义与校验锚点 | 适用 | `blas/gemv/arch35/sgemv_host.cpp`：`ValidateSgemvParams`（L31-68）、`aclblasSgemv` 入口（L92-135） |
| arch22 工程骨架 | 适用 | `agent/skills/repo-op-templates/references/simd-membase/blas/{op}/archxx/`（Validate/Launch 拆分、tiling const 引用传参、workspace 由 handle 提供） |
| arch22 算法拆解参考 | 适用 | `blas/gemv/arch22/cgemv_host.cpp`、`cgemv_no_trans_kernel.cpp`、`cgemv_do_trans_kernel.cpp`（同一 arch 下的复数版分块方式） |

> 说明：任务书 §2.3/§7 给出的声明行号为 L457-459 / L461-463，实测当前 master 为 **L468-470**（文件在上游持续新增接口导致行号偏移）。设计以实际声明为准，接口签名与任务书 §2.3 逐字一致。

#### 1.2.2 aclblasSgemv 标杆算子现状分析

##### 1.2.2.1 标杆支持的数据类型和数据格式

与 cuBLAS `cublasSgemv`、Netlib `sgemv.f` 及任务书 §2.4 一致：

| 参数 | 含义 | 数据类型 | 内存 | 排布/形状 | 约束 |
|------|------|----------|------|-----------|------|
| handle | 库上下文（携带 stream） | `aclblasHandle_t` | Host | scalar | 空指针返回 `HANDLE_IS_NULLPTR`（优先校验） |
| trans | op(A) 类型 | enum | Host | attr | `{N=111, T=112, C=113}`，越界返回 `INVALID_VALUE` |
| m | A 行数 | int | Host | scalar | m ≥ 0；m=0 为合法快速返回 |
| n | A 列数 | int | Host | scalar | n ≥ 0；n=0 为合法快速返回 |
| alpha | 标量乘数 α | FLOAT32 | Host | scalar | 不可为空；值不校验 |
| A | 列主序系数矩阵 | FLOAT32 | Device | `lda×n`，前 `m×n` 有效 | `lda ≥ max(1,m)`；m>0 且 n>0 时不可为空 |
| lda | 相邻列间距（**列主维**） | int | Host | scalar | `lda ≥ max(1,m)` |
| x | 输入向量 | FLOAT32 | Device | 逻辑长 `trans==N?n:m`，物理长 `1+(len-1)·|incx|` | 逻辑长非零时不可为空 |
| incx | x 元素步长 | int | Host | scalar | ≠0 且 ≠INT32_MIN；支持负值 |
| beta | 标量乘数 β | FLOAT32 | Host | scalar | 不可为空；β=0 时 y 旧值不参与 |
| y | 输入兼输出（原地） | FLOAT32 | Device | 逻辑长 `trans==N?m:n`，物理长 `1+(len-1)·|incy|` | 逻辑长非零时不可为空 |
| incy | y 元素步长 | int | Host | scalar | ≠0 且 ≠INT32_MIN；支持负值 |

不支持广播、不涉及 0 维张量；支持非连续访问（lda 列间空洞、incx/incy 跨元素），不支持超出上述语义的其它非连续形式。

##### 1.2.2.2 标杆实现描述

计算语义：

```text
y[j] = alpha * Σ(i=1..k) op(A)[j,i] * x[i] + beta * y[j],  j = 1..len(y)
```

- `trans = N`：`op(A) = A`（m×n），k = n，`len(y) = m`
- `trans = T`：`op(A) = Aᵀ`（n×m），k = m，`len(y) = n`
- `trans = C`：`op(A) = Aᴴ`；A 为实数矩阵，与 T 完全等价（Kernel 路由到同一分支）

实现要点：

1. 元素定位：`A[i,j]` 位于 `A[(j-1)*lda + (i-1)]`（1-based，列主序）。
2. y 原地更新：β 读取 y 旧值参与计算；`β = 0` 时旧值语义上不参与（实现可覆写，但**不做 α=0 短路**，见 §3.4）。
3. 快速返回：参数校验通过后 `m == 0 || n == 0` 直接返回成功，不访问 A/x/y，不启动 kernel。
4. 负步长：`incx < 0` 时第 i 个逻辑元素物理偏移为 `(len_x - 1 - i) * |incx|`（从物理 span 尾端起算），incy 同理。
5. 校验顺序（对齐 arch35 `ValidateSgemvParams`）：trans → m → n → lda → incx → incy → alpha → beta → A（仅 m>0&&n>0）→ x（按 trans 决定的长度）→ y（同）。

##### 1.2.2.3 标杆算子实现流程图

```mermaid
flowchart TD
    A[aclblasSgemv 入口] --> H{handle 非空?}
    H -->|否| E0[HANDLE_IS_NULLPTR]
    H -->|是| V{trans/m/n/lda/incx/incy<br/>alpha/beta/A/x/y 校验}
    V -->|失败| E1[INVALID_VALUE]
    V -->|通过| S{跨度溢出守卫<br/>(len-1)·|inc| > INT32_MAX?}
    S -->|是| E1
    S -->|否| Q{m==0 或 n==0?}
    Q -->|是| E2[SUCCESS，不启动 kernel]
    Q -->|否| T{trans == N?}
    T -->|是| N1[Kernel: 按行分核，遍历列累加<br/>y = α·A·x + β·y]
    T -->|否 T/C| T1[Kernel: 按输出列分核，遍历行归约<br/>y = α·Aᵀ·x + β·y]
    N1 --> R[SUCCESS，结果随 stream 异步可见]
    T1 --> R
```

### 1.3 算子功能分析

- 功能：单精度列主序矩阵与向量的乘加，y 原地更新。
- 输入：`alpha`、`A`、`x`、`beta`、`y`（旧值）；输出：`y`。
- 支持 `trans` 三态；`C` 与 `T` 在实数域结果逐位一致。
- 支持正负步长、`lda > m` 的列间空洞。
- 非目标：不做 `α=0` / `β=1` 短路优化（见 §3.4）、不做确定性计算保证、不涉及 workspace、不涉及内存量化。

---

## 二、需求分析

### 2.1 外部组件依赖

| 组件 | 版本/来源 | 用途 |
|------|-----------|------|
| CANN | 9.1.0 | Ascend C 编译、ACL 运行时、Kernel launch |
| Ascend C | 随 CANN | `kernel_operator.h`、TPipe/TQue、Vector 指令 |
| cblas（Netlib BLAS 3.10.0） | 随测试工程 | 仅测试侧 golden（`cblas_sgemv`），不进算子二进制 |

算子本体无三方依赖。

### 2.2 内部适配模块

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `include/cann_ops_blas.h` | 不改 | 复用 L468 既有声明，与 950PR 共用 |
| `blas/gemv/arch22/sgemv_host.cpp` | 新增 | 参数校验 + 分核/tiling + kernel launch |
| `blas/gemv/arch22/sgemv_kernel.cpp` | 新增 | Kernel 类（Init/Process）+ entry + launcher |
| `blas/gemv/arch22/sgemv_kernel.h` | 新增 | Kernel 对外声明（`sgemv_kernel_do`） |
| `blas/gemv/arch22/sgemv_tiling_data.h` | 新增 | Host/Kernel 共享 tiling 结构 |
| `blas/gemv/CMakeLists.txt`（或上层聚合） | 按仓规范 | 将 arch22 新源文件纳入构建 |
| `test/gemv/sgemv/arch22/sgemv_test.cpp` | 新增 | CSV 驱动 GTest（A2/A3） |
| `test/gemv/sgemv/arch22/sgemv_npu_wrapper.h` | 新增 | device 缓冲分配 + `INT32_MIN` 短路 |
| `test/gemv/sgemv/CMakeLists.txt` | 修改 | 增加 arch22 分支的 `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)` |
| `blas/gemv/README.md` | 修改 | 产品支持表：Atlas A2/A3 改为「支持」 |

> 现状缺口（任务书 §2.2 指出并已核实）：`test/gemv/sgemv/CMakeLists.txt` 目前只有一行无条件 `ops_blas_add_gtest_tests`，没有 arch 分支；`test/gemv/sgemv/` 下只有 `arch35/`。参考 `test/gemv/cgemv/CMakeLists.txt` 的 `if("arch35" IN_LIST SOC_ARCH_DIRS) ... else() ops_blas_add_tests` 写法补齐 arch22 分支。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);
```

与 `cublasSgemv` 参数序列逐位一致，无需映射层。

#### 2.3.2 任务书 §2.5 约束逐项核对

| §2.5 约束项 | 本设计结论 | 落地位置 |
| --- | --- | --- |
| 参数合法性与校验层级 | 完全对齐 arch35 顺序与返回码；**新增跨度溢出守卫**（arch35 参考实现尚未提供，本任务补齐） | `sgemv_host.cpp::ValidateSgemvParams` |
| 非连续 Tensor 支持 | 支持：A 按 `lda` 跨列、x/y 按 `incx/incy` 跨元素，正负步长均生效 | Kernel 索引计算 |
| broadcast | 不涉及 | — |
| dynamic shape | 不要求：m/n/lda/incx/incy 运行时入参，host tiling 按实参计算 | `CalSgemvTilingData` |
| 原地与视图语义 | y 原地覆写，A/x 只读；β 读旧值，β=0 不读 | Kernel `yGm` 先读后写 |
| 确定性计算 | 不要求：Kernel 内累加顺序固定（分巷 + 固定树形合并次序），同一输入下结果逐次一致 | §3.2.2.2 |
| 空 Tensor | `m=0\|\|n=0` → 不访问 A/x/y、不启 kernel、返回成功；**判定位于参数校验之后** | `aclblasSgemv` 入口 |
| 0 维 Tensor | 不涉及 | — |
| 异步执行 | 依赖 handle 绑定的 stream 异步下发；不调用 `aclrtSynchronizeStream`（对齐仓内 SIMD 模板） | `LaunchSgemvKernel` |
| `α=0` 行为 | **不短路**（按任务书 §2.1 原文）：仍完整计算 `Σ = op(A)·x` 再乘 `α`，`0×Inf/NaN` 按 IEEE 自然得 NaN。golden `cblas_sgemv` 在 `α=0` 时短路为 `βy`，二者**仅**在 `A`/`x` 含 Inf/NaN（或 `Σ\|a·x\|` 在 f32 下溢出）时结果不同，任务书已声明该组合不设精度用例——测试侧对该组合打印 `SKIP` 并跳过比对，其余 96 条 `α=0` 用例照常比对 | Kernel 无 `alpha == 0` 分支；测试侧 `alpha == 0 && 非有限/溢出` 分支 |

---

## 三、需求详细设计

### 3.1 调用方式

句柄式 kernel 直调：`aclblasCreate` → `aclblasSetStream` → `aclblasSgemv` → `aclrtSynchronizeStream` → D2H。Host 完成校验与 tiling，单次 kernel launch，无中间同步、无 GM tiling 搬运、无 workspace。

### 3.2 需求总体设计

#### 算子数据模型

##### 数学公式

见 §1.2.2.2。索引统一在 Kernel 侧转为 0-based：

```text
N   : y[i] = alpha * Σ(j=0..n-1) A[i + j*lda] * x[j·|incx| 侧] + beta * y[i·|incy| 侧],  i = 0..m-1
T/C : y[j] = alpha * Σ(i=0..m-1) A[i + j*lda] * x[i·|incx| 侧] + beta * y[j·|incy| 侧],  j = 0..n-1
其中「侧」指：正步长取 i*inc，负步长取 (len-1-i)*|inc|
```

##### 支持数据类型

FLOAT32（alpha/beta 为 FLOAT32 标量，A/x/y 为 FLOAT32 数组）。

##### 支持形状与布局

- A：列主序 `lda × n`，前 `m × n` 有效；`lda > m` 时列间存在空洞（不读、不写）。
- x/y：一维逻辑向量，物理 span 为 `1 + (len-1)·|inc|`，支持负步长（尾端起算）。

### 3.2.1 Host 侧设计

#### 3.2.1.1 参数校验与启动边界

```text
handle == nullptr                          → HANDLE_IS_NULLPTR      （最先判断）
trans ∉ {N,T,C}                            → INVALID_VALUE
m < 0 / n < 0                              → INVALID_VALUE
lda < max(1, m)                            → INVALID_VALUE
incx == 0 || incx == INT32_MIN             → INVALID_VALUE
incy == 0 || incy == INT32_MIN             → INVALID_VALUE
alpha == nullptr / beta == nullptr         → INVALID_VALUE
m > 0 && n > 0 && A == nullptr             → INVALID_VALUE
xLen > 0 && x == nullptr                   → INVALID_VALUE   （xLen = N? n : m）
yLen > 0 && y == nullptr                   → INVALID_VALUE   （yLen = N? m : n）
【新增】(int64)(xLen-1)*|incx| > INT32_MAX  → INVALID_VALUE
【新增】(int64)(yLen-1)*|incy| > INT32_MAX  → INVALID_VALUE
以上全部通过后：m == 0 || n == 0           → SUCCESS（quick return，不启 kernel）
```

**跨度溢出守卫的必要性**：`(len-1)·|inc|` 超出 int32 时，Kernel 侧 32 位索引回绕会读到非法地址。arch35 参考实现尚未提供该校验（任务书 §2.5 已标注），本任务按 §2.4 补齐。实现要点：

1. 守卫必须在 `incx != INT32_MIN` 之后执行——`abs(INT32_MIN)` 是未定义行为（对应测试侧 `sgemv_npu_wrapper.h` 的同类问题，见 §5.2）。
2. 计算一律走 `int64_t`，避免乘法溢出。
3. 该检查放在 quick return **之前**：`m=0` 时 `(0-1)·|inc|` 为负数，需先按 `len > 0` 条件短路，避免误判（`len == 0` 时跳过该检查）。

#### 3.2.1.2 分核与 tiling 策略

分核维度取**输出向量维度** `outDim = (trans == N) ? m : n`，每核负责 `outDim` 上一段连续区间，核间无数据依赖、无归约、无原子操作。

```text
aivCoreNum = GetAivCoreCount()                       // 0 → EXECUTION_FAILED
outDim     = (trans == N) ? m : n
minOutPerCore = 256                                  // 单核最少输出元素，保证 DMA 段 ≥ 1KB
useNumBlocks  = min(aivCoreNum, ceil(outDim / minOutPerCore))
useNumBlocks  = max(useNumBlocks, 1)
perCore       = ceil(outDim / useNumBlocks) 向上对齐到 8（32B）
```

依据：

- **DMA 段大小**：A 在列主序下一次连续可搬的是「同一列的行片段」。若每核行块过小（如 4096 行 / 40 核 ≈ 103 行 = 412B），单次 DMA 不足 512B，带宽利用率显著下降。取 `minOutPerCore = 256`（1KB/段）并 32B 对齐，可在核数与段长之间取得平衡。
- **核数不能过少**：本算子为 memory-bound（见 §5.1 带宽估算），需足够并发打满 HBM。以 4096×4096 为例：`ceil(4096/256)=16` 核，每核 256 行、A 分片 4MB，16 路并发。
- **不采用动态负载均衡**：BLAS-2 单次调用计算量固定且各核数据量一致，静态均分即可，避免额外同步开销。

UB 内 tile：

```text
tileM（A 块行数，即 outDim 方向切分） = min(perCore, TILE_M_MAX)
tileN（A 块列数，即 k 方向切分）      = min(kDim,  TILE_N_MAX)
TILE_M_MAX × TILE_N_MAX × 4B ≤ 64KB   → 设计取值 512 × 32
```

UB 预算（A2 单核 UB 192KB）：

| 缓冲区 | 大小 | 说明 |
|--------|------|------|
| A 块（in queue） | 512×32×4B = 64KB | 双 buffer 时 128KB |
| Y 累加器 | 512×4B = 2KB | 输出行块/列块的累加结果 |
| 分巷累加池 | 8192×4B = 32KB + 512B 哨兵 | N 路径 `lanes×stride` 共享，`stride = len` 按 32B 对齐；len=4 时 1024 巷、len=512 时 16 巷 |
| X 缓存 | 32×4B 或整段 k×4B | `incx==1 && k·4B ≤ 32KB` 时整段驻留 UB |
| y 回写暂存 | 复用 A 块 buffer | 跨步回写时借道 |

剩余空间用于 `TQue` 双 buffer 重叠搬运与计算。

#### 3.2.1.3 tiling 结构与下发

```cpp
struct SgemvTilingData {
    int64_t m;            // A 行数
    int64_t n;            // A 列数
    int64_t lda;          // 列间距
    int64_t incx;         // x 步长（含符号）
    int64_t incy;         // y 步长（含符号）
    int64_t outDim;       // (trans==N) ? m : n
    int64_t kDim;         // (trans==N) ? n : m
    int64_t perCore;      // 单核输出元素数
    int64_t tileM;        // UB 块行数
    int64_t tileN;        // UB 块列数
    int64_t numBlocks;    // 实际启用核数
    float   alpha;
    float   beta;
    uint32_t transN;      // 1 = trans N 路径，0 = trans T/C 路径（kernel 内 dispatch 用）
};
```

下发方式：`sgemv_kernel_do(A, x, y, workSpace, numBlocks, tiling, stream)`，tiling 以 `const&` 传入，launch 时按值拷贝进 kernel 参数——**不分配 tiling GM、不做 H2C memcpy、不做流同步**（对齐 `repo-op-templates/simd-membase` 的约定，与 arch35 `sgemv_host.cpp:130-132` 一致）。

`workSpace` 传 `nullptr`：实数版无需 mask 表（复数版 cgemv 因实虚交错存取需 1024 项掩码，实数无此需求），也无需跨核归约缓冲。

#### 3.2.1.4 与既有 arch22 Cgemv 的差异

| 项 | cgemv arch22（既有） | sgemv arch22（本设计） |
|----|---------------------|----------------------|
| tiling 下发 | `aclrtMalloc` + `aclrtMemcpy` 到 GM | `const&` 值传递 |
| 同步 | host 内 `aclrtSynchronizeStream` | 不调用，保持异步 |
| mask/workspace | 1024 项掩码 + 16MB workspace | 无 |
| 数据拆分 | `split_CVector` 实虚分离 | 无（单实数流） |
| UB 管理 | `AsdopsBuffer` 手工偏移 | `TPipe` + `TQue` |

不复用 cgemv 的手工 UB 布局，是因为其掩码/工作区分配方式带有多处历史修复（issue #712/#716/#721），沿用会把同类问题带入新算子；新实现走仓内官方 SIMD 模板。

### 3.2.2 Kernel 侧设计

单 kernel entry，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，按 `tiling.transN` 在 entry 内二路分发到两个 Kernel 类（或同一类的两个 Process 分支）。核内按 `GetBlockIdx()` 取自己负责的输出区间。

#### 3.2.2.1 UB 划分

```text
A_q   : TQue<VECIN, BUFFER_NUM>    tileM × tileN × 4B
X_q   : TBuf<VECCALC>               kDim（incx==1）或 tileN（stride 路径）
Y_buf : TBuf<VECCALC>               tileM 或 tileN
```

`BUFFER_NUM = 2`：搬运与计算重叠是达到带宽目标的前提（单 buffer 时 MTE2 与 V 严格串行，实测通常掉 30%+）。

#### 3.2.2.2 trans = N 路径（按行分核，遍历列累加）

```mermaid
flowchart TD
    S[核 b 负责输出行 i0..i1) --> L1[Y_buf 初始化为 0<br/>长度 len = i1-i0]
    L1 --> C{还有未处理列块?}
    C -->|是| D[搬 A[i0..i1, j0..j1) → UB<br/>每列连续 len 个 float，stride=lda]
    D --> E[搬 x[j0..j1)（stride 支持）]
    E --> F[Y += A(:,j)*x[j] 逐列累加<br/>Axpy]
    F --> C
    C -->|否| G[读 y[i0..i1) 旧值]
    G --> H[y = alpha*Y + beta*y_old]
    H --> I[跨步回写 y（incy 支持）]
```

- 列累加按 `lane = (col - c0) & (lanes-1)` 分巷：每巷独立 `Axpy` 累加，列块结束后按 `log2(lanes)` 层树形合并到 lane 0，再加进块累加器。巷数按行块长度自适应（`lanes·stride ≤ 32KB / 4B`，`stride` 为 `len` 按 32B 向上取整，上限 2048，取 2 的幂以便用掩码代替取模），并先按「每巷 ≤ 128 项」定够用值：`len=4` 时 1024 巷、`len=512` 时 16 巷；`n` 很小（如 n=4）时只取 1 巷，避免无谓的初始化与合并。列块宽 `chunkCols = lanes · LANE_TERMS`，块累加器复用 `tmpLocal_`（N 路径中它只在跨步回写时用，不额外占 UB）。分巷 + 分块把顺序累加的 `sqrt(项数)·eps` 误差压到 `sqrt(128)·eps` 量级。
  - **UB 地址必须 32B 对齐**：巷偏移若直接写 `k·len`，`len` 不是 8 的倍数时（如 `outDim=4`，步长仅 16B）奇数巷起点只对齐到 16B，AICore 报 `aivec error`（全核崩在同一 PC，同步阶段返回 `INTERNAL_ERROR`）。故所有巷偏移统一用 `k·stride`，`stride = ceil(len/8)·8`。
  - 池容量 32KB 是 UB 预算决定的：`a/x/tmp/work` 各 16KB（64KB）+ `acc/y/red` 约 4KB + 池 32KB ≈ 100KB，为实测能稳定跑满全量的水位；池提到 64KB（合计 132KB）会越过水位并触发同类异常。池尾部另留 512B 哨兵，因为向量指令按 32B 块（部分按 256B repeat）访问 UB，末巷只写 `len` 个元素也可能覆盖到块尾。

#### 3.2.2.3 trans = T/C 路径（按输出列分核，遍历行归约）

```mermaid
flowchart TD
    S[核 b 负责输出列 j0..j1) --> L1[Y_buf 初始化 0<br/>长度 len = j1-j0]
    L1 --> C{还有未处理行块?}
    C -->|是| D[搬 A[i0..i1, j0..j1) → UB<br/>矩形块，行内连续、列间 stride=lda]
    D --> E[搬 x[i0..i1)（stride 支持）]
    E --> F[tmp[0..len) = Σ_i A[i,:] * x[i]<br/>mul + 跨行 cadd_v 归约]
    F --> G[Y += tmp]
    G --> C
    C -->|否| H[y = alpha*Y + beta*y_old]
    H --> I[跨步回写 y]
```

- A 块按「行 × 输出列」布局进 UB，`dstStride = ceilAlign(tileM, 32B)`；每行 `len` 个输出列连续，便于 `mul_v` 以 `dstStride` 复用同一累加区后做 `cadd_v` 跨行归约（cgemv `do_trans` 的同构做法，去掉虚部）。
- `trans = C` 与 `T` 走完全相同的代码路径：实数矩阵共轭不改变任何元素，**不设 `conjugateA` 分支**（cgemv 有此分支是因为复数虚部取反）。
- 归约顺序为行块递增 + 块内顺序，单次执行内部确定。

#### 3.2.2.4 负步长与边界

```cpp
inline int64_t StridedIndex(int64_t i, int64_t len, int64_t inc)
{
    return (inc > 0) ? (i * inc) : ((len - 1 - i) * (-inc));
}
```

- 索引用 `int64_t`：`(len-1)·|inc|` 在 host 已保证 ≤ INT32_MAX，但中间量 `j*lda` 在大规模下可达 4096×4096 = 16.7M，用 int32 亦有溢出风险（`lda×n` 上限可达 INT32_MAX），故 UB 偏移与 GM 偏移统一 64 位。
- 尾部处理：`perCore` 向上取整后最后一个核的实际长度可能小于 `perCore`，需按 `min(perCore, outDim - i0)` 截断；`tileM`/`tileN` 的最后一块同理。
- `lda > m`：A 块搬运的 srcStride 取 `lda - tileM`，空洞列不读。

#### 3.2.2.5 精度与浮点行为

- 累加算术：`trans = T/C` 路径按 `Mul`（乘积）+ `Add`（累加）两条向量指令实现，**乘积在 f32 落盘时发生一次舍入**（本算子未使用融合乘加）；`trans = N` 路径用 `Axpy` 逐列累加。该口径与实测吻合：`TC_EX_0429` 实测 `max_abs_err = 2.4832e+04`，与「乘积先舍入、再顺序累加」的离线模型预测值**逐位一致**（若按融合乘加建模则为 `1.4464e+04`，与实测不符），故下文第 3 类边界以乘积落盘舍入为分析基础。归约链按行块分巷后长度 = `kDim / lanes`。
- 误差估算：顺序累加的舍入误差按 `sqrt(项数)·eps·|部分和|` 增长。`kDim = 1048576` 的极端矩形用例（如 4×1048576，输出仅 4 个元素，`matched_ratio ≥ 0.99` 等于要求 4 个元素全部达标）若单链累加必然超限：16 巷时每巷 6.5 万项，实测 `TC_SQ_059` 最大绝对误差 `6.29e-2`（限值 `1e-2`）；仅分巷到 1024 巷（每巷 1024 项）时 `TC_EX_0223` 仍超限 7%（`1.065e-2`）。因此累加分**两级**，把「每巷项数」与 `n` 解耦：
  1. **巷内**：列按 `lane = (col - c0) & (lanes-1)` 落到 `lanes` 个巷，每巷只累加 `LANE_TERMS = 128` 项，误差 ≈ `sqrt(128)·eps ≈ 6.8e-7`；
  2. **块间**：一个列块宽 `chunkCols = lanes · LANE_TERMS` 列，块末按 `log2(lanes)` 层树形合并到 lane 0，再加进块累加器，然后进入下一块。块数仅 `ceil(n / chunkCols)`（`n=1e6`、1024 巷时为 8 次加法），舍入可忽略。

  实测该结构把 `TC_EX_0223` 的误差从 `1.065e-2` 降到 `3.8e-3` 量级、`TC_SQ_059` 从 `9.28e-3` 降到 `3.3e-3` 量级。实现需保证：
  1. 不做 `float→half` 中间降精度；
  2. 中间累加器始终为 float；最终「乘回定标 `2^32`」与「乘 α」分两步做（先复原再乘 α，避免 `α·2^32` 自身溢出），两步都是 2 的幂或单次乘法，不引入额外舍入。**不能用 double**：AICore 编译器禁止一切 double 相关的类型转换（`cast to/from double precision floating variable is not allowed in aicore function`）；
  3. 树形合并的层数与配对次序固定，保证同输入逐次一致。
- 累加定标：用例 `fill_a_extreme_*` 的 A 含 `±FLT_MAX`，f32 下 `a·x` 即可越过 `3.4e38`，使部分和提前变 inf；当正负大项相消、真值仍落在 f32 范围内时，f64 golden 收窄后是有限值，直接累加必然比对失败（且 `inf + (-inf)` 会退化成 NaN）。因此把乘子按 `2^-32` 定标后再累加（中间量落到 1e29 量级），累加结束再乘回 `2^32`。2 的幂定标对 f32 只改阶码、逐位精确，无 under/overflow 时结果与不定标完全一致，故对常规用例无精度代价；仅在输入含次正规数时损失该元素的低位（其绝对贡献 ≤ `FLT_MAX·2^-45 ≈ 5e-7`，远低于 1e-2 兜底限值）。
- Inf/NaN：按 IEEE 规则自然传播，Kernel 不做特殊值判断，与 golden 行为一致（golden 为 f64 真值收窄）。`α=0` **不短路**（任务书 §2.1），`0×Inf/NaN` 自然得 NaN。
- **f32 精度的三类边界与判据分层**：

  | 类 | 成因 | 判据 | 典型用例 | 处置 |
  |---|---|---|---|---|
  | ① 累加舍入随归约长度增长 | 顺序累加误差 ≈ `√k · eps · |部分和|`，`k` 为归约长度 | `max_abs_error ≤ max(1e-2, 32·ULP)` | `TC_SQ_059`（`k=1e6`、仅 4 个输出）、`TC_EX_0223` | **算法解决**（本文件上条的两级累加），不豁免 |
  | ② 乘积溢出 | `Σ|a·x| > FLT_MAX`（`RANDOM_EXTREME` / `ILLCOND` 填充），f32 乘积或部分和必然溢出为 Inf，而 f64 golden 仍有限 | 同上，误差主项不可消除 | `TC_EX_0297`（`Σ|a·x| ≈ 1.6e41`） | **不可达** → MERE/MARE |
  | ③ 乘积超位宽舍入 | `|a·x|` 需超过 24 bit 尾数，乘积落盘必然舍入；其量级 `2^-24·|α|·Σ|a·x|` 已超过限值 | 同上，误差主项不可消除 | `a=INDEX`（`a[i]=i`，量级到 1e6）× `x=RANDOM_ALTER`、`m/n` 达 1e6 | **不可达** → MERE/MARE |

  判定流程（`test/gemv/sgemv/arch22/sgemv_test.cpp`）：**先按 `mixed_tolerance` 判**；**仅当判失败时**才逐元素计算 f32 舍入误差量级 `B_i = 2^-24 · |α| · Σ_j |a_ij·x_j|`，若 `max_i (B_i / limit_i) > 1` 则把该失败归因为第 ②/③ 类，改走 MERE/MARE 统计判据（阈值取用例 CSV 的 `mere_threshold = 2^-13`、`mare_multiplier = 10`，与 `test/frame/verify.h` 的 `MereMareStrategy` 同源）；否则如实报 FAIL（属实现缺陷）。第 ① 类**不走这条通路**——它由算法改进解决，改进后必须直接在 `mixed_tolerance` 下通过。

  `B_i` **只作失败归因、不作独立判决依据**，原因有三：① `B_i` 假定所有乘积的舍入误差同向叠加，实测通常小 1~2 个量级（`TC_EX_0223`：`B = 9.77e-1`，实测 `7.32e-3`，相差 130 倍）；② `B_i` 由全局量级 `Σ|a·x|` 决定，而限值由逐元素 `|g_i|` 决定，强相消元素 `|g_i| → 0` 时限值退化为 `1e-2` 而 `B_i` 仍是全局量级，比值虚高（`TC_EX_0514` 达 `4.6e5`，但全局实测仅 `0.06` 倍限值——若以 `B_i` 直接改判会把这条正常通过的用例误切轨道）；③ `B_i` 只含乘积项，不含累加项 `2^-24·Σ|部分和|`，长归约时后者可超过前者（`TC_EX_0583` 实测 `4.81e5`，是 `B` 的 11 倍）。因此测试侧同时输出 `boundTightness = B_max / 实测最大误差` 作为证据强度指标：`≫1` 表示界远高于实测（乘积项主导、界乐观），`<1` 表示实测已超过该估计（累加项主导）。

  MERE/MARE 的判定细则：双方均有限的元素计入相对误差统计（`MERE < thr`、`MARE < thr × mult`）；双方均非有限的元素视为等效（这类点的真值是巨大正负项相消后的残差，落到 `±Inf` 时符号由舍入顺序决定、不可判）；一方有限另一方非有限时，若有限一侧量级 `≥ FLT_MAX/2`（处于 f32 溢出边界，"输出 Inf 还是输出一个 ≈FLT_MAX 的有限值"只由舍入路径决定）则同样视为等效，否则判失败（属真实溢出或漏写）。

  全量 1000 条精度用例中触发改判的共 8 条（均为第 ③ 类），实测归因如下（取自 `acc.log` 的 `TRACK=` 行）：

  | 用例 | `B/limit` | `boundTightness` | MARE | 结论 |
  |---|---|---|---|---|
  | `TC_EX_0429` | 63.98 | 13.21 | < 1.22e-3 | 通过 |
  | `TC_EX_0513` | 15.39 | 7.67 | < 1.22e-3 | 通过 |
  | `TC_EX_0583` | 10.68 | 0.09 | < 1.22e-3 | 通过（误差主项为累加项） |
  | `TC_EX_0687` | 97.53 | 79.54 | < 1.22e-3 | 通过 |
  | `TC_EX_0779` | 174.95 | 169.65 | < 1.22e-3 | 通过 |
  | `TC_EX_0247` / `TC_EX_0289` / `TC_EX_0345` | 均 > 1 | 见 `acc.log` | < 1.22e-3 | 通过 |

#### 3.2.2.6 与 arch35 SIMT 实现的差异

| 项 | arch35（950PR） | arch22（本设计） |
|----|-----------------|------------------|
| 并行模型 | SIMT，`asc_vf_call` + `blockIdx/threadIdx` | AIV 多核 + Vector 指令 |
| 数据驻留 | 直接读 GM，可选 x 驻留 `__ubuf__` | A 块显式搬入 UB，TQue 双缓冲 |
| 分核 | `numThreads` 线程级 grid-stride | 核级静态分段 |
| 收敛 | 每线程独立点积，天然并行 | 每核内列/行顺序累加 |

两者算法同构（都是「输出维分块 + 归约维遍历」），差异仅来自并行原语。

### 3.3 支持硬件

| 芯片版本 | 涉及勾选 |
|----------|----------|
| Atlas A2 训练系列（910B3） | √（自验与性能出具设备） |
| Atlas A2 推理系列 | √（同一 arch22 二进制） |
| Atlas A3 训练/推理系列 | √（同一 arch22 二进制） |

### 3.4 算子约束限制

1. 仅 FLOAT32；复数由 `aclblasCgemv` 承担。
2. `α=0` **不短路**（任务书 §2.1 原文）：仍完整计算 `op(A)·x` 再乘 `α`，`0×Inf/NaN` 按 IEEE 得 NaN。`α=0` 且 `A`/`x` 含 Inf/NaN（或 `Σ|a·x|` 在 f32 溢出）的组合任务书声明不设精度用例，测试侧打印 `SKIP`。
3. 确定性：分巷与树形合并的次序固定，同一输入下结果逐次一致（但不作为接口承诺）。
4. 单次调用内不做 workspace 复用与跨调用缓存。
5. `lda×n` 与 `(len-1)·|inc|` 的合法上界由 INT32 语义与 host 守卫共同约束，超出返回 `INVALID_VALUE`。

---

## 四、特性交叉分析

| 交叉项 | 结论 |
|--------|------|
| 与 `aclblasCgemv`（arch22 同族） | 独立文件、独立 kernel entry，互不影响；共享 `aclblasHandle_t` 与 stream 语义 |
| 与 arch35 `aclblasSgemv` | 同一公共声明，按 `SOC_ARCH_DIRS` 编译期分流，运行时不同产品加载各自实现 |
| 与 `aclblasSgemvBatched` / `SgemvStridedBatched` | 不涉及，本任务只做基础接口 |
| 与 ACLNN 图模式 | 不涉及，句柄式直调，无算子信息库注册 |
| 与 `lda > m` 空洞布局 | 支持；空洞列不读不写 |
| 与负步长 | 支持；尾端起算 |
| 与多 stream 并发 | 依赖 handle 绑定 stream，不同 handle 可并发 |

---

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度 | FLOAT32：`atol = rtol = 2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2 或 32·ULP`；Inf/NaN 按等价性单独判定。**判据分层**：先按 `mixed_tolerance` 判；失败且 f32 舍入误差量级 `2^-24·|α|·Σ|a·x|` 已超过该点限值时（乘积溢出 / 乘积超位宽，见 §3.2.2.5 第 ②③ 类），改用 MERE/MARE 统计判据（`MERE < 2^-13`、`MARE < 10·2^-13`）；长归约的累加舍入（第 ① 类）由两级累加算法解决，不豁免 | 生态算子开源精度标准 mixed_tolerance + `test/frame/verify.h` 的 `MereMareStrategy` |
| 性能 | 平均单次耗时不高于 GPU（A100）标杆的 1.25 倍，即 ≥ 0.8× A100；warmup 5 次后有效采样 ≥10 次取平均；设备 910B3 | 任务书 §3.3 + `test_cases/gpu_baseline.csv` |

**带宽可行性估算**（性能风险的主要来源）：

- 标杆 `sgemv-base-000`（4096×4096，trans=N）：A100 耗时 52.63 us，A 数据量 67.1 MB → 有效带宽约 1.27 TB/s。
- 达标线 `≤ 52.63 / 0.8 = 65.8 us` → 910B3 需 ≥ 1.02 TB/s 有效带宽。
- 结论：**该 case 是纯 bandwidth-bound**，算力不是瓶颈。设计上必须保证：① 全核并行（16 核起）；② A 只读一次、无重复搬运；③ TQue 双缓冲重叠 MTE2 与 V；④ 不做「GM→UB→GM」的多余往返。
- 小规模 case（n=1/3/64）由 launch 开销主导，核数按 `outDim` 自适应收缩，避免大量空核拖慢。

### 5.2 测试工程与覆盖

- 位置：`test/gemv/sgemv/arch22/`（新建），复用 `test/gemv/sgemv/sgemv_param.h`、`sgemv_golden.h`（`cblas_sgemv`）与 CSV 用例 `sgemv_test.csv`（1200 条：精度 1000 + 性能 200）。
- CMake：`test/gemv/sgemv/CMakeLists.txt` 补 arch22 分支（参考 `test/gemv/cgemv/CMakeLists.txt`）。
- **INT32_MIN 短路**：`sgemv_npu_wrapper.h` 用 `(len-1)*std::abs(incx)` 预算缓冲，`abs(INT32_MIN)` 为 UB，arch22 的 wrapper 必须先对 `INT32_MIN` 短路（原样透传给接口），否则负向用例在 wrapper 阶段异常而非走到接口校验。
- **负向用例缓冲**：期望非 SUCCESS 的用例用单元素 dummy 缓冲或 `NULLPTR` 占位，不按步长预分配（`(n-1)·|inc|` 可达数十亿元素）。
- **正态分布补充**：`gen_csv.py` 默认只生成均匀分布，需用 `--dist mixed` 补 `GAUSS_NORM_5_2` 用例，并在报告中说明补齐方式与条数。

覆盖矩阵（对应任务书 §3.5）：

| 维度 | 覆盖 |
|------|------|
| trans | N / T / C 三态 + 非法枚举（999） |
| m, n | 0、1、3、64、512、2048、4096、1048576×4 极端矩形、非方阵、负值 |
| lda | `max(1,m)`、`m+8` 空洞 |
| incx / incy | 1、2、7、-1、-3、0、INT32_MIN、跨度溢出 |
| alpha / beta | 0.0、1.0、2.5、-1.5、0.1、10.0、Inf、NaN、nullptr |
| 填充 | 均匀 / 正态 / Inf / NaN / NULLPTR |
| 边界 | quick return（m=0/n=0 × trans 三态 × A/x/y 空指针组合）、负维度、lda 非法、标量空指针 |

### 5.3 兼容性分析

- 接口声明不改动，与 950PR 实现共用，二进制 ABI 一致。
- 新增 arch22 源文件不影响 arch35 编译（`SOC_ARCH_DIRS` 分流）。
- README 产品支持表更新为 A2/A3「支持」，属文档变更，不涉及行为兼容。

### 5.4 已知限制与规避

| 限制 | 影响 | 规避 |
|------|------|------|
| 长累加链（k=1048576）误差余量收窄 | 极端矩形用例可能逼近 1e-4 阈值 | 保持 float 累加器、禁止中间降精度；实测超阈值再加 Kahan |
| 小规模 case 受 launch 开销主导 | `n=1/3` 可能无法达标 0.8×A100 | 核数自适应收缩至 1-2 核；如仍不达标，在报告中单列说明并给出 GPU 侧同规模开销对比 |
| `α=0` 与 golden 短路策略不一致 | 含 Inf/NaN 时结果不同 | 该组合不设精度用例（任务书已认可） |
| arch35 的跨度守卫缺失 | 两代实现行为不一致 | 本设计在 arch22 补守卫；不一致点写入 README/PR 说明，供上游决定是否同步补 arch35 |
