# 【CANN社区任务】aclblasCgemm3m 算子设计文档（Atlas A2/A3，arch22）

| 项 | 内容 |
|---|---|
| 任务 | 11月社区任务-aclblasCgemm3m算子开发（A2/A3） |
| 团队 | EulerTeng |
| 算子 | `aclblasCgemm3m` |
| 实现目录 | `blas/gemm/arch22/` |
| 测试目录 | `test/gemm/cgemm3m/arch22/` |
| 对标接口 | cuBLAS `cublasCgemm3m` |
| 基线 commit | `master` @ `55ef791`（2026-11-25） |

> **与本团队 10 月 950 版本设计文档的关系**：本团队已在
> `tasklist/10-aclblasCgemm3m-950/Euler_teng/docs/design.md` 提交过同一算子的
> arch35（Ascend 950PR）设计文档并合入。两者**算子语义与 3m 数学分解完全一致**，
> 本文沿用该文的 §1.2.1 数学定义与 §3.1.5 共轭转置处理结论，不再重复推导；
> **差异全部集中在 arch22 的工程落地**（仓内基础设施不同、无共享 launcher、
> Phase 0 产出的缓冲布局不同、性能判据设备不同），这些在本文 §1.2.2、§2.4、
> §3.2、§3.3、§4.1.2 重写。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 社区任务「11月社区任务-aclblasCgemm3m算子开发（A2/A3)」。仓内
`blas/gemm/arch35/cgemm3m_host.cpp` 已有 arch35 实现、`test/gemm/cgemm3m/arch35/`
已有对应测试，A2/A3（arch22）侧缺失。本任务补齐 arch22 实现与测试，使
`aclblasCgemm3m` 在 Atlas A2/A3 系列产品上可用。

## 1.2 背景介绍

### 1.2.1 3m 算法的数学定义

复数乘法 `(a+bi)(c+di)` 的常规实现需 4 次实数乘（`ac`、`bd`、`ad`、`bc`）。
3m 算法（高斯复数乘法技巧）用 3 次实数乘完成。本任务书 §2.1 给出两条代数等价的分解：

**非融合路径**

```
P0 = op(Ar)·op(Br)
P1 = op(Ai)·op(Bi)
P2 = op(Ar+Ai)·op(Br+Bi)
Re = P0 − P1
Im = P2 − P0 − P1
```

**融合路径**

```
P0 = op(Ar+Ai)·op(Br)
P1 = op(Ar)·op(Bi−Br)
P2 = op(Ai)·op(Br+Bi)
Re = P0 − P2
Im = P0 + P1
```

验证融合路径（标量形式，`a+bi` 乘 `c+di`）：

```
P0 = (a+b)c,  P1 = a(d−c),  P2 = b(c+d)
Re = P0 − P2 = (a+b)c − b(c+d) = ac − bd    ✓
Im = P0 + P1 = (a+b)c + a(d−c) = ad + bc    ✓
```

两条路径**代数等价但 FP32 舍入过程不同**，任务书 §2.1 明确「不能把一种路径的误差
分析套用到另一种」。本设计的选路决策见 §3.2.1.5。

相对 4 乘法路径，3m 以 25% 的乘法量换取更差的数值条件数——重组阶段的
`P0 − P2` 与 `P0 + P1` 是**相近量相减/相加**，当 `|Re|` 远小于 `|P0|`、`|P2|` 时会发生
抵消性误差放大。这是 cuBLAS 把 `cublasCgemm3m` 与 `cublasCgemm` 并列提供、
而非替换后者的原因，也是本设计 §4.1.4 精度风险分析的起点。

### 1.2.2 仓库现状分析（本设计的起点）

本节为 arch22 重写，与 950 版本文档的同名小节结论不同。

**（1）接口声明已存在，本任务不新增**

`include/cann_ops_blas.h` 已声明 `aclblasCgemm3m`（任务书 §7.3 指为 L584），
参数序列与 `aclblasCgemm` 一致。任务书 §7.3 要求「禁止定义私有平行 API」，
本设计据此**不新增任何声明**，仅在 arch22 提供定义。

**（2）arch22 已有的复数 GEMM 基础设施（本设计的主要复用对象）**

`blas/gemm/arch22/` 现有文件：

| 文件 | 规模 | 作用 |
|---|---|---|
| `cgemm_host.cpp` | 24 KB | `aclblasCgemm` 的 host 实现，含校验、workspace 规划、三段式编排 |
| `cgemm_kernel.cpp` | 15 KB | Phase 0（去交错拆分）与 Phase 2（复数重组）的 AIV kernel |
| `cgemm_kernel.h` | 2.0 KB | kernel_do 声明 |
| `cgemm_tiling_data.h` | 2.3 KB | tiling 结构体 |
| `gemm_host.cpp` / `gemm_kernel.cpp` / `gemm_kernel.h` | 24+62+2.8 KB | 实数 GEMM |

`cgemm_kernel.cpp` 文件头注释（:17-22）说明了既有的三段式架构：
Phase 0（AIV）把交错复数拆成两个独立的实数列主序平面，
Phase 1 由 host 驱动、通过 arch22 的实数 GEMM 完成四次实数乘，
Phase 2（AIV）做复数重组。

关键在于 Phase 1 的四次实数乘是一张**声明式的表**
（`cgemm_host.cpp` 内 `RunCgemmRealProducts()`）：

```cpp
const CgemmRealProduct products[4] = {
    {1.0F, ws.ar, ws.br, true, ws.re, ws.prodRe},        // re = Ar*Br
    {signs.aiBi, ws.ai, ws.bi, false, ws.re, ws.prodRe}, // re += -(s_a*s_b) * Ai*Bi
    {signs.arBi, ws.ar, ws.bi, true, ws.im, ws.prodIm},  // im = s_b * Ar*Bi
    {signs.aiBr, ws.ai, ws.br, false, ws.im, ws.prodIm}, // im += s_a * Ai*Br
};
```

每项描述「系数 × 左平面 × 右平面 → 目标平面」，由 `RunOneCgemmProduct()` 统一通过
`aclblasSgemmEx` 下发到实数 GEMM，并由 `AccumulateRealPlane()` 做带系数累加。
**这意味着从 4 乘法改为 3 乘法，核心是替换这张表与重组系数，而不是重写整条流水线。**

**（3）arch22 与 arch35 的结构性差异（决定本设计的文件划分）**

arch35 把复数 GEMM 的 launcher 抽成了共享函数：

```
blas/gemm/arch35/cgemm_host.cpp:701   aclblasStatus_t LaunchComplexGemm(..., bool forceThreeProducts)
blas/gemm/arch35/cgemm_host.cpp:737   aclblasCgemm   -> LaunchComplexGemm(..., false)
blas/gemm/arch35/cgemm3m_host.cpp:22  aclblasCgemm3m -> LaunchComplexGemm(..., true)
blas/gemm/arch35/gemm_host_common.h:100  声明
```

因此 arch35 的 `cgemm3m_host.cpp` 只有 23 行。
**arch22 没有 `gemm_host_common.h`，也没有 `LaunchComplexGemm`**——`cgemm_host.cpp` 内的
`LaunchCgemmKernel()` 是文件内静态函数，只服务 `aclblasCgemm`。本设计据此需要先做一次
最小抽取（§2.4），使 arch22 的文件布局与 arch35 对齐。

**（4）arch35 的实现不能整体照搬**

arch22 不提供 arch35 的向量函数调用编程模型。实测核对：

```bash
grep -rn "asc_vf_call" blas/*/arch22/ | wc -l    # 0
```

（注：若同时 grep `Interleave` 会有 185 处命中，但那些是 `SplitInterleaved` 一类
做复数实虚部拆分的普通标识符，与编程模型无关，不应作为判据。）

因此 Phase 0/Phase 2 必须走 arch22 自己的 AIV kernel 路径，即复用
`cgemm_kernel.cpp` 的既有写法，而非移植 `cgemm3m_rounding_kernel.cpp`。

**（5）同批次任务的既有实现不冲突**

`blas/gemm/arch22/cgemm_host.cpp` 由上游于 2026-11-24 合入（`5ebfa3c`），
提供 `aclblasCgemm`；本任务新增的是 `aclblasCgemm3m`，两者并列，
与 arch35 的并列关系一致，不存在重复实现。

---

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemm/arch22/` 实现 `aclblasCgemm3m`，计算

```
C = alpha * op(A) * op(B) + beta * C
```

- `alpha`/`beta` 为 complex64 标量（Host 内存），`A`/`B`/`C` 为 complex64 矩阵（Device 内存，列主序）；
- `transa`/`transb ∈ {OP_N, OP_T, OP_C}`，复数语义下 `OP_C` 为共轭转置、`OP_T` 为非共轭转置；
- `op(A)` 为 m×k、`op(B)` 为 k×n、`C` 为 m×n；
- 复数乘法按 3m 分解为 3 次实数矩阵乘。

## 2.2 需求拆解

| 编号 | 子项 | 说明 |
|---|---|---|
| R1 | 入参校验 | 按任务书 §2.4 逐参数返回状态码，校验顺序见 §3.2.1.1 |
| R2 | Quick return | `m=0`/`n=0` no-op；`k=0`/`alpha=0` 只做 `C = beta*C` |
| R3 | Phase 0 拆分 | 去交错，并按 3m 融合路径产出组合平面（含共轭吸收） |
| R4 | Phase 1 三次实数 GEMM | 替换既有 4 项表为 3 项表 |
| R5 | Phase 2 重组 | `Re = P0 − P2`、`Im = P0 + P1`，再施加 alpha/beta |
| R6 | 选路 | 按形状与片上容量在融合/非融合间选择（§3.2.1.5） |
| R7 | 测试工程 | `test/gemm/cgemm3m/arch22/`，1200 条 CSV（精度 1000 + 性能 200） |
| R8 | 性能打印 | 每条 TC_PF 打印 `[PERF] case=... avg_us=...` |
| R9 | 双口径性能数据 | `msprof op` kernel Task Duration（正式）+ `aclrtEvent`（交叉验证） |

## 2.3 外部组件依赖

| 依赖 | 用途 | 说明 |
|---|---|---|
| `aclblasSgemmEx` | Phase 1 的实数 GEMM | 既有 `cgemm` 已用此路径，本设计沿用 |
| `gemm_kernel_do` | 实数 GEMM kernel | 见 §2.4 的 CMake 登记要求 |
| cblas（Netlib BLAS 3.10.0） | 精度 golden | 随测试工程提供，`cblas_cgemm` |
| `msprof op` | 性能正式判据采集 | 任务书 §3.3 |

## 2.4 内部适配模块（新增/修改文件清单）

**新增**

| 文件 | 内容 |
|---|---|
| `blas/gemm/arch22/cgemm3m_host.cpp` | `aclblasCgemm3m` 定义，薄壳，转调共享 launcher |
| `test/gemm/cgemm3m/arch22/cgemm3m_test.cpp` | arch22 的 GTest 用例装配 |
| `test/gemm/cgemm3m/arch22/cgemm3m_test.csv` | 1200 条，来自任务配套 `test_cases/` |
| `test/gemm/cgemm3m/arch22/gpu_baseline.csv` | 200 条 `gpu_ms`，来自任务配套 |
| `test/gemm/cgemm3m/arch22/readme.md` | 测试步骤（任务书 §4 交付件 2 要求） |

> **测试工程的公共层已存在，本任务不重复创建**：`test/gemm/cgemm3m/` 下的
> `cgemm3m_golden.h`、`cgemm3m_verify.h`、`cgemm3m_param.h`、`cgemm3m_profile.h`、
> `run_self_test.py`、`run_memcheck.py`、`test_gate.py` 均为跨 arch 共用，
> arch35 版本已在用。本任务只新增 `arch22/` 子目录下的 arch 专属文件
> （对照：`arch35/cgemm3m_test.cpp` 为 308 行），避免触及任务书 §5 的
> 「多出一些无用的代码、测试和编译文件」反例。

**修改**

| 文件 | 改动 | 理由 |
|---|---|---|
| `blas/gemm/arch22/cgemm_host.cpp` | 把文件内静态的 `LaunchCgemmKernel()` 抽成带 `forceThreeProducts` 形参的非静态 `LaunchComplexGemm()`；Phase 0/1/2 按该形参分支；校验与 quick return 一并移入（见 §3.2.1.0） | 与 arch35 的分工对齐；避免复制一份近乎相同的 host 流程 |
| `blas/gemm/arch22/cgemm_kernel.cpp` | 新增两个 AIV kernel：`cgemm_split_sum_kernel`（Phase 0 去交织的同一趟里写出和平面）与 `cgemm3m_fold_kernel`（Phase 1b，一趟把三个积折成 `re`/`im`，并在最后一个 K 分块内联精度筛查）。Phase 2 的 `cgemm_combine_kernel` 原样复用、未改动 | arch22 无法复用 arch35 的 kernel；两个新 kernel 的实测依据见 §3.2.1.4a |
| `blas/gemm/arch22/cgemm_kernel.h` | 新 kernel_do 声明 | — |
| `blas/gemm/arch22/cgemm_tiling_data.h` | `Cgemm3mFoldTilingData`（含内联筛查的参数） | — |
| `blas/gemm/arch22/cgemm_host_common.h`（新建） | 声明抽取后的 `LaunchComplexGemm()` | arch22 无 `gemm_host_common.h`，跨 TU 调用需要一处声明（见 §3.2.1.0） |
| `test/gemm/cgemm3m/CMakeLists.txt` | 当前仅在 `arch35 IN_LIST SOC_ARCH_DIRS` 时注册测试目标，否则打印 `Skipping cgemm3m tests: arch35 is required`；需放开 arch22 | **不改则 arch22 测试二进制根本不会生成，而且是静默跳过**；任务书 §2.2 亦点名要求注册 arch22 目标 |
| `blas/CMakeLists.txt` | 单算子构建（`--ops=cgemm3m`）时保留 `blas/gemm_ex/arch22/`（`aclblasSgemmEx` 后端）、`blas/scal/arch22/`、以及 `blas/gemm/arch22/cgemm_host.cpp`+`cgemm_kernel.cpp` | 现有保留规则按 `^${_test_name}_` 前缀匹配，`cgemm3m` 匹配不到 `cgemm_host.cpp`；`gemm_ex`/`scal` 的保留条件只对 `"cgemm" IN_LIST TEST_NAMES` 生效 |
| `blas/gemm/README.md` | 补 `aclblasCgemm3m` 的 arch22 产品支持 | 任务书 §4 交付件 4 要求 |

> **本表已按 arch22 实测修订**。初稿曾沿用本团队 950（arch35）文档的 CMake 结论并
> 标注「arch22 同构」，复核后发现三条都不适用：
> ① `cmake/test.cmake::_ops_blas_has_blas_op_sources` 是 glob 驱动
> （`blas/*/${arch}/${op_name}_*.cpp`），能自动命中 `cgemm3m_host.cpp`，**无需改动**；
> ② `blas/CMakeLists.txt` 里并不存在「算子→源文件」映射表，是文件名前缀规则；
> ③ `_GEMM_KERNEL_DEP_OPS`（`"cherk;cher2k;cherkx"`）只约束 arch35 的
> `gemm_kernel.cpp`，与 arch22 无关，arch22 的 `cgemm` 本身也不在该表内。
> **跨 arch 照搬 CMake 结论是不安全的**，本文其余章节不再做此类外推。

> **关于「修改既有文件」的说明**：任务书 §5 举了反例「文件个数明显不符合正常数量」。
> 本设计选择**抽取共享 launcher 而非新建一份平行 host 流程**，正是为了让 arch22 的
> 文件数与 arch35 一致（arch35 为 `cgemm_host.cpp` + 23 行的 `cgemm3m_host.cpp`）。
> 若改为完全独立实现，arch22 会多出一份约 500 行的重复 host 代码，反而触及该反例。

## 2.5 与仓内既有实现的关系

| 关系 | 说明 |
|---|---|
| `aclblasCgemm`（arch22） | 并列算子，共用抽取后的 `LaunchComplexGemm`，`forceThreeProducts=false` 时行为与现状等价 |
| `aclblasCgemm3m`（arch35） | 同名算子的另一 arch，数学分解一致，但校验口径有两处 per-API 差异（§3.2.1.1）、默认路径选择亦不同（§3.2.1.5） |

**关于「纯重构」说法的订正**：抽取会把入参校验从公开入口 `aclblasCgemm`
（`cgemm_host.cpp:522-531`）挪进 launcher 内部（对齐 arch35 的
`LaunchComplexGemm` 做法），**这改变了 `aclblasCgemm` 的控制流，不是严格意义上的
纯重构**。因此回归验证不能只靠"既有测试通过"：

> `test/gemm/cgemm/arch22/cgemm_test.csv` 只有 79 条用例，空指针部分仅覆盖
> `alpha_null`/`beta_null`/`a_null`/`b_null`/`c_null`，**没有 `alpha=0` 叠加
> `A=nullptr` 的组合**。所以它既不足以证明重构没破坏行为，也发现不了 §3.2.1.1
> 「差异一」那类问题。

本设计据此要求：**抽取前先跑一遍 `cgemm` 既有测试留存基线输出，抽取后逐条比对
返回码**，并在自测报告中附上两次输出，而不是只写一句"既有测试通过"。

---

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```
C = alpha * op(A) * op(B) + beta * C
```

其中 `op(X) = X`（`OP_N`）、`X^T`（`OP_T`）、`X^H`（`OP_C`）。

### 3.1.2 支持数据类型

| 参数 | 类型 |
|---|---|
| `A`/`B`/`C` | `aclblasComplex`（COMPLEX64，实部/虚部各 FLOAT32） |
| `alpha`/`beta` | `aclblasComplex`（Host 内存） |
| `m`/`n`/`k`/`lda`/`ldb`/`ldc` | `int` |

复数类型以 `include/cann_ops_blas_common.h` 的定义为准。

### 3.1.3 支持形状

- `m, n, k ≥ 0`；`m=0` 或 `n=0` 为合法 no-op；
- `transa=N` 时 `lda ≥ max(1,m)`，否则 `lda ≥ max(1,k)`；
- `transb=N` 时 `ldb ≥ max(1,k)`，否则 `ldb ≥ max(1,n)`；
- `ldc ≥ max(1,m)`；
- 测试覆盖至 4096³（任务书 §3.5），自验包内存预算为单用例 host 侧 ≤ 512 MB。

### 3.1.4 接口定义

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
```

声明已存在于 `include/cann_ops_blas.h`，本任务不新增。

### 3.1.5 3m 分解与共轭转置的交互

`op` 取 N/T/C 三种。实数路径下 T 与 C 等价，**复数路径下 C 为共轭转置，与 T 有本质区别**：

```
op = N:  Ar, Ai 原样
op = T:  Ar^T, Ai^T
op = C:  Ar^T, −Ai^T      ← 虚部取负
```

**3m 恒等式对 `(Ar, Ai)` 的符号无任何假设**，共轭仅相当于把 `Ai` 替换为 `−Ai`，
恒等式依然成立。但**共轭在 arch22 落在哪一层，与 arch35 不同，这点必须说清楚**：

| | arch35 | **arch22（本任务）** |
|---|---|---|
| 共轭落点 | **Phase 0 拆分阶段**，`DeinterleaveElement()` 的 `conjugate` 形参写出虚部时取负 | **积表的符号系数**。拆分 kernel 写的是**原始虚部** |
| 依据 | `cgemm_deinterleave_common.h` | `cgemm_split_kernel_do()` 无 conjugate 形参；`ResolveCgemmSigns()` 返回 `{-sa*sb, sb, sa}`，`sa = (transa==OP_C) ? -1 : 1` |

记 `sa`、`sb` 为两侧的共轭符号，arch22 四乘法路径实际计算的是：

```
Re = Ar·Br − sa·sb·(Ai·Bi)
Im = sb·(Ar·Bi) + sa·(Ai·Br)
```

即等价于用 `Ai' = sa·Ai`、`Bi' = sb·Bi`。因此 **3m 路径必须自己施加这两个符号**：

| 位置 | 做法 |
|---|---|
| 和平面 | `Ar + sa·Ai`、`Br + sb·Bi`（累加虚部平面时用 `sa`/`sb` 作系数，而非 1） |
| `P1 = Ai·Bi`（原始平面）的系数 | `−sa·sb`，恰好等于既有的 `signs.aiBi`，`Re` 与 `Im` 两处同值 |

> ⚠️ **本节是实现阶段实测订正的**。初稿照 arch35 写成「共轭吸收在 Phase 0 拆分」，
> 据此实现后真机实测 `transa=C` / `transb=C` 的用例成批失败
> （典型 `TC_L0_005`（`basic_NC_n4`）：实测 `(26.34, −19.40)`，golden `(−54.16, 32.29)`），
> 根因就是 3m 路径沿用了原始虚部平面却未施加 `sa`/`sb`，等价于把 `OP_C` 当成了 `OP_T`。
> **跨 arch 照搬结论不安全**——这是本文第二次因此修订（另一处见 §2.4 的 CMake 结论）。

转置本身仍交给实数 GEMM 的 `transMode` 表达：`ta = (transa == ACLBLAS_OP_C) ? ACLBLAS_OP_T : transa`
的归一化，3m 路径沿用不变。

#### 3.1.5.1 Phase 0 产出的平面（arch22 版本）

arch22 既有 `CgemmWorkspace` 为 4 乘法路径准备了 4 个输入平面
（`ar`、`ai`、`br`、`bi`）。本设计取非融合分解（§3.2.1.5），所需输入平面为：

| GEMM | 左操作数 | 右操作数 |
|---|---|---|
| `P0` | `Ar` | `Br` |
| `P1` | `Ai` | `Bi` |
| `P2` | `Ar + Ai` | `Br + Bi` |

**既有的 4 块平面（`ar`/`ai`/`br`/`bi`）全部直接复用**，只新增 2 块组合平面
`Ar+Ai`、`Br+Bi`。这是选非融合的一个附带好处：融合路径要的是
`Ar+Ai`、`Bi−Br`、`Br+Bi` 三块新平面，且 `bi` 不再单独需要，改动面更大。

**workspace 代价（按实际分配口径，不按"平面块数"粗算）**

必须说明三件事，否则这笔账会算错：

1. `CgemmWorkspace` 除了常驻平面还有 **`scratch`**，`CgemmTotalFloats()`
   （`cgemm_host.cpp:220-240`）的实际分配是
   `2*planeA + 2*planeB + 4*planeR + scratch`，其中
   `scratch = 2*(planeA + planeB + 4*planeR) + MARGIN`——**scratch 比全部常驻平面还大**，
   它是转交给实数 GEMM 子句柄的工作区。
2. 三种平面尺寸**不相等**（`planeA = ldAr×k`、`planeB`、`planeR = ldRe×n`），
   不能用"一块平面"当统一单位。
3. `prod` 中间缓冲不是冗余：`RunOneCgemmProduct()`（:349）的
   `direct = overwrite && (alpha == 1.0F)`、`overwrite = (kBase == 0 && firstWriter)`；
   一旦 `k > CGEMM_K_CHUNK` 触发 k 分块，`kBase>0` 的迭代 `direct=false`，
   必须经 `prod` 再累加。

   > **订正（实现阶段实测）**：原文「**4096³ 必然走这条路径**」不成立。
   > `CGEMM_K_CHUNK` 已改为条件生效 —— 实数入口的 whole-grid 后端要求
   > m/n/k 三者都落在 [2048, 8192]（`CanUseWholeGridSquare`），而把 k 压到 1024
   > 恰好让它掉出下界、把最快的通路关掉。4096³ 现在**单块覆盖整个 K**，
   > 实测提速 ×2.17（35565 µs → 16360 µs）。详见 §3.2.1.4a。

据此，本设计相对 4 乘法路径的真实增量是 **`+2×planeA/planeB` 量级的组合平面**
（A 侧 1 块、B 侧 1 块），中间积沿用既有 `re`/`im`/`prod` 不新增（见 §3.2.1.4）。

> **这笔账必须在实现阶段用 4096³ 实测复核**。随包 `test_cases/README.md` 已预警：
> 「Cgemm3m 若 3M workspace 导致超限，可在自测报告中说明后过滤该 case，
> **不得静默删除**」——说明官方预期到了这个风险。本设计把它列为 §4.1.2 的
> 显式风险项而非小问题。

## 3.2 算子实现

### 实现方案

沿用 arch22 既有三段式：Phase 0（AIV 去交错+组合）→ Phase 1（AIC 三次实数 GEMM）
→ Phase 2（AIV 复数重组）。

#### 3.2.1 host 侧设计

##### 3.2.1.0 抽取边界：这一刀切在哪里

这是本设计最需要讲清楚的工程决定。arch22 的现状是：

| 符号 | 位置 | 链接可见性 |
|---|---|---|
| `aclblasCgemm`（公开入口） | `cgemm_host.cpp:522` | `extern "C"` |
| 入参校验三件套 | `ValidateCgemmParams` :143 / `ValidateCgemmLeadingDims` :123 / `ValidateCgemmPointers` :178 | **匿名 namespace 内 `static`** |
| `LaunchCgemmKernel` | :456（namespace 闭合于 :512） | **匿名 namespace 内 `static`** |

校验目前由公开入口 `aclblasCgemm` 调用（:522-531），**不在** `LaunchCgemmKernel` 内。

> **若 `cgemm3m_host.cpp` 只是「薄壳转调 launcher」而不动校验，它将跳过全部入参校验
> 与 quick return**，CSV 中约 38 条 `TC_ED_*` 负向用例（空指针、非法枚举、负维度、
> 非法前导维）会全部 FAIL。这是初稿未交代清楚的地方。

**本设计的切法**：把校验与 quick return 一并移入 `LaunchComplexGemm()`，
与 arch35 的分工一致（arch35 的校验正是在 `LaunchComplexGemm:711-723` 内）。于是：

```
aclblasCgemm   (cgemm_host.cpp)    -> LaunchComplexGemm(..., /*forceThreeProducts=*/false)
aclblasCgemm3m (cgemm3m_host.cpp)  -> LaunchComplexGemm(..., /*forceThreeProducts=*/true)
                                      ^ 薄壳，约 20 行，对齐 arch35 的 cgemm3m_host.cpp
```

`LaunchComplexGemm()` 需要跨 TU 可见，因此**新建
`blas/gemm/arch22/cgemm_host_common.h` 放声明**（arch22 无现成的
`gemm_host_common.h`，§1.2.2(3) 已核实）。三个 `Validate*` 函数保持
匿名 namespace 内的 `static` 不变——它们只被同一 TU 内的 launcher 调用。

> **形参命名**：沿用仓内既有命名 `forceThreeProducts`（arch35
> `cgemm_host.cpp:704`、`gemm_host_common.h:103`），不自造 `use3m`。

##### 3.2.1.1 校验顺序

严格按任务书 §2.5 的「校验顺序」小节：**先结构参数，后 quick return**。

| 序 | 检查 | 失败返回 |
|---|---|---|
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `transa`/`transb` 不在枚举内 | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | `m < 0` 或 `n < 0` 或 `k < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `lda`/`ldb`/`ldc` 低于下界 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `alpha`/`beta` 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | quick return：`m=0 || n=0` | `SUCCESS`，不引用 A/B/C，不启 kernel；此时 A/B/C 允许为 nullptr |
| 7 | **读出 alpha 值**（见下「alpha 的读取时机」） | — |
| 8 | `m>0 && n>0` 时 `C` 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | `k>0 && alpha≠0` 时 `A`/`B` 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | quick return：`k=0 || alpha=0` | 只做 `C = beta*C`，A/B 允许为 nullptr |

> **契约取证优先级**（本团队 2026-11-11 Crot 的教训）：
> 同 arch 兄弟算子实现 > 任务书 §2.4/§2.5 > 配套 CSV 的 `expect_result`。

**本算子与 `aclblasCgemm` 存在两处 per-API 差异，不能直接复用后者的校验**。
这是本节最需要注意的地方，逐条给出证据：

**差异一：`alpha=0` 时豁免 A/B 空指针检查（必须新增）**

arch22 既有的 `ValidateCgemmPointers()` 签名里**没有 alpha**，判据只有 `k > 0`：

```cpp
// blas/gemm/arch22/cgemm_host.cpp:178-190
static aclblasStatus_t ValidateCgemmPointers(
    int k, const aclblasComplex* A, const aclblasComplex* B, const aclblasComplex* C)
{
    if (C == nullptr) { ... return ACLBLAS_STATUS_INVALID_VALUE; }
    if (k > 0 && (A == nullptr || B == nullptr)) { ... return ACLBLAS_STATUS_INVALID_VALUE; }
    return ACLBLAS_STATUS_SUCCESS;
}
```

而本任务书 §2.5「空 Tensor 处理」要求 `alpha = 0` 时允许 A/B 为 nullptr，
配套 CSV 亦有对应正向用例：

```
cgemm3m_test.csv:391
TC_ED_390,alpha0_a_null,ACLBLAS_OP_N,ACLBLAS_OP_N,16,16,16,0,0,NULLPTR,16,...,ACLBLAS_STATUS_SUCCESS
```

即 `m=n=k=16`、`alpha=(0,0)`、`A=NULLPTR`，期望 `SUCCESS`。
**若原样复用 `ValidateCgemmPointers()`，该用例必然 FAIL**，精度 1000 条拿不到 ALL PASS。

arch35 已给出标准修法，且只对 3m 生效：

```cpp
// blas/gemm/arch35/cgemm_host.cpp:722-723
const bool skipProduct = forceThreeProducts && alphaVal == std::complex<float>(0.0f, 0.0f);
st = ValidateCgemmPointers(skipProduct ? 0 : k, A, B, C, opName);
```

本设计采用同一写法：把 `alpha=0` 等价成 `k=0` 传入，复用既有函数体，不改其签名。

**差异二：非法枚举的返回码——arch35 的做法不能照搬（必须刻意不抄）**

arch35 对 3m 把 `INVALID_ENUM` 重映射成了 `INVALID_VALUE`（对齐 cuBLAS）：

```cpp
// blas/gemm/arch35/cgemm_host.cpp:713
return forceThreeProducts && st == ACLBLAS_STATUS_INVALID_ENUM ? ACLBLAS_STATUS_INVALID_VALUE : st;
```

**本任务书 §2.3 明确否定了这一行为**：「枚举类非法取值返回
`ACLBLAS_STATUS_INVALID_ENUM`（说明：cuBLAS 对非法 trans 返回
`CUBLAS_STATUS_INVALID_VALUE`，此处为 ops-blas 约定，**属有意差异**）」。
配套 CSV 亦然：

```
cgemm3m_test.csv:397  TC_ED_396,invalid_transa,999,...,ACLBLAS_STATUS_INVALID_ENUM
cgemm3m_test.csv:398  TC_ED_397,invalid_transb,...,999,...,ACLBLAS_STATUS_INVALID_ENUM
```

**本设计不实现该重映射。** 这是「与 arch35 分工对齐」的唯一例外，
实现时须显式注释说明，避免后来者"顺手补齐"成 arch35 的样子。

> **一句话总结这两条**：arch35 的 3m 校验段「取其一、弃其二」——
> 取 `skipProduct ? 0 : k`，弃 `INVALID_ENUM → INVALID_VALUE`。

**alpha 的读取时机（arch22 特有的工程约束）**

arch22 既有代码把 alpha/beta 的读取放在 `LaunchCgemmKernel()` 内部，
且走的是异步 D2H：

```cpp
// blas/gemm/arch22/cgemm_host.cpp:464
const aclblasStatus_t readRet = CBlas3ReadScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream, OP_TAG);
```

而校验发生在公开入口 `aclblasCgemm`（:522-531），**在此之前**。
要实现「差异一」的 `alpha=0` 豁免，alpha 的值必须在指针校验**之前**拿到，
因此本设计把 alpha 读取上移到指针校验之前（上表第 7 步）。

> **待确认（§5.1 第 1 项）**：任务书 §2.4 与 §3.1.2 均约定 alpha/beta 指向
> **Host 内存**，arch35 也是直接 `alpha->real` 解引用；但 arch22 既有代码按 device
> 指针做 D2H 异步拷贝。两者口径不一致，需在实现阶段实测确认 arch22 对 Host 指针
> 的处理路径，再决定上移后用哪种读法。**此项不影响本文其余设计。**

##### 3.2.1.2 workspace 规划

扩展既有 `PlanCgemmWorkspace()`：`forceThreeProducts=true` 时在既有布局之外追加
A 侧、B 侧各一块组合平面（`Ar+Ai`、`Br+Bi`），其余（`re`/`im`/`prod`/`scratch`）
沿用不变，具体口径与实测要求见 §3.1.5.1。平面前导维沿用既有 `ldAr`/`ldBr`/`ldRe` 约定。

##### 3.2.1.3 Phase 0：去交错并产出组合平面

复用既有 `RunCgemmSplits()` 的去交错与共轭吸收逻辑，在其后增加组合平面的生成：
`Ar+Ai`、`Bi−Br`、`Br+Bi`。组合为逐元素运算，由 AIV 完成。

> **为什么不在 GEMM 里做组合**：实数 GEMM 接口不接受「两矩阵之和」作为操作数，
> 组合必须物化成独立平面。这是 3m 多占一块 workspace 的根本原因。

##### 3.2.1.4 Phase 1：三次实数 GEMM

非融合的重组是 `Re = P0 − P1`、`Im = P2 − P0 − P1`。**不需要把三个积各存一块再做
一次独立重组**——既有的 `AccumulateRealPlane(dst, src, scale, overwrite)`
（`cgemm_host.cpp:322-330`）已经支持「带系数累加到目标平面」，
把每个积按系数分发到 `re`/`im` 即可：

| 积 | 目标平面与系数 |
|---|---|
| `P0 = Ar·Br` | `re` ← +1（首写）；`im` ← −1 |
| `P1 = Ai·Bi` | `re` ← −1（累加）；`im` ← −1 |
| `P2 = (Ar+Ai)·(Br+Bi)` | `im` ← +1（累加） |

累加完成后 `re`/`im` 直接就是 `Re(op(A)·op(B))` 与 `Im(op(A)·op(B))`。

> **这带来一个比省内存更重要的结果**：`re`/`im` 的布局与 4 乘法路径**完全一致**，
> 因此 **Phase 2 可以直接复用既有的 `cgemm_combine_kernel_do`**。
>
> 这一条实现后仍然成立 —— `cgemm_combine_kernel` 原样复用、一个字没改。
> 但原文顺带下的「不需要新增 3m 重组 kernel」与「`cgemm_kernel.h` /
> `cgemm_tiling_data.h` 不必改动」两句**已被实现阶段的实测推翻**：重组本身
> 下沉成了新的 Phase 1b 内核 `cgemm3m_fold_kernel`，Phase 0 也新增了
> `cgemm_split_sum_kernel`，两个头文件都有改动。理由见 §3.2.1.4a。
>
> 本设计初稿曾主张「三个积必须分别保留、需新增重组 kernel」，设计阶段复核时认为
> 该主张不成立：重组公式只要求线性组合，而线性组合正是 `AccumulateRealPlane`
> 的既有能力；多造一个 kernel 反而会撞上任务书 §5 的
> 「多出一些无用的代码、测试和编译文件」反例。
>
> **订正（实现阶段按 msprof 实测推翻）**：按系数分发到 `re`/`im` 在**功能上**
> 确实可行且已跑通，但**性能上不可接受**。实现后用 msprof 采 kernel Task Duration
> 发现：`cgemm_accum_kernel` 每次调用要跑 **10 趟**（4 趟算和平面、
> 2 趟 P0、2 趟 P1、2 趟 P2），每趟都读写一整个平面；256³ 用例上它独占
> **47.3%** 的 kernel 时间，而真正做矩阵乘的 Cube 只占 **19.3%**。
> 最终按 §3.2.1.4a 改为两个新内核。关于 §5 反例的顾虑：新增的不是「无用的代码」——
> 它们替换掉了 10 次下发与十余趟平面访存，有逐条实测支撑。

实现上把 `products[4]` 替换为 3m 的 3 项表（每项需分发到两个目标的，
由调用方对同一 `prod` 结果做两次 `AccumulateRealPlane`）：

```cpp
// sa/sb 为共轭符号（OP_C 时为 -1），见 §3.1.5；和平面已按 Ar+sa*Ai / Br+sb*Bi 生成
const Cgemm3mTarget p0Targets[2] = {{ws.re, 1.0F},          {ws.im, -1.0F}};
const Cgemm3mTarget p1Targets[2] = {{ws.re, signs.aiBi},    {ws.im, signs.aiBi}};  // signs.aiBi == -sa*sb
const Cgemm3mTarget p2Targets[1] = {{ws.im, 1.0F}};
// P0 = Ar*Br, P1 = Ai*Bi（原始平面）, P2 = (Ar+sa*Ai)*(Br+sb*Bi)
```

`RunOneCgemmProduct()`、k 方向分块（`CGEMM_K_CHUNK`）、
`avoidLeftTransDirect` 规避、转置归一化等逻辑全部复用，不改动。

##### 3.2.1.4a 订正：实际落地的 Phase 1b 结构与实测依据

本节是**实现阶段按实测改写设计**的记录，与 §3.2.1.4 的设计阶段结论不一致之处以本节为准。

**为什么改。** 按 §3.2.1.4 的方案实现并跑通后，用完整 `msprof --task-time=on`
采 `op_summary_*.csv` 的逐 kernel Task Duration（TC_PF_0001 = 256³，15 次下发）：

| kernel | 类型 | 次数 | 总耗时 | 占比 |
|---|---|---|---|---|
| `cgemm_accum_kernel` | AI_VECTOR_CORE | 150 | 470.7 µs | **47.3%** |
| `gemmex_fallback_wholegrid_square_nn_kernel` | **AI_CORE** | 45 | 191.9 µs | 19.3% |
| `cgemm_split_kernel` | AI_VECTOR_CORE | 30 | 174.6 µs | 17.6% |
| `cgemm_combine_kernel` | AI_VECTOR_CORE | 15 | 102.3 µs | 10.3% |
| `cgemm_risk_kernel` | AI_VECTOR_CORE | 15 | 55.0 µs | 5.5% |

`AI_VECTOR_CORE 80.7% / AI_CORE 19.3%` —— 真正做矩阵乘的 Cube 只占两成，
八成是 AIV 搬运。`cgemm_accum_kernel` 150 次 / 15 = **每次 10 趟**，与积表逐项对得上。

**改成什么。** 分三步，每步都有前后实测：

| 步骤 | 做法 | accum 趟数 |
|---|---|---|
| 1 | 新增 `cgemm_split_sum_kernel`：和平面在 Phase 0 去交织的同一趟里写出（加数已在 UB，**不多读一次 GM**） | −4 |
| 2 | 新增 `cgemm3m_fold_kernel`：三个积各写独立平面（`prodRe`/`prodIm`/`p2`，都是已有的），一趟融合算出 `re`/`im` | −6，+1 |
| 3 | 精度筛查内联进 fold 的最后一个 K 分块（那时 `im` 已是终值、P2 总和也在手），移除独立的 `cgemm_risk_kernel` | 下发 8→7 |

访存从「12 次平面读 + 6 次平面写」降到「3 次读（累加时 5 次）+ 2 次写」。
`cgemm_accum_kernel` 从主路径完全消失（纠正通路仍用它，那里每个积只喂一个平面）。

**效果（msprof kernel 口径，单次调用）：**

| 用例 | 形状 | 改前 | 改后 | `gpu_ms` 折合 | 倍率 |
|---|---|---|---|---|---|
| TC_PF_0001 | 256³ | 66.30 µs | **37.23 µs** | 26.01 µs | 0.392 → **0.699** |
| TC_PF_0002 | 1024³ | 396.93 µs | **294.55 µs** | 458.75 µs | 1.156 → **1.557** |
| TC_PF_0003 | 2048³ | 1974.71 µs | **1593.82 µs** | 3388.93 µs | 1.716 → **2.126** |

**等价性如何保证。** 不靠推导，逐项核对后再用数值实验确认：

- `re`/`im` 的线性组合在 1~4 个 K 分块、`aiBi = ±1` 下穷举模拟，与原积表**逐位一致**；
- 累加分支的**求和顺序必须保持** `im = ((im − P0) + coefP1·P1) + P2`。
  3m 里 `P2 ≈ P0 + P1 + Im`，P2 与 P0 都远大于 Im；先减 P0 把运行值压小、
  把对 P2 的相消留到最后。实现时一度写成 `((im + P2) + u) − P0`，
  float32 模拟 2 万次样本显示最大绝对误差放大 **1.52 倍**，真机上
  `TC_MN_216` 因此从通过（maxError 0.0146 / 限值 0.03125）变为失败（0.0381）。
- 分块归约时 `ws.p2` 每块被 GEMM 覆写，筛查需要的是**整个** P2，
  故新增 `ws.p2Acc` 由 fold 顺带累加。曾误判为「只影响启发式、可以省」，
  实测代价是丢掉 `TC_MN_216` 的纠正 —— 该用例在 4 乘法下通过，
  而纠正通路正是把 Im 按 4 乘法重算。`p2Acc` 只在真会分块时分配，
  **4096³ 走 whole-grid 单块，额外开销 0 MB**。

**验收判据（每轮重复）：** 精度失败清单与改动前**逐条相同**（`diff` 为空）、
`ALLOC_FAILED` 为 0、`aclblasCgemm` 88 条回归全过、`[PERF]` 140 行。

##### 3.2.1.5 选路：融合 vs 非融合

任务书 §2.1 给出两条分解，§2.5「确定性计算要求」允许「由实现按形状和片上容量选择，
不按测试编号或随机种子分派」。

**arch35 的既有选择是「默认非融合」，而不是默认融合**——这一点必须先说清楚，
因为它直接决定本设计的默认路径：

```cpp
// blas/gemm/arch35/cgemm_host.cpp:202-208
static uint32_t SelectCgemmFusedTile(const CgemmLaunchContext& context)
{
    const bool unitOutput =
        context.alpha == std::complex<float>(1.0f, 0.0f) && context.beta == std::complex<float>(0.0f, 0.0f);
    if (!unitOutput) { return 0; }      // ← 非 alpha=1/beta=0 一律不融合
    ...
```

`fusedTile == 0` 时 `standardThreeProducts = 1`（:379），拆分 kernel 走的是
**非融合**布局 `Ar, Ai, Ar+Ai`（`cgemm_deinterleave_common.h:38-42`），
即任务书的非融合分解 `P0=Ar·Br, P1=Ai·Bi, P2=(Ar+Ai)(Br+Bi)`。

**更关键的是：arch35 的融合路径需要「纠正重算」才达标。** 任务书 §2.1 把它写作
「融合路径（**含纠正重算**）」，arch35 对应的实现是 `layout.correctRounding`：

```cpp
// blas/gemm/arch35/cgemm_host.cpp:234-238
layout.correctRounding = allowRounding && layout.fusedTile != 0 && context.forceThreeProducts &&
                         context.m == context.n && context.n == context.k && context.k >= 2048 &&
                         context.k <= 4096 && context.lda == context.m && ...
                         context.alpha == std::complex<float>(1.0f, 0.0f) && context.beta == ...(0.0f, 0.0f);
```

配合整个 `cgemm3m_rounding_kernel.cpp`（499 行，筛选高风险元素后串行 FP32 重算）。
也就是说，**融合路径在 arch35 上是一条窄路径（方阵 + 2048≤k≤4096 + 前导维紧凑 +
alpha=1/beta=0），而且还挂着一套纠正机制**。

**本设计的选路决策**

| 项 | 决策 | 理由 |
|---|---|---|
| 默认路径 | **非融合** | 与 arch35 默认一致；不依赖纠正重算 |
| 融合路径 | **本期不实现** | 其收益需配套纠正重算才安全，而纠正 kernel 属 arch35 专有实现、arch22 无对应基础设施（§1.2.2(4)）；贸然只抄融合布局而不抄纠正机制，正是 §4.1.3 所列「纯实×纯虚、Re 理论为 0」抵消误差最剧烈的场景 |
| 若非融合性能不达标 | 再评估融合 + 纠正重算，作为增量 | 届时须同步实现纠正机制，不单独引入融合布局 |

非融合的重组公式为 `Re = P0 − P1`、`Im = P2 − P0 − P1`，
三个积各自需要写入两个目标平面（见 §3.2.1.4），**不按测试用例名或随机数分派**。

##### 3.2.1.6 Phase 2 与 alpha/beta

重组与标量施加合并在一个 AIV kernel：

```
Re(AB) = P0 − P2
Im(AB) = P0 + P1
C = alpha * (Re + i·Im) + beta * C
```

`beta = 0` 时不读 `C` 旧值（任务书 §2.1），由 kernel 内分支保证，
覆盖「`beta=0` 且 `C` 含 NaN」的用例（任务书 §3.5 补充特殊场景）。

#### 3.2.2 kernel 侧设计

| kernel | 位置 | 职责 |
|---|---|---|
| 既有拆分 kernel | `cgemm_kernel.cpp` | 去交错 + 共轭吸收，复用 |
| 新增组合 kernel | `cgemm_kernel.cpp` | 产出 `Ar+Ai`、`Bi−Br`、`Br+Bi` |
| 新增 3m 重组 kernel | `cgemm_kernel.cpp` | `Re=P0−P2`、`Im=P0+P1` + alpha/beta |

分核策略沿用既有 `RunCgemmCombine()` 的
`blocks = min(aivCoreNum, max(1, n))`，按列切分。

> **kernel_do 返回值的口径选择**：上游近期有一批「`kernel_do` 返回发射状态，
> 失败不再假成功」的修复（如 `55ef791`、`5a01644`）。但 **arch22 `cgemm` 现有的三个
> launcher 仍全是 `void`**（`cgemm_kernel.h:35/39/42-43`），错误靠
> `aclrtGetLastError(ACL_RT_THREAD_LEVEL)`（`cgemm_host.cpp:504`）兜底。
>
> 本设计**沿用既有的 `void` + `aclrtGetLastError` 口径**，不在同一组文件里引入第二种
> 风格。理由：本任务新增的组合 kernel 与既有拆分 kernel 同属 Phase 0，混用两种错误
> 传播方式会让 `cgemm_host.cpp` 的错误处理分叉；而把既有三个一并改成返回状态，
> 等于在上游刚合入的文件上追加一笔与本任务无关的重构，评审时更难解释。
> 若维护者要求统一改为返回状态，本设计可配合，作为独立提交。

## 3.3 支持硬件

| 项 | 值 |
|---|---|
| 适配硬件 | Atlas A2 系列产品、Atlas A3 系列产品 |
| 性能测试设备 | Atlas 800T A2（910B3），任务书 §3.3 指定 |
| SoC | arch22 |
| CANN | 任务书 §3.1 要求 9.1.0 |

> 自验证只需覆盖一种款型（任务书 §3.1）：精度与性能均在 910B3 上完成。
> 两款共用同一份 arch22 代码，blockdim 由设备上报核数决定，无按款型写死的常量，
> 故结论可共用；此点在自测报告中说明。

## 3.4 算子约束限制

| 约束 | 内容 |
|---|---|
| 非连续 Tensor | 不支持，按 `lda`/`ldb`/`ldc` 的列主序带填充布局访问 |
| broadcast | 不支持 |
| dynamic shape | 不要求，host 侧按运行时入参完成 tiling 与 launch |
| 原地语义 | `C` 原地覆写；调用方须保证 `C` 可写区与 `A`/`B` 可读区不重叠 |
| 确定性 | **不要求**（任务书 §2.5）。3m 路径与 K 方向归约顺序由实现按形状选择，同一输入两次执行不保证逐位一致 |
| 异步 | kernel 在 handle 绑定的 stream 上异步执行，读回前须同步 |

---

# 4. 可维可测分析

## 4.1 精度标准/性能标准

### 4.1.1 精度标准

golden 由 CPU 侧 `cblas_cgemm`（Netlib BLAS 3.10.0）生成，**实部、虚部分别比对**。
COMPLEX64 的阈值按 FLOAT32 分量执行（任务书 §3.2）：

| 项 | 值 |
|---|---|
| `rtol` | 2^-13（1.2e-4） |
| `atol` | 2^-13（1.2e-4） |
| `required_matched_ratio` | 0.99 |
| `max_abs_error_limit` | 1e-2 或 32·ULP |

判定：满足 `|actual − golden| ≤ atol + rtol·|golden|` 的分量计入 matched；
用例需同时满足 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max_abs_error_limit`。

> 任务书 §3.2.1 明确「3M 算法与直接复数乘法的舍入过程不同，结果与 `cblas_cgemm`
> 不保证逐位一致，按本节阈值判定」——**本算子不追求逐位一致**，这与本团队
> Crot 任务（逐位一致）是不同口径，不可套用。

### 4.1.2 性能标准与风险

**判据口径（本批次与 10 月 Crot 任务相反，须特别注意）**

| 口径 | 地位 | 采集方式 |
|---|---|---|
| `msprof op` kernel Task Duration | **正式判据** | warmup 5 次不计入，之后连续 10 次取算术平均 |
| `aclrtEvent` 端到端 `avg_us` | 仅交叉验证，**不得替代正式判据** | GTest 打印 `[PERF] case=... avg_us=...` |

任务书 §3.3.2 原文：「该事件计时与 `gpu_ms`（kernel-only）口径不完全一致，仅作交叉验证，
不作为正式判据」；§3.3.3：「使用 `msprof op` 在 NPU 上采集单算子 kernel task duration
…作为 kernel 单次耗时」。随包 `verify_performance.py` 文件头亦自述为
「性能交叉验证 / 快速跑测（非正式判据）」。

**判据**：NPU kernel 平均单次耗时 ≤ `gpu_baseline.csv` 的 `gpu_ms` × 1.25（即 ≥ 0.8× 性能）。
随包 `verify_performance.py` 的常量为 `MIN_RATIO = 0.8`（:33）与 `MIN_AVG_RATIO = 0.8`（:34），
即**单 case 与平均倍率均需 ≥ 0.8**。

**格式硬约束**：GTest 须为**每一条** TC_PF 用例打印
`[PERF] case=<case_name> ... avg_us=<数值>`。解析不到即判 FAIL——
本团队上一任务曾因只给部分形状打印、字段名不符，导致脚本报 `PASS=0 / FAIL=200`
而算子本身其实达标。本设计把该打印列为 R8 独立需求项，在性能调优之前先验证可解析。

**已识别的性能风险**

| 风险 | 说明 | 应对 |
|---|---|---|
| 小尺寸被下发开销淹没 | 端到端口径下 ACL 单次下发约 8.7 µs，存在 host 地板 | 正式判据是 kernel-only，不受此影响；交叉验证的差异在报告中说明 |
| Phase 0 的 AIV 开销被低估 | 组合平面需在既有拆分之外再过一遍 A/B（**两趟 AIV**），而 arch35 是一趟写完三块（`cgemm_deinterleave_common.h:37-46`） | 小尺寸下可能吃掉 25% 的 GEMM 收益；若实测确认，再评估把组合并入拆分 kernel 做成一趟 |
| workspace 变大 | 相对 4 乘法增加 A/B 侧各一块组合平面，叠加 `scratch`（比常驻平面还大）后 4096³ 有超限风险 | 按 §3.1.5.1 实测复核；超限按随包 README 指引在报告中说明并过滤，**不静默删除** |
| 性能基线键不匹配 | `verify_performance.py` 的判定键含 `alpha_real/alpha_imag/beta_real/beta_imag/lda/ldb/ldc`，任何键不匹配算 `NO_REF`，而 **`NO_REF` 默认判失败** | 用例参数须与 `gpu_baseline.csv` 逐键对齐；**不使用 `--allow-noref` 掩盖** |
| CPU 负载影响 | 本团队实测：一次并发编译把某形状从 54 µs 推到 82 µs | **测性能时不并发编译**，写入自验证步骤说明 |
| DVFS 降频 | 连续满载降频，不遵守冷却协议时重复性 ±1.2% 且整体偏慢约 10% | 每次计时前空转 300 ms、窗口约 30 ms |

**自设冗余目标（非任务书要求）**：在硬判据之上自设
`npu_avg ≤ gpu_ms × 1.1875`（= 1.25 × 0.95）作为 5% 安全垫，用于抵御随机波动。
执行顺序为**先让官方脚本全绿并固化一个可交付快照，再去压余量**。

### 4.1.3 已识别的首要精度风险与缓解设计

3m 的重组是相近量相加减（§1.2.1），当结果实部远小于中间积量级时会放大相对误差。
最危险的用例形态：

| 形态 | 来源 | 说明 |
|---|---|---|
| 纯实 × 纯虚 | 任务书 §3.5「正向用例的复数输入同时覆盖实部/虚部全零、纯实、纯虚、一般复数四类」 | `Re` 理论为 0，`P0 − P2` 为两个非零量相减 |
| 大 k 累加 | k 至 4096 | 累加误差随 k 增长 |
| ±1e4 标量 | 任务书 §3.5 alpha/beta 特殊值 | 放大绝对误差 |

缓解：
1. `max_abs_error_limit` 取 `max(1e-2, 32·ULP)`，ULP 随量级放大，对大值用例有保护；
2. 判定按分量（实部/虚部各自）而非复数模，避免一方为 0 时相对误差失真；
3. 若实测发现某类形态擦边，优先调整**选路阈值**（让该形态走非融合或 4 乘法等价路径），
   而非放宽阈值——阈值是任务书规定的，不可改。

> 本小节为设计期的风险识别，**实测结论待真机复核后写入自测报告**。
> 本团队上一任务的教训正是「下午断言没问题、晚上发现还有问题」，
> 故本文不预先断言精度结论。

## 4.2 确定性计算

任务书 §2.5 明确「不要求」确定性：3m 路径与 K 方向归约顺序由实现按形状和片上容量选择，
**不按测试编号或随机种子分派**，同一输入两次执行不保证逐位一致，验收按 §3.2 阈值判定。

## 4.3 兼容性分析

- 接口声明未改动，ABI 不变；
- `aclblasCgemm`（arch22）的**对外行为**不变（返回码与计算结果一致），但内部控制流有
  变动——校验与 quick return 移入 launcher，故不按「纯重构」对待，回归验证方式见 §2.5；
- arch35 的 `aclblasCgemm3m` 不受影响；
- 新增算子不影响既有算子的编译收集与测试门控（§2.4 的 CMake 登记为增量）。

## 4.4 测试方案

**用例来源**：任务配套 `test_cases/`，`gen_csv.py` 以固定种子 `20261124` 生成 1200 条
（同种子逐字节可复现）：精度 1000 条、性能 200 条（`TC_PF` 前缀）。

**CSV 列格式**（随包 README 给定）：

```
case_name,description,transa,transb,m,n,k,alpha_real,alpha_imag,a_fill,lda,
b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,
mere_threshold,mare_multiplier,random_seed
```

> **一处任务书与随包 README 的冲突，以 README 为准**：
> 任务书 §3.5 把正态分布填充 token 写作 `RANDOM_GAUSS_0_1`，
> 而 `test_cases/README.md` 明确「`GAUSS` 是 method，不能写成 `RANDOM_GAUSS_0_1`」，
> 正确 token 为 `GAUSS_NORM_0_1`。本设计按 README 实现，并在自测报告中记录该差异。

**覆盖面**：trans 全组合（9 种）× 尺寸扫描 × 标量特殊值 × 前导维 padding ×
填充 token × 边界负向；补充场景含 `m/n/k=0`、`alpha=0`、`beta=0 且 C 为 NaN`、
`k=0 且 beta≠1`、`OP_C` 用非对称复数矩阵验证共轭转置（结果须与 `OP_T` 不同）、
`ldc` padding 列哨兵写保护、Inf/NaN 传播。

**三类验证**

| 类别 | 命令 | 判据 |
|---|---|---|
| 精度 | `verify_accuracy.py` | ALL PASS（1000 条） |
| 性能 | `msprof op`（正式）+ `verify_performance.py`（交叉） | 单 case 与平均倍率均 ≥ 0.8 |
| 内存 | 任务书 §3.4 为「不涉及」 | 见下 |

> **内存一项的处理**：任务书 §3.4 写明「不涉及」，但 §4 的 `task_submission` 目录
> 仍明列 `4.1 内存自验证报告.xlsx` 与 `4.2 内存自验证日志.log` 共 7 个文件。
> 本设计据此**仍交付 7 个文件**，其中内存两份按「§3.4 不涉及」填写并注明依据，
> 而非留空或省略。

---

# 5. 待澄清事项与风险说明

## 5.1 仍需确认

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| 1 | alpha/beta 的地址空间 | 任务书 §2.4、§3.1.2 与 arch35 均按 **Host 内存**（直接解引用）；arch22 既有代码按 device 指针做 D2H 异步拷贝（`CBlas3ReadScalarsFromDevice`，:464） | 影响 §3.2.1.1 第 7 步 alpha 读取的实现写法，不影响校验语义与返回码 |
| 2 | 任务书 §4 交付件 2 描述的 `repo-task-case-gen` 产物（`cases.json`/`inputs/`/`golden/`/`facts.json`） | 随包 `test_cases/` 实际提供的是 CSV + `gen_csv.py` 流程，无上述产物 | 交付形态以随包实际内容为准，并在报告中说明 |
| 3 | 4096³ 的 device workspace 是否超限 | 随包 README 已预警「若 3M workspace 导致超限，可在自测报告中说明后过滤该 case，不得静默删除」 | 需实测复核（§3.1.5.1），超限时按 README 指引处理，**不静默删除用例** |

> **已关闭**：CANN 版本——开发机已升级至 9.1.0
> （`innerversion=V100R001C11SPC001B243`），与任务书 §3.1 一致，不再构成版本差异风险。
> 另：本设计不实现融合路径（§3.2.1.5），原「选路阈值待定」一项随之关闭。

## 5.2 已自行查明并关闭

| # | 问题 | 结论 | 证据 |
|---|---|---|---|
| 1 | 接口是否需新增声明 | 否，已存在 | 任务书 §7.3；`include/cann_ops_blas.h` |
| 2 | arch35 实现能否整体移植 | 否，arch22 无 `asc_vf_call` 编程模型 | `grep -rn "asc_vf_call" blas/*/arch22/` → 0 |
| 3 | arch22 是否已有可复用的复数 GEMM 骨架 | 是，三段式 + 声明式积表 | `blas/gemm/arch22/cgemm_kernel.cpp` 文件头；`cgemm_host.cpp` `RunCgemmRealProducts()` |
| 4 | arch22 是否已有共享 launcher | 否，需抽取 | arch35 有 `LaunchComplexGemm`（`cgemm_host.cpp:701`），arch22 无 |
| 5 | 性能正式判据是哪一套 | `msprof op` kernel Task Duration | 任务书 §3.3.2/§3.3.3/§7.4；`verify_performance.py` 文件头自述 |
| 6 | 正态分布 token 写法 | `GAUSS_NORM_0_1` | `test_cases/README.md`（与任务书 §3.5 冲突，以 README 为准） |
| 7 | `aclblasCgemm`（arch22）是否与本任务冲突 | 否，并列算子 | 上游 `5ebfa3c`（2026-11-24）新增，对应 arch35 的并列关系 |
| 8 | 能否原样复用 `aclblasCgemm` 的入参校验 | **否**，`alpha=0` 豁免 A/B 空指针这一条代码里不存在 | `cgemm_host.cpp:178-190` 的 `ValidateCgemmPointers` 无 alpha 形参；CSV:391 `TC_ED_390` 期望 SUCCESS |
| 9 | 能否照搬 arch35 的 3m 校验段 | **部分**，取 `skipProduct ? 0 : k`、弃 `INVALID_ENUM→INVALID_VALUE` 重映射 | arch35 `cgemm_host.cpp:713/722-723`；任务书 §2.3 把 cuBLAS 的 INVALID_VALUE 列为有意差异；CSV:397/398 期望 INVALID_ENUM |
| 10 | arch35 的默认路径是融合还是非融合 | **非融合** | `SelectCgemmFusedTile`（arch35 `cgemm_host.cpp:202-208`）非 `alpha=1/beta=0` 一律返回 0；`standardThreeProducts` 走 `Ar, Ai, Ar+Ai` 布局 |
| 11 | 3m 是否必须新增重组 kernel | **设计阶段判否，实现阶段按实测改判是** | 按系数分发功能上可行且已跑通，但 msprof 实测 `cgemm_accum_kernel` 每次调用 10 趟、占 256³ kernel 时间的 47.3%（Cube 仅 19.3%）。最终新增 `cgemm_split_sum_kernel` 与 `cgemm3m_fold_kernel`，单次 66.30 µs → 37.23 µs。见 §3.2.1.4a |
| 12 | `test/gemm/cgemm3m/CMakeLists.txt` 是否已支持 arch22 | **否**，硬门控 arch35-only 且静默跳过 | 该文件 :11-16 |
| 13 | arch22 的共轭落在哪一层 | **积表符号系数**，不是 Phase 0 拆分（与 arch35 相反） | `cgemm_split_kernel_do()` 无 conjugate 形参；`ResolveCgemmSigns()` 返回 `{-sa*sb, sb, sa}`。实现阶段实测订正，见 §3.1.5 |
| 14 | 单算子构建 `--ops=cgemm3m` 需要保留哪些源 | `gemm/arch22/cgemm_{host,kernel}.cpp` + `gemm_ex/arch22/` + `scal/arch22/` | 实测链接失败 `undefined reference to LaunchComplexGemm`；前缀规则 `^${_test_name}_` 匹配不到 `cgemm_*` |

---

# 6. 验收交付件对照

| 序号 | 交付件 | 本设计对应 |
|---|---|---|
| 1 | 算子设计文档 | 本文，以 PR 形式提交 cann-competitions 并**待合入后**再提交验收 |
| 2 | 自测用例及测试代码 | `test/gemm/cgemm3m/arch22/`，含 readme 说明复现步骤；配套 `test_cases/` 原样打入交付包 |
| 3 | 自测报告 | 腾讯文档模板；含用例参数、精度对比截图（实部/虚部分别）、两套口径性能数据截图 |
| 4 | 待验收代码地址 | 私仓链接 + 分支 + 算子目录；已邀请 `Ascend-CANN` 为开发者；`blas/gemm/README.md` 标明产品支持 |
| — | `task_submission/` | 私仓根目录 7 个文件（内存两份按 §3.4「不涉及」填写） |

> **提交时序**（任务书 §4 交付件 1 + 本团队 2026-08 两次打回的教训）：
> 设计文档 PR **合入之后**才能提交验收，仅「已提交」不算数。
