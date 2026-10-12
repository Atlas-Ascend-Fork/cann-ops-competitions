# aclblasCgbmv 算子设计文档

> 任务：9 月社区任务 - aclblasCgbmv 算子开发（A2/A3）
> 目标仓：`ops-blas`，实现路径 `blas/gbmv/arch22/`，测试路径 `test/gbmv/cgbmv/arch22/`
> 对标接口：cuBLAS `cublasCgbmv`（语义参考 Netlib BLAS `cgbmv.f`）

---

## 一、需求背景

### 1.1 需求来源

通过**社区任务**完成开源仓算子贡献。在昇腾 NPU（Atlas A2 / A3 系列产品，arch22 / DAV_2201，性能测试设备 Atlas 800T A2 (910B3)）上使用 Ascend C 开发 `aclblasCgbmv`（complex64 一般带状矩阵-向量乘）算子，对齐 cuBLAS `cublasCgbmv` 语义，验收通过后合入 `ops-blas` 开源仓。

| 项 | 取值 |
|----|------|
| 任务书算子 | `aclblasCgbmv` |
| 目标硬件 | Atlas A2 系列产品 / Atlas A3 系列产品（arch22，SOC `ascend910b*` / `ascend910_93*`） |
| CANN 版本 | 9.1.0 |
| 参考标杆 | cuBLAS `cublasCgbmv`，语义参考 Netlib BLAS `cgbmv.f` |
| 精度 golden | cblas（Netlib BLAS）`cblas_cgbmv` |
| 代码仓 | `ops-blas`，实现 `blas/gbmv/arch22/`，测试 `test/gbmv/cgbmv/arch22/` |

### 1.2 背景介绍

#### 1.2.1 ops-blas 仓内 gbmv 家族现状分析

`ops-blas` 仓内 `blas/gbmv/` 已有实数单精度实现 `aclblasSgbmv`，但其产品支持情况为：

| 产品 | aclblasSgbmv |
|------|--------------|
| Ascend 950PR / Ascend 950DT | 支持 |
| Atlas A3 训练/推理系列产品 | **不支持** |
| Atlas A2 训练/推理系列产品 | **不支持** |

即：gbmv 家族当前**只有 arch35 的实数实现**，`blas/gbmv/` 下不存在 `arch22` 目录，复数版 `aclblasCgbmv` 接口在仓内完全缺失。本任务需要同时补齐两件事：

1. 新增 **arch22 架构**的 gbmv 实现（A2/A3 首次支持 gbmv）；
2. 新增 **complex64 数据类型**的 gbmv 接口 `aclblasCgbmv`。

因此本任务**不能**直接移植 `blas/gbmv/arch35/` 的实现。已核对仓内源码确认：`blas/gbmv/arch35/sgbmv_kernel.cpp` 使用 **SIMT 编程模型**（`#include "simt_api/asc_simt.h"`、`__simt_vf__`、`asc_vf_call`、`threadIdx.x`），而 **arch22 不支持 SIMT**（`agent/skills/repo-op-templates/SKILL.md` 编程模型表：「SIMT … 仅 arch35」）。架构目录与 SOC 的映射见根 `CMakeLists.txt`：`ascend910b* → arch22`、`ascend910_93* → arch22`、`ascend950* → arch35`，即 **arch22 恰好覆盖 A2 + A3**。

#### 1.2.2 关键先例：`blas/gemv/arch22/` 已有 complex64 Level-2 实现

仓内**已存在**一个与本任务高度同构的实现：`aclblasCgemv`（complex64 矩阵-向量乘）。`blas/gemv/README.md` 原文：

```
### aclblasCgemv
#### 产品支持情况
- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持
```

它是 gemv 家族中**唯一** A2/A3 支持的接口（同文件 `aclblasSgemv` 对 A2/A3 均为「不支持」），文件布局为：

```
blas/gemv/arch22/ascblasCgemv_utils.h        复数解交织/再交织/复数乘工具
blas/gemv/arch22/cgemv_host.cpp              aclblasCgemv 入口
blas/gemv/arch22/cgemv_no_trans_kernel.cpp   trans=N
blas/gemv/arch22/cgemv_do_trans_kernel.cpp   trans=T/C
```

注意目录层级为 `blas/<family>/<arch>/`，**无 `cgemv/` 中间层**；`blas/CMakeLists.txt` 以 `*/${arch_dir}/*.cpp` 自动 GLOB，故新建 `blas/gbmv/arch22/` **无需改动 CMake**。

**这是本设计最重要的参考资产**：complex64 在 arch22 AIV 上怎么拆实部/虚部、怎么写回，已有可直接复用的工程答案（详见 §3.4.1）。

#### 1.2.3 两种 arch22 编码风格的取舍（设计决策）

核对源码后发现仓内 arch22 存在**两套并存的风格**，必须明确选型：

| | 既有 `blas/gemv/arch22/cgemv_*`（现网代码） | `agent/skills/repo-op-templates` 的 simd-membase 模板（仓方推荐） |
|---|---|---|
| UB 管理 | `AsdopsBuffer<ArchType::ASCEND_V220>` + **手工字节偏移** | `TPipe` / `TQue` |
| 搬运 | `gm_to_ub_align` / `matrix_gm2ubuf` / `ub_to_gm_align`（仓内 `common/iterator/iterator.h`） | `DataCopyPad` |
| 同步 | 裸 `SET_FLAG/WAIT_FLAG/PIPE_BARRIER` | `SetFlag/WaitFlag<HardEvent::...>` |
| 计算 | `muls_v/add_v/sub_v/cadd_v` + intrinsic `vmla`/`vreducev2` | AscendC 矢量 API |
| Tiling 下发 | `aclrtMalloc` + `aclrtMemcpy` 到 GM，kernel 手写偏移解析 | **按值传参**（模板原文：「host 侧以 const 引用传入 kernel_do（无 GM 设备内存分配，无同步）」） |
| Workspace | 自行 `aclrtMalloc` 16MB（实际未用） | **从 handle 获取**（模板原文：「禁止自行 aclrtMalloc」） |
| 核数 | 硬编码 `numBlocks = 8` | 动态获取（违反即触犯编码规则 R2） |

**本设计选择：以 simd-membase 模板为骨架（host 拆分、kernel.h、按值 tiling、handle workspace、动态核数），仅复用既有 cgemv 的复数解交织/再交织技法。** 理由：

1. 既有 cgemv arch22 代码**触犯了仓内多条编码规则**——硬编码核数违反 `R2-动态获取CoreNum`，`incx≠1` 分支里的标量 `for` 循环逐元素拆分违反 `R1-禁止逐元素操作`，自行 `aclrtMalloc` workspace 违反模板约定。照抄会在 MR 评审被拦。
2. 每次调用 `aclrtMalloc` + `aclrtMemcpy` + `aclrtSynchronizeStream` + `aclrtFree` 三块 Device 内存的开销，对本任务 **10.665 us / 17.328 us** 的小 shape 性能预算是致命的（§5.2 case 1/2）。按值 tiling 可完全消除这部分开销。

> ⚠️ **既有 cgemv arch22 的一个已知缺陷，本设计必须规避**：其 host 仅按 `trans == ACLBLAS_OP_N` 二分派发，`OP_T` 与 `OP_C` 都走 `cgemv_do_trans`，而 kernel 内 `_transA` 读出后**从未被使用**（全文仅声明处出现一次）。即 arch22 的 Cgemv 对 `OP_C` 很可能返回的是 `OP_T` 的结果。本算子的 `OP_C` 必须真正实现共轭（对 A 的虚部取负），且自测用例 `TC_L0_005/006`、`TC_SQ_*_C`、性能 case 5 均为 `OP_C`，不容回避。

**参考资源路径：**

| 资源 | 路径 | 用途 |
|------|------|------|
| 接口声明 | `ops-blas/include/cann_ops_blas.h` | `aclblasCgbmv` 声明**需新增**（供其他产品线共用） |
| 实现目录 | `ops-blas/blas/gbmv/arch22/` | 本任务新建 |
| 算子 README | `ops-blas/blas/gbmv/README.md` | 已存在（Sgbmv），需追加 Cgbmv 章节与产品支持表 |
| 测试目录 | `ops-blas/test/gbmv/cgbmv/arch22/` | 本任务新建 |
| 代码模板 | `agent/skills/repo-op-templates/references/simd-membase/` | arch22 可用模板（SIMD membase） |
| 测试模板 | `agent/skills/repo-test-develop/references/test/{op}/` | GTest + CSV 六文件模板 |
| README 规范 | `docs/zh/develop/readme_develop_guide.md` | 算子 README 编写指引 |

#### 1.2.4 标杆算子支持的数据类型和数据格式

| 项 | 取值 |
|----|------|
| 输入/输出类型 | `aclblasComplex`（complex64：float real + float imag 交织存储，8 字节/元素） |
| 矩阵 A | m×n 一般带状矩阵，**列主序带状打包**存储于 `lda×n` 数组，`lda ≥ kl+ku+1`，Device 内存，只读 |
| 向量 x | 逻辑长度 `n`（trans=N）/ `m`（trans=T/C），步长 `incx`（可负），Device 内存，只读 |
| 向量 y | 逻辑长度 `m`（trans=N）/ `n`（trans=T/C），步长 `incy`（可负），Device 内存，原地读写 |
| 标量 alpha/beta | Host 内存复数标量指针 |
| 数据排布 | ND |

**带状打包存储索引换算（0-based，本设计统一采用 0-based）：**

```
A(i, j)  存放于  AB[(ku + i - j) + j * lda]        0 ≤ i < m, 0 ≤ j < n
有效范围：max(0, j-ku) ≤ i ≤ min(m-1, j+kl)
```

即主对角线位于打包数组的第 `ku` 行（0-based），上方 `ku` 行放上对角带，下方 `kl` 行放下对角带；左上 `ku×ku` 与右下 `kl×kl` 三角区为未定义填充，**不得引用**。

由此导出本设计反复使用的三个量（**列视角**，第 j 列）：

```
iLo(j) = max(0, j - ku)                  # 该列在 A 中的首个有效行
iHi(j) = min(m - 1, j + kl)              # 该列在 A 中的末个有效行
len(j) = iHi(j) - iLo(j) + 1             # 该列有效元素个数，内部列恒为 bandW
d0(j)  = max(0, ku - j)                  # 该列在打包数组中的起始行
```

**关键性质**：第 j 列的 `len(j)` 个有效元素在内存中**连续**，起始地址为 `AB + (j*lda + d0(j)) * 8B`。其中 `bandW = kl + ku + 1`。这是本设计选择"以列为基本搬运单元"的根本原因。

对称地（**行视角**，第 i 行）：

```
jLo(i) = max(0, i - kl)
jHi(i) = min(n - 1, i + ku)
```

第 i 行的有效元素 `A(i, j)` 在内存中的偏移为 `(ku + i - j) + j*lda = (ku + i) + j*(lda - 1)`，即**等差分布，公差为 `lda-1` 个元素**。这一性质保证行向量也能用一次带 stride 的 DMA 取回。

#### 1.2.5 标杆算子实现描述

cuBLAS `cublasCgbmv` / Netlib `cgbmv` 计算定义：

```
y := alpha * op(A) * x + beta * y
```

其中：

- `trans = ACLBLAS_OP_N`：`op(A) = A`（m×n），x 长 n，y 长 m
- `trans = ACLBLAS_OP_T`：`op(A) = Aᵀ`（n×m），x 长 m，y 长 n
- `trans = ACLBLAS_OP_C`：`op(A) = Aᴴ`（共轭转置，n×m），x 长 m，y 长 n
- `m` / `n` 恒为矩阵 A 的行数 / 列数，**不随 trans 翻转**；A 存储恒为 `lda×n`

展开为逐元素形式：

```
trans = N:   y[i] = alpha * Σ_{j=jLo(i)}^{jHi(i)}  A(i,j)       * x[j]  + beta * y[i]   , i ∈ [0, m)
trans = T:   y[j] = alpha * Σ_{i=iLo(j)}^{iHi(j)}  A(i,j)       * x[i]  + beta * y[j]   , j ∈ [0, n)
trans = C:   y[j] = alpha * Σ_{i=iLo(j)}^{iHi(j)}  conj(A(i,j)) * x[i]  + beta * y[j]   , j ∈ [0, n)
```

复数乘法：`(a+bi)(c+di) = (ac − bd) + (ad + bc)i`；共轭：`conj(a+bi) = a − bi`。

**步长语义（对齐 Netlib）**：设逻辑长度为 `len`，若 `inc > 0` 则逻辑第 k 个元素位于 `v[k*inc]`；若 `inc < 0` 则位于 `v[(len-1-k)*(-inc)]`。等价于：**基址前移 `(len-1)*|inc|`，步长取 `inc`（带符号）**，即 `base = v + (inc > 0 ? 0 : (len-1)*(-inc))`，逻辑第 k 个元素为 `base[k*inc]`。本设计在 host 侧完成该归一化。

**边界与快速返回语义（对齐 Netlib `cgbmv.f`）：**

| 条件 | 行为 |
|------|------|
| `m == 0 或 n == 0` | 合法 no-op，返回 SUCCESS，不读写任何 Device 内存 |
| `alpha == 0 且 beta == 1` | no-op（y 不变） |
| `alpha == 0`（beta ≠ 1） | 退化为 `y = beta * y`，**不读 A、不读 x** |
| `beta == 0` | **不读原 y**（y 可为未初始化 / Inf / NaN），直接写 `alpha*op(A)*x` |
| `beta == 1` | 跳过缩放，直接累加 |

> `alpha == 0` 与 `beta == 0` 的"不读"是**正确性要求而非性能优化**：测试用例 `TC_ED_233 (beta0_y_nan)` 以 NaN 填充 y 且 `beta=0`，期望输出不含 NaN；若实现中执行 `0 * NaN` 会得到 NaN 从而判错。

#### 1.2.6 标杆算子流程图

```
┌──────────────────────────────────────────────┐
│        aclblasCgbmv(handle, trans, ...)      │
└───────────────────────┬──────────────────────┘
                        ▼
          ┌─────────────────────────────┐
          │ handle == nullptr ?         │──yes──→ HANDLE_IS_NULLPTR
          └─────────────┬───────────────┘
                        ▼ no
          ┌─────────────────────────────┐
          │ 参数校验（第一批）：         │
          │  trans 枚举 / m,n ≥ 0 /     │──fail─→ INVALID_VALUE
          │  incx ≠ 0 / incy ≠ 0 /      │
          │  alpha,beta ≠ nullptr        │
          └─────────────┬───────────────┘
                        ▼ pass
          ┌─────────────────────────────┐
          │ m == 0 || n == 0 ?          │──yes──→ SUCCESS（no-op，不触碰 Device）
          └─────────────┬───────────────┘
                        ▼ no
          ┌─────────────────────────────┐
          │ 参数校验（第二批）：         │
          │  0 ≤ kl ≤ m-1 /             │──fail─→ INVALID_VALUE
          │  0 ≤ ku ≤ n-1 /             │
          │  lda ≥ kl+ku+1 /            │
          │  A,x,y ≠ nullptr             │
          └─────────────┬───────────────┘
                        ▼ pass
          ┌─────────────────────────────┐
          │ alpha==0 且 beta==1 ?       │──yes──→ SUCCESS（no-op）
          └─────────────┬───────────────┘
                        ▼ no
          ┌─────────────────────────────┐
          │ 步长归一化 + Tiling 计算     │
          │ 选择分派路径(trans×步长)     │
          └─────────────┬───────────────┘
                        ▼
          ┌─────────────────────────────┐
          │ Launch kernel（AIV 多核）    │
          │  各核计算互不相交的 y 区段   │
          └─────────────┬───────────────┘
                        ▼
                    SUCCESS
```

> **校验顺序说明（设计决策）**：任务书 §2.4 要求 `0 ≤ kl ≤ m-1`、`0 ≤ ku ≤ n-1`，但 `m = 0` 时该约束自相矛盾（`m-1 = -1`），而任务书同时要求 `m = 0` 为合法 no-op。本设计的解法是**把 kl/ku/lda 与 A/x/y 空指针校验放在零维快速返回之后**，使 `m=0 / n=0` 无条件返回 SUCCESS。该顺序与自测用例 `TC_ED_230/231/232`（`m=0, n=4, kl=0, ku=0` 等期望 SUCCESS）一致，也与 "A/x/y 仅在参与计算时才要求非空" 的描述一致。

---

## 二、需求分析

### 2.1 需求描述

使用 Ascend C（SIMD membase 编程模型）在 arch22 架构上实现 `aclblasCgbmv`，kernel 直调方式，支持 complex64 数据类型，参数语义与 cuBLAS `cublasCgbmv` 完全对齐，并满足任务书 §3.2 精度标准与 §3.3 性能标准。

### 2.2 需求拆解

1. 正确实现带状打包存储的索引换算（`A(i,j) ↔ AB[(ku+i-j) + j*lda]`），正确处理 `lda > kl+ku+1` 的 padding 场景与带外三角区不引用；
2. 支持 `trans` 三枚举（N / T / C），其中 C 为共轭转置；
3. 支持任意非零步长 `incx` / `incy`，含负步长；
4. 实现 alpha / beta 的全部特殊值语义（含 `alpha==0` 不读 A/x、`beta==0` 不读 y）；
5. 实现完整的参数合法性校验与 no-op 语义；
6. 多核并行且无跨核归约，满足 §3.3 五个典型 case 的达标耗时；
7. 补齐 `include/cann_ops_blas.h` 接口声明、`blas/gbmv/README.md` 文档、`test/gbmv/cgbmv/arch22/` 测试工程。

### 2.3 外部组件依赖

| 依赖组件 | 版本/说明 | 用途 |
|----------|-----------|------|
| ACL (AscendCL) | CANN 9.1.0 | Device 内存管理、stream、kernel launch |
| Ascend C kernel 运行时 | CANN 9.1.0 | `DataCopyPad`/`GatherMask`/`Gather`/`Mul`/`Add`/`Sub`/`ReduceDataBlock`/`ArithProgression` 等；UB 管理方式按 §1.2.3 选型 |
| `platform_ascendc` | CANN 9.1.0 | 运行时查询 AIV 核数、UB 容量（不硬编码芯片规格） |
| GTest | 系统已装 | 参数化测试框架 |
| **Netlib** CBLAS / LAPACK | 系统已装 | golden `cblas_cgbmv` 生成。**必须是 Netlib，不得链到 OpenBLAS**，详见 §3.6 风险 R8 |

### 2.4 内部适配模块

| 内部模块 | 说明 |
|----------|------|
| `include/cann_ops_blas.h` | **新增** `aclblasCgbmv` 声明 |
| `include/cann_ops_blas_common.h` | `aclblasStatus_t` / `aclblasOperation_t` / `aclblasComplex` 既有定义，复用 |
| `common/helper/aclblas_handle_internal.h` | handle 结构、stream 获取、workspace 管理 |
| `common/helper/kernel_constant.h` | UB 容量、buffer 数等常量 |
| `common/helper/host_utils.h` | host 侧工具函数，含 **`GetAivCoreCount()`**（kernel 直调场景取 AIV 核数的正确途径） |
| `common/iterator/iterator.h` | `gm_to_ub` / `gm_to_ub_align` / `ub_to_gm_align` / `matrix_gm2ubuf` 等 arch22 搬运助手 |
| `common/compute/simd.h` | `muls_v` / `add_v` / `sub_v` / `mul_v` / `cadd_v` 等 arch22 矢量助手 |
| `blas/gemv/arch22/ascblasCgemv_utils.h` | **复用**：`split_CVector()` 复数解交织、`mulSComplex()` 复数标量乘 |
| `log/log.h` | `OP_LOGE` / `OP_LOGD` / `OP_LOGI`（禁止 printf / std::cout） |
| `test/frame/`（`csv_loader.h` / `blas_test.h` / `fill.h` / `verify.h`） | CSV 驱动 ST 框架 |
| `test/utils/cblas_compat.h` | `ToCblasOp` 等枚举映射、Fortran BLAS 声明 |

### 2.5 算子原型

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t          handle,   // ops-blas 句柄（携带 stream），Host
    aclblasOperation_t       trans,    // ACLBLAS_OP_N / OP_T / OP_C
    int                      m,        // A 的行数（不随 trans 翻转）
    int                      n,        // A 的列数（不随 trans 翻转）
    int                      kl,       // 下对角带数
    int                      ku,       // 上对角带数
    const aclblasComplex    *alpha,    // 复数标量，Host 内存
    const aclblasComplex    *A,        // 带状打包矩阵，lda×n，Device，只读
    int                      lda,      // 前导维，lda ≥ kl+ku+1
    const aclblasComplex    *x,        // 输入向量，Device，只读
    int                      incx,     // x 步长，可负，≠ 0
    const aclblasComplex    *beta,     // 复数标量，Host 内存
    aclblasComplex          *y,        // 输出向量（原地），Device
    int                      incy);    // y 步长，可负，≠ 0
```

### 2.6 与标杆算子的功能范围对比

| 功能点 | cuBLAS `cublasCgbmv` | 本设计 | 说明 |
|--------|----------------------|--------|------|
| 数据类型 | complex64 | complex64 | 一致 |
| trans 分支 | N / T / C | N / T / C | 一致（C 为共轭转置） |
| 带状打包存储 | 列主序，lda ≥ kl+ku+1 | 一致 | 支持 lda padding |
| 非方阵 | 支持 | 支持 | m ≠ n 任意 |
| 正负步长 | 支持 | 支持 | 按 Netlib 语义归一化 |
| alpha/beta 特殊值 | 支持 | 支持 | 含 `alpha=0` 不读 A/x、`beta=0` 不读 y |
| 零维 no-op | 支持 | 支持 | m=0 或 n=0 返回 SUCCESS |
| Inf/NaN 传播 | IEEE754 | IEEE754 | 不做特殊钳位 |
| 确定性计算 | 不保证 | 不保证 | 任务书 §2.5 不要求 |
| 标量位置 | Host/Device 指针 | **仅 Host** | 任务书 §2.4 明确 alpha/beta 为 Host 内存 |
| 内存重叠 | 未定义 | 未定义 | 任务书 §2.5：y 与 A/x 不允许重叠 |

---

## 三、详细设计

### 3.1 调用方式

采用 **kernel 直调（ops-blas 句柄式 BLAS 接口）**：

- 上层调用 `aclblasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)`；
- host 侧 `cgbmv_host.cpp` 完成 handle 校验、参数校验、no-op 判定、步长归一化、tiling 计算、分派路径选择、kernel launch；
- device 侧 `cgbmv_kernel.cpp` 执行带状矩阵-向量乘；
- 异步语义依赖 `aclblasSetStream` 绑定的 stream，调用方读回结果前须 `aclrtSynchronizeStream`。

**代码交付件布局**（遵循 `repo-op-templates` 的 2 层目录 + host 拆分规范）：

```
blas/gbmv/
├── README.md                          -- 追加 aclblasCgbmv 章节与产品支持表
└── arch22/
    ├── cgbmv_tiling_data.h            -- CgbmvTilingData（host/kernel 共享，按值传递）
    ├── cgbmv_kernel.h                 -- 两个 kernel_do 签名（host/kernel 共用）
    ├── cgbmv_no_trans_kernel.cpp      -- Device 侧：trans=N
    ├── cgbmv_do_trans_kernel.cpp      -- Device 侧：trans=T/C（模板实参 CONJ 区分）
    └── cgbmv_host.cpp                 -- Host 侧 API 入口 + Validate/Launch 拆分 + dlog
```

文件数与既有 `blas/gemv/arch22/`（utils + host + 2 个 kernel）同级，符合任务书 §5「文件个数明显不符合正常数量」的反例规避要求。复数解交织工具优先**复用** `blas/gemv/arch22/ascblasCgemv_utils.h` 而非新建副本；若该头文件不便跨算子引用，则在 `cgbmv_kernel.h` 内以同名内联函数实现，并在 PR 说明中注明来源。

命名遵循仓规范：目录/文件 snake_case，结构体 `CgbmvTilingData`，kernel 类 `CgbmvAIV`，入口 `cgbmv_kernel`，启动器 `cgbmv_kernel_do`，公共 API `aclblasCgbmv`。

### 3.2 总体设计

#### 3.2.1 核心思路：统一为"每个输出元素一次复数点积"

带状矩阵-向量乘的难点在于 A 的两种访问方向（按列 / 按行）与输出的归约方向冲突。本设计的核心选择是：

> **按输出元素分核**：每个 AIV 核负责输出向量 y 的一段**互不相交**的连续区间，每个输出元素由一次独立的复数点积算出。

由此得到两个关键收益：

1. **零跨核归约、零 workspace、零核间同步**——各核写 y 的不同位置，不存在写冲突，不需要 `CrossCoreSetFlag/WaitFlag`，也不需要 atomic add 或中间 workspace。这对小 shape（性能 case 1/2 规模仅 256/512）尤其重要，因为跨核同步的固定开销会直接吃掉 10 us 级的预算。
2. **三个 trans 分支共用同一套计算内核**——差异被压缩到"如何把 A 的那一段取进 UB"和"虚部符号"两点上。

三个分支的映射关系：

| trans | 输出元素 | 需要的 A 向量 | 需要的 x 向量 | A 的内存形态 |
|-------|---------|--------------|--------------|-------------|
| N | `y[i]`，i ∈ [0, m) | 第 i **行**：`A(i, jLo(i)..jHi(i))` | `x[jLo(i)..jHi(i)]` | 打包数组的一条**对角线** → 列块连续搬入 + UB 内 `Gather`（§3.4.2） |
| T | `y[j]`，j ∈ [0, n) | 第 j **列**：`A(iLo(j)..iHi(j), j)` | `x[iLo(j)..iHi(j)]` | **内存连续** → 单次 burst |
| C | `y[j]`，j ∈ [0, n) | 同 T，取共轭 | 同 T | **内存连续** → 单次 burst |

两条路径最终都把 A 段与 x 段落到 UB 的**对齐起始地址**，于是 UB 内所有向量指令的操作数起始地址恒为 32B 对齐——这规避了"带状结构的滑动窗口偏移（`iLo(j)` 随 j 递增 1）"必然带来的非对齐访问问题。该对齐要求是**硬约束而非性能建议**（见 §3.6 风险 R1），因此这一点是本设计结构选择的根据，而不是事后优化。

#### 3.2.2 整体流程

```
host:   handle 校验 → 参数校验(第一批) → 零维 no-op → 参数校验(第二批)
        → alpha=0&&beta=1 no-op → 步长归一化 → Tiling → 选分派路径 → launch

kernel: 按 blockIdx 取本核输出区间 [outStart, outEnd)
        → 逐输出元素（按 tile 流水）:
             CopyIn : A 段 + x 段 → UB
             Compute: 解交织 → 复数乘 → 归约 → ×alpha → +beta*y_old
             CopyOut: 写回 y
```

### 3.3 Host 侧设计

#### 3.3.1 参数校验（`ValidateCgbmvParams`）

严格按 §1.2.6 流程图的两批顺序。全部失败分支以 `OP_LOGE` 打印入参后返回。

**本顺序与仓内同族 `aclblasSgbmv` 的现网实现逐步对齐**（`blas/gbmv/arch35/sgbmv_host.cpp`），不是自创：

| 序 | 位置 | 校验项 | 返回码 |
|----|------|--------|--------|
| 0 | `aclblasCgbmv` 入口 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 1 | `aclblasCgbmv` 入口 | `m < 0` 或 `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| — | `aclblasCgbmv` 入口 | **`m == 0 \|\| n == 0` → 直接 `return ACLBLAS_STATUS_SUCCESS`** | — |
| 2 | `ValidateCgbmvParams` | `trans ∉ {OP_N, OP_T, OP_C}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `ValidateCgbmvParams` | `kl < 0` 或 `kl > m-1` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `ValidateCgbmvParams` | `ku < 0` 或 `ku > n-1` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `ValidateCgbmvParams` | `lda < kl + ku + 1` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `ValidateCgbmvParams` | `incx == 0` 或 `incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `ValidateCgbmvParams` | `alpha == nullptr` 或 `beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `ValidateCgbmvParams` | `A == nullptr` 或 `x == nullptr` 或 `y == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | Launch 前 | `GetAivCoreCount() == 0` | `ACLBLAS_STATUS_EXECUTION_FAILED` |

现网 `aclblasSgbmv` 的对应代码（节选，印证第 1 步与零维返回的相对顺序）：

```cpp
CHECK_RET(h != nullptr, OP_LOGE("aclblasSgbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
CHECK_RET(m >= 0 && n >= 0, OP_LOGE(...); return ACLBLAS_STATUS_INVALID_VALUE);
if (m == 0 || n == 0) { return ACLBLAS_STATUS_SUCCESS; }          // ← 零维返回早于 kl/ku/lda/指针校验
aclblasStatus_t st = ValidateSgbmvParams(trans, m, n, kl, ku, lda, incx, incy, alpha, beta, A, x, y);
```

**两处与 Sgbmv 的有意差异：**

1. Sgbmv 只校验 `kl >= 0` / `ku >= 0`，**不设上界**；本任务书 §2.4 明确要求 `kl ≤ m-1`、`ku ≤ n-1`，故本设计增加上界校验。静态核对 1200 条用例：无任何 SUCCESS 用例违反该上界，增加它不会误杀（但该上界也未被负向用例覆盖，属"按任务书从严"）。
2. `trans` 非法枚举，本任务书要求返回 `INVALID_VALUE`（而非某些算子用的 `INVALID_ENUM`），用例 `TC_ED_241` 期望值即 `ACLBLAS_STATUS_INVALID_VALUE`，以任务书与用例为准。

> **零维返回必须早于 kl/ku 校验**，否则 `m=0` 时 `kl ≤ m-1 = -1` 恒不成立，会把合法 no-op 误判为非法。该顺序已由 §5.5 的可执行化回归对 1200 条用例验证（0 条不一致），并与现网 Sgbmv 一致。

#### 3.3.2 步长归一化

```cpp
lenX = (trans == ACLBLAS_OP_N) ? n : m;     // x 的逻辑长度
lenY = (trans == ACLBLAS_OP_N) ? m : n;     // y 的逻辑长度
xBase = x + ((incx > 0) ? 0 : (int64_t)(lenX - 1) * (-incx));
yBase = y + ((incy > 0) ? 0 : (int64_t)(lenY - 1) * (-incy));
// 逻辑第 k 个元素 = xBase[k * incx]（incx 带符号），y 同理
```

kernel 只感知 `(xBase, incx)` / `(yBase, incy)`，不再关心正负，负步长的复杂性被完全收敛在 host 的两行代码里。

#### 3.3.3 分核策略

```
outLen     = lenY                                    # 输出元素总数 = 待分配的任务量
aivNum     = GetAivCoreCount()                       # 仓内 host_utils.h 助手，禁止硬编码
useCoreNum = min(aivNum, ceil(outLen / MIN_ELEM_PER_CORE))
perCore    = ceil(outLen / useCoreNum)
本核区间   = [blockIdx * perCore, min((blockIdx+1) * perCore, outLen))
```

> **AIV 核数必须运行时查询，且应使用仓内助手**：本任务为 kernel 直调（无 `gert::TilingContext`），因此**不能**用 `platform_ascendc::PlatformAscendC::GetCoreNumAiv()`（其需从 `context->GetPlatformInfo()` 构造）。仓内同族 `blas/gbmv/arch35/sgbmv_host.cpp` 的做法是调用 `common/helper/host_utils.h` 提供的 `GetAivCoreCount()`，返回 0 时报 `ACLBLAS_STATUS_EXECUTION_FAILED`，本设计沿用。规格侧：CANN 文档对 Atlas A2 系列只给出系列级规格——「Atlas A2 推理系列产品、Atlas A2 训练系列产品（**20/24 个 AICore**）」，且 arch2201 的 AIC:AIV 配比为 **1:2**，即 A2 系列的 AIV 数为 **40 或 48**（随具体款型而异）。**官方文档中查不到「Ascend 910B3 = 40 AIV」这一具体型号的规格条目**，因此本设计不写死任何核数常量，一律用 `platform_ascendc::PlatformAscendC::GetCoreNumAiv()` 获取。编码规则 `R2-动态获取CoreNum` 明文禁止硬编码核数（反例即 `constexpr int numBlocks = 28;`）——而既有 `blas/gemv/arch22/cgemv_host.cpp` 恰恰写死了 `numBlocks = 8`，不可照抄。

- `MIN_ELEM_PER_CORE` 取 1（gbmv 单个输出元素的计算量正比于 `bandW`，带宽大时单元素也值得独占一个核）。但为避免极小 shape 下多核启动开销反超收益，额外加一条阈值：当 `outLen * bandW < SINGLE_CORE_THRESHOLD`（经验值，调优确定）时强制 `useCoreNum = 1`。性能用例含 `pf_small_1`（m=n=1）等极小 case，该阈值直接决定这些 case 的耗时。
- **负载均衡性**：带状矩阵的边界列/行 `len(j) < bandW`（首 `ku` 列与末 `kl` 列），内部列恒为 `bandW`。当 `bandW ≪ outLen` 时边界不均衡可忽略；当 `bandW` 接近 `outLen`（如用例 `TC_BW` 的 `kl=ku=31, m=n=32`、性能用例 `pf_bw_2047`）时首尾核任务量约为中间核的一半。本版采用**均分输出元素**的简单策略，在 §3.6 列为可选优化项（按 `Σlen` 前缀和均分）。

#### 3.3.4 UB 分块策略

arch2201（A2/A3 共用）的 UB 总容量为 **192KB**，但**用户实际可用容量小于该值**：框架与编译器会预留 256B，默认还会再占用 8KB，即默认可用上限为 `192KB − 256B − 8KB ≈ 183.75KB`（开启 `--cce-disable-asc-reserved-ubuf` 时为 `192KB − 256B`）。因此 **不得按 192KB 整块规划**，实现中统一用 `ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize)` 运行时查询（注意 `ubSize` 必须为 `uint64_t`）。

单次处理的基本单元是一个输出元素所需的 `bandW` 长复数向量对。UB 预算：

| Buffer | 内容 | 元素数 | 字节（fp32 计） |
|--------|------|--------|----------------|
| `aQue`（double buffer） | A 段，complex64 交织 | `TB` | `2 × TB × 8` |
| `xQue`（double buffer） | x 段，complex64 交织 | `TB` | `2 × TB × 8` |
| `aRe/aIm` | A 段解交织 | `TB` ×2 | `TB × 8` |
| `xRe/xIm` | x 段解交织 | `TB` ×2 | `TB × 8` |
| `tmp0/tmp1` | 乘积与归约中间量 | `TB` ×2 | `TB × 8` |
| `yQue` | y 读回/写出 | 小常量 | — |

总计约 `TB × 56 B`。按默认可用上限 183.75KB 再留余量取 160KB，得 `TB_max ≈ 2900`；设计取 `TB = 2048` 时占用 `2048 × 56 B = 112 KB`，相对 183.75KB 上限有充分余量。设计取：

```
TB = min(bandW, TB_MAX)            TB_MAX = 2048（32B 对齐、2 的幂，便于 mask/repeat 规划）
numTiles = ceil(bandW / TB)
```

- `bandW ≤ 2048` → `numTiles = 1`，一个输出元素一次搬运算完，流水最简。**全部 5 个任务书性能 case 均落在此路径**（bandW 分别为 33 / 65 / 129 / 257 / 321）；
- `bandW > 2048` → 分 tile 累加部分和，tile 间复用同一组 UB buffer。对 1200 条自测用例做静态统计，**仅 4 条合法用例的 `bandW > 2048`，全集最大 `bandW = 4095`**（`kl=ku=2047, m=n=4096`，即 `TC_PF_1073` 一类带宽扫描用例，其 A 规模达 128 MB），故 `numTiles ≤ 2`。

> `TB = 2048` 的另一个作用是**给 `DataCopyPad` 的 `blockCount` 设上界**。trans=N 分支的 `blockCount` 等于单行有效元素数，静态统计其全集最大值为 **4095**；经 `TB` 分块后恒 `≤ 2048`，从而与 `blockCount` 的硬件上限（**4095**，`uint16_t`，已核实）保持安全距离。

**Double buffer**：`aQue` / `xQue` 均设 `BUFFER_NUM = 2`，使第 k 个输出元素的 Compute 与第 k+1 个的 CopyIn 重叠。由于每个输出元素的搬运量固定为 `bandW×8B×2`，当 `bandW` 较小（如 case 1 的 33 → 264B）时 DMA 处于小包状态，此时**沿输出维度批量化**（一次 CopyIn 覆盖 `TJ` 个输出元素）是主要优化手段，见 §3.6 优化项 O1。

#### 3.3.5 分派路径规划（kernel 入口 × 模板实参）

> **用词更正**：kernel 直调工程**没有**自定义算子工程里的 `TILING_KEY_IS` 分派机制（`KERNEL_TASK_TYPE` 文档亦注明 Tiling 模板编程走 `ASCENDC_TPL_KERNEL_TYPE_SEL`）。仓内 arch22 的实际做法是**按形态拆分独立 kernel 入口**——`blas/gemv/arch22/` 即拆成 `cgemv_no_trans` 与 `cgemv_do_trans` 两个 `__global__` 函数，由 host 侧分派。simd-membase 模板也给出同样的范式：「多路径分发…`if (incx == 1 && incy == 1) { … } else { … }` … tiling 中编码 dispatch 信息，kernel_do 内分支」。
>
> 因此下表描述的是 **host 侧分派矩阵 + kernel 模板实例化**，不是 tilingKey 数值。

`trans` 的虚部符号与 A 的搬运形态是 kernel 内最频繁的分支，必须编译期确定；步长是否为 1 决定 x 段是整段 burst 还是 stride 搬运，差异同样显著。据此规划 **6 条分派路径**：

| kernel 入口 | 模板实参 | 含义 |
|-------------|----------|------|
| `cgbmv_no_trans_kernel` | `<CONJ=false, IS_CONT=true>` | trans=N，`incx==1 && incy==1` |
| `cgbmv_no_trans_kernel` | `<CONJ=false, IS_CONT=false>` | trans=N，任意步长 |
| `cgbmv_do_trans_kernel` | `<CONJ=false, IS_CONT=true>` | trans=T，连续 |
| `cgbmv_do_trans_kernel` | `<CONJ=false, IS_CONT=false>` | trans=T，任意步长 |
| `cgbmv_do_trans_kernel` | `<CONJ=true,  IS_CONT=true>` | trans=C，连续 |
| `cgbmv_do_trans_kernel` | `<CONJ=true,  IS_CONT=false>` | trans=C，任意步长 |

N 与 T/C 拆成**两个 kernel 入口**，因为二者取 A 的方式本质不同（对角线 `Gather` vs 整列连续）；T 与 C 则共用入口、以模板布尔实参 `CONJ` 区分，使虚部符号成为**编译期常量**，消除内层循环分支且不重复源码。

> ⚠️ **必须真正实现 `CONJ`**：既有 `blas/gemv/arch22/` 的 Cgemv 把 `OP_T`/`OP_C` 合流且 kernel 内 `_transA` 从未使用（§1.2.3），是本任务要规避的缺陷。

`alpha == 0` 的 `y = beta*y` 路径**不单独开入口**，而是在 kernel 内按 `tiling.alphaIsZero` 提前跳过 A/x 的 CopyIn——它既不改变分核方式也不改变搬运形态，一次运行时判断的开销可忽略，而多一个入口就多一份 kernel 二进制。`beta == 0` / `beta == 1` 同理，用 `tiling.betaMode ∈ {ZERO, ONE, GENERAL}` 运行时区分。

#### 3.3.6 TilingData 结构

```cpp
struct CgbmvTilingData {
    int32_t m;              // A 行数
    int32_t n;              // A 列数
    int32_t kl;             // 下带宽
    int32_t ku;             // 上带宽
    int32_t lda;            // 前导维
    int32_t bandW;          // kl + ku + 1
    int32_t outLen;         // 输出元素数：N → m，T/C → n
    int32_t inLen;          // 输入向量逻辑长度：N → n，T/C → m
    int32_t incx;           // 归一化后带符号步长
    int32_t incy;           // 归一化后带符号步长
    int32_t useCoreNum;     // 实际使用核数
    int32_t perCore;        // 每核输出元素数（末核可能更少）
    int32_t tileLen;        // TB：单次处理的 band 元素数
    int32_t numTiles;       // ceil(bandW / TB)
    float   alphaRe;        // alpha 实部（Host 标量随 tiling 下发）
    float   alphaIm;
    float   betaRe;
    float   betaIm;
    uint32_t alphaIsZero;   // alpha == (0,0)
    uint32_t betaMode;      // 0=ZERO, 1=ONE, 2=GENERAL
};
```

**下发方式：按值传参，不碰 Device 内存。** 依据仓方 simd-membase 模板原文——「Tiling 传递方式：host 侧以 const 引用传入 `kernel_do`（**无 GM 设备内存分配，无同步**）」「host 侧计算后以 const 引用传入 kernel_do，kernel 以 **by value** 接收（运行时 launch 参数自动拷贝）」。同族 `blas/gbmv/arch35/sgbmv_host.cpp` 即如此（全文无 `aclrtMalloc`/`aclrtMemcpy`）：

```cpp
void cgbmv_do_trans_kernel_do(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace,
                              const CgbmvTilingData& tiling, uint32_t numBlocks, void* stream);
```

三点设计约束：

1. **alpha / beta 拆成 4 个 `float` 字段**（`alphaRe/alphaIm/betaRe/betaIm`），而非直接放 `aclblasComplex`——与既有 `blas/gemv/arch22/cgemv_host.cpp` 的字段惯例一致，kernel 侧再重组为 `aclblasComplex`。由此 **host 标量零拷贝下发**，不引入任何 H2D。
2. **结构体内不含数组**，符合编码规则 `R4-TilingData禁止数组`。本设计的分核信息用 `useCoreNum`/`perCore` 两个标量表达（各核区间由 `blockIdx` 现算），而非 `rowStart[MAX_CORE]` 这类数组。
3. **workspace 从 handle 获取**（`GetEffectiveWorkspace(h)`），模板明文「禁止自行 `aclrtMalloc`」。本算子实际不需要 workspace，传 `nullptr` 即可。

> 这是相对既有 `blas/gemv/arch22/cgemv_host.cpp` 的**关键改进**：后者每次调用都 `aclrtMalloc` + `aclrtMemcpy` + `aclrtSynchronizeStream` + `aclrtFree` 三块 Device 内存（tiling / mask / 16MB workspace，其中 workspace 实际未被 kernel 使用）。对本任务 case 1 的 10.665 us 预算，这类固定开销是不可接受的。

#### 3.3.7 日志（强制规范）

| 级别 | 内容 |
|------|------|
| `OP_LOGE` | 参数校验失败（打印全部入参）、workspace/launch 失败 |
| `OP_LOGD` | TilingData 全字段 |
| `OP_LOGI` | 分派路径、blockDim、useCoreNum |

禁止 `printf` / `std::cout`。

### 3.4 Kernel 侧设计

#### 3.4.1 复数在 AIV 上的处理方式

complex64 在内存中是 `[re0, im0, re1, im1, ...]` 交织布局，而 AIV 向量指令按 fp32 逐元素操作。本设计采用**解交织（deinterleave）后用实数向量指令组合**的方式：

```
CopyIn 后：aUb (fp32 视图，长度 2*TB) = [re0, im0, re1, im1, ...]
解交织：   aRe = aUb 的偶数位；aIm = aUb 的奇数位
```

**实现方式已由仓内既有代码与官方文档双重确认，不再是待定项**：使用 `GatherMask`（底层 intrinsic 为 `vreducev2`）的内置固定 pattern。

- **pattern = 1**：取**偶数索引**元素（index 0, 2, 4, …）→ **实部**
- **pattern = 2**：取**奇数索引**元素（index 1, 3, 5, …）→ **虚部**
- 输出为**紧凑排列**，有效元素个数由 `rsvdCnt` 返回；A2 对 b32 的 pattern 张量类型为 `uint32_t`，固定模式 1–7 全支持。

> ⚠️ **措辞陷阱**：官方文档用的是「索引」（0-based）。pattern 1 取的是第 1、3、5… 个元素（偶数**索引**），pattern 2 取的是第 2、4、6… 个元素（奇数**索引**）。用"偶数位/奇数位"的口语描述极易写反，实现与检视时一律以 0-based 索引为准。

仓内 `blas/gemv/arch22/ascblasCgemv_utils.h` 的 `split_CVector()` 正是这一做法（`vreducev2(..., /*pattern*/1, ...)` → 实部，`pattern=2` → 虚部），**可直接复用其封装**：

```cpp
// 来自 blas/gemv/arch22/ascblasCgemv_utils.h（已验证可用于 arch22 complex64）
split_CVector(dst_r, dst_i, src, num_complex);   // src: 交织的 2*num_complex 个 float
```

**A2 的两个使用约束**（官方文档）：调用 `GatherMask` 前后的 mask 模式会被重置为 Normal（若此前为 Counter 模式需显式设回）；`dst`/`src0` 的 LocalTensor 起始地址须 32B 对齐。

解交织后，复数点积在 fp32 域展开为：

```
设 s = ±1（trans=C 时 s = -1，表示对 A 取共轭；N/T 时 s = +1）

dotRe = Σ ( aRe*xRe  −  s*aIm*xIm )
dotIm = Σ ( aRe*xIm  +  s*aIm*xRe )
```

向量指令序列（每 tile）：

```
Mul(t0, aRe, xRe);  Mul(t1, aIm, xIm);  (s>0) Sub(t0,t0,t1) : Add(t0,t0,t1)   → 规约 → dotRe
Mul(t0, aRe, xIm);  Mul(t1, aIm, xRe);  (s>0) Add(t0,t0,t1) : Sub(t0,t0,t1)   → 规约 → dotIm
```

即 **4 次 Mul + 2 次 Add/Sub + 2 次规约**产出一个复数输出元素，浮点运算量与标杆的「4 乘 2 加 / 元素」一致，无冗余计算。

**规约原语的选型（重要，不能想当然用 `ReduceSum`）**：

| 候选 | 结论 |
|------|------|
| `ReduceSum`（全量求和 → 1 标量） | **不采用**。官方文档明确：「该接口内部通过**软件仿真实现**，性能可能不及硬件指令实现的 `ReduceDataBlock` 和 `ReduceRepeat` 接口，**不推荐对性能敏感的场景使用**」，且需额外 `sharedTmpBuffer` |
| `ReduceRepeat`（每 repeat 归约 → 1 标量/repeat） | 可用，但 **float32 下 `mask ≤ 64`**（256B/repeat 限制），而本算子 `cnt` 最大 4095，无法一次覆盖 |
| **二叉树 `Add` 折半 + `ReduceDataBlock` 收尾** | **采用**。见下 |

采用方案：把长度 `cnt` 的乘积向量**折半相加**——`Add(t[0:h], t[0:h], t[h:2h])`，`h` 从 `cntPad/2` 逐级减半至 8；每级两个操作数在 UB 中的起始偏移均为 `h` 个 float = `4h` 字节，**只要 `h ≥ 8` 就天然满足 32B 对齐**。降到 8 个 float（恰好 1 个 DataBlock）后，用一次 `ReduceDataBlock`（每 DataBlock 归约出 1 个元素）收尾得到标量。

- 全程只用 `Add` + 一次 `ReduceDataBlock`，均为硬件指令，无软件仿真、无 `sharedTmpBuffer`；
- 共 `log2(cntPad) − 3` 次 `Add`，对 `cnt=321`（性能 case 5）约 6 次，开销可忽略；
- 全程操作数 32B 对齐，不触碰 §3.6 R1 的非对齐约束。

> ⚠️ **接口改名（CANN 9.1.0 起生效，直接影响本任务）**：`WholeReduceSum → ReduceRepeat`、`BlockReduceSum → ReduceDataBlock`、`PairReduceSum → ReducePairElem`。本任务目标版本为 **CANN 9.1.0**，故**必须使用新名**；若在 ≤9.0.0 环境编译需换回旧名（注意 `BlockReduceSum → ReduceDataBlock` 还调整了 `mask` 与 `repeatTime` 的参数顺序）。
>
> ⚠️ **`ReduceDataBlock` 在 A2 上的行为差异**：若某个 DataBlock 内所有元素都被 mask 排除，A2 **不写入目的操作数、保留其原值**（而非填 0）。因此收尾前必须显式初始化目的位置，或保证 mask 覆盖完整 DataBlock。
>
> 仓内 `blas/gemv/arch22/cgemv_do_trans_kernel.cpp` 使用的是仓内自有的 `cadd_v()` 行内归约助手，也可作为等价实现，实现阶段二选一并实测对比。

最后的标量收尾（每个输出元素一次，标量域完成）：

```
resRe = alphaRe*dotRe − alphaIm*dotIm
resIm = alphaRe*dotIm + alphaIm*dotRe
if   betaMode == ZERO :  y[k] = (resRe, resIm)                       // 不读原 y
elif betaMode == ONE  :  y[k] = (resRe + yRe, resIm + yIm)
else                  :  y[k] = (resRe + betaRe*yRe − betaIm*yIm,
                                 resIm + betaRe*yIm + betaIm*yRe)
```

**alpha 在归约之后施加一次**，而非在每个乘法上施加——既省运算量，也减少一次舍入。

#### 3.4.2 A 段的搬运（三个 trans 分支的唯一实质差异）

**trans = T / C（输出 `y[j]`，取第 j 列）**

第 j 列的 `len(j)` 个有效元素连续，一次 burst 搬完：

```cpp
// GM 起始元素下标（complex 计）
srcOff  = (int64_t)j * lda + max(0, ku - j);
copyLen = min(m-1, j+kl) - max(0, j-ku) + 1;       // = len(j)
DataCopyPad(aUb, aGm[srcOff], {1, copyLen * 8 /*B*/, 0, 0, 0}, padParams);
```

单次 `blockCount=1`、`blockLen = len(j)*8B` 的连续搬运，DMA 效率最优。

对应的 x 段同样连续（`incx==1` 时）：

```cpp
xOff = max(0, j - ku);                              // = iLo(j)
DataCopyPad(xUb, xGm[xOff * incx], {1, copyLen * 8, 0, 0, 0}, padParams);
```

**trans = N（输出 `y[i]`，取第 i 行）**

由 §1.2.4，第 i 行的有效元素在打包数组中等差分布，公差 `lda-1` 个元素（`(lda-1)*8` 字节）。**直觉方案是用一次 `blockLen = 8B`、`blockCount = cnt`、`srcStride = (lda-2)*8B` 的 stride 搬运把整行取进 UB——但该方案在 A2 上不成立，已废弃**：

> ❌ **废弃理由（A2 硬约束）**：`DataCopyPad` 的功能说明原文——「当每个搬运的数据块长度（`blockLen`）**非 32 字节对齐**时，**每一个数据块都需要填充数据至 32 字节对齐**」。而「Compact 模式」（紧凑不补齐）的产品支持为 **Ascend 950PR / 950DT，A2 不支持**（A2 只有不含 `mode` 模板参数的原型）。
>
> 后果：`blockLen = 8B` 的每个复数元素在 UB 中会被补齐占满 **32B**，落到 UB 的不是"连续的 `cnt` 个复数"，而是 `[re, im, ×6 个填充]` 的重复排布。这同时导致：①UB 占用膨胀 4 倍；②后续 `GatherMask` 解交织的 pattern 完全失配；③有效带宽利用率仅 25%。**原设计"落到 UB 后同样是连续的 `cnt` 个复数"的说法是错误的。**

**改用方案：列块连续搬入 + UB 内对角线 `Gather`**

第 i 行的元素落在打包数组的一条**对角线**上（`d = ku+i-j`，j 增 1 则 d 减 1）。因此：

1. **GM → UB（大粒度连续搬运）**：按列块 `[j0, j0+TJ)` 一次搬入 `bandW × TJ` 的二维块——
   `blockCount = TJ`、`blockLen = bandW*8 B`、`srcStride = (lda-bandW)*8 B`、`dstStride` 按 `bandWPad` 补齐（单位 32B）。
   `blockLen` 为 `bandW*8` 字节（如 case 4 的 257×8 = 2056 B），是**大块连续 burst**，DMA 效率高；紧凑 `lda = bandW` 时 `srcStride = 0`，退化为单次超大 burst。
2. **UB 内取对角线**：用 `Gather` + 字节偏移表抽出第 i 行。偏移表为等差数列，**公差 `(bandWPad-1)*8` 字节**，可用 `ArithProgression` 在 UB 内一次生成，无需 host 上传、无需额外 GM buffer。实部与虚部各做一次 `Gather`（虚部偏移表 = 实部 + 4 字节），直接产出**已解交织**的 `aRe` / `aIm`，省掉一次 `GatherMask`。

`Gather` 的关键事实（官方文档）：A2 **支持**（含 float）；`srcOffset` **单位为字节**（不是元素下标）；官方示例本身就是一个逆序 gather；**不支持源操作数与目的操作数使用同一块内存**；`repeatTime ∈ [0,255]`，每次迭代完成 8 个 DataBlock。

**备选方案**：`TransDataTo5HD` 把 `bandW × TJ` 块转置为行主序后直接连续取行。A2 支持 float，b32 的转置粒度为 16 个地址序列 × 每 DataBlock 8 个元素（单 repeat 512B）。相比 `Gather` 方案多一次转置开销，但省掉偏移表；二者实现阶段实测对比，接口封装一致。

> `blockCount` 上限为 **4095**（`uint16_t`），本方案的 `blockCount = TJ`（≤ 几十），远低于上限，不存在越界风险——这也是相对废弃方案（`blockCount = cnt`，全集最大 4095，紧贴上限）的一个额外好处。

**非连续步长（`TK_*_STRIDE`）**

`incx ≠ 1` 时 x 段改为 `blockCount = cnt, blockLen = 8B, srcStride = (|incx|-1)*8B` 的 stride 搬运。

`incx < 0` 需要额外处理：host 已把基址前移、步长取负，GM 地址随 k **递减**，而 `DataCopyPad` 的 `srcStride` 是无符号量，无法表达负向步进。等价变换如下：令 `k' = cnt-1-k`、`xStart = xBase + (cnt-1)*incx`（incx<0 时这是该段的**最低地址**），则 `x_k = xStart[k' * |incx|]`，即**从 xStart 以正步长 `|incx|` 搬运，得到的是逆序的 x 段**。

由于本设计的计算是**点积（求和与顺序无关）**，只需让 A 段与 x 段的元素一一对应即可，于是只需把其中一个在 UB 内翻转。实现方案：

- **主方案**：`Gather` + 预置**降序索引张量**（索引在 Init 阶段按最大 tile 长度生成一次、常驻 UB，每次调用零额外构造开销），对 x 段做一次 UB 内逆序；
- **备选方案**：逐元素 `DataCopyPad`（`cnt` 次 8B 拷贝）按逻辑序搬入。正确性无疑义，但**触犯编码规则 `R1-禁止逐元素操作`**（既有 arch22 cgemv 的 `incx≠1` 分支正因此踩雷），仅作为 `Gather` 不可用时的最后退路。

> 两种方案封装为同一个内联函数 `LoadVectorStrided(dst, gm, base, inc, cnt)`，上层无感。
>
> **注**：本设计早期版本曾写"UB 内 `Reverse` 接口"，经文档核查**该接口并不存在**（Ascend C kernel 侧无 Reverse/Flip/逆序接口；`aclnnFlip` 是 host 侧 aclnn 单算子，不适用），已更正为 `Gather` + 递减字节偏移表——官方 `Gather` 文档的调用示例本身就是一个逆序样例。偏移表**单位为字节**（float32 逆序为 `[(n-1)*4, …, 8, 4, 0]`，不是元素下标），且 `Gather` **不支持源与目的使用同一块内存**。见 §3.6 风险 R5。

`incy ≠ 1` 仅影响单个 y 元素的读写地址，按标量偏移计算即可，无额外成本。

> 全部性能用例均为 `incx = incy = 1`（生成器有断言强制），因此 `TK_*_STRIDE` 路径只承担精度用例（`TC_INC` 36 条步长全组合 + `TC_EX` 部分），功能正确性优先于极致性能。

#### 3.4.3 Kernel 流程图

```
┌──────────────────────────────────────────────────────────┐
│  cgbmv_kernel                                            │
│  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)          │
│  模板实例：CgbmvAIV<TRANS, IS_CONT>                       │
└───────────────────────────┬──────────────────────────────┘
                            ▼
      ┌───────────────────────────────────────────┐
      │ Init: blockIdx = GetBlockIdx()            │
      │  outStart = blockIdx * perCore            │
      │  outEnd   = min(outStart+perCore, outLen) │
      │  TPipe 分配 aQue/xQue(double) + 计算 buf   │
      └───────────────────┬───────────────────────┘
                          ▼
      ┌───────────────────────────────────────────┐
      │ alphaIsZero ?                             │
      │   yes → 仅做 y = beta*y，逐元素标量处理   │──→ CopyOut → 结束
      └───────────────────┬───────────────────────┘
                          ▼ no
      ┌───────────────────────────────────────────┐
      │ for k in [outStart, outEnd):   ← 输出元素  │
      │   dotRe = dotIm = 0                       │
      │   for t in [0, numTiles):     ← band 分块  │
      │     ┌───────── CopyIn ─────────┐          │
      │     │ TRANS=N : 列块连续搬+对角 │          │
      │     │           Gather 取行     │          │
      │     │ TRANS=T/C: A 列段 连续搬  │          │
      │     │ x 段（连续 / stride / 反转）│         │
      │     └──────────┬───────────────┘          │
      │     ┌───────── Compute ────────┐          │
      │     │ GatherMask 解交织 →aRe/aIm│         │
      │     │ GatherMask 解交织 →xRe/xIm│         │
      │     │ 4×Mul + 2×Add/Sub        │          │
      │     │  (CONJ=true → aIm 取反)   │          │
      │     │ 折半 Add + ReduceDataBlock│          │
      │     │  → 累加 dot               │          │
      │     └──────────┬───────────────┘          │
      │   标量收尾: ×alpha，按 betaMode 合并原 y   │
      │   ┌───────── CopyOut ─────────┐           │
      │   │ 写 yGm[k * incy]（1 个 complex）│       │
      │   └───────────────────────────┘           │
      └───────────────────────────────────────────┘
                          ▼
                 各核独立结束（无核间同步）
```

#### 3.4.4 与标杆算子流程的差异点

| 差异点 | 标杆（Netlib `cgbmv`） | 本设计 | 原因 |
|--------|------------------------|--------|------|
| 并行度 | 单线程双层标量循环 | AIV 多核按输出元素分块 | 利用多核降低延迟 |
| 归约方向 | trans=N 用列 AXPY（`y(i) += temp*A(k+i,j)`） | trans=N 改为**按行点积** | 列 AXPY 的输出区间随列滑动，UB 累加器起始地址无法保持 32B 对齐；按行点积可让每次 DMA 落到对齐的 UB 起点，且天然避免跨核写冲突 |
| 复数运算 | 语言级 complex 乘法 | 解交织 + fp32 向量 Mul/Add | AIV 无原生复数向量指令 |
| 共轭处理 | 运行时 `CONJG()` | 编译期模板实参 `CONJ`（T/C 共用入口） | 消除内层分支 |
| 带外区 | 循环边界裁剪 | host 预计算 `iLo/iHi`、kernel 按 `len` 搬运 | 不引用带外三角区，且搬运长度精确 |
| alpha 施加 | 先 `temp = alpha*x(jx)` 再乘加 | **归约后施加一次** | 省运算量、少一次舍入 |
| beta 缩放 | 独立的 y 缩放前置循环 | 与点积结果在同一次 CopyOut 中合并 | 避免对 y 的二次读写 |
| 快速返回 | `IF (M.EQ.0 .OR. N.EQ.0 ...) RETURN` | host 侧判定，不下发 kernel | 省掉一次 launch 开销 |

### 3.5 支持硬件

与任务书 §3.1 一致：

| 支持的芯片版本 | 是否支持 |
|----------------|---------|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（含 Atlas 800I A2） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（含 Atlas 800I A3） | √ |
| Ascend 950PR / 950DT | — （不在本任务范围，由既有 arch35 路径承担） |

SOC：`ascend910b*` / `ascend910_93*`，架构目录 `arch22`（DAV_2201）。CANN 9.1.0。
精度自验证设备：Atlas 800T A2 (910B3)；性能自验证设备：Atlas 800T A2 (910B3)。

### 3.6 关键风险、优化项与应对

| 编号 | 项 | 说明与应对 |
|------|-----|-----------|
| **R1** | UB 操作数必须 32B 对齐（**硬约束，非性能问题**） | 已核实：官方《通用说明和约束》规定 UB 操作数起始地址 32Byte 对齐，`Mul`/`Add`/`Sub`/`Axpy` 各页均指向该约束；文档把 `Local1[7]` 起算明确称为「**错误示例**」，并写明违反「可能导致未定义行为或数据异常」。带状结构的滑动窗口（`iLo(j)` 随 j 递增 1）天然产生非对齐偏移，本设计通过「每个输出元素独立搬入对齐 UB 起点」从根上规避。 |
| **R2** | ~~解交织 API 语义待定~~ → **已关闭** | 已双重确认：官方文档给出 `GatherMask` 固定 pattern 1 = 偶数索引、2 = 奇数索引、输出紧凑；仓内 `blas/gemv/arch22/ascblasCgemv_utils.h` 的 `split_CVector()` 即为该用法的可用实现，直接复用。 |
| **R3** | ~~trans=N 的 8B stride DMA 效率~~ → **方案已更换** | 原方案因 A2 不支持 `DataCopyPad` Compact 模式而**不成立**（每个 8B 块在 UB 内补齐到 32B），已改为「列块连续搬入 + UB 内对角线 `Gather`」（§3.4.2）。剩余风险转为 `Gather` 的指令开销，实现后用 msOpProf 核对 case 4（m=n=2048, bandW=257, 达标 30.640 us）。 |
| **R4** | 小 shape 的 launch 开销占比 | case 1（256×256）达标仅 10.665 us，固定开销占比高。应对：**按值传递 TilingData**（仓方 simd-membase 模板原文：「host 侧以 const 引用传入 kernel_do，无 GM 设备内存分配，无同步」），alpha/beta 作为 4 个 float 字段随 tiling 下发；workspace 从 handle 获取（模板：「禁止自行 aclrtMalloc」）；无核间同步；小 shape 强制单核。**既有 `blas/gemv/arch22/cgemv_host.cpp` 每次调用 malloc+memcpy+sync+free 三块 Device 内存，是本任务必须避开的反例。** |
| **R5** | Ascend C **无** `Reverse` 接口 | 已核实证伪：kernel 侧不存在 Reverse/Flip/逆序接口（`aclnnFlip` 是 host 侧 aclnn 单算子，不适用）。负步长改用 `Gather` + **递减字节偏移表**（官方 `Gather` 示例本身即逆序），A2 支持 `Gather`/`Gatherb`，**不支持 `Scatter`**。 |
| **R6** | 归约接口在 CANN 9.1.0 改名 | `WholeReduceSum → ReduceRepeat`、`BlockReduceSum → ReduceDataBlock`、`PairReduceSum → ReducePairElem`。本任务目标为 **CANN 9.1.0，必须用新名**。`ReduceSum`（全量）为软件仿真实现、官方不推荐性能敏感场景，本设计改用「折半 `Add` + `ReduceDataBlock` 收尾」（§3.4.1）。 |
| **R7** | arch22 测试工程形态与任务要求不一致 | 仓内 `test/gemv/cgemv/CMakeLists.txt` 为 `if("arch35" IN_LIST SOC_ARCH_DIRS) ops_blas_add_gtest_tests(...) else() ops_blas_add_tests(...)`，即 **arch35 走 GTest+CSV，arch22 走独立 `main()` 样例**（`test/gemv/cgemv/arch22/cgemv_test.cpp` 自带 main、自带 golden、`EPSILON=1e-3`，不读 CSV）。而本任务要求用 1200 条 CSV 驱动。应对见 §5.4。 |
| **R8** | golden 必须是 Netlib，**不能是 OpenBLAS** | 仓内 `test/gemv/cgemv/README.md` 原文警告：两者复数求和次序不同，「同一份算子在 OpenBLAS 参考下曾出现 23 条 max-abs 失败，换回 Netlib 后同一集合全部通过」；且「很多发行版的 `libblas.so`（Netlib）**不导出** `cblas_*` 符号，系统里唯一提供 `cblas_cgemv` 的往往是 OpenBLAS，直接链接会**静默**用错参考实现」。本算子为复数规约，同等敏感。 |
| **O1** | 输出维度批量化 | `bandW` 小时（case 1 仅 33）单输出元素的 DMA 仅 264B，处于小包区。优化：一次 CopyIn 覆盖 `TJ` 个连续输出元素（T/C 分支即 `bandW×TJ` 二维连续块；紧凑 `lda=bandW` 时退化为单次大 burst），归约用 `ReduceRepeat` 批量产出（**注意 float32 下 `mask ≤ 64`**，故仅 `bandW ≤ 64` 可一次 repeat 覆盖一个输出，否则仍需折半 Add）。**性能用例全部为紧凑 lda，该优化收益最直接。** |
| **O2** | trans=N 的转置备选 | 用 `TransDataTo5HD` 将 `bandW×TJ` 块转置为行主序后直接连续取行，替代 UB 内 `Gather`。A2 支持 float，b32 粒度为 16 地址序列 × 每 DataBlock 8 元素。与 `Gather` 方案实测对比择优。 |
| **O3** | 按 `Σlen` 均分负载 | `bandW` 接近 `outLen` 时首尾核任务量约为中间核一半。改为按各输出元素的 `len` 前缀和均分区间。影响 `TC_BW` 类与 `pf_bw_*` 类用例。 |

#### R3 量化分析：trans=N 的访存局部性（支撑列块方案的核数/分块取值）

> **本小节的前提已随 §3.4.2 的方案更换而变化**：原「8B stride 逐元素取行」方案已因 A2 不支持 Compact 模式而废弃，改为「列块连续搬入 + UB 内对角线 Gather」。下面的 Cache 局部性分析**仍然成立且仍然重要**，因为它决定了列块方案中「同一列块被多少个输出行复用」，进而决定分核粒度。

CANN 文档明确：「**所有通过搬运单元读写 GM 的数据都缺省被缓存在 L2 Cache**」，且「核外 L2Cache 以 Cache Line 为单位加载数据，根据硬件规格不同，Cache Line 大小不同（128/256/512Byte 等）」。

关键在于 trans=N 的访存模式与 Cache Line 的对齐关系。由 §1.2.4，第 i 行第 j 个有效元素的字节偏移为：

```
byteOff(i, j) = [ (ku + i) + j * (lda - 1) ] * 8
```

固定 j、令 i 递增 1，偏移只前进 **8 字节**。也就是说：**对同一个 j，连续 16 行（128B Cache Line）/ 32 行（256B）/ 64 行（512B）的元素落在同一条 Cache Line 上**。本设计按连续行区间分核，每个核处理 `perCore` 个连续行，因此：

- 一条 Cache Line 被同一个核复用 16～64 次，**同一列块的数据被本核的多个输出行反复命中**，实际 GM 流量 ≈ A 的实际大小；
- 每个核在「推进一个 j」时的活跃 Cache Line 数约为 `bandW` 条。以最吃紧的 case 4（bandW=257）为例，按 128B 行计为 `257 × 128B ≈ 33 KB`/核；即便 48 核全开也仅约 1.5 MB 的活跃工作集，远小于 L2 容量量级，**不会发生容量性替换**。

**对列块方案的直接推论**：按连续行区间分核时，一个列块一旦搬进 UB，可服务本核 `TJ` 范围内的全部输出行，GM 侧不重复读；因此 trans=N 的真实风险不在带宽，而在 **UB 内 `Gather` 的指令开销**——由 O2（`TransDataTo5HD` 转置）作为备选对冲。

> 注：L2 容量的具体数值在 CANN 文档中**未对 A2/910B3 给出**（文档里出现的 192MB 是「假设，AI 处理器的 L2 Cache 大小为 192MB」的教学假设，不可当规格引用）。上面的结论只依赖「活跃工作集 ~1.5MB 属于小量级」这一弱假设，不依赖具体容量数字。必要时可用 `GetCoreMemSize(CoreMemType::L2, size)` 运行时查询。

---

## 四、特性交叉分析

| 特性 | 说明 |
|------|------|
| **多核并行** | 按输出元素分块，各核写 y 的不相交区间；**无跨核归约、无 workspace、无 `CrossCoreSetFlag/WaitFlag`** |
| **workspace** | 不申请。tiling 在 host 栈计算，alpha/beta 随 tiling 下发 |
| **原地语义** | y 原地读写。`betaMode != ZERO` 时先读原 y 再写回，读写同一元素、无跨元素依赖，故与分核顺序无关 |
| **内存重叠** | 任务书 §2.5 规定 y 与 A/x 不允许重叠，不做检测（与 cuBLAS 一致，重叠为未定义行为） |
| **带状性** | 仅引用 `max(0,j-ku) ≤ i ≤ min(m-1,j+kl)` 范围内元素；左上 `ku×ku`、右下 `kl×kl` 三角区与 `lda > kl+ku+1` 的 padding 区**不读取**，故这些位置的 Inf/NaN 不会污染结果 |
| **trans 分支** | N 按行点积、T/C 按列点积；C 在编译期把 A 虚部符号取反，与 T 共用模板 |
| **负步长** | host 侧归一化为 `(base, 带符号 inc)`；kernel 内 x 段用"正步长 stride 搬运 + UB 内 `Gather` 逆序"（Ascend C 无 `Reverse` 接口，见 R5） |
| **特殊值** | Inf/NaN 按 IEEE754 自然传播，不钳位。`alpha=0` 不读 A/x、`beta=0` 不读 y，保证 `TC_ED_233/234`、`TC_FL_*` 的 Inf/NaN 用例与 cblas 一致 |
| **精度** | 归约为 fp32 累加，顺序与 cblas 不同（任务书 §2.5 明确不要求确定性）。alpha 在归约后施加，比标杆少一次舍入 |
| **并发** | kernel 只读 A/x、写 y 的互斥区间，无全局可变状态，可与其他算子在不同 stream 上安全并发 |
| **多代际** | 实现位于 `blas/gbmv/arch22/`，与既有 `blas/gbmv/arch35/`（Sgbmv）路径完全隔离，互不影响 |

---

## 五、可维可测分析

### 5.1 精度标准

对齐任务书 §3.2 与[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64 | 2⁻¹³ (1.22e-4) | 2⁻¹³ (1.22e-4) | 0.99 | 1e-2 或 32×ULP（规约场景可放宽至 2ULP） |

- golden 由 cblas（Netlib BLAS）`cblas_cgbmv` 单标杆生成；
- 比对对象为输出向量 y 的**参与计算维度全量**（trans=N 取 m 个、T/C 取 n 个），复数按实部/虚部分别按 FLOAT32 判定；
- 逐元素通过条件 `|actual − golden| ≤ atol + rtol × |golden|`；
- 整体通过条件 `matched_ratio ≥ 0.99` **且** `max_abs_error ≤ limit`；
- 精度模式：Level-2 BLAS 按仓规范取 `PrecisionMode::MERE_MARE`，CSV 提供 `mere_threshold = 0.0001220703125`（2⁻¹³）、`mare_multiplier = 10` 两列。

### 5.2 性能标准

测试设备 Atlas 800T A2 (910B3)，msprof 采集 kernel `Task Duration(us)`，5 次 warmup 后有效采样 ≥10 次取平均。达标口径：`NPU 平均单次 kernel 耗时 ≤ gpu_ms / 0.8`。

| case | m | n | kl | ku | bandW | trans | alpha/beta | A 字节 | 达标耗时(us) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 256 | 16 | 16 | 33 | N | 1/0 | 66 KB | 10.665 |
| 2 | 512 | 512 | 32 | 32 | 65 | N | 1/1 | 260 KB | 17.328 |
| 3 | 1024 | 1024 | 64 | 64 | 129 | T | 1/0 | 1.01 MB | 25.875 |
| 4 | 2048 | 2048 | 128 | 128 | 257 | N | 1/0 | 4.02 MB | 30.640 |
| 5 | 4096 | 4096 | 64 | 256 | 321 | C | 1/0 | 10.03 MB | 52.487 |

**访存量**：本设计对每个输出元素各搬一段 A 和一段等长的 x，故**名义 GM 流量约为 `2 × A 字节`**（x 段在相邻输出元素间高度重叠，经 L2 后实际流量更接近 `1 × A`）：

| case | A 字节 | 名义流量(2×A) | 达标耗时(us) |
|---|---|---|---|
| 1 | 66 KB | 132 KB | 10.665 |
| 2 | 260 KB | 520 KB | 17.328 |
| 3 | 1.01 MB | 2.02 MB | 25.875 |
| 4 | 4.02 MB | 8.04 MB | 30.640 |
| 5 | 10.03 MB | 20.06 MB | 52.487 |

**可行性判断**：算子为纯访存密集型（每个 complex64 元素仅 4 乘 2 加），性能由访存而非算力决定。

> **关于带宽基准的说明**：Atlas 800T A2 / Ascend 910B3 的 HBM 理论带宽，**CANN 官方文档中未给出**——文档一律用脱敏表述（如「某款 AI 处理器的 GM 峰值带宽约为 1.8TB/s」、「假设…GM 的带宽约为 1.6TB/s」），均未绑定具体型号。因此本节**不以某个带宽数字作为达标论证**，仅作量级判断：即便按文档出现过的最低档 0.8 TB/s 估算，case 5 的 20 MB 也在 25 us 量级，相对 52.487 us 预算仍有裕量。精确带宽须用 msOpProf 实测（`Memory.csv` 的 `aiv_gm_to_ub_bw`）后回填到自测报告。

真正的风险分布是：

- **case 1 / 2**（数据量 ≤520 KB）：**瓶颈在 kernel launch 与 tiling 的固定开销，不在带宽**。这是 §3.2.1 选择"无跨核同步、无 workspace、标量随 tiling 下发"架构以及 R4 各项措施的直接动因；
- **case 4**（trans=N，8 MB，预算 30.640 us）：对应 R3，经 §3.6 的 Cache Line 量化分析后判断带宽不是瓶颈，剩余风险是小 burst 的 DMA 发射开销，由 O1 应对；
- **case 3 / 5**：流量与预算之比最宽松，风险最低。

### 5.3 自验用例

随任务提供 `test_cases/cgbmv_test.csv` 共 **1200 条**（固定种子 20260823，可复现）：

| 类别 | 前缀 | 条数 | 覆盖点 |
|------|------|------|--------|
| L0 基础 | `TC_L0` | 6 | trans 全枚举 × 小尺寸 |
| L1 尺寸 | `TC_SQ` | 69 | 23 种尺寸（1→2048）× 3 枚举 |
| L2 标量 | `TC_AB` | 24 | 8 组 alpha/beta 特殊值 × 3 枚举 |
| L3 非方阵 | `TC_RC` | 36 | fat / thin × 3 枚举 |
| L4 前导维 | `TC_LD` | 9 | `lda = kl+ku+1+4` padding |
| L4b 带宽 | `TC_BW` | 13 | kl/ku = 0 / 上界 / 全带 / 仅上带 / 仅下带 |
| L4c 步长 | `TC_INC` | 36 | incx×incy ∈ ±1/±2/±3 全组合 |
| L5 填充 | `TC_FL` | 12 | 全零 / 交替 / 极端值 / Inf / NaN（A 与 x 各一遍） |
| L5b 覆盖 | `TC_CV` | 24 | 中等尺寸 × 3 枚举 |
| L6 边界负向 | `TC_ED` | 19 | 零维 / beta=0 不读 y / alpha=0 / 空指针 / 非法枚举 / 非法 lda / 负维度 / 零步长 / kl,ku 负值 |
| EX 扩展 | `TC_EX` | 752 | 尺寸×trans×带宽×标量×步长×padding 确定性采样 |
| PF 性能 | `TC_PF` | 200 | 5 条任务书典型 case（逐参数精确匹配）+ 小尺寸 + 方阵对数扫描 + 带宽扫描 + 宽/窄网格 + 大尺寸混合 |

**本设计对各负向用例的返回码映射**见 §3.3.1 校验表，其中 `TC_ED_235 (null_handle)` → `HANDLE_IS_NULLPTR`，其余负向用例均 → `INVALID_VALUE`。

### 5.4 测试工程交付件

遵循 `repo-test-develop` 的六文件 ST 结构，落在 `test/gbmv/cgbmv/`：

```
test/gbmv/cgbmv/
├── cgbmv_param.h                -- 参数结构体（字段按 API 参数顺序，数组参数用 BlasFillMode）
├── cgbmv_golden.h               -- CPU golden，调用 cblas_cgbmv，签名与 API 一致
├── CMakeLists.txt               -- 测试注册（统一 main 来自 test/frame/test_main.cpp）
└── arch22/
    ├── cgbmv_npu_wrapper.h      -- NPU wrapper，封装全部 ACL 操作
    ├── cgbmv_test.cpp           -- GTest 入口：TEST_F（空 handle）+ TEST_P（CSV 驱动）
    └── cgbmv_test.csv           -- CSV 用例表
```

**⚠️ 一个必须在编码前拍板的工程问题（风险 R7）**：仓内 **arch22 的测试先例并不是 GTest+CSV**。`test/gemv/cgemv/CMakeLists.txt` 的实际内容是

```cmake
if("arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} cgemv_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
```

即 arch35 走 GTest+CSV，**arch22 走 `ops_blas_add_tests`（带独立 `main()` 的样例程序）**——`test/gemv/cgemv/arch22/cgemv_test.cpp` 自带 `main`、自带 `ComputeGoldenCgemvN/T`、`EPSILON = 1e-3f`，完全不读 CSV，也不使用 `cgemv_param.h` / `cgemv_golden.h`。

而本任务要求用随任务提供的 **1200 条 CSV** 驱动自验。两者冲突，处理方案：

1. **首选**：为 arch22 启用 GTest+CSV（`ops_blas_add_gtest_tests`）。`test/frame/` 框架本身与芯片无关（`csv_loader.h` / `blas_test.h` / `fill.h` / `verify.h` 均无 arch 依赖），cgemv 的 `if(arch35)` 是该算子自身的选择而非仓级限制。**需在编码第一步用最小用例验证 `ops_blas_add_gtest_tests` 在 arch22 下能正常注册与运行**——这是本设计唯一需要"先试一下才能确定"的工程点。
2. **兜底**：若框架确实只支持 arch35 的 GTest 路径，则在 arch22 下提供 `cgbmv_test.cpp` 自带 main + 自实现 CSV 读取，复用 `cgbmv_golden.h`，保持 1200 条用例覆盖不打折。

另外，`test/CMakeLists.txt` 的探测逻辑要求文件**严格命名** `cgbmv_test.cpp`，且 `ops_blas_has_op_sources_for_soc(cgbmv ...)` 要求 `blas/gbmv/arch22/` 下确有 cgbmv 源码，二者缺一测试会被静默 skip。

- A 的带状填充使用框架的 `makeBlasBanded`，CSV 中 `RANDOM` 一律显式带值域（`RANDOM_NORM_5_5` = [−5,5]），符合仓内 HIGH 级检视项；
- **golden 必须链接 Netlib BLAS，不能是 OpenBLAS**（风险 R8）。构建前须确认 `cblas_cgbmv` 实际解析到 Netlib：很多发行版的 `libblas.so`(Netlib) 不导出 `cblas_*`，系统里唯一提供该符号的往往是 OpenBLAS，**链接会静默成功但参考实现用错**。仓内 `test/gemv/cgemv/README.md` 记录了同类事故：同一算子在 OpenBLAS 参考下 23 条 max-abs 失败，换回 Netlib 后全部通过；
- 构建与执行：`bash build.sh --ops=cgbmv --run [--device=N]`，通过标准为 `[  PASSED  ] N tests.` 且 Summary `Failed: 0`；
- 性能用例（`TC_PF` 前缀）仅用于本地性能自验，**PR 前从 ST 交付目录移除**，避免混入正式交付（仓规范明确要求）；
- kernel 耗时采集：
  ```bash
  msprof op --application="./build/test/gbmv/cgbmv/cgbmv_test --gtest_filter=*TC_PF_0001*" --output=./prof_cgbmv
  ```
  取 `OPPROF_*/OpBasicInfo.csv` 中对应 kernel 的 `Task Duration(us)` 求平均。

### 5.5 设计自验（编码前，评审阶段完成）

本设计在动手编码前，已用独立脚本对**设计本身**做了两项实证，避免把错误的索引数学或校验顺序带进实现：

**（1）索引数学与三分支分解的数值验证**

构造 21 组覆盖性形状（含 `1×1`、`2×3`、仅上带 `kl=0,ku=127`、仅下带 `kl=127,ku=0`、满带 `kl=m-1,ku=n-1`、极扁/极瘦 `16×1024` / `1024×16`、`lda` padding 0/2/3/4/8 等），**把带外三角区与 lda padding 区填成 NaN 毒值**，逐项验证：

| 验证项 | 结论 |
|--------|------|
| 打包索引公式 `A(i,j) = AB[(ku+i-j) + j*lda]` | 通过 |
| 列视角：`iLo/iHi/len/d0` 正确，且该列 `len` 个元素**在内存中连续** | 通过 |
| 行视角：第 i 行有效元素**等差、公差 `lda-1`**，且下标恒落在 `lda×n` 数组内 | 通过 |
| 三个 trans 分支（N 按行点积 / T 按列点积 / C 按列点积取共轭）对比 dense 参考 | 通过 |
| 步长归一化 `base = (inc>0 ? 0 : (len-1)*|inc|)`，逻辑第 k 元素 `base[k*inc]` | 通过 |
| `alpha=0` 不读 A/x、`beta=0` 不读 y（y 填 NaN 时结果不得含 NaN） | 通过 |

共 **3024 次**端到端比对（21 形状 × 3 枚举 × 8 组 alpha/beta × 6 组步长组合，含 `incx/incy` 的 ±1/±2/±3 与正负混合），相对误差判据 `1e-12`（远严于 float32），**全部通过**。毒值设计保证了任何越界引用或 beta 短路错误都会立即以 NaN 形式暴露。

**（2）参数校验表的可执行化回归**

把 §3.3.1 的校验表实现为可执行判定函数，对随任务提供的 **1200 条 CSV 用例**逐条判定并与 `expect_result` 比对：**0 条不一致**。这同时验证了 §1.2.6 中"把 kl/ku/lda 与 A/x/y 空指针校验放到零维快速返回之后"这一设计决策与用例集完全吻合。

该统计同时产出了若干驱动设计取值的事实（已回填至 §3.3.4、§3.4.2、§5.2）：

| 统计项 | 值 | 对设计的影响 |
|--------|-----|-------------|
| `bandW` 最大值 | 4095（仅 4 条用例 > 2048） | `TB=2048` 时 `numTiles ≤ 2` |
| trans=N 单行最大元素数（= `blockCount`） | 4095 | 经 `TB` 分块后 ≤ 2048 |
| `lda > kl+ku+1`（padding）用例 | 159 条 | padding 路径必须正确 |
| 含负步长 / 非单位步长用例 | 271 / 370 条 | **STRIDE 路径占精度用例 37%，不是边缘路径** |
| `beta = 0` 用例 | 696 条（69%） | beta=0 快速路径是**主路径**，非优化项 |
| `alpha = 0` 用例 | 195 条 | alpha=0 快速路径同样高频 |
| 单用例 A 最大规模 | 128 MB | 测试工程需注意 Device 内存预算 |

> 两项脚本随测试工程一并提交，置于 `test/gbmv/cgbmv/` 的开发辅助目录（不计入 ST 交付件，PR 前移除）。

### 5.6 兼容性分析

| 兼容性项 | 说明 |
|----------|------|
| 接口兼容 | 新增算子，无存量接口变更。`aclblasCgbmv` 声明加入 `include/cann_ops_blas.h`，供其他产品线共用，**不得**建私有平行 API |
| 签名对齐 | 参数序列与 `cublasCgbmv` 一一对应，无需映射说明 |
| 数据格式 | complex64 内存布局与仓内既有复数算子（`aclblasCgemv` / `aclblasCdotc` / `aclblasCdotu`）一致 |
| 多代际 | arch22 专用实现位于 `blas/gbmv/arch22/`，不触碰既有 `arch35` 路径；`aclblasSgbmv` 的产品支持情况不受影响 |
| 文档 | `blas/gbmv/README.md` 追加 `aclblasCgbmv` 章节，产品支持表标注 Atlas A2/A3 系列产品（含 Atlas 800I A2/A3）：支持 |
| 语义对齐 | 零维 no-op、负步长、alpha/beta 特殊值、非法参数返回码均与 cuBLAS / Netlib 对齐 |

---

## 六、交付件清单

| 序号 | 交付件 | 位置 |
|------|--------|------|
| 1 | 算子设计文档 | 本文档，以 PR 形式提交至 `cann-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/<task>/<gitcode-id>/docs/design.md` |
| 2 | 算子实现 | `ops-blas`：`include/cann_ops_blas.h`（声明）、`blas/gbmv/arch22/`（4 个文件）、`blas/gbmv/README.md`（追加章节） |
| 3 | 测试工程 | `ops-blas`：`test/gbmv/cgbmv/`（6 个文件，含 README 说明复现步骤） |
| 4 | 自测报告 | `task_submission/`：自验证步骤说明、精度/性能/内存报告 xlsx 与日志 log |
| 5 | 代码仓 | 个人仓分支 + 邀请 `Ascend-CANN` 为开发者 |
