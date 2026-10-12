# aclblasCgbmv 算子设计文档（Atlas A2/A3，arch22）

> 对标 cuBLAS `cublasCgbmv`；任务：9 月社区任务编号 70「9月社区任务-aclblasCgbmv算子开发(A2/A3)」。
> 本文四个 `（required）` 章节与官方 `resources/design_template.md` 的一级标题逐字对齐。

# 需求背景（required）

## 需求来源

**9月社区任务-aclblasCgbmv算子开发(A2/A3)**：要求参考 cuBLAS `cublasCgbmv`，在 Atlas A2/A3 系列产品上
以 Ascend C 开发单精度复数（complex64）一般带状矩阵向量乘算子 `aclblasCgbmv`，完成算子设计、开发、测试全流程。

任务书原文给出的权威入口（以下链接均逐字取自任务书，非凭记忆拼接）：

| 用途 | 链接 |
| --- | --- |
| 社区任务列表（设计文档 PR 目标仓） | <https://gitcode.com/cann/cann-competitions/tree/master/04_tasks/01_community-task-2026/tasklist> |
| 设计文档模板 | <https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md> |
| 【社区任务】流程及注意事项 | <https://gitcode.com/org/cann/discussions/39> |
| 生态算子开源精度标准 | <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md> |
| 目标代码仓 | <https://gitcode.com/cann/ops-blas> |
| 免费开发环境（HiDevLab webIDE） | <https://hidevlab.huawei.com/online-develop-intro?from=hiascend> |

- 实现目录 `blas/gbmv/arch22/`，测试目录 `test/gbmv/cgbmv/arch22/`（任务书 §5 原文要合入的两个目录）
- 交付物（任务书 §4）：算子设计文档、自测用例与测试代码、自测报告（精度/性能/内存）、待验收代码地址
- 验收设备：性能基准为 Atlas 800T A2 (910B3)（任务书 §3.1、§3.3）；
  Atlas 800I A2 与 Atlas 800I A3 均需完成验收（任务书 **§7 第 4 条**原文）
- 任务编号：`70`。依据为社区任务仓 `cann/cann-ops-competitions` 内
  `04_tasks/01_community-task-2026/docs/README.md` 第 281 行原文
  `| 70 | 9月社区任务-aclblasCgbmv算子开发(A2/A3) |`；设计文档提交目录为该任务在
  `04_tasks/01_community-task-2026/tasklist/09-70-aclblasCgbmv-A2A3/` 下的个人目录
  `YO_Y_O_/docs/design.md`（该仓 `tasklist/` 共 455 个任务目录，其中 401 个采用「编号-算子名」形态）

> 本文不重复罗列任务平台的截止时间与激励金额：本地任务书正文只出现「9月社区任务」字样，未含日期与金额的原文，
> 相关数字应以平台任务详情页为准，本设计不将其作为任何技术结论的前提。

## 背景介绍

一般带状矩阵向量乘（GBMV）是 BLAS-2 核心例程之一，用于带宽远小于阶数的稀疏结构矩阵与向量做乘加，
典型场景为差分方程求解、带状特征值问题与有限差分核函数。cuBLAS 以 `cublasCgbmv` 覆盖单精度复数场景，
ops-blas 需对齐该能力以支撑上层生态迁移。

**仓内现状（已逐条实测核对，非推测）**：

| 项 | 现状 | 取证方式 |
| --- | --- | --- |
| `blas/gbmv/` | 已存在，但只有 `arch35/sgbmv_{host,kernel}.cpp` + `sgbmv_tiling_data.h`（单精度**实数**、950 系） | 目录列举 |
| `blas/gbmv/arch22/` | **不存在**，本任务新建 | 目录列举 |
| `aclblasCgbmv` 接口声明 | **不存在**；`include/cann_ops_blas.h:358` 仅有 `aclblasSgbmv` | grep |
| 全仓 `cgbmv`/`zgbmv` 调用点 | 零命中 | grep |
| 同架构复数参考实现 | `blas/gemv/arch22/cgemv_*`、`blas/scal/arch22/cscal_*`、`blas/axpy/arch22/caxpy_*` | 目录列举 |
| arch22 复数 gather 先例 | `extensions/complexmatdot/arch22/`（唯一） | 目录列举 |
| `blas/gbmv/README.md` | 存在，仅列 `aclblasSgbmv`，产品支持表标注 A2/A3「不支持」 | 文件读取 |

# 需求分析（required）

## 算子功能

计算 `y = alpha * op(A) * x + beta * y`，其中：

- `A` 为 `m×n` 一般带状矩阵，含 `kl` 条下对角带与 `ku` 条上对角带，**以列主序打包存放在 `lda×n` 数组**，要求 `lda ≥ kl+ku+1`；
- `op(A)` 由 `trans` 决定：`ACLBLAS_OP_N` → `A`（`m×n`）；`ACLBLAS_OP_T` → `Aᵀ`；`ACLBLAS_OP_C` → `Aᴴ`（后两者为 `n×m`）；
- **`m`/`n` 恒表示 `A` 的行数/列数，不随 `trans` 翻转**；
- `x`、`y` 为 COMPLEX64 向量，按 `incx`/`incy` 步长存取，**步长可为负**（cuBLAS 语义）；
- `x` 长度为 `n`（N）或 `m`（T/C）；`y` 长度为 `m`（N）或 `n`（T/C）；
- `m == 0` 或 `n == 0` 为合法 no-op，返回成功且不执行计算。

计算公式与索引定义见「详细设计 → 算子分析」。

## 公开接口与范围

在 `include/cann_ops_blas.h` 新增任务书 §2.4 规定的 13 参数接口（声明风格与相邻 `aclblasSgbmv` 一致，
仓内算子声明均为裸 `aclblasStatus_t`，无导出宏，故不添加导出宏）：

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku,
    const aclblasComplex *alpha, const aclblasComplex *A, int lda,
    const aclblasComplex *x, int incx, const aclblasComplex *beta,
    aclblasComplex *y, int incy);
```

能力范围：

| 维度 | 支持 | 不支持 / 不要求 |
| --- | --- | --- |
| 数据类型 | COMPLEX64（`aclblasComplex`，实部/虚部各 float） | COMPLEX128、实数类型（实数已有 `aclblasSgbmv`） |
| 布局 | 列主序带状打包 | 行主序、CSR/COO 稀疏格式 |
| 步长 | `incx`/`incy` 任意非零整数（含负值） | `inc == 0` |
| 广播 / Tensor 输入 | 否 | 超出 `lda/incx/incy` 语义的非连续 Tensor |
| Shape | `m/n/kl/ku` 为运行时入参，由 host tiling 支持 | 不要求 dynamic shape 图模式 |
| 确定性 | 不要求确定性累加顺序 | — |
| 内存别名 | — | `y` 与 `A`/`x` 不允许重叠；不返回视图 |
| 异步语义 | 依赖 `aclblasSetStream` 绑定 stream | — |

# 详细设计（required）

## 算子分析

### 数学公式与打包索引

1-based 定义：`A(i,j) = Apack(ku+1+i-j, j)`，主对角线位于打包数组第 `ku+1` 行。
**该式不来自任务书文本**（任务书只规定「列主序打包于 `lda×n`、`lda ≥ kl+ku+1`、`ku` 条上对角、
`kl` 条下对角，按 cuBLAS `cublasCgbmv` 对齐」，见 §2.1 与 §7 第 3 条），而是 cuBLAS/LAPACK 惯例；
其正确性已由 Netlib 系 `cgbmv` 实测确认（见「精度标准与验证」）。
仅当 `max(1, j-ku) ≤ i ≤ min(m, j+kl)` 时元素在带内，左上 `ku×ku` 与右下 `kl×kl` 三角区**不引用**。

0-based 落地形式（kernel 实际使用）：

```
aIdx   = (ku + i - j) + lda * j                    // 列主序，列跨度为 lda
带内条件：max(0, j-ku) ≤ i ≤ min(m-1, j+kl)
```

`lda > kl+ku+1` 时多出的行是 padding，同样不引用。带外元素**不得影响结果**，本设计以对抗用例验证该性质。

### trans 对带结构的影响（最高风险点）

`op(A)` 的上下对角带**互换**：`A` 有 `kl` 条下带、`ku` 条上带，则 `Aᵀ`/`Aᴴ` 有 `ku` 条下带、`kl` 条上带。
因此转置分支的裁剪边界必须以 `ku` 为下界、以 `kl` 为上界：

```
trans = N  ：固定输出行 i，列范围 j ∈ [max(0, i-kl), min(n, i+ku+1))
trans = T/C：固定输出列 j，行范围 i ∈ [max(0, j-ku), min(m, j+kl+1))
```

`ACLBLAS_OP_C` 的共轭**施加于 `A` 的元素**（`Aᴴ(j,i) = conj(A(i,j))`），不作用于 `x`，也不作用于 `beta*y`。

### 负步长语义

逻辑第 `t` 个元素位于物理位 `i0 + t*inc`，其中 `i0 = (1-len)*inc`；`inc < 0` 时即
`x[t] = X[(len-1-t) * |inc|]`。所需物理存储长度为 `1 + (len-1)*|inc|`。`y` 的写回使用同一映射，
故不同输出位置互不重叠。

### alpha / beta 短路语义

沿用仓内既有模板写法，规避 `0 × NaN = NaN` 污染：

```
alphaTerm = (alpha == 0) ? 0 : alpha * acc
y[yIdx]   = (beta == 0) ? alphaTerm : alphaTerm + beta * y[yIdx]
```

即 `beta == 0` 时不读原 `y`（`y` 可为未初始化或空指针），`alpha == 0` 时结果不受 `A`/`x` 中 NaN/Inf 影响。

### 已验证性与验证方法

本节四条语义（打包索引式、`trans` 上下带互换、共轭施加对象、负步长偏移）**不依赖自洽推断**，
已用五条相互独立的实现路径交叉确认，其中含一个外部权威实现。方法与数据见「可维可测分析 → 精度标准与验证」。

设计方法论说明（重要）：最初仅用「两条不同结构的本机实现互校」，但二者共用同一套索引公式，
属**共享假设盲区**——若公式本身与 cuBLAS 定义有系统性偏差，互校仍会报零缺陷。因此引入了第三方权威实现
（Netlib 系 BLAS `cgbmv`）作为外部锚点，把该风险关闭。此判断路径保留在文档中，供后续同类复数算子复用。

## 算子实现

### 总体分期

| 阶段 | 目标 | 内容 |
| --- | --- | --- |
| v1（点亮） | 首次上板即与 golden 对齐 | 标量 `GetValue` 循环，与已在 CPU 侧逐元素验证过的分解式同构 |
| v2（性能） | 达标 §性能标准 全部 case | 向量化四级改造，按收益确定性排序实施（见下） |

v2 排序依据是「收益确定性与改造风险」，而非实现难度：

1. `trans=T/C` 分支改连续段 `DataCopy`（该分支对角元素公差为 1，收益与风险均已由索引性质确定）；
2. `trans=N` 分支改 gather/GatherMask（对角元素公差为 `lda-1`，仓内唯一先例 `extensions/complexmatdot/arch22/`）；
3. real/imag 分量分离缓冲（参照 `blas/gemv/arch22/cgemv_no_trans_kernel.cpp:61`）；
4. 分带流水与 double buffer（受 UB 预算约束，见「资源与调度」）。

### Host 侧设计

**参数校验（两级 + 顺序即语义）**：

```
第一级（无条件）：
  handle == nullptr                      → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
  alpha == nullptr || beta == nullptr    → ACLBLAS_STATUS_INVALID_VALUE
  m < 0 || n < 0                         → ACLBLAS_STATUS_INVALID_VALUE
  m == 0 || n == 0                       → ACLBLAS_STATUS_SUCCESS        // no-op 早退
  trans ∉ {OP_N, OP_T, OP_C}             → ACLBLAS_STATUS_INVALID_VALUE
  kl < 0 || kl > m-1                     → ACLBLAS_STATUS_INVALID_VALUE
  ku < 0 || ku > n-1                     → ACLBLAS_STATUS_INVALID_VALUE
  lda < kl+ku+1                          → ACLBLAS_STATUS_INVALID_VALUE
  incx == 0 || incy == 0                 → ACLBLAS_STATUS_INVALID_VALUE
第二级（仅参与计算时判空）：
  alpha != 0 时 A == nullptr 或 x == nullptr → ACLBLAS_STATUS_INVALID_VALUE
  beta  != 0 时 y == nullptr                 → ACLBLAS_STATUS_INVALID_VALUE
```

**允许返回码集合**：`include/cann_ops_blas_common.h:18` 的 `aclblasStatus_t` 共 12 个取值（0–11），
本算子只返回 3 个——`SUCCESS(0)`、`INVALID_VALUE(3)`、`HANDLE_IS_NULLPTR(9)`。

**逐参数返回码矩阵（14 项，与官方负向用例一一对应）**：

| 参数 | 触发条件 | 返回码 | 官方用例 |
| --- | --- | --- | --- |
| handle | `== nullptr` | `HANDLE_IS_NULLPTR` | TC_ED_235 |
| trans | 不在 `{OP_N,OP_T,OP_C}` | `INVALID_VALUE` | TC_ED_241（`trans=999`） |

（任务书 §2.5 原文已直接规定「非法参数与非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`」，
因此该返码不必靠用例反推。）
| m | `< 0` | `INVALID_VALUE` | TC_ED_243 |
| n | `< 0` | `INVALID_VALUE` | TC_ED_244 |
| m / n | `== 0` | **`SUCCESS`（短路 no-op）** | TC_ED_230 / 231 / 232 |
| kl | `< 0` 或 `> m-1` | `INVALID_VALUE` | TC_ED_247 |
| ku | `< 0` 或 `> n-1` | `INVALID_VALUE` | TC_ED_248 |
| alpha | `== nullptr` | `INVALID_VALUE` | TC_ED_236 |
| beta | `== nullptr` | `INVALID_VALUE` | TC_ED_237 |
| A | 参与计算且 `== nullptr` | `INVALID_VALUE` | TC_ED_238 |
| x | 参与计算且 `== nullptr` | `INVALID_VALUE` | TC_ED_239 |
| y | 参与计算且 `== nullptr` | `INVALID_VALUE` | TC_ED_240 |
| lda | `< kl+ku+1` | `INVALID_VALUE` | TC_ED_242 |
| incx / incy | `== 0` | `INVALID_VALUE` | TC_ED_245 / 246 |

**三条由官方用例反证出的硬约束**（照抄邻近算子会导致验收失败）：

1. **非法枚举返回 `INVALID_VALUE`，不得返回 `INVALID_ENUM`。**
   官方 `TC_ED_241`（`trans=999`）期望 `INVALID_VALUE`；仓内 arch35 `sgbmv_host.cpp:35-37` 同此语义，可作先例。
   需注意仓内另有 67 个文件（如 `blas/dgmm`、`blas/geam`）对非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，
   属同名库内的另一套规；本任务以任务书与官方用例为准。
2. **`m==0 || n==0` 的 no-op 早退必须先于 `kl ≤ m-1` / `ku ≤ n-1` 校验。**
   `TC_ED_230`（`m=0, kl=0`）按字面规则 `0 > -1` 会判非法，而官方期望 `SUCCESS`；
   `TC_ED_231/232` 同理。顺序颠倒将误杀 3 条官方用例（已用变异实现实测复现）。
3. **`lda` 判据使用未钳位的 `kl+ku+1`。**
   `TC_ED_247`（`kl=-1, ku=2, lda=2`）中 `lda == kl+ku+1` 成立，其非法性来自 `kl<0`；
   若先对 `kl` 取 `max(kl,0)` 再比 `lda`，返回码虽同但归因日志错误，排障与评审均受阻。

**分核策略**：输出维 `outDim = (trans==N) ? m : n`。GBMV 的输出行之间**无数据依赖、不需要跨核归约**
（这是相对 GEMM 的结构性优势），故按连续输出行段切块即可：

```
aivCoreNum = GetAivCoreCount()                       // arch22 现有 host 约定
useNumBlocks = min(ceil(outDim / MIN_ROWS_PER_BLOCK), aivCoreNum)
rowsPerBlock = ceil(outDim / useNumBlocks)
```

余出的行分配给前几个块（与仓内满核优先原则一致）。arch22 `cgemv_host.cpp` 现用 `numBlocks = 8` 固定值，
本算子改为按 `outDim` 动态取，避免小 case 空转与大 case 欠并行——该点列为真机需验证项。

**tiling 数据**：`CgbmvTilingData` 全部索引字段用 **int64**（理由见「资源与调度」），字段含
`m, n, kl, ku, lda, trans, numBlocks, rowsPerBlock, bandChunk, incx, incy, xDim, yDim,
alphaReal, alphaImag, betaReal, betaImag, maskOffset`。

**资源分配与释放**：mask 缓冲（arch22 约定 `MASK_OFFSET_BASE = 1024`、`uint32_t` dtype）、workspace、
tiling 设备缓冲均在 host 侧一次分配，异常路径逐层回退释放，与 `cgemv_host.cpp` 的 `CHECK_RET` 链一致。

### Kernel 侧设计

沿用 arch35 `sgbmv` 已验证的双路径骨架，按 arch22 复数访存规范重写：

1. **GM 直读路径**：输出行按块分配，逐行在带内列范围做复数乘加。适用于 `incx ≠ 1`、`outDim` 过小、
   或 x 分片超出 UB 容量的场景。
2. **UB 缓存路径**：`incx == 1` 且块内所需 x 分片可容于 UB 时，先整片搬入 UB 再计算，削减 GM 重复访存。
3. **复数分量处理**：参照 `cgemv/arch22` 的 real/imag 分离思路，用 `GatherMask` 把交错复数拆为
   实部/虚部两个连续向量，计算后 `Gather` 复原写回；共轭分支对 `A` 的虚部取符号，不触碰 `x`。
   arch22 上的具体 API 组合待真机确认（见「待实测清单」第 1 项）。
4. **写回**：按 alpha/beta 短路式写 `y`，物理位置由负步长映射决定。

**决定指令选择的访存结构性质（已用偏移集合单独证明，非经验推断）**：

| 分支 | 同一条对角带 `d = i-j` 上元素在打包数组中的排布 | 可用指令形态 |
| --- | --- | --- |
| `trans=N` | **公差 `lda-1`**（跨列但非连续） | gather / GatherMask 类 |
| `trans=T/C` | **公差 1（连续段）** | `DataCopy` 整段搬运 |

这条性质直接给出 v2 的实施顺序：先做 T/C 的 `DataCopy`（结构性收益确定），N 分支再上 gather。

### 资源与调度

**形状与访存规模（由官方 1200 条用例统计得到的实测上界）**：

| 量 | 上界 | 换算 |
| --- | --- | --- |
| `m, n` | 4096 | — |
| `kl, ku` | 2047 | — |
| `lda` | 4095 | — |
| 单条对角带元素数 | 4095 | **32KB**（8B/元素） |
| `x`/`y` 必需存储长度 | 6142 元素 | **48KB** |
| 最大被引用下标 | 16,773,119 | 128MB |

**索引宽度决策**：官方域内不发生 int32 溢出，但任务书未给参数上界，故 tiling 与 kernel 索引**一律 int64**——
跟 arch22 `cgemv_host.cpp` 的写法，不跟 arch35 `sgbmv` 的 uint32 习惯。

**UB 预算**：arch22 UB = 192KB（见 `cgemv_no_trans_kernel.cpp:61`），双缓冲则单半区 96KB。
最大带宽 case 上「一条带 32KB + x 32KB + y 32KB」**恰好占满半区、零余量**，因此不能整带常驻，必须分带流式：

```
bandChunk ≤ floor((96KB − 输出尾段预留 8KB − 安全余量 2×8KB) / 8B) = 10240 元素
```

该算式实现为 host 与 kernel 共用的单一函数 `CgbmvBandChunk()`，避免两处各写一份预算而失配。

**并行度**：`useNumBlocks` 依 `outDim` 动态取（见 Host 侧），AIV 核数由 `GetAivCoreCount()` 提供。

## 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（含 Atlas 800I A2、Atlas 800T A2 910B3） | 支持（本任务实现于 `blas/gbmv/arch22/`） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（含 Atlas 800I A3） | 支持（同 arch22，任务书 §7 第 4 条要求 A3 一并验收） |
| Ascend 950PR / Ascend 950DT | 不在本任务范围（该 arch 已有 `sgbmv`，复数扩展另议） |

构建接线按目录约定自动生效：`CMakeLists.txt:36-48` 将 `ascend910b*` / `ascend910_93*` 映射到 `arch22`，
`ascend950*` 映射到 `arch35`，故 `-DSOC_VERSION=ascend910b` 即命中本实现。

## 算子约束限制

- 仅 COMPLEX64；列主序带状打包；不支持广播与非连续 Tensor（超出 `lda/incx/incy` 语义者）。
- 不要求确定性累加顺序，故不同核数/分块下的结果容差按 §精度标准 判定，不承诺逐位一致。
- `y` 与 `A`/`x` 不得内存重叠。
- 不校验带外元素取值：带外区域（含 `lda` padding 行）允许为任意值乃至 NaN/Inf，实现必须保证不影响结果。
- `incx`/`incy` 为 0 判非法，不支持零步长广播语义。

# 可维可测分析（required）

## 精度标准与验证

### 判定标准

任务书 §3.2 与生态算子开源精度标准
（<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>）：
COMPLEX64 的实部与虚部分别按 FLOAT32 混合容差判定，

```
rtol = atol = 2^-13 ≈ 1.2207e-4
逐元素通过：|actual - golden| ≤ atol + rtol × |golden|
整体通过：matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32×ULP
```

`max_abs_error` 仅统计 actual 与 golden 均为有限的元素；双 NaN 与同号 Inf 按官方标准判匹配，
其余非有限差异计入未匹配。

### golden 来源

CPU golden 采用参考 BLAS 的 `cblas_cgbmv`（与 Netlib 同源）。框架已具备该链路：
`cmake/test.cmake:216-229` 执行 `find_path(cblas.h)` + `find_library(REFBLAS_LIB NAMES blas)`，
且复数以 `void*` 传参已是既有约定——`cblas_ccopy`（`test/copy/ccopy/arch35/ccopy_golden.h:33`）、
`cblas_sgbmv`（`test/gbmv/sgbmv/sgbmv_golden.h:39`）。因此本任务**不自造参考实现**，直接复用官方规；
仅当构建机的精简 BLAS 未导出 `cblas_cgbmv` 时启用 CPU 侧兜底实现（与下述已验证实现同源），
由 CMake `check_c_source_compiles` 探测符号后决定。

### 已完成的本机验证（不依赖 NPU，全部可复现）

**路径与结果**：

| 路径 | 实现结构 | 结果 |
| --- | --- | --- |
| A | 逐元素定义式 `y = α·A·x + β·y` | 基准 |
| B | 逐对角带累加（≈ kernel 结构） | 与 A 全等 |
| C | LAPACK 带状打包 + 对角累加（独立推导） | 与 A 全等 |
| 外部权威 | `scipy.linalg.blas.cgbmv`（Netlib 系 Fortran 实现） | 88/165 可比，0 失败 |
| C++ 语言层 | g++ 6.3.0 编译的 golden + 分核分解，参考值由 Netlib 现算注入 | 495 条 0 失败 |

覆盖 `trans ∈ {N,T,C}` × `m,n ∈ {1,2,3,5,7,8,15,16,17,31,32,33,64,65,128}` × `kl/ku` 边界
（0、1、上界 `m-1`/`n-1`、全带、仅上带、仅下带）× `lda` padding × `incx/incy ∈ {±1,±2,±3}` × 极瘦/极胖矩阵。

| 精度层 | 用例数 | 互校不合格 | 带外引用缺陷 | NaN/短路缺陷 | 最大绝对误差 |
| --- | --- | --- | --- | --- | --- |
| complex128（语义层） | 165 | 0 | 0 | 0 | 1.812e-14 |
| complex64（设备同精度） | 165 | 0 | 0 | 0 | 2.413e-05 |
| 对 Netlib 外部权威 | 88（覆盖率 53.3%） | 0 | 0 | — | 1.160e-05 |

complex64 的 2.413e-05 优于 `atol = 1.22e-4`，说明该语义在单精度下不会误伤容差。

**C++ 语言层实测（`g++ 6.3.0`，495 条取自官方 SUCCESS 用例的分层抽样）**：
抽样规则为「`lda*n ≤ 60000` 的小用例全入 + 最大 6 条 + 保证 `(trans × |incx| × |incy|)` 组合覆盖」，
元素预算 2.5M；每条用例的参考值由 `scipy.linalg.blas.cgbmv` 现算写入二进制，
C++ 侧不重复实现 BLAS 公式，从而保证「被测语言与交付语言一致」：

```
[1] partition bijection: combos=19200 bad=0
[2] cases=495  netlib_comparable=359  fail_vs_netlib=0  fail_vs_partition=0  bitwise_equal=203
    worst_abs_err_vs_netlib=1.726335e-003  worst_abs_err_partition_vs_serial=2.324150e-004
    per_trans: N=169 T=166 C=160 ; fail N=0 T=0 C=0
[3] status rules: mismatch=0 (null-pointer-driven=6)
=== RESULT: ALL_PASS ===
```

`worst_abs_err=1.73e-003` 出现在 `m=4096,n=1024,kl=1023,ku=1023` 这类累加项极多的用例，
属 float32 累加顺序差异而非公式差异（其所在用例仍满足 `atol + rtol·|golden|` 判定）。

**校验器有效性（故障注入）**：不注入缺陷的测试不作为证据。共 6 类注入全部被检出——

| 注入 | 缺陷 | 检出结果 |
| --- | --- | --- |
| F1 | 交换「非法参数检查」与「no-op 早退」顺序 | 误杀 3 条官方 SUCCESS 用例 ✓ |
| F2 | 打包索引式写反 | 触发越界 ✓ |
| F3 | 负步长按正步长处理 | 115/165 不合格、最大误差 7.3e+01 ✓ |
| F4 | 带外未裁剪 | 触发越界 ✓ |
| G1 | 打包行号符号写反（`ku-(i-j)`） | 151/165 与基准不符 ✓ |
| G2 | 对角范围 `d` 上下界互换 | 120/165 与基准不符 ✓ |

还原正确实现后全部重新通过，故上表「零缺陷」结论非假阴性。

**返回码全量对撞**：把上文校验规则套到官方 1200 条用例的 `expect_result`：

```
可比对用例 = 1194   一致 = 1194   冲突 = 0
指针空值类 = 6 条（其中 5 条由 CSV 直接编码，1 条靠测试侧构造）
顺序变异（F1）误杀的官方 SUCCESS 用例 = 3 条
```

### 已知覆盖缺口（不作无条件的合格声明）

`scipy` 的 f2py 包装对 `cgbmv` 额外断言 `m ≥ ku+kl+1`，该条件**严于** BLAS/cuBLAS 规范
（规范只要求 `lda ≥ kl+ku+1`，允许 `m < kl+ku+1`）。按该判据**逐条重数官方用例**：
1200 条中有 **332 条**满足 `m < kl+ku+1`（其中期望 `SUCCESS` 331 条），族分布为
`TC_EX` 283 / `TC_CV` 21 / `TC_RC` 18 / `TC_BW` 7 / `TC_ED` 2。
这些合法用例在本机**未获得外部对标**，必须在真机以 `cblas_cgbmv` 补齐。
（另：本机 `selftest` 实际对标的是 495 条 SUCCESS 抽样，受 `cases.bin` 体积预算与
`trans×|incx|×|incy|` 组合覆盖选取，因此“本机未对标”包含未抽样与宽带宽两类，不能混为一谈。）

### 自验步骤

使用任务随附 `cgbmv_test.csv`（1200 条，固定随机种子），执行：

```bash
python verify_accuracy.py --repo <ops-blas> --soc ascend910b3 --csv cgbmv_test.csv
```

先过 `TC_L0` 6 条基础用例，再打全量精度用例；若官方用例未覆盖任务书 **§3.5 第 4 条**
（小 shape 基础、shape 扫描、填充模式、对齐偏移、边界与负向）要求的场景则自行补充。
**空指针用例在官方 CSV 中的两种编码方式（本轮实测确认）**：

| 用例 | 编码列 | 取值 | 驱动方式 |
| --- | --- | --- | --- |
| TC_ED_236 / 237 | `alpha_real` / `beta_real` | 字面量 `null` | CSV 可驱动，但参数层必须识别 `null`（否则 `parseFloat("null")` 抛异常或静默变 0） |
| TC_ED_238 / 239 / 240 | `a_fill` / `x_fill` / `y_fill` | `NULLPTR` | CSV 可驱动，复用框架已有的 `BlasFillMode::M_NULLPTR` |
| TC_ED_235 | 无值列（仅 `description=null_handle`） | — | 必须测试侧构造 |

因此 6 条空指针用例中**仅 handle 一条需要 `TEST_F` 直接构造**，其余 5 条由 CSV 驱动；
`cgbmv_param.h` 已实现 `IsNullToken`/`ParseFloatOrNull`，且测试侧**不得**在 `y` 为空时
把它 `resize(1)`（那会把负向用例变成正向用例）。另需保证 `beta=0` 时 `y` 可为空的正向用例覆盖，
以锁定 §3.1.4 的短路语义。

## 性能标准与验证

### 达标口径（已用官方数据逐条验算）

任务书 §3.3 给出 5 条 case 的达标耗时。经与 `gpu_baseline.csv` 的 `gpu_ms` 列逐条对算，
两者严格满足同一关系式（|Δ| ≤ 0.0005us，即任务书数值就是按下式三位四舍五入）：

```
达标耗时(us) = gpu_ms(us) / 0.8
```

| case | m | n | kl | ku | trans | gpu_ms | gpu_us | gpu_us/0.8 | 任务书达标(us) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-000 | 256 | 256 | 16 | 16 | N | 0.008532 | 8.532 | 10.6650 | 10.665 |
| base-001 | 512 | 512 | 32 | 32 | N | 0.013862 | 13.862 | 17.3275 | 17.328 |
| base-002 | 1024 | 1024 | 64 | 64 | T | 0.020700 | 20.700 | 25.8750 | 25.875 |
| base-003 | 2048 | 2048 | 128 | 128 | N | 0.024512 | 24.512 | 30.6400 | 30.640 |
| base-004 | 4096 | 4096 | 64 | 256 | C | 0.041990 | 41.990 | 52.4875 | 52.487 |

因此本设计把统一口径固化为 **NPU 平均单次 kernel 耗时 ≤ gpu_ms / 0.8**，
`gpu_baseline.csv` 其余 195 条（`cgbmv-perf-*`）虽未在任务书表格中列出，同样可据此逐条换算目标值，
用于回归时判断优化是否退化。

**优先盯住的难例**（GPU 耗时最大，容差最紧）：

| case | 形状 | GPU | NPU 目标 |
| --- | --- | --- | --- |
| cgbmv-perf-072 | N 4096×4096 kl=ku=2047（全带） | 219.9us | 274.9us |
| cgbmv-perf-170 | C 3416×2694 kl=924 ku=857 | 196.1us | 245.1us |
| cgbmv-perf-124 | C 3233×3563 kl=949 ku=570 | 171.3us | 214.2us |
| cgbmv-perf-064 | T 4096×4096 kl=ku=512 | 130.3us | 162.9us |
| cgbmv-perf-005 | N 1×1（最小题，看固定开销） | 3.4us | 4.3us |

`perf-005/006/007` 这类 1×1/2×2/4×4 用例约束的是**启动与 tiling 固定开销**，与带宽优化无关，
是 `useNumBlocks` 动态取值最容易被拖累的地方，需单独观测。

### 采集方法

性能数据为 COMPLEX64 场景平均单次耗时，有效采样 >10 次取平均（msprof 自带 5 次 warmup），
用 `--gtest_filter` 收敛到单条用例避免 kernel 混叠：

```bash
msprof op --application="./build/test/gbmv/cgbmv/cgbmv_test --gtest_filter=*TC_PF*" \
          --output=./prof_cgbmv
```

统计 `OPPROF_*/OpBasicInfo.csv` 的 `Task Duration(us)` 求平均；
`PipeUtilization.csv` 与 `op_summary_*.csv` 用于判断瓶颈（向量流水占比、MTE2/MTE3 占比、Wait 占比）。
CPU 域调试与仿真仅用于功能检查与瓶颈定位，正式性能结论由规定设备实测给出。

### 内存占用

Host 侧按 `lda*n`、`1+(len-1)*|inc|` 精确计算必需存储，不使用宽松上界；
设备侧内存以 workspace + mask + tiling + 输入输出缓冲计。自测报告需给出每条 case 的实测值与理论值对照，
理论值可由 `02_语义验证/cgbmv_resource_sizing.py` 复算。

## 兼容性分析

新增公开声明位于 `include/cann_ops_blas.h`（与既有 `aclblasSgbmv` 同段），供其他产品线共用。
实现位于 `blas/gbmv/arch22/`，测试与用例位于 `test/gbmv/cgbmv/arch22/`。
`blas/gbmv/README.md` 需新增 `aclblasCgbmv` 小节，产品支持表把 A2/A3 系列标注为「支持」
（现该文件仅列 `aclblasSgbmv` 且 A2/A3 标「不支持」，属两接口分列，不改动 sgbmv 原有条目）。

**构建接线（逐条取证，结论是无需改动任何上层构建脚本）**：

| 接线点 | 机制与证据 |
| --- | --- |
| 算子源码收集 | `blas/CMakeLists.txt:14,18-25` `file(GLOB)` 递归 + `ARCH_SPECIFIC_DIRS` 区分 arch20/22/35 |
| SOC → arch 映射 | `CMakeLists.txt:36-48`：`ascend910b*`/`ascend910_93*` → `arch22` |
| 测试目录发现 | `test/CMakeLists.txt:17` GLOB 家族目录，注释明确支持嵌套（如 `test/gbmv/sgbmv/`） |
| 测试目标注册 | 新增 `test/gbmv/cgbmv/CMakeLists.txt` 一行 `ops_blas_add_gtest_tests(${OPS_BLAS} cgbmv_test)`，与 `test/gbmv/sgbmv/CMakeLists.txt:11` 同形 |
| 参考 BLAS | `cmake/test.cmake:216-229` 已要求 `cblas.h` + blas 库 |
| 符号导出 | 仓内算子声明均为裸 `aclblasStatus_t`，无 `ACLBLAS_API` 宏 |

**与测试框架对接时发现的两个陷阱**（会直接影响负向用例是否真的被执行）：

1. `test/frame/csv_loader.h:285-297` 的 `parseOpTrans` 对未识别 token **静默回退 `ACLBLAS_OP_N`**，
   而官方 `TC_ED_241` 的 `trans` 列写的是裸数字 `9`。若照常规写法解析，这条「非法枚举」用例
   会被转成合法计算，**测了等于没测**。对策：测试参数层保留原始 token 与可识别标记，
   未识别时以裸 int 构造 `aclblasOperation_t`（已实现）。
2. `test/frame/fill.h` 只有 `makeBlasComplexMatrix`（line 884），**没有复数版 banded/strided 填充**。
   按其文档化约定（实部用 `seed`、虚部用 `seed+1000`）在测试内实现局部复数填充；
   `A` 侧按 `lda*n` 全量随机填充，使带外同样是随机值，从而真正检验「实现只引用带内元素」。

# 待实测清单（本设计尚未证明、不得作为验收依据的部分）

1. arch22 复数访存的 API 组合（`GatherMask` 实虚分离 vs 交错直接计算）与 UB 定容。
2. `alpha == 0` 时是否跳过 `A`/`x` 读取循环（正确性已定，性能策略未定）。
3. 与真机 `cblas_cgbmv` golden 的逐位/容差比对，尤其 332 条 `m < kl+ku+1` 的宽带宽用例。
4. §性能标准 的 5 条 base 与 195 条 perf 用例实测达标情况。
5. `kl ≤ m-1` / `ku ≤ n-1` 的负向行为与 cuBLAS 是否完全一致（`m=0` 子情形已由官方用例固定为 `SUCCESS`，
   其余仍需真机确认）。
6. 6 条空指针用例在真机上的返回码断言（5 条由 CSV 的 `null` / `NULLPTR` 编码驱动，
   `handle` 一条由 `TEST_F` 构造；均已在测试侧实现，待执行）。
7. `GetAivCoreCount()` 在 910B3 上的实际返回值，及其对 `useNumBlocks` 动态策略的影响。
8. kernel 首次编译的语法层风险：tiling 以结构体指针读取 vs arch22 现有的字节偏移读法；
   `TEST_F` 挂在参数化 fixture 上（仓内已有先例 `test/chemm/arch22/chemm_test.cpp:124`）。

---

## 附：本机证据与复现方式

| 证据 | 位置 | 说明 |
| --- | --- | --- |
| 官方任务书 | `00_任务书与用例/aclblasCgbmv_A2A3_task_doc.md` | 任务随附 zip 解出，zip md5 `18ec50397570e922edd992892c9b6de5` |
| 官方用例 | `00_任务书与用例/cgbmv_test.csv` | 1200 条，md5 `100d510bcf29db95f4aab9d9a22da97e`（交付副本逐位一致） |
| GPU 基线 | `00_任务书与用例/gpu_baseline.csv` | 200 条，含 5 条 base |
| 三路径互证 + 故障注入 | `02_语义验证/cgbmv_semantics_check.py`、`cgbmv_fault_selfcheck.py`、`cgbmv_path_c_diag_check.py` | python |
| 外部权威对标 | `02_语义验证/cgbmv_external_conformance.py` | 输出覆盖率矩阵，禁止在有跳过时声称「全部一致」 |
| 返回码对撞 | `02_语义验证/cgbmv_status_rule_check.py`、`cgbmv_status_code_audit.py` | 1194 条对撞 + 顺序变异自检 |
| 形状/UB 预算 | `02_语义验证/cgbmv_resource_sizing.py` | int32 溢出判定、分带预算 |
| C++ 语言层自证 | `03_代码/local_verify/cpp_selftest/` | g++ 6.3.0 实测 `ALL_PASS`，含 `cases.bin` 可直接运行 |
| API 真实性静态校验 | `03_代码/local_verify/api_truth_lint.py` | 67 个标识符回查仓内 7.9MB 文本，`RED 0` |

复现入口：`03_代码/local_verify/build_selftest_linux.sh <官方CSV路径>`。
