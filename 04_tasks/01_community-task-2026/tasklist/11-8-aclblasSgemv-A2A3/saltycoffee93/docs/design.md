# aclblasSgemv 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasSgemv算子开发 任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/gemv/`（arch22 含 cgemv 双 kernel）、`blas/gemv/arch35/sgemv_host.cpp`、`include/cann_ops_blas.h`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasSgemv 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/gemv/arch22/`（实现）与 `test/gemv/sgemv/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度实数矩阵-向量乘加能力（BLAS-2），
对标 cuBLAS `cublasSgemv`（cuBLAS Library 文档 §2.6.2）。

## 背景介绍

### 算子功能定位

BLAS-2 列主序矩阵-向量乘加：`y = α·op(A)·x + β·y`，op(A) 由 trans 决定（N：A；T/C：Aᵀ，实数下 C≡T），y 原地更新。

数学公式：

```
y[j] = α × Σ_{i=1..k} op(A)[j,i] × x[i] + β × y[j]
trans=N：k=n、len(y)=m；trans=T/C：k=m、len(y)=n
A[i,j] 位于列主序偏移 (i−1) + (j−1)×lda（1-based）
β=0 时 y 输入值不被使用；参数校验通过后 m=0 或 n=0 quick return
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasSgemv | cublasSgemv | 单精度实数 GEMV：trans∈{N,T,C}、列主序 + lda 跨列、正负步长、原地输出 |

### 业界对标

- 对标接口：cuBLAS `cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)`（cuBLAS Library §2.6.2）。
- Golden 语义来源：Netlib BLAS `sgemv.f`（https://netlib.org/blas/sgemv.f）。
- 语义差异点（相对仓内既有实现）：
  1. **alpha=0 不短路**：实现侧按 `y = 0×op(A)x + βy` 计算（仍读 A/x）；golden `cblas_sgemv` 在 alpha=0
     时短路为 y=βy（不读 A/x）——二者仅在 A/x 含 Inf/NaN 时结果不同，该组合**不设精度用例**（任务书 §2.5）；
  2. **跨度溢出守卫须补齐**：arch35 `sgemv_host.cpp`（ValidateSgemvParams，L31-107）未提供
     `(逻辑长度−1)×|inc|` 的 int64 溢出守卫，A2/A3 实现须按任务书 §2.4 补齐（issue #866/#860 口径）；
  3. **编程模型**：arch22 无 SIMT，arch35 kernel 不可平移 → arch22 SIMD 双 kernel（N/T 分文件，
     沿用同目录 cgemv arch22 的 do_trans/no_trans 拆分形态）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/gemv/arch22/`（该目录**已存在**，含 cgemv 复数实现（no_trans/do_trans 双 kernel 分文件）→ **并列加文件**）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同目录 arch22 复数实现（主复用源） | `blas/gemv/arch22/cgemv_{host,no_trans_kernel,do_trans_kernel}.*` | arch22 工程骨架、trans 双 kernel 拆分、列主序访问、host 校验序 |
| 同算子 arch35 实现 | `blas/gemv/arch35/sgemv_{host,kernel,tiling_data}.*` | ValidateSgemvParams 校验序（L31-107）/tiling 规划平移；kernel 不平移（SIMT） |
| arch22 列流式累加范式 | `blas/gemv/arch22/cgemv_no_trans_kernel.cpp` | trans=N 列流式：x 驻 UB、逐列乘累加到 y（P4 性能范式，知识库 02-gemv族） |

### 接口规格

```cpp
aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
                             const float* alpha, const float* A, int lda,
                             const float* x, int incx, const float* beta, float* y, int incy);
```

- 唯一声明位置：`include/cann_ops_blas.h` **L468**（现场核验；任务书 §2.3 写 L461-463、§7.1 写 L457-459，
  以仓内实际为准）。**本次任务不修改公共头**（声明已存在、与 950PR 共用 → 红线 3）。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| trans | op(A) 类型，Host 内存 | aclblasOperation_t | — | ∈ {OP_N, OP_T, OP_C}；不在枚举集合返回 `ACLBLAS_STATUS_INVALID_VALUE` | — |
| m | A 的行数，Host 内存 | int | int32 | 任意非负 int32；m<0 → INVALID_VALUE；校验通过后 m=0 或 n=0 quick return | — |
| n | A 的列数，Host 内存 | int | int32 | 同 m | — |
| alpha | 标量 α 指针，Host 内存 | const float* | FLOAT32 | nullptr → INVALID_VALUE；值不校验（含 0/Inf/NaN）；alpha=0 不短路 | 标量 |
| A | 列主序系数矩阵，Device 内存，只读 | const float* | FLOAT32 | m>0 且 n>0 时 nullptr → INVALID_VALUE；元素值不校验 | lda×n（前 m×n 有效） |
| lda | A 主维（列间间距），Host 内存 | int | int32 | lda ≥ max(1,m)，否则 INVALID_VALUE | — |
| x | 输入向量，Device 内存，只读 | const float* | FLOAT32 | 逻辑长度（N: n / T,C: m）非零时 nullptr → INVALID_VALUE | 一维，物理长度 `1+(len−1)*|incx|` |
| incx | x 步长，Host 内存 | int | int32 | 非零且非 INT32_MIN；0/INT32_MIN 或跨度溢出（int64 计算）→ INVALID_VALUE | — |
| beta | 标量 β 指针，Host 内存；β=0 时 y 旧值不被使用 | const float* | FLOAT32 | nullptr → INVALID_VALUE；值不校验 | 标量 |
| y | 输入兼输出（原地覆写），Device 内存 | float* | FLOAT32 | 逻辑长度（N: m / T,C: n）非零时 nullptr → INVALID_VALUE | 一维，物理长度 `1+(len−1)*|incy|` |
| incy | y 步长，Host 内存 | int | int32 | 同 incx | — |

### 数学公式

```
y ← α·op(A)·x + β·y（op(A) = A / Aᵀ / Aᴴ≡Aᵀ）
```

与 §3.2 golden（cblas_sgemv，输入转 float64 真值计算后按 FLOAT32 收窄）同一形态；
行内点积归约顺序由实现决定（§2.5 确定性口径），按阈值判定。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-2 单精度 GEMV：FLOAT32 列主序矩阵、trans∈{N,T,C}、
lda 跨列、正负步长、原地输出；精度 rtol=atol=2^-13（matched_ratio≥0.99）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `y ← α·op(A)·x + β·y`（原地）。golden 白嫖仓内 `sgemv_golden.h` 的 `cblas_sgemv`（Netlib 3.10.0，任务书 §3.5/§7.4）；注意 golden alpha=0 短路差异（不设该组合精度用例） |
| 2 | 接口原型 | 12 参签名（上节），声明于 `include/cann_ops_blas.h` L468（现场核验），本次不改公共头 |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | **仅 FLOAT32**（A/x/y/alpha/beta 全 FLOAT32，任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 实现，不受影响 |
| 6 | 精度要求 | rtol = atol = 2^-13；required_matched_ratio = 0.99；max_abs_error ≤ 1e-2 or 32×ULP；Inf/NaN 按 §3.2 等效规则；golden 为 float64 真值收窄口径（任务书 §3.2，mixed_tolerance_standard，opbase commit 28a1ce52） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定，trans 三态均考核。典型 case（实测 CSV 读数，4096×4096，lda=4096，incx=incy=1）：OP_N → ≤65.79us；OP_T → ≤64.26us；OP_C → ≤65.28us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase 双 kernel**（N/T 分文件，沿同目录 cgemv arch22 形态）：OP_N 列流式累加（y 驻 UB，逐列乘累加，P4）+ OP_T/OP_C 行分块点积 + 跨核/块内归约。死路①：arch35 kernel 平移（SIMT）；死路②：单 kernel 内 trans 分支（同族已验证双 kernel 形态更清晰、tiling 更简单） |
| 9 | 评测来源 | 精度 golden：Netlib BLAS 3.10.0 `cblas_sgemv`（float64 真值收窄）；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V9，对齐 arch35 ValidateSgemvParams L31-107 + 任务书 §2.4 补强，顺序不可调换）**：

| 序 | 条件 | 返回码 |
|----|------|--------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `trans ∉ {OP_N, OP_T, OP_C}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `m < 0 \|\| n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V4 | `m == 0 \|\| n == 0` | `ACLBLAS_STATUS_SUCCESS`（quick return：不访问 A/x/y、不启动 kernel） |
| V5 | `lda < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `alpha == nullptr \|\| beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V7 | `incx == 0 \|\| incy == 0 \|\| incx == INT32_MIN \|\| incy == INT32_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V8 | `A == nullptr`（m>0 且 n>0 时）；x/y 逻辑长度非零时 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| V9 | `(逻辑长度−1)*\|inc\| 超 int32 范围（int64 计算，x/y 分别按各自逻辑长度；arch35 缺该守卫，本任务补齐）` | `ACLBLAS_STATUS_INVALID_VALUE` |

> V4（m/n=0 quick return）位于 V5..V9 之前：任务书 §2.4 明示"参数校验通过后 m=0 或 n=0 时不访问 A、x、y"——
> m/n 已过 V3 合法性，后续判空/跨度检查对 0 长度向量无意义，先返回。

---

# 详细设计（required）

## 算子分析

### 数学公式

`y[j] = α·Σ_i op(A)[j,i]·x[i] + β·y[j]`，元素定位 `(i−1)+(j−1)×lda`（列主序，1-based）。
行内点积为归约运算：累加顺序实现自定（不保证跨次一致），按阈值判定。

### 支持数据类型

A/x/y/alpha/beta 统一 FLOAT32；中间乘加保持 FLOAT32（行内点积累加 FLOAT32，
精度余量由 float64 golden 收窄口径与 2^-13 阈值覆盖）。
特殊值按 IEEE 传播；alpha=0×Inf/NaN=NaN 组合不设精度用例（golden 短路差异，§2.5）。

### 支持形状

| 场景 | 约定 |
|------|------|
| 矩阵 A | 列主序 lda×n 存储，前 m×n 有效；lda=max(1,m) 紧凑与 lda=m+padding（列间空洞）均支持 |
| 向量 | x 逻辑长度 N:n / T,C:m；y 逻辑长度 N:m / T,C:n；物理长度按步长展开，调用方保证有效 |
| m=0 或 n=0 | quick return：成功、不访问任何指针 |
| m=1 或 n=1 | 退化矩阵（向量-向量），边界用例覆盖 |
| 非方阵 | m≠n 支持（如 2048×512），tiling 按运行时 m/n 独立计算 |
| 步长语义 | 正负步长均生效（负步长自尾端起算）；0/INT32_MIN 拒绝 |
| 广播 / 0 维 | 不涉及（BLAS-2） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（沿同目录 cgemv arch22 的 trans 双 kernel 形态）：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **K_NO_TRANS**（sgemv_no_trans） | `trans==OP_N` | y[j] = α·Σ_i A[j,i]·x[i] + β·y[j]：**列流式累加**，y 驻 UB | 是（基线 3 条中 OP_N） |
| **K_DO_TRANS**（sgemv_do_trans） | `trans==OP_T \|\| OP_C` | y[j] = α·Σ_i A[i,j]·x[i] + β·y[j]：列=A 的行方向，行分块点积 + 归约 | 是（OP_T/OP_C） |

> OP_C 对实数矩阵与 OP_T 完全等价（任务书 §2.1），host 侧将 OP_C 直接映射到 do_trans 路径
> （tiling 记 trans 归一化后的 N/T 两态），不设第三条 kernel 路径。

数据流（K_NO_TRANS，列流式，y 驻 UB）：

```
GM y ──DataCopy──▶ UB yAcc（按 m 分块，长度 perCoreM）
                     │ β 缩放一次（β=0 置零、不读 GM y 旧值）
GM x ──DataCopy──▶ UB xBuf（整条 x 或分段驻留）
   for each column j (tile):
      GM A[:,j] ──DataCopy──▶ UB aCol
      yAcc += α·x[j]·aCol        （Muls+Add / Axpy 向量指令）
GM y ◀──DataCopy── UB yAcc（写回）
```

数据流（K_DO_TRANS，行分块点积）：

```
   for each y 分块 [j0, j0+cnt)（按核均分 len(y)=n）:
      UB acc ← 0
      for each k 分块:
         GM x[k..] ──DataCopy──▶ UB xTile
         GM A[k.., j0..]（跨列步进 lda）──▶ UB aTile
         acc += Σ aTile·xTile        （行内 Mul+ReduceSum）
      y[j0..] = α·acc + β·y[j0..]（β=0 不读 y 旧值）──DataCopy──▶ GM y
```

#### host 侧设计

**分核策略**：
- OP_N：按 y 长度 m 分核（每核一段连续 y 区间，逻辑序；负 incy 由物理基址映射消化）；
- OP_T/C：按 y 长度 n 分核；每核内 k 维（m）再按 tile 循环；
- 核数 `numBlocks = min(GetAivCoreCount(), len(y))`；kernel 侧自算
  `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（R4：tiling 不按核数组分配）。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- OP_N：yAcc 每核 8192 float（32 KiB）+ xBuf 驻留 + aCol tile 双缓冲；n 大时 x 分段驻留、外层循环；
- OP_T：aTile 按列块搬运（DataCopy stride=lda 跨列，或按行 Gather 备选），tile 尺寸平衡 UB 与跨列 DMA 效率；
- β 语义：β=0 时 yAcc 初始化为 0、**不发起对 GM y 旧值的读**；β≠0 时先读旧值缩放；
- 尾块 DataCopyPad 对齐补零（补零×x 为 0，不污染结果）。

**tilingkey 规划策略**：tiling 字段分支（kernel 文件已按 trans 拆分，路径内再以
`incx==1&&incy==1` 连续/步长字段分支）；Tiling 字段：transN（归一化 N/T）、m、n、lda、alpha、beta、
perCoreN、remainder、tileK、tileM、incx、incy（命名对齐 arch35 `sgemv_tiling_data.h` 与同目录 cgemv）。

**数据检测**：按 V1..V9 顺序执行（见需求拆解节）：
- V4（quick return）先于 V5..V9（任务书 §2.4"参数校验通过后"口径）；
- V9 为本任务新增守卫（arch35 未提供）：x、y 分别按**各自逻辑长度**（随 trans 变化）int64 计算，阈值 INT32_MAX/INT32_MIN 双边；
- alpha/beta 为 nullptr 判空（V6）不依赖其值——标量值本身不校验。

**其他 host 侧约定**：
- workspace：OP_T 路径跨核归约若采用两段式则需要（每核 partial 槽）；OP_N 列流式无跨核归约
  （y 按核不重叠）→ workspace 大小随路径动态返回，OP_N 传 0/nullptr（任务书 §3.4 不涉及内存判据）；
- 同步：异步下发，host 不同步（任务书 §2.5）；
- 日志：`OP_LOGE` 各校验失败分支，`OP_LOGD` tiling 决策（trans 归一化、分核结果）。

#### kernel 侧设计

**Init/Process 两阶段**（K_NO_TRANS 为例）：

```
Init:   解析 tiling → myOffset/myCount → SetGlobalBuffer → InitBuffer(yAcc/xBuf/aCol)
Process:
   1) β≠0：DataCopy GM y → yAcc，Muls(yAcc, yAcc, β)；β=0：Duplicate(yAcc, 0)（不读 GM）
   2) for each column tile j:
        MTE2: DataCopy aCol ← GM A[myOffset.., j]（列连续，lda 间隔跨列）
        MTE2: DataCopy xBuf ← GM x[j..]（驻留或分段）
        MTE2→V: SetFlag/WaitFlag
        V:    Muls(t, xBuf[j], α) → Axpy(yAcc, aCol, t, cnt)   // yAcc += α·x[j]·aCol
   3) incy==1：DataCopy GM y ← yAcc；incy≠1：asc_copy_ub2gm_align(dst_gap) 写回
```

**K_DO_TRANS 要点**：y 按核分块后行内点积用 `Mul + ReduceSum` 向量指令链；
GM A 跨列访问用列块 DMA（连续 m 段、列距 lda）或 DataCopy 逐列循环（tileK 实测定案）；
β 混合在块末一次完成。

**步长与负步长**：x/y 负步长按物理基址 `(len−1−i)*|inc|` 映射（同族 caxpy arch22 PhysicalIndex 范式）；
OP_N 路径 y 负步长在最终写回用 ub2gm 步长 DMA；x 负步长在取列标量时做逻辑↔物理换算（标量读，无 DMA 反转）。

**workspace 布局**（仅 OP_T 两段式时）：

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | numBlocks × 4B × perCoreY | 每核 partial 点积结果（若单 kernel 栅栏归约则仅标量槽） |

**同步事件配对**：MTE2→V、V→MTE3/MTE2 显式 SetFlag/WaitFlag；双缓冲掩盖搬运时延。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| trans 双 kernel | N/T 各自最优数据流，无运行时分支损耗（同目录 cgemv 已验证形态） |
| OP_N 列流式（P4） | y 驻 UB、x 标量广播、A 列连续 DMA——带宽型瓶颈下最优；y 只读写各一次 |
| OP_T 行分块 + ReduceSum | 行内点积全向量指令；A 跨列 DMA 尽量成块 |
| 双缓冲流水 | aCol/aTile 2 槽，MTE2 与 V 重叠 |
| β=0 免读 | 不读 GM y 旧值，省一次向量搬运 |
| host 零同步 | tiling const 引用传参，无 H2D、无 aclrtMalloc（OP_N 无 workspace） |

**未采用的路线及原因**：
- **arch35 kernel 平移**：SIMT 线程模型 + 缺跨度守卫——死路；
- **单 kernel trans 运行时分支**：数据流差异大（列流式 vs 行点积），分支恶化 tiling 与流水——拆分；
- **OP_T 整行驻留**：m 大时超出 UB，行分块循环替代。

---

## 支持硬件

| 芯片版本 | 支持情况 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | **支持**（本任务目标） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | **支持**（本任务目标） |
| Ascend 950PR / Ascend 950DT | 支持（由 arch35 既有实现提供，本次不改） |

---

## 算子约束限制

逐条对齐任务书 §2.5：

| 约束项 | 结论 |
|--------|------|
| 参数合法性与校验层级 | V1..V9 全序执行；trans 非法枚举 / m,n<0 / lda<max(1,m) / inc 0 或 INT32_MIN / alpha,beta nullptr / A,x,y 判空 / 跨度溢出 → INVALID_VALUE；m=0 或 n=0 → quick return 成功；标量值不校验 |
| 非连续 Tensor 支持 | 支持：lda 跨列空洞、incx/incy 步长（含负步长尾端起算）；0/INT32_MIN 拒绝 |
| broadcast 规则 | 不涉及（BLAS-2） |
| dynamic shape 要求 | 不要求：m/n/lda/inc 运行时入参，host tiling 按运行时规模计算 |
| 原地与视图语义 | y 原地覆写（β 读旧值；β=0 不读）；A、x 只读；不返回视图 |
| 确定性计算要求 | 不要求：行内点积累加顺序不保证跨次一致；单次执行内部确定 |
| 空 Tensor 处理 | 等价 m=0 或 n=0：不访问数据、不启动 kernel、返回成功 |
| 0 维 Tensor 处理 | 不涉及 |
| 异步执行 | 经 handle 绑定 stream 异步下发；读取输出前须 aclrtSynchronizeStream |

**已知限制（如实标注）**：
1. alpha=0 不短路（实现口径），与 golden 短路口径在 A/x 含 Inf/NaN 时结果不同——该组合不设精度用例（任务书 §2.5 已声明）；
2. arch35 与 arch22 的跨度溢出守卫行为不同（arch35 缺失），README per-arch 差异表须同步说明；
3. OP_T 路径 A 跨列 DMA 的 tile 策略在大 lda（深 padding）下效率待 S4 实测，备选 Gather 方案保留。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_sgemv`（输入转 float64 真值计算，按 FLOAT32 收窄落盘），经仓内 `sgemv_golden.h` | 任务书 §3.2/§7.4 |
| 普通精度（rtol/atol） | 2^-13 ≈ 1.2e-4（FLOAT32 档） | 任务书 §3.2，mixed_tolerance_standard（opbase commit 28a1ce52） |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | 1e-2 or 32 × ULP | 同上 |
| 判定式 | 逐元素 `\|actual − golden\| ≤ atol + rtol*|golden|`（y 全向量） | 任务书 §3.2 |
| Inf/NaN 特殊值 | 同为 NaN / 同为同号 Inf / 一方 NaN 另一方 ±Inf → 等效判过；一方有限另一方非有限 → 判不过；非有限值不计入统计、单独记录 | 任务书 §3.2 |
| alpha=0 特例 | 实现 alpha=0 不短路 vs golden 短路——该组合不设精度用例；alpha≠0 的 Inf/NaN 用例照常覆盖 | 任务书 §2.5 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定），trans 三态均考核 | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{sgemv_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/gemv/sgemv/arch22/`（仓内当前只有 arch35），并在 `test/gemv/sgemv/CMakeLists.txt` 补齐 arch22 GTest 注册（`ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)`，任务书 §2.2）；
- 用例覆盖（§3.5）：m/n ∈ {0,1,3,64,512,2048} 及 m≠n 非方阵；trans 三态全组合（含非法枚举负向）；alpha ∈ {0,1,2.5,−1.5,0.1,10}；beta ∈ {0,1,0.6,−0.5}（β=0 语义样本）；lda ∈ {max(1,m), m+8}（列间空洞）；incx/incy ∈ {1,2,7,−1,−3}（正负步长）+ 0/INT32_MIN/跨度溢出负向；A/x/y 含 Inf/NaN；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(len−1)*|inc|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **新增实现，无 ABI 变更**：`aclblasSgemv` 声明已存在于 `include/cann_ops_blas.h`（L468），本次不改公共头；
- **对 arch35 影响**：无。按 SOC 选目录编译互不覆盖；
- **对 arch22 同目录 cgemv 影响**：无。并列新文件（aclblasSgemv vs aclblasCgemv），接口独立；
- **测试构建影响**：`test/gemv/sgemv/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支不变；
- **文档影响**：`blas/gemv/README.md` 产品支持表新增 aclblasSgemv「A2/A3：支持」，并同步 per-arch 差异
  （arch22 新增跨度溢出守卫、alpha=0 不短路口径）。
