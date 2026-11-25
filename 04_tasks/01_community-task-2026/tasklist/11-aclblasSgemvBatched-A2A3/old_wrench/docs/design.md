# aclblasSgemvBatched 类算子设计文档

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：S=FP32；HSH/HSS=FP16；TST/TSS=BF16（半精度入参以 uint16_t 承载位模式）
> 合入位置：`ops-blas/blas/gemv_batched/arch22/`；测试：`ops-blas/test/gemv_batched/arch22/`
> 任务书：`aclblasSgemvBatched_AtlasA2_A3_task_doc.md`（**v2，2026-11-24 修订版**；下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——gemvBatched 类 5 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemv_batched/arch22/`；测试代码 → `test/gemv_batched/arch22/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件（设计文档 / 自测用例与代码 / 自测报告 / 待验收代码地址，§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入。
- 本设计与仓内现状的关系：上游已于 2026-11-23 合入 arch22 实数版五件套（commit `feb4f9c`），但其 host 契约为旧口径；本任务书 v2（11-24 修订）重新发布后，本设计 = **沿用已合入架构基线 + v2 契约对齐层**（差异清单见"总体架构与数据流"）。

## 背景介绍

### 算子功能定位

批量矩阵-向量乘：对 batchCount 个独立 (A_i, x_i, y_i) 三元组分别执行

```
y_i = alpha * op(A_i) * x_i + beta * y_i,   i = 0..batchCount-1
```

Aarray/xarray/yarray 为 Device 侧指针数组，各 batch 之间不要求连续（§2.1.1）。

| 接口 | 对标 | A / x / y dtype | 作用 |
|---|---|---|---|
| aclblasSgemvBatched | cuBLAS `cublasSgemvBatched`｜Netlib `sgemv` | FP32 / FP32 / FP32 | 单精度批量 gemv |
| aclblasHSHgemvBatched | cuBLAS `cublasHSHgemvBatched` | FP16 / FP16 / FP16 | 半精度入出 |
| aclblasHSSgemvBatched | cuBLAS `cublasHSSgemvBatched` | FP16 / FP16 / FP32 | 半精度入、单精度出 |
| aclblasTSTgemvBatched | cuBLAS `cublasTSTgemvBatched` | BF16 / BF16 / BF16 | bf16 入出 |
| aclblasTSSgemvBatched | cuBLAS `cublasTSSgemvBatched` | BF16 / BF16 / FP32 | bf16 入、单精度出 |

alpha/beta 恒为 FLOAT32 Host 标量；半精度/bf16 变体在 FP32 域完成计算后按输出 dtype 收窄（§2.1.6）。

### 业界对标

| 项 | cuBLAS 对应变体 | 本算子 |
|---|---|---|
| 参数序列 | handle/;trans/m/n/alpha/Aarray/lda/xarray/incx/beta/yarray/incy/batchCount | 逐参数一致（§2.3） |
| A 物理布局 | A 恒为 lda×n、lda ≥ max(1,m)，与 trans 无关；转置只作用于 op(A) | 同（§2.1.3、§9 修订） |
| trans 取值 | N/T/C | N/T/C，**实数下 C≡T 不共轭**（§2.1.4） |
| 特值语义 | alpha=0 只做 y=beta·y 不读 A/x；beta=0 不读 y 旧值 | 同（§2.1.5） |
| 非法 trans 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 统一口径，§2.3 说明） |
| 维数与 batch | int | 同（§2.3） |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口：Host C API 校验 + tiling，`<<<>>>` 直调 AIV kernel，经 handle 绑定 stream 异步执行，§2.2/§2.5）。
- 代码落位：`blas/gemv_batched/arch22/gemv_batched_{host,kernel}.cpp + gemv_batched_kernel.h + gemv_batched_tiling_data.h`（复用 2026-11-23 已合入的同名五件套文件，在其上做 v2 对齐，不新建族目录）；测试复用/扩展 `test/gemv_batched/` 通用工程并注册 arch22 目标。
- 公共头：`include/cann_ops_blas.h` L287-308 **5 条声明均已存在，无需修改**（grep 核验；§2.3"无需新增"与仓内现状一致）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemv_batched/arch22/gemv_batched_*`（11-23 合入） | ★ 主基线 | dtype 模板分发、AIV 按 batch 切核、N 路列流式累加 / T 路逐列点积、FP32 域计算与收窄、strided 访问与负步长折算 | host 校验层按 §2.5 重排 + OP_C + quick return；README trans 行同步 |
| `blas/gemv_batched/arch35/gemv_batched_*` | ◆ 参考 | 变体分派与 UB 预算 tiling 思路（Transpose 切 dotTile×outTile） | arch22 沿用固定 TILE=2048 方案，不动 arch35 |
| `blas/gemv_batched/arch22/cgemv_batched_*` | ◆ 参考 | 指针数组工程组织、族内错误码口径 | 复数路径不涉及 |
| `blas/common/helper/gemv_batched_span_check.h`、`gemv_batched_layout.h` | 共享基建 | INT32 跨度上限拒绝（issue #860/#848 口径） | 直接调用，不重写 |

### 接口规格

签名与任务书 §2.3 逐字一致（以 `include/cann_ops_blas.h` 为准，已存在）：

```cpp
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float *const Aarray[], int lda,
    const float *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasTSTgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasTSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
```

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 指向有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| trans | op(A) 运算类型 | attr | int | {OP_N, OP_T, OP_C}，实数 C≡T；其它 → `INVALID_ENUM` | 标量 |
| m | A 的行数 | attr | int | m ≥ 0；负 → `INVALID_VALUE`；0 为合法 quick return | 标量 |
| n | A 的列数 | attr | int | n ≥ 0；负 → `INVALID_VALUE`；0 为合法 quick return | 标量 |
| alpha | 标量乘数（FP32） | scalar | FLOAT32 | 任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| Aarray | A[i] 指针数组（Device） | tensor | 按变体 | 每个 A[i] 为 lda×n（与 trans 无关）；batchCount>0 且 m,n>0 时 nullptr → `INVALID_VALUE` | batchCount 个 lda×n |
| lda | A 前导维 | attr | int | lda ≥ max(1,m)（与 trans 无关）；低于下界 → `INVALID_VALUE` | 标量 |
| xarray | x[i] 指针数组（Device） | tensor | 按变体 | 每个x[i] 长 n（N）/ m（T、C）；batchCount>0 且逻辑长度>0 时 nullptr → `INVALID_VALUE` | batchCount 个向量 |
| incx | x 元素步长 | attr | int | 任意非零 int（含负）；0 → `INVALID_VALUE` | 标量 |
| beta | 标量乘数（FP32）；beta=0 不读 y 旧值 | scalar | FLOAT32 | 任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| yarray | y[i] 指针数组（Device，原地输出） | tensor | 按变体 | 每个 y[i] 长 m（N）/ n（T、C）；各 y[i] 互不重叠；判空同 xarray | batchCount 个向量 |
| incy | y 元素步长 | attr | int | 任意非零 int（含负）；0 → `INVALID_VALUE` | 标量 |
| batchCount | batch 数 | attr | int | ≥ 0；负 → `INVALID_VALUE`；0 为合法 no-op | 标量 |

> 变体 dtype 约定（§2.4）：S=FP32/FP32/FP32；HSH=FP16/FP16/FP16；HSS=FP16/FP16/FP32；TST=BF16/BF16/BF16；TSS=BF16/BF16/FP32；`uint16_t` 仅承载 FP16/BF16 位模式。

### 数学公式

```
y_i = alpha * op(A_i) * x_i + beta * y_i
op(A_i) = A_i            (trans = N，m×n)
op(A_i) = A_i^T          (trans = T 或 C，n×m，实数无共轭)
```

A_i 列主序、恒为 lda×n；x_i 长度 = N?n:m，y_i 长度 = N?m:n（§2.1.2/§2.1.3）。半精度/bf16 变体计算与 golden 判定域见"算子分析·数学公式"。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现 5 个句柄式批量 gemv 接口：单 AIV kernel 模板覆盖 5 组 dtype 组合，支持 trans N/T/C（C≡T）、任意合法前导维与非零（含负）步长、指针数组形态的 batch 组织；精度达生态算子开源混合容差标准（按输出 dtype 查表），性能达 GPU 标杆 0.8×（单 case 判定）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | `y=α·op(A)x+β·y`；批量 = 逐 batch 独立 gemv，无 batch 间归约——可完全复用仓内已合入 arch22 gemv_batched 的 N 路列流式 / T 路逐列点积两条路径 |
| 2 | 接口原型 | §2.3 五签名（见上），返回 `aclblasStatus_t`；公共头 L287-308 已存在零改动 |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | 5 组组合（S/HSH/HSS/TST/TSS），alpha/beta 恒 FP32；FP16/BF16 以 uint16_t 承载 |
| 5 | 目标芯片与 arch 映射 | A2/A3（910b 系）→ arch22 / dav-2201；构建 `--soc=ascend910b3`；950PR/DT（arch35）已有实现不受影响 |
| 6 | 精度要求 | §3.2：输出按 dtype 查 mixed 表——FP16 rtol=atol=2^-10、BF16 2^-7、FP32 2^-13；ratio ≥0.99；max_abs_error ≤ 1e-1（半精度）/ 1e-2（FP32）或 32 ULP；golden = 逐 batch cblas_sgemv |
| 7 | 性能预期 | §3.3：单 case NPU 平均 kernel 耗时 ≤ 1.25×gpu_ms（≥0.8× GPU）；msprof op Task Duration 口径，warmup5 + 10 样本均值；端到端事件计时仅交叉验证 |
| 8 | 代码架构选型 + 可行性 | AIV 单 kernel 模板（dtype 为模板参数 + tiling.dtype 分发），host 侧 batch 均分切核；已排除死路：cube 路径（gemv 无矩阵级复用收益）、双 kernel 拆 N/T（一次发射更优）；11-23 已合入同架构实现 = 可行性旁证 |
| 9 | 评测来源 | 任务书自验包 test_cases/（5 变体各 1200 条 + gpu_baseline.csv 200 条）；仓内 ATK 测试框架 + verify_{accuracy,performance}.py |
| 10 | v2 契约对齐层 | 相对 11-23 已合入实现需补 4 点：① m/n/batchCount=0 → SUCCESS quick return（现为 INVALID_VALUE）；② 接受 OP_C≡T（现为 INVALID_ENUM）；③ 校验顺序按 §2.5 重排（trans 提到 m/n 前）；④ 指针条件判空 + quick return 前置结构校验；README trans 行同步 |
| 11 | 退化与特值 | alpha=0 → y=βy 不读 A/x（kernel 内已有分支）；beta=0 → 不读 y（含 y 为 NaN 场景）；负步长按 BLAS 反向遍历（起点折算 (1-count)·inc） |
| 12 | workspace | 无（指针数组直接 GM 读取，UB 内完成全部暂存） |

# 详细设计

## 算子分析

### 数学公式

逐 batch（i = 0..batchCount-1）独立计算，无 batch 间交互：

```
N 路（trans=OP_N，op(A_i) 为 m×n）：
    y_i[r] = alpha * Σ_c A_i[r + c*lda] * x_i[c] + beta * y_i[r],   r = 0..m-1, c = 0..n-1
T/C 路（op(A_i) = A_i^T 为 n×m，实数无共轭）：
    y_i[c] = alpha * Σ_r A_i[r + c*lda] * x_i[r] + beta * y_i[c],   c = 0..n-1, r = 0..m-1
```

x_i 元素地址 = xarray[i] + k·incx，y_i 元素地址 = yarray[i] + k·incy（负步长按 BLAS 反向遍历语义，起点折算 (1-count)·inc）。计算域：FP16/BF16 输入先 cast 到 FP32，全部乘加在 FP32 完成，按 y 的输出 dtype（FP16/BF16/FP32）收窄（§2.1.6）。alpha/beta 为 FP32 Host 标量，随 tiling 结构传值。

特值语义（§2.1.5）：alpha=0 → 只做 y=β·y，**不读 A/x**（x 含 Inf/NaN 不传播）；beta=0 → **不读 y 旧值**（y 含 NaN 亦安全，输出 = alpha·op(A)x 或 0）。

### 支持数据类型

| 变体 | A | x | y | alpha/beta | kernel 模板实例 | dtype 码 |
|---|---|---|---|---|---|---|
| S | FP32 | FP32 | FP32 | FLOAT32 | GemvBatchedKernel<float,float> | 1 |
| HSH | FP16 | FP16 | FP16 | FLOAT32 | GemvBatchedKernel<half,half> | 0 |
| HSS | FP16 | FP16 | FP32 | FLOAT32 | GemvBatchedKernel<half,float> | 2 |
| TST | BF16 | BF16 | BF16 | FLOAT32 | GemvBatchedKernel<bfloat16_t,bfloat16_t> | 3 |
| TSS | BF16 | BF16 | FP32 | FLOAT32 | GemvBatchedKernel<bfloat16_t,float> | 4 |

dtype 码沿用仓内已合入实现取值（arch22 `gemv_batched_host.cpp` 五入口 → `GemvBatchedImpl<TIn,TOut>(..., dtype)`）。

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m, n | ≥ 0（int） | m=0 或 n=0 → quick return：SUCCESS、不启动 kernel、Aarray/xarray/yarray 允许 nullptr（§2.5 空 Tensor 行） |
| batchCount | ≥ 0（int） | 0 → no-op：SUCCESS、不下发 kernel、指针数组允许 null（§2.1.7） |
| lda | ≥ max(1, m)，与 trans 无关 | 低于下界 → INVALID_VALUE（§2.4） |
| incx/incy | 任意非零 int（含负、含大步长 {7,8,16,31,64,127} 抽样） | 0 → INVALID_VALUE |
| 覆盖档位（测试口径 §3.5） | m,n ∈ {0,1,2,3,5,7,16,31,32,33,63,64,65,127,128,129,255,256,511,512,1023,1024,2047,2048,4095,4096}；batchCount ∈ {0,1,2,3,8,16,64,256} | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

单 AIV kernel 模板 + host 侧 batch 均分切核；5 个 C 入口收敛到一个模板实现：

```
aclblas{S,HSH,HSS,TST,TSS}gemvBatched   (5 个 C API)
        └─ GemvBatchedImpl<TIn,TOut>(dtype)      ← host：校验→quick return→切核→tiling→发射
                └─ gemv_batched_kernel_do(...)    ← `<<<usedCoreNum,,,stream>>>` 直调
                        └─ gemv_batched_kernel_impl   (AIV, extern "C")
                                └─ GemvBatchedKernel<TIn,TOut>::Init/Process
                                        ├─ trans=0 → ProcessNormalBatch（N 路列流式）
                                        └─ trans=1 → ProcessTransBatch（T/C 路逐列点积）
```

**v2 契约对齐层**（相对 2026-11-23 已合入实现，host 侧）：

| # | 对齐项 | 现状（arch22 host） | v2 目标 |
|---|---|---|---|
| 1 | quick return | m/n/batchCount ≤0 → INVALID_VALUE | 负值 → INVALID_VALUE；**0 → SUCCESS** 且不启动 kernel |
| 2 | OP_C | trans ∉{N,T} → INVALID_ENUM | 接受 **OP_C≡T**（映射 trans=1）；仅非法枚举报错 |
| 3 | 校验顺序 | m/n→指针→trans | §2.5：handle→trans→m/n→lda→incx/incy→batchCount→alpha/beta→Aarray/xarray/yarray→quick return |
| 4 | 指针判空 | 无条件 | **条件判空**（仅 batchCount>0 且 m,n>0 时报 INVALID_VALUE），quick return 场景允许 null |

数据流（N 路，列流式、y 驻 UB）：

```
GM: Aarray[i]/xarray[i]/yarray[i] --DataCopyPad(8B)--> UB: ptr
GM: y_i --(beta≠0 时, strided)--> UB: yIn --cast FP32--> acc := beta*acc
for 每列 c (0..n-1):
    GM: x_i[c] (strided) --> UB: xIn/x32, 取标量 scale = alpha*x32[c]
    GM: A_i 列段 [c*lda+mOff, mCnt] --DataCopyPad--> UB: aIn --cast--> a32
    UB: acc += a32 * scale                      (Muls + Add)
UB: acc --cast 收窄--> yIn --(strided)--> GM: y_i
```

数据流（T/C 路，逐列点积）：

```
GM: y_i --(beta≠0 时)--> acc
for 每输出列 c (0..n-1):
    dot = 0
    for rOff in m 分段(TILE=2048):
        GM: A_i 列段 [c*lda+rOff, rCnt] --> aIn --cast--> a32
        GM: x_i[rOff..rOff+rCnt) (strided) --> xIn --cast--> x32
        a32 *= x32; dot += ReduceSum(a32)       (Mul + ReduceSum + 标量累加)
    acc[c] += alpha * dot
GM: y_i <--(strided, 收窄后)-- acc
```

#### host 侧设计

- **分核策略**（满核原则、均分与余块）：`coreNum = GetAivCoreCount()`；`batchPerCore = ⌈batchCount/coreNum⌉`；`usedCoreNum = ⌈batchCount/batchPerCore⌉`（不满 batch 时收缩用核数，避免空核）；`batchTail = batchCount − (usedCoreNum−1)·batchPerCore`（尾核余块）。每核处理连续区间 `[blockIdx·batchPerCore, +myBatches)`，batch 间无依赖、无核间通信。
- **数据分块和内存优化策略**：m/n 按 `GEMV_BATCHED_ARCH22_TILE = 2048` 分段（tiling_data.h 常量）；UB 七 buffer（aIn/a32/xIn/x32/yIn/acc/work 各 2048×float）+ 归约/指针/标量小 buffer（64）+ 仅非单位步长时启用 4096×float 的 gather/scatter 窗口 `WIN_MAX`；拷贝与计算靠 HardEvent 交叉等待（MTE2↔V↔S）形成软流水；指针数组元素按需 8B 直读，不整体搬 GM→UB。
- **tilingkey 规划策略**：本工程为直调 kernel，无独立 tilingkey 机制，分支承载于 tiling 结构字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| dtype | 0/1/2/3/4 | HSH/S/HSS/TST/TSS 模板分发 |
| trans | 0 / 1 | N 路 / T·C 路（host 侧 C→1 折算） |
| alpha / beta | ==0.0F 特值 | kernel 内隐式分支：alpha=0 跳过 A/x 读取；beta=0 以 Duplicate(0) 初始化 acc 不读 y |
| usedCoreNum/batchPerCore/batchTail | 切核三元组 | 核间 batch 划分 |

- **数据检测**（校验顺序与异常码，逐条对齐 §2.4/§2.5；结构合法后才执行 quick return）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | trans ∉ {N,T,C} | INVALID_ENUM |
| 3 | m<0 或 n<0 | INVALID_VALUE |
| 4 | lda < max(1,m) | INVALID_VALUE |
| 5 | incx==0 或 incy==0 | INVALID_VALUE |
| 6 | batchCount<0 | INVALID_VALUE |
| 7 | alpha/beta == nullptr | INVALID_VALUE |
| 8 | batchCount>0 且 m,n>0 时 Aarray/xarray/yarray == nullptr | INVALID_VALUE |
| 9 | 形状跨度（span_check，见下） | INVALID_VALUE（不下发 kernel） |
| 10 | 以上全过后：batchCount==0 或 m==0 或 n==0 → 返回 SUCCESS，不下发 kernel | — |

  跨度上限（复用 `blas/common/helper/gemv_batched_span_check.h`，issue #860 口径，实数 elemFactor=1）：A 元素跨度 `(n-1)·lda + m ≤ INT32_MAX`；x 跨度 `1+(dotDim-1)·|incx|`、y 跨度 `1+(outDim-1)·|incy| ≤ INT32_MAX`（dotDim=N?n:m，outDim=N?m:n，含负步长组合）；总输出 `batchCount·outDim ≤ INT32_MAX`（kernel 侧核间切分为 uint32 算术）。
  kernel 发射失败（aclError ≠ SUCCESS）→ `EXECUTION_FAILED` 并记日志。

#### kernel 侧设计

- **Init 阶段**：读 `GetBlockIdx()`，按切核三元组判定 active 与本核 batch 区间；Aarray/xarray/yarray 以 `GlobalTensor<uint64_t>` 绑定（长度 batchCount）；按 dtype 布局初始化 UB buffer（单位步长不初始化窗口 buffer）。
- **Process 阶段**：dtype 分发 → `ProcessTyped<TIn,TOut>`：逐 batch 用 `ReadDevicePtr`（DataCopyPad 8B + GetValue）取 A_i/x_i/y_i 设备指针；按步长符号折算 x/y 起始偏移（负步长 `(1-count)·inc`）；按 span 上界绑定 GM tensor；trans 分派 N/T 路径。
- 两路计算流程见"总体架构与数据流"ASCII 图；N 路对每个 m 段先 `InitYAcc`（beta=0 → Duplicate(0)，否则读 y·cast·Muls(beta)），alpha≠0 时逐列累加，最后收窄写回；T 路对每个 n 段先 `InitYAcc`，alpha≠0 时逐列 `ComputeColumnDot`（m 分段 Mul+ReduceSum 标量累加），`acc += alpha·dot` 后写回。
- **workspace 布局**：无 workspace；指针数组（调用方 Device 内存）直接按元素读取。

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| batch 均分切核 | 满核原则 + usedCoreNum 收缩 | 充分利用 A2/A3 全部 AIV 核，小 batchCount 不空转（待实测回填） |
| N 路列流式 | y 驻 UB，A 按列段流过，单次 GM 读即消费 | 免 A 的整体 UB 驻留，支持大 m/n（4096 档）超 UB 规模 |
| FP32 计算域 | half/bf16 cast 后乘加 | 精度达 mixed 标准（半精度 2^-10/2^-7），避免半精度累加漂移 |
| MTE2/V/S 软流水 | HardEvent 交叉等待 | 隐藏搬运延迟（待 msprof 回填占比） |
| 标量特值分支 | alpha/beta 特值免读对应操作数 | 特值用例省带宽（对 §3.5 特值占比有效） |
| 已排除路线 | cube 路径（gemv 无矩阵级收益）、N/T 双 kernel 拆分（多一次发射开销） | — |

性能判据逐 case 独立：NPU 平均 Task Duration ≤ 1.25×gpu_ms；gpu_baseline 200 条实测分布——S max 0.2687ms / avg 0.0254ms，HSH 0.4063/0.0284，HSS 0.4072/0.0288，TST 0.4066/0.0284，TSS 0.4071/0.0289（ms）。带宽线能否达标以设备 msprof 实测为准（设计阶段不判决）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列产品 | ✔（本任务目标，arch22） |
| Atlas A2 推理系列产品 | ✔（同 arch22） |
| Atlas A3 训练系列产品 | ✔（本任务目标，arch22） |
| Atlas A3 推理系列产品 | ✔（同 arch22） |
| Ascend 950PR / 950DT | （arch35 既有实现，本任务不涉及） |

## 算子约束限制

| 约束项 | 内容（§2.5 逐条） |
|---|---|
| 校验层级 | 全部结构参数在 host 侧校验，不对 Device 张量做元素级扫描 |
| 非连续支持 | incx/incy 非零跨步（含负步长反向遍历）；A 按 lda 列主序带填充；batch 起址由指针数组描述，不要求 batch 间连续 |
| broadcast | 不支持（各 batch 独立操作数） |
| dynamic shape | 天然支持（m/n/lda/incx/incy/batchCount 运行时入参，host 按实际值 tiling） |
| 原地语义 | y[i] 原地覆写；调用方保证 y[i] 与 A[i]/x[i] 不重叠、各 y[i] 互不重叠；不返回视图 |
| 确定性 | 不保证浮点累加逐位一致；按 §3.2 阈值判定 |
| 空 Tensor | batchCount=0 / m=0 / n=0 → SUCCESS no-op，不引用 A/x/y |
| 0 维 Tensor | 不涉及（指针数组 + 维数描述形态） |
| 异步执行 | handle 绑定 stream 异步；返回≠完成，读回前须同步 stream |
| 跨度上限 | span_check 三条上界（见"数据检测"），超限 INVALID_VALUE 不下发 kernel |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden = 逐 batch Netlib cblas_sgemv（FP64 累加后按输出 dtype 收窄落盘，或 cblas FP32 对照，差异在自测报告说明）；判定 = `|actual−golden| ≤ atol + rtol·|golden|` 逐 batch 逐元素，matched_ratio ≥ 0.99 **且** max_abs_error ≤ 上限时通过；rtol=atol：FP16 2^-10(1e-3)、BF16 2^-7(7.8e-3)、FP32 2^-13(1.2e-4)；max_abs_error_limit：1e-1（FP16/BF16）/ 1e-2（FP32）或 32 ULP | §3.2；生态算子开源精度标准 mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel：msprof op 单算子 Task Duration，warmup 5 + 连续 10 次取均值，**单 case NPU avg ≤ 1.25×gpu_ms**（=达 GPU 0.8×），逐 case 判定；GPU 基线 = cuBLAS 对应变体同参数 kernel-only 采集；端到端 aclrtEvent avg_us 仅交叉验证不作判据 | §3.3；test_cases/gpu_baseline.csv |

## 测试概览

- 用例规模：5 变体 × 1200 条（S 变体实测分布：L0 6 / L1 39 / L2 21 / L3 8 / L4 14 / L5 28 / L6 28 / EX 856 / PF 200）＝ 精度 1000 + 性能 200 每变体（§3.5 口径：精度全量 5000 + 性能子集 1000）；负向随覆盖集设计（非法 trans/负值/零步长/非法 lda/指针数组成员 y[i] 重叠等）。
- 特殊场景：batchCount=0；m=0/n=0 quick return；beta=0 且 y 含 NaN；alpha=0 且 x 含 NaN；FP16/BF16 Inf/NaN 传播；lda 紧凑/+1/+8/+64 padding；incx/incy ±1/±2/±3 + 大步长抽样；性能子集 alpha/beta 显式保留（约 12% 非默认标量）。
- 性能口径：msprof op（正式）+ GTest `[PERF] case=... avg_us=...` 事件计时（交叉验证，verify_performance.py 解析对表 gpu_baseline.csv）。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + ATK 测试工程，CSV 驱动）。

## 兼容性分析

- 公共头零改动（5 条声明已在 `include/cann_ops_blas.h` L287-308），无接口面波及。
- 行为变更（v1→v2 对齐，均在 arch22 范围内）：m/n/batchCount=0 的返回码 INVALID_VALUE→SUCCESS；OP_C 由 INVALID_ENUM→按 T 接受；校验顺序重排；README trans 行与约束节同步补充 OP_C 与 quick return 口径。既有 `test/gemv_batched/arch22/gemv_batched_test.csv` 中相关负向预期需同步为 v2 口径（golden.h 已按 v2 判 lda，无需改）。
- 跨 arch：arch35 / cgemv_batched 路径不触碰；共享基建 `blas/common/helper/gemv_batched_{span_check,layout}.h` 原样复用。
- 新增文件 vs 已合入文件：在 2026-11-23 已合入五件套上原位修改（host 重排校验层），不新增文件，符合"文件个数正常、无冗余"的合入规范（§5）。

