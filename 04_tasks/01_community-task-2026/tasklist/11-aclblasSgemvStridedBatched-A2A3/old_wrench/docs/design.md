# aclblasSgemvStridedBatched 类算子设计文档

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：S=FP32；HSH/HSS=FP16；TST/TSS=BF16（半精度入参以 uint16_t 承载位模式；alpha/beta 恒 FLOAT32）
> 合入位置：`ops-blas/blas/gemv_strided_batched/arch22/`；测试：`ops-blas/test/gemv_strided_batched/arch22/`
> 任务书：`aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——gemvStridedBatched 类 5 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemv_strided_batched/arch22/`；测试代码 → `test/gemv_strided_batched/arch22/`（复用通用测试工程并注册 arch22 目标，§1/§2.2/§5）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 本设计与仓内现状的关系：上游已于 2026-11-23 合入 arch22 实数版五件套（commit 21:09，cann-robot）；本设计 = **沿用已合入架构 + 任务书契约逐条对齐**（对齐项见"总体架构与数据流"）。

## 背景介绍

### 算子功能定位

跨步批量矩阵-向量乘：batchCount 个 (A_i, x_i, y_i) 三元组由**基址 + 元素偏移**定位（无指针数组）：

```
y + i*stridey = alpha * op(A + i*strideA) * (x + i*stridex) + beta * (y + i*stridey),  i = 0..batchCount-1
```

| 接口 | 对标 | A / x / y dtype | 作用 |
|---|---|---|---|
| aclblasSgemvStridedBatched | cuBLAS `cublasSgemvStridedBatched`｜Netlib `sgemv` | FP32 / FP32 / FP32 | 单精度跨步批量 gemv |
| aclblasHSHgemvStridedBatched | cuBLAS `cublasHSHgemvStridedBatched` | FP16 / FP16 / FP16 | 半精度入出 |
| aclblasHSSgemvStridedBatched | cuBLAS `cublasHSSgemvStridedBatched` | FP16 / FP16 / FP32 | 半精度入、单精度出 |
| aclblasTSTgemvStridedBatched | cuBLAS `cublasTSTgemvStridedBatched` | BF16 / BF16 / BF16 | bf16 入出 |
| aclblasTSSgemvStridedBatched | cuBLAS `cublasTSSgemvStridedBatched` | BF16 / BF16 / FP32 | bf16 入、单精度出 |

### 业界对标

| 项 | cuBLAS 对应变体 | 本算子 |
|---|---|---|
| 参数序列 | handle/trans/m/n/alpha/A/lda/strideA/x/incx/stridex/beta/y/incy/stridey/batchCount | 逐参数一致（§2.3；维数与 batchCount 用 int，stride 用 int64_t） |
| batch 定位 | 基址 + i·stride（元素偏移，允许间隙） | 同（§2.1 公式） |
| A 物理布局 | A_i 恒为 lda×n、lda ≥ max(1,m)，与 trans 无关 | 同（§2.1.3——转置只作用于 op(A)） |
| trans 取值 | N/T/C | N/T/C，实数下 **C≡T 不共轭**（§2.1.4） |
| 特值语义 | alpha=0 只做 y=beta·y 不读 A/x；beta=0 不读 y 旧值 | 同（§2.1.5） |
| 非法 trans 返回码 | CUBLAS_STATUS_INVALID_VALUE | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 统一口径，§2.3 说明） |
| stride 约束 | stride > 0、相邻 batch 不重叠 | 同（§2.4 stride 行、§2.5 stride 合法性行） |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口：Host 校验 + tiling，`<<<>>>` 直调 AIV kernel，handle 绑定 stream 异步执行，§2.2/§2.5）。
- 代码落位：`blas/gemv_strided_batched/arch22/gemv_strided_batched_{host,kernel}.cpp + gemv_strided_batched_kernel.h + gemv_strided_batched_tiling_data.h`（复用 2026-11-23 已合入四件套，在其上做契约对齐，不新建文件）。
- 公共头：`include/cann_ops_blas.h` L311-337 **5 条声明均已存在，无需修改**（grep 核验）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_*`（11-23 合入） | ★ 主基线 | dtype 模板分发（5 变体）、batch 均分切核（maxPerCore/usedCoreNum/batchTail）、N 路列流式 / T 路逐列点积、FP32 域计算与收窄、strided 访问与负步长折算、**stride 直接寻址**（基址+i·stride 纯地址算术） | host 校验层按任务书逐条核对（OP_C ✓ quick return 次序 ✓ stride>0 ✓ 均已合规；alpha=0 短路/条件判空/strideA≥lda·n 口径开发会话终验）；README 补 A2/A3 |
| `blas/gemv_strided_batched/arch35/gemv_strided_batched_host.cpp` | ◆ 任务书指定参考 | arch35 的 stride 偏移、变体分派与缩放逻辑 | arch22 沿用现网方案，不动 arch35 |
| `blas/gemv_batched/arch22/`（姊妹任务同族） | ◆ 对照 | 指针数组形态的工程对称（kernel 结构同源） | batch 定位层差异（指针数组 8B 读 vs 地址算术） |
| `test/gemv_strided_batched/` 工程 | 既有 | 通用 gemv_strided_batched_test 工程、golden.h/param.h | 注册 arch22 目标；csv 按自验包口径核对 |

### 接口规格

签名与任务书 §2.3 逐字一致（= `include/cann_ops_blas.h` L311-337 既有声明）：

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| trans | op(A) 运算类型 | attr | int | {N,T,C}，实数 C≡T；其它 → `INVALID_ENUM` | 标量 |
| m | A 的行数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 quick return | 标量 |
| n | A 的列数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 quick return | 标量 |
| alpha | 标量乘数（FP32） | scalar | FLOAT32 | 任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| A | 第 0 个 batch 矩阵基地址（列主序） | tensor | 按变体 | batchCount>0∧m,n>0 时 nullptr → `INVALID_VALUE` | batchCount 个 lda×n（第 i 个 = A+i·strideA） |
| lda | A 前导维 | attr | int | ≥max(1,m)（与 trans 无关）；低于 → `INVALID_VALUE` | 标量 |
| strideA | 相邻 batch A 的元素偏移 | attr | int64 | **>0 且 ≥lda×n（不重叠）**；违反 → `INVALID_VALUE` | 标量 |
| x | 第 0 个 batch 向量基地址 | tensor | 按变体 | batchCount>0 且逻辑长度>0 时 nullptr → `INVALID_VALUE` | batchCount 个向量（第 i 个 = x+i·stridex） |
| incx | x 元素步长 | attr | int | 任意非零 int（含负）；0 → `INVALID_VALUE` | 标量 |
| stridex | 相邻 batch x 的元素偏移 | attr | int64 | >0 且不重叠；违反 → `INVALID_VALUE` | 标量 |
| beta | 标量乘数（FP32）；beta=0 不读 y | scalar | FLOAT32 | 任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| y | 第 0 个 batch 输出基地址（原地覆写） | tensor | 按变体 | batchCount>0 且逻辑长度>0 时 nullptr → `INVALID_VALUE` | batchCount 个向量（第 i 个 = y+i·stridey） |
| incy | y 元素步长 | attr | int | 任意非零 int（含负）；0 → `INVALID_VALUE` | 标量 |
| stridey | 相邻 batch y 的元素偏移 | attr | int64 | >0 且不重叠；违反 → `INVALID_VALUE` | 标量 |
| batchCount | batch 数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | 标量 |

> 变体 dtype 约定（§2.4）：S=FP32/FP32/FP32；HSH=FP16/FP16/FP16；HSS=FP16/FP16/FP32；TST=BF16/BF16/BF16；TSS=BF16/BF16/FP32；alpha/beta 恒 FLOAT32 Host 标量；uint16_t 仅承载位模式。

### 数学公式

```
y_i = alpha * op(A_i) * x_i + beta * y_i,   A_i = A + i·strideA, x_i = x + i·stridex, y_i = y + i·stridey
op(A_i) = A_i（trans=N，m×n）；A_i^T（trans=T 或 C，n×m，实数无共轭）
A_i 列主序恒为 lda×n；x_i 长度 = N?n:m，y_i 长度 = N?m:n
```

x_i/y_i 元素地址 = 基址 + i·stride + k·inc（负步长按 BLAS 反向遍历，起点折算 (1-count)·inc）。半精度/bf16 变体 FP32 域计算后按输出 dtype 收窄（§2.1.6）。特值：alpha=0 → 只做 y=beta·y（**不读 A/x**，Inf/NaN 不传播）；beta=0 → 不读 y 旧值（y 含 NaN 亦安全）。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现 5 个句柄式跨步批量 gemv 接口：单 AIV kernel 模板覆盖 5 组 dtype 组合，batch 由基址+stride 直接寻址（无指针数组读）、按 batch 均分切核；支持 trans N/T/C（C≡T）、任意合法前导维与非零（含负）步长、stride>0 且不重叠的跨步批量组织；精度按输出 dtype 查生态混合容差表（golden = 逐 batch cblas_sgemv），性能达 GPU 标杆 0.8×（单 case 判定）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 跨步批量 = 逐 batch 独立 gemv + 基址偏移定位；可完全复用仓内已合入 arch22 实现（11-23）的 N 路列流式 / T 路逐列点积两路径与 batch 切核 |
| 2 | 接口原型 | §2.3 五签名（见上；stride 为 int64_t），返回 `aclblasStatus_t`；公共头 L311-337 已存在零改动 |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4（含 stride>0/不重叠约束） |
| 4 | 支持数据类型 | 5 组组合（S/HSH/HSS/TST/TSS），alpha/beta 恒 FP32 |
| 5 | 目标芯片与 arch 映射 | A2/A3（910b 系）→ arch22；构建 `--soc=ascend910b3`；arch35 已有实现不动 |
| 6 | 精度要求 | §3.2：S/HSS/TSS 按 FLOAT32（rtol=atol=2^-13）、HSH/TST 按 FLOAT16/BFLOAT16（2^-10/2^-7）查 mixed 表；ratio ≥0.99；max_abs 1e-2（FP32）/1e-1（半精度）或 32 ULP；golden = 逐 batch cblas_sgemv（FP64 累加收窄或 FP32 对照） |
| 7 | 性能预期 | §3.3：单 case NPU Task Duration ≤1.25×gpu_ms（≥0.8× GPU）；gpu_baseline 每变体 200 条实测——S max=0.3245ms/avg=0.0198ms，HSH 0.2290/0.0174，HSS 0.2315/0.0175，TST 0.2292/0.0171，TSS 0.2307/0.0175 |
| 8 | 代码架构选型 + 可行性 | AIV 单 kernel 模板（dtype 模板 + tiling.dtype 分发）+ batch 均分切核 + stride 直接寻址；已排除死路：指针数组中转（多一次间接读，跨步形态不需要）、双 kernel 拆 N/T（单发射更优）；11-23 已合入同架构实现 = 可行性旁证 |
| 9 | 评测来源 | 任务书自验包 test_cases/（5 变体各 1200 条 csv = 精度 1000 + 性能 200；gpu_baseline 各 200 行） |
| 10 | 契约对齐层 | 相对 11-23 已合入实现：grep 实证 OP_C 接受 ✓、lda/stride>0 校验 ✓、INT_MIN 防护 ✓、quick return 位于结构校验后 ✓；开发会话终验项 = alpha=0 短路、指针条件判空次序、strideA≥lda·n 重叠判定口径、README A2/A3 产品行 |
| 11 | 特值与退化 | batchCount=0/m=0/n=0 → SUCCESS no-op（A/x/y 可 null，不下发 kernel）；alpha=0 → y=beta·y 不读 A/x；beta=0 → 不读 y（NaN 安全）；负步长反向遍历 |
| 12 | workspace | 无（stride 地址算术直接寻址，UB 内完成全部暂存） |

# 详细设计

## 算子分析

### 数学公式

逐 batch（i = 0..batchCount-1）独立计算，无 batch 间交互：

```
N 路（trans=OP_N，op(A_i) 为 m×n）：
    y_i[r] = alpha * Σ_c A_i[r + c*lda] * x_i[c] + beta * y_i[r],   r = 0..m-1, c = 0..n-1
T/C 路（op(A_i) = A_i^T 为 n×m，实数无共轭）：
    y_i[c] = alpha * Σ_r A_i[r + c*lda] * x_i[r] + beta * y_i[c],   c = 0..n-1, r = 0..m-1
其中 A_i = A + i·strideA（列主序恒 lda×n），x_i = x + i·stridex，y_i = y + i·stridey
```

x_i 元素地址 = x_i 基址 + k·incx，y_i 同理（负步长按 BLAS 反向遍历，起点折算 (1-count)·inc）。计算域：FP16/BF16 输入先 cast FP32，乘加在 FP32 完成，按 y 输出 dtype 收窄（§2.1.6）。特值语义：alpha=0 → 只做 y=beta·y（不读 A/x）；beta=0 → 不读 y 旧值（NaN 安全）。

### 支持数据类型

| 变体 | A | x | y | alpha/beta | kernel 模板实例 | dtype 码 |
|---|---|---|---|---|---|---|
| S | FP32 | FP32 | FP32 | FLOAT32 | GemvStridedBatchedKernel<float,float> | 1 |
| HSH | FP16 | FP16 | FP16 | FLOAT32 | GemvStridedBatchedKernel<half,half> | 0 |
| HSS | FP16 | FP16 | FP32 | FLOAT32 | GemvStridedBatchedKernel<half,float> | 2 |
| TST | BF16 | BF16 | BF16 | FLOAT32 | GemvStridedBatchedKernel<bfloat16_t,bfloat16_t> | 3 |
| TSS | BF16 | BF16 | FP32 | FLOAT32 | GemvStridedBatchedKernel<bfloat16_t,float> | 4 |

dtype 码沿仓内已合入实现取值（host 五入口 → 模板实现 + dtype 码）。

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m, n | ≥ 0（int） | m=0 或 n=0 → quick return：SUCCESS、不启动 kernel、A/x/y 允许 nullptr（§2.5） |
| batchCount | ≥ 0（int） | 0 → no-op：SUCCESS、不下发 kernel、基址允许 null（§2.1.7） |
| lda | ≥ max(1, m)，与 trans 无关 | 低于下界 → INVALID_VALUE |
| strideA | > 0 且 ≥ lda×n（相邻 batch A 不重叠） | strideA ≤ 0 或 < lda×n → INVALID_VALUE |
| stridex/stridey | > 0 且 ≥ 各 batch 向量逻辑跨度（1+(len-1)·abs(inc)） | ≤ 0 或跨步重叠 → INVALID_VALUE |
| incx/incy | 任意非零 int（含负；INT_MIN 防护） | 0 → INVALID_VALUE |
| 覆盖档位（§3.5） | m,n ∈ {0..4096 全档}；batchCount ∈ {0,1,2,3,8,16,64,256}；stride 紧凑+带间隙 | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

单 AIV kernel 模板 + host 侧 batch 均分切核；5 个 C 入口收敛到一个模板实现；batch 由 **stride 地址算术**直接定位（无指针数组间接读）：

```
aclblas{S,HSH,HSS,TST,TSS}gemvStridedBatched   (5 个 C API)
        └─ GemvStridedBatchedImpl<TIn,TOut>(dtype)   ← host：校验→quick return→切核→tiling→发射
                └─ gemv_strided_batched_kernel_do(...)  ← 直调 AIV kernel（usedCoreNum 核）
                        └─ gemv_strided_batched_kernel_impl
                                ├─ trans=0 → ProcessNormalBatch（N 路列流式）
                                └─ trans=1 → ProcessTransBatch（T/C 路逐列点积）
```

**契约对齐层**（相对 11-23 已合入实现，grep 实证的合规项与待终验项）：

| # | 项 | 现状（arch22 host） | 契约目标 |
|---|---|---|---|
| 1 | OP_C 接受 | trans ∈ {N,T,C} 校验 ✓（L36-38） | 保持（C 折算 trans=1） |
| 2 | quick return 次序 | m/n/batchCount=0 在结构校验后 ✓（L81） | 保持 |
| 3 | stride 正性 | strideA/stridex/stridey ≤ 0 → INVALID_VALUE ✓（L40-42） | 保持 + 补 strideA ≥ lda×n 的可判定重叠口径终验 |
| 4 | alpha=0 短路 | 待终验（kernel 内分支/独立 scale 路径） | alpha=0 不读 A/x |
| 5 | README | 产品行核对 | 补 Atlas A2/A3 支持情况 |

数据流（N 路，列流式、y 驻 UB）：

```
for i ∈ 本核 batch 区间:  A_i = A + i·strideA（地址算术）
    GM: y_i --(beta≠0 时, strided)--> UB yIn --cast FP32--> acc := beta*acc
    for 每列 c: GM x_i[c] (strided) --> 标量 scale = alpha*x32[c]
                GM A_i 列段 [c*lda+mOff, mCnt] --> aIn --cast--> a32
                acc += a32 * scale
    UB: acc --cast 收窄--> yIn --(strided)--> GM: y_i
```

数据流（T/C 路，逐列点积）：

```
for i ∈ 本核 batch 区间:
    for 每输出列 c: dot = Σ_r（m 分段: A_i 列段 × x_i 段 → Mul → ReduceSum 累加）
                    acc[c] += alpha * dot（先 beta·y_i 初始化）
    GM: y_i <--(strided, 收窄后)-- acc
```

#### host 侧设计

- **分核策略**：`coreNum = GetAivCoreCount()`；`batchPerCore = ⌈batchCount/coreNum⌉`；`usedCoreNum = ⌈batchCount/batchPerCore⌉`（收缩空核）；`batchTail` 尾核余块。每核连续 batch 区间，无核间通信。
- **数据分块和内存优化策略**：m/n 按 TILE 分段（沿仓内常量口径）；UB 布局 aIn/a32/xIn/x32/yIn/acc/work + strided 窗口 buffer（仅非单位步长）；MTE2/V/S HardEvent 软流水；batch 定位零间接读（stride 纯地址算术）。
- **tilingkey 规划策略**：直调工程分支承载于 tiling 字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| dtype | 0/1/2/3/4 | HSH/S/HSS/TST/TSS 模板分发 |
| trans | 0 / 1 | N 路 / T·C 路（host 侧 C→1 折算） |
| alpha / beta | ==0.0F 特值 | kernel 隐式分支：alpha=0 跳过 A/x；beta=0 以 0 初始化 acc 不读 y |
| strideA/stridex/stridey | int64 | batch 基址偏移 |
| usedCoreNum/batchPerCore/batchTail | 切核三元组 | 核间 batch 划分 |

- **数据检测**（校验顺序逐条对齐 §2.5；结构合法后才执行 quick return）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | trans ∉ {N,T,C} | INVALID_ENUM |
| 3 | m<0 或 n<0 | INVALID_VALUE |
| 4 | lda < max(1,m) | INVALID_VALUE |
| 5 | incx==0 或 incy==0（含 INT_MIN 防护） | INVALID_VALUE |
| 6 | strideA ≤ 0 或 strideA < lda·n；stridex/stridey ≤ 0 或小于向量逻辑跨度 | INVALID_VALUE |
| 7 | batchCount < 0 | INVALID_VALUE |
| 8 | alpha/beta == nullptr | INVALID_VALUE |
| 9 | batchCount>0 且 m,n>0 时 A/x/y 基址 == nullptr | INVALID_VALUE |
| 10 | 全过后：batchCount==0 或 m==0 或 n==0 → SUCCESS（不下发 kernel） | — |

  kernel 发射失败 → `EXECUTION_FAILED` 记日志（沿 11-25 上游修正的发射状态上报口径）。

#### kernel 侧设计

- **Init 阶段**：blockIdx → 本核 batch 区间；按 dtype 布局 UB buffer；GM 绑定以第 0 batch 基址 + span 上界（batchCount·stride + 单 batch 跨度，uint64 防溢出）。
- **Process 阶段**：dtype 分发 → 逐 batch：`baseA = A + i·strideA`（int64 地址算术）→ 按 trans 分派 N/T 路径（两路径与 gemv_batched 同构：InitYAcc（beta=0 → Duplicate(0)）/ alpha≠0 时累加 / 收窄写回，负步长起点折算）。
- **workspace 布局表**：无 workspace；batch 定位零间接读；指针不落 GM。

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| batch 均分切核 | 满核原则 + usedCoreNum 收缩 | 全 batch 并行（待 msprof 回填） |
| stride 直接寻址 | 基址+i·stride 地址算术 | 相对指针数组形态省每 batch 8B 间接读 |
| N 路列流式 | y 驻 UB、A 列段流过 | 大 m/n 超 UB 规模可支撑 |
| FP32 计算域 | half/bf16 cast 后乘加 | 精度达 mixed 标准 |
| 标量特值分支 | alpha/beta 特值免读对应操作数 | 特值用例省带宽 |
| 已排除路线 | 指针数组中转（跨步形态不需要）；N/T 双 kernel 拆分（多一次发射） | — |

判据逐 case 独立：NPU ≤ 1.25×gpu_ms；gpu_baseline 每变体 200 条实测（S max=0.3245ms / HSH 0.2290 / HSS 0.2315 / TST 0.2292 / TSS 0.2307；avg 0.017-0.020ms，cublas 对应变体 kernel-only）。达标以设备 msprof 实测为准（设计阶段不判决）。

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
| 校验层级 | 全部结构参数（含 stride 三元组）host 侧校验；不扫描 Device 数据 |
| stride 合法性 | strideA/stridex/stridey > 0；相邻 batch 区域不重叠（A 最小间距 strideA ≥ lda×n；x/y 按逻辑跨度）；违反 → INVALID_VALUE |
| 非连续支持 | incx/incy 非零跨步（含负）；A 按 lda 带填充；batch 间允许间隙（stride 带间隙） |
| broadcast | 不支持 |
| dynamic shape | 天然支持（全部结构参数运行时入参驱动 tiling） |
| 原地语义 | y 原地覆写；y 可写区与 A/x 可读区不重叠、各 batch y 区域不重叠由调用方保证；不返回视图 |
| 确定性 | 不保证逐位一致；按 §3.2 阈值判定 |
| 空 Tensor | batchCount=0 / m=0 / n=0 → SUCCESS no-op，不引用 A/x/y |
| 0 维 Tensor | 不涉及 |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden = 逐 batch Netlib cblas_sgemv（对 batch i 以 A+i·strideA / x+i·stridex / y+i·stridey 独立调用；FP64 累加后按输出 dtype 收窄落盘，或 FP32 对照，差异写自测报告）；判定 = `|actual−golden| ≤ atol + rtol·|golden|` 逐 batch 逐元素；S/HSS/TSS 按 FLOAT32（2^-13）、HSH/TST 按 FLOAT16/BFLOAT16（2^-10/2^-7）查 mixed 表；ratio ≥0.99 且 max_abs ≤ 1e-2/1e-1 或 32 ULP 时用例通过 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel：msprof op Task Duration（warmup5+10 均值）单 case ≤1.25×gpu_ms（≥0.8× GPU）；GTest `[PERF] avg_us`（aclrtEvent）仅交叉验证（verify_performance.py） | §3.3；test_cases/*/gpu_baseline.csv |

## 测试概览

- 用例规模：精度全量 5000（5 变体各 1000）+ 性能子集 1000（5 变体各 200，按变体 × 规模档分层）；csv 实测每变体 1200 行（1000+200）。
- 覆盖：S/HSH/HSS/TST/TSS 全变体；trans N/T/C 全覆盖（**OP_C 与 OP_T 结果一致**——实数交叉验证点）；m/n 至 4096 全档；batchCount {0,1,2,3,8,16,64,256}（性能 1/8/64）；incx/incy ±1/±2/±3 + 大步长抽样 {7,8,16,31,64,127}；stride 紧凑+带间隙；负向（非法 trans/负值/零步长/非法 lda/**stride≤0/相邻 batch 重叠**/空基址）。
- 特殊场景：batchCount=0；m=0/n=0 quick return；beta=0 且 y 含 NaN；alpha=0 且 x 含 NaN（不读 A/x）；半精度/bf16 变体 Inf/NaN 传播；FP16/BF16 输入按 dtype 生成覆盖 Inf/NaN。
- 性能口径：msprof op（正式）+ GTest `[PERF] avg_us`（交叉验证）；性能子集显式保留 alpha/beta 列（约 12% 非默认标量）。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + `test/gemv_strided_batched/arch22/` ATK 工程）。

## 兼容性分析

- 公共头零改动（L311-337 既有 5 条声明），无接口面波及。
- 在 11-23 已合入四件套上原位核对/修改（host 校验层对齐项最小化改动），不新增文件；arch35 与 gemv_batched 路径零触碰；注意上游 11-25 仍在活跃维护本文件（east_yang 发射状态修正），改动保持最小面以降低合并冲突。
- `test/gemv_strided_batched/CMakeLists.txt` 注册 arch22 目标；既有 arch35 测试不动。
- README：`blas/gemv_strided_batched/README.md` 核对补 A2/A3 产品支持情况。
- 无既有合法调用方行为变化（仅非法参数场景错误码口径核对）。

