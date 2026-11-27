# aclblasSgemvStridedBatched 类算子设计文档（Atlas A2/A3）

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：S=FP32；HSH/HSS=FP16；TST/TSS=BF16（半精度入参以 `uint16_t` 承载位模式）；alpha/beta 恒为 FLOAT32
> 合入位置：`ops-blas/blas/gemv_strided_batched/arch22/`；测试：`ops-blas/test/gemv_strided_batched/arch22/`
> 任务书：`aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——gemvStridedBatched 类 5 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemv_strided_batched/arch22/`；测试代码 → `test/gemv_strided_batched/arch22/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入（§4）。

## 背景介绍

### 算子功能定位

跨步批量矩阵-向量乘算子族（5 个 API）：`y + i·stridey = alpha·op(A + i·strideA)·(x + i·stridex) + beta·(y + i·stridey)`，i = 0..batchCount−1，偏移以元素为单位。与指针数组版 gemvBatched 的区别：batch 间地址由 **stride 偏移**描述（允许带间隙），基址只有一套。

| 变体 | A | x | y | alpha/beta |
|---|---|---|---|---|
| S | FP32 | FP32 | FP32 | FLOAT32 |
| HSH | FP16 | FP16 | FP16 | FLOAT32 |
| HSS | FP16 | FP16 | FP32 | FLOAT32 |
| TST | BF16 | BF16 | BF16 | FLOAT32 |
| TSS | BF16 | BF16 | FP32 | FLOAT32 |

### 业界对标

- 对标接口：cuBLAS `cublasSgemvStridedBatched` / `cublasHSHgemvStridedBatched` / `cublasHSSgemvStridedBatched` / `cublasTSTgemvStridedBatched` / `cublasTSSgemvStridedBatched`（数学语义参考 Netlib `sgemv`）。
- 语义要点：A[i] **恒为 lda×n 列主序**（lda ≥ max(1,m)，与 trans 无关），转置只作用于 op(A_i)；`strideA ≥ lda×n` 为最小间距；实数算子 `OP_C ≡ OP_T`；alpha/beta 恒为 FLOAT32 Host 标量；半精度/bf16 变体计算在 FP32 完成后按输出 dtype 收窄；strideA/stridex/stridey 为 int64_t 且必须 > 0。
- 有意差异：非法 trans 返回 `ACLBLAS_STATUS_INVALID_ENUM`（cuBLAS 返回 INVALID_VALUE）。

### 工程形态

- 工程模式：Host C API + Ascend C kernel 直调（handle 绑定 stream，§2.2）。
- 代码落位：`blas/gemv_strided_batched/arch22/` 新建 `gemv_strided_batched_host.cpp` / `gemv_strided_batched_kernel.cpp`（当前仅有 arch35）。
- 仓内近亲：`blas/gemv_strided_batched/arch35/gemv_strided_batched_host.cpp` 已实现 5 变体，可参考 stride 偏移、变体分派与缩放逻辑。5 个声明均已在 `include/cann_ops_blas.h`（约 L311-337），无需修改。

### 接口规格

5 个签名与 cuBLAS 逐参数一致（S 版示例）：

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

HSH/HSS/TST/TSS 的 A/x/y 元素类型按变体替换为 `const uint16_t*` / `uint16_t*`（FP16/BF16 位模式承载），其余相同。

参数规格固定表（§2.4 逐行）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | - | - | 非空，否则 HANDLE_IS_NULLPTR | - |
| trans | op(A[i]) 类型 | attr int | - | {OP_N, OP_T, OP_C}；C≡T，非法→INVALID_ENUM | - |
| m/n | A[i] 行/列数 | attr int | - | ≥0；=0 合法 quick return | - |
| alpha/beta | FLOAT32 标量（Host） | scalar | FLOAT32 | 非空；beta=0 不读 y 旧值 | - |
| A/x/y | 各 batch 基地址 | tensor | 按变体 | 基址 host 校验；batch 区域不重叠 | A[i] lda×n（与 trans 无关） |
| lda | A[i] 前导维度 | attr int | - | ≥max(1,m)（与 trans 无关） | - |
| strideA/stridex/stridey | 相邻 batch 元素偏移 | attr int64 | - | >0；最小间距 strideA ≥ lda×n，重叠→INVALID_VALUE | - |
| incx/incy | x/y 内步长 | attr int | - | 任意非零 int（含负，反向遍历）；0→INVALID_VALUE | - |
| batchCount | batch 数 | attr int | - | ≥0；=0 合法 no-op | - |

### 数学公式

`y[i] = alpha·op(A[i])·x[i] + beta·y[i]`，其中 A[i] = A + i·strideA、x[i] = x + i·stridex、y[i] = y + i·stridey；`trans=OP_N` 时 op(A[i]) 为 m×n（x 长 n、y 长 m），否则为 n×m（x 长 m、y 长 n）。

# 需求分析（required）

## 需求描述

以 Ascend C 在 arch22 上实现 5 变体跨步批量 gemv：stride 线性偏移取代指针数组（基址 + batch id × stride 即可取址，无指针数组搬运开销），batch×行块二维分核，变体按 dtype 分派计算路径（FP32 累加后按 y dtype 收窄）；支持 N/T/C 全组合、任意非零步长与完整 quick return 语义；精度按输出 dtype 对应阈值判定，性能达 GPU kernel 标杆 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | batch 间无依赖完全并行；batch 地址 = 基址 + i·stride 线性公式（比指针数组版少一次数组读取）；gemv 计算量 = m×n FMA/batch |
| 2 | 接口原型 | 5 个声明已在 `include/cann_ops_blas.h` L311-337，无需改公共头 |
| 3 | 参数规格 | §2.4 固定表；stride 为 int64 元素偏移且 >0；FP16/BF16 以 uint16_t 承载 |
| 4 | 支持数据类型 | S=FP32；HSH/HSS=FP16；TST/TSS=BF16；y dtype 按变体（FP16/FP32/BF16/FP32） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → arch22（ascend910b3），落 `blas/gemv_strided_batched/arch22/` |
| 6 | 精度要求 | 按输出 dtype 执行 mixed_tolerance_standard：FP32 档 rtol=atol=2^-13、FP16 档 2^-10、BF16 档 2^-7，matched_ratio≥0.99（§3.2，可溯源） |
| 7 | 性能预期 | kernel-to-kernel：`msprof op` Task Duration ≤ `gpu_ms` × 1.25（0.8 倍 GPU 标杆，§3.3，可溯源） |
| 8 | 代码架构选型 | 单套 batch×行块二维分核骨架 + stride 线性取址 + 变体模板分派；可行性依据：arch35 五变体同构实现已在仓、gemvBatched（指针数组版）骨架可平移。已排除死路——逐 batch 多次 launch（下发开销大）、通用指针数组方案（本任务 stride 语义下无需间接层） |
| 9 | 评测来源 | golden：按变体以 FP32（float64 累加）参考计算后按 y dtype 收窄；性能 baseline=`gpu_baseline.csv`（cuBLAS 同名接口 kernel-only） |

# 详细设计（required）

## 算子分析

### 数学公式

`y[i] = alpha·op(A[i])·x[i] + beta·y[i]`；分量式（trans=N）：`y[i](r) = alpha·Σ_j A[i](r,j)·x[i](j) + beta·y[i](r)`（r = 0..m−1）；trans=T 时对 A[i] 按列方向累加。

### 支持数据类型

| 变体 | 计算 | 输出 |
| --- | --- | --- |
| S | FP32 FMA | FP32 |
| HSH/TST | uint16 拆包 → FP32 FMA | FP16 / BF16（round-to-nearest 收窄） |
| HSS/TSS | uint16 拆包 → FP32 FMA | FP32（免收窄） |

### 支持形状

| 参数 | 形状约束 | 退化 |
| --- | --- | --- |
| A[i] | lda×n 列主序（与 trans 无关），batch 间距 ≥ lda×n | m=0/n=0 quick return |
| x[i]/y[i] | 长度按 trans（x: n/m，y: m/n），步长可负 | alpha=0 不读 A/x；beta=0 不读 y 旧值 |
| batchCount | ≥0 | =0 合法 no-op |

## 算子实现

### 实现方案

#### 总体架构与数据流

```
host: 校验(handle→trans→m/n→lda→incx/incy→strideA/stridex/stridey→batchCount
      →alpha/beta→A/x/y 基址) → quick return 判定 → tiling → 单次 launch
数据流: 核内按 (基址 + batch_id×stride) 线性取址 → GM(A[i] 列块 + x[i])
        → UB 拆包(H/T 变体)/FP32 FMA → beta 折算 → GM(y[i]) 原地写回
```

| kernel 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| 主路径 | batchCount>0 且 m,n>0 且非 quick return | batch×行块二维并行 gemv |
| alpha 缩放路径 | alpha=0 且 beta≠1 | 仅 y=beta·y，不读 A/x |
| quick return | batchCount=0 或 m=0 或 n=0；（alpha=0 且 beta=1） | 不 launch |

#### host 侧设计

- **分核策略**：batch 维 × y 行块二维编址满核均分；大批次小矩阵时 batch 维主导，小批次大矩阵时行块网格主导；
- **数据分块和内存优化策略**：stride 三元组作为标量参数传入 kernel，核内线性取址（零间接层）；A[i] 列块搬运 + double buffer；H/T 变体拆包在 UB 内与搬运流水重叠，统一 FP32 累加；stride 连续等距时按 batch 维做向量化合并访问；
- **tilingkey 规划策略**：key 由 {变体(S/HSH/HSS/TST/TSS), trans(N/T), 步长单位性, 尾块标志} 编码，编译期展开拆包与 gather 分支；
- **数据检测**：按 §2.5 校验顺序逐参数校验（含 stride>0 与最小间距检查），任一结构参数非法立即返回；结构合法后 quick return；不对 Device 数据做元素级扫描。

#### kernel 侧设计

- Init：按全局核号解码 (batch id, 行块)，以 `基址 + batch_id×stride` 计算 A[i]/x[i]/y[i] 地址；
- Process：CopyIn（A[i] 列块、x[i]）→ Compute（FP32 FMA 行累加）→ CopyOut（beta 折算、按变体收窄 y dtype、原地写回）；
- workspace：无额外 GM workspace。

#### 性能优化要点（终态）

- 优化项→预期效果：stride 线性取址→无指针数组搬运、取址开销 O(1)；单 launch 二维分核→消除逐 batch 下发开销；FP32 累加→半精度路径精度与吞吐兼得。实测按 §3.3 口径（warmup 5、有效采样 10、msprof op）回填自测报告。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2（800T A2/910B3） | ✔（本任务目标） |
| Atlas A3 | ✔（本任务目标） |
| Atlas 950PR | arch35 已有实现，不涉及 |

## 算子约束限制

- A[i] 恒为 lda×n 列主序带填充布局；x/y 按步长语义访问（支持负步长）；batch 间由 stride 描述、允许间隙但不得重叠；不支持广播；
- y 原地覆写，各 batch 的 A/x/y 区域不重叠（调用方保证）；
- incx/incy 不支持 0；strideA/stridex/stridey 必须 >0；batchCount=0 与 m/n=0 为合法 quick return；
- 非确定性：浮点累加顺序随实现变化，验收按阈值判定；异步执行语义同 §2.5。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 按输出 dtype 执行 mixed_tolerance_standard：FP32 档 rtol=atol=2^-13（max_abs≤1e-2 或 32ULP）、FP16 档 2^-10（≤1e-1 或 32ULP）、BF16 档 2^-7（≤1e-1 或 32ULP）；matched_ratio≥0.99；golden 按 FP32（float64 累加）参考计算后按 y dtype 收窄 | §3.2 + mixed_tolerance_standard |
| 性能标准 | msprof op kernel Task Duration ≤ gpu_ms×1.25（cuBLAS 同名接口 kernel-only 基线）；[PERF] avg_us 仅交叉验证 | §3.3 |

## 测试概览

- 用例分类：精度全量（CSV 驱动，5 变体全覆盖）+ 性能子集（变体×规模档分层）+ 负向（非法枚举、m/n/lda/batchCount 越界、stride≤0、零步长、空基址）；
- 特殊场景：batchCount=0、m/n=0、alpha=0、beta=0 且 y=NaN、OP_C≡OP_T 一致性、负步长反向遍历、带间隙 stride（strideA > lda×n）、紧凑 stride（strideA = lda×n）、FP16/BF16 次正规与 Inf/NaN；
- 性能口径：msprof op 正式判据（warmup 5 + 采样 10 取平均）；
- 复现入口：`bash build.sh --ops=sgemv_strided_batched --soc=ascend910b3`，自验包 readme 步骤。

## 兼容性分析

- 5 个声明均已存在，不修改公共头；arch22 新建文件对 arch35 及其他产品线无影响；
- 测试在 `test/gemv_strided_batched/CMakeLists.txt` 注册 arch22 目标，独立编译单元；
- README.md 补充 5 变体的 Atlas A2/A3 支持说明，无兼容性风险。