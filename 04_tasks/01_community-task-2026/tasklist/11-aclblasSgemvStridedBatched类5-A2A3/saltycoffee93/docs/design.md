# aclblasSgemvStridedBatched 类算子设计文档（S/HSH/HSS/TST/TSS）

> 目标芯片：Atlas A2/A3（arch22，自验证 SOC=ascend910b3）｜CANN 9.1.0｜合入位置 `ops-blas/blas/gemv_strided_batched/arch22/`
> 任务书：`aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）
> 仓内现状基线：ops-blas master `55ef791`（2026-11-25 源码核验）。本文撰写时未上板，凡需 NPU 实测的结论均标注"待实测"，不含未经测试的性能数字。

# 需求背景

## 需求来源

- 任务名称/批次：11 月社区任务 —— aclblasSgemvStridedBatched 类算子开发（A2A3，5 个变体）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → 实现 `blas/gemv_strided_batched/arch22/`；测试 `test/gemv_strided_batched/arch22/`（任务书 §2.2、§5）。
- 验收流程：设计文档以 PR 形式提交至 cann-competitions 仓 tasklist，评审合入后进入开发与验收（任务书 §4）。

## 背景介绍

### 算子功能定位

跨步批量矩阵-向量乘算子族（5 个 API）：`y + i·stridey = alpha * op(A + i·strideA) * (x + i·stridex) + beta * (y + i·stridey)`（i = 0..batchCount−1，偏移以元素为单位）。与 gemvBatched 的差异仅在 batch 定位方式：stride 偏移替代 Device 指针数组。

| 变体 | A/x dtype | y dtype | 对标 |
|---|---|---|---|
| aclblasSgemvStridedBatched | FP32 | FP32 | `cublasSgemvStridedBatched` |
| aclblasHSHgemvStridedBatched | FP16 | FP16 | `cublasHSHgemvStridedBatched` |
| aclblasHSSgemvStridedBatched | FP16 | FP32 | `cublasHSSgemvStridedBatched` |
| aclblasTSTgemvStridedBatched | BF16 | BF16 | `cublasTSTgemvStridedBatched` |
| aclblasTSSgemvStridedBatched | BF16 | FP32 | `cublasTSSgemvStridedBatched` |

alpha/beta 恒为 FLOAT32（Host 标量）；FP16/BF16 以 `uint16_t` 位模式承载。

### 业界对标

| 项 | cuBLAS 对应变体 | 本算子 |
|---|---|---|
| 参数序列 | 15 参数（含 strideA/stridex/stridey） | 逐参数一致（§2.3） |
| trans | N/T/C，实数 C≡T | 同 |
| A 物理形状 | 恒为 lda×n，与 trans 无关；lda ≥ max(1,m) | 同（§2.1 第 3 条） |
| stride 约束 | strideA/stridex/stridey > 0 且相邻 batch 区域不重叠（strideA 最小 lda×n） | 同；违背 → `INVALID_VALUE` |
| 步长 | incx/incy 任意非零（含负，反向遍历） | 同；0 → `INVALID_VALUE` |
| n=0/m=0/batchCount=0 | quick return，不豁免结构校验 | 同（§2.5） |
| 非法 trans 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |

### 工程形态

- 工程模式：Kernel 直调（Host C API 校验/分派/tiling，经 handle 绑定的 stream 直调 NPU kernel，§2.2）。
- 仓内现状（master `55ef791` 源码核验）：

| 内容 | 现状 | 证据 | 本任务工作 |
|---|---|---|---|
| 公开声明 | 5 个均已存在 | `include/cann_ops_blas.h` 约 L311-337 | 零公共头改动 |
| arch22 实现 | **已合入**（5 变体） | `blas/gemv_strided_batched/arch22/gemv_strided_batched_host.cpp` 等 4 文件；提交 `41f26c9`（2026-11-23，5 变体 arch22 实现与单元测试）、修复 `1401419`（2026-11-25，kernel_do 返回发射状态） | 按任务书口径核对，测试用例对齐 |
| arch35 参考 | 已有 5 变体 | `blas/gemv_strided_batched/arch35/gemv_strided_batched_host.cpp` | 参考口径 |
| 测试工程 | arch22 已有 | `test/gemv_strided_batched/arch22/gemv_strided_batched_test.cpp` 等 | 用例按任务包口径对齐扩展 |
| README | 5 个变体章节已有 | `blas/gemv_strided_batched/README.md` L35-865 | 核对补充 |

- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_*`（已合入） | ★ 主对象 | 5 变体统一分派（`GemvStridedBatchedImpl`）、stride 偏移定位、FP32 累加收窄路径 | 仅做任务书口径核对 |
| `blas/gemv_batched/arch22/gemv_batched_*`（已合入） | ◇ 参考 | 同族批量 GEMV（指针数组版） | 不改动 |
| `blas/gemv_strided_batched/arch35/gemv_strided_batched_*` | ◇ 参考 | 变体分派与缩放逻辑 | 不复制代码 |

### 接口规格

5 个声明均已存在于 `include/cann_ops_blas.h`（约 L311-337），**无需新增或修改**（§2.3）。以 `aclblasSgemvStridedBatched` 为例（其余 4 个仅 dtype 不同）：

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

参数规格（逐行对齐 §2.4；dtype 按变体表）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 句柄，携带 stream | scalar | — | 非空；空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 标量 |
| trans | op(A[i]) 运算类型 | attr | int | {N, T, C}；C≡T；其它 → `INVALID_ENUM` | — |
| m / n | A[i] 行/列数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 quick return | — |
| alpha / beta | FLOAT32 标量乘数 | scalar | FLOAT32 | 非空；alpha=0 只做 `y=beta*y`；beta=0 不读 y 旧值 | 标量 |
| A | 第 0 个 batch 矩阵基地址 | tensor | 按变体 | batchCount>0 且 m,n>0 时非空 | 每 A[i] 为 lda×n（与 trans 无关） |
| lda | A[i] 前导维 | attr | int | ≥max(1,m)（与 trans 无关） | 标量 |
| strideA | 相邻 batch A 的元素偏移 | attr | int64 | >0 且相邻区域不重叠（最小 lda×n）；否则 `INVALID_VALUE` | 标量 |
| x / incx / stridex | x 数组基地址/步长/批量偏移 | tensor/attr | 按变体 | x 非空（逻辑长度>0）；incx≠0；stridex>0 不重叠 | 长度 n（N）或 m（T/C） |
| y / incy / stridey | y 数组基地址/步长/批量偏移（原地覆写） | tensor/attr | 按变体 | 同上 | 长度 m（N）或 n（T/C） |
| batchCount | batch 数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | — |

### 数学公式

```text
y + i·stridey = alpha * op(A + i·strideA) * (x + i·stridex) + beta * (y + i·stridey)，i = 0..batchCount−1
op(A_i)：N 为 m×n；T/C 为 n×m（实数 C≡T，不共轭）
A[i] 物理形状恒为 lda×n 列主序；相邻 batch 区域不重叠由 stride 语义保证。
FP16/BF16 变体：计算在 FP32 完成后按输出 y 的 dtype 收窄（§2.1 第 6 条）。
```

# 需求分析

## 需求描述

在 Atlas A2/A3（arch22）上提供 5 个句柄式跨步批量 GEMV 接口：stride 偏移批量语义、trans N/T/C、任意非零元素步长（含负）、batchCount 0..256；golden 按 cblas_sgemv 逐 batch 生成，输出按变体 dtype 分档判定；性能达 GPU 标杆 0.8 倍（kernel-to-kernel）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 逐 batch 独立 GEMV（带宽型 BLAS-2），batch 由 stride 偏移定位（地址 = 基址 + i·stride，规则连续可算术化）。已合入的 arch22 实现为 5 变体统一分派（`GemvStridedBatchedImpl`），与 gemv_batched（指针数组版）共享单 batch 计算结构 |
| 2 | 接口原型 | §2.3 签名（15 参数），返回 `aclblasStatus_t`；5 个声明均已存在，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"参数表；异常行为逐条对齐 §2.4（含 stride>0 不重叠约束） |
| 4 | 支持数据类型 | S 全 FP32；HSH/TST 全 FP16/BF16；HSS/TSS 输入 FP16/BF16、输出 FP32；alpha/beta 恒 FP32 |
| 5 | 目标芯片与 arch 映射 | A2（`ascend910b*`）与 A3（`ascend910_93*`）→ arch22（`build.sh` 的 SOC→arch 映射，L164）；arch35 已有实现，不在本任务范围 |
| 6 | 精度要求 | §3.2：golden=`cblas_sgemv` 逐 batch（float64 累加按输出 dtype 收窄）；S/HSS/TSS 按 FLOAT32 档（2^-13、1e-2 或 32ULP），HSH/TST 按 FLOAT16/BF16 档（2^-10、1e-1 或 32ULP），ratio≥0.99 |
| 7 | 性能预期 | §3.3：单 case NPU kernel 平均耗时 ≤ `gpu_ms`/0.8；口径 `msprof op` Task Duration，warmup 5 + 10 样本；5 变体各 200 条性能子集 |
| 8 | 代码架构选型 + 可行性 | arch22 已合入 5 变体统一实现（`41f26c9`，2026-11-23；`1401419` 修复发射状态）：5 个入口 → 统一 `GemvStridedBatchedImpl` → 批量 kernel，stride 偏移为纯算术定位。本任务以口径核对为主，测试用例对齐任务包（每变体 1200 条）。NPU 侧达标性**待实测** |
| 9 | 评测来源 | 任务包 `test_cases/`（5 个子目录）：每变体精度 1000 条 + 性能 200 条，`gpu_baseline.csv` 全回填；CSV 安装到 `test/gemv_strided_batched/<arch>/` 由通用 `gemv_strided_batched_test` 加载 |

# 详细设计

## 算子分析

### 数学公式

见"接口规格 · 数学公式"节。BLAS-2 带宽型：每 batch 的主要流量为 A[i]（lda×n，按 trans 决定行/列遍历方向）与 x/y 向量；batch 定位为纯算术偏移（基址 + i·stride），无需指针数组间接层。

### 支持数据类型

| 项 | 类型 | 说明 |
|---|---|---|
| A/x/y | FP32 / FP16 / BF16（按变体表） | uint16_t 位模式承载 FP16/BF16 |
| alpha/beta | FLOAT32 | Host 标量 |
| 计算精度 | FP32 累加后按输出 dtype 收窄 | 5 变体统一 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m/n | ≥0 | 0 为合法 quick return（结构校验后） |
| batchCount | ≥0 | 0 为合法 no-op |
| lda | ≥max(1,m)，与 trans 无关 | 低于下界 → INVALID_VALUE |
| incx/incy | 非零（可负） | 0 → INVALID_VALUE |
| strideA/stridex/stridey | >0 且相邻 batch 区域不重叠（strideA ≥ lda×n） | 违背 → INVALID_VALUE |

## 算子实现

### 实现方案

#### 总体架构与数据流

已合入的 arch22 实现为 **5 变体统一入口 + 批量 kernel**（`gemv_strided_batched_host.cpp`）：

| 变体路径 | 触发条件 | 职责 |
|---|---|---|
| quick return | batchCount=0 或 m=0/n=0（结构校验后） | host 直接返回 |
| y = beta*y 快路 | alpha=0 | 仅按 incy 步长缩放各 y[i] |
| 批量 GEMV 主路径 | 其余 | stride 算术定位 + batch×分核二维切分，逐 batch GEMV，FP32 累加按输出 dtype 收窄 |

数据流：

```text
第 wid 号工作项 = (batch i, 输出区间)：基址 A + i·strideA（x/y 同理，纯算术定位）
GM: A[i]（lda×n 列主序，按 trans 决定遍历方向）, x[i]（步长 incx）─[FP32 乘累加]─> 部分结果
GM: y[i] ─[y = alpha·(·) + beta·y_old，按变体 dtype 收窄，步长 incy 写回]─> GM: y[i]
```

#### host 侧设计

**分核策略**：工作项 = batchCount × 输出向量分段，按 AIV 核数线性分派；stride 连续定位使相邻工作项的 A[i] 访问天然局部化。

**tilingkey 规划策略**：无 tilingkey 宏分支，5 变体经统一 tiling 结构传参（变体号决定读写 dtype 转换）。

**数据检测**（校验顺序逐条对齐 §2.5）：

1. handle 非空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. trans ∈ {N, T, C} → 否则 `INVALID_ENUM`
3. m/n ≥ 0 → 否则 `INVALID_VALUE`
4. lda ≥ max(1,m) → 否则 `INVALID_VALUE`
5. incx/incy ≠ 0 → 否则 `INVALID_VALUE`
6. batchCount ≥ 0 → 否则 `INVALID_VALUE`
7. alpha/beta 非空 → 否则 `INVALID_VALUE`
8. **quick return**：batchCount=0 或 m=0 或 n=0 → SUCCESS（A/x/y 可空）
9. A/x/y 非空（逻辑长度>0）→ 否则 `INVALID_VALUE`
10. strideA/stridex/stridey > 0 且相邻区域不重叠 → 否则 `INVALID_VALUE`
11. alpha=0 → 仅 `y = beta*y`，不读 A/x

#### kernel 侧设计

```text
[AIV 批量 GEMV kernel（按变体 dtype 实例化）]
 Init   ：解析 tiling → 求本核 (batch, 输出区间) 工作项
 Process：读 A[i] 行/列（trans 决定）与 x[i] → FP32 乘累加
          → y = alpha·(·) + beta·y_old → 按输出 dtype 收窄，步长 incy 写回
```

文件落位（已合入，本任务按口径核对）：

| 文件 | 内容 |
|---|---|
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_host.cpp` | 5 个入口 + `ValidateGemvSbParams` + `GemvStridedBatchedImpl` |
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_kernel.cpp/.h`、`gemv_strided_batched_tiling_data.h` | 批量 GEMV kernel 与 tiling |
| `test/gemv_strided_batched/arch22/` | 通用 `gemv_strided_batched_test` + CSV（复用顶层 golden/param） |

#### 性能优化要点

本文未上板，以下为设计依据与规划，实测数据在性能阶段以 `msprof op` 回填。由任务包 `gpu_baseline.csv` 算出的 NPU 允许耗时（`gpu_ms`/0.8，括号内为 `gpu_ms`）：

| 代表 case（sgemv_strided_batched） | trans | m×n×batch | 允许耗时 |
|---|---|---|---|
| sgemv_strided_batched-base-200 | N | 64×32×64 | 0.012 ms（0.010） |
| sgemv_strided_batched-base-199 | C | 17×3×16 | 0.012 ms（0.010） |
| sgemv_strided_batched-base-198 | T | 3×1023×8 | 0.013 ms（0.010） |

| 优化项 | 依据 | 状态 |
|---|---|---|
| 单 launch 批量分派（batch, 输出段） | stride 纯算术定位，无指针数组间接层 | 已合入实现沿用 |
| A 列主序 + trans 双向遍历 | N 按行读（合并访问），T/C 按列读（合并访问 x） | 沿用 |
| FP32 累加统一 | 5 变体共用一套乘累加逻辑，收窄在最末端 | 沿用 |
| 未采用：Cube 通道 | BLAS-2 带宽型，Cube 无收益 | 排除 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练/推理系列产品（`ascend910b*`，arch22） | ✓ 本任务目标；自验证设备 Atlas 800T A2（910B3） |
| Atlas A3 训练/推理系列产品（`ascend910_93*`，arch22） | ✓ 本任务目标（§3.1：自验证覆盖一种款型即可） |
| 昇腾 950PR / 950DT（arch35） | 已有实现（`blas/gemv_strided_batched/arch35/`），不在本任务范围 |

## 算子约束限制

逐条对齐 §2.5：

- 参数合法性与校验层级：host 入口按上表顺序校验；各 batch 的 A/x/y 区域不重叠由 stride 语义与调用方保证；不对 Device 张量做元素级扫描。
- 非连续 Tensor：x/y 支持 incx/incy 非零跨步（含负，反向遍历）；A 按 lda 列主序带填充；不使用 AutoContiguous/ViewCopy。
- broadcast：不支持。
- dynamic shape：不要求，m/n/lda/incx/incy/batchCount 为运行时入参。
- 原地与视图语义：y[i] 原地覆写；相邻 batch 区域不重叠。
- 确定性计算：不要求。
- 空 Tensor：batchCount=0、m=0/n=0 合法 quick return（A/x/y 可空）。
- 0 维 Tensor：不涉及。
- 异步执行：kernel 在 handle 绑定的 stream 上异步执行。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度逐元素与用例判定 | `\|actual − golden\| ≤ atol + rtol·\|golden\|` 记为 matched；用例须同时满足 matched_ratio ≥ 0.99 与 max_abs_error ≤ 限值。S/HSS/TSS 输出按 FLOAT32 档（2^-13、1e-2 或 32ULP）；HSH/TST 按 FLOAT16/BF16 档（2^-10、1e-1 或 32ULP） | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md`（pin `28a1ce52`） |
| 精度 golden | `cblas_sgemv` 逐 batch（float64 累加按输出 dtype 收窄落盘） | 任务书 §3.1、§3.2 |
| 性能 | 单 case：NPU kernel 平均耗时 ≤ `gpu_ms` / 0.8 | 任务书 §3.3.2 |
| 性能口径 | `msprof op` 的 kernel task duration（正式判据）：warmup 5 次不计入，之后 10 次有效样本取算术平均；GTest `[PERF] avg_us` 仅作交叉验证 | 任务书 §3.3.1、§3.3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

## 测试概览

任务包 5 个子目录（sgemv/hsh/hss/tst/tss_gemv_strided_batched），每变体精度 1000 条 + 性能 200 条（合计 5000 + 1000）：

| 维度 | 覆盖 |
|---|---|
| 变体 | S / HSH / HSS / TST / TSS 全覆盖 |
| trans | N / T / C 全覆盖 + 非法枚举负向 |
| m/n | 尺寸池 0→4096，负值负向 |
| alpha/beta | 特殊值 0/1/−1/±1e4 + 随机值；beta=0 且 y NaN、alpha=0 且 x NaN |
| lda | 紧凑 + +1/+8/+64 padding，非法负向 |
| incx/incy | ±1/±2/±3 + 大步长 {7,8,16,31,64,127}（含负），零步长负向 |
| strideA/stridex/stridey | >0 不重叠；含最小间距（strideA = lda×n）与跨步重叠负向 |
| batchCount | 0、1、2、3、8、16、64、256；性能子集 1/8/64 |
| H 特殊 | FP16/BF16 Inf/NaN 传播、收窄边界 |

- 测试工程：`test/gemv_strided_batched/arch22/` 已有通用 `gemv_strided_batched_test` + CSV；任务包各变体 CSV 安装到 `test/gemv_strided_batched/<arch>/` 后加载。
- 复现入口：`bash build.sh --ops=sgemv_strided_batched --soc=ascend910b3`；任务包 `verify_accuracy.py` / `verify_performance.py` 按 README 步骤执行。
- 负向与专项重点：stride ≤ 0 或跨步重叠负向、batchCount=0（不引用数据）、半精度 Inf/NaN 传播、OP_C 与 OP_T 结果一致。

## 兼容性分析

- 公共头 `include/cann_ops_blas.h` **零改动**：5 个声明均已存在（约 L311-337）。
- arch22 实现与测试已合入（`41f26c9`/`1401419`），本任务为口径核对与用例对齐；arch35 实现与 `blas/gemv_batched/`（指针数组版）零改动、零回归风险。
- `blas/gemv_strided_batched/README.md` 的 5 个变体章节（L35-865）按核对结果补充 arch22 支持情况。

