# gemvStridedBatched 类 5 算子开发（A2/A3）设计文档

> 依据 cann-competitions design_template.md 结构编写；提交前请核对模板是否有更新。

# 需求背景（required）

## 需求来源

昇腾社区任务「11月社区任务-gemvStridedBatched类5算子开发（A2/A3）」，对标 cuBLAS `cublasSgemvStridedBatched` / `cublasHSHgemvStridedBatched` / `cublasHSSgemvStridedBatched` / `cublasTSTgemvStridedBatched` / `cublasTSSgemvStridedBatched`，基于 ops-blas 开源仓补齐 Atlas A2/A3（arch22）实现与测试。

## 背景介绍

### gemvStridedBatched 算子现状分析

跨步批量矩阵-向量乘（Strided Batched GEMV）是 BLAS 批量接口族的基础成员，数学定义：

```text
y + i*stridey = alpha * op(A + i*strideA) * (x + i*stridex) + beta * (y + i*stridey),  i = 0..batchCount-1
```

ops-blas 仓现状：

| 内容 | 现状 |
| --- | --- |
| `blas/gemv_strided_batched/arch35/` | Ascend 950PR 已实现 5 个精度变体（S/HSH/HSS/TST/TSS），含完整 tiling（转置路径 dotTile/outTile 二分搜索）与 UB 布局规划 |
| `blas/gemv_strided_batched/arch22/` | Atlas A2/A3（dav-2201）存在基础种子实现，但与任务书 §2.5 要求存在差距：① quick return 位于参数校验之前，违反「结构合法后才允许 quick return」的校验顺序；② 缺少转置路径的 dotTile/outTile 切分与 UB 缓冲规划，性能未对标 GPU 基线 |
| `include/cann_ops_blas.h` L311-337 | 5 个接口声明已存在，本任务不修改头文件 |
| `test/gemv_strided_batched/arch35/` | 通用测试工程可复用，arch22 目标需在 `test/gemv_strided_batched/CMakeLists.txt` 注册 |

接口参数序列与 cuBLAS 逐参数一致（handle、trans、m、n、alpha、A、lda、strideA、x、incx、stridex、beta、y、incy、stridey、batchCount），A 恒为 lda×n 列主序、`lda ≥ max(1, m)`（与 trans 无关，依据 cuBLAS 13.5 与任务书 11-24 修订口径）。

### 变体 dtype 约定

| 变体 | A | x | y | alpha/beta |
| --- | --- | --- | --- | --- |
| S | FP32 | FP32 | FP32 | FLOAT32 |
| HSH | FP16 | FP16 | FP16 | FLOAT32 |
| HSS | FP16 | FP16 | FP32 | FLOAT32 |
| TST | BF16 | BF16 | BF16 | FLOAT32 |
| TSS | BF16 | BF16 | FP32 | FLOAT32 |

FP16/BF16 以 `uint16_t` 位模式承载；半精度变体在 FP32 域完成累加后按输出 dtype 收窄。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 上使用 Ascend C 实现 5 个 `aclblas*GemvStridedBatched` 接口，通过 handle 绑定 stream 直调 NPU kernel，精度与性能达到任务书验收标准。

## 需求拆解

1. 5 个精度变体全覆盖，`trans ∈ {OP_N, OP_T, OP_C}`（实数 C≡T），支持 incx/incy 任意非零步长（含负步长）与 batch 间带间隙 stride
2. 按 §2.5 校验层级实现 host 侧参数校验：handle → trans → m/n → lda → incx/incy → strideA/stridex/stridey → batchCount → alpha/beta → A/x/y → quick return；结构非法立即返回对应状态码，仅结构合法后允许 quick return（batchCount=0 或 m=0 或 n=0）
3. alpha=0 仅执行 y=beta*y（不读 A/x）；beta=0 不读 y 旧值
4. 精度：与 cblas_sgemv golden 对比，按生态算子开源精度标准（matched_ratio ≥ 0.99）
5. 性能：910B3 上 msprof op kernel Task Duration ≤ gpu_ms × 1.25（≥0.8× GPU 标杆）
6. 补齐 `test/gemv_strided_batched/arch22/` 测试工程与算子 README 文档

# 详细设计（required）

## 算子分析

### 数学公式

```
y[i] = alpha · op(A[i]) · x[i] + beta · y[i]      （原地覆写 y）
op(A[i]) = A[i]        （trans = OP_N，m×n）
op(A[i]) = A[i]^T      （trans = OP_T / OP_C，实数不共轭，n×m）
```

A[i] 列主序、恒为 lda×n；x[i] 长度 n（N）或 m（T/C）；y[i] 长度 m（N）或 n（T/C）。

### 支持数据类型

FP32、FP16、BF16（按 5 变体 dtype 表组合）；alpha/beta 恒为 FLOAT32 host 标量。

### 支持形状

- m, n ≥ 0；batchCount ≥ 0；不支持广播（每 batch 独立操作数）
- incx/incy 任意非零 int（含负步长，按 BLAS 反向遍历语义）
- strideA/stridex/stridey > 0，允许带间隙，相邻 batch 区域不得重叠（strideA ≥ lda×n）

## 算子实现

### 实现方案

#### host 侧设计

**1. 参数校验（对齐任务书 §2.5 校验层级）**

修复种子实现中 quick return 先于校验的问题，统一为：

```
handle(nullptr → HANDLE_IS_NULLPTR)
→ trans(非法枚举 → INVALID_ENUM)
→ m/n < 0、batchCount < 0（→ INVALID_VALUE）
→ lda < max(1,m)（→ INVALID_VALUE）
→ incx/incy ∈ {0, INT_MIN}（→ INVALID_VALUE）
→ strideA/stridex/stridey ≤ 0（→ INVALID_VALUE）
→ alpha/beta nullptr（→ INVALID_VALUE）
→ batchCount>0 且 m,n>0 时 A/x/y nullptr（→ INVALID_VALUE）
→ quick return：batchCount=0 或 m=0 或 n=0 → SUCCESS（不启动 kernel）
```

**2. 分核策略**

- 按 batch 维均分：`batchPerCore = ceil(batchCount / coreNum)`，`usedCoreNum = ceil(batchCount / batchPerCore)`，余数 batch 由最后一个核处理（batchTail）
- 单核内按输出轴（N 路径按 m，T 路径按 outTile 行块）展开 SIMT 线程，`numThreads` 按 `batchPerCore × outSize` 动态计算并夹在 [SIMT_MIN, SIMT_MAX]

**3. tiling 设计（移植 arch35 的转置切分策略至 arch22 资源参数）**

- N 路径：每 batch 整行向量一次性载入，`outTile = m`，无 dotTile 切分
- T 路径：转置点积沿 dot 轴切分。以 UB 预算为约束做二分搜索（`BinaryMaxRowsCols`）：先求容纳整 dot 轴的最大 out 行数，装不下则先按 32B/元素对齐约束定 dotTile，再反求 outTile
- UB 布局按 256B 对齐划分子缓冲（inA / inx / inY / matTmp / vecTmp），预置 VL staging 余量，避免转置尾行越界读（arch35 issue #495 同款问题预防）；total 以 uint64 累加防截断（issue #743 同款）
- tiling 数据结构 `#pragma pack(8)`，4B 字段在前、int64 stride 字段在后，保证 strideA/stridex/stridey 8 字节对齐
- tilingkey 规划：dtype（5 变体）× trans（N/T）分支由模板参数 `TIn/TOut/TRANS_T/HSS_MODE` 编译期分派，不设运行时 tilingkey

**4. arch22 与 arch35 的差异适配**

| 项 | arch35（950PR） | arch22（A2/A3）适配 |
| --- | --- | --- |
| UB 总量 | 248KB - 40KB = 208KB | 按 dav-2201 平台 UB 实际容量通过平台接口获取，tile 常量以 arch22 头文件为准（GEMV_SB_ARCH22_TILE=2048 / WIN_MAX=4096 起步，实测调优） |
| 分核 | GetAivCoreCount | 同接口，核数随硬件变化，分核逻辑参数化 |
| kernel 发射 | — | 保留种子实现已修复的发射状态检查：`kernel_do` 返回非 ACL_SUCCESS 时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，不再假成功 |

#### kernel 侧设计

- 沿用模板类 `GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>` 结构（自 arch35 移植），Init 阶段解析 tiling / 初始化 Queue，Process 阶段 CopyIn → Compute → CopyOut 三段流水，双缓冲（BUFFER_NUM=2）覆盖 GM↔UB 搬运
- N 路径：按行做向量点积，FP16/BF16 输入先转 FP32 域累加，输出按 dtype 收窄；HSS/TSS 的 y 为 FP32 直接写回
- T 路径：按 outTile 行块 × dotTile 列块二维遍历，dot 轴分段累加（matTmp/vecTmp 临时缓冲），行内用 VL 宽度向量指令做规约
- beta=0 时跳过 y 旧值读取（y 含 NaN 亦正确覆写）；alpha=0 时跳过 A/x 读取与乘加，仅做 y=beta*y
- batch 基址：`A + (int64_t)i*strideA` 等 64 位偏移计算，防止大 batch 乘法溢出

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3） | √ |
| Atlas A3 | √ |

## 算子约束限制

- 不支持广播；不支持复数（OP_C≡OP_T）；不要求确定性浮点累加顺序
- 各 batch 的 A/x/y 区域不得重叠，由调用方保证

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas_sgemv 逐 batch 生成；FP32 rtol=atol=1.2e-4，FP16 1e-3，BF16 7.8e-3；matched_ratio ≥ 0.99 且 max_abs_error 达标 | 生态算子开源精度标准（opbase mixed_tolerance_standard.md）+ 任务书 §3.2 |
| 性能标准 | 910B3 上 msprof op kernel Task Duration 平均 ≤ gpu_ms × 1.25（warmup 5 + 有效样本 10）；端到端 avg_us 仅交叉验证 | 任务书 §3.3 + test_cases/gpu_baseline.csv |

## 兼容性分析

接口声明已存在于 `include/cann_ops_blas.h`，本任务仅补齐 arch22 实现与测试，不新增/修改公开 API，无兼容性影响。

## 测试设计

- 复用 `test/gemv_strided_batched/arch35/` 通用测试工程，注册 `test/gemv_strided_batched/arch22/` 目标
- 用例：精度全量 5000 条（每变体 1000）+ 性能子集 1000 条（每变体 200，gpu_baseline.csv 随任务附件提供）
- 负向覆盖：非法 trans/负 m,n/lda 越界/incx=0/stride≤0/batch 重叠/nullptr；特殊场景：quick return、beta=0 且 y=NaN、alpha=0 且 x=NaN、Inf/NaN 传播、±步长与带间隙 stride
