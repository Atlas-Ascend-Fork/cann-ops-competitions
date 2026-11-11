# aclblasChemv（A2/A3）算子设计文档

## 需求背景（required）

### 需求来源

11 月社区任务：在 Atlas A2/A3 系列产品上使用 Ascend C 开发单精度复数（complex64）Hermitian 矩阵-向量乘算子 `aclblasChemv`，与 cuBLAS `cublasChemv` 核心功能、参数语义完全对齐，补齐 ops-blas 仓 BLAS Level 2 能力。

- 任务书：`aclblasChemv_AtlasA2_A3_task_doc.md`（本文范围真值源；配套 `test_cases/` 为用例与 GPU 基线真值源）
- 设计文档提交：`cann-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/11月社区任务-aclblasChemv算子开发(A2A3)/OpenGood_WeChat/docs/design.md`，以 PR 形式提交（即本 PR）
- 目标代码仓：https://gitcode.com/cann/ops-blas （`blas/symv/arch22/` + `test/symv/chemv/arch22/`）
- 适配硬件：Atlas A2（性能测机 800T A2 / 910B3）与 Atlas A3（arch22）；CANN 9.1.0
- 精度 golden：`cblas_chemv`（Netlib CBLAS 复数单精度，`CblasColMajor`），随测试工程提供

### 背景介绍

#### 接口与仓内现状分析

调研基线：ops-blas master（2026-11-11 sparse clone 实查）。

| 项 | 仓内现状 | 与本任务关系 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 已有 `aclblasSsymv`/`aclblasCsymv`/`aclblasChpmv` 等同族声明，**无 `aclblasChemv`** | 按任务书 §2.3 新增声明（签名与 `cublasChemv` 参数序列一致，维数参数 int，同族 `aclblasSsymv` 口径），供其他产品线共用，禁止私有平行接口 |
| `blas/symv/arch22/` | 仅有实数 `ssymv`（host/kernel/tiling 三件套） | 全新增量，落同目录 `chemv_*` |
| `blas/symv/arch35/` | 已有复数 `csymv`（arch35） | 复数语义与 `ComplexScalarSource` 标量双读口径（issue #869）的参照实现，本任务不改 arch35 |
| `test/symv/` | `ssymv`/`csymv` 两套均**只在 CMakeLists 的 arch35 分支注册 GTest**（`ops_blas_add_gtest_tests`），arch22 分支退回非 GTest runner（输出整数毫秒） | 按任务书 §2.2 随本任务补齐：`test/symv/chemv/CMakeLists.txt` arch22 分支调用 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)` |

#### 功能分析

`aclblasChemv` 计算 `y = alpha·A·x + beta·y`（全 complex64，列主序，步长 incx/incy 可正可负）。与一般矩阵-向量乘（Csymv）的关键差异是 **Hermitian 语义**（Netlib `chemv.f` 逐式同源）：

1. 仅 `uplo` 指定三角被引用，未引用三角不得读取，按 `A = A^H` 共轭对称推断；
2. 对角元素仅实部参与计算（虚部假定 0），即贡献 `alpha·x(j)·real(A(j,j))`；
3. 非对角共轭出现在两处：列 J 的散射段 `y(I) += alpha·x(J)·conj(A(I,J))`（I 在引用三角内），以及对角行项 `y(J) += alpha·conj(Σ x(I)·A(I,J))`（对引用段求和后取共轭）。

特殊语义（任务书 §2.1/§2.5）：`n=0` 合法 no-op；`alpha=(0,0)` 且 `beta=(1,0)` quick return（A/x/y 可全空）；`beta=(0,0)` 时 y 旧值不必有效（不读 y），但 y 非空要求不豁免；alpha/beta 支持Host/Device 双存储（运行时按指针属性探测，对齐 csymv arch35 `ComplexScalarSource` 口径，issue #869）；`(n-1)×|incx|` 跨度守卫（host 侧 int64 计算，对齐 ssymv arch22 #807 口径）；`incx/incy = 0 或 INT32_MIN` 拒绝（#780）。

## 需求分析（required）

### 需求描述

按任务书 §2.3 签名实现 `aclblasChemv` 句柄式接口，host 绑定 stream 直调 arch22 kernel；交付 arch22 实现与 GTest 测试工程；精度按 FLOAT32 标准（实/虚部分别：`|a-g| ≤ 2^-13(1+|g|)`，matched_ratio ≥ 0.99，max_abs ≤ 1e-2 或 32×ULP，含 Inf/NaN 等效规则）；性能在 910B3 上不低于 GPU 标杆 0.8 倍（Avg time ≤ gpu_baseline × 1.25）。

### 需求拆解

1. **host 侧**（`blas/symv/arch22/chemv_host.cpp`）：参数校验（任务书 §2.5 顺序）、alpha/beta 标量双读（#869）、quick return 判定、跨度守卫（#807）、均衡分核 tiling、kernel 直调。
2. **kernel 侧**（`chemv_kernel.cpp`，AIV）：两段式（phase 0 标量 y / phase 1 逐列贡献原子累加，沿用仓内 ssymv arch22 已验证结构）；Hermitian 三要素（散射取共轭、对角仅实部、求和后取共轭）；复数交错布局的向量化乘加，不引入 FMA 融合（与 golden 舍入序一致）。
3. **接口声明**：`include/cann_ops_blas.h` 新增 `aclblasChemv`（唯一头文件改动）。
4. **测试工程**（`test/symv/chemv/arch22/`）：任务书 1200 条 CSV 全量 + `GAUSS_NORM_5_2` 正态分布补充用例（§3.5-3 要求 gen_csv.py 未覆盖部分）+ 负向用例 dummy 缓冲区规则；arch22 GTest 注册补齐；性能采集 warmup 5 + 有效采样 ≥10 取平均（msprof task duration 口径）。
5. **文档**：算子 README（产品支持表标注 Atlas A2/A3：支持）、测试 README（复现步骤）。

## 详细设计（required）

### host 侧设计

校验与下发次序（非法/空计算场景不访问 Device 数据，任务书 §2.5 口径）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `n == 0` → `SUCCESS`（不下发）
4. `uplo` 非法 → `ACLBLAS_STATUS_INVALID_ENUM`；`lda < max(1,n)` → `INVALID_VALUE`；`incx/incy == 0 或 INT32_MIN` → `INVALID_VALUE`（#780）
5. `alpha/beta == nullptr`（n>0）→ `INVALID_VALUE`
6. 读标量（`ComplexScalarSource` 探测 host/device，device 侧 `aclrtMemcpyAsync` D2H 后同步，与 csymv arch35 逐式同源，#869）
7. `alpha=(0,0)` 且 `beta=(1,0)` → `SUCCESS`（quick return，A/x/y 允许全空）
8. A/x/y 判空（按 §2.4 条件：`alpha≠(0,0)` 且 n>0 时 A/x 非空；非 quick-return 时 y 非空）→ `INVALID_VALUE`
9. 跨度守卫：`(n-1)×|incx|`、`(n-1)×|incy|` 在 int64 域计算，超出 uint32 索引域 → `INVALID_VALUE`（与 ssymv arch22 #807 拒绝式守卫一致）
10. 计算 tiling，`chemv_kernel_do(...)` 直调下发

tiling 字段（对齐 `CsymvTilingData` 既有形态）：`n/lda/uplo/alphaReal/alphaImag/betaReal/betaImag/incx/incy/alphaIsDevice/betaIsDevice/useCoreNum/phase`。`beta=(0,0)` 时 phase 0 跳过（等价 Netlib `TEMP2` 起点取 ZERO，保证 y 旧值可为 NaN）。

### 分核及分块设计

- **列均衡拆分**：核 c 负责列区间 `[n·c/C, n·(c+1)/C)`（u64 域），尾核最多多 1 列；`C = min(n, AIV 核数)`。
- 列 J 的 A 段在列主序下为连续区间（UPPER：行 `[0, J-1]`；LOWER：行 `[J+1, n-1]`），散射目标 y 段同为连续区间——单列内搬运与原子写均连续。
- phase 1 每列 A 段按 ≤16 KiB chunk 流水（VECIN 双缓冲）；x 在每核入口一次搬入并解交错为 xR/xI 平面（n×8B ≤ 32 KiB，UB 可全量驻留）。

### kernel 侧设计

**两段式结构**（沿用 ssymv arch22 的 phase 0/phase 1 模式）：

- **phase 0**：`y = beta·y`。`beta=(0,0)` 时整段跳过（不读 y，兼容 y=NaN 输入用例）；否则 core 0 单核覆盖写（非原子），支持 incy 步长寻址。
- **phase 1**：各核对自有列区间计算 `alpha·A·x` 贡献并原子累加进 y（`SetAtomicAdd(float)`，复数交错布局即实/虚部成对原子加，与 ssymv arch22 相同口径）。每列 J（以 UPPER 为例，LOWER 对称）：
  1. 搬入引用段 `A[0..J-1][J]`，与 xR/xI 平面做复数乘归约：`s = Σ x(I)·A(I,J)`（乘法四条、加法两条，不用 FMA——与 Netlib `TEMP` 计算的舍入次序一致，避免融合舍入差在极端值上翻转 Inf/NaN）；
  2. 散射：对段内 `y(I) += alpha·x(J)·conj(A(I,J))`（逐 chunk 向量化乘加后原子写回，共轭在乘法侧完成）；
  3. 对角行项：`y(J) += alpha·x(J)·real(A(J,J)) + alpha·conj(s)`（对角仅取实部；`conj(s)` 由求和结果取虚部反号得到，`s` 的累加序与 Netlib 列内 DO 循环一致）。

**Inf/NaN 传播**：与 golden（Netlib）逐式同源，不做任何零跳过或特殊值分支——`0×Inf/NaN` 不定式、累加溢出的最终形态随同一路径自然传播（任务书 §3.2 Inf/NaN 等效规则据此判定）；golden 侧由测试框架的 `cblas_chemv`（Netlib 3.10.0）按同参数生成，两侧均按 Hermitian 约束（仅引用三角、对角虚部置 0）计算。

**UB 预算**（192 KiB）：xR/xI 平面 2×16 KiB + A 列段双缓冲 2×16 KiB + 归约临时 ≈ 64 KiB（峰值列段按 chunk 切流，不随 n 增长），余量充足。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B 系列，arch22） | √（性能测机） |
| Atlas 800I A3（arch22） | √ |
| Ascend 950（arch35） | 不涉及（csymv arch35 为参照实现，不在本任务改动范围） |

### 算子约束限制

- 仅 complex64；A 为 n×n Hermitian 方阵，列主序存于 lda×n；不支持超出 lda/incx/incy 语义的非连续视图（任务书本批次不要求）。
- 无 broadcast、不要求 dynamic shape、不要求确定性计算；y 原地覆写。
- A/x/y 内存重叠未定义；异步语义依赖 `aclblasSetStream`，调用方读回前须同步。

## 可维可测分析（required）

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 实/虚部分别按 FLOAT32：`|a-g| ≤ atol + rtol·|g|`（rtol=atol=2^-13），matched_ratio ≥ 0.99，max_abs ≤ 1e-2 或 32×ULP；Inf/NaN 按任务书等效规则单独判定；golden 为 `cblas_chemv`（CblasColMajor，对角仅实部） | 任务书 §3.2 + 生态算子精度标准 |
| 性能 | 910B3，msprof op 模式 task duration（us），warmup 5 + 有效采样 ≥10 取平均；判据 Avg ≤ gpu_baseline × 1.25。四条典型 case 门槛：n=512 → 18.30us；n=1024 → 26.39us；n=2048 → 37.61us；n=4096 → 81.62us（baseline 14.64/21.11/30.09/65.30us × 1.25，见 gpu_baseline.csv chemv-base-000~003）。实测数据【待上机回填】 | 任务书 §3.3 |

### 测试工程与兼容性分析

- **目录**：`test/symv/chemv/arch22/`（chemv_test.cpp、chemv_perf.cpp、chemv_npu_wrapper.h、chemv_test.csv、gpu_baseline.csv、README.md）+ `test/symv/chemv/CMakeLists.txt`（arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)`，补齐仓内 arch22 GTest 缺口）。
- **用例**：任务包 1200 条 CSV 全量（L0 基础/L1 尺寸/L2 标量/L4 前导维与步长/L5 填充含 Inf/NaN/L6 边界负向/PF 性能）；按 §3.5-3 补充 `GAUSS_NORM_5_2` 正态分布用例（扩展 gen_csv 或独立 CSV，纳入精度自验并在自测报告说明补齐方式与数量）。
- **负向用例缓冲区**：期望非 SUCCESS 的用例以单元素 dummy 缓冲（或 nullptr）占位，不按 `(n-1)×|inc|` 预分配（int 域即溢出），接口在读数据前返回预期错误码即判定通过（任务书 §3.5 规则）。
- **兼容性**：代码改动仅 `include/cann_ops_blas.h`（新增声明）、`blas/symv/arch22/`（新增三件套）、`test/symv/chemv/`（新增）；不动 arch35、不动既有 ssymv/csymv 与公共头文件其余部分；`blas/symv/README.md` 产品支持表补 Atlas A2/A3。
- **私仓交付**：gcw_LxlU8H9L/ops-blas 分支 `feat/chemv-arch22`，邀请 Ascend-CANN 为开发者；`task_submission/` 目录按模板放置自验证步骤、精度/性能/内存报告与日志。

### 风险评估

| 风险 | 应对 |
| --- | --- |
| 小尺寸 GPU 基线极低（n≤16 约 6.1–7.6us）导致 1.25× 门槛贴近 launch 开销地板 | 小尺寸走紧凑分支（单核、x 常驻、免 chunk 流水）；性能报告按 §3.3 口径实测回填，与 910B3 平台公开数据交叉对照 |
| 原子累加顺序不定导致与 golden 累加序差异放大极端值偏差 | 散射项为逐列独立元素累加（每 y(I) 仅来自本列一次贡献 + phase 0 一次），非跨列重排；列内归约 s 保持 Netlib 顺序，精度风险收敛于 FLOAT32 标准内 |
| `GAUSS_NORM_5_2` 补充用例与 gen_csv.py 默认分布不一致造成覆盖争议 | 补充 CSV 与生成脚本改动一并提交，自测报告列明条数与覆盖类别 |
| 平台检查器/CI 波动（本周期 GitCode 机器人状态不稳） | PR 发起后即 /compile；标签被摘则按仓内惯例重新 /compile（hesc #2757 判例：同一 PR 重复 4 次属正常） |
