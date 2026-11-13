# aclblasSaxpy（Atlas A2/A3）算子设计文档

## 需求背景（required）

### 需求来源

11 月社区任务：在 Atlas A2/A3 系列产品上使用 Ascend C 开发单精度实数向量运算算子 `aclblasSaxpy`（BLAS Level 1，`y = alpha·x + y`），对标 cuBLAS `cublasSaxpy`（参数语义、返回码与快速返回口径以其官方文档 §2.5.4 为唯一依据），补齐 ops-blas 仓 A2/A3 产品线 BLAS-1 能力缺口。

- 任务书：`aclblasSaxpy_AtlasA2_A3_task_doc.md`（本文范围真值源；配套 `test_cases/` 为用例与 GPU 基线真值源）
- 设计文档提交：`cann-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/11-11-aclblasSaxpy-A2A3/OpenGood_WeChat/docs/design.md`（沿用该任务已合入判例 #2913 的目录命名 `11-11-aclblasSaxpy-A2A3`），以 PR 形式提交
- 目标代码仓：https://gitcode.com/cann/ops-blas （`blas/axpy/arch22/` + `test/saxpy/arch22/`）
- 适配硬件：Atlas A2/A3（arch22），性能测机与自验证设备 Atlas 800T A2（910B3）；CANN 9.1.0
- 精度 golden：`cblas_saxpy`（Netlib BLAS 3.10.0，float64 真值计算后按 FLOAT32 收窄，与仓内 `saxpy_golden.h` 一致），随测试工程提供

### 背景介绍

#### 接口与仓内现状分析

调研基线：ops-blas master（2026-11-12 sparse clone 实查）。

| 项 | 仓内现状 | 与本任务关系 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasSaxpy` 声明已存在（L477，与 950PR 共用），`aclblasCaxpy` 同在（L355） | 不改头文件；实现共用声明，禁止另立私有平行接口 |
| `blas/axpy/arch35/` | **`saxpy` 单精度实数版已有**（saxpy_host.cpp / saxpy_kernel.cpp / saxpy_tiling_data.h / saxpy_kernel.h 四件套）：host 校验顺序（handle→n≤0 no-op→ValidateSaxpyParams）、连续/跨步双 tiling（`CalSaxpyTilingDataContiguous` / `CalSaxpyTilingDataStrided`）、AIV 分块 + SIMT 双实现 | 任务书 §1/§2.1/§2.5 点名的行为对齐参照（ValidateSaxpyParams 与入口 L20–160）；本任务不改 arch35 |
| `blas/axpy/arch22/` | **仅复数版 `caxpy` 四件套**（caxpy_host / caxpy_kernel / caxpy_tiling_data / caxpy_dispatch_policy），单精度实数 saxpy 缺位 | 全新增量：`blas/axpy/arch22/saxpy_*`；负步长/INT_MIN/跨度溢出的 arch22 判例从 caxpy_host.cpp 移植（`incx==0`、`INT_MIN` 拒绝、`CheckedVectorSpan` int64 域守卫） |
| `test/saxpy/` | 仅有 arch35 工程，CMakeLists 只在 `ascend950` 分支注册 GTest（`ops_blas_add_gtest_tests`），A2/A3 构建退回非 GTest runner（输出整数毫秒） | 按任务书 §2.2 补齐 arch22 分支：`ops_blas_add_gtest_tests(${OPS_BLAS} saxpy_test)` |

结论：arch35 的 saxpy 是行为与工程形态的权威参照，arch22 的 caxpy 是 arch22 工程惯例与步长守卫判例的载体；本任务以两者为骨架做 arch22 单精度实数移植与测试工程补齐，无头文件改动、无 arch35 改动。

#### 功能分析

`aclblasSaxpy` 计算 `y[j] = alpha·x[k] + y[j]`（k = 1+(i−1)·incx，j = 1+(i−1)·incy，i = 1..n），全 FLOAT32，y 原地覆写，x 只读。关键语义（任务书 §2.1/§2.4/§2.5）：

1. **n≤0 合法 no-op**：不访问 alpha/x/y，直接返回 SUCCESS（Netlib `saxpy.f` 的 `IF (N.LE.0) RETURN`；注意 no-op 先于指针校验——n≤0 时 alpha/x/y 可为 nullptr，这是与"先校验后 no-op"类算子的顺序差异点）。
2. **alpha 为 Host 标量指针**：值不校验（任意 FLOAT32 含 Inf/NaN）；`alpha=0.0` 时按 Netlib `IF (DA.EQ.ZERO) RETURN` 直接返回（y 不变且不读 x）。与 950PR arch35 实现对齐：arch35 在 host 侧同步解引用 `*alpha` 后随 tiling 下发（值语义，无 Device 标量指针路径）。
3. **步长语义**：incx/incy 支持正负步长（非连续访问即 BLAS 步长语义本身），负步长自物理 span 尾端起算；`incx/incy == 0` 或 `INT32_MIN` 拒绝（INVALID_VALUE）；`(n−1)×|incx|`、`(n−1)×|incy|` 超出 int32 表示范围拒绝（host 侧 int64 计算，issue #780/#866 口径，与仓内 caxpy arch22 `CheckedVectorSpan` 同构）。
4. **返回码口径**（cuBLAS 文档）：handle nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n>0 时 alpha/x/y nullptr 或步长非法 → `ACLBLAS_STATUS_INVALID_VALUE`。
5. **异步语义**：经 handle 绑定 stream 异步下发，返回成功仅表示 Host 已提交；读回前须 `aclrtSynchronizeStream`。
6. **确定性**：逐元素单次执行内确定，跨次执行 bit 级一致（纯逐元素运算，无归约顺序问题）。

## 需求分析（required）

### 需求描述

按任务书 §2.3 签名实现 `aclblasSaxpy`（handle, n, alpha, x, incx, y, incy），host 校验 + tiling + kernel 直调；交付 arch22 实现与 GTest 测试工程；精度按 FLOAT32 标准（实部单通道：`|a−g| ≤ 2^-13(1+|g|)`，matched_ratio ≥ 0.99，max_abs ≤ 1e-2 或 32×ULP，Inf/NaN 按等效规则）；性能在 910B3 上平均单次耗时 ≤ GPU 标杆 × 1.25（gpu_baseline.csv 逐 case）。

### 需求拆解

1. **host 侧**（`blas/axpy/arch22/saxpy_host.cpp`）：校验链（handle → n≤0 → ValidateSaxpyParams（alpha/x/y 判空、incx/incy ≠0 且 ≠INT32_MIN、CheckedVectorSpan 跨度守卫）→ alpha==0.0 quick return → tiling → 直调下发），全部对齐 arch35 saxpy_host.cpp 的口径与函数形态。
2. **kernel 侧**（`saxpy_kernel.cpp/.h`）：连续快路径（incx==1 && incy==1，对齐块搬运 + 向量化 FMA 乘加，UB 双缓冲流水）+ 通用步长路径（gather/scatter 按步长寻址，负步长经基准指针归一），行内逐元素运算、无原子操作。
3. **tiling**（`saxpy_tiling_data.h`）：沿用 arch35 字段形态（totalN/perCoreN/remainder/tileSize/alpha + 跨步路径 useCoreNum/startOffset/calCount），字段排布按 arch22 caxpy 惯例 `static_assert` 锁偏移。
4. **测试工程**（`test/saxpy/arch22/`）：任务包 1000 条精度 CSV + 200 条性能子集全量加载；`GAUSS_NORM_5_2` 正态分布补充用例（§3.5 要求，gen_csv.py 默认只出均匀分布）；负向用例 dummy 缓冲区规则；CMakeLists arch22 分支 GTest 注册补齐。
5. **文档**：算子 README（产品支持表标注 Atlas A2/A3：支持）+ 测试 README（复现步骤）。

## 详细设计（required）

### host 侧设计

校验与下发次序（非法/空计算场景不访问 Device 数据；与 arch35 saxpy_host.cpp 逐分支对齐）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `n <= 0` → `SUCCESS`（no-op，不访问 alpha/x/y——先于指针校验，Netlib/cuBLAS 口径）
3. `alpha/x/y == nullptr`（n>0）→ `ACLBLAS_STATUS_INVALID_VALUE`
4. `incx == 0 || incy == 0` → `INVALID_VALUE`；`incx == INT32_MIN || incy == INT32_MIN` → `INVALID_VALUE`
5. 跨度守卫：`(n−1)×|incx|`、`(n−1)×|incy|` 在 int64 域计算，超 int32 表示范围 → `INVALID_VALUE`（CheckedVectorSpan 同构，#780/#866）
6. `alpha == 0.0f` → `SUCCESS`（Netlib quick return，y 不变、不读 x；host 侧解引用后值比较）
7. tiling：`incx==1 && incy==1` 走连续路径（`totalN` 均分核 + 32 元素对齐块 + 尾块归末核），否则走跨步路径（useCoreNum = min(n, AIV 核数， SAXPY_MAX_CORE_NUM)，均衡 startOffset/calCount）
8. `saxpy_kernel_do(x, y, tiling, numBlocks, handle->stream)` 直调下发

alpha 为 Host 标量：host 侧解引用 `*alpha` 后以值随 tiling 下发（与 arch35 一致，无 Device 标量指针分支；任务书 §2.5 异步执行行明确"alpha 为 Host 内存标量，算子在 Host 侧同步解引用后下发"）。

### 分核及分块设计

- **连续快路径**（incx==1 && incy==1）：`numBlocks = min(totalN, AIV核数)`，每核 `perCoreN = floor(totalN/numBlocks)` 向下取整到 32 元素对齐单元，尾块 `remainder` 归末核；核内按 tile（≤32K 元素，UB 双缓冲）切片流水。
- **通用步长路径**：均衡分段（startOffset[i]/calCount[i]），段内步长寻址；负步长在 host 侧归一——保持逻辑序（i = 0..n−1，元素地址 = base + i×inc×4B），x/y 基准指针按负步长折叠到物理首地址（arch22 caxpy 同手法），kernel 内只处理非负步长间隔访问。
- 无跨核数据依赖、无归约，任一 y 元素仅被一个核写一次（同 incy 段不重叠），天然无原子需求；确定性由此保证。

### kernel 侧设计

**连续快路径**：x/y tile 搬入 UB（DataCopy 对齐块，尾块 DataCopyPad），向量化乘加 `y = alpha·x + y`（一条向量 FMA 或 mul+add 两条，fp32 精度与标量序完全一致——逐元素运算无累加链，不引入舍入差异），结果写回 y。

**通用步长路径**：stride ≠ 1 时标量载入/存储或按 mask 向量化（以连续段 gather 为优选：步长 ≥1 时同向相邻元素地址间隔固定，可按窗口小批量搬入 UB 再解交织；caxpy arch22 已验证该窗口手法），逐元素 `y_j += alpha·x_k`。

**Inf/NaN**：全值域按 IEEE 自然传播，无任何特殊值分支；golden 同参数由 cblas_saxpy 生成， Inf/NaN 等效判定规则由测试框架承担。

**UB 预算**：连续路径 x/y 双缓冲 2×2×tile×4B（tile ≤32K → ≤512 KiB 超限时取 tile=16K 级别对齐 192 KiB 预算；实际以 `UB_SIZE/(2×2×4)` 对齐计算，与 arch35 `CalSaxpyTilingDataContiguous` 同式）；跨步路径窗口小批量，预算更小。无 GM workspace。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B 系列，arch22） | √（性能测机/自验证设备） |
| Atlas 800I A3（arch22） | √ |
| Ascend 950PR/DT（arch35） | 已有独立实现（blas/axpy/arch35/saxpy），本交付不涉及、不改动 |

### 算子约束限制

- 仅 FLOAT32；不支持 n、incx、incy 以外的张量视图语义（BLAS 步长即非连续访问本身）。
- y 原地覆写；x 与 y 内存重叠行为未定义（调用方保证，同 cuBLAS 口径）。
- 异步执行依赖 handle 绑定 stream，读回前须同步；不涉及 broadcast/dynamic shape/确定性特殊要求（逐元素天然 bit 级确定）。
- alpha 仅支持 Host 标量指针（任务书 §2.4），Device 标量不在本批次范围。

## 可维可测分析（required）

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | FLOAT32：`|a−g| ≤ atol + rtol·|g|`（rtol=atol=2^-13），matched_ratio ≥ 0.99，max_abs ≤ 1e-2 或 32×ULP；Inf/NaN 按任务书等效规则单独统计；golden 由 cblas_saxpy float64 计算后 FLOAT32 收窄 | 任务书 §3.2 + opbase mixed_tolerance 标准（commit 28a1ce5） |
| 性能 | 910B3，msprof op 模式 task duration（us），warmup 5 + 有效采样 ≥10 取平均；判据 Avg ≤ gpu_baseline × 1.25。标杆要点（gpu_baseline.csv）：大向量 n=1048576 → 12.07us 门槛（15.09us）、n=2097152 → 16.40us（20.50us）、n=4194304 → 34.82us（43.52us）；小向量 n≤512 一档 GPU 约 6.5–7.3us（门槛 8.2–9.1us，贴近 launch 地板，为性能主攻点）。实测数据【待上机回填】 | 任务书 §3.3 |

### 测试工程与兼容性分析

- **目录**：`test/saxpy/arch22/`（saxpy_test.cpp、saxpy_perf.cpp、saxpy_npu_wrapper.h、saxpy_test.csv、gpu_baseline.csv、README.md）；`test/saxpy/CMakeLists.txt` 增 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS} saxpy_test)`（补齐仓内 arch22 GTest 缺口，任务书 §2.2 点名）。
- **用例**：任务包 1000 条精度 CSV + 200 条性能子集全量（边界域 n∈{0,1,1024,16384,262144,1048576}、alpha∈{0.0,1.0,2.5,−1.5,0.1,10.0}、步长 {1,2,7,−1,−3}；负向域 handle 空/n>0 指针空/inc=0/INT32_MIN/跨度溢出 n=3×inc=INT_MAX 与 n=2^30×inc=8）；按 §3.5 补 `GAUSS_NORM_5_2` 正态分布用例（扩展 gen_csv 或独立 CSV，纳入精度自验并在自测报告说明补齐方式与数量）。
- **负向用例缓冲区**：期望非 SUCCESS 的用例以单元素 dummy 缓冲（或 nullptr）占位，不按 `(n−1)×|inc|` 预分配（int 域即溢出）；n=0 与步长 quick return 用例无需按步长分配（任务书 §3.5 规则）。
- **兼容性**：代码改动仅 `blas/axpy/arch22/`（新增 saxpy 四件套）、`test/saxpy/`（新增 arch22 工程 + CMakeLists 分支）；不动 arch35、不动头文件（声明已在）、不动 caxpy；`blas/axpy/README.md` 产品支持表补 Atlas A2/A3。
- **私仓交付**：gcw_LxlU8H9L/ops-blas 分支 `feat/saxpy-arch22`，邀请 Ascend-CANN 为开发者；`task_submission/` 目录按模板放置自验证步骤、精度/性能/内存报告与日志。

### 风险评估

| 风险 | 应对 |
| --- | --- |
| 小向量 GPU 标杆贴近 launch 地板（n≤512 约 6.5us，门槛 8.2us） | 连续路径单核小批量分支（免多核调度开销、tile 一步到位）；实测按 §3.3 口径回填，与平台公开数据交叉对照 |
| 步长路径向量化收益有限（gather/scatter 开销） | 以达标为先：步长路径 GPU 标杆同样宽松（非连续访存），必要时标量实现保正确性，性能主攻连续大向量 |
| GAUSS_NORM_5_2 补充用例覆盖争议 | 补充 CSV 与生成脚本改动一并提交，自测报告列明条数与覆盖类别 |
| 设计文档赛道已有多份合入（同任务 7 份），评审基准被抬高 | 本文锚点全部落在仓内源码与任务书条款（arch35 行为对齐表、arch22 caxpy 判例、CMakeLists 缺口），差异化在"对齐口径逐条可验证"，无预填性能数字 |
