# aclblasIcamin（Atlas A2/A3）算子设计文档

## 需求背景（required）

### 需求来源

11 月社区任务：在 Atlas A2/A3 系列产品上使用 Ascend C 开发单精度复数向量最小模元素索引算子 `aclblasIcamin`（`result = argmin_i(|Re(x[k])| + |Im(x[k])|)`，1-based，平局取最小索引），与 cuBLAS `cublasIcamin` 核心功能、参数语义完全对齐，补齐 ops-blas 仓 A2/A3 BLAS-1 索引归约能力。

- 任务书：`aclblasIcamin_AtlasA2_A3_task_doc.md`（本文范围真值源；配套 `test_cases/` 为用例与 GPU 基线真值源）
- 设计文档提交：`cann-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/11-aclblasIcamin-A2A3/OpenGood_WeChat/docs/design.md`（沿用该任务已合入判例 #2997/#2965 的目录命名），以 PR 形式提交
- 目标代码仓：https://gitcode.com/cann/ops-blas （`blas/iamin/arch22/` 新建 + `test/iamin/icamin/arch22/` 新建）
- 适配硬件：Atlas A2/A3（arch22），性能测机 Atlas 800T A2（910B3）；CANN 9.1.0
- golden：测试工程内 cblas 风格 CPU 参考循环（Netlib 无 icamin 例程，语义以 cuBLAS 文档为最高优先级）

### 背景介绍

#### 接口与仓内现状分析

调研基线：ops-blas master（2026-11-13 sparse clone 实查）。

| 项 | 仓内现状 | 与本任务关系 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasIcamin` 声明已存在（**L249**，任务书 §2.3 写 L241 为笔误；`aclblasIsamin` 同在 L248），与 950PR 共用 | 不改头文件；实现共用声明，禁止私有平行接口 |
| `blas/iamin/arch35/` | `icamin` **复数版已有**（icamin_host.cpp / icamin_kernel.cpp / icamin_kernel.h / icamin_tiling_data.h 四件套）+ `isamin` 实数版 | 任务书 §2.2 点名的归约实现参照；本任务不改 arch35 |
| `blas/iamin/arch22/` | **不存在（目录需新建）** | 全新增量：arch22 版 icamin 四件套 |
| `test/iamin/icamin/` | arch35 工程齐备（test.cpp / csv / golden.h / npu_wrapper.h / param.h），CMakeLists 只在 arch35 分支注册 GTest，A2/A3 构建退回非 GTest runner | 按任务书 §2.2 补齐 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)`；测试工程参照 arch35 模式新建 arch22 目录，CSV 列格式对齐 `icamin_param.h`（x 列 `NULLPTR` 表示空指针负向） |

#### 功能分析

`aclblasIcamin` 在复数向量 x 中找 `|Re|+|Im|`（BLAS icamin 惯例模，非欧几里得模）最小的元素，输出 1-based 索引。关键语义逐条锚定（arch35 `icamin_kernel.cpp` 头注释与实现即权威语义文档）：

1. **严格小于更新**（cblas 口径）：以首个元素的模为基准，之后仅 `val < bestVal` 才更新——平局保最小索引（arch35 `IcaminAccumulate` L65–79）。
2. **NaN 语义**：NaN 与任何值比较均为 false，永不被选中；若 x[0] 为 NaN，基准即 NaN，任何后续元素都无法取代，结果为 1（含全 NaN 输入，任务书 §3.2 补充说明）。向量 fold 中 NaN 模值重写为 +Inf 保证 ReduceMin 稳定；全局元素 0 为 NaN 时属主核在发布前强制写入 `(NaN, idx 0)`，使跨核合并与顺序扫描一致（arch35 L31–35）。
3. **quick return**：`n == 0` 或 `incx < 1`（含 0 与负步长）不触发 kernel，`result` 设备侧写 0（`aclrtMemsetAsync`，arch35 `WriteQuickReturn`）后返回 SUCCESS；先于 x 判空（n=0 时 x 可为 nullptr）。`result == nullptr` 恒为 INVALID_VALUE（任何路径，含 quick return）。
4. **跨度守卫（×2 复数系数）**：`(n−1)×incx` 在 int64 域计算，超过 int32 上限即拒绝（arch35 额外考虑复数交错存储的 `×2` 字节展开，issue #745 口径）。
5. **bit-exact 判定**：输出为 INT32 索引，验证为 `actual == golden` 精确一致。归约顺序无关性论证见详细设计——这是本算子正确性的核心命题。

## 需求分析（required）

### 需求描述

按任务书 §2.3 签名实现 `aclblasIcamin(handle, n, x, incx, result)`，host 校验 + tiling + kernel 直调；交付 arch22 实现与 GTest 测试工程；输出 INT32 索引按精确一致判定；性能在 910B3 上 Avg ≤ GPU 标杆 × 1.25。

### 需求拆解

1. **host 侧**（`blas/iamin/arch22/`）：校验链与 arch35 `ValidateIcaminParams` 逐分支对齐（handle → n<0 → result 判空 → n==0/incx<1 quick return → x 判空 → 跨度×2 守卫），quick return 走设备 memset，tiling 双路径（连续 32B 对齐切片 / 跨步通用）。
2. **kernel 侧**：两级归约——各核内 UB 向量 fold（Abs + 解交错 + 求模 + ReduceMin(calIndex)）产出每核 (val, idx) 对，reduce kernel 单核合并写 1-based 结果；SIMT 标量路径兜底（跨步扫描与 NaN/Inf 切片回退精扫）。
3. **测试工程**（`test/iamin/icamin/arch22/`）：任务包 CSV 全量 + GAUSS_NORM_5_2 正态补充用例 + NaN/平局/quick return 专项；arch22 GTest 注册补齐。
4. **文档**：算子 README（产品支持表标注 Atlas A2/A3：支持）+ 测试 README（复现步骤）。

## 详细设计（required）

### host 侧设计

校验与下发次序（与 arch35 `icamin_host.cpp` 逐分支对齐）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `result == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`（任何 n，含 quick return 路径——任务书 §2.4 result 行明确）
4. `n == 0 || incx < 1` → `WriteQuickReturn`：`aclrtMemsetAsync(result, sizeof(int), 0, ..., stream)` 设备侧写 0，返回 SUCCESS（不触发 kernel；此路径先于 x 判空，n=0 时 x 可为 nullptr）
5. `x == nullptr` → `INVALID_VALUE`
6. 跨度守卫：`lastComplex = (n−1)×incx`（int64 域），`lastComplex < 0 || lastComplex > INT32_MAX || lastComplex > INT64_MAX/2` → `INVALID_VALUE`（×2 为复数交错存储 float 对系数，arch35 同式，issue #745）
7. tiling：`numBlocks = min(n, AIV核数)`；连续路径（incx==1）切片按 4 复数（32B）向上对齐，`useCoreNum` 反推、`lastCoreN` 归末核；跨步路径标量扫描任意对齐
8. workspace 校验：每核 2 float（val/idx）× 对齐 64 → 检查 handle workspace 容量后 `icamin_kernel_do(...)` 直调

tiling 字段（与 arch35 `IcaminTilingData` 同构）：`totalN/perCoreN/lastCoreNum/useCoreNum/tileSize/nthreads/incx`；`ICAMIN_TILE_COMPLEX = 4096`（4096 复数 = 32KB/槽，双缓冲 + 模值 scratch 合计 ≈117KB，对齐 arch35 UB 预算口径，arch22 上按实际 UB 大小同式核算）。

### 分核及分块设计

- 各核负责连续切片（连续路径 32B 对齐，跨步路径均分+末核补余），核间无数据依赖；
- 每核产出**一个** (val, idx) 对写入 workspace 槽位（val/idx 各 float，32B 对齐槽）；
- 归约阶段 `icamin_reduce_kernel<<<1>>>` 按槽序合并：种子取首个非空槽，严格小于更新——**槽序即元素序**（核 c 的切片整体小于核 c+1 的切片），因此合并顺序等价于全局顺序扫描，平局自然归最小索引。

### kernel 侧设计

**归约顺序无关性论证（bit-exact 正确性核心）**：

模值比较 `val < bestVal` 为全序严格弱化（NaN 重写为 +Inf 后仍保序），最小值具有唯一"首达位置"——所有等于最小值的元素中索引最小者。逐对归约满足结合律与交换律（取 (min val, 对应最小 idx) 的字典序最小对），故任意归约树的结果与顺序扫描一致；实现进一步以"槽序=元素序"消解对 idx 的额外仲裁需求。因此 INT32 结果与 golden（顺序循环）bit-exact 一致，与核数、tile 大小无关。

**连续路径（incx==1，向量 fold）**：tile 4096 复数流水——MTE2 搬入交错复数 → Abs（复数绝对值逐分量）→ DeInterleave（re/im 分离）→ Add（|re|+|im| 模值）→ `ReduceMin(calIndex)` 产出本 tile (val, idx) 落 UB 槽；MTE2 与 V 流水通过深度 2 TQue 重叠（arch35 同式）。tile 内 NaN 模值重写 +Inf 防止单点污染 ReduceMin。

**跨步路径（incx≥2）**：SIMT 标量扫描（每线程一个复数元素，`nthreads` 按 perCoreN 128 对齐并封顶），`IcaminAccumulate` 逐元素严格小于更新（跨步访存无向量收益，正确性优先；GPU 标杆对非连续 case 同样宽松）。

**NaN/Inf 兜底精扫**：某核合并后的最小值不严格小于 FLT_MAX（全 NaN/Inf 切片，或模恰为 FLT_MAX）时，该切片转元素级重扫，按顺序语义仲裁 NaN 前缀与 FLT_MAX 边界（arch35 同式）；全局 x[0] 为 NaN 时属主核发布前强制 `(NaN, idx 0)`，保证跨核合并等于顺序扫描。

**输出**：reduce kernel 写 `result = bestIdx + 1`（0-based 内部 → 1-based 输出）；quick return 路径已由 host memset 写 0。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B 系列，arch22） | √（性能测机） |
| Atlas 800I A3（arch22） | √ |
| Ascend 950PR/DT（arch35） | 已有独立实现（blas/iamin/arch35/icamin），本交付不涉及、不改动 |

### 算子约束限制

- incx 仅支持 ≥1（负步长与 0 为 quick return 写 0，与 cublasIcamin 口径一致——任务书 §2.4 明确，区别于 Saxpy 族"负步长反向遍历"）。
- 不支持 broadcast / dynamic shape / 原地语义（单向量归约到标量索引）；x 只读，result 为独立标量输出。
- 异步执行依赖 handle 绑定 stream，读回 result 前须同步。
- 输入模值恰为 FLT_MAX 且为全局最小的边界情形由兜底精扫仲裁（非 Undefined，测试覆盖）。

## 可维可测分析（required）

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | INT32 索引精确一致（`actual == golden`，matched_ratio 退化为单值判定）；quick return 用例判 `result == 0`；NaN/平局/FLT_MAX 边界逐条对齐 §3.2 语义；golden 为 cblas 风格顺序循环（strict < 更新） | 任务书 §3.2 |
| 性能 | 910B3，msprof op 模式 task duration（us），warmup 5 + 有效采样 ≥10 取平均；判据 Avg ≤ gpu_baseline × 1.25。标杆要点（gpu_baseline.csv）：n=1048576 → 22.70us 门槛（18.16us 基准）、n=2097152 → 27.03us（21.62us）、n=4194304 → 45.76us（36.61us）；小向量 n≤512 一档 GPU 约 8.0–8.3us（门槛 ~10.3us，launch 地板区，为性能主攻点）。实测数据【待上机回填】 | 任务书 §3.3 |

### 测试工程与兼容性分析

- **目录**：`test/iamin/icamin/arch22/`（icamin_test.cpp、icamin_perf.cpp、icamin_npu_wrapper.h、icamin_test.csv、gpu_baseline.csv、README.md）+ `test/iamin/icamin/CMakeLists.txt` 增 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)`；CSV 列格式对齐 `test/iamin/icamin/icamin_param.h`（x 列 `NULLPTR` 即空指针负向），golden 复用仓内 `icamin_golden.h` CPU 参考。
- **用例**：任务包 CSV 全量（n 覆盖 0/1/小质数/2 的幂±1/≥2^20；incx 覆盖 1/2/3 与 0/负步长 quick return；全零/正负交替/极端值/Inf/NaN 填充）+ GAUSS_NORM_5_2 正态分布补充（§3.5 要求，gen_csv.py 默认仅均匀，补齐方式与条数写入自测报告）+ 专项：平局（重复最小模元素多索引）、x[0]=NaN（结果=1）、全 NaN（结果=1）、模恰为 FLT_MAX、负 n / result 空指针 / x 空指针负向（dummy 缓冲规则按 §3.5：不按步长预分配，单元素或 nullptr 占位）。
- **兼容性**：代码改动仅 `blas/iamin/arch22/`（新增四件套）、`test/iamin/icamin/`（新增 arch22 工程 + CMakeLists 分支）；不动 arch35（isamin/icamin 均不动）、不动头文件（声明已在 L249）、不动既有测试；`blas/iamin/README.md` 产品支持表补 Atlas A2/A3。
- **私仓交付**：gcw_LxlU8H9L/ops-blas 分支 `feat/icamin-arch22`，邀请 Ascend-CANN 为开发者；`task_submission/` 目录按模板放置自验证步骤、精度/性能/内存报告与日志。

### 风险评估

| 风险 | 应对 |
| --- | --- |
| 小向量 GPU 标杆贴近 launch 地板（n≤512 约 8.1us，门槛 10.3us） | 连续路径小向量单核分支（免多核调度与 reduce 二次 launch）；与 arch35 同场景公开数据交叉对照，实测按 §3.3 口径回填 |
| 归约平局/NaN 仲裁与 golden 不一致 | 语义全部锚定 arch35 已实现模式（strict <、NaN→+Inf、idx0 强制发布、兜底精扫）；专项用例逐条验证（平局/x[0]=NaN/全NaN/FLT_MAX） |
| 跨步路径性能弱（SIMT 标量扫描） | GPU 标杆非连续 case 同样宽松且任务书 incx 仅 {1,2,3} 小步长；正确性优先，性能主攻连续大向量 |
| 赛道已 7 份合入，评审基准被抬高 | 本文差异化在"语义级对齐表"（6 条语义逐条到 arch35 源码行号）与"归约顺序无关性论证"（bit-exact 命题的完整证明），无预填数字 |
| GAUSS_NORM_5_2 补充用例覆盖争议 | 补充 CSV 与生成脚本改动一并提交，自测报告列明条数与覆盖类别 |
