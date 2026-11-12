# aclblasScasum 算子设计文档（arch22 / Atlas A2/A3）

> 社区任务：11月社区任务-aclblasScasum算子开发（A2/A3）
> 目标仓库：`https://gitcode.com/cann/ops-blas`，目录 `blas/asum/arch22/`
> 说明：本文为最终设计文档，含 A2（910B3）上机实测结果；报告截图见 `task_submission/`。

## 需求来源

在 Atlas A2/A3（arch22）上使用 Ascend C 实现 `aclblasScasum`，对齐 cuBLAS `cublasScasum` / Netlib `scasum`：

```text
result = Σ (|real(x[k])| + |imag(x[k])|)，k = 1 + (i-1)*incx，i = 1..n
```

输入为 complex64 向量，输出为 FLOAT32 实数标量；接口声明已在 `include/cann_ops_blas.h:239` 提供，与 950PR 共用。

## 测试介绍

### 现状

ops-blas 已有：
- arch22 实数实现 `blas/asum/arch22/sasum_host.cpp` / `sasum_kernel.cpp`（AIV + Abs + ReduceSum + AtomicAdd 范式）；
- arch35 complex64 实现 `blas/asum/arch35/scasum_host.cpp` / `scasum_kernel.cpp`（SIMT 范式）；
- arch35 测试 `test/asum/scasum/arch35/`；arch22 原先仅有非 GTest runner。

### 本任务改动（已完成）

- 新增 arch22 complex64 实现：`blas/asum/arch22/scasum_tiling_data.h`、`scasum_host.cpp`、`scasum_kernel.cpp`。
- 新增 arch22 GTest：`test/asum/scasum/arch22/scasum_test.cpp`、`scasum_npu_wrapper.h`、`scasum_test.csv`（1220 条：任务书 1200 条 + 追加 20 条 `GAUSS_NORM_5_2` 正态分布用例）。
- **`include/cann_ops_blas.h` 中 `aclblasScasum` 声明已存在**（第 239 行，与 950PR 共用，本任务无需新增声明，arch22 实现直接对接该公共 API）。
- **测试 CMake 的 arch22 注册已补齐**：`test/asum/scasum/CMakeLists.txt` 现为 `if(("arch22" IN_LIST SOC_ARCH_DIRS) OR ("arch35" IN_LIST SOC_ARCH_DIRS))` 分支统一调用 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)`，A2/A3 构建不再退回非 GTest runner。
- 更新 `blas/asum/README.md`、`docs/zh/api_list.md` 的 A2/A3 支持矩阵。

## 需求分析

### 需求描述

- 功能：单精度复数向量绝对值分量之和，归约为 FLOAT32 标量。
- 语义：`n <= 0` 或 `incx <= 0` 为合法 quick return：不触发 kernel，`result = 0.0f`，返回 `ACLBLAS_STATUS_SUCCESS`。
- 异常：`handle == nullptr` → `HANDLE_IS_NULLPTR`；`result == nullptr`（任意 n）→ `INVALID_VALUE`；`n > 0 && incx > 0 && x == nullptr` → `INVALID_VALUE`；`(n-1)*incx` 超出 int32 → `INVALID_VALUE`。
- 性能：平均单次耗时不高于 A100 标杆的 1.25 倍（≥0.8×）。

### 需求拆解

1. host 侧参数校验、quick return、tiling、tiling 按值随 kernel 下发（API 异步，不在内部同步 stream）。
2. kernel 侧 AIV 向量归约：complex64 视为 2n float，`Abs + ReduceSum`，多核 atomic add。
3. incx==1 连续 DMA；incx>1 成对 gather（Re/Im 不拆散）。
4. arch22 GTest：CSV 驱动、精度/quick return/负向/性能全路径。

## 详细设计

### 算子分析

#### 数学公式

```text
result = Σ (|real(x[k])| + |imag(x[k])|)
```

#### 支持数据类型

- 输入：complex64（`aclblasComplex`，实部/虚部各 float32，交错存储）。
- 输出：FLOAT32 标量。

#### 支持形状

逻辑一维 `[n]`；物理长度 `1 + (n-1)*|incx|` 个复数元素。

### 算子实现

#### host 侧设计

文件：`blas/asum/arch22/scasum_host.cpp`

1. 校验顺序：handle → result → x（仅 n>0 && incx>0）→ quick return → span。
2. quick return：`aclrtMemcpy(result, 4, &zero, 4, H2D)` 后返回 SUCCESS；不解引用 device 指针。
3. span 守卫：`span = (int64)(n-1)*incx`，`span >= INT32_MAX` 返回 INVALID_VALUE（float 视图长度 `2*(span+1)` 需落在 uint32）。
4. tiling：`GetAivCoreCount()` 取核，上限 40，并按 `ceil(n / 1024)` 做小 n 缩核（n ≤ 1024 时单核），避免多核调度/SyncAll/atomic 的固定开销；按 4 复数（8 float）粒度分配 `startOffset[]` / `calNum[]`。
5. 下发：tiling 结构按值随 kernel 参数传入，`scasum_kernel_do(..., tiling, useCoreNum, stream)`；无 per-call `aclrtMalloc`/H2D/free，也不在 API 内 `aclrtSynchronizeStream`（保持异步，调用方按需同步后读 result）。

#### kernel 侧设计

文件：`blas/asum/arch22/scasum_kernel.cpp`

1. complex64 按 2n float 视图：complex i 对应 float 2i（Re）、2i+1（Im）；`result = Σ|float|`。
2. Init：
   - 从按值传入的 tiling 结构解析 n/incx/useCoreNum/startOffset/calNum；
   - `GlobalTensor<float>` 长度 `2*((n-1)*incx+1)`（incx>1）或 `2*n`（incx==1）；
   - 初始化 UB buffer：80KB 输入队列 ×2、ReduceSum workspace、8 float 核内累加器、1 float 输出；
   - 先把 outGM[0] 写 0，`SyncAll()` 保证所有核的 atomic add 在其后。
3. Process（`SetAtomicAdd<float>`）：
   - 按 `maxDataCount=10240` 复数分迭代，tile 间软件流水：先预取下一 tile，再计算当前 tile，让 MTE2 搬运与向量计算重叠；
   - incx==1：`DataCopy` 连续搬入 `2*count` floats；
   - incx>1：按 `(offset+i)*incx*2` 成对 `GetValue` gather；
   - `Abs` + `ReduceSum` 得到 tile 部分和，累加到 UB 标量累加器；
   - 循环结束后每核只做 1 次 atomic add 写回 outGM，避免每 tile 一次 atomic 在单地址上串行化；
   - 尾部按 8 float 对齐补 0；`DataCopyPad` 的 uint16 字节上限用 8192 复数分块规避。
4. `SetAtomicNone()` 结束。

#### 分核策略

- 粒度：4 复数（8 float，32B）。
- 实际核数：`min(n, GetAivCoreCount(), 40, ceil(n/1024))`；n ≤ 1024 时单核，n 较大时用满 40 核。
- `totalBlockNum = n/4`，`remain = n%4`；
- `totalBlockNum==0`：单核处理 remain；
- `totalBlockNum<=cores`：每核一块，最后一块吸收 remain；
- 否则均分，前 `remainBlock` 核多一块，最后一块吸收 remain。

#### 数据分块与 UB

- 输入队列：2 × 80KB；
- ReduceSum workspace：按 20480 float 的最大 repeat 计算约 1.3KB；
- 输出：1 float；
- 总计约 165KB，低于 A2/A3 UB 上限。

#### tilingkey

不涉及：complex64 单一数据类型、incx 运行时分支在 kernel 内处理，无需 tilingkey。

## 支持硬件

| 芯片版本 | 架构 |
|---|---|
| Atlas 800T A2 / Atlas 800I A2 | arch22（ascend910b） |
| Atlas 800T A3 / Atlas 800I A3 | arch22（ascend910_93） |

## 算子约束限制

- 不支持广播；单向量归约。
- 不支持非连续 tensor 的额外 leading dimension；非连续仅由 incx 表达。
- `incx <= 0` 按 cuBLAS/Netlib 语义 quick return，不遍历向量。
- 归约顺序不要求逐位一致，按精度阈值判定。

## 可维可测分析

### 精度标准/性能指标

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | FLOAT32 标量：atol=rtol=2^-13，matched_ratio≥0.99，max_abs_error≤1e-2 或 32 ULP | 生态算子开源精度标准 |
| 性能 | 平均单次耗时 ≤ A100 标杆 1.25 倍（≥0.8×） | 任务书 §3.3 / gpu_baseline.csv |
| 内存 | 不涉及 | 任务书 §3.4 |

### 兼容性分析

新算子，不涉及历史版本兼容；接口与 `aclblasSasum` 同构，声明位于公共头文件 `include/cann_ops_blas.h`（已存在），供其他产品线复用。

## 测试设计

- GTest + CSV：复用 `test/asum/scasum/scasum_param.h` / `scasum_golden.h`。
- 用例：
  - 任务书 `scasum_test.csv` 1200 条；
  - 追加 20 条 `GAUSS_NORM_5_2` 正态分布用例（n=16/64/256/1024/4096/65536，incx=1/2/3），纳入精度自验证；
  - 其中 200 条 `TC_PF_` 前缀为性能用例，精度运行时由 `verify_accuracy.py` 单独统计。
- 路径：
  - 正常：device buffer + CPU golden（double 累加）+ `verifyScalar`；
  - quick return：result=0.0f；
  - 负向：null handle / null result / null x / span 溢出；
  - 性能：`[PERF] case=... n=... incx=... avg_us=...`，warmup 10 + 100 次 aclrtEvent 计时。

## 实测结果（Atlas 800T A2 / 910B3，CANN 9.1.0）

### 精度

`verify_accuracy.py --repo <ops-blas> --soc ascend910b --device 0`（未传 `--csv`，使用仓内合并后 CSV；`--gtest_filter=-*TC_PF*` 排除性能用例）：

- 官方脚本汇总：**PASS=1020 / FAIL=0（共 1020），结论 ALL PASS**，覆盖精度/边界/quick return/负向全路径。
- 含 20 条 `GAUSS_NORM_5_2` 正态分布补充用例，全部通过。
- 日志：`task_submission/2.2 精度自验证日志.log`；报告与截图见 `2.1 精度自验证报告.xlsx`。

### 性能

`verify_performance.py --repo <ops-blas> --soc ascend910b --device 0`；GTest `[PERF]` 输出（warmup 后连续下发取均值，aclrtEvent 计时）。性能优化点：tiling 按值传递、去掉 per-call sync；tile 间软件流水预取；每核只做 1 次 atomic add；小 n 按 `ceil(n/1024)` 缩核；kernel 内清零 + `SyncAll`。

- 官方脚本汇总：**PASS=200 / FAIL=0 / NO_REF=0**（200 条 `TC_PF_` 用例全部达标，判据 ratio = A100 标杆 / NPU 耗时 ≥ 0.8）。
- 全量 ratio：最小值 **1.36**、中位 **1.45**、最大 3.49。
- 任务书 §3.3 对齐的三个大 n 用例（incx=1）实测：

| n | incx | NPU avg_us | A100 标杆 (gpu_ms) | ratio(GPU/NPU) | verdict |
|---|---|---|---|---|---|
| 1048576 | 1 | 13.140 | 0.0198656 | 1.51 | PASS |
| 2097152 | 1 | 15.643 | 0.0221525 | 1.42 | PASS |
| 4194304 | 1 | 20.480 | 0.0332971 | 1.63 | PASS |

> 说明：`[PERF] avg_us` 为 aclrtEvent 计时口径（含下发开销）；如需进一步交叉验证，可用 msprof 的 op 口径采集 kernel 纯执行时间比对。全量结果以 `task_submission/v5_results/results_npu_scasum_v5.csv`、`performance_console_v5.log` 为准；首次提交版本的日志保留在 `3.2 性能自验证日志.log`。

### 内存

任务书 §3.4 明确不涉及内存要求；`4.1/4.2` 交付件按“不涉及”填写。
