# 需求背景（required）

## 需求来源

- 任务：11月社区任务-aclblasScasum算子开发(A2A3)，在 Atlas A2/A3（arch22）上开发单精度复数向量绝对值分量之和算子 `aclblasScasum`。
- 团队及提交账号：skytosky。
- 依据：任务方提供的《aclblasScasum_AtlasA2_A3 任务书》及配套 `test_cases/`（1200 条 CSV 用例、200 条 GPU 基线、生成与校验脚本）。
- 目标代码仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，算子代码入 `blas/asum/arch22/`，测试代码入 `test/asum/scasum/arch22/`；硬件 Atlas A2/A3，CANN 9.1.0。
- 本 PR 为设计文档评审稿；实现代码在评审通过后按 ops-blas 流程单独提交。

## 背景介绍

### aclblasScasum 功能与工程现状

Scasum 计算单精度复数向量各元素实部与虚部绝对值之和 `result = Σ (|Re(x_k)| + |Im(x_k)|)`，对标接口为 cuBLAS `cublasScasum`，计算语义参考 Netlib BLAS `scasum.f`（1-based Fortran 索引、`incx` 步长、`n≤0`/`incx≤0` quick return）。

以本设计核对时的 ops-blas 提交 `5bd9514`（2026-11-10）为工程基线，现状如下：

- 公共接口已就绪：`include/cann_ops_blas.h` L239 已声明 `aclblasScasum`（handle, n, x, incx, result），与 950PR 共用，**本任务无需修改公共头文件**。
- arch35 语义金标准已存在：`blas/asum/arch35/` 已有完整 `scasum_host.cpp / scasum_kernel.cpp / scasum_tiling_data.h`（SIMT 实现，float2 交错视图 + warp 级归约 + SyncAll 跨核），其校验顺序（handle → result → x → 跨度守卫）、quick return（H2D 写 0.0f）、`issue #764`（kernel 下发失败透传）与 `issue #805`（步长跨度 uint32 溢出拒绝）口径可直接移植。
- arch22 同族脚手架已存在：`blas/asum/arch22/` 已有实数 `sasum_host.cpp / sasum_kernel.cpp`（AIV dataflow 模板类、双缓冲、Abs+ReduceSum、浮点原子累加回写），任务书明确以"扩展复数路径"方式复用。
- 测试框架已就绪：`test/asum/scasum/` 已有 `scasum_param.h / scasum_golden.h` 与 arch35 GTest 注册；`CMakeLists.txt` 当前仅有 arch35 分支，**须随本任务补齐 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)` 注册**（任务书同款要求）。
- 已知差距（本设计逐项修复）：现行 `sasum_host.cpp`（arch22）缺少 quick return 与 result/x 空指针校验、固定 8 核分块未吃满 40 AIV、host 侧 tiling H2D 拷贝 + `aclrtSynchronizeStream` 同步式收尾与任务书异步语义不符；本设计按任务书语义实现，不沿用该写法。

本任务采用 Ascend C kernel 直调（句柄式 BLAS 接口），不涉及 TBE 原型、ACLNN 两阶段接口或算子注册 JSON。

# 需求分析（required）

## 需求描述

在 CANN 9.1.0 上实现符合任务书与 cuBLAS `cublasScasum` 语义的 arch22 版本：COMPLEX64 交错存储输入、FLOAT32 标量输出，`incx` 正步长语义与 `n≤0`/`incx≤0` quick return，参数校验返回值与任务书 §2.4 一致；精度满足生态算子 FLOAT32 混合容差标准（标量比对）；200 条 GPU 基线 case 逐条达到 0.8× GPU 性能。

## 需求拆解

1. 功能：`result = Σ_{k=0}^{n-1} (|Re(x_k)| + |Im(x_k)|)`，`x_k` 物理偏移按 `incx` 复数步长；复数 `aclblasComplex` 为 `{float real; float imag;}` 交错存储（8B/元素），`x` 物理长度 `1+(n-1)×|incx|` 个复数元素。
2. 边界与负向：handle 空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；result 为 nullptr（任意 n，含 quick return 路径）→ `ACLBLAS_STATUS_INVALID_VALUE`；`n>0 && incx>0` 时 x 为 nullptr → `ACLBLAS_STATUS_INVALID_VALUE`；`(n-1)×|incx|` 复数跨度超 uint32 float 索引域 → `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，issue #805 同口径）；校验通过后 `n≤0` 或 `incx≤0` 走 quick return（不触发 kernel，result 写 0.0f，返回 SUCCESS）。
3. 特殊值：元素不校验；IEEE FP32 逐项传播；接近 FLT_MAX 的大 n 累加可溢出为 Inf，与 golden 按等效性判定（同为 Inf/NaN 或一方 NaN 一方 ±Inf 视为通过）。
4. 精度：golden 由 cblas（Netlib `scasum`）生成，输出为单 FLOAT32 标量，按标量混合容差判定；归约累加顺序不要求逐位一致（任务书 §3.2 明示，本设计的交错 float 视图口径依赖此条，见详细设计）。
5. 性能：msprof op 采集 task duration，warmup 5 次 + 有效采样 ≥10 次取平均；200 条基线 case 逐条 ≤ `gpu_ms/0.8`。
6. 交付：按任务书第 4 节完成设计文档 PR、自测用例、自测报告（xlsx+log 七件套）与 `task_submission/` 目录，私仓邀请 Ascend-CANN。

# 详细设计（required）

## 算子分析

### 数学公式

以下标 0 基。`x_k` 为第 k 个复数元素（实部 `r_k`、虚部 `i_m`），交错存储于 float 数组 `f[0..2n)`（`f[2k]=r_k, f[2k+1]=i_m`）：

```text
result = Σ_{k=0}^{n-1} ( |r_k| + |i_m| )
```

**关键等价（本设计核心简化）**：上式与"对交错 float 视图全量取绝对值求和"在数学上恒等：

```text
result = Σ_{j=0}^{2n-1} |f[j]|        （f 为 x 的 float 交错视图，共 2n 个 FLOAT32）
```

即 Scasum 在数值上等价于"以 float 视角对 2n 个元素做 Sasum"，**无需区分实部/虚部、无需去交错**。与逐元素配对口径的差异仅是 FP32 加法结合顺序，落在任务书 §3.2"归约累加顺序不要求逐位一致"的豁免范围内。

### 支持数据类型与形状

- 输入 COMPLEX64（float 交错视图计算，不降精度、不引入 TF32/HF32），输出 FLOAT32 标量（Device 内存）。
- n≥0（运行时标量，无 dynamic shape 要求）；`incx≥1` 参与计算，`incx≤0` quick return。
- x 物理跨度（float 视图）`2×(1+(n-1)×|incx|)` 个 FLOAT32；不涉及广播、不涉及原地；result 为独立输出标量。
- 测试规模最大 n=2^22（基线 case scasum-base-002）；设计不据此设置接口上限。

### 地址计算

以 float 为单位、64 位中间量计算，防溢出：

```text
复数 k 的 float 起始偏移：base = int64(k) × int64(incx) × 2      （k = 0..n-1）
float 视图总量：totalFloatNum = 2 × n
跨度守卫：int64(n-1) × int64(incx) × 2 不得超过 uint32 索引域（issue #805 同口径，较 arch35 再乘系数 2）
```

## 算子实现

### 实现方案

核心思路：**把"复数绝对值和"等价改写为"float 视图的 Sasum"**，归约结构与仓内 `blas/asum/arch22/sasum_kernel.cpp` 完全同构（Abs + ReduceSum + 原子累加回写），从而以最小增量风险获得一个已被仓内验证的 AIV dataflow 实现；host 侧校验与 quick return 逻辑从 arch35 `scasum_host.cpp` 移植（口径与 950 完全一致）。整体流程：参数校验 → quick return 判定 → tiling 计算 → 启动 kernel（异步，不等待）→ 返回，读回前由调用方同步 stream。

#### Host 侧设计

参数校验顺序（移植 `blas/asum/arch35/scasum_host.cpp` 现行口径，含 issue #764/#805 修复）：

1. handle 为空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. result 为 nullptr → `ACLBLAS_STATUS_INVALID_VALUE`（任意 n，含 quick return 路径，先于 quick return 判定）；
3. `n>0 && incx>0` 时 x 为 nullptr → `ACLBLAS_STATUS_INVALID_VALUE`；
4. quick return：`n≤0 || incx≤0` → `aclrtMemcpy` H2D 写 `0.0f` → `ACLBLAS_STATUS_SUCCESS`（不启动 kernel；结果为 Device 内存的写零语义，与 arch35 一致）。负 n / 负 incx / 大 |incx|×负 n 组合一律在此成功返回；
5. 跨度守卫（仅正常路径，QR 之后才可达）：`int64(n-1) × int64(incx) × 2 > UINT32_MAX` → `ACLBLAS_STATUS_INVALID_VALUE`（float 视图索引域，较 arch35 复数口径多 ×2，同 issue #805 拒绝式守卫）。守卫后置于 QR 与上游 arch35 现行顺序一致：若先行，负象限组合会因 `(n-1)×incx` 为正误判 INVALID_VALUE（如 n=incx=-65536），且 `n=incx=INT32_MIN` 时 ×2 形式溢出 int64；QR 先行后守卫前提 `incx>0` 恒成立，带符号式与任务书 `|incx|` 字面式在整数域严格等价（2×span_c 恒为偶数，无边界奇偶歧义）；
6. 填充 tiling POD，经 handle 取 stream，`scasum_kernel_do` 分派启动，**不做任何同步等待**（异步语义，任务书 §2.5；修正现行 sasum host 的同步式收尾）。

Tiling 为 POD 结构体随 kernel 参数按值传递（arch22 已验证可行，见 `sasum_kernel_do` 的 arch35-style 重载；不申请 device tiling 缓冲、无 H2D 拷贝）：

```cpp
constexpr uint32_t SCASUM_MAX_CORE_NUM = 40;   // Atlas 800T A2 (910B3) 单芯片 AIV 数

struct ScasumTilingData {
    int64_t  n;          // 复数元素个数
    int64_t  incx;       // 复数步长（原值）
    uint32_t useCoreNum; // 实际持有数据的核数
    uint32_t perCoreN;   // 每核基础复数元素数
    uint32_t remainder;  // 前 remainder 个核各多承担 1 个元素
};
```

分核策略（较现行 sasum 固定 8 核的升级点）：

- `useCoreNum = min(GetAivCoreCount(), n)`（910B3 实测 40 AIV），满核优先；
- 按 `perCoreN = n / useCoreNum`、`remainder = n % useCoreNum` 均分复数元素，各核偏移由 kernel 侧自派生（arch35 同式：`offset(i)=i×perCoreN+min(i,remainder)`），**大 n 全程满核并行**；
- 小 n（n≤40）单核/少核启动，避免核间同步开销倒挂（对应基线 n=1..128 的延迟型 case）。

#### Kernel 侧设计

沿用 `blas/asum/arch22/sasum_kernel.cpp` 的 AIV dataflow 模板类范式（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，`<<<blocks, nullptr, stream>>>` 启动），以 float 为计算类型扩展复数路径，主要结构与 sasum 一致：

1. **Init**：清零 result（`DataCopyPad` 写 0 + `SyncAll`，保证原子累加起点确定），解析 tiling，按 `maxDataCount = 80KB/4B` 规划单次搬运量，`InitBuffer` 双缓冲（`BUFFER_NUM=2`）；
2. **CopyIn**：
   - `incx==1`（性能 case 主路径，float 视图完全稠密）：`DataCopyPad` 连续装载该核分段内的 `2×dataCount` 个 FLOAT32（含尾块 pad），GM→UB 由双缓冲与计算重叠；
   - `incx>1`（正确性路径，基线 case 无此场景）：按复数步长 gather，每复数元素以 `GetValue` 标量循环取 2 个 float（沿用 sasum 现行标量循环口径；预留 `GatherMask` 向量化 gather 作为优化项，仓内 `blas/dgmm/arch22/` 已有用法先例）；
3. **Compute**：`Abs` 全段 → `ReduceSum` 得分段部分和（`workBuf` 按 ReduceSum 对齐要求预留）；
4. **CopyOut**：`SetAtomicAdd<float>` + `DataCopyPad` 4B 原子累加到 result（沿用 sasum 现行跨核累加范式；累加顺序不定，任务书豁免）；全程 `SetAtomicNone` 收尾。

关键正确性约束：

- 交错 float 视图的等价性依赖"输入确为 8B 对齐的复数交错存储"——`aclblasComplex` 定义（`cann_ops_blas_common.h`）与 ACL 复数布局保证此前提；`incx=1` 时 base 指针 8B 对齐，DataCopyPad 以 32B 块粒度搬运、pad 值 0 参与 Abs/ReduceSum 不改变结果；
- 特殊值：Abs/ReduceSum 按 IEEE FP32 传播；Inf/NaN 用例与 golden 等效性判定（任务书 §3.2 补充说明）；大 n×极值的 Inf 溢出场景为非负累加的合法结果，golden 同为浮点累加亦溢出，单独标注；
- 不做去交错、不引入 FMA 收缩；除 Abs/ReduceSum 外不引入额外舍入步骤；
- 尾块与极小 n（n=1、2 等）走同一模板的 pad 分支，不单设特例。

**备选路径（SIMT 移植）**：arch35 `scasum_kernel.cpp` 为 SIMT 实现（`__simt_vf__` float2 视图 + `asc_reduce_add` warp 归约 + SyncAll），仓内 `blas/dot/arch22/cdot_kernel.cpp` 已证明 `__simt_vf__` 在 arch22 可用。若 M4 阶段 AIV 路径性能不达标（见性能优化方案），直接移植该 kernel 作为备选（仅需调整核数上限与 launch 配置），不阻塞主线路径。

### 性能优化方案

本算子为纯带宽瓶颈（单遍读 8B/复数 + 写 4B 标量）。按 `gpu_baseline.csv` 主要大 case 折算带宽需求：

| case | n | 输入读量 | GPU 基线 us | 上限（×1.25）us | 要求聚合读带宽 |
| --- | --- | --- | --- | --- | --- |
| base-000 | 2^20 | 8.4MB | 19.87 | 24.83 | ≥338GB/s |
| base-001 | 2^21 | 16.8MB | 22.15 | 27.69 | ≥606GB/s |
| base-002 | 2^22 | 33.6MB | 33.55 | 41.94 | ≥801GB/s |
| perf-003~ (n≤128) | 1~128 | ≤1KB | ~10.6-12.0 | ~13.3-15.0 | 启动延迟型，靠最小启动路径 |

- 大 n：40 AIV 满核 + 双缓冲连续 `DataCopyPad`，目标聚合读带宽 ≥ 设备实测上限的 75%；**case base-002 要求 ≥801GB/s，是本项目最大性能风险**，M2 阶段先以带宽探针（对齐长度连续读）实测 910B3 可达聚合带宽，若实测不可达则：①开启 `GatherMask`/对齐块调优将 DataCopy 效率推向带宽上限；②仍不达标时按社区流程就该 case 提出基线复议（口径先询任务方，参照 GemmEx PR 先例）。
  **M2 探针实测结论（2026-11-11，910B3 cannlab 实例）**：以同构 sasum（满核 40 AIV + 双缓冲）为载体、`aclrtEventElapsedTime` 计时（warmup 5 + 采样 20），33.6MB（base-002 等读量）单遍内核时间 ~19-22us，对照预算 41.94us **余量约 2×**；实测聚合读带宽 ~1.1-1.55TB/s，满核分核较现行固定 8 核提升 3-4 倍。**带宽风险解除，无需基线复议**；正式 msprof task duration 数据在 M4 补采（本共享实例 msprof op 通道异常，与 GemmEx #2421 相同）。
- 小 n（延迟型）：host 路径无多余拷贝与同步（POD tiling 按值传递 + 异步下发），kernel 单核直达，task duration 预期个位数 us，对 13.3us 上限保留余量；
- 调优与验证手段：msprof 采集 `OpBasicInfo.csv` 按 kernel 名统计 `Task Duration(us)` 均值（同 `test_cases/README.md` 口径）；参数扫描：`maxDataCount`、`BUFFER_NUM`、分核数上限；AIV/SIMT 双路径以 M4 实测数据决策。

### 工程组织与构建安装

```text
include/cann_ops_blas.h                 # L239 声明已存在，不改
blas/asum/README.md                     # 产品支持表更新：Atlas A2/A3 支持
blas/asum/arch22/                       # 新增（blas/CMakeLists.txt 自动收集 *.cpp，无需算子侧 CMake）
    scasum_host.cpp                     # 校验、quick return、tiling、启动（移植 arch35 口径）
    scasum_kernel.cpp                   # AIV 模板类：float 视图 Abs+ReduceSum+原子累加
    scasum_tiling_data.h                # POD tiling（host/kernel 共用）
test/asum/scasum/arch22/                # 新增
    scasum_npu_wrapper.h                # 直调 wrapper（对齐 arch35 命名）
    scasum_test.cpp                     # CSV 驱动 GTest（复用 test/asum/scasum/scasum_param.h、scasum_golden.h）
    scasum_test.csv                     # 随任务 1200 条用例（含补充的正态分布用例）
    gpu_baseline.csv                    # 随任务 200 条性能基线
test/asum/scasum/CMakeLists.txt         # 补 arch22 分支：ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)
```

构建按仓内 `build.sh --soc ascend910b3`（Release）；测试注册修复任务书指出的"同族 CMakeLists 仅 arch35 注册、A2/A3 构建退回非 GTest runner（整数毫秒输出）"问题，使 910B3 输出 GTest 精细计时。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（含 Atlas 800T/I A2，910B3） | √（本次实现与自验；性能测试设备 910B3） |
| Atlas A3 系列产品（含 Atlas 800I/T A3） | √（同 arch22 产物，验收阶段复验） |

## 算子约束限制

- 仅 COMPLEX64 输入、FLOAT32 标量输出；无 alpha/beta、无批量、无广播；x 与 result 不得重叠；
- `n≤0` 或 `incx≤0` 为合法 quick return（写 0.0f 返回成功），不遍历向量；
- 异步执行，读回前须同步 handle 绑定的 stream；
- 不要求逐位确定性（跨核原子累加顺序不定），浮点顺序差异须落在任务书精度标准内；
- 步长间隙、DataCopyPad 对齐 padding 不参与结果（毒化 + 哨兵用例验证）。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验收标准 | 来源 |
| --- | --- | --- |
| 精度 golden | cblas（Netlib `scasum`），单标杆比对，测试框架 `scasum_golden.h` 自实现 | 任务书 §3.1 |
| 标量判定 | abs(actual-golden) ≤ atol + rtol×abs(golden)，rtol=atol=2⁻¹³（1.2e-4） | 任务书 §3.2 |
| 上限 | max_abs_error ≤ 1e-2（或 32×ULP）；CSV `mere_threshold=1.22e-4`、`mare_multiplier=10` | 任务书 §3.2、用例表 |
| 特殊值 | Inf/NaN 等效性判定（同为 NaN/同号 Inf/一方 NaN 一方 ±Inf 通过）；quick return golden=0；大数溢出 Inf 一致性单独标注 | 任务书 §3.2 |
| 性能 | msprof op task duration，warmup 5 次 + 有效采样 ≥10 次平均；200 条逐条 ≤ gpu_ms/0.8 | 任务书 §3.3 |
| 内存 | 无额外 workspace（tiling 按值传递、原子累加直写 result）；记录 UB 用量与 NPU 峰值显存 | 任务书 §3.4 |

测试组织（沿用配套 1200 条 CSV，并按任务书 §3.5-3 补齐）：

- 全量执行 `scasum_test.csv`（TC_L0 基础 / TC_SQ 规格扫描 / 边界 / 负向 / 特殊值分类）；
- **正态分布补充**：`gen_csv.py` 默认仅均匀分布，按任务书补充 `GAUSS_NORM_5_2`（μ∈[-5,5]、σ∈[0.1,2]，实部虚部独立）用例约 50% 配比，扩展生成器并纳入精度自验，补齐方式与数量在自测报告中说明；
- 负向用例遵循 dummy-buffer 规则：期望非 SUCCESS 或 quick return 的 case 不按 `(n-1)×|incx|` 预分配（INT32_MIN 步长在 int 域即溢出），以单元素占位或 nullptr 验证错误码；
- 性能/内存：200 条基线逐条 msprof 计时，关联键 `n/incx`，输出 performance 记录并归档自测日志（sha256 绑定提交号）；
- 内存：记录峰值显存与 UB 用量，验证无越界/泄漏。

## 实施计划与里程碑

| 里程碑 | 内容 | 门禁 |
| --- | --- | --- |
| M1 设计评审 | 本设计文档 PR 合入 cann-ops-competitions | 评审通过 |
| M2 环境与工程打通 | ops-blas 于 910B3 编译部署；arch22 GTest 注册生效；带宽探针实测（风险项前置确认） | 编译零错误、探针数据归档 |
| M3 正确性 | host 校验 + kernel 全路径实现；1200 条 + 正态补充用例精度通过 | 精度达标（含特殊值/负向） |
| M4 性能迭代 | msprof 采集 200 条基线对标；AIV/SIMT 双路径决策；调优扫描 | 逐条 ≤ gpu_ms/0.8 |
| M5 验收交付 | 自测报告七件套、readme、task_submission、私仓邀请 Ascend-CANN、提交 ops-blas PR | 验收通过 |

## 兼容性分析

新增 arch22 实现与测试，不改公共接口签名、枚举值、状态码语义与其他算子行为；`aclblasScasum` 声明已存在（与 950PR 共用），本任务仅补 arch22 实现与测试，与 arch35（950）实现互不影响；`blas/asum/README.md` 产品支持表由"不支持"更新为"支持（Atlas A2/A3）"，与实际验证范围一致。

## 参考资料

- 任务书及配套用例：`aclblasScasum_AtlasA2_A3_task_doc.md`、`test_cases/`（gen_csv.py / scasum_test.csv / gpu_baseline.csv / verify_accuracy.py / verify_performance.py）。
- ops-blas 工程基线 `5bd9514`（2026-11-10）：`blas/asum/arch35/`（scasum SIMT 参考实现与校验口径）、`blas/asum/arch22/`（sasum AIV 脚手架）、`blas/dot/arch22/cdot_kernel.cpp`（arch22 SIMT 可用性先例）、`blas/dgmm/arch22/`（GatherMask 用法先例）、`test/asum/scasum/`（测试框架与 golden）。
- [Netlib scasum.f](https://www.netlib.org/blas/scasum.f)、[cuBLAS Scasum](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-asum)：接口语义、quick return 与特殊值行为参考。
- [生态算子开源精度标准（混合容差）](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
- Ascend C 官方文档：数据搬运（DataCopyPad 非对齐/尾块）、矢量计算 API（Abs/ReduceSum）、`KERNEL_TASK_TYPE_DEFAULT`（www.hiascend.com/document）。
- [社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)、[cann-competitions 设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
