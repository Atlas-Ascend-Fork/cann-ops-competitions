# aclblasIsamin 算子设计文档

# 需求背景（required）

## 需求来源
- 任务名称/批次：2026 年 11 月社区任务 — aclblasIsamin 算子开发（A2A3）。
- 合入目标仓：https://gitcode.com/cann/ops-blas ，算子代码 `blas/iamin/arch22/`（新建，与 arch35 同族同目录分层），测试代码 `test/isamin/arch22/`。
- 验收流程：cann-competitions 社区任务流程（任务书 §4 交付件 / §5 PR 申请合入）。

## 背景介绍
### 算子功能定位
- aclblasIsamin 在单精度实数向量 x 中查找绝对值最小元素的最先出现位置：`result = argmin_i |x[j]|`（j = (i−1)·incx，0 基展开），返回 1-based int 索引；并列取最先出现（最小索引）；结果写 Device 标量。
- n=0 或 incx<1 时快速返回：result 写 0 并返回 SUCCESS；n<0 返回 INVALID_VALUE。
- NaN 语义：首元素为 NaN 时基准置 FLT_MAX，后续元素中 NaN 跳过（比较式 `!(absVal != absVal) && absVal < minAbs`），NaN 不被选为最小元素；全部元素为 NaN 时结果为 1。

### 业界对标
- 对标 cuBLAS `cublasIsamin`（cuBLAS 官方文档 §2.5.2）：参数语义、返回码、快速返回口径以该文档为唯一依据；Netlib 参考库无 isamin.f。
- 与同族复数接口 `aclblasIcamin`（blas/iamin/arch35 已实现）、实数 `aclblasIsamax`（blas/iamax/arch22 已合入）同族同工程模式。

### 工程形态
- 工程模式：Ascend C kernel 直调（host 入口 + 校验 + tiling + device kernel，handle 绑定 stream）。
- 代码落位：`blas/iamin/arch22/`（isamin_host.cpp / isamin_kernel.cpp / isamin_kernel.h / isamin_tiling_data.h）；测试落位 `test/isamin/arch22/`，并在 `test/isamin/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} isamin_test)`（当前仅 arch35 注册，随本任务补齐）。
- 仓内近亲：`blas/iamin/arch35/isamin_*`（950PR 实数绝对值最小归约已合入，算法/host 校验/tiling 直接移植）；`blas/iamax/arch22/isamax_*`（实数 argmax 同级归约 arch22 骨架与 CMake 接入样例）。
- 现场核验：`include/cann_ops_blas.h` **L242 已有 `aclblasIsamin` 声明**（任务书 §2.3 写 L240，行号小幅漂移），签名与 §2.3 逐参一致，本任务无需改公共头。

### 接口规格
- 接口声明（与 include/cann_ops_blas.h L242 既有声明逐参一致）：

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

- 参数规格表（官方栏目）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream），Host 内存 | scalar | - | nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | 向量元素个数，Host 内存 | scalar | int32 | n≥0；n<0 返回 INVALID_VALUE；n=0 或 incx<1 快速返回：result 写 0 返回 SUCCESS | - |
| x | 输入向量，Device 只读 | tensor | FLOAT32 | 全值域（含 Inf/NaN，比较语义见上）；n>0 且 incx≥1 时 nullptr 返回 INVALID_VALUE | 一维，物理 1+(n−1)×\|incx\| |
| incx | x 步长，Host 内存 | scalar | int32 | incx≥1 正常计算；incx<1 快速返回（result=0，不报错）；(n−1)×incx 超 int32 返回 INVALID_VALUE（host int64） | - |
| result | 绝对值最小元素最先出现位置（1-based），Device 标量 | scalar | INT32 | 正常路径 [1,n]；快速返回为 0；任意 n（含快返路径）下 nullptr 返回 INVALID_VALUE | [1] |

### 数学公式
- `result = 1 + min{ i : |x[j]| = min_k |x[k]| }`（0 基 i，1 基输出，NaN 不入选）；与 §3.2 golden（测试工程内按 cuBLAS 语义编写的 CPU 参考实现，float32 直接计算）同一判定形态，bit-exact。

# 需求分析（required）

## 需求描述
- 使用 Ascend C 在 Atlas A2/A3（arch22）上以 kernel 直调实现 aclblasIsamin：FLOAT32 向量绝对值最小索引归约，输出 1-based int（bit-exact），n=0/incx<1 快速返回 result=0，NaN 跳过语义，性能达到 cublasIsamin（910B3 实测）平均单次耗时的 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可复用的代数量 | abs 归约 argmin；分解为"逐核局部 (模值, 索引) 归约 → 跨核汇总"，平局规则最先出现胜出；arch35 isamin 已合入实现（含 NaN 跳过语义）整体移植，iamax arch22 提供同级归约骨架 |
| 2 | 接口原型 | include/cann_ops_blas.h L242 声明已存在（现场核验），无需新增；与 aclblasIcamin/aclblasIsamax 同族对齐 |
| 3 | 参数规格 | 见接口规格表；异常码：HANDLE_IS_NULLPTR / INVALID_VALUE；快速返回返回 SUCCESS 且 result=0 |
| 4 | 支持数据类型 | x：FLOAT32；result：INT32 标量（Device） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（arch22）；自验覆盖 910B3（性能设备 Atlas 800I A2 910B3）；标杆 cublasIsamin |
| 6 | 精度要求 | 输出 INT32 索引按 actual==golden 精确比对（matched_ratio 1.0 / max_abs_error 0 为该精确比对的形式化表述），无浮点容差；golden=测试工程内按 cuBLAS 语义编写的 CPU 参考实现（float32 直接计算，不涉及收窄） |
| 7 | 性能预期 | msprof op（warmup 5 + 采样 ≥10 取平均）≤ GPU 标杆 1.25 倍；标杆 gpu_baseline.csv |
| 8 | 代码架构选型 + 可行性 | 移植 blas/iamin/arch35/isamin：各核 UB 内向量指令批量算 Abs + 局部 argmin（严格小于更新保最小索引，NaN 跳过），再跨核汇总。已排除死路：单核全量扫描（大规模带宽受限）；Abs 后走浮点 min 容差比对（输出是索引，必须精确）；NaN 参与最小值比较（违反 cuBLAS 语义，须显式跳过） |
| 9 | 评测来源 | golden：测试工程内 CPU 参考实现（按 cuBLAS 文档语义）；性能标杆：cublasIsamin（gpu_baseline.csv）；用例：test_cases CSV + GTest |

# 详细设计（required）

## 算子分析
### 数学公式
- 归约：`result = 1 + min{ i : |x[j]| == min_k |x[k]| }`，比较基于 float32 绝对值，NaN 元素跳过、不入选；全 NaN 向量结果为 1。

### 支持数据类型
- x：FLOAT32；result：INT32（Device 标量）。

### 支持形状
- 一维向量，物理长度 1+(n−1)×|incx|（仅 incx≥1 时按步长访问）。
- 退化：n=0 → 快速返回（result=0）；incx<1 → 快速返回（result=0，不进入计算）；n<0 → INVALID_VALUE。

## 算子实现
### 实现方案
#### 总体架构与数据流
- 两级归约数据流：GM(x 段)→UB（double buffer）→向量指令批量计算 Abs → 各核 UB 内局部 argmin（严格小于更新 + NaN 跳过）→ 各核局部 (abs, 索引) 写全局汇总缓冲 → 跨核汇总得全局 (最小 abs, 最小索引) → +1 写 result（arch35 isamin 同款方案）。

#### host 侧设计
- **分核策略**：n 按 AIV 核数均分（均分 + 余块），每核处理一段逻辑下标（物理下标 = 逻辑 × incx），满核原则；n 小于核数时按实际 n 启用。
- **数据分块和内存优化策略**：x 段分 tile 载入 UB，double buffer；abs 与局部 argmin 在寄存器/UB 内完成，仅各核最终局部结果落一次 GM。
- **tilingkey 规划策略**：无路径分叉需求（incx≥1 单一路径，快返在 host 侧短路），不引入 tiling key；n/incx 走 tiling 结构体。
- **数据检测（校验顺序，对齐 arch35 isamin_host.cpp 校验入口 L29-53）**：
  1. handle==nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR；
  2. n<0 → INVALID_VALUE；
  3. result==nullptr（任意 n，含快返路径）→ INVALID_VALUE；
  4. (n−1)×incx 超 int32（host int64）→ INVALID_VALUE；
  5. n==0 或 incx<1 → 快速返回：写 result=0 返回 SUCCESS（不进入计算）；
  6. n>0 且 incx≥1 时 x==nullptr → INVALID_VALUE。
- **输出下发**：result 为 Device 标量，kernel 直写；异步执行依赖 handle 绑定 stream，读回前须同步 stream。

#### kernel 侧设计
- Init：解析 tiling（n/incx/各核起止/汇总缓冲布局），SetGlobalBuffer，InitBuffer。
- Process：逐 tile CopyIn → 向量指令批量 Abs → 局部 argmin（首元素 NaN 时基准置 FLT_MAX；比较式 `!(absVal != absVal) && absVal < minAbs` 跳过 NaN 并保最小索引）→ 各核写局部结果 → 跨核汇总 → result = 全局索引 +1 写出。
- 流程图：

```mermaid
flowchart LR
    A[host: 校验+分核+tiling] --> B[各核 CopyIn: x 段 GM→UB]
    B --> C[向量指令批量 Abs]
    C --> D[局部 argmin: 严格小于更新+NaN 跳过]
    D --> E{tile 完?}
    E -- 否 --> B
    E -- 是 --> F[局部结果写全局缓冲]
    F --> G[跨核汇总 → result=idx+1]
```

- workspace 布局：跨核汇总全局缓冲（各核局部 (abs, 索引) 对，核数量级）；除此无额外 workspace。

#### 性能优化要点（终态）
- 优化项：向量指令批量 Abs；UB double buffer；各核仅一次局部结果落 GM。目标 ≤1.25× cublasIsamin 标杆，实测数据回填自测报告。
- 未采用路线：单核全量扫描；浮点容差比对（输出是索引，必须精确）；NaN 参与比较（语义错误）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2/A3 训练系列（910B，arch22） | ✅（本任务目标，自验覆盖 910B3） |
| Atlas 950PR（arch35） | ❌（arch35 已有 isamin/icamin 实现，本任务不改） |

## 算子约束限制
- 不支持负步长反向遍历（incx<1 为合法快速返回，result=0）；向量非连续由 incx≥1 表达。
- 不涉及原地更新（result 为独立输出）、broadcast、dynamic shape；归约划分顺序不要求固定，但须满足"最先出现（最小索引）"语义与 golden bit-exact 一致。
- 异步执行：依赖 handle 绑定 stream，读回 result 前须同步 stream。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出 INT32 索引按 actual==golden 精确比对（matched_ratio 1.0 / max_abs_error 0 为其形式化表述），无浮点容差；golden=测试工程内按 cuBLAS 语义编写的 CPU 参考实现（float32 直接计算）；NaN 跳过语义与 golden 一致；快返路径 result==0 | 任务书 §3.2 |
| 性能标准 | FLOAT32 平均单次耗时 ≤ GPU 标杆（cublasIsamin 同规模同 dtype 同步长）×1.25；msprof op 采集，warmup 5 + 有效采样 ≥10 取平均，仅 SUCCESS 计入 | 任务书 §3.3 + test_cases/gpu_baseline.csv |

## 测试概览
- 测试工程：`test/isamin/arch22/` 新建（CSV + C++ GTest，参照仓内 test/isamin/ arch35 工程模式与 isamin_param.h/isamin_golden.h 范式），补 `test/isamin/CMakeLists.txt` arch22 注册。
- 覆盖：尺寸扫描（0/1/小质数/2 的幂 ±1/非对齐/大规模）、步长 ≥1 组合与 incx<1 快返（期望 result=0 & SUCCESS）、平局构造用例（重复绝对值取最先出现）、NaN 语义用例（首元素 NaN 基准 FLT_MAX、中间 NaN 跳过、全 NaN 结果 1）、±Inf/±0 特殊值、负向（n<0、result/x 空指针、跨度溢出期望 INVALID_VALUE；handle 空指针期望 HANDLE_IS_NULLPTR）、性能用例。
- 输入生成：均匀 [-5,5] 50% + 正态 GAUSS_NORM_5_2 50%；负向用例不按 (n−1)×incx 预分配缓冲（dummy 占位）；任务自带 gen_csv.py 仅均匀分布，正态用例本任务补齐并在自测报告说明。
- 性能口径：msprof op，warmup 5 + 有效采样 ≥10 取平均；复现入口见交付件测试 README。

## 兼容性分析
- 公共头零改动：`aclblasIsamin` 声明已存在于 include/cann_ops_blas.h L242（现场核验，任务书行号 L240 有小幅漂移），本任务不新增不修改任何声明。
- `blas/iamin/arch22/` 新建目录与 isamin 文件、`test/isamin/arch22/` 新建测试工程与 arch22 CMake 注册，均不影响 arch35 既有行为。
