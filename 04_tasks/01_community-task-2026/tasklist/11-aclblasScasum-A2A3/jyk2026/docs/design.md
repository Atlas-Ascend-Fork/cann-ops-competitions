# aclblasScasum 算子设计文档

# 需求背景（required）

## 需求来源
- 任务名称/批次：2026 年 11 月社区任务 — aclblasScasum 算子开发（A2A3）。
- 合入目标仓：https://gitcode.com/cann/ops-blas ，算子代码 `blas/asum/arch22/`，测试代码 `test/asum/scasum/arch22/`。
- 验收流程：cann-competitions 社区任务流程（任务书 §4 交付件 / §5 PR 申请合入）。

## 背景介绍
### 算子功能定位
- aclblasScasum 计算单精度复数向量 x 各元素实部与虚部绝对值之和：`result = Σ (|real(x[k])| + |imag(x[k])|)`，输出非负 FLOAT32 实数标量（Device），逐项非负累加，大 n × 极端值下可溢出到 Inf。
- n≤0 或 incx≤0（含 0 与负步长）时 quick return：不触发 kernel，直接写 result=0.0f 并返回 SUCCESS，不遍历向量。

### 业界对标
- 对标 cuBLAS `cublasScasum`：参数语义、quick return 口径以 cuBLAS 文档为唯一依据；语义参考 Netlib `scasum.f`（https://www.netlib.org/blas/scasum.f）。
- 与同族实数接口 `aclblasSasum`（include/cann_ops_blas.h L237，blas/asum/arch22 已合入）同构（int n / 步长 / result 输出）。

### 工程形态
- 工程模式：Ascend C kernel 直调（host 入口 + 校验 + tiling + device kernel，handle 绑定 stream）。
- 代码落位：`blas/asum/arch22/`（scasum_host.cpp / scasum_kernel.cpp / scasum_kernel.h / scasum_tiling_data.h，在同目录实数 sasum 基础上扩展复数路径；tiling 结构参考 `blas/asum/arch35/sasum_tiling_data.h`）；测试落位 `test/asum/scasum/arch22/`，并在 `test/asum/scasum/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)`（当前仅 arch35 注册，随本任务补齐）。
- 仓内近亲：`blas/asum/arch35/scasum_*`（950PR 复数绝对值和归约已合入，算法/host 校验/tiling 直接移植）；`blas/asum/arch22/sasum_*`（实数同族 arch22 骨架：分核归约与 CMake 接入样例）。
- 现场核验：`include/cann_ops_blas.h` **L239 已有 `aclblasScasum` 声明**（任务书 §2.1 写 L236，行号小幅漂移），签名与 §2.3 逐参一致，本任务无需改公共头。

### 接口规格
- 接口声明（与 include/cann_ops_blas.h L239 既有声明逐参一致）：

```cpp
aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);
```

- 参数规格表（官方栏目）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream），Host 内存 | scalar | - | nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | x 复数元素个数，Host 内存 | scalar | int32 | n≤0（含负维度）合法 quick return：写 result=0.0f 返回 SUCCESS，不触发 kernel | - |
| x | 单精度复数向量（实/虚交错存储），Device 只读 | tensor | COMPLEX64 | 实/虚取值 FLOAT32 全集；n>0 且 incx>0 时 nullptr 返回 INVALID_VALUE | 逻辑 [n]，物理 1+(n−1)×\|incx\| |
| incx | x 步长，Host 内存 | scalar | int32 | incx≤0 合法 quick return（写 result=0.0f，不遍历）；(n−1)×\|incx\| 超 int32 返回 INVALID_VALUE（host int64，issue #805） | - |
| result | (|Re|+|Im|) 之和，Device 标量 | scalar | FLOAT32 | 非负实数（溢出场景可为 Inf）；任意 n（含快返路径）下 nullptr 返回 INVALID_VALUE | 单值 |

### 数学公式
- `result = Σ_{k=1..n} ( |Re(x[k])| + |Im(x[k])| )`（逐项非负累加）；与 §3.2 golden（cblas scasum，Netlib 复数实现）同一计算形态，按浮点阈值判定（非 bit-exact）。

# 需求分析（required）

## 需求描述
- 使用 Ascend C 在 Atlas A2/A3（arch22）上以 kernel 直调实现 aclblasScasum：COMPLEX64 向量的 |Re|+|Im| 绝对值和归约，输出 FLOAT32 标量，n≤0/incx≤0 快速返回 result=0.0f，精度按 mixed_tolerance FLOAT32 阈值（标量比对）对齐 cblas scasum golden，性能达到 cublasScasum（910B3 实测）平均单次耗时的 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可复用的代数量 | 逐元素 (|Re|+|Im|) 非负求和归约；可分解为"各核 UB 内向量指令批量 Abs/Add + 局部部分和 → 跨核汇总"；arch35 scasum 已合入实现整体移植，arch22 sasum 提供实数归约骨架 |
| 2 | 接口原型 | include/cann_ops_blas.h L239 声明已存在（现场核验），无需新增；与 aclblasSasum 同构 |
| 3 | 参数规格 | 见接口规格表；异常码：HANDLE_IS_NULLPTR / INVALID_VALUE；quick return 返回 SUCCESS 且 result=0.0f |
| 4 | 支持数据类型 | x：COMPLEX64；result：FLOAT32 标量（Device） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（arch22）；自验精度 910B3 或 A3、性能 910B3；标杆 cublasScasum |
| 6 | 精度要求 | 标量比对（matched_ratio 退化为单值判定）：rtol=atol=2^-13，max_abs 1e-2 or 32×ULP；归约顺序不要求与 golden 逐位一致，不做 bit-exact；Inf/NaN 等效规则；溢出用例按 Inf 一致性判定 |
| 7 | 性能预期 | msprof op（warmup 5 + 采样 ≥10 取平均）≤ GPU 标杆 1.25 倍；标杆 gpu_baseline.csv |
| 8 | 代码架构选型 + 可行性 | 移植 blas/asum/arch35/scasum：各核 UB 内向量指令批量算 |Re|+|Im|（Abs 实/虚后 Add），tile 内 ReduceSum 局部部分和，跨核汇总一次落 result。已排除死路：单核全量扫描（大规模带宽受限）；float64 中间累加（无精度收益——逐项非负且按 FLOAT32 阈值判定，多一次精度转换徒增开销）；实/虚分两次遍历（一次遍历双通道 Abs 即可） |
| 9 | 评测来源 | golden：cblas scasum（Netlib 复数实现，quick return 用例固定 0.0f 无需调 cblas）；性能标杆：cublasScasum（gpu_baseline.csv）；用例：test_cases CSV + GTest（CSV 列格式对齐 test/asum/sasum/sasum_param.h） |

# 详细设计（required）

## 算子分析
### 数学公式
- `result = Σ ( |Re(x[k])| + |Im(x[k])| )`，k 按 incx 步长取值；逐项非负，累加和单调不减，极端值 × 大 n 可溢出 Inf（golden 同为浮点累加亦溢出，按 Inf 一致性判定）。

### 支持数据类型
- x：COMPLEX64（实/虚各 float32，交错存储，n 复数元素 = 2n float）；result：FLOAT32（Device 标量）。

### 支持形状
- 一维向量，物理长度 1+(n−1)×|incx| 复数元素（仅 incx>0 时按步长访问）。
- 退化：n≤0 → quick return（result=0.0f）；incx≤0 → quick return（result=0.0f，不遍历）。

## 算子实现
### 实现方案
#### 总体架构与数据流
- 两级归约数据流：GM(x 段)→UB（double buffer）→向量指令批量计算 |Re|+|Im|（实/虚双通道 Abs 后 Add）→ tile 内 ReduceSum 累加局部部分和 → 各核部分和写全局汇总缓冲 → 跨核汇总（末核或二段归约）一次写 result（arch35 scasum 同款方案）。

#### host 侧设计
- **分核策略**：n 按 AIV 核数均分（均分 + 余块），每核处理一段逻辑下标（物理下标 = 逻辑 × incx），满核原则；n 小于核数时按实际 n 启用。
- **数据分块和内存优化策略**：x 段分 tile 载入 UB，double buffer；实/虚通道并行 Abs；部分和仅各核落一次 GM，避免原子竞争。
- **tilingkey 规划策略**：无路径分叉需求（incx>0 单一路径，快返在 host 侧短路），不引入 tiling key；n/incx 走 tiling 结构体。
- **数据检测（校验顺序）**：
  1. handle==nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR；
  2. result==nullptr（任意 n，含快返路径）→ INVALID_VALUE；
  3. (n−1)×|incx| 超 int32（host int64，issue #805）→ INVALID_VALUE；
  4. n>0 且 incx>0 时 x==nullptr → INVALID_VALUE；
  5. 校验通过后 n≤0 或 incx≤0 → quick return：写 result=0.0f 返回 SUCCESS（不触发 kernel）。
- **输出下发**：result 为 Device 标量，kernel 直写；异步执行依赖 handle 绑定 stream，读回前须同步 stream。

#### kernel 侧设计
- Init：解析 tiling（n/incx/各核起止/汇总缓冲布局），SetGlobalBuffer，InitBuffer。
- Process：逐 tile CopyIn → 实/虚双通道 Abs + Add → tile 内 ReduceSum 累加部分和 → 各核部分和写全局缓冲 → 跨核汇总 → result 写出。
- 流程图：

```mermaid
flowchart LR
    A[host: 校验+分核+tiling] --> B[各核 CopyIn: x 段 GM→UB]
    B --> C[双通道 Abs + Add]
    C --> D[tile 内 ReduceSum 部分和]
    D --> E{tile 完?}
    E -- 否 --> B
    E -- 是 --> F[部分和写全局缓冲]
    F --> G[跨核汇总 → result]
```

- workspace 布局：跨核汇总全局缓冲（各核部分和，核数量级）；除此无额外 workspace。

#### 性能优化要点（终态）
- 优化项：实/虚双通道并行 Abs；向量指令批量计算 + ReduceSum；UB double buffer；各核仅一次部分和落 GM。目标 ≤1.25× cublasScasum 标杆，实测数据回填自测报告。
- 未采用路线：float64 中间累加（按 FLOAT32 阈值判定无精度收益）；原子操作逐核竞争累加；实/虚分两次遍历。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2/A3 训练系列（910B，arch22） | ✅（本任务目标，自验精度 910B3 或 A3、性能 910B3） |
| Atlas 950PR（arch35） | ❌（arch35 已有 scasum/sasum 实现，本任务不改） |

## 算子约束限制
- 不支持负步长反向遍历（incx≤0 为合法 quick return，result=0.0f）；向量非连续由 incx>0 表达。
- 不涉及原地更新（result 为独立输出）、broadcast、dynamic shape；归约累加顺序不保证与参考实现逐位一致（无确定性要求），按精度阈值判定。
- 异步执行：依赖 handle 绑定 stream，读回 result 前须同步 stream。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出 FLOAT32 标量比对：\|actual−golden\| ≤ atol+rtol×\|golden\|（rtol=atol=2^-13，max_abs 1e-2 or 32×ULP；matched_ratio 退化为单值判定）；golden=cblas scasum；Inf/NaN 等效规则；大 n × 极端值溢出用例按 Inf 一致性判定并单独标注；quick return 用例 golden 固定 0.0f | mixed_tolerance_standard，任务书 §3.2 |
| 性能标准 | COMPLEX64 平均单次耗时 ≤ GPU 标杆（cublasScasum 同规模同 dtype 同步长）×1.25；msprof op 采集，warmup 5 + 有效采样 ≥10 取平均，仅 SUCCESS 计入 | 任务书 §3.3 + test_cases/gpu_baseline.csv |

## 测试概览
- 测试工程：`test/asum/scasum/arch22/` 新建（CSV + C++ GTest；CSV 列格式对齐 test/asum/sasum/sasum_param.h：n / incx / x 填充模式 / result 空指针标记），补 `test/asum/scasum/CMakeLists.txt` arch22 注册。
- 覆盖：尺寸扫描（0/1/小质数/2 的幂 ±1/非对齐/大规模）、步长 ≥1 组合与 incx≤0 快返（含 0 与负步长，期望 result=0.0f & SUCCESS）、n≤0 快返、大 n × 极端值（接近 FLT_MAX）溢出用例（Inf 一致性，单独标注）、±Inf/NaN 分量用例、负向（result/x 空指针、跨度溢出期望 INVALID_VALUE；handle 空指针期望 HANDLE_IS_NULLPTR）、性能用例。
- 输入生成：均匀 [-5,5] 50% + 正态 GAUSS_NORM_5_2 50%（实/虚独立采样）；负向用例不按 (n−1)×|incx| 预分配缓冲；任务自带 gen_csv.py 仅均匀分布，正态用例本任务补齐并在自测报告说明。
- 性能口径：msprof op，warmup 5 + 有效采样 ≥10 取平均；复现入口见交付件测试 README。

## 兼容性分析
- 公共头零改动：`aclblasScasum` 声明已存在于 include/cann_ops_blas.h L239（现场核验，任务书行号 L236 有小幅漂移），本任务不新增不修改任何声明。
- `blas/asum/arch22/` 新增 scasum 文件（与既有实数 sasum 并列）、`test/asum/scasum/arch22/` 新建测试工程与 arch22 CMake 注册，均不影响同目录 sasum 与 arch35 既有行为。
