# aclblasCrot 算子设计文档

# 需求背景（required）

## 需求来源
- 任务名称/批次：2026 年 11 月社区任务 — aclblasCrot 算子开发（A2A3）。
- 合入目标仓：https://gitcode.com/cann/ops-blas ，算子代码 `blas/rot/arch22/`，测试代码 `test/rot/crot/arch22/`。
- 验收流程：cann-competitions 社区任务流程（任务书 §4 交付件 / §5 PR 申请合入）。

## 背景介绍
### 算子功能定位
- aclblasCrot 完成单精度复数向量 Givens 平面旋转：对长度 n 的复向量 x、y 逐元素应用 (c, s) 旋转，两向量均原地更新，c 为 float 实标量、s 为复标量。
- 数学公式：`x[i] ← c·x[i] + s·y[i]`；`y[i] ← c·y[i] − conjg(s)·x[i]`（y 更新使用旋转前的原始 x，实现须先暂存）。

### 业界对标
- 对标 cuBLAS `cublasCrot`：参数序列、返回码、快速返回口径以 cuBLAS 官方文档为唯一依据。
- 语义参考 Netlib/LAPACK crot 语义（含 conjg(s)；Netlib BLAS 本身无 crot 例程，golden 按该语义在测试工程内实现）。
- 与同族 `aclblasCsrot`（c 为实、s 为实，include/cann_ops_blas.h L246，blas/rot/arch22 已合入）、`aclblasSrot`（实数，blas/rot/arch35 已合入）同族同工程模式。

### 工程形态
- 工程模式：Ascend C kernel 直调（host 入口 + 校验 + tiling + device kernel，handle 绑定 stream）。
- 代码落位：`blas/rot/arch22/`（crot_host.cpp / crot_kernel.cpp / crot_kernel.h / crot_tiling_data.h，与同目录 csrot 分层一致）；测试落位 `test/rot/crot/arch22/`，并在 `test/rot/crot/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)`（当前仅 arch35 注册，随本任务补齐）。
- 仓内近亲：`blas/rot/arch35/crot_*`（**950PR 复数旋转已合入实现，算法/host 校验/tiling 直接移植**）；`blas/rot/arch22/csrot_*`（arch22 工程骨架与 CMake 接入样例）。
- 现场核验：`include/cann_ops_blas.h` **L250 已有 `aclblasCrot` 声明**，签名与任务书 §2.3 逐参一致（与 950PR 共用，禁止另立私有平行接口，本任务无需改公共头）。

### 接口规格
- 接口声明（与 include/cann_ops_blas.h L250 既有声明逐参一致）：

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const aclblasComplex* s);
```

- 参数规格表（官方栏目）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream），Host 内存 | scalar | - | nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | x/y 元素个数，Host 内存 | scalar | int32 | n≥0；n<0 返回 INVALID_VALUE；n=0 合法快速返回 SUCCESS 且不修改 x/y | - |
| x | 复向量 x，原地更新为 c·x+s·y，Device | tensor | COMPLEX64 | n>0 时 nullptr 返回 INVALID_VALUE | 逻辑 [n]，物理 1+(n−1)×\|incx\| |
| incx | x 步长，可正可负，Host 内存 | scalar | int32 | ≠0 且非 INT32_MIN；(n−1)×\|incx\| 超 int32 返回 INVALID_VALUE（host int64） | - |
| y | 复向量 y，原地更新为 c·y−conjg(s)·x（用原始 x），Device | tensor | COMPLEX64 | n>0 时 nullptr 返回 INVALID_VALUE | 逻辑 [n]，物理 1+(n−1)×\|incy\| |
| incy | y 步长，Host 内存 | scalar | int32 | 同 incx | - |
| c | 旋转余弦标量（实数）指针，Host 或 Device 内存 | scalar | FLOAT32 | n>0 时 nullptr 返回 INVALID_VALUE；值不校验 | [1] |
| s | 旋转正弦标量（复数）指针，Host 或 Device 内存 | scalar | COMPLEX64 | n>0 时 nullptr 返回 INVALID_VALUE；值不校验 | [1] |

- c/s 内存位置：Host 或 Device 均可，运行时经 aclrtPointerGetAttributes 自动判定（对齐 blas/rot/README.md 同族 aclblasSrot 约束口径）。

### 数学公式
- `x' = c·x + s·y`，`y' = c·y − conjg(s)·x`（逐元素，x' 计算使用原始 y、y' 计算使用原始 x）；与 §3.2 golden（测试工程内按 Netlib/LAPACK crot 语义实现）同一计算形态。

# 需求分析（required）

## 需求描述
- 使用 Ascend C 在 Atlas A2/A3（arch22）上以 kernel 直调实现 aclblasCrot：COMPLEX64 dtype、正负步长、n=0 快速返回、c/s 双内存位置探测，精度按 mixed_tolerance FLOAT32（实部/虚部分别）对齐 crot 语义 golden，性能达到 cublasCrot（910B3 实测）平均单次耗时的 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可复用的代数量 | 逐元素 Givens 旋转，无依赖链可按核并行；arch35 crot 已合入实现整体移植（含 conjg(s) 与原始 x 暂存顺序），arch22 csrot 提供工程骨架 |
| 2 | 接口原型 | include/cann_ops_blas.h L250 声明已存在（现场核验），无需新增；参数序列与 cublasCrot 一一对应 |
| 3 | 参数规格 | 见接口规格表；异常码：HANDLE_IS_NULLPTR / INVALID_VALUE / SUCCESS（n=0 快返） |
| 4 | 支持数据类型 | x/y：COMPLEX64；c：FLOAT32；s：COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（arch22）；自验精度 910B3 或 A3、性能 910B3；标杆 cublasCrot |
| 6 | 精度要求 | 实部/虚部分别 FLOAT32：rtol/atol=2^-13，ratio 0.99，max_abs 1e-2 or 32×ULP；误差仅两次复数乘加舍入，正常实现远优于阈值；Inf/NaN 等效规则 |
| 7 | 性能预期 | msprof op（warmup 5 + 采样 ≥10 取平均）≤ GPU 标杆 1.25 倍；标杆 gpu_baseline.csv |
| 8 | 代码架构选型 + 可行性 | 移植 blas/rot/arch35/crot_*：UB 载入 x/y 段，寄存器暂存原始 x，向量指令完成复数乘加后写回。已排除死路：x/y 就地顺序更新不做暂存（y 更新会读到已改写的 x，违反原始 x 语义）；GM 上二次遍历暂存（UB 暂存即可，避免额外访存） |
| 9 | 评测来源 | golden：测试工程内按 Netlib/LAPACK crot 语义（含 conjg(s)）实现（Netlib 无 crot 例程）；性能标杆：cublasCrot（gpu_baseline.csv）；用例：test_cases CSV + GTest |

# 详细设计（required）

## 算子分析
### 数学公式
- 逐元素：`x'[i] = c·x[i] + s·y[i]`；`y'[i] = c·y[i] − conjg(s)·x[i]`，i=0..n−1（0 基）；s 的复共轭仅作用于 y 更新项。

### 支持数据类型
- x/y：COMPLEX64；c：FLOAT32；s：COMPLEX64。

### 支持形状
- 一维向量，物理长度 1+(n−1)×|incx| / 1+(n−1)×|incy|。
- 退化：n=0 → 快速返回 SUCCESS（不修改 x/y）；n<0 → INVALID_VALUE。

## 算子实现
### 实现方案
#### 总体架构与数据流
- 数据流：GM(x 段, y 段)→UB（double buffer）→寄存器暂存原始 x→计算 x'、y'→GM 写回。单次遍历同时产出两个输出，无跨核依赖。

#### host 侧设计
- **分核策略**：n 按 AIV 核数均分（均分 + 余块归末核），连续段（|inc|=1）按对齐块切块，跨步段按逻辑下标均分，满核原则。
- **数据分块和内存优化策略**：x/y 段分 tile 载入 UB，double buffer；每 tile 内一次暂存原始 x、一次算出 x'/y'，访存量 2R+2W 每元素，无冗余遍历。
- **tilingkey 规划策略**：连续（incx==incy==1）与跨步两路径区分（连续走 DataCopy 对齐块，跨步走逐元素下标计算）；c/s 的 Host/Device 来源在 host 侧消解，kernel 不感知。
- **数据检测（校验顺序，#765/#861 口径）**：
  1. handle==nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR（最优先，quick return 不得先于判空）；
  2. n<0 → INVALID_VALUE；n==0 → 直接 SUCCESS（不修改 x/y）；
  3. incx/incy==0 或 INT32_MIN、(n−1)×|inc| 超 int32（host int64）→ INVALID_VALUE；
  4. n>0 时 x/y/c/s 任一 nullptr → INVALID_VALUE。
- **c/s 处理（双内存位置）**：运行时 aclrtPointerGetAttributes 判定 Host/Device；Host 指针同步解引用、Device 指针经 stream 读取，统一转为值随 tiling 下发（对齐同族 srot/crot 口径）；标量值不校验（含 Inf/NaN，按 IEEE 自然传播）。

#### kernel 侧设计
- Init：解析 tiling（n/incx/incy/各核起止/c/s 值），SetGlobalBuffer，InitBuffer。
- Process：CopyIn x/y 段 → 暂存原始 x → `x' = c·x + s·y`（复数乘法展开 4 次 float 乘加）→ `y' = c·y − conjg(s)·x_orig` → CopyOut。跨步路径按逻辑下标 → 物理下标映射（负步长自尾端起算）。
- 流程图：

```mermaid
flowchart LR
    A[host: 校验+c/s 双位置探测] --> B[分核: n 均分]
    B --> C[CopyIn: x 段+y 段 GM→UB]
    C --> D[暂存原始 x]
    D --> E[x'=c·x+s·y; y'=c·y-conjg s·x]
    E --> F[CopyOut: x',y' 写回 GM]
    F --> G{tile 完?}
    G -- 否 --> C
    G -- 是 --> H[结束]
```

- workspace 布局：无额外 workspace（原始 x 暂存在 UB/寄存器，x/y 原地覆写）。

#### 性能优化要点（终态）
- 优化项：x/y double buffer 流水；原始 x 暂存于寄存器/UB 免二次访存；连续路径对齐块 DataCopy。目标 ≤1.25× cublasCrot 标杆，实测数据回填自测报告。
- 未采用路线：GM 中转暂存 x（多余 1R1W）；两输出分两个 kernel（重复载入 x/y）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2/A3 训练系列（910B，arch22） | ✅（本任务目标，自验精度 910B3 或 A3、性能 910B3） |
| Atlas 950PR（arch35） | ❌（arch35 已有 crot 实现，本任务不改） |

## 算子约束限制
- 向量非连续由 incx/incy 表达（含负步长，自尾端起算）；零步长与 INT32_MIN 拦截。
- x/y 均原地更新不返回视图；y 更新必须使用旋转前原始 x（实现侧暂存保证）。
- 不涉及 broadcast / dynamic shape；n=0 快速返回。
- 异步执行：依赖 handle 绑定 stream，读结果前须同步 stream；c/s 标量在提交前消解，Device 标量经 stream 读取。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | x/y 实部/虚部分别 FLOAT32：rtol=atol=2^-13(1.2e-4)，ratio 0.99，max_abs 1e-2 or 32×ULP；golden 在测试工程内按 Netlib/LAPACK crot 语义（含 conjg(s)）实现；Inf/NaN 等效规则单独判定 | mixed_tolerance_standard，任务书 §3.2 |
| 性能标准 | COMPLEX64 平均单次耗时 ≤ GPU 标杆（cublasCrot）×1.25；msprof op 采集，warmup 5 + 有效采样 ≥10 取平均，仅 SUCCESS 计入 | 任务书 §3.3 + test_cases/gpu_baseline.csv |

## 测试概览
- 测试工程：`test/rot/crot/arch22/` 新建（CSV + C++ GTest，参照仓内 test/rot/arch35 模式），补 `test/rot/crot/CMakeLists.txt` arch22 注册。
- 覆盖：尺寸扫描（0/1/小质数/2 的幂 ±1/非对齐/大规模）、步长 ±1/±2/±3 与正负组合、标量特殊值（c=0/1、s=(0,0)/(1,0)/纯虚/大值）、边界与负向（n=0 期望 SUCCESS、空指针/零步长/INT32_MIN/跨度溢出期望 INVALID_VALUE、handle 空指针期望 HANDLE_IS_NULLPTR）、Inf/NaN 特殊值用例、性能用例。
- 输入生成：均匀 [-5,5] 50% + 正态 GAUSS_NORM_5_2 50%（实/虚独立采样）；负向用例不按 (n−1)×|inc| 预分配缓冲（dummy 占位）；任务自带 gen_csv.py 仅均匀分布，正态用例本任务补齐并在自测报告说明。
- 性能口径：msprof op，warmup 5 + 有效采样 ≥10 取平均；复现入口见交付件测试 README。

## 兼容性分析
- 公共头零改动：`aclblasCrot` 声明已存在于 include/cann_ops_blas.h L250（现场核验），本任务不新增不修改任何声明。
- `blas/rot/arch22/` 新增 crot 文件、`test/rot/crot/arch22/` 新增测试工程与 arch22 CMake 注册，均不影响同目录 csrot 与 arch35 既有行为。
