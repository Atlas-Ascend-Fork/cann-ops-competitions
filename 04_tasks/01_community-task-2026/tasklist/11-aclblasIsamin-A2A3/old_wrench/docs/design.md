# aclblasIsamin 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasIsamin算子开发（A2A3）；算子 aclblasIsamin（对标 cuBLAS cublasIsamin，CUBLAS_Library.pdf §2.5.2；Netlib 参考库无 isamin.f，golden 按 cuBLAS 文档语义由 CPU 参考实现生成）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/iamin/arch22/`（**新建**）+ `test/isamin/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-ops-competitions MR）。
- 环境：CANN 9.1.0；性能设备 Atlas 800T A2 (910B3)；接口声明已在 `include/cann_ops_blas.h`（L240，与 950PR 共用），不改公共头。

## 背景介绍
### 算子功能定位
单精度**实数**向量绝对值最小元素索引归约：`result = argmin(i=1..n) |x[j]|`（j = 1+(i−1)×incx），并列时取**最先出现**的 i，输出 1-based int 索引；n=0 或 incx<1 时 result=0 quick return。补齐 Blas 仓 Atlas A2/A3 BLAS-1 实数最小索引归约缺口。

### 业界对标
- cublasIsamin（CUBLAS_Library.pdf §2.5.2）：参数语义、返回码与快速返回口径**唯一依据**。
- Netlib 参考库无 isamin.f：golden 由按 cuBLAS 文档语义编写的 CPU 参考实现生成（测试工程内直接计算 1-based 索引）。
- NaN 语义（任务书 §2.1 明确）：首元素为 NaN 时基准置 FLT_MAX，后续元素中的 NaN 跳过（`!(absVal != absVal) && absVal < minAbs`），NaN 不会被选为最小元素；若全部元素均为 NaN，结果为 1。

### 工程形态
- Kernel 直调（与仓内 `blas/iamin/arch35` 既有实现同一工程模式：host 入口 + tiling + device kernel）。
- 代码落位：`blas/iamin/arch22/`（新建，arch22 目录需新建；参考 arch35 兄弟实现）。
- 仓内近亲：`blas/iamin/arch35/isamin_host.cpp`（校验入口 L29-53，行为对齐锚点）+ `isamin_kernel/tiling`（实数最小索引归约全套，直接平移 arch22）；`blas/iamax/arch22/isamax_*`（arch22 侧索引归约工程先例）。
- 测试工程 `test/isamin/arch22/` 新建，并在 `test/isamin/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} isamin_test)` 注册（任务书 §2.2 明确要求）。

### 接口规格
```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```
（声明见公共头 L240，与 950PR 共用声明，禁止另立私有平行接口）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | 非空；nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | 向量元素个数 | int | attr | 任意 int32；n<0 → INVALID_VALUE；n=0 或 incx<1 → quick return（result=0，SUCCESS） | - |
| x | 输入向量（Device，只读） | const float* | FLOAT32 | 一维，物理长度 1+(n−1)×\|incx\|；全值域含 Inf/NaN；n>0 且 incx≥1 时非空 | [1+(n−1)·\|incx\|] |
| incx | x 步长 | int | attr | incx≥1 正常计算；incx<1 quick return（result=0）；(n−1)×incx 超 int32 → INVALID_VALUE | - |
| result | 输出 1-based 索引（Device） | int* | INT32 | 独立输出 shape [1]；正常 [1,n]，quick return 为 0；nullptr（任意 n，含快速返回路径）→ INVALID_VALUE | [1] |

### 数学公式
result = argmin_{i=1..n} |x[j]|，j = 1+(i−1)×incx；并列时取最先出现的 i；result 为 1-based 索引，正常 [1,n]；n=0 或 incx<1 → result=0。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调在 Atlas A2/A3（arch22/ascend910b）实现 aclblasIsamin 全语义（绝对值最小索引、首见并列语义、NaN 跳过语义、quick return、int64 跨度守卫），输出与 golden **bit-exact 精确相等**，性能单 case 达 0.8×A100 cublasIsamin。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | argmin(\|x\|) + 首见索引；arch35 isamin 全套（host 校验 L29-53 + tiling + kernel）平移 arch22，arch22 isamax 提供工程先例——纯实数，无复数机制，改动量最小 |
| 2 | 接口原型 | 见上节；声明已在公共头 L240，不改头文件 |
| 3 | 参数规格 | 见参数规格表；校验分支逐条对齐 blas/iamin/arch35/isamin_host.cpp 校验入口 L29-53；返回码口径以 cuBLAS 官方文档为准（§2.4/§2.5） |
| 4 | 支持数据类型 | 输入 FLOAT32 / 输出 INT32 索引 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / ascend910b*；自验仅需覆盖 910B3（§3.1） |
| 6 | 精度要求 | **bit-exact 整数索引精确比对**（actual==golden，matched_ratio 1.0、max_abs_error 0 的形式化表述）；x 的 FLOAT32 值仅为生成域不参与比对；NaN 跳过语义按 §2.1（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv（§3.3）；FLOAT32 输入场景 |
| 8 | 代码架构选型+可行性 | 多核局部 argmin + 全局归约两段式（arch35 同款）；已排除死路：原子写 result（破坏首见索引语义）、host 遍历（数据在 Device）、欧氏/其他模口径（实数无此问题，Abs 即 SCABS1） |
| 9 | 评测来源 | 精度 golden=测试工程内 CPU 参考实现（cuBLAS 文档语义，直接计算 1-based）；性能标杆=gpu_baseline.csv（cublasIsamin 同规模同 dtype 同步长） |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景。两段归约：段内 argmin（严格小于更新；NaN 处理按 `!(absVal != absVal) && absVal < minAbs` 跳过，首元素 NaN 时基准置 FLT_MAX）→ 各核局部 (minAbs, minIdx) 写 workspace → 核间归约取全局最小（并列取最先出现的小索引）→ result 写 1-based 索引。

### 支持数据类型
输入 FLOAT32（单精度实数）；输出 INT32 标量索引。

### 支持形状
n≥0 任意；n=0 quick return；incx≥1（覆盖 1、2、7、16）；incx<1 quick return；n 覆盖 {0,1,2,3,1024,16384,262144,1048576} 及溢出负向组合（n=4 且 incx=INT_MAX、n=2³⁰ 且 incx=8）；±0、重复元素（并列）、NaN、Inf 输入均支持。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| isamin_multi_core（唯一变体） | n>0 且 incx≥1 | 各核分片：MTE2 搬本核 float 段（步长 incx 寻址）→ V 求绝对值 → 标量域严格小于比较（NaN 跳过，首见索引保留）→ 局部 (minAbs, minIdx) 写 workspace（每核 2 word）→ 尾核全局归约取 min（并列取小 idx）写 result |

数据流：GM(x) → MTE2 → UB（float 段双缓冲）→ V Abs → 标量比较 → workspace（GM 小缓冲）→ 全局归约 → MTE3 写 result（int 标量）。步长 incx>1 时按元素步进寻址（间隔元素跳过），MTE2 Gather 场景退化为标量/小段搬运，S5 按 incx 分布精调。

#### host 侧设计
- 分核策略：numBlocks=min(GetAivCoreCount(), 上限)；n 均分 + 尾核补余（perCoreN/lastCoreN，arch35 CalcIsaminTiling 同式）。
- 数据分块与内存优化：单核 UB 双缓冲；纯向量带宽瓶颈；incx>1 时有效流量 = n×4B，与物理跨度无关。
- tilingkey 规划：无独立 tilingkey；TilingData 字段：totalN/useCoreNum/incx/perCoreN/lastCoreN（arch35 同构）。
- 数据检测（校验分支，逐条对齐 arch35 isamin_host.cpp L29-53 + 任务书 §2.5）：handle 判空（HANDLE_IS_NULLPTR）→ n<0（INVALID_VALUE）→ result 判空（INVALID_VALUE，任意 n）→ n=0 或 incx<1 quick return（aclrtMemsetAsync 写 result=0，issue #738 口径：result 为 device 地址，不得 host 解引用）→ x 判空（n>0 且 incx≥1）→ 跨度守卫 (n−1)×incx∈[0,INT32_MAX]（int64 计算）。异步下发，读结果前须同步 stream。

#### kernel 侧设计
```
isamin_multi_core:
  [阶段1 局部] for 本核子区间按块:
    [MTE2] float 段（incx 步进寻址）搬 UB
    [V] Abs（向量）
    [标量] !(absVal != absVal) && absVal < minAbs 更新 minAbs/minIdx
           （NaN 跳过；首元素 NaN → 基准 FLT_MAX）
  [阶段2 全局] workspace[coreId]={minAbs,minIdx} → 尾核读全 workspace，
    取 min(minAbs)，并列取 min(minIdx) → result = minIdx+1（1-based）
```
workspace 布局：2 word/核 × numBlocks，64 word 对齐；无其他依赖。

#### 性能优化要点（终态，S5 回填实测）
- 大 n 多核满载 + 双缓冲逼近带宽；Abs 向量化单趟；workspace 批量读回。
- incx=1 主路径（性能用例主体）优先整段连续搬运；incx>1 慢路不拖累主路径（分支隔离）。

## 支持硬件
| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22，910B 系；自验 910B3） | ✅ 本任务目标 |
| Atlas 950（arch35） | ❌（已有 isamin arch35 实现） |

## 算子约束限制
- incx<1 不进入计算（result=0 快速返回，任务书 §2.5）。
- 步长语义即非连续访问（间隔元素跳过）；无广播；result 独立输出不复用输入内存。
- 确定性：并列最小取最先出现索引，单次执行内部确定；内存 case 不涉及（§3.4）。

# 可维可测分析（required）

## 精度标准/性能标准
| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | result（INT32）与 golden 索引**完全一致**（不允许任何错位；matched_ratio 1.0、max_abs_error 0）；golden=CPU 参考实现直接计算 1-based；NaN/±0/并列语义按 §2.1 | 任务书 §3.2 + mixed_tolerance_standard.md（28a1ce5 版） |
| 性能 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv | 任务书 §3.3 |

## 测试概览
- 用例分层（§3.5）：边界域（n=0/1、incx<1 → result=0、并列最小值含重复元素与 ±0、NaN 样本）、负向域（handle/result/x 空指针（含 n=0 路径）、n<0、跨度溢出）、精度全量 1000 条（均匀 [-5,5] 50% + 正态 GAUSS_NORM_5_2 50%，全值域含 Inf/NaN）、性能子集 200 条（每 dtype 按 §2.4 dtype 面个数分摊——本算子单 dtype FLOAT32）。
- 期望码：空指针/负 n/跨度溢出 → INVALID_VALUE；handle → HANDLE_IS_NULLPTR；quick return → SUCCESS+result=0。
- 负向用例缓冲区：不按 (n−1)×|incx| 预分配（dummy/NULLPTR 占位，§3.5 规则）。
- 性能口径：msprof op，warmup5+≥10 均值；复现步骤见 test_cases/README.md。

## 兼容性分析
- 不改公共头（声明已在 L240）；`blas/iamin/arch22/`、`test/isamin/arch22/` 均为新建目录；`test/isamin/CMakeLists.txt` 在 arch22 分支补 GTest 注册（任务书明确要求，波及面仅测试构建，不影响 arch35 既有注册）。

---
### 与官方模板的差异说明
按套件《设计文档模板》无编号六段结构书写；性能要点为规划值，终态数据待 S5 实测回填。
