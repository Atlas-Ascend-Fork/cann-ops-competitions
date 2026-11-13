# aclblasIcamin 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasIcamin算子开发（A2A3）；算子 aclblasIcamin（对标 cuBLAS cublasIcamin；Netlib 标准参考库**无 icamin 例程**，语义细节以 cuBLAS 官方文档为最高优先级来源）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/iamin/arch22/`（**新建**目录）+ `test/iamin/icamin/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-ops-competitions MR）。
- 环境：CANN 9.1.0；性能设备 Atlas 800T A2 (910B3)；接口声明已在 `include/cann_ops_blas.h`（L241，与 950PR 共用），不改公共头。

## 背景介绍
### 算子功能定位
单精度复数向量最小模元素索引归约：`result = argmin_i (|Re(x[k])| + |Im(x[k])|)`，i=1..n，k=1+(i−1)·incx，输出 **1-based** int 索引；并列取**最小索引**；n=0 或 incx<1 时 result=0 quick return。补齐 Blas 仓 Atlas A2/A3 BLAS-1 复数最小索引归约缺口。

### 业界对标
- cublasIcamin：参数序列、返回码与快速返回口径唯一依据（CUBLAS_Library.pdf）。
- 复数"模"按 BLAS icamin 惯例为 **1-范数 |Re|+|Im|**（SCABS1 口径，非欧氏模），实现与 golden 同口径。
- NaN 语义（任务书 §3.2 明确）：golden 以严格小于（`absVal < bestVal`）更新索引，NaN 与任何值比较均为 false，**NaN 不会被选为最小元素**；若首元素为 NaN，基准即为 NaN，后续任何元素都不小于它，结果为 1——实现须对齐该语义。
- 语义差异点：与 Icamax 同为"严格比较 + 首见索引"语义，仅比较方向相反；归约顺序不影响结果（IEEE 浮点比较确定）。

### 工程形态
- Host C API + Kernel 直调；代码落位 `blas/iamin/arch22/`（新建；参考 `blas/iamin/arch35/` 中 aclblasIsamin / aclblasIcamin 归约实现）。
- 仓内近亲：`blas/iamin/arch35/isamin_*`（实数最小索引归约：ValidateIsaminParams 校验入口 L29-53 + perCoreN/lastCoreN tiling）、`blas/iamin/arch35/icamin_*`（复数版，直接平移）、`blas/iamax/arch22/isamax_*`（arch22 侧索引归约工程骨架）。
- 测试工程 `test/iamin/icamin/arch22/` 新建，CSV 列格式对齐 `test/isamin/isamin_param.h`（x 列写 NULLPTR 表示空指针负向），并在 `test/iamin/icamin/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)` 注册（任务书 §2.2 明确要求）。

### 接口规格
```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```
（声明已在公共头 L241，与 950PR 共用，禁止私有平行 API）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | 非空；nullptr → HANDLE_IS_NULLPTR | - |
| n | 复数元素个数 | int | attr | n≥0；n<0 → INVALID_VALUE；n=0 → quick return（result=0，SUCCESS） | - |
| x | 复数向量（Device，只读） | const aclblasComplex* | COMPLEX64 | 交错存储，n 个复数 = 2n 个 float；n>0 且 incx≥1 时非空；值域 FLOAT32 全集 | [1+(n−1)·\|incx\|] |
| incx | x 步长 | int | attr | incx≥1 正常计算；incx<1（含 0 与负步长）合法 quick return（result=0，不触发 kernel）；(n−1)×\|incx\| 超 int32 → INVALID_VALUE（host 侧 int64，issue #745） | - |
| result | 输出 1-based 索引（Device） | int* | INT32 | 单值标量；0（quick return）或 [1,n]；nullptr（任意 n，含快速返回路径）→ INVALID_VALUE | [1] |

### 数学公式
result = argmin_{i=1..n} SCABS1(x[k])，k = 1+(i−1)·incx；SCABS1(x) = |Re(x)| + |Im(x)|；并列取最小 i；输出 1-based；n=0 或 incx<1 → result=0。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调在 Atlas A2/A3（arch22/ascend910b）实现 aclblasIcamin 全语义（1-范数模、最小索引并列语义、NaN 不胜出、quick return、int64 跨度守卫），输出与 golden **bit-exact 精确相等**，性能单 case 达 0.8×A100 cublasIcamin。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | argmin(|Re|+|Im|) + 最小索引；arch35 icamin（复数版归约）平移 arch22 + arch35 isamin（host 校验骨架 L29-53）——双近亲拼接，无新机制；与 Icamax 仅差比较方向，内核结构完全同构 |
| 2 | 接口原型 | 见上节；声明已在公共头，不改头文件 |
| 3 | 参数规格 | 见参数规格表；校验顺序对齐 arch35 ValidateIsaminParams（handle→n<0→result 判空→quick return→x 判空→跨度守卫） |
| 4 | 支持数据类型 | 输入 COMPLEX64 / 输出 INT32 索引 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / ascend910b*；自验 910B3（§3.1） |
| 6 | 精度要求 | **bit-exact 整数索引精确相等**（actual==golden，1-based 逐位相等）；NaN 语义按严格小于更新（NaN 不被选中，首元素 NaN → 结果 1）（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv（§3.3） |
| 8 | 代码架构选型+可行性 | 多核局部 argmin + 全局归约两段式（arch35 icamin 同款）；已排除死路：原子写 result（破坏最小索引语义）、欧氏模（与 golden 不一致即全错）、host 遍历（数据在 Device） |
| 9 | 评测来源 | 精度 golden=测试工程内 CPU 参考循环（cblas 风格，|Re|+|Im| 严格小于取最小索引，直接产出 1-based）；性能标杆=gpu_baseline.csv |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景。两段归约：段内 argmin（严格小于更新，保留首见最小索引，NaN 不胜出）→ 各核局部 (bestVal, bestIdx) 写 workspace → 核间归约取全局最小（并列取小索引）→ result 写 1-based 索引。

### 支持数据类型
输入 COMPLEX64（实/虚 float32 交错）；输出 INT32 标量索引。

### 支持形状
n≥0 任意；n=0 quick return；incx≥1（1、2、3 等）；incx<1 quick return；n 至 ≥2²⁰ 大规模；全零（并列取索引 1）、正负交替、极端值、Inf/NaN 输入均支持。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| icamin_multi_core（唯一变体） | n>0 且 incx≥1 | 各核按交错分片取 n 的子区间：MTE2 搬本核复数段 → V 计算 SCABS1（|Re|+|Im|）→ 标量域严格小于比较（bestIdx 保留首见最小索引；NaN 比较为 false 不更新，首元素 NaN 时基准为 NaN、结果停留本段首索引）→ 局部 (val,idx) 写 workspace（每核 2 word）→ 尾核全局归约取 min（并列取小 idx）写 result |

数据流：GM(x) → MTE2 → UB（复数段双缓冲）→ V 绝对值求和 → 标量比较 → workspace（GM 小缓冲）→ 全局归约 → MTE3 写 result（int 标量）。

#### host 侧设计
- 分核策略：numBlocks=min(GetAivCoreCount(), 上限)；n 均分 + 尾核补余（perCoreN/lastCoreN，arch35 CalcIsaminTiling 同式）。
- 数据分块与内存优化：单核 UB 双缓冲搬 2n float；纯向量带宽瓶颈，S5 精调块长。
- tilingkey 规划：无独立 tilingkey；TilingData 字段：totalN/useCoreNum/incx/perCoreN/lastCoreN（arch35 isamin 同构）。
- 数据检测（校验顺序，逐条对齐 §2.4/§2.5 与 arch35 ValidateIsaminParams L29-53）：handle 判空（HANDLE_IS_NULLPTR）→ n<0（INVALID_VALUE）→ result 判空（INVALID_VALUE，任意 n）→ n=0 或 incx<1 quick return（aclrtMemsetAsync 写 result=0，不 host 解引用 device 地址，issue #738 口径）→ x 判空 → 跨度守卫 (n−1)×|incx|∈[0,INT32_MAX]（int64，issue #745）。异步下发。

#### kernel 侧设计
```
icamin_multi_core:
  [阶段1 局部] for 本核子区间按块:
    [MTE2] 复数段搬 UB
    [V] SCABS1: Abs(re)+Abs(im)（向量）
    [标量] 严格小于更新 bestVal/bestIdx（NaN 比较为 false 不更新）
  [阶段2 全局] workspace[coreId]={val,idx} → 尾核读全 workspace，
    取 min(val)，并列取 min(idx) → result = bestIdx+1（1-based）
```
workspace 布局：2 word/核 × numBlocks，64 word 对齐；无其他依赖。

#### 性能优化要点（终态，S5 回填实测）
- 大 n 多核满载 + 双缓冲逼近带宽；小块 n 单核短路阈值实验（沿 isamax arch22 先例）。
- 候选：SCABS1 向量化单趟、workspace 批量读回。

## 支持硬件
| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22，910B 系；自验 910B3） | ✅ 本任务目标 |
| Atlas 950（arch35） | ❌（已有 icamin/isamin arch35 实现） |

## 算子约束限制
- 不支持负步长反向遍历（incx<1 → result=0，任务书 §2.4）。
- 非连续访问仅由 incx≥1 表达；无广播；result 独立输出；无 leading dimension padding 场景。
- 归约顺序不影响结果（IEEE 比较确定），支持 bit-exact 判定。

# 可维可测分析（required）

## 精度标准/性能标准
| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 整数索引 **bit-exact**（actual==golden 逐位相等，matched_ratio=1）；golden=CPU 参考循环直接产出 1-based；NaN 语义对齐 §3.2 | 任务书 §3.2 + mixed_tolerance_standard.md（28a1ce5 版） |
| 性能 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv | 任务书 §3.3 |

## 测试概览
- 用例分类：精度（小 shape 基础 / shape 扫描至 ≥2²⁰（n=2/3/5/7/2 的幂±1）/ 填充模式：全零、正负交替、极端值 / Inf·NaN 特殊值 / incx=1,2,3）、quick return（n=0、incx=0、负步长 → SUCCESS 且 result=0）、负向（handle/result/x 空指针、n<0、跨度溢出 → 对应错误码）、性能 200 条。
- CSV：n/incx/填充模式对齐 `test/isamin/isamin_param.h`，x 列 NULLPTR 标记空指针负向（§3.5-1）。
- 期望码：空指针/负 n/跨度溢出 → INVALID_VALUE；handle → HANDLE_IS_NULLPTR；quick return → SUCCESS+result=0。
- 正态分布用例（GAUSS_NORM_5_2）扩展补齐并纳入精度自验（§3.5 补充要求）。

## 兼容性分析
- 不改公共头（声明已在 L241）；`blas/iamin/arch22/` 为新建目录（族目录下首个 arch22 实现，文件结构与 arch35 对齐）；测试 CMakeLists 在 arch22 分支补 GTest 注册（任务书明确要求，波及面仅测试构建）。

---
### 与官方模板的差异说明
按套件《设计文档模板》无编号六段结构书写；性能要点为规划值，终态数据待 S5 实测回填。
