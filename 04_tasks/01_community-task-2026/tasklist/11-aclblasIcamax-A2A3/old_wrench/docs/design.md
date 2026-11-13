# aclblasIcamax 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasIcamax算子开发（A2A3）；算子 aclblasIcamax（对标 cuBLAS cublasIcamax；Netlib `icamax.f`）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/iamax/arch22/`（实现）+ `test/iamax/icamax/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-ops-competitions MR）。
- 环境：CANN 9.1.0；性能设备 Atlas 800T A2 (910B3)；接口声明已在 `include/cann_ops_blas.h`（L239，与 950PR 共用），不改公共头。

## 背景介绍
### 算子功能定位
单精度复数向量最大"模"元素索引归约：`result = argmax_i (|Re(x[k])| + |Im(x[k])|)`，i=1..n，k=1+(i−1)·incx，输出 **1-based** int 索引；并列最大取**最小索引**；n=0 或 incx≤0 时 result=0 quick return（不支持负步长反向遍历）。补齐 Blas 仓 Atlas A2/A3 BLAS-1 复数索引归约缺口。

### 业界对标
- cublasIcamax："（smallest）index" 语义 + 严格大于才更新（Netlib `icamax` 比较语义）。
- Netlib `icamax.f`：**复数"模"为 1-范数口径 SCABS1 = |Re|+|Im|，禁止欧氏模**（实现与 golden 同口径，bit-exact 判定基础）。
- 特殊值语义（严格大于比较）：首个 Inf 元素胜出；NaN 元素不会胜出（首元素为 NaN 时结果停留索引 1）；以 cblas golden 精确一致为判定。

### 工程形态
- Host C API + Kernel 直调；代码落位 `blas/iamax/arch22/`（目录已有实数版 `aclblasIsamax`，复用其框架）。
- 仓内近亲：`blas/iamax/arch22/isamax_*`（实数索引归约骨架：host 校验/tiling/quick-return memset + 多核归约内核）、`blas/iamax/arch35/icamax_*`（复数版：复数 1-范数比较 + 两字 workspace 归约，直接平移到 arch22）；测试工程放 `test/iamax/icamax/arch22/`，并在 `test/iamax/icamax/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} icamax_test)` 注册（任务书 §2.2 明确要求）。

### 接口规格
```cpp
aclblasStatus_t aclblasIcamax(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```
（声明已在公共头 L239，与 950PR 共用，禁止私有平行 API）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | 非空；nullptr → HANDLE_IS_NULLPTR | - |
| n | 复数元素个数 | int | attr | n≥0；n<0 → INVALID_VALUE；n=0 → quick return（result=0，SUCCESS） | - |
| x | 复数向量（Device，只读） | const aclblasComplex* | COMPLEX64 | 交错存储，n 个复数 = 2n 个 float；n>0 且 incx≥1 时非空；值域 FLOAT32 全集 | [1+(n−1)·incx]（incx≥1） |
| incx | x 步长 | int | attr | incx≥1 正常计算；incx≤0（含 0 与负步长）合法 quick return（result=0）；(n−1)×incx 超 int32 → INVALID_VALUE（host 侧 int64，issue #745） | - |
| result | 输出 1-based 索引（Device） | int* | INT32 | 单值标量；正常 1..n，quick return 为 0；nullptr（任意 n，含快速返回路径）→ INVALID_VALUE | [1] |

### 数学公式
result = argmax_{i=1..n} SCABS1(x[k])，k = 1+(i−1)·incx；SCABS1(x) = |Re(x)| + |Im(x)|（1-范数，非欧氏模）；并列取最小 i；输出 1-based；n=0 或 incx≤0 → result=0。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调在 Atlas A2/A3（arch22/ascend910b）实现 aclblasIcamax 全语义（1-范数模、最小索引并列语义、Inf/NaN 特殊值、quick return、int64 跨度守卫），输出与 golden **bit-exact 精确相等**，性能单 case 达 0.8×A100 cublasIcamax。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | argmax(|Re|+|Im|) + 最小索引；arch35 icamax（复数归约）全量平移 arch22，arch22 isamax（实数索引归约框架）承担 host 骨架——两者拼接即可，无新机制 |
| 2 | 接口原型 | 见上节；声明已在公共头，不改头文件 |
| 3 | 参数规格 | 见参数规格表；校验顺序对齐 arch35 ValidateIcamaxParams（handle→n<0→result 判空→quick return→x 判空→跨度守卫） |
| 4 | 支持数据类型 | 输入 COMPLEX64 / 输出 INT32 索引 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / ascend910b*；自验 910B3（§3.1） |
| 6 | 精度要求 | **bit-exact 整数索引精确相等**（NPU 索引 ≠ golden 索引即失败）；浮点阈值不适用输出比对；1-范数口径；NaN/Inf 语义对齐 Netlib 严格大于更新（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv（§3.3） |
| 8 | 代码架构选型+可行性 | 多核局部 argmax + 全局归约两段式（arch35 icamax 同款）；已排除死路：原子写 result（会破坏最小索引语义）、欧氏模比较（与 golden 不一致即全错） |
| 9 | 评测来源 | 精度 golden=cblas_icamax（0-based +1 转 1-based，对齐 test/isamax/isamax_golden.h 范式）；性能标杆=gpu_baseline.csv |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景。两段归约：段内 argmax（严格大于更新，保留首见最小索引）→ 各核局部 (bestVal, bestIdx) 写 workspace → 核间归约取全局最大（并列取小索引）→ result 写 1-based 索引。

### 支持数据类型
输入 COMPLEX64（实/虚 float32 交错）；输出 INT32 标量索引。

### 支持形状
n≥0 任意；n=0 quick return（result=0）；incx≥1（1、2、3 等）；incx≤0 quick return；n 至 ≥2²⁰ 大规模；全零/常量（并列取索引 1）输入均支持。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| icamax_multi_core（唯一变体） | n>0 且 incx≥1 | 各核按交错分片取 n 的子区间：MTE2 搬本核复数段 → V 计算 SCABS1（|Re|+|Im| 分量绝对值相加）→ 标量域严格大于比较（bestIdx 初值取全局首索引，保证并列取小索引）→ 局部结果写 workspace（每核 2 word：val+idx）→ 最后完成归约的核（或尾核）读 workspace 做全局比较写 result |

数据流：GM(x) → MTE2 → UB（复数段双缓冲）→ V 绝对值求和 → 标量比较（index 语义在标量域，不进向量谓词）→ workspace（GM 小缓冲）→ 全局归约 → MTE3 写 result（int 标量）。

#### host 侧设计
- 分核策略：numBlocks=min(GetAivCoreCount(), 上限)；n 均分 + 尾核补余（perCoreN/lastCoreN，arch35 CalcIcamaxTiling 同式）。
- 数据分块与内存优化：单核内 UB 双缓冲搬 2n float；无矩阵，纯向量带宽瓶颈，S5 精调每核块长。
- tilingkey 规划：无独立 tilingkey；TilingData 字段：totalN/useCoreNum/incx/perCoreN/lastCoreN（arch35 同构）。
- 数据检测（校验顺序，逐条对齐 §2.4/§2.5 与 arch35 ValidateIcamaxParams L15-53）：handle 判空（HANDLE_IS_NULLPTR）→ n<0（INVALID_VALUE）→ result 判空（INVALID_VALUE，任意 n）→ n=0 或 incx<1 quick return（aclrtMemsetAsync 写 result=0，不 host 解引用 device 地址，issue #738 口径）→ x 判空 → 跨度守卫 (n−1)×incx∈[0,INT32_MAX]（int64，issue #745）。异步下发。

#### kernel 侧设计
```
icamax_multi_core:
  [阶段1 局部] for 本核子区间按块:
    [MTE2] 复数段搬 UB
    [V] SCABS1: Abs(re)+Abs(im)（向量）→ 局部 bestVal/bestIdx（严格 > 更新，
        NaN 不胜出；首 Inf 直接锁定）
  [阶段2 全局] workspace[coreId]={val,idx} → 尾核读全 workspace，
    取 max(val)，并列取 min(idx) → result = bestIdx+1（1-based）
```
workspace 布局：2 word/核 × numBlocks，64 word 对齐（arch35 WORKSPACE_WORDS_PER_CORE/ALIGN 同值）；无其他依赖。

#### 性能优化要点（终态，S5 回填实测）
- 大 n（≥2²⁰）时多核满载 + 每核双缓冲即可逼近带宽；S5 关注小块 n 的核间开销（可单核短路的规模阈值实验，沿 isamax arch22 先例）。
- 候选：SCABS1 向量化（Abs+Add 单趟）、workspace 读回用一次 MTE2 批量。

## 支持硬件
| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22，910B 系；自验 910B3） | ✅ 本任务目标 |
| Atlas 950（arch35） | ❌（已有 icamax arch35 实现） |

## 算子约束限制
- 不支持负步长反向遍历（incx≤0 → result=0，任务书 §2.1-4）。
- 非连续访问仅由 incx≥1 表达；无广播；result 独立输出。
- 归约划分顺序不要求固定，但结果必须"首个最大者"语义，与 golden 精确一致。

# 可维可测分析（required）

## 精度标准/性能标准
| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 整数索引 **bit-exact**（actual==golden，matched_ratio=1 才通过）；golden=cblas_icamax 0-based +1；1-范数口径；NaN 不胜出/首 Inf 胜出语义对齐 | 任务书 §3.2 + mixed_tolerance_standard.md（28a1ce5 版） |
| 性能 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv | 任务书 §3.3 |

## 测试概览
- 用例分类：精度（小 shape 基础 / shape 扫描至 ≥2²⁰ / 填充模式：全零、常量、正负交替、极端值 / Inf·NaN 特殊值 / incx=1,2,3）、quick return（n=0、incx=0、负步长 −1/−2/−3 → SUCCESS 且 result=0）、负向（handle/result/x 空指针、n<0、跨度溢出 → 对应错误码）、性能 200 条。
- CSV：n/incx/填充模式列对齐 `test/isamax/isamax_param.h` 参数定义，扩展 result 列 NULLPTR 标记负向（§3.5-1）。
- 期望码：空指针/负 n/跨度溢出 → INVALID_VALUE；handle → HANDLE_IS_NULLPTR；quick return → SUCCESS+result=0。
- 正态分布用例（GAUSS_NORM_5_2）扩展补齐并纳入精度自验（§3.5 补充要求）。

## 兼容性分析
- 不改公共头（声明已在 L239）；`blas/iamax/arch22/` 并入既有目录新增 icamax_* 文件；测试 CMakeLists 在 arch22 分支补 GTest 注册（任务书明确要求，波及面仅测试构建）。

---
### 与官方模板的差异说明
按套件《设计文档模板》无编号六段结构书写；性能要点为规划值，终态数据待 S5 实测回填。
