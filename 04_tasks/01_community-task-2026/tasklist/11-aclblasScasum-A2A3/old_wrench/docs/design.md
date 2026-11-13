# aclblasScasum 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasScasum算子开发（A2A3）；算子 aclblasScasum（对标 cuBLAS cublasScasum；Netlib `scasum.f`）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/asum/arch22/`（实现）+ `test/asum/scasum/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-ops-competitions MR）。
- 环境：CANN 9.1.0；性能设备 Atlas 800T A2 (910B3)；接口声明已在 `include/cann_ops_blas.h`（L236，与 950PR 共用），不改公共头。

## 背景介绍
### 算子功能定位
单精度复数向量绝对值分量之和归约：`result = Σ (|real(x[k])| + |imag(x[k])|)`，k = 1+(i−1)·incx，输入 COMPLEX64 向量、输出 FLOAT32 实数标量；n≤0 或 incx≤0 时 result=0.0f quick return（不触发 kernel）。补齐 Blas 仓 Atlas A2/A3 BLAS-1 复数归约缺口。

### 业界对标
- cublasScasum（CUBLAS 文档 cublas-t-asum）：参数序列、返回码与快速返回口径唯一依据。
- Netlib `scasum.f`：golden 语义（`cblas_scasum` 生成真值）；物理长度 1+(n−1)·|incx| 个复数元素。
- 语义差异点：归约累加顺序不要求与参考实现逐位一致（非 bit-exact，按阈值判定）；逐项 |Re|+|Im| 非负累加，大 n × 接近 FLT_MAX 元素时结果可溢出到 Inf（golden 同为浮点累加亦溢出，按 Inf 一致性判定并单独标注）。

### 工程形态
- Host C API + Kernel 直调；代码落位 `blas/asum/arch22/`（同目录已有实数版 sasum_host.cpp / sasum_kernel.cpp，扩展复数路径；tiling 结构可参考 `blas/asum/arch35/sasum_tiling_data.h`）。
- 仓内近亲：`blas/asum/arch22/sasum_*`（实数绝对值和归约骨架：host 校验/tiling/quick-return + 多核归约内核）、`blas/asum/arch35/scasum_*`（复数版，直接平移 arch22）、`blas/iamax/arch22/isamax_*`（arch22 侧多核归约工程先例）。
- 测试工程 `test/asum/scasum/arch22/` 新建，CSV 列格式对齐 `test/asum/sasum/sasum_param.h`（n / incx / x 填充模式 / result 空指针标记），并在 `test/asum/scasum/CMakeLists.txt` arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)` 注册（任务书 §2.2 明确要求）。

### 接口规格
```cpp
aclblasStatus_t aclblasScasum(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    float* result);
```
（声明已在公共头 L236，与 950PR 共用，禁止私有平行 API）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | 非空；nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| n | 复数元素个数 | int | attr | 任意整数；n≤0（含负维度）合法 quick return（result=0.0f，SUCCESS，不触发 kernel） | - |
| x | 复数向量（Device，只读） | const aclblasComplex* | COMPLEX64 | 交错存储，n 个复数 = 2n 个 float；物理长度 1+(n−1)·\|incx\|；n>0 且 incx>0 时非空；值域 FLOAT32 全集 | [1+(n−1)·\|incx\|] |
| incx | x 步长 | int | attr | incx>0 正常计算；incx≤0（含 0 与负步长）合法 quick return（result=0.0f，不遍历）；(n−1)·\|incx\| 超 int32 → INVALID_VALUE（host 侧 int64，issue #805） | - |
| result | 输出 \|Re\|+\|Im\| 之和（Device） | float* | FLOAT32 | 独立输出标量，非负（溢出可为 Inf）；nullptr（任意 n，含 quick return 路径）→ INVALID_VALUE | [1] |

### 数学公式
result = Σ_{i=1..n} ( |Re(x[k])| + |Im(x[k])| )，k = 1+(i−1)·incx；输出 FLOAT32 标量；n≤0 或 incx≤0 → result=0.0f。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调在 Atlas A2/A3（arch22/ascend910b）实现 aclblasScasum 全语义（|Re|+|Im| 分量归约、quick return、int64 跨度守卫），精度满足 mixed_tolerance FLOAT32 标量比对口径，性能单 case 达 0.8×A100 cublasScasum。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | Σ(\|Re\|+\|Im\|)；arch35 scasum（复数版归约）平移 arch22 + arch22 sasum（实数归约框架）——归约骨架全复用，仅数据通道换复数（2n float 交错搬运） |
| 2 | 接口原型 | 见上节；声明已在公共头 L236，不改头文件 |
| 3 | 参数规格 | 见参数规格表；校验顺序对齐 §2.5：result 判空（任意 n）→ handle 判空 → n≤0 或 incx≤0 quick return → x 判空（n>0 且 incx>0）→ 跨度守卫（int64，issue #805） |
| 4 | 支持数据类型 | 输入 COMPLEX64 / 输出 FLOAT32 标量 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / ascend910b*；自验 910B3（§3.1） |
| 6 | 精度要求 | 标量比对：\|actual−golden\| ≤ atol+rtol×\|golden\|（rtol=atol=2⁻¹³），max_abs≤1e-2 或 32ULP；matched_ratio 退化为单值判定；Inf/NaN 等效规则（同为 NaN/同号 Inf/NaN↔±Inf 等效通过）；quick return 用例 golden 固定 0.0f（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv（§3.3） |
| 8 | 代码架构选型+可行性 | 多核分片求和 + 全局归约两段式（arch35 scasum 同款）；已排除死路：复数整体乘法路径（无乘法，纯 Abs+Add）、按复数元素单值搬运（交错 2n float 连续段更优） |
| 9 | 评测来源 | 精度 golden=cblas_scasum（Netlib 复数实现）；性能标杆=gpu_baseline.csv（A100 实测） |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景。两段归约：段内 Σ(|Re|+|Im|) 向量化（复数段按 2n float 视角 Abs 后求和）→ 各核局部和写 workspace → 核间求和归约 → result 写 float 标量。逐项非负累加保证无符号翻转误差；溢出为 IEEE 语义自然 Inf。

### 支持数据类型
输入 COMPLEX64（实/虚 float32 交错）；输出 FLOAT32 标量。

### 支持形状
n 任意整数（n≤0 quick return）；incx>0（1、2、3 等）；incx≤0 quick return；n 至 ≥2²⁰ 大规模；全零、正负交替、极端值（大 n×接近 FLT_MAX 的溢出场景）、Inf/NaN 输入均支持。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| scasum_multi_core（唯一变体） | n>0 且 incx>0 | 各核分片：MTE2 搬本核复数段（2n float 交错，incx 步进寻址）→ V 逐 float 求绝对值 + 段内求和归约（ReduceSum）→ 局部和写 workspace（每核 1 word）→ 尾核读全 workspace 求和 → 写 result |

数据流：GM(x) → MTE2 → UB（复数段双缓冲）→ V（Abs → ReduceSum）→ workspace（GM 小缓冲）→ 全局求和归约 → MTE3 写 result（float 标量）。incx>1 时按复数元素步进搬运，MTE2 退化为小段/标量 Gather，S5 按 incx 分布精调。

#### host 侧设计
- 分核策略：numBlocks=min(GetAivCoreCount(), 上限)；n 均分 + 尾核补余（arch35 sasum tiling 同式）。
- 数据分块与内存优化：单核 UB 双缓冲；纯向量带宽瓶颈；主路径（incx=1）整段连续搬运。
- tilingkey 规划：无独立 tilingkey；TilingData 字段：totalN/useCoreNum/incx/perCoreN/lastCoreN（参考 arch35/sasum_tiling_data.h）。
- 数据检测（校验顺序，逐条对齐 §2.5）：result 判空（INVALID_VALUE，任意 n，含 quick return 路径）→ handle 判空（HANDLE_IS_NULLPTR）→ n≤0 或 incx≤0 quick return（aclrtMemsetAsync/单元素 memset 写 result=0.0f，SUCCESS）→ x 判空（n>0 且 incx>0，INVALID_VALUE）→ 跨度守卫 (n−1)×|incx|∈[0,INT32_MAX]（host 侧 int64，issue #805）。元素值不校验。异步下发，读结果前同步 stream。

#### kernel 侧设计
```
scasum_multi_core:
  [阶段1 局部] localSum = 0
    for 本核子区间按块:
      [MTE2] 复数段搬 UB（双缓冲）
      [V] Abs（逐 float，含实/虚部）→ 段内 ReduceSum 累入 localSum
  [阶段2 全局] workspace[coreId]=localSum → 尾核读全 workspace 求和
    → MTE3 写 result（float；溢出自然为 Inf）
```
workspace 布局：1 word/核 × numBlocks，64 word 对齐；无其他依赖；无原子操作。

#### 性能优化要点（终态，S5 回填实测）
- Abs+ReduceSum 融合单趟；双缓冲消除 MTE2/V 空泡；workspace 批量读回。
- 大 n 多核满载逼近带宽；incx=1 主路径与 incx>1 慢路分支隔离。

## 支持硬件
| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22，910B 系；自验 910B3） | ✅ 本任务目标 |
| Atlas 950（arch35） | ❌（已有 scasum arch35 实现） |

## 算子约束限制
- incx≤0 不遍历向量（result=0.0f quick return，任务书 §2.1-3）。
- 非连续访问仅由 incx 表达；无广播；result 独立输出；无 leading dimension padding 场景。
- 累加顺序不保证与 golden 逐位一致（按阈值判定）；大 n × 极端值可溢出 Inf（与 golden 一致性判定）。

# 可维可测分析（required）

## 精度标准/性能标准
| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 标量比对：\|actual−golden\| ≤ atol+rtol×\|golden\|（rtol=atol=2⁻¹³，ratio 单值判定），max_abs≤1e-2 或 32ULP；Inf/NaN 等效规则；quick return 用例 golden=0.0f | 任务书 §3.2 + mixed_tolerance_standard.md（28a1ce5 版） |
| 性能 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv | 任务书 §3.3 |

## 测试概览
- 用例分类：精度（小 shape 基础 / shape 扫描至 ≥2²⁰（n=2/3/7/2 的幂±1）/ 填充模式：全零、正负交替、极端值 / 大 n×极端值溢出用例（Inf 一致性，报告单独标注）/ Inf·NaN 特殊值 / incx=1,2,3）、quick return（n≤0（负 n 为边界用例）、incx=0、负步长 → SUCCESS 且 result=0.0f）、负向（result/x 空指针、跨度溢出 → 对应错误码）、性能 200 条。
- CSV：对齐 `test/asum/sasum/sasum_param.h` 参数定义（n / incx / 填充模式 / result 空指针标记）。
- 期望码：result 空指针（任意 n）/ n>0∧incx>0 时 x 空指针 / 跨度溢出 → INVALID_VALUE；handle → HANDLE_IS_NULLPTR；quick return → SUCCESS+result=0.0f。
- 正态分布用例（GAUSS_NORM_5_2）扩展补齐并纳入精度自验（§3.5 补充要求）。

## 兼容性分析
- 不改公共头（声明已在 L236）；`blas/asum/arch22/` 并入既有目录扩展复数路径（新增 scasum 文件，不动 sasum 实数实现）；测试 CMakeLists 在 arch22 分支补 GTest 注册（任务书明确要求，波及面仅测试构建）。

---
### 与官方模板的差异说明
按套件《设计文档模板》无编号六段结构书写；性能要点为规划值，终态数据待 S5 实测回填。
