# aclblasChemv 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasChemv算子开发（A2A3）；算子 aclblasChemv（对标 cuBLAS cublasChemv；Netlib `chemv.f`）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/symv/arch22/`（实现）+ `test/symv/chemv/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-ops-competitions MR）。
- 环境：CANN 9.1.0；性能设备 Atlas 800T A2 (910B3)；公共头新增声明（任务书 §2.3）。

## 背景介绍
### 算子功能定位
单精度复数 Hermitian 全矩阵-向量乘：`y = α·A·x + β·y`，A 为 n×n Hermitian（A=Aᴴ）复数矩阵，列主序全存储于 lda×n；仅 uplo 指定三角被引用，未引用部分不读取、由共轭对称性隐含；对角元素虚部假定为 0、不参与计算。补齐 Blas 仓 Atlas A2/A3 BLAS-2 Hermitian 能力缺口。

### 业界对标
- cublasChemv：参数序列、返回码与快速返回口径唯一依据（uplo∈{UPPER, LOWER}，列主序，正负步长）。
- Netlib `chemv.f`：golden 语义（`cblas_chemv` 生成真值，对角仅取实部）。
- 语义差异点：本任务 α=(0,0) 且 β=(1,0) 可 quick return（Netlib 语义）；β=(0,0) 时 y 旧值不必有效；α=(0,0) 时 A/x 允许为空指针（校验按标量值判定，需先读标量——仓内 csymv arch35 ComplexScalarSource 双读取口径，issue #869）。

### 工程形态
- Host C API + Kernel 直调，与同族 `aclblasSsymv`（实数）、`blas/symv/arch22/chemv_*`（若 master 已合入则逐行参考）同工程模式。
- 代码落位：`blas/symv/arch22/`（并入既有目录）。
- 仓内近亲：`blas/symv/arch22/ssymv_*`（实数全矩阵版骨架）、`blas/symv/arch22/csymv_*`（复数对称版——与 Hermitian 仅差"对角取实"与"未引用三角不读"）、`blas/symv/arch35/csymv_host.cpp`（ComplexScalarSource 标量双读取锚点）。
- 附带工程修复（任务书 §2.2 明确要求）：`test/symv/chemv/CMakeLists.txt` 在 arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)`（同族 CMakeLists 当前仅 arch35 注册，A2/A3 构建退回非 GTest runner）。

### 接口规格
```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```
（新增声明，签名与 cuBLAS `cublasChemv` 参数序列一致，维数参数 int，与同族实数 `aclblasSsymv` 口径一致）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | 非空；nullptr → HANDLE_IS_NULLPTR（最优先） | - |
| uplo | A 存储模式 | aclblasFillMode_t | attr | ∈{UPPER, LOWER}；非法 → ACLBLAS_STATUS_INVALID_ENUM（本任务书 §2.4 明确枚举错误返回 INVALID_ENUM，与 Chbmv 的 INVALID_VALUE 口径不同，逐任务书执行） | - |
| n | Hermitian 矩阵阶数 | int | attr | n≥0；n<0 → INVALID_VALUE；n=0 合法 no-op | - |
| alpha | 复数标量 α 指针 | const aclblasComplex* | COMPLEX64 | Host/Device 双读取（按指针属性探测，issue #869）；n>0 时非空 | - |
| A | Hermitian 矩阵（Device，只读） | const aclblasComplex* | COMPLEX64 | 列主序 lda×n 全存储；仅 uplo 三角被引用；对角虚部视为 0；α≠(0,0) 且 n>0 时非空 | [lda,n] |
| lda | 前导维 | int | attr | lda≥max(1,n)；违反 → INVALID_VALUE | - |
| x | 输入向量（Device，只读） | const aclblasComplex* | COMPLEX64 | α≠(0,0) 且 n>0 时非空；跨度溢出守卫同 y | [1+(n−1)·\|incx\|] |
| incx | x 步长 | int | attr | ≠0 且 ≠INT32_MIN，正负均支持；(n−1)×\|incx\|>可表示范围 → INVALID_VALUE（host 侧 int64，issue #807/#780；对齐 ssymv arch22 大跨度守卫） | - |
| beta | 复数标量 β 指针 | const aclblasComplex* | COMPLEX64 | Host/Device 双读取（同 α）；β=(0,0) 时 y 旧值不必有效；n>0 时非空 | - |
| y | 输入兼输出向量（原地，Device） | aclblasComplex* | COMPLEX64 | n>0 且非（α=(0,0)∧β=(1,0)）时非空；跨度溢出同 x | [1+(n−1)·\|incy\|] |
| incy | y 步长 | int | attr | 同 incx | - |

### 数学公式
y[j] = α × Σ(i=1..n) A(i,j) × x[i] + β × y[j]，j = 1..n；
UPPER：i>j 的 A(i,j) 按 A(j,i) 共轭读取（不读未存储侧）；LOWER：i<j 同理；对角元素 A(j,j) 只取实部；β=(0,0) 不读 y 旧值；A(i,j) 位于列主序偏移 (i−1)+(j−1)×lda（1-based）。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调（SIMD 向量内核）在 Atlas A2/A3（arch22/ascend910b）实现 aclblasChemv 全语义（uplo 两三角、对角取实、共轭读取、α/β 特值 quick return、Host/Device 标量双探测、正负步长与跨度守卫），精度满足 mixed_tolerance FLOAT32 口径全绿，性能单 case 达 0.8×A100 cublasChemv。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | y=α·A·x+βy；Hermitian 全矩阵 = 复数对称版 csymv 的对角修正（对称版对角不取共轭/取全元素，Hermitian 版对角取实、非对角共轭读取）——arch22 csymv 机制全复用，改动集中在读侧共轭与对角分支 |
| 2 | 接口原型 | 见上节；新增声明进公共头（§2.3） |
| 3 | 参数规格 | 见参数规格表；校验顺序对齐 §2.5：handle → n<0 → n=0 no-op → uplo/lda/incx/incy/α/β 指针 → 读标量判（α=(0,0)∧β=(1,0)）直接成功 → 按标量值条件判空 A/x/y |
| 4 | 支持数据类型 | COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / ascend910b*；自验 910B3（§3.1） |
| 6 | 精度要求 | mixed_tolerance FLOAT32：rtol=atol=2⁻¹³，ratio≥0.99，max_abs≤1e-2 或 32ULP；实部/虚部分别比对；Inf/NaN 等效规则（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv（§3.3） |
| 8 | 代码架构选型+可行性 | host（校验+标量双探测+tiling+直调）+ SIMD 内核按输出行分块；已排除死路：SIMT（仅 arch35）、复用 csymv 原样（对角虚部参与计算会破坏 Hermitian golden 一致性，必须改对角分支） |
| 9 | 评测来源 | 精度 golden=cblas_chemv（Netlib 单精度复数实现，对角取实）；性能标杆=任务包 gpu_baseline.csv |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景；kernel 按输出行 j 分块：UPPER 时列 j 的贡献 = 对角 A(j,j)·x[j]（取实）+ Σ(i<j) A(j,i)ᴴ·x[i]（按存储上三角共轭读取）+ Σ(i>j) A(i,j)·x[i]（直接读，同一列内行连续）；LOWER 镜像。两种读法合并为"每列一次连续段搬运 + 段内按 i<j/i>j 分支共轭"。

### 支持数据类型
COMPLEX64（全操作数）。

### 支持形状
n≥0 任意；n=0 no-op；n=1（纯对角，仅取实部乘 α 加 βy）、非对齐 n、lda=n 紧凑与 lda>n padding、|incx|>1 与负步长（自 span 尾端反向）均支持。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| chemv_upper | uplo=UPPER | 输出行块按核交错；每列 j 搬运上三角列切片 A(j..min(n,j+?)…) 不适用——上三角列 j 的被引用行为 i≤j，即列头连续段 [1..j]（i≤j 直接读）+ 行 i>j 部分按行读 A(j,i) 转置段；实现按"对角及以下（本列）连续段 + 跨列转置段"两路 |
| chemv_lower | uplo=LOWER | 列 j 的被引用行为 i≥j，即列内连续段 [j..n]（i≥j 直接读）+ 行 i<j 部分按 A(j,i)ᴴ 读下三角对应行段 |

数据流：GM(A,x,y) → MTE2 → UB（列切片 + 转置行段 + x/y 段双缓冲，事件交替）→ V 复数乘累加（y_re += a_re·x_re − a_im·x_im；y_im += a_re·x_im + a_im·x_re；共轭读侧 a_im 取负；对角项 y[j] += a_re·x[j] 实数乘）→ MTE3 → GM(y 原地)。UB 规划对齐 ssymv arch22 AsdopsBuffer 分区，复数路径缓冲倍宽。

#### host 侧设计
- 分核策略：输出行 n 按行块 ceil 切分按核交错；numBlocks=min(GetAivCoreCount(), 上限)。
- 数据分块与内存优化：行块/列切片由 tiling 携带（初版沿 ssymv 常量，S5 实测精调）；软件流水 copy_in(k+1)。
- tilingkey 规划：无独立 tilingkey，uplo/quickreturn 由 TilingData 字段分发；字段：uplo/n/lda/incx/incy/alpha/beta（标量 Host/Device 已由 host 归一为寄存器值下发，消除 kernel 侧二次探测）。
- 数据检测（校验顺序，逐条对齐 §2.5）：handle 判空 → n<0 → n=0 no-op（SUCCESS）→ uplo 枚举（INVALID_ENUM）→ lda≥max(1,n) → incx/incy（0/INT32_MIN）→ 跨度溢出（int64）→ α/β 指针非空（n>0）→ 读 α/β 标量（Host 直接读；Device 指针按 ComplexScalarSource 异步读回）→ （α=(0,0)∧β=(1,0)）→ SUCCESS；→ 条件判空 A/x/y：α≠(0,0) 时 A/x 非空、y 非空（除非 α=(0,0)∧β=(1,0)）；β=(0,0) 不豁免 y 非空。异步下发。

#### kernel 侧设计
```
chemv_upper / chemv_lower:
  for idxLoop in [0, 行块数) 按核交错:
    [MTE2] y_in×β 累入 Ybase（β=(0,0) 跳过）
    每列 j（块内）:
      [MTE2] 列内被引用连续段 + （UPPER 时）转置行段 / x 段 搬 UB
      [V] 直接读段：复数乘累加；共轭读段：a_im 取负后乘累加；
          对角元素：仅 a_re×x_re 累入实部、虚部加 a_re×x_im 之外不加 a_im 项（a_im=0）
    [MTE3] Ybase 写回 y（负步长反向寻址）
  事件 ID0/ID1 双缓冲流水
```
无 workspace；无原子操作。α=(0,0) 或 β=(1,0) 的 quick return 在 host 短路，不启动 kernel。

#### 性能优化要点（终态，S5 回填实测）
- 事件化双缓冲；UPPER 转置段与直接读段的搬运合并排序以提升连续性；βy 先行复用 y 段寄存。
- 候选：行块大小联调、numBlocks 上限、大 n 下转置段分块策略（对齐 ssymv arch22 实数版经验）。

## 支持硬件
| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22，910B 系；自验 910B3） | ✅ 本任务目标 |
| Atlas 950（arch35） | ❌（另行任务） |

## 算子约束限制
- 不支持超出 lda/incx/incy 语义的非连续访问；无广播；y 原地覆写；确定性不要求。
- 未引用三角不读取（实现不产生越界读）；对角虚部假定为 0（不校验）。
- α=(0,0) 时 A/x 可为 nullptr（不读）；α=(0,0)∧β=(1,0) 时 y 亦可为 nullptr。

# 可维可测分析（required）

## 精度标准/性能标准
| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | rtol=atol=2⁻¹³，ratio≥0.99，max_abs≤1e-2 或 32ULP（golden 动态）；实部/虚部分别比对；Inf/NaN 等效规则 | 任务书 §3.2 + mixed_tolerance_standard.md（28a1ce5 版） |
| 性能 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv | 任务书 §3.3 |

## 测试概览
- 用例分类：精度（uplo 全组合小尺寸 / 尺寸扫描 / 标量特殊值 (0,0),(1,0),纯虚,大值 / lda padding / 步长 ±1,±2,±3 与负步长 / Inf/NaN / β=(0,0) 时 y=NaN 用例）、负向（空指针按标量值条件、非法枚举、非法 lda、负维度、零步长）、性能 200 条。
- 期望码：非法枚举 → INVALID_ENUM；非法 lda/负维度/零步长/条件空指针 → INVALID_VALUE；handle → HANDLE_IS_NULLPTR；n=0 与 α=(0,0)∧β=(1,0) → SUCCESS。
- 负向用例缓冲区：不按 (n−1)×|inc| 预分配，dummy/NULLPTR 占位（§3.5 规则）；quick return 用例同样不分配。
- 性能口径：msprof op，warmup5+≥10 均值；正态用例（GAUSS_NORM_5_2）扩展补齐。

## 兼容性分析
- 新增公共头声明（仅追加）；`test/symv/chemv/CMakeLists.txt` 补 arch22 GTest 注册（任务书明确要求的连带修复，波及面仅测试构建，不影响算子实现与其他 arch）。

---
### 与官方模板的差异说明
按套件《设计文档模板》无编号六段结构书写；性能要点为规划值，终态数据待 S5 实测回填。
