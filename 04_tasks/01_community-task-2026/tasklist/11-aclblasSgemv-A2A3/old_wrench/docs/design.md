# aclblasSgemv 算子设计文档（Atlas A2/A3，arch22）

# 需求背景（required）

## 需求来源
- 任务：11月社区任务-aclblasSgemv算子开发（A2A3）；算子 aclblasSgemv（对标 cuBLAS cublasSgemv，CUBLAS_Library.pdf §2.6.2）。
- 合入目标仓与目录：https://gitcode.com/cann/ops-blas `blas/gemv/arch22/`（实现）+ `test/gemv/sgemv/arch22/`（测试），开发分支 master。
- 验收流程：hiascend 社区任务（cann-community-task），设计文档先于代码提交（cann-competitions MR）。

## 背景介绍
### 算子功能定位
单精度矩阵-向量乘加：`y = α·op(A)·x + β·y`，op(A) ∈ {A, Aᵀ, Aᴴ}（实数矩阵 Aᴴ≡Aᵀ），y 原地更新。补齐 Blas 仓 Atlas A2/A3 系列 BLAS-2 矩阵-向量能力缺口。

### 业界对标
- cublasSgemv（cuBLAS §2.6.2）：参数序列、返回码与快速返回口径唯一依据；A 列主序、支持正负 incx/incy。
- Netlib BLAS sgemv.f：golden 语义（测试真值生成依据，仓内 sgemv_golden.h 的 cblas_sgemv float32 直算）。
- 语义差异点：本实现 α=0 不短路（仍按公式计算），golden cblas_sgemv α=0 短路为 y=βy——两者仅当 A/x 含 Inf/NaN 时不同（0×Inf=NaN），故该组合不设精度用例（任务书 §2.1）。

### 工程形态
- Host C API + Kernel 直调（与同仓 `blas/gemv/arch22/` 既有 ascblasCgemv 同工程模式：host 校验/tiling + `extern "C" __global__ __aicore__ __vector__` SIMD 内核 `<<<numBlocks, nullptr, stream>>>` 直调）。
- 代码落位：`blas/gemv/arch22/`（host 三件 + utils 头，并入既有目录不新建族目录）；接口声明复用 `include/cann_ops_blas.h` 既有 aclblasSgemv（与 950PR 共用，不改公共头）。
- 仓内近亲：arch22 ascblasCgemv（工程骨架，复数版同构简化）；arch35 sgemv_host.cpp（ValidateSgemvParams 校验语义锚点 L31-107）。

### 接口规格
```cpp
aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, const float* x, int incx,
    const float* beta, float* y, int incy);
```
（与 include/cann_ops_blas.h L471-475 既有声明逐参一致）

参数规格固定表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄（携带 stream） | aclblasHandle_t | attr | aclblasCreate 返回非空句柄；nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| trans | 矩阵操作类型 N/T/C | aclblasOperation_t | attr | ∈{N,T,C}；非法 → INVALID_VALUE | - |
| m | A 的行数 | int | attr | m≥0；m<0 → INVALID_VALUE；m=0 quick return | - |
| n | A 的列数 | int | attr | n≥0；n<0 → INVALID_VALUE；n=0 quick return | - |
| alpha | 标量 α 指针（Host） | const float* | FLOAT32 | 非空；nullptr → INVALID_VALUE；值不校验 | - |
| A | 列主序系数矩阵（Device） | const float* | FLOAT32 | 维度 lda×n 前 m×n 有效；m>0 且 n>0 时非空；NaN/Inf 自然传播 | [lda,n] |
| lda | 列主序相邻列间距 | int | attr | lda ≥ max(1,m)；违反 → INVALID_VALUE | - |
| x | 输入向量（Device） | const float* | FLOAT32 | 逻辑长度 lenX（N:n / T、C:m）；非零即非空；(lenX−1)×\|incx\| 超 int32 → INVALID_VALUE（int64 计算） | [1+(lenX−1)·\|incx\|] |
| incx | x 步长 | int | attr | ≠0 且 ≠INT32_MIN，正负均支持 | - |
| beta | 标量 β 指针（Host） | const float* | FLOAT32 | 非空；β=0 时 y 输入不被读取 | - |
| y | 输入兼输出向量（原地，Device） | float* | FLOAT32 | 逻辑长度 lenY（N:m / T、C:n）；非零即非空；跨度溢出同 x 口径 | [1+(lenY−1)·\|incy\|] |
| incy | y 步长 | int | attr | ≠0 且 ≠INT32_MIN，正负均支持 | - |

### 数学公式
y[j] = α × Σ(i=1..k) op(A)[j,i] × x[i] + β × y[j]，j = 1..len(y)；
op(A)=A（N，k=n、len(y)=m）或 Aᵀ（T/C，k=m、len(y)=n）；A[i,j] 位于列主序偏移 (i−1)+(j−1)×lda（1-based）；β=0 不读 y 输入值。

# 需求分析（required）

## 需求描述
用 Ascend C kernel 直调（SIMD 向量内核）在 Atlas A2/A3（arch22/ascend910b）实现 aclblasSgemv 全语义（含正负步长、lda 空洞、跨度溢出守卫），精度满足 mixed_tolerance FLOAT32 口径全绿，性能单 case 达 0.8×A100 cublasSgemv（≤1.25× 标杆耗时）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | y=α·op(A)x+βy；实数版无共轭（C≡T）；列主序访问与 arch22 cgemv 同构（复数机制全删，搬运/分块保留） |
| 2 | 接口原型 | 见上节；声明已在公共头，不另立接口（任务书 §7.1） |
| 3 | 参数规格 | 见参数规格表；校验顺序对齐 arch35 ValidateSgemvParams + A2/A3 补跨度溢出守卫（§2.4/§2.5，issue #866/#860） |
| 4 | 支持数据类型 | FLOAT32（单接口单 dtype） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3（910B 系）→ arch22 / SOC=ascend910b*；自验覆盖 910B3（§3.1） |
| 6 | 精度要求 | mixed_tolerance：rtol=atol=2⁻¹³，ratio≥0.99，max_abs≤1e-2 或 32ULP；Inf/NaN 等效规则（§3.2） |
| 7 | 性能预期 | msprof op task duration（warmup5+≥10 均值）≤1.25×gpu_baseline.csv（§3.3）；4096×4096 N 标杆 52.63µs |
| 8 | 代码架构选型+可行性 | host（校验+tiling+直调）+ 双 SIMD 内核（no_trans / do_trans），事件化软件流水；已排除死路：SIMT（simt_api 仅 arch35）、arch35 SIMT 内核不可移植 |
| 9 | 评测来源 | 精度 golden=cblas_sgemv（sgemv_golden.h）；性能标杆=任务包 gpu_baseline.csv（A100 实测） |

# 详细设计（required）

## 算子分析
### 数学公式
同需求背景；kernel 内按输出维分块：N 按 M 行分块（每块点积一行跨 n 列），T/C 按 N 列分块（每块归约一列 m 元素）。

### 支持数据类型
FLOAT32（A/x/y/α/β 全单精度实数）。

### 支持形状
m,n≥0 任意；m=0 或 n=0 quick return（不启动 kernel）；非方阵 m≠n、lda>m 列间空洞、|incx/incy|>1 间隔、负步长（自 span 尾端反向）均支持。

## 算子实现
### 实现方案
#### 总体架构与数据流

| 内核变体 | 触发条件 | 职责 |
|---|---|---|
| sgemv_no_trans | trans=N | 按 M×szVert 行块 × N×szHorz 列块分片；A 列块经 matrix_gm2ubuf 搬 UB，x 标量广播，Axpy 融合乘加累积 Ybase；多核按行块交错分配（idxLoop % aiv_num == aiv_id） |
| sgemv_do_trans | trans=T/C | 按 N×szVert 列块输出维分片；A 沿列（连续维）成段搬 UB，x 段乘 A 后跨段归约（mul_v/vmla+cadd_v），尾部位掩码处理非整向量段；C 与 T 同路径（实数无共轭） |

数据流：GM(A,x,y) → MTE2 → UB（Abase/Xbase/Ybase 双缓冲 + 事件 ID 流水）→ V 计算乘加/归约 → MTE3 → GM(y 原地)。UB 规划对齐 cgemv arch22 的 AsdopsBuffer 分区（A 块 2×32KB ping-pong、Y/X 基址区、tmp 缓冲），实数版无实虚分离缓冲。

#### host 侧设计
- 分核策略：numBlocks = min(GetAivCoreCount(), 上限)；输出维（N: m / T: n）按行/列块 ceil 切分后按核交错，满核优先、余块自然落在尾部核。
- 数据分块与内存优化：szVert/szHorz 由 tiling 携带（初版沿 cgemv arch22 常量 512×16(N)/16×512(T)，S5 按带宽实测精调）；sectionDim 控制单核连续处理列组数，软件流水 copy_in(k+1) 前置。
- tilingkey 规划：无独立 tilingkey（单 kernel 双变体由 trans 直接分发）；TilingData 字段：trans/m/n/lda/incx/incy/sectionDim/alpha/beta（int64 对齐 + float 标量，kernel 侧按偏移读）。
- 数据检测（校验顺序，逐条对齐 §2.4）：handle→trans 枚举→m/n→lda→incx/incy（0/INT32_MIN）→跨度溢出（int64：(lenX−1)×\|incx\|、(lenY−1)×\|incy\|>INT32_MAX）→alpha/beta 非空→A（m>0&&n>0）→x（lenX>0）→y（lenY>0）；返回码 INVALID_VALUE / HANDLE_IS_NULLPTR；quick return 在校验全过后。异步下发（不同步流）。

#### kernel 侧设计
```
no_trans（trans=N）:
  for idxLoop in [0, numVert*numSection) 按核交错:
    [MTE2] A 行块列片 + x 列片 + (首列组) y_in 搬 UB（matrix_gm2ubuf 三分支：对齐快路/半对齐/逐列兜底——处理 lda 空洞与非对齐）
    [V] y_in×β 累入 Ybase；逐列 j：Ybase += A[:,j]×x[j]（Axpy 融合，尾部位掩码）
    [MTE3] Ybase 写回 y（负步长反向寻址）
  事件 ID0/ID1 交替构成 copy_in(k+1) 与 compute(k) 流水

do_trans（trans=T/C）:
  for idxVert 列块（输出维）按核交错:
    [MTE2] y_in×β 累入 Ybase；逐 idxHorz 行段: A 段(列连续) + x 段 搬 UB
    [V] 段内 mul_v/vmla 乘加 → cadd_v 跨行归约 → Ybase 累加（尾部 SetVectorMask 位掩码）
    [MTE3] Ybase 写回 y
```
无 workspace 依赖（tiling 走 GM 小缓冲或直调参数，S2 按 csrot 先例定型）；无原子操作。

#### 性能优化要点（终态，S5 回填实测）
- 事件化 ping-pong 双缓冲（P14 同款）；A 搬运对齐快路优先（matrix_gm2ubuf 分支选择）；Axpy 单指令融合乘加。
- 候选：szVert/szHorz/sectionDim 联调、numBlocks 上限放开（Crot P30：40 核 cap 可弃）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列 / Atlas A2 推理系列（910B 系） | ✔ 本任务目标（arch22） |
| Atlas A3 训练系列 / Atlas A3 推理系列 | ✔（同 arch22 门禁随仓声明） |
| Ascend 950PR / 950DT | 已有 arch35 实现（本任务不涉及） |

## 算子约束限制
（逐项对齐任务书 §2.5）
- 非连续：支持（lda 空洞、incx/incy 间隔与负步长；inc=0/INT32_MIN/跨度溢出返回 INVALID_VALUE）。
- broadcast：不涉及（无操作数间广播）。
- dynamic shape：不要求（m/n/lda/inc 运行时入参，host tiling 现算）。
- 原地与视图：y 原地覆写（β=0 不读旧值），不返回视图；A/x 只读。
- 确定性：不要求（归约顺序不保证跨次一致）。
- 空 Tensor：m=0/n=0 quick return，不访问数据。
- 0 维 Tensor：不涉及。
- 异步执行：stream 异步下发，返回成功仅表示已提交。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | FLOAT32 mixed_tolerance：rtol=atol=2⁻¹³(1.2e-4)，matched_ratio≥0.99，max_abs_error≤1e-2 或 32ULP；Inf/NaN 等效判定（同 NaN/同号 Inf/NaN↔±Inf 过，有限值 vs Inf/NaN 不过，非有限值单独统计须全过）；golden=cblas_sgemv float32 直算 | 任务书 §3.2；opbase mixed_tolerance_standard（28a1ce5） |
| 性能标准 | msprof op task duration：warmup 5 次 + 有效采样 ≥10 次算术平均，Avg time ≤ 1.25×gpu_baseline.csv（单 case ≥0.8×GPU）；设备 Atlas 800T A2 (910B3) | 任务书 §3.3 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

## 测试概览
- 用例：任务包 sgemv_test.csv（1200 条，含 TC_L0 基础/TC_ED 边界/TC_NM 负向/TC_PF 性能分层）+ 正态分布补充（GAUSS_NORM_5_2，μ∈[-5,5] σ∈[0.1,2]，扩展 gen_csv.py 生成并纳入全量，自测报告说明数量与覆盖）；覆盖 trans 三值、m/n∈{0,1,3,64,512,2048} 与非方阵、lda∈{max(1,m),m+8}、inc∈{1,2,7,−1,−3}、α∈{0,1,2.5,−1.5,0.1,10}、β∈{0,1,0.6,−0.5}、负向含跨度溢出（如 m=n=4、incx=INT_MAX）。
- 负向缓冲区规则：非 SUCCESS 用例以单元素 dummy/nullptr 占位，不按跨度分配（§3.5）。
- 复现入口：test/gemv/sgemv/arch22/（GTest，BLAS_TEST_CASES 过滤）+ 交付件 03 步骤说明（msprof 采集口径）。

## 兼容性分析
- 新增 arch22 实现，不改公共头、不改 common 框架；arch35 既有 950 实现与测试零波及。
- blas/gemv/README.md 补 aclblasSgemv A2/A3 支持声明（唯一公共文件改动，属任务书 §4/§5 要求的 README 更新）。
- 测试注册：cmake/test.cmake 按 SOC_ARCH_DIRS 自动发现 blas/gemv/arch22/sgemv_*.cpp 后 sgemv_test 目标在 A2/A3 构建自动生效（与仓内 crot 先例同机制，无冗余 CMake 改动）。
