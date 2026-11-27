# aclblasCherk3mEx 算子设计文档（Atlas 950PR）

> 目标芯片：昇腾 950PR（arch35，SOC=ascend950）｜CANN 9.1.0+
> 数据类型：A ∈ {COMPLEX64, COMPLEX8I(int8交错)}，C = COMPLEX64
> 合入位置：`ops-blas/blas/herk/arch35/`；测试：`ops-blas/test/herk/cherk3m_ex/arch35/`
> 任务书：`aclblasCherk3mEx_Atlas950PR_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasCherk3mEx 算子开发（昇腾 950PR，arch35）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/herk/arch35/`；测试代码 → `test/herk/cherk3m_ex/arch35/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入（§4）。

## 背景介绍

### 算子功能定位

单精度复数 Hermitian 矩阵秩-k 更新（扩展版）：`C = alpha·op(A)·op(A)^H + beta·C`，C 为 n×n Hermitian 复数矩阵，仅 uplo 三角被读写、对角虚部强制置 0。Ex 体现在两点：A 支持 ACL_INT8 复数（每元素 (int8 real, int8 imag) 交错 2 字节，对应 cuBLAS `CUDA_C_8I`）；alpha/beta 为实数 FLOAT32 标量。

### 业界对标

- 对标接口：cuBLAS `cublasCherk3mEx`（数学语义参考 Netlib `cherk`）。
- 有意差异：① 非法 uplo/trans 枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（cuBLAS 返回 INVALID_VALUE）；② `trans = OP_T` 不属于 Hermitian 秩-k 更新定义域，返回 INVALID_ENUM（cuBLAS 文档虽列 OP_T 但公式未定义之）；③ Atype/Ctype 组合不支持集返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
- 数值差异：3M 舍入与直接复数乘不同，与 `cblas_cherk` 不保证逐位一致，按阈值判定。

### 工程形态

- 工程模式：Host C API + Ascend C kernel 直调（handle 绑定 stream）。
- 代码落位：`blas/herk/arch35/` 新增 `cherk3m_ex_host.cpp` / `cherk3m_ex_kernel.cpp`；**需在 `include/cann_ops_blas.h` 新增声明**（该接口 1125 前不存在）。
- 仓内近亲：非 Ex `cherk_host/cherk_kernel`（参数校验、uplo 三角处理、对角虚部置 0、缩放逻辑）；`blas/gemm/arch35/cgemm3m_host.cpp` 与 `cgemm3m_rounding_kernel.cpp`（3M 分解与舍入处理）。**不得依赖/调用未合入的 `aclblasCherkEx`，3M 三次实数乘必须同一次 kernel launch 内完成，不得用多次 GEMM 调用拼接**（§2.2）。

### 接口规格

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const float* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

参数规格固定表（§2.4 逐行）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | - | - | 非空 | - |
| uplo | C 存储三角模式 | attr int | - | {UPPER, LOWER}，非法→INVALID_ENUM | - |
| trans | op(A) 类型 | attr int | - | {OP_N, OP_C}；OP_T/非法→INVALID_ENUM | - |
| n/k | C 阶数 / 秩-k 维度 | attr int | - | ≥0；n=0 no-op；k=0 秩-k 贡献 0 | - |
| alpha/beta | 实数标量 | scalar | FLOAT32 | 非空 | - |
| A | 输入矩阵，列主序 | tensor | COMPLEX64 / COMPLEX8I | void*，按 Atype 解释 | n×k（N）或 k×n（C） |
| Atype | A 的数据类型 | attr int | {ACL_COMPLEX64, ACL_INT8} | 组合不支持→NOT_SUPPORTED | - |
| lda | A 前导维度 | attr int | - | ≥max(1,n)（N）或 max(1,k)（C） | - |
| C | n×n 输出，原地覆写 | tensor | COMPLEX64 | 仅 uplo 三角读写 | ldc×n |
| Ctype | C 的数据类型 | attr int | {ACL_COMPLEX64} | 其他→NOT_SUPPORTED | - |
| ldc | C 前导维度 | attr int | - | ≥max(1,n) | - |

### 数学公式

`C = alpha·op(A)·op(A)^H + beta·C`，**乘积是 X·X^H 而非 X·X^T**。Hermitian 专用 3M 分解（§2.1 第 7 条）：令 X = op(A) = Xr + i·Xi，B = X^H ⇒ Br = Xr^T、Bi = −Xi^T，取正号形式：

- P0 = Xr·Xr^T，P1 = Xi·Xi^T，P2 = (Xr+Xi)·(Xr−Xi)^T；
- 实部 = P0 + P1，虚部 = P2 − P0 + P1。

错误地套用通用 X·X^T 对称分解会把虚部算成 `Xi·Xr^T + Xr·Xi^T`（正确值 `Xi·Xr^T − Xr·Xi^T`），属实现错误，设计中以用例显式防呆（§3.5 `TC_3M_*` 虚部抵消用例）。

# 需求分析（required）

## 需求描述

以 Ascend C 在 arch35 上实现 Hermitian 秩-k 更新：同一次 launch 内完成 3 次实数矩阵乘的 3M 分解，支持 A 的 COMPLEX64/INT8 双 dtype 入口与 Hermitian 三角语义，精度达 FLOAT32 分量阈值，性能达 GPU kernel 标杆 0.4×（单 case）/平均 0.5× 以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | X·X^H 的 Hermitian 3M 分解（上式）；C 仅 uplo 三角需计算，对称半区免费；对角虚部恒 0 可免算 |
| 2 | 接口原型 | 新增声明于 `include/cann_ops_blas.h`，与 cuBLAS 逐参对应（§2.3） |
| 3 | 参数规格 | §2.4 固定表；A 为 void* 按 Atype 分派解释 |
| 4 | 支持数据类型 | (A=ACL_COMPLEX64,C=COMPLEX64) 与 (A=ACL_INT8,C=COMPLEX64)；INT8 分量精确提升 float 后计算 |
| 5 | 目标芯片与 arch 映射 | 昇腾 950PR → arch35（ascend950），落 `blas/herk/arch35/` |
| 6 | 精度要求 | FLOAT32 分量 rtol=atol=2^-13、matched_ratio≥0.99、max_abs≤1e-2 或 32ULP（§3.2，可溯源） |
| 7 | 性能预期 | kernel-to-kernel：单 case Task Duration ≤ gpu_ms×2.5，全 case 平均倍率 ≥0.5（§3.3，可溯源） |
| 8 | 代码架构选型 | 单 launch 融合 3M kernel：P0/P1/P2 共享 A 的搬运与三角分块，片上完成 (Xr+Xi)/(Xr−Xi) 合成；可行性依据：非 Ex cherk（校验/三角/缩放骨架）+ cgemm3m（3M 分解与舍入）均已在仓。已排除死路——三次独立 GEMM 拼接（违反 §2.2 且引入额外 GM 往返）、套用 X·X^T 对称分解（虚部符号错误） |
| 9 | 评测来源 | golden = cblas_cherk（Netlib 3.10.0，INT8 先精确提升）；性能 baseline = `gpu_baseline.csv`（cuBLAS cublasCherk3mEx kernel-only） |

# 详细设计（required）

## 算子分析

### 数学公式

见"数学公式"节：P0/P1/P2 三项共享 X 的分块搬运；上三角/下三角由 uplo 选择输出侧写入范围，另一三角不写。

### 支持数据类型

| 输入 | dtype | 处理 |
| --- | --- | --- |
| A | COMPLEX64 | 直接按 float32 对计算 |
| A | COMPLEX8I（ACL_INT8） | 交错 (int8,int8) 2B/元素，CopyIn 阶段拆包并精确提升 float，无精度损失 |
| alpha/beta | FLOAT32 | Host 标量 |
| C | COMPLEX64 | 原地读写 uplo 三角 |

### 支持形状

| 矩阵 | 形状约束 | 退化 |
| --- | --- | --- |
| A | n×k（trans=N）或 k×n（trans=C），带 lda 填充 | k=0/alpha=0 不引用 A |
| C | n×n（uplo 三角），带 ldc 填充 | n=0 no-op；beta=0 不读旧值 |

## 算子实现

### 实现方案

#### 总体架构与数据流

```
host: 校验(uplo/trans→n/k→lda/ldc→Atype/Ctype→指针) → quick return 判定 → tiling → 单次 launch
数据流: GM(A 实/虚分量) → UB 拆包(int8)/直接读取 → 片上合成 Xr+Xi、Xr−Xi 分块
        → 3 路实数 FMA（P0/P1/P2 累加）→ 三角掩码写回 GM(C)，对角虚部清 0
```

| kernel 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| 3M 主路径 | k>0 且 alpha≠0 | 单 launch 内完成 P0/P1/P2 三路实数矩阵乘与虚/实部合成 |
| beta 缩放路径 | k=0 或 alpha=0 且 beta≠1 | 仅 uplo 三角 C=beta·C |
| quick return | n=0；或 (k=0 或 alpha=0) 且 beta=1 | 不 launch |

#### host 侧设计

- **分核策略**：按 C 的 uplo 三角做满核均分（三角分块映射到核阵列，边缘尾块单独分派）；k 维在片内累加，避免跨核归约。
- **数据分块和内存优化策略**：A 按 k 方向 double buffer 分块搬入；int8 输入的拆包提升在 UB 内与搬运流水重叠；P0/P1/P2 复用同一批 A 分块，中间量不出片上；C 只写 uplo 三角，padding 列不动。
- **tilingkey 规划策略**：key 由 {Atype(COMPLEX64/INT8), uplo, trans, 尾块标志} 编码，编译期展开拆包与三角掩码分支。
- **数据检测**：按 §2.5 校验顺序逐参数校验；Atype/Ctype 支持集外返回 NOT_SUPPORTED；结构合法后再 quick return；不做 Device 元素级扫描。

#### kernel 侧设计

- Init：解析 tiling、按 key 选择拆包与三角掩码常量；
- Process：CopyIn（A 分块 + C 旧值按需）→ Compute（int8 拆包/合成 (Xr+Xi)、(Xr−Xi) → 三路 FMA → 实部 P0+P1、虚部 P2−P0+P1 组装）→ CopyOut（beta 折算、三角掩码写回、对角虚部强制置 0）；
- workspace：无额外 GM workspace；中间合成全部在 UB。

#### 性能优化要点（终态）

- 三路乘共享搬运与分块→访存约为 1 次直接复数 GEMM 的 1/3 量级；int8 入口带宽减半（2B/元素）→搬运敏感场景受益；double buffer→搬运与 FMA 重叠。实测按 §3.3.1 口径（warmup 5、有效采样 10、msprof op Task Duration）回填自测报告。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas 950PR | ✔（本任务目标） |
| Atlas A2/A3（arch22） | 不涉及 |

## 算子约束限制

- trans 仅 {OP_N, OP_C}；OP_T 显式拒绝（INVALID_ENUM）；
- (Atype, Ctype) 仅两种支持组合，其余 NOT_SUPPORTED；
- 仅连续列主序带填充布局；不支持广播、非连续布局；C 原地覆写且仅 uplo 三角可写；
- 非确定性：K 归约顺序随形状变化，验收按阈值判定；异步执行语义同 §2.5。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 分量按 FLOAT32：rtol=atol=2^-13，matched_ratio≥0.99，max_abs≤1e-2 或 32ULP；golden=cblas_cherk（INT8 精确提升后调用），实部/虚部分别比对 | §3.2 + mixed_tolerance_standard |
| 性能标准 | msprof op Task Duration：单 case ≤ gpu_ms×2.5 且全 case 平均倍率 ≥0.5（cuBLAS cublasCherk3mEx kernel-only 基线）；[PERF] avg_us 仅交叉验证 | §3.3/§3.3.1 |

## 测试概览

- 用例分类：精度全量 1000 条（含 `TC_3M_*` 专项 4 条：交替/极值填充下虚部抵消、int8 极值、对角虚部置 0）；性能子集 200 条（dtype×规模档分层）；
- 覆盖：uplo×trans 全组合、lda/ldc padding（+1/+8/+64）、n/k 边界（0/1/2/3/…/4096）、alpha/beta 特殊值（0/±1/±1e4）、Atype=INT8 交错输入与边界值、负向（非法枚举、OP_T、不支持组合、lda/ldc 越界、空指针）；
- 性能口径：msprof op 正式判据 + verify_performance.py 交叉验证；int8 性能沿用 COMPLEX64 基线自行补采（§3.3.6）；
- 复现入口：`bash build.sh --ops=cherk3m_ex --soc=ascend950`，`msprof op --application=…--gtest_filter=*TC_PF*`。

## 兼容性分析

- 本接口为新增声明（append 到 `include/cann_ops_blas.h`），不修改既有符号，对 `aclblasCherk`/`aclblasCherkEx` 无依赖、无行为影响；
- 新增文件独立（`cherk3m_ex_*`），不改动既有 herk 编译目标；README 按仓规范补充章节；
- 无公共头修改波及面，无兼容性风险。