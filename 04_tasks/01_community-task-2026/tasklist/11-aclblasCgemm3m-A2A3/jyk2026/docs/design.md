# aclblasCgemm3m 算子设计文档（Atlas A2/A3）

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：COMPLEX64（实部/虚部各 FLOAT32）
> 合入位置：`ops-blas/blas/gemm/arch22/`；测试：`ops-blas/test/gemm/cgemm3m/arch22/`
> 任务书：`aclblasCgemm3m_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasCgemm3m 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemm/arch22/`；测试代码 → `test/gemm/cgemm3m/arch22/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件（设计文档 / 自测用例与测试代码 / 自测报告 / 待验收代码地址，§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入。

## 背景介绍

### 算子功能定位

单精度复数三乘法通用矩阵乘：`C = alpha * op(A) * op(B) + beta * C`，其中 op ∈ {N, T, C}，`OP_C` 为共轭转置、`OP_T` 为非共轭转置（§2.1）。核心价值在于用 Gauss 3M 算法把 1 次复数矩阵乘分解为 3 次实数矩阵乘，减少复数乘法次数、增大实数乘加的并行粒度。

### 业界对标

- 对标接口：cuBLAS `cublasCgemm3m`（数学语义参考 Netlib `cgemm`）。
- 语义差异点：非法 trans 枚举 cuBLAS 返回 `CUBLAS_STATUS_INVALID_VALUE`，ops-blas 约定返回 `ACLBLAS_STATUS_INVALID_ENUM`（§2.3，有意差异）；alpha/beta 固定指向 Host 内存（与 cuBLAS 默认 pointer mode 一致）。
- 数值差异：3M 分解的中间项舍入与直接复数乘法不同，结果不保证与 `cblas_cgemm` 逐位一致，按 §3.2 阈值判定。

### 工程形态

- 工程模式：Host C API + Ascend C kernel 直调（handle 绑定 stream，§2.2）。
- 代码落位：`blas/gemm/arch22/`，在 `gemm_host.cpp` 增加 `aclblasCgemm3m` 入口，`gemm_kernel.cpp/.h` 增加 3M 复数 kernel 路径。
- 仓内近亲：`blas/gemm/arch35/cgemm3m_host.cpp`（arch35/950PR 已有 3M 实现，可参考路径选择与缩放逻辑）；同目录 `aclblasSgemm` arch22 实现提供 arch22 的 tiling/分核基线。**禁止**把实现放入 `blas/gemm3m/`（那是实数 Sgemm3m 接口族）。

### 接口规格

签名与 `include/cann_ops_blas.h`（L584）一致，本任务无需新增声明：

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

参数规格固定表（§2.4 逐行）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | - | - | 非空 | - |
| transa/transb | op(A)/op(B) 运算类型 | attr int | - | {OP_N, OP_T, OP_C}，非法枚举返回 INVALID_ENUM | - |
| m/n/k | op(A) 行数 / op(B) 列数 / 归约维 | attr int | - | ≥0；m或n=0 合法 no-op；k=0 只做 C=beta*C | - |
| alpha/beta | 复数标量（Host 内存） | scalar | COMPLEX64 | 非空（quick return 时仍须非空指针） | - |
| A/B | 输入矩阵，列主序 | tensor | COMPLEX64 | 连续带填充布局；不与 C 重叠 | op(A) m×k、op(B) k×n |
| lda/ldb/ldc | 前导维度 | attr int | - | lda≥max(1,m或k)、ldb≥max(1,k或n)、ldc≥max(1,m) | - |
| C | 输出矩阵，原地覆写 | tensor | COMPLEX64 | 仅写逻辑 m×n 区域 | m×n |

### 数学公式

`C = alpha * op(A) * op(B) + beta * C`；3M 分解（§2.1）：

- 非融合路径：P0 = op(Ar)·op(Br)，P1 = op(Ai)·op(Bi)，P2 = op(Ar+Ai)·op(Br+Bi)，`op(A)·op(B) = (P0 − P1) + i·(P2 − P0 − P1)`；
- 融合路径（含纠正重算）：P0 = op(Ar+Ai)·op(Br)，P1 = op(Ar)·op(Bi−Br)，P2 = op(Ai)·op(Br+Bi)，`op(A)·op(B) = (P0 − P2) + i·(P0 + P1)`。

两条路径代数等价但 FP32 舍入不同，误差分析不能互相套用。

# 需求分析（required）

## 需求描述

以 Ascend C 在 arch22 上实现复数 3M 矩阵乘：3 次实数 SGEMM kernel 复用 + 复数标量缩放，支持 N/T/C 全组合、任意非零规模与带填充布局，精度达 FLOAT32 分量阈值（rtol/atol=2^-13，matched_ratio≥0.99），性能达 GPU kernel 标杆 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | 3M 两套等价分解（§2.1）；实部/虚部可拆成 3 次实 SGEMM；op(N/T/C) 作用于实部、虚部矩阵的方式一致，可统一入口 |
| 2 | 接口原型 | `include/cann_ops_blas.h` L584 已有声明，参数序列与 `aclblasCgemm` 一致，无需改公共头 |
| 3 | 参数规格 | §2.4 固定表，alpha/beta 为 Host 指针，A/B/C 为 Device 列主序 |
| 4 | 支持数据类型 | COMPLEX64（内部按 FLOAT32 分量计算） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → arch22（ascend910b3），落 `blas/gemm/arch22/` |
| 6 | 精度要求 | FLOAT32 分量 rtol=atol=2^-13、matched_ratio≥0.99、max_abs≤1e-2 或 32ULP（§3.2，可溯源） |
| 7 | 性能预期 | kernel-to-kernel：`msprof op` Task Duration ≤ `gpu_ms` × 1.25（即 0.8 倍 GPU 标杆，§3.3，可溯源） |
| 8 | 代码架构选型 | 3 次实 SGEMM 复用 + host 侧 3M 调度；可行性依据：arch35 已有同构实现、arch22 SGEMM 基线达标；已排除死路——直接在 kernel 内做复数乘（乘加粒度差、寄存器压力大于 3M 拆分）与把实现放 `blas/gemm3m/`（接口族不符，§2.2 禁止） |
| 9 | 评测来源 | golden = CPU cblas（Netlib 3.10.0）随测试工程提供；性能 baseline = `gpu_baseline.csv`（cuBLAS cublasCgemm3m kernel-only） |

# 详细设计（required）

## 算子分析

### 数学公式

见"数学公式"节：两套 3M 分解共享同一实 SGEMM 内核，仅乘子矩阵组合不同（Br+Bi、Ar+Ai、Bi−Br 需要时在片上逐元素合成）。

### 支持数据类型

COMPLEX64（实部/虚部各 FLOAT32）；标量 alpha/beta 同型，Host 内存。

### 支持形状

| 矩阵 | 形状约束 | 退化 |
| --- | --- | --- |
| op(A) | m×k；物理 lda×k（N）或 lda×m（T/C） | m=0/n=0 no-op；k=0 只做 C=beta*C |
| op(B) | k×n；物理 ldb×n（N）或 ldb×k（T/C） | 同上 |
| C | m×n；物理 ldc×n，仅写逻辑区，padding 列哨兵保护 | beta=0 不读旧值（C 可含 NaN） |

## 算子实现

### 实现方案

#### 总体架构与数据流

Host 侧把 1 次 Cgemm3m 调度为 3 次实 SGEMM kernel（复用 arch22 SGEMM 基线的分块与流水）：

```
host: 校验 → quick return 判定 → 3M 路径选择 → 下发 3 次 SGEMM（复数标量折叠进各次调用）
数据流: GM(A/B 实虚部) → L1/L0 分块 → UB 逐元素合成（Ar+Ai / Br+Bi / Bi−Br）→ 实 SGEMM 累加 → GM(C)
```

| kernel 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| 非融合路径 | 片上容量充足、中等规模 | 3 次标准 SGEMM，P2 的操作数需片上合成 |
| 融合路径 | 大 K、大矩阵，减少一次乘法访存 | 3 次含纠正重算的 SGEMM，Bi−Br 片上合成 |
| beta-C 路径 | alpha=0 或 k=0 | 仅 C=beta*C，不启动 SGEMM |

#### host 侧设计

- **分核策略**：沿用 arch22 SGEMM 的满核均分（按 C 的 m×n 网格切核，余块尾核处理）；三次 SGEMM 串行下发、同 stream 内顺序执行。
- **数据分块和内存优化策略**：复用 SGEMM 的 L1/L0 double buffer tile；中间矩阵不落 GM——P0/P1/P2 由同一 C 缓冲按序累加（乘子以 (1,0)/(−1,0)/(1,±i) 等价折算为实部加减），避免 3 份 m×n 临时空间。
- **tilingkey 规划策略**：key 由 {路径(融合/非融合), transa/transb 组合, 尾块标志} 编码，编译期展开消除内层分支。
- **数据检测**：按 §2.5 校验顺序——handle→trans 枚举→m/n/k→lda/ldb/ldc→alpha/beta/A/B/C 指针；结构合法后再 quick return；不做 Device 元素级扫描。

#### kernel 侧设计

- Init：解析 tiling（分块坐标、路径 key）、设置 CopyIn 窗口；
- Process：CopyIn（A/B 分块，负/共轭转置在搬运或合成阶段处理 OP_T/OP_C 语义：OP_C 在 UB 合成 Ar+Ai 阶段做实部取反）→ Compute（FMA 累加）→ CopyOut（beta 折算：beta=0 直写、beta≠0 先读旧值缩放）；
- workspace：融合路径无需额外 GM workspace；非融合路径 P2 操作数合成在 UB 内完成，workspace 布局仅含对齐填充。

#### 性能优化要点（终态）

- 优化项→预期效果：3M 复用实 SGEMM 内核→乘加并行粒度对齐 SGEMM 基线；中间量片上合成→GM 访存从 5 次矩阵读写降到 2 读 1 写；double buffer→搬运与计算重叠。实测数据在开发阶段按 §3.3 口径（warmup 5、采样 10、msprof op）回填自测报告。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2（800T A2/910B3） | ✔（本任务目标） |
| Atlas A3 | ✔（本任务目标） |
| Atlas 950PR | arch35 已有实现，不涉及 |

## 算子约束限制

- 仅支持连续列主序带填充布局，不支持任意非连续布局与 broadcast（§2.5）；
- 不使用 AutoContiguous/ViewCopy，不产生相关临时空间；
- C 与 A/B 存储不得重叠；C 原地覆写；
- 非确定性：路径选择与 K 归约顺序随形状变化，两次执行不保证逐位一致；
- 异步执行：结果在 handle 绑定 stream 上生效，读回前须同步。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 分量按 FLOAT32 阈值：rtol=atol=2^-13，matched_ratio≥0.99，max_abs≤1e-2 或 32ULP；逐元素 \|actual−golden\|≤atol+rtol·\|golden\|，实部/虚部分别比对 | §3.2 + mixed_tolerance_standard |
| 性能标准 | msprof op kernel Task Duration ≤ gpu_ms×1.25（cuBLAS cublasCgemm3m kernel-only 基线）；GTest avg_us 仅交叉验证 | §3.3 |

## 测试概览

- 用例分类：精度全量 1000 条（CSV 驱动，ATK 框架加载）；性能子集 200 条（small/medium/large 分层，含约 12% 非默认标量）；负向用例覆盖非法枚举、非法维度/前导维、空指针（§3.5）。
- 特殊场景：m/n/k=0、alpha=0、beta=0 且 C=NaN、k=0 且 beta≠1、OP_C 非对称矩阵、ldc padding 哨兵、Inf/NaN 传播。
- 性能口径：warmup 5 + 采样 10 取平均；msprof op Task Duration 为正式判据。
- 复现入口：`bash build.sh --ops=cgemm3m --soc=ascend910b3`，自验包 readme 步骤。

## 兼容性分析

- 新增 arch22 路径不修改公共头（声明已存在），不影响 arch35 及其他产品线；
- `gemm_host.cpp`/`gemm_kernel.cpp` 的增量入口对既有 `aclblasSgemm`/`aclblasCgemm` 路径无行为改动；
- README.md 补充 `aclblasCgemm3m` 的 Atlas A2/A3 支持说明，无兼容性风险。
