# aclblasCtrsmBatched 算子设计文档

> 任务：9月社区任务 `aclblasCtrsmBatched`（A2/A3），任务表编号 **79**  
> 目标仓：https://gitcode.com/cann/ops-blas  
> 实现目录：`blas/trsmbatched/arch22/`  
> 测试目录：`test/trsmbatched/ctrsmbatched/arch22/`  
> 适配硬件：Atlas 800I/T A2、Atlas A3（arch22 / dav-2201）  
> CANN 版本：9.1.0  
> 模板：https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md  
> CheckList：https://docs.qq.com/sheet/DUHVGUFdmSFRjVFFU?tab=000001  
> 提交位置：`04_tasks/01_community-task-2026/tasklist/09-79-aclblasCtrsmBatched-A2A3/{TeamName}/docs/design.md`

| 项 | 内容 |
|----|------|
| 算子 | aclblasCtrsmBatched |
| 硬件 / CANN | Atlas A2/A3（arch22）/ 9.1.0 |
| 文档日期 | 2026-10-12 |
| TeamName | Hidden_wolf |
| 对标接口 | cuBLAS `cublasCtrsmBatched`；精度 golden：Netlib / cblas `ctrsm` 逐 batch |

---

# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务——《9月社区任务-aclblasCtrsmBatched算子开发(A2/A3)》（`docs/README.md` 任务表第 **79** 项）。要求在 Atlas A2/A3 系列产品上基于 Ascend C（可结合 CATLASS / Matmul API）开发单精度复数（COMPLEX64）**批量**三角线性系统求解算子 `aclblasCtrsmBatched`，完成设计、开发、测试全流程；验收通过后合入 https://gitcode.com/cann/ops-blas 。

工程模式为 **ops-blas 句柄式 Kernel 直调**（非 aclnn / 非图算子）：Host 完成参数校验与 tiling 组装后，经 `handle` 绑定的 stream 下发 NPU Kernel。

## 背景介绍

### aclblasCtrsmBatched 算子实现优化

本算子为 BLAS Level-3 族批量三角求解接口。昇腾生态中无对应 TBE / ACLNN 图算子实现，不以 TBE 路径为标杆。按任务书采用 Kernel 直调，语义对齐 cuBLAS `cublasCtrsmBatched`，精度对标 Netlib `ctrsm`（由 cblas 逐 batch 生成 golden）。

| 角色 | 路径 / 链接 | 说明 |
|------|-------------|------|
| 语义标杆 | [cuBLAS `cublasCtrsmBatched`](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsmbatched) | 批量三角求解、side/uplo/trans/diag、解原地覆写 B |
| 算法 / 特殊值标杆 | [Netlib `ctrsm.f`](https://www.netlib.org/blas/ctrsm.f) | 前向/回代顺序、UNIT 对角、alpha=0、Inf/NaN 跳零分支 |
| 仓内同族 | `blas/trsmbatched/`（已有 arch35 `aclblasStrsmBatched`） | Host 校验、handle/workspace、测试工程布局参考；arch22 路径需独立实现 |
| 公共类型 | `include/cann_ops_blas_common.h` | `aclblasComplex`、枚举与状态码 |
| 接口声明 | `include/cann_ops_blas.h` | 公共头声明，不另设 A2/A3 私有平行接口 |

相对实数 `aclblasStrsmBatched`：元素类型为 `aclblasComplex`；`trans=ACLBLAS_OP_C` 须共轭转置（不可等价为 T）；对角除法为复数除法；累加与规约按 FP32 分量进行。

相对单矩阵 `aclblasCtrsm`：本算子处理 **uniform batch**——各 batch 共享 `(m,n,lda,ldb,side,uplo,trans,diag)`，矩阵地址来自 Device 侧指针数组 `A[]` / `B[]`，batch 间相互独立。

### aclblasCtrsmBatched 算子现状分析

#### 标杆支持的数据类型和数据格式

与 cuBLAS `cublasCtrsmBatched`、Netlib `ctrsm` 及任务书 §2.4 一致（本任务相对 cuBLAS **收紧**：`alpha` 仅支持 Host 指针）：

| 参数 | 含义 | 数据类型 | 支持 dtype | 内存 | 排布 / 形状 | 约束 |
|------|------|----------|------------|------|-------------|------|
| handle | 库上下文，绑定 stream | `aclblasHandle_t` | — | Host | scalar | 空指针 → `HANDLE_IS_NULLPTR` |
| side | 左 / 右乘 | enum | `{LEFT, RIGHT}` | Host | attr | 非法 → `INVALID_VALUE` |
| uplo | 上 / 下三角 | enum | `{UPPER, LOWER}` | Host | attr | 非法 → `INVALID_VALUE` |
| trans | `op(A)` | enum | `{N, T, C}` | Host | attr | 非法 → `INVALID_VALUE` |
| diag | 单位 / 非单位对角 | enum | `{NON_UNIT, UNIT}` | Host | attr | 非法 → `INVALID_VALUE` |
| m | B 行数；LEFT 时 A 阶 | int | m ≥ 0 | Host | scalar | m < 0 非法；m=0 合法 no-op |
| n | B 列数；RIGHT 时 A 阶 | int | n ≥ 0 | Host | scalar | n < 0 非法；n=0 合法 no-op |
| alpha | 复数标量乘数 | COMPLEX64 | 实/虚 FLOAT32 | Host | scalar | 不可为 nullptr；仅 Host |
| A | 批量三角矩阵指针数组 | COMPLEX64 | 实/虚 FLOAT32 | Device | 列主序；LEFT：各 `m×m`（存 `lda×m`）；RIGHT：各 `n×n`（存 `lda×n`） | alpha≠0 时数组及元素不可空；UNIT 不读对角 |
| lda | A 前导维 | int | LEFT：≥max(1,m)；RIGHT：≥max(1,n) | Host | scalar | 否则非法 |
| B | 右端 / 解矩阵指针数组（原地） | COMPLEX64 | 实/虚 FLOAT32 | Device | 列主序，各 `m×n`（存 `ldb×n`） | 数组及元素不可空；各 B[i] 有效区不重叠 |
| ldb | B 前导维 | int | ≥ max(1,m) | Host | scalar | 否则非法 |
| batchCount | 批次数 | int | ≥ 1 | Host | scalar | <1 → `INVALID_VALUE` |

不支持广播；不要求 dynamic shape；不支持超出 lda/ldb 语义的非连续 Tensor。

#### 标杆实现描述

对每个 batch `i = 0 .. batchCount-1` 独立求解，解 **原地覆写** `B[i]`：

```text
side=LEFT :  op(A[i]) * X[i] = alpha * B[i]
side=RIGHT:  X[i] * op(A[i]) = alpha * B[i]
op(A) = A | Aᵀ | Aᴴ   （由 trans = N / T / C 决定）
```

要点：

1. `m==0` 或 `n==0` 为合法 no-op（返回成功，不执行计算）；`batchCount` 必须 ≥ 1。
2. `alpha=(0,0)` 时不引用 A，逐批将 B[i] 有效 `m×n` 区域置零后返回成功。
3. `uplo` 指定引用的三角部分；`diag=UNIT` 时对角视为 1，不读存储对角；`NON_UNIT` 取实际对角，本算子 **不做奇异性检测**。
4. 所有矩阵列主序；仅更新 B 的有效列主序区域，lda/ldb padding 不被改写。
5. 异步依赖 `aclblasSetStream`；读回 Device 结果前须同步 stream。
6. 确定性：不要求浮点累加逐位一致。

#### 标杆流程图

```mermaid
flowchart TD
    A[进入 aclblasCtrsmBatched] --> H0{handle 非空?}
    H0 -->|否| E0[HANDLE_IS_NULLPTR]
    H0 -->|是| B1{枚举 / m,n / lda,ldb / batchCount / alpha 合法?}
    B1 -->|否| E1[INVALID_VALUE]
    B1 -->|是| P[D2H 校验 A[]/B[] 元素非空]
    P -->|否| E1
    P -->|是| C{m==0 或 n==0?}
    C -->|是| E2[SUCCESS no-op]
    C -->|否| Z{alpha==(0,0)?}
    Z -->|是| ZERO[下发 B 有效区置零]
    Z -->|否| T[组装 Tiling / 选择小规模或分块路径]
    ZERO --> E2
    T --> K[Kernel 直调下发]
    K --> E2
```

### aclblasCtrsmBatched 功能分析

| 项 | 内容 |
|----|------|
| 功能 | 批量三角求解，解原地写回各 `B[i]` |
| 输入 | handle, side, uplo, trans, diag, m, n, alpha, A[], lda, B[], ldb, batchCount |
| 输出 | B[]（原地） |
| 数据类型 | COMPLEX64 |
| 模式 | side×uplo×trans×diag = **24** 组合，全部支持 |
| 广播 | 不支持 |

---

# 需求分析（required）

## 需求描述

在 arch22 上实现 `aclblasCtrsmBatched`，满足：

1. 功能与 cuBLAS `cublasCtrsmBatched` 对齐（含 OP_T / OP_C 区分；解原地覆写 B；alpha 仅 Host）。
2. 精度满足生态开源混合容差标准（COMPLEX64 按 FLOAT32 同档）：rtol = atol = 2^-13，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32×ULP。
3. 性能：Atlas 800T A2 (910B3) 上任务书 5 个 case 的平均单次耗时不高于标杆。

| case | batchCount | m | n | side | uplo | trans | diag | 达标耗时（Avg，us） |
|------|------------|---|---|------|------|-------|------|---------------------|
| 1 | 64 | 256 | 256 | LEFT | LOWER | N | NON_UNIT | 895.1 |
| 2 | 128 | 384 | 512 | LEFT | UPPER | C | NON_UNIT | 5551 |
| 3 | 32 | 1024 | 1024 | RIGHT | LOWER | T | UNIT | 13884 |
| 4 | 8 | 2048 | 2048 | LEFT | UPPER | N | NON_UNIT | 21816 |
| 5 | 2 | 4096 | 4096 | RIGHT | UPPER | C | UNIT | 40969 |

## 需求拆解

1. **Host**：参数与指针数组校验、quick return（零维 / alpha=0）、平台资源查询、tiling 组装、Kernel 启动（异步依赖 stream）。
2. **Kernel**：小规模 Vector 直接求解；大规模「对角 panel 向量求解 + 非对角 Cube/Matmul 更新」；复数 AoS↔SoA、转置/共轭打包。
3. **测试**：CSV 驱动 GTest（`test/trsmbatched/ctrsmbatched/arch22/`），golden 对齐 Netlib `ctrsm` 逐 batch；覆盖 24 枚举、边界负向、Inf/NaN 与性能 case。

---

# 详细设计（required）

## 算子分析

### 数学公式

记 `T = op(A)`。LEFT：

```text
x[k,j] = (alpha·B₀[k,j] − Σ_p T[k,p]·x[p,j]) / T[k,k]
下三角：p < k；上三角：p > k。UNIT 时省略除法。
```

RIGHT：

```text
x[i,k] = (alpha·B₀[i,k] − Σ_p x[i,p]·T[p,k]) / T[k,k]
```

大规模路径将 RIGHT 归一化为左侧求解 `C·Y = alpha·D`（`C` 由 side/trans 映射得到，含共轭），最后按映射写回列主序 B。

复数乘法（FP32）：`(ar+iai)(br+ibi) = (ar·br−ai·bi) + (ar·bi+ai·br)i`。

非对角更新（复数拆成四次 FP32 GEMM）：

```text
Yr -= Cr·Yr − Ci·Yi
Yi -= Cr·Yi + Ci·Yr
```

### 支持数据类型

COMPLEX64（`aclblasComplex`：实部/虚部各 float32，AoS 交错）。计算与累加使用 FP32；不启用 HF32 / FP16 降精度。

### 支持形状

- A[i]：LEFT 为逻辑 `m×m`，RIGHT 为逻辑 `n×n`，存储前导维 lda。
- B[i]：逻辑 `m×n`，存储前导维 ldb。
- batchCount ≥ 1；uniform batch（共享维度与标志）。
- 无广播；m/n/batchCount 为运行时入参。

## 算子实现

### 实现方案

#### Host 侧设计

文件规划：`blas/trsmbatched/arch22/ctrsmbatched_host.cpp`

1. **校验顺序**：handle → side/uplo/trans/diag 枚举 → m/n ≥ 0 → batchCount ≥ 1 → lda/ldb 前导维 → alpha 非空 → B 数组非空 →（alpha 非零时）A 数组非空。
2. **指针数组元素校验**：在绑定 stream 上 D2H 读回 Device 指针表并逐元素判空；任一空指针在任何 B 写入前返回 `INVALID_VALUE`。
3. **Quick return**：`m==0` 或 `n==0` → `SUCCESS`（仍要求合法 Host 参数与必需指针）。
4. **alpha=0**：下发仅写 B 有效区的置零 Kernel，不访问 A。
5. **Tiling / 分核**：
   - 记 `Q` 为系数阶（LEFT=m，RIGHT=n），`R` 为右端项数（LEFT=n，RIGHT=m）。
   - **小规模**：当 `Q` 较小且 UB 可容纳系数 + RHS 片时，按 `(batch, RHS 分片)` 分给多个 AIV；每个核完整拥有其 RHS 的依赖链，不跨核拆同一三角依赖。
   - **大规模**：对角 panel 宽度 `P`（候选 32/64），AIV 做 panel 内回代；AIC 做未解区域的四次 FP32 Matmul 更新；按 `(batch 窗口, 未解行块, RHS 列块)` 展开 AIC 任务。
   - 平台核数 / UB / L1/L0 通过 `PlatformAscendC` 运行时查询，禁止写死单一机型常量。
6. **下发**：`ctrsmbatched_kernel_do(tiling, stream)`；返回 SUCCESS 表示已校验并下发（或合法 no-op），不代表 Device 已算完。

**主要 tiling 字段（示意）：**

| 字段 | 含义 |
|------|------|
| m, n, Q, R, lda, ldb | 原始 / 归一化尺寸与寻址 |
| side, uplo, trans, diag | 模式 |
| alphaReal, alphaImag | Host 标量快照 |
| path, rhsTile, panelWidth | 小规模 / 大规模路径与分块 |
| batchStart, waveCount, blockDim | batch 窗口与多核调度 |
| workspaceOffsets / cubeTiling | 工作区与 Matmul 参数 |

#### Kernel 侧设计

文件规划：`blas/trsmbatched/arch22/ctrsmbatched_kernel.cpp`（MIX：AIV + AIC）

| 路径 | 条件（示意） | 行为 |
|------|--------------|------|
| Zero | alpha=0 | 仅写 B 有效 `m×n` 为零 |
| Small Vector | Q 小且 UB 足够 | 搬入有效三角与 RHS；UB 内递推；交织写回 |
| Blocked Mix | 其余大规模 | 打包 C/D 为 SoA FP32 → panel 向量求解 → Cube 四实数乘更新残差 → 散出到列主序 B |
| Special-value fallback | 引用数据含 Inf/NaN | 按 Netlib 跳零分支走 Vector，避免密集 GEMM 的 `0·Inf` 污染 |

**数据流（大规模一轮 panel）：**

```text
AIV: 求解对角块 J 写回 Y[J,:]
  → AIC: Cr·Yr / Ci·Yi → AIV: 更新 Yr[U,:]
  → AIC: Cr·Yi / Ci·Yr → AIV: 更新 Yi[U,:]
  → 下一 panel
```

**UB / Workspace：**

- 小规模：预算约 `8·Qp² + 24·Qp·Sp + Uaux`（系数实虚 + RHS + 临时 / 双缓冲）。
- 大规模：Cr/Ci、Yr/Yi、G0/G1 等平面；单批次主要量级约 `8·Qp² + 16·Qp·Rp`，按可用 workspace 做 batch 窗口复用。
- 使用句柄 `EnsureDefaultWorkspace` / `GetEffectiveWorkspace`；不足时回退流式 Vector，不向不存在容量下发 Cube。

**同步：** MTE2↔Vector↔MTE3 用队列 / 事件；AIC/AIV 交接用跨核 flag 或同 stream 多 kernel 顺序；调用方读回前同步 handle 的 stream。

#### 工程文件

| 文件 | 职责 |
|------|------|
| `ctrsmbatched_host.cpp` | 校验、tiling、launch |
| `ctrsmbatched_kernel.cpp` | MIX Kernel 入口与路径分派 |
| `ctrsmbatched_kernel.h` | `ctrsmbatched_kernel_do` 声明 |
| `ctrsmbatched_tiling_data.h` | `CtrsmBatchedTilingData` |
| `include/cann_ops_blas.h` | API 声明 |
| `test/trsmbatched/ctrsmbatched/arch22/*` | CSV + GTest |

合入目录以任务书第 5 节为准（`blas/trsmbatched/arch22/`）；第 2.2 节写的 `blas/trsm/` 与仓内同族及第 5 节不一致，本设计按第 5 节与仓内 `trsmbatched` 族路径执行。

---

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|----------------|----------|
| Atlas 800I/T A2 | √ |
| Atlas A3 系列 | √ |

（自验证 / 性能以 Atlas 800T A2 (910B3) 为主，与任务书 §3.1 一致；§7 要求 A2 与 A3 均完成验收。）

---

# 算子约束限制

| 约束项 | 内容 |
|--------|------|
| 参数合法性 | 见任务书 §2.4 / §2.5；非法返回对应 `aclblasStatus_t` |
| 非连续 Tensor | 仅支持 lda / ldb 语义 |
| broadcast | 不涉及 |
| dynamic shape | 不要求（m/n/batchCount 运行时入参） |
| 原地语义 | 各 B[i] 原地覆写；各 B[i] 有效区不得重叠；A 与 B 不得破坏性重叠 |
| 确定性 | 不要求 |
| 空 / 零维 | m=0 或 n=0 合法 no-op；alpha=(0,0) 置零 B |
| 异步 | 依赖 `aclblasSetStream`；读回前须同步 |
| alpha | 仅 Host 指针（相对 cuBLAS 收紧） |
| batchCount | 必须 ≥ 1（0 / 负值非法） |

---

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度标准 | COMPLEX64：rtol=atol=2^-13；`|a-g| ≤ atol + rtol·|g|`；matched_ratio≥0.99；max_abs_error≤1e-2 或 32×ULP | 任务书 §3.2 / 生态开源混合容差标准 |
| 性能标准 | case1–5 平均单次耗时 ≤ 上表标杆；warmup 后有效采样 >10 次取平均 | 任务书 §3.3 |
| 自测手段 | CSV + GTest；golden 为 cblas `ctrsm` 逐 batch；可辅 msprof 统计完整逻辑调用下全部相关 kernel | 任务书 §3.5 |

## 测试覆盖设计

| 类别 | 覆盖要点 |
|------|----------|
| 语义基础 | 24 组枚举；含非零虚部；LEFT/RIGHT 非方阵；T 与 C 区分 |
| 形状 / 对齐 | 0、1、质数、2 的幂及 ±1、lda/ldb padding；padding 哨兵不被改写 |
| 批次 | batchCount 1～1024；非相邻矩阵地址；batch 间数据隔离 |
| 标量 | alpha=0/1/−1/纯虚/一般复数 |
| 负向 | 空 handle/alpha/A/B、非法枚举、负 m/n、非法 lda/ldb、batchCount≤0、指针表元素为空 |
| 特殊值 | 被引用 Inf/NaN；UNIT 对角 / 未引用三角上的哨兵 |
| 性能 | 任务书 5 case + 配套性能 CSV |

## 兼容性分析

新算子，不涉及兼容性分析。接口声明放入 `include/cann_ops_blas.h`，禁止定义产品私有平行 API。相对 cuBLAS：本任务仅 Host alpha，且 `batchCount < 1` 返回 `INVALID_VALUE`。

## 参考资料

1. 任务书：`aclblasCtrsmBatched_A2A3_task_doc.md`  
2. 设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`  
3. 设计文档 CheckList：https://docs.qq.com/sheet/DUHVGUFdmSFRjVFFU?tab=000001  
4. ops-blas：https://gitcode.com/cann/ops-blas  
5. 生态精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md  
6. cuBLAS trsmBatched / Netlib `ctrsm.f`
