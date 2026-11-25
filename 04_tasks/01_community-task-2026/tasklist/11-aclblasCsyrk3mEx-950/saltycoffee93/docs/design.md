# aclblasCsyrk3mEx 算子设计文档

> 目标芯片：Ascend 950PR（arch35，SOC=ascend950）｜CANN 9.1.0+｜合入位置 `ops-blas/blas/syrk/arch35/`
> 任务书：`aclblasCsyrk3mEx_Atlas950PR_task_doc.md`（下文 §n 均指该任务书章节）
> 仓内现状基线：ops-blas master `55ef791`（2026-11-25 源码核验）。本文撰写时未上板，凡需 NPU 实测的结论均标注"待实测"，不含未经测试的性能数字。

# 需求背景

## 需求来源

- 任务名称/批次：11 月社区任务 —— aclblasCsyrk3mEx 算子开发（950）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → 实现 `blas/syrk/arch35/`；测试 `test/syrk/csyrk3m_ex/arch35/`（任务书 §2.2、§5）。
- 验收流程：设计文档以 PR 形式提交至 cann-competitions 仓 tasklist，评审合入后进入开发与验收（任务书 §4）。

## 背景介绍

### 算子功能定位

`aclblasCsyrk3mEx` 实现单精度复数（complex64）**对称**矩阵秩-k 更新（3M 扩展版）：`C = alpha * op(A) * op(A)^T + beta * C`，仅读写 uplo 指定三角，对角虚部为有效数据（不置 0）。

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasCsyrk3mEx | cuBLAS `cublasCsyrk3mEx`｜Netlib `csyrk.f`（数学语义） | 对称秩-k 更新，3M 分解，A 支持 COMPLEX64/INT8 两种输入 |

与本仓既有接口的边界：非 Ex 的 `aclblasCsyrk`（`blas/syrk/arch35/csyrk_host.cpp`）是直接复数乘路径，可作参考但接口独立；`aclblasCsyrkEx` 是 1125 新增的另一个独立接口，本任务**不得调用或依赖**（§2.2）。

### 业界对标

| 项 | cuBLAS cublasCsyrk3mEx | 本算子 |
|---|---|---|
| trans 定义域 | N/T/C，对称语义下 OP_C 与 OP_T 等价（不共轭） | 同：{OP_N, OP_T, OP_C}，OP_C 按 OP_T 处理（§2.1） |
| 对角虚部 | 有效数据，不置 0 | 同（与 Hermitian 族 cherk 的关键差异，§2.1 第 3 条） |
| alpha/beta | 复数标量 | 同（COMPLEX64，Host/Device 指针位置均支持，§2.5） |
| Atype/Ctype | CUDA_C_8I / CUDA_C_32F 组合 | `ACL_INT8`（复元素按 int8 实/虚交错 2 字节）/ `ACL_COMPLEX64`；其余 → `NOT_SUPPORTED` |
| 非法 uplo/trans 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| 数值口径 | 3M 舍入 | 同为 3M 舍入，与 `cblas_csyrk`（非 3M）不保证逐位一致，按 §3.2 阈值判定 |

### 工程形态

- 工程模式：Kernel 直调（Host C API 校验/分派/tiling，经 handle 绑定的 stream 直调 NPU kernel，§2.2）。
- 仓内现状（master `55ef791` 源码核验）：

| 内容 | 现状 | 证据 | 本任务工作 |
|---|---|---|---|
| 公开声明 | **不存在** `aclblasCsyrk3mEx` | `include/cann_ops_blas.h` L792 为 `aclblasCsyrkx`、L809 为 `aclblasCsyrk` | **新增声明** |
| arch35 实现 | 仅非 Ex `csyrk`/`ssyrk`（合 4198 行） | `blas/syrk/arch35/csyrk_*`、`ssyrk_*` | **新增** `csyrk3m_ex_*` |
| 测试工程 | 仅 `test/syrk/csyrk/`、`test/syrk/ssyrk/` | 目录清单核验 | 新建 `test/syrk/csyrk3m_ex/` 并注册目标 |
| README | 无 Csyrk3mEx 章节 | `blas/syrk/README.md` L279 仅 `### aclblasCsyrk` | 新增章节 |

- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/syrk/arch35/csyrk_*`（非 Ex） | ★ 主模板 | 参数校验顺序、uplo 三角处理、Phase 0 拆实虚（Ar/Ai/AiNeg，`csyrk_kernel.cpp` 文件头 L11-19）/ Phase 2 复数 alpha/beta 组装的 AIV 结构、alpha=0/k=0 短路 | 乘积从"4 次实数 GEMM"换成"3M 对称专用 3 次实数乘（单 launch）"；**对角虚部保留**（非置 0） |
| `blas/gemm/arch35/cgemm3m_strided_batched_kernel.cpp` | ◆ 参考 | **单 launch 内三累加的 3M SIMT 直接法**、第二操作数转置读取 | 对称分解第二操作数按 X^T 读（不取共轭），虚部符号与 Hermitian 不同 |
| `blas/gemm/arch35/cgemm3m_rounding_kernel.cpp` | ◇ 参考 | 3M 舍入风险筛查/纠正思想 | 按需借鉴 |

### 接口规格

`aclblasCsyrk3mEx` 为新增接口，需在 `include/cann_ops_blas.h` 新增声明（紧随同族 `aclblasCsyrk` 之后），与 cuBLAS `cublasCsyrk3mEx` 逐参数对应（§2.3）：

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

参数规格（逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 句柄，携带 stream | scalar | — | 非空；空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 标量 |
| uplo | C 的存储三角 | attr | int | UPPER / LOWER；其它 → `INVALID_ENUM` | — |
| trans | op(A) 运算类型 | attr | int | {N, T, C}；C 与 T 等价不共轭；其它 → `INVALID_ENUM` | — |
| n | 方阵 C 的阶数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | — |
| k | 秩-k 的 k 维度 | attr | int | ≥0；负 → `INVALID_VALUE`；0 时秩-k 贡献为 0 | — |
| alpha | 复数标量乘数 | scalar | COMPLEX64 | 非空；空 → `INVALID_VALUE`；Host/Device 指针位置均支持 | 标量 |
| A | 输入矩阵（列主序） | tensor | COMPLEX64 / COMPLEX8I | n>0 且 k>0 时非空；INT8 为 (int8 real, int8 imag) 交错 2 字节 | n×k（N）或 k×n（T/C） |
| Atype | A 的类型枚举 | attr | int | {ACL_COMPLEX64, ACL_INT8}；其它 → `NOT_SUPPORTED` | — |
| lda | A 前导维 | attr | int | N：≥max(1,n)；T/C：≥max(1,k)；否则 `INVALID_VALUE` | 标量 |
| beta | 复数标量乘数 | scalar | COMPLEX64 | 非空；=(0,0) 不读 C 旧值 | 标量 |
| C | 输出对称矩阵（原地覆写） | tensor | COMPLEX64 | n>0 时非空；仅 uplo 三角被读写；对角虚部保留 | 逻辑 n×n；物理 ldc×n |
| Ctype | C 的类型枚举 | attr | int | 仅 ACL_COMPLEX64；其它 → `NOT_SUPPORTED` | — |
| ldc | C 前导维 | attr | int | ≥max(1,n)；否则 `INVALID_VALUE` | 标量 |

### 数学公式

对称专用 3M 分解（§2.1 第 7 条原文；乘积是 X·X^T，**不得套用 Hermitian 的 X·X^H 分解**）：

```text
X = op(A) = Xr + i·Xi（Atype=ACL_INT8 时 int8 实/虚先精确提升为 float）
非融合：P0 = Xr·Xr^T，P1 = Xi·Xi^T，P2 = (Xr+Xi)·(Xr+Xi)^T
        Re = P0 − P1，Im = P2 − P0 − P1
C = alpha·(Re + i·Im) + beta·C_old，仅写 uplo 三角，对角虚部保留
```

三次实数矩阵乘必须在**同一次 kernel launch** 内完成（三路累加），不得用三次独立 GEMM 调用拼接。3M 的 FP32 舍入与直接复数乘不同，与 `cblas_csyrk` 不保证逐位一致，按 §3.2 阈值判定。

# 需求分析

## 需求描述

用 Ascend C 在昇腾 950PR（arch35）上实现句柄式接口 `aclblasCsyrk3mEx`：complex64 对称秩-k 更新，3M 分解单 launch 完成，支持 uplo×trans(N/T/C) 组合与 Atype=COMPLEX64/INT8 双输入；精度达生态算子开源混合容差标准 FLOAT32 分量档，性能达 GPU 标杆 0.4×/0.5× 双条件（kernel-to-kernel）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 对称 3M：P0=XrXr^T、P1=XiXi^T、P2=(Xr+Xi)(Xr+Xi)^T，Re=P0−P1、Im=P2−P0−P1。非 Ex csyrk 的"拆实虚 → 实数 GEMM → 组装"骨架可平移（其 Phase 0 已产出 Ar/Ai/AiNeg 三平面），乘积数 4→3；对角虚部保留为有效数据 |
| 2 | 接口原型 | §2.3 签名（14 参数，含 Atype/Ctype），返回 `aclblasStatus_t`；**公共头需新增声明** |
| 3 | 参数规格 | 见"接口规格"参数表；异常行为逐条对齐 §2.4（含 NOT_SUPPORTED 组合） |
| 4 | 支持数据类型 | C/输出恒 COMPLEX64；A 为 COMPLEX64 或 INT8（复 int8 交错存储，精确提升）；alpha/beta 为复数 COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | 昇腾 950PR → arch35（`--soc=ascend950`）；仅 950PR，无 arch22 要求 |
| 6 | 精度要求 | §3.2：FLOAT32 分量档 rtol=atol=2^-13、ratio≥0.99、max_abs_error≤1e-2 或 32ULP；golden=`cblas_csyrk`（INT8 先精确提升；float64 可用 `cblas_zsyrk`），3M 舍入不保证逐位一致 |
| 7 | 性能预期 | §3.3：**双条件**——单 case NPU ≤ 2.5×`gpu_ms`（≥0.4×）且全部 case 平均倍率 ≥0.5；口径 `msprof op` Task Duration，warmup 5 + 10 样本；int8 性能沿用 COMPLEX64 基线自行补采 |
| 8 | 代码架构选型 + 可行性 | 单 launch 融合 3M kernel（硬约束，§2.1 第 7 条）：SIMT 直接法起步（三累加结构照 `cgemm3m_strided_batched_kernel`，正确性优先），大规模 Cube 融合变体为性能备选。**已排除**：三次独立 GEMM 调用拼接（明令禁止）、Hermitian X·X^H 分解（虚部符号错误）、依赖未合入的 `aclblasCsyrkEx` |
| 9 | 评测来源 | 任务包 `test_cases/`：精度 1000 条（含 `TC_3M_*` 专项 4 条）+ 性能 200 条（全 COMPLEX64，`gpu_baseline.csv` 全回填） |

# 详细设计

## 算子分析

### 数学公式

见"接口规格 · 数学公式"节。实现要点：三次实数乘 P0、P1、P2 在**同一次 kernel launch** 内以三路累加同时推进（对每个输出元素、沿 K 一遍循环同时累计 `p += Xr·Xr^T、q += Xi·Xi^T、r += (Xr+Xi)·(Xr+Xi)^T` 的三个部分和），输出 `Re = p − q`、`Im = r − p − q`——三累加结构与 `cgemm3m_strided_batched_kernel.cpp` L29-40 同构，第二操作数按**转置读取（不取共轭）**。

### 支持数据类型

| 项 | 类型 | 说明 |
|---|---|---|
| A | COMPLEX64 或 COMPLEX8I（int8 实/虚交错 2 字节） | INT8 在 kernel 内精确提升为 float（无损） |
| C | COMPLEX64 | 仅 uplo 三角被读写，**对角虚部保留**（对称语义） |
| alpha/beta | COMPLEX64，Host 内存 | 复数组装的缩放系数（与非 Ex csyrk 的 Phase 2 同口径） |
| 计算精度 | FP32 | 950PR SIMT/Cube 均为 FP32 计算 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| n | ≥0 | n=0：结构校验通过后 SUCCESS，不引用 A/C、不启动 kernel |
| k | ≥0 | k=0 或 alpha=0：秩-k 贡献为 0；beta=1 quick return，beta≠1 仅做 uplo 三角 `C = beta*C`（不引用 A） |
| lda/ldc | 见参数表下界 | 允许 padding；低于下界 → INVALID_VALUE |
| trans | N/T/C（C≡T） | 非法枚举 → INVALID_ENUM |
| Atype/Ctype | 支持集见参数表 | 非支持组合 → NOT_SUPPORTED |

仅写 uplo 三角；另一半由 C = C^T 隐含，不写入。C 与 A 存储不得重叠（调用方保证）。

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（变体/触发条件/职责）：

| 变体 | 触发条件 | 职责 | kernel 数 |
|---|---|---|---|
| 融合 3M 主 kernel（SIMT 直接法） | 全尺寸通用 | 单 launch 内三路累加完成 X·X^T，第二操作数转置读取（C≡T 不共轭）；复数 alpha/beta 内联；仅写 uplo 三角、对角虚部保留 | 1 |
| beta 缩放 kernel | k=0 或 alpha=0 且 beta≠(1,0) | 仅 uplo 三角 `C = beta*C`（复数 beta） | 1 |
| quick return | n=0；或 alpha=0 且 beta=(1,0) | host 直接返回，不发射 | 0 |

数据流（融合主 kernel，Atype 双类型）：

```text
GM: A(COMPLEX64 或 int8x2)──[SIMT 单 kernel 单 launch]──> 每个输出元素(i,j)：
      沿 K 累加：p += Xr(i,l)·Xr(j,l)，q += Xi(i,l)·Xi(j,l)，r += (Xr+Xi)(i,l)·(Xr+Xi)(j,l)
      [INT8：读入 (int8,int8) 精确转 float 后参与同一循环]
      Re = p−q，Im = r−p−q
GM: C_old ─[AIV 组装：C = alpha·(Re + i·Im) + beta·C_old，仅 uplo 三角掩码写回，对角虚部保留]─> GM: C
```

实现组织可二选一（实现期按 950PR 实测裁决）：a) 三累加全部在 SIMT 主 kernel 内完成（组装内联，单 launch 最彻底）；b) 参照非 Ex csyrk 的 Phase 结构，Phase 0 拆实虚后由 Cube 侧单 launch 完成三乘积、Phase 2 组装——两者均满足"同一次 launch"约束。workspace：方案 a 零中间平面；方案 b 需拆实虚平面（Ar/Ai/AiSum，非 Ex csyrk 已有同款 Phase 0）与乘积平面。

#### host 侧设计

**分核策略**：SIMT 直接法按 950PR 核数对输出 uplo 三角分块（每核若干列带或行带，三角内列首尾裁剪）；若引入 Cube 变体，则按非 Ex csyrk 的满核分块模式。

**数据分块和内存优化策略**：直读原位 A（含 lda padding 的列主序访问）；INT8 数据量减半，读带宽收益天然存在；uplo 三角内输出块按对称性无重复计算。

**tilingkey 规划策略**：仓内 arch35 syrk/gemm 族无 tilingkey 宏分支，host 显式选择 kernel。本算子的分派字段：

| 条件 | 分支 |
|---|---|
| n=0 | quick return SUCCESS |
| k=0 ∨ alpha=0 | beta=(1,0) quick return；否则 beta 缩放 kernel |
| trans=N/T/C | N 直读；T/C 转置读取（C 不取共轭），经 tiling 字段传入 |
| Atype | COMPLEX64 直读 / INT8 读时提升 |
| uplo | UPPER/LOWER 三角掩码 |

**数据检测**（校验顺序逐条对齐 §2.4/§2.5）：

1. handle 非空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. uplo ∈ {UPPER, LOWER} → 否则 `INVALID_ENUM`
3. trans ∈ {N, T, C} → 否则 `INVALID_ENUM`
4. n ≥ 0、k ≥ 0 → 否则 `INVALID_VALUE`
5. lda ≥ max(1, trans=N ? n : k)、ldc ≥ max(1, n) → 否则 `INVALID_VALUE`
6. alpha、beta 非空 → 否则 `INVALID_VALUE`
7. Atype ∈ {ACL_COMPLEX64, ACL_INT8}、Ctype = ACL_COMPLEX64 → 否则 `NOT_SUPPORTED`
8. n>0 且 k>0 时 A 非空、n>0 时 C 非空 → 否则 `INVALID_VALUE`
9. 结构合法后 quick return：n=0 → SUCCESS（A/C 可空）；k=0 或 alpha=0 → A 可空，beta≠(1,0) 时仍执行 uplo 三角 `C = beta*C`

#### kernel 侧设计

```text
[融合 3M SIMT kernel]
 Init   ：解析 tiling（n,k,lda,ldc,trans,uplo,Atype,alpha,beta 内联）→ 求本核三角分区
 Process：按输出列带遍历 uplo 三角 → 逐元素三路 K 累加（INT8 读时提升）
          → 复数 alpha/beta 组装 → uplo 掩码写回（对角虚部原样保留）
[beta 缩放 kernel]
 Process：uplo 三角逐列 C = beta*C（复数 beta；beta=0 清零，不读旧值）
```

文件落位：

| 文件 | 内容 |
|---|---|
| `include/cann_ops_blas.h` | **新增** `aclblasCsyrk3mEx` 声明（任务书 §2.3 授权） |
| `blas/syrk/arch35/csyrk3m_ex_host.cpp` | 公开入口：校验、分派、标量读回（Host/Device 双位置）、发射 |
| `blas/syrk/arch35/csyrk3m_ex_kernel.cpp/.h`、`csyrk3m_ex_tiling_data.h` | 融合 3M kernel 与 beta 缩放 kernel |
| `test/syrk/csyrk3m_ex/`（顶层 golden/param/verify）+ `test/syrk/csyrk3m_ex/arch35/` | 测试工程（对齐 `test/syrk/csyrk/` 组织），注册 CMake 目标 `csyrk3m_ex_test` |

#### 性能优化要点

本文未上板，以下为设计依据与规划，实测数据在性能阶段以 `msprof op` 回填。由 `gpu_baseline.csv` 算出的 NPU 允许耗时（双条件较宽：单 case ≤ `gpu_ms`/0.4 且平均倍率 ≥0.5；括号内为 `gpu_ms`，全部 COMPLEX64）：

| 代表 case | uplo/trans | n×k | 允许耗时 |
|---|---|---|---|
| csyrk3m_ex-perf-195 | UPPER/T | 1648×3668 | 3.92 ms（1.57） |
| csyrk3m_ex-perf-196 | UPPER/N | 3006×842 | 2.83 ms（1.13） |
| csyrk3m_ex-perf-198 | LOWER/N | 1490×2355 | 2.16 ms（0.87） |
| csyrk3m_ex-perf-199 | LOWER/T | 714×1582 | 0.48 ms（0.19） |

| 优化项 | 依据 | 状态 |
|---|---|---|
| SIMT 直接法单 launch（3 累加一次过） | 天然满足"同一次 launch"硬约束；3M 相对非 Ex csyrk 的 4 次实数乘少 25% 乘加量 | 首版实现（正确性优先） |
| 零 workspace 直读 | 无拆实虚平面与乘积平面的 GM 往返 | 首版实现 |
| INT8 读时提升 | int8 数据量为 COMPLEX64 的 1/4，读带宽收益；精确无损 | 首版实现 |
| 大 n 的 Cube 融合变体（Phase 0 + 3 组 mmad 同 launch + Phase 2） | 若 SIMT 在大 n（3006+）不满足 0.4×，改 Cube 主算力 | 备选，S5 按实测裁决 |
| 未采用：三次独立 GEMM 调用拼接 | 任务书 §2.1 第 7 条明令禁止 | 排除 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| 昇腾 950PR（arch35，`--soc=ascend950`） | ✓ 本任务目标；自验证覆盖 950PR（§3.1） |
| Atlas A2/A3（arch22） | 不在本任务范围 |
| Ascend 950DT（arch35） | 按仓内惯例由同 arch 实现覆盖，本任务不单独验证 |

## 算子约束限制

逐条对齐 §2.5：

- 参数合法性与校验层级：host 入口按上表顺序校验；Atype=INT8 按交错 (int8, int8) 解释；不对 Device 张量做元素级扫描；对称性由调用方保证（对角虚部为有效数据，计算后原样保留）。
- 非连续 Tensor：不支持，仅 lda/ldc 描述的列主序带填充布局；不使用 AutoContiguous/ViewCopy。
- broadcast：不支持。
- dynamic shape：不要求，n、k 为运行时入参。
- 原地与视图语义：C 原地覆写，仅 uplo 三角读写；C 可写区域与 A 可读区域不得重叠。
- 确定性计算：不要求；3M 路径与 K 归约顺序由实现按形状选择。
- 空 Tensor：n=0 合法 no-op；k=0 或 alpha=0 时 A 可空，beta≠(1,0) 仍做 `C = beta*C`。
- 0 维 Tensor：不涉及；标量支持 Host 与 Device 两种指针位置。
- 异步执行：kernel 在 handle 绑定的 stream 上异步执行。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度逐元素与用例判定 | `\|actual − golden\| ≤ atol + rtol·\|golden\|` 的实部/虚部分量记为 matched；用例须同时满足 matched_ratio ≥ 0.99 与 max_abs_error ≤ 限值。按 FLOAT32 分量档：rtol=atol=2^-13（1.2e-4），max_abs_error 限值 1e-2 或 32 ULP | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md`（pin `28a1ce52`） |
| 精度 golden | `cblas_csyrk`（Netlib `csyrk`，非 3M 直接复数乘）；`Atype=ACL_INT8` 时 int8 实/虚部先精确提升为 float 再调用；float64 参考用 `cblas_zsyrk`。3M 舍入不保证逐位一致，按阈值判定 | 任务书 §3.1、§3.2 第 1 条与 golden 精度域说明 |
| 性能 | 双条件：单 case NPU ≤ 2.5×`gpu_ms`（≥0.4×）**且**全部 case 加速比算术平均 ≥ 0.5 | 任务书 §3.3.2 |
| 性能口径 | `msprof op` 的 kernel Task Duration（正式判据）：warmup 5 次不计入，之后 10 次有效样本取算术平均，除以 1000 转 ms 后与 `gpu_ms`（GPU kernel-only，nsys）比较；GTest `[PERF] avg_us` 仅作交叉验证。int8 性能沿用 COMPLEX64 基线由开发者补采 | 任务书 §3.3、§3.3.1、§3.3.2 |
| 内存 | 不涉及 | 任务书 §3.4 |

## 测试概览

用例来自任务包 `test_cases/csyrk3m_ex_test.csv`（固定随机种子，逐条可复现），精度 1000 条 + 性能 200 条：

| 类别 | 前缀 | 条数 | 覆盖 |
|---|---|---|---|
| L0 基础 | TC_L0 | 8 | uplo × trans(N/T/C) × 小尺寸 |
| 方阵尺寸 | TC_SQ | 136 | 尺寸池方阵（含 4095/4096） |
| 三角/转置键 | TC_TK / TC_BK | 12 + 12 | uplo×trans 组合与块三角场景 |
| 标量 | TC_AB | 32 | 复数 alpha（(0,0)/(1,0)/(0,1)/±1e4 与随机值）、beta（(0,0)/(1,0) 与随机值） |
| 前导维 | TC_LD | 18 | 紧凑与 +1/+8/+64 padding |
| 填充值域 | TC_FL | 12 | 均匀/正态/交替/极端/Inf/NaN |
| 边界负向 | TC_ED | 38 | n/k=0、alpha=0+A NaN、beta=0+C NaN、空指针、非法枚举、Atype/Ctype 不支持、负维度 |
| 3M 专项 | TC_3M | 4 | 交替/极值填充下的实/虚部抵消、int8 极值、对角虚部保留 |
| 扩展精度 | TC_EX | 696 | 尺寸 × uplo × trans × Atype × 标量采样 |
| 性能 | TC_PF | 200 | 全 COMPLEX64；uplo×trans 组合与规模分档；含与典型 case 同参条目 |

- 测试工程：新建 `test/syrk/csyrk3m_ex/`（顶层 golden/param/verify 对齐 `test/syrk/csyrk/` 组织）与 `test/syrk/csyrk3m_ex/arch35/`，注册 CMake 目标 `csyrk3m_ex_test`。
- 复现入口：`bash build.sh --ops=csyrk3m_ex --soc=ascend950` → `build/test/syrk/csyrk3m_ex/csyrk3m_ex_test`；任务包 `verify_accuracy.py` / `verify_performance.py` 将 CSV 安装到 `test/syrk/csyrk3m_ex/<arch>/csyrk3m_ex_test.csv` 后加载执行。
- 负向与专项重点：对角虚部非零输入（验证保留不置 0）、`alpha=0` 且 A 为 NaN（不引用 A）、`beta=0` 且 C 为 NaN（不读旧 C）、`TC_3M_*` 实/虚部抵消与 int8 极值。

## 兼容性分析

- 公共头 `include/cann_ops_blas.h` **新增** `aclblasCsyrk3mEx` 一条声明（任务书 §2.3 明确"目前不存在，需新增"），不改动既有声明与 ABI。
- 新增文件仅 `blas/syrk/arch35/csyrk3m_ex_*` 与 `test/syrk/csyrk3m_ex/`；已合入的非 Ex `csyrk`/`ssyrk` 实现与测试零改动、零回归风险。
- 与 1125 新增的 `aclblasCsyrkEx`（独立接口）互不依赖、代码不交叉（任务书 §2.2）。
- `blas/syrk/README.md` 新增 `aclblasCsyrk3mEx` 章节按仓内规范补充算子说明与产品支持情况。
- Atype/Ctype 的 `NOT_SUPPORTED` 返回码为新增组合语义，仅作用于本接口，不影响其他算子。

