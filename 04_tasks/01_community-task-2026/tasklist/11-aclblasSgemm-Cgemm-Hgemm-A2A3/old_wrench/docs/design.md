# aclblasSgemm / aclblasCgemm / aclblasHgemm 算子设计文档

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：S = FLOAT32；C = COMPLEX64（实/虚部各 FP32）；H = FLOAT16（`aclblasHalf`，计算域 FP32）
> 合入位置：`ops-blas/blas/gemm/arch22/`；测试：`ops-blas/test/gemm/arch22/`（扩展既有通用 gemm_test 工程，不新建子目录）
> 任务书：`aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——gemm 类 3 个算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemm/arch22/`（§1/§5）；测试代码 → `test/gemm/arch22/`（扩展既有 `gemm_test.cpp` / `gemm_test.csv`，§2.2 明令不新建 `test/gemm/sgemm/arch22/`）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 公共头现状（grep 实证）：`aclblasSgemm`（L575）与 `aclblasCgemm`（L579）**声明已存在**；`aclblasHgemm`（非 batch）**不存在，需新增声明**（建议紧随 aclblasCgemm 之后，同族风格）。
- 仓内实现现状（源码级核验，2026-11-25 快照）：`aclblasSgemm` arch22 已在 `gemm_host.cpp` L459 导出（含 gemm_kernel.cpp 多形态基建）；`aclblasCgemm` arch22 已于 2026-11-24 合入（`cgemm_host.cpp` L517，拆平面→子句柄 SgemmEx 四积→combine）。**本任务的实质增量 = aclblasHgemm（新增声明 + arch22 实现）**；本设计按任务书三接口契约全覆盖，并逐一对齐仓内现状。

## 背景介绍

### 算子功能定位

单矩阵通用矩阵乘（非批量），三 dtype 变体：

```
C = alpha * op(A) * op(B) + beta * C
```

| 接口 | 对标 | dtype（A/B/C/alpha/beta） | 作用 |
|---|---|---|---|
| aclblasSgemm | cuBLAS `cublasSgemm`｜Netlib `sgemm` | FLOAT32 | 单精度 GEMM |
| aclblasCgemm | cuBLAS `cublasCgemm`｜Netlib `cgemm` | COMPLEX64 | 复数 GEMM（**直接复数乘，非 3M**） |
| aclblasHgemm | cuBLAS `cublasHgemm`（Netlib 无 half 参考） | FLOAT16（aclblasHalf） | 半精度 GEMM（FP32 计算域） |

### 业界对标

| 项 | cuBLAS 对应变体 | 本算子 |
|---|---|---|
| 参数序列 | handle/transa/transb/m/n/k/alpha/A/lda/B/ldb/beta/C/ldc | 逐参数一致（§2.3） |
| OP_C 语义 | S/H：≡T；C：共轭转置（≠T） | 同（§2.1——Cgemm 须以非对称复矩阵用例区分 C/T） |
| 布局 | 列主序 + lda/ldb/ldc | 同 |
| H 标量 | alpha/beta 为 half | 同（`aclblasHalf`） |
| 非法 trans 返回码 | CUBLAS_STATUS_INVALID_VALUE | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| Cgemm 与 Cgemm3m | 两个独立接口（后者 3M 分解） | 同——本任务 Cgemm 按 `(a+bi)(c+di)=(ac−bd)+(ad+bc)i` 直接计算，**禁止混入 3M 文件**（§2.2 补充说明） |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口，handle 绑定 stream 异步执行，§2.2）。
- 代码落位：`blas/gemm/arch22/`——`gemm_host.cpp` 增加入口（Cgemm/Hgemm；Cgemm 已由 `cgemm_host.cpp` 就位）、`gemm_kernel.cpp/.h` 增加 dtype kernel 路径（或按仓内规范拆分文件，§2.2）。
- 公共头：新增 `aclblasHgemm` 一条声明（仅增不改）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemm/arch22/gemm_{host,kernel}.cpp`（Sgemm 已导出） | ★ 主基线 | L459 入口骨架（校验顺序/条件判空/特值短路/tiling）+ 62KB kernel 基建（rows/tiles/wholegrid + pack/accumulate/epilogue/zero 多形态） | Hgemm 以 half 模板实例叠加（FP32 累加器 + cast 进出） |
| `blas/gemm/arch22/cgemm_*`（11-24 合入） | ★ 已就位 | Cgemm 复数四积三段流水（拆平面→子句柄 SgemmEx→combine） | 本设计仅契约核对（直接复数乘 ✓ 非 3M ✓ 条件判空口径核对） |
| `blas/gemm/arch35/{cgemm_host, gemm_host, hgemmStridedBatched_*}.cpp` | ◆ 任务书指定参考 | dtype 分派 / 复数路径 / **FP16 计算路径**（hgemmStridedBatched） | arch22 重写发射层（SDK 兼容） |
| `test/gemm/arch22/` + `gemm_{golden,param}.h` | 既有 | 通用 gemm_test 工程 | 扩展 dtype/op 列，不新建子目录 |

### 接口规格

签名与任务书 §2.3 一致（S/C = L575/L579 既有声明；H = 新增声明）：

```cpp
aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc);

aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);

// 新增声明（aclblasHalf 为 FP16 16-bit 位模式类型）
aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, const aclblasHalf* B, int ldb,
    const aclblasHalf* beta, aclblasHalf* C, int ldc);
```

参数规格（官方固定表，逐行对齐 §2.4；三算子同构，dtype 列区分）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| transa/transb | op(A)/op(B) 运算类型 | attr | int | {N,T,C}；S/H 中 C≡T，Cgemm 中 C=共轭转置；其它 → `INVALID_ENUM` | 标量 |
| m / n / k | GEMM 维数 | attr | int | ≥0；负 → `INVALID_VALUE`；m/n=0 合法 no-op；k=0 只做 C=beta·C | 标量 |
| alpha | 标量乘数（Host） | scalar | S=FLOAT32；C=COMPLEX64；H=FLOAT16 | 任意有限值（FP16 可表示域）；nullptr → `INVALID_VALUE` | 标量 |
| A | 输入矩阵（列主序，Device） | tensor | 同 alpha | m>0∧n>0∧k>0∧alpha≠0 时 nullptr → `INVALID_VALUE` | N: lda×k；T/C: lda×m |
| lda | A 前导维 | attr | int | N: ≥max(1,m)；T/C: ≥max(1,k)；低于 → `INVALID_VALUE` | 标量 |
| B | 输入矩阵（列主序，Device） | tensor | 同 alpha | 同 A 条件 | N: ldb×n；T/C: ldb×k |
| ldb | B 前导维 | attr | int | N: ≥max(1,k)；T/C: ≥max(1,n)；低于 → `INVALID_VALUE` | 标量 |
| beta | 标量乘数（Host）；beta=0 不读 C | scalar | 同 alpha | 同 alpha | 标量 |
| C | 输出矩阵（列主序，原地覆写，Device） | tensor | 同 alpha | m>0∧n>0 时 nullptr → `INVALID_VALUE`；仅写逻辑 m×n | ldc×n |
| ldc | C 前导维 | attr | int | ≥max(1,m)；低于 → `INVALID_VALUE` | 标量 |

### 数学公式

```
C[r,c] = alpha · Σ_{t=0..k-1} op(A)[r,t]·op(B)[t,c] + beta · C[r,c]
S/H（实数）：直接乘加；op = N → 原矩阵，T/C → 转置（无共轭）
C（复数）：(a+bi)(c+di) = (ac−bd) + (ad+bc)i 直接展开；OP_C 的共轭折入 op(A)/op(B) 的虚部取反
H 数据通路：FP16 输入/标量 → FP32 → FP32 乘加累加 → IEEE round-to-nearest 收窄 FP16 写 C
```

仅写 C 逻辑 m×n 区域（ldc padding 列不动）。特值：alpha=0 或 k=0 → 仅 `C = beta·C`（不引用 A/B；beta=0 → 直写 0 不读 C）；m/n=0 → no-op（A/B/C 可 null）。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上提供三个句柄式单矩阵 GEMM 接口：Sgemm（FP32，仓内既有，契约核对）、Cgemm（COMPLEX64 直接复数乘，11-24 已合入，契约核对）、**Hgemm（FP16 全链 + FP32 计算域，本任务实质增量：新增公共头声明 + arch22 实现）**；支持 transa×transb 9 组合（Cgemm 的 OP_C 共轭语义独立）、任意合法前导维、quick return 与特值短路；精度 S/C 按 FP32 / H 按 FP16 达生态混合容差，性能达 GPU 标杆 0.8×（单 case 判定）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 三接口同构 GEMM；S 的 arch22 多形态基建（rows/tiles/wholegrid）可直接模板化给 H；C 的四积三段流水已就位（直接复数乘语义核对无误） |
| 2 | 接口原型 | §2.3 三签名（S=L575、C=L579 既有；H 新增声明插在 Cgemm 后），零既有声明修改 |
| 3 | 参数规格 | 见"接口规格"表；条件判空逐条对齐 §2.4 |
| 4 | 支持数据类型 | S=FP32；C=COMPLEX64；H=FP16（含 alpha/beta），H 计算域 FP32 |
| 5 | 目标芯片与 arch 映射 | A2/A3（910b 系）→ arch22；构建 `--ops=sgemm,cgemm,hgemm --soc=ascend910b3`；arch35 参考实现不动 |
| 6 | 精度要求 | §3.2：S/C 按 FP32 分量（rtol=atol=2^-13，max_abs 1e-2/32ULP；golden=cblas_sgemm/cblas_cgemm，FP64 累加收窄）；H 按 FP16（2^-10/1e-1；golden=FP32（FP64 累加）→FP32→FP16 双段收窄，不用 cblas_hgemm） |
| 7 | 性能预期 | §3.3：单 case NPU Task Duration ≤1.25×gpu_ms（≥0.8× GPU）；gpu_baseline 实测 S max=8.8202ms/avg=0.3548ms（67 条）、C max=29.1271ms/avg=1.2504ms（67 条）、H max=1.6786ms/avg=0.0703ms（66 条） |
| 8 | 代码架构选型 + 可行性 | H 主路径 = gemm 基建增加 `<half>` 模板实例（FP32 累加器 + cast 进出）；备选 = cast 复合路径（A/B 预 cast FP32 → 子句柄 SgemmEx(ACL_FLOAT) → C 收窄回写）；已排除死路：FP16 原生累加（精度漂移，golden 为 FP32 域）、混入 3M 文件（§2.2 禁止）；可行性旁证 = S 基建与 arch35 hgemmStridedBatched FP16 路径均已合入 |
| 9 | 评测来源 | 任务书自验包 test_cases/{sgemm,cgemm,hgemm}/（csv 402/401/400 行 ≈ 精度 333/333/334 + 性能 67/67/66；gpu_baseline 各 67/67/66 条） |
| 10 | 特值与退化 | m/n=0 no-op；k=0 或 alpha=0 → 只做 C=beta·C（A/B 可 null；C 条件判空）；beta=0 不读 C（NaN 安全） |
| 11 | 错误码口径 | 非法 transa/transb → INVALID_ENUM；维度/ld/指针 → INVALID_VALUE |
| 12 | workspace | H 主路径无额外 workspace（核内 cast）；备选路径需 m·k+k·n+m·n FP32 暂存（size_t 防溢出）；S/C 既有路径不动 |

# 详细设计

## 算子分析

### 数学公式

逐元素（r=0..m-1，c=0..n-1）：

```
C[r,c] = alpha · Σ_{t=0..k-1} op(A)[r,t] · op(B)[t,c] + beta · C[r,c]
S：FP32 直接乘加
C：复数直接乘（非 3M）：Σ (Ar+i·Ai)(Br+i·Bi) → 分量 ac−bd / ad+bc；OP_C 的共轭 = 对应操作数虚部取反后再转置
H：FP16 存储 → FP32 域乘加 → IEEE RN 收窄 FP16（与 §3.2.4 golden 的 FP64 累加→FP32→FP16 口径在 FP32 域内一致）
```

op 定义：N → 原矩阵；T → 转置；C → Cgemm 中共轭转置（虚部取反 + 转置）、S/H 中 ≡T。特值：alpha=0 或 k=0 → 乘加项跳过（不引用 A/B）；beta=0 → C 旧值不读（NaN 安全）；m/n=0 → no-op。

### 支持数据类型

| 接口 | A/B/C 存储域 | alpha/beta | 计算域 | golden 判定档 |
|---|---|---|---|---|
| aclblasSgemm | FP32 | FP32（Host） | FP32 | FLOAT32 |
| aclblasCgemm | COMPLEX64 | COMPLEX64（Host） | FP32 分量 | FLOAT32 分量 |
| aclblasHgemm | FP16（aclblasHalf） | FP16（Host） | FP32（cast 进出） | FLOAT16 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m, n | ≥ 0 | 0 → SUCCESS no-op（A/B/C 可 null，不下发 kernel） |
| k | ≥ 0 | 0 → 仅 C=beta·C（A/B 可 null；C 条件判空） |
| lda/ldb/ldc | N: max(1,m)/max(1,k)/max(1,m)；T/C: 换位 | 低于下界 → INVALID_VALUE |
| 覆盖档位（§3.5） | m/n/k ∈ {0..4096 全档，含 4095³/4096³ 立方}；transa×transb 9 组合 | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

三接口统一挂 `blas/gemm/arch22/`，S 既有、C 已合入、H 为本任务增量：

```
aclblasSgemm  (gemm_host.cpp L459，既有)   ── FP32 基建：rows/tiles/wholegrid + pack/accumulate/epilogue/zero
aclblasCgemm  (cgemm_host.cpp L517，11-24 合入) ── 拆平面 → 子句柄 aclblasSgemmEx 四积 → combine
aclblasHgemm  (gemm_host.cpp 新增入口，本任务增量):
 └─ 校验（§2.5 顺序 + 条件判空，复用 Sgemm 骨架）→ quick return（m/n=0）
 ├─ 特值旁路: alpha=0 或 k=0 → C=beta·C kernel（beta=0 直写 0；beta=1 直接 SUCCESS）
 └─ 主路径: alpha/beta FP16→FP32 提升 → 实数 GEMM 基建 <half> 模板实例
        （A/B 面板 FP16 读入 → cast FP32 → FP32 乘加累加 → IEEE RN 收窄写 C；
          OP_C≡T 折算；ldc padding 哨兵）
 备选路径（模板侵入过大时）: A/B 预 cast 到 workspace FP32 → 子句柄 SgemmEx(ACL_FLOAT)
        → C 收窄回写（代价 = 额外 FP32 暂存带宽，大 shape 由 GEMM 主导）
```

kernel 变体划分表：

| 变体 | 触发条件 | 职责 |
|---|---|---|
| FP32 基建多 kernel（既有） | Sgemm 全形态；H 备选路径 | rows/tiles/wholegrid + pack/accumulate/epilogue/zero |
| `<half>` 模板实例（新增） | Hgemm 主路径 | 同上形态，面板 cast FP32、FP32 累加、收窄写回 |
| 拆平面/四积/combine（已合入） | Cgemm | 复数直接乘四积路径 |
| beta 缩放旁路 kernel | alpha=0 或 k=0（三接口共用语义） | C=beta·C（beta=0 写 0） |

#### host 侧设计

- **分核策略**：沿用 Sgemm 基建自适应（rows 形态小矩阵 / tiles 分块 / wholegrid 整阵 + K 分块归约）；H 模板实例继承同一 tiling 口径。
- **数据分块和内存优化策略**：H 主路径面板级 cast（UB 内完成，无 GM 暂存）；备选路径 workspace = (m·k + k·n + m·n)×FP32（size_t 乘加对防溢出）。
- **tilingkey 规划策略**：分支承载于 tiling 字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| dtype | 0/1/2 | S（float）/ H（half）/ 备选复合路径 |
| transa/transb | 折叠 {N,T}² | 4 形态（Cgemm 保留 C 独立符号） |
| alphaZero/betaZero/betaOne | 0/1 | 特值旁路 |

- **数据检测**（校验顺序逐条对齐 §2.5 注；结构合法后再 quick return，三接口同构）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | transa ∉ {N,T,C} | INVALID_ENUM |
| 3 | transb ∉ {N,T,C} | INVALID_ENUM |
| 4 | m/n/k < 0 | INVALID_VALUE |
| 5 | lda/ldb/ldc 低于对应下界 | INVALID_VALUE |
| 6 | alpha/beta == nullptr | INVALID_VALUE |
| 7 | m>0∧n>0∧k>0∧alpha≠0 时 A/B == nullptr | INVALID_VALUE |
| 8 | m>0∧n>0 时 C == nullptr | INVALID_VALUE |
| 9 | 全过后：m==0 或 n==0 → SUCCESS（不下发 kernel） | — |
| 10 | k==0 或 alpha==0 → beta==1 直接 SUCCESS；否则 beta 缩放旁路（不引用 A/B） | — |

#### kernel 侧设计

- H 主路径 `<half>` 实例：在既有 pack/accumulate/epilogue 流水上加 cast 节点——pack 阶段 A/B 面板 FP16→FP32；accumulate 阶段 FP32 乘加（累加器 FP32）；epilogue 阶段 alpha·P+beta·C（C 旧值 FP16 读入 cast）后 IEEE RN 收窄写回；ldc padding 列不触碰。
- 流程图（H 主路径，N 形态示例）：

```
GM A(FP16) ──pack+cast──> workspace/UB A'(FP32)
GM B(FP16) ──pack+cast──> workspace/UB B'(FP32)
A'·B' ──accumulate(K 分块 FP32 归约)──> P(FP32)
GM C(FP16) ──cast──> β·C_old ──> alpha·P + beta·C ──RN 收窄──> GM C(FP16)
```

- **workspace 布局表**：

| 路径 | 区 | 用途 |
|---|---|---|
| H 主路径 | 基建自管 pack/积暂存 | 与 Sgemm 相同布局（dtype 缩放后） |
| H 备选 | A'/B'/C' FP32 暂存 | cast 复合路径（m·k + k·n + m×n） |
| S/C | 既有 | 不动 |

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| H 复用 S 基建形态 | 模板实例化（同一 rows/tiles/wholegrid 自适应） | H 继承 FP32 基建已调优切分（待 msprof 回填） |
| 面板级 cast | FP16→FP32 在 pack 阶段合并 | 免独立 cast kernel 的额外整阵往返 |
| FP32 累加域 | 精度对齐 FP16 阈值 + FP32 golden | 免 FP16 累加漂移 |
| 已排除路线 | FP16 原生累加（精度）；混入 3M 文件（§2.2 禁止）；新建测试子目录（§2.2 禁止） | — |

判据逐 case 独立：NPU ≤ 1.25×gpu_ms；gpu_baseline S max=8.8202ms/avg=0.3548ms、C max=29.1271ms/avg=1.2504ms、H max=1.6786ms/avg=0.0703ms（cublas 同名接口 kernel-only）。达标以设备 msprof 实测为准（设计阶段不判决）。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列产品 | ✔（本任务目标，arch22） |
| Atlas A2 推理系列产品 | ✔（同 arch22） |
| Atlas A3 训练系列产品 | ✔（本任务目标，arch22） |
| Atlas A3 推理系列产品 | ✔（同 arch22） |
| Ascend 950PR / 950DT | （arch35 既有实现，本任务不涉及） |

## 算子约束限制

| 约束项 | 内容（§2.5 逐条） |
|---|---|
| 校验层级 | 结构参数（含 alpha/beta/A/B/C 指针）host 侧校验；不扫描 Device 数据 |
| 非连续支持 | 仅 lda/ldb/ldc 带填充布局；不用 AutoContiguous/ViewCopy |
| broadcast | 不支持 |
| dynamic shape | 天然支持（m/n/k/ld/trans 运行时入参驱动 tiling） |
| 原地语义 | C 原地覆写；C 与 A/B 不重叠由调用方保证；不返回视图 |
| 确定性 | 多核切分与 K 归约顺序由实现决定，不按用例编号或种子分派；不保证逐位一致 |
| 空 Tensor | m/n=0 no-op（A/B/C 可 null）；k=0 或 alpha=0 只做 C=beta·C |
| 0 维 Tensor | 不涉及（标量 Host 指针） |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | S：golden=cblas_sgemm（FP64 累加收窄或 FP32 对照），FLOAT32 档 rtol=atol=2^-13、max_abs ≤1e-2/32ULP；C：golden=cblas_cgemm（实/虚分量分别比对），FLOAT32 分量档；H：golden=FP32（FP64 累加）→FP32→FP16 双段收窄（不用 cblas_hgemm），FLOAT16 档 rtol=atol=2^-10、max_abs ≤1e-1/32ULP；三者 matched_ratio ≥0.99 逐用例联合判定 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel：msprof op Task Duration（warmup5+10 均值）单 case ≤1.25×gpu_ms（≥0.8× GPU）；GTest `[PERF] avg_us`（aclrtEvent）仅交叉验证（verify_performance.py，未解析到判 FAIL） | §3.3；test_cases/*/gpu_baseline.csv |

## 测试概览

- 用例规模：精度全量 1000（S/C/H ≈ 333/333/334）+ 性能子集 200 分摊（按算子 × small ≤64KiB / medium ≤2MiB / large >2MiB 分层，剔除 m/n/k=0）；csv 实测 S 402/C 401/H 400 行（含表头）；性能子集须同时产出两套口径数据。
- 覆盖：transa×transb 9 组合（**Cgemm 的 OP_C 用非对称复矩阵验证与 OP_T 结果不同；S/H 的 OP_C 与 OP_T 结果一致**）；m/n/k 至 4096³ 立方档；ld 紧凑 + +1/+8/+64 padding；alpha/beta 特殊值（S float32；C 复数 (0,0)/(1,0)/(0,1)/(−1,0)/±1e4；H FP16 可表示值 + ±65504）；负向（非法枚举/负值/ld 下界/空指针条件）。
- 特殊场景：m/n/k=0；alpha=0 且 A/B 为 NaN（不引用）；beta=0 且 C 为 NaN（不读旧值）；k=0 且 beta≠1；ldc padding 哨兵；Inf/NaN 传播；H 含次正规数/接近上限值/Inf/NaN 的溢出与次正规处理。
- 测试落位：**扩展 `test/gemm/arch22/` 既有 gemm_test.cpp/csv**（加 op/dtype 列分派三接口），不新建子目录（§2.2 明令）。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + test/gemm/arch22/ ATK 工程）。

## 兼容性分析

- **公共头新增声明**：`include/cann_ops_blas.h` 仅新增 `aclblasHgemm` 一条（紧随 aclblasCgemm，同族风格），不改既有声明；新增导出符号对既有链接面无破坏。
- S/C 既有实现零改动（Cgemm 已由 11-24 合入，本设计仅契约核对）；H 在 `gemm_host.cpp` 增加入口 + `gemm_kernel.cpp/.h` 增加 half 模板路径（或按仓内规范拆分文件，控制文件数量符合 §5 反例约束）；**禁止把 Cgemm/Hgemm 实现混入 cgemm3m 文件**（§2.2 补充说明）。
- `test/gemm/arch22/` 既有工程扩展用例；arch35 与 cgemm3m 路径零触碰。
- README：`blas/gemm/README.md` 补三接口章节（含 H 全 FP16 标量语义、Cgemm OP_C 共轭语义、A2/A3 支持情况）。
- 无既有合法调用方行为变化（S/C 原样；H 为新接口）。

