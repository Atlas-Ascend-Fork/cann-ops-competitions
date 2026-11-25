# aclblasCgemm3m 算子设计文档

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：complex64（实部/虚部各 FP32，`aclblasComplex` 定义见 `include/cann_ops_blas_common.h`）
> 合入位置：`ops-blas/blas/gemm/arch22/`；测试：`ops-blas/test/gemm/cgemm3m/arch22/`
> 任务书：`aclblasCgemm3m_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasCgemm3m 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemm/arch22/`（与同族 aclblasSgemm 同目录，§1/§5）；测试代码 → `test/gemm/cgemm3m/arch22/`（在 `test/gemm/cgemm3m/CMakeLists.txt` 注册 arch22 目标，§2.2）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 禁入声明：**不得放入 `blas/gemm3m/`**——该目录是实数 aclblasSgemm3m（3 个实矩阵沿 K 拼接）接口族的实现目录，与复数 aclblasCgemm3m 非同族（§2.2 末段）。

## 背景介绍

### 算子功能定位

单精度复数三乘法通用矩阵乘（3M GEMM）：用 Gauss 3M 算法把一次复数矩阵乘分解为 **3 次实数矩阵乘**（传统复数乘需 4 次），降低实数乘加总量、增大并行粒度：

```
C = alpha * op(A) * op(B) + beta * C
```

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasCgemm3m | cuBLAS `cublasCgemm3m`｜Netlib `cgemm`（数学语义） | complex64 GEMM，3M 分解加速 |

### 业界对标

| 项 | cuBLAS cublasCgemm3m | 本算子 |
|---|---|---|
| 参数序列 | handle/transa/transb/m/n/k/alpha/A/lda/B/ldb/beta/C/ldc | 逐参数一致（§2.3） |
| transa/transb | N/T/C，C=共轭转置 | 同（复数下 C 与 T **结果不同**，§2.1.1） |
| 布局 | 列主序，lda/ldb/ldc | 同（A 为 lda×k（N）/lda×m（T/C）；B 为 ldb×n（N）/ldb×k（T/C）） |
| 标量 | alpha/beta 复数 | 同（Host 内存指针，§2.4 说明） |
| 非法 trans 返回码 | CUBLAS_STATUS_INVALID_VALUE | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| 舍入 | 3M 与直接复数乘舍入过程不同 | 同——与 aclblasCgemm 结果不保证逐位一致，按 §3.2 阈值判定（§2.1 对标说明） |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口，handle 绑定 stream 异步执行，§2.2/§2.5）。
- 代码落位：`blas/gemm/arch22/` 现有同族五件套（gemm_host/kernel、cgemm_host/kernel 等，2026-11-24 已合入 arch22 复数 cgemm）；本任务**由 cgemm_host.cpp 增加 aclblasCgemm3m 入口、kernel 侧增加 3M 复数路径**（按仓内规范可拆分文件）。
- 公共头：`include/cann_ops_blas.h` L584 声明**已存在，无需修改**（grep 核验；L765 的 Cgemm3mStridedBatched 为另一接口，不在本任务范围）。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemm/arch22/cgemm_{host,kernel}.cpp`（11-24 合入） | ★ 主基线 | 拆平面（cgemm_split_kernel）→ 实平面乘积（子句柄 aclblasSgemmEx + K 分块 + cgemm_accum_kernel）→ 复数 combine（cgemm_combine_kernel）三段流水；workspace 规划（PlanCgemmWorkspace）；Validate* 校验三件套 | 4 个实乘积 → **3 个**；新增和差平面打包 kernel；combine 公式换 3M |
| `blas/gemm/arch35/cgemm3m_host.cpp` + `cgemm3m_rounding_kernel.cpp` | ◆ 参考 | arch35 锚点：LaunchComplexGemm(forceThreeProducts) 分发、HandleCgemmAlphaZero 特值短路；融合纠正路径全套（select/pack/prefix_recompute/permute） | arch35 把 3m 的非法 trans 映射 INVALID_VALUE，与任务书 §2.3 冲突——**按任务书返回 INVALID_ENUM**；融合路径留作性能迭代选项 |
| `blas/gemm3m/arch35/sgemm3m_*` | ◇ 对照 | 3M 接口族命名/落位参照（非同族，不混用） | 不复用其实现 |
| `common/helper/complex_blas3_host_utils.h` | 共享基建 | 复数 BLAS3 host 侧公共工具 | 直接调用 |

### 接口规格

签名与任务书 §2.3 一致（= `include/cann_ops_blas.h` L584 既有声明，与 aclblasCgemm 参数序列完全一致）：

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

参数规格（官方固定表，逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| transa | op(A) 运算类型 | attr | int | {N,T,C}，C=共轭转置；其它 → `INVALID_ENUM` | 标量 |
| transb | op(B) 运算类型 | attr | int | 同 transa；其它 → `INVALID_ENUM` | 标量 |
| m | op(A) 与 C 的行数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | 标量 |
| n | op(B) 与 C 的列数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | 标量 |
| k | 缩并维 | attr | int | ≥0；负 → `INVALID_VALUE`；0 只做 C=beta·C | 标量 |
| alpha | 复数标量乘数（Host） | scalar | COMPLEX64 | 实/虚部任意有限 float；nullptr → `INVALID_VALUE` | 标量 |
| A | 输入矩阵（列主序，Device） | tensor | COMPLEX64 | m>0∧n>0∧k>0∧alpha≠0 时 nullptr → `INVALID_VALUE` | N: lda×k；T/C: lda×m |
| lda | A 前导维 | attr | int | N: ≥max(1,m)；T/C: ≥max(1,k)；低于下界 → `INVALID_VALUE` | 标量 |
| B | 输入矩阵（列主序，Device） | tensor | COMPLEX64 | 同 A 条件 | N: ldb×n；T/C: ldb×k |
| ldb | B 前导维 | attr | int | N: ≥max(1,k)；T/C: ≥max(1,n)；低于下界 → `INVALID_VALUE` | 标量 |
| beta | 复数标量乘数（Host）；beta=0 不读 C | scalar | COMPLEX64 | 同 alpha | 标量 |
| C | 输出矩阵（列主序，原地覆写，Device） | tensor | COMPLEX64 | m>0∧n>0 时 nullptr → `INVALID_VALUE`；仅写逻辑 m×n 区域 | m×n（物理 ldc×n） |
| ldc | C 前导维 | attr | int | ≥max(1,m)；低于下界 → `INVALID_VALUE` | 标量 |

### 数学公式

```
C = alpha * op(A) * op(B) + beta * C，列主序复数矩阵乘
```

3M 分解（§2.1，记 Ar/Ai/Br/Bi 为实虚部平面）：

- **非融合路径（默认）**：P0 = op(Ar)·op(Br)，P1 = op(Ai)·op(Bi)，P2 = op(Ar+Ai)·op(Br+Bi)，
  `op(A)·op(B) = (P0 − P1) + i·(P2 − P0 − P1)`
- **融合路径（含纠正重算）**：P0 = op(Ar+Ai)·op(Br)，P1 = op(Ar)·op(Bi−Br)，P2 = op(Ai)·op(Br+Bi)，
  `op(A)·op(B) = (P0 − P2) + i·(P0 + P1)`

两条分解代数等价、中间项 FP32 舍入不同，误差分析不可互换；实现按形状与片上容量选择路径（§2.1）。共轭（OP_C）通过和差平面打包时的符号折叠实现（Ai 以 −Ai 进入和式）。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现 complex64 句柄式 3M GEMM 接口 aclblasCgemm3m：沿用仓内 arch22 复数 cgemm 的"拆平面 → 实平面乘积 → 复数 combine"三段流水，把 4 个实乘积替换为 3M 分解的 3 个（非融合路径为默认，融合纠正路径为大 shape 性能选项）；支持 transa/transb 9 种组合（含 OP_C 共轭）、任意合法前导维、quick return 与特值短路；精度按 FP32 分量达生态混合容差（golden = cblas_cgemm 直接复数乘），性能达 GPU 标杆 0.8×（单 case 判定）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 3M 分解把复数 GEMM 转成 3 个实数 GEMM + 2 个和差平面打包 + 1 个 3M combine；实平面乘积可整体复用仓内 arch22 cgemm 的 SgemmEx 子句柄链路（11-24 已合入） |
| 2 | 接口原型 | §2.3 签名（= L584 既有声明），返回 `aclblasStatus_t`，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | A/B/C/alpha/beta 全 COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | A2/A3（910b 系）→ arch22；构建 `--soc=ascend910b3`；950PR（arch35）已有实现不动 |
| 6 | 精度要求 | §3.2：FP32 分量 rtol=atol=2^-13、matched_ratio ≥0.99、max_abs_error ≤ 1e-2 或 32 ULP；golden=cblas_cgemm（FP64 累加收窄或 FP32 对照，差异写报告）；3M 与直接乘舍入差异由阈值吸收 |
| 7 | 性能预期 | §3.3：单 case NPU 平均 Task Duration ≤ 1.25×gpu_ms（≥0.8× cublasCgemm3m）；gpu_baseline 200 条实测 max=22.04ms / avg=1.39ms |
| 8 | 代码架构选型 + 可行性 | 三段流水 + 3 实乘积（子句柄复用已合入 SgemmEx 编排）；已排除死路：单 kernel 全融合 SIMT（大 shape 吞吐不足）、blas/gemm3m/ 目录落位（实数接口族，任务书明令禁止）；可行性旁证 = arch35 同代数实现已合入 |
| 9 | 评测来源 | 任务书自验包 test_cases/（1201 条 csv = 精度 1000 + 性能 200；gpu_baseline.csv 201 行）；仓内 ATK 框架 + verify_{accuracy,performance}.py |
| 10 | 特值与退化 | m/n=0 → SUCCESS no-op；k=0 或 alpha=0 → 只做 C=beta·C（beta=(1,0) 时直接 SUCCESS 不动 C）；beta=0 不读 C（含 C 为 NaN）；A/B 在 k=0 或 alpha=0 时可 null，m>0∧n>0 时 C 必须非空 |
| 11 | 错误码口径 | 非法 transa/transb → `INVALID_ENUM`（任务书 §2.3；不沿用 arch35 现网把 3m 映射 INVALID_VALUE 的做法，README 记录差异） |
| 12 | workspace | 拆平面 4 面 + 和差面 2 面 + 积暂存 + SgemmEx scratch，按 `CGEMM_WORKSPACE_ALIGNMENT` 对齐布局；总量经 CheckedMulSize/AddWorkspaceRegion 防溢出 |

# 详细设计

## 算子分析

### 数学公式

逐元素（列主序，r=0..m-1，c=0..n-1）：

```
C[r,c] = alpha * Σ_j op(A)[r,j]·op(B)[j,c] + beta * C[r,c],  j = 0..k-1
```

3M 非融合路径展开（默认实现路径；op 的转置/共轭作用见下）：

```
P0[r,c] = Σ_j op(Ar)[r,j]·op(Br)[j,c]
P1[r,c] = Σ_j op(Ai)[r,j]·op(Bi)[j,c]
P2[r,c] = Σ_j op(Ar+s_a·Ai)[r,j]·op(Br+s_b·Bi)[j,c]     ← 和差平面（打包时折叠共轭符号）
Re(P)   = P0 − P1
Im(P)   = P2 − P0 − P1
C = alpha·P + beta·C（复数 epilogue）
```

其中 `s_a = (transa==C) ? −1 : +1`（B 同理）：**OP_C 的共轭转置 = 转置 + 虚部取反**，转置形态交给实平面 GEMM 的 transa/transb（C→T 映射），取反折入和差平面与 P1 的符号（P1 = op(Ai)·op(Bi) 在双共轭下符号为 s_a·s_b·P1）。融合路径公式见"需求背景·数学公式"，作为大 shape 性能选项（其中间项舍入与非融合不同，切换不改阈值判定）。

特值：alpha=0 或 k=0 → P 项整体跳过，仅执行 `C = beta·C`；beta=0 → C 旧值不读取（C 含 NaN 亦安全）；beta=(1,0) 且 alpha=0/k=0 → 不动 C 直接返回。

### 支持数据类型

| 对象 | 类型 | 说明 |
|---|---|---|
| A / B / C / alpha / beta | COMPLEX64（实虚部各 FP32） | 接口层 |
| Ar/Ai/Br/Bi/和差平面/积平面/combine 中间量 | FP32 | kernel 内部实平面 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m, n | ≥ 0 | 0 → SUCCESS no-op，不引用 A/B/C、不下发 kernel（§2.1.4） |
| k | ≥ 0 | 0 → 只做 C=beta·C（A/B 可 null；C 在 m>0∧n>0 时必须非空）（§2.1.5） |
| lda/ldb/ldc | N: lda≥max(1,m)、ldb≥max(1,k)；T/C: lda≥max(1,k)、ldb≥max(1,n)；ldc≥max(1,m) | 低于下界 → INVALID_VALUE |
| 覆盖档位（§3.5） | m/n/k ∈ {0,1,2,3,5,7,16,31,32,33,63,64,65,127,128,129,255,256,511,512,1023,1024,2047,2048,4095,4096}，含 4096³ 立方档；transa×transb 9 组合全覆盖 | — |
| 写区域 | 仅写 C 逻辑 m×n；ldc padding 列不动（§2.1.6） | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

沿用仓内 arch22 复数 cgemm 的三段流水（2026-11-24 已合入、同目录），乘积段 4→3：

```
aclblasCgemm3m (cgemm_host.cpp 新增入口)
 └─ 校验/quick return/特值短路
 └─ Phase 0  拆平面:   A → Ar,Ai；B → Br,Bi                (cgemm_split_kernel, 复用)
 └─ Phase 0.5 打包:    As = Ar + s_a·Ai；Bs = Br + s_b·Bi  (cgemm3m_pack_kernel, 新增)
 └─ Phase 1  三个实乘积 (子句柄逐个调 aclblasSgemmEx, K 分块):
        P0 = op(Ar)·op(Br)      (首写积, alpha=1 直写 P0 区)
        P1 = op(Ai)·op(Bi)      (积暂存 + accum 缩放累加)
        P2 = op(As)·op(Bs)      (积暂存 + accum)
 └─ Phase 2  3M combine:  re=P0−P1, im=P2−P0−P1, C=alpha·P+beta·C
                                                     (cgemm3m_combine_kernel, 新增)
特值旁路: alpha=0/k=0 → 跳过 Phase 0-1, 仅 combine 语义的 C=beta·C epilogue kernel；
          beta=(1,0) 时直接 SUCCESS 不发 kernel (对齐 arch35 HandleCgemmAlphaZero 模式)
```

kernel 变体划分表：

| 变体 | 触发条件 | 职责 |
|---|---|---|
| cgemm_split_kernel | 常规路径 | 复数交织面 → 4 个实平面（列主 packed） |
| cgemm3m_pack_kernel | 常规路径 | 和差平面 As/Bs（符号折叠共轭） |
| aclblasSgemmEx ×3 | 常规路径 | 实平面 GEMM（满核由 SgemmEx 内部自管；K 按 CGEMM_K_CHUNK 分块） |
| cgemm_accum_kernel | 非直写积 | prod 暂存 → 缩放累加进积平面（复用） |
| cgemm3m_combine_kernel | 常规路径 | 3M 公式 + complex alpha/beta epilogue → 交织写 C（ldc padding 哨兵保护） |
| combine 旁路（alphaZero=1） | alpha=0 或 k=0 | 仅 C=beta·C（betaZero=1 时不读 C） |

数据流：GM 交织复数面 →(split)→ GM 实平面（workspace）→(pack)→ GM 和差面 →(SgemmEx)→ GM 积平面 →(combine)→ GM C。各段 kernel 在 handle 绑定 stream 上顺序下发（同流保序）。

#### host 侧设计

- **分核策略**：乘积段的满核切分委托给 aclblasSgemmEx（仓内实数 GEMM 自带 rows/tiles/wholegrid 多形态切核）；split/pack/combine 为元素型 AIV kernel，按输出元素 m×k / k×n / m×n 以 1024 元素块切 `blocks = min(aivCoreNum, ceil(elems/1024))`。
- **数据分块和内存优化策略**：K 方向按 `CGEMM_K_CHUNK` 分块循环（aKStride/bKStride 按 transa/transb 取 ldAr 或 1），避免 workspace 一次性驻留全量积；积平面 P1/P2 复用 prod 暂存区；workspace 布局（`CGEMM_WORKSPACE_ALIGNMENT` 对齐）：[Ar|Ai|Br|Bi|As|Bs|re|im|prod|SgemmEx scratch]，总量经 CheckedMulSize/AddWorkspaceRegion 溢出防护，超限时返回 INVALID_VALUE。
- **tilingkey 规划策略**：直调工程无独立 tilingkey，分支承载于 tiling 结构字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| transa/transb（折叠后 ta/tb） | N/T（C 已折成符号+T） | SgemmEx 形态选择 |
| s_a/s_b | ±1 | pack 符号、P1 缩放 |
| alphaZero / betaZero | 0/1 | 特值旁路（不读 A/B / 不读 C） |
| hasTemp | 0/1 | combine 直写 C（小 shape 无暂存）或经积平面 |

- **数据检测**（校验顺序逐条对齐 §2.4/§2.5，结构合法后才 quick return）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | transa ∉ {N,T,C} | INVALID_ENUM |
| 3 | transb ∉ {N,T,C} | INVALID_ENUM |
| 4 | m/n/k < 0 | INVALID_VALUE |
| 5 | lda 低于下界（按 transa） | INVALID_VALUE |
| 6 | ldb 低于下界（按 transb） | INVALID_VALUE |
| 7 | ldc < max(1,m) | INVALID_VALUE |
| 8 | alpha/beta == nullptr | INVALID_VALUE |
| 9 | m>0∧n>0 时 C == nullptr | INVALID_VALUE |
| 10 | m>0∧n>0∧k>0∧alpha≠0 时 A/B == nullptr | INVALID_VALUE |
| 11 | 全过后：m==0 或 n==0 → SUCCESS（不下发 kernel） | — |
| 12 | alpha==0 或 k==0 → A/B 判空跳过，走 C=beta·C 旁路（beta=(1,0) 直接 SUCCESS） | — |

#### kernel 侧设计

- **Init 阶段**（各元素型 kernel）：按 tiling 元素总数与 blockIdx 划分本核区间，绑定 GM 输入/输出与 UB buffer（128B 对齐段拷贝）。
- **Process 阶段**：
  - split：交织复数面读入（float2 视角），实/虚分离写两平面（复用 cgemm_split_kernel 的列主 packed 写序：col×packedLd+row）。
  - pack：`As = Ar + s_a·Ai`、`Bs = Br + s_b·Bi`（同形状平面二元加，符号来自 tiling）。
  - combine：逐元素 `re=P0[r,c]−P1[r,c]`、`im=P2[r,c]−P0[r,c]−P1[r,c]`，再按 complex alpha/beta epilogue（复乘展开 αr·re−αi·im / αr·im+αi·re；betaZero 时跳过读 C）交织写 C 逻辑区域，padding 列不触碰。
- 流程图（常规路径）：

```
split ──→ pack ──→ SgemmEx(P0) ──→ SgemmEx(P1)+accum ──→ SgemmEx(P2)+accum ──→ combine3m ──→ C
  A,B        As,Bs     Ar×Br→P0直写     Ai×Bi→prod→re          As×Bs→prod→im       re/im+α/β
```

- **workspace 布局表**（float 元素，均按 CGEMM_WORKSPACE_ALIGNMENT 对齐）：

| 区 | 大小 | 用途 |
|---|---|---|
| Ar/Ai | ldAr×kA_each | A 拆平面（ldAr = CeilAlign(按 trans 的行数, 16)） |
| Br/Bi | ldBr×kB_each | B 拆平面 |
| As/Bs | 同 Ar/Br | 和差平面 |
| re/im | ldRe×n | 积平面（hasTemp=0 时直接以 C 为积区） |
| prod | ldRe×n | 非直写积暂存 |
| scratch | SgemmEx 子句柄 | 实 GEMM 内部 workspace |

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| 3M 替代 4M | 实乘积 4→3（约 0.75× 乘加总量） | 大 k 档收益主项（待 msprof 实测回填） |
| 首写直写 | P0 以 alpha=1 直写积区，省一次 accum | 消 1 轮 m×n 读改写 |
| K 分块复用 | CGEMM_K_CHUNK 循环 + prod 暂存 | workspace 有界，支持 4096³ 立方档 |
| 和差面打包融合符号 | pack 单遍完成 As/Bs 与 OP_C 折叠 | 不额外发取反 kernel |
| 融合纠正路径（备选） | arch35 cgemm3m_rounding 全套（select/pack/prefix_recompute/permute） | P2 带宽受限时启用；默认不开（复杂度/收益待实测） |
| 已排除路线 | 单 kernel 全融合 SIMT；blas/gemm3m/ 目录落位 | 前者大 shape 吞吐不足；后者任务书禁止 |

判据逐 case 独立：NPU ≤ 1.25×gpu_ms；gpu_baseline 200 条 max=22.04ms / avg=1.39ms（cublasCgemm3m kernel-only）。0.75× 乘加量是纸面方向，达标与否以设备 msprof 实测为准（设计阶段不判决）。

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
| 校验层级 | 结构参数全在 host 侧校验；不扫描 Device 数据 |
| 非连续支持 | 仅 lda/ldb/ldc 带填充布局；不用 AutoContiguous/ViewCopy、不产生相关临时空间 |
| broadcast | 不支持 |
| dynamic shape | 天然支持（运行时入参驱动 tiling） |
| 原地语义 | C 原地覆写；C 可写区与 A/B 可读区不重叠由调用方保证；不返回视图 |
| 确定性 | 3M 路径与 K 归约顺序按形状/片上容量选择，不按用例编号或种子分派；两次执行不保证逐位一致 |
| 空 Tensor | m=0/n=0 no-op；k=0/alpha=0 只做 C=beta·C |
| 0 维 Tensor | 不涉及 |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | golden = CPU cblas_cgemm（Netlib cgemm，直接复数乘；FP64 累加收窄或 FP32 对照，差异在自测报告说明）；实部/虚部分别逐元素判定 `|actual−golden| ≤ atol + rtol·|golden|`；rtol=atol=2^-13（FLOAT32 分量）、matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP 时用例通过；3M 与直接乘的舍入差异由阈值吸收 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel：msprof op Task Duration，warmup 5 + 10 样本均值，单 case NPU avg ≤ 1.25×gpu_ms（≥0.8× cublasCgemm3m kernel-only 基线）；GTest `[PERF] avg_us`（aclrtEvent，warmup5+10）仅交叉验证 | §3.3；test_cases/gpu_baseline.csv |

## 测试概览

- 用例规模：精度全量 1000 + 性能子集 200（csv 实测 1201 行含表头）；transa×transb 9 组合全覆盖；m/n/k 至 4096³ 立方档；性能子集按 small ≤64KiB / medium ≤2MiB / large >2MiB 分层（剔除 m/n/k=0），约 12% 非默认 alpha/beta。
- 特殊场景：m/n/k=0；alpha=0 且 A/B 含 NaN（只做 beta·C 不引用）；beta=0 且 C 含 NaN；k=0 且 beta≠1；OP_C 非对称复矩阵（与 OP_T 结果不同）；ldc padding 列哨兵写保护；Inf/NaN 传播；复数输入覆盖全零/纯实/纯虚/一般四类。
- 性能口径：msprof op（正式）+ GTest `[PERF] avg_us`（交叉验证，verify_performance.py 对表 gpu_baseline.csv）。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + `test/gemm/cgemm3m/` ATK 工程：run_self_test.py / run_memcheck.py / test_gate.py）。

## 兼容性分析

- 公共头零改动（L584 既有声明），无接口面波及。
- 新增入口挂在 `blas/gemm/arch22/cgemm_host.cpp`（任务书 §2.2 指定），kernel 新增 cgemm3m_pack/cgemm3m_combine 两变体（可拆分文件，按仓内规范控制文件数量）；arch35 实现与 `blas/gemm3m/` 实数族零触碰。
- 错误码差异记录：arch35 现网把 3m 非法 trans 的 INVALID_ENUM 映射为 INVALID_VALUE，本实现按任务书 §2.3 返回 INVALID_ENUM，README 的 aclblasCgemm3m 章节记录该口径；README 同步补 Atlas A2/A3 产品支持情况（§2.2）。
- `test/gemm/cgemm3m/CMakeLists.txt` 注册 arch22 目标；arch35 测试不动。
- 对既有调用方无行为变更（新接口，无历史调用方）。

