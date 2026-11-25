# aclblasSgemmBatched / aclblasHgemmBatched 算子设计文档

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：S = FLOAT32 全链；H = FLOAT16 全链（A/B/C/alpha/beta 均为 `aclblasHalf`，计算域 FP32）
> 合入位置：`ops-blas/blas/gemm_batched/arch22/`；测试：`ops-blas/test/gemm_batched/sgemm_batched/arch22/`（S/H 复用同一用例集，按 op 列分派）
> 任务书：`aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasSgemmBatched / aclblasHgemmBatched 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemm_batched/arch22/`（与既有 cgemm_batched_* 同目录，§1/§2.2/§5）；测试代码 → `test/gemm_batched/sgemm_batched/arch22/`（在 `test/gemm_batched/sgemm_batched/CMakeLists.txt` 注册 arch22 目标）。
- 验收流程：社区任务 IT 系统四件套交付件（§4）；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 评审合入。
- 上游现状：arch22 目录仅有复数 cgemm_batched（10 月合入家族），**S/H 实数版空白**（ls-tree 实证）——本任务为全新实现，无先手竞争。

## 背景介绍

### 算子功能定位

批量通用矩阵乘：对 batchCount 个独立 (A[i], B[i], C[i]) 三元组分别执行标准 GEMM，各 batch 共享 transa/transb、m/n/k、lda/ldb/ldc、alpha、beta，数据与地址相互独立：

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  i = 0..batchCount-1
```

| 接口 | 对标 | A/B/C/alpha/beta | 作用 |
|---|---|---|---|
| aclblasSgemmBatched | cuBLAS `cublasSgemmBatched`｜Netlib `sgemm` | FLOAT32 | 单精度批量 GEMM |
| aclblasHgemmBatched | cuBLAS `cublasHgemmBatched`（Netlib 无 half 参考） | FLOAT16（aclblasHalf） | 半精度批量 GEMM（FP32 计算域） |

### 业界对标

| 项 | cuBLAS 对应变体 | 本算子 |
|---|---|---|
| 参数序列 | handle/transa/transb/m/n/k/alpha/Aarray/lda/Barray/ldb/beta/Carray/ldc/batchCount | 逐参数一致（§2.3） |
| transa/transb | N/T/C | 同——实数算子 **OP_C ≡ OP_T 不共轭**（§2.1） |
| 布局 | 列主序 + 指针数组 | 同（A: lda×k（N）/lda×m（T/C）；B: ldb×n（N）/ldb×k（T/C）；C: ldc×n） |
| H 标量 | alpha/beta 为 half | 同（`aclblasHalf`；与 gemvBatched 半精度变体的 FP32 标量不同） |
| 非法 trans 返回码 | CUBLAS_STATUS_INVALID_VALUE | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| quick return | m/n=0、batchCount=0 no-op；k=0/alpha=0 只做 C=beta·C | 同（§2.1.3-2.1.5） |

### 工程形态

- 工程模式：**Ascend C kernel 直调**（句柄式 BLAS 接口，handle 绑定 stream 异步执行，§2.2）。
- 代码落位：`blas/gemm_batched/arch22/` 下新增 S/H 的 host 与 kernel 文件（或扩展既有 gemm_batched 导出入口，§2.2）；与同目录 cgemm_batched_* 并存。
- 公共头：`include/cann_ops_blas.h` 声明**已存在（S L606 / H L616，grep 实证），无需修改**。
- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemm_batched/arch35/gemm_batched_host.cpp`（S）+ `hgemmBatched_host.cpp`（H） | ★ 任务书指定参考 | ValidateSgemmBatchedParams（L47-93）、LaunchEarlyExit / LaunchAlphaBetaPostProcess（特值旁路）、CalcGemmTiling + ApplyColumnMajorSwap（batch+tile 二级切分）、EnsureWorkspace | arch22 重写发射层（SDK 兼容性，任务书禁止直接复制）；勿陷入同文件后半 CgemmBatched 特化 tuning 群 |
| `blas/gemm_batched/arch22/cgemm_batched_*`（同目录兄弟） | ★ | arch22 工程模式：CgbtDispatch 多路分发（beta 旁路/小形状向量路/direct/general）、CgbtPlanWorkspace 指针数组 workspace 规划、校验骨架 | 实数版砍掉 deinterleave/combine 复数专属段 |
| `blas/gemm/arch22/gemm_{host,kernel}.cpp`（实数 GEMM 基建） | ◆ 底座 | 大矩阵路径：rows/tiles/wholegrid + pack/accumulate/epilogue/zero 多 kernel 形态；子句柄调用编排模式 | batch 外层循环 + batch 维切核 |
| `test/gemm_batched/sgemm_batched/` + `gemm_batched_npu_common.h` | 既有 | S/H 复用用例集按 op 列分派的测试组织 | 新建 arch22/ 注册目标 |

### 接口规格

签名与任务书 §2.3 一致（= `include/cann_ops_blas.h` L606/L616 既有声明）：

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount);

aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const Carray[], int ldc,
    int batchCount);
```

参数规格（官方固定表，逐行对齐 §2.4；S/H 同构，dtype 列区分）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 库上下文句柄，携带 stream | scalar | - | 有效句柄；nullptr → `HANDLE_IS_NULLPTR` | 标量 |
| transa/transb | op(A)/op(B) 运算类型 | attr | int | {N,T,C}，C≡T；其它 → `INVALID_ENUM` | 标量 |
| m / n / k | GEMM 维数 | attr | int | ≥0；负 → `INVALID_VALUE`；m/n=0 合法 no-op；k=0 只做 C=beta·C | 标量 |
| alpha | 标量乘数（Host） | scalar | S=FLOAT32；H=FLOAT16 | 任意有限值（FP16 可表示域）；nullptr → `INVALID_VALUE` | 标量 |
| Aarray | A[i] Device 指针数组 | tensor | 同 alpha | batchCount>0∧m>0∧n>0∧k>0∧alpha≠0 时 nullptr → `INVALID_VALUE`；元素有效性由调用方保证 | batchCount 个（N: lda×k；T/C: lda×m） |
| lda | A 前导维 | attr | int | N: ≥max(1,m)；T/C: ≥max(1,k)；低于 → `INVALID_VALUE` | 标量 |
| Barray | B[i] Device 指针数组 | tensor | 同 alpha | 同 Aarray 条件 | batchCount 个（N: ldb×n；T/C: ldb×k） |
| ldb | B 前导维 | attr | int | N: ≥max(1,k)；T/C: ≥max(1,n)；低于 → `INVALID_VALUE` | 标量 |
| beta | 标量乘数（Host）；beta=0 不读 C | scalar | 同 alpha | 同 alpha | 标量 |
| Carray | C[i] Device 指针数组（原地覆写） | tensor | 同 alpha | batchCount>0∧m>0∧n>0 时 nullptr → `INVALID_VALUE`；C[i] 互不重叠且不与 A[j]/B[j] 重叠 | batchCount 个 ldc×n |
| ldc | C 前导维 | attr | int | ≥max(1,m)；低于 → `INVALID_VALUE` | 标量 |
| batchCount | batch 数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 合法 no-op | 标量 |

### 数学公式

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
op(A[i])：N→A[i]（m×k）；T/C→A[i]^T（m×k，实数无共轭）
逐元素：C[i][r,c] = alpha · Σ_t op(A)[r,t]·op(B)[t,c] + beta · C[i][r,c]
```

仅写各 C[i] 逻辑 m×n 区域（ldc padding 列不动）。特值：alpha=0 或 k=0 → 仅 `C = beta·C`（beta=0 → 直写 0 不读 C）；beta=0 → C 旧值不读取。H 变体：输入/标量 FP16 → FP32 域乘加 → IEEE round-to-nearest 收窄 FP16 落盘（与 §3.2.4 golden 口径一致）。

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现 2 个句柄式批量实数 GEMM 接口（S=FP32 全链、H=FP16 全链+FP32 计算域）：host 多路分发（特值旁路 / 小矩阵大批量批量 kernel / 大矩阵分块复用实数 GEMM 基建），batch 维为切核主维度（batchCount 至 65536）；支持 transa×transb 9 组合（OP_C≡T）、任意合法前导维、指针数组批量组织；精度 S 按 FP32 / H 按 FP16 达生态混合容差，性能达 GPU 标杆 0.8×（单 case 判定）。

## 需求拆解（必须回答表）

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 批量 GEMM = 逐 batch 独立标准 GEMM（共享形状/标量）；arch22 实数 GEMM 基建（pack/accumulate/epilogue/wholegrid）可直接复用为大矩阵路径底座 |
| 2 | 接口原型 | §2.3 双签名（= L606/L616 既有声明），返回 `aclblasStatus_t`，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"表；**条件判空**逐条对齐 §2.4 |
| 4 | 支持数据类型 | S：全 FP32；H：全 FP16（含 alpha/beta），FP32 计算域 |
| 5 | 目标芯片与 arch 映射 | A2/A3（910b 系）→ arch22；构建 `--ops=sgemm_batched,hgemm_batched --soc=ascend910b3`；arch35 已有实现不动 |
| 6 | 精度要求 | §3.2：S 按 FP32（rtol=atol=2^-13，golden=cblas_sgemm 逐 batch，FP64 累加收窄）；H 按 FP16（rtol=atol=2^-10，max_abs 1e-1，golden=FP32（FP64 累加）→FP32→FP16 双段收窄） |
| 7 | 性能预期 | §3.3：单 case NPU Task Duration ≤1.25×gpu_ms（≥0.8× GPU）；gpu_baseline 实测 S max=8.8079ms/avg=0.4470ms、H max=1.6636ms/avg=0.0699ms（各 100 条） |
| 8 | 代码架构选型 + 可行性 | host 三路分发：① 小矩阵大批量 → 单 kernel 指针数组直读（batch 满核切分）；② 大矩阵 → 按 batch 分块子句柄复用 arch22 实数 GEMM 基建；③ 特值 beta 缩放旁路；已排除死路：全用例逐 batch 子句柄（65536 档发射爆炸）、全融合单 kernel（4096³ 档 tile 管理复杂）；可行性旁证 = arch35 同代数实现 + arch22 GEMM 基建均已合入 |
| 9 | 评测来源 | 任务书自验包 test_cases/{sgemm_batched,hgemm_batched}/（各 601 行 csv = 500 精度 + 100 性能；gpu_baseline 各 101 行） |
| 10 | 特值与退化 | m/n/batchCount=0 no-op；k=0 或 alpha=0 → 只做 C=beta·C（A/B 数组可 null；Carray 在 m>0∧n>0 必须非空）；beta=0 不读 C（NaN 安全） |
| 11 | 错误码口径 | 非法 transa/transb → INVALID_ENUM；维度/ld/batchCount/指针 → INVALID_VALUE |
| 12 | workspace | ② 路由 GEMM 基建自管；① 路无 workspace（指针数组直读 + 核内 UB）；③ 路无 workspace；host 侧总控防溢出（size_t 乘加对） |

# 详细设计

## 算子分析

### 数学公式

逐 batch（i = 0..batchCount-1）独立计算（FP32 计算域；S 的存储域亦 FP32，H 存储域 FP16）：

```
C[i][r,c] = alpha · Σ_{t=0..k-1} op(A[i])[r,t] · op(B[i])[t,c] + beta · C[i][r,c]
op(A[i]) = A[i]（transa=N）；A[i]^T（transa=T/C，实数无共轭）
```

H 变体数据通路：A/B/alpha/beta FP16 → FP32 → FP32 乘加累加 → IEEE round-to-nearest 收窄 FP16 写 C（与 §3.2.4 golden 的 FP64 累加→FP32→FP16 口径在 FP32 域内一致）。特值：alpha=0 或 k=0 → 乘加项跳过（不引用 Aarray/Barray 元素数值）；beta=0 → C 旧值不读（含 NaN 安全，直写 alpha·op(A)op(B) 或 0）。

### 支持数据类型

| 变体 | A/B/C 存储域 | alpha/beta | 计算域 | kernel 模板实例 |
|---|---|---|---|---|
| S | FP32 | FP32（Host） | FP32 | GemmBatchedKernel<float,...> |
| H | FP16（aclblasHalf） | FP16（Host） | FP32（进出 cast） | GemmBatchedKernel<half,...> |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m, n | ≥ 0 | 0 → SUCCESS no-op（三数组可 null，不下发 kernel） |
| k | ≥ 0 | 0 → 仅 C=beta·C（Aarray/Barray 可 null；Carray 条件判空） |
| batchCount | ≥ 0 | 0 → SUCCESS no-op（不引用任何数组元素） |
| lda/ldb/ldc | 按_trans 取下界（N: max(1,m)/max(1,k)/max(1,m)；T/C 换位） | 低于 → INVALID_VALUE |
| 覆盖档位（§3.5） | m/n/k ∈ {0..4096 全档，含 4096³ 立方}；batchCount ∈ {0,1,2,3,8,16,64,256,1024,4096,16384,65536}（后 4 档配 1×1×1） | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

host 校验 + 特值旁路 + 三路正态分发（batch 维为切核主维度）：

```
aclblas{S,H}gemmBatched (gemm_batched arch22 新增 host)
 └─ 校验（§2.5 顺序 + 条件判空）→ quick return（m/n/batchCount=0）
 ├─ 特值旁路: alpha=0 或 k=0 → 批量 beta 缩放 kernel（C[i]=beta·C[i]，逐 batch strided；
 │            beta=0 直写 0；beta=1 且 alpha=0/k=0 → 直接 SUCCESS 不发 kernel）
 ├─ ① 小矩阵大批量路（m·n·k ≤ 批量阈值 或 batchCount ≥ 大批量阈值）:
 │      gemm_batched_small_kernel（单发射，指针数组直读）：
 │      batchPerCore = ⌈batchCount/coreNum⌉ 均分 + 尾核余块；核内逐 batch 完整 GEMM
 │      （A[i]/B[i]/C[i] 指针经数组元素 8B 直读；FP32 计算域；写 C 逻辑区域）
 └─ ② 大矩阵路: 按 batch 分块（块大小由 GEMM 基建 workspace 上限决定），子句柄复用
        blas/gemm/arch22 实数 GEMM 基建（rows/tiles/wholegrid + pack/accumulate/epilogue）：
        逐 batch 以 alpha=1、beta 折算调用，或按块聚合发射；H 变体在 cast 层适配
```

kernel 变体划分表：

| 变体 | 触发条件 | 职责 |
|---|---|---|
| beta 缩放批量 kernel | alpha=0 或 k=0 | 逐 batch C[i]=beta·C[i]（strided；beta=0 写 0） |
| gemm_batched_small_kernel（S/H 模板） | 小矩阵或 batchCount ≥ 1024 档 | batch 满核切分，核内逐 batch 完整乘加（指针数组直读） |
| ② 路 GEMM 基建多 kernel（既有） | 单 batch 计算量大 | pack→accumulate→epilogue / wholegrid / rows 形态（基建自选择） |

数据流（① 路）：GM Aarray/Barray/Carray（指针数组）→(8B 直读)→ GM A[i]/B[i] →（核内 UB 分块）→ FP32 累加 → GM C[i]（原地）。

#### host 侧设计

- **分核策略**：① 路 batch 满核均分（maxPerCore/usedCoreNum/batchTail 三元组，同 gemv_batched arch22 口径）；② 路切分交由 GEMM 基建（其内部 rows/tiles/wholegrid 自适应）+ host 侧 batch 分块循环。
- **数据分块和内存优化策略**：① 路核内按 m/n/k 以固定 TILE 分块流过 UB（无 GM 暂存）；② 路复用基建 workspace（pack 缓冲 + 积暂存）；路径选择的批量阈值（如 m·n·k ≤ 32³ 或 batchCount ≥ 1024）以实测校准后在 tiling 常量固化。
- **tilingkey 规划策略**：直调工程分支承载于 tiling 字段：

| 字段 | 取值 | 触发分支 |
|---|---|---|
| dtype | 0/1 | S（float）/ H（half）模板分发 |
| path | 0/1/2 | ① 批量 kernel / ② GEMM 基建 / ③ beta 缩放 |
| transa/transb | 折叠为 {N,T}² | 4 形态（C→T 折算） |
| alphaZero/betaZero/betaOne | 0/1 | 特值分支 |

- **数据检测**（校验顺序逐条对齐 §2.5 注；结构合法后再 quick return）：

| 序 | 检查 | 异常码 |
|---|---|---|
| 1 | handle == nullptr | HANDLE_IS_NULLPTR |
| 2 | transa ∉ {N,T,C} | INVALID_ENUM |
| 3 | transb ∉ {N,T,C} | INVALID_ENUM |
| 4 | m/n/k < 0 | INVALID_VALUE |
| 5 | lda/ldb/ldc 低于对应下界 | INVALID_VALUE |
| 6 | batchCount < 0 | INVALID_VALUE |
| 7 | alpha/beta == nullptr | INVALID_VALUE |
| 8 | batchCount>0∧m>0∧n>0∧k>0∧alpha≠0 时 Aarray/Barray == nullptr | INVALID_VALUE |
| 9 | batchCount>0∧m>0∧n>0 时 Carray == nullptr | INVALID_VALUE |
| 10 | 全过后：m==0 或 n==0 或 batchCount==0 → SUCCESS（不下发 kernel） | — |
| 11 | k==0 或 alpha==0 → beta==1 直接 SUCCESS；否则 ③ 路（不引用 A/B） | — |

  kernel 发射失败（aclError ≠ SUCCESS）→ `EXECUTION_FAILED` 记日志。

#### kernel 侧设计

- **① 路 gemm_batched_small_kernel（模板 T=float/half）**：
  - Init：blockIdx → 本核 batch 区间；绑定三指针数组 GM tensor；UB 布局（A/B/C tile buffer + FP32 累加器）。
  - Process：逐 batch——指针数组元素 8B 直读取 A[i]/B[i]/C[i] 地址；k 分块流过（op(A)/op(B) 按 trans 折叠的读取下标：N 直读列段、T 换位读）；FP32 乘加进 C-tile 累加器；beta≠0 时先读 C[i] 旧值乘 beta；收窄写 C[i] 逻辑 m×n（ldc padding 不触碰）。
- **③ 路 beta 缩放 kernel**：逐 batch 逐元素 `C=beta·C`（strided；beta=0 → Duplicate(0) 不读）。
- 流程图（① 路）：

```
for b ∈ 本核 batch 区间:
    aPtr/bPtr/cPtr = ReadPtr(Aarray/Barray/Carray, b)
    acc := (beta≠0) ? beta·C[i] : 0
    for t ∈ k 分块:
        载 op(A)[tile,t] 与 op(B)[t,tile]（cast FP32）
        acc += op(A)·op(B)（FP32 域）
    C[i][tile] := narrow(acc)
```

- **workspace 布局表**：①/③ 路无 workspace；② 路由 GEMM 基建自管（pack/积暂存）；host 以 size_t 乘加对防 batchCount·元素数溢出。

#### 性能优化要点（终态）

| 优化项 | 机制 | 效果 |
|---|---|---|
| batch 满核切分（① 路） | batchPerCore 均分 + 尾核 | 65536×1×1 档单发射吃满核（vs 逐 batch 发射 65536 次） |
| 大矩阵走已调优基建（② 路） | 复用 arch22 GEMM rows/tiles/wholegrid | 4096³ 档继承既有性能形态 |
| 指针数组直读 | 8B DataCopyPad + GetValue | 免整体数组 H2D 式搬运 |
| H 的 FP32 计算域 | cast 进出 | 精度对齐 FP16 阈值 + FP32 golden 口径 |
| 已排除路线 | 全用例逐 batch 子句柄（65536 档发射爆炸）；全融合单 kernel（大档 tile 管理复杂） | — |

判据逐 case 独立：NPU ≤ 1.25×gpu_ms；gpu_baseline S max=8.8079ms/avg=0.4470ms、H max=1.6636ms/avg=0.0699ms（cublas 对应接口 kernel-only）。达标以设备 msprof 实测为准（设计阶段不判决）。

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
| 校验层级 | 结构参数（含指针数组本体）host 侧校验；不扫描数组元素/Device 张量；元素有效性与重叠由调用方保证 |
| 非连续支持 | 各 batch 按 lda/ldb/ldc 带填充布局；batch 间地址独立（指针数组描述）；不用 AutoContiguous/ViewCopy |
| broadcast | 不支持（batch 共享形状但数据不广播） |
| dynamic shape | 天然支持（m/n/k/ld/batchCount 运行时入参驱动 tiling） |
| 原地语义 | C[i] 原地覆写；C[i] 与 A[j]/B[j] 不重叠、C[i] 间不重叠由调用方保证；不返回视图 |
| 确定性 | batch 调度/多核切分/K 归约顺序由实现决定，不按用例编号或种子分派；不保证逐位一致 |
| 空 Tensor | m/n/batchCount=0 no-op；k=0 或 alpha=0 只做 C=beta·C（A/B 数组可 null；Carray 条件判空） |
| 0 维 Tensor | 不涉及（标量 Host 指针） |
| 异步执行 | stream 上异步；返回≠完成 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | S：golden = 逐 batch cblas_sgemm（FP64 累加收窄或 FP32 对照，差异写报告），FP32 阈值 rtol=atol=2^-13、max_abs ≤1e-2 或 32 ULP；H：golden = 测试工程 FP32（FP64 累加）→FP32→FP16 双段收窄（不用 cblas_hgemm），FP16 阈值 rtol=atol=2^-10、max_abs ≤1e-1 或 32 ULP；两者 matched_ratio ≥0.99 逐用例联合判定 | §3.2；mixed_tolerance_standard（pin 28a1ce5） |
| 性能标准 | 正式判据 kernel-to-kernel：msprof op Task Duration（warmup5+10 均值）单 case ≤1.25×gpu_ms（≥0.8× GPU）；GTest `[PERF] avg_us`（aclrtEvent）仅交叉验证（verify_performance.py，未解析到判 FAIL） | §3.3；test_cases/*/gpu_baseline.csv |

## 测试概览

- 用例规模：精度全量 1000（S 500 + H 500）+ 性能子集 200（各 100，按算子 × small ≤64KiB / medium ≤2MiB / large >2MiB 分层，剔除 m/n/k=0 或 batchCount=0）；csv 实测每算子 601 行。
- 覆盖：transa×transb 9 组合；m/n/k 至 4096³ 立方档；batchCount 至 65536（后 4 档 1×1×1）；ld 紧凑 + +1/+8/+64 padding；alpha/beta 特殊值 0/1/−1/0.5/±1e4（H 取 FP16 可表示值）；负向（非法枚举/负值/零步长 n/a/ld 下界/空指针条件）。
- 特殊场景：m/n/k=0；alpha=0 且 A/B 为 NaN（不引用）；beta=0 且 C 为 NaN（不读旧值）；k=0 且 beta≠1；batchCount=0（不引用数组）；**OP_C 与 OP_T 结果一致**（实数交叉验证）；ldc padding 哨兵；Inf/NaN 传播；H 含次正规数/接近上限值/Inf/NaN 的溢出与次正规处理。
- 性能口径：msprof op（正式）+ `[PERF] avg_us`（交叉）；性能子集须同时产出两份数据。
- 复现入口：交付件 03《自验证步骤说明》（自验包 readme + `test/gemm_batched/sgemm_batched/arch22/` ATK 工程，S/H 按 op 列分派）。

## 兼容性分析

- 公共头零改动（L606/L616 既有声明），无接口面波及。
- 新增 S/H host 与 kernel 文件（与 cgemm_batched_* 同目录并存，或扩展 gemm_batched_host.cpp 导出入口——按文件规模与仓内规范定，控制文件数量符合 §5 反例约束）；arch35 实现与 cgemm_batched 路径零触碰。
- `test/gemm_batched/sgemm_batched/CMakeLists.txt` 注册 arch22 目标；既有 arch35 测试不动。
- README：`blas/gemm_batched/README.md` 补 S/H 章节（含 H 全 FP16 标量语义、OP_C≡T、A2/A3 支持情况）。
- 无既有调用方（arch22 新接口面），无行为变更波及。

