# aclblasSgemmBatched / aclblasHgemmBatched 算子设计文档（Atlas A2/A3）

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：S=FLOAT32；H=FLOAT16（`aclblasHalf`，IEEE FP16 位模式）
> 合入位置：`ops-blas/blas/gemm_batched/arch22/`；测试：`ops-blas/test/gemm_batched/sgemm_batched/arch22/`（S/H 复用同一 GTest 用例集，按 `op` 列分派）
> 任务书：`aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasSgemmBatched / aclblasHgemmBatched 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemm_batched/arch22/`；测试代码 → `test/gemm_batched/sgemm_batched/arch22/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入（§4）。

## 背景介绍

### 算子功能定位

批量通用矩阵乘：`C[i] = alpha·op(A[i])·op(B[i]) + beta·C[i]`，i = 0..batchCount−1。每个 batch 独立执行一次标准 GEMM；各 batch 共享 transa/transb、m/n/k、lda/ldb/ldc、alpha、beta，A/B/C 数据与地址通过 Device 侧指针数组（Aarray/Barray/Carray）相互独立。

| 接口 | 对标 | dtype | 作用 |
|---|---|---|---|
| `aclblasSgemmBatched` | cuBLAS `cublasSgemmBatched`（语义参考 Netlib `sgemm`） | FLOAT32 | 单精度批量 GEMM |
| `aclblasHgemmBatched` | cuBLAS `cublasHgemmBatched` | FLOAT16（aclblasHalf） | 半精度批量 GEMM |

### 业界对标

- 实数算子，`OP_C` 与 `OP_T` 等价（不做共轭）。
- 有意差异：非法 trans 枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（cuBLAS 返回 INVALID_VALUE）；alpha/beta 固定 Host 内存（与 cuBLAS 默认 pointer mode 一致）。
- H 的 golden：Netlib 无 half 参考实现，以 FP32（float64 累加）计算后收窄到 FP16 作为参考（§3.2）。

### 工程形态

- 工程模式：Host C API + Ascend C kernel 直调（handle 绑定 stream，§2.2）。
- 代码落位：`blas/gemm_batched/arch22/`，与现有 `cgemm_batched_*` 同目录，新增 S/H 的 host 与 kernel 文件。
- 仓内近亲：`blas/gemm_batched/arch35/gemm_batched_host.cpp`（S）与 `hgemmBatched_host.cpp`（H）已有 arch35 实现，可参考 batch 调度、指针数组处理与 FP16 缩放路径；arch22 单矩阵 SGEMM/HGEMM 提供分块基线。

### 接口规格

两接口声明均已存在于 `include/cann_ops_blas.h`（S L606、H L616），无需新增：

```cpp
aclblasStatus_t aclblasSgemmBatched(handle, transa, transb, m, n, k,
    const float* alpha, const float* const Aarray[], int lda,
    const float* const Barray[], int ldb, const float* beta,
    float* const Carray[], int ldc, int batchCount);
aclblasStatus_t aclblasHgemmBatched(handle, transa, transb, m, n, k,
    const aclblasHalf* alpha, const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb, const aclblasHalf* beta,
    aclblasHalf* const Carray[], int ldc, int batchCount);
```

参数规格固定表（§2.4 逐行）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | - | - | 非空，否则 HANDLE_IS_NULLPTR | - |
| transa/transb | op(A)/op(B) 类型 | attr int | - | {OP_N, OP_T, OP_C}，非法→INVALID_ENUM；C≡T | - |
| m/n/k | 三维规模 | attr int | - | ≥0；m或n=0 no-op；k=0 只做 C=beta*C | - |
| alpha/beta | 标量（Host 内存） | scalar | S=FP32；H=FP16 | 非空 | - |
| Aarray/Barray | Device 指针数组（长度 batchCount） | tensor | S=FP32；H=FP16 | 数组本体 host 校验；元素有效性调用方保证 | 每元素指向 lda×k（N）等 |
| lda/ldb/ldc | 前导维度 | attr int | - | lda≥max(1,m或k)、ldb≥max(1,k或n)、ldc≥max(1,m) | - |
| Carray | 输出指针数组，原地覆写 | tensor | 同上 | C[i] 互不重叠且不与 A[j]/B[j] 重叠 | 每元素 ldc×n |
| batchCount | batch 数量 | attr int | - | ≥0；=0 合法 no-op | - |

### 数学公式

`C[i] = alpha·op(A[i])·op(B[i]) + beta·C[i]`（i = 0..batchCount−1），列主序；仅写各 C[i] 的逻辑 m×n 区域。

# 需求分析（required）

## 需求描述

以 Ascend C 在 arch22 上实现 S/H 双 dtype 批量 GEMM：Device 指针数组的 batch 调度 + 复用单矩阵 GEMM 分块流水，支持 N/T/C 全组合、batchCount 0..65536、带填充布局与完整 quick return 语义；S 精度按 FLOAT32 阈值、H 按 FLOAT16 阈值（matched_ratio≥0.99），性能达 GPU kernel 标杆 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | batch 间无数据依赖，可完全并行；单 batch 即标准 SGEMM/HGEMM，分块与流水全部复用 arch22 单矩阵基线 |
| 2 | 接口原型 | `include/cann_ops_blas.h` L606/L616 已有声明，无需改公共头 |
| 3 | 参数规格 | §2.4 固定表；指针数组本体 Device 内存，alpha/beta Host 内存 |
| 4 | 支持数据类型 | S=FLOAT32；H=FLOAT16（aclblasHalf） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → arch22（ascend910b3），落 `blas/gemm_batched/arch22/` |
| 6 | 精度要求 | S：FLOAT32 阈值 rtol=atol=2^-13、max_abs≤1e-2 或 32ULP；H：FLOAT16 阈值 rtol=atol=2^-10、max_abs≤1e-1 或 32ULP；matched_ratio≥0.99（§3.2，可溯源） |
| 7 | 性能预期 | kernel-to-kernel：`msprof op` Task Duration ≤ `gpu_ms` × 1.25（0.8 倍 GPU 标杆，§3.3，可溯源） |
| 8 | 代码架构选型 | host 侧一次性读入指针数组 + 按 batch×分块二维分核的单 launch 方案；可行性依据：arch35 同构实现已在仓、arch22 单矩阵 GEMM 基线达标。已排除死路——每 batch 单独 launch（下发开销随 batchCount 线性放大）、host 侧展开指针数组为连续大矩阵（batch 地址不保证连续/等距） |
| 9 | 评测来源 | golden：S=cblas_sgemm 逐 batch；H=FP32(float64 累加) 收窄 FP16；性能 baseline=`gpu_baseline.csv`（cuBLAS 同名接口 kernel-only） |

# 详细设计（required）

## 算子分析

### 数学公式

见"数学公式"节。batch 内标准三重循环 GEMM，K 维分块累加；batch 间纯并行。

### 支持数据类型

| 接口 | dtype | 计算口径 |
| --- | --- | --- |
| aclblasSgemmBatched | FLOAT32 | FP32 FMA |
| aclblasHgemmBatched | FLOAT16 | UB 内 FP16 存取、FP32 累加（对标 cuBLAS Hgemm 语义），写回转 FP16 |

### 支持形状

| 参数 | 形状约束 | 退化 |
| --- | --- | --- |
| A[i]/B[i] | lda×k（N）/ lda×m（T）等，带填充 | m=0/n=0 no-op；k=0 只做 C=beta*C |
| C[i] | ldc×n，逻辑 m×n，padding 列哨兵保护 | beta=0 不读旧值（C 可含 NaN） |
| batchCount | 0..65536 | =0 合法 no-op，不引用任何数组 |

## 算子实现

### 实现方案

#### 总体架构与数据流

```
host: 校验(handle→trans→m/n/k→lda/ldb/ldc→batchCount→alpha/beta/数组本体)
      → quick return 判定 → tiling(batch×分块二维切分) → 单次 launch
数据流: GM(Aarray/Barray/Carray 指针数组) → 核内按 batch id 取址
        → GM(A[i]/B[i] 分块) → L1/L0 double buffer → FMA 累加 → GM(C[i]) 原地写回
```

| kernel 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| S 主路径 | batchCount>0 且 m,n>0 且非 quick return | FP32 批量 GEMM |
| H 主路径 | 同上 | FP16 存取 + FP32 累加批量 GEMM |
| beta 缩放路径 | k=0 或 alpha=0 且 beta≠1 | 仅 C[i]=beta·C[i]，不读 A/B |
| quick return | m=0 或 n=0 或 batchCount=0；或(k=0/alpha=0 且 beta=1) | 不 launch |

#### host 侧设计

- **分核策略**：二维分核——batch 维 × C[i] 的 m×n 分块网格联合编址，优先拉满核数；大 batchCount（最高 65536，小矩阵）时 batch 维主导，小 batch 大矩阵时分块网格主导；
- **数据分块和内存优化策略**：指针数组一次性搬入 UB/寄存器，核内按 batch id 直接取址（避免 host 解引用 Device 指针）；A/B 分块 L1/L0 double buffer；H 路径 FP16 搬运带宽减半、FP32 累加保精度；
- **tilingkey 规划策略**：key 由 {dtype(S/H), transa, transb, 尾块标志} 编码，编译期展开转置读取与尾块分支；
- **数据检测**：按 §2.5 校验顺序：handle→trans 枚举→m/n/k→lda/ldb/ldc→batchCount→alpha/beta 与三数组本体；结构合法后 quick return；不对 Device 数组元素做元素级扫描。

#### kernel 侧设计

- Init：按全局核号解码 (batch id, 分块坐标)，从指针数组取 A[i]/B[i]/C[i] 基址；
- Process：CopyIn（A[i]/B[i] 分块）→ Compute（FMA 累加，H 路径 FP32 累加）→ CopyOut（beta 折算后原地写回 C[i]，仅逻辑 m×n 区域）；
- workspace：无额外 GM workspace。

#### 性能优化要点（终态）

- 优化项→预期效果：单 launch 二维分核→消除逐 batch 下发开销；指针数组核内取址→支持任意 batch 布局零拷贝；H 路径 FP16 搬运→带宽减半、FP32 累加保精度。实测按 §3.3 口径（warmup 5、有效采样 10、msprof op）回填自测报告。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2（800T A2/910B3） | ✔（本任务目标） |
| Atlas A3 | ✔（本任务目标） |
| Atlas 950PR | arch35 已有实现，不涉及 |

## 算子约束限制

- 每个 batch 仅支持列主序带填充布局，不支持广播、任意非连续布局（§2.5）；
- 不使用 AutoContiguous/ViewCopy；C[i] 原地覆写，C[i] 之间及与 A[j]/B[j] 不重叠（调用方保证）；
- 指针数组元素有效性由调用方保证，host 不做元素级扫描；
- 非确定性：batch 调度与 K 归约顺序随形状变化，验收按阈值判定；异步执行语义同 §2.5。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | S：FLOAT32 阈值 rtol=atol=2^-13、max_abs≤1e-2 或 32ULP；H：FLOAT16 阈值 rtol=atol=2^-10、max_abs≤1e-1 或 32ULP；matched_ratio≥0.99；golden=S=cblas_sgemm 逐 batch、H=FP32(float64 累加)收窄 FP16 | §3.2 + mixed_tolerance_standard |
| 性能标准 | msprof op kernel Task Duration ≤ gpu_ms×1.25（cuBLAS 同名接口 kernel-only 基线）；[PERF] avg_us 仅交叉验证 | §3.3 |

## 测试概览

- 用例分类：精度全量 1000 条（S 500 + H 500）；性能子集 200 条（各 100，算子×规模档分层，含约 12% 非默认标量）；batchCount 覆盖 0/1/2/3/8/16/64/256/1024/4096/16384/65536（后 4 档用 1×1×1 小矩阵）；
- 特殊场景：m/n/k=0、batchCount=0、alpha=0、beta=0 且 C=NaN、k=0 且 beta≠1、OP_C≡OP_T 一致性、ldc padding 哨兵、FP16 次正规/接近上限/Inf-NaN；
- 性能口径：msprof op 正式判据（warmup 5 + 采样 10 取平均）；
- 复现入口：`bash build.sh --ops=sgemm_batched,hgemm_batched --soc=ascend910b3`，自验包 readme 步骤。

## 兼容性分析

- 新增 arch22 路径不修改公共头（两声明均已存在），不影响 arch35 及其他产品线；
- `blas/gemm_batched/arch22/` 新增 S/H 文件对既有 `cgemm_batched_*` 无行为改动；测试目标在 `test/gemm_batched/sgemm_batched/CMakeLists.txt` 独立注册；
- README.md 补充 S/H 的 Atlas A2/A3 支持说明，无兼容性风险。