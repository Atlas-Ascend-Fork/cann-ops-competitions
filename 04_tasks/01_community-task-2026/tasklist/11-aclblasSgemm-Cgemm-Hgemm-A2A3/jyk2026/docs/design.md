# aclblasSgemm / aclblasCgemm / aclblasHgemm 算子设计文档（Atlas A2/A3）

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：S=FLOAT32；C=COMPLEX64（实部/虚部各 FLOAT32）；H=FLOAT16（`aclblasHalf`，IEEE FP16 位模式）
> 合入位置：`ops-blas/blas/gemm/arch22/`；测试：`ops-blas/test/gemm/arch22/`（复用既有 `gemm_test.cpp`/`gemm_test.csv` 扩展用例，不新建 `test/gemm/sgemm/arch22/`）
> 任务书：`aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——gemm 类 3 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/gemm/arch22/`；测试代码 → `test/gemm/arch22/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入（§4）。

## 背景介绍

### 算子功能定位

单矩阵通用矩阵乘三件套：`C = alpha·op(A)·op(B) + beta·C`，op(A) m×k、op(B) k×n、C m×n，列主序。S=FP32 实数、C=COMPLEX64 复数（直接复数乘加 `(a+bi)(c+di)=(ac−bd)+(ad+bc)i`，不走 3M）、H=FP16 实数（FP16 存取、FP32 累加）。

### 业界对标

- 对标接口：cuBLAS `cublasSgemm` / `cublasCgemm` / `cublasHgemm`（S/C 语义参考 Netlib `sgemm`/`cgemm`；H 以 cuBLAS 语义与 FP32 golden 口径为准）。
- 语义差异点：非法 trans 枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（cuBLAS 返回 INVALID_VALUE，有意差异）；alpha/beta 固定 Host 内存。
- Cgemm 中 `OP_C` 为共轭转置、`OP_T` 为非共轭转置，二者结果不同；S/H 为实数算子，`OP_C` 等价 `OP_T`。Cgemm 与 `aclblasCgemm3m` 参数序列相同但实现语义不同（后者为 3M 分解），**禁止把 Cgemm 实现混入 3M 文件**（§2.2）。

### 工程形态

- 工程模式：Host C API + Ascend C kernel 直调（handle 绑定 stream，§2.2）。
- 代码落位：`blas/gemm/arch22/gemm_host.cpp` 当前仅导出 `aclblasSgemm`；本任务在同一目录补齐 Cgemm/Hgemm 入口与 kernel 路径。
- 仓内近亲：`blas/gemm/arch35/cgemm_host.cpp`（Cgemm 950PR）、`gemm_host.cpp`（Sgemm）、`hgemmStridedBatched_host/kernel.cpp`（FP16 计算路径）；本任务的 arch22 `aclblasSgemm` 实现即为 S 基线。
- 声明状态：S（L575）/C（L579）已存在；**H（非 batch）需在 `include/cann_ops_blas.h` 新增声明**（紧随 Cgemm 之后）。

### 接口规格

三者参数序列分别与 cuBLAS 同名接口逐参数对应：

```cpp
aclblasStatus_t aclblasSgemm(handle, transa, transb, m, n, k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc);
aclblasStatus_t aclblasCgemm(handle, transa, transb, m, n, k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
aclblasStatus_t aclblasHgemm(handle, transa, transb, m, n, k,   // 新增声明
    const aclblasHalf* alpha, const aclblasHalf* A, int lda,
    const aclblasHalf* B, int ldb, const aclblasHalf* beta,
    aclblasHalf* C, int ldc);
```

参数规格固定表（§2.4 逐行）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | - | - | 非空，否则 HANDLE_IS_NULLPTR | - |
| transa/transb | op(A)/op(B) 类型 | attr int | - | {OP_N, OP_T, OP_C}；C 中 C=共轭转置、S/H 中 C≡T | - |
| m/n/k | 三维规模 | attr int | - | ≥0；m或n=0 no-op；k=0 只做 C=beta*C | - |
| alpha/beta | 标量（Host 内存） | scalar | S=FP32；C=COMPLEX64；H=FP16 | 非空 | - |
| A/B | 输入矩阵，列主序 | tensor | S=FP32；C=COMPLEX64；H=FP16 | 连续带填充布局；不与 C 重叠 | op(A) m×k、op(B) k×n |
| lda/ldb/ldc | 前导维度 | attr int | - | lda≥max(1,m或k)、ldb≥max(1,k或n)、ldc≥max(1,m) | - |
| C | 输出，原地覆写 | tensor | 同上 | 仅写逻辑 m×n 区域 | m×n |

### 数学公式

`C = alpha·op(A)·op(B) + beta·C`；Cgemm 直接复数乘加，不采用 3M 分解。

# 需求分析（required）

## 需求描述

以 Ascend C 在 arch22 上实现 S/C/H 三个单矩阵 GEMM：共享同一分块-流水骨架、按 dtype 分派 kernel 路径（S=FP32 FMA，C=复数 FMA，H=FP16 存取 + FP32 累加），支持 N/T/C 全组合、带填充布局与完整 quick return 语义；S 按 FLOAT32 阈值、H 按 FLOAT16 阈值判定精度，性能达 GPU kernel 标杆 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | 三算子共享 GEMM 骨架，仅 dtype 与 OP_C 语义不同；复用 arch22 Sgemm 基线的分核/流水，C/H 只替换数据通路 |
| 2 | 接口原型 | S/C 已有声明无需改；H 新增声明于 `include/cann_ops_blas.h`（同族风格） |
| 3 | 参数规格 | §2.4 固定表，alpha/beta Host 内存，A/B/C Device 列主序 |
| 4 | 支持数据类型 | S=FLOAT32、C=COMPLEX64、H=FLOAT16 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → arch22（ascend910b3），落 `blas/gemm/arch22/` |
| 6 | 精度要求 | S：FP32 阈值 rtol=atol=2^-13、max_abs≤1e-2 或 32ULP；C：分量同 FP32；H：FP16 阈值 rtol=atol=2^-10、max_abs≤1e-1 或 32ULP；matched_ratio≥0.99（§3.2，可溯源） |
| 7 | 性能预期 | kernel-to-kernel：`msprof op` Task Duration ≤ `gpu_ms` × 1.25（0.8 倍 GPU 标杆，§3.3，可溯源） |
| 8 | 代码架构选型 | 单套 GEMM 骨架 + dtype 模板化 kernel 路径；可行性依据：arch22 Sgemm 基线已在仓、arch35 同构 C/H 实现可参考。已排除死路——Cgemm 走 3M 分解（任务书明确不走且禁混入 3M 文件）、H 全 FP16 累加（精度不足 FP32 累加口径） |
| 9 | 评测来源 | golden：S=cblas_sgemm、C=cblas_cgemm（Netlib 3.10.0）、H=FP32(float64 累加)收窄 FP16；性能 baseline=`gpu_baseline.csv`（cuBLAS 同名接口 kernel-only） |

# 详细设计（required）

## 算子分析

### 数学公式

`C = alpha·op(A)·op(B) + beta·C`；Cgemm 分量式：实部 = ac−bd、虚部 = ad+bc（a,b,c,d 为对应实虚部），OP_C 在搬运阶段做共轭。

### 支持数据类型

| 接口 | dtype | 计算口径 |
| --- | --- | --- |
| aclblasSgemm | FLOAT32 | FP32 FMA |
| aclblasCgemm | COMPLEX64 | 复数 FMA（实虚部 FP32 分量），OP_C 共轭 |
| aclblasHgemm | FLOAT16 | UB 内 FP16 存取、FP32 累加，写回转 FP16 |

### 支持形状

| 矩阵 | 形状约束 | 退化 |
| --- | --- | --- |
| op(A) | m×k；物理 lda×k（N）或 lda×m（T/C） | m=0/n=0 no-op；k=0 只做 C=beta*C |
| op(B) | k×n；物理 ldb×n（N）或 ldb×k（T/C） | 同上 |
| C | m×n；物理 ldc×n，padding 列哨兵保护 | beta=0 不读旧值（C 可含 NaN） |

## 算子实现

### 实现方案

#### 总体架构与数据流

```
host: 校验(handle→trans→m/n/k→lda/ldb/ldc→alpha/beta/A/B/C) → quick return 判定
      → dtype 分派 → tiling → launch
数据流: GM(A/B 分块) → L1 → L0 double buffer → UB FMA 累加 → beta 折算 → GM(C) 写回
```

| kernel 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| S 路径 | aclblasSgemm | FP32 批量 FMA |
| C 路径 | aclblasCgemm | 复数 FMA（4 实数乘 2 加/元素对），OP_C 搬运期共轭 |
| H 路径 | aclblasHgemm | FP16 搬运 + FP32 累加，写回转 FP16 |
| beta 缩放路径 | k=0 或 alpha=0 且 beta≠1 | 仅 C=beta·C，不读 A/B |
| quick return | m=0 或 n=0；或(k=0/alpha=0 且 beta=1) | 不 launch |

#### host 侧设计

- **分核策略**：按 C 的 m×n 分块网格满核均分，余块尾核处理；大 K 时按 K 切段片上累加，不跨核归约；
- **数据分块和内存优化策略**：L1/L0 double buffer；H 路径 FP16 搬运带宽减半；C 路径实虚部交织访存按列块对齐；beta 折算在写回阶段完成（beta=0 直写）；
- **tilingkey 规划策略**：key 由 {dtype(S/C/H), transa, transb, 尾块标志} 编码，编译期展开转置/共轭读取分支；
- **数据检测**：按 §2.5 校验顺序：handle→trans 枚举→m/n/k→lda/ldb/ldc→alpha/beta/A/B/C 指针；结构合法后 quick return；不做 Device 元素级扫描。

#### kernel 侧设计

- Init：解析 tiling、按 key 选择 dtype/转置分支；
- Process：CopyIn（A/B 分块）→ Compute（dtype 分派 FMA；C 路径 OP_C 在 CopyIn 期做实部取反）→ CopyOut（beta 折算、仅写逻辑 m×n 区域）；
- workspace：无额外 GM workspace。

#### 性能优化要点（终态）

- 优化项→预期效果：三 dtype 共享骨架→一套流水优化三算子受益；H 路径 FP16 搬运→带宽减半；double buffer→搬运与 FMA 重叠。实测按 §3.3 口径（warmup 5、有效采样 10、msprof op）回填自测报告。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2（800T A2/910B3） | ✔（本任务目标） |
| Atlas A3 | ✔（本任务目标） |
| Atlas 950PR | arch35 已有实现，不涉及 |

## 算子约束限制

- 仅支持连续列主序带填充布局，不支持任意非连续布局与 broadcast（§2.5）；
- 不使用 AutoContiguous/ViewCopy；C 原地覆写且与 A/B 不重叠；
- 非确定性：K 归约顺序随形状变化，验收按阈值判定；异步执行语义同 §2.5。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | S/C：FLOAT32 分量阈值 rtol=atol=2^-13、max_abs≤1e-2 或 32ULP；H：FLOAT16 阈值 rtol=atol=2^-10、max_abs≤1e-1 或 32ULP；matched_ratio≥0.99；golden=S=cblas_sgemm、C=cblas_cgemm（实虚部分别比对）、H=FP32(float64 累加)收窄 FP16 | §3.2 + mixed_tolerance_standard |
| 性能标准 | msprof op kernel Task Duration ≤ gpu_ms×1.25（cuBLAS 同名接口 kernel-only 基线）；[PERF] avg_us 仅交叉验证 | §3.3 |

## 测试概览

- 用例分类：在既有 `test/gemm/arch22/gemm_test.cpp`/`gemm_test.csv` 基础上扩展 S/C/H 用例（精度全量与性能子集按任务书口径分层），不新建 sgemm 子目录；负向覆盖非法枚举、负值规模、lda/ldb/ldc 越界、空指针；
- 特殊场景：m/n/k=0、alpha=0、beta=0 且 C=NaN、k=0 且 beta≠1、Cgemm 的 OP_C 与 OP_T 结果不同验证（S/H 中二者一致）、ldc padding 哨兵、FP16 次正规/接近上限/Inf-NaN；
- 性能口径：msprof op 正式判据（warmup 5 + 采样 10 取平均）；
- 复现入口：`bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`，自验包 readme 步骤。

## 兼容性分析

- H 为新增声明（append 到 `include/cann_ops_blas.h`），S/C 声明不变，无既有符号修改；
- `blas/gemm/arch22/gemm_host.cpp` 的增量入口对既有 `aclblasSgemm` 路径无行为改动；Cgemm 实现独立于 3M 文件；
- README.md 补充三算子的 Atlas A2/A3 支持说明，无兼容性风险。