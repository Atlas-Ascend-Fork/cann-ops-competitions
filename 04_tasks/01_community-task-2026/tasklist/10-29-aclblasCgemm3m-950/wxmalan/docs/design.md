# 需求背景

## 需求来源

昇腾 CANN 算子社区竞赛任务 —— Atlas 950PR 社区算子开发任务（2026），任务书文件：`aclblasCgemm3m_Atlas950PR_task_doc.md`。

## 背景介绍

cuBLAS 提供了 `cublasCgemm3m` 接口，使用 3m 算法优化复数矩阵乘法。3m 算法的核心思想是将一次复数乘法 `(a+bi)(c+di)` 分解为三次实数乘法，从而减少计算量。

当前昇腾 CANN ops-blas 仓库中已有以下相关实现：
- **aclblasSgemm3m**（`blas/gemm3m/arch35/`）：实数 3m GEMM，输入为已分解的 m×3k 和 3k×n 实数矩阵，算子内部仅做三次实数 GEMM 累加。
- **aclblasCgemm**（`blas/gemm/arch35/cgemm_host.cpp`）：复数 GEMM，已实现 deinterleave（含 ThreeProducts 3m 分解模式）和 epilogue（3-product 重组）模板。

本次任务要求实现 **aclblasCgemm3m**，即复数 3m 算法 GEMM 算子，对标 `cublasCgemm3m`。与 sgemm3m 不同，cgemm3m 的输入是标准复数矩阵（A 为 m×k，B 为 k×n），算子内部需完成：
1. 复数到实数的 3m 分解（deinterleave）
2. 三次实数 GEMM 计算
3. 结果重组与 alpha/beta 缩放（epilogue）

现有 cgemm 实现已具备 ThreeProducts 模式的 deinterleave 和 epilogue 模板，cgemm3m 应在此基础上复用改造，而非独立实现。

# 需求分析

## 需求描述

在昇腾 Atlas 950PR（arch35）平台上，基于 Ascend C 开发单精度复数（complex64）3m 算法 GEMM 算子 `aclblasCgemm3m`，功能对标 cuBLAS `cublasCgemm3m`，精度和性能达到任务书验收标准。

## 需求拆解

1. **接口声明**：在 `include/cann_ops_blas.h` 中新增 `aclblasCgemm3m` API 声明，参数类型为 `aclblasComplex*`，alpha/beta 为复数标量。
2. **算子实现**：
   - 复用 `blas/gemm/arch35/cgemm_deinterleave_common.h` 中的 ThreeProducts 3m 分解模板
   - 复用 `blas/gemm/arch35/cgemm_epilogue_kernel.cpp` 中的 3-product epilogue 重组逻辑
   - 基于现有 `cgemm_host.cpp` 架构改造，强制使用 ThreeProducts 路径（现有 cgemm 中 alpha=(1,0) 且 beta=(0,0) 时已走此路径，cgemm3m 需扩展到全 alpha/beta 范围）
   - 支持 OP_N / OP_T / OP_C 全转置组合（9 种）
   - 支持共轭转置（OP_C 需对虚部取反）
3. **边界处理**：
   - m=0 或 n=0：quick return
   - k=0：在 stream 上完成 C = beta * C（不依赖 host 同步拷贝，不依赖 workspace）
4. **无 workspace 依赖**：算子内部管理所有临时内存，不要求调用方通过 `aclblasSetWorkspace` 预分配
5. **测试工程**：
   - 目录：`test/gemm/arch35/`（非 `test/gemm3m/`）
   - 支持复数标量参数（alpha_real/alpha_imag, beta_real/beta_imag）
   - 1200 条精度用例 + 200 条性能用例（CSV 驱动）
   - 精度验收：逐元素判定 `|actual - golden| <= atol + rtol × |golden|`（FLOAT32 分量：atol=rtol=2^-13≈1.22e-4），matched_ratio >= 0.99，max_abs_error <= 1e-2
   - 性能验收：单 case >= 0.4x GPU，平均 >= 0.5x GPU
6. **确定性计算**：同输入逐位一致

# 详细设计

## 算子分析

### 数学公式

**标准复数 GEMM**：
```
C = alpha * op(A) * op(B) + beta * C
```

其中 A、B、C 为复数矩阵，alpha、beta 为复数标量。

**3m 分解公式**：

对于复数乘法 `(a + bi)(c + di)`，3m 算法分解为三次实数乘法：
```
P0 = (Ar + Ai) * Br        // 实部和 × B实部
P1 = Ar * (Bi - Br)        // A实部 × B虚部减实部
P2 = Ai * (Br + Bi)        // A虚部 × B实部加虚部
```

结果重组：
```
Re(C) = P0 - P2            // = Ar*Br - Ai*Bi
Im(C) = P0 + P1            // = Ar*Bi + Ai*Br
```

对于矩阵乘法，上述逐元素分解推广为三次实数矩阵乘法：
```
P0 = (Ar + Ai) * Br        // m×n 实数 GEMM
P1 = Ar * (Bi - Br)        // m×n 实数 GEMM
P2 = Ai * (Br + Bi)        // m×n 实数 GEMM
```

### 支持数据类型

| 参数 | 数据类型 | 说明 |
|------|----------|------|
| A, B, C | COMPLEX64 (complex64) | 每个元素 = 2 × FLOAT32 (实部 + 虚部)，interleaved 存储 |
| alpha, beta | COMPLEX64 (complex64) | 复数标量，通过指针传入 |
| 内部计算 | FLOAT32 | 实数 GEMM 部分使用 FP32 cube 指令 |

### 支持形状

- m, n, k >= 0
- m=0 或 n=0：quick return，不执行计算
- k=0：仅执行 C = beta * C，不引用 A/B
- beta=0 时 C 的原值不被读取
- 矩阵维度上限受 L1 缓冲和内存预算约束（单用例 host 侧 <= 512MB）
- 性能方阵扫描范围：256 ~ 4096

## 算子实现

### 实现方案

整体架构复用现有 `blas/gemm/arch35/cgemm_host.cpp` 的 Host 端框架。现有 cgemm 已同时支持 TwoProducts（3m）和 FourProducts（Gauss）两条路径：当 alpha=(1,0) 且 beta=(0,0) 时走 ThreeProducts 路径，否则走 FourProducts 路径。cgemm3m 需强制使用 ThreeProducts 路径，并在此基础上扩展 alpha/beta 复数缩放支持。核心流程分为三个阶段：

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Deinterleave   │ ──> │  3x Real GEMM    │ ──> │    Epilogue      │
│  (复数→3路实数)   │     │  (cube 计算)      │     │ (重组+alpha/beta) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

#### host侧设计：

**总体流程**：
1. 参数校验（维度、转置枚举、lda/ldb/ldc 合法性）
   - 非法转置枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`（与任务书 §2.3/§2.4 一致）
2. 边界处理：
   - m=0 或 n=0：quick return，返回 SUCCESS，不写 C
   - k=0 或 alpha=(0,0)：走 beta 缩放路径（C = beta * C），不引用 A/B（与 cgemm `HandleCgemmAlphaZero` 一致）
   - beta=(0,0) 时不读取 C 的原值
3. Tiling 参数计算
4. 分配片上 workspace（deinterleave 后的实/虚部分离矩阵 + 3 个 product 中间结果）
5. Launch kernel：deinterleave → 3x GEMM → epilogue

**tiling策略：**

##### 1. 分核策略：

- 复用 cgemm 的 tiling 框架，基于 `TilingData` 结构体传递参数
- Cube core 分配：arch35 的 DAV_3510 有 28 个 AIC (Cube) core（AIC:AIV = 1:2），每个 core 处理一个 tiling 分块。核数通过运行时 API `GetAicCoreCount()` 获取，不硬编码
- GEMM 分块尺寸：BASE_M=128, BASE_N=128, BASE_K=64（与 cgemm/sgemm3m 一致）
- 3 次实数 GEMM 串行执行（共享同一组分块参数），每次处理一对实部/虚部组合

##### 2. 数据分块和内存优化策略：

- **L1 缓冲**：512KB，2 路 ping-pong 双缓冲，用于加载 A/B 的分块数据
- **L0 缓冲**：L0A 64KB（存放左矩阵，FRACTAL_NZ 格式）+ L0B 64KB（存放右矩阵，FRACTAL_ZN 格式）+ L0C 256KB（存放计算结果，FRACTAL_NZ 格式）
- **片上 workspace**：
  - Deinterleave 输出：A 的 3m 预变换数据（realOut=Ar+Ai, imagOut=Ar, auxiliary=Ai，共 3 × m×k × 4B）+ B 的 3m 预变换数据（realOut=Br, imagOut=Bi-Br, auxiliary=Br+Bi，共 3 × k×n × 4B）
  - 3 个 Product 结果：t1/t2/t3（各 m×n × 4B）
  - 总计 workspace 由算子内部管理，通过 handle 的默认 workspace（`EnsureDefaultWorkspace`）分配
- **数据搬运**：使用 `DataCopy` 指令实现 L1→L0 的 ping-pong 流水搬运，隐藏搬运延迟

##### 3. tilingkey规划策略：

- tiling key 编码维度信息（m/n/k 的分块数量）和转置组合
- 3m 路径新增 `ThreeProducts` 标记（区别于 cgemm 的 `FourProducts`）
- 转置组合共 9 种（OP_N/OP_T/OP_C × OP_N/OP_T/OP_C），通过 tiling key 分发到不同 kernel 分支
- 共轭转置（OP_C）在 deinterleave 阶段对虚部取反，后续计算与 OP_T 路径一致

#### kernel侧设计：

**Deinterleave Kernel**：
- 输入：interleaved complex64 矩阵 A（m×k）和 B（k×n）
- 输出：3m 分解后的 FLOAT32 矩阵（非直接的实部/虚部分离）
  - 对于 A：realOut = Ar+Ai, imagOut = Ar, auxiliary = Ai（按 `cgemm_deinterleave_common.h` L37-40 的 ThreeProducts 模板）
  - 对于 B：realOut = Br, imagOut = Bi-Br, auxiliary = Br+Bi
  - 注意：ThreeProducts 模式下输出并非直接的 Ar/Ai/Br/Bi，而是 3m 公式的预变换数据
- 共轭处理：当 transa=OP_C 时，对 Ai 取反（即 imag 分量取反后再参与 3m 预变换）；当 transb=OP_C 时，对 Bi 取反
- 使用 AIV（向量）指令完成 interleaved→separated 的数据重排
- 参考实现：`cgemm_deinterleave_common.h` L37-44 的 ThreeProducts 模板

**Compute Kernel（3x Real GEMM）**：
- 调用现有 `gemm_kernel`（FP32 实数 GEMM cube kernel）执行三次矩阵乘：
  - P0 = (Ar+Ai) × Br
  - P1 = Ar × (Bi-Br)
  - P2 = Ai × (Br+Bi)
- 每次 GEMM 使用 tensor_api 的 `Te::Mmad` 指令
- 三次 GEMM 串行调度，共享 tiling 参数

**Epilogue Kernel**：
- 输入：t1, t2, t3（各 m×n FLOAT32，3 个 product 中间结果，可能按 kPartitions 分段）
- 输出：interleaved complex64 矩阵 C
- k-partitions 累加：当 K 维度被切分为多个 partition 时，epilogue 逐 partition 读入 t1/t2/t3 并累加 Re/Im 中间结果（参考 `cgemm_epilogue_kernel.cpp` L74-85 的 `RunPlanarOutputTile`）
- 重组公式（对每个 k-partition 累加后）：
  ```
  Re(product) = t1 - t3          // = P0 - P2
  Im(product) = t1 + t2          // = P0 + P1
  ```
- Alpha/Beta 缩放（复数运算）：
  ```
  C = (alpha_r + alpha_i*j) * (Re + Im*j) + (beta_r + beta_i*j) * (C_r + C_i*j)
  ```
  展开为：
  ```
  C_r = alpha_r*Re - alpha_i*Im + beta_r*C_r_old - beta_i*C_i_old
  C_i = alpha_r*Im + alpha_i*Re + beta_r*C_i_old + beta_i*C_r_old
  ```
- 使用 SIMT 向量化处理，支持 `AlphaZero`/`BetaZero` 模板特化优化（跳过 alpha 或 beta 计算分支）
- 当 alpha=(1,0)、beta=(0,0) 且 ThreeProducts 时，可走 `LaunchPlanarOutput` 快速路径（直接重组写回，跳过 SIMT epilogue）
- 参考实现：`cgemm_epilogue_kernel.cpp` L143-166（SIMT epilogue）、L61-100（PlanarOutput 快速路径）

**k=0 边界处理**：
- 当 k=0 时，跳过 deinterleave 和 compute 阶段
- 直接执行 epilogue：C = beta * C（复数 beta 缩放）
- 使用 AIV kernel 在 stream 上完成，不依赖 host 同步拷贝
- 参考实现：`cgemm_host.cpp` 中 `HandleCgemmAlphaZero` 的 beta 缩放路径

**alpha=0 快速路径**：
- 当 alpha=(0,0) 时，不引用 A/B，直接执行 C = beta * C（与 k=0 共用同一 beta 缩放路径）
- 参考实现：`cgemm_host.cpp` L401-403 `LaunchCgemmKernel` 中的 alpha=0 判断

**Onchip Input 快速路径**：
- 当 alpha=(1,0)、beta=(0,0) 且 k 较小时（k <= CGEMM_HALF_L0_FLOAT_ELEMENTS / CGEMM_FUSED_TILE），可走 onchip input 融合路径
- 该路径将 deinterleave + 3x GEMM + epilogue 融合为单个 kernel，减少全局内存访问
- 参考实现：`cgemm_host.cpp` 中 `UseCgemmOnchipInput` 和 `LaunchCgemmOnchipInput`

## 支持硬件

| 硬件平台 | 芯片版本 | 支持状态 |
|----------|----------|----------|
| Atlas 950PR | arch35 (DAV_3510) | ✅ 支持 |

**DAV_3510 关键硬件参数**（参考 Ascend C 官方文档 npu_arch_3510）：

| 参数 | 规格 |
|------|------|
| AIC (Cube) Core 数量 | 28（运行时通过 `GetAicCoreCount()` 获取） |
| AIV (Vector) Core 数量 | 56（AIC:AIV = 1:2） |
| UB | 256KB |
| L1 Buffer | 512KB |
| L0A Buffer | 64KB（FRACTAL_NZ） |
| L0B Buffer | 64KB（FRACTAL_ZN） |
| L0C Buffer | 256KB（FRACTAL_NZ） |
| Cube 单拍算力 | FP16 16×16 矩阵乘 |
| 核间通信 | SSBuffer（AIC↔AIV），支持 1:1 和 1:2 模式 |
| SIMT | 每个 AIV 4 个 Warp Scheduler，128KB Register File |

## 算子约束限制

1. **数据类型限制**：仅支持 COMPLEX64（实部/虚部各为 FLOAT32），不支持 COMPLEX128 或其他精度
2. **存储格式**：仅支持列主序（column-major），不支持行主序
3. **转置操作**：支持 OP_N / OP_T / OP_C，传入其他枚举值返回 `ACLBLAS_STATUS_INVALID_VALUE`
4. **内存约束**：单用例 host 侧内存占用 <= 512MB，限制了最大可处理的矩阵尺寸
5. **Workspace**：算子内部管理 workspace，不依赖调用方通过 `aclblasSetWorkspace` 预分配外部 workspace
6. **确定性**：相同输入保证逐位一致的确定性输出
7. **3m 算法精度**：由于 3m 分解引入额外的加减法运算，在极端数值条件下可能比标准 4-product 算法有略大的舍入误差，但仍在 FP32 精度验收范围内
8. **代码提交规范**：提交的文件组织结构须与仓内同类算子保持一致，文件数量应在合理范围内，不得包含冗余的代码、测试或编译文件（参见任务书 §5 典型反例1/2）。复用现有 cgemm 模板时，应在原有文件基础上扩展或新增必要的独立文件，而非复制出多余的副本。算子代码提交至 `blas/gemm/arch35/`，测试代码提交至 `test/gemm/arch35/`（参见任务书 §5）

# 可维可测分析

## 精度标准/性能标准

### 精度标准

任务书 §3.2 引用生态算子开源精度标准（混合容差标准），COMPLEX64 按 FLOAT32 分量（实部、虚部分别）判定。

**逐元素通过条件**：`|actual - golden| <= atol + rtol × |golden|`

| 指标 | 阈值 | 说明 |
|------|------|------|
| rtol（相对容差） | 2^-13 ≈ 1.22e-4 | FLOAT32 分量相对容差 |
| atol（绝对容差） | 2^-13 ≈ 1.22e-4 | FLOAT32 分量绝对容差 |
| required_matched_ratio | >= 0.99 | 满足逐元素通过条件的元素占比 |
| max_abs_error_limit | <= 1e-2 | 全元素最大绝对误差上限 |

**Inf/NaN 特殊值判定规则**（任务书 §3.2 第 4 条）：
- 当 golden 或 NPU 输出中出现 Inf/NaN 时，不适用上述判定式
- golden 与 actual 同为 NaN、同为同号 Inf，或一方为 NaN 而另一方为 ±Inf，视为等效，判定通过
- 一方为有限值、另一方为 Inf/NaN，判定不通过
- 非有限值元素不计入 matched_ratio 与 max_abs_error 统计，需单独记录并全部按上述规则判定通过

**验收判定**：有限值元素满足 matched_ratio >= 0.99 且 max_abs_error <= 1e-2，且非有限值元素全部判定通过。使用任务书提供的 `verify_accuracy.py` 脚本自动验收。

### 性能标准

| 指标 | 阈值 | 说明 |
|------|------|------|
| 单 case 性能比 | >= 0.4 | 单条性能用例的 NPU 时间 / GPU 时间 <= 1/0.4 = 2.5x |
| 平均性能比 | >= 0.5 | 所有性能用例的平均 NPU 时间 / 平均 GPU 时间 <= 1/0.5 = 2.0x |

**性能基线**：基于 cuBLAS `cublasCgemm3m` 实测数据（`gpu_baseline.csv`，200 条用例），标杆环境为 NVIDIA 驱动 550.104.05、CUDA 12.4。使用任务书提供的 `verify_performance.py` 脚本自动验收。

### 测试用例覆盖

| 类别 | 前缀 | 条数 | 覆盖维度 |
|------|------|------|----------|
| L0 基础 | TC_L0 | 18 | 3×3 转置全组合 × 基础尺寸 |
| L1 方阵 | TC_SQ | 108 | 27 种方阵尺寸 × 4 组转置 |
| L1 矩形 | TC_MN | 81 | 9 组非方阵 × 9 组转置 |
| L2 标量 | TC_AB | 32 | 8 组 alpha/beta × 4 组转置 |
| L4 前导维 | TC_LD | 16 | padding(+4/+8) × 4 组转置 × 4 种尺寸 |
| L5 填充 | TC_FL | 18 | 6 种数据填充模式 |
| L5b 覆盖 | TC_CV | 32 | 中等尺寸 × 4 组转置 |
| L6 边界 | TC_ED | 32 | 0 维/k=0/beta=0/空指针/非法枚举/负维度 |
| EX 扩展 | TC_EX | 663 | 确定性采样补足 |
| PF 性能 | TC_PF | 200 | 4 条任务书典型 case + 方阵对数扫描(256~4096) |
| **合计** | | **1200** | |

## 兼容性分析

**需要兼容性分析**：

1. **与 cublasCgemm3m 的接口兼容性**：
   - 函数签名、参数语义与 cuBLAS `cublasCgemm3m` 保持一致
   - 转置枚举映射：ACLBLAS_OP_N/OP_T/OP_C 对应 CUBLAS_OP_N/OP_T/OP_C
   - 数据布局：列主序 interleaved complex64

2. **与现有 cgemm 实现的兼容性**：
   - cgemm3m 复用 cgemm 的 deinterleave 和 epilogue 模板，不修改 cgemm 原有代码路径
   - 新增代码以独立文件或在现有文件中以条件编译/模板参数区分，提交至 `blas/gemm/arch35/` 目录，不包含冗余文件（参见任务书 §5 代码提交规范）
   - cgemm（4-product Gauss）和 cgemm3m（3-product 3m）作为两个独立 API 共存

3. **与 sgemm3m 的关系**：
   - sgemm3m 的输入约定（A 为 m×3k，3m 分解由调用方完成）与 cgemm3m（A 为 m×k，算子内部完成分解）不同
   - cgemm3m 不依赖 sgemm3m 的实现，而是基于 cgemm 架构改造
   - cgemm3m 算子代码提交至 `blas/gemm/arch35/`，测试代码提交至 `test/gemm/arch35/`，与 sgemm3m（`blas/gemm3m/arch35/`）和 cgemm（`blas/gemm/arch35/`）各自独立

4. **CANN 版本兼容性**：
   - 要求 CANN >= 9.1.0 (ASC_DEVKIT_GE_9_1)
   - 使用 Ascend C tensor_api 和 AIV/AIC 指令，需确认目标版本 API 稳定性
