# aclblasGemmGroupedBatchedEx 算子设计文档（Atlas A2/A3）

| 文档版本 | 日期 | 作者/团队 | 说明 |
|---|---|---|---|
| v1.0 | 2026-10-28 | CANN lab operator engineer | 初稿：设计阶段性能可达性已估算；验收按计划执行 |

> 配套任务书：`aclblasGemmGroupedBatchedEx_A2_task_doc.md`（社区 10 月任务）
> 提交仓：代码 `gitcode.com/cann/ops-blas`；设计文档 `gitcode.com/cann/cann-competitions`
> 本文件为设计阶段文档，提交仓见 §交付与目录。

---

# 需求背景（required）

## 需求来源

- 社区任务「10 月社区任务-aclblasGemmGroupedBatchedEx 算子开发（A2A3）」，补齐 Blas 仓能力。
- 对标接口：NVIDIA cuBLAS `cublasGemmGroupedBatchedEx`。
- 验收合入仓：`gitcode.com/cann/ops-blas`，算子目录 `blas/gemm_grouped_batched_ex/arch22`；测试目录 `test/gemm_grouped_batched_ex/arch22`。

## 背景介绍

### 算子定位与数学定义

`aclblasGemmGroupedBatchedEx` 是 **BLAS Level-3 分组批量通用矩阵乘法（Grouped Batched GEMM）** 的扩展接口。与单 GEMM（`aclblasGemmEx`）区别在于：它一次性接收 `groupCount` 个分组，每组 `groupSize[g]` 个**相互独立**的矩阵乘，所有子问题在一次 API 调用内完成（这正是其性能优势来源——把 launch 开销摊薄到全部子问题上）。

数学定义（逐组 `g`、逐子问题 `i`）：

```
C_{g,i} = alpha_g * op(A_{g,i}) * op(B_{g,i}) + beta_g * C_{g,i}
```

- `op(A)` 形状 `m_g × k_g`，`op(B)` 形状 `k_g × n_g`，`C` 形状 `m_g × n_g`。
- 矩阵按**列主序（Column-Major）**存储，前导维 `lda/ldb/ldc` 逐组约束见 §参数与异常行为矩阵。
- `op` 由 `transaArray[g]` / `transbArray[g]` 决定：`N`=不转置、`T`=转置、`C`=共轭转置（实数路径下 `C` 与 `T` 等价）。

### 存储布局与一项必要的勘误

任务书 §2.4 规定 `alphaArray`/`betaArray` 为「Device 或 Host 内存」，而 cuBLAS 原接口 `cublasGemmGroupedBatchedEx` 的 `alpha`/`beta` 为 **Host 指针**（`const void* alpha`，标量，非数组指针）。这是任务书与对标接口的唯一口径偏差。设计决策见 §风险与待确认项 #3：默认按 **Host 指针**读取（与 cuBLAS 一致），并在 Host 侧即可完成 `alpha==0` 的快速分支判断；若社区要求支持 Device 指针，则增加一次 Device→Host 拷贝。

### 对标接口与现状分析

| 接口/实现 | 现状 | 与本设计的关系 |
|---|---|---|
| cuBLAS `cublasGemmGroupedBatchedEx` | 已发布，参数序列与本算子一致（handle 及参数顺序一一对应） | 功能/语义/边界行为的金标准；性能基线 `gpu_baseline.csv` 即 cuBLAS 实测 |
| ops-blas `include/cann_ops_blas.h` 已有声明 | 仓内已存在的函数原型，与现网/950PR 共用 | 本实现**复用同一声明**，禁止定义产品私有平行接口 |
| ops-blas `blas/gemm_grouped_batched_ex/arch22` | 当前为空，待开发 | 本算子代码落地目录 |
| ops-blas `test/gemm_grouped_batched_ex/arch22` | 仓内 `param.h` 读 snake_case + 长度1广播 | 与配套 CSV（camelCase + 严格长度）冲突，见 #2 |

### 功能分析（逐条枚举）

1. **分组批量计算**：`groupCount` 组，每组 `groupSize[g]` 个独立 `C=α·op(A)·op(B)+β·C`。
2. **转置语义**：`transa/transb ∈ {N,T,C}`，实数路径 `C≡T`，`op` 维度相应交换。
3. **主路径 dtype**：FLOAT32（`Atype/Btype/Ctype=ACL_FLOAT`，`computeType=ACLBLAS_COMPUTE_32F`）。扩展类型（FP16/BF16/FP8/INT8）见 #4。
4. **维度与 lda/ldb/ldc 约束**：与 cuBLAS 一致（详见 §参数与异常行为矩阵）。
5. **边界 quick-return**：
   - `groupCount==0` → `SUCCESS`，**不再校验其余参数**；
   - `m_g==0` 或 `n_g==0` 或 `groupSize[g]==0` → 合法 no-op；
   - `k_g==0` → 仅做 `C = beta·C`，不引用 A/B；
   - `alpha==0` → 跳过矩阵乘，`C = beta·C`（**位精确 EXACT**）。
6. **异步执行**：依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream。

---

# 需求分析（required）

## 需求描述（接口签名）

```cpp
aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[],
    const void* alphaArray,
    const void* const Aarray[], aclDataType Atype, const int ldaArray[],
    const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray,
    void* const Carray[], aclDataType Ctype, const int ldcArray[],
    int groupCount, const int groupSize[],
    aclblasComputeType_t computeType);
```

## 需求拆解

- **功能**：分组批量 GEMM，列主序，N/T/C 转置，原地覆写 C。
- **接口异常**：handle 空→`HANDLE_IS_NULLPTR`；非法枚举→`INVALID_ENUM`；非法 dtype/computeType→`NOT_SUPPORTED`；其余非法值→`INVALID_VALUE`（校验顺序见矩阵）。
- **精度**：FLOAT32，`rtol=atol=2⁻¹³`，`matched_ratio≥0.99`，`max_abs_error≤1e-2 或 32×ULP`；`alpha==0` 路径位精确。
- **性能**：Atlas 800T A2 (910B3) 上平均单次耗时 ≤ 1.25× `gpu_baseline.csv` 标杆（即 ≥0.8× GPU）；msprof op `Task Duration(us)`，warmup 5 + 有效采样 >10 取均值。
- **工程**：基于 ops-blas 框架，句柄式接口，kernel 直调 NPU；A2/A3 同源（arch22）。
- **自验**：1200 用例（1000 精度 + 200 性能）全过；golden 由 `cblas_sgemm`（Netlib）生成。

## 验收核对矩阵（任务书条款 → 怎么测 → 证据 → 结论）

| 任务书条款 | 要求 | 我的实现/策略 | 证据文件 | 结论 |
|---|---|---|---|---|
| §3.2 精度 | FLOAT32 `|a-g|≤atol+rtol·|g|`，matched≥0.99，max_abs≤1e-2/32ULP | 逐元素混合容差；fp32 cube 累加；NaN/Inf 按规则等效 | `verify_accuracy.py` 输出 + 自测报告 §精度 | 设计可达 ✅ |
| §3.2 alpha=0 EXACT | alpha=0 且指针非空→`C=beta·C` 位精确 | Host 读 alpha，alpha==0 跳过 matmul 直接 `C=beta*C` | TC_AB/TC_ED `alpha=0` 用例 | ✅ |
| §3.3 性能 | ≤1.25× `gpu_baseline.csv`（≥0.8×） | 单 kernel 启动吞全部子问题；cube 计算；bucketed tiling 复用；host 侧任务分布与 epilogue 条件跳过；详见 §详细设计与 §性能标杆与达标分析 | `perf_gbe` warmup5 + >10 次采样（与 gpu_ms 同口径） | 设计可达 ✅（A2/A3 同源 arch22，按计划验证） |
| §3.4 内存 | 不涉及 | 主动给出 workspace/UB 峰值上界 | 自测报告 §内存 | ✅ |
| §3.5 自验 | 1200 用例全覆盖 | 配套 CSV 全量 + GTest | `gemm_grouped_batched_ex_test.csv`（1200 行） | 计划验证 ✅ 设计可达 |
| §2.5 异常/边界 | 各状态码语义 | 校验顺序表 + TC_ED(42) | TC_ED 用例 | ✅ |
| §2.1 转置 N/T/C | 与 cuBLAS 一致 | TC_L0 + TC_MN 转置组合 | 转置用例 | ✅ |
| §2.2 工程模式 | ops-blas 框架，arch22 | 代码落地 `blas/.../arch22` | 代码 PR | ✅ |

---

# 详细设计（required）

## 算子分析

### 数学公式与依赖方向

数据流（一次 API 调用内）：

```mermaid
flowchart LR
    H[Host: 参数校验+问题扁平化] --> W[(GM Workspace: 问题描述符数组)]
    W --> K[AIC Cube Kernel: 单 launch 遍历全部子问题]
    K --> CI[CopyIn: 按 transa/transb/lda/ldb 读 A/B]
    CI --> CP[Compute: Cube Matmul, fp32 累加]
    CP --> CO[CopyOut: alpha·AB + beta·C 写回]
    CO --> S[Stream 同步后返回]
```

- **依赖方向**：各 `(g,i)` 子问题**相互独立、无跨问题依赖** → 可完全并行到 cube 核；无串行归约。
- **关键架构选择**：**单列（一次 kernel launch）处理全部 `problemCount = Σ groupSize[g]` 个子问题**，把 launch/框架固定开销摊薄到全部子问题（避免极小 case 落入单次调用固定开销主导的区间，详见 §性能标杆与达标分析）。

### 支持数据类型

| 类型 | 支持范围 | 处置 |
|---|---|---|
| `Atype/Btype/Ctype` = `ACL_FLOAT`，`computeType` = `ACLBLAS_COMPUTE_32F` | 主路径，全实现 | ✅ |
| FP16 / BF16 / FP8 / INT8 等混合组合 | 任务书 §2.1.3 提及「在支持范围内覆盖」 | 当前返回 `NOT_SUPPORTED`（与负向用例期望一致），见 #4 |

### 支持形状

- `m_g, n_g, k_g, groupCount, groupSize[g] ≥ 0`；runtime 入参（dynamic shape，由 host tiling 支持）。
- 逐组 `lda/ldb/ldc` 约束：`transa=N` 时 `lda_g ≥ max(1,m_g)`，否则 `lda_g ≥ max(1,k_g)`；`transb` 同理；`ldc_g ≥ max(1,m_g)`。
- 非连续 Tensor：**不支持**（超出 lda/ldb/ldc 语义的非连续访问）。

### 参数与异常行为矩阵（校验顺序，须与仓内同族一致）

> 严格按任务书 §2.5 顺序：`dtype 组合 → groupCount<0 → groupCount==0 早返 → groupCount>0 校验各 host 数组非空并逐组校验维度/前导维 → problemCount=ΣgroupSize 校验矩阵指针数组`。

| 校验步骤 | 条件 | 返回码 |
|---|---|---|
| 0 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 1 | `Atype/Btype/Ctype/computeType` 非受支持集 | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 2 | `groupCount < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `groupCount == 0` | `ACLBLAS_STATUS_SUCCESS`（**不再校验其余**） |
| 4 | `groupCount > 0` 时任一 host 数组指针为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `transa/transb` 取值非 `{N,T,C}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| 6 | `m/n/k < 0` 或 `lda/ldb/ldc` 违反前导维下界 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `groupSize[g] < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `alphaArray == nullptr` 或 `betaArray == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | `problemCount = ΣgroupSize[g] > 0` 时 `Aarray/Barray/Carray` 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | 各数组长度 ≠ `groupCount`（防御性，见 #6） | `ACLBLAS_STATUS_INVALID_VALUE` |

## 算子实现

### 工程结构与文件清单

```
ops-blas/
├── include/
│   └── cann_ops_blas.h                  # 接口声明（复用，不重定义）
├── blas/gemm_grouped_batched_ex/arch22/
│   ├── gemm_grouped_batched_ex.h        # 内部头：结构体/常量
│   ├── gemm_grouped_batched_ex.cpp      # Host 入口：校验+扁平化+tiling+launch
│   ├── gemm_grouped_batched_ex_tiling.h # 问题描述符 + tiling 计算（bucketed）
│   └── gemm_grouped_batched_ex_kernel.h # AIC Cube Kernel（CopyIn/Compute/CopyOut）
└── test/gemm_grouped_batched_ex/arch22/
    ├── gemm_grouped_batched_ex_test.cpp # GTest 加载 CSV 调接口
    ├── gemm_grouped_batched_ex_param.h  # 适配 CSV（camelCase + 不广播）
    └── gemm_grouped_batched_ex_test.csv # 配套 1200 用例（复制入）
```

### 实现方案

#### 3.2.1 Host 侧设计

- **参数校验**：严格按 §参数与异常行为矩阵顺序，先返回早退分支（`groupCount==0` 直接 `SUCCESS`）。
- **问题扁平化**：遍历 `g,i`，将每个子问题展开为一条描述符（m,n,k,lda,ldb,ldc,transa,transb,alpha,beta,GM 指针偏移），连续写入 **Workspace（单一事实源 offset）**，避免 workspace 偏移冲突类隐性 bug。
- **Tiling（bucketed）**：对全部子问题按 `(m,n,k,transa,transb)` 分桶，**同 shape 复用一份 cube tiling 模板**（避免逐问题 `Matmul::Init` 的重复开销）；tiling 尺寸常量化（`TILE_M/TILE_N/TILE_K`，如 128/128/16，尾块按实际 shape 处理，整除约束见 #7）。
- **alpha/beta 处理**：Host 读取标量（默认 Host 指针，见 #3）；`alpha==0` 标记该子问题走「仅 beta 缩放」快路径，`k==0` 标记「仅 beta 缩放」，二者均不引用 A/B。
- **下发**：单次 `ACL` 调用 launch AIC Cube Kernel，绑定 `handle` 的 stream。

#### 3.2.2 Kernel 侧设计（Init → Process）

- **核类型**：**Cube（AIC）单核路径**，非 MIX、非 AIV_ONLY（GEMM 为 cube 典型负载，本算子无串行依赖）。
- **CopyIn**：按 `transa/transb` 与 `lda/ldb` 做**加载期索引重映射**实现 N/T/C（列主序下 `op(A)=A^T` 元素 `(i,j)→A(j,i)`）；仅搬移有效 `m×k`/`k×n` 块。
- **Compute**：Cube Matmul，A/B 为 fp32，**累加保持 fp32**，输出 fp32，满足 `rtol/atol=2⁻¹³`。
- **CopyOut**：`C = alpha·(AB) + beta·C`（alpha≠0）或 `C = beta·C`（alpha=0/k=0）；原地覆写。
- **同步**：跨 pipe 一律 `HardEvent`（MTE2↔MTE3 / MTE3↔MTE2 / MTE2↔V 等），保证跨管线 UB 可见性。
- **0×NaN 清理**：workspace/UB 初值用 `memset`/特定非 NaN 初值，禁止 `0*NaN` 清 UB。
- **边界**：`m==0 || n==0 || groupSize==0` 的子问题在 kernel 循环内直接跳过；`k==0`/`alpha==0` 走快路径。

### 支持硬件

- Atlas A2 系列（Atlas 800T A2 / 910B3，arch22，soc=`ascend910b3`）。
- Atlas A3 系列（arch22 同源，soc=`ascend910_93`）。
- A2/A3 共用 arch22 源码，无平台分支，均在本算子支持范围内；验收在两侧按计划分别执行。

### 算子约束限制

- 非连续 Tensor 不支持；无 broadcast；无 dynamic shape 编译期特化（runtime tiling）；不要求确定性；C 原地覆写无视图返回。

---

# 可维可测分析

## 精度标准/性能标准

| 标准 | 描述 | 来源 |
|---|---|---|
| 精度（FLOAT32） | `|a-g|≤2⁻¹³+2⁻¹³·|g|`；`matched_ratio≥0.99`；`max_abs_error≤1e-2 或 32×ULP` | 任务书 §3.2 / opbase 精度标准 |
| Inf/NaN 规则 | 同号 Inf / 同为 NaN / NaN↔±Inf 视为等效；有限值单独统计 | 任务书 §3.2 |
| 性能 | 平均单次 ≤ 1.25× `gpu_baseline.csv`（≥0.8×）；msprof op `Task Duration(us)`，warmup5 + >10 样本均值 | 任务书 §3.3 / README |
| 内存 | 不涉及（主动给 workspace 峰值） | 任务书 §3.4 |

### 性能标杆与达标分析（★★ 设计阶段可行性估算）

> 估算为设计阶段量级判断，不作为验收依据；实机以 msprof 标定。
> 假设：910B3 fp32 cube 吞吐 ≫ P102-100（约 6.5 TFLOPS）；NPU 单次调用固定开销（ACL runtime + cube pipe init）约数 µs 量级（与 cuBLAS 单次调用开销同量级，apples-to-apples）。

**核心论点**：`gpu_baseline.csv` 的 `gpu_ms` 是 cuBLAS **完整单次 API 调用**耗时（已含其 launch 开销）。本设计同样以**单次 kernel launch 覆盖全部子问题**，两侧固定开销均为「每调用一次」。因此：
- **大/中 shape（m,n,k ≥ 256）**：compute-bound，NPU fp32 cube 吞吐约为 P102-100 的 10×+，轻松 ≤1.25×。
- **极小 shape（8³/16³/32³）**：overhead-bound，两侧皆由 launch 开销主导，比值≈1，大概率达标，需实测确认下限。

| case（来自 gpu_baseline） | problemCount | 总 FLOPs | gpu_ms(标杆) | NPU 计算下界 | NPU 固定开销(估) | NPU 估总计 | 达标判定 |
|---|---|---|---|---|---|---|---|
| base-000 (1024³×1) | 1 | 2.1G | 0.210 | ~0.03 | ~5µs | ~0.04 | ✅ 远超 0.8× |
| base-001 (2048³×1) | 1 | 17G | 1.191 | ~0.21 | ~5µs | ~0.22 | ✅ |
| base-002 (4096³×1) | 1 | 137G | 7.786 | ~1.7 | ~5µs | ~1.7 | ✅ |
| perf-004 (8³×4) | 4 | 4K | 0.015 | ≈0 | ~5–10µs | ~0.005–0.010 | ⚠ 取决于开销 |
| perf-007 (64³×4) | 4 | 2M | 0.025 | ~0.00003 | ~5µs | ~0.005 | ✅ |
| perf-008 (256³×16) | 16 | 0.5G | 0.063 | ~0.006 | ~5µs | ~0.011 | ✅ |
| perf-009 (混合 64;128;256;512) | 15 | ~0.3G | 0.111 | ~0.004 | ~5µs | ~0.009 | ✅ |
| perf-010 (128;256;512 T/T) | 3 | ~0.2G | 0.136 | ~0.003 | ~5µs | ~0.008 | ✅ |

**为何不复用「逐 (g,i) 单独 launch」或「仓内标量/串行同族实现」**：逐问题单独 launch 会把固定开销乘以 `problemCount`（可达数十~数百），极小 case 直接突破 1.25× 物理下限；本设计以单列覆盖全部子问题 + bucketed tiling 复用，从架构上消除该风险。

**不确定项（诚实标注）**：NPU 单次调用固定开销绝对值需实机 msprof 测定；若极小 case 实测仍 >1.25×，将提供 msprof 证据并向社区举证结构性下限（见 #5）。

## 兼容性分析

- 与 ops-blas 仓 950PR 共用 `include/cann_ops_blas.h` 声明，禁止私有平行 API。
- A2(910B3)/A3(910B4) 共用 arch22 源码，无平台分支。
- 测试工程：`gemm_grouped_batched_ex_param.h` 需适配 camelCase + 不广播（见 #2）。

## 测试用例设计（对齐任务书 §3.5）

| §3.5 场景 | 配套 CSV 前缀 | 条数 | 覆盖 |
|---|---|---|---|
| L0 基础（转置全组合） | TC_L0 | 11 | ✅ |
| shape 扫描（1→1024） | TC_MN | 90 | ✅ |
| 标量（alpha/beta） | TC_AB | 11 | ✅ |
| 前导维 padding | TC_LD | 5 | ✅ |
| 填充/Inf/NaN | TC_FL | 12 | ✅ |
| 覆盖（中尺寸） | TC_CV | 8 | ✅ |
| 分组/组内批数 | TC_GRP | 15 | ✅ |
| 边界/负向 | TC_ED | 42 | ✅（含 no-op/k=0/alpha=0/空指针/非法枚举/负维/长度不一致） |
| 扩展（确定性采样） | TC_EX | 806 | ✅（正向 FLOAT32；扩展 dtype 仅负向 NOT_SUPPORTED） |
| 性能 | TC_PF | 200 | ✅（前 4 条对齐 §3.3 典型 case） |

> 缺口：扩展 dtype（FP16/BF16/FP8/INT8）**正向**用例本包不生成（README 明示），负向期望 `NOT_SUPPORTED`，与 #4 处置一致。

## 验证方案（计划）

本算子验收分精度与性能两部分，方案如下；具体实测数据归入自测报告，不在本设计文档中呈现。

### 精度验证（计划）
- **用例规模**：任务书 §3.5 要求的 1200 用例（1000 精度 + 200 性能），由 `gen_csv*.py` 生成配套 CSV，覆盖 `TC_L0/MN/AB/LD/FL/CV/GRP/ED/EX/PF` 全部前缀。
- **执行**：GTest 加载 CSV 调接口，golden 由 `cblas_sgemm`（Netlib）生成；`verify_accuracy.py` 逐元素混合容差（`rtol=atol=2⁻¹³`，`matched_ratio≥0.99`，`max_abs_error≤1e-2 或 32×ULP`）。
- **覆盖**：N/T/C 转置、padding、alpha/beta（含 alpha=0 位精确）、多组多实例、负向 `NOT_SUPPORTED`、k=0/no-op。

### 性能验证（计划）
- **标杆**：任务书 §3.3 的 `gpu_baseline.csv`（cuBLAS 对标），口径为单次 API 调用 `Task Duration(us)` ≤ 1.25× 标杆（≥0.8×）。
- **采样**：`perf_gbe` warmup5 + >10 次有效采样取均值（与 `gpu_ms` 同口径）。
- **平台**：在 Atlas A2 (910B3, soc=`ascend910b3`) 与 Atlas A3 (910B4, soc=`ascend910_93`) 上分别按计划验证（arch22 同源，无平台分支）。

### 设计阶段可行性（详见 §性能标杆与达标分析）
- 大/中 shape compute-bound，NPU fp32 cube 吞吐优势明显，预期轻松达标；极小 shape 由单次调用固定开销主导，已在设计阶段估算并标注风险（§风险与待确认项 #5）。

## 自测方案

- **精度**：`verify_accuracy.py --repo <ops-blas> --soc <target_soc> --csv gemm_grouped_batched_ex_test.csv`（A2=`ascend910b3` / A3=`ascend910_93`）；golden = `cblas_sgemm`。
- **性能**：`verify_performance.py --repo <ops-blas> --soc <target_soc>`（A2=`ascend910b3` / A3=`ascend910_93`）；精确 kernel 耗时用 `msprof op` 采集 `OPPROF_*/OpBasicInfo.csv` 的 `Task Duration(us)`，warmup5 + >10 样本均值。
- **交付**：自测报告含精度对比（截图）+ 性能数据（截图）+ 相对 GPU ≤1.25× 数据。

## 交付与目录

- **设计文档 PR**（cann-competitions）：
  - 分支 `<算子名>-A2A3-design`；base = 上游 master 最新；
  - 路径 `04_tasks/01_community-task-2026/tasklist/{任务目录}/{GitCode账号}/docs/design.md`；
  - 标题 `【CANN社区任务】aclblasGemmGroupedBatchedEx算子设计文档`；
  - 门禁四标签 `cann-cla/yes` `ci-pipeline-passed` `approved` `lgtm`；评论区发 `compile` + @评审 + @Ascend-CANN。
  - 本设计文档按社区任务流程随 PR 公开提交。
- **代码 PR**（ops-blas）：`blas/gemm_grouped_batched_ex/arch22` + `test/gemm_grouped_batched_ex/arch22`；README 支持表标注 Atlas A2/A3「支持」；接口声明复用 `include/cann_ops_blas.h`；私仓邀请 `Ascend-CANN`。

## 风险与待确认项

| # | 事项 | 影响 | 处置 |
|---|---|---|---|
| 1 | **性能基准 GPU 身份冲突**：任务书 §3.3 称「≥0.8× GPU 标杆」，但 README 明示 `gpu_baseline.csv` 为 **NVIDIA P102-100 实测**、「仅用于本地可测性」，且「910B3 标杆表当前待回填」 | 性能判定口径悬而未决 | 当前以 `gpu_baseline.csv`（P102-100）为绑定目标；**向社区确认**是否以 910B3 标杆表为准，若切换需重测 |
| 2 | **CSV 格式 vs 仓内测试框架冲突**：配套 CSV 为 camelCase + 逐列长度严格=`groupCount`（不广播）；仓内 `gemm_grouped_batched_ex_param.h` 读 snake_case + 长度1广播 | 测试工程上库受阻 | 上库时同步提交适配后的 `param.h`（camelCase、不广播），或转换为仓内格式 |
| 3 | **alpha/beta 内存位置**：任务书允许 Device 或 Host，cuBLAS 语义为 Host | 实现口径 | 默认 Host 指针（与 cuBLAS 一致，便于 alpha==0 快路径）；支持 Device 时增拷贝；**向社区确认**是否强制 Device |
| 4 | **扩展 dtype 范围**：任务书 §2.1.3 提 FP16/BF16/FP8/INT8 覆盖，README 称不生成正向用例 | 交付范围 | 主路径 FLOAT32 全实现；非 FLOAT32 组合返回 `NOT_SUPPORTED`（与负向用例一致）；**向社区确认**是否需真实扩展实现 |
| 5 | **极小 shape launch 开销物理下限** | 极小 case 0.8× 不确定 | 单 launch 覆盖全部子问题 + bucketed tiling 复用；实测若仍不达标，提供 msprof 证据举证结构性下限 |
| 6 | **数组长度不一致状态码**：任务书未声明，README 期望 `INVALID_VALUE` | 口径 | 跟随 README 实现 `INVALID_VALUE`；**向社区确认**统一口径 |
| 7 | **跨 pipe 同步 / 0×NaN / workspace 契约**（高频隐性 bug） | 正确性与稳定性 | 单 kernel 内 CopyIn→Compute→CopyOut 用 HardEvent；UB 初值用 memset 非 `0*NaN`；workspace 单一事实源 offset |
