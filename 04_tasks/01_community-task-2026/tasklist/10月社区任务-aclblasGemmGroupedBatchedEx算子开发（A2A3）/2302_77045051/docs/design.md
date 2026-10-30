# 【社区任务】aclblasGemmGroupedBatchedEx算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 10月社区任务-aclblasGemmGroupedBatchedEx算子开发（A2/A3） |
| 队名 / GitCode 账号 | 2302_77045051（个人参与，提交目录名为 GitCode 用户名） |
| 目标仓库 | gitcode.com/cann/ops-blas（算子与测试）、cann-ops-competitions（设计文档） |
| 目标硬件 | Atlas A2 系列产品（性能验收设备 Atlas 800T A2 / 910B3，arch22）与 Atlas A3 系列产品 |
| CANN 版本 | CANN 9.1.0 |
| 主路径数据类型 | 单精度实数 FLOAT32（ACL_FLOAT / ACLBLAS_COMPUTE_32F） |

## 一、需求背景（required）

### 1.1 需求来源

Ascend CANN 社区任务：在 Atlas A2/A3 产品上，基于 ops-blas 开源仓工程框架，使用 Ascend C 编程语言以 Kernel 直调方式开发分组批量扩展版通用矩阵乘算子 `aclblasGemmGroupedBatchedEx`，与 cuBLAS `cublasGemmGroupedBatchedEx` 的核心功能与参数语义完全对齐，用于补齐 Blas 仓能力；验收通过后合入 ops-blas 开源仓 `blas/gemm_grouped_batched_ex/arch22` 目录。

### 1.2 背景介绍

#### 1.2.1 标杆来源、TBE 源码与算子信息库核查

本任务是向 ops-blas 新增公共 BLAS 接口 `aclblasGemmGroupedBatchedEx` 的 Ascend C Kernel 直调实现，不是既有 ACLNN/TBE 算子的迁移优化。因此下列审核项的结论是"不适用"，而不是缺少材料：

| 审核对象 | 核查结论 | 依据及替代参考 |
| --- | --- | --- |
| 原 TBE 算子源码路径 | 不适用：任务书未给出、仓内不存在该接口的 TBE/ACLNN 实现 | 数学与 API 语义以 cuBLAS `cublasGemmGroupedBatchedEx` 官方文档与 Netlib `sgemm.f` 为准 |
| 算子信息库路径 | 不适用：本任务不注册 ACLNN 算子、不使用算子信息库 | 公共接口声明位于 `ops-blas/include/cann_ops_blas.h`，接口名非 aclnn* |
| 可复用工程实现 | 适用 | ops-blas 仓已有 BLAS 算子（如 `blas/syr2/` 等同族句柄式接口）的 handle/stream/启动框架、构建自动发现机制与 `test/` CSV 参数化 GTest 测试框架 |

接口声明必须放入 `include/cann_ops_blas.h`，与现网/950PR 共用，禁止定义产品私有平行接口。

#### 1.2.2 标杆算子现状分析

**语义标杆**：cuBLAS `cublasGemmGroupedBatchedEx`。groupCount 个分组，第 g 组含 groupSize[g] 个独立矩阵乘，第 g 组第 i 个矩阵乘计算：

```
C_{g,i} = alphaArray[g] * op(A_{g,i}) * op(B_{g,i}) + betaArray[g] * C_{g,i}
```

- op(X) 由 transaArray[g]/transbArray[g] 决定：N 不转置、T 转置、C 共轭转置（实数路径下 C 与 T 等价）；
- A、B、C 均为列主序（Column-Major）存储，第 g 组 op(A) 为 m_g×k_g、op(B) 为 k_g×n_g、C 为 m_g×n_g；
- alphaArray/betaArray 为逐组标量数组，任务书 §2.4 定义其位于 Device 或 Host 内存；
- 边界行为对齐 cuBLAS / Netlib sgemm quick return 语义（详见 3.2.4）。

**精度标杆（Golden）**：cblas（Netlib BLAS 3.10.0 实数实现 `cblas_sgemm`）逐组生成，随任务测试工程提供。

**工程标杆**：ops-blas 仓内同族句柄式 BLAS 算子（Host 校验 → handle 绑定 stream → tiling → 异步 launch 的框架与测试接入方式），仅复用工程接入方式，数值实现独立完成。

#### 1.2.3 标杆算子实现流程（cuBLAS 语义）

```
aclblasGemmGroupedBatchedEx(handle, transaArray, transbArray, mArray, nArray, kArray,
                            alphaArray, Aarray, Atype, ldaArray, Barray, Btype, ldbArray,
                            betaArray, Carray, Ctype, ldcArray, groupCount, groupSize, computeType)
  │
  ├─ handle 为空? ── 是 ──> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
  ├─ dtype/computeType 组合不支持? ── 是 ──> ACLBLAS_STATUS_NOT_SUPPORTED
  ├─ groupCount < 0? ── 是 ──> ACLBLAS_STATUS_INVALID_VALUE
  ├─ groupCount == 0? ── 是 ──> 直接返回 SUCCESS（不再校验其余参数）
  ├─ host 数组为空 / 逐组维度、前导维、枚举非法? ── 是 ──> 对应错误码
  ├─ problemCount = Σ groupSize[i] > 0 且 Aarray/Barray/Carray 为空? ── 是 ──> INVALID_VALUE
  ├─ 对每个 (g, i)：m_g=0 或 n_g=0 ──> no-op，不读不写
  │                 k_g=0 ──> 仅 C = beta_g * C，不引用 A、B
  │                 alpha_g=0 ──> C = beta_g * C（不引用 A、B；结果位精确）
  └─ 其余 ──> 逐组逐矩阵执行 GEMM，异步返回 SUCCESS
```

### 1.3 aclblasGemmGroupedBatchedEx 算子功能分析

**计算公式**：对 g = 0..groupCount-1、i = 0..groupSize[g]-1：

```
C_{g,i}[p,q] = alpha_g * Σ_{r=0..k_g-1} op(A)[p,r] * op(B)[r,q] + beta_g * C_{g,i}[p,q]
```

- **输入**：handle，transaArray/transbArray，mArray/nArray/kArray，alphaArray，Aarray+Atype+ldaArray，Barray+Btype+ldbArray，betaArray，groupCount，groupSize，computeType（共 19 个形参）；
- **输出**：Carray（各矩阵原地覆写，列主序）；
- **支持数据类型**：主路径 FLOAT32（Atype/Btype/Ctype = ACL_FLOAT，computeType = ACLBLAS_COMPUTE_32F）；扩展接口声明的 FP16/BF16/FP8/INT8 组合本次按 README 宣称范围返回 NOT_SUPPORTED 或后续覆盖，负向用例期望 `ACLBLAS_STATUS_NOT_SUPPORTED`；
- **广播**：不涉及。各分组各矩阵的 A/B/C 为独立操作数。

## 二、需求分析（required）

### 2.1 需求描述

使用 Ascend C 编程语言、基于 ops-blas 工程框架实现 `aclblasGemmGroupedBatchedEx` 句柄式 BLAS 接口，在 Atlas A2/A3（arch22）上通过 handle 绑定 stream 直调 NPU Kernel，满足：

1. 与 cuBLAS `cublasGemmGroupedBatchedEx` 参数序列、语义完全对齐（handle 及参数顺序一一对应，无需映射说明）；
2. FLOAT32 主路径全量转置组合（N/T/C × N/T/C，C≡T）与逐组异构 m/n/k/ld；
3. 精度满足生态算子开源精度标准 FLOAT32 档位；
4. 性能达到 GPU 标杆 0.8 倍以上（平均单次耗时 ≤ 1.25× GPU）。

### 2.2 需求拆解

1. 实现公共接口 `aclblasGemmGroupedBatchedEx`（19 参，签名以 `include/cann_ops_blas.h` 已有声明为准）；
2. Host 侧参数校验（含 §2.5 规定的校验顺序与 7 类错误码）、no-op/quick-return 语义、分组展开与调度；
3. Kernel 侧分组批量 GEMM：tile 级负载均衡、Cube 矩阵乘 + Vector epilogue、转置适配、尾块处理；
4. 特殊路径：k=0 仅 beta 缩放、alpha=0 位精确缩放、beta=0 不读 C 旧值、Inf/NaN 透传；
5. 自测：覆盖任务书 200 条用例与测试包 1200 条用例（1000 精度 + 200 性能），输出自测报告；
6. 交付：算子 README（A2/A3 标注支持）、测试工程 `test/gemm_grouped_batched_ex/`、PR 合入 ops-blas。

### 2.3 外部组件依赖

| 组件 | 适配状态 | 用途 |
| --- | --- | --- |
| CANN 9.1.0 Ascend C / ACL Runtime | 已适配 | 编译 Kernel、管理 Device 内存、stream 异步启动 |
| Atlas A2（910B3，arch22）/ Atlas A3 | 已适配 | 正式验收硬件 |
| cuBLAS / Netlib BLAS | 不作为运行时依赖 | 仅作 API/数学语义参考；Golden 由 cblas_sgemm 独立生成 |

不新增 CATLASS、TBE、ACLNN 或其他生产第三方依赖。

### 2.4 内部适配模块

| 模块 | 仓内位置 | 适配内容 |
| --- | --- | --- |
| 公共 BLAS API | `include/cann_ops_blas.h` | 使用已有声明，与现网/950PR 共用，不新增平行接口 |
| Host 实现与启动 | `blas/gemm_grouped_batched_ex/arch22/gemm_grouped_batched_ex_host.cpp` | 参数校验、分组展开、tiling、workspace 管理、stream launch |
| Host–Kernel ABI | `blas/gemm_grouped_batched_ex/arch22/gemm_grouped_batched_ex_tiling_data.h` | 问题描述表 / tile 任务表 / tiling 结构定义 |
| Kernel | `blas/gemm_grouped_batched_ex/arch22/gemm_grouped_batched_ex_kernel.cpp` | 分组批量 GEMM Kernel 与缩放 Kernel（Cube+Vector） |
| 算子 README | `blas/gemm_grouped_batched_ex/README.md` | 产品支持表标注 Atlas A2/A3 支持 |
| 构建与测试 | `test/gemm_grouped_batched_ex/` | CSV 参数化 GTest、NPU wrapper、精度/性能记录（结构参考主仓 BLAS 算子测试代码） |

### 2.5 设计原则与非目标

| 原则 | 设计约束 |
| --- | --- |
| 单一公共接口 | 只实现 `cann_ops_blas.h` 已有声明的 `aclblasGemmGroupedBatchedEx`，不定义产品私有 API |
| 语义对齐 | 转置/前导维/no-op/quick-return/错误码语义以任务书 §2.4/§2.5 为准；空数组指针语义取 ops-blas 实现口径，不要求与 cuBLAS 一致 |
| 正确性优先 | 通用路径覆盖全部合法输入；性能分派不得缩小 API 支持范围，不适用时回退通用路径 |
| 证据驱动优化 | tile 参数、核数、缓冲级数由平台信息与 profiling 决定，不按用例名称/基线值特判 |
| 无隐式同步 | API 只向 handle 绑定 stream 入队；Host 侧不做隐式 `aclrtSynchronizeStream`，读回结果由调用方同步 |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；Host 指针标量经 stream 上的异步拷贝进入 Device workspace |

非目标：FP16/BF16/FP8/INT8 数值实现（本次仅负向 NOT_SUPPORTED）、arch20/arch35 移植、全仓框架重构、确定性计算保证（任务书 §2.5 不要求）。

## 三、详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

见 1.3。列主序存储：元素 (p, q) 地址为 `base + p + q * ld`（逐矩阵独立 base 指针）。

#### 3.1.2 支持数据类型

| 参数 | 数据类型 | 存储位置 |
| --- | --- | --- |
| A/B/C 矩阵 | ACL_FLOAT（float32） | Device，A/B 只读，C 原地读写 |
| alphaArray / betaArray | float32（长度 groupCount） | Device 或 Host（任务书 §2.4），见 3.3.4 |
| 维度/前导维/分组数组 | int | Host |
| transa/transb | aclblasOperation_t ∈ {N, T, C} | Host，实数路径 C ≡ T |
| computeType | ACLBLAS_COMPUTE_32F | Host |

#### 3.1.3 支持形状与布局

| 项目 | 支持范围 |
| --- | --- |
| 维度 | m_g, n_g, k_g ≥ 0（逐组独立），运行时入参 |
| 分组 | groupCount ≥ 0；groupSize[g] ≥ 0；problemCount = Σ groupSize[i] |
| 存储顺序 | 列主序 ND，前导维逐组独立 |
| 前导维约束 | transa=N：lda_g ≥ max(1, m_g)，否则 lda_g ≥ max(1, k_g)；transb=N：ldb_g ≥ max(1, k_g)，否则 ldb_g ≥ max(1, n_g)；ldc_g ≥ max(1, m_g) |
| 非连续 Tensor | 不要求（超出 ld 语义的非连续访问不支持） |
| dynamic shape | 不要求（host tiling 支持运行时入参） |
| 原地语义 | C 原地覆写，不返回视图 |

### 3.2 需求总体设计

#### 3.2.1 调用方式

ops-blas 句柄式 BLAS API + Ascend C Kernel 直调。调用方创建 handle 并以 `aclblasSetStream` 绑定 stream，将 Host 侧描述数组与 Device 侧矩阵指针数组传入；Host 完成校验、分组展开、tiling 与 workspace 准备后，仅向绑定 stream 异步入队 Kernel，立即返回；调用方读回 C 前自行同步该 stream。

```
创建 handle 并 aclblasSetStream 绑定 stream
        │
aclblasGemmGroupedBatchedEx(...)
        │
Host：校验 → quick return 判定 → 分组展开/问题表/tile 表构建
      → （alpha/beta 为 Host 指针时）stream 上异步 H2D 到 workspace
        │
arch22 Ascend C Kernel（单次 launch 覆盖全部分组）
        │
调用方同步 stream 后读取 C
```

#### 3.2.2 算子数据模型

Host 侧将分组描述展开为 problemCount 个扁平子问题（编号 p = 0..problemCount-1），每个子问题携带：

```
ProblemDesc {
  int32 m, n, k;            // 本矩阵的 m/n/k
  int32 lda, ldb, ldc;
  int8   transa, transb;    // 归一化后 C→T
  int32  groupIdx;          // 用于读取 alpha_g/beta_g
  const float* A; const float* B; float* C;   // Device 指针（来自 Aarray/Barray/Carray）
}
```

再按 3.3.2 的 tiling 规则展开为 tile 任务表。两级表结构使 Kernel 不感知"分组"概念，仅消费统一的 tile 任务序列，天然实现组间/组内负载均衡与大小矩阵混合调度。

#### 3.2.3 列主序到 Cube 计算的映射（转置策略）

Ascend C Matmul（Cube）按行主序左操作数布局约定工作。利用恒等式 `C = op(A)·op(B) ⟺ C^T = op(B)^T·op(A)^T` 与"列主序存储 X ≡ 行主序存储 X^T（同一内存）"，将每个列主序子问题零拷贝映射为行主序子问题：

| 列主序问题 | 行主序等价问题 |
| --- | --- |
| M, N, K | M' = n，N' = m，K' = k |
| opa = transa, opb = transb | transa' = transb，transb' = transa |
| A 矩阵（lda） | 充当行主序 B' 操作数（ld' = lda） |
| B 矩阵（ldb） | 充当行主序 A' 操作数（ld' = ldb） |
| C 矩阵（ldc，列主序 m×n） | 即行主序 C'（n×m）同一内存（ld' = ldc） |

四种 (transa, transb) 组合映射后仍为四种种转置组合，每个操作数恰以原始 op 语义进入 Matmul 的 isTrans 标志（`SetTensorA(pLocal, isTrans)` / `SetTensorB(pLocal, isTrans)`），不引入额外转置拷贝；输出 tile 直接按列主序地址写回 C。

**布局适配层（正确性回退）**：若 CANN 9.1.0 在 arch22 上对 FLOAT32 的转置装载（isTrans）受限，则该操作数在 CopyIn 阶段经 GM→UB 分块搬运后由 Vector 单元做显式转置（strided DataCopy + 向量重排），再以 isTrans=false 进入 Cube；该回退仅影响该操作数吞吐，不影响语义与覆盖范围。最终采用哪种方式以目标设备编译/ profiling 证据为准。

#### 3.2.4 边界与特殊值语义（对齐任务书 §2.5）

| 场景 | 行为 |
| --- | --- |
| groupCount = 0 | 直接返回 SUCCESS，不再校验其余参数 |
| m_g = 0 / n_g = 0 / groupSize[g] = 0 | 合法 no-op：不生成任何任务，不读不写对应矩阵 |
| k_g = 0 | 仅做 C = beta_g * C，不引用 A、B（生成 SCALE 任务） |
| alpha_g = 0（精确 == 0.0f，含 -0.0；NaN 不为零） | 不引用 A、B 的乘积：C = beta_g * C；beta=1 时 C 逐位不变、beta=0 时置零、其余逐元素缩放，结果位精确（EXACT）。判定与分支在 Kernel 侧 epilogue 完成（alpha 可能位于 Device，Host 不可见）；若 alphaArray 为 Host 指针，Host 读取后可为整组 alpha=0 直接跳过 matmul 任务生成（优化项） |
| beta_g = 0 | C 不必是有效输入：epilogue 不读 C 旧值，输出 C = alpha_g * A·B（规避 0×NaN/Inf 污染） |
| Inf/NaN 输入 | 逐元素透传：乘加路径按 IEEE 语义产生 Inf/NaN，不截断、不特判；0×Inf、Inf-Inf 等不定式最终形态随累加顺序，按任务书 §3.2 特殊值规则比对 |

### 3.3 Host 侧设计

Host 侧职责：校验公共接口契约 → 展开/过滤子问题 → tiling 与任务表构建 → workspace 准备（含 Host 标量异步落地）→ 在绑定 stream 上异步 launch。Host 不做任何同步。

#### 3.3.1 参数校验与启动边界（校验顺序按任务书 §2.5）

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 2 | Atype/Btype/Ctype/computeType 非法或不支持组合（如 ACL_DOUBLE、999） | ACLBLAS_STATUS_NOT_SUPPORTED |
| 3 | groupCount < 0 | ACLBLAS_STATUS_INVALID_VALUE |
| 4 | groupCount == 0 | ACLBLAS_STATUS_SUCCESS（不再校验其余参数） |
| 5 | groupCount > 0 且 transaArray/transbArray/mArray/nArray/kArray/ldaArray/ldbArray/ldcArray/groupSize/alphaArray/betaArray 任一为 nullptr | ACLBLAS_STATUS_INVALID_VALUE |
| 6 | 逐组：transaArray[g]/transbArray[g] ∉ {N, T, C} | ACLBLAS_STATUS_INVALID_ENUM |
| 7 | 逐组：m_g/n_g/k_g/groupSize[g] < 0，或 lda/ldb/ldc 不满足 3.1.3 约束 | ACLBLAS_STATUS_INVALID_VALUE |
| 8 | 组内数组长度与 groupCount 不一致（防御性，如 len_mismatch_kArray 用例） | ACLBLAS_STATUS_INVALID_VALUE |
| 9 | problemCount = Σ groupSize[i] > 0 且 Aarray/Barray/Carray 任一为 nullptr | ACLBLAS_STATUS_INVALID_VALUE |
| 10 | 其他合法输入 | 计算任务表并异步 launch，返回 SUCCESS |

说明：多个非法条件组合时按上表首个命中返回；`groupCount=0` 时所有数组列（含指针数组）均不校验，验证 §2.5 的顺序语义。

#### 3.3.2 分核、分块与任务调度策略

**问题展开与过滤**：Host 遍历 (g, i) 生成 ProblemDesc；m_g=0/n_g=0/groupSize[g]=0 的矩阵不产生任务；k_g=0 的矩阵产生 SCALE 任务；其余产生 GEMM tile 任务。

**tile 划分**：对每个 GEMM 子问题按尺寸分档选择 (TM, TN, TK)（候选如 128×128 / 64×128 / 32×64，TK 取 Cube 单次 Mmad 的 K 步进，FLOAT32 按 16 对齐基准；最终档位由平台 L0A/L0B/L0C/UB 容量与 profiling 确定）。tiles_p = ceil(m_g/TM_g) × ceil(n_g/TN_g)，连同 (tileM0, tileN0) 追加进全局 tile 表。

**persistent 调度（负载均衡）**：

```
launchGrid = min(totalTiles + scaleTasks, aivNum)   // aivNum 由平台信息获取，910B3 以运行时 GetPlatformInfo 为准
kernel 内：for (task = blockId; task < totalTasks; task += gridDim) { 处理 task }
```

- 每个 tile/task 由唯一 block 处理，无跨核写冲突、无原子操作；
- 异构分组（大小矩阵混合、组内批数不一）由全局任务序列自然均衡：单大矩阵（如 4096³ 产生数百个 tile）自动铺满全部核，海量小矩阵（如 8³×16 组）以单核一 tile 粒度摊开，避免"按组绑核"的倾斜；
- 大 case（tiles_p ≥ 2×aivNum）等价于普通单 GEMM 的多核切分，无性能损失，无需特判。

**UB/内存优化**：Kernel 内按 TM×TN 分块经 Ascend C Matmul API（`matmul.h`，TCubeTiling）完成 K 循环，iterate 自带多级缓冲（double buffer）；UB 需求 = A'/B' 分块 + C tile + 旧 C tile + 标量缓存，按目标平台 UB 容量（GetPlatformInfo 获取）校验档位；L0A/L0B/L0C 由 Matmul 框架管理。

**尾块**：`SetTail` 处理 m/n 非 tile 倍数余量；K 余量由 iterate 尾块处理；ld 大于紧凑值的 padding 列/行不访问。

#### 3.3.3 tilingKey / 分派策略

本工程为 ops-blas Kernel 直调，不使用 ACLNN/TBE 框架 tilingKey。等价分派信息经 Host launch 参数与任务表传递：

| 分派 | 条件 | 策略 |
| --- | --- | --- |
| Kernel 模板变体 | (transa', transb') ∈ {N,T}×{N,T}（C 归一化为 T） | `template<int TA, int TB>` 实例化 4 份，Host 按组内主导组合选择 launch（组内混合转置时按 tile 归属选择变体，或经布局适配层统一到单一变体，以 profiling 定） |
| GEMM 任务 | k_g > 0 | Cube Matmul + Vector epilogue |
| SCALE 任务 | k_g = 0（Host 可见 kArray） | 纯 Vector 缩放 kernel（读 beta_g，含 beta=0 置零分支） |
| no-op | m/n/groupSize = 0 | 无任务 |

分派仅依赖输入布局与规模，不读取 case 名称、seed、基线值。

#### 3.3.4 workspace 与标量（alpha/beta）处理

**workspace 构成**（一次调用，绑定 stream 上的异步 H2D）：

```
sizeof(workspace)
  = problemCount × sizeof(ProblemDesc)      // ≈ 48–64 B/问题
  + totalTasks  × sizeof(TileTask)          // ≈ 8–12 B/tile
  + groupCount  × 2 × sizeof(float)         // alpha/beta 落地缓存（仅当为 Host 指针时）
```

对 200 条性能用例的量级（problemCount ≤ 数百、totalTiles ≤ 数千），workspace 为 KB 量级，远小于矩阵数据本身，满足交付件"workspace 相对 GPU ≤ 1.2×"要求（具体数值随自测报告出具）。

**alpha/beta 双位置处理**：Kernel 统一从 Device 地址读取逐组标量。

- alphaArray/betaArray 为 Device 指针：任务表直接携带该指针；
- 为 Host 指针：以 `aclrtPointerGetAttributes` 判别属性后，将 groupCount×2 个 float 经 `aclrtMemcpyAsync`（绑定 stream）拷入 workspace 标量区，任务表携带落地地址。

**workspace 复用**：与 handle 关联缓存（同 stream 顺序复用、按需扩容），避免每次调用 aclrtMalloc/Free 的小 case 开销（性能用例 base-004 量级 NPU 预算仅约 18.9 us，launch+准备开销是主要风险点）。

#### 3.3.5 索引与 ABI 边界

- 元素地址统一在 64 位下计算：`addr = base + (int64)p + (int64)q * ld`，禁止 int 先乘后转（大 m×ld 溢出）；
- TilingData/任务表按值或只读 Device 常量区传递；Host 提交后修改 Host 数组不影响已入队任务（描述表已拷贝）；
- int 入参范围不表示硬件可分配相应内存，超大尺寸仅由 Host 负向校验覆盖，不作为 Device 运行证明。

### 3.4 Kernel 侧设计

#### 3.4.1 Kernel 结构

主 GEMM Kernel（Cube+Vector 混合）按"初始化 → 逐任务处理"两阶段：

```
Init:    GetPlatformInfo / 任务表指针 / Matmul 对象初始化（按 TA/TB 模板实例）
Loop:    task = blockId; task < totalTasks; task += gridDim
  ├─ GEMM task:
  │    CopyIn  : GM(A' 分块、B' 分块、beta_g>0 时旧 C 分块) → UB（含布局适配层转置回退）
  │    Compute : Matmul iterate K 循环 → L0C（FLOAT32 Mmad 累加）
  │    Epilogue: L0C → UB；读 alpha_g/beta_g 分支——
  │        alpha_g != 0:  C = alpha_g * AB + (beta_g != 0 ? beta_g * C_old : 0)   [beta=0 不读 C_old]
  │        alpha_g == 0:  C = beta_g * C_old（beta=1 位不变 / beta=0 置零 / 其余缩放，位精确）
  │    CopyOut : UB → GM（列主序地址，仅有效元素）
  └─ SCALE task: C = beta_g * C（同 epilogue 缩放分支，纯 Vector）
```

- 单一 Kernel launch 覆盖全部分组与批数（SCALE 任务并入同一任务表，避免多 Kernel 串行 launch）；
- 每 tile 唯一写入，无原子、无全局归约、无跨核同步；
- MTE2→V、V→Cube、Cube→V、V→MTE3 依赖经 Ascend C 事件/pipe 机制约束，K 循环 double buffer。

#### 3.4.2 Ascend C 实现与标杆流程差异及原因

| 维度 | cuBLAS / Netlib 语义标杆 | Ascend C 实现 | 差异原因与正确性约束 |
| --- | --- | --- | --- |
| 执行位置 | GPU/CPU 内部实现 | arch22 Host 异步 launch + persistent Kernel | 适配 ops-blas 句柄与 stream；API 不在 Host 同步 |
| 分组调度 | cuBLAS 内部按 problem 调度 | Host 展开扁平 tile 表，Kernel grid-stride 认领 | 异构分组负载均衡；每 tile 唯一写入 |
| 布局 | 列主序直读 | 恒等式交换 A/B 零拷贝映射为行主序 + isTrans | 不引入额外转置；回退路径语义等价 |
| 累加顺序 | cblas 顺序固定 | Cube 分块 K 累加，顺序不同 | 非位精确要求场景按 rtol/atol 2⁻¹³ 判定；alpha=0 缩放路径单独保证位精确 |
| 标量位置 | cuBLAS pointer mode | 任务表统一 Device 地址（Host 标量异步落地） | Kernel 只读 Device；不引入 Host 同步 |
| no-op/quick return | 不访问 Device | Host 过滤零任务；k=0/alpha=0 走缩放分支 | 对齐任务书 §2.5；beta=0 不读 C 旧值 |

#### 3.4.3 浮点顺序与特殊值

- 乘加按 `alpha*(A·B)` 先聚合再与 `beta*C_old` 相加的顺序，有限值满足同一公式；Inf/NaN 对重结合敏感，不因代数等价改写表达式；
- `beta_g == 0.0f`（含 -0）分支不加载 C 旧值，规避 0×Inf/NaN 污染（任务书：beta=0 时 C 不必为有效输入）；
- `alpha_g == 0.0f` 分支不使用 AB 结果（可能含 Inf），保证 C = beta*C 位精确；
- ±0 符号位不做强制归一，EXACT 用例仅覆盖任务书定义的缩放路径。

### 3.5 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas 800T A2（910B3，arch22） | √ | 性能验收设备（任务书 §3.1/§3.3） |
| Atlas A2 系列其他款型 / Atlas A3 系列 | √ | 同 arch22 架构；自验证覆盖一款款型即可（精度建议 910B3 或 A3），README 产品支持表标注"支持" |
| Ascend 950PR（arch35）/ 310P（arch20） | × | 接口声明共用，Kernel 不在本任务范围 |

### 3.6 算子约束限制

1. 主路径仅 FLOAT32（ACL_FLOAT × 3 + ACLBLAS_COMPUTE_32F）；FP16/BF16/FP8/INT8 及不支持 computeType 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；
2. 不支持超出 lda/ldb/ldc 语义的非连续内存访问；不支持广播（无此语义）；
3. transa/transb 仅 {N, T, C}，实数下 C≡T，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；
4. 异步接口：读回结果前须同步绑定 stream；API 内部不同步；
5. 不保证确定性计算（任务书不要求）；别名（各矩阵间指针重叠）不在本期契约内；
6. 空数组指针语义以本设计 3.3.1 为准（ops-blas 口径），不要求与 cuBLAS 一致。

## 四、特性交叉分析

| 交叉特性 | 组合 | 设计处理与验证重点 |
| --- | --- | --- |
| 转置 × 形状 | (N/T/C)² × 1→1024 扫描 × 矩形 × 奇数/非对齐/2ⁿ±1 | 4 变体模板全覆盖；尾块 SetTail；ld padding 列不访问 |
| 分组 × 批数 | groupCount ∈ {0,1,2,3,4,5,8} × groupSize ∈ {0,1,…,64} × 组内混合 | 扁平 tile 表统一调度；零任务 no-op；组内转置混合按 tile 归属分派 |
| quick return × 指针 | groupCount=0 / m=0 / n=0 / groupSize=0 × 指针数组为空 | 校验顺序 3.3.1：groupCount=0 先返回 SUCCESS 不查其余；problemCount>0 才查 A/B/C 指针 |
| k=0 / alpha=0 / beta 组合 | k=0 × beta∈{0,1,其他}；alpha=0 × beta∈{0,1,其他} | SCALE 任务 + epilogue 分支；beta=1 位不变、beta=0 置零、位精确 EXACT 断言 |
| 特殊值 × 算术路径 | A/B/C 含 Inf/NaN × 一般 alpha/beta | 透传不特判；beta=0 不读 C；任务书 §3.2 Inf/NaN 等效规则单独判定与统计 |
| 标量位置 | alpha/beta ∈ {Host, Device} 指针 × 正常/缩放路径 | 属性判别 + stream 异步落地；Kernel 只读 Device 地址 |
| 性能分派 × 通用性 | 4096³ 单问题 × 8³×16 组微批 × 异构分组 | 同一 persistent 框架覆盖；无按 case 特判；小 case 依赖单 launch 与 workspace 复用压开销 |
| 异步 × 原地 | 绑定 stream × 连续调用 / C 原地覆写 | 无 Host 同步；tile 唯一写入；同 stream 顺序性由任务提交序保证 |

## 五、可维可测分析

### 5.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | FLOAT32：rtol = atol = 2⁻¹³ (1.2e-4)，逐元素 `|actual-golden| ≤ atol + rtol×|golden|`；matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32×ULP | 任务书 §3.2、生态算子开源精度标准（mixed_tolerance_standard） |
| 特殊值 | Inf/NaN 不适用容差式：双 NaN / 同号 Inf / NaN↔±Inf 视为等效；一方有限另一方非有限判失败；非有限值单独统计，不计入 matched_ratio | 任务书 §3.2 |
| 位精确特例 | alpha=0（指针非空）时 C = beta*C 需位精确（EXACT） | 任务书 §3.2 |
| 性能 | Atlas 800T A2 (910B3)，FLOAT32，msprof op 模式采集单算子 task duration；warmup 5 次后有效采样 10 次取算术平均；判据：平均单次耗时 ≤ GPU 标杆 1.25 倍（≥ 0.8× GPU），逐 case 对标 gpu_baseline.csv | 任务书 §3.3 |
| 内存 | workspace 相对 GPU ≤ 1.2×，随自测报告出具 | 任务书 §4 交付件 3 |

**任务书典型性能 case 门槛**（gpu_baseline.csv 前 4 行，NPU 上限 = gpu_ms × 1.25）：

| case | groupCount×groupSize | m=n=k | GPU 标杆 (ms) | NPU 上限 (ms) |
| --- | --- | --- | --- | --- |
| TC_PF_1001 / base-000 | 1×1 | 1024 | 0.210142 | 0.2627 |
| TC_PF_1002 / base-001 | 1×1 | 2048 | 1.191470 | 1.4893 |
| TC_PF_1003 / base-002 | 1×1 | 4096 | 7.785660 | 9.7321 |
| TC_PF_1004 / base-003 | 2×1 | 1024×2 | 0.401152 | 0.5014 |

4096³ 单 case 折算有效算力约 2×4096³ / 9.73 ms ≈ 14.1 TFLOPS（FLOAT32），处于 910B3 Cube 常规可达区间；微批小 case（base-004 起，GPU 约 15.1 us）NPU 预算约 18.9 us，主要开销在 launch 与任务表准备，已按单次 launch + workspace 复用设计应对。

### 5.2 测试工程与数据流

```
test_cases/gen_csv.py（固定种子 20261028，可复现）
  → gemm_grouped_batched_ex_test.csv（1200 条 = 1000 精度 + 200 性能）
  → C++ CSV 参数化 GTest（ops-blas test 框架，加载 CSV 调用 aclblasGemmGroupedBatchedEx）
      → 精度：CPU golden（Netlib cblas_sgemm 逐矩阵生成）× NPU 输出全矩阵逐元素比对
      → 性能：msprof op 采集 Task Duration(us)，warmup 5 + 采样 10 取平均
verify_accuracy.py / verify_performance.py（--repo /path/to/ops-blas --soc ascend910b3）
```

- 覆盖随任务提供的全部用例：L0 基础 11、shape 扫描 90、标量 11、前导维 5、填充 12、覆盖 8、分组/批数 15、边界/负向 42、扩展 806、性能 200（含任务书 200 条对应集合）；
- 负向用例比对 `expect_result` 状态码（handle 空、非法枚举 999、非法 dtype/computeType、负维度/负组数/负批数、非法 ld、空指针、数组长度不一致），不参与数值比对；
- 用例不足场景（边界/负向/特殊值）按任务书 §3.5.4 自行补充，期望状态码以 §2.4/§2.5 为准；
- msprof 命令：`msprof op --application="./build/test/gemm_grouped_batched_ex/gemm_grouped_batched_ex_test --gtest_filter=*TC_PF_1001*" --output=./prof_xxx`，在 OPPROF_*/OpBasicInfo.csv 按 kernel 名统计 Task Duration。

### 5.3 覆盖要求

| 维度 | 覆盖内容 |
| --- | --- |
| 转置 | transa×transb 全 9 组合（含 OP_C 及组内混合） |
| 形状 | 0/1、小质数、2 的幂及 ±1、非对齐、矩形，直至 1024+；m/n/k 各自独立扫描 |
| 标量 | 组内同标量、组间差异化 alpha/beta；特殊值 0、1、大值、-1；均匀 [-5,5] 与正态各 50% |
| 前导维 | 紧凑最小值 + padding(+4/+8)、仅 lda/仅 ldc、转置混合 |
| 分组 | groupCount 0/1/小/中/2 的幂；groupSize 0/1/小/中（至 64）与组内混合批数 |
| 边界/负向 | 零维/零组/零批 no-op、k=0、alpha=0、空指针（alpha/beta/A/B/C）、handle 空、非法枚举/dtype/computeType、负值、非法 ld、长度不一致 |
| 特殊值 | Inf/NaN 填充（A/B/C 分别）、按 §3.2 规则单独判定 |
| 性能 | 200 条：4 条典型 case + 小尺寸 + 分组异构 + 转置组合 + 矩形 + 混合 |
| 异步 | 非默认 stream、同 stream 连续调用、读回前同步 |

### 5.4 内存预算

单用例 Host 侧 ≤ 512MB（测试工程设计值）；典型性能 case 4096³ 各矩阵约 64MB、A/B/C 合计约 192MB，在预算内。workspace 构成见 3.3.4，KB 量级，随自测报告出具与 GPU 的对比（≤1.2×）。

### 5.5 兼容性分析

新增公共 API，使用 `include/cann_ops_blas.h` 已有声明，不改变现有任何 BLAS 接口的参数与行为；接口供现网/950PR 及其他产品线共用，不创建私有平行 API。arch22 目录内实现，其他架构不提供伪实现、不宣称支持；如未来移植，复用本数学契约与测试资料，独立实现 Kernel 与资源参数。

## 参考资料

1. 任务书：aclblasGemmGroupedBatchedEx_A2_task_doc.md（10月社区任务包）
2. ops-blas 开源仓：https://gitcode.com/cann/ops-blas （CONTRIBUTING.md、include/cann_ops_blas.h、test 框架）
3. cuBLAS cublasGemmGroupedBatchedEx：https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmGroupedBatchedEx
4. Netlib BLAS sgemm：https://www.netlib.org/blas/sgemm.f
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
6. Ascend C 算子开发文档 / 接口文档（CANN 9.1.0）
7. 社区任务流程及注意事项：https://gitcode.com/org/cann/discussions/39

## 目录

- 一、需求背景（required）
  - 1.1 需求来源 / 1.2 背景介绍（标杆核查、现状分析、流程） / 1.3 功能分析
- 二、需求分析（required）
  - 2.1 需求描述 / 2.2 需求拆解 / 2.3 外部组件依赖 / 2.4 内部适配模块 / 2.5 设计原则与非目标
- 三、详细设计（required）
  - 3.1 算子分析（公式、类型、形状）
  - 3.2 总体设计（调用方式、数据模型、列主序映射、边界语义）
  - 3.3 Host 侧设计（校验、调度、tiling/分派、workspace/标量、ABI）
  - 3.4 Kernel 侧设计（结构、差异表、浮点顺序与特殊值）
  - 3.5 支持硬件 / 3.6 算子约束限制
- 四、特性交叉分析
- 五、可维可测分析（精度/性能标准、测试工程、覆盖、内存、兼容性）
