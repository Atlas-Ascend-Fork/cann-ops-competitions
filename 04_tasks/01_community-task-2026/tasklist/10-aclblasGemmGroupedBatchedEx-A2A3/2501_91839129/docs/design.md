# 需求背景（required）

## 需求来源

[10月社区任务-aclblasGemmGroupedBatchedEx 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center) 要求基于 [ops-blas](https://gitcode.com/cann/ops-blas) 开源仓，使用 Ascend C 在 Atlas A2/A3 系列产品（含 Atlas 800T A2 / Atlas 800T A3，对应 arch22）上补齐单精度实数（float32）扩展版分组批量通用矩阵乘法算子 `aclblasGemmGroupedBatchedEx`，完成算子设计、开发、测试全流程。接口语义参考 cuBLAS 的 grouped GEMM 扩展接口，验收范围以任务书及随附的 1000 条精度用例、200 条性能用例为准。

## 背景介绍

分组批量 GEMM 将若干 GEMM 实例组织成 `groupCount` 个“组”。同一组内的所有实例共享形状（`m/n/k`）、转置方式、前导维（`lda/ldb/ldc`）与标量系数（`alpha/beta`）；不同组之间这些属性可以不同。组 `g` 含 `groupSize[g]` 个实例，实例总数为 `Σ groupSize[g]`。每个实例的 A/B/C 矩阵采用**列主序**存储，通过 Device 指针数组按实例展平传入（组内连续、组间顺序排列），不要求地址连续。

实现复用 `include/cann_ops_blas.h` 中的公共接口声明，仅在 `blas/gemm_grouped_batched_ex/arch22/` 新增 Host 与 Kernel，不改动 arch35 的接口或实现，也不新增产品私有平行接口。

# 需求分析（required）

## 功能与接口

对每个实例 `idx`（属于组 `g`）计算：

\[
C_{idx}(r,c)\leftarrow\alpha_g\sum_{p=0}^{k_g-1}\operatorname{op}(A_{idx})(r,p)\,\operatorname{op}(B_{idx})(p,c)+\beta_g\,C_{idx}(r,c),
\quad 0\le r<m_g,\;0\le c<n_g.
\]

`op(X)` 在 `N` 模式取原矩阵、`T` 模式取转置、`C` 模式取共轭转置；实数路径中 `C` 与 `T` 等价。`op(A)` 为 `m×k`，`op(B)` 为 `k×n`，输出 `C` 为 `m×n`，原地覆写。

```cpp
aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void* alphaArray,
    const void* const Aarray[], aclDataType Atype, const int ldaArray[],
    const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray, void* const Carray[], aclDataType Ctype, const int ldcArray[],
    int groupCount, const int groupSize[], aclblasComputeType_t computeType);
```

| 参数 | 语义与范围 |
| --- | --- |
| handle | Host 上的有效句柄，计算下发到其绑定的 stream。 |
| transaArray、transbArray | 长度 `groupCount`，每组取 `ACLBLAS_OP_N/T/C`。 |
| mArray、nArray、kArray | 长度 `groupCount`，每组的非负维度。 |
| alphaArray、betaArray | 长度 `groupCount` 的 FP32 标量数组（本设计按 Host 驻留处理，见下文限制）。 |
| Aarray、Barray | 长度 `Σ groupSize` 的 Device 指针数组，按实例展平，每项指向只读 Device 矩阵。 |
| Carray | 长度 `Σ groupSize` 的 Device 输出指针数组，每项指向可写且互不重叠的 Device 矩阵。 |
| Atype、Btype、Ctype | 本 arch22 设计支持组合为 `ACL_FLOAT/ACL_FLOAT/ACL_FLOAT`。 |
| ldaArray、ldbArray、ldcArray | 长度 `groupCount`。`lda≥max(1,m)`（N）或 `max(1,k)`（T/C）；`ldb≥max(1,k)`（N）或 `max(1,n)`（T/C）；`ldc≥max(1,m)`。 |
| groupCount | 非负 int。为 0 时为空操作，返回成功。 |
| groupSize | 长度 `groupCount`，每组实例数（非负）。 |
| computeType | 支持 `ACLBLAS_COMPUTE_32F`；其余返回 `NOT_SUPPORTED`。 |

FP16/BF16/FP8/INT8 等扩展组合不在本次 arch22 支持集内，返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；README 按产品分别标明支持范围，保留 arch35 既有能力。

**已知限制：** 本实现假定 `alphaArray/betaArray` 为 **Host 驻留**内存（与随任务提供的 GTest 测试框架一致——该框架以 Host 标量指针调用接口，1000/200 条用例均不涉及 Device 标量）。若调用方传入 Device 标量指针，需后续补充指针属性查询分支；该增强不影响本任务验收范围。

## 参数校验与边界行为

Host 先按固定顺序校验，再判断 no-op，最后下发计算：

1. `handle==nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. 数据类型/计算类型非 FP32 组合 → `ACLBLAS_STATUS_NOT_SUPPORTED`；
3. `groupCount<0` → `ACLBLAS_STATUS_INVALID_VALUE`；`groupCount==0` → 成功（空操作）；
4. 任一 per-group Host 数组（trans/m/n/k/alpha/lda/ldb/beta/ldc/groupSize）为空 → `ACLBLAS_STATUS_INVALID_VALUE`；
5. 非法转置枚举 → `ACLBLAS_STATUS_INVALID_ENUM`；负 `m/n/k/groupSize` 或前导维低于下限 → `ACLBLAS_STATUS_INVALID_VALUE`；
6. 实例总数 `>0` 且 `Aarray/Barray/Carray` 任一为空 → `ACLBLAS_STATUS_INVALID_VALUE`；实例总数 `==0` → 成功。

上述指针校验只检查数组本身，不在 Host 解引用 Device 指针数组。

| 条件（完成参数校验后） | 行为 | 访存要求 |
| --- | --- | --- |
| groupCount=0 或实例总数=0 | 返回成功，不下发计算。 | 不读取矩阵或标量。 |
| 实例 m=0 或 n=0 | 该实例跳过。 | 不读取其矩阵。 |
| k=0 或 alpha=0，beta=1 | C 保持原位（identity）。 | 不读 A/B，不写 C。 |
| k=0 或 alpha=0，beta=0 | C 有效元素写正零（EXACT）。 | 不读 A/B 或旧 C。 |
| k=0 或 alpha=0，其余 beta | 逐元素执行 `C=beta*C`。 | 不读 A/B，保留 C padding。 |
| 常规乘加，beta=0 | 仅写 `alpha*op(A)*op(B)`。 | 不读旧 C，避免 `0*Inf/NaN` 污染。 |

实现以两个逐实例标志刻画上述语义：`hasGemm = (m>0 && n>0 && k>0 && alpha≠0)`，`writesC = (m>0 && n>0 && (hasGemm || beta≠1))`。无任何 `writesC` 实例时整体返回成功，不分配 workspace、不下发 Kernel。

# 详细设计（required）

## 算子分析

### 地址映射

列主序下，以元素为单位的访问偏移为：

\[
a(r,p)=\begin{cases}r+p\,lda,&transa=N\\p+r\,lda,&transa=T/C\end{cases},\quad
b(p,c)=\begin{cases}p+c\,ldb,&transb=N\\c+p\,ldb,&transb=T/C\end{cases},\quad
c(r,c)=r+c\,ldc.
\]

Host 把每个组的属性展平成逐实例的 `GroupedGemmExProblem` 记录（原始列主序形状与前导维、指针数组下标、alpha/beta、路径标志、workspace 偏移），随 `GroupedGemmExHeader` 一并拷到 Device tiling 区。偏移、分块与 workspace 字节数均用经过溢出检查的 64 位运算。

### 计算结构：列主序到行主序的免转置映射

核心恒等式是把列主序结果当作行主序来写：

\[
C_{cm}\ (\text{按行主序读}) \;=\; (C_{cm})^{\mathsf T}\ (\text{行主序}) \;=\; \operatorname{op}(B)^{\mathsf T}\,\operatorname{op}(A)^{\mathsf T}.
\]

即令 Matmul 的 `M=n, N=m, K=k`，计算行主序 `Y(n×m)=op(B)^T · op(A)^T`，则 `Y` 的行主序字节布局恰好等于 `C` 的列主序布局。于是 Matmul 的左操作数来自 **B 缓冲**、右操作数来自 **A 缓冲**，且在满足前导维约束时可**直接原地读取**，无需显式转置数据。

## 算子实现

### Host 侧设计

1. 完成接口校验与 no-op 判定（上表语义）。
2. 逐实例构建 `GroupedGemmExProblem`，并据通用输入属性（转置、前导维、k、alpha/beta）选择执行路径标志：
   - **smallK**：`hasGemm && transA=N && k≤16`，走 AIV 向量路径；
   - **useDirect**：cube 能原地读 A/B——非转置操作数的 GM stride 被 Matmul 强制为 `orgM/orgN`，故要求 `transA=T` 或 `lda==m`，且 `transB=N` 或 `ldb==n`；否则走 pack 把操作数整理成紧凑行主序；
   - **directC**：`hasGemm && alpha==1 && beta==0`，cube 直接把结果写入 C（`orgKc=ldc`），省去 epilogue 与 Ydense。
3. 规划 workspace，由两个相互独立的标志驱动：`useDirect==0` 的 cube 实例需 `Adense(n×k)+Bdense(k×m)`（pack 操作数）；`directC==0` 的 cube 实例需 `Ydense(n×m)`。因此 `directC` 实例仅省去 Ydense——只有当它**同时** `useDirect`（前导维紧凑）时才完全不占 workspace；smallK 与纯 beta 实例不占 workspace。各块按 512B（128 float）对齐。因 cube 全量覆写其 Ydense、epilogue 对纯 beta 实例以 `Y=0` 代入，**无需对 workspace 清零**。
4. 查询 AIV/AIC 核数，在 handle 绑定的 stream 上顺序下发各阶段 Kernel；运行时失败按仓内状态码传播，返回前 `aclrtSynchronizeStream`。

### Kernel 侧设计

最多四个 Kernel，Host 按需只下发有工作量的阶段（例如 `alpha=1,beta=0` 的常规用例只发 dense 一个 Kernel，避免空 Kernel 的固定启动开销）。同一 stream 顺序下发，依靠流序提供 `pack → dense → epilogue` 依赖；smallK 独立于前三者：

- **pack（AIV）**：仅处理非 `useDirect` 的 cube 实例。按 `Adense[r][c]=op(B)[c][r]`、`Bdense[r][c]=op(A)[c][r]` 把列主序操作数整理成紧凑行主序，连续行用 `DataCopy`、转置方向逐元素搬运，吸收 padding 前导维。每个实例由单核独占，避免跨核写同一 cache line。
- **dense（AIC）**：FP32 `matmul::Matmul`（`MatmulType<GM, ND, float>`），静态 tiling 单核形状上限 4096、base 128/128/64。`useDirect` 实例以转置标志 + `SetOrgShape(n,m,orgKa,orgKb,orgKc)` 原地读 B/A，其余读紧凑的 Adense/Bdense；`directC` 实例 `orgKc=ldc` 直写 C，否则 `orgKc=m` 写 Ydense。两处关键性能设计：
  - **二维输出分块铺满 cube 核**：把输出 `(M=n 行 × N=m 列)` 切成 `128×512` 的 tile（均 32B 对齐、分别为 baseM/4×baseN 的整数倍），所有实例的全部 tile 组成一个全局任务流，以 `task % coreNum == coreId` 轮询分核。相比"一实例一核/只切行"，单个大实例也能占满全部 cube 核，多实例用例天然负载均衡。K 轴由 `IterateAll` 内部流式归约。
  - **Matmul 对象全程复用**：`Init` 移出 tile 循环，一次创建、跨所有 tile/实例复用（每 tile 仅 `SetOrgShape/SetSingleShape/SetTensorA/B/IterateAll`）；避免逐 tile 重新分配 L0/L1 的开销。
  - 约束：`m`（Matmul N）与 `k`（Matmul K）不跨 tile 再切分，故要求单实例 `m≤4096` 且 `k≤4096`（覆盖本任务最大 4096；超出者不在本任务范围）。
- **epilogue（AIV）**：处理 `writesC && !directC && !smallK` 的实例，向量化融合 `C[:,j]=alpha*Y[j,:]+beta*C[:,j]`（C 列与 Y 行均为连续 m 元素，`DataCopy+Muls+Add`）。纯 beta 实例以 `Y=0` 代入即得 `C=beta*C`。
- **smallK（AIV）**：对 `transA=N` 的小 k 实例，利用 `C[:,j]=alpha·Σ_p op(B)[p,j]·op(A)[:,p]`——每个 `op(A)[:,p]` 是连续 m 元素，整列化为 k 次向量 `Axpy`。A 分片按行块一次装载后在列块内复用，B 标量按对齐步长预存 UB。

**FP32 说明：** cube 路径经 Ascend C Matmul API 以 FP32 输入/累加执行（与仓内 ssymm arch22 FP32 路径一致），不以 FP16/BF16 替代本任务的 FP32 计算。smallK 路径为 Vector 单元原生 FP32。

### 性能优化（按实测演进）

性能以官方 `msprof op` 的 device 侧 task duration 衡量（见"性能标准与测试方法"）。关键优化及其对附件基线的 device 中位比值（越小越好）：

| 阶段 | 关键改动 | device 中位比值 |
| --- | --- | --- |
| 基线 | 每实例单核 + 全量 pack | 很差（墙钟口径曾误显 278×） |
| +免转置直喂 | 紧凑前导维直接原地喂 cube，仅 padding 才 pack | — |
| +直写 C | `alpha=1,beta=0` 省 epilogue 与 Ydense | — |
| +smallK 向量路径 | `k≤16, transA=N` 走 Vector Axpy（并修复极端值精度，见下） | ~1.25× |
| +Matmul 复用 + 按需启动 | Init 移出循环复用对象；空闲 Kernel 不发射 | — |
| **+二维输出分块（最终）** | **输出 `128×512` tile 铺满全部 cube 核** | **0.86×** |

二维分块是决定性的一招：单个大实例也能占满 cube 核。典型大用例 device 实测：4096³ 由 10.6ms 降到 7.27ms（0.93×基线）、2048³ 1.01ms（0.85×）、1024³ 0.18ms（0.86×），4096³ 持续算力约 18.8 TFLOPS。早期墙钟口径的"放慢几十倍"是 host 下发/同步固定开销造成的假象，device 口径才是正确衡量。

## 支持硬件与工程组织

| 产品 | 支持范围 |
| --- | --- |
| Atlas A2 系列（含 Atlas 800T A2、910B3） | FLOAT32 主路径，arch22。 |
| Atlas A3 系列（含 Atlas 800T A3） | FLOAT32 主路径，arch22。 |

实现位于 `blas/gemm_grouped_batched_ex/arch22/`（`*_host.cpp` / `*_kernel.cpp` / `*_tiling_data.h`），测试位于 `test/gemm_grouped_batched_ex/`（公共 `*_param.h` / `*_golden.h` 与 `arch22/` 下的 `*_test.cpp` / `*_npu_wrapper.h` / CSV）。沿仓内 GTest、CSV 与 CMake 按 SoC 自动注册机制接入，使用公共头既有签名，不新增产品私有 API。

# 可维可测分析（required）

## 精度标准与测试方法

以 Netlib BLAS 实数 `cblas_sgemm` 逐实例生成 golden，比较每个输出矩阵的有效 `m×n` 元素，并以哨兵核对 padding 未被改写。FLOAT32 采用仓内 mixed-tolerance 判据：匹配比例 ≥ 0.99，逐元素绝对误差上限 `max(1e-2, 32*ULP)`，`rtol/atol` 取 FP32 标准值；CSV 的 MERE/MARE 配置不替代该判据。非有限值按任务书等效规则统计（同为 NaN、同号 Inf 等效）。

测试覆盖随任务提供的 **1000 条精度用例**（基础、全转置组合、M/N 扫描、alpha/beta、前导维 padding、分组、特殊值、边界/负向、扩展遍历）与 **200 条性能用例**。负向用例仅比对返回状态码。

## 性能标准与测试方法

性能设备为 Atlas 800T A2（910B3）。判据存在两档：任务书 §3.3 要求 `T_NPU ≤ 1.25·T_GPU`，§7.4（特别注意事项）要求 `T_NPU ≥ GPU数据` 即 `T_NPU ≤ 1.0·T_GPU`，报告两档均列出。以 `gpu_baseline.csv` 的 200 条基线逐例对齐（`groupCount/trans/m/n/k/groupSize`）比对。正式采集须用官方 `msprof op` 采 task duration（随附 README 要求 5 次 warmup 后 >50 次有效样本取平均，异常样本剔除并说明）。

**基线设备说明：** `gpu_baseline.csv` 无设备字段；随附 README 自相矛盾——一处称 `gpu_ms` “已由 A100 实测回填”，另一处称“当前为 NVIDIA P102-100 实测值”。故本文不断言基线型号，仅按附件数值逐例比对。

**device 实测结论（`msprof op`，910B3，每例 5 warmup + 51 有效样本，全量 200 条）：** 对附件基线 device 中位比值 **0.857×**，`≤1.25×`（§3.3 口径）**179/200**、`≤1.0×`（§7.4 口径）**139/200**、`≤2×` **195/200**，最优 0.11×。未达标 21 条集中在极小形状（8/16/32 方阵）与小 k（k≤32）大 n 用例——固定开销/访存占比高；其中少数 small-k 用例若改走 cube 会更快（见优化方向）。这与同族 `aclblasSgemmGroupedBatched` 社区提交实测 200/200 达标相互印证——`≤1.25×` 在 910B3 上对 FP32 分组批量 GEMM 是可达的。

**测量口径订正：** 早期曾用 GTest 内 `chrono` 墙钟（含 Host malloc/launch/同步开销）误判为"放慢几十倍、门槛不可达"；改用 `msprof op` 的 device task duration 后，真实计算性能与基线基本持平或更优。正式数据一律以 `msprof op` 为准。

**进一步优化方向（针对剩余极小形状）：** 小形状自适应 tile 分档（避免宽 tile 的并行度浪费）、beta 折叠入 C 以对所有 alpha/beta 走零 workspace 路径、smallK 双缓冲 + 细粒度 `SetFlag/WaitFlag`。

**workspace：** 性能用例（`alpha=1,beta=0`）全部命中 directC，几乎不占 workspace；需 pack/epilogue 的实例按 `Adense+Bdense+Ydense` 规模使用，均为一次性规划、无常规调用反复申请。

## 兼容性分析

仅新增 arch22 实现与对应测试，公共接口、状态码及 arch35 支持范围保持兼容。A2/A3 按仓内 SoC 配置分别编译，性能结论以 910B3 实测为依据。
