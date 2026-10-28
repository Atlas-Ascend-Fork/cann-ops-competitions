# 需求背景

## 需求来源

[10月社区任务-aclblasGemmBatchedEx算子开发（A2A3）](https://www.hiascend.com/activities/task-center)要求在 Atlas A2/A3 系列产品上，基于 [ops-blas](https://gitcode.com/cann/ops-blas) 开源仓工程框架，使用 Ascend C 编程语言实现扩展版批量通用矩阵乘法算子 `aclblasGemmBatchedEx`。接口语义、精度标准与性能判据以该任务书为依据，对标 cuBLAS `cublasGemmBatchedEx`。

## 背景介绍

`aclblasGemmBatchedEx` 对 `batchCount` 个维度一致的独立矩阵三元组计算 `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`。矩阵均为列主序，通过 Device 指针数组 `Aarray/Barray/Carray` 传入，各批次共享 m/n/k、alpha/beta 与 leading dim。`transa/transb` 支持 N/T/C 三种操作（实数路径下 C 与 T 等价）。本算子补齐 ops-blas 仓批量 GEMM Ex 能力，接口声明复用 `include/cann_ops_blas.h`，与现网及 950PR 共用，禁止定义产品私有平行接口。

# 需求分析

## 算子功能

对第 `i` 个批次（`i = 0..batchCount-1`）计算：

```
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
```

其中 `op(A_i)` 为 m×k、`op(B_i)` 为 k×n、`C_i` 为 m×n，均为列主序。`op(X)` 由 trans 参数决定：N 取原矩阵、T 取转置、C 取共轭转置（实数下与 T 等价）。

边界行为对齐 cuBLAS / Netlib `sgemm` quick return 语义：

- `m=0`、`n=0` 或 `batchCount=0` 为合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`；
- `k=0` 时只做 `C = beta * C`，不引用 A、B；
- `alpha=0` 时跳过矩阵乘，`C = beta * C` 逐元素精确（EXACT 校验）；`beta=0` 时不引用 C 的旧值（含 0×Inf 不产生 NaN 的语义）；
- 参数校验先于 no-op 判定：`batchCount > 0` 时指针数组非空，alpha/beta 指针非空，lda/ldb/ldc 满足前导维约束，非法枚举/非法维度/不支持组合返回对应错误码。

## 公开接口与范围

在 `include/cann_ops_blas.h` 既有声明上实现，函数原型与 cuBLAS 对齐：

```cpp
aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* const Aarray[], aclDataType Atype, int lda,
    const void* const Barray[], aclDataType Btype, int ldb,
    const void* beta,
    void* const Carray[], aclDataType Ctype, int ldc,
    int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);
```

数据类型面按算子 README 宣称覆盖：Atlas A2/A3（arch22，910B3 无 FP32/FP8 Cube）支持 FP16 输入（COMPUTE_16F / COMPUTE_32F）与 BF16 输入（COMPUTE_32F），FP8 等不支持组合在 host 校验即返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。`m/n/k/batchCount` 为运行时入参，由 host tiling 支持任意合法组合；不要求非连续 Tensor、broadcast 与确定性计算；C 原地覆写，各批次输出区域不得重叠。算子依赖 `aclblasSetStream` 绑定 stream，读回结果前须同步 stream。

# 详细设计

## 算子分析

### 数学公式

`C_i(r, c) = alpha * Σ_p op(A_i)(r, p) * op(B_i)(p, c) + beta * C_i(r, c)`，r∈[0,m)、c∈[0,n)、p∈[0,k)。fp16/bf16 输入在 Cube 上以 fp32 累加，输出按 Ctype 量化；alpha/beta 合成在 fp32 域完成后单次量化，保证混合精度语义。

### 支持数据类型

FP16（COMPUTE_16F / COMPUTE_32F）、BF16（COMPUTE_32F）；alpha/beta 为 Host 侧 float 标量指针。

### 支持形状

m、n、k、batchCount 运行时任意（≥0）；覆盖 1、小质数、2 的幂及 ±1、非对齐值直至大规模（2048 量级）；ld 支持紧凑与多档 padding。

## 算子实现

总体架构为「裸 Cube 矩阵乘主路径 + AIV 向量/标量辅助路径」分层调度：AIC Cube 承担矩阵乘主体，AIV 承担转置 staging、alpha/beta 尾处理、微形状与小批量路径，host 依据形状与标量参数选择执行路径。

### Host 侧设计

**参数校验**：句柄为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；transa/transb 非法返回 `ACLBLAS_STATUS_INVALID_ENUM`；m/n/k/batchCount 为负返回 `ACLBLAS_STATUS_INVALID_VALUE`；alpha/beta 为空、`batchCount>0` 时指针数组为空、前导维不满足 `transa=N ? lda≥max(1,m) : lda≥max(1,k)` 等约束均返回 `ACLBLAS_STATUS_INVALID_VALUE`；dtype/computeType/algo 不在支持集返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。校验通过后先判 no-op 与 quick return。

**列主序视角转换**：C(m×n 列主 ldc) 与 C^T(n×m 行主 ld=ldc) 内存布局等价。host 将问题交换为行主视角 `C^T = op(B)^T · op(A)^T`（m↔n、lda↔ldb、transa↔transb 互换），使 Cube 按 行主 × 行主 → 行主 的自然方向计算，C 直接落在用户缓冲区。

**执行路径决策**（按序判定）：

1. `k=0` 或 `alpha=0`：early-exit，AIV 全列 `[0, ldc)` 执行 `C = quantize(beta * C)`（golden 校验覆盖整个 ldc·n 缓冲，含 ld-padding；beta=0 时置零且不读 C）；`beta=1` 且 `alpha=0` 为纯 no-op。
2. `alpha==1 && beta==0`（直写类）：
   - 微工作量（m·n·k·batch ≤ 阈值）：单次 AIV 向量 GEMM kernel 直读列主输入，免 Cube 装载路径；
   - `m%16≠0`：m-split——主行 `[0, mMain)`（mMain = m 向下取 16 对齐）走 Cube 直写，尾行 `[mMain, m)` 走 AIV 向量 GEMM（rowBase 偏移），两路写区不相交；
   - 其余：Cube 直写。
3. 非 direct 且形状带微维/尾块（min(m,n,k)<16 或 m%16≠0 或 k%16≠0）且工作量在阈值内：AIV 标量 GEMM，纯标量 UB 运算，不涉及 Cube 微结构。
4. 其余：fp32 temp 路径——Cube 将原始积写独立 temp，host 同步后 AIV epilogue 执行 `C = quantize(alpha * temp + beta * C)`。

直写条件由 fixpipe 写出粒度决定：Fixpipe 以 16 元素分形对齐写 GM，`nSize%16≠0` 时写出区间扩展到 `[curN, ceil16(curN))`；仅当 m（swap 后的 Cube n 维）16 对齐时该扩展为零，可安全直写用户 C 且不触碰 ld-padding。`alpha≠1` 或 `beta≠0` 时量化前需 fp32 合成，统一走 temp；temp 行距 `tempLd = max(ldc, ceil16(min(m,128)))`，使分形扩展落入本行 padding。

**staging 转置**：`transa≠N` 时 B'-operand（= A 的 k×m 列主）不满足 Cube L0B 的行主装载要求（arch22 L0B 不支持列主 K×N 源），host 先下发 AIV staging kernel 将 A 逐 16×16 子块 Gather 转置为行主 workspace（偏移表按字节单位预生成），Cube 从 staged 缓冲装载。staging workspace 按 (perBatch 字节数, batchCount) 键驻留缓存，同形状重复调用免分配与指针数组 H2D；换形状时在流排空后重分配。

**任务分解与启动**：Cube task = (batch, 128×128 输出块)，k 循环全内循环，`usedCores = min(AIC 核数, 总任务数)`；各 AIV kernel 以列/块为单位网格化，块数取 `min(AIV 核数, 任务数)`。temp/staging 的逐 batch 基址补齐到 32B。全部 kernel 在 handle 绑定 stream 上按序启动。

### Kernel 侧设计

**Cube 主 kernel**（AIC_ONLY，BASE_M=BASE_N=128、BASE_K=64）：单 task 内 k 分块循环，每迭代执行装载原语链

```
Nd2Nz(GM → L1 NZ 布局) → LoadData3D(L1 → L0A/L0B) → Mmad(L0 × L0 → CO1, fp32 累加)
```

k 循环结束后 Fixpipe 将 CO1 写 GM（直写路径 F322F16/F322BF16 单次量化；temp 路径 NoQuant 写 fp32）。装载参数：arch22 Nd2Nz 元素映射 `elem(n,d) = base + n·srcDValue + d`（d 连续），`dstNzC0Stride` 取 k 向对齐值；LoadData3D 以 `l1W=dstNzC0Stride`、`channelSize/mExtension` 对应 k/n 对齐量装载 L0 分形；k 分块经再平衡保证单块 `curK≥16`（避免 d0=1 分形的装载边界）。

**管道同步设计**（对齐仓内 cherk/cher2k 已验收信封）：Nd2Nz 下发后 `Set<MTE2_MTE1>`，LoadData3D 前 `Wait<MTE2_MTE1>`；LoadData3D 后 `Set<MTE1_M>`，Mmad 前 `Wait<MTE1_M>`；Mmad 后全量屏障并 `Set<M_MTE1>`，下一迭代 LoadData3d 前 `Wait<M_MTE1>`（L0 复用保护）；task 首部 `Wait<FIX_M>` 等待上一 task 的 Fixpipe 退役（CO1 复用保护），Fixpipe 前 `Set/Wait<M_FIX>` 确认 Mmad 完成，Fixpipe 后 `Set<FIX_M>`；kernel 尾部 `Wait<FIX_M> + Wait<M_MTE1>` 排空，保证 kernel 完成时全部 GM 写已落地，同流后续 kernel 无需额外同步即可读。kernel 初始化调用 `SetHF32Mode(HF32Mode::DISABLE)`（HF32 模式下 fp16 Mmad 吞吐显著劣化）。

**边界与硬件细节**：curM=1 时 Mmad 以 m=2 发射（CO1 填充行由 Fixpipe 按 mSize 裁剪）；L1 尾块区先以 64 元素分块 Fill 零填充再由 Nd2Nz 重写有效区，规避大 blockNum 的静默截断；AIV 向量指令 count 上限 256；子 32B 的二维块写出改为逐列一维 DataCopyPad（任意字节粒度已验证）；AIV 侧标量读取向量写入的 UB 数据前执行 `V_S` 事件握手，标量写入的 Gather 偏移表在读取前执行 `S_V` 事件握手。

**AIV 向量 GEMM kernel**（微形状与 m-split 尾行路径）：以 16×16 输出块为任务，A/B 各以 16×16 tile 经 DataCopyPad（目的侧补齐 16）装载并 Cast 至 fp32；转置需求（tA=T / tB=N 的装载方向）经预生成字节偏移表 Gather 完成；MAC 采用 `Axpy(acc[jj], aTile[pp], bScalar, 16)` 单指令形式（acc[jj][ii] += A(ii,pp)·B(pp,jj)，lane 为行方向）；k 全程分块累加，尾块以 alpha 量化后逐列写出，仅写有效行（不触碰 ld-padding）。beta=0 的该路径无 C 读取。

**AIV 标量 GEMM kernel**（非 direct 微维/尾块兜底）：按输出列任务化，B 列向量与 A 16 行块分别装载，标量域 fp32 乘加累加后合成 `alpha·acc + beta·C_orig` 并量化写回，语义完整覆盖 alpha/beta 任意组合。

**AIV epilogue / early-exit kernel**：temp 路径按行分块（CHUNK=256）读取 fp32 原始积，与 beta·C 在 fp32 域合成后量化写回，仅写 `[0, m)` 有效区（ld-padding 保留原值）；early-exit 按列处理全 `[0, ldc)`。

### 资源与调度

Cube 单 task：A1/B1 各 16KB（L1）、A2/B2 各 16KB（L0）、CO1 64KB（L0C），单核内 k 循环顺序复用，无跨 task 共享；AIV 向量 GEMM 单任务 UB 约 15KB；AIV 标量 GEMM UB 约 15KB（含 2048 元素 B 列缓冲，k 超 2048 时按块累加）；epilogue/early-exit UB 按 256 元素分块。staging workspace 大小 = batchCount × k × ceil16(m) × 2B + 指针数组；temp 大小 = batchCount × tempLd × n × 4B + 指针数组，均按批 32B 对齐。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I/T A3 | √ |

算子代码位于 `blas/gemm_batched_ex/arch22`，自验证设备 Atlas 800T A2 (910B3)。

## 算子约束限制

- dtype 面按 README 宣称：FP16 / BF16；FP8 及其它组合返回 NOT_SUPPORTED；
- 仅支持列主序与 lda/ldb/ldc 语义内的访问，不支持超 ld 语义的非连续 Tensor；
- C 原地覆写，各批次 C 缓冲不得重叠；beta≠0 时 C 为有效输入；
- 非 ld-padding 区外的填充区域不被写路径触碰（量化直写路径仅写 `[0,m)` 行、`[0,n)` 列）；
- 浮点累加顺序与参考实现不同，结果非 bit-exact（`alpha=0` 的 `C=beta·C` 除外，该路径位精确）；
- 性能口径要求调用方传入 Device 指针数组（ACL 契约），矩阵数据需驻留 Device 内存。

# 可维可测分析

## 精度标准与验证

golden 由 cblas（Netlib BLAS 3.10.0 实数 `cblas_sgemm`）逐批次生成，按 Ctype 混合容差判定（生态算子开源精度标准）：FP16 `rtol=atol=2^-10`、BF16 `rtol=atol=2^-7`，`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max(1e-2, 32×ULP)`；逐元素通过条件 `|actual-golden| ≤ atol + rtol×|golden|`；Inf/NaN 按同 NaN/同号 Inf/一 NaN 一 ±Inf 等效规则单独判定，不计入有限值统计；`alpha=0` 用例位精确（EXACT）。测试工程位于 `test/gemm_batched_ex/`，CSV 驱动 GTest：覆盖 shape 扫描（1、质数、2 的幂 ±1、非对齐、至 2048 量级）、转置全组合（N/T/C × N/T/C）、批数扫描（0/1/小/中/大/2 的幂）、ld 紧凑与 padding、alpha/beta 特殊值（0/1/负/大值/空指针）、Inf/NaN 输入、全部负向用例（空句柄/非法枚举/负维度/非法前导维/空指针数组），配套 README 说明复现步骤。

## 性能标准与验证

测试设备 Atlas 800T A2 (910B3)，测试输入矩阵驻留 Device 内存，`gpu_baseline.csv` 提供同形状 GPU 标杆耗时（按 dtype 比例推导），判据：平均单次耗时 ≤ GPU 标杆 / 0.8。采样方式：每用例 5 次 warmup 后连续采样取平均，异常样本（返回非 SUCCESS）不计入并说明。200 条性能用例覆盖 2 的幂立方体（256³~2048³ 级）与批量组合、奇数/非对齐形状（49/97/198/628/981 级）、全部转置组合、批数 1~202 扫描。测试框架与用例随算子合入 `test/gemm_batched_ex/`，CSV 中精度与性能用例分列（TC_EX_* / TC_PF_*），验收人可按 README 步骤复现精度与性能数据。
