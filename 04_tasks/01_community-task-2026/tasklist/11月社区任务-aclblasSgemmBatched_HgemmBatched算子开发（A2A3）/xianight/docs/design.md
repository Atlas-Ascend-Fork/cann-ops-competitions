# aclblasSgemmBatched / aclblasHgemmBatched 算子设计（Atlas A2/A3）

## 需求背景

本任务为 ops-blas 的 arch22 后端补齐两个指针数组批量 GEMM 接口。调用方可在一条 stream 上提交一批形状相同、物理地址各自独立的矩阵乘法，用于多路小矩阵并行，也用于单批大矩阵。需求依据为 11 月任务网页提供的《aclblasSgemmBatched / aclblasHgemmBatched 算子开发任务书》及其 S/H 两套测试资料。公开附件未标注对应的 Git commit。[任务网页](https://www.hiascend.com/activities/task-center/details/55123ab3caa64d089f525f646155094c) · [任务讨论](https://gitcode.com/cann/ops-blas/discussions/94)

仓内已有公开声明、arch35 的 S/H 实现、arch22 的复数批量 GEMM 和实数 GEMM。arch35 提供参数传递和半精度缩放的参考，arch22 的实数 GEMM 提供 Cube 调用和 FP32 模式的参考。实现范围为 `blas/gemm_batched/arch22/` 的设备代码及 Host 入口、相应构建选择、共用 GTest，以及算子 README 的产品支持表。[接口源码](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/include/cann_ops_blas.h) · [arch35 批量实现](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm_batched/arch35) · [arch22 实数 GEMM](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/arch22)

## 需求分析

两个入口共用下列参数顺序。S 的 T 是 float，H 的 T 是 aclblasHalf，H 的位模式按 IEEE binary16 解释。handle 提供执行 stream。alpha、beta 在 Host，三张指针表及每个矩阵在 Device。标量和形状在 batch 间共享，C 原地更新。

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* const Aarray[], int lda,
    const float* const Barray[], int ldb, const float* beta, float* const Carray[],
    int ldc, int batchCount);
aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha, const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb, const aclblasHalf* beta, aclblasHalf* const Carray[],
    int ldc, int batchCount);
```

| 矩阵 | 物理存储（列主序） | 前导维度下界 |
| --- | --- | --- |
| A | N 时 lda × k，T/C 时 lda × m | N 时 max(1,m)，T/C 时 max(1,k) |
| B | N 时 ldb × n，T/C 时 ldb × k | N 时 max(1,k)，T/C 时 max(1,n) |
| C | ldc × n，只更新每列的前 m 个元素 | max(1,m) |

按 handle、枚举、尺寸、前导维度、batchCount、标量和条件指针顺序检查。空 handle 返回 HANDLE_IS_NULLPTR，非法 trans 返回 INVALID_ENUM，其余结构错误返回 INVALID_VALUE（均为 ACLBLAS_STATUS 系列）。零尺寸也校验枚举、尺寸、前导维度和两个标量指针。C 的不同 batch 之间互不重叠，且与各 batch 的 A、B 不重叠，元素地址的有效性由调用方保证。

| 结构合法后的条件 | 所需表 | 动作 |
| --- | --- | --- |
| batchCount、m 或 n 为零 | 无 | 返回 SUCCESS，无设备任务 |
| k 或 alpha 为零 | C | beta 为 1 时直接成功，为 0 时写零，其余情况按列缩放 C |
| 其余情况 | A、B、C | 批量乘加，beta 为零时略过 C 旧值加载 |

## 详细设计

### 算子分析与标杆

对每个 batch b、行 i、列 j，计算 `P(i,j) = sum_p op(A_b)(i,p) * op(B_b)(p,j)`，再写 `C_b(i,j) = alpha * P(i,j) + beta * C_b(i,j)`。实数下 C 与 T 等价。A 的元素下标在 N/T 时分别为 `i+p*lda`、`p+i*lda`，B 的下标分别为 `p+j*ldb`、`j+p*ldb`。C 的下标始终为 `i+j*ldc`。每批约 2mnk 次浮点运算，读取 m×k 和 k×n 逻辑元素，写入 m×n 个结果。小矩阵和大批量容易受指针访问与下发开销限制；大型方阵主要受乘加吞吐限制。

标杆流程取 cuBLAS 公共 API 定义，S 的公开算法参照 Netlib SGEMM。下图描述数学流程。cuBLAS 内部的 kernel 调度并非本设计的依据。[cuBLAS 批量 GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmbatched) · [Netlib SGEMM](https://www.netlib.org/blas/sgemm.f)

```text
检查 BLAS 参数 → 空输出立即返回
                   ↓ 有输出
              k=0 或 alpha=0 → 按 beta 缩放或清零 C
                   ↓ 其余
      对每批取 A/B/C → 按 trans 取元素做 K 归约 → 应用标量 → 写回 C
```

### Host 与 Tiling

1. 两个入口共享结构校验和调度。按输入的实际类型读取 Host 标量并转为 FP32，下发时复制标量值。仅检查指针表本体，避免逐 batch 在 Host 读回地址。检查每次 kernel 发射状态，异步错误由调用者同步 stream 时获取。
2. 用 dtype、两项实际转置、是否只缩放、beta 是否为零、尺寸和物理步长选择模板或分支。传给设备的 tiling 包括 m/n/k、lda/ldb/ldc、batch 起点与数量、标量、块形状和尾块。batch、矩阵偏移与块数的乘积在乘法前提升到 64 位，工作区字节数单独检查溢出。
3. 普通路径把输出划成 BM × BN 的小块。设 R=ceil(m/BM)，S=ceil(n/BN)，块序号 t 对应 batch=t/(R×S)，其余序号映射到输出块行、列。并行度取平台可用核数与任务数的较小值，每核以启动核数作步长领取后续块。同一输出块只由一个核完成 K 归约。小矩阵以整矩阵或整批中的多个矩阵为单位分配，减少多次启动；长条矩阵沿长边增加块数。
4. 通用 Cube 路径从 BM=BN=64、BK=32 的候选开始，按 SoC 能力和实际形状评估扩大到 128 的收益。令 s 为输入元素字节数，A/B 双缓冲面板的净数据量为 2×(BM+BN)×BK×s，L0C 的 FP32 输出块为 4×BM×BN 字节。实际预算使用分形对齐后的各维、Matmul API 的 L1/L0 缓冲与保留量。向平台查询各级容量，切换块形状时分别核验 L1、L0A/B、L0C 和 UB；尾块按剩余行列及 K 长度裁剪。
5. 常用 alpha=1、beta=0 且写出布局支持时直接输出。需要后处理时，用句柄工作区保存 FP32 乘积并按 batch 或输出块组分段。每段记录实际行数和对齐的中间行距，工作区可容纳的块数决定段长。工作区由库的句柄机制在 stream 顺序内复用。申请失败时使用可独立运行的直接计算路径或返回相应错误。

### Kernel 与数据流

```text
从 Device 表取本批地址
    ├─ 仅缩放  → AIV 按列写逻辑区
    ├─ 小矩阵  → 合并多批下发，在寄存器或 UB 内归约并应用标量
    └─ 普通矩阵 → Cube 分块计算 Bᵀ × Aᵀ，得到列主序乘积的行主序视图
                       ├─ 可直接输出 → 写回各 C 地址
                       └─ 需后处理   → FP32 工作区 → 同 stream 的 AIV 缩放、合并并写回
```

设备核从三张表按 batch 索引取地址，同一块复用地址和标量。以行主序 Matmul 表达 `op(B)ᵀ × op(A)ᵀ`，左右操作数分别绑定 B、A，使用实际 ldb、lda 设置物理跨度。K 维面板分次搬入片上，先初始化 L0C，后续 K 块累加。S 路径关闭 HF32 截位，H 路径以 FP32 累加，完成 alpha/beta 运算后一次舍入为 FP16。输入搬运的尾部在片上补零，输出只写逻辑有效区。非对齐写出采用精确长度搬运或按目标对齐单元安排唯一写者，避免两个核对同一缓存行做读改写。

beta=0 使用独立写出分支，k=0 或 alpha=0 的分支不取 A/B 表项。标量或向量路径和 Cube 路径均以相同的特殊值语义为目标。双缓冲的同一槽位由搬入完成事件交给计算，计算完成后才能重新装载。分为两个 kernel 时通过同一 stream 保证乘积可见。测试将覆盖 FP16 次正规输入、上溢和 Inf/NaN；若设备指令冲刷次正规值，采用保留 FP16 位模式的转换路径，按实际误差逐项复核。

### 硬件、依赖与兼容性

依赖 CANN 的 Ascend C、运行时和 ops-blas 句柄。目标设备为 Atlas A2、A3（arch22）。按任务书 §3.1 规划 CANN 9.1.0 自验，选 Atlas 800T A2 / 910B3 采集正式性能数据。S 参考 Netlib BLAS 3.10.0 的 cblas_sgemm，H 的参考结果由测试工程生成。两类输入均为列主序带前导维度的独立实矩阵。调用者负责管理输入、指针表及 C 的生命周期直至 stream 完成。复用现有导出符号，通过架构选择源码及共用测试入口；arch35 的构建、测试路径独立。

## 可维可测分析

- 在 `test/gemm_batched/sgemm_batched/arch22/` 建立统一的参数化 GTest。按 S/H 数据源标记 op，以 (op,case_name) 为键，保留原 CSV、填充值、种子和独立的随机序列。测试真实上传 Device 指针表，使用非等距地址、不同 batch 数据、C 的 padding 哨兵，特别覆盖条件空指针、k=0、alpha=0、beta=0 下 NaN 未读分支与 OP_C。核对九种转置组合、4095³/4096³、大批量 1×1 和 65536 个指针。编译入口为 `bash build.sh --ops=sgemm_batched,hgemm_batched --soc=ascend910b3`。
- 每个算子执行 500 条精度用例。负向用例核对确切返回码，正向用例按 batch 核对每个逻辑元素和填充哨兵。S 以 cblas_sgemm 为参考，H 的量化输入与标量扩展到 float64，乘加后收窄到 FP32，再按就近偶数舍入到 FP16。遇相同符号的无穷和成对 NaN 时单独核对，记录不一致的非有限值。有限输出以 `abs(error) <= atol + rtol * abs(golden)` 统计匹配占比，S 的 atol=rtol=2⁻¹³，H 为 2⁻¹⁰。每例匹配率至少 0.99，并核对最大绝对误差上限（S 为 0.01，H 为 0.1，或按参考值对应的 32 ULP）；记录最差元素、绝对误差及 ULP。
- 每个算子 100 条性能用例，以 dtype、transA/B、m/n/k、lda/ldb/ldc、batchCount、alpha/beta 匹配同参 A100 基线。设备缓冲预先准备，5 次预热后连续成功执行 10 次。保留 GTest 的 aclrtEvent 平均耗时和 msprof op 原始 Task Duration。多 kernel 的一次 BLAS 调用合计其全部计算及后处理 kernel 耗时，再对 10 次求平均。每例要求平均 kernel 微秒数 ≤ 1250 × gpu_ms，另列 GPU/NPU 比值。记录 SoC、软件版本、设备负载、样本数和异常。内存验收按 §3.4 标明不涉及。
- 自验步骤从完整实现 commit 开始，登记输入、基线、golden、实际输出与脚本版本，按当前运行的原始日志生成报告及所需截图。用例缺失、重复、程序异常或没有时间样本时均记录失败。测试材料保留输入与输出的对应关系，文档引用可随包获取的文件。
