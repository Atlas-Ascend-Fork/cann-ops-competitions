# aclblasCgemm3m 算子设计（Atlas A2/A3）

## 需求背景

`aclblasCgemm3m` 在调用方指定的流上执行单精度复数矩阵乘加。目标是在 CANN 9.1.0 的 Atlas A2/A3 上提供与 cuBLAS `cublasCgemm3m` 对应的列主序 BLAS 接口。复数乘法采用 Gauss 3M 分解，将矩阵乘法的主体交给三次实数矩阵乘。[任务书及附件](https://www.hiascend.com/p/resource/202611/21b52bc81a1f43b0b5f59e0f337e9665.zip)

## 需求分析

输出在原地址更新为 `C = alpha · op(A) · op(B) + beta · C`。`op(A)`、`op(B)` 和 C 的逻辑形状分别为 `m×k`、`k×n` 和 `m×n`。每个操作数可独立选择 N（原矩阵）、T（转置）或 C（共轭转置）。例如，当 A 只有一个元素 `1+2i`、B 只有一个元素 `3+4i`，且 alpha=1、beta=0、transb=N 时，transa=T 得到 `−5+10i`，transa=C 得到 `11−2i`。这个例子可直接检出漏取共轭的问题。

## 详细设计

### 算子原型

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
```

| 参数 | 类型、位置与约束 |
| --- | --- |
| handle | 有效的 BLAS 句柄。kernel 提交到绑定的 stream。 |
| transa、transb | 分别取 `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 或 `ACLBLAS_OP_C`。 |
| m、n、k | 非负整数。 |
| alpha、beta | Host 侧非空标量指针。每个 `aclblasComplex` 包含两个 FP32 分量。 |
| A、lda | Device 输入。N 时按 `lda × k` 存储，`lda ≥ max(1,m)`。T/C 时按 `lda × m` 存储，`lda ≥ max(1,k)`。 |
| B、ldb | Device 输入。N 时按 `ldb × n` 存储，`ldb ≥ max(1,k)`。T/C 时按 `ldb × k` 存储，`ldb ≥ max(1,n)`。 |
| C、ldc | Device 上的原地输出，物理尺寸 `ldc × n`，`ldc ≥ max(1,m)`。调用方保证可写区域与 A、B 的可读区域互不重叠。 |

### Host 参数校验与分派

1. 按顺序检查句柄、转置枚举、维度、前导维度和标量指针。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法维度或必需的空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. 结构参数合法时，m 或 n 为零立即返回成功，允许 A、B、C 为空，无需下发 kernel。对于非空输出先检查 C。若 k=0 或 alpha=0，允许 A、B 为空，只执行 C 的缩放。beta=1 可直接返回，beta=0 逐列写零，其他 beta 读取旧值做复数缩放。
3. 一般乘加路径需要有效的 A、B。Host 依据逻辑尺寸、矩阵布局和片上容量选择输出分块及 K 分块。地址偏移、块数和工作区字节数用检查过的宽整数计算。临时存储随流上的最后一次使用完成后复用，调用方可连续异步调用。

### Kernel 计算与访存

按输出块分配计算，每块沿 K 归约。常规矩阵先批量打包两个操作数，由 Cube 完成三个实数乘积，再由 Vector 重构复数输出。小矩阵可由一次 Vector 发射完成归约与缩放。对适合片上分块的 NN 矩阵，同一组 Cube 和 Vector 协同打包、计算并按有效行写回，减少中间数据的往返。载入 A、B 时按各自前导维度定位列，在分离实部和虚部时完成转置寻址及共轭的虚部取负。记变换后的分量为 Ar、Ai、Br、Bi。对每个 K 块形成 Ar+Ai 和 Br+Bi，进行以下三项实数矩阵乘并累加 FP32 部分和。

| 乘积 | 计算 |
| --- | --- |
| P | Ar · Br |
| Q | Ai · Bi |
| R | (Ar+Ai) · (Br+Bi) |

最终实部为 P−Q，虚部为 R−P−Q。对结果乘复数 alpha，再按 beta 分支处理原 C，仅写回每列前 m 个元素。beta=0 的分支从写回前移除对旧 C 的载入，避免未初始化值或 NaN 影响输出。分块尾部按实际长度搬运，计算单元需要的对齐填充放在内部工作区。临近抵消及长 K 用例用独立参考检查分量误差，并据此验证累加与中间量的精度。

在 `blas/gemm/arch22/` 接入现有公开声明，复用该架构已有的列主序分量拆分与基础实数 GEMM 能力。在同一流上依序完成分量拆分、三项实数矩阵乘和输出组合。输出组合阶段筛查非有限结果，并在小输出中检查分量相消，在工作量上限内检查长 K 归约误差被 alpha 放大的风险。对命中的输出元素，在同一输出组合 kernel 中用扩展数值范围重新计算三项实数乘积 U=(Ar+Ai)·Br、V=Ar·(Bi−Br)、W=Ai·(Br+Bi)。由 U−W、U+V 恢复实部和虚部，再完成复数缩放和写回。按输出规模、归约长度和筛查结果控制纠正重算的范围。两组分解的舍入顺序不同，分别用长 K、相消、纯实和纯虚输入验证。按全调用的 kernel 耗时评估拆分、求和、乘积、纠正和写回的总成本。新增架构测试放在 `test/gemm/cgemm3m/arch22/`，并补充测试构建入口及产品支持说明。[ops-blas 源码](https://gitcode.com/cann/ops-blas)

## 可维可测分析

| 验证项 | 方法与判据 |
| --- | --- |
| 功能、精度 | 运行官方全部 1,000 条精度用例。按任务书 §3.2(4)，在运行前统一选定主参考。对全部数值输出采用相同的 FP64 参考策略。矩阵乘加将输入和标量升至复数双精度，使用 Netlib BLAS 3.10.0 的 `cblas_zgemm` 沿完整 K 以 FP64 累加，最后逐分量收窄至 complex64，作为全量精度判定依据。只需缩放 C 的用例在同一精度域中计算。对同一批全部数值输出另用 Netlib `cblas_cgemm` 直接计算 FP32 对照，报告分别列出两套参考下的误差与判定，并说明差异。实、虚分量分别使用 `atol=rtol=2⁻¹³`，匹配率至少 0.99，最大绝对误差满足任务书 §3.2 规定的 `1e−2 或 32 ULP` 界限。 |
| 状态与内存访问 | 覆盖九组 N/T/C、空句柄、非法枚举、各前导维度不足、空矩阵、条件性空指针及参与计算的 Inf/NaN 传播。以 NaN 旧 C 检查 beta=0，以空 A/B 检查 alpha=0、k=0，逐列比对 C 的 padding 哨兵；覆盖非对齐尾块、4095³ 和 4096³。 |
| 性能 | A2 开发自验对 910B3 和 910B4 使用相同流程与门限，在原始记录中保存实际 SoC 和 CANN 版本。依据任务书 §2.4 和 §3.3(3)(5)，报告保留附件全部 200 个编号。140 个合法输入各预热 5 次，再计量 10 次成功调用。60 个输入的 lda 或 ldb 小于当前转置方式规定的下界，逐例验证 `ACLBLAS_STATUS_INVALID_VALUE` 和整个 C 缓冲区逐字节保持，并在报告中列为异常样本，排除计时和倍率统计。另以独立编号仅将不足的前导维度提高到合法下界，验证同一种子的 60 个补充输入；现有 GPU 标杆均未匹配修正后的完整参数。按一次 BLAS 调用归集其全部 kernel 的 `msprof op` Task Duration 后取平均；将微秒换为毫秒，与相同转置、尺寸、标量及前导维度的 `gpu_ms` 对齐。逐例要求 NPU 平均耗时 ≤ 1.25 × GPU 耗时，并计算这 140 个合法输入的平均倍率，要求至少达到 0.8。 |
| 端到端交叉验证 | 相同流上用 `aclrtEvent` 计量预置 Device 缓冲上的 10 次下发，记录 GTest 的 `[PERF] avg_us`；分别保留与 kernel 采集对应的用例和日志。 |
| 工程兼容性 | 在 CANN 9.1.0 下按目标 SoC 构建 `--ops=cgemm3m` 并运行对应 GTest；检查公开符号、单算子构建依赖、既有 `aclblasCgemm` 入口和 arch35 目标。 |
