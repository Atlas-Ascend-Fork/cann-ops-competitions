# aclblasHgemmStridedBatched 算子设计文档（Atlas 950PR）

# 1. 需求背景（required）

## 1.1 需求来源

依据《aclblasHgemmStridedBatched 算子开发任务书》及其附带测试用例，在 ops-blas 中补充半精度固定步长批量 GEMM。对标 cuBLAS `cublasHgemmStridedBatched`，面向 Atlas 950PR 和 CANN 9.1.0，使用 Ascend C kernel 直调，复用同族 GEMM 工程能力。

## 1.2 背景介绍

批量矩阵乘适用于多组大小一致的小矩阵计算；以固定元素步长定位每批，减少逐次接口调用。输出原地更新，调用方以列主序及前导维描述矩阵。公共接口供各产品线共用，950PR 实现放在 `blas/gemm/arch35/`，不新增私有平行 API。

# 2. 需求分析（required）

## 2.1 需求描述

对批次 `b∈[0,batchCount)` 计算 `C_b = alpha * op(A_b) * op(B_b) + beta * C_b`。A、B、C 及 host 标量均为 FLOAT16，矩阵位于设备内存；各批通过非负 `long long` 元素步长定位。`op` 支持 N、T、C，实数下 C 等价 T。结果写回 C 的逻辑区域；不属于任何输出逻辑区域的 padding 保持不变。

## 2.2 需求拆解

1. 公共头文件新增指定签名；handle 绑定 stream 异步直调设备 kernel。
2. 支持转置全组合、运行时维度、前导维 padding、固定 batch 步长，包括 A/B 重叠只读和零步长；不支持其他任意视图或维度广播。
3. 完成参数校验、空规模 no-op、k=0 缩放、beta=0 不读取旧 C；覆盖 Inf/NaN 和非法参数。
4. 设计满足任务书精度、性能及验证要求，不以未验证的优化代替正确性。

## 2.3 接口及参数契约

```cpp
aclblasStatus_t aclblasHgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, long long strideA,
    const aclblasHalf* B, int ldb, long long strideB,
    const aclblasHalf* beta, aclblasHalf* C, int ldc, long long strideC,
    int batchCount);
```

| 参数 | 内存及约束 |
| --- | --- |
| handle | Host 句柄，空时 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| transa/transb | Host 枚举 N/T/C，非法返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| m/n/k, batchCount | Host int，非负；负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha/beta | Host FLOAT16 标量指针，均不可为空，包括 no-op 场景 |
| A/B | Device FLOAT16 输入，非空运算且 k>0 时均不可为空，即使 alpha=0 也须校验 |
| C | Device FLOAT16 原地输出；beta=0 时无需有效旧值。当前实现对 k=0、beta=0、C 为空按任务书参数表返回成功，此例外遵循参数表及随附 TC_ED_145，C 有效时仍执行置零 |
| lda/ldb/ldc | int，分别不小于 `max(1, A物理行数)`、`max(1, B物理行数)`、`max(1,m)` |
| strideA/B/C | long long，相邻批次首地址的元素偏移，均非负 |

参数非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。no-op 之前依次检查 handle、枚举、维度/批数、前导维、步长、标量指针；随后 `m=0`、`n=0` 或 `batchCount=0` 返回成功，不访问 A/B/C。非空运算时再检查所需设备指针。`k=0` 不引用 A/B，执行 `C=beta*C`；beta=0 直接置零而不读旧 C。成功返回表示分派成功，异步执行错误由 stream 同步时报告。

# 3. 详细设计（required）

## 3.1 算子分析

### 数学公式

对 `0≤r<m, 0≤c<n, 0≤b<batchCount`，按以下公式计算并将结果舍入到 FLOAT16：

```text
C[b*strideC+r+c*ldc] = alpha * Σ(p=0..k-1) opA(b,r,p)*opB(b,p,c)
                         + beta * C_old[b*strideC+r+c*ldc]
opA = A[b*strideA + (transa==N ? r+p*lda : p+r*lda)]
opB = B[b*strideB + (transb==N ? p+c*ldb : c+p*ldb)]
```

A 的物理形状在 N 下为 m×k，否则为 k×m；B 在 N 下为 k×n，否则为 n×k。C 为 m×n。计算及地址偏移使用足够宽的整数，乘加及字节转换先检查溢出，调用方保证内存范围有效，异步执行完成前保持设备缓冲有效。

### 支持数据类型和形状

仅支持 FLOAT16，统一 batch 的 m/n/k 及操作标志相同。规格允许运行时非负维度，测试覆盖 0、1、质数、非对齐及至 16384 的尺寸、至 65536 的批数，实际可执行规模受地址空间和设备内存约束。复杂度约为 `2*batchCount*m*n*k` 次浮点运算，忽略缩放与加法；逻辑输出量为 `2*batchCount*m*n` 字节。

## 3.2 算子实现

### 3.2.1 Host 和 Tiling 设计

入口放入 `include/cann_ops_blas.h`，实现接入 `blas/gemm/arch35/` 的同族 GEMM 直调框架；复用 handle 对应 stream、平台信息及仓内的构建注册。读取非空 host 标量后按值传递，避免设备读取 host 指针，不隐式同步 stream。

完成 Hgemm 专属校验和 FP16 host 标量向 FP32 转换后，内部 `HgemmStridedBatchedLaunch` 构造 tiling 并通过 handle stream 直调专用 FLOAT16 kernel。空规模直接返回；k=0 使用共享 epilogue 缩放或清零 C。输入 A/B 为零步长时仍批量执行；只有 C 批次地址区间可能重叠时，按 batch 拆分并在同一 stream 依次提交。公共 Hgemm 不调用公开 Ex 接口。

通过 `C^T = B^T * A^T` 将列主序映射为共享模板的行主序。tile 按形状选择，具体配置与选择条件见 `hgemm_strided_batched_launch.cpp`。输入 FP16、L0C 使用 FP32。alpha=1、beta=0 时直接转 FP16 写 C；其他标量组合分配逻辑输出大小的 FP32 工作区，再由 epilogue 缩放并写回。工作区大小及地址范围均进行溢出检查。

### 3.2.2 Kernel 设计

每个计算任务独占一个 C tile，按 batch 偏移和列主序地址搬入 A/B 块；N/T/C 使用相应的地址映射和同族模板支持的转置搬运。K 维逐块累加，输入 FP16，采用 FP32 累加及缩放，最后一次转换为 FP16 写出；具体指令和布局以 950PR 可用模板为准。尾块采用边界掩码或填零，禁止访问矩阵外区域；只写 C 的有效元素。

beta=0 时不读取旧 C，避免旧 C 中 NaN 污染输出；其他 beta 读取 C 并计算。k=0 使用共享 epilogue 线性分核，仅访问逻辑 C；beta=0 直接写零。共享 Cube 模板位于 `gemm/arch35/gemm_strided_cube.h`，Ex 内核也通过 include 复用。FP16 转置搬运的 kStep 保持实际 fractal 步长，禁止误扩为偶数；Mmad 的 M 维按16对齐，避免 M=1 触发紧凑布局与 ND2NZ 输入不符。L1 按 tile 实际空间分配，L0 容量使用静态断言约束。

通用 Cube 路径使用双 L1 缓冲，常规64/128/256空间tile使用双 L0，192空间tile使用单 L0及较长K块。`MTE1_MTE2` 防止 L1 被提前覆盖，`M_MTE1` 防止 L0 被提前覆盖，`FIX_M` 防止输出完成前复用 L0C。wide tile 只在任务数量足以填满计算核时使用，192 tile 仅在不增加两个空间维度分块数时替代256 tile，不按测试用例编号分派。

针对紧凑 NN、alpha=1、beta=0 的小方阵提供专用路径。8³ 使用 SIMT，每线程按列主序计算一个输出并完整累加8项；16³/32³ 在 batchCount 可被8整除时，将八批输入分别沿 M/N 拼接，执行一次 Cube 乘法，再用 `Fixpipe ndNum` 一次写出八个对角块。交叉批次乘积不会写入结果。对非 NN、非紧凑、其他标量或未满八批的情况回退到通用路径。专用路径同样不读取 beta=0 的旧 C。

输入 batch 可以重叠并被多次只读。当前实现检测到输出地址区间可能重叠时，将各 batch 在同一 stream 上依次提交，按批次顺序更新；每次写回后结果为 FP16，再供后续重叠批次读取。任务书对输出重叠与附件测试指导存在冲突，本实现明确采用顺序更新作为扩展语义；A/B 与 C 的别名不受支持。

### 3.2.3 代码组织

| 位置 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 公共声明，复用既有类型、状态码 |
| `blas/gemm/arch35/` | Hgemm Host、内部 launch、专用 kernel 及共享 Cube 模板 |
| `blas/gemm_strided_batched_ex/arch35/` | 共享 tiling 数据结构、epilogue；Ex include 共享 Cube 模板 |
| `blas/gemm/hgemm_strided_batched.md` | 接口、产品支持表、参数与边界语义 |
| `test/gemm/hgemmStridedBatched/arch35/` | GTest、CSV 与可复现测试说明 |

## 3.3 支持硬件

| 芯片/软件 | 支持范围 |
| --- | --- |
| Atlas 950PR | 支持，arch35 实现 |
| CANN | 9.1.0 |

## 3.4 算子约束限制与待澄清事项

- 仅列主序 FLOAT16，host 标量、统一维度和固定元素步长；不提供任意张量视图、不同批次尺寸或确定性保证。
- 任务书 2.1 要求非空计算的 C 非空；2.4/2.5/3.5 又允许 `k=0,beta=0` 时 C 为空，但同时要求置零。空地址无法写入零；当前实现对该空指针例外返回成功，对非空 C 执行置零，与随附 TC_ED_145 的期望一致。
- 任务书允许 `strideC=0` 或小于存储宽度并要求覆盖，附件测试指导将 C 重叠写视为未定义且不构造；本实现对重叠输出定义逐批顺序更新。stride=0 的 A/B 是地址复用，不涉及维度广播。
- 性能正文规定 5 次预热后 10 次有效样本、单 case ≥0.4× 且平均 ≥0.5× GPU；第 7 节又要求 >10 次且 ≥GPU。本次按§3.3明确的采样与判据执行，保留原始样本；不宣称达到§7的另一口径。

# 4. 可维可测分析

## 4.1 精度标准

参考逐 batch Netlib BLAS 3.10.0 `cblas_sgemm`，将 FP16 输入和标量转为 FP32 运算，再按 IEEE 754 最近偶数舍入为 FP16。逐 batch 全量检查 C 逻辑元素及 padding。有限值逐元素满足 `abs(actual-golden)≤2^-10+2^-10*abs(golden)` 的比例至少 0.99，最大绝对误差需满足任务书 `1e-1 或 32*ULP` 限制，两项同时成立。实现要求每个有限元素误差不超过 max(0.1,32*ULP(golden))，并记录逐 batch 的最大误差。

非有限值单独统计：双方均 NaN、同号 Inf、或一方 NaN 另一方任意 Inf 视为等效；有限与非有限不等效，非有限元素全部满足才通过。对大 k 溢出或精度偏差保留独立参考和误差随 k 分析，不通过放宽阈值代替证据。

## 4.2 测试方案

使用任务包全部1200条 CSV 和 ops-blas GTest，分别执行均匀/正态两轮，保留特殊填充和固定标量；另以36条配对分布用例覆盖随机 alpha/beta。覆盖全部 N/T/C 组合、不同 m/n/k 和 batch、紧凑与 padding、零步长和重叠读取、0/±1/大值/Inf/NaN。额外测试三个16384维度、65536批、beta=-1及三种no-op的先行校验；尾块回归覆盖486种形状/转置组合。固定种子及正态mu/sigma写入日志。

负向覆盖空 handle、非法枚举、负尺寸/批数/stride、前导维不足、空标量与必要设备指针；验证负 stride 即使 batchCount=0 仍报错。空 m/n/batch 不读设备指针；k=0 不读 A/B，beta=0 且 C 含 NaN 应写零。对 C padding 加哨兵检查，验证 A/B 只读；在可用的设备内存检查工具下核查越界与竞态。另验证非默认 stream 的调用及缓冲生命周期。重叠 C 按顺序语义验证，k=0、beta=0、C 为空按 TC_ED_145 验证。

## 4.3 性能标准和测量

使用 `gpu_baseline.csv` 对齐各性能用例，在 950PR 使用 msprof op 采集 task duration，完成 5 次预热后取 10 次成功执行的算术平均，单位 us；记录异常样本与测量环境。逐 case `GPU耗时/NPU耗时≥0.4`，全部 case 倍率算术平均≥0.5，且精度合格；第7节的另一口径单独说明，不混称通过。任务的200条性能用例均 alpha=1、beta=0，直接调用单个内核；其他含 epilogue 的调用不能用矩阵乘部分时间代表完整调用。每轮恢复原始 C，恢复及 golden 计算不计入调用时间。

## 4.4 兼容性和可维护性

公共 API 使用现有类型、状态码和 handle 生命周期，不改变既有 API 或 stream。校验、地址映射、tiling、计算和测试参考分离，保留通用路径以覆盖对齐和尾块；优化必须通过相同功能和精度测试。记录编译配置、设备、版本、用例参数和随机种子，测试报告单独保存，不纳入设计文档。
