# 需求背景

## 需求来源

10 月社区任务《aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx 任务书》及其三个接口的 CSV 和测试指导。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，设计分析所依据的 `master` 代码提交为 `05b96ea22cea926ecfd43fc7ac691a795522bd61`。本文是待评审的实现与验证方案，不代表功能或性能已通过验收。

## 背景介绍

这三个句柄式接口计算列主序的扩展通用矩阵乘法，调用者通过 handle 绑定 stream。当前代码仓的 `blas/gemm_ex/` 只有 `arch35` 实现：`aclblasGemmEx` 已声明；`aclblasSgemmEx` 使用 15 参数旧接口；`aclblasCgemmEx` 缺少公共声明。`blas/gemm/arch22/` 已有实数 GEMM 的分块、打包和 Matmul 代码，可作为 A2/A3 的工程参考，但不能直接假定其参数校验、特殊值或性能满足本任务。

# 需求分析

## 需求描述

- 为 Atlas A2/A3 增加 `blas/gemm_ex/arch22/` 实现，在统一头文件 `include/cann_ops_blas.h` 暴露三个接口，并由 `test/gemm_ex/` 承载可复现的测试。
- 三个接口分别与任务书中的 cuBLAS 参数顺序对应：`GemmEx` 19 参数、`SgemmEx` 和 `CgemmEx` 各 17 参数。后两者均无 `computeType` 和 `algo`。复数类型使用仓库的 `aclblasComplex` 和 `ACL_COMPLEX64`，不另造公开类型。
- 主路径分别是 FP32 实数和交错存储的 complex64；支持运行时形状、前导维、N/T/C 及 alpha/beta。要求覆盖完整输出矩阵、边界、异常、特殊值与性能。
- 根据任务书，自验证至少覆盖一款 A2/A3 设备；性能验收指定 Atlas 800T A2 (910B3)。开发验证环境使用 CANN 9.1.0，黄金值使用 Netlib BLAS 3.10.0 的 `cblas_sgemm`/`cblas_cgemm`。

## 需求拆解

| 项目 | 方案与交付边界 |
| --- | --- |
| 公共接口 | 保留 `GemmEx` 的 19 参数声明；将 `SgemmEx` 改为任务书规定的 17 参数；新增 17 参数 `CgemmEx`，同步声明、定义、导出、文档、示例和调用方。 |
| A2/A3 实现 | 实数与复数均采用 Ascend C kernel 直调和 handle stream；由 `arch22` 组织专属 host、kernel 和 tiling。 |
| 950 兼容性 | 同步适配 `arch35` 的 `SgemmEx` 实现及其调用方；公共 ABI 改变须由维护者评审，不能仅修改头文件或为 A2/A3 定义平行接口。 |
| 精度和异常 | 任务书 §2.5 的参数检查顺序、返回码和短路语义适用于三个接口；复数实部与虚部分别验证。 |
| 扩展数据类型 | FP16/BF16/其他混合组合列为待评审支持矩阵；按确认的组合实现并补充正向测试，支持集外组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。附件没有这些组合的正向用例，不能据此宣称已覆盖。 |
| 交付 | 设计评审并合入后实施；提交源码、测试、README、自测报告及日志，功能与性能门槛达到后向代码仓提交 PR。 |

# 详细设计

## 算子分析

### 数学公式与布局

`C = alpha * op(A) * op(B) + beta * C`，其中 `op(A)` 为 `m × k`、`op(B)` 为 `k × n`、`C` 为 `m × n`。所有矩阵按列主序存储，元素地址按元素计：

| 操作数 | N | T / C |
| --- | --- | --- |
| A 的 `(i,p)` | `A[i + p*lda]` | `A[p + i*lda]` |
| B 的 `(p,j)` | `B[p + j*ldb]` | `B[j + p*ldb]` |
| C 的 `(i,j)` | `C[i + j*ldc]` | 不适用 |

实数 C 与 T 等价；复数 C 对索引到的元素再取共轭，T 不取共轭。`lda/ldb/ldc` 允许大于最小值，padding 不参与计算且不得被覆盖。所有大小、偏移、对齐及 workspace 运算使用足够宽的无符号整数并先检查溢出。

### 支持数据类型及接口

| 接口 | 主路径 A/B/C | 标量 | 其他参数 | 状态 |
| --- | --- | --- | --- | --- |
| `aclblasGemmEx` | `ACL_FLOAT` / `ACL_FLOAT` / `ACL_FLOAT` | `const void*` 指向 FP32 | `ACLBLAS_COMPUTE_32F`、`ACLBLAS_GEMM_DEFAULT` | 计划实现 |
| `aclblasSgemmEx` | `ACL_FLOAT` / `ACL_FLOAT` / `ACL_FLOAT` | `const float*` | 无 computeType/algo | 计划实现 |
| `aclblasCgemmEx` | `ACL_COMPLEX64` / `ACL_COMPLEX64` / `ACL_COMPLEX64` | `const aclblasComplex*` | 无 computeType/algo | 计划实现 |
| 扩展组合 | FP16/BF16 等任务书允许的组合 | 由最终组合定义 | 按接口实际参数约束 | 最低支持组合待评审，未验证 |

接口原型以任务书 §2.3 为准；数据指针的地址空间和标量指针模式遵循该版本 ops-blas 公共接口约定，方案按现有 GEMM Ex 的 host 标量、device 矩阵路径设计，设备标量支持范围须在接口评审时明确，不凭 cuBLAS 的其他 pointer mode 推断。

### 支持形状与边界

`m,n,k >= 0`；`lda >= max(1, m)`（A 为 N）或 `max(1, k)`（A 为 T/C）；`ldb >= max(1, k)`（B 为 N）或 `max(1, n)`（B 为 T/C）；`ldc >= max(1, m)`。

按任务书指定顺序检查 handle、维度、前导维、标量指针、dtype 支持、转置枚举、矩阵指针，再检查仅属于 `GemmEx` 的 computeType/algo，最后执行 quick return。无效枚举、非法值、未支持组合和空 handle 分别返回任务书指定状态。前导维检查早于转置枚举检查，对多重非法输入的处理必须用附件负向用例固定；不同接口附件对非法 dtype 的期望有差别，测试须逐条核对，不合并为单一推断。

- 全部必要检查完成后，`m=0` 或 `n=0` 返回成功，不启动计算。
- `k=0` 或 `alpha=0` 时不读取 A/B，执行 `C=beta*C`；`beta=1` 保持 C，`beta=0` 对有效 C 写零且不读旧值。`alpha=0` 的缩放按任务书做 EXACT 对比。
- 按任务书的特殊约定，`beta=0` 且 C 为空时返回成功且不访问 C，包含 `k=0` 分支；这项行为以任务书为准，不泛化为对所有 BLAS 接口的结论。
- A/B 在 `k>0` 且 alpha 非零时必须有效；beta 非零时 C 必须有效。所有短路仍服从上述检查顺序。

## 算子实现

### 实现方案

以 `blas/gemm/arch22/` 的 Matmul 和打包机制为参考，先取得当前源码在目标环境的基线结果，再设计能够满足上述接口边界的共享内部计算路径。复用仅在参数验证、workspace、stream 生命周期及结果语义一致时成立；不得让已有 `aclblasSgemm` 的不同指针检查覆盖本任务规则。

#### Host 侧设计

1. 在公共头文件中使用任务书签名；`arch22` 实现校验参数并依据接口、dtype、转置和边界分发。将 `GemmEx` 与 `SgemmEx` 的共同实数计算委托内部实现，避免复制两套 kernel。
2. 获取 handle 绑定的 stream 与有效 workspace，查询当前设备的核数与资源。按形状和前导维选择直接计算或打包，生成只含本次调用参数的 tiling；不硬编码设备号。
3. 检查 workspace 大小、对齐和整数溢出；复用库既有的用户/库 workspace 所有权与增长契约。多个异步调用的临时区须有明确的流序和生命周期，避免并发覆盖、提前释放或跨 handle 共享可变状态；资源不足返回明确错误。
4. 保持 stream 异步执行；不得为了缩放或暂存矩阵而隐式将完整矩阵回传 CPU。更新接口后同步修改 `arch35` 定义及其所有仓内调用，并由评审确定已发布 15 参数 ABI 的迁移处理。

#### Kernel 侧设计

1. 实数路径以 `C^T = op(B)^T * op(A)^T` 映射列主序，针对紧凑输入考虑直接 Matmul，对非紧凑和尾块考虑先打包；将实际前导维传入打包索引，输出只写 `m × n` 的有效区域。Matmul 的转置模板、tiling 设置及实际绑定的转置标志必须一致，尾块单独处理。
2. 复数候选路径先把交错复数分块拆为实部和虚部，并在打包时按 C 转置对虚部取负。对 `A=Ar+iAi`、`B=Br+iBi` 计算四个实数乘积，得到 `P=ArBr-AiBi`、`Q=ArBi+AiBr`；合并阶段对复数 alpha/beta 和旧 C 执行乘加后写回交错结果。四次乘法是待验证的正确性基线，不预先宣称满足性能；复用分块临时区以控制 workspace。避免未经精度验证的三乘法化简。
3. 处理 `k=0`/`alpha=0` 的缩放 kernel 仅访问有效 C，`beta=0` 不加载旧 C；`beta=1` 允许直接返回。复数缩放逐分量实现，并验证任务要求的位精确行为。
4. 依据运行时资源规划分核与分块，给尾部和 padding 明确边界。数据搬运先验证地址、长度和对齐；单次 `DataCopyPad` 的块长度控制在 16 KiB 以内，缓冲按对齐及安全余量规划。先进行小形状验证，再扩大规模。

### 性能优化方案

在正确性基线建立后，对实数的紧凑路径、打包代价、分核与访存进行 profiling；复数优先分析拆分、四次乘法和合并的流量及启动开销，再评估可证明语义和精度的融合与复用。优化前后使用相同输入、设备、软件和计时范围，不以缩减用例或放宽精度获取性能。

### 支持硬件

| 产品 | 设计目标 | 验证要求 |
| --- | --- | --- |
| Atlas A2 | `arch22` | 功能/精度与 910B3 性能验收待执行 |
| Atlas A3 | `arch22` | 功能/精度待执行；A3 结果不替代 910B3 性能结果 |
| Ascend 950 | 公共接口变更的兼容范围 | 需构建并回归受影响的 `arch35` 接口和测试 |

### 算子约束限制

本任务不涉及广播、任意非列主序步长或额外 batch 维度。输入矩阵只读，C 原地更新；按前导维访问的 padding 不属于有效输出。主路径与扩展组合的支持声明必须以实现和相应平台测试为依据，完成验证前不得把产品支持表改为已支持。

# 可维可测分析

## 精度标准与性能标准

| 验证项 | 方法与通过条件 |
| --- | --- |
| 接口与异常 | 三组附件各 1,200 条，其中 1,000 条精度/边界和 200 条性能；按 case ID 检查实际返回码及参数校验顺序，补齐缺失的扩展类型正向与组合冲突用例。 |
| 数值精度 | Netlib `cblas_sgemm`/`cblas_cgemm` 为单 golden；FP32 和复数实部/虚部分别按 `atol=rtol=2^-13`、matched ratio ≥ 0.99 和最大绝对误差 ≤ `1e-2` 或 32 ULP 判定。FP16/BF16 如被支持，分别采用任务书 `2^-10`/`2^-7` 档位。 |
| 特殊值 | 对 NaN/Inf 单独计数并按任务书 §3.2 的等效规则逐元素判定；不能把非有限值混入有限值误差统计。`alpha=0` 缩放单独按 EXACT 校验。 |
| 性能 | 在 910B3 上 warmup 5 次，采集超过 50 次有效样本，按实际 kernel task duration 和完整调用路径统计；逐 case 满足 `NPU时间 <= GPU基线时间 / 0.8`。多 kernel 路径须说明各 kernel 与一次接口调用的对应关系及汇总口径，不挑选单个较快 kernel。 |
| 资源和集成 | 检查对齐、尾块、前导维 padding、workspace 不足、溢出和重复异步调用；完成公开头文件、符号、测试工程及受影响架构回归。任务书未设置单独的内存验收上限。 |

附件性能脚本读取 GTest 整数毫秒总耗时，包含 host 和 golden 开销，只能作为诊断，不能代替 msprof kernel 样本；脚本对失败用例和无参考数据的处理也不能作为验收结论。附件精度脚本在解析到零条用例时仍可能返回零退出码，验收应独立核对进程状态、预期用例数和逐条结果。附件 README 对 GPU 型号分别出现 A100 和 P102-100 的描述；在正式比较前需确认基线来源、设备和计时范围。任务书称至少 10 次有效采样，附件建议超过 50 次，本方案采取后者并保留原始样本及异常结果。

测试报告记录代码完整 SHA、SoC、CANN 版本、用例文件哈希、命令、退出码、逐用例结果和性能样本，区分 PASS、FAIL、SKIP、TIMEOUT 和 PENDING；没有覆盖、没有有效样本或未找到基线都不能记为 PASS。

## 兼容性分析与待评审事项

1. `aclblasSgemmEx` 从 15 参数改为 17 参数属于公开 API/ABI 变更，旧二进制调用约定不能假定兼容。需要维护者确认迁移方案、版本发布和 950 回归范围；本任务不自行引入产品私有的平行接口。
2. `GemmEx` 的扩展混合精度最低支持组合与附件缺失的正向用例需要评审明确；不支持的合法组合和非法 dtype 的返回码按任务书与各用例逐项核对。
3. 任务书 §2.5 的 `C=nullptr && beta=0` 与常见 GEMM 输出要求不同；此处按任务书实现，并请评审确认与公共库契约的关系。
4. 附件精度/性能说明、GPU 基线设备描述和采样次数存在差异；性能结论待基线来源及多 kernel 统计口径确认。目标平台的 CANN 9.1.0、Netlib 3.10.0 和 910B3 资源均须在验收时实际核验。

## 参考资料

- [ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)
- [ops-blas GemmEx 实现及说明](https://gitcode.com/cann/ops-blas/tree/master/blas/gemm_ex)
- [生态算子混合容差精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
- [cuBLAS GEMM 扩展接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx)
- [Netlib SGEMM](https://www.netlib.org/blas/sgemm.f) 和 [Netlib CGEMM](https://www.netlib.org/blas/cgemm.f)
