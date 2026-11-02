# aclblasGemmEx / aclblasSgemmEx / aclblasCgemmEx（A2/A3）设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 10 月社区任务：aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx（A2/A3） |
| 贡献者 | Y673789476 |
| 目标仓库 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) |
| 开发环境 | CANN 9.1.0；自验使用 Ascend 910B3 |
| 文档状态 | 设计评审稿；下述精度、性能均为验收目标，不代表已经测得或达标 |
| 代码分析基线 | ops-blas `1e270a3`，最终实现及自测报告记录完整 commit |

# 需求背景（required）

## 需求来源

依据《aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx 任务书》及配套 CSV，为 Atlas A2/A3 补齐三个扩展矩阵乘法接口。接口参数和返回值以任务书为主，通用 BLAS 行为参考 cuBLAS 和 Netlib。

## 背景介绍

三个接口计算列主序矩阵的 `C = alpha * op(A) * op(B) + beta * C`。GemmEx、SgemmEx 主路径为 FLOAT32，CgemmEx 主路径为 COMPLEX64。实数接口的 OP_C 等价于转置；复数接口的 OP_C 必须额外对虚部取负。

最新仓库的 `blas/gemm/arch22/` 已有 FP32 GEMM 实现，可复用其布局适配、Cube 分块和后处理。`blas/gemm_ex/` 目前为 arch35 实现，A2/A3 扩展接口仍需补齐。仓内已有 CgemmEx 声明使用 `int typeA/typeB/typeC`，与任务书要求的 `aclDataType` 不同；SgemmEx 仍是旧签名。不能简单复制任务书中“仓内无 CgemmEx”的旧状态描述。

# 需求分析（required）

## 需求描述

使用 Ascend C kernel 直调方式实现三个接口，复用 handle 绑定的 stream，以异步方式下发。代码放入 `blas/gemm_ex/arch22/`，测试放入 `test/gemm_ex/`，公共声明位于 `include/cann_ops_blas.h`，与 950 产品线共用。

| 接口 | 参数个数 | 主路径 dtype | 差异参数 |
| --- | ---: | --- | --- |
| aclblasGemmEx | 19 | A/B/C 均为 ACL_FLOAT，FP32 累加 | computeType、algo |
| aclblasSgemmEx | 17 | A/B/C 均为 ACL_FLOAT，FP32 累加 | 无 computeType、algo |
| aclblasCgemmEx | 17 | A/B/C 均为 ACL_COMPLEX64，实虚交错 | alpha、beta 为 aclblasComplex |

主路径优先覆盖附件规定的全部用例。FP16/BF16 等组合不因名称带 Ex 就默认支持；实现中明确支持集，超出支持集返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。若验收要求加入扩展组合，则同步补充 dtype 分发、量化 golden 和相应阈值测试，不能仅修改文档为“支持”。

## 需求拆解

1. 修正 SgemmEx 为任务书的 17 参签名，统一 CgemmEx 的 dtype 类型，并同步 arch35 定义和调用点。
2. 共享三个接口的参数校验顺序、状态码和 quick return 规则。
3. 实数常规路径复用现有 arch22 GEMM，并增加直接 Matmul/对齐打包混合后端；极小 NN 使用 AIV，标量放大及小输出长 K 场景使用顺序累加 AIV。
4. 复数按场景选择极小 NN AIV、顺序累加 AIV、实数嵌入单 GEMM 或拆分四 GEMM，支持 N/T/C 全组合。
5. 验证配套 3,000 条精度用例、72 条必要标量分布补充用例和 600 条性能用例，提交可复现的精度、性能报告及日志；全量结论以最终实测报告为准。

# 详细设计（required）

## 算子分析

### 数学公式

设 `X=op(A)=Ar+iAi`，`Y=op(B)=Br+iBi`，则：

```text
P0 = Ar * Br
P1 = Ai * Bi
P2 = Ar * Bi
P3 = Ai * Br
R = P0 - P1
I = P2 + P3
C.real = alpha.real*R - alpha.imag*I + beta.real*C.real_old - beta.imag*C.imag_old
C.imag = alpha.real*I + alpha.imag*R + beta.real*C.imag_old + beta.imag*C.real_old
```

以上为通用四乘法路径，避免三乘法额外引入的和差消去误差。对转置使用实数 GEMM 的转置分发，对共轭在拆分时取虚部负号。beta 为零时不加载 C 旧值。符合条件的 alpha=1、beta=0 场景另采用实数嵌入单 GEMM，数值敏感及极小场景使用 AIV，具体分发见下文。

### 支持形状与布局

`m,n,k >= 0`，op(A) 为 m×k，op(B) 为 k×n，C 为 m×n。以列主序和前导维描述物理布局，允许前导维 padding；不支持额外任意 stride 或广播。

- A：N 时 `lda >= max(1,m)`，T/C 时 `lda >= max(1,k)`。
- B：N 时 `ldb >= max(1,k)`，T/C 时 `ldb >= max(1,n)`。
- C：`ldc >= max(1,m)`。
- 地址计算使用足够宽的整数，workspace 分区和分配尺寸独立于逻辑元素数量，避免 padding 与有效元素混淆。

## 算子实现

### 3.2.1 Host 侧设计

#### 公共参数校验

严格采用任务书顺序：handle → 维度非负 → 前导维 → alpha/beta 非空 → dtype 支持集 → trans 枚举 → A/B/C 指针 → GemmEx 的 computeType/algo → no-op。

| 情况 | 处理 |
| --- | --- |
| handle 为空 | HANDLE_IS_NULLPTR |
| 负维度、非法前导维、必需的指针为空 | INVALID_VALUE |
| trans 枚举非法 | INVALID_ENUM |
| dtype/计算类型/算法不支持 | NOT_SUPPORTED |
| m=0 或 n=0 | 参数校验完成后返回 SUCCESS，不发 kernel |
| k=0 或 alpha=0 | 不引用 A/B，仅执行 beta 缩放 |
| alpha=0、beta=1 | C 保持不变，避免多余读写 |
| beta=0、C=nullptr | 按任务书特殊约定直接 SUCCESS，不读写 C，包括 k=0 |
| beta=0、C 有效 | 不读取 C 旧值；仅缩放分支直接写零 |

alpha、beta 按任务书约定为 Host 标量；矩阵为 Device 内存。不会为了检查矩阵数值而在生产接口中拷回 CPU。

#### 实数路径与 tiling

GemmEx、SgemmEx 共享同一个 Ex 分发入口。参数检查及短路处理完成后，先为极小 NN 选择单 AIV kernel，避免 Cube 布局转换和多次启动开销；再为 alpha 放大或小输出长 K 选择保持 Netlib 运算次序的 AIV 路径。其他场景进入共享实数后端，按转置、前导维、对齐和规模复用 `blas/gemm/arch22/` 或调用直接 Matmul/对齐打包混合路径。分发仅依赖公开输入参数，不依赖测试名称或 golden 结果。当前极小路径的实数边长上限为 16，其他阈值随正确性和实测调整；阈值属于实现策略，不构成 API 支持边界。对现有 Sgemm 的复用须显式配置单算子构建依赖，确保 `--ops=gemm_ex` 不缺链接符号。

保留 FP32 精度，不用未经精度验证的 HF32/低精度替换来满足性能指标。特殊缩放路径由向量 kernel 处理，避免 k=0 或 alpha=0 时触发无意义的 GEMM。

CgemmEx 在非缩放分支中依次考虑极小 NN AIV、任一 alpha 分量放大或小输出长 K 的顺序累加 AIV、alpha=1 且 beta=0 的实数嵌入单 GEMM，最后使用通用四 GEMM。当前极小复数边长上限为 32，嵌入路径覆盖至各维 2048；实际阈值允许依据实测继续调整。AIV 路径不构造四乘积平面；嵌入路径与四乘法使用各自独立的 workspace 布局。

#### 复数 workspace 与异步生命周期

通用四乘法的 handle workspace 按 32 字节或更高要求对齐，分为 A/B 实虚平面、四个乘积平面、实数 GEMM 临时区域。拆分后的 A/B 保持其物理行列方向及转置标志，乘积使用紧凑 m×n 布局。嵌入单 GEMM 路径则保存 4mk 个 FP32 A 元素、2kn 个 FP32 B 元素及实数后端 scratch，输出直接使用交错 C，不分配四个乘积平面。

紧凑布局的主要存储上界为 `4 * (2*m*k + 2*k*n + 4*m*n)` 字节，再加对齐和实数 GEMM scratch；具体大小以 tiling 实际布局为准。Host 先一次性确保整个 workspace，再将不重叠的 scratch 子区域交给实数后端。不得让嵌套 GEMM 的 workspace 扩容覆盖拆分数据或尚未完成的乘积。

所有拆分、矩阵乘、合成 kernel 在同一个 handle stream 顺序下发。生产接口不在每次调用结束强制同步，不释放仍被 stream 引用的存储；使用仓库既有 workspace 生命周期管理。测试在读回或复用结果前同步。

### 3.2.2 Kernel 侧设计

#### 实数 GEMM

复用 arch22 的 AIC Cube 计算路径及 AIV 布局/alpha-beta 后处理。新增后端在满足分发条件时直接以实际 leading dimension 配置 Matmul，避免紧凑假设；其他形状通过 AIV 多行带步长搬运，将 A/B 放入对齐的物理布局并将补齐区置零，再执行 Cube Matmul。输入、输出均按运行时 m/n/k/ld 计算地址；alpha=1、beta=0 时可直接输出 C，其余使用临时乘积及带 ldc 的裁剪/缩放后处理。任意合法 leading dimension 的支持并不意味着所有形状均不打包，而是由直接与打包两种路径共同覆盖。

#### 极小 NN AIV

实数、复数极小 NN 问题按输出列/片段划分核心，在一个 AIV kernel 内完成矩阵乘和 alpha/beta 运算，减少多 kernel 启动开销。块数受实际任务量限制；仍处理 padding、beta=0 不读旧 C 等语义。极小分发优先于通用顺序/矩阵后端。

#### 数值敏感场景的顺序累加

标量放大或小输出长 K 场景中，归约重排的舍入差异可能被放大。顺序 AIV 路径保持 Netlib 对应转置分支的循环运算次序及 alpha/beta 应用位置，沿输出元素进行向量并行，沿 K 按顺序累加，不以树形归约替代逐步求和。复数路径使用独立实虚乘加和 N/T/C 地址映射。该路径优先保证规定精度，可能增加大矩阵耗时；其性能必须使用真实数据验证，不能沿用四乘法路径的性能结论。

#### 复数拆分

AIV 将交错 `[real,imag]` 批量搬入 UB，使用向量 gather 或等价向量操作拆成两个 FP32 平面，必要时对虚部取负。按连续物理行片段划分核心，输出互不重叠。padding 和尾片段使用带长度的搬运，不越过分配边界。

#### 复数实数嵌入单 GEMM

对于 alpha=1、beta=0 且落在已选规模范围的场景，利用 `[Ar,-Ai; Ai,Ar] * [Br;Bi]` 将复数乘法嵌入一个 `(2m)×n×(2k)` 的 FP32 GEMM。A 的输出行按实/虚交错组织，使实数 GEMM 可用 `2*ldc` 直接写回复数 C。打包阶段应用 N/T/C 地址映射与共轭符号，并保留匹配实数后端的物理转置布局。该路径用额外 A/B 打包空间换取减少 GEMM 启动和通用合成 kernel；它与四乘法的舍入路径不同，须各自接受同一精度标准和实际性能验证。

#### 复数合成

AIV 分块加载 P0/P1/P2/P3，用 Sub/Add 得到 R/I，再执行复数 alpha 缩放；仅 beta 非零时加载 C 旧值，执行复数 beta 缩放并累加。最终按 C 的列主序和 ldc 写回交错输出，不能覆写 padding。缩放专用分支不读取未生成的乘积平面。


#### 按布局与规模选择的性能路径

所有分支只依赖接口参数，不读取用例编号、golden 或历史测量结果。计算保持 FP32，Cube 路径关闭 HF32。

| 场景 | 实现策略 |
| --- | --- |
| 对齐常规矩阵与大矩阵 | 复用 arch22 GEMM 或静态 Matmul 模板；保留一般 alpha/beta 语义 |
| 短 K 实数 NN | 按输出面积分块，单轮任务使用简化的纯 AIC 路径；对齐临时结果后由 AIV 按实际 ldc 搬回 |
| 短 K 实数双转置 | AIV 按 A/B 各自布局打包 NZ，纯 AIC 计算，再裁剪输出；较长短 K 可直接从原布局加载 |
| 需转置的窄矩阵或长 K | 在 AIV 侧预先形成 NZ；输入双缓冲预取，避免重复 ND2NZ 转换 |
| 复数短 K NN | 将 A 的实虚对与交换符号后的实虚对打包，并打包 B；使用一次实数 GEMM，最后搬回有效输出 |
| 复数窄输出、长 K NN | 采用两段 K 划分与确定次序的 AIV 求和；输入 A 按 512 字节行跨度对齐，部分结果保持紧凑布局 |
| 复数转置嵌入 | 在一次 AIV 打包中完成共轭符号处理及 NZ 转换；短 K 使用纯 AIC 写入对齐临时结果，再由 AIV 按实际 ldc 搬回；窄 B 使用独立打包块，改善任务分布 |
| 标量放大、小输出长 K及任务书特殊值场景 | 使用既有顺序累加/缩放分支满足规定精度；不以降低精度换性能 |

NN 中短 K（97～160）且 M≤2048 时，输入直接从 GM 搬运并转换至 L1，Cube 输出写入对齐暂存区后由独立向量核搬回 C，避免原混合核的全局输入打包和跨核等待。L1 容量覆盖该范围，保留多轮块间及 K 片段间的数据依赖。

未预打包且 K≤32 的实数 NN 专用路径以 64 列为一块，行片段根据形状均衡到不超过 512 行；单轮内直接计算当前核心的一块。NN 输出搬运仅在 UB 容量允许且保留至少 75% 原核心数时调整分核，使每个核心负责的列数为 8 的整数倍，保证各核心相对 C 的写入偏移为 32 字节的整数倍。该分核调整仅用于 NN 分支，转置搬运保持原策略。

复数转置窄 B 分支的 A 打包索引利用 NZ 布局周期：只生成 512 个索引种子，再按每组 64 字节偏移扩展到整块，减少每个 AIV 核的索引初始化指令和向量遍历量。索引运算保持整数精确，缓冲区复用仍保留必要的流水依赖。

复数短 K 输入打包按总任务数选择单轮编译分支。单轮任务省去后续循环及末尾的缓冲区复用等待；多轮任务仅在确实有后续任务时执行 MTE3→V、MTE3→MTE2 反向依赖。输入读取、向量变换、输出写入之间的生产者与消费者依赖保持完整。

单次即可完成的输出搬运使用独立编译分支，循环间反向同步仅在缓冲区还需复用时执行。同步依据官方 [静态 Tensor 编程说明](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html)。

短 K 手写 Cube 内核对匹配的 Mmad/Fixpipe 数据块启用 UnitFlag。跨 tile 复用 L1、L0A/B、L0C 时分别维护 MTE1→MTE2、M→MTE1、Fixpipe→M 依赖；单轮任务直接计算当前核心的一块，省去循环和跨 tile 控制。混合 AIC/AIV 路径通过事件保证暂存结果写完再读取、读取完再复用。所有 padding 均计入 workspace 分配，实际输出只写有效行列。

实现参考官方 [UnitFlag 机制](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute_key_features/UnitFlag.html)、[AIV ND2NZ 优化](https://asc.gitcode.com/guide/operator_practice/best_practices/matmul_tuning_cases/aiv_nd2nz_conversion.html)、[Split-K](https://asc.gitcode.com/guide/operator_practice/best_practices/matmul_tuning_cases/matmul_multi_core_split_k.html) 和 [CATLASS 对齐建议](https://gitee.com/ascend/catlass/blob/master/docs/catlass_optimize_guidance.md)，是否采用由任务书用例的实测结果决定。

#### 分核、流水及优化次序

向量搬运以对齐片段为单位，按实际任务量限制 block 数，不为极小输入盲目启动全部核心。CopyIn/Compute/CopyOut 通过队列和流水事件确保数据依赖；仅在能减少等待且不超出 UB 预算时使用双缓冲。

复用的 NN 打包路径中，MTE2 搬运与尾部标量补零可能写入同一 cache line，必须在标量补零前完成相应搬运同步，不能只依据逻辑索引不重叠就假定安全。实现已补充相应同步；对齐打包路径还须保证 UB 清零、MTE2、MTE3 和下一轮缓冲区复用之间的依赖。局部调试结果不替代完整精度验收。

优化次序为：典型正确性 → 真实 kernel 计时 → 减少布局搬运/启动 → 调整 tile 和分核。禁止 CPU 代算主结果、漏计拆分/合成 kernel、跳过失败性能用例或修改基线数值来得到达标结论。

## 支持硬件

| 目标产品 | 设计范围 | 自验计划 |
| --- | --- | --- |
| Atlas A2 | arch22 | 910B3，精度与性能完整验证 |
| Atlas A3 | arch22 | 同架构设计覆盖；未上机前不声称完成 A3 实测 |
| Ascend 950 | 共用公共声明 | 同步签名与调用点，保留既有实现；不将 A2 数据用作 950 验证结果 |

## 算子约束限制

无广播、无任意非连续 stride，无新指针模式扩展。C 原地覆写；A/B 与 C 的额外重叠不作为新增保证。FP32/COMPLEX64 之外的支持集必须以实际实现和自测为准。超出 handle workspace 可用容量时返回仓库规定的分配错误，不继续发出使用无效存储的 kernel。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 口径 | 来源 |
| --- | --- | --- |
| Golden | Netlib BLAS 3.10.0：cblas_sgemm / cblas_cgemm | 任务书 |
| FP32、COMPLEX64 | atol=rtol=2^-13；匹配比例≥0.99；最大绝对误差 ≤ max(1e-2, 32×ULP(g_low))，g_low 取最大误差点的 FLOAT32 golden 值 | 任务书 §3.2 |
| 复数比对 | 实部、虚部分别全矩阵统计，不仅比较模长 | 任务书 |
| 特殊缩放 | alpha=0 的 beta 缩放使用 EXACT 判定；beta=1 直接保留输入 | 任务书 |
| 非有限值 | 同 NaN、同号 Inf、NaN 与 ±Inf 视为等效；有限/非有限不等效；非有限元素单列且须全部通过 | 任务书 |
| 性能 | 每个 case 的 NPU 平均时间≤对应 GPU baseline×1.25 | 任务书 §3.3 |

Golden 使用官方 Reference-LAPACK/Netlib BLAS 3.10.0 源码，不修改其计算源码；当前可复现构建采用 GCC/GFortran 9，C 编译选项为 `-O3 -DNDEBUG -ffp-contract=off`，Fortran 为 `-O2 -DNDEBUG -ffp-contract=off -frecursive -fPIC`。关闭浮点乘加收缩用于固定逐步乘加的舍入行为。默认 O3 下编译器可能使用 FMA，舍入结果与该构建不同，开发阶段出现过由此相关的精度差异，因此不能声称精度结果与编译选项无关。验收包须记录源码版本、编译器版本、完整构建命令及所链接 golden 库，最终精度结论仅对应报告记录的环境。

附件三个 CSV 各 1,200 条，按 3,000 条原始精度和 600 条性能执行，不能按任务书旧的“200 条总计”缩减。为覆盖任务书标量分布，每个接口补充 24 条用例（均匀、正态各 12 条），精度/接口校验共 3,072 条；这些是待全量验证的用例规模，不是通过数量。只补接口适配、模板遗漏场景及验收必要覆盖，不开展无关防御测试。配套 verify_accuracy.py 明确排除 TC_PF；精度与性能分别按各自用例集合验收。最大误差上限按任务书引用的 opbase 混合容差标准 §2.1.2 以最大误差点为锚，日志保留该点值及动态上限。

CSV loader 支持三个接口的字段，含实/复 alpha、beta、null/NULLPTR、dtype、computeType/algo。负向用例只校验返回值，避免用非法尺寸分配矩阵。测试期望与任务书不一致的地方，例如非法 dtype，按任务书统一为 NOT_SUPPORTED，并在报告记录调整原因及原始值。

性能用例先执行一次初始化调用，再执行 5 次应用预热及 11 次有效调用，满足任务书“>10 次”的要求。beta=0 时无须在相邻调用间重复复制无效旧 C；beta 非零时恢复相同初始 C。正式数据来自 `msprof op`/`msopprof BasicInfo` 的 `Task Duration(us)`，统一使用 application replay；工具自身的 warm-up 参数也设为 5。对一次接口调用内所有打包、计算和搬回 kernel 求和，再对 11 次有效样本取算术平均。

pct51 容器中原工具将逻辑设备 0 转为宿主物理设备 4 后发生 HDC 错误。使用独立工具副本及官方 msopcom 源码构建的 injection 库，通过专用环境变量保留容器设备编号。该兼容修改不改变计时数据，未覆盖已安装的 CANN、驱动或工具。交付记录原源码、修改补丁、构建方法及工具哈希，不称其为未经修改的原版工具。

采集可在一次应用中顺序运行多条用例，最多每批 40 条。以 GTest 用例边界和 profiler 的 kernel 启动日志，关联每个 kernel 的全局采集序号与原始 CSV；要求记录一一对应。每条用例的每个 kernel 序列必须恰好组成 17 次完整接口调用，只取第 7～17 次。全部批次使用同一算子库、测试程序、用例及基线哈希，保留批次清单、原始日志和逐启动映射。任何缺失、重复、调用数不完整或构建变化均终止验收，不筛选样本、不替换失败项。

标准 `msprof` 的应用采集只用于优化诊断，不作为最终验收数据。GTest 总耗时及 `PERF_EVENT` 事件时间也不作为报告中的 NPU kernel 时间。最终结论必须来自完整 600 项正式采集及全量精度验证。


附件基线说明存在 A100 与 P102-100 两种相互冲突的来源描述。本次按配套 `gpu_baseline.csv` 原始数值开展对比，报告标为“附件基线（README 性能节描述为 P102-100；来源描述待澄清）”，不宣称 A100 实测或已获官方确认。基线来源澄清不影响先完成实现和数据采集；若维护方指定其他基线，则使用同一真实 NPU 数据重算并重新检查达标，不自行放宽 1.25 倍阈值。

## 兼容性分析

公共头文件按照任务书使用 19/17/17 参，不增设 A2 专用平行 API。SgemmEx 删除旧 algo 并新增三个 dtype 参数，必须同时更新 arch35 定义及仓内调用；CgemmEx 统一为 aclDataType，既有 arch35 私有类型映射在实现内部处理。此次签名变更需要在 README 明确列出迁移方式，不保证旧签名二进制兼容。

代码与测试遵循仓库结构，尽量共享校验和计算逻辑。设计 PR 与代码 PR 分开。交付顺序为先提交验收 ZIP，包含实现源码及 commit、自验命令、精度明细、600 条性能结果、原始日志及环境版本；代码验收审核通过后再提交代码 PR。已有设计 PR 继续用于设计评审，不代表代码验收通过。评审通过、代码合入和性能达标各自记录真实状态。

### 验证状态记录原则

正式验收数据使用上述独立工具副本采集。原工具的失败记录保留在兼容性说明中，不混入正式结果。通过数量、最终构建哈希和代码 commit 以验收包内同一次完整自测记录为准；设计评审及代码合入状态分别记录。

## 参考资料

- [ops-blas 仓库](https://gitcode.com/cann/ops-blas)
- [社区任务目录规范](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
- [设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)
- [生态混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
- [Netlib BLAS](https://www.netlib.org/blas/)
