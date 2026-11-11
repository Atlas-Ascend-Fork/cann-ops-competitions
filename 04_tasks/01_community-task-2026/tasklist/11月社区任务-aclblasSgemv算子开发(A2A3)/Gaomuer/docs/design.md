# 需求背景（required）

## 需求来源

本设计对应《11月社区任务 aclblasSgemv 算子开发任务书（Atlas A2/A3）》，为 ops-blas 的 arch22 补齐单精度实数矩阵向量乘加能力。开发语言为 Ascend C，采用 kernel 直调模式，目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)。本文为设计方案，不代表已完成实现或真机验收。

## 背景介绍

### aclblasSgemv 功能与实现现状

算子执行列主序 BLAS-2 运算 `y = alpha * op(A) * x + beta * y`。仓内已有公共接口声明、arch35 实现和同族测试设施；本任务复用公共 ABI、handle/stream 和测试框架，在 arch22 实现 A2/A3 路径，不复制 950 专属 SIMT 指令。

参考位置为 `include/cann_ops_blas.h`、`blas/gemv/arch35/sgemv_host.cpp`、`test/gemv/sgemv/sgemv_golden.h`。开发时以目标分支实际源码为准，不依赖任务书行号。测试 CMake 若已统一调用 `ops_blas_add_gtest_tests`，仅确认 arch22 文件收集和测试注册，不再重复注册。

# 需求分析（required）

## 需求描述

在 CANN 9.1.0 上实现 FLOAT32 的 N/T/C 三种模式，支持合法列主维、正负向量步长、运行时尺寸与异步原地更新。适配 Atlas A2/A3，自验证在 910B3 完成。全部精度与性能指标以任务书和随附原始测试材料为依据，不沿用其他算子的阈值、golden 或计时方法。

## 需求拆解

1. 保持公共接口签名、Host 标量指针、Device 数据指针及返回码一致。
2. 参数校验先于快速返回，补齐 int32 步长跨度溢出守卫。
3. 分别针对列主序 N 与 T/C 设计连续搬运和分块累加，提供通用非单位步长路径。
4. 保证 beta=0 不读取旧 y；不引入 alpha=0 短路。
5. 覆盖官方 1000 条精度、全部 200 条性能，补充正态分布、溢出、保护区与非有限值测试。
6. 使用 910B3 的 msprof op task duration 验证逐条 NPU 平均耗时不超过 GPU 的 1.25 倍。

# 详细设计（required）

## 算子分析

### 数学公式

以零基索引记 `A(i,j)=A[i+j*lda]`。N 模式输出长度 L=m、归约长度 K=n；T/C 模式 L=n、K=m，实数下 C 与 T 等价。

`y_out[r] = alpha * sum(k=0..K-1, op(A)[r,k] * x_logical[k]) + beta * y_old[r]`。

当 beta 为正零或负零时省去旧 y 的读取和乘法，不能以 `0*y_old` 代替，否则旧 y 中 NaN/Inf 会污染结果。alpha=0 仍执行乘加路径。

### 接口与数据类型

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, const float* x,
    int incx, const float* beta, float* y, int incy);
```

| 参数 | 存储与类型 | 合法域及用途 |
| --- | --- | --- |
| handle | Host 句柄 | 非空，携带执行流 |
| trans | Host 枚举 | N/T/C，C 按 T 执行 |
| m、n | Host int32 | 非负矩阵行列数 |
| alpha、beta | Host FLOAT32 指针 | 不可空，允许任意标量值，不做数值域检查 |
| A | Device FLOAT32 | 只读，列主序，物理容量 lda*n |
| lda | Host int32 | 至少 max(1,m)，列间可有 padding |
| x | Device FLOAT32 | 只读，逻辑长度 N 模式为 n，否则 m |
| y | Device FLOAT32 | 原地输出，逻辑长度 N 模式为 m，否则 n |
| incx、incy | Host int32 | 非零且非 INT32_MIN，支持负步长 |

### 支持形状与地址映射

不限制为官方 CSV 列举的形状；在合法地址范围和设备内存容量内支持运行时非负 int32 尺寸。矩阵元素地址、分块索引及字节数使用 64 位运算，不先在 int32 中乘法再转型。

对于逻辑长度 len>0 的向量，物理跨度为 `span=(len-1)*abs(int64(inc))`。span 超过 INT32_MAX 返回 INVALID_VALUE。负步长的起点为 span，逻辑元素 i 的地址为 `start+i*int64(inc)`；正步长起点为 0。len=0 时不计算 len-1，不进行 abs(INT32_MIN)。矩阵 padding 和向量间隙均不参与计算，且不得被修改。

## 算子实现

### 3.2.1 Host 侧设计

按顺序执行：

1. 检查 handle；空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 检查 trans、m/n、lda、incx/incy、alpha/beta 指针；非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. 根据 trans 计算 x/y 逻辑长度，校验跨度和指针；A 在 m>0 且 n>0 时不可空，x/y 在各自逻辑长度>0 时不可空。
4. 参数校验通过后，若 m=0 或 n=0，返回 SUCCESS，不读写设备数据，不下发 kernel。
5. 非空输入读取 Host alpha/beta 的值，查询平台 AIV 核数、UB 容量，生成 tiling，绑定 handle stream 异步下发。

资源查询失败遵循仓内执行失败口径，不硬编码设备核心数。异步成功表示下发成功，调用方读取 y 前须同步执行流。

#### 分核与 tiling

基础路径按输出块分核，每个输出仅由一个核写回，避免跨核写冲突和浮点原子累加。`blockDim=min(可用AIV核数, 输出块数)`，小形状限制活跃核数以减少下发及调度成本。

tiling 包含 m/n/lda、L/K、64 位 incx/incy 与起点、alpha/beta、输出块大小 R、归约块大小 Q、尾块长度、分核任务数及路径标识。路径区分 N、T/C、连续/跨步和可选 split-K，不以具体 case 编号、随机种子或输入数据值分派。

UB 预算按 `矩阵tile + x tile + 累加向量 + y tile + 归约临时区 + 对齐余量` 计算。启用双缓冲时只对搬入缓冲计入两份，确保总量小于平台实际 UB 容量。初始 R/Q 选为 DMA 和向量指令友好的尺寸，再用 msprof 实测调整，不将设计候选尺寸当作已测最优值。

### 3.2.2 Kernel 侧设计

#### N 模式：连续行块更新

输出块对应连续的 R 行。每次搬入 A 的 Q 列各自的 R 个有效行元素，加载对应 x 元素，按列完成向量乘加至 FP32 累加器。lda 只作为列间距，不能将列尾 padding 当成有效矩阵。x 在核内复用；同一输出块的全部归约完成后统一乘 alpha，按 beta 分支合并旧 y。

#### T/C 模式：连续列内点积

每个输出对应 A 的一列，沿 m 方向分块连续搬运；在 UB 内与 x tile 相乘并归约。一次处理多个输出列以复用 x，使用分组累加后合并，减轻长累加链误差。C 不新增复数共轭逻辑。

#### 步长、尾块与数据搬运

单位步长采用连续 DMA；非单位步长将合法有效元素打包为 UB 连续向量，负步长按逻辑映射读取。采用目标 CANN 9.1.0 的 arch22 可用 DataCopy/DataCopyPad、向量乘加和归约接口，具体重载编译确认，禁止直接复制 arch35 SIMT API。

非对齐尾块仅访问有效 GM 区间，通过受支持的 padding 搬运与 mask 处理；不能依赖调用者多分配对齐空间。跨步 y 写回只写逻辑元素，不能用包含间隙的整块覆盖。TPipe 在 kernel 入口生命周期内创建，搬运/计算/写回之间设置正确事件同步。

#### 小输出、长归约优化

当 L 很小而 K 很大导致核利用不足时，评估 split-K：第一阶段按输出及归约段计算 FP32 部分和到独立 workspace，第二阶段固定顺序归约，并且仅一次应用 alpha、beta。workspace 用量按输出块和分段数有界计算，使用仓库 handle workspace 机制，空间不足回退到无额外 workspace 的单核归约路径。不得在热路径同步 malloc/free。

split-K 是否启用以全部精度和逐条性能验证决定；若采用多 kernel，实现性能统计必须对同一次 API 调用的所有阶段求和，不能只统计最快或最后一个 kernel。

#### 精度与优化边界

输入和输出采用 FLOAT32，不使用 FP16/BF16/HF32 降精度替代。精度目标严格对齐 §3.2 的 FLOAT64 真值计算后收窄结果，不能把普通 FP32 累加或 FP32 BLAS 的结果当作验收真值。NPU 实现评估分块多累加器、乘积低位补偿、补偿归约及 alpha/beta 最终合并的补偿；对有限输入的中间溢出、下溢和强相消单独验证，不能仅因最终输出类型为 FLOAT32 就接受中间精度丢失。需要缩放时使用可恢复的数值缩放并验证极值与特殊值语义，不改变原始测试输入。具体路径是否采用以完整精度与逐条性能结果为准。

禁止通过缩小随机输入范围、放宽阈值、替换 golden、跳过失败 case 达标。补偿路径须核查目标编译器优化后的实际舍入行为。编译选项遵循仓库默认，不预设额外文件级优化属性。

## 工程组织与兼容性

预期源码置于 `blas/gemv/arch22/sgemv_host.cpp`、`sgemv_kernel.cpp` 和必要的 kernel/tiling 头文件；测试置于 `test/gemv/sgemv/arch22/`，复用公共参数、golden、DeviceBuffer 和 BlasTest。按实际工程收集规则补齐 GTest 注册，不新建独立构建体系、不重复公共接口声明、不改变 arch35 行为。

同步维护 `blas/gemv/README.md` 的支持产品、参数、约束和调用示例。提交前检查 arch22 选源、符号唯一性、API→Host→Kernel 调用链，以及 arch35 的编译兼容。架构差异留在对应目录，公共修改须说明必要性并验证已有路径。

## 支持硬件

| 产品 | 设计支持 | 本任务要求的实测 |
| --- | --- | --- |
| Atlas A2（910B3） | 是，arch22 | 精度、完整性能和工程构建 |
| Atlas A3 | 是，arch22 | 适配构建；任务书不要求独立真机自验 |

## 算子约束逐项核对

| §2.5 约束 | 设计对应措施 |
| --- | --- |
| 参数合法性 | Host 校验及 int64 跨度计算，错误码测试；校验在 quick return 前 |
| 非连续输入 | lda>m、正负 incx/incy，间隙保持 |
| broadcast | 不涉及，形状由 m/n/trans 确定 |
| dynamic shape | 无图模式要求，运行时标量参与 tiling |
| 原地与视图 | y 原地写回，A/x 只读，不返回视图 |
| 确定性 | 不承诺跨实现/设备位级一致，不依赖原子写输出 |
| 空 Tensor | m/n 为零且参数合法时无 kernel、无数据访问 |
| 0维 Tensor | 不涉及，alpha/beta 为 Host 标量指针 |
| 异步执行 | 使用 handle stream，不在算子内部强制同步 |

调用方须保证有效设备指针、足够容量及符合 BLAS 使用约定的缓冲区关系；不新增 x/y 或 A/y 重叠支持承诺。大跨度错误在分配或访问前返回，不能将极端形状测试中的分配失败误记为算子参数校验通过。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验收口径 |
| --- | --- |
| Golden | 严格采用§3.2：原始FLOAT32输入提升到FLOAT64计算，最后一次收窄为FLOAT32；Netlib BLAS 3.10.0的float32结果仅作辅助对照，不用于验收判定 |
| 有限值 | atol=rtol=2^-13，matched_ratio>=0.99；同时满足任务书 1e-2/32ULP 最大误差约束 |
| 非有限值 | 同为NaN、同号Inf、NaN与任意Inf等效；异号Inf不等效；有限与非有限不等效；独立统计且全部匹配 |
| 性能 | 910B3，msprof op，5次预热后至少10个有效样本，逐case NPU_us<=GPU_ms*1000/0.8 |
| 全量 | 官方1000条精度、200条性能全部执行，报告缺失/失败/未完成，不以子集代替 |

对最大误差同时输出 absolute error、ULP 比值及判定边界；按任务书“1e-2 or 32*ULP”的逐元素可接受上限执行，不用其他元素的较大 ULP 放宽当前元素。全非有限输出的有限统计记为不适用，非有限判定仍必须全部通过。

参考计算直接使用与 NPU 相同的原始 FLOAT32 数据，将 A、x、旧 y 及 alpha、beta 提升到 FLOAT64；按列主序、trans 和逻辑步长计算点积及 alpha/beta 合并，最终仅一次转换为 FLOAT32 并保存 golden。beta=0 时不读取旧 y；合法空形状保持 y 不变。参考计算独立于 NPU 的分块和补偿实现，不复用其求和路径。对参考程序本身交叉验证布局、负步长、特殊值及最终收窄，保留输入、golden、运行环境和逐元素误差证据。

## 测试设计

保留任务原始 CSV、GPU 基线与脚本原件并计算哈希。逐条核验 case_name、参数和返回码，结果必须能回溯到原 case。性能基线按 `(trans,m,n,alpha,beta,lda,incx,incy)` 关联，缺失、重复、单位不明一律不计通过。

保持官方均匀分布用例，同时为每条适用随机数值用例补充同参数的正态分布配对用例，使随机数据测试均匀/正态各占50%；非随机边界和负向用例单独计数。正态采用 GAUSS_NORM_5_2，记录每条 seed、mu、sigma，A/x/y 的 NPU 与 golden 输入相同。追加数量依据实际 CSV 随机用例统计写入报告，不提前虚报固定数量。

补充覆盖：N/T/C、m/n为0与1、非方阵、1048576×4及4×1048576、4095/4097/4098、lda padding、正负非单位步长、偏移指针、alpha/beta全值域、beta=0旧y含NaN/Inf、空指针组合、INT32_MIN与span溢出。A/x、矩阵padding、向量间隙和缓冲区保护值分别检查不变。

非法参数用单元素dummy缓冲或指定空指针直接调用接口，不生成golden、不按非法跨度分配。合法负步长且非空的计算用例必须分配完整物理跨度；不能把 inc<0 本身误当作快速返回条件。

性能以同一份预置输入执行，恢复y初值和H2D/D2H不计入task duration；性能采样前后验证精度，分清算子返回码、异步执行状态、实际有效样本数。msprof按单case采集并保留原始OpBasicInfo及调用映射，多kernel按API调用分组求和。ACL event数据可作辅助诊断，不替代任务书msprof验收数据。优化优先通过msprof定位小形状启动开销、大矩阵带宽、跨步搬运与归约瓶颈，再进行同条件对照。

## 任务材料差异与处理原则

1. §3.1 的 `cblas_sgemm` 逐batch与本算子不符，§3.5及§7.4的float32参考与§3.2存在差异。本设计以§3.2为准：FLOAT32输入提升为FLOAT64，按列主序及逻辑步长进行FLOAT64乘加，最终收窄FLOAT32。beta=0不读取旧y，alpha=0仍遵循公式；不得切换成OpenBLAS或float32累加判定。参考计算与NPU实现分离，保留输入、golden和逐元素指标，并以独立FLOAT64实现交叉核验参考计算。
2. 测试README称msprof仅辅助、ACL event用于判定，与任务书§3.3相反；以任务书msprof为主，同时保留脚本输出供复核。脚本适配须记录diff，不改GPU基线或阈值。
3. 任务书明确排除alpha=0且A/x含Inf/NaN的精度组合，但测试README仍描述该组合。实施前逐行核对实际CSV；若存在冲突条目，保留原件及单独结果并请求任务方确认，不静默删例，也不让golden短路改变NPU要求的语义。
4. 文本部分写“lda>n有列间空洞”，应为lda>m；采用公式及参数表一致的列主序语义。任务书列出的CMake旧状态需在实际开发版本重新核验。
5. 设计不以未经确认的材料矛盾宣称验收达标；最终提交前须把确认后的口径、脚本版本及全部结果固定到交付commit。

## 兼容性分析与交付计划

不修改公共ABI，不影响950已有实现。A2/A3共享arch22路径，设备资源由平台查询。普通非法参数在Host返回，数值特殊值不在Host扫描；大尺寸采用分块，内存不足不通过隐藏截断绕过。

验收前提交：设计文档、源代码、官方与补充测试、逐条精度和性能结果、完整原始日志、构建环境/版本/commit/库哈希、可复现步骤及规范README。按官方 task_submission.zip 模板在私仓根目录建立 task_submission，保留“1 自验证步骤说明.md”“2.1 精度自验证报告.xlsx”“2.2 精度自验证日志.log”“3.1 性能自验证报告.xlsx”“3.2 性能自验证日志.log”“4.1 内存自验证报告.xlsx”“4.2 内存自验证日志.log”；内存项按任务书标记不涉及并说明，不伪造内存通过记录。

验收说明填写完整commit链接、分支和算子目录；提交验收前核验Ascend-CANN开发者访问权限。设计通过后实施开发，测试验收通过再向ops-blas提交代码PR，按检视要求修改并合入。本文未包含已完成测试或已邀请账号的声明。

## 参考资料

- [社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [cuBLAS GEMV接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv)
- [Netlib SGEMV](https://netlib.org/blas/sgemv.f)
- [ops-blas贡献规范](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)
- [混合精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
