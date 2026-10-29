# 需求背景（required）

## 需求来源

2026 年 10 月社区任务「aclblasCgemm3m 算子开发（950）」。目标硬件 Atlas 950PR，CANN 9.1.0。
接口名 `aclblasCgemm3m`，工程目录名 `aclblasCgemm3m950`；950 后缀不进入公共 API。
设计按 cann-ops-competitions 的官方 design_template.md 章节编写，模板获取日期 2026-10-28，
仓库提交 `17cd8ea3d7a5fa9d0053da4eaca4656c0bc8a9ad`。设计尚未提交社区评审。

## 背景介绍

基于 ops-blas 提交 `05b96ea22cea926ecfd43fc7ac691a795522bd61`。
实现位于个人 Fork `gzh1630/ops-blas` 的 `feature/aclblas-cgemm3m-950` 分支，
[已推送提交](https://gitcode.com/gzh1630/ops-blas/commit/dd4553b08d43a2ad1e4894f6e5b22d73ab1567f2)
为 `dd4553b08d43a2ad1e4894f6e5b22d73ab1567f2`。
既有 arch35 Cgemm 已有三乘法预处理、Cube 计算、向量重组及融合流水线，但只在 alpha=1、beta=0
时选择三乘法。新增接口复用这些 kernel，在所有需要 A×B 的场景强制选择三乘法。
原 `aclblasCgemm` 保持原来的路径选择和返回值语义。

# 需求分析（required）

## 需求描述

`C = alpha * op(A) * op(B) + beta * C`，complex64、列主序，支持 N/T/C 九种组合和合法 lda/ldb/ldc。
标量指针位于 Host，矩阵指针位于 Device。通过 handle 中的 stream 异步提交计算。

## 需求拆解

| 项目 | 行为 |
|---|---|
| 空 handle | HANDLE_IS_NULLPTR |
| 非法枚举、负维度、非法前导维、空标量指针 | INVALID_VALUE |
| m=0 或 n=0 | 参数校验后 SUCCESS，不写 C |
| k=0 或 alpha=0 | 不引用 A/B，只计算 beta*C |
| m,n>0 | C 必须非空 |
| beta=0 | 不读取旧 C |
| 输出 padding | 保持原始字节 |
| 重复输入 | 固定调度和归并顺序，输出逐位一致 |

# 详细设计（required）

## 算子分析

令转置及共轭操作后的 A=Ar+iAi、B=Br+iBi：

- P0=(Ar+Ai)×Br
- P1=Ar×(Bi−Br)
- P2=Ai×(Br+Bi)
- Re(A×B)=P0−P2；Im(A×B)=P0+P1。

再执行复数 alpha/beta 线性组合。三个产品使用 FP32 输入与累加，不降为 FP16/HF32。
输入为 complex64；m/n/k 为非负 int，leading dimension 遵循标准列主序约束。
不支持 A/B 与 C 存储重叠、广播或任意非连续步长。

## 算子实现

### Host 侧设计

新增 `blas/gemm/arch35/cgemm3m_host.cpp` 导出公共 API，转入共享 `LaunchComplexGemm`。
`forceThreeProducts=true` 决定三产品工作区和计算路径；旧接口传 false。
新接口将非法枚举映射为任务要求的 INVALID_VALUE，并允许 alpha=0 时 A/B 为空。

按平台 API 获取 AIC/AIV 数量，按可用 tile 数限制实际核数；每个输出 tile 有唯一写入者。
workspace 通过 handle 既有管理器复用，计算前检查尺寸乘法与区域对齐溢出。
同一 handle/stream 的资源生命周期遵循上游约定；测试在取回结果前同步 stream。

一般路径：AIV 拆分交错复数并处理共轭 → Cube 的三个实数 GEMM → AIV 重组及 alpha/beta。
短 K 且 alpha=1、beta=0 时使用既有片上融合路径；输出 tile 足够多时采用既有大 tile 融合流水。
融合计算使用双缓冲与跨核事件管理 L1/L0/UB 生命周期，最后写回保留用户 ldc。

K 分区按 L1 面板容量决定，按固定 part 次序合并，避免原子浮点累加造成不确定性。
非融合三乘法路径联合搜索输出 tile 与 1～4 个 K 分区，小输出保留占用率策略；
融合路径每个分区最多两个 L1 面板，在 UB 中按固定次序合并。
两组大矩阵各 4096 点的独立长双精度诊断显示 RMS 误差约降低一半，仍不等同 Netlib 验收通过。
三乘法仅分配三份全局乘积工作区；完整候选对比与验证记录随 IT 验收材料提供。
对于窄输出、受限 K 且输出 tile 至少覆盖 3/4 Cube 核的情况，直接分面板在片上准备输入，
避免全局变换平面和额外内核启动；并行度不足时继续使用独立乘积路径。

连续方阵 m=n=k、2048≤k≤4096、alpha=1/beta=0 时启用部分和轨迹及稀疏舍入纠正。
该范围由纠正内核的固定 UB 容量和较长归约的适用范围限定，其他形状/布局/标量走通用路径。
将每段 K 长度限制在 128，利用同一 Cube 流水线的模板特化输出实/虚部分和以及路径积分。
为每个输出分配一个 float 路径值、一个最坏情况下的选中索引；输入排列、能量和分核计数
也放在 handle 工作区内。C 仍只占调用者声明的 ldc×n 个复数。

### Kernel 侧设计

复用 `cgemm_kernel.cpp`、`cgemm_compute_kernel.cpp`、`cgemm_epilogue_kernel.cpp`。
预处理对共轭输入虚部取负，构造辅助和差平面；Cube 按输入方向与列主序交换方案读取。
三产品输出按固定顺序重组；非单位 alpha 或非零 beta 使用通用 epilogue。
alpha=0/k=0 使用单独缩放路径；beta=0 模板分支消除旧 C 读取。
既有短 K 路径对非有限结果的标量修正保持不变，三次矩阵乘法分解保持启用。

新增 `cgemm3m_rounding_kernel.cpp`：根据转置方向准备连续 A 行/B 列，融合分段排列和能量统计，
筛选后只对选中输出重算。对于部分和 s0→s1、实际段长 d，累计
`I += d*(s0²+s0*s1+s1²)/9`，分别维护实部/虚部，输出二者最大值。
风险方差近似为 `2^-48*(I + EA*EB*sum(d²)/(18*k²))`；固定系数 3.75 与 0.01 比较。
这是顺序舍入的统计近似，不是严格误差界；不能仅凭公式宣称任意输入均通过精度。
所有分支只依赖形状、布局和当前数值，golden 只用于 Host 测试。

纠正中 64 个 SIMD lane 分别处理连续 K 段。第一遍形成三乘法的复数项和段和，
并行前缀和估计各段初值；第二遍从该初值按段内次序累加，最后归约各段的末值减初值。
输入按段交错预排列，使向量加载连续；两个输入缓冲与输出缓冲交替搬运。
固定 K 段次序、固定前缀扫描和归约结构保证同输入重复结果一致；不使用浮点原子累加。
Cube 统计特化使用 224 KiB UB（目标平台 API 实测 248 KiB），纠正内核显式使用 165.25 KiB。
当前集成版的逐项状态和证据随 IT 验收材料提供。

## 支持硬件

| 芯片 | 支持 |
|---|---|
| Atlas 950PR / arch35 | 是，CANN >=9.1 的对应开发套件 |

## 算子约束限制

矩阵必须位于对应设备，用户保证分配空间覆盖有效 leading dimension；
传入任意未分配地址不属于可由 Host 参数校验恢复的输入。
不新增 950 专用平行接口；其他架构暂不实现本 API。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 判据 | 来源 |
|---|---|---|
| 精度 | 实部/虚部独立，atol=rtol=2^-13，匹配率>=99%，绝对误差<=0.01或32ULP | 任务 §3.2 |
| 非有限值 | NaN/NaN、NaN/Inf、同号Inf匹配；有限/非有限或异号Inf失败 | 任务 §3.2 |
| 性能 | 全部200条 GPU/NPU>=0.4，倍率算术平均>=0.5 | 任务 §3.3 |
| 正式计时 | msprof op，预热5次，有效10次取均值 | 任务 §3.3 |
| 开发计时 | msprof application task duration求和，预热10次，60次去最大10次 | 项目 AGENTS.md |

测试位于 `test/gemm/cgemm3m`，包括原始1200条CSV、补充边界测试、逐位确定性检查、
严格设备采样汇总脚本和Netlib 3.10.0构建脚本。复现见该目录 README.md。
工具故障与算子数值失败分别记录，不将开发计时或OpenBLAS结果替代正式要求。
已归档的本地自验结果：Netlib 精度与逐位重复性 1206/1206，原 Cgemm 回归 51/51，
正式 msprof op 性能 200/200；倍率 `GPU_us/NPU_us` 最低 0.406155449、平均 0.651992516。
这些结果仅代表本地自验，官方 CI、设计评审和华为验收尚未完成。
任务书 §7.4 的“有效采样 >10”“达到 ≥ GPU数据”与 §3.3 的详细性能判据冲突，
本设计按 §3.3 自验，申请验收时提请任务方明确适用标准。

## 兼容性分析

复用已有资源管理、kernel及构建框架；旧Cgemm调用传false保留调度语义。
CMake登记新接口的开发套件版本门槛，低版本不构建不具备实现的测试目标。
新增公共符号，不更改既有API参数、结构体布局或枚举取值。

## 资料来源

- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [ops-blas](https://gitcode.com/cann/ops-blas)
- [Netlib/LAPACK 3.10.0 官方版本](https://github.com/Reference-LAPACK/lapack/releases/tag/v3.10.0)

以上在线资料于2026-10-28核对；任务包原文保留在项目根目录ZIP中。
