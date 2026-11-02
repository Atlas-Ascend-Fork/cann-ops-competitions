# aclblasCher Ascend 950PR 算子设计

状态（2026-10-30）：已在 Ascend 950PR / CANN 9.1.0 上编译通过，1002 项精度测试及 200 项 msprof op 性能测试全部通过。精度包含 n=4097 的缓存/分块边界验证。性能保留 3200 次真实硬件采样，同时核对前 10 次和全部 11 次有效采样均值；原始数据已下载并独立复核。本文记录开发者自验证结果，社区评审按任务流程办理。报告附图来自实际 XML/CSV 的结果视图。

# 需求背景（required）

## 需求来源

2026 年 10 月社区任务“aclblasCher 算子开发（950）”。任务书随包提供，
验收目标为 Ascend 950PR、CANN 9.1.0。设计按社区 design_template.md 的必填章节组织。

## 背景介绍

为 ops-blas 补充 complex64 Hermitian 秩 1 更新接口：

```text
A(i,j) <- A(i,j) + alpha * x(i) * conjugate(x(j))
```

alpha 是 Host float，x 和 A 是 Device complex64。矩阵为列主序，
仅引用和写入 uplo 指定的三角区域。对角虚部视为零，更新后写成零。

## 仓库接入检查

基准仓库为 cann/ops-blas，基准提交为 `05b96ea22cea926ecfd43fc7ac691a795522bd61`。
本地开发分支为 `feat/cher-ascend950`。

| 检查项 | 结果 |
| --- | --- |
| 公共接口 | 基准 include/cann_ops_blas.h 无 aclblasCher；在 extern C 区内新增 |
| 代码目录 | 遵照任务书放入 blas/syr/arch35/，文件以 cher_ 开头 |
| 架构 | ascend950 映射 arch35 / dav-3510 |
| 实现发现 | blas/CMakeLists.txt 递归收集 arch35 源码，按 cher_ 前缀保留单算子源码 |
| 测试发现 | test/CMakeLists.txt 可发现 test/syr/cher；cmake/test.cmake 可发现 cher_ 源码 |
| 测试注册 | 新增 test/syr/cher/CMakeLists.txt，复用 ops_blas_add_gtest_tests |
| 参考实现 | 同族 Ssyr、Csyr；RegBase 复数装载/交织参照 cgetrf_batched |
| 外部依赖 | CANN、GTest、CBLAS/LAPACK；golden 要求 Netlib BLAS 3.10.0 |
| 当前硬件验证 | 950PR 上 SDK 编译、1002 项精度和 200 项性能均通过 |

构建发现依靠文件命名和现有扫描规则，不修改全仓构建逻辑。
仓库主分支包含较新的 SDK 兼容代码；实际服务器必须记录 CANN 与 asc-devkit 版本，
用 9.1.0 验证后才能确认兼容性。Host 接口测试不等同于 SDK 编译验证。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

| 参数 | 约定 |
| --- | --- |
| handle | 非空；使用绑定 stream 异步下发 |
| uplo | ACLBLAS_UPPER 或 ACLBLAS_LOWER |
| n | 非负，0 为提前返回 |
| alpha | 非空 Host float 指针；值复制进 tiling，不保留指针 |
| x | 物理长度至少 1+(n-1)*abs(incx)，只读 |
| incx | 有符号 complex 元素步长；不为 0 或 INT_MIN |
| A | lda*n 个 complex 元素，列主序、原地更新；与 x 不重叠 |
| lda | n>0 时至少 n |

INT_MIN 拒绝规则来自附带测试 README 和 TC_ED_150，是相对任务书基础 incx≠0 的增强约束。
不扩展 Device alpha、广播、其他 dtype 或其他芯片支持。

## 需求拆解

1. 公共 ABI、Host 校验和提前返回顺序。
2. 连续数据 SIMD 与一般有符号步长实现。
3. 指定三角和 padding 的精确写保护。
4. CBLAS golden、CSV 参数化 GTest、非有限值及边界覆盖。
5. 逐 case 的 msprof kernel 性能采集与基线匹配。
6. 上机后补充实测结果、日志和验收报告。

# 详细设计（required）

## 算子分析

### 数学公式

令 `x(i)=ri+i*ii`，对每列 j，先计算 `tr=alpha*rj`、`ti=-alpha*ij`。

```text
delta.real = ri*tr - ii*ti
delta.imag = ri*ti + ii*tr
A.real += delta.real
A.imag += delta.imag            (i != j)
A.imag = 0                      (i == j)
```

运算次序对齐 Netlib CHER 的列标量折叠。x(j) 两分量均为零时，
按 Netlib 语义跳过该列更新，仅将对角虚部置零，避免零乘 Inf/NaN 改写其他元素。
alpha=0 则整矩阵保持不变，包括对角虚部。

### 支持数据类型

alpha 为 FLOAT32；x/A 为交织存储的 COMPLEX64。

### 支持形状

n、lda 为运行时参数；x 的物理长度至少为 1+(n-1)*abs(incx)，A 为 lda*n 个复数。
输入物理布局由 lda/incx 描述。索引乘法先提升到 64 位，避免 n*lda 或 n*incx 在 int 域溢出。
API 不负责从裸指针推断分配大小；调用方必须提供足够存储。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

严格按照任务书 §2.5：

```text
handle==null -> HANDLE_IS_NULLPTR
n<0          -> INVALID_VALUE
n==0         -> SUCCESS（不检查剩余参数、不读取 alpha）
非法 uplo    -> INVALID_ENUM
lda<n        -> INVALID_VALUE
incx==0 或 INT_MIN -> INVALID_VALUE
alpha==null  -> INVALID_VALUE
*alpha==±0   -> SUCCESS（x/A 可为空，不下发 kernel）
x/A==null    -> INVALID_VALUE
查询 AIV 核数；为 0 -> EXECUTION_FAILED
组装 tiling 并在 handle stream 上下发
```

入口仅处理基本校验、提前返回和 ValidateCherParams/LaunchCherKernel 调用。
不分配 workspace，不同步 stream。异步执行错误由调用方同步时发现。
实际下发核数 `min(n, GetAivCoreCount())`，不硬编码板卡核数。

##### 1 分核策略

按列循环分配给实际启动的 AIV 核；每个选定三角输出只由一个核写入，不需要原子操作。
实际核数由运行时板卡能力和 n 共同决定，避免固定板卡核数假设。

##### 2 数据分块和内存优化策略

Host 只传递 n、lda、uplo、alpha、incx 五个 tiling 字段，不分配 workspace。
SIMD 路径列内按最多 4096 个复数分块、每批处理两列，复用 x 的前缀缓存和 A 原地缓冲；具体内存布局见 kernel 侧设计。

##### 3 tilingkey规划策略

本项目采用 kernel 直调，不注册额外的框架 tiling key。
kernel 根据 incx=1 或其他合法步长选择 SIMD/SIMT 路径；n=0、alpha=0 在 Host 侧提前返回。

#### 3.2.2 kernel侧设计

优先评估 SIMD：连续步长的三角列段可批量 DMA 并用 RegBase 进行复数运算，采用 SIMD 主路径。
任意有符号 incx 的物理跨度可能超出有限 UB 和 DMA stride 字段，故通用路径采用 SIMT GM 寻址。
未将稀疏输入整体复制到连续 workspace，避免同步、多次 kernel 和额外内存依赖。

| 条件 | 路径 | 分工 |
| --- | --- | --- |
| incx=1 | SIMD RegBase | 核间循环分列，每批两列，列内按最多 4096 个复数分块 |
| 其他合法 incx | SIMT | 核间循环分列，线程处理该列三角范围内的不同行 |

SIMD：TPipe 在 kernel 入口创建，以引用传入初始化；对象不持有 TPipe。
每核使用两个 A 槽、两个备用 x 槽和一个 x 前缀缓存，最大 UB 用量 165376 字节。
A 槽每个容纳 4096 个复数，在寄存器读完当前片段后原地写回 UB，不另设输出缓冲。
备用 x 槽和前缀缓存各追加 64 个复数的 UB 空间，保证非对齐向量装载的预取不越界。
每核启动时一次性缓存 x 的前 4096 个复数，既供列标量取值，也直接供向量计算读取，
避免每列重复从 GM 搬运相同的 x。超出缓存的列标量或行段回退到原始 GM 访问/分块搬运。
缓存的 LOWER 行起点可能不对齐，采用 LoadUnAlign 和寄存器 DeInterleave；
A 装载/写回使用 DIST_DINTLV_B32 / DIST_INTLV_B32，每个寄存器组处理 64 个复数。
两个列槽共用一次 MTE2→V、对角处理和 MTE3→MTE2 屏障，并融合为一次 VF 调用。
尾段采用有效元素 mask，GM DMA 长度按实际字节数设置，不越过三角边界。

负步长映射：`physical(i)=(n-1-i)*(-incx)`；正步长为 `i*incx`。
测试和实现均传原始物理起始指针，不在 wrapper 中重新打包向量。

每个输出由唯一核/线程写入，不需要 atomics。UPPER 只写 row<=col，
LOWER 只写 row>=col，padding row>=n 从不写入。
不同路径都遵循列标量折叠和零列短路；对角结果虚部直接置零。

## 支持硬件

| 产品 | 代码适配范围 | 当前验证状态 |
| --- | --- | --- |
| Ascend 950PR | arch35 / ascend950 | CANN 9.1.0 编译、完整精度自测及性能数值门槛通过 |

## 算子约束限制

不支持输入输出重叠、Device alpha 或超出 incx/lda 语义的布局。
实测结论仅对应下述已记录环境和任务原始用例；其他硬件和 SDK 版本需另行验证。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 判据 |
| --- | --- |
| 参考实现 | Netlib BLAS 3.10.0 cblas_cher，CblasColMajor |
| 分量精度 | 实虚分别：atol=rtol=2^-13，有限值 matched_ratio>=0.99 |
| 最大误差 | 最大绝对误差点阈值 max(1e-2,32*ULP(golden)) |
| 非有限值 | 按任务书 NaN/Inf 规则单独判断，不计入有限值统计 |
| 工程误差 | CSV MERE/MARE 作为补充，固定生态阈值不受 CSV 列调整影响 |
| 对角虚部 | 更新时严格为零；alpha=0/no-op 时保持原样 |
| 写保护 | 非操作三角、padding 用 NaN payload、-0、subnormal 等哨兵逐字节比较 |
| 精度用例 | 原始 1000 条 CSV，加空 handle 和正态/忽略对角虚部补充测试 |
| 性能用例 | 原始 200 条，每条 GPU_us / NPU_us >=0.4 |
| 采样 | 每条先预热 5 次，随后 60 次 kernel 耗时均值；A 每次调用前恢复 |

原始 CSV 保持逐字节一致并记录 SHA256。普通随机填充按种子奇偶选择均匀/正态，
实虚独立随机；正态参数使用框架 GAUSS 的 μ∈[-5,5]、σ∈[0.1,2]。
额外覆盖随机实数 alpha 和非零/NaN 输入对角虚部。

性能基线存在重复 shape key，不能通过 shape 字典覆盖。
验证器按原始生成器的性能用例序号匹配 baseline ID，再核对 uplo/n/alpha/incx。
NPU 数据读取 msprof 任务时间线 `op_summary_*.csv` 的 CHER kernel `Task Duration(us)`，
按任务开始时间排序，检查 AI_VECTOR_CORE 类型和设备/流/任务标识唯一。
独立 OpBasicInfo.csv 也可严格解析，但不混用导出格式。
不用 GTest wall time，不接受缺基线、零用例、跳过用例或缺失 profiler 数据为成功。
采集格式或 kernel 命名与 SDK 不一致时应显式报错并上机适配，不能降低判据。

## 兼容性分析

新增独立 aclblasCher ABI，不改变 Ssyr/Csyr 或其他产品线的实现。
Cher 实现和测试只在 arch35 构建。接口可被公共头文件包含；其他架构不承诺具有该符号。

## 算力到账后的验证顺序

1. 记录驱动、固件、CANN、asc-devkit、SoC、源码 commit 与 Netlib 版本；核对 950PR。
2. 构建 cher 并检查导出符号和 CMake 实际源码清单。
3. 跑空 handle、n=0、alpha=0、边界和小尺寸用例。
4. 跑全部 1000 条精度 CSV 与补充测试，保存 XML、日志及截图。
5. 先采集典型规模，再采集完整 200 条性能；定位 SIMD 屏障/搬运成本并调优。
6. 回归全部精度和写保护检查，整理实测报告和可复现命令。
7. 按社区任务流程提交设计与代码 PR，处理评审。

## 实测记录（2026-10-30）

| 项目 | 实测结果 |
| --- | --- |
| 系统 / 板卡 | Ubuntu 22.04.5 x86_64 / Ascend 950PR |
| 驱动 / 固件 | 25.7.rc1.10 / 9.0.0.110.220；板卡 Compatibility=OK |
| CANN / 编译器 | CANN 9.1.0，bisheng，GCC 11.4.0，CMake 3.22.1 |
| Golden | Netlib BLAS/LAPACK 3.10.0（Ubuntu 3.10.0-2ubuntu1），显式核对实际动态链接路径 |
| 编译 / ABI | Cher 真实 SDK 编译成功；动态库导出 aclblasCher |
| 精度 | 1002/1002 GTests，通过、无失败/禁用/错误；含 1000 条原始 CSV 和 2 项补充测试 |
| 补充边界 | 正态输入测试含 10 个子检查，覆盖 UPPER/LOWER 的 n=4097、lda=4105 |
| 性能 | 200/200 原始用例通过；msprof op application replay，3200 次硬件采样，每例剔除 5 次应用预热后保留 11 次有效采样 |
| 全部有效采样最小性能比 | 0.4453713031，判据为 GPU 基线耗时 / NPU kernel 平均耗时 >= 0.4 |
| 前 10 次采样最小性能比 | 0.4230375402，同样逐例检查 >= 0.4 |
| 最差性能用例 | TC_PF_1178，UPPER，n=530；GPU 基线 5.907 us，NPU 11 次平均 13.26309073 us，前 10 次平均 13.9632998 us |

GPU 数据来自任务随附基线，本轮未重新测量 GPU。通过任务性能门槛不代表每条用例均快于 GPU。
NPU 性能取 msprof op 的 OpBasicInfo Task Duration(us)，不使用 GTest 总耗时或主机计时。
完整精度证据标识为 `20261030T053126130082Z`；性能证据为 `20261030T053216294343Z`。
精度 XML 含 902 组实际计算的实部/虚部指标，其中 10 组为补充子检查；108 条原始 no-op/负向用例按状态和矩阵保护检查。
独立构建目录还复验了自验证步骤中的完整配置、编译、导出符号及动态链接检查。
原始 XML、CSV、profiler 数据、编译日志、动态库及源文件哈希单独归档，不提交到源码仓。

2026-10-29 的 task-trace 数据和早期失败采集不混入以上均值。
本次容器通过驱动公开映射查询确认：物理 1 对应逻辑 0，profiling 通道接口接受逻辑 0。
进程内适配器只在双向映射验证成立且显式启用时转换 profiling API 的输入/返回设备编号，
驱动错误码、硬件采样和时长保持原样，不修改驱动安装文件或云权限。
具体编译和复现命令见 test/syr/cher/README.md。
msprof 自带预热参数显式设为 5，应用每次调用前恢复 A；使用 application replay 重新初始化输入，避免原地更新累积。
为兼顾第 3.3 节的 10 次和第 7 节的超过 10 次要求，报告同时保留前 10 次及全部 11 次有效采样的均值，两项门槛均逐例通过。
独立复核包括源码 SHA256、完整 XML 身份/状态、全部 CSV 的数字启动顺序、原始硬件 duration/profiling 文件、950PR 元数据及两项均值。

## 交付与社区流程

1. 按官方设计模板复核本文，在 cann-competitions 对应任务目录提交设计 PR 并完成评审合入。
2. 按官方 task_submission.zip 附件字段提供精度及性能 Excel 报告、实部/虚部分量明细、结果附图、自验证步骤、完整原始硬件数据和日志。
3. 个人代码仓为 https://gitcode.com/gcw_HuYbgqHW/ops-blas-cher-950 ，开发分支 feat/cher-ascend950，算子目录 blas/syr/arch35；验收材料记录已发布的具体 commit 链接。
4. 按任务书在个人仓邀请 Ascend-CANN 为开发者。
5. 按社区流程提交 ops-blas 代码 PR，处理评审；本地测试通过不等于 PR 已合入或社区已验收。
