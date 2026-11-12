# aclblasCrot A2/A3 算子设计修订 V1

本方案在 Atlas A2/A3 的 arch22 路径实现 complex64 原地 Givens 旋转，支持合法正负步长及 c/s 独立 Host/Device 位置。连续路径采用 T=2880 的双槽搬运、128-float 交替系数、按实际输入范围选择普通或补偿计算，并在同一 kernel 内以 `CrotAIV<DensePath>` 对既有路径进行编译期实例化；显式 UB 占用 181.5KiB。

2026-11-11，在 910B3/CANN 9.1.0 上完成最终自测：两次完整标准构建入口均退出 0，同版正式精度 **3955/3955**、正式性能 **FULL_200：200/200** 通过，README 示例编译和运行均退出 0，执行前后测试程序、算子库、CSV 和 GPU 基线哈希一致。完整链于当日 18:46:54（Asia/Shanghai）完成。以下结论属于本次自验证，不代表官方平台代码或性能验收结论。

[首设计 PR #2813](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2813)已于 2026-11-11 16:42:14（Asia/Shanghai）评审通过并合入，已合入 head 为 `4f8f17aba5399601b421d38d67bc82941778a3a6`。本次 V1 是后续设计更新，补充短系数广播、T2880、连续布局常量化与模板路径实例化，原精度判据、软件补偿、转置及早 y 写回依赖保持。2026-11-11 参与者确认官网已进入“任务开发中/已分配”，此状态来自参与者确认。设计维护：cylus；接口、平台或实现方案变化时同步维护。

源码最终冻结提交为 `b0a24f6730fcc155f2c3f82603980b83cf174934`；下列实现文件与最终标准构建及完整验证产物绑定。

| 冻结文件 | SHA256 |
| --- | --- |
| `blas/rot/arch22/crot_kernel.cpp` | `1dc22ec56d682fd221f72cef58863fd6aa7aae805f056552fb5f41aa041f035f` |
| `blas/rot/arch22/crot_tiling_data.h` | `410600f7c3e27d3b66eab06dab22d400d0ab02ca32d43c3c4c64150c7c94fa31` |
| `test/rot/crot/arch22/crot_test.cpp` | `3d3b48f01e9ec27a3222989917e1e0c850d7d651d1d1134c5d01acafc8386ed3` |

本设计按[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织。

# 需求背景（required）

## 需求来源

| 项目 | 本次绑定或要求 |
| --- | --- |
| 任务 ID | `7a42143a91d54df58451707188133781` |
| 官方页面 | [11 月 aclblasCrot A2/A3 任务](https://www.hiascend.com/activities/task-center/details/7a42143a91d54df58451707188133781?menu=trends) |
| 任务书 | 官方附件中的 aclblasCrot_AtlasA2_A3_task_doc.md（[下载 ZIP](https://www.hiascend.com/p/resource/202611/4251fdbf6acd4a7b83cbfd245ae80016.zip)） |
| 原始附件 | `4251fdbf6acd4a7b83cbfd245ae80016.zip`，2,279,086 字节 |
| 附件 SHA256 | `74B5FB984C7852BB166311F2DF736D274C41A51BFB3EF85CF4CBEB36D5B9588A` |
| 任务契约记录 | 开发仓 requirements/7a42143a91d54df58451707188133781.md；本设计完整列出评审所需接口、算法与验收要求 |
| 实现上游 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) |
| 绑定基线 | `5bd95145a9c499e43ebad44a313a1029c63d54bc` |
| 官方设计渠道 | [cann-ops-competitions 任务目录](https://gitcode.com/cann/cann-ops-competitions/tree/master/04_tasks/01_community-task-2026/tasklist)的设计 PR |
| 软件与设备 | CANN **9.1.0**；Atlas A2/A3，arch22；自验证覆盖一种款型，性能测试任务书指定 Atlas 800T A2（910B3） |
| 当前环境 | 910B3、CANN 9.1.0、CMake 3.31.10；最终标准构建绑定原 Netlib 3.10.0 和 GTest 1.10，Bisheng -O3、dav-2201；完整精度与性能同版验证 |
| 期限和分配 | 官方接口截止时间为 2026-11-23 00:00（Asia/Shanghai）；参与者已确认官网“任务开发中/已分配”（2026-11-11） |

## 背景介绍

在 Atlas A2/A3 上使用 Ascend C 实现单精度复数向量 Givens 旋转。算子原地更新 x、y；c 为实数，s 为复数；y 更新必须使用 s 的共轭及旋转前的 x。使用 ops-blas 句柄携带的 stream 直调 NPU kernel，最终在官方仓交付实现、测试、README 与自测材料。

### 基线实现现状

已只读核实绑定基线：公共头文件已声明 `aclblasCrot`；`blas/rot/arch35/` 已有 Crot 的 Host、Kernel、Tiling；`test/rot/crot/` 已有 Param、Golden、README 和 arch35 测试。`test/rot/crot/CMakeLists.txt` 已无条件调用 `ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)`。arch22 目录目前只有 csrot，无本任务 Crot 实现或测试。

因此，任务书所述“新增公共声明、补 arch22 GTest 注册”属于较早现状：本任务保持已有声明，不重复新增；补齐 arch22 测试源后核实已有注册实际选中目标架构，仅在真实缺项时修改 CMake。arch35 的负 n/零步长规则和 SIMT 路径不能直接用于本任务。

# 需求分析（required）

## 需求描述

实现 complex64、ND 一维 x/y 旋转，支持合法正负步长及 c/s 独立 Host/Device 指针。在任务指定的参数校验顺序下返回状态码，满足 FLOAT32 分量精度和逐性能 case 的 GPU 对标要求。

只实现任务书、验收标准、目标仓强制规范及直接必要的验证，不从旧任务或公共模板追加功能、安全检查或验收门槛。

## 需求拆解

| 范围 | 需求及来源 |
| --- | --- |
| 公共接口 | 复用 `include/cann_ops_blas.h` 的唯一 `aclblasCrot` 签名；任务书 §2.3 |
| 计算 | x'=c*x+s*y；y'=c*y-conjg(s)*原始x；任务书 §2.1 |
| 参数 | handle 优先、负 n 非法、零 n 返回成功、非零且非 INT32_MIN 步长、int32 跨度限制；§2.4/§2.5 |
| 内存位置 | x/y 为 Device，c/s 分别自动判断 Host/Device；§2.4 |
| 正确性 | 全部随附精度 case 的契约一致版本及要求的补充用例；分量混合容差、Inf/NaN；§3.2/§3.5 |
| 性能 | msprof op task duration，5 次 warmup 后至少 10 个有效样本，每 case NPU≤GPU×1.25；§3.3 |
| 内存 | §3.4 不涉及独立内存指标；交付表仍要求内存材料，按澄清项处理 |
| 交付 | 设计 PR、源码/测试/README、自测报告、私仓 `task_submission` 及规定平台验收；§4/§5 |

# 详细设计（required）

## 算子分析

### 数学公式

记 x=xr+i·xi、y=yr+i·yi、s=a+i·b，每个逻辑元素计算：

```text
xr_new = c*xr + (a*yr - b*yi)
xi_new = c*xi + (a*yi + b*yr)
yr_new = c*yr - (a*xr + b*xi)
yi_new = c*yi - (a*xi - b*xr)
```

四个输出均使用同一组旧 xr、xi、yr、yi。先把两个输入搬入 UB（片上统一缓冲区），保留原始分量，再生成两路输出，最后回写。不能先写 x，再从 GM 读取已更新的 x 来算 y。

计算采用 float32 向量操作，不降精度。大系数强相消场景需要控制直接 float32 乘加的舍入误差，因此使用软件补偿：将尾数低 12 位清零得到高部，再由减法得到低部；四项展开计算乘积残差，TwoSum 计算加法残差，最终合并修正量。当前 A2 上 Axpy/FusedMulAdd 的实测不提供这里所需的单次舍入 FMA，不能据 API 名称使用 FMA 残差公式。补偿实现保持 double Golden 和验收阈值不变，最终标准重建产物已通过 3955 项完整精度。

为减少普通数值范围内的补偿开销，实现按实际输入 tile 和 c/s 的绝对值指数分派。记系数最大原始 FP32 指数为 Ec、四个输入分量最大指数为 Em；只有 Ec≤190、Em≤190 且 Ec+Em≤258 才用直接乘加。这保证系数、输入幅值各小于 2^64，乘积幅值上界 AM≤64。以不依赖 RNE 的 u=2^-23、γ5=5u/(1−5u) 估计，三个乘积及两次加减误差≤γ5·3AM≈1.14441e−4，小于 atol=2^-13；下溢/FTZ 项上界 6·2^-62 不改变该结论。指数提取通过绝对值的整数位比较保留 NaN/Inf 标记，非有限数和超界 tile 均走补偿路径。归约只覆盖有效元素，读取一次归约结果用于分支，不以测试编号、固定系数或预知数据选择路径。

范围归约使用保序位编码，避免逐lane整数到浮点转换：Abs及整数Max产生非负FP32位型B，将B右移1位写到独立converted，再按float做ReduceMax。最大结果的位型右移22位恢复原指数；原指数0/1映射后可能被FTZ归零，因此恢复值至少取1，作为保守界。原Inf/NaN映到[1.5,2)的正常数，高于所有有限输入的编码，必恢复指数255并拒绝普通路径。归约源不与result/workspace重叠，原190/258条件及误差界不变。

四个输出始终使用保存的旧 x/y。范围判定已证明全部输入和系数有限时，si=0可用通用实旋转 x'=c·x+sr·y、y'=c·y−sr·旧x；c/s不限定为性能用例中的固定值。非有限或超界输入仍走完整四公式补偿，不用零系数短路改变0×Inf语义。编译器浮点选项与最终源码、产物和测量证据绑定。最终标准重建产物分别完成 3955 项精度与 200 条性能验证，精度和性能各按对应标准独立判定。

### 支持数据类型与形状

`aclblasComplex` 使用目标仓定义，实部/虚部各 float32。x/y 逻辑形状 [n]，n>0 时物理跨度为 `1+(n-1)*abs(inc)`；间隙元素保持原值。c 为 float 标量指针，s 为 complex64 标量指针。无广播、转置、批量、uplo 或额外 leading dimension。

公共 API 保持：

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n,
    aclblasComplex* x, int incx,
    aclblasComplex* y, int incy,
    const float* c, const aclblasComplex* s);
```

### 参数校验与快速返回

| 顺序 | 条件 | 结果 |
| --- | --- | --- |
| 1 | handle 为空，包括 n=0 | HANDLE_IS_NULLPTR |
| 2 | n<0 | INVALID_VALUE |
| 3 | n=0 | SUCCESS；不读标量、不启动 kernel、不改 x/y |
| 4 | n>0，任一步长为 0 或 INT32_MIN | INVALID_VALUE |
| 5 | n>0，任一 (n-1)×abs(inc)>INT32_MAX | int64 计算后判定，返回 INVALID_VALUE |
| 6 | n>0，x/y/c/s 任一为空 | INVALID_VALUE |
| 7 | 参数合法 | 查询标量位置、生成 tiling、异步启动 |

步骤 4–6 返回同一非法值状态，不增加未要求的细分错误优先级。负步长合法：逻辑 i（0-based）对应物理索引为 `i*inc`（inc>0）或 `(n-1-i)*abs(inc)`（inc<0）。字节地址乘 8 后继续用宽整数，不窄化为 int32。

任务书没有规定 x/y 互相重叠或 c/s 与输出重叠的别名行为；本轮不额外增加此支持或检查，按明确的独立向量范围推进，并作为评审澄清项。

## 算子实现

### 实现方案

arch22 采用 **SIMD membase**，使用局部 TPipe、TBuf、DataCopyPad、向量运算及明确的依赖同步。保留两个路径：incx=incy=1 的连续性能路径，以及其它合法步长的批量 DMA、UB 向量重排路径。步长路径只搬运有效复数，不读改写完整物理窗口，不搬出间隙元素。遵守仓内 R1：不通过 SetValue/GetValue 或逐元素 DataCopy/DataCopyPad 循环处理向量，也不为小尾块引入该串行路径。不会复制 arch35 的 SIMT 或 Compact DMA 专有路径。

数学上每复数约 20 次浮点操作；连续路径每元素读 x/y 共 16 字节、写回 x/y 共 16 字节。32n 字节是理想流量估算而非实测。实际耗时还包括范围归约、向量计算、地址准备和同步，不能仅据理想流量或某一 PMU 占比确定瓶颈。

### Host 侧设计

#### 文件与职责

| 位置 | 本任务改动 |
| --- | --- |
| `include/cann_ops_blas.h` | 已有声明，核实签名并复用；不重复声明 |
| `blas/rot/arch22/crot_host.cpp` | ValidateCrotParams、LaunchCrotKernel；入口负责 handle/n 优先检查和调度 |
| `blas/rot/arch22/crot_kernel.h` | Host/Kernel 共用启动器声明 |
| `blas/rot/arch22/crot_tiling_data.h` | 共享字段及常量 |
| `blas/rot/arch22/crot_kernel.cpp` | 标量读取、搬运、计算和回写 |
| `blas/rot/README.md` | 增加 A2/A3 Crot 支持和 arch22 语义，不改变 arch35 声明的差异 |
| `test/rot/crot/arch22/` | 新增架构专属 GTest、wrapper、Golden 入口、分量比较器、CSV 和步骤说明 |
| `test/rot/crot/` | 复用适用的 Param 和已有 CMake；不把 arch22 参数或精度规则写入 arch35 的共享实现 |

任务书明确指定的 `blas/rot/arch22/` 优先于通用模板目录示例。保留仓规范要求的 Host 拆分、共用 kernel.h 和 dlog；不增加无用编译或测试文件。

#### 标量位置和 stream

Host 使用 aclrtPointerGetAttributes 分别查询 c/s。Host 标量读取一次，通过 tiling 传 cValue/sReal/sImag；Device 标量指针原样传 kernel，以两个独立标志选择来源，覆盖 Host/Host、Host/Device、Device/Host、Device/Device。

Device 标量在各核开始处理前读一次，不由 Host 解引用，不做 D2H 或 stream 同步取值。沿用目标仓 Crot 的指针查询失败处理。通过 OP_LOGE/OP_LOGD/OP_LOGI 记录错误、tiling 与启动，不引入 printf/std::cout。

使用 handle stream 异步启动；本算子不申请 Device workspace，不为正常调用附加同步。调用者或测试在读回前同步。错误转换遵循目标仓直调机制，提交成功不冒充设备已完成计算。

#### 分核、Tiling 与 UB

Tiling 字段包括 n、incx/incy、每核元素划分参数、tileSize、路径键、c/s 位置标志和 Host 标量值。采用仓内按值传递 tiling 的直调方式，Device 标量是独立 GM 参数，避免逐调用 malloc tiling 缓冲。

通过仓内 GetAivCoreCount 动态查询平台 AIV 数量，不硬编码核数。常规路径选择 B≤min(可用 AIV 数量,n)，按完整复数分配逻辑连续区间；商/余数字段给出每核范围，不在 tiling 中存核分配数组。连续主路径按四个复数分配整块，剩余尾部归一个核，降低搬运边界开销；小 n 与大 DMA 间隔的处理见下文。不能照搬 csrot 按 float 分量切分的方式。

仓内 `HardwareInfo<ASCEND_V220>::ubSize` 为物理 **192KiB**；共享 kernel_constant.h 的 248KiB 不能作为 arch22 容量。当前 CANN 9.1 的 TMP_UB_OFFSET/TOTAL_VEC_LOCAL_SIZE 为 **184KiB**，且未启用取消编译器预留区的选项，因此不把物理 192KiB 全部当作可分配容量。连续路径统一使用 **T=2880**、补偿子块 **q=512**，显式 UB 为 **185856 bytes = 181.5KiB**，距 184KiB 可分配边界保留 2560 bytes。此实现预算不是新增任务验收门槛。所有视图起点 32B 对齐，有效长度单独传入。

连续路径仅分配 xStage、yStage 和一个统一辅助 TBuf，所有辅助视图均通过同一个 denseAux 的显式偏移建立，不依赖独立 TBuf 之间的邻接关系：

| 区域 | float 数量 | 字节数 | 用途与复用 |
| --- | ---: | ---: | --- |
| xStage | 4T | 46080 | 两个 2T 交织输入槽，计算后在原槽保存 x 输出 |
| yStage | 4T+128 | 46592 | 两个 2T 交织输入槽，首尾各留 256B；旧 y 保留至 x 计算完成 |
| denseAux | 8T+256 | 93184 | 两个 outY、swap、transpose、128-float 系数及 512B 地址空隙 |
| 合计 | **16T+384** | **64T+1536 = 185856** | **181.5KiB**，范围归约工作区从当前 outY 槽划出 |

按 denseAux 起点计，全部区间为半开字节区间：

| 视图 | 区间 | 生命周期 |
| --- | --- | --- |
| outY 槽 0 | `[0,23040)` | 当前槽范围检查、归约结果与最终 y 输出；写回完成前保留 |
| outY 槽 1 | `[23040,46080)` | 与槽 0 交替，允许上一槽 y 写回与当前槽计算重叠 |
| 地址空隙 | `[46080,46336)` | 保留原 256B bank-group 相位差 |
| swap | `[46336,69376)` | 范围检查中间值或普通复数路径的成对交换结果 |
| transpose | `[69376,92416)` | 两遍转置临时区，按 x/y 次序复用 |
| 地址空隙 | `[92416,92672)` | 系数视图累计相移 512B |
| coefficient | `[92672,93184)` | 128 floats，重复保存 `(-si,+si)`；descriptor 和初始化长度一致 |
| 补偿复用区 | `[46336,87296)` | 复用 swap/transpose 内 20q floats，保持双 outY 与 coefficient 独立 |

这些相对偏移用于保持双源指令的 bank-group 相位安排，性能按正式采样判定。两个输出槽、转置临时区和系数区大小均由公共常量计算；连续 Stage、Dense、OutputY、SlowView 和范围归约视图不再从运行时 tile 字段重复构造同一尺寸。Host 只在 incx=incy=1 时选择连续路径，并设置同一个 `CROT_TILE_CONTIGUOUS=2880`；每核剩余元素和尾部 count 仍动态计算。

进入补偿路径时，在 aux 的 46336 字节处依次布局旧值 4q、输出 4q、算术临时 8q、高部/索引 4q，共 20q floats=40KiB，到 87296 字节止；不接触两个 outY 槽或系数区。容量断言为 `20*512 <= 4*T`。每个子块先算完四个输出再写到当前 xStage/outY 子范围，不能覆盖先前 y 输出或另一槽尚在写回的数据。补偿后按现有逻辑失效快路缓存；同槽再次装入前等待上一轮写回。

正负步长路径保持 T=256：x/y 的 32B 槽共64T bytes，旧值16T、输出16T、算术临时32T、索引40T，总 **168T bytes = 42KiB**；不按物理跨度分配窗口。Gather 的源和目的分离；API 的显式工作区计入以上布局。算子不申请全局临时 workspace，句柄与测试框架已有分配不计为算子新增空间。

### Kernel 侧设计

#### 单 kernel 的路径实例化

`CrotAIV` 使用 `template<bool DensePath>`，`IsContiguous()` 返回该编译期布尔值。唯一 `crot_kernel` 入口仍接收原 tiling 和 GM 指针；仅在入口按 `tiling.pathKey == CROT_PATH_CONTIGUOUS` 选择 `CrotAIV<true>` 或 `CrotAIV<false>`。一个公共 TPipe 和 SetAtomicNone 位于分支之前，每次调用只执行一个实例的 Init/Process，Host 启动器仍只有一次同名 kernel launch，ABI 不变。

true 实例使用上述连续双槽路径。false 实例同时覆盖 STRIDED 和 SINGLE_ELEMENT：保留 tile256 的串行 ProcessTile、42KiB 布局、正负步长地址公式、Gather/PackOutput 与原同步。超出 DMA 间隔字段的合法单元素路径仍每核一元素、DMA blockCount=1、gap=0，不被模板化改成连续路径。

实例化使共享 helper 中的连续/步长条件可由编译器常量折叠；不按 case、系数值或预知输入选择实例。局部 `.template ReinterpretCast<...>` 是依赖成员模板的 C++ 语法消歧义。数学、精度判据、转置及事件顺序保持不变；模板可能影响生成代码体积和寄存器分配，最终效果以实际编译和验证结果为准，不宣称已经获得逐指令归因。

#### 连续路径

1. 按完整复数区间分核，每个tile读入交织x/y，GM有效长度始终为8t字节。双槽交替装入；处理当前槽前可预载下一槽。对当前实际2t个float分量检查范围，尾部padding不参加归约。
2. 范围允许普通计算时，直接在交织布局中计算。si=0用前述实旋转；一般复数先成对交换旧x，得到 y'=c·旧y−sr·旧x+(-si,+si)·swap(旧x) 并保存在当前outY；旧x的用途全部完成后，再用同一交换临时区计算 x'=c·旧x+sr·旧y+(-si,+si)·swap(旧y)，输出到当前xStage。交替系数只初始化128 floats：Duplicate128，再以偶数lane掩码做2次repeat的Muls(-1)。每次系数乘法用两条Mul，normal mask=64、repeat=ceil(2t/128)≤45、dst/src0 repeat stride=16个32B块、src1 repeat stride=0；第二条的源、目的和系数起点均加64 floats。两条覆盖互不重叠的64-lane半区，每个有效分量恰好乘一次；末尾round128仅访问转置已写且已分配的padding，Add和GM写回仍只处理有效2t。每分量仍为三乘两加，Axpy不依赖单次舍入FMA。
3. 配对交换使用A2支持的两遍TransDataTo5HD<float>，在固定16项地址元数据中改变行列映射，使输出[i]=输入[i^1]。每repeat处理128个float，repeat=ceil(2t/128)；单repeat时src/dstRepStride=0，否则均16个32B块。两遍之间及结果读取前有V屏障；尾部可访问的padding仍在本槽分配范围内，且有效配对结果不依赖padding值。地址列表循环只生成固定指令元数据，不逐元素计算。
4. 超界或非有限输入时，将大tile分成q≤512的子块；完整2880元素块为5×512+320。GatherMask pattern 1/2保序拆分实/虚部，走软件补偿四公式，再用索引Gather恢复交织。计算和回写严格按当前有效count。
5. 每次回写只写8t字节。复杂普通路径在y结果完成后，以V→MTE3事件slot+2提前搬出outY，同时继续用未改写的StageY计算x；末尾以事件slot写x。实路径、补偿与stride仍在末尾写xy。每槽仅在最后写回后发一次MTE3→MTE2完成事件，MTE3顺序保证它同时覆盖早y与晚x；同槽复用前才等待。每tile局部标志区分早y，防止重复写回。末尾消费两个槽完成事件（未使用槽消费初始化事件），再做PIPE_ALL收尾。

CANN 9.1 的 [Interleave](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_rearrange_ISASI/Interleave.md) / DeInterleave 明确不支持 A2/A3，不能迁用 arch35 的对应调用。本方案使用当前SDK已实编并验证的TransDataTo5HD<float>、GatherMask/Gather，不要求GM指针额外32B对齐。

#### 正负步长路径

同一逻辑 tile [L,L+t) 对应 x/y 两段配对元素。对每个向量，令 a=abs(inc)，DMA 按物理升序搬运：正步长起点为 L×a，负步长起点为 (n−L−t)×a，地址及字节间隔均先用 64 位计算。常规批量搬运参数如下：

| 方向 | blockCount | blockLen | srcStride | dstStride |
| --- | --- | --- | --- | --- |
| GM→UB | t | 8 字节 | (a−1)×8 字节 | 0 个 32B dataBlock |
| UB→GM | t | 8 字节 | 0 个 32B dataBlock | (a−1)×8 字节 |

搬运区使用 VECIN/VECOUT 位置，算术平面可用 VECCALC；使用 A2/A3 支持的不带 PaddingMode 模板参数的 DataCopyPad 原型，其原有布局按每块补齐 32B；不调用仅新架构支持的带 mode 重载。GM→UB 每个 8B 复数补为一个 32B 槽，前两个 float 为有效数据。一次批量 DMA 覆盖整个 tile，既不对每个复数发起串行 DMA，也不搬运 gap。UB→GM 从这些 32B 对齐槽仅写各自的 8B 有效数据，padding 丢弃，gap 从未成为写回数据。

输入索引由 CreateVecIndex 生成逻辑 j，再用整数 Muls/Adds 得到字节偏移：正步长为 32j/32j+4，负步长为 32(t−1−j)/32(t−1−j)+4。Gather 分别产生四个旧分量平面，负步长因此恢复为逻辑顺序。四个输出计算完成后，再用向量索引和 Gather 构造物理升序的输出槽：槽号通过标量移位版 ShiftRight 得到，负步长反转槽对应的逻辑元素，奇偶 lane mask 选择同一 TBuf 内的实/虚平面。仅槽内前两个 float 会写到 GM，其余 lane 可以重复合法分量值；Gather 的 count 限定为有效槽范围，不访问尾块外的反向索引。

DataCopyExtParams 的 A2/A3 srcStride、dstStride 为 32 位无符号间隔，最大 UINT32_MAX；不能在窄类型内计算 (a−1)×8。若任一向量的该间隔超限，任务既有 (n−1)×a≤INT32_MAX 约束推出 **n≤4**。该小 n 路径每个逻辑元素分配一个 AIV，每核仅处理一个 tile，DMA blockCount=1、srcStride=dstStride=0，直接用 64 位 GM 地址定位；这不是核内逐元素循环，也不改变合法参数范围。当前环境查询得到 40 个 AIV，满足该路径 n≤4；实现仍动态查询核数，不把合法大步长改判 INVALID_VALUE，也不退回 R1 禁止的循环。

DataCopyPad 的非对齐能力不免除 UB 起点 32B 对齐要求。不能把 GM blockLen 上取整为 32B，或将输出槽按 8B 间隔放入 UB。官方说明 UB→GM 丢弃补齐数据；本方案没有依据要求添加原子加或整窗口合并。连续主路径按四复数分块可减少核边界，但不将 x/y 指针或步长同时满足 32B 分界作为新增输入限制。最小正确性阶段覆盖跨核相邻复数、尾块、混合正负步长和 gap 保持，验证实际写回行为。

所需 API 的产品支持已与官方文档和绑定基线交叉核实：`CreateVecIndex<int32_t>`、`Muls/Adds<int32_t>`、**移位量为标量**的 `ShiftRight<int32_t>`、`Gather<float>` 支持 A2/A3；Gather 的 offset 类型为 uint32_t、单位为字节，源和目的不能重叠。移位量为 Tensor 的另一 ShiftRight 重载以及 Scatter 不用于本方案。仓内 arch22 cgemm_strided_batched_direct_kernel.cpp 已有对应索引生成、批量 8B DMA 和 Gather 调用。当前 CANN 9.1.0 上的组合已编译，负步长、mask/尾块及同步经过最终完整精度验证；后续改变搬运或索引逻辑时复验相关用例。

#### 同步与原地正确性

MTE2完成后才读UB计算；每路结果完成后才允许对应MTE3写回，同槽再次复用前等待两路全部写回。双槽预载使用显式槽地址，不更改activeSlot；范围归约的result/workspace只放在当前outY。两路y输出独立，向量临时区顺序复用；提前写GM_y不改变StageY的旧输入。指数分派仅归约当前有效tile，V_S后读取标量；无核间共享中间结果或核间同步。

## 支持硬件

| 产品 | 本任务支持目标 | 验证安排 |
| --- | --- | --- |
| Atlas A2（arch22） | 支持 | 910B3、CANN 9.1.0；最终标准重建、3955 项精度、200 条性能及 README 示例均通过 |
| Atlas A3（arch22） | 支持 | 按任务允许，自验证覆盖 A2/A3 中一种 |
| Ascend 950 等 arch35 | 本任务不新增 | 保留既有实现与测试语义 |

CANN 要求 9.1.0；不能用 8.5/9.0 镜像或旧环境通过记录代替本次工具预检。

## 算子约束限制

仅 complex64 x/y、float c、complex64 s；n 为 int；合法步长为非零、非 INT32_MIN 的 int，且元素跨度不超过 INT32_MAX。原地更新但不返回视图。无 broadcast、动态 shape 图推导或额外批量接口。别名范围按前述澄清项处理，不从 arch35 继承本任务未要求的特殊路径。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 本任务要求 | 来源 |
| --- | --- | --- |
| 精度 | 实/虚部分别 atol=rtol=2^-13、matched_ratio≥0.99、最大绝对误差 ≤max(1e-2, 32×ULP(g_low)) | 任务书 §3.2 |
| 非有限值 | 同 NaN、同号 Inf、NaN 对 ±Inf 等效；有限/非有限不等效；单独统计且全部匹配 | 任务书 §3.2 |
| 性能 | 每 case 的 msprof op 平均 task duration ≤1000×gpu_ms/0.8 | 任务书 §3.3 |
| 采样 | 5 次 warmup 不计，连续成功有效采样≥10，算术平均 | 任务书 §3.3 |
| 内存 | §3.4 不涉及独立内存指标；交付材料按 N/A 说明，未单独测量 | §3.4 与 §4 |

已回读任务书指定的[固定版本精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) §2.3：与更高精度实现的单一标杆直接比较。本任务固定使用下述 double Golden；有限值混合容差和最大绝对误差均比较 double(actual) 与 double Golden，容差中的 |golden| 也取 double 参考值，不先将参考结果收窄为 float32。

FLOAT32 最大绝对误差上限取 max(1e-2, 32×ULP(g_low))；g_low 仅取最大绝对误差位置的 double Golden 按 RNE（round-to-nearest-even）收窄至 float32 后的值。不能用全 Tensor 最大幅值或 actual 代替，不能仅用现有 MERE/MARE 代替任务指标。除该 ULP 锚点外，收窄仅用于标准规定的非有限值分类：double Golden 收窄后为同号 Inf 时，可与实际输出的同号 Inf 匹配，不能因收窄前有限而直接判为失配。非有限值不进入 ULP 或有限误差统计；只有非有限值时报告实际数量，不能虚构有限统计。

## 正确性用例和 Golden

测试复用适用的 CrotParam、CSV 加载和 BlasTest，使用 arch22 局部 Golden 入口与 wrapper。将实际送入 NPU 的 float32 x/y/c/s 分量提升为 double，保存旧 x/y，按 [Netlib 3.10.0 crot 公式](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/SRC/crot.f)独立计算 x'=c*x+s*y、y'=c*y-conjg(s)*旧x；中间运算和参考输出均保留 double。有限值误差使用该 double 参考，仅非有限值分类和 ULP 锚点按上述标准 RNE 收窄到 float32。标杆在运行前固定，不根据 NPU 结果切换参考实现或舍入口径。

Golden 不调用不存在的 cblas_crot，不复用 kernel 分块或索引函数。共享 crot_golden.h 接受负 n 和零步长，其错误码规则不能用于 arch22；其中 volatile float 的逐步舍入乘加也不作为本任务高精度标杆。保留 arch35 已声明的规则，Netlib BLAS 3.10.0 依赖按任务和仓框架准备。

在 arch22 目录实现任务专用分量比较器，不改变公共 Verifier：公共实现的逐元素 32ULP 硬限、将配对非有限值计入比率分母等行为不等于本任务标准；arch35 测试仅接受 NaN/NaN 或同号 Inf，也不能替代任务的 NaN/Inf 等效规则。分别输出 x/y 实部/虚部的有限元素数、混合容差匹配数及比率、最大绝对误差及按该误差位置 g_low 计算的上限，以及非有限值总数和通过数。matched_ratio 分母仅含有限值比较项，全部非有限项须按任务规则通过。

| 阶段 | 验证内容 |
| --- | --- |
| 最小闭环 | n=1、小非对齐长度、纯虚 s/一般复数 s，同时检查 x/y，捕捉共轭和旧 x 错误 |
| 跨核搬运验证 | n=4、incx=1/incy=-1 覆盖 4 核相邻单块写回；TC_EX_0570 的 n=129、incx=2/incy=-1 在当前 40 AIV 下由 9 核各处理 4 元素、31 核各处理 3 元素，覆盖同一 32B 边界附近的批量 gap 写回。检查有效值及间隙，不增加原子或 GM 32B 对齐限制 |
| 参数边界 | 空 handle 优先级、负 n、零 n、空 x/y/c/s、零步长、INT32_MIN、两向量跨度边界及溢出 |
| 功能覆盖 | ±1/±2/±3 组合、质数/2幂±1/大规模、四种标量位置、偏移和尾块 |
| 数值覆盖 | 均匀和正态、特殊 c/s、全零/交替/极值、Inf/NaN |
| 完整回归 | 原始 1000 精度 case 的契约一致派生版本，加实际需求缺项；原件不修改 |
| 布局回归 | 合法连续输入中的局部有限大值，覆盖同核快→多补偿子块→快，检查全部x/y输出及缓存重建；不改变CSV或精度标准 |
| 非有限归约回归 | 连续输入中的单个 NaN/Inf 位于首 repeat、后续 repeat 和尾部，与另一输入同 lane 有限大值配对，检查四个输出分量；不改变分量判据 |
| 框架闭环 | CSV、GTest 枚举/执行/结果的 case 数一致，不漏测/重复或把错误状态记 PASS |

原始 CSV 无 GAUSS token，c/s 也未覆盖任务要求的随机分布，派生用例已补齐均匀/正态覆盖并记录数量。现有 GAUSS_NORM_5_2 填充虽随机选择 μ∈[-5,5]、σ∈[0.1,2]，还会将采样值 clamp 至 [-5,5]；任务书的正态描述未规定该截断。本任务在 arch22 局部数据准备中按规定 μ/σ 采样且不额外截断，实/虚部独立采样，报告明确与现有 token 的差异、seed 和覆盖数量，不修改全局 fill 行为。非法或 n=0 用例使用 dummy/规定空指针，不按非法跨度预分配；合法负步长正常分配和计算。实际派生 CSV 含 3948 个精度用例和原始 200 个性能用例；普通输入均匀/正态各 1962 项，随机标量均匀/正态各 981 项。特殊值、非法参数、空输入和性能用例单独统计。派生脚本、清单和原始 SHA256 保留。最终精度集合为 **3948 条 CSV + 7 个固定 GTest = 3955 项**，已在最终标准重建产物上全部通过。正式精度步骤退出 0、耗时 137.162207368 秒，无缺失、额外、重复或失败；7 个固定测试均执行且未跳过。证据为 `runs/final/b0a24f6730-01/evidence/accuracy/accuracy-summary.json`，XML SHA256 为 `e85c7e59c507a07f870ed94ffd9e2e8d4b3499e8d09003883aea9369d7fc2a96`。正式精度、性能和 README 示例使用同一测试/库/CSV 绑定，执行前后哈希一致。

| 固定 GTest | 验证目的 |
| --- | --- |
| `HostScalarsAndTail` | Host 标量与非整块尾部 |
| `DeviceScalarsAndNegativeIncrement` | Device 标量及负步长 |
| `IndependentScalarLocations` | c/s 独立 Host/Device 位置 |
| `DenseRangeTransitionPreservesOutputs` | 同核普通→补偿→普通切换、输出和缓存复用 |
| `DenseNonFiniteReductionPreservesOutputs` | n=4099 的 6 个小场景；NaN/Inf 在元素 1/33/103，覆盖首/后 repeat 与尾部，避免非有限标记或同 lane 有限大值丢失 |
| `ValidationPrecedence` | handle/n 等错误优先级 |
| `InvalidStridesUseDummyBuffers` | 非法步长按契约返回，测试不按非法跨度预分配 |

固定测试与 CSV 分开统计，6 个非有限小场景属于一个固定 GTest，不扩写成额外 CSV 或改变验收基数。

## 性能采集和优化

正式性能使用当前目标环境的 msprof op，按单 case 采集，统计一次 API 的全部 kernel task duration，单位 us。固定版本的 warmup/采样机制已通过真实采集、解析和样本映射核对；API 返回 SUCCESS 且精度检查通过的调用才计为有效样本。正式结论使用该工具的 Task Duration。

官方[工具说明](https://gitcode.com/Ascend/msopprof/blob/tag_MindStudio_26.1.0.B102_001/docs/zh/user_guide/attention.md)明确独立 msopprof 与 msprof op 接口一致；如采用独立官方版本，记录真实入口、版本和产物哈希，不使用自造外壳。正式 runner 显式执行5次 Crot 预热，再执行10次真实 API 调用；每次复原该 case 输入，拷贝置于 kernel 计时之外。依任务书§3.3/7保留官方 kernel replay 的 warm-up=5，应用热身和工具内部热身分开记录，均不计入10次真实 API 样本。当前实现每 API 单 kernel、无前置 helper，使用 launch-skip-before-match=5、launch-count=10 排除应用热身。工具内部 PMU 重放不能充当10次 API 样本。

固定 B102 的 [kernel replay](https://gitcode.com/Ascend/msopcom/blob/tag_MindStudio_26.1.0.B102_001/csrc/runtime/inject_helpers/ProfDataCollect.cpp) 会备份、恢复受跟踪的 aclrtMalloc 分配，最终执行[原始单次 launch](https://gitcode.com/Ascend/msopcom/blob/tag_MindStudio_26.1.0.B102_001/csrc/runtime/HijackedFuncOfKernelLaunch.cpp)；这不免除本算子的原地结果验证。application 模式内置 warm-up 实为 D2D 拷贝，不能作为5次 Crot 热身的证据。正式用例固定同一 replay 模式，并保留 profiling 下的有效状态、golden 检查及完整样本映射；最终 200 条性能用例各完成 15 次真实 API 精度检查，正式均值各来自 10 个有效官方样本；完整结果及采样映射按 case 独立保存。

性能 CSV 有 200 个连续步长 case，包含小 n、2幂边界、1048576/2097152/4194304 和不同标量位置，最终逐条完成完整采集。原地输入在每次有效调用前恢复，恢复排除在 kernel duration 外，避免重复旋转改变分布或溢出。测量使用单卡串行执行，绑定源码、产物、命令和环境。

原始 GPU CSV 的 200 行与原始 TC_PF 的 200 条按行序逐一对齐，n/incx/incy 全部一致；gen_csv.py 按 pf_rows 同序写入 GPU 基线。按原始性能用例序号与 GPU 行号建立一一映射，保留 case 名称、标量位置和值、填充及 seed；不按 shape 字典覆盖重复行。此映射依据生成代码和行序核对，附件没有 GPU 原始采样记录，已回填耗时来源仍以提供方为准。

最终性能范围为 **FULL_200**：200 条全部通过，每条保留 10 个官方有效 Task Duration 样本，完整性能步骤耗时 2627.682659986 秒。GPU/NPU 比率按 `gpu_us / npu_mean_us` 定义，任务门槛为该比率 ≥0.8，等价于 NPU 均值≤GPU/0.8。

| 最终完整性能数据 | NPU 均值（us） | GPU 基线（us） | NPU 上限（us） | 结果 |
| --- | ---: | ---: | ---: | --- |
| TC_PF_1001，n=1048576，实系数 | 16.9039999 | 18.9781332 | 23.7226665 | PASS |
| TC_PF_1028，n=1048576，复系数 | 22.2340001 | 18.8245334 | 23.53066675 | PASS |

全 200 条的最小 GPU/NPU 比率为 **0.8466552719**，对应 **TC_PF_1028**，高于 0.8 门槛。以上均值和比率均取最终标准重建的 FULL_200 数据；`performance/performance.csv` SHA256 为 `8bb4065179eba2dd3dc06fa26f52786eb71f2a59cdaa79f55710976f2875e71a`。

### 最终自验证与产物绑定

源码版本为 `b0a24f6730fcc155f2c3f82603980b83cf174934`，平台为 910B3/CANN 9.1.0，编译使用 CMake 3.31.10、Bisheng -O3、dav-2201，无 debug 宏。最终验证完整链在 2026-11-11 18:46:54（Asia/Shanghai）完成，结果为 `FINAL_CHAIN_PASS`。

| 步骤 | 已验证结果 |
| --- | --- |
| `bash build.sh --ops=crot --soc=Ascend910B3` | 完整标准入口退出 0 |
| `bash build.sh --ops=crot --soc=Ascend910B3 --run --device=0` | 完整标准入口退出 0；实际执行 4155 项：4148 CSV（3948 精度+200 性能参数）+7 固定，0 失败；未指定性能 case 时的专用 Profile 入口按设计跳过 1 项 |
| 正式精度 runner | 3955/3955 PASS，3948 CSV+7 固定，0 失败/缺失/额外/重复，固定测试 0 跳过 |
| 正式性能 runner | FULL_200，200/200 PASS，10 有效样本/case，官方基线行序及 CSV 读回通过 |
| README 示例 | 编译退出 0，运行退出 0；使用同一最终算子库 |
| 执行前后绑定 | 下列测试程序、算子库、用例 CSV 和 GPU 基线 SHA256 全部一致 |

| 绑定对象 | SHA256 |
| --- | --- |
| `build/test/rot/crot/crot_test` | `c4284e0deac9fa74b41eadfa0cf50e2e821ba9d749e9953b77918d35ffad37ac` |
| `build/libops_blas.so` | `c2810d508b15646fcd7afef773764e3b88f6d3475ee8e5d7ee9936f5b7a614af` |
| `test/rot/crot/arch22/crot_test.csv` | `0f8f5417e34fa3c20f706a3b1449e25ef70078fdc0361eafec84d7c251769f97` |
| 原始 GPU 基线 CSV | `ea643c8221ffa08ec1590001132debc27c8cc1e7eec177e1563f315a7a758384` |

完整命令、退出码、环境和前后哈希见 `runs/final/b0a24f6730-01/evidence/result.json`；正式精度和性能明细分别位于其 `accuracy/` 与 `performance/` 目录，README 输出为 `readme-example-run.log`。完整归档共 126924343 字节，SHA256 为 `adf29a552633f48a2f3c5726442fd1379b95718116e2d01ea3f736b244396d33`；14680 个证据文件逐项哈希回读通过，校验记录为 `runs/final/b0a24f6730-01/verification.json`。这些证据说明本次自测结果，不代替官方平台验收。

## 兼容性分析

保持公共 Crot 签名和 arch35 现有语义，新增 arch22 产品实现。错误码、Golden 入口、比较器和正态数据准备放在 arch22 局部，不修改 arch35 或公共 Verifier/fill 的既有行为。已有 CMake 注册应复用，构建按 SOC 选择架构，只对真实缺项改动。

## 构建与验证环境

本次必需工具已在绑定的 910B3/CANN 9.1.0 环境完成实际操作验证，覆盖构建及 Ascend C 编译链接、ACL/设备执行、CSV/GTest/Netlib、精度指标解析、官方 msprof op 采集及解析输出和报告生成。工具及逐项操作证据记录在开发仓任务契约中。

最终标准入口使用独立的 **CMake 3.31.10** 真实 ELF：`preflight/deps/cmake-3.31.10-pypi/cmake/data/bin/cmake`，SHA256 `dfeb6e3350440a8feaa8a63b550cc0d431ebe0be25e01af6568c68451ffc8506`。通过官方 `CMAKE_TOOLCHAIN_FILE` 环境入口绑定 Netlib 3.10.0、GTest 1.10 和 CANN 9.1.0；toolchain SHA256 为 `2fb80d9cdd6852edf0d149248c16e0d065579f72abe02331582800f5f09aa233`。最终两次完整 build.sh 标准入口及同产物精度/性能/示例运行均已通过；具体命令和依赖解析记录见最终证据。

正式性能使用官方独立 msopprof B102，其接口与 msprof op 一致，ASCEND_HOME_PATH 绑定实际 CANN 9.1 SDK。SDK、运行时、NPU 或必需工具变化时，只复验受影响项；同一任务且环境未变时复用本次证据。

## 冲突和待确认项

| 编号 | 问题 | 设计处理/下一项验证 |
| --- | --- | --- |
| D1 | README/CSV 放行负 n、零步长，与 §2 arch22 契约冲突 | 以任务书明确 INVALID_VALUE 规则为准；保留原件，派生用例记录修订 |
| D2 | README/性能脚本按事件 avg_us 判定，与 §3.3 msprof 冲突 | 已采用官方 msopprof B102，其接口与 msprof op 一致，最终完整采集、解析及报告链已完成 |
| D3 | GPU 基线重复 shape 键 | 已核实200行顺序与生成代码；按原始case序号对应基线行，不用shape覆盖；正式报告保留映射证据 |
| D4 | §3.5 含 trans/uplo/批数/Aarray 模板内容 | 不扩展接口，以 §2.4/§2.5 实际入参设计用例 |
| D5 | §3.5 把 inc≤0 写入 quick return 缓冲说明 | 负步长合法计算，不能 dummy/no-op；仅非法及零 n 使用 dummy |
| D6 | 内存“不涉及”与必交内存报告/日志并存 | 按 §3.4 提供 N/A 说明，未单独测量内存指标；UB 静态预算见实现设计，不冒充实测值 |
| D7 | 任务书的声明/注册现状过时 | 绑定基线已具备，复用并核实 arch22 选择，避免重复 |
| D8 | 跨向量别名未明确 | 不自行扩展别名支持；32ULP已按指定固定标准明确为最大误差位置golden的ULP |
| D9 | arch22 DMA/索引及大 stride 分核验证 | 最终标准重建的 3955 项精度已通过；动态 AIV=40，TC_EX_0570 覆盖 40 核批量跨核 8B 写回，合法大 stride 单元素路径单独覆盖 |
| D10 | 目标环境与工具实测进度 | 910B3+CANN 9.1.0 全部必需工具已通过，最终标准重建、3955 项精度、FULL_200 性能和 README 示例均已完成自验证；官方平台验收另行记录 |

## 参考依据

- 任务书 §2–§5：功能、边界、环境、验收与交付。
- 绑定 ops-blas 基线 `5bd95145a9c499e43ebad44a313a1029c63d54bc`：`include/cann_ops_blas.h`、`blas/rot/arch35/crot_host.cpp`、`crot_kernel.cpp`、`test/rot/crot/CMakeLists.txt`、`crot_golden.h`。
- arch22 复数重排参考：绑定基线 `blas/dgmm/arch22/cdgmm_kernel.cpp` 中 GatherMask/Gather；目标组合已在本次 CANN 9.1.0 编译和运行。
- 绑定基线 `blas/gemm_strided_batched/arch22/cgemm_strided_batched_direct_kernel.cpp:41–60,108–139`：向量索引、批量 8B DataCopyPad 与 Gather；其超大 stride 逐元素循环不沿用。`blas/common/arch/hardware.h:27` 给出 arch22 192KiB；`blas/common/helper/host_utils.h:75–82` 动态查询 AIV；`blas/herk/arch22/cherk_kernel.cpp:992–1007` 为按值 tiling 启动依据。
- 官方 [DataCopyPad GM→UB](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.html)和 [DataCopyPad UB→GM](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.html)：A2/A3 使用不带 mode 的原型，UB 起点 32B 对齐，GM 间隔单位为字节，写回丢弃补齐数据；当前 SDK 的实际调用组合已编译并通过本次精度验证。
- 官方 API 产品/类型支持：[CreateVecIndex](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_padding/CreateVecIndex.html)、[Muls](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Muls.html)、[标量 ShiftRight](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/logical_compute/ShiftRight.html)、[Gather](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/scatter_gather/Gather.html)；以当前 9.1.0 编译验证具体调用组合。[CANN 9.1 Scatter](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/scatter_gather/Scatter_ISASI.md) 不支持 A2/A3。
- 仓内 `agent/skills/repo-coding-rules/references/R1-禁止逐元素操作.md`：未给 UB 或小尾块串行循环豁免。测试隔离依据为共享 `test/rot/crot/crot_golden.h:45–56,74–93`、`test/frame/verify.h:340–373`、`test/rot/crot/arch35/crot_test.cpp:130–159,208–214` 和 `test/frame/fill.h:243–252`；具体差异已列入本设计。
- 仓内 `agent/skills/repo-op-templates/SKILL.md`：SIMD membase、Host 拆分、共用 kernel.h 与日志要求；任务书指定目录优先。
- 已回读 [cuBLAS rot](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rot) 的 Crot 标量类型及 x/y 原地接口；复数共轭公式和负步长起点以任务书及 [Netlib 3.10.0 crot.f 原文](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/SRC/crot.f) 为直接依据。Netlib 使用 STEMP 保留旧 x，负步长一基起点为 (1−n)inc+1，对应本设计零基 (1−n)inc；其旋转归一化背景不新增为本任务的输入限制。Netlib 的 n≤0 返回及零步长行为不作为本任务 arch22 状态码依据；cuBLAS 通用 rot 展示式也不单独用作复数共轭细节证明。

## 交付状态

| 内容 | 状态与范围 |
| --- | --- |
| 首设计 | PR #2813 已评审通过并合入；本次 V1 为后续设计更新 |
| 实现 | 源码冻结提交 `b0a24f6730fcc155f2c3f82603980b83cf174934`；公共 ABI、单 kernel 启动及 arch35 语义保持 |
| 标准构建与自测 | 两次完整标准入口、正式 3955 项精度、FULL_200 性能和 README 示例均通过，使用同一绑定产物 |
| 官方验收与代码发布 | 按任务平台流程另行提交和记录；本设计不将自测通过表述为官方验收通过 |
