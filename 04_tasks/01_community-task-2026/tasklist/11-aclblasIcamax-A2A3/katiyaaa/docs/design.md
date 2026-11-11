# 需求背景（required）

## 需求来源

本设计对应 11 月社区任务 aclblasIcamax（Atlas A2/A3），依据任务发布的 `aclblasIcamax_AtlasA2_A3_task_doc.md`、`test_cases/README.md`、用例 CSV、GPU 基线及验证脚本。提交团队为 katiyaaa。本文件为开发前设计，不代表已完成性能或精度验收。

## 背景介绍

### 算子功能与工程现状

在 complex64 向量中查找 `abs(real)+abs(imag)` 最大的元素，返回首个最大者的 1-based INT32 索引。该运算不是欧氏模，也不是分别对实部、虚部求 argmax。

ops-blas 的 `include/cann_ops_blas.h` 已声明 aclblasIcamax；`blas/iamax/arch35/` 已有 950 实现，`blas/iamax/arch22/` 已有 Isamax。此次扩展 arch22，不新增平行接口，不覆盖 arch35 实现。参考同族参数校验、handle/stream/workspace 管理和测试组织，但不直接移植 arch35 的 SIMT 或架构专用指令。

# 需求分析（required）

## 需求描述

使用 Ascend C kernel 直调实现 A2/A3 aclblasIcamax，遵循 cuBLAS 参数语义及任务指定的 Netlib BLAS 3.10.0 golden。输出索引逐例精确相等，性能逐例满足 NPU 平均耗时不超过 GPU 的 1.25 倍。

```cpp
aclblasStatus_t aclblasIcamax(
    aclblasHandle_t handle, int n, const aclblasComplex* x,
    int incx, int* result);
```

| 参数 | 位置及类型 | 约束 |
| --- | --- | --- |
| handle | Host 句柄 | 非空，使用句柄绑定的 stream |
| n | Host int | 负数非法；零为 quick return |
| x | Device complex64 | 正常路径非空，逻辑长度 n，实虚交错，各 float32 |
| incx | Host int | 正数按复数元素计步长；零和负数 quick return，不反向遍历 |
| result | Device int* | 始终非空，独立 INT32 标量，正常结果 1..n，quick return 为 0 |

## 需求拆解与口径澄清

1. 校验顺序：handle 空返回 HANDLE_IS_NULLPTR；n<0 或 result 空返回 INVALID_VALUE；之后处理 n=0 或 incx≤0；正常路径才检查 x 和跨度。因此 quick return 允许 x 为空，但不允许 result 为空。
2. 正常路径用 int64 计算 `(n-1)*incx`，超过 INT32_MAX 返回 INVALID_VALUE。复数索引到 float/字节偏移继续使用 64 位，避免乘 2 或乘 8 时溢出。
3. 首元素的模为 NaN 时固定返回 1；首元素非 NaN 时忽略后续 NaN。相同最大值，包括多个 Inf，取最小逻辑索引；全零及正负零并列返回 1。
4. 任务书部分写“新增声明”，但当前公共头文件已有声明：复用既有声明，避免重复符号。A2/A3 选择 arch22，950 保持 arch35。
5. 环境固定 CANN 9.1.0、Netlib BLAS 3.10.0；不能用测试 README 的“指定版本及以上”放宽环境，也不能以 OpenBLAS 替换指定 golden。
6. 任务书性能主口径为 msprof op task duration；配套 README/脚本 Event 口径不同。两种结果分别记录，主验收按任务书，提交时明确差异请评审确认，不能静默替换指标。
7. 任务书内存要求“不涉及”，但交付表要求内存报告：仍提供报告与日志，记录输入、输出及 workspace，不编造内存性能门槛。

# 详细设计（required）

## 算子分析

### 数学公式

以零基逻辑索引 j 表示输入访问：`a[j] = abs(Re(x[j*incx])) + abs(Im(x[j*incx]))`，j∈[0,n)。正常输出为首个最大值的 j+1。模计算使用 FP32 加法，禁止改成 FP64 求和或平方和，以保持 Netlib SCABS1 的溢出、并列语义。

串行参考状态以 a[0] 初始化，仅当新值严格大于当前最大值才更新。并行实现先保留全局首元素 NaN 标志，然后将其他 NaN 标为无效候选；有效候选按“模更大优先，模相等时索引更小优先”合并。空分区也使用显式无效标志。不能让每个分区的首个 NaN 阻塞该分区后续有效元素。

### 数据类型与形状

仅支持 complex64 输入和 INT32 索引输出。incx≥1 时物理复数长度为 `1+(n-1)*incx`；无广播，无多维 Tensor stride 接口。n 为运行时参数，分区和分块覆盖所有满足跨度约束且存储充足的规模，不按 CSV 的特定尺寸硬编码结果。

## 算子实现

### Host 侧设计

完成前述校验后，从 handle 取得 stream。quick return 用该 stream 的 `aclrtMemsetAsync` 将 result 的四字节置零，不启动归约、不读取 x；运行时调用失败映射为仓内状态码，不无条件返回成功。

正常路径查询 AIV 核数及 UB 容量。小向量优先单核，避免跨核合并开销；大向量按连续逻辑区间分核，区间均衡且互不重叠。核数不超过可用 AIV 数或非空分区数。阈值是可实测调优的调度参数，不根据用例 ID、填充值或 golden 选择路径。

Tiling 包含 n、incx、每核起点/数量、tile 元素数、有效核数及路径标记。字段遵循同族框架传参方式，不建立 ACLNN executor 或私有 handle。tile 大小由 UB 预算确定，预算计入交错输入、实虚拆分/模值、索引、有效掩码及归约临时区；双缓冲启用时计入两份输入空间，保留平台要求的余量。索引始终以整数保存，不转为 float 以免大于 2^24 时失真。

多核路径使用 handle 提供的 workspace 保存每核候选，记录至少 FP32 最大模、INT32 逻辑索引及有效性。各核槽位按搬运粒度对齐，检查实际可用容量；不足返回明确错误，不能越界使用或创建不受控的全量临时张量。workspace 生命周期遵循 handle 和 stream 约定，不在异步完成前释放或复用。

### Kernel 侧设计

连续路径按 tile 搬入交错 float 数据，在 UB 中拆分实虚部，执行 Abs、Add 和带索引归约。具体向量接口仅选用 CANN 9.1.0 arch22 支持的接口；实现前核对接口的对齐、mask、repeat 和临时空间约束。尾部仅访问真实有效输入，填充 lane 标为无效，不能让补零参与全零并列选择。

步长路径按逻辑 j 访问 `x[j*incx]`。优先比较小步长窗口搬入后抽取与逐元素安全读取两种方案；大步长不搬运整个物理跨度。Gather 若使用字节偏移数组，必须验证其位宽范围，超界改用分段基址或 64 位标量寻址。不得因 DMA 向上对齐读取用户分配范围之外。

每核维护有效最大模及最小全局逻辑索引，处理跨 tile 并列。单核直接写 j+1。多核先提交候选，再在同一 stream 下启动合并 kernel；stream 顺序提供阶段间依赖，避免不可靠的跨核自旋和未刷新的标志。最终合并检查全局首元素 NaN 标志，按统一比较规则输出唯一 INT32。

两阶段方案作为正确性基线。若 msprof 表明二次启动限制小中规模性能，优先调整单核阈值；只有验证 arch22 同步机制及全部边界后才考虑融合跨核归约。性能核算须包含两阶段全部 task duration，不能仅上报较快的第一阶段。

### 工程集成

```text
blas/iamax/
├── README.md                         # 更新 A2/A3 产品支持及语义
└── arch22/
    ├── icamax_host.cpp
    ├── icamax_kernel.cpp
    └── icamax_kernel_impl.h          # 仅在实现拆分有必要时新增
test/iamax/icamax/
├── CMakeLists.txt                    # 补齐 arch22 GTest 注册
├── icamax_param.h                    # 复用，必要时兼容扩展
├── icamax_golden.h                   # 固定 Netlib cblas 及 +1 转换
└── arch22/
    ├── icamax_npu_wrapper.h
    ├── icamax_test.cpp
    └── icamax_test.csv
```

源码目录由框架自动收集，不添加冗余 CMakeLists。测试 CMake 在 arch22 分支调用 `ops_blas_add_gtest_tests(${OPS_BLAS} icamax_test)`，保持 arch35 注册可用，避免同一配置重复注册。使用公共头文件类型，不重定义 complex ABI。按贡献指南使用标准许可头、状态码、日志和编译选择；后续需求 issue、代码 PR 分别按社区流程提交。独立验收材料不混入上游算子源码。

## 支持硬件

| 产品 | 设计支持 | 验证计划 |
| --- | --- | --- |
| Atlas A2，910B3 | arch22 | CANN 9.1.0 编译及全量精度、性能 |
| Atlas A3 | arch22 | 独立目标编译；有设备再补实机，不把编译写成实机验证 |

任务书允许自验证覆盖一种款型；正式性能采用指定 910B3。不能将 A3 耗时直接与 910B3 门槛的验证结论混用。

## 算子约束限制

不支持负步长遍历、广播和输入输出重叠。不扫描输入数值合法性，Inf/NaN 按 Netlib 规则处理。正常路径访问合法性由调用方分配足够的 Device 存储保证。执行异步，调用方必须在读取 result 或释放资源前同步绑定 stream；实现不为便于测试而增加全设备同步。

# 可维可测分析

## 精度标准与测试矩阵

冻结原任务书、生成器、验证脚本、1200 行用例和 200 行 GPU 基线的哈希，保留原件；不得重生成覆盖官方输入或修改判定阈值。原 1000 精度用例逐条执行，原 200 性能用例完整保留，结果与 ID 一一对应。

golden 明确链接 Netlib BLAS 3.10.0，保存构建配置、动态链接信息及库哈希。正常路径使用 `cblas_icamax(n,x,incx)+1`；quick return 直接期望 0；负向用例比较状态码，不调用不合法的 cblas。输出 INT32 完全相等才通过，没有 99% 容忍率，也不按实虚浮点容差判定。

官方生成器仅覆盖均匀随机，不能宣称已满足 50% 正态要求。另建补充 CSV，以官方随机分布用例数量为基础配对生成等量 `GAUSS_NORM_5_2` 正态用例，使随机分布子集均匀/正态数量 1:1；mu∈[-5,5]、sigma∈[0.1,2]，实虚独立，保存随机种子。特殊值与负向用例单独统计，不混入分布比例。

补充覆盖：首个/后续/分区首个 NaN、全 NaN、多个 Inf、有限分量相加溢出 Inf、跨核跨 tile 并列、正负零、1-范数与欧氏模排序相反的向量、单元素、对齐边界及偏移视图、INT32_MAX 跨度边界、超界跨度、INT32_MIN 步长、空 handle、各 quick return 与空 x/result 的组合。反复执行、非默认 stream 和结果哨兵检查同步、覆盖与越界问题。

所有负向和 quick return 用例使用 NULLPTR 或单元素 dummy，不按巨大 n/步长提前分配。正常大规模压力用例先安全计算字节数并检查资源，无法运行应记录未覆盖而不是标为通过。

## 性能标准与优化验证

逐一执行原 200 条 TC_PF；通过参数/ID 校验与 GPU 基线一对一关联，禁止仅测试前三条或用几何平均掩盖失败。`gpu_us=1000*gpu_ms`，`speedup=gpu_us/npu_avg_us`，每条须 ≥0.8，等价于 `npu_avg_us≤1.25*gpu_us`。使用未四舍五入的数值判定。

正式采集使用 msprof op，排除其 5 次 warmup，至少 10 次有效执行取算术平均。每条 case 独立采集或提供可追溯调用映射，保存原始 OpBasicInfo.csv、任务名、阶段、样本数、状态和聚合公式；多阶段逐次求和后取平均。失败调用记录错误并先解决，不通过删样本制造通过结果。H2D/D2H、造数和 CPU golden 不纳入 kernel 时间。

原 `verify_performance.py` 的 Event avg_us 单独保存并报告，不改写其原始输出为 msprof 数据。若两套判定结论不同，列出差异并向验收方确认，不混用单位或挑选有利口径。依 msprof 的搬运、计算、同步和启动开销调整核数、tile、缓冲策略，所有优化重新执行受影响边界及完整性能矩阵。

## 内存、交付及兼容性

内存无独立性能门槛，但记录正常输入占用 `8*(1+(n-1)*incx)` 字节、输出 4 字节和实际 workspace，验证哨兵、泄漏及生命周期。保留可复现的编译/安装/测试步骤、设备和 CANN 版本、commit、原始日志、整数精度截图、200 条性能表和 profiler 证据。

私仓 `task_submission` 按模板提供 `1 自验证步骤说明.md`、`2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`、`3.1 性能自验证报告.xlsx`、`3.2 性能自验证日志.log`、`4.1 内存自验证报告.xlsx`、`4.2 内存自验证日志.log`。验收填写完整 commit 链接、分支和目录，并按流程邀请 Ascend-CANN。提交前验证压缩包文件数、体积、内容完整性和解压可读性。

设计评审通过后进入实现；当前不声称任何实机测试达标。共享测试辅助文件的修改须兼容 arch35，并回归现有 Icamax/Isamax 构建及测试选择；无必要不改其他算子。设计文档在竞赛仓维护，代码仓提供 README/API 使用说明，设计通过链接引用。

## 参考资料

- [ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)
- [公共接口](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h)
- [arch22 同族实现](https://gitcode.com/cann/ops-blas/tree/master/blas/iamax/arch22)
- [arch35 Icamax 参考](https://gitcode.com/cann/ops-blas/tree/master/blas/iamax/arch35)
- [Netlib icamax](https://www.netlib.org/blas/icamax.f)
- [cuBLAS Icamax](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amax)
