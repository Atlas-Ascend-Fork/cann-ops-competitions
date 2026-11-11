# aclblasIcamax（Atlas A2/A3）设计文档

## 需求背景（required）

### 需求来源

[11月社区任务](https://www.hiascend.com/activities/task-center/details/3187f64ea116468190cabc7779a7cb03?menu=guide)要求在 [cann/ops-blas](https://gitcode.com/cann/ops-blas) 实现 complex64 向量最大一范数元素的最小索引。设计依据为任务网页提供的 [任务书及测试附件](https://www.hiascend.com/p/resource/202611/a65f5dbb923445e18a746550581653f4.zip)，归属 2026 年 11 月。下载原文没有可核实的 Git commit，不以下载哈希代替任务书版本；实现参考仓版本为 `5bd95145a9c499e43ebad44a313a1029c63d54bc`。

### 背景介绍

`aclblasIcamax` 的公共声明已存在，950 产品线有 arch35 实现，A2/A3 的 arch22 目前提供同族 `aclblasIsamax`。本方案在已有句柄式 BLAS 工程内补充 arch22 实现，复用句柄绑定 stream 和 workspace，保持公共接口和其他架构兼容。语义依据 cuBLAS Icamax 与 Netlib BLAS 3.10.0 `icamax`。

## 需求分析（required）

### 接口与参数

```cpp
aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n,
                            const aclblasComplex* x, int incx, int* result);
```

| 参数 | 位置与类型 | 约束及行为 |
| --- | --- | --- |
| handle | Host 句柄 | 空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | Host int | 复数元素数；负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| x | Device complex64，只读 | 实虚部 float32 交错；正常路径不能为空；物理元素数 `1+(n-1)*incx` |
| incx | Host int | 正值为复数元素步长；零和负值均快速返回，不反向遍历 |
| result | Device int 标量 | 任意路径不能为空；正常返回逻辑 1-based 索引；快速返回写 0 |

按 handle、负 n/result 空、quick return、正常输入和跨度的顺序校验。`n==0 || incx<=0` 时允许 x 为空，用绑定 stream 上的设备清零操作写 0 后返回 SUCCESS，不启动归约。正常路径以 int64 计算 `(n-1)*incx`，超过 INT32_MAX 返回 INVALID_VALUE。复数到 float/字节偏移继续使用 64 位计算，避免乘 2/8 溢出。返回值、输入只读、stream 顺序及结果缓冲区的生命周期沿用仓库接口约定。

### 验收目标

输出整数索引与 Netlib CBLAS golden 精确一致。覆盖官方 1000 条精度和 200 条性能用例，补充正态分布、地址偏移、并列值、NaN/Inf 和参数组合。性能在 910B3、CANN 9.1.0 上以任务书 §3.3 的 msprof op task duration 为主：5 次预热、至少 10 个有效样本取均值，每个性能用例 `NPU_us <= 1.25 * GPU_us`。测试 README 使用 Event 时间的说明与正文有差异，Event 时间另行记录，不能混作 kernel 耗时。自验证只需一种硬件款型；A3 实机结果仅在实际执行后报告。任务没有内存达标阈值，仍按交付模板记录分配量和内存检查信息。

## 详细设计（required）

### 算子分析

对逻辑索引 `i=0..n-1`，令 `s[i]=abs(real(x[i*incx]))+abs(imag(x[i*incx]))`，按 float32 相加。结果是 `1+min{i | s[i] 为最大值}`，不计算平方或开方。连续输入需读取约 8n 字节，复杂度 O(n)，大规模以访存和归约为主要开销，小规模主要受启动开销影响。

比较状态是 `(score, logical_index)`：更大 score 优先，相同 score 取更小 index。首个元素的 score 若为 NaN，Netlib 的严格大于更新语义使结果固定为 1；其他位置 NaN 忽略。Inf 以相同的并列规则取首个。有限分量相加溢出为 Inf 也按该规则处理。正负零得到相同 score；尾部填充值和无有效候选的核使用负哨兵并带无效索引，不能胜过真实零。

### Host/Tiling 设计

实现放在 `blas/iamax/arch22/`，主要分为 host 参数校验/分发和 kernel 计算，必要时用一个小头文件共享 tiling 结构。复用 `include/cann_ops_blas.h` 中的公共声明，不创建平行接口。参数以 kernel 参数或按值 tiling 传递，避免每次调用分配、上传 tiling 或同步 stream。

根据实际 AIV 核数及 n 选择核数，小输入用单核直接输出，较大输入按连续逻辑区间均分。核边界优先按向量块对齐，余数分配到有限核，尾块只处理有效元素。每核以固定 UB 容量分 tile；初始考虑每 tile 4096 个复数，实机编译和性能分析决定阈值，不依据输入值或测试编号分派。

多核各写独立 workspace 候选，采用句柄已有 workspace；写入槽按 32 字节分隔，所需空间为 O(核数)，启动前核验指针和容量。首先采用同 stream 上第二个小归约 kernel 合并候选，以 kernel 边界保证可见性；若实测启动开销成为瓶颈，可在平台支持范围内采用经过验证的核间同步单 kernel 方案。单核无需 workspace。不会在设备任务结束前释放工作空间或让并发 stream 复用同一候选区。

### Kernel 设计

连续输入通过 MTE2 搬入 UB，做 Abs 和相邻实虚分量求和，再归约 `(score,index)`。采用 Ascend C arch22 向量 API，不依赖 arch35 的寄存器/SIMT 专有路径。队列或明确事件保证 MTE2→V、V→S、S/V→MTE3 的可见性；跨 tile 保持当前候选，不将索引转为 float 数值，索引以 int32/uint32 保存。

NaN 在归约前用比较掩码替换为负哨兵；原始首元素 NaN 状态独立处理。若硬件归约 API 的并列顺序不能保证最小索引，则先求最大 score，再对等于该 score 的有效索引求最小值。浮点相加不改变顺序或改为更高精度累加；对于次正规数与特殊值，验证硬件浮点模式，必要时增加保持 float32 语义的精确路径。

正步长非连续输入按物理窗口搬运并抽取，窗口长度 `1+(tileCount-1)*incx` 必须落在 UB 和实际输入范围内；步长很大时用 64 位寻址分段搬运或标量读取，不能为了对齐越过调用方缓冲区。连续尾块使用精确字节数的 DataCopyPad 等接口，不依赖调用方额外 padding。输出只写 4 字节，不覆盖相邻内存。性能优化保留这些边界和语义约束。

### 支持硬件及资源约束

| 产品 | 实现目标 | 自验证计划 |
| --- | --- | --- |
| Atlas A2（910B3） | arch22 支持 | 精度、性能及复现验证 |
| Atlas A3 | arch22 支持 | 兼容构建；实机未执行时明确标注 |

CANN 9.1.0；UB 与 AIV 核数通过平台信息和编译资源报告核实。以 4096 个复数计算，双缓冲输入约 64 KiB、score 约 16 KiB，索引、掩码和归约临时区另计；总量须小于实际 UB 限额并预留空间。DataCopy 对齐单位、repeat 上限与索引容量按 arch22 API 限制拆分。流水和归约不能复用仍被另一单元读取的缓冲。设备工作区需按实际使用字节核验，避免把默认 32 MiB 句柄工作区误报为本算子新增开销。

## 可维可测分析

### 测试组织与参考实现

在 `test/iamax/icamax/arch22/` 增加 CSV 驱动 GTest，`test/iamax/icamax/CMakeLists.txt` 为 arch22 调用 `ops_blas_add_gtest_tests(${OPS_BLAS} icamax_test)`，保留 arch35 注册。golden 用 Netlib BLAS 3.10.0 `cblas_icamax()+1`；快速返回直接校验 0，非法参数校验状态码。测试结果必须检查执行数量和进程退出码，避免只通过少量用例即判全通过。

冻结官方 CSV 和 GPU 基线，不用生成器的空基线覆盖实测数据。为原有均匀分布精度场景追加同参数的 `GAUSS_NORM_5_2` 用例，使随机精度场景均匀/正态各占 50%；实虚分量独立采样，正态均值位于 [-5,5]、标准差位于 [0.1,2]，记录 seed 及参数。原 CSV 所有行仍执行。

补充跨 tile/核的相同最大值、非首元素 NaN、首元素 NaN、首个 Inf、全零与常量、欧氏模与一范数排序相反的数据、次正规数、float32 加法溢出、n=1 且极大 incx、跨度溢出、INT32_MIN 步长、空 handle/result/x 的组合以及非 32 字节对齐的输入与输出保护区。非法参数和快速返回用例只分配 dummy 或空缓冲，禁止按非法跨度申请巨量内存。每次调用后同步绑定 stream 再读回，校验输入未改、输出保护区及两次调用不同数据，防止覆盖和状态残留。

### 性能及内存验证

对 200 条 TC_PF 用例预先分配输入、输出和工作区，H2D、造数、golden、D2H 均在计时外。逐用例使用 msprof op 隔离采集，记录预热和有效次数、设备负载、原始 OpBasicInfo.csv 与算子 SUCCESS 状态。多 kernel 方案每次调用的相关 task duration 求和，再跨调用取均值，不只挑最快 kernel。GPU 基线按官方性能 CSV 顺序及 n/incx/填充关联，保留同 shape 不同填充的独立行。Event 时间另存以便核对。

记录最大规模及异常路径的分配/释放和 workspace 大小，检查重复调用后资源不持续增长；内存模板没有任务阈值的项目注明不适用并说明依据。测试失败、工具不可用和未测项分别记录，不能用估计值代替实际测量。

### 复现与交付

从完整验收 commit 获取代码，提供 CANN 环境初始化、Netlib/GTest 依赖准备、构建、全部精度及性能执行、解析日志和从本次结果生成 Excel/截图的顺序命令。保留必要 CSV、基线、配置、原始输出与日志及《复现附件说明》，不依赖开发机私有文件。报告脚本与测试工程一起版本化。

按任务模板在仓库 `task_submission/` 提供自验证步骤及精度、性能、内存三套报告/日志。最终 ZIP 包含必需附件，使用标准 DEFLATE/STORE、低于 100,000,000 字节，检查全部 CRC、完整解压内容和引用路径。验收前刷新任务书并处理变化；独立检出验证部署、功能、解析和报告生成。评审、CI、合入与自测状态各自记录。
