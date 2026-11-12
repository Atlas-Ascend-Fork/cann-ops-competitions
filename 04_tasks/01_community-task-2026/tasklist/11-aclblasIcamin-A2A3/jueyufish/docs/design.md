# 需求背景（required）

## 需求来源

11月社区任务“aclblasIcamin 算子开发（A2/A3）”。需求依据任务包
`aclblasIcamin_AtlasA2_A3_task_doc.md`，文档按社区[设计模板][template]组织。
参与账号为 `jueyufish`。

## 背景介绍

基于 ops-blas 的 Ascend C kernel 直调框架，补齐 `blas/iamin/arch22/` 实现，
复用现有公开接口、句柄、stream、workspace 和 `aclblasComplex` 类型。
实现、测试、原始日志及数值结果对应私有仓库的[固定提交][code-commit]
`6156d6c09de7d8d0b06e4095168d2a724e7a9bc5`，评审者需具备该仓库的访问权限。
补充的 GitCode 页面截图及含截图的精度/性能工作簿对应[后续提交][screenshot-commit]
`081709d7120b14871d2aa43a0b4fc5cb137cbbbf`，原数值证据保持不变。

当前已完成 A2（910B3）自验证；A3 尚未实测。已取得展示归档证据的真实
GitCode 页面截图，非服务器原执行现场截图；原有 PNG 仍为报告渲染预览。
本文提交设计评审，社区评审、任务验收及上游合入均尚未完成；以下结果均为开发者自测记录。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);
```

输入 x 为 Device 上交错存储的 complex64，result 为 Device 上单个 INT32。
第 i 个逻辑元素（从0开始）的模为
`fabsf(x[i*incx].real) + fabsf(x[i*incx].imag)`；返回最小模的 **1-based** 索引，
模相同时返回最小索引。这里的模不是欧几里得模。

| 项目 | 行为 |
|---|---|
| 空 handle | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n<0 或空 result | `ACLBLAS_STATUS_INVALID_VALUE` |
| n=0 或 incx<1 | 不启动计算 kernel；在绑定 stream 上将 Device result 清零，成功返回 |
| 正常计算的空 x | `ACLBLAS_STATUS_INVALID_VALUE` |
| 地址跨度 | Host 用 int64 检查 `(n-1)*incx <= INT32_MAX`，超出返回 INVALID_VALUE |
| NaN | 首逻辑元素任一分量为 NaN 时返回1；后续 NaN 不更新最小值 |
| Inf | 全部为 Inf/被忽略的 NaN 时保留最早合法候选；首元素非NaN时有限值优先 |
| 异步 | 在 handle 的 stream 上下发，API 不主动同步；调用方读回前同步 stream |

参数检查先处理 handle，再处理 n/result，再处理 quick return，最后检查 x
和地址跨度。负步长不做反向遍历；`INT32_MIN` 属 quick return，不能先取绝对值。
公开声明已存在于 `include/cann_ops_blas.h`，不新增私有平行接口。

## 需求拆解

1. 对齐复数 abs1 最小索引、同模首索引、参数检查及 quick return 语义。
2. 采用 arch22 向量指令实现连续与跨步扫描，使用有界 UB 和两阶段多核归约。
3. 补齐 arch22 的 GTest 注册，保留任务原始用例，补充正态分布及边界测试。
4. 按任务要求在 CANN 9.1.0 / 910B3 完成精度和 msprof 性能自验，保存可追溯证据。
5. 保持公共 ABI 和 arch35 实现兼容，明确 A3、截图和社区验收的当前状态。

# 详细设计（required）

## 算子分析

### 数学公式

```text
m[i] = float32(abs(real(x[i * incx])) + abs(imag(x[i * incx])))
best = m[0], index = 0
对 i = 1..n-1：仅当 m[i] < best 时更新 best 和 index
result = index + 1
```

以上描述正常计算路径；quick return 返回0。严格小于更新保留同模首索引，
并定义首元素 NaN 的行为，不能用忽略所有 NaN 的普通 argmin 替代。

### 支持数据类型

| 参数 | 数据类型 | 内存与布局 |
|---|---|---|
| x | COMPLEX64，两个 FLOAT32 分量 | Device，实部/虚部交错存储 |
| n / incx | INT32 | Host 标量 |
| result | INT32 | Device 单值输出 |
| handle | aclblasHandle_t | Host 句柄，携带 stream 和 workspace |

### 支持形状

输入逻辑形状为 `[n]`，输出为单个索引。正常计算时 `n>0`、`incx>=1`，物理
长度为 `1+(n-1)*incx` 个复数元素，并要求 `(n-1)*incx<=INT32_MAX`。
`n=0` 或 `incx<1` 走 quick return；`n<0` 报错。无广播或额外 leading dimension。

### 计算与访存特点

算子是内存读取和最小索引归约。complex64 模只需两个 float32 绝对值和一次
float32 加法，输出不存在浮点容差。CPU golden 初始化为首元素模，再按逻辑
顺序仅在严格小于时更新，因此必须单独保留首 NaN 的语义。

目标为 Atlas A2/A3 的 arch22（DAV_2201），不使用 arch35 的 SIMT 接口。
物理长度为 `1+(n-1)*incx` 个复数元素；正常路径采用64位 GM 偏移。
调用方负责提供足够的存储空间及保证异步执行期间的生命周期。

## 算子实现

### 实现方案

#### Host 侧

1. 按上述合同校验参数，quick return 使用 `aclrtMemsetAsync`。
2. 查询当前平台 AIV 核数，以 `ceil(n/4096)` 限制使用核数，避免小向量启动过多核。
3. 每核逻辑区间向64个复数元素对齐，最后一核处理剩余区间。
4. 单核场景直接写 result；多核场景取得 handle workspace，每核占32字节，
   检查 workspace 非空且大小足够。
5. 同一 stream 先下发扫描 kernel，多核时再下发一次归并 kernel。

Tiling 仅携带 `totalN`、`incx`、`perCoreN`、`useCoreNum`。核数来自平台接口，
不固定写死为某张卡的核数。

#### Kernel 侧

每核初始候选为 `(+Inf, 本核起始索引)`。读取全局首元素检查 NaN；若首元素
为 NaN，各核发布索引0，最终返回1。若首元素两个分量的绝对值位模式均为0，
已经达到模值的全局下界，各核直接发布 `(0,0)`，无须扫描剩余输入。

`incx=1` 且本核不少于8个元素时使用连续布局流水线：

1. 每 tile 最多4096个复数，输入采用深度2的 `TQue`。当前 tile 出队后立即
   发起下一 tile 的 DMA，使 MTE2 与当前向量计算重叠。
2. `DataCopyPad` 只读取真实输入长度；最后32字节块及之后的尾部 lane 补
   +Inf，使归约长度可向64个复数对齐。输入地址允许偏移，无越界读取。
3. 对交错输入执行 Abs，通过 `GatherMask` 的偶/奇模式拆分实、虚平面，
   再执行 Add。自比较及 Select 将 NaN 模替换为 +Inf。
4. 每 tile 的 `ReduceMin(calIndex=true)` 写入一个独立32字节 UB 槽。最多
   收集64个 tile 后再同步 V/S，按顺序标量合并，减少每 tile 的同步等待。
   固定64槽也保证任意 n 的 UB 占用有界。
5. 队列管理 MTE2/V 及 V/MTE2 依赖。下一 tile 的 Abs 后有 V 屏障，在后续
   `GatherMask` 覆写 real/imag 前保证上一 tile 的 ReduceMin 已消费完毕；
   批次合并结束后用 S/V 事件保护结果槽的重用。

`2<=incx<=64` 且本核不少于8个元素时使用跨步 Gather 路径：

1. 每 tile 的逻辑容量为 `floor(4096/incx/64)*64`，使所读取物理跨度不超过
   4096个复数元素。按实际字节长度 `DataCopyPad` 搬入，输入偏移及尾部无需
   32字节对齐，读取不越过最后一个有效复数元素。
2. 预生成 Gather 字节偏移，将交错输入拆成实、虚平面；执行 Abs 和 Add。
3. 未使用的尾部 lane 设为 +Inf；用自比较识别 NaN，并替换为 +Inf。
4. `ReduceMin(calIndex=true)` 输出 tile 最小值及首索引。按 tile 顺序用严格
   小于更新本核候选，保持跨 tile 同值选择最早索引。
5. MTE2/V、V/S、V/MTE2 同步保护 DMA、向量归约、标量读取及 UB 复用。

`incx>64` 或本核不足8个元素时采用标量逻辑扫描，避免为大步长搬入无用间隔。
该路径只读取指定复数元素，使用64位字节/元素地址计算。

单核扫描直接精确写回4字节结果。多核扫描各自写 workspace 的独立32字节槽，
槽内为 float32 最小模和 uint32 逻辑索引。归并 kernel 在同一 stream 的扫描
完成后执行，按 `(模, 索引)` 比较所有槽，写回 `index+1`。没有原子操作或
kernel 内跨核屏障。

### 资源与工程组织

扫描阶段 UB 使用固定容量，随布局选择不同输入缓冲：

| 缓冲 | incx=1 | incx>1 |
|---|---:|---:|
| 输入 | 2×32KiB | 32KiB |
| 实/虚平面 | 2×16KiB | 2×16KiB |
| Gather 字节偏移 | 0 | 16KiB |
| NaN mask | 512B | 512B |
| 归约 scratch | 4KiB | 4KiB |
| 批量 tile 结果槽 | 64×32B | 0 |
| 输出槽 | 32B | 32B |
| 合计 | 104992B | 86560B |

以上为代码请求的 UB 容量，均低于 arch22 的192KiB；队列与事件由 TPipe 管理。
多核 workspace 为 `useCoreNum*32` 字节。API 不新增 Device 分配。

| 路径 | 用途 |
|---|---|
| [blas/iamin/arch22/icamin_host.cpp][host] | 校验、tiling、异步调度 |
| [blas/iamin/arch22/icamin_kernel.cpp][kernel] | 扫描与归并 |
| [blas/iamin/arch22/icamin_kernel.h][kernel-header] | 内部 launch 声明 |
| [blas/iamin/arch22/icamin_tiling_data.h][tiling] | tiling及容量常量 |
| [test/iamin/icamin/arch22/][test-dir] | GTest、CSV、自验说明 |
| [test/iamin/icamin/CMakeLists.txt][gtest-cmake] | arch22 GTest 注册 |

## 支持硬件

| 产品 | 设计目标 | 自验证状态 |
|---|---|---|
| Atlas A2，性能型号 Atlas 800T A2（910B3） | arch22 向量实现 | 已在 910B3 / CANN 9.1.0 实测 |
| Atlas A3 | arch22 实现适配目标 | 尚未实测，不据 A2 结果宣称 A3 已通过 |

任务书§3.1允许自验证覆盖一种款型。本次仅以 910B3 自测结果申明已验证范围。

## 算子约束限制

不修改 arch35 实现或公共 ABI。正常输入为正步长一维向量；无广播、无原地
更新、无额外 leading dimension。小尺寸与大步长路径是功能通路，性能验收
仍按任务给定的200条连续布局用例逐条检查。subnormal 保留、同值首索引和
特殊值行为已在 910B3 做专项验证；A3 仍需相应设备验证。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 判定方法 | 标准来源 |
|---|---|---|
| 精度 | 成功用例输出 INT32 索引与 CPU golden 精确相等；负向用例核对状态码；quick return 核对0 | 任务书§2.1、§3.2；[测试说明][test-readme] |
| 性能 | 有效样本不少于10次，取 msprof 单次 kernel task duration 均值；多 kernel 调用累加；GPU/NPU≥0.8 | 任务书§3.3、§7.4；任务随附[GPU 基线][gpu-baseline] |
| 覆盖 | 保留全部1000条精度与200条性能用例；有效普通随机输入均匀/正态各50%，补充边界和特殊值 | 任务书§3.5；[完整用例 CSV][test-csv] |
| 内存 | 无验收阈值；静态资源与进程观测单独说明，不作通过判据 | 任务书§3.4 |

## 兼容性分析

复用已有 `aclblasIcamin` 声明及状态码，不改变公共 ABI。新实现位于 arch22
分支，arch35 源码保持不变；测试工程仅补齐 arch22 的 GTest 注册。异步行为
遵循 handle 绑定的 stream；调用方继续负责输入、输出及 workspace 的生命周期。

## 精度与覆盖

保留全部原始1000条精度、200条性能 CSV。对403条有效普通均匀随机精度输入
逐一追加正态配对，使有效普通随机精度输入均匀/正态各占50%。实部、虚部
独立生成，正态均值在[-5,5]、标准差在[0.1,2]，不截断尾部。

追加5个具名测试覆盖空handle、偏移、跨tile/core同值、步长空洞、混合NaN、
Inf、FLT_MAX加法溢出、正负零、subnormal、负维度和quick return空result。
精度总计1408个GTest，性能200个，完整运行1608个。负向和quick return仅
分配dummy，输出始终使用Device地址，并用输出哨兵检查写回范围。

CPU golden 是仓库 cblas 风格的严格小于循环；标准Netlib没有icamin接口。
所有成功用例用 `EXPECT_EQ(actual, golden)` 比较整数索引。验证脚本同时检查
进程状态和用例完整性，失败、漏测及缺少基线不能记为通过。

取证模式 `ICAMIN_ACCURACY_LOG=1` 输出逐次 `[ACCURACY]` 记录。实际采集1434条，
由1403条CSV调用和5个固定GTest中的31个子场景组成；总GTest数仍为1408。
记录实际/预期索引与状态码、同步状态、输出guard及偏移，错误返回的索引记NA。
独立构建的 `icamin_evidence_test` 链接原性能算子库；manifest验证原性能二进制、
库及对象文件在取证构建前后与运行后指纹不变，避免补日志影响已采集性能证据。

## 性能采集与证据边界

正式环境为 CANN 9.1.0、Atlas 800T A2（910B3）。每例采用 `msprof op` kernel
task duration，warmup后保留至少10个成功有效调用，取算术平均。一次调用含
扫描和归并时，两阶段耗时必须相加。逐例阈值为
`NPU_avg_us <= gpu_baseline.gpu_ms * 1000 / 0.8`。

GTest另外输出5次warmup、20次event采样的诊断均值，不能替代正式msprof口径。
保存源码commit、硬件/CANN信息、执行命令、完整精度日志、每条性能原始数据
与汇总；报告在实际运行后填写，不预填通过结论。任务书§3.4不设内存指标，
交付模板中的内存材料按实际资源使用与验证证据填写并说明适用范围。

本次容器的 profiler 设备映射需要修复。采集使用独立 msopprof 副本，将官方
`a0eb0c6` 映射修复移植到配套源码 `b4ffdb3d6ba7acdb7bbfcf359f6e4dd5b587a4f4`，
通过包装脚本接受 `msprof op` 命令形式。具体补丁、构建和复现步骤见
[修复补丁][profiler-patch]及[自验证说明][steps]；原 CANN 安装保持不变。

精度逐项日志由独立 `icamin_evidence_test` 链接已完成性能采样的原算子库生成。
[取证清单][provenance]记录构建前后及运行后原性能二进制、库和对象的指纹一致。
[源码审计][audit]记录 kernel 逐字节一致，另3个 host/header 文件仅初始许可证
注释不同、程序正文一致。补日志不改变已采集性能版本。

补充进程观测得到Linux `os.wait4` 最大RSS229156 KiB；23次NPU快照中16次可见
目标进程，最大观测占用209 MB。两者包括测试框架和输入等，非算子独占内存；
NPU为周期采样，不能作为真实峰值或无阈值任务的内存通过判据。

设计文档、[测试代码与 README][test-dir]、[自验步骤][steps]、原始精度/性能/内存报告
及日志集中在 `6156d6c` 的[交付目录][submission-dir]。
`081709d` 的[截图目录][screenshots]保存真实 GitCode 页面截图，展示前述提交的
`accuracy.log`、`performance.csv` 和 `summary.json`，属于归档证据展示，非服务器原执行现场截图。
[截图清单][screenshot-manifest]记录来源 URL、原文件及图片 SHA256，以及工作簿补图检查结果。
新版[精度工作簿][accuracy-report-with-screenshots]和[性能工作簿][performance-report-with-screenshots]
各增加“归档证据截图”页；原工作表、数据及公式缓存保持不变，内存工作簿未修改。
原有[索引预览][numeric-preview]与[固定子场景预览][fixed-preview]仍为真实数据报告的渲染预览。

## 本次实机自测

实测环境为 910B3 / CANN 9.1.0，驱动26.1.1，Ubuntu22.04 / AArch64，
完整版本记录见[环境信息][environment]。

| 自验证项目 | 实测结果 | 固定版本证据 |
|---|---|---|
| 精度 | 1408/1408 个 GTest 通过；1434条调用记录含31个固定子场景 | [报告][accuracy-report]、[原始日志][accuracy-log] |
| 性能 | 200/200项自测达标；每项10次有效调用；最低 GPU/NPU=0.8857595，阈值0.8 | [报告][performance-report]、[日志][performance-log]、[逐例 CSV][performance-csv] |
| 大输入专项 | 3/3通过；含同核跨 tile 和跨64-tile批次；最大 n=16777216 | [专项日志][large-boundaries] |
| 验证脚本 | 16项回归通过 | [脚本测试日志][script-tests] |
| 内存补充观测 | Host 最大RSS229156 KiB；NPU进程最大观测209 MB；不构成内存验收通过结论 | [报告][memory-report]、[记录与局限][memory-log] |

[原始 profiler 归档][raw-profiles]保留逐例采样依据；[服务器材料归档][server-materials]
及[校验清单][server-checksums]保留环境、构建和二进制等材料。

A3 尚未实测。已补充的截图展示归档证据，非服务器原执行现场。
上述为开发者自验证结果，社区设计评审、任务验收及上游合入均尚未完成。


[code-commit]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/commit/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5
[code-dir]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/tree/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/blas/iamin/arch22
[test-dir]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/tree/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/test/iamin/icamin/arch22
[submission-dir]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/tree/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission
[template]: https://gitcode.com/cann/cann-ops-competitions/blob/c5130d91be660a45cb6a67e6c5fa73faaa0fd04a/04_tasks/01_community-task-2026/resources/design_template.md
[host]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/blas/iamin/arch22/icamin_host.cpp
[kernel]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/blas/iamin/arch22/icamin_kernel.cpp
[kernel-header]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/blas/iamin/arch22/icamin_kernel.h
[tiling]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/blas/iamin/arch22/icamin_tiling_data.h
[gtest-cmake]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/test/iamin/icamin/CMakeLists.txt
[test-readme]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/test/iamin/icamin/arch22/README.md
[test-csv]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/test/iamin/icamin/arch22/icamin_test.csv
[gpu-baseline]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/test/iamin/icamin/arch22/gpu_baseline.csv
[steps]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/1%20%E8%87%AA%E9%AA%8C%E8%AF%81%E6%AD%A5%E9%AA%A4%E8%AF%B4%E6%98%8E.md
[accuracy-report]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/2.1%20%E7%B2%BE%E5%BA%A6%E8%87%AA%E9%AA%8C%E8%AF%81%E6%8A%A5%E5%91%8A.xlsx
[accuracy-log]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/2.2%20%E7%B2%BE%E5%BA%A6%E8%87%AA%E9%AA%8C%E8%AF%81%E6%97%A5%E5%BF%97.log
[performance-report]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/3.1%20%E6%80%A7%E8%83%BD%E8%87%AA%E9%AA%8C%E8%AF%81%E6%8A%A5%E5%91%8A.xlsx
[performance-log]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/3.2%20%E6%80%A7%E8%83%BD%E8%87%AA%E9%AA%8C%E8%AF%81%E6%97%A5%E5%BF%97.log
[memory-report]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/4.1%20%E5%86%85%E5%AD%98%E8%87%AA%E9%AA%8C%E8%AF%81%E6%8A%A5%E5%91%8A.xlsx
[memory-log]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/4.2%20%E5%86%85%E5%AD%98%E8%87%AA%E9%AA%8C%E8%AF%81%E6%97%A5%E5%BF%97.log
[environment]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/environment-full.txt
[provenance]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/manifest.json
[audit]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/source-audit.json
[large-boundaries]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/large-boundaries.log
[script-tests]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/script-tests.log
[performance-csv]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/performance.csv
[profiler-patch]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/msopcom-device-fix.patch
[raw-profiles]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/verification_raw.tar.gz
[server-materials]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/server_materials.tar.gz
[server-checksums]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/server_materials_sha256.json
[numeric-preview]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/2.1_numeric_checks.png
[fixed-preview]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/6156d6c09de7d8d0b06e4095168d2a724e7a9bc5/task_submission/evidence/2.1_fixed_checks.png
[screenshot-commit]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/commit/081709d7120b14871d2aa43a0b4fc5cb137cbbbf
[screenshots]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/tree/081709d7120b14871d2aa43a0b4fc5cb137cbbbf/task_submission/evidence/screenshots
[screenshot-manifest]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/081709d7120b14871d2aa43a0b4fc5cb137cbbbf/task_submission/evidence/screenshots/manifest.json
[accuracy-report-with-screenshots]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/081709d7120b14871d2aa43a0b4fc5cb137cbbbf/task_submission/2.1%20%E7%B2%BE%E5%BA%A6%E8%87%AA%E9%AA%8C%E8%AF%81%E6%8A%A5%E5%91%8A.xlsx
[performance-report-with-screenshots]: https://gitcode.com/jueyufish/ops-blas-icamin-a2a3/blob/081709d7120b14871d2aa43a0b4fc5cb137cbbbf/task_submission/3.1%20%E6%80%A7%E8%83%BD%E8%87%AA%E9%AA%8C%E8%AF%81%E6%8A%A5%E5%91%8A.xlsx
