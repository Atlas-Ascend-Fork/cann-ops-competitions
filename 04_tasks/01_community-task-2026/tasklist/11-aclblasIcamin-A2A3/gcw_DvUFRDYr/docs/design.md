# 需求背景（required）

## 需求来源

11月社区任务“aclblasIcamin算子开发（A2A3）”。参与账号为
`gcw_DvUFRDYr`（childishfoam），任务包为
`aclblasIcamin_AtlasA2_A3_task_doc.md` 及随包 `test_cases/`。

## 背景介绍

在 ops-blas 已有公开接口及 Ascend C kernel 直调框架中，补齐 Atlas A2/A3
的 complex64 最小1-范数模元素索引归约。复用公共句柄、stream、状态码和
workspace 管理，不另设 API，不修改 arch35 实现。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle, int n,
    const aclblasComplex* x, int incx, int* result);
```

声明已存在于 `include/cann_ops_blas.h`。x 为 Device 上实虚部交错的
complex64，result 为 Device 上一个 INT32。对逻辑元素
`x[i * incx]`（0<=i<n）计算 float32 的
`fabs(real) + fabs(imag)`，返回最小模的最小逻辑索引加1。
不使用欧几里得模，也不返回物理下标。

## 需求拆解

1. 与任务书的参数、quick return、严格小于比较及首元素 NaN 语义对齐。
2. 新建 `blas/iamin/arch22/` 的 Host、Kernel 和共享 tiling 数据。
3. 新建 `test/iamin/icamin/arch22/`，补齐其 CMake GTest 注册。
4. 保留随包全部1000条精度与200条性能用例及GPU基线，补充正态分布、
   自然对齐偏移、分块/分核 tie、特殊值、stream 与缓冲区保护检查。
5. 仅在任务允许的一款 A2/A3 上自验；性能优先使用910B3。
   CANN版本9.1.0。未实测前不填写通过结果。

# 详细设计（required）

## 算子分析

### 数学公式与数值规则

```text
m[i] = float32(abs(real(x[i*incx])) + abs(imag(x[i*incx])))
result = 1 + min { i | m[i] 为按任务书比较规则得到的最小值 }
```

CPU golden 采用随工程提供的 cblas 风格顺序循环：以首元素模为初值，
仅在 `candidate < best` 时更新。标准 Netlib BLAS 没有 icamin 例程，
不虚构 cblas_icamin 调用；Netlib 3.10.0 作为仓内测试依赖按要求保留。
输出采用整数索引精确相等判定，不套用浮点容差。

首元素含 NaN 时结果固定为1；其他 NaN 不更新最小值。Kernel 将普通
归约中的 NaN 模替换为正Inf，分核结果另外保留首元素NaN标记。
全Inf、全NaN、同模及正负零均返回最早逻辑索引。模加法采用float32，
允许有限分量之和溢出为Inf，与CPU golden保持相同舍入语义。

### 参数与状态码

按下列顺序校验，避免负向用例分配或访问大缓冲区：

| 顺序 | 条件 | 行为 |
| --- | --- | --- |
| 1 | handle为空 | HANDLE_IS_NULLPTR |
| 2 | n<0或result为空 | INVALID_VALUE |
| 3 | n=0或incx<1，含INT32_MIN | 在handle stream上aclrtMemsetAsync写result=0，成功返回；不启动计算Kernel，不要求x非空 |
| 4 | 正常计算路径x为空 | INVALID_VALUE |
| 5 | int64计算(n-1)*incx>INT32_MAX | INVALID_VALUE |
| 6 | 多核路径workspace为空或不足 | EXECUTION_FAILED，不下发计算 |

正常路径的全局float偏移及字节计算使用64位，不因复数的2个分量而
在int32中再次乘2。输入仅要求float的4字节自然对齐；搬运使用
DataCopyPad，不对未对齐地址强制float2或32字节对齐加载。

## 算子实现

### Host与Tiling

通过现有 `GetAivCoreCount()` 查询实际AIV核数，不写死950或A2的核数。
每核目标至少4096个复数元素；每核区间按64个逻辑元素对齐，
区间连续且无重叠，尾核裁剪至n。

单位步长每tile最多4096个复数。incx在2至64时按步长缩小逻辑tile，
使物理搬运窗口不超过4096个复数；更大步长每tile64个逻辑元素，
使用64位标量GM读取，不搬运巨大的空洞。首版阈值以真机测量校核，
不改变接口和验收口径。

单核路径直接写结果。多核路径使用handle已有workspace，每核占32字节，
记录一个float32模与一个uint32原始逻辑下标，剩余字节用于隔离写回，
总需求为 `cores * 32` 字节。调用过程中不额外aclrtMalloc、
不Host回读输入或中间结果、不同步stream。

### Kernel

1. 通过depth-2输入队列搬入tile。常规步长按物理窗口DataCopyPad搬运，
   较大步长标量打包填入UB；尾部只搬运真实字节，
   参与对齐归约的无效项填正Inf。
2. 单位步长及已打包数据先对交错实/虚部Abs，再以PairReduceSum相邻
   两项相加；其他步长Gather实/虚部后Abs、Add。两条路径均以一次
   float32加法得到每个复数的模。Compare与Select将非首元素NaN归约值
   规范化为正Inf；ReduceMin同时得到tile最小值和首个局部下标。
3. 每批最多保存32个tile结果，再按模和原始下标合并，保证UB占用
   不随n增长；批间同模保留更小下标。首元素NaN在核0明确保留。
4. 单核直接以DataCopyPad写4字节索引。多核将局部结果写入各自32字节
   槽，随后在同一stream启动一个归并Kernel，批量读槽并Gather模值，
   ReduceMin选择最早获胜核，取该核保存的原始下标加1。

两个Kernel在同一stream顺序执行，避免跨核自旋、全局原子争用和
workspace复用竞态。同模时核区间顺序与原始下标顺序一致。
V/MTE/S之间使用队列或HardEvent建立依赖；主核UB预算不超过约140KiB，
适配A2/A3，不使用arch35 SIMT、VF、DeInterleave特有路径。

### 测试与构建接入

修改 `test/iamin/icamin/CMakeLists.txt`，在arch22与arch35均注册
`ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)`。
公共API与arch35源码、数据和测试不改动。

原始CSV固定1200行，其中精度1000、TC_PF_200；C++与Python分别校验
绝对条数、唯一名称、字段与分类，避免CSV截断后测试集合同步缩水。
原始873条均匀分布精度行分别补一个同参数的正态分布行；另补8对
均匀/正态偏移场景与40条固定语义场景，共929条补充用例。
随机分布精度行的均匀/正态为881:881；有效计算路径的分布也独立审计。
正态均值在[-5,5]、标准差在[0.1,2]，实虚部独立采样并记录参数和种子。
另外测试空handle和同一handle绑定stream下的结果顺序。

负向、quick return和跨度溢出用例仅分配dummy输入；result始终使用
Device缓冲区，禁止把Host栈变量当作Device result。
输入、输出分别增加前后guard，并覆盖额外4字节偏移，逐字节检查
输入只读、输出仅修改一个INT32。对照值使用真实CPU参考循环。

## 支持硬件

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2训练/推理系列，包括Atlas 800T A2（910B3） | 本任务新增arch22实现 |
| Atlas A3训练/推理系列，包括Atlas 800T A3 | 本任务新增arch22实现 |
| Ascend 950系列 | 保持现有arch35实现，本任务不新增实测要求 |

## 算子约束限制

仅支持complex64到Device INT32标量。incx<1为quick return，不做负步长遍历。
正常路径跨度受任务书INT32_MAX限制。无广播、leading dimension或原地输出。
异步返回后，x、result及workspace须存活至绑定stream完成；同一handle的
workspace复用沿其stream有序执行，跨stream并发应使用独立handle。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 本任务实施口径 | 来源 |
| --- | --- | --- |
| 精度 | 全部原始及补充用例，INT32 actual==golden；状态码、quick return、guard分别核对 | §2.4、§3.2、§3.5 |
| 分布 | 原CSV不改，补足正态，均匀/正态成对；特殊值单列 | §3.5 |
| 性能 | 固定200条GPU基线，逐例GPU/NPU>=0.8 | §3.3 |
| 采样 | msprof op Task Duration(us)，5次warmup后至少10个有效算子样本的算术平均；多Kernel单次耗时取各阶段之和 | §3.3、§7.4 |
| 辅助计时 | aclrtEvent数据单列，不能替代任务书的Kernel耗时 | 任务书优先于随包指导中Event主判据的冲突表述 |
| 内存 | §3.4不涉及内存验收；七文件模板中的内存报告和日志如实写“不涉及”，不新增峰值实测任务 | §3.4、§4 |
| 交付 | 设计PR、可复现测试工程、个人验收仓、task_submission七文件及所需证据 | §4 |

报告中的执行状态来自真机原始日志和Profiler记录，不能由用例清单推定通过。
正式交付前必须核实本设计PR实际Merged并记录时间；其后平台验收由参与者
本人提交。代码PR按本任务官方测试通过后的流程申请，不以本地自测代替
官方通知。SSH与Token不进入源码、文档、日志或交付包。

## 兼容性分析

复用公开aclblasIcamin ABI和complex64布局；只补arch22实现、对应GTest注册与
产品支持说明。既有arch35实现及其他接口维持原状，不扩展私有平行接口。
