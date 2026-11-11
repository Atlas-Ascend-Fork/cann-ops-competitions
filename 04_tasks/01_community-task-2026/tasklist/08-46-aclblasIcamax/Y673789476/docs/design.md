# 需求背景（required）

## 需求来源
2026 CANN 社区任务：aclblasIcamax A2/A3，任务附件 a65f5dbb923445e18a746550581653f4。贡献者 Y673789476。

## 背景介绍
基于 ops-blas Ascend C kernel 直调框架补齐 arch22 复数最大模索引接口。设计基线为 ops-blas 5bd95145a9c499e43ebad44a313a1029c63d54bc。复用公共声明、handle/stream/workspace 管理及测试框架，参考 arch35 的参数校验语义；不移植其架构专用 SIMD/SIMT 指令。

# 需求分析（required）

## 需求描述
接口保持 aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)。x 为交错 float32 实虚部的 Device complex64 向量，result 为 Device int32 标量。

## 需求拆解
- 输出 1-based 逻辑索引，按 abs(real)+abs(imag) 比较，同模取最小索引。
- handle 空返回 HANDLE_IS_NULLPTR；n<0 或 result 空返回 INVALID_VALUE；随后 n=0 或 incx<=0 在绑定 stream 写零返回成功，不读 x，不启动归约。
- 正常路径 x 空及 int64(n-1)*incx>INT32_MAX 返回 INVALID_VALUE。字节地址和实虚部偏移使用 64 位计算。
- 支持 Atlas A2/A3、CANN 9.1.0；精度与 Netlib BLAS 3.10.0 cblas_icamax 精确一致。

# 详细设计（required）

## 算子分析
### 数学公式
令 s(i)=abs(Re(x[i*incx]))+abs(Im(x[i*incx]))，i=0..n-1。正常有限值结果为最小的 argmax(s)+1。全局首元素的 s 为 NaN 时固定返回 1；其余 NaN 不参与竞争，Inf 并列仍取首个。加法使用 float32，禁止改为欧氏模或提高计算精度改变比较结果。
### 支持数据类型和形状
输入 complex64，一维逻辑长度 n；正步长物理长度 1+(n-1)*incx。输出 int32，索引不转换为 float，避免大于 2^24 时失去精度。不支持广播、负步长遍历及原地输出。

## 算子实现
### Host 侧设计
严格按上述顺序校验。quick return 使用 aclrtMemsetAsync(result,sizeof(int),0,sizeof(int),handle->stream)，错误映射 EXECUTION_FAILED。正常路径以值传递 tiling，包含 n、incx、每核逻辑区间和 tile 大小。查询平台 AIV 数与 UB 容量；小向量单核，大向量连续逻辑分段，按 tile 数限制核数，避免空核。使用 handle 有效 workspace，校验容量，避免每次 malloc/free 和强制同步。workspace 的所有权、复用规则保持现有 handle 契约。

### Kernel 侧设计
实现放在 blas/iamax/arch22/，使用该架构可用的 Ascend C 搬运、Abs、Add、比较与归约接口。连续路径分块搬运交错分量，拆分或按成对通道求绝对值和；步长路径按物理覆盖范围分块并提取有效元素，较大步长采用有界逐元素加载作为通用路径。所有搬运均检查有效尾长，禁止为 DMA 对齐读取用户分配外的数据。

每个 tile 产生 (最大值,最小逻辑索引,有效标志)。先屏蔽 NaN 和填充通道，再求最大值，并对相等通道做整数最小索引归约；跨 tile 和跨核使用同样的比较规则。额外保留全局首元素 NaN 标志，不能让每个分片的首个 NaN 阻断本分片后续有效值。无有效值的分片输出无效标志。

小向量在单核直接写 result；多核先写各核对齐的候选记录，再在同一 stream 发起单核合并 kernel，以 kernel 顺序保证可见性，避免未经验证的跨核自旋同步。两个 kernel 的总耗时共同计入算子性能。workspace 大小随核数线性增长，每核记录按 32 字节对齐；最终写回只覆盖 4 字节结果。缓冲区复用前完成相应流水事件同步。默认单缓冲，基于 UB 预算和实测决定是否启用双缓冲。

### 工程集成
沿用 include/cann_ops_blas.h 的现有声明及公共 complex 类型，不新增平行 API。新增 arch22 的 host/kernel 与必要共享 tiling 定义；不改动 arch35 行为。测试放在 test/iamax/icamax/arch22/，补齐 icamax/CMakeLists.txt 的 arch22 GTest 注册，避免同时构建架构时重复注册。README 标明 A2/A3 支持、构建和测试命令、步长及异常语义。

## 支持硬件
| 产品 | 支持 |
| --- | --- |
| Atlas A2 | 是 |
| Atlas A3 | 是 |

## 算子约束限制
调用方保证输入与结果 Device 指针有效、容量充分且不重叠；在读取结果或复用输入之前同步绑定 stream。性能目标通过实测确认，不作为本设计阶段已完成结论。

# 可维可测分析

## 精度标准/性能标准
| 项目 | 标准与方法 |
| --- | --- |
| 精度 | Netlib 3.10.0 cblas_icamax 的 0-based 输出加 1，与 NPU int32 逐值精确相等；quick return 单独验证 0 |
| 性能 | 910B3、msprof op，5 次 warmup 后至少 10 次成功采样，算术平均 us <= 对应 GPU baseline 的 1.25 倍；多 kernel 耗时按一次调用汇总 |
| 内存 | 任务无额外内存阈值；记录 workspace、设备占用及检查方法，验证无越界与泄漏 |

完整纳入附件 CSV，并补充均匀/正态各占随机输入用例 50%：实虚部独立生成，正态 mu 属于 [-5,5]、sigma 属于 [0.1,2]，固定随机种子并记录参数。覆盖 n=0/1/2/3/7、2 的幂及相邻值、至少 2^20；incx=1/2/3/0/-1/-2/-3/INT32_MIN；空 handle/x/result、负 n、跨度溢出；全零、常量、正负交替、极值、首位/中间 NaN、多个 Inf、跨 tile/核并列、尾块、不同输入地址偏移，以及大索引整数精度。

负向及 quick return 用例仅分配单元素 dummy 或空指针，不按非法跨度分配；调用前初始化输出哨兵，验证返回状态后同步并读回。测试通过 GTest 加载 CSV，保存逐 case 状态、期望索引、实际索引和精确比较结果。性能记录完整原始 profiling、有效样本数、设备型号与版本；非 910B3 数据单独标明，不替代指定设备性能结论。

## 兼容性分析
共享接口不变，新增 arch22 能力，不改变 arch35。设计 PR 独立提交后保持固定，不随实现修改；实现细节、调优记录与实测证据归入代码分支和自测报告。验收包及个人仓 task_submission 按任务模板提供步骤、精度/性能/内存 xlsx 与日志；用户提交验收 zip 后才创建代码 PR。
