# 需求背景（required）

## 需求来源

9月社区任务-aclsparseSpSV算子开发（A2/A3）。目标仓库为
`cann/ops-sparse`，公开接口为 `include/cann_ops_sparse.h` 中的
`aclsparseSpSV_createDescr/destroyDescr/bufferSize/analysis/solve/updateMatrix`。

## 背景介绍

SpSV 求解稀疏三角系统 `op(A)Y = alpha X`。现有上游实现位于
`sparse/spsv/arch35/`，依赖 Ascend 950 SIMT，不能直接编译用于 A2/A3。
本任务新增 DAV_2201 的 Ascend C 实现，并保持现有公开 ABI 和 arch35
接口行为可维护。

# 需求分析（required）

## 需求描述

支持 FP32 与 complex64，CSR、CSC、COO、SLICED_ELL 四种稀疏格式，
I32 Device 索引，base 0/1，LOWER/UPPER，UNIT/NON_UNIT，以及 N/T/H。
complex64 的 H 操作对矩阵 values 共轭。支持 Host/Device alpha、X/Y
同 Device 地址原地求解、未排序及重复坐标、确定性执行和调用 stream。

BufferSize/Analysis 接受 NULL vecX/vecY 描述符；Solve 要求向量描述符、
尺寸、类型及 Device values 有效。Analysis 绑定矩阵、可选向量、参数、
handle/stream、pointer mode 和外部 workspace；改变这些对象或参数后
必须重新 Analysis。GENERAL 更新完整 values，DIAGONAL 更新每行对角值，
两种更新均复用已分析的 pattern。

## 需求拆解

1. 新增 `sparse/spsv/arch22/` Host、Tiling 和 Kernel。
2. 将公共参数约束及生命周期检查放在 SpSV 公共头文件中，架构后端独立。
3. 在 NPU 中完成结构校验、格式规范化、转置/共轭、更新与三角求解。
4. 添加 `test/spsv/arch22/` C++ UT/ST 和官方脚本所需的 PyTorch 适配层。
5. 按任务包脚本执行精度、性能、内存验收，保留原始结果及 Profiler 证据。

# 详细设计（required）

## 算子分析

### 数学公式

有效下三角系统按 `r=0..m-1` 前代，有效上三角按 `r=m-1..0` 回代：

`Y[r] = (alpha*X[r] - sum(Aeff[r,c]*Y[c])) / diagonal[r]`。

求和仅包含求解方向中已完成的依赖行。UNIT 的分母固定为 1，忽略存储的
对角值。NON_UNIT 对角由全部重复对角项按规范遍历次序相加；缺失或为零
时按 IEEE 浮点语义传播 INF/NAN。目标三角之外的存储项忽略。

### 支持数据类型

`ACL_FLOAT` 与 `ACL_COMPLEX64` 的 A/X/Y/computeType 必须一致。
complex64 存储为连续 FP32 实部、虚部。Device 索引为 I32；Host 尺寸
元数据保留公开接口的 int64_t，并对 I32 上限与字节数溢出进行检查。

### 支持形状

A 为动态 `m x m` 方阵，X/Y 为长度 m 的连续向量。支持 m/nnz 为 0/1、
空行、长行、非对齐尾块、未排序坐标及重复坐标，不以测试编号或固定规模分流。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

Host 验证枚举、尺寸、类型、连续性、Device 指针、workspace 对齐和生命周期。
BufferSize 只进行元数据和安全字节数计算，不读取输入数据。

Analysis 在调用 stream 提交 NPU 规范化 Kernel。Kernel 校验 row offsets、
坐标范围、SELL slice 结构，构建 0-based 的有效 CSR。对转置路径交换坐标，
对 H 同时共轭 values。Host 仅回读固定大小的校验状态，以便立即返回明确错误；
不把稀疏索引或 values 回读到 Host，不进行 CPU 求解。该结构校验阶段有一次
stream 同步，Solve 和 UpdateMatrix 的计算下发保持异步。

描述符保存 Analysis 快照、workspace 地址与大小、规范化布局、有效 nnz、
当前 values 引用，以及格式、dtype、op、fill/diag、base 和 stream 信息。
Analysis 的可选向量为空时，首次 Solve 绑定向量；非空时保持对象与 values
指针一致。数据所有权仍归调用者；workspace 在异步操作完成前保持有效。

GENERAL 根据规范化 CSR 中的原始位置映射读取 newValues，必要时共轭。
DIAGONAL 在固定 pattern 内将重复对角项归并成一个新对角值，其他重复对角
位置置零；缺失对角不会通过更新新增稀疏坐标。

Workspace 采用单份规范化 CSR，包含运行时状态、row offsets、column indices、
values、原始位置映射及 m 个临时 scatter 计数。各段按 64 字节对齐，首地址
要求 512 字节对齐。申请大小上界为：

`512 + align64(4*(m+1)) + align64(4*nnz) + align64(valueBytes*nnz)`
`+ align64(4*nnz) + align64(4*m)`。

complex64 的 P-03 对应 workspace 约 65 MiB，最终值以 BufferSize 查询为准。
核数及 UB/L2 容量从实际平台获取，不按型号硬编码。m=0 不启动 Kernel；
nnz=0 走 UNIT 缩放或 NON_UNIT 奇异传播路径。

#### 3.2.2 kernel侧设计：

规范化采用单核确定性三遍遍历：统计有效目标行 nnz、前缀和、按源存储遍历
次序 scatter。CSR/CSC 遍历各压缩段，COO 遍历原始位置，SELL 按描述符的
slice 高度与列槽布局遍历。每项先校验再寻址；非法结构设置状态并停止。
避免原子 scatter 的不确定次序，也避免第二份完整转置副本。

首版求解由一个 Vector Core 顺序执行依赖行，使用 UB 中的块缓存读取 row
offsets、列索引、values、X 与已完成 Y。缓存未命中通过 DataCopyPad 搬入，
写缓存按块回写；所有 UB 起始位置和 DMA 长度满足对齐要求。尾块只搬运有效
元素。借助 MTE2→Scalar、Scalar→MTE3 同步避免读未完成搬运或提前覆盖。
依赖结果从同一 Y 缓存读取，回写及失效顺序明确，不使用跨核忙等。

FP32 使用补偿累加减小长行误差；complex64 的实部和虚部分别补偿。复数除法
使用按较大分量缩放的稳定公式，并对零、INF/NAN 做显式分支。每行遍历与舍入
顺序固定，重复执行获得 bitwise deterministic 结果。

原地求解仅在写回当前行前读取该行 X。求解方向之外的未来 RHS 不被覆盖；
X 读取缓存与 Y 写缓存的顺序保持该约束。Device alpha 在 Kernel 内读取，
遵循用户 stream 上先前操作的顺序。

GENERAL 和 DIAGONAL 更新只操作规范化 workspace values，A/X 保持只读。
对常规缩放、values 更新及复制等无依赖工作，可按实际 AIV 数量做连续分块；
主求解是否增加分层并行由实测结果决定，相关设计和测试同步更新。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 / Ascend910B3 | √，主开发及优先性能自验证 |
| Atlas A2 / Ascend910B4 | √，独立验收 |
| Atlas A3 / 环境实际提供的 Ascend910_93 型号 | √，独立验收 |

CANN 使用 9.1.0 及后续配套版本。三个目标均使用 arch22/DAV_2201；
实际 SOC、核数、L2、CANN/驱动版本分别记录。

## 算子约束限制

仅 DEFAULT 算法；A 为方阵，Device 索引为 I32，X/Y 为连续向量。
不支持部分重叠但地址不同的 X/Y，支持完全同址。GENERAL values 与原 pattern
等长，DIAGONAL values 为 m 个元素。指针 API 无法获知任意裸指针的逻辑长度，
调用者须提供声明的有效存储范围。

不额外持有与输入规模同量级的 Host 中间结果。输入 buffer 和描述符不可在
异步完成前释放；裸指针释放后重新分配到相同地址的情况不承诺被生命周期
快照识别。初版 Analysis 的结构状态回读有一次同步，文档和测试明确记录。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度 | FP64/complex128 CPU Golden；FP32 及复数各分量 rtol=2^-10、atol=2^-16，匹配率≥0.99，逐元素误差≤max(1e-2,32×ULP) | 任务书及官方精度脚本 |
| 确定性 | 四格式、N/T/H、Host/Device alpha、更新和原地场景重复执行 bitwise 一致 | 任务书及 C++ UT/ST |
| 性能 | 每个场景 GPU Event median_us / NPU 同调用范围耗时≥0.25；预热10次、采样30次，报告中位数/P90、Analysis、Solve及 workspace | 任务包性能脚本与完整逐 case GPU 基线 |
| 内存 | 按附带脚本：输入输出>500MB 时相对 GPU allocated 峰值增加≤5%，或显式 workspace≤实际 L2 | compare_sparse_ops_memory.py；口径冲突按脚本执行 |
| 执行位置 | Profiler 展示 Host、NPU Kernel、workspace 和 handle stream | 任务书 |

C++ UT/ST 覆盖四格式、dtype、base、fill/diag、N/T/H、pointer mode、NULL 可选
向量、GENERAL/DIAGONAL、空/奇异、长行、重复/未排序、异常索引、workspace、
生命周期和原地。官方 200 条交付精度、1000 条泛化及206条性能/内存逐型号执行。
计时范围采用官方三阶段真实 hook，包含脚本规定的更新与求解，不以单 Kernel
耗时替代同调用范围总耗时。官方脚本与资产保持原样。

## 兼容性分析

公开接口原型及 enum 不变。新增 arch22 后端与公共 Host 检查辅助，保留 arch35
实现；共享描述符增加的内部状态不暴露为 ABI。CMake 根据 SOC 只编译对应
架构，防止同名公共符号重复定义。A2/B3、A2/B4、A3 分别构建并验收，arch35
涉及公共逻辑的功能回归在 950 环境执行后记录，不能以 A2 结果代替。
