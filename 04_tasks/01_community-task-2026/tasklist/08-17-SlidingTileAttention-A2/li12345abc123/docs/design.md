# 需求背景（required）

## 需求来源

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 8月社区任务-SlidingTileAttention算子开发（A2） |
| 任务目录 | 04_tasks/01_community-task-2026/tasklist/08-17-SlidingTileAttention-A2 |
| 作者 | li12345abc123 |
| 联系邮箱 | 2909834810@qq.com |
| 文档版本 | V0.1，2026-10-13，设计评审稿 |
| 目标工程 | ops-transformer/experimental/attention/sliding_tile_attention |
| 接口形式 | 两阶段 aclnn 接口，Ascend C 算子工程 |
| 目标平台 | A2，性能验收设备为 Ascend 910B3；CANN 9.0.0 及以上 |

功能和验收要求来自本任务《SlidingTileAttention算子开发任务书》及随附 slidingTileAttention_testCase 测试包。本文采用当前社区任务仓库的[设计模板](../../../../resources/design_template.md)。

功能参考固定为 FastVideo 提交 4ddcdf541f32b63b5c684016c903658e2e2b6f67，对应[接口实现](https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py)和[滑动窗口计算实现](https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/st_attn_triton.py)。固定参考版本，避免上游更新改变窗口、文本或布局语义。

## 背景介绍

SlidingTileAttention（STA）面向视频生成模型。视频 Token 按三维 Tile 排布，每个 Head 的图像 Query 访问局部视频窗口和有效文本 KV；文本 Query 访问全局有效 KV。其主要收益来自减少图像 Query 对无关视频 Token 的计算。

直接构造整个序列的稠密 Mask 或 Attention Score 会产生随序列长度平方增长的中间数据。本设计将稀疏关系表示为窗口分组和索引，用紧凑的变长 Attention 执行有效计算，保持输出顺序、窗口边界和文本语义一致。

主方案为“相同可见窗口分组 + Q/K/V 打包 + 变长 FlashAttention + 输出逆映射”。完整 Tile、常见 Head Dimension 优先复用仓库现有 Ascend C FlashAttention 计算能力；后续以分块融合路径覆盖底层能力不能直接承载的合法形状，并减少打包开销。

工程职责划分如下，设计文档本身不包含算子源码交付：

| 文件或模块 | 职责 |
| --- | --- |
| op_host/op_api/aclnn_sliding_tile_attention.h、.cpp | 公共两阶段接口、参数校验、执行计划、输出处理 |
| op_host/op_api/sta_packed_attention.h、.cpp | 窗口分组、索引计划、TND 打包及变长 Attention 调度 |
| 仓库现有 FlashAttention L0/Host/Kernel | 主路径 Attention 的 Tiling、Cube/Vector 计算与同步 |
| 后续新增通用 Tiling/Kernel | 部分 Tile、超大 Group、维度分段及直接读取融合路径 |
| selfcheck 与 ATK 适配 | 公式 Golden、边界用例、公共接口调用和原始 ATK 验证 |

# 需求分析（required）

## 需求描述

### 接口与参数

~~~cpp
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    aclTensor *output,
    const aclIntArray *const *windowSize,
    uint64_t windowSizeLen,
    int64_t textLength,
    bool hasText,
    const char *seqShape,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSlidingTileAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
~~~

| 参数 | 语义和校验 |
| --- | --- |
| q、k、v、output | BNSD 四维 Tensor，Shape 与 dtype 完全一致；B、N、S、D 均大于 0；支持 FLOAT16、BFLOAT16；不支持 Tensor 广播 |
| windowSize | 每项包含三个正奇数；windowSizeLen 为 1 时广播属性到所有 Head，为 N 时逐 Head 生效 |
| textLength | hasText 为 true 时表示有效文本 KV 数量，满足 0≤textLength≤S−T×H×W；hasText 为 false 时有效值按 0 处理 |
| seqShape | 正整数三维网格字符串，如 30x48x80；校验格式、整数范围及乘积溢出 |
| output | 独立输出，Shape 与 q 一致；不修改输入 q、k、v |
| workspaceSize、executor | 第一阶段返回工作空间大小和执行计划；第二阶段在指定 Stream 上执行 |

Head 数统一记作 N。任务书 window_size 参数表中的“长度为 H”按每个 Head 的窗口属性解释，避免与空间高度 H 混淆。

### 功能语义的确定

固定参考实现使用 6×8×8 的视频 Tile，完整 Tile 含 384 个 Token。三个随附图像网格均能整除该 Tile。输入已采用 Tile-major 排布，算子不应再次将其当成普通三维栅格顺序重排。

任务书数学描述将窗口写成 Token/Grid 单位再除以 Tile Size，而性能用例包含 (1,1,1)、(3,3,3) 等奇数窗口。结合固定版本代码，这些属性实际表示沿三个方向的 **Tile 数量**。本设计按官方用例和参考实现执行 Tile 窗口语义，不将这些数值再次除以 6、8、8。

令 I=T×H×W，L=hasText ? textLength : 0。图像区间为 [0,I)，有效文本 KV 区间为 [I,I+L)。[I+L,S) 中的尾部容量不作为有效 KV。hasText 为 true 时，参考实现仍对输入中已有的文本尾部 Query 计算全局 Attention，并返回原始长度 S 的输出，不能把这些 Query 的输出统一置零。

hasText 为 false 时 L 为 0，textLength 不影响计算；当 S 大于 I 时，设计将尾部 Query 按全局 Query 规则处理，使其访问图像 KV。该扩展需补充独立边界用例，固定参考代码未覆盖的形状不宣称已与 GPU 完成一致性验证。

STA 不包含 Causal Mask、Dropout、GQA 或 MQA。每个 Batch、Head 独立计算。

## 需求拆解

| 模块 | 工作内容 | 完成判据 |
| --- | --- | --- |
| aclnn 接口 | 参数检查、执行计划、工作空间、Stream 执行及独立输出 | 签名一致，非法参数报错，输入保持不变 |
| 稀疏窗口 | Tile-major 索引、逐 Head 窗口、边界整体平移、超大窗口 | 可见 KV 与参考一致，无重复 Token 或越界 |
| 文本处理 | 有效文本 KV、全局文本 Query、尾部容量 | 填充 KV 不参与 Softmax，原始 S 保持不变 |
| 计算路径 | 分组打包、变长 Attention、逆映射；通用融合补充路径 | 两种 dtype、合法形状满足精度标准 |
| 性能 | 两个任务书场景，覆盖完整执行流程 | 910B3 上按统一计时口径达到任务书要求 |
| 验证 | 原始 ATK、A100 参考、边界及正态分布补充测试 | 保存可复现输入、版本、结果和日志，不修改判据 |

# 详细设计（required）

## 算子分析

### 数学公式

单个 Batch、Head 的输入为 Q、K、V，形状均为 S×D。Token i 的可见 KV 集合记作 \(\mathcal K(i)\)：

\[
O_i=
\frac{\sum_{j\in\mathcal K(i)}
\exp(Q_iK_j^\mathsf T/\sqrt D)\,V_j}
{\sum_{j\in\mathcal K(i)}
\exp(Q_iK_j^\mathsf T/\sqrt D)}.
\]

图像 Query 所属 Tile 坐标为 \((q_t,q_h,q_w)\)。完整 Tile 网格为 \((G_t,G_h,G_w)=(T/6,H/8,W/8)\)，部分 Tile 扩展使用对应向上取整值。对一个方向，网格长度为 G、窗口为正奇数 w：

\[
m=\min(w,G),\qquad
a=
\begin{cases}
0,&w\ge G,\\
\min(\max(q-\lfloor w/2\rfloor,0),G-w),&w<G.
\end{cases}
\]

该方向访问 Tile 区间 [a,a+m)。三个方向的笛卡尔积构成局部窗口。边界窗口整体移入合法区域；窗口超过网格时覆盖该方向全部 Tile。

图像 Query 的集合为局部窗口内有效视频 Token 与 [I,I+L) 的并集；尾部 Query 的集合为 [0,I+L)。Softmax 只在有效集合上归一化。

完整 Tile 下，Tile 编号和 Token 索引为：

\[
u=(q_tG_h+q_h)G_w+q_w,\qquad
i=384u+r,\quad 0\le r<384.
\]

部分 Tile 的设计使用每个 Tile 的有效 Token 数和前缀和定位，不能继续使用 384u 作为起点。Tile 内部使用 t、h、w 顺序，并只排列有效坐标。

### 支持数据类型

| 数据项 | 类型 |
| --- | --- |
| 输入与最终输出 | FLOAT16、BFLOAT16，与输入 dtype 一致 |
| Token 索引、长度和偏移 | 按底层接口要求使用整数；Host 采用带溢出校验的 64 位整数计算 |
| Softmax 最大值、指数和、累计状态 | FP32 |
| QK 点积及 PV 累加 | 优先使用支持 FP32 累加的路径；乘法和转换模式通过两种 dtype 的精度验证 |

对齐 Head Dimension 时仅补零，Attention Scale 保持 1/sqrt(D)，不能替换为对齐后维度的 Scale。

### 支持形状

对外目标遵守任务书的正维度 BNSD 合同，不把测试包的枚举当作接口范围。随附测试主要覆盖 D 为 16、32、64，以及 30x48x80、36x48x48、18x48x80 三个网格。

完整 Tile 且 D 不大于 128 是当前原型的已实现分派范围。任意正网格、部分 Tile、D 大于 128 等任务书允许的输入属于本设计的扩展范围，需在交付前完成实现和验证。资源不足按实际运行限制报错，不通过截断输入或改变语义处理。

## 算子实现

### 实现方案

仅在同一 Batch、Head 内，将可见 KV 列表相同的图像 Tile 合为一个 Group。将 Group 内 Query 按确定顺序拼接，KV 由其可见视频 Token 和有效文本 Token 组成；每个 Token 在该 Group 的 KV 列表中只出现一次。文本尾部 Query 单独构成全局 Group。

用变长 Attention 对各 Group 独立计算，再通过逆置换恢复原始 BNSD 顺序。同一 Group 的 Query 具有相同可见集合，拼接不会让不同 Query 的 Softmax 相互混合。每个 Query 只归属一个 Group，输出逆映射一一对应。不同 Batch、Head 即使窗口坐标相同，也不能共享数值 KV。

~~~mermaid
flowchart LR
    A["BNSD Q/K/V"] --> B["连续化与 D 尾部补零"]
    C["网格 / 窗口 / 文本属性"] --> D["窗口分组与索引计划"]
    B --> E["Gather 打包"]
    D --> E
    E --> F["TND 变长 FlashAttention"]
    F --> G["分段输出拼接"]
    G --> H["逆索引恢复顺序"]
    H --> I["裁剪 D 并写入独立 BNSD 输出"]
~~~

#### 3.2.1 host侧设计：

##### 1. 参数检查与执行计划

第一阶段检查指针有效性、四维 Shape/dtype 一致性、正维度、seqShape 格式及乘积溢出、I≤S、windowSize 项数和奇数窗口，以及有效文本长度。不符合要求时使用工程现有 aclnn 状态码和校验机制返回错误，不增加自定义错误码。

索引、累计序列长度和中间 Tensor 生命周期纳入 executor。第二阶段提交已生成的计划，调用方按第一阶段返回大小提供 workspace；任务在指定 Stream 上执行，不在算子内部进行设备级全局同步。

嵌套 windowSize 属性必须完整参与计划复用条件。当前原型使用不缓存的第一阶段计划；后续只有缓存键完整包含网格、窗口、有效文本长度、Shape 和 dtype 等属性后才启用缓存，防止复用错误窗口。

##### 2. 分组索引与布局转换

每个 Batch、Head 内遍历 Tile，用窗口起点三元组作为分组键。窗口长度在该 Head 内固定，起点相同即对应相同可见视频集合。记录：

- qIndex：打包 Query 到原始 Token 的映射。
- kvIndex：打包 KV 到原始 Token 的映射，K 与 V 共用索引。
- inverse：原始 Query 到打包输出的逆映射。
- qEnd/kvEnd：各 Group 的累计 Query/KV 长度，二者分别维护。

输入连续化后转换为以 Token 为行的视图，利用 Gather 打包为 [packedTokens,1,Dp] 的 TND 表示。Dp 是底层路径支持的对齐维度。索引作为计划常量传到设备，其构造、传输成本必须记录。各调用独立管理数据生命周期，不修改调用方 Tensor 的 Shape、Storage 或内容。

##### 3. 分核与数据分块策略

Group 是逻辑独立序列，不等同于一个 AI Core。主路径由复用的 FlashAttention Tiling 将 Group 内 Query 分块映射到可用核心；每个 Query 块扫描该 Group 的 KV 块，保留同一 Softmax 状态。

按 Group 边界拆分底层调用，使单次变长 Attention 长度处于可承载范围内。当前原型采用约一百万打包 Token 的拆分预算，它是内部容量参数，不是公开 S 上限。实际核心数量和片上资源由平台能力查询，不写死某张卡的核数。

调度可按 Q_group×KV_group×Dp 估算计算量，避免仅按 Token 数均分造成长窗口任务集中。单个 Group 已超出预算时，不能拆成多个独立 KV Softmax 后简单拼接，必须采用保持在线归一化状态的融合路径或经过验证的等价机制。

##### 4. 工作空间与内存开销

令 R=B×N×S 为 Query 总数，P 为各 Group 打包 KV 数之和，G 为 Group 数，FP16/BF16 元素字节数为 2。主要逻辑容量如下：

| 中间数据 | 容量估算 |
| --- | --- |
| 打包 Query 和打包结果 | 约 2×R×Dp×2 字节 |
| 打包 K 和 V | 约 2×P×Dp×2 字节 |
| Query、KV、逆置换索引 | INT64 情况下约 8×(2R+P) 字节 |
| 两套累计长度 | INT64 情况下约 16G 字节，另含固定元数据 |
| 其他 | 连续化、补零、分段拼接、输出恢复及底层 FlashAttention 临时空间 |

这些为容量分析，实际峰值受生命周期复用、对齐和底层临时空间影响，以 executor 返回的 workspace 和实测为准。实现不构造全局 S×S Score/Mask。

不同 Group 的窗口可能重叠，打包方案会复制重叠 KV，不能描述为“无额外 Device 拷贝”。当前原型先打包整个计划，再拆分 FlashAttention 调用；拆分调用本身尚不能降低完整打包缓冲区的峰值。

后续按 Group 批次打包和释放临时数据，将活动 KV 限制在预算内；超大 Group 使用直接从原始 KV 分块读取的融合路径。这是实现计划，不能写为已完成的内存测试结论。任务书 3.4 未规定定量内存验收指标。

##### 5. 路径与 Tiling 选择

| 路径 | 选择条件 | 方案 |
| --- | --- | --- |
| Packed 路径 | 完整 Tile、底层支持的 dtype/Head Dimension/长度 | 分组、Gather、仓库 L0 FlashAttention、输出恢复 |
| 部分 Tile 路径 | T/H/W 不能整除 6/8/8 | 按有效 Token 前缀和生成索引和长度，不引入无效 Token |
| 通用融合路径 | 单 Group 容量超限、D 超出底层能力或打包成本过高 | Query/KV/D 分块、直接读取、在线 Softmax |

分派根据 Shape、窗口、有效文本和平台能力决定，不识别官方用例编号或随机种子。Packed 路径复用现有 FlashAttention TilingKey；新增融合 Kernel 根据 dtype、尾块和维度分段选择内部 TilingKey，不修改公共参数规避不支持输入。

#### 3.2.2 kernel侧设计：

##### 1. 主计算路径

当前主路径组合仓库的布局处理、Gather、变长 FlashAttention 和输出恢复算子，其中 Attention 使用仓库已有 Ascend C 实现。L0 接口、TND 和累计序列长度能力以对应源码和运行版本为准。

每个 Query 块依次进行 Cube QK 计算、Vector Scale/Softmax 更新、Cube PV 计算和 Vector 输出累计。各 Group 由变长序列边界隔离，不跨 Group 访问 KV。完成后逆映射恢复 Query 顺序、裁去 D 补零尾部并写入用户 output。

##### 2. 在线 Softmax 与通用维度处理

每个 Query 行维护 FP32 最大值 m、指数和 l、未归一化输出 z。当前块 Score 为 s，块最大值为 \(m_b\)：

\[
m'=\max(m,m_b),\quad
\alpha=\exp(m-m'),\quad
p=\exp(s-m'),
\]
\[
l'=\alpha l+\sum p,\qquad
z'=\alpha z+pV,\qquad
O=z/l.
\]

初始 m=−∞、l=0、z=0，首个有效块直接建立状态；跳过空块，避免对空块计算无定义的 −∞−(−∞)。所有输出 D 分块必须按同一 alpha 缩放，不能仅更新当前 D 分段。

通用融合设计在 D 方向分块累加 QK，一个 KV 块的完整点积完成后再执行 Softmax。PV 按输出 D 分段计算，但共享同一 m、l。状态较大时保存到受控 GM 临时区，不对各 D 分段独立归一化。该路径用于覆盖正 D，目前仍需实现和验证，不能以增加公开 D≤128 限制代替。

##### 3. 局部存储与流水

分别为 Q、K/V、Score、Softmax 状态和输出分块分配资源。Query Block、KV Block、D Block 根据实际 UB/L1/L0 容量、dtype、尾块及双缓冲需求选择，不能只统计主 Tensor 而忽略临时状态。

CopyIn/Compute/CopyOut 流水与 Cube、Vector、数据搬运之间的事件同步配套。KV 双缓冲仅在资源校验通过时启用。不同 Query 块的输出位置不重叠；同一 Query 的多个 KV 块保持累计状态依赖。

部分 Tile、Query/KV 尾块和 D 尾块显式处理有效范围。不存在的 KV 不进入紧凑列表，或在归一化前被有效 Mask 排除；不能仅把其 V 清零而仍计入 Softmax 分母。

##### 4. 性能优化方向

首先避免完整 Attention 和稠密 Mask，再分析整个算子的实际耗时。相同窗口分组减少独立调用，使边界处共享可见集合的 Query 合并计算。小窗口、低 D 场景重点检查索引、Gather、恢复顺序开销；大窗口重点控制 KV 重复和流水，必要时融合直接读取。

安全复用属性计划，分别报告第一阶段和设备时间。FP16、BF16 独立验证，不能用一种 dtype 的结果替代另一种。

## 支持硬件

| 平台 | 范围 |
| --- | --- |
| A2 / Ascend 910B 系列 | 算子功能目标，对应 DAV_2201、arch22 |
| Ascend 910B3 | 任务书指定性能验收设备 |
| GPU A100 | 固定 FastVideo 参考和 ATK 对比，不运行 NPU 实现 |

记录驱动、固件、芯片、CANN、算子库、ATK、PyTorch/CUDA 和参考代码版本。A100 参考使用该设备支持的实现，不能将仅面向其他 GPU 架构的二进制作为基线。

## 算子约束限制

1. q、k、v、output 为相同四维 BNSD Shape 和 dtype，不支持 Tensor 广播或不同 Q/K/V Head 数。
2. windowSize 为一项广播属性或 N 项逐 Head 属性，各维为正奇数。
3. 输入为视频 Tile-major 后接文本容量，不自动将 raster-major 解释为 Tile-major。
4. 不增加 Causal、Dropout、GQA、MQA 等未定义行为。
5. 文本填充 KV 不参与计算；有效文本数量和尾部容量分别处理。
6. 部分 Tile、通用 D、超大 Group 尚需扩展验证，不能把原型限制写成任务书限制。
7. 任务书随机网格比固定 FastVideo 支持范围更广。非参考直接支持的网格使用独立公式 Golden，说明验证依据，不宣称已通过原始 A100 路径。

# 可维可测分析

## 精度标准/性能标准

### 精度判据

| dtype | atol | rtol | matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FLOAT16 | 2^-9 | 2^-9 | ≥0.99 | max(0.1, 32×2^-10)=0.1 |
| BFLOAT16 | 2^-6 | 2^-6 | ≥0.99 | max(1, 32×2^-7)=1 |

有限值元素按 abs(actual−reference)≤atol+rtol×abs(reference) 判定匹配；整例同时满足匹配率和最大绝对误差门槛才通过。按照[生态精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)，任务书的“A or 32×ULP”取二者较大值，ULP 取 dtype 在 1.0 处的单位精度，因此这两种 dtype 的上限分别为 0.1 和 1。非有限输出单独记录并判定失败，不让 NaN 在比较中被遗漏。

原始 ATK 判据同时保留。随附 JSON 另含 high_precision 比较配置，不能仅检查任务书混合容差后宣称 ATK 已通过。两套判据都检查和留存，不修改官方脚本、输入分布或阈值。

### 性能目标和计时口径

| 场景 | B/N/D | 图像网格 | 文本 | 逐 Head 窗口 | A100 任务书数据 |
| --- | --- | --- | --- | --- | --- |
| 混合窗口 | 1/8/64 | 30×48×80 | 128 | (1,1,1)、(3,3,3)、(3,5,5)、(5,5,5)、(1,3,3)、(3,3,5)、(5,5,7)、(7,7,7) | 9 Calls；总耗时 64.204 ms；平均 7.134 ms |
| 相同窗口 | 1/8/64 | 36×48×48 | 无 | 所有 Head 为 (3,3,3) | 8 Calls；总耗时 11.691 ms；平均 1.461 ms |

“0.8倍性能”可能指吞吐或耗时。按吞吐不低于 GPU 的 0.8 倍解释，总耗时上限分别为 80.255 ms、14.61375 ms；按耗时不大于 GPU 的 0.8 倍解释，分别为 51.3632 ms、9.3528 ms。本文采用后者作为较严格的开发目标，正式阈值以官方口径为准，不将该解释写成已确认的验收结论。

比较完整一次 forward 的等价工作量。A100 数据同时含多次调用总和与平均值，不能将融合后的全部 Head 耗时与 GPU 单 Head 平均耗时直接比较。

采集覆盖打包、Attention、拼接、逆映射和输出写入，另报告端到端 aclnn 时间及第一阶段成本。预热后同步测量，保留多次测量和原始记录，排除并行测试干扰。不只统计内部 FlashAttention Kernel，以免隐藏 STA 布局处理成本。两个场景均测试 FP16、BF16。

### 测试设计与执行计划

| 分类 | 覆盖内容 | 验证方式 |
| --- | --- | --- |
| 官方用例 | 原始 50 个 ATK 用例，25 个 FP16、25 个 BF16 | A100 固定参考与 A2 对比，保留 JSON、标准、报告 |
| 任务性能 | 上表两种配置、两种 dtype | 910B3 完整流程 Profiling 和统一口径比较 |
| 窗口 | 广播、逐 Head、角点、中心、窗口大于网格 | 对照显式集合，检查平移和去重 |
| 文本 | textLength=0、有效文本、填充、hasText=false | 无效 KV 不影响结果，全局 Query 与公式一致 |
| 布局 | B>1、N>1、连续和非连续 Tensor、恢复顺序 | 无跨 Batch/Head 混用，输出顺序正确 |
| 形状边界 | 小网格、部分 Tile、D 尾块和 D>128 | 独立公式 Golden，验证通用路径 |
| 随机补充 | Q/K/V 为 N(0,1)，合法奇数窗口、任务书网格范围 | 固定输入和种子，记录匹配率和最大误差 |
| 非法参数 | 空指针、维数/dtype/Shape 不一致、窗口非法、字符串非法、溢出 | 返回失败，不启动越界计算 |

官方验证按随附 README 的多机流程：部署 A100 参考 Server、配置 A2 节点、加载原始 all_sta.json，再由执行插件调用两阶段 aclnn 接口。测试包 Python 插件尚需连接 NPU 接口；适配层只将网格枚举映射为字符串、窗口三元组映射为 aclIntArray 列表，不改变计算规则或验收阈值。

README 引用场景 YAML，而当前包已经提供 all_sta.json，应直接保留和使用该 JSON，不因缺少生成配置而生成另一套“官方”用例。逐 Head 混合窗口需要额外性能配置，不能由属性广播的精度用例代替。

保存原始输入、Golden、输出、各判据明细、版本和日志。后续性能采集尽量对同一输入保存输出，与已验证 Golden 对比；输入、语义或路径覆盖改变时才重新生成对应参考。摘要截图对应实际日志，不能用计划值或局部自验替代正式通过结果。

本版本为设计评审稿。已实现的 Packed 原型范围和计划扩展范围在正文分别说明；本文不声明完成 A100/ATK 正式精度验收或达到性能门槛，正式结论由完成验证后的实际日志和报告给出。

## 兼容性分析

新增 aclnnSlidingTileAttention 接口和独立算子目录，不修改已有 FlashAttention 的公开数值语义、其他算子接口或原始验收资产。主路径使用仓库已有 L0 能力，新增路径通过明确分派接入；失败时返回实际错误，不回退为改变窗口语义的完整 Attention。

最低目标版本为 CANN 9.0.0。当前原型使用 CANN 9.1.0 的能力需要逐项核对最低版本支持情况，不能仅凭 9.1.0 编译成功宣称 9.0.0 已兼容。声明构建依赖，使用实际运行库，不将链接 Stub 作为运行库，不替换系统 CANN 库。

保持输入输出 dtype、任务书属性格式和两阶段约定；内部 TND 打包、D 对齐和分组拆分对调用方透明。后续文档更新同步反映已实现范围、实际验证版本和正式测量结果。
