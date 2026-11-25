# aclblasSgemvStridedBatched（Atlas A2/A3）算子设计

## 文档信息

| 项目 | 值 |
| --- | --- |
| 任务 | 11 月社区任务 gemvStridedBatched 类 5 算子开发 |
| 目标 | Atlas A2/A3，CANN 9.1.0，ops-blas |

# 需求背景（required）

## 需求来源

社区任务要求在 Atlas A2/A3 上补齐跨步批量实数 GEMV，沿用 ops-blas 公共声明和 handle 绑定的 stream。工作基线为 ops-blas master 的 `55ef791b77f3677b45da3468b0a8b17941424f84`。该基线已包含五个 arch22 入口。设计覆盖任务包的参数契约、批内并行和验证方法。[社区任务详情](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)

# 需求分析（required）

## 需求描述

第 i 批的 A、x、y 相对各自基地址的元素偏移为 i 乘以相应的 batch stride。A 按列主序存储，物理尺寸为 lda × n，lda 的下界为 max(1,m)。N 的 x/y 长度为 n/m，T 和 C 的 x/y 长度为 m/n。C 对实数与 T 等价。向量内部步长可为负，首个逻辑元素从本批向量占用区间的尾端起算。

| API 前缀 | A/x | y | alpha/beta |
| --- | --- | --- | --- |
| S | FP32 | FP32 | FP32 |
| HSH | FP16 | FP16 | FP32 |
| HSS | FP16 | FP32 | FP32 |
| TST | BF16 | BF16 | FP32 |
| TSS | BF16 | FP32 | FP32 |

## 需求拆解

- 共用五个公开 API 的校验、批间地址计算和分核规则。
- 实现非转置和转置的向量计算，覆盖五种输入输出组合。
- 接入任务 CSV，采集逐用例精度、正式 kernel 耗时及事件计时。

# 详细设计（required）

## 算子分析

对每个批次 `i` 原地计算 `y_i ← alpha · op(A_i) · x_i + beta · y_i`。每个操作数的批间步长以元素为单位，向量步长用于单批内寻址。

矩阵跨度采用 lda × n。长度为 L 的向量占用 1 + (L - 1) × abs(inc) 个元素。多个 batch 的各自首地址以 64 位元素偏移定位。选核结合批数和输出数，批数较少的形状由多个核处理单批内的独立输出。

## 算子实现

### Host 与分核

Host 先校验 handle、枚举、尺寸、lda、向量步长、batch stride、batchCount 和标量指针。空规模通过结构校验后直接返回成功，并允许 A、x、y 为空。活跃批次使用 A 的 lda × n 和向量实际占用跨度计算各自的最小 batch stride。末批地址检查使用除法判断乘法是否可表示，避免有符号偏移及地址计算溢出。空 handle 和非法 trans 返回各自的状态码。

将 y 的工作按批次和批内物理顺序展平，核数取实际 AIV 数与输出分组数的较小值。每个核处理一个连续工作区间。跨核边界向 y 的 32 字节 DMA 数据块起点对齐，相邻批紧挨时同样共享一个边界计算，因此一个 DMA 数据块由同一核处理。负步长在设备端把物理输出区间反向映射为逻辑输出区间。非空常规请求在 handle 的 stream 上下发一个 AIV kernel。

### 核内计算和数据搬运

- N 常规路径使用双缓冲成组搬运多列，使用 Brcb 扩展 x 系数后逐列块乘加归约。单行紧凑矩阵直接计算点积，少行紧凑矩阵成组搬入后用 Gather 提取各行。大批量、紧凑且行数较多的形状扩大行块。
- 归约维较大的 N 用例可按矩阵列分核。各核将 FP32 部分和写入 handle workspace 的独立槽位，同步后按输出区间归并。Host 根据形状、步长及 workspace 的容量和对齐选择分列或输出维分核。
- T/C 路径成组搬运列并复用 x 块。短行归约使用 WholeReduceSum，长行分块后按物理 pitch 使用二维 ReduceSum。
- 非单位步长的输入按跨度选择连续窗口或二维 DMA，再用 Gather 提取逻辑元素并处理反向顺序。跨步输出按物理顺序排列到 UB，二维 DMA 逐段写回逻辑元素。
- 所有变体在片上按 FP32 运算。高倍率路径保存乘积与归约的 FP32 舍入残差。低精度输出在缩放与合并旧 y 之后收窄。

alpha 为零时跳过 A、x 的搬运与归约。beta 为零时覆写 y 的逻辑元素，无需读取旧 y。批间间隙和向量步长跳过的存储位置通过精确长度的 DMA 保留。片上缓冲按有界矩阵 tile 和向量窗口申请。批内分列使用已有的 handle workspace，部分和槽按 AIV 核划分。

## 支持硬件

| 目标 | 编译参数 | 实现目录 |
| --- | --- | --- |
| Atlas A2 910B3 | ascend910b3 | arch22 |
| Atlas A3 | ascend910_93 | arch22 |

## 算子约束限制

阶数和 batchCount 非负。向量步长非零且排除 INT_MIN。batch stride 为正并覆盖该操作数的完整单批跨度。应用保证 y 的写入范围与 A 和 x 的读取范围独立。同一 handle 按绑定 stream 下发，读回前同步 stream。

# 可维可测分析

## 精度标准和性能标准

任务 CSV 通过测试进程的环境变量接入，共五份数据，每份 1,000 条精度与 200 条性能用例。输入按变体量化后构造参考值。逐批以 CBLAS DGEMV 累加，转为 FP32 后按 y 类型收窄，符合任务书 §3.2 允许的 FP64 golden 口径。精度按输出类型的 mixed tolerance 判定，同时逐字节检查 y 的间隙。N(0,1) 按任务包定义及原 seed 生成。

性能验收在 CANN 9.1.0 和 910B3 上执行。每个用例预热五次，有效采样十次。使用 `msprof op` 的 Task Duration 与相同矩阵布局及参数的 A100 基线逐例对比，NPU 平均耗时上限为基线的 1.25 倍。`aclrtEvent` 平均耗时保留在同一结果表中用于辅助核对。

## 兼容性分析

变更位于 arch22 实现和对应测试入口。五个 API 复用现有公开头文件，A2/A3 共用 devkit 的 dav-2201 目标。
