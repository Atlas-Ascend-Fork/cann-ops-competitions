# 需求背景

## 需求来源

[11月社区任务 — aclblasIsamin 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/b8d4f23bc41c4c58b64414e16dcf9e34)，贡献者 Student_ZZX。

## 背景介绍

ops-blas 已在 `blas/iamin/arch35/` 提供 950PR 实现。本设计在 arch22 补齐同一公开接口，使用 Ascend C kernel 直调，供 Atlas A2/A3 调用。自验环境为 A2／910B3、CANN 9.1.0。

# 需求分析

复用 `include/cann_ops_blas.h` 的函数声明：

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n,
    const float* x, int incx, int* result);
```

| 参数 | 位置和类型 | 含义 |
| --- | --- | --- |
| handle | Host，库句柄 | 提供设备上下文及执行 stream |
| n | Host，int | 逻辑向量长度 |
| x | Device，FP32 ND | 读取下标 `i * incx` 的元素 |
| incx | Host，int | 正步长以元素为单位 |
| result | Device，INT32[1] | 写最早的最小绝对值位置，索引从 1 开始 |

从第一个元素的绝对值初始化候选。首元素为 NaN 时，以 FLT_MAX 和索引 1 初始化；后续 NaN 跳过，只有绝对值严格更小时更新。因此全 NaN 返回 1，首元素为 NaN 且其余只含 Inf 或 FLT_MAX 时同样保留 1。首元素为 Inf 时，后续任何有限值均可胜出。±0 等值，按位置决定先后。

# 详细设计

## Host 校验与 tiling

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。n 为负数或 result 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. n 为 0 或 incx 小于 1 时，在句柄 stream 上用 `aclrtMemsetAsync` 写一个 INT32 的 0。该路径允许 x 为空，跳过输入和计算 kernel。
3. 计算路径校验 x，使用 64 位乘法计算最后有效元素偏移 `(n - 1) * incx`，大于 INT32_MAX 时返回 INVALID_VALUE。元素偏移乘元素字节数仍以 64 位计算。
4. tiling 按值传递逻辑长度、步长和启用核数。短向量由单 AIV 执行，避免跨核等待。长向量用当前设备可用的 AIV 核分配连续逻辑区间，优先均衡切分并维持连续路径的 32B 起点对齐。
5. 多核使用句柄 workspace 存放按 32B 分隔的候选值和整数位置，Host 在下发前校验指针和容量。提交和写回均由同一 stream 排序。Host 路径无逐次分配或同步。

## Kernel

- 连续路径按 UB 容量分块读取。整块用对齐搬运，尾部按实际字节数搬运。对数据取绝对值后，把 NaN 排除出有限最小值搜索。块内使用支持返回首个最小值索引的 Ascend C 归约，归约只覆盖有效元素。全 Inf/NaN 块不会替换全局首元素候选。实际尾块、NaN 掩码与硬件索引编码由 910B3 定向用例验证。
- 跨步路径按有界跨度读取并在 UB 选取逻辑元素；当步长大到单次跨度无法装入 UB 时，用有界逐元素搬运。任何路径均不将 stride 间隔元素计入比较，最后一个有效元素后不读取额外字节。
- 每核按绝对值和 1-based 逻辑索引比较候选。各核将结果写入独占对齐槽，全部参与核完成同步后，由核 0 对候选再次归约。工作区的索引使用 INT32，不将大索引转换为浮点参与比较。最终只写 result 的 4 字节。
- 连续长向量的主要开销是 GM 读取。在 A2 上用 1M、2M、4M 锚点定位带宽与同步开销，再调整分块和核数。小尺寸优先降低单次 kernel 延时。

## 支持硬件与约束核对

| 任务书 §2.5 | 设计 |
| --- | --- |
| 参数和异常 | 按上述顺序校验；result 的空指针检查覆盖快速返回 |
| 非连续 | 正步长选择有效元素；负步长及零步长快速返回 |
| 广播、0 维 Tensor | 一维向量 API，无广播或 0 维输入 |
| 动态 shape | host tiling 读取运行时 n/incx；无需图模式支持 |
| 原地与视图 | x 只读，输出为独立 Device INT32 |
| 确定性 | 值相等时取最小的逻辑索引，核数及分块不改变结果 |
| 空 Tensor | n 为 0 只在 stream 上写 0 |
| 异步 | 调用返回表示下发完成，调用方同步后读取；同一 stream 上连续提交保持顺序 |

适配范围为 Atlas A2/A3 的 arch22。950PR 保留已有 arch35 路径。跨度上限和 workspace 需求由 host 守卫；跨步大于连续快路径可承载跨度时走有界搬运。

# 可维可测分析

| 项目 | 验证方式 |
| --- | --- |
| 构建与发现 | A2 上运行 `bash build.sh --soc=ascend910b3 --ops=isamin --device=0`（按实际空闲卡替换设备号）。arch22 使用仓内参数化 GTest。默认列表以最后一个斜杠后的用例名与原始 CSV 逐一比对，覆盖 1000 个精度和 200 个性能用例 |
| 精度 | CPU golden 与 Device INT32 精确相等。保留原 CSV，补正态分布、空句柄、空输出、合法大步长、乘积溢出、±0、Inf/NaN、跨块并列和同 stream 连续调用。负向用例使用单元素 Device 占位或空指针 |
| 性能 | 910B3 上按用例独立采集 msprof op 原始 task duration，排除 5 次预热，至少取 10 次有效样本。200 条逐条对照官方 A100 `gpu_ms`（换算为 µs），门槛 `NPU_us <= 1.25 * GPU_us`，报告保留 GPU/NPU、原始采样和完整日志。GTest Event 数据单独记录，便于复现随包脚本 |
| 交付 | 保存命令、CANN/SoC、提交 SHA、二进制和库标识。按任务书提交设计、可访问代码 commit 和自验证材料。内存指标按任务书 §3.4 标注不涉及 |

主任务书 §2.4–2.5 与当前 arch35 实现要求快速返回路径也校验 result，因此沿用该行为。随包 README 对空 result 和 Event 判定的描述有出入；报告分别保留任务书 §3.3 的 msprof 判定和随包脚本口径。此 PR 供设计评审，编译和 910B3 结果将在实现后提供。

AI 辅助说明：使用 OpenAI Codex（GPT-6），根据任务书、ops-blas 既有实现及官方设计模板辅助整理方案与测试计划。
