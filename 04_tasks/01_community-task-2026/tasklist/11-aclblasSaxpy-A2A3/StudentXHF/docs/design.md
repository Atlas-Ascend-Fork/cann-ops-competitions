# aclblasSaxpy（Atlas A2/A3）设计方案

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/6ffdb3f781a044588935d6c7d33c3b96)要求在 `ops-blas` 新增 Atlas A2/A3 的 FP32 SAXPY 实现，补齐 BLAS Level-1 支持。复用既有 `aclblasSaxpy` API 和 handle 的 stream；新增 `blas/axpy/arch22` 实现和 `test/saxpy/arch22` 自验，不修改 arch35 实现或公共 ABI。

## 需求分析

```cpp
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha,
                            float* x, int incx, float* y, int incy);
```

对逻辑下标 `i ∈ [0,n)`，计算 `y[py(i)] ← alpha × x[px(i)] + y[py(i)]`。步长 `inc>0` 时物理下标为 `i×inc`，负步长为 `(n−1−i)×abs(inc)`；各向量物理长度为 `1+(n−1)×abs(inc)`。只更新 y 的选中元素，间隔位置和 x 不改写；不涉及广播、图模式或额外用户 workspace。

## 详细设计

### Host 校验与调度

1. handle 判空并返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；其后 `n<=0` 直接成功，不读取指针或提交 kernel。
2. 正长度校验 alpha/x/y 非空、步长非零且不为 INT32_MIN；先扩为 int64，再取绝对值并计算 `(n−1)×abs(inc)`，超过 INT32_MAX 返回 `ACLBLAS_STATUS_INVALID_VALUE`。不按溢出的跨度分配测试缓冲。
3. 从 Host 读取 alpha；±0 直接成功，不访问 x/y。其他标量按 FP32 运算，包括 NaN/Inf。通过已绑定的 stream 异步直调 AIV，Host 不强制同步。
4. 获取设备 AIV 数，按至少一个 32 字节向量块划分连续逻辑区间，尽量均衡核间工作。小输入减少启用核数；参数结构按值下发，不分配 GM 临时空间。

### Device 数据路径

- **连续步长 (1,1)**：每核按 UB 容量分块，分别读入 x、旧 y，执行向量乘标量、向量加，再写回同一 y 地址。UB 使用双缓冲以重叠搬运与计算；尾块采用字节长度搬运，写回只覆盖有效元素。核间无归约或同步。
- **其他步长**：按 BLAS 逻辑下标独立定位 x/y 的物理地址；小块分批将有效元素搬到对齐的 UB 槽位，向量运算后按原位置写回，不对间隔元素做读改写。常见同向步长可用成组搬运，方向不同或跨度不适合单条搬运时以精确长度分次搬运。不采用 arch35 SIMT 依赖。
- 初版先确认连续路径与混合步长正确性；再根据 910B3 实测带宽调节连续 tile、核数和双缓冲。任务包的 200 条性能基线均为连续步长。

### 任务书 §2.5 约束核对

| 约束项 | 设计结论 |
| --- | --- |
| 参数合法性与校验层级 | 先校验 handle，再处理 `n<=0`。正长度下，alpha、x 或 y 为空，步长为零或 `INT32_MIN`，或最大元素偏移 `(n−1)×abs(inc)` 超过 `INT32_MAX`，均返回 `ACLBLAS_STATUS_INVALID_VALUE`。alpha 的 FP32 值不作合法性限制。 |
| 非连续 Tensor | 支持 x、y 独立的正负步长。负步长从各自的物理跨度末端起算，写回时保留 y 的间隔元素。 |
| 广播 | 不涉及，x、y 都参与 n 个逻辑元素的逐元素运算。 |
| 动态形状 | 不要求图模式动态形状。Host 按运行时传入的 n 计算 tiling。 |
| 原地与视图 | y 复用原缓冲区写回，x 只读，无视图输出。 |
| 确定性 | 无跨核归约，每个逻辑元素由固定运算顺序独立计算。同一设备重复执行时，以相同的初始 x、y、alpha 和步长获得一致结果。 |
| 空 Tensor | handle 有效时，`n=0` 直接返回成功，不读取向量或下发 kernel。 |
| 0 维 Tensor | 不涉及，x、y 是一维 Device 向量，标量由 Host 指针 alpha 提供。 |
| 异步执行 | Host 读取 alpha 后，经 handle 绑定的 stream 异步下发。调用方读取输出前需用 `aclrtSynchronizeStream` 或等价机制等待完成。 |

若最大元素偏移超过 `INT32_MAX`，调用方可在各子向量偏移合法的前提下分段调用。分段时应保持每段的 BLAS 下标映射，负步长按该段的最低物理地址传入指针。

## 可维可测分析

| 项目 | 自验证计划 |
| --- | --- |
| 硬件与软件 | Atlas 800T A2（910B3）、CANN 9.1.0；A3 为兼容目标，按任务书只要求 910B3 自验。 |
| 精度 | 官方 1000 条精度用例，并补正态填充。覆盖 n<=0、零 alpha、负步长、尾块、NaN/Inf、空指针与跨度溢出；检查间隔元素不变。按任务书 §3.2 记录 FP32 mixed tolerance 及非有限值。 |
| 性能 | 逐条匹配官方 200 条 GPU 基线。用 `msprof op` 采集 910B3 kernel task duration，5 次预热后至少 10 次有效采样；每条 NPU 均值 ≤ GPU 基线 1.25 倍，统一换算为微秒。Event 计时单独保留用于定位，不混入验收统计。 |
| 交付 | arch22 GTest + CSV、复现步骤、精度与性能报告及日志。内存项按任务书标注不涉及。 |

本 PR 仅提交设计。编译、精度与性能结果将以 910B3 实测记录为准。
