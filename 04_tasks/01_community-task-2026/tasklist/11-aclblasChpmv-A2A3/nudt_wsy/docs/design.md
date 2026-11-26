# aclblasChpmv 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/c596a05d90b9471683e41fcf4d026ae5)。目标平台：Atlas A2/A3（arch22）。

计算 `y = alpha·A·x + beta·y`，矩阵 `A` 为 Hermitian，只提供列主序压缩三角 `AP`。

## 需求分析

`AP` 长度为 `n(n+1)/2`，`uplo` 选择唯一存储区。输出为原地更新的 `y`。对角只读取实部，非存储侧通过共轭恢复。`incx`、`incy` 可为负且不得为零，负步长的逻辑首地址由长度和步长计算。

### 公共接口

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

## 详细设计

### Host

校验 handle、三角枚举、维度和步长，按任务书顺序处理空向量和必需指针。用 64 位计算 packed 偏移。依据 `n`、步长和向量核数选择短向量单核或输出分段方案。

### 计算与数据搬运

每个向量核写一段互不重叠的 `y`。按列分块加载 AP，存储边的连续片段用批量搬运，另一侧按 packed 坐标映射并取共轭。`x` 连续时直接分段搬运；稀疏步长用小块重排，复数乘加在片上完成。归约后统一做 alpha/beta 融合，按实际步长写回；`beta=0` 时不取 `y` 旧值。

### 实现侧重点

优先完善 `alpha=0` 和 `beta=0` 的数据访问边界。用无效旧值验证实际路径只读取参与运算的向量。

## 可维可测分析

用任务书参考实现比较输出实部和虚部，覆盖上下三角、非零对角虚部、单元素、负步长、非单位步长、标量快路径及填充区。精度 1000 例，性能 200 例。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 NPU 平均耗时不得超过 GPU 基线的 1.25 倍。事件计时用于快速回归。
