# aclblasChpr2 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/7a1275ee6b654459957fc27c8dde32b0)。目标平台：Atlas A2/A3（arch22）。

在压缩三角上原地计算 `A ← A + alpha·x·yᴴ + conj(alpha)·y·xᴴ`，并把更新后的对角虚部置零。

## 需求分析

`AP` 为 `n(n+1)/2` 个 `COMPLEX64` 元素。只读写 `uplo` 指定的一侧。`x` 和 `y` 分别使用 `incx` 和 `incy`，支持负步长；零步长与 `INT_MIN` 按任务书报错。

### 公共接口

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* AP);
```

## 详细设计

### Host

按任务书处理空规模、零标量、步长及空指针。依据实际 packed 元素数分配核，而不是平分列数；三角列长度变化时用 64 位前缀偏移切分工作。

### 计算与数据搬运

每个核独占连续的一段 AP，通过 packed 首尾偏移定位列和列内行范围。分片加载 x/y，按当前列广播系数，向量化计算两项复数外积并批量回写。单独处理对角落点与跨列尾部，所有写入都在所选三角内。

### 实现侧重点

步长为单位时直接搬运 x/y 片；步长较大时将逻辑区间收拢后再广播列向量，控制稀疏读放大。

## 可维可测分析

比较 packed 输出实虚分量，单列核对对角虚部、未使用填充及零标量语义。覆盖上下三角、长短列边界、负步长和非法步长。精度 1000 例，性能 200 例。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 NPU 平均耗时不得超过 GPU 基线的 1.25 倍。事件计时用于快速回归。
