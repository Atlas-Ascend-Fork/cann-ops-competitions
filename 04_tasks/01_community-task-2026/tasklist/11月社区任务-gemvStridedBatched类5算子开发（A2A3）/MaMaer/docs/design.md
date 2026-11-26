# gemvStridedBatched 类五个接口 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)。目标平台：Atlas A2/A3（arch22）。

对五种类型变体执行 `y[b] ← alpha·op(A[b])·x[b] + beta·y[b]`。相邻 batch 由显式的元素偏移分隔。

## 需求分析

`strideA/stridex/stridey` 为 64 位元素间距，必须为正且覆盖各自所在 batch 的占用范围。`incx/incy` 是 batch 内向量步长，支持负值且不得为零。实数 `OP_C` 等价 `OP_T`。

### 公共接口

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

## 详细设计

### Host

用 64 位计算 `lda*n`、向量占用和 batch 偏移，拒绝非法跨步与重叠区域。按地址对齐、间距是否紧凑、有效输出长度和批量大小确定 tile。

### 计算与数据搬运

每个核用基址加 `batch*stride` 定位数据。N 路径合并连续行搬运，T/C 路径以列点积组织片上归约。紧凑、对齐的 batch 可批量搬运；有间隙或稀疏步长时只读取实际数据范围，按输出类型写回。

### 实现侧重点

按 batch 规模和 `stride` 紧凑程度拆分采样，核对优化只是减少无效访问，结果对任意合法间距保持一致。

## 可维可测分析

五种变体各 1000 条精度、200 条性能。除常规转置和标量外，独立覆盖紧凑/带间隙布局、负元素步长、重叠和非正 batch 间距，检查填充不被覆盖。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 NPU 平均耗时不得超过 GPU 基线的 1.25 倍。事件计时用于快速回归。
