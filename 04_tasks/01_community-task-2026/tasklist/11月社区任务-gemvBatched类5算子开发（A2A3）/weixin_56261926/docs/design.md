# gemvBatched 类五个接口 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)。目标平台：Atlas A2/A3（arch22）。

对每个 batch 执行 `y[b] ← alpha·op(A[b])·x[b] + beta·y[b]`。任务覆盖 S、HSH、HSS、TST、TSS 五种类型组合。

## 需求分析

使用设备指针数组描述 A、x、y。物理矩阵为列主序 `lda×n`，转置只改变读法。公共维度、步长和 batch 数保持一致，数据类型按五个原型分别执行。负元素步长调整逻辑起点；各输出向量区域互不重叠。

### 公共接口

```cpp
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float *const Aarray[], int lda,
    const float *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSTgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
```

## 详细设计

### Host

校验公共形状、枚举、步长和数组指针。依据 batch 数与输出长度决定按 batch 粒度合并任务还是把单个 batch 的输出轴分块；只填写计算需要的 tiling 参数。

### 计算与数据搬运

在设备侧读取各 batch 的指针。N 路径按列读入连续行片段并广播 `x`；T/C 路径按列累加点积，并向量化处理 K 轴。小尺寸每核可处理多个 batch，大尺寸增加输出维并行。混合精度路径以 FP32 累加并在最终写回时转换目标类型。

### 实现侧重点

对转置的列点积保留 FP32 部分和，必要时对极长归约按任务片合并。对比混合精度写回前后的误差。

## 可维可测分析

五种变体各 1000 条精度、200 条性能。逐 batch 检查输出，覆盖 N/T/C、非单位步长、空维度、两类标量快路径及指针数组边界。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 NPU 平均耗时不得超过 GPU 基线的 1.25 倍。事件计时用于快速回归。
