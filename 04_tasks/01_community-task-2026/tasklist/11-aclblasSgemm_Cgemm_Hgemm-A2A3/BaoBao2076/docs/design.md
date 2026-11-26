# aclblasSgemm / aclblasCgemm / aclblasHgemm 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/4b45f7b048224317a2ec62eab1e8df52)。目标平台：Atlas A2/A3（arch22）。

实现列主序 `C ← alpha·op(A)·op(B) + beta·C` 三种接口。S 为 FP32、H 为 FP16，C 为复数 FP32。

## 需求分析

S/H 的 `OP_C` 等价 `OP_T`；C 路径执行共轭转置。标量由 Host 传入，类型与各接口一致。Cgemm 使用直接复数乘法，不走 3M。`alpha=0` 或 `k=0` 走缩放；`beta=0` 不读取 C 旧值。

### 公共接口

```cpp
aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc);

aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);

aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* A, int lda,
    const aclblasHalf* B, int ldb,
    const aclblasHalf* beta,
    aclblasHalf* C, int ldc);
```

## 详细设计

### Host

复用 arch22 同族的接口校验与分块结构，按转置确定 leading dimension 的最小合法值。按矩阵形状、片上容量和类型选择小矩阵或 Cube 路径。每个输出 tile 由单核独占。

### 计算与数据搬运

S/H 路径分块搬运和累加，H 以 FP32 累加再按接口写回。复数路径拆分输入实虚平面，按 `ArBr−AiBi` 和 `ArBi+AiBr` 累加，处理共轭后做复数 alpha/beta 融合。对窄边和尾块做掩码写回。

### 实现侧重点

在转置加载处决定直接读取或片内重排，常见对齐路径使用批量搬运。用带 padding 的 leading dimension 核对访存区。

## 可维可测分析

按任务书参考实现比较三个算子。重点覆盖九种转置组合、leading dimension 填充、空矩阵、半精度舍入及 NaN 不读取约束。精度 1000 例、性能 200 例，按三算子和规模分层。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 NPU 平均耗时不得超过 GPU 基线的 1.25 倍。事件计时用于快速回归。
