# aclblasSgemmBatched / aclblasHgemmBatched 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/55123ab3caa64d089f525f646155094c)。目标平台：Atlas A2/A3（arch22）。

分别对设备指针数组的各项执行 `C[b] ← alpha·op(A[b])·op(B[b]) + beta·C[b]`。

## 需求分析

三组指针数组在 Device，维度、leading dimension 和转置参数由各 batch 共享。S 使用 FP32 接口，H 使用 FP16 接口；N/T/C 均支持，实数 C 等价 T。各 `C[b]` 的可写区域独立。

### 公共接口

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount);

aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const Carray[], int ldc,
    int batchCount);
```

## 详细设计

### Host

在入口校验 batch 数、尺寸、ld 和指针。矩阵较小时合并多个 batch 到核级任务；矩阵较大时并行处理 `(batch, tileM, tileN)`。无输出的维度走空返回，零归约或零 alpha 走 C 缩放。

### 计算与数据搬运

解析设备指针后按列主序分块读取，不假设各 batch 在同一连续段。双缓冲搬运与计算交替，H 使用 FP32 累加。输出块仅在写回前融合 alpha/beta；`beta=0` 跳过 C 旧值读取，避免不必要的 GM 访问。

### 实现侧重点

将无有效乘积的 batch 走缩放，并按任务书检查空指针要求。用带 NaN 的 A、B 与 C 分别回归标量快路径。

## 可维可测分析

S/H 各 500 条精度、100 条性能。覆盖 0/1/多 batch、九种转置组合、非紧凑 leading dimension、两类标量快路径、地址隔离及内存边界。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 NPU 平均耗时不得超过 GPU 基线的 1.25 倍。事件计时用于快速回归。
