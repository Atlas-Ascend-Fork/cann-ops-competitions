# aclblasCgemm3m 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/cae74cb900484862a2b43b24c0f7d9d3)。目标平台：Atlas A2/A3（arch22）。

用三次实数乘实现 `COMPLEX64` 矩阵乘加 `C ← alpha·op(A)·op(B) + beta·C`。

## 需求分析

列主序，两个转置参数各支持 N/T/C；标量为 Host 内存中的 `COMPLEX64`。C 原地更新。`alpha=0` 或 `k=0` 走缩放；`beta=0` 不读 C 旧值。

### 公共接口

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

## 详细设计

### Host

校验 enum、维度、lda/ldb/ldc 及活跃路径所需的指针。根据是否转置、对齐和 tile 的局部容量决定就地切片或规整输入；按输出 tile 映射核。

### 计算与数据搬运

把 `op(A)`、`op(B)` 的实虚分量统一为 `Ar/Ai`、`Br/Bi`，对 C 转置预先取虚部符号。三项 `P=Ar·Br`、`Q=Ai·Bi`、`R=(Ar+Ai)·(Br+Bi)` 重建实部 `P−Q`、虚部 `R−P−Q`。保留 FP32 中间值，tile 完成后再执行复数标量融合与边缘写回。

### 实现侧重点

按三组累加缓冲总量决定是否启用双缓冲。长 K 优先复用已读取的实虚片和逐片求和结果。

## 可维可测分析

用直接复数参考比较各分量，覆盖九种转置形态、消去误差、零标量、短 K、尾块及非默认标量。精度 1000 例，性能 200 例。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 NPU 平均耗时不得超过 GPU 基线的 1.25 倍。事件计时用于快速回归。
