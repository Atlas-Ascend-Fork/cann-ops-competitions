# aclblasCherk3mEx 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/5f0045789d554794aca26d146e93d6b7)。目标平台：Ascend 950PR（arch35）。

对指定三角执行 Hermitian 秩-k 更新 `C ← alpha·X·Xᴴ + beta·C`。`OP_N` 取 A，`OP_C` 取 A 的共轭转置，`OP_T` 按非法枚举处理。

## 需求分析

列主序；A 支持 `ACL_COMPLEX64` 或交错复数 `ACL_INT8`，C 为 `ACL_COMPLEX64`。alpha 和 beta 为实数；只读写 `uplo` 指定一侧，输出对角虚部按任务书置零。

### 公共接口

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const float* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

## 详细设计

### Host

依据类型、K 尺寸及有效三角块数确定 tiling。按输出块分核；内部块与跨对角块使用同一坐标映射，避免依赖用例编号。

### 计算与数据搬运

在同一次 kernel 发射内计算 `P=Xr·Xrᵀ`、`Q=Xi·Xiᵀ`、`R=(Xr+Xi)·(Xr−Xi)ᵀ`。取实部 `P+Q`、虚部 `R−P+Q`；先规范化转置与共轭再执行。按 K 块累加，INT8 的实虚和差使用更宽表示。尾部和对角块按掩码写回。

### 实现侧重点

仅给跨对角块生成写回掩码，完整三角块按矩阵 leading dimension 整段写回；用不对齐的 n 检查边缘。

## 可维可测分析

比较有效三角实部和虚部，覆盖 `OP_N/OP_C`、上下三角、两种输入类型、零归约、特殊值和 3M 专项。精度 1000 例，性能 200 例。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 GPU/NPU ≥ 0.4，全部用例倍率算术平均 ≥ 0.5。事件计时用于快速回归。
