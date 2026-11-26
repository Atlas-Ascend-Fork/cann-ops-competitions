# aclblasCsyrk3mEx 设计文档

## 需求背景

来源：[官方任务书页面](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6)。目标平台：Ascend 950PR（arch35）。

对指定三角执行复数对称秩-k 更新 `C ← alpha·X·Xᵀ + beta·C`。`OP_T` 与 `OP_C` 等价，操作数不做共轭。

## 需求分析

列主序；A 支持 `ACL_COMPLEX64` 或交错复数 `ACL_INT8`，C 为 `ACL_COMPLEX64`。alpha 和 beta 为复数。只更新 `uplo` 指定三角，对角可具有非零虚部。

### 公共接口

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

## 详细设计

### Host

校验枚举、类型组合和 `lda/ldc`。按 dtype、K 和三角输出块数选择 Cube 块和搬运策略。仅分配覆盖所选三角的任务，斜跨对角的 tile 记录三角边界。

### 计算与数据搬运

加载 `Xr/Xi` 后在同一次 kernel 发射内完成三次实数乘：`P=Xr·Xrᵀ`、`Q=Xi·Xiᵀ`、`R=(Xr+Xi)·(Xr+Xi)ᵀ`。复原实部 `P−Q`、虚部 `R−P−Q`。分块沿 K 累加，INT8 数据的中间求和用足够宽的表示，最后与复数 alpha/beta 融合，按三角掩码写回。

### 实现侧重点

实虚和与原始分量在同一 K 片复用，三个乘积的累加均在一条发射路径内。保留代表性强相消输入验证组合精度。

## 可维可测分析

用任务书参考实现核对实虚分量，覆盖两种输入类型、N/T/C、上下三角、3M 专项、空规模及未选三角不写。精度 1000 例，性能 200 例。

精度按任务书 §3.2 的容差、匹配比例和最大误差联合判定；复数分别核对实、虚分量。内存按 §3.4 检查越界、步长和尾块。

性能使用 `msprof op` 的 kernel Task Duration。预热 5 次不计入，10 次有效样本取算术平均；换算单位后与同用例 GPU kernel-only 基线比较。单例 GPU/NPU ≥ 0.4，全部用例倍率算术平均 ≥ 0.5。事件计时用于快速回归。
