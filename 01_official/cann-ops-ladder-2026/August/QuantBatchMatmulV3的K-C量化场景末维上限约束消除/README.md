# QuantBatchMatmulV3的K-C量化场景末维上限约束消除

**难度系数：4.0**

## 算子说明

本赛题面向 `QuantBatchMatmulV3` 的能力扩展：在保持算子语义不变前提下，**消除 `x1/x2` 最后一维 `65535` 上限限制**，并支持 **输入维度为二维 ND** 的 `transposeX1/transposeX2` **全组合**。

**计算过程如下：** 对输入做矩阵乘得到 `x1 @ x2`，再乘 **`scale`**，再乘 **`pertokenScale`**，得到输出 `out`。

$$
out = x1@x2 * scale * pertokenScale
$$

张量命名、类型与维度详见下表。

相似类型算子的实现可参考开源仓：

https://gitcode.com/cann/ops-math

https://gitcode.com/cann/ops-nn

https://gitcode.com/cann/ops-cv

https://gitcode.com/cann/ops-transformer

https://gitcode.com/cann/cann-samples

## 算子规格

|参数名|输入/输出|描述|数据类型|数据格式|维度|
|-|-|-|-|-|-|
|`x1`|输入|左输入张量，参与矩阵乘|`INT8`（重点）|`ND`|`(m, k)` 或 `(k, m)`（由 `transposeX1` 决定）|
|`x2`|输入|右输入张量，参与矩阵乘|`INT8`（重点）|`ND`|`(k, n)` 或 `(n, k)`（由 `transposeX2` 决定）|
|`scale`|输入|与 **channel / `n` 维**对齐的缩放（可与 `n` 同长或长度为 1）|`FLOAT32`|`ND`|`(n,)` 或 `(1,)`|
|`pertokenScale`|输入|与 **token / `m` 维**对齐的缩放|`FLOAT32`|`ND`|`(m,)`|
|`transposeX1`|输入|是否对 `x1` 末两维转置后参与计算|`bool`|-|-|
|`transposeX2`|输入|是否对 `x2` 末两维转置后参与计算|`bool`|-|-|
|`out`|输出|矩阵乘后经 `scale`、`pertokenScale` 链式缩放的结果|`BFLOAT16`|`ND`|`(m, n)`|

### 维度范围

* **维度数量**：仅支持二维 `ND` 输入；
* **关键约束变化**：**不再限制** `x1/x2` 的最后一维大小 `<= 65535`（包括 `k` 或 `n` 作为“最后一维”出现的各类 transpose 组合场景）。

## 任务要求

### **正确性要求**

   * 与cpu标杆相比，误差满足双千分之一：误差超过 0.1% 的用例数量不超过 0.1%；
   * 支持 `transposeX1/transposeX2` 全组合（`false/true` 四种组合）；
   * `k` 或 `n` 等参与“最后一维”的长度超过 `65535` 的场景必须通过。

### **性能要求**

   * 能力扩展后，在常见规模下性能不明显退化；
   * 对大维度（如 `k > 65535`）与 transpose 场景给出可量化性能优化或持平结果。

## 关键技术方向

* **大维度切分与归约**：对 `k/n` 超 `65535` 的轴做分块与分段归约，避免对单轴长度做出固定假设。
* **transpose 组合统一**：将 `transposeX1/transposeX2` 的四组合尽量走统一数据路径，降低分支开销。
* **二维映射与访存复用**：二维输入场景下优化任务划分与地址计算，减少访存开销。
* **流水并行与双缓冲**：搬运与计算重叠，提升大 `k` 下吞吐稳定性。
