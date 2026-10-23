# 算子性能优化赛题：QuantBatchMatmulV3的K-C量化场景性能优化

**难度系数：4.0**

## 算子说明

本赛题面向 **K-C 量化**下的二维 **量化 MatMul**（GEMM）。**K-C 量化**（首次出现说明：**pertoken-perchannel**，即左矩阵沿 **token / `m` 维**一组缩放、右矩阵沿 **channel / `n` 维**一组缩放的组合量化模式），以 **INT8** 两端输入与给定 `scale`、`pertokenScale`为载体，按 `(m,k) × (k,n)` 理解与评测。

**计算过程如下：** 对 `INT8` 输入做矩阵乘得到 `x1 @ x2`，再乘 **`scale`**，再乘 **`pertokenScale`**，得到 **`BFLOAT16` 输出**。

$$
out = x1@x2 * scale * pertokenScale
$$

张量命名、类型与维度详见下表。

计算特征为 **小 M/N、大 K**，对访存、流水与并行映射敏感。

相似类型算子的实现可参考开源仓：

https://gitcode.com/cann/ops-math

https://gitcode.com/cann/ops-nn

https://gitcode.com/cann/ops-cv

https://gitcode.com/cann/ops-transformer

https://gitcode.com/cann/cann-samples

## 算子规格

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度 |
| ------ | --------- | ---- | -------- | -------- | ---- |
| `x1` | 输入 | 左矩阵，INT8 量化乘的左端 | `INT8` | `ND` | `(m, k)` |
| `x2` | 输入 | 右矩阵，INT8 量化乘的右端 | `INT8` | `ND` | `(k, n)` |
| `scale` | 输入 | K-C 中与 **channel / `n` 维**对齐的缩放（可与 `n` 同长或长度为 1 再广播） | `FLOAT32` | `ND` | `(n,)` 或 `(1,)` |
| `pertokenScale` | 输入 | K-C 中与 **token / `m` 维**对齐的缩放 | `FLOAT32` | `ND` | `(m,)` |
| `out` | 输出 | 矩阵乘后经 `scale`、`pertokenScale` 链式缩放的结果 | `BFLOAT16` | `ND` | `(m, n)` |

### 维度范围

- `m <= 256`
- `n <= 256`
- `32768 <= k <= 65535`

（即标准 GEMM 的 `m、n、k`。）


## 任务要求

### **正确性要求**
   - 与cpu标杆相比，误差满足双千分之一：误差超过 0.1% 的用例数量不超过 0.1%；

### **性能要求**
   - 重点优化 `m <= 256, n <= 256, 32768 <= k <= 65535`；
   - 相较基线有可量化性能提升；

## 关键技术方向

- **并行分块与任务映射**：小 `M/N`、大 `K` 的 tile 与核间划分。  
- **访存与缓存复用**：降低带宽、提高片上重用。  
- **INT8 乘加与向量化**：减少冗余转换与同步。  
- **scale 与 pertoken 乘性链路融合**：对齐文档数据流 `(x1@x2) → * scale → * pertokenScale`，减少中间写回。  
- **BFLOAT16 输出与舍入**：满足基线语义前提下的高效转换。  
- **流水并行与双缓冲**：搬运与计算重叠。
