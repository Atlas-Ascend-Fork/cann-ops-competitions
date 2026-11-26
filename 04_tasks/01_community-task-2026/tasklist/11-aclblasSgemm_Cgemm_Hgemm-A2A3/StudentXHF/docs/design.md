# Sgemm、Cgemm、Hgemm 的 Atlas A2/A3 设计

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/4b45f7b048224317a2ec62eab1e8df52)要求在 Atlas A2/A3 上交付三种列主序 GEMM。基线 `6df9aa04` 的 `ops-blas` 中已有 arch22 的 Sgemm、Cgemm 和 `aclblasHgemm` 公共声明。沿用现有实现和接口，补齐 Hgemm 的 arch22 路径，并对三个接口运行本批任务的统一验证。

## 需求分析

`aclblasSgemm`、`aclblasCgemm`、`aclblasHgemm` 均按 `C = α·op(A)·op(B) + β·C` 原地更新 C。参数顺序为 `handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc`。S、C、H 分别存储 FP32、交错复数 FP32、IEEE binary16，H 执行 FP32 累加并在写回前舍入为 binary16。N、T、C 三种操作均支持，复数 C 为共轭转置，实数 C 与 T 等价。A、B、C 均按给定的前导维度寻址，只写 C 的逻辑 `m×n` 区域。

## 详细设计

### Host 校验与调度

1. 校验顺序依次为 handle、转置枚举、非负维度、前导维度、Host 标量指针及实际访问的 Device 指针。前导维度下界为 `max(1, transa==N ? m : k)`、`max(1, transb==N ? k : n)` 和 `max(1, m)`。非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法尺寸或指针返回 `ACLBLAS_STATUS_INVALID_VALUE`，空 handle 使用专用状态码。
2. 结构参数合法后，m 或 n 为零时直接返回成功。k 为零或 α 为零时，仅对 C 执行 β 缩放。β 为零时写结果而无需加载 C 的旧值。Host 将半精度标量解码为 FP32 后按值传给 kernel。
3. 复用现有 stream 和 workspace 机制。用 64 位计算矩阵元素偏移、工作区大小和分块数量，在乘加越界或工作区不足时借助已有状态码报错。工作区随 Handle 管理，分区使用同一 stream 上的执行顺序。

### 算法与 tiling

- **Sgemm** 复用 arch22 的 AIC Matmul 和整网格实数 GEMM 后端。主路径以 FP32 计算，紧凑矩阵按输出块直写，带填充或尾块的形状用批量 DMA 完成输入打包与结果写回。小规模及大标量场景采用向量累加路径，避免中间结果提前舍入。
- **Cgemm** 按转置方式和前导维解包实部、虚部，分别计算四个实数乘积并应用复数 α、β。常用标量且输入紧凑的窄矩阵或长 K 矩阵复用 CgemmEx，将两个交错复数矩阵展开为一次实数 GEMM，并恢复交错输出。长 K 路径在 FP32 中规约分片后裁掉补齐部分。
- **Hgemm** 以内部行主序计算 `Cᵀ = op(B)ᵀ·op(A)ᵀ`，使用 FP16 输入和 FP32 累加。读取路径按前导维、转置和对齐选择直读或 DMA 打包，对尾块补零。长 K 和较小输出网格可将归约分给多个 AIC，再由 AIV 合并。一般路径在 UB 中应用 α、β，以精确长度写回 FP16 输出。对齐且 α=1、β=0 时可直接写入 C。
- 默认按输出块保持数据归属，避免不同核写同一 C 元素。依据 AIC/AIV 核数和片上空间限制选择分块。极大形状在工作区受限时分批处理输出块，偏移和紧邻块写回通过边界用例核验。

### 任务书 §2.5 约束核对

| 约束 | 设计结论 |
| --- | --- |
| 参数合法性 | Host 按上述顺序校验结构参数，张量元素由调用方提供。 |
| 非连续 Tensor | 支持由 lda、ldb、ldc 描述的列间填充。任意 stride 视图超出接口范围。 |
| 广播 | 三个操作数均按给定矩阵形状计算，行列无广播。 |
| 动态形状 | 运行时根据 m、n、k 和转置模式计算分块与下发参数。 |
| 原地与视图 | 结果写回 C 原缓冲区，β 为零时跳过旧 C 读取。调用方保证 C 与 A、B 的逻辑区域不重叠。 |
| 确定性 | 归约顺序由实现及形状决定，精度以 §3.2 的混合容差和最大绝对误差判据为准。 |
| 空 Tensor | m 或 n 为零时允许 A/B/C 为空，k 或 α 为零时允许 A/B 为空。 |
| 0 维 Tensor | 参数为矩阵间接指针或 Host 标量指针。 |
| 异步执行 | 使用 handle 绑定的 stream，读取输出前由调用方同步。 |

## 可维可测分析

- 在 Atlas 800T A2（910B3）和 CANN 9.1.0 下编译 arch22。通过同一个 `test/gemm/arch22` 测试目标按类型运行官方用例，保留原始 CSV。
- S、C 按任务书 §3.2 以 CPU CBLAS 执行 FP64 参考计算并收窄到 FP32，分别记录复数实部和虚部。H 以量化后的 FP16 输入执行 FP64 累加，经 FP32 舍入至 FP16。运行 1000 条官方精度用例，覆盖九种转置组合、前导维填充、非默认标量、错误码、空矩阵和 FP16 特殊值。逐例记录精度判据和 padding 核验。
- 性能覆盖 200 条基线，使用 `msprof op` 记录每次调用的 kernel task duration。预热 5 次，统计 10 次有效采样。多 kernel 路径计入单次调用的完整 device 核耗时。同步保存端到端事件时间、原始日志、GPU/NPU 耗时和每条比值，正式判据为 GPU/NPU ≥0.8。
- 性能数据按转置、维度、前导维和标量匹配官方 `gpu_baseline.csv`。FP32、复数和 FP16 的精度结果按各自类型阈值统计。内存项按任务书 §3.4 记录适用情况。
