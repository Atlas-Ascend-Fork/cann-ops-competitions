# Sgemm/Cgemm/Hgemm（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/4b45f7b048224317a2ec62eab1e8df52)要求补齐 arch22 通用矩阵乘三种精度。沿用 `aclblasSgemm`、`aclblasCgemm`，按任务书新增非批量 `aclblasHgemm` 声明。

## 需求分析

C ← α·op(A)op(B)+βC。A/B/C 均使用列主序与可填充的 lda/ldb/ldc；逻辑输出 m×n、归约 k。支持 N/T/C，复数 C 分支同时共轭，实数 C 等价 T。S 为 FP32，C 为 interleaved complex64，H 为 FP16；H 的累加及 α/β 运算在 FP32 完成，存回时一次 FP16 舍入。矩阵与标量类型按各公开接口一致。

## 详细设计

### Host 与 ABI

三接口共享参数骨架 handle、transa/transb、m/n/k、alpha、A/lda、B/ldb、beta、C/ldc。检查句柄、枚举、非负维度、物理行数要求、Host 标量指针以及实际使用的矩阵指针。m/n=0 免读 A/B/C；k=0 或 α=0 只缩放 C，允许 A/B 为空；β=0 不读旧 C。用 64 位计算最大索引。H 在 `include/cann_ops_blas.h` 添加声明并接入目标架构编译，S/C 复用原 ABI；发射使用 handle 绑定的 stream。

### Kernel、精度与优化

按 m/n 划输出 tile，每块单核独占，用 k 面板累加。装载器统一转换 N/T/C 索引；连续列段使用 DMA，padding 只参与寻址，尾段以掩码处理。H 走原生半精度输入、FP32 累加矩阵通路；S 以足精度 FP32 路径为基准，若引入 Cube 分量分解，需按全量 FP32 阈值验收后再选择；不直接按 FP16 截断输入。C 使用四路实数积 (ArBr、AiBi、ArBi、AiBr)，以 FP32 按符号累加并在片上重组；共轭在输入装载完成。α/β 缩放与写回融合为单一 epilogue。K 流水对面板搬运双缓冲，双缓冲超容量时缩小面板。小形状走低启动成本路径，CPU 只计算 tiling，不调度逐 tile kernel。优先避免 split-K 的全量部分和和额外发射。

## 可维可测分析

实现置于 `blas/gemm/arch22/`，扩展 `test/gemm/arch22/` 内现有用例。各精度使用任务书附件和 CBLAS/FP32 golden；C 按实虚分别比较，S/C 采用 FP32 档，H 采用 FP16 档，匹配率≥0.99 且满足任务书最大误差门槛。覆盖九种转置组合、非标准 leading dimension、0 维、k=0、任意 α/β 和填充哨兵。按附件分层的 200 条性能数据用 `msprof op` 取热身后 10 次有效 kernel 均值，目标 NPU≤1.25×GPU；保留端到端数据作为迭代参考。当前仅提交方案，精度与性能待实测。
