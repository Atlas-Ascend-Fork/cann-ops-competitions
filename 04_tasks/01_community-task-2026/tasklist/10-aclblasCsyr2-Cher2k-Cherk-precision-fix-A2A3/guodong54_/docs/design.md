# Csyr2 / Cher2k / Cherk 精度修复设计（A2/A3）

## 需求背景

修复三个算子 arch22 实现中对未操作三角及前导维 padding 的写入问题。对齐后的整块写回，即使只对非操作区原子加零，也可能改变 NaN payload 或负零的位模式，因此需要裁剪实际写回范围。

本任务沿用 `include/cann_ops_blas.h` 中的公开接口，不新增 API，不修改 arch35 实现。

## 算子功能

| 算子 | 计算功能 |
|---|---|
| Csyr2 | `A = alpha*x*y^T + alpha*y*x^T + A`，支持正、负非零步长 |
| Cher2k | `trans=N` 时为 `C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C`；`trans=C` 时改为相应的共轭转置乘积 |
| Cherk | `trans=N` 时为 `C = alpha*A*A^H + beta*C`；`trans=C` 时为 `C = alpha*A^H*A + beta*C` |

矩阵采用列主序，元素为 complex64。Csyr2 的 alpha 为 Host 复数标量；Cher2k 的 alpha、beta 和 Cherk 的实数 alpha、beta 沿用原 Device 标量约定。只更新 uplo 指定三角；Cher2k、Cherk 在执行更新时将对角虚部置零，合法 quick return 保持原行为。

## 实现方案

### Host 与分块

保留参数校验、返回码、stream 及步长语义，根据逻辑尺寸、布局、块容量和可用核数选择计算路径。workspace 用于输入重排及中间乘积，不用于备份并恢复被覆盖的另一半区。

### Kernel 与写回

- **Csyr2**：按列确定上、下三角有效范围。下三角对角块跳过各列前缀，非对齐短段在 UB 内整理后精确搬出；对齐主体继续批量搬运。原子累加只作用于指定半区。
- **Cher2k**：保留 Cube 乘积计算与后处理。完整位于操作区的块批量写回，跨对角块逐列裁剪，尾块按逻辑尺寸限制长度。
- **Cherk**：采用矩形输出块，完整块使用 2D 搬运，对角块逐列裁剪。共轭转置单面板路径直接生成所需虚部，减少镜像中间结果的读取和整理。

三个算子的 arch22 Kernel 关闭编译器自动同步，使用显式事件与屏障处理搬入、计算、搬出及 UB 复用依赖。Host 不新增流同步。

## 验证方案

复用任务 CSV 和新增 `TC_BIT_*`，覆盖双 uplo、转置、尾块、padding、步长、quick return 和非法参数。操作区按任务规定的 Netlib 3.10.0 参考及混合容差检查；另一三角与 padding 对调用前快照逐 bit 比较，哨兵覆盖不同 NaN payload、±0、±Inf、非规范数及有限值。

对浮点归约顺序引起的差异，另用 FP64 参考辅助定位，不将该诊断直接等同于任务指定参考的验收通过。CSV 与任务书的返回码冲突单独记录。

性能采用预热后超过 50 次采样的算术均值，与任务门槛比较；A2/A3 验证结果及未完成项分别记录。本文描述设计，不声明全部验收已完成。

## 支持硬件与代码位置

目标为 Atlas A2/A3（arch22）。实现位于 `blas/syr2/arch22/`、`blas/her2k/arch22/`、`blas/herk/arch22/`；测试位于相应 arch22 测试目录及 `test/triangle_preservation/`。

- [任务书参考 PR !2383](https://gitcode.com/cann/cann-ops-competitions/pull/2383)
- [修复代码分支](https://gitcode.com/guodong54_/ops-blas/tree/fix/arch22-triangle-csyr2)
