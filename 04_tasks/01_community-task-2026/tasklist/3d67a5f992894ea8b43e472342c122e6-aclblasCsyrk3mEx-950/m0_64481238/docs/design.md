# aclblasCsyrk3mEx设计文档

## 需求背景

本设计对应 2026 年 11 月社区任务，目标是在 Ascend 950PR（arch35）的 ops-blas 中提供下列 BLAS 接口。实现采用 Ascend C，由 handle 绑定的 stream 异步执行。[社区任务书与测试包](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6)

| 工程项 | 位置 | 接入方式 |
| --- | --- | --- |
| 实现代码 | `blas/syrk/arch35/` | 在 `blas/syrk/arch35/` 下新增 `aclblasCsyrk3mEx` 入口与 3M kernel 路径；建议新建 `csyrk3m_ex_host.cpp` / `csyrk3m_ex_kernel.cpp`，或按仓内规范扩展现有 csyrk host/kernel。3M 三次实数乘必须在同一次 kernel launch 内完成，不得用三次独立 GEMM 调用拼接。 |
| 参考实现 | `blas/syrk/arch35/csyrk_host.cpp`、`csyrk_kernel.cpp/.h` 等；`blas/gemm/arch35/cgemm3m_host.cpp`、`cgemm3m_rounding_kernel.cpp` | 非 Ex `csyrk` 可参考参数校验、uplo 三角处理、对称语义下对角虚部保留与缩放；`cgemm3m` 可参考 3M 分解与舍入处理。对称秩-k 的乘积是 X·X^T，注意与 Hermitian 的 X·X^H 区分。 |
| 测试代码 | `test/syrk/csyrk3m_ex/arch35/` | 仓内已有 `test/syrk/csyrk`、`test/syrk/ssyrk`；需新建 `csyrk3m_ex` 子目录并注册构建目标 |

## 需求分析

C ← alpha·X·Xᵀ + beta·C，X = op(A)

### 公开接口

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



### 参数校验与数据布局

| 约束项 | 内容 |
| --- | --- |
| 参数合法性与校验层级 | host/API 在本接口入口按任务书 §2.4 异常行为逐参数校验并返回对应状态码；uplo、trans、n、k、lda、ldc 等整型与枚举取值由 host 校验，Atype、Ctype 按支持集校验，Atype=ACL_INT8 时按交错 (int8 real, int8 imag) 解释 A；不对 Device 张量做元素级扫描；张量数据本身是否满足对称或 Hermitian 约束由调用方保证，计算按公式自然进行。 |
| 非连续 Tensor 支持 | 不涉及，A 与 C 的访问跨度由 lda/ldc 描述的列主序带填充布局访问；不要求支持超出前导维度语义的任意非连续内存布局，不使用 AutoContiguous/ViewCopy，不产生相关临时空间。 |
| broadcast 规则 | 不支持，A 与 C 为独立操作数，shape 之间不做广播。 |
| dynamic shape 要求 | 不要求，n、k 为运行时入参，Host 侧按实际取值完成 tiling 与 launch。 |
| 原地与视图语义 | C 为原地覆写语义：输入旧值参与 beta 项计算后写回同一块内存，接口不返回视图，输出张量即入参 C 本体；仅 uplo 指定三角被读写，调用方须保证 C 的可写区域与 A 的可读区域不重叠。 |
| 确定性计算要求 | 不要求，3M 路径（融合/非融合、K 方向归约）由形状与片上容量决定，不按测试编号或随机种子分派；同一输入两次执行不保证逐位一致；验收按任务书 §3.2 阈值判定。 |
| 空 Tensor 处理 | n = 0 为合法 no-op：C 逻辑尺寸为 0×0，接口返回 `ACLBLAS_STATUS_SUCCESS`，不引用 A/C、不启动 kernel；k = 0 时不发生秩-k 更新，A 可不必有效；beta = (1,0) 时 quick return，beta ≠ (1,0) 时仍需对 C 的 uplo 三角做 beta 缩放。 |
| 0 维 Tensor 处理 | 不涉及，本接口不接受 0 维 Tensor 入参；矩阵经指针配合 n、k、lda、ldc 描述，标量以标量指针传入，Host 与 Device 两种指针位置均支持。 |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream，kernel 在 handle 绑定的 stream 上异步执行；接口返回不代表 Device 计算完成，Host 读回 Device 结果前须同步该 stream。 |

**校验顺序**：先校验 handle，再校验 uplo/trans，然后校验 n/k、lda/ldc，最后校验 alpha/beta/Atype/Ctype 与 A/C 指针等结构参数；结构合法后再执行 quick return。n=0 时允许 A、C 为 NULL，接口不引用它们。k=0 或 alpha=(0,0) 时 A 可为 NULL；但 n>0 时 C 必须非空，beta≠(1,0) 时仍需执行 C=beta·C。

## 详细设计

### 在选定三角内计算复对称三乘积更新

以 X=op(A) 为逻辑左矩阵，右矩阵取 Xᵀ，右侧虚部块保留原符号。对每个有效输出块计算 P=Xr·Br、Q=Xi·Bi、R=(Xr+Xi)·(Br+Bi)，再以 (P−Q)+i(R−P−Q) 重组。三个实数乘积在一次 kernel 启动内沿 K 累加。

### 片上数据与回写

Host 枚举与 uplo 相交的输出块，为每块分配唯一写者。完全落在有效三角内的块按列写回，穿过对角线的块逐列截取有效行。交错 complex-int8 的两个有符号分量分别提升为 FP32，再进行符号处理与预加，以覆盖 −128。使用复数 alpha、beta，保留对角虚部。OP_C 按任务约定映射到 OP_T。 由片上容量限制 (M,N,K) 块长，重用输入面板，避免三份全尺寸乘积写入 GM。

Host 根据逻辑输出范围、数据类型和设备可用片上容量生成切分参数。核内缓冲按同时驻留的输入块、累加块和转换临时区分别计数；开启双缓冲时计入第二份输入块。超出容量时先缩短归约或输入段，再缩小输出块。搬运尾段使用实际有效长度。

## 可维可测分析

### 可手算的回归用例

取 X=(1+i,2−i)ᵀ、alpha=1、beta=0，上三角有效元素依次为 2i、3+i、3−4i。在下三角填入哨兵并检查逐字节保持。对同一组整数分量分别以 complex64 与交错 int8 输入，比较输出。

### 精度和性能验收

算子输出结果需要与 cuBLAS `cublasCsyrk3mEx` 的输出一致；golden 由 CPU 侧 CBLAS **非 3M** 参考实现 `cblas_csyrk` 生成（Netlib BLAS 3.10.0/CBLAS，参考 https://www.netlib.org/blas/csyrk.f ），实部、虚部分别比对；CBLAS 无 `csyrk3mEx` 专用符号，Ex 变体按 Atype 精确提升后调用 `cblas_csyrk`（`Atype=ACL_INT8` 时每个复元素的 int8 实部/虚部各自升为 float；`Atype=ACL_COMPLEX64` 直接使用）；float64 精度域可改用 `cblas_zsyrk` 后将结果收窄到 complex64。3M 与直接复数乘的舍入过程不同，结果不保证逐位一致，按本节阈值判定。

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|---|---|---|---|---|---|---|
| rtol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| atol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| required_matched_ratio | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| max_abs_error_limit | 1e-1 or 32 * ULP | 1e-1 or 32 * ULP | 1e-2 or 32 * ULP | 1e-1 or 32 * ULP | 1e-0 or 32 * ULP | 1e-0 or 32 * ULP |

按任务书同时核对匹配比例和最大绝对误差。异常参数用例直接断言返回状态码；正常用例在绑定 stream 同步后回读有效输出及边界哨兵。记录失败的用例名、输入种子、布局、返回值与最大误差所在元素。

| 任务包用例文件 | CSV 数据行 | 对应 GPU 基线行 | 首条用例 |
| --- | ---: | ---: | --- |
| `csyrk3m_ex_test.csv` | 1200 | 200 | `TC_L0_001` |

正式计时采用 `msprof op` 的 kernel Task Duration。预热 5 次后采集 10 次有效执行并取算术平均，将微秒换算为毫秒后与同一用例的 `gpu_ms` 比较。每例 NPU 平均耗时 ≤ 2.5 × GPU 耗时，全部性能用例 GPU/NPU 加速比的算术平均 ≥ 0.5。对交错 INT8 输入另行补采基线并记录环境。

### 修订记录

2026-11-25，m0_64481238，提交初版设计。精度、性能及实机执行结果待实现后验证。
