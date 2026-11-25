# aclblasCgemm3m设计文档

## 需求背景

本设计对应 2026 年 11 月社区任务，目标是在 Atlas A2/A3（arch22）的 ops-blas 中提供下列 BLAS 接口。实现采用 Ascend C，由 handle 绑定的 stream 异步执行。[社区任务书与测试包](https://www.hiascend.com/activities/task-center/details/cae74cb900484862a2b43b24c0f7d9d3)

| 工程项 | 位置 | 接入方式 |
| --- | --- | --- |
| 实现代码 | `blas/gemm/arch22/` | 与同族 `aclblasSgemm` 的 arch22 实现同目录；由 `gemm_host.cpp` 增加 `aclblasCgemm3m` 入口，`gemm_kernel.cpp/.h` 增加 3M 复数 kernel 路径（也可按仓内规范拆分文件） |
| 参考实现 | `blas/gemm/arch35/cgemm3m_host.cpp` | arch35（Ascend 950PR）已有实现，可参考 3M 路径选择与缩放逻辑，不得直接复制 SDK 不兼容代码 |
| 测试代码 | `test/gemm/cgemm3m/arch22/` | 参考 `test/gemm/cgemm3m/arch35/`；需在 `test/gemm/cgemm3m/CMakeLists.txt` 中注册 arch22 目标 |

## 需求分析

C = alpha * op(A) * op(B) + beta * C

### 公开接口

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



### 参数校验与数据布局

| 约束项 | 内容 |
| --- | --- |
| 参数合法性与校验层级 | host/API 在本接口入口按任务书 §2.4 异常行为逐参数校验并返回对应状态码；handle、transa/transb、m/n/k、lda/ldb/ldc、alpha/beta/A/B/C 指针均在 host 侧校验，不对 Device 张量做元素级扫描；A/B/C 数据本身是否有效由调用方保证，计算按公式自然进行。 |
| 非连续 Tensor 支持 | 不涉及，A/B/C 按 lda/ldb/ldc 描述的列主序带填充布局访问；不要求支持超出前导维度语义的任意非连续内存布局，不使用 AutoContiguous/ViewCopy，不产生相关临时空间。 |
| broadcast 规则 | 不支持，A/B/C 为独立操作数，shape 之间不做广播。 |
| dynamic shape 要求 | 不要求，m/n/k/lda/ldb/ldc 与 transa/transb 均为运行时入参，Host 侧按实际取值完成 tiling 与 launch。 |
| 原地与视图语义 | C 为原地覆写语义：C 旧值参与 `beta*C` 累加后写回同一块内存，接口不返回视图，输出张量即入参 C 本体；调用方须保证 C 的可写区域与 A/B 的可读区域不重叠。 |
| 确定性计算要求 | 不要求，3M 路径（融合/非融合）与 K 方向归约顺序由实现按形状和片上容量选择，不按测试编号或随机种子分派；同一输入两次执行不保证逐位一致；验收按任务书 §3.2 阈值判定。 |
| 空 Tensor 处理 | m = 0 或 n = 0 为合法 no-op：接口直接返回 `ACLBLAS_STATUS_SUCCESS`，不引用 A/B/C，不启动 kernel，此时 A/B/C 允许为 nullptr；k = 0 或 alpha = 0 时只做 `C = beta*C`，允许 A/B 为 nullptr，但 m>0 且 n>0 时 C 必须非空。 |
| 0 维 Tensor 处理 | 不涉及，本接口不接受 0 维 Tensor 入参；矩阵经指针配合 m/n/k 与 lda/ldb/ldc 描述，标量以标量指针传入。 |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream，kernel 在 handle 绑定的 stream 上异步执行；接口返回不代表 Device 计算完成，Host 读回 Device 结果前须同步该 stream。 |

**校验顺序**：先校验 handle、transa/transb、m/n/k、lda/ldb/ldc、alpha/beta/A/B/C 指针等结构参数；结构合法后再执行 quick return。m=0 或 n=0 时允许 A/B/C 为 nullptr，接口不引用它们；k=0 或 alpha=0 时允许 A/B 为 nullptr；m>0 且 n>0 时 C 必须非空。

## 详细设计

### 对复数输出块做 Gauss 三乘积分解

先按转置枚举得到两个逻辑操作数 U 和 V，在 OP_C 装载阶段将虚部取负。对同一输出块沿 K 迭代，计算 P=Ur·Vr、Q=Ui·Vi、R=(Ur+Ui)·(Vr+Vi)，实部为 P−Q，虚部为 (R−P)−Q。三个累加块在 K 循环结束后统一乘复数 alpha，再合并 beta·C。

### 片上数据与回写

按 M、N 输出块分核，优先选取能容纳三个累加块的形状；输入实虚分量按 K 小块拆分和预加，避免在 GM 生成整幅预加矩阵。尾块用有效 M、N、K 限制读写。用同一输入比较相消样本与非 3M golden，区分舍入误差和转置符号错误。

Host 根据逻辑输出范围、数据类型和设备可用片上容量生成切分参数。核内缓冲按同时驻留的输入块、累加块和转换临时区分别计数；开启双缓冲时计入第二份输入块。超出容量时先缩短归约或输入段，再缩小输出块。搬运尾段使用实际有效长度。

## 可维可测分析

### 可手算的回归用例

取 A=[1+2i]、B=[3+4i]，alpha=1、beta=0，NN 的预期为 −5+10i；仅 A 用 OP_C 时预期为 11−2i。再以大实虚部且结果接近零的输入覆盖三乘积重组误差。

### 精度和性能验收

算子输出结果需要与 cuBLAS `cublasCgemm3m` 的输出一致；golden 由 CPU 侧 cblas 参考实现 `cblas_cgemm`（Netlib `cgemm`，请参考链接：https://www.netlib.org/blas/cgemm.f ）生成，实部、虚部分别比对。3M 算法与直接复数乘法的舍入过程不同，结果与 `cblas_cgemm` 不保证逐位一致，按本节阈值判定。

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|---|---|---|---|---|---|---|
| rtol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| atol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| required_matched_ratio | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| max_abs_error_limit | 1e-1 or 32 * ULP | 1e-1 or 32 * ULP | 1e-2 or 32 * ULP | 1e-1 or 32 * ULP | 1e-0 or 32 * ULP | 1e-0 or 32 * ULP |

按任务书同时核对匹配比例和最大绝对误差。异常参数用例直接断言返回状态码；正常用例在绑定 stream 同步后回读有效输出及边界哨兵。记录失败的用例名、输入种子、布局、返回值与最大误差所在元素。

| 任务包用例文件 | CSV 数据行 | 对应 GPU 基线行 | 首条用例 |
| --- | ---: | ---: | --- |
| `cgemm3m_test.csv` | 1200 | 200 | `TC_L0_001` |

正式计时采用 `msprof op` 的 kernel Task Duration。预热 5 次后采集 10 次有效执行并取算术平均，将微秒换算为毫秒后与同一用例的 `gpu_ms` 比较。在 910B3 上逐例检查 NPU 平均耗时 ≤ 1.25 × GPU 耗时。

### 修订记录

2026-11-25，m0_64481238，提交初版设计。精度、性能及实机执行结果待实现后验证。
