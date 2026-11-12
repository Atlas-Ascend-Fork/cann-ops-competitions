# aclblasCrot 算子设计（Atlas A2/A3）

## 需求背景

### 需求来源

11 月 CANN 社区任务「aclblasCrot 算子开发（A2/A3）」。基于 `cann/ops-blas`，使用 Ascend C 实现单精度复数向量的 Givens 平面旋转。

### 背景介绍

接口对标 [cuBLAS ROT](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rot)，复数计算与负步长寻址参考 [Netlib/LAPACK CROT](https://netlib.org/lapack/lapack-3.1.1/html/crot.f.html)。基线仓已提供 `aclblasCrot` 公共声明和 arch35 实现，本任务补齐 Atlas A2/A3 的 arch22 实现。

## 需求分析

### 需求描述

在 handle 绑定的 stream 上异步执行，原地更新两个 complex64 向量。旋转余弦 c 为 FP32 实数，旋转正弦 s 为 complex64。c、s 指针各自支持 Host 或 Device 内存。

### 需求拆解

1. 支持任意合法的正负步长，并按 arch22 契约校验参数。
2. 使用同一对旋转前的 x、y 元素计算两个输出，保持步长间隙中的数据。
3. 连续向量使用 AIV 分块搬运和向量计算。通过仓内 GTest 和 msprof op 分别验证精度和性能。

## 详细设计

### 算子分析

对逻辑下标 i（0 ≤ i < n），以 `pos(i, inc) = (inc > 0 ? i : n - 1 - i) * |inc|` 计算物理下标。令 k = pos(i, incx)，j = pos(i, incy)，原地写回：

```text
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - conj(s) * x[k]
```

x、y 的逻辑形状均为 ND 一维 `[n]`。当 n > 0 时，每个缓冲区至少包含 `1 + (n - 1) * |inc|` 个复数，每个复数以交织的 FP32 实部、虚部存储。

### 公开接口

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx,
    aclblasComplex* y, int incy, const float* c, const aclblasComplex* s);
```

| 参数 | 类型和位置 | 含义 |
| --- | --- | --- |
| handle | `aclblasHandle_t`，Host | BLAS 句柄，绑定下发 stream |
| n | `int`，Host | 参与旋转的复数元素数 |
| x / y | `aclblasComplex*`，Device | 输入向量及原地输出 |
| incx / incy | `int`，Host | 对应向量的步长，单位为复数元素 |
| c | `const float*`，Host 或 Device | 旋转余弦 |
| s | `const aclblasComplex*`，Host 或 Device | 旋转正弦 |

### 算子实现

#### Host 侧设计

先校验 handle，空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。n < 0 返回 `ACLBLAS_STATUS_INVALID_VALUE`，n = 0 返回 `ACLBLAS_STATUS_SUCCESS`。n > 0 时，x、y、c、s 中任一指针为空，或者任一步长为 0、INT32_MIN 或 `(n-1)*|inc|` 超过 INT32_MAX，均返回 `ACLBLAS_STATUS_INVALID_VALUE`。跨度以 int64 计算。

以 `aclrtPointerGetAttributes` 分别查询 c 和 s 的位置。Host 标量随 tiling 参数按值传入，Device 标量在 kernel 中从 GM 读取，读取遵循 handle 绑定的 stream 顺序。算子无需全局 workspace。调用方在读取 Device 输出前同步该 stream。

#### 分核与 Kernel 侧设计

按四个复数元素划分一组，启动核数取设备可用 AIV 核数与组数的较小值，组数余量分给前几个核。末组截断到有效长度。

- 当 incx = incy = 1 且该核处理超过四个复数元素时，使用连续路径。以每向量 3,072 个 float 为一 tile，使用 DataCopyPad 搬运交织数据。Gather 交换实虚分量，按 `cross(z) = (-si*zi, si*zr)` 计算交叉项，得到 `s*y = sr*y + cross(y)` 和 `conj(s)*x = sr*x - cross(x)`。乘法和加减使用独立 FP32 指令。输入、输出双缓冲和四个临时缓冲共 144 KiB UB。
- 其他情况走逐元素路径。用 64 位 float 偏移读取原始 x/y，将结果暂存在 32 字节对齐的 UB 槽中，DataCopyPad 仅写回每槽有效的 8 字节。每核最多暂存 256 对结果，占 16 KiB UB。

### 支持硬件

目标硬件为 Atlas A2/A3（arch22），含 Atlas 800T A2 和 Atlas 800T A3。性能验收使用 Atlas 800T A2（910B3），工具链为 CANN 9.1.0。

### 算子约束限制

仅处理一维 complex64 向量。两个输出缓冲区各自原地更新，跨步间隙保持原值。部分重叠的 x/y 视图不属于本任务的验证范围。

## 可维可测分析

### 精度、性能和内存标准

| 项目 | 验收标准 | 来源 |
| --- | --- | --- |
| 精度 | 按 Netlib 公式构造 FP32 golden，对 x/y 实部、虚部独立比较。atol = rtol = 2⁻¹³，有限值匹配比例 ≥ 0.99，绝对误差上限为 1e-2 或 32 ULP。非有限值按任务书规则单独判定。 | 任务书 §3.2 |
| 性能 | 200 条用例使用 msprof op。预热 5 次后取至少 10 个有效 kernel task duration 的均值，每例 ≤ GPU 基线的 1.25 倍。 | 任务书 §3.3 |
| 内存 | 全局 workspace 为 0 字节，内存验收项不涉及。 | 任务书 §3.4 |

实机复现在 `ops-blas` 根目录执行 `bash build.sh --soc=ascend910b --ops=crot`，再按 [验收仓测试说明](https://gitcode.com/gcw_HuYbgqHW/ops-blas-crot-a2-a3/blob/d590030594b264cf43dd00263d68cd5b0dd43b24/test/rot/crot/README.md) 运行 `test/rot/crot/arch22/verify.py` 的精度和性能模式。Atlas A3 编译参数为 `--soc=ascend910_93`。

当前本地 CPU 模拟已通过 2,964 条精度和接口用例，200 条性能用例的调用流程已在 CPU 模拟中跑通。CANN 9.1.0 设备编译、NPU 精度及性能采集待实机验证。

### 兼容性分析

沿用 `include/cann_ops_blas.h` 已有签名和复数 ABI。arch22 对负 n 和零步长执行任务书的拒绝规则。现有 arch35 实现及其参数契约保持原样。
