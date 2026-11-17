# Atlas 950PR 单精度复数对角矩阵乘法 aclblasCdgmm 设计

## 需求背景

本设计对应 CANN 训练营华南师范大学专场的 `aclblasCdgmm` 任务，开发标识为 `Poyais-cloud`。目标是在 CANN 9.1.0 的 Atlas 950PR 上，提供与 cuBLAS `cublasCdgmm` 对齐的列主序复数对角矩阵乘法。应用可以直接用一维向量缩放矩阵的行或列，无需构造稠密对角矩阵。[官方任务页](https://www.hiascend.com/activities/task-center/details/dcfe51817ba0422a90b643f3e611cfb7)

`ops-blas` 基线 `9565ef4` 的公共头文件已声明该接口，950 系列的 `blas/dgmm/arch35/` 提供实数 `aclblasSdgmm`；其他产品线的复数版本具有自己的布局及产品支持约定。本设计新增 arch35 的复数实现，沿用公共 API，将产品线之间的布局、mode 和前导维差异记入接口文档。[ops-blas](https://gitcode.com/cann/ops-blas)

## 需求分析

```cpp
aclblasStatus_t aclblasCdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx,
    aclblasComplex* C, int ldc);
```

| 参数 | 契约 |
| --- | --- |
| `handle` | 有效的 BLAS 句柄；调用使用此句柄绑定的 stream |
| `mode` | `ACLBLAS_SIDE_LEFT` 逐行缩放，`ACLBLAS_SIDE_RIGHT` 逐列缩放 |
| `m, n` | 非负整数，分别表示行数与列数；任一为零时，合法参数组合执行空操作 |
| `A, C` | 设备侧列主序 complex64 矩阵。每列含 `m` 个逻辑元素，起始地址分别相隔 `lda`、`ldc` 个复数 |
| `lda, ldc` | 均至少为 `max(1, m)` |
| `x, incx` | 设备侧对角向量，逻辑长度 `L` 在 LEFT 时为 `m`，RIGHT 时为 `n`；非空调用至少可访问 `1 + (L-1) * abs(incx)` 个物理位置，步长可正可负且必须非零 |

对零基的行号 `i` 和列号 `j`，LEFT 使用 `k=i`，RIGHT 使用 `k=j`。当 `incx > 0` 时，向量偏移为 `k*incx`；当 `incx < 0` 时偏移为 `(L-1-k)*abs(incx)`。两种模式都计算复数乘法 `C[j*ldc+i] = A[j*lda+i] * x[offset]`。只处理逻辑矩阵；`C` 的列尾 padding 必须保留调用前的值。

允许 `A == C` 且 `lda == ldc` 的原地调用。调用方须保证 `C` 与 `x` 不重叠，A/C 部分重叠的行为未定义。

## 详细设计

### Host 侧

1. 校验句柄。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 校验 mode、维数、非零步长以及前导维。非法组合返回 `ACLBLAS_STATUS_INVALID_VALUE`。先校验前导维，再决定是否空操作，例如 `n=0, lda=0` 仍返回错误。
3. 空矩阵返回 `ACLBLAS_STATUS_SUCCESS`，允许数据指针为空。非空矩阵要求 A、x、C 均为有效指针，并检查同址时的前导维约束。
4. 以 64 位整数计算 `E=m*n`，查询设备可用 AIV 核数。启动 block 数为 `min(可用核数, ceil(E/256))`，核数查询为零时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
5. 在句柄绑定的 stream 上异步启动 kernel。各次调用的维度、步长和模式按值传递。该方案没有设备全局工作区，读取设备结果前由调用方同步。

### Kernel 侧

每个 AIV block 发起含 256 个线程的 SIMT 计算。线程起始逻辑索引为 `blockIdx.x * blockDim.x + threadIdx.x`，下一个索引相隔 `gridDim.x * blockDim.x`。这些等差序列互不重叠，覆盖 `[0, E)`。

对索引 `p`，列号为 `p / m`，行号为 `p % m`。矩阵与向量的复数索引和全局步长都使用 64 位运算；`incx` 提升至 64 位后再取相反数。以一个复数的两个浮点分量为例：

```text
A(i,j) = ar + i·ai
x(k)   = xr + i·xi
C.real = ar*xr - ai*xi
C.imag = ar*xi + ai*xr
```

一个线程先读取自身矩阵元素和对应向量元素的四个分量，再写回自己的输出位置。相邻行访问临近的 A 和 C 元素。LEFT 且步长绝对值大于 1 时会产生跨步读取；RIGHT 模式的一列复用同一个系数。实机按矩阵长宽比、mode、步长和前导维分组评估耗时，优先排查不满足单用例门槛的布局。

### 工程与兼容性

| 文件或目录 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 复用产品线共享的 `aclblasCdgmm` 声明 |
| `blas/dgmm/arch35/cdgmm_host.cpp` | 参数检查、空操作、分核及 stream 分发 |
| `blas/dgmm/arch35/cdgmm_kernel.cpp` | 列主序索引、负步长寻址及复数乘法 |
| `test/dgmm/cdgmm/arch35/` | 官方 CSV、官方 golden、设备 GTest |
| `blas/dgmm/README.md`、`docs/zh/api_list.md` | 区分 arch35 与 A2/A3 的乘法模式、布局和前导维语义 |

## 可维可测分析

### 精度与边界

官方数据包含 1,200 行用例，含 200 个性能用例。正确性对照任务包内的 `cdgmm_golden.h`：CPU 将 float 输入提升至 double，展开复数运算，再收窄至 float。A、x 的实部和虚部分别独立填充，按官方 CSV 中的随机种子、均匀分布、截断正态分布和特殊值配置构建输入。正态用例的均值、标准差及截断区间采用任务包约定。

分别统计实、虚部的有限值个数、匹配个数、匹配率及最大绝对误差。逐元素采用 `atol=rtol=2^-13`，每个分量的有限值匹配率应至少为 99%，绝对误差上限为 `max(0.01, 32*ULP(golden))`。Inf/NaN 单独统计：两者均为 NaN、均为同号 Inf，或一方为 NaN 另一方为 Inf 时接受；有限值与非有限值的配对失败。只对输出逻辑区计算误差，额外逐字节检查 padding。

重点回归 LEFT/RIGHT、非方阵、不同前导维、正负步长、空矩阵、非法 mode、空指针以及两个 A/C 同址场景：同前导维成功写回；不同前导维拒绝。分布、参数和用例名随日志保留，便于复现失败位置。

### 性能与复现

目标环境为 Atlas 950PR 和 CANN 9.1.0。设备节点编译并运行：

```bash
bash build.sh --soc=ascend950 --ops=cdgmm --run
```

对每个性能用例分别采集 msprof 设备 task duration，预热 5 次，至少取 10 次成功调用的耗时算术平均。用同一组 `m,n,mode,lda,ldc,incx` 匹配 H100 基线，先将 GPU 与 NPU 时间换为相同单位。每例 `GPU耗时/NPU耗时` 至少为 0.4，200 例比值的算术平均至少为 0.5。记录设备、软件版本、代码 commit 和原始日志，使验收方可以重复同一测试。[官方任务包](https://www.hiascend.com/p/resource/202611/db40646ec9c847abbd42564c9b773433.zip)

### 已完成的预验证

已在 Windows 上编译 Host 参数检查及按 CPU 方式执行的 SIMT 索引和计算逻辑，完成 64 组数值与布局检查，并核对官方 CSV 中的 17 条非法参数用例。Atlas 950PR 上的算子编译、精度、异步调用及性能采样待验证。
