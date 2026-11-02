# 【CANN社区任务】aclblasCher 算子设计文档（950PR）

目标仓库：<https://gitcode.com/cann/ops-blas>
目标实现目录：`blas/syr/arch35/`、`test/syr/cher/arch35/`
目标硬件：Ascend 950PR（DAV_3510，arch35）
贡献者：`skytosky`

# 需求背景

## 需求来源

本设计对应 2026 年 10 月社区任务“aclblasCher 算子开发（950PR）”。任务要求在 ops-blas 中新增单精度复数 Hermitian 秩1更新接口，完成公开声明、Host、arch35 Kernel、测试和自验材料。

`aclblasCher` 与已有 L3 `aclblasCherk` 不同，不得混用接口。仓内已有实数 `aclblasSsyr` 和复数 `aclblasCsyr` 的 arch35 实现，可参考其 handle/stream、SIMT 调度、三角遍历和负步长处理，但 Cher 的实数 alpha、共轭外积及对角虚部置零需要独立实现。

## 功能分析

计算：

```text
A = alpha·x·x^H + A
```

其中 `alpha` 为 FP32 实数，`x` 为 complex64 向量，`A` 为 complex64 Hermitian 矩阵。仅 `uplo` 指定三角被引用和更新，另一三角不读不写；对角虚部更新后强制为 0。`n=0` 或 `alpha=0` 是合法 quick return，A 保持不变。

# 需求分析

## 接口定义

在 `include/cann_ops_blas.h` 新增公共声明，供各产品线共用，禁止定义 950PR 私有平行 API：

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha,
    const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| `handle` | ops-blas 上下文和 stream | handle | 不涉及 | 有效非空 | 标量 |
| `uplo` | 指定引用/更新三角 | 枚举 | UPPER/LOWER | 非法值返回 `ACLBLAS_STATUS_INVALID_ENUM` | 标量 |
| `n` | Hermitian 矩阵阶数 | int32 | int32 | `n>=0`，`n=0` quick return | 标量 |
| `alpha` | 实数标量乘数 | Host 指针 | FLOAT32 | 非空，`alpha=0` quick return | 标量 |
| `x` | 复数输入向量，只读 | Device Tensor | complex64 | 按 incx 取 n 个元素 | 逻辑长度 n |
| `incx` | x 的元素步长 | int32 | int32 | 非零，支持负步长 | 标量 |
| `A` | 原地更新的 Hermitian 矩阵 | Device Tensor | complex64 | 仅 uplo 三角可写 | `[lda,n]` 列主序 |
| `lda` | A 的前导维 | int32 | int32 | `lda>=max(1,n)` | 标量 |

参数校验顺序采用同族 Csyr 口径：先检查 handle 和 `n<0`，`n=0` 直接成功返回；`n>0` 时依次检查 uplo、lda、incx 和 alpha 指针，再判断 `alpha=0` quick return，最后要求 x/A 非空。quick return 不启动 kernel，不读取或改写 A。

# 详细设计

## 数学与寻址

对逻辑元素 `x[i]`，物理复数下标为：

```text
start = incx >= 0 ? 0 : -(n-1)·incx
index(i) = start + i·incx
```

每个操作区元素 `(row,col)` 计算：

```text
v = alpha·x[row]
A[row + col·lda].real += v.real·x[col].real + v.imag·x[col].imag
A[row + col·lda].imag += v.imag·x[col].real - v.real·x[col].imag
```

当 `row==col` 时，完成上述更新后将 imag 置为 0。复数乘法保持明确 FP32 运算顺序，避免为性能改变 Inf/NaN 传播语义。

UPPER 仅处理 `row<=col`，LOWER 仅处理 `row>=col`。另一三角和 `n<=row<lda` 的 padding 不读不写，不能通过补全 Hermitian 或事后恢复实现。

## Host 与 Tiling 设计

1. 复用同族 handle/stream、参数校验和异步 launch 结构。
2. 使用 `GetAivCoreCount()` 获取运行时核数，按 n 和实际核数确定 block 数，不写死 950PR 核数。
3. Tiling 传递 `n/lda/uplo/incx/alpha`；alpha 为 FP32 Host 标量，直接取值，不引入复数 alpha 或 Device 标量同步。
4. quick return 在 Host 完成，workspace 非必需；不申请与 n 无关的持久 Device 缓存。
5. 首版采用 direct SIMT 正确路径；若 profiling 证明固定调度或 tile 化有收益，再以单变量实验扩展，不改变公开语义。

## Kernel 侧设计

采用 AIV 入口 + `asc_vf_call` 的混合 SIMT 模式，与同族 arch35 工程一致，不新增 `--enable-simt`。

```text
for row in grid_stride_rows:
    load x[row] once and compute alpha*x[row]
    for col in uplo_range(row, n):
        load x[col]
        update A[row + col*lda]
    if row is diagonal:
        A[row + row*lda].imag = 0
```

每个 GM 元素只有一个所属线程写回，避免原子操作。矩阵索引使用 64 位中间量，防止大 `lda×n` 溢出。首版不使用 UB 保存整矩阵，GM 仅访问 x 和操作区 A；若后续引入 tile，跨三角 tile 也必须按同一 uplo 条件裁剪写回。

## 内存与同步

在同一 stream 上异步执行，调用方同步后读取结果。无独立输出矩阵、无 CPU fallback、无整块 A 副本。性能采样时复用输入/workspace（若有），恢复 A 的操作不计入 Kernel 耗时。

# 支持硬件与约束

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / DAV_3510 / arch35 | √ |

仅 complex64 输入矩阵和向量，alpha 为 FP32；支持 UPPER/LOWER、任意非零 incx（含负步长）、lda padding 和 quick return。公开 ABI 为新增 `aclblasCher`，不得修改 `aclblasCherk` 或其他既有接口。

# 可维可测分析

## 精度与性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 操作区精度 | 实部/虚部分别 `atol=rtol=2^-13`，匹配率≥0.99，最大误差满足动态 32 ULP 上限 | 任务书 §3.2 |
| 非操作区 | 未引用三角与 lda padding 保持调用前哨兵值 | 任务书 §3.2 |
| 特殊值 | 按任务书 Inf/NaN 分类与等效规则独立判定，不以有限值通过替代 | 任务书 §3.2 |
| 性能 | 全部性能 case 平均单次耗时不高于 GPU 标杆 2.5 倍，即≥0.4×GPU | 任务书 §3.3 |
| 采样 | msprof op，5 次 warmup 后至少 10 次有效采样取平均 | 任务书 §3.3、§7 |

Golden 使用 Netlib BLAS 3.10.0 的 `cblas_cher`（CblasColMajor）。仅比较 uplo 操作区；另一三角与 padding 单独检查哨兵，计算本身不要求 bit-exact。

## 测试设计

1. 复用随包 `cher_test.csv` 和验证脚本，不删减原功能、精度、性能或负向用例。
2. 覆盖 UPPER/LOWER、n=0/1/小质数/2 的幂及±1/非对齐值、alpha 零/正负/大值、incx=±1/±2/±3、lda 紧凑与 padding。
3. 覆盖空指针、非法枚举、非法 lda、负 n、incx=0 和 quick return，核对返回码及 A 不变。
4. 操作区按 Hermitian 约束生成，非操作区使用不同哨兵；另覆盖 Inf/NaN、全零、交替值和极端值。
5. 使用 A2/A3 不涉及；950PR 上记录 CANN、驱动、Netlib、代码 commit 和原始日志。设计阶段不宣称精度或性能已达标。

## 兼容性与风险

只新增公共符号、arch35 实现和对应测试，不影响 Ssyr/Csyr/Cherk。主要风险为负步长起点、uplo 越界写、对角虚部未置零和 FP32 非有限值传播。通过步长专项、padding 哨兵、对角检查及同协议性能复测控制。

# 实施与交付计划

1. 新增公共声明、Host、direct SIMT Kernel 和 README，先完成参数与边界正确性。
2. 接入 `test/syr/cher/arch35/` GTest 和随包 CSV，完成精度、哨兵和异常路径验证。
3. 按同协议采集性能；若未达 0.4×GPU，先 profiling 再做单变量优化，不删减用例。
4. 交付设计文档、实现与测试代码、精度/性能自测报告及个人仓代码地址。
