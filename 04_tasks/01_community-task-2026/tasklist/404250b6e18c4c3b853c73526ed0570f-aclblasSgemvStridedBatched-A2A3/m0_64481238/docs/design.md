# gemvStridedBatched 类五算子设计文档

## 需求背景

本设计对应 2026 年 11 月社区任务，目标是在 Atlas A2/A3（arch22）的 ops-blas 中提供下列 BLAS 接口。实现采用 Ascend C，由 handle 绑定的 stream 异步执行。[社区任务书与测试包](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)

| 工程项 | 位置 | 接入方式 |
| --- | --- | --- |
| 实现代码 | `blas/gemv_strided_batched/arch22/` | 新增 5 个变体（S/HSH/HSS/TST/TSS）入口与 kernel；当前仅有 `arch35/`，arch22 需按仓内规范新建 `gemv_strided_batched_host.cpp`/`gemv_strided_batched_kernel.cpp` |
| 参考实现 | `blas/gemv_strided_batched/arch35/gemv_strided_batched_host.cpp` | arch35（Ascend 950PR）已实现 5 个变体，可参考 stride 偏移、变体分派与缩放逻辑；不得直接复制 SDK 不兼容代码 |
| 测试代码 | `test/gemv_strided_batched/arch22/` | 复用/扩展 `test/gemv_strided_batched/arch35/` 的通用 `gemv_strided_batched_test` 工程，在 `test/gemv_strided_batched/CMakeLists.txt` 注册 arch22 目标 |

## 需求分析

y[b] ← alpha·op(A[b])·x[b] + beta·y[b]

### 公开接口

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

| 变体 | A | x | y（Y 的 dtype） | alpha / beta |
| --- | --- | --- | --- | --- |
| S | FP32 | FP32 | FP32 | FLOAT32 |
| HSH | FP16 | FP16 | FP16 | FLOAT32 |
| HSS | FP16 | FP16 | FP32 | FLOAT32 |
| TST | BF16 | BF16 | BF16 | FLOAT32 |
| TSS | BF16 | BF16 | FP32 | FLOAT32 |

### 参数校验与数据布局

| 约束项 | 内容 |
| --- | --- |
| 参数合法性与校验层级 | host/API 在本接口入口按任务书 §2.4 异常行为逐参数校验并返回对应状态码；handle、trans、m/n、lda、strideA/stridex/stridey、incx/incy、batchCount、alpha/beta 标量指针、A/x/y 基址均在 host 侧校验，不对 Device 张量做元素级扫描。 |
| 校验顺序 | `handle` → `trans` → `m/n` → `lda` → `incx/incy` → `strideA/stridex/stridey` → `batchCount` → `alpha/beta` → `A/x/y` → quick return；任一结构参数非法立即返回对应状态码，只有结构合法后才允许执行 quick return。 |
| stride 合法性 | 要求 `strideA

**quick return 说明**：`batchCount=0` 或 `m=0` 或 `n=0` 时允许 A/x/y 为空，接口不引用它们、不启动 kernel；`alpha=0` 时只执行 `y=beta*y`，不读取 A/x。

## 详细设计

### 以 64 位批间距定位 GEMV 的三个操作数

每个工作单元先计算 b·strideA、b·stridex、b·stridey 三个 64 位元素偏移，再叠加矩阵列主序偏移及向量负增量的首项修正。Host 根据 lda·n 与两向量的跨度 1+(len−1)·abs(inc) 校验批间距。

### 片上数据与回写

按 (批号,输出段) 分核。N 路径一次处理多行并沿列扫描，连续搬入 A 的行段，让同一个 x[j] 服务于这段行。T/C 路径以连续列段做点积。跨步 x、y 按元素增量读写，负增量按 (1−len)·inc 修正首项。短向量在同一核上顺序处理多个批，长向量沿归约轴分段；每个输出始终由唯一核写回。五个入口共用 FP32 累加，在最终合并 beta·y 后按输出类型转换一次。

Host 根据逻辑输出范围、数据类型和设备可用片上容量生成切分参数。核内缓冲按同时驻留的输入块、累加块和转换临时区分别计数；开启双缓冲时计入第二份输入块。超出容量时先缩短归约或输入段，再缩小输出块。搬运尾段使用实际有效长度。

## 可维可测分析

### 可手算的回归用例

令 m=2、n=3、lda=4、strideA=14，两批间保留填充。对 N 和 T 分别计算向量跨度，取 incx=−2、incy=2。逐批对照 CBLAS，同时检查批间空隙与 y 的奇数位哨兵；再用过小 stridey 验证重叠检查。

### 精度和性能验收

输出 y[i] 与 cuBLAS 对应变体一致；golden 由 CPU 侧 Netlib `cblas_sgemv` 逐 batch 生成：对每个 batch i 以 `op(A + i*strideA)`、`x + i*stridex`、`y + i*stridey` 独立调用一次 `cblas_sgemv`，FP32 计算后按输出 y 的 dtype 收窄。S/HSS/TSS 输出按 FLOAT32 判定，HSH/TST 输出分别按 FLOAT16/BFLOAT16 判定。

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|---|---|---|---|---|---|---|
| rtol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| atol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| required_matched_ratio | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| max_abs_error_limit | 1e-1 or 32 * ULP | 1e-1 or 32 * ULP | 1e-2 or 32 * ULP | 1e-1 or 32 * ULP | 1e-0 or 32 * ULP | 1e-0 or 32 * ULP |

按任务书同时核对匹配比例和最大绝对误差。异常参数用例直接断言返回状态码；正常用例在绑定 stream 同步后回读有效输出及边界哨兵。记录失败的用例名、输入种子、布局、返回值与最大误差所在元素。

| 任务包用例文件 | CSV 数据行 | 对应 GPU 基线行 | 首条用例 |
| --- | ---: | ---: | --- |
| `hsh_gemv_strided_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `hss_gemv_strided_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `sgemv_strided_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `tss_gemv_strided_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `tst_gemv_strided_batched_test.csv` | 1200 | 200 | `TC_L0_001` |

正式计时采用 `msprof op` 的 kernel Task Duration。预热 5 次后采集 10 次有效执行并取算术平均，将微秒换算为毫秒后与同一用例的 `gpu_ms` 比较。在 910B3 上逐例检查 NPU 平均耗时 ≤ 1.25 × GPU 耗时。

### 修订记录

2026-11-25，m0_64481238，提交初版设计。精度、性能及实机执行结果待实现后验证。
