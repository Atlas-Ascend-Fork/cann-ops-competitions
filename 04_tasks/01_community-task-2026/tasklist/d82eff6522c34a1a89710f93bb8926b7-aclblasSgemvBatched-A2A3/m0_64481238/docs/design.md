# gemvBatched 类五算子设计文档

## 需求背景

本设计对应 2026 年 11 月社区任务，目标是在 Atlas A2/A3（arch22）的 ops-blas 中提供下列 BLAS 接口。实现采用 Ascend C，由 handle 绑定的 stream 异步执行。[社区任务书与测试包](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)

| 工程项 | 位置 | 接入方式 |
| --- | --- | --- |
| 实现代码 | `blas/gemv_batched/arch22/` | 新增 5 个变体（S/HSH/HSS/TST/TSS）入口与 kernel；当前目录仅有 `cgemv_batched_*`（复数实现），可按仓内规范复用文件或新增 `gemv_batched_host.cpp`/`gemv_batched_kernel.cpp` |
| 参考实现 | `blas/gemv_batched/arch35/gemv_batched_host.cpp` | arch35（Ascend 950PR）已实现 5 个变体，可参考变体分派、tiling 与缩放逻辑；不得直接复制 SDK 不兼容代码 |
| 测试代码 | `test/gemv_batched/arch22/` | 复用/扩展 `test/gemv_batched/arch35/` 的通用 `gemv_batched_test` 工程，在 `test/gemv_batched/CMakeLists.txt` 注册 arch22 目标；`test/gemv_batched/cgemv_batched/arch22/` 可参考复数版本组织 |

## 需求分析

y[b] ← alpha·op(A[b])·x[b] + beta·y[b]

### 公开接口

```cpp
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float *const Aarray[], int lda,
    const float *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasTSTgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);
aclblasStatus_t aclblasTSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
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
| 参数合法性与校验层级 | host/API 在本接口入口按任务书 §2.4 异常行为逐参数校验并返回对应状态码；handle、trans、m/n、lda、incx/incy、batchCount、alpha/beta 标量指针、Aarray/xarray/yarray 指针数组均在 host 侧校验，不对 Device 张量做元素级扫描。 |
| 校验顺序 | `handle` → `trans` → `m/n` → `lda` → `incx/incy` → `batchCount` → `alpha/beta` → `Aarray/xarray/yarray` → quick return；任一结构参数非法立即返回对应状态码，只有结构合法后才允许执行 quick return。 |
| 非连续 Tensor 支持 | 支持 `incx`/`incy` 非零跨步访问 x/y（含负步长，按 BLAS 反向遍历语义）；矩阵按 lda 列主序带填充访问；每个 batch 的起始地址由 Aarray/xarray/yarray 指针数组描述，不要求 batch 间连续。 |
| broadcast 规则 | 不支持，每个 batch 为独立操作数，shape 之间不做广播。 |
| dynamic shape 要求 | 不要求，m/n/lda/incx/incy/batchCount 均为运行时入参，Host 侧按实际取值完成 tiling 与 launch。 |
| 原地与视图语义 | y[i] 为原地覆写：旧值参与 `beta*y` 后写回同一块内存；调用方须保证 y[i] 可写区域与 A[i]/x[i] 可读区域不重叠，且各 y[i] 之间不重叠；接口不返回视图。 |
| 确定性计算要求 | 不要求，浮点累加顺序不保证逐位一致；同一输入两次执行允许存在舍入差异，验收按任务书 §3.2 阈值判定。 |
| 空 Tensor 处理 | `batchCount=0`、`m=0` 或 `n=0` 为合法 quick return：直接返回 `ACLBLAS_STATUS_SUCCESS`，不引用 A/x/y、不启动 kernel，此时 Aarray/xarray/yarray 允许为 nullptr。 |
| 0 维 Tensor 处理 | 不涉及，本接口不接受 0 维 Tensor 入参；向量与矩阵经指针数组配合 m/n、lda、incx/incy 描述。 |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream，kernel 在 handle 绑定的 stream 上异步执行；接口返回不代表 Device 计算完成，Host 读回 Device 结果前须同步该 stream。 |

**quick return 说明**：`batchCount=0` 或 `m=0` 或 `n=0` 时允许 A/x/y 为空，接口不引用它们、不启动 kernel；`alpha=0` 时只执行 `y=beta*y`，不读取 A/x。

## 详细设计

### 通过设备指针数组定位每批 GEMV 的操作数

每个工作单元先以批号 b 从设备数组取得 A[b]、x[b]、y[b]；同一批的多个输出复用这三个地址。各批可来自不同分配，Host 只传入数组基址，设备按真实成员指针寻址。

### 片上数据与回写

按 (批号,输出段) 分核。N 路径一次处理多行并沿列扫描，连续搬入 A 的行段，让同一个 x[j] 服务于这段行。T/C 路径以连续列段做点积。跨步 x、y 按元素增量读写，负增量按 (1−len)·inc 修正首项。短向量在同一核上顺序处理多个批，长向量沿归约轴分段；每个输出始终由唯一核写回。五个入口共用 FP32 累加，在最终合并 beta·y 后按输出类型转换一次。

Host 根据逻辑输出范围、数据类型和设备可用片上容量生成切分参数。核内缓冲按同时驻留的输入块、累加块和转换临时区分别计数；开启双缓冲时计入第二份输入块。超出容量时先缩短归约或输入段，再缩小输出块。搬运尾段使用实际有效长度。

## 可维可测分析

### 可手算的回归用例

构造两块不同的 A 以及独立分配的 x 和 y，将第二批的指针放到数组首项。用逐批 CBLAS 校验输出顺序，验证设备按数组寻址。以非方阵、负 incx 和非单位 incy 覆盖 T 分支，并检查 y 中未选中的元素。

### 精度和性能验收

输出 y[i] 与 cuBLAS 对应变体一致；golden 由 CPU 侧 Netlib `cblas_sgemv` 逐 batch 生成：对每个 batch i 以 `op(A_i)`、`x_i`、`y_i` 独立调用一次 `cblas_sgemv`，FP32 计算后按输出 y 的 dtype 收窄。S/HSS/TSS 输出按 FLOAT32 判定，HSH/TST 输出分别按 FLOAT16/BFLOAT16 判定。

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|---|---|---|---|---|---|---|
| rtol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| atol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| required_matched_ratio | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| max_abs_error_limit | 1e-1 or 32 * ULP | 1e-1 or 32 * ULP | 1e-2 or 32 * ULP | 1e-1 or 32 * ULP | 1e-0 or 32 * ULP | 1e-0 or 32 * ULP |

按任务书同时核对匹配比例和最大绝对误差。异常参数用例直接断言返回状态码；正常用例在绑定 stream 同步后回读有效输出及边界哨兵。记录失败的用例名、输入种子、布局、返回值与最大误差所在元素。

| 任务包用例文件 | CSV 数据行 | 对应 GPU 基线行 | 首条用例 |
| --- | ---: | ---: | --- |
| `hsh_gemv_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `hss_gemv_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `sgemv_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `tss_gemv_batched_test.csv` | 1200 | 200 | `TC_L0_001` |
| `tst_gemv_batched_test.csv` | 1200 | 200 | `TC_L0_001` |

正式计时采用 `msprof op` 的 kernel Task Duration。预热 5 次后采集 10 次有效执行并取算术平均，将微秒换算为毫秒后与同一用例的 `gpu_ms` 比较。在 910B3 上逐例检查 NPU 平均耗时 ≤ 1.25 × GPU 耗时。

### 修订记录

2026-11-25，m0_64481238，提交初版设计。精度、性能及实机执行结果待实现后验证。
