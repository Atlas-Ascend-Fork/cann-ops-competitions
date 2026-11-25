# aclblasChpr2设计文档

## 需求背景

本设计对应 2026 年 11 月社区任务，目标是在 Atlas A2/A3（arch22）的 ops-blas 中提供下列 BLAS 接口。实现采用 Ascend C，由 handle 绑定的 stream 异步执行。[社区任务书与测试包](https://www.hiascend.com/activities/task-center/details/7a1275ee6b654459957fc27c8dde32b0)

| 工程项 | 位置 | 接入方式 |
| --- | --- | --- |
| 实现 | `blas/spr2/arch22/` | 新增 A2/A3 设备实现 |
| 参考 | `blas/spr2/arch35/` | 参考同名接口及边界处理 |

## 需求分析

AP(i,j) ← AP(i,j) + alpha·x[i]·conj(y[j]) + conj(alpha)·y[i]·conj(x[j])

### 公开接口

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* AP);
```



### 参数校验与数据布局

| 约束项 | 内容 |
| --- | --- |
| 参数合法性与校验层级 | host/API 在本接口入口按任务书 §2.4 异常行为逐参数校验并返回对应状态码；n、incx、incy、uplo 等整型与枚举取值由 host 校验，不对 Device 张量做元素级扫描；AP 数据本身是否满足 Hermitian 约束由调用方保证，计算按公式自然进行。 |
| 非连续 Tensor 支持 | 不涉及，x、y 的访问跨度由 incx/incy 描述（可为负值，按反向遍历语义访问），压缩矩阵 AP 按列主序无间隙布局访问；不要求支持超出步长语义的任意非连续内存布局，不使用 AutoContiguous/ViewCopy，不产生相关临时空间。 |
| broadcast 规则 | 不支持，各张量参数为独立操作数，shape 之间不做广播。 |
| dynamic shape 要求 | 不要求，矩阵阶数、向量长度与步长均为运行时入参，Host 侧按实际取值完成 tiling 与 launch。 |
| 原地与视图语义 | AP 为原地覆写语义：输入旧值参与累加后写回同一块内存，接口不返回视图，输出张量即入参 AP 本体；调用方须保证 AP 的可写区域与 x、y 的可读区域不重叠。 |
| 确定性计算要求 | 不要求，浮点乘加按实现内部归约顺序计算，同一输入两次执行不保证逐位一致；验收按任务书 §3.2 阈值判定。 |
| 空 Tensor 处理 | n = 0 为合法 no-op：此时 AP 逻辑长度为 0、x 与 y 逻辑长度均为 0；接口先按任务书 §2.5 校验顺序完成结构参数校验（uplo/incx/incy/alpha），校验通过后返回 `ACLBLAS_STATUS_SUCCESS`，不引用 AP/x/y、不启动 kernel；结构参数非法时仍返回 §2.4 对应错误（与 cuBLAS 一致）。 |
| 0 维 Tensor 处理 | 不涉及，本接口不接受 0 维 Tensor 入参；向量经指针配合 n、incx/incy 描述，压缩矩阵经指针配合 n 描述，标量以标量指针传入，Host 与 Device 两种指针位置均支持。 |
| 异步执行 | 依赖 aclblasSetStream 绑定 stream，kernel 在 handle 绑定的 stream 上异步执行；接口返回不代表 Device 计算完成，Host 读回 Device 结果前须同步该 stream。 |

**校验顺序（与 cuBLAS `cublasChpr2` 错误检查一致）**：handle 判空 → `n < 0` → `alpha` 指针判空 → `uplo` → `incx`/`incy`（含 `INT_MIN`）→ **`n = 0` quick return** → `x`/`y`/`AP` 指针判空 → `alpha = (0,0)` quick return。结构参数在 `n = 0` 时同样校验，非法取值仍返回对应错误；`n = 0` 只豁免 tensor 指针校验与 kernel 启动。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余结构参数非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。

## 详细设计

### 按压缩列段更新 Hermitian 矩阵

以列 j 的一段有效行作为工作单元。上三角的列首为 j(j+1)/2，下三角的列首为 j(2n−j+1)/2；下三角列内偏移为 i−j。均以 64 位元素索引计算。核间按累计元素数切分，对长列再切段，使核间工作量相近。

### 片上数据与回写

每列先取两个系数 alpha·conj(y[j]) 和 conj(alpha)·conj(x[j])，复用到整段行。将 x、y 的对应行载入 UB，与旧 AP 完成两次复数乘加。对角单独用旧实部参与运算并写零虚部。跨步向量按逻辑索引装载；负步长的首项偏移为 (1−n)·inc。

Host 根据逻辑输出范围、数据类型和设备可用片上容量生成切分参数。核内缓冲按同时驻留的输入块、累加块和转换临时区分别计数；开启双缓冲时计入第二份输入块。超出容量时先缩短归约或输入段，再缩小输出块。搬运尾段使用实际有效长度。

## 可维可测分析

### 可手算的回归用例

取 n=2、alpha=1、x=(1,i)、y=(1,1)，以零矩阵初始化。上三角按列应为 (2,1−i,0)，下三角应为 (2,1+i,0)。再向对角旧虚部注入 NaN，核对更新后的对角实部与虚部。

### 精度和性能验收

算子输出结果需要与 cuBLAS `cublasChpr2` 的输出一致；golden 由 CPU 侧 cblas 参考实现 chpr2 生成（请参考链接：https://www.netlib.org/blas/chpr2.f ），实部、虚部分别比对；

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|---|---|---|---|---|---|---|
| rtol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| atol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
| required_matched_ratio | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| max_abs_error_limit | 1e-1 or 32 * ULP | 1e-1 or 32 * ULP | 1e-2 or 32 * ULP | 1e-1 or 32 * ULP | 1e-0 or 32 * ULP | 1e-0 or 32 * ULP |

按任务书同时核对匹配比例和最大绝对误差。异常参数用例直接断言返回状态码；正常用例在绑定 stream 同步后回读有效输出及边界哨兵。记录失败的用例名、输入种子、布局、返回值与最大误差所在元素。

| 任务包用例文件 | CSV 数据行 | 对应 GPU 基线行 | 首条用例 |
| --- | ---: | ---: | --- |
| `chpr2_test.csv` | 1200 | 200 | `TC_L0_001` |

正式计时采用 `msprof op` 的 kernel Task Duration。预热 5 次后采集 10 次有效执行并取算术平均，将微秒换算为毫秒后与同一用例的 `gpu_ms` 比较。在 910B3 上逐例检查 NPU 平均耗时 ≤ 1.25 × GPU 耗时。

### 修订记录

2026-11-25，m0_64481238，提交初版设计。精度、性能及实机执行结果待实现后验证。
