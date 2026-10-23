# aclblasCsrot 算子设计文档

> 目标硬件：Ascend 950PR（arch35）；CANN 9.1.0。
> 数据类型：complex64（实部/虚部各 float32）+ float 实数旋转因子 c/s。
> 对标接口：cuBLAS `cublasCsrot`，语义参考 Netlib `csrot.f`。
> 说明：本窗口仅产出设计文档，不提交 PR。

---

## 一、任务概述

在昇腾 NPU（Ascend 950PR）上使用 Ascend C kernel 直调方式，基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）工程框架，实现句柄式 BLAS 接口 `aclblasCsrot`：对向量 x、y 施加 Givens 平面旋转 G = [[c, s], [-s, c]]，通过 handle 绑定 stream 直调 NPU kernel。

- 算子实现代码位置：`blas/rot/arch35/`。
- 测试代码位置：`test/rot/csrot/arch35/`。
- 接口声明已存在于 `include/cann_ops_blas.h`（第 222-224 行），本次不新增、不修改公共头，禁止定义 950PR 私有平行接口。

---

## 二、功能分析

### 2.1 接口签名

```cpp
aclblasStatus_t aclblasCsrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx,
    aclblasComplex* y, int incy, const float* c, const float* s);
```

### 2.2 Givens 旋转公式

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]
```

- **y 的更新必须使用原始 x[k]**，不能使用已更新后的 x[k]。
- `i = 1..n`；`k = 1 + (i-1)*incx`；`j = 1 + (i-1)*incy`（1-based 索引，兼容 Fortran）。

### 2.3 数据类型与运算规则

- `x`、`y` 为单精度复数向量 complex64（`aclblasComplex`，实部和虚部均为 float32）。
- `c`、`s` 为 float 实数标量。
- 实数标量 × 复数向量按「实部、虚部分别乘以该实数」执行：

```
Re(x'[k]) = c * Re(x[k]) + s * Re(y[j]);   Im(x'[k]) = c * Im(x[k]) + s * Im(y[j]);
Re(y'[j]) = c * Re(y[j]) - s * Re(x[k]);   Im(y'[j]) = c * Im(y[j]) - s * Im(x[k]);
```

### 2.4 参数校验矩阵

| 参数/条件 | 校验行为 |
|-----------|----------|
| handle | 为 nullptr 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（优先级最高，先于 n<=0） |
| n | n <= 0（含负数）为合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`，且**不读写 x/y**（对齐 Netlib `IF (N.LE.0) RETURN`） |
| x/y/c/s | n > 0 时任一为 nullptr，返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| (n-1)*incx / (n-1)*incy | 超出 int32 范围时返回 `ACLBLAS_STATUS_INVALID_VALUE`；**host 侧先用 int64 计算再收窄** |
| incx / incy | 正、负、零均**不拦截**（均合法） |
| c==1 且 s==0 | **不短路**，正常执行旋转公式 |
| 路径选择 | incx==1 且 incy==1 走连续路径；否则走 stride 路径 |
| 负步长 | 从向量尾端起算，沿负方向步进 |
| 零步长 | 按 Netlib 语义对同一元素反复旋转 |

---

## 三、需求分析

### 3.1 外部组件依赖

| 组件 | 适配状态 | 用途 |
|------|----------|------|
| CANN 9.1.0 Ascend C / ACL runtime | 已适配 | 编译 Kernel、管理 Device 内存、在 stream 上启动 Kernel |
| Ascend 950PR（arch35） | 已适配 | 本任务唯一正式验收硬件 |
| cuBLAS / Netlib | 不作运行时/生产依赖 | 仅作 API/数学语义参考；golden 由独立 CPU 参考实现生成 |

不新增 CATLASS、GEMM、TBE、ACLNN 或其他生产第三方依赖。

### 3.2 内部模块

| 模块 | 仓内位置 | 适配内容 |
|------|----------|----------|
| 公共 BLAS API | `include/cann_ops_blas.h` | `aclblasCsrot` 声明已存在（第 222-224 行），直接复用 |
| 句柄与 stream | `blas/rot/arch35/csrot_host.cpp` | 从 `aclblasHandle_t` 取已绑定 stream，异步 launch |
| Kernel 与 tiling ABI | `blas/rot/arch35/csrot_kernel.{cpp,h}`、`csrot_tiling_data.h` | 连续/stride 两条 kernel 路径与 `CsrotTilingData` |
| 构建与测试 | `test/rot/csrot/` | CSV 参数化 GTest、NPU wrapper、独立 CPU golden 与性能记录 |

### 3.3 依赖定义（类型/状态码/运行时判定）

- 接口签名：`include/cann_ops_blas.h:222-224`。
- 状态码：`include/cann_ops_blas_common.h:18-31`（`ACLBLAS_STATUS_SUCCESS=0`、`ACLBLAS_STATUS_INVALID_VALUE=3`、`ACLBLAS_STATUS_HANDLE_IS_NULLPTR=9` 等）。
- `aclblasComplex`：`include/cann_ops_blas_common.h:68-71`（`typedef struct { float real; float imag; } aclblasComplex;`）。
- 运行时指针属性判定：`aclrtPointerGetAttributes`（`aclrtPtrAttributes`，判 `location.type == ACL_MEM_LOCATION_TYPE_DEVICE`）。

---

## 四、详细设计

### 4.1 host 校验顺序（严格按此顺序）

1. ① `handle` 判空 → 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. ② `n <= 0` → 直接返回 `ACLBLAS_STATUS_SUCCESS`，no-op，不读写 x/y；
3. ③ `n > 0` 检查 `x`/`y`/`c`/`s` 非空 → 任一空返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. ④ 用 int64 检查 `(n-1)*incx` 和 `(n-1)*incy` 是否超出 int32 范围 → 超出返回 `ACLBLAS_STATUS_INVALID_VALUE`；
5. ⑤ 用 `aclrtPointerGetAttributes` 分别判定 `c`/`s` 指针位置（host/device）→ 失败返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. ⑥ 生成 tiling，按 `incx`/`incy` 分派连续/stride 路径；
7. ⑦ 在 handle 绑定 stream 上异步 launch（Host 不做 `aclrtSynchronizeStream`）。

### 4.2 连续路径与 stride 路径

- **连续路径**：仅限 `incx == 1` 且 `incy == 1`。
- **stride 路径**：其余所有情况，包括正非 1、负数、零及其混合步长。

### 4.3 int64 索引防溢出与负/零步长处理

- Host 侧 `(n-1)*stride` 先以 int64 计算，再收窄判断是否超出 int32；负步长先提升到 int64 再取相反数，避免 `INT_MIN` 取反溢出。
- 负步长不能使用 `n-1-i` 作为通用索引。
- 对 `incx < 0`：x 的起始物理偏移为 `(n-1) * abs(incx)`；第 i 个逻辑元素（i 从 0 开始）的物理偏移为 `(n-1) * abs(incx) + i * incx`。
- 对 `incy < 0` 同理。
- 相关计算使用 int64，且 `abs(INT_MIN)` 不能在 int32 中计算（须先提升到 int64 再取绝对值）。
- 这与文档中的 1-based 公式 `k = 1 + (i-1)*incx`、`j = 1 + (i-1)*incy` 的负步长语义一致。
- 零步长：对同一元素反复旋转，需串行回退（仅 block0.thread0 按 Netlib 顺序执行 n 次），避免并发读写同一地址产生竞态。

### 4.4 c/s host/device/mixed 三态统一处理

- host 指针 → host 解引用一次，将标量写入 tiling（`cosValue`/`sinValue`），kernel 用标量广播；
- device 指针 → 原样透传 device 指针到 GM（kernel 从 GM 读值），**不做 D2H 拷贝、不做 stream 同步**；
- `c`/`s` 独立判定，支持 host/device、device/host 混合（用 `cIsDevice`/`sIsDevice` 字段区分）；
- `aclrtPointerGetAttributes` 失败 → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- `n <= 0` 时不做 c/s 指针判定（no-op 直接返回，不触发指针属性查询）。

### 4.5 y 使用原始 x

计算 `y'[j] = c * y[j] - s * x[k]` 时，`x[k]` 必须是原始值。实现上须在 x 被覆写前先完成 y 的计算所需原始 x 的读取（寄存器/UB 保留原始 x，或先读后写），保证 in-place 安全。

### 4.6 tiling 数据字段与每核任务划分

`CsrotTilingData` 字段（参照 `blas/rot/arch35/srot_tiling_data.h`）：

- 公共：`tilingKey`（0=连续，1=stride）、`totalN`、`cosValue`、`sinValue`、`cIsDevice`、`sIsDevice`；
- 连续路径：`perCoreN`、`remainder`、`tileSize`；
- stride 路径：`incx`、`incy`、`nthreads`。

每核任务划分：

- 连续路径：`perCoreN = n / numBlocks`，`remainder = n % numBlocks`，前 remainder 个核各多 1 元素；`tileSize` 按 UB 与 32B 对齐。
- stride 路径：`nthreads = clamp(ceilDiv(n, numBlocks) 向上对齐 SIMT_MIN_THREAD_NUM, SIMT_MAX_THREAD_NUM)`，grid-stride 循环多核并行。
- 零步长串行回退：任一 inc 为 0 时，仅 block0.thread0 顺序执行 n 次，其余线程/block 直接返回。

### 4.7 arch35 代码文件建议结构

```
blas/rot/arch35/csrot_host.cpp        # Host 参数校验、tiling、三态判定、stream launch
blas/rot/arch35/csrot_kernel.cpp      # 连续 SIMD kernel + stride SIMT kernel + 零步长串行回退
blas/rot/arch35/csrot_kernel.h        # csrot_kernel_do 声明
blas/rot/arch35/csrot_tiling_data.h   # CsrotTilingData 结构
```

### 4.8 测试目录

```
test/rot/csrot/arch35/                # csrot_test.cpp / csrot_npu_wrapper.h / csrot_test.csv
test/rot/csrot/                       # csrot_param.h / csrot_golden.h / README.md / CMakeLists.txt
```

---

## 五、精度与性能标准

### 5.1 精度标准（complex64 实部/虚部）

- 实部/虚部分别按 FLOAT32 判定；
- `rtol = atol = 2^-13`（约 `1.2e-4`）；
- `matched_ratio >= 0.99`；
- `max_abs_error <= 1e-2` 或 `32xULP`。

逐元素通过条件：`|actual - golden| <= atol + rtol * |golden|`；用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 判通过。

### 5.2 性能标准（Ascend 950PR，complex64，incx=incy=1）

- 先 warmup 5 次，再有效采样 >50 次取平均；
- n=1048576 上限 21.69 us；
- n=2097152 上限 43.19 us；
- n=4194304 上限 119.38 us。

---

## 六、测试覆盖矩阵

| 维度 | 覆盖点 |
|------|--------|
| n | 0、负数（no-op）、1、小规模、2 的幂及 2 的幂 ±1、大规模（≥2^20） |
| incx/incy | 正、负、零、连续（incx=incy=1）、stride（其余组合） |
| c/s 指针位置 | host / device / mixed（cs_ptr_mode） |
| 负向用例 | 空 handle、空 x、空 y、空 c、空 s、int32 偏移溢出 |
| 特殊值 | c=1,s=0 不短路；复数实部/虚部、Inf/NaN、极端值 |
| 精度 | 实部/虚部独立比对，MERE≤2^-13、MARE≤10·2^-13 |

CSV 列（与任务包 `csrot_test.csv` 表头一致）：

```
case_name, description, n, x, incx, y, incy, c, s, cs_ptr_mode, expect_result, mere_threshold, mare_multiplier, random_seed
```

- `cs_ptr_mode` 仅允许 `host` / `device` / `mixed`。

---

## 七、支持硬件与约束

- 本次范围：Ascend 950PR / arch35（NPU_ARCH=dav-3510）。
- arch22（Atlas A2/A3）不在本次范围，不提供伪实现或错误声明支持。
- 目标代码目录：`blas/rot/arch35/`；测试目录：`test/rot/csrot/arch35/`。
- 不改变既有接口签名（`include/cann_ops_blas.h` 已声明，禁止定义 950PR 私有平行接口）。
- 不在本窗口提交 PR。

---

## 关键事实逐条自查

- [x] 接口签名一致；
- [x] 数学公式一致；
- [x] y 使用原始 x；
- [x] handle 校验优先；
- [x] n<=0 no-op；
- [x] 空指针校验；
- [x] int64 偏移检查；
- [x] 正/负/零步长；
- [x] 连续/stride 分派；
- [x] c/s host/device/mixed；
- [x] 精度阈值；
- [x] 性能阈值；
- [x] CSV 列；
- [x] 950PR/arch35，arch22 不在范围。
