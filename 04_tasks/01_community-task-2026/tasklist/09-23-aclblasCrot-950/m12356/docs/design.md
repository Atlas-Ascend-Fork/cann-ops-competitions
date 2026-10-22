# aclblasCrot 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | aclblasCrot（单精度复数向量 Givens 平面旋转） |
| 任务名称 | CANN训练营南科大-aclblasCrot算子开发(950) |
| 任务来源 | 昇腾任务广场 `b6f4d038761348f5955c0abd6d7e5e85`；活动留言板 `gitcode.com/org/cann/discussions/318` |
| 目标硬件 | Ascend 950PR |
| 目标架构 | arch35（DAV_3510），UB 248 KB |
| 软件版本 | CANN 9.1.0（asc-devkit 9.1） |
| 开发仓 | ops-blas（`https://gitcode.com/cann/ops-blas`） |
| 实现目录 | `blas/rot/arch35/`；测试目录 `test/rot/crot/arch35/` |
| 参考资料 | 任务书 `aclblasCrot_Atlas950PR_task_doc.md`；仓内 `blas/rot/arch35/srot_*`、`blas/axpy/arch35/caxpy_*`、`blas/rot/arch22/csrot_*` |
| 文档模板 | cann-ops-competitions `resources/design_template.md` |
| 文档版本 | v1.0 |
| 编写日期 | 2026-10-20 |

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| v1.0 | 2026-10-20 | 算子开发组 | 首版：需求分析、详细设计、精度/性能口径、优化审查、风险与交付件 |

---

# 一、需求背景（required）

## 1.1 需求来源

本任务来自「CANN 训练营 南方科技大学专场」任务广场，任务名称为 **CANN训练营南科大-aclblasCrot算子开发(950)**，发布时间 2026-10-18，截止时间 2026-11-15。任务书要求：

- 参考 cuBLAS `cublasCrot` 语义，在 Ascend 950PR 上使用 Ascend C 开发 `aclblasCrot`；
- 支持两个单精度复数（complex64）向量应用 Givens 平面旋转；
- 验收通过后合入昇腾算子开源仓 ops-blas：实现入 `blas/rot/arch35/`，测试入 `test/rot/crot/arch35/`。

活动留言板 discussion 318 是本次训练营的统一答疑入口，包含学习资料、昇腾社区与 AtomGit 账号绑定、任务引导、在线课程与作业提交说明，不包含额外技术规格。技术规格以任务书为准。

## 1.2 背景介绍

Givens 旋转是 BLAS Level-1 的基础操作，广泛用于 QR 分解、线性方程组求解、特征值计算等数值算法。对复数向量 x、y 施加旋转：

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - conjg(s) * x[k]     // 使用旋转前的原始 x[k]
```

其中 `c` 为实数余弦，`s` 为复数正弦，`x`、`y` 原地更新。`y` 的更新包含 `s` 的共轭，这是 LAPACK `crot` 与 cuBLAS `cublasCrot` 的关键语义，不能套用实数 `srot` 或实正弦 `csrot`。

## 1.3 现状分析

ops-blas 仓 rot 族的现状：

| 接口/实现 | 架构 | 状态 | 可复用点 |
| --- | --- | --- | --- |
| `aclblasSrot` | arch35 | 已实现，连续 SIMD + 步长 SIMT 双路径 | Host 校验顺序、`aclrtPointerGetAttributes` 标量定位、TilingData 传值、内核直调、CSV 测试框架 |
| `aclblasCsrot` | arch22 | 仅冒烟测试；Host 侧 malloc workspace + H2D 拷贝 tiling + 流内同步 | 复数分量展开思路；其 Host 开销模式不应照搬 |
| `aclblasCrot` | — | 头文件无声明，仓内无实现、无 CSV 测试工程 | 需要新增声明、实现与测试 |

`include/cann_ops_blas.h` 已有 `aclblasCsrot`、`aclblasSrot` 声明，但**没有** `aclblasCrot`。本任务需新增公共声明，禁止定义 950PR 私有平行接口。

## 1.4 设计目标与非目标

**目标**

1. 功能与 cuBLAS `cublasCrot` / Netlib `crot` 对齐：复数 `s`、共轭更新、原始 `x` 语义、负步长尾端起算、零步长不拦截、`n <= 0` no-op。
2. 精度满足生态算子开源精度标准（complex64 实部/虚部分别按 FP32 判定）。
3. 性能达到任务书 §3.3 的三条标杆：n=1048576 / 2097152 / 4194304（incx=incy=1）。
4. 新增 `test/rot/crot/arch35` CSV 驱动测试工程，可复现精度、性能、内存结果。
5. 代码结构、文件数量与仓内同族算子一致，便于评审与维护。

**非目标**

- 不要求 broadcast、dynamic shape、确定性计算、额外 leading dimension padding。
- 不为 stride 路径做过度性能优化：任务书性能 case 全部为 `incx=incy=1`，优化预算集中在连续路径；步长路径保证功能正确与合理性能。

---

# 二、需求分析（required）

## 2.1 需求描述

使用 Ascend C 在 Ascend 950PR 上实现 `aclblasCrot`：对 complex64 向量 x、y 原地施加 Givens 旋转。接口采用句柄式 BLAS 风格，通过 handle 绑定 stream 直调 NPU kernel。

## 2.2 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 复数旋转公式、`conjg(s)`、原始 x 语义 | CSV 精度用例 + Netlib 语义 FP32 golden |
| R2 | `include/cann_ops_blas.h` 新增公共声明 | 头文件评审 + 编译 |
| R3 | `n <= 0` no-op；`n > 0` 空指针校验；handle 空指针优先 | 负向用例 |
| R4 | incx/incy 正、负、零；负步长尾端起算 | 步长组合用例 |
| R5 | `(n-1)*|inc|` 超出 int32 返回 `ACLBLAS_STATUS_INVALID_VALUE` | 溢出用例 |
| R6 | c、s 支持 Host/Device 指针，逐个独立判定 | `cs_ptr_mode` host/device/mixed |
| R7 | 精度达标 | 逐元素 atol/rtol + matched_ratio + max_abs_error |
| R8 | 性能达标 | 三条标杆 case，warmup 5 次、采样 > 50 次取平均 |
| R9 | README 产品支持表标注 Ascend 950PR：支持 | 文档评审 |
| R10 | 代码合入 ops-blas 指定目录，文件结构对齐 srot | PR 评审 |

## 2.3 接口定义

在 `include/cann_ops_blas.h` 新增（签名与任务书一致）：

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy,
    const float* c,
    const aclblasComplex* s);
```

`aclblasComplex` 定义于 `include/cann_ops_blas_common.h`：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

内存布局为 AoS（实部、虚部相邻），单个复数 8 字节。

## 2.4 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | dtype | 排布 | 维度 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | ops-blas 上下文句柄，携带 stream | scalar | — | — | — | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 输入 | 元素个数 | scalar | int | — | — | n ≤ 0 合法 no-op，返回 SUCCESS，不修改 x/y |
| x | 输入/输出 | 复数向量，原地更新为 `c*x + s*y` | tensor | COMPLEX64 | ND | 逻辑 [n]，物理 1+(n-1)*\|incx\| | n>0 且 nullptr → INVALID_VALUE |
| incx | 输入 | x 步长，可正/负/零 | scalar | int | — | — | `(n-1)*|incx|` 超 int32 → INVALID_VALUE |
| y | 输入/输出 | 复数向量，原地更新为 `c*y - conjg(s)*x` | tensor | COMPLEX64 | ND | 逻辑 [n]，物理 1+(n-1)*\|incy\| | n>0 且 nullptr → INVALID_VALUE |
| incy | 输入 | y 步长，可正/负/零 | scalar | int | — | — | 同上 |
| c | 输入 | 旋转余弦标量指针 | scalar | FLOAT32 | — | [1] | n>0 且 nullptr → INVALID_VALUE；Host/Device 均可 |
| s | 输入 | 旋转正弦标量指针（复数） | scalar | COMPLEX64 | — | [1] | n>0 且 nullptr → INVALID_VALUE；Host/Device 均可 |

**返回值**：`aclblasStatus_t`，状态码与 `cann_ops_blas_common.h` 一致。本算子使用 `ACLBLAS_STATUS_SUCCESS`、`ACLBLAS_STATUS_INVALID_VALUE`、`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`、`ACLBLAS_STATUS_INTERNAL_ERROR`（核数获取失败等内部错误）。

## 2.5 语义与边界约束

| 约束项 | 设计结论 |
| --- | --- |
| 参数校验优先级 | handle 空指针优先于 `n<=0` 早返回；`n<=0` 早返回优先于 x/y/c/s 空指针校验（对齐 srot 与 Netlib `IF (N.LE.0) RETURN`） |
| no-op | n ≤ 0 返回 SUCCESS，不读不写 x/y，不校验 c/s |
| 非连续 | 由 incx/incy 表达；支持正、负、零步长；零步长走 stride 路径，不拦截 |
| 负步长 | 从向量尾端起算：逻辑第 i 个元素物理索引 `(n-1-i)*|inc|` |
| 零步长语义 | 同一元素被连续旋转 n 次，必须串行执行，不能多线程并发读写同一地址 |
| 原地更新 | x、y 原地写回；`y'` 必须使用旋转前的 `x`；实现需先暂存原始分量 |
| 别名 | 不假设 x、y 不重叠；逐元素语义下，每个逻辑 i 先读原始 x/y 再写回，保持 Netlib 单步语义 |
| 标量指针 | c、s 各自独立判定 Host/Device；允许 c Host + s Device |
| 异步 | 不在接口内同步 stream；调用方读回结果前自行 `aclrtSynchronizeStream` |
| 空 Tensor / 0 维 | 统一归入 n ≤ 0 no-op |
| 确定性 | 不要求；多核划分固定，结果本身确定 |
| 精度 | complex64 实部/虚部分别按 FP32 混合容差判定 |

## 2.6 验收口径摘要

- **精度**：atol = rtol = 2^-13（1.2207e-4），matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32×ULP；逐元素条件 `|actual-golden| ≤ atol + rtol*|golden|`，实部、虚部分别统计。
- **性能**：Ascend 950PR，COMPLEX64 输入，warmup 5 次、有效采样 > 50 次取平均；三条 case 的 NPU 平均单次耗时分别不超过 21.96 us、43.70 us、119.52 us。
- **功能覆盖**：小 shape、shape 扫描、步长组合、标量特殊值、填充模式、对齐偏移、零维/负维/空指针/非法步长负向、INF/NAN。

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

对 i = 1..n，逻辑索引转物理索引（1-based 兼容 Fortran）：

```
k = 1 + (i-1)*incx
j = 1 + (i-1)*incy

x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - conjg(s) * x[k]      // x[k] 为旋转前的原始值
```

对应 Netlib/LAPACK `crot.f` 的 `TEMP` 暂存语义（任务书引用 `reference/cublas_docs/netlib/crot.f` 第 75-77 行）。

### 3.1.2 复数分量展开

记 `x = xr + i·xi`，`y = yr + i·yi`，`s = sr + i·si`，`c` 为实数。复数乘法 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`，共轭 `conjg(s) = sr - i·si`。

```
x'r = c*xr + (sr*yr - si*yi)
x'i = c*xi + (sr*yi + si*yr)

y'r = c*yr - (sr*xr + si*xi)
y'i = c*yi - (sr*xi - si*xr) = c*yi - sr*xi + si*xr
```

所有分量运算均为 FP32，无类型转换。

### 3.1.3 数据类型与内存布局

| 逻辑对象 | 物理表示 | 说明 |
| --- | --- | --- |
| complex64 向量 | 连续 float32 数组，长度 2n | `[xr0, xi0, xr1, xi1, ...]` AoS |
| 步长为 s 的访问 | 第 i 个复数的 float 偏移 `i*|s|*2` | 8 字节对齐，天然满足 float2 访问 |
| 标量 c | 1 个 float | Host 或 Device |
| 标量 s | 2 个连续 float | Host 或 Device，`sPtr[0]=sr, sPtr[1]=si` |

### 3.1.4 形状与步长

- 逻辑一维 `[n]`；不涉及 broadcast、维度补齐、leading dimension。
- 物理跨度 `span = 1 + (n-1)*|inc|` 个复数（零步长时实际只访问元素 0）。
- Host 测试按跨度分配 Device 缓冲；算子只按逻辑索引与步长访问。

### 3.1.5 关键语义

1. **原始 x**：`y'` 计算必须使用旋转前的 `x`。连续路径在 UB 中保留原始 x 的实/虚平面直到两侧结果都算完；SIMT 路径先读入寄存器再写回。
2. **共轭**：`y'` 使用 `conjg(s)`，即 `-si` 参与实部、`+si` 参与虚部（见 3.1.2），不可写成 `s`。
3. **负步长**：Netlib 在进入循环前将起始索引设为 `IX = (-N+1)*INCX + 1`，等价于从尾端起算；kernel 用 `(n-1-i)*|inc|` 实现。
4. **零步长**：Netlib 的 `IX` 恒定，第 i 次迭代读写的是上一次已更新的同一元素，属于串行累积；多线程并行会产生竞争并与参考结果不一致。
5. **单位旋转不短路**：`c==1,s==0` 正常走公式，不做特判（对齐 srot 与 Netlib）。

## 3.2 总体架构

### 3.2.1 逻辑视图

| 模块 | 职责 | 核心文件 |
| --- | --- | --- |
| API 层 | 公共接口、参数校验、标量定位、Tiling、内核下发 | `include/cann_ops_blas.h`、`blas/rot/arch35/crot_host.cpp` |
| Kernel 层 | 连续 SIMD 路径、步长 SIMT 路径、复数旋转 | `blas/rot/arch35/crot_kernel.cpp/.h`、`crot_tiling_data.h` |
| 测试层 | CSV 驱动 GTest、CPU golden、封装 | `test/rot/crot/CMakeLists.txt`、`test/rot/crot/arch35/*` |
| 文档层 | 设计文档、算子 README、自测报告 | `docs/design.md`、`blas/rot/README.md`、自测报告 |

模块依赖：

```
aclblasCrot (host API)
  ├─ 参数校验 / 标量定位 / Tiling 计算
  └─ crot_kernel_do
       ├─ tilingKey=0 → crot_aiv_kernel  (连续 SIMD)
       └─ tilingKey=1 → crot_simt_kernel (步长 SIMT)
```

### 3.2.2 开发视图

```
ops-blas/
├── include/
│   └── cann_ops_blas.h                 # 新增 aclblasCrot 声明
├── blas/rot/
│   ├── README.md                       # 增补 aclblasCrot 章节与产品支持表
│   └── arch35/
│       ├── crot_host.cpp               # 新增
│       ├── crot_kernel.cpp             # 新增
│       ├── crot_kernel.h               # 新增
│       └── crot_tiling_data.h          # 新增
└── test/rot/crot/
    ├── CMakeLists.txt                  # 新增
    └── arch35/
        ├── crot_test.cpp               # 新增
        ├── crot_param.h                # 新增
        ├── crot_golden.h               # 新增
        ├── crot_npu_wrapper.h          # 新增
        └── crot_test.csv               # 随任务提供，按需裁剪/扩充
```

构建无需修改 `blas/CMakeLists.txt`：该文件自动收集 `blas/*/*.cpp`，并按 `SOC_ARCH_DIRS`（ascend950 → arch35）收集架构目录源码。测试通过 `test/rot/crot/CMakeLists.txt` 注册 `crot_test`。

### 3.2.3 运行视图

```
调用方                           Host (aclblasCrot)                   Device
  │                                   │                                 │
  ├─ aclblasSetStream(handle, s) ────▶│                                 │
  ├─ aclblasCrot(...) ───────────────▶│ handle 校验                      │
  │                                   │ n<=0 ? → SUCCESS                 │
  │                                   │ x/y/c/s 校验                     │
  │                                   │ aclrtPointerGetAttributes(c/s)   │
  │                                   │ 选择 tilingKey、算 TilingData    │
  │                                   │ crot_kernel_do ────────────────▶ │ kernel 执行
  │◀──────── 立即返回（异步） ─────────│                                 │
  ├─ aclrtSynchronizeStream ──────────────────────────────────────────▶ │ 等待完成
  └─ D2H 读回结果                                                        │
```

接口内部**不做** workspace 申请、tiling H2D 拷贝、stream 同步。

## 3.3 Host 侧设计

### 3.3.1 入口与校验顺序

```cpp
extern "C" aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx,
    aclblasComplex* y, int incy, const float* c, const aclblasComplex* s)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n <= 0) return ACLBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return LaunchCrotKernel(handle, n, x, incx, y, incy, c, s);
}
```

校验优先级与 srot 一致：handle → no-op → 指针。这样 `handle==nullptr && n==0` 返回 HANDLE_IS_NULLPTR，`n<=0` 时不因空指针误报。

### 3.3.2 参数校验与错误码

| 条件 | 返回码 | 说明 |
| --- | --- | --- |
| handle == nullptr | HANDLE_IS_NULLPTR | 最高优先级 |
| n <= 0 | SUCCESS | no-op，不碰 x/y/c/s |
| n > 0 且 x 或 y == nullptr | INVALID_VALUE | 空指针 |
| n > 0 且 c 或 s == nullptr | INVALID_VALUE | 空指针 |
| 标量指针属性查询失败 | INVALID_VALUE | 与 srot / scalex 一致 |
| `(n-1)*|incx|` 或 `(n-1)*|incy|` 超 int32 | INVALID_VALUE | 防止 kernel 偏移溢出 |
| AIV 核数为 0 | INTERNAL_ERROR | 平台信息异常 |
| 其余 | SUCCESS | 已下发 |

### 3.3.3 标量指针位置判定

c（float）与 s（complex64）各自独立判定：

```cpp
struct CrotScalarPack {
    uint32_t cIsDevice;
    uint32_t sIsDevice;
    float cValue;
    float sRealValue;
    float sImagValue;
    const float* cArg;        // Host: nullptr；Device: 原指针
    const float* sArg;        // 指向 2 个 float
};

aclblasStatus_t ResolveCrotScalars(
    const float* c, const aclblasComplex* s, CrotScalarPack& pack)
{
    aclrtPtrAttributes attr{};
    if (aclrtPointerGetAttributes(c, &attr) != ACL_SUCCESS)
        return ACLBLAS_STATUS_INVALID_VALUE;
    pack.cIsDevice = (attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    if (aclrtPointerGetAttributes(s, &attr) != ACL_SUCCESS)
        return ACLBLAS_STATUS_INVALID_VALUE;
    pack.sIsDevice = (attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);

    pack.cValue     = pack.cIsDevice ? 0.0f : *c;
    pack.sRealValue = pack.sIsDevice ? 0.0f : s->real;
    pack.sImagValue = pack.sIsDevice ? 0.0f : s->imag;
    pack.cArg = pack.cIsDevice ? c : nullptr;
    pack.sArg = pack.sIsDevice ? reinterpret_cast<const float*>(s) : nullptr;
    return ACLBLAS_STATUS_SUCCESS;
}
```

- Host 标量：Host 侧解引用一次，写入 TilingData，kernel 用标量广播，不访问 GM。
- Device 标量：指针原样下发，kernel 在入口读取一次；不做 D2H 拷贝、不做同步。
- 允许 c Host + s Device 的混合组合。

### 3.3.4 TilingKey 选择

| tilingKey | 条件 | 路径 |
| --- | --- | --- |
| 0 | `incx == 1 && incy == 1` | 连续 SIMD（任务书性能 case 全走此路径） |
| 1 | 其余（正非 1、负、零及混合） | 步长 SIMT |

连续路径内部再按 `alignedCopy` 标志选择搬运 API：x/y 基址 32B 对齐时用 `DataCopy`，否则用 `DataCopyPad`。每核尾块长度是 unit 的整数倍（≥64 B），对齐时同样用 `DataCopy`。该分支只在每次搬运时判断，不增加 kernel 二进制数量。

### 3.3.5 TilingData

```cpp
struct CrotTilingData {
    // 公共
    uint32_t tilingKey;      // 0 连续 SIMD / 1 步长 SIMT
    uint32_t totalN;         // 逻辑元素个数
    float    cValue;         // Host c；cIsDevice 时为 0 占位
    float    sRealValue;     // Host s.real；sIsDevice 时为 0 占位
    float    sImagValue;     // Host s.imag；sIsDevice 时为 0 占位
    uint32_t cIsDevice;      // 1 = kernel 读 cArg
    uint32_t sIsDevice;      // 1 = kernel 读 sArg[0..1]

    // 连续路径
    uint32_t unitSize;       // 固定 8（8 个复数 = 64 B）
    uint32_t unitPerCore;    // 每核 unit 数
    uint32_t unitExtra;      // 前 unitExtra 个核各多 1 个 unit
    uint32_t tailN;          // totalN % unitSize，由 0 号核处理
    uint32_t tileSize;       // 每 tile 复数个数（2560，8 的倍数）
    uint32_t alignedCopy;    // 1 = 整块可用 DataCopy

    // 步长路径
    int32_t  incx;
    int32_t  incy;
    uint32_t nthreads;       // 每 block 线程数
};
```

TilingData 按值作为 kernel 参数传递，不申请 Device 内存、不做 H2D 拷贝。

### 3.3.6 分核与溢出检查

**连续路径**

1. 取 `unitSize=8`（8 个复数 = 64 B）。`alignedCopy` 由 x/y 基址是否 32B 对齐决定，与 unit 划分解耦；非对齐时整块走 `DataCopyPad`，不影响分核。
2. `units = n / unitSize`，`tailN = n % unitSize`。
3. `numBlocks = units == 0 ? 1 : min(units, aivCoreNum)`；空核由 `len==0` 早退。
4. `unitPerCore = units / numBlocks`，`unitExtra = units % numBlocks`；前 `unitExtra` 个核各多处理 1 个 unit。
5. 每核起点 `start = (blockIdx*unitPerCore + min(blockIdx, unitExtra)) * unitSize`，长度 `len = (unitPerCore + (blockIdx < unitExtra ? 1 : 0)) * unitSize`。
6. `tailN` 由 0 号核在主循环后处理。所有核区间 `[start, start+len)` 精确拼接，无重叠、无空洞。

**步长路径**

1. 先做 int64 溢出检查：
   ```cpp
   int64_t lastX = int64_t(n - 1) * incx;
   int64_t lastY = int64_t(n - 1) * incy;
   if (lastX > INT32_MAX || lastX < INT32_MIN ||
       lastY > INT32_MAX || lastY < INT32_MIN)
       return ACLBLAS_STATUS_INVALID_VALUE;
   ```
2. `numBlocks = clamp(ceilDiv(n, SIMT_MIN_THREAD_NUM), 1, aivCoreNum)`。
3. `nthreads = clamp(ceilAlign(ceilDiv(n, numBlocks), 128), 128, 2048)`。
4. 线程数按每核实际元素量缩放，避免小 n 启动满额线程。

**Tile 大小**

```
可用 UB = 248 KB，预留 16 KB 余量
单复数 UB 成本 = 22 个 float（见 3.4.4）
tileSize = alignDown((248KB - 16KB) / (22 * 4B), 256)
         = alignDown(237568 / 88, 256) = 2560 个复数
```

2560 个复数时 UB 实际占用 225280 B（220 KB），余量 28672 B（28 KB）。

### 3.3.7 内核下发

```cpp
void crot_kernel_do(uint8_t* x, uint8_t* y, uint8_t* cPtr, uint8_t* sPtr,
                    uint32_t numBlocks, const CrotTilingData& tiling, void* stream);
```

- 连续：`crot_aiv_kernel<<<numBlocks, nullptr, stream>>>(x, y, cPtr, sPtr, tiling);`
- 步长：`crot_simt_kernel<<<numBlocks, nullptr, stream>>>(x, y, cPtr, sPtr, tiling);`

两个独立 kernel 入口，避免 SIMD 与 SIMT 代码共存导致的寄存器压力和编译问题（沿用 srot 的组织方式）。不使用 workspace。

## 3.4 Kernel 侧设计

### 3.4.1 连续路径（tilingKey=0）

#### 数据流与流水线

```
GM x ──DataCopy/DataCopyPad──▶ UB xAos ──DeInterleave──▶ xr, xi
GM y ──DataCopy/DataCopyPad──▶ UB yAos ──DeInterleave──▶ yr, yi
                                                          │
                          t0,t1 = 复数旋转（见 3.4.3）◀────┘
                                 │
                          Interleave(t0,t1) ──▶ outAosX ──DataCopy/DataCopyPad──▶ GM x
                          （复用 t0,t1 算 y 分量）
                          Interleave(t0,t1) ──▶ outAosY ──DataCopy/DataCopyPad──▶ GM y
```

采用 TQue 输入双缓冲 + TQue 输出双缓冲，使 MTE2（搬入）、Vector（计算）、MTE3（搬出）三级重叠：

```cpp
TQue<TPosition::VECIN, 2>  inQueueX;
TQue<TPosition::VECIN, 2>  inQueueY;
TQue<TPosition::VECOUT, 2> outXQue;   // x' 输出双缓冲
TQue<TPosition::VECOUT, 2> outYQue;   // y' 输出双缓冲
TPipe pipe;
pipe.InitBuffer(inQueueX, 2, tileSize * 2 * sizeof(float));  // num=2 双缓冲
pipe.InitBuffer(inQueueY, 2, tileSize * 2 * sizeof(float));
pipe.InitBuffer(outXQue, 2, tileSize * 2 * sizeof(float));
pipe.InitBuffer(outYQue, 2, tileSize * 2 * sizeof(float));
```

输出用深度 2 的 `TQue<VECOUT>`：`AllocTensor` 得到 2×tileSize 浮点的槽位，Interleave 一次
写满（`dst` 与 `dst[count]` 两半）；`EnQue/DeQue` 由框架插入 V→MTE3 事件，`DataCopy` 后
`FreeTensor` 交还槽位，复用同一槽位时框架插入 MTE3→V 事件。全部同步交给队列，不使用手动
`FetchEventID/SetFlag/WaitFlag`——950PR 实测手动事件在“warmup 后 sync、再连续下发”场景会
挂死（NPU 100% 空转），改为队列后 200 条 PF 用例全部正常。

主循环采用输入预取 + 计算/搬出同轮：

```text
CopyIn(0)
for i in [0, tileCount):
    if i+1 < tileCount: CopyIn(i+1)        // MTE2 预取，与 Compute(i) 并行
    ComputeAndStore(i)                     // Vector 计算 + Interleave + MTE3 写回
                                           // 输出槽位复用由 TQue 内部同步
```

#### 分核与分块

- 每核处理一段连续逻辑区间（见 3.3.6），区间内按 `tileSize=2560` 复数分块。
- `fullTile = len / tileSize`，`tailTile = len % tileSize`；每核尾块长度是 unit（8 复数，64 B）的整数倍，对齐时用 `DataCopy`，非对齐时用 `DataCopyPad`。
- 空核（len==0）立即 return，避免空循环和无效同步。
- 每块起点均为 `tileSize` 的整数倍（相对每核起点），且 `tileSize` 为 8 的倍数；在 `alignedCopy=1` 时使用 `DataCopy`，否则 `DataCopyPad`。

#### 原地安全

- `x`、`y` 的原始分量在 `DeInterleave` 后保存在 `xr/xi/yr/yi` 平面中，直到 `x'` 与 `y'` 都计算完成才释放。
- 输出使用 x'/y' 各自的 AoS 双缓冲队列（`outXQue/outYQue`），与输入缓冲分离；`x'` 先写回 GM 后，`y'` 仍使用 UB 中的原始 `x` 分量计算，不读取 GM，因此语义正确。
- 不使用 `GlobalTensor::SetValue/GetValue`（性能黑名单 API），全部通过 DataCopy 系列搬运。

#### 尾块处理

- 每核尾块：`DataCopyPad` 以字节精确长度搬运，GM 侧只需 1 字节对齐。
- 全局尾块 `tailN = n % unitSize`（<8 个复数）：由 0 号核在主循环后处理，最多 7 个复数，用标量循环完成。
- 所有区间用 `[start, start+len)` 表示，`len` 由 unit 计算直接得出，**不使用减法反推尾长**，避免取负导致越界写。

### 3.4.2 步长路径（tilingKey=1）

#### SIMT 映射

```
globalThread = blockIdx.x * blockDim.x + threadIdx.x
gridStride   = gridDim.x * blockDim.x
for (i = globalThread; i < n; i += gridStride) {
    jx = negX ? (n-1-i) : i
    jy = negY ? (n-1-i) : i
    offX = jx * absIncX * 2      // float 偏移
    offY = jy * absIncY * 2
    // 读原始 x/y → 计算 x'/y' → 写回
}
```

- `absInc` 用 int64 取绝对值后转 uint32，避免 `incx == INT32_MIN` 时取负溢出。
- 每个线程先读入两个复数的 4 个 float，计算完两侧结果后一起写回；原始值保存在寄存器中，天然满足“y' 使用原始 x”。
- 相邻线程在 `|inc|==1` 时不走此路径（由连续路径处理）；其余步长下访存不连续，属于功能路径，不做强制合并优化。

#### 零步长串行回退

零步长会在同一地址上串行累积，必须严格按 i 递增顺序执行。采用**VF 入口守卫**：`zeroIncX/zeroIncY` 作为参数传入 VF，除 block0.thread0 外的所有线程在循环前 `return`，串行循环结束后同样 `return`，不再进入并行走廊：

```cpp
if ((zeroIncX != 0) || (zeroIncY != 0)) {
    if ((blockIdx.x != 0U) || (threadIdx.x != 0U)) return;
    for (uint32_t i = 0; i < n; i++) { /* 读改写后 return */ }
    return;
}
```

这样整次调用只启动一个 VF（grid 仍按 `numBlocks` 配置，守卫使其等价于单线程），且 `zeroInc/negInc` 判断不进入并行循环体。

串行循环内：

```
jx = zeroIncX ? 0 : (negX ? n-1-i : i)
jy = zeroIncY ? 0 : (negY ? n-1-i : i)
```

零步长一侧的 `absInc` 置 1 仅为地址计算合法，索引固定为 0。

#### 分支与线程配置

- 把 `zeroInc/negInc` 等整次调用一致的条件提升到 VF 外，避免每个线程重复判断（SIMT Rule 001）。
- 线程数按每核元素量取 128~2048，小 n 不启动满额线程。
- 复数读写天然 8 字节对齐，可评估 `float2` 短向量访存以减半 load/store 指令数（P1 优化，见第 5 章）。

#### SIMT 舍入序列

四个分量用显式 `fmaf` 展开，顺序与 CPU golden 的 `std::fma` 完全一致：

```cpp
newXr = fmaf(-si, yi, fmaf(sr, yr, c * xr));
newXi = fmaf(si, yr, fmaf(sr, yi, c * xi));
newYr = fmaf(-si, xi, fmaf(-sr, xr, c * yr));
newYi = fmaf(si, xr, fmaf(-sr, xi, c * yi));
```

必须显式写出：普通的 `c*xr + sr*yr - si*yi` 在编译后是逐项舍入，而 golden 是“一次乘法 + 两次 FMA”。CSV 的 TC_EX 极值标量组（`c=1e10`、`s=±1e10`）会把这一 ULP 级差异放大到 1e3~1e4，触发逐元素 `32×ULP` 上限（实测 43 条失败）；改为显式 `fmaf` 后 AIV/SIMT 两条路径与 golden 逐位一致，TC_EX 全部 `maxAbsErr=0`。

### 3.4.3 复数计算序列

每个 tile 内，x/y 已解交织为 `xr, xi, yr, yi` 四个 float 平面，长度均为 `tileSize`。使用两个临时平面 `t0, t1`，先算 `x'` 并写回，再复用 `t0, t1` 算 `y'`。

| 步骤 | 操作 | 表达式 |
| --- | --- | --- |
| 1 | Muls | `t0 = c * xr` |
| 2 | Muls | `t1 = c * xi` |
| 3 | Axpy | `t0 += sr * yr` |
| 4 | Axpy | `t0 += (-si) * yi` |
| 5 | Axpy | `t1 += sr * yi` |
| 6 | Axpy | `t1 += si * yr` |
| 7 | Interleave + 搬出 | `out = interleave(t0,t1)` → GM x |
| 8 | Muls | `t0 = c * yr` |
| 9 | Muls | `t1 = c * yi` |
| 10 | Axpy | `t0 += (-sr) * xr` |
| 11 | Axpy | `t0 += (-si) * xi` |
| 12 | Axpy | `t1 += (-sr) * xi` |
| 13 | Axpy | `t1 += si * xr` |
| 14 | Interleave + 搬出 | `out = interleave(t0,t1)` → GM y |

计算量：4 次 `Muls` + 8 次 `Axpy`，共 12 条向量指令；相比逐项 `Muls + Add/Sub` 的 20 条指令减少 40%。

**舍入策略**：`Axpy` 使用 FMA，单次舍入，而 Netlib 参考实现按每步舍入。srot 在 `y` 腿（45° 旋转、x≈y 时接近抵消）曾因 FMA 与参考不一致而改用 `Muls + Muls + Add`。crot 的 `y'` 同样存在抵消风险，因此：

- 默认实现采用 12 指令 `Axpy` 版本，golden 按同一“一次乘法 + 两次 FMA”顺序展开；
- 950PR 实测：全部连续路径精度用例（含 `FLT_MAX`/`Inf`/`NaN` 填充的 TC_FL 组）与 golden 的 `maxAbsErr` 为 0，无需切换到逐项舍入版本；
- 若后续更换平台或参考实现导致近抵消场景超限，可把 `y'` 腿改为 `Muls + Add/Sub`（`x'` 腿保留 `Axpy`），通过编译期常量控制，最终只保留通过精度验证的一条。

### 3.4.4 UB 预算

tileSize = 2560 个复数：

| 缓冲 | 数量 | 大小 |
| --- | --- | --- |
| inQueueX | 2 块 × 2×2560 float | 40,960 B |
| inQueueY | 2 块 × 2×2560 float | 40,960 B |
| outXQue | 2 块 × 2×2560 float | 40,960 B |
| outYQue | 2 块 × 2×2560 float | 40,960 B |
| xr, xi, yr, yi | 4 × 2560 float | 40,960 B |
| t0, t1 | 2 × 2560 float | 20,480 B |
| **合计** | 22 × 2560 float | **225,280 B（220 KB）** |

UB 可用 248 KB，余量 28 KB。约束写入代码（Host/Kernel 共用同一常量）：

```cpp
constexpr uint32_t CROT_UB_FLOATS_PER_COMPLEX = 22;
constexpr uint32_t CROT_TILE_COMPLEX =
    ((UB_SIZE - 16 * 1024) / (CROT_UB_FLOATS_PER_COMPLEX * sizeof(float)) / 256) * 256;  // 2560
static_assert(CROT_TILE_COMPLEX * CROT_UB_FLOATS_PER_COMPLEX * sizeof(float) <= UB_SIZE,
              "Crot contiguous UB budget exceeded");
```

若后续增加临时平面或提高缓冲深度，必须同步下调 `tileSize`，并在 Host 侧用同一公式计算，禁止 Host/Kernel 各写一套常量。

### 3.4.5 TilingKey 与分支汇总

| 分支 | 条件 | 使用资源 | 说明 |
| --- | --- | --- | --- |
| tilingKey=0, alignedCopy=1 | 连续且 unit=8 | DataCopy + 双缓冲 | 性能主路径，三条标杆 case |
| tilingKey=0, alignedCopy=0 | 连续但 x/y 基址非 32B 对齐 | DataCopyPad + 双缓冲 | 功能兜底 |
| tilingKey=1, zero=0 | 任一步长非 1 且无零步长 | SIMT 并行 | 功能路径 |
| tilingKey=1, zero=1 | 任一步长为 0 | SIMT 单线程串行 | 语义路径 |

## 3.5 API 映射与验证记录

| 计算步骤 | API | 签名/用法 | 平台 | 约束 | 验证状态 |
| --- | --- | --- | --- | --- | --- |
| GM→UB 整块搬入/搬出 | `DataCopy` | `DataCopy(dst, src, count)` | arch35 | 仅 32B 严格对齐时使用；count 为元素数 | ✅ 仓内 `caxpy` arch35 已用 |
| GM↔UB 非对齐/尾块 | `DataCopyPad` | `DataCopyPad(dst, src, DataCopyExtParams{1, bytes, 0,0,0}, padParams)` | arch35 | GM 侧 1B 对齐；UB 自动 pad；尾块只消费有效长度 | ✅ 仓内 `srot` arch35 已用 |
| 复数解交织 | `DeInterleave` | `DeInterleave(dstR, dstI, srcAos, count)` | arch35 | `count` 为 float 个数 = 2×复数个数；输出两个平面 | ✅ 仓内 `caxpy` arch35 已用；正式开发按 CANN 9.1 API 文档复核 membase 签名 |
| 复数交织 | `Interleave` | `Interleave(dst0, dst1, srcR, srcI, count)` | arch35 | `dst0/dst1` 为同一输出缓冲的两段；`count` 为复数个数 | ✅ 仓内 `caxpy` arch35 已用；同上复核 |
| 标量乘 | `Muls` | `Muls(dst, src, scalar, count)` | arch35 | count ≤ 255×64 时单 repeat；超过需分段 | ✅ 仓内 `srot/csrot` 已用 |
| 乘加 | `Axpy` | `Axpy(dst, src, scalar, count)` | arch35 | FMA 语义；精度敏感路径需验证 | ✅ 仓内 `srot/csrot` 已用 |
| 加减 | `Add/Sub` | `Add(dst, s0, s1, count)` | arch35 | 逐项舍入备选路径 | ✅ 仓内 `srot` 已用 |
| SIMT 启动 | `asc_vf_call` | `asc_vf_call<vf>(dim3{x,1,1}, args...)` | Atlas 350 / 950PR | 参数仅支持裸指针与基础类型，不支持结构体；VF 不能是成员函数 | ✅ 仓内 `srot/caxpy` 已用；官方文档 [asc_vf_call](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00267.html) |
| SIMT 限定符 | `__simt_vf__` / `LAUNCH_BOUND` | `LAUNCH_BOUND(2048)` | arch35 | 线程数 1~2048；决定寄存器预算 | ✅ 官方 [SIMD/SIMT 混合算子](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_10040.html) |
| 标量指针位置 | `aclrtPointerGetAttributes` | `aclrtPointerGetAttributes(ptr, &attr)` | Runtime | 返回 `attr.location.type`；失败按 INVALID_VALUE | ✅ 仓内 `srot` 已用；官方 [统一寻址](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/API/runtimeapi/docs/zh/api_ref/11-05_unified_addressing.md) |
| 核数获取 | `GetAivCoreCount()` | Host 工具函数 | arch35 | 返回 0 视为平台异常 | ✅ 仓内 `srot/caxpy` 已用 |
| 任务类型 | `KERNEL_TASK_TYPE_DEFAULT` | `KERNEL_TYPE_AIV_ONLY` | arch35 | 仅 AIV | ✅ 仓内 `srot/caxpy` 已用 |

**验证检查清单**

- [x] `DataCopy` 仅用于 32B 对齐块，非对齐全部走 `DataCopyPad`
- [x] `DeInterleave/Interleave` 的 count 单位（float vs 复数）在实现中区分清楚
- [x] `Axpy` 的 FMA 舍入风险有精度用例与回退路径
- [x] `asc_vf_call` 不传结构体，只传裸指针与标量
- [x] `aclrtPointerGetAttributes` 对 c、s 分别调用，支持混合位置
- [x] 不修改 `$ASCEND_HOME_PATH` 等 SDK 安装目录

## 3.6 内存与 Workspace

| 项目 | 结论 |
| --- | --- |
| Device workspace | 不需要；原地算子，无临时 GM |
| Host 侧 malloc | 不需要；TilingData 按值传参 |
| Tiling H2D 拷贝 | 不需要 |
| 接口内 stream 同步 | 不需要 |
| 额外内存 | 仅 Device 输入输出缓冲，由调用方分配 |

与 `csrot` arch22 的 `aclrtMalloc(workspace) + aclrtMalloc(tiling) + H2D + aclrtSynchronizeStream` 模式相比，本设计每次调用少 2 次 malloc、1 次 H2D、1 次同步，对小 n 和性能采样都有直接收益。

## 3.7 支持硬件

| 芯片版本 | 支持 | 说明 |
| --- | --- | --- |
| Ascend 950PR | √ | 主目标，arch35 |
| Ascend 950DT | √（同 arch35，随仓内编译配置） | 同一实现 |
| Atlas A2 / A3 系列 | × | 无 arch22 实现；接口声明为公共声明，不提供 A2/A3 内核 |

README 产品支持表需标注：Ascend 950PR：支持。

## 3.8 约束限制

1. 仅支持 complex64 向量；c 为 float，s 为 complex64。
2. 不支持 broadcast、dynamic shape、leading dimension padding。
3. x、y 必须为 Device 内存；c、s 可为 Host 或 Device。
4. n ≤ 0 为合法 no-op；不做参数错误返回。
5. incx/incy 为 0 合法，走串行语义路径，性能不做保证。
6. `(n-1)*|inc|` 超 int32 返回 INVALID_VALUE。
7. 接口不负责 stream 同步。
8. 不定义 950PR 私有平行接口，声明统一放入 `include/cann_ops_blas.h`。

---

# 四、可维可测分析（required）

## 4.1 精度标准

依据任务书与生态算子开源精度标准（complex64 实部/虚部分别按 FP32）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64（实部/虚部分别） | 2^-13（1.2e-4） | 2^-13（1.2e-4） | 0.99 | 1e-2 或 32×ULP |

逐元素通过条件：

```
|actual - golden| ≤ atol + rtol * |golden|
```

用例通过条件：`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max(1e-2, 32×ULP)`。

测试工程口径（ops-blas frame）：调用 `applyMixedTolerance` 后显式覆盖 `mixedRtol = mixedAtol = 2^-13`（frame 对 FP32 的默认值为 2^-10 / 2^-16），再用 `Verifier::verifyVector` 对实部、虚部分别统计；CSV 中的 `mere_threshold` / `mare_multiplier` 两列保留以兼容任务书模板，但不再作为判定开关。`MixedToleranceStrategy` 对完全相等（含同号 Inf）或两侧同为 NaN 的元素跳过，仅一侧为特殊值时判失败。

**误差来源**：每个输出分量最多经历 3 次乘法和 2 次加法（或 FMA），正常实现应远优于阈值。重点风险场景是 45° 旋转且 x≈y 的抵消场景。

## 4.2 性能标准

| case | n | incx | incy | 标杆耗时（us） | 对应带宽需求（4 条流） |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,048,576 | 1 | 1 | 21.96 | ≈1.53 TB/s |
| 2 | 2,097,152 | 1 | 1 | 43.70 | ≈1.54 TB/s |
| 3 | 4,194,304 | 1 | 1 | 119.52 | ≈1.12 TB/s |

- 流量模型：每个复数读 x、读 y、写 x、写 y，共 32 字节；n=1M 时约 32 MiB。
- 950PR 理论 HBM 带宽约 1.6 TB/s；case 1/2 需要约 95% 峰值带宽，case 3 余量较大。
- 性能测试会 warmup 5 次并采样 >50 次，工作集 32~128 MiB。1M/2M 的工作集小于 L2（约 112 MB），重复采样时可能 L2 常驻，这是达到标杆的重要前提；因此**不做 L2 bypass**。
- 判定：NPU 平均单次耗时 ≤ 标杆耗时（GPU 基线 / 0.4）。基线回填前标记 NO_REF，仅采集 NPU 耗时。
- kernel 精确耗时用 `msprof op` 采集，统计 `OpBasicInfo.csv` 的 `Task Duration(us)`。

## 4.3 测试设计

### 4.3.1 测试工程结构

参照 `test/rot/srot/arch35` 新建 `test/rot/crot/arch35`：

| 文件 | 作用 |
| --- | --- |
| `crot_test.cpp` | GTest 主程序：加载 CSV、分配 Device 缓冲、调用 `aclblasCrot_npu`、CPU golden 比对、精度判定 |
| `crot_param.h` | CSV 参数解析：n、x/y 填充模式、incx/incy、c、s_real/s_imag、expect_result、mere/mare、random_seed、cs_ptr_mode |
| `crot_golden.h` | CPU golden：按 Netlib crot 语义（含 conjg(s)、原始 x）实现，不依赖 cblas 的 crot（BLAS 无该例程） |
| `crot_npu_wrapper.h` | 封装：Host→Device 拷贝、c/s 的 host/device/mixed 位置构造、D2H 回读、同步 |
| `crot_test.csv` | 随任务提供的 1200 条用例，可直接使用或按需扩充 |
| `README.md` | 测试步骤、编译命令、CSV 列说明、精度/性能判定口径，保证验收人可复现 |
| `CMakeLists.txt` | `ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)` |

### 4.3.2 用例分类

| 类别 | 前缀 | 条数 | 覆盖 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 8 | 小尺寸 × ±1/±2 步长 |
| L1 尺寸 | TC_SQ | 38 | 1→1048576，含质数、2 的幂 ±1、非对齐 |
| L2 步长 | TC_INC | 36 | incx × incy 全组合（±1/±2/±3） |
| L3 标量 | TC_CS | 20 | 单位旋转、90°、45°、复数 s、纯虚 s、负 c、大值、极端值 |
| L5 填充 | TC_FL | 12 | 均匀/全零/交替/极端/Inf/NaN |
| L5b 指针位置 | TC_PM | 6 | c/s 的 host/device/mixed |
| L6 边界 | TC_ED | 10 | n=0/负 n no-op；零步长；空指针 |
| EX 扩展 | TC_EX | 870 | 尺寸 × 步长 × 标量 × 指针位置确定性采样 |
| PF 性能 | TC_PF | 200 | 3 条标杆 + 小尺寸 + 规模扫描，全部 inc=1 |

**输入分布实现**：任务书要求 x/y 的 50% 用例取均匀分布 [-5,5]、50% 取正态分布（μ∈[-5,5]，σ∈[0.1,2]），实部/虚部独立采样；c、s 同样要求均匀 + 正态并含特殊值。

经核对，ops-blas `test/frame/fill.h` **已内置** GAUSS 模式（`M_GAUSS` / `GaussGenerator`），语法为 `GAUSS_<muHi>_<sigHi>`，语义为 μ∈[-muHi,muHi]、σ∈[0.1,sigHi]、元素截断到 [-muHi,muHi]；仓内 `test/gbmv/cgbmv/arch35` 已有 `GAUSS_2.422_1.518` 等实例。因此**无需扩展测试框架的 fill 模式**，只需让生成器使用现有模式：

- `gen_csv.py` 默认 `--dist mixed`：x/y 张量交替使用 `RANDOM_NORM_5_5` 与 `GAUSS_5_2`，各占 50%；c/s 标量按同一比例生成随机值，`TC_CS` 保留特殊值全集。
- 该实现按**用例级 50/50**（半个用例集均匀、半个正态）。若验收要求“单个 tensor 内 50% 元素均匀 + 50% 元素正态”，则需新增 MIXED fill 模式；当前口径需在测试 README 中写明。
- `crot_test.csv` 已用 Python 3.12.7 重新生成并自检：x 的 `GAUSS_5_2` 为 500/994、y 为 493/994（各约 50%），特殊填充与其他类别按原规则保留，性能用例 200 条全部 `incx=incy=1`；原始均匀 CSV 备份为 `crot_test.uniform.bak.csv`。

### 4.3.3 补充测试

CSV 无法表达的用例，用独立 `TEST_F` 补充（对齐 srot）：

| 用例 | 期望 |
| --- | --- |
| `NullHandle` | HANDLE_IS_NULLPTR |
| `NullHandleTakesPrecedenceOverQuickReturn` | `handle=nullptr, n=0` → HANDLE_IS_NULLPTR |
| `QuickReturnSkipsDataPointerValidation` | `n=0/-1` 且全空指针 → SUCCESS |
| `NullCsRejected` | c 或 s 为 nullptr（n>0）→ INVALID_VALUE |
| `StrideOverflowRejected` | `(n-1)*inc` 超 int32（inc=INT32_MAX/INT32_MIN）→ INVALID_VALUE |

c/s 的 host/device/mixed 组合由 CSV 的 `TC_PM` 用例覆盖，不另设 `TEST_F`。

### 4.3.4 Golden 实现

Netlib 的 `crot.f` 是标准 BLAS 例程（CBLAS 绑定为 `cblas_crot`），但任务书已明确“参考 BLAS 库无 crot 例程”，且目标环境是否导出该符号无法预先保证；仓内复数算子（如 caxpy）也采用标量 golden 而非 cblas。因此测试工程在 `crot_golden.h` 内按 Netlib crot 语义实现等效 golden，使用 **FP32** 计算并按算子相同的“一次乘法 + 两次 fma”顺序展开。这样在 `FLT_MAX` 极值输入下，golden 与任意 FP32 实现（含 cublas）一样发生中间溢出，Inf/NaN 分布一致；若使用 double 累加，极值乘积不溢出，会出现“golden 有限、实际 Inf/NaN”的假失败。

```cpp
for (int i = 0; i < n; ++i) {
    int64_t ix = incx < 0 ? int64_t(n - 1 - i) * (-int64_t(incx)) : int64_t(i) * incx;
    int64_t iy = incy < 0 ? int64_t(n - 1 - i) * (-int64_t(incy)) : int64_t(i) * incy;
    float xr = x[ix].real, xi = x[ix].imag, yr = y[iy].real, yi = y[iy].imag;
    float xrOut = std::fma(sr, yr, c * xr);
    xrOut = std::fma(-si, yi, xrOut);
    float xiOut = std::fma(sr, yi, c * xi);
    xiOut = std::fma(si, yr, xiOut);
    float yrOut = std::fma(-sr, xr, c * yr);   // 使用原始 x
    yrOut = std::fma(-si, xi, yrOut);
    float yiOut = std::fma(-sr, xi, c * yi);
    yiOut = std::fma(si, xr, yiOut);
    x[ix] = {xrOut, xiOut};
    y[iy] = {yrOut, yiOut};
}
```

零步长时 `ix/iy` 固定为 0，严格串行累积。精度判定采用任务书 §3.2 的混合容差（实部、虚部分别判定）：`|actual-golden| ≤ atol + rtol*|golden|`，`rtol = atol = 2^-13`，并要求 `matched_ratio ≥ 0.99`、逐元素误差不超过 `max(1e-2, 32×ULP)`。NaN/Inf 用例的判定交给 `MixedToleranceStrategy`：两侧同为 NaN 或完全相等（含同号 Inf）时跳过；仅一侧为特殊值时判失败，从而同时覆盖传播正确性与特殊值不匹配。

### 4.3.5 复现命令

```bash
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./crot_test.csv
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --skip-build --device 1
python verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600
msprof op --application="./build/test/rot/crot/crot_test --gtest_filter=*TC_PF*" --output=./prof_crot
```

`TC_PF_*` 在 GTest 内走独立性能分支（与仓内 scasum 一致）：Device 缓冲一次性分配，warmup 5 次后连续下发 100 次、末尾同步一次，打印 `[case] avg_single_us=...`，并跳过 CPU golden，避免 host 端计算主导计时；同形状的正确性由其余约 1000 条精度用例覆盖。

### 4.3.6 自测报告内容

- 用例参数表（n、incx/incy、标量、填充、指针位置）
- 实部/虚部分别比对的精度结果与截图
- 三条标杆 case 的性能数据与截图
- 内存占用（本算子无 workspace，记录输入输出缓冲）
- 失败用例的原始数据与定位

### 4.3.7 950PR 实测结果

环境：Ascend 950PR（NPU 0，HBM 131072 MB）、CANN 9.1.0、Ubuntu 22.04、x86_64。

**精度**：`--gtest_filter=-*TC_PF*` 共 1005 条（1000 条 CSV + 5 条独立边界用例）全部通过。
TC_EX 极值标量组 871 条在 SIMT 显式 FMA 修正后 `maxAbsErr=0`，与 CPU golden 逐位一致；
AIV 连续路径（含 `FLT_MAX`/`Inf`/`NaN` 填充的 TC_FL 组）同样全部通过。首轮 43 条 SIMT
失败（42 条 TC_EX + 1 条零步长）已定位为逐项舍入与 golden 的 FMA 顺序差异，详见 3.4.2。

**性能**（`TC_PF_1001/1002/1003`，`incx=incy=1`，warmup 5 + 100 次连续下发）：

| n | 测试内平均（含 host 下发） | msprof Task Duration 中位数 | 任务书门限 | 余量 |
| --- | --- | --- | --- | --- |
| 1,048,576 | 9.68 us | 9.85 us（min 9.39） | 21.96 us | 2.27× |
| 2,097,152 | 15.84 us | 16.04 us（min 15.54） | 43.70 us | 2.76× |
| 4,194,304 | 28.14 us | 28.46 us（min 27.83） | 119.52 us | 4.25× |

注：

- `gpu_baseline.csv` 的 `gpu_ms` 尚未回填，`verify_performance.py` 将判定为 NO_REF；上表门限取任务书给定的 NPU 目标值。
- 本机 `msprof op`（动态插桩）报 `Profiling channel start failed`，改用 `msprof` 任务级采集读取 `op_summary` 的 `Task Duration(us)`；每条 case 采集 106 次 kernel（1 次 wrapper + 5 warmup + 100 采样）。
- 200 条 TC_PF 全部通过；小 n 单次约 1.9~3.2 us，耗时随 n 单调增长。
- 连续 100 次下发时 x/y 工作集（≤67 MB）小于 L2 容量，表中是 L2 辅助下的稳态值；任务书同样按“连续多次下发取平均”验收，口径一致。

**调试中发现并修复的问题**：

- SIMT 标量路径初版用 `c*xr + sr*yr - si*yi`，实测呈逐项舍入行为，与 golden 的 FMA 序列在 `c=1e10` 极值组相差 1~2 ULP 并被放大，43 条用例触发逐元素 `32×ULP` 上限；改为显式 `fmaf` 后 0 失败。
- AIV 输出初版用 4 个 TBuf + 手动 `FetchEventID/SetFlag/WaitFlag` 做 ping-pong；单次调用正常，但“warmup 后同步、再连续下发”时 NPU 100% 空转（首条 1M 性能用例 60 s 未完成）。改为 `TQue<VECOUT, 2>` 后同步交给框架，200 条 PF 全部通过。

## 4.4 兼容性分析

- 新增接口，不修改已有 `aclblasSrot`、`aclblasCsrot` 行为。
- 仅在 `include/cann_ops_blas.h` 增加声明，不引入私有头文件或平行 API。
- 新增源文件放在 `blas/rot/arch35/`，CMake 自动收集；ascend910b/910_93 构建时不会收集 arch35 源文件，不影响 A2/A3 产物。
- README 增加 `aclblasCrot` 章节与产品支持表，其他算子章节不变。

---

# 五、性能优化与设计审查

本章回答两个问题：**当前设计的瓶颈在哪里**，以及**还有哪些可以优化的地方**。结论按优先级分为 P0（已纳入设计）、P1（建议实施并用实测决定）、P2（备选或明确排除）。

## 5.1 流量与开销模型

### 5.1.1 数据流量

| n | 单向量字节 | 4 条流总流量 | 标杆耗时 | 需求带宽 |
| --- | --- | --- | --- | --- |
| 1,048,576 | 8 MiB | 32 MiB | 21.96 us | ≈1.53 TB/s |
| 2,097,152 | 16 MiB | 64 MiB | 43.70 us | ≈1.54 TB/s |
| 4,194,304 | 32 MiB | 128 MiB | 119.52 us | ≈1.12 TB/s |

本算子的理论最小流量就是“读 x、读 y、写 x、写 y”，没有 workspace、没有 golden 在 kernel 内、没有额外中转。设计已做到最小流量，**不应再引入任何额外的 GM 读写**。

### 5.1.2 计算量

每个 tile 采用 12 条向量指令（4 `Muls` + 8 `Axpy`），每条处理 `tileSize` 个 float。

以 n=1M、32 个 AIV 核、64 lane/指令、1.65 GHz 估算：

```
总 lane 操作 ≈ 12 × 1,048,576 = 12.58 M
每核        ≈ 393 K lane 操作
每核周期    ≈ 393K / 64 ≈ 6,144 cycle
每核计算时间 ≈ 6,144 / 1.65GHz ≈ 3.7 us
```

对比内存时间约 20 us，计算约占 19%。**结论：算子是 memory-bound，主要矛盾是 GM 带宽与搬算重叠，不是向量指令数。** 这决定了优化顺序：

1. 先保证 4 条流走满带宽（对齐、连续、全核、双缓冲）。
2. 再考虑减少向量指令（20→12 已做）。
3. 不建议为 v1 引入 RegBase 全寄存器链路，收益上限受内存带宽限制。

### 5.1.3 固定开销

固定开销包括：接口参数校验、2 次 `aclrtPointerGetAttributes`、Tiling 计算、kernel 下发、测试框架的 H2D/D2H 与 golden。小 n 时固定开销占比高，优化重点是**不增加每次调用的 Host 侧分配与同步**。

## 5.2 已纳入设计的优化（P0）

| 编号 | 优化点 | 依据/证据 | 预期收益 | 风险 |
| --- | --- | --- | --- | --- |
| P0-1 | 去掉 workspace malloc、Tiling H2D、接口内 stream 同步 | `csrot` arch22 的 Host 模式每次调用 2 次 malloc + 1 次 H2D + 1 次同步；`srot` arch35 已改为 TilingData 传值 | 小 n 显著；大 n 减少 Host 抖动 | 无，功能等价 |
| P0-2 | TQue 双缓冲 + 预取循环 | srot arch35 为单缓冲（`InitBuffer(..., 1, ...)`）；crot 计算量更大、4 条流，单缓冲会串行 | 搬算重叠，理论吞吐提升 20%~50%；目标 case 依赖此项 | UB 翻倍，已用 22×tile 预算约束 |
| P0-3 | 每核连续区间 + 全核并行 + 空核早退 | srot 的 even split；连续区间保证 GM 合并访问 | 负载均衡、合并访存 | 区间必须精确拼接，已用 unit 计算保证 |
| P0-4 | 输出双缓冲（x'/y' 各一个深度 2 的 TQue） | caxpy 用 3 个 AoS + 6 个平面；crot 用 2 个输出队列，保证 MTE3 与下一块 Vector 重叠 | 输出搬运与计算重叠，避免单输出缓冲的串行等待 | UB 增加 8×tile；需保证原始 x 分量在写 y 前不被覆盖 |
| P0-5 | 最小 4 条流、无 workspace | 原地算子，无临时 GM | 达到理论最小流量 | 无 |
| P0-6 | 32B 对齐 `DataCopy` 快路 + `DataCopyPad` 尾块 | `DataCopy` 要求严格 32B 对齐，`DataCopyPad` 开销更高 | 主路径减少搬运开销 | 对齐条件不满足时退回 Pad，功能不受影响 |
| P0-7 | 零步长在 VF 入口守卫，不在并行循环内判断 | SIMT Rule 001：整次调用一致的条件应提升到 VF 外 | 非 block0.thread0 立即 return，保证串行语义 | 单 VF 内多一层 early-return |
| P0-8 | 不使用 `SetValue/GetValue` | API 黑名单，效率极低 | 避免标量逐点访存 | 无 |
| P0-9 | 不做 L2 bypass | 1M/2M 工作集 32/64 MiB < L2 约 112 MB，重复采样可能 L2 常驻 | 保留标杆达标的关键余量 | 需用 msprof 确认命中情况 |

## 5.3 建议实施的二级优化（P1，用实测决定）

| 编号 | 优化点 | 当前设计 | 建议动作 | 预期收益 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | 向量指令 20→12 | 默认 4 `Muls` + 8 `Axpy` | 先跑精度；若 45°/抵消场景超限，`y'` 腿改逐项舍入 | 计算时间约 40%，总时延约 5%~8%（memory-bound 下） | 混合容差 + msprof VEC 时间 |
| P1-2 | 32B 对齐分核 | `unitSize=8` 已有 | 确保标杆 n 全走 `DataCopy`；用日志确认 `alignedCopy=1` | 减少搬运指令开销 | 对比 `DataCopyPad` 版本 |
| P1-3 | 平面缓冲 bank 对齐 | 连续排布 | 在 `xr/xi/yr/yi/t0/t1` 基址间加 32B/64B padding，错开 bank | 若存在 bank conflict，预计 0~10% | msprof bank/cycle + A/B |
| P1-4 | SIMT 短向量访存 | 逐 float 读写 | 使用 `float2` 一次读写一个复数；相邻线程按 8B 连续 | stride 路径指令数减半 | 仅功能路径，抽查时延 |
| P1-5 | 小 n 自适应核数 | `numBlocks=min(units, cores)` | 对 n < 阈值（如 4096）减少 block 数，避免“一核仅几个复数”的启动浪费 | 小 n 固定开销下降 | 小 shape 扫描 A/B |
| P1-6 | Tile 大小调参 | 2560 复数 | 在 2048 / 2560 / 2816 间 A/B；以实测最优为准 | 可能 2%~5% | 三条标杆 case |
| P1-7 | Host 标量查询开销 | 2 次 `aclrtPointerGetAttributes` | 测量单次开销；确认没有重复校验/重复查询；不做指针缓存（不安全） | 避免固定开销吃掉小 n 收益 | 端到端 vs kernel 时间差 |
| P1-8 | 尾块均摊 | 全局尾块由 0 号核处理（<8 个复数） | 保持现状；若尾块超过一个 tile，再改为多核 Pad | 当前尾块极小，收益有限 | 非 2 的幂 n 用例 |

**P1 实施纪律**

- 一次只改一个变量；每次保留带实测时延的备份版本。
- 先看 `msprof` 的 kernel 时间与 pipe 占比，再决定改搬运还是改计算。
- 不用“文档建议”直接定参数；核数、tile、线程数都以目标机实测为准。
- 小 shape 与三条标杆分别验证，避免为大 n 优化牺牲小 n。

## 5.4 备选路线与排除理由（P2）

| 路线 | 结论 | 理由 |
| --- | --- | --- |
| RegBase / VF 全寄存器链路（`register-resident-vector-chain`） | v1 不采用，保留为备选 | 已检索 arch35 VF 范例库：该条目为 `integration_skeleton`，适用于 VEC 主导的向量链。本算子计算约占内存时间 19%，瓶颈在 GM 带宽；全寄存器化只能减少 UB 读写，不能减少 4 条 GM 流。且仓内 rot 族使用经典 membase API，引入 RegBase 会增加模板适配与版本风险。若 profiling 显示 VEC 占比 > 40%，再按范例库前置条件评估 |
| `paired-segment-rotate`（RoPE 类） | 排除 | 该范例针对前后分段旋转（`x*cos + rotate(x)*sin`）；crot 是两个独立向量的 Givens 旋转，数据布局与语义不同，不能硬套 |
| 三缓冲 | 排除 | 三缓冲用于 Cube+Vector 融合；本算子纯 Vector，双缓冲足够 |
| L2 Cache bypass | 排除 | 工作集小于 L2 且重复采样，bypass 会让标杆变慢；仅在数据量远超 L2 时才有收益 |
| `GatherMask` 解交织 | 排除 | `DeInterleave/Interleave` 是 arch35 原生重排指令，仓内 caxpy 已验证；GatherMask 主要用于 A2/A3 的奇偶拆分 |
| 直接在交错 AoS 上算（邻居交换 + 符号向量） | 排除 | 需要额外的 swap/符号乘法，向量指令数多于“解交织 + 平面计算 + 交织”，且可读性差 |
| 单位旋转 `c==1,s==0` 短路 | 排除 | 破坏 Netlib 语义；srot 明确不短路 |
| 缓存 c/s 指针位置 | 排除 | 指针位置可能随调用变化，缓存会导致错误；2 次属性查询是可接受成本 |
| 接口内同步 | 排除 | 破坏异步语义；调用方负责同步 |
| 多流并行 | 不适用 | 单 handle 单 stream，数据依赖同一对 x/y；拆流会增加同步与复杂度 |

## 5.5 优化验证计划

1. **基线**：先用当前设计跑通精度与三条标杆，记录 end-to-end 与 msprof kernel 时间。
2. **瓶颈判定**：比较 kernel 时间与理论内存时间；若差距大，看 MTE2/MTE3/VEC 占比。
3. **按序实施**：P1-2（对齐）→ P1-6（tile）→ P1-1（指令数）→ P1-3（bank）→ P1-5（小 n）→ P1-4（stride）。
4. **A/B 规则**：同一 case、同一环境、同一采样协议；warmup 5 次、>50 次平均；一次只动一个变量。
5. **停止条件**：当剩余时间主要由 kernel 启动/下发固定开销构成，且没有源码级手段可压时定稿；不再为 1%~2% 消耗提交次数。
6. **回归**：每次优化后重跑全部精度用例，尤其是 45°、共轭、负步长、零步长、device 标量。

---

# 六、风险评估与应对

| 编号 | 风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| R1 | `Axpy` 的 FMA 舍入与逐项舍入参考不一致，45°/近抵消场景相对误差放大 | 精度用例失败 | golden 已采用与 kernel 相同的 fma 展开；若外部参考为逐项舍入版本，保留 `Muls + Add/Sub` 备选路径，按混合容差实测切换 |
| R2 | 零步长多线程并发读写同一地址，结果与 Netlib 串行累积不一致 | 功能错误、结果不稳定 | 入口级串行回退（block0.thread0）；专项用例覆盖 incx=0、incy=0、两者为 0 |
| R3 | `(n-1)*inc` 在 int32 溢出，kernel 地址偏移越界 | 非法访存 | Host 用 int64 计算并检查上下界，超界返回 INVALID_VALUE |
| R4 | c/s 指针位置判定错误或混合位置处理不当 | 读到错误标量/非法访问 | 分别查询、分别记录标志；host/device/mixed 三类用例 |
| R5 | `DeInterleave/Interleave` 的 membase 签名或 CANN 9.1 可用性与仓内版本不一致 | 编译失败/结果错误 | 开发前按 CANN 9.1 API 文档复核；以 caxpy 为编译基线；必要时退回 `DataCopyPad` 分列搬运 |
| R6 | UB 预算与 tile 公式在 Host/Kernel 各写一套，导致越界或浪费 | 编译失败/运行时异常 | Host 计算 tileSize 写入 TilingData；Kernel 只用传入值；`static_assert` 兜底 |
| R7 | 标杆 case 1/2 需要约 95% 峰值带宽，难达标 | 性能验收失败 | 双缓冲 + 对齐 + 全核 + 保留 L2；尽早 msprof；必要时调 tile/核数 |
| R8 | 仓内无 crot CSV 测试工程，需新建 | 测试无法复现 | 照 srot 模式搭建；直接使用任务提供的 1200 条 CSV；补充独立 TEST_F |
| R9 | 头文件声明与已有接口冲突或引入私有 API | 评审不通过 | 严格使用任务书签名；只改公共头文件；不新增私有头 |
| R10 | 交付件遗漏（README 支持表、测试 README、自测报告、私仓邀请） | 验收延迟 | 按第 7 章清单逐项勾选 |
| R11 | 输入分布口径不一致（用例级 50/50 vs 单 tensor 元素级 50/50） | 用例覆盖被判不足 | 默认按用例级 50/50 生成并在测试 README 写明口径；如验收要求元素级混合，再新增 MIXED fill 模式 |

---

# 七、交付件清单

| 序号 | 交付件 | 具体要求 | 状态 |
| --- | --- | --- | --- |
| 1 | 算子设计文档 | 本文档放入 `04_tasks/01_community-task-2026/tasklist/{任务目录}/{TeamName}/docs/design.md`，以 PR 提交评审 | 本文 |
| 2 | 算子代码 | `include/cann_ops_blas.h` 声明；`blas/rot/arch35/crot_host.cpp`、`crot_kernel.cpp`、`crot_kernel.h`、`crot_tiling_data.h` | 待开发 |
| 3 | 测试代码 | `test/rot/crot/CMakeLists.txt`、`arch35/crot_test.cpp`、`crot_param.h`、`crot_golden.h`、`crot_npu_wrapper.h`、`crot_test.csv`、测试 `README.md`（复现步骤） | 待开发 |
| 4 | 算子 README | 更新 `blas/rot/README.md`：函数原型、参数、约束、调用示例；产品支持表标注 Ascend 950PR：支持 | 待更新 |
| 5 | 自测报告 | 用例参数、实部/虚部精度结果与截图、性能数据与截图、内存占用 | 待生成 |
| 6 | 待验收代码地址 | 个人仓链接、分支、算子目录；邀请账号 `Ascend-CANN` 为开发者 | 待提交 |
| 7 | 上游 PR | 实现合入 `blas/rot/arch35/`，测试合入 `test/rot/crot/arch35/` | 待提交 |

---

# 八、迭代规划

| 迭代 | 目标 | 代码 | 测试 | 出口条件 |
| --- | --- | --- | --- | --- |
| 迭代一 | 骨架与连续路径 | 头文件声明、Host 校验、TilingData、tilingKey=0 单缓冲版 | CSV 框架 + L0 基础用例 | 连续路径功能正确，小 shape 精度通过 |
| 迭代二 | 双路径与边界 | 步长 SIMT、零步长串行、host/device 标量、溢出检查 | L2/L3/L5b/L6 全覆盖 | 全部精度用例通过，负向行为正确 |
| 迭代三 | 性能与优化 | 双缓冲、对齐 DataCopy、Axpy 指令压缩、tile 调参 | PF 200 条 + msprof | 三条标杆达标，优化 A/B 有记录 |
| 迭代四 | 交付与合入 | README、自测报告、代码清理 | 全量回归 | 交付件齐全，PR 通过评审 |

---

# 九、参考资料

1. 任务广场：`https://www.hiascend.com/activities/task-center/details/b6f4d038761348f5955c0abd6d7e5e85`
2. 活动留言板：`https://gitcode.com/org/cann/discussions/318`
3. 任务书：`https://www.hiascend.com/p/resource/202610/9d50463f8a8b4cbfa5ef6ea92995e451.zip`
4. ops-blas 开源仓：`https://gitcode.com/cann/ops-blas`
5. 算子设计文档模板：`https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md`
6. 社区任务提交规范：`https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md`
7. 生态算子开源精度标准：`https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md`
8. cuBLAS `cublasCrot`：`https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rot`
9. Netlib/LAPACK `crot`：`https://www.netlib.org/lapack/explore-html/da/df6/group__complex__blas__level1_ga4b55e64b6b4f48e1c6e25f3dc53b5b06.html`
10. Ascend C 算子开发文档：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html`
11. 算子开发接口文档：`https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html`
12. `asc_vf_call`：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00267.html`
13. `DeInterleave`：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0406.html`
14. `Interleave`：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0405.html`
15. 统一寻址与 `aclrtPointerGetAttributes`：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/API/runtimeapi/docs/zh/api_ref/11-05_unified_addressing.md`

## 附：仓内取证文件

| 结论 | 取证文件 |
| --- | --- |
| Host 校验顺序、标量定位、TilingData 传值 | `blas/rot/arch35/srot_host.cpp` |
| 连续 SIMD + 步长 SIMT、零步长串行回退 | `blas/rot/arch35/srot_kernel.cpp` |
| 复数交错布局与 `DeInterleave/Interleave` | `blas/axpy/arch35/caxpy_kernel.cpp` |
| UB 248KB、SIMT 线程上下限 | `blas/common/helper/kernel_constant.h` |
| 头文件 rot 族声明 | `include/cann_ops_blas.h` |
| 复数类型定义 | `include/cann_ops_blas_common.h` |
| CSV 测试与 golden 模式 | `test/rot/srot/arch35/srot_test.cpp`、`srot_param.h`、`srot_golden.h` |
| arch35 构建路由（ascend950） | `build.sh`、`blas/CMakeLists.txt` |

