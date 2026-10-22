# aclblasCrot 算子设计文档

> 本文档依据 `aclblasCrot_Atlas950PR_task_doc.md`（Ascend 950PR 任务书）与《昇腾社区任务开发与交付规范》（ASCEND-TASK-SPEC-001）编写，面向社区任务评审。
> 算子实现提交至 ops-blas 仓 `blas/rot/arch35/`，测试代码提交至 `test/rot/crot/arch35/`。

---

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 — 算子开发任务 `aclblasCrot`（Atlas 950PR）。

- 任务书：`aclblasCrot_Atlas950PR_task_doc.md`
- 上游仓库：https://gitcode.com/cann/ops-blas ，目标目录 `blas/rot/arch35/`
- 接口声明：`include/cann_ops_blas.h`
- 语义基线：cuBLAS `cublasCrot` / Netlib LAPACK `crot.f`

## 背景介绍

### 现状

ops-blas 仓的 `rot` 算子族提供 Givens 平面旋转。实数版本 `aclblasSrot` 已完成 arch35（Ascend 950PR）适配；复数版本 `aclblasCsrot` 的 arch35 适配由同族任务覆盖；**复数正弦版本 `aclblasCrot` 在 arch35 上尚无实现，且公共头文件中没有声明**。本任务基于 Ascend C 新增该算子。

当前仓库与本任务的缺口：

1. **接口缺失**：`include/cann_ops_blas.h` 只有 `aclblasCsrot`（c/s 均为实数）与 `aclblasSrot`，没有 `aclblasCrot`；后者的 `s` 为复数，需要新增声明并保证可与其它产品线共用。
2. **无 arch35 实现**：`blas/rot/arch35/` 只有 `srot_{host,kernel,tiling_data,kernel.h}`，无复数 rot 算子。
3. **参考实现非标**：Netlib/LAPACK `crot.f` 的 `y` 更新使用 `conjg(s)`，且必须使用旋转前的原始 `x`；该语义无法由实数 `srot` 直接套用。
4. **无 cblas 参考**：参考 BLAS 库不提供 `crot` 例程（只有 `cblas_srot`/`cblas_csrot`/`cblas_crotg`），因此测试工程的 CPU golden 需按 `crot.f` 语义自行实现。

### 官方来源与参考基线

- 任务书 §2.1 给出公式与 `conjg(s)` 语义；§3.2 给出精度标准；§3.3 给出性能标杆；
- Netlib/LAPACK `SRC/crot.f`（已核对原文）：`x' = c*x + s*y`；`y' = c*y - conjg(s)*x`（使用旋转前 x）；
- 参考基线实现：同族 `blas/rot/arch35/srot`（实数双路径，工程脚手架）、`blas/rot/arch22/csrot`（复数布局参考）。

### 算子功能分析

对两个等长复数向量 `x`、`y` 做原地 Givens 旋转，旋转因子 `c` 为**实数**、`s` 为**复数**：

```
x'[i] = c * x[i] + s * y[i]
y'[i] = c * y[i] - conjg(s) * x[i]      (使用旋转前的 x[i])
```

与 `csrot` 的关键区别：`s` 是复数、`y` 的更新含共轭，因此存在复数乘法的实/虚交叉项，不能像 `csrot` 那样把复向量当作 2n 个独立 float lane 处理。典型用途同 rot 族：QR 分解等数值算法中的平面旋转。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `aclblasCrot` 算子：

- 数据类型：complex64（单精度复数向量，实部/虚部各 float32）+ float 实数 `c` + complex64 `s`
- 适配硬件：Ascend 950PR / 950DT（arch35），CANN 9.1.0
- 完整对齐 netlib `crot.f` 的边界语义与 `conjg(s)` 语义
- 性能不低于任务书 §3.3 的标杆耗时

## 需求拆解

1. **接口声明**：`aclblasCrot(aclblasHandle_t, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c, const aclblasComplex* s)`，加入 `include/cann_ops_blas.h`，禁止 950PR 私有平行接口；
2. **原地运算**：x、y 既是输入又是输出，`y'` 必须使用旋转前 `x`；
3. **边界语义对齐 netlib**：`n <= 0` 为 no-op 且不校验其余参数；`handle == nullptr` 优先返回；`inc` 可正/负/零，负步长从尾端起算、零步长对锚点元素串行旋转；
4. **标量指针位置自动判定**：`c`、`s` 可分别位于 host 或 device，二者独立；
5. **int32 溢出防护**：`(n-1)*|inc|` 超 int32 返回 `INVALID_VALUE`；
6. **测试工程**：新建 `test/rot/crot/arch35/`，原样使用任务包 1200 条 CSV 用例（1000 精度 + 200 性能）；
7. **精度/性能达标**：精度按实部/虚部分别判定；性能满足 21.96 / 43.7 / 119.52 us。

# 详细设计（required）

## 对象与执行器分析

### 计算与复用边界

- 复用：ops-blas 的 handle/stream 基础设施、`CheckPtrLocation`、`GetAivCoreCount`、`CeilDiv/CeilAlign`（`blas/common/helper/`）；
- 复用：同族 `srot` 的 SIMT 启动形态（`__simt_vf__` + `asc_vf_call` + `KERNEL_TASK_TYPE_DEFAULT`）；
- 不修改：`blas/rot/arch22/` 的 `csrot`、`blas/rot/arch35/` 的 `srot`、其它算子与公共代码；
- 新增：`blas/rot/arch35/crot_{host,kernel,tiling_data,kernel.h}` 与 `test/rot/crot/`。

### 支持数据类型与配置

| 参数 | 类型 |
|------|------|
| x, y | complex64（`aclblasComplex`，8 字节：float real + float imag，交错存储） |
| c | float32 |
| s | complex64（2 个连续 float） |
| 计算精度 | float32 |
| n / incx / incy | int（Host） |
| handle | aclblasHandle_t（携带 stream） |

### 输入输出与异常

- x、y 原地更新（in-place），不返回视图；
- 逻辑一维 `[n]`，物理跨度 `1+(n-1)*|inc|` 个复数；
- 异常返回：

| 条件 | 返回 |
|---|---|
| handle == nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（优先于一切） |
| n <= 0 | `ACLBLAS_STATUS_SUCCESS`（no-op，不校验 x/y/c/s） |
| n > 0 且 x/y/c/s 任一为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| `(n-1)*|inc|` 超出 int32 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `aclrtPointerGetAttributes` 查询失败 | `ACLBLAS_STATUS_INVALID_VALUE` |

## 实现方案

### 主机侧设计

文件：`blas/rot/arch35/crot_host.cpp`、`crot_tiling_data.h`、`crot_kernel.h`

1. 参数校验与 quick return（顺序见上表）；
2. 标量指针位置判定：`CheckPtrLocation` 分别判断 `c`（1 个 float）与 `s`（2 个连续 float）位于 host 还是 device；host 时解引用写入 tiling，device 时指针原样转发、不做 D2H、不额外同步；
3. tiling（`CrotTilingData`）：`totalN`、`incx/incy`、`nthreads`、`cIsDevice/sIsDevice`、`cosValue/sRealValue/sImagValue`；
4. 分核：`numBlocks = min(ceilDiv(n, SIMT_MIN_THREAD_NUM), aivCoreNum)`；`nthreads = min(alignUp(ceilDiv(n, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)`；核数运行时获取；
5. stride 范围检查：`(n-1)*|incx|`、`(n-1)*|incy|` 在 int64 中计算，超 `INT32_MAX` 拒绝；
6. 启动：`<<<numBlocks, nullptr, stream>>>` 直调，无 workspace。

### 设备侧设计

文件：`blas/rot/arch35/crot_kernel.cpp`，单 SIMT 路径覆盖全部步长组合：

- `__global__ __aicore__` 入口 + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，`asc_vf_call<crot_simt_compute>` 启动每核 SIMT 线程；
- 每线程一次 `float2`（8 B）读写一个复数元素，grid-stride 循环；
- 寻址：`inc>0 → jx=i`；`inc<0 → jx=n-1-i`；`inc==0 → jx=0`；float 偏移 `2*jx*|inc|`；
- 零步长：仅 `block0.thread0` 串行执行 n 轮，复现 netlib 串行累积；
- 每元素先读 `xr/xi/yr/yi` 到寄存器，算完先写 x、后写 y。

### 内存生命周期

- 原地操作，无 workspace、无临时设备内存申请；
- kernel 内原始值在寄存器中存活至两次写回完成，x/y 别名安全；
- Host 侧 tiling 为按值传参，无长期驻留。

## 支持硬件与环境

| 项 | 值 |
|---|---|
| 适配芯片 | Ascend 950PR（arch35 / dav-3510）；950DT 同架构 |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`） |
| OS / 架构 | Ubuntu 22.04.5 LTS / x86_64 |
| 编译器 | gcc 11.4.0；CANN 自带 bisheng |
| 测试框架 | ops-blas `build.sh --soc=ascend950 --ops=crot` + C++ GTest |

## 约束限制

- `n` 为 `int`；`n <= 0` 合法 no-op；
- `(n-1)*|incx|` 与 `(n-1)*|incy|` 必须落在 int32 范围内，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`；
- `c` 为实数 float32，`s` 为 complex64；`c == 1 && s == 0` 的单位旋转不短路；
- 原地运算，`x`/`y` 必须位于 Device 内存；
- 零步长按 netlib 串行语义实现，大 n 零步长性能不作保证；
- 本任务仅交付并验证 Ascend 950PR；A2/A3 不支持。

# 可维可测分析

## 精度/性能/内存标准与判定方法

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 实部/虚部分别按 FLOAT32 判定：rtol = atol = 2^-13，matched_ratio ≥ 0.99，case-level max_abs_error ≤ 1e-2 | 任务书 §3.2；生态算子开源精度标准 |
| 精度判定方法 | 测试工程 `crot_golden.h` 按 netlib `crot.f` 语义以 FP32 + 显式 `std::fma` 实现 golden；逐元素 `|actual-golden| ≤ atol + rtol*|golden|`，逐用例统计 matched_ratio 与 max_abs_error | 2.1 测试方案 §2.3 |
| 性能 | n=1048576 / 2097152 / 4194304（incx=incy=1）平均单次耗时 ≤ 21.96 / 43.7 / 119.52 us | 任务书 §3.3 |
| 性能判定方法 | 常驻设备 buffer、正确性预检、warmup 5 次、连续 100 次调用、循环结束单次同步取平均；另报 min/max | 2.1 测试方案 §7.5 |
| 内存 | 不涉及（无 workspace） | 任务书 §3.4 |

## 已完成自验数据（真实数据）

| 项 | 实测 |
|---|---|
| 全量非性能用例 | 1024 / 1024 PASS |
| 任务包 `verify_accuracy.py` | PASS=1000 / FAIL=0 / ALL PASS |
| 全量用例（含性能过滤） | 1227 / 1227 PASS |
| 性能 1M | 12.50 us（标杆 21.96，1.76x） |
| 性能 2M | 21.17 us（标杆 43.70，2.06x） |
| 性能 4M | 39.55 us（标杆 119.52，3.02x） |
| 白盒分支 | 59 条活分支覆盖 57 条（2 条已论证不可黑盒覆盖） |

## 证据获取与待完成项

**证据**（随自测报告提交）：

- 精度原始日志、性能原始日志、构建日志；
- 复现命令与固定提交：`xiang_ye1/ops-blas` 分支 `feat/crot-arch35`；
- 测试工程 README：`test/rot/crot/README.md`（构建/运行/过滤/复现步骤）。

**待完成项**：

1. 设计文档 PR 评审通过并合入（G1）；
2. 平台【提交验收】与验收测试（G2）；
3. 验收通过后提交代码 PR 与需求 issue（G3）。

## 兼容性分析

接口 `aclblasCrot` 的参数序列与 `cublasCrot` 一一对应（handle 及参数顺序），无需额外映射说明（任务书 §3.5.2）。声明加入公共头文件 `include/cann_ops_blas.h`，可与其它产品线共用；不定义 950PR 私有接口。

`blas/rot/arch22/` 的 `csrot` 与 `blas/rot/arch35/` 的 `srot` 均不改动，按 SOC 分派，不影响既有产品线。
