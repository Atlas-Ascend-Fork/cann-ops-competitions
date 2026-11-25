> 依据官方任务：https://www.hiascend.com/activities/task-center/details/c596a05d90b9471683e41fcf4d026ae5
> 任务书：11月社区任务-aclblasChpmv算子开发（A2/A3）；讨论帖 ops-blas #86。
> 精度 ACC1000 / 性能 PF200 / 正式判据 msprof op ≥0.8×GPU kernel（任务书 §3.2/§3.3）。

> 依据官方任务：https://www.hiascend.com/activities/task-center/details/c596a05d90b9471683e41fcf4d026ae5
> 任务书：11月社区任务-aclblasChpmv算子开发（A2/A3）；讨论帖 https://gitcode.com/cann/ops-blas/discussions/86
> 精度 ACC1000 / 性能 PF200 / 正式判据 msprof op kernel ≥0.8×GPU（任务书 §3.2/§3.3）；性能测试手段沿用我方 msprof 2-pass 套件。

# 设计文档头信息

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 202611社区任务-aclblasChpmv算子开发（A2/A3） |
| 提交团队 | Light_Uu |
| 任务讨论 | [ops-blas discussions #86](https://gitcode.com/cann/ops-blas/discussions/86) |
| 适配硬件 | Atlas A2/A3，`arch22`；首要验证 `ascend910b3`，A3 用同架构 `ascend910_93*` |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas` |
| 代码目标目录 | `blas/spmv/arch22/`（新增 `chpmv_*`，与既有 `sspmv_*` 并存） |
| 测试目标目录 | `test/spmv/chpmv/arch22/` |
| 文档状态 | 设计文档提交 |

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务「aclblasChpmv算子开发（A2/A3）」。在 `ops-blas` 为 Atlas A2/A3（`arch22` / DAV_2201）补齐单精度复数 Hermitian **packed** 矩阵-向量乘 `aclblasChpmv`。

语义裁决顺序：

1. cuBLAS `cublasChpmv`；
2. Netlib / CBLAS `cblas_chpmv`；
3. 仓内既有 arch35 实现 `blas/spmv/arch35/chpmv_host.cpp` / `chpmv_kernel.cpp`（校验与 quick-return 口径）；
4. 仓内 arch22 实数同族 `blas/spmv/arch22/sspmv_*`（A2 可落地的 Ascend C 写法）。

## 背景介绍

### 算子语义

计算：

```text
y := alpha * A * x + beta * y
```

其中 `A` 为阶 `n` 的 Hermitian 矩阵，以 **packed** 列主序只存上/下三角（长度 `n(n+1)/2`）。对角元虚部按 Hermitian 约定视为 0。`alpha`/`beta` 为 Host 侧 `aclblasComplex` 标量指针；`AP`/`x`/`y` 为 Device 侧。

公共接口（已声明，本任务不改 ABI）：

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

### 现状分析

| 路径 | 状态 |
| --- | --- |
| 公共声明 | `include/cann_ops_blas.h` 已有 |
| arch35（950） | 已实现；主路径使用 **SIMT**（`simt_api/asc_simt.h`、`__simt_callee__`） |
| arch22（A2/A3） | **缺失**；同目录仅有实数 `sspmv`（纯 Ascend C `TPipe`/`DataCopy`/`Muls`/`Add`，无 SIMT） |

**关键约束（源码核实）**：arch35 `chpmv_kernel.cpp` 依赖 SIMT。仓内设计惯例写明「arch22 无 SIMT，不能复用」。因此 **禁止** 把 arch35 kernel 原样拷到 arch22；只能对齐 Host 口径与数学语义，用 arch22 Ascend C 重写。

### 参数与异常

| 参数 | 约束 |
| --- | --- |
| handle | 空 → `HANDLE_IS_NULLPTR` |
| n | `n < 0` → `INVALID_VALUE`；`n == 0` → SUCCESS（不碰其它指针） |
| uplo | 仅 `UPPER`/`LOWER`，否则 `INVALID_ENUM` |
| incx/incy | 非 0；建议同时拒绝 `INT_MIN`（与 Chpr2/Sspmv 族一致） |
| alpha/beta | `n>0` 时非空；**Host 内存**（arch35 用 `CheckPtrLocation` 拒 Device） |
| AP/x/y | `n>0` 时非空 Device |
| span | `(n-1)*|inc|` 用 int64 校验防溢出 |

## 口径裁决

1. `n == 0` 优先于一切指针校验（与 arch35 `ChpmvValidateInput` 一致）。
2. `alpha == (0,0) && beta == (1,0)` → no-op SUCCESS，不读 AP/x、不写 y（arch35 `ChpmvIsNoOp`）。
3. `alpha == (0,0) && beta != (1,0)` → 只做 `y := beta * y`，不读 AP/x。
4. 对角：累加时只用 `Re(AP[diag])`，虚部忽略。
5. 负步长：物理 span 固定，逻辑下标映射与 Chemv/Sspmv 一致。
6. 返回码优先级：`HANDLE_IS_NULLPTR` > `INVALID_ENUM`/`INVALID_VALUE` > `EXECUTION_FAILED`。

# 需求分析（required）

## 需求描述

在 `blas/spmv/arch22/` 用 Ascend C kernel 直调实现 `aclblasChpmv`，handle 绑定 stream 异步下发；不新增私有平行接口；不使用 arch35 SIMT/tensor_api。

## 需求拆解

1. Host：校验、quick-return、标量 Host 解引用、核数感知 tiling。
2. Kernel：packed 寻址 + Hermitian 共轭补全 + 正负步长 + complex64 乘加。
3. 连续步长走 SIMD 双缓冲；非连续用「物理 span 搬运 + UB 逻辑重排」（对齐 Saxpy/Sspmv arch22 做法）。
4. Golden：Netlib/CBLAS `chpmv`（或 `zhpmv` 提升再收窄），按任务书 mixed_tolerance。
5. 性能：按任务书 GPU 基线与采样口径（以正式任务书为准）。

# 详细设计（required）

## 算子分析

### 数学公式

对输出行 `r`：

```text
y[r] = alpha * Σ_j H(r,j) * x[j] + beta * y[r]
```

packed 下标（0 基）：

- UPPER 列 `j`：基址 `j(j+1)/2`，行 `i∈[0,j]` 元素在 `base+i`；
- LOWER 列 `j`：基址 `j(2n-j+1)/2`，行 `i∈[j,n)` 元素在 `base+(i-j)`。

Hermitian 补全：读到的三角外元素取共轭；对角虚部置 0。

### 支持数据类型

COMPLEX64（`aclblasComplex`，实虚交错 float）。

### 支持形状

Hermitian packed 向量长度 `n(n+1)/2`；向量 `x`/`y` 长度与 `incx`/`incy` 相关；覆盖 `n=0` 到大规模及正负步长。

## 算子实现

### 实现方案

#### 目录

```text
blas/spmv/arch22/
  chpmv_host.cpp
  chpmv_kernel.cpp / .h
  chpmv_tiling_data.h
  （保留既有 sspmv_* 不动）
test/spmv/chpmv/arch22/
  CSV + GTest + README
```

#### Host 侧设计

对齐 arch35 校验顺序，但 tiling 只服务 AIV SIMD 路径：

- `strategy`：连续单位步长 + 规模足够 → 行块 SIMD；否则 → 通用路径（span pack）。
- 不启动 SIMT VF；`threads` 字段若保留仅作兼容占位或删除。
- 分核：优先满核；不能均分时余块分配到前几个核。按 `n` 与 UB 预算动态调整使用核数。
- UB：complex 为 float 的 2 倍，tile 行数相对 sspmv 减半重算；按需 double buffer。

#### Kernel 侧设计

以 `SpmvAIV`（sspmv arch22）为骨架扩展到 complex64，流程为 Init → Process（CopyIn / Compute / CopyOut）：

1. `AP`/`x`/`y` 用 `GlobalTensor<float>` 按实虚交错（每复数 2 float）或 `aclblasComplex` 等价布局访问；
2. 行块归约：UB 内累加 real/imag，对角只取实部；
3. 非连续：先把 x/y 的物理窗口拷入 UB，再按逻辑下标 gather（禁止依赖 SIMT 逐线程 GM 寻址）；
4. `TPipe` + 双缓冲 `DataCopyPad` + `Mul`/`Add`/`Muls` 向量指令；禁止 `tensor_api`、`DeInterleave`（A5 API）。

### 与 arch35 的差异声明

| 项 | arch35 | 本设计 arch22 |
| --- | --- | --- |
| 编程模型 | SIMT + 可选 SIMD | 仅 Ascend C AIV |
| 头文件 | `simt_api/asc_simt.h` | `kernel_operator.h` only |
| 可复用 | Host 口径、数学 | Kernel 结构参考 sspmv arch22 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3（同 arch22） | √ |

## 算子约束限制

1. 禁止 arch35 SIMT / Te / `tensor_api` 直接移植到 arch22。
2. `alpha`/`beta` 为 Host 侧指针；`AP`/`x`/`y` 为 Device 侧。
3. 对角虚部忽略；仅使用对角实部参与计算。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | ACC1000；golden=`cblas_chpmv`；按任务书 mixed_tolerance | 任务书 §3.2 / discussions #86 |
| 性能标准 | PF200；NPU Avg ≤ 1.25× A100（≥0.8×）；msprof op Task Duration | 任务书 §3.3 |

## 兼容性分析

公共 API 已声明，本任务不改 ABI；新增 arch22 实现路径，不影响既有 arch35 / sspmv。

## 测试与验收

1. 精度：任务 CSV + 边界（n=0/1、正负步长、alpha/beta 特例、上下三角）。
2. 性能：任务书 TC_PF；msprof op warm-up=5 + 有效采样 ≥10。
3. 回归：不得破坏既有 `sspmv` arch22 用例。

## 风险

1. packed + Hermitian 共轭在 SIMD 路径的索引易错 → 用小 n 穷举对照 golden。
2. 非连续路径无 SIMT，大 `|inc|` 时 MTE 占比高 → 性能靠连续主路径达标。
3. UB 预算：complex 是 float 的 2 倍，tile 行数需按 sspmv 减半重算。
