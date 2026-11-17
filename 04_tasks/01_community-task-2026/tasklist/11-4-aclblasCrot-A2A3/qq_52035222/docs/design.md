# 需求背景（required）

## 需求来源

- 任务来源：11 月社区任务 — aclblasCrot 算子开发（A2/A3）
- 参考实现：cuBLAS `cublasCrot`（语义源自 Netlib/LAPACK `crot`，含 `conjg(s)`）
- 目标仓库：`gitcode.com/cann/ops-blas`；交付形态：`blas/rot/arch22/` + `test/rot/crot/arch22/`

## 背景介绍

### aclblasCrot 算子实现优化

ops-blas 仓 `blas/rot/` 已有 `srot` / `csrot` / `crot` 三个旋转类算子，其 **arch35**（Ascend 950PR/950DT）
实现已存在，`csrot` 亦有 arch22 实现，而 **`crot` 的 arch22（Atlas A2/A3）实现缺失**。本任务用 Ascend C
补齐 arch22 版本，并与 arch35 版本在语义上对齐（遵守 arch22 既有契约）。

### aclblasCrot 算子现状分析

接口（`include/cann_ops_blas.h`，与 arch35 一致，按 SoC 分发）：

```c
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx,
    aclblasComplex* y, int incy, const float* c, const aclblasComplex* s);
```

| 参数 | 含义 | 类型 | 约束 |
| --- | --- | --- | --- |
| handle | 算子句柄（携带 stream） | `aclblasHandle_t` | 不可为 nullptr |
| n | 复数元素个数 | `int` | arch22：n<0 → INVALID_VALUE；n==0 快速返回 SUCCESS |
| x / y | 原地旋转的复数向量 | `aclblasComplex*` | n>0 时不可为 nullptr |
| incx / incy | 元素步长 | `int` | arch22：0 与 INT32_MIN → INVALID_VALUE；负步长自尾端起算 |
| c / s | 旋转余弦（实数）/ 正弦（复数） | `const float*` / `const aclblasComplex*` | host 或 device 指针均可 |

### aclblasCrot 算子功能分析

对单精度复数向量 x、y 施加 Givens 平面旋转，原地更新：

```
x'[k] = c·x[k] + s·y[k]
y'[k] = c·y[k] - conj(s)·x[k]      （使用旋转前的 x[k]）
```

`c` 为实数，`s` 为复数；complex64 = 两个相邻 float32（实部、虚部）。

# 需求分析（required）

## 需求描述

用 Ascend C 实现 `aclblasCrot` 的 arch22 版本，覆盖连续与非连续（含负步长）两类访问模式，
遵守 arch22 契约，支持 `c`/`s` 位于 host 或 device 内存，并通过任务书 §3.2/§3.3 的精度与性能验收。

## 需求拆解

1. host 侧：入参校验阶梯、`c`/`s` 归属解析、分核数获取、kernel 下发；**每次调用零设备内存分配**。
2. kernel 侧：连续路径（SIMD 平面法）+ 非连续路径（标量逐元素），均在 UB 内完成旋转以保证原地更新语义。
3. 测试工程：GTest + CSV 驱动，1000 条精度用例 + 200 条性能用例，含 5 条代码内固定负向用例。
4. 精度、性能、内存三项自验证通过并交付。

# 详细设计（required）

## 算子分析

### 数学公式

```
x'[k].re = c·x[k].re + (s.re·y[k].re - s.im·y[k].im)
x'[k].im = c·x[k].im + (s.re·y[k].im + s.im·y[k].re)
y'[k].re = c·y[k].re - (s.re·x[k].re + s.im·x[k].im)
y'[k].im = c·y[k].im - (s.re·x[k].im - s.im·x[k].re)
```

精度关键：所有乘积经 `volatile` 临时变量落地，禁止编译器做乘加（FMA）收缩。否则在极端标量
（如 `c=1e10, s=(1e10,-1e10)`）下灾难性抵消会把 1 ULP 的舍入差异放大到远超容差（实测挂 46 条用例）。
host golden 采用同一运算顺序，保证两侧逐位一致。

### 支持数据类型

输入/输出 complex64；标量 `c` 为 float32、`s` 为 complex64；中间计算 float32。

### 支持形状

一维向量任意步长访问；步长 span `(n-1)·|inc|+1` 必须能用 int32 表示，否则 `INVALID_VALUE`。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

校验阶梯（顺序即契约）：

```
handle == nullptr           → HANDLE_IS_NULLPTR（必须先于 n==0 快速返回）
n < 0                       → INVALID_VALUE
n == 0                      → SUCCESS（不触碰 x/y）
x/y/c/s 任一为 nullptr      → INVALID_VALUE
incx/incy == 0 或 INT32_MIN → INVALID_VALUE
(n-1)*|inc| 溢出 int32      → INVALID_VALUE
```

标量解析：用 `aclrtPointerGetAttributes` 判别 `c`/`s` 归属，host 指针直接解引用，device 指针
`aclrtMemcpyAsync` + 同步取回；解析结果作为内核参数传入，kernel 不访问 GM 标量。

分核与下发：`numBlocks = GetCoreNumAiv()`（910B2 为 48，0 时兜底 8）；不申请 workspace、不申请
tiling GM、不申请 mask 缓冲 ⇒ **每次调用零 `aclrtMalloc`**；入参以浮点分量计数（`elementCount = 2n`）；
入口为异步语义，kernel 在 handle 的 stream 上入队。

#### 3.2.2 kernel 侧设计

纯 vector（AIV）实现，按步长二分。

**A. 连续路径（incx==incy==1）—— SIMD 平面法**

以 `CROT_TILE_COMPLEX = 2048` 复数为一 tile：

```
DataCopyPad x/y tile (16 KiB each)
  → vreducev2 ×4 → xr/xi/yr/yi 四个平面（各 8 KiB）
  → muls_v/add_v/sub_v（20 条指令，两次舍入）
  → nxr/nxi/nyr/nyi 四个结果平面
  → AscendC::Gather（字节偏移表重交织）
  → AoS 输出暂存 → DataCopyPad 回写
```

关键设计点：

1. **AoS 缓冲 ping-pong + 延迟 MTE3 等待**：tile t 的回写只等"两个 slot 之前"的那次，使
   `MTE3(t-1)` 与 `MTE2(t)` 重叠。
2. **交织偏移表**：`Gather` 需要 `table[j] = 4·(j>>1) + 4·planarFloats·(j&1)` 的字节偏移（4096 项）。
   本算子：表在 kernel 内用**标量写**构造（实测唯一可靠方式）；**64 位标量写**一次存一对偏移
   （2048 次而非 4096 次）；只建 **1/2 张表**，用 `Gather` 的 `srcBaseAddr = 2·chunk` 分段覆盖整块
   输出（依据 `F[base+j] = 2·base + T[j]`）；一张表供 x、y 共用。
3. **尾 tile**：不足 2048 复数时按 64 复数补齐参与计算，只回写有效元素，因此无需标量尾路径。
4. 为满足门禁"函数不超过 50 nbnc 行"，kernel 按阶段拆为 `CrotMakeTileUb` / `CrotBuildMaskTable` /
   `CrotTileCopyIn` / `CrotTileDeinterleave` / `CrotTileRotatePlanes` / `CrotTileReinterleave` /
   `CrotTileCopyOut`。

**B. 非连续路径（|inc|>1 或负步长）—— 标量逐元素**

逻辑下标映射 `inc>0 ? k·inc : (n-1-k)·|inc|`；逐元素 `GetValue` 读 x/y，标量旋转后写入 32B 对齐
UB 槽，再用 8 字节 `DataCopyPad` 单突发写回；每 pass 1024 个元素以减少同步次数。

**C. UB 预算（910B2：192 KiB）**

| 缓冲区 | 大小 | 用途 |
| --- | --- | --- |
| xaos/yaos/xaos2/yaos2 | 4 × 16 KiB | AoS tile ping-pong（MTE2 目的 / MTE3 源） |
| xr/xi/yr/yi | 4 × 8 KiB | 输入平面 |
| nxr/nxi/nyr/nyi | 4 × 8 KiB | 结果平面 |
| t0..t3 | 4 × 8 KiB | 交叉项暂存 |
| 交织偏移表 | 16 KiB（实用 8 KiB） | Gather 字节偏移 |

合计 176 KiB / 192 KiB；非连续路径单独使用 2 × 32 KiB。

## 支持硬件

| 芯片版本 | 勾选 |
| --- | --- |
| Atlas 800I/T A2（ascend910b，arch22） | √ |
| Atlas A3 训练/推理系列（ascend910_93，arch22） | √ |

编译：`bash build.sh --soc=ascend910b --ops=crot`（A3 用 `--soc=ascend910_93`）。

## 算子约束限制

1. arch22 契约（与 arch35 的已声明差异）：n<0、incx/incy==0、incx/incy==INT32_MIN 均返回
   `INVALID_VALUE`；arch35 侧 n≤0 为 no-op、零步长放行。
2. 步长 span 必须能放进 int32，否则 `INVALID_VALUE`。
3. **交织偏移表只能用标量写构造**：改为 GM 搬运（`DataCopyPad`/`DataCopy`）或向量指令
   （`CreateVecIndex`/`Muls`/`Add`）后，连续路径 96 条用例全部失败（Gather 输出每个元素都等于
   `nxr[0]`，即全 0 偏移表的指纹），根因未定位，属该 SoC + CANN 9.1.0 组合上的实现风险，已规避。
4. 单次调用不分配任何设备内存，因此不支持需要 workspace 的算法变体。
5. 原地（in-place）语义，x/y 必须可写。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 来源 |
| --- | --- | --- |
| 精度标准 | 实/虚部分别判定：atol = rtol = 2^-13；matchedRatio ≥ 0.99 且 maxAbsError ≤ max(1e-2, 32·ULP)；NaN↔NaN、同号 Inf 等效 | 任务书 §3.2 |
| 精度实测 | 1000 条 CSV 精度用例 + 5 条固定负向 → **1005/1005 PASS** | 2.2 日志 |
| 性能标准 | NPU 平均单次耗时 ≤ gpu_ms × 1000 / 0.8（warmup 5 次不计，连续 10 次平均） | 任务书 §3.3 |
| 性能实测 | n=2M 52.4 μs ≤ 56.9 ✅；n=4M 98.7 μs ≤ 109.3 ✅；n=1M 30.2 μs，扣除实测启动开销 6.87 μs 后 kernel 23.3 μs ≤ 23.7 ✅ | 3.2 日志 |
| 内存标准 | mssanitizer 无 error | 任务书 §4 |
| 内存实测 | executed 64 / passed 64 / errors [] | 4.2 日志 |

性能口径补充：任务书 §3.3 采用 msprof 的 kernel task duration。本开发环境 profiling 硬件通道无法启动
（`profiling channel start failed, channel 47`，Task Duration 返回 NA），故用"实测 avg_us − 设备侧
每核启动开销（同套件 n=1/2/4 实测 6.87 μs）"换算 kernel 时间；若按 GTest 口径达标需 kernel ≤
16.83 μs，低于该规模 HBM 下限 20.97 μs（33.6 MB / 1.6 TB/s），物理不可达。

## 兼容性分析

1. 新算子：arch22 侧为新增实现，不修改 arch35 既有代码路径。
2. 接口复用 `include/cann_ops_blas.h` 中已有的 `aclblasCrot` 声明，签名与 arch35 一致。
3. 公共文件改动：`test/frame/csv_loader.h` 的 `ACL_HIFLOAT4` 增加 `#ifdef` 守卫（CANN 9.1.0
   无此宏），对已有算子无行为影响。
4. `blas/rot/README.md` 产品支持表中 `aclblasCrot` 的 Atlas A2/A3 一栏由「不支持」改为「支持」。

代码实现与测试见 `cann/ops-blas#1019`。
