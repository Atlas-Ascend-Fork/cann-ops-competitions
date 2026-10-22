# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务（CANN 训练营·南科大）：在 Ascend 950PR 上使用 Ascend C 编程语言实现
`aclblasCrot` 算子，补齐 ops-blas 仓 rot 算子族在 complex64 上的 Givens 平面旋转能力，
验收通过后合入 `cann/ops-blas` 开源仓 `blas/rot/arch35/`。

## 背景介绍

### 算子实现现状分析

ops-blas 仓 `blas/rot/` 已有两个算子：

| 算子 | 位置 | 数据类型 | 产品支持 |
| --- | --- | --- | --- |
| `aclblasSrot` | `blas/rot/arch35/` | FP32 实数 | Ascend 950PR / 950DT |
| `aclblasCsrot` | `blas/rot/arch22/` | FP32 complex | Atlas A2 / A3 |

`aclblasCrot` 在公共头 `include/cann_ops_blas.h` 中**尚无声明**，也没有任何实现。
复数旋转与 `aclblasCsrot` 的区别在于旋转正弦 `s` 为复数且 y 的更新含共轭：
`Csrot` 的 s 是实数、更新为 `y = c*y - s*x`；`Crot` 的 s 是复数、更新为 `y = c*y - conj(s)*x`。

参考 BLAS 库（cblas）**没有 crot 例程**，因此 golden 无法直接调用参考实现，
需要按 Netlib/LAPACK `crot.f` 语义自行实现（见 §可维可测分析）。

### 算子功能分析

对两个等长 complex64 向量施加 Givens 平面旋转，x、y 原地更新：

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - conj(s) * x[k]        （使用旋转前的 x[k]）
```

其中 `c` 为实数（旋转余弦），`s = sr + i·si` 为复数（旋转正弦）。

输入：x、y（各 n 个 complex64）、c（float）、s（complex64）
输出：x、y（原地更新）
支持数据类型：COMPLEX64（实部/虚部各 float32），c 为 FLOAT32

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Ascend 950PR（arch35）上实现 `aclblasCrot` 句柄式 BLAS 接口，
通过 handle 绑定 stream 直调 NPU kernel；接口声明新增至 `include/cann_ops_blas.h`，
实现代码位于 `blas/rot/arch35/`，测试工程位于 `test/rot/crot/arch35/`。

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx,
    aclblasComplex* y, int incy, const float* c, const aclblasComplex* s);
```

## 需求拆解

1. 功能：复数 Givens 旋转，y 使用旋转前的 x，且 s 取共轭。
2. 步长语义：incx/incy 可正、可负、可为零；负步长从向量尾端起算；零步长按参考实现串行复用首元素。
3. 边界语义：n ≤ 0 为合法 no-op（返回 SUCCESS，不修改 x/y）；n > 0 时 x/y/c/s 为空指针返回 INVALID_VALUE；
   handle 为空返回 HANDLE_IS_NULLPTR。
4. 标量位置：c/s 为标量指针，可为 Host 或 Device 内存，由算子运行时独立判定。
5. 精度：实部/虚部分别按 FLOAT32 标准判定，`atol = rtol = 2^-13`，`matched_ratio >= 0.99`。
6. 性能：三个标杆 case（n=1048576/2097152/4194304）kernel 平均耗时不超过标杆值。
7. 通用性：接口须可被其他产品线共用，禁止定义 950PR 私有平行接口。

# 详细设计（required）

## 算子分析

### 数学公式

i = 0…n-1（0-based，等价于任务书 1-based）：

```
k = (incx >= 0) ? i*incx : (n-1-i)*|incx|
j = (incy >= 0) ? i*incy : (n-1-i)*|incy|

x'[k] = c*x[k] + s*y[j]
y'[j] = c*y[j] - conj(s)*x[k]     （使用旋转前的 x[k]）
```

实数/虚部展开（全部 float 运算）：

```
s*Y       : re = sr*yr - si*yi ,  im = sr*yi + si*yr
conj(s)*X : re = sr*xr + si*xi ,  im = sr*xi - si*xr
x' re = c*xr + (sr*yr - si*yi)      x' im = c*xi + (sr*yi + si*yr)
y' re = c*yr - (sr*xr + si*xi)      y' im = c*yi - (sr*xi - si*xr)
```

### 支持数据类型

COMPLEX64（实部/虚部各 FLOAT32）；标量 c 为 FLOAT32，s 为 COMPLEX64。

### 支持形状

x/y 逻辑一维 [n]，物理长度 `1 + (n-1)*|inc|`；不做动态 shape，n 为运行时入参。

## 算子实现

### 实现方案

按步长是否连续分两条路径，由 host 侧 `tilingKey` 选择：

| tilingKey | 条件 | 实现 | 原因 |
| --- | --- | --- | --- |
| 0 | `incx==1 && incy==1` | SIMD（AIV membase，GM↔UB 搬运算 Vector） | 连续访存可打满带宽 |
| 1 | 其他（含负步长、零步长） | SIMT（grid-stride loop） | 跨步场景逐元素计算更简洁 |

#### 3.2.1 host 侧设计

**参数校验顺序**（与 netlib 及同族算子一致）：
`handle == nullptr` → `HANDLE_IS_NULLPTR`；`n <= 0` → `SUCCESS`（提前返回，不触碰 x/y）；
`n > 0` 时 x/y 或 c/s 为空 → `INVALID_VALUE`。

**标量位置判定**：对 c、s 分别调用 `aclrtPointerGetAttributes`，
`ACL_MEM_LOCATION_TYPE_DEVICE` 时把指针原样下传（kernel 从 GM 读标量），
否则在 host 侧解引用一次、把标量存入 TilingData 并下传空指针。两者相互独立（支持 host+device 混合）。

**分核策略**：连续路径按 AIV 核数均分，`perCoreN = n / numBlocks`，
余数 `remainder = n % numBlocks` 分给前若干个核（每个多 1 个元素）。
不做块对齐取整，避免 `n / numBlocks < 单块元素数` 时出现空核；GM 侧只要求 1 字节对齐，
因此非 32B 对齐的核起点与尾块都合法。

**数据分块与内存优化**：连续路径按 `CROT_TILE_COMPLEX = 4096` 个复数为一个 tile，
单核内循环处理。UB 需 14 个 tile 大小的 float buffer（x/y 的 AoS 输入 4 份、
拆解后的实/虚部 4 份、结果 4 份、交织暂存 1 份、标量/临时 1 份），
占用 `14 × 4096 × 4B = 224 KiB`，低于 UB 上限 248 KiB。
AoS 输入 buffer 在交织写回时复用，省下 2 份 buffer。

**溢出保护**：`(n-1)*|inc|` 用 int64 计算，超过 INT32_MAX 返回 `INVALID_VALUE`
（kernel 侧用 uint32 索引，避免回绕）。

**SIMT 路径线程数**：`nthreads = ceilAlign(ceilDiv(n, numBlocks), SIMT_MIN_THREAD_NUM)`，
上限 `SIMT_MAX_THREAD_NUM`；块数取 `ceilDiv(n, SIMT_MIN_THREAD_NUM)` 与 AIV 核数的较小值。

#### 3.2.2 kernel 侧设计

**连续路径（`crot_aiv_kernel`）**：Init / Process 两阶段，Process 内按 tile 循环：

1. `DataCopyPad` GM→UB，一次搬 2·count 个 float（AoS，实虚交错）。
2. `DeInterleave` 把 AoS 流拆成纯实部、纯虚部向量。
3. 20 条标量广播 Vector 指令完成四个分量的计算。**每个乘积单独 `Muls`、每个加减单独 `Add/Sub`，
   不做 FMA 收缩**，与 CPU golden 同序，因此结果逐位一致。
4. `Interleave` 交织回 AoS 并写回复用的输入 buffer。
5. `DataCopyPad` UB→GM 写回。

同步使用 `MTE2_V` / `V_MTE3` / `MTE3_MTE2` 三对手工 flag，最后一个 tile 的 flag 在 Process 末尾统一 drain。
所有 Vector/permute 指令的计数与偏移统一按 32B 取整（`align_up(elementCount, 8)`），
避免尾块（如 n=1 时只有 8B）在 UB 侧地址未对齐导致 vector core exception；多算出的尾部被丢弃。

**跨步路径（`crot_simt_kernel`）**：`asc_vf_call` + grid-stride 循环，每线程处理一个复数
（读 2 个 float、写 2 个 float）。

- 负步长：`jx = n-1-i`，`|inc|` 用 int64 计算避免 `INT32_MIN` 取负 UB。
- 零步长：netlib 中 IX 常数不前进，N 次迭代作用在同一地址、串行累加（第 i 次读到的是第 i-1 次写回的值）。
  因此当 `incx==0` 或 `incy==0` 时只保留 `block0/thread0` 执行 N 次循环，其余线程立即返回，
  与参考实现逐位一致；非零那一侧仍在循环内按 i 递增访问。
- 原地安全：先把 `xr/xi/yr/yi` 全部读入寄存器，四个结果算完再写回，保证 y' 使用旋转前的 x。

**标量来源**：`cIsDevice` / `sIsDevice` 由 host 传入；device 指针时从 GM 读取（s 读连续两个 float），
host 指针时使用 TilingData 中的标量。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

编译参数 `--npu-arch=dav-3510`（`Ascend950PR_9579`，`NpuArch=3510`）。
注意 `.ini` 中的 `CCEC_AIC_version=dav-c310-cube` 是 CCEC 核内部版本，**不是**合法的 `--npu-arch` 取值
（用它会在编译期报 `Unsupported NPU architecture or soc`）。

## 算子约束限制

- 仅支持 complex64；不涉及 broadcast、dynamic shape。
- incx/incy 零步长为合法输入，按参考实现串行语义执行（不拦截）。
- `(n-1)*|inc|` 超出 int32 表示范围时返回 `INVALID_VALUE`。
- 无 workspace 需求；不支持确定性计算要求（逐元素运算本身确定）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FP32 判定：`atol=rtol=2^-13`，`matched_ratio>=0.99`，`max_abs_error<=max(1e-2, 32*ULP)` | 生态算子开源精度标准（mixed_tolerance_standard.md） |
| 性能标准 | 三个标杆 case kernel 平均耗时不超过基准：21.96 / 43.70 / 119.52 us | 任务书 §3.3（H100 基线 / 0.4） |

**实测结果**（2026-10-21，Ascend 950PR）：

| case | n | 预算 | 实测 kernel 平均 | 余量 |
| --- | --- | --- | --- | --- |
| 1 | 1048576 | 21.965 us | 17.22 us | 21% |
| 2 | 2097152 | 43.705 us | 31.06 us | 29% |
| 3 | 4194304 | 119.517 us | 66.42 us | 44% |

精度：1000/1000 用例 PASS（实部、虚部分别满足阈值）。

**golden 说明**：参考 BLAS 无 crot 例程，`test/rot/crot/crot_golden.h` 按 Netlib/LAPACK `crot.f` 语义实现，
并在 `test/rot/crot/arch35/crot_test.cpp` 中与算子结果逐项比对；
另有一份独立的 Python 参考实现做交叉验证（含零步长串行累加），5 组手算场景与硬件输出逐位一致。

## 兼容性分析

新算子，`include/cann_ops_blas.h` 仅**新增**声明，不修改既有接口签名，不涉及兼容性分析。
同族既有接口保持原样：`aclblasSrot`（arch35，指针标量）、`aclblasCsrot`（arch22，`int64_t` + 按值标量）。
