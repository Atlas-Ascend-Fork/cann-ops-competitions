# aclblasChpr2（A2/A3）算子设计文档

> 范围说明：本 PR 仅提交设计文档，不含算子实现与测试代码。设计依据任务方《aclblasChpr2_AtlasA2_A3 任务书》与 `ops-blas @ 252d42d6` 源码；实现代码按 ops-blas 仓流程单独提交，文档不声称精度/性能已实测达标。

# 需求背景（required）

## 需求来源

- 任务：11 月社区任务-aclblasChpr2 算子开发（A2/A3），含税奖金 8665.50 元，最晚验收 2026-12-07。
- 依据：任务书 §2.1~§2.5（功能、接口、参数、校验顺序）与 §3.2~§3.5（精度、性能、自验口径），配套 `test_cases/`（精度全量 1000 条、性能子集 200 条）。
- 目标仓：[cann/ops-blas](https://gitcode.com/cann/ops-blas)。实现代码入 `blas/spr2/arch22/`，测试代码入 `test/spr2/chpr2/arch22/`。
- 硬件：Atlas A2/A3（arch22，DAV_2201）；CANN 9.1.0；精度 golden 由 Netlib CBLAS 3.10.0 生成。

## 背景介绍

### 算子功能与工程现状

`aclblasChpr2` 对 Hermitian 压缩矩阵执行复数秩-2 更新：

```text
A := alpha * x * y^H + conj(alpha) * y * x^H + A
```

- `A` 为 n×n Hermitian 矩阵，按列主序逐列无间隙 packed 存储 `uplo` 指定三角，共 `n(n+1)/2` 个 `aclblasComplex`（实部/虚部各 FP32），原地更新，无 `lda`、无 `beta`。
- `x`、`y` 为逻辑长度 n 的复数向量，`incx/incy` 可为任意非零整数（含负步长，负步长按反向遍历语义）。
- 两项贡献互为共轭转置，求和后结果天然 Hermitian；对角元素虚部在计算后强制置 0。
- 对标接口为 cuBLAS `cublasChpr2`，数值语义参考 Netlib `chpr2.f`。

以 `ops-blas @ 252d42d6`（2026-11-25）为基线：

| 能力 | arch35（950PR/DT） | arch22（A2/A3） | 本任务 |
|---|---|---|---|
| 公共声明 | `include/cann_ops_blas.h` L515 已声明 | 共用同一头文件 | **无需修改** |
| 复数秩-2 实现 | `blas/spr2/arch35/chpr2_host.cpp`、`chpr2_kernel.cpp`、`chpr2_kernel.h`、`chpr2_tiling_data.h` | 无 | 新增 arch22 |
| 同族实数实现 | arch35 `sspr2` | `blas/spr2/arch22/sspr2_host.cpp`、`sspr2_kernel.cpp`、`sspr2_tiling_data.h` | 复用 slab 划分、分核、原地更新脚手架 |
| 复数 AIV 先例 | — | `blas/axpy/arch22/caxpy_*`（交错↔平面、跨步 Gather）、`blas/gbmv/arch22/cgbmv_*` | 复用复数打包与跨步访问 |
| 产品支持表 | 支持 | **不支持**（`blas/spr2/README.md`） | 本任务补齐并更新 README |
| 测试资产 | `test/spr2/chpr2/arch35/`：golden、param、CSV | 无 arch22 注册 | 新增 arch22 GTest 与注册 |

### 标杆核查

| 审核对象 | 结论 | 依据 |
|---|---|---|
| 原 TBE 算子源码 | 不适用 | handle 式 BLAS 接口，不涉及 ACLNN/TBE 迁移 |
| 算子信息库 JSON | 不适用 | 不注册 ACLNN 算子 |
| 可复用工程实现 | 适用 | arch35 `chpr2_*`（packed 下标定位、对角置 0）、arch22 `sspr2_*`（slab 流水、负步长、UB 预算）、arch22 `caxpy`（复数平面化与跨步 Gather） |

### 现状差距

1. arch35 `chpr2` 为 SIMT packed 下标并行：512 线程按 packed 元素划分区间、`FindColumn` 二分定位列、列标量缓存、零列跳过；**arch22 无 SIMT**，不能直接移植，需改写为 AIV slab 流水。
2. arch22 `sspr2` 是实数实现：`SSPR2_ARCH22_MAX_N=4096`、`SEG_TILE=2048`、按 4B 元素做 UB 预算；复数元素 8B，tile 与 UB 预算需按 2 倍元素宽度重算。
3. arch35 实现对 `n=0` 提前 quick return，且 `uplo` 非法返回 `INVALID_VALUE`；任务书 §2.5 要求 `n=0` 不豁免结构参数校验、非法 `uplo` 返回 `INVALID_ENUM`。arch22 新实现必须按任务书口径，并作为 arch35 后续对齐项。
4. arch22 `sspr2` 的校验顺序为 `uplo → handle → n ≤ 0 → inc → 指针 → alpha`，与本任务书要求相反，不能照搬。

# 需求分析（required）

## 需求描述

在 CANN 9.1.0、Atlas A2/A3 上实现与 cuBLAS/Netlib 语义一致的 `aclblasChpr2`：COMPLEX64、packed 原地更新、UPPER/LOWER、正负步长、`n=0` 与零 alpha 边界；精度满足 COMPLEX64（FP32 分量）混合容差，性能达到 GPU kernel 基线的 0.8 倍以上。

## 接口定义

接口声明已存在于 `include/cann_ops_blas.h`（L515），本任务不新增/不修改声明：

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

返回值：`aclblasStatus_t`；成功为 `ACLBLAS_STATUS_SUCCESS`，非法枚举 `INVALID_ENUM`，其余结构参数非法 `INVALID_VALUE`，空句柄 `HANDLE_IS_NULLPTR`。

## 需求拆解

1. **功能**：对每个存储元素 `(i,j)` 计算
   `AP(i,j) += alpha*x_i*conj(y_j) + conj(alpha)*y_i*conj(x_j)`；
   `i==j` 时只累加实部，输出虚部置 0。
2. **数据类型**：`alpha`、`x`、`y`、`AP` 均为 `aclblasComplex`（re/im 各 FP32）；不支持 half/bf16/双精度扩展。
3. **形状**：`n ≥ 0` 且 `n ≤ 4096`（沿用 arch22 `spr2` 族形状约束）；`AP` 元素数 `n(n+1)/2`；`x/y` 物理元素数 `1+(n-1)·|inc|`，且 `≤ 12288`（arch22 同族向量长度上限）；全程用 `uint64/int64` 计算，并校验字节数不超 `size_t`。
4. **边界**：`n=0` 合法 no-op 但不豁免结构参数校验；`alpha=(0,0)` quick return 且不改 `AP`；`alpha=0` 不掩盖空 `x/y/AP`（指针校验在前）。
5. **步长**：`incx/incy` 任意非零整数（`INT_MIN` 除外）；负步长从物理分配起点反向遍历。
6. **校验**：严格按任务书 §2.5 顺序；非法枚举返回 `INVALID_ENUM`，其余结构参数非法返回 `INVALID_VALUE`，空句柄返回 `HANDLE_IS_NULLPTR`。
7. **精度**：golden 由 `cblas_chpr2`（CblasColMajor）生成，实部/虚部分别比对；FLOAT32 分量取 `rtol=atol=2^-13`、`matched_ratio≥0.99`、`max_abs≤1e-2`。
8. **性能**：`msprof op` 采集 kernel `Task Duration`，warmup 5 次 + 连续 10 次取均值，NPU ≤ GPU `gpu_ms`×1.25（即 ≥0.8× GPU kernel 性能）。
9. **交付**：设计文档 PR、arch22 实现、1000 条精度用例与 200 条性能用例自测报告、README 支持矩阵更新。

# 详细设计（required）

## 算子分析

### 数学公式

```text
A := alpha * x * y^H + conj(alpha) * y * x^H + A
```

对 packed 元素 `(i,j)`（只处理存储三角）：

```text
term1 = alpha * x_i * conj(y_j)
term2 = conj(alpha) * y_i * conj(x_j)
A(i,j) = (A(i,j) + term1) + term2          (i != j)
A(i,i) = ((Re(A(i,i)) + Re(term1)) + Re(term2)) + 0i
```

展开为实部/虚部（`a=alpha`，`x=xr+i·xi`，`y=yr+i·yi`）：

```text
Re(term1) = ar*(xr*yr + xi*yi) - ai*(xi*yr - xr*yi)
Im(term1) = ar*(xi*yr - xr*yi) + ai*(xr*yr + xi*yi)
Re(term2) = ar*(yr*xr + yi*xi) + ai*(yi*xr - yr*xi)
Im(term2) = ar*(yi*xr - yr*xi) - ai*(yr*xr + yi*xi)
```

packed 下标（0 基，列主序）：

```text
UPPER (i <= j):  idx(i,j) = j*(j+1)/2 + i
LOWER (i >= j):  idx(i,j) = j*(2n-j-1)/2 + (i-j)
```

### 支持数据类型

| 参数 | 类型 | 位置 | 说明 |
|---|---|---|---|
| `handle` | 句柄 | Host | 携带 stream |
| `uplo`、`n`、`incx`、`incy` | 标量 | Host | `inc != 0` 且 `!= INT_MIN` |
| `alpha` | COMPLEX64 | Host/Device | Host 直接读取；Device 经一次 D2H + stream 同步取值 |
| `x`、`y` | COMPLEX64 | Device（只读） | 交错 `{real, imag}`，8B/元素 |
| `AP` | COMPLEX64 | Device（读写） | packed，长度 `n(n+1)/2` |

### 支持形状

- `n ∈ [0, 4096]`；`AP` 元素数 `n(n+1)/2 ≤ 8,390,656`（n=4096 时约 64 MiB）。
- `1+(n-1)·|incx| ≤ 12288`、`1+(n-1)·|incy| ≤ 12288`；`|inc|` 用 `int64` 计算，避免 `INT_MIN` 取绝对值溢出（`INT_MIN` 在校验层直接拒绝）。
- 负步长物理起点 `(1-n)·inc`（`int64`），逻辑第 k 个元素地址为 `start + k·inc`。

## 算子实现

### 实现方案

#### 3.2.0 整体执行流程图

```mermaid
graph TD
    A[调用 aclblasChpr2] --> B[Host 校验<br/>handle→n<0→alpha 空→uplo→inc→n=0→x/y/AP 空]
    B --> C[形状/溢出守卫<br/>n<=4096, 1+(n-1)|inc|<=12288]
    C --> D{alpha == 0?}
    D -- 是 --> E[返回 SUCCESS<br/>AP 不变、不启动 Kernel]
    D -- 否 --> F[Tiling<br/>packed 区间分核 + 列 slab]
    F --> G[AIV Kernel 单次启动]
    G --> H[打包 x/y 到 UB<br/>连续 DataCopy 或跨步 Gather]
    H --> I[按列搬入 AP slab]
    I --> J[定列向量化复数乘加<br/>对角只算实部]
    J --> K[写回 AP<br/>对角虚部置 0]
    K --> L[异步返回，调用方同步 stream]
```

#### 3.2.1 Host 侧设计

**参数校验顺序（严格按任务书 §2.5）**：

| 顺序 | 检查 | 返回 |
|---|---|---|
| 1 | `handle == nullptr` | `HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `INVALID_VALUE` |
| 3 | `alpha == nullptr` | `INVALID_VALUE` |
| 4 | `uplo ∉ {UPPER, LOWER}` | `INVALID_ENUM` |
| 5 | `incx == 0 \|\| incx == INT_MIN` | `INVALID_VALUE` |
| 6 | `incy == 0 \|\| incy == INT_MIN` | `INVALID_VALUE` |
| 7 | `n == 0`（结构参数已校验） | `SUCCESS`，不引用 tensor、不启动 |
| 8 | `x/y/AP` 任一为空 | `INVALID_VALUE` |
| 9 | `n > 4096` 或向量长度越界或字节数溢出 `size_t` | `INVALID_VALUE` |
| 10 | `alpha == (0,0)`（标量值已解析） | `SUCCESS`，AP 不变、不启动 |
| 11 | 计算 Tiling、单次启动 | `SUCCESS`（异步） |

> 关键差异：`n=0` 只豁免 tensor 指针校验与 kernel 启动，**不豁免** handle/n/alpha/uplo/inc 校验。零 alpha 判断必须放在 `x/y/AP` 指针校验之后，避免“零 alpha + 空指针”被 quick return 掩盖。与 arch35 现有实现（n=0 提前返回、uplo 返回 `INVALID_VALUE`）不同，以任务书为准。

**标量位置处理**：

- 用 `CheckPtrLocation` 探测 `alpha` 位于 Host 还是 Device。
- Host：直接解引用；Device：`aclrtMemcpyAsync` D2H 2×4B + 同步 handle stream 后取值。
- 仅当进入第 10 步（n>0 且指针齐全）才解析标量值，`n=0` 与结构错误路径不触发同步。
- 位置探测失败返回探测错误码；不支持的混合语义不涉及（本接口只有 alpha 一个标量）。

**Tiling 字段**：

| 字段 | 含义 |
|---|---|
| `n`、`uplo` | 规模与存储三角 |
| `alphaReal/alphaImag` | 解析后的复数 alpha |
| `incx/incy`、`xStart/yStart` | 步长与负步长物理起点 |
| `packedLength` | `n(n+1)/2`（`uint64`） |
| `usedCoreNum` | 实际 AIV 核数 |
| `slabElems` | 单次 AP slab 元素数（复数口径） |
| `xPackMode/yPackMode` | 0=连续直搬，1=跨步 Gather，2=标量兜底 |
| `colBaseTab` | 可选：列基址增量表偏移（大 n 时减少重复乘法） |

**分核与负载均衡**：

- 工作量为 packed 元素；按 packed 区间 `[begin,end)` 连续划分给各核，天然按列长加权均衡。
- `blocks = min(GetAivCoreCount(), ceil(packedLength / MIN_ELEMS_PER_CORE))`；`MIN_ELEMS_PER_CORE` 取 4096，避免小 n 时核数过大导致固定开销占比高。
- 每核区间起点用列基址公式定位首列，区间内逐列推进；列内剩余元素不足一个 vector 时走尾块，不做越界写。
- **每个 packed 元素只归一个核、只写一次**，无原子加、无跨核写冲突，结果逐位确定。

**UB 预算（184 KiB 口径）**：

| 缓冲 | 规格 | 字节 |
|---|---|---|
| `x` packed（实/虚平面） | n≤4096，2×4096×4B | 32 KiB |
| `y` packed（实/虚平面） | n≤4096，2×4096×4B | 32 KiB |
| AP slab × 2（双缓冲） | 2048 复数 × 8B × 2 | 32 KiB |
| term/临时（4 个乘积平面） | 2048 × 4B × 6 | 48 KiB |
| 对齐/尾块余量 | — | ≤ 8 KiB |
| 合计 | — | ≤ 152 KiB < 184 KiB |

小 n 时按实际长度申请；n 较大时优先保证双缓冲，`slabElems` 取 `min(2048, n)` 并在列内分块。

#### 3.2.2 Kernel 侧设计

采用 AIV dataflow，单次 launch：

1. **向量打包（CopyIn）**：
   - `inc==1`：`DataCopy` 连续搬入；`inc==-1` 按反向起点搬入后逻辑反转。
   - `|inc|>1`：按“稠密载入跨度 + `Gather`”打包到 UB，仅保留逻辑 n 个元素；跨步过大导致跨度超 UB 时退化为标量 `SetValue` 打包（`packMode=2`）。
   - 打包同时完成交错→平面化（`xr/xi`、`yr/yi` 四个平面），后续计算只访问平面。
2. **列 slab 搬入**：按列基址公式计算首地址，UPPER 列 j 长度 `j+1`，LOWER 列 j 长度 `n-j`；`DataCopyPad` 搬入 AP slab，双缓冲预取下一 slab。
3. **定列向量化计算**：固定列 j 时 `x_j`、`y_j` 为标量，列内元素 i 构成向量：
   ```text
   AP_col[i] += alpha * x[i] * conj(y_j) + conj(alpha) * y[i] * conj(x_j)
   ```
   先算 `t1 = alpha*conj(y_j)`、`t2 = conj(alpha)*conj(x_j)` 两个复数标量，再对向量做两次复数乘加，减少重复标量运算。
4. **对角处理**：命中对角元素时只取 `Re(AP)` 与两个 term 的实部，写回 `Im=0`；对角输入虚部不参与运算。
5. **写回**：`DataCopyPad` 按列实际元素数写回，`slab` 边界与列尾均不越界；UPPER/LOWER 只写存储三角。
6. **同步**：`TQue` 管 MTE2→V→MTE3，slab 复用前 `PipeBarrier`；Kernel 退出前保留全屏障（`sspr2` 实测去掉尾部屏障会偶发挂死）。

**复数乘法的舍入口径**：

- 4 个实数乘积分别舍入后再加减，不使用会把整体收缩为 FMA 的写法；
- 元素级顺序固定为 `(AP + term1) + term2`，与 arch35 `StoreActiveElement` 一致；
- 每个元素单核独占，归约顺序不受调度影响，同输入结果逐位可复现。

#### 3.2.3 算子泛化设计

- **n=0**：Host 短路，不启动 kernel。
- **n=1**：仅处理 `AP[0]` 对角，实部累加、虚部置 0。
- **正负步长**：逻辑 k → 物理 `start + k·inc`；负步长 `start=(1-n)·inc`。
- **极端规模**：n=4096 时 packed 约 64 MiB，按列 slab 流式处理，不整块进 UB。
- **零 alpha**：不读 `x/y/AP`，不启动 kernel。
- **特殊值**：Inf/NaN 按 IEEE 传播；对角虚部输入被忽略。
- **重叠**：`AP` 与 `x/y` 不得重叠，由调用方保证，接口不做 Device 扫描。

#### 3.2.4 复数与 Hermitian 专项

- **共轭方向**：`x*y^H` 对 `y` 取共轭，`conj(alpha)*y*x^H` 对 `x` 取共轭；平面化后只需对虚部平面取负，避免整体重排。
- **对角置 0 契约**：非 quick return 路径一律写 `Im(AP(i,i))=0`，即使该列 `x/y` 全零；golden 由 `cblas_chpr2` 生成，天然为 0，比对时实部/虚部分别判。
- **原地安全**：每个 packed 元素由所属核先读后写一次，不存在读改写竞争。
- **与 arch35 差异**：arch35 按 packed 元素并行、每线程独立定位列；arch22 按列 slab 向量化，两者数值顺序一致（元素级顺序相同），仅调度不同。

### 数据检测

- 空 handle、负 n、空 alpha、非法 uplo、零/`INT_MIN` 步长、空 x/y/AP：按上表返回。
- `n > 4096`、`1+(n-1)|inc| > 12288`：`INVALID_VALUE`（arch22 族形状约束）。
- `packedLength·8 > SIZE_MAX`：`INVALID_VALUE`。
- 零 alpha 在 `x/y/AP` 指针校验之后判断，不掩盖空指针错误。
- `n=0` 叠加非法 uplo/inc/alpha 空：返回对应结构错误，不因 `n=0` 豁免。

### 支持硬件

| 芯片版本 | 勾选 |
|---|---|
| Atlas A2 训练/推理系列 | √ |
| Atlas A3 训练/推理系列 | √ |
| Ascend 950PR/DT（arch35 既有） | 不改动 |

### 算子约束限制

- 不支持广播；`AP` 原地覆写，`AP` 与 `x/y` 不得重叠。
- `alpha` 支持 Host/Device 指针；Device 标量会引入一次 stream 同步（n>0 时）。
- 仅支持列主序 packed 布局，不支持任意非连续 Tensor。
- 对角元素输出虚部固定为 0（与 Netlib 保留旧虚部不同，按 ops-blas/任务书契约）。
- arch22 形状上限沿用 `spr2` 族：`n ≤ 4096`、向量物理长度 `≤ 12288`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | COMPLEX64 实部/虚部分别按 FP32：`rtol=atol=2^-13`，`matched_ratio≥0.99`，`max_abs≤1e-2`；golden 为 `cblas_chpr2` | 任务书 §3.2 |
| 性能 | `msprof op` kernel `Task Duration`，warmup 5 + 10 次均值 ≤ GPU `gpu_ms`×1.25 | 任务书 §3.3 |
| 交叉验证 | `verify_performance.py` 的 `TC_PF` 事件计时仅作趋势，不作正式判据 | 任务书 §3.3 |

## 测试用例规划

复用任务书 `test_cases/` 与 arch35 `chpr2_test.csv`，arch22 新增 GTest 注册：

| 类别 | 条数 | n 范围 | 覆盖点 |
|---|---|---|---|
| `basic` | 4 | 4–8 | UPPER/LOWER 冒烟 |
| `ex` | 800 | 1–2048 | 常规形状、两三角 |
| `pf` | 200 | 1–4096 | 性能基线 |
| `sq` | 46 | 1–2048 | 方阵边界 |
| `inc` | 72 | 16–64 | ±1/±2/±3 与大步长 |
| `cv` | 16 | 10–400 | 共轭方向正确性 |
| `fill` | 18 | 32 | 特殊填充 |
| `alpha` | 16 | 32 | (0,0)/(1,0)/(0,1)/一般复数 |
| `n0` | 8 | 0 | 空矩阵 + n=0 结构校验 |
| `null` | 4 | 16 | 空指针（含零 alpha + 空指针） |
| `neg` | 2 | -1 | 负 n |
| `alpha0` | 5 | 32 | 零 alpha（AP 不变） |
| 其他边界 | 9 | — | 非法 uplo、零步长、大 n、对角虚部非零 |

预期：1186 条 SUCCESS、14 条 `INVALID_VALUE`；性能子集覆盖 UPPER/LOWER、正负步长与小/中/大三档规模。

## 兼容性分析

- 新增 arch22 实现，不改公共头与 arch35 行为；arch35 的 `n=0`/`uplo` 差异作为后续独立修复项，不在本 PR 混入。
- 更新 `blas/spr2/README.md` 的 A2/A3 支持矩阵与 `test/spr2/chpr2/CMakeLists.txt` 的 arch22 目标。
- 保持 `include/cann_ops_blas.h` L515 声明不变，不新增私有 API。

## 风险与降级预案

| 风险 | 影响 | 预案 |
|---|---|---|
| arch35 与任务书校验顺序不一致 | 评审争议/测试期望不一致 | 以任务书为准实现，文档显式列出差异；补充“n=0 + 非法结构参数”用例 |
| 对角虚部口径（清 0 vs 保留） | 与 golden 不一致 | 按任务书清 0；golden 用 cblas 对照，单独断言输出虚部为 0 |
| 复数 8B 使 slab 变小 | 大 n 带宽利用率下降 | `slabElems=2048` 起步、双缓冲；按列长自适应，长列优先 |
| 跨步 Gather 成本高 | 小 n 性能抖动 | 小 n 标量打包；中等 n 用 Gather；性能不足时按列批量打包 x/y |
| 大 n 单核区间跨列多 | 列基址重复计算 | 预生成列基址表或增量递推，避免每元素一次乘法 |
| 尾块非对齐写 | 越界/踩踏 | 一律 `DataCopyPad` + 实际元素数；尾块不做向量化越界读 |

## 附录：修订记录

| 日期 | 版本 | 说明 |
|---|---|---|
| 2026-11-26 | v1.0 | 依据任务书与 `ops-blas @ 252d42d6` 完成设计；明确 arch22 列 slab 向量化与任务书校验顺序 |
