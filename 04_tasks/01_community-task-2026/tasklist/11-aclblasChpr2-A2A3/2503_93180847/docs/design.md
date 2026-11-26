# aclblasChpr2 算子设计文档（Atlas A2/A3）

> 适配硬件：Atlas A2 训练/推理系列（910B3/B4）、Atlas A3 训练/推理系列（910_93），均走 arch22
> CANN：9.1.0　｜　dtype：COMPLEX64（实部/虚部各 float32）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务 — aclblasChpr2 算子开发（A2/A3）。
- 合入目标仓：<https://gitcode.com/cann/ops-blas>，算子代码 `blas/spr2/arch22/`，测试代码 `test/spr2/chpr2/arch22/`。
- 交付分支（私仓）：`https://gitcode.com/2503_93180847/ops-blas`（分支 `feat/chpr2-arch22-spec-align`）。
- 代码 PR：向 `cann/ops-blas` master 提交实现与测试 PR。
- 设计文档提交：`cann-ops-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/{任务编号}-aclblasChpr2-A2A3/2503_93180847/docs/design.md`。
- 验收流程：社区任务流程（任务书 §4 交付件 / §5 PR 申请合入）。

## 背景介绍

### 算子功能定位

aclblasChpr2 完成单精度复数 Hermitian 矩阵的 packed 秩-2 更新：

```
A ← A + α·x·yᴴ + conj(α)·y·xᴴ
```

其中 A 为 n×n Hermitian 矩阵，以 packed 列主序格式仅保存 `uplo` 指定三角的 n(n+1)/2 个复数元素，无前导维度参数；x、y 为 n 元素复数向量。两项贡献互为共轭转置，结果天然 Hermitian，对角元素虚部在计算后强制置零。A 原地覆写。

该算子补齐 ops-blas 在 Atlas A2/A3（arch22）上的复数 Hermitian rank-2 更新能力；同一接口在 Ascend 950PR/950DT（arch35）已有实现。

### 现状分析

- 接口声明已存在于 `include/cann_ops_blas.h`，与 arch35 共用，**不另立私有平行接口**（任务书 §2.3）。
- 仓内 `blas/spr2/arch35/` 有 arch35 实现（SIMT 路径），可作为语义基线；其 `n=0` 与非法 `uplo` 的返回码契约与 arch22 不同（见下述"与仓内同族算子的关系"）。
- 仓内 `blas/spr2/arch22/` 已存在一份 aclblasChpr2 实现（commit `87a95e9`，MR !1038，2026-11-23 合入；`55ef791` 于 11-25 修复其发射状态未检查问题）。**本任务在该实现基础上做规格对齐与性能验收补齐，不重写计算路径**：
  - 该实现的数学核心（packed 双布局、负步长、对角置零、复数乘展开）经逐行核对正确，予以保留；
  - 其 Host 校验契约与任务书 §2.4/§2.5 不符（`uplo` 返回 `INVALID_VALUE`、`n=0` 在结构校验之前 quick return），本次修正；
  - 其测试用例按实现行为反写（`TC_ED_175` 期望 SUCCESS 等），无法发现上述偏差，本次换为任务书规格用例；
  - 其用例最大 n=2048 且无性能用例、无 `[PERF]` 打点，本次补齐。
- 同族参考：`blas/spr2/arch22/sspr2_host.cpp`（对称实数版，同为 packed 秩-2 更新）提供 arch22 工程骨架与分核范式；`test/her2k/cher2k/arch22/` 提供 arch22 复数 Hermitian 算子的测试与性能打点范式。

### 业界对标

- 对标接口 cuBLAS `cublasChpr2`：参数序列、返回码语义、quick return 口径以该文档为唯一依据。
- 语义参考 Netlib BLAS `chpr2.f`：`IF (N.EQ.0) RETURN`、`IF (INCX.EQ.0 .OR. INCY.EQ.0) INFO=...`、负步长起点 `IX=(-N+1)*INCX+1`。
- 行为对齐任务书 §2.5：结构参数（uplo/incx/incy/alpha）在校验层级上先于 `n=0` quick return，与该文档"n=0 不豁免结构参数校验"的表述一致。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 `blas/spr2/arch22/` 实现 `aclblasChpr2`，支持：

1. COMPLEX64 单一 dtype 面（alpha 为复数标量，x/y/AP 为 complex64）；
2. `uplo ∈ {ACLBLAS_UPPER, ACLBLAS_LOWER}`，双 packed 布局；
3. 任意合法非零 `incx`/`incy`（含负步长），排除 `INT_MIN`；
4. `n = 0` 与 `alpha = (0,0)` 为合法 quick return，不修改 AP；
5. AP 原地覆写，x/y 只读，不产生中间张量、无额外 workspace；
6. 异步执行：经 handle 绑定的 stream 下发，Host 侧不插入同步；
7. 精度、性能达到任务书 §3.2/§3.3 口径。

## 需求拆解

1. Host 侧按任务书 §2.5 的固定顺序完成两段式校验，返回码对齐 §2.3/§2.4；
2. 按列分核 + 行分块（tile）的双层划分，保证每元素被且仅被处理一次；
3. packed 索引在 64 位域计算，覆盖 n=4096 的 8390655 最大偏移；
4. 正负步长统一为"逻辑下标 → 物理偏移"映射，负步长按 Netlib 语义反向遍历；
5. 对角元素单独路径：仅累加增量实部，虚部强制 0；
6. 测试工程：CSV 驱动 GTest，1200 条（1000 精度 + 200 性能），含 `[PERF]` 打点供交叉验证。

## 接口与约束

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

| 参数 | 类型 | 内存位置 | 描述 | 约束与异常行为 |
|---|---|---|---|---|
| handle | `aclblasHandle_t` | Host | 库上下文，携带 stream | `nullptr` → `HANDLE_IS_NULLPTR` |
| uplo | `aclblasFillMode_t` | Host | 存储三角：UPPER / LOWER | 非法枚举 → **`INVALID_ENUM`** |
| n | `int` | Host | 矩阵阶数 | `n < 0` → `INVALID_VALUE`；`n = 0` 合法 no-op（结构参数仍校验） |
| alpha | `const aclblasComplex*` | Host | 复数标量 | `nullptr` → `INVALID_VALUE`；数值不校验（含 0/Inf/NaN） |
| x | `const aclblasComplex*` | Device | 输入向量，逻辑长度 n | `n > 0` 且 `nullptr` → `INVALID_VALUE`；物理长度 `1+(n-1)*|incx|` |
| incx | `int` | Host | x 步长 | `0` / `INT_MIN` → `INVALID_VALUE` |
| y | `const aclblasComplex*` | Device | 输入向量，逻辑长度 n | 同 x |
| incy | `int` | Host | y 步长 | 同 incx |
| AP | `aclblasComplex*` | Device | packed Hermitian 矩阵（原地） | `n > 0` 且 `nullptr` → `INVALID_VALUE`；长度 `n(n+1)/2` |

**Host 校验顺序（严格对齐任务书 §2.5）**

```
1. handle == nullptr                          → HANDLE_IS_NULLPTR
2. n < 0                                      → INVALID_VALUE
3. alpha == nullptr                           → INVALID_VALUE
4. uplo ∉ {UPPER, LOWER}                      → INVALID_ENUM
5. incx == 0 || incx == INT_MIN
   || incy == 0 || incy == INT_MIN            → INVALID_VALUE
6. n == 0                                     → SUCCESS（quick return，仅豁免下述两项）
7. x == nullptr || y == nullptr || AP == nullptr → INVALID_VALUE
8. 缓冲区长度溢出（1+(n-1)*|inc| 与 n(n+1)/2 超出 size_t 域） → INVALID_VALUE
9. alpha == (0,0)                             → SUCCESS（quick return，不读取 x/y/AP 数值）
```

关键点：第 6 步只豁免 tensor 指针检查与 kernel 启动，**不豁免第 3~5 步**，故 `n=0` 叠加非法 uplo、零步长、`INT_MIN`、空 alpha 时仍返回对应错误。

# 详细设计（required）

## 算子分析

### 数学公式

```
AP = alpha · x · yᴴ + conj(alpha) · y · xᴴ + AP
```

按列 j 展开（i 为行下标，仅遍历 `uplo` 指定三角）：

```
AP[i,j] += alpha·x[i]·conj(y[j]) + conj(alpha)·y[i]·conj(x[j])
AP[j,j]  : 仅取上式实部，虚部置 0
```

对固定列 j 预先算出两个复数乘子，使向量部分退化为"复数向量 × 复标量"：

```
t1 = alpha · conj(y[j])
t2 = conj(alpha · x[j]) = conj(alpha) · conj(x[j])
AP[:,j] += x[:] · t1 + y[:] · t2
```

复数乘展开为实部/虚部两步（实现内的交错布局）：

```
t1.re = aR·yjR + aI·yjI      t1.im = aI·yjR − aR·yjI
ax.re = aR·xjR − aI·xjI      ax.im = aR·xjI + aI·xjR
t2 = conj(ax) → t2.re = ax.re, t2.im = −ax.im
```

### 支持数据类型

| 参数 | 类型 | 说明 |
|---|---|---|
| alpha | COMPLEX64 | 实部/虚部各 float32，Host 指针 |
| x / y | COMPLEX64 | 逻辑长度 n，物理元素数 ≥ 1+(n−1)·abs(inc) |
| AP | COMPLEX64 | n(n+1)/2 个元素，原地覆写 |

### 支持形状

不涉及高维 shape。向量由 `n` 与 `incx/incy` 描述，压缩矩阵由 `n` 描述；不支持 broadcast、dynamic shape 专项适配、0 维 Tensor 入参。

### 负步长语义

入参指针指向**逻辑首元素**，负步长时向低地址遍历：

```
phys(k) = k · inc            (inc > 0)
phys(k) = (n−1−k) · |inc|    (inc < 0)
```

等价 Netlib `chpr2.f` 的起点 `IX = (−N+1)·INCX + 1`。该映射由 `Chpr2VecPhysIdx` 统一实现，正负步长共用同一段计算代码。

## 算子实现

### 实现方案

#### Host 侧设计

**（1）两段式校验**

`ValidateChpr2Structure(uplo, alpha, incx, incy)` 与 `ValidateChpr2Tensors(n, x, incx, y, incy, AP)` 分离，主流程按上节 9 步顺序调用。结构参数校验位于 `n == 0` 之前，tensor 指针与缓冲区溢出检查位于其后。

**（2）tiling 与分核**

```
numBlocks = min(n, GetAivCoreCount())
totalOrds = ceil(n / numBlocks)
colOf(blockIdx, ord) = ord·numBlocks + (ord 为奇数 ? numBlocks−1−blockIdx : blockIdx)
```

按列分配并对奇数轮做反序（蛇形），使列跨度不均（UPPER 的后段列更长、LOWER 的前段列更长）时各核负载大致均衡。tiling 结构体传递 `n / uplo / numBlocks / alphaReal / alphaImag / incx / incy`，无动态 shape。

**（3）launch 与错误传播**

`chpr2_kernel_do` 返回 `aclError`：发射前清除线程级残留错误，发射后检查并返回；Host 侧失败时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，不静默假成功。

#### Kernel 侧设计

**（1）索引表预构建**

启动时用向量指令生成三张表，避免列循环内重复计算：

- `splitIdx`：实部/虚部拆分步长（`2·k`），供 `Gather` 取实部；
- `packIdx`：共轭重排偏移（实部正序、虚部反序交错），供一次 `Gather` 得到 `x·t1` 所需的交错布局；
- `shiftIdx`：非 4 对齐行起点时的重排偏移。

**（2）分块加载**

按 `CHPR2_ARCH22_TILE_ROWS = 1024` 行切 tile，按步长分四种装载策略（均为实测后选定）：

| 条件 | 搬运方式 | 说明 |
|---|---|---|
| `inc = 1` | 一次连续 `DataCopyPad` | 直接落交织布局 |
| `2 ≤ inc ≤ CHPR2_ARCH22_STRIDED_CHUNK_MAX(31)` | 分段窗口：`chunk = (4096 / inc)` 向下取整到 8 的倍数，每段一次连续 `DataCopyPad` | 段数 1~8；chunk 取 8 的倍数保证目标地址 32B 对齐 |
| `inc > 31` | 一次多维 `DataCopyPad`（`blockCount = curLen`、`blockLen = 8B`、`srcStride = (inc−1)·8B`） | 读放大 1 倍；UB 侧每块占 32B 对齐槽位 |
| `inc < 0` | 跨度 ≤ 4096 时整窗连续拷贝；否则逐元素落到 32B 槽位（全部下发后只等一次 MTE2） | 负步长物理下标随逻辑下标递减 |

搬运后的「按步长抽取 + 实虚部交织」用**一次向量 `Gather`** 完成：构造交织索引 `idx[2i] = base + stepHalf·2i`、`idx[2i+1] = base + stepHalf·2i + 4`（单位字节），直接压紧进 `ubRaw`。索引由 `CreateVecIndex/ShiftRight/Muls/Sub/Add/Duplicate` 构造，临时张量复用 `bufWork_`。实部/虚部拆分与共轭重排仍走第（1）步的 `splitIdx/packIdx`，列内更新路径不变。

启动 block 数与逻辑分核数解耦：`tiling.numBlocks = min(n, aivCoreNum)` 不变，实际启动数 `launchBlocks = max(numBlocks, min(aivCoreNum, 8))`，多余的 block 在 `Init` 中按 `blockIdx >= numBlocks` 直接返回。

**（3）列内更新**

对 `[rowLo, rowHi)` 段：搬入 AP 旧值 → 计算 `x·t1` 与 `y·t2`（复数向量 × 复标量）→ 两次 `Add` 累加 → 回写。`x[j]`、`y[j]` 若落在当前 tile 内直接取 UB，否则单独搬 8 字节。

**（4）对角元素**

对角位置 `A(j,j)` 单独处理：读取旧实部，加上增量实部（`Re(x_j·t1) + Re(y_j·t2)`），虚部写 0。输入的 `A(j,j)` 虚部不参与任何计算。

**（5）整列零跳过**

若 `x[j]` 与 `y[j]` 同时为零，该列增量为零，仅需把对角虚部置 0，跳过整段搬入搬出。

### 性能优化方案

| 手段 | 目的 |
|---|---|
| 按列分核 + 蛇形映射 | 均衡 UPPER/LOWER 的三角列长度差异 |
| 1024 行 tile | 限制单次 UB 占用，适配 n 到 4096 的 4 轮 tile |
| 步长分档装载 | 小步长用连续窗口（1 次 MTE2），中等步长用分段窗口（2~8 次），大步长用多维拷贝（读放大 1 倍）；兼顾 MTE2 次数与读放大 |
| 向量 `Gather` 解交织 | 取代逐元素 `GetValue/SetValue`（每 tile `curLen×2` 次标量操作）；`TC_PF_1005` 由 28.7us 降到 14.5us |
| 极小 n 补足启动 block 数 | `blockDim = 1/2` 的 kernel 比 `blockDim ≥ 4` 多约 3us；补足到 8 后 n=1/2 由 6.6us 降到 3.5us |
| 索引表预构建 | 把每列的偏移计算从 O(n) 降为一次 `Gather` |
| 整列零跳过 | `alpha=0` 之外的零向量场景直接省去整列搬运 |
| 尾块精确访问 | 只读写实际元素，不触碰分配范围外内存 |

### 约束核对（任务书 §2.5）

| 约束项 | 落实情况 |
|---|---|
| 参数校验层级 | Host 入口按 §2.4 逐参数校验，不对 Device 张量做元素级扫描 |
| 非连续 Tensor | 不支持；跨度由 incx/incy 描述，AP 按列主序无间隙访问；不使用 AutoContiguous/ViewCopy |
| broadcast | 不支持 |
| dynamic shape | 不要求；阶数、步长均为运行时入参 |
| 原地与视图 | AP 原地覆写，不返回视图 |
| 确定性计算 | 不要求；按 §3.2 阈值判定 |
| 空 Tensor | `n=0` 合法 no-op，结构参数仍校验 |
| 0 维 Tensor | 不涉及 |
| 异步执行 | kernel 在 handle 绑定的 stream 上异步执行；Host 读回前须同步 |

## 测试与验收

### 测试工程

| 项 | 设计 |
|---|---|
| 用例来源 | 任务书规格 1200 条（1000 精度 + 200 性能），CSV 驱动 |
| CSV 列 | `case_name,description,uplo,n,alpha_real,alpha_imag,x_fill,incx,y_fill,incy,ap_fill,expect_result,mere_threshold,mare_multiplier,random_seed` |
| 填充 token | `RANDOM_NORM_5_5`（U[−5,5]）、`RANDOM_GAUSS_0_1`（N(0,1)）、`RANDOM_ALTER`、`RANDOM_EXTREME`、`RANDOM_DIAG_IMAG`、`VALUE_NORM_{0,INF,NAN}`、`NULLPTR` |
| token 实现位置 | `RANDOM_GAUSS_0_1` 与 `RANDOM_DIAG_IMAG` 在算子测试侧实现（公共 `test/frame/fill.h` 的 `METHOD_PATTERN_VAL` 规则中 `GAUSS` 属 Method 而非 Pattern，无法表达；且不宜为单算子改动全仓公共文件） |
| 对角虚部注入 | 仅当 `ap_fill == RANDOM_DIAG_IMAG` 时注入有限非零值，不再对全部用例无条件写 NaN |
| 精度判定 | 实部、虚部分别统计；atol=rtol=2⁻¹³、matched_ratio≥0.99、max_abs_error≤1e-2；另断言对角虚部严格为 0 |
| 性能打点 | `aclblasChpr2_npu_benchmark`：warmup 5 次不计时，`aclrtEvent` 夹住 10 次下发取平均；用例名以 `TC_PF` 开头时打印 `[PERF] case=<name> n=<n> avg_us=<v> warmup=5 iterations=10` |
| 构建 | `bash build.sh --ops=chpr2 --soc=ascend910b3`（族级 `--ops=spr2` 不覆盖本算子） |

### 验收口径

| 项 | 口径 |
|---|---|
| 精度 | 任务书 §3.2：COMPLEX64 按 FLOAT32 分量判定，实部/虚部分离 |
| 性能（正式） | `msprof op` 采集 kernel `Task Duration(us)`；warmup 5 + 10 次有效样本取均值；判据 `NPU_kernel_avg_us ≤ 1.25 × gpu_ms × 1000` |
| 性能（交叉验证） | `verify_performance.py` 解析 `[PERF] avg_us`，与 `gpu_baseline.csv` 比对；含下发开销，仅作趋势判断 |
| 用例规模 | 精度 1000 条、性能 200 条；性能子集与 `gpu_baseline.csv` 200 条逐条对应（按 uplo/n/incx/incy 四元组匹配） |

### 自验证结果

**实机验证（Atlas A2 / Ascend 910B3，CANN 9.0.0，CANNLab）**

| 项 | 方法 | 结果 |
|---|---|---|
| 编译 | `./build.sh --ops=chpr2 --soc=ascend910b3` | 通过 |
| 精度全量 | `chpr2_test --gtest_filter=-*TC_PF*`；golden 为 Netlib `cblas_chpr2`，实部/虚部分别比对 | **1001/1001 通过**（1000 条 CSV + 1 条空 handle 负向），耗时 338~340 s |
| 性能（正式） | `msprof op` 采集 kernel `Task Duration(us)`，每条 2 次取中位数；判据 `gpu_us / npu_us ≥ 0.8` | 带步长 12 条全部达标（最低 1.14）；n=1/2 为 3.52 us（倍率 1.44 / 1.51）；全量 200 条逐条结果见自测报告 `3.1 / 3.2` |
| 性能（交叉验证） | `aclrtEvent` 事件计时（含下发开销）对 `gpu_baseline.csv` | 189/200 ≥ 0.8；未达 0.8 的 11 条均为 n ≤ 256 的单位步长用例，事件耗时 11~15 us 由下发开销支配，kernel 口径下均达标 |
| 回归纪律 | 每次 kernel 改动后重跑精度 1001 条 | 全部通过 |

优化前后对照（事件计时，同一台机器）：

| 用例 | n / incx | 优化前 | 优化后 |
|---|---|---|---|
| TC_PF_1006 | 1024 / 8 | 0.14 | 1.26 |
| TC_PF_1150 | 2404 / 8 | 0.51 | 1.93 |
| TC_PF_1007 | 4096 / 127 | 1.02 | 2.56 |
| TC_PF_1158 | 3246 / 64 | 0.88 | 2.69 |

**离线验证（提交前完成，不依赖 NPU）**

| 项 | 方法 | 结果 |
|---|---|---|
| Host 校验契约 | 将两段校验抽为纯 C++，用 CSV 中全部负向与 `n=0` 用例驱动 | 22/22 通过（含 n=0 叠加非法 uplo/零步长/INT_MIN/空 alpha、n<0 叠加非法 uplo） |
| packed 索引与分核覆盖 | 离线模拟 `Chpr2ColBase`/`Chpr2VecPhysIdx`/`Chpr2ColOf` 与 tile 分段，n ∈ [1,64] 全量 + 127…4096 边界档 × 双 uplo × 多档 numBlocks | 544 组全过：偏移无越界/重复/空洞，列分配与 tile 分段各覆盖恰好一次 |
| 内存规模 | n=4096 时 AP 单份 64 MB | 测试工程 host 侧约 192 MB + device 64 MB，在 512 MB 预算内 |

## 已知限制与规避

| 限制 | 说明 | 规避 |
|---|---|---|
| `alpha` 仅支持 Host 标量指针 | ops-blas 的 BLAS handle 未提供 pointer mode（仅 blasLt 有），仓内 BLAS 算子一律 host 解引用标量；传入 Device 指针会非法访问 | 在算子 README 明确该边界；如需 Device 指针，需先为 BLAS 侧引入 pointer mode，超出本任务接口范围 |
| `n=0` 语义与 `sspr2` 不同 | 本算子按任务书在结构校验之后 quick return；`sspr2` 在参数校验之前返回 | 设计文档与 PR 说明中显式记录该差异及其依据（任务书 §2.5） |
| 对角虚部非零用例仅 1 条 | 任务书 §3.5 要求"包含"该场景，当前仅 `TC_ED_231` | 如需加强，可按 `RANDOM_DIAG_IMAG` token 追加用例，无需改代码 |

## 文件清单

| 文件 | 动作 | 说明 |
|---|---|---|
| `blas/spr2/arch22/chpr2_host.cpp` | 改 | 两段式校验；`uplo` 改 `INVALID_ENUM`；`n=0` 后移；启动 block 数与逻辑分核数解耦 |
| `blas/spr2/arch22/chpr2_kernel.cpp` | 改 | 仅 `LoadVectorTile`：步长分档装载 + 向量 `Gather` 解交织；列内更新与索引表不变 |
| `blas/spr2/arch22/chpr2_tiling_data.h` | 改 | 新增 `CHPR2_ARCH22_STRIDED_CHUNK_MAX`、`CHPR2_ARCH22_MIN_LAUNCH_BLOCKS` 两个策略常量 |
| `test/spr2/chpr2/arch22/chpr2_test.cpp` | 改 | 新增 2 个 token 分支；对角注入改条件触发；新增 `[PERF]` 打点 |
| `test/spr2/chpr2/arch22/chpr2_npu_wrapper.h` | 改 | 新增 benchmark 接口（事件计时 + warmup） |
| `test/spr2/chpr2/arch22/chpr2_test.csv` | 替换 | 任务书规格 1200 条 |
| `blas/spr2/README.md` | 改 | 校验顺序、`INVALID_ENUM`、`n=0` 语义、arch22/arch35 实现分述、标量指针边界 |

未改动：`test/frame/fill.h`、任何 `CMakeLists.txt`、`include/` 下头文件。

## 关联链接

- 任务书：`aclblasChpr2_AtlasA2_A3_task_doc.md`
- 对标接口：<https://docs.nvidia.com/cuda/cublas/index.html>（`cublasChpr2`）
- BLAS 参考实现：<https://www.netlib.org/blas/chpr2.f>
- 精度标准：<https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>
- ops-blas 仓：<https://gitcode.com/cann/ops-blas>
- 同族参考实现：`blas/spr2/arch35/chpr2_host.cpp`、`blas/spr2/arch22/sspr2_host.cpp`
- 同族测试范式：`test/her2k/cher2k/arch22/cher2k_test.cpp`（`[PERF]` 打点）、`test/her2k/cher2k/arch22/cher2k_npu_wrapper.h`（事件计时）
