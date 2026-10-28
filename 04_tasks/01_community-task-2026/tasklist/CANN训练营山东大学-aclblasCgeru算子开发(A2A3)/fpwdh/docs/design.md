# aclblasCgeru 算子设计文档（Atlas A2/A3）

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| V1.0 | 2026-10-24 | 初始版本 |

# 需求背景（required）

## 需求来源

本需求来源于 CANN 社区任务「CANN训练营山东大学-aclblasCgeru算子开发(A2A3)」。任务书要求在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3 系列产品，架构 arch22）上使用 Ascend C kernel 直调方式实现单精度复数无共轭秩 1 更新 `aclblasCgeru`，完成算子设计、开发、测试全流程，验收通过后合入 [ops-blas](https://gitcode.com/cann/ops-blas)（目录 `blas/ger/`），测试代码合入 `test/ger/cgeru/arch22/`。

- CANN 版本：9.1.0
- 性能测机：Atlas 800T A2（910B3）
- 实现目录：`blas/ger/arch22/`（A2/A3 共用 arch22）

## 背景介绍

### aclblasCgeru 算子功能

列主序（Column-Major）复数矩阵原地更新：

```
A = alpha * x * y^T + A
```

逐元素：`A[i][j] = alpha * x[i] * y[j] + A[i][j]`（i=0..m-1，j=0..n-1，含步长时按 incx/incy 索引）。**y 不取共轭**，虚部符号不取反——这是 geru 与同族 `aclblasCgerc`（`A = alpha * x * y^H + A`）的唯一语义差异，实现与 golden 比对公式均不得引入共轭。

| 项 | 内容 |
|----|------|
| 输入 | alpha（Host 复数标量）、x（Device，m 元素 COMPLEX64）、y（Device，n 元素 COMPLEX64） |
| 输出 | A（Device，lda×n COMPLEX64，原地更新前 m×n 部分） |
| 数据类型 | COMPLEX64（实部/虚部各 float32，`include/cann_ops_blas_common.h` 的 `aclblasComplex`） |
| 存储布局 | A 列主序；x/y 按 incx/incy 步长索引（支持正/负步长） |
| 广播 | 不涉及 |

### 现状分析

| 现状 | 差距 |
|------|------|
| `include/cann_ops_blas.h` L148 已有 `aclblasCgeru` 声明（950PR 版本合入时新增，PR #454） | 声明已存在，与 950PR 同名接口共用，本任务无需改动头文件 |
| `blas/ger/arch22/` 已有 `aclblasSger`（实数 GER 三件套范式） | 无复数无共轭 GER 的 arch22 实现 |
| `blas/gerc/arch22/` 已有 Cgerc（共轭版） | 共轭语义，且不支持 incx/incy/lda 通用场景，golden 也不可复用 |
| `blas/ger/arch35/` 已有 950PR 版 Cgeru | arch35 为 SIMT/VF 编程模型，arch22 不可复用，需按 arch22 向量模型实现 |

本实现不定义产品私有平行接口，与 950PR 同名接口共用 `include/cann_ops_blas.h` 中的同一声明。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22）上实现与 cuBLAS `cublasCgeru` / Netlib `cgeru` 语义一致的句柄式 BLAS 接口：COMPLEX64、列主序、正/负步长、lda padding、m=0/n=0 与 alpha=(0,0) no-op、handle 绑定 stream 异步执行、参数非法返回规定状态码。

## 算子原型

```cpp
aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);
```

## 参数说明

与任务书 §2.4 参数表一致：

| 参数 | 异常行为 |
|---|---|
| handle | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（首查） |
| m / n | < 0 → `ACLBLAS_STATUS_INVALID_VALUE`；= 0 → 合法 no-op，返回 SUCCESS 不计算 |
| alpha | nullptr → `ACLBLAS_STATUS_INVALID_VALUE`；(0,0) → 不写 A，返回 SUCCESS |
| incx / incy | = 0 或 INT_MIN → `ACLBLAS_STATUS_INVALID_VALUE`；支持负步长（从尾部反向索引） |
| x / y / A | m>0 且 n>0 且 alpha 非零时为 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | < max(1, m) → `ACLBLAS_STATUS_INVALID_VALUE` |

## 需求拆解

1. Host 侧：参数校验（含上述全部异常路径与 OP_LOGE 日志）、二维分块 tiling、kernel 直调 launch（handle 绑定 stream，纯异步）。
2. Kernel 侧：arch22 AIV 向量模型；连续访存（incx=incy=1）快速路径 + 通用步长路径（±1/±2/±3 任意组合、lda padding）。
3. 精度：COMPLEX64 实部/虚部按 FLOAT32 分量判定，rtol=atol=2^-13（1.2e-4），required_matched_ratio=0.99，max_abs_error_limit=1e-2 or 32*ULP；golden 由 cblas（Netlib cgeru，y 不共轭）生成，全矩阵比对。
4. 性能：910B3 实测平均单次耗时（warmup 5 次、采样 >50 次取平均）不高于标杆：512²≤5.7us、1024²≤10.1us、2048²≤55.71us、4096²≤219.35us。
5. 交付：设计文档 PR 合入、测试代码（`test/ger/cgeru/arch22/`，CSV 驱动 GTest）、自测报告、个人仓（邀 Ascend-CANN 为开发者）。

# 详细设计（required）

## 算子分析

### 数学公式

对每一列 j（0..n-1）：

```
coeff_j = alpha * y[j]                 # 无共轭：y[j] 直接参与乘积
A[i][j] = A[i][j] + coeff_j * x[i]     # i = 0..m-1
```

复数展开（cr/ci 为 coeff_j 实/虚部，xr/xi 为 x[i] 实/虚部）：

```
A.r += cr*xr - ci*xi
A.i += cr*xi + ci*xr
```

对照共轭版 gerc（`coeff_j = alpha * conj(y[j])`）：实部公式相同，虚部第二项符号相反。kernel 中该差异体现为虚部合成的一次 `sub` ↔ `add`，实现与 golden 均按上式（无共轭）落实。

**特殊值语义**（对齐 cublas/Netlib 的 IEEE 行为）：alpha 实部或虚部为 0 时跳过对应 0×Inf/0×NaN 分量乘，避免引入 cblas 不会产生的 NaN；乘加不做 FMA 收缩（乘、加分离发射），保证 0×Inf=NaN 等 IEEE 结果与 golden 一致。Inf/NaN 用例按「golden 与算子同公式」对拍，matched_ratio 与 max_abs_error 按任务书阈值判定。

### 支持数据类型与形状

COMPLEX64。m、n 为运行时入参（m,n ≥ 0，测试覆盖 1~4096）；A 列主序 lda×n，仅更新前 m×n；x 物理长度 1+(m-1)·|incx|，y 物理长度 1+(n-1)·|incy|。

## 算子实现

### Host 侧（`blas/ger/arch22/cgeru_host.cpp`）

校验顺序（每个错误路径带 OP_LOGE）：

```
handle==nullptr → HANDLE_IS_NULLPTR
→ m<0 / n<0 → INVALID_VALUE
→ alpha==nullptr → INVALID_VALUE
→ incx==0||INT_MIN / incy==0||INT_MIN → INVALID_VALUE
→ lda<max(1,m) → INVALID_VALUE
→ m==0 || n==0 → SUCCESS（no-op，不 launch）
→ alpha==(0,0) → SUCCESS（不写 A，不 launch）
→ x/y/A==nullptr → INVALID_VALUE
→ 寻址范围溢出检查（size_t 全量下标校验）→ INVALID_VALUE
```

tiling 与分核：

- `GetAivCoreCount()` 取可用 AIV 核数；
- 二维分块（rowBlocks × colBlocks，乘积 ≤ AIV 核数）：在「最大化占用核数」前提下最小化单块元素数、其次最小化 x/y 缓存拷贝代价、优先更多列块（列块内 A 按列连续，DMA 效率高）；
- 列主序下每个列块内 A 逐列连续（长度 m），天然适合按列组搬运；行块数覆盖 m≫n 的窄矩阵场景；
- 负步长起点 host 侧折算：`xStart=(incx<0)?(m-1)·|incx|:0`、`yStart` 同理，kernel 统一按下标正算；
- tiling 结构体（`cgeru_tiling_data.h`）：m/n/lda/alpha 实虚/incx/incy(int64)/xStart/yStart/rowBlocks/colBlocks/tilingKey；kernel 以值传参，无 GM workspace（全部使用 UB）。

launch：`cgeru_kernel_do(x, y, A, tiling, numBlocks, handle->stream)`，单次 launch，库内不做任何 `aclrtSynchronize*`（异步语义，读回前由调用方同步 stream）。

### Kernel 侧（`blas/ger/arch22/cgeru_kernel.cpp`）

单一 kernel 入口 `cgeru_kernel`（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`），类封装 + `TPipe`/`TQue`（VECIN/VECOUT/VECCALC，BUFFER_NUM=2 ping-pong），处理流程 CopyIn → Compute → CopyOut，事件 `MTE2_V` / `V_MTE3` / `MTE3_MTE2` 同步——与仓内 `sger`（arch22）范式一致。**A 的写回一律 MTE3 DataCopy/DataCopyPad**（arch22 多核标量 GM 写不可靠，见仓内既有实践）。

按 block 的 tile 划分：`blockIdx → (rowBlock, colBlock)`，行均匀余数分配，与 arch35 版相同的通用 tile 折算。

**数据搬运（按步长分三种拷贝模式）**：

- inc==1：连续 `DataCopyPad`；
- inc>1：突发-间隙 `DataCopyPad`（nBurst=元素数、lenBurst=8B、srcGap=(|inc|-1)·8B）；
- inc<0：kernel 内按 `(len-1-idx)·|inc| + start` 折算下标，标量读入 UB（读 GM 标量安全；写回仍走 MTE3）。

A 块搬运：列组（G 列）× 行段（R 行）二维突发：`nBurst=G、lenBurst=R·8B、srcGap=(lda-R)·8B、dstGap=对齐 padding`；lda==m 且对齐时合并为单长突发。G、R 按 192KB UB 预算静态折算（预留 x/y/临时缓冲）。

**计算（交错域 pair-swap 法，x 每行段预处理一次）**：

1. 行段内 x 拷入 UB 后构造两个向量：`xKeep`（原样交错 [xr,xi,…]）与 `xSwapNeg`（对内交换并虚部取负 → [-xi,xr,…]，由 Gather 交换模板 + 常量向量 [-1,1,…] 逐元素乘实现，模板/常量每行段构造一次）；
2. 每列 j 从 UB 中的 y 取 `y[j]`，标量合成 `coeff_j = alpha·y[j]`（含纯实/纯虚 shortcut 与 IEEE 特殊值路径）；
3. 列更新（每列 4 条向量指令，全交错域，A 无需解交织）：

```
t1 = xKeep   * cr        (Muls)
t2 = xSwapNeg * ci       (Muls)
u  = t1 + t2             (Add)     # [cr·xr - ci·xi, cr·xi + ci·xr] ✓ 无共轭
A列 = A列 + u             (Add)
```

4. MTE3 写回（源突发参数镜像拷入参数）。

**流水**：x/y 预处理与 A 列组搬运重叠（ping-pong 双缓冲），y 的 MTE2 完成后再参与计算，先算当前块再预取下一块。

**性能预算（910B3，HBM 带宽量级 1.6TB/s）**：

| case | 数据量（读A+写A） | 标杆 | 需有效带宽 | 路径 |
|---|---|---|---|---|
| 512² | 4.2MB | 5.7us | 0.74TB/s | 快路径，512 融 8 列突发，launch 开销占比高 |
| 1024² | 16.8MB | 10.1us | 1.66TB/s | 快路径，L2 常驻（16.8MB 工作集），融合 4 列 |
| 2048² | 67MB | 55.71us | 1.20TB/s | 快路径，多核列并行 |
| 4096² | 268MB | 219.35us | 1.22TB/s | 快路径，多核列并行，纯流式 |

全部性能 case（任务包 TC_PF 200 条）均为 incx=incy=1 连续访存，走快路径。

### 目录与文件结构（对齐仓内 sger 范式，不新增多余文件）

```
blas/ger/arch22/cgeru_host.cpp        # Host：校验/tiling/launch
blas/ger/arch22/cgeru_kernel.cpp      # Kernel：AIV 向量实现（含拷贝模式分支）
blas/ger/arch22/cgeru_tiling_data.h   # tiling 结构与常量
test/ger/cgeru/arch22/cgeru_test.cpp        # CSV 驱动 GTest（复用 test/ger/cgeru 公共层）
test/ger/cgeru/arch22/cgeru_npu_wrapper.h   # NPU 调用包装
test/ger/cgeru/arch22/cgeru_perf.cpp        # 性能采集
test/ger/cgeru/arch22/cgeru_test.csv        # 1200 条用例（任务包提供格式）
test/ger/cgeru/arch22/gpu_baseline.csv      # 性能基线（gpu_ms，阈值=×1.25）
```

`include/cann_ops_blas.h` 不改动（声明已存在）；`blas/ger/README.md` 产品支持表补 A2/A3「支持」。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I A3 | √ |
| Ascend 950PR/DT | 已有 arch35 实现，不在本任务范围 |

# 算子约束限制

- m≥0、n≥0；incx≠0、incy≠0（含 INT_MIN 防护）；lda≥max(1,m)；alpha 不可为 nullptr
- 不支持超出 incx/incy/lda 语义的额外非连续内存访问
- 无广播；A 原地覆写，不返回视图；不涉及 dynamic shape 与确定性计算特殊要求
- 异步执行：依赖 handle 绑定 stream，读回 Device 结果前须同步 stream
- COMPLEX64 单一数据类型

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | golden=cblas（Netlib cgeru，**y 不共轭**）单标杆；COMPLEX64 实部/虚部按 FLOAT32 分量：rtol=2^-13、atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 or 32*ULP；逐元素 `|a-g| ≤ atol + rtol·|g|`；全矩阵比对 | 任务书 §3.2 + 生态算子开源精度标准 |
| 性能 | 910B3；warmup 5 + >50 次平均；msprof `OpBasicInfo.csv` kernel `Task Duration` 口径；512/1024/2048/4096 方阵 ≤ 5.7/10.1/55.71/219.35 us；参考全表 200 条 ≤ gpu_baseline.csv·1.25 | 任务书 §3.3 + 测试指导 |
| 内存 | 不涉及（无 GM workspace） | 任务书 §3.4 |

## 测试方案

- 框架：复用 ops-blas `test/ger/cgeru/` 公共层（`cgeru_golden.h`/`cgeru_param.h`/CMakeLists），新增 `arch22/` 子目录（结构与已合入的 `arch35/` 一致）；CSV 驱动 GTest，golden 由测试工程内 cblas（Netlib 复数实现）生成。
- 用例：任务包 `cgeru_test.csv` 1200 条——精度 1000 条（L0 基础 6 / 尺寸扫描 23 / 标量 8 / 矩形 12 / 前导维+步长 39（±1/±2/±3 全组合 36）/ 填充含 Inf/NaN/全零/交替/极端 18 / 边界负向 16 / 扩展 878）+ 性能 200 条（含 4 条任务书典型 case）。
- 自验报告：精度逐条 PASS/FAIL + 实/虚部分别统计截图；性能 4 典型 case + 全表对照；二次运行一致性抽测。

## 兼容性分析

新增 arch22 实现挂接已有公共声明，不修改任何已合入算子（`aclblasCgerc`/`aclblasSger`/arch35 `aclblasCgeru`）的语义与代码；`blas/ger/README.md` 仅追加支持表条目。
