# 需求背景（required）

## 需求来源

CANN 训练营哈工深专场社区任务“aclblasChemv 算子开发（950）”。依据随任务提供的
`aclblasChemv_Atlas950PR_task_doc.md`、`test_cases/`（`chemv_test.csv`、
`gpu_baseline.csv`、精度与性能校验脚本）及任务方后续澄清实现。

目标仓：<https://gitcode.com/cann/ops-blas>，基线 commit
`8081a87d2a6f926c85705b9a2a90bd22b9edc9e8`。该基线已有 `aclblasSsymv`、`aclblasCsymv`，
尚无 `aclblasChemv`。

## 背景介绍

### aclblasChemv 算子实现优化

在 Ascend 950PR 上使用 Ascend C 编程语言补齐 ops-blas 的单精度复数 Hermitian
矩阵-向量乘接口 `aclblasChemv`，通过 handle 绑定 stream 直调 kernel 实现，代码放在
`blas/symv/arch35/`（与实数参考算子 `aclblasSsymv` 同族目录）。

### 参考算子现状分析

| 参考算子 | 可复用 | 不可复用 |
| --- | --- | --- |
| `aclblasSsymv`（arch35） | handle/stream 绑定、kernel 直调框架、GTest 工程组织 | 实数语义，无共轭 |
| `aclblasCsymv`（arch35） | RegBase tile、64 列列切分、UB 预算写法、测试 CSV 结构 | 复数**对称**矩阵，不对角虚部置零、不做共轭；其 RegBase 先搬完整矩形 tile 再掩码对角，违反“未引用三角不得读取” |

Chemv 为 Hermitian（`A = A^H`），对角虚部假定为 0、不参与计算，未引用三角不得读取。
因此不能通过改名调用 Csymv 得到正确语义。

# 需求分析（required）

## 需求描述

实现与 cuBLAS `cublasChemv` 参数序列一致、语义对齐 Netlib `chemv.f` 的接口：

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

计算 `y = alpha * A * x + beta * y`，A、x、y、alpha、beta 均为 complex64。

## 需求拆解

1. 公共 API 声明放入 `include/cann_ops_blas.h` 供其他产品线共用，不定义 950PR 私有接口。
2. 功能语义：仅读 `uplo` 指定三角；未引用部分按共轭对称推断；对角仅取实部；
   列主序 `lda×n` 存储；`incx`/`incy` 支持正负非零步长。
3. 参数校验顺序严格按任务书 §2.5，不沿用 Netlib 源码顺序。
4. 快速返回：`n=0` 直接成功；`alpha=(0,0)` 且 `beta=(1,0)` 直接成功。
5. 数据访问约束：`alpha=0` 不读 A/x；`beta=0` 不读旧 y；未引用三角不得读取。
6. 精度：Netlib BLAS 3.10.0 `cblas_chemv` 单 golden，实虚分别按 FLOAT32 标准判定。
7. 性能：每性能 case 平均单次耗时 ≤ 2.5× H100 基线。
8. 交付：算子 README、测试工程与 CSV、自测报告、设计文档评审、私仓代码。

# 详细设计（required）

## 算子分析

### 数学公式

```
y = alpha * A * x + beta * y,  A = A^H (Hermitian)
```

对逻辑行 r、列 c：
- 存储三角内：`A[r][c]` 取自 `a[r + c*lda]`
- 未存储三角：`A[r][c] = conj(a[c + r*lda])`
- 对角：`A[r][r].real = a[r + r*lda].real`，虚部忽略且**不载入**

Netlib 的逐输出依赖顺序（顺序随 uplo 不同）：

```text
UPPER：对角  → temp2 分组（j<r）→ 直接项（j>r）
       y[r] = beta*y[r] + real(alpha*x[r])*d + alpha*temp2 + sum_{j>r} temp1[j]*A[r][j]
LOWER：直接项（j<r）→ 对角  → temp2 分组（j>r）
```

### 支持数据类型

输入输出均为 `aclblasComplex`（complex64，实部/虚部各 float32）；`n`、`lda`、
`incx`、`incy` 为 int；内部按 FP32 复数运算。

### 支持形状

A 为 `lda×n` 列主序；x、y 为 n 元素向量，物理长度至少 `1+(n-1)*|inc|`。
不涉及广播、非连续 tensor、batch 或视图。

## 算子实现

### 实现方案

#### Host 侧设计

**参数校验顺序**（严格按任务书 §2.5）：

```text
handle 判空 → n<0 → n==0 返回成功 → uplo/lda/incx/incy/alpha/beta 指针校验
→ 读标量后判定 alpha=0 且 beta=1 的 quick return → A/x/y 判空 → 地址跨度可表示性
```

`lda < max(1,n)`、`incx==0`、`incy==0`、`INT_MIN` 步长、以及字节跨度不可表示时返回
`ACLBLAS_STATUS_INVALID_VALUE`；非法 uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM`。

**标量来源**：alpha 按任务书参数表为 Host；beta 支持 Host/Device 两种。
Device beta 且 alpha≠0 时由 kernel 在绑定 stream 内读取，不引入同步；
Device beta 且 alpha=0 时，为正确判定 quick return 及 y 判空，需先按 stream 顺序
同步再 D2H 解析 beta。该同步例外与 `INT_MIN` 拒绝均为任务书未逐条明确的实现决策，
已在设计评审中提请确认。

**分核**：C 为平台返回的 AIV 核数（`GetAivCoreCount()`，为 0 时返回错误，不硬编码核数）。

| 条件 | 路径 | 分核 |
| --- | --- | --- |
| n < 16 或 alpha=0 | 基础 SIMT | `B=min(C,ceil(n/128))`，线程独占整行 |
| 16 ≤ n ≤ 1024 | 有序 RegBase | `rowTile=64`，`B=min(C,ceil(n/64))` |
| n > 1024 | 有序 RegBase | 8 行对齐均衡行块，`B=min(C,ceil(n/32))` |

n>1024 时 `waves=ceil(n/(64C))`、`balancedRows=ceil(n/(C*waves))`、
`rowTile=min(64,8*ceil(balancedRows/8))`；kernel 侧按商余把 `ceil(n/32)` 个 32 行单位分给各
block，余数分到直接项较多的一端（UPPER 低行、LOWER 高行），block 内以 64 行为步长。
所有行区间互不重叠且覆盖 `[0,n)`，每行只有一个写者。

`n≤1024` 强制用满 64 行块：若按均衡公式，中小尺寸会退化为 8~24 行窄块
（实测核数下 n=64/512/1024 分别落到 8/16/24 行，即通道占用 12.5%/25%/37.5%），
且每个 block 重复一遍 O(n) 的 x staging；固定 64 行后这些尺寸通道占用恒为 100%。

#### Kernel 侧设计

单 kernel、输出行独占，无 GM workspace、无核间同步、无原子操作。

**列切分**：固定 64 列一个 tile，列 j 始终递增累加，保持 Netlib 的 FP32 依赖顺序，
不做跨列树规约或部分和重排。

**tile 分类与搬运**：
- 完全位于直接三角（UPLO=UPPER 时 r<c，LOWER 时 r>c）：列主序整块 DMA
  （`asc_copy_gm2ub_align`，每列 `rows` 个连续复数），与访存布局一致，
  消费端用 `LoadAlign<DIST_DINTLV_B32>` 对齐拆分实虚。
- 完全位于镜像三角：行主序 DMA + 每行 8 字节 padding 破坏 512B 银行冲突，
  计算时用寄存器 `Gather` 交换实虚并按符号位 XOR 施加共轭。
- 对角 tile：**始终走受保护标量 SIMT staging**，逐元素判定 row/col 关系，
  只载入合法三角与对角实部，绝不留出所选三角；对角虚部从不载入。

**staging 线程数（关键参数）**：对角 tile staging 取 **1024 线程**（每线程 4 次载入）。
取 128 线程时每线程串行 32 次标量 GM 读，完全暴露在内存延迟下；n=4096 有 64 个对角 tile、
共约 52.4 万次标量读，按 6 cycle/read @2GHz 估约 28 μs，与当时缺口（134−101.8=32 μs）吻合。
改为 1024 线程后该串行链缩短至 1/8，n=4096 由 134 μs 降至 88.65 μs（msprof op 实测；
同尺寸全量性能测试落在 83.1~89.9 μs）。
加到 2048 线程无进一步收益（已饱和）。该参数与 n 无关，全尺寸统一。

**UB 预算**：`4*(MB*64*(64*2+8) + 4R + 4*64)` 字节，`R=n<=4096?64*ceil(n/64):64`，
`MB=n>2048?2:1`。各尺寸实际占用：n=1024 为 51.0 KiB、n=2048 为 67.0 KiB、
n=4096 为 **133.0 KiB**。平台 UB 总量 248 KiB（`kernel_constant.h` 的 `UB_SIZE`），
n=4096 扣除该动态区与 8 KiB 预留后剩余 107 KiB 供 Data Cache 与编译器 spill 使用。
单一 `extern __ubuf__ char dynamic_buf[]`，launch 明确传入计算后的动态 UB 大小
（`chemv_kernel_do` 的 `orderedUbBytes`），不与 TPipe 混用。

**流水与同步**：n>2048 时矩阵 tile 双缓冲（EVENT_ID1 交叉），下一 tile 的 DMA 与当前 VF
重叠；其余尺寸单 buffer 串行。DMA 使用 `V_MTE2`/`MTE2_V` 事件保护复用与完成，
VF 之间用 `PIPE_V` barrier 保证生产/消费顺序。

**特殊语义分支**：alpha=0 走缩放/置零路径，不读 A/x；beta=0 输出前完全不读旧 y；
beta=1 跳过不必要的复数乘法。不使用假定无 Inf/NaN 的 unsafe fast-math。

#### 已评估并实测证伪的方案

以下候选均经真机实测否定，保留结论以避免重复投入：

| 方案 | 实测结果 | 结论 |
| --- | --- | --- |
| 8 行单位负载均衡 | 1025–2048 段不达标由 16 例升至 36 例 | 40 行块需两趟 64-lane 列扫描，抵消均衡收益 |
| 80 行宽核（64 planar + 16 packed） | n=4096 由 110.9 升至 207.0 μs | 逐列 Gather 代价大于省下的第二趟扫描 |
| n>2048 用 128 列 tile | n=4096 由 115.8 升至 132.2 μs（3 轮可复现） | UB 足迹与 staging 成本超过减少的轮次 |
| 镜像 tile 一次性 SIMT 转置 | n=2048 由 42.7 升至 110.6 μs | 每 tile 重读整块，比寄存器 Gather 更贵 |
| n>3584 改 64 行单位 | 全量不达标数 68→73 | 均衡收益被每核两趟扫描抵消 |
| 消除全部寄存器 Gather（探针） | n=4096 仍 117–123 μs，与基线相同 | 证明 Gather 不占时间，该方向整体排除 |
| 二维行列切分 + workspace 归约 | 重构未收敛到正确结果 | 当前实现已满足性能要求，无需引入 workspace 与第二 kernel |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | ✅（arch35，CANN 9.1.0） |
| 其他产品 | 本实现未适配 |

## 算子约束限制

- 不涉及广播、非连续 tensor、batch、视图语义。
- 不提供 A/x/y 之间重叠别名的保证。
- 调用方须同步 handle 绑定的 stream 后再回读 y。
- 不使用 ACLNN 算子注册 tiling key，无私有平行接口。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 实部、虚部分别：`\|actual-golden\| ≤ atol + rtol*\|golden\|`，`atol=rtol=2^-13`；有限值 matched_ratio ≥ 0.99；max_abs_error ≤ 1e-2 | 任务书 §3.2 + 生态算子开源精度标准 mixed_tolerance_standard（FLOAT32 列） |
| 非有限值 | golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 一方 ±Inf 视为等效；非有限值不计入 matched_ratio，须全部满足等效规则 | 任务书 §3.2 |
| 性能 | 每性能 case 平均单次耗时 ≤ 2.5 × H100 基线；采样用 `msprof op` 采集单算子 task duration，5 次 warmup 不计入，随后有效样本取算术平均 | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

golden 为 Netlib BLAS 3.10.0 `cblas_chemv(CblasColMajor, ...)` 单标杆，
对角虚部按 Hermitian 语义置零；不自行编写参考实现替代。

## 兼容性分析

新增算子，不涉及既有接口兼容性；公共声明放入 `include/cann_ops_blas.h`，
与其他产品线共用同一 `aclblasChemv` API，不引入 950PR 私有接口。

## 可测试性

- 测试工程在 `test/symv/chemv/arch35/`：CSV 驱动 GTest，精度 golden 由 Netlib 实现生成。
- 覆盖：1000 条精度用例（uplo 全组合、尺寸扫描、标量特殊值、lda padding、
  ±步长、边界与负向用例、Inf/NaN）、200 条性能用例，参数与任务包逐条一致。
- 补充用例：null-handle 校验顺序、随机标量/特性交叉、比较器单测、
  非有限标量与退化形状、Host 标量快照与 Device beta 排队语义。
- 未引用三角与 padding 做 NaN 投毒，A/x 只读校验，y 存储间隙与首尾 guard 逐字节校验。
  投毒通过不单独作为“未读未引用三角”的证明，需结合 kernel 地址检查。

# 交付与验收计划

| 序号 | 交付件 | 状态 |
| --- | --- | --- |
| 1 | 设计文档（本文件，按官方模板） | 待提交 cann-competitions PR 评审 |
| 2 | 自测用例及测试代码（含 README 复现步骤） | 已完成 |
| 3 | 自测报告（参数、精度实虚截图、性能数据与截图） | 已完成 |
| 4 | 待验收代码地址（个人仓 + Ascend-CANN 开发者邀请 + 算子 README） | 待完成 |

后续：正式 PR 合入 `gitcode.com/cann/ops-blas` 的 `blas/symv/arch35/` 与
`test/symv/chemv/arch35/`，遵循 CONTRIBUTING 与 pre-commit 规范，
不包含冗余文件（已移除实验头文件与死模板路径）。
