# aclblasChemv（Atlas A2 / A3）算子设计文档

| 文档版本 | 日期 | 作者/团队 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-11-11 | m0_56676577 | Atlas A2/A3（arch22 / DAV_2201）四级自适应分派方案：小尺寸单核全驻 UB、中尺寸无同步 AIV 行条带、大尺寸交错实矩阵同构全 FP32 AIC Cube + AIV 双缓冲协同、非单位步长与极值/非有限数精确守卫 |

> **实现与测试位置**：个人 `ops-blas` 仓 `blas/symv/arch22/`（`chemv_tiling_data.h`、`chemv_host.cpp`、`chemv_kernel.cpp`）、公开头文件 `include/cann_ops_blas.h` 及测试目录 `test/symv/chemv/arch22/`。本 PR 仅提交设计文档，不包含算子源代码、测试代码或原始采样日志。

# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 2026——「11月社区任务-aclblasChemv算子开发（A2/A3）」。任务要求基于昇腾算子开源仓 [ops-blas](https://gitcode.com/cann/ops-blas) 工程框架，使用 Ascend C 算子直调模式开发单精度复数（`COMPLEX64`）厄米特（Hermitian）矩阵-向量乘法算子 `aclblasChemv`，适配 Atlas A2 系列（性能标定设备 Atlas 800T A2 / Ascend 910B3）与 Atlas A3 系列产品（`arch22` / `dav-2201`），接口签名与语义对齐 cuBLAS `cublasChemv` 及 Netlib BLAS `chemv` 参考实现。

## 背景介绍

### aclblasChemv算子实现优化

`chemv`（Complex single-precision Hermitian Matrix-Vector Multiply）是 BLAS Level 2 核心接口，广泛应用于量子化学哈密顿量作用、雷达阵列协方差矩阵处理、无线通信 MIMO 信道估计与科学计算特征值求解（如 Lanczos / Arnoldi 迭代）等场景。

相较普通复数矩阵-向量乘 `cgemv`，`chemv` 同时具有存储半角压缩与共轭对称反射两重特性：
1. **严格存储边界约束**：矩阵 $A \in \mathbb{C}^{n \times n}$ 按列主序存放于 $lda \times n$ 缓冲区中，仅 `uplo` 指定的上三角或下三角被引用，未引用三角可能包含脏数据（测试中显式注入 `NaN` 哨兵），物理 DMA 读取必须严格裁剪在有效三角内；对角元素满足 $A_{ii} = \overline{A_{ii}} \in \mathbb{R}$，其虚部存储槽位假定为 $0$ 且同样不得读取或参与 `0 * NaN` 运算。
2. **双向耦合访存与小矩阵延迟敏感**：每个非对角元素 $A_{ij}$ 既对顺列方向贡献 $A_{ij}(\alpha x_j)$，又对共轭反射方向贡献 $\alpha(\overline{A_{ij}} x_i)$。在小尺寸（$n \le 64$）场景下 GPU 基线耗时低至 $6 \sim 8\,\mu\text{s}$，受限于 Kernel 启动与跨核同步开销；在大尺寸（$n = 2048 \sim 4096$）场景下矩阵容量达 $16 \sim 64\,\text{MiB}$，纯粹依赖 Vector 逐列拆分交错复数易受 `GatherMask` 拆包吞吐与跨核归约带宽限制。因此需要结合 `dav-2201` 架构的 AIV Vector 与 AIC Cube（全 IEEE-754 FP32 模式）设计分档协同方案。

### aclblasChemv算子实现路径和相关API路径

- 公共接口声明路径：`include/cann_ops_blas.h`（新增 `aclblasChemv` 声明，全产品线共用）
- Host 分派与参数校验路径：`blas/symv/arch22/chemv_host.cpp`
- Tiling 结构定义路径：`blas/symv/arch22/chemv_tiling_data.h`
- Kernel 实现路径：`blas/symv/arch22/chemv_kernel.cpp`
- GTest 精度/性能测试工程路径：`test/symv/chemv/CMakeLists.txt`、`test/symv/chemv/arch22/`

### aclblasChemv算子现状分析

上游 `ops-blas` 仓现状如下：
- `blas/symv/arch22/` 已有实数对称算子 `aclblasSsymv`，仅处理 `float` 标量与实对称反射（无共轭变号、无实虚交错存储、无复数标量双源探测）；
- `blas/symv/arch35/` 面向新架构实现了 `aclblasCsymv`（复对称非共轭 $A = A^T$，对角元素含虚部，与厄米特矩阵 $A = A^H$ 数学定义不同）；
- `include/cann_ops_blas.h` 尚无 `aclblasChemv` 声明，`blas/symv/arch22/` 无 `chemv` 实现，`test/symv/chemv/CMakeLists.txt` 仅在 `arch35` 分支注册了 GTest 测试目标。

对标接口参数能力如下表所示：

| 参数 | 输入/输出/属性 | 参数含义 | 数据类型 | 排布/维度 | 约束与异常行为 |
| --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | ops-blas 库上下文句柄（绑定 `aclrtStream`） | `aclblasHandle_t` | Host 标量 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 属性 | 指定引用 `A` 的上三角或下三角 | `aclblasFillMode_t` | Host 枚举 | 仅支持 `ACLBLAS_UPPER` / `ACLBLAS_LOWER`，其余返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | 输入 | 厄米特矩阵 `A` 的阶数 | `int` | Host 标量 | `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` 直接返回 `SUCCESS` |
| `alpha` | 输入 | 复数标量乘数指针（运行时自动探测 Host/Device） | `const aclblasComplex*` | COMPLEX64 | `n > 0` 时 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | 输入 | 厄米特矩阵（仅引用 `uplo` 三角与实对角） | `const aclblasComplex*` | Column-Major, `lda × n` | `n > 0 && alpha != (0,0)` 时 `nullptr` 返回 `INVALID_VALUE` |
| `lda` | 输入 | 矩阵 `A` 的列主序前导维度 | `int` | Host 标量 | `lda < max(1, n)` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | 输入 | 输入复数向量 | `const aclblasComplex*` | `1 + (n-1)*|incx|` | `n > 0 && alpha != (0,0)` 时 `nullptr` 返回 `INVALID_VALUE` |
| `incx` | 输入 | 向量 `x` 的步长（支持正/负非零步长） | `int` | Host 标量 | `incx == 0`、`INT32_MIN` 或 `(n-1)*|incx| > UINT32_MAX` 返回 `INVALID_VALUE` |
| `beta` | 输入 | 复数标量乘数指针（运行时自动探测 Host/Device） | `const aclblasComplex*` | COMPLEX64 | `n > 0` 时 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | 输入/输出 | 原地更新复数向量（`beta=(0,0)` 时不读旧值） | `aclblasComplex*` | `1 + (n-1)*|incy|` | `n > 0` 且非 `(alpha==(0,0) && beta==(1,0))` 时 `nullptr` 返回 `INVALID_VALUE` |
| `incy` | 输入 | 向量 `y` 的步长（支持正/负非零步长） | `int` | Host 标量 | `incy == 0`、`INT32_MIN` 或 `(n-1)*|incy| > UINT32_MAX` 返回 `INVALID_VALUE` |

### aclblasChemv算子功能分析

计算公式：$y \leftarrow \alpha A x + \beta y$，其中 $A = A^H \in \mathbb{C}^{n \times n}$。
- 输入：`handle`、`uplo`、`n`、`alpha`、`A`、`lda`、`x`、`incx`、`beta`、`y`（当 $\beta \ne 0$）、`incy`
- 输出：原地覆写 `y` 的 $n$ 个逻辑元素（当 $|\text{incy}| > 1$ 时不修改元素间隙内存）
- 支持数据类型：`aclblasComplex`（单精度复数 `COMPLEX64`，实部/虚部各为 IEEE-754 `float32`）
- 支持广播：不涉及

# 需求分析（required）

## 需求描述

在 `ops-blas` 仓 `arch22` 路径下使用 Ascend C 直调模式实现 `aclblasChemv`，在 `include/cann_ops_blas.h` 中补齐公共接口声明，在 `test/symv/chemv/` 下补齐 `arch22` GTest 注册、正态分布补充用例与隔离式 Netlib BLAS 3.10.0 `cblas_chemv` 标杆校验。实现需严格遵守任务书定义的参数校验顺序、三角与实对角访问边界、Host/Device 标量自动识别及特殊值传播语义，精度满足生态算子开源精度标准（FLOAT32 分量独立判定），性能在 Atlas 800T A2（910B3）上达到标杆 GPU 的 $0.8\times$ 以上（即单算子平均耗时 $\le 1.25 \times \text{GPU baseline}$）。

## 需求拆解

1. **接口与校验完备性**：在 `include/cann_ops_blas.h` 新增标准 `aclblasChemv` 声明；严格实现「`handle` 判空 $\to n < 0 \to n == 0$ quick return $\to$ `uplo/lda/incx/incy/alpha/beta` 固定参数校验 $\to$ 标量读取与 `alpha==(0,0) && beta==(1,0)` quick return $\to$ `A/x/y` 条件判空」优先级；使用 `int64_t` 拒绝 `inc == INT32_MIN` 与 `(n-1)*|inc| > UINT32_MAX` 跨度溢出。
2. **Hermitian 物理隔离与数值语义**：仅对 `uplo` 指定三角发起 DMA，对角元素仅读取 4 字节实部（绝不加载虚部槽位）；`alpha == (0,0)` 时不读取 `A/x`，`beta == (0,0)` 时不读取 `y` 旧值，`beta == (1,0)` 时直接保留 `y` 旧值而不执行冗余复数乘法（避免单分量 `Inf` 触发 `0 * Inf = NaN` 跨分量污染）；保持与 Netlib `chemv.f` 一致的 $\alpha$ 结合位置与 $\beta y$ 先行累加顺序。
3. **全场景自适应高性能架构**：覆盖正/负任意步长、任意前导维 padding、Host/Device 标量及 $n = 1 \sim 4096+$ 全尺寸区间，无需调用方手工传入额外 workspace（大尺寸路径内部复用 `handle` 默认 workspace，分配受限时平滑回退零 workspace 路径）。
4. **可复现实验与双标尺验证**：通过 `dlmopen(LM_ID_NEWLM, ...)` 在独立动态链接命名空间加载 Netlib BLAS 3.10.0 `libblas.so.3.10.0`（杜绝进程内 OpenBLAS 符号劫持 `chemv_`）；补齐多尺度正态分布与白盒边界用例；完成精度与性能自验。

# 详细设计（required）

## 算子分析

### 数学公式

设复数标量 $\alpha = \alpha_r + i\alpha_i$，$\beta = \beta_r + i\beta_i$，向量逻辑下标 $k \in [0, n)$ 在正/负步长 $\text{inc}$ 下的物理元素偏移为：

$$
\text{pos}(k, \text{inc}) = \begin{cases} k \cdot \text{inc}, & \text{inc} > 0 \\ (1 - n + k) \cdot \text{inc}, & \text{inc} < 0 \end{cases}
$$

对于 `uplo = ACLBLAS_UPPER`，仅访问 $0 \le i \le j < n$ 的上三角元素，第 $r$ 个逻辑输出分量为：

$$
y_r \leftarrow \beta y_r + \operatorname{Re}(A_{rr}) (\alpha x_r) + \alpha \sum_{i=0}^{r-1} \overline{A_{ir}} x_i + \sum_{j=r+1}^{n-1} A_{rj} (\alpha x_j)
$$

对于 `uplo = ACLBLAS_LOWER`，仅访问 $0 \le j \le i < n$ 的下三角元素，第 $r$ 个逻辑输出分量为：

$$
y_r \leftarrow \beta y_r + \sum_{j=0}^{r-1} A_{rj} (\alpha x_j) + \operatorname{Re}(A_{rr}) (\alpha x_r) + \alpha \sum_{i=r+1}^{n-1} \overline{A_{ir}} x_i
$$

**交错实矩阵同构映射（大尺寸 Cube 直算数学基础）**：
列主序下形状为 $B \times B$（$B = 128$）的复数子块 $A_{\text{blk}}$ 在全局内存中物理存储为 $B$ 列、每列连续 $2B$ 个 `float32`（实虚交错 `[Re, Im, Re, Im, ...]`）。将其直接视为行主序形状为 $B \times 2B$ 的实矩阵 $M$（即 $M_{c, 2r} = \operatorname{Re}(A_{r,c})$，$M_{c, 2r+1} = \operatorname{Im}(A_{r,c})$），构造右乘辅助实矩阵 $X_{\text{refl}} \in \mathbb{R}^{2B \times 16}$（偶行填 $[x_r, x_i, 0, \dots]$，奇行填 $[x_i, -x_r, 0, \dots]$）与 $TX_{\text{stored}} \in \mathbb{R}^{B \times 16}$（填 $[(\alpha x)_r, (\alpha x)_i, 0, \dots]$）：
- **反射项**：$D = M \cdot X_{\text{refl}} \in \mathbb{R}^{B \times 16}$，其前两列恰好给出 $\sum_r \overline{A_{r,c}} x_r$ 的实部与虚部；
- **顺列项**：$S = M^T \cdot TX_{\text{stored}} \in \mathbb{R}^{2B \times 16}$，偶行第 0 列与奇行第 1 列之差给出实部 $\operatorname{Re}(A) (\alpha x)_r - \operatorname{Im}(A) (\alpha x)_i$，奇行第 0 列与偶行第 1 列之和给出虚部 $\operatorname{Re}(A) (\alpha x)_i + \operatorname{Im}(A) (\alpha x)_r$。

由此，内部非对角复数子块**无需 AIV 预先拆分/打包实虚平面**，AIC Cube 单元通过 `Nd2Nz` 一次读取原始列主序交错矩阵即可在关闭 `HF32` 的全 IEEE-754 FP32 模式下同时完成顺列与反射两路矩阵乘，HBM 矩阵读取流量严格保持 $1\times$ 半角体积（$n=4096$ 时约 $64\,\text{MiB}$）。

### 支持数据类型

- 矩阵 `A`、向量 `x`、向量 `y`、标量 `alpha`、标量 `beta`：`COMPLEX64`（`aclblasComplex`，实部/虚部均为 `float32`）
- 内部计算与累加精度：全路径采用 IEEE-754 `float32`（AIC Cube 显式调用 `SetHF32Mode(HF32Mode::DISABLE)` 禁用 TF32/HF32 截断）

### 支持形状

- 矩阵阶数 `n`：运行时 `int` 入参，支持 $n \ge 0$ 任意规模（含 $n=0$、质数、2 的幂及 $\pm 1$、非对齐尾块及 $n > 4096$ 超大尺寸）
- 前导维 `lda`：支持 $lda \ge \max(1, n)$ 紧凑及任意奇偶 padding（含超过 16-bit stride 上限的超大前导维回退）
- 步长 `incx`、`incy`：支持任意正/负非零整数步长（受 `int64` 物理跨度上限守卫保护）

## 算子实现

### 实现方案

#### 3.2.0 整体执行流程图

```text
aclblasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
  │
  ├─► 1. 参数分级校验：handle -> n<0 -> n==0(SUCCESS) -> uplo/lda/incx/incy/alpha/beta
  ├─► 2. 标量位置探测（缓存属性）：Host 直读 / 缺指针时同步探测 Device -> alpha==0&&beta==1(SUCCESS) -> A/x/y 条件判空
  │
  └─► 3. 运行时自适应四路径 Kernel 分派（绑定 handle->stream）
        ├── Path 1 [TINY_SINGLE] : incx=incy=1, n ≤ 48
        │     └─► 1 AIV Core，无 SyncAll/Workspace，整块有效三角驻留 UB 折半树归约直写 y
        ├── Path 2 [MED_ROW8/16] : incx=incy=1, 48 < n ≤ 512
        │     └─► 多核 AIV（8 行或 16 行条带独占），无 SyncAll/Workspace，2D DataCopyPad 读水平顺列带+垂直反射带+对角微块
        ├── Path 3 [MIX_CUBE128] : incx=incy=1, 512 < n ≤ 4096, 2*lda ≤ 65535, Workspace 就绪
        │     ├─► Phase 1 (AIV) : 分片构造 X/TX 右乘矩阵 + 错峰并行清洗对角块与末尾非对齐边缘块 + 有限安全域检测
        │     ├─► SyncAll<false>()
        │     ├─► Phase 2 (AIC) : 连续块区间均分 numTiles，L1/L0/CO1 全双缓冲 IEEE-754 FP32 Cube Mmad 计算顺列与反射分块
        │     ├─► SyncAll<false>()
        │     └─► Phase 3 (AIV) : 双缓冲流式线性 Add 合并分块结果 + Gather 抽取实虚部 + 叠加 beta*y 与实对角项写回 y
        │         （若 Phase 1 检测到 Inf/NaN 或 |val|>1e18 极值，自动切换至 Path 4 精确顺序计算）
        └── Path 4 [GENERIC]     : incx != 1 || incy != 1 || n > 4096 || alpha == 0 || 超大 lda
              └─► 多核 AIV 按逻辑行独占循环分块（CHUNK=256），严格遵循 Netlib 累加顺序与正/负步长映射
```

#### 3.2.1 host侧设计：

**tiling策略**：
`chemv_host.cpp` 首先完成任务书规定的分级参数校验，随后通过 `aclrtPointerGetAttributes` 判断 `alpha`、`beta` 位于 Host 还是 Device，并在 `handle` 内部缓存最近一次标量指针属性以消除重复驱动查询开销：当 `A/x/y` 均非空且标量在 Device 侧时，不在 Host 侧做阻塞 `aclrtMemcpy`，而是将设备指针直接传入 Kernel 通过 8 字节 `DataCopyPad` 异步读取，保证整条流零宿主停顿。

**任务均分与批量搬运**：
- 在中尺寸行条带路径（`MED_ROW8` / `MED_ROW16`）中，按输出行切分为宽度 $W \in \{8, 16\}$ 的独立行块，使总条带数 $\lceil n / W \rceil \le \text{AIV 核数}$，每个活跃 AIV 核仅承担唯一 1 个行条带并通过单条二维 `DataCopyPad` 批量搬运跨列数据；
- 在大尺寸混合路径（`MIX_CUBE128`）中，将全部 $F(F+1)/2$ 个 $128 \times 128$ 有效三角块（$F = \lceil n / 128 \rceil$）拉平为一维连续区间 `[tLo, tHi)` 等分给全部 AIC 核，核间块数差严格 $\le 1$，彻底消除三角行列长度递增带来的长尾不均。

##### 1. 分核策略：

Host 通过 `GetAivCoreCount()` 与 `GetAicCoreCount()` 动态查询硬件核数（910B3 为 40 AIV / 20 AIC）并缓存在 `handle` 中：
- `CHEMV_PATH_TINY_SINGLE`（$n \le 48$）：固定 `blocks = 1`，消除多核启动与同步开销；
- `CHEMV_PATH_MED_ROW8`（$48 < n \le 320$）：`blocks = min(ceil(n / 8), availableAiv)`；
- `CHEMV_PATH_MED_ROW16`（$320 < n \le 512$）：`blocks = min(ceil(n / 16), availableAiv)`；
- `CHEMV_PATH_MIX_CUBE128`（$512 < n \le 4096$）：以 `KERNEL_TYPE_MIX_AIC_1_2`（带 `__schedmode__(1)`）启动 `aicBlocks = min(max(numTiles, ceil(cleanBlocks / 2)), availableAic)` 个复合组（对应 $2 \times \text{aicBlocks}$ 个 AIV 子核）；
- `CHEMV_PATH_GENERIC`：`blocks = min(n, availableAiv)`，每核按 `row = blockIdx; row < n; row += cores` 独占输出行。

##### 2. 数据分块和内存优化策略：

**UB 内存规划（严格控制在 910B3 单核 192 KiB 上限内）**：
- `ChemvTinySingleKernel`：`matAosBuf`（32 KiB）+ 6 个计算平面 `planesBuf`（96 KiB）+ 向量与索引缓冲（< 6 KiB），峰值约 **134 KiB**；
- `ChemvMedRowBlockKernel<8/16>`：固定单次条带容量 `MAX_STRIP_COMPLEX = 4096`，`rawBuf`（32 KiB）+ `planesBuf`（96 KiB）+ 常驻 `x/tx` 向量（8 KiB）+ 广播/交织索引表（16.5 KiB）+ 对角微块缓冲（6 KiB），峰值 **158.5 KiB**；
- `ChemvMixAivCore<128>`：`cleanBuf`（128×256×4B = 128 KiB，Phase 1 用于整块清洗对角/边缘块，Phase 3 复用为 `stored/refl` Ping-Pong 归约缓冲）+ `workBuf`（32 KiB）+ `idxBuf`（25 KiB），峰值 **185 KiB**；
- `ChemvCubeEngine<128>`（AIC 侧硬件缓冲）：L1 双缓冲 `a1[2]`（256 KiB）+ `b1Refl[2]`（32 KiB）+ `b1Stored[2]`（16 KiB）= **304 KiB $\le$ 512 KiB**；L0A 双缓冲 `a2[2]` = **64 KiB $\le$ 64 KiB**；L0B 双缓冲 `b2[2]` = **8 KiB $\le$ 64 KiB**；L0C 双缓冲 `c1Refl[2] + c1Stored[2]` = **48 KiB $\le$ 128 KiB**。

**GM Workspace 规划**：
仅 `CHEMV_PATH_MIX_CUBE128` 使用 `EnsureDefaultWorkspace` 申请内部工作区（各段按 1024B 对齐）：包含安全回退标记段 `offFlag`、右乘矩阵段 `offXMat` / `offTxMat`、实对角紧凑段 `offDiagReal`、预清洗对角/边缘块段 `offClean`（最多 $2F-1$ 块，内部满块零拷贝直读原始 `A`），以及分块部分和段 `offRefl` / `offStored`。在 $n = 4096$（$F = 32$）最大场景下总计约 **21.0 MiB**，完全落在 `ops-blas` 默认 32 MiB 句柄工作区内且常驻 192 MiB L2 Cache；若用户绑定了极小自定义 workspace 导致容量不足，Host 自动无感回退至零 workspace 的精确路径。

##### 3. tilingkey规划策略：

直调算子通过按值传递轻量结构体 `ChemvTilingData`（64 字节以内）控制设备侧路径选择，由 `tiling.path` 字段区分 5 个独立编译的核函数入口（`chemv_tiny_single_kernel`、`chemv_med_row8_kernel`、`chemv_med_row16_kernel`、`chemv_mix_cube128_kernel`、`chemv_generic_kernel`），避免在单核函数内混合所有分支造成寄存器溢出与 I-Cache 污染。

**数据检测**：
在进入向量/Cube 批量归约前，使用无分支向量指令 `Mul(sq, v, v) + Sub(diff, sq, sq) + WholeReduceSum` 一次性检测输入向量/对角元是否包含 `Inf`、`NaN` 或 $|v| \ge 10^{19}$（平方上溢为 `+Inf` 进而 `Inf - Inf = NaN`）。常规有限数据零开销通过；一旦命中非有限值或接近 `FLT_MAX` 的极值（如 `2e38` 溢出抵消用例），立即转入 `ChemvGenericKernel` 按 Netlib 严格标量块顺序求值，杜绝填充零乘 `Inf` 产生伪 `NaN`。

#### 3.2.2 kernel侧设计：

##### 1. 小尺寸全矩阵驻留核（`ChemvTinySingleKernel`, $n \le 48$）
单 AIV 核运行，零跨核屏障：先按有效三角列长裁剪 `DataCopyPad` 将引用元素搬入预清零 UB 矩阵，并用 4 字节步长 `DataCopyPad` 单独提取实对角元（完全跳过虚对角槽位）；经 `GatherMask` 拆分为实虚平面后，顺列方向用折半树形 `Add` 归约列贡献，反射方向用 `repStride` 广播 `Mul` 配合 `WholeReduceSum` 一次求出全部列共轭内积，最后纯向量合成写回 `y`。

##### 2. 中尺寸无同步多核行条带核（`ChemvMedRowBlockKernel<8/16>`, $48 < n \le 512$）
每个 AIV 核独占输出向量 `y` 的连续 8 行或 16 行条带 `[r0, r0 + W)`，全程无需 `SyncAll` 或 Workspace：
- **顺列水平带**：位于存储三角内的水平矩形 `A[r0:r0+W, cLo:cHi]` 通过单条二维 `DataCopyPad`（原生支持奇偶任意字节 `srcStride`）整块搬入 UB，利用向量化倍增构造的广播索引表将 `tx` 展开 $W$ 倍后做连续向量乘与折半折叠；
- **反射垂直带**：位于存储三角内的垂直矩形 `A[kLo:kHi, r0:r0+W]` 同样以单条二维 `DataCopyPad` 搬入 UB，用 `x` 跨列广播乘法与 `WholeReduceSum` 求出 $W$ 列反射内积；
- **对角微块**：单独裁剪搬运唯一一个 $W \times W$ 对角微块的严格三角与 4B 实对角元，在 UB 内完成合并后单次写出 `y[r0:r0+W)`。

##### 3. 大尺寸混合全 FP32 Cube 协同核（`ChemvCubeEngine<128>` + `ChemvMixAivCore<128>`, $512 < n \le 4096$）
采用三阶段两屏障（`SyncAll<false>()`）流水架构：
- **Phase 1（AIV 错峰预处理）**：核 `0..F-1` 分片读取各自 128 个 `x` 元素并通过向量化 `Gather` 打包 `X_refl` / `TX_stored`；与此同时，错位分配的空闲 AIV 核并行将 $F$ 个对角块的严格指定三角（剔除对角元与未引用半角）及最多 $F-1$ 个末尾越界块清洗至 `offClean`，4B 实对角元紧凑存入 `offDiagReal`；
- **Phase 2（AIC 全流水双缓冲 Cube）**：每个 AIC 核处理连续块区间 `[tLo, tHi)`；内部满块直接通过 `Nd2Nz` 从原始 `A` 加载，对角/边缘块从 `offClean` 加载；开启 `A1/B1` 双缓冲、`A2/B2` 双缓冲（反射切片 $K=64$、顺列切片 $K=32$）与 `CO1` 双缓冲（4 个硬件 `FIX_M` 事件独立轮转），实现 `MTE2`、`MTE1`、`Cube MAC`、`Fixpipe` 四级硬件管道完全重叠；
- **Phase 3（AIV 线性归约收尾）**：利用 `Gather` 的线性分配律 $\operatorname{Gather}(\sum_k S_k) = \sum_k \operatorname{Gather}(S_k)$，每个输出块对应的 AIV 核先以 `MTE2-V` 双缓冲做满吞吐连续 `Add` 累加顺列/反射分块，循环外仅执行唯一 1 次 `Gather` 抽取实虚分量，叠加 $\beta y$ 与实对角项写回 `y`。

##### 4. 通用步长与精确守卫核（`ChemvGenericKernel`）
支持任意正/负 `incx`、`incy`、超大 $n > 4096$ 与极值/非有限数场景：按 `Physical(index, inc)` 精确计算负步长起点偏移，严格按 Netlib 顺序先算 $\beta y$（$\beta = 1$ 时原样拷贝保留单侧 `Inf`），再按 `uplo` 顺序组合实对角项、顺列 `A * (alpha * x)` 与反射 `alpha * sum(conj(A) * x)`，写出仅更新逻辑坐标对应的 8 字节复数槽位，确保步长间隙内存逐比特不变。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（Ascend 910B3，`arch22` / `dav-2201`） | √ |
| Atlas A3 系列（`arch22`） | √ |

## 算子约束限制

1. 仅支持单精度复数 `COMPLEX64`（`aclblasComplex`）与列主序矩阵存储，不支持广播或超出 `lda/incx/incy` 语义的不规则张量视图；
2. 矩阵阶数 `n >= 0`，前导维 `lda >= max(1, n)`；步长 `incx != 0`、`incy != 0`、不可为 `INT32_MIN`，且物理下标跨度 `(n - 1) * |inc| <= UINT32_MAX`；
3. 仅访问 `uplo` 指定三角及对角元素的实部；`y` 为原地更新 Device 缓冲区，异步执行绑定于 `handle` 关联的 `aclrtStream`，调用方在主机侧读取结果前须同步该 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 单标杆对比 Netlib BLAS 3.10.0 `cblas_chemv`（`CblasColMajor`，通过 `dlmopen` 独立命名空间隔离加载并校验符号来源）。输出 `y` 实部与虚部独立按 FLOAT32 混合容差判定：逐元素满足 `|actual - golden| <= 2^-13 + 2^-13 * |golden|` 的比例 `matched_ratio >= 0.99`，且 100% 有限值元素满足硬上限 `max_abs_error <= max(1e-2, 32 * ULP(golden))`；非有限值元素（`Inf/NaN`）按任务书等效规则全部匹配。用例集覆盖原始 1,200 条 CSV（含 200 条 `TC_PF`）+ 补充 110 条多尺度正态/均匀/边界 CSV + 7 项白盒专项 GTest（未引用三角与虚对角 `NaN` 毒化、步长间隙金丝雀、Host/Device 标量溢出抵消 witness、微小用户 workspace 回退等），共计 1,317 项 GTest | 任务书 §3.2 & [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 在 Atlas 800T A2（910B3）上对 `test_cases/gpu_baseline.csv` 全部 200 条 `COMPLEX64` 性能用例预热 5 次后连续有效采样 $\ge 10$ 次（实采 20 次）取算术平均，要求单算子平均耗时 $\le 1.25 \times \text{GPU baseline}$（即性能达到 GPU 基线的 $0.8\times$ 以上）。典型规模在 910B3 上的实测核耗时：$n=64$ 约 $5.7\,\mu\text{s}$（上限 $9.49\,\mu\text{s}$）、$n=512$ 约 $13.7\,\mu\text{s}$（上限 $18.30\,\mu\text{s}$）、$n=1024$ 约 $22.6\,\mu\text{s}$（上限 $26.39\,\mu\text{s}$）、$n=2048$ 约 $33.0\,\mu\text{s}$（上限 $37.61\,\mu\text{s}$）、$n=4096$ 约 $62.2\,\mu\text{s}$（上限 $81.62\,\mu\text{s}$） | 任务书 §3.3 & `test_cases/gpu_baseline.csv` |
| 内存标准 | 不涉及（任务书 §3.4 明确内存要求不涉及；算子内部 workspace 复用 `aclblasHandle_t` 默认池，不新增对外内存接口） | 任务书 §3.4 |

## 兼容性分析

- 本算子为 `ops-blas` 仓 `arch22` 架构新增算子：在 `include/cann_ops_blas.h` 中新增与 cuBLAS `cublasChemv` 参数序列一致的标准公开声明 `aclblasChemv`，供 Atlas A2/A3 及其他产品线统一共用，未引入任何私有平行接口；
- 实现代码完全隔离在 `blas/symv/arch22/` 新增源文件中，不影响同目录既有 `aclblasSsymv` 及 `arch35` 分支已有算子的编译、符号导出与运行行为；
- `test/symv/chemv/CMakeLists.txt` 仅在 `arch22` 分支补齐 `ops_blas_add_gtest_tests` 注册与 Netlib 3.10.0 引用校验，保持 `arch35` 原有构建逻辑不变。
