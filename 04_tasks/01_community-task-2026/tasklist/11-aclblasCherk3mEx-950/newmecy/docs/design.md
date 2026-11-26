# 需求背景（required）

## 需求来源

本算子来源于 CANN 社区任务 2026（ops-blas 开源仓社区任务，任务编号：11-aclblasCherk3mEx-950，代码仓：https://gitcode.com/cann/ops-blas ）。要求在昇腾 NPU（Ascend 950PR，`arch35` / `DAV_3510` 架构）上，使用 Ascend C 编程语言并基于 ops-blas 开源仓工程框架，开发基于 3M 算法的单精度复数 Hermitian 矩阵秩-k 更新扩展算子 `aclblasCherk3mEx`，对标 cuBLAS `cublasCherk3mEx` 接口与 Netlib BLAS `cblas_cherk` 数学语义，完成算子设计、实现与自验证，验收通过后合入昇腾算子开源仓。

## 背景介绍

### aclblasCherk3mEx 算子实现现状分析

ops-blas 开源仓中 `blas/herk/arch35/` 已有非 Ex 版本的 `cherk_host.cpp`，其采用“1 次 Deinterleave AIV + 4 次 GEMM AIC + 1 次 Combine AIV = 6 次 Kernel Launch”的 4M 拼接方案，并需要高达 $8nk + 16n^2$ 字节（`4096×4096` 时需 384 MiB）的全局 Workspace。该模式既不满足 `aclblasCherk3mEx` 对 3M 算法（三次实数乘必须在单次 Kernel Launch 内完成）的要求，也不支持 `Atype = ACL_INT8` 扩展输入，且在 HBM 带宽上存在显著冗余。本任务在 `blas/herk/arch35/` 下独立新增 `cherk3m_ex_*` 源文件，并在 `include/cann_ops_blas.h` 新增公开声明。

当前对标接口支持的能力与参数契约如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 库上下文句柄，携带 stream | aclblasHandle_t | aclblasHandle_t | 非空有效句柄 | 标量 |
| uplo | 指定读写 C 的上三角或下三角 | aclblasFillMode_t | 枚举 | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` | 标量（属性） |
| trans | op(A) 的运算类型（N 不转置 / C 共轭转置） | aclblasOperation_t | 枚举 | `{ACLBLAS_OP_N, ACLBLAS_OP_C}`；`ACLBLAS_OP_T` 非法 | 标量（属性） |
| n | 输出 Hermitian 矩阵 C 的阶数 | int | int | $n \ge 0$；$n = 0$ 为合法 no-op | 标量（属性） |
| k | 秩-k 更新归约维 | int | int | $k \ge 0$；$k = 0$ 时跳过 A，若 $\beta \ne 1$ 执行 $C = \beta C$ | 标量（属性） |
| alpha | 实数标量乘数（Host 或 Device 指针） | const float* | FLOAT32 | 非空；与 beta 同为 Host 或同为 Device，禁止混用 | 标量 |
| A | 输入复数矩阵（Device，列主序） | const void* | `ACL_COMPLEX64` / `ACL_INT8` | $n>0 \land k>0 \land \alpha \ne 0$ 时非空 | `OP_N`: $n \times k$；`OP_C`: $k \times n$ |
| Atype | 输入矩阵 A 的数据类型枚举 | aclDataType | `ACL_COMPLEX64` / `ACL_INT8` | `ACL_INT8` 为 `(int8 re, int8 im)` 交错 2 字节 | 标量（属性） |
| lda | A 的前导维度（单位：复元素） | int | int | `OP_N`: $\text{lda} \ge \max(1, n)$；`OP_C`: $\text{lda} \ge \max(1, k)$ | 标量（属性） |
| beta | 实数标量乘数（Host 或 Device 指针） | const float* | FLOAT32 | 非空；$\beta = 0$ 时不读旧 C | 标量 |
| C | 输出 Hermitian 矩阵（Device，列主序，原地更新） | aclblasComplex* | `ACL_COMPLEX64` | $n > 0$ 时非空；仅 `uplo` 指定三角被读写 | $n \times n$（物理 $\text{ldc} \times n$） |
| Ctype | 输出矩阵 C 的数据类型枚举 | aclDataType | `ACL_COMPLEX64` | 仅支持 `ACL_COMPLEX64` | 标量（属性） |
| ldc | C 的前导维度（单位：复元素） | int | int | $\text{ldc} \ge \max(1, n)$ | 标量（属性） |

计算公式：

$$C \leftarrow \alpha \cdot \operatorname{op}(A) \cdot \operatorname{op}(A)^H + \beta \cdot C, \qquad \alpha, \beta \in \mathbb{R}$$

### aclblasCherk3mEx 算子功能分析

- **算子功能**：使用 3M 复数乘法分解算法，对 $n \times n$ 复数 Hermitian 矩阵 $C$ 的 `uplo` 指定三角执行原地秩-k 更新，并将对角元素 $C_{i,i}$ 的虚部强制清零。
- **输入**：`handle`、`uplo`、`trans`、`n`、`k`、`alpha`、`A`、`Atype`、`lda`、`beta`、`Ctype`、`ldc`。
- **输出**：`C`（原地覆写，仅更新 `uplo` 指定的上三角或下三角区域；非 `uplo` 三角及 $\text{ldc} > n$ 的 padding 区域保持逐比特零改写）。
- **支持数据类型**：
  - 支持 `(Atype = ACL_COMPLEX64, Ctype = ACL_COMPLEX64)` 与 `(Atype = ACL_INT8, Ctype = ACL_COMPLEX64)` 两种组合；
  - `alpha`、`beta` 为实数 `FLOAT32` 标量（支持 Host/Device 双模式，禁止混用）。
- **支持广播**：不支持广播（标准 BLAS Level-3 矩阵运算语义）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Ascend 950PR（`arch35` / `DAV_3510`）上实现 `aclblasCherk3mEx` 算子：

1. **单次 Kernel Launch 3M 硬约束**：禁止采用多 GEMM 拼接方案，Hermitian 3M 的三次实数矩阵乘必须在同一次 Kernel Launch 内于片上融合完成，且全局 Workspace 为 0 字节。
2. **双数据类型支持**：支持 `ACL_COMPLEX64` 与交错 2 字节 `ACL_INT8` 输入，输出统一为 `ACL_COMPLEX64`；`ACL_INT8` 在片上无损提升为 FP32 后进入统一 3M 流水，避免 `int8` 饱和截断导致的精度损失。
3. **严格的三角保护与对角实数化**：仅读写 `uplo` 指定的半三角区域，保护非 `uplo` 三角与 `ldc` padding 区域哨兵零污染；对角元素 $C_{i,i}$ 虚部强制清零（含 $\beta \ne 0$ 及退化缩放路径）。
4. **完备的参数校验与边界处理**：严格按顺序校验句柄、枚举（拦截 `ACLBLAS_OP_T`）、维度、数据类型组合、前导维、Host/Device 标量一致性及矩阵指针；支持 $n=0$、$k=0$、$\alpha=0$、$\beta=0$（不读旧 C）及 $\beta=1$ 的快速返回与退化处理。
5. **精度与性能双百达标**：精度满足生态算子开源精度标准（Mixed Tolerance `rtol = atol = 2^-13`，匹配率 $\ge 99\%$，最大绝对误差 $\le \max(10^{-2}, 32 \times \text{ULP})$）；性能在 Ascend 950PR 上通过 `msprof op` 采集 kernel Task Duration，满足单 case $\ge 0.4000\times$ 且全量 200 例均值 $\ge 0.5000\times$ GPU 基线。

## 需求拆解

1. **公开接口与 Host 校验状态机**：在 `include/cann_ops_blas.h` 新增 13 参数声明；在 Host 侧实现全量参数合法性校验、Host/Device 标量自动探测与 4 路径路由决策。
2. **Hermitian 3M 代数分解与列主序零转置映射**：构造 $P_0 = X_r X_r^T$、$P_1 = X_i X_i^T$、$P_2 = (X_r + X_i)(X_r - X_i)^T$，结合列主序到行主序的转置等价映射，消除额外转置与符号翻转开销。
3. **全尺度 4 路径快慢分流**：按规模路由至 Path 0（Quick Return）、Path 1（Scale/Zero-only AIV）、Path 2（AIV-only SIMT Direct 3M）与 Path 3（`KERNEL_TYPE_MIX_AIC_1_2` On-Chip 3M，含 `Mix64` 与 `Mix128` 双规格）。
4. **三角 Tile 裁剪与 2D Swizzle 调度**：仅计算有效三角块（节省约 50% 算力），结合对角 Tile 优先调度与 L2 感知 2D 映射提升缓存命中率。
5. **3M 数值安全防线**：强制禁用 HF32、大 K 分段归约（`reductionSpan`）抑制大数相消误差，并在 AIV Consumer 集成零扫描按位非有限（`Inf`/`NaN`）检测与 `__simt_vf__` 精确分量修复。

# 详细设计（required）

## 算子分析

### 数学公式

令 $X = \operatorname{op}(A) = X_r + i X_i \in \mathbb{C}^{n \times k}$，其中：
- `trans = ACLBLAS_OP_N`：$X = A$，即 $X_r = A_r,\ X_i = A_i$；
- `trans = ACLBLAS_OP_C`：$X = A^H$，即 $X_r = A_r^T,\ X_i = -A_i^T$（拆解虚部时乘以 `-1.0f`）。

定义辅助实数矩阵 $X_+ = X_r + X_i$ 与 $X_- = X_r - X_i$，构造三次实数矩阵乘：

$$P_0 = X_r X_r^T, \qquad P_1 = X_i X_i^T, \qquad P_2 = X_+ X_-^T = (X_r + X_i)(X_r - X_i)^T$$

由此得到 Hermitian 3M 合并公式：

$$\operatorname{Re}(X X^H) = P_0 + P_1, \qquad \operatorname{Im}(X X^H) = P_2 - P_0 + P_1 = X_i X_r^T - X_r X_i^T$$

最终输出更新为（对角元素 $i = j$ 处强制虚部置 `0.0f`）：

$$C_{\text{re}} \leftarrow \alpha (P_0 + P_1) + \beta C_{\text{re}}, \qquad C_{\text{im}} \leftarrow \alpha (P_2 - P_0 + P_1) + \beta C_{\text{im}}$$

**列主序到行主序零转置等价映射**：输出 `C` 为列主序存储（$C(i, j)$ 位于 `C[j * ldc + i]`），物理上等价于以列号 $j$ 为行、行号 $i$ 为列的行主序矩阵 $C^T[j, i]$。利用 $(P_0)^T = X_r X_r^T$、$(P_1)^T = X_i X_i^T$、$(P_2)^T = X_- X_+^T$，在行主序 Cube `Mmad(L0C, L0A, L0B)` 中：
- 左操作数（M 维 = 输出列号 $j$，SubBlock 0）组装三通道 $(X_r, X_i, X_-)$；
- 右操作数（N 维 = 输出行号 $i$，SubBlock 1）组装三通道 $(X_r, X_i, X_+)$；
- Cube 经 `Fixpipe` 输出至 UB 时坐标 `(mOff, nOff)` 直接对应 `C` 的列偏移 `jBase = mOff` 与行偏移 `iBase = nOff`，实现零额外转置。

### 支持数据类型

- **输入矩阵 `A`**：`ACL_COMPLEX64`（交错 8 字节：`float re, float im`）或 `ACL_INT8`（交错 2 字节：`int8_t re, int8_t im`）。
- **输出矩阵 `C`**：`ACL_COMPLEX64`（交错 8 字节：`float re, float im`）。
- **标量 `alpha` / `beta`**：实数 `FLOAT32`（`float`）。
- **片上计算域**：统一采用 IEEE FP32 计算，AIC 初始化强制关闭 HF32（`SetHF32Mode(DISABLE)`）。`ACL_INT8` 经 `int8 -> half -> float` 片上无损提升后进入统一 3M 流水，避免 $X_\pm \in [-256, 255]$ 在 `int8` 下溢出。

### 支持形状

- $n \ge 0, k \ge 0$，覆盖任意小维度、非对齐素数维度（`1, 3, 5, 7, 31, 33, 127, 1023, 4095`）及大矩阵（最高 `4096×4096`）；
- 前导维满足 $\text{lda} \ge \max(1, n)$（`OP_N`）或 $\text{lda} \ge \max(1, k)$（`OP_C`），$\text{ldc} \ge \max(1, n)$；
- 支持 $n = 0$、$k = 0$ 等合法退化边界。

## 算子实现

### 实现方案

整体采用 **4 路径快慢分流 + 0 全局 Workspace** 架构，所有计算路径均在单次 Kernel Launch 内完成：

1. **Path 0（Quick Return）**：$n = 0$ 或 $((k = 0 \lor \alpha = 0) \land \beta = 1)$，Host 直接返回 `ACLBLAS_STATUS_SUCCESS`。
2. **Path 1（Scale/Zero-only AIV）**：$n > 0 \land (k = 0 \lor \alpha = 0) \land \beta \ne 1$，启动 AIV-only 内核 `Cherk3mExScaleSimt`，仅更新 `uplo` 三角 $C = \beta C$ 并置对角虚部为 `0.0f`（$\beta = 0$ 不读旧 `C`）。
3. **Path 2（Direct 3M SIMT AIV）**：$n \le 64$ 的极小/延迟敏感矩阵，启动 AIV-only SIMT 内核 `Cherk3mExDirectSimt`，避免 AIC/AIV 跨核握手开销，支持 Warp Shuffle K 归约。
4. **Path 3（`KERNEL_TYPE_MIX_AIC_1_2` On-Chip 3M）**：中大矩阵采用 1 AIC : 2 AIV 耦合架构，包含双分支：
   - **Path 3A（`Mix64`）**：`TILE = 64, PANEL_K = 64, INNER_K = 16`，通用中大矩阵主路径；
   - **Path 3B（`Mix128`）**：`TILE = 128, PANEL_K = 32`，当 `!isInt8 && n >= 256 && k <= 32` 时启用，针对超薄 K 场景大幅压低调度与写回开销。

#### 3.2.1 host侧设计：

##### tiling策略：
- **Path 1 (`Cherk3mExScaleTilingData`)**：传递 `n, ldc, betaVal, isUpper, isBetaZero`。
- **Path 2 (`Cherk3mExDirectTilingData`)**：传递 `n, k, lda, ldc, alphaVal, betaVal, isUpper, isConjTrans, isInt8, isBetaZero`。
- **Path 3 (`Cherk3mExMixTilingData`)**：根据是否满足 `!isInt8 && n >= 256 && k <= 32` 确定 `useTile128`（选 `TILE = 128` 或 `64`），计算二维网格 `gridDim = CeilDiv(n, tileSize)` 与有效三角块数 `totalTriTiles = gridDim * (gridDim + 1) / 2`；在 `Mix64` 且 `k > 1024` 时设置 `reductionSpan = 512` 启用分段 K 归约。

##### 任务均分：
- **三角列镜像配对（Path 1）**：将列 `col` 与列 `n - 1 - col` 配对分配给同一 AIV 核，使每核每轮处理的总元素数恒为 $n + 1$，消除三角列长不均导致的尾核长尾。
- **三角 Tile 裁剪与均衡（Path 3）**：仅对 `uplo` 对应的 $T_{\text{tri}} = G(G+1)/2$ 个有效三角 Tile 编号（`UPPER` 取 $m_{\text{tile}} \ge n_{\text{tile}}$，`LOWER` 取 $m_{\text{tile}} \le n_{\text{tile}}$），各物理核组按持久化步进循环领取 Tile；`Mix128` 采用对角 Tile 优先调度（`DecodeMixTile128`）与 2D 局部性映射提升 L2 命中率。

##### 批量搬运：
- GM 到 UB 的加载统一采用 `DataCopyPad`，按 `ACL_COMPLEX64`（8 字节/元素）或 `ACL_INT8`（2 字节/元素）精确计算 `blockLen` 并自动零填充非对齐尾块；
- UB 写回 GM 时，非对角 Tile 由两个 AIV SubBlock 各自通过一次 2D `DataCopyPad` 精准写出有效矩形区；对角 Tile 逐列裁剪出仅属于 `uplo` 的连续片段写出，绝对不触碰非 `uplo` 半区及 `ldc` padding。

##### 1. 分核策略：
- **Path 1**：启动 `min(aivCoreNum, n)` 个 AIV 核。
- **Path 2**：启动 `min(aivCoreNum, CeilDiv(n * (n + 1) / 2, 16))` 个 AIV 核，每核 512 线程（16 Warps）。
- **Path 3**：按满核优先且不浪费核的原则，设置 `usedCores = min(aicCoreNum, totalTriTiles)`，以 `KERNEL_TYPE_MIX_AIC_1_2` 启动 `usedCores` 个核组（每组 1 AIC + 2 AIV；AIV 通过 `groupIdx = GetBlockIdx() / 2` 与 `subIdx = GetSubBlockIdx()` 绑定所属 AIC）。

##### 2. 数据分块和内存优化策略：
Ascend 950PR 单核组片上硬件容量为 L0C 256 KiB、L0A/L0B 各 64 KiB、L1 512 KiB、单 AIV UB 248 KiB（计算工作区上限 192~224 KiB，避开 `224..256 KiB` SIMT DCache 区）。以主路径 `Mix64`（`TILE = 64, PANEL_K = 64, INNER_K = 16`）为例：
- **L0C**：2-Stage Ping-Pong × 3 路 `64×64` FP32 累加器 = **96 KiB**（占比 37.5%）。
- **L0A / L0B**：各 2-Slot Ping-Pong × `64×16` FP32 = **8 KiB / 8 KiB**（占比 12.5%）。
- **L1**：2-Stage Ping-Pong ×（左 3 路 + 右 3 路）`64×64` FP32 = **192 KiB**（占比 37.5%）。
- **UB（每 AIV）**：Fixpipe 3 路接收区（`24 KiB`）+ 分段累加/旧 C/Interleave 区（$\le 32\text{ KiB}$）+ Producer 输入变换区（`80 KiB`），单 Stage 全驻留峰值 **136 KiB**（双 Stage 输出缓冲模式下 **160 KiB**），严格处于安全上限内。
- **全局 Workspace**：全路径片上直连，**0 字节**全局 Workspace，零运行期 `aclrtMalloc` 开销。

##### 3. tilingkey规划策略：
通过 Host 侧静态路径路由直接分发对应 Kernel 入口，并在 `Cherk3mExMixTilingData` 中通过紧凑字段控制核内分支（避免过多编译期模板膨胀）：
- 入口路由：Path 1 → `cherk3m_ex_scale_kernel_do`；Path 2 → `cherk3m_ex_direct_kernel_do`；Path 3 → `cherk3m_ex_mix_kernel_do`；
- Path 3 核内路由字段：`useTile128`（切换 `Mix64` / `Mix128`）、`isInt8`、`isConjTrans`、`isUpper`、`isBetaZero`、`hasBeta`、`isUnitAlphaBeta`、`reductionSpan`。

##### 数据检测：
Host 侧严格按以下顺序执行防御式参数校验（不在 Host 对 Device 张量做元素级扫描）：
1. `handle == nullptr` → 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `uplo \notin \{UPPER, LOWER\}` 或 `trans \notin \{OP\_N, OP\_C\}`（含 `OP_T`）→ 返回 `ACLBLAS_STATUS_INVALID_ENUM`；
3. `n < 0` 或 `k < 0` 或 `lda < max(1, trans == OP_N ? n : k)` 或 `ldc < max(1, n)` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. `(Atype, Ctype)` 不属于 `{(ACL_COMPLEX64, ACL_COMPLEX64), (ACL_INT8, ACL_COMPLEX64)}` → 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；
5. `alpha == nullptr` 或 `beta == nullptr` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. `n == 0` → 执行 Path 0 Quick Return，返回 `ACLBLAS_STATUS_SUCCESS`；
7. 探测 `alpha`/`beta` 指针位置（Host/Device），若混用或读取失败 → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
8. `C == nullptr` 或 `(k > 0 && alpha != 0 && A == nullptr)` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`。

#### 3.2.2 kernel侧设计：

内核执行分为 **Init** 与 **Process** 两个阶段：

1. **Init 阶段**：
   - AIC 侧首行执行 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)` 禁用 HF32，初始化 L1/L0A/L0B/L0C 缓冲区与双缓冲同步事件（`INPUT_FREE[0..1]`、`M_MTE1[0..1]`、`FIX_M[0..1]`）；
   - AIV 侧初始化 `TPipe`、UB 缓冲（Producer 变换区与 Consumer 合并区）及跨核空闲信号（`FREE_FLAG[0..1]`、`MTE3_MTE2`）。
2. **Process 阶段（以 Path 3 `KERNEL_TYPE_MIX_AIC_1_2` 三阶段双缓冲耦合流水为例）**：
   - **Producer（AIV SubBlock 0 / 1 → L1）**：
     - SubBlock 0 读取左面板 $A(j)$，SubBlock 1 读取右面板 $A(i)$；若 `isInt8 == 1`，在 UB 执行 `int8 -> half -> float` 无损 `Cast`；若 `isConjTrans == 1`，在变换阶段同步完成转置与虚部取反；
     - 分别在 UB 生成左三通道 $(X_r, X_i, X_-)$ 与右三通道 $(X_r, X_i, X_+)$，转换为 `NZ`/`ZN` 布局写入 `L1[stage]`，通过 `CrossCoreSetFlag(INPUT_READY + stage)` 通知 AIC。
   - **Compute（AIC L1 → L0A/L0B → L0C → Fixpipe UB）**：
     - AIC 等待 `INPUT_READY` 后，通过 `L0A`/`L0B` 2-Slot 乒乓依次执行 3 路 FP32 `Mmad`（$P_0 = X_r X_r^T$、$P_1 = X_i X_i^T$、$P_2 = X_- X_+^T$），并立即通过 `INPUT_FREE` 释放 L1 槽位；
     - 完成当前 Tile（或单段 `reductionSpan`）K 累加后，等待 `FREE_FLAG`，经 `Fixpipe(L0C -> UB, dualDstCtl=1)` 将 $P_0, P_1, P_2$ 上/下半区分别直写至 SubBlock 0 与 SubBlock 1 的 UB，并置位 `READY_FLAG`。
   - **Consumer（AIV SubBlock 0 / 1 UB → GM）**：
     - 接收 $P_0, P_1, P_2$ 后计算 $\text{Re} = P_0 + P_1$、$\text{Im} = P_2 - P_0 + P_1$（若启用 `reductionSpan` 则先累加至 UB 驻留 `(sumRe, sumIm)`）；
     - 执行 $\alpha$ 缩放与 `hasBeta` 时的旧 $C$ 叠加（`isBetaZero == 1` 时跳过旧 $C$ 加载），对角元素强制 `Im = 0.0f`；
     - 顺带对有效输出行按位与 `0x7F800000` 执行非有限检测，若触发 `Inf`/`NaN` 则在片内调用 `__simt_vf__` 精确分量修复（`#pragma clang fp contract(off)` + `volatile float`），确保与 `cblas_cherk` 一致；
     - 通过 `DataCopyPad` 严格按 `uplo` 边界写回 GM，随后置位 `FREE_FLAG`。
   - **DrainPipeline**：退出主循环前，AIC 与 AIV 严格 1:1 配对排空所有残余 `HardEvent` 与 `CrossCore` 信号，确保任意奇偶 Tile 数下零死锁。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（`arch35` / `DAV_3510`） | √ |

## 算子约束限制

1. **数据类型约束**：仅支持 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)` 两种 `(Atype, Ctype)` 组合；不支持双精度复数或其他类型。
2. **转置枚举约束**：`trans` 仅支持 `ACLBLAS_OP_N` 与 `ACLBLAS_OP_C`，传入 `ACLBLAS_OP_T` 返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. **布局与重叠约束**：矩阵仅支持标准列主序带前导维布局；调用方需保证 `A` 与 `C` 内存区域不重叠；不支持广播。
4. **标量指针约束**：`alpha` 与 `beta` 支持同为 Host 指针或同为 Device 指针，禁止 Host 与 Device 指针混用。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | CPU 标杆统一为非 3M 的 Netlib `cblas_cherk`（`ACL_INT8` 先无损提升为 `float` 后调用）；按 `uplo` 三角实部、虚部分别比对，满足 Mixed Tolerance：`rtol = atol = 2^-13 ≈ 1.2207e-4`、`matched_ratio >= 0.99`、`max_abs_error <= max(1e-2, 32 * ULP)`（同时统计 MERE/MARE）；非 `uplo` 三角与 `ldc` padding 哨兵零改写，对角虚部清零；全量 1000 条精度用例 100% PASS。 | 生态算子开源精度标准（mixed_tolerance_standard）、任务书 §3.2 |
| 性能标准 | 在 Ascend 950PR 上对全量 200 条 `TC_PF` 性能用例使用 `msprof op` 采集底层 kernel `Task Duration(us)`（1 precheck + 5 warmup + 12 samples，去毛刺后取 10 稳态样本均值），与 `gpu_baseline.csv` 的 `gpu_ms` 做 kernel-to-kernel 对比；达标红线为单 case 倍率 $\ge 0.4000\times$ 且全量 200 例算术平均倍率 $\ge 0.5000\times$（内部预留 5% 安全余量目标：单 case $\ge 0.4200\times$，均值 $\ge 0.5500\times$）。 | 任务书 §3.3、`gpu_baseline.csv` |

## 兼容性分析

1. **接口与头文件兼容**：在 `include/cann_ops_blas.h` 新增标准 13 参数 `aclblasCherk3mEx` 声明，状态码与数据类型枚举完全复用 `include/cann_ops_blas_common.h`，不影响已有算子 ABI。
2. **构建与版本兼容**：源码独立落位于 `blas/herk/arch35/cherk3m_ex_*`，自动匹配 `blas/CMakeLists.txt` 单算子筛选规则，无需改动全局构建配置；面向 CANN 9.1.0+（`#if ASC_DEVKIT_GE_9_1` 保护 `tensor_api` / `FixpipeParamsC310` 路径），与现有 `arch35` 其他算子完全解耦共存。
