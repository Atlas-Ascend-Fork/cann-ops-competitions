# 需求背景（required）

## 需求来源

本设计文档对应 **CANN 社区任务 2026（11 月社区任务）— `aclblasCsyrk3mEx` 算子开发（Ascend 950PR）**，面向昇腾 `ops-blas` 开源算子库（`https://gitcode.com/cann/ops-blas`），基于 Ascend C 在 Ascend 950PR（`arch35` / `DAV_3510`）上新增实现支持混合输入精度与对称 Gauss 3M 算法的高性能复数对称秩-k更新接口 `aclblasCsyrk3mEx`。

## 背景介绍

### `aclblasCsyrk3mEx` 算子定位与对标接口

`aclblasCsyrk3mEx` 对标 cuBLAS 的 `cublasCsyrk3mEx`（13 参数，列主序，句柄式接口），属于 BLAS Level-3 复数对称秩-k更新（Symmetric Rank-K Update）的扩展版本：

- **算法特性**：采用 **Gauss 3M** 复数乘法分解，将对称外积 $X X^T$ 所需的 4 次实数矩阵乘缩减为 3 次实数矩阵乘，理论降低 25% Cube 计算量；且三次实数乘须在**单次融合 kernel launch 内完成**，不得拆分为三次独立 GEMM 调用；
- **多输入精度支持**：在标准 `ACL_COMPLEX64` 之外支持 `Atype = ACL_INT8`（交错 `(int8 real, int8 imag)` 存储，对应 `CUDA_C_8I`），在计算前无损提升为 FP32 后输出 `ACL_COMPLEX64`；
- **仓内现状分析**：`ops-blas` 仓内此前无 `Ex` 后缀的 `csyrk3m_ex` 接口与实现（仅有非 Ex 的 `aclblasSsyrk` / `aclblasCsyrk` 及 `cgemm3m`）；本算子独立实现于 `blas/syrk/arch35/`，不依赖 `aclblasCsyrkEx`，并在 `include/cann_ops_blas.h` 与 `cmake/asc_devkit_version.cmake` 完成注册。

### `aclblasCsyrk3mEx` 参数与接口规格

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n,
    int k,
    const aclblasComplex* alpha,
    const void* A,
    aclDataType Atype,
    int lda,
    const aclblasComplex* beta,
    aclblasComplex* C,
    aclDataType Ctype,
    int ldc);
```

| 参数 | 参数含义 | 内存位置 | 数据类型 / 支持枚举 | 约束 | 形状（列主序） |
| --- | --- | --- | --- | --- | --- |
| `handle` | ops-blas 句柄（携带 stream 与默认 workspace） | Host | `aclblasHandle_t` | 非 `nullptr` | 标量句柄 |
| `uplo` | 输出矩阵更新三角区域 | Host | `ACLBLAS_UPPER` / `ACLBLAS_LOWER` | 仅支持上述枚举 | 枚举 |
| `trans` | 输入矩阵 `A` 转置模式 | Host | `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` | 对称语义下 `OP_C` 等价于 `OP_T`（不共轭） | 枚举 |
| `n` | 对称矩阵 `C` 阶数 | Host | `int` | `n >= 0`（`n == 0` 为合法 no-op） | 标量 |
| `k` | 归约维度 | Host | `int` | `k >= 0` | 标量 |
| `alpha` | 复数标量乘数 | Host 或 Device（自动探测） | `const aclblasComplex*`（`ACL_COMPLEX64`） | 非 `nullptr` | 标量 |
| `A` | 输入矩阵 | Device | `const void*`（按 `Atype` 解释） | `n > 0 && k > 0 && alpha != (0,0)` 时非 `nullptr` | `OP_N`: $(n \times k)$；`OP_T/C`: $(k \times n)$ |
| `Atype` | 输入矩阵 `A` 数据类型 | Host | `ACL_COMPLEX64` / `ACL_INT8` | 其余合法类型返回 `NOT_SUPPORTED` | 枚举 |
| `lda` | 矩阵 `A` 主维 | Host | `int` | `OP_N`: `lda >= max(1, n)`；`OP_T/C`: `lda >= max(1, k)` | 标量 |
| `beta` | 复数标量乘数 | Host 或 Device（自动探测） | `const aclblasComplex*`（`ACL_COMPLEX64`） | 非 `nullptr` | 标量 |
| `C` | 对称输入/输出矩阵 | Device | `aclblasComplex*`（`ACL_COMPLEX64`） | `n > 0` 时非 `nullptr`；仅更新 `uplo` 三角 | $(n \times n)$ |
| `Ctype` | 矩阵 `C` 数据类型 | Host | `ACL_COMPLEX64` | 其余合法类型返回 `NOT_SUPPORTED` | 枚举 |
| `ldc` | 矩阵 `C` 主维 | Host | `int` | `ldc >= max(1, n)` | 标量 |

### 算子功能分析

- **核心计算**：$C = \alpha \cdot \operatorname{op}(A)\operatorname{op}(A)^T + \beta \cdot C$；
- **对称语义（非 Hermitian）**：$C_{ij} = C_{ji}$，**对角元素虚部为有效数据，计算后原样保留，不置 0**；`ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 完全等价（不做共轭）；
- **写保护与退化语义**：
  - 仅读写 `uplo` 指定三角（含对角线），另一三角与 `ldc > n` 的 padding 行严格不改写；
  - `k == 0 || alpha == (0,0)` 时不引用 `A`：`beta == (1,0)` 直接 quick return，`beta != (1,0)` 仅执行 `C = beta * C`；
  - `beta == (0,0)` 时不读取 `C` 旧值（`C` 旧值中的 NaN/Inf 不进入结果）。

---

# 需求分析（required）

## 需求描述

在 Ascend 950PR（`arch35` / `DAV_3510`）上使用 Ascend C 实现 `aclblasCsyrk3mEx` 算子，完整支持 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)` 两种数据类型组合、`uplo × trans` 全 6 种枚举组合、Host/Device 双标量指针模式、前导维 padding、极值/非有限值（Inf/NaN）语义及全量负向参数校验；同时满足单次 kernel launch 内完成对称 3M 三次实数矩阵乘、精度通过全量 1000 例混合容差校验、性能达到 `msprof op` kernel-to-kernel 口径下单用例 $\ge 0.4\times$ GPU 且 200 例平均 $\ge 0.5\times$ GPU 的验收要求。

## 需求拆解

1. **接口与工程集成**：在 `include/cann_ops_blas.h` 声明 `aclblasCsyrk3mEx`，在 `cmake/asc_devkit_version.cmake` 注册 `csyrk3m_ex`，在 `blas/syrk/arch35/` 实现 Host/Kernel 代码并更新 `blas/syrk/README.md`，支持 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 干净构建且不破坏既有 `csyrk`；
2. **对称 3M 单次融合 Kernel 架构**：在单次 MIX（AIC:AIV = 1:2）kernel launch 中融合完成 $P_0, P_1, P_2$ 三次实数矩阵乘与 3M 组合、$\alpha/\beta$ Epilogue 及三角精确写回；
3. **Ascend 950PR 高性能计算与访存优化**：充分发挥 DAV_3510 `HF32` Cube 8× 吞吐优势（`COMPLEX64` 采用 `HF32×3` 补偿、`INT8` 采用无损 1× `HF32`），通过 $O(NK)$ 预打包 + 256B 对齐防 bank 冲突步长 + `BS=128, PANEL_K=128, INNER_K=32` 逐乘积 L1 双 bank 乒乓流水线打满 Cube 与 MTE2；
4. **数值精度与边界鲁棒性**：覆盖大 $k$ 分段 3M 抵消、大 $k$ 对角线精确 Kahan 3M 细化、非有限值自动检测与 Netlib 语义重算、小规模极端 $|\alpha|$ 补偿直算以及非 `uplo`/padding 写保护；
5. **完备自验体系**：在 `test/syrk/csyrk3m_ex/` 构建 GTest 测试工程，完成 1000 例精度（含 `TC_3M_*` 与负向用例）、`NullHandle`/`HostAndDeviceScalarPointerModes`、200 例正式 `msprof op` 性能及 5 例 `ACL_INT8` 性能补充自验证。

---

# 详细设计（required）

## 算子分析

### 数学公式

令 $X = \operatorname{op}(A) = X_r + i X_i \in \mathbb{C}^{n \times k}$，其中 `Atype = ACL_INT8` 时每个交错对 $(a_r, a_i) \in [-128, 127]^2$ 先精确转换为 FP32。对称秩-k乘积按对称 Gauss 3M 展开为：

$$
\begin{aligned}
P_0 &= X_r X_r^T, \qquad P_1 = X_i X_i^T, \qquad P_2 = (X_r + X_i)(X_r + X_i)^T \\
R &= P_0 - P_1, \qquad I = (P_2 - P_0) - P_1 \\
C_{\text{new}} &= (\alpha_r + i\alpha_i)(R + i I) + (\beta_r + i\beta_i) C_{\text{old}}
\end{aligned}
$$

对所有满足上三角（$i \le j$）或下三角（$i \ge j$）的元素更新 $C(i, j)$，对角线 $C(i, i)$ 的虚部严格保留。

### 支持数据类型

| 组合 | `Atype` | `Ctype` | `alpha` / `beta` | 计算模式 |
| --- | --- | --- | --- | --- |
| 1 | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `HF32×3`（主路径）/ Native `FP32`（小规模或极端 $|\alpha|$） |
| 2 | `ACL_INT8`（交错 2B） | `ACL_COMPLEX64` | `ACL_COMPLEX64` | 精确无损 `1× HF32`（每 256 步分段归约） |

### 支持形状

- 任意 $n \ge 0, k \ge 0$（包括 $n=0$、$k=0$、非对齐奇数/质数尺寸、$k \gg n$ 宽矩阵、$n \gg k$ 窄矩阵、以及 $n, k$ 至 4096 及以上的大矩阵）；
- 任意满足 `lda >= max(1, op_rows)`、`ldc >= max(1, n)` 的紧凑或带 padding 前导维布局。

---

## 算子实现

### 实现方案

代码结构精简为 4 个文件（位于 `blas/syrk/arch35/`）：
- `csyrk3m_ex_tiling_data.h`：DAV_3510 硬件常量与 5 个 Tiling 结构体定义；
- `csyrk3m_ex_kernel.h`：Kernel launch 封装函数声明；
- `csyrk3m_ex_host.cpp`：参数校验、标量指针探测、四路径自适应 Tiling 与分发；
- `csyrk3m_ex_kernel.cpp`：AIV Scale、Direct SIMT 3M、On-chip MIX 3M、AIV Prep 预打包与 Single-Launch Fused MIX 3M 全部核函数实现。

#### 3.2.1 Host 侧设计（`csyrk3m_ex_host.cpp`）

##### 1. 参数校验与数据检测

按规范顺序执行结构性校验：
1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `uplo` / `trans` 非法 → `ACLBLAS_STATUS_INVALID_ENUM`；
3. `n < 0 || k < 0 || lda < minLda || ldc < max(1, n) || alpha == nullptr || beta == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
4. `(Atype != ACL_COMPLEX64 && Atype != ACL_INT8) || Ctype != ACL_COMPLEX64` → `ACLBLAS_STATUS_NOT_SUPPORTED`；
5. `n == 0` 直接返回 `ACLBLAS_STATUS_SUCCESS`；`C == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. 通过 `CheckPtrLocation` 自动探测 `alpha`/`beta` 指针位置（Host 指针零开销直读；Device 指针走异步 D2H + `aclrtSynchronizeStream`）；
7. `k > 0 && !isAlphaZero && A == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；`(k == 0 || isAlphaZero) && isBetaOne` 直接 quick return。

##### 2. 分核策略

- **优先满核且按实际任务上限裁剪**：
  - AIV Scale 路径：`blocks = clamp(n, 1, aivCores)`；
  - Direct SIMT 3M 路径：按三角总元素数 $n(n+1)/2$ 估算所需 block 数，裁剪至 `[1, aivCores]`；
  - Prep 预打包路径：按对齐列数 `wsCols` 分核，`prepBlocks = clamp(wsCols, 1, aivCores)`；
  - Fused / On-chip MIX 3M 路径：按上/下三角有效 Tile 总数 $\text{triTiles} = \frac{\text{divN}(\text{divN}+1)}{2}$ 分核，`usedCores = clamp(triTiles, 1, aicCores)`，其中 `aicCores = GetAicCoreCount()`。
- **三角均匀分块、无跨核屏障**：每个 MIX block 独立按 `tileIdx = block, block + usedCoreNum, ...` 步进，利用无浮点 `sqrt` 的整数二分反演函数 `DecodeTriTile` 直接将 `tileIdx` 映射到三角内有效二维块坐标 `(row, col)`，彻底跳过另一半非 `uplo` 区域；且不使用跨核 `SyncAll`，在任意物理核数或算力切分下均无死锁。

##### 3. 数据分块与内存优化策略

充分对齐 DAV_3510（Ascend 950PR）单核组片上容量（UB = 248 KiB、L1 = 512 KiB = 双 256 KiB Bank、L0A/L0B = 64 KiB、L0C = 256 KiB），全部布局由编译期 `static_assert` 强制校验：

- **Tile 尺寸选择（`SelectMixTileSize`）**：
  - `n <= 256`：`BS = 32`（提高小矩阵核间并行度）；
  - `256 < n < 896`：`BS = 64`；
  - `n >= 896` 且 `triTiles(128) >= aicCores` 且 `ubBytes >= 244 KiB`：`BS = 128`（相比 `BS=64` 将 GM→L1 访存流量减半，同时 $3 \times 128 \times 128 \times 4\text{B} = 192\text{ KiB} \le 256\text{ KiB}$ 恰好在 L0C 中同时常驻 $P_0, P_1, P_2$）。
- **Workspace 256B 对齐与 Bank 冲突消除**：
  - 预打包路径将 `wsRows` 向上对齐至 **64 float（256 Bytes）**，彻底消除 `wsRows % 32 == 16` 时每隔一行跨 128B 缓存行导致的 MTE2 带宽腰斩；
  - 当 `wsRows % 256 == 0 && wsRows + 64 <= 65535` 时额外偏移 `+64`（如 `4096 -> 4160`），打破 HBM / L2 2 的幂次步长组相联冲突；
  - 通过 `EnsureDefaultWorkspace(h, totalWsBytes)` 复用 `handle` 内置 workspace；当超过 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）或 `wsRows > 65535` 时自动无感回退零 workspace 的 Path 3a（On-chip MIX），绝不报错。
- **分段归约跨度（`SelectReductionSpan`）**：
  - `ACL_INT8`：`reductionSpan = 256`（因 $256 \times 256^2 = 2^{24}$，每 256 步在 FP32 累加器内 100% 比特无损）；
  - `ACL_COMPLEX64`：`k > 2048` 时 `reductionSpan = 1024`（每 1024 步在 UB 即时执行 `P2 - P0 - P1` 抵消合并，防止大 $k$ 累加漂移触及 `1e-2` 单点上限），`k <= 2048` 时单段累加零额外开销。

##### 4. 四路径路由（TilingKey）规划策略

| 路径 | 触发条件 | 启动 Kernel | Workspace | 适用场景 |
| --- | --- | --- | --- | --- |
| Path 1（纯缩放） | `(k == 0 \|\| alpha == (0,0)) && beta != (1,0)` | `csyrk3m_ex_scale_kernel` (AIV) | 0 | 退化分支，不触碰 `A` |
| Path 2（Direct SIMT 3M） | `n <= 28 \|\| (n <= 64 && max(\|alpha\|) > 100)` | `csyrk3m_ex_direct_kernel` (AIV SIMT) | 0 | 超小矩阵零 Cube 开销；极端 $|\alpha|$ 启用 Dekker/TwoSum 补偿 |
| Path 3b（主路径：Prep + Fused 3M） | `(n > 128 \|\| k > 256)` 且 workspace 适配 2 GiB | `csyrk3m_ex_prep_kernel` (AIV) + `csyrk3m_ex_fused_kernel` (MIX 1:2) | $O(NK)$ 复用 | 中大矩阵主力路径，单次 $O(NK)$ 预打包 + 单次 launch 融合 3M Cube |
| Path 3a（On-chip 融合 3M） | `28 < n <= 128 && k <= 256` 或 workspace 超限回退 | `csyrk3m_ex_mix_kernel` (MIX 1:2) | 0 | 小规模 MIX 零额外 launch 延迟，及超大矩阵零 workspace 兜底 |

Cube 计算精度模式：
- `Atype == ACL_INT8` → `INT8_HF32`（`HF32Mode::ENABLE`，1× `Mmad`，100% 精确且 8× Cube 吞吐）；
- `Atype == ACL_COMPLEX64 && !extremeAlpha && n*n*k >= 4M` → `HF32X3`（`HF32Mode::ENABLE`，3× `Mmad`：$x_l y_h + x_h y_l + x_h y_h$，2.67× Cube 吞吐 + 20~21 位尾数精度）；
- 其余小工作量或极端 $|\alpha|$ → Native `FP32`（`HF32Mode::DISABLE`）。

---

#### 3.2.2 Kernel 侧设计（`csyrk3m_ex_kernel.cpp`）

整体执行流程如下：

```
Host: 参数校验 & 标量探测 & 四路径 Tiling 路由
  │
  ├──► [Path 3b 主路径]
  │      ├─ Stage 1: csyrk3m_ex_prep_kernel (AIV-only, 2-slot UB 65KiB×2 乒乓)
  │      │     GM A ──(MTE2)──► UB ──(64-lane SIMD 解交织 + Xr+Xi + HF32 RNE残差)──► MTE3 ──► WS (3或6平面, 256B对齐, padding补0)
  │      │
  │      └─ Stage 2: csyrk3m_ex_fused_kernel (MIX_AIC_1_2, 单次 launch 融合全部 3M)
  │            ├─ AIC (Cube):
  │            │    WS ──(硬件 Nd2Nz MTE2, BS=128 采用 PANEL_K=128 逐乘积双256KiB L1 Bank 乒乓)──► L1
  │            │    L1 ──(LoadA2/LoadB2, INNER_K=32, 双32KiB L0A/L0B 乒乓)──► L0A / L0B
  │            │    L0A×L0B ──(3M: P0, P1, P2 同核常驻 L0C 192KiB)──► Fixpipe(dualDstCtl=1, ROW_MAJOR) ──► AIV0/AIV1 UB
  │            └─ AIV0 / AIV1 (各处理半个 Tile):
  │                 ├─ 多段归约: MergePartVf (Re=P0-P1, Im=(P2-P0)-P1 并累加)
  │                 ├─ 读 C_old (beta!=0 时 DataCopyPad + 对角块 MaskDiagCOldVf 屏蔽非 uplo)
  │                 ├─ FinalEpilogueVf (3M 组合 + alpha/beta 复数缩放 + 交错打包 + 零开销指数最大值探测)
  │                 ├─ 常规写回: 非对角整块 DataCopyPad / 上三角逐行 DataCopyPad / 下三角 ShiftLowerDiagRowsVf+DataCopyPad
  │                 ├─ 大 k (k>=511, C64) 对角块: 跳过对角写回 → ExactDiagElementsVf (16-warp 纯 __gm__ Kahan 3M 精算)
  │                 └─ 异常修复 (仅 maxExp==0x7f800000): 写回前触发 RepairTileNonFiniteVf (纯 __gm__ Netlib 顺序精确重算)
  │
  ├──► [Path 3a]: csyrk3m_ex_mix_kernel (AIV 片上实时解交织+NZ 转置 → L1，AIC 单次 launch 融合 3M，零 GM workspace)
  ├──► [Path 2]:  csyrk3m_ex_direct_kernel (AIV SIMT 3M，可选 Dekker TwoProd + TwoSum 补偿)
  └──► [Path 1]:  csyrk3m_ex_scale_kernel (AIV 仅缩放 uplo 三角，零访问 A)
```

关键微架构优化细节：

1. **AIV 预打包零散写流水线（`csyrk3m_ex_prep_kernel`）**：
   - 使用双槽 UB（每槽 `2048` 复元素 = 65 KiB，总计 130 KiB ≤ 248 KiB），以 `MTE2 / V / MTE3` 三级事件双缓冲流水执行；
   - `ACL_COMPLEX64` 通过 `DIST_DINTLV_B32` 解交织，在单次 SIMD 遍历内同步产出 $X_r, X_i, X_r+X_i$ 与 RNE 残差 $x - \text{round}_{\text{HF32}}(x)$（Inf/NaN 残差掩码置 0）；
   - `ACL_INT8` 通过 `DIST_UNPACK4_B8` + `Cast<int32_t, int8_t>(ZERO)` + `Cast<float, int32_t>` + `DeInterleave` 零标量循环完成精确升精与 3 平面生成；
   - Padding 行/列在 `prep` 阶段精确清零，保证后续 AIC 硬件 `Nd2Nz` 补齐 16 倍数边界无脏数据。

2. **`BS=128` 宽 K 逐乘积（Per-Product）L1 乒乓流水线**：
   - 相比 `PANEL_K=32` 一次性加载 12 平面（导致 `trans=T/C` 下 `Nd2Nz` 连续行长 `dValue` 仅 32 float = 128B、MTE2 带宽跌至 73.8 GB/s），本设计在 `BS=128` 启用 `PANEL_K=128, INNER_K=32`；
   - 每个 L1 Bank（256 KiB）存放单个乘积的 4 个 $128 \times 128$ 平面（`leftH/leftL/rightH/rightL` 各 64 KiB；对角块复用左平面减半至 128 KiB），在 `(kOuter, prod)` 上跨双 L1 Bank 异步乒乓预取；
   - 无论 `trans=N` 还是 `trans=T/C`，每次 `Nd2Nz` 突发均达 **128 float = 512 Bytes 连续传输**，描述符数缩减 4×；
   - `LoadA2`/`LoadB2` 支持传入外层步长 `outerDepth`（`srcStride = CeilDiv(outerDepth, 16)`），内层 `kInner = 0, 32, 64, 96` 切片加载至双 32 KiB L0A/L0B，每个 L1 Bank 驱动 12 次 HF32 `Mmad`，完全掩盖下一个 L1 Bank 的 HBM 预取。

3. **精准数值保护与严格写保护**：
   - **对角线 Kahan 3M 细化（`ExactDiagElementsVf`）**：针对 `HF32×3` 对角线上 $(x_l)^2 \ge 0$ 同号累加漂移，在 `!IsInt8A && k >= 511` 时以 16-warp（512 线程）纯 `__gm__` SIMT Kahan-TwoSum 精算 $O(n)$ 对角元素（严禁在 MIX SIMT 中访问 `__ubuf__`）；
   - **非有限值自动修复（`RepairTileNonFiniteVf`）**：在 `FinalEpilogueVf` 中通过向量位与+规约最大值零额外开销检测 `Inf/NaN`；仅当检测到非有限值时在写回 `C` 前触发纯 `__gm__` Netlib `csyrk.f` 顺序重算，确保 `TC_FL_*` 与 `TC_3M_415` 精确通过且常规路径零开销；
   - **严格三角与 padding 写保护**：对角块读 `C_old` 后通过 `MaskDiagCOldVf` 掩蔽非 `uplo` 位置（防止 `beta * NaN` 污染），写回时上三角逐行按有效长度 `DataCopyPad`，下三角经 `ShiftLowerDiagRowsVf` 左对齐后按精确字节长度写出，保证非 `uplo` 与 `ldc > n` padding 行 0 改写。

---

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（`arch35` / `DAV_3510`） | √ |

---

## 算子约束限制

1. 依赖 CANN `asc-devkit >= 9.1`（`ASC_DEVKIT_GE_9_1`）；
2. `Atype` 仅支持 `ACL_COMPLEX64` 与 `ACL_INT8`，`Ctype` 仅支持 `ACL_COMPLEX64`；
3. `uplo` 仅支持 `ACLBLAS_UPPER` / `ACLBLAS_LOWER`，`trans` 支持 `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C`（`OP_C` 不共轭，等价 `OP_T`）；
4. 维度与主维约束：`n >= 0`、`k >= 0`、`lda >= max(1, trans == OP_N ? n : k)`、`ldc >= max(1, n)`；
5. `alpha`、`beta` 不可为 `nullptr`；`n > 0` 时 `C` 不可为 `nullptr`；`n > 0 && k > 0 && alpha != (0,0)` 时 `A` 不可为 `nullptr`；
6. 输出矩阵 `C` 为对称（非 Hermitian）矩阵，仅更新 `uplo` 指定三角，对角元素虚部原样保留，不置 0。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 实测验收结果（Ascend 950PR, CANN 9.2.0） | 标准来源 |
| --- | --- | --- | --- |
| 精度标准 | `ACL_COMPLEX64` 实部、虚部分别按 FLOAT32 混合容差判定：`rtol = atol = 2^-13 (1.2207e-4)`，`matched_ratio >= 0.99`，`max_abs_error <= max(1e-2, 32*ULP)`；非 `uplo` 与 padding 行逐位精确匹配；CPU golden 采用 Netlib `cblas_csyrk`/`cblas_zsyrk` | `verify_accuracy.py` **1000 / 1000 ALL PASS**（`FAIL=0, MISSING=0`），叠加 `NullHandle` 与 `HostAndDeviceScalarPointerModes` 共 **1002 / 1002 PASSED**；全部保护区哨兵 0 改写 | 任务书 §3.2 / CANN 生态算子精度标准 |
| 性能标准（正式 `msprof op`） | `msprof op` 采集 `OpBasicInfo*.csv` `Task Duration(us)`（warmup 5 次不计入，后 10 次有效样本算术平均，多 kernel 累加转 ms），与 `gpu_baseline.csv`（GPU kernel-only `nsys`）对比：单 case `gpu_ms / npu_ms >= 0.4` 且全部 200 例算术平均 `>= 0.5` | 200 例 `TC_PF_*` **200 / 200 PASS（FAIL=0）**；算术平均倍率 **0.7798×**（中位数 0.7705×），单用例范围 **0.4398× ~ 1.4498×**（`4096×4096 C` 11.70 ms = 0.727×，`4096×4096 N` 12.38 ms = 0.698×）；`ACL_INT8` 5 例补充对标 C64 `gpu_ms` 为 **1.192× ~ 2.442×（平均 1.728×）**；`verify_performance.py` 交叉验证 200/200 PASS（平均 0.787×） | 任务书 §3.3 / §3.3.1 |
| 内存标准 | 任务书 §3.4 明确不涉及 | 片上缓冲编译期 `static_assert` 全部通过；workspace 超 2 GiB 自动回退零 workspace 路径 | 任务书 §3.4 |

## 兼容性分析

- 本算子为 `ops-blas` 新增接口 `aclblasCsyrk3mEx`，声明新增于公共头文件 `include/cann_ops_blas.h`（与 cuBLAS `cublasCsyrk3mEx` 参数序列一一对应），不修改任何既有算子的符号签名或行为；
- 已验证 `bash build.sh --ops=csyrk --soc=ascend950` 与 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 均独立编译通过，无符号冲突与回归。
