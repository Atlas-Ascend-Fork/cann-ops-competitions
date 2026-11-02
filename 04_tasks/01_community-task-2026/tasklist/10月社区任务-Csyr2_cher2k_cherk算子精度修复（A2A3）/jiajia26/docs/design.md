# aclblasCsyr2/aclblasCher2k/aclblasCherk 精度修复设计文档（A2/A3，arch22）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | aclblasCsyr2/aclblasCher2k/aclblasCherk 精度修复-A2/A3 |
| 开源仓 | https://gitcode.com/cann/ops-blas |
| 适配硬件 | Atlas A2 / Atlas A3（arch22）；性能主门禁 Atlas 800I A2 (910B3) |
| CANN 版本 | CANN 9.1.0 |
| 开发语言 | Ascend C（仅 arch22 目录内修复，不动 arch35/950PR） |
| 设计目标 | 非操作区二进制一致（硬门禁）+ 操作区混合容差 + 性能不差于标杆 |

---

# 需求背景（required）

## 需求来源

社区任务《aclblasCsyr2/aclblasCher2k/aclblasCherk 精度修复-A2/A3》任务书。本任务为**已有算子精度修复**：接口冻结、功能冻结，仅修复「更新 `uplo` 指定三角时改写三角矩阵非操作区域」的精度缺陷。

## 背景介绍

`aclblasCsyr2`（对称秩 2 更新）、`aclblasCher2k`（厄米特秩 2k 更新）、`aclblasCherk`（厄米特秩 k 更新）已合入 ops-blas 的 A2/A3（arch22）实现，语义对标 cuBLAS `cublasCsyr2/cublasCher2k/cublasCherk`（Netlib `ssyr2/cher2k/cherk`）。三者均为**原地（in-place）三角更新**：列主序物理布局为 `ld × n` 个 `aclblasComplex`（complex64）元素，仅 `uplo` 指定三角（含对角）被引用与更新；严格另一三角与前导维 padding（`n ≤ i < ld`）属于**非操作区**，语义上不得被读后写、不得被 tiling 覆盖，调用后必须与调用前**逐 bit 一致**（含 NaN payload、±0、±Inf、非规范数）。

当前实现在部分路径上以「矩形 tile / 等长列块」为写回粒度，未按三角边界裁剪，或以「先算脏再恢复」的方式借用非操作区作为计算缓冲，导致非操作区出现清零、数值改写或位型破坏，不满足上述硬门禁。

# 需求分析（required）

## 需求描述

在 `blas/syr2/arch22/`、`blas/herk/arch22/`、`blas/her2k/arch22/` 目录内完成修复，使三个算子同时满足：

1. **非操作区二进制一致**（任务书 §2.0 硬门禁）：非操作三角 + `lda`/`ldc` padding 与调用前输入 bit-exact（`memcmp`，`matched_ratio = 1.0`）；quick return 时整个 `ld × n` 缓冲区不变。
2. **操作区混合容差**（任务书 §3.2.1）：COMPLEX64 实部/虚部按 FLOAT32 判定，rtol = atol = 2^-13，`matched_ratio ≥ 0.99`，`max_abs_error ≤ max(兜底, 32*ULP)`；Hermitian 族对角虚部强制置 0。
3. **接口与功能冻结**：不改 `include/cann_ops_blas.h` 签名；公式、quick return、对角置 0、负步长规则、Host 同步语义、返回码全部不变。
4. **性能不差于任务书 §3.3 标杆**（910B3，COMPLEX64，warmup + >50 次采样均值）。
5. **无规模无关的持久 Device 缓存**（任务书 §3.4 代码检视项）。

## 现状与根因分析

修复前对 arch22 现有实现的写回路径逐条走读，确认以下非操作区破坏点（行号以 master 当前版本为准）：

### blas/syr2/arch22（aclblasCsyr2）

| # | 位置 | 模式 | 非操作区后果 |
| --- | --- | --- | --- |
| R1 | `csyr2_device_chunks.h` `PackedUpperChunk` → `CopyOutCols` | 组内 b 列统一 `blockLen = cur` 批量写回；各列计算长度 `nv[kk]` 按三角裁剪（`valid = j0+kk+1-i0`），但写回长度取组内最大值，短列超出段输出的是 `Duplicate(0)` 的填充值 | UPPER 组内靠前列的严格下三角区（非操作区）被**清零** |
| R2 | `csyr2_device_chunks.h` `PackedLowerChunk` | 先 `Axpy` 全长 `cur*2` 把整块算脏，再用 `Adds(seg, src, 0.0f, prefix*2)` 从读回的旧 A「恢复」前缀（读后写模式）；且 `ProcessGroup` 中 `row0 = j0 & ~3` 向下 4 对齐，写回起点低于列操作区起点 | 恢复用 `Adds` 依赖浮点加法：**`-0 + 0 = +0`（符号位丢失）、sNaN 被 quiet 化（payload bit 改写）**，哨兵位型被破坏；若前缀恢复被对齐裁剪遗漏则直接数值污染 |
| R3 | `csyr2_device_xy.h` `FinishInterleavedChunk`、`csyr2_device_lower_window.h` 窗口路径 | 对列前缀（非操作三角行段）执行 `Duplicate(aOut, 0.0f, prefix*2)` 后整段写回 | 非操作三角被**直接清零** |
| R4 | 读侧 `CopyInCols` / `LoadXYWindow` | LOWER/UPPER 组按对齐行块读取旧 A，起点低于三角边界时把未引用三角读入参与计算（`R2` 的恢复依赖） | 未引用三角「被引用」，语义违规；同时放大带宽开销 |

### blas/herk/arch22（aclblasCherk）

direct kernel 已具备 tile 级 `diagonalMask` / `triangleMask` 与 `StoreDirectUnitUpperTile` 的逐列前缀写示范，但主路径仍存在：

| # | 位置 | 模式 | 非操作区后果 |
| --- | --- | --- | --- |
| R5 | `cherk_direct_kernel.cpp` `StoreDirectTile` 主分支 | 对角 tile（`fullTile && oldLoaded`）走**矩形 tile 批量写**：非三角区依赖「先读旧 C → `Select` 掩码合并 → 写回」恢复。合并虽为位操作，但 `ApplyDirectScalars` 的 `alpha`/`beta` 标量乘作用域若未按掩码屏蔽整 tile，非三角区数值经过乘法流水后位型被改写（sNaN quiet 化、Inf×0→NaN 等） | 对角 tile 非三角区位型破坏/数值污染；且「读旧合并」本身违反「不得读后写」，多一次 C 读带宽 |
| R6 | `cherk_direct_kernel.cpp` 尾块路径 | `n` 非 `DIRECT_EPILOGUE_TILE` 倍数时 `validRows/validCols` 裁剪后仍按矩形提交，掩码表（`CHERK_ARCH22_META_*`，host 预生成）只覆盖整 tile 尺寸；`ldc > n` 的 padding 行段写回长度未按 `n` 再次收紧 | 尾块 padding 位段被写 |
| R7 | `cherk_compute_kernel.cpp` / `cherk_kernel.cpp`（Cube 主路径） | tile 调度按 `mBase/nBase` 三角筛选，tile 内部写回按矩形 `DataCopy` 提交，斜边 tile 缺乏逐元素/逐列掩码 | 斜边 tile 非三角区被 Cube 结果覆盖 |

### blas/her2k/arch22（aclblasCher2k）

| # | 位置 | 模式 | 非操作区后果 |
| --- | --- | --- | --- |
| R8 | `cher2k_kernel_postprocess.cpp` / `cher2k_kernel_layout.cpp` | 与 herk 同框架的 tile 级后处理写回，alpha 复数（含 `conj(alpha)` 项）与双输入 A/B 使 tile 内合并分支更多；对角置 0、`trans=N/C` 两形态的正交组合下，同一斜边 tile 缺少统一裁剪 | 同 R5–R7 |

## 需求拆解

1. 建立统一**写回不变式**：任一 GM 提交 `(offset, blockLen, stride)` 必须落在该列/该 tile 与操作区的交集内（§3.1）。
2. csyr2：消除 R1–R4（§3.2）。
3. cherk：R5–R7 修复，direct 主路径从「读旧合并」切换为「按掩码裁剪的精确提交」（§3.3）。
4. her2k：与 cherk 同方案（§3.4）。
5. 测试接入：复用任务包 CSV + 新增 `TC_BIT_*` 哨兵用例与 `check_untouched_bitwise.h` GTest（§3.5）。
6. 性能与内存约束自证（§3.6）。

# 详细设计（required）

## 3.1 总体设计：写回不变式与三类 tile 处理

**写回不变式**（所有算子、所有路径）：设列 `j` 的操作区行区间为
- UPPER：`[0, j]`，长度 `len(j) = j + 1`
- LOWER：`[j, n-1]`，长度 `len(j) = n - j`

则任一 GM 写回必须满足：`offset ∈ 操作区` 且 `offset + blockLen ≤ 操作区终点`；`dstStride = (ld - 实际写回复元素数) * sizeof(aclblasComplex)`，保证 `[n, ld)` padding 永不出现在任何 `blockLen` 内。违反即评审不通过。

tile 分类处理（列主序，`TILE_M × TILE_N`，`rowBase/colBase` 为 tile 左上角）：

| tile 类型 | 判定 | 写回策略 |
| --- | --- | --- |
| **纯三角 tile** | tile 完全位于操作三角内（UPPER：`colBase - rowBase ≥ TILE_N`；LOWER：`rowBase - colBase ≥ TILE_M`） | 保持现有整块矩形提交（性能最优路径不变） |
| **斜边 tile**（跨三角边界，含对角 tile） | 其余 | **按列精确提交**：对 tile 内每一列 `j = colBase + c`，有效行数 `r = clamp(len(j) - rowBase, 0, TILE_M)`；`r = 0` 跳过，否则单列 `DataCopyPad(blockLen = r * 8 字节)`，UB 内用列内偏移 `c * TILE_M` 定位。逐列长度互不相同，不能用单一 `blockCount/blockLen` 批量 |
| **对角 tile**（Hermitian 族附加） | `rowBase == colBase` | 在斜边处理基础上，UB 内对角虚部先置 0（`Duplicate(zero)` + `Select(diagonalMask)`），再按上表提交；`beta=0` 时**不读旧 C**，非三角区零提交 |

配套规则：

1. **禁止读后写恢复**：删除 csyr2 `Adds(src, 0)` 恢复、cherk/cher2k「读旧 C → Select 合并 → 整 tile 写回」；非操作区既不读、更不写。 cher2k/cherk 中 `beta ≠ 0` 需要读 C 时，**只读 tile ∩ 操作区交集**（读偏移与长度同步裁剪）。
2. **禁止前缀/尾部清零**：删除所有 `Duplicate(0) + 整段写回` 的填充提交。
3. **对齐不改写回起点**：行块起点不得为对齐便利而低于三角边界（删除 `row0 = j0 & ~3` 类下取整，改为写回偏移内对齐）；不足 32B 的提交一律 `DataCopyPad`（`blockLen` 精确字节），禁止凑整。
4. **读侧裁剪**：旧 A/C 读回的 `(offset, len)` 同样按操作区交集裁剪；未引用三角零读。
5. **标量乘作用域**：所有 `Muls/Axpy/Select` 的 `count` 参数等于裁剪后的有效元素数，不存在「全 tile 乘、掩码事后恢复」。

## 3.2 aclblasCsyr2 修复设计

| 根因 | 修复 |
| --- | --- |
| R1 `PackedUpperChunk` 统一长度写回 | `CopyOutCols` 改为逐列提交：列 `j = j0 + kk` 写回长度 `len = min(cur, max(j+1 - rowBase, 0))`，`len > 0` 时 `CopyOutFloats(base + j*lda*2, src[kk*pitchFloats], len*2)`（内部维持现有 32B 对齐走 `DataCopy` / 否则 `DataCopyPad` 的分流）；批量 `DataCopyPad` 仅保留在「组内各列 len 相同」的纯三角情形 |
| R2 `PackedLowerChunk` 读后写 | 计算起点改为列内偏移 `off = max(j0+kk - rowBase, 0)`，`Axpy` 从 `aOut[kk*pitch + off*2]` 起作用长度 `(cur - off)*2`；前缀 `[0, off)` 不计算、不提交；删除 `Adds` 恢复段；写回长度按列变为 `cur - off`（斜边 tile 逐列提交，同 R1） |
| R3 前缀清零 | `FinishInterleavedChunk` / 窗口路径删除 `Duplicate(0, prefix*2)`；GM 写偏移改为 `gmComplexOff + prefix`，长度 `(cur - prefix)*2` |
| R4 `row0 = j0 & ~3` 下对齐 | `ProcessGroup` 行块从 `row0 = j0` 起（不因 DMA 对齐而扩域）；块内 DMA 对齐需求由 `DataCopyPad` 承担 |
| R4' 读侧 | `CopyInCols` 起点同步改为操作区交集（`max(rowBase, j)`），读长同步收紧 |

保持不变：6 个 tilingKey 分派、`n=0`/`alpha=(0,0)` quick return、负步长 `kx = 1-(n-1)*incx` 起点、对称不共轭（对角虚部不特殊处理）、Buffer 划分与 `usedCoreNum` 列前缀负载均衡（纯 tiling 函数 `csyr2_tiling.h` 除删除对齐下取整外不动）。

## 3.3 aclblasCherk 修复设计

1. **direct kernel 主路径**（R5）：`StoreDirectTile` 对角 tile 一律走 `StoreDirectDiagonalTile` / 逐列前缀提交（推广 `StoreDirectUnitUpperTile` 已验证的「selected prefix per column」模式到全部 alpha/beta 组合）；删除 `fullTile && oldLoaded` 的矩形批量分支与对应 `LoadDirectOldC` 全 tile 读。`beta ≠ 0` 时旧 C 读取按列交集裁剪（`LoadDirectOldC` 增加 `(rowOff, len)` 参数，`blockLen` 交集化）。
2. **标量乘作用域**（R5 配套）：`ApplyDirectScalars` 的 `alpha` 乘、`beta` 乘与对角置 0 的 `Select` 全部改为「裁剪后长度」作用域，斜边 tile 按列循环内完成。
3. **尾块与 padding**（R6）：`validRows/validCols` 收紧到 `min(tile, n - base)` 后，**再与 `len(j)` 求交**；掩码元数据表（`CHERK_ARCH22_META_*`）host 生成端补齐非整 tile 尺寸的掩码行段，或改为 device 端按 `rowBase/colBase/n` 闭式生成列长度（择一，倾向后者：省 host 预生成表，消除 meta 与运行时参数失配风险）。
4. **Cube 主路径**（R7）：`cherk_compute_kernel.cpp` 斜边 tile 的 C 写回复用同一「逐列精确提交」helper；纯三角 tile 维持矩形提交。
5. **workspace**（内存约束）：epilogue 中转 workspace 保持 tile 级生命周期（逐 tile 覆盖写），不新增与 n/k 无关的持久缓存；「保护非操作区」不引入任何额外 Device 内存。
6. 功能冻结项复核：`trans=T` 返回 `ACLBLAS_STATUS_INVALID_VALUE`、`k=0 且 beta=1` quick return、`beta=0` 时 C 无效输入不读 C、对角虚部置 0，均在现有 host/device 分支上保留。

## 3.4 aclblasCher2k 修复设计

与 §3.3 同方案、同 helper：

- 差异点 1：alpha 为 COMPLEX64 且含 `conj(alpha)` 双项——两项 `Axpy/Muls` 的作用域同样按裁剪长度；
- 差异点 2：A、B 双输入，pack 阶段（`cher2k_kernel_layout.cpp`）不涉及 C 写回，仅 epilogue/postprocess（`cher2k_kernel_postprocess.cpp`）按 §3.1 三类 tile 处理；
- 差异点 3：对角虚部置 0 与 `trans=N/C` 正交组合共用同一对角 tile 分支，无额外路径。

## 3.5 验证设计

1. **功能与操作区精度**（复用任务包 CSV，不删减原用例）：安装到 `test/syr2/csyr2/arch22/`、`test/herk/cher2k/arch22/`、`test/herk/cherk/arch22/`，CSV + GTest 框架；golden 按 Netlib BLAS **3.10.0**（Cher2k/Cherk 用 cblas 复数实现，Csyr2 由测试工程按 `ssyr2` 语义自实现复数参考），混合容差按任务书 §3.2.1。
2. **非操作区 bit-exact（新增，硬门禁）**：接入 `test_cases/check_untouched_bitwise.h` 到三个算子 arch22 GTest——调用前 Host 快照完整 `ld × n` 缓冲，同步后 D2H，非操作三角 + padding 逐元素 `memcmp`（real/imag 视作 uint32 比较）；所有返回 `SUCCESS` 的用例（含 TC_LD/TC_EX padding 场景）全覆盖。
3. **TC_BIT_* 哨兵用例**：`append_bit_cases.py` 追加（不改原 1200 行）——Csyr2 22 条、Cher2k/Cherk 各 24 条；UPPER/LOWER × `ld ∈ {n, n+8, n+32}` × 哨兵（NaN 不同 payload、±Inf、±0、非规范数、随机有限值）× `trans ∈ {N, C}`（Hermitian 族）；`n ∈ {1, 7, 8, 16, 63, 64, 128}`。
4. **quick return 用例**：`n=0`；`alpha=0`；`(alpha=0 或 k=0) 且 beta=1`——整缓冲 bit 不变。
5. **负向用例**：`incx/incy=0`、`n/k/lda/ldb/ldc` 越界、`trans=T`、非法枚举、nullptr——返回码对齐 cuBLAS，不写回。
6. **性能**：910B3 上按 §3.3 标杆 3+3+3 个 case 及仓内 `TC_PF_*`，warmup + >50 采样均值；A3 全量跑通并记录。
7. **环境记录**：自测报告记录 CANN、NPU 型号、驱动、Netlib 3.10.0 版本。

## 3.6 性能与内存影响评估

1. 斜边/对角 tile 逐列提交的额外 MTE3 事务数：每 tile ≤ `TILE_N` 次（仅边界 tile），纯三角 tile 批量路径不变；按 §3.3 标杆 case（n=512~2048）估算边界 tile 占比 ≤ 2%，期望耗时增量远低于标杆余量，最终以实测为准。
2. 删除「读旧合并」后，斜边 tile 少一次 C 读，直接路径带宽净下降。
3. csyr2 删除 `Adds` 恢复段后，向量计算长度按列收紧，UB 占用不增。
4. 无新增全局 Device 缓存；workspace 维持 tile 级复用。

## 3.7 代码修改范围

| 文件 | 修改内容 |
| --- | --- |
| `blas/syr2/arch22/csyr2_device_chunks.h` | R1/R2/R4：`CopyOutCols` 逐列精确提交；`PackedUpperChunk/PackedLowerChunk` 裁剪计算与写回长度；`ProcessGroup` 去除行起点下对齐 |
| `blas/syr2/arch22/csyr2_device_xy.h` | R3：`FinishInterleavedChunk` 去前缀清零、写回偏移/长度裁剪 |
| `blas/syr2/arch22/csyr2_device_lower_window.h`、`csyr2_device_columns.h`、`csyr2_device_init.h` | R3/R4 同模式核查与修复（前缀清零、读起点裁剪） |
| `blas/syr2/arch22/csyr2_host.cpp`、`aclblas_csyr2_host.h` | 如 tiling 派生字段调整则同步（接口与校验不变） |
| `blas/herk/arch22/cherk_direct_kernel.cpp` | R5/R6：`StoreDirectTile` 对角 tile 一律逐列前缀提交；`LoadDirectOldC`/`ApplyDirectScalars` 交集化；尾块掩码闭式化 |
| `blas/herk/arch22/cherk_compute_kernel.cpp`、`cherk_kernel.cpp`、`cherk_tiling_data.h` | R7：斜边 tile 逐列精确提交 helper 接入 |
| `blas/her2k/arch22/cher2k_kernel_postprocess.cpp`、`cher2k_kernel_layout.cpp`、`cher2k_tiling_data.h` | R8：同 §3.4 |
| `test/syr2/csyr2/arch22/`、`test/herk/cher2k/arch22/`、`test/herk/cherk/arch22/` | 任务包 CSV 安装 + `TC_BIT_*` 追加 + `check_untouched_bitwise.h` 接入 GTest + README |

不改：`include/cann_ops_blas.h`、`include/cann_ops_blas_common.h`、`arch35/`（950PR）全部文件、Host 同步语义、返回码定义。

## 3.8 风险与对策

| 风险 | 对策 |
| --- | --- |
| 逐列提交引入 MTE3 事务碎片，性能回退 | 仅边界 tile 逐列；对齐时尽量合并同长度相邻列；§3.3 标杆 case 与 `TC_PF_*` 全量实测把关 |
| 掩码元数据 host/device 生成切换引入新失配 | 采用 device 闭式生成为主；TC_BIT_* 的 `ld ∈ {n+8, n+32}` 与 `n ∈ {63,64,128}` 交叉用例兜底验证 |
| `DataCopyPad` 字节级提交与 UB 对齐约束冲突 | 维持现有 `blockLen % 32` 分流；UB 侧列 pitch 保持 32B 对齐，GM 侧由 Pad 语义保证不越界 |
| 三算子交付不一致 | 同一 PR（或一组关联 PR）交付，缺一不合入 |

---

## 交付与合入

- PR 目标：https://gitcode.com/cann/ops-blas ，仅含 §3.7 所列 arch22 实现与 test 目录；三算子同一 PR 或关联 PR。
- 验收环境：Atlas A2（910B3）+ Atlas A3 双卡精度与 bit-exact 结果、910B3 性能对标结果、CANN 9.1.0、Netlib BLAS 3.10.0 版本记录。
- 参考资料：任务书 §6 所列混合容差标准、cuBLAS/Netlib 文档与 Ascend C 开发文档。
