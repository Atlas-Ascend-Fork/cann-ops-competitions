# aclblasCsyr2 / aclblasCher2k / aclblasCherk 精度修复设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 ID | `fc6145cabc15459ab08b3996cad3b8f9` |
| 任务名称 | 10 月社区任务：aclblasCsyr2 / aclblasCher2k / aclblasCherk 算子精度修复（A2/A3） |
| 贡献者 | 2401_88896705 |
| 适配硬件 | Atlas A2 / Atlas A3（arch22）；性能门禁设备 Atlas 800I A2（910B3） |
| CANN | 9.1.0 |
| 三方参考 | Netlib BLAS 3.10.0 |
| 分析基线 | `cann/ops-blas` `c340cfa0a6c5732d74e0cfa4c7cec9a336793b82`（2026-11-03 快照）；任务包起始基线 `05b96ea22cea926ecfd43fc7ac691a795522bd61` |
| 文档状态 | 设计评审稿；代码未实现，未上板，实测数据另见自测报告 |

---

## 1 需求背景

### 1.1 需求来源

- 任务页面：https://www.hiascend.com/activities/task-center/details/fc6145cabc15459ab08b3996cad3b8f9
- 官方讨论：https://gitcode.com/cann/ops-blas/discussions/66
- 任务包：本任务 ZIP，内含 `aclblasCsyr2_Cher2k_Cherk精度修复_A2A3_task_doc.md` 与 `test_cases/`
- 设计模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
- 设计提交说明：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md
- 混合容差标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
- 代码仓库：https://gitcode.com/cann/ops-blas
- 社区任务流程：https://gitcode.com/org/cann/discussions/39

### 1.2 背景介绍

本任务是在已有 aclblasCsyr2、aclblasCher2k、aclblasCherk 接口上修复精度与写回语义，不是重新开发算子。要求保持公开接口、公式、quick return、列主序、负步长、Hermitian 对角规则与异步 stream 语义不变，只修改 arch22 实现与对应测试。

Csyr2 位于 `blas/syr2/arch22/`；Cher2k 位于 `blas/her2k/arch22/`；Cherk 位于 `blas/herk/arch22/`。任务书将 Cher2k/Cherk 统一写作 `blas/herk/arch22`，与当前仓库实际目录不一致；本设计以仓库实际目录为准。

### 1.3 已有实现的问题

对当前 master 的 arch22 写回路径逐条做静态审计，问题集中在两类：

1. **非操作区被写**：写回 tile 的矩形范围大于当前 `uplo` 的操作区，或为了 32B 对齐把 LOWER 行起点向下扩展到严格上三角，随后补零/整块写回。
2. **非操作区被读后写**：为“保留”非操作区旧值，先用 `GetValue/SetValue` 或 `Adds(old, 0.0f)` 合并旧 C，再整块写回。即使最终 bit 多数相同，也会改变 `-0.0`、NaN payload、非规范数，且违反“非操作区零写”。

典型位置：

| 类型 | 典型实现 | 后果 |
| --- | --- | --- |
| 整 tile 矩形写回 | Csyr2 `PackedUpperChunk` 的 `CopyOutCols(gmComplexOff, aOut, cur, b)` | UPPER 每列 `j+1` 行之后的行被写回，严格下三角被覆盖 |
| 对齐前缀整块写回 | Csyr2 `StoreLowerBatch`、`FinishInterleavedChunk`、`FinishPlanarChunk`；Cher2k `cher2kSg3StorePartial`、`Cher2kStorePlanarUnitTile`；Cherk `StoreDirectDiagonalTile` | LOWER 对齐前缀属于严格上三角，补零后整块写回会把上三角清零 |
| 旧值读改写 | Cher2k `cher2kSg3StorePartial`、`Cher2kStorePlanarUnitTile`；Cherk `StoreDirectTile` 的 `fullTile && oldLoaded` 分支、`StoreDirectDiagonalTile` 的 `Adds(oldAos, 0.0f)` | 非操作区被读后写，特殊值 bit 可能变化 |
| 整列写回 | Cher2k `Cher2kSmallGemvOuterStore` / `Cher2kSmallGemvOuterBatch` | UPPER 按 `n` 行整列写回，严格下三角被覆盖，并预读整块旧 C |

此外，操作区精度还受数值路径影响：

- 长 k 归约的消减误差：当前代码已有 `Cher2kSmallFusedAdd` 补偿，但只覆盖部分小矩阵路径，需确认大规模/k 长归约路径是否全部使用补偿累加；
- 极端有限输入在中间乘法中溢出：当前 Cherk 已有非有限结果探测与局部处理，需确认 generic Cube 路径是否覆盖所有非有限输出；
- 若实测操作区超出混合容差，按 §3.4 的兜底方案补补偿累加与扩展范围重算，不以非操作区修复替代。

本设计的结论边界：上述为非操作区问题的**源码级证据**；操作区是否超差、以及具体失败 shape，必须由任务包 `csvrunner`、`check_untouched_bitwise` 与 benchmark 上板确认。未取得这些结果前，不声称操作区已修复。

---

## 2 需求分析

### 2.1 功能与接口

| 编号 | 要求 | 设计响应 |
| --- | --- | --- |
| F1 | 三个算子公式、参数顺序、标量归属不变 | 不改公开头文件、不改公式 |
| F2 | 只更新 `uplo` 指定三角（含对角） | 写回区间按 `uplo` 逐列裁剪 |
| F3 | Hermitian 输出对角虚部为 0 | 保留现有对角置 0 逻辑，作用域改为选中元素 |
| F4 | Csyr2 对角虚部不做特殊处理 | 保持对称公式，不清零 |
| F5 | quick return 不改矩阵 | `n==0`、`alpha==0`、`k==0 && beta==1` 等路径保持 |
| F6 | 负步长、stream、非连续语义不变 | 不改 Host 下发与 stream 绑定 |
| F7 | 不改 arch35、不新增平行 API | 仅改 arch22 |

### 2.2 非操作区硬门禁

严格另一三角与 `ld` padding 必须与调用前输入逐 bit 相同，`matched_ratio = 1.0`。覆盖合法计算、`lda/ldc` padding、哨兵 NaN、quick return；负向用例返回非 `SUCCESS` 时不要求写回，但一旦返回 `SUCCESS` 也必须一致。

### 2.3 性能与交付

- 性能按任务书 §3.3 的 9 项标杆，A2（910B3）为主门禁，A3 跑通并记录；
- 三个算子同一任务交付，不拆分；
- 设计文档 PR 合入 cann-competitions，代码与测试 PR 合入 ops-blas。

---

## 3 详细设计

### 3.1 算子分析

数据类型为 `aclblasComplex`（实部/虚部各 FLOAT32），不扩展 dtype、广播或高维 ND。

| 算子 | 公式 | 标量归属与形状 |
| --- | --- | --- |
| Csyr2 | `A := alpha*x*y^T + alpha*y*x^T + A`，普通转置，不共轭 | 复数 `alpha` 在 Host；x/y 在 Device，逻辑长度 n，incx/incy 非零、支持负值；A 为列主序 `lda × n` |
| Cher2k N | `C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C` | 复数 `alpha`、实数 `beta` 在 Device；A/B 为 `n × k`，C 物理缓冲 `ldc × n` |
| Cher2k C | `C := alpha*A^H*B + conj(alpha)*B^H*A + beta*C` | A/B 为 `k × n`，其余同上 |
| Cherk N | `C := alpha*A*A^H + beta*C` | 实数 `alpha/beta` 在 Device；A 为 `n × k` |
| Cherk C | `C := alpha*A^H*A + beta*C` | A 为 `k × n` |

参数合法性、最小 leading dimension 与 trans 支持范围见 §3.6。

### 3.2 Host、分核与 tiling 设计

- 继续使用现有 Ascend C kernel 直调与 `aclblasSetStream` 绑定 stream；Host 不新增同步，不改变 Device/Host 参数归属。
- 不新增 tilingkey，不新增外部 workspace，不新增与 `n/k` 无关的持久 Device 缓存。
- 写回边界由运行时 `uplo`、列号 `j`、tile 行起点 `r0`、tile 行尾 `r1` 和 `n` 计算，不依赖用例 ID、固定 shape 或主机侧查表。
- 完全选中 tile 沿用原二维批量 DMA；跨对角线 tile 逐列提交，逐列长度在 kernel 内计算，不改变 tiling 结构。
- 分核与 tile 所有权保持原状；逐列写回不跨核重分配输出。
- quick return 在 Host 入口保持“直接返回、不启动 kernel”，确保整个缓冲不变。
- 负步长 `incx/incy` 仍由原 kernel 处理；本次只改输出写回，不重写输入索引。
- 大 `ld` 保持 64-bit 偏移；`strideBytes > 0xffffffff` 时沿用原有回退路径。

### 3.3 Kernel：非操作区裁剪 / 掩码写回

#### 3.3.1 统一写回契约

对输出 tile，设覆盖列 `[c0, c1)`、行 `[r0, r1)`，每列 `c` 的有效区间：

```text
UPPER: s(c) = r0,             e(c) = min(r1, c + 1)
LOWER: s(c) = max(r0, c),     e(c) = min(r1, n)
```

只提交 `[s(c), e(c))`：

```text
DataCopyPad(
    dst = C[(c * ld + s(c)) * 2],
    src = lane(c) + (s(c) - r0) * 2,
    blockLen = (e(c) - s(c)) * 8 字节)
```

tile 分两类：

1. **完全选中 fast path**：UPPER 判据 `r1 <= c0 + 1`，LOWER 判据 `r0 >= c1 - 1`，保留原二维批量 DMA。
2. **跨对角线 crop path**：逐列计算 `[s(c), e(c))`，逐列 1D DMA。数量级为每个 panel 对角线相交 tile，远小于总 tile 数。

实现约束：

- 禁止 `Duplicate(prefix, 0)` + 整块写；禁止 `Adds(old, 0.0f)` 合并旧值；禁止 `GetValue/SetValue` 保留非操作区。
- UB lane 有效段起点偏移 `2*(s(c)-r0)` 个 float；`DataCopyPad` 使用非 32B 起始地址。
- 若目标版本对 UB 源起始地址有 32B 限制，使用现有 `Gather` + 偏移表把有效段压到 lane 起始；Csyr2 用 `idxBuf_`，Cher2k/Cherk 用 `interleaveOffsets`。不新增 GM workspace。
- 所有 DMA `e(c) <= n`，padding 不参与写回；二维批量 DMA 保留 `ld - cur` 的 `dstStride`。

#### 3.3.2 Csyr2

- `PackedUpperChunk`：完全选中时保留 `CopyOutCols`；跨对角线时逐列写 `min(cur, j+1-i0)` 行，源 `aOut[kk*pitchFloats]`，目的 `C[(j*lda+i0)*2]`。
- `StoreLowerBatch`：逐列计算 `prefix = max(0, (j0+kk) - r0)`（Window 路径 `r0=row0`，packed 路径 `r0=row0+off`）；只写 `r0+prefix` 起 `cur-prefix` 行，源 `aOut[kk*pitchFloats + prefix*2]`。
- 删除 `ApplyLowerPrefixMask`、`MaskLowerWindowPrefix`、`MaskLowerFullPrefix`、`ZeroLongPrefixes` 及其调用；LOWER 前缀只计算不写回。
- `FinishInterleavedChunk`：不再补零前缀，写 `CopyOutFloats(gmComplexOff*2 + prefix*2, aOut[prefix*2], (cur-prefix)*2)`。
- `FinishPlanarChunk`：`CopyInFloats` 从 `(gmComplexOff+prefix)*2` 读 `valid=cur-prefix` 个复数；`Gather` 只处理有效段；写回 `gmComplexOff*2 + prefix*2`。LOWER unit 路径 `xyOff` 用 `(i0+prefix)-jBegin_`。
- `PackedLowerChunk` 已不在 LOWER 主路径调用，删除，避免保留错误写法。
- `ProcessColumn` LOWER 分支的 `i0=j&~3` 前缀由上述 `prefix` 处理，后续 chunk `prefix=0`。
- UPPER 单列/批量路径已按 `j+1` 写回，保持不变。

#### 3.3.3 Cher2k

- `cher2kSg3StorePartial`（LOWER）：删除标量旧值保留；逐列 `prefix=max(0, c-rowBase)`，只写 `validRows-prefix` 行，目的 `C[(c*ldc+rowBase+prefix)*2]`。
- `Cher2kStorePlanarUnitTile`（LOWER）：删除 `GetValue/SetValue` 与对齐前缀整块写；改为逐列从 `prefix` 起写。`Cher2kStoreDirectUnitTile` 已正确，仅审计。
- `Cher2kDirectStoreTile`（LOWER 对角）：删除 `Adds(old,0)` 合并；`loadOld` 仅由 `beta != 0` 决定；`beta != 0` 时按列加载操作区，不读严格上三角。
- `Cher2kSmallGemvOuterStore` / `Cher2kSmallGemvOuterBatch`：`unitAlphaNoBeta && upper` 时不再预读整块旧 C；按列只写 `min(n, colBase+col+1)` 行。
- `Cher2kTinySquareProcessColumn` 已按列裁剪，保持不变。
- `cher2k_k2_outer_kernel` 当前无 Host 调用点，仅审计；若后续启用，按同一规则处理。

#### 3.3.4 Cherk

- `StoreDirectTile`：`diagonalTile` 为真时无条件走逐列 `StoreDirectDiagonalTile`，删除 `fullTile && oldLoaded` 的二维整块分支。
- `StoreDirectDiagonalTile` LOWER：`prefix=min(validRows, absCol-rowBase)`，只写 `validRows-prefix` 行；删除 `oldAos` 合并与 `preservedRows`。
- `LoadDirectEpilogue` / `LoadDirectOldC`：`loadOld=(beta != 0)`；`beta != 0` 时按列只加载 `[s(c),e(c))`。
- `ApplyDirectScalars`：删除 `mergeTriangle`/旧值合并；`alpha` 乘、`beta` 乘、对角虚部置 0 的 `Select` 只作用于有效元素。
- `StoreDirectUnitUpperTile`（UPPER unit-alpha/beta-zero）已按列裁剪，保持不变。
- `cherk_compute_kernel.cpp` 的 `StoreSelectedTile` 与 `cherk_kernel.cpp` 的 Tiny/小尺寸路径已按列裁剪，不做功能修改；只确认 LOWER 的 `firstRow/validRows` 无符号下溢。
- `LoadAndApplyBeta` 若加载整 tile 旧 C，仅影响带宽，不产生非操作区写；实现时按列裁剪读取范围。

### 3.4 操作区数值路径保障

- 保留当前已有的补偿/非有限处理：`Cher2kSmallFusedAdd`、Cherk 非有限探测与对角规则。
- 实现后先跑原 CSV；若操作区出现消减误差或极值溢出超差，按以下顺序处理：
  1. 将补偿累加扩展到对应长 k 归约路径；
  2. 对有限输入产生的非有限结果，从已驻留输入按原公式局部重算；
  3. 扩展范围重算使用两项 float 有效数 + 显式指数，避免中间范围溢出；
  4. 原输入本身含 Inf/NaN 时保留 Netlib 运算顺序，不用重算改变语义。
- 所有重算在 Device 完成，不回退 CPU，不新增持久 Device 缓存。
- 该节为**条件方案**：是否启用由原 CSV 的机读结果决定，不在设计阶段假定操作区已通过。

### 3.5 支持硬件与兼容性

| 项目 | 要求 |
| --- | --- |
| 硬件 | Atlas A2、Atlas A3；性能门禁 Atlas 800I A2（910B3） |
| CANN | 9.1.0；自测报告记录驱动版本 |
| Netlib BLAS | 3.10.0，A2/A3 与自测报告使用同一版本 |
| 架构 | 只改 arch22；arch35/950PR 不动 |
| API | 公开头文件签名、枚举值、参数归属不变 |
| 数据布局 | 列主序、负步长、`lda/ldb/ldc` padding 语义不变 |
| 异步 | 保持 stream 绑定，不在生产 Host 入口新增同步 |
| 内存 | 不新增与 `n/k` 无关的持久 Device 缓存；不用“先写满再拷回” |

### 3.6 算子约束限制

| 约束项 | Csyr2 | Cher2k | Cherk |
| --- | --- | --- | --- |
| n/k | n≥0 | n≥0, k≥0 | n≥0, k≥0 |
| 最小 ld | lda≥max(1,n) | N: lda/ldb≥max(1,n)；C: lda/ldb≥max(1,k)；ldc≥max(1,n) | N: lda≥max(1,n)；C: lda≥max(1,k)；ldc≥max(1,n) |
| 步长 | incx/incy≠0，支持负值 | 不涉及 | 不涉及 |
| uplo | UPPER/LOWER | UPPER/LOWER | UPPER/LOWER |
| trans | 不涉及 | N/C；T 返回 INVALID_VALUE，非法枚举 INVALID_ENUM | 同左 |
| 标量 | alpha 复数、非空 | alpha 复数、beta 实数、均非空 | alpha/beta 实数、均非空 |
| 对角 | 不清零虚部 | Hermitian 输出对角虚部置 0 | 同左 |
| 空/quick return | n=0 或 alpha=(0,0)：成功且不改 A | n=0；或 (alpha=0 或 k=0) 且 beta=1：成功且不计算 | n=0；或 (alpha=0 或 k=0) 且 beta=1：成功且不计算 |
| 原地/视图 | A 原地覆写 | C 原地覆写 | C 原地覆写 |
| 确定性 | 不要求 | 不要求 | 不要求 |
| 高维/广播 | 不支持 | 不支持 | 不支持 |

### 3.7 代码修改范围

| 文件 | 修改内容 | 规模 |
| --- | --- | --- |
| `blas/syr2/arch22/csyr2_device_chunks.h` | `PackedUpperChunk` 双路径；`FinishInterleavedChunk`/`FinishPlanarChunk` 前缀裁剪；删除 `PackedLowerChunk` | 中 |
| `blas/syr2/arch22/csyr2_device_lower_window.h` | 删除掩码函数；`StoreLowerBatch` 逐列裁剪 | 中 |
| `blas/syr2/arch22/csyr2_device_lower_packed.h` | 删除前缀掩码调用；`ProcessLowerPackedOut` 裁剪 | 小 |
| `blas/syr2/arch22/csyr2_device_columns.h` | LOWER `prefix` 传递与 `FinishPlanarChunk` 调用点 | 小 |
| `blas/syr2/arch22/csyr2_ascend_kernel.h` | 删除掩码声明；新增 `prefix` 形参 | 小 |
| `blas/her2k/arch22/cher2k_kernel_postprocess.cc` | `cher2kSg3StorePartial`、`Cher2kDirectStoreTile` 裁剪 | 中 |
| `blas/her2k/arch22/cher2k_kernel_cube.cc` | `Cher2kStorePlanarUnitTile` 裁剪；`Cher2kStoreDirectUnitTile` 审计 | 中 |
| `blas/her2k/arch22/cher2k_kernel_entry.cc` | small gemv 外积去旧 C 预读；Tiny 路径审计 | 中 |
| `blas/herk/arch22/cherk_direct_kernel.cpp` | `StoreDirectTile` 对角强制逐列；`StoreDirectDiagonalTile` / `ApplyDirectScalars` / `LoadDirectEpilogue` 去旧值合并 | 中 |
| `blas/herk/arch22/cherk_compute_kernel.cpp` | 审计，无功能修改 | 0 |
| `blas/herk/arch22/cherk_kernel.cpp` | 审计，无功能修改 | 0 |
| `test/syr2/csyr2/arch22/` | 安装任务包 CSV；接入 `check_untouched_bitwise.h` | 小 |
| `test/herk/cher2k/arch22/` | 安装任务包 CSV；接入 bit 检查 | 小 |
| `test/herk/cherk/arch22/` | 安装任务包 CSV；接入 bit 检查 | 小 |

不修改公开头文件、CMake 公共接口、arch35 实现；不新增算子名、tilingkey 或平行 API。

### 3.8 实现顺序

1. 先落 Csyr2、Cher2k、Cherk 的非操作区裁剪写回；
2. 接入 `check_untouched_bitwise.h` 与 `TC_BIT_*`，在 A2 跑原 CSV + 新增 bit 用例；
3. 根据操作区结果决定是否启用 §3.4 数值兜底；
4. A2 跑 9 项性能与 `TC_PF_*`；
5. A3 重复精度/bit/性能并记录；
6. 输出自测报告，再提 ops-blas PR。

---

## 4 可维可测分析

### 4.1 精度标准

分两层，两层同时通过：

| 层 | 判定 |
| --- | --- |
| 操作区 | 逐元素 `|actual-golden| ≤ atol + rtol*|golden|`；FLOAT32/COMPLEX64 实部虚部：`rtol=atol=2^-13`，`matched_ratio≥0.99`，`max_abs_error≤max(1e-2, 32*ULP(g_low))`；±Inf/NaN 按标准 2.3 节 |
| 非操作区 | 严格另一三角 + `ld` padding 逐 bit `memcmp`，`matched_ratio=1.0` |
| 对角 | 属于操作区；Cher2k/Cherk 对角虚部 golden 与实现均按 0；Csyr2 不清零 |

golden：Csyr2 由测试工程按 Netlib `ssyr2` 语义自实现复数参考；Cher2k/Cherk 由 Netlib BLAS 3.10.0 cblas 生成。

### 4.2 测试用例与验证链

- 复用任务包原 CSV，不删减功能/精度/性能/负向用例：
  - `test_cases/csyr2/syr2_test.csv` → `test/syr2/csyr2/arch22/`
  - `test_cases/cher2k/cher2k_test.csv` → `test/herk/cher2k/arch22/`
  - `test_cases/cherk/cherk_test.csv` → `test/herk/cherk/arch22/`
- 原 CSV 1200 行保持不变；新增 `TC_BIT_*`：Csyr2 22 条、Cher2k 24 条、Cherk 24 条，由 `append_bit_cases.py` 追加。
- `TC_BIT_*` 覆盖 `UPPER/LOWER × 紧凑 ld / ld=n+8 / ld=n+32 × 哨兵 NaN payload、±Inf、极端值、随机有限值`；n 覆盖 1、7、8、16、63、64、128；Cher2k/Cherk 再正交 `trans=N/C`。
- GTest 调用前：Host 快照完整 `ld*n` 缓冲；调用并同步后 D2H；操作区走混合容差，非操作区 + padding 走 `memcmp`；所有返回 `SUCCESS` 的计算用例都检查，含 `TC_LD`/`TC_EX`。
- quick return 用例比较整个缓冲；负向用例返回非 `SUCCESS` 时不要求写回。
- 测试脚本：`gen_csv.py`、`verify_accuracy.py`、`verify_performance.py`、`check_untouched_bitwise.py`、`append_bit_cases.py`；README 写清编译、安装 CSV、跑精度+bit+性能的复现步骤。

### 4.3 性能标准

设备 Atlas 800I A2（910B3）；COMPLEX64；warmup 后有效采样 >50 次（建议 51 次）取平均；平均单次耗时不超过下表。

| 算子 | case | 参数 | 标杆（us） |
| --- | --- | --- | --- |
| Csyr2 | 1 | UPPER n=512 alpha=(1,0) incx=incy=1 lda=512 | 7.54 |
| Csyr2 | 2 | LOWER n=1024 lda=1024 | 9.72 |
| Csyr2 | 3 | UPPER n=2048 lda=2048 | 19.05 |
| Cher2k | 1 | UPPER N n=1024 k=1024 | 1458.91 |
| Cher2k | 2 | UPPER N n=2048 k=2048 | 5240.68 |
| Cher2k | 3 | LOWER C n=1024 k=1024 | 1081.90 |
| Cherk | 1 | UPPER N n=1024 k=1024 | 738.09 |
| Cherk | 2 | UPPER N n=2048 k=2048 | 2616.28 |
| Cherk | 3 | LOWER C n=1024 k=1024 | 485.94 |

其余性能项用仓内 `TC_PF_*` 与 `gpu_baseline.csv`。计时包括 API 下发到 stream 完成，不含分配、H2D/D2H、golden 校验；不用 GTest 整例耗时替代。

### 4.4 内存与兼容性

- 不设额外内存门禁；代码检视确认无与 `n/k` 无关的持久 Device 缓存，无“先写满整块再拷回”。
- 只改 arch22；arch35 无调用变化；保留版权头、错误码与资源释放路径。
- 不改公开接口、枚举数值、参数归属、quick return 与 Hermitian 规则。

### 4.5 风险与规避

| 风险 | 规避 |
| --- | --- |
| `DataCopyPad` UB 源起始 32B 对齐限制 | 实测；受限时用 `Gather` 压到 lane 首，复用现有偏移表 |
| 逐列 DMA 增加开销 | 完全选中 tile 保留二维批量；仅对角 tile 逐列；性能项验证 |
| LOWER 前缀计算浪费 | 前缀最多 3 个复数（Csyr2）或 tile 列宽（Cher2k/Cherk）；必要时按 prefix 裁剪计算 |
| `beta!=0` 旧 C 读取漏列 | 逐列 `[s(c),e(c))` 读取；CSV 覆盖 beta≠0/≠1 |
| 长 k 消减/极值溢出 | 先跑原 CSV；超差启用 §3.4 补偿/重算 |
| NaN/-0 语义 | 删 `Adds(...,0)`、补零、标量保留 |
| 大 ld 32 位截断 | 保持 64-bit 偏移；超阈值走逐列回退 |
| 快速返回误写 | Host 直接 return；quick return 用例比较整缓冲 |
| 目录/任务名重复 | 本 PR 使用任务包同名目录；如评审要求统一目录，文档内容不变，按评审指定平移 |

### 4.6 当前状态与待办

- [x] 源码审计与非操作区写回方案
- [x] 设计文档
- [ ] 代码实现
- [ ] A2 原 CSV + `TC_BIT_*` + bit 检查
- [ ] A2 9 项性能 + `TC_PF_*`
- [ ] A3 精度/bit/性能
- [ ] 自测报告
- [ ] ops-blas 实现 PR

本文档为设计评审稿，不宣称已完成验收；上板结果另见自测报告。

---

## 5 参考

1. 任务页面：https://www.hiascend.com/activities/task-center/details/fc6145cabc15459ab08b3996cad3b8f9
2. 官方讨论：https://gitcode.com/cann/ops-blas/discussions/66
3. 混合容差标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
4. ops-blas：https://gitcode.com/cann/ops-blas
5. Netlib BLAS 3.10.0：https://github.com/Reference-LAPACK/lapack/releases/tag/v3.10.0
6. Netlib ssyr2：https://www.netlib.org/blas/ssyr2.f
7. Netlib cher2k：https://www.netlib.org/blas/cher2k.f
8. Netlib cherk：https://www.netlib.org/blas/cherk.f
9. 设计模板：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
10. 设计提交说明：https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/README.md
11. 社区任务流程：https://gitcode.com/org/cann/discussions/39
