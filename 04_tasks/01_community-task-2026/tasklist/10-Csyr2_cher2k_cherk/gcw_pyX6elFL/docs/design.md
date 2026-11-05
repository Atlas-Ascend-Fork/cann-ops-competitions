# 需求背景（required）

## 需求来源

昇腾社区任务：Csyr2/cher2k/cherk 算子精度修复（A2/A3）。

任务地址：https://www.hiascend.com/activities/task-center/details/fc6145cabc15459ab08b3996cad3b8f9

任务书：https://www.hiascend.com/p/resource/202610/d7d6833b9324453688ba4f76cbef5f4d.zip

任务讨论：https://gitcode.com/cann/ops-blas/discussions/66

代码仓：https://gitcode.com/cann/ops-blas

本设计以 `f1e288f72c835bd1b4ece2c3119a181656a66a01` 为分析基线。当前处于设计与基线复现准备阶段，未提交实机通过结论。

## 背景介绍

三个接口均已有 arch22 实现。本任务保持接口和 BLAS 语义，修复三角更新时的输出写范围，使非操作三角及前导维 padding 与输入逐 bit 一致，同时满足操作区混合容差和 A2 性能要求。

| 接口 | 当前实现目录 | 当前测试目录 |
| --- | --- | --- |
| aclblasCsyr2 | `blas/syr2/arch22/` | `test/syr2/csyr2/arch22/` |
| aclblasCher2k | `blas/her2k/arch22/` | `test/her2k/cher2k/arch22/` |
| aclblasCherk | `blas/herk/arch22/` | `test/herk/cherk/arch22/` |

任务书将 Cher2k 列在 `herk/arch22` 下，当前代码实际位于 `her2k/arch22`；按现有目录实施，提交评审时说明映射，不新增平行 API 或实现目录。

### 当前实现分析

以下结论来自源码检查，尚需 A2/A3 基线运行确认影响范围：

1. Csyr2 的部分批量路径按矩形搬出，未操作的三角位置填零后仍参与 AtomicAdd。`csyr2_device_chunks.h` 的 PackedUpperChunk/CopyOutCols、`csyr2_device_lower_window.h` 的 StoreLowerBatch 均需审查。对有限普通数加零可能保持数值，但不能保证负零、NaN payload 和非规范数逐 bit 不变，也不满足非操作区零写。
2. Cher2k 存在对齐起点向前扩展、读取旧值后整段写回，以及合并旧 C 后整块写回的路径。重点是 Cher2kStorePlanarUnitTile、Cher2kSg3StorePartial、Cher2kDirectStoreTile 和 small/outer 路径。即使拷贝能够保留数值，也应按操作三角裁剪实际写地址。
3. Cherk 的 StoreDirectDiagonalTile 在 LOWER 分支复制旧前缀后写整列；StoreDirectTile 在特定完整对角 tile 上整块写回；StoreOuterK2 也会合并旧值后提交完整列。这些路径需要限制输出范围。
4. 现有测试不能直接作为本任务验收结论。Csyr2 的 SameComplex 使用浮点比较；Cher2k 的普通 CSV 验证未覆盖全部 padding 的 bit 比较，且将最大绝对误差兜底放宽至 2.5e-2；Cherk 共用测试将其放宽到与 k 相关的值。正式自验不沿用这些放宽。

# 需求分析（required）

## 需求描述

三个算子在 A2/A3 上满足同一组条件：

1. 公开签名、参数顺序、枚举、返回码和标量归属不变。
2. 原地更新仅写入 uplo 指定三角；另一三角及 `ld > n` 的 padding 零写。
3. 合法 quick return 不写整个输出缓冲，保留原对角虚部和特殊比特。
4. 操作区实部、虚部分别按 FLOAT32 混合容差验收。
5. 保持绑定 stream 的异步执行；不增加 Host 同步、不回退 CPU。
6. A2/A3 均提交功能、精度与 bit 检查；A2/910B3 达到任务书性能标杆。

## 需求拆解

| 工作项 | 交付内容 | 通过条件 |
| --- | --- | --- |
| 固定基线 | 源码 SHA、任务包 SHA、原始用例与环境记录 | 源码、二进制、日志可相互对应 |
| 输出地址修复 | 三个 arch22 算子的边界写回 | 非操作区零写；正常结果通过混合容差 |
| 语义回归 | quick return、对角、负步长、异常和标量检查 | 返回码及输出符合任务书 |
| 测试补齐 | 原 CSV、TC_BIT、哨兵、bit 检查、可靠运行脚本 | 无漏测、跳过或静默失败 |
| 性能验证 | 设备单次平均耗时及对应基线 | 有 warmup、有效采样 >50、逐 case 达标 |
| 验收材料 | 设计、自测代码、自测报告、代码 commit | 三个算子一起提交 |

# 详细设计（required）

## 算子分析

### 数学公式

Csyr2：`A = alpha*x*y^T + alpha*y*x^T + A`。普通转置，不共轭；对角虚部按普通复数运算更新。

Cher2k：

- N：`C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C`。
- C：`C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C`。
- alpha 为 complex64，beta 为 float32。

Cherk：

- N：`C = alpha*A*A^H + beta*C`。
- C：`C = alpha*A^H*A + beta*C`。
- alpha、beta 均为 float32。

Hermitian 族正常计算时忽略输入对角虚部，输出对角虚部为零；quick return 优先于输出更新。

### 支持数据类型与形状

矩阵、向量均为 complex64，实虚分量为 float32。列主序地址为 `M(i,j)=M[i+j*ld]`。Csyr2 支持 n≥0 和非零正负 incx/incy；Cher2k/Cherk 支持 n,k≥0、trans=N/C。前导维约束、无效参数和空指针行为遵照任务书及既有公开接口，不借修复增加新语义。

### 区域划分

| 区域 | UPPER | LOWER |
| --- | --- | --- |
| 操作区 | `0 <= i <= j < n` | `0 <= j <= i < n` |
| 非操作三角 | `j < i < n` | `0 <= i < j < n` |
| padding | `n <= i < ld, 0 <= j < n` | 同左 |

对角属于操作区。只有合法 quick return 才对整个输出缓冲执行不变性检查。

## 算子实现

### Host 侧设计

保持公开校验顺序、handle 绑定 stream、标量位置判定及当前分发接口。对本任务边界修复无必要的 Host 重构不纳入修改。

Csyr2：n=0 或 alpha=(0,0) 为 quick return；负步长逻辑起点按 `(n-1)*abs(inc)` 取反向遍历，不改变传入基地址规则。

Cher2k/Cherk：n=0，或无乘积项且 beta=1，为 quick return。若标量在 Device，沿用设备端判定，不通过新增 D2H/stream 同步判断 no-op；每次调用重新读取标量，避免地址不变、值变化时错误复用分支。beta=0 时计算结果不依赖旧 C 的数值。

workspace 沿用随 n/k 计算的临时空间。边界重排优先复用现有 UB 队列及索引缓冲；如需少量额外 UB staging，在 tiling 中计入容量。不为保护区域增加持久 Device 副本或整矩阵备份。

### Kernel 侧设计

#### 统一写范围

对输出 tile 的每列 j，将行区间 `[rowBase,rowBase+validRows)` 与操作区求交：

- UPPER：`begin=rowBase`，`end=min(rowBase+validRows,j+1,n)`。
- LOWER：`begin=max(rowBase,j)`，`end=min(rowBase+validRows,n)`。
- `end <= begin` 时不发起输出搬运。

完全处于操作区的 tile 保留现有批量搬运。跨对角或含尾部的 tile 按列提交交集；每段长度严格为 `(end-begin)*sizeof(aclblasComplex)`，不向前覆盖另一三角，也不向后覆盖 padding。

#### UB 对齐与特殊值

不能简单把 GM 起点裁剪后，直接以非对齐的 UB 子地址搬出。A2/A3 的 DataCopyPad 要求 UB 起点对齐，GM 允许非对齐。设计采用对齐 staging：按整数位模式 Gather/重排有效片段到 32 字节对齐的 UB 起点，再按有效字节数搬出。

重排不使用 Adds(x,0) 复制需保留的比特；不对非操作区做 AtomicAdd(0)。UB staging 的补齐数据不提交到 GM。队列释放和 scratch 复用前等待对应 MTE3 完成；保留现有 V/MTE2/MTE3 事件依赖。

接口约束参考：https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.html

实施时以目标 CANN 9.1.0 头文件、编译结果和 A2/A3 运行结果核验 API 可用性，不引入仅新版本支持的模式。

#### Csyr2

保留向量复数计算和已验证的核间分工。批量 UPPER 在靠近对角时按每列真实长度搬出；批量 LOWER 跳过对齐前缀，只提交从该列对角开始的更新量。AtomicAdd 若继续用于有效操作区，必须只覆盖该有效区。

通用 strided、windowed、full-window、tiny 和 packed 路径均纳入检查，不能只改默认入口。完全操作区内的多列块保留 2D 搬运；仅边界块增加重排，限制性能影响。

#### Cher2k

Cube 计算及复数乘加公式首先保持。统一裁剪 postprocess 输出：取消 LOWER 对角列对齐前缀读回写回；取消 direct、small、outer 路径用旧 C 合并另一三角后整块写出的做法。

N/C、unit/general alpha、beta=0/非零和不同 tiling 路径遵守同一写范围。beta 非零时仅将操作区旧值用于结果；beta=0 时跳过旧值参与计算。正常计算的对角虚部在操作区内置零。

#### Cherk

检查 StoreDirectDiagonalTile、StoreDirectTile、StoreSelectedTile、StoreOuterK2 及 tiny 路径。完整对角 tile 也属于边界块，不能因 validRows/validCols 达到整块大小就整块写出。

LOWER 不再复制前缀旧值后整列提交；outer K2 路径不再合并整个旧列提交。正常计算的对角虚部为零；无乘积但 beta≠1 时只缩放操作区。

#### 数值精度

先在严格标准下固定基线并分类失败：越界写、语义错误、有限值误差、非有限值分类。写回裁剪完成后，若操作区仍失败，针对失败形状定位复数展开、累加顺序和中间精度，不通过修改参考值、删除用例或扩大阈值解决。

如需改变计算路径，补充设计变更和该路径的 A2/A3 数值及性能对照，不默认重写全部计算实现。

### 计划修改范围

| 范围 | 计划修改 |
| --- | --- |
| 三个算子的 arch22 Kernel/内部头文件 | 输出裁剪、对齐 staging、必要的事件依赖和容量核算 |
| 三个算子的 arch22 测试 | 原 CSV 对接、哨兵填充、memcmp、严格混合容差、quick return 全缓冲检查 |
| 三个算子的 arch22 验证脚本和 README | 路径、完整性校验、设备耗时解析、复现步骤 |
| 公开接口与 arch35 | 不修改 |

Cherk 当前 arch22 包含 arch35 测试源。严格验收补齐应隔离在 arch22 中，不通过修改 arch35 测试改变其他架构行为。

## 支持硬件

| 硬件 | 验证要求 |
| --- | --- |
| Atlas A2 | 功能、精度、bit 检查；910B3 性能主门禁 |
| Atlas A3 | 功能、精度、bit 检查；性能跑通并记录 |

## 算子约束限制

保持列主序和既有 lda/ldb/ldc 语义，不扩展任意 stride、broadcast、数据类型或新 API。三个算子作为一个任务验收。内存要求通过代码检视，不新增任务书未要求的独立内存测试门禁。

# 可维可测分析

## 精度标准/性能标准

精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md

| 指标 | 标准 |
| --- | --- |
| 操作区分量容差 | rtol=atol=2^-13，matched_ratio≥0.99 |
| 最大误差 | 按任务书及标准的 `max(1e-2,32*ULP)` 口径；不额外引入 k 倍上限或 2.5e-2 放宽 |
| 非操作区 | 与调用前快照 memcmp，100% 一致 |
| quick return | 整个输出缓冲 memcmp 一致 |
| 参考库 | Netlib BLAS 3.10.0，记录构建来源和实际加载链 |
| 性能 | warmup 后有效采样 >50 次，A2 各 case 不高于任务书标杆 |

Csyr2 golden 按任务书指定的 Netlib ssyr2 语义实现复数参考；Cher2k/Cherk 首先保留任务书指定的 Netlib 3.10.0 CBLAS 参考。更高精参考可作为诊断记录，未获验收口径确认前不代替正式 golden。

### 用例与检查接入

原任务包共 3670 条用例：Csyr2 1222 条、Cher2k 1224 条、Cherk 1224 条。每个算子保留原 1000 条非性能及 200 条性能；额外 TC_BIT 分别为 22/24/24 条，包中已包含，不重复追加。

每个返回 SUCCESS 的有效计算用例，在调用前保存完整 `ld*n` 输入，调用并同步对应 stream 后读回。操作区走混合容差，非操作三角及 padding 走 bit 检查。哨兵由 uint32 模式写入，覆盖有限值、±0、不同 NaN payload、±Inf、非规范数；不对哨兵先做浮点运算。

quick return 另加全缓冲检查，不能只调用忽略操作区的检查函数。普通计算时 Hermitian 对角虚部严格检查为零；对角原虚部为非零或特殊值的输入也需覆盖。

覆盖 UPPER/LOWER、N/C、紧凑 ld 及 n+8/n+32、n=1/7/8/16/63/64/128，以及大尺寸和各专门分发边界；Csyr2 加入正负步长。补充回归必须由具体问题或分发路径支持，不以删减原 CSV 缩短测试。

### 性能用例

| 算子 | uplo/trans | n/k | 上限（us） |
| --- | --- | --- | --- |
| Csyr2 | UPPER | n=512 | 7.54 |
| Csyr2 | LOWER | n=1024 | 9.72 |
| Csyr2 | UPPER | n=2048 | 19.05 |
| Cher2k | UPPER/N | 1024/1024 | 1458.91 |
| Cher2k | UPPER/N | 2048/2048 | 5240.68 |
| Cher2k | LOWER/C | 1024/1024 | 1081.90 |
| Cherk | UPPER/N | 1024/1024 | 738.09 |
| Cherk | UPPER/N | 2048/2048 | 2616.28 |
| Cherk | LOWER/C | 1024/1024 | 485.94 |

其余 TC_PF 及对应 gpu_baseline.csv 保留。测量从输入与标量就绪后开始，使用绑定 stream 的设备事件或等价可核验计时，记录 warmup、有效迭代数、单位和原始输出。不得以 GTest 包含 CPU golden 的总耗时替代平均单次耗时。CSV 基线已含倍率折算，不能对任务书上限再次放宽。

### 运行结果完整性

校验测试进程返回码、GTest XML、预期与实际 case 清单。超时、崩溃、零用例、缺失、重复、跳过、解析失败均不判通过。性能的执行失败、缺基线和无测量值分别记录，不以 NO_REF 代替通过。保留 stdout/stderr、XML、环境记录和精度/性能结果表。

## 兼容性分析

本任务不改变 ABI/API、架构目录映射和返回码约定。需要更新的 Csyr2 TC_ED_151 原 CSV 期望为 INVALID_VALUE，但讨论帖评审已明确返回 INVALID_ENUM；原件保留，修正版和 diff 随交付件提交。

现有代码的额外设备/Host 标量路径和回归测试保留，目标是修复指定缺陷，避免回归。所有声称通过的结果必须来自本次固定版本在对应设备上的运行。

## 已知问题与实施前条件

1. 设计提交时说明 Cher2k 当前目录映射，避免评审按旧路径寻找代码。
2. 讨论帖中的高精 golden、NaN/Inf 分类问题尚待明确答复；保留双链诊断和失败，不改写官方结论。
3. HiDevLab 当前 A2/A3 资格待开通；原 A2 实例标签为 CANN 9.0.0，不作为 CANN 9.1.0 验收环境。须取得 A2/A3、核对软件版本后执行板测。
4. 先完成设计评审；随后按批准方案实施、全量自验、提交验收。本文不把静态检查或 CPU 分析写成 NPU 通过证据。
