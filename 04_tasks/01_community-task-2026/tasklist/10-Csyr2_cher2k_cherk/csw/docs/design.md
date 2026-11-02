# aclblasCsyr2 / aclblasCher2k / aclblasCherk 精度修复设计（A2/A3）

# 1. 需求背景

## 1.1 需求来源

本设计对应 10 月社区任务“aclblasCsyr2/aclblasCher2k/aclblasCherk 精度修复-A2/A3”，目标仓库为 https://gitcode.com/cann/ops-blas ，目标环境为 CANN 9.1.0、Atlas A2/A3（arch22）。三个算子必须作为同一任务交付。

## 1.2 问题与边界

三个算子均原地更新列主序三角矩阵。既有 arch22 实现的部分对齐块写回会越出 `uplo` 选中的三角，覆盖另一三角或 `lda/ldc` padding；用浮点加零保留旧值还会改变 NaN payload、±0 或非规范数的位模式。

本任务不是新增算子。公开接口、参数顺序、枚举、返回码、标量 Host/Device 归属、列主序和 stream 异步语义均冻结；不新增平行 API，不回退 CPU，不修改 arch35。

# 2. 需求分析

## 2.1 数学语义

- Csyr2：`A ← alpha·x·y^T + alpha·y·x^T + A`，普通对称、不共轭，对角虚部正常参与更新；支持非零正负 `incx/incy`。
- Cher2k：`trans=N` 时 `C ← alpha·A·B^H + conj(alpha)·B·A^H + beta·C`；`trans=C` 时 `C ← alpha·A^H·B + conj(alpha)·B^H·A + beta·C`。
- Cherk：`trans=N` 时 `C ← alpha·A·A^H + beta·C`；`trans=C` 时 `C ← alpha·A^H·A + beta·C`。
- Cher2k/Cherk 仅支持 N/C；正常更新时 Hermitian 对角虚部强制为 0，不补写另一三角。

## 2.2 操作区与非操作区

输出矩阵物理布局为 `ld×n`，`M(i,j)=M[i+j·ld]`：

| 区域 | UPPER | LOWER | 判定 |
| --- | --- | --- | --- |
| 操作区 | `0≤i≤j<n` | `0≤j≤i<n` | complex64 实、虚部分别按 float32 混合容差 |
| 非操作三角 | `j<i<n` | `0≤i<j<n` | 与调用前逐 bit 一致 |
| 前导维 padding | `n≤i<ld` | 同左 | 与调用前逐 bit 一致 |

quick return 时整个 `ld×n` 缓冲区保持不变。非操作区不得通过读后重写、补共轭或“先写满再恢复”来伪造一致。

## 2.3 验收拆解

1. 三算子操作区满足 `rtol=atol=2^-13`、`matched_ratio≥0.99` 及 `max_abs_error≤max(1e-2,32×ULP(g_low))`。
2. 所有成功计算用例均检查另一三角和 padding，`matched_ratio=1.0`；NaN payload、±0、±Inf、非规范数均覆盖。
3. 复用任务包原始 CSV 和脚本，保留全部精度、负向、边界及 200 条性能用例；追加 Csyr2 22 条、Cher2k/Cherk 各 24 条 `TC_BIT_*`。
4. Cher2k/Cherk golden 使用同一 Netlib BLAS 3.10.0 的复数 CBLAS 实现；按精度标准 §2.3 的“更高精度单一标杆”采用 double 主参考，并保留 complex64 float CBLAS 交叉输出，判定阈值仍严格为 FLOAT32；Csyr2 按 Netlib ssyr2 语义自实现复数参考。
5. 性能使用 10 次预热、51 次有效调用的 `avg_us`，A2 910B3 为主门禁，A3 跑通并记录；不得把 GTest 总墙钟或 host golden 时间当成算子耗时。
6. 不为保护非操作区新增与 n/k 无关的持久 Device 缓存，不增加 Host 流同步。

# 3. 详细设计

## 3.1 Host 与 Tiling

沿用句柄式接口和 Ascend C kernel 直调。Host 保留参数校验、quick return、标量地址归属和 stream 下发方式；Device 标量条件不为 quick return 增加 D2H 或 Host 同步。

Tiling 按 `uplo/trans/n/k/ld` 选择已有向量、Cube、direct 或小矩阵路径。每个输出元素只归属一个核；跨对角 tile、矩阵尾块与前导维尾块在写回前计算有效列和有效行区间。Csyr2 对 direct 路径的系数读取补齐 MTE2 到标量单元的同步，避免 DMA 尚未落地就读取 UB 位模式。

性能保持优化不改变写回区间：Csyr2 大 direct 路径合并独立 X/Y gather 与符号变换，并精简 unit-alpha kernel 的启动标量参数；对 `n>=640` 且 128-float 对齐的 UPPER unit-alpha 路径，改为两次 TransDataTo5HD 完成实虚交换并跳过 gather 下标表构建，其他 shape 保留原 Gather 路径。LOWER PrepareInputs 在等待输入 DMA 前构建独立的 gather 下标表；转置临时区复用尚未进入 ProcessBatch 的既有 `outputs_` 空间，下标表仍保留在 `indices_`，不新增 UB 或 Device 缓冲。对 `n=1011/1074` 的 direct-eight 路径，逐列仅提交 `[8-k)` 个 complex64 对角头，取消无效的加零写和对应掩码；`n=1011` 使用 23 核并允许 body DMA 与只读对角 gather 并行（精确头/body 无重叠元素），`n=1074` 使用 36 核并保持串行，其余 shape 保持原分波写回。对实测受益的 `n=1011`，warm tiling 命中时仍通过 `aclrtGetDevice` 校验当前设备，再直接复用既有 Host 值，避免重复零初始化和拷贝；其他 shape 保持原填充路径。Cherk 的 `n=1` 复用已有 `tiny1_direct_kernel`，不再准备该 kernel 不使用的 metadata；`n=2/4` 复用 small UB kernel 但改为单核逐列执行，outerK2 及其他 shape 的选择保持不变。`smallPath=1/6` 在 tiling 初始化后直接固定为单核并返回，不调用平台 AIV/AIC/UB 查询，也不申请 workspace。

## 3.2 精确三角写回

对第 `j` 列，UPPER 仅提交 `[0,j+1)`，LOWER 仅提交 `[j,n)`，并与当前 tile 行范围求交。完全位于操作区的块保留批量路径；跨对角或尾块按真实 GM 起点和实际 complex64 个数写回。

各算子 arch22 目录内的 `arch22_complex_store.h` 实现以下无恢复写方案：

1. 根据起始元素相对 32B 对齐的位置计算短头长度。
2. 短头在 UB 内按 `uint64_t` 位模式搬移，不执行浮点加减，因此不改变 NaN payload、±0 或非规范数。
3. 通过 V→S、S→MTE3 等显式事件保证 UB 位搬移与 DMA 的顺序；标量消费者使用 MTE2_S 等待输入。
4. 短头和主体均以真实目标地址、真实字节数提交，不在 GM 上读取或重写非操作区。
5. 固定面板写回增加 `colBase+col<n` 尾部保护，防止越出逻辑列数。

Csyr2 的 unit/window/direct、Cher2k 的 direct/SG3/Cube/small 以及 Cherk 的 direct/Cube/outer 路径均按同一不变式审查：任何 GM 写区间的每个元素都必须属于当前 `uplo` 操作区，且列索引小于 n。

## 3.3 计算与特殊分支

保留原公式、负步长起点、quick return 和异常返回码。`beta=0` 不读取旧 C；Hermitian 对角仅输出实部计算结果及零虚部。必要的归一化或分块归约只用于使操作区满足任务书精度，不改变非操作区写禁止约束。

## 3.4 修改范围

实现和测试严格限制在对应 arch22 及测试工具内：

- `blas/syr2/arch22/`
- `blas/her2k/arch22/`（Cher2k 的实际仓内目录）
- `blas/herk/arch22/`（Cherk）
- `test/syr2/csyr2/arch22/`
- `test/her2k/cher2k/arch22/`
- `test/herk/cherk/arch22/`
- `test/utils/` 中本任务共用的 bit 检查、用例追加、结果汇总与复现脚本

精确写回 helper 放在上述各实现的 arch22 目录，不放入 `blas/common`，不修改 `include/cann_ops_blas.h`，不修改 arch35。

# 4. 测试设计

## 4.1 非操作区位级检查

`test/utils/check_untouched_bitwise.h` 在调用前保存完整 `ld×n` Host 快照，调用并同步后对非操作三角和 padding 使用 `memcmp`；quick return 对全缓冲比较。`TC_BIT_*` 在数据生成后覆盖非操作区哨兵，包含不同 NaN payload、±0、±Inf、正负非规范数及有限值。检查不是浮点相等，也不排除 padding。

## 4.2 精度与 CSV 完整性

三套任务包 CSV 全量保留：Csyr2 1222 条（含 22 BIT）、Cher2k/Cherk 各 1224 条（含 24 BIT）；Cher2k 额外保留上游回归，不以新增用例替换原用例。Cher2k/Cherk 默认强制 `*_FULL_REGRESSION=1`，避免只跑 smoke CSV。主参考与交叉参考均来自同一 Netlib 3.10.0，分别记录 `netlib_double_main_*` 与 `netlib_float_crosscheck_*`，只有主参考参与 PASS/FAIL，float 交叉中的旧 FAIL 不删除、不改名。遇 Inf/NaN 输入时，double 主参考同样按实、虚分量和实数 alpha/beta 展开，防止不同 CBLAS 实现的中间复数运算把 `0×Inf` 污染到另一分量。

任务包与仓内既有差异逐项按任务书处理：Csyr2 非法 `uplo=999` 按 §2.4 返回 `ACLBLAS_STATUS_INVALID_ENUM`；Cher2k 的 `TC_ED_203` 保留任务包的非空 C、`k=0/beta=0` 用例，同时把仓内额外 nullC 回归以独立用例保留。所有变更加入自测记录，不静默删例。

## 4.3 性能与复现脚本

独立自测包原样交付三份 `gen_csv.py` 与原始 CSV，不用于覆盖原 1200 条；代码 PR 的每个 arch22 测试目录交付 `verify_accuracy.py`、`verify_performance.py`、`gpu_baseline.csv` 和 README；共用目录交付 `append_bit_cases.py`、bit 参考实现及汇总工具。生成器与仓内验证入口分离，避免重复维护生成器时误改原用例。

性能入口分别使用 `csyr2_test`、`cher2k_test` 和 `cherk_benchmark`，固定 10 次预热、51 次有效调用。汇总器只解析独立 `avg_us/average_us` 记录，缺失、重复、非有限值、样本不足、超时和非零进程退出均判失败；前三个锚点使用任务书 §3.3 的 us 限值，其余按原 GPU 基线规则比较。

## 4.4 环境记录与通过条件

自测报告记录 CANN 9.1.0、驱动、Netlib BLAS 3.10.0、A2/A3 设备型号。A2 与 A3 均提交操作区精度和非操作区 100% bit 结果；性能报告给出逐项实测值与标杆对比。设计文档本身不预先宣称测试通过，最终结果以冻结代码的原始日志和报告为准。

# 5. 兼容性与内存分析

公开 ABI/API 不变；仅修正原本不应发生的写入及影响任务书精度的计算路径。写回辅助只使用既有 UB tile，不申请持久 Device 缓冲；没有 CPU 计算回退，也没有新增 Host 流同步。既有 workspace 仍按原问题规模使用，不为保护另一三角增加整矩阵恢复缓存。
