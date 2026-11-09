# aclblasLtMatmul A2/A3 设计方案

贡献者：`cylus`（个人）。版本：0.5，2026-10-28。本文为设计评审稿，不是验收通过报告。

当前完整精度测试已通过：`accuracy-full-03` 实跑 1000 例，1000 PASS、0 FAIL，进程退出码 0。运行时间为 2026-10-28 04:36:55.141376–04:41:29.948834 UTC，测试器记录耗时 274.5838885 秒。另有 build06 补充测试 6/6 通过。性能采集尚未形成有效证据，不宣称性能达标。

# 需求背景

## 需求来源

本方案对应 [aclblasLtMatmul A2/A3 社区任务](https://www.hiascend.com/activities/task-center/details/96a87ecddfe140e7bc29fc6959205ac5)，依据官方任务书及其测试附件编写。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，实现维护于 [cylus 的开发分支](https://gitcode.com/cylus/ops-blas/tree/cylus/aclblasltmatmul-a2)，目标实现目录为 `blas/lt_matmul/arch22`，复用 `blasLt` 中现有公开入口、句柄与描述符。

设计结构参考 [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/b11e879347ec583b6f0a309aeade8050c5d541ee/04_tasks/01_community-task-2026/resources/design_template.md)。附件归档 SHA-256 为 `d7669f02610c31523240ec5e66f2121ee3b2e80d145a7554ebc281f21dab0c4f`，包含 1000 条精度用例和 200 条性能用例。

## 背景介绍

为 A2/A3 补齐 cuBLASLt 风格的单精度实数矩阵乘加能力，与现有 arch35 共用公开接口，避免增加产品私有 API。实现需要兼顾转置、非紧凑行跨度、C/D 原地计算、标量缩放、错误码及空矩阵行为，并提供真实的精度、性能与内存证据。

# 需求分析

## 需求描述

实现 `D = alpha * op(A) * op(B) + beta * C`。使用已有 16 参数 ABI：

```cpp
aclblasLtMatmul(lightHandle, computeDesc,
               alpha, A, Adesc, B, Bdesc,
               beta, C, Cdesc, D, Ddesc,
               algo, workspace, workspaceSizeInBytes, stream);
```

参数类型和符号以 `include/cann_ops_blasLt.h` 为准，不增删参数，不省略调用方 workspace 或 stream。矩阵布局由各自 descriptor 指定；alpha/beta 的位置由 pointer mode 指定；algo=null 时执行自动选择。

## 需求拆解

| 项目 | 本次范围 | 验证方式 |
|---|---|---|
| 数学计算 | FLOAT32 乘法、累加与缩放 | Netlib BLAS 3.10.0 对照 |
| 布局 | ROW、单 batch、N/T/C、独立 ld、C==D | 完整 CSV、padding 哨兵 |
| 边界 | 空输出、k=0、alpha=0、beta=0、条件空指针 | 官方边界与负向用例 |
| 精度 | 混合容差、最大误差锚点、非有限值、EXACT | 全部 1000 例严格检查 |
| 性能 | 200 条给定形状的完整 API 设备耗时 | msprof 原始记录与 GPU 基线 |
| 交付 | 设计、代码、测试、复现材料与原始报告 | 固定提交版本，公开证据待补充 |

# 详细设计

## 算子分析

### 数学公式

```text
P[i,j] = sum(t=0..k-1, op(A)[i,t] * op(B)[t,j])
D[i,j] = alpha * P[i,j] + beta * C[i,j]
```

实数 C 转置等价于 T。A/B/C 为输入，除 C==D 指定原地场景外不能修改。只写 D 的逻辑元素，保留行尾 padding。

### 支持数据类型

A/B/C/D 与 scaleType 均为 `ACL_FLOAT`，computeType 为 `ACLBLAS_COMPUTE_32F`，default epilogue。不将混合精度或 Bias/ReLU/GELU 纳入本任务主路径。Cube 使用完整 FP32，关闭 HF32；向量与小矩阵后端也使用 FP32。

HOST 标量直接读取。DEVICE 标量当前采用 Host fallback：先同步传入 stream，再分别复制 4 字节 alpha/beta，供条件空指针校验及后端选择。因此 DEVICE 标量调用不承诺完全异步，基础 DEVICE alpha/beta 补充用例已通过，异常恢复边界仍待补充。

### 支持形状

矩阵采用 ROW、单 batch，batchCount=1、stridedBatchOffset=0，m/n/k 非负。附件最大 m/n/k 均为 4096，此范围不等同于接口上限。

| 矩阵布局 | 物理形状 | 归一后 ld 下限 |
|---|---|---|
| A，N | m×k | max(1,k) |
| A，T/C | k×m | max(1,m) |
| B，N | k×n | max(1,n) |
| B，T/C | n×k | max(1,k) |
| C、D | m×n | max(1,n) |

ld=0 按物理行宽归一。C!=D 的 ldc/ldd 可不同；C==D 时归一后的布局必须一致。尺寸、偏移与字节数采用宽整数及溢出检查。

## 算子实现

### 实现方案

保留共享公开入口，通过架构分派调用 arch22 Host 实现。后端按矩阵形状和 alpha/beta 选择，不识别测试编号或输入数据分布。

#### Host 侧设计

校验顺序为 `handle → 五个描述符与 alpha/beta 判空 → pointerMode → m/n 为零返回 → 布局 → 必要数据指针 → workspace`。m/n=0 不访问矩阵数据，也不启动 kernel。其他分支如下：

| 条件 | 行为 |
|---|---|
| alpha=0 或 k=0 | 不读取 A/B，计算 `D=beta*C` |
| beta=0 | 不读取 C，包括其旧值中的 NaN |
| alpha=0/k=0 且 beta=0 | 将 D 逻辑区域置零 |
| alpha=0/k=0 且 beta=1 | 位精确复制 C；C==D 时保留原值 |
| 有非空输出 | 要求 D 非空，按实际计算需要校验 A/B/C |

无效 handle 返回 `NOT_INITIALIZED`；空描述符、所需数据指针或非法维度/ld/batch/stride 返回 `INVALID_VALUE`；非法 trans 枚举返回 `INVALID_ENUM`；不支持的类型与 order 返回 `NOT_SUPPORTED`。C=null 且 beta!=0 采用官方负向 CSV 的 `INVALID_VALUE` 要求；正文对此存在表述差异，提请评审确认。

algo 非空时按 `max_workspace_bytes` 检查实际容量及非空要求；非空 workspace 必须 16 字节对齐。algo=null 且 workspace=0 的默认路径可按需要使用 handle 内缓存。workspace=null 且 size>0 本身不单独构成错误，仍由实际算法合同决定。

heuristic 当前只提供 arch22 实际支持的 128×128×64、policy0、splitK=1 候选，按 `m*align8(n)*4` 申报通用 raw 容量；m/n/k 为零或专用内存受限路径可申报零。它不接收 alpha/beta，因此该容量是通用路径上限。16³ 显式通用算法真实使用 1024 字节，容量不足必须返回错误。

kernel 使用调用方 stream。启动前后通过 `aclrtGetLastError(ACL_RT_THREAD_LEVEL)` 消费并传播已出现的 Runtime 错误为 `EXECUTION_FAILED`；设备异步错误仍需调用方同步检查。分配、回收、同步及 DEVICE 标量复制失败后的完整恢复序列待专项验证。

#### Kernel 侧设计

| 分派条件 | 后端 | 中间 GM |
|---|---|---|
| 无乘积，仅清零/复制/缩放 | AIV epilogue | 无 raw |
| 需要乘积，`abs(alpha)>1 && k>16` | 按 K 递增的 AIV Axpy | 不申请 raw |
| 0<m,n,k<=16，或极窄矩阵 raw 超内存预算 | 顺序 FP32 小矩阵后端 | 自动路径直接写 D；显式正容量算法写 raw 后 epilogue |
| 其他完整 GEMM | Cube FP32 MatmulImpl | 满足直接输出条件时无 raw，否则 raw+epilogue |

**Cube 路径。** 使用 NN/NT/TN/TT 四种模板，C 归一为 T；静态 tile 为 128×128×64，单段 MAX_K=8192，保留真实 lda/ldb/ldRaw。K 超出单段时由 Host 分段累加。M/N/K 尾块按有效范围处理。`alpha==1 && beta==0 && ldd%8==0` 时直接写 D；其他情况先写 `m*align8(n)*4` 字节 raw，再由 AIV 按独立 ldc/ldd 应用 alpha/beta。多 kernel 耗时必须完整统计。

**有序 Axpy 路径。** 每块负责一行的最多 64 列，每次搬运 256 个 K 项，逐 K 执行向量 Axpy。B 不转置时用 beta*C 初始化输出，将 alpha*A 的结果作为逐 K 系数；B 转置时 Gather 出独立列，先累加乘积，再应用 alpha/beta。A/B 转置与 ld 均保留。beta=0 不读 C；C==D 先读当前块 C 再写 D。静态 UB 请求 67,584 字节，无中间 GM raw。此路径用于控制放大系数下的 FP32 累加及缩放顺序，可能明显慢于 Cube，性能必须另测。

**小矩阵与 epilogue。** 小矩阵采用顺序 FP32 计算，避免微小形状的 Cube 启动成本，并保留极端值的中间溢出行为。显式正容量算法先将未缩放乘积写 raw，再 epilogue；自动路径或内存受限零容量算法直接写 D。仅缩放时 beta=1 采用位精确复制，beta=0 不读 C，alpha=0/k=0 不读 A/B。

#### Workspace 与并发

默认算法需要 raw 而调用方未提供时，handle 按 stream 保存独立缓冲；同一 stream 依赖队列有序复用，不同 stream 不共享 raw。mutex 覆盖缓存选择与 kernel 入队。

对每次非空调用，按四个 layout 的物理行分配量计算 `B=floor(1.2*sum(rows*normalizedLd*4))`，扣除非空调用方 workspace 的完整容量，再检查**整个 handle 所有 stream**的保留缓冲合计。超预算时先同步设备，再释放全部缓存，避免同步可能已被调用方销毁的旧 stream 句柄。新增或扩容前再次为新请求预留预算。非空但不足实际 raw 的调用方 workspace 返回 `INVALID_VALUE`，不额外分配一份空间补足。

HOST 缓存充足且预算满足时不主动同步；DEVICE 标量、单 stream 缓存增长、全 handle 超限回收及销毁存在同步。不能承诺所有调用完全异步，也不能把后端无 raw 写成整个 API 的 HBM 峰值为零。

统计必须包含调用方预分配、各 stream 缓存与回收峰值。m/n=0 按合同提前返回保留的缓存、其他 handle、分配器粒度以及调用方自身超过预算的容量不受上述内部预算完全约束。C==D 去重口径仍需单列：如 257×17×2 的某原地用例，raw 为 24,672 字节，四 layout 合计 37,144 字节，去重后为 19,668 字节；去重比例约 1.2544，尚不能声称这一口径满足 1.2 倍要求。内部缓存是否符合任务书“workspace 由调用方分配”的要求，提请评审确认。

## 支持硬件

| 产品 | 实现目标 | 已有证据 |
|---|---|---|
| Atlas A2 / Ascend910B3 | arch22 | 单卡 64 GiB HBM、CANN 9.1.0，已编译并执行精度测试 |
| Atlas A3 | arch22 | 代码目标覆盖，尚无 A3 实机结果 |

最终报告补充 driver、firmware、编译器、源码提交与依赖版本，不将 A2 结果表述为 A3 实测通过。

## 算子约束限制

仅覆盖本任务的 FLOAT32、ROW、单 batch、default epilogue；不支持广播和任意非连续视图。仅承诺指定的 C==D 别名场景，D 与 A/B 重叠、C/D 部分重叠没有已确认的官方语义。高 alpha 有序后端和全 handle 预算已在 build06 编译集成，完整官方精度集合已通过 1000/1000；性能、实测 HBM 及专项恢复验证仍待补充。

# 可维可测分析

## 精度标准

Golden 为 Netlib BLAS 3.10.0 `cblas_sgemm`，对所有逻辑输出比较。有限元素满足 `abs(actual-golden)<=2^-13+2^-13*abs(golden)` 的比例必须 >=0.99。

任务书直接引用的 [官方混合容差标准](https://gitcode.com/cann/opbase/blob/d5dc8e47d0d92affee17b490837bf37cdaafd0d8/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) 第 93、146 行确定最大误差规则：选择有限 actual/golden 对中的最大绝对误差位置 i*，取该位置 golden 转为输出类型后的 g_low，要求 `max_abs_error<=max(0.01,32*ULP(g_low))`。ULP 为绝对间距，不是全局最大位序距离。并列位置采用固定顺序并保留最严格并列点诊断；0/非规格化数及最大有限值不能产生错误的零或无限阈值。

非有限值按任务附件单独比较：NaN/NaN、同号 Inf/Inf、NaN/Inf 可匹配；有限/非有限及异号 Inf 不匹配。非有限值不计入有限误差统计，专门比较须全部通过；alpha=0 额外执行 EXACT。

| 验证批次 | 结果 | 适用范围 |
|---|---|---|
| 当前完整运行 full03，2026-10-28 04:36:55–04:41:29 UTC | 1000 实收，1000 PASS、0 FAIL，退出码 0 | build06；耗时 274.5838885 秒 |
| 历史完整运行 full02，2026-10-28 04:11:35–04:16:10 UTC | 1000 实收，995 PASS、5 FAIL，退出码 1 | 修正前历史记录，不代表当前结果 |
| 剩余问题修正的独立验证 | 11/11 通过 | 局部证据，不能替代全量 |
| build06 补充边界与 stream 用例 | 6/6 通过 | DEVICE 标量、原地 ld 拒绝、条件空 A/B、workspace 对齐、双 stream 隔离 |
| 200 条性能验收 | 尚无有效完整结果 | 不宣称性能达标 |

以下为修正前 full02 的 5 条历史失败，均为 alpha=100、beta=0；它们已在 full03 完整回归中通过，表中误差仅保留用于说明修正背景：

| 用例 | m×n×k / 转置 | 最大绝对误差 | 锚点阈值 |
|---|---|---:|---:|
| TC_EX_392 | 2×9×2000 / NN | 0.08203125 | 0.0625 |
| TC_EX_584 | 513×257×1000 / NC | 0.16796875 | 0.125 |
| TC_EX_648 | 7×4×2000 / NC | 0.0703125 | 0.0625 |
| TC_EX_712 | 1000×2000×2000 / NT | 0.4609375 | 0.25 |
| TC_EX_824 | 1000×31×1025 / TN | 0.16015625 | 0.125 |

build06 补充测试 6/6 通过：DEVICE alpha/beta、C==D 时不同 ld 的拒绝、alpha=0 和 k=0 时空 A/B、workspace 非对齐拒绝、两个 stream 的 scratch 隔离。该结果独立于官方 1000 例，不扩展为所有 Runtime 失败恢复或全部缓存预算场景已验证。

严格检查测试进程状态、预期用例集合、重复/缺失/跳过及逐例断言，不把包装脚本零退出码单独视为成功。补充测试覆盖 padding、ld=0、不同 ldc/ldd、原地布局、条件空指针、workspace 对齐/容量、多 stream 缓存回收、DEVICE 标量与失败恢复。当前逐例原始日志和冻结源码提交的公开链接待补充。

## 性能标准

使用 msprof 的设备 task duration，以完整 API 调用为统计单元。raw+epilogue 等多 kernel 路径必须聚合全部设备任务；不以 GTest wall time 或 Golden 计算时间替代。计划每例 5 次预热、至少 51 次有效采样，保留原始值与均值。

附件有 200 条 GPU 基线、198 个唯一形状/转置键。任务正文要求 >=0.8× GPU，末尾要求 >=1.0×；同时报告两项，以 >=1.0× 为优化目标，正式门槛提请确认。基线说明同时出现 A100 和 P102-100/CUDA12.5，来源尚待确认。内存要求也同时出现输入输出量 1.2 倍及 GPU workspace 1.2 倍，附件未提供 GPU workspace，不构造缺失数据。

当前 profiler 参数传递及采集通道仍在排查，首轮试采已中止，没有可用于达标判断的有效数据。有序 Axpy 对性能的影响、完整 200 例统计及实测 HBM 峰值均待补充。

## 兼容性分析

公开 ABI、handle 与 descriptor 类型保持不变，新增分派和算法范围限制于 arch22。arch35 路径保持既有实现；最终交付需附 arch35 构建和必要回归证据。HOST 常态异步、DEVICE 标量同步 fallback 和预算回收的设备同步均明确披露。

完整官方精度集合已通过；设计评审后继续补齐公开精度原始报告、性能、内存与错误恢复报告。代码及测试的公开提交、可复现构建命令、依赖版本、环境快照和原始日志链接随实现进展追加；未完成项不标记为验收通过。


## 本次进展对应源码

开发分支：[cylus/aclblasltmatmul-a2](https://gitcode.com/cylus/ops-blas/tree/cylus/aclblasltmatmul-a2)。

本次提交版本：[`60fe449d7390`](https://gitcode.com/cylus/ops-blas/commit/60fe449d73908b730b7aca91c5121fe81f1ee729)。该提交保存已验证精度的首版实现和复现测试；正式性能验证仍在进行。
