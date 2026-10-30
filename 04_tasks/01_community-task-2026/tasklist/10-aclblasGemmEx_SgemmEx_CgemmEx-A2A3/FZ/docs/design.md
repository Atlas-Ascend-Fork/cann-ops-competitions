# 需求背景（required）

## 需求来源

2026 年 10 月社区任务：aclblasGemmEx / aclblasSgemmEx / aclblasCgemmEx 算子开发（A2/A3）。
提交者：[FZ](https://gitcode.com/gcw_WnKFcIRe)。本文用于设计评审。
结构参照[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/a18791828054cc11674e1f64943e9e1beeb4a4d5/04_tasks/01_community-task-2026/resources/design_template.md)。

## 背景介绍

在 ops-blas 的 `blas/gemm_ex/arch22/` 增加三接口实现，保留共用公开 API。
GemmEx 为 19 参；SgemmEx 和 CgemmEx 为 17 参，不含 computeType/algo。
950 的 SgemmEx 同步适配公开签名，内部计算方案保持原实现。

# 需求分析（required）

## 需求描述

计算列主序 `C = alpha * op(A) * op(B) + beta * C`。
GemmEx/SgemmEx 的 A/B/C、alpha/beta 为 FLOAT32；CgemmEx 为 COMPLEX64。
支持 N/T/C、非紧凑前导维、零维及 alpha/beta 边界值，校验先于 no-op。

## 需求拆解

1. 三接口共享参数校验，区分非法枚举与合法但不支持的类型。
2. 用 Cube 实现主矩阵乘，用 AIV 实现紧凑化、复数拆分和 alpha/beta 收尾。
3. 保证连续调用、尾块、padding 和多 K 块正确；工作区的生命期覆盖异步执行。
4. 对实部、虚部分别执行任务精度合同；性能使用 msprof op 的 task duration。

# 详细设计（required）

## 算子分析

### 数学公式与数据类型

GemmEx/SgemmEx 仅支持三参同为 ACL_FLOAT；GemmEx 仅支持 ACLBLAS_COMPUTE_32F 和
ACLBLAS_GEMM_DEFAULT。CgemmEx 仅支持三参同为 ACL_COMPLEX64，复数交错排列为 real、imag。
三接口均不广播；逻辑形状为 A: m×k、B: k×n、C: m×n。转置输入的物理形状按 op 变换。

复数以四个实数乘积表示，`sA/sB = transA/transB 为 C ? -1 : 1`：

```text
P0 = op(Ar) op(Br)     P1 = op(Ai) op(Bi)
P2 = op(Ar) op(Bi)     P3 = op(Ai) op(Br)
Pr = P0 - sA*sB*P1     Pi = sB*P2 + sA*P3
C  = alpha*(Pr + i*Pi) + beta*C
```

## 算子实现

### 公共接口

```cpp
aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const void* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb,
    const void* beta, void* C, aclDataType Ctype, int ldc, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb,
    const float* beta, void* C, aclDataType Ctype, int ldc);

aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb,
    const aclblasComplex* beta, void* C, aclDataType Ctype, int ldc);
```

alpha/beta 为 Host 标量指针；A/B/C 为 Device 数据指针。

### Host 侧设计

校验顺序：handle → m/n/k → lda/ldb/ldc → alpha/beta → dtype → trans → A/B/C 指针 →
GemmEx 的 computeType/algo → no-op。合法不支持类型返回 NOT_SUPPORTED，非法 dtype 数值
返回 INVALID_VALUE，非法转置返回 INVALID_ENUM。

列主序内存直接视为转置矩阵的行主序布局，内部计算 `C^T = op(B)^T op(A)^T`。
输入的 ld 含 padding 时先紧凑化；复数拆分同时紧凑化。

实数输出 m、n 均不大于 64 时，采用 AIV 顺序 FP32 乘加。该区域不足一个 Cube 输出 tile；
沿递增 K 执行乘加使小输出矩阵的舍入与溢出行为更接近 Netlib 顺序实现。
N 分支先缩放 B、初始化 beta*C 后更新；T/C 分支先归约，再缩放 alpha 并加入 beta*C。
选择仅依赖形状，不依赖测试编号、输入内容或运行模式。该路径可能增加小矩阵的标量调度开销。

其余实数路径和复数的前缀乘积使用 Cube。实数 alpha=1、beta=0、ldc=m 时直接输出 C；
否则写紧凑乘积工作区，再收尾。复数 K≤2048 使用四个完整乘积平面；
K>2048 使用四个 K=2048 的前缀乘积，再由 AIV 按递增 K 合并每项复数乘积并累加后缀。
该选择仅依赖 K；没有按用例 ID、输入数值或测量模式分支。

工作区取自 handle：实数 `[packB][packA][acc]`，复数 `[Br][Bi][Ar][Ai][P0][P1][P2][P3]`。
长 K 复数额外分配 `[orderedReal][orderedImag]` 两个输出平面；前缀 P0..P3 始终只读，
避免 profiler 单 kernel 回放时再次累加到已覆盖的输入。收尾的 combinedAcc 标志区分四乘积与
已合并实虚部。各段按 32B 对齐，使用同一对齐平面跨度。禁止每次异步下发后立即释放工作区。

### Kernel 侧设计

| Kernel | 分块与同步 |
|---|---|
| Cube GEMM | 输出 tile 256×256，baseM/N/K=64/256/32，静态 tiling K 上限 8192，实际 K 每 2048 分块；禁用 HF32 |
| 实数顺序乘加 | 每个 AIV 核处理若干输出列；每次装载 K=128 的 UB panel，MulAddDst 沿 K 更新 |
| compact | 每行每段最多 16384 个 float，独立 DMA，按 srcLd 跳过 padding |
| complex split | 每行每段最多 4096 个复数；GatherMask 提取实虚部，尾部从 UB 标量读取 |
| complex ordered suffix | 每个 AIV 核处理 64×96 输出块，K panel=16；Gather 只读取一列 A 和每个 B 值的 8-lane 副本，Mul repeat 步长广播；逐 K 保持乘积、实虚合并及累加顺序 |
| real/complex epilogue | 逐列、最多 1024 行；仅搬运有效行，保留 C padding |

Cube 的 SetOrgShape 保留完整 K 作为 GM stride，SetSingleShape 使用当前 kChunk；
首个 K 块覆盖输出，其余块累加。缩短归约块不会改变底层矩阵行跨度。
实际归约块统一取 2048，以降低长 FP32 累加链的舍入误差。4096³ 原先相对 Netlib 的
max_abs_error=0.0100098；缩短归约链后为 0.0090332，阈值仍为 0.01。
2048 也保留短 K 的原计算顺序；统一 1024 曾使 K=2000、alpha=100 的 SgemmEx 用例从
maxAbsErr=0.375 增至0.578125，超过0.5。2048 分块恢复该用例至0.375，且4096用例通过。
这项修改在 probe13 联合定点通过，开发回归的阶段和限制见下文；不代表最终版本一次全量验收。

复数项内部保留独立乘积与加减，不使用融合乘加替代。实际 Netlib 溢出反例表明，
有限输入的独立乘积结果可能为 Inf，而 FMA 会得到有限值，违反有限/非有限合同。
alpha=(1,0) 收尾直接保留实虚部，避免 0×Inf 污染本应有限的另一分量；K=1/4096 的永久回归
覆盖此情形。beta=(1,0) 直接分别累加 C 实虚部，向量段与尾段均避免额外的 0×Inf；
m=1/33 与 K=1/4096 的回归检查非有限分量和 padding。上述分支只依据 API 标量参数，
不依赖用例 ID 或测量模式。

复数交错恢复的 Gather 结果在 MTE3 写回之前使用 PIPE_ALL 屏障。
标量尾处理始终在 UB 内进行，避免多核标量写 GM cache line 冲突或读到旧 GM 数据。
逐行 DMA 和有界 chunk 同时避免 padding 串入有效区与超出 UB。

alpha=0 或 k=0 时仅缩放 C，beta=0 不读旧 C，beta=1 不写 C；beta=0 且 C=nullptr 为成功 no-op。
这些路径仍先完成参数校验。

## 支持硬件

| 产品 | 实现/验证范围 |
|---|---|
| Atlas A2/A3（arch22） | 三接口实现；当前实机验证为 910B4-1、CANN 9.1.0-beta.1 |
| Ascend 950（arch35） | SgemmEx 公开签名适配，已编译；未在 950 硬件运行 |

## 算子约束限制

不增加广播、混合 dtype 或其他算法支持。接口为异步执行；调用方负责 stream 完成前保持输入、输出
和句柄有效。不把 910B4-1 的性能测量标记为任务书要求的 910B3 验收结果。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 判定 |
|---|---|
| 有限值精度 | atol=rtol=2^-13，matched_ratio≥0.99，max_abs_error≤max(1e-2,32ULP)；ULP 按用例 golden 最大量级计算并在报告注明 |
| 非有限值 | 同为 NaN、NaN/Inf、同号 Inf 等效；异号 Inf、有限/非有限不等效；单独统计且全部须通过 |
| alpha=0 | 有限值逐位比较 beta*C；非有限值按上述合同检查 |
| 性能 | 5 次预热，11 次有效样本；平均完整 API 的 task duration≤对应 GPU 基线/0.8 |

永久回归覆盖：严格非有限值比较器；三个入口的空 handle；复数同输入/变输入连续调用；
k=2048/2049/4097 的九种复数转置组合、跨输出 tile 边界、padding 和复数尾部；k=2047/2048/2049/8193 的四种实数转置组合；AIV/Cube 分界附近的实数连续调用。
临时八平面诊断已删除，日志保留在自测证据中。`GEMM_EX_DIAGNOSTICS=1` 可对实数及复数最大
误差元素调用 FP64 helper，辅助区分 NPU 与 Netlib 两侧舍入误差；不参与判定或计时。
每个性能 case 先验证输出、再预热和采样、最后复查输出与 padding，失败数据不计作性能成绩。

## 兼容性分析

SgemmEx 的公开签名按任务书统一为 17 参，旧调用点必须补齐 Atype/Btype/Ctype 并去掉 algo；
仓内 arch35 实现与测试已经适配并完成编译。CgemmEx 声明为全产品共用，当前仅 arch22 实现。
没有引入未合入社区 PR，也没有替换已验证容器的运行时或驱动。

## 当前验证结果与待完成事项

截至 2026-10-29，本地计算实现快照为 `024802a330b6f9a775a4aa855245623293d60458`。
本 PR 仅提交设计文档。以下为 910B4-1 / CANN 9.1.0-beta.1 / Netlib 3.10.0 的开发结果，
不能替代任务指定 910B3 / CANN 9.1.0 的正式验收。

| 项目 | 开发验证结果 | 边界 |
|---|---|---|
| GemmEx | 1206/1206 通过 | accuracy14 阶段；此后未改实数数值路径 |
| SgemmEx | 1201/1201 通过 | accuracy14 阶段；此后未改实数数值路径 |
| CgemmEx CSV | 1200 个用例均有通过证据 | 按 case ID 核对的分阶段覆盖，不是最终版本单次全量通过 |
| 最终定点回归 | 10/10 通过 | 9 个定点与 PF1057；采样前后均通过正确性检查 |
| 最终 beta=(1,0) 专项 | 274/274 通过 | 全部对应 CSV 用例 |
| arch35 SgemmEx | 库及测试目标编译通过 | 未做 950 实机验证 |

PF1057（m=n=1024、k=4096）采用 5 次预热、11 次有效 API 样本。
每个 API 按 split、4 个 GEMM、ordered suffix 和 epilogue 共 7 个 kernel 求和，
共核对 77 条 OpBasicInfo 记录；采样前后精度与 padding 检查通过。
完整 API 平均 task duration 为 **12603.056 us**，GPU 基线为 1906.66 us，
GPU/NPU 比值为 **0.151286**，低于要求的 0.8；对应 NPU 目标为 **2383.325 us**。
有序后缀平均 12034.161 us，约占总 kernel 时间的 95.5%，仍是主要性能瓶颈。

后续需在保持原精度及非有限值合同的前提下优化长 K 后缀，并完成最终版本三接口全量回归、
任务指定硬件上的正确性和全量性能验收。当前不声明性能达标或任务全部完成。
