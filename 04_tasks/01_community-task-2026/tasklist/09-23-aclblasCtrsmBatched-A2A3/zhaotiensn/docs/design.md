# 需求背景（required）

## 需求来源

本算子来源于 9 月社区任务《aclblasCtrsmBatched 算子开发（A2A3）》，任务书位于
`blas/9月社区任务-aclblasCtrsmBatched算子开发(A2A3)/aclblasCtrsmBatched_A2A3_task_doc.md`。
实现目标是将单精度复数批量三角求解能力加入 `ops-blas`，接口语义对齐 cuBLAS
`cublasCtrsmBatched`，并适配 Atlas A2/A3 的 arch22 架构。

## 背景介绍

### aclblasCtrsmBatched 算子实现目标

`aclblasCtrsmBatched` 对一批独立且尺寸一致的复数三角线性系统执行求解，输入矩阵
`B[i]` 原地覆写为解矩阵 `X[i]`。实现采用 ops-blas 的句柄式 BLAS 接口，通过 handle
绑定的 stream 异步下发 Ascend C MIX kernel。

算子实现路径：

```text
接口声明：include/cann_ops_blas.h
Host 实现：blas/trsmbatched/arch22/ctrsmbatched_host.cpp
Kernel 实现：blas/trsmbatched/arch22/ctrsmbatched_kernel.cpp
Tiling 数据：blas/trsmbatched/arch22/ctrsmbatched_tiling_data.h
```

### aclblasCtrsmBatched 参数与能力分析

| 参数 | 参数含义 | 数据类型 | 支持取值/约束 | 存储与形状 |
| --- | --- | --- | --- | --- |
| handle | ops-blas 上下文及 stream | `aclblasHandle_t` | 有效句柄；空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | Host |
| side | A 的作用方向 | enum | `LEFT` / `RIGHT` | Host |
| uplo | 三角部分 | enum | `UPPER` / `LOWER` | Host |
| trans | A 的变换 | enum | `N` / `T` / `C` | Host |
| diag | 对角线类型 | enum | `NON_UNIT` / `UNIT` | Host |
| m、n | B 的行数和列数 | `int` | 均须 `>= 0`；任一为 0 时 no-op | Host |
| alpha | 复数缩放因子 | `COMPLEX64` | Host 指针；空指针非法；零值不引用 A | Host |
| A | 批量三角矩阵指针数组 | `COMPLEX64` | 设备指针数组；LEFT 时为 m×m，RIGHT 时为 n×n | Device，列主序 |
| lda | A 的 leading dimension | `int` | LEFT：`lda >= max(1,m)`；RIGHT：`lda >= max(1,n)` | Host |
| B | 批量右端/解矩阵指针数组 | `COMPLEX64` | 设备指针数组，原地覆写；每个 B[i] 不得重叠 | Device，列主序 |
| ldb | B 的 leading dimension | `int` | `ldb >= max(1,m)` | Host |
| batchCount | 批量大小 | `int` | `>= 1` | Host |

计算公式：

```text
side=LEFT :  op(A[i]) * X[i] = alpha * B[i]
side=RIGHT:  X[i] * op(A[i]) = alpha * B[i]
```

其中 `op(A)` 由 `trans` 决定：`N` 为 A，`T` 为转置，`C` 为共轭转置；`uplo` 指定
引用的三角区域，`diag=UNIT` 时对角线视为 1 且不读取对角元素。

### aclblasCtrsmBatched 功能分析

- 支持 COMPLEX64（单精度复数）输入、输出和 alpha。
- 支持 LEFT/RIGHT、UPPER/LOWER、N/T/C、UNIT/NON_UNIT 的组合。
- 每个 batch 独立求解，A/B 为设备侧指针数组，矩阵按列主序存储。
- 支持 m/n 为 0 的合法 no-op 和 alpha 为零时的 B 原地清零路径。
- 不要求 broadcast、动态 rank 或非连续视图；m、n、batchCount 为运行时标量。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas A2/A3（arch22）上实现 `aclblasCtrsmBatched`，满足任务书
规定的 cuBLAS 兼容语义、参数校验、混合容差精度标准和 5 条性能门槛。接口通过
`aclblasHandle_t` 的 stream 绑定实现异步执行，B[i] 原地保存求解结果。

## 需求拆解

1. 实现 LEFT/RIGHT 两种求解方向及 N/T/C 变换，保证 batch 间地址和数据互不混淆。
2. 实现 UPPER/LOWER、UNIT/NON_UNIT、alpha 缩放、零维 no-op 和 alpha=0 特殊路径。
3. 在 Host 侧完成句柄、枚举、维度、leading dimension、指针数组和 batchCount 校验。
4. 使用 Ascend C MIX_AIC_1_2 方案，将 AIV 三角求解与 AIC 复数 GEMM 尾块更新流水化。
5. 通过 panel、双 AIV、列拆分、批次分组和 block-inverse 快路径覆盖不同规模和 batch 形态。
6. 精度满足 COMPLEX64 混合容差：`rtol=atol=2^-13`、`matched_ratio>=0.99`，且
   `max_abs_error<=max(1e-2,32*ULP)`。
7. 性能满足任务书 5 条门槛；每个 case 先 warm-up，再以不少于 10 次有效样本求平均。
8. 提供 CSV 驱动 GTest、cblas golden、测试 README、自测日志和 profiler 原始证据。

# 详细设计（required）

## 算子分析

### 数学公式

对每个 batch `i`：

```text
LEFT :  op(A[i]) * X[i] = alpha * B[i]
RIGHT:  X[i] * op(A[i]) = alpha * B[i]
```

矩阵为列主序布局。为统一 kernel 的行式处理，Host 将列主序问题利用“列主序矩阵与其
转置的行主序存储逐位一致”的性质转换为等价的行主序问题。转换关系为：

```text
side 翻转，m/n 互换，uplo 翻转，trans 和 diag 保持不变，A/B 地址透传。
```

### 支持数据类型

- A、B、alpha：COMPLEX64，即交错存储的两个 FLOAT32 分量。
- Kernel 内部将复数乘法编码为实数 2×2 block 运算，AIC 使用 FLOAT32 Cube GEMM。
- Host 对 alpha 非 `(1,0)` 的输入先将 B 逐 batch 预乘，kernel 主路径固定使用 alpha=1。

### 支持形状

- `m,n >= 0`；`m=0` 或 `n=0` 时直接返回成功，不下发计算 kernel。
- LEFT：A[i] 为 m×m，B[i] 为 m×n；RIGHT：A[i] 为 n×n，B[i] 为 m×n。
- `lda`、`ldb` 遵循 BLAS leading dimension 约束；矩阵实际有效区域按列主序访问。
- batchCount 支持运行时批量大小；大 batch 在 Host 侧按组下发以控制 workspace 峰值。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

##### 1. 参数校验与特殊路径

Host 入口 `aclblasCtrsmBatched` 首先校验 handle、枚举、m/n、alpha、A/B、lda/ldb 和
batchCount。非法参数返回任务书规定的状态码。m/n 为零时返回成功；alpha 为零时不读取
A，按 batch 将 B 的有效 m×n 区域清零。

##### 2. alpha 和数据布局处理

alpha 不为 `(1,0)` 时，Host 在设备数据拷回/映射后逐 batch 对 B 预乘 alpha，kernel
统一执行 alpha=1 的求解路径。根据 side/trans 计算 `needTA` 和 `conjA`，必要时将 A
规范化为交错 FLOAT32 工作区；B 在 kernel 内完成 AoS/SoA 解交织、对齐补齐和必要转置。

##### 3. 分核与 tiling 策略

Host 计算有效三角维 `kDim`、右端列数 `nCols` 及其 8-float 对齐尺寸，并根据平台查询
得到的 AIC 核数设置 `numSplits`、`dualAivMode`、`nb` 和 kernel block 数：

- `nb=16` 用于 `kDim<=1024`，`nb=32` 用于更大三角矩阵。
- `kDim>=128` 且 `nColsAligned>=128` 时优先启用双 AIV；小 batch 时按列拆分以铺满 AIC。
- 非 dual、forward、偶数 batch 且规模足够大时启用 block-inverse 快路径。
- batch 按组下发，每组下发前同步 stream，避免复用共享 tiling/workspace 时产生数据竞争。
- workspace 按 A 工作区、B/GEMM 工作区、Xneg 双缓冲区和系统 workspace 对齐分配；通过
  分组控制单组占用，规避单组 2GB 限制。

##### 4. 列主序到 kernel 内部布局映射

Host 在 `FillCtrsmTilingBasicParams` 中保存原始 m/n、lda/ldb 和变换标志，并将转置后的
side/uplo/m/n 写入 tiling。A/B 指针数组仍保持 batch 顺序透传，kernel 统一按映射后的
行式三角系统执行。

##### 5. kernel 下发

Host 生成 Cube matmul tiling，拷贝指针数组、tiling 和 Cube 配置到设备 workspace，随后调用
`ctrsmbatched_mix12_kernel_do`。每个 batch group 使用独立的 `batchStart`，kernel 根据
`numBlocks`、`numSplits` 和 `dualAivMode` 访问对应的 A/B 地址。

#### 3.2.2 kernel 侧设计

Kernel 使用 `KERNEL_TYPE_MIX_AIC_1_2`，由 1 个 AIC 和 2 个 AIV 协同工作。

##### 1. AIV panel 三角求解

三角矩阵沿求解方向切分为 panel。AIV 将对角 panel 搬入 UB，完成复数前代/回代、对角
复数除法和尾行 Axpy，并把解写回工作区，同时生成供 AIC 使用的 `Xneg` 交错平面。
双 AIV 模式按列范围拆分任务；`FLAG_TRSV` 用于通知 AIC 当前 panel 已完成。

##### 2. AIC 尾块更新

AIC 对当前 panel 以下的尾部执行：

```text
B_trail -= A_trail * X_panel
```

复数乘法拆成实数 block 矩阵的两次 GEMM，通过 Cube tiling 执行。`FLAG_GEMM` 通知 AIV
尾块更新完成；`Xneg` 按 16 个 panel 分组双缓冲，支持跨组流水。

##### 3. block-inverse 快路径

对非 dual、非 split、forward、偶数 batch 且 `kDim,nCols>=128` 的场景：

1. AIV 预计算每个 16×16 对角块的逆，并编码为 `[[Re,-Im],[Im,Re]]`。
2. AIC 对每个 panel 执行 `Y=M_apply·B_panel`，再执行尾部更新。
3. 通过 FIX/MTE2 事件保证 apply 的输出先于 trailing GEMM 读取。
4. AIV 回写与 AIC 下一 panel 计算重叠；两个 16 行块合并一次转置回写，降低 PIPE_ALL 次数。

该路径将小 panel 的 AIV 复数除法和固定开销转移到 Cube，主要用于任务书 case1 家族，
其余规模仍使用通用 panel 求解路径。

##### 4. 数据转换与回写

kernel 使用专用 gather 偏移表完成 AoS 到实部/虚部平面的解交织，按照 `needTA`、`conjA`
和 padding 状态构造有效 A/B 工作区；求解完成后将结果重新交织并按原始 ldb 写回 B。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas 800T A2 / Atlas 800I A2 | √ | 当前实机为 Atlas 800T A2（910B3），arch22 |
| Atlas A3 训练/推理系列（含 Atlas 800I A3） | √ | 使用同一 arch22 实现；本次未进行 A3 实机采样 |
| Ascend 950PR / Ascend 950DT | × | 复数批量三角求解仅提供 arch22 实现 |

## 算子约束限制

- 不支持 broadcast；A[i]、B[i] 为独立矩阵，所有 batch 共享 m/n、lda/ldb 和枚举属性。
- A、B 为设备侧指针数组，alpha 为 Host 指针；各 B[i] 不得互相重叠。
- `diag=UNIT` 时对角元素视为 1，不读取其存储值；`NON_UNIT` 不做奇异性检测。
- 不支持超出 lda/ldb 语义的非连续视图，不要求动态 rank 或确定性逐位一致。
- 异步语义依赖 `aclblasSetStream`；读取设备结果前必须同步对应 stream。
- alpha 为零时 A 可以为空但 B 仍必须有效；普通非零 alpha 路径要求 A/B 数组有效。
- 大 batch 通过 Host 分组下发规避单组 workspace 过大，但整个调用仍保持原始 batch 顺序。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 功能标准 | 对每个 batch 独立求解，结果原地写回 B；支持 LEFT/RIGHT、UPPER/LOWER、N/T/C、UNIT/NON_UNIT | 任务书 §2.1、§2.3 |
| 参数标准 | 非法 handle、枚举、维度、leading dimension、空指针和 batchCount 返回约定状态码；零维和 alpha=0 按约定处理 | 任务书 §2.4、§2.5 |
| 精度标准 | `rtol=atol=2^-13`，`matched_ratio>=0.99`，`max_abs_error<=max(1e-2,32*ULP)`；cblas 逐 batch golden | 任务书 §3.2、生态混合容差标准 |
| 性能标准 | 5 个 COMPLEX64 门槛 case 的 Avg time 不高于 895.1/5551/13884/21816/40969 us；有效采样不少于 10 次 | 任务书 §3.3 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

## 兼容性分析

- 参数顺序、side/uplo/trans/diag 语义、列主序布局和 B 原地覆写语义与 cuBLAS
  `cublasCtrsmBatched` 对齐。
- 对外接口声明放在 `include/cann_ops_blas.h`，使用 ops-blas 的 `aclblasHandle_t`、状态码
  和 stream 机制；不改变已有 `aclblasStrsmBatched` 接口。
- 代码按 arch22 独立实现，A2/A3 共用同一 Host/kernel/tiling 数据结构；Ascend 950 系列不提供
  该复数版本实现。
- 测试工程位于 `test/trsmbatched/ctrsmbatched/`，采用 CSV 驱动 GTest 和 Netlib cblas golden，
  可复现边界、特殊值、负向和性能 case。
