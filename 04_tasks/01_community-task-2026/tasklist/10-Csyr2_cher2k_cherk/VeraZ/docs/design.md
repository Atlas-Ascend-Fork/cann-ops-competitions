# 需求背景

## 需求来源

社区任务：aclblasCsyr2/aclblasCher2k/aclblasCherk精度修复-A2/A3。任务书为 `aclblasCsyr2_Cher2k_Cherk精度修复_A2A3_task_doc.md`，实现仓库为 https://gitcode.com/cann/ops-blas 。

## 背景介绍

三个复数 BLAS 算子已经在 Atlas A2/A3 的 arch22 实现。现有实现更新 `uplo` 指定的三角区域时可能覆盖未引用的另一三角及前导维 padding。本任务修复写回边界，同时保持既有公开接口、计算语义、异步行为与性能。三个算子作为一个任务交付。

# 需求分析

## 需求描述

对原地输出矩阵 M（Csyr2 为 A，Cher2k/Cherk 为 C），按列主序寻址 `M(i,j)=M[i+j*ld]`，`0≤j<n`。UPPER 的操作区为 `0≤i≤j`，LOWER 为 `j≤i<n`，均包含对角线。未选中三角及每列 `n≤i<ld` 的 padding 是非操作区；这些位置的实部、虚部必须与调用前逐 bit 一致，Kernel 不得对其写入。quick return 时整个缓冲区保持不变。

## 需求拆解

1. 限定在 arch22 修复三算子，沿用现有 API、参数顺序、枚举、返回码和标量内存归属；不新增平行接口。
2. 操作区满足相应 BLAS 公式及混合容差；未选中三角和 padding 零写，不以先写满再恢复或补全共轭矩阵代替。
3. 支持步长、前导维、边界、异常和 quick return；Hermitian 算子正常计算时对角虚部置零。
4. 保持 handle 绑定 stream 的异步执行，不回退 CPU；A2/A3 均完成精度和 bit 检查，A2 满足任务书性能标杆。

# 详细设计

## 算子分析

### 数学公式

Csyr2：`A ← A + αxyᵀ + αyxᵀ`，为普通转置，不共轭；对角虚部参与普通计算。向量逻辑元素按 `start+i*inc` 访问，`inc<0` 时零基起点为 `(1-n)*inc`。

Cher2k：`trans=N` 时 `C ← αABᴴ + conj(α)BAᴴ + βC`；`trans=C` 时 `C ← αAᴴB + conj(α)BᴴA + βC`。α 为复数、β 为实数。

Cherk：`trans=N` 时 `C ← αAAᴴ + βC`；`trans=C` 时 `C ← αAᴴA + βC`。α、β 为实数。两类 Hermitian 更新只引用所选三角，正常计算对角虚部置零；不向另一侧写共轭值。

### 支持数据类型和形状

矩阵、向量及 Csyr2/Cher2k 的复数标量为 complex64（实、虚部各 float32）；Cherk 的 α、β 及 Cher2k 的 β 为 float32。矩阵为列主序，逻辑阶数 n，物理输出为 `ld×n`；N 时输入矩阵逻辑形状为 `n×k`，C 时为 `k×n`。支持 `n,k≥0`、两种 uplo 及 Hermitian 算子的 N/C 操作；Csyr2 支持非零正负 incx/incy。输入和输出的前导维依任务书约束。

## 算子实现

### 实现方案

#### Host 侧设计

复用 ops-blas 现有句柄式入口和校验：检查 handle、uplo、n、步长、前导维及必要指针；Cher2k/Cherk 检查 k 和 trans，合法但不支持的 OP_T 返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`。实际校验顺序和返回码按现有接口及任务书保持一致。Csyr2 的 α 在 Host；Cher2k/Cherk 的 α、β 在 Device，禁止为判断标量而增加 Host 同步或改变地址归属。

Tiling 传递 n、k、ld、uplo、trans、步长及分块信息。按输出三角有效工作量分配核和 tile，同一输出元素只归属一个核；划分时考虑 UB 容量、计算暂存和硬件对齐。对于小规模减少启动与分块开销；对于大规模保留适用的向量/Cube 计算和流水，以性能测试决定块大小。不能因为 tile 尾部对齐把写回扩大到非操作区。

#### Kernel 侧设计

1. 对每个 tile 根据其行列范围分类：完全位于非操作区则跳过；完全位于操作区则可采用经过边界证明的批量搬运和计算；跨对角线或矩阵尾部的 tile 走裁剪写回路径。
2. 对每列 j，将有效行裁剪到 UPPER 的 `[0,j+1)` 或 LOWER 的 `[j,n)`，并与当前 tile 范围求交；只计算和提交交集。列内地址使用 `j*ld+i`，不把 `n` 之后的 padding 纳入写回。操作区输入 C 仅在 β 非零时读取；β 为零时不依赖其原值。输入向量和矩阵按各自步长及前导维加载。
3. 对边界采用可在目标 A2/A3 上证明只写指定元素的写回方式；若常规块搬运有对齐扩写，改用支持精确写入的方式或重新划分边界。不能通过读取、重写未选中三角来消除覆盖。核间写入范围必须互斥；确认指令写入粒度不会因相邻核产生隐式覆盖。这些硬件写回能力和实际路径须在实现与测试时验证。
4. 复数运算按上述公式执行；Hermitian 对角只使用旧值实部，并在非 quick return 的输出中将虚部置零。Csyr2 对角不作此处理。
5. Csyr2 的 `n=0` 或 `α=(0,0)` quick return；Cher2k/Cherk 的 `n=0` 或 `(α=0 或 k=0) 且 β=1` quick return，所有输出 bit 不变。设备标量相关条件在设备侧处理，避免 Host 同步。`α=0` 或 `k=0` 且 β 不为 1 时仅对操作区执行 β 缩放和 Hermitian 对角处理。

#### 修改范围及资源

修改 `blas/syr2/arch22/` 和 `blas/herk/arch22/` 内必要的 Tiling、Kernel 与写回实现；测试分别置于 `test/syr2/csyr2/arch22/`、`test/herk/cher2k/arch22/`、`test/herk/cherk/arch22/`。不修改公开头文件签名；不改 arch35，除非共享 Host 校验确有必要且验证不影响 950。中间存储随 tile 和问题规模规划，不申请与 n/k 无关的持久 Device 缓存。

## 支持硬件

| 芯片 | 支持范围 |
| --- | --- |
| Atlas A2 | arch22，功能、精度和性能验证；性能主门禁为 Atlas 800I A2 910B3 |
| Atlas A3 | arch22，功能、精度和性能跑通并记录 |

## 算子约束限制

仅支持任务书定义的步长和前导维，不支持额外广播或高维 ND 运算。Csyr2 的 `lda≥max(1,n)`；Cher2k/Cherk 的 N 模式 `lda/ldb≥max(1,n)`、C 模式 `lda/ldb≥max(1,k)`，输出 `ldc≥max(1,n)`。Cherk 不使用 B/ldb。quick return 不强行修改对角虚部；Host 侧不做流同步，调用方在读回前同步。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 操作区精度 | complex64 实虚分量分别按 float32 比较：rtol=atol=2^-13，匹配率≥0.99，最大绝对误差≤max(1e-2, 32×ULP(g_low))；Inf/NaN 按标准规则 | https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md |
| 非操作区 | 每个实虚分量逐 bit 比较，包含未选中三角及 padding，匹配率 100%；覆盖 NaN payload、±0、±Inf、非规范数 | 任务书 §2.0、§3.2 |
| 性能 | A2 910B3 预热后有效采样超过 50 次，平均耗时不高于下表及任务包已有性能用例标杆；A3 跑通并记录 | 任务书 §3.3 |

| 算子 | uplo / trans | n / k | 上限（μs） |
| --- | --- | --- | --- |
| Csyr2 | UPPER | 512 / — | 7.54 |
| Csyr2 | LOWER | 1024 / — | 9.72 |
| Csyr2 | UPPER | 2048 / — | 19.05 |
| Cher2k | UPPER / N | 1024 / 1024 | 1458.91 |
| Cher2k | UPPER / N | 2048 / 2048 | 5240.68 |
| Cher2k | LOWER / C | 1024 / 1024 | 1081.90 |
| Cherk | UPPER / N | 1024 / 1024 | 738.09 |
| Cherk | UPPER / N | 2048 / 2048 | 2616.28 |
| Cherk | LOWER / C | 1024 / 1024 | 485.94 |

Csyr2 性能行的 α=(1,0)、incx=incy=1、lda=n；其他性能参数按任务书和已有基线配置。测试使用 CANN 9.1.0 和 Netlib BLAS 3.10.0：Cher2k/Cherk 用 CBLAS 生成 golden，Csyr2 按 Netlib ssyr2 语义自实现复数参考。报告记录 CANN、驱动、设备及 Netlib 版本。

复用任务包三套 CSV 和脚本，不删减原有功能、精度、性能、负向用例；将 `check_untouched_bitwise.h` 接入对应 GTest，对每个成功用例在调用前后比较整个物理缓冲的非操作区。追加 `TC_BIT_` 哨兵用例：Csyr2 22 条、Cher2k/Cherk 各 24 条；覆盖两种 uplo、紧凑及 `ld=n+8/n+32`、Hermitian N/C、非对齐和对角边界，含不同 NaN payload、有限值、±Inf、±0 和非规范数。另测零规模、标量 quick return、β=0、负步长、非法枚举/参数、跨 tile 边界及特殊值。测试前快照 `ld×n`，调用并同步后 D2H，操作区按混合容差、非操作区按 memcmp 检查。

## 兼容性分析

沿用公开 API、数据类型、参数位置、返回码、流语义和已有运算公式；修复只改变原本不应被覆盖的内存行为。三个算子的 A2/A3 功能、bit 一致性与性能结果均须实测，当前设计不代表测试已通过。写回指令粒度、不同对齐条件下的核间互斥、性能及与现有实现的兼容性均为实施阶段验证项。
