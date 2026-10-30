# 需求背景（required）

## 需求来源

[10月社区任务-aclsparseCgtsv2算子开发（950）](https://www.hiascend.com/activities/task-center/details/3c8d06bce4f54557b895de27e3737e0f)。实现提交至 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse) 的 `master` 分支，采用 aclsparse Legacy C++ API 与 Ascend C Kernel 直调工程。接口能力对标任务书指定的 CUDA Toolkit / cuSPARSE 13.3 Update 1 `cusparseCgtsv2`。

## 背景介绍

`aclsparseCgtsv2` 求解复数三对角线性方程组 `AX=B`，一组系数对应一个或多个右端项，结果原地覆盖 B。系数矩阵不要求对角占优，因此需要部分主元选取及相邻行交换，不能仅提供无主元 Thomas 求解。

现有 `sparse/gtsv2/arch35/` 的 Sgtsv2 提供实数 SIMT 实现，可参考其 Host、stream 和 Kernel 调用方式。该实现每个线程处理一个 RHS，并重复进行系数消去。本任务新增独立 complex64 类型及 ABI，结合共享分解、多 RHS 并行求解与有界检查点重放，保留现有实数接口。此接口无对应 TBE 迁移实现，不使用 aclnn 工程模式。

# 需求分析（required）

## 需求描述

使用 C++ 和 Ascend C 实现 `aclsparseCgtsv2_bufferSizeExt`、`aclsparseCgtsv2`，面向 Ascend 950、DAV_3510、`arch35`。核心分解和求解在 NPU 完成，执行使用 handle 绑定的 stream。

## 需求拆解

1. 提供独立复数类型、workspace 查询、参数校验和异步执行接口。
2. 通过复数模长比较选取主元，维护行交换、消去乘子和第二上对角填充；工作区允许时复用共享分解，否则分块重放主元递推。
3. 保持系数只读、B 原地输出及 padding 不变，支持动态尺寸、空 RHS、确定性复跑和异常边界。
4. 按指定官方 ATK 验证数值与合同，逐条验证 203 条性能及内存要求，并回归与 A2/A3 共用的接口和 Host 代码。

# 详细设计（required）

## 算子分析

### 数学公式与布局

采用零基下标，`A[i,i]=d[i]`、`A[i,i-1]=dl[i]`、`A[i,i+1]=du[i]`，边界 `dl[0]=du[m-1]=0+0i`。每个 RHS 满足：

\[
dl_iX_{i-1,j}+d_iX_{i,j}+du_iX_{i+1,j}=B_{i,j},\qquad 0\le i<m,\quad 0\le j<n.
\]

首末行省略越界项。`dl/d/du` 是长度 m 的连续 complex64 数组；B 是列主序 `[ldb,n]`，地址为 `B[i+j*ldb]`。只读写前 m 行，`m<=i<ldb` 的 padding 保持不变。n 表示 RHS 数量；没有 batchCount、广播、转置或任意 stride 入口。

### 算子原型

在 `include/cann_ops_sparse.h` 中新增一次以下类型和接口：

```cpp
typedef struct {
    float real;
    float imag;
} aclsparseComplex_t;

static_assert(sizeof(aclsparseComplex_t) == 8);
static_assert(alignof(aclsparseComplex_t) == alignof(float));

aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B,
    int ldb, size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B,
    int ldb, void *pBuffer);
```

| 参数 | 契约 |
| --- | --- |
| handle | 有效的 aclsparse 句柄，执行采用其绑定 stream。 |
| m、n、ldb | 运行时 int32；`m>=3`、`n>=0`、`ldb>=max(1,m)`。 |
| dl、d、du | NPU 只读 complex64 连续输入，均为 `[m]`。 |
| B | NPU complex64 列主序 `[ldb,n]` 输入输出，有效区原地覆盖为 X。 |
| pBufferSizeInBytes | 非空 Host size_t 输出指针，接收 workspace 字节数。 |
| pBuffer | NPU 临时工作区，容量不小于查询值，首地址按 128 字节对齐。 |

`n=0` 时仍校验 handle、m/n/ldb 和查询输出指针；查询返回 0，执行不启动求解 Kernel，允许四个数据指针和 workspace 为空。`n>0` 时四个数据指针均非空，执行还检查 workspace 非空与对齐。空 handle 返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`，非法整数、空必需指针、未对齐 workspace 或字节数溢出返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；运行时错误沿用仓内状态码映射。

raw C ABI 不携带 dtype、分配长度或 stride 元数据，不能检测这些不可见条件；调用者及适配层负责类型、布局、长度及边界系数。执行接口没有 workspace 容量参数，调用者须足量分配。所有设备存储保持有效直至 stream 完成；使用同一 B 或 workspace 的并发调用由调用者串行化。

## 算子实现

### Host 侧设计

查询和执行共用整数校验、溢出检查与 workspace 布局计算，使用 size_t/int64_t 检查尺寸乘法、地址跨度和向上对齐。查询不读取设备系数或 B。执行取得 stream 后按序下发分解、求解及必要的数值修正，不增加 Host 等待或 D2H 数值计算。

中间复数的实部、虚部分别保存独立 int64 指数，指数累加、乘除、范数平方及对齐差值均使用64位；公开尺寸参数仍为int32。最终收窄先判断FLOAT32的上下溢范围。工作区允许时，共享区保存 U 的三条对角线、消去乘子和 pivot，每个 RHS 槽保存解与修正量；原始 B 保留至该列修正及范围检查结束。对于收窄边界需要更高精度的列，保留原 B，转交四项 FP32 展开重算。

工作区按目标设备 L2 容量限制。设 `A8=align128(8m)`、`A4=align128(4m)`，普通共享因子区为 `S=16*A8+A4`，每个普通 RHS 槽为 `L=6*A8`。每个检查点槽为

\[
C(m,k)=128+\operatorname{align128}(152\lceil m/k\rceil)
+\operatorname{align128}(192\min(m,k))+1536,
\]

其中 k 为块宽，包含块首状态、一个重放窗口、算术状态及控制字。启用 PCR 时另预留 `P=(4+2*n)*A8+align128(4*(4*p+1))`，p 为 PCR 核数；未启用时 P 为零。若 `S+L+C+P` 超过预算，取消整列因子和普通 RHS 槽，只使用检查点，令 S、L 为零。并行槽数 q 同时受 n、AIV 核数和剩余容量约束，保证查询总量 `S+q*(L+C)+P` 不超过目标设备的128 MiB L2。块宽根据尺寸和窗口/检查点容量选择，不依赖用例编号。

查询与执行共享布局。RHS 按 q 列分波处理；保存、重放与回代使用同一 stream 顺序下发，不在 Host 读取设备标志，也不增加隐藏分配。实际对齐分配与框架内存仍按任务口径独立记录。

### Kernel 侧设计

1. **带主元分解。** 共享分解和检查点重放采用相同的相邻主元递推，在内部存储中维护当前对角、上对角、第二上对角、乘子和 pivot。每步比较当前主对角与下一行下对角的欧氏模长，等模时保留当前行；需要交换时记录相邻行置换和第二上对角填充。平方模的乘积与求和携带独立指数，避免直接对原始幅度平方导致溢出。三条输入系数始终只读。
2. **多 RHS 求解。** 工作区允许时，不同 AIV/SIMT 执行单元领取不同列并复用共享因子。前代严格按 pivot 顺序交换 RHS 并消去；回代使用

   \[
   X_{i,j}=\frac{Y_{i,j}-U_{i,i+1}X_{i+1,j}-U_{i,i+2}X_{i+2,j}}{U_{i,i}},
   \]

   其中越界项省略。列主序下连续行访问合并搬运，寄存器保留相邻递推状态；SIMT 处理分支和复数标量运算，SIMD/UB 可用于批量搬运、残差和实虚分量计算。跨 Kernel 依赖使用同一 stream 排序，核内搬运与计算显式同步。
3. **数值处理。** 复数运算在 FP32 实虚分量上展开，除法的共轭乘积与平方模以带指数的实数乘加求值，避免中间乘积溢出。极小主元或大中间解不能提前收窄为零或 Inf；范围保护表示以 FP32 尾数和独立指数分别保存因子、解及修正量的实部和虚部，避免共同归一化丢失较小分量。因子的每个分量用高低两个 FP32 数保留递推中的舍入残差，分解和使用因子求解时保留低位；解向量仍使用单 FP32 尾数与独立指数。用原始 A/B 计算补偿残差 `R=B-AX`，复用因子求解 `AΔX=R` 并更新 X，达到停止条件后按 RNE 写回 complex64。最终值接近FLOAT32溢出阈值时，在写回前改用四项FP32展开重算，按任务Golden收窄规则区分有限值与Inf。修正和提高内部精度均不是病态系统精度的先验保证，仍按规定Golden和逐分量指标验证。
4. **有界检查点重放。** 大系统不保存完整因子：前向分段执行主元递推，保存各块首的递推状态和原始 RHS。随后逆序重放一个块并回代，再复用窗口；跨块的两个后续解保持宽指数四项表示，最终输出才收窄。相邻块已被输出覆盖的首行 RHS 从保存的原值读取，保证原地执行正确。多 RHS 使用独立槽并按波次复用，避免总工作区随 n 无界增长。
5. **并行优化与边界。** 对适合行并行的尺寸，先通过并行循环约减（PCR）生成候选；仅当系数满足严格对角占优及范围条件，且全部 RHS 的原方程补偿残差通过误差界检查时接受。检查在 NPU 上完成，候选与原 B 分开保存，任一条件不满足即从原输入进入完整带主元路径。多 RHS 复用分解，长系统采用有界分块重放。算法选择只依赖输入属性和数学条件，不按用例编号分派。尾块和非对齐列跨度只更新有效元素，避免覆盖相邻列或 padding；不使用输出累加原子操作。

真正奇异系统无独立 info 输出。接口成功表示异步提交成功，不保证系统非奇异；自测分别记录接口返回、stream 状态及输出，验证不崩溃、不越界。Inf/NaN 通过独立特殊值套件记录分类与位置行为，不用于普通精度通过。

### 工程组织

| 位置 | 内容 |
| --- | --- |
| `include/cann_ops_sparse.h` | 公开复数类型和两个独立 ABI。 |
| `sparse/cgtsv2/` | 公共尺寸、布局工具和接口 README。 |
| `sparse/cgtsv2/arch35/` | Host 调度、Ascend C Kernel 和 launch 声明。 |
| `sparse/cgtsv2/acceptance/` | 独立验收辅助：任务用例、ATK 配置、Python 适配及验收说明；验收后提交上游实现 PR 前移除。 |
| `test/cgtsv2/` | 按仓内构建机制组织的正式测试。 |
| `test/cgtsv2/arch35/` | C++ 接口、Host/Kernel UT/ST。 |

沿用仓内 OP_LIST、SOC 架构过滤与测试构建机制；不创建重复 build 目录。检查点纯SIMT与既有混合模式核分别生成共享库，沿同一标准安装入口安装到同一lib64目录，主库使用相对依赖路径；调用者继续使用原公开接口。公共层不引用 950 专属 Device 头文件。A2/A3 与 A5 合入顺序变化时复用已合入公共类型和 Host 代码，消除重复定义并执行交叉回归。

## 支持硬件与约束限制

支持 Ascend 950（DAV_3510、arch35、A5），采用 CANN 9.0.0 及后续配套版本。Python/ATen 适配使用 PyTorch 2.7+、torch_npu 26.0.0+ 的配套组合；生产接口无 CPU fallback。实际硬件、驱动、固件、CANN 及编译选项随自测报告记录，不以架构名称代替上板验证。

# 可维可测分析

## 精度标准与测试方法

CPU Golden 将同一份已量化 complex64 输入提升为 complex128，以 SciPy/LAPACK 带主元带状求解。实部、虚部分别满足 `|actual-golden|<=2^-13+2^-13*|golden|`，两部分匹配率均不低于 0.99。各分量最大有限绝对误差同时不超过 `max(1e-2,32*ULP(g_low))`，g_low 取该分量最大误差点 Golden 收窄为 float32 后的值。

常规精度中 NaN 不记通过；NPU 为 Inf 而 Golden 有限时，仅 Golden 按 RNE 收窄到 float32 后为同号 Inf 才通过溢出特例。独立特殊值套件逐位置核对 NaN/Inf mask 和 Inf 符号，其余有限分量逐点满足混合容差，不使用 0.99 匹配率或最大误差 ULP 上限。

采用官方 ATK 26.9.8（提交 `9504bee6076820ca74259fb919a790a12ef88cf8`）、Python 3.10/3.11、NumPy<2，沿用随包 Tensor/Attr YAML、约束插件及原生 `atk case`、`atk task` 命令。随包 200 条和验收另生成的 800 条精度用例独立运行；四输入各自保持 50% 均匀、50% 正态分布并正交组合，不以对角占优输入替代。

完整有限因子域另做覆盖审计：包含 `[1,2^20]` 内合法幂次、幂次减一、padding 与分布组合，按 `ldb<=2^20`、总元素 `<=2^34` 过滤。200+800 不替代全域审计；超出单机资源的合法组合明确记录需求与未执行状态。

C++ UT/ST 覆盖类型布局、m/n/ldb 边界、空 RHS、单/多 RHS、连续交换、零主对角但非奇异、病态及真正奇异、实虚极大幅度差和单分量输出溢出、Inf/NaN、非法参数、workspace 对齐与保护区、整数溢出、非默认 stream、只读系数、B 原地输出与 padding，以及查询/分配/执行/释放全流程。另用非奇异下双对角矩阵和独立解析解覆盖大行数下超出int32范围的中间指数差，检查64位指数链；不将生成用例的2^20边界当作公开ABI的m上限。检查点路径补充跨块、分段保存、列槽复用和大尺寸预算边界，验证原地覆盖后仍使用正确的原始RHS。非奇异输入恢复原 B 后在同设备、软件与配置下复跑，核对确定性。适配层拒绝不代替真实 C ABI 证据。

## 性能与内存标准

全部 203 条固定性能用例逐条满足 `cuSPARSE Kernel总耗时中位数 / NPU Kernel总耗时中位数 >= 0.3`。P-01、P-02、P-03 的 `(m,n,ldb)` 为 `(8192,64,8192)`、`(4096,64,4096)`、`(7168,128,7168)`，GPU 基线分别为 640.896、628.944、1220.048 μs；其余 200 条采用附件 CSV，不以均值掩盖单例未达标。

正式验收每例预热至少 10 次、采样至少 30 次。GPU 用 NCU，NPU 用官方 Profiler/msopprof，合计一次求解内全部 Kernel，包含分解、求解和修正。每次恢复原始 B 后才计时，排除恢复、数据准备、首次编译和拷贝；workspace 复用。同一 stream Event 另行包围求解并单列结果，不参与 Kernel 倍率。`n=0` 不定义倍率。开发阶段先运行受影响的最小用例，形成候选后扩大正式验证。

两种采集口径均报告内存：B 仅计一次，输入输出共 `(3*m+ldb*n)*8` 字节；workspace 查询值、128 字节对齐实际分配、恢复 B 的夹具副本分别列示。记录设备 free/total；使用框架 allocator 时另列 allocated/reserved 基线、峰值和增量，注明单位与同步点。内存满足任务书两条件之一：输入输出超过 500 MB 时达到规定的 GPU/NPU 额外内存比例，或固有 workspace 不超过实际目标设备 L2 Cache 容量。设备全局 free/total 不直接归因于本算子。

保留源码版本、环境、种子、ATK 原生结果、覆盖审计、确定性复跑及 NPU Dispatch/Profiler 证据，分别报告功能、精度、性能、内存和交叉回归。

## 兼容性分析

新增 complex64 类型及 Cgtsv2 符号，保持 Sgtsv2 类型、签名和行为。公共类型单一定义，设备代码按架构分离；回归原有实数接口和受影响的公共 Host 路径，不引入隐式全局同步。
