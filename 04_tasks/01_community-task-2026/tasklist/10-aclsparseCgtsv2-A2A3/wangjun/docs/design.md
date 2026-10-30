# 需求背景（required）

## 需求来源

[10月社区任务-aclsparseCgtsv2算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/b8289b3a952d4e13a9e79c3a242a8338)。实现提交至 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse) 的 `master` 分支，采用 Legacy C++ API 与 Ascend C Kernel 直调工程。接口能力对标任务书指定的 cuSPARSE `cusparseCgtsv2`。

## 背景介绍

`aclsparseCgtsv2` 求解复数三对角线性方程组 `AX=B`。一组系数对应一个或多个右端项，求解结果原地覆盖 B。系数矩阵不要求对角占优，因此需要支持部分主元选取和相邻行交换。现有 `aclsparseSgtsv2(float*)` 可参考工程调用流程；本任务新增独立 complex64 ABI，保留原有实数接口。

# 需求分析（required）

## 需求描述

使用 C++ 和 Ascend C 实现 `aclsparseCgtsv2_bufferSizeExt`、`aclsparseCgtsv2`，支持 Atlas A2/A3 的 DAV_2201（`arch22`）计算。核心分解与求解在 NPU 完成，调用提交至 handle 绑定的 stream。

## 需求拆解

1. 新增由两个连续 float 分量组成的公开复数类型及两个接口，提供参数校验、workspace 查询和异步求解。
2. 实现按复数模长比较的部分主元消去，保存行交换产生的第二上对角，普通路径的全部 RHS 复用同一分解，范围保护后备按活跃核保留私有状态。
3. 保持三条输入对角线只读，原地更新 B 的有效区，保留 leading dimension 对应的 padding；覆盖动态尺寸、空 RHS、确定性及异常边界。
4. 按任务书使用官方 ATK 验证精度，在 910B3 验证全部 203 条性能用例，完成 910B4、A3 功能与精度回归。

# 详细设计（required）

## 算子分析

### 数学公式与数据布局

采用零基下标，`A[i,i]=d[i]`，`A[i,i-1]=dl[i]`，`A[i,i+1]=du[i]`，边界 `dl[0]=du[m-1]=0+0i`。对每列 RHS 求解：

\[
dl_iX_{i-1,j}+d_iX_{i,j}+du_iX_{i+1,j}=B_{i,j},
\quad 0\leq i<m,\quad 0\leq j<n,
\]

首末行省略越界项。`dl/d/du` 是长度为 m 的连续 complex64 数组。B 以列主序存储，地址为 `B[i+j*ldb]`，完整存储形状为 `[ldb,n]`；仅 `0<=i<m` 参与求解，`m<=i<ldb` 保持原值。n 是右端项数量，无 batchCount、广播、转置或任意 stride 入口。

### 算子原型

在 `include/cann_ops_sparse.h` 中新增以下公开定义和声明：

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
| handle | 有效的 aclsparse 句柄，执行使用其绑定 stream。 |
| m、n、ldb | 运行时 int32 参数；`m>=3`、`n>=0`、`ldb>=max(1,m)`。 |
| dl、d、du | NPU 只读输入，均为 `[m]` 的 complex64 连续数组。 |
| B | NPU 输入输出，complex64 列主序 `[ldb,n]`，有效区原地覆盖为 X。 |
| pBufferSizeInBytes | 非空 Host size_t 输出指针，返回本次求解所需 workspace 字节数。 |
| pBuffer | 调用者分配的 NPU workspace，容量不小于查询结果，首地址按 128 字节对齐。 |

`n=0` 时，合法 handle 和整数参数仍需校验；查询返回 0，执行不启动求解 Kernel，允许数据及 workspace 指针为空。`n>0` 时，四个数据指针必须非空，执行还检查 workspace 非空与地址对齐。空 handle 返回仓内 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；非法整数、空必需指针、未对齐 workspace 或字节数溢出返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。stream 获取失败返回对应仓内状态码；Kernel 异步执行错误由调用者检查 stream 执行结果。

raw C ABI 不携带实际 dtype、分配长度或 stride 元数据；调用者及适配层负责保证这些条件。执行接口不传 workspace 实际容量，调用者须按查询结果足量分配。输入、B 和 workspace 必须保持有效直至对应 stream 完成；同一 workspace 不同时服务于相互独立的并发求解。

## 算子实现

### Host 侧设计

Host 共用一份参数校验与 workspace 布局计算，使用 size_t/int64_t 检查乘法、对齐及 B 地址跨度溢出。查询仅根据参数计算容量，不读取设备数据。执行取得 handle stream，选择分核及 UB 分块配置；普通路径依次下发分解与求解，检查点路径下发融合求解。

普通求解路径的 workspace 保存四组长度 m 的复数因子和 int32 pivot。全系统状态放入 UB 的小尺寸路径不需要额外 GM 状态；其余普通路径保留因子、候选解、补偿残差、独立指数及 pivot，用于范围保护和候选拒绝后的完整求解。单核进入范围后备后不再使用普通因子，两者共用存储；多核执行保留共享普通因子和各核独立后备状态。设 `C=align128(8m)`、`P=align128(4m)`、活跃核数为 K，检查点间隔 `q=ceil(sqrt(m))`，则 `n>0` 时：

\[
W=\begin{cases}
4C+P,&\text{全系统 UB 路径},\\
9C+P,&\text{单核分块路径},\\
4C+P+K(9C+P),&\text{多核分块路径},\\
128(\lceil m/q\rceil+\min(m,q)),&\text{检查点重算路径}.
\end{cases}
\]

K 不超过实际 AIV 数及 RHS 数；不对齐列跨度或 B 地址使用单核，避免跨列搬运边界冲突。普通分块路径按实际 L2 容量限制私有状态的复制数。若单份 `9C+P` 超过 L2，选择检查点重算路径，多个 RHS 依次复用一份有界工作区；Host 同时核对该路径的 UB 与 L2 容量。实际内存仍按任务书在目标设备验证。查询与执行共用布局和分核规则，不引入公开接口的尺寸上限；调用者按同一设备、相同尺寸、布局和数据地址查询后分配工作区。私有候选解接受后仍原地写回 B。

接口按 stream 顺序异步下发，内部不增加 Host 等待或 D2H 系数计算。真正奇异系统没有独立 info 输出参数：Host 成功提交不表示系统非奇异，不为读取设备主元状态强制同步；自测记录接口返回码、stream 执行结果和输出分类，并验证不越界、不崩溃。

### Kernel 侧设计

1. **带主元分解。** 将系数复制到内部工作区，逐行比较当前主对角与下一行下对角的复数欧氏模长，等模时保留当前行。根据选择执行消去或相邻行交换，保存乘子、pivot 和第二上对角，输入系数保持不变。模长比较使用缩放形式，复数除法采用缩放或分支求值，避免直接平方造成不必要的溢出、下溢。
2. **多 RHS 求解。** 各核处理互不重叠的 RHS 列，复用同一份因子；先按 pivot 顺序进行交换与前代，再按 U 的两条上对角回代：

   \[
   X_{i,j}=\frac{Y_{i,j}-U_{i,i+1}X_{i+1,j}-U_{i,i+2}X_{i+2,j}}{U_{i,i}},
   \]

   首末行按有效项处理。复数乘加在 FP32 实部、虚部上展开；范围敏感路径为因子、解和残差保留独立指数，最终写回时缩窄。UB 内采用分量布局或交错布局的分块载入，降低逐元素 GM 访问。
3. **并行与边界。** 多 RHS 以列为主要并行维度，长系统使用分块搬运及前代、回代仿射扫描。少 RHS 中等尺寸可在实际系数占优和范围检查通过后使用五轮并行约化；候选解经原方程补偿残差及范围检查接受后写回 B，拒绝时转入完整带主元、独立指数的求解路径。跨 Kernel 依赖依靠同一 stream 排序，核内显式维护搬运与计算同步。处理不对齐列跨度时，避免不同核写入同一搬运边界块；尾块仅更新有效元素，padding 保持不变。固定分派及主元规则，不使用输出累加原子操作。
4. **有界工作空间。** 检查点路径融合带主元消去和前代，每隔 q 行保存当前主对角、上对角、前代 RHS 及块首原始 RHS。逆序处理各块，重算块内已确定的因子与前代结果后回代，仅保留相邻两个已解值；跨块从检查点读取尚需使用的原始 RHS，避免原地写回覆盖。每个复数分量用两个 FP32 尾数及独立整数指数表示，最终按整数舍入写回 complex64；系数、暂存行与输出均分块搬运。

### 工程组织

| 位置 | 内容 |
| --- | --- |
| `include/cann_ops_sparse.h` | 复数类型与两个公开接口。 |
| `sparse/cgtsv2/` | 公共 Host 工具、接口说明。 |
| `sparse/cgtsv2/arch22/` | A2/A3 Host 调度、tiling 与 Ascend C Kernel。 |
| `test/cgtsv2/` | 公共 Golden、测试参数、ATK 配置及运行说明。 |
| `test/cgtsv2/arch22/` | C++ UT/ST、接口及设备端回归。 |

遵循仓内 SOC 选择与测试构建模式，不建立重复构建目录。A2/A3 与 A5 的公共类型、接口及 Host 工具保持单一定义，设备实现分别位于对应架构目录；后合入的 PR 基于上游公共代码完成冲突处理和交叉回归。

## 支持硬件与约束

支持任务环境提供的 Atlas A2 910B3、910B4 和 Atlas A3 具体型号，软件为 CANN 9.0.0 及后续配套版本。功能、精度按型号验证，性能仅在 910B3 验收。Python 适配若使用 PyTorch，采用 PyTorch 2.7 及以上、torch_npu 26.0.0 及后续配套版本。不以 arch22 共用源码替代各型号的实际验证。

# 可维可测分析

## 精度标准与测试方法

CPU Golden 将同一份已量化 complex64 输入提升为 complex128，使用 SciPy/LAPACK 带主元带状求解。实部、虚部分别满足混合容差 `|actual-golden|<=2^-13+2^-13*|golden|`，两部分匹配率均不低于 0.99。每一分量的最大有限绝对误差同时不超过 `max(1e-2,32*ULP(g_low))`，g_low 取该分量最大误差点的 Golden 收窄为 float32 后的值。

常规精度中 NaN 不计通过；NPU 为 Inf 而 Golden 有限时，仅 Golden 按 RNE 收窄到 float32 后为同号 Inf 才按溢出特例通过。独立特殊值套件逐位置检查 NaN/Inf 分类与 Inf 符号，其余有限分量逐点满足上述混合容差，不使用 0.99 匹配率或最大误差 ULP 上限。奇异与非有限输入的状态和输出另行记录。

使用任务指定的 ATK 26.9.8（提交 `9504bee6076820ca74259fb919a790a12ef88cf8`）、Python 3.10/3.11、NumPy<2。沿用随包 Tensor/Attr YAML、插件及官方 `atk case`、`atk task` 命令，独立运行附件 200 条与验收额外生成的 800 条精度用例。完整有限因子域另作覆盖审计，保持四输入各 50% 均匀、50% 正态分布及正交组合，不以对角占优变换替代常规精度输入。

C++ UT/ST 和上板测试覆盖类型布局、m/n/ldb 边界、单/多 RHS、非零 padding、需要连续 pivot、零主对角但非奇异、病态及真正奇异系统、Inf/NaN、空指针、workspace 对齐与边界、整数溢出、非默认 stream、系数只读、B 原地更新及完整查询/分配/执行/释放流程。非奇异输入在同设备、软件及配置下恢复原 B 后复跑，检查输出一致。大型合法因子组合保留覆盖与资源需求记录，不因未执行而记为通过。

## 性能与内存标准

正式性能验收采用任务附件固定的 203 条 n>0 用例，在 910B3 上逐条计算 `cuSPARSE Kernel总耗时中位数 / NPU Kernel总耗时中位数`，每条不低于 0.25，全部倍率算术平均值不低于 0.35。三条主场景为 `(m,n,ldb)=(8192,64,8192)`、`(4096,64,4096)`、`(7168,128,7168)`，GPU 基线分别为 732.032、717.120、1390.480 μs；不只对三条主场景取均值。

每例预热不少于 10 次、采样不少于 30 次。GPU 用 NCU，NPU 用官方 Profiler，求和一次求解调用包含的全部 Kernel；每次求解前恢复 B，恢复及数据准备不计时，workspace 复用。同一 stream 的 Event 辅助耗时独立记录，排除 B 恢复，不参与 Kernel 倍率计算。开发阶段使用相关用例作最小验证，阶段方案形成后执行完整验收采样。

两种采集口径均记录内存。B 仅计一次，输入输出字节数为 `(3*m+ldb*n)*8`；workspace 查询值、128 字节对齐实际申请量、恢复 B 的夹具副本分别列示。同时记录设备 free/total，以及使用框架 allocator 时的 allocated/reserved 基线、峰值和增量，注明单位、同步点及统计范围。内存满足任务书两项条件之一：输入输出超过 500 MB 时满足指定的 GPU/NPU 额外内存比例，或固有 workspace 不超过目标硬件 L2 Cache 容量；按实际设备核验。

保留源码版本、CANN/驱动/固件、完整 SOC、ATK 种子、原生结果、覆盖审计和 NPU Dispatch/Profiler 记录，分别报告功能、精度、性能、内存及跨平台回归。

## 兼容性分析

新增 Cgtsv2 类型与符号，保持现有 Sgtsv2 的类型、签名与行为；公共定义兼容后续 A5 实现，平台调度分离。回归既有实数接口及共享 Host 路径，不增加 CPU fallback 或隐式全局同步。
