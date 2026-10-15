# 需求背景

## 需求来源

本设计依据[10月社区任务—单精度实数 LU 分解、求解和批量求逆（950）](https://www.hiascend.com/activities/task-center/details/cd66edbecb8840b3bbcd39035146a970)及其任务书，为 [ops-solver](https://gitcode.com/cann/ops-solver) 设计单精度实数 LU 分解、基于 LU 的求逆和小矩阵直接批量求逆接口。

范围以任务书 §2.3 的接口定义为准：四个计算接口 `Sgetrf`、`Sgetri`、`SgetriBatched`、`SmatinvBatched`，以及两个 `_bufferSize` 查询接口，作为一组功能设计。本文说明实现方案及后续验证计划。

## 背景介绍

LU 分解通过部分主元行交换将矩阵分解为 `P*A=L*U`，可用于重复求解线性系统及矩阵求逆。批量接口处理一组尺寸相同、地址独立的矩阵，适用于大量小矩阵和中等规模矩阵的求逆。

接口采用 ops-solver Host C API 与 Ascend C/CATLASS Kernel 直调模式。矩阵、主元、info 和 workspace 位于 Device，公开布局为列主序。计算使用现有 handle 绑定的 stream。

## 现状分析

源码参考固定于 `ops-solver` 提交 `a29362cfa605c69b1f330feadbff21e686eb9502`。

| 项目 | 当前源码行为 | 任务要求 |
| --- | --- | --- |
| 公开声明 | `Sgetrf/Sgetri` 使用 `aclError` 和 `int64_t`，无查询接口 | `aclsolverStatus_t`、32 位 `int`、独立 workspace 查询 |
| 数据入口 | Host 分配内部 Device 缓冲、复制矩阵，结束后同步并拷回 | 直接处理调用方 Device 指针 |
| leading dimension | Host 要求 `lda==n` | 列主序，支持合法 padding |
| info | Host 直接初始化 `*info` | Device 上真实写入成功、非法参数和奇异位置 |
| `Sgetri` | Kernel 中调用 `custom_lu` 后求逆 | 接收已分解 LU 和主元，不再次分解输入 |
| 实数批量接口 | 尚无本任务两个批量公开接口 | 增加基于 LU 的批量求逆及 `n<=32` 直接求逆 |
| handle | 已有 Create/Destroy/SetStream/GetStream | 复用既有管理接口 |

现有 LU、三角求解和矩阵更新代码可作为算法与工程参考。其固定分核和 C220 条件分支不能直接证明 950PR 适配，需要按目标产品生成 tiling 和计算路径。

# 需求分析

## 需求描述

在 Ascend 950PR 上支持 FLOAT32、列主序、部分主元和无主元分解，正确处理矩形矩阵、padding、空问题、奇异矩阵及批量独立状态。提供调用方 workspace 查询，批量接口按既定参数组织内部临时空间。

## 需求拆解

| 功能 | 输入与输出语义 | 最低覆盖范围 |
| --- | --- | --- |
| `Sgetrf` | 原始 `m*n` 矩阵 → 原地打包 LU、1-based 主元和 info | `m,n` 至 4096，方阵与矩形 |
| `Sgetri` | 方阵 LU 和主元 → 原地逆矩阵和 info | `n` 至 4096 |
| `SgetriBatched` | Device 指针数组中的 LU → 独立输出数组 | `n` 至 4096，batchSize 至 1000000 |
| `SmatinvBatched` | Device 指针数组中的原始矩阵 → 独立逆矩阵 | `n<=32`，batchSize 至 1000000 |
| workspace 查询 | Host 返回计算所需的 float 元素个数 | 与当前形状及产品路径一致 |
| 执行与状态 | 使用 handle 的 stream；Device 写 info；固定输入确定性 | 非默认 stream、连续调用、失败项独立 |

官方用例还包含 `Sgetrf` 的 `n=8168` 和 `SmatinvBatched` 的 `batchSize=10000000`。前述区间是验收下限，设计按分块和宽整数偏移覆盖包内更大规模，不将正文下限作为硬编码拒绝条件。`SmatinvBatched` 的 `n<=32` 则是接口硬约束。

# 详细设计

## 公开接口与数据契约

在 `include/cann_ops_solver_common.h` 增加任务书规定的枚举：

```cpp
typedef enum {
    ACLSOLVER_OP_N = 0,
    ACLSOLVER_OP_T = 1,
    ACLSOLVER_OP_C = 2
} aclsolverOperation_t;
```

本组计算接口不接收 `trans` 参数。枚举值按任务书与 CUDA 对齐。

在 `include/cann_ops_solver.h` 声明：

```cpp
aclsolverStatus_t aclsolverSgetrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, float *A,
    int lda, int *Lwork);

aclsolverStatus_t aclsolverSgetrf(
    aclsolverHandle_t handle, int m, int n, float *A, int lda,
    float *Workspace, int *devIpiv, int *devInfo);

aclsolverStatus_t aclsolverSgetri_bufferSize(
    aclsolverHandle_t handle, int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSgetri(
    aclsolverHandle_t handle, int n, float *A, int lda,
    int *devIpiv, float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSgetriBatched(
    aclsolverHandle_t handle, int n, const float *const Aarray[], int lda,
    const int *PivotArray, float *const Carray[], int ldc,
    int *infoArray, int batchSize);

aclsolverStatus_t aclsolverSmatinvBatched(
    aclsolverHandle_t handle, int n, const float *const A[], int lda,
    float *const Ainv[], int lda_inv, int *info, int batchSize);
```

矩阵元素地址为 `A[i+j*lda]`，有效行数之外的 padding 保持不变。Host 参数包括 handle、整数维度和查询结果 `Lwork`；矩阵、主元、info 和执行 workspace 均为 Device 缓冲。

批量 `Aarray/Carray/A/Ainv` 自身也位于 Device，各元素保存对应矩阵的 Device 地址。`PivotArray` 是连续的 `n*batchSize` 个 int，第 `b` 个矩阵的主元起点为 `b*n`。不以连续三维 Tensor 替代指针数组。

`devIpiv/PivotArray=nullptr` 表示无主元 LU。非空主元应对应同一输入 LU；Kernel 检查合法行号范围，调用方保证其与分解结果一致。批量求逆的输入与输出不重叠，各矩阵输出区域互不重叠。

## 数学与算法设计

### Sgetrf：分块部分主元 LU

对于 `m*n` 矩阵，令 `k=min(m,n)`：

$$
P A=L U,\quad
L\in\mathbb{R}^{m\times k},\quad
U\in\mathbb{R}^{k\times n}.
$$

按宽度 `b` 划分面板，当前起点为 `s`，实际面板宽度为 `min(b,k-s)`。

1. 在 `A[s:m,s:s+b]` 内逐列分解。主元从当前列尚未消元的全部有效行中选取，以绝对值最大、行号最小为确定规则，写出全局 1-based 行号。
2. 在面板内执行行交换、除法与消元。将同一主元序列按正序应用到已形成的左侧 L 区域和右侧未处理列，使所有区域具有一致行序。
3. 解 `L11*U12=A12`，得到当前面板右侧上三角块。
4. 更新 `A22=A22-L21*U12`，进入下一面板。

最后一个面板和更新块均使用实际行列数。高矩形矩阵保留完整下梯形 L，宽矩形矩阵在最后一个主元后仍计算剩余 U 列。

无主元路径省略搜索与行交换。零主元按精确零判断，记录最小的全局位置；对应列不执行除零，继续按 LAPACK 分解语义处理可完成的部分，不将近零值自动判为奇异。

### Sgetri：基于已分解 LU 求逆

由 `P*A=L*U` 得到：

$$
L Y=P,\qquad U X=Y,\qquad X=A^{-1}.
$$

将单位矩阵按列划为宽度 `r` 的 RHS 块。每块先按主元正序对相应单位矩阵列执行行置换，得到 `P*E_j`，随后进行 L 前代与 U 回代，写出该组逆矩阵列。`devIpiv=nullptr` 时 RHS 不置换。

单矩阵接口原地输出，因此第一步将有效 LU 因子保存到调用方 workspace，再开始覆盖 A。后续所有求解读取保存的 LU，不读取已经被逆矩阵覆盖的 A 区域。无需存放完整单位矩阵，RHS 块在 Device 上生成。

求逆前检查 U 对角，首个零对角通过 `devInfo` 报告；该矩阵不继续三角除法，其数值输出不作为有效逆矩阵使用。

### SgetriBatched：已分解 LU 的批量求逆

各矩阵使用相同的三角求解算法。输入 LU 保持只读，逆矩阵写入 `Carray[b]`，主元使用 `PivotArray+b*n`。

小矩阵在 UB 中保留 LU 与 RHS/结果，一个核心循环处理多个批次。较大矩阵按批次组和 RHS 列块调度，使用受限的 GM 临时区，计算后写出对应矩阵列。一个矩阵内的依赖按固定顺序执行，批次间并行，每个 batch 的 info 独立维护。

### SmatinvBatched：小矩阵直接求逆

`n<=32` 时，每个任务将原始矩阵搬入 UB，在同一任务内完成部分主元 LU 和单位 RHS 的前代、回代，再写出逆矩阵。主元仅为内部临时量，不新增公开输出参数。

单份 32 阶 FP32 矩阵为 4096 字节。LU、逆矩阵/RHS、行交换暂存和主元可在 UB 内组织；同时驻留的矩阵数根据实际 UB 预算确定。`n=33` 在 Host 参数检查阶段返回非法参数，不进入大矩阵求逆路径。

## Host 侧设计

### 参数检查与 info

复用现有 `aclsolverCreate/Destroy/SetStream/GetStream`。Host 校验 handle、非负维度、leading dimension、顶层空指针和 workspace 查询/长度参数，并使用仓库已有 `aclsolverStatus_t`。

| 条件 | 返回值及 info |
| --- | --- |
| 合法执行计划已下发 | `ACLSOLVER_STATUS_SUCCESS`；计算结果通过 Device info 表达 |
| 空 handle | `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR` |
| Host 可判定的非法参数 | `ACLSOLVER_STATUS_INVALID_VALUE`；可写 info 在调用 stream 中置为 `-i` |
| 内存申请失败 | `ACLSOLVER_STATUS_ALLOC_FAILED` |
| 不支持的产品路径 | `ACLSOLVER_STATUS_ARCH_MISMATCH` |
| Runtime/Kernel 下发失败 | 按已有状态码映射为执行失败或内部错误 |
| 奇异矩阵 | Host 下发状态与数值状态分开；Device info 为首个零主元位置 |

参数序号不计 handle，例如：

| 接口 | `lda` 错误 | 其他示例 |
| --- | --- | --- |
| `Sgetrf` | `devInfo=-4` | `m=-1` 对应 `-1` |
| `Sgetri` | `devInfo=-3` | `Lwork` 不足对应 `-6` |
| `SgetriBatched` | 各可写 info 为 `-3` | 非法 `ldc` 对应 `-6` |
| `SmatinvBatched` | 各可写 info 为 `-3` | `n=33` 对应 `-1` |

Device 内的主元内容、指针数组元素和输入输出地址关系由入口校验 Kernel 检查；失败项写 `-i` 并停止访问其矩阵。是否要求 Host 同步读取这类校验结果以返回 `INVALID_VALUE`，在评审中确认，不能把下发成功视为已经验证 Device 参数内容。

空问题在基本维度检查后处理。单矩阵 `m=0` 或 `n=0` 时不访问 A、主元和 workspace，按任务书要求向可写 `devInfo` 写 0；批量 `batchSize=0` 时不访问数组，`n=0` 且 batch 非零时只初始化对应 info。

### Workspace 查询与布局

查询接口仅返回 Host `Lwork`，不读取矩阵内容、不启动分解。`Lwork` 单位为 float 元素个数，调用方申请字节数为 `sizeof(float)*Lwork`。

计算与查询共用布局计算函数。设 `Align` 为目标路径要求的字节对齐，分区包含：

| 接口 | workspace 分区 |
| --- | --- |
| `Sgetrf` | 面板/行交换暂存、主元归约暂存、矩阵更新 packing 与确定性累加临时区、控制数据 |
| `Sgetri` | 保存有效 LU 的 `n*n` 个 float、`n*r` RHS 块、三角求解与更新临时区、控制数据 |

```text
W_getrf = sum(Align(each getrf partition))
W_getri = Align(4*n*n) + Align(4*n*r) + W_tri(n,r) + W_control
Lwork = ceil(W_bytes / sizeof(float))
```

整数控制区在 float workspace 内按其自身对齐划分。offset、padding、矩阵体积和批次偏移使用宽整数计算，转换为 `int Lwork` 前检查可表示范围。

`Sgetri` 校验 `Lwork` 不小于查询值。`Sgetrf` 的执行接口没有长度参数，调用方须按相同形状的查询结果分配，不能根据该原型承诺检测实际分配长度。工作区不保存跨调用有效的 LU 身份，使用前初始化本次需要的控制数据。

### 批量临时空间与异步寿命

小矩阵 UB 路径不需要随 batchSize 增长的矩阵 GM scratch。大矩阵路径一次处理 `q` 个矩阵、每矩阵 `r` 个 RHS 列，scratch 约为 `q*W_tri(n,r)` 加共享 tiling/control 分区。`q` 根据计算资源和工作区预算选择，不为全部 batch 同时分配完整中间矩阵。

批量接口无 workspace 参数，内部 scratch 由现有 handle 的内部结构管理，不增加公开管理接口。分配、使用、复用和释放遵守以下顺序：

1. 为本次调用取得容量足够的 scratch，记录所属 Device/context 和最后使用的 stream/event。
2. 在当前 stream 上依次下发批次组，各组按流内顺序复用 scratch。
3. 最后一个使用者之后记录完成事件；同 stream 后续调用可按流内顺序复用。
4. handle 更换 stream 时，复用前由新 stream 等待完成事件；扩容产生的旧缓冲在完成后回收。
5. `aclsolverDestroy` 完成内部资源清理后释放 scratch 和事件，调用方 stream 仍由调用方管理。

Host 不在 Kernel 下发后立即释放尚在使用的缓冲。实现优先使用配套 CANN 支持的流序内存能力；无对应能力时使用上述事件管理。对同一 handle 的调用与 SetStream 操作按顺序执行，不同 handle 的 scratch 相互独立。

## Kernel 与 tiling 设计

### 数据搬运与分核

Host tiling 包含 m/n、lda/ldc、batchSize、面板宽度、RHS 块宽度、有效尾块、workspace offset 及选主元模式。核数与 UB/L1/L0 资源从平台查询。

矩阵读取保留公开列主序与 leading dimension。内部 packing 可转换为矩阵乘法所需布局，写回只覆盖有效行列；padding 使用哨兵值验证保持不变。

面板主元搜索由 AIV 执行。大面板按行段产生 `(abs(value), row)` 候选，再按固定顺序归约，保证相等最大值选择最小行号。矩阵更新和多个 RHS 块按不相交输出 tile 分配，每个 tile 的 K 方向累加顺序固定。

### 阶段依赖

初版采用同 stream 多 Kernel 串联面板分解、面板外行交换、TRSM 和 trailing update。下一阶段只在前一阶段完成后读取其结果，不使用依赖驻留核数量的全局自旋屏障。

求逆的 LU 保存、对角检查、RHS 置换、前代、回代及写回也按该 stream 排序。批量失败项由 Device info 控制后续访问，其他矩阵继续处理。

主元交换按序执行；info 使用确定性归约记录首个零主元。矩阵更新采用固定 K 分块及唯一写出者，避免不同核对同一结果进行无序浮点原子累加。

### 计算类型

面板分解、主元比较、三角求解使用 FP32。大矩阵更新拟使用目标产品支持的 FP32 输入与累加的 Ascend C/CATLASS 路径，关闭会改变有效输入精度的 HF32 等模式。

现有源码虽实例化 `MatmulCustom<float>`，仍需核对 950PR 编译目标、对应 Matmul 指令与精度配置。实现阶段先验证 FP32 GEMM/TRSM 的残差与确定性，再接入整体 LU；不将 FP16 降精度路径作为本设计的精度基础。

## 工程与兼容性

| 位置 | 计划变更 |
| --- | --- |
| `include/cann_ops_solver.h` | 升级 Sgetrf/Sgetri 原型，新增查询与实数批量接口 |
| `include/cann_ops_solver_common.h` | 增加 operation 枚举，复用现有 status 类型 |
| `src/sgetrf/`、`src/sgetri/` | Device 接口、workspace、950PR Kernel |
| `src/sgetri_batched/`、`src/smatinv_batched/` | 批量 Host 与 Kernel |
| `src/utils/` | 共享布局计算及内部 scratch 寿命管理 |
| `test/` 对应目录 | 接口、精度、状态、padding 与确定性测试 |
| `docs/zh/`、`docs/api_list.md`、README | 公开契约、调用步骤、版本配套及接口列表 |

Sgetrf/Sgetri 参数序列及语义变化构成 API/ABI 升级，旧二进制调用不能直接沿用。测试和调用样例同步改为：创建 handle、设置 stream、查询 Lwork、申请 Device 缓冲、计算、同步并读回 info/结果。若仓库需要保留旧入口，另行评审命名与过渡期，不能让两个原型共享同一 C 符号。

本任务不改动复数计算接口的公开语义；公共 handle 的内部资源变更需保留现有接口回归。

## 支持硬件与约束

目标为 Ascend 950PR，CANN 9.0.0 及以上且与 ops-solver README 的验证版本配套。实现使用 ACL、Ascend C 和 CATLASS 工程，不依赖 CUDA 执行计算。

支持 FLOAT32、列主序和统一 batch 尺寸，不支持广播。`Sgetri` 输入为 LU，`SmatinvBatched` 输入为原始矩阵；单矩阵原地输出，批量独立输出。调用方负责保持 Device 缓冲有效，并在读回结果前同步其 stream。自构造用例按任务书将输入占用控制在 4 GB 范围内。

## 待确认事项

| 官方材料差异或接口限制 | 本设计处理 | 评审确认点 |
| --- | --- | --- |
| 正文称“五个接口”，§2.3 为四个计算加两个查询；测试说明出现 B/nrhs/trans | 以六个明确原型设计，保留规定枚举 | 是否存在尚未列出的 Sgetrs 交付要求 |
| 要求非法 Device 内容返回 Host 错误码，同时要求避免无必要同步 | Host 检查与 Device info 分开；入口阻止失败项继续计算 | Device 内容错误是否必须在返回前同步确认；LU 与主元一致性的可校验边界 |
| `smatinvbatched` GPU 金标有两条 `perf=null,error=2` | 保留功能结果和原始金标状态，不生成倍率 | 是否补充对应 GPU 性能基线 |

# 可维可测分析

## 精度标准

单位舍入 `eps=2^-24`。CPU 参考分解与求逆使用 FLOAT32，同 case、同残差实现计算 `ratio_cpu`；FP64 可辅助诊断，不替代同精度 CPU 比例。

### LU 残差

$$
\mathrm{ratio}=\frac{\|\widehat{A}-A\|_1}
{n\|A\|_1\varepsilon}.
$$

先由打包 LU 构造 `L*U`，再将主元按 `min(m,n)` 到 1 的逆序执行行交换，得到原始行序的 `A_hat`。矩形输入的分母仍使用列数 n。

```text
ratio <= max(5*ratio_cpu, 0.3)
```

### 求逆残差

$$
\mathrm{ratio}=\frac{\|I-C A\|_1}
{n\|A\|_1\|C\|_1\varepsilon}.
$$

`Sgetri/SgetriBatched` 的 A 由输入 LU 与主元还原；`SmatinvBatched` 使用原始 A。批量逐矩阵判定：

| 接口 | 阈值 |
| --- | --- |
| `Sgetri` | `max(5*ratio_cpu,0.05)` |
| `SgetriBatched` | `max(5*ratio_cpu_i,0.05)` |
| `SmatinvBatched` | `max(5*ratio_cpu_i,0.5)` |

奇异矩阵按 info 验证，不对无有效逆矩阵的结果作上述残差通过判断。

## 功能与覆盖计划

官方用例与金标按参数完整匹配，覆盖以下全量清单：

| 接口 | 官方用例文件 | 条数 |
| --- | --- | --- |
| `Sgetrf` | `sgetrf/cases.json` | 148 |
| `Sgetri` | `sgetri/cases_brief.json` | 200 |
| `SgetriBatched` | `sgetribatched/cases.json` | 149 |
| `SmatinvBatched` | `smatinvbatched/cases.json` | 215 |

清单当前未覆盖 padding、空问题及无主元等全部接口边界，补充验证包括：

- 方阵、高矩形、宽矩形、非整块尾部；最小 leading dimension 及 `+8/+32` padding。
- 有主元、无主元、相等绝对值主元；打包 LU 和 1-based 主元还原。
- 成功 info=0、首个零主元 info>0、非法参数 info=-i；一个 batch 失败而其他 batch 正常。
- `n=1`、`batchSize=1/0`、空矩阵；matinv 的 `n=32/33`。
- 单矩阵原地求逆的 LU 保存；批量 Device 指针数组及输入输出分离。
- 固定原始输入重复执行，逐位比较矩阵、主元与 info；重复前重新恢复原始输入。
- 非默认 stream、同 stream 连续调用、SetStream 后内部 scratch 依赖、Destroy 清理。

分解数据覆盖任务书要求的均匀/正态分布与对角占优，并单独覆盖奇异、近奇异和零范数矩阵。生成参数、随机种子和每个 batch 的结果均保留在验证记录中。

## 性能标准与验证

目标为每条可比用例：

```text
T_GPU / T_NPU >= 0.35
```

GPU 时间读取各算子 `bench_result.json` 的 `perf.avg_ms`，转换为与 NPU 相同单位。按 m/n、lda/ldc/lda_inv、batch、pivot 等参数匹配，不能仅按矩阵阶数匹配。

NPU 使用 ops-solver 原生测试入口和 msprof，按单条 case 预热后采样。一次接口调用由多个 Kernel 组成时，关联完整调用的全部计算 Kernel，按调用数求平均；workspace 查询、准备 LU 的 getrf 及数据拷贝与目标计算的计时范围分别记录，不能只选最快的 GEMM 子 Kernel。

任务书九个代表规格作为重点核对项：getrf 的 1024 方阵、4096 方阵及 2048×1024 矩形；getri 的 1024、4096；getriBatched 的 32×1024 batch、128×128 batch；matinvBatched 的 16×1024 batch、32×256 batch。正式结论覆盖官方全量可比较记录。

`smatinvbatched` 中 n=2、4 且 batch=10000000 的两条 GPU 记录缺少耗时，保留其原始错误状态并请求补充基线，不将其标为性能通过。官方 `Sgetri` 金标的 operator 名称需按任务书说明的“getrf 后求解 AX=I”链核对，求逆阶段不计 getrf 时间。

## 可维护性与交付

公开接口与计算布局、workspace、tiling 使用同一套定义。矩阵更新和三角求解只共享本组接口需要的内容。设计、API 文档与原生测试同步说明列主序、Device 指针、Lwork 单位和 info 时序。

后续实现验证记录源提交、工具链、驱动、产品型号、准确命令、原始日志、逐 case 精度与性能结果。本文不填入尚未执行的测试数值。

## 参考资料

- [官方任务页](https://www.hiascend.com/activities/task-center/details/cd66edbecb8840b3bbcd39035146a970)及其任务书、四组 case/bench 文件。
- [ops-solver 公开接口](https://gitcode.com/cann/ops-solver/blob/a29362cfa605c69b1f330feadbff21e686eb9502/include/cann_ops_solver.h)。
- [ops-solver Sgetri Kernel](https://gitcode.com/cann/ops-solver/blob/a29362cfa605c69b1f330feadbff21e686eb9502/src/sgetri/sgetri_kernel.cpp)。
- [NVIDIA cuSolver](https://docs.nvidia.com/cuda/cusolver/index.html)、[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)。
- [LAPACK SGETRI](https://netlib.org/lapack/explore-html/d8/ddc/group__getri.html)。
