# 需求背景（required）

## 需求来源

《10月社区任务-aclsparseCgtsv2算子开发（A2A3）》：为 Atlas A2/A3 的 ops-sparse 提供 complex64 三对角线性方程组求解，接口对标 cuSPARSE Cgtsv2。目标代码仓为 `cann/ops-sparse` 的 `master`；设计仓为 `cann/cann-ops-competitions`。上游 master 当前没有 Cgtsv2 任务目录；本设计拟新建 `10-aclsparseCgtsv2-A2A3` 任务目录，目录名称和位置可由社区评审调整。本文为待评审的设计方案，尚未获得社区批准；文中的原型和实验仅用于说明设计取舍，不代表最终实现或验收通过。

## 背景介绍

求解单个三对角矩阵 A 和多个 RHS 的 `A X = B`；系数向量为 `dl/d/du`，其第 0 个下对角元素和最后一个上对角元素为 0。B 是列主序 `[ldb,n]`，结果原地覆盖其前 m 行。目标为 A2/A3 的 DAV_2201（arch22），核心计算在 NPU，禁止 CPU fallback。现有 `Sgtsv2` 的 float ABI 不变。

# 需求分析（required）

## 需求描述

新增 `aclsparseComplex_t`，两个连续 float 分量 `real`、`imag`，在 C++ 编译期验证大小为 8、对齐等于 float。新增 `aclsparseCgtsv2_bufferSizeExt` 和 `aclsparseCgtsv2`，分别查询与执行；ABI 参数顺序与任务书一致。`m>=3`、`n>=0`、`ldb>=max(1,m)`；`n=0` 查询返回 0 并跳过求解。`n>0` 系数及 B 地址非空，执行 workspace 非空、128 字节对齐且由调用者按查询值足量分配。设备输入只读，B padding 保持不变。

```cpp
typedef struct { float real; float imag; } aclsparseComplex_t;
aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, const aclsparseComplex_t *dl,
    const aclsparseComplex_t *d, const aclsparseComplex_t *du,
    const aclsparseComplex_t *B, int ldb, size_t *pBufferSizeInBytes);
aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n, const aclsparseComplex_t *dl,
    const aclsparseComplex_t *d, const aclsparseComplex_t *du,
    aclsparseComplex_t *B, int ldb, void *pBuffer);
```

查询的大小输出为 host 标量；系数、B 和执行 workspace 为设备地址。调用者持有 handle、输入、输出及 workspace，直至 stream 上相关操作完成；禁止调用期间复用互相重叠的可写缓冲区。`n=0` 时仍检查 handle、尺寸以及查询输出指针，但不要求系数、B 或 workspace。拟定错误语义：空 handle 返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`；无效尺寸、必需指针为空、workspace 未满足对齐或字节计算溢出返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。stream 获取失败和运行时启动错误按仓库现有状态传播；异步设备错误需在 stream 同步时检查。具体映射仍需随目标仓库接口评审并以真实 C ABI 测试核对。

## 需求拆解

1. Host 校验 handle、维度、空指针、输出地址、workspace 对齐及涉及字节数和索引的整数溢出；原始 ABI 无法检查真实 dtype、分配长度或 stride。
2. 支持复数模长比较的相邻行部分主元选取，维护行交换形成的第二上对角项；全部 RHS 应使用相同矩阵分解结果和相应交换。
3. 调用在 handle stream 异步提交；调用者负责输入、B、workspace 生命周期直到 stream 完成。不可因未同步的设备奇异结果而声称 Host 已检测到奇异。
4. 实测功能、精度、性能、内存、确定性和跨平台回归；真实奇异和非有限值按独立套件记录，不计作常规精度通过。

# 详细设计（required）

## 算子分析

### 数学公式

`A[i,i]=d[i]`，`A[i,i-1]=dl[i]`（i≥1），`A[i,i+1]=du[i]`（i<m−1），求 `A X = B`。对 i 从 0 到 m−2，比较当前对角主元与下一行下对角元素的复数模，选择主元并按需要交换相邻行，将消元乘数、更新后的对角和两个上对角以及 pivot 记录于工作区；前向应用交换及消元于每列 B，再使用一、二上对角反向回代。主元比较及复数除法需设计防止平方或分母中间溢出，确定相等时一致的选择规则。

### 支持数据类型

仅 complex64，即以两分量 float 表示的复数；公开接口严格使用 `aclsparseComplex_t*`。不添加 float ABI 重载、不支持广播。

### 支持形状

运行时动态 m/n/ldb。三条系数长度 m；B 的有效解是 `B[i+j*ldb]`（0≤i<m、0≤j<n）；其余 padding 不写。n=0 是合法空 RHS。调用者保证连续布局、真实容量和 `dl[0]=du[m-1]=0`。

## 算子实现

### 实现方案

#### 数据流与职责

| 阶段 | 输入与职责 | 输出或验证 |
| --- | --- | --- |
| Host 查询 | 校验 handle、输出地址与形状；n=0 直接返回 0 | 返回所选实现所需的字节数；不读取设备数据 |
| Host 执行 | 校验尺寸、必需地址、workspace 对齐和整型范围，取得 handle stream | 在该 stream 上提交设备工作；调用方保留所有缓冲区直至完成 |
| 设备求解 | 只读系数，原地更新 B 有效区；部分主元及其形成的填充项必须保留 | 相同输入和配置确定性复跑；padding 不变 |
| 测试与验收 | CPU complex128 Golden、官方 ATK、C++ 调用及设备性能采集 | 按 SoC、代码版本和用例分别记录通过、失败或未执行 |

#### Host 侧设计

在 `include/cann_ops_sparse.h` 新增类型和两个声明，在 `sparse/cgtsv2/` 放公共校验与工具、`arch22/` 放调度和 Kernel；配套 `test/cgtsv2/arch22/` 和 CMake 接入。按目标库的真实构建及符号导出机制集成。参数校验先处理 handle、query 输出指针和 m/n/ldb，再处理 n=0；n>0 时校验所有数据指针、工作区及整数上界，查询与执行共享相同安全计算。B 的最大访问索引及字节数也需做溢出检查。128 字节向上对齐前先检查加法溢出。实际工作区布局和精确字节式在实现与设备容量测量后确定并固定由查询返回，不能假定现有实数算子的大小适用于复数。

通过 handle 获取 stream，按相同 stream 顺序启动分解、RHS 求解等所需 Kernel，不在接口内部无故同步。Host 返回参数及 launch 相关状态；无法从异步调用即时推断数值奇异或设备执行最终结果。多个调用只访问各自工作区和 B，调用者对并发别名负责。

当前正确性原型 `b26406821b91aa7c7ec9281d3a85b491ad61e72f` 的工作区布局可供复现与评审（不代表最终性能版本）：以 128 字节对齐的基址为起点，0～`80m-1` 字节是 20m 个 float 尾数，`80m`～`160m-1` 字节是 10m 个 int64 指数，`160m`～`161m-1` 字节是 m 个 pivot 字节；分别对应对角、第一上对角、第二上对角、消元乘数及一列 RHS 这五组扩展复数。每个复数分为实、虚两部分，每部分存两个 float 和一个 int64。`n>0` 查询 `align_up(161*m,128)` 字节，`n=0` 为 0；计算乘法、加法和向上对齐之前必须检查溢出。指数段起点 `80m` 对所有合法整数 m 均满足 8 字节自然对齐，末尾补齐不参与计算。查询和执行必须使用完全相同的公式；该布局不得与将来的快速路径公式混淆。

例如 `m=2^20` 时原型 workspace 为 168,820,736 字节。910B3 的 L2 在目标环境通过运行时能力接口测得 201,326,592 字节，但该测量不能外推到其他 SoC；本数值只用于预算，实际需分别计入 IO、框架分配、对齐开销、原生 handle workspace 和快速路径状态。任务内存条件按实际 workload 的对应判据核验，不能仅由 workspace 小于 L2 推断整体内存通过。快速路径若增加缓冲，其查询值必须覆盖所有可能发生的路径；若预算不足则选择有界分组或保守回退，不能执行超出查询值的布局。

#### Kernel 侧设计

**正确性原型。** 对角、上对角和临时数据存入扩展指数表示：每个实分量用两个 float 尾数及 int64 指数表示，复数运算组合实部分量，比较复数模长时避免直接用 float 平方产生中间上溢。设备按相邻行进行部分主元消元，记录 pivot、乘数及交换形成的第二上对角，再逐列进行前向变换和回代；用户系数不改，写回仅覆盖 B 的有效区。当前 checkpoint 实际以单个 AIV 执行因子分解和全部 RHS，故无跨核共享工作区冲突；此实现用于正确性基线而非性能达标声明。设备实现需对 int64 指数运算、极端抵消、次正规数、舍入及 NaN/Inf 分类分别验证；当前尚未证明所有合法有限输入与所有长度都不会发生指数溢出。

**待评审的可回退快速路径。** 仅对输入数值满足可验证条件的矩阵尝试，不按尺寸、种子或基准编号特判。以 `δ_i=max(|Re(d_i)|,|Im(d_i)|)-(|Re(dl_i)|+|Im(dl_i)|)-(|Re(du_i)|+|Im(du_i)|)` 作保守判定，边界处忽略不存在的副对角项，要求所有输入有限且 `δ=min_i δ_i>0`。对每组 RHS 保留原 B，使用两个候选缓冲进行分块复数 Jacobi 迭代，halo 从上一轮只读缓冲获取；每轮通过同一 stream 的独立 kernel 保证跨块次序，轮数上限只控制开销，不作为精度证明。

试算结束后从原始系数、B 和候选解计算残差，只有经过保守认证才允许写回；认证失败或数值非有限时保持原 B，转入支持部分主元的通用路径。候选认证将每个 float32 分量按整数乘 `2^-149` 精确解码，复数残差实虚分量按 `2^-298` 累加，以 9 个 64 位 limb 表示有符号结果；令 `R` 为各行各 RHS 的 `|Re(r)|+|Im(r)|` 最大值，δ 按 `2^-149` 表示为正整数时比较 `R<=δ<<136`。该条件在整数运算无溢出且实现正确时给出相对精确数学解的分量绝对误差上界 `2^-13`；它本身不证明对 complex128 Golden 的误差，参考解舍入预算和实际任务比较仍须独立验证。分量解码、负零、非有限值、进位/借位、位移边界、UB 与 cache 行同步必须独立证明和测试；认证无法建立时直接回退。

快速路径的候选额外空间为每组 `2*align_up(8*m,64)*g` 字节的双缓冲，加上精确认证和归约状态，以及通用回退 workspace；所有段需检查溢出并计算对齐，g 为一次处理的 RHS 数。该方案尚未接入生产；已有局部研究探针，但完整方案、证明和性能均未验证，最终布局、状态大小和守护命中率均待评审。不同核写入的 B 列必须避免缓存行共享；不能保证安全时单写者或回退。`DataCopyPad` 搬运分块不超过 16 KiB，并为 UB 对齐与尾块留足余量。通用路径的串行依赖和扩展复数运算成本较高，203 条性能目标仍需逐例 profiler 验证；严禁通过牺牲 pivot 或数值正确性达标。奇异输入按非崩溃、无越界要求记录，不承诺具体奇异错误码或有限解。

## 支持硬件

| 平台 | 计划功能及精度 | 性能验收 |
| --- | --- | --- |
| Atlas A2 910B3 | 是 | 是 |
| Atlas A2 910B4 | 是 | 否 |
| Atlas A3 任务环境具体型号 | 是 | 否 |

各平台须分别绑定完整 SoC、CANN、驱动、固件、提交和测试结果；当前表为任务要求，不将单一设备诊断扩展为其他平台通过。

## 算子约束限制

无 batchCount、独立 X、转置视图或任意 stride；有效 B 区域必须允许原地覆盖。workspace 在异步执行结束前保持有效。raw 指针不可用于推断分配长度或 dtype。Inf/NaN 不作为常规数值精度通过；分类、位置和符号由特殊套件独立检查。workspace 实际申请及对齐空间、测试复位副本和 allocator 统计分开记录。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 输入 complex64 升 complex128，SciPy/LAPACK 带主元 Golden；实部和虚部分别 `atol=rtol=2^-13`，匹配率≥0.99，并满足任务书最大有限绝对误差动态 ULP 上限 | 当前任务书；混合容差标准提交 a5e8e716743b20bfb2ffa16e0213c4aa73b823b3 |
| 性能 | 仅 910B3，203 条逐例 GPU/NPU Kernel 总耗时中位数比≥0.25，全部算术平均≥0.35；每例预热≥10、采样≥30，B 恢复排除计时；Event 单列 | 当前任务书及性能附件 |
| 内存 | 输入输出超过 500 MB 时额外内存不超过 GPU 总量 5%，或固有 workspace 不超过目标 L2；逐例记录查询、实际分配、框架峰值和设备快照 | 当前任务书 |

验证矩阵分别覆盖 910B3、910B4 和任务环境的 A3：每个平台运行公开接口、边界、只读/padding、确定性、普通数值和特殊语义；只有 910B3 与同输入 GPU 逐例比较 Kernel 性能。B3 上 checkpoint 已完成随包 200、额外 800、特殊值 8 和适配合同 12 的诊断；这些结果仅绑定该 checkpoint 和实际测试环境，不覆盖其他 SoC、完整组合或最终验收。适配层拒绝不等于 raw C ABI 的错误码验证。

采用官方 ATK 26.9.8，核对提交 `9504bee6076820ca74259fb919a790a12ef88cf8`；运行随包 200 条及独立额外 800 条，并单独生成和审计完整有限域、特殊及合同测试。C++ UT/ST 验证空指针、边界、workspace 对齐、溢出、只读系数、B padding、stream 生命周期和重复执行确定性；适配层拒绝不替代真实 API 证据。保留命令、版本、代码 SHA、退出码、原始日志、Profiler 样本及未执行项。随包 GPU CSV 缺原始 profiler、采样参数和设备软件版本证据，正式性能对比前需补齐。

## 兼容性分析

新增符号及类型，与既有 Sgtsv2 float ABI 独立；A2/A3 和 A5 的公共 Host 代码若发生并行变更，以实际合入顺序解决冲突并做交叉回归。设计尚未获得社区批准；当前代码为正确性原型，完整验收和性能优化仍待完成。

## 参考与未决事项

- 任务源码基线：`cann/ops-sparse` master `e0580fa8991ab473c5cc06259c2c44f2db412aac`，参考 `sparse/gtsv2/arch22` 和对应测试；其 float 接口及参数校验并不替代本任务合同。
- 社区设计流程：[`04_tasks/01_community-task-2026/README.md`](https://gitcode.com/cann/cann-ops-competitions/blob/e1ab79ca0d495afce93dfb1e85f95f546b74d2cf/04_tasks/01_community-task-2026/README.md)。设计须经社区评审；该设计 PR 的批准与最终合入状态需单独核验。
- 拟新建的任务目录和名称可由社区评审调整，关联 Issue 尚待确认；原型以外的快速路径、数值稳定性、指数边界、硬件资源、特殊值及正式性能均须设计评审与实际验证。
