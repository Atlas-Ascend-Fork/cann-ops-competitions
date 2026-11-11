# aclblasChemv（Atlas A2/A3）算子设计

# 需求背景

## 需求来源

- 任务：[11月社区任务-aclblasChemv算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/cbfd47808791498d94595dd9e48500aa)。
- 任务 ID：`cbfd47808791498d94595dd9e48500aa`；贡献者：`AIAFKK`。
- [任务书与测试包](https://www.hiascend.com/p/resource/202611/6faee17b760b4accb74904f9ec97ed3f.zip)，SHA256：`7900794fc773fb50a9980f36aa8b43bb7b4361d877bc2f1dd1e55bbc4b5aafe9`。
- [任务讨论](https://gitcode.com/cann/ops-blas/discussions/80)；[社区任务流程](https://gitcode.com/org/cann/discussions/39)。
- 设计使用[社区模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)，实现目标为 `cann/ops-blas`。

## 背景介绍及现状

`Chemv` 完成单精度复数 Hermitian 矩阵与向量乘加，用于复数线性代数计算。矩阵按列主序存放完整二维数组，但只引用 `uplo` 指定三角；另一三角由共轭对称性隐含，对角虚部不参与计算。

本设计查阅的 ops-blas 基线为 `5bd95145a9c499e43ebad44a313a1029c63d54bc`。该版本尚无公共 `aclblasChemv` 声明及 `test/symv/chemv/` 测试目录。参考 `ssymv` 的 arch22 工程接入、`csymv` arch35 的 Host/Device 标量识别，但不照搬实数对称运算或 950 专用 kernel。

# 需求分析

## 功能与接口

计算 `y = alpha * A * x + beta * y`，所有数值为 `aclblasComplex`（实、虚部各 FLOAT32）。接口添加到公共头文件 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

| 参数 | 内存及形状 | 语义和约束 |
| --- | --- | --- |
| handle | Host 句柄 | 已创建且有效；使用其绑定 stream |
| uplo | Host 枚举 | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n、lda | Host int | `n >= 0`；非空运算 `lda >= max(1,n)` |
| alpha、beta | Host 或 Device 标量指针 | 按运行时指针属性分别识别；不假设两者位置相同 |
| A | Device，列主序 `lda*n` complex64 | 仅指定三角只读；对角虚部忽略 |
| x | Device，逻辑 n 元素 | 物理容量至少 `1+(n-1)*abs(incx)`，支持带符号步长 |
| y | Device，逻辑 n 元素 | 原地输出，物理容量同 incy；`beta=0` 不读旧值 |
| incx、incy | Host int | 非 0、非 INT32_MIN；跨度按 int64 校验 |

输入、输出接口与 cuBLAS `cublasChemv` 参数顺序及 Netlib `chemv` 语义对齐。只支持 lda/incx/incy 描述的布局，无广播、额外视图或任意非连续 Tensor；不要求确定性模式。

## 要求拆解

1. 补齐公共接口及 arch22 Host/kernel，遵守任务书校验顺序、短路和异步语义。
2. 正确处理 Hermitian 共轭、对角虚部、padding、正负步长以及特殊标量。
3. 新建 CSV 驱动 GTest，使用 Netlib BLAS 3.10.0 的 `cblas_chemv` 单标杆，实虚分开严格判定。
4. 保留全部 1200 条官方 CSV（1000 条非性能用例、200 条性能用例），补齐正态分布和交叉边界。
5. 在任务指定环境取得全量精度及逐例性能证据，提供可复现步骤、源码地址与规定格式报告。

# 详细设计

## 算子分析

令 `B` 为根据指定三角解释出的逻辑 Hermitian 矩阵：

```text
stored(i,j) = (uplo == UPPER ? i <= j : i >= j)
B(i,i) = real(A[i + i*lda])
B(i,j) = stored(i,j) ? A[i + j*lda] : conj(A[j + i*lda])  (i != j)
pos(k,n,inc) = inc > 0 ? k*inc : (n-1-k)*(-inc)
y[pos(i,n,incy)] = alpha * sum_j(B(i,j)*x[pos(j,n,incx)]) + beta*y_old[pos(i,n,incy)]
```

矩阵地址及步长偏移使用 64 位计算，避免先在 32 位乘法中溢出。算法计算量 O(n²)、逻辑矩阵读取量 O(n²)，无需创建完整 Hermitian 矩阵副本。

## Host 侧设计

### 参数校验与短路

依次执行以下步骤；顺序本身纳入测试。

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` 立即成功，不访问其他指针、不启动 kernel。
3. 检查 uplo（非法返回 `INVALID_ENUM`），再检查 lda、incx、incy、alpha/beta 非空（非法返回 `INVALID_VALUE`）。
4. 拒绝 0 和 INT32_MIN 步长。以 int64 计算 `(n-1)*abs(inc)`，按任务书引用的同族 arch22 大跨度守卫拒绝超出 `UINT32_MAX` 的逻辑元素跨度；后续地址计算仍为 64 位。
5. 取得标量值或确认其 Device 来源；`alpha=0,beta=1` 时不引用 A/x/y。`alpha=0` 时允许 A/x 为空，其余写回路径要求 y 非空；`alpha!=0` 时要求 A/x 非空。

### 标量位置与异步生命周期

借鉴 `csymv` arch35 的 `ComplexScalarSource`：分别查询 alpha/beta 指针属性，Host 值复制入按值传递的 tiling，Device 值在目标 stream 的 kernel 中读取。正常 Device 标量路径不为了主机判断而无条件同步。

当 A/x/y 缺失，必须确定标量才能给出同步返回码：仅此必要路径先同步 handle 绑定的 stream，再同步回读 Device 标量，判断空指针是否合法。回读在函数返回前完成，避免异步复制引用栈缓冲区的生命周期问题；任何传输或同步错误映射为库错误码，不能把失败当作零标量。Host 和 Device 混合位置分别覆盖。普通路径由 kernel 实现 `alpha=0,beta=1` 短路；Device 标量更新与 kernel 读取依靠同一 stream 顺序保证，不缓存标量数值。

### 分核和 tiling

使用平台查询取得 AIV 核数，按输出行分配互不重叠的工作范围。每个输出元素由唯一核计算并写回，避免跨核原子加和跨核归约。连续输出按 32 字节写回粒度组织行组；带步长路径同样确保写回不覆盖其他核或非逻辑元素，不能只凭逻辑索引不重叠就假定搬运范围无交叉。小 n 限制活跃核数，避免空核；较大 n 在单核内部按列分块计算点积。首版采用单次 kernel 启动，把 `alpha*A*x` 与 `beta*y` 融合写回。

tiling 保存 n、lda、uplo、带符号 incx/incy、Host 标量值、Device 标量标志、核数与分块参数。它不包含依赖临时 Host 对象生命周期的 Device 地址。正常运算无每次调用的矩阵展开、额外 Device 分配或全局工作区。

## Kernel 侧设计

### Init 与搬运

建立 A/x/y 的 GlobalTensor，初始化有界 UB 队列。只有相应分支确实引用数据时才读取它：alpha 为零不读 A/x，beta 为零不读旧 y，对角元单独读取实部。

对输出行 i，沿 j 将访问分成对角线两侧和对角元三个区段。存储在当前列的区段连续搬入，另一侧按 lda 映射收集后在 UB 中整理为连续的复数分量；镜像区段虚部取反。所有搬运范围限制在指定三角和有效元素内，尾块按有效长度搬运，不跨入未引用三角或 padding。带符号 stride 在地址生成阶段映射到正确物理元素，不对负步长直接使用无符号乘法。

### Compute 与写回

复数点积拆成实部 `ar*xr-ai*xi` 和虚部 `ar*xi+ai*xr`，采用 FLOAT32 向量乘加及分块规约。先核验完整用例中累加顺序对 Netlib 单精度标杆的影响；若误差集中于长规约、抵消或大标量族，针对该根因选择分段补偿等路线，并复测受影响全族。不得以更改 golden 或放宽阈值解决精度失败。

最后结合 alpha、beta 得到复数输出；`beta=0` 分支不执行 `0*y_old`，`alpha=0` 分支不执行 `0*(A*x)`，避免引入不应存在的 NaN。写回只覆盖本核拥有的 y 逻辑元素，保留 stride 间隙和外围哨兵；队列和事件保证搬运、向量计算及写回的依赖，不使用未完成的数据。对同一输出没有并行写者。

### 性能优化路径

先建立完整精度/性能基线，依据失败族处理：

| 可能瓶颈 | 数据与判别依据 | 对应优化 |
| --- | --- | --- |
| 小 n 启动占比高 | task duration 随 n 变化、核利用率 | 单次融合 kernel，减小核数与 tiling 开销 |
| 大 n 非连续矩阵读取 | DMA/向量时序与实测带宽 | 调整行块与列块，使复用和连续段搬运增加；保持三角访问语义 |
| 重复读取 x | 分块读取量与 UB 占用 | 核内复用 x 分块，评估双缓冲覆盖搬运 |
| 大规模规约误差 | 实虚逐点误差、最大误差位置及根因族 | 分段规约或补偿，测量精度收益和性能成本 |

分块阈值根据 910B3 真机证据确定，性能统计覆盖所有形状与当前候选。单次局部改善不替代全量验收。

## 工程接入

| 位置 | 计划内容 |
| --- | --- |
| `include/cann_ops_blas.h` | 公共 aclblasChemv 声明 |
| `blas/symv/arch22/chemv_host.cpp` | 参数、标量来源、tiling 和 stream 调度 |
| `blas/symv/arch22/chemv_kernel.cpp` 及必要头文件 | AIV kernel、tiling 及 launcher |
| `blas/symv/README.md` | 接口、约束、Atlas A2/A3 支持表 |
| `test/symv/chemv/CMakeLists.txt` | arch22 调用 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)` |
| `test/symv/chemv/arch22/` | CSV、GTest、封装及采集入口 |
| `test/symv/chemv/` | 参数解析、Netlib golden、判据和复现 README |

沿用 ops-blas 自动源码收集和 GTest 构建入口，不额外建立平行库/API。实施中按职责组织必要文件，避免冗余工具副本。

## 支持硬件和限制

| 项目 | 设计目标与验证安排 |
| --- | --- |
| Atlas A2/A3 | 使用共同 arch22 Ascend C 路线；自验证按任务书覆盖一种款型 |
| 正式性能设备 | Atlas 800T A2（910B3） |
| CANN | 9.1.0 |
| Golden | Netlib BLAS 3.10.0，实际链接的 `cblas_chemv` |
| 内存专项验收 | 任务书 §3.4 不涉及；实现仍应确保边界与生命周期正确 |
| 当前验证状态 | 尚未取得本题真机测试结果；本稿为设计评审，不声明达标 |

# 可维可测分析

## 精度判据

Golden 在 `CblasColMajor` 下调用 Netlib `cblas_chemv`，使用原始单精度输入。实、虚各形成一个长度 n 的向量，分别计算以下指标并同时通过。

| 指标 | 判据 |
| --- | --- |
| 逐有限点匹配 | `abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)` |
| 匹配率 | 有限值比较点 `matched_ratio >= 0.99` |
| 最大绝对误差 | `max_abs_error <= max(1e-2, 32*ULP(g_low))`；g_low 为本分量最大误差点的 FLOAT32 golden，不使用全向量最大值替代 |
| 非有限值 | 两者 NaN、两者同号 Inf，或 NaN 对 ±Inf 均等效；有限/非有限及异号 Inf 失败 |

非有限点从有限统计中剔除并单独记录，必须全部通过；没有有限点时报告有限统计“不适用”，依非有限点规则判定。来源为任务书 §3.2 及[固定精度标准版本](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。特殊值采用本题正文明确规则。MERE/MARE 仅作诊断，不替代上述判据。

## 测试覆盖和参考链验证

保留官方 CSV 的全部 case ID、参数和原始分布；另行补充确定种子的均匀/正态随机用例，使随机类 A/x/y 和 alpha/beta 覆盖均匀 [-5,5] 与正态（μ∈[-5,5]、σ∈[0.1,2]）各 50%，实虚独立采样。特殊值不并入随机分布配额。新增 case 独立编号，并在报告列出分布数量、种子及覆盖。

| 要求族 | 验证内容 |
| --- | --- |
| 基础/尺寸 | UPPER/LOWER、0、1、小质数、2 的幂及 ±1、非对齐、大规模 |
| Hermitian | 手算小矩阵验证共轭符号；未引用三角和对角虚部置 NaN/Inf；指定三角保持有限 |
| 布局 | lda 紧凑/多个 padding、incx/incy 的 ±1/±2/±3；输入只读与 y 间隙哨兵 |
| 标量 | 0/1/纯虚/大值、随机两种分布；Host/Device 四种位置组合，stream 内更新 Device 标量 |
| 短路 | n=0 与无效其他参数组合；alpha0beta1 三数据指针为空；alpha0 无 A/x；beta0 旧 y 为 NaN |
| 异常 | 空 handle、空标量、非法 uplo/lda/n/stride、INT32_MIN、大跨度及错误优先级 |
| 非有限 | NaN、±Inf、有限与非有限交叉；逐分量原始计数与结果 |
| 异步 | 非默认 stream、前置写入与后置读回在同 stream；读回前同步 |
| 参考链 | 已知小矩阵结果和受控错误验证 Golden/判据；性能缺样本、重复映射、错误计时来源回归 |

期望失败和 quick-return 用例只使用单元素 dummy 或空指针，不为非法巨大跨度分配缓冲区。合法负步长运算则按其真实跨度分配。全量运行持续收集普通数值失败；设备异常、越界或挂起时停止受影响部分并保留日志。

构建测试采用 `bash build.sh --ops=chemv --soc=ascend910b` 及仓内 GTest。明确链接 Netlib 3.10.0 的库/头路径并记录构建身份，不能把默认 OpenBLAS 自动发现当作指定 golden 已满足。正式长跑前先测算代表尺寸的生成、golden、拷贝和采集耗时，复用有效输入或结果减少重复成本，完整覆盖仍保留。

## 性能判据与采集

任务书 §3.3 的 msprof op 单算子 task duration 是正式口径。先 warmup 5 次，再采集至少 10 次返回 SUCCESS 的有效调用，取算术平均；异常样本保留并说明。每个 case 均要求 `NPU_avg_us <= gpu_ms*1000/0.8`。

完整保留 `gpu_baseline.csv` 的 200 行及 ID。官方表存在相同参数但不同耗时的重复行，按官方 case 顺序并校验全参数建立逐行映射，禁止以参数字典覆盖前一条基线。若复用相同输入规格的采集，仍分别对所有原始基线判定，不采用较宽值替换较严值。耗时绑定 case、调用、kernel、有效样本数和原始 profiler 文件；若后续引入多 kernel，完整计入每次调用的全部任务，不只挑最短阶段。

附件 README 中的 aclrtEvent 完整调用耗时可作诊断，但不替代正文 task duration。GTest 整数毫秒、Host golden 时间或数据拷贝时间不作为算子性能。精度与性能都针对同一最终代码及动态库；报告逐例 PASS/FAIL/未运行，不只报均值汇总。

## 兼容性与交付

新增共用 API，不改变其他算子的行为；arch22 实现与其他架构按现有构建规则分离。代码、CSV、golden 依赖和 profiler 入口一同提供复现说明，记录源码完整 commit、产物哈希、设备、软件、命令和原始结果。

按任务模板提供设计文档、自测代码、实虚精度报告与截图、性能报告与截图、源码地址及 commit。代码仓根 `task_submission/` 同步规定的七个材料文件；内存项依据正文“不涉及”标注，不制造内存通过记录。设计评审与合入、真机自验、平台验收及实现合入分别记录，当前仅申请本设计评审。
