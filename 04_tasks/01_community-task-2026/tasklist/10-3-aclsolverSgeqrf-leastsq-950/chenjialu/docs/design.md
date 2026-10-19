# Ascend 950 单精度实数 QR 分解与最小二乘接口与验收设计

# 需求背景（required）

本设计对应[10月社区任务-单精度实数QR分解与最小二乘(950)](https://www.hiascend.com/activities/task-center/details/de13b54566654239b5fdba6ccc2b718a?menu=tasks)。接口及验收要求依据任务书与[2026-10-17 官方更新附件](https://www.hiascend.com/p/resource/202610/a909baec14b6404191a241a43c872616.zip)，目标代码仓为 [cann/ops-solver](https://gitcode.com/cann/ops-solver)。

本任务按新版任务书交付 Sgeqrf、Sorgqr、Sormqr、SSgels 四个计算接口及对应的四个 workspace 查询接口。任务方已在[官方回复](https://gitcode.com/cann/ops-solver/discussions/6?parent_id=932bdf1dd132446ca950a36c8e51c898#tid-6fc332c897e24529b739c4ee755f079c)中明确 Xlarft 不做开发验证，本文按四接口的当前必验范围对齐。

| NPU 接口 | 对标接口 | 功能 |
| --- | --- | --- |
| `aclsolverSgeqrf` | `cusolverDnSgeqrf` | 对实矩阵进行 QR 分解。 |
| `aclsolverSorgqr` | `cusolverDnSorgqr` | 根据输入反射器表示生成显式 Q。 |
| `aclsolverSormqr` | `cusolverDnSormqr` | 将 Q 或 Q 的转置作用于矩阵 C。 |
| `aclsolverSSgels` | `cusolverDnSSgels` | 求解方阵或超定最小二乘问题，输出独立解 X。 |

# 需求分析（required）

## 环境与调用约定

目标硬件为 Ascend 950PR，CANN 版本为任务书要求的 9.0.0 及以上，并与目标仓配套；本次测试环境为 CANN 9.1.0、Linux x86_64。算子使用 AscendC/CATLASS 与 ACL runtime，公开接口沿用 ops-solver 的 C API、handle 和调用方 stream。NumPy/SciPy LAPACK 只用于测试，不是 NPU 运行依赖。

调用方创建 handle 并设置 stream，按 `_bufferSize` 查询值分配 workspace，再调用计算接口；测试在 stream 同步后回读结果。查询与执行使用相同参数。前三个接口的 workspace 长度按 FLOAT32 元素数计，SSgels 按字节计。

## 数据布局与输入输出

输入与输出为 FLOAT32，采用列主序，元素地址为 `base[row + column * ld]`。leading dimension 可以大于有效行数；有效区域和 padding 分别校验。

| 接口 | 有效规模与布局约束 | 输出及输入保持要求 |
| --- | --- | --- |
| Sgeqrf | m、n 覆盖 1～4096，支持宽矩形、方阵和长矩形；`lda>=max(1,m)`。 | 原地覆盖 A，上三角或上梯形保存 R，严格下三角保存反射器尾部；TAU 长度为 `min(m,n)`。 |
| Sorgqr | `0<=k<=n<=m`，m 覆盖至 4096；`lda>=max(1,m)`。 | 原地输出 m×n 的 Q，tau 保持不变。 |
| Sormqr | C 为 m×n，m、n 覆盖至 4096；`nq=(side==LEFT ? m : n)`，`0<=k<=nq`，`lda>=max(1,nq)`，`ldc>=max(1,m)`。 | LEFT 计算 `op(Q)*C`，RIGHT 计算 `C*op(Q)`，原地更新 C；A 和 tau 保持不变。 |
| SSgels | `0<=n<=m`，m、n 覆盖至 4096，nrhs 覆盖至 128；`ldda,lddb>=max(1,m)`，`lddx>=max(1,n)`。 | 求解 `min_X ||A*X-B||_F`；dA 允许被破坏，dB 保持不变，dX 独立存储。 |

Sormqr 支持 LEFT/RIGHT 与 N/T/C，实数 T/C 具有相同转置语义。

# 详细设计（required）

## 公开接口


公开类型位于 `cann_ops_solver_common.h`，已有 `aclsolverHandle_t` 和 `aclsolverStatus_t` 继续复用：

```cpp
typedef enum { ACLSOLVER_OP_N=0, ACLSOLVER_OP_T=1, ACLSOLVER_OP_C=2 } aclsolverOperation_t;
typedef enum { ACLSOLVER_SIDE_LEFT=0, ACLSOLVER_SIDE_RIGHT=1 } aclsolverSideMode_t;
```

公开原型位于 `cann_ops_solver.h`：

```cpp
aclsolverStatus_t aclsolverSgeqrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSgeqrf(
    aclsolverHandle_t handle, int m, int n, float *A, int lda,
    float *TAU, float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSorgqr_bufferSize(
    aclsolverHandle_t handle, int m, int n, int k,
    const float *A, int lda, const float *tau, int *lwork);

aclsolverStatus_t aclsolverSorgqr(
    aclsolverHandle_t handle, int m, int n, int k,
    float *A, int lda, const float *tau, float *work, int lwork, int *devInfo);

aclsolverStatus_t aclsolverSormqr_bufferSize(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const float *A, int lda, const float *tau,
    const float *C, int ldc, int *lwork);

aclsolverStatus_t aclsolverSormqr(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const float *A, int lda, const float *tau,
    float *C, int ldc, float *work, int lwork, int *devInfo);

aclsolverStatus_t aclsolverSSgels_bufferSize(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    float *dA, int ldda, float *dB, int lddb, float *dX, int lddx,
    void *dwork, size_t *lwork_bytes);

aclsolverStatus_t aclsolverSSgels(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    float *dA, int ldda, float *dB, int lddb, float *dX, int lddx,
    void *dWorkspace, size_t lwork_bytes, int *niter, int *dinfo);
```

| 参数类别 | 内存位置与行为 |
| --- | --- |
| handle、枚举、维数、leading dimension、workspace 长度 | 这些参数在 Host 上解释。 |
| A、TAU/tau、C、dA、dB、dX | 这些参数指向 Device 数据，Host 不解引用矩阵内容。 |
| Sgeqrf/Sorgqr/Sormqr 的 Workspace/work | 这些参数指向 Device FLOAT32 缓冲区，长度按元素个数计。 |
| SSgels 的 dWorkspace | 该缓冲区位于 Device，长度按字节计。 |
| `_bufferSize` 的长度输出 | 这些参数指向 Host 标量，查询不改变矩阵数据，也不启动数值计算。 |
| devInfo/dinfo | 这些参数指向 Device INT32 标量；写入与数值计算遵守同一 stream 顺序。 |
| niter | 该参数指向 Host INT32 输出标量，报告实际迭代次数。 |

## 参数、状态与边界契约

非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`。带 info 的计算接口在 info 可写时写入 `-i`，i 为去掉 handle 后从 1 开始的参数位置；成功时真实写入 0。info 自身为空时返回错误，不尝试写入。

| 计算接口 | 非 handle 参数的位置（从 1 开始） |
| --- | --- |
| Sgeqrf | m=1，n=2，A=3，lda=4，TAU=5，Workspace=6，Lwork=7，devInfo=8。 |
| Sorgqr | m=1，n=2，k=3，A=4，lda=5，tau=6，work=7，lwork=8，devInfo=9。 |
| Sormqr | side=1，trans=2，m=3，n=4，k=5，A=6，lda=7，tau=8，C=9，ldc=10，work=11，lwork=12，devInfo=13。 |
| SSgels | m=1，n=2，nrhs=3，dA=4，ldda=5，dB=6，lddb=7，dX=8，lddx=9，dWorkspace=10，lwork_bytes=11，niter=12，dinfo=13。 |

合法空问题成功返回，仍按接口要求提供 info、niter 和查询长度输出。不存在有效元素的矩阵指针可以为空；非空问题的 leading dimension 约束保持不变。Sorgqr 在 k=0 且 m、n>0 时输出单位矩阵前 n 列；Sormqr 在 k=0 时不改变 C；SSgels 在 n=0 或 nrhs=0 时写入 niter=0 和成功信息。查询值为零时允许对应 workspace 指针为空。

相同原始输入在同一 stream 上串行重复执行，输出、tau 和 info 必须逐位一致。每次测试恢复原始输入，不将前次原地输出作为下次输入。矩阵有效区域之外的 padding 不得改写。

## 工程与验收入口

公开声明位于 `include/cann_ops_solver.h` 与 `include/cann_ops_solver_common.h`。四接口源码分别位于 `src/sgeqrf/`、`src/sorgqr/`、`src/sormqr/` 和 `src/ssgels/`；正式测试位于 `test/` 下同名目录，接口说明位于 `docs/zh/` 下。构建、加载和测试沿用目标仓标准流程，不要求调用方改变矩阵布局或使用额外绑定接口。

私有验收分支在 `src/sgeqrf/acceptance/` 集中提供自测用例、自测代码、自测说明及验收操作指南。该目录仅服务本次验收，验收通过后、提交上游代码 PR 前删除，正式测试不依赖该目录。验收人员从私有分支获取最新提交，不要求检出固定 commit。

# 特性交叉分析

| 交叉特性 | 验收检查 |
| --- | --- |
| 输入输出与原地修改。 | 按各接口契约检查可写区域、只读输入与独立输出。 |
| leading dimension 与 padding。 | 分别核对有效元素和 padding 哨兵，不以紧凑布局替代调用方布局。 |
| side/trans 与矩阵形状。 | 覆盖合法枚举组合、矩形输入、k<n 和非法组合。 |
| workspace 与动态 shape。 | 查询不修改输入；按当前调用查询值分配后能正确执行。 |
| stream 与确定性。 | 在非默认 stream 上恢复原始输入重复执行并逐位比较。 |

# 可维可测分析

## 精度标准与测试方法


测试保留原始 A、B 和 C，并用任务书指定的 CPU FLOAT32 参考链路计算同 case 的参考残差。`epsilon=2^-24`；矩阵 1-范数是各列绝对值之和的最大值，向量 1-范数是元素绝对值之和。

| 接口 | 残差与通过标准 |
| --- | --- |
| Sgeqrf | `r=||R-Q^T*A_original||_1/(m*||A_original||_1*epsilon)`，要求 `r<=max(5*r_cpu,0.1)`。Q 由 NPU 反射器升精度到 FLOAT64 后在 CPU 构造为完整 m×m 矩阵。 |
| Sorgqr | `r=||I-Q^T*Q||_1/(m*epsilon)`，要求 `r<=max(5*r_cpu,1.5)`。 |
| Sormqr | `r=||op2-op1||_1/(nq*||C_original||_1*epsilon)`，要求 `r<=max(5*r_cpu,0.1)`。op1 是被测输出；op2 使用同一份 NPU 反射器由 Sorgqr 生成完整 Q 后计算。nq 是 Q 的阶数，即 LEFT 时为 C 的行数，RIGHT 时为 C 的列数。 |
| SSgels，m=n | 对每个右端项 j 计算 `r_j=||B_j-A*X_j||_1/(max(m,n)*||A||_1*||X_j||_1*epsilon)`，取最大值，要求 `r<=max(5*r_cpu,0.015)`。 |
| SSgels，m>n | `R=B-A*X`，`r_opt=||R^T*A||_1/(||A||_1*||B||_1*max(m,n,nrhs)*epsilon)`，要求 `r_opt<=10*r_opt_cpu`。 |

Sormqr 按任务书在 Sorgqr 通过后验证，避免使用未经验证的参考 Q。分母为零的特殊输入按所引用 LAPACK 测试例程的规则处理，不能通过产生 NaN 或跳过比较判为通过。

## 测试范围与标准流程

使用官方更新附件内未经修改的生成器、冻结 CPU 参考、精度判据与 GPU 基线，按 case_id 对应每条结果。四接口共 694 条精度用例与 12 条非法参数检查：

| 接口 | 精度及性能用例数 | 非法参数检查数 |
| --- | ---: | ---: |
| Sgeqrf | 174 | 3 |
| Sorgqr | 165 | 3 |
| Sormqr | 189 | 3 |
| SSgels（官方目录 sgels） | 166 | 3 |
| 合计 | 694 | 12 |

补充检查覆盖空问题、最小规模、padding、k=0、k<n、nrhs=0、1、8、32、128、workspace 不足、非法指针和枚举、非默认 stream、只读输入以及重复执行的一致性。仓内保留的 Xlarft 代码与样例仅为附加兼容内容，不增加本任务必验项。

下面的标准构建同时保留已验证的 Xlarft 附加样例构建项；任务全量自测仍只覆盖上述四接口。按目标仓 [QUICKSTART](https://gitcode.com/cann/ops-solver/blob/master/docs/QUICKSTART.md)及 `build.sh` 的实际参数执行标准构建：

```bash
bash build.sh --soc=ascend950 --ops=sgeqrf,sorgqr,sormqr,xlarft,ssgels
```

在准备好 Python 测试依赖后，仓库标准 `build.sh --run` 运行默认样例；该样例不代表任务全量覆盖。完整验收按照私有分支 `src/sgeqrf/acceptance/ACCEPTANCE.md` 的单一路径操作，使用公开 C API 调用、官方验证脚本和 CANN 官方性能采集工具。报告记录环境、用例、动态库身份、判定与日志，开发者自测与最终验收结论分别说明。

## 性能标准与计时口径

每条用例关联官方更新包 `package/perf_baseline.json` 中的 `avg_ms`，逐项要求 `T_NPU <= T_GPU / 0.35`，等价于 `T_GPU/T_NPU >= 0.35`。四接口全部 694 条有效基线均纳入，原缺失的 108 条 SSgels 已包含，不用估计值替代正式基线。

性能测试使用同 shape、FLOAT32、列主序和一致的输入恢复方式，独立标记预热与测量轮。使用 CANN 9.1 官方 `msprof op` 的 BasicInfo 采集，从官方导出的 `OpBasicInfo.csv` 读取 `Task Duration(us)`；通过官方 MSTX 范围选择待测计算 API。每次公开 API 的全部 Vector/Cube kernel 累加后，按测量轮求算术平均；保留逐 kernel 明细及调用关联，不以某个局部 kernel 时间代表完整接口。

Host 内存分配、拷贝、workspace 查询和 CPU 参考生成不计入 kernel 时间。Sorgqr/Sormqr 的输入准备不计入相应接口，SSgels 本身需要的全部 NPU 计算计入完整接口。任务书未要求内存性能指标；如提供缓冲区容量或内存保护区检查，单列其方法与覆盖，不冒充峰值内存或其他工具结果。

## 待对齐事项

SSgels 数值失败的状态及正 dinfo 编码仍以任务书补充或评审结论为准，不据此排除任务书范围内的合法输入。四接口范围、950PR 目标、精度要求和 0.35 性能门限保持任务书规定。

## 参考资料

1. [官方任务](https://www.hiascend.com/activities/task-center/details/de13b54566654239b5fdba6ccc2b718a?menu=tasks)及[官方更新附件](https://www.hiascend.com/p/resource/202610/a909baec14b6404191a241a43c872616.zip)定义接口和验收标准。
2. [社区任务提交说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)与[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)提供提交要求。
3. [ops-solver 快速入门](https://gitcode.com/cann/ops-solver/blob/master/docs/QUICKSTART.md)与[构建脚本](https://gitcode.com/cann/ops-solver/blob/master/build.sh)提供标准构建和默认测试流程。
4. [官方 msOpProf 指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)提供官方性能采集流程。
5. [cuSolver DN 文档](https://docs.nvidia.com/cuda/cusolver/index.html)用于核对公开接口语义。
