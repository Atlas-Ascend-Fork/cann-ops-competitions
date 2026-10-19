# Atlas 950PR 单精度实数 LU 分解与批量求逆算子设计

任务编号：`10-1-aclsolverSgetrf-batch-950`。团队及提交人：`gai0718`。日期：2026-10-18。

# 需求背景（required）

## 需求来源

本设计对应 10 月社区任务“单精度实数 LU 分解、求解和批量求逆（950）”。以任务书 `Atlas950_Sgetrf_Sgetri_SgetriBatched_SmatinvBatched_task_doc.md` 的 §2.3 接口、§3 验收标准及同包各算子的 `package` 为依据。目标仓库为 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，目标设备为 Ascend 950PR，工程模式为 Host C API 加 AscendC/CATLASS Kernel 直调。

## 背景介绍

LU 分解和求逆共享三角因子与主元。Sgetrf 从原矩阵产生因子，Sgetri 与 SgetriBatched 使用已有因子，SmatinvBatched 对小原矩阵直接求逆。四个计算接口和两个 workspace 查询一起交付。

源码调研基线为 `cann/ops-solver@b0a2247317f7a1f01341b247c8fa5d7d9d9d7e2e`。

| 调研项 | 基线行为 | 本次设计 |
| --- | --- | --- |
| Sgetrf/Sgetri 公开接口 | `aclError`、旧参数、Host 数据及内部搬运 | 任务书规定的 `aclsolverStatus_t`、32 位 `int`、Device 数据 |
| leading dimension | 实数原型要求 `lda==n` | 按有效行数校验，以实际 ld 寻址 |
| Sgetri 输入 | 原型内部重新执行分解 | 直接消费 packed LU 与主元 |
| workspace | 内部申请，无公开查询 | 单矩阵由调用方查询和分配 |
| 950 适配 | 构建有 `ascend950/dav-3510/arch35` 映射；旧 LU 使用硬件同步地址和固定分核 | 为 arch35 建立分核、搬运和同步路径 |
| 批量实数接口 | 待新增 | Device 指针数组、逐矩阵 info、独立输出 |

# 需求分析（required）

## 需求描述

公开原型逐项采用任务书 §2.3：

```cpp
aclsolverStatus_t aclsolverSgetrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverSgetrf(
    aclsolverHandle_t handle, int m, int n, float *A, int lda,
    float *Workspace, int *devIpiv, int *devInfo);
aclsolverStatus_t aclsolverSgetri_bufferSize(
    aclsolverHandle_t handle, int n, float *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverSgetri(
    aclsolverHandle_t handle, int n, float *A, int lda, int *devIpiv,
    float *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverSgetriBatched(
    aclsolverHandle_t handle, int n, const float *const Aarray[], int lda,
    const int *PivotArray, float *const Carray[], int ldc,
    int *infoArray, int batchSize);
aclsolverStatus_t aclsolverSmatinvBatched(
    aclsolverHandle_t handle, int n, const float *const A[], int lda,
    float *const Ainv[], int lda_inv, int *info, int batchSize);
```

在 `cann_ops_solver_common.h` 补充 `aclsolverOperation_t`，N/T/C 分别为 0/1/2。以上接口没有 trans、nrhs 或 B 参数，开发范围采用其明确签名。

矩阵为 FLOAT32，主元和 info 为 INT32。A、workspace、主元、info、批量地址数组及数组指向的矩阵均位于 Device。查询输出 Lwork 位于 Host，单位为 float 元素数。列主序元素地址为 \(A_{r,c}=A[r+c\,lda]\)。Sgetrf 有效尺寸为 \(m\times n\)，其余为 \(n\times n\)，独立校验每个输入/输出 ld。

| 接口 | 核心语义 | 支持范围 |
| --- | --- | --- |
| Sgetrf | 原地输出 L/U；可选部分主元；1-based ipiv | m 为 0…4096，n 为 0…8192，覆盖长宽矩形和 padding |
| Sgetri | 从 LU 与主元原地输出逆 | n 为 0…4096，支持无主元 |
| SgetriBatched | 从只读 LU 指针数组写独立 C | n 为 0…4096，batch 为 0…1000000 |
| SmatinvBatched | 从只读原矩阵写独立逆矩阵 | n 为 0…32，batch 为 0…1000000 |

各接口支持任意满足最小行跨度要求的 ld，设备分配容量由调用方提供。内部所有乘加、字节数、batch 偏移采用经溢出检查的 64 位计算。

## 需求拆解

1. 统一六个公开函数、返回状态、参数编号及 Device 内存归属。
2. 实现矩形 LU、确定性部分主元、已有 LU 的求逆与小矩阵融合批量求逆。
3. 支持 padding、非连续矩阵地址、空问题、奇异和非法参数的 info。
4. 使用调用方 stream，建立临时资源在多流下的有效期。
5. 按当前 package 原始造数和冻结参考运行精度、info、确定性及逐用例 msprof 性能自测。

# 详细设计（required）

## 算子分析

### 数学公式与主元

LU 满足 \(PA=LU\)。严格下三角存储 L 的非对角元，L 对角隐含为 1；上三角及对角存储 U。第 k 步主元为剩余行中绝对值最大且行号最小的元素，记录 \(ipiv[k]=p+1\)，并交换整行。无主元路径令 \(P=I\)。

已有 LU 求逆执行 \(LY=P\)、\(UC=Y\)。构造单位阵后正序回放主元，前代求 Y、回代求 C。重构原矩阵用于残差检查时，先计算 LU，再逆序回放行交换。

### 分块 LU

当 \(\min(m,n)>256\)，或 \(m>64\) 且 \(n\ge4m\) 时，采用 64 列外层面板、8 列内部面板，尾随更新使用 Cube 矩阵乘法。其它形状使用 16 列面板（m≤2048）或 8 列面板；适用的矮宽矩阵将左侧方块分解后按输出列块并行求解。面板内每列固定次序选主元、交换、缩放和更新。完成后计算 \(U_{12}=L_{11}^{-1}A_{12}\)，更新 \(A_{22}\leftarrow A_{22}-L_{21}U_{12}\)。面板循环次数由 \(\min(m,n)\) 决定，矩形和尾块按有效范围屏蔽。

面板主元搜索由单个 AIV 执行，候选比较先绝对值后最小行号；行交换、三角求解及尾随更新再按列块分核。历史 L 列与尾随列一起交换。零主元记录首个正 info，跳过该列的除零，继续形成确定的分解输出。

### 已有 LU 求逆

Sgetri 先把有效 LU 完整保存到调用方 workspace，内部行跨度按 8 个 float 对齐，再按 RHS 列块生成置换单位阵，执行前代和回代，写回 A 的有效列。n≤128 使用小规模求解，5…32 阶复用批量小逆路径；较大有主元问题采用分块求解与 Cube 更新，无主元问题采用补偿点积求解。只读副本覆盖完整 LU 的后续生命周期。

SgetriBatched 直接读取每项 LU 和 `PivotArray+i*n`，将结果写到相应 C。n≤32 使用融合向量路径；33…128 阶使用普通或 tiled 中阶路径，普通路径 RHS 宽度为 64。n>128 使用受 512 MiB scratch 预算约束的 chunk，三角对角块在 AIV 求解，块间贡献使用固定次序的 Cube 矩阵更新。输入 LU 与 padding 保持原值。

### 小矩阵批量路径

n≤32 时每个矩阵在一个任务内完成读取、计算和写回。SmatinvBatched 融合部分主元 LU 与两次三角求解；SgetriBatched 使用传入 LU。紧密布局的 2/4/8/16 阶大批量使用专用向量路径，其余尺寸与 padding 使用通用分组路径。2×2 路径使用向量指令生成地址校验索引，并以 FMA 残差加 Newton 乘法修正倒数。分散矩阵地址进入逐项路径。

每核处理固定矩阵区间，矩阵间独立。分组数量 q 由 UB 预算决定：至少计算 LU、RHS/结果、主元、归约临时区、掩码、双缓冲和 API scratch 的同时存活字节数；超预算时减小 q 或切换分块路径。最后一组仅处理有效 batch。

向量掩码接口 `UpdateMask(uint32_t&)` 会递减其计数参数。LU 尾随更新和三角求解保留原始有效 lane 数，另以 remaining 创建掩码，保证至少一个完整向量的分组也执行全部数学更新。25…30 阶使用 32 个矩阵分组时，按实际核数处理完整波；不足半波的尾部以 8 或 16 个矩阵分组另行下发。两段使用调用方同一 stream，覆盖完整 batch，并将全部内核耗时相加。

18…32 阶原矩阵求逆在输入、输出均紧密连续、整组 info 均为 0 且启用融合路径时，将输出按每项 \(n^2\) 个 float 连续打包。预先缓存紧凑槽和对齐槽两套索引，紧凑组以一次 DataCopyPad 写出 \(count\cdot n^2\) 个 float。奇数阶紧凑槽减少 UB bank 冲突及 DMA 分段；padding、分散指针和含失败项的组使用对齐槽及逐项写出。每组重新选择布局，保持奇异矩阵输出、输入和调用方 padding 原值。

## 算子实现

### 3.2.1 Host 侧设计

#### 校验、状态及 info

复用 create/destroy/set/get stream。空 handle 返回 HANDLE_IS_NULLPTR；资源申请失败返回 ALLOC_FAILED；维数、ld、必需指针和可知 workspace 长度错误返回 INVALID_VALUE。devInfo 可用时，在调用方 stream 上以短 kernel 写负参数编号。

| 计算接口 | 不计 handle 的参数编号 |
| --- | --- |
| Sgetrf | m=1，n=2，A=3，lda=4，Workspace=5，devIpiv=6，devInfo=7 |
| Sgetri | n=1，A=2，lda=3，devIpiv=4，Workspace=5，Lwork=6，devInfo=7 |
| SgetriBatched | n=1，Aarray=2，lda=3，PivotArray=4，Carray=5，ldc=6，infoArray=7，batchSize=8 |
| SmatinvBatched | n=1，A=2，lda=3，Ainv=4，lda_inv=5，info=6，batchSize=7 |

先校验 handle、负维数、合法 ld 和 info，再分空问题。m=0 或 n=0 的单矩阵写 info=0；n=0、batch>0 的批量逐项写 0；batch=0 返回成功且无需访问零长数组。非空路径验证必需数据指针。nullptr 主元选择无主元路径。SmatinvBatched 在 n>32 时按第 1 参数报错。

成功计算写 info=0，数值奇异写首个 U 对角为 0 的 1-based 位置。批内奇异项单独结束求逆，其余项继续。Device 中的主元按每步合法范围检查；指针数组元素判空并以 64 位地址区间检查同项输入输出重叠。各输出由独立任务写入，输入矩阵和输出矩阵之间的跨项别名按调用契约排除。

Host 可见标量错误立即返回。Device 内容检查在调用方 stream 上执行并写负 info；其同步返回状态与异步执行的边界列在末尾评审事项中。

#### Workspace 和资源生命周期

查询和执行共用 workspace 尺寸计算函数，依据算法分支计算对齐行跨度和各段偏移。矩阵行跨度按 8 个 float 对齐，大规模暂存段按 512B 划分，并预留 Cube scratch。Lwork 为所需字节数换算的 float 元素数，超出 int 表示范围返回错误。

| 接口 | 空间组成 |
| --- | --- |
| Sgetrf | 普通路径为 0；分块路径为 64×n 个 float 的负 U 暂存及 Cube scratch |
| Sgetri | n≤128 为两份对齐矩阵；较大问题为四份矩阵、状态区及 Cube scratch |
| SgetriBatched | n>128 使用 handle 内部的有界 batch chunk 与求解 scratch，小 n 使用 UB |
| SmatinvBatched | 小矩阵、主元及 RHS 驻留 UB，结果与 info 写入调用者缓冲 |

查询只依赖形状、布局和平台，不读取 A 内容。Sgetri 检查 Lwork 至少达到查询值；Sgetrf 调用方按查询值提供空间。合法空问题查询返回 0。workspace 与矩阵、主元、info 分离。

内部资源按创建 context、stream 和容量管理。槽位与 stream 绑定，同流入队顺序允许复用足够容量的槽位；切换 stream 后使用独立槽位。最后一个使用 kernel 后记录事件，正常返回路径保留在途缓冲。SetStream 切换保留旧流槽位；Destroy 等待自身完成事件后回收，调用方 stream 由调用方销毁。互斥锁保护 workspace 选择及相关入队过程。

#### 平台与工程接入

使用平台查询的核数、UB/L1 容量，构建目标为 `ascend950`、`dav-3510`、`arch35`。面板、求逆、批量任务按运行时形状和可用资源分派。初版以同 stream 的阶段 kernel 边界表达全局依赖，满足正确性后再评估融合。

沿用仓库的 `build.sh` 和 CMake 源码收集方式，更新公开头、Sgetrf/Sgetri 声明定义及测试调用。单矩阵入口保留在原 `src/sgetrf/`、`src/sgetri/`，批量入口和公共数值组件集中于 `src/lu_real/`，950 内核位于 `arch35/`；四个算子的测试和 adapter 保留各自标准目录。单配置默认采用 Release，适配器显式启用 Host 转换优化；ASC 内核头文件加入增量构建依赖。

### 3.2.2 Kernel 侧设计

矩阵访问始终使用实际 ld；内部 tile 可重排为适合向量或矩阵指令的布局。CopyOut 只覆盖有效行列。UB 中向量目标、行 pitch 和临时段满足实际指令对齐，GatherMask 的整块写范围单独保留。

AIV 完成面板、主元、三角对角块和小矩阵路径；AIC/CATLASS 或 AscendC Matmul 执行尾随更新。首先验证完整 FLOAT32 乘加路径的残差，再选择满足标准的设备矩阵路径。小主元使用安全除法，保持非零主元及 subnormal 的数值语义。

MTE2、V、MTE3 之间显式建立事件依赖。双缓冲计算完整同时存活容量。每个输出 tile 唯一写者，三角递推与 K 块次序固定。主元 tie-break、首个失败 info 的写者与顺序固定，控制区在每次调用初始化，实现同输入、同 stream 串行重跑逐位一致。

小矩阵求逆在 Scalar 读取 Device 地址数组后，用 `S_MTE2` 事件同步，再把同一 UB 段用于矩阵搬入。针对 n=1、batch=8 的实际插桩 WAR 报告补齐这一依赖，逐项执行内存及竞争检测复核。

## 支持硬件

本次适配和自验使用 Ascend950PR、驱动 25.7.rc1.10、CANN 9.1.0。最终报告记录软件包、编译器、驱动、NPU 型号和仓库 README 配套关系。

## 算子约束限制

采用任务书公开六个函数，矩阵为列主序 FLOAT32。批量地址数组可以指向分散分配的矩阵；输入与输出独立。调用方保持所有 Device 数据及 workspace 有效直到 stream 完成。不涉及广播、图融合或 PyTorch 封装。

# 可维可测分析

## 精度标准/性能标准

### 当前附件与覆盖清单

本次附件内 `canonical_cases.json` 与 `perf_baseline.json` 审计结果如下。任务书 §3.5 指定 package 内有效用例为实际自验清单。

| package | 精度例数 | 有效 GPU 性能基线 | 额外 info 契约 |
| --- | ---: | ---: | ---: |
| sgetrf | 99 | 99 | 3 |
| sgetri | 200 | 200 | 3 |
| sgetribatched | 150 | 150 | README 指定 1 |
| smatinvbatched | 171 | 171 | README 指定 1 |
| 合计 | 620 | 620 | 8 |

原始 ZIP 中的两个批量 `cases/index.json.gz` 是 Git LFS 指针。已通过 `cann/ops-solver.git/info/lfs/objects/batch` 下载真实对象，文件大小分别为 25,988,089 和 41,480,274 字节，SHA-256 均与指针 OID 一致；本地与 950PR 材料目录已替换为实际文件，并保留指针备份。验证器读取这些文件中的冻结 sample_map、ratio_cpu 和 info 参考。单矩阵 index 已包含参考值。原始 cases/bench 和 package 均保留，分别记录版本与指纹。

精度接入使用附带 solver_adapter v2 协议：Host 紧凑行主序传输数据在 reset 阶段转换为 Device 列主序和真实 padding，execute 仅调用目标公开 API，readback 在 harness 同步后读取结果、主元和 info。不同算子的 descriptor 指向对应公开函数地址。准备/恢复输入、前置 LU 和数据搬运位于目标 kernel 计时之外。

单矩阵按包内 `gen_data.py` 原始规则造数，已有 LU 求逆使用该包生成的 LU32/ipiv。批量按 k≤5 代表内容与完整 sample_map 现场展开；目标 API 接收原规格整批输入。读取完整输出，先检查同内容所有槽位逐位一致，再检查代表内容残差；info 契约逐槽比对。分核、算法和数据路径只依赖输入参数及平台资源。

验证程序可通过 `LU950_CACHE_HOST_TRANSPOSE=1` 缓存 Host 行列布局转换。它逐位比较完整 Host 输入以判断缓存命中，每次 reset 上传完整 Device 输入及主元，并恢复输出及 info；缓存保存紧凑输入和列主序副本，开关记录于实测元数据。新增回归覆盖大分组随机矩阵、缓存命中、输入内容变更和 padding。

验证脚本保留首次输出副本，最后一次回读复用输出数组，并以 1 MiB 字节块比较完整结果，覆盖 signed-zero 和 NaN payload。profiler 在内存中仅保存关联与求和所需的4列，完整原始 CSV 继续归档。按容器实际 cgroup 内存限额选择采集分组，物理 batch、10次预热、25次正式采样及全部内核求和规则保持一致。

### 残差

矩阵 1-范数取最大列绝对值和，单位舍入为 \(\varepsilon=2^{-24}\)。LU 逆序回放主元还原矩阵；已有 LU 求逆以同一因子和主元还原 A。

$$
r_{LU}=\frac{\|P^{-1}LU-A\|_1}{n\|A\|_1\varepsilon},\qquad
r_{inv}=\frac{\|I-CA\|_1}{n\|A\|_1\|C\|_1\varepsilon}.
$$

| 接口 | 精度阈值 |
| --- | --- |
| Sgetrf | \(r\le\max(5r_{cpu},0.3)\) |
| Sgetri/SgetriBatched | \(r\le\max(5r_{cpu},0.05)\) |
| SmatinvBatched | \(r\le\max(5r_{cpu},0.5)\) |

使用包内原始判定脚本和冻结 FLOAT32 LAPACK 参考值，记录现场 NumPy/SciPy/OpenBLAS、CPU 指令路径和线程配置。奇异/info 用例按契约单列。自构造用例覆盖均匀/正态、有/无主元、等值主元、奇异、小主元、padding 与非连续指针数组。

### 性能

使用 CANN 9.1 的 `msprof --type=db` 采集完整内核记录，再由官方 exporter 导出 `op_summary*.csv`；每例保留独立 stream，按实际 stream ID 关联用例。每次目标 API 所需的全部初始化、校验、重排和计算 kernel 按调用求和，再对全部正式迭代取算术均值。预热 10 次，采样 25 次；GPU 原始记录的 iters 为 25。保留完整原始采集数据与全部样本。GPU `avg_ms` 和 NPU 的 us 统一单位，逐例要求 \(T_{NPU}\le T_{GPU}/0.35\)。Sgetri/SgetriBatched 的前置分解不计入求逆。预热和恢复输入独立标记，每次迭代保持同一初始状态。

每例记录参数、seed、输入/库/源码指纹、CPU/NPU ratio、info、确定性、GPU 时间、NPU 时间、倍率、判定及原始日志路径。保留慢例和失败记录，优化后重跑受影响范围。内存按任务书 §3.4 记录 workspace、内部 scratch 和回收情况。

### 边界与复现

补充 m/n/batch=0、n=1/32/33、负维数、非法 ld、必需指针为空、workspace 少一个元素、无主元、主元非法、输入输出重叠、混合奇异 batch、padding 哨兵、非默认流、切换流和重复 bit-wise 检查。测试 README 给出标准构建、adapter、精度、性能命令和环境依赖。

已完成的连续输出回归覆盖 18/21/27/29/31/32 阶、batch=4097 的多波和尾组，混合奇异组、带 padding 和分散指针，以及 27 阶 batch=1777 的较大分组。输入、输出、指针数组和 info 分配前后各设 128 个哨兵元素，重复三次逐位比较，并检查 FLOAT32 LAPACK 参考及 FP64 残差。已有定向插桩记录随附；附加检查的固定数量不作为交付条件。

## 兼容性分析

Sgetrf/Sgetri 原型升级涉及 API/ABI，统一迁移声明、实现、示例、测试及文档。新增批量函数和 workspace 查询；其它算子继续使用原有接口。头文件、实现、导出符号及 adapter 一起校验。新代码遵循仓库许可证与贡献规范。

## 当前自验结果与版本来源

交付代码为 [gai0718/ops-solver-lu950@6f970a7](https://gitcode.com/gai0718/ops-solver-lu950/commit/6f970a7865634216f8531862c18858e9f6f771f8)。Ascend-CANN 已有私仓开发者权限。

已完成的全量基线 `bf19edc073ed3577a071b0097d48b3a2375104bb` 通过620项精度和8项info，性能619/620，唯一超时为 smatinvbatched-0154。连续输出优化以该基线的候选补丁构建并完成以下定向复测，随后纳入交付提交 `6f970a7`：

| 用例 | n / batch | NPU均值（us） | 门槛（us） | GPU/NPU | 精度、确定性、性能 |
| --- | --- | ---: | ---: | ---: | --- |
| smatinvbatched-0152 | 29 / 16384 | 1203.03528 | 1292.0 | 0.375883 | 通过 |
| smatinvbatched-0154 | 29 / 524288 | 36244.63792 | 38256.285714 | 0.369426 | 通过 |

交付报告复用完整基线并采用上述已有定向复测，按case_id汇总为精度及info628/628、性能620/620；每条记录注明实际构建版本。原始全量报告、历史失败、候选补丁及两项复测的完整采样均保留。各项均执行原规格完整batch，预热10次、采样25次，沿用原包检查器及性能门槛。

任务书§3.4明写“内存不做要求”，交付内存报告据实记录已有观测。已完成的功能及定向诊断作为补充材料，不将附加固定数量或完整插桩设为交付条件。

## 评审需确认事项

1. Device 指针数组元素和主元内容由同流 Device 校验，内容错误写入负 info；Host 返回状态报告参数校验和下发结果。
2. 多 kernel API 采用“每次调用全部 kernel 求和后取均值”的性能口径。


## 参考资料

- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)、[社区流程 #39](https://gitcode.com/org/cann/discussions/39)、[本任务讨论 #2](https://gitcode.com/cann/ops-solver/discussions/2)。
- 已合入设计 [PR #2174](https://gitcode.com/cann/cann-ops-competitions/pull/2174) 和 [PR #2251](https://gitcode.com/cann/cann-ops-competitions/pull/2251)：参考接口契约、确定性主元、分块和资源生命周期；用例统计使用本次 package 重新审计结果。
- 计算实现基于 xlvrNlfxKe 的公开 [ops-solver PR #106](https://gitcode.com/cann/ops-solver/pull/106)，提交 `b5c581244e6c46019b4ba66e0cfc844b02e93583`。保留原版权与 CANN Open Software License Agreement Version 2.0；本分支补充接口兼容、正式适配器、2×2 向量索引与倒数修正、Release 构建及增量依赖，并使用当前官方包重新自验。源码来源说明随代码交付。
- [cuSolver](https://docs.nvidia.com/cuda/cusolver/index.html)、[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)、[LAPACK getri](https://netlib.org/lapack/explore-html/d8/ddc/group__getri.html)、[CATLASS](https://gitcode.com/cann/catlass)。
