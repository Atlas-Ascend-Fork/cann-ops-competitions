# 需求背景（required）

## 需求来源

10月社区任务 aclblasHgemmStridedBatched 算子开发（950）。报名团队：m0_52603995。
目标环境为 Atlas 950PR、CANN 9.1.0；实现集成至 cann/ops-blas。
源码基线：master，095ed7d0fba93ab48f2572705cd0c929d8cdf358。
本文件为开发设计，后文验证步骤属于计划，实际结果以可追溯自测报告为准。

## 背景介绍

`aclblasHgemmStridedBatched` 对固定步长定位的一批半精度矩阵执行
`C_b = alpha * op(A_b) * op(B_b) + beta * C_b`。
列主序 BLAS 的 leading dimension 和 batch stride 分别表示列间距、批间距；两者均以元素计。
现有仓库包含 FP32 strided batched GEMM、GEMM batched 的 FP16 dtype traits、arch35 Cube 搬运与累加模板，
但公共头文件没有本任务要求的半精度接口和 `aclblasHalf` 类型。

实现以仓库句柄式 BLAS 工程为基础，复用同族 Cube 模板，不引入额外框架入口。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasHgemmStridedBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, long long strideA,
    const aclblasHalf* B, int ldb, long long strideB,
    const aclblasHalf* beta,
    aclblasHalf* C, int ldc, long long strideC,
    int batchCount);
```

公共声明放在 `include/cann_ops_blas.h`，共用类型 `aclblasHalf` 表示 IEEE 754 binary16 的16位存储，
在 `include/cann_ops_blas_common.h` 中定义。Host 用 ACL 的半精度转换读取标量；device 按 half 解释相同位布局。
alpha/beta 是 Host 指针，矩阵是 Device 指针；通过 handle 绑定的 stream 异步提交计算。
不新增 950PR 专有平行 API，不将矩阵指针数组作为本接口输入。

## 需求拆解

| 项目 | 行为 |
|---|---|
| 数据类型 | A/B/C 和 alpha/beta 均为 FP16；乘积累加与 epilogue 使用 FP32 |
| 布局 | column-major；A 的物理形状为 m×k 或 k×m，B 为 k×n 或 n×k，C 为 m×n |
| 转置 | N/T/C 的九种组合；实数 C 与 T 等价，映射为四种实际 kernel 布局 |
| 前导维 | lda≥max(1,A物理行数)，ldb≥max(1,B物理行数)，ldc≥max(1,m) |
| batch定位 | 首地址加 batchIndex×stride；strideA/B/C 为非负 long long |
| no-op | m=0、n=0或batchCount=0，在必须参数校验后成功返回，不访问A/B/C |
| 退化 | k=0只执行C=beta*C；beta=0直接写零，不读取旧C；alpha=0使用相同缩放路径 |
| 指针 | alpha/beta始终非空；活动计算且k>0时A/B必须非空，与alpha取值无关 |
| 输出指针 | 活动计算且k>0或beta≠0时C必须非空；k=0、beta=0且C为空按任务§2.4成功返回 |
| 原地 | 只更新C各矩阵的m×n逻辑元素，保留leading dimension padding及批间空隙 |
| 范围 | 新增 arch35 实现和公共声明；其他产品线继续共用声明，不声称已实现其kernel |

# 详细设计（required）

## 算子分析

### 数学公式

对批号 `b`、输出行 `i`、输出列 `j`：

```text
A_b = A + b * strideA
B_b = B + b * strideB
C_b = C + b * strideC
Aop(i,p) = transa==N ? A_b[i + p*lda] : A_b[p + i*lda]
Bop(p,j) = transb==N ? B_b[p + j*ldb] : B_b[j + p*ldb]
C_b[i+j*ldc] = half_RNE(alpha*sum_p(float(Aop)*float(Bop)) + beta*float(C_b[i+j*ldc]))
```

beta=0 的实际实现跳过旧 C 的加载和乘法，避免旧 C 中 NaN 污染结果。
alpha=0 或 k=0 按 BLAS 缩放分支处理。FP16 输入先按值解释，Cube 使用 FP32 累加，最后只进行一次 FP16 舍入。

### 支持形状

m/n/k/batchCount 是运行时参数；支持非对齐尺寸、非方阵、前导维 padding、非紧凑及零输入步长。
输入 stride 小于矩阵存储宽度仅导致共享只读数据，不需要复制或拒绝。
任务书要求覆盖 strideC 小于矩阵宽度及0，而随包 README 将重叠输出称为未定义并不生成相关用例；
本设计对这些输入采用按 batch 升序在同一 stream 串行排队的行为，避免跨 batch 写竞争。
本轮验收范围按随包 README 排除多 batch 输出重叠；不将该类输入加入正式精度/性能用例。上述保序是实现的防竞争措施，不声称该场景已经通过真机验证。

## 算子实现

### Host侧设计

校验顺序：handle → transa/transb枚举 → m/n/k和batchCount → lda/ldb/ldc → 非负stride → alpha/beta判空 → no-op → 活动计算矩阵指针。
返回码分别使用 `HANDLE_IS_NULLPTR`、`INVALID_ENUM`、`INVALID_VALUE`。
特别覆盖 `batchCount=0,strideA<0` 和 no-op 但标量为空：不能因提前返回掩盖非法参数。

对会实际访问的矩阵，采用 checked uint64 算术计算
`(batchCount-1)*stride + (cols-1)*ld + rows`，并检查元素转字节以及基址加偏移是否溢出。
这些检查在 no-op 后执行，不为从不访问的矩阵构造虚假地址。

取得 handle stream、平台实际 AIC/AIV 核数和可用 workspace；kernel启动失败与workspace不足返回仓内对应错误码。
不修改全局流；已有 workspace 足够时仅异步提交。库自有 workspace 不足一个 tile 时沿用 handle 的扩容机制，该机制会同步本 handle stream 后重新分配；用户 workspace 不扩容，容量不足返回错误。

### 分核与分块

沿现有 GEMM 的映射计算 `C^T = op(B)^T * op(A)^T`，交换左右操作数和 m/n，
将列主序输出等价为行主序引擎视图；物理 ld、batch stride 与转置标志随操作数一起交换。

初始 tile 为 128×128，L1 的 K chunk 为128，L0 的 K 步长为64。
尾块用 `min(tile, remaining)` 控制有效元素；加载布局与 Fixpipe 输出布局跟随真实尾块尺寸。
将 `(batch, tileRow, tileCol)` 展平为64位任务号，核以 `task=coreId; task<taskCount; task+=coreCount` 轮转。
小矩阵多 batch 因此可以并行填满设备，无需每批启动一对 kernel。

按可用 handle workspace 将任务分为若干 wave；每个 workspace槽仅保存一个输出 tile 的 FP32 结果。
`slots <= workspaceBytes / (tileM*tileN*sizeof(float))`，尾 wave 只处理实际任务数。
一个 wave 顺序提交 Cube kernel 和 AIV epilogue，再复用相同workspace；同一stream的顺序保证覆盖前已消费。
输出矩阵重叠时 wave 限定在单 batch 内，batch 升序排队。

### Kernel侧设计与API序列

复用 `blas/gemm_batched/arch35/gemm_batched_cube_impl.h` 中的 FP16 dtype traits、
`GbProcessMNTiles<1>`、L1/L0双缓冲、FP32 `Mmad` 和相关同步约定。

```text
Host validate / read half scalars / checked offset and wave planning
  if no-op: SUCCESS
  if k==0 or alpha==0:
    launch scale kernel: logical C elements only; beta==0 writes zero
  else for each wave in handle stream:
    Cube: decode batch/tile
      Te::MakeTensor / layout for column-major operands
      Te::Copy GM -> L1 -> L0A/L0B, same-family double buffering
      Mmad FP16 x FP16 -> FP32 L0C, traverse all K chunks
      Fixpipe / Te::Copy L0C -> float workspace tile
    AIV: read float tile
      alpha multiply in FP32
      if beta!=0: load old half C, Cast to float, beta multiply and Add
      Cast FP32 -> half using round-to-nearest-even
      write logical C slice, preserving padding
```

Cube、向量和搬运流水使用模板既有 event；workspace依赖由 stream 保序。
不以 Host 读取 Device 数据计算，不使用原子累加拼接输出，不以逐元素 Host 循环替代 GEMM。

### 片上资源与Workspace

以下为初始 128×128、Kchunk=128、L0K=64 的资源预算，最终构建应逐项校验实际缓冲及平台容量。

| 资源 | 用途 | 计划字节 |
|---|---|---:|
| L1 | A/B half，各双缓冲 | 2×(128×128×2+128×128×2)=131072 |
| L0A | half，双缓冲 | 2×128×64×2=32768 |
| L0B | half，双缓冲 | 2×64×128×2=32768 |
| L0C | FP32累加tile，双缓冲 | 2×128×128×4=131072 |
| UB | AIV整tile：AB float、旧C half/float、结果half，四个单缓冲 | 12×128×128=196608字节（192KiB），小于arch35的248KiB；按32字节对齐 |
| GM workspace | 每个在途tile一个FP32槽 | slots×128×128×4；由handle容量限制 |

FP16乘法、FP32累加使用Cube原生能力，无需先把全部 A/B 转成 FP32。
workspace 使用量不超过32MiB，不随完整 batchCount×m×n 无限增长；用户空间不足一个tile时返回ALLOC_FAILED。输出转换使用CAST_RINT（ties-to-even）。

### 工程文件

```text
include/cann_ops_blas_common.h                  # aclblasHalf共用类型
include/cann_ops_blas.h                         # 公共API声明
blas/gemm/arch35/hgemmStridedBatched_host.cpp    # 校验、分wave、句柄/stream
blas/gemm/arch35/hgemmStridedBatched_kernel.cpp  # Cube与epilogue/scale
blas/gemm/arch35/hgemmStridedBatched_*.h         # 小型kernel接口/tiling
blas/gemm/README.md                            # API、约束、950PR产品支持
cmake/asc_devkit_version.cmake                 # 9.1 tensor_api支持门控
 test/gemm/hgemmStridedBatched/                 # CMake、参数、golden、README
 test/gemm/hgemmStridedBatched/arch35/           # CSV/GTest及验证脚本
```

文件名遵守仓库自动源文件/测试发现规则；沿用根构建，不创建另一套算子工程。

## 支持硬件

| 产品 | 支持 |
|---|---|
| Ascend 950PR | 目标支持；真机验证结果在自测报告中独立列示 |

## 算子约束限制

仅 FP16；标量位于 Host，矩阵位于 Device；同一 handle 的 workspace 复用遵守仓内并发约定。
调用方保证输入输出缓冲实际容量，接口无法从裸指针推断分配大小。
A/B 与 C 跨缓冲别名不作为新增支持项；batch 输出重叠采用上述保序方案。
不要求额外 dynamic-shape 框架接入或 bit-exact 确定性。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 执行与判据 | 来源 |
|---|---|---|
| 用例 | 保留随包1200条：1000精度、200性能；补充分布、padding、边界、非重叠输出步长与错误校验 | §3.5、§4.2 |
| Golden | Netlib BLAS3.10.0逐batch `cblas_sgemm` FP32，随后RNE转FP16；核实实际加载库 | §3.1/3.2/3.5 |
| 有限精度 | atol=rtol=2^-10，matched_ratio≥0.99，max_abs_error≤0.1或32ULP；逐batch全矩阵验证 | §3.2 |
| 非有限精度 | 同NaN、同号Inf、NaN与任意Inf等效；有限/非有限及异号Inf不等效；单独计数且全通过 | §3.2.5 |
| 输入分布 | uniform[-5,5]与normal(mu∈[-5,5],sigma∈[0.1,2])各50%；另保留特殊值 | §3.5 |
| 性能采集 | msprof op，5次warmup+10有效样本；一次API有多个kernel时对其task duration求和；单位us | §3.3 |
| 性能判据 | 同时报告逐case GPU/NPU及全部倍率的算术均值；§3.3要求单例≥0.4、平均≥0.5 | §3.3 |
| 采样口径 | 本轮按任务专项条款§3.3执行5次warmup和10个有效样本，单case≥0.4、平均≥0.5；保留原始任务材料便于追溯 | §3.3 |
| 内存专项 | 任务明确不涉及；实现仍校验边界/偏移并验证padding保留 | §3.4 |

有限值统计不包含非有限值；空有限集合按全体非有限判据检查，不能因除零或空统计误报失败或通过。
MERE/MARE仅为辅助数据，不替代任务混合容差。
测试进程失败、超时、未执行、用例缺失、msprof样本缺失均保持失败/未执行状态。
随包 GTest 总墙钟耗时包含数据准备与 CPU golden，不用于替代任务要求的 NPU task duration。

精度测试先完成九种布局、尾块、不同batch、alpha/beta退化、padding及错误参数的小集，再完成全部用例。
性能只有设备真实执行成功后才记录；通过的结果绑定源码/产物、设备环境、输入及判据，条件未变时复用。
若大k出现FP16数值异常，先定位并提供任务§3.2.4所需误差与golden证据，不直接放宽阈值。

## 性能分析与优化计划

通过 batch/tile展平提升小矩阵并行度，通过L1/L0双缓冲重叠搬运和计算，通过wave限制workspace峰值。
真实profile将用于判断双kernel开销、workspace流量和tile利用率；根据证据调整tile或alpha=1/beta=0快路径。
所有优化均重新验证受影响精度/性能用例；当前设计不预先承诺实测倍率。

## 兼容性分析

新增公共符号及16位类型，不修改既有GEMM签名。N/T/C与列主序语义和任务接口一致。
950 tensor_api代码受CANN9.1门控；不对旧产品线添加不实支持声明。
原始任务CSV保留，局部比较器按本任务特殊值规则实现，避免改变全仓其他算子测试判据。

## 分阶段交付

1. 设计阶段：本设计文档和可评审PR，按流程登记设计链接。
2. 开发阶段：Host/Kernel、公共类型与声明、算子README、GTest/CSV/复现脚本；实现与设计评审并行。
3. 自验阶段：完整精度/性能报告、原始日志、真实结果截图、环境与源码版本记录；内存按不涉及说明。
4. 验收交接：私仓分支、commit链接、开发者权限证明、ZIP和人工填写文本；设计需完成评审合入，必测需通过后方可申请验收。
5. 正式上游代码PR及兑奖：在验收通过并获后续通知后开展。
