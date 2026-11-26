# aclblasSgemmBatched / aclblasHgemmBatched 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 11 月 aclblasSgemmBatched/aclblasHgemmBatched 算子开发（A2/A3） |
| 目标硬件与软件 | Atlas A2/A3，arch22，CANN 9.1.0；当前自验使用 910B3 |
| 工程模式 | Ascend C kernel 直调，使用 handle 绑定的 stream |
| 接口与数据类型 | S：FP32；H：FP16；alpha/beta 为同类型 Host 标量 |
| 存储布局 | 列主序、Device 指针数组、各 batch 地址独立 |
| 实现目录 | `ops-blas/blas/gemm_batched/arch22/` |
| 测试目录 | `ops-blas/test/gemm_batched/sgemm_batched/arch22/` |
| 当前验证 | 附件 CSV 的 1000 条精度/接口用例及 200 条性能用例正确性全部通过；正式性能尚未完成 |

# 需求背景

## 需求来源

本设计对应 11 月社区任务「aclblasSgemmBatched/aclblasHgemmBatched 算子开发（A2/A3）」，以任务书附件 `50d7f6f055c948219f697c99ae5b0367.zip` 为接口和验收依据。[任务页面](https://www.hiascend.com/activities/task-center/details/55123ab3caa64d089f525f646155094c)。

## 背景介绍

批量 GEMM 通过设备指针数组描述多组地址独立的矩阵，在同一组形状、转置和标量参数下计算矩阵乘。基于 ops-blas 现有句柄、stream、workspace 和 Ascend C 工程扩展 arch22，保留公开接口与现有许可证。


# 需求分析

## 需求描述

每个 batch 独立计算 `C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]`。A、B、C 均为列主序；实数算子 `OP_C` 与 `OP_T` 等价。S 的输入输出及标量为 FP32；H 为 FP16，计算采用 FP32 累加，最终按就近偶数规则写回 FP16。

## 接口定义

复用 `include/cann_ops_blas.h` 中已有声明，参数顺序与任务书 §2.3 一致：

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* const Aarray[], int lda,
    const float* const Barray[], int ldb, const float* beta,
    float* const Carray[], int ldc, int batchCount);

aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha, const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb, const aclblasHalf* beta,
    aclblasHalf* const Carray[], int ldc, int batchCount);
```

`op(A)` 为 m×k，`op(B)` 为 k×n，C 为 m×n。A 的存储范围为 lda×k（N）或 lda×m（T/C）；B 为 ldb×n（N）或 ldb×k（T/C）；C 为 ldc×n。三个指针数组长度均为 batchCount。

## 需求拆解

1. Aarray/Barray/Carray 本身及其元素指向设备内存；alpha/beta 为 Host 标量。各 batch 的矩阵地址可以不连续。
2. 支持任意合法 lda/ldb/ldc，只改写 C 的逻辑区域，保留列尾 padding。调用者保证 C 之间以及 C 与输入之间不重叠。
3. m、n 或 batchCount 为零时不访问矩阵，不启动计算；k 或 alpha 为零时只计算 beta*C，允许 A/B 数组为空；beta 为零时不读取旧 C。
4. 通过绑定 stream 异步执行。内核完成与接口返回是不同事件，测试在同一 stream 同步后读取结果。
5. 覆盖附件 S/H 各 500 条精度/接口用例及各 100 条性能用例，并另测边界组合和随机输入。

# 详细设计

## 算子分析

| 参数 | 合法范围与访问规则 |
| --- | --- |
| m/n/k/batchCount | 非负；负值返回 INVALID_VALUE |
| transa/transb | N/T/C；其他值返回 INVALID_ENUM |
| lda | N 时至少 max(1,m)，T/C 时至少 max(1,k) |
| ldb | N 时至少 max(1,k)，T/C 时至少 max(1,n) |
| ldc | 至少 max(1,m) |
| alpha/beta | 必须非空，类型与算子一致 |
| Aarray/Barray | 只有实际执行乘积时要求非空 |
| Carray | 非空输出时必须非空，包括不改变 C 的 beta=1 路径 |

Host 校验依次处理 handle、转置、尺寸、前导维、batchCount、标量及必要数组。空输出允许数组为空，但仍校验结构参数。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；负尺寸、非法前导维或必要指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。设备指针数组元素的有效性与不重叠性由调用方保证，Host 不回读扫描。

矩阵偏移和 tile 总数采用 uint64_t；计算 batch 与单矩阵 tile 数的乘积前检查溢出。工作区按当前容量确定窗口，无法容纳一个 tile 时返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

## 算子实现

### Host 与工程接入

实现置于 `blas/gemm_batched/arch22/`，复用 `include/cann_ops_blas.h` 的两个声明。测试统一由 `test/gemm_batched/sgemm_batched/arch22/` 加载 S/H 两份原始 CSV，保留全部用例与验收阈值。

S/H 使用共用的 Host 校验与调度模块，两个入口仅处理标量类型。常规乘积通过 Ascend C Matmul API 实现，FP32 与 FP16 输入分别实例化；累加及临时结果为 FP32，S 关闭 HF32。乘积 kernel 后在同一 stream 下发 AIV epilogue，完成 alpha/beta 运算，并按 ldc 写回逻辑区域。H 直接读取原生 FP16 输入，写回使用 `CAST_RINT` 就近偶数舍入。

```text
aclblasSgemmBatched / aclblasHgemmBatched
  └─ 共用 Host 校验、读取 Host 标量
      ├─ 空输出 → 返回
      ├─ k=0 或 alpha=0
      │   ├─ beta=1 → 返回
      │   └─ 其他 beta → AIV 清零或缩放
      ├─ m*n≤64 且 k≤32 → 单次 AIV 批量小矩阵计算
      └─ 常规矩阵 → 按 workspace 容量遍历窗口
          ├─ AIC：FP32/FP16 乘积 → FP32 临时 tile
          └─ AIV：alpha/beta、数值范围处理、写回 C
```

### 分路径策略

| 路径 | 输入条件 | 核心处理 | 验证重点 |
| --- | --- | --- | --- |
| P0 空输出 | m=0、n=0 或 batchCount=0 | Host 校验后返回 | 空数组与非法结构组合、无设备访问 |
| P1 仅缩放 | k=0 或 alpha=0 | AIV 清零/缩放；beta=1 直接返回 | 不读 A/B，beta=0 不读旧 C，padding 不变 |
| P2 小矩阵批量 | 当前 m*n≤64 且 k≤32；P0/P1 已排除 | 一次 AIV 下发，各核跨步处理独立 batch | 全部地址独立、高 batch 数、尾部任务 |
| P3 常规矩阵 | P0/P1/P2 之外的合法输入 | 128×128 Cube 分块乘积，再由 AIV 完成 alpha/beta 与写回 | N/T/C、非方阵、带填充和非默认标量 |
| P4 后处理融合探索 | alpha=1、beta=0 且输出布局满足硬件写回约束 | 探索直接输出或融合后处理，省去临时结果往返 | ldc、尾块、FP16 舍入、相邻区域不被覆盖 |
| P5 数值范围处理 | P2 或 S 的 P3 后处理中出现非有限候选结果，或 abs(alpha)>2^64 | 在设备内按指数归一化、双 float 补偿重新计算受影响元素 | 中间溢出、抵消、NaN/Inf 分类及最终舍入 |

P2/P3 的当前条件来自工作量划分，后续用相邻形状、不同 batch 数及实测交叉点调整。P4 尚未实现；P5 已接入当前版本并通过附件极值用例，还需补充独立生成的强抵消和宽指数分布测试。所有路径使用实际输入参数和设备端计算结果分流。

### 并行与访存

以 `(batch, tileM, tileN)` 描述独立任务，设备直接读取指针数组，避免 Host 回读数组和逐 batch 同步。列主序可转换为交换操作数的行主序视图；转置与原始前导维必须一起映射。

常规矩阵优先采用 Cube 的 FP32 累加，比较不同 tile 和 K 分块的搬运量、并行度及尾块比例。S/H 的缩放和后处理只搬运逻辑区域，利用 DataCopyPad 和片上 Cast；缩放/收窄时按真实输出长度写回。相邻输出块不得因为 DMA 对齐覆盖其他块或 padding。

当前 P3 的输出 tile 为 128×128，K 每段不超过 4096，尾块按实际 m/n/k 长度传给 Matmul。计算 `C^T = op(B)^T * op(A)^T` 的行主序视图，交换 A/B 的角色并同步映射 trans 与 leading dimension。K 的第一段写入临时 tile，后续段在同一任务中累加，H 到最终写回才转换为 FP16。

当前 P2 每个核处理 `batch = blockIdx + j*blockNum`，逐元素累加点积后按列写回。P1 将 C 的每列拆为最多 4096 元素的连续片段。两者均将多个 batch 合并到一次下发中，分别按 batch 数或片段总数限制参与核数。

### 工作区与同步

定义总任务数 `tasks = batchCount * ceil(m/128) * ceil(n/128)`，每个 FP32 临时 tile 占 `128*128*4 = 65536` 字节。默认工作区预算为 32 MiB，因此每个窗口最多 512 个 tile；用户工作区更小时按其实际容量缩减窗口。

```text
窗口 firstTask ... firstTask+taskCount-1
  workspace: [FP32 tile 0][FP32 tile 1] ... [FP32 tile taskCount-1]
  同一 stream: Cube 写临时结果 → AIV 读临时结果并写 C → 下一窗口复用
```

工作区只保存当前窗口的输出 tile，不为所有 batch 完整展开矩阵。AIC/AIV 核数分别按平台查询值及当前任务数取较小值；AIV 内显式处理搬入、计算与搬出间的事件依赖，写回后再复用片上缓冲。

workspace 通过已有 handle 管理，按当前可用容量分批或分窗口执行。复用区域须等待同 stream 中先前消费者完成；用户提供的工作区不足时沿用仓内错误约定，不越界使用。优化前后记录 kernel 序列及额外搬运，避免只测主 GEMM 而遗漏转换或后处理。

### 数值计算

常规路径按 FP32 累加执行乘积、alpha 缩放与 beta 累加。`beta=0` 时跳过旧 C 的读取；P1 的 `beta=0` 直接生成零。

P5 在 Device 上将有限输入拆为带符号尾数和二次幂指数，先确定各乘积及 beta*C 项的最大指数，再按该指数归一化。尾数乘积采用 Dekker 拆分，以两个 float 保存高低部分并补偿累加，最后恢复指数。遇到真实 NaN/Inf 输入则按浮点乘加传播处理。CPU FP64 仅用于测试 golden，不参与算子执行。

该处理针对中间范围溢出；对仍产生有限结果的严重抵消场景，还需用独立数据复测误差并完善触发条件。验收按任务书误差阈值判定，不声明与 FP64 或 cuBLAS 逐位一致。

### 支持硬件与约束

目标为 Atlas A2/A3 arch22、CANN 9.1.0。当前已连接的开发设备为单卡 910B3；A3 尚未实测。采用现有平台查询获取核数与片上容量，不把开发容器的设备编号写入实现。接口不增加私有参数，不修改 arch35 语义。

# 可维可测分析

## 精度与性能标准

| 项目 | 验证方法 |
| --- | --- |
| 完整性 | 逐个记录 S/H 各 600 行，检查名字、数量、返回码和退出状态；区分 500 条精度与 100 条性能 |
| 输入 | 固定 seed 的均匀 [-5,5]、标准正态、极值、Inf/NaN；H 输入和标量先收窄到 FP16 |
| Golden | 按任务书 §3.2 第 4 项：S 使用 FP64 累加后转 FP32；H 使用 FP64 累加后依次转 FP32、FP16。本轮使用 OpenBLAS 0.3.20 的 cblas_dgemm，报告注明此路径与直接 Netlib FP32 计算的舍入差异 |
| 精度 | S atol=rtol=2^-13；H atol=rtol=2^-10；匹配率至少 99%；逐元素额外检查 max(固定绝对上限,32ULP)，固定上限 S=0.01、H=0.1 |
| 完整写回 | 对 C 的 padding 与边界保护区逐字节检查；验证输入及指针表未被修改 |
| 正式性能 | 910B3 上 msprof op，预热 5 次后取得 10 次有效样本；同一调用所有必需 kernel 均计入，与对应 GPU kernel 基线比较，均值不超过基线的 1.25 倍 |
| 内存项 | 按任务书 §3.4 为 N/A；额外执行保护区及输入不变性检查 |
| 辅助性能 | 按任务书记录 ACL event 夹连续 10 次下发的平均耗时，用于趋势分析，与正式 kernel 耗时分列 |
| 复现 | 保存源码 commit、环境、CSV 哈希、输入 seed、执行命令、逐条结果及原始性能样本 |

测试逐项匹配原始 CSV，S/H 名称分别带 `S_`/`H_` 前缀，避免同名用例合并。性能基线匹配键包含算子、转置、m/n/k、lda/ldb/ldc、alpha/beta 和 batchCount。正式采集须把一个接口调用实际执行的全部窗口、Cube 及后处理 kernel 计入该次样本，预热与正确性调用不计入 10 次有效测量。

当前测试使用逆序指针表描述各 batch 的独立区域，在 C 的边界及 padding 放置哨兵，校验 A/B、指针表与非逻辑输出区域未改写。后续补充更多独立分配方式、非默认 stream、邻近 tile 边界及同 shape 不同数值分布的随机测试。Host 返回码与设备结果分别断言。

## 工程与复现

| 文件/模块 | 职责 |
| --- | --- |
| `sgemm_batched_host.cpp`、`hgemmBatched_host.cpp` | S/H 公开入口 |
| `sgemm_batched_host_impl.h` | 共用校验、路径选择与窗口调度 |
| `sgemm_batched_plan.h` | Host/Kernel 共享参数与启动声明 |
| `sgemm_batched_kernel.cpp` | 缩放、小矩阵、Cube 乘积和 AIV 写回 |
| `sgemm_batched_range.h` | Device 指数归一化及补偿计算 |
| `sgemm_batched_test.cpp`、`community_s.csv`、`community_h.csv` | 1200 项 CSV 驱动自验 |

本轮代码提交为 `f165b6113979be9f18d64ffbae68ceb3771d2e51`，位于个人私仓 `community-gemm-batched-a2a3` 的 `develop/gemm-batched-a2a3` 分支。Ascend-CANN 已获开发者权限。构建与运行示例（进入 ops-blas 根目录，CANN 环境已加载）：

```bash
cmake -S . -B build -DSOC_VERSION=ascend910b3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DASC_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/cmake \
  -DBUILD_TEST=ON -DTEST_NAMES=sgemm_batched -DTEST_DEVICE_ID=0
cmake --build build --target sgemm_batched_test -j4
set -o pipefail
LD_LIBRARY_PATH="$PWD/build:/usr/lib/aarch64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH" \
BLAS_DEVICE_ID=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  ./build/test/gemm_batched/sgemm_batched/sgemm_batched_test \
  --gtest_output=xml:sgemm_hgemm_selftest.xml 2>&1 | tee sgemm_hgemm_selftest.log
```

原始 CSV SHA256：

- S：`6615bb4ddcf336a58d802519b334a1e5d2080493bd7f2ab3f9c70e28f32d1364`
- H：`ff5eb4d59dd5572f476c370344840ed7a851f4dd8d1cfdf4fe429c6feb95c3c8`

## 当前状态与提交安排

截至 2026-11-25，本轮在 910B3/CANN 9.1.0 的完整自验退出码为 0，结果如下：

| 接口 | 精度与接口用例 | 性能用例正确性 | 正式性能 |
| --- | --- | --- | --- |
| aclblasSgemmBatched | 500/500 通过 | 100/100 通过 | 尚未完成 |
| aclblasHgemmBatched | 500/500 通过 | 100/100 通过 | 尚未完成 |

辅助 Event 计时与 GPU 基线的交叉比较中，S 的 40/100 项、H 的 9/100 项不超过 1.25 倍。两种计时口径不同，此数据用于筛选优化方向；当前优先改善大 batch 小矩阵的逐元素访存，以及短 K 大输出的 tile 利用率和后处理开销。

本题 `msprof op` 尝试报告 0 success / 20 failed，错误为 `Profiling channel start failed`；驱动日志显示 profiling HDC 通道查询返回 -1，尚未获得正式 kernel 耗时。下一步排查采集环境，完成优化和正式性能复测后更新自验报告。

设计文档先评审合入，代码继续在个人私仓开发。验收材料按任务书 §4 整理至 `task_submission/`，包含自验证步骤、精度与两种口径的性能报告、原始日志、测试用例包及待验收 commit；内存项标记 N/A。任务书要求的冻结输入、golden、cases.json 与 facts.json 等交付内容一并核对。自验材料齐备后申请验收，收到通过通知后再向 ops-blas 提交 Issue 与代码 PR。

## 兼容性分析

保持公开 S/H 接口、状态码、列主序、设备指针数组和异步 stream 契约。新增实现独立置于 arch22；公共句柄与 workspace 管理沿用仓内接口。S/H 共用的实现文件需同时纳入单算子构建，避免仅构建 H 时遗漏 kernel。
