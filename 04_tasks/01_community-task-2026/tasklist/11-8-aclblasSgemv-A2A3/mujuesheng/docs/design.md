# aclblasSgemv 算子设计说明（Atlas A2/A3）

## 1. 需求背景

本设计对应 CANN 2026 年 11 月社区任务 aclblasSgemv 算子开发（A2/A3），任务编号 11-8。贡献者 GitCode 账号为 `mujuesheng`。

目标是在 `cann/ops-blas` 的 `blas/gemv/arch22/` 补齐 FP32 矩阵向量乘，复用 `include/cann_ops_blas.h` 的公共接口。测试位于 `test/gemv/sgemv/arch22/`。设计文档提交到 `cann/cann-ops-competitions` 对应任务目录，代码验收及后续代码合入按任务书另行办理。

本文描述当前冻结的 v17 实现。目标自测环境是 910B3、CANN 9.1.0；已有开发实测来自 Ascend910_9362（A3）、CANN 9.0.0，不能替代目标环境验收。

## 2. 需求分析与接口

### 2.1 数学功能

对列主序 FP32 矩阵 A 原地更新向量 y：

```text
y = alpha * op(A) * x + beta * y
```

`A[i + j*lda]` 为第 i 行、第 j 列元素。`trans=N` 时 op(A)=A，x、y 逻辑长度分别为 n、m；`trans=T/C` 时为 A 的转置，长度分别为 m、n。实数 C 与 T 等价。

### 2.2 公共接口

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);
```

| 参数 | 含义和约束 |
| --- | --- |
| handle | 有效句柄，使用其绑定 stream 和有效 workspace |
| trans | ACLBLAS_OP_N、ACLBLAS_OP_T、ACLBLAS_OP_C |
| m、n | A 的行数、列数，均非负 |
| alpha、beta | 非空 Host 标量指针，值不作合法性过滤 |
| A、lda | Device 列主序矩阵，lda >= max(1,m)；m、n 均非零时 A 非空 |
| x、incx | Device 输入向量，步长非零且不为 INT32_MIN；逻辑长度非零时指针非空 |
| y、incy | Device 原地输出向量，步长约束同 x；逻辑长度非零时指针非空 |

向量物理跨度采用 int64 计算，要求 `(length-1)*abs(inc) <= INT32_MAX`。负步长从物理 span 尾端起算。矩阵列间 padding 和向量步长间隔不参与运算。

### 2.3 返回码与边界

先检查 handle；空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。非法枚举、尺寸、lda、步长、必需指针或向量跨度返回 `ACLBLAS_STATUS_INVALID_VALUE`。参数校验通过后，m=0 或 n=0 返回成功，不启动 kernel、不访问 Device 数据。非空问题未能取得可用 AIV 核心数时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

alpha=0 不设置短路，仍按完整乘加语义计算。beta=0 时不读取旧 y，避免旧 y 的 NaN/Inf 污染输出。任务排除 alpha=0 且 A/x 包含非有限数的精度组合；该组合不用于证明与会短路的 FP32 CBLAS 等价。

## 3. 详细设计

### 3.1 Host 调度与资源

Host 只进行参数检查、形状计算、路由选择和异步下发；不读取 Device 数据或在 CPU 上计算结果。入口复用 handle 的 stream 和 workspace，不增加 Host 侧同步或内存分配。多阶段在同一 stream 中顺序启动。

快速路径的共同条件是 `alpha=1、beta=0、lda=m、incx=incy=1`。路由按输入参数、核心数和资源条件选择，不依赖用例 ID、随机种子或预存结果。

| 路由 | 条件 | 主体计算 | Workspace |
| --- | --- | --- | --- |
| N 小输出、长 K | m 为 4 或 16，n>=65536，满足快速路径条件且 workspace 足够 | 完整 K 分片的部分和 kernel，再合并 | AIV 核心数 × 64 × sizeof(float) |
| N 列分片 | m,n>=128，m*n>=65536，满足快速路径条件和 UB/workspace 守卫 | 按列分片并形成部分 y，再合并 | Round64(m) × parts × sizeof(float) |
| T/C 流水 | m<=8192，满足快速路径条件 | 缓存 x，双槽搬入 A，批量乘法与逐列归约 | 无外部 workspace |
| 其他快速路径 | 未进入上述专用路径的连续输入 | N 输出分块或 T/C 分块点积 | 无外部 workspace |
| 通用精度回退 | 非快速路径参数 | 支持 lda padding、signed strides、一般 alpha/beta 的补偿计算 | 无外部 workspace |

专用分片路径所需 workspace 不足或为空时，继续使用不需要该 workspace 的计算路径。列分片用 uint64/int64 计算对齐量和容量。

### 3.2 N 小输出、长 K

此时输出少、归约长，按输出分核会闲置多数核心。将完整 K 划分给多个 AIV，各核计算其 K 区间对全部输出的贡献。输入 A/x 使用双 buffer 预取；循环中按 lane 累加，完整 K 区间结束后进行树归约，减少反复归约开销。第二 kernel 合并所有部分和，覆盖完整 K，不提前停止。

m=4 路径的 UB 峰值估算约 176.25 KiB。部分和按 64 个 float 的间隔存放，使用上述 workspace 容量守卫。

### 3.3 N 列分片

列主序下沿行搬运连续，将若干列组成批次，复用连续 DMA。令 `pitch=Round8(m)`，batch 在 pitch<=4096 时为 8，否则为 4；Host 要求 `(batch+3)*pitch*sizeof(float)+32 <= 184 KiB`。`parts=min(AIV核心数,ceil(n/8))`，各分片覆盖不同列区间。

单核 K 区间长度<=128 时，采用批内 lane 累加，再进行一次批内树归约；更长区间保留分批补偿累加。所有分片写各自 workspace 区域；第二 kernel 按 128 个输出一块合并。专用路径只用于 beta=0，因此合并阶段无需读取旧 y。

### 3.4 T/C 连续点积流水

列主序中每个输出对应一列 A 与 x 的点积。缓存完整 x，根据 m 和 UB 预算选取不超过 8 列的 batch。两槽输入队列搬入矩阵块；同批独立列的 Mul 集中下发后同步，再逐列执行 ReduceSum。归约结果经 Gather 整理，使用输出队列搬出。

每个核心负责连续输出区间，按实际启动核心数等分。预算包含两槽矩阵、缓存 x、乘法临时量、归约 scratch 和输出队列，不超过 184 KiB。尾列、非对齐长度按有效元素计数处理。

### 3.5 通用计算与精度

一般标量、lda padding、非单位步长及正负步长进入通用精度路径。使用 FP32 两项展开、Dekker 乘积分解、幂次缩放和补偿归约减小误差，归约后执行 alpha/beta 合成。它不是任意混合动态范围下的精确实数算法；未测输入不得宣称全部精度达标。

其余连续输入使用常规 Vector 分块；N 常规路径使用输出块和归约块，T/C 点积按 1024 元素分段。所有路径遍历完整归约域。

### 3.6 同步、尾部和读写边界

核内通过队列及 pipeline barrier 管理搬运和计算依赖。部分和 kernel 与合并 kernel 通过同一 stream 的启动顺序建立依赖，不使用跨核忙等。尾块借助 DataCopyPad 等方式填充无效 lane；只写有效输出，不覆盖向量间隔和尾部 guard。A、x 为只读；不新增 broadcast 或返回视图。

当前 v17 生产实现使用 Vector。Cube 布局原型属于独立实验，没有纳入本设计对应的冻结候选，也不据此宣称 Cube/Vector 已实现重叠。

## 4. 任务约束逐项核对

| 任务约束 | 设计处理 |
| --- | --- |
| 参数合法性及校验顺序 | 检查 handle 后检查参数及跨度，最后处理空尺寸 |
| 非连续输入 | 支持 lda>=m 的列间 padding 和非零正负 incx/incy |
| Broadcast | 不涉及，维度由 m、n、trans 唯一确定 |
| Dynamic shape | 运行时参数生成调度数据，不要求图模式 dynamic shape |
| 原地与视图 | y 原地覆写，A/x 只读，不返回视图 |
| 确定性 | 不承诺跨设备、跨版本逐位一致；单次归约无竞争更新 |
| 空 Tensor | 校验后 m=0 或 n=0 不启动计算、不访问 Device 数据 |
| 0 维 Tensor | BLAS 标量及尺寸接口，不涉及通用 Tensor 的 0 维输入 |
| 异步执行 | 绑定 stream 下发，调用者读取结果前负责同步 |

## 5. 可维可测分析

### 5.1 精度验证

原任务 1000 项精度/边界和 200 项性能用例保留，另补 1000 项高斯输入。两条原始 alpha=0/nonfinite 组合与任务排除域冲突，只在派生 CSV 修正并记录 provenance，原始输入保留。高斯生成器的实际截断范围为 [-5,5]。

任务书同时出现 FP64 真值计算后收窄 FP32、以及 FP32 Netlib CBLAS 两种 golden。当前并行记录两套结果：同一 v17 manifest 下，2202 项 GTest 的 FP64/返回码/边界检查通过；2168 次数值比较中，FP32 CBLAS 有 198 项未通过，包括性能子集的 TC_PF_1025、TC_PF_1027。主办方采用口径尚待确认，不能表述为两套精度全部通过。

精度标准按任务 mixed_tolerance 要求检查匹配比例及最大误差。返回码、beta=0 配 NaN 旧 y、signed strides、padding、尾块及 guard 另行覆盖。v17 没有执行独立 122 项 holdout，其他候选的结果不计入 v17。

### 5.2 性能验证

官方要求在 910B3、CANN 9.1.0 上以 msprof op task duration 计时，预热 5 次，至少 10 个有效样本取算术平均；每项 NPU 时间不超过相应 GPU 基线的 1.25 倍。

已有开发记录：A3/CANN 9.0.0 的 200 项分四批采集，每项 11 个有效样本，四批源码、二进制、输入 manifest 一致，200/200 满足比较门槛。多 kernel 路由按一次 API 各阶段 task duration 求和，不包含 launch 间隙。原 GPU 表 200 行、197 个参数键，重复键沿用原比较脚本的末行规则。

最紧项 TC_PF_1046 均值 25.747787636 us，比较门槛 25.9839995 us，裕量较小。开发设备结果不证明 B3 达标，也不保证重复测量始终通过。B3/9.1 数据及官方验收结果仍待补齐。

### 5.3 可追溯性

对应上游开发基线为 `6f92fa8b49ef6fb62dd83351723ea3896caaeeab`。v17 主 kernel 文件 SHA256 为 `0bae7d17f3e6aefcd59ac578ec176a3c70e63882edd8fc9d54f533b7aaa761d9`，开发实测 libops_blas.so SHA256 为 `979a1fd1cf4dc1ce9b1a0a2ae61eadacb87d6e43e1c178fa74f4322e00a4361e`。完整源码、原始日志、CSV、XML 和 manifest 在验收材料中提供，私仓根目录同步提交 task_submission 七项材料。

本 PR 仅提交设计说明，不包含代码变更，不将文档提交、PR 创建或 PR 合入等同于代码门禁或任务验收通过。

## 6. 兼容性与已知限制

公共 API 复用既有声明，不新增平行私有接口；实现限定 arch22，不修改 arch35 实现。910B3/CANN 9.1.0 的编译、精度、性能和仓内正式门禁待补测。FP32 CBLAS 的 198 项差异待口径确认或后续修复；UB 资源需在目标平台再次核对。

## 7. 参考资料

- [本任务页面](https://www.hiascend.com/activities/task-center/details/a8c0b7d0985349c59503ec0c34ca7344)
- [社区任务提交说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
- [官方设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [目标代码仓 cann/ops-blas](https://gitcode.com/cann/ops-blas)
- [Netlib SGEMV 参考语义](https://netlib.org/blas/sgemv.f)
