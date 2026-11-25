# aclblasCgemm3m 算子设计（Atlas A2 / A3）

## 1 需求背景

### 需求来源

[11 月社区任务：aclblasCgemm3m 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/cae74cb900484862a2b43b24c0f7d9d3)。[ops-blas](https://gitcode.com/cann/ops-blas) 已声明 `aclblasCgemm3m`，并有 950PR 的 arch35 实现。本设计为 arch22 补齐 Ascend C 实现和 GTest；不修改公共声明、其他架构的实现、测试输入或验收阈值。

### 背景介绍

`Cgemm3m` 对单对复数矩阵计算乘积，并用复数系数把结果累加到 C。这个接口的区别不仅是输入类型：三乘法将一次复数矩阵乘拆成三次实矩阵乘，适合在 Cube 上并行；但拆分、相加再相减及 K 向归约改变了浮点舍入顺序。实部、虚部必须分别验证，不能用两个结果的复模相近代替分量正确。

## 2 需求分析

### 接口与数据约定

沿用 `include/cann_ops_blas.h` 中的声明：

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc);
```

`m,n,k` 非负；`alpha,beta` 是 Host 指针，矩阵为 Device 指针。每个 `aclblasComplex` 含相邻的两个 float32，依次为实部和虚部。列主序下元素 (i,j) 位于 `i+j*ld`；每个前导维度的单位是**复数元素**而非字节。下表写的是**转置前**物理矩阵的有效形状与下界：

| 矩阵 | 不转置（N） | 转置 / 共轭转置（T/C） |
| --- | --- | --- |
| A | m × k，lda ≥ max(1,m) | k × m，lda ≥ max(1,k) |
| B | k × n，ldb ≥ max(1,k) | n × k，ldb ≥ max(1,n) |
| C | m × n，ldc ≥ max(1,m) | 不适用 |

C 原地覆写。调用方保证 C 可写且不与 A、B 重叠。LD 可大于有效行数，列间填充不能当数据，也不能被写坏。不支持批量、广播、任意步长视图或 Device 侧标量。

### 边界及错误行为

先检查 handle（空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）、转置枚举（非法返回 `ACLBLAS_STATUS_INVALID_ENUM`）、维度、LD 和标量指针（非法返回 `ACLBLAS_STATUS_INVALID_VALUE`）。这几项不会因为空矩阵而跳过。结构参数合法后：

| 条件 | 对矩阵的访问 | 返回行为 |
| --- | --- | --- |
| m=0 或 n=0 | 不引用 A/B/C，允许三者为空 | SUCCESS，不发射 kernel |
| k=0 或 α=0，且输出非空 | 不引用 A/B，C 必须非空 | 仅更新 C←βC；β=1 可直接返回 |
| β=0，且输出非空 | 不读取旧 C，即使旧值为 NaN 也不参与运算 | 写逻辑 C，保留列尾填充 |
| 其余情况 | A/B/C 都必须非空 | 在 handle 绑定的 stream 上异步下发 |

成功返回表示接口完成下发，不表示 Device 计算已结束；读回前同步同一条 stream。

## 3 详细设计

### 算子分析与三乘积分解

令 `X=op(A)=Xr+iXi`、`Y=op(B)=Yr+iYi`，形状分别为 m×k、k×n。T 仅交换索引，C 在交换索引的同时对虚部取负。令 `Z=X·Y`，最终写回 `C=αZ+βC`。以下“·”均指实矩阵乘；两组中间积不能混用。

| 算法路径 | 三次实矩阵乘 | 重建 Z 的实部、虚部 |
| --- | --- | --- |
| 非融合 | P=Xr·Yr，Q=Xi·Yi，R=(Xr+Xi)·(Yr+Yi) | P−Q，(R−P)−Q |
| 融合（含纠正重算） | P=(Xr+Xi)·Yr，Q=Xr·(Yi−Yr)，R=Xi·(Yr+Yi) | P−R，P+Q |

展开后一行可得 `P−R=Xr·Yr−Xi·Yi`、`P+Q=Xr·Yi+Xi·Yr`。两种公式代数等价，但 FP32 的中间舍入并不等价。切换融合策略时，打包、Cube 的乘积槽顺序、分片重建及误差纠正必须同时切换；不能只换写回公式。

### Host：尺寸、分核与工作区

Host 根据 m/n/k、转置、物理跨度、列间空隙、标量以及设备的 AIC/AIV 数量选择路径，不使用用例编号或随机种子。先以宽整数计算每个非空矩阵跨度 `(物理列数−1)×LD+物理行数` 并检查地址容量，再计算所需的私有平面空间。所有 kernel 都使用同一条 stream。

| 条件 / 代价 | 分工方式 |
| --- | --- |
| 微小矩阵（m≤32、n≤64、k≤32，且 A/B 实际跨度分别不超过 4096 个复数） | 输入可放入 UB，AIV 按输出列分工，避免启动三个实 GEMM 和分配 GM 中间平面。 |
| 输出块足够多 | AIV 将输入解交错并打包，AIC 将输出划分为二维块并按块处理三组实乘积；根据形状选择独立或合并的向量写回。 |
| 输出块少而 K 长 | 合适时按 K 分片，以扩充同时可执行的工作数。各片先写入独立位置，同一输出块由一个归约所有者按顺序合并，避免并发更新 C。 |
| 不满足专用核的尺寸或布局条件 | 调用 arch22 实数 GEMM 后端三次，复用同一 stream；子调用 scratch 与操作数和结果平面分离。 |

打包后 A/B 各有三块 FP32 平面。若每个 A、B 平面占用分别为 a、b 个 float，每个输出平面占 c 个 float，K 共 s 片，工作区主体为 `4×[3a+3b+3cs]` 字节；通用后端另加 scratch。a/b/c 按**私有布局**对齐，不能拿来替换公开接口的 LD。Host 复用 handle 的有效工作区并检查总大小，超限或申请失败按库约定返回分配错误。

### Device：打包、矩阵乘与写回

1. **打包。** AIV 从逻辑行列解交错，按 N/T/C 选择索引并处理共轭。私有补齐区域填零，不能读借用方的列间空隙来补 K；两边的三个平面按上表产生。对于狭长或有列间空隙的输入，分块搬运或在私有布局中转置，使 Cube 可连续读面板。
2. **实数矩阵乘。** Cube 将装入 L1/L0 的面板分块相乘，在 FP32 累加器中生成 P、Q、R；专用 Cube 路径显式禁用 HF32。短 K 和少量输出块时，将多个乘积或不同 K 分片分发到不同核；尾块只对应有效逻辑结果，不能在用户矩阵边界外写回。
3. **归约及误差纠正。** 长 K 分片先重建各段的实、虚部分量，再累计到同一逻辑元素。需要做数值纠正时，依据运行中前缀和的舍入风险选择片段，在向量侧读取本片输入并补偿，而不识别用例或查询参考结果。纠正项须沿用融合公式 **P−R、P+Q** 的中间量，不得为追求参考值改用另一套分解。
4. **输出。** AIV 先计算 `αZ`，仅在 β 非零时读取原 C 并累加 `βC`，再交错写回复数。紧凑 C 可连续输出；有 padding 时逐列或按带间隔的块写有效行。混合核中，Cube 发布已完成的块或阶段，向量核只在对应信号就绪后消费，私有槽不得提前覆写。

生产实现归属 `blas/gemm/arch22/`：Host 入口和策略在 `cgemm3m_host.cpp`；设备打包、微型计算和写回在 `cgemm3m_kernel.cpp`、`cgemm3m_pack_aiv.h`；Cube 及长 K 归约在 `cgemm3m_cube.cpp`、`cgemm3m_long_aiv.h`。公共接口说明归属 `blas/gemm/README.md`，arch22 的测试入口在 `test/gemm/cgemm3m/arch22/`。

### 硬件及兼容性

目标为 Atlas A2/A3（arch22），开发和性能基准环境为 CANN 9.1.0、Atlas 800T A2（910B3）。A3 为适配目标，不能从 A2 数字推导 A3 的精度或性能。两代复用同一公共 BLAS ABI，不改变 950PR 的 arch35 编译选择与行为。

## 4 可维可测分析

| 验收项 | 方法与判据 | 任务书条款 |
| --- | --- | --- |
| 精度 | 官方 1000 条精度用例；使用 Netlib BLAS 3.10.0 的 `cblas_cgemm` 作参考。实部、虚部各自要求至少 99% 的有限元素满足 `abs(actual−golden) ≤ (1+abs(golden))/8192`，另需最大绝对误差不超过 `max(0.01,32×ULP)`；ULP 取该分量最大误差所对应的 FP32 参考值。另查非有限值和 C 的 padding。 | §3.2、§3.5 |
| 性能 | 原 200 条性能用例；每条预热 5 次，再连续执行 10 次取平均。正式值为 `msprof op` 的单次各 kernel Task Duration 之和；单位统一后要求 `NPU ≤ 1.25 × GPU`。只与同参数的 cuBLAS 基线配对。另记录 `aclrtEvent` 括住 10 次调用的 `[PERF] avg_us`，仅交叉验证。 | §3.3 |
| 负向与边界 | 检查非法枚举、LD 和指针优先级，零阶 / 零乘积的输入是否被引用，β=0 且旧 C 为 NaN，九种转置组合、非规则尾块、长 K、复数 α/β 及非紧凑 C。 | §2.4、§2.5、§3.5 |
| 内存 | 不设独立性能或内存达标指标。工作区容量、边界和 padding 完整性由接口与功能测试覆盖。 | §3.4 |

**原附件异常的处理。** 本期 200 条性能输入中有 60 条的 lda 或 ldb 小于 §2.4 的下界，CSV 却期望 SUCCESS。例如 `TC_PF_0005` 为 transa=N、m=4096、lda=64。接口应返回 INVALID_VALUE，不发射 kernel；保留原行和报错，不记作 0 微秒，也不拿修改参数后的时间和原 GPU 标杆比较。是否调整用例及对应标杆以任务方澄清为准。
