# aclblasSgemv 算子设计文档（Atlas A2/A3，arch22）

> 文档性质：开发设计说明。接口与语义以本任务书及 ops-blas 既有声明/实现为准；本工作区未包含算子源码、编译产物或 910B3 实测结果，因此本文的 kernel/tiling 内容是实现方案，不代表已完成代码或已通过验收。实际实现完成后，应将源码路径、tiling 参数及验证数据回填到相应章节。

## 1. 设计目标与范围

在 ops-blas 的 `blas/gemv/arch22/` 中实现 Ascend C kernel 直调版 `aclblasSgemv`，为 Atlas A2/A3 提供 FLOAT32、BLAS-2 矩阵向量乘加。实现复用公开头文件中的接口，不增加私有平行 API；Host 负责参数校验、空问题快速返回和 tiling 下发，Device kernel 完成计算。测试按要求放在 `test/gemv/sgemv/arch22/`，并在该算子的 CMake arch22 分支注册 GTest。

目标语义为列主序矩阵上的

```text
y ← α · op(A) · x + β · y
```

其中 `op(A)` 由 `trans` 决定。本文只覆盖任务要求的 FLOAT32 `aclblasSgemv`；性能目标、精度阈值和自验环境按任务书第 3 节。

## 2. 接口与参数语义

沿用 `include/cann_ops_blas.h` 中的既有声明：

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);
```

| 参数 | 语义与约束 |
| --- | --- |
| `handle` | 非空库句柄，计算提交到其绑定的 stream；空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。 |
| `trans` | 仅接受 `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C`；实数场景下 C 与 T 等价。非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| `m`, `n` | A 的行数、列数，均须非负。负数返回 `INVALID_VALUE`。 |
| `A`, `lda` | A 为列主序；元素 `A(i,j)` 的偏移为 `i + j*lda`（0-based），`lda >= max(1,m)`。`m>0 && n>0` 时 A 不可为空。允许 `lda>m`，列间填充不参与计算。 |
| `x`, `incx` | N 时逻辑长度 `n`，T/C 时长度 `m`。步长以元素计，任意非零 int32 均支持，`INT32_MIN` 不支持。负步长从物理 span 尾端对应的逻辑首元素开始。逻辑长度非零时 x 不可为空。 |
| `alpha`, `beta` | 指向 Host FLOAT32 标量，不可为空；标量值不做合法性筛选。 |
| `y`, `incy` | N 时逻辑长度 `m`，T/C 时长度 `n`。y 为原地输入输出；`beta==0` 时不得读取旧 y。步长规则与 x 相同。逻辑长度非零时 y 不可为空。 |

令 `k=n, len(y)=m`（N），或 `k=m, len(y)=n`（T/C）。计算为

```text
N: y[j] ← α · Σ(i=0..n-1) A[j + i·lda] · x[i] + β · y[j]
T/C: y[j] ← α · Σ(i=0..m-1) A[i + j·lda] · x[i] + β · y[j]
```

向量逻辑元素地址需考虑增量。对逻辑长度 `L`、非零增量 `inc`，物理跨度为 `(L-1)*|inc|`。Host 用 int64 计算跨度，若超过 int32 可表示范围则返回 `INVALID_VALUE`；先拒绝 `INT32_MIN`，避免绝对值溢出。负增量时逻辑首地址偏移为 `(L-1)*|inc|`，之后按负步长遍历。

### 2.1 校验顺序与快速返回

Host 校验在提交 kernel 前完成。建议按仓内参考实现顺序检查：

1. `handle==nullptr` 返回 `HANDLE_IS_NULLPTR`。
2. 校验 `trans`、`m/n`、`lda`、`incx/incy`（含 `INT32_MIN` 和跨度溢出）、`alpha/beta` 非空。
3. 对 `m>0 && n>0` 要求 A 非空；按 `trans` 求 x/y 逻辑长度，仅长度非零时要求对应指针非空。
4. 上述校验通过后，若 `m==0 || n==0`，返回 SUCCESS，不读取设备数据、不启动 kernel、不修改 y。
5. 非空计算问题准备 tiling 并异步下发 kernel。

此顺序意味着 quick return 不豁免其他属性校验。例如 `m=0` 时仍需合法 `lda/incx/incy` 且 `alpha/beta` 非空；A 可为空，x/y 是否可空由各自逻辑长度决定。alpha=0 不短路，依公式计算，因此 A/x 中的 NaN/Inf 可传播；任务书明确不将 alpha=0 且 A/x 含非有限值的组合用作精度用例，因为 Netlib golden 在 alpha=0 时可能走不同短路路径。beta=0 时则必须遵循“不读取 y 旧值”的接口语义。

## 3. Host 与 Device 结构

### 3.1 Host 入口

Host 入口负责：

- 完成上述参数及地址跨度校验，错误时立即返回对应状态码。
- 读取 Host 标量 `alpha`、`beta`，将标量值写入 tiling 数据；不得将 Host 标量指针当作 Device 指针传递。
- 对空问题 quick return。
- 根据 `trans`、输出长度和归约长度选择 kernel 方向及 tiling；运行时维度参与计算，不依赖静态图 shape。
- 使用 handle 对应的 stream 配置 kernel launch。launch 成功表示已提交，不表示设备执行已完成。

### 3.2 Kernel 映射

每个输出元素对应一条长度为 `k` 的点积，随后乘 alpha 并融合 beta/y。计算索引统一使用 64 位整数或经证明不会溢出的索引表达式；Host 已检查向量跨度，矩阵偏移按 `row + col*lda` 计算。`lda>m` 的填充元素永不读入。

**N（非转置）**：一个输出是 A 的一行与 x 的点积，但列主序下相邻归约项在 A 中相隔 `lda`。优先将相邻输出行映射到同一计算块，使同一列上读取的 A 元素相邻/近邻；沿归约维分段累加，末段归约后写 y。该映射适合复用每个 x 元素，并改善列主序访问合并程度。若单输出归约很长，则使用多个归约分块并做分层归约，避免单个任务持有过长累加链；是否启用跨块临时归约由设备资源与性能实测决定。

**T/C（转置）**：一个输出对应 A 的一列与 x 的点积，归约维 m 在内存中连续。以输出列为并行维、归约维为连续加载维，连续加载 A 与 x，分段累加后写 y。C 与 T 共用同一路径。

**向量步长**：设备侧以逻辑索引换算物理偏移；负步长基址采用 Host 传入设备缓冲区的物理 span 起点，并在 kernel 中按 `(L-1)*abs(inc)` 确定逻辑首元素。y 的写回仅覆盖逻辑元素位置，不覆盖步长间隙。

### 3.3 Tiling 设计

Tiling 采用运行时 shape 驱动，携带 `m,n,lda,incx,incy,trans,alpha,beta`，以及以下经编译期/设备信息确认的分块参数：输出 tile 大小 `T_out`、归约 tile 大小 `T_red`、归约分段数及所需临时缓冲区信息。输出任务数按 `ceil(len(y)/T_out)` 计算。N 与 T/C 可使用不同 tile 参数，因为二者矩阵访问方向不同。

分块策略：

| 维度/场景 | 映射策略 | 目的 |
| --- | --- | --- |
| 输出维 `len(y)` | 切为 `T_out` 个输出一组，多个核块并行处理 | 提供足够并行度，并复用向量/控制信息 |
| 归约维 `k` | 每个输出/输出组按 `T_red` 分段累加 | 控制寄存器与单任务工作量；覆盖长向量 |
| N | 同一 tile 协同处理邻近行，按列推进 | 缓解列主序下单行归约跨列步长访问 |
| T/C | 按列输出，列内连续归约 | 利用 A 列连续性，提高访存合并 |
| 短归约/小输出 | 单阶段直接累加并写回 | 降低 launch/临时空间开销 |
| 长归约 | 视资源情况分段归约；采用局部 partial 与第二阶段合并，或在单 CTA 内分段累加 | 限制单任务资源占用；数值顺序允许由实现决定 |

性能实现应先以无临时 workspace 的单阶段方案作为基线；若 A2 资源限制或长归约性能表明需要多阶段归约，再引入部分和缓冲区，并在 README/设计文档中记录分配、生命周期和 launch 次数。任务书的“内存要求不涉及”不等于可以无说明增加外部持久内存；临时区须由框架按 stream 生命周期管理。具体 `T_out/T_red`、核函数向量化宽度、是否采用二阶段归约属于平台调优项，必须以 CANN 9.1.0、910B3 的编译与性能数据定值；本工作区没有这些实测依据，故不伪造具体数值。

### 3.4 数值计算

累加使用 FLOAT32，允许 FMA 和并行分块归约；任务不要求跨次确定性，浮点归约顺序可由 tiling 决定。需按任务书 mixed_tolerance 判定：逐元素 `|actual-golden| <= 2^-13 + 2^-13*|golden|`，用例同时满足 matched ratio ≥0.99 和 max absolute error ≤1e-2（或 32 ULP 口径）方通过。Inf/NaN 按任务书的非有限值等价规则单独判定。

## 4. 任务约束逐项核对

| 任务书约束 | 设计处理 | 实现/验收核对点 |
| --- | --- | --- |
| 接口与声明 | 复用 `cann_ops_blas.h` 既有 `aclblasSgemv` 声明，不另建接口 | 签名、参数顺序、返回类型与头文件一致 |
| 行为与错误码 | 采用 §2.1 的 Host 校验及 cuBLAS/仓内 Sgemv 口径 | 空句柄、非法 trans、负维度、非法 lda/stride、空标量/数据指针逐项核验 |
| 跨度溢出 | Host int64 算 `(L-1)*abs(inc)`，拒绝超 int32 范围；先排除 `INT32_MIN` | 负向用例使用 dummy/null 缓冲，不按溢出跨度分配 |
| 非连续布局 | A 按 lda 列间距；x/y 按 incx/incy；正负步长均支持 | `lda>m`、非单位及负步长、步长间隙哨兵值不变 |
| 广播 | 无广播 | m/n/trans 唯一决定各向量逻辑长度 |
| 动态 shape | Host tiling 读取本次运行时参数 | 多组 m/n 复用同一实现，不依赖编译时固定 shape |
| 原地/视图 | y 原地更新，A/x 只读，不生成视图 | 检查 y 逻辑位置，间隙不被覆盖 |
| 确定性 | 不承诺跨次归约位级一致 | 以误差标准验收，不将归约顺序作为接口承诺 |
| 空问题 | 校验后 m=0 或 n=0 成功返回，不读 A/x/y、不启动 kernel | 各 trans 与指针空值组合覆盖；非法参数仍先报错 |
| 0 维 | 不涉及独立 0 维操作数 | 通过 m/n=0 quick return 表达空问题 |
| 异步执行 | 绑定 handle stream 异步提交 | 读取 y 前同步 stream；状态 SUCCESS 仅表示提交成功 |
| alpha/beta 特殊语义 | alpha=0 不短路；beta=0 不读旧 y | 避免 alpha=0+非有限 A/x golden 不一致；beta=0+y=NaN/Inf 覆盖 |
| 工程模式 | Ascend C kernel 直调，Host 校验 + tiling + Device kernel | 路径 `blas/gemv/arch22/`；测试 `test/gemv/sgemv/arch22/`；CMake arch22 注册 GTest |
| 精度与用例 | FLOAT32，Netlib/cblas golden；1000 精度用例与正态补充用例 | `verify_accuracy.py` 结果和日志归档；报告列明补充用例数量 |
| 性能 | 910B3，msprof op task duration；warmup 5，有效样本至少 10 | 按任务书 GPU baseline 逐 case 验证 NPU 平均耗时 ≤ GPU×1.25 |
| 内存 | 任务书标记“不涉及” | 不提交独立内存验收结论；若实现使用临时 workspace，记录其必要性与大小 |

## 5. 已知限制与规避措施

1. **当前工作区不含算子源仓。** 无法核实仓内 `arch22` 骨架、具体 API 名称、头文件行号或 CMake 上下文。实现时应基于目标 ops-blas 分支的 arch22 兄弟算子复制最小骨架，并对照 `arch35/sgemv_host.cpp`，避免臆造仓内接口。
2. **具体 tiling 参数尚未经过 A2/A3 profiling。** 文中给出的是访存方向和分块原则。以 910B3 的编译、精度及性能测试选择 `T_out/T_red`；对 N/T/C 和小/大归约分别调优，逐 case 对照 baseline。若未达标，先检查访存映射、归约分块及 launch 数，再调整 tile。
3. **极端尺寸可能触发索引或资源压力。** 使用足够宽的偏移计算，Host 校验 stride span；检查 `lda*n` 及偏移计算的整数宽度。对超长归约限制单任务资源，必要时分段归约；负向用例不得按巨型非法 stride 分配缓冲。
4. **alpha=0 与非有限输入的参考路径差异。** 按任务书排除此组合的精度用例；alpha 非零时仍覆盖 Inf/NaN 传播。beta=0 需确保 kernel 分支不加载旧 y。
5. **正态分布输入不是生成器默认项。** 按任务书扩展 `gen_csv.py` 或补充 CSV，填充 `GAUSS_NORM_5_2`，将其纳入精度验证并在自测报告中记录数量及生成方法。
6. **GTest 注册需补齐 arch22。** 任务书指出当前 CMake 仅 arch35 注册 GTest；实现需按要求在 arch22 分支调用 `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)`，并在 wrapper 中对 `INT32_MIN` 先短路，确保接口校验用例能到达 Host 入口。
7. **本文不包含实测报告。** 当前目录只有任务书和测试包，未提供编译日志、精度日志、性能日志或目标设备。不得将设计预期写作验收通过；完成设备自验后，应另交任务书规定的步骤说明、精度/性能日志及报告文件。

## 6. 自验证与验收计划

### 6.1 精度与参数行为

- 在 `test/gemv/sgemv/arch22/` 建立 CSV 驱动 GTest，并执行完整 1000 条精度集；在测试报告中注明默认均匀分布和补充正态分布用例的数量。
- 覆盖 N/T/C、非方阵、`lda>m`、正负及非单位 inc、`alpha/beta` 组合、空问题及所有负向校验项。
- 精度 golden 与阈值遵循任务书 §3.2；记录 matched ratio、最大绝对误差、非有限值判定以及逐用例状态。
- 核验 beta=0 时 y 旧值不被使用；核验 stride 间隙、lda padding 不被误读/覆盖；kernel 完成前后的异步行为通过 stream 同步验证。

### 6.2 性能

- 在 Atlas 800T A2 910B3、CANN 9.1.0 运行性能子集 200 条。
- 使用任务书指定 msprof op 采集单算子 task duration；warmup 5 次不计入，之后至少 10 个 SUCCESS 样本取算术平均。
- 逐 case 与 `test_cases/gpu_baseline.csv` 比较，判据 `NPU_avg_us <= 1.25 * GPU_baseline_us`。异常样本排除并在报告说明。
- 分别分析 N 与 T/C、大输出、小输出、长归约和短归约场景；只有实测后才能填写最终性能结论。

### 6.3 合入前检查

- 确认实现和测试路径、arch22 GTest 注册、构建及仓内门禁通过。
- 清理无用代码和临时文件；为算子目录补充 README，说明接口、布局、构建/运行方式和 arch22 限制。
- 将本文档随 PR 合入；按任务书模板组织 `task_submission` 中的步骤、精度、性能和内存材料。内存项按任务书“不涉及”如实说明，不编造测试结果。

## 7. 参考资料

1. 任务书：[`aclblasSgemv_AtlasA2_A3_task_doc.md`](./aclblasSgemv_AtlasA2_A3_task_doc.md)。
2. cuBLAS Library，§2.6.2：<https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf>。
3. Netlib BLAS `sgemv.f`：<https://netlib.org/blas/sgemv.f>。
4. ops-blas gemv 实现：<https://gitcode.com/cann/ops-blas/tree/master/blas/gemv>。
5. ops-blas 贡献指南：<https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md>。
6. mixed_tolerance 标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>。
7. 测试包复现说明：[`test_cases/README.md`](./test_cases/README.md)。


