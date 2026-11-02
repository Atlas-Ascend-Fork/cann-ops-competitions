# aclblasGemmBatchedEx 算子设计文档

# 需求背景

## 需求来源

[10月社区任务-aclblasGemmBatchedEx算子开发（A2A3）](https://www.hiascend.com/activities/task-center)要求在 Atlas A2/A3 系列产品上，基于 [ops-blas](https://gitcode.com/cann/ops-blas) 开源仓工程框架，使用 Ascend C 编程语言实现扩展版批量通用矩阵乘法算子 `aclblasGemmBatchedEx`。

接口语义、精度标准和性能判据以任务材料、实际 `test_cases`、验证脚本和 `gpu_baseline.csv` 为准。算子接口复用 `include/cann_ops_blas.h`，不定义产品私有的平行接口。

仓库和主要材料位于：

```text
/mnt/workspace/gitCode/ops-blas
```

主要参考路径包括：

- `docs/10月社区任务-aclblasGemmBatchedEx算子开发（A2A3）/test_cases/`
- `docs/task_submission/3.5 性能验收结论.md`
- `docs/task_submission/3.6 性能优化尝试记录.md`
- `docs/task_submission/optimization_loop/`
- `docs/MSPROF_USAGE.md`
- `blas/gemm_batched_ex/`
- `test/gemm_batched_ex/`

验收材料实际包含 1000 条精度用例和 200 条性能用例，共 1200 条。设计文档不以任务描述中的概括数字替代实际 CSV 和脚本口径。

## 背景介绍

`aclblasGemmBatchedEx` 对 `batchCount` 个维度一致的独立矩阵三元组执行：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
```

其中 `i = 0 ... batchCount-1`，A、B、C 均为列主序矩阵，通过 device pointer array `Aarray/Barray/Carray` 传入。各批次共享 `m/n/k`、`transa/transb`、`alpha/beta` 和 leading dimension，C 原地覆写，各批次的 C 存储区域不得重叠。

`transa/transb` 支持 N、T、C 三种操作。实数路径下 C 与 T 等价。`op(A_i)` 的逻辑形状为 `m×k`，`op(B_i)` 的逻辑形状为 `k×n`，C 的逻辑形状为 `m×n`。

性能实现需要同时处理以下硬件和接口特点：

1. Cube 计算适合对齐的分形 tile，用户矩阵却可能具有任意合法的 M/N/K 和 leading dimension。
2. 短 K、K tail 和非对齐 N 会放大 MTE2、ND2NZ、L1/L0 装载和 Fixpipe 的固定开销。
3. C 的列主序和非对齐 `ldc` 可能导致 strided Fixpipe 写回。
4. arch22 的 L0A/L0B 容量限制了 128-wide tile 和双缓冲组合。
5. 高层 Matmul task API 对 task 生命周期、非连续 GM 输出和异步搬运有明确限制。
6. AIC Cube、AIV 打包/转置/尾处理和事件同步的总耗时必须按实际 Task Duration 计算。

### 当前实现状态

当前安全基线为 Stage B，源码基线为 `7825e50`。已经验证并接入的专用路径如下：

- **1087**：`micro_1087_pack_kernel` + `micro_1087_grid_kernel`。AIV 负责 A padding 和 B 转置打包，AIC 负责 19 个窄 N 输出块，K chunk 为 128，直接写回 C。
- **1115**：`CONFIG_NORM + OUTER_PRODUCT` 专用路径，使用 `singleCoreK=64`、`basicK=64`、`preloadN=2`、`enableDoubleCache` 和连续 block 到 task 的映射。

当前真实验收状态：

| 项目 | 结果 |
| --- | --- |
| 精度用例 | 1200/1200 PASS |
| 性能用例 | 198 PASS / 2 FAIL / 0 ERROR |
| 1087 | PASS，约 14.71–14.96us |
| 1115 | PASS，约 18.56–19.13us |
| 1111 | 数值 PASS，约 33.3–33.7us，性能 FAIL |
| 1129 | 当前安全路径约 48.859us，性能 FAIL |
| 1075 | PASS |
| 4 条典型大 case | PASS |

因此当前文档描述的是已验证实现和后续设计目标的组合，不能把当前状态写成 200/200 已完成。

# 需求分析

## 算子功能

对第 `i` 个批次执行：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
```

其中：

- `op(A_i)` 为 `m×k`；
- `op(B_i)` 为 `k×n`；
- `C_i` 为 `m×n`；
- 矩阵使用列主序；
- `trans=N` 取原矩阵，`trans=T` 取转置，`trans=C` 取共轭转置，实数路径下 C 与 T 等价。

边界行为对齐 cuBLAS/Netlib SGEMM 的 quick return 语义：

- `m=0`、`n=0` 或 `batchCount=0` 是合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`；
- `k=0` 时只执行 `C=beta*C`，不读取 A、B；
- `alpha=0` 时跳过矩阵乘，执行 `C=beta*C`；
- `beta=0` 时不得读取旧 C；
- 参数校验先于 no-op 判定；
- 非法维度、非法枚举、非法 leading dimension、空指针和不支持组合返回对应状态码。

## 公开接口与范围

接口声明复用 `include/cann_ops_blas.h`，函数原型为：

```cpp
aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* const Aarray[],
    aclDataType Atype,
    int lda,
    const void* const Barray[],
    aclDataType Btype,
    int ldb,
    const void* beta,
    void* const Carray[],
    aclDataType Ctype,
    int ldc,
    int batchCount,
    aclblasComputeType_t computeType,
    aclblasGemmAlgo_t algo);
```

设计目标支持范围按参考文档定义：

| 项目 | 支持范围 |
| --- | --- |
| 产品 | Atlas A2/A3 |
| 输入类型 | FP16、BF16 |
| 计算类型 | FP16 输入支持 COMPUTE_16F/COMPUTE_32F；BF16 输入支持 COMPUTE_32F |
| 输出类型 | 按 Ctype 量化输出，具体组合以 README 和测试材料为准 |
| 不支持类型 | FP8 及未实际实现和验证的组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 形状 | `m/n/k/batchCount` 运行时合法非负值 |
| 布局 | 列主序、BLAS leading dimension |
| 批处理 | device pointer array；不要求 broadcast |
| 计算语义 | C 原地覆写，不要求确定性计算 |

设计目标支持和当前已经通过的验收范围需要分开理解：README、实际代码和 `test_cases` 必须只声明已经实现并验证的 dtype/computeType 组合，不能因设计目标存在就扩大产品声明。

## 接口约束

leading dimension 约束如下：

| 参数 | 条件 | 约束 |
| --- | --- | --- |
| `lda` | `transa=N` | `lda >= max(1,m)` |
| `lda` | `transa=T/C` | `lda >= max(1,k)` |
| `ldb` | `transb=N` | `ldb >= max(1,k)` |
| `ldb` | `transb=T/C` | `ldb >= max(1,n)` |
| `ldc` | 所有情况 | `ldc >= max(1,m)` |

`aclblasSetStream` 绑定执行 stream，调用方在读回结果前需要按 ACL 约定同步 stream。A、B、C 矩阵必须驻留 device memory，各批次 C 缓冲区不得重叠。

# 详细设计

## 算子分析

### 数学公式

```text
C_i(r,c) = alpha * sum(p=0..k-1) op(A_i)(r,p) * op(B_i)(p,c)
           + beta * C_i(r,c)
```

其中 `r∈[0,m)`、`c∈[0,n)`、`p∈[0,k)`。FP16/BF16 输入在 Cube 上使用对应的累加模式；COMPUTE_32F 路径在 fp32 域完成累加及 alpha/beta 合成，再按 Ctype 量化一次。

### 支持数据类型

设计范围为：

- FP16 输入：COMPUTE_16F 或 COMPUTE_32F；
- BF16 输入：COMPUTE_32F；
- alpha/beta：按接口和 computeType 使用 Host 侧标量；
- FP8 及其他未验证组合：Host 侧返回 NOT_SUPPORTED。

精度实现必须与实际 README、公共接口和 CSV 测试保持一致。若某个组合只有设计方案而没有代码和真实测试证据，不能将其写成已验收能力。

### 支持形状

`m`、`n`、`k` 和 `batchCount` 为运行时参数，覆盖：

- 1、质数、2 的幂、2 的幂加减 1；
- 对齐和非对齐的 M/N/K；
- 短 K 和 K tail；
- 紧凑 leading dimension 和多种 padding；
- 大规模矩阵及多批次组合。

不支持广播、任意额外的非连续 Tensor view 或批次间 C 缓冲区重叠。

## 算子实现

总体架构采用“Cube 矩阵乘主路径 + AIV 转置/打包/尾处理路径 + Host 形状分发”的分层结构：

- AIC/Cube 负责主要矩阵乘和 Mmad；
- AIV 负责转置 staging、padding、尾块复制以及必要的 epilogue；
- Host 根据转置属性、矩阵尺寸、leading dimension、标量参数和对齐属性选择通用路径或专用路径；
- 所有新增辅助 kernel 的 Task Duration 都计入总耗时。

### 3.2.1 Host 侧设计

#### 参数校验

Host 侧执行顺序如下：

1. 检查 `handle`，为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 检查 `transa/transb`，非法值返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. 检查 `m/n/k/batchCount`，负值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 检查 `alpha/beta`、batch pointer array 和 leading dimension。
5. 检查 Atype/Btype/Ctype、computeType 和 algo 是否在支持集合中。
6. 参数合法后再处理 no-op、`k=0`、`alpha=0` 和 `beta=0`。

当 `batchCount>0` 时，Aarray、Barray、Carray 不能为空。参数校验不能因 m/n/batchCount 为零而提前跳过测试材料要求的指针和 leading dimension 检查。

#### 列主序视角转换

C 为 `m×n` 列主序矩阵时，其内存布局等价于 C 转置后的行主序视角。实现可以将问题转换为：

```text
C^T = op(B)^T * op(A)^T
```

对应地交换 m/n、A/B、lda/ldb 和转置关系，使 Cube 使用适合当前 arch22 的方向。该转换必须同时更新：

- A/B 的物理地址和 stride；
- tile 的 M/N 坐标；
- C 的列主序写回坐标；
- K tail 的 padding 范围；
- Fixpipe 的 mSize/nSize。

不得只交换形状而忽略物理 leading dimension 或 pointer array。

#### 路径决策

Host 按以下顺序选择路径：

1. **退化路径**：`k=0` 或 `alpha=0` 时使用 AIV early-exit，对 C 执行 beta 缩放；`alpha=0,beta=1` 可以直接 no-op。
2. **微形状路径**：当矩阵乘工作量很小、存在 M/N/K 微维或无法高效使用 Cube 时，使用 AIV 向量/标量路径，避免 Cube 固定开销。
3. **专用性能路径**：当转置、尺寸、leading dimension、batch、alpha/beta 和对齐条件完整匹配时，进入 1087 或 1115 专用 kernel。
4. **通用 Cube 路径**：其他合法输入使用 Stage B 通用路径。

专用分发必须基于矩阵属性和对齐条件，不能按 `TC_PF_1087`、`TC_PF_1115` 等用例名称判断。

#### staging 和 workspace

当转置输入的物理布局不适合 Cube 的 L0 装载时，使用 AIV staging 将输入按 16×16 子块转为 Cube 友好的 NZ/连续布局。workspace 按 batch 和形状计算，并保证每个 batch 的基址按 32B 对齐。

当 C 的直接 Fixpipe 写回不能满足 leading dimension 或尾块要求时，可以使用 padded temporary workspace，再执行有效区域复制。但是 padded workspace、copy 和尾块裁剪都属于算子总成本，必须在性能采样中计入。

#### 任务和核数分配

Cube task 基本单位为一个 batch 的输出 tile。Host 根据总 task 数和 AIC 核数计算 `usedCores`，尽量使用满核；不能均分时，将余数分配给前面的核。AIV 辅助 kernel 根据 tile 数和 AIV 核数分配任务。

1087 和 1115 的专用映射如下：

- 1087：19 个窄 N 输出块，内部 K chunk=128，AIV pack 和 AIC grid 分开计时。
- 1115：连续 block 到 task 的映射，同一 N 行内的 M task 连续，复用 A tile；K=51 使用单个 64 对齐 K block。

### 3.2.2 Kernel 侧设计

#### Cube 主路径

单个 Cube task 的基本流水为：

```text
GM -> L1 -> L0A/L0B -> Mmad -> L0C -> Fixpipe -> GM(C)
```

K 分块时，每个块执行：

```text
LoadData/ND2NZ -> L0 装载 -> Mmad 累加
```

K tail 通过对齐后的 padding 处理，padding 区必须清零，Mmad 只将有效区域写回 C。对于 `alpha/beta` 非 direct path，先将原始积写入 fp32 temporary，再由 AIV epilogue 完成合成和量化。

#### 事件和 buffer 生命周期

L1、L0A、L0B、L0C 的复用必须按照 producer/consumer 配对事件执行：

1. MTE2 完成 GM 到 L1 的搬运后，才能执行 L1 到 L0 的装载。
2. L0A/L0B 数据准备完成后，才能启动 Mmad。
3. Mmad 完成后，才能复用 L0A/L0B 或启动 Fixpipe。
4. Fixpipe 完成后，才能复用 L0C 或写入同一 C tile。
5. 双缓冲下，每个 slot 使用独立的状态和事件，不能以无条件 PipeBarrier 替代真实依赖。

全局 SyncAll、per-tile CrossCore wait 或不匹配的 SetFlag/WaitFlag 可能使同步成本超过 Cube 节省，除非有实测证据，否则不作为默认结构。

#### 1087 专用 kernel

1087 的已验证结构为：

```text
micro_1087_pack_kernel  (AIV)
  A padding + B transpose/pack

micro_1087_grid_kernel  (AIC)
  19 output blocks
  K chunk = 128
  first chunk direct write
  later chunks use verified atomic accumulation
  direct C writeback
```

真实结果：

- 数值 `matched=1.000000`；
- `max_abs=5.49316406e-4`；
- 3 次独立 msprof 约 `14.708us / 14.714us / 14.964us`；
- pack 和 grid 的 Task Duration 均计入。

该路径必须冻结，除非新证据证明存在不改变数值和性能的安全修改。

#### 1115 专用 kernel

1115 的已验证结构为 `CONFIG_NORM + OUTER_PRODUCT`：

- `singleCoreK=64`；
- `basicK=64`；
- `preloadN=2`；
- `enableDoubleCache`；
- `SetOrgShape/SetSingleShape` 移出 task 循环，只有 tail 重新设置；
- 连续 block→task 映射；
- K=51 进入单个 64 对齐 K block。

当前 1115 的 15 次独立 msprof 中 14 次低于 19.131us，四组 3 次采样连续通过，典型结果包括 `18.669us`、`18.840us`、`18.696us`。该路径不得被 1111/1129 的实验修改覆盖。

#### AIV 向量/标量辅助路径

AIV 辅助路径用于：

- 微形状矩阵乘；
- M/N 尾行或尾列；
- A/B 转置 staging；
- padded C 的有效区域复制；
- `alpha=0`、`k=0` 等 early-exit；
- fp32 temporary 到用户 C 的 epilogue。

AIV 侧按 16×16 或 32B 对齐的数据块组织搬运。尾块必须按有效 M/N 裁剪，不能把分形 padding 写入用户有效区域。

#### 1111 和 1129 后续结构

1111 当前数值正确但性能约 33.3–33.7us，主要瓶颈为 C 的 Fixpipe 写回。后续可以探索 per-tile C staging、连续 workspace、Fixpipe 与下一 tile 的重叠以及减少 strided writeback 的布局，但 staging/copy/epilogue 必须计入总耗时。

1129 当前 `basicM/basicN` 任一为 128 且 `basicK=128` 时出现 `AICORE_EXCEPTION 507015`，根因与 L0A/L0B 双缓冲容量和生命周期有关；64/64 虽可运行但约 69.08us，不具备性能价值。后续应先验证 128×128×64、128×64×64 或 64×128×64 的实际 L0 分配和单/双缓冲组合，再进行性能采样。

### 3.2.3 资源和调度

设计时需要审计每个 tile 的资源：

| 资源 | 设计要求 |
| --- | --- |
| L1 | A/B tile 按 K/M/N 对齐，复用前等待搬运完成 |
| L0A/L0B | 根据 basicM/basicN/basicK 计算实际字节数，确认单/双缓冲不越界 |
| L0C | 按 M/N tile 和 Fixpipe 格式分配，确认复用顺序 |
| UB | AIV pack、epilogue 和 copy 按 32B 对齐，尾块先清零再写有效数据 |
| workspace | 每个 batch 和 tile 的基址对齐，所有 copy 成本计入性能 |
| AIC 核 | 根据 Cube task 数均分，避免只有少量核工作 |
| AIV 核 | pack/copy task 独立均分，必要时与 AIC 流水重叠 |

任何 128-wide 方案必须在 kernel 运行前打印或计算 L0A/L0B 的实际分配、slot 间距、offset 和总字节数，不能只根据理论 tile 尺寸判断容量足够。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas 800I/T A2 | √ | 目标验证环境，Ascend910B3，arch22 |
| Atlas 800I/T A3 | √ | 需要在对应 SoC/CANN 环境重新构建和验证 |

算子代码位于 `blas/gemm_batched_ex/arch22`，测试工程位于 `test/gemm_batched_ex`。当前性能数据来自 Atlas 800T A2/Ascend910B3，不能直接外推到未实测的芯片或 CANN 版本。

## 算子约束限制

- dtype 支持范围以 README、公共接口和实际测试为准；设计目标为 FP16/BF16，FP8 及其他未验证组合返回 NOT_SUPPORTED。
- 仅支持列主序和 `lda/ldb/ldc` 语义内的访问，不支持额外的任意非连续 Tensor view。
- 不支持广播；每个 batch 使用独立 device pointer。
- C 原地覆写，各批次 C 缓冲不得重叠。
- `beta!=0` 时 C 是有效输入；`beta=0` 时不得读取旧 C。
- 直接写回路径只能写用户有效的 `[0,m)×[0,n)` 区域；padded workspace 的 padding 不得泄漏到用户有效区。
- 浮点累加顺序与参考 BLAS 可能不同，允许在验收容差内误差；`alpha=0` 等精确路径按 EXACT 规则验证。
- 性能口径要求矩阵和 device pointer array 驻留 Device 内存，并使用真实 Task Duration。
- arch22 的 L0 容量、Fixpipe strided 写回、高层 Matmul task API 和 AIC/AIV 事件能力限制可用的 tile 结构。

# 可维可测分析

## 精度标准与验证

golden 由 cblas/Netlib SGEMM 按 batch 生成，逐 batch、逐矩阵比较 C。精度脚本和 CSV 是验收契约，覆盖：

- N/T/C 转置组合；
- 紧凑和 padding leading dimension；
- 对齐、非对齐和短 K；
- 小矩阵、质数、2 的幂、±1 形状和大矩阵；
- batchCount=0、1、小批量和较大批量；
- alpha/beta 的 0、1、负值和较大值；
- Inf/NaN 和所有负向参数校验；
- FP16/BF16 的容差和 alpha=0 的 EXACT 语义。

当前真实精度结果为：

```text
1200/1200 PASS
```

每个新性能路径还必须先通过以下最小门禁，再进行 msprof：

1. TT/TN/NN 最小数值 case；
2. TC_L0；
3. TC_ED_216 和 TC_EX_255；
4. 目标 case 连续调用至少 50 次；
5. 相关精度分块和完整精度回归。

数值 mismatch、stream error、hang、segfault、AICORE_EXCEPTION 或重复调用数据污染均判定该候选失败。

## 性能标准与验证

性能判定以实际性能 CSV、`gpu_baseline.csv` 和验证脚本为准。当前关键性能门槛如下：

| case | 门槛 | 当前结果 | 判定 |
| --- | ---: | ---: | --- |
| 1087 | `<18.423us` | 14.71–14.96us，含 pack+grid | PASS |
| 1111 | `<30.586us` | 33.3–33.7us | FAIL |
| 1115 | `<19.131us` | 18.56–19.13us | PASS |
| 1129 | `<36.598us` | 约 48.859us | FAIL |

标准采样方式：

- warmup=10；
- 有效 samples=20；
- runs=30；
- retries=1；
- 至少 3 次独立采样；
- 使用 `msprof --task-time=l1 --ai-core=on` 或仓库已验证的等价脚本；
- 记录 mean、median、min、max 和原始 Task Duration；
- pack、staging、copy、reduction、epilogue 和辅助 kernel 全部计入；
- 不删除慢样本，不以一次擦线结果宣称通过。

完整验收目标为：

```text
TC_PF：200 PASS / 0 FAIL / 0 ERROR
精度：1000 PASS / 0 FAIL
TC_PF_1075：PASS
4 条典型大 case：全部 PASS
```

当前为 `198/200`，因此仍未完成最终验收。

## 兼容性分析

算子兼容性要求如下：

- 使用仓库现有公共接口、handle、stream 和状态码；
- 保持 cuBLAS/Netlib 风格的转置、列主序、leading dimension、alpha/beta 和 batch 语义；
- 通用路径和专用路径对相同合法输入产生一致结果；
- 专用路径仅对满足完整条件的输入启用，其他输入回退到安全实现；
- 不修改其他 BLAS 算子和验收 CSV；
- 不把未在目标 SoC/CANN 组合上验证的 dtype 或硬件能力写成已支持。

主要兼容性风险来自 CANN 9.1/arch22 的底层限制：L0 双缓冲容量、Fixpipe 的非连续 GM 写回、高层 Matmul API 的异步输出约束，以及 AIC/AIV 事件同步成本。涉及这些能力的修改必须在目标设备上重新执行精度、稳定性和性能验证。

## 交付和当前遗留项

交付应包含：

1. 算子源码和构建接入；
2. 公共接口声明和参数校验；
3. 测试工程、CSV 驱动入口和 README；
4. 1000 条精度结果及原始日志；
5. 200 条性能结果及 profiler 摘要；
6. 1075 和 4 条典型 case 回归；
7. 每轮优化的 patch、数值日志、稳定性日志和性能 CSV；
8. 设计文档和自测报告。

在上述两项解决并完成全量复验前，不得写“200/200 验收完成”。
