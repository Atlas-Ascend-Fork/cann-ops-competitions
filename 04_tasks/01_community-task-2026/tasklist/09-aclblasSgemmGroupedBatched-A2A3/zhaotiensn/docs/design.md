# 需求背景（required）

## 需求来源

本设计对应 2026 年 9 月社区任务“aclblasSgemmGroupedBatched 算子开发（A2/A3）”，目标是在 ops-blas 仓库中补齐 Atlas A2/A3（arch22）上的 FP32 分组批量矩阵乘法实现，并按照任务书完成精度、边界和性能自验证。

## 背景介绍

`aclblasSgemmGroupedBatched` 是 BLAS Level 3 分组批量 GEMM 接口，对标 cuBLAS `cublasSgemmGroupedBatched`。与普通 batched GEMM 相比，该接口允许每个 group 使用独立的矩阵尺寸、转置方式、leading dimension、缩放系数和 batch 数量；组内矩阵通过扁平 Host 指针数组传入，指针元素指向 Device 矩阵。

任务书要求支持 Atlas 800I A2/A3 以及 Atlas A2/A3 系列产品。仓库已有 arch35 实现，本任务新增 arch22 Host/Kernel 路径，并保持公共头文件中的 `aclblasSgemmGroupedBatched` 接口不变。

## 现状分析

输入矩阵为 FP32 列主序存储。对第 `g` 个 group 和组内第 `j` 个 batch，扁平下标为：

```text
idx = sum(groupSize[t], t < g) + j
C[idx] = alpha[g] * op(A[idx]) * op(B[idx]) + beta[g] * C[idx]
```

其中 `op(A)` 和 `op(B)` 分别由 `transaArray[g]`、`transbArray[g]` 决定，支持 `N` 和 `T`。接口无 stride/broadcast 语义，A/B/C 均通过独立地址描述。

支持能力如下：

| 参数/能力 | 设计支持 |
| --- | --- |
| 数据类型 | FP32 输入、输出和计算 |
| 分组形状 | 每组独立 `m/n/k`、`lda/ldb/ldc`、alpha、beta、groupSize |
| 转置 | NN、NT、TN、TT 四种组合 |
| 存储布局 | BLAS 列主序，支持 leading dimension padding |
| 空计算 | groupCount、groupSize、m、n 为零时合法 no-op |
| 特殊标量 | `k=0` 或 `alpha=0` 时执行 beta-only |
| 指针语义 | Host 指针数组，元素指向 Device 矩阵地址 |

# 需求分析（required）

## 需求描述

使用 Ascend C/CATLASS 在 Atlas A2/A3 arch22 上实现 FP32 `aclblasSgemmGroupedBatched`，满足任务书规定的公共接口、参数校验、列主序矩阵语义、四种转置组合、空维和 beta-only 规则，并达到任务书给出的精度和性能门槛。

## 需求拆解

1. 保持 `include/cann_ops_blas.h` 中公共 API 签名和异步 handle/stream 语义不变。
2. 校验 handle、数组指针、枚举、维度、groupSize 和 leading dimension；非法参数返回规定状态码。
3. 正确处理不同 group 的独立形状、标量和扁平 batch 地址偏移。
4. 为大矩阵、中等矩阵、转置/非紧凑矩阵和 beta-only 场景选择合适的执行路径。
5. 通过基础回归、扩展合法精度/边界用例以及任务书 5 个性能 case。

# 详细设计（required）

## 算子分析

### 数学公式

对 `g = 0 ... groupCount-1`，组内 `j = 0 ... groupSize[g]-1`：

```text
idx = prefixGroupSize[g] + j
C[idx] = alpha[g] * op(A[idx]) * op(B[idx]) + beta[g] * C[idx]
```

矩阵均按列主序解释。若 `transa=N`，A 的逻辑形状为 `m x k`；若 `transa=T`，A 的物理形状为 `k x m`。B 的规则同理，C 的物理形状为 `m x n`。

### 支持数据类型

仅支持单精度实数 FP32。`ACLBLAS_OP_C` 对实数算子非法，不提供共轭转置语义。

### 支持形状和边界

- `groupCount >= 0`，每组 `m/n/k/groupSize >= 0`；
- `m=0`、`n=0` 或 `groupSize=0` 的 group 为合法 no-op；
- `k=0` 等价于 `C = beta * C`；
- `alpha=0` 跳过矩阵乘法，仅执行 `C = beta * C`；
- 非零计算 group 的 A/B/C 地址必须有效，leading dimension 满足 BLAS 下界；
- 不支持跨矩阵的隐式 broadcast 或 stride 参数。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

Host API 入口按“校验、构造参数、分配 workspace、下发 kernel”执行：

1. 校验 handle 非空、groupCount 和数组参数合法；仅接受 N/T 枚举。
2. 按 group 累加 `groupOffset`，检查 batch 总数和 task 数量的整数溢出。
3. 对 active group 检查 `lda/ldb/ldc` 约束以及每个 batch 的 A/B/C Device 地址；零维或零 batch group 不访问矩阵地址。
4. 将 Host 参数转换成设备侧 `GroupParam` 数组和扁平的 A/B/C 地址数组。`GroupParam` 保存维度、转置、leading dimension、alpha/beta、tile 尺寸和临时 buffer 大小。
5. 根据矩阵形状、转置、leading dimension、alpha/beta 选择 Kernel 工作模式，并计算 256B 对齐的 workspace 布局。
6. 按 handle 绑定的 stream 异步复制 tiling 和地址数组，依次下发所需 kernel；接口调用本身不隐式同步。

##### 1. 分组和任务索引

设备侧 task 使用 `(groupIdx, batchIdx, mTile, nTile, kTile)` 描述唯一计算单元。组内 batch 通过 prefix offset 映射到扁平地址数组；不同 group 的形状和 scale 只从自身 `GroupParam` 读取，避免参数串组。

##### 2. Workspace 和内存布局

workspace 分为：

- `GroupParam[]`：每组固定大小参数；
- `uint64_t A/B/C pointer[]`：扁平 Device 地址；
- packed task descriptor：记录 tile 偏移、有效尺寸、K 分段状态；
- packed A/B/C 临时区：按 chunk 复用，保存 row-major packed tile 和 Cube 部分结果。

所有区域按设备访问要求对齐，并在 Host 侧做容量检查。packed 缓冲区只保留当前 chunk 所需槽位，避免 workspace 随总 task 数线性增长。

##### 3. Tiling 和路径选择

路径选择遵循以下优先级：

| 条件 | 工作模式 | 目的 |
| --- | --- | --- |
| `groupCount=0` 或无有效计算 | no-op | 直接成功返回 |
| `k=0` 或 `alpha=0` | beta-only | 只更新 C，避免无效读 A/B |
| 紧凑大矩阵、`alpha=1,beta=0` | direct Cube | 直接使用 Cube GEMM，减少搬运 |
| 紧凑中等矩阵、多 batch | direct task queue | 单次 launch 批量处理 batch，降低 Host launch 开销 |
| 转置、padding、尾块或 alpha/beta 融合 | PACK-CUBE-EPILOGUE | 统一 packed row-major 语义并隔离尾块 |
| 极瘦矩阵或不适合 Cube 的特殊形状 | AIV fallback | 使用标量/向量分块保证边界稳定性 |

#### 3.2.2 Kernel 侧设计

Kernel 侧采用 Ascend C Vector Core 和 CATLASS/Cube 组合实现，按 tiling 的 phase 字段进入不同处理函数。

##### 1. Direct Cube

对满足条件的大矩阵，Host 将 BLAS 列主序问题映射到 Cube 的矩阵布局。转置映射如下：

| BLAS 组合 | Cube 模式 |
| --- | --- |
| NN | `CBLAS3_GEMM_TRANS_NONE` |
| NT | `CBLAS3_GEMM_TRANS_RIGHT` |
| TN | `CBLAS3_GEMM_TRANS_LEFT` |
| TT | `CBLAS3_GEMM_TRANS_BOTH` |

TT 使用 `MatmulTransBoth`，避免分别解释左右操作数时产生布局不一致。

##### 2. Direct task queue

中等紧凑矩阵为每个 batch 创建 task descriptor，在一个 kernel launch 中由不同 block 处理不同 batch。每个 block 直接读取原始 A/B/C Device 地址，采用 logical one-core Cube GEMM，不再为每个 batch 单独下发 Host launch。

##### 3. PACK-CUBE-EPILOGUE

该路径分为三个按 stream 顺序执行的 phase：

1. **PACK**：AIV 按转置状态和 leading dimension 从列主序 A/B 抽取有效区域，写入 row-major packed tile；尾块无效元素补零。
2. **CUBE**：Cube 对 packed A/B 计算 tile 或 K 分段 partial C，task descriptor 保存 C tile 所有权和 K 段状态。
3. **EPILOGUE**：AIV 执行 `alpha * partial + beta * C_old`，根据原始 `ldc` 以列主序写回 C 的有效区域。

##### 4. AIV fallback 和特殊路径

普通小矩阵采用 M/N/K 分块和 K 维累加，尾块仅写有效行列。beta-only 小尺寸使用标量访问，较大尺寸使用向量分块。对 `m >= 65536 && n <= 8` 的极瘦矩阵使用无 padding 的标量路径，避免尾部 padding lane 影响结果。

##### 5. 同步和错误处理

不同 phase 在同一 handle stream 上按顺序下发，依赖 stream 顺序保证 packed 数据可见性。Kernel 不修改 Host 参数；Host 对 workspace、地址复制和 kernel launch 的 ACL 返回值进行检查并转换为 aclblas 状态码。

##### 6. Legacy AIV 的异构 UB 规划

legacy AIV kernel 会在一次 launch 中按六类 buffer 的最大值初始化 UB。异构 group
分别取最大值时，六个最大值可能来自不同 group，单组均合法但组合后接近或超过实际
UB 容量。Host 在确认不走 direct Cube、direct task 和 PACK-CUBE-EPILOGUE 后，统一将
active group 的 `tileM/tileK/tileN` 限制到 64（`tileK=0` 的 beta-only group 保持
beta-only buffer 语义），并重新计算对齐尺寸与各 buffer 大小，再执行总容量检查。
该规划只影响 legacy fallback，避免改变高性能 Cube 路径的原始 tile 选择。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3） | √ |
| Atlas A2 系列其他产品 | √ |
| Atlas 800I/T A3 | √ |
| Atlas A3 系列其他产品 | √ |

arch22 默认映射到 `ascend910b*` 和 `ascend910_93*` SOC。arch35 现有实现保持不变。

## 算子约束限制

- 仅支持 FP32；
- 仅支持 `ACLBLAS_OP_N` 和 `ACLBLAS_OP_T`；
- 不支持广播和 stride 语义；
- Aarray/Barray/Carray 必须是 Host 指针数组，元素必须指向 Device 矩阵；
- C 与 A/B 不允许存在未定义的重叠写入语义；
- 计算结果采用异步 stream 语义，读取 Device C 前必须同步对应 stream；
- 当前实现不承诺逐位确定性，只满足任务书 mixed tolerance 精度标准。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 接口和状态码 | 合法参数成功，非法参数返回任务书规定状态码；zero dimension/group 为合法 no-op | 任务书 §2.4、§2.5 |
| FLOAT32 精度 | `atol=rtol=2^-13`，matched ratio >= 0.99，max abs error <= 1e-2 或 32 ULP | 任务书 §3.2、生态 mixed tolerance 标准 |
| 基础回归 | 覆盖 NN/NT/TN/TT、标量、padding、边界和负向参数 | 任务书 §3.5、自测 CSV |
| 扩展精度 | 生成器默认 1000 条精度用例，包含组间异构、尺寸扫描、batch、padding、特殊值 | 测试指导 README |
| 性能 | 5 个典型 case 有效采样至少 10 次，平均 NPU kernel/device 区间不高于 `gpu_ms / 0.8` | 任务书 §3.3、测试指导 README |

任务书典型性能门槛如下：

| Case | 形状和 batch | 门槛 |
| --- | --- | ---: |
| TC_PF_0177 | 两组 256^3，batch `[64,64]` | 351.286 us |
| TC_PF_0178 | 两组 512^3，batch `[128,64]` | 3714.125 us |
| TC_PF_0179 | 三组 1024^3，batch `[64,128,64]`，B 转置 | 37651.875 us |
| TC_PF_0180 | 两组 2048^3，A 转置，batch `[128,128]` | 298380 us |
| TC_PF_0181 | 两组 4096^3，batch `[64,64]` | 1188440 us |

## 测试方案

1. 使用 `cmake --build` 或 `build.sh --ops=gemm_grouped_batched --soc=ascend910b3` 编译 arch22 库和 GTest。
2. 运行基础 CSV，确认 54 条回归用例稳定通过。
3. 使用 `test_cases/gen_csv.py` 生成或恢复扩展 CSV，运行 `verify_accuracy.py` 排除 `TC_PF` 后统计精度和边界结果。
4. 对 5 个典型性能 case 预热 5 次，采样至少 10 次；优先使用 msprof 获取 `Task Duration(us)`，若设备 profiling channel 不可用，则记录故障并使用同一 stream 的 `aclrtEvent` 设备区间作为替代证据。
5. 对源码执行 `git diff --check` 和 OAT 扫描，交付前删除 build、profiler、Python cache 等临时产物。

## 兼容性分析

本实现复用公共 `aclblasSgemmGroupedBatched` 声明，不改变已有 arch35 接口和行为。新增文件位于 `blas/gemm_grouped_batched/arch22/` 和 `test/gemm_grouped_batched/arch22/`，由 SOC 到 arch22 的现有 CMake 自动选择。公共 README 的产品支持表新增 A2/A3 支持说明，其他产品仍使用各自已有架构实现。

