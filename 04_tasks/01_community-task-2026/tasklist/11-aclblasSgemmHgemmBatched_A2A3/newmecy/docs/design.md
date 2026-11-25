# 需求背景（required）

## 需求来源

本算子来源于 CANN 社区任务 2026（ops-blas 开源仓社区任务，任务编号：11-aclblasSgemmHgemmBatched_A2A3，代码仓：https://gitcode.com/cann/ops-blas ）。要求在昇腾 NPU（Atlas A2/A3 系列产品，含 Atlas 800T A2 / Atlas 800T A3）上，使用 Ascend C 编程语言并基于 ops-blas 开源仓工程框架，开发句柄式（handle-based BLAS）批量通用矩阵乘算子 `aclblasSgemmBatched`（单精度 FP32）与 `aclblasHgemmBatched`（半精度 FP16，`aclblasHalf`），分别对标 cuBLAS `cublasSgemmBatched` / `cublasHgemmBatched` 接口与 Netlib BLAS `sgemm` / `hgemm` 参考计算语义，完成算子设计、实现与自验证，验收通过后合入昇腾算子开源仓。

## 背景介绍

### aclblasSgemmBatched / aclblasHgemmBatched 算子实现现状分析

在 ops-blas 开源仓中，批量矩阵乘目录 `blas/gemm_batched/` 下已有面向 Ascend 950（arch35）的实现（`arch35/gemm_batched_host.cpp` 与 `arch35/hgemmBatched_host.cpp`），且 `blas/gemm_batched/arch22/` 下已有复数批量矩阵乘 `cgemm_batched_*` 以及 `blas/gemm/arch22/` 单矩阵乘作为工程参考。但目前代码仓缺少面向 Atlas A2/A3 系列（`arch22` / `dav-2201` 指令集）的实数批量通用矩阵乘 `aclblasSgemmBatched` 与 `aclblasHgemmBatched` 的正式实现。

算子接口已预先声明于 `include/cann_ops_blas.h`（`aclblasSgemmBatched` 位于 L606，`aclblasHgemmBatched` 位于 L616），无需修改公共头文件。对标 cuBLAS 接口所支持的能力及参数规范如下表所示：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 库上下文句柄，携带 stream | aclblasHandle_t | aclblasHandle_t | 非空有效句柄 | 标量 |
| transa | op(A) 的运算类型（N/T/C） | aclblasOperation_t | 枚举 | 实数算子下 OP_C 与 OP_T 等价 | 标量（属性） |
| transb | op(B) 的运算类型（N/T/C） | aclblasOperation_t | 枚举 | 实数算子下 OP_C 与 OP_T 等价 | 标量（属性） |
| m | op(A) 与 C 的行数 | int | int | m ≥ 0，m=0 为合法 no-op | 标量（属性） |
| n | op(B) 与 C 的列数 | int | int | n ≥ 0，n=0 为合法 no-op | 标量（属性） |
| k | op(A) 的列数、op(B) 的行数 | int | int | k ≥ 0，k=0 仅做 C=beta*C | 标量（属性） |
| alpha | 标量乘数，位于 Host 内存 | const T* | S: float; H: aclblasHalf | 非空指针 | 标量（Host 内存） |
| Aarray | 输入矩阵 A 的 Device 指针数组 | const T* const[] | S: float; H: aclblasHalf | 长度 batchCount；逐批 Device 首址 | 列主序，lda×k (N) 或 lda×m (T/C) |
| lda | A 矩阵的前导维度 | int | int | N: lda ≥ max(1, m); T/C: lda ≥ max(1, k) | 标量（属性） |
| Barray | 输入矩阵 B 的 Device 指针数组 | const T* const[] | S: float; H: aclblasHalf | 长度 batchCount；逐批 Device 首址 | 列主序，ldb×n (N) 或 ldb×k (T/C) |
| ldb | B 矩阵的前导维度 | int | int | N: ldb ≥ max(1, k); T/C: ldb ≥ max(1, n) | 标量（属性） |
| beta | 标量乘数，位于 Host 内存 | const T* | S: float; H: aclblasHalf | 非空指针；beta=0 时不读 C 旧值 | 标量（Host 内存） |
| Carray | 输出矩阵 C 的 Device 指针数组（原地覆写） | T* const[] | S: float; H: aclblasHalf | 长度 batchCount；逐批 Device 首址 | 列主序，ldc×n（逻辑 m×n） |
| ldc | C 矩阵的前导维度 | int | int | ldc ≥ max(1, m) | 标量（属性） |
| batchCount | 批量大小 | int | int | batchCount ≥ 0，batchCount=0 为合法 no-op | 标量（属性） |

计算公式：

$$C[i] = \alpha \cdot \operatorname{op}(A[i]) \cdot \operatorname{op}(B[i]) + \beta \cdot C[i], \quad i = 0, 1, \dots, \text{batchCount}-1$$

### aclblasSgemmBatched / aclblasHgemmBatched 算子功能分析

- **算子功能**：对 `batchCount` 个相互独立的同规模矩阵三元组并行执行标准通用矩阵乘累加（Batched GEMM），并将结果原地写回 $C[i]$ 对应的逻辑 $m \times n$ 内存区域。
- **输入**：`handle`、`transa`、`transb`、`m`、`n`、`k`、`alpha`、`Aarray`、`lda`、`Barray`、`ldb`、`beta`、`ldc`、`batchCount`。
- **输出**：`Carray`（原地覆盖输出，仅更新每个 $C[i]$ 的逻辑 $m \times n$ 区域，前导维 padding 列保持不变）。
- **支持数据类型**：
  - `aclblasSgemmBatched`：`alpha`、`beta`、`Aarray`、`Barray`、`Carray` 均为单精度浮点型 `FLOAT32`（`float`）。
  - `aclblasHgemmBatched`：`alpha`、`beta`、`Aarray`、`Barray`、`Carray` 均为半精度浮点型 `FLOAT16`（`aclblasHalf`）。
- **支持广播**：不支持广播。本算子遵循标准 BLAS Level-3 矩阵乘契约，各矩阵操作数独立寻址；所有 Batch 共享相同的维度规模与转置属性，但各批次数据相互独立，无广播行为。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，在 Atlas A2/A3（`arch22`）硬件平台上实现 `aclblasSgemmBatched` 与 `aclblasHgemmBatched` 两个算子：

1. **接口兼容**：参数序列与语义与 cuBLAS `cublasSgemmBatched` / `cublasHgemmBatched` 完全对齐，通过 handle 绑定 stream 实现 NPU 异步执行。
2. **布局与转置**：支持输入与输出矩阵按列主序（Column-Major）带填充（leading dimension）访问；支持 `transa` / `transb` 分别取 `ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）、`ACLBLAS_OP_C`（实数算子下与转置等价）。
3. **指针数组与独立批次**：通过位于 Device 侧的二级指针数组 `Aarray`、`Barray`、`Carray` 寻址各批次矩阵首地址，支持各批次矩阵在物理内存上非连续分布。
4. **原地更新与写保护**：$C[i]$ 采用 In-place 原地覆写语义；当 $ldc > m$ 时，严格保护超出逻辑行范围的 padding 区域不被破坏。
5. **完备的边界与退化处理**：
   - $m=0 \lor n=0 \lor \text{batchCount}=0$：合法空操作（quick return），直接返回 `ACLBLAS_STATUS_SUCCESS`，不引用指针数组元素，不启动 Kernel。
   - $k=0 \lor \alpha=0$：不执行矩阵乘；当 $\beta \ne 1$ 时仅执行 $C[i] = \beta \cdot C[i]$（$\beta=0$ 时置零）；当 $\beta=1$ 时直接成功返回；此时不引用 `Aarray` 与 `Barray`。
   - $\beta=0$：不读取 $C[i]$ 初始值（即使旧数据包含 NaN 也不受影响）。
6. **精度与性能双百达标**：
   - 精度满足生态算子开源精度标准（Sgemm 匹配率 $\ge 99\%$，容差 $2^{-13}$；Hgemm 匹配率 $\ge 99\%$，容差 $2^{-10}$）。
   - 性能在 Atlas 800T A2 (910B3) 上通过 `msprof op` 真实采集 kernel Task Duration，达到 cuBLAS GPU 标杆耗时（`gpu_ms`）的 $0.8\times$ 以上性能（即 $\text{Task Duration} \le \text{gpu\_ms} / 0.8$）。

## 需求拆解

1. **Host 入口校验与防御**：实现严格的参数合法性前置检查，按照 `handle` $\to$ 枚举值 $\to$ 维度非负 $\to$ 前导维下界 $\to$ 标量指针 $\to$ 指针数组 的确定性顺序校验，并映射标准错误码。
2. **列主序到硬件 Cube 的数学映射**：利用列主序转置等价性 $C^T = \operatorname{op}(B)^T \cdot \operatorname{op}(A)^T$，将 BLAS 列主序输入直接重解释映射为行主序 Cube MMAD 计算，避免全局显存转置搬运开销。
3. **两级任务网格与分核调度**：设计结合 Batch 维度与空间维（$M, N$）的二维切分策略，将任务均匀分发至 Atlas A2/A3 的物理 AIC/AIV 核心，支持大 Batch 小矩阵与单 Batch 大矩阵的自适应负载均衡。
4. **FP32 与 FP16 算子内核流水**：
   - Sgemm：采用 strict FP32 纯精度计算（关闭 HF32 模式），确保满足严格的 $2^{-13}$ 混合容差标准。
   - Hgemm：采用 FP16 数据搬入与高吞吐 Cube 乘加，并在 L0C 累加器中保持 FP32 精度累加，写回时转换回 FP16，兼顾吞吐与数值稳定性。
5. **标量 Epilogue 与快慢路径分流**：
   - 快路径（$\alpha=1.0, \beta=0.0$）：Cube 计算结果直接写回 $C[i]$，跳过中间暂存与向量 Epilogue，达成最优性能。
   - 通用路径（$\alpha \ne 1.0 \lor \beta \ne 0.0$）：通过向量核 Epilogue 或 Matmul 原位累加完成标量缩放与旧值叠加。
   - 退化路径（$k=0 \lor \alpha=0$）：绕过 Cube 单元，由 AIV 向量核直接执行 $C[i] = \beta \cdot C[i]$ 快速缩放。
6. **测试集与验证自动化**：集成 1200 条全量测试用例（Sgemm 600 条 + Hgemm 600 条，含 1000 条精度用例与 200 条性能用例），支持故障注入、边界覆盖与真实硬件 Profiling 结果比对。

# 详细设计（required）

## 算子分析

### 数学公式

算子计算逻辑为逐批执行独立 GEMM：

$$C[i] = \alpha \cdot \operatorname{op}(A[i]) \cdot \operatorname{op}(B[i]) + \beta \cdot C[i], \quad i \in [0, \text{batchCount}-1]$$

由于 BLAS 规范中所有矩阵均按**列主序（Column-Major）**存储，在昇腾 NPU 硬件上，利用矩阵转置恒等式：

$$C[i]^T = \operatorname{op}(B[i])^T \cdot \operatorname{op}(A[i])^T$$

令：
- $X[i] = \operatorname{op}(B[i])^T$，逻辑规模为 $n \times k$；
- $Y[i] = \operatorname{op}(A[i])^T$，逻辑规模为 $k \times m$；
- $D[i] = C[i]^T$，逻辑规模为 $n \times m$。

则 $D[i] = X[i] \cdot Y[i]$ 为标准**行主序（Row-Major）**矩阵乘法：
- 当 $\text{transb} = \text{ACLBLAS\_OP\_N}$ 时，以列主序物理存储且前导维为 $ldb$ 的 $B$ 矩阵，其首行元素在内存中沿列跨步分布，但在行主序视角下，其转置矩阵 $B^T$ 正好呈现连续紧凑分布，能够直接以零内存拷贝的方式由 Cube 单元通过 ND2NZ 指令高效加载；
- 当 $\text{transa} = \text{ACLBLAS\_OP\_N}$ 时，同理处理 $A$ 矩阵。
- 实数算子下共轭转置 $\text{ACLBLAS\_OP\_C}$ 与普通转置 $\text{ACLBLAS\_OP\_T}$ 等价，不进行共轭取反。

### 支持数据类型

1. **`aclblasSgemmBatched`**：
   - 矩阵元素：`float`（单精度 IEEE-754 FP32，4 字节）。
   - 标量系数：`alpha`、`beta` 均为 `const float*`（Host 内存）。
   - 片上计算模式：AIC Cube 单元采用 IEEE FP32 MMAD 乘加，显式关闭 HF32 转换（`SetHF32(false)`），确保累加精度满足 $2^{-13}$ 容差标准。
2. **`aclblasHgemmBatched`**：
   - 矩阵元素：`aclblasHalf`（半精度 IEEE-754 FP16，2 字节）。
   - 标量系数：`alpha`、`beta` 均为 `const aclblasHalf*`（Host 内存）。
   - 片上计算模式：AIC Cube 单元采用 FP16 输入乘法，在 L0C 内部以 FP32 浮点累加器保存中间态，写回 GM 时收窄为 FP16，有效防止大矩阵乘加溢出。

### 支持形状

- 任意满足 $m \ge 0, n \ge 0, k \ge 0, \text{batchCount} \ge 0$ 的非负整数；
- 矩阵前导维约束：
  - $\text{transa} = \text{ACLBLAS\_OP\_N}$ 时 $lda \ge \max(1, m)$；$\text{transa} \in \{\text{OP\_T}, \text{OP\_C}\}$ 时 $lda \ge \max(1, k)$；
  - $\text{transb} = \text{ACLBLAS\_OP\_N}$ 时 $ldb \ge \max(1, k)$；$\text{transb} \in \{\text{OP\_T}, \text{OP\_C}\}$ 时 $ldb \ge \max(1, n)$；
  - $ldc \ge \max(1, m)$。
- 全面覆盖小维度（$1 \le m, n, k \le 16$）、非对齐尾块（如 $31, 63, 65$）、大维度（$1024, 2048, 4096$）及非连续跨步 padding 场景。

## 算子实现

### 实现方案

总体架构采用句柄式 stream 直调方案，区分核心计算通路与边界退化通路：

1. **Host 调度层**：负责入参防御校验、零维/零标量快速返回判定、Workspace 资源分配与 Tiling 结构体封装。
2. **Device 核心计算通路**：
   - **Path A（Cube 快路径：$\alpha=1.0, \beta=0.0$）**：启动 `GemmBatchedKernel`，AIC 核心遍历分配的批次与分块，通过 Device 指针数组读取对应批次首地址，完成 MMAD 乘加后直接写回 $C[i]$ 逻辑区域。
   - **Path B（通用路径：$\alpha \ne 1.0 \lor \beta \ne 0.0$）**：Cube 核心计算中间结果至 Handle 预分配的临时 Workspace 中，紧随同 stream 启动 AIV Vector Epilogue Kernel 完成 $C[i] = \alpha \cdot \text{Temp}[i] + \beta \cdot C[i]$ 原位写回；或对于支持的形状采用 Matmul Epilogue 原位融合。
   - **Path C（退化缩放路径：$k=0 \lor \alpha=0.0$）**：不启动 Cube 核心，直接分发 AIV Vector Scale Kernel 执行 $C[i] = \beta \cdot C[i]$（$\beta=0$ 时快速置零）。

#### 3.2.1 host侧设计：

##### tiling策略：
- **分块粒度选择**：
  - 在 Atlas A2/A3 上，Cube 基本分形计算单元为 $16 \times 16$。
  - 对于大矩阵（$m, n, k \ge 256$），Tile 尺寸设置为 $\text{tileM} = 128, \text{tileN} = 128, \text{tileK} = 128$（或 $64 \times 64 \times 64$），以充分利用 L1/L0 片上缓冲区；
  - 对于瘦高/扁平矩阵，按主维度做非对称分块（如 $256 \times 64$ 或 $64 \times 256$）；
  - 对于超小矩阵（$m, n \le 16$），Tile 尺寸设为最小分形 $16 \times 16$，避免引入过多 padding 无效计算。
- **K 轴切分与多级流水**：采用片内 K 循环外加双缓冲（Ping-Pong Buffer），使 MTE2 从全局内存搬运至 L1 与 Cube MMAD 计算高度重叠。

##### 任务均分：
- **空间与 Batch 联合网格**：
  - 令 $m\text{Blocks} = \lceil m / \text{tileM} \rceil$，$n\text{Blocks} = \lceil n / \text{tileN} \rceil$。
  - 单个 Batch 的分块数为 $\text{tilesPerBatch} = m\text{Blocks} \times n\text{Blocks}$。
  - 全局总任务数为 $\text{totalTasks} = \text{batchCount} \times \text{tilesPerBatch}$。
- **核心映射与均衡**：
  - 获取物理 AIC 核心数 $N_{\text{aic}}$（Atlas 800T A2 上为 20）。
  - 设置启动 Block 数 $\text{usedAicCores} = \min(\text{totalTasks}, N_{\text{aic}})$。
  - 每个核心采用网格步进持久循环（Grid-Stride Loop）：从核编号 $\text{blockIdx}$ 开始，以 $\text{usedAicCores}$ 为步长遍历所有任务：
    $$\text{taskId} \in [\text{blockIdx}, \text{blockIdx} + \text{usedAicCores}, \dots < \text{totalTasks}]$$
  - 每个任务解算为三元组：
    $$\text{batchId} = \text{taskId} / \text{tilesPerBatch}$$
    $$\text{localTileId} = \text{taskId} \% \text{tilesPerBatch}$$
    $$\text{tileMIdx} = \text{localTileId} / n\text{Blocks}, \quad \text{tileNIdx} = \text{localTileId} \% n\text{Blocks}$$
  - 这种调度方式保证当 $\text{batchCount} \gg N_{\text{aic}}$ 时，各核分配到的批次数量极度均衡；当单 Batch 大矩阵时，各核分配到的空间 Tile 均匀分布。

##### 批量搬运：
- 搬运基准为 32 字节对齐（256-bit 突发传输）。
- 对于连续列，使用 `DataCopy` 高速块搬运；
- 对于非对齐边界或带 $lda/ldb$ 步长的子矩阵，采用 `DataCopyPad` 或 2D 参数块搬运，并在尾块设置哨兵掩码，避免跨界非法读写。

##### 1. 分核策略：
1. **满核优先原则**：只要总任务数 $\ge N_{\text{aic}}$，一律打满硬件所有可用 AIC 核心（20 核）；若任务数较少，则动态收缩核心数至实际任务数，避免空跑核的调度开销。
2. **AIV 协同分核**：对于向量 Epilogue 或纯 Scale 路径，根据 AIV 物理核心数（40 核）进行划分，每个核心分配连续的列块，保证写回带宽最大化。

##### 2. 数据分块和内存优化策略：
1. **L1 Bank 冲突消除**：L1 缓冲区采用物理 Bank 隔离划分，Ping-Pong 双缓冲交替放置在不同物理存储页，避免 MTE1 与 MTE2 同时读写 L1 产生总线冲突。
2. **Workspace 动态管理与零运行期分配**：算子利用 handle 预分配机制（`EnsureDefaultWorkspace`），默认复用 32 MiB 缓存空间（不足时阶梯扩容至最高 2 GiB），彻底消除 API 调用过程中的同步 `aclrtMalloc` / `aclrtFree` 开销。
3. **原地 Padding 保护**：在写回 $C[i]$ 时，只对 $m \times n$ 的有效矩形区域发起 GM 写入，跨步间隙（$ldc - m$）通过行步长跳过，严格保护调用方在 padding 区预置的哨兵数据。

##### 3. tilingkey规划策略：
采用整数字段直接传参或位掩码组合：
- `tilingKey = 0`：Sgemm 纯直出路径（$\alpha=1.0, \beta=0.0$）；
- `tilingKey = 1`：Sgemm 通用路径（含标量 Epilogue）；
- `tilingKey = 2`：Hgemm 纯直出路径（$\alpha=1.0, \beta=0.0$）；
- `tilingKey = 3`：Hgemm 通用路径（含标量 Epilogue）；
- `tilingKey = 4`：退化标量缩放路径（$k=0 \lor \alpha=0.0$）。

##### 数据检测：
严格遵循以下校验顺序：
1. `handle != nullptr`，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `alpha != nullptr && beta != nullptr`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`；
3. `m >= 0 && n >= 0 && k >= 0 && batchCount >= 0`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. `transa, transb \in {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}`，否则返回 `ACLBLAS_STATUS_INVALID_ENUM`；
5. 前导维下界校验：
   - $\text{transa} = \text{OP\_N} \implies lda \ge \max(1, m)$；$\text{transa} \in \{\text{OP\_T}, \text{OP\_C}\} \implies lda \ge \max(1, k)$；
   - $\text{transb} = \text{OP\_N} \implies ldb \ge \max(1, k)$；$\text{transb} \in \{\text{OP\_T}, \text{OP\_C}\} \implies ldb \ge \max(1, n)$；
   - $ldc \ge \max(1, m)$；任一不满足返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. 结构合法性校验后检查快速返回（Quick Return）：
   - 若 $m=0 \lor n=0 \lor \text{batchCount}=0$，立即返回 `ACLBLAS_STATUS_SUCCESS`，允许数组指针为 `nullptr`；
   - 若 $k=0 \lor \alpha=0$：允许 `Aarray`、`Barray` 为 `nullptr`；但若 $m>0 \land n>0$，`Carray` 必须非空且有效；若 $\beta=1$，直接成功返回；若 $\beta \ne 1$，启动退化缩放 Kernel。
7. 正向计算路径非空校验：当 $\text{batchCount}>0 \land m>0 \land n>0 \land k>0 \land \alpha \ne 0$ 时，`Aarray`、`Barray`、`Carray` 必须全部非空，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`。

#### 3.2.2 kernel侧设计：

算子 Kernel 执行分为 `Init` 与 `Process` 两大阶段：

1. **Init 阶段**：
   - 初始化 `TPipe` 内存管道，申请必要的 L1、L0A、L0B、L0C 与 UB 队列/缓冲；
   - 初始化 `matmul::MatmulImpl` 矩阵乘计算对象，配置形状参数、数据格式（`CubeFormat::ND`）与计算精度（Sgemm 关闭 HF32，Hgemm 开启 FP32 累加）；
   - 从入参中解包 Device 指针数组基地址 `AarrayGm`、`BarrayGm`、`CarrayGm`。
2. **Process 阶段**：
   - **外层持久循环**：当前核循环遍历归属的 `taskId`，通过 `batchId` 解算出对应批次。
   - **指针解引用**：从 `AarrayGm[batchId]`、`BarrayGm[batchId]`、`CarrayGm[batchId]` 中分别读取当前 Batch 的矩阵物理起始地址，分别绑定为当前的 `GlobalTensor`。
   - **CopyIn（搬入）**：MTE2 单元根据当前空间 Tile 的坐标与前导维跨步，将矩阵分块由 GM 加载至 L1 缓冲区，并自动进行双缓冲交替。
   - **Compute（计算）**：
     - MTE1 将 L1 数据转换为分形排布载入 L0A 与 L0B；
     - Cube MMAD 指令驱动矩阵乘加，结果写入 L0C；
     - 沿 K 轴完成全部子块累加。
   - **CopyOut（搬出）与 Epilogue**：
     - 若为直出模式，Fixpipe 直接将 L0C 数据按照列主序转置映射写回 $C[i]$ 目标显存；
     - 若为 Epilogue 模式，数据暂存至临时 Workspace，AIV 向量核执行标量乘加 $C[i] = \alpha \cdot \text{Temp} + \beta \cdot C[i]$ 后写回；
     - 严格确保仅写入有效矩阵元素，Padding 列保持原样。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2 推理产品 | √ |
| Atlas 800T A2 训练产品 | √ |
| Atlas 800T A3 训练产品 | √ |

实现严格基于 `arch22`（`dav-2201`）指令集与 Ascend C 基础/高阶 API（禁止使用 `arch35` 特有的 SIMT API 或 Tensor API），构建与测试在 CANN 9.1.0 稳定环境下验证。

## 算子约束限制

1. **数据类型**：`aclblasSgemmBatched` 仅支持单精度 `FLOAT32`；`aclblasHgemmBatched` 仅支持半精度 `FLOAT16`。不支持复数或混合精度（混合精度由 `GemmEx` 系列承担）。
2. **数据布局**：矩阵仅支持标准 BLAS 列主序（Column-Major）带前导维度布局；不支持任意非连续步长视图或 Row-Major 输入。
3. **批次组织**：批次为 Uniform Batch，所有 Batch 共享相同的维度 $m, n, k$、前导维 $lda, ldb, ldc$ 及转置枚举；不支持 Ragged/Grouped 异构形状。
4. **内存重叠**：调用方必须保证各批次 $C[i]$ 存储空间互不重叠，且 $C[i]$ 不得与任意 $A[j]$ 或 $B[j]$ 重叠。
5. **参数校验**：严格在 Host 入口拦截非法枚举、负维度、低于下界的前导维及空指针，返回对应 BLAS 标准状态码。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 1. `aclblasSgemmBatched` 对标 CPU Netlib `cblas_sgemm`（采用 float64 累加后收窄为 FP32）；满足生态算子开源精度标准：$rtol = 2^{-13} \approx 1.2\times 10^{-4}$，$atol = 2^{-13} \approx 1.2\times 10^{-4}$，匹配率 $\ge 99\%$，最大绝对误差上限 $\le \max(10^{-2}, 32 \times \text{ULP})$。<br>2. `aclblasHgemmBatched` 对标 CPU FP32 浮点累加（float64 累加后收窄为 FP32，再按 IEEE 最近偶数舍入收窄至 FP16）；满足生态算子开源精度标准：$rtol = 2^{-10} \approx 9.77\times 10^{-4}$，$atol = 2^{-10} \approx 9.77\times 10^{-4}$，匹配率 $\ge 99\%$，最大绝对误差上限 $\le \max(10^{-1}, 32 \times \text{ULP})$。<br>3. 全量 1000 条精度用例（Sgemm 500 条 + Hgemm 500 条）通过率必须达到 100%（0 FAIL / 0 SKIP）。 | 生态算子开源精度标准（mixed_tolerance_standard）、任务书 §3.2 |
| 性能标准 | 1. 测试硬件：Atlas 800T A2 (910B3)。<br>2. 正式判据（kernel-to-kernel）：采用 `msprof op` 在 NPU 上真实采集单算子底层 kernel Task Duration（微秒 us）；执行前 warmup 5 次（不计入样本），后续连续有效采样 10 次取算术平均。<br>3. 达标门槛：NPU 平均耗时 $\le \text{gpu\_ms} / 0.8$（即性能达到 A100 GPU 同规格 cuBLAS 基线耗时的 $0.8\times$ 以上）。<br>4. 辅助判据：GTest 端到端 `[PERF] avg_us`（通过 `aclrtEvent` 计时）仅用于交叉验证，不替代硬件底层 Task Duration 正式判据。<br>5. 全量 200 条性能用例（Sgemm 100 条 + Hgemm 100 条）达标率必须达到 100%（0 FAIL / 0 SKIP）。 | 任务书 §3.3、gpu_baseline.csv |

## 兼容性分析

1. **接口声明兼容**：接口原型完全复用 `ops-blas` 统一对外头文件 `include/cann_ops_blas.h` 中已有的声明（`aclblasSgemmBatched` 与 `aclblasHgemmBatched`），无需对公共对外头文件进行任何新增或修改，确保与历史版本和上层调用栈 ABI 100% 兼容。
2. **状态码兼容**：所有返回值统一采用 `cann_ops_blas_common.h` 中定义的标准 `aclblasStatus_t` 枚举，错误码语义与 cuBLAS / CANN BLAS 系列接口规范保持严格一致。
3. **架构编译兼容**：实现文件独立落位于 `blas/gemm_batched/arch22/`，由 `blas/CMakeLists.txt` 根据目标 SoC（`ascend910b*` / `ascend910_93*`）自动检索并编入 `arch22` 库，不影响现有的 `arch35` 构建流程，实现多架构解耦共存。
