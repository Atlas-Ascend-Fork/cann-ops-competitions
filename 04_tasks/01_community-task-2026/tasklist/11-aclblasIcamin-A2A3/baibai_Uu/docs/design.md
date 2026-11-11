# 需求背景（required）

## 需求来源

- 任务来源：202611 社区任务讨论（ops-blas，A2/A3），任务名「aclblasIcamin 算子开发(A2/A3)」。
- 开源仓库：https://gitcode.com/cann/ops-blas
- 任务范围：在仓库内新增 BLAS L1 级 `aclblasIcamin` 的 Atlas A2/A3（910B / arch22）实现，与仓库已有 arch35 实现并存，形成按架构分目录的可插拔算子族。
- 对标基线：cuBLAS `cublasIcamin`（A100）、Netlib/cblas `icamin`（精度与语义基线）。

## 背景介绍

`Icamin` 属于 BLAS Level 1 的 index-returning reduction（`i?amin` / `i?amax` 家族），用于求复向量中 Manhattan 模（|Re| + |Im|）最小元素的 **1-based** 索引，是共轭梯度、最小残差选择、pivot 搜索、稀疏预条件等数值库中常用的原语。与 `Icamax` 一起构成对称的一对算子，实现范式高度一致。

仓库现状与差距：

| 项目 | 现状 | 目标 |
| --- | --- | --- |
| 架构支持 | 仅有 arch35 目录实现 | 新增 `blas/iamin/arch22/` 下 `icamin_*` 实现 |
| 硬件 | Atlas A5 系（可使用 SIMT 特性） | Atlas 800I/T A2（910B）、Atlas A3（arch22），**禁止依赖 AscendC SIMT / asc_simt** |
| 访存 | SIMT 下天然支持 stride | 需用 membase gather / 标量地址计算实现 `incx > 1` |
| 归约 | 已有带索引归约范式 | 复用 `Icamax/Isamax` 的「模值计算 + WholeReduceMin(带 index)」范式 |
| 并行 | - | AIV_ONLY 多核拆分 + 两阶段归约合并 |

因此本次开发的核心是把 arch35 上已验证的数学语义，迁移到 arch22 的 Ascend C 约束模型下（无 SIMT、指令受限、访存需显式描述），并保证与 cblas 语义 bit-exact 对齐。

# 需求分析（required）

## 需求描述

实现 `aclblasIcamin`，在设备上对复向量 `x` 求 Manhattan 模最小元素的索引：

```
result = argmin_i ( |Re(x[i])| + |Im(x[i])| ),  i = 0 .. n-1
```

输出为 **1-based** 索引（即 `result = i + 1`）。若多个元素同时取到最小值，取**最小索引**（first occurrence）。

API（handle 式 kernel 直调，Ascend C 工程模式）：

```c
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle,
                              int n,
                              const aclblasComplex* x,
                              int incx,
                              int* result);
```

关键语义要求：

1. `x` 与 `result` 均为 device 侧地址；`handle` 提供 stream、workspace、调度配置。
2. `incx` 为元素步长；`incx == 1` 为连续访问，`incx > 1` 为跨步访问，`incx <= 0` 触发 quick return。
3. quick return：`n <= 0` 或 `incx <= 0` 时，直接向 `*result` 写 `0`（表示无有效元素），不进入 kernel 计算，不返回错误。
4. 并列（tie）取最小索引，需与 cblas `icamin` / cuBLAS `cublasIcamin` 完全一致。
5. 精度要求范围限定为**正步长** `incx >= 1`。

## 需求拆解

| 编号 | 拆解项 | 说明 | 优先级 |
| --- | --- | --- | --- |
| R1 | 模值计算 | `mag = |Re| + |Im|`，COMPLEX64 拆成两个 FLOAT32 分量，Abs + Add | P0 |
| R2 | 带索引最小值归约 | 求 `min(mag)` 及其首次出现索引，采用 `WholeReduceMin(带 index)` 范式 | P0 |
| R3 | 连续访存路径 | `incx == 1`，向量化连续 load，按 512B 对齐分块 | P0 |
| R4 | 跨步访存路径 | `incx > 1`，membase 地址偏移 + gather / 标量读取（禁止 SIMT） | P0 |
| R5 | 多核并行 | AIV_ONLY 分核，各核局部 `(minVal, minIdx)`，再做跨核合并 | P0 |
| R6 | tie-break | 比较使用严格 `<` 或 `(value, index)` 字典序归约，保证取最小索引 | P0 |
| R7 | quick return | `n <= 0` / `incx <= 0` → `*result = 0`，host 侧完成，不下发 kernel | P0 |
| R8 | 索引输出 | device 侧 int32 标量结果，1-based | P0 |
| R9 | tiling 体系 | tilingkey 覆盖连续/跨步、单核/多核、quick return | P0 |
| R10 | 精度验证 | 索引 bit-exact；浮点中间量按 FLOAT32 混合容差 | P0 |
| R11 | 性能验收 | NPU Task Duration ≤ A100 cuBLAS median / 0.8 | P0 |
| R12 | 约束文档化 | n 上限、对齐、incx 取值、workspace 需求等 | P1 |
| R13 | 用例与 CI | 覆盖 quick return、tie、大 n、跨步、非对齐地址 | P1 |

# 详细设计（required）

## 算子分析

### 数学公式

对长度为 `n`、步长为 `incx` 的复向量 `x`，定义第 `i` 个有效元素（0-based 逻辑序）：

```
x_i = x[i * incx],  i = 0, 1, ..., n-1
mag_i = |Re(x_i)| + |Im(x_i)|
```

求解：

```
result = 1 + argmin_{i} mag_i        (1-based 索引)
```

并列规则：

```
若 mag_p == mag_q 且 p < q，则选择 p
等价写法：result = 1 + min{ i : mag_i == min_j mag_j }
```

quick return：

```
if (n <= 0 || incx <= 0) result = 0; return SUCCESS;
```

说明：

- 模定义采用 Manhattan 模（`|Re| + |Im|`），与 Netlib `icamin` / cblas `icamin` 一致，**不是**欧氏模 `sqrt(|Re|² + |Im|²)`；两者在该算子的语义下排序结果通常一致，但必须按 BLAS 定义使用 Manhattan 模以保证边界情形 bit-exact。
- 归约过程不涉及加法累加误差放大，仅含 Abs/Add/Compare，因此最终索引可做到 bit-exact。

### 支持数据类型

| 参数 | 数据类型 | 说明 |
| --- | --- | --- |
| `x` | COMPLEX64 | 即 float32 real + float32 imag 交错存储，单元素 8 字节 |
| `result` | int32 | device 侧标量，1-based 索引；quick return 写 0 |
| 中间量 | FLOAT32 | `|Re|`、`|Im|`、`mag` 均按 float32 计算与比较 |
| 索引 | int32（内部局部可用 uint32） | 归约索引与输出索引均为 int32 |

不支持：COMPLEX16（double 精度）、COMPLEX32FP16 / COMPLEX32BF16 等半精度复数类型、float16/bfloat16 实数输入。如后续扩展，通过 tilingkey 高位做 dtype 区分扩展位预留。

### 支持形状

| 参数 | 形状 | 约束 |
| --- | --- | --- |
| `x` | `[1 + (n-1) * incx]` 逻辑元素数 | 实际显存读取范围为 `x[0 .. (n-1)*incx]` |
| `n` | 标量 int | `1 <= n <= 2^31 - 1`；`n <= 0` 触发 quick return |
| `incx` | 标量 int | `incx >= 1`；`incx <= 0` 触发 quick return |
| `result` | 标量 int（device 指针） | 单元素 4 字节，32B 对齐友好但非强制 |

- `x` 的起始地址按 32B 对齐时可走全向量化路径；否则走非对齐路径（性能下降，功能正确）。
- 单元素 8 字节，`incx == 1` 时数据物理连续，可按 256B/512B 块处理。
- 不支持批量（strided batched / grouped）接口，本任务只做单向量单结果。

### 接口定义

```c
typedef enum {
    ACLBLAS_STATUS_SUCCESS = 0,
    ACLBLAS_STATUS_INVALID_PARAM = 1,
    ACLBLAS_STATUS_RUNTIME_ERROR = 2,
    /* 其余取值沿用 aclblasStatus_t 现有定义 */
} aclblasStatus_t;

typedef struct { float real; float imag; } aclblasComplex;

aclblasStatus_t aclblasIcamin(aclblasHandle_t handle,
                              int n,
                              const aclblasComplex* x,
                              int incx,
                              int* result);
```

行为定义：

| 输入条件 | 行为 | 返回 |
| --- | --- | --- |
| `handle == NULL` 或 `result == NULL` | 参数校验失败 | `ACLBLAS_STATUS_INVALID_PARAM` |
| `n > 0` 且 `x == NULL` | 参数校验失败 | `ACLBLAS_STATUS_INVALID_PARAM` |
| `n <= 0` 或 `incx <= 0` | host 侧向 `*result` 写 `0`，不下发 kernel | `ACLBLAS_STATUS_SUCCESS` |
| 其他 | 下发 kernel，异步于 handle 的 stream，结果写 `*result` | `ACLBLAS_STATUS_SUCCESS` |

调用模式：handle 式 kernel 直调（不走 IR 定义/算子编译流水线），host 侧完成参数校验、quick return、tiling 计算与 kernel launch，kernel 侧为 Ascend C 实现。

## 算子实现

### 实现方案

整体采用「三段式」：

1. **模值计算**：向量化读取 complex64，拆 real/imag，`Abs` + `Add` 得 `mag`（float32）。
2. **局部归约**：每个 AIV 在其数据段内求 `(localMinVal, localMinIdx)`，索引语义为「值最小者保留其首次出现索引」。
3. **全局合并**：跨核按 `(value, index)` 字典序做 `WholeReduceMin` 风格合并（value 更小优先；value 相等时 index 更小优先），最终索引 `+1` 后写 device 标量。

并列处理两种等价实现，取其一（推荐 A）：

- **方案 A（配对归约）**：把 `(mag, index)` 作为一对做带索引的最小值归约，比较器使用字典序，天然满足 tie 取最小索引。
- **方案 B（两遍法）**：先全局求 `minVal`，再顺序/并行扫描取第一个 `mag == minVal` 的索引。多一次遍历，不推荐用于大 `n`。

#### Host侧设计

##### 1. 分核策略

- **模式**：AIV_ONLY（纯向量核，无 scalar 计算需求），与 `Icamax/Isamax` 保持同一范式，便于代码复用与 review。
- **核数选取**：`useCoreNum = min(deviceAivNum, ceil(n / minElemPerCore))`，其中 `minElemPerCore` 取「单核一次 UB 处理块数 × 块内元素数」的量级（经验值约 64K~256K 元素），避免小 `n` 时多核启动与归约开销盖过收益。
- **数据切分**：`n` 按核均分，余数分摊到前 `n % useCoreNum` 个核；每个核处理连续的逻辑下标区间 `[start_i, end_i)`，实际物理地址为 `x + start_i * incx`。
- **跨核合并**：
  - 各核写出 `(localMinVal, localMinIdx)` 到 workspace（每核 8/16 字节，按 32B 对齐 padded）；
  - 由 0 号 AIV（或通过第二次极小规模 kernel/同一 kernel 的收敛分支）对 `useCoreNum` 个候选做标量级字典序比较，得到全局结果，`+1` 后写 `*result`。
  - `useCoreNum == 1` 时省略合并阶段，直接写结果，作为低延迟快路径。
- **quick return**：`n <= 0` 或 `incx <= 0` 时 host 直接 `aclrtMemcpyAsync(result, &zero, 4, H2D)`（或设备侧 0 值填充），返回成功，不参与分核。

##### 2. 数据分块和内存优化策略

- **分块粒度**：以 UB 容量为约束，按 256B/512B 对齐单位切块（complex64 单元素 8B，512B 块 = 64 元素；连续路径可按 4~8 块批量流水）。
- **连续路径（incx == 1）**：
  - 直接向量化 load complex64（等价于 float32×2 交错）；
  - 在 UB 内做 real/imag 分离（可通过 reshape/位拼接视图），`Abs` 后 `Add` 得 `mag`；
  - 就地局部归约，UB 峰值占用控制在单 buffer 量级（模值 + 索引两片区域），不做大规模中间缓存。
- **跨步路径（incx > 1）**：
  - **禁止使用 AscendC SIMT / asc_simt**（A5 才提供）；
  - 采用 membase 模式：以标量地址计算每个元素物理偏移 `base + (start + k) * incx * 8B`，通过 **membase gather** 或按小批（如 32/64 元素）标量地址连续装载的方式把跨步数据聚拢到 UB 连续区，再走与连续路径相同的向量化模值计算；
  - gather 批大小可调，兼顾访存合并效率与 UB 占用。
- **索引生成**：在局部段内为每个元素生成「段内全局逻辑下标」向量（int32），与 `mag` 一同参与字典序归约，避免索引错位。
- **内存访问**：`x` 只读，`result` 单次写，workspace 仅每核 32B 对齐的候选对；全程无额外 device 内存分配（workspace 来自 handle 或 host 侧一次性预留）。
- **精度考虑**：`mag` 全程 float32，不做累加，无误差放大；比较用严格 `<`，浮点 NaN 语义按「NaN 不优于任何有效值」处理，并在约束中声明。

##### 3. tilingkey规划策略

tilingkey 采用「低位 = 调度形态，高位 = 数据类型/扩展位」编码，具体规划如下：

| tilingkey | 编码含义 | 触发条件 | Kernel 路径 |
| --- | --- | --- | --- |
| 0 | quick return | `n <= 0` 或 `incx <= 0` | 不下发 kernel，host 写 0 |
| 1 | 连续 + 单核 | `incx == 1` 且 `useCoreNum == 1` | 连续 load、局部归约、直接写结果 |
| 2 | 连续 + 多核 | `incx == 1` 且 `useCoreNum > 1` | AIV 分核 + 0 号核合并 |
| 3 | 跨步 + 单核 | `incx > 1` 且 `useCoreNum == 1` | membase gather/标量读 + 局部归约 |
| 4 | 跨步 + 多核 | `incx > 1` 且 `useCoreNum > 1` | gather + 分核 + 合并 |
| 5 | 连续 + 非对齐 + 单核 | `incx == 1` 且 `x` 非 32B 对齐 | 头尾标量补齐 + 主体向量化 |
| 6 | 连续 + 非对齐 + 多核 | 同上且多核 | 同上 + 分核合并 |
| 高位 8~15 | dtype 扩展位 | 预留 COMPLEX16 / FP16 等 | 预留，本期不启用 |

说明：

- tiling 结构体包含：`n`、`incx`、`useCoreNum`、`perCoreElem`、`remainElem`、`ubBlockElem`、`gatherBatch`、`x`/`result`/`workspace` 指针。
- tilingkey 由 host 侧 tiling 函数根据输入参数与指针对齐性计算，kernel 侧据此分支，保证单一 kernel 二进制覆盖多形态。
- 所有分支的数学语义完全一致，仅访存与并行形态不同。

#### Kernel侧设计

Kernel 为 Ascend C 实现，AIV_ONLY，结构如下（文字描述，不给出可复制源码）：

1. **初始化阶段**：
   - 解析 tiling，计算本核逻辑区间 `[start, end)` 与物理起始地址 `x + start * incx`；
   - 初始化局部最优：`bestVal = +INF`（float32）、`bestIdx = INT32_MAX`（语义上保证任意有效元素优先）。

2. **数据装载阶段**：
   - `incx == 1`：按 512B 块连续 load complex64 到 UB；
   - `incx > 1`：以 membase 地址偏移 + gather（或小批标量地址装载）把元素聚拢到 UB 连续区；
   - 非对齐首尾元素用标量路径单独处理，保证覆盖全部 `n` 个元素。

3. **模值计算阶段**：
   - real、imag 分离为两个 float32 向量；
   - `Abs` 各自取绝对值，`Add` 得 `mag = |Re| + |Im|`；
   - 生成本块元素的全局逻辑索引向量 `idx = start + k`（int32）。

4. **局部归约阶段**：
   - 在 UB 内对 `(mag, idx)` 做带索引的最小值归约，比较规则为字典序：`mag` 严格小者胜出；`mag` 相等时 `idx` 小者胜出；
   - 采用 `WholeReduceMin`（带 index）语义：若平台接口直接提供索引输出则直接使用；否则以「值最小者保留其索引」的配对归约实现（每轮把最小值广播后再掩码筛选对应索引）；
   - 逐块与历史 `bestVal/bestIdx` 合并，比较使用严格 `<` 保证 first-occurrence 语义。

5. **跨核合并阶段**（多核形态）：
   - 各核把 `(bestVal, bestIdx)` 写入 workspace 对应槽位（32B 对齐）；
   - 0 号核对 `useCoreNum` 个候选做同规则字典序比较，得到全局 `(minVal, minIdx)`；
   - 最终写 `*result = minIdx + 1`（1-based）。

6. **边界与异常**：
   - 单核形态直接写结果，跳过合并；
   - 索引范围校验：`minIdx ∈ [0, n-1]`，若出现越界说明输入异常，按约束处理并在测试中拦截；
   - NaN 输入：不参与「更小」比较（按「NaN 不优于任何有效值」约定），约束中声明为未定义支持项。

## 支持硬件

| 硬件型号 | 架构代号 | 支持情况 | 备注 |
| --- | --- | --- | --- |
| Atlas 800I A2（910B） | arch22 | ✅ | 性能测试设备之一 |
| Atlas 800T A2（910B3） | arch22 | ✅ | **性能测试标准设备** |
| Atlas A3 | arch22 | ✅ | 与 A2 同套 kernel 二进制/源码 |
| Atlas A5 | arch35 | ❌（本期不做） | 已有 arch35 目录实现，保持不动 |

实现目录规划（仓库：https://gitcode.com/cann/ops-blas）：

```
blas/iamin/
├── arch35/                 # 已存在，保持不动
└── arch22/
    ├── icamin_def.h        # 数据结构、tiling 结构体、常量定义
    ├── icamin_tiling.h/.cpp# tiling 计算与 tilingkey 规划
    ├── icamin_host.cpp     # 参数校验、quick return、kernel launch
    ├── icamin_kernel.cpp   # Ascend C kernel（AIV_ONLY）
    └── CMakeLists.txt      # arch22 编译目标
```

## 算子约束限制

| 编号 | 约束 | 说明 |
| --- | --- | --- |
| C1 | 仅支持 COMPLEX64 输入、int32 输出 | 不支持 complex128 / 半精度复数 |
| C2 | 仅支持正步长 `incx >= 1` | 精度验证范围限定为正步长 |
| C3 | `incx <= 0` 或 `n <= 0` 触发 quick return | 向 `*result` 写 `0`，返回 SUCCESS，不视为错误 |
| C4 | `n <= 2^31 - 1` | 索引与计数使用 int32 |
| C5 | `x` 与 `result` 必须为 device 侧有效地址 | 且二者不得别名重叠 |
| C6 | 不支持 batch / grouped 变体 | 单向量单结果 |
| C7 | 不依赖 AscendC SIMT / asc_simt | 跨步访问仅用 membase gather / 标量地址计算 |
| C8 | A2/A3 上 AIV_ONLY 实现 | 不下发 scalar 计算核 |
| C9 | NaN 输入行为未定义 | 约定「NaN 不优于任何有效值」，测试不覆盖 |
| C10 | 与 cuBLAS 对 incx 为负时的行为差异 | 本实现按 quick return 处理，与 cblas 文档中负步长起点规则不同，需在 API 文档中明示 |
| C11 | 并列取最小索引 | 必须与 cblas `icamin` bit-exact 一致 |
| C12 | 需 handle 提供 stream 与 workspace | workspace 大小约 `useCoreNum * 32B`（32B 对齐） |

# 可维可测分析

## 精度标准/性能标准

### 精度标准

| 类别 | 标准 | 验收方式 |
| --- | --- | --- |
| **索引结果（归约索引类）** | **bit-exact** | 与 cblas `icamin` / Netlib 逐 case 严格相等（含 tie 情形），任何 n、incx、输入分布下 `result` 必须完全一致 |
| 浮点中间量 | FLOAT32 混合容差 | `mag = |Re| + |Im|` 等中间值按 FLOAT32 精度计算，允许混合精度规范下的相对误差量级（~1e-6）；因最终只输出索引，容差不直接影响输出，仅用于内部一致性与 tie 边界判定 |
| tie 边界 | 严格相等判定 | 构造 `mag` 完全相等的多元素 case，验证返回最小索引 |

测试用例覆盖（精度）：

| 用例 | 覆盖点 |
| --- | --- |
| `n=1` | 单元素，返回 1 |
| 全等元素（`mag` 全相同） | tie 取最小索引（返回 1） |
| 最小值在首/中/末 | 索引边界正确 |
| 非对齐起始地址 | 头尾补齐路径 |
| `incx = 2/3/8` | 跨步 gather 路径 |
| `n <= 0`、`incx <= 0` | quick return 写 0 |
| 随机大 n | 与 cblas 逐 case 比对（≥1000 组随机） |
| 极端幅值（subnormal、±0） | Abs/Add 边界 |

### 性能标准

- **测试设备**：Atlas 800T A2（910B3），软件版本与仓库 CI 声明的 CANN 版本一致。
- **对标基线**：NVIDIA A100 上 cuBLAS `cublasIcamin`（含 warmup 后的中位数）。
- **验收公式**：

  ```
  NPU Task Duration  ≤  A100 cuBLAS Duration(median) / 0.8
  ```

  即 NPU 端到端 kernel 耗时需达到 A100 cuBLAS 的 80% 以内（相对性能 ≥ 0.8 倍基线的倒数口径，按题述公式执行）。
- **测量口径**：使用 msprof / profiler 抓取 **NPU Task Duration**，warmup ≥ 10 次，采样 ≥ 100 次取 median；A100 侧同口径抓取 cuBLAS 调用耗时 median。
- **典型 case 表**：

  | Case | n | incx | 数据


<!-- design revision: CLA single-commit / design.md only -->
