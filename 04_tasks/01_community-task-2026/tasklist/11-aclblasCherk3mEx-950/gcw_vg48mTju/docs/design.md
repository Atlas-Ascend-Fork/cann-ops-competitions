# aclblasCherk3mEx 算子设计

## 1. 需求背景

### 1.1 需求来源

11 月 CANN 社区任务《aclblasCherk3mEx 算子开发任务书》。在 Ascend 950PR 上使用 Ascend C 实现扩展的单精度复数 Hermitian 秩-k 更新。函数签名对标 [cuBLAS `cublasCherk3mEx`](https://docs.nvidia.com/cuda/cublas/index.html#cublascherk3mex)，工程入口沿用 `ops-blas` 的 handle 和 stream。

### 1.2 仓内实现现状

| 参考代码 | 可复用的设计 |
| --- | --- |
| `blas/herk/arch35/cherk_host.cpp`、`cherk_kernel.cpp` | 列主序寻址、句柄与标量处理、三角写回和对角虚部处理。现有乘积路径拆成四次独立实数 GEMM，扩展接口需要自己的 3M kernel。 |
| `blas/gemm/arch35/cgemm_onchip_kernel.h`、`cgemm_compute_kernel.cpp` | 交错复数解包、FP32 Cube 乘加、片上分块、AIV/AIC 同步和分段归约。Hermitian 的操作数方向及写回边界单独定义。 |
| `blas/common/helper/host_utils.h`、`aclblas_handle_internal.h` | 标量地址位置查询、stream 和设备资源查询。 |
| `test/herk/cherk/` | CSV 参数化 GTest、CBLAS golden 及设备资源管理。 |

## 2. 需求分析

### 2.1 接口与布局

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha,
    const void* A, aclDataType Atype, int lda, const float* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

| 参数 | 类型与形状 | 约束 |
| --- | --- | --- |
| `handle` | BLAS 句柄 | 使用该句柄绑定的 stream。 |
| `uplo` / `trans` | 枚举 | `UPPER` 或 `LOWER`；`OP_N` 或 `OP_C`。`OP_T` 返回 `INVALID_ENUM`。 |
| `n` / `k` | 非负整数 | 分别为 C 的阶数和秩-k 更新的归约维度。 |
| `A` / `Atype` | 只读 Device 矩阵 | `ACL_COMPLEX64` 为交错 FP32 实部和虚部，每个复元素 8 B。`ACL_INT8` 表示交错 `int8` 实部和虚部，每个复元素 2 B。 |
| `lda` | A 的列步长，以复元素计 | N 模式下 `lda ≥ max(1,n)`，A 逻辑形状 `[n,k]`、存储形状 `[lda,k]`。C 模式下 `lda ≥ max(1,k)`，A 逻辑形状 `[k,n]`、存储形状 `[lda,n]`。 |
| `alpha` / `beta` | FP32 实标量指针 | 支持 Host 和 Device 标量。 |
| `C` / `Ctype` | 原地更新的 Device 矩阵 | `Ctype = ACL_COMPLEX64`。逻辑形状 `[n,n]`，存储形状 `[ldc,n]`。 |
| `ldc` | C 的列步长，以复元素计 | `ldc ≥ max(1,n)`。 |

选定三角的 C 元素参与更新，对角输入虚部忽略。写回时对角输出虚部置为 `+0.0f`，另一三角和每列末尾的 padding 保持原状。调用方保证 A 的可读区域与 C 的可写区域分离。`lda`、`ldc` 只描述列主序的列间跨度。

### 2.2 需求拆解

1. 实现参数校验、退化路径、两种 A 类型及两种转置模式。
2. 为每个输出块在一次 kernel 启动内完成 Hermitian 3M、三角内的 `alpha/beta` 更新及原地写回。
3. 以独立 CBLAS 参考按实部和虚部验精度，以 `msprof op` 逐例采集设备耗时并匹配附件基线。

## 3. 详细设计

### 3.1 数学模型与地址计算

设 `X = op(A) = R + iI`，结果为 `C ← α·X·Xᴴ + β·C`。以下使用从 0 开始的索引。

| `trans` | `X(i,t)` 来源 | 逻辑实部和虚部 |
| --- | --- | --- |
| `OP_N` | `A[i + t*lda]` | `R = A.real`，`I = A.imag` |
| `OP_C` | `A[t + i*lda]` | `R = A.real`，`I = -A.imag` |

INT8 两个分量各自符号扩展并精确提升到 FP32，再做变号与加减。例如 C 模式下 `int8(-128)` 的虚部变成 `+128.0f`，在原始 8 位整数上取负会溢出。C 的地址始终是 `i + j*ldc`。元素偏移及乘字节宽度后的地址使用 64 位整数。

三次实数乘和组合式如下。第二操作数来自 `Xᴴ`，因此交叉项是 `I·Rᵀ − R·Iᵀ`。

```text
P0 = R · Rᵀ
P1 = I · Iᵀ
P2 = (R + I) · (R − I)ᵀ

U.real = P0 + P1
U.imag = (P2 − P0) + P1
```

以 `n=2, k=1, X=[1+2i, 3+4i]ᵀ` 为例，`U(0,1)=11+2i`，`U(1,0)=11−2i`，对角实部分别为 5 和 25。N 模式用该列作为 A，C 模式用 `[1−2i, 3−4i]` 作为 A 的一行，两者的选定三角应一致。该手算例用于检查共轭、矩阵左右操作数和列主序回写方向。

### 3.2 Host 侧设计

按以下顺序校验，结构参数合法后才处理快速返回。表中状态码省略 `ACLBLAS_STATUS_` 前缀。

| 顺序 | 检查 | 错误返回 |
| --- | --- | --- |
| 1 | `handle != nullptr` | `HANDLE_IS_NULLPTR` |
| 2 | `uplo`、`trans` 属于支持枚举 | `INVALID_ENUM` |
| 3 | `n`、`k` 非负，`lda`、`ldc` 满足各自下界 | `INVALID_VALUE` |
| 4 | `alpha`、`beta` 非空，`Atype/Ctype` 为两种支持组合 | 空标量 `INVALID_VALUE`，不支持的组合 `NOT_SUPPORTED` |
| 5 | 正阶数时 `C` 非空，秩-k 项有效时 `A` 非空 | `INVALID_VALUE` |

`n=0` 时允许 A、C 为空，结构校验完成后直接返回，不解引用数据或标量。正阶数时通过仓库的 `CheckPtrLocation` 分别识别两个标量的地址位置。Host 标量读入 tiling。Device 标量首版沿句柄 stream 读回，在 Host 需要作退化分派及返回空 A 状态时等待标量读回完成，保证获取同一 stream 上此前写入的值。标量读取可能等待；随后的矩阵 kernel 在绑定 stream 上异步下发。C 正阶数时必须有效，即使后续命中 quick return。

| 条件 | 下发路径 | 数据访问 |
| --- | --- | --- |
| `n=0` | 直接返回 | 无 A/C 访问 |
| `k=0` 或 `alpha=0`，且 `beta=1` | 直接返回 | C 保持原样，包括原对角虚部，允许空 A |
| `k=0` 或 `alpha=0`，且 `beta≠1` | AIV 三角缩放 | 允许空 A。`beta=0` 直接写零，其他值读选定三角并缩放 |
| `k>0` 且 `alpha≠0` | 小任务 AIV 3M 或常规 MIX 3M | 读取 A。`beta=0` 时跳过旧 C 读取 |

Host 通过平台接口获取 AIC/AIV 资源。tiling 记录 `n/k/lda/ldc`、类型及转置、三角方向、标量、输出块尺寸、K panel 深度、归约跨度、块数量与启用核数。分派依据是形状、布局和平台容量。范围判断及字节乘法均检查整数溢出后再下发。

### 3.3 分核、分块与片上预算

常规路径取方形输出块，令 `Q=ceil(n/B)`。上三角按列枚举 `tileRow ≤ tileCol`，列 `c` 的起始任务编号是 `c(c+1)/2`。下三角枚举 `tileRow ≥ tileCol`，列 `c` 起始编号为 `c(2Q−c+1)/2`。两者各有 `Q(Q+1)/2` 个任务，使用 64 位编号。每个工作组按 `blockIdx + t*usedCoreNum` 取任务，单组负责块内全部 K 段。非对角块整体处于选定三角，对角块逐列裁剪，矩阵尾块按有效行列数裁剪。

首版从 `B=64`、K panel 深度 `64` 起步。若三角块数不足以覆盖可用 AIC，可评估 `B=32` 提升并行度。极小矩阵或短 K 时使用 AIV，每个向量任务负责一段同列输出元素，在寄存器内完成三个实数点积。待上板后以 AIV 和 MIX 的实测交叉点设置分派阈值。

| 存储位置 | `B=64`、K panel 为 64 的初始规划 |
| --- | --- |
| L1 | 左右各 3 个 FP32 `64×64` panel，双缓冲，共 `2×6×64×64×4 = 192 KiB`，另计格式对齐。 |
| L0A / L0B | 每侧每份 FP32 panel 为 16 KiB。首版每侧预留两个槽，共 32 KiB，依照实际 `Mmad` 装载粒度切分。 |
| L0C | 三个 FP32 `64×64` 累加块约 48 KiB。完成整个归约段后转出，槽再次使用前等待消费者释放。 |
| UB | AIV 输入打包阶段复用原始交错缓冲与三组实数 panel。输出阶段按半块处理，三组结果共 24 KiB，复数累加及误差补偿各最多 16 KiB，另为 C 搬入/搬出与掩码保留空间。按输入和输出同时存活的峰值校验，不重复计算可复用区域。 |

容量按仓内 arch35 `cgemm_tiling_data.h` 的片上限制和实际布局对齐核算，并在编译期校验固定缓冲偏移。设计将三组中间量限制在块内，不为每次计算分配三张 `n×n` GM 临时矩阵。输入、输出槽共用规则按阶段明确，首版先验证单输出槽，再启用双缓冲。

### 3.4 Kernel 数据流

1. **AIV 备料。** 根据 trans 把两个行面板解包为逻辑 `R/I`，准备左侧 `R/I/(R+I)` 和右侧 `R/I/(R−I)`。使用 FP32 实数和 Cube 所需的分形布局。对于非紧凑 `lda` 按列跨度搬入，K 尾部在片上补零，GM 只读有效元素。
2. **AIC 计算。** 在同一次 MIX kernel launch 中，依次对三组 panel 执行 FP32 `Mmad`。每个归约段开始时清零 L0C，后续 panel 累加到对应的 `P0/P1/P2`。利用仓内原生 FP32 搬运和乘加路径，首版关闭 HF32 降精度模式。三个结果在本组内完成归约并通过 Fixpipe 转给 AIV。
3. **AIV 合并。** 各段先转换成复数增量，再对复数增量按顺序累加。待该块全部 K 段完成，仅执行一次 `alpha` 缩放与 `beta*C` 更新。`beta=0` 时从零起算。只读取选定三角，且对角只读取旧 C 的实部。
4. **AIV 写回。** 两个 AIV 子核按不相交的列片段处理回写。有效元素使用精确长度的搬运。对角短段、尾部及非对齐地址使用按元素或按有效字节数写回；GM 端不补写完整对齐块。对角虚部在最终写回前统一置零。

MIX 组内采用仓内 `cgemm` 的 AIV/AIC 同步方式。输入槽按“可写 → AIV 完成写入 → AIC 完成读取 → 可写”流转，输出槽由 Fixpipe 完成后通知 AIV，AIV 完成读取后释放。同步序号随 panel 和归约段递增，尾块和空子片段也执行既定握手，避免一侧提前退出。参考官方 [CrossCoreSetFlag](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0273.html) 的 950PR 组内同步规则，实际以 CANN 9.1.0 编译及设备验证为准。

### 3.5 舍入与特殊值

- `P2−P0+P1` 在虚部接近零时容易发生抵消。常规路径初始用 128 的 K 归约跨度，每段用原生 FP32 累加并先合并为复数，再对复数段和做补偿归约。以全量精度用例比较 64、128、256 的跨度，记录每类形状的误差和设备耗时。小任务采用同一分解，避免不同路径因共轭符号或重复缩放产生系统性偏差。
- 验证 `+/-1e4` 标量、实虚部抵消、INT8 极值和长 K 时的绝对误差。若原生 FP32 分段仍碰到精度上限，先收短归约段并提高段间归约精度。评估 3M 预加减与合并误差时保留每个实数乘的分量，定位误差发生在预加减、乘加还是组合步骤。
- 非有限输入的 `Inf−Inf` 及 `0×Inf` 可能在 3M 中引入额外 NaN。打包阶段记录非有限值标记，回写前对受影响输出分量走同一次 launch 内的 `cblas_cherk` 语义补算，分别比较 NaN、同号 Inf 和有限值。标记来自参与乘积的有效输入，填充区不参与扫描。对角实部按两分量的平方和处理。纯缩放和 quick return 路径先判定是否引用操作数，以覆盖 `alpha=0` 且 A 含 NaN、`beta=0` 且旧 C 含 NaN 的场景。

### 3.6 工程文件

| 位置 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 公共函数声明。 |
| `blas/herk/arch35/cherk3m_ex_host.cpp` | 参数校验、标量读取、tiling 及下发。 |
| `blas/herk/arch35/cherk3m_ex_kernel.cpp`、`cherk3m_ex_tiling_data.h` | AIV 缩放、小任务 3M、MIX 3M 及 tiling 字段。 |
| `test/herk/cherk3m_ex/arch35/` | CSV、GTest、golden 和设备端辅助验证。 |
| `blas/herk/README.md`、`test/herk/cherk3m_ex/README.md` | 接口与硬件说明、测试和性能采集步骤。 |

`bash build.sh --ops=cherk3m_ex --soc=ascend950` 构建目标库与测试，GTest 目标为 `cherk3m_ex_test`。新增入口依赖公共 handle 和 arch35 基础能力，与独立开发的 `aclblasCherkEx` 无链接依赖。自验环境为 Ascend 950PR、CANN 9.1.0。

## 4. 可维可测分析

### 4.1 功能与精度

按附件 CSV 原样驱动。附件实际包含 1,000 个精度/功能用例和 200 个 `TC_PF_*` 用例，精度部分包含 4 个 `TC_3M_*` 专项。使用附件种子实现 `RANDOM_GAUSS_0_1`、`RANDOM_INT8`、`VALUE_INT8_EXTREME`、`RANDOM_DIAG_IMAG` 等填充规则。静态参考用 Netlib BLAS 3.10.0 CBLAS `cblas_cherk`，INT8 分量精确提升后再构造 golden。高精度定位可使用 `cblas_zherk`，正式精度结论按任务书要求的 CBLAS 参考口径记录。

| 检查项 | 判定 |
| --- | --- |
| 有限数精度 | 选定三角实部与虚部分开统计。`atol=rtol=2⁻¹³`，以 `abs(actual−golden) ≤ atol + rtol·abs(golden)` 计算匹配率，各分量组达到 `0.99`。逐分量核对 `1e-2` 绝对误差或 `32×ULP` 上限，并报告最大绝对误差。 |
| NaN/Inf | 单独核对 NaN 类别和 Inf 符号，参与结果匹配计数；有限值照常统计。 |
| 存储约束 | 计算或缩放后对角虚部为零，未选三角和 C padding 做位模式比较。quick return 保留完整 C 原值。 |
| 参数及运行时 | 覆盖空指针、非法枚举/类型/维度/跨度、INT8 交错布局、Host/Device 标量、同一 stream 上先写标量再调用算子。 |

参考 [Netlib CHERK](https://www.netlib.org/blas/cherk.f) 的三角与标量语义，结合任务书 §3.2 的 FP32 通过条件。日志保留用例名、参数、状态、实虚部分开的匹配率与误差、非有限结果，以及边界哨兵的检查结果。附件 CSV 的 `mere_threshold`、`mare_multiplier` 可用于框架诊断，验收结果记录上述分量指标。

### 4.2 性能与验收

附件的 200 条正式性能用例均为 `ACL_COMPLEX64`、紧凑 `lda/ldc`，`alpha=1, beta=0`，实际 `n/k` 的范围均为 1 至 4096。GPU 基线按附件顺序与 `TC_PF_*` 逐行核对 `(uplo,trans,n,k,Atype,Ctype)` 后绑定到用例编号。两组参数各出现两次，分别保留对应的 `gpu_ms` 记录，避免用形状作字典键覆盖已有记录。INT8 补充选择 N/UPPER、C/LOWER 两组典型规模，复用相同形状的 COMPLEX64 对照线并独立记录。

| 用例 | 参数 | GPU 基线（µs） | NPU 单例上限（µs） |
| --- | --- | ---: | ---: |
| `TC_PF_1001` | N / UPPER，`1024×1024` | 258.9367 | 647.3418 |
| `TC_PF_1002` | N / UPPER，`2048×2048` | 1439.9266 | 3599.8165 |
| `TC_PF_1003` | C / LOWER，`1024×1024` | 243.9159 | 609.7898 |
| `TC_PF_1004` | C / LOWER，`2048×2048` | 1270.0883 | 3175.2208 |
| `TC_PF_1007` | N / UPPER，`n=k=1` | 2.6368 | 6.5920 |

上表是附件 GPU 数据按单例标准换算得到的预算。上板先跑 `TC_PF_1007` 至 `TC_PF_1011` 核对小任务启动开销，再跑前四条典型用例核对 3M 吞吐与两种转置布局。使用 `msprof op` 按 kernel 名匹配 `Task Duration(us)`，预热 5 次，每例取至少 10 个成功且有效的设备样本的算术平均。按 `gpu_ms×1000 / npu_us` 计算倍率。正式 200 例逐例达到 `0.4`，整体倍率算术平均达到 `0.5`。`aclrtEvent` 输出的 `[PERF] avg_us` 用于趋势对照。内存性能验收按任务书 §3.4 为不涉及。

### 4.3 开发验证顺序

1. 落地 Host 校验、AIV 缩放和小任务 3M，先验证手算符号样例、INT8 极值、对角和退化分支。
2. 在 9.1.0 上验证 64×64 块的 FP32 `Mmad`、Fixpipe 行列映射及 AIV/AIC 同步。按尾块、非紧凑布局和长 K 的顺序扩展覆盖。
3. 跑完 1,000 条精度/功能用例与补充测试，按实际误差确定归约跨度。再跑 200 例正式性能及 INT8 补充样本，归档 XML、原始 CSV、环境和采集命令。
