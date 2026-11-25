# aclblasCherk3mEx（Atlas 950PR）算子设计

## 1. 需求背景

### 1.1 需求来源

承接“11月社区任务-aclblasCherk3mEx 算子开发（950）”，为 Ascend 950PR（arch35）补齐单精度复数 Hermitian 秩-k 更新的 3M 扩展接口。设计依据随任务包提供的 `aclblasCherk3mEx_Atlas950PR_task_doc.md` 与 `test_cases/`（`cherk3m_ex_test.csv`、`gpu_baseline.csv`、`gen_csv.py`、`verify_accuracy.py`、`verify_performance.py`），按[设计模板](../../../../resources/design_template.md)组织。

任务列表（`docs/README.md`）尚未给出本任务编号，故先在完整任务名目录下提交，后续以官方目录映射为准。

### 1.2 背景介绍

cuBLAS 的 `cublasCherk3mEx` 在 Hermitian 秩-k 更新上额外提供 3M（Gauss 三次实数乘）算法入口，并允许 A 以 complex-int8 存储而 C 为 complex64。本任务要求在 ops-blas 中以 Ascend C 直调 kernel 的方式实现同名语义接口。

源码现状核对基于 ops-blas 提交 `55ef791b77f3677b45da3468b0a8b17941424f84`：

| 用途 | 现有文件 | 结论 |
| --- | --- | --- |
| 公共声明与复数类型 | `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h` | 无 `aclblasCherk3mEx`，需新增声明；`aclblasComplex` 为实/虚各 float32 |
| 同族非 Ex 实现 | `blas/herk/arch35/cherk_host.cpp`、`cherk_kernel.cpp/.h`、`cherk_tiling_data.h` | 校验顺序、alpha/beta 指针定位、uplo 三角与对角虚部置 0 可直接沿用 |
| 3M 分解参考 | `blas/gemm/arch35/cgemm_deinterleave_common.h`、`cgemm3m_rounding_kernel.cpp` | 提供 3M 三平面拆分与重组模板 |
| 单 launch 融合参考 | `blas/gemm/arch35/cgemm_onchip_kernel.h`、`cgemm_compute_kernel.cpp`（`cgemm_fused::PublishProducts`）、`cgemm_ex_cube3_kernel.cpp`、`cgemm_ex_combine_impl.h` | 提供 AIC/AIV 混核（`KERNEL_TYPE_MIX_AIC_1_2`）逐 tile 三次乘积 + `CrossCoreSetFlag` 同步 + `Fixpipe` 直写 AIV UB 的重组范式 |
| 测试框架 | `test/herk/cherk/{arch35,arch22}`、`test/herk/cherk/CMakeLists.txt` | 需新建 `test/herk/cherk3m_ex/` 并注册目标 |
| 构建收集与门控 | `blas/CMakeLists.txt`、`cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS` | `cherk3m_ex_*` 源文件按文件名前缀自动收集，无需登记；仅当实现引用 tensor_api 或链接其他算子的 kernel 源时才需在对应列表登记（见 §3.5） |

关键差异：现有 `aclblasCherk` 走的是 **4M 非融合路径**——AIV deinterleave 后串行下发 4 次 `gemm_kernel_do`（`Ar·Ar^T`、`Ai·Ai^T`、`Ai·Ar^T`、`Ar·Ai^T`），再由 `cherk_combine_kernel_do` 在第二次 launch 中重组。该结构不能直接满足本任务“三次实数矩阵乘必须在同一次 kernel launch 内完成”的约束，且中间结果占 `4 × tempLdc × n` 个 float 的 GM 临时区。本设计因此另建融合 kernel，而非扩展现有 cherk host 的 launch 序列。

## 2. 需求分析

### 2.1 接口定义

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const float* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

语义：`C = alpha · op(A) · op(A)^H + beta · C`，列主序，C 仅 uplo 指定三角被读写，对角元素虚部计算后强制置 0。alpha、beta 为实数（FLOAT32）标量，支持 Host 与 Device 两种指针位置（混用拒绝）。

| 条件（按校验顺序） | 返回值 / 行为 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo ∉ {UPPER, LOWER}`、`trans ∉ {OP_N, OP_C}`（含 `OP_T`） | `ACLBLAS_STATUS_INVALID_ENUM` |
| `n < 0` 或 `k < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1, n)`（trans=N）/ `lda < max(1, k)`（trans=C），或 `ldc < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha`/`beta` 为 nullptr，或 n>0 且 k>0 时 `A == nullptr`，或 n>0 时 `C == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `(Atype, Ctype) ∉ {(ACL_COMPLEX64, ACL_COMPLEX64), (ACL_INT8, ACL_COMPLEX64)}` | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `n == 0` | `ACLBLAS_STATUS_SUCCESS`，不引用 A/C，不启动 kernel |
| `k == 0` 或 `alpha == 0`，且 `beta == 1` | 直接返回 `ACLBLAS_STATUS_SUCCESS`（C 不变，不读 A） |
| `k == 0` 或 `alpha == 0`，且 `beta != 1` | 只做 C 的 uplo 三角 `C = beta·C`，对角虚部置 0，不引用 A |
| 其余合法输入 | 在 handle 绑定的 stream 上异步下发；Host 读回前须同步该 stream |

校验顺序按任务书 §2.5：handle → uplo/trans → n/k → lda/ldc → alpha/beta 与 Atype/Ctype 及 A/C 指针；结构合法后再执行 quick return。`trans == ACLBLAS_OP_T` 在枚举一层就返回 `INVALID_ENUM`，不进入形状校验。

### 2.2 需求拆解

1. 在 `include/cann_ops_blas.h` 新增 `aclblasCherk3mEx` 声明，参数序列与 `cublasCherk3mEx` 逐项对应。
2. 在 `blas/herk/arch35/` 新增 3M 融合 kernel 路径，三次实数乘在同一次 launch 内完成。
3. 支持 `Atype ∈ {ACL_COMPLEX64, ACL_INT8}`、`Ctype = ACL_COMPLEX64`；complex-int8 按 `(int8 real, int8 imag)` 交错 2 字节解释。
4. 支持 uplo × trans 四种组合、padding 前导维、`k=0`/`alpha=0` 的 beta-only 路径。
5. 新增 `test/herk/cherk3m_ex/arch35/` 测试工程，CSV 驱动 1000 条精度 + 200 条性能用例。
6. 精度不低于 `cblas_cherk`（FLOAT32 分量阈值），性能达 GPU `cublasCherk3mEx` kernel 的 0.4×/0.5×。

## 3. 详细设计

### 3.1 数学定义

令 `X = op(A) = Xr + i·Xi`（`Atype=ACL_INT8` 时先做精确 int8→float 提升，无舍入）。本算子的乘积是 `X·X^H` 而**不是** `X·X^T`，因此 3M 必须按 `B = X^H` 分解：

```text
X·X^H = (Xr + i·Xi)(Xr^T − i·Xi^T)
      = (Xr·Xr^T + Xi·Xi^T) + i·(Xi·Xr^T − Xr·Xi^T)
```

取非融合正号形式：

```text
P0 = Xr·Xr^T          P1 = Xi·Xi^T          P2 = (Xr+Xi)·(Xr−Xi)^T
Re(C_new) = alpha·(P0 + P1) + beta·Re(C)
Im(C_new) = alpha·(P2 − P0 + P1) + beta·Im(C)
```

代入 `P2 = P0 − Xr·Xi^T + Xi·Xr^T − P1` 可验证 `P2 − P0 + P1 = Xi·Xr^T − Xr·Xi^T`。**若误用 `X·X^T` 的对称分解，虚部会算成 `Xi·Xr^T + Xr·Xi^T`（符号错误）**，这是本算子最需要防的实现缺陷，测试侧由 `TC_3M_*` 虚部抵消用例专门覆盖。

三个乘积的两个操作数都来自 A 的实/虚平面，第二操作数是 `Xr^T`、`Xi^T`、`(Xr−Xi)^T`（均为正号形式，不需要额外取负）。`A` 的物理布局随 trans 改变（n×k 或 k×n），因此左右操作数的取法如下：

| trans | Xr | Xi | P0 = ··^T | P1 = ··^T | P2 = ··^T |
| --- | --- | --- | --- | --- | --- |
| `OP_N` | `Ar`（n×k） | `Ai`（n×k） | `Ar·Ar^T` | `Ai·Ai^T` | `(Ar+Ai)·(Ar−Ai)^T` |
| `OP_C` | `Ar^T`（n×k） | `−Ai^T`（n×k） | `Ar^T·Ar` | `Ai^T·Ai` | `(Ar−Ai)^T·(Ar+Ai)` |

`OP_C` 行的 `Xr`/`Xi` 由 `A^H = Ar^T − i·Ai^T` 直接得到：不需要任何显式转置，只需在 deinterleave 时交换实/虚平面的读写方向并把 `Xi` 取负；`P1 = Xi·Xi^T = (−Ai^T)(−Ai) = Ai^T·Ai`，符号自然抵消，因此 `P1` 仍按正号累加。

仓内 `gemm_kernel_do(a,b,c)` 的 fixpipe 实际写出 `(a·b)^T`（见 `cherk_host.cpp` 中 `t3/t4` 操作数互换的注释）。`OP_N` 路径据此交换左右操作数即可得到 `Ar·Ar^T`，不需额外的搬运转置步骤；`OP_C` 路径本身已是 `A` 的存储序，左乘右乘顺序反过来写。

### 3.2 方案选择

| 方案 | 三次乘积的落点 | 是否满足任务约束 | 结论 |
| --- | --- | --- | --- |
| A. 沿用 cherk 的 4M 多 launch | 4 次 `gemm_kernel_do` + 1 次 combine kernel | 否（跨 launch） | 排除 |
| B. 单 launch，3 次独立 GEMM kernel 序列 | 同一 launch 内 3 次 `cgemm_ex_cube3_do` 式调用 | 形式满足但需 3× `tempLdc×n` GM 临时区 | 作为回退 |
| C. 单 launch，AIC/AIV 混核逐 tile 融合（本设计） | 每 tile 内 3 个 Mmad → L0C → 直接重组写 C | 是 | **采用** |

方案 C 复用 `cgemm_onchip_kernel.h` 已验证的混核范式：AIC 负责 `Mmad` 与 `Fixpipe`，AIV 负责 deinterleave、重组与写回，两者用 `CrossCoreSetFlag/CrossCoreWaitFlag` 按 tile 阶段握手（`INPUT_READY/INPUT_FREE` 双缓冲标志），kernel 类型 `KERNEL_TYPE_MIX_AIC_1_2`。三个乘积由 `Fixpipe<..., ROW_MAJOR_UB>` 直接写到 AIV 的 UB（同 `cgemm_fused::PublishProducts`），不经过 GM；相比方案 B，GM 临时区从 `O(tempLdc·n)` 降为 0，避免了 Cherk 当前路径在 n=4096 时的数百 MB 中间缓冲。

### 3.3 Host 侧设计

**流程**：校验 → 标量解析 → 平台信息 → tiling → 下发 1 个融合 kernel（或 beta-only kernel）。两条路径的中间数据全部在片上，不使用 GM workspace。

1. **标量解析**：`CheckPtrLocation` 分别判定 alpha/beta 位于 Host 还是 Device（对齐 `cherk_host.cpp` 中 issue #628 的修复方式）。两者须同为 Host 或同为 Device，混用返回 `ACLBLAS_STATUS_INVALID_VALUE`；Host 指针直接取值，Device 指针经 `ReadCherkScalarsFromDevice` 在绑定 stream 上做 D2H 读取后使用。取值结果写入按值 tiling，其后的 quick-return 判定与 kernel 都不再访问标量指针。
2. **融合分块**：

   - `BM = BN = 64`，为 Cube fractal 16 的整数倍；每核 L0C 需要 3 个 `64×64×4B = 16KiB` 累加器（P0/P1/P2），共 48KiB，在 256KiB L0C 内留出双缓冲余量。
   - 输出暂存在片上：每核 AIV 的 UB 为三个乘积预留 2 个流水级 × 3 个 `64×64×4B = 16KiB` 平面，共 96KiB，由 `Fixpipe` 直写（见 §3.4）；950PR 单核 UB 为 248KiB，剩余部分容纳 A 面板与 C 回读缓冲。
   - `k` 方向按 `tileKChunk = 64` 切片累加。每个 k-tile 在 L1 中存放左右操作数各 3 个平面（`Xr`、`Xi`、`Xr+Xi` 与 `Xr^T`、`Xi^T`、`(Xr−Xi)^T`），每平面 `64×64×4B = 16KiB`，每级 6 × 16KiB = 96KiB；双缓冲共 192KiB，在 512KiB 的 L1 容量内，级数与 `cgemm_onchip_kernel.h` 的 `CGEMM_L1_PING_PONG_STAGES = 2` 一致。
   - 分核采用 2D 切分：先按 `BM`/`BN` 得到 `mTiles × nTiles`，再在 `mBlocks × nBlocks ≤ AIC 核数` 约束下选利用率最高的组合（沿用 `CalcMultiCorePartition` 的枚举策略），蛇形遍历 tile 索引以提升 A/B panel 的 L2 复用。只调度 uplo 三角（含对角块）以内的 tile，三角之外的 tile 不计算、不写回。AIC 核数运行时动态获取，不硬编码（950PR 各款型不同：9579 为 28，9589 为 32，9599 为 36）。
3. **tilingkey 规划**（host 侧枚举，kernel 侧模板分发）：

   | 维度 | 取值 | 用途 |
   | --- | --- | --- |
   | uplo | {UPPER, LOWER} | 决定写回的三角与对角掩码 |
   | trans | {N, C} | 决定三个乘积的操作数取法与转置标志 |
   | Atype | {COMPLEX64, INT8} | 决定 deinterleave 的读类型与提升方式 |
   | 形状档位 | {小形状向量/SIMT, 常规混核 Cube} | 由 tile 数与总工作量 `n²·k` 共同判定：小形状走向量/SIMT 以避免混核启动开销；判据须覆盖 n 大 k 小的形状（如 n=1031、k=8，单 case 上限约 14µs），阈值见 §5 |

   组合数 2×2×2×2 = 16，由 host 写出 32 位 tilingkey；同一输入两次执行的 tiling 与分派完全由形状决定，不含随机因素。

4. **beta-only 路径**（`k==0` 或 `alpha==0`，`beta!=1`）：下发一个轻量 AIV kernel，按行分段对 uplo 三角做 `C = beta·C`，并把对角元素虚部清零；`beta==0` 时不读旧 C。

### 3.4 Kernel 侧设计

```mermaid
flowchart LR
    subgraph AIV["AIV (每 tile)"]
      D[A 面板搬入 UB<br/>int8/float 分流] --> T[拆分为 Xr、Xi、Xr±Xi<br/>三平面写入 L1]
      T --> F
      R[读 UB 中的三乘积平面] --> M[重组 Re/Im<br/>alpha/beta 融合]
      M --> W[三角掩码 + 对角虚部置 0<br/>写回 C]
    end
    subgraph AIC["AIC (每 tile)"]
      F[P0 = Xr·Br] --> G[P1 = Xi·Bi]
      G --> H[P2 = (Xr+Xi)·(Xr−Xi)]
      H --> X[Fixpipe 直写 AIV UB 三平面]
    end
    T -.CrossCoreSetFlag INPUT_READY.-> F
    X -.CrossCoreSetFlag READY.-> R
```

1. **Init**：`TPipe` 初始化 UB 队列（输入面板、重组输出、C 回读）、L1 双缓冲（每级左右各三平面）、L0A/L0B 双缓冲、L0C 三个累加器；AIC 侧预置 `FIX_M`/`M_MTE1` 事件，AIV 侧预置 `MTE3_MTE2` 与流水级 free 标志，`DrainPipeline` 在收尾时等待所有反向标志，避免退出时残留事件。
2. **CopyIn（AIV）**：按 (k-tile, panel) 从 GM 读 A 面板。`ACL_COMPLEX64` 走 `DataCopyPad` 直接搬交错实/虚 float；`ACL_INT8` 走 int8 搬运后 `Cast` 到 float（int8→float 精确）。随后逐元素产出三平面：`Xr`、`Xi`、`Xr±Xi`；B 侧平面取 `Br`/`Bi`，按 §3.1 表格决定是否取负、是否交换左右操作数。整块 16 对齐，尾部 tile 用 `DataCopyPad` 按实际行数补齐并清零，避免越界读。
3. **Compute（AIC）**：每个 k-tile 依次发 3 个 `Mmad`，写入 3 个独立 L0C 累加器；首个 k-tile 用 `UNIT_FINAL` 初始化累加器，后续 k-tile 累加。三个 Mmad 之间只做 `M_MTE1` 事件保护，L0A/L0B 双缓冲让下一个 k-tile 的搬运与前一个的 Mmad 重叠。输入与累加均为 FP32，kernel 入口调用 `SetHF32Mode(HF32Mode::DISABLE)` 关闭 HF32，与 `cgemm_compute_kernel.cpp` 的 FP32 Cube 核一致。
4. **CopyOut（AIC→AIV）**：k 方向累加完成后，`Fixpipe<float, float, ROW_MAJOR_UB>` 把三个 L0C 累加器直接写到 AIV 的 UB 输出平面（每核 2 个流水级轮换），随后 `CrossCoreSetFlag<PIPE_FIX>(READY_FLAG + stage)`；AIV 在 `PIPE_MTE3` 上 `CrossCoreWaitFlag` 后直接在该 UB 平面上重组，用完再以 `CrossCoreSetFlag<PIPE_MTE3>` 归还 `FREE_FLAG`，AIC `CrossCoreWaitFlag<PIPE_FIX>` 到该标志后才复用这一级。三个乘积全程不经过 GM。
5. **重组与写回（AIV）**：向量化计算 `Re = P0 + P1`、`Im = P2 − P0 + P1`，再按 `alpha`/`beta` 融合（`beta==0` 时完全不读 C）；按 uplo 生成三角掩码，只写指定三角，`i == j` 处的虚部强制置 0；非 uplo 三角与 `ldc` padding 列保持原值不写。写回用 `DataCopyPad` 按实际列宽搬出，避免破坏 padding。
6. **确定性**：k 切分与 tile 归属只由 `n/k/uplo/trans/核数` 决定，不按用例编号或随机种子分派；同一输入两次执行得到相同的累加顺序。

### 3.5 工程集成

| 内容 | 文件 | 说明 |
| --- | --- | --- |
| 公开声明 | `include/cann_ops_blas.h` | 新增 `aclblasCherk3mEx` 原型（紧邻 `aclblasCherk`） |
| Host 实现 | `blas/herk/arch35/cherk3m_ex_host.cpp` | 校验、tiling、launch |
| Kernel 实现 | `blas/herk/arch35/cherk3m_ex_kernel.cpp`、`.h`、`cherk3m_ex_tiling_data.h` | 混核融合 kernel 与按值 tiling 结构 |
| 构建门控 | `cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS` 增加 `"CHERK3M_EX\|/herk/arch35/cherk3m_ex\|cherk3m_ex"` | 仅当实现引用 tensor_api 时需要：asc-devkit < 9.1 时整体跳过该算子并给出 skip 原因；源文件本身按文件名前缀自动收集 |
| 算子文档 | `blas/herk/README.md`、`docs/zh/api_list.md` | 补 `aclblasCherk3mEx` 章节与索引行 |
| 测试 | `test/herk/cherk3m_ex/CMakeLists.txt`、`cherk3m_ex_param.h`、`cherk3m_ex_golden.h`、`arch35/cherk3m_ex_npu_wrapper.h`、`arch35/cherk3m_ex_test.cpp`、`arch35/cherk3m_ex_test.csv`、`README.md` | `ops_blas_add_gtest_tests(${OPS_BLAS} cherk3m_ex_test)`；精度与性能用例同在一份 CSV，以 `TC_PF` 前缀区分 |

编译与自验入口：`bash build.sh --ops=cherk3m_ex --soc=ascend950`。

不依赖 `aclblasCherkEx`（与本算子同为 11 月新增的独立接口，可能尚未合入），也不复用 arch22 的 `cherk` 契约。

### 3.6 约束与边界

- **不支持广播**：A 与 C 为独立操作数，无 shape 广播语义。
- **不涉及动态 shape**：n、k 为运行时入参，host 按实际值 tiling。
- **不涉及非连续 Tensor**：访问跨度由 lda/ldc 的列主序带填充布局描述，不使用 `ViewCopy`。
- **原地输出**：C 旧值参与 beta 项后写回同一块内存；仅 uplo 三角被读写，调用方保证 C 可写区与 A 可读区不重叠。
- **Hermitian 输入假设**：C 的 Hermitian 性质由调用方保证；本算子不做元素级扫描校验，按公式自然计算，只在写回时强制对角虚部为 0。
- **数值口径**：3M 的 FP32 中间舍入与直接复数乘不同，结果与 `cblas_cherk` 不保证逐位一致，按 §4.1 阈值判定。

### 3.7 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

## 4. 可维可测分析

### 4.1 精度标准

golden 为 CPU 侧 Netlib CBLAS 3.10.0 的 `cblas_cherk`（`CblasColMajor`）；`Atype=ACL_INT8` 时先把 int8 实/虚部分别精确提升为 float 再调用，float64 参考用 `cblas_zherk` 收窄。实部、虚部分别比对，仅比对 uplo 三角（含对角）。

| 项 | 取值 |
| --- | --- |
| rtol / atol（FLOAT32 分量） | 2⁻¹³ ≈ 1.2207e-4 |
| required_matched_ratio | 0.99 |
| max_abs_error_limit | 1e-2（或 32×ULP） |
| 判定 | `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2` |
| 工程口径 | `mere_threshold = 2⁻¹³`、`mare_multiplier = 10.0` |

专项校验：对角 `|Im| ≤ atol`；未引用三角与 `ldc` padding 用哨兵值写保护，调用后逐字节比对未被改写。

3M 的虚部 `Im = P2 − P0 + P1` 在强抵消时（结果远小于三个乘积本身）舍入误差大于直接复数乘，除阈值判据外由 `TC_3M_*` 用例专项覆盖；是否需要限长累加等误差控制手段，按真机精度数据在 §5 中跟踪。

### 4.2 性能标准

测试设备昇腾 950PR。正式判据为 kernel-to-kernel：`gpu_baseline.csv` 的 `gpu_ms` 来自 GPU `cublasCherk3mEx` 的 nsys kernel-only 耗时；NPU 侧用 `msprof op` 采集单算子 `Task Duration(us)`（warmup 5 + 有效采样 10 取平均）。

| 判据 | 阈值 |
| --- | --- |
| 单 case | NPU kernel 平均耗时 ≤ `gpu_ms` / 0.4（即 ≥ 0.4× GPU） |
| 全部 case 平均倍率 | ≥ 0.5 |
| int8 路径 | 由开发者补充采集，沿用同参 `ACL_COMPLEX64` 的 `gpu_ms` 基线 |

`[PERF] avg_us`（`aclrtEvent` 夹 10 次下发）仅作快速回归，由 `verify_performance.py` 解析，不作正式判据；未解析到该行的用例判 FAIL。

### 4.3 用例设计

`gen_csv.py`（默认种子 20261112）生成 1200 条：精度 1000 条（`TC_L0`/`TC_3M` 等前缀）+ 性能 200 条（`TC_PF` 前缀）。

- 覆盖 uplo × trans 全组合 × Atype（COMPLEX64/INT8）× 尺寸扫描（含 0、2 的幂与 ±1、非对齐 33/65/127/1025）× k 非对称（k≪n、k≫n）× 前导维 padding（+1/+8/+64）× 填充分布（`RANDOM_NORM_5_5`、`RANDOM_GAUSS_0_1`、`RANDOM_INT8`、`VALUE_INT8_EXTREME`、`RANDOM_DIAG_IMAG`）。
- 负向：非法 uplo/trans（含 `OP_T`）、负 n/k、lda/ldc 低于下界、`Atype/Ctype = ACL_FLOAT`、`handle == nullptr`（在参数装配层固定传 nullptr，CSV 无法表达）。
- 特殊场景：`alpha=0` 且 A=NaN（验证不引用 A）、`beta=0` 且 C=NaN（验证不读旧 C）、`k=0` 且 `beta≠1`、对角虚部非零输入。
- 性能前 4 条为 N/1024、N/2048、C/1024、C/2048 典型 case，其余在 512MB 预算（`8×(n·k+n²) ≤ 512MB`）内做小/中/大档分布。

### 4.4 兼容性分析

新增接口，不改变既有符号与行为。`aclblasCherk`（4M 路径）与 arch22 `cherk` 保持原状，两套实现共存且互不调用；`include/cann_ops_blas.h` 仅追加原型，不用宏改名。`trans=OP_T` 返回 `INVALID_ENUM` 是按 Hermitian 定义域收窄的有意差异（cuBLAS 文档将其列入合法枚举但公式未定义），需在自测报告说明。

## 5. 风险与未验证项

| 项 | 说明 |
| --- | --- |
| 融合 kernel 的 L0C 占用 | 3 个 `64×64` 累加器共 48KiB，理论上可在 256KiB L0C 内再容纳 2 级流水；实际双缓冲级数须在 950PR 真机上按 `Mmad` 与 `Fixpipe` 时序实测后再定，文档给出的是设计目标而非实测值 |
| 小形状档位阈值 | 向量/SIMT 路径与混核 Cube 路径的切换点由真机测试决定；判据须同时考虑 tile 数与 `n²·k`，因为基线中存在 n 大 k 小、上限很紧的形状（n=1031、k=8 时单 case 上限约 14µs），当前仅给出候选区间 |
| 3M 虚部舍入 | `Im = P2 − P0 + P1` 的抵消误差量级待真机确认；若超出 §4.1 阈值，参照 `cgemm_onchip_kernel.h` 的 `reductionSpan` 做 K 方向限长累加，或对高风险元素用直接复数乘重算 |
| int8 性能 | 任务书只要求开发者补充采集，融合路径下 int8 仅影响 AIV 侧 deinterleave，Cube 侧与 COMPLEX64 完全一致，故不预期性能回退；此结论待实测 |
| 性能数据 | 设计阶段尚无任何真机编译与 `msprof op` 数据；文中不含任何实测性能或精度数字，相关结论均须真机验证 |
| 目录映射 | 任务列表未给出本任务编号，目录名暂用完整任务名，合入前以官方映射为准 |
