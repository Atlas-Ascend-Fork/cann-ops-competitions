# 需求背景

## 需求来源

[11月社区任务-aclblasChemv算子开发（A2/A3)](https://www.hiascend.com/activities/task-center/details/cbfd47808791498d94595dd9e48500aa) 要求在 Atlas A2/A3 系列产品上实现单精度复数 Hermitian 矩阵-向量乘，接口与验收范围以该任务书为依据。目标代码仓库为 [ops-blas](https://gitcode.com/cann/ops-blas)，实现目录 `blas/symv/arch22/`，测试目录 `test/symv/chemv/arch22/`。本任务为补齐 BLAS 仓能力，仓内当前无 `aclblasChemv` 声明与实现。

## 背景介绍

`aclblasChemv` 计算 `y = alpha*A*x + beta*y`，其中 `alpha`、`beta`、`A`、`x`、`y` 均为 `COMPLEX64`。`A` 为 `n×n` Hermitian 矩阵，满足 `A = A^H`，按列主序全存储于 `lda×n` 数组，仅 `uplo` 指定的三角被引用，未引用部分由共轭对称性隐含、不得读取，对角元素的虚部假定为 0 且不参与计算。`x`、`y` 为 `n` 元素向量，支持正负步长。语义对标 cuBLAS `cublasChemv`，参考实现为 Netlib `chemv`。

仓内同族已有 `aclblasSsymv`（`blas/symv/arch22/`，实数对称）与 `aclblasCsymv`（`blas/symv/arch35/`，复数对称）。Hermitian 与 symmetric 的差别不止于数据类型：对称矩阵满足 `A[j][i] = A[i][j]`，Hermitian 满足 `A[j][i] = conj(A[i][j])`，因此由存储三角推导未存储三角时必须取共轭，且对角元素必须按实数处理。直接复用 symv 的数值路径会在全部非对角元素上产生系统性偏差。

# 需求分析

## 算子功能

设 `A` 的存储三角由 `uplo` 指定。对 `0 ≤ j < n`：

- `y[j] = alpha * ( Σ_{i≠j} A_H[j][i] * x[i] + Re(A[j][j]) * x[j] ) + beta * y[j]`，其中 `A_H[j][i]` 在 `(j,i)` 落于存储三角时等于 `A[j][i]`，否则等于 `conj(A[i][j])`。

边界语义按 Netlib `chemv`：`n = 0` 为合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS` 且不访问任何指针；`alpha = (0,0)` 且 `beta = (1,0)` 时亦直接返回成功，此时 `A`、`x`、`y` 均允许为空；`alpha = (0,0)` 且 `beta ≠ (1,0)` 退化为 `y = beta*y`，不读取 `A` 与 `x`；`beta = (0,0)` 时 `y` 的旧值不必有效，但 `y` 指针本身不得为空。`y` 原地覆写，不返回视图。

## 公开接口与范围

在 `include/cann_ops_blas.h` 新增任务书规定的 10 参数接口，维数参数采用 `int`，与同族 `aclblasSsymv`、`aclblasCsymv` 的既有声明口径一致，供其他产品线共用，不定义私有平行 API：

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

`aclblasComplex` 以 `include/cann_ops_blas_common.h` 的定义为准（实部、虚部各一个 `float`）。支持 `COMPLEX64`、列主序、`lda` 填充与 `incx/incy` 的正负步长。`n` 在运行时确定，不要求 dynamic shape。`alpha`、`beta` 支持 Host 与 Device 两种内存，由运行时按指针属性探测，口径对齐仓内 `csymv` arch35 的 `ComplexScalarSource`（issue #869）。不支持超出 `lda/incx/incy` 语义的非连续访问，不涉及广播，不要求结果逐位确定。

# 详细设计

## 算子分析

以 `uplo = ACLBLAS_UPPER` 为例。列主序下矩阵元素 `(i, j)` 位于 `j*lda + i`，上三角意味着对第 `j` 列持有 `i ∈ [0, j]` 的元素，且这段元素在内存中连续。遍历列 `j` 一次读取该连续段，可同时产出两份互不重复的贡献：

- **内积贡献**：`y[j] += alpha * Σ_{i<j} conj(A[i][j]) * x[i]`。该式来自 `A[j][i] = conj(A[i][j])`，是 Hermitian 相对 symv 唯一需要共轭的一侧。
- **缩放贡献**：`y[i] += alpha * A[i][j] * x[j]`，`i < j`。该元素本身即存储元素，不取共轭。
- **对角贡献**：`y[j] += alpha * Re(A[j][j]) * x[j]`，仅计一次，虚部按 0 处理。

`uplo = ACLBLAS_LOWER` 时第 `j` 列持有 `i ∈ [j, n)`，两份贡献的角色对称交换，对角元素同样单独处理。两种 `uplo` 的实现共用同一套 chunk 逻辑，由存储三角的方向参数区分读取区间，并由一个判定函数确认当前 chunk 是否跨越对角线，跨越时将对角元素从缩放贡献中剔除，避免重复累加。

复数向量运算采用仓内 arch22 既有做法：把 `COMPLEX64` 的 Global 指针按 `float` 解释，交织的实部、虚部经 `GatherMask` 分离到两组 UB 缓冲，后续全部以实数向量指令组合复数乘加。`GatherMask` 以 repeat 为粒度（每 repeat 固定 256 字节、32 个 `float`），因此分离前须保证 chunk 长度与缓冲区按该粒度对齐，尾部按有效长度单独处理。结果写回 `y` 需要由分离布局恢复交织布局，该方向为散布操作，不使用 `GatherMask`，改用 `Gather` 还原偏移后写出。共轭只需对分离出的虚部取负，与仓内 `blas/dot/arch22` 的 `cdot` 共轭分支一致，无需额外数据搬运。

`incx/incy ≠ 1` 时元素在 Global 内不连续。物理位置换算统一在 64 位域进行：正步长取 `logical * |inc|`，负步长取 `(n-1-logical) * |inc|`；`(n-1) * |inc|` 在 32 位域会回绕，Host 侧守卫先行拒绝超出可表示范围的调用（对齐 issue #807）。

## 算子实现

### Host 侧设计

Host 严格按任务书规定的顺序校验，该顺序与仓内同族既有实现不同，不沿用：`handle` 判空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n = 0` 立即返回 `ACLBLAS_STATUS_SUCCESS`，此时不再检查任何其他参数、不访问任何指针，`uplo`、`lda`、`incx`、`incy`、`alpha`、`beta` 即使全部非法也返回成功；随后校验 `uplo` 取值，非法返回 `ACLBLAS_STATUS_INVALID_ENUM`；校验 `lda ≥ max(1, n)`、`incx ≠ 0`、`incy ≠ 0`、`alpha` 与 `beta` 非空，不满足返回 `ACLBLAS_STATUS_INVALID_VALUE`。`incx` 或 `incy` 等于 `INT32_MIN` 无条件拒绝（对齐 issue #780），该判定独立于跨度守卫，因为 `n = 1` 时跨度为零不会触发守卫。跨度守卫在 Host 侧以 64 位计算 `(n-1) * |incx|` 与 `(n-1) * |incy|`，超出 32 位索引域返回 `ACLBLAS_STATUS_INVALID_VALUE`。

通过上述检查后读取标量。`alpha`、`beta` 使用 `common/helper/host_utils.h` 的 `ReadScalarFromHostOrDevice` 的 `aclblasComplex` 重载，该重载按指针属性区分 Host 与 Device 来源，仓内 `cgemv` arch22 已采用同一路径。读到标量后判定：`alpha = (0,0)` 且 `beta = (1,0)` 直接返回成功；`alpha ≠ (0,0)` 时 `A`、`x` 不得为空；除上一条快速返回外 `y` 不得为空；违反返回 `ACLBLAS_STATUS_INVALID_VALUE`。

Host 不分配 Device 工作区、不在算子内同步 stream。tiling 结构按值传入 kernel，包含 `n`、`lda`、`uplo`、`useCoreNum`、`alpha` 与 `beta` 的实虚部、`incx`、`incy` 与执行阶段标识。核心数经平台接口查询后按向量核上限截断，并以 `n` 为上界；`incx` 或 `incy` 不为 1 时退回单核，避免跨步写回的跨核交叠。计算在 `handle` 绑定的 stream 上下发，调用方读取 Device 结果前自行同步该 stream。

### Kernel 侧设计

核函数为纯 AIV（`KERNEL_TYPE_AIV_ONLY`），不使用 cube。每个向量核按列号取模认领列，对认领的每一列按 UB 容量切 chunk 顺序处理。单 chunk 的处理分三步：搬入该列 chunk 的 `A` 连续段与对应的 `x` 段（`incx = 1` 走连续 DMA，否则逐元素取值）、分离实虚部后完成复数乘加、写回两份贡献。

内积贡献在 chunk 内先做复数逐元素乘（虚部取负实现共轭），再对实部、虚部分别归约，得到该 chunk 对 `y[j]` 的标量贡献。缩放贡献为复数标量乘向量，标量为 `alpha * x[j]`，结果长度等于 chunk 长度（跨越对角线的 chunk 减一）。对角元素单独成一个长度为 1 的 chunk 处理，只取实部参与计算，使该元素不进入任何缩放贡献路径。

`beta` 的施加与多核累加的关系是本算子唯一的结构性取舍，两条路线如下：

- **路线 A（两段式）**：先由单核覆盖写 `y = beta*y`，再在原子累加模式下由各核累加 `alpha*A*x` 的各份贡献。该结构与仓内 `ssymv` arch22 一致，其源码记录了放弃单次启动的原因：同一次启动内「一个核读改写 `y`」与「其余核原子累加 `y`」存在跨核竞态，实测在 `n = 255`、UPPER 下产生与 `(beta-1)` 成比例的偏差。两段分两次 kernel 启动，由同一 stream 的顺序保证互斥。
- **路线 B（单段式）**：按 `y` 的下标区间分核，每个核独占一段 `y`，在核内完成 `y_seg = beta*y_seg + alpha*(A*x)_seg`。各段输出互不相交，既不需要原子累加，也不需要第二次启动，因而不存在上述竞态。代价是每个核为求自己这段 `y` 需要同时读取 `A` 的列内三角段与行内三角段，其中一侧在列主序下为跨步访问。同族的 `csymv` arch35 的 tiling 结构中没有阶段标识字段，表明复数 symv 在该架构上即以单段式实现。

两条路线的正确性均可由上述分析保证，差异集中在性能：路线 A 多付一次 kernel 启动开销，在小 `n` 场景占比显著；路线 B 省去该开销但访存效率在大 `n` 场景可能下降。本设计两条均实现至可运行，在 Atlas 800T A2 (910B3) 上以同一构建连续测量三次，按 `TC_PF` 用例的逐例比值在小尺寸档与大尺寸档分别比较后择一固定，不以推理替代实测。择定结果与两组原始数据写入自验证报告。

UB 预算按 192 KB 规划。`A` chunk 与 `x` chunk 各需实部、虚部两份缓冲，缩放贡献的输出需一份与 chunk 等长的复数缓冲，另需归约临时区与 `GatherMask` 的偏移表。`COMPLEX64` 每元素 8 字节，分离后每分量 4 字节，单 chunk 的复数元素数按上述总量反推并向下取到 `GatherMask` 的 repeat 粒度整数倍。

### 资源与调度

列的认领以核号为起点、以启动核数为步长轮转，使三角形状带来的列长差异在核间自然摊平：上三角的长列与短列交替落到不同核上，不额外做负载均衡计算。`n` 小于启动核数时按 `n` 截断核数，避免空核参与启动。

`incx` 或 `incy` 不为 1 时退回单核执行。跨步场景下 `y` 的写回位置由 `(n-1-logical) * |inc|` 等式决定，多核分段无法保证写回区间不交叠，单核执行以正确性优先；该场景在性能用例集中不出现（`TC_PF` 全部为 `incx = incy = 1` 的连续访存）。

两段式路线的第一段固定单核执行，第二段使用全部认领核数；单段式路线只有一次启动。两条路线的 tiling 结构共用同一定义，由阶段标识字段区分，便于 A/B 实测时切换同一构建产物。

## 支持硬件

目标为 Atlas A2/A3 系列产品，使用 `blas/symv/arch22/` 目录。精度自验证在 910B3 或 A3 款型上完成，性能验收设备按任务书为 Atlas 800T A2 (910B3)。`blas/symv/README.md` 的产品支持表标注 Atlas A2/A3 系列产品：支持。CANN 版本为 9.1.0。

## 算子约束限制

只支持 `COMPLEX64`、列主序、`lda ≥ max(1, n)`、`incx ≠ 0` 且 `incy ≠ 0`。`A` 仅 `uplo` 指定三角被引用，未引用三角不读取；对角元素虚部按 0 处理，调用方写入的对角虚部不影响结果。`y` 原地覆写。`alpha` 与 `beta` 支持 Host/Device 指针；`A`、`x`、`y` 为 Device 内存。不支持非连续 Tensor、不涉及广播、不要求 dynamic shape、不要求确定性计算。浮点累加顺序不保证与参考实现逐位一致。

# 可维可测分析

## 精度标准与验证

以 Netlib CBLAS `cblas_chemv`（`CblasColMajor`）生成 Golden，比较输出向量 `y` 的全部 `n` 个逻辑元素（含步长位置），实部与虚部分别统计。Golden 按 Hermitian 约束计算：仅读取 `uplo` 指定三角，对角元素只取实部，未引用三角按共轭对称补全；用例生成时未引用三角的填充值仅为占位，Golden 与算子均不依赖该值。

`COMPLEX64` 的实部、虚部分别按 FLOAT32 档的混合容差标准判定：`rtol = atol = 2^-13 ≈ 1.2207e-4`，逐元素通过条件为 `|actual - golden| ≤ atol + rtol * |golden|`，用例通过条件为 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2 或 32×ULP`。Inf/NaN 按任务书规则单独判定：Golden 与实际同为 NaN、同为同号 Inf，或一方为 NaN 而另一方为 ±Inf，视为等效通过；一方有限而另一方非有限，判定不通过。非有限元素不计入 `matched_ratio` 与 `max_abs_error` 统计，单独记录其总数与其中通过数。

测试工程另按仓内惯例以 CSV 列配置 MERE/MARE 策略（`mere_threshold = 2^-13`、`mare_multiplier = 10.0`），用于 GTest 的逐条 PASS/FAIL。两套口径并存：标准比对按 dtype 固定阈值执行，不随用例列变化；报告分别给出两套结果。

用例以随任务提供的 `chemv_test.csv` 为基础，共 1200 条，其中精度 1000 条、性能 200 条。精度用例覆盖 `uplo` 全组合的小尺寸基础场景、`n ∈ [0, 2048]` 的尺寸扫描（含 0、1、小质数、2 的幂及其 ±1、33/65/127/1025 等非对齐值）、八组 `alpha`/`beta` 标量组合、`lda` 紧凑与 `n+1`/`n+8`/`n+16` 填充、`incx × incy` 在 ±1/±2/±3 的全组合、均匀与全零与交替与极端值与 Inf 与 NaN 的填充模式，以及边界与负向用例。边界用例包含 `n = 0` 叠加全部非法参数仍期望成功、`alpha=(0,0)` 且 `beta=(1,0)` 配空指针期望成功、`beta=(0,0)` 配 `y` 为 NaN 期望成功、`beta=(0,0)` 配 `y` 为空期望 `ACLBLAS_STATUS_INVALID_VALUE`、非法枚举期望 `ACLBLAS_STATUS_INVALID_ENUM`、非法 `lda`、负 `n`、`incx/incy` 为 0 与 `INT32_MIN`。期望非成功的用例以单元素占位缓冲或空指针提供输入，不按 `(n-1)×|inc|` 预分配，避免测试工程自身分配失败导致误判。

任务书 §3.5 要求的正态分布用例（`μ ∈ [-5,5]`、`σ ∈ [0.1,2]`，填充 token `GAUSS_NORM_5_2`）不在随包生成器的默认输出中，由本任务扩展 `gen_csv.py` 补齐并纳入精度自验证，报告中说明补齐方式、用例数量与覆盖范围。

精度日志与 JUnit XML 保留 case 名称、实虚部各自的匹配比例、最大绝对误差及其位置对应的 Golden 值、非有限元素统计与失败位置；参数与随机种子保留在原始 CSV。

## 性能标准与验证

在 Atlas 800T A2 (910B3) 上测量 `TC_PF` 的 200 条用例，判据为平均单次耗时不高于 `test_cases/gpu_baseline.csv` 中 A100 标杆耗时的 1.25 倍，即达到标杆 0.8 倍以上性能，逐条用例判定。基线采集条件为 NVIDIA A100-SXM4-80GB、CUDA 12.8 / cuBLAS 12.8.4、`cublasChemv`、Host pointer mode。任务书 §3.3 的四条典型 case 对应上限如下（`alpha=(1,0)`、`beta=(0,0)`、`incx=incy=1`、`lda=n`）：

1. `n=512, UPPER`：A100 `14.643 us`，不高于 `18.304 us`。
2. `n=1024, UPPER`：A100 `21.111 us`，不高于 `26.389 us`。
3. `n=2048, LOWER`：A100 `30.089 us`，不高于 `37.611 us`。
4. `n=4096, LOWER`：A100 `65.297 us`，不高于 `81.621 us`。

全部 200 条的标杆分布为最小 `6.144 us`、中位 `36.454 us`、最大 `93.303 us`，对应上限分别为 `7.680 us`、`45.568 us`、`116.629 us`。最小尺寸档的上限接近单次 kernel 启动的量级，是本算子性能风险最集中的区间，也是上文两条实现路线取舍的主要依据。

采样方法存在两处来源，本设计同时执行并分别记录：任务书 §3.3 规定使用 `msprof op` 采集单算子 task duration，自带 5 次 warmup 不计入样本，warmup 后连续执行不少于 10 次取算术平均；随任务提供的 `verify_performance.py` 与 `test_cases/README.md` 则以 GTest 在预置 Device 缓冲上 warmup 后用 `aclrtEvent` 夹住连续下发取均值（`avg_us`，含下发开销），并声明 `msprof op` 的 task duration 仅供交叉验证、不参与判定。两者的差值即单次下发开销，在最小尺寸档足以改变判定结论。本设计按脚本口径输出 `[PERF] case=<case_name> ... avg_us=<数值>` 供 `verify_performance.py` 解析判定，同时以 `msprof op` 采集 task duration 作为交叉验证列入报告，并就以哪一口径作为验收判据请求澄清。算子执行返回 `ACLBLAS_STATUS_SUCCESS` 方计入有效样本，异常样本不计入并在报告中说明。

## 兼容性分析

新增算子，不涉及既有接口的兼容性变更。公开声明新增于 `include/cann_ops_blas.h`，位于同族 `aclblasCsymv` 之后，签名与其逐参数同形，不新增私有平行 API，不改动已接入算子的声明与行为。Host 与 AIV Kernel 实现位于 `blas/symv/arch22/`，与同目录的 `ssymv` 共存且不共享 tiling 结构定义。测试工程位于 `test/symv/chemv/arch22/`，目录形态对齐仓内 `test/symv/csymv/`，包含 CSV 用例、Golden、参数解析、NPU wrapper 与 GTest 主体。`test/symv/chemv/CMakeLists.txt` 在 arch22 分支调用 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)`；仓内同族 `test/symv/csymv/CMakeLists.txt` 与 `test/symv/ssymv/CMakeLists.txt` 当前仅在 arch35 分支注册 GTest，A2/A3 构建退回非 GTest runner 并只输出整数毫秒，本任务在自身目录补齐该注册，不修改同族其他算子的构建文件。`blas/symv/README.md` 增补本算子的产品支持、参数说明与调用方式。
