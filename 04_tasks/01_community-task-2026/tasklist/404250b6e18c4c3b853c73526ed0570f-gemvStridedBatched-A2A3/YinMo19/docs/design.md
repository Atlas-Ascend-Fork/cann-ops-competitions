# gemvStridedBatched 类五个接口设计（Atlas A2/A3）

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)要求在 CANN 9.1.0 上实现并验证 S/HSH/HSS/TST/TSS 五个跨步批量 GEMV 接口。本文以 11 月 24 日修订的任务书为准：转置改变运算，不改变矩阵的物理存储。

任务书发布时所述的「仅有 arch35」与当前源码不同。[上游 !1066](https://gitcode.com/cann/ops-blas/pull/1066) 已实现 arch22 五个入口及计算核；[!1338](https://gitcode.com/cann/ops-blas/pull/1338) 已修复 kernel 发射错误的传递。本方案基于 [ops-blas `55ef791`](https://gitcode.com/cann/ops-blas/commit/55ef791b77f3677b45da3468b0a8b17941424f84)，不重复提交这两项工作。增量集中在参数校验、任务附件接入和经实测定位的性能瓶颈；不增加公共 API，不改变 arch35。

## 需求分析

对第 `b` 批计算 `y_b ← alpha · op(A_b) · x_b + beta · y_b`。三块数据的基址分别偏移 `b*strideA`、`b*stridex`、`b*stridey`，单位均为各自类型的**元素**。所有模式下 `A_b(r,c)` 的地址均为 `A + b*strideA + c*lda + r`。N 模式下 x/y 的逻辑长度为 n/m；T、C 模式下为 m/n。数据为实数，C 与 T 共用路径。

| `aclblas` 接口后缀 | A、x | y |
| --- | --- | --- |
| `SgemvStridedBatched` | FP32 | FP32 |
| `HSHgemvStridedBatched` | FP16 | FP16 |
| `HSSgemvStridedBatched` | FP16 | FP32 |
| `TSTgemvStridedBatched` | BF16 | BF16 |
| `TSSgemvStridedBatched` | BF16 | FP32 |

沿用 `include/cann_ops_blas.h` 的参数顺序。`uint16_t` 只承载 16 位浮点位模式。alpha、beta 为 Host FP32 标量；A、x、y 为 Device 地址。每批独立，原地更新 y；不广播，不要求支持 y 与 A/x 别名，不触碰 y 的填充和批间间隙。

## 详细设计

### Host：结构校验先于空操作

五个接口复用 `blas/gemv_strided_batched/arch22/` 的 Host 分派。当前 `GemvStridedBatchedImpl` 在尺寸或批数为零时直接返回，且 `ValidateGemvSbParams` 只校验 stride 为正。前者会把 `m=0, incx=0` 当作合法调用，后者会把 `batchCount=2, strideA=lda*n-1` 下发到设备。设计将两种情况分开处理：

1. 按 `handle → trans → m/n → lda → incx/incy → 三个 stride → batchCount → alpha/beta → A/x/y` 判断。handle 为空返回 `HANDLE_IS_NULLPTR`，非法 trans 返回 `INVALID_ENUM`，其余非法值返回 `INVALID_VALUE`（状态前缀均为 `ACLBLAS_STATUS_`）。因此 `trans=999, batchCount=0` 仍为非法枚举。`lda >= max(1,m)` 与转置无关。inc 不得为零；附件将 `INT_MIN` 也列为负例，与现有 arch22 行为一致，保留并在评审中注明与任务书“任意非零 int”的差异。
2. 三个 stride 始终须为正。存在相邻且非空的 batch 时检查区间占用：A 至少 `int64_t(lda)*n`，x/y 至少 `1+(len-1)*abs(int64_t(inc))`。先提升再乘、不用 `abs(int)`。最后一批的起始偏移加最大访问下标必须可表示，先用除法判界再乘。独立输入指针间的实际别名由调用方保证，不通过读取 Device 数据验证。
3. alpha/beta 指针始终非空。`batchCount=0 || m=0 || n=0` 时不需要 A/x/y，不查核数、不发射；附件 `TC_ED_191` 在 `m=0,n=16,strideA=1` 时即属此类。非空问题须先检查 A/x/y，`alpha=0` 免**读取内容**，不免基址检查。
4. 合法非空调用传递 FP32 标量值、布局和 dtype 到已有 tiling 数据，在 `handle->stream` 异步发射。不在 API 内同步；保留上游的发射失败返回 `EXECUTION_FAILED` 路径。

### Kernel：复用列主序路径，按实测选择分核粒度

既有 AIV 核在 N 模式沿列搬入连续行片，在 UB 中累加多个输出；T/C 模式按列搬入连续的归约片，与 x 做 FP32 点积并合并分段和。负 inc 的起点为 `(len-1)*abs(int64_t(inc))`，随后按有符号 inc 访问。窄类型先扩展为 FP32，写回时才按输出 dtype 舍入。`alpha=0` 跳过 A/x 的搬入和乘积；`beta=0` 用零初始化累加器，不搬入 y。跨度大时只写实际 y 元素，不用覆盖间隙的整窗写回。

沿用单核私有 UB：现有主片 2048 个元素，非单位步长用至多 4096 个元素的局部窗口，越过窗口时逐点搬运；尾片只拷贝有效字节。复用缓冲前保证 MTE2、Vector/Scalar、MTE3 的生产者与消费者按事件顺序完成。不新建 GM workspace 或临时转置矩阵。

当前只按 batch 分核，`batchCount=1` 时单核处理全部输出。先按 N/T、批数、向量步长分别采集基线。若大矩阵小批数受限，拟在**连续、对齐的输出**上按 `(batch, 输出片)` 分核：N 分行、T/C 分列，每个核独立完成一段输出的完整归约，不跨核累加。输出片边界按实际 y 地址和 dtype 对齐 32 B，不能保证相邻核写入安全时（含非单位 incy、紧邻且非对齐的 batch 边界）让同一核拥有共享写回块。小形状避免过度切分。是否启用以同一机器上的前后精度及 kernel 时长决定，不预设加速比。

## 可维可测分析

### 测试接入与精度

保留原 `gemv_strided_batched_test` 回归目标。在 `test/gemv_strided_batched/arch22/` 接入附件分变体 CSV，并使任务脚本能定位对应测试目标。附件没有 `dtype` 列，须由变体选择具体 API；把 `aarray_fill/xarray_fill/yarray_fill` 映射到实际输入，`null` 标量和 `NULLPTR` 张量必须传入真实空指针。负向用例先走返回码检查，不根据非法形状分配缓冲；不能只用 S 入口代测五个变体。以不同哨兵填充 y 的步长空隙与批间间隙，计算后验证未被改写。

五份 CSV 各含 1000 条精度/边界用例和 200 条 `TC_PF`。覆盖 N/T/C、非方阵及带 padding 的 lda、正负 inc、相邻批重叠、空形状组合非法参数、未读侧含 NaN、FP16/BF16 的 Inf/NaN。每批以**实际量化后的输入**调用 Netlib 3.10.0 `cblas_sgemv` 作为 FP32 参考，再按输出 dtype 舍入；报告说明与任务书允许的 FP64 累加路径的差异。特殊浮点值单独检查类别和 Inf 符号，不把 NaN 放进普通误差排序。

按输出 dtype 取 `atol=rtol`：FP32 为 `2^-13`，FP16 为 `2^-10`，BF16 为 `2^-7`。以 `|实际值-参考值| <= atol + rtol*|参考值|` 统计逻辑 y 元素，匹配率至少 0.99，且同时满足任务书最大绝对误差限制（FP32 `1e-2`，FP16/BF16 `1e-1`，或按标准的 32 ULP 判据）。不以 CSV 的 MERE/MARE 两列替代生态判据。

### 正式性能与交付条件

在 Atlas 800T A2（910B3）、CANN 9.1.0 上，先验证性能输入正确，再用 `msprof op` 逐例采集 `Task Duration(us)`。同一 case 预热 5 次，取其后 10 次成功发射且执行成功的有效样本均值；失败或缺样单列，不计为通过。比较 `mean_us / 1000 <= 1.25 * gpu_ms`。以变体、trans、m/n、lda、三 stride、两 inc、batchCount、alpha/beta 对齐 A100 基线。任务书修订指出 T/C 基线须重测，附件说明另提示非默认标量待复测；键能匹配、单元格非空不证明数值已重测，正式报告须记录基线版本和来源。

辅助计时在预置 Device 缓冲上用同一 stream 的 `aclrtEvent` 包住 10 次下发，不含分配、H2D/D2H 或 golden，输出 `[PERF] case=... avg_us=...`。不把事件结果当作 kernel 时长；含非零 beta 的正确性检查使用初始 y 的独立副本，避免拿连续原地更新的结果与单次 golden 比较。任务书不要求专项内存性能报告；地址边界和未写区域仍须验证。

正式验收记录设备、工具链、源码 commit、用例、基线版本和原始日志；设计 PR 不包含达标声明。附件中未附 §4 提到的 `cases.json`、`facts.json` 和冻结 `inputs/`、`golden/`，在[任务讨论 #91](https://gitcode.com/cann/ops-blas/discussions/91)确认完整自验包及基线后再提交验收。
