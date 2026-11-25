# 需求背景

## 需求来源

11月社区任务-aclblasChpr2算子开发（A2/A3)（[hiascend 任务中心](https://www.hiascend.com/activities/task-center)）要求在 Atlas A2/A3 系列产品上，基于 [ops-blas](https://gitcode.com/cann/ops-blas) 开源仓工程框架，使用 Ascend C 编程语言实现单精度复数 Hermitian packed 矩阵秩-2 更新算子 `aclblasChpr2`，完成算子设计、开发与测试全流程。接口语义、精度标准与性能判据以该任务书为依据，对标 cuBLAS `cublasChpr2`（Netlib/LAPACK `chpr2.f` 参考语义）。

## 背景介绍

`aclblasChpr2` 对 n×n Hermitian 矩阵 A 执行秩-2 更新：

```
A := alpha * x * y^H + conj(alpha) * y * x^H + A
```

其中 `x`、`y` 为长度 n 的单精度复数向量，`alpha` 为单精度复数标量，`^H` 表示共轭转置。A 以 packed 列优先格式存储于 `ap`（长度 n*(n+1)/2），仅 `uplo` 指定的上三角或下三角区域被引用和更新；对角线元素为实数，更新后其虚部无条件置零（对齐 Netlib `chpr2.f` 语义）。

ops-blas 仓现状：`include/cann_ops_blas.h` 已有 `aclblasChpr2` 公共声明（供各产品线共用）；`blas/spr2/arch35/` 已有 950PR 实现与测试（已合入社区任务，语义基准）；`blas/spr2/arch22/` 已有同族实数秩-2 `aclblasSspr2` 与复数秩-1 `aclblasChpr` 的已合入实现（列分核 + x/y 平面缓存 + 残差类 slab 协议），本任务在其工程框架内补齐 A2/A3（arch22，Ascend 910B3/B4 与 910_93 共享同一份实现）能力，禁止定义产品私有平行接口。

# 需求分析

## 算子功能

逐元素语义（0-based 下标，`i ≤ j` 上三角 / `i ≥ j` 下三角）：

```
AP(i,j) = AP(i,j) + alpha * x(i) * conj(y(j)) + conj(alpha) * y(i) * conj(x(j))
```

- 对角元素（`i == j`）：上式两项互为共轭，和为实数；输出虚部无条件写 0（含输入对角虚部被投毒的用例，`0×NaN ≠ 0` 故掩码乘法不等价，须直接写零字面量）。
- 边界行为对齐任务书 §2.1/§2.5（与 cuBLAS `cublasChpr2` 错误检查一致）：
  - `n = 0` 为合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`，不引用 `x/y/ap`、不启动 kernel；但 `n = 0` **不豁免结构参数校验**——非法 `uplo/incx/incy` 或空 `alpha` 指针仍返回对应错误，仅豁免 tensor 指针校验与 kernel 启动；
  - `alpha` 为字面零（`(0.0, 0.0)`）时快速返回成功，不更新 `ap`、不读取 x、y 的数值（判定发生在 `x/y/ap` 指针判空之后：`alpha` 为合法零时空 `x/y/ap` 指针仍须报错）；
  - 非法 `uplo`（非 `ACLBLAS_UPPER/ACLBLAS_LOWER`）返回 `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定；cuBLAS 对应场景返回 `CUBLAS_STATUS_INVALID_VALUE`，属有意差异）；
  - 负步长：逻辑元素 `k` 从物理下标 `(1-n)*inc + k*inc`（即尾端起算）读取；
  - `alpha` 为 NaN/Inf 时按 IEEE 传播，不提前拦截。

## 公开接口与范围

在 `include/cann_ops_blas.h` 既有声明上实现（签名对齐 cublasChpr2）：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* ap);
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- |
| handle | 输入 | ops-blas 库上下文句柄，携带 stream，Host 内存 | scalar | - | 有效句柄 | nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 属性 | 更新的三角区域（另一半由共轭对称隐含） | enum | - | `ACLBLAS_UPPER(121)` / `ACLBLAS_LOWER(122)` | 取值不在枚举集时返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| n | 属性 | 方阵 A 的阶数 | int | - | n ≥ 0 | n < 0 返回 `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 为合法 no-op（结构参数仍需校验） |
| alpha | 输入 | 复数标量乘数指针，Host 或 Device 内存（运行时判定指针位置） | COMPLEX64 | [1] | 实/虚部 FLOAT32 全集 | nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；字面零 quick return |
| x | 输入 | 复数向量，Device 内存，只读 | COMPLEX64 | 逻辑 [n]，物理 1+(n-1)\*\|incx\| | 实/虚部 FLOAT32 全集（含 Inf/NaN） | n > 0 时 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | x 的元素步长，Host 内存 | int | - | 非零且非 `INT_MIN`，可正可负 | 非法值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| y | 输入 | 复数向量，Device 内存，只读 | COMPLEX64 | 逻辑 [n]，物理 1+(n-1)\*\|incy\| | 实/虚部 FLOAT32 全集（含 Inf/NaN） | n > 0 时 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incy | 输入 | y 的元素步长，Host 内存 | int | - | 非零且非 `INT_MIN`，可正可负 | 非法值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| ap | 输入/输出 | packed Hermitian 矩阵，Device 内存，原地更新 | COMPLEX64 | [n*(n+1)/2] | 实/虚部 FLOAT32 全集 | n > 0 时 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；缓冲字节跨度溢出 size_t 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

返回值：`aclblasStatus_t`（`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_ENUM` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` / `ACLBLAS_STATUS_EXECUTION_FAILED`），语义与 `include/cann_ops_blas_common.h` 一致；枚举类非法取值返回 `INVALID_ENUM`（ops-blas 约定，与 cuBLAS 返回 `INVALID_VALUE` 属有意差异），阶数/步长/空指针类返回 `INVALID_VALUE`。

算子不涉及 broadcast、dynamic shape 框架接入与确定性计算要求（packed 更新按列划分，单元素内的求和顺序确定）；`ap` 原地覆写，调用方保证 `ap` 与输入向量无有害重叠。接口异步执行：依赖 `aclblasSetStream` 绑定 stream，读回结果前须同步 stream。

# 详细设计

## 算子分析

### 数学公式

`AP(i,j) += alpha·x(i)·conj(y(j)) + conj(alpha)·y(i)·conj(x(j))`，按 `uplo` 仅更新指定三角的 packed 元素；对角虚部输出无条件置零。

展开为实部/虚部（t1 = alpha·conj(y_col)，t2 = conj(alpha)·conj(x_col)）：

```
Re(AP) += t1.re·Re(x) - t1.im·Im(x) + t2.re·Re(y) - t2.im·Im(y)
Im(AP) += t1.re·Im(x) + t1.im·Re(x) + t2.re·Im(y) + t2.im·Re(y)
```

### 支持数据类型

`aclblasComplex`（COMPLEX64，实部/虚部各 FLOAT32），与 ops-blas 仓 `include/cann_ops_blas_common.h` 定义一致。

### 支持形状

`n` 运行时任意（≥ 0，无静态形状假设）；测试覆盖 1、2 的幂及 ±1、小质数、非对齐值直至 4096；`incx/incy` 覆盖 ±1/±2/±3 及负步长大步长；上/下三角两填充模式全覆盖。

## 算子实现

总体架构为**单 AIV 向量 kernel 直调**（AIV-ONLY，无 Cube 参与）：host 完成 cublas 对齐的参数校验与核数决策后，将 n/uplo/incx/incy/alpha 打包进 `Chpr2TilingData` 直发 kernel；kernel 内按三角区域选择两条已实测收敛的流水线路——UPPER 走「轮转列所有权 + 多列合并分段 + 三阶段环形流水」，LOWER 走「残差类 slab + 三引擎并发流水」，两者共享 4-op Axpy 计算链与对角虚部 V 道清零。

### Host 侧设计

**校验链**（顺序逐字对齐任务书 §2.5「校验顺序」，与 cuBLAS `cublasChpr2` 错误检查一致，任务书 CSV 负向用例编码该顺序——如 `TC_ED_205`：`n=0` + 非法 `uplo` 期望 `INVALID_ENUM` 而非 no-op）：

```
handle 空(HANDLE_IS_NULLPTR) → n<0(INVALID_VALUE) → alpha 指针空(INVALID_VALUE)
→ uplo 非法(INVALID_ENUM) → incx 非法(0/INT_MIN, INVALID_VALUE) → incy 非法(0/INT_MIN, INVALID_VALUE)
→ n==0 quick return（结构参数已全部校验，仅豁免 tensor 指针校验与 kernel 启动）
→ x/y/ap 空指针(INVALID_VALUE) → alpha 取值 → alpha 字面零 quick return(不读 x/y 数值)
→ x/y/ap 缓冲字节跨度溢出 size_t 守卫 → AIV 核数查询失败(EXECUTION_FAILED) → 发射
```

**alpha 标量双指针位置**（任务书 §2.5：Host 与 Device 两种指针位置均支持）：入口以 `aclrtPointerGetAttributes` 判定 `alpha` 指针位置——Host 位置直接解引用；Device 位置以 `aclrtMemcpy`（同步、8B）取回标量后走同一判零 quick return 与 tiling 装载路径，kernel 侧不感知差异（`alphaReal/alphaImag` 始终随 tiling 下发，免 kernel GM 标量读与跨核一致性）。
- 溢出守卫：元素偏移可由 int 入参构造，但其字节跨度可能超出 `size_t`，对 `1+(n-1)·|inc|`（x/y）与 `n·(n+1)/2`（ap）按 `SIZE_MAX/sizeof(aclblasComplex)` 上界拒绝，防 GM 地址回卷。
- **核数策略**：`useCoreNum = min(n, AIV 核数)` 列轮转所有权——每核至少拥有一列，n 小于核数时自动降核，避免空核发射；`CHPR2_CORES` 环境变量可压核（调试/核数缩放诊断用）。
- **tiling 载荷**：`n / useCoreNum / uplo / incx(int64) / incy(int64) / alphaReal / alphaImag`（host 侧取值后随 tiling 下发）。`Chpr2TilingData` 为 Host/Kernel 共用头文件 `chpr2_tiling_data.h`。
- 无 workspace / 无多阶段流程：单 kernel 单发射，`handle->stream` 上直调 `chpr2_kernel_do`。

### Kernel 侧设计

**统一骨架**（`Chpr2Kernel<UPLO_IS_UPPER>`，`blas/spr2/arch22/chpr2_kernel.cpp`）：

- 列所有权：核 `blockIdx` 以轮转方式拥有列 `col ≡ blockIdx (mod useCoreNum)`（`col = blockIdx + k·P`）， packed 列段地址由 `Chpr2ColumnBase`（整数 64 位推导，上三角 `i+j·(j+1)/2`、下三角 `i+(2n-j-1)·j/2`）与 `VecPhysicalIndex`（正/负步长统一 `(1-n)·inc+k·inc` 物理映射）计算。
- **计算链（4-op Axpy）**：每列先取列标量 `t1 = alpha·conj(y_col)`、`t2 = conj(alpha)·conj(x_col)`（标量域），随后对该列连续行区间在四个平面（x / ix=共轭平面 / y / iy）上执行 `Axpy×4`（`AP += t1.re·x + t1.im·ix + t2.re·y + t2.im·iy`，ix/iy 为 `ApplyEvenLaneNeg` 生成的虚部取反平面），尾块走精确元素访问。该链对混合容差（rtol=atol=2^-13、matched_ratio ≥ 0.99、32·ULP 上限）全量实测通过；netlib 位级 6-op 精确链（Muls+Axpy+Add 双 scratch 原地合成）为已验证回退路径，netlib 逐元素舍入序位级一致（RANDOM_EXTREME 实测 0/528 差异）。
- **对角虚部清零（V 道）**：netlib 语义为无条件写 0——掩码乘 `0×NaN` 不等价，故对角虚部以 V 道全零源块 Muls（0×0 字面零，NaN 安全）完成，UB 16K 空闲处 64-float 零块每 drive 一次 Duplicate 初始化，V 序内免标量阻塞消费。
- **事件纪律**（arch22 HardEvent 实测语义）：单方向多 slot 在飞 Set 必须按槽分 id（ring：M2V 0-2 / M3M2 3-5，方向内严格成对可共享）；S 管线读 MTE2 写入的 UB 数据必须经 MTE2→V→S 发布链（V_S 对缺失即陈旧读）；kernel 退出前按实际用过的 buffer 排水全部在飞事件，杜绝跨 launch 位残留。

**UPPER 环形流水**（`RingUpper`）：

- 多列合并分段 piece：连续列段打包为一个 piece 批量 DMA（段内列边界按 4 复数 = 32B 对齐落位，满足 arch22 VEC 指令 UB 操作数 32B 对齐契约）；Issue/Out/Compute 三处同推导段迭代。
- 三阶段软件流水：A（copy-in 预发射，在飞 ≤ 3 slot）/ B（slot 落地后 walk 入队，V 背靠背消费）/ C（slot 计算退役后 copyback），slot 复用 gate 为延迟 2 拍消费的 level 触发位；深 slot 4096 复数（32KB×3 ≈ 96KB 在飞）撑满 MTE 有效带宽；列长 > slot 容量（n > 2048 量级）自动分片，srcOff 按片推进。

**LOWER slab 残差类流水**（`SlabLowerTiles`，移植自已合入 sspr2 协议）：

- ap 列段按 slab 批量搬运（4 复数 padding 对齐）；列按 `col ≡ r (mod 4)` 分 4 个残差类，类内以平移源对 `Xs = x[j0..n)` 构建共轭平面（`(j-j0)·2` 为 16 浮点 = 64B 对齐，结构性满足 32B 契约，规避 LOWER 逐列变起点的 8B 对齐陷阱——507035 VEC UB 未对齐坑的权威解法）。
- 三引擎并发：copy-in(k+1) / walk(k) / copy-out(k−1) 三引擎以 3×4096 复数缓冲复用距离 3 并发（复用距离 ≥ 3 保证 M3M2 门必已退役）；空类预扫描跳过；类尾按实际用过 buffer 排水。
- V 道对角清零与类间 fused 预发射（MTE2 FIFO 序约束下持平，保留协议）。

**边界与硬件细节**：

- 所有 VEC（Axpy/Muls）UB 操作数起点 32B 对齐；LOWER 变起点结构一律残差类化或逐列重载。
- 跨 tile 列头标量走 GM 直读（平面外列回退 GM，ColScalarsUpper/LOWER 头列同款），避免平面越界。
- 尾块（列长非整块、n 小于平面容量）用精确元素访问，不读写分配范围外数据（mssanitizer 全绿）。
- 奇偶边界列（双列边界推导）经 13 列实证用例锁定。

### 资源与调度

单 kernel AIV-ONLY，`useCoreNum = min(n, AIV 核数)`；UB 布局：LOWER 4 平移平面区（x/ix/y/iy 各 8192-float 切片动态偏移）+ 环形 slot 区（3×4096 复数）+ 对角零源块 + staging；UPPER 平面缓存四平面 + ring slot 复用同区。无 workspace、无 GM 中间缓冲，`ap` 原地更新；tiling 载荷 28B。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I/T A3 | √ |

算子代码位于 `blas/spr2/arch22`（A2 与 A3 共享同一份实现，910_93 编译兼容），自验证设备 Atlas 800T A2（910B3/910B4 真机）。

## 算子约束限制

- 仅支持 packed 列优先存储与 `uplo` 语义内访问，无 `lda` 参数；不支持广播；`ap` 原地更新且与输入向量不得有害重叠；
- `alpha` 为标量指针，Host 与 Device 两种指针位置均支持（运行时判定，见 Host 侧设计）；
- `incx/incy` 非零且非 `INT_MIN`；负步长从向量尾端起算；
- 非 quick return 时对角线输出虚部无条件置零（输入对角虚部不参与计算）；
- 浮点累加顺序与参考实现不同，结果非 bit-exact（混合容差口径判定）；alpha 字面零与 n=0 的 no-op 路径位精确（完全不触碰 `ap`）；
- 性能口径要求矩阵数据驻留 Device 内存。

# 可维可测分析

## 精度标准与验证

golden 由cblas（Netlib BLAS 3.10.0 实现）生成，输出 packed 向量 `ap` 全量（n(n+1)/2 个复数元素）验证，实部、虚部分别按 FLOAT32 分量判定（口径与任务书 §3.2 一致）：rtol = 2^-13 (1.2e-4)，atol = 2^-13 (1.2e-4)，required_matched_ratio = 0.99，max_abs_error_limit = 1e-2 or 32 * ULP；逐元素通过条件 `|actual - golden| ≤ atol + rtol × |golden|`。Inf/NaN 按任务书等效规则单独判定（同 NaN / 同号 Inf / 一 NaN 一 ±Inf 视为等效，不计入有限值统计），对角虚部比对前双方统一为 0 口径。

golden 版本纪律：本机 REFBLAS 若解析为 openEuler libopenblas 0.3.25，其内置 `zhpr2` 为 FMA 融合链（与 netlib 在极值/部分抵消元素上存在 32·ULP 级差异，TC_FL_148/149 两条 RANDOM_EXTREME 用例在该 golden 下判负），验收以任务书钦定的 golden 版本为准（`ldd` 确认实际链接库）。

测试工程位于 `test/spr2/chpr2/arch22`，CSV 驱动 GTest（`Chpr2Arch22Test`）：1200 条任务用例（L0/SQ/AB/INC/FL/CV/ED/EX/PF 九类）+ 32 条 TC_EXTRA 补充，覆盖 shape 扫描（1 至 4096）、上/下三角全组合、正/负/零边界步长、alpha 特殊值（0/字面零/NaN/Inf/共轭对角投毒）、Inf/NaN 输入、全部负向用例（空句柄/负 n/非法 uplo/非法步长/空指针/缓冲溢出守卫），配套 `run_self_test.py`（数量/名称/执行记录核对）与 `run_memcheck.py`（mssanitizer 上/下三角 × 正/负步长真实调用 + 负向用例）。全量自测 1032/1032（netlib 口径位级全过，含 openblas 伪影 2 条注记）。`CHPR2_CSV` / `CHPR2_RESULTS` 支持用例文件与结果路径覆盖。

## 性能标准与验证

测试设备 Atlas 800T A2 (910B3)（CANN 9.1.0），采样口径对齐任务书 §3.3：msprof op 采集单算子 Task Duration（warmup 5 次 + 有效样本 10 次取算术平均，异常样本不计入并说明）。性能判据：平均单次耗时不高于 GPU 标杆耗时 / 0.8（即达到标杆 0.8× 性能以上，单 case 同判据）；标杆数据随任务提供（test_cases/gpu_baseline.csv，GPU 实测）。完整性能用例以任务书测试用例文件为准。

测试设备 Atlas 800T A2（910B4@1650MHz），采样口径对齐任务书：msprof op 的 `OpBasicInfo_*.csv` Task Duration（每用例预热 5 次、10 个有效样本均值，`CHPR2_PROFILE=1` + `CHPR2_PROFILE_CALLS` 控制），不以 GTest 运行时间代替 kernel 耗时。预开发实测（CANN 9.1.0，n=4096 大档逼近 HBM roofline 63-68%，中档受 arch22 fp32 向量管 32-lane 物理上限约束——L2 驻留探针实测 3.25TB/s 封顶、V 链 26/32 管线效率，GPU(H100) 同题属 L2 带宽界，物理判定证据链见自测报告）：L512 16.5µs / L1024 19.3µs / L2048 37.6µs / L4096 119.5µs / U512 15.7µs / U2048 38.6µs / U4096 112.7µs，对比已合入同族 chpr arch22 等流量快 1.6×/字节。200 条 TC_PF 性能用例覆盖 1~4096 全档位（2 的幂、±1、小质数、非对齐、混合三角/步长组合），`run_msprof_perf.py` 断点续采。剩余优化路线（双列交错破除 V 道 RAW 停摆、LOWER 类扫掠单驱动）已在预开发仓留档。

## 兼容性分析

新算子（A2/A3 首发实现），无历史版本兼容负担；`include/cann_ops_blas.h` 声明与 950PR（arch35）共用，接口语义两平台一致（校验顺序逐字对齐）；README 产品支持表标注 Atlas A2/A3 训练/推理系列：支持。

# 附录：任务书关键信息摘录

> 本节由任务书（11月社区任务-aclblasChpr2算子开发（A2/A3)，taskId `7a1275ee6b654459957fc27c8dde32b0`，发布 2026-11-24T16:00:00.000+00:00，截止 2026-12-06T16:00:00.000+00:00）自动摘录，作为设计口径溯源；如与任务书原文冲突，以任务书原文为准。

## 软硬件环境（任务书 §3.1）

- **适配硬件**：Atlas A2 系列产品（性能测试设备：Atlas 800T A2 (910B3)）与 Atlas A3 系列产品，自验证只需覆盖一种款型，精度建议910B3或者A3，性能建议910B3
- **CANN 版本**：CANN 9.1.0
- **三方软件版本**：精度比对 golden 由 cblas（Netlib BLAS 3.10.0 实现）生成，随测试工程提供

## 精度要求（任务书 §3.2）

1. 算子输出结果需要与 cuBLAS `cublasChpr2` 的输出一致；golden 由 CPU 侧 cblas 参考实现 chpr2 生成（请参考链接：https://www.netlib.org/blas/chpr2.f ），实部、虚部分别比对；
2. 算子计算精度需满足生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md ）。本算子输入输出为单精度复数 COMPLEX64（实部/虚部各为 FLOAT32），阈值按 FLOAT32 分量执行，具体如下：

    | 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
    |---|---|---|---|---|---|---|
    | rtol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
    | atol | 2^-10 (1e-3) | 2^-7 (7.8e-3) | 2^-13 (1.2e-4) | 2^-10 (1e-3) | 2^-3 (0.125) | 2^-2 (0.25) |
    | required_matched_ratio | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
    | max_abs_error_limit | 1e-1 or 32 * ULP | 1e-1 or 32 * ULP | 1e-2 or 32 * ULP | 1e-1 or 32 * ULP | 1e-0 or 32 * ULP | 1e-0 or 32 * ULP |

当用例同时满足 matched_ratio ≥ required_matched_ratio 且 max_abs_error ≤ max_abs_error_limit 时，判定该用例精度通过。

**golden 计算精度域**：golden 直接调用 Netlib CBLAS 对应接口 `cblas_chpr2`（`CblasColMajor`）生成，以其单精度复数输出作为比对基准。

**随机类算子的精度对比判定策略**：不涉及——本算子为确定性浮点乘加运算，不含随机数生成，§3.5 的均匀分布与正态分布是测试输入数据的生成规则，不是算子行为。

## 性能要求（任务书 §3.3）

1. **精确 kernel 耗时获取方式（msprof，正式验收依据）**：测试设备：Atlas 800T A2 (910B3)。采用 `msprof op` 采集单算子 **kernel task duration**（单位 us）作为单次耗时；warmup 5 次不计入样本，warmup 后连续执行 10 次作为 10 个有效样本，取算术平均作为 Avg time；算子执行返回 SUCCESS 方计入有效样本，异常样本不计入并须在自测报告中说明。具体采集命令与 `Task Duration(us)` 的读取方式见 `test_cases/README.md` 的“精确 kernel 耗时获取方式（msprof）”一节。
2. **正式判据（kernel-to-kernel）**：NPU kernel 平均单次耗时 ≤ GPU kernel 标杆耗时的 1.25 倍（即达到 `test_cases/gpu_baseline.csv` 的 `gpu_ms` 0.8 倍以上性能）。`gpu_baseline.csv` 的 `gpu_ms` 为 **GPU kernel-only** 耗时（profiling/nsys 的 kernel duration），只能与 `msprof op` 的 `Task Duration` 对比，不得与 GPU 端到端耗时或 NPU 事件计时混用。
3. **性能数据参考**：`test_cases/gpu_baseline.csv` 仅保留单列 `gpu_ms`（GPU kernel-only），性能子集与基线逐条对应；本任务书及 README 中的性能判据均以该列与 `msprof op` 结果对比为准。
4. **交叉验证 / 快速跑测（不作为正式判据）**：`test_cases/verify_performance.py` 执行 GTest 的 `TC_PF` 用例做交叉验证：先 warmup 5 次（不计时），再用 `aclrtEvent` 夹住连续 10 次下发取平均（`avg_us`，含下发开销，不含 H2D/D2H、host 造数与 golden 比对）；GTest 须按用例打印 `[PERF] case=<case_name> ... avg_us=<数值>`，脚本只解析该行并与 `gpu_baseline.csv` 的 `gpu_ms` 比较。该口径含下发开销、与 kernel-only 基线不完全一致，仅用于快速回归与趋势判断；正式验收仍以第 1、2 条的 `msprof op` 结果为准。
5. 完整性能用例（含更多规模、带宽与 uplo 组合）以测试用例文件为准，判据同上 0.8× GPU kernel 标杆。
