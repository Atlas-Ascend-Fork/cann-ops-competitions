# 需求背景（required）

## 需求来源

CANN 社区任务 2026，任务 10-1：单精度实数 LU 分解、求解和批量求逆（Ascend 950PR）。贡献者：`zhanghua145`。实现目标仓库：[cann/ops-solver](https://gitcode.com/cann/ops-solver)。

任务依据为 `Atlas950_Sgetrf_Sgetri_SgetriBatched_SmatinvBatched_task_doc.md` 及随附的四个算子测试包（sgetrf / sgetri / sgetribatched / smatinvbatched，各含 cases.json + bench_result.json）。本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织。

## 背景介绍

带部分主元的 LU 分解 `P·A = L·U` 是通用（非对称）线性求解与矩阵求逆的基础。任务要求在 Ascend 950PR 上交付与 cuSolver DN / cuBLAS / LAPACK 对齐的 4 个 C API：`aclsolverSgetrf`（LU 分解，含 `devIpiv == nullptr` 不选主元分支）、`aclsolverSgetri`（基于因子的原地求逆，LAPACK 语义）、`aclsolverSgetriBatched`（基于因子的 out-of-place 批量求逆）、`aclsolverSmatinvBatched`（n ≤ 32 原始矩阵直接批量求逆）。全部数值计算由 NPU AI Core 完成，禁止 CPU fallback。

ops-solver 仓内已有 `sgetrf` / `sgetri` 简化原型，但与对标接口不对齐（Host 指针、强制 lda==n、info 未写入、无 bufferSize 查询），本任务须将其升级为公开接口定义并新增两个批量接口。本团队在 9 月已完成同仓 Cholesky 五接口（09-73，残差 768/768 全过）的全部交付件与踩坑知识（blocked 分解的 AIV/AIC 分工、批量接口的 watchdog/一致性/清零问题族），本设计直接继承其已验证架构与构件。

任务核心特征（决定设计）：

1. **四接口一任务**：一并交付，不允许拆分验收；同一 PR 或一组关联 PR 合入。
2. **矩形矩阵与选主元是两个新增硬点**：sgetrf 全量 148 条中 136 条为 m≠n 矩形（m ∈ [2,4096]，n ∈ [2,8168]，含 m=64×n=4947 极宽形态）；部分主元引入列内归约（确定主元）、面板内与跨面板行交换（laswp）两类 Cholesky 没有的数据移动。
3. **性能标杆反算**（real fp32：getrf ≈ 2/3·m·n·min(m,n)，getri ≈ 2n³，小 n 批量为带宽/launch 主导）显示标杆分四段：

| 编号 | 接口 | 规格 | GPU 耗时 | GPU 有效算力 | 门槛（NPU ≤） | 分段判读 |
| --- | --- | --- | --- | --- | --- | --- |
| P-01 | Sgetrf | m=n=1024 | 2.035 ms | 0.70 TF/s | 5.814 ms | 利用率不足段：Cholesky 同型锚点实收 0.73 ms（1/2 规模 flops），LU 预计 1.5~2 ms，富余大 |
| P-02 | Sgetrf | m=n=4096 | 12.012 ms | 5.65 TF/s | 34.320 ms | 算力对抗段：blocked 尾部 GEMM（fp16 三拆 Ozaki）跑 Cube；Cholesky 4096 实收 6.2 ms（1/2 flops），LU 预计 13~18 ms |
| P-03 | Sgetrf | m=2048, n=1024 | ~2.4 ms（邻近 shape） | ~1.2 TF/s | ~6.9 ms | 过渡段，同上 |
| P-04 | Sgetri | n=1024 | 0.941 ms | 2.3 TF/s | 2.689 ms | 2n³=2.1 GFLOP；两条 blocked trsm 的尾部更新走 Cube，预计 ≈2 ms |
| P-05 | Sgetri | n=4096 | 7.811 ms | 4.4 TF/s | 22.318 ms | 137 GFLOP @ Cube ≈10 TF/s ≈ 14 ms + 对角块串行相，预计 16~19 ms |
| P-06 | SgetriBatched | n=32 × 1024 | ~0.04 ms（邻近 batch 插值） | — | ~0.12 ms | launch/带宽主导段：融合 kernel，数据 2 MB |
| P-07 | SgetriBatched | n=128 × 128 | ~0.15 ms（插值） | — | ~0.43 ms | 同上 |
| P-08 | SmatinvBatched | n=16 × 1024 | ~0.028 ms | — | ~0.08 ms | launch 主导：单融合 kernel（≈5.5 µs）+ 0.5 MB 搬运 |
| P-09 | SmatinvBatched | n=32 × 256 | ~0.074 ms | — | ~0.21 ms | 同上 |

批量段的两个极端（均在全量 case 内，非锚点）：sgetriBatched n=2 × 10⁶（GPU 0.033 ms → 门槛 0.093 ms，数据 ~48 MB，带宽下限 ~25 µs，融合分块 launch 后可达）；n=3691 × 32（GPU 14.43 s → 门槛 41.3 s，cublasSgetriBatched 大 n 小批为已知弱项，3.2 TFLOP 在 NPU blocked 管线 ≈0.4 s，>80× 富余）。smatinvbatched 覆盖 n ∈ [2,32] 全整数 × batch ∈ [128, 10⁷]（n=2/4 至 10⁷；两条 10⁷ case GPU 侧采集失败、无基线，按无门槛处理）。

内部目标：单矩阵大档 ≥0.6× GPU，批量段 ≥1×，全量 712 条逐条过门槛。**验收范围 = 测试包全量 case（sgetrf 148 + sgetri 200 + sgetribatched 149 + smatinvbatched 215，共 712 条），上表仅为锚点抽样。**

# 需求分析（required）

## 需求描述

共同交付四种计算能力及两个 workspace 查询接口：

| 接口 | 数学语义 | 输入 | 输出 | info 语义 |
| --- | --- | --- | --- | --- |
| `aclsolverSgetrf_bufferSize` | — | m、n、A、lda | Host Lwork | workspace 元素个数（float 计数） |
| `aclsolverSgetrf` | `P·A = L·U`（部分主元；devIpiv=nullptr 时不选主元 `A = L·U`） | m×n 矩阵 A | 原位 LU 因子 + devIpiv（1-based，长度 min(m,n)） | 0 / -i / i>0（U(i,i)=0，分解止于第 i 步） |
| `aclsolverSgetri_bufferSize` | — | n、A、lda | Host Lwork | 元素个数 |
| `aclsolverSgetri` | `A⁻¹·A = I`（A 为 LU 因子） | LU 因子 A + devIpiv | 原位逆矩阵 | 0 / -i / i>0（U 的第 i 个对角元为 0） |
| `aclsolverSgetriBatched` | 逐矩阵 `C_i = inv(A_i)` | Device 指针数组 Aarray（LU 因子）+ PivotArray（长度 n·batchSize，nullptr=无主元） | 独立输出 Carray（不得与 Aarray 重叠） | infoArray[i]：0 / k（逐矩阵） |
| `aclsolverSmatinvBatched` | 逐矩阵直接求逆（等价 getrf+getri 短路） | Device 指针数组 A（原始矩阵） | 独立输出 Ainv | info[i]：0 / k；n ∉ [0,32] 报错，禁止转调 getriBatched |

规模与泛化（验收下限，允许超范围声明）：getrf m, n ∈ [1,4096]（测试包实测至 n=8168，按更大范围实现）覆盖方阵与矩形；getri n ∈ [1,4096]；getriBatched n ∈ [1,4096]、batchSize ∈ [1,10⁶]；matinvBatched n ∈ [1,32]、batchSize ∈ [1,10⁷]（n=33 必须 INVALID_VALUE）。`lda/ldb/ldc/lda_inv ≥ max(1,·)` 的 padding 全程合法（禁止要求 ==n）。

工程模式：**ops-solver Host C API + AscendC/CATLASS Kernel 直调**；列主序、全 Device 指针、真实写 info（`0 / -i / i>0`）；计算走调用方 stream、禁止无谓 Host 同步；相同输入同 stream 串行多次执行输出（含 ipiv/info）**bit-wise 一致**，主元选取为确定规则（列内绝对值最大、并列取最小下标）；`aclsolverStatus_t` 返回码；补 `aclsolverOperation_t`（N/T/C 对齐 cublasOperation_t）枚举。性能门槛 0.35× GPU（950PR 实测，msprof kernel 口径，与 bench_result.json 逐条关联）。

## 需求拆解

1. 补齐公开 C ABI（6 函数 + `aclsolverOperation_t` 枚举），升级/替换仓内不对齐原型；Host 三段校验 + 异步 stream 行为；-i 编号以 cuSolver/LAPACK xerbla 序冻结（A=4 类：handle=1 起不计、m/n=2、A=3、lda=4、Workspace=5、devIpiv=6、devInfo=7 等，开发初期与 cuSolver 手册对拍冻结写入头文件注释）。
2. 单矩阵 getrf：blocked 右视 + 部分主元。四类数据移动（对角/面板分解、面板内行交换、已算 L 列的 laswp、尾部 GEMM 更新）在 AIV（向量/交换/归约）与 AIC（Cube GEMM）间分工；矩形 m<n 的宽尾（超过第 min(m,n) 列部分）仅做行交换不做消元。
3. getri：`inv(A)` = ipiv 重放 → blocked trsm `L·Y = P·I` → blocked trsm `U·X = Y`，重相输出；对角块串行求解（s-scale 误差控制，继承已验证的寄存器累加模式）+ 尾部 Cube GEMM 更新；U(i,i)=0 检测写 info。
4. 批量接口：Device 指针数组真批量。getriBatched 复用 getri 构件（小 n 融合向量化、大 n 逐矩阵 blocked）；smatinvBatched 单融合 kernel（n ≤ 32 全矩阵驻 UB，含选主元 Gauss-Jordan/分块消元），巨 batch 按 watchdog 上限分块 launch（实测 ~2¹⁸ 矩阵/kernel），info 核内按归属核清零与直写。
5. info 全走 Device 写入路径；空问题（m=0/n=0/batchSize=0）成功返回。
6. 覆盖全量 712 条精度（两步判定：逐元素 + LAPACK 残差 SGET01/SGET03，含 CPU fp32 参考链路 ratio_cpu）、性能、info、边界、padding、非默认 stream、确定性测试。
7. 交付设计文档、四算子实现 + 测试 + 文档（`src/sgetrf/ + src/sgetri/ + src/sgetri_batched/ + src/smatinv_batched/`）、task_submission 交付件，完成评审与验收。

# 详细设计（required）

## 算子分析

### 数学公式

- **Sgetrf**：`P·A = L·U`。A 为 m×n；L 为 m×k 单位下三角（k=min(m,n)），U 为 k×n 上三角（梯形）。原地：对角及以上存 U，严格以下存 L 非对角元。`devIpiv[i] ∈ [1,m]`（1-based）为第 i 步行交换记录。`devIpiv == nullptr` 时不选主元（语义对齐 cusolverDnSgetrf）。SGET01 残差重放：`L·U` 先按 ipiv **逆序**（i = k→1）回放行交换还原到 A 行序再比 `‖L·U−A‖₁`。
- **Sgetri**：`A⁻¹`。输入为 getrf 因子。等价三步：① 将单位阵按 ipiv 正序展开为 `P·I`（行重排）；② 解 `L·Y = P·I`（单位下三角，n 个右端）；③ 解 `U·X = Y`（非单位上三角）。X = A⁻¹ 原地覆盖。
- **SgetriBatched**：逐矩阵同上，输入 Aarray[i] 为已分解因子 + PivotArray 线性排列（第 i 矩阵占 [i·n, (i+1)·n)），输出独立 Carray。
- **SmatinvBatched**：对原始 A[i] 直接求逆（getrf+getri 短路），仅 n ≤ 32；奇异时 info[i] = 首个零主元位置（语义同 getrf）。

### 支持数据类型

矩阵 FLOAT32；ipiv/PivotArray/info(Array) INT32；维数 int（32-bit 对齐 CUDA legacy API）；Lwork 为 float 元素个数。

### 支持形状

m×n 矩形（列主序，元素地址 `base + row + col·ld`）；批量接口为 Device 指针数组（Host 不解引用）；Carray/Ainv 与输入不得重叠（Host 无法校验重叠时按契约约定，检测到同址即 INVALID_VALUE）；不支持 broadcast、不要求图融合。

## 算子实现

### 实现方案

总体架构：四算子独立目录（`src/sgetrf|sgetri|sgetri_batched|smatinv_batched/`，各含 host + kernel 文件），跨算子构件（对角块因子、laswp、批量调度）收敛到共享头。**继承 9 月 Cholesky 已实机验证的四构件骨架**（对角块 AIV 分解 / 面板 / 尾部 Cube GEMM（fp16 三拆 Ozaki，LU 中 B 操作数由 L21ᵀ 换为 U12 的一般 GEMM 形态）/ 批量矩阵级分核），新增选主元与行交换两个向量构件。

平台事实约束（950PR / arch35，实测经验）与应对：

| 平台事实 | 设计应对 |
| --- | --- |
| 64 AIV + 28 AIC，MIX 单 kernel 不可用 | 向量阶段与 Cube 阶段分 kernel 同 stream 顺序下发 |
| AIC_ONLY 无向量指令 | Cube 阶段一律 Matmul 高阶 API / CATLASS |
| UB ≈192KB/core；HBM ~1.6TB/s | b=64 时面板（n×64×4B）+ 双 staging ≤ UB；批量小 n 全矩阵驻 UB |
| kernel 时长 watchdog（巨 batch ~2¹⁸ 矩阵/kernel 即杀） | 批量按 ≤32768 矩阵分块 launch；n=2×10⁶ 门槛 93 µs 下 4 块 launch ≈22 µs 可承受 |
| runtime memset 对 launch 序列无执行序保证 | info scratch 核内按核清零（每核一次对齐 MTE3 拷贝） |
| 同内容孪生位一致性（A0 契约） | 奇数尾矩阵自配对过双矩阵路径 |
| kernel 间 GM 陈旧 | 跨 kernel 读写点插 DataCacheCleanAndInvalid |
| launch ≈5.5 µs/次 | 小 n 单 kernel；n=4096 blocked ≈100 次下发 ≈0.55 ms，占预算 2~4% |
| HF32 精度不足 | Cube fp32 GEMM 一律 SetHF32(false)；fp16 三拆 hi/lo 表示误差 2⁻²² 按 SGET03 预算复核 |

#### 3.2.1 host 侧设计

**三段校验链**（首个失败即返回）：① handle 空 → NOT_INITIALIZED（不写 info）；② 标量非法（m/n/batchSize < 0、n>32（matinv）、lda(ldc/lda_inv) < max(1,·)、Lwork < 查询值（getri）、devIpiv 与选主元约定不一致）→ INVALID_VALUE + 同 stream Device 写 info=-i；③ 缓冲空指针（规模 > 0 时）→ 同上。空问题成功返回（info 可写时清零）。-i 编号开发初期对拍 cuSolver 冻结。

**tiling 策略**：`{m, n, lda, batch, path, b, coreRanges, ws 布局}` 经 GM/参数传入；路径按 shape 现场选择：

| 算子 | 路径 | 条件 |
| --- | --- | --- |
| Sgetrf | S-整矩阵单 kernel | min(m,n) ≤ 128（全有效列驻 UB；矩形时面板取 m×min(m,n) 有效区） |
| Sgetrf | L-blocked 多 kernel | min(m,n) > 128，b=64（spike 复核 128） |
| Sgetri | S-单 kernel | n ≤ 128 |
| Sgetri | L-双 trsm blocked | n > 128 |
| SgetriBatched | BV 融合向量化 / BL 逐矩阵 blocked | n ≤ 64 / n > 64 |
| SmatinvBatched | BV 融合（唯一路径） | n ≤ 32（n>32 Host 报错） |

**Lwork 契约**（32B 对齐，float 元素数）：getrf = 面板紧致 staging（m·b）+ L11 逆 staging（b²）+ 行交换双缓冲（2·m）+ Cube P tile（128×128 上界）+ 标志区（16）= `m·b + b² + 2m + 16384 + 16`；getri = 逆矩阵工作缓冲（n·ldw，ldw=n+b 对齐）+ 同类杂项。n=0/m=0 返回 0。两批量接口无 Workspace 参数（对齐 cuBLAS）。

#### 3.2.2 kernel 侧设计

**aclsolverSgetrf（blocked 右视 + 部分主元）**——L 路径每 stage k（块宽 b）四步：

| 步骤 | kernel | 核型 | 职责 |
| --- | --- | --- | --- |
| ① 面板分解 | sgetrf_panel | AIV | 对 A[k:m, k:k+b] 做 b 步列消元：每步列内 **绝对值最大（并列取最小行号）** 归约选主元 → 面板内行交换 → 消元列归一（除以主元）→ 面板尾部向量更新。产出 L11/L21/U11/U12 面板 + 本段 ipiv（全局 1-based）。主元为 0 → info=全局步号 + stopFlag，**本 stage 内停**（前 k-1 列因子与 ipiv 已定） |
| ② 行交换回放 | sgetrf_laswp | AIV×多核 | 将 ① 的行交换应用到：已算完的 L 列区（rows k:m, cols 0:k，按 32B 行段分核）+ U12 面板行 + 矩形宽尾（m<n 时 cols k+b:n 整段）。纯向量 Gather/Scatter 行重排，无原子 |
| ③ 面板规整 | sgetrf_panel_gemm | AIC | `L21 = A21·inv(U11)`（TRSM 化为 GEMM：先 ① 内联算 inv(U11) b×b）；lda≠n 时先紧致 staging |
| ④ 尾部更新 | sgetrf_update | AIC | `A22 −= L21·U12`：一般 GEMM(M=m−k−b, N=n−k−b, K=b)，fp16 三拆 Ozaki（继承已验证 trail cube，B 操作数换 U12）；beta 通路或 C-tile staging + AIV 减法回写 |

S-小 n 路径：单 AIV kernel，整矩阵驻 UB 逐列右视（选主元 + laswp 在 UB 内完成，无跨 kernel 同步），info 检测同上。**devIpiv=nullptr**：跳过选主元与 ②（不写 ipiv），消元照常（数值不稳定为语义要求）。**早退守卫**：②③④ 入口读 stopFlag，非零空转——Host 不做同步，后续 stage 照常下发但不改数据，保证 info 语义与 bit-wise 确定性。

**aclsolverSgetri（双 blocked trsm）**：

| 相 | kernel | 核型 | 职责 |
| --- | --- | --- | --- |
| ① 行重排 | getri_permute | AIV | `Y0 = P·I`：按 ipiv 正序重放，构造 n×n 单位阵的行重排（等价列 scatter：列 j 的第 i 个非零元落行 ipiv 序）——以 n 个单位列向量的置换形式写入 workspace 的 Y 缓冲 |
| ② 前代 | getri_trsm_l | AIV 对角块 + AIC 尾部 | 自左向右按 b 列分块：对角块 `Y11 ← L11⁻¹·Y11`（AIV 全寄存器 s-scale 递推，继承已验证 SolveBlock——修正项按自身尺度寄存器累加、一次折叠，对 fp32 舍入不放大）；尾部 `Y21 −= L21·Y11`（Cube GEMM，K=b，N=b 的倍数列） |
| ③ 回代 | getri_trsm_u | AIV + AIC | 自右向左镜像：`X ← U⁻¹·Y`；对角块含 **U(i,i)=0 检测** → info=i + stopFlag |
| ④ 回写 | getri_store | AIV | X 原地覆盖 A（兼容任意 lda：strided 回写） |

S-小 n 路径单 kernel（n ≤ 128 全程 UB）。精度预算：trsm 更新走 fp16 三拆 GEMM（表示误差 2⁻²²），SGET03 阈值 `max(5·ratio_cpu, 0.05)`——大 n 时 5·ratio_cpu（ratio_cpu = O(1)~O(10)）为主导界，与 Cholesky potrs 实测同型可行；开发期以全量 200 条 + 残差复核验证，超阈值时尾部 GEMM 升级 3-split+e 平面（继承 spotri 已验证方案）。

**aclsolverSgetriBatched**：矩阵级分核（batchSize 连续区间切分到 64 AIV，核间零交互；kernel 从 GM 指针数组取基址）。两档：

1. BV（n ≤ 64）：融合 kernel。每矩阵 n ≤ 64 全因子驻 UB（64×64×4 = 16KB）；核内三步全向量化：ipiv 重放（Gather 行重排）→ 前代（列消元，多矩阵可打包到同一 64-lane 向量操作，n 为 2 的幂时双矩阵交织）→ 回代；out-of-place 直写 Carray[i]。奇异检测逐矩阵 info[i]=k。巨 batch 按 watchdog 上限分块 launch（块内每核清零自己的 info 槽段）。
2. BL（n > 64）：Host 按批索引循环复用单矩阵 getri 序列（kernel 按批号取基址；该段 batch 受 4G 内存约束小，launch 次数 = batch×阶段数，门槛富余 >80×）。

**aclsolverSmatinvBatched（BV 融合，唯一路径）**：n ≤ 32 原始矩阵直接求逆。每矩阵 32×32×4 = 4KB，UB 单平面可驻 8+ 矩阵；融合 kernel 内：选主元 Gauss 消元（列内绝对值最大归约 + 行交换 + 消元，向量化列步进，n=2 的幂时双矩阵 lane 交织复用已验证 Dual 模式）→ 对右端 I 同步消元 → 回代 → 原地判 info（零主元 → info[i]=首个零主元步号）→ 直写 Ainv[i]。n>32 Host 直接 INVALID_VALUE（不转调 getriBatched）。巨 batch 分块 launch 同上。

**确定性**：主元规则确定（|v| 最大、并列最小行号）；无原子、无跨核归约、固定块内归约顺序；stopFlag 早退不改已写数据 ⇒ 同输入同 stream 串行多次执行**含 ipiv/info bit-wise 一致**。**数值控制**：Cube GEMM 关 HF32、fp32 L0C 累加；选主元比较用 fp32 绝对值向量 Max 归约（无 sqrt/div 参与选择路径）；对角块消元除以主元为 IEEE div。

**Spike 验证计划**（开发前小规模实验，结论回填归档）：

- S1 getrf 大 n（P-01/02/03）：候选 A. 四步 blocked（b=64）+ fp16 三拆尾部 GEMM；B. b=128 面板 TRSM 走 Cube；C. 纯 AIV blocked。判定门：大档最优 ≥0.21× GPU（门槛 60%）才全量开发；<0.175× 熔断复盘。附带标定 laswp 行交换在 4096 规模的实际开销（占预算预估 <8%，若 >20% 需合并 ①② 为单 kernel）。
- S2 批量（P-06..P-09 + 极端）：候选 A. 矩阵级分核 + BV 融合；B. 逐矩阵串行。扫描 watchdog 分块上限（32768 vs 65536）、双矩阵交织的 n 边界、n=2×10⁶ 的 4-launch 时序。
- S3 getri（P-04/05）：候选 A. 双 blocked trsm + Cube 尾部；B. inv(U) 显式求逆 + GEMM（n³×3 形态）；b ∈ {64,128}。精度对照 SGET03 全量 200 条。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950（Ascend 950PR，arch35/dav-3510） | √ |

CANN ≥ 9.0.0，与 ops-solver 仓 README 已验证配套版本一致；性能金标环境 CUDA cuSolver/cuBLAS。

## 算子约束限制

- 仅 FLOAT32；getrf m,n ∈ [1,8168+]、getri n ∈ [1,4096]、getriBatched n ∈ [1,4096] × batch ∈ [1,10⁶]、matinvBatched n ∈ [0,32] × batch ∈ [1,10⁷]（允许超范围声明）；
- `SmatinvBatched` n>32 必须 INVALID_VALUE，禁止静默转调；`SgetriBatched` 输出与输入不得重叠；
- `Sgetri/SgetriBatched` 输入必须是 getrf 因子 + 一致主元约定（devIpiv nullptr 须两端一致，不跨调用校验，对齐 cuSolver/LAPACK）；
- 不支持 broadcast、不要求图融合；case 构造输入内存 ≤ 4G；核心计算禁止 CPU fallback、禁止依赖 CUDA。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 两步判定：① 逐元素 vs float64 golden（NumPy/SciPy LU/inv）；② 不过时按 LAPACK 残差复核：getrf SGET01 `‖L·U−A‖₁/(n·‖A‖₁·ε)`（ipiv 逆序回放还原）判 `≤ max(5·ratio_cpu, 0.3)`；getri/getriBatched SGET03 `‖I−C·A‖₁/(n·‖A‖₁·‖C‖₁·ε)` 判 `≤ max(5·ratio_cpu, 0.05)`；matinvBatched SGET03 判 `≤ max(5·ratio_cpu, 0.5)`；ε=2⁻²⁴，batched 逐矩阵判定；info 正确性属精度（奇异 10% 构造用例必须给出正确正值下标，含 U(i,i)=0 位置） | 任务书 §3.2.1 + [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 全量 case：NPU 平均单次 kernel 耗时 ≤ bench_result.json perf.avg_ms / 0.35（950PR 实测）；msprof 采集（收敛单条用例防混叠），OpBasicInfo.csv Task Duration 按调用平均，多 kernel 按单次调用全部 kernel 合计；wall-clock 为保守上界口径（launch 开销计入） | 任务书 §3.3 |
| 确定性 | 合法输入重复执行输出含 ipiv/info bit-wise 一致 | 任务书 §2.1/§3.2 |

**测试方案**：ops-solver 测试工程（`build.sh --pkg --soc=ascend950 --ops=…`）+ 任务包用例全量回归（四包 gen/verify 脚本）+ 内置 gtest 功能矩阵：四链路（getrf；getrf+getri；getrf+getriBatched；matinvBatched）、矩形（m>n、m<n、极宽 m=64×n=4947 类）、选主元开/关（ipiv nullptr 各至少一组）、info 三态（0/i/-i 每参数至少 1 条）、批量边界（batchSize=0/1；n=1；matinv n=32 合法 / n=33 报错）、空问题、lda/ldc/lda_inv 最小值与 +8/+32 padding、确定性（含奇异用例）、非默认 stream。输入构造按任务书：50% 均匀 [-5,5] + 50% 正态后对角占优，10% 不加占优构造奇异/近奇异。**Day-1 清点全量 712 条（148+200+149+215，其中 2 条 GPU 基线缺失）写入 PROGRESS，验收只认全量数字。**

**风险与兜底**：

| 风险 | 缓解/兜底 |
| --- | --- |
| laswp 行交换引入额外数据移动（Cholesky 无此项） | 交换按 32B 行段分核纯向量重排；② 与 ① 合并为单 kernel 备选；S1 专项标定占比 |
| 矩形宽尾（m<n）更新退化 | 宽尾仅行交换不消元（数学上必要仅 laswp）；紧致 staging 兼容任意 lda |
| getri 尾部 fp16 三拆精度 vs SGET03 0.05 下限 | 大 n 由 5·ratio_cpu 主导；超限时升 3-split+e 平面（spotri 已验证方案） |
| 巨 batch watchdog / 位一致性 / memset 时序 | 继承 Cholesky 三修法：≤32768 分块、自配对尾、核内清零——已实机验证 |
| n=2×10⁶ 批量门槛 93 µs | 数据 48 MB 带宽下限 25 µs + 4 launch 22 µs，S2 时序验证；超限则提高分块上限减少 launch |
| -i 编号与 cuSolver 口径不一致 | 对拍冻结后写头文件注释 |
| msprof 口径波动 | 双口径（Task Duration 求和 + wall-clock 上界）互备 |

## 兼容性分析

仓内 sgetrf/sgetri 现存原型为 Host 指针简化版，本任务**原地升级为公开 Device 指针契约**（任务书点名不得以简化语义交付）；返回值 `aclsolverStatus_t`，如需短暂兼容现存 `aclError` 原型在头文件与文档给出对照表，最终验收以公开接口为准。`aclsolverOperation_t` 新增枚举值 0/1/2 对齐 cublasOperation_t。handle/stream 复用仓内既有管理接口。交付同步更新 README、docs/api_list.md 与 docs/zh/ 四算子文档；四接口同一 PR 或一组关联 PR 合入，不得只合入部分。
