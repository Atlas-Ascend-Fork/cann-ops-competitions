# 单精度复数LU分解、求解和批量求逆（Atlas 950）设计文档

> 作者：skytosky；日期：2026-10-18。本文为拟实施设计（设计阶段口径）：文中规模上限、资源预算与性能余量为规划输入与实测参照，不视为已通过的测试结论；实现完成、950PR 实测后同步刷新。
>
> 任务依据：10月社区任务"单精度复数LU分解、求解和批量求逆(950)"（任务页 237f144aab534448b58a123650b2fb35），任务书 `Atlas950_Cgetrf_Cgetrs_CgetriBatched_CmatinvBatched_task_doc.md`。目标仓 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，工程模式为 ops-solver Host C API + AscendC/CATLASS Kernel 直调；原型调研基线为本设计分支生成时 ops-solver master（原型现状另见 1.3）。

# 需求背景（required）

## 需求来源

在 Ascend 950PR（Atlas 950，arch35/DAV_3510）上交付稠密复数线性求解与批量求逆能力，四个计算接口加两个 workspace 查询接口作为同一任务一并交付、不得拆分验收，最终合入 cann/ops-solver：

| 交付接口 | 对标 |
| --- | --- |
| `aclsolverCgetrf` / `aclsolverCgetrf_bufferSize` | `cusolverDnCgetrf` / `cusolverDnCgetrf_bufferSize` |
| `aclsolverCgetri` / `aclsolverCgetri_bufferSize` | LAPACK `cgetri`（调用约定对齐 cuSolver DN） |
| `aclsolverCgetriBatched` | `cublasCgetriBatched` |
| `aclsolverCmatinvBatched` | `cublasCmatinvBatched`（n>32 必须报错，禁止转调 getriBatched） |

**任务书差异备案**：任务书文件名含 `Cgetrs`、部分位置出现"五个接口"及 `nrhs`/`B`/`trans` 字样，但第 2.1/2.3/7 节定义的计算接口均为 Cgetrf、Cgetri、CgetriBatched、CmatinvBatched，原型亦无 Cgetrs。本设计按四个计算接口实施，不自行增删公开接口；若维护者确认需补 Cgetrs，再按增量扩展。`aclsolverOperation_t` 仍按任务书 2.3 节要求在公共头补齐（枚举 0/1/2 对齐 `cublasOperation_t`）。

## 背景介绍与现状

复数 LU 分解（部分主元，`P·A = L·U`）是稠密复数线性代数的基础设施；LU 因子 + 主元序列可复用于单矩阵求逆（getri）与批量求逆（getriBatched），小矩阵（n≤32）另有直接批量求逆（CmatinvBatched，等价 getrf+getri 短路融合）。

ops-solver 仓内现有原型与本任务公开接口**语义不对齐**，本任务实质是接口语义重构 + 设备计算路径重写，而非改名：

| 原型现状 | 任务要求 |
| --- | --- |
| Host 矩阵指针，host 侧 H2D/D2H 搬运并同步 stream | 矩阵/ipiv/info/workspace 全 Device 指针，异步执行 |
| `lda` 强制等于 `n` | 列主序 + `lda ≥ max(1,m)` 任意 padding |
| `info` 不写回 | 真实写入 0 / −i / i（U(i,i)=0），批量逐 batch 独立写 |
| 无 `_bufferSize` 查询 | getrf/getri 各配 bufferSize，与执行端共享 workspace 规划 |
| CmatinvBatched n≥32 转调 getriBatched | 该行为必须删除；n>32 返回非法参数 |
| 连续 `[batch,n,n]` 批张量 | Device 指针数组（array of device pointers），矩阵可不连续 |

## 用例与金标数据核查（本设计的规模依据）

对任务包逐文件核查（数据来源：各算子目录 `cases.json` / `bench_result.json`）：

| 算子 | 用例数 | 关键极值（实测自用例文件） | GPU 金标覆盖 |
| --- | --- | --- | --- |
| Cgetrf | 148 | m∈[2,4096]，**n 最大 8168**（超正文 4096）；全部带选主元；无 lda>m padding 例 | 148/148 |
| Cgetri | 200 | n∈[2,4096] | 200/200 |
| CgetriBatched | 89 | n∈[2,3049]，batch∈[32,1000000] | **89 有 perf；bench 文件另含 56 条 perf=null 条目** |
| CmatinvBatched | 215 | n∈[2,32]；**batch 最大 10000000**（超正文 1e6） | 215/215 |

两点提请维护者确认（不阻塞开发）：① Cgetrf n>4096 与 CmatinvBatched batch>1e6 的用例超出正文"规模与泛化"下限，实现按"允许支持更大范围并在文档中声明"路线覆盖之（workspace/索引/batch 分块规划按实测极值设计）；② CgetriBatched 的 56 条 perf=null 条目按任务书"如 case 无 prof 数据则为非自验必跑项"处理，性能验收以 89 条有基线用例为准。

# 需求分析（required）

## 接口与功能拆解

| 接口 | 输入 | 输出 | 核心语义 |
| --- | --- | --- | --- |
| Cgetrf_bufferSize | m、n、A、lda | Host Lwork | 返回 COMPLEX64 元素数；只规划、不读 A 内容、不下发计算 kernel |
| Cgetrf | 原始 m×n 复矩阵 | 原地 LU 因子、devIpiv、devInfo | 方阵/矩形；devIpiv=nullptr 时 `A=L·U` 不选主元；1-based 主元 |
| Cgetri_bufferSize | n、A、lda | Host Lwork | 同上规划语义 |
| Cgetri | LU 因子 + devIpiv | 原地逆矩阵、devInfo | 只求逆不二次分解；Lwork 不足报错 |
| CgetriBatched | Device 指针数组 Aarray、可选 PivotArray | Carray、infoArray | out-of-place（C 与 A 不重叠）；逐 batch 独立 info |
| CmatinvBatched | Device 指针数组 A（原始矩阵） | Ainv、info | 直接求逆；0≤n≤32，n>32 报 INVALID_VALUE |

公共约束：列主序（元素地址 `base + row + col·ld`），仅访问有效区域、不踩 padding；所有维数为 `int`（32 位，对齐 CUDA legacy API）；矩阵、workspace、ipiv、info、指针数组均在 Device，Host 侧只有标量维数、handle 与 Lwork 查询输出；复用 `aclsolverCreate/Destroy/SetStream/GetStream`，计算全部下发到调用方 stream，禁止 CPU fallback；空问题（m/n=0、batchSize=0）成功返回且不访问矩阵与 workspace。

确定性要求（任务书 2.1.7）：相同输入、相同 stream 串行多次执行，输出含 ipiv/info **bit-wise 一致**——主元归约用固定分组固定合并序，浮点累加无原子竞争，kernel 调度静态确定。

## 规模与边界

| 接口 | 声明支持范围 | 说明 |
| --- | --- | --- |
| Cgetrf | m,n ∈ [1, 8192]（矩形覆盖 m≠n） | 覆盖用例实测 n=8168 并留裕量；官方声明下限 4096 |
| Cgetri | n ∈ [1, 4096] | 对齐用例 |
| CgetriBatched | n ∈ [1, 4096]，batch ∈ [1, 1000000] | 对齐正文；分块调度不物化全量中间阵 |
| CmatinvBatched | n ∈ [1, 32]，batch ∈ [1, 10000000] | n=33 报错；batch 上限按实测用例放宽声明 |

自构造用例输入总内存控制在任务书规定的 4 GB 内；索引与偏移一律先宽化到 64 位再运算，跨 int 边界的 `(b·n)`、`(ld·n)` 乘积做溢出检查。

# 详细设计（required）

## 公开类型与 ABI

`cann_ops_solver_common.h` 补充 C 可调用类型（布局与 `cuComplex` 一致，8 字节）：

```c
typedef struct { float real; float imag; } aclFloatComplex;

typedef enum {
    ACLSOLVER_OP_N = 0,
    ACLSOLVER_OP_T = 1,
    ACLSOLVER_OP_C = 2
} aclsolverOperation_t;
```

六个公开函数声明与任务书 2.3 节逐字一致（参数名、顺序、Device/Host 归属不变，`int` 不改 `int64_t`）。新增类型与接口同时支持 C/C++ 调用（C++ 侧提供与 `std::complex<float>` 互转的内联辅助，公开头保持 C 可调用）。旧不兼容原型（Host 指针版 Cgetrf/Cgetri）不与新 C ABI 同时导出：调用点、测试与文档随本任务一次性迁移，不用重载掩盖 ABI 变化。返回值统一 `aclsolverStatus_t`（仓内已有 12 状态码；内存失败与执行失败分别映射 `ALLOC_FAILED`/`INTERNAL_ERROR`，不混同为参数非法）。

## 数学设计

### LU 分解（P·A = L·U）

令 k=min(m,n)。第 j 步在行 [j, m) 内选主元，规则对齐 LAPACK `CGETF2`/`ICAMAX`：比较 `|Re| + |Im|`，相等取行号最小者——**不采用复数欧氏模**，二者可能选出不同主元，验收以任务书残差判据为准，不假设与任何后端逐位一致。主元写 1-based 行号；`devIpiv == nullptr` 完全跳过选主元与换行。

幅值比较用防溢出的缩放表示；复数除法用缩放分支（先除以 |Re|、|Im| 中较大者再算 `c²+d²`），不对大幅值分量无条件平方。零主元按实部虚部同时为零判定，不用人为阈值把近奇异判成奇异；零主元写入 devInfo（1-based 列号）后停止除以该主元，可定义的剩余步骤按语义完成。

### LU 因子求逆（getri）

由 `P·A = L·U` 得 `A⁻¹ = U⁻¹·L⁻¹·P`，实现分三段独立 kernel：

1. **dinv(U)**：U 上三角原地求逆，自下而上按块列推进（对角块求逆 + 块列回代），块列间无依赖可并行；
2. **L 多右端求解**：`X = L⁻¹·X`，单位下三角，自上而下按块行推进；注意单位对角从约定读取，不读 packed LU 中实际存放 U 对角线的位置；
3. **置换重排**：以整数行映射 p[i]=i 出发、按 j 递增交换 p[j] 与 p[ipiv[j]−1] 构造列置换索引，输出按映射重排，**不物化稠密置换矩阵**。

与 LAPACK `CGETRI` 的 DTRTRI + DLASWP + DTRSM 序列数学等价；用不少于两次交换同一对行的主元序列验证置换次序（单次交换用例无法证明顺序正确）。无主元（ipiv=nullptr）时跳过第 3 段。求逆前扫描 U 对角线记录最小零对角索引，失败矩阵跳过求逆并在 devInfo 报告。

### 批量求逆（getriBatched）与小矩阵直接求逆（CmatinvBatched）

getriBatched 输入是**已分解的 LU 因子**（可由循环调用 Cgetrf 构造，本任务不交付 CgetrfBatched），逐矩阵套用上述三段；C 从只读 LU 出发独立生成，不覆盖输入。

CmatinvBatched 对原始矩阵做 getrf+getri 短路融合（n≤32）：片上完成带部分主元的消元与两次三角求解；输出不覆盖输入；首个零主元写入对应 info。n>32 在 Host 直接拒绝，不转调。

## Host 设计

### 校验、错误码与 info 异步语义

校验顺序：handle（空 → `HANDLE_IS_NULLPTR`）→ Host 标量（负维数、lda/ldc/lda_inv 下限、Lwork 指针、batchSize<0 → `INVALID_VALUE`）→ 顶层指针非空检查 → 取 stream 下发。空问题先行短路，避免 n=0 时计算 `(n−1)·ld` 类偏移。

**info 写入走设备侧**：非法参数（−i，i 为不含 handle 的参数序号）与计算失败（i）均由轻量 device kernel 写入，Host 不解引用 Device 指针；返回成功仅表示异步下发成功。参数编号：Cgetrf m=1/n=2/A=3/lda=4/Workspace=5/devIpiv=6/devInfo=7；Cgetri n=1/A=2/lda=3/devIpiv=4/Workspace=5/Lwork=6/devInfo=7；CgetriBatched n=1/Aarray=2/lda=3/PivotArray=4/Carray=5/ldc=6/infoArray=7/batchSize=8；CmatinvBatched n=1/A=2/lda=3/Ainv=4/lda_inv=5/info=6/batchSize=7。批量负值口径与"依赖 Device 内容的错误是否要求返回前可见"属实现前需与维护者对齐项：本设计取"Host 可判定错误走返回值并同步写 info；Device 数据错误写 info、不引入隐藏同步"，文档与测试显式记录该行为。

错误检查分层：Host 只查标量、顶层指针、Lwork 与偏移算术溢出；Device 预检 kernel 可读指针数组元素，检查空元素、可判定的输入输出重叠与主元值域，失败矩阵不进入计算；不以试读任意地址的方式"验证指针有效性"，也不声称仅凭 A[i]/C[i] 检查能覆盖所有跨 batch 重叠。

### workspace 规划（bufferSize 与执行端共享）

规划按字节、返回按 COMPLEX64 元素数（`Lwork = ceil(totalBytes/8)`），全部用带溢出检查的 64 位无符号算术；两端口调用同一规划函数，避免查询值与执行需求漂移：

| 路径 | 规划项（初版预算） |
| --- | --- |
| Cgetrf | 三角面实虚分离暂存 ≤ `8·(m·b + b·n)` 字节（b=面板宽）；主元归约记录；GEMM tile 临时区（按 tiling 查询） |
| Cgetri | 完整输出 X `8·n²` 字节 + 行映射 `4·n` + 三角求解 tile 临时区（不同时保存第二份 LU） |
| CgetriBatched | 复用用户 C 作右端/结果，内部仅有界 chunk 的行映射、状态与 tile 暂存 |
| CmatinvBatched | 优先片上（UB），无 Device workspace；分阶段后备路径按 chunk 分配 |

生命周期不重叠的区域才可复用；首版不做跨阶段双缓冲，若后续引入流水重叠同步扩大查询值。getrf 查询无主元模式参数，统一按可选主元路径预留。getri 校验调用方 Lwork 不小于同参查询值，不足报错。batch 接口无用户 workspace：小矩阵走片上路径，大 batch 以有界 chunk 切分调度范围（chunk 只切调度、不省略矩阵、不把指针数组整批搬到 Host）。

### 依赖编排与 tiling

kernel 下发序列按"预检 → 面板分解 → 面板外换行 → 块行三角求解 → 尾随更新"循环，全部排同一 stream，无全局自旋屏障；每个输出 tile 唯一归属一个 kernel 实例，避免原子累加次序不确定性。tiling 传递 m/n、lda/ldc、batch 范围、面板起点与有效宽度、workspace 偏移；核数与片上容量经平台查询获得，不硬编码。尾块按有效行列掩码访问，padding 不参与计算与归约。

## Kernel 设计

### 总体：双模态 + 复数拆实

复数 GEMM 一律展开为四个 FP32 实数积（`Re(AB)=ArBr−AiBi`，`Im(AB)=ArBi+AiBr`），实虚分离布局进 Cube（arch35 有 FP32 MMAD；AoS 复数不能直接进 Cube），分离/回写开销计入算子路径；精度合规必须 FP32，不引入半精度截断。依我们同平台 GEMM 微基准实测，arch35 严格 FP32 Cube 参考实现口径峰值 ≈ 20.3 TFLOPS。

小矩阵与大矩阵分界阈值、面板宽度 b 由 950PR 单变量实验标定（候选 b=32，与 fp32 baseK=8 及 UB 布局对齐），设计文档不预设最优值。

### Cgetrf：分块右视（大）/ 单 kernel SIMT（小）

**小路径**（规模低于阈值，小形状门禁 37~63µs 级决胜）：单个 kernel 内完成逐列选主元与消元，SIMT 多线程按行分块，固定归约树选主元；目标是把 launch 链压到 1~2 个 kernel。

**大路径**（分块右视，五阶段循环）：

1. **面板分解**（AIV）：处理 [j, m) 全高面板，逐列选主元（记录全局行号）、乘子计算与面板内更新；主元归约按固定线程分组两极合并；
2. **换行**（laswp，AIV/AIC 轻量）：按面板内主元产生顺序，将交换施加到先前已产出的 L 列与右侧未处理列；面板内已执行的交换不重复执行；
3. **块行三角求解**：`U12 = L11⁻¹·A12`，沿 U12 列块并行；
4. **尾随更新**（Cube FP32）：`A22 −= L21·U12`，复数拆四实数积，M/N/K 三维 tile 分核；
5. 推进至 k=min(m,n)，矩形（m<n）时 U 的横向阶梯与最后梯形块由列掩码处理。

零主元后该列跳过消元并置 info；不同 batch（循环调用场景）的 info 互不覆盖。

### Cgetri / CgetriBatched：分段并行

小矩阵将一个（或若干个）完整矩阵分到一个核，片上完成三段；大矩阵沿块列/块行并行——dinv(U) 自下而上块列间并行，L 求解自上而下块行串行推进、块行内列并行，置换重排按列块并行。批量 kernel 直接从 GM 读取 `Aarray[b]`/`Carray[b]` 指针值（Device 指针数组不在 Host 展开），PivotArray 偏移 `b·n`，不假定矩阵连续。

### CmatinvBatched：融合片上路径

n≤32 单矩阵 LU+X 共 16·n² ≤ 16 KB，192 KB UB 扣除临时区后一核可保有多个矩阵；消元为复数标量乘减全行向量化（复数标量乘复数行 = 四组实数向量乘加），主元选取与换行片上完成。batch 采用静态确定性分配（核 c 处理 c、c+C、c+2C…号矩阵，每矩阵唯一拥有者）；超大 batch 由 Host 侧有界 chunk 循环覆盖到 1e7，chunk 边界不重不漏。若资源约束导致融合路径不可行，回退"有界 Device 暂存的分阶段路径"，不拒绝合法 n≤32 用例。

## 工程集成

修改范围：`include/cann_ops_solver*.h`、`src/cgetrf/`、`src/cgetri/`、`src/cgetri_batched/`、`src/cmatinv_batched/`、对应 `test/` 四目录、`docs/zh/` 四篇、`README` 与接口列表。复数分离/合成、TRSM、laswp 等公共辅助放现有公共目录，不复制多个高重复大文件；保留上游许可证头，遵循 ops-solver CONTRIBUTING（无冗余文件、kernel 结构清晰）。目标硬件仅声明 Ascend 950PR（arch35 编译配置 `--soc=ascend950`）；其他平台保持既有构建行为，新增同名符号不产生重复链接。开发分支独立 clone，不与其他任务共用工作区。

# 可维可测分析（required）

## 精度标准

按任务书 3.2.1 执行 LAPACK 残差判据（ε=2⁻²⁴，ratio_cpu 来自同 case 的 COMPLEX64 CPU 参考链路；complex128 仅用于残差求值与诊断，不替代同精度 ratio_cpu）：

| 接口 | 判据（溯源） | NPU 通过条件 |
| --- | --- | --- |
| Cgetrf | CGET01：ipiv 逆序回放行交换后 `‖LU−A‖₁/(n·‖A‖₁·ε)` | ≤ max(5·ratio_cpu, 0.5)；矩形分母 n 取列数 |
| Cgetri | CGET03：`‖I−C·A‖₁/(n·‖A‖₁·‖C‖₁·ε)`，A 为 LU 因子按 ipiv 还原 | ≤ max(5·ratio_cpu, 0.05) |
| CgetriBatched | CGET03 逐矩阵 | 每矩阵 ≤ max(5·ratio_cpu_i, 0.05)，任一超限该用例不通过 |
| CmatinvBatched | CGET03 逐矩阵（A 为原始输入） | 每矩阵 ≤ max(5·ratio_cpu_i, 0.5) |

非有限结果单独判失败；零分母与奇异矩阵走明确分支，不依赖 NaN 比较得出 PASS。CPU 基线脚本随测试代码交付（固定种子，可复现）。

## 性能标准与测量设计

门禁：每 case `T_NPU ≤ T_GPU/0.35`（GPU 金标取各目录 `bench_result.json` 的 `perf.avg_ms`；Cgetri 金标为 GPU 上 getrf+AX=I 全程，其分解时间不计入本接口耗时）。本设计预算测算（按用例 FLOPS 估算 ÷ 金标预算）：Cgetrf 最紧 case（3692×7444）约需 3.8 TFLOPS 有效算力，Cgetri n=4096 口径约 3.7 TFLOPS——相对 arch35 FP32 Cube 实测峰值 ~20.3 TFLOPS 余量约 5 倍，**性能门禁主要风险不在算力而在小形状固定开销**（同平台经验：多 kernel launch 链在 37~63µs 门禁下即告负、窄形状 Cube 利用率塌方），故小路径单 kernel 化与面板/尾随形状的 tiling 对策列为 M3 首要验证项。

测量：950PR 上 msprof op 采集 `OpBasicInfo.csv` 的 Task Duration（us），按 kernel 名求平均单次耗时；采集收敛到单条用例避免混叠；原地接口每轮重置输入；多 kernel 实现同时报告各 kernel 均值与整接口设备计算总和，不以最快单 kernel 代替整接口；记录 CANN/驱动版本、NPU 型号与候选提交 hash，profile 原始证据随自测报告归档。

## 覆盖规划

覆盖任务包全部官方用例（148+200+89+215，含 perf=null 的非必跑条目仅跑精度），另自构造：矩形与 2 的幂±1 维数、lda/ldc/lda_inv 的 +8/+32 padding、devIpiv/PivotArray 为空（无主元）、奇异/近奇异、非法维数与非法 lda（info=−i、返回值）、空问题（m/n=0、batchSize=0）、n=32/33 边界、batch=1 与 chunk 边界、非默认 stream、Device 指针数组非连续布局。确定性：同输入同 stream 串行 ≥2 次，输出有效元素 + ipiv + info 逐位比对。回归映射：

| 设计约束 | 回归方法 |
| --- | --- |
| C ABI 与列主序 | 独立 C 最小调用工程；编译期 static_assert 复数大小/偏移；非对称复矩阵排除隐含转置 |
| 主元与置换次序 | 等幅取最小行号、两次不对易交换、无主元路径、首个零主元、近奇异非零主元 |
| 残差链路 | 保留原矩阵与种子；ipiv 逆序回放；区分原始矩阵与 packed LU |
| workspace 契约 | 查询/执行同源；Lwork 少一元素报错；连续调用复用 |
| 设备数组与大 batch | 分散分配、重叠检测用例、非法主元、chunk 前后边界矩阵抽查 |
| 确定性 | 重复执行逐位比对（含 ipiv/info） |

执行顺序：Host 单测（参数编号/溢出/空问题）→ 小矩阵正确性 → 矩形/padding/分块边界 → 全量官方用例 → 性能实验（正确性候选冻结后再调优）。

## 兼容性与风险

旧接口升级属有意 ABI 迁移：公开头、示例、测试、文档同步；既有非本任务算子语义不动；不引入 CUDA 依赖。主要风险与对策：

| 风险 | 对策 |
| --- | --- |
| 大矩阵面板串行成为瓶颈 | 面板内归约分组并行 + 面板宽 b 标定；预算余量 5× 下优先保正确性 |
| 小形状 launch 链超门禁 | 小路径单 kernel 化列为 M3 首验项（同平台实证教训） |
| n=8168 等超正文用例 | 索引 64 位化 + 分块覆盖；范围声明放宽并备案 |
| 异步 info 错误口径 | 已在 3.3 节给出建议口径，评审对齐后冻结，不验收时隐去 |
| Cgetrs 任务书差异 | 备案待确认（1.1 节），不自行扩接口 |
