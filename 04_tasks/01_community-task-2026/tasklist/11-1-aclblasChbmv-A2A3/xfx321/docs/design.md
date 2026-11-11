# aclblasChbmv（Atlas A2/A3）算子设计文档

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务「aclblasChbmv 算子开发（A2A3）」（任务书：`aclblasChbmv_AtlasA2_A3_task_doc.md`）。任务要求在 Atlas A2/A3 系列产品上使用 Ascend C 开发单精度复数 Hermitian 带状矩阵-向量乘算子 `aclblasChbmv`，补齐 ops-blas 开源仓 Blas 能力，对标 cuBLAS `cublasChbmv`（语义参考 Netlib `chbmv`）。

## 背景介绍

### aclblasChbmv 算子功能

计算 `y = alpha * A * x + beta * y`。A 为 n×n Hermitian 带状矩阵（k 条次对角线 + k 条超对角线），按列主序带状格式存储于 lda×n 数组（实际仅 (k+1)×n 有效），仅 `uplo` 指定的带形三角被引用，另一侧由 Hermitian 共轭对称性隐含补全；alpha、beta、A、x、y 均为单精度复数（complex64）。

ops-blas 仓现状：

- 实数同族算子 `aclblasSsbmv` 已在 `blas/sbmv/arch22/` 落地，采用「行块分解 + 每存储元素双用」的零 workspace、无原子架构，带状存储约定、分核成本模型、strided 向量处理均已验证，是本算子最直接的工程与算法参考；
- 复数向量算子 `aclblasCsymv`（`blas/symv/arch22/cgemv_*`）已给出复数在 UB 内平面化（real/imag 分离缓冲）运算的仓内先例；
- `aclblasChbmv` 为新增接口，`include/cann_ops_blas.h` 中尚无声明，需按任务书 §2.3 新增。

# 需求分析（required）

## 需求描述

使用 Ascend C 以 kernel 直调方式实现 `aclblasChbmv`：句柄式 BLAS 接口（handle 绑定 stream），complex64 输入输出，支持 UPPER/LOWER 两种带形三角、任意非零步长 incx/incy、beta=(0,0) 时 y 旧值不必有效、n=0 合法 quick return、对角元素仅实部参与计算。接口声明放入 `include/cann_ops_blas.h` 与其他产品线共用，实现代码放 `blas/sbmv/arch22/`。精度按 FLOAT32 生态标准（rtol=atol=2⁻¹³，matched_ratio≥0.99），性能在 Atlas 800T A2 (910B3) 上达到 GPU 标杆的 0.8 倍以上（单 case 耗时 ≤ GPU×1.25）。

## 需求拆解

1. **接口与类型**：复数类型复用仓内 `cann_ops_blas_common.h` 已有的 `aclblasComplex`（real/imag 各 float32，8 字节交错），不新增类型；在 `include/cann_ops_blas.h` 的 `aclblasSsbmv` 声明之后新增 `aclblasChbmv` 签名（维数参数 int，与 Ssbmv 口径一致）。
2. **Host 侧**：按任务书 §2.5 拍板顺序做参数校验（handle 判空最优先、n=0 quick return 不得先于判空）；lda ≥ k+1 与步长溢出校验在 int64 域进行；k > n-1 按有效带宽 min(k, n-1) 钳制；非法 uplo 返回 `ACLBLAS_STATUS_INVALID_VALUE`（§7.5 与 Csymv 的 INVALID_ENUM 差异已由研发确认按本任务书执行）。
3. **Kernel 侧**：在 Ssbmv 行块分解架构上扩展 Hermitian 语义——镜像引用从转置（Aᵀ）改为共轭转置（Aᴴ），对角元素单独成 pass 且仅取实部；复数运算在 UB 内平面化（real/imag 分离），共轭退化为虚部平面取负。
4. **正确性细节**：beta=(0,0) 时必须跳过 y 旧值读取与乘法（y 可能含 NaN/Inf，0×NaN≠0）；带状数组中不对应矩阵元素的区域（UPPER 左上 k×k、LOWER 右下 k×k）不得引用。
5. **测试工程**：按 `test/sbmv/ssbmv` 结构搭建 `test/sbmv/chbmv/arch22`，加载随包 1200 条 CSV 用例；golden 采用随包 cblas_chbmv（fp64 真值、fp32 收窄）；自行补充 GAUSS_NORM_5_2 正态分布用例；含白盒不变式测试。
6. **性能**：msprof op 模式采集，5 次 warmup + ≥10 次有效采样取平均，196 条性能用例逐一对照 `gpu_baseline.csv` 判定 ≤1.25×。

# 详细设计（required）

## 算子分析

### 数学公式

```
y = alpha · A · x + beta · y
```

- A 为 n×n Hermitian 带状矩阵：A(i,j) = conj(A(j,i))，|i-j| ≤ k 以外元素为 0；
- 复数乘：`(a_r + a_i·i)(b_r + b_i·i) = (a_r·b_r − a_i·b_i) + (a_r·b_i + a_i·b_r)·i`；
- Hermitian 对角元素虚部假定为 0，对角项 A(i,i)·x(i) 仅实部参与：`acc_r[i] += A_r·x_r[i]`，`acc_i[i] += A_r·x_i[i]`；
- 输出合成：`y_out = alpha · acc + beta · y_old`（alpha=(0,0) 时按 `y = beta·y` 计算且不保证跳过 A/x 读取；beta=(0,0) 时跳过 y_old 项）。

### 带状存储与 Hermitian 约束

列主序带状存储（与仓内 Ssbmv 内核注释一致的记法，已对照 Netlib/cblas 参考验证）：

| uplo | 存储映射 | 对角位置 | 未引用区域 |
|---|---|---|---|
| UPPER | A(t, j) = M(j−k+t, j)，t = 0..k | t = k（第 k+1 行） | 数组左上 k×k 三角 |
| LOWER | A(t, j) = M(j+t, j)，t = 0..k | t = 0（第 1 行） | 数组右下 k×k 三角 |

关键差异（相对实数 Ssbmv）：镜像引用必须取**共轭**。以 LOWER 为例，存储元素 A(t,j)=M(j+t,j) 有两次引用：

1. **本列散布**：y[j+t] += A(t,j) · x[j]（固定列 j，A 段与 y 段在 GM 中均连续）；
2. **镜像引用**：y[j] += conj(A(t,j)) · x[j+t]（固定 t，跨列 gather，stride=lda；x 侧连续切片）。

UPPER 对称地成立（散布 y[j−k+t] += A(t,j)·x[j]，镜像 y[j] += conj(A(t,j))·x[j−k+t]）。对角元素（LOWER t=0 / UPPER t=k）自共轭，不能计入双用（会重复累加），单独走对角 pass。

### 公开接口原型

```cpp
// include/cann_ops_blas.h，紧随 aclblasSsbmv 声明之后新增
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

### 支持数据类型与形状

| 项 | 值 |
|---|---|
| dtype | COMPLEX64（8 字节，real/imag 两个 float32 交错，复用仓内 `aclblasComplex`） |
| A | 带状数组逻辑形状 (k+1)×n，物理 lda×n 列主序，只读 |
| x / y | 至少 1+(n−1)·|incx| / 1+(n−1)·|incy| 个元素；y 原地覆写 |
| n / k | n ≥ 0（n=0 合法 quick return）；k ≥ 0，k>n−1 时有效带宽钳制为 min(k, n−1) |
| lda | ≥ k+1（int64 域比较） |
| incx / incy | ≠ 0 且 ≠ INT32_MIN，可负 |

### 参数与属性

校验顺序（任务书 §2.5 #765/#861 口径）：

```
handle 判空 → n < 0? → n == 0 直接返回 SUCCESS → uplo 枚举 → k < 0?
→ lda < k+1（int64）→ incx/incy == 0 或 INT32_MIN → (n−1)·|incx|、(n−1)·|incy|
int64 溢出 → alpha/beta/A/x/y 判空（n > 0 时）
```

| 非法输入 | 返回码 |
|---|---|
| handle == nullptr | ACLBLAS_STATUS_HANDLE_IS_NULLPTR (9) |
| n < 0、k < 0、lda < k+1、incx/incy ∈ {0, INT32_MIN}、步长跨度 int32 溢出、uplo 非法、n>0 时任一指针为空 | ACLBLAS_STATUS_INVALID_VALUE (3) |
| n == 0（通过校验后） | ACLBLAS_STATUS_SUCCESS，不执行计算 |

注：非法 uplo 按 `INVALID_VALUE` 返回（任务书 §7.5 已与研发确认；仓内 `aclblasCsymv` 当前返回 `INVALID_ENUM`，差异由研发后续统一，本算子不跟随）。

## 算子实现

### 总体方案

沿用 Ssbmv 已验证的**行块分解、每存储元素双用**架构，零 workspace、无原子操作、单 kernel 启动（tiling 传值），扩展三点：

1. 镜像引用由转置改为**共轭转置**（平面布局下仅虚部取负，无额外开销）；
2. 全部向量运算在 **UB 平面域**（real/imag 分离缓冲）进行，GM 侧保持 8 字节交错布局；
3. 对角独立成 pass（仅实部乘法），避免双用重复累加。

行块分解：核 b 拥有行区间 [r0, r1)，r1 = min(n, r0 + blockRows)。每个存储元素 A(t,j) 的两次引用都由其落点行所归属的核完成——本列散布落 y[j−k+t]（或 y[j+t]），镜像落 y[j]——因此每行恰好被一个核写入一次，无需跨核同步。A 的读取量至多两倍于存储量。

### Host 侧设计

- 校验顺序如上表；全部维数/跨度比较在 int64 域（防 INT32_MAX 回绕穿透，issue #780/#866 口径）；
- `n=0` 在判空与枚举校验之前直接返回 SUCCESS（quick return 不先于 handle 判空）；`beta=(0,0)` 不豁免 y 非空校验；
- k > n−1 时 host 侧将有效带宽 `kb = min(k, n−1)` 写入 tiling，kernel 仅引用 kb；
- 分核数：`numBlocks = min(aivCoreNum, cap)`，cap = ceil(n / 64)，并按 Ssbmv 的带感知成本模型 `CalSsbmvDiagCoreNum` 复算（块固定开销 + (2kb+1)·ceil(rowsPerCore/64)·迭代成本 + 总字节/带宽(c)，逐 c 取最优）——复数版系数按 8 字节元素重标定；
- workspace：常规路径为 0（行块私有累加在 UB 内完成）；仅当 x/y 物理区间重叠且超出 handle 自带 workspace 时按 Ssbmv 同款策略增长或报错；
- 一次 kernel 启动，`ChbmvTilingData` 按值传入（n、kb、lda、incx、incy、numBlocks、行块划分、uplo、betaIsZero 标志等，int32 域，结构体 <256B）。

### Kernel 侧设计

`ChbmvKernel` 类，`Init()` 读 tiling 与 GM 指针，`Process()` 按行块执行三 pass：

```
Per row-block [r0, r1)（UB 内常驻 acc_r/acc_i 累加器）:
  ① 列散布 pass：对本块覆盖的每个存储列 j，
     DataCopyPad 取 A 带段（连续）→ 平面化 → x[j] 标量广播复数乘
     → 移位对齐累加进 acc（DataCopyPad leftPadding 保持 32B 对齐基址）
  ② 镜像 pass：对固定 t（0..kb），GatherMask 跨列取 A(t, j) 段（stride=lda）
     → 平面化、虚部取负（conj）→ 与连续 x 切片平面复数乘累加进 acc
  ③ 对角 pass：取对角元素实部 → 实标量 × 复数向量（2 次向量乘）累加
  尾处理：acc ← alpha·acc（alpha=(0,0) 即清零）；beta≠0 时 acc ← acc + beta·y_old
  → 交织回交错布局 → Scatter/DataCopyPad 写回 y[r0..r1)
```

- **strided x**（|incx|≠1）：初始化一次性构建 `BuildOffsetTable` 偏移表（strideBytes=incx·8），镜像 pass 的 x 切片用 GatherMask 间接寻址；负步长按物理位置映射 `phys = base + (n−1−logical)·stride`（Ssbmv `XPhysicalPos` 同款），逻辑顺序遍历不变；
- **strided y**（|incy|≠1）：写回走 staging（Ssbmv `SSBMV_Y_STAGE_FLOATS` 同款，按复数元素数减半），再按步长散布写回；负 incy 反向写；
- **宽带分片**：kb 大于 UB 单段容量时（镜像 A 段 stride=lda 连续列 gather 段长 (kb+8) 复数），按 `MIRROR_TILE`（复数元素 4096）分片多次 pass 归约，Ssbmv 同款；
- **beta=(0,0)**：tiling 携带 `betaIsZero` 标志，kernel 跳过 y_old 装载与 `beta·y` 乘加（IEEE 下 0×NaN=NaN，必须整项跳过而非乘零）；
- **未引用区域**：所有访存均由 (t, j) 有效索引生成，UPPER 左上 / LOWER 右下三角天然不被寻址（白盒用例以 Inf/NaN 填充验证）。

### 复数运算的实现要点（UB 平面化）

GM 交错 → UB 平面：`GatherMask(dst_r, src, offTable_even, 0, cnt)`、`GatherMask(dst_i, src, offTable_odd, 4, cnt)`（偶数 float 下标为实部、+4 字节基址取虚部；偏移表整块共享、init 时一次构建，仓内 `gtsv2_strided_batch` / `cgemv` 已验证该用法）。

平面域核心运算（每 64 元素向量批）：

| 运算 | 平面公式 | 向量指令 |
|---|---|---|
| 复数乘累加（散布） | acc_r += A_r·x_r − A_i·x_i；acc_i += A_r·x_i + A_i·x_r | 4 Mul + 1 Add + 1 Sub + 2 Add |
| 共轭乘累加（镜像） | acc_r += A_r·x_r + A_i·x_i；acc_i += A_r·x_i − A_i·x_r | 同上（交叉项符号翻转） |
| 共轭准备 | conj(A) = (A_r, −A_i) | 平面化后 0 条指令（仅镜像乘法符号选择） |
| 对角项 | acc += A_r·x（实标量×复向量） | 2 MulBroadcast |
| 标量合成 | acc ← alpha·acc + beta·y_old | 广播复数乘（cgemv `mulSComplex` 先例） |

写回：平面 → 交错（GatherMask 反向或模式写回），保持 GM 侧与输入相同的 8 字节交错布局。

### Tiling 与分核策略

- 每核行块下限 64 行（一个向量宽度），上限 `CHUNK_ROWS_CAP` 分块防 UB 溢出（Ssbmv 同款）；
- 分核数由带感知成本模型决定：窄带大 n 时最优核数远小于「每核 64 行」经验值（模型自动收敛）；总带宽封顶 2.367TB/s、单核 74GB/s，复数版每元素字节数 ×2 后重标定；
- n 小于 64 的用例单核覆盖；n=1/k=0 退化为标量路径复用同一 pass 结构（散布+对角合并）。

UB 预算（arch22，192KB 可用）：

| 缓冲 | 规模 | 说明 |
|---|---|---|
| acc_r / acc_i | 2 × chunkRows × 4B | 行块累加器 |
| y_old_r / y_old_i | 2 × chunkRows × 4B | beta 项（betaIsZero 时不装载） |
| A 带段平面 | 2 × 8192 × 4B（每 plane） | MIRROR_TILE 分片上限 |
| x 窗口平面 | 2 × 8192 × 4B | 镜像切片 |
| 偏移表 ×2 | 2 × 8192 × 4B | GatherMask 共享 |
| staging | 8192 × 4B | 仅 strided y 分配 |
| 对齐余量 | 16 floats × 各缓冲 | DataCopyPad 尾块 |

宽帶极端（kb≈2047）：A/x 段按 4096 复数元素分片后峰值 UB ≈ 160KB，满足 192KB 约束；普通带宽（k≤512）全程单分片。

### 确定性设计

任务书不强制确定性，但本设计天然确定：每行仅归属一个核、核内累加顺序固定（列散布按列升序 → 镜像按 t 升序 → 对角收尾 → alpha/beta 合成），同输入复跑位级一致（纳入不变式测试）。

### 异常处理

- 全部校验在 host 侧同步完成，非法入参不发起 kernel 启动；
- 特殊值行为：A/x 含 Inf/NaN 时按 IEEE 传播（累加路径自然产生）；golden 与 NPU 输出同为 NaN、同号 Inf 或 NaN/±Inf 互现按等效判定（§3.2 口径）；Inf/NaN 用例不与 alpha=(0,0) 组合（任务书 §2.1-4，随包用例 TC_FL_209/210/212/213 均为 alpha=(1,0) 组合，一致）；
- n=1：k 被钳制为 0，仅对角 pass 生效；
- x/y 物理重叠（原地调用）：按 Ssbmv 策略经 handle workspace 中转或显式报错，不做隐式假设。

### 性能预算与优化方案

带宽主导型算子。A 读取量 ≤ 2× 存储量（双用），最坏性能用例（chbmv-perf-119，n=4076，k=929，lda=930）：

- A 存储 930×4076×8B ≈ 30.3MB，双用读 ≈ 60.7MB；x/y 合计 ≈ 130KB；
- HBM 带宽按 1.6TB/s、访存效率 70% 估算：≈ 54μs；
- 向量算力：7.58M 存储复数元素 × 16 向量指令 / 64 lane ÷ 20 AIV ÷ 1.8GHz ≈ 48μs，与访存重叠；
- GPU 标杆 698.7μs，预算上限 873μs（1.25×）→ 预估裕量 >10×。

代表性 case 预算表（GPU 基线 ×1.25 为判据线）：

| case | uplo | n | k | GPU (μs) | 判据线 (μs) | NPU 粗估 (μs) |
|---|---|---|---|---|---|---|
| chbmv-perf-119 | UPPER | 4076 | 929 | 698.7 | 873.3 | ~50 |
| chbmv-perf-081 | LOWER | 2048 | 2047 | 443.0 | 553.7 | ~55 |
| chbmv-perf-080 | LOWER | 2048 | 1024 | 400.4 | 500.5 | ~35 |
| chbmv-base-003 | LOWER | 4096 | 64 | 53.0 | 66.3 | ~12（小带宽启动开销主导） |
| 基线中位 case | — | — | — | 159.2 | 199.0 | ~15–30 |

后备优化（实测不达标时逐级启用）：
1. 分核成本模型系数用 msprof 实测回归（Ssbmv 上线时同流程）；
2. 小 n/k 用例（<20μs 量级）减少 GatherMask 表构建开销，x/y 连续时退化为纯 DataCopy 路径；
3. 双用合并读：镜像 pass 复用散布 pass 已装载的列段缓存（k 窄、行块内列全被覆盖时省一次 A 读取）。

### 支持硬件

- Atlas 800T A2（910B3，arch22）：精度 + 性能验收设备；
- Atlas A3 系列：精度自验覆盖（kernel 直调框架同族适配，AIV 数量等平台参数经 `platform_ascendc` 运行时获取，无硬编码）。

### 算子约束限制

- 不支持超出 lda/incx/incy 语义的非连续内存访问；
- 对角元素虚部不读取（垃圾值不影响结果，白盒验证）；
- alpha=(0,0) 不做短路（含 Inf/NaN 的 x 用例不得与 alpha=(0,0) 组合，随包用例已遵守）；
- 无 broadcast、无 dynamic shape 要求；y 原地覆写不返回视图；
- 确定性不强制（设计天然满足）。

### 代码落点与构建接入

```
ops-blas/
├── include/cann_ops_blas.h                  # 新增 aclblasChbmv 声明（Ssbmv 之后）
├── blas/sbmv/arch22/
│   ├── chbmv_host.cpp                       # 校验、tiling、分核、启动
│   ├── chbmv_kernel.cpp                     # AIV 内核（三 pass + 平面复数运算）
│   ├── chbmv_kernel.h                       # 内核入口签名
│   └── chbmv_tiling_data.h                  # tiling 结构（host/kernel 共享）
├── blas/sbmv/README.md                      # 产品支持表追加 aclblasChbmv：Atlas A2/A3 支持
└── test/sbmv/chbmv/
    ├── chbmv_param.h / chbmv_golden.h / chbmv_npu_wrapper_impl.h   # 仿 ssbmv 共享层
    └── arch22/
        ├── chbmv_test.cpp                   # GTest：加载随包 1200 条 CSV + 白盒不变式
        ├── chbmv_npu_wrapper.h              # NPU 封装
        ├── chbmv_perf.cpp + chbmv_perf_baseline.csv  # msprof op 性能采集
        ├── chbmv_test.csv                   # 随包用例 + GAUSS_NORM_5_2 补充用例
        └── README.md                        # 复现步骤
```

构建接入沿用 sbmv 族 CMake 模式（`test/sbmv/ssbmv/CMakeLists.txt` 同款，新增 chbmv 子目录条目），无根构建改动。

# 测试设计

## 精度测试

随包 `chbmv_test.csv` 共 1200 条，分组全量执行：

| 分组 | 条数 | 覆盖 |
|---|---|---|
| TC_L0 | 6 | UPPER/LOWER × 基础小尺寸（n=4/8/12, k=1） |
| TC_SQ | 90 | 尺寸扫描：n=0/1/小质数/2 的幂±1/非对齐 |
| TC_IN | 72 | 带宽扫描：k=0/1/小带宽/满带 k=n−1（n≤2048, k≤2047） |
| TC_LD | 12 | lda padding：紧凑 k+1 与多档 padding |
| TC_CV | 16 | 步长组合：±1/±2/±3 |
| TC_FL | 18 | 标量特殊值：alpha/beta=(0,0)/(1,0)/纯虚/大值，y=0、A/x 含 Inf/NaN |
| TC_AB | 16 | 负向：空指针（A/x/alpha/beta/y）、非法 uplo(999)、n=−1、k=−1、lda=k、incx/incy=0、INT32_MIN、INT32_MAX 跨度溢出 |
| TC_ED | 33 | 边界组合 |
| TC_EX | 737 | 正交随机扩展（均匀分布） |
| TC_PF | 200 | 性能用例（精度同跑，incx=incy=1） |

自补用例：扩展 `gen_csv.py` 增加 `GAUSS_NORM_5_2`（μ∈[−5,5]、σ∈[0.1,2]）正态分布生成，覆盖 alpha/x/y/A 四参数 × 均匀/正态 50% 混合要求，计划 ≥40 条（UPPER/LOWER、k=0/1/中等带宽/满带、n∈{1,16,64,256,1024} 正交），纳入精度自验并在自测报告说明。

golden：随包 cblas_chbmv（fp64 真值、fp32 收窄、CblasColMajor、Hermitian 约束解释 A）；判定 `|actual−golden| ≤ atol + rtol·|golden|`（rtol=atol=2⁻¹³），matched_ratio≥0.99 且 max_abs_error ≤ max(1e−2, 32·ULP动态)；Inf/NaN 按 §3.2 等效规则单独统计。

白盒不变式测试（自研补充）：
1. **beta=(0,0) 不读 y**：y 填充 NaN/Inf 位模式，结果与 y 填 0 完全一致；
2. **对角虚部忽略**：对角虚部填随机垃圾，结果与填 0 一致；
3. **未引用区域隔离**：UPPER 左上 / LOWER 右下 k×k 填 Inf/NaN，结果不受影响；
4. **确定性**：同输入复跑 3 轮位级一致；
5. **实数退化**：虚部全 0 输入与 `aclblasSsbmv` 结果交叉一致（同架构对拍）。

## 性能测试

- 工具：msprof op 模式采集单算子 task duration；5 次 warmup 不计样本，连续 ≥10 次有效样本取算术平均；
- 判据：196 条性能用例（含 base 4 条）逐一满足 Avg ≤ gpu_baseline.csv 对应条目 × 1.25；
- 异常样本不计入并在自测报告说明。

# 可维可测分析

## 精度标准/性能标准

- 精度：FLOAT32 标准 rtol=atol=2⁻¹³（1.2e−4）、matched_ratio≥0.99、max_abs_error≤max(1e−2, 32·ULP)；fp32 平面累加的理论误差 ~√(2k+1)·2⁻²⁴（最坏 k=2047 行内 ~2.6e−6），低于容差两个数量级，无需 fp64 中间累加；
- 性能：Atlas 800T A2 (910B3)，msprof op 口径，0.8× GPU（≤1.25× 耗时）逐 case 判定；
- 可测性：测试全量 CSV 驱动、golden 独立于实现（fp64 cblas）、白盒不变式独立于 golden；内核 pass 结构与 Ssbmv 同构，代码评审可对照已合入实现逐 pass 比对。

## 兼容性分析

- 复数类型复用 `cann_ops_blas_common.h::aclblasComplex`，不新增类型、不改既有 ABI；
- 接口声明进 `include/cann_ops_blas.h` 共用头，参数序列与 cublasChbmv 一一对应，无私有平行接口；
- 非法 uplo 返回 `INVALID_VALUE`（任务书 §7.5 拍板口径），与 Csymv 的 `INVALID_ENUM` 差异已在本文档与代码注释中显式记录，便于后续仓内统一；
- n=0 quick return、beta=0 不读 y、对角虚部忽略三项语义与 cuBLAS/Netlib 严格对齐，向后兼容既有 BLAS 生态调用习惯。
