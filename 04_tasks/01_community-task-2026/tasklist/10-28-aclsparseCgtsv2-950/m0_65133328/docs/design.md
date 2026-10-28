# aclsparseCgtsv2 算子设计文档（Ascend 950）

| 提交信息 | 内容 |
| --- | --- |
| 任务编号 | 10月28日社区任务-aclsparseCgtsv2 |
| 队名（GitCode 用户名） | m0_65133328 |

版本：最终验证提交 `b7371205d4da27e659beef3fbdb8d0967a745eb2`，2026-10-28。已通过 Cgtsv2 33/33、Sgtsv2 67/67 条 GTest，以及官方 ATK 发布 200/200、额外 800/800、特殊语义 8/8、合同 12/12。固定203条正式Kernel、独立Event和独立memory均已完成。

# 需求背景（required）

## 需求来源

社区任务《aclsparseCgtsv2 算子开发》，适用 Ascend 950（A5、DAV-3510、`arch35`）。代码目标仓库为 [cann/ops-sparse](https://gitcode.com/cann/ops-sparse)，目标分支 `master`。本文结构对应 [社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

接口语义以 CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 的 `cusparseCgtsv2` 为基线：单精度复数、部分主元选取、多 RHS、B 原地更新、workspace 查询以及异步 stream 执行。

## 背景介绍

三对角线性方程组可由三条一维复数对角线表示。带部分主元选取的消去允许在遇到零主对角或较小主元时交换相邻方程，并保留交换产生的第二上对角填充，覆盖无主元 Thomas 算法之外的非奇异输入。

初始读取版本为 `e0580fa`，最终开发分支已合并上游 `09fc9e4`。仓库已有 FP32 `aclsparseSgtsv2` 的 Legacy C++ API、Host 校验、SIMT 主元求解和测试工程。complex64 接口新增独立公开类型及符号，沿用仓库的架构目录和构建模式。

# 需求分析（required）

## 需求描述

使用 C++、Ascend C 和 ops-sparse 工程实现 `aclsparseCgtsv2_bufferSizeExt`、`aclsparseCgtsv2`。核心计算在 Ascend 950 NPU 上完成，CPU 负责参数校验、调度和测试 Golden。

| 项目 | 需求 |
| --- | --- |
| 数据类型 | complex64，连续 `float real` / `float imag`，大小 8 字节、按 `float` 对齐 |
| 系统与 RHS | 单个三对角系统，`m >= 3`，`n >= 0` 个 RHS；`n` 为列数 |
| 系数存储 | `dl/d/du` 各为连续 `[m]`，只读；`dl[0]=du[m-1]=0` |
| B 存储 | 列主序 `[ldb,n]`，`ldb >= m`；仅前 `m` 行原地写回解 |
| 动态能力 | `m/n/ldb` 均为运行时整数，每次调用重新校验与调度 |
| 算法 | NPU 值域筛选满足安全占优条件的系统进入 PCR，其余采用复数模长主元、相邻行交换及第二上对角填充 |
| workspace | 查询、分配 128 字节对齐地址、执行、等待 stream 完成后释放 |
| 空 RHS | `n=0` 返回零 workspace，执行无求解 Kernel |
| 异步与确定性 | 使用 handle 的 stream；相同配置、恢复原始 B 后输出复现 |

## 需求拆解

1. 在公开头文件增加复数布局、编译期断言和两个独立 C ABI 声明。
2. 实现 Host 参数校验、可见地址/字节数溢出校验、workspace 查询及 stream 调度。
3. 实现 A5 SIMT PCR 与高精度复数部分主元路径，保留系数和 B 的 padding。
4. 接入 C++ UT/ST、官方 ATK 200 条精度、额外 800 条精度及独立特殊/合同场景。
5. 在固定的 203 条性能输入上采集完整求解 Kernel、Event 和内存，保存原始证据。
6. 完成公开 ABI、A2/A3 源码兼容及可用硬件上的回归，输出设计、代码和自测材料。

# 详细设计（required）

## 算子分析

### 数学公式

使用零基下标。矩阵元素为 \(A_{i,i}=d_i\)、\(A_{i,i-1}=dl_i\)（\(1\le i<m\)）、\(A_{i,i+1}=du_i\)（\(0\le i<m-1\)），其余为零。对每一 RHS 列 \(j\) 求解：

$$
AX_{:,j}=B_{:,j},\qquad 0\le j<n.
$$

B 的实际元素地址为 `B[i + j*ldb]`；workspace 偏移和字节数以宽整数计算。

### 支持数据类型

```cpp
typedef struct {
    float real;
    float imag;
} aclsparseComplex_t;
```

通过 `static_assert` 检查 `sizeof(aclsparseComplex_t)==8`、`alignof(aclsparseComplex_t)==alignof(float)`，并在测试中验证 `real/imag` 偏移及 C ABI 可调用性。设备端按相同实虚分量布局加载与写回。

### 支持形状

输入为三条连续 `[m]` 系数及列主序 `[ldb,n]` B，合法整数条件是 `m>=3`、`n>=0`、`ldb>=m`。B 的 `[0,m)` 行参与求解，`[m,ldb)` 行按字节保持原值。

精度覆盖设计按任务附件保留 `2^20` 上边界及规定的有限因子域。实现依据公开整数接口和实际地址空间进行校验，测试因子域与接口动态能力分别记录。

## 算子实现

### 公开接口

```cpp
aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B,
    int ldb, size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B,
    int ldb, void *pBuffer);
```

### 3.2.1 Host 侧设计

#### 参数与状态

公共 Host 实现位于 `sparse/cgtsv2/cgtsv2_host.cpp`，workspace 和字节数校验集中在 `cgtsv2_workspace.h`。查询入口先检查 handle 与查询输出地址，再调用公共尺寸/数据地址校验；执行入口复用同一校验，然后检查 workspace。空 RHS 仍验证 `m/ldb`，随后返回零 workspace 或成功；数据和 workspace 地址在该路径可为空。`n>0` 时四个数据地址均要求非空，执行要求足量、128 字节对齐的 workspace。

乘法先检查 `size_t` 上限，再计算 `ldb*n*sizeof(aclsparseComplex_t)`、workspace 总字节与对齐增量；B 和 workspace 的原始字节范围另检查 `int64_t` 偏移上限。空 handle 返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`，尺寸、数据地址、workspace 地址或范围无效返回 `ACL_SPARSE_STATUS_INVALID_VALUE`；获取 stream 的失败状态原样传出，AIV 核数为零返回 `ACL_SPARSE_STATUS_INTERNAL_ERROR`。

#### workspace 与生命周期

最新主元路径为每个 RHS 保存四条长度 `m` 的 scaled complex 数组：分解后的主对角 \(D'\)、第一上对角 \(U'\)、第二上对角 \(U''\) 和消去后的 RHS \(B'\)。每个中间复数包含实部和虚部各一个 float-pair 尾数，以及共享的 int32 二进制指数，共 20 字节。workspace 头部另保留 128 字节用于 NPU 路由标志。`n>0` 时所需字节为：

$$
W=128+\operatorname{align\_up}(4mn\times20,128).
$$

PCR 复用同一数据区，两 bank 各保存三条系数与全部 RHS，每 bank 为 `(3*m+m*n)*8` 字节。`n>=1` 时，双 bank 的总量不大于 `80*m*n` 字节。`n=0` 的查询值仍为零。

查询仅计算大小。调用方为至少 `W` 字节的有效区准备 128 字节对齐设备地址；普通对齐分配可额外申请 `W+127` 字节再向上对齐，并记录查询值、实际分配值和原始分配地址。当前 C++ 测试额外设置前后 128 字节 guard，workspace backing 分配为 `W+127+256` 字节，对齐后核验有效区之外的 guard。释放时使用原始分配地址。

`arch35/cgtsv2_host.cpp` 的 `Cgtsv2Dispatch` 从 `aclsparseGetStream` 获取 stream，在该 stream 清零 workspace 标志并顺序提交 Prepare、各轮 PCR、Finish、Commit 和主元 Kernel。各 Kernel 读取设备标志决定执行对应路径；Host 不读回数据或路由标志，保持异步。数据、B 和 workspace 生命周期覆盖 stream 完成点。测试在复用 workspace、恢复 B 和读回结果时使用相同 stream 的顺序及必要同步。

#### 分核与构建

主元路径以 RHS 为并行维，每 block 为 32 个线程，线程通过 grid-stride 遍历 RHS 列；block 数为 `min(AIV核数,ceil(n/32))`。Prepare/PCR/Finish/Commit 以 `m*n` 个有效元素为并行维，独立计算 parallel block 数。`m/n/ldb/blocks/parallelBlocks` 通过 `Cgtsv2TilingData` 传入，设备地址不存入可持久化 Host 状态。

公共 Host、workspace、复数基础运算、float-pair 和 scaled 数学辅助位于 `sparse/cgtsv2/`；A5 调度、PCR 与主元 Kernel 位于 `sparse/cgtsv2/arch35/`，由仓库 `sparse/CMakeLists.txt` 自动按 SOC 收集。根 CMake 在 `ascend950*` 构建中定义 `OPS_SPARSE_CGTSV2_ARCH35`，使公共执行入口调用 A5 Dispatch；其他架构的非空求解入口返回 `ACL_SPARSE_STATUS_NOT_SUPPORTED`。`test/cgtsv2/CMakeLists.txt` 通过 `ops_sparse_add_gtest_tests` 注册本平台 UT/ST。构建继续使用根目录 `build/`，命令为：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=cgtsv2 --soc=ascend950
bash build.sh --ops=cgtsv2 --soc=ascend950 --run
```

### 3.2.2 Kernel 侧设计

#### NPU 路由与 PCR

Prepare 检查所有行系数以及有效 B 分量。PCR 要求 \(|d_i|>1.001(|dl_i|+|du_i|)\)，所有系数与 B 的实虚分量有限且绝对值不超过 \(2^{16}\)，每行主对角至少一个分量的绝对值不小于 \(2^{-16}\)。系数边界按零处理。任意条件不满足时设置设备回退标志，合法输入由 scaled 主元路径继续求解。

满足条件的系统进入并行循环规约（PCR）。步长 \(h\) 依次为 `1,2,4,...`。每行消去距离 \(h\) 的左右两项；令 \(\alpha_i=-l_i/d_{i-h}\)、\(\beta_i=-u_i/d_{i+h}\)，对应越界项取零，则一轮更新为：

$$
d'_i=d_i+\alpha_i u_{i-h}+\beta_i l_{i+h},\quad
b'_i=b_i+\alpha_i b_{i-h}+\beta_i b_{i+h},
$$

$$
l'_i=\alpha_i l_{i-h},\qquad u'_i=\beta_i u_{i+h}.
$$

各轮在两个 GM bank 之间切换，系数按行共享，RHS 对所有列并行更新。Reduce 检查规约结果是否有限；Finish 计算商并暂存于 scratch，同时检查商是否有限。随后独立 Commit Kernel 在全局回退标志为零时写回 B。任一步检测异常均设置回退标志，Commit 保留原始 B，由后续 scaled 主元 Kernel 重新求解。各阶段通过同一 stream 的执行顺序建立依赖。

路由依据 NPU 读到的实际系数与 B 值，每次调用重新判断。Host 提交过程不进行数据同步或 CPU 求解，输入案例编号不参与分支。

#### 复数运算及主元比较

复数乘加按实部/虚部展开。主元比较使用共同缩放：令 \(p\) 为当前主元，\(l\) 为下一行下对角候选，取 \(s=\max(|\Re p|,|\Im p|,|\Re l|,|\Im l|)\)。有限且 \(s>0\) 时比较：

$$
(\Re p/s)^2+(\Im p/s)^2\quad\text{与}\quad(\Re l/s)^2+(\Im l/s)^2.
$$

基础 `cgtsv2_math.h` 中的 `KeepPivot` 在缩放量为零时保留原主元，模长相等也保留原主元。高精度路径保留 float-pair 低位，并结合二进制指数进行归一化比较。非有限输入由主元路径处理，通过专门套件核查分类与输出。

复数除法由同一公共头文件的 `Div` 实现归一化 Smith 算法。对 \(a/b\)，当 \(|\Re b|\ge|\Im b|\) 时，定义 \(r=\Im b/\Re b\)、\(a_r=\Re a/\Re b\)、\(a_i=\Im a/\Re b\)，计算：

$$
\Re(a/b)=\frac{a_r+a_i r}{1+r^2},\qquad
\Im(a/b)=\frac{a_i-a_r r}{1+r^2}.
$$

另一分支按虚部分量归一化并交换对应实虚表达式。分母形成 `1+r*r`，控制分母运算范围。float-pair 运算使用保序加减及 FMA 误差项保存约 48 位有效尾数，除法在初始商上进行残差修正；相关表达式按 IEEE 次序编译。

额外 800 条集合中有 75 例 complex128 Golden 的解分量达到约 `1e43~1e94`。最新 scaled complex 为 float-pair 尾数增加 int32 二进制指数，中间运算按指数对齐和归一化，保存该类超出 float32 指数范围的有限值。最终写 B 时统一收窄为 complex64，常规溢出特例按 Golden 的 float32 RNE 收窄与符号判定。

#### 带部分主元消去

对第 `i` 行消去从 `i=1` 到 `m-1`。寄存器保持上一行当前主元 `p`、第一上对角 `u` 和 RHS `b`；当前原始行读入 `l=dl[i]`、`d=d[i]`、`v=du[i]`、`c=B[i+j*ldb]`。

当上一行主元模长不小于候选时，取 \(q=l/p\)；`l` 的实虚部均为零时直接令 `q=0`。保存上一行 \((p,u,0,b)\)，下一行更新为：

$$
p_{\mathrm{next}}=d-qu,\qquad
u_{\mathrm{next}}=v,\qquad
b_{\mathrm{next}}=c-qb.
$$

需要相邻行交换时，取 \(q=p/l\)，保存上一行 \((l,d,v,c)\)，其中 `v` 为第二上对角填充；下一行更新为：

$$
p_{\mathrm{next}}=u-qd,\qquad
u_{\mathrm{next}}=-qv,\qquad
b_{\mathrm{next}}=b-qc.
$$

已定形的上一行系数与 RHS 写入对应列的四条高精度 workspace 数组。原始 `dl/d/du` 始终作为只读系数访问，消去过程在 workspace 中保存 \(B'\)。

#### 回代与内存访问

先解末行，再逆序处理：

$$
X_{i,j}=\frac{B'_{i,j}-U'_iX_{i+1,j}-U''_iX_{i+2,j}}{D'_i}.
$$

第一、第二上对角项分别按有效行界加入。`D'/U'/U''/B'` 和后续使用的解保持 scaled float-pair 表示；B 仅接收每个最终解的 complex64 收窄值，回代继续使用高精度解寄存器。Kernel 的 B 读写均限定 `0<=i<m`，实际列间距采用 `ldb`；所有跨列和 workspace 地址采用宽整数。每个 RHS 线程只写自己的 workspace 与 B 列，执行次序固定。

#### 时间和空间

主元路径每列顺序消去与回代，工作量为 \(O(mn)\)。PCR 路径每轮并行处理全部有效元素，共 \(\lceil\log_2m\rceil\) 轮，工作量为 \(O(mn\log m)\)，依赖深度为 \(O(\log m)\)。两个路径共享 \(O(mn)\) workspace。正式性能采集累计一次求解的 Prepare、所有规约、Finish、Commit 及主元 Kernel 耗时；回退路径中已提交但快速返回的 Kernel 仍属于该次调用范围。

## 支持硬件

| 芯片 | 架构与实现 |
| --- | --- |
| Ascend 950PR / 950DT | DAV-3510，`arch35` |
| Atlas A2/A3 | 现有算子由 `arch22` 选择；新增 Cgtsv2 的设备实现范围由各平台 PR 分别维护 |

当前远端硬件为 Ascend 950PR，CANN 安装路径为 `/usr/local/Ascend/cann-9.1.0`。首次设备状态查询曾出现 `Critical` 及 TaiShan Core Hang 告警；后续 runtime 初始化、设备分配与 Kernel 执行成功，最终提交已通过 33 条 Cgtsv2 GTest。告警和实际运行结果均保留在环境与自测记录，报告另列实际驱动、固件和工具版本。

## 算子约束限制

输入系数满足三对角边界零值，四个数据数组按 complex64 和约定布局分配。raw 指针 ABI 的实际 dtype、分配长度和 stride 由调用方或适配层保证。B 是原地输出，padding 保持原值；无额外 `X` 或 `batchCount` 参数。

真正奇异系统独立记录提交状态、stream 同步状态和输出分类；常规精度集合使用非奇异有限输入。NaN/Inf 的独立语义以任务特殊值套件判定。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判定方法 | 来源 |
| --- | --- | --- |
| 常规精度 | 实部、虚部分别混合容差 `rtol=atol=2^-13`，匹配率各不小于 0.99 | 任务书及 opbase 现行混合容差标准 |
| 最大有限绝对误差 | 各分量最大有限误差不大于 `max(1e-2,32*ULP(g_low))`，锚点使用该误差位置的 float32 Golden | 任务书第 3.2 节 |
| Golden | 正式 ATK 使用同一 complex64 输入升为 complex128 的 SciPy/LAPACK 带主元带状求解；C++ GTest 使用独立 `std::complex<double>` 带主元带状参考实现 | 任务附件与 `test/cgtsv2/cgtsv2_golden.h` |
| 精度集合 | 随包 200、官方额外生成 800，分别运行；完整组合、特殊值和合同独立覆盖 | 任务书第 3.5 节及 ATK YAML |
| 确定性 | 同设备/版本/配置，恢复原始 B 后重复执行，保存逐次输出一致性结果 | 任务书第 2.1 节 |
| 性能 | 每条性能用例 `GPU Kernel 中位数 / NPU Kernel 中位数 >= 0.3` | 固定 203 条性能清单 |
| 采样 | 每例预热至少 10 次、正式样本至少 30 次；先累计每次求解全部 Kernel，再取中位数 | 任务书第 3.3 节 |
| Event | 同一 stream Event 包围求解；恢复 B 在计时前完成，单独列示 | 任务书第 3.3 节 |
| 内存 | 逐例报告 IO、查询/对齐 workspace、恢复副本、设备 free/total 与适用 allocator 基线/峰值/增量 | 任务书第 3.4 节 |

NPU Kernel 采用 TorchNPU Level1 CPU+NPU Profiler。每个样本与 B 恢复分别标记，通过完整 HostToDevice flow ID、精确设备端点 `(pid,tid,ts)` 和唯一主机提交范围逐项归属；所有求解 compute duration 相加。恢复计算与 DMA 单列并排除，缺失或歧义 flow 触发失败。CPU 与设备时间轴边界差异仅作为诊断。Kernel、memory、Event 由三个独立进程依次采集；每个进程跨用例复用 runtime，各例重新准备输入和 workspace，并保持 10/30 及样本内 workspace 复用。原生 handle 的 4 MiB 工作区通过 `aclsparseGetWorkspace` 查询，与 Torch allocator 和 Cgtsv2 workspace 分列。

正式性能清单包含 P-01 `8192/64/8192`、P-02 `4096/64/4096`、P-03 `7168/128/7168`（分别为 `m/n/ldb`）及 200 条扩展。GPU Kernel 基线与输入属性按附件 CSV 对齐。`n=0` 无求解 Kernel，按空操作语义验证。

内存验收逐项核对任务书两条条件：大于 500 MB 的 IO 场景比较双方同口径总量及额外内存，或固有 workspace 不大于目标设备 L2 容量。本机通过 ACLRT capability attribute 302 实际查询得到 L2 容量 134,217,728 字节（128 MiB）。按最新 workspace 查询公式，P-03 为 `128+80*7168*128=73,400,448` 字节，小于该容量。203条独立memory实测均满足workspace≤L2条件，查询、实际分配及所有内存分项见正式报告。

## 测试设计

| 层次 | 主要检查 |
| --- | --- |
| 类型/ABI | 大小、对齐、分量偏移、C/C++ 编译调用、原有 Sgtsv2 声明和符号 |
| Host UT | 空 handle/查询输出、`m<3`、`n<0`、`ldb<m`、各数据空指针、workspace 为空/未对齐、乘法和对齐溢出、合法 `n=0` |
| NPU 数值 ST | 单/多 RHS、非默认 `ldb`、普通随机、对角占优、需要 pivot、连续交换、零主对角但非奇异、病态与尺度边界 |
| 内存/stream | 系数只读、B 原地、padding、workspace 边界、异步 stream、等待完成后释放、重复调用无泄漏 |
| 特殊语义 | 奇异系统状态/输出，NaN 分类 mask、Inf 位置与符号、其余有限分量逐点容差 |
| 官方 ATK | 原生 Tensor/Attr 生成、200 与额外 800 分组执行、完整组合/特殊/合同覆盖审计及原始结果 |
| 性能/内存 | 固定 203 条同输入、10/30 采样、完整 Kernel、独立 Event、内存基线/峰值与原始 trace |

官方 ATK 使用 26.9.8，对应任务指定源码提交 `9504bee6076820ca74259fb919a790a12ef88cf8`。实际环境优先 Python 3.10/3.11、NumPy < 2。原生生成命令从随包 `test_cases/atk/` 执行：

```bash
atk case -f case_designs/accuracy.yaml -p atk_plugins.py -s 1028
atk case -f case_designs/accuracy_acceptance_800.yaml -p atk_plugins.py -s 20261029
atk task -c result/accuracy/json/all_accuracy.json \
  -n nodes_accuracy.yaml -p . -tk accuracy -sp --disable_id_seed
atk task -c result/accuracy_acceptance_800/json/all_accuracy_acceptance_800.json \
  -n nodes_accuracy.yaml -p . -tk accuracy -sp --disable_id_seed
```

生成与运行分别保存版本、种子、原生 JSON/CSV/XLSX 和覆盖审计。适配模块 `test/cgtsv2/torch_extension/cgtsv2_extension.py` 通过 `ctypes` 加载真实 `libops_sparse.so`，注册 workspace 查询与原地 NPU 执行接口，按 ATK Tensor 约定向 C++ ABI 传入设备地址，并将 handle 绑定到 `torch.npu.current_stream()`。性能使用任务同口径采集器，并保留每次样本与 Profiler trace。适配层与官方套件已进入测试目录，官方 NPU ATK 运行状态见下表。

## 兼容性分析

新增 `aclsparseComplex_t`、`aclsparseCgtsv2_bufferSizeExt`、`aclsparseCgtsv2`，既有 `aclsparseSgtsv2(float*)` 的类型和声明保持原值。公共类型仅定义一次；后续其他架构 PR 基于已合入头文件复用定义。

架构独有设备实现放在 `arch35`，公共 Host 工具以无 A5 设备依赖的类型与函数组织。A2/A3 验证分为源码构建、公开 ABI 和设备执行三项，报告分别给出可执行结果与对应环境。

## 交付与验证状态

交付目录按任务书保留设计原文件、源码、C++ UT/ST、完整 ATK 附件和根目录 `task_submission`。精度、性能、内存三个 `.xlsx` 报告与对应 `.log` 使用任务规定文件名，原始采样和 Profiler 记录作为附件。

截至 2026-10-28，最终提交及保留的历史验证状态如下：

| 项目 | 当前结果与范围 |
| --- | --- |
| CANN 9.1.0 / Ascend 950PR 构建 | scaled complex 与 PCR 实测版本已通过 |
| C++ GTest | 最终提交 Cgtsv2 33/33、Sgtsv2 67/67 通过 |
| NPU 数值 | 已测版本覆盖相邻/连续主元交换、第二上对角填充、复数模长、大/小量级、病态、单/多 RHS 与动态 padding |
| P-01 / P-02 / P-03 形状 | `8192/64/8192`、`4096/64/4096`、`7168/128/7168` 的 C++ 数值测试通过；使用 GTest 对角占优输入 |
| 原地与内存边界 | 数值夹具检查系数逐字节只读、padding、B/workspace 前后 guard 及恢复 B 后两次逐字节确定性，相关 GTest 全部通过 |
| 空 RHS 与奇异安全 | 已完成实际已创建 handle 的空 RHS 测试及奇异系统完成/guard/实际输出记录，相关 GTest 通过 |
| 首轮 float32 ATK | 发布集合 198/200，额外集合 651/800；失败项为精度上限问题，原始结果保留 |
| 未缩放 float-pair ATK | 历史版本发布集合 200/200，额外集合 722/800，原始结果保留 |
| scaled complex ATK | 发布集合 200/200、额外集合 800/800；其中 992 条常规数值、8 条合法空 RHS（每组各 4 条） |
| ATK 特殊与合同 | scaled 实测版本特殊语义 8/8、合同 12/12，与普通数值精度独立记录 |
| ATK 实际执行与原始记录 | `logs/final_20261028_v2` 四套 Task Summary 均为 Pass；NPU 求解和 CPU Golden 均实际运行，精度结果列于 `cpu_cpu_golden` |
| 最终提交固定 | `b737120`；7901 字节 scaled 头已强制编译，6 个设备依赖与最终提交逐文件一致，全部精度与回归已复跑 |
| 完整有限因子域 | 完整数值与设备运行尚未完成，保留覆盖审计和资源执行待办 |
| P-01 preflight | Kernel 中位数 1152.419 μs、GPU/NPU 约 0.556131，属于正式全量采集前的预采样 |
| 正式203条性能与内存 | Kernel、独立Event、独立memory各203/203完成；全部Kernel倍率≥0.3，最低0.530739、算术平均2.625433；workspace全部≤128 MiB L2 |
| A2/A3 交叉回归 | Arch22 构建与 Host 链接通过，workspace 查询为384字节，非空执行返回 NOT_SUPPORTED、空操作返回成功；没有执行 Arch22 设备 Kernel |

P-01～P-03 的数值结果验证相应形状的求解功能。正式性能使用附件冻结输入、逐条基线和完整采样流程。最终提交已完成全部精度复跑；固定 203 条性能与内存结果按实际采集状态更新设计评审 PR 与自测报告。

社区流程已核对 `【社区任务】流程及注意事项` #39，主帖最后修改时间为 2026-09-22。按 competitions 仓库现有 `任务目录/账号/docs/design.md` 结构，设计评审提交位置采用 `04_tasks/01_community-task-2026/tasklist/10-28-aclsparseCgtsv2-950/m0_65133328/docs/design.md`，标题采用 `【社区任务】aclsparseCgtsv2（Ascend 950）算子设计文档`。代码开发、剩余验证与设计评审并行推进，评审状态按实际外部流程更新。

## 正式性能结果（2026-10-28）

代码`b737120`；NPU每例10次预热、30次采样，三phase依次执行。GPU采用任务附件CSV；GPU原始NCU与版本记录未附。完整flow、所有dispatch Kernel、Kernel/Event样本中位数和逐例workspace已独立复核，203/203通过。

| 场景 | GPU Kernel μs | NPU Kernel μs | GPU/NPU | GPU Event μs | NPU Event μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| P-01 | 640.896 | 1207.555 | 0.530739 | 664.976 | 1238.004 |
| P-02 | 628.944 | 563.196 | 1.116741 | 641.184 | 592.710 |
| P-03 | 1220.048 | 2028.899 | 0.601335 | 1224.944 | 2058.478 |

内存采用任务条件2验收：最大查询workspace为73,400,448 bytes，实测L2为134,217,728 bytes。原生handle的4,194,304 bytes工作区单列；Torch allocator峰值与设备全局快照分别报告。各例准备前Torch allocated均为0。
