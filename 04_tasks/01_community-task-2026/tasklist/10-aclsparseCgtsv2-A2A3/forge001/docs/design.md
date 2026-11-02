# 需求背景(required)

## 需求来源

本需求来源于 2026 年 10 月 CANN 社区任务《aclsparseCgtsv2 算子开发（A2/A3）》，要求面向 Atlas A2/A3 训练系列产品，使用 C++、Ascend C 与 ops-sparse 算子工程开发复数三对角线性方程组求解接口 `aclsparseCgtsv2`，完成算子设计、开发、测试全流程，代码最终提交 ops-sparse（`sparse/cgtsv2/arch22/`）。接口与能力、精度、性能标杆均为 NVIDIA cuSPARSE `cusparseCgtsv2`（接口语义以 CUDA Toolkit 13.3 所含 cuSPARSE 13.3 为准；cuSPARSE 仅作标杆，NPU 实现使用 aclsparse + Ascend C）。

## 背景介绍

### aclsparseCgtsv2算子功能现状分析

三对角线性方程组求解是数值计算的基础内核，广泛出现于大型模型训练的隐式计算与预处理环节（如 Llama 3.1 70B、Qwen3-235B、DeepSeek-V3 等主场景对应的优化器与递推计算步），其求解效率直接影响端到端训练/推理吞吐。对标接口 `cusparseCgtsv2` 求解 `A·X = B` 的复数（complex64）三对角系统，核心语义如下：

1. **系数矩阵**：m×m 三对角矩阵 A 由三条只读一维对角线给出——下对角 `dl[m]`、主对角 `d[m]`、上对角 `du[m]`，边界 `dl[0] = du[m-1] = 0` 由调用方保证；求解全程三条对角线不得写改。
2. **右端项与解**：B 为列主序连续存储的 `[ldb, n]` 矩阵（元素 (i,j) 位于 `b[i + j*ldb]`），前 m 行为有效求解区；`n` 为 RHS 列数（**不是 batchCount**）。解 X **原地覆盖** B 的前 m 行（无独立 X 输出缓冲）；padding 行（m ≤ i < ldb）执行前后逐元素不变。
3. **算法硬红线**：必须执行带**部分主元选取**（partial pivoting）的消去——按复数模长比较选取主元、交换相邻方程行、每次交换同步更新全部 RHS 列、维护交换造成的第二上对角填充项（fill-in），再对上三角系统回代；**不得仅实现无主元 Thomas 算法**。
4. **接口族与调用序**（aclsparse Legacy C++ API Kernel 直调工程模式，非 aclnn 两段式）：
   1. `aclsparseCgtsv2_bufferSizeExt` 查询所需 workspace 字节数（n=0 返回 0，允许不分配）；
   2. 调用方按查询值足量分配 **128 字节对齐** workspace；
   3. `aclsparseCgtsv2` 在 handle 携带的 stream 上**异步执行**（执行 ABI 不传 workspace 实际大小）。
5. **新增独立复数 ABI**：本算子在 `include/cann_ops_sparse.h` 新增 `aclsparseComplex_t`（连续 float real + float imag 分量，complex64 布局，编译期断言 `sizeof == 8`、`alignof == alignof(float)`），不修改实数同族 `aclsparseSgtsv2(float*)` 既有声明；仓内当前不存在 Cgtsv2 公开接口，属全新接口开发。

# 需求分析(required)

## 需求描述

在 Atlas A2/A3 产品上基于 ops-sparse 开源仓工程框架，以 aclsparse Legacy C++ API Kernel 直调方式实现 `aclsparseCgtsv2` 复数三对角求解接口，Host 侧完成参数校验、workspace 查询与 NPU 调度，使用 Ascend C 开发 arch22 Kernel，核心计算在 NPU 上完成。接口声明新增于 `include/cann_ops_sparse.h`，签名与 cuSPARSE 同名参数逐项对应：

```c
typedef struct {
    float real;
    float imag;
} aclsparseComplex_t;

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

## 需求拆解

1. **功能语义**：求解带部分主元选取的复数三对角方程组 `A·X = B`，各 RHS 列独立求解、列间无数据依赖；`n = 0` 为合法空 RHS（查询返回 0、执行跳过、B 不变、不要求 workspace）；奇异系统保证完成执行、不崩溃、不越界，输出分类独立记录。
2. **参数校验与错误码**：Host 层校验空句柄、`m < 3`、`n < 0`、`ldb < max(1, m)`、`n > 0` 时各指针空、workspace 空指针/未 128 字节对齐、m/n/ldb 相关乘法上溢，越界返回参数错误（`aclsparseStatus_t` 仓内既有枚举，不新增枚举值）；raw C ABI 只传地址和整数，dtype/分配长度/stride 等不可见元数据由调用方保证。
3. **精度**：complex64 输出按实部/虚部分别按 FLOAT32 混合容差验收；CPU Golden 对同一份 complex64 量化输入升为 complex128 后按带状 `(1,1)` 求解（SciPy/LAPACK 带主元）。
4. **性能**：性能验收仅 910B3，主口径为 Kernel 总耗时中位数，全部 203 条性能用例每条倍率 ≥ 0.25、算术平均 ≥ 0.35。
5. **确定性**：相同设备、软件版本、输入及配置下复跑（恢复原始 B 后重复执行）输出一致，并保存复跑记录。
6. **工程化**：代码提交 ops-sparse master（`sparse/cgtsv2/` 公共层 + `arch22/` 平台层、`test/cgtsv2/` 测试），公共算法与平台实现解耦，支撑跨平台交叉回归；自测套件统一使用官方 ATK（精度 200 条随包 + 800 条验收、性能 203 条）。

# 详细设计(required)

## 算子分析

### 数学公式

以零基下标表示，`A(i,i) = d(i)`、`A(i,i-1) = dl(i)`（1 ≤ i < m）、`A(i,i+1) = du(i)`（0 ≤ i < m-1），其余元素为 0；边界 `dl(0) = du(m-1) = 0`。对每个 RHS 列 j（0 ≤ j < n）求解：

```
d(0)·X(0,j) + du(0)·X(1,j)                              = B(0,j)
dl(i)·X(i-1,j) + d(i)·X(i,j) + du(i)·X(i+1,j) = B(i,j)  (1 ≤ i < m-1)
dl(m-1)·X(m-2,j) + d(m-1)·X(m-1,j)                      = B(m-1,j)
```

求解采用带部分主元选取的高斯消去：沿 m 逐步进行复数模长主元比较（含 NaN/Inf 确定分支）与相邻行交换，消去下邻元素并同步更新全部 RHS 列与第二上对角填充项，得到上三角系统后沿 m 逆向回代得解，解原地写回 B 的前 m 行。全部消去/回代中间量在 complex64 域内计算（实部/虚部 float32 分量对），Golden 侧才升 complex128。

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| dl、d、du、B | complex64（`aclsparseComplex_t`，实部/虚部各 float32），四个 Tensor 统一 |
| m、n、ldb | int32 运行时标量 |
| workspace | 连续字节地址（uint8），128 字节对齐 |

### 支持形状

- dl/d/du：ND 连续一维，长度 m；
- B：ND 列主序连续，存储形状 `[ldb, n]`，有效求解区 `[m, n]`；
- 约束：m ≥ 3、n ≥ 0、ldb ≥ max(1, m)；不接受广播、转置视图或非约定 stride。

## 算子实现

### 实现方案

总体思路：本算子为 Recurrence（沿 m 的顺序依赖递推链）+ Sparse（三对角带宽结构）+ DynamicShape（m/n/ldb 运行时标量）形态，计算瓶颈是沿 m 的标量串行递推链（延迟受限）而非批量向量计算。设计主线为「**分解与 RHS 解耦 + RHS 列间并行**」的两阶段映射：消去/回代沿 m 严格顺序（每 RHS 列串行链固定），列间按 block 切分并行、无原子与写冲突；主元决策与消去系数只依赖 dl/d/du、与 B 无关（固定交换序列）。

#### host侧设计：

##### 1. 分核策略：

Host 沿「Validate 参数校验 / ComputeWorkspaceBytes workspace 查询 / Launch 下发」三段组织。下发阶段拆分为两个 Kernel 入口、同一 stream 顺序执行：因子分解 Kernel 固定 **1 个 AIV block**（顺序递推链不可并行，其结果供全部 RHS 列共享重放）；RHS 应用 Kernel 取 **numBlocks = min(n, AIV 核数)**，每 block 负责一段连续列区间，只读写自己的列区间。核数通过运行时接口动态获取、禁止写死；n 小于核数时 block 数收缩为 n，无空转 block；每列计算量相同，按列均分天然负载均衡。

##### 2. 数据分块和内存优化策略：

- **定长分块搬运**：GM↔UB 按定长 chunk 分块（chunk 按目标平台 UB 容量锚定，占用与 m/n 无关，无随规模增长风险），统一使用 `DataCopyPad`——GM 端偏移任意（B 按列访问的 `j*ldb` 级偏移）、UB 端起始 32 字节对齐、尾块非对齐合法，天然覆盖任意 m 的尾块与 padding 场景；
- **workspace 紧凑布局**：workspace 容纳消去乘子、消去后主/上对角、第二上对角填充项（4 条 complex64 流）与行交换标志（1 条 uint8 流）的线性紧凑布局，随 m 线性增长且远小于目标平台 L2 容量，满足内存验收条件（方案固有 workspace 不超过目标硬件 L2 Cache）；`bufferSizeExt` 返回字节数（n=0 返回 0），调用方按 128 字节对齐足量分配，覆盖 stream 异步执行全程后释放；
- **统一数据出口**：所有 GM 写经 UB 暂存后由 MTE3 发出，配合屏障同步保证"读完成才写"——B 原地更新且前向阶段块内读/写区间重叠，必须先整块读入 UB、UB 内计算、再写回，杜绝读-写竞争；
- **延迟隐藏**：递推为标量链，采用软件预取（计算第 k 步预读第 k+1 个元素）隐藏 UB 标量读延迟；搬运→计算、计算→写回→缓冲复用分别以事件同步/屏障定序。

##### 3. 路径分派策略（kernel 直调，无 TilingKey）

本算子为 aclsparse Legacy Kernel 直调模式，不经 TilingKey/框架二进制匹配流程，而由 **tiling 标量字段承载全部分派信息**：Host 将 m、n、ldb、实际 block 数等标量序列化后随启动参数下发，kernel 侧读取标量字段完成分支。dtype 仅 complex64 一组组合、m/n/ldb 全为运行时标量（kernel 内分块循环 + 尾块特判统一覆盖，含 m=3 最小阶、任意尾块长度、ldb padding）、n=0 与参数校验在 Host 层短路、奇异系统按 IEEE-754 自然语义传播不进 kernel 分支——因此单一 kernel 模板路径即可覆盖全部合法输入，无运行时多路分派。

#### kernel侧设计：

**因子分解（factor，单 block，沿 m 串行，与 B/n 无关）**：顺序扫描 m 步，每步依次执行——① 复数模长主元比较：采用缩放比较实现（max-abs 归一化后比较平方和），避免直接平方在中大模长输入下的上溢；任一操作数含 NaN、全零或 Inf 退化场景均固定走"不交换"分支，保证比较分支确定；② 相邻行交换判定与消去：消去乘子与状态更新中的复数除法统一采用 Smith 比例法（按除数最大分量缩放后再展开），避免直接公式分母平方上溢，运算序列固定；③ 产出消去乘子、消去后主对角/上对角、第二上对角填充项与行交换标志写入 GM workspace。交换决策只依赖三条对角线、与 B 无关，形成固定交换序列——这是确定性复跑与跨平台数值一致性锚点。

**RHS 应用与回代（apply，按 RHS 列区间并行）**：每 block 对所辖列区间内每列执行两遍：**前向应用**（沿 m 顺序）按固化的交换标志逐列重放相邻行交换（数学上等价于串行算法"每次交换同步更新全部 RHS 列"——交换决策已固化，各列独立重放），并以消去乘子做复数乘加更新；**回代**（沿 m 逆向）逐行计算 `x(i) = (rhs'(i) − du'(i)·x(i+1) − du2'(i)·x(i+2)) / d'(i)`，除法与消去共用同一防溢出实现。解经 UB 暂存原地写回 B 的前 m 行，任何路径严格不触碰 `[m, ldb)` padding 行；各 block 列区间互斥，无原子操作、无跨核写冲突。

**边界处理**：n=0 由 Host 短路（不下发任何 Kernel）；m=3 最小阶由循环自然覆盖；尾块长度任意仅改分块循环边界、不改运算序列；奇异/零主元/Inf/NaN 系数按 IEEE-754 自然传播（不崩溃、不越界、无替代错误码）；`k+1 ≥ m`、`i+1 ≥ m` 等越界风险步显式跳过。

## 支持硬件

Atlas A2 训练系列产品（910B3、910B4）与 Atlas A3 训练系列产品，对应架构目录 `sparse/cgtsv2/arch22/`（DAV_2201）；功能与精度验收覆盖全部型号，性能验收仅 910B3。软件栈 CANN 9.0.0 及后续配套版本。

## 算子约束限制

- m ≥ 3、n ≥ 0、ldb ≥ max(1, m)；n = 0 为合法空操作（查询返回 0、执行跳过、B 不变）；m/n/ldb 为运行时动态标量，每次执行内尺寸固定；
- dl/d/du 求解全程只读，`dl[0] = du[m-1] = 0` 由调用方保证；B 原地输出（无独立 X），padding 行 `[m, ldb)` 执行前后逐元素不变；
- workspace 由调用方按查询值足量分配且 128 字节对齐，覆盖 stream 异步执行全程；执行 ABI 不传 workspace 大小；
- 仅支持 ND 列主序连续存储，不支持广播、转置视图或非约定 stride；raw C ABI 不承诺检测 dtype/分配长度/stride 等不可见元数据；
- 奇异系统完成执行、不崩溃、不越界，非有限输出不计入数值精度通过；
- 仅支持 aclsparse Legacy C++ API 直调，不支持 ACLNN 调用与 GE 图模式；torch_npu 单算子适配为可选项。

# 可维可测分析(required)

## 精度标准/性能标准

**精度标准**：按生态算子开源精度标准执行。CPU Golden 对同一份 complex64 量化输入升为 complex128 后，以带状 `(1,1)` 带主元求解（SciPy/LAPACK，band 构造 `ab[0,1:]=du[:-1]`、`ab[1]=d`、`ab[2,:-1]=dl[1:]`），解与 NPU 输出按实部、虚部分别按 FLOAT32 混合容差判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| complex64（实部/虚部按 FLOAT32 分量） | 2^-13 | 2^-13 | 0.99 | max(1e-2, 32 × ULP) |

逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`；匹配率按分量统计且均不低于 0.99；非有限值不参与最大绝对误差/ULP 锚点计算；常规比对的溢出特例按精度标准（Golden 按 RNE 收窄为 FLOAT32 后同号 Inf 才通过）执行。NaN/Inf 分类与符号一致性另设独立特殊值套件判定（不替代常规精度判定）；验收含随包 200 条 + 官方 ATK 生成 800 条精度用例、特殊值套件与拒绝合同用例。

**性能标准**：测试设备 Atlas 800I A2（910B3），complex64 输入。主口径 = Kernel 总耗时中位数：GPU 侧 NCU 采集 `cusparseCgtsv2`，NPU 侧 Profiler 采集相同求解范围内所有 Kernel 耗时求和；每条 case 预热不少于 10 次、正式采样不少于 30 次；不计入首次编译、数据生成、初始化、数据拷贝与 B 恢复，每次求解前恢复原始 B 再计时，workspace 在预热/采样期复用；Event 辅助口径单独列示、不用于倍率验收。验收阈值：全部 203 条（3 主场景 + 200 扩展）每条倍率 ≥ 0.25、算术平均 ≥ 0.35。三条主场景（GPU 标杆来自随包基线文件）：

| 编号 | 场景 | m / n / ldb | GPU Kernel 总耗时中位数（μs） | 目标 |
|---|---|---|---|---|
| P-01 | Llama 3.1 70B | 8192 / 64 / 8192 | 732.032 | 倍率 ≥ 0.25 |
| P-02 | Qwen3-235B-A22B | 4096 / 64 / 4096 | 717.120 | 倍率 ≥ 0.25 |
| P-03 | DeepSeek-V3 | 7168 / 128 / 7168 | 1390.480 | 倍率 ≥ 0.25 |

**可维可测手段**：C++ UT/ST 覆盖接口、Host 校验链、Kernel、复数类型布局与完整调用流程（查询 → 对齐分配 → 异步执行 → 释放）；必测项含 dl/d/du 只读性、B 原地与 padding 不变性（机器断言）、确定性复跑记录、n=0 空操作、m=3 最小阶、尾块/padding 组合、对角占优与强制主元交换两条消去路径、零主元/奇异系统行为、workspace 对齐与生命周期边界、全部参数校验负向用例；精度路径另覆盖模长上溢极值、subnormal、零 RHS 等数值路径。

## 兼容性分析

1. **接口兼容**：参数序列与 cuSPARSE `cusparseCgtsv2` 一一对应；`aclsparseComplex_t` 为新增独立 ABI（编译期断言保证 complex64 布局），不修改 `aclsparseSgtsv2(float*)` 既有声明与行为，对仓内已有功能无前向/后向影响。
2. **工程兼容**：实现落点沿用仓内 `sparse/<算子>/archXX`、`test/<算子>/archXX` 组织；公共 Host/工具与平台层（arch22）解耦，支撑公共算法跨平台交叉回归；按既有构建模式接入根 CMakeLists，不新增重复 build 目录。
3. **环境兼容**：基于 CANN 9.0.0 及后续配套版本与 Ascend C 开发，无新增三方软件依赖（Golden 随测试资产提供）；异步执行依赖 handle 携带 stream，读回结果前须同步 stream；报告记录实际 CANN/驱动/固件/硬件型号与代码提交版本。
