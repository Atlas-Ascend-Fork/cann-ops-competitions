# 需求背景（required）

## 需求来源

本设计对应社区任务“指数分布后处理融合算子开发”，算子类型为 `FusedExponential`，对外接口为 `aclnnFusedExponential`。需求依据为随任务发布的 `FusedExponential_task_doc.md` 和 `FusedExponential-自验包`，设计结构遵循社区任务官方设计模板。

本文为设计评审稿，描述拟实现方案和验收方法，不将设计目标表述为已经完成的编译、实机或性能结论。

## 背景介绍

### 现有实现及优化目标

op-plugin 26.1.0 的 `ExponentialKernelNpuOpApi.cpp` 在均匀采样后，通过 `neg_ → add_ → 比较 → masked_fill_ → log_ → mul_` 完成指数变换。比较步骤产生独立 bool mask，连续的计算步骤增加 Kernel 调度及 GM 读写。

本任务融合采样后的逐元素变换，输入为调用者已生成的均匀样本，不实现随机数生成器，不管理 seed、offset 或随机状态。计算在 UB 内完成，连续场景直接写入用户输出，消除全量 GM 中间张量和 bool mask。原始链仅作为语义与性能参考；精度以任务自验包提供的 CPU float64 golden 为准。

### 功能分析

设输入为 `u`、率参数为 `λ`，输出为 `out`：

```text
epsilon = finfo(u.dtype).eps / 2
t = min(1 - u, 1 - epsilon)
out = -log(t) / lambda
```

输入与输出 shape、dtype 相同，不广播。不同元素之间无数据依赖，无跨核归约。输入采样域为 `[0, 1)`；仅对 `u` 接近 0 的一侧保护，不钳制 `u → 1` 的重尾一侧。

# 需求分析（required）

## 需求描述

在 A2/A3 上以 Ascend C 实现融合算子，提供标准 aclnn 两段式接口，支持 FLOAT16、BFLOAT16、FLOAT32、连续及非连续 Tensor、独立输出及合法原地输出。通过官方 1000 条精度、200 条性能和内存自验，并补充接口异常及边界测试。

## 需求拆解

| 项目 | 实现及验收要求 |
| --- | --- |
| 接口 | 保持任务书给定的两个函数签名及参数顺序，λ 为 Host DOUBLE scalar |
| 精度 | 使用包内 float64 golden；逐 dtype 满足混合容差及最大绝对误差限制 |
| 性能 | 200 条逐条满足拼接链 Device 耗时 / 融合 Device 耗时 ≥ 1.0 |
| 内存 | 连续路径不申请元素数相关的 GM 临时空间，workspace 目标为 0 |
| 布局 | 输入、输出非连续分别处理；保留 storage offset 与视图逻辑顺序 |
| 边界 | 空张量、0 维单元素、λ=+inf、非法参数、数值收窄溢出及下溢 |
| 确定性 | 相同输入、λ、设备和执行路径重复调用逐位一致 |
| 工程 | 接入 ops-math 的注册、构建、安装和测试链，不另建平行算子框架 |

## 任务表述差异与本设计口径

以下将可由公式、测试包及仓库规范判定的事项明确化；涉及不同层次约束的事项给出本设计选择，供设计评审确认，不声称已获得任务方解释。

| 差异 | 依据与采用口径 |
| --- | --- |
| 参数表写 rank 1～8，后文接受 0 维 | 冻结包含 177 条 `shape=[]`，因此支持 rank 0～8。0 维是一个元素，不是空张量；含 0 长度维才是空张量 |
| 参数表称独立输出，约束又要求原地 | 两种均支持：独立 out 不修改 u；同一逻辑视图的完全别名允许原地覆盖。不同布局的部分重叠不能按直接原地 Kernel 处理，交由框架检查或快照路径，不能产生核间读写竞争 |
| 参数表 BF16 ε 近似值写为 2^-9 | 按明确公式及 golden：FP16 为 2^-11，BF16 为 **2^-8**，FP32 为 2^-24。BF16 的 `finfo.eps` 是 2^-7，除以 2 后不是 2^-9 |
| 五算子链与六次 Kernel | 五个算术/填充阶段之外还有一次产生 mask 的比较；性能严格测包内包含比较的完整链，不漏算 `ge` |
| 低精度链与 float64 golden | §3.5.1 和 `function_FusedExponential.py` 明确先 float64 计算，再收窄到输出 dtype。采用该精度标杆，不复制原链每一步的低精度舍入，不替换 golden |
| λ 非法值错误由哪层返回 | §2.5 和 facts.json 将 `λ≤0/NaN` 校验归于 torch 适配层；本设计在适配层执行 `λ>0` 检查。直接 aclnn 调用的数值域不追加与该口径冲突的 λ 拒绝逻辑，仍校验 scalar 非空及 DOUBLE 类型；正值域外不承诺采样语义。该层次选择需在 API 文档明示 |
| λ=+inf | 对合法输入输出全 0，单独选择填零路径，避免 `inf × 0` 等中间运算；不生成随机输入，不读取 u 元素 |
| 数学输出严格正与存储结果 | 实数公式为正，但 dtype 收窄可下溢为 +0 或上溢为 +inf；不得为了强制非零而改变 golden。λ=+inf 本身要求输出 0 |
| u 的采样域与低精度生成 | 按实际存储的输入值计算。BF16 下 `1-2^-10` 收窄可能得到 1，此时公式产生 +inf；记录输入位模式并按非有限值规则核对，不私改冻结用例或把 u 强行截小 |
| workspace=0 与文档 512 字节示例 | 0 为连续路径设计目标；文档中的 512 是示例，不能当作豁免。若框架有固定运行开销，须给出来源、实测大小、是否随 N 增长及必要性说明，提交评审，不能隐藏为 0 |
| 性能比值方向 | 统一 `ratio=baseline_us/under_test_us`，≥1.0；与反向比值≤1.0等价。使用未四舍五入数值判断，避免 0.99996 显示为 1.0000 后误判 |
| 指定目录与通用命名 | 任务书 §5 指定 `experimental/math/FusedExponential`，通用开发指南推荐 `fused_exponential`。本设计暂按任务指定目录，内部源文件用小写下划线；合入目录如需改名，由维护者在评审中统一，不同时保留两套目录 |

# 详细设计（required）

## 算子分析

### 数学公式与边界

对合法有限输入可令 `v=max(u, epsilon)`，计算 `-log(1-v)/λ`，与任务公式等价。使用最大值代替显式 bool mask，不在 GM 中保存比较结果。

对 `u=0` 返回 `-log(1-epsilon)/λ` 的输出 dtype 表示；`u=1` 返回 +inf（有限正 λ），`u>1` 的对数输入为负，产生 NaN。u 不做 Host 值域扫描。NaN 传播不得依赖可能忽略 NaN 的 Min/Max 指令行为，补充显式验证及必要的 UB 内处理。

| dtype | ε | 计算与落盘 |
| --- | --- | --- |
| FLOAT16 | 0.00048828125（2^-11） | 转 FP32 计算，最终一次舍入回 FP16 |
| BFLOAT16 | 0.00390625（2^-8） | 转 FP32 计算，最终一次舍入回 BF16 |
| FLOAT32 | 0.000000059604644775390625（2^-24） | FP32 计算及输出，小输入使用稳定计算分支 |

λ 保留 Host double 语义，不先将原始 λ 截成 float 后再取倒数，以免极小/极大有限 double 变成 0/inf。任务采样范围 `[0.002,10000]` 使用 double 计算缩放系数后传递 FP32 系数；范围外有限正值使用 mantissa/exponent 分解的缩放路径，避免倒数本身溢出或下溢使可表示输出错误归零/变 inf。缩放顺序需按输出量级验证，不能把范围外 λ 静默夹紧到测试范围。

### 接口定义

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor *u, const aclScalar *lambd, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFusedExponential(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

| 参数 | 位置/类型 | 约束 |
| --- | --- | --- |
| u | Device tensor | ND，FP16/BF16/FP32，rank 0～8，允许非连续，合法采样域由调用者保证 |
| lambd | Host aclScalar | DOUBLE，合法功能域为 λ>0，包含 +inf；适配层拒绝非正值及 NaN |
| out | Device tensor | 与 u 同逻辑 shape 和 dtype；可独立存储或合法原地别名，允许非连续 |
| workspaceSize | Host uint64_t 输出 | 返回完整执行链所需 workspace，不能仅返回融合 Kernel 的工作区大小 |
| executor | Host 执行器输出 | 用仓库执行器生命周期管理，按两段式调用契约使用 |
| workspace/stream | 执行阶段 | 使用传入 stream；workspace 在执行完成前有效；结果回读前同步 |

空指针按仓库通用接口规范返回 `ACLNN_ERR_PARAM_NULLPTR`；dtype、shape、rank、格式及 scalar 类型不匹配返回 `ACLNN_ERR_PARAM_INVALID`。空张量先完成必要的元数据校验，再返回空执行器及零 workspace，不用空张量绕过错误参数检查。第二阶段沿用公共执行器的错误码和 stream 管理。

### 支持形状

连续数据展平为 N 个元素，不对具体 shape 建白名单。使用 64 位元素数及 GM 偏移，检查维度连乘和字节数溢出；分块长度进入指令参数前检查其位宽上限。非连续数据按逻辑 shape 计算，不把底层 storage 的填充区计入有效元素。

## 算子实现

### 实现方案

整体流程为：L2 元数据检查与布局处理 → L0 融合节点 → Host Tiling → AI Vector Core 分块 CopyIn/Compute/CopyOut → 必要时输出 ViewCopy。连续主路径必须将计算直接落到调用者 out，不能照搬“分配全量临时输出再复制”的通用例子。

#### 3.2.1 Host 侧设计

**接口与注册。** OpDef 注册三种输入输出 dtype、ND 和 A2/A3 支持配置。L2 保留 `aclScalar*` 外部签名，Host 读取 DOUBLE 后转换为内部缩放元数据，λ 不作为额外 GM Tensor 搬运。内部属性和 TilingData 显式携带系数、缩放指数及填零标志，避免借用 float 属性保存完整 double。采用仓库已有注册与 L0 调度能力；新增符号须在最终 `libcust_opapi.so` 中导出。

**布局与别名。**

1. 连续输入、连续输出：直接绑定 u/out，无中间输出；完全同址同布局允许原地，每个 tile 完成读入后才覆盖其对应区间。
2. 非连续输入：复用框架 Contiguous/AutoContiguous，先按视图语义生成连续输入。
3. 非连续输出：分配必要的连续结果并通过框架 ViewCopy 写回指定视图，不覆盖视图外元素。
4. 非连续原地：先完成输入快照，再计算、写回；由执行器依赖关系保证顺序。
5. 部分重叠或内部重叠视图按框架支持范围检查，不进入无保护的直接原地路径。无定义的多元素写同一存储位置不作为支持承诺。

布局转换工作区计入 GetWorkspaceSize 的返回值，连续零 workspace 约束不被扩大为“所有非连续路径也零 workspace”。直接输出绑定与内存复用能力需通过最小 aclnn 原型验证，禁止仅通过把返回值写成 0 满足内存要求。

**分核策略。** 从平台读取 AI Vector Core 数和 UB 容量。设输入元素字节数为 s，DMA 对齐粒度为 32 字节，逻辑分配粒度 `A=32/s`。以 `B=ceil(N/A)` 个块均衡分配到 C 个有效核：`q=B/C`、`r=B%C`，核 k 负责 `q+(k<r)` 个块，起始块为 `k*q+min(k,r)`。最后一个核的有效长度裁剪到 N，不访问越界地址。

小张量采用一个或少量核，避免空核和调度开销；大张量按平台核数扩展。C 和 tile 大小由 N、dtype、平台资源决定，不按官方 case id 或具体 shape 特判。空 Tensor 不启动 Kernel；rank 0 按 N=1 处理。

**UB 分块。** 设每 tile 为 T 个元素、队列深度为 Q（单/双缓冲），预留临时空间 R，FP32 工作向量数为 W，则按保守上界选择：

```text
2 * Q * T * s + 4 * W * T + R <= 可用 UB 字节数
```

输入/输出队列按实际生命周期分配；W 包括减法、对数、稳定计算和选择所需缓冲，不能只计算输入输出。T 向 DMA/矢量指令共同支持的粒度下取整，最小一个合法 tile。初版先采用可审计的保守缓冲预算，经过依赖分析再复用 UB，不能因别名复用破坏流水同步。

**TilingData。** 包含 `totalLength`、每核块数及余数、`tileLength`、尾块有效长度计算依据、缩放系数/指数和路径标志。元素总量及偏移使用 64 位；dtype 编译特化。TilingKey 只区分通用计算与 +inf 填零等确有必要的路径，不按 rank 建多套 Kernel。

#### 3.2.2 Kernel 侧设计

1. **Init**：解析 TilingData，绑定核的 GM 区间，初始化 TPipe、输入输出队列及 FP32 临时区。
2. **CopyIn**：有效元素搬入 UB；FP16/BF16 转 FP32。使用目标 A2/A3 支持的尾块搬运接口，补齐仅发生在 UB，不能越界读写 GM。
3. **Compute**：在 UB 中计算 ε 下限保护、对数和 λ 缩放；最终转回目标 dtype。所有比较中间状态留在寄存器/UB。
4. **CopyOut**：只写有效输出元素。双缓冲遵循入队、出队、释放和搬运/计算依赖，核间不共享临时区。

普通区间采用 `Adds/Muls/Maxs/Ln/Cast` 等目标平台可用的 Ascend C 矢量原语组合，具体重载按 CANN 9.1.0 的 A2/A3 编译结果确认，不引用仅适用于 950 的接口。

小输入下直接求 `1-u` 可能损失相对精度。拟在 `v=max(u,epsilon)<2^-8` 时使用正项展开：

```text
-log(1-v) ≈ v * (1 + v * (1/2 + v * (1/3 + v/4)))
```

截断余项不超过 `v^5/(5*(1-v))`，阈值处约为 1.83e-13；这只是实数截断误差界，不代替 FP32 舍入、Ln 误差、缩放与最终收窄的全量测试。其余区间计算 `-Ln(1-v)`。使用 UB 内选择，不产生 GM mask。阈值邻域、u=ε 两侧及 λ 小值放大场景单独测试。性能不足时先用 profiler 判断稳定分支成本，再优化实现，不更换 golden 或放宽容差。

`λ=+inf` 走直接填零分支。有限正 λ 的极端缩放按指数分解执行，输出的 inf、subnormal 和 0 需与 float64 golden 收窄一致；必要时为极端值单独路径，不能让普通路径的倒数溢出污染最终结果。

### 内存与复杂度

连续主路径算法复杂度 O(N)，每元素一次输入读取、一次输出写回，理论 GM 流量约 `2*N*s`，算法 workspace 为 0。UB 空间 O(T)，与全量 N 无关；无跨核同步、原子更新或全量统计缓冲。

独立输出的逻辑张量内存为 `input_bytes+output_bytes`；原地输出物理存储可复用，但官方报告仍按输入+输出+workspace 的指定口径填写，不用物理别名减少统计值来制造内存优势。非连续转换开销另列。若实际 aclnn 框架返回固定 workspace，查明属于运行设施还是中间张量，并按任务书例外条款说明。

### 工程组织与构建

任务指定算子目录下按标准 aclnn 工程组织必要文件：

```text
experimental/math/FusedExponential/
├── CMakeLists.txt
├── README.md
├── docs/design.md
├── op_api/                 # aclnn 两段式接口、L0 封装及声明
├── op_host/                # OpDef、InferShape、*_tiling.cpp、必要构建配置
├── op_kernel/              # fused_exponential.cpp、实现及 TilingData/Key 头
├── examples/               # 可编译运行的 aclnn 示例
└── tests/                  # API/Tiling UT、边界及集成测试
```

源文件命名使用 `fused_exponential_*`；Tiling 源文件保留 `_tiling` 标记以便构建发现。InferShape 复制输入 shape，InferDtype 保持输入 dtype；图模式支持不扩大为任务未要求的动态 shape 承诺。只增加构建实际需要的配置，不复制其他算子的无关目录、下载包、日志、二进制或临时脚本。是否需单独 op_graph 文件由仓库模板和注册方式决定。

使用仓库 `build.sh`，构建 experimental 算子包时设置 `--experimental`、`--ops`、`--soc` 和独立 vendor。A2 使用 `ascend910b` 系列对应配置，A3 使用仓库的 `ascend910_93` 配置，最终型号参数以目标版本支持列表为准。CANN 9.1.0 与当前上游工具链分别验证，不能将新版本编译成功视为最低版本兼容成功。

安装后校验库路径、两个 aclnn 导出符号、Kernel 注册与实际执行链；配置 `ASCEND_CUSTOM_OPP_PATH`、`ATK_CUSTOM_OPP_PATH` 指向自己的安装包，并保存动态加载证据，防止测到系统内置同名算子。

## 支持硬件

| 硬件 | 设计支持 | 验证计划 |
| --- | --- | --- |
| Atlas A2 / Ascend 910B3 | 是 | 任务要求的完整精度、性能、内存自验及补充测试 |
| Atlas A3 | 是 | 平台配置和工具链编译检查；取得设备后补充实机，不将编译结果标成实机通过 |

## 算子约束限制

仅支持 ND 和 FP16/BF16/FP32，输入输出同 dtype/shape；rank 0～8，无广播。输出存储须满足框架可写视图规则。不支持把不同布局的部分重叠直接当作安全原地更新。输入均匀分布及独立性由调用者保证；本算子只计算确定性变换，不检验分布，不负责抽样。用户需遵循异步执行器、stream、Tensor 和 workspace 生命周期。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 判据 | 来源 |
| --- | --- | --- |
| FP16 | atol=rtol=2^-10，matched_ratio≥0.99；最大误差限制见下文 | 任务书 §3.2 |
| BF16 | atol=rtol=2^-7，matched_ratio≥0.99；最大误差限制见下文 | 任务书 §3.2 |
| FP32 | atol=rtol=2^-13，matched_ratio≥0.99；最大误差限制见下文 | 任务书 §3.2 |
| 性能 | 200 条数据完整，每条 baseline_us/under_test_us≥1.0 | 任务书 §3.3、官方性能脚本 |
| 内存 | 无全量中间 GM 张量；连续 workspace=0，例外需有必要性证据 | 任务书 §3.4 |

逐元素比较 `abs(actual-golden) <= atol+rtol*abs(golden)`；最大绝对误差仅统计双方有限值，限制为 `max(兜底值,32*ULP(g_low))`，`g_low` 是最大误差点的 golden 收窄值。FP16/BF16 兜底值为 0.1，FP32 为 0.01。NaN、inf 按生态标准的非有限值规则独立检查，不能用忽略 NaN 的最大值函数掩盖异常。

### 官方精度测试

冻结文件实查共 1000 条：FP16 338、BF16 331、FP32 331；rank 0～8，其中 177 条 0 维，29 条含零长度维。320/种加40补量描述的是生成策略，不代表最终三个 dtype 各恰好320条。

原样使用 `function_FusedExponential.py` 的 CPU float64 golden、case id 随机种子和 `--slice_input non_contiguous --slice_input_ratio 0.4`。固定 ATK 26.9.8 与实际 torch/torch_npu 版本，保存用例及 golden 实现哈希、输入生成配置；环境/生成器变更后重建 golden，避免缓存与新输入错配。

29 条空张量的登记不等于29条自动通过。仅当日志证明在算子调用前的非连续归一化 `aclnnInplaceCopy` 报 161002，且 case 在登记表中，才分类为工具缺陷；其他错误仍为待修复。独立 aclnn 测试补齐所有空张量语义，报告分别列总数、实际执行通过、工具阻塞、精度失败与执行失败。

### 补充功能与异常测试

| 维度 | 覆盖内容 |
| --- | --- |
| 元数据 | 三 dtype × rank 0～8；0 长度、单元素、奇数长度、32字节边界及 tile/核边界前后 |
| 布局 | 连续、转置、步长切片、storage offset；非连续输入/输出分别与组合测试；视图外哨兵不变 |
| 别名 | 独立输出输入保持不变；连续原地与非连续原地；部分重叠不发生静默损坏 |
| 数值 | u=0、ε及两侧邻近表示值、稳定算法切换点、各 dtype 最大小于1的值、收窄到1的边界 |
| λ | 0.002、0.1、1、1000、10000、+inf；测试域外可表示正 double 的极小/极大值及缩放边界 |
| 非法调用 | 空参数、dtype不支持或不一致、shape不一致、rank超限、非DOUBLE scalar；适配层λ=0/负数/NaN拒绝 |
| 非有限值 | u=1、u>1、u<0、NaN/inf 的算术行为另列，不混为合法均匀分布用例 |
| 执行 | 多stream隔离、重复调用逐位一致、传入设备编号、空执行路径、workspace大小/生命周期 |

Frozen JSON 的正常采样范围不能代替对上述异常路径的补测。原地测试每次从同一输入快照开始，不能把已经变换的输出作为下一次均匀样本。

### 官方性能测试与优化

官方子集实查200条：FP16 58、BF16 77、FP32 65，不含0维和空张量。逐 id 保存相同型号、dtype、shape、λ、融合耗时、拼接耗时、比值及判定。缺少任何一侧数据均不得算通过。

拼接链按包内 Profiler Level1 采集，窗口外预热10次、窗口内60次，`op_statistic.csv` 全表 `Total Time(us)` 之和除以60；融合使用 ATK `performance_device` 一次调用的 Device 耗时。不得混入 CPU wall time、漏掉比较 Kernel 或用只测 Ln 的时间替代整链。核对输入克隆、初始化操作是否处于官方规定窗口，保留原始 trace。

脚本提供展示比值时会四舍五入；最终验收逐条按原始 `baseline_us` 和 `under_test_us` 判断，不只读中位数或展示比值。优化按 small/medium/large 分析：小规模关注启动、分核和多余搬运，大规模关注有效带宽、UB利用和流水重叠；异常用例用 msprof/Profiler 检查实际 Kernel 数和时间，优化后重跑全部精度及200条性能。

### 内存、工具和交付

官方 workspace 探针覆盖三 dtype × 两种 shape × 连续/非连续，逐项保存输入/输出字节数、GetWorkspaceSize 实测值、总量及基线对比；额外增加小/大规模和原地路径检查。源码审查及 trace 验证无 GM mask，不能只凭 workspace 数值声称已消除全部中间输出。

按自验包执行 `run_accuracy.sh`、`run_performance.sh`、`run_memory.sh`，再生成报告。脚本版本与环境留痕；ATK 安装不可用时先解决依赖，不能用自写测试替代正式验收流程。已有 LOG、模板数据及示例数值不作为本算子的实测结论。

个人交付仓的 `task_submission` 必须包含以下7个文件，XLSX 中按模板嵌入真实终端截图，数据与日志一致：

```text
task_submission/
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx
└── 4.2 内存自验证日志.log
```

同时交付设计文档、算子源码及README、官方测试包与补充用例。保存代码完整 commit、构建安装及加载路径、软硬件版本、原始报告和数据哈希，确保被测代码等于交付代码。验收说明提供完整 commit 链接、分支和目录；验收前完成指定账号访问授权。上游代码 PR 仅包含规范要求的算子交付件，独立验收日志和下载产物不混入算子源码目录。

## 兼容性分析

新增接口，不修改既有算子语义或随机数接口。复用已有公共依赖，不改公共测试阈值、其他算子注册和版权策略。针对 A2/A3 使用平台能力查询，避免硬编码950资源。标准构建、安装、aclnn实际调用及非连续框架路径分别测试，不能以单独 Kernel 测试替代端到端验证。

社区设计评审与 ops-math 上游需求/代码评审是不同阶段。按社区要求提交设计文档；按目标仓 CONTRIBUTING 完成需求讨论、关联 Issue、CLA、CI、Committer/Maintainer 检视后合入。本文不预先宣称评审、验收或合入通过。

## 参考依据及核对版本

核对日期：2026-10-14。ops-math 基线：`11aa1c753cc9deac8597a495b6a1b359d412c359`。

- 任务资料：`FusedExponential_task_doc.md`，§2～§5；自验包 `readme.md`、`docs/`、`test_cases/function_FusedExponential.py`、`cases.json`、`perf/cases.json`、`facts.json` 和 `scripts/`。
- [社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。保留需求背景、需求分析、详细设计、可维可测分析四个一级章节，移除模板中的 Addcdiv 示例内容。
- [ops-math 贡献指南](https://gitcode.com/cann/ops-math/blob/11aa1c753cc9deac8597a495b6a1b359d412c359/CONTRIBUTING.md)。
- [AI Core 开发指南](https://gitcode.com/cann/ops-math/blob/11aa1c753cc9deac8597a495b6a1b359d412c359/docs/zh/develop/aicore_develop_guide.md)。
- [文档贡献指南](https://gitcode.com/cann/ops-math/blob/11aa1c753cc9deac8597a495b6a1b359d412c359/docs/CONTRIBUTING_DOCS.md)。
- [Exp aclnn 工程参考](https://gitcode.com/cann/ops-math/blob/11aa1c753cc9deac8597a495b6a1b359d412c359/experimental/math/exp/op_api/aclnn_exp.cpp)。仅参考执行器与布局处理，不能照搬其全量临时输出路径。
- [op-plugin 26.1.0 Exponential 参考实现](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)。
- [生态算子混合容差精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
