# 需求背景（required）

## 需求来源

社区任务 10-12「10月社区任务-指数分布后处理融合算子开发」，算子 `FusedExponential`，对外接口 `aclnnFusedExponential`（aclnn 两段式）。平台 A2/A3，落地仓库 [cann/ops-math](https://gitcode.com/cann/ops-math)（任务指定目录 `experimental/math/FusedExponential`）。需求依据为任务书 `FusedExponential_task_doc.md` 与官方自验包（冻结精度 1000 条 + 性能 200 条 + 内存口径）；本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，为设计评审稿，不将设计目标表述为已完成的实测结论。

## 背景介绍

### 功能定义

输入均匀样本 `u` 与率参数 `λ`（Host DOUBLE scalar），输出指数分布样本：

```text
epsilon = finfo(u.dtype).eps / 2
t = min(1 - u, 1 - epsilon)
out = -log(t) / lambda
```

对应 op-plugin `ExponentialKernelNpuOpApi` 的采样后处理链 `neg_ → add_ → ge(比较) → masked_fill_ → log_ → mul_`（六个 Kernel，含一次产生 bool mask 的比较）。本算子将六步融合为单个 Kernel，消除全量 GM 中间张量与 bool mask；随机数生成不在本算子范围（输入为调用方已生成的均匀样本，不管理 seed/offset/随机状态）。精度以自验包 float64 CPU golden 为准，不以原链逐算子舍入为标杆。

### 接口签名（任务书给定，保持参数顺序）

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor *u, const aclScalar *lambd, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnFusedExponential(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

### 任务关键口径（决定设计的客观事实）

| 口径 | 内容 |
| --- | --- |
| dtype | FLOAT16 / BFLOAT16 / FLOAT32，输入输出同 shape 同 dtype，不广播 |
| ε | FP16=2^-11（0.00048828125）、BF16=**2^-8**（0.00390625）、FP32=2^-24（参数表近似值 2^-9 与公式矛盾，以公式及 golden 为准） |
| rank | 参数表 1～8，冻结用例含 177 条 `shape=[]`，按 rank 0～8 支持；0 维是 1 个元素，含 0 长度维才是空张量 |
| 输出形态 | 独立 out 与合法原地（同逻辑视图完全别名）均支持 |
| λ | Host DOUBLE；λ>0 校验在 torch 适配层；λ=+inf 合法（输出全 0）；直接 aclnn 调用不追加 λ 拒绝逻辑，仅校验非空与 DOUBLE |
| 精度 | float64 先算后收窄到输出 dtype；逐 dtype 混合容差 + 最大绝对误差 |
| 性能 | `ratio = 拼接链 Device 耗时 / 融合 Device 耗时 ≥ 1.0`，基线为含比较的完整六步链 |
| 内存 | 连续路径不申请元素数相关 GM 临时空间，workspace 目标 0 |
| 确定性 | 相同输入/λ/设备/路径重复调用逐位一致 |

# 需求分析（required）

## 需求描述

在 A2/A3 上以 Ascend C 实现单 Kernel 融合算子，支持 FP16/BF16/FP32、rank 0～8、连续与非连续 Tensor、独立与合法原地输出；通过官方 1000 条精度、200 条性能与内存自验，并补充接口异常与边界测试；接入 ops-math 注册、构建、安装与测试链，不另建平行框架。

## 需求拆解

1. 元数据与校验：nullptr→`ACLNN_ERR_PARAM_NULLPTR`，dtype/rank/shape 不匹配→`ACLNN_ERR_PARAM_INVALID`；空张量完成校验后返回零 workspace 空执行器。
2. 连续主路径：单 Kernel 直写 out，零 workspace，O(N) 一读一写。
3. 布局路径：非连续输入 Contiguous 化、非连续输出 ViewCopy 写回、非连续原地快照重算，转换空间如实计入 workspace。
4. 数值路径：FP16/BF16 转 FP32 计算、最终一次收窄；小 `1-u` 稳定分支；`λ=+inf` 填零分支；极端 λ 指数分解缩放。
5. 工程交付：op_api/op_host/op_kernel/tests/docs + 示例，A2（ascend910b）与 A3（ascend910_93）双配置编译。

# 详细设计（required）

## 算子分析

### 数学公式与边界

令 `v = max(u, ε)`，则 `out = -log(1-v)/λ`，与任务公式恒等且不需要显式 bool mask（比较结果以 Max 选择吸收进计算）。边界行为：

| 输入 | 行为 | 依据 |
| --- | --- | --- |
| `u = 0` | `-log(1-ε)/λ` 的输出 dtype 表示 | ε 下限保护 |
| `u = 1` | +inf（有限正 λ） | `1-v = 0`；不做钳制，采样域 `[0,1)` 由调用方保证 |
| `u > 1` / NaN / ±inf | NaN / 非有限值按 IEEE 传播 | NaN 不依赖 Min/Max 忽略语义，显式用例验证 |
| `λ = +inf` | 输出全 0，独立填零分支，不读 u | 避免 `inf×0` |
| 收窄溢出/下溢 | 允许结果为 +inf / +0，与 float64 golden 收窄一致 | 不为"严格正"私改 golden |

| dtype | 计算路径 | 落盘 |
| --- | --- | --- |
| FLOAT16 / BF16 | UB 内 Cast 上行 FP32 → 计算 → 单次 Cast 收窄 | 一次舍入 |
| FLOAT32 | FP32 直接计算，`v < 2^-8` 走稳定分支 | — |

λ 保留 Host double 全程：内部以"尾数 + 指数"分解传入 Kernel（避免原始 λ 截成 float 后取倒数在极小/极大 λ 下提前溢出/下溢），采样范围 `[0.002, 10000]` 内退化为单 FP32 系数路径。范围外有限正 λ 不静默夹紧。

### 支持形状

连续数据展平为 N 个元素（64 位元素数与偏移，校验连乘溢出）；非连续按逻辑 shape 计算，不计入 storage 填充区；不对具体 shape 建白名单。

## 算子实现

### 实现方案

流程：L2 校验与布局判定 → Host Tiling → AI Vector Core 分块 CopyIn/Compute/CopyOut →（需要时）ViewCopy。连续主路径直接写调用方 out，禁止"全量临时输出再复制"。

#### 3.2.1 host 侧设计

**注册**：OpDef 注册三种 dtype、ND、A2/A3；L2 保留 `aclScalar*` 签名，Host 读 DOUBLE 后转内部缩放元数据（系数/指数/填零标志入 TilingData），λ 不走 GM Tensor。新增符号在 `libcust_opapi.so` 导出。

**分核**：设元素字节 `s`、DMA 对齐 32B、逻辑粒度 `A=32/s`、块数 `B=ceil(N/A)`，均衡分配到 C 个有效核：核 k 得 `q+(k<r)` 块（`q=B/C, r=B%C`），起始 `k·q+min(k,r)`，末核裁剪到 N。小张量收敛到少量核，不按 case id 特判。

**UB 分块**：`2·Q·T·s + 4·W·T + R ≤ 可用UB`（Q 队列深度、T tile 元素数、W 为 FP32 工作向量数、R 预留）；T 取 DMA/矢量指令公共粒度向下取整。

**TilingData**：`totalLength`、每核块数/余数、`tileLength`、尾块有效长度、缩放系数与指数、路径标志；元素总量与偏移 64 位；TilingKey 仅区分通用 / +inf 填零，不按 rank 建多 Kernel。

#### 3.2.2 kernel 侧设计

1. **CopyIn**：有效元素搬入 UB，FP16/BF16 上行 FP32；尾块补齐仅在 UB，不越界 GM。
2. **Compute**（UB 内，无 GM mask）：
   - `v = max(u, ε)`（Maxs 原语，吸收 masked_fill）；
   - `w = 1 − v`；`w < 2^-8` 时走正项稳定分支 `-log(1-v) ≈ v·(1 + v·(1/2 + v·(1/3 + v/4)))`（截断余项 `v^5/(5(1-v))`，阈值处 ≈1.8e-13，远低于 FP32 舍入），否则 `-Ln(w)`；
   - 乘 λ 缩放系数（有限路径），收窄回目标 dtype；
   - NaN 显式传播验证（不依赖可能忽略 NaN 的 Min/Max）。
3. **CopyOut**：只写有效元素；双缓冲按入队/出队/释放依赖，核间不共享临时区。

**布局路径**：非连续输入先 Contiguous；非连续输出经连续结果 + ViewCopy 写回视图；非连续原地先快照输入再计算写回（执行器依赖保序）；部分重叠视图不进入直接原地路径。转换 workspace 如实计入 GetWorkspaceSize，零 workspace 承诺仅限连续主路径。

### 内存与确定性

连续主路径 O(N)：每元素一读一写，GM 流量 `2·N·s`，算法 workspace 0（若框架返回固定开销，须给出实测大小与来源说明，不得隐藏）。分核等分 + 余数前缀，无原子、无跨核同步，逐位确定。

### 工程组织

```text
experimental/math/FusedExponential/
├── CMakeLists.txt / README.md / docs/design.md
├── op_api/      # aclnn 两段式接口
├── op_host/     # OpDef、InferShape（复制输入）、*_tiling.cpp
├── op_kernel/   # fused_exponential.cpp、TilingData/Key
├── examples/    # aclnn 可编译示例
└── tests/       # UT、边界、精度/性能自验脚本接入
```

构建走仓库 `build.sh --experimental --ops ... --soc ascend910b|ascend910_93`；CANN 9.1.0 与上游工具链分别验证；安装后以 `ASCEND_CUSTOM_OPP_PATH` 指向自装包并留存动态加载证据，防测到系统内置同名算子。

### 测试与验收

| 项 | 内容 |
| --- | --- |
| 精度 | 自验包全量 1000 条；另补边界（u∈{0,1,ε 邻域,NaN,±inf}、λ∈{+inf, 极小/极大有限}、rank0、空张量） |
| 性能 | 200 条逐条 ratio≥1.0（未四舍五入判断）；profiler 拆解稳定分支占比 |
| 内存 | 连续路径 workspace=0 核验；非连续路径 workspace 与转换开销单列 |
| 异常 | 空指针、dtype/rank 非法、out shape 不匹配、scalar 非 DOUBLE |
| 确定性 | 同输入串行 3 次逐位一致 |

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| BF16 ε 取值（2^-8 vs 参数表 2^-9）口径争议 | 以公式 `finfo.eps/2` + golden 反算为唯一标准，设计评审明示 |
| 稳定分支拖慢主路径 | 阈值分支占比极小；profiler 实测后可改谓词执行统一路径，精度以 golden 复验 |
| 非连续原地别名语义复杂 | 快照路径兜底正确性；重叠视图语义在 API 文档明示，不承诺未定义行为 |
| 950 与 A2/A3 原语集差异 | 仅引用 A2/A3 可用矢量原语（Adds/Muls/Maxs/Ln/Cast），不引用 arch35 专属接口 |
