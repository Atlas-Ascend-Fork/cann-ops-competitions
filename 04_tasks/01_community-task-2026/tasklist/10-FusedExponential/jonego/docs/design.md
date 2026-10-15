# FusedExponential 算子设计文档（Atlas A2/A3）

版本：v1.0（设计评审稿），2026-10-14

贡献者：jonego

代码仓：`cann/ops-math`，新增 `experimental/math/FusedExponential/`。

状态：首版已完成 A2 构建及接口小样验证。原包精度为 996 条通过、4 条在 ATK
构造空张量切片时失败，尚未完成全部验证，正式性能未开展。本文供方案评审，
不作为验收通过的证据；任务书与原自验包保持原样。

## 1. 需求背景（required）

任务为[10月社区任务-指数分布后处理融合算子开发](https://www.hiascend.com/activities/task-center/details/e179b493000a43378f416316cb544264)。以本地 `FusedExponential_task_doc.md`
及 `FusedExponential-自验包` 为需求和验收来源。基线源码仓为
`cann/ops-math`，开发起点 `11aa1c753cc9deac8597a495b6a1b359d412c359`。

输入已经是均匀随机样本，算子不生成随机数：

```text
epsilon = finfo(u.dtype).eps / 2
out = -log(min(1-u, 1-epsilon)) / lambda
```

支持 FP16、BF16、FP32，输出与输入同类型、同形状；DOUBLE Host 标量 lambda。
支持 A2 (`ascend910b`) 和 A3 (`ascend910_93`)，按任务要求在 910B3 上自验。
CANN >= 9.1.0，torch/torch_npu >= 2.7.1。

参考 [op-plugin 26.1.0 实现](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)
的 neg → add → 边界判断及 mask_fill → log → mul 后处理。
各步骤都是逐元素运算，无跨元素依赖，适合融合为一个 AIV kernel。
融合在 UB 内完成边界判断及数学变换，消除 bool mask 的 GM 中间张量和多次中间结果搬运。
非连续布局的连续化及回写仍由框架处理，其辅助 kernel 和内存计入相应路径成本。

文档结构参考[社区设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
和[设计文档 PR #1858](https://gitcode.com/cann/cann-ops-competitions/pull/1858/diffs)，
其中其他算子的技术方案、性能数字及历史验证结论不作为本任务依据。

## 2. 需求分析（required）

### 2.1 接口、布局与边界

手写 L2 接口保留任务书的 aclScalar 签名，内部将 double 标量转换为计算属性：

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor *u, const aclScalar *lambd, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnFusedExponential(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

L2 负责参数校验、连续化与非连续输出回写；Tiling 再检查类型、shape 和秩上界 8。
独立输出和完全相同视图的原地输出均采用逐元素路径。任意部分重叠视图不作为额外承诺。
零维 shape `[]` 处理一个元素；含零长度轴的输入不访问 GM。
lambda 为正有限数时计算公式，+inf 时直接输出正零；L2 接口拒绝 lambda<=0 或 NaN。
类型不支持、shape 不匹配等返回 `ACLNN_ERR_PARAM_INVALID`，空接口对象返回
`ACLNN_ERR_PARAM_NULLPTR`。不扫描输入 u 的值域；合法采样域为 [0,1)。
无广播，同输入要求确定性计算；执行在入参 stream 上异步进行，调用方同步后读回。

### 2.2 需求拆解与示例

精度标杆使用原包 CPU PyTorch FP64 变换后收窄的结果，性能标杆使用原包 NPU
拼接链。功能实现、完整精度、性能不劣化及内存占用分别验证。

例如 FP32 输入 `u=[0,0.25,0.5,0.75]`、lambda=2，输出依次为
`-log(1-2^-24)/2`、`-log(0.75)/2`、`-log(0.5)/2`、`-log(0.25)/2`。
首元素为 epsilon 保护的小正数；lambda=+inf 时输出全零。

## 3. 详细设计（required）

### 3.1 工程结构与 Host 流程

```text
experimental/math/FusedExponential/
├── CMakeLists.txt
├── docs/design.md
├── op_api/aclnn_fused_exponential.{h,cpp}
├── op_host/FusedExponential_{def,infershape,tiling}.cpp
├── op_host/fused_exponential_plan.h
├── op_kernel/FusedExponential.cpp
├── op_kernel/fused_exponential{,_tiling_data,_tiling_key}.h
├── examples/test_aclnn_fused_exponential.cpp
└── tests/test_plan.cpp
```

- Host：公共 L2 接收 DOUBLE aclScalar lambd，用 double 计算缩放参数后以 ListFloat
  属性送入内部算子，另传 zero_output 属性。公共接口签名不变；避免 Device FP64。
  连续 out 直接作为 kernel 输出，非连续 out 通过 executor 临时张量及 ViewCopy 回写。
  连续 out 先创建规范视图，保留指针和 offset，使 Tiling 使用逻辑长度，而非整个
  backing storage 长度。第一阶段完成参数校验、标量解析、空操作和执行计划构建。
- Tiling：读取真实 AIV 核数、UB 容量、元素数和 dtype；按 32B 块均衡分核；
  传递逻辑长度、每核块数、余块核数、tile 元素数、epsilon 和缩放因子。
- Kernel：每核负责独立连续区间，循环执行 CopyIn -> FP32 计算 -> CopyOut。
  核内使用输入/输出队列，计算中间数据仅放 UB，不生成 GM bool mask。
- lambda 缩放：常用范围将 `-1/lambda` 在 Host 用 double 计算后收窄为 float。
  极端有限 lambda 使用 frexp 分解为多个有限 FP32 缩放因子，避免倒数提前溢出。
  +inf 使用单独零输出标志，避免 `inf * 0`。

### 3.2 分块、容量与同步

设输入元素宽度 s 为 2 或 4B，块元素数 A=32/s，总块数 B=ceil(N/A)。
使用核数 C 不超过平台 AIV 核数和 B；初版每核目标至少 2048 个元素，减少小张量调度。
核 i 的块起点为 `i*(B/C)+min(i,B%C)`，前 `B%C` 个核多一个块。
最后一个区间按真实 N 截断，不因为对齐而读写用户缓冲区之外的数据。

输入和输出队列深度各为 2，另有 3 个 FP32 UB 临时张量。
每 tile 的预算为 `(4*s+12)*tileElements` 字节，另预留 1024B；tile 长度
向下对齐到 64 元素并上限 8192。低于最低预算的平台返回失败。
使用 DataCopyPad 搬运真实字节数，UB 容量对齐，GM 访问长度不向上扩大。
队列 EnQue/DeQue 完成搬运与计算之间的同步；有依赖的 Vector 指令显式 PIPE_V barrier。
不需要跨核同步。连续路径声明用户 workspace 为 0；框架实际返回量以原内存探针为准。

有偏移的连续视图及尾块须上卡检查，不能只凭相对分块对齐判断物理写入安全。
tile 长度为 0 或平台资源不足时返回失败；索引使用 64 位整数。
双缓冲仅是资源配置，其流水重叠及收益需实测。

初步小样中，空调用实际 workspaceSize=0，非空连续调用返回 512B。
该固定开销的用途和必要性仍需结合框架实现与原内存探针确认，依任务书
“特殊情况申请少量内存需要给出必要性说明”报告，不能把实际返回值写成 0。
非连续路径额外临时张量由 executor 管理，工作区随规模增长，须分别记录。
调用方在 stream 完成前保持输入、输出和 workspace 有效；缩放属性由 executor
创建的数组持有，不将局部栈数组作为异步参数保存。

### 3.3 Kernel 数值方案

指定 golden 保持原包 `function_FusedExponential.py`：FP64 计算后收窄输出。
Device 将低精度输入先精确扩展到 FP32，所有中间计算使用 FP32。
初版采用相消修正，避免 `1-u` 的 FP32 舍入在小 u 处放大相对误差：

```text
q = max(u, epsilon)
t = 1 - q
d = 1 - t
y = log(t) * (q / d)
out = y * (-1/lambda)
```

在合法输入范围 q>=epsilon，d 非零；当 t 精确时 q/d=1。
这只是候选数值实现，是否符合任务精度必须由原 golden 和全量测试确定。
输出使用最近舍入转回输入类型。同输入的确定性、特殊值、低精度转换均需上卡验证。

### 3.4 性能优化方向与风险

| 场景 | 设计及优化方向 | 需验证的风险 |
| --- | --- | --- |
| 小输入 | 减少启用核数，降低调度开销 | 分派阈值、空张量及标量语义 |
| 大连续输入 | 均衡分核、tile 内融合，减少 GM 流量 | 尾块、偏移、UB 容量与同步 |
| log 计算 | FP32 中间计算及相消修正 | 小 u、重尾、低精度舍入及极端 lambda |
| 非连续布局 | 框架连续化及输出回写 | stride/offset、原地别名、辅助内存与搬运成本 |
| 流水调整 | 根据 profiler 调整 tile、队列和同步 | 保留数据依赖；不能预先宣称提速 |

优化围绕一个有证据的瓶颈，保留失败候选与原始测量。出现开发问题时回查 learning
对应正文及示例；优化无进展时重新审视假设。任何实现、构建或验证依赖改变后，
重新冻结候选，从头完成全量精度，通过后才测试该候选性能。

## 4. 支持硬件与算子约束（required）

| 产品 | 构建配置 | 当前状态 |
| --- | --- | --- |
| Atlas A2 | ascend910b | 已在 910B3 构建及运行，完整验收未完成 |
| Atlas A3 | ascend910_93 | 已声明支持配置，尚未完成该平台编译与实机验证 |

采用 A2/A3 的 Ascend C 基础 Vector API。支持 FP16/BF16/FP32、ND、秩不超过 8、
标量、空张量、非连续视图和完全相同视图的原地调用；不新增随机数状态。
硬件支持声明、编译通过和实机验证通过分别记录。

## 5. 可维可测分析（required）

### 5.1 精度及功能验证

| 要求 | 原文 | 实现/检查位置 | 验证方式 | 当前状态 |
| --- | --- | --- | --- | --- |
| 两段式接口、dtype、shape | 2.3、2.4 | op_host / op_api | 符号、头文件、参数调用检查 | ABI 及 FP32 小样已验证，专项待补齐 |
| 非连续输入/输出、原地、空输入 | 2.5 | AutoContiguous / Kernel | 原包 + 接口专项 | 小样通过，原包 4 条工具失败待闭环 |
| FP64 golden，逐 dtype 容差 | 3.2、3.5.1 | 原包 golden 与 ATK | 原始 1000 条，从头全量 | 首轮 996 通过、4 条工具执行失败 |
| 匹配比例和最大误差 | 3.2 | 原包比较器 | 记录 ATK 版本与实际判据 | 待验证 |
| 无 GM 中间 mask / workspace | 3.4 | Kernel / Tiling | 原内存探针、源码审查 | 待验证 |
| 200 条性能不劣化 | 3.3 | 融合 kernel | 全量精度符合后运行原性能脚本 | 未开始 |
| 设计、README、七份报告及日志 | 4、5 | 本目录 / task_submission | 同源证据清单 | 设计稿就绪，其余材料待完成 |

精度阈值原样采用任务书：FP16 atol=rtol=2^-10，BF16 为 2^-7，FP32 为 2^-13；
matched_ratio >=0.99，且满足任务书最大误差限制。不能把 99% 元素比例当成只需
99% 用例通过。原包中的工具缺陷按任务书说明分析实际日志，不仅凭编号排除失败。

执行顺序：接口小样诊断 -> 核验测试加载的是当前库 -> 冻结源码/构建/依赖 ->
从头运行原 1000 条精度及有效补充项 -> 全部完成并符合 -> 原 200 条性能 ->
原内存自验 -> 同源材料归档。使用原分阶段脚本，不调用会在精度失败后继续性能的总入口。
任何实现、测试或依赖改变后重新全量精度。未通过、未比较、漏测和未结束都不进入性能。

核对比较器实际阈值，并注入错误输出验证其拦截能力。检查所有逻辑输出及 NaN/Inf
分类，保护区单独检查。补充覆盖非法 lambda、+inf、极端有限值、非连续输出、
原地调用、偏移视图、尾块、分派边界、连续调用和确定性。

### 5.2 性能与内存验证

精度全部完成后使用原 200 条性能子集。基线是 NPU neg/add/ge/mask_fill/log/mul 链，
按原脚本预热 10 次、采集 60 次；逐用例要求拼接链平均 Device 耗时除以融合耗时
不小于 1.0。计时边界及输入恢复遵循原脚本，不以 CPU golden、Host 总耗时或
最佳单次代表性能。内存按输入字节数、输出字节数和实际 workspaceSize 计量，
分别解释连续与非连续路径，正式性能及内存结论目前未产生。

### 5.3 当前证据与交付计划

候选 006 环境为 CANN 9.1.0、torch/torch_npu 2.7.1、ATK 26.9.8、910B3。
Host 分块属性测试和 31 条 FP32 接口小样通过，原 1000 条 CPU golden 完整生成。
原包精度为 996 条通过，960/966/969/971 四条在 ATK 的
`_build_non_contiguous_slice_view` → `aclnnInplaceCopy` 报 161002，未进入本算子。
任务书 §3.5.1 将该情形列为工具缺陷；保留具体堆栈并补足真实接口验证，不能
仅按 known_tool_defects 编号自动记为通过。

最终交付包含源码、README、API 文档、调用示例、原测试及补充用例、设计文档和
任务指定七份自验材料。保存源码/库哈希、构建命令、依赖、设备和原始日志，
使报告与受测版本一致。设计文档先在社区任务仓评审；代码完成验证后按任务书
提交至 `cann/ops-math`，分别记录文档评审、代码 PR 与验收状态。

## 6. 兼容性分析

新增接口保持任务书的 aclTensor/aclScalar 两段式签名。ListFloat/Int 属性只用于
内部传参，不增加公共参数。不替换已安装 CANN 包作为测试对象，通过独立 vendor
加载本实现。A2/A3 共享源码并按 SoC 构建，不继承参考 GEMM 的归约、工作区、
设备分工或性能指标。epsilon、布局和异常处理以当前任务为准。
