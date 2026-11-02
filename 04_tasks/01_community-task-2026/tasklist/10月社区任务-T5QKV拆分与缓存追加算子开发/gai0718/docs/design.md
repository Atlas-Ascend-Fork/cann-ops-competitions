# T5SplitQkvAppendCache 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 【xLLM】【2026年10月社区任务】T5QKV拆分与缓存追加算子开发 |
| taskId | `299045990ccf44b99caae895c11a3c58` |
| 目标仓库 | `cann/cann-ops-competitions` |
| 目标目录 | `04_tasks/01_community-task-2026/tasklist/10月社区任务-T5QKV拆分与缓存追加算子开发/gai0718/docs/design.md` |
| PR 标题 | `【CANN社区任务】T5SplitQkvAppendCache 算子设计文档` |
| 评审通知 | `@condfuse_3 @fullt @Ascend-CANN` |
| 文档状态 | 本地自测通过；社区设计评审及 xLLM 代码 PR 分别跟踪 |

官方已确认本任务按 Ascend910_9382 单 die、单逻辑设备验收：48 个 Vector Core、24 个 Cube Core、64 GiB HBM、192 MiB L2。Issue #62 的回复同时确认性能按 `max_wall_us` 绝对时间验收，每次调用都分配新输出并完整复制 past。GitCode 设计 PR #2526 位于 `gai0718/cann-ops-competitions` 的 `design/t5qkv-20261029` 分支；算子源代码及自测交付位于独立实现仓库，xLLM 合入由单独代码 PR 跟踪。

## 需求背景（required）

### 需求来源

本文对应“题目3-T5QKV拆分与缓存追加算子开发”任务，目标是在 Ascend 910C / Atlas A3 上使用 Ascend C 实现单 token 解码阶段的 QKV 拆分和 KV cache 追加，并提供 ACLNN、PyTorch NPU 和 Kernel 直调入口。

### 背景介绍

输入融合张量 `qkv` 按全部 Q、全部 K、全部 V 的顺序排列，形状为 `[B,1,3HD]`。算子完成三项操作：

1. 从 Q 段生成独立连续的 `query`，形状为 `[B,H,1,D]`；
2. 将 `pastKey` 与当前 token 的 K 拼接到新分配的 `key`，形状为 `[B,H,L+1,D]`；
3. 将 `pastValue` 与当前 token 的 V 拼接到新分配的 `value`，形状为 `[B,H,L+1,D]`。

算子只搬运 FP16/BF16 的 16 位存储单元，不执行浮点运算，因此输入中的正负零、Inf 和 NaN 位模式可以逐位复制到输出。

## 需求分析（required）

### 需求描述

实现单 token 解码阶段的 QKV 拆分与 KV cache 追加，提供 ACLNN、PyTorch NPU 和 Kernel 直调入口。

### 需求拆解

1. 拆分 Q、K、V，并将当前 token 追加到历史 Key/Value。
2. 支持 FP16、BF16、连续输入和分批 cache 复制。
3. 保持输入不变，每次都分配独立输出并完整复制历史数据。
4. 按任务要求的绝对 wall time 标准完成性能验证。

### 2.1 接口定义

| 参数 | 形状 | 类型 | 说明 |
| --- | --- | --- | --- |
| `qkv` | `[B,1,3HD]` | FP16 / BF16 | Q、K、V 三段连续排列 |
| `pastKey` | `[B,H,L,D]` | 与 `qkv` 一致 | 历史 Key cache |
| `pastValue` | `[B,H,L,D]` | 与 `qkv` 一致 | 历史 Value cache |
| `query` | `[B,H,1,D]` | 与输入一致 | 独立连续输出 |
| `key` | `[B,H,L+1,D]` | 与输入一致 | 历史 Key 加当前 token |
| `value` | `[B,H,L+1,D]` | 与输入一致 | 历史 Value 加当前 token |
| `headNum` | 标量 | int | 必须等于 H |
| `headDim` | 标量 | int | 必须等于 D |

取值范围为 \(1\leq B\leq64\)、\(1\leq H\leq128\)、\(1\leq D\leq256\)、\(0\leq L\leq8192\)，所有输出元素总数不超过 \(2^{28}\)。输入位于同一 NPU、连续、类型一致且布局为 ND。输出位于同一 NPU、连续且各自存储区间独立。

### 2.2 计算语义

令 `row = b*H+h`，`P=L*D`。QKV 的每个 batch 内部按 Q、K、V 三段排列，因此：

```text
Q 源元素：qkv[(3*b*H + h)*D + d]
K 当前 token：qkv[((3*b + 1)*H + h)*D + d]
V 当前 token：qkv[((3*b + 2)*H + h)*D + d]
```

输出地址按连续的 `[B,H,T,D]` 布局计算：

```text
query[b,h,0,d] = qkv[(3*b*H + h)*D + d]
key[b,h,t,d]   = pastKey[b,h,t,d]       , 0 <= t < L
key[b,h,L,d]   = qkv[((3*b + 1)*H + h)*D + d]
value[b,h,t,d] = pastValue[b,h,t,d]     , 0 <= t < L
value[b,h,L,d] = qkv[((3*b + 2)*H + h)*D + d]
```

当 \(L=0\) 时，Key 和 Value 只写入当前 token。\(L>0\) 时，每次将全部历史数据复制到本次新输出。

### 2.3 入口和错误处理

- ACLNN 第一阶段接口 `aclnnT5SplitQkvAppendCacheGetWorkspaceSize` 检查指针、shape、dtype、连续性、属性范围和输出形状，创建 executor 并返回 workspace 大小；第二阶段接口在调用方 stream 上异步执行。
- `run(inputs, attributes)` 创建三个独立 NPU 输出。首次元数据组合调用 ACLNN 第一阶段并将 executor 设置为 repeatable，后续调用更新变化的地址并执行第二阶段。
- `run_direct(inputs, attributes)` 使用同一套元数据检查，直接调用生成的 Ascend C launcher。
- `prepare(inputs, attributes)` 在真正下发前完成 Python 侧参数检查，非法属性、非法 rank、CPU 输入、非连续输入和尺寸越界均返回错误。
- 输出、输入和 workspace 的生命周期覆盖当前 stream 上的全部异步操作；算子不修改输入。

## 详细设计（required）

### 算子分析

#### 支持数据类型

输入、历史 cache 和输出支持 FP16 与 BF16，计算路径按 16 位存储单元逐位搬运。

#### 支持形状

主接口支持 \([B,1,3HD]\) 的 QKV、\([B,H,L,D]\) 的历史 cache，以及对应的查询和追加后 cache 输出；具体取值范围见接口定义。

#### 算子约束限制

输入必须连续、位于同一 NPU、类型一致且布局为 ND；输出必须连续并相互独立。多 token、GQA、RoPE 和分页 cache 不在本接口范围内。

### 3.1 工程结构

```text
t5qkv_operator/
├── op_host/
│   ├── t5_split_qkv_append_cache_def.cpp
│   ├── t5_split_qkv_append_cache_infershape.cpp
│   ├── t5_split_qkv_append_cache_tiling.cpp
│   └── t5_split_qkv_append_cache_tiling.h
├── op_kernel/
│   ├── t5qkv_impl.h
│   └── t5_split_qkv_append_cache.cpp
├── torch_extension/
│   └── t5qkv_bridge.cpp
├── direct/
│   └── t5qkv_direct.cpp
├── examples/aclnn_call.cpp
└── solution.py
```

工程通过 CANN `custom_op` 模板生成 ACLNN 两阶段 API、算子协议、tiling 注册和 `aclrtlaunch` 声明。目标 compute unit 为 `ascend910_93`，构建参数使用 `ASCEND_COMPUTE_UNIT=ascend910_93`，对应任务书指定的 `Ascend910_9382`。

### 3.2 Host 侧 Shape、Dtype 和 Tiling

Shape 推导从 `pastKey` 读取 \(B,H,L,D\)，生成三个输出 shape：`[B,H,1,D]`、`[B,H,L+1,D]` 和 `[B,H,L+1,D]`。输入和输出 dtype 统一为 qkv 的 FP16 或 BF16。

TilingData 保存 `batch`、`heads`、`length`、`headDim` 和任务总数。Q 按每个 batch 的 \(H\times D\) 元素切成最多 32768 个 16 位元素的任务，K/V 按缓存行切 tile 并追加当前 token。Block 数取任务数与 48 个 Vector Core 的较小值。显式全局 workspace 为 0。

### 3.3 Kernel 设计

Kernel 使用两个 64 KiB UB tile，按 GM→UB→GM 搬运。分支依据输入形状和工作集大小选择：

| 条件 | 分工与缓存策略 |
| --- | --- |
| \(L=0\) | Q/K/V 按每个 batch 的连续 \(H\times D\) 区域搬运 |
| \(LD\ge65536\) 且 \(BH\ge48\) | 将 K/V 行交错编号，按核心步长分配整行，每行分 tile 并追加 token |
| 其他非空历史 | 通用的 Q、历史 tile、追加 token 任务映射 |
| 单份历史缓存元素数 \(BHLD\ge8\times1024^2\) | pastValue 使用 L2 bypass；pastKey 按 64 KiB tile 交错选择普通缓存与 bypass，保留约一半历史 Key；输出使用普通缓存 |

L2 bypass 改变搬运使用的缓存路径。每次仍逐字节读取完整历史并写入全新的输出。pastKey 的偶数 tile 使用普通缓存、奇数 tile 使用 bypass，将缓存集合分散到整个 Key；大尺寸用例保留约 32 MiB 输入和 128 MiB 输出。K/V 保持交错整行分工。

大缓存搬运先检查实际 GM 起址加元素偏移后的 32 字节对齐，以及元素数为 16 的倍数；满足时使用普通 `DataCopy`，其他起址和尾块按实际字节数使用 `DataCopyPad`。输入和输出分别检查，因此带 storage offset 的连续输入仍保持正确边界。FP16 和 BF16 使用同一个 `uint16_t` 搬运路径。

Q、K、V 三个输出分别由独立任务写入。历史数据和追加数据的目标区间不重叠，多个 block 不写同一输出元素。

### 3.4 ACLNN 与 Kernel 直调

ACLNN 第一阶段将输入输出描述符和属性登记到 executor，调用框架生成的 AICore launcher；第二阶段调用 `CommonOpExecutorRun`，stream 由调用方传入。ACLNN 输出使用普通 REQUIRED 输出描述，不采用输入 Ref 或原地 cache 语义。

Python bridge 按线程缓存一个 repeatable executor，键包含完整形状、dtype、device 和 stream。地址变化时调用 `aclSetInputTensorAddr`、`aclSetOutputTensorAddr`；每次调用均重新执行第二阶段内核。同地址输入在设备上改写后也会重新读取。

Kernel 直调入口使用同一份 `T5SplitQkvAppendCacheTilingData` 和同一份 Ascend C kernel，实现独立 launcher。直调调用方负责构造 ND tensor 描述、分配不重叠的设备输出，并在调用方 stream 上同步。

### 3.5 内存和生命周期

算子不申请显式全局 workspace。片上 UB 使用两个 64 KiB tile，输出字节数为主要新增设备存储。调用方每次重新分配输出；释放输出后同步 stream，再检查活跃张量分配回到调用前水平。

Python bridge 每次分配新的设备块，将其分成三段互不重叠且有边界填充的 StorageImpl，并设置完整 ND 元数据。三个 StorageImpl 通过共同所有者保持设备块有效，任一结果保留时其他段也维持有效。

输入、外部预分配输出和 workspace 登记 stream 使用关系。当前流新分配的输出使用分配器自身的流序回收，避免额外完成事件延迟释放。输出再次作为另一条 stream 的输入时，bridge 将共同所有者登记到消费流。连续视图按逻辑起始指针建立 ACL 描述符。

### 支持硬件

| 项目 | 版本 |
| --- | --- |
| 硬件 | Ascend 910C / Atlas A3，Ascend910_9382，单 die、单逻辑设备；48 Vector Core、24 Cube Core、64 GiB HBM、192 MiB L2 |
| CANN | 9.1.0 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 模块版本 `2.10.0.post4`；2026-11-02 实测完整 git revision 为 `5dd8ef3f9b375b5ae4a83538d5785754148c3302`，对应任务指定 `git5dd8ef3` 构建 |
| NumPy | 2.2.6 |
| ml_dtypes | 0.6.0 |

## 可维可测分析

Issue #62 评论 197308996、197309122、197309170、197309272 确认按 `max_wall_us` 绝对阈值验收。每次调用为 Q/K/V 分配相互独立的新输出，逐位复制全部 `pastKey/pastValue` 历史前缀，再写入当前 token；即使地址或内容与先前调用相同，也执行完整复制。缓存策略按设备能力选择。

### 精度标准/性能标准

#### 精度标准

使用任务包 `generate_data.py` 的固定种子生成 35 个公开用例，使用 `golden.py` 计算 CPU 标杆。FP16/BF16 输出按原始 16 位编码逐位比较，同时检查 shape、dtype、连续性、设备、输入不变和存储独立性。额外覆盖 12 个固定种子和 storage offset 用例。

### 兼容性分析

验证器逐项覆盖 28 类非法输入和属性场景，并分别执行 `prepare`、`run`、`run_direct`。工程测试覆盖两个独立 stream、重复调用回收、提前释放压力、输出生命周期、连续追加，以及新增的地址交替、输入原地改写、输出作为另一流输入后提前释放、大缓存输入连续改写。共 196 条记录。

#### 性能标准

原始 benchmark 固定预热 50 次、5 轮、每轮 200 次，取轮均值中位数；计时包括新输出分配、完整历史复制、ACLNN 提交和完成同步。正式重建后独立三组完整 50/200/5 均为 10/10：007 为 71.979/71.648/71.925 微秒，017 为 71.580/71.691/71.134 微秒；最小余量分别为 3.021 和 5.309 微秒。此前同方案另三组完整测量也全部通过，连续六组全部达标。

正式重建、精度 35/35、工程 196/196、原生 70/70 和实际大页/活跃分配回收均通过。全部原始五轮及失败历史在 `reports/`，详见 `docs/final_validation.md`。

## 6. 构建和安装

```bash
bash build_all.sh --python python3 --jobs 4 --run-tests --device 0
source private_install/activate_t5qkv.bash
```

安装路径默认是工程内 `private_install/`，通过 `--install-prefix` 指定。激活脚本强制设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`TASK_QUEUE_ENABLE=0` 和 `PYTORCH_NPU_ALLOC_CONF=page_size:1g`。先激活环境再启动 Python，使大页分配配置在分配器初始化时生效。Python bridge 在安装包生成的 ACLNN 头文件和 launcher 头文件基础上编译，独立 C++ 样例位于 `examples/aclnn_call.cpp`。

复现和性能测量均固定上述四个环境变量；构建、激活和复测脚本覆盖调用 shell 中继承的值。大页模式属于 torch_npu 标准分配器配置，原始性能脚本、输入定义和绝对阈值保持原样；报告保存该配置、是否回退及活跃分配峰值。环境采集在独立短进程完成后退出，性能控制器保持无 NPU 上下文。

## 7. 变更范围

本设计覆盖任务书规定的单 token、MHA 布局和连续 cache。多 token、GQA、RoPE、分页 cache 和原地写回不属于当前接口。性能按每个用例的 `max_wall_us` 绝对阈值验收；正式验收口径和完整 past 复制语义已由 Issue #62 维护者确认。完整测试结果与社区评审状态记录在版本化交付材料中。

## 8. AI 辅助说明

- AI Agent：OpenAI Codex
- AI Model：GPT-6
- 辅助范围：读取任务书和 CANN 参考工程，整理接口、地址推导、tiling、kernel、生命周期和验证方案。
- 本文未包含凭据、服务器地址或未核实的验收结果。

