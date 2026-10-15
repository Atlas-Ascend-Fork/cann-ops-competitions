# Wan2.2-I2V TaylorSeer、Offload 与 Prefetch 设计说明书（Atlas A3）

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 10 月社区任务：Wan2.2-I2V TaylorSeer、Offload 与 Prefetch（A3） |
| 贡献者 | Louis（GitCode：`Louislu68888`） |
| 设计状态 | RFC，待社区评审；本文定义实现与验收合同，不声明功能、质量或性能已经通过 |
| 关联 RFC Issue | [cann-recipes-infer #430](https://gitcode.com/cann/cann-recipes-infer/issues/430) |
| 代码仓 | [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) |
| 设计仓 | [cann/cann-competitions](https://gitcode.com/cann/cann-competitions)，`04_tasks/01_community-task-2026/tasklist/` |
| 调研基线 | `cann-recipes-infer`：`bbd278fe5ef772daa310df680865a6826ae6d909`；代码实施前复核上游变化 |
| 设计模板基线 | `cann-competitions`：`83aeaa5887560b5d0036d01f2e2ae3b0d2b1a469`，采用社区任务设计模板的四个主体章节 |
| 目标平台 | Atlas A3，单卡；CANN ≥ 8.5.0、PyTorch 2.7.1、torch_npu v7.3.0，Python 建议 3.12 |
| 本次交付 | RFC 与设计文档 PR；代码 PR、A3 实测与验收报告为后续阶段 |

## 1. 需求背景

### 1.1 需求来源与范围

需求依据为《Wan2.2-I2V TaylorSeer、Offload 与 Prefetch 任务书（A3）》及其 `test_cases/README.md`。接收版本的 SHA-256 分别为 `f6bf5b500e0e223cc13a22f300bd022b836425ea3889a85989e65d4070af240d`、`b099ef71770a4fd1b106facfe6d32cd4293b7e6ad2d0a2422ec6e01095075730`。这些哈希用于固定需求版本，不代表模型权重或运行结果。

本任务将仓内已有 TaylorSeer 数学能力接入 Wan2.2-I2V A14B，在不改变生成规格的前提下，通过减少 DiT 完整计算、缓存 Host offload 和异步 H2D prefetch，争取达到端到端平均时延降低至少 50% 的验收要求。论文和其他设备上的加速比不能作为本任务结果。

本期不新增独立 kernel、aclnn 接口、量化、Attention 替换或并行策略；不扩展多卡与 DiT-Cache 组合；不提交权重、输入图片、输出视频或大型 profiling 文件。既有 Wan 模型的增强仍放在 `models/wan2.2-i2v/`，不按“新增裸模型”迁移到 `integration/custom/`。新 LLM Checklist 中 KV Cache、Packed Sequence、在线服务与图模式等不适用项应注明原因。

### 1.2 已核实的仓库事实

以下事实均来自调研基线的源码，文件链接见第 4.8 节。

| 位置 | 现状 | 设计影响 |
| --- | --- | --- |
| `module/dit_cache/cache_method.py` | `CacheManager.from_config` 能选择 TaylorSeer，但 Wan 入口仅传 `num_steps`，Taylor 构造还需要 `double_stream_layers`、`single_stream_layers` | 只改 `method` 会遇到缺失层数，必须增加 Wan 适配层 |
| `wan/cache/cache_block.py`、`generate.py` | 只为两个专家的第 0 个 block 安装 TeaCache/FBCache 包装 | Taylor 不能沿用首块跳过整个 DiT 的控制语义 |
| `wan/modules/model.py` | 首块后若 `should_skip` 为真则退出整个 block 循环，循环结束调用 `post_cache_update` | 新适配器须明确实现兼容协议，防止误跳后续组 |
| `wan/image2video.py` | 实际 denoising 循环每步选择 high/low 专家；单卡顺序调用 cond、uncond，再组合 CFG 并执行 scheduler | 步号由外层循环提供，不能按 block 数或 CFG 调用次数推算 |
| `generate.py` | 先执行 `sampling_steps=2` 的预热，再进行正式生成，缓存管理器在此前创建 | 每次 `generate` 必须重新建立请求状态，不能把预热历史带入正式生成 |
| 公共 Taylor 导数函数 | 第一次完整计算因历史不足直接返回；默认存在 offload 流，但缺少本任务的有界 Host 池及下一组预取协议 | 适配器显式初始化零阶项；异步生命周期由模型侧搬运组件负责 |
| `14b_single.yaml` 与参数解析 | 单卡未显式配置 `offload_model` 时，入口将其解析为 `true` | 消融中的“offload 开关”指新增缓存管理，不应误称原始基线完全无模型 offload |

现有 `_prepare_model_for_timestep` 的非活动模型迁移条件使用 `device.type == 'cuda'`，入口同时加载 `transfer_to_npu`。设计阶段仅确认这个源码事实；A3 上的实际设备类型、模型驻留与搬运须测量，不能据此声称已有 NPU offload 完全正确。

## 2. 需求分析

### 2.1 固定支持矩阵与实验基线

| 维度 | 首期合同 |
| --- | --- |
| 模型/权重 | `i2v-A14B`、Wan2.2-I2V-A14B-BF16，两个专家权重 revision 和每个分片 SHA-256 在首次运行前固定 |
| 生成规格 | 配置 `size: "640*360"`、`frame_num: 61`、`sample_steps: 40`、`base_seed: 100`，相同输入图、正负 prompt、solver、shift、guide scale |
| 实际形状 | 记录经输入宽高比、VAE stride、patch 对齐得到的实际宽高、latent 和 token shape；四档完全相同，不能仅比较 YAML 的 size 字符串 |
| 并行 | 单进程、单逻辑设备，`world_size=cfg_size=ulysses_size=ring_size=tp_size=1`，不启用 DiT/T5 FSDP 或 VAE 并行 |
| CFG | 单卡 cond、uncond 顺序执行，独立历史；`enable_separate_cfg: true` |
| 数值 | 既有 BF16 模型计算；组残差、Taylor 因子及 Taylor 累加使用 FP32；返回时转换为组入口张量 dtype，不改变模型其他 autocast 规则 |
| 缓存张量 | 连续稠密 `[B,S,C]`；首期 B=1，S/C 从实际模型与输入取得，非连续入口通过显式连续副本处理并计入预算 |
| 请求 | 同一进程单 pipeline 串行生成；并发或重入在任何异步任务入队前报错 |
| 其他组合 | A2/950、多卡、动态图捕获、不同精度权重、并行 CFG 等不在本次支持声明内；明确拒绝不支持的 Taylor 组合 |

当前未取得属于本任务的 A3 实测环境、权重与样本，因此实机型号、可用 HBM/Host 容量、OS、驱动/固件、实际安装版本、权重 hash、输入 hash、基线耗时均标为**待实测固定**。它们是性能实验的进入条件，不能用另一项任务的机器或历史结果代填。参考软件组合取任务书和 Wan 专属 README，不用仓库面向其他模型的通用推荐版本覆盖。

### 2.2 需求追踪

| 编号 | 要求 | 设计落点 | 验证入口/证据 |
| --- | --- | --- | --- |
| R1 | RFC 固定基线、版本、输入、配置和计时口径 | 2.1、4.2 | RFC 环境与实验 manifest |
| R2 | full/predict、导数、warmup/cutoff 接入 Wan | 3.1—3.3 | 调度单测、组残差测试、逐步 JSONL |
| R3 | 配置公开、非法参数快速失败 | 3.4 | YAML 参数化异常测试，非零退出码 |
| R4 | request/expert/CFG 隔离 | 3.2、3.5 | 专家边界、预热后正式生成、连续请求 |
| R5 | 缓存及非活动模型 offload、释放与复用 | 3.5—3.7 | 搬运计数、驻留检查、Host/NPU 内存曲线 |
| R6 | 异步 prefetch、event wait、有界 buffer | 3.6 | 延迟注入、顺序扰动测试与 A3 时间线 |
| R7 | 与 `npu_prefetch` 区分 | 3.6.4 | 配置说明及默认关闭核验 |
| R8 | 四档消融、平均时延降低 ≥50% | 4.2—4.3 | 每档 ≥1 次预热、≥3 次正式生成及原始统计 |
| R9 | 最终质量、稳定性与异常清理 | 4.1、4.4 | ≥5 组样本、2 人盲测、连续 ≥5 次生成、中断 |
| R10 | 回退、现有方法与公共调用方回归 | 4.5 | NoCache/FBCache/TeaCache、HunyuanVideo 回归 |
| R11 | 文档、License、CI 与交付材料 | 4.6—4.7 | 代码 PR、检查日志、验收报告与附件索引 |

## 3. 详细设计

### 3.1 模块划分与接入方式

复用 `CacheManager` 的配置与方法容器，复用 `TaylorSeer` 的有限差分及 Taylor 公式；Wan 的专家选择、CFG、请求状态和搬运队列放在模型目录。首期方案不修改 `module/dit_cache/`。

| 计划文件 | 职责 |
| --- | --- |
| `wan/cache/taylorseer.py` | `configure_wan_cache`、`WanTaylorContext`、组级 Taylor adapter、full/predict 与统计 |
| `wan/cache/cache_transfer.py` | 普通 Host 缓存、有界 pinned staging、NPU slot、copy streams、events、预算与 drain |
| `wan/cache/__init__.py` | 模型侧初始化与包装入口导出 |
| `generate.py` | 加载并严格校验配置，选择旧方法或 Taylor 包装，统一计时与报告 |
| `wan/image2video.py` | 真实 scheduler 步长、request 生命周期、expert/CFG 上下文、模型迁移及 `finally` 清理 |
| `config/14b_taylorseer.yaml` | 从 `14b_single.yaml` 复制全部字段，增加 Taylor 配置；保留原单卡基线文件 |
| 模型 README、`config/README.md`、`docs/models/wan2.2-i2v/taylorseer.md` | 启用、参数、原理、限制、回退与排障 |
| `models/wan2.2-i2v/tests/` | 拟新增无权重逻辑测试及 A3 集成、自验脚本；最终遵守仓库测试发现规则 |

`configure_wan_cache` 先检查方法枚举和原始 YAML，再调用已有加载器。旧三种方法继续走既有 `CacheManager.from_config` 与首块包装；Taylor 分支安装实现同一容器协议的 Wan façade，不能先构造缺少层数的公共 Taylor，也不能先给首块装 Tea/FBCache wrapper 后再叠加 Taylor wrapper。

每个专家、CFG 分支对应独立 Taylor 数学实例，显式给出 `double_stream_layers=0`、`single_stream_layers=G`。这里 `single_stream[group_id]["total"]` 只作为公共算法的存储槽映射，**不表示 Wan 具有公共实现中的 single/double-stream 架构**。实例的公共 `offload` 设为 `false`；模型搬运组件在调用数学函数前提供已经就绪的 NPU 因子，不复用公共类内部的异步 offload 路径。

façade 的 `should_skip` 始终为 `false`，`post_cache_update` 为 no-op；组出口负责更新，不让 `WanModel.forward` 提前结束整个 block 循环。模型级 embeddings、timestep embedding/projection、输出 head、unpatchify、CFG 合成与 scheduler 每步都执行；组内 block 的 modulation 运算属于组残差近似的覆盖范围，predict 时随原 block 一同跳过。无需新增 tiling、workspace C ABI 或独立 kernel，相关模板章节为 N/A。

模型侧拟提供以下私有接口，不新增公共 ABI；异常向入口传播，不返回可被误当作有效预测的空 tensor。

| 私有接口 | 输入/输出与生命周期 |
| --- | --- |
| `configure_wan_cache(config_path, pipeline) -> WanTaylorContext` | 读取配置与专家结构；旧方法沿原路径初始化；不在配置阶段建立跨请求历史 |
| `begin_request(request_id, timesteps, shape_signature) -> None` | 独占 pipeline，T 来自实际 timesteps，清空状态、解析预算；timesteps 只读 |
| `begin_step(s, expert_id) -> StepPlan` | 一步一次；生成两个 CFG 共用的 full/predict 计划、绑定局部 k；StepPlan 只读 |
| `begin_branch(is_cond: bool) -> None` | 显式选择分支；不可由 block 自行翻转 |
| `group_forward(group_id, x, original_blocks, model_args) -> Tensor` | 借用模型输入；full 输出原 block 结果，predict 输出独立结果；缓存拥有 detached 残差/因子，不借用可写输入 |
| `finish_step() -> None` | 两分支成功后推进一次 k 和统计，拒绝不完整组遍历 |
| `close(error=None) -> Statistics` | 停止提交、drain、释放请求资源；可重复清理而不重复释放，保留首次异常及统计快照 |

### 3.2 缓存单元、索引与所有权

#### 3.2.1 连续 block 分组残差

全模型每块独立缓存 FP32 高阶因子会使 Taylor-only 的显存成本过高，故拟使用连续 block 组作为缓存单元。设置 `cache_group_size=g`，总层数 L 来自 `len(model.blocks)`，`G=ceil(L/g)`，第 j 组包含半开区间 `[j*g, min((j+1)*g,L))`。建议起点 `g=4`，尾组可不足四层；g=1 保留逐 block 对照能力。分组是待验证的 Wan 适配选择，不能宣称已达到所需质量或最优性能。

full 步完整运行组内所有原始 block，保存 `F_j(s)=float32(x_out)-float32(x_in)`。predict 步在组首返回 `cast_to_input_dtype(float32(x_in)+F_hat_j(s))`，同组其余 block wrapper 恒等返回；仍正常遍历后续组。预测值不进入导数历史。组内无需保存中间 block 的历史，组入口副本不能与可写输出共享存储。

三个 Taylor 消融档必须使用相同 g、阶数、调度、计算/缓存 dtype。改变 g 需要更新 RFC 配置记录并重跑三档及质量测试；不得在某档隐式少缓存、换低精度或改变生成规格。

#### 3.2.2 状态主键

主键为 `(request_id, expert_id, expert_epoch, cfg_branch, group_id)`；记录模型权重指纹、shape/dtype/device、组边界、最新 full 步号、实际最高阶、buffer generation。`expert_id ∈ {high, low}`，`cfg_branch ∈ {cond, uncond}`。原始 block_id 与 group_id 同时记录，不能把 CFG 调用数当作 denoising step。

每次 `generate` 获取单 pipeline 执行锁并分配新 request；从实际 scheduler 获取 T，再创建空缓存。两步预热也是独立请求。专家首次激活或重新激活时增加 epoch、重置局部步号与导数历史，首次调用强制 full；即使采样器以后出现 high→low→high，也不恢复旧 epoch 的预测历史。当前单调 scheduler 下退役专家的历史在最后消费完成后直接释放，避免两专家成熟缓存同时常驻。

同一步 cond/uncond 用独立缓存但共用 full/predict 决策；调用顺序由外层显式 `begin_branch` 指定，禁止依赖首块 `cfg_step` 翻转。状态形状或权重指纹不匹配时使该请求失败并清理，不能把其他请求张量当作一次正常缓存命中。受支持的全新输入在下一请求重新建池。另在每次请求开始核验两个专家的 grid/shape 签名；`WanModel.forward` 仅在 `freqs_list is None` 时生成依赖 grid 的 RoPE 缓存，因此形状变化时必须在旧任务 drain 后清空并重建 `freqs_list` 等形状相关非 parameter 缓存，不能只重置 Taylor 池。该通用正确性修正同样应用于四档；测试覆盖不同宽高比连续请求。

### 3.3 调度与数学合同

令 s 为当前请求的实际 denoising 索引，范围 `[0,T)`；k 为当前 expert epoch 内的局部索引；w、K、c 分别为 warmup、skip interval、cutoff。候选 full 条件为：

```text
full = (k < w) or ((k - w) % K == 0) or (s >= T - c)
       or (当前专家刚激活) or (任一所需组/CFG没有有效零阶历史)
```

K 是相邻周期 full 的步距，K=3 表示中间最多两步 predict；k=w 本身也 full，延续公共实现的区间定义。w 按每个专家 epoch 生效，c 指全请求最后 c 步；不是最后 c 次 CFG 调用。请求总步数以实际 scheduler 长度为准；两步预热时 w=2、c=2 的并集使所有步 full，不能沿用正式生成的 T=40。若任一缓存缺失，在该步开始前将所有组及两个 CFG 分支统一提升为 full，并记录原因。

例如未发生专家切换且 T=10、w=2、K=3、c=2 时，full 索引为 `0,1,2,5,8,9`，predict 为 `3,4,6,7`。如专家从 s=5 开始切换，新的 k 从 0 起，s=5、6、7 都为 full。正式 40 步的完整表由实际 timestep/boundary 推导并存入日志，不能预填专家分界步。

公共数学实例的 `current["step"]` 与 `activated_steps` **统一存专家 epoch 内的局部 k**；全局 s/timestep 只用于请求 cutoff、外层调度和日志。这样 `current.step > warmup-2` 的升阶门槛在每个专家均遵循局部 warmup。数学单位是 denoising 索引，以下 a 与预测自变量均为局部索引（单 epoch 内与全局差值相等）。最新两个 full 步为 `a_prev`、`a_new`，`Δ=a_new-a_prev>0`；只在 full 更新：

```text
D_new[0] = F(a_new)
D_new[r] = (D_new[r-1] - D_prev[r-1]) / Δ
F_hat(k) = sum(D_last[r] * (k-a_last)^r / r!, r=0..q_eff)
```

以上有限差分递推与仓内实现一致，不把它宣称为解析高阶导数或非均匀采样下的精确多项式拟合。初次 full 显式保存 `D[0]`，修补公共函数首次直接返回造成的空历史；随后复用 `derivative_approximation`、`taylor_formula`。最高配置阶为 q，最多 q+1 份因子；历史不足时按公共 warmup 条件逐阶增长，使用已有的 `q_eff≤q`，不伪造高阶零值为“已有历史”。q=2、w=2 的首次三次连续 full 可逐步得到 0、1、2 阶。

外层 `begin_step` 一次性绑定全局 s、局部 k、局部 activated_steps 与 full/predict 类型，所有组运行结束且两分支成功后才提交该步统计。对公共 `_get_step_type` 的副作用做模型侧隔离，不使用 `update_layer_counter` 推进实际步号，避免 CFG 翻倍计数及最后一层预选 `s=T`。若计算中断，该步标为 aborted，不计入完整生成结果。

### 3.4 配置与快速失败

`14b_taylorseer.yaml` 的模型字段完整复制 `14b_single.yaml`，明确固定 `model_args.offload_model: true`，Taylor 配置建议起点如下。预算 `null` 表示按下述公式推导并在运行日志打印解析值，不表示无限额。

```yaml
dit_cache:
  method: "TaylorSeer"
  enable_separate_cfg: true
  params:
    n_derivatives: 2
    skip_interval_steps: 3
    warmup: 2
    cutoff_steps: 2
    offload: false
    prefetch: false
    prefetch_distance: 1
    cache_group_size: 4
    max_inflight: 2
    host_cache_budget_bytes: null
    pinned_budget_bytes: null
    device_cache_budget_bytes: null
```

| 参数 | 精确语义与校验 |
| --- | --- |
| `method` | 只允许 NoCache/FBCache/TeaCache/TaylorSeer；拼写错误不能落入公共 loader 的 NoCache 兜底 |
| `n_derivatives` | 整数，首期支持 0、1、2、3，拒绝 bool、负数、浮点和更高阶；0 为零阶复用对照 |
| `skip_interval_steps` | 整数 K≥1；K=1 全 full，作为等价性测试 |
| `warmup` | 整数 w≥1；按实际请求执行 `min(w,T)` 次专家局部 warmup，w≥T 时全 full |
| `cutoff_steps` | 非负整数 c；实际使用 `min(c,T)`，c=0 不追加尾部 full；w/c 可重叠 |
| `offload` | 严格布尔；控制新增 Taylor 缓存向 Host 驻留与有界搬运，不替代已有 `offload_model` |
| `prefetch` | 严格布尔；只有 Taylor 且 offload=true 时可开启，否则报错 |
| `prefetch_distance` | 首期仅支持整数 1，单位为当前专家/CFG 遍历中的下一个缓存组；不跨 denoising step，不跨专家猜测未来内容 |
| `cache_group_size` | 整数 `1≤g≤L`，读取两个专家结构后分别校验；若结构不同，建立各自组映射 |
| `max_inflight` | 首期固定 2 个缓存组 NPU slot（当前消费、下一组），超出支持值报错；D2H 队列另设固定单组上限 |
| 三个 `*_budget_bytes` | `null` 或正整数，不接受 bool；单位 byte，分别限制普通 Host 因子、pinned staging、NPU 缓存相关活跃张量；解析后均为有限上限 |

校验顺序：YAML 类型/未知字段/枚举→方法和开关组合→并行及运行模式→模型结构与组映射→实际 shape/内存预算→分配与首次推理。错误包含配置键、实值及允许范围，抛出 `ValueError` 并由入口非零退出；容量不足明确报出请求量和上限。NoCache 带 Taylor 专属 params、prefetch 无 offload、字符串假布尔、多卡 Taylor 都不能静默忽略。npu_prefetch 不列入首期启用选项。

### 3.5 请求与专家生命周期

| 状态 | 进入动作与不变量 | 退出条件 |
| --- | --- | --- |
| NEW | 获取执行锁，建立 request、配置快照、shape 和专家计划；缓存为空 | 预算和结构校验完成 |
| ACTIVE | 当前 expert/epoch、CFG、group 唯一；生产者和消费者通过事件通信 | 所有 scheduler 步完成，或切换专家/出现异常 |
| SWITCHING | 停止给旧专家派新搬运，等待其最后 use 与 D2H/H2D；释放退役缓存和临时引用 | 非活动模型已迁回 Host，活动模型就绪，新专家 history 为空 |
| DRAINING | 禁止入队，等待已提交任务，汇总结果并保存统计快照 | stream/event/slot 不再被任何任务使用 |
| CLOSED | 释放缓存、staging、events 和请求引用，释放锁；保留小型统计快照供入口打印 | 下一次调用新建 request |
| FAILED | 记录首个异常与失败阶段，进入同一清理流程；不可将部分结果记 PASS | 清理成功后可新建请求；设备错误导致 drain 失败时进程非零退出 |

模型切换保持既有串行迁移策略，首期不预取整套下一专家权重。Taylor 路径使用实际 NPU 设备判定，清理/迁移必须覆盖 parameters、registered buffers，以及 freqs 等模型额外持有的设备张量，验证后才将模型标为 HOST。不能在 copy/compute 尚引用权重时调用迁移，也不能用 `empty_cache()` 代替释放活跃引用。

模型 offload 和 Taylor 缓存 offload 各自报告搬运量。四档固定相同既有模型 offload 设置；所有模型驻留相关修正（设备类型判断、非 parameter 张量的迁移/释放等）都先单独记录原始上游 B0 与修正后 B0，再把相同修正应用于四档。不能只进入优化档以夸大缓存收益；无缺陷时保留原模型迁移语义。

请求异常采用 `try/finally` drain；中断时先停止提交、完成已提交事件、再释放所有权。设备上下文已失败时不尝试把部分迁移模型重新用于下一请求，保留失败并要求重建进程。包装原始 forward 只安装一次，保存恢复引用，关闭方法或销毁 pipeline 时在 drain 后恢复，防止反复包装累积。模型实例可复用，缓存内容不能跨请求复用。

### 3.6 Offload、Prefetch 状态机与同步

#### 3.6.1 存储与所有权

offload=false 时，当前专家两个 CFG 的系数常驻 NPU。offload=true 时，完整系数存普通 CPU 内存，使用有界 pinned staging 完成 CPU↔NPU 搬运。只锁页当前消费/预取所需的少量缓存，不把所有高阶历史都设为 pinned。

transfer manager 唯一拥有 Host 正本、staging、NPU slot 和每个 generation 的事件；Taylor 实例在计算期间只借用就绪因子，调用后移除引用。一个 group bundle 含该组的 0..q_eff 阶因子。两个 H2D staging/NPU slot 分别用于当前组、下一组；额外一个 D2H staging 用于更新后的 bundle。Host writer 按固定顺序处理 D2H 完成后的 CPU 复制，队列上限为 1，禁止后台积压随步数增长。

#### 3.6.2 单 bundle 状态机

| 转移 | 动作及事件 | 可释放/覆写条件 |
| --- | --- | --- |
| EMPTY → DEVICE_READY | 首次 full 生成 F 并初始化零阶项；compute stream 记录 `produced` | 原始 x_in 等临时值在相应 compute use 完成后释放 |
| DEVICE_READY → D2H_PENDING | D2H stream 等待 `produced`，复制新 bundle 到专用 pinned staging，记录 `d2h_done` | NPU 源在 `d2h_done` 前保持强引用 |
| D2H_PENDING → HOST_READY | Host writer 等待 `d2h_done` 后复制到普通 Host 正本，原子发布新 generation | staging 在 Host memcpy 完成后复用；旧正本在无读者后释放 |
| HOST_READY → H2D_PENDING | Host 正本复制到空闲 pinned staging；H2D stream 等待目标 slot 的 `last_use`，入队 H2D 并记录 `h2d_ready` | Host/staging 在对应 H2D 完成前不得覆盖 |
| H2D_PENDING → IN_USE | compute stream `wait_event(h2d_ready)`，核验主键和 generation 后读取 | 事件必须属于本 bundle，不能借用旧请求事件 |
| IN_USE → HOST_READY/DEVICE_READY | 预测或导数更新完成后记录 `last_use`；预测不修改 Host 正本 | NPU slot 仅在 `last_use` 完成后可用于下组 |
| 任意状态 → DRAINING → RELEASED | 停止入队，按依赖顺序 join Host writer、同步 streams、删除所有引用 | 未完成传输的张量不可提前回池 |

生命周期事件与计时事件分开；slot 使用 `(request_id, slot_id, generation)` 防止旧事件误满足新请求。只有上一 generation 的所有 wait/record 完成后才允许重用事件。allocator 的流记录机制若用于跨流 tensor 生命周期，须在目标 torch_npu 版本验证；同时保留明确强引用和完成事件，不能仅假定 Python 变量退出即可安全释放。

#### 3.6.3 正常时序

1. 冷启动或新专家的 EMPTY 组没有历史，不提交 H2D、不等待不存在的 Host generation；直接 full 运行原 block、初始化零阶项并按需 D2H。已有历史的请求/分支在开始时 demand-load 第 0 组，需要就绪才启动消费，不声称首组有可隐藏窗口。
2. 在消费组 j 前，若开启 prefetch，先为组 j+1 申请空 slot、仅当该组存在本次会读取的已发布 Host generation 时入队 H2D；再在 compute stream 执行组 j 的原 block 计算或 Taylor 预测。
3. 组 j 的消费者只等待自己的 `h2d_ready`；组 j+1 不受组 j 的数学数据依赖约束，其历史来自已完成的 full 步，因此可以提前复制。不能提前搬运尚未生成的新因子。
4. full 在分配或计算新 bundle 之前必须取得唯一的新因子生产 credit；上一 bundle 的 NPU 源在 `d2h_done` 完成并解除引用后才返还该 credit，不能先生成下一组再排队。随后用当前组旧因子计算新因子，compute 记录 `produced`，D2H 等该事件后搬出。新旧 bundle 仅在该组更新窗口共存；Host 正本发布后释放旧 generation；D2H staging 另在 Host memcpy 完成后才可复用。冷启动初始化零阶项也消耗相同 credit。
5. 组 j+1 消费前等待自己的 H2D 事件；继续轮转两个 slot。尾组不跨 CFG/专家盲目预取，分支尾等待未完成的发布后再推进上下文。
6. offload=true、prefetch=false 使用同一传输与回收协议，但只在使用时请求当前组；这构成消融中的同步需求加载对照。
7. D2H/H2D 各用独立 copy stream，compute 使用当前模型计算流，避免 D2H 等待当前计算时阻塞已可执行的下一组 H2D。队列满时等待最老的合法完成事件形成背压，不能覆盖尚在使用的 slot。

按上述顺序先入队生产 event 再入队消费者 wait，依赖图没有“copy 等尚未提交的 compute、compute 又等该 copy”的循环。非 offload 路径不创建搬运队列；prefetch 关闭不产生提前 H2D。正常 group 循环不用全设备同步；生命周期结束和计时边界统一同步。

预取是否有收益取决于计算窗口与带宽，特别是预测步本身较短，不能保证 H2D 被完全隐藏。验收须由时间线证明至少一项关键 H2D 与当前计算实际重叠，并测量消费侧可见等待是否下降。

#### 3.6.4 与权重 L2 预取的区别

这里的 prefetch 为 pinned Host→NPU 缓存传输。`torch_npu.npu_prefetch` 是将设备权重提前放入 L2 的另一类优化，不能完成 CPU 缓存搬回 NPU，不能用一次调用替代本节协议。本期保持关闭；后续只有在 profiling 确认目标 MatMul 为 memory-bound、依赖窗口安全且单点验证有效后，再单独评审、实验并报告，不能混入本次四档归因。

### 3.7 内存预算与设计取舍

令 `b=4*B*S*C_hidden` 为一份 FP32 特征 byte 数，q 为最高阶，G 为组数，当前专家 CFG 分支数为 2，则成熟系数上界 `C_cache=2*G*(q+1)*b`，单 bundle 上界 `U=(q+1)*b`。所有字节乘法先做整数范围及上限检查；显式记录申请值、峰值活跃值和 allocator reserved。

| 对象 | 预算方式 |
| --- | --- |
| Taylor-only 系数 | NPU 常驻 C_cache；另加旧新更新共存、FP32 入口/残差/累加临时与模型/激活，不把 C_cache 当总峰值 |
| Offload Host 正本 | C_cache，加一个发布窗口的旧新共存 U，即自动预算 C_cache+U |
| Pinned staging | 两个 H2D bundle 加一个 D2H bundle，自动预算 3U；不按全部 C_cache 锁页 |
| Offload NPU slot | 两个 bundle slot；当前旧 bundle 是消费 slot 的借用视图，不再额外复制一份 |
| 新因子与计算 scratch | 取得唯一生产 credit 后更新单组最多另加 U，工作张量保守预留 4b；offload 缓存相关自动上限为 3U+4b，非 offload 为 C_cache+U+4b |
| 非缓存开销 | 活动模型、必要的 VAE/T5、真实激活、非 parameter 张量、框架 workspace 和 allocator 碎片，另以同规格基线与运行测量确认 |

上述公式是初始内存规划上界；实现时建立全部实际 allocation 的清单，发现额外临时量必须修订上界，不能静默超过配置预算。自定义预算小于需求时启动失败。Host 可用内存/锁页额度和 NPU 可用内存需实机核验；缓存自动预算不等于可用硬件容量，启动时还要保留非缓存开销与安全余量。任一完整分配失败立即清理并非零退出，不自动切换消融档或重试改规格。

条件示例：上游 A14B 配置 L=40、C_hidden=5120，61 帧、16:9 输入按当前形状公式得到 S=14080 时，一份 FP32 特征为 275 MiB。q=2、逐块缓存需 `2*40*3*275 MiB=64.45 GiB` 的单专家系数；g=4、10 组时降为 16.11 GiB。这里未计权重与其他峰值，且 S 依赖实际输入，不能作为所有图片的实测值。

采用 g=4 的原因是降低缓存容量并保留多个可预取单元；拒绝直接假设 40 块高阶 FP32 缓存一定能在 A3 Taylor-only 档容纳。分组会改变预测误差，需与 g=1 对照并通过盲测。首期也拒绝全模型预取造成双专家权重重叠，选择可控的缓存预取；普通 Host 正本加小型 pinned 池优于默认锁页全部历史。均属于待评审取舍。

## 4. 可维可测分析

### 4.1 功能与算法验证

| 测试类别 | 必测场景 | 判定 |
| --- | --- | --- |
| 调度 | K=1/3、w=1/2/≥T、c=0/2/≥T、T=2/40、专家在首/中/尾切换 | 逐步表与 3.3 的规则一致，统计总数不超过实际 T |
| 数学 | 常量/线性可解析特征，手工有限差分表，非等距 full 间隔，q=0/1/2/3 | 独立 CPU FP64 参考核对因子与预测；逻辑测试转换 FP32 后 atol=1e-5、rtol=1e-4，不将该容差当作视频质量阈值 |
| 组包装 | g=1/4、尾部不足 g、全 full、predict 组首/组内、head 与 scheduler 调用计数 | full 原算子路径和输出保持一致，predict 只按组替代，不误触发全模型 break |
| 隔离 | 两专家、两个 CFG、相同/不同输入连续请求、预热后正式生成、专家再次激活 | 主键无串用，首次状态为空，导数只来自同 key 的真实 full |
| 配置 | 未知方法/字段、非法类型/范围、不支持的并行组合、预算不足 | 首次重计算前报错，退出码非零，无部分队列残留 |
| 传输 | offload 开关、prefetch 开关、延迟注入、过期 generation、slot 背压 | 搬回因子逐元素一致，无提前消费、过早覆写或事件循环依赖 |
| 数值健康 | 因子、预测及输出 NaN/Inf 检查 | 异常保留失败，不纳入成功性能样本 |

逻辑/CPU 测试只能证明调度和数学合同，不能证明 A3 DMA 重叠、安全的 NPU allocator 行为或性能。FP32→输入 dtype 的全 full 对照不得凭“理论等价”标 PASS，需实际验证无替换时保持原模型执行。

### 4.2 四档消融与计时口径

| 档位 | 方法 | 新缓存 offload | 缓存 prefetch | 既有 model offload |
| --- | --- | --- | --- | --- |
| B0 baseline | NoCache | 关闭 | 关闭 | true，固定当前单卡行为 |
| B1 taylorseer | TaylorSeer | false | false | true |
| B2 taylorseer_offload | TaylorSeer | true | false | true |
| B3 taylorseer_offload_prefetch | TaylorSeer | true | true | true |

四档均通过 `models/wan2.2-i2v/infer.sh` 启动，使用对应 `YAML_FILE_NAME`。B0 使用 `14b_single.yaml`，B1 以新增 `14b_taylorseer.yaml` 为源；B2/B3 保存完整解析后的配置快照。既有脚本内设置 YAML 的方式遵循模型 README，不假设 shell 外部变量一定能覆盖脚本赋值。

正式实验前固定：代码完整 SHA、上游补齐文件的 commit、两个专家权重 revision/分片 hash、输入图 hash、正负 prompt hash、完整 YAML 与解析结果 hash、硬件/软件清单、计时边界及输出规格。无这些元信息的计时不能用于验收。无 profiler 的正式运行和单独 profiling 运行分开保存；二者配置相同，不能用 profiler 开销污染正式延迟。

每档至少一次**同规格完整生成预热**，再进行至少三次正式生成。仓内两步预热仅作运行初始化，不能替代同规格预热。每次正式运行的新请求起点一致，缓存不沿用预热；模型初始驻留状态也固定，不能让某档独享已经留在设备上的专家。

主口径采用当前入口的生成请求范围：计时前全设备同步；从 `WanI2V.generate` 调用开始，包括图像/文本处理、VAE encode、denoising、CFG/scheduler、模型和缓存搬运、VAE decode、请求 drain；返回后同步全部计算及 copy streams，再停止 `perf_counter`。权重加载、进程启动、预热和 MP4 编码/文件写出不计入该主口径，另报 `bash infer.sh` 总 wall time及保存时间以便审计。此口径需在性能实验前获 RFC 共识，四档同时采用，禁止只报告 DiT 子段替代端到端。

现有入口只同步一个新建 stream，不能由此证明全部执行完成；四档需采用同样的完整同步边界。正式计时期间不逐层全设备同步；阶段事件只用于独立 profiling 运行。失败、OOM、中断、重试都保存原始记录；最终通过的三次样本不能通过丢弃不稳定结果筛选而来。

### 4.3 统计与性能归因

对于每个固定输入，记录各档原始 `t1,t2,t3`，计算均值、中位数、极差；附每阶段耗时、full/predict 数和比例、按 byte/direction/object 分类的 H2D/D2H、可见等待、峰值 allocated/reserved 及 Host RSS/pinned 值。

```text
latency_reduction = 1 - mean(B3) / mean(B0)
speedup = mean(B0) / mean(B3)
通过：latency_reduction >= 0.50（等价 speedup >= 2.0）
```

至少五组固定输入的四档均各正式运行三次，先给出逐样本比较，再按每组等权计算总体平均值，不挑最快样本。若社区指定唯一性能样本，在首次测量前于 RFC 固定，不能事后替换。性能结论同时受质量、无 OOM/重试及稳定性门槛约束。

full-step 比例按真正 denoising step 计数；另报完整执行的原始 block 次数与理论 `sum_s(2*L_expert(s))` 分母，组预测导致跳过的 wrapper 不能被计为完整 block。每条审计记录包含 request、s/timestep、expert/epoch、CFG、group/block 范围、full/predict 原因、q_eff、命中、generation、传输字节和错误状态。

Profiler 必须回答：B1 相对 B0 哪些 DiT 计算实际消失；B2 哪些缓存/非活动模型驻留到 Host；B3 至少一个关键 H2D 是否与组计算重叠、消费前是否存在对应 event wait、可见等待是否降低。DMA 时间与计算时间存在重叠，不能简单相加当作请求时延；Host 阶段、device event 时长和 wall time 分别标注。

### 4.4 质量、内存与稳定性

质量样本至少覆盖人物、动物、快速运动、镜头变化、复杂背景，各组固定输入与 prompt。生成结果随机匿名编号，隐藏使用档位，由两名指定验收人独立判断主体/语义、动作、时序一致性和伪影；至少 B0/B3 成对盲测，保存三档 Taylor 的全部视频索引。允许像素和具体内容差异，不允许 prompt 语义错误、明显噪声或明显质量退化。两位验收人逐样本给出意见；分歧保留并复审，不能用单人意见或平均数掩盖质量失败。

本期不以 LPIPS/CLIP 等数值阈值代替盲测；若增加辅助指标，先在 RFC 固定实现版本、预处理和阈值，再进行测量。盲测媒体作为受控附件/制品提供 URL 和 SHA-256，不进 Git。

四档分别采集模型加载后、生成峰值、生成结束以及同进程连续五次生成后的 NPU allocated/reserved、Host RSS、普通缓存 byte、pinned byte 和 outstanding transfer 数。正式测试覆盖正常结束、计算中断、H2D/D2H 中断、CPU buffer 分配失败、预算不足。请求关闭后 live cache 与 transfer 引用归零；allocator 保留池要与活跃泄漏区分，固定形状充分预热后重复运行不出现持续增长趋势。

稳定性测试同时包含至少一次人为中断后的清理与再次生成；设备上下文不可恢复的失败记录非零退出，并在新进程重试功能验证，不把该重试计为一次成功性能样本。可增加更长重复运行观察，不能以五次之外的短时片段覆盖已发生的 OOM 或泄漏。

### 4.5 回退与公共模块回归

关闭 Taylor：选择 NoCache 并撤销 Taylor 包装/请求资源，恢复基线生成规格和原方法；关闭 offload：因子留在 NPU；关闭 prefetch：按需加载，不提交下一组预取。运行中不允许热切换这些开关，必须在请求结束后重新配置。

Wan2.2 的 NoCache/FBCache/TeaCache 必须做相同输入的功能、质量与代表性性能回归。首期不改公共模块，仍以 HunyuanVideo 直接调用路径作共享依赖冒烟；若实现阶段任何改动进入 `module/dit_cache/`，需重新枚举所有直接调用方并扩展至功能、精度与代表性性能，至少覆盖 Wan2.2、HunyuanVideo 的三种既有方法。

回归性能使用同机、配对输入与相同统计口径，先重复建立基线波动范围，再固定允许区间；吞吐不得低于基线波动下界，延迟不得超过对应上界，不在看到结果后放宽阈值。未运行的平台/路径标为未执行，不以单卡 A3 结果外推。

### 4.6 风险、评审重点与阶段安排

| 风险 | 检测与处置 |
| --- | --- |
| 分组近似导致质量下降 | 固定五类样本盲测；调整 g/K/w/q 后重新固定配置、重跑三档，不能更改验收规格 |
| Taylor-only 或 Host offload 预算过大 | 运行前公式与实际 allocation 预检；配置不足明确失败，记录各项占用；g=4 并非容量保证 |
| 预测步太短，搬运无法隐藏 | 时间线和可见等待归因；不能用理论跳步率承诺 2×，若目标未达到如实报告失败 |
| 专家边界、CFG 顺序或两步预热污染状态 | 显式上下文、每请求重置、epoch 主键、trace 对照与异常注入 |
| 跨流悬挂引用或过期事件 | generation、单一所有者、有界队列、消费前 wait、finally drain |
| 现有模型设备判断/计时存在差异 | A3 核验，必要修正先建立对称基线，不只修优化档 |
| 公共数学类与模型 facade 不兼容 | 首次零阶初始化、协议单测，should_skip 恒 false，禁用内部层计数推进 |

阶段顺序：①发布 RFC Issue 与本设计 PR，评审缓存粒度、调度和计时合同；②在确认的 A3 环境固定基线与权重/样本 manifest；③实现 Taylor-only 并核验状态/质量；④加入有界 offload 与 prefetch，完成竞争和清理测试；⑤完成四档、盲测、公共回归和 CI 后提交代码验收材料。设计 PR 创建不等于已批准或已合入，2× 目标不等于已达成。实施日期以设计评审及 A3/权重可用时间确定，不虚填截止日。

评审需重点确认：g=4 的组残差近似与 FP32 缓存、每专家 warmup/全请求 cutoff、生成请求端到端计时边界、五样本等权性能汇总，以及实际 A3 内存上限。实现前的未知环境字段由基线采集补齐；设计可先评审，性能实验不得跳过这些前置条件。

### 4.7 交付与验收状态

后续代码 PR 关联 RFC Issue 和设计 PR，提交配置、模型侧实现、测试、README 和技术文档，保留 Apache 2.0 等来源声明。TaylorSeer 官方仓仅作算法参考，不能把其外部代码未经许可证核查直接复制入本仓。

私仓 `task_submission/` 按任务书交付以下文件：`1 自验证步骤说明.md`、`2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`、`3.1 性能自验证报告.xlsx`、`3.2 性能自验证日志.log`、`4.1 内存自验证报告.xlsx`、`4.2 内存自验证日志.log`。报告关联逐次原始结果、manifest、四档完整 YAML、盲测表和 profiling 制品，数值可重算；本任务存储空间下降指标为 N/A，但内存验证不可省略。

| 项目 | 本设计提交时状态 |
| --- | --- |
| 任务书、测试要求、官方源码与模板核对 | 已完成文档调研 |
| 实现、A3 基线、四档性能与内存 | 未执行，属于后续实施阶段 |
| 最终视频盲测、连续运行与异常清理 | 未执行，不声明质量/稳定性通过 |
| 代码 `pre-commit run --all-files`、仓库 CI、A3 实机验收 | 待代码阶段执行，不以设计检查替代 |
| 设计批准/合入 | 由社区评审决定，不能与 PR 已创建混同 |

### 4.8 参考资料

- [社区任务目录与提交规则](https://gitcode.com/cann/cann-competitions/blob/83aeaa5887560b5d0036d01f2e2ae3b0d2b1a469/04_tasks/01_community-task-2026/README.md)、[设计模板](https://gitcode.com/cann/cann-competitions/blob/83aeaa5887560b5d0036d01f2e2ae3b0d2b1a469/04_tasks/01_community-task-2026/resources/design_template.md)。
- [公共 CacheManager/TaylorSeer](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/module/dit_cache/cache_method.py)。
- [Wan 入口与预热/计时](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/models/wan2.2-i2v/generate.py)、[生成与专家切换](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/models/wan2.2-i2v/wan/image2video.py)。
- [现有首块缓存](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/models/wan2.2-i2v/wan/cache/cache_block.py)、[Wan block 遍历](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/models/wan2.2-i2v/wan/modules/model.py)。
- [单卡配置](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/models/wan2.2-i2v/config/14b_single.yaml)、[配置说明](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/models/wan2.2-i2v/config/README.md)、[模型 README](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/models/wan2.2-i2v/README.md)。
- [贡献指南](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/CONTRIBUTION.md)、[新模型 Checklist](https://gitcode.com/cann/cann-recipes-infer/blob/bbd278fe5ef772daa310df680865a6826ae6d909/docs/common/new_model_checklist.md)。
- [TaylorSeer 论文](https://arxiv.org/abs/2503.06923)、[官方算法参考](https://github.com/Shenyi-Z/TaylorSeer)、[Wan2.2 上游 A14B 配置](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/configs/wan_i2v_A14B.py)。
- [torch_npu.npu_prefetch 接口语义](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/torch_npu-npu_prefetch.md)：用于区分设备 L2 预取；实际运行版本的支持能力仍须单点核验。
