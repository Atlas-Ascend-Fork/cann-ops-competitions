# Qwen3-1.7B BF16 单卡模型样例设计

## 1. 需求背景

### 1.1 需求来源

- [10月社区任务：Qwen3-1.7B 模型样例](https://www.hiascend.com/activities/task-center/details/73500202c71f4d30bc512daeb749f00d)，任务编号 10-17。
- [任务书及测试内容](https://www.hiascend.com/p/resource/202610/6167013f90e2467daddfc9afd27be9b1.zip)。本文按 A2 版本说明交付范围。
- 目标代码仓：[cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)。已有技术讨论：[RFC Issue #458](https://gitcode.com/cann/cann-recipes-infer/issues/458)。该 Issue 用于讨论，不表示设计已通过评审。

### 1.2 现有实现与问题

仓库已提供 Qwen3-8B 与 Qwen2.5-7B-Instruct 的统一执行器接入。Qwen3-1.7B 与 Qwen3-8B 使用同一 Dense 解码器结构，可复用 `QwenForCausalLM` 和 `Qwen3Config`，无需复制模型 forward。

1.7B 的官方配置采用 `tie_word_embeddings=true`，官方 checkpoint 可不包含独立的 `lm_head.weight`。现有线性层有独立的权重后处理过程，直接将 embedding 参数对象绑定到 head 会使存储形状受到转置影响；仅跳过 head 则会留下未初始化权重。接入时需要明确 tied head 的加载与后处理边界。

现有离线入口还需要保证请求级输出上限被执行，并对完整权重覆盖和生成配置提供可追溯验证。共享修改同时影响已有 Qwen 模型，必须回归。

## 2. 需求分析

### 2.1 交付范围

| 项目 | 范围 |
| --- | --- |
| 模型 | 官方 Qwen/Qwen3-1.7B Dense，完整层数 |
| 精度与部署 | BF16、单卡、1TP、Offline |
| 执行模式 | `eager`、`npugraph_ex`；交付 YAML 默认图模式 |
| 输入与输出 | 默认最大输入 2048 token，最大新增输出 128 token |
| batch | 默认 1，验证 2 个不等长请求的 Packed Sequence |
| 硬件 | Atlas A2，实测芯片 Ascend 910B3 |
| 非交付项 | 量化、Online、多卡并行验证、独立 Ascend C 算子、性能提升比例 |

### 2.2 需求拆解

1. 增加 `qwen3_1_7b` 注册和 BF16 单卡 YAML，配置及结构参数取自官方 checkpoint。
2. 完整加载必需权重，支持 tied head，缺失、重复、形状错误和意外权重明确失败。
3. 复用既有 Prefill/Decode、Packed Sequence、Paged KV Cache 和图执行流程。
4. 保证请求在 EOS 或输出预算处结束，非法容量配置在权重/KV 分配前失败。
5. 提供固定样本、参考实现映射、性能与内存基线以及共享模型回归材料。

## 3. 详细设计

### 3.1 模型结构和计算

模型结构由官方 `config.json` 读取。当前冻结 revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，模型、tokenizer 与 chat template 均取该 revision。

| 参数 | 官方配置值 |
| --- | ---: |
| hidden size | 2048 |
| intermediate size | 6144 |
| decoder 层数 | 28 |
| attention heads / KV heads | 16 / 8 |
| head dimension | 128 |
| vocabulary size | 151936 |
| RoPE theta | 1000000 |
| tied word embeddings | true |
| QK-Norm / attention bias | true / false |

上述数值用于核对配置，不写入模型规模分支。Q/K/V 投影、QK-Norm、RoPE、Attention、输出投影、残差、RMSNorm、SwiGLU 和最终词表投影均复用已有实现。FFN 计算为：

$$\operatorname{MLP}(x)=\left[\operatorname{SiLU}(xW_g^T)\odot xW_u^T\right]W_d^T.$$

QKV 和 gate/up 的融合由现有线性层完成，保持 checkpoint 的逻辑分片顺序。Prefill 使用每条序列最后一个有效位置的隐藏状态产生 logits；Decode 使用当前 token 的隐藏状态。无独立 kernel、host tiling、UB 分块或 tiling key 设计；算子执行与内存布局由现有 torch/torch_npu 路径负责。

### 3.2 注册、配置与入口

在 `executor/core/support_models.py` 注册 `qwen3_1_7b`，指向已有 `QwenForCausalLM` 与 `Qwen3Config`。复用 `Qwen3Config`，补充官方 tied embedding 配置，不创建第二套模型类。

交付 `models/qwen/config/qwen3_1_7b_1tp.yaml`：

```yaml
model_config:
  model_name: "qwen3_1_7b"
  model_path: "your_model_path"
  dtype: "bfloat16"
  exe_mode: "npugraph_ex"
  with_ckpt: true
  enable_profiler: false
  enable_cache_compile: false
  enable_static_kernel: false
  enable_weight_nz: false

data_config:
  dataset: "default"
  input_truncated_len: 2048
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  seed: 2026

parallel_config:
  world_size: 1
  attn_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1

scheduler_config:
  max_new_tokens: 128
  batch_size: 1
  max_prefill_tokens: 2048
  mem_fraction_static: 0.25
  block_size: 128
```

权重路径保留占位符，由使用者填入实际下载目录。1.7B 交付 ND 权重配置，不新增 NZ 优化。batch=2 用例记录实际容量配置，不将默认 batch=1 的 YAML 作为 batch=2 证据。

从仓库 `models/qwen/` 目录启动：

```bash
bash ../../executor/scripts/infer.sh --model qwen --mode offline \
  --yaml qwen3_1_7b_1tp.yaml
```

`--model qwen` 选择模型目录；YAML 的 `model_name` 选择注册项。两者含义不同。

### 3.3 权重加载与审计

`QwenForCausalLM.load_weights` 遍历 checkpoint 一次，使用现有 TP-aware loader 加载，不在 1.7B 分支复制加载逻辑。

1. 从实际模型参数建立所需来源集合。合并 QKV 对应 q/k/v 三个来源，gate/up 对应两个来源；融合来源与拆分来源重复覆盖同一投影时失败。
2. 在 TP 截取前按官方完整逻辑形状检查 Dense projection、embedding 和 head；禁止依靠框架只记录 shape 错误后继续运行。
3. tied head 与 embedding 保持独立参数存储。加载 embedding 的同一 checkpoint tensor 时，经各自 loader 初始化二者，再分别执行已有后处理，避免转置互相影响。
4. checkpoint 若同时携带 head，要求其完整 tensor 与 embedding 严格相等；比较在 TP 截取前进行，避免局部分片相同掩盖完整权重不一致。
5. 重复名称、意外名称和缺失必需来源均抛出异常。仅对已知可重建 RoPE buffer 和显式减少层数的既有调试路径作限定处理；完整层数验收不使用减少层数选项。
6. 既有量化路径可缺省的 smooth scales 保持原初始化规则并列入审计；不将这一例外应用于 BF16 必需权重。

成功后输出 checkpoint tensor 数、已加载参数数、tied head 来源、可选默认项、派生 buffer、减少层数忽略项和缺失列表。失败不能生成成功审计；重试测试重新提供完整 checkpoint，不复用耗尽的迭代器。

### 3.4 输出预算与异常处理

离线返回结果的有效输出上限取全局 `max_new_tokens`、请求自身上限与已有有效 EOS 长度中的最小值。保持既有固定 Decode 调度，仅约束返回段；不宣称每个请求在其上限处立即停止计算。EOS 位于被预算截去的部分时，将返回的结束原因改为 `length`，防止文本与原因不一致。

模型初始化前校验输入上限、batch、输出上限、Prefill 预算和 block size 为正整数，拒绝布尔值；输入加输出预算不超过官方位置容量，静态内存比例在 `(0, 1]`。执行器已有并行配置校验继续负责 world size 与各 TP 的合法组合。

### 3.5 Packed Sequence、KV Cache 与图模式

沿用现有 `ForwardMetaData`、累计序列长度、slot mapping 和 block table。Prefill 与 Decode 均由现有缓存管理器管理页的申请、写入、读取和回收，不增加 1.7B 专用缓存类型。

验证 batch=2 时使用不等长输入，并与独立请求/参考运行对照，观察输出预算、请求结束、后续请求和缓存释放。连续 20 次请求用于检查缓存驻留稳定性；最大输入、输出、batch 组合单独采样峰值。图模式按固定 shape 预热，正式 Decode 的计时与编译/预热分开。

### 3.6 共享影响与兼容性

直接修改位置为 Qwen 模型注册、配置读取、Qwen 权重加载和离线输出上限。Qwen 公共模型直接使用者包含 Qwen3-8B、Qwen2.5-7B-Instruct 与新增 1.7B；离线输出上限属于执行器公共路径，使用 mock executor 测试不同请求预算、EOS 和边界行为，其他离线模型仍由仓库 CI 补充覆盖。

已有 Qwen3-8B 和 Qwen2.5-7B-Instruct 使用相同机器、相同模式和相同权重分别采集修改前后结果。tied 与 untied、拆分与融合来源、加载顺序和 TP 形状由单元测试覆盖。未改变 Attention 数学计算、KV 布局或通信协议。

## 4. 可维可测分析

### 4.1 标准与参考映射

| 检查 | 方法与通过条件 |
| --- | --- |
| 功能与文本质量 | 至少五类固定样本：中文、英文、知识问答、摘要/改写、简单推理；保留参考与实际完整输入/输出 token 和文本，相关、通顺，无乱码、异常重复、明显语义断裂或可稳定辨识退化 |
| 模式一致性 | 相同输入、参数和 checkpoint 下，eager 与 npugraph_ex 对应请求生成 token 一致 |
| 参考实现 | 同 revision 的 Transformers，BF16，`enable_thinking=false`，greedy，seed=2026，temperature=0、top_p=1、top_k=0；统一 chat template、右侧截断和输出/EOS 预算 |
| 性能 | 1.7B 建立首份基线，无优化百分比门槛；每个 shape 预热 1 次、正式 3 次，报告逐次值、均值和中位数 |
| 共享回归 | 两款既有 Qwen 模型在同平台的输出保持一致；预设 Prefill/Decode 时延容差为各自基线的 10%，不跨 A2/A3 比较 |
| 内存 | 最大输入/输出/batch 无 OOM，记录加载后与 Prefill/Decode 峰值；连续 20 请求，预热后增长上限 32 MiB，并拒绝持续单调或分窗口低水位增长，检查 KV 块回收与重复输出 |
| 权重 | 完整来源和形状审计通过，缺失必需项为零，异常 checkpoint 明确失败 |

本任务不以 CPU 小规模投影误差代替 BF16 NPU 端到端文本质量标准。CPU 数值测试使用独立 `F.linear` 公式核对 QKV、SwiGLU 和 head，容差为 `atol=1e-6, rtol=1e-5`；加载失败后完整重试采用逐参数严格相等。

### 4.2 测试矩阵

| 类别 | 用例 |
| --- | --- |
| 基础/精度 | 五类固定自然语言样本，eager/npugraph_ex，batch=1/2，完整参考输出 |
| 长度与结束 | 短输入、2048 token、超长输入右截断、输出上限 1/128、EOS、全局与请求上限组合 |
| Packed/KV | 两个不等长请求、连续 20 请求、结束回收及后续请求隔离 |
| 性能 | 输入 128/512/1024/2048，输出上限 128，batch=1，npugraph_ex；每形状 warmup=1、repeat=3 |
| 内存 | 最大输入 2048、输出 128、batch=2；加载后、Prefill、Decode、请求完成后采样，Profiler 开启 memory |
| 负向配置 | 非正整数、布尔伪整数、位置容量超限、非法内存比例及并行配置 |
| 负向权重 | 缺失、重复、错误 shape、未知权重、tied head 不一致、融合/拆分混用重复 |
| 验证工具 | 不完整记录、错误执行配置、伪内存峰值、错配 revision/样本/模板、旧成功报告、输出覆盖输入或源码、CLI 缩写与选项终止符 |
| 共享回归 | Qwen3-8B、Qwen2.5-7B-Instruct 的代表性 Prefill/Decode 及完整输出 |

### 4.3 已完成的开发自验

下列结果为开发自验，不能替代设计评审、仓库 CI 或官方验收。

A2 环境为 Ascend 910B3、单逻辑设备、64 GiB 标称 HBM，Ubuntu 20.04.5 aarch64、Python 3.11.4、CANN 8.5.2、PyTorch 2.8.0+cpu、torch_npu 2.8.0.post4、Transformers 5.0.0、驱动 25.5.0。固件版本未从环境中获取，不以推断值补填。实际可用设备内存为 65,452,113,920 bytes。

测量基于上游 `bbd278fe5ef772daa310df680865a6826ae6d909` 和候选 `88a8e30d8f6cf3570243d60fd52274fe51be9562`。随后仅更新验证工具、测试与材料，模型运行时和交付配置的文件哈希保持一致；不将保留测量写成最终工具版本重新采集。

| 输入 token | Prefill 中位数（ms） | Decode 中位数（ms/step） |
| ---: | ---: | ---: |
| 128 | 71.825 | 12.760 |
| 512 | 73.058 | 12.810 |
| 1024 | 78.605 | 12.859 |
| 2048 | 77.649 | 13.003 |

- 18 组模式/batch 配对请求 token 一致，保留了官方参考完整输出；最终验收材料还需给出逐样本文本质量对比结论。
- 权重审计为 311 个 checkpoint tensor、227 个参数，tied head 从 embedding 初始化，必需权重缺失为零。
- 最大 NPU allocated 为 4.713 GiB；第 6 至 20 次请求 Host RSS、NPU allocated/reserved 的净变化均为零。该指标不代表整个进程的 HBM 总占用。
- 四组共享回归的最大时延增幅为 7.562%，低于预设 10% 容差。
- 完整 CPU 回归 346 项通过，包含新增的加载数值、执行证据和边界测试；最终 CPU 完整组运行环境为 Python 3.11.15、CANN 9.0.0、PyTorch/torch_npu 2.8.0 系列。
- 更严格的最终校验器复核 A2 的 20 份、A3 的 16 份保留运行记录通过；该复核不等同于重新执行 NPU 测量。
- 改动文件格式检查通过。全仓 `pre-commit run --all-files` 仍有既有尾空白、末尾换行和脚本执行权限问题；仓库 CI 尚未执行，不声称全部门禁通过。
- A2 Profiler 索引包含 807 个文件；采集中有 32 条不完整 memory event 警告，保留原始日志，不将该部分产物写成无警告的完整内存 trace。

### 4.4 交付和维护

代码提交包含注册、YAML、必要共享修复、固定样本、可执行验证/比较脚本、单元测试、基线 metadata 和 Qwen README。metadata 关联实际采集 commit、配置、环境、权重 revision、文件哈希、计时和日志索引。

权重、数据集、二进制与大日志不入 Git；提供获取方式及校验信息。公开文档不包含机器登录信息或开发环境私有路径。验收脚本异常返回非零，失败报告不能复用旧成功状态，输出路径不能覆盖输入证据或验证源码。

设计 PR 评审合入后，按社区任务流程提供代码与自验材料进入官方测试；根据官方通知完成目标代码仓的公开 PR、CI、评审与合入。代码进入公开 PR 前同步目标分支，若共享依赖发生变化，重新评估并补充受影响验证，不沿用旧 commit 名义声称新版本已实测。
