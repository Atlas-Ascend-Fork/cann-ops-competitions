# Qwen3-1.7B BF16 单卡样例设计文档

状态：设计补充评审稿（2026-10-20）。此前已提交实现仓 RFC Issue #480，但遗漏社区任务 competitions 仓设计 PR；本稿补齐正式设计提交流程，不回溯声称开发前已获设计评审。设计评审及合入状态以本 PR 实际结果为准。

# 需求背景（required）

来源：10月社区任务《Qwen3-1.7B模型样例开发任务书（A2）》及随附test_cases；任务目录10-17-Qwen3-1.7B-sample。参考既有Qwen3-8B样例，在cann-recipes-infer新增更低资源的Qwen3-1.7B BF16单卡离线样例，复用框架入口和推理后端。

# 需求分析（required）

交付独立注册名与1TP YAML、完整加载与运行说明；支持Prefill/Decode、Paged KV Cache、Packed Sequence、不等长batch至少2、eager和npugraph_ex；完成固定官方权重与参考推理对照、边界/非法参数/内存测试，回归公共路径涉及的Qwen3-8B及Qwen2.5-7B-Instruct。无量化、不含Online PD或多卡优化。功能接口遵循任务书，不将不同后端逐token相等扩张为未经确认的质量标准。

# 详细设计（required）

## 1. 实现范围与接口

- 在 executor/core/support_models.py 注册 qwen3_1_7b，复用 models.qwen.models.modeling_qwen.QwenForCausalLM 和 Qwen3Config，不借用 qwen3_8b 名称加载 1.7B。
- 新增 models/qwen/config/qwen3_1_7b_1tp.yaml；沿用现有 forward 接口及官方离线启动入口。
- 核对官方 hidden/FFN size、层数、Q/KV heads、head_dim、vocab、RoPE、QK-Norm、bias、embedding/lm_head、权重名称与 shape。兼容处理按配置驱动，不硬编码模型规模；审计必需权重、重复/额外项及实际加载 dtype，缺失必需权重明确失败。
- 复用 Paged KV Cache、Packed Sequence；支持 batch 至少 2 的不等长输入、npugraph_ex 和 eager 对照。
- 已定位的离线 max_new_tokens=1 返回边界问题采用最小修复；列出公共 Scheduler 直接调用方及影响范围，既有 Qwen3-8B、Qwen2.5-7B-Instruct 逐项回归，其他未实机覆盖路径不写成通过。

## 2. 固定代码、权重与环境

上游基线：6d060311ae75f6105fe4216e20566875da5a8838。候选版本随提交记录完整 commit/变更文件 SHA256，不把旧设备记录冒充后续代码版本的全矩阵结果。代码仓库后续使用 Miomugi 名下的官方 fork；PR 关联本 Issue。

官方模型与 tokenizer 使用同一 revision，无量化或权重格式转换：
- Qwen/Qwen3-1.7B：70d244cc86ccca08cf5af4e1e306ecf908b1ad5e。
- Qwen/Qwen3-8B（回归）：b968826d9c46dd6066d109eabc6255188de91218。
- Qwen/Qwen2.5-7B-Instruct（回归）：a09a35458c702b33eeacc393d103063234e8bc28。

1.7B 两份 safetensors SHA256：169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5、912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9。tokenizer.json SHA256：aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4。完整资产清单、逐文件校验与官方获取方法随基线交付。

实际设备：Atlas A2 / Ascend910B3，板卡 IT21HMDA_Bin6，单卡 HBM 65536 MiB；物理卡 1 映射逻辑 npu:0；Host MemTotal 257547764 KiB。Ubuntu 20.04、aarch64，Python 3.11.4，驱动 25.5.0。容器板卡查询 Firmware Version 返回 NA，固件独立版本尚无法取得，不以驱动版本代填。

CANN toolkit 9.0.0 及匹配 910b ops 9.0.0；PyTorch 2.8.0+cpu、torch_npu 2.8.0.post4、Transformers 5.0.0、NumPy 1.26.4、PyArrow 20.0.0。交付依赖清单、实际动态库路径及硬件探测记录。受控回归固定 CPU0–15、OMP_NUM_THREADS=8、CPU_AFFINITY_CONF=2,npu0:0-15；保存实际线程亲和。环境、权重或配置变更时更新本 Issue 并重采受影响项目。

## 3. 完整交付 YAML 与启动方式


代码基线和官方权重准备完成后，模型路径替换为本机 checkpoint 目录；以下为交付配置：

```yaml
model_config:
  model_name: "qwen3_1_7b"
  model_path: "/path/to/Qwen3-1.7B"
  exe_mode: "npugraph_ex"
  with_ckpt: true
  enable_profiler: false
  enable_cache_compile: false
  enable_static_kernel: false
data_config:
  dataset: "default"
  input_truncated_len: 2048
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  seed: 42
parallel_config:
  world_size: 1
  attn_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1
scheduler_config:
  max_new_tokens: 128
  batch_size: 1
```

从 models/qwen/ 执行：
bash ../../executor/scripts/infer.sh --model qwen --mode offline --yaml qwen3_1_7b_1tp.yaml

模型目录参数为 qwen，注册名在 YAML 中为 qwen3_1_7b。测试派生 YAML 只调整已声明的输入上限、batch、输出长度、exe_mode 和采集开关，逐份留存；性能单卡串行、单请求并发，禁止其他 NPU 任务同时运行。

## 4. 功能、精度与内存自验

精度参考为同 revision 的官方 Qwen3-1.7B、Transformers 5.0.0 BF16 generate，不以 8B 为精度参考。固定 seed=42、greedy、相同 tokenizer/chat template、官方默认 thinking 开启、相同输入 token/截断、EOS 与 max_new_tokens=128；保留所有显式参数及参考接口默认值映射。按任务书 3.2 检查输出相关、通顺，无乱码、异常重复或明显语义断裂，无可稳定辨识退化；不同后端不强加逐 token 相同要求，输出差异和达到长度上限的情况均保留说明。不临时引入或放宽 logits 容差。

固定五类样本 v1，文件 tests/data/prompts.json SHA256：db51bd58a39b2967e26a74e6c2a667cc27fe3bb4b8205079471c2a8a5304f1fe：
1. 中文：用两句话解释为什么天空通常是蓝色的。
2. 英文：Explain the difference between a list and a tuple in Python in two sentences.
3. 知识：法国的首都是哪里？只回答城市名称。
4. 摘要：请总结这段话：学校把旧停车场改成了花园，学生每周轮流浇水，收获的蔬菜用于食堂。
5. 推理：小明有3个苹果，又买了2个，送给朋友1个。现在有几个苹果？请简短回答。

功能矩阵：eager/npugraph_ex 基础 Prefill/Decode；短输入、实际 token 长度 2048 及超长截断；batch=1/2 不等长请求；输出 1/128 边界和 EOS；token 词表范围及 NaN/Inf 检查；配置文件、模型名、模型路径/权重、模式、输入长度、batch、输出上限、并行字段等非法输入明确报错，失败脚本非零退出。保存完整输入输出 token/text、参数映射和 graph 编译计数；稳态 Decode 不持续重编译。截断使用既有数据入口并保存截断前后 token，参考端消费相同实际输入，避免模板重复处理。

容量上限为已验证的 64GB 级单卡、batch2、每请求输入2048/输出128。记录权重加载后、加载峰值、推理峰值的 Host/NPU 内存，以及连续20请求结束时 allocated/reserved、KV 分配回收，无 OOM、无持续泄漏。当前容量核算：checkpoint 逻辑张量 4,063,479,808 bytes；35 个 KV 块（含 null 块）逻辑预算 513,802,240 bytes；已记录加载后 allocated 4,085,602,304 bytes，加载峰值 5,331,310,592 bytes，容量推理峰值 5,060,591,104 bytes，Host 加载采样峰值 12,144,529,408 bytes。allocator 数值不等于全设备占用，保留驱动/运行库/workspace 余量，不承诺更小板卡支持。

## 5. 性能采集与回归评价方案

1.7B 仓内此前无性能基线，本任务建立首份 Prefill/Decode 基线，不虚构优化前 1.7B 数据、不设置优化提升比例。矩阵为 npugraph_ex、输入实际 token 数128/512/1024/2048、输出上限128、batch1；完整模型推理，不用小算子替代。

共享路径回归：Qwen3-8B 与 Qwen2.5-7B-Instruct，各 eager/npugraph_ex、输入128/512、输出上限128、batch1。原版和候选使用同卡、同权重、同输入输出参数、同软件环境，每侧独立5次完整请求预热，再5次正式请求。原版与候选均采集 Profiler，正式采集开销和窗口一致；区分加载/编译/预热/稳态，不把主机 Python 代码耗时变化直接解释为 NPU 计算退化。

主要证据为前后计算图/算子序列及纯 NPU kernel 时间：记录算子名称、类型、顺序、shape、dtype、format、BlockNum，能取得时补图连接和依赖；缺失的图信息明确注明。元数据相同不冒充 kernel 二进制或完整依赖图相同。Prefill 和稳态 Decode 分开，Decode 使用匹配步次和实际长度；报告 kernel duration 总和，以及重叠区间合并后的设备 busy 时间，排除 host 下发等待和无 kernel 的空白区间，不将重叠执行重复计作墙钟。

波动判定：完整保留两侧逐次值、均值、中位数、min/max、标准差和相对差。以相同 case 原版重复采集的设备侧波动区间作为参照；计算图/算子工作量一致、候选设备耗时处于该波动范围且无稳定可复现退化时，按正常波动解释。小幅越界时保留其绝对/相对差和重复配对证据，不仅凭单次更快或更慢下结论；不事后挑选轮次，不凭空规定一个百分比掩盖稳定退化。具体设备侧标杆值随完整采集追加本 Issue，维护者如有指定容差按确认口径执行。旧的同步 host wall time 表仍保留并标明计时范围，不改写历史失败记录。


# 自验方案与当前结果

上文第4、5节给出完整检查协议及采集口径。当前代码PR：https://gitcode.com/cann/cann-recipes-infer/pull/1184 。实现提交：6083e1d4b42cf48456f07c2bdf0962983139578d，官方fork分支qwen3-1-7b-adaptation。实现仓RFC：https://gitcode.com/cann/cann-recipes-infer/issues/480 。三类提交分别记录，本设计PR不替代代码PR。

- A2功能、五类文本质量对照、Packed/KV、batch与输出边界、异常及内存已有实机记录；HF/eager/graph可能存在文本token差异，按任务书质量判据记录，不能称所有后端逐token一致。
- 共享7B/8B路径已完成8组配对，每侧独立5次预热、5次正式采集；440个配对采样步kernel有序元数据及工作量一致，回归输入输出一致。16阶段kernel耗时中位差-1.216%至+1.248%，8项加快、8项变慢。该证据不证明完整依赖图或kernel二进制完全相同。
- 历史同步Host墙钟仍有5/16阶段超出原严格边界，不删除历史记录。设备侧口径为用户要求并公开提交评议，不能冒称官方已批准该变更；最终性能判定待维护者确认。
- 当前代码提交CI849通过、CLA通过；这不代表设计文档已评审或代码已合入。容器Firmware Version返回NA，固件版本未知，不能用驱动版本代填。
- 实测源码版本/生产文件哈希与文档规范修复版本分别留存；不将后续规范提交称为全矩阵重跑。

# 交付与评审事项

代码、YAML、测试/复现脚本、小型基线与逐次结果置于实现仓规定目录，说明见models/qwen/agentic/baseline及模型README；权重、大日志和凭据不入仓。验收附件提供一份总报告、复现说明、来源索引和轻量证据，原始profiler归档保留SHA256，可按评审要求补充。完整验收代码：https://gitcode.com/Miomugi/cann-recipes-infer/commit/6083e1d4b42cf48456f07c2bdf0962983139578d?ref=qwen3-1-7b-adaptation 。

本次申请评审方案及接口范围、跨后端质量判断、共享路径回归与设备侧性能口径。按社区任务平台反馈，设计PR评审合入后再提交验收；若维护者要求修改，更新方案并重测受影响项，不将待审事项写成已通过。
