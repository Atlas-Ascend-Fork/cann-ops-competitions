# Gemma-4-E2B 纯文本推理接入设计文档

> 作者:qwegugugaga|关联:RFC #447(cann-recipes-infer)、代码 PR #1068|任务:10月社区任务·Gemma-4-E2B 纯文本推理接入(A2)

## 需求背景(required)

### 需求来源

10 月社区任务·Gemma-4-E2B 纯文本推理接入(Atlas A2 单卡 1TP,batch=1,Offline)。在统一执行器 cann-recipes-infer 中接入 `google/gemma-4-E2B`(5.1B dense,BF16 约 10.2GB)的语言路径,并按官方 text_config 支持三大特有结构。

### 背景介绍

#### 对标模型与计算语义

- 对标实现:transformers v5.5.0 `modeling_gemma4.py`(官方语义权威),参考环境与开发环境两套隔离(任务书 §3.1)。
- 权重:`google/gemma-4-E2B@d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`(sha256 前缀 76dc84a5),官方嵌套 `text_config`。
- 计算语义要点:35 层(4:1 sliding:full 分组)、逐层输入(hidden_size_per_layer_input=256)、跨层 KV 共享(num_kv_shared_layers=20,消费层 15~34)、双宽 MLP(前 15 层 6144/共享区 12288)、双类型 RoPE(head_dim 256/512)、QK-Norm 吸收 `1/sqrt(d)`(`softmax_scale=1.0`)、`final_logit_softcapping=30`、tie embedding。

#### 仓库实现现状

- 基座 `models/gemma_4/`(26B-A4B)已接入统一执行器,提供 PagedAttention sliding/full 双 block pool 与任务书建议入口;本任务 E2B(密集模型)经 spike 与方案评估采用**模型自管 KV(路线 P)**,不走 PA 路径(依据见 RFC #447 §5b 与本文详细设计)。
- 统一执行器对图模式(npugraph_ex)要求模型实现 `main_decode` 并走 cache_compile;模型契约(config/infer_config/comm_manager/prefix)与 ForwardMetaData 元数据驱动。

### 任务材料差异与设计假设

- 官方 checkpoint 不含 chat_template:按 RFC #447 问题①预案,精度对齐两端固定仓内 26B jinja 模板,定案变更则按约定重采。
- 数据集协议:RFC 原钉 5-shot,因滑窗 prefill T>512 缺口(见约束限制)改为 0-shot 全量,偏离已声明待追认。
- 判据:RFC §4 组合判据测试前固定;实测如实呈现,不静默调阈。

## 需求分析(required)

### 需求描述

按任务书完成 E2B 语言路径接入:三大必做结构逐项实现并对齐官方语义;eager 对齐后必验 npugraph_ex Decode;完成精度对齐、数据集评测、性能矩阵、内存与生命周期、滑窗边界、权重加载审计等全部验收项。

### 接口与参数分析

- 模型契约:实现 `Gemma4E2BForCausalLM(config, infer_config, comm_manager, prefix)`;前置校验 batch_size=1、world_size=1、`layer_types` 长度/共享层数/双宽一致性,非法配置构造期显式报错。
- 元数据:ForwardMetaData 驱动(prefill/decode、actual_seq 各形态字段);`get_cache_info()=None` 显式声明不走 PA 路径,缓存生命周期由 `init_cache/load_weights` 承接。

### YAML 与启动约定

- `config/gemma_4_e2b_bf16_1tp.yaml`(eager,精度对齐主配置)与 `config/gemma_4_e2b_bf16_1tp_npugraph_ex.yaml`(Decode 图模式);启动入口 `bash executor/scripts/infer.sh --model gemma_4_e2b --mode offline --yaml ...`;CI 轻量配置 `config/ci/` 供 `test_all_case.sh` 使用。
- 运行前按部署机 CANN 布局填写 `executor/scripts/set_env.sh` 的 cann_path(上游占位符保持原样)。

### 需求拆解

D0 spike(算子形态与路线初裁)→ 模型代码与真权重全链 → 代码评审与缺陷修复收敛 → 精度对齐链(参考环境/NPU 双端)→ 图模式攻坚 → 全项验收实测 → 文档与交付。

### 交付件与证据规划

models/gemma_4_e2b/(模型代码+YAML×2+CI 配置+infer.sh 入口+tests 40 件+README+requirements×2)与 docs/models/gemma_4_e2b/ 三文档(适配说明/KV 共享设计/验证报告);验证报告内含任务书要求对照、判据状态表与证据 sha256 索引;全部脚本与数据产物随 tests/ 归档。

## 详细设计(required)

### 算子分析

#### 数学公式

- 逐层输入(PLE):`embed_tokens_per_layer(ids)→×√256`;hidden 侧 `per_layer_model_projection(embeds×1536^-0.5)→RMSNorm`;层内 `gate→gelu_tanh→×PLE→projection→post RMSNorm→+residual`,末尾乘该层 `layer_scalar`。
- 注意力:QK-Norm 后 RoPE(sliding 层 head_dim 256 局部频率,full 层 512 全频率),`softmax_scale=1.0`(QK-Norm 已吸收 `1/sqrt(d)`);full 层全因果,sliding 层窗口 512。
- 跨层 KV 共享:消费层(15~34)无 k_proj/v_proj/k_norm,复用共享起点前最后一个同型层(层 13 滑窗/层 14 full)post-norm、post-RoPE 的 K/V;消费者 Q 独立。
- 双宽 MLP:前 15 层 intermediate 6144,共享区 12288,GELU(tanh) 门控。
- 输出:final logits ×softcap 变换(`final_logit_softcapping=30`)+ tie embedding。

#### 支持数据类型与形状

BF16 计算;hidden 1536;per-layer input 256;prefill 打包 1D(TND 语义,≤2048);decode 单步。

### 算子实现

#### 实现方案

##### 整体推理流程图

```
prefill: ids → PLE 双路注入 → 35 层循环[源层写 KV→共享区读源 KV] → norm → softcap → logits
decode : 单 token → 同一 35 层循环(源层写当前步 KV→消费层经环形窗/全长视图读源) → logits
```

#### Host 侧设计

##### 配置与权重加载

- 独立 `Gemma4E2BConfig`(嵌套 text_config 解包+E2B 标记校验;`has_no_defaults_at_init=True` 规避 transformers to_diff_dict 默认实例构造;顶层 model_type/architectures 合并后 pop)。
- 权重逐名审计:checkpoint 键 2011=1411 跳过(视觉/音频)+540 载入+60 消费层按声明缺省;参数视角 541(540+tie lm_head 1→2 展开)。

##### 共享 KV 与缓存所有权

路线 P(模型自管):`get_cache_info()=None` 脱离 PA 路径;持 KV 层 15 层(0~14:12 滑窗+3 full)全长连续缓冲(约 40.1MB)+28 个滑窗层环形暂存(约 14.7MB),合计不足权重 0.6%。消费层零缓存接口,decode 跨层读源层缓冲。

##### 共享 KV Cache 生命周期流程图

```
新请求 prefill: 源层写 k_buf/v_buf[0:T], 滑窗层预置尾窗(wk/wv[0:min(T,512)])
逐步 decode  : 源层 index_copy 当前步;滑窗层环形移位+追加(sel/slot 公式,窗口语义精确)
请求边界    : prefill 重置全部游标态(张量驱动, 无 Python 跨步状态)
```

##### 缓存接入的静态依据与原型通过条件

- full 层对定长全长缓冲视图做 FIA(sparse_mode=0,actual_seq_kvlen 截断),spike 实测与精确 shape 位级一致。
- 滑窗层单 FIA 环形窗(prefill 预置尾窗升序+每步移位追加),数学上等价全前缀窗注意力;torchair 复合图(双 FIA+where)在 CANN 9.0.0 存在确定性 vector core 异常(507035),单 FIA 方案规避(spike v10~12)。

##### Prefill 与 Decode 的区别

prefill 与 decode 走同一套层实现;差异仅在元数据(mask/actual_seq)与写入槽位——decode 写入槽位=position_ids(张量驱动),图重放安全、无跨步 Python 状态。

##### 图模式与异常处理

npugraph_ex:cache_compile 编译路径(绑定 main_decode+warmup+set_stance(skip_guard_eval_unsafe)),图/eager 模式内 6 步位级一致+第二请求生命周期一致,稳态零重编译。T>512 滑窗层 prefill 触发运行时 warning 护栏(RFC #447 问题条,见约束限制)。

#### Kernel 侧设计

模型级接入,不新增 kernel;注意力用 CANN FIA 原语(full=BNSD sm3 因果 / 滑窗=TND sm4+pre_tokens=512 形态,与仓内 26B 基座同款),逐 token workaround 覆盖 T>512 域。

### 复杂度与内存分析

全长缓冲 15 层(0~14:12 滑窗+3 full)×2×2176×HKV(1)×D(256/512)×2B≈40.1MB;滑窗环形暂存 28 层×2×512×256×2B≈14.7MB;合计约 0.5% 权重量级。实测四时点:载入 9.45GB/prefill 峰/decode 峰/20 请求后,无泄漏。

### 支持硬件

Atlas A2 单卡(1 die,1TP,batch=1);CANN 9.0.0+torch_npu 2.8.0.post4+PyTorch 2.8.0+Python 3.10。

### 算子约束限制

- 滑窗 prefill T>512:sm4 窗口约束在 T=513 起不被执行(≥1024 掩码路径无语义),为 RFC 发布后补充验证的实测发现;T≤512 验证域内有效,T>512 运行时 warning+逐 token decode 过渡方案(已验证),正式提请官方。
- batch>1、多卡不在首期范围,构造期显式报错。
- 图编译缓存跨进程复用在 CANN 9.0.0 不保证签名兼容,升级模型代码后需清 graph_cache_* 目录(README 清理节)。

## 可维可测分析

### 精度标准/性能标准

#### 精度标准

- 10 固定样本(五类各 2)与参考(transformers 5.5.0 CPU)同输入逐 token 对比:7/10 全等,3 例分歧均为 bf16 精确并列/1ulp 翻转(翻转集 {1,2,5} 在 A3/A2 双硬件完全一致,分歧前 15~77 步全同,首步 top-1 10/10 一致);双模式端到端 10/10 逐 token 一致。
- RFC §4 判据(如实,A2 实测):①7/10(未达,修订案随 RFC 流程)②0/10(0.2 绝对阈值与 top-32 尾部小幅值元素冲突,提幅值分区相对阈值;A3 轮 1/10,同量级尾部噪声)③10/10(29~31/32,达成);500 题子集双端 Δ1.2pp≤3pp(3σ=2.63pp)。

#### 性能测试设计

128/512/1024/2048×128 输出、batch=1、双模式、warmup1+run3,均值/中位;decode eager ~106ms/tok vs npugraph_ex ~13.4~13.9ms/tok(≈7.8×,A2 实测);infer.sh 全链双模式 46.84→13.70ms(3.42×);Profiler 口径复测(A3 轮,profile_memory=True)≈4.8×。

#### 内存与生命周期测试

四时点(载入/prefill 峰/decode 峰/20 请求后)无泄漏;连续 20 请求输出 finite 且互不相同(零串扰)。

#### 边界与异常用例细化

滑窗边界 511/512/513/514 单点+环形连续扫 1..520 全过;非法配置(batch>1/world>1/layer_types 不一致)构造期报错;`test_all_case.sh --models gemma_4_e2b` 退出码 0+rank0 有效输出。

### 分层验证计划

模块级(spike 形态矩阵/权重审计)→模型级(对齐/判据/e2e)→系统级(infer.sh、test_all_case、Profiler);全量复验通过(位级复现+逐题一致+性能区间)。

### 兼容性分析

公共代码改动仅 support_models.py 注册 4 行(纯新增);与既有 models/gemma_4/(26B-A4B)命名空间隔离,不影响其执行路径;`Gemma4E2BConfig` 独立实现,`layer_types` 显式列表拷贝(不受引用污染类问题影响)。
