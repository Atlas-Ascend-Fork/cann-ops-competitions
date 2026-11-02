# 需求背景（required）

## 需求来源

本设计对应「10月社区任务-daVinci-MagiHuman FA MXFP8 量化任务书（A5）」及配套 `test_cases/README.md`，目标代码仓为 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)。流程采用[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)、[社区任务目录说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)和目标仓 `CONTRIBUTION.md`。

提交团队为 `chilpirate`，活动报名已审核通过。原设计 [PR #2143](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2143) 已于2026-10-14评审通过并合入；环境、固定权重和完整BF16基线已完成独立复核。本补充固定最终路线为PScale256、scale_alg1、十层标准BF16回退且不做K中心化，并说明公共默认路径隔离和执行期持有关系释放。最终源两case精度、26项接口及HY代表回归已实测通过；正式性能、连续内存和profile仍在采集。补充评审保持OPEN，最终验收尚未完成。关联 [RFC #432](https://gitcode.com/cann/cann-recipes-infer/issues/432)。

## 背景介绍

daVinci-MagiHuman 通过 40 层 DiT 联合生成视频和音频。固定上游配置的 hidden size 为 5120，head dimension 为 128，Q 为 40 heads，K/V 为 8 heads，属于 GQA。base 模式的 `local_attn_layers` 为空；局部 Attention 使用 flex 的 Q/K 区间与 softmax LSE 合并，不可直接用全序列稠密 Attention 替换。

模型现有 `flash_attn_func` 和 `flex_flash_attn_func` 使用 BF16 `npu_fused_infer_attention_score`。公共 `module/fa_quant/` 已有动态 MXFP8 量化与 FA v2 封装；当前封装以 BSND 输入、TND 计算和基础非 sparse Attention 为主，错误信息、layout 和 GQA 契约仍需检查。当前直接调用方为 HunyuanVideo Attention 和 `module/unified_sp/core.py`；实施前再次检索并冻结影响清单。

本任务仅量化 Attention 的 Q/K/V 激活，复用现有 CANN 算子。MatMul 权重量化、RoPE 融合、缓存和图模式不纳入收益。

# 需求分析（required）

## 需求描述

在 Ascend 950 单卡上，使标准 Attention 和经 RFC 选定的真实额外 case 执行动态 MXFP8 FA。输出保持原 dtype、shape、GQA 和局部范围语义；未支持组合明确回退或报错。通过中间张量比较、视频盲测、完整生成性能、峰值内存和公共调用方回归后，整理私仓与自验材料并提交平台首次验收；后台测试通过后再提交正式代码PR。

## 需求拆解

| 工作 | 产物与通过条件 |
| --- | --- |
| 基线与 RFC | 固定代码、权重、完整有效配置、输入、软件与硬件；950 上运行 BF16 base 和 profiling |
| 能力验证 | 实机验证真实 Q40/KV8/D128 的动态量化、scale 与 FA v2；探测长度、输出 dtype 和 LSE 能力 |
| 公共接口 | 保持现有位置参数调用兼容；新增参数使用关键字；校验 layout、长度、head、dtype 与设备 |
| 模型接入 | 启停与回退开关贯通 CLI、配置和实际 Attention；保留 flex 区间及 LSE 语义 |
| 精度 | Q/K/V、scale、FA 和 DiT block 达到预设门限；5 组固定样本由 2 名指定验收人盲测 |
| 性能与内存 | 每档至少 1 次预热、3 次正式完整生成；FA、含量化/layout 的 Attention 和端到端均改善，NPU 峰值降低 |
| 兼容与交付 | 直接调用方回归、HunyuanVideo single_fp8 代表性回归、pre-commit、CI、文档及私仓七项材料 |

# 详细设计（required）

## 算子分析

### 数学公式

对 query head \(h\)，对应 KV head 为 \(g(h)=\lfloor h/(H_q/H_{kv})\rfloor\)。原计算为：

\[
O_h=\operatorname{softmax}\left(Q_h K_{g(h)}^T / \sqrt{D} + M\right)V_{g(h)}.
\]

Q/K 按最后一维、V 按公共封装规定的 token 维执行 `npu_dynamic_mx_quant`，分别得到 FP8 值和 E8M0 scale；scale 直接交给 `npu_fused_infer_attention_score_v2`。daVinci显式选择内置`scale_alg=1`，公共旧调用默认保留算法0。补充方案中，实际量化分支另外向FA v2传入PScale256；PScale是内部Attention概率量化的缩放参数，不是三路Q/K/V scale的替代品。真实轴、scale shape 与 padding 契约以所安装版本的接口和实机检查为依据，冻结到 RFC。运行路径不执行「反量化后调用 BF16 FA」。

flex 的同一 Q 区间如果有多个 K 区间，需要按各区间的 softmax LSE 合并：

\[
L=\operatorname{logsumexp}(L_1,\ldots,L_m),\qquad O=\sum_i \exp(L_i-L)O_i.
\]

不能简单求平均、丢弃区间或把区间外token纳入Attention。本次不提供MXFP8 LSE返回契约；多K整个Q区间固定保留原BF16分段计算和LSE合并，并记录回退原因，不以未验证能力替代原语义。

### 支持数据类型

| 张量 | 设计契约 |
| --- | --- |
| 量化前 Q/K/V | 目标模型 Attention 当前使用 BF16；三者在同一 NPU、维度和长度合法 |
| 量化后 Q/K/V | `torch.float8_e4m3fn`，由动态 MXFP8 算子实际生成 |
| Q/K/V scale | 算子返回的 scale，FA 使用 `torch_npu.float8_e8m0fnu` 对应类型，禁止全 1 或伪造值 |
| FA 输出 | 对应输入的原始 dtype；目标 BF16 计算及外部 dtype 恢复行为保持一致 |

公共封装原有 `dst_type` 参数的支持集合、`out_dtype` 枚举/类型、scale 存储形式及非有限值检查方式以安装版本为准。非法 dtype 提前报错，不接受通过硬编码 FP8 类型掩盖参数不一致。

### 支持形状

标准模型输入为 Q `[B,Sq,40,128]`、K/V `[B,Skv,8,128]`，单卡 base 为 B=1。BSND 转 TND 时保持 batch 边界，`actual_seq_qlen` 与 `actual_seq_kvlen` 使用各样本的真实累计长度。不能用 query 长度代替 key 长度，也不能重复展开 GQA 的 K/V heads。先验证现有 BSND 公共接口即可；本轮不新增任意多样本 packed-varlen 接口。

flex 输入为 TND 或 B=1 的 BSND，由现有 Q/K range 切出真实子区间。空区间、不完整覆盖、非法 head 比、非连续 tensor 和长度不整齐等情况分别检查；合法非连续输入通过显式 contiguous/layout 转换处理并计入成本。非法结构在调用 NPU 算子前报错。

固定配置预计含 3584 个视频 tokens、126 个音频 tokens和实际文本 tokens；4350 只是静态上限估计，不能作为实际序列长度证据。文本编码先 pad/trim 到640，再由 SingleData 去除 padding；conditional/unconditional 的有效文本长度需分别记录。正式设计将附上各层、各步的真实 shape、长度和调用次数。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

1. `config/fa_mxfp8.yaml` 沿用base原权重和完整生成配置，仅增加量化、回退开关；`infer.sh fa_mxfp8` 与CLI贯通，默认关闭。固定策略在 `FAQuantizationConfig` 声明：PScale256、scale_alg1、标准BF16层 `[0,3,5,6,7,10,26,27,30,31]`，不做K中心化；完整有效配置与实际计数留证。
2. 公共 `fa_quant.py` 与 `__init__.py` 恢复固定基仓bbd278fe的Git原字节，分别3760B/SHA`7a820fdd735578e34082ca5303599ea3fc13681df346b4970ebb8c85dd8a20a1`、717B/SHA`1a872da61691963c403019a3fe0bf85e79583bf4e15b5dabd54de3a1bc801194`。历史CRLF运行副本的SHA单列，原Hunyuan/UnifiedSP函数与导入保持。
3. Magi显式从 `module.fa_quant.extended` 导入扩展，不替换旧公共函数。扩展保留位置参数，新增keyword-only `scale_alg=0, quant_scale_p=None`；通用校验、量化、layout、真实长度和GQA处理放在原公共目录。PScale是同设备FP32[1]，不替代Q/K/V的原生scale。
4. Attention对象保存构造时的enabled和layer_idx，沿CP包装/custom-op显式传递，生产forward不切换全局开关。标准十层保留完整BF16 FA，其余受支持标准层直接执行动态MXFP8及FA v2并恢复外部shape/dtype。不双算筛选输出、不动态扩充回退名单。
5. flex单K按实际切片长度执行MX，允许Q_S≠KV_S，包括local第0层；标准名单不应用于flex。多K整个Q区间保留原BF16子FA和LSE合并，不混合不同精度的LSE。
6. 合法但未验证的FP16、CP>1，以及标准batch>1/head_dim非128按fallback开关回退或报错；非法设备、dtype、shape、GQA、长度/ranges直接报错。flex只接受TND或singleton BSND。fallback=false也拒绝固定十层精度回退。OOM/设备错误保留原异常。
7. pipeline每次生成通过ContextVar拥有独立PScale，首次MX调用才惰性创建设备Tensor，生成内复用；原Step6同步完成后、视频VAE前释放，异常退出由finally释放并恢复外层上下文。不建立全局设备缓存、不逐FA增加同步。
8. 无SR且CP=1时，两flags在同一原Step6同步之后只pop已被process_output消费的 `data_proxy._saved_data['simple_packed_data']`，下次process_input按原流程重建。不清其他键，不改最终latent、未证明SR/CP路径或MM/FFN/VAE/RNG/cache数学。
9. 普通入口完整生成成功后输出一次主机审计，记录配置、层号、shape/dtype、MX/回退及PScale创建释放。异常不输出成功记录；正式运行不逐FA打印、同步或导出Tensor，数值/profile独立采集。
10. 原冻结BF16、同释放修复BF16、同修复MX分别列出。原峰值门限仍比较最终MX与原冻结同case BF；同修复BF仅控制共享清理，不新增第二官方门限。后两者持平不能声称FA带来额外降峰，也不能把共同proxy释放全部归给量化。

#### 3.2.2 kernel侧设计

不新建 kernel 工程，不修改 CANN tiling。使用现有 `npu_dynamic_mx_quant` 和 `npu_fused_infer_attention_score_v2`，其 query/key/value quant mode 沿用已验证公共实现的 6/6/8 契约。验收 profile 必须同时出现三路动态量化和 MXFP8 FA v2；还需结合调用参数和 tensor dtype/scale 证据，不能仅凭同名算子证明计算精度。

## 支持硬件

| 项目 | 已确认或待冻结条件 |
| --- | --- |
| 芯片 | `asys` 实测 Ascend 950PR_9579 V100，28 AI Core |
| 设备 | 单卡，HBM 131072 MB；初始设备状态健康 |
| Host | x86_64、openEuler 24.03 LTS-SP3，Host 内存 754 GiB |
| 驱动 | `npu-smi` / driver 25.7.rc1.10；固件记录随完整环境证据补齐 |
| CANN | 9.1.0，来自安装包记录 |
| 已验证软件 | 独立 Python 3.12.12、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post4；NPU 可用，单卡张量运算检查成功 |
| MagiCompiler | 官方 commit `7ee218354ca7bd1ca385c270a898326330267206`，固定源码在本地磁盘构建 wheel；导入版本 1.0.0，模型所用 API 导入成功 |

## 算子约束限制

- 本次验收固定单卡 base T2V，448×256、5 秒、32 steps、CFG=2、生成 seed=42、NoCache。初始化 seed与生成 seed分别记录。A2/A3 的 README 性能数据不能充当950 基线。 这里CFG=2指cfg_number=2的条件/无条件两次forward；视频guidance在t>500时用配置值5、t<=500时为2，音频始终5，不将cfg_number误作全程guidance数值。
- case A 为默认 base，局部层为空；需要运行计数证据确认。case B 使用同一 base 权重和完整 pipeline，仅在独立测试配置显式设置 `arch_config.local_attn_layers=[0]`，保留全部生成规格与 `frame_receptive_field=11`。它改变第0层 Attention 的可见域，已按原设计固定，并以同配置 BF16-local 为对照；不替代 case A 的基线。
- case B 每帧视频 Q 对应局部视频 K 和全部音频/文本 K 两个区间，音频/文本 Q 对应一个全序列 K 区间。静态预计每次 flex 有32个双 K 区间和1个单 K 区间；实际 ranges和单 K MXFP8、多 K 回退计数必须留证。保留另一短 prompt 的 base GQA 作为真实不同 S 的辅助用例，不把同一运行重复计为多个 case。
- CP>1 的分布式数值与性能不在单卡验证覆盖内，默认不宣称支持。回退行为仍需检查。
- 多 K flex、未验证 sparse 及实际变长接口保持明确的支持/回退清单。测试中使用构造输入检查 API 边界时，将其标为接口测试，不能替代真实模型 case。

# 可维可测分析

## 精度标准/性能标准

### 基线冻结与证据

infer 基线 commit 为 `bbd278fe5ef772daa310df680865a6826ae6d909`，固定上游 commit 为 `209209b7086eba2020c5439265221495a8357322`。按照模型 README 非覆盖合并上游 inference，并记录覆盖文件与补入文件清单。权重从 README 指定官方模型来源准备，各文件 revision、size、SHA256 和实际本地校验保存在 `weight_manifest.json`；已有共享 Wan2.2 VAE 单独核验。不能只记录下载元数据而称本地文件校验通过。

先执行 `ENABLE_PROFILER=1 bash infer.sh base`。记录完整解析后配置、有效 CLI、所有权重、官方 `example/assets/prompt.txt` 的实际输入内容和文件 hash、软件版本、命令、返回码、原始日志、视频与 profile 获取路径。两档使用同一源码准备方式、依赖、权重和生成输入。

### 已完成的独立能力检查

2026-10-14 在上述 950 环境运行现有公共 helper，Q40/KV8/D128 的以下三个合成输入均返回 BF16、原 Q shape 且无 NaN/Inf：

| Q 长度 | KV 长度 | FA 输出 NRMSE | FA 输出 cosine |
| --- | --- | --- | --- |
| 128 | 128 | 0.0564886 | 0.9984179 |
| 4350 | 4350 | 0.0588906 | 0.9982820 |
| 257 | 4350 | 0.0589234 | 0.9982783 |

实测 Q/K/V 量化输出为 `float8_e4m3fn`，scale 为 `uint8` 存储；Q/K scale shape 为 `[S,N,2,2]`，KV 长度 4350 时 V scale 为 `[68,8,128,2]`。FA v2 实际传入 quant modes 6/6/8、Q40/KV8、TND 和各自累计长度；当前 wheel 接受 NPU 上的 int64 Tensor 长度。保留原始 operator schema、调用参数、JSON 和日志。

这些合成检查只证明已安装算子的基础能力，不作为真实模型case、视频质量、完整DiT block精度或性能验收。官方权重36个文件共78,072,129,263字节和共享Wan2.2 VAE已逐文件完成本地校验。后续完整BF16生成、profile、一次预热和三次正式计时及独立复核均已完成；Total中位为27919.267703ms，三次极差54.617364ms，peak allocated为53,499,608,576字节。一个active去噪步的80次BF16 FA合计93.112129ms，采样窗口不等同完整生成。逐轮原始记录见RFC。

### 数值检查

对相同 Q/K/V 输入比较 BF16 与 MXFP8 FA，另比较同一层输入下完整 DiT block 的输出。收集首层首步和经 RFC 指定的后续层/步，区分单模块误差与去噪传播误差。反量化仅用于离线误差检查，不放入推理路径。

定义 `NRMSE=||Xtest-Xref||₂/max(||Xref||₂,1e-12)`；NRMSE和cosine对CPU FP64展平向量归约，修正大张量FP32统计产生cosine大于1的问题，原错误记录保留。完全相同的非零向量cosine为1；不截断不同向量的统计结果。记录MAE、最大绝对误差、NRMSE、cosine、均值、标准差、p1/p50/p99、min/max、NaN/Inf数量、FP8最大有限值占比和重建超范围比例。零范数reference单独记录绝对误差，不给出虚假的cosine。

| 对象 | 实验前提出的门限 |
| --- | --- |
| 同一输入的 Q/K/V 量化重建 | 各自 NRMSE ≤ 0.08，cosine ≥ 0.995 |
| 相同输入 FA 输出 | NRMSE ≤ 0.10，cosine ≥ 0.99 |
| 同一输入的完整 DiT block | NRMSE ≤ 0.05，cosine ≥ 0.995 |
| 稳定性 | Q/K/V、scale、FA、block 无 NaN/Inf；scale 不是伪造值，FP8 clipping 比例为 0 |

FP8 最大值占比不等同 clipping；先按实际 MXFP8 block scale 计算可表示范围，再判定 clipping。门限提交评审后冻结，失败保留原始结果，不根据量化结果提高容忍度。

### 最终质量

固定五组prompt，覆盖单人、多人、肢体动作、镜头运动和复杂背景，并包含可检查的对白或环境声。BF16/MXFP8保持seed、权重、生成参数和音视频导出条件一致。导出随机A/B标签和隐藏映射，由开发者与老师分别观看并作实际自验记录；记录主体、语义、构图、动作、时序、色彩、音画一致性、伪影及稳定可辨识的退化。两人均未发现稳定退化才通过，不用自动指标或代理人结论代替真人观看。

### 性能

每个真实目标 case、每档至少 1 次完整生成预热和 3 次正式完整生成。单次计时起止处 NPU synchronize；模型加载不混入生成时间。分别记录 Base（去噪）、Decode（音视频 VAE）和 Total，逐次值及平均值、中位数、极差。正式计时不启用 profile、tensor 导出或逐调用同步。

BF16 波动范围固定定义为三次正式 Total 的 `max-min`。通过条件为 `median(BF16)-median(MXFP8)>range(BF16)`，同时 FA 累计耗时降低，计入 Q/K/V 量化、layout 和恢复成本后的 Attention 累计耗时降低。profile 使用独立重复运行，按算子和 Attention range统计累计耗时及调用次数。不能把 FA 内核收益直接当作端到端收益。

### 内存

两档分别记录加载后、每次生成峰值和连续 5 次完整生成后的 NPU allocated/reserved、进程 Host RSS及 `npu-smi` 使用量。使用同一环境和独立新进程；峰值统计在每次生成前重置。通过条件为 MXFP8 NPU peak allocated 小于 BF16；峰值 reserved和设备使用量也报告。输出移至 CPU/文件后释放中间引用，最后记录同步后的稳定状态；若连续增长，调查并保留失败记录。回退与非法参数异常后的资源释放单独验证。

## 兼容性分析

`module/fa_quant` 的旧签名、默认返回类型、dtype和现有 HunyuanVideo 行为保持兼容。新增参数使用 keyword-only；默认行为不启用模型特性。接口用例覆盖标准长度、不同 Q/K 长度、TND/BSND、合法非连续输入、GQA head 比、非法 dtype/layout/长度、配置关闭和回退。

公共接口修改前后检索所有直接调用方并留存结果，对全部调用方做 API/功能验证。HunyuanVideo `single_fp8.yaml` 在相同权重、MatMul 配置、prompt和环境下比较公共模块修改前后，执行代表性精度和至少 1 次预热、3 次正式性能测试。它仅用于回归；其中的权重量化收益不能归入 daVinci 本任务。回归性能下降不超过原 baseline `max-min`，精度沿用调用方原有门限并补充原始结果。

交付模型 README、`docs/models/davinci-magihuman/` 优化说明、公共模块文档、测试和脚本；执行 `pre-commit run --all-files`、仓库 CI和950 实机检查。私仓 `task_submission/` 按任务书提交步骤说明、精度/性能/内存三份 xlsx 和三份原始日志。权重、输入和视频大文件只附受控获取位置与校验信息。代码 PR 关联 RFC、设计 PR、各项自验结果；未完成项明确列出，不申请完成验收。

## 实测精度问题与补充方案（2026-10-16）

### 最终选择与历史边界

最终生产使用标准十层BF16名单 `{0,3,5,6,7,10,26,27,30,31}`、scale_alg1、PScale256，不做K中心化；local0单K仍MX，多K完整BF/LSE。PScale依据保留为[官方固定MXFP8 golden](https://gitcode.com/cann/ops-transformer/blob/7f642b905f1c095051fa3669063ddd708b40cb92/experimental/attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test/fia_fullquant_mxfp8_golden.py)及原真实capture诊断；官方master源码不冒作本环境二进制同版本。最终调用仍以安装接口和实测参数为准。

原算法/仅PScale矩阵失败、旧九回退加layer0标准K中心化的人评FAIL、全生成峰值比BF高512B FAIL，以及旧HY长负载CLI/Health/观察器失败均留在RFC及原交接，不追认通过。七天长批次已撤回、未上传/执行；逐块同步/节流/冷却只属旧诊断包装，已退出本次正式测量和代表回归，不作为验收前置。十回退路径已有五组视频的本人/老师共同相对质量意见，不伪造成十条独立评分，也不替代最终源全部验收。

### 最终源功能与两case精度

原BF权重、loader、参数前缀和post-load保持，实际参数元素15,299,419,136，170个FP32及160个BF16参数对象；没有静态量化权重导出。15个实际加载模块匹配manifest `c871b2d29074441941c11b7291bf052c850ed8cf47e0f09c15373e033072d4e4`，精度collector为 `74ed45f781497a5f401bd7eb40043fcd0e204edf6defa65ebd447b07e59e2335`。

base BF/MX各一次完整functional自然0、125帧和音轨有效、HealthOK；BF媒体SHA复现冻结参考。MX实际5760次动态量化、1920次FA v2、640次BF，PScale创建/释放各1，Step7/8无已完成packed owner或scalar。单次冷启动耗时/峰值不充作正式收益。

两case精度原exit均0，覆盖forward `{1,2,31,32,63,64}`，即step0/15/31条件/无条件×全部40层，每case240个完整hidden+deferred及240个FA对比。向后返回BF参考维持同输入轨迹，不能解释为自由MX视频与BF逐位相同。

| case | 完整层通过；最大NRMSE / 最小cosine | 真实MX FA通过；最大NRMSE / 最小cosine | QKV通过；最大NRMSE / 最小cosine | 单列BF回退 |
| --- | --- | --- | --- | --- |
| base | 240/240；0.0196313195 / 0.9998072913 | 180/180；0.0999333636 / 0.9950035000 | 54/54；0.0270421577 / 0.9996343117 | 60/60位相同 |
| local [0] | 240/240；0.0195175163 / 0.9998095276 | 186/186；0.0980676367 / 0.9951858982 | 72/72；0.0270578765 / 0.9996338778 | 54/54位相同 |

QKV详细样本为真实MX层1/20/39，local另含第0层6个真实单K。Q/K axis=-1、V axis0，E4M3FN/uint8原生E8M0一致；全部零clipping、无NaN/Inf、scale码无255。BF回退max_abs/NRMSE0、cos1，不计作量化通过。完整精度媒体复现各自冻结BF SHA、125视频帧及218音频帧，Health全OK最高93°C。双路比较和CPU统计的时间/峰值不进入benchmark。

最终26项NPU接口原exit0、26/26通过，清理后allocated0/reserved69,206,016B。CP元数据模拟及错误注入的覆盖限制保持，不宣称多卡或真实设备故障恢复通过。准确源、六位置、门限、scale及媒体均已独立复核；原件索引为 `task_work/final_validation_actual_20261016` 的两case `precision.json/source_loaded.json` 与 `final_interfaces_20261016/data/interfaces.json`。

### HY真实调用方代表回归

原设计要求代表性精度及至少1预热+3正式，没有50步完整视频前提。本次保留single_fp8原权重、在线MXFP8 Linear/MM、NoCache、720×1280/129帧/seed42；原sampler真实完成首double block后以明确测试结束信号停止，不执行后续block/scheduler/VAE。

首block实际8次量化MM、11次动态量化、1次FA v2，240个在线Linear真实E4M3FN权重/uint8 scale。保留原未裁剪QKV各 `[1,119056,24,128]`、cu `[0,118811,119056]` 及maxlen/batch；原预处理后有效 `[1,118811,24,128]`，仍覆盖padding回填。此HY权重量化是原有路径，不计为Magi权重改造。

同一真实输入分别执行原Hunyuan Attention和 `UnifiedSPAttention._npu_mxfp8_attention`，before/after交替各1预热+3正式。16回放每次3动态量化+1FA v2，输出均finite/逐字节一致，NRMSE0/cos1。before/after默认源去CRLF后相同，after helper/init准确等固定Git身份；旧函数globals不变，没有reference替代after输出。

| caller | before中位(ms) | after中位(ms) | 差值(ms) | before极差(ms) |
| --- | ---: | ---: | ---: | ---: |
| Hunyuan Attention | 727.902565 | 728.244979 | +0.342414 | 9.678540 |
| UnifiedSP MXFP8 Attention | 737.687225 | 739.277962 | +1.590737 | 5.629425 |

两者差值在原波动内，满足公共兼容不退化条件，不宣称FA加速。runner116.026秒、外层119.687秒自然完成；20条Health/fault全OK/0、最高87°C。无逐块冷却、核数或MatMul参数改动，原件 `hunyuan_public_representative_20261016/result.json`、log/遥测已独立复核。范围为原真实首block及两本地真实caller，full_generations/scheduler_steps/vae_decodes均0；不覆盖50步媒体、多卡/ring或完整after block，旧失败不被覆盖。

### 正式结果与剩余交付

2026-10-16最终源批次的6个阶段均自然exit0，外层总耗时6188.678345秒；该外层时间包含加载和冷却，不作生成时延。每case、每flag在独立新进程执行1次完整预热+5次完整生成，前3次统计性能，5次全部记录内存；独立profile不混入正式Total。输入、权重、seed和规格与各自冻结BF相同，Health全OK、最高93°C。表内时间单位为ms，内存为原始字节B，显示取6位小数，门限使用未舍入原值。

| case | 路径 | 三次正式Total (ms) | 平均 (ms) | 中位 (ms) | 极差 (ms) |
|---|---|---|---:|---:|---:|
| base | 原冻结BF | 27965.956653 / 27911.339289 / 27919.267703 | 27932.187882 | 27919.267703 | 54.617364 |
| base | 同修复BF | 27990.706676 / 27906.497555 / 27827.286722 | 27908.163651 | 27906.497555 | 163.419954 |
| base | 最终MX | 27221.339751 / 27147.323873 / 27251.645850 | 27206.769825 | 27221.339751 | 104.321977 |
| local [0] | 原冻结BF | 28268.151776 / 28321.637001 / 28331.693559 | 28307.160779 | 28321.637001 | 63.541783 |
| local [0] | 同修复BF | 28254.983078 / 28248.976837 / 28340.922065 | 28281.627327 | 28254.983078 | 91.945228 |
| local [0] | 最终MX | 27782.266747 / 27757.198797 / 27736.370237 | 27758.611927 | 27757.198797 | 45.896510 |

base相对原冻结BF的Total中位收益697.927952ms，大于原极差54.617364ms；local收益564.438204ms，大于原极差63.541783ms。同修复BF至MX的中位收益分别684.157804ms和497.784281ms。各档三次正式max/min≤1.10，未以profile或预热值替换正式计时。

去噪/视频VAE/音频VAE的中位另列（ms），不把FA kernel收益当Total，也不要求各阶段中位相加等于Total中位。

| case | 路径 | 去噪中位 (ms) | 视频VAE中位 (ms) | 音频VAE中位 (ms) |
|---|---|---:|---:|---:|
| base | 原冻结BF | 26663.085034 | 570.507460 | 264.268600 |
| base | 同修复BF | 26605.318996 | 573.849515 | 264.245615 |
| base | 最终MX | 25888.459780 | 573.968212 | 264.616412 |
| local [0] | 原冻结BF | 27034.314861 | 577.706090 | 264.509662 |
| local [0] | 同修复BF | 26969.748530 | 576.068514 | 265.047830 |
| local [0] | 最终MX | 26451.409651 | 571.500750 | 264.628501 |

独立profile匹配窗口如下：

| case | 原BF → MX原生FA累计 (ms) | 原BF → MX完整Attention累计 (ms) | MX动态量化kernel：次数 / ms | MX独立layout kernel：次数 / ms |
|---|---:|---:|---:|---:|
| base | 93.897873 → 67.512226 | 93.921371 → 80.923057 | 180 / 2.988917 | 0 / 0.000000 |
| local [0] | 93.122753 → 68.394269 | 124.862971 → 118.556318 | 186 / 3.039769 | 512 / 3.359375 |

独立profile匹配去噪step2、model forward5/6，均含80个完整Attention infra事件；base为80条、local为208条原生FA kernel（含BF回退），按同case原BF匹配窗口比较。完整事件包含Q/K/V量化、layout、恢复和主机调度成本，不能拿kernel之和代替它。base没有单独归因的layout kernel，不表示布局或主机开销为零。原base早期445的93.112129ms仅作历史profile，当前匹配对照采用93.897873ms。

五次完整生成内存（三方归因）：

| case | 原冻结BF峰值 (B) | 同修复BF峰值 (B) | 最终MX峰值，五次各自值 (B) | MX − 原BF (B) |
|---|---:|---:|---:|---:|
| base | 53499608576 | 53496823808 | 53496823808 × 5 | -2784768 |
| local [0] | 53499608576 | 53496823808 | 53496823808 × 5 | -2784768 |

两case的最终MX五次完整生成峰值均比各自原冻结BF严格低2,784,768B，满足原数值门槛；同修复BF的五峰也均为53,496,823,808B。MX与同修复BF持平，未证明FA额外降低完整生成峰值；不能把共享proxy生命周期清理全部归因于量化。同修复BF用于归因，不是新增的第二项官方峰值门限。

两case、两flag的五次释放后NPU allocated均为51,694,914,048B（仍包含模型常驻分配），极差0；完整生成peak reserved均为54,349,791,232B。释放后reserved在前几次扩大后持平：BF为[54223962112,54265905152,54349791232,54349791232,54349791232]B，MX为[54223962112,54286876672,54349791232,54349791232,54349791232]B。Host RSS有平台或小幅变化，不能声称完全不变；VMS、设备全局使用量、加载后状态和逐次原值另存原始runs/遥测，不把npu-smi设备占用当allocator allocated。

| case | 路径 | 五次释放后Host RSS (B，按顺序) | 极差 (B) |
|---|---|---|---:|
| base | 同修复BF | `4708360192, 4710457344, 4710457344, 4712554496, 4712554496` | 4194304 |
| base | 最终MX | `4640907264, 4643004416, 4643004416, 4643004416, 4643004416` | 2097152 |
| local [0] | 同修复BF | `4563111936, 4565209088, 4567306240, 4567306240, 4567306240` | 4194304 |
| local [0] | 最终MX | `4587319296, 4587319296, 4587319296, 4587315200, 4589412352` | 2097152 |

每次完整生成实际原生计数：

| case | 关闭：BF FA v1 | 开启：动态MX量化 / MX FA v2 / BF FA v1 |
|---|---:|---:|
| base | 2560 | 5760 / 1920 / 640 |
| local [0] | 6656 | 5952 / 1984 / 4672 |

每次MX生成的PScale创建/释放各1；Step7/8对已完成packed owner和PScale的弱引用均已失效。该观察不等同所有输入别名或opaque allocator状态全部释放。每side预热加五次的媒体SHA内部一致，BF分别逐字节复现冻结参考；125个视频帧及音频有效且finite。MX自由生成与BF媒体不要求逐字节相同。

正式原件索引：`task_work/final_validation_actual_20261016`下的`final_{base,local}_{bf16,mxfp8}_benchmark_20261016/{runs,source_loaded,device_telemetry}.json`及两MX profile目录；`{base,local}_final_attention_kernel_attribution.json`关联实际trace_view.json/kernel_details.csv。精度原件在两case precision目录，HY原件为`hunyuan_public_representative_20261016/result.json`。汇总`task_work/final_delivery_20261016/final_evidence_reconciliation.json`记录58份来源身份；正式原包`final_formal_20261016_raw.tar.gz`为3,364,649B、SHA256 `81d7fc8358d723dc17b0cf2c696199e56f1df96d0ad70d73714f41e5894e5824`。以上是受控证据索引，权重和视频不纳入源码提交。

本轮原始实测汇总的机器门限已满足；独立复核、代码评审和平台验收分别记录，不以本表代替批准。补充设计评审仍OPEN，代码PR、CI和平台验收未据此完成。任务文件定向hooks通过；全仓pre-commit实际exit1，三项已知基仓失败与本任务文件不相交，不能写全仓检查或CI全绿。
