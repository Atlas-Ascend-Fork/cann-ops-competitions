# 需求背景（required）

## 需求来源

本设计对应「10月社区任务-daVinci-MagiHuman FA MXFP8 量化任务书（A5）」及配套 `test_cases/README.md`，目标代码仓为 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)。流程采用[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)、[社区任务目录说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)和目标仓 `CONTRIBUTION.md`。

本文为待评审设计，提交团队为 `chilpirate`，用户已确认活动报名审核通过。950 软件环境和权重校验已完成，独立合成算子 probe 成功；完整 BF16 模型基线仍在采集。文中门限是实验前提出的验收条件，尚无模型量化通过结论。关联 RFC：[https://gitcode.com/cann/cann-recipes-infer/issues/432](https://gitcode.com/cann/cann-recipes-infer/issues/432)。设计 PR：[#2143](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2143)；基线运行记录齐备且设计评审通过后冻结修订版，再开展模型量化接入。

## 背景介绍

daVinci-MagiHuman 通过 40 层 DiT 联合生成视频和音频。固定上游配置的 hidden size 为 5120，head dimension 为 128，Q 为 40 heads，K/V 为 8 heads，属于 GQA。base 模式的 `local_attn_layers` 为空；局部 Attention 使用 flex 的 Q/K 区间与 softmax LSE 合并，不可直接用全序列稠密 Attention 替换。

模型现有 `flash_attn_func` 和 `flex_flash_attn_func` 使用 BF16 `npu_fused_infer_attention_score`。公共 `module/fa_quant/` 已有动态 MXFP8 量化与 FA v2 封装；当前封装以 BSND 输入、TND 计算和基础非 sparse Attention 为主，错误信息、layout 和 GQA 契约仍需检查。当前直接调用方为 HunyuanVideo Attention 和 `module/unified_sp/core.py`；实施前再次检索并冻结影响清单。

本任务仅量化 Attention 的 Q/K/V 激活，复用现有 CANN 算子。MatMul 权重量化、RoPE 融合、缓存和图模式不纳入收益。

# 需求分析（required）

## 需求描述

在 Ascend 950 单卡上，使标准 Attention 和经 RFC 选定的真实额外 case 执行动态 MXFP8 FA。输出保持原 dtype、shape、GQA 和局部范围语义；未支持组合明确回退或报错。通过中间张量比较、视频盲测、完整生成性能、峰值内存和公共调用方回归后，提交代码 PR。

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

Q/K 按最后一维、V 按公共封装规定的 token 维执行 `npu_dynamic_mx_quant`，分别得到 FP8 值和 E8M0 scale；scale 直接交给 `npu_fused_infer_attention_score_v2`。真实轴、scale shape 与 padding 契约以所安装版本的接口和实机 probe 为依据，冻结到 RFC。运行路径不执行「反量化后调用 BF16 FA」。

flex 的同一 Q 区间如果有多个 K 区间，需要按各区间的 softmax LSE 合并：

\[
L=\operatorname{logsumexp}(L_1,\ldots,L_m),\qquad O=\sum_i \exp(L_i-L)O_i.
\]

不能简单求平均、丢弃区间或把区间外 token 纳入 Attention。FA v2 的 MXFP8 LSE 支持尚待 probe。若不能取得满足现有合并语义的 LSE，多 K 区间继续使用完整的原 BF16 实现，并记录回退原因。

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

1. 新增 `config/fa_mxfp8.yaml`，复制 `base.yaml` 的完整字段，仅增加 `fa-mxfp8: true`、`fa-mxfp8-fallback: true`。补齐 `infer.sh` 模式入口和 CLI 参数解析，使设置传到实际模型 Attention；默认关闭。
2. 复用 `module.fa_quant.npu_mxfp8_attn`。模型仅保留开关、CP/flex 条件判断、range/LSE 组合与回退；通用量化/layout/长度/GQA 参数处理放在已有公共目录，禁止复制平行通用实现。
3. 公共接口保持现有 `npu_mxfp8_attn(q,k,v,dst_type,softmax_scale)` 调用成立；若新增 layout 或实际长度参数，使用 keyword-only。默认返回一个原 shape 的 tensor。第一版 flex 单 K 路径直接包装 singleton BSND，不需要修改返回类型或引入 LSE 参数。
4. 标准路径量化 Q/K/V，传真实 scale、Q/KV head 数、`1/sqrt(D)`、正确 actual lengths 与非 sparse 参数，调用 FA v2，再恢复外部 shape/dtype。
5. 单 K 区间 flex 复用同一公共实现：Q 为 `[1,126+L_text,40,128]`，K/V 为 `[1,S,8,128]`，分别传 Q/KV 的真实长度。多 K 的整个 Q 区间按原逻辑 BF16 回退，保留两次或多次子 Attention 和 LSE 合并。本版不宣称 flex 视频查询已量化；多 K MXFP8 若以后扩展，需另行确认量化 LSE 契约与误差。
6. CP>1、未支持 sparse/layout/dtype/shape 等组合：回退开关打开时走原 BF16 分支，按原因输出可审计日志；关闭回退时明确报错。参数错误不作为普通能力回退吞掉，OOM、设备异常等错误保留原始上下文并抛出。
7. 审计日志记录模式、分支、shape、heads、长度、dtype、scale shape和回退次数。性能正式运行避免每次打印或 CPU/NPU 同步；诊断、profile、数值和性能分别运行。

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

- 本次验收固定单卡 base T2V，448×256、5 秒、32 steps、CFG=2、生成 seed=42、NoCache。初始化 seed与生成 seed分别记录。A2/A3 的 README 性能数据不能充当950 基线。
- case A 为默认 base，局部层为空；需要运行计数证据确认。case B 提议使用同一 base 权重和完整 pipeline，仅在独立测试配置显式设置 `arch_config.local_attn_layers=[0]`，保留全部生成规格与 `frame_receptive_field=11`。它改变第0层 Attention 的可见域，需 RFC 评审认可，并以同配置 BF16-local 为对照；不替代 case A 的基线。
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

这些检查只证明已安装算子的基础能力，不作为真实模型 case、视频质量、完整 DiT block 精度或性能验收。冷启动时延不计入性能报告。官方权重 36 个文件共 78,072,129,263 字节和共享 Wan2.2 VAE 已逐文件完成本地 SHA256 校验；完整 BF16 生成、profile 和正式重复计时仍待采集。

### 数值检查

对相同 Q/K/V 输入比较 BF16 与 MXFP8 FA，另比较同一层输入下完整 DiT block 的输出。收集首层首步和经 RFC 指定的后续层/步，区分单模块误差与去噪传播误差。反量化仅用于离线误差检查，不放入推理路径。

定义 `NRMSE=||Xtest-Xref||₂/max(||Xref||₂,1e-12)`；cosine 对 FP32 展平向量计算。记录 MAE、最大绝对误差、NRMSE、cosine、均值、标准差、p1/p50/p99、min/max、NaN/Inf 数量、FP8 最大有限值占比和重建超范围比例。零范数 reference 单独记录绝对误差，不给出虚假的 cosine。

| 对象 | 实验前提出的门限 |
| --- | --- |
| 同一输入的 Q/K/V 量化重建 | 各自 NRMSE ≤ 0.08，cosine ≥ 0.995 |
| 相同输入 FA 输出 | NRMSE ≤ 0.10，cosine ≥ 0.99 |
| 同一输入的完整 DiT block | NRMSE ≤ 0.05，cosine ≥ 0.995 |
| 稳定性 | Q/K/V、scale、FA、block 无 NaN/Inf；scale 不是伪造值，FP8 clipping 比例为 0 |

FP8 最大值占比不等同 clipping；先按实际 MXFP8 block scale 计算可表示范围，再判定 clipping。门限提交评审后冻结，失败保留原始结果，不根据量化结果提高容忍度。

### 最终质量

固定五组 prompt，覆盖单人、多人、肢体动作、镜头运动和复杂背景，并包含可检查的对白或环境声。BF16/MXFP8 保持 seed、权重、生成参数和音视频导出条件一致。导出随机 A/B 标签和隐藏映射，由 2 名指定验收人独立观看；记录主体、语义、构图、动作、时序、色彩、音画一致性、伪影及稳定可辨识的退化。两人均未发现稳定退化才通过。验收人尚待指定，不用自动指标或代理人结论代替真实盲测。

### 性能

每个真实目标 case、每档至少 1 次完整生成预热和 3 次正式完整生成。单次计时起止处 NPU synchronize；模型加载不混入生成时间。分别记录 Base（去噪）、Decode（音视频 VAE）和 Total，逐次值及平均值、中位数、极差。正式计时不启用 profile、tensor 导出或逐调用同步。

BF16 波动范围固定定义为三次正式 Total 的 `max-min`。通过条件为 `median(BF16)-median(MXFP8)>range(BF16)`，同时 FA 累计耗时降低，计入 Q/K/V 量化、layout 和恢复成本后的 Attention 累计耗时降低。profile 使用独立重复运行，按算子和 Attention range统计累计耗时及调用次数。不能把 FA 内核收益直接当作端到端收益。

### 内存

两档分别记录加载后、每次生成峰值和连续 5 次完整生成后的 NPU allocated/reserved、进程 Host RSS及 `npu-smi` 使用量。使用同一环境和独立新进程；峰值统计在每次生成前重置。通过条件为 MXFP8 NPU peak allocated 小于 BF16；峰值 reserved和设备使用量也报告。输出移至 CPU/文件后释放中间引用，最后记录同步后的稳定状态；若连续增长，调查并保留失败记录。回退与非法参数异常后的资源释放单独验证。

## 兼容性分析

`module/fa_quant` 的旧签名、默认返回类型、dtype和现有 HunyuanVideo 行为保持兼容。新增参数使用 keyword-only；默认行为不启用模型特性。接口用例覆盖标准长度、不同 Q/K 长度、TND/BSND、合法非连续输入、GQA head 比、非法 dtype/layout/长度、配置关闭和回退。

公共接口修改前后检索所有直接调用方并留存结果，对全部调用方做 API/功能验证。HunyuanVideo `single_fp8.yaml` 在相同权重、MatMul 配置、prompt和环境下比较公共模块修改前后，执行代表性精度和至少 1 次预热、3 次正式性能测试。它仅用于回归；其中的权重量化收益不能归入 daVinci 本任务。回归性能下降不超过原 baseline `max-min`，精度沿用调用方原有门限并补充原始结果。

交付模型 README、`docs/models/davinci-magihuman/` 优化说明、公共模块文档、测试和脚本；执行 `pre-commit run --all-files`、仓库 CI和950 实机检查。私仓 `task_submission/` 按任务书提交步骤说明、精度/性能/内存三份 xlsx 和三份原始日志。权重、输入和视频大文件只附受控获取位置与校验信息。代码 PR 关联 RFC、设计 PR、各项自验结果；未完成项明确列出，不申请完成验收。
