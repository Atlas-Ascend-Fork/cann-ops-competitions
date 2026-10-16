# Wan2.2-I2V 双专家 MatMul INT8 量化设计（Atlas A3）

开发者：joeb213，邮箱：gai25213050348@sina.com。

本文件按[社区任务流程#39](https://gitcode.com/org/cann/discussions/39)提交至competitions对应任务目录，代码仓设计PR [!1057](https://gitcode.com/cann/cann-recipes-infer/merge_requests/1057)另行保留。

## 1 需求背景（required）

### 1.1 需求来源

本设计对应“10月社区任务-Wan2.2-I2V MatMul INT8 量化（A3）”，实现位于
`cann/cann-recipes-infer` 的 `models/wan2.2-i2v/`。
社区RFC：[cann/cann-recipes-infer#461](https://gitcode.com/cann/cann-recipes-infer/issues/461)；
原个人RFC：[joeb213/cann-recipes-infer#1](https://gitcode.com/joeb213/cann-recipes-infer/issues/1)；
实现 PR：[cann/cann-recipes-infer!1046](https://gitcode.com/cann/cann-recipes-infer/merge_requests/1046)。

Wan2.2-I2V A14B 按扩散时间步使用 high-noise 和 low-noise 两套 DiT，每套包含 40 个 block。
Self-Attention、Cross-Attention 的 q/k/v/o 与 FFN 两层共形成每专家 400、双专家 800 个候选 Linear。
两个专家分别采集敏感度、生成回退清单和转换制品，并在实际切换、offload 后检查状态。

通过双专家profiling分别统计Attention/FFN热点，按敏感度选择量化范围。原始统计与实测结果存放材料私仓。

### 1.2 目标与范围

1. 双专家执行真实 W8A8 INT8 MatMul，输出与原 Linear 的 shape、BF16 dtype 和 bias 语义一致。
2. 用固定校准集确定敏感层，再用新的独立样本验证逐层、完整 block 和最终视频质量。
3. Atlas A3 同机、同配置下，端到端中位耗时下降超过 BF16 三次极差，MatMul 累计耗时降低。
4. 量化权重存储低于 BF16，运行峰值显存不高于 BF16；连续五次生成保持量化状态正确。
5. 提供确定性转换、严格加载校验、关闭量化与按层 BF16 回退，以及可复现的脚本和报告。

FA/Softmax、RoPE、归一化、残差、VAE、T5、Head 和时间/文本嵌入沿用现有计算路径。
正式对照使用单卡 eager、NoCache、CPU offload。

## 2 需求分析与固定基线（required）

### 2.1 来源与运行规格

| 项目 | 固定值 |
|---|---|
| infer 基线 | `bbd278fe5ef772daa310df680865a6826ae6d909` |
| 上游 Wan2.2 | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`，按模型 README 非覆盖补齐 |
| 原始权重 | ModelScope `Wan-AI/Wan2.2-I2V-A14B-BF16`，revision `d51dfab242334a3afee55ead4d7aaafb07502742` |
| 生成参数 | `i2v-A14B`、最大面积 `640*360`、61 帧、40 steps、seed 100、UniPC、shift 5、CFG 3.5 |
| 固定示例实际形状 | 416×544，latent `[16,16,68,52]`，sequence 14144 |
| 专家执行次数 | high/low 为 15/25 步、30/50 次 CFG forward |
| 正式硬件 | Atlas A3 / Ascend910_9382，单卡 device 0，64 GiB HBM |
| 正式软件 | Ubuntu 22.04 aarch64、CANN 9.0.0、Python 3.12.11、torch 2.7.1+cpu、torch_npu 2.7.1.post2、driver 26.1.1 |
| 测量 | 每档完整 40 步预热一次、三次正式生成，另采完整 40 步 profile |

输入、prompt、种子、采样器、offload、cache 与实际输出形状在 BF16/INT8 两档保持一致。
计时包含条件编码、去噪、专家迁移/offload 与 VAE decode，结束前同步 NPU；
模型加载、预热、视频编码和 profile 解析单独记录。

### 2.2 需求拆解

| 需求 | 实现与验证 |
|---|---|
| 算子契约 | `Int8Linear` 管理权重/scale/bias，复用公共 W8A8 `apply_weights` |
| 配置入口 | YAML → 模型私有适配器 → CLI → 模型构建；校验方案、粒度和专家路径 |
| 双专家权重 | 分专家、逐 shard 转换，元数据绑定来源、配置和完整层清单 |
| 设备迁移 | 保留 CPU 规范 buffer，按当前设备创建并复用 runtime view |
| 敏感层回退 | 校准后冻结 `ignored_layers`，从原始 checkpoint 恢复 BF16 权重 |
| 精度与质量 | 同输入逐层/block 对照、五组固定视频、两名指定验收人盲测 |
| 性能与内存 | 三次原始计时、真实 kernel flow、权重存储、HBM 与 Host 连续记录 |
| 工程交付 | 配置与使用说明、自验工具、七件材料、RFC、代码 PR 和设计 PR |

## 3 详细设计（required）

### 3.1 量化数学与 dtype 契约

将 Linear 输入展平为 \(X\in\mathbb{R}^{M\times K}\)，原权重记为
\(W\in\mathbb{R}^{N\times K}\)。权重按输出通道对称静态量化：

$$
s_{w,n}=\begin{cases}\max_k|W_{n,k}|/127,&\max_k|W_{n,k}|>0\\1,&\text{全零通道}\end{cases}
$$

$$
Q_{w,n,k}=\operatorname{clip}\!\left(\operatorname{round}(W_{n,k}/s_{w,n}),-127,127\right).
$$

转换使用 round-to-even。原生 `torch_npu.npu_dynamic_quant` 对 BF16 输入按 token 动态量化，
得到 INT8 \(Q_x\) 和 FP32 \(s_x\)。MatMul 计算 INT8×INT8、INT32 累加，
融合两种 scale 与原 BF16 bias，返回 BF16，再恢复输入的前导维度。

| 对象 | 持久布局 / 算子输入 |
|---|---|
| 权重 | INT8 `[N,K]`，默认以 ND 转置 view 提供 `[K,N]` |
| 权重 scale | FP32 `[N,1]`，算子入口展平为 `[N]`，全部有限且为正 |
| 激活 | BF16 `[...,K]` → `[M,K]` → INT8 `[M,K]` |
| 激活 scale | FP32 `[M]`；全零 token 的 scale=0 与 INT8 全零按原生算子处理 |
| bias / 输出 | 原 BF16 bias；BF16 `[M,N]` → `[...,N]` |

运行时复用 `CompressedTensorsW8A8Int8LinearMethod.apply_weights`。
模型私有 wrapper 负责 FP32 scale 保持、严格加载和迁移状态；公共 `module/quantization/` 保持原实现。

### 3.2 校准、敏感度与回退计划

五组 calibration 样本覆盖人物、动物、运动、镜头变化和复杂背景。
来源、许可、图片校验值、prompt和seed在材料私仓的manifest中固定。

在 BF16 轨迹 steps 0/14/15/39、条件与无条件 CFG 分支捕获同一输入。
v2 每层均匀采样 256 个 token，记录整体与逐 token 相对 L2、余弦、绝对误差及激活统计；
每专家 blocks 0/20/39 用完整输入对照。

| 用途 | 判定条件 |
|---|---|
| 固定验收阈值 | relative L2 ≤ 0.05、cosine ≥ 0.995、全部有限；全零参考时 max abs ≤ 1e-6 |
| v2 校准选择 | 所有观察整体 relative L2 ≤ 0.025、逐 token 最大 relative L2 ≤ 0.05，并满足原余弦/有限值/零参考规则 |
| 敏感层 | 任一校准条件失败即加入所属专家 `ignored_layers`，使用源 BF16 权重 |

选择条件和 ignored 清单在新的独立 v2 验证前冻结。最终计划为：

| 专家 | 候选 Linear | INT8 | BF16 目标回退 |
|---|---:|---:|---:|
| high-noise | 400 | 302 | 98 |
| low-noise | 400 | 312 | 88 |
| 合计 | 800 | 614 | 186 |

独立验证与校准分离。针对激活离群值，采用更严格的校准选择条件并冻结ignored清单；原始观察和调整依据保存于材料私仓。

### 3.3 确定性转换与元数据

`wan.modules.quantization_layers.weight_convert` 逐专家、逐原始 shard 转换，
写入独立输出目录；两专家成功后生成 `conversion.complete.json`。
根配置记录量化 schema、专家目录及量化规则，各专家保留 config、safetensors shard/index 与
`quantization_metadata.json`。

`wan-i2v-w8a8-v1` 元数据记录：专家身份、完整模型配置和摘要、来源 revision、
源 config/index/shard 校验值、量化与 ignored 分区、每层 NK shape、dtype、scale/bias，
以及全部输出 tensor 的 shape/dtype/shard 和量化 shard 大小/校验值。

加载先核对来源、配置、索引和制品，再用 meta 构造模型、安装目标层并 `strict=True` 加载。
缺失、重复、错专家、shape/dtype 错误、非法 scale、制品校验错误均明确报错。
额外 runtime ignored 层从已验证的原始 BF16 checkpoint 恢复。
对相同输入重复转换，并对跨设备重建制品进行一致性校验；记录保存于材料私仓。

### 3.4 专家切换、offload 与布局

规范 INT8 权重、FP32 scale、BF16 bias 注册为持久 buffer。
显式 dtype 转换保持 scale 的 FP32 与 bias 的 BF16。
CPU offload 清除设备 runtime view；迁回 NPU 时按当前设备和布局准备；
同设备重复 `.to()` 与前向复用已准备的 view。

默认布局为 `nd_view`，显式支持 `nd_contiguous` 与 `nz29`。
CPU 保存规范 ND 数据；选择 NZ 时在设备恢复阶段重建布局。
自验记录每层准备次数、scale 状态、专家驻留设备以及 CPU→NPU→CPU 往返结果。

### 3.5 配置与入口

`config/14b_quantization.yaml` 包含 enable、W8A8、动态激活、scale 粒度、
两专家 weight/scale metadata 路径、ignored 清单、BF16 fallback 与 layout。
模型私有配置适配器校验 YAML，并透传实际 CLI 参数；Python 入口再次核对原 YAML。
关闭量化时使用原 BF16 checkpoint。

```bash
cd models/wan2.2-i2v
YAML_FILE_NAME=14b_bf16_benchmark.yaml bash infer.sh
YAML_FILE_NAME=14b_quantization.yaml bash infer.sh
```

运行前按 `config/README.md` 设置权重、输入和输出路径，并保持两档非量化参数一致。
校准、转换与严格加载命令见实现 PR 中的 `docs/models/wan2.2-i2v/quantization.md`。

## 4 验证设计与判定

| 验证项 | 方法 | 通过条件 |
|---|---|---|
| 双专家功能 | 加载、切换、offload、重复迁移、关闭量化、ignored回退 | 配置和状态正确，无NaN/Inf或重复量化 |
| 异常输入 | 缺失/负值/NaN scale、错专家、重复层、来源/config/ignored不匹配 | 明确报错并返回非零 |
| 中间张量 | 同一捕获输入逐层与代表block对照 | relative L2≤0.05、cosine≥0.995；零参考maxabs≤1e-6，全部有限 |
| 最终视频 | 两名指定验收人各五组、七维评价 | 由本人确认有无可稳定辨识的质量退化 |
| 性能 | 每档完整预热一次、正式三次，另采完整profile | 同机同负载，中位减时超过BF16极差，目标MatMul累计耗时下降 |
| 内存与存储 | 权重制品、峰值HBM、连续五次Host/NPU和量化状态 | 权重存储下降，峰值HBM不高于BF16，连续状态稳定 |
| 可复现性 | 确定性转换、来源校验、配置与测量脚本 | 第三方可按文档启动和测量 |

测量工具将生成metadata、trace、算子统计和日志。测试数据、原始实测结果、视频与评分记录提交开发者材料私仓的 `task_submission` 分支，公开目标仓保留实现、配置、测试工具和必要文档。

## 5 指定验收与材料流程

- 公开代码PR：[!1046](https://gitcode.com/cann/cann-recipes-infer/merge_requests/1046)。
- 社区RFC：[cann/cann-recipes-infer#461](https://gitcode.com/cann/cann-recipes-infer/issues/461)。
- 原个人RFC：[joeb213/cann-recipes-infer#1](https://gitcode.com/joeb213/cann-recipes-infer/issues/1)，保留并互相关联。
- 官方流程设计PR：[competitions !2221](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2221)。
- 材料真实分支：[task_submission](https://gitcode.com/joeb213/cann-recipes-infer/tree/task_submission)。

指定验收人为梁横溢（lianghengyi）、李祎杰（yijie19），两人的确认时间与结论均待本人填写。盲测基于公开代码PR展开，沿用现有视频和A/B展示，映射保存在开发侧。
材料分支为每位验收人提供独立盲测包与查看说明，完成后请本人直接在社区RFC评论区回复结论，并注明代码及材料提交ID。
fdu25213050348、m0_73533491的原记录作为开发侧补充评价保存于私仓。

七项必交材料、原始实验结果及从公开PR迁出的结果快照均保留在私仓。代码仓和设计PR按检视意见继续迭代；设计审核、合入、第三方复现和指定验收确认按实际状态记录，完成前保持待验收。
