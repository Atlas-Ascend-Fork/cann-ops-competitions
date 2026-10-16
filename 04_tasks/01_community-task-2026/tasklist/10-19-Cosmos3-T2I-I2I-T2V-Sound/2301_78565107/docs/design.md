# Cosmos3 适配 T2I、I2I 与 T2V+Sound 开发任务书（设计计划）

提交账号：2301_78565107（Kousaka）。任务：10-19-Cosmos3-T2I-I2I-T2V-Sound。

本文说明拟实施方案、测试协议及交付计划，不包含实测成绩或完成声明。

# 需求背景（required）

## 1. 需求来源

依据《Cosmos3 适配 T2I、I2I 与 T2V+Sound 场景(A2, A3)》及随附 `test_cases/README.md`。目标代码仓为 [cann/cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai)，沿用 `world_model/cosmos3/` 的 overlay、`npu_adapt.sh` 和 adaptor_patches 机制，技术说明更新至 `docs/world_model/cosmos3/README.md`。代码 PR 关联开发前的 RFC/Issue。

## 2. 背景和目标

既有 NPU 样例覆盖 T2V、I2V、V2V；锁定的 Cosmos 上游已定义 text2image、image2image 及 text2video 的 sound 能力。本任务拟以最小设备兼容改动接通 T2I、I2I、T2V+Sound，保持上游生成语义和既有三场景默认行为。不得以无声视频代替音频支持，不训练模型、不重写音频 tokenizer，不以减少分辨率、帧数或采样步数规避回归。

# 需求分析（required）

## 3. 版本和资产冻结

| 项目 | 计划配置及约束 |
| --- | --- |
| Recipes 起点 | db9b2453ad70ac919f8c42e02a61d16db63f4e66；记录 overlay 与原有 patch 哈希 |
| Cosmos 上游 | a61b292ccbc337c25c0c82d2a8cca75feef96cee，OpenMDW-1.1；保留许可证及版权声明 |
| 模型 | nvidia/Cosmos3-Nano，revision e59a53c25979a090fa8706c9acc0c254a6e89b92；不用缺少音频能力的 Edge 变体 |
| 附加 VAE | Wan-AI/Wan2.2-TI2V-5B，revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e 中的 Wan2.2_VAE.pth |
| 资产清单 | transformer 分片、配置、文本/图像编码资产、image VAE、sound tokenizer/AVAE 均记录相对路径、来源 revision、大小和 SHA256；下载后逐文件校验 |
| 平台 | Atlas A3 与 Atlas A2 各自建立改动前基线及改动后验证，不以跨平台结果代替同平台对照 |
| 软件 | 计划采用 CANN 9.0.0、PyTorch 2.10.0、torch_npu 2.10.0、Python 3.13；依照样例依赖安装，并冻结实际 Python 补丁版本、完整依赖、驱动、固件和编码库版本 |
| 通用运行配置 | 单卡，parallelism-preset=latency，resolution=480，seed=0，--no-guardrails；保持上游各组件 dtype/autocast 设置并记录，不自行改变精度类型 |
| 测试资产 | 六组固定 JSON、prompt、输入媒体哈希、实际解析后的帧数、步数、宽高比、采样器、编码参数在正式对照前写入 RFC 清单；两平台共用同一清单 |

权重及媒体采用官方来源；镜像下载仅作为传输方式，必须核对相同 revision 和文件哈希。补丁只针对锁定上游，不隐式升级版本。安装后核查实际运行库和设备推理，不能仅以安装目录存在判定环境正确。

## 4. 范围与交付接口

六个场景共用 `python -m cosmos_framework.scripts.inference`，由 torchrun 拉起。旧场景使用锁定上游 `inputs/omni/t2v.json`、`i2v.json`、`v2v.json`；新增配置遵守同一 JSON 契约。不另造替代官方推理链路，不实现 CP/CFGP/FSDP 优化。

| 场景 | 配置和语义 | 检查目标 |
| --- | --- | --- |
| T2I | model_mode=text2image，固定 prompt/seed，沿用单帧默认配置 | 仅一张有效 vision.jpg，尺寸和格式正确，模型输出有限 |
| I2I | model_mode=image2image，提供本地 vision_path 和 prompt | 正确预处理、条件编码及 image VAE；替换输入图能改变输出 |
| T2V+Sound | model_mode=text2video，enable_sound=true，带 sound 资产的 checkpoint | 音视频均生成，非空非全零音频，时长对齐并可解封装和播放 |
| T2V/I2V/V2V | 保留现有 JSON、默认参数、attention 和本地权重路径 | 改动前后同平台原始输出、逐帧图像、性能及显存回归 |

# 详细设计（required）

## 5. 基线及改动边界

先分别在两平台用既有 overlay 和 npu_adapt.sh 建立未修改基线，保存三类旧场景的原始 latent/模型输出、最终制品、配置、耗时及内存。原始基线与适配版分目录保存，禁止覆盖。检查确定性及运行错误时保留证据，不把无法比较计为通过。

实现集中于 Cosmos3 样例目录：检查 device 字符串、stream、autocast、同步和 attention 调用，复用既有 CANN backend；新增分支仅在对应模式或 sound 开关启用时生效。默认模式、无 sound 路径、本地 checkpoint 及并行参数的解释保持兼容。如必须修改公共 module 或共享 backend，先列出全部直接调用方，补齐关闭新功能时的功能、精度和代表性性能回归。

## 6. T2I / I2I

T2I 沿用上游 text2image 参数解析、单帧 latent、VAE 解码和图像后处理，不把视频生成后抽帧作为替代。I2I 沿用 image2image 的输入读取、缩放/宽高比、条件编码及采样路径，校验 vision_path 存在、媒体类型和尺寸合法；不以忽略图像条件的 T2I 分支替代。

提供最小合法 JSON 和参数说明。测试配置显式冻结上游实际解析值，避免省略字段后因版本漂移而改变工作量。I2I 除固定输入回归外，固定其余参数、替换不同图像，验证条件张量及最终输出确实变化。

## 7. T2V+Sound

沿用上游 sound 开关和音视频联合生成接口，核对 AVAE/tokenizer 的配置、权重、dtype、device 和返回结构。音频模块按需加载，禁用 sound 时不额外占用音频常驻资源；重复请求正确释放临时张量并重置状态。CPU 编码/封装保留上游职责，GPU 专属依赖若不能在 NPU 使用则在样例内做等价适配，不改变采样算法。

预检 sound tokenizer、checkpoint 能力和编码器；缺失依赖须非零退出并给出可定位错误。封装后检查视频轨、音频轨、采样率、声道数、有效采样数和有效时间戳，并解封装检查非空、非全零和播放效果。拟冻结 A/V 有效时长差阈值为一个视频帧周期加一个音频采样周期（1/fps + 1/sample_rate 秒）；编码器声明的 priming/padding 依容器有效时间戳处理并单独记录，不事后截短内容以通过检查。

## 8. 输出、异常与可复现性

每次调用使用独立输出目录，保留解析后的 sample_args 和输出索引，防止上游已存在输出跳过机制污染计时。测试辅助脚本负责调用官方入口、记录和比较，不替换实际生成流程。原始张量采集仅用于对照，关闭时不能改变正式推理路径；profiling 独立运行，不将 profiler 开销混入性能结果。

覆盖六类异常：缺失 sound tokenizer、缺失编码器、错误输入类型、缺失 vision_path、非法尺寸、错误 checkpoint。分别断言非零退出、错误信息指向具体配置或资产，禁止吞异常或静默降级为无声输出。

# 可维可测分析

## 9. 双平台验收矩阵

A2 与 A3 分别执行下列全部项目；权重和六组输入一致，每个平台内部使用同一环境比较改动前后。

| 类别 | 执行与判定 |
| --- | --- |
| 新场景功能 | T2I、I2I、T2V+Sound 按第 4、6、7 节检查，原始输出无 NaN/Inf；保存结构检查与真实制品证据 |
| 旧场景精度 | T2V、I2V、V2V 原始输出 shape、dtype 和逐元素值一致，拟采用 atol=0、rtol=0；仅容器元数据变化时，解码后逐帧像素必须一致。发现原始非确定性先定位并向维护者反馈，不自行放宽阈值 |
| 旧场景性能 | 每场景改动前后各至少预热 1 次、正式 3 次，记录每次端到端耗时、中位数和基线波动区间；按任务书“稳态端到端中位耗时不劣于各自基线重复测量波动下界”验收，不以单次最快值代替中位数 |
| 新场景性能 | 每场景预热至少 1 次、正式至少 3 次，记录完整端到端原始耗时和中位数，不虚设新增场景加速门槛 |
| 内存 | 六场景各开启 profile_memory，记录 Host 峰值与单卡 NPU 峰值；旧场景无无法解释的增长 |
| 连续调用 | 同一服务/模型生命周期内新旧场景混合至少 5 次，逐次记录结束后显存、输出目录和状态；扣除可解释框架缓存后不持续增长、不 OOM、不串请求 |
| 异常 | 六类异常逐项验证，非零退出且错误可定位 |
| 工程 | overlay/patch 可从干净上游应用，统一 CLI 可运行，License、pre-commit 与仓库 CI 合规 |

性能的波动下界统计解释在 RFC 中于正式测量前确认；同时保留基线 min/median/max 与全部原始数据，不用事后选择有利口径。端到端计时固定起止范围并在设备完成后停止计时，冷启动/权重加载与稳态记录分列，前后保持相同定义。

## 10. 复现命令模板

```bash
export COSMOS_CHECKPOINT=/path/to/Cosmos3-Nano
SCENE=t2v  # 依次替换为 i2v、v2v、t2i、i2i、t2vs
RUN_ID=run01
torchrun --nproc-per-node=1 -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i inputs/omni/${SCENE}.json -o outputs/${SCENE}/${RUN_ID} \
  --checkpoint-path "${COSMOS_CHECKPOINT}" \
  --resolution=480 --seed=0 --no-guardrails
```

复现说明提供环境安装、资产获取和校验、原版基线、适配补丁、六场景运行、异常验证、比较及汇总的完整顺序。所有路径用相对路径或占位符，不包含私人地址及凭据。

## 11. 兼容性、风险与处理

若锁定模型或提交缺少能力，先核查上游契约及资产完整性；确需升级时经 RFC 记录并重建两平台旧场景基线，不混用版本结果。若资源不足，先按既有官方配置分析峰值，不改变精度、采样步数或输出规格规避。对音频依赖、封装延迟、NPU 不支持路径和重复调用泄漏逐项定位；错误归因区分输入、资产、适配代码和运行库，不把基础设施失败冒充测试通过。

# 实施与交付计划

1. 提交设计计划与 RFC，冻结上游、资产和测试协议，建立两平台旧三场景基线。
2. 接通三类新场景、补齐依赖预检与按需加载，完成单场景 smoke 和异常检查。
3. 完成两平台六场景、旧场景精度/性能、混合调用及 profile_memory 验证。
4. 整理复现说明、自验收报告、逐次 CSV、日志索引、真实截图和制品哈希；逐项对应任务书，不把计划或冒烟测试写为全量通过。
5. 提交仅包含本样例 overlay、patch、配置、测试和文档的代码 PR，关联 RFC，完成 pre-commit 和 CI 后申请合入。

交付目录沿用 `world_model/cosmos3/` 与 `docs/world_model/cosmos3/README.md`。不提交完整上游代码、权重、输入媒体、生成媒体、凭据或大体积日志；报告给出可追溯的制品地址和哈希。验收包根目录提供总说明、总结果表和自验收报告，详细证据以索引关联，避免重复堆叠目录。
