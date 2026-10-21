# 需求背景

## 背景与目标

HY-Embodied-0.5-X 面向具身场景理解，采用 MoT-2B 架构，总参数约 4B、单次激活约 2B；上游推理依赖特定 Transformers 与 CUDA attention 接口。
本方案拟复用官方 `HyEmbodiedPipeline` 契约，通过小范围可维护适配在 Atlas A3 上提供图像＋文本推理，建立可复现的首份 NPU 评测基线。
精度目标是官方 human-verified evaluation 全量 3,640 题 Accuracy ≥80%，即至少 2,912 题正确；当前不填写未完成的实测总分或预计成绩。
支持本地图像＋prompt 的单样本／批量调用和 direct／thinking；不训练、不使用 EmbSpatial-SFT，不将视频、分布式推理或模型量化扩展为本次交付。
OpenAI-compatible server 为可选项，本方案以 CLI／Python runner 为入口，不将部署服务列为新增验收条件。

# 需求分析

## 功能与约束

| 类别 | 设计要求 |
| --- | --- |
| NPU 执行 | 视觉编码器、MoT 路由、语言主干、KV Cache、lm head 的核心计算在 NPU 执行，不静默回落 CPU |
| 适配方式 | 复用官方模型和 processor；不复制完整上游、不伪造 CUDA 包、不自动改走 CPU |
| 可配置性 | 显式设置权重／processor 路径、device、dtype、token 上限、temperature、top-p、seed、thinking |
| 数据与计分 | 原问题、原四选项、原图像及原序；逐题唯一；GT 只用于独立计分 |
| 复现 | 固定源版本、依赖、资产、协议和配置；不匹配快速报错；不自动升级 |
| 修改范围 | 新增单个模型样例，不计划修改仓库公共模块或迁移其他模型代码 |

文件解码、图像预处理、tokenizer、结果解析和计分属于主机职责；整数／bool 元数据传输与模型浮点激活的 CPU 计算分开审查。
如实际开发确需公共代码修改，应单独列明所有直接调用样例及功能／精度回归；不将无关公共清理纳入本设计。

## 目标平台与资源预算

| 项目 | 拟锁定值及范围 |
| --- | --- |
| 实机 | HiDevLab Atlas 800T A3；既有 DMI／npu-smi 记录可作平台依据 |
| 卡与芯片 | 实例暴露两个 Ascend910_9382 逻辑芯片；本次仅使用一个 `npu:0`，不张量并行 |
| 单设备 HBM | 运行时报告约 62,740 MiB 可用；不将两个芯片合并为一个 128 GiB 设备 |
| 系统／架构 | Ubuntu 22.04、ARM64 |
| HDK／固件 | 26.1.1／9.0.0.9.220；验收清单记录实际读取值 |
| CANN／Python | 9.1.0／3.12（既有实例为 3.12.13） |
| PyTorch／torchvision | 2.10.0／0.25.0 |
| torch_npu | 2.10.0.post6 |
| 精度 | BF16 权重和主计算；保留上游必要稳定性计算的 dtype，索引保持整数 |
| 峰值预算 | 拟单设备 allocated 峰值 <60 GiB；总占用不得超过实际可用 HBM，均待实测验证 |

allocated、reserved、运行时其他占用及 Host RSS 分别报告，不能仅凭 allocated 预算证明不会 OOM；不宣称其他逻辑芯片已完成模型验证。

## 模型、数据和版本锁

| 对象 | 固定版本／revision |
| --- | --- |
| HY 官方源码 | [Tencent-Hunyuan/HY-Embodied-0.5-X](https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X/tree/cf23678a76dc9c445ab7f7aea7a9a8f60e910a63)：`cf23678a76dc9c445ab7f7aea7a9a8f60e910a63` |
| Transformers | [固定源码](https://github.com/huggingface/transformers/tree/9293856c419762ebf98fbe2bd9440f9ce7069f1a)：`9293856c419762ebf98fbe2bd9440f9ce7069f1a` |
| 模型权重／processor | [tencent/HY-Embodied-0.5-X](https://huggingface.co/tencent/HY-Embodied-0.5-X/tree/79f8301d8a80dd03ba778c58d6429e414e9ef5d3)：`79f8301d8a80dd03ba778c58d6429e414e9ef5d3` |
| EmbSpatial 官方仓 | [固定源码](https://github.com/mengfeidu/EmbSpatial-Bench/tree/75792fc5957b0142fab8ff1d47574ffcf73e02c9)：`75792fc5957b0142fab8ff1d47574ffcf73e02c9` |
| human-verified 数据 | [Phineas476/EmbSpatial-Bench](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench/tree/e2733cacf0f20371518c687176e881d79644e61e)：`e2733cacf0f20371518c687176e881d79644e61e` |
| 本地实现参考 | `0395e5ee1b71d1c9b30e91b8de34cc9a75648932`，仅作本设计技术素材，不是批准或验收版本 |

最终实现提交须在设计通过后另行绑定；文件大小、SHA256、下载来源和依赖 wheel 清单放在资产／运行索引，不在正文堆叠逐文件摘要。
HY 源码和权重采用 Tencent HY Community License Agreement，不误写为 Apache-2.0；数据卡标注 CC-BY-4.0，交付保留原作者归属和许可原文。
新增代码遵循目标仓许可，第三方依赖逐项保留许可／告知清单；不据 metadata 标签替代实际 LICENSE，也不作额外法律结论。
权重、数据、base64 图像、缓存、完整大型输出／trace 和任何凭据均不提交代码仓。

# 详细设计

## 模块划分与数据流

```text
原图＋原问题／四选项 → 官方 processor／chat template（主机）
  → 视觉编码器 → MoT 路由／attention → 语言主干／KV／lm head（NPU）
  → 原生成过程 → 最终答案解析 → 独立计分与分类统计（主机）
```

`hy_npu` 负责版本／资产校验、NPU 加载、attention 适配、可选路由及设备审计；`evaluation` 负责数据白名单、协议、子进程监督、结果与独立复算。
上游文件保持原样；只在导入边界校验固定源码并替换 CUDA FlashAttention 映射，复用已有 NPU wrapper 和 `npu_fusion_attention`。
保留因果方向、视觉双向覆盖、padding、KV 及原逐样本投影形状；这些语义仍须用固定输入的目标设备结果验证，不静态承诺跨后端 bit-exact。
Liger 不作为本推理样例的必装依赖；不启用未经验证的自动设备映射、CPU offload、图编译或自动依赖升级。

## 路由、静态资源与设备检查

拟启用的路由优化只覆盖无梯度、单 token、当前全部有效且纯文本的解码：按原逐样本二维 GEMM 执行，再 stack。
prefill、视觉混合、padding 或不满足约束的布局回到原 helper；只读取必要 bool 元数据，不搬运浮点激活，也不保存跨样本张量缓存。
原生 attention 静态 mask 在构造期通过原 helper、同一 `torch.device` key 幂等初始化；不调用 model forward／generate 作隐式预热。
加载基线在权重、设备检查和真实静态资源初始化完成后采集；不能把首次合法静态分配误判为上一题 KV 残留，也不以缓存清空掩盖增长。

两种审计模式均保留 native PrivateUse1 fallback blocker、参数／buffer 放置、全部 module 输出与 KV 设备检查、根 logits NaN／Inf 检查。
native blocker 对安装时实际会选择 backend fallback 的已知 schema 注册抛错实现，保留已有 NPU／合法 composite 实现；如存在晚注册未覆盖项须明确核查，不能声称永久覆盖未来算子。
`enable_python_dispatch_audit=True` 为通用默认值，额外逐可见 ATen 检查输入／输出浮点数组设备和含 NPU 输入操作首次遇到的分派条目。
recipe 拟采用显式 False 减少主机审计开销，作为本次设计待评审选择；False 仅省去该 Mode，保留路由上下文及上述检查，并记录实际模式和失去的额外覆盖。
False 不能实时发现模块内部显式 CPU 浮点计算后转回 NPU 的闭环；模块输出在 NPU 不足以单独证明全部核心计算在 NPU，两种覆盖不宣称等价。
将固定源码审阅、native 阻断负例、设备／trace 记录结合验证；PythonMode 也可能改变 composite 分解或分配路径，模式切换必须目标数值与性能实测。

## 数据准备与 GT 隔离

只读取官方 `embspatial_bench.json`，不下载／读取 `embspatial_sft.json`；校验源文件、revision、唯一 ID、四个选项及 0-based answer。
按原 JSON 数组顺序规范化为 JSONL；图像保持原始字节，不裁剪或重编码，保存图像内容摘要和来源关联。
标准化记录保留计分所需 GT／relation，但模型输入构造只允许原图、问题和四选项；去除 objects／bbox 等非协议字段，GT 不进入消息、模板或生成参数。
模型文件与数据传输到目标机后再实际复验；规范化文件同时绑定源数据与自身摘要，禁止仅信清单自述或传输成功状态。

| relation | close | far | above | under | left | right | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 样本数 | 612 | 594 | 596 | 602 | 616 | 620 | 3,640 |

## 评测协议与答案解析

拟固定 generation＋thinking，正式 batch size=1，单题只使用一张对应图像；direct／批量仅作功能与稳定性测试，不混入正式成绩。
采用权重原始 chat template；图像由固定 Qwen2VLImageProcessorFast 处理，min_pixels=25,088、max_pixels=4,194,304、patch=16、merge=2、temporal_patch=1。
RGB、mean／std=[0.5,0.5,0.5]、rescale=1/255、resample=3；不因显存或成绩更改预处理，不追加外部 resize。
不添加 system、few-shot 答案、relation 或坐标；精确 prompt 模板如下：

```text
{question}
A. {original option 0}
B. {original option 1}
C. {original option 2}
D. {original option 3}
Select exactly one option. Put only its uppercase letter (A, B, C, or D) inside <answer></answer> for your final answer.
```

| 项目 | 拟固定值 |
| --- | --- |
| 生成 | max_new_tokens=32768、temperature=0.05、top_p=1、top_k=50、do_sample=True、use_cache=True |
| 随机数 | 基础 seed=42，按 `sha256-question-id-v1` 派生逐题 seed，恢复不依赖已运行题数 |
| 特殊 token | BOS=120000、EOS=[120020]、PAD=120002；其余默认由固定生成配置锁定 |
| 时限 | 每题生成 600 秒；profiler 导出／分析另记 3600 秒，不扩大生成时限 |

解析优先接受最终闭合 `<answer>` 内唯一大写 A/B/C/D；拟保留显式最终答案行后备，其可接受语法随解析器和配置一次性冻结，不从普通解释中猜字母。
thinking 的输入预填 `<think>` 与生成 suffix 分开记录，结构校验按真实 completion prefix 处理；不扫描推理文本或依据 GT 挑选答案。
缺失、矛盾、多答案、非法格式、空文本均按不可解析计错；有真实 token 但 decode 为空时保留 token 数和 EOS／长度终止事实，不伪造推理异常。
主指标为 `correct / 3640 × 100%`；六类各报告 correct／total／accuracy，宏平均不替代主指标。无法解析、OOM、超时和其他异常均保留在分母中。

## 调用、异常与断点恢复

以下为最终实现接口示意，不在设计阶段执行；`SAMPLE_DIR`、`HY_WORK`、`IMAGE`、`PROMPT`、`OUT` 由操作者填写为样例目录、仓外资产、真实输入和新输出目录。
`MODEL_REVISION` 取上表权重 revision；`RUN_CONFIG`、`DATASET_JSONL`、`RUN_MANIFEST` 分别指冻结的仓外配置、规范化数据和首次运行清单，不是新的可调协议。

```bash
cd "$SAMPLE_DIR"
python -B infer.py \
  --model "$HY_WORK/assets/HY-Embodied-0.5-X" \
  --upstream-path "$HY_WORK/sources/HY-Embodied-0.5-X" \
  --weight-revision "$MODEL_REVISION" \
  --device npu:0 --dtype bfloat16 --inference-routing \
  --no-python-dispatch-audit --image "$IMAGE" --prompt "$PROMPT" \
  --thinking --max-new-tokens 32768 --temperature 0.05 \
  --top-p 1 --top-k 50 --seed 42 --audit-output "$OUT/single-audit.json"
```

批量接口使用 `--batch-jsonl`／`--batch-size`；诊断可用 `--python-dispatch-audit`。CLI 的 seed 同时用于初始化与生成，正式工具则保持初始化 seed=42 并逐题传入派生 seed。
正式接口为 `python -B evaluate_embspatial.py --config "$RUN_CONFIG" --dataset-path "$DATASET_JSONL" --output-path "$OUT/results.jsonl"`；首次不使用 resume 或 partial 选项。
独立复算接口为 `python -B score_results.py --results "$OUT/results.jsonl" --manifest "$RUN_MANIFEST" --output-prefix "$OUT/rescored"`，无需加载模型；变量均为该次实际清单路径。
每题落盘原始输出、真实 token／终止信息、解析／GT／正确性、时延、内存和错误；真实零生成 token、非有限 logits 或设备异常保留错误，不改参数重试。
持久 worker 串行处理；超时后 terminate／kill／join 并确认退出，不能让未退出的模型进程污染下一题；故障题计错后可为后续题创建新 worker。
恢复按 question_id 去重，校验实现、配置、数据及图像绑定；不一致则拒绝恢复并完整新跑，不人工修正原记录或拼接不同协议的题目。

# 可维可测分析

## 验证矩阵与通过依据

| 项目 | 计划与证据 |
| --- | --- |
| 功能／参考契约 | 本地权重加载、单图与两图 batch、direct／thinking；同输入对照官方 Pipeline 契约，注明参考实际后端 |
| 核心 NPU 执行 | 参数／buffer、module／KV、finite 检查与 native 无 Mode 阻断负例、真实 CPU＋NPU trace；不只看最终输出设备 |
| 固定 smoke | 10 题覆盖六类：mp3d_0／1／2／3／4／5／6／7／10／13；不按答案或成功率挑选 |
| profiling | 每个固定样本至少 5 次，共 50 次；开启 profile_memory、record_shapes，保存完整 native trace 和索引 |
| 异常／长输出 | 缺图、坏图、不可解析、空 decode、OOM／超时；长输出用独立受控压力记录真实长度与清理，强制长度参数另记，不算正式精度 |
| 状态与内存 | 连续 100 题；20 个固定真实样本连续两轮，检查上一轮图像／KV／输出残留和状态污染 |
| 正式精度 | 设计通过后冻结实现与协议，完成 3,640 条唯一结果，六类计数一致、正确数≥2,912，独立复算 |
| 工程／交付 | CPU 工具合同、pre-commit、目标仓 CI、目标实机及复现材料；均按实际结果记录，未完成不标通过 |

固定 smoke／故障注入只证明其实际覆盖范围，不替代完整评测；受控截断或强制长输出不改变正式解码参数，也不把“自然生成特定长度”设为任务新增门槛。

## 性能与内存口径

TTFT 从本题输入准备前计时，到首个实际生成 token 的 NPU 同步完成；包含 processor，不把 streamer 的 prompt 回调算首 token。
runner E2E 包含输入处理、生成、末尾同步、decode 和 token 列表整理；排除 worker 启动、资产校验、清理采样与 profiler 导出，这些另记完整外层墙钟。
输出吞吐为可获得完整计时的真实输出 token 总数／对应 runner E2E 总和；说明样本范围与计时条目数，不冒称纯 decode 吞吐，不剔除不可解析但正常生成的记录。
汇总 TTFT／E2E 平均值、P50、P95、总吞吐与全流程耗时；profile 仪器开销与无 profiler 正式测量分列，不只报告最快样本，不要求强制性能提升。
记录资源初始化后 loaded、预热后及各阶段的 current／peak allocated、reserved 和 Host RSS；基线是实际读数，不锁死历史绝对字节值。
拟以 Host 64 MiB 作为波动诊断界限，NPU cleanup current allocated 与 loaded 精确相等并记录正负差；这是待评审工程判据，不误称任务书自带阈值。
reserved／peak 不替代 current 清理证据，不逐题 empty_cache；最终须无持续增长、OOM 或跨样本状态残留，判据不在看到结果后放宽。
trace 在官方分析完成后无损压缩、回读校验并保存索引与备份；保留仓外原始证据，不将全量 trace 公开分卷上传设为正式评测前置条件。
