> 文档状态：设计评审稿。
>
> 本 PR 仅提交设计文档；实现代码与 A2/A3 验收材料在代码仓 cann-recipes-embodied-ai 的 PR #210 中提交（CI 已通过、CLA 已签）。

# Alpamayo-R1 NPU 图模式与权重预取设计文档

## 1. 背景

Alpamayo-R1 样例基于上游 NVlabs/alpamayo 固定提交 `e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073`，通过 `patches/ar-1.patch` 完成昇腾基础适配。样例 README 记录的历史基线（CANN 8.5.0 环境）：单次生成 10 组驾驶轨迹与 CoC 文本约 7.32 s，Physical-AI-AV `chunk_0000` 固定 100 clips 的 minADE(10)=0.5676。

对 eager 基线的 torch_npu profiler 全量采集显示负载为 launch-bound：单次推理约 75077 次 kernel 下发、host 侧下发耗时约 471 ms，MatMul 家族仅占约 14% 的 kernel 时间，元素级小算子约占 60%。据此确定两个优化方向：

1. 将稳定计算子图（flow-matching 去噪的 expert 前向、VLM 自回归 decode 单步）接入 NPU Graph capture/replay，消除 host 侧逐算子下发；
2. 对 profiling 证明为 memory-bound 且存在安全依赖窗口的 MatMul 权重（`action_in_proj` / `action_out_proj`，GEMM 的 M 维为扩散 token 数，算力利用率低、权重读取主导单次时延）使能 side-stream 预取。

## 2. 目标与非目标

### 2.1 目标

1. A2 / A3 双平台上 expert 前向与 VLM decode 单步真实 capture 并 replay，warmup → capture → replay 生命周期可观测（captures / replays / fallbacks / evictions 计数）且可安全回退；
2. 提供独立 `enable_graph` 与 `enable_prefetch` 配置，全部默认关闭；关闭时不安装任何 wrapper，完全恢复 `ar-1.patch` 原路径；
3. graph+prefetch 稳态端到端耗时低于各自平台 eager 基线，且改善幅度超过基线波动；
4. 四种配置（eager / graph-only / prefetch-only / graph+prefetch）的输出与 eager 逐元素一致，不修改默认输入、采样步数、轨迹数量或采样参数；
5. 固定 100 clips 的 minADE 在 1/3/5/10 轨迹入口均不劣于同平台 eager 基线。

### 2.2 非目标

- 不更换模型权重、精度类型、轨迹数、扩散步数或数据集来获得性能收益；
- 不新增量化、模型并行或自定义算子；
- 不以单个 kernel 的理论收益替代端到端实测。

## 3. 版本与资产

| 项目 | 值 |
|---|---|
| Recipes 基线 | `cann/cann-recipes-embodied-ai` master |
| 上游提交 | `e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073` |
| 实现分支 | `feat/alpamayo-r1-graph-prefetch`（PR #210，CI 通过） |
| 权重 | `nvidia/Alpamayo-R1-10B`（bf16，SHA256 已随验收材料存档） |
| A2 环境 | Atlas 800I A2（910B）/ CANN 9.0.0 / torch 2.8.0 + torch-npu 2.8.0 / transformers 4.57.1 / python 3.11（hidevlab 镜像 py3.11-a2-arm-20261012） |
| A3 环境 | Atlas 800I A3（910C）/ CANN 9.1.0 / torch 2.7.1 + torch-npu 2.7.1.post4 / transformers 4.57.1 / python 3.11 |
| 评测数据 | Physical-AI-AV `chunk_0000` 固定 100 clips，seed 42 |

## 4. 图模式设计

### 4.1 图边界

| 子图 | 边界 | 说明 |
|---|---|---|
| expert 前向 | `step_fn` 内 10 次去噪调用 | 输入形状固定；prompt KV 池化为只读层进图（容量 512 对齐分桶），step KV 写池尾零拷贝 view，FIAS bool mask 屏蔽池尾未写区；签名只取决于桶容量，跨 clip 单图复用 |
| VLM decode 单步 | generate 循环内每步 forward | StaticCache 池（地址跨 rollout 稳定）+ FIAS causal step mask + 静态 position_ids；prefill 恒 eager |
| 图外 | 初始噪声、top-p 采样、Euler 步进、crop、轨迹后处理 | RNG 全部图外，图内无随机数消费 |

### 4.2 关键设计决策

- **统一 kernel**：非因果 attention 分支统一路由到 `npu_fused_infer_attention_score`（NPU Graph 白名单算子，与 `npu_fusion_attention` 对相同输入位一致），四种消融模式共享同一 kernel 与 KV 布局——这是档间输出逐位一致的前提；
- **统一池化布局**：expert 池化 KV 随 kernel 统一层安装在全部四种模式（eager 也走池，只是不 capture），softmax 归约顺序在任何模式下相同。若 eager 走紧贴 cat、graph 走池尾 step，归约顺序不同会引入约 1 ulp 差异并被 10 步去噪混沌放大至约 4e-3 的轨迹偏差（实测定位过）；
- **桶化签名 + LRU**：prompt 长度按 512 对齐分桶（实测 3024-3047 的长度全部落入 3072 单桶，跨 100 clips 零重捕获），桶数超限驱逐最旧图，prompt 超池容量或训练态一律回退 eager 并计数；
- **torch 2.8 兼容**：replay 路径（静态 buffer 拷贝与 replay 调用）包 `torch.inference_mode()`——torch 2.8 严格执行 inference-tensor 原地更新规则，2.7 上该包裹为无操作。

### 4.3 实现攻关要点（实机验证过）

| 现象 | 根因 | 解法 |
|---|---|---|
| 图捕获 ACL 107025 | `npu_fusion_attention` 内部 stream 不 join | 非因果分支统一改 FIAS |
| FIAS + mask 捕获仍 107025 | BSH 布局 + mask 组合内部开流 | mask 路径切 BNSD |
| BNSD 报 EZ0037 | BNSD 要求 key/value 连续 | transpose 后显式 `.contiguous()` |
| eager 与 graph 输出差 4e-3 | step KV 位置不同改变 softmax 归约顺序 | 四档统一池化布局 |
| 逐 clip 重捕获 92 次 | prompt 长度逐 clip 微变进签名 | 桶化签名 + LRU 驱逐 |

## 5. 预取设计

- profiling 证据：目标 GEMM 的 M 维为扩散 token 数（约 40）远小于 K/N，算力利用率低、权重读取主导单次时延，属带宽主导型负载；
- 独立 NPU stream + event：expert 前向入口（固定调用位置，与图模式组合时无额外重编译）对目标权重发起 reduce 模式预取，目标模块 forward pre-hook 中 `wait()`，CPU 侧零同步；
- `enable_prefetch=False` 时不创建 stream、不插入任何调用。

## 6. 验证协议

- **minADE 口径**：seed 在每个评测 arm 开头设置一次，RNG 状态在 100 个 clip 间连续推进（不逐 clip 重置）；clip 集合取数据集接口 `get_clip_chunk` 映射后的第一个 chunk 全集（每 chunk 至多 100 clips）；minADE 为 100 clips 等权平均，单采样时间点 `t0_us=5100000`，单轮运行，无过滤；1/3/5/10 轨迹入口为同一次 10 轨迹生成中前 k 条取 min；
- **四档消融**：每档 warmup ≥1 + 正式 ≥4，记录首次迭代耗时（含 FIAS 安装、池化分配与 capture）、稳态逐次耗时、目标模块 host 耗时、kernel 变化与显存峰值；
- **输出一致性**：同进程 PARITY 链（四模式各 ≥2 runs 逐元素比对 + 落盘 json）+ 数据集级 `--save-ref` / `--compare-ref` 逐 clip FAIL-loud 比对链；
- **稳定性**：gp 档连续 ≥20 次推理，profiler 开启 `profile_memory`，验证零失败、显存曲线平坦、输出零漂移、开关组合语义正确。

## 7. 双平台实测结论摘要（设计对应的实现已验证）

- 四种配置输出与 eager **逐位一致**：minADE(10) 在 100 clips 上四模式全同（A2 与 A3 各自平台内 13 位小数全等，1/3/5/10 档全同）；
- 性能：A2 graph+prefetch 稳态 **-22.2%**（gp 7.09-7.14 s vs eager 9.07-9.18 s，分布无重叠，基线波动 1.1%；数据集负载口径 -22.4%）；A3 长跑口径 -0.65%（24 次 eager 基线，分布无重叠）；expert host 耗时 A2 6.61 s → 0.86 s（-87%）；
- 稳定性：A2 24 runs / A3 23 runs 零失败，显存峰值曲线完全平坦（44.47 GB 恒定）；
- 详细数据表随 PR #210 验收材料提交。

## 8. 交付计划

| 交付件 | 位置 |
|---|---|
| 实现（npu_opt 六文件 + patch + 测试三件） | PR #210（CI 通过、CLA 已签） |
| 样例 README 与优化文档更新 | PR #210 内 |
| RFC / 口径讨论 | Issue #141 |
| 双平台原始数据与验收报告 | 平台验收材料（不入库，RL 红线） |

