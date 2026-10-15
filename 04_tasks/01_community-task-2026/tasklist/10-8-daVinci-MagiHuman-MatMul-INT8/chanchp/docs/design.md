# daVinci-MagiHuman MatMul INT8 量化设计文档（Atlas A3）

| 字段 | 内容 |
|---|---|
| **任务** | HiAscend 社区任务「daVinci-MagiHuman MatMul INT8 量化（Atlas A3）」 |
| **作者 / 交付账号** | GitCode `chanchp` |
| **实现目标仓** | [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) · `models/davinci-magihuman/` |
| **关联 RFC Issue** | https://gitcode.com/cann/cann-recipes-infer/issues/459 |
| **上游 MagiHuman commit** | `209209b7086eba2020c5439265221495a8357322` |
| **状态** | 设计已锁定；Atlas A3 自验完成；实现代码暂存私仓，按任务要求择机正式交付 |

---

## 1. 背景与目标

### 1.1 背景

daVinci-MagiHuman DiT 主干热点线性层由 `BaseLinear` / `NativeMoELinear` 承载，覆盖每个 block 的：

- `attention.linear_qkv` / `attention.linear_proj`
- `mlp.up_gate_proj` / `mlp.down_proj`

当前默认走 BF16 `torch.matmul`。本任务在 Atlas A3 上将上述 MatMul 切换为真实 W8A8 INT8，并保证可配置、可验证、可回退。

### 1.2 目标

1. **真实性**：`npu_dynamic_quant` + `npu_quant_matmul` 执行 INT8 MatMul；Profiler 可观测 QuantMatMul 类算子。
2. **精度**：中间张量无 NaN/Inf；≥5 组固定 prompt 盲测相对 BF16 无可稳定辨识退化。
3. **性能**：同配置下端到端关键阶段中位耗时低于 BF16，且降幅超过 BF16 三次极差。
4. **内存**：INT8 权重存储低于 BF16；记录加载/峰值显存。
5. **可复现**：离线转换、配置启停、自验流程可重复。

### 1.3 锁定方案

| 维度 | 方案 |
|---|---|
| 模式 | W8A8 |
| 权重 | 离线对称 per-output-channel INT8，scale float32 |
| 激活 | 在线 dynamic per-token INT8 |
| 运行时 | `npu_dynamic_quant(..., quant_mode="pertoken")` → `npu_quant_matmul(..., pertoken_scale=, bias=BF16, output_dtype=bfloat16)` |
| 权重布局 | 推理前转置为 `[K, N]`，并采用 NPU 友好内部格式驻留（与 `npu_quant_matmul` 契约一致） |
| 代码归属 | `models/davinci-magihuman/inference/model/quantization_layers/`（不强制改 `module/quantization/`） |
| 回退 | `ignored_layers` + BF16；缺失/错误 scale **必须报错**，禁止占位 scale |
| 反伪量化 | 禁止 BF16→INT8→立即反量化→BF16 MatMul 冒充 INT8 |

---

## 2. 范围

### 2.1 In Scope

- 替换目标 Linear / MoE Linear 的 MatMul 路径；保持 MoE dispatch/undispatch 语义。
- `config/int8.yaml`、`int8_config.json`、`bash infer.sh int8`。
- 离线权重转换、数值检查、profiling、盲测与异常路径自验材料。

### 2.2 Out of Scope

- RoPE / RMSNorm / Softmax-FA / 残差加法等非 MatMul。
- 独立 aclnn/AscendC kernel。
- 提交量化权重大文件（只交方法、元数据与校验信息）。

---

## 3. 基线冻结

### 3.1 环境（实机）

| 项 | 值 |
|---|---|
| 硬件 | Atlas A3 单卡 |
| Python | 3.12 |
| CANN | 9.1.0 |
| PyTorch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| MagiCompiler | 1.0.0 |

### 3.2 推理配置

| 参数 | 值 |
|---|---|
| 模式 | base T2V |
| 分辨率 / 时长 | 448×256，5 s |
| steps / CFG / seed | 32 / 2 / **42**（seed 写在 engine_config） |
| Cache | `dit_cache.method: NoCache` |
| Prompt | `example/assets/prompt.txt`（另备 ≥4 组盲测固定文案） |

INT8 与 BF16 除量化字段、量化权重路径、输出目录外完全一致。

### 3.3 性能测量

- 预热 ≥1 次；正式 ≥3 次；计时前同步 NPU。
- 主指标：Step3 Basic Resolution Evaluation（denoise）中位耗时。
- BF16 波动上界 = max−min；INT8 降幅须超过该上界。
- Profiler：`ENABLE_PROFILER=1`，目录 `prof/<mode>_prof/`。

---

## 4. 架构与配置

### 4.1 目录（计划）

```
models/davinci-magihuman/
├── inference/model/quantization_layers/
│   ├── int8.py              # Int8BaseLinear / Int8NativeMoELinear
│   ├── apply.py             # 替换与加载
│   └── weight_convert.py    # 离线 per-channel 量化
├── config/int8.yaml
├── config/int8_config.json  # quantization_config 块
├── tools/convert_weights.py
└── infer.sh                 # 新增 int8 mode
```

### 4.2 quantization_config（示意）

```json
{
  "enabled": true,
  "mode": "w8a8",
  "activation_method": "dynamic",
  "weight_granularity": "per_channel",
  "activation_granularity": "per_token",
  "quantized_weight_path": "/path/to/daVinci-MagiHuman-int8",
  "ignored_layers": [
    "block.layers.0",
    "block.layers.39.attention.linear_proj"
  ],
  "fallback_dtype": "bfloat16"
}
```

敏感层默认 BF16 回退；可用配置调整。

### 4.3 权重制品契约

- `quant_state.pth`：`{layer}.weight_int8`、`{layer}.weight_scale`（及可选 bias）
- `quant_metadata.json`：`scheme=w8a8`、层清单、shape/dtype
- 缺键 / shape 不匹配 → RuntimeError

### 4.4 接入点

模型 checkpoint 加载后、`.to(device)` 前/后按配置调用 `apply_int8_quantization`；支持环境变量指向独立 `quantization_config.json`，默认关闭不影响 base。

---

## 5. 自验计划与实机结果摘要

### 5.1 功能 / 异常

| 用例 | 期望 | 实机 |
|---|---|---|
| enabled=false | 行为同 BF16 | PASS |
| 缺失 quant_state | 报错 | PASS |
| 缺失某层 weight/scale | 报错，拒占位 | PASS |
| 全部 ignore | 明确报错（无替换） | PASS |

### 5.2 性能（Step3 denoise，seed=42）

| 配置 | runs (s) | 中位 | 极差 |
|---|---|---|---|
| BF16 | 35.50 / 35.55 / 35.40 | **35.50** | 0.15 |
| INT8 | 27.78 / 27.75 / 27.74 | **27.75** | 0.04 |

相对 BF16：**约 −21.8%**，降幅 ≫ BF16 极差。

### 5.3 Profiler / 显存

- Profiler 可见 `QuantMatmulWeightNz`、`DynamicQuantV2`（证明真实 INT8 路径）。
- DiT 加载后 allocated：BF16 ≈28.5 GB → INT8 ≈20.4 GB。

### 5.4 盲测

固定 5 组 prompt（默认样例 + 单人/双人/走动/静物），BF16/INT8 成对产物已生成；主观结论由指定验收人填写。

---

## 6. 交付与风险

### 6.1 交付件

1. 本设计文档（本 PR，competition 仓）。
2. recipes-infer 侧 RFC Issue #459。
3. 实现代码与配置（按任务要求提交至目标仓 / 私仓邀请 Ascend-CANN 查阅）。
4. 私仓 `task_submission/` 自验材料（步骤说明、性能/精度/内存日志、盲测清单）。

### 6.2 风险

| 风险 | 缓解 |
|---|---|
| NZ/内部格式未生效导致 INT8 更慢 | 加载期校验 format；Profiler 复核算子名 |
| 首尾层精度敏感 | ignored_layers 回退 |
| 伪量化误报性能 | 强制 quant_matmul 路径 + Profiler 证据 |

### 6.3 后续

- 补齐 README/docs 启用说明。
- 正式代码 PR 关联 Issue #459（与实现窗口对齐，避免过早公开实现细节）。
- 验收人口测结论与最终合入。
