# 【社区任务】SplitQkvRmsnormMrope 算子设计文档（Atlas 800T A2）

> 算子：`SplitQkvRmsnormMrope`（aclnn 工程化；融合 QKV+Gate 拆分 → Q/K 逐 head RMSNorm → Q/K 三轴 MRoPE 旋转，适用于 Qwen3-VL 系列）
> 目标仓：https://gitcode.com/cann/ops-transformer ，落位 `experimental/posembedding/`（目录结构参照仓上算子：op_host / op_kernel / op_api / 测试用例）
> 目标硬件：Atlas 800T A2（910B3，arch22）；CANN 9.1.0+；开发语言 Ascend C
> 任务：9月社区任务-split_qkv_rmsnorm_mrope算子开发（tasklist 目录 `09-76-split_qkv_rmsnorm_mrope`）
> 对标基线：vLLM-Ascend Triton 实现 `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`（`split_qkv_rmsnorm_mrope_kernel`）
> 文档性质：**设计评审稿**（设计文档评审通过后进入开发，算子代码与 NPU 实测不在本 PR 范围内）

---

# 一、需求背景（required）

## 1.1 需求来源

来源于 CANN 社区任务《9月社区任务-split_qkv_rmsnorm_mrope算子开发》。vLLM-Ascend 中该算子当前使用 Triton 实现，任务要求使用 Ascend C 重新实现并完成 aclnn 算子工程化，利用 Vector Core 流水优化，整体性能达到开源 Triton 版本的 **2 倍以上**（算子 NPU 运行时延 < baseline / 2），验收通过后合入 ops-transformer 仓 `experimental/posembedding`。

## 1.2 背景介绍

### 1.2.1 标杆实现现状分析

随任务提供的 Triton 基线（Atlas 800T A2 实测，固定参数 `head_size=256, rope_dim=64, mrope_section=[11,11,10], eps=1e-6, interleaved=True`）：

| case | num_tokens | num_q_heads | num_kv_heads | Triton baseline（µs） | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| decode_t1_q16_kv4 | 1 | 16 | 4 | 636.240 | Qwen3-VL-4B/9B tp1 decode |
| small_t16_q16_kv4 | 16 | 16 | 4 | 643.246 | 小 batch |
| medium_t128_q16_kv4 | 128 | 16 | 4 | 678.923 | 中等 batch |
| prefill_t2048_q16_kv4 | 2048 | 16 | 4 | 641.700 | prefill |
| prefill_t8192_q16_kv4 | 8192 | 16 | 4 | 669.499 | 长序列 prefill |
| decode_t1_q16_kv2 | 1 | 16 | 2 | 640.793 | Qwen3-VL-35B tp2 decode |
| prefill_t2048_q16_kv2 | 2048 | 16 | 2 | 669.931 | Qwen3-VL-35B tp2 prefill |

**关键观察**：基线耗时在 num_tokens = 1 → 8192 范围内近乎平坦（636~679 µs），说明 Triton 路径由**固定调度开销主导**（小/中 case）并受**访存效率制约**（长序列 case）。Ascend C aclnn 单 kernel 直调可消除该固定开销，长序列 case 则需逼近有效带宽上限（见 §3.2.3 带宽分析）。

### 1.2.2 算子语义（golden 口径）

输入 `qkv [num_tokens, q_size + gate_size + 2*kv_size]`（bf16），行内布局为逐 head 交错的 `[Q_head0 | Gate_head0 | Q_head1 | Gate_head1 | … | K | V]`：

1. **Split**：Q/Gate 以 head 为粒度交错拆分（每段 head_size=256）；K、V 各连续 kv_size；
2. **RMSNorm（Q、K 逐 head，fp32 累积）**：`y = x · rsqrt(mean(x²) + eps) · weight (+ bias)`；
3. **MRoPE（Q、K 每 head 的前 rope_dim=64 维）**：32 个坐标对 `(x1, x2) = (x[c], x[c+32])`，`x1' = x1·cos − x2·sin`、`x2' = x2·cos + x1·sin`；`cos = cos_sin[axis(c), t, c]`、`sin = cos_sin[axis(c), t, c+32]`，`cos_sin [3, num_tokens, 64]`（每轴平面内 `[cos(32) | sin(32)]`）；
4. **输出**：q_output / k_output（归一化+旋转）、v_output / gate_output（**直通，逐位一致**），均 bf16。

轴映射（32 对坐标 → temporal/height/width 三轴）：

- `interleaved=True`：`axis(c) = c mod 3`（三轴轮转，与 mrope_section=[11,11,10] 的计数一致）；
- `interleaved=False`：`axis(c) = 0 (c<11) / 1 (11≤c<22) / 2 (c≥22)`（分段连续）。

两种模式的轴索引表均为 32 项**编译期常量**，kernel 内无需运行时分支。

## 1.2.3 平台前提（arch22）

无 SIMT/VF（arch35 专属），以标准 Ascend C 向量指令 + 多 AIV 分核（约 20 核，运行时读取）实现；UB 数百 KiB 量级，单 token 行（kv4：10240×bf16 = 20 KB；fp32 展开为 40 KB）可整体驻留 UB。

---

# 二、需求分析（required）

## 2.1 需求描述

aclnn 原型（对齐随任务 `op.json`）：

| 类别 | 名称 | 类型（bf16, ND） | 形状 | 备注 |
| --- | --- | --- | --- | --- |
| 输入 | qkv | bfloat16 | [num_tokens, q_size+gate_size+2·kv_size] | 融合 QKV+Gate 投影输出 |
| 输入 | q_weight / k_weight | bfloat16 | [head_size=256] | RMSNorm 权重 |
| 输入 | cos_sin | bfloat16 | [3, num_tokens, 64] | 三轴 cos/sin |
| 输入（可选） | q_bias / k_bias | bfloat16 | [256] | RMSNorm 偏置 |
| 输出 | q_output / gate_output | bfloat16 | [num_tokens, q_size] | q_size = num_q_heads×256 |
| 输出 | k_output / v_output | bfloat16 | [num_tokens, kv_size] | kv_size = num_kv_heads×256 |

属性：`num_q_heads=16`、`num_kv_heads=4`、`eps=1e-6`、`interleaved=true`、`has_gate=true`。

aclnn 两段式接口：`aclnnSplitQkvRmsnormMropeGetWorkspaceSize(...)` + `aclnnSplitQkvRmsnormMrope(...)`，workspace 由框架分配、接口异步执行于绑定 stream。

## 2.2 需求拆解

1. **精度**：Q/K 输出与 Triton 版本（golden.py，torch fp32 中转）逐元素对比 `max_diff < 1e-3`；V/Gate 直通**完全一致**（case.json 阈值 [0.001, 0] / [0, 0]）；
2. **性能**：7 个 case 均满足 NPU 时延 < baseline/2（msprof kernel 耗时口径）；
3. **工程化**：按 ops-transformer `experimental` 既有算子结构交付（op_host / op_kernel / op_api / 测试），含算子 README；
4. **语义覆盖**：interleaved / non-interleaved 两种 RoPE 模式、可选 bias、has_gate 开关（无 gate 时 gate_output 为空、qkv 宽度收缩）；
5. **测试**：基于随任务 `op.json/case.json/golden.py/generate_data.py` 构建 ascendOpTest 用例，7 case 全覆盖并支持形状扩展。

## 2.3 设计原则与非目标

**原则**：语义严格对齐 golden（fp32 内部计算、bf16 出入）；单 kernel 单趟完成 Split→Norm→RoPE→写出，消除中间 GM 往返；V/Gate 走纯搬运路径不进计算单元。

**非目标**：不修改 vLLM-Ascend 侧调用方式（另行集成）；不支持 head_size/rope_dim 泛化（Qwen3-VL 固定 256/64，泛化以 tiling 参数预留但不作为验收项）。

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

对 token t、Q/K 的每个 head h（head 内 256 维，前 64 维参与旋转）：

```
RMSNorm:  μ₂ = (1/256)·Σ_{i<256} xᵢ² ； yᵢ = xᵢ · rsqrt(μ₂ + eps) · wᵢ (+ bᵢ)
MRoPE:    对 c ∈ [0,32)：
          x1 = y[c],  x2 = y[c+32]
          out[c]    = x1·cos[t, a(c)]     − x2·sin[t, a(c)]
          out[c+32] = x2·cos[t, a(c)]     + x1·sin[t, a(c)]
          （cos/sin 取自 cos_sin[a(c), t, ·]，a(c) 为 §1.2.2 常量轴表）
          y[64:256] 直通
V/Gate:   out = in（bf16 逐位一致）
```

### 3.1.2 支持数据类型

| 项 | 类型 |
| --- | --- |
| 输入/输出 | bfloat16（ND） |
| 内部计算 | float32（RMSNorm 方差累积、旋转、加权均在 fp32 进行，输出 cast 回 bf16） |
| 属性 | Host 标量（int/float/bool） |

### 3.1.3 支持形状

固定 `head_size=256`、`rope_dim=64`、`mrope_section=[11,11,10]`；`num_tokens ∈ [1, 8192]`（用例覆盖 1/16/128/2048/8192）；num_q_heads=16；num_kv_heads ∈ {2, 4}（qkv 行宽 10240 / 9216）。

## 3.2 算子实现

### 3.2.0 工程落位与总体结构

```
experimental/posembedding/split_qkv_rmsnorm_mrope/
├── op_api/       # aclnn 两段式接口
├── op_host/      # tiling、shape/attr 校验、workspace 规划
├── op_kernel/    # Ascend C kernel（token 级并行，AIV）
└── tests/        # op.json / case.json / golden.py / generate_data.py
```

```mermaid
graph TD
    A[aclnnSplitQkvRmsnormMrope] --> B[Host: shape/attr 校验 + tiling 计算]
    B --> C[token 块均分到 AIV 核]
    C --> D[Kernel: 每 token 一行 qkv 搬入 UB bf16→fp32]
    D --> E[Split: Q/Gate 逐 head 交错拆分 K/V 连续段]
    E --> F[RMSNorm: 逐 head 平方规约→rsqrt→加权(+bias)]
    F --> G[MRoPE: 常量轴表 gather cos/sin 旋转前 64 维]
    G --> H[Cast fp32→bf16]
    H --> I[四路 CopyOut: q/k/v/gate_output]
    D --> J[V/Gate 纯搬运路径 直通写出]
```

### 3.2.1 Host 侧设计

1. 校验：qkv 行宽 = `num_q_heads·256·(has_gate?2:1) + 2·num_kv_heads·256`；cos_sin 形状 [3, num_tokens, 64]；权重/偏置长度 256；dtype 全 bf16；
2. tiling：token 维均分——`tokensPerCore = ceil(num_tokens / coreNum)`，大核/小核尾块均衡；写入 POD tiling（num_tokens、行宽、head 计数、eps、模式位 interleaved/has_gate/hasBias）；
3. workspace：仅 tiling 与轴表常量（若有），无大数据中转（单趟原地流水，§3.2.2）。

### 3.2.2 Kernel 侧设计（token 级并行 + 单趟流水）

1. **CopyIn**：每核按 token 块循环，整行 qkv（20 KB / 18.4 KB）`DataCopy` 入 UB，双缓冲与计算重叠；cos_sin 的三个轴平面按 token 行同步搬入；
2. **Cast & Split**：bf16→fp32 向量 Cast 后在 UB 内拆分——Q head h 位于 `[h·512, h·512+256)`、Gate 位于 `[h·512+256, (h+1)·512)`（纯 UB 偏移，无 GM 再访问）；K、V 为连续段；
3. **RMSNorm（逐 head）**：平方 → `WholeReduceSum`（256 lanes）→ 加 eps → `rsqrt`（标量广播）→ 乘权重（+bias）；Q 16 head + K 4 head 顺序处理，规约结果经 Vector-Scalar 流水回填；
4. **MRoPE**：每 token 先按常量轴表从三个轴平面 gather 出 32+32 的 cos/sin 向量（UB 内，token 内 16+4 个 head 复用），再对每个 Q/K head 的前 64 维执行旋转（两组向量乘加），其余 192 维直通；
5. **写出**：q/k 头 cast 回 bf16 后按去交错布局连续写出（q_output 行内 head 连续）；**V/Gate 走纯搬运路径**（bf16 原样 CopyOut，不经计算单元，保证逐位一致并省算力）；
6. **模式位编译期分离**：interleaved / has_gate / has_bias 以模板参数实例化，消除运行时分支。

### 3.2.3 访存与带宽分析（性能设计核心）

以最大 case `prefill_t8192_q16_kv4` 计：读 qkv 8192×10240×2B = 160 MB + cos_sin ≈ 3 MB，写四输出 8192×10240×2B = 160 MB，**总流量 ≈ 328 MB**；目标 < 669.499/2 ≈ 334.7 µs，要求有效带宽 ≈ 980 GB/s——**逼近乃至超出 A2 平台常规有效带宽**，是本设计最大风险点。对策：

- 满核 token 并行（20 AIV 全部参与，无空闲核）；
- 单趟流水（无中间 GM 落盘、无 kernel 间往返）；行级 20 KB 连续突发搬运；
- 计算全部在 UB 内 fp32 向量完成，与 MTE2/MTE3 双缓冲充分重叠；
- 小/中 case（≤2048 tokens）由调度开销优势自然达标（基线固定开销 ~640 µs vs aclnn 直调数十 µs 量级）；
- 开发阶段先以纯搬运 micro-benchmark 标定平台有效带宽上限，再确认 t8192 达标裕量；若存在缺口，优化方向为 CopyOut 合并与 32B 对齐突发写（以实测迭代）。

### 3.2.4 精度处理要点

- 方差累积 fp32（对齐 golden 的 `float()` 中转），eps 以 fp32 参与运算；
- bf16↔fp32 Cast 采用向量指令舍入模式与 torch bf16 舍入一致（RNE）；
- V/Gate 不做任何数值变换（逐位一致，阈值 0）；
- 旋转的乘加顺序与 golden 表达式一致（先乘后加减，无融合重排），保证 `max_diff < 1e-3` 裕量。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B3，arch22） | √ |

## 3.4 算子约束限制

1. head_size=256、rope_dim=64、mrope_section=[11,11,10] 固定（Qwen3-VL）；interleaved 默认 true；
2. has_gate=false 时 gate_output 为空张量、qkv 行宽收缩为 `q_size + 2·kv_size`；
3. q_bias/k_bias 为可选输入，缺省不做偏置加法；
4. 输入输出均为 bf16/ND；内部 fp32；不支持其他 dtype 与排布；
5. num_tokens=0 视为非法输入（aclnn 框架校验层面报错），不在 kernel 内处理。

---

# 四、可维可测分析

## 4.1 精度标准

| 输出 | 判据 | 来源 |
| --- | --- | --- |
| q_output / k_output | 与 golden（`golden.py:calc_expect_func`，torch fp32 中转）逐元素 `max_diff < 1e-3` | case.json `err_threshold [0.001, 0]` |
| v_output / gate_output | 逐元素完全一致（阈值 0） | case.json `err_threshold [0, 0]` |

测试数据由随任务 `generate_data.py` 生成（qkv/cos_sin 值域 [-1,1]，权重 1.0）；7 case 全覆盖，并补充 non-interleaved、含 bias、has_gate=false 的扩展用例（golden 同口径生成）。

## 4.2 性能标准

判据：各 case NPU 运行时延 < Triton baseline / 2（基线见 §1.2.1 表；目标值分别为 318.1 / 321.6 / 339.5 / 320.9 / 334.7 / 320.4 / 335.0 µs）；msprof 采集 kernel 耗时，先 warmup 再有效采样取平均。

## 4.3 兼容性分析

- 新增算子目录独立落位 `experimental/posembedding`，不修改仓内既有算子与公共接口；
- aclnn 两段式接口与框架标准一致，可被 vLLM-Ascend 后续以 aclnn 调用替换 Triton 路径（集成不在本任务范围）；
- Triton 原路径保留，两条实现可并行对照验证。

## 4.4 实现与验证状态

| 项 | 状态 |
| --- | --- |
| 设计文档（本文档） | 本 PR 提交评审 |
| 算子实现（op_api/op_host/op_kernel） | 待开发（评审通过后启动） |
| 测试工程（tests/，ascendOpTest 用例） | 待开发，基于随任务 op.json/case.json/golden.py |
| 精度/性能自验证 | 待实现后按 §4.1/§4.2 执行并出具报告 |

# 五、参考资料

1. 任务书《split_qkv_rmsnorm_mrope 任务书》及随任务 `op.json / case.json / golden.py / generate_data.py`；
2. vLLM-Ascend Triton 基线：`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`；
3. ops-transformer 开源仓（experimental 算子结构与贡献指南）：https://gitcode.com/cann/ops-transformer ；
4. Ascend C 算子开发文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html ；
5. Ascend C 算子开发接口文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html 。
