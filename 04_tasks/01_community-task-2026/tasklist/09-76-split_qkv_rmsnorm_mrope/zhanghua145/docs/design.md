# SplitQkvRmsnormMrope 算子设计文档（Atlas 800T A2）

> 依据《split_qkv_rmsnorm_mrope 任务书》编写，按官方设计模板组织，四个必需一级章节齐全。
> 交付目标仓：https://gitcode.com/cann/ops-transformer （`experimental/posembedding/split_qkv_rmsnorm_mrope/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 9 月社区任务「split_qkv_rmsnorm_mrope 算子开发」。该算子在 vllm-ascend 中当前为 Triton 实现（`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`，kernel 名 `split_qkv_rmsnorm_mrope_kernel`），服务于 Qwen3-VL 系列模型推理。目标：使用 Ascend C 重写（Vector 核流水优化），在 Atlas 800T A2 上达到 Triton 基准 **2 倍以上**性能。

| 材料 | 用途 |
| --- | --- |
| 任务书 `split_qkv_rmsnorm_mrope算子开发.md` | 功能、接口、精度/性能门槛与交付要求 |
| `op.json` | aclnn 工程算子原型（输入输出/attr 定义） |
| `case.json`（7 条测试 shape）+ `golden.py` + `generate_data.py` | golden 语义与测试数据 |
| ops-transformer 仓 `experimental/posembedding/` | 目标目录与工程模式参照 |

## 1.2 背景介绍

### 算子语义：一次读入、四路输出的融合归一化旋转

算子将线性层输出的融合 QKV+Gate 张量（`[num_tokens, q_size + gate_size + 2·kv_size]`）单遍处理：

```text
布局: [Q_head0 | Gate_head0 | Q_head1 | Gate_head1 | ... | K | V]
      （Q 与 Gate 按 head 粒度交错，各 q_size；K/V 各 kv_size，连续尾段）

1. Split:        拆出 Q/G/K/V（golden.py 实测语义：qg 每 512 元素中前 256 为 Q_head、后 256 为 Gate_head）
2. RMSNorm(Q/K): 逐 (token, head) 在 fp32 下 x·rsqrt(mean(x²)+eps)·w（bias 可选）
3. MRoPE(Q/K):   对每 head 前 rope_dim=64 维做三轴旋转，配对 (i, i+32)，i∈[0,32)
                 cos = cos_sin[axis(i), t, i]，sin = cos_sin[axis(i), t, i+32]
                 out_first = first·cos − second·sin；out_second = second·cos + first·sin
                 axis(i)：interleaved 按 i%3 循环指派（0/1/2 共 11/11/10 对）；
                          non-interleaved 按 [0,11)/[11,22)/[22,32) 连续分段
4. 输出: q_output / k_output（bf16），v_output / gate_output（直通，bit 级一致）
```

`cos_sin` 布局为 `[3, num_tokens, 64]`：轴 0/1/2 = temporal/height/width，每行前 32 为 cos、后 32 为 sin（golden.py 索引方式实测确认）。

### 现状分析（ops-transformer 与基线）

基于 ops-transformer master（`experimental/posembedding/`）实测核对：

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `experimental/posembedding/` | 已有三个算子：`rope_matrix`、`rotary_position_embedding3d`、`rotary_stride` | **工程模式参照**：`npu_ops_transformer_ext` torch 扩展 + Ascend C kernel 直调（`torch_interface.cpp` 注册、`op_host/<op>_tiling.h`、`op_kernel/<op>.h`、`tests/test_<op>.py`） |
| 仓内算子能力 | 仅 RoPE/位置编码类；**无 Split+RMSNorm+MRoPE 融合算子** | 本任务为新增（缺口即全部工作量） |
| 基线 | vllm-ascend Triton 单融合 kernel | 对标对象；7 条 case 基线 636.2–678.9 µs（任务书 §3.5） |

### 功能缺口与机会

1. **融合算子缺失**：当前链路在 vllm-ascend 由 Triton 承载，Triton-NPU 后端向量化不足、发射开销高；
2. **单遍数据流红利**：算子本质是"读一遍 qkv、写一遍四输出"的带宽受限型（无重用、计算密度低），Ascend C 单融合 kernel 可将访存压至理论下限（§3.2.5 反算），这是 2× 目标的来源；
3. V/Gate 为纯直通段，须与计算段同 tile 流水搬出，避免二次读 qkv。

# 2. 需求分析（required）

## 2.1 需求描述

在 `experimental/posembedding/split_qkv_rmsnorm_mrope/` 以 Ascend C 实现单融合 Vector kernel（AIV-only）及 torch 扩展接口。能力达到：bfloat16 输入输出、内部 fp32；`head_size=256、rope_dim=64、mrope_section=[11,11,10]、eps=1e-6`（attr 可配）；支持 `interleaved` 真/假两种旋转模式与 `has_gate` 真/假；q_bias/k_bias 可选。精度：Q/K 与 golden 逐元素 `max_diff < 1e-3`，V/Gate 完全一致。性能：7 条 case 的 NPU 运行时延 < Triton 基线 / 2。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | Split 语义（Q/Gate head 级交错、K/V 连续尾段） | 任务书 §2.3 / golden.py | §3.2.2 索引映射（Q_h0 @ h·512，G_h0 @ h·512+256） |
| R02 | RMSNorm 逐 head fp32（eps 入 rsqrt、weight、可选 bias） | 任务书 §2.3 | 向量 ReduceSum + Rsqrt + Mul（+Add） |
| R03 | MRoPE 三轴（interleaved / non-interleaved） | 任务书 §2.3 | 32 项轴索引 LUT 双表（§3.2.2），cos/sin 交叉 gather |
| R04 | V/Gate 直通 bit 级一致 | 任务书 §3.2 | 同 tile bf16 原样搬出，不经计算通路 |
| R05 | 7 条 shape（t∈{1,16,128,2048,8192}，q16_kv4 / q16_kv2） | case.json | token×head 任务网格分核（§3.2.3） |
| R06 | 精度：Q/K < 1e-3；V/Gate = 0 | 任务书 §3.2 / case.json err_threshold | fp32 中间、单次舍入回 bf16 |
| R07 | 性能：NPU 时延 < baseline/2 | 任务书 §3.3 | §3.2.5 访存反算 + 口径说明 |
| R08 | 工程合规（ops-transformer 结构） | 任务书 §5 | §3.2.0 文件布局（对齐 rope_matrix） |
| R09 | 可测可复现 | 任务书 §3.5 | case.json + golden.py + generate_data.py + tests/ |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

设 head 内向量 `x ∈ ℝ²⁵⁶`（fp32 展开），对 Q/K 的每个 (t, h)：

```text
RMSNorm:  y = x · rsqrt( (1/256)·Σ_{d<256} x_d² + eps ) · w_d (+ b_d)
MRoPE（仅作用于 y[0:64]，配对 (i, i+32)）:
          y_i'     = y_i · cos_{a(i),t,i}     − y_{i+32} · sin_{a(i),t,i}
          y_{i+32}' = y_{i+32} · cos_{a(i),t,i} + y_i · sin_{a(i),t,i}
          a(i) = interleaved ? (i%3==1 ∧ i≤33 ? 1 : i%3==2 ∧ i≤30 ? 2 : 0)
                             : (i<11 ? 0 : i<22 ? 1 : 2)
输出:     q_output[t, h·256+d] = bf16(y'_d)；其余 d ∈ [64,256) 直接输出 bf16(y_d)
          gate/v_output: bf16 原样直通（无计算）
```

### 3.1.2 支持数据类型

bfloat16 输入输出；内部计算 fp32（cast 进、单次舍入出）。cos_sin/权重/偏置同为 bfloat16 输入、使用前 fp32 展开。

### 3.1.3 支持形状

`qkv [T, W]`，`W = q_size + gate_size + 2·kv_size`（q_size = num_q_heads·256，kv_size = num_kv_heads·256，gate_size = has_gate ? q_size : 0）；`cos_sin [3, T, 64]`；输出 `q [T, q_size]`、`k/v [T, kv_size]`、`gate [T, q_size]`。attr：`num_q_heads / num_kv_heads / eps / interleaved / has_gate`（op.json）。典型 W = 10240（q16_kv4）或 9216（q16_kv2），行字节 20480/18432，均 32B 对齐；head 段 512B 对齐。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内 rope_matrix 先例）

```text
experimental/posembedding/split_qkv_rmsnorm_mrope/
├── CMakeLists.txt
├── README.md                                  # 支持 limitation + 编译测试步骤
├── torch_interface.cpp                        # torch 扩展注册 + kernel 直调（npu_ops_transformer_ext 模式）
├── inc/split_qkv_rmsnorm_mrope_extern.h
├── op_host/split_qkv_rmsnorm_mrope_tiling.h   # TilingData + Host 划分
├── op_kernel/
│   ├── split_qkv_rmsnorm_mrope.h
│   └── split_qkv_rmsnorm_mrope.cpp            # 融合 kernel（AIV）
└── tests/
    ├── test_split_qkv_rmsnorm_mrope.py        # 精度（golden.py 对比）+ 性能（Triton 基线）
    ├── case.json / golden.py / generate_data.py  # 任务包随附
    └── data/                                  # generate_data.py 产物
```

### 3.2.1 Host 侧设计

校验：shape/attr 一致性（`W == 2·q_size + 2·kv_size`（has_gate）或 `q_size + 2·kv_size`；cos_sin 末维 = 64 = rope_dim；head_size 固定 256 与 rope_dim = 64 以断言形式校验，超集形状后续扩展）。tiling：

```text
任务网格: (tokenTile, head)，head ∈ [0, num_q_heads + num_kv_heads)
tokenTile 按 UB 预算取值（每 token 每头 512B 装载，tile 取 8~32 token）
分核: 总 tile 数 ÷ AIV 核数均分，余量前几核 +1（满核优先）
attr 下发: num_q_heads/num_kv_heads/eps/interleaved/has_gate/has_bias +
           轴 LUT（32 项，按 interleaved 二选一，Host 预生成入 tiling）
```

### 3.2.2 Kernel 侧设计（本任务技术主体）

**总体：单融合 AIV kernel，"tokenTile × head" 任务粒度，一遍读 qkv、一遍写四输出。**

```text
对认领的每个 (tokenTile, h) 任务：
  ① Q 头（h < num_q_heads）：
     装载 qkv[t0:t0+tt, h·512 : h·512+512]（Q 与 Gate 同段连续读入 UB）
     ├─ Q 半段 [0:256)：bf16→fp32 → ReduceSum(x²) → r = rsqrt(mean+eps)
     │   → x̂ = x·r·w(+b) → 前 64 维按 LUT 取 axis，gather cos/sin
     │   （cos_sin[axis, t, i] / [axis, t, i+32]，fp32 展开）
     │   → 旋转配对 (i, i+32) → bf16 → 写 q_output
     └─ Gate 半段 [256:512)：bf16 原样 → 写 gate_output（直通）
  ② K 头（h ≥ num_q_heads，记 kh = h − num_q_heads）：
     装载 qkv[t, keyBase + kh·256 : +256]（连续）
     同 ① 归一化+旋转 → 写 k_output
  ③ V 段：与 ①② 无计算依赖，由独立 (tokenTile, vChunk) 任务直通搬运
     qkv[t, valBase : valBase+kv_size] → v_output（bf16 原样）
```

要点：

- **归一化规约**：head_size = 256 元素 fp32 规约在 UB 内完成（向量 ReduceSum，256 元素一次成对折叠加尾处理），无 GM 中间；
- **cos/sin 交叉 gather**：`cos_sin` 按 (axis, t) 行预装 UB（每 token 64 元素 × 用到的轴行），轴由 LUT 索引——32 对配对仅需 32 cos + 32 sin，按 LUT 从三轴行中选择；interleaved 的轴指派呈 `0,1,2,0,1,2,...`（边界截断）模式，LUT 化后无逐元素分支；
- **旋转实现**：first/second 两半向量对位 Mul/Mul/Sub/Add（各 32 元素，fp32），输出交错写回 [0:64) 区间；
- **直通段零计算**：V/Gate 与计算段共用 tile 装载（Q/Gate 本就同段），V 独立任务，全 bf16 DataCopy，不经向量单元，保证 bit 级一致；
- **尾块**：tokenTile 对 T 取余的尾核按实际 token 数收缩装载（32B 对齐：行 20480/18432B、head 段 512B 均天然对齐，无逐元素尾掩码）；
- **任务类型**：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，无 workspace、无 GM 中间张量。

### 3.2.3 调用与执行流程图

```text
torch (vllm-ascend 替换点)                     Device (AIV × cores)
────                                           ────────────────────
split_qkv_rmsnorm_mrope(qkv, w_q, w_k, cos_sin, ...)
  ├─ shape/attr 校验 + TilingData（网格/LUT/分核）
  └─ kernel 直调（torch_npu 当前 stream）──→  融合 kernel
                                                ├─ (tokenTile, qHead): Q norm+mrope → q_out
                                                │                        Gate 直通 → gate_out
                                                ├─ (tokenTile, kHead): K norm+mrope → k_out
                                                └─ (tokenTile, vChunk): V 直通 → v_out
```

### 3.2.4 关键技术点与风险

| # | 点 | 处置 |
| --- | --- | --- |
| 1 | 轴 LUT 两模式语义 | 以 golden.py `_axis_indices` 为唯一语义源，Host 生成 32 项 LUT；单测对两模式各断言轴分布（11/11/10） |
| 2 | cos_sin 布局（后半 sin） | 索引 `i` / `i+32` 与 `[3,T,64]` 布局写死并加静态断言；7 条 case 的 golden 全量回归兜底 |
| 3 | bf16 直通一致性 | V/Gate 不进计算通路、同 tile 搬运；case.json err_threshold [0,0] 逐 case 锁定 |
| 4 | decode（T=1）并行度仅 20 个 head 任务 | 数据量 20KB，耗时由发射开销主导（目标 <318µs 余量大）；单 kernel 单发射已是下界 |
| 5 | 大规模档带宽下限 | 见 §3.2.5 反算与口径说明——首要联调验证项 |

### 3.2.5 性能设计反算（R07）

融合单遍的不可避免访存 = 读 qkv 一遍 + 写四输出一遍 ≈ `2 × W × T × 2B`：

| case | T | 融合流量 | 基线 µs | 目标 µs | 目标隐含带宽 |
| --- | --- | --- | --- | --- | --- |
| decode_t1_q16_kv4 | 1 | 41 KB | 636.2 | <318.1 | —（发射开销主导） |
| small_t16 | 16 | 0.66 MB | 643.2 | <321.6 | 2 GB/s |
| medium_t128 | 128 | 5.2 MB | 678.9 | <339.5 | 16 GB/s |
| prefill_t2048 | 2048 | 84 MB | 641.7 | <320.9 | 262 GB/s |
| prefill_t8192 | 8192 | 336 MB | 669.5 | <334.7 | >1 TB/s（超出 HW 规格） |
| decode_t1_q16_kv2 / prefill_t2048_q16_kv2 | — | 同比例 | 640.8 / 669.9 | <320.4 / <335.0 | 同上档 |

判读：

- **基线近乎平坦（636→669µs，T 增 8192 倍仅 +5%）**，表明其耗时由固定开销主导、未反映逐 token 的设备侧数据搬运随规模扩展——`t8192` 档"目标隐含带宽"超出硬件规格正源于此口径差异；
- **验收口径**：以 **msprof 采集双方 kernel 设备时间同口径对比**（NPU 时延 < baseline/2），并在自测报告中同时给出端到端时延对照；
- **设备时间维度**：Triton-NPU 向量化/发射劣势 + 潜在多段发射，使其实际设备时间显著高于融合 Ascend C kernel（后者按 §3.2.2 单遍流达理论下限流量；2048 档 262GB/s 在 910B3 HBM（约 400GB/s 量级）规格内可达）；`t8192` 档为首要在联调初期以 msprof 双侧实测确认项，若口径澄清后仍有缺口，以 tile/双缓冲调优收敛。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B3） | √ |

## 3.4 算子约束限制

- 固定 `head_size=256、rope_dim=64、mrope_section=[11,11,10]`（attr 扩展预留：eps/interleaved/has_gate/heads 已可配，section 常量内建）；
- 输入 bfloat16 / ND；qkv 行宽须满足 `W = q_size + gate_size + 2·kv_size`；cos_sin 末维 64；
- q_bias/k_bias 可选（缺省无偏置，与 case.json `calc_expect_func_no_bias` 一致）；
- 无 workspace、无 GM 中间张量；不支持非 32B 对齐的行宽扩展（当前 W 均 32B 对齐）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | Q/K 逐元素 vs golden（torch fp32 参考实现 golden.py）`max_diff < 1e-3`；V/Gate 误差为 0（bit 级一致） | 任务书 §3.2；case.json err_threshold |
| 性能标准 | 7 条 case NPU 运行时延 < Triton 基线/2（msprof 同口径 kernel 设备时间；报告附端到端对照） | 任务书 §3.3 |

## 4.2 测试方案

- **数据**：`generate_data.py` 按 case.json 生成 qkv/cos_sin 的 bin（value_range ±1，固定种子）；权重/偏置由框架按 value_range 填充；
- **精度**：`tests/test_split_qkv_rmsnorm_mrope.py` 遍历 7 条 case，调用 golden.py 期望函数（ascend op test 框架 `expect_func` 挂接）逐输出比对（Q/K 1e-3，V/Gate 0）；interleaved 两种模式各补一组离线断言（轴分布 11/11/10）；
- **性能**：同脚本内以 torch profiler / msprof 采集本算子与 vllm-ascend Triton 基线的设备时间，输出 7 条 case 对照表（目标：每条 < baseline/2）；
- **复现**：按 rope_matrix README 流程——`npu_ops_transformer_ext` 构建 wheel、安装后在 tests/ 执行脚本；步骤写入本算子 README。

## 4.3 兼容性分析

- 纯新增算子目录，不改仓内既有算子与公共头；
- torch 扩展入口按 `npu_ops_transformer_ext` 命名空间注册，vllm-ascend 侧替换为后续独立工作（本任务交付仓内算子与接口）；
- 输入输出 dtype/布局与 Triton 版完全一致（含 interleaved 模式），调用方无感知迁移。

# 5. 参考链接

1. 任务书与测试包：`split_qkv_rmsnorm_mrope算子开发.md`、`op.json`、`case.json`、`golden.py`、`generate_data.py`；
2. ops-transformer 仓：https://gitcode.com/cann/ops-transformer （`experimental/posembedding/rope_matrix/` 工程先例、`experimental/npu_ops_transformer_ext` 构建模式）；
3. vllm-ascend Triton 基线：`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`；
4. Ascend C 算子开发文档：https://www.hiascend.com/document 。
