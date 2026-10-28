# KaiserWindow 算子设计文档

时间：2026 年 10 月 28 日

作者：Baibai_Hei

## 1 需求背景

### 1.1 需求来源

本项目对应 10 月社区任务「torch_npu 支持 kaiser_window 接口（A2/A3/A5）」。目标是在昇腾 NPU 上基于 Ascend C 实现与 `torch.kaiser_window`（PyTorch 2.7.1）语义一致的 Kaiser 窗生成算子，交付落点为 SiP 开源仓（https://gitcode.com/cann/sip）：

- Kernel 直调接口：`asdKaiserWindow` / `asdKaiserWindowGetWorkspaceSize`
- Torch 侧封装：`torch_sip.asd_kaiser_window`（sip_pta，`TORCH_LIBRARY` 注册）

硬件分档：A2/A3 走 ascend910b（arch22）AIV 向量路径；A5 走 ascend950（arch35）SIMT VF 路径。软件配套 CANN 9.3.0、torch / torch_npu 2.7.1。

### 1.2 背景介绍

#### 1.2.1 现状与缺口

| 参考对象 | 现状 | 本设计用途 |
| --- | --- | --- |
| `torch.kaiser_window` | CPU / GPU 可用；NPU 无原生实现 | 语义与精度对标基线 |
| SiP 仓 `ops/base` | 无 `kaiser_window` 目录 | 新增 kernel / tiling / operation |
| sip_pta | 无 `asd_kaiser_window` | 新增 PTA 封装与单测 |
| Aten `calc_i0`（Math.h） | Cephes Chebyshev 分段 | Device / Host I0 系数与路径划分依据 |

当前调用只能落到 CPU，或由多个逐元素小算子拼接，产生与输出等长的中间张量且多次下发。本任务用单次 kernel 直写输出，workspace 恒为 0。

#### 1.2.2 数学定义

```text
N      = window_length           (periodic = True)
N      = window_length - 1       (periodic = False)
alpha  = N / 2
r[i]   = (i - alpha) / alpha
t[i]   = sqrt(|1 - r[i]^2|)
out[i] = I0(beta * t[i]) / I0(beta),   i = 0 .. window_length - 1
```

`I0` 为第一类零阶修正贝塞尔函数。工程上按 Aten/Cephes 分段：

- Path A（`x <= 8`）：`I0(x) = Exp(x) * ChbevlA(x/2 - 2)`
- Path B（`x > 8`）：`I0(x) = Exp(x) * ChbevlB(32/x - 2) / Sqrt(x)`（除法自变量钳制 `ARG_EPS`）

等价稳定形式可用 `i0e`：`out[i] = i0e(beta*t[i]) / i0e(beta) * exp(beta*(t[i]-1))`；本实现采用 Host 预计算 `invI0Beta = 1/I0(beta)`，核内 `CalcI0(arg) * invI0Beta`。

边界与任务书一致：`window_length=0` 空张量；`=1` 短路为 `[1]`；`beta<0` 与 `|beta|` 等价；`beta` 为 NaN/±inf 且长度 ≥2 时输出全 NaN。

## 2 需求分析

### 2.1 需求描述

实现 KaiserWindow，使 `torch_sip.asd_kaiser_window(window_length, periodic=True, beta=12.0, *, dtype, layout, device, requires_grad)` 在 A2/A3/A5 上生成与 CPU `torch.kaiser_window` 同语义、同精度阈值的 1 维连续窗序列。

### 2.2 需求拆解

1. SiP 入口校验（shape/dtype/连续性、`windowLength>=0`）与 AspbStatus 返回码
2. Host tiling：`alpha`、`invI0Beta`、分核、A2 pathB 区间、半窗镜像策略
3. A2/A3：Ascend C 向量核（TPipe + Exp/Sqrt/Chebyshev + 可选 Gather 镜像）
4. A5：arch35 SIMT VF（MulArch35 兼容 5-GM ABI，`expf`/`sqrtf`，禁止 GM GetValue/SetValue）
5. sip_pta 封装：dtype 面 FLOAT16/BFLOAT16/FLOAT32，缺省 float32；device 仅 npu
6. 精度 1000 + 性能 200 自验（混合容差；A2/A3 vs A100 0.8x，A5 vs H100 0.4x）

### 2.3 外部依赖

| 组件 | 要求 | 验证 |
| --- | --- | --- |
| CANN 9.3.0 + Ascend C | 910B 向量 / 950 SIMT 编译运行 | set_env + 上板 |
| SiP / asdsip | kernel 直调与 libasdsip | 接口联调 |
| torch 2.7.1 + torch_npu 2.7.1 | PTA 与 golden | verify_accuracy / verify_torch_api |
| 任务书 test_cases | CSV + 脚本契约 | gen_csv 固定种子复现 |

## 3 详细设计

### 3.1 算子分析

#### 3.1.1 接口定义

SiP（extern "C"）：

```c
AsdSip::AspbStatus asdKaiserWindowGetWorkspaceSize(size_t *workspaceSize);

AsdSip::AspbStatus asdKaiserWindow(
    aclTensor *outTensor,
    int64_t    windowLength,
    bool       periodic,
    double     beta,
    void      *stream,
    void      *workspace);
```

Torch：`torch_sip.asd_kaiser_window(...)`，参数语义对齐 `torch.kaiser_window`；layout 仅 strided，requires_grad 由封装层透传，不进 kernel。

#### 3.1.2 支持数据类型与形状

| 项 | 内容 |
| --- | --- |
| out dtype | FLOAT16 / BFLOAT16 / FLOAT32（核内 FP32 计算，写回时收窄） |
| shape | `[windowLength]`，1 维连续；0 合法空张量 |
| 输入张量 | 无；仅 host 标量 + 预分配 out |
| workspace | 恒 0 |

#### 3.1.3 工程落点（SiP）

```text
core/base/kaiser_window.cpp
include/base_api.h
ops/base/kaiser_window/          # operation / kernel / tiling / op_kernel
ops/include/params/kaiser_window.h
configs/op_list.yaml             # ascend910b: true, ascend950: true
sip_pta/csrc/base/kaiser_window.cpp
sip_pta/torch_sip/__init__.py
sip_pta/test/base/kaiser_window.py
```

A2/A3 与 A5 代码分目录组织，共用同一 Host 参数语义。

### 3.2 Host 侧设计

#### 3.2.1 参数处理顺序

| 顺序 | 条件 | 结果 |
| --- | --- | --- |
| 1 | `windowLength < 0` | `ACL_ERROR_INVALID_PARAM` |
| 2 | out 非 1 维 / shape 不匹配 / 非连续 | `ACL_ERROR_INVALID_PARAM` |
| 3 | dtype 非 FP16/BF16/FP32 | `ACL_ERROR_UNSUPPORTED_DATA_TYPE` |
| 4 | `windowLength == 0` | 成功，不下发 kernel，`workspaceSize=0` |
| 5 | `windowLength == 1` | Host/封装写 `1.0`，不下发公式核 |
| 6 | `beta` 取绝对值参与；NaN/Inf 自然传播 | 正常 tiling + 下发 |
| 7 | 正常路径 | 填 tiling，启动对应 SoC kernel |

#### 3.2.2 公共标量预计算

- `computeLen = periodic ? windowLength + 1 : windowLength`（periodic 时按 N=windowLength 生成后丢弃末元，输出长仍为 windowLength）
- `alpha = 0.5f * (computeLen - 1)`（与 Aten 一致：`N/2`）
- Host 侧同算法 `I0(beta)` → `invI0Beta = 1 / I0(beta)`，核内只做乘法，避免重复求 `I0(beta)`

#### 3.2.3 A2/A3（ascend910b）分核与 tiling

Tiling 字段要点：`blockNum`、`numPerCore`、`tailNumLastCore`、`tileLen`、`outLen`、`computeLen`、`beta`/`alpha`/`invI0Beta`、`pathBLo`/`pathBHi`、`halfLen`、`isPeriodic`。

分核策略：

- `outLen <= 2048`：单核，`tileLen=256`
- 更大窗：满可用 AIV 核；`tileLen` 默认 2048；超大非周期半窗路径用 `tileLen=4096`
- 按 tile 均分到各核，尾核处理剩余元素

Path B 区间（仅当 `beta > 8`）：由 `arg = beta * t(i) > 8` 反解索引区间 `[pathBLo, pathBHi)`，使大块 tile 可走对齐向量分段，减少混合路径。

半窗策略（性能）：

- `outLen >= 1M` 时只算前半（periodic：`outLen/2+1`；非周期：`(outLen+1)/2`）
- 非周期：Device 上 Gather 反向镜像写后半
- 周期：后半配对与非周期不同，大窗后半可由 Host 配对填充，降低 AIV TaskDuration

#### 3.2.4 A5（ascend950）分核与 tiling

Tiling：`outLen`、`computeLen`、`beta`、`alpha`、`invI0Beta`。

- `blockNum = GetBlockNum()`，按元素均分（前 `remain` 核多 1 个）
- 每核 `Simt::VF_CALL`，`LAUNCH_BOUND(256)`
- 不设 pathB 区间：SIMT 逐点 `CalcI0` 自然分支
- workspace=0；5 参 GM 签名对齐 MulArch35（`gm_dummy0/1` 占位 + `gm_out` + workspace + tiling），保证 dav-c310 VF 启动稳定

### 3.3 Kernel 侧设计

#### 3.3.1 A2/A3 向量核流程

编程模型：`KERNEL` + `TPipe`；过程 Init → Process（ping-pong CopyOut(i-1)/Compute(i)）→ 可选 Mirror。

1. **FillArg**：`ArithProgression` 生成索引 → `(i-alpha)/alpha` → `Abs(1-r^2)` → `Sqrt` → `*beta` 得 `arg`
2. **Exp(arg)**：硬件 `AscendC::Exp`
3. **分段求 I0**：
   - 全 A / 全 B：直接 `EvalPathA` / `EvalPathB`
   - 小 tile（≤256）混合：双路径算完后标量按索引挑选（避免近 0 误入 Path B 产生 Inf）
   - 大 tile 混合：将 Path B 区间扩到 8 对齐后向量分段（`MulAddDst` Clenshaw，A 截断 12 项 / B 截断 6 项）
4. **CopyOut**：`DataCopyPad` 写 GM
5. **Mirror（非周期半窗）**：从 GM 读回前半 → Gather 字节偏移反转 → 写对称后半

Path B 除法前对 `arg` 做 `Maxs(..., ARG_EPS)`，与 A5 `xc` 钳制一致。

#### 3.3.2 A5 SIMT 核流程

```text
for i in thread_stride(startOffset, calNum):
    rel = (i - alpha) * invAlpha
    inside = |1 - rel^2|
    arg = beta * sqrtf(inside)
    out[i] = CalcI0(arg) * invI0Beta
```

`CalcI0`：完整 Cephes A(30)/B(25) 系数 + `expf`/`sqrtf`；`x<=8` Path A，否则 Path B。输出按模板 dtype 收窄。无 GM GetValue/SetValue。

#### 3.3.3 与对标实现的关键对齐点

| 点 | 处理 |
| --- | --- |
| L=1 | 禁止走公式（periodic 公式会得 `1/I0(beta)`） |
| 端点 `r^2>1` | 开方前取绝对值，避免 NaN |
| periodic | `computeLen = L+1` 语义等价于按 N=L 丢末元 |
| 负 beta | Host 取绝对值 |
| FP16/BF16 | 核内 FP32，写回转换 |

### 3.4 Torch / PTA 封装

1. `NPUGuard` + `at::empty({window_length}, dtype, device=npu)`
2. `asdKaiserWindowGetWorkspaceSize` → 申请（本算子 0）→ `asdKaiserWindow`
3. 非法 dtype/device：`TORCH_CHECK`；不静默回退 CPU
4. 自测：`sip_pta/test/base/kaiser_window.py`；任务书 `verify_torch_api.py` 覆盖默认值与短路语义

### 3.5 目录与产品支持

| 项 | 内容 |
| --- | --- |
| 私仓原型 | `Baibai_Hei/kaiser_window`：`ascend910b/accept/`（A2 向量）、`ascend950/`（A5 SIMT）、各自 `pytorch_adapter/` |
| SiP 合入目标 | `ops/base/kaiser_window` + `sip_pta`（设计评审通过后） |
| A2 / A3 | 支持（ascend910b） |
| A5 | 支持（ascend950 / arch35） |

## 4 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列 | √ |
| Atlas A3 系列 | √ |
| Ascend 950 系列（A5） | √ |

## 5 算子约束限制

- 仅 1 维连续 out；不支持 broadcast、非连续、原地、视图
- dtype 不含 FLOAT64 / 整型 / BOOL / 复数
- 确定性：同参数两次执行逐位一致
- `beta` / `periodic` 不做取值校验码；按任务书边界表传播
- workspace 恒 0，无中间 GM 张量

## 6 可维可测分析

### 6.1 精度标准

对标 CPU `torch.kaiser_window`；golden 用 fp64 现算再按输出 dtype 收窄，同 dtype 混合容差：

| dtype | rtol / atol | matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- |
| FLOAT16 | 2^-10 | ≥0.99 | max(1e-1, 32×ULP) |
| BFLOAT16 | 2^-7 | ≥0.99 | max(1e-1, 32×ULP) |
| FLOAT32 | 2^-13 | ≥0.99 | max(1e-2, 32×ULP) |

非有限 beta 的全 NaN 用例按「两侧同 NaN」判定，不进 max_abs_error。

### 6.2 性能标准

- A2/A3：`T_A100 / T_NPU >= 0.8`
- A5：`T_H100 / T_NPU >= 0.4`
- 计时：预热 10 + 采集 60，Device 平均单次；标杆见任务书 csv

### 6.3 兼容性

新算子，无历史 aclnn/TBE 兼容负担；接口仅 SiP + sip_pta。

### 6.4 测试与验收方案

| 项 | 方法 |
| --- | --- |
| 精度 | `verify_accuracy.py --device npu`（1000 case） |
| 接口 | `verify_torch_api.py --device npu` |
| 性能 | `verify_performance.py --tier a2a3|a5`（200 case） |
| 内存 | `workspaceSize==0` 且无中间 GM |
| 私仓回归 | A2 accept / A5 arch35 全量精度脚本（含 torch_npu hook） |

## 7 风险与后续

1. 设计文档合入路径：`04_tasks/01_community-task-2026/tasklist/10-KaiserWindow-A2A3A5/Baibai_Hei/docs/design.md`（若 master 尚无任务目录，评审路径项按流程忽略）。
2. 代码合入 SiP PR 待设计评审通过后提交；A2/A5 原型已在私仓验证精度。
3. A3 与 A2 共用 ascend910b 实现，上板分别采集性能。
