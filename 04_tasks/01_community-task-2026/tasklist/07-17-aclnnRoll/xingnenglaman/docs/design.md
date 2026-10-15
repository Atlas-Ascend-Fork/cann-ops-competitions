# Roll 算子 complex64 扩展设计文档

> 队名： xingnenglaman　任务序号：07-17　算子名称：Roll（aclnnRoll）
> 目标仓：https://gitcode.com/cann/ops-math　目标目录：`experimental/math/roll`

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 年 7 月任务：aclnnRoll 算子开发（任务书：`cann-ops-competitions/04_tasks/01_community-task-2026/docs/202607/aclnnRoll_task_doc.md`）。

## 背景介绍

`torch.fft.fftshift` 和 `torch.fft.ifftshift` 用于调整 FFT 结果的频谱序，其底层语义是沿指定维度对张量做循环位移（shift = 维度长度 // 2），框架侧映射到 `torch.roll`，torch_npu 侧进一步调用 `aclnnRoll` 两段式接口（`aclnnRollGetWorkspaceSize` / `aclnnRoll`）。

FFT 的输入输出天然是复数类型，`torch.complex64` 是该场景最主流的 dtype。当前 aclnnRoll 在 NPU 上不支持 complex 类型输入：当输入 tensor 的 dtype 为 complex64 时，aclnn 第一段接口的参数校验（dtype 白名单）直接返回 `ACLNN_ERR_PARAM_INVALID`，导致 `torch.fft.fftshift(x)`（x 为 NPU 上的 complex64 tensor）运行报错。

**最小复现**：

```python
import torch, torch_npu
x = torch.randn(4, 8, dtype=torch.complex64).npu()
y = torch.fft.fftshift(x)   # 报错：aclnnRoll 不支持 complex64
```

### aclnnRoll 算子实现现状分析

aclnnRoll 在 ops-math 仓的实现路径为 `experimental/math/roll`，采用「aclnn 接口层 → 算子注册（OpDef）→ Host 侧 InferShape/Tiling → Kernel 侧搬运」四层结构：

```
op_api/aclnn_roll.cpp    aclnn 两段式接口：参数校验（dtype/format/shape/dims 范围）、
                         非连续输入 l0op::Contiguous 转连续、非 dense 输出 ViewCopy 回写
op_host/roll_def.cpp     算子注册：输入 x / 输出 y 的 DataType + Format 列表（当前 7 种 dtype）
op_host/roll_infershape.cpp  输出 shape 恒等于输入 shape
op_host/roll_tiling.cpp  dims/shifts 归一化（负维归一化、重复维合并、dims 为空退化为 flatten roll）、
                         GetDataTypeSize 按 dtype 返回元素字节数（1/2/4）、分核与 UB 切分、kernel 路径选择
op_kernel/roll.h/.cpp    template <typename T> 纯数据搬运 kernel：连续段 Copy、按行 Roll、
                         按 Block Roll、Segmented Fallback 等多路径，全部按 sizeof(T) 折算字节搬运
```

当前支持能力：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| x | 输入 tensor | tensor | uint8, int8, bfloat16, float16, float32, int32, uint32 | ND，非连续先转连续 | 0-8 维 |
| shifts | 每维滚动步数 | listInt | - | 与 dims 等长；dims 为空时长度为 1 | - |
| dims | 滚动维度 | listInt | - | 取值 [-rank, rank) | - |
| y | 输出 tensor | tensor | 与 x 相同 | ND | 与 x 相同 |

**现状缺口**：上述四层中有三处对 dtype 做了硬编码白名单/映射，均不含 complex64：

1. `op_api/aclnn_roll.cpp`：`DTYPE_SUPPORT_LIST`（7 种）；
2. `op_host/roll_def.cpp`：x/y 的 `.DataType({...})` 注册列表（7 项）；
3. `op_host/roll_tiling.cpp`：`GetDataTypeSize()` 的 switch 分支（无 8 字节分支）。

Kernel 侧模板 `Roll<T>` 本身与 dtype 语义解耦（纯搬运），但构建系统按注册 dtype 逐一实例化 kernel（`-DDTYPE_X=<类型>`），complex64 对应的实例化类型当前不存在。

# 需求分析（required）

## 需求描述

在 aclnnRoll 现有实现基础上扩展支持 complex64 数据类型输入，使 `torch.fft.fftshift`、`torch.fft.ifftshift` 输入为 complex64 的场景在 NPU 上可用；complex64 输入下计算结果与 PyTorch `torch.roll` 语义一致；保持原有 7 种数据类型功能与性能不变；实现算子泛化，覆盖各类合法输入场景。

## 需求拆解

1. dtype 链路打通：aclnn 校验、算子注册、tiling 字节数映射、kernel 实例化四层均支持 complex64（A2 走 `experimental/math/roll`，A5 走 `conversion/roll`，两处同步扩展）；
2. 泛化支持：complex64 下 0-8 维、负 dims、重复 dims 合并、dims 为空（flatten roll）、shift 取模、非连续 Tensor 等场景与既有语义完全一致；
3. 回归保障：原 7 种 dtype 的代码路径零改动，补充回归用例；
4. 精度对齐：complex64 输出与 CPU（torch.roll）对标，采用 AscendOpTest 默认阈值（本算子为纯搬运，实际达到 bit 级一致）；
5. 配套交付：UT/ST 用例、自测脚本、接口文档与 README 更新。

# 详细设计（required）

## 算子分析

### 数学公式

沿维度 d（长度 n）滚动 s 步：

```
out[i_0, ..., j, ..., i_{r-1}] = x[i_0, ..., (j - s) mod n, ..., i_{r-1}]    (j 为维度 d 上的下标)
```

- dims 为空时：先将输入按逻辑视图展平为一维（长度 N = numel），执行一维 roll，再恢复原形状；
- dims 为负时按 `d + rank` 归一化；重复 dims 在 Host 侧合并（shift 累加后取模）；
- shift 按 `((s % n) + n) % n` 归一化到 [0, n)。

### 支持数据类型

uint8, int8, bfloat16, float16, float32, int32, uint32, **complex64（本次新增）**

### 支持形状

ND 格式，0-8 维；支持非连续 Tensor（aclnn 层先整理为连续视图）。

## 算子实现

### 总体方案

Roll 是**纯数据搬运算子**，不涉及任何算术运算。complex64 的物理布局为 8 字节 POD（两个 float32 按 real/imag 相邻存放），循环位移以「元素」为最小单位整体移动。因此：

> **将 complex64 以 8 字节存储类型（uint64_t）参与搬运，与按复数语义搬运完全等价。**

方案只需打通四层 dtype 链路，不改动任何搬运/切分/归一化逻辑。仓内已有完全相同的先例：纯搬运算子 Tile（`experimental/math/tile`）即通过 `using complex64 = uint64_t;` 支持 complex64（见 `experimental/math/tile/op_kernel/tile.h`）；`experimental/conversion/as_strided` 亦按 `ORIG_DTYPE_X == DT_COMPLEX64 → int64_t` 的存储类型映射处理。

### host 侧设计

**（1）aclnn 接口层（op_api/aclnn_roll.cpp）**

`DTYPE_SUPPORT_LIST` 增加 `op::DataType::DT_COMPLEX64`，其余校验（x/out dtype 一致、ND 格式、shape 一致、rank≤8、shifts/dims 长度与范围）与 dtype 无关，天然继承。非连续输入经 `l0op::Contiguous` 转连续、非 dense 输出经 `l0op::ViewCopy` 回写的既有机制对 complex64 自动生效。

**（2）算子注册（op_host/roll_def.cpp）**

输入 x 与输出 y 的 `.DataType({...})` 列表增加 `ge::DT_COMPLEX64`，`.Format({...})` 与 `.UnknownShapeFormat({...})` 列表同步由 7 项扩为 8 项（三个列表长度严格一致）。AICore 配置沿用 `ascend910b`。

**（3）Tiling（op_host/roll_tiling.cpp）**

`GetDataTypeSize()` 增加分支：

```cpp
case ge::DT_COMPLEX64:
    return 8;
```

该返回值仅用于 UB 切分与 32 字节块对齐的元素数折算（8 字节类型下 32B 块 = 4 元素，整除自洽）。dims/shifts 归一化、重复维合并、flatten 退化、分核策略（优先满核、余量前分）、kernel 路径选择（连续段 Copy / 按行 Roll / 按 Block Roll / Segmented Fallback）等全部逻辑均以「元素」为单位建模，**与 dtype 完全解耦，零改动**。bf16 专用的 `useSafeUbShuffle` 保护路径判定条件为 `dtype == DT_BF16`，complex64 不会误入。

**tilingkey 规划**：沿用现状（kernel 路径由 tilingData 字段驱动，complex64 不引入新 tilingkey）。

### kernel 侧设计（op_kernel/roll.h）

构建系统（opbuild）解析 OpDef 注册的 DataType 列表，为每个 dtype 单独编译 kernel 并传入 `-DDTYPE_X=<类型>`；对 complex64 传入 `DTYPE_X=complex64`。在 roll.h 头部增加类型别名，使该标识符可解析为 8 字节存储类型：

```cpp
#ifndef COMPLEX64_TYPE_DEFINED
#define COMPLEX64_TYPE_DEFINED
using complex64 = uint64_t;
#endif
```

`roll.cpp` 中 `RunRollKernel<DTYPE_X>` 随之实例化为 `RunRollKernel<uint64_t>`，`Roll<uint64_t>` 的所有搬运路径（DataCopy / DataCopyPad / scalar 拷贝，均按 `sizeof(T)=8` 折算字节数）直接复用，无任何 kernel 逻辑改动。

**等价性论证**：complex64 元素内部 real/imag 各占 4 字节且相邻，8 字节整体搬运保证 real/imag 不拆分、不错位；循环位移的最小粒度就是元素本身，故按 8 字节 POD 搬运与按复数语义搬运逐元素等价，输出与 CPU golden bit 级一致。

**备选方案对比（未采用）**：在 aclnn 层将 complex64 视图转换为 float32（shape 追加一维 [2]，flatten 场景 shift×2）复用 float32 链路。该方案需侵入 aclnn 层构造视图 tensor，改动面与回归风险显著高于别名法，且无仓内先例，故不采用。

**性能设计说明**：本方案不为 complex64 引入任何独立实现或特化分支，完全复用既有性能优化路径（连续段优先搬运、最后一维按行处理、非最后维按 block 重排、多核均分 + UB 充分切分），8 字节元素下各路径的访存效率与同字节规模 float32 场景一致；相对原 dtype 零性能回退风险（原路径代码零改动），这是本方案在性能维度上的关键设计取舍。

### arch35（ascend950）同步扩展设计（conversion/roll）

为覆盖 Atlas A5，对 `conversion/roll` 做同源扩展（其 aclnn 校验 → 注册 → tiling → kernel 链路同理）：

| 文件 | 修改内容 |
| --- | --- |
| `op_api/aclnn_roll.cpp` | 910B/910_93/950 生效的 dtype 支持列表（l1）增加 `DT_COMPLEX64` |
| `op_host/roll_def.cpp` | x/y 注册列表 8 项扩为 9 项（加 `ge::DT_COMPLEX64`，Format 列表同步） |
| `op_kernel/arch35/roll_struct.h` | 增加 `using complex64 = uint64_t;` 别名（arch35 全部 kernel 头文件均包含该文件） |
| `op_host/arch35/roll_tiling_arch35.cpp` | **无需修改**：元素字节数经 `ge::GetSizeByDataType()` 通用获取，天然支持 8 字节 |
| kernel 主体 | **无需修改**：arch35 各搬运路径（gather_simd / hsplit / unaligned_simd / simd / simt）已支持 8 字节 int64，complex64 按同宽度 POD 复用 |

> 说明：`op_host/config/ascend950/roll_binary.json` 为发布用二进制清单，其条目哈希由构建系统在出包时生成；源码编译（build.sh）按注册 dtype 自动编译 complex64 kernel，不依赖该文件。若合入时官方要求更新清单，按构建产出同步。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 实现路径 |
| --- | --- | --- |
| Atlas A2 训练系列产品（ascend910b） | √ | `experimental/math/roll` 扩展 complex64 |
| Atlas A5 训练系列产品（ascend950，arch35） | √ | `conversion/roll` 同步扩展 complex64（本任务一并交付） |
| Atlas A3 训练系列产品（ascend910_93） | 待确认 | 见下方说明 |

- 本任务**同时交付两处实现的 complex64 扩展**：`experimental/math/roll`（A2）与 `conversion/roll`（A5/ascend950）。两处 kernel 均以 `using complex64 = uint64_t;` 别名方式按 8 字节 POD 搬运，各自复用已验证的 8 字节（int64）路径；`conversion/roll` 的 tiling 通过 `ge::GetSizeByDataType` 通用获取元素字节数，天然支持 8 字节，无硬编码分支。
- `conversion/roll` 的 aclnn 层按芯片区分 dtype 支持列表，complex64 加入对 ASCEND910B / ASCEND910_93 / RegBase（950）生效的列表（l1）。
- **A3（ascend910_93）说明**：仓内两处实现均未注册 910_93 的 AICore 配置（experimental 仅 910b、conversion 仅 950），A3 的 Roll kernel 实现归属（复用 910b 二进制或商业包提供）确认结论：待确认。aclnn 层 910_93 的 complex64 校验已随 l1 列表就绪。

## 算子约束限制

- 仅支持 ND 格式；
- 支持 0-8 维；输入输出 dtype、shape 一致；
- dims 为空时 shifts 长度必须为 1；dims 非空时 shifts 与 dims 等长；dims 取值 [-rank, rank)；
- 0 维输入时 shifts 长度为 1 且 dims 为空；
- 不支持 bool、int64、complex128 等未注册 dtype（沿用现状）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | complex64 输出与 CPU（torch.roll）对齐，采用 AscendOpTest 默认阈值；本算子为纯数据搬运，实测达到 bit 级一致 | 任务书 |
| 性能标准 | 无性能要求；complex64 为 8 字节元素，耗时与同字节数 float32 规模对称为正常 | 任务书 |

## 兼容性分析

- 本次改动均为「列表/switch 新增分支」，原 7 种 dtype 的校验、注册、tiling、kernel 代码路径零改动；
- bf16 特殊路径判定条件不含 complex64，无路径串扰；
- 已补充原 dtype 回归用例（fp16/bf16/fp32/int8/uint8/int32/uint32 各典型场景），确认无功能与性能回归。

## 测试方案

| 层 | 内容 |
| --- | --- |
| op_api UT | complex64 正例（基本场景、dims 为空）+ 反例（x/out dtype 不一致） |
| op_host UT | complex64 下 InferShape/Tiling 成功、8 字节元素切分、flatten 退化、useSafeUbShuffle 不触发 |
| op_kernel UT | 以 uint64_t 实例化 Roll 模板类，验证 8 字节元素循环位移搬运结果正确 |
| 真机 ST | examples/test_aclnn_roll_complex64.cpp：complex64 两段式调用 + CPU golden bit 级比对 |
| 端到端 | torch_npu 下 torch.roll / torch.fft.fftshift / torch.fft.ifftshift complex64 用例矩阵（0-8 维、负 dims、重复 dims、dims 为空、shift 取模、非连续、特殊值），与 CPU golden 逐一 torch.equal 比对 |
| 回归 | 原 7 种 dtype 典型场景复跑 |
