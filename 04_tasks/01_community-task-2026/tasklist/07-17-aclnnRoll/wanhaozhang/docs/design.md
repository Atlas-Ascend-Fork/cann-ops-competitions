# aclnnRoll complex64 扩展算子设计文档

- 团队：wanhaozhang
- 任务：202607-aclnnRoll
- 目标仓库：cann-ops/ops-math

# 需求背景（required）

## 需求来源

本设计对应 CANN 2026 年 7 月社区任务“aclnnRoll 算子开发”。任务要求在现有
`aclnnRoll` 实现上新增 `complex64` 输入输出能力，使
`torch.fft.fftshift`、`torch.fft.ifftshift` 的主流复数场景能够在昇腾 NPU
上执行，同时保持原有数据类型的功能和性能。

## 背景介绍

`Roll` 按指定维度循环移动张量元素。PyTorch 的 `torch.fft.fftshift` 和
`torch.fft.iftshift` 会调用 `torch.roll` 完成频谱重排，FFT 结果通常为
`complex64`。原有 `aclnnRoll` 仅支持 `uint8`、`int8`、`bfloat16`、
`float16`、`float32`、`int32`、`uint32`，因此复数 FFT 结果会在接口参数
检查阶段被拒绝。

Roll 只改变元素位置，不对元素执行复数运算。`complex64` 由两个连续的
`float32` 分量组成、单元素占 8 字节，因此可将每个复数元素视作一个不可拆分的
64 位数据块进行搬运，完整保留实部、虚部以及 NaN、Inf、正负零等位模式。

### 现有实现能力分析

| 参数 | 参数含义 | 支持数据类型 | 数据格式 | 形状 | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入张量 | uint8、int8、bfloat16、float16、float32、int32、uint32 | ND | 0～8 维 | 支持 |
| `shifts` | 各维循环位移 | int64 列表 | - | 与 `dims` 等长；`dims` 为空时长度为 1 | - |
| `dims` | 位移维度 | int64 列表 | - | 每项位于 `[-x.dim(), x.dim()-1]` | - |
| `out` | 输出张量 | 与 `x` 相同 | ND | 与 `x` 相同 | 由 API 层适配 |

现有 Host Tiling 会规范化负维度和负位移、合并重复维度的位移，并根据有效位移维度、
数据类型字节数、GM 对齐要求和 AIV 核数切分任务。Kernel 侧已包含展平、单维、
末维、多维等搬运路径。

# 需求分析（required）

## 需求描述

在不改变 `aclnnRoll` 接口原型和现有语义的前提下：

1. 新增 `complex64` 输入输出支持，结果与 `torch.roll` 对齐。
2. 支持 ND 格式、0～8 维、正负位移、负维度、重复维度、空 Tensor 和非连续输入。
3. 保持原有七种数据类型功能不变。
4. 适配 Atlas A2、A3、A5 训练系列产品，CANN 8.5.0 及以上。

## 需求拆解

1. API 参数校验白名单加入 `complex64`。
2. OpDef 输入输出 dtype 注册加入 `DT_COMPLEX64`。
3. Tiling 将 `complex64` 元素大小识别为 8 字节。
4. Kernel 将 `complex64` 映射为 64 位无符号搬运类型，避免复数算术依赖。
5. 注册 A2、A3、A5 对应 SoC 配置。
6. 新增 API UT、Tiling UT、真机 ST 和原有 dtype 回归用例。
7. 输出可复现的构建、精度和性能测试材料。

# 详细设计（required）

## 算子分析

### 数学公式

设输入张量为 `x`，第 `d` 维长度为 `N_d`，该维规范化后的位移为：

```text
s_d = ((shift_d % N_d) + N_d) % N_d
```

输出坐标 `o = (o_0, ..., o_{n-1})` 对应的输入坐标为：

```text
i_d = (o_d - s_d + N_d) % N_d
y[o_0, ..., o_{n-1}] = x[i_0, ..., i_{n-1}]
```

当同一维度在 `dims` 中出现多次时，先对位移累加再取模：

```text
s_d = (Σ shift_k) mod N_d, 其中 dims_k = d
```

当 `dims` 为空时，将输入逻辑展平成一维后执行循环移动。标量输入保持不变。

### 支持数据类型

`uint8`、`int8`、`bfloat16`、`float16`、`float32`、`int32`、`uint32`、
`complex64`。

### 支持形状

- ND 格式，0～8 维。
- 输入输出形状和 dtype 必须一致。
- 维度可为负数，范围为 `[-rank, rank-1]`。
- 支持重复维度、任意正负位移和大于维长的位移。
- 支持空 Tensor。
- API 层支持非连续输入；输出非稠密时通过 `ViewCopy` 写回。

## 算子实现

### 实现方案

#### Host 侧设计

**API 参数检查与图构造**

1. 检查 `x`、`shifts`、`out`、`workspaceSize`、`executor` 非空。
2. 检查输入输出 dtype、ND 格式和形状一致，rank 不超过 8。
3. 标量要求 `shifts` 长度为 1 且 `dims` 为空；普通张量要求 `shifts` 与
   `dims` 等长，或 `dims` 为空且 `shifts` 长度为 1。
4. 对非稠密输入插入 `Contiguous`；输出为非稠密视图时先生成临时连续结果，再由
   `ViewCopy` 写回。
5. 空 Tensor 直接返回，workspace 为 0。

**Tiling 数据准备**

Tiling 保存总元素数、维度数、各维 shape/stride/shift、有效位移维数、单一有效
维度信息、分核元素数和 UB 元素数。位移统一规范到 `[0, N_d-1]`，重复维度的位移
在 Host 侧合并。

`complex64` 的元素大小设置为 8 字节。UB 预算为 64 KiB，因此：

```text
ubElements = 64 KiB / 8 B = 8192
```

此值保证一个复数元素始终作为完整的 8 字节单元进入和离开 UB。

**分核策略**

1. 从平台获取可用 AIV 核数，按 `ceil(totalNum / coreNum)` 计算初始单核元素数。
2. 单核起止位置至少按 32 字节 GM block 对齐；根据最后一个有效位移维度进一步按
   行或连续块对齐，减少跨核边界的零碎搬运。
3. 小于等于 4 KiB 的常规小张量优先单核，避免多核调度开销。
4. 最后一核处理尾块，`lastCoreElements` 不越界。
5. workspace 固定为 0，TilingKey 使用统一调度模式；具体数据路径由 TilingData
   在 Kernel 内选择。

#### Kernel 侧设计

**complex64 扩展**

构建系统为 `complex64` 生成 `ORIG_DTYPE_X == DT_COMPLEX64`。Kernel 入口在该
条件下定义：

```cpp
using RollDataType = uint64_t;
```

其他 dtype 保持原 `DTYPE_X`。`uint64_t` 与 `complex64` 同为 8 字节，Roll 又不
执行算术，因此该映射是严格的逐元素位拷贝，不改变实部和虚部的任何 bit。

**执行流程**

1. `Init` 根据 block id 计算本核 `[startIndex, elementCount]`，初始化 GM Tensor
   和 64 KiB UB 队列。
2. `Process` 根据 TilingData 选择搬运路径：
   - 无有效位移：连续复制；
   - 展平或一维：按源连续区间拆成最多两段循环搬运；
   - 首维单维：按连续大块从源位置搬运；
   - 末维单维：按行分段，在 UB 内完成首尾拼接；
   - 非末维单维：按 `shape[d] * stride[d]` 块搬运；
   - 多维：按最后有效维度选择行级或块级批量搬运；
   - 极端边界：回退到通用索引映射和分段搬运。
3. 使用 `DataCopyPad` 处理非 32 字节整块尾部，保证不越界。

**性能优化**

- 优先识别连续源区间，使用批量 `DataCopyPad`，避免逐元素处理。
- 末维按多行批量搬入 UB，再完成循环拼接。
- 非末维按连续 block 计算最大可合并段数，减少 GM 指令数。
- 分核边界与 GM block、行长或 roll block 对齐。
- `complex64` 使用原生 8 字节搬运单元，无实虚拆分、类型转换和额外 workspace。

## 支持硬件

| 支持的芯片版本 | SoC 配置 | 涉及勾选 |
| --- | --- | --- |
| Atlas A2 训练系列 | `ascend910b` | √ |
| Atlas A3 训练系列 | `ascend910_93` | √ |
| Atlas A5 训练系列 | `ascend950` | √ |

## 算子约束限制

任务书声明无额外约束。接口通用约束如下：

- `x` 与 `out` 的 dtype、shape 必须一致，且格式为 ND。
- rank 范围为 0～8。
- `dims` 非空时必须与 `shifts` 等长。
- `dims` 为空时 `shifts` 必须仅含一个值。
- `dims` 每项必须位于 `[-rank, rank-1]`。

# 可维可测分析

## 测试方案

| 类别 | 覆盖项 |
| --- | --- |
| complex64 基本功能 | 一维、二维、三维、正负位移 |
| 泛化语义 | 负维度、大位移取模、重复维度、多维同时滚动 |
| 边界 | 0 维、空 Tensor、8 维、单元素、4 KiB 阈值附近 |
| 内存布局 | 非连续输入、非稠密输出写回 |
| 回归 | uint8、int8、bfloat16、float16、float32、int32、uint32 |
| 硬件 | A2、A5 真机编译/运行；A3（`ascend910_93`）目标编译、安装包完整性与 Kernel 产物检查 |
| 性能 | 1024×1024 complex64，多次预热后统计设备执行耗时 |

日志需记录 CANN 版本、SoC、分支、commit、测试命令、逐用例结果、总体结果和性能数据。
由于测试环境的 CANN 已包含同名 `Roll`，自测包使用独立 vendor
`roll_custom`，运行时通过 `LD_PRELOAD` 优先加载该包的
`libcust_opmaster_rt2.0.so`，确保执行的是本次交付的 Host Tiling 和
Kernel，而非环境内置实现。该措施只用于隔离验证，不改变公开算子名与
ACLNN 接口。

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | complex64 与 CPU `torch.roll` 对齐，采用 AscendOpTest 默认阈值；同时进行逐元素 bit 一致性检查 | 社区任务书 |
| 回归标准 | 原有七种 dtype 的全部自测用例通过 | 社区任务书 |
| 性能标准 | 任务书无硬性性能门槛；记录真机性能数据，且 complex64 扩展不改变原 dtype Kernel 类型和路径 | 社区任务书 |

## 兼容性分析

- ACLNN 函数原型、属性语义、输出形状和 workspace 约定不变。
- 原有 dtype 仍使用原 `DTYPE_X` 实例化，complex64 条件分支仅在编译期生效。
- Host Tiling 对原类型的字节数和既有路径不做修改。
- 新增 A3、A5 SoC 注册，不影响 A2 配置。
- 非连续输入仍由既有 `Contiguous`/`ViewCopy` 机制处理。
- 独立 vendor 和 `LD_PRELOAD` 仅用于自测环境的同名算子隔离，正式集成仍注册公开算子 `Roll`。

## 风险与应对

| 风险 | 应对措施 |
| --- | --- |
| 编译器不提供可直接实例化的复数 Kernel 类型 | 使用等宽 `uint64_t` 作为纯搬运容器 |
| 复数元素被拆分导致实虚部错位 | Tiling 按 8 字节计算，Kernel 所有索引均以 8 字节元素为单位 |
| 跨核或尾块非对齐访问 | 32 字节边界切分并使用 `DataCopyPad` |
| 扩展影响原 dtype | 添加七种原 dtype 回归 ST，并保持编译期类型分支隔离 |
| 不同 SoC 编译差异 | 分别对 `ascend910b`、`ascend910_93`、`ascend950` 生成二进制包 |
| CANN 内置同名 `Roll` 抢先注册 | 自测使用独立 vendor，并预加载该 vendor 的 opmaster 库 |
