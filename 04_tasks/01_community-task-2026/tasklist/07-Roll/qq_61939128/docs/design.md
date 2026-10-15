# aclnnRoll 算子设计文档

# 需求背景（required）

## 需求来源

任务要求在 `ops-math` 开源仓已有 `aclnnRoll` 算子基础上扩展 `complex64` 输入输出能力，使 NPU 侧能够承接 `torch.fft.fftshift`、`torch.fft.ifftshift` 在 `complex64` Tensor 上的调用链路。

开发目录为 `experimental/math/roll`，参考样例目录为 `conversion/roll`。

## 背景介绍

`torch.roll` 用于沿指定维度对 Tensor 执行循环位移。`torch.fft.fftshift` 与 `torch.fft.ifftshift` 会基于 roll 语义移动频谱数据的位置，FFT 场景下输入通常为 `torch.complex64`。当前 NPU 侧 `aclnnRoll` 未支持 `complex64`，框架侧调用 `fftshift` 或 `ifftshift` 时，如果下沉到 `aclnnRoll` 且输入 dtype 为 `complex64`，会因 dtype 校验或算子注册不支持而报错。

Roll 算子的核心计算不是数值运算，而是按 PyTorch roll 语义做元素级重排。对 `complex64` 而言，每个元素由连续的两个 `float32` 组成，大小为 8 字节。实现时必须将一个复数元素作为不可拆分的最小搬运单元，保持实部和虚部的相对位置不变。

## 当前实现现状分析

### experimental/math/roll 当前能力

当前 `experimental/math/roll` 已提供 aclnn API、L0 API、host tiling、Ascend C kernel、UT、示例和接口文档。

| 模块 | 路径 | 当前能力 |
| --- | --- | --- |
| aclnn API | `experimental/math/roll/op_api/aclnn_roll.cpp` | 参数校验、非连续输入整理、非连续输出回写、调用 L0 `Roll` |
| L0 API | `experimental/math/roll/op_api/roll.cpp` | 向 launcher 注册 AICore Roll kernel |
| 算子定义 | `experimental/math/roll/op_host/roll_def.cpp` | 注册 `Roll` 输入输出 dtype、format、attr 和 AICore config |
| shape 推导 | `experimental/math/roll/op_host/roll_infershape.cpp` | 输出 shape 与输入 shape 保持一致 |
| host tiling | `experimental/math/roll/op_host/roll_tiling.cpp` | 归一化 shifts/dims、合并重复维度 shift、计算 stride、分核、UB 切分和 tiling data |
| kernel | `experimental/math/roll/op_kernel/roll.h`、`roll.cpp` | 使用 `GlobalTensor<T>`/`LocalTensor<T>` 和 `DataCopyPad` 完成一维、末维、单活动维、多活动维 roll |
| 测试 | `experimental/math/roll/tests` | 覆盖基础 API、tiling 和 kernel smoke case |

当前支持 dtype 为 `uint8`、`int8`、`bfloat16`、`float16`、`float32`、`int32`、`uint32`，未包含 `complex64`。其中 API dtype 白名单、op_def dtype 列表、tiling dtype size 分支均需要扩展。

### conversion/roll 参考能力

`conversion/roll` 是较完整的历史参考实现，具备多 tiling key、SIMD/SIMT 分支和较丰富测试。该实现同样未支持 `complex64`，但可参考如下策略：

1. host 侧根据 dtype 字节数计算 cache line、UB 元素数和分核策略。
2. `dims` 为空时按展平后的一维 Tensor 执行 roll。
3. `dims` 非空时支持负维度、重复维度和 shift 归一化。
4. 空 Tensor、0 维 Tensor、非连续 Tensor 需要独立处理。
5. golden 使用 `torch.roll` 对齐语义。

### 当前缺口

| 缺口 | 影响 |
| --- | --- |
| aclnn dtype 白名单未包含 `DT_COMPLEX64` | complex64 输入在 `aclnnRollGetWorkspaceSize` 阶段失败 |
| `Roll` op_def 未注册 `ge::DT_COMPLEX64` | 算子编译/注册侧无法生成 complex64 内核规格 |
| tiling `GetDataTypeSize` 未处理 `ge::DT_COMPLEX64` | complex64 会被误按 1 字节计算，导致分核和 UB 元素数错误 |
| kernel 未显式说明 complex64 存储搬运方式 | 直接模板化为复数类型可能受 `DataCopyPadExtParams<T>` 零值初始化和类型 API 支持限制影响 |
| 测试未覆盖 complex64 | 无法证明新 dtype 与 PyTorch `torch.roll`、`fftshift`/`ifftshift` 场景对齐 |

# 需求分析（required）

## 需求描述

在已有 `aclnnRoll` 实现基础上新增 `complex64` dtype 支持。扩展后：

1. 输入 `x` 和输出 `out` 支持 `COMPLEX64`，且二者 dtype、shape、format 一致。
2. `complex64` 结果与 PyTorch `torch.roll` 语义一致。
3. 原有 dtype 的功能和性能路径不被破坏。
4. 支持泛化输入，包括 0 到 8 维、空 Tensor、0 维 Tensor、负 shifts、负 dims、重复 dims、dims 为空的展平 roll、非连续输入和非连续输出。
5. 满足 `torch.fft.fftshift`、`torch.fft.ifftshift` 对 `complex64` 的调用需求。

## 需求拆解

1. API 层扩展：`aclnnRoll` 参数校验支持 `op::DataType::DT_COMPLEX64`，保持 `x` 与 `out` dtype 一致。
2. op_def 扩展：`Roll` 输入输出 dtype 列表新增 `ge::DT_COMPLEX64`，format 保持 `ND`。
3. tiling 扩展：host 侧将 `ge::DT_COMPLEX64` 的元素大小识别为 8 字节，所有按字节数计算的分核、对齐、UB 容量逻辑按 8 字节元素生效。
4. kernel 扩展：kernel 按 8 字节存储单元搬运 complex64，不做实部/虚部拆分和数值计算。
5. 文档扩展：更新 README/API 文档的 dtype、约束和示例说明。
6. 测试扩展：增加 API UT、tiling UT、kernel UT、ST/AscendOpTest 泛化用例和 `fftshift`/`ifftshift` 场景用例。

# 详细设计（required）

## 算子分析

### 数学公式

设输入 Tensor 为 `x`，输出 Tensor 为 `out`。当 `dims` 非空时，对每个输出坐标 `o = (o0, o1, ..., on-1)`，源坐标为：

```text
src[d] = (o[d] - normalized_shift[d] + shape[d]) % shape[d], d in active dims
src[d] = o[d], d not in active dims
out[o] = x[src]
```

当 `dims` 为空时，先将输入按逻辑连续视图展平为一维，设总元素数为 `N`：

```text
out_flat[i] = x_flat[(i - shift + N) % N]
```

重复维度的 shift 在 host 侧累加并按该维 shape 取模：

```text
normalized_shift[dim] = sum(shifts mapped to dim) % shape[dim]
```

对 `complex64`：

```text
out[o].real = x[src].real
out[o].imag = x[src].imag
```

实现上不分别计算实部和虚部，而是将一个 `complex64` 元素作为 8 字节整体搬运。

### 支持数据类型

| 参数 | 当前支持 dtype | 新增 dtype |
| --- | --- | --- |
| x | `uint8`、`int8`、`bfloat16`、`float16`、`float32`、`int32`、`uint32` | `complex64` |
| out | 与 x 相同 | `complex64` |

`complex128` 不在本任务范围内，仍返回 `ACLNN_ERR_PARAM_INVALID`。

### 支持形状

1. 支持 `ND` format。
2. 支持 0 到 8 维 Tensor。
3. 支持空 Tensor，输入为空时无需下发 kernel。
4. 支持 0 维 Tensor。0 维 Tensor 下 `dims` 必须为空，`shifts` 长度必须为 1，输出等于输入。
5. `dims` 为空时，`shifts` 长度必须为 1。
6. `dims` 非空时，`shifts` 和 `dims` 长度必须一致。
7. `dims` 元素取值范围为 `[-rank, rank - 1]`。
8. 支持非连续输入，API 层先整理为连续 Tensor；支持非连续输出，kernel 结果生成后通过 `ViewCopy` 回写。

## 算子实现

### 实现方案

#### 1. aclnn API 侧设计

修改 `experimental/math/roll/op_api/aclnn_roll.cpp`：

1. 在 `DTYPE_SUPPORT_LIST` 中新增 `op::DataType::DT_COMPLEX64`。
2. `CheckDtypeValid` 继续要求 `x` 与 `out` dtype 完全一致。`complex64` 输入只允许 `complex64` 输出。
3. `CheckFormatValid` 保持仅支持 `ND`。
4. `CheckShapeValid` 保持输入输出 shape 一致、rank 不超过 8。
5. `CheckArraySize`、`CheckDimsRange` 复用现有逻辑。
6. 非连续输入仍通过 `l0op::Contiguous` 整理，非连续输出仍通过临时输出加 `ViewCopy` 回写。

该层不需要对 `complex64` 做 cast。Roll 是重排算子，cast 会改变用户语义并引入无意义开销。

#### 2. op_def 侧设计

修改 `experimental/math/roll/op_host/roll_def.cpp`：

1. `Input("x").DataType(...)` 增加 `ge::DT_COMPLEX64`。
2. `Output("y").DataType(...)` 增加 `ge::DT_COMPLEX64`。
3. `Format(...)` 与 `UnknownShapeFormat(...)` 同步增加一个 `ge::FORMAT_ND`。
4. attr 保持不变：`shifts` 为 required listInt，`dims` 为 optional listInt，默认空数组。
5. AICore config 按任务要求覆盖 Atlas A2/A3/A5 训练系列产品；当前样例已有 `ascend910b`，后续按仓库支持方式补齐对应 SoC 配置或 binary config。

#### 3. host tiling 侧设计

修改 `experimental/math/roll/op_host/roll_tiling.cpp`：

1. `GetDataTypeSize` 新增：

```cpp
case ge::DT_COMPLEX64:
    return 8;
```

2. `totalNum` 仍表示逻辑元素个数，不按实部/虚部翻倍。
3. `ubElements = UB_BYTES / typeSize`，因此 complex64 的 UB 元素容量按 8 字节计算。
4. `elementsPerBlock = 32 / typeSize`，complex64 下为 4 个元素对齐到 32B。
5. `elementsPerBandwidthBlock = 512 / typeSize`，complex64 下为 64 个元素对齐到 512B。
6. 现有 `activeDimCount`、`activeDim`、`outerSize`、`dimSize`、`innerSize`、`activeShift` 计算不依赖 dtype，可复用。
7. 现有针对 `BF16`、`UINT8`、`FLOAT16` 的特殊性能分支保持原逻辑，不对 complex64 误套这些特化分支。complex64 默认走通用分核与搬运策略。

host tiling 产出的关键字段如下：

| 字段 | 含义 | complex64 处理 |
| --- | --- | --- |
| `totalNum` | 逻辑元素总数 | 不翻倍 |
| `dimNum` | roll 维度数 | 与当前逻辑一致 |
| `shapes[]` | 每维 shape | 与当前逻辑一致 |
| `strides[]` | 连续逻辑 stride | 以元素为单位 |
| `shifts[]` | 每维归一化 shift | 以元素为单位 |
| `usedCoreNum` | 实际使用核数 | 按 8 字节元素分核 |
| `perCoreElements` | 每核处理元素数 | 以 complex64 元素为单位 |
| `ubElements` | UB 可容纳元素数 | `64KB / 8` |

#### 4. kernel 侧设计

当前 kernel 入口为：

```cpp
RunRollKernel<DTYPE_X>(x, y, tiling);
```

Roll kernel 内部所有索引均以元素为单位，搬运长度通过 `current * sizeof(T)` 转为字节数。为保证 complex64 作为整体搬运，设计采用“存储类型映射”：

```cpp
template <typename T>
struct RollStorageType {
    using Type = T;
};

template <>
struct RollStorageType<complex64> {
    using Type = int64_t;
};
```

kernel 入口根据 `DTYPE_X` 选择实际搬运类型：

```cpp
using StorageT = typename RollStorageType<DTYPE_X>::Type;
RunRollKernel<StorageT>(x, y, tiling);
```

这样 complex64 路径中：

1. GM 地址被解释为连续的 8 字节元素数组。
2. 每次 `DataCopyPad` 的 `blockLen` 为 `elementCount * 8`。
3. UB 中的手动置换和赋值按 8 字节元素进行。
4. 不发生复数算术，不改变 NaN、Inf、符号位等 bit pattern。
5. 实部与虚部始终绑定在同一个 8 字节元素中移动。

如果 Ascend C 当前环境确认 `DataCopyPad`、`LocalTensor` 和 `GlobalTensor` 已完整支持 `complex64` 类型，也可以直接使用 `RunRollKernel<complex64>`。但从实现稳定性角度，8 字节整型存储映射更适合 Roll 这类纯搬运算子，可规避 `DataCopyPadExtParams<T>` 对 complex pad value 的类型限制。

涉及函数保持不变：

1. `ComputeInputIndex`：计算输出线性索引对应的输入线性索引。
2. `CopySegment`：按连续片段搬运。
3. `CopyFlattenRollBySource`：处理 dims 为空或一维展平 roll。
4. `CopyLastDimRollByRows`：处理仅末维 roll。
5. `CopySingleDimRollByBlocks`：处理单活动非末维 roll。
6. `CopyMultiDimLastDimRollByRows`、`CopyMultiDimNonLastRollByBlocks`：处理多维 roll。

#### 5. 文档与配置设计

需要同步更新：

1. `experimental/math/roll/README.md`：dtype 列表新增 `complex64`，说明面向 `fftshift`/`ifftshift` 场景。
2. `experimental/math/roll/docs/aclnnRoll.md`：API 参数、约束、返回错误和示例 dtype 新增 `complex64`。
3. `experimental/math/roll/op_host/config` 或构建生成配置：确保 complex64 规格被纳入二进制/简化 key 生成流程。
4. 如仓库分类文件或算子列表需要人工维护，补充 Roll complex64 支持说明。

## 支持硬件

| 支持的产品/芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 训练系列产品 | √ |
| Atlas A5 训练系列产品 | √ |

当前 `experimental/math/roll` 样例中 AICore config 已包含 `ascend910b`。按任务要求交付时，需要结合仓库现有 SoC 命名和 binary config 生成规则补齐 A3/A5 对应配置。

## 算子约束限制

1. 仅支持 `ND` format。
2. `x` 与 `out` 的 dtype 必须一致。
3. `x` 与 `out` 的 shape 必须一致。
4. 最大支持 8 维 Tensor。
5. `dims` 为空时，`shifts` 长度必须为 1。
6. `dims` 非空时，`shifts` 长度必须与 `dims` 长度一致。
7. `dims` 取值范围为 `[-rank, rank - 1]`。
8. 本任务新增支持 `complex64`，不新增 `complex128`。
9. complex64 只做元素重排，不做复数数值计算和类型转换。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | `complex64` 输出与 CPU/PyTorch `torch.roll` 结果一致；采用 AscendOpTest 默认阈值。由于 Roll 是搬运类算子，预期可做到逐元素 bit 级一致。 | 社区任务书 |
| 性能标准 | 任务书无显式性能要求；新增 `complex64` 不应影响原有 dtype 性能路径。 | 社区任务书 |
| 回归标准 | 原有 dtype UT/ST 全部通过，原有 dtype 的参数校验、非连续处理、空 Tensor 和 0 维 Tensor 行为不变。 | 当前实现与社区任务书 |

## 测试设计

### API UT

在 `experimental/math/roll/tests/ut/op_api/test_aclnn_roll.cpp` 增加：

1. `complex64` 正常路径：shape `{2, 3}`，`dims={1}`，`shifts={1}`，返回 `ACLNN_SUCCESS`。
2. `complex64` dims 为空：shape `{2, 3, 4}`，`dims={}`，`shifts={5}`，返回 `ACLNN_SUCCESS`。
3. dtype mismatch：`x=ACL_COMPLEX64`，`out=ACL_FLOAT`，返回 `ACLNN_ERR_PARAM_INVALID`。
4. unsupported dtype：`ACL_COMPLEX128` 返回 `ACLNN_ERR_PARAM_INVALID`。
5. 0 维 complex64：`dims={}`，`shifts={1}`，返回 `ACLNN_SUCCESS`。
6. invalid dims range：complex64 下 `dims` 越界仍返回 `ACLNN_ERR_PARAM_INVALID`。

### host tiling UT

在 `experimental/math/roll/tests/ut/op_host/test_roll_tiling.cpp` 增加：

1. `ge::DT_COMPLEX64` 基础末维 roll，校验 `totalNum`、`dimNum`、`activeDimCount`、`activeDim`、`activeShift`。
2. `ge::DT_COMPLEX64` dims 为空展平 roll，校验 `shapes[0]=totalNum`、`strides[0]=1`、`shifts[0]=PositiveMod(shift,totalNum)`。
3. `ge::DT_COMPLEX64` 负 dims 和负 shifts，校验归一化结果。
4. `ge::DT_COMPLEX64` 重复 dims，校验 shifts 累加取模。
5. 校验 `ubElements` 按 8 字节计算，即 `UB_BYTES / 8`。

### kernel UT

在 `experimental/math/roll/tests/ut/op_kernel/test_roll.cpp` 增加 complex64 smoke：

1. host 数据使用 `std::complex<float>` 或等价的 `{real, imag}` 连续 8 字节数据。
2. 构造一维 roll、末维 roll、多维 roll 的 tiling data。
3. kernel 输出回读后与预期元素顺序比较，实部/虚部必须保持成对移动。

### ST/泛化测试

在 ST 或 AscendOpTest 用例中覆盖：

| 场景 | shape | dims | shifts |
| --- | --- | --- | --- |
| 一维基础 | `[17]` | `[0]` | `[3]` |
| 展平 roll | `[2, 3, 4]` | `[]` | `[5]` |
| 末维 roll | `[4, 5, 7]` | `[-1]` | `[2]` |
| 多维 roll | `[3, 4, 5]` | `[0, 2]` | `[1, -2]` |
| 重复 dims | `[2, 3, 4]` | `[1, -2]` | `[1, 2]` |
| shift 超过维长 | `[5, 6]` | `[1]` | `[19]` |
| 空 Tensor | `[0, 3]` | `[1]` | `[1]` |
| 0 维 Tensor | `[]` | `[]` | `[1]` |
| 8 维边界 | `[2, 1, 3, 1, 4, 1, 5, 2]` | `[0, 2, 4, 7]` | `[1, 2, -3, 1]` |
| fftshift 偶数维 | `[8, 16]` | `[0, 1]` | `[4, 8]` |
| fftshift 奇数维 | `[7, 9]` | `[0, 1]` | `[3, 4]` |
| ifftshift 奇数维 | `[7, 9]` | `[0, 1]` | `[4, 5]` |

golden 使用 PyTorch：

```python
y = torch.roll(x, shifts=shifts, dims=dims)
```

对 `fftshift`/`ifftshift` 场景，可直接使用：

```python
fftshift_y = torch.fft.fftshift(x, dim=dims)
ifftshift_y = torch.fft.ifftshift(x, dim=dims)
```

## 兼容性分析

1. 对原有 dtype：只在 dtype 列表、op_def 列表和 dtype size 分支中追加 complex64，不改变原有 dtype 的分支条件和搬运逻辑。
2. 对 API 行为：`complex64` 遵循现有 shape、format、dims、shifts 校验规则；原有错误码保持不变。
3. 对 kernel 行为：complex64 使用 8 字节存储类型搬运，不影响 `uint8`、`int8`、`bf16`、`fp16`、`fp32`、`int32`、`uint32` 的模板实例化。
4. 对非连续 Tensor：沿用现有 `Contiguous` 和 `ViewCopy` 流程，不额外增加 complex64 特殊处理。
5. 对精度：Roll 不做数值计算，因此 NaN、Inf、正负零、任意实虚部 bit pattern 都应原样移动。

## 风险与规避

| 风险 | 影响 | 规避措施 |
| --- | --- | --- |
| complex64 被误按两个 float32 处理 | 实部/虚部可能被拆散，结果错误 | tiling 以元素为单位，kernel 以 8 字节存储类型搬运 |
| tiling dtype size 未更新 | 分核、UB 容量、对齐长度错误 | `GetDataTypeSize` 明确返回 8，并增加 tiling UT |
| kernel 直接使用 `complex64` 类型编译不通过 | DataCopy pad value 或 LocalTensor 类型受限 | 使用 `RollStorageType<complex64>::Type = int64_t` 的存储映射方案 |
| 特化性能分支误作用于 complex64 | 可能引入未验证路径 | complex64 默认走通用路径，仅保留 dtype 无关的搬运优化 |
| 原有 dtype 回归 | 影响已支持场景 | 保留并执行原有 UT/ST，新增 complex64 不修改原有判断顺序 |
