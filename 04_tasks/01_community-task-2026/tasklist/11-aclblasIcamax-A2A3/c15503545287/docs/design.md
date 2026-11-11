# aclblasIcamax 算子设计文档（A2/A3）

# 需求背景

## 需求来源

11月社区任务《aclblasIcamax 算子开发（A2A3）》要求在 Atlas A2/A3 上使用 Ascend C 实现单精度复数向量最大模元素索引，代码合入 [ops-blas](https://gitcode.com/cann/ops-blas)。本设计依据任务书第 2、3、4 节，说明接口语义、归约方案及自验证方法。

## 背景介绍

BLAS 的复数向量最大模索引按实部、虚部绝对值之和比较，输出 1-based 索引。ops-blas 已提供公共接口声明和 arch35 实现；本任务补充 arch22 实现及其测试，复用已有 handle 和 stream 机制。

# 需求分析

## 需求描述

在 CANN 9.1.0 环境下支持 Atlas A2/A3 的 `aclblasIcamax`，输出索引与任务书指定的 Netlib BLAS 参考结果精确一致。正常计算读取 device 复数向量并写 device INT32 标量，接口以 handle 绑定的 stream 异步执行。

## 需求拆解

1. 支持 complex64 输入及正整数步长，按 `abs(real)+abs(imag)` 比较，平局取最小逻辑索引。
2. 校验句柄、输出、维度、输入与有效步长寻址范围；零长度或非正步长写 0。
3. 实现分块归约，覆盖非对齐尾块、Inf、NaN 和跨块平局。
4. 建立 arch22 CSV 驱动 GTest，自验完整任务用例及补充的边界、异常、正态分布用例。
5. 在 910B3 进行任务规定的性能采样，逐项对照 GPU 基线；保留原始样本及用例标识。

# 详细设计

## 算子分析

### 数学公式

对 `i=0,...,n-1`，令 `m[i] = abs(x[i*incx].real) + abs(x[i*incx].imag)`，正常路径返回第一个取得最大值的逻辑索引加 1，不使用欧氏模代替上述比较量。比较使用 FP32 加法结果，按 CPU 参考循环严格大于更新；首元素模为 NaN 时返回 1，其他 NaN 不参与更新。

### 接口、数据类型与形状

```cpp
aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n,
                              const aclblasComplex* x, int incx, int* result);
```

| 参数 | 位置与类型 | 语义 |
| --- | --- | --- |
| handle | Host，aclblasHandle_t | 有效句柄，包含绑定 stream |
| n | Host，int | 逻辑元素个数 |
| x | Device，complex64 | 实虚部分别为 FP32、交错存储；正常路径物理长度至少为 1+(n-1)*incx 个复数 |
| incx | Host，int | 相邻逻辑元素的复数步长 |
| result | Device，int32 | 输出一个 1-based 索引，quick return 输出 0 |

## 算子实现

### Host 侧设计

1. 检查 handle、result、n；handle 为空返回 HANDLE_IS_NULLPTR，result 为空或 n<0 返回 INVALID_VALUE。
2. n=0 或 incx<1 时，使用绑定 stream 异步写 device result=0，不启动计算 kernel。正常路径检查 x，使用 int64 计算 `(n-1)*incx`，超过 INT32_MAX 返回 INVALID_VALUE。
3. 动态获取可用 AIV 核数与 UB 容量，按 n、incx 和对齐约束确定核数、每核逻辑区间及 tile 大小。地址乘积使用足够宽的整数，避免复数到浮点偏移转换时溢出。
4. 从现有 handle 工作区获取中间结果空间并校验容量；按 stream 顺序调度局部归约和最终归约，检查可获得的运行时错误，保证异步工作区生命周期。小规模可以使用单核路径减少调度，阈值由实测确定。

### Kernel 侧设计

- 将逻辑向量划成连续区间。incx=1 采用连续批量搬运、向量绝对值与实虚部求和；incx>1 按平台支持的跨步搬运能力分块取数。按 UB 容量选择 tile，尾块使用合法长度搬运并为填充位置设置无效标记。
- 每个候选项保存有效标记、FP32 模值和逻辑索引。全局首项 NaN 时结果固定为 1；其余 NaN 标记为无效。有效项按模值降序、索引升序合并。多个 Inf 或其他同分值元素取最小逻辑索引；不能把 NaN 与 Inf 简单视作同一候选。索引全程保持整数，避免浮点转换丢失精度。
- 各核写独立 workspace 槽位，由后续归约阶段合并，或在单核路径直接输出。两个阶段在同一 stream 中按顺序提交，避免无保障的核间可见性假设。最终将逻辑索引加 1，仅写出一个 INT32 标量，不覆盖 result 之外的内存。
- 小规模、单位步长和跨步场景分别评估搬运、分核和启动开销；双缓冲、tile 和核数以实际资源约束和性能测量为依据，不预设加速结果。

### 工程组织

实现位于 `blas/iamax/arch22/`；公共声明使用现有 `include/cann_ops_blas.h`。测试放入 `test/iamax/icamax/arch22/`，在 `test/iamax/icamax/CMakeLists.txt` 的 arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} icamax_test)`；更新算子 README 的 A2/A3 支持信息。CPU golden 使用 Netlib BLAS 3.10.0 的 `cblas_icamax`，其 0-based 返回值加 1 后与 NPU 输出精确比较。补足溢出和异常测试，避免测试包装器将 host 输出指针传作 device 指针。

## 支持硬件

| 芯片 | 支持范围 |
| --- | --- |
| Atlas A2 | 目标适配；性能测试使用 910B3 |
| Atlas A3 | 目标适配 |

## 算子约束限制

不支持负步长反向遍历；incx<1 为 quick return。有效步长下逻辑末元素偏移不得超过 INT32_MAX。x 和 result 为 device 指针，调用方在读取结果前同步绑定 stream。实际 A2/A3 覆盖和资源占用以设备实测为准。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 设计和验证方法 | 标准来源 |
| --- | --- | --- |
| 精度 | INT32 索引与 Netlib cblas golden 逐例完全相等，比较语义为严格大于更新；执行原 1000 精度用例并补充分布、特殊值、边界和错误组合 | 任务书 §3.2、§3.5 |
| 性能 | 910B3 上按任务书使用 msprof op，5 次 warmup 后不少于 10 次有效采样；平均 task duration 不超过对应 GPU 基线的 1.25 倍 | 任务书 §3.3 |
| 内存 | 无单独内存性能阈值；按交付要求记录工作区占用和内存自验证材料 | 任务书 §3.4、§4 |

测试覆盖 n=0/1、幂及邻值、大规模、incx=1/2/3 和非正步长、INT32_MIN、寻址越界、空指针、全零、相同最大值跨 tile/核、首位与非首 NaN、Inf 与极端值；负向及 quick return 使用单元素或空输入占位。保留原用例参数、seed 和分布，补充正态分布及均匀分布并统计比例。补充区分 1-范数与欧氏模的专项数据。每次构建记录源码、库、环境和逐例结果。

性能记录需保留 case ID 与原基线行，重复参数不得用键覆盖。任务包 README/脚本采用 event 计时，与任务书 msprof 判据存在差异；本设计按任务书准备 msprof 验证，同时保留 event 数据，最终验收口径向任务方确认。两阶段实现统计一次接口调用涉及的全部 kernel，并统一 us/ms 单位。

## 兼容性分析

沿用已公开的公共 API 和 handle/stream 约定，不新增私有平行接口。arch22 与现有 arch35 实现按平台构建选择；quick return、NaN 和错误处理由测试验证。按任务要求准备自测报告、日志及复现步骤，实际硬件覆盖在自测报告中说明。
