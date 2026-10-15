# FusedExponential 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 10 月社区任务：指数分布后处理融合算子开发 |
| 提交者 | gcw_fiAo4tDr |
| 算子 / ACLNN 接口 | FusedExponential / aclnnFusedExponential |
| 分类 | 一元 Elementwise，无广播、无跨元素依赖 |
| 目标平台 | A2/A3，arch22 / DAV_2201；本次自验证仅 Ascend910B3 |
| 已验证环境 | CANN 9.1.0、AArch64、Ascend910B3、40 AIV、192 KiB UB |
| 文档范围 | 设计方案及自验证摘要；算子实现与完整验收材料在实现工程交付 |

## 修订记录

| 日期 | 版本 | 修改人 | 内容 |
| --- | --- | --- | --- |
| 2026-10-14 | 1.0 | gcw_fiAo4tDr | 根据任务书、已实现方案和 910B3 自验证结果整理设计 |

# 1. 需求背景

## 1.1 需求来源

本设计对应社区任务《FusedExponential（neg+add+mask_fill+log+mul 融合）算子任务书》。参考 op-plugin 26.1.0 的指数分布后处理：均匀采样之后，通过 neg、add、比较、mask_fill、log、mul 完成指数变换。

任务目标是将后处理融合为一个 Ascend C Vector Kernel，消除比较产生的 bool mask 中间 GM 张量，降低内存占用和多次 kernel 调度开销。

随机源由调用方提供。算子仅对输入 `u` 做确定性变换，不生成随机数；相同输入、lambda 和 dtype 下应逐位一致。

## 1.2 参考实现与现状

参考文件：[ExponentialKernelNpuOpApi.cpp（26.1.0）](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)。逻辑链为：

```text
u -> neg -> add(1) -> ge(1-eps) -> mask_fill(1-eps) -> log -> mul(-1/lambda)
```

任务书逻辑上称五算子链，性能自验实际采集包含 bool 比较，因此标杆为六个 kernel。融合内核用标量 `Mins` 直接限制上界，保持公式语义，不创建 bool mask。

# 2. 需求分析

## 2.1 数学契约

\[
\mathrm{out}_i = -\log\!\left(\min(1-u_i,1-\varepsilon)\right)/\lambda,
\quad \varepsilon=\mathrm{finfo}(\mathrm{dtype}(u)).\mathrm{eps}/2.
\]

这是对 `1-u` 的上限保护，只保护 `u` 很小的一侧；不对接近 1 的重尾一侧增加下限 clamp。

| dtype | finfo.eps | epsilon |
| --- | --- | --- |
| FLOAT16 | 2^-10 | 2^-11 |
| BFLOAT16 | 2^-7 | **2^-8** |
| FLOAT32 | 2^-23 | 2^-24 |

任务书 BF16 近似说明写为 2^-9，但其公式及自验 golden 明确使用 `torch.finfo(u.dtype).eps / 2`。BF16 有 7 个显式尾数位，1 附近间距为 2^-7，除以 2 后为 2^-8；设计和实现使用该定义，不采用近似说明值。

## 2.2 参数与边界行为

| 参数 | 位置 / 类型 | 约束 |
| --- | --- | --- |
| u | Device aclTensor；FP16/BF16/FP32，ND | 只读，调用方保证 u∈[0,1)；支持连续和非连续视图 |
| lambd | Host aclScalar；DOUBLE | lambda>0；0、负数、NaN 在 ACLNN 校验层拒绝；+inf 输出全 0 |
| out | Device aclTensor，与 u 相同 dtype/逻辑 shape | 支持独立输出及 out==u 原地输出；支持非连续视图 |

- 不支持广播；storage/view rank 上限 8。
- rank 0 的 scalar shape 含一个元素；存在零维长度的 tensor 才是空张量，两者分别处理。
- 空张量仍先完成参数校验，查询返回 workspace=0，二阶段无需启动计算 kernel。
- ACLNN 不扫描 u 值域。u>1 会使 log 自变量为负，输出 NaN；u==1 可输出 +inf。合法重尾结果超出目标 dtype 范围时允许 +inf。
- lambda 为 +inf 时显式写全 0，避免先计算 log 的特殊值再乘 0 产生 NaN。
- 采用传入 stream 异步执行，读取结果前调用方同步 stream。
- 保证 out==u 的原地语义；任意不同 view 之间的部分重叠不作为扩展承诺。

## 2.3 验收目标

按任务自验包进行 1000 条精度、200 条性能和内存检查。连续 input/output 场景无中间 GM 张量，workspace 为 0；非连续场景允许框架连续化和 ViewCopy 所需空间，单独如实记录。

任务书支持 A2/A3，自验证仅需 910B3。其他型号和 CANN 版本的运行兼容性不能由本次实测直接推定，尤其需复核第 3.6 节的零 workspace 参数 ABI。

# 3. 详细设计

## 3.1 工程与模块职责

```text
FusedExponential/
├── CMakeLists.txt、README.md
├── docs/                 # 需求、设计、ACLNN 接口、自验证说明
├── examples/             # 两段式调用示例
├── op_api/               # L2 参数校验、视图处理；L0 kernel 入队
├── op_host/              # OpDef、InferShape / InferDataType、arch22 Tiling
├── op_kernel/            # arch22 入口、模板类、TilingData、TilingKey
└── tests/ut、tests/st     # Host/API UT、真实 ACLNN ST
```

实现 PR 目标目录按任务书为 `ops-math/experimental/math/FusedExponential`。工程导出处理大写算子目录与 CANN snake_case opFile 约定；设计文档在竞赛仓的本提交目录独立评审。

## 3.2 ACLNN 两段式接口

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor* u, const aclScalar* lambd, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnFusedExponential(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);
```

L2 处理顺序：

```text
创建 executor -> 检查指针、dtype、shape、rank、ND、DOUBLE lambda
  -> 空张量快速返回
  -> Contiguous(u)
  -> Host double 计算 scale=-1/lambda，再收窄为 float
  -> 连续 out 直接入队 / 非连续 out 分配临时 tensor
  -> 归一化逻辑 shape 的 CreateView（保留 GetViewOffset）
  -> l0op::FusedExponential
  -> 必要时 ViewCopy 至非连续 out
  -> 返回 workspaceSize 与 executor
```

公共接口的 `aclScalar*` 不直接传给内部 Float 属性。内部 OpDef 属性名保留 `lambd`，实际值为预计算的 scale；该转换既保持公共 DOUBLE 标量契约，也减少 device 端除法。

L0 保持输入输出对应关系，推导 shape/dtype 并入队单个融合 kernel。InferShape 复制输入逻辑 shape，InferDataType 复制输入 dtype。

## 3.3 Host Tiling：核间切分

通过 `PlatformAscendC` 查询 AIV 核数和 UB 大小，按输入 origin/view 的元素数切分。不能使用 storage 元素数：连续切片可能与较大的底层 storage 共享内存，计算范围必须限制在实际 view。

设 N 为总元素数，e 为 dtype 字节数，C 为运行时 AIV 核数：

```text
candidateCore = min(C, max(1, ceil(N*e / 4096)))
blockFactor = align_up(ceil(N / candidateCore), 512)  # 元素单位
blockDim = ceil(N / blockFactor)
```

每核长度 `min(blockFactor, N-blockIdx*blockFactor)`，没有跨核依赖。N=0 时设 blockDim=1 和单缓冲模板；ACLNN 空张量路径不启动此 kernel。

小输入限制核数，避免调度开销过高；大输入使用多核，提高数据搬运和 Vector 计算吞吐。Host 校验平台数据、dtype、输入输出 numel 和所需属性。

## 3.4 UB 切分与模板路由

UB 预留 8 KiB，N>=4096 配置双缓冲，较小输入配置单缓冲。tile 长度按 256B 对齐，并以单核 blockFactor 为上限。

| 路径 | UB buffer | 双缓冲每元素字节 | 单缓冲每元素字节 |
| --- | --- | ---: | ---: |
| FP32 | input queue、output queue | 16 | 8 |
| FP16/BF16 | input queue、output queue、复用的 FP32 workBuf | 12 | 8 |

```text
usableUB = runtimeUB - 8192
ubFactor = floor_align(usableUB / bytesPerElement, 256/e)
ubFactor = min(ubFactor, align_up(blockFactor, 256/e))
```

典型 192 KiB UB 下，双缓冲 FP32 tile 上限 11,776 元素，FP16/BF16 上限 15,616 元素。实际值以运行时资源和 blockFactor 为准。

TilingKey 为 `(dtype, bufferMode)`，3 种 dtype×2 种模式形成 6 个模板实例，不依赖每元素动态判断。

```cpp
struct FusedExponentialTilingData {
    int64_t totalNum;
    int64_t blockFactor;
    int64_t ubFactor;
    float scale;
    float epsilon;
};
```

## 3.5 Kernel 数据流与同步

每个 tile 执行 `CopyIn -> Compute -> CopyOut`：

1. `DataCopyPad` 将有效元素从 GM 搬至输入 queue，EnQue/DeQue 管理搬运和计算依赖。
2. FP16/BF16 用 `Cast(..., CAST_NONE)` 转 FP32 workBuf；FP32 直接从输入 LocalTensor 计算至输出 LocalTensor。
3. 使用 `Muls(-1) -> Adds(1) -> Mins(1-eps) -> Ln -> Muls(scale)`。依赖的 Vector 操作之间用 `PipeBarrier<PIPE_V>()` 同步。
4. 半精度输出用 `CAST_RINT` 收窄，workBuf 在 Cast 完成后复用。
5. 输出 queue 搬回 GM，`DataCopyPad` 只写有效字节，不写尾部保护区。

`Mins` 等价于比较后将超过上界的值替换为标量，无需 bool mask。变换全在 UB 内完成，不将 neg、add、clamp、ln 等中间结果落 GM。

原地场景先将本 tile 输入完整读入 UB，再写回对应 GM 区间；各 tile/核写入的区间不重叠。独立输出场景保持输入只读。

## 3.6 零 workspace 的 CANN 9.1.0 ABI 处理

连续 input/output 时，Tiling 使用 `GetWorkspaceSizes(0)`，L0 不声明 OP_WORKSPACE。实测 `GetWorkspaceSizes(1)[0]=0` 会被 executor 规划为 512B；workspace 数量必须为 0 才满足严格验收。

该版本实际 launch 参数表为 `[u, out, tiling, ...]`，但动态编译包装仍要求四参数源入口。实现保留包装形式，在 device 侧从 `get_para_base()[2]` 获取真实 tiling 地址，不读取 workspace、不隐藏申请 GM 空间。

这是针对 **DAV_2201 / CANN 9.1.0** 的实测 ABI 兼容处理。升级 CANN 或迁移设备时必须检查包装、参数槽和连续场景 workspace，并重新运行 ABI/ST；不能宣称对所有 CANN 9.1.0 以上版本已验证。

# 4. 性能优化方案

- 将标杆的比较和五次后处理融合为单个 kernel，减少 launch 和 GM 往返。
- 用 `Mins` 标量上界替代 bool mask，减少 GM 分配和比较输出。
- 半精度只保留一个 FP32 workBuf，逐步骤原地复用，控制 UB 占用。
- Host 预计算 scale，将 DOUBLE 标量转换为内部 float 乘法系数。
- 大 shape 多核并行，小 shape 减少核数，按 shape 选择 buffer 配置。
- 连续视图保留 offset，直接读写 caller 内存；仅非连续视图走框架复制。

缓冲区配置为后续流水优化保留空间；性能结论以真实 profiler/ATK 数据为准，不从“双缓冲”配置直接推定搬运计算完全重叠。

# 5. 可维可测分析

## 5.1 精度与性能判据

精度使用自验包 CPU float64 五算子组合 golden，按输出 dtype 收窄。ATK default mixed tolerance v2 配置与任务书的十进制阈值口径一致：

| dtype | atol / rtol | matched ratio | 固定绝对误差限（或 32 ULP） |
| --- | --- | --- | --- |
| FP16 | 1e-3 | >=0.99 | 0.1 |
| BF16 | 7.8e-3 | >=0.99 | 0.1 |
| FP32 | 1.2e-4 | >=0.99 | 0.01 |

性能使用任务包 200 条固定子集和提供的 NPU 拼接链耗时，逐条要求 `ratio=拼接链耗时/融合耗时>=1`。内存查询之外执行二阶段并同步 stream，确认零 workspace 路径可以真实运行。

## 5.2 测试覆盖及自验证摘要

2026-10-14，Ascend910B3 / CANN 9.1.0 实测：

| 项目 | 结果 | 范围与说明 |
| --- | --- | --- |
| 正式 ATK 精度 | 1000/1000 | 三 dtype、逻辑 shape、非连续输入及边界 |
| 性能 | 200/200 达标 | ratio 最小 3.0561、中位 5.5747、最大 10.8617 |
| 内存 | 12/12 实际执行成功 | 6 连续 input/output workspace=0；6 非连续框架 workspace 如实记录 |
| 独立严格 ACLNN ST | 141/141 | 尾部、UB边界、多核、rank0/8、空张量、原地、offset/stride、guard、输入只读及确定性 |
| 独立 Host/API UT | 106/106 | 参数校验、Host 注册/Tiling、空两阶段、L0调度；行覆盖率84.2%、函数覆盖率80% |
| 单场景 msprof | 每调用 1 个融合 kernel | FP32、1,048,577 元素场景，仅FusedExponential统计行 |

原版 ATK 26.9.8 首轮为 996 条通过，4 条在 non-contiguous 空切片 helper 造数阶段失败，尚未进入算子。修补 helper 的 slice stop 为 `original_shape[axis]*2`，保留原备份和失败日志；CPU helper 39 组回归通过后全量复跑为 1000/1000。冻结 cases、golden 和性能基线均未修改。

独立 UT/ST 和本次自验证不等同于社区 CI。ops-math 社区工程已实际打包、安装并通过 141 条 ST；社区全套 UT、静态检查与外部 CI 属于实现 PR 的后续合入检查，不在本设计 PR 中声称通过。

## 5.3 兼容性与维护风险

| 风险 | 维护要求 |
| --- | --- |
| CANN launch ABI 变化 | 复核零 workspace 参数表并重跑连续场景 ST |
| 设备/UB资源变化 | 运行时查询资源；新设备独立验证，不继承910B3结论 |
| BF16 epsilon理解差异 | 保持 finfo/2 数学定义及边界 golden |
| 非连续临时workspace | 区分框架复制空间和融合kernel自身空间 |
| 构建自动生成API头 | 安装公共头保持 aclScalar ABI，核对生成包与源码 |

# 6. 参考资料

- [社区任务参与与设计文档提交要求](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
- [社区任务官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [op-plugin 26.1.0 参考实现](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)
- [生态混合容差精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
- [ATK 工具仓](https://gitcode.com/Ascend/ATK)
- [ops-math 贡献指南](https://gitcode.com/cann/ops-math/blob/master/CONTRIBUTING.md)
- [ops-math AI Core 开发指南](https://gitcode.com/cann/ops-math/blob/master/docs/zh/develop/aicore_develop_guide.md)
