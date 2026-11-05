# GemmEx / SgemmEx / CgemmEx（A2/A3）算子设计文档

作者：TreamTik

# 需求背景（required）

## 需求来源

本设计对应[10月社区任务：3个GemmEx类算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/2b4e298cbf6e4ed89e7fbcfaea1a5b02)。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，实现目录为 `blas/gemm_ex/arch22/`，公共接口位于 `include/cann_ops_blas.h`。

设计面向 Atlas A2/A3，软件环境为 CANN 9.1.0。任务书允许选择一种款型自验证；本轮环境为 A2 910B2，任务书正式性能设备为 910B3。不同设备的结果分别标注。

## 背景介绍

三个接口执行扩展矩阵乘法，使用句柄绑定的 stream，支持转置、共轭转置和前导维 padding。两个实数接口共用 FP32 计算路径，复数接口在实虚交错布局上执行 COMPLEX64 运算。

# 需求分析（required）

## 需求描述

计算公式为 `C = alpha * op(A) * op(B) + beta * C`。例如，N/N 模式下，A 为 `m×k`、B 为 `k×n`，输出 C 为 `m×n`，并原地覆写 C 的有效区域。

| 接口 | 公共签名 | A/B/C 主路径 | alpha/beta |
| --- | --- | --- | --- |
| `aclblasGemmEx` | 19 参数，含 computeType、algo | `ACL_FLOAT` | `const void*` 指向 Host FP32 |
| `aclblasSgemmEx` | 17 参数，无 computeType、algo | `ACL_FLOAT` | `const float*`，Host FP32 |
| `aclblasCgemmEx` | 17 参数，无 computeType、algo | `ACL_COMPLEX64` | `const aclblasComplex*`，Host 复数 |

GemmEx 的 arch22 主路径仅接受 `ACLBLAS_COMPUTE_32F` 与 `ACLBLAS_GEMM_DEFAULT`。本设计的 arch22 支持集不包含 FP16/BF16/FP8 混合组合，不用主路径 CSV 证明扩展类型支持。

## 布局与边界

矩阵采用列主序，元素 `(r,c)` 的偏移为 `r+c*ld`。N 表示原矩阵，T 表示转置，C 表示共轭转置；实数 C 与 T 等价。复数按 FP32 实部、虚部交错存储。

| 前导维 | trans=N | trans=T/C |
| --- | --- | --- |
| lda | `>=max(1,m)` | `>=max(1,k)` |
| ldb | `>=max(1,k)` | `>=max(1,n)` |
| ldc | `>=max(1,m)` | `>=max(1,m)` |

支持上述前导维产生的 padding。不支持其他任意步长或广播。A/B 只读；各路径只写 C 的 `m×n` 有效区，保留列间 padding。

`m/n/k` 非负。所有参数检查通过后，m=0 或 n=0 返回成功且不启动计算。k=0 或 alpha=0 时只执行 beta 缩放，不引用 A/B。beta=0 时不读取 C 旧值；任务书允许的 `beta=0 && C=nullptr` 返回成功，整个调用不访问 C。alpha/beta 指针始终需要有效。

# 详细设计（required）

## Host 侧设计

`gemm_ex_host.cpp` 提供三个公开入口。`ValidateFloatCall` 共用两个实数接口的检查，`ValidateComplexCall` 检查复数接口。

检查顺序为：handle → m/n/k → lda/ldb/ldc → alpha/beta → dtype 支持集 → transA/transB → 必需矩阵指针 → GemmEx 的 computeType/algo → no-op。该顺序也适用于多个非法条件同时出现的调用。

空 handle 返回 `HANDLE_IS_NULLPTR`；负维度、非法前导维和必需空指针返回 `INVALID_VALUE`；非法转置返回 `INVALID_ENUM`；支持集外的 dtype、computeType 或 algo 返回 `NOT_SUPPORTED`。状态码均使用 `ACLBLAS_STATUS_` 前缀。

Host 根据形状、转置、标量、对齐和容量选择计算路径，不读取矩阵内容计算结果。输入极值检查和软件 FP32 舍入在 Device 路径内完成。路径选择不依赖 case 编号、随机种子或性能基线。

## 实数计算路径

两个实数接口调用 `RunRealGemm`，由 `gemm_ex_real_kernel.cpp` 及相关 kernel 实现。

| 条件 | 路径与设计目的 |
| --- | --- |
| alpha=1、beta=0，m/n/k 均不超过32，mnk不超过4096，A/B物理跨度各不超过8192个元素 | Tiny kernel 将输入保留在片上，减少启动和反复搬运 |
| mnk不超过4096，或 alpha/beta 非默认值 | Vector kernel 按 K 顺序累加，直接应用标量和旧 C |
| 其他正常矩阵乘 | Cube `MatmulImpl` 按输出 tile 分核，关闭 HF32，使用 FP32 计算 |
| k=0 或 alpha=0 | Vector epilogue 只缩放有效 C；beta=1 保持原值，beta=0 直接置零 |

主 Cube 路径按真实 lda/ldb 读取矩阵，按尾块有效尺寸计算，不统一复制整矩阵。输出 tile 使用32、64、128等基础轴长，短 K 场景可扩大单核区域以复用输入。默认 alpha/beta 可按 ldc 直接写回 C，省去额外输出阶段；短 K NN 的 NZ 专用路径先将中间结果写入工作区，再转换为逻辑 C。

短 K、窄矩阵和低输出 tile 数量场景使用各自受几何、对齐和片上容量约束的 kernel。其余场景保持通用 Cube 路径。小输出、较大标量或极值输入的有序路径使用 `GemmSbFp32::Fma` 处理 FP32 舍入，保持在 Device 上计算；精度与性能仍逐用例实测。

## 复数计算路径

`RunComplexGemm` 在 `gemm_ex_complex_kernel.cpp` 和相关 kernel 中完成复数运算。对 `A=Ar+iAi`、`B=Br+iBi`，乘积使用四个实数乘积：

```text
Prr = Ar * Br       Pii = Ai * Bi
Pri = Ar * Bi       Pir = Ai * Br
P = (Prr - Pii) + i * (Pri + Pir)
C = alpha * P + beta * C
```

复数正常路径分为以下几类：

- m/n/k 均不超过32且 A/B 物理跨度各不超过4096个复数元素时，Tiny kernel 在片上读取输入并完成输出。
- `m*n<=128 && k>64` 时，Ordered kernel 按 K 顺序完成小输出归约，控制分解与启动代价。
- 默认复数标量、NN、短 K（不超过32）、m/n均大于32且输出至少65536个元素时，可采用成对打包与 Cube 计算，直接写交错输出。
- 其他路径拆分实虚平面，再计算四个实数乘积并由 epilogue 合成。大工作量将四乘积合并到一个 Cube kernel；较小工作量调用共享实数路径。

拆分阶段按原始 lda/ldb 读取有效数据，并在转置标记为 C 时应用共轭符号。平面行跨度向8个 FP32对齐；K 按转置与长度对齐至32或128的倍数，新增区域置零。通用四乘积路径在长 K 时按256分片，以控制平面相减前的误差。

窄转置矩阵、部分中等尺寸转置矩阵和窄长 K NN 场景各有受几何条件约束的 kernel。打包和输出阶段使用分组或二维搬运，保留 C 的 padding。仅缩放路径直接处理实虚交错 C，不分配矩阵乘中间平面。

## 工作区、同步与兼容性

复数通用路径的工作区包括后端 scratch、A/B 实虚平面和四个乘积平面。scratch 至少64 MiB，并覆盖对齐输出容量；各平面追加前检查总容量和地址范围。后端临时 handle view 的可用空间限定在 scratch 段，避免覆盖后续平面。

Host 使用 handle 的工作区和 stream 提交计算。CopyIn、Vector/Scalar/Cube计算与CopyOut按生产者、消费者依赖同步。不同 kernel 在同一 stream 中按提交顺序执行；调用方同步该 stream 后再读取结果。工作区复用、连续调用和非默认 stream 行为纳入专项验证。

三个接口共用公共 C ABI。SgemmEx 统一为17参数形式，并同步公共头文件、arch35定义及调用者；CgemmEx 的公共 dtype 统一为 `aclDataType`。arch35保留既有类型分派，其编译、链接和回归单独验证，不用arch22结果替代。

# 可维可测分析

## 三个独立测试入口

三个程序分别注册对应接口的 GTest suite，共用测试工具但各自读取相邻 CSV。测试资产位于 `test/gemm_ex/arch22/`。

| 可执行程序 | 测试输入 | GPU 基线 |
| --- | --- | --- |
| `gemm_ex_test` | `gemm_ex_test.csv` | `gemm_ex_gpu_baseline.csv` |
| `sgemm_ex_test` | `sgemm_ex_test.csv` | `sgemm_ex_gpu_baseline.csv` |
| `cgemm_ex_test` | `cgemm_ex_test.csv` | `cgemm_ex_gpu_baseline.csv` |

用 `(interface, case_name)` 唯一标识用例。调用、报告和 XML 按完整 case_name 精确核对，不以相同后缀或相同形状替代身份。性能基线先按对应接口及原始行关联，再核对转置与 m/n/k；重复形状保留为不同用例。GPU CSV 的 `gpu_ms` 乘1000转换为us。

## 精度标准与覆盖

Golden 使用固定 Netlib BLAS 3.10.0：实数调用 `cblas_sgemm`，复数调用 `cblas_cgemm`。检查整个逻辑 C，复数实部、虚部分别判断。

有限值逐元素采用 `abs(actual-golden)<=2^-13+2^-13*abs(golden)`。每个分量要求匹配率至少99%，并检查 `1e-2` 或 `32*ULP` 的绝对误差门槛；当前检查按逐点较宽阈值记录，保留 golden、最大误差及判定细节供复核。alpha=0 的所有 beta 缩放逐位 EXACT 比较，不以普通容差放行。

非有限值另行统计。NaN/NaN、同号Inf/Inf及NaN/任意符号Inf按任务规则等效；有限/非有限不匹配，异号Inf不匹配。非有限元素不计入有限值匹配率，且须全部通过对应规则。

附件三个测试 CSV 各1200条，分别为1000条非性能用例和200条性能用例。完整计划覆盖3600条原始用例，其中600条性能用例也先检查精度。额外覆盖padding、地址偏移、输入只读、保护区、零维、k=0、空指针、参数检查顺序、Host标量与非默认stream。

原始材料中的200条描述与附件规模存在差异。200条混合抽样清单仅为辅助覆盖，不作为本设计完整附件验证的替代。随机输入还须核对均匀/正态各50%的要求；不从填充标签推断该覆盖已满足。

## 性能标准与采集

本轮性能验证使用普通 `msprof --application`。测试程序先预热3次，再执行5次正式调用，取5次完整调用的算术平均。每次调用汇总全部kernel耗时，包括打包、全部乘积与epilogue，不只计算主Cube时间。各阶段分别核对记录数量，排除缺失样本和失败调用。

任务书的正式性能采集要求使用 `msprof op --application`，从 `OpBasicInfo*.csv` 读取 `Task Duration(us)`。3.3节规定5次warmup后取10个有效样本，4/7节要求有效样本超过10次，附件README另有超过50次描述。本轮普通msprof的3+5结果单独记录，不作为任务书正式性能已通过的证据。

任务书逐case要求 `NPU_avg_us<=GPU_us*1.25`，即 `GPU_us/NPU_avg_us>=0.8`。GPU基线只称“任务附件GPU基线”；材料对GPU型号描述不一致，不据此指定已核验型号。NPU event、GTest总耗时和历史性能结果仅作辅助分析。

## 当前验证状态与限制

本次提交为设计文档评审。已完成任务材料、实现代码、三独立测试入口及目录契约的静态核对。本轮源码的构建、完整精度、600条正式性能、内存插桩与arch35兼容性结果尚未齐备，不声明全量通过或达到正式验收标准。

SgemmEx 的 `TC_ED_218/219/220` 与 CgemmEx 的 `TC_ED_213/214/215` 对dtype=999的期望值，与任务正文的`NOT_SUPPORTED`存在差异。原始CSV和门槛保持不变，差异在自测报告单列并由任务方确认。

FP16/BF16/FP8扩展组合、A3真机覆盖、不同设备性能可比性、异步工作区生命周期与EXACT特殊值行为仍需按最终交付范围验证。设计评审不替代代码评审和原始测试证据。

AI参与方案分析、资料核对与文档整理。本文由TreamTik提交，验证结论以本轮实际源码、库哈希、原始日志和正式评审记录为依据。
