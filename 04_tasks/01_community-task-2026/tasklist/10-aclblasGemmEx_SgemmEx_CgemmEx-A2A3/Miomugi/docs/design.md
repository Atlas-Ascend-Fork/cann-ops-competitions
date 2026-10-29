# 需求背景（required）

## 需求来源

10月社区任务：aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx 算子开发（A2/A3）。依据任务书 `aclblasGemmEx_SgemmEx_CgemmEx_A2_task_doc.md` 及随附三组测试 CSV、验证脚本制定本方案。本文是开发前设计，不包含尚未取得的实测成绩。

任务附件归档 SHA256：`0d2616cbcb8066f4fee7cac7e7a8bc892423f5ae2f01ffb587d9874a6d27ce24`。

## 背景介绍与仓内现状

本次检查 ops-blas master 提交 `05b96ea22cea926ecfd43fc7ac691a795522bd61`：`blas/gemm_ex/` 只有 arch35；GemmEx 为 19 参，SgemmEx 仍为 15 参，CgemmEx 尚无接口。另有 `blas/gemm/arch22/` 的实数 SGEMM，可作为 FP32 后端复用候选；arch35 的 BlockMmad 实现不是可直接复制到 arch22 的已验证方案。

代码目录目标为 `blas/gemm_ex/arch22/`，测试目录为 `test/gemm_ex/`。公共声明统一在 `include/cann_ops_blas.h`，不新建产品私有平行 API。

# 需求分析（required）

## 需求描述

三个接口计算列主序 `C = alpha * op(A) * op(B) + beta * C`。GemmEx/SgemmEx 主路径 FLOAT32；CgemmEx 主路径 COMPLEX64，实虚交错，alpha/beta 为复数。N/T/C 组合均覆盖；实数 C 与 T 等价，复数 C 为共轭转置。

| 接口 | 公共签名与主路径 | 工程处理 |
| --- | --- | --- |
| aclblasGemmEx | 19 参；void 标量与矩阵、三 dtype、computeType/algo；FP32 | 新增 arch22 分发，主路径共享实数计算后端 |
| aclblasSgemmEx | 17 参；float 标量、void 矩阵、三 dtype，无 computeType/algo；FP32 | 修正当前 15 参声明，同步 arch35 定义、测试和示例调用 |
| aclblasCgemmEx | 17 参；aclblasComplex 标量、void 矩阵、三 dtype；complex64 | 新增公共声明与 arch22 复数实现 |

不将旧 15 参 SgemmEx 保留为同名 C ABI 重载；公共接口变更要由维护者审定，并对 arch35 做编译与调用点回归。新架构支持表在验证后更新，不提前宣称实机支持已完成。

## 需求拆解与边界

1. 实现任务书规定的校验顺序：handle → 维度 → leading dimensions → alpha/beta → dtype → 转置枚举 → A/B/C 指针 → GemmEx computeType/algo → no-op。
2. m/n 为零仅在校验完成后成功返回；k=0 或 alpha=0 不访问 A/B，只处理逻辑 C。beta=0 不读取 C 旧值；任务明确规定 beta=0 且 C=nullptr 时成功返回且不写 C，包括 k=0 分支。
3. m/n/k 非负；按转置模式检查 lda/ldb/ldc。padding 不参与数值结果且不应被修改；无广播或任意步长视图支持。
4. 按§3.5与附件自验范围，GemmEx/SgemmEx接受三矩阵同为ACL_FLOAT，CgemmEx接受三矩阵同为ACL_COMPLEX64；不增加额外混合精度支持。非法aclDataType值（例如999）返回INVALID_VALUE；合法枚举但不支持的类型/组合返回NOT_SUPPORTED，保留原CSV预期。
5. 通过 handle 绑定 stream 异步执行，结果读回前测试程序同步。避免在正常计算路径中进行 Host 全矩阵拷贝和计算。

# 详细设计（required）

## 算子分析

### 数学与存储

对实数路径，C 的每个逻辑元素为 alpha 乘 K 维点积再加 beta 乘原 C。FP32 输入不得为提速而未经精度验证降为 HF32/FP16。

对复数路径令 op(A)=Ar+iAi、op(B)=Br+iBi：

```text
Pr = Ar*Br - Ai*Bi
Pi = Ar*Bi + Ai*Br
Cr' = alpha_r*Pr - alpha_i*Pi + beta_r*Cr - beta_i*Ci
Ci' = alpha_r*Pi + alpha_i*Pr + beta_r*Ci + beta_i*Cr
```

共轭在输入映射时对虚部变号，转置同时映射物理索引；ld 按复数元素计数。beta=0 的 epilogue 不加载原 C。complex64 四实 GEMM 分解是候选实现，归约/融合顺序可能影响误差，必须通过 golden 全量验证；不预设三实 GEMM 分解可满足精度。

## 算子实现

### Host 与公共工程

- 为三接口组织共享维度/枚举/指针校验及紧凑 tiling 描述，接口差异单独处理；错误码按上述非法枚举与不支持组合两层规则区分，与官方CSV预期一致。
- 实数路径优先验证可否在 Ex 完整校验后复用现有 arch22 SGEMM 计算层；现有 SGEMM 的 no-op/指针校验顺序与本任务不完全一致，不直接无条件转发。若抽取内部计算接口，保持原 SGEMM 对外行为并回归测试。
- workspace 按输入打包、分块临时乘积及必要 epilogue 计算，使用 handle 的既有机制；尺寸乘法和字节偏移检查溢出，避免写入调用者未分配的空间。不能在 stream 计算完成前复用或释放临时存储。
- 只在必要模块中共享后端，不复制整套旧算子代码。修改构建注册，使 `--ops=gemm_ex` 选择 arch22 的正确符号；同时检查 arch35 的公共声明一致性。

### FP32 kernel 与 tiling

- 基于现有 arch22 FP32 Matmul 路径验证 N/T、列主序、非紧凑 ld 和尾块，再补充 Ex 校验及缩放行为。实数 C 操作映射为 T。
- 大矩阵使用多核 M/N 分块和 K 累加；紧凑对齐路径减少 pack 与 epilogue。小矩阵采用低启动开销路径；选择依据为公开维度/布局，不按 case ID 或 baseline 特判。
- 禁止未验证地启用截断精度模式。长 K、非零均值输入、大 alpha 和抵消场景优先验证。
- k=0、alpha=0 使用仅访问逻辑 C 的设备缩放/清零路径；beta=1 保持原值，padding 保持，EXACT 规则按任务执行。

### complex64 kernel 与同步

- 首版建立保守正确性路径，验证复数标量、共轭、尾块及特殊值；随后实现拆分实虚的打包与四路实数乘积复用方案，评估中间缓冲、带宽及 kernel 启动成本。
- 尽量按块复用 A/B 实虚面板，在 epilogue 合成复数并应用 alpha/beta；避免无必要的四份全尺寸中间输出。可先用易审查的分阶段方案建立基线，再在证据支持下融合。
- 分块覆盖只写各自输出区域；GM 阶段按同 stream 顺序执行。单核内 MTE/Vector/Cube 交接遵守所用 API 同步契约；不假定函数重构保持隐式同步，不使用无依据全局同步掩盖问题。
- FMA 与复数展开顺序以 Netlib 3.10.0 实际构建为参考，锁定编译器、选项、库哈希和运行时加载路径。不能把其他复数算子的成功结果当本算子精度保证。

## 支持硬件与限制

目标 Atlas A2/A3，CANN 9.1.0，正式性能使用 910B3。任务允许自验证覆盖一种款型，计划优先 910B3 完成功能和性能；A3 没有实测时不写成双端均完成。950PR/arch35 的既有支持不应因公共接口修正而失去编译能力，缺少该设备时如实标明运行回归范围。

仅提供任务确认的 dtype 支持集；不承诺 FP8 在 arch22 可用。不要求新增内存 profiling 门槛，但实现必须保证 workspace 生命周期与越界安全。

# 可维可测分析

## 精度标准与测试覆盖

- Golden：Netlib BLAS 3.10.0 的 cblas_sgemm/cblas_cgemm，需在测试环境记录获取来源、源码版本、构建参数、实际库 SHA256。当前尚未安装或开展 NPU 测试，不将任务指定版本写成已验证构建。
- FP32 及 complex64 的实/虚分别按 rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤max(1e-2,32*ULP(g_low))判定，g_low按官方标准为最大误差点的golden值；保留固定0.01的诊断结果。
- alpha=0 的缩放场景检查任务要求的 EXACT。非有限值按本任务 §3.2.5 判定：同 NaN、同号 Inf、NaN 与 ±Inf 等效；有限与非有限不等效；异号 Inf 不按同号规则通过。非有限元素单列，不进入有限值误差统计。该规则不照搬历史 Ctrmm 的分类要求。
- 附件实际三接口各 1200 行（1000 精度、200 性能），合计 3600 行与 600 性能点。计划全覆盖，不按正文“200 条覆盖三个接口”缩减。
- 测试工程按接口显式分发并支持复数 alpha/beta、内联 null/NULLPTR、dtype、expect_result；保留原始 CSV 和 SHA，不在开发中静默改官方期望。
- 追加缺失的多错误叠加校验、padding 哨兵、alpha=0/k=0 不读取 A/B、beta=0 不读取原 C、C=nullptr 不写、非有限与有限混合、共轭差异、长 K 抵消等用例。源码、CSV、输入生成器与种子随测试工程提供。

## 性能标准与采样

任务性能边界为各主路径每点 NPU 平均耗时不高于提供 GPU baseline 的 1.25 倍（GPU/NPU≥0.8）。正式结论使用 msprof op 的设备 task duration，不用 GTest 总时间或 Host wall time 替代。一次接口调用包含 pack/compute/epilogue 等多个 kernel 时纳入该调用必要设备工作的完整统计，不只挑主 GEMM kernel。

每次预热/采样前恢复初始 C，恢复与计算同 stream 有序，恢复 DMA 不计入算子时间；记录每次有效调用及状态。每批5预热+64正式，保留全部样本；用户要求每点至少3个独立批次的GPU/NPU均>0.85作为内部稳定性目标，官方≥0.8门槛单列。不得选择最好窗口或丢弃失败批次。核实 msprof 实际 warmup 行为，不能仅据工具名字假定已完成预热。

附件脚本解析 GTest 整数 ms，该结果包含测试开销，不能作为正式 task duration；需要补充 profiler 解析与逐次表，保留原脚本来源。GPU baseline 的型号说明存在 A100/P102-100 冲突，固定随附 CSV，不改值，按用户确认直接使用附件表作性能比较，不宣称已独立核验GPU型号。

## 兼容性与交付

SgemmEx 签名变更影响 arch35 的实现、测试 wrapper、README 示例及外部调用者，提交前搜索全仓并同步适配。设计 PR 与实现 PR 分开；最终交付官方模板自测报告、完整精度/性能逐例数据与截图、测试步骤、代码和正式 CSV。代码 PR 需当前 SHA 的 CI，编译、设备测试、人工评审、合入状态分别记录。

## 需评审澄清的附件不一致

| 项目 | 已确定执行口径 |
| --- | --- |
| 用例规模 | 全量3600条，含600性能条目 |
| 有效采样 | msprof，每批5预热+64正式，至少3批逐例检查内部>0.85目标 |
| GPU来源 | 原附件CSV，不修改数值，不虚称独立核验型号 |
| dtype错误码 | 非法枚举INVALID_VALUE；合法但不支持NOT_SUPPORTED；原CSV不改 |
| 支持范围 | §3.5和官方用例涉及的FP32/complex64组合，不扩展额外混合精度 |
| 精度最大误差 | 按原标准采用max(0.01,32*ULP(g_low))，记录golden锚点及动态上限 |

以上为本次用户确认执行口径；设计评审与设备验收仍需实际完成。

## 实施里程碑

1. 固定上游和附件、提交设计、明确冲突与签名兼容范围。
2. 接入 arch22 公共工程，先完成两个 FP32 接口及边界行为，验证 golden 与测试工程。
3. 完成 complex64 正确性、共轭/复数标量与 EXACT 路径，再做拆分与性能优化。
4. 在 910B3/CANN9.1.0 全量自验并逐点性能验证；处理 arch35 编译回归、当前提交 CI 和自测报告。

时间取决于实机、评审澄清及复杂路径性能，设计阶段不承诺已经达标或具体测试完成时间。
