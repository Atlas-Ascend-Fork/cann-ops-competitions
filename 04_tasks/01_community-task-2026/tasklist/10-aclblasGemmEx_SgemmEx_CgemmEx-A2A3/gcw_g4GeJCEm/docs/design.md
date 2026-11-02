# 需求背景（required）

## 需求来源

10月社区任务：aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx 算子开发（A2/A3）。依据任务书与随附三组官方测试 CSV（各 1200 行 = 1000 精度/边界 + 200 性能）制定本方案。

## 背景介绍与仓内现状

设计依据 ops-blas master（2026-10-28）全仓检索结果：

- `blas/gemm_ex/` 仅有 arch35 实现，README 产品支持表标注 Atlas A2/A3「不支持」；
- `aclblasGemmEx` 已有 19 参声明，与 cuBLAS 同构；`aclblasSgemmEx` 为 15 参强类型旧签名（缺三 dtype 参、多 algo 参）；`aclblasCgemmEx` 全仓零命中；
- `blas/gemm/arch22/` 已有经验证的 FP32 SGEMM 后端（统一 epilogue 处理 alpha/beta），可复用为实数计算层；
- arch22 设备代码不支持跨 TU 链接，kernel 与其 launcher 须在同一编译单元。

实现落 `blas/gemm_ex/arch22/`，测试落 `test/gemm_ex/`（三接口共用契约测试工程），公共声明统一进 `include/cann_ops_blas.h`，不新建产品私有平行 API。

# 需求分析（required）

## 需求描述

三接口计算列主序 `C = alpha * op(A) * op(B) + beta * C`。GemmEx/SgemmEx 主路径 FLOAT32；CgemmEx 主路径 COMPLEX64（实虚交错），alpha/beta 为复数标量。N/T/C 组合全覆盖，实数路径 C 与 T 等价，复数路径 C 为共轭转置。

| 接口 | 公共签名 | 工程处理 |
| --- | --- | --- |
| aclblasGemmEx | 19 参（含 computeType/algo） | arch22 新增分发，共享实数计算后端 |
| aclblasSgemmEx | 17 参（无 computeType/algo） | 修正 15 参旧声明，同步 arch35 定义、测试 wrapper、README 示例 |
| aclblasCgemmEx | 17 参（复数标量） | 新增公共声明与 arch22 复数实现 |

## 需求拆解与边界

1. 参数校验严格按任务书 §2.5 全序，no-op 短路置于全部校验之后。
2. 错误码两层规则与官方 CSV 期望一致：非法枚举值 → INVALID_VALUE；合法但不受支持的枚举 → NOT_SUPPORTED；trans 非法枚举 → INVALID_ENUM。
3. 支持集为三矩阵同 dtype：实数接口 `ACL_FLOAT`、复数接口 `ACL_COMPLEX64`。扩展混合精度组合（FP16/BF16/FP8）一律 NOT_SUPPORTED——材料包未生成扩展 dtype 正向用例，支持集外实现属无验收证据的范围扩张，本设计不做。
4. 边界行为：m/n=0 合法 no-op；k=0 或 alpha=0 不读 A/B，只做 `C = beta*C`；beta=0 且 C 为空指针时返回 SUCCESS 且全程不写 C。
5. 经 handle 绑定 stream 异步执行，workspace 走 handle 既有机制并做显式溢出检查。

# 详细设计（required）

## 算子分析

实数路径为 alpha 乘 K 维点积加 beta 乘原 C，FP32 输入不降精度。

复数路径采用实/虚平面化：打包阶段把交错复数展开为实部、虚部两个平面（共轭转置在打包时对虚部变号），以一次实数矩阵乘同时产出实/虚两个平面，再由合并阶段施加复数 alpha/beta 并交织写回。该结构 FLOPs 与复数乘法理论量一致，且不改变累加舍入序（与 Netlib cgemm 的 4 乘结构对齐）。

## 算子实现

- **实数路径**：完全复用已合入 master 的 arch22 SGEMM 后端（含其 alpha/beta epilogue 与 k=0/alpha=0 缩放路径），不修改该后端文件，对其零回归风险。
- **复数路径**：打包 → 一次实数矩阵乘 → 合并，三阶段均以连续访存为主、分块处理，UB 占用与 m/k 无关，避免跨步搬运带来的不确定性。
- **微形状**：小尺寸场景优先减少 launch 次数以摊薄固定开销，融合与回退的交叉点以 msprof 实测为准，阈值作可调旋钮。
- **构建注册**：`--ops` 保留 `blas/gemm/arch22/`、`blas/scal/arch22/` 与 `blas/gemm_ex/` 源。
- arch35 的 `aclblasSgemmEx` 同步为 17 参，dtype 限定 `ACL_FLOAT`，其余校验行为不变。

## 支持硬件与限制

| 产品 | 设计目标 | 验证要求 |
| --- | --- | --- |
| Atlas A2 (910B3) | arch22 | 功能/精度全量 + 性能实测 |
| Atlas A3 | arch22 | 功能/精度；性能以 910B3 为准 |
| Ascend 950 | arch35 既有支持 | 17 参签名变更的编译与调用点回归 |

限制：不支持非连续张量、广播、batch 维度（任务书明确不涉及）；扩展混合精度组合 NOT_SUPPORTED；未在 A3 实测前不宣称双端性能达标。

# 可维可测分析

## 精度标准与测试覆盖

- Golden 由 Netlib CBLAS 生成：实数 `cblas_sgemm`、复数 `cblas_cgemm`，随测试工程提供并记录版本指纹。
- 判定按生态标准：`|actual-golden| ≤ atol+rtol·|golden|`（atol=rtol=2⁻¹³），matched_ratio ≥ 0.99，max_abs_error ≤ max(1e-2, 32×ULP)；COMPLEX64 实部/虚部分别判定。
- Inf/NaN 按任务书 §3.2.5 等效规则单独判定，不入 ratio/max_abs 统计；因判定式与非有限值规则的组合需自定义校验器实现。
- alpha=0 用例做位精确比对；负向用例只比对返回码。官方三 CSV 全量执行、不改期望值，并以 SHA256 固定用例版本。

## 性能标准与采样

910B3 实机，msprof op 模式采集 task duration，warmup 后连续有效采样取算术平均；一次调用的全部 kernel 纳入统计。判据为逐例 NPU 均值 ≤ gpu_baseline.csv × 1.25。

## 兼容性与交付

1. SgemmEx 15→17 参属公共 ABI 变更：arch35 定义、测试 wrapper、README 原型与示例全量同步，950 侧回归范围即 arch35 sgemm_ex 测试套件。
2. 交付件：设计文档（本文）、测试工程（三二进制 + CSV + 可复现步骤 README）、自测报告（精度逐例 + 性能逐点 + msprof 截图）、个人仓邀请 Ascend-CANN。
3. 里程碑：设计评审合入 → FP32 双接口实机验证 → 复数路径实机验证 → 全量精度与性能 + 自测报告。
