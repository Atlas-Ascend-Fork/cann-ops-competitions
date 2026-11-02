# aclblasHgemm 算子设计文档（终稿 v1.0）

> 交付件 1，提交 cann-competitions 仓。结构对齐 design_template.md。
> 状态：待精度全绿后提交 PR。

# 需求背景

## 需求来源

10 月社区任务：在 Ascend 950PR（arch35，CANN 9.1.0）以 Ascend C 开发 `aclblasHgemm`，补齐 ops-blas 仓 Blas 能力。验收通过后 PR 合入。

## 背景介绍

对标 cuBLAS `cublasHgemm`：`C = α·op(A)·op(B) + β·C`，FLOAT16 列主序，op∈{N,T,C}。
现状态：`blas/gemm/arch35/` 已有 aclblasSgemm(FP32)/aclblasCgemm 模板栈。本算子在此基础上新增 fp16 路径。

# 需求分析

1. 接口层：`aclblasHalf` 类型 + `aclblasHgemm` 声明入 `cann_ops_blas.h`
2. host 层：校验层（§2.5 逐字）、tiling、双路径派发
3. kernel 层：panel 管线（大 case）+ 模板回退（小 case/通用 α/β）
4. 测试层：1200 条 CSV 用例（1000 精度 + 200 性能）、golden=fp32 参考舍入 fp16
5. 性能：单 case ≥0.4× GPU 且均值 ≥0.5×（H100 标杆）

# 详细设计

## 算子分析

C[i][j] = α × Σ op(A)[i][p]·op(B)[p][j] + β×C[i][j]，列主序。
累加中间精度 fp32（L0C），输出 fp16（Fixpipe F322F16 单次舍入）。

## 算子实现

### host 侧设计

**校验层**（任务书 §2.5 逐字对齐）：
- handle 判空 → 枚举 → 维度 → 前导维 → alpha/beta 判空（先于 m/n 短路）→ m/n 短路 → 指针判空（A/B 需 α≠0 前置）→ 分支派发

**双路径派发**：
- **panel 路径**（α=1 ∧ β=0 ∧ m≥256 ∧ n≥128 ∧ k≥256）：直接 fixpipe 输出，零后处理开销
- **模板路径**（其余全部）：te:: dtype 模板化 kernel + fp32 epilogue（workspace fp16 中间 + Cast/Muls/Add fp32 算术 + 单次 RN→fp16）

### kernel 侧设计

**Panel 管线**（大 case 高性能路径——kernel 实现结构源自同仓 MR !737（作者 gcw_fY1MdJ3B，commit e1e19ffe，CANN OSL 2.0），本提交做了 tiling 参数适配、host 派发条件与仓框架集成，详见 NOTICE 文件）：
- 每 core 处理一个 256×128 输出 tile（grid-stride + GROUP_M=8 L2 带状分组）
- k-panel 256 双缓冲 L1（A 128KB + B 64KB = 192KB/stage，L1 ≥448KB 实测支撑）
- INNER_K=64 步进 L0A/L0B，L0 双 stage 流水
- L0C fp32 跨 k-panel 累加，最终 Fixpipe F322F16 直出 fp16 C
- 关键 flag 编排：Wait 在循环顶（等上轮释放）、Set 在循环尾（释放本轮），GM→L1 装载与 L1→L0+Mmad 计算重叠

**模板路径**（小 case / 通用 α/β）：
- dtype-templated te:: tensor API（DT/DATA_C0 贯穿）
- L0C fp32 累加 → Fixpipe → workspace → fp32 epilogue kernel

**退化路径**：k=0 或 α=0 → C=β·C（memset 或 fp16 scale kernel）

### tilingKey 规划

无需 tilingKey：路径选择由 host 侧条件判断（非编译期分支），trans 组合为模板参数。

## 性能设计

| case | 耗时 | TFLOPS | 倍率 |
|---|---|---|---|
| 512³ NN | 0.010ms | 27.9 | 0.94× |
| 1024³ NN | 0.019ms | 114.3 | 0.80× |
| 2048³ TN | 0.060ms | 285.0 | 0.47× |
| 4096³ NT | 0.415ms | 331.2 | 0.44× |
| 8192³ NN | 3.19ms | 344.4 | 0.43× |
| 200 条均值 | — | — | **1.267** |

- 344T ≈ 估算理论峰值（378.5T）的 91%
- 关键发现：arch35 L1 实测 ≥448KB（hardware.h 原标注 32KB 为错误值），是 panel 管线可行性的前提

# 测试与验证

- 精度：1200 条 CSV 驱动 GTest（golden=fp32 参考→RN→fp16，混合容差 rtol/atol=2⁻¹⁰、ratio≥0.99、max_abs≤max(1e-1,32ULP)）
- 负向：40 条边界用例（校验顺序、空指针、非法枚举/维度/前导维）
- 性能：200 条 TC_PF 用例 vs H100 标杆（gpu_baseline.csv）

# 风险与已知限制

| 风险 | 对策 |
|---|---|
| 硬件 vs 软件 fp32 累加序差致边缘精度差 | L0C fp32 累加 + fp32 epilogue 已最大化精度；残余差异为架构固有 |
| 512×512×1024 唯一 perf 差 0.025 | 待 tile 适配调优 |
