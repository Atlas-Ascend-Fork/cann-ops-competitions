# 需求背景（required）

## 需求来源

本设计文档对应 2026 年 11 月社区任务 `aclblasSgemm` / `aclblasCgemm` / `aclblasHgemm`（Atlas A2/A3 系列产品）。任务要求在昇腾 NPU 上基于 Ascend C 编程语言、依托 ops-blas 开源仓工程框架，以 kernel 直调方式实现三个通用矩阵乘接口，完成设计、开发与测试全流程。验收通过后实现代码合入 `blas/gemm/arch22/`，测试代码合入 `test/gemm/arch22/`，接口声明使用 `include/cann_ops_blas.h`。CANN 版本 9.1.0。

依据材料：任务书 `aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md`、配套测试用例目录（`sgemm/`、`cgemm/`、`hgemm/` 的 CSV、GPU 基线、生成与校验脚本、测试指导 README）。

## 背景介绍

### 算子功能

三个接口均为 BLAS Level-3 矩阵乘，统一计算公式：

```text
C = alpha * op(A) * op(B) + beta * C
```

| 接口 | 数据精度 | 转置语义 |
| --- | --- | --- |
| `aclblasSgemm` | FLOAT32 | `OP_C` 与 `OP_T` 等价，不做共轭 |
| `aclblasCgemm` | COMPLEX64（实部/虚部各 FLOAT32） | `OP_C` 为共轭转置，`OP_T` 为非共轭转置，二者结果不同 |
| `aclblasHgemm` | FLOAT16（IEEE binary16） | `OP_C` 与 `OP_T` 等价，不做共轭 |

矩阵按列主序存储：`op(A)` 为 m×k，`op(B)` 为 k×n，`C` 为 m×n。`beta=0` 时不读取 C 旧值；`m=0` 或 `n=0` 为合法 quick return；`k=0` 或 `alpha=0` 时仅执行 `C = beta * C`，不引用 A/B。C 与 A/B 存储不得重叠。

### 精度口径

- S、C 的精度 golden 由 Netlib `sgemm` / `cgemm` 在 CPU 侧生成；Cgemm 实部、虚部分别比对。
- H 的 golden 由测试工程按 FLOAT32 计算后收窄到 FLOAT16。
- 判定采用混合容差：`|actual - golden| <= atol + rtol * |golden|`，S/C 按 FLOAT32 容差，H 按最终 FLOAT16 输出误差判定。

### 仓内现状与复用

| 层次 | 关键文件 | 现状与本任务动作 |
| --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` | `aclblasSgemm`（L575）、`aclblasCgemm`（L579）、`aclblasHgemm`（L589）声明齐备，无需修改 |
| Sgemm 实现 | `blas/gemm/arch22/gemm_host.cpp` / `gemm_kernel.cpp` | 已有 arch22 FLOAT32 实现，直接复用 |
| Cgemm 实现 | `blas/gemm/arch22/cgemm_host.cpp` / `cgemm_kernel.cpp` | 已有 arch22 COMPLEX64 实现，按任务书回归验证 |
| Hgemm 实现 | `blas/gemm/arch22/` | 新增 arch22 FP16 实现及配套测试 |
| 测试框架 | `test/gemm/arch22/`、`test/gemm/cgemm/arch22/`、`test/gemm/hgemm/arch35/` | 复用 GTest + CSV 结构，补充 `test/gemm/hgemm/arch22/` |

# 需求分析（required）

## 需求描述

在 `blas/gemm/arch22/` 完成 `aclblasSgemm`、`aclblasCgemm`、`aclblasHgemm` 的 arch22 版本，覆盖 `transa/transb` 全枚举、列主序与前导维（含 padding）、`m/n/k` 全尺寸（含 0 与 4095/4096）、标量特殊值、`beta=0` 不读旧 C、`alpha=0/k=0` 只做 `beta*C`、边界与负向用例。精度满足任务书 §3.2 混合容差标准，性能满足 §3.3 的 warmup 与有效采样要求。

## 需求拆解

1. **接口与校验闭环**：host 按任务书顺序完成 handle、枚举、维度、前导维、标量指针与数据指针校验，非法输入返回对应状态码且不发起 kernel。
2. **矩阵乘计算**：S/C 复用仓内已有 arch22 路径；H 新增 arch22 FP16 路径，FP16 输入经 FLOAT32 workspace 参与矩阵乘，最终按 FP16 写回。
3. **标量与退化语义**：`beta=0` 不读 C；`m/n=0` quick return；`k=0` 或 `alpha=0` 仅缩放 C；`OP_C` 对 C 为共轭转置、对 S/H 等价转置。
4. **测试覆盖**：精度全量 1000 条、性能子集 200 条、负向与特殊值用例，按算子分摊并保留 alpha/beta 性能判定键。
5. **交付材料**：按任务书 §4 输出 `task_submission/` 与算子 README。

# 详细设计（required）

## 算子分析

### 数学公式

```text
C = alpha * op(A) * op(B) + beta * C
```

S/H 为实数乘加；Cgemm 为复数乘加：

```text
(a + bi)(c + di) = (ac - bd) + (ad + bc)i
```

### 参数语义

| 参数 | 类型与约束 |
| --- | --- |
| `handle` | `aclblasHandle_t`，nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `transa/transb` | `ACLBLAS_OP_N/T/C`；非法值返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m/n/k` | 非负整数；`m/n=0` 合法 quick return |
| `alpha/beta` | Host 标量指针；`beta=0` 时 C 旧值不被读取 |
| `A/B/C` | Device 列主序矩阵；`k>0` 且 `alpha≠0` 时 A/B 不可为空；C 不可为空 |
| `lda/ldb/ldc` | 满足任务书前导维下界；padding 列不参与计算 |

## 算子实现

### Sgemm 与 Cgemm

复用 `blas/gemm/arch22/` 已有实现：Sgemm 采用 host 分块 + Cube 实数矩阵乘 + AIV epilogue；Cgemm 采用 AIV 拆实/虚平面、四次实数矩阵乘、AIV 复数组合回写。本任务回归验证，不重复造轮子。

### Hgemm 新增 arch22 实现

Hgemm 采用 FP16 → FLOAT32 复用 Sgemm → FP16 写回的三段式实现，避免新增第二套 Cube 矩阵乘：

```text
A_half,B_half --[AIV cast]--> A_f32,B_f32
A_f32,B_f32 --[复用 aclblasSgemm]--> product_f32
product_f32,C_half --[AIV epilogue]--> C_half = alpha*product + beta*C_old
```

- `h2f`：AIV kernel 将 A/B 从 FP16 逐列搬运到 FLOAT32 workspace，保持列主序与前导维。
- `f2h`：AIV kernel 逐列完成 `alpha*product + beta*C_old`，再按 FP16 舍入写回。
- 复用 `aclblasSgemm` 时通过子句柄将 Sgemm workspace 限制在独立 scratch 切片，避免覆盖仍存活的 FP32 中间量。
- `k=0` 或 `alpha=0` 时只执行 `f2h` 的 beta 缩放路径，不分配 A/B/product。

### 边界与错误处理

- `handle` 空 → `HANDLE_IS_NULLPTR`；非法枚举 → `INVALID_ENUM`；负维度、非法前导维、空标量、空指针 → `INVALID_VALUE`。
- `m/n=0` 先于指针校验返回成功；`k=0/alpha=0` 时 A/B 不引用。
- 输出只写 C 的 m×n 有效区，`ldc` padding 列不触碰。

## 支持硬件

| 支持的芯片版本 | 本设计说明 |
| --- | --- |
| Atlas 800T A2（910B3，arch22） | 精度/性能验收设备 |
| Atlas 800I A2（arch22） | 同 arch22 数据流，随产品包构建回归 |
| Atlas 800I A3（arch22） | 同 arch22 数据流，自验后按目标包构建回归 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | S/C 按 FLOAT32 混合容差判定，C 实部/虚部分别比对；H 按 FP16 输出误差判定；Inf/NaN 单列匹配 | 任务书 §3.2、生态算子开源精度标准 |
| 性能标准 | 910B3，warmup 5 次 + 有效采样 ≥10 次取平均；正式判据为 `msprof op` kernel Task Duration，GTest `[PERF] avg_us` 仅作交叉验证 | 任务书 §3.3 |

## 自验用例覆盖

- 全算子覆盖 S/C/H；`transa/transb` 9 种组合全枚举；非法枚举负向。
- 尺寸扫描 0、1、质数、2 的幂及 ±1 至 4096；`m/n/k=0` 与退化组合。
- `alpha/beta` 特殊值、`alpha=0` 且 A/B 为 NaN、`beta=0` 且 C 为 NaN；`lda/ldb/ldc` 紧凑与 padding。
- Cgemm 的 `OP_C` 用非对称复数矩阵验证共轭转置；S/H 的 `OP_C` 与 `OP_T` 结果一致。
- Inf/NaN、FP16 次正规/上下限值；性能子集显式保留 alpha/beta 列。

测试工程：`test/gemm/arch22/`（S）、`test/gemm/cgemm/arch22/`（C）、`test/gemm/hgemm/arch22/`（H，新增）。

## 兼容性分析

新增接口与新增目录，不修改既有接口行为；arch22 与 arch35 实现互不影响。接口签名与任务书给定参数序列一致，无需额外映射。

# 交付与合入

| 交付件 | 说明 |
| --- | --- |
| 设计文档 | 本文档，经 PR 评审合入 tasklist 对应目录 |
| 实现代码 | `blas/gemm/arch22/`（个人 fork 仓，邀 Ascend-CANN 为开发者） |
| 测试代码 | `test/gemm/arch22/`、`test/gemm/cgemm/arch22/`、`test/gemm/hgemm/arch22/`，readme 说明复现步骤 |
| 自测报告 | `task_submission/` 按任务书 §4 模板提交 7 项材料，含精度/性能对比与截图 |
| 算子 README | `blas/gemm/README.md` 增加 S/C/H 接口说明，产品支持表标注 Atlas A2/A3 支持 |
