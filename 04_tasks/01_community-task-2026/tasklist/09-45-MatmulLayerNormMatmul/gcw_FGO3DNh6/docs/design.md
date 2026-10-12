# MatmulLayerNormMatmul 算子设计文档

## 版本记录

| 版本 | 日期 | 作者团队 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-09-16 | `gcw_FGO3DNh6` | 首版设计：需求分析、融合方案选型、Host/Kernel 分层设计、片上资源预算、性能达标可行性论证、测试设计 |

**目标任务目录与提交路径**

```text
04_tasks/01_community-task-2026/tasklist/09-45-MatmulLayerNormMatmul/gcw_FGO3DNh6/docs/design.md
```

**目标代码仓与分册目录**（任务书《PR 申请合入》规定）

```text
https://gitcode.com/cann/catlass
  experimental/matmul/matmul_layer_norm_matmul/    # 算子样例目录
  tests/optest/                                    # optest 测试框架接入
```

**说明**：本文中的 TileShape、流水深度、`BM` 等数值均为**设计候选**，不是已验证的最优值；精度与性能结论一律以 950PR 实测为准，本文不将估算当作实测结果。

---

# 一、需求背景（required）

## 1.1 需求来源

本需求来自 CANN 2026 年 **9 月社区任务「MatmulLayerNormMatmul 算子开发」**，任务编号 `09-45-MatmulLayerNormMatmul`。

| 项目 | 内容 |
| --- | --- |
| 任务书 | `9月社区任务-MatmulLayerNormMatmul算子开发/MatmulLayerNormMatmul_task_doc.md` |
| 任务测试集 | `MatmulLayerNormMatmul_测试集.csv`，116 条 `(M0, K0, N0, M1)` 组合，附 Ascend 950PR 实测标杆时延 |
| 适配硬件 | Ascend 950（性能验收使用 Ascend 950PR） |
| 开发语言 | Ascend C，基于 CATLASS 模板库选型 |
| 开源仓 | <https://gitcode.com/cann/catlass> |
| 算子目录 | `experimental/matmul/matmul_layer_norm_matmul/` |
| 设计文档模板 | <https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md> |
| 参考合入 PR | <https://gitcode.com/cann/catlass/pull/678> |

## 1.2 背景介绍

### 1.2.1 算子定位与数学定义

`Matmul → LayerNorm → Matmul` 是 Transformer 类模型中反复出现的计算片段：第一次投影得到隐层激活，对隐层做**行级**归一化，再投影回输出维度。三个阶段的数学定义如下。

阶段一，第一个 Matmul（`A0` 为 `(M0, K0)`，`B0` 为 `(K0, N0)`）：

```text
C0[m, n] = sum_k A0[m, k] * B0[k, n]，  C0 形状 (M0, N0)
```

阶段二，LayerNorm（对 `C0` 的每一行独立计算，归一化维为 `N0`，`eps = 1e-6` 固定）：

```text
mean[m]       = (1 / N0) * sum_n C0[m, n]
var[m]        = (1 / N0) * sum_n (C0[m, n] - mean[m])^2
C0_norm[m, n] = (C0[m, n] - mean[m]) / sqrt(var[m] + eps) * gamma[n] + beta[n]
```

阶段三，第二个 Matmul（`B1` 为 `(N0, M1)`）：

```text
C1[m, p] = sum_n C0_norm[m, n] * B1[n, p]，  C1 形状 (M0, M1)，唯一对外输出
```

### 1.2.2 小算子拼接基线的开销结构

性能标杆是小算子拼接方案：

```python
c0       = torch.mm(a0, b0)                                        # launch 1
c0_norm  = torch.nn.functional.layer_norm(c0, (n0,), gamma, beta, 1e-6)  # launch 2
c1       = torch.mm(c0_norm, b1)                                   # launch 3
```

该链路的开销可分为三类，且随 shape 变化此消彼长：

| 开销类别 | 来源 | 随 shape 的变化趋势 |
| --- | --- | --- |
| 固定 launch 开销 | 3 次 kernel launch 的调度、下发与首尾 flush | 小 shape 时占比极高 |
| 中间激活的 HBM 往返 | `C0` 被写 1 次、读 1 次（LayerNorm），`C0_norm` 被写 1 次、读 1 次（Matmul1） | 正比于 `M0 * N0`，大 `N0` 时占比高 |
| LayerNorm 的纯带宽时间 | LayerNorm 自身无 Cube 计算，是一段暴露在关键路径上的搬运+归约 | 正比于 `M0 * N0`，无法与 Cube 重叠 |

以测试集中的典型 shape 为例，`C0` 的规模差异极大：

| case | `(M0, K0, N0, M1)` | `C0` 大小（FP16） | 说明 |
| --- | --- | --- | --- |
| idx 1 | `(128, 768, 2048, 768)` | 512 KiB | 三次 launch 的固定开销主导 |
| idx 81 | `(1024, 4096, 4096, 4096)` | 8 MiB | 两类开销相当 |
| idx 116 | `(2048, 8192, 8192, 4096)` | 32 MiB | 中间激活带宽主导 |

**融合算子的收益来源因此有三条**：① 3 次 launch 合并为 1 次；② LayerNorm 由 AIV 承担，可与 AIC 的 Cube 计算在行块之间重叠，把这段纯带宽负载从关键路径上摘掉；③ `C0`/`C0_norm` 以行块为粒度在片上闭环生产与消费，不再走 HBM 往返。

### 1.2.3 9 月任务书相对 7 月版的增量要求

社区 2026 年 7 月已发布过同名任务（`07-04-MatmulLayerNormMatmul`）。本设计在动笔前逐条比对了 7 月与 9 月两版任务书，**确认任务测试集完全相同**（116 例、同一份标杆时延，逐字节 diff 无差异），9 月的实质变化只有两项：

| 增量项 | 7 月版 | 9 月版 | 对本设计的影响 |
| --- | --- | --- | --- |
| ATK 泛化测试 | 未要求 | **要求泛化 ≥ 200 例精度通过**，可用 `catlass-atk-support` skill 生成 | 第 4.2 节新增 ATK 四件套交付与 240 例泛化矩阵设计 |
| PR 提交目录 | 未规定 | **明确要求** `experimental/matmul/${op_name}/` + `tests/optest/` 文件树，并列出「不希望出现的内容」黑名单 | 第 3.2.3 节的文件清单严格按任务书目录树组织；第 4.4 节将黑名单登记为提交门禁 |
| 测试集口径 | `./self_test_case/matmul_layer_norm_matmul/` | `./test_case/MatmulLayerNormMatmul_测试集.csv` | 内容一致，按 9 月路径为准 |

即：9 月任务**没有放宽**任何指标，只增加了交付件要求。本设计按 9 月版（更严）执行。同族 7 月已提交的三份设计文档（`2301_79503900`、`dududu121`、`xfx321`）可作为格式与深度基线参考。

### 1.2.4 目标仓库现状核查

设计动笔前对 CATLASS 主干做了一轮组件级核查，结论直接决定本设计的取舍。核查对象为 catlass master 稀疏克隆。

| 核查项 | 结论 | 证据 |
| --- | --- | --- |
| `Arch::Ascend950` 片上资源 | L1 512 KiB、UB 248 KiB、L0A/L0B 64 KiB、L0C 256 KiB、BIAS 4 KiB、FIXBUF 16 KiB | `include/catlass/arch/arch.hpp` 中 `struct Ascend950` |
| 编译架构宏 | `CATLASS_ARCH=3510` | 根 `CMakeLists.txt` 默认 2201；`examples/CMakeLists.txt` 将其转为 `--npu-arch=dav-${CATLASS_ARCH}` |
| **LayerNorm / RMSNorm 实现** | **不存在**。全仓库 `LayerNorm` / `layer_norm` / `layernorm` / `RMSNorm` 关键字 0 命中 | `include/` 下所有 `norm` 命中均为 `SetMaskNorm`、`LoadDist::DIST_NORM` 等无关词 |
| **行方向 reduce 的通用 tile** | **不存在**。`include/**/*reduce*` 无文件 | 现有行归约均内联在 block epilogue 内，直接用微指令 `ReduceSum`/`ReduceMax` 或 `WholeReduceSum` |
| L0C → UB 通路 | 存在，`CopyL0CToUBTla`，含 `NO_SPLIT` / `SPLIT_M`（`dualDstCtl=1`）/ `SPLIT_N`（`dualDstCtl=2`）三种模式 | `include/catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp`、`copy_l0c_to_dst.hpp` |
| Fixpipe 直出 FP16 能力 | 能力存在（`CopyL0CToDstQuantMode<Ascend950, float, half, NO_QUANT>` → `QuantMode_t::F322F16`），但**仓库现有样例的 16 位落 UB 只用了 bf16，无 half 先例** | `copy_l0c_to_dst.hpp`；`examples/62` 用 bf16 落 UB |
| UB → L1 通路 | 存在，`CopyUb2L1Tla`，支持 RowMajor → zN，仅 `CATLASS_ARCH == 3510` 编译 | `include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp`，FA 样例已用 |
| 跨核同步原语 | `CrossCoreSetFlag` / `CrossCoreWaitFlag` / `CrossCoreFlagWithReverse` / `CrossCoreSetFlagWithReverse` / `MAX_REVERSE_DEPTH = 15` / `FFTS_MAX_FLAG = 7` 均存在 | `include/catlass/arch/cross_core_sync.hpp` |
| AIC/AIV 全核栅栏 | `AscendC::SyncAll<false>()` 有先例，但需 AIC/AIV 两侧严格配对 | `include/catlass/gemm/kernel/matrix_inverse.hpp` 等多处 |
| MIX kernel 的 AIC/AIV 配对 | `__mix__(1, 2)`（1 AIC : 2 AIV）；AIV 侧用 `GetBlockIdx() / GetSubBlockNum()` 或 `/ GetTaskRation()` 映射回同一 AIC | `examples/83_ascend950_hstu_infer`、`include/catlass/gemm/kernel/matmul_mix_fixpipe_opti.hpp` |
| example 注册宏 | `catlass_example_add_executable(NAME OPTYPE ...)`，`OPTYPE ∈ {cube, mix}`；`experimental/` 下同样使用该宏 | `examples/CMakeLists.txt` |
| **`experimental/` 的构建接入** | 根 `CMakeLists.txt` **未** `add_subdirectory(experimental)`，`experimental/` 下也无顶层 CMakeLists | 见第 4.4 节风险 R-07 |
| 最近的同族参考 | `experimental/matmul/` 下已有 `78_matrix_inverse`、`ascend950_fp4_mx_quant_matmul`、`ascend950_fp8_e4m3_quant_matmul` | `git ls-tree` 结果 |

**核查的直接结论**：本算子的核心新增点是 **LayerNorm 的行归约与仿射 epilogue**，以及**让第二个 Matmul 的 A 操作数来自 L1（而非 GM）的通路**。CATLASS 现成能力可覆盖 Fixpipe、UB→L1、跨核同步与 Matmul 本体，遵循 CATLASS「最大化复用、最小化创新」原则，新增面收敛在 epilogue 及其行归约 tile 上。

---

# 二、需求分析（required）

## 2.1 外部契约

### 2.1.1 参数表

| 参数 | 方向 | dtype | 逻辑 shape | 物理布局 | 元素 stride |
| --- | --- | --- | --- | --- | --- |
| `A0` | 输入 | FP16 | `(M0, K0)` | RowMajor | `(K0, 1)` |
| `B0` | 输入 | FP16 | `(K0, N0)` | ColumnMajor | `(1, K0)` |
| `B1` | 输入 | FP16 | `(N0, M1)` | ColumnMajor | `(1, N0)` |
| `gamma` | 输入 | FP32 | `(N0,)` | 连续 | `(1)` |
| `beta` | 输入 | FP32 | `(N0,)` | 连续 | `(1)` |
| `C1` | **唯一输出** | FP16 | `(M0, M1)` | RowMajor | `(M1, 1)` |
| `eps` | 编译期常量 | FP32 | 标量 | — | 固定 `1e-6`，不开放为入参 |

`B0` 的元素地址为 `base + k + n * K0`，`B1` 的元素地址为 `base + n + p * N0`。Torch 侧可用 `torch.empty((N0, K0), dtype=torch.float16).t()` 构造列主 `B0`。

**布局校验要点**：ColumnMajor 的 `B0`/`B1` 在 Torch 视角表现为具有列主 stride 的二维 Tensor，**不能**等同于「必须 contiguous」。Host 侧必须校验逻辑 shape 与实际 stride，禁止用无条件 `.contiguous()` 把列主数据改成行主后仍沿用原地址公式——那会静默产生错误结果。对退化维度（如 `K0` 或 `N0` 为 1）按实际地址等价性检查布局，避免仅因 singleton stride 不同而拒绝合法输入。

### 2.1.2 样例入口（CATLASS example 形态）

```cpp
// experimental/matmul/matmul_layer_norm_matmul/matmul_layer_norm_matmul.cpp
struct MatmulLayerNormMatmulParams {
    uint32_t m0;   // A0 行数
    uint32_t k0;   // Matmul0 归约维
    uint32_t n0;   // LayerNorm 归一化维，同时是 Matmul1 的归约维
    uint32_t m1;   // Matmul1 输出列数
    uint8_t* a0;       // FP16, (M0, K0), RowMajor
    uint8_t* b0;       // FP16, (K0, N0), ColumnMajor
    uint8_t* b1;       // FP16, (N0, M1), ColumnMajor
    uint8_t* gamma;    // FP32, (N0,)
    uint8_t* beta;     // FP32, (N0,)
    uint8_t* c1;       // FP16, (M0, M1), RowMajor —— 唯一对外输出
    uint8_t* workspace;
};
```

optest / PyTorch 侧（对齐 `.agents/skills/catlass-example-to-pytest`）：

```python
c1 = torch_catlass.matmul_layer_norm_matmul(a0, b0, b1, gamma, beta)
# a0: (M0, K0) fp16 ; b0: (K0, N0) fp16 列主 ; b1: (N0, M1) fp16 列主
# gamma/beta: (N0,) fp32 ; 返回 (M0, M1) fp16
```

`mean` / `var` / `rstd` 等中间量全部由算子内部 workspace 承载，不出现在任何对外接口上。

## 2.2 需求拆解

| 编号 | 子需求 | 设计响应 | 验证方式 |
| --- | --- | --- | --- |
| R1 | 单次 kernel launch 完成三阶段 | 一个 Ascend 950 MIX kernel（`__mix__(1,2)`），内部编排 `Matmul0 → LayerNorm → Matmul1` | msprof kernel 列表、代码审查 |
| R2 | `C0`/`C0_norm` 无中间显存读写 | 主路径 `C0` 驻留片上（`L0C → UB → L1`），Matmul1 的 A 直接取自 L1；大 `N0` 档退化为 L2 驻留 slab | 源码无对应 GM 地址 + 内存规划无对应分配 + profiling 访存路径 |
| R3 | `mean`/`var` 由内部 workspace 管理、不对外暴露 | stats 驻留 UB scratch；对外接口只返回 `C1` | 接口审查、pytest 断言输出唯一 |
| R4 | dtype 契约固定 | `A0/B0/B1/C1` FP16，`gamma/beta` FP32，FP32 累加 | 负向用例、pytest dtype 断言 |
| R5 | shape 与布局正确 | 校验四组维度联动关系及列主 stride | shape/stride 正反向用例 |
| R6 | LayerNorm 沿 `N0` 全行 | 每行块完整持有 `N0`，FP32 两遍中心化统计 | golden 对比、常数行/零方差用例 |
| R7 | 任务测试集 116 例全过 | 按 CSV 原始 idx 参数化 | optest 全量执行 + 汇总日志 |
| R8 | ATK 泛化 ≥ 200 例 | 独立构造 240 例泛化矩阵（见 4.2.2） | ATK 四件套 + manifest |
| R9 | 平均性能 > 1.1× | 片上复用、AIC/AIV 重叠、按 `N0` 分档 | `msprof op` 逐 case 数据 + 平均比值 |
| R10 | 精度达标 | 与拼接基线逐段对齐舍入边界（见 3.1.3） | MERE/MARE、NaN/Inf 检查 |
| R11 | 交付件合规 | example + optest 接入 + ATK + README + 自验证报告 | PR checklist、clean build |

## 2.3 验收边界

### 2.3.1 测试集 shape 域

任务测试集共 116 例，取值域与分布如下（已实测统计）：

| 维度 | 取值 | 各值出现次数 |
| --- | --- | --- |
| `M0` | 128 / 512 / 1024 / 2048 | 30 / 30 / 30 / 26 |
| `K0` | 768 / 2048 / 4096 / 8192 | 36 / 36 / 32 / 12 |
| `N0` | 2048 / 3072 / 4096 / 8192 | 36 / 35 / 33 / 12 |
| `M1` | 768 / 2048 / 4096 | 39 / 39 / 38 |

该集合**不是**上述取值域的完整笛卡尔积，验收按 CSV 逐行执行。全部 `K0`/`N0`/`M1` 均为 256 的整数倍，`M0 ∈ {128,512,1024,2048}` 均为 128 的整数倍，对 Cube 的 16 元素分形与 FP16 的 32 B 对齐天然友好；但实现仍须覆盖非对齐尾块，**不能把测试集的对齐特征写成对外接口限制**。

### 2.3.2 性能验收的第一性口径

任务书表述为「算子整体性能需达成 1.1 倍小算子拼接性能（即平均的标杆时延 / 测试时延 > 1.1）」。该表述存在两种可解析的口径，**两者的优化优先级完全不同**，必须先行澄清：

| 口径 | 定义 | 权重分布 |
| --- | --- | --- |
| A：逐例比值的平均 | `S_avg = (1/116) * Σ_i (T_ref,i / T_fused,i) > 1.1` | 116 例等权。小 shape 因数量多（`M0=128` 有 30 例）且提升空间大，对指标影响大 |
| B：平均时延之比 | `mean(T_ref) / mean(T_fused) = Σ T_ref,i / Σ T_fused,i > 1.1` | 大 shape 主导：`N0=8192` 的 12 例占标杆总时长 **38.6%**，`M0=2048` 的 26 例占 **45.0%** |

本设计**同时满足两种口径**作为自验目标，并在自验报告中原样给出逐例 `S_i`、`S_avg`、`Σ T_ref / Σ T_fused`、P50 与最差 case，不用辅助指标替代主指标。该口径差异已登记为待确认项（见 4.4 T-01）。

### 2.3.3 标杆时延基线

| 指标 | 数值 |
| --- | --- |
| 116 例标杆时延合计 | 15395.6 µs |
| 按口径 B 达标的测试时延上限 | 13996.0 µs |
| 标杆时延算术平均 | 132.721 µs |
| 最小 / 最大标杆时延 | 19.005 µs（idx 1）/ 1188.58 µs（idx 116） |

## 2.4 性能达标可行性分析

这一节回答评审最关心的问题：**1.1× 靠什么拿到，余量有多大，哪一档是binding constraint。** 下面的估算全部基于测试集给出的标杆时延，不引入任何猜测的性能数字。

### 2.4.1 标杆自身的效率分布

按 `FLOPs = 2*M0*K0*N0 + 2*M0*N0*M1` 折算 116 例标杆的等效算力：

| 分位 | 等效算力 |
| --- | --- |
| 最低（idx 1） | 42.4 TFLOPS |
| P25 | 135.3 TFLOPS |
| 中位 | 235.9 TFLOPS |
| P75 | 282.0 TFLOPS |
| 最高（idx 99） | 398.9 TFLOPS |

按 `M0` 分组的视角：

| 分组 | 例数 | 等效算力 P50 | 最低 | 最高 | 标杆时长合计 | 占总时长 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0=128` | 30 | 96.5 | 42.4 | 154.7 | 1458.7 µs | 9.5% |
| `M0=512` | 30 | 232.6 | 111.0 | 347.7 | 2497.1 µs | 16.2% |
| `M0=1024` | 30 | 271.1 | 164.5 | 366.1 | 4516.8 µs | 29.3% |
| `M0=2048` | 26 | 294.3 | 116.7 | 398.9 | 6923.0 µs | 45.0% |

**关键观察一**：`M0=128` 档等效算力仅 42–155 TFLOPS，远低于同硬件可达的 ~399 TFLOPS，说明这些 case 由 launch 与拖尾开销主导，融合后提升空间最大。

**关键观察二**：大 shape 档（`M0=2048`）已达到 294–399 TFLOPS，即基线在 Cube 上**已经接近跑满**。这意味着融合算子在大 shape 上**几乎没有 Cube 效率余量**，收益只能来自「移除 LayerNorm 的独立 kernel、移除中间激活 HBM 往返、合并 launch」。

### 2.4.2 余量分解：融合能拿到的绝对上限

若融合算子把两个 GEMM 都跑到基线自身观测到的最高效率（398.9 TFLOPS），且 LayerNorm 完全被隐藏、launch 开销为零，则：

| 分组 | 标杆合计 | 该理想模型下合计 | 比值（绝对上限） |
| --- | ---: | ---: | ---: |
| `M0=128` | 1458.7 µs | 410.8 µs | 3.551 |
| `M0=512` | 2497.1 µs | 1643.3 µs | 1.520 |
| `M0=1024` | 4516.8 µs | 3286.6 µs | 1.374 |
| `M0=2048` | 6923.0 µs | 5507.3 µs | 1.257 |
| **全量** | **15395.6 µs** | **10848.1 µs** | **1.419** |

**结论**：即使做到「零开销 + 全程打满历史最高 Cube 效率」，全量上限也只有 **1.419×**。1.1× 的目标在 1.419× 的天花板下并非唾手可得，且 `M0=2048` 档的余量只有 1.257×，是真正的 binding constraint。

### 2.4.3 只靠「省显存流量」够不够——不够

融合消除的中间显存流量为 `8 * M0 * N0` 字节（`C0` 写+读、`C0_norm` 写+读）。按 1.5 TB/s 的 HBM 有效带宽折算（该带宽为本节估算的假设输入，仅用于量级判断，非实测硬件指标）：

| case | 标杆时延 | 消除流量 | 折算节省 | 仅此项的比值 |
| --- | ---: | ---: | ---: | ---: |
| idx 1 `(128,768,2048,768)` | 19.005 µs | 2.00 MiB | 1.40 µs | 1.079 |
| idx 54 `(512,4096,4096,4096)` | 121.857 µs | 16.00 MiB | 11.18 µs | 1.101 |
| idx 81 `(1024,4096,4096,4096)` | 216.301 µs | 32.00 MiB | 22.37 µs | 1.115 |
| idx 116 `(2048,8192,8192,4096)` | 1188.580 µs | 128.00 MiB | 89.48 µs | 1.081 |

**结论**：仅消除中间显存读写，绝大多数 case 只能到 1.08–1.12×，整体压在 1.1× 附近。**因此本设计必须额外拿到两件事**：① LayerNorm 的独立 kernel 时间被彻底移除并与 Cube 重叠；② 融合后两个 GEMM 的 Cube 效率不劣化。

### 2.4.4 主导模型：移除 LayerNorm kernel + 合并 launch

更贴近本设计目标的模型：融合算子相对基线**仅**移除「LayerNorm 独立 kernel 的 3 次 `C0` HBM 往返」与「2 次 kernel launch」，两个 GEMM 的 Cube 效率与基线持平（不减分也不加分）。取单次 launch 固定开销 `L`：

| `L` | 口径 A：逐例比值平均 | 口径 B：时延之比 | 逐例最低 | 逐例最高 | 低于 1.1 的例数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 µs | 1.2020 | 1.1437 | 1.052 | 1.518 | 12 |
| 3.0 µs | 1.2516 | 1.1638 | 1.065 | 1.590 | 11 |
| 4.0 µs | 1.3086 | 1.1846 | 1.067 | 1.909 | 9 |

取 `L = 3.0 µs` 时，融合后合计约 **13229 µs**，相对口径 B 的上限 13996 µs 有 **5.8% 的裕度**。

**但必须同时看到风险面**：在该模型下，逐例低于 1.1 的 case 集中在 `N0=8192` 档，其标杆时长占比 **37–39%**：

| case | `(M0,K0,N0,M1)` | 标杆时延 | 模型节省 | 模型比值 |
| --- | --- | ---: | ---: | ---: |
| idx 107 | `(128, 8192, 8192, 4096)` | 166.572 µs | 10.194 µs | 1.0652 |
| idx 113 | `(1024, 8192, 8192, 4096)` | 624.156 µs | 39.554 µs | 1.0677 |
| idx 116 | `(2048, 8192, 8192, 4096)` | 1188.580 µs | 73.109 µs | 1.0655 |

### 2.4.5 由可行性分析导出的三条设计约束

上述估算给出三条**hard design constraint**，直接决定第 3 章的技术选型：

1. **`N0=8192` 档不得为片上驻留牺牲 Cube 效率。** 该档 12 例贡献 38.6% 的标杆时长，而其模型比值仅 1.065–1.093；任何 Cube 效率损失都会直接击穿整体指标。因此该档禁止采用「把 `BM` 压到 16 以换取整行 L1 驻留」的方案（见 3.2.1 的复用度推导），改用 L2 驻留 slab。
2. **LayerNorm 必须与 Cube 真正重叠，不能串行。** 若 LayerNorm 在融合 kernel 内仍串行执行，本模型的收益退化为 2.4.3 的水平（≈1.08–1.12×），不达标。
3. **小 shape 档不必过度优化。** 9 例标杆 < 30 µs 的 case 合计仅 231.8 µs（1.5%），但按口径 A 每例等权；在 `L=3 µs` 假设下仅靠 launch 合并即可拿到 1.25–1.59×。这里的关键是**不要引入额外的固定开销**（多余的 workspace 初始化、多余的同步、过大的核启动数），而不是追求极限。

---

# 三、详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式与依赖方向

```text
Phase 0  C0[m, n]        = Σ_k A0[m, k] * B0[k, n]
Phase 1  mean[m]         = (1/N0) * Σ_n C0[m, n]
         var[m]          = (1/N0) * Σ_n (C0[m, n] - mean[m])^2
         rstd[m]         = 1 / sqrt(max(var[m], 0) + 1e-6)
Phase 2  C0_norm[m, n]   = (C0[m, n] - mean[m]) * rstd[m] * gamma[n] + beta[n]
Phase 3  C1[m, p]        = Σ_n C0_norm[m, n] * B1[n, p]
```

依赖方向（决定分核与同步）：

| 结果 | 依赖 | 归约方向 | 可否跨核拆分 |
| --- | --- | --- | --- |
| `C0` | `A0`, `B0` | `K0`（Matmul0 归约维） | 可沿 `M0`/`N0` 拆分；沿 `K0` 拆分需跨核累加 |
| `mean`/`var` | `C0` **整行** | `N0`（**全行**） | **不可**沿 `N0` 跨核拆分，否则需跨核归约 |
| `C0_norm` | `C0`, `mean`, `var`, `gamma`, `beta` | 逐元素 | 随 `C0` 的行块归属 |
| `C1` | `C0_norm`, `B1` | `N0`（Matmul1 归约维） | 沿 `N0` 拆分需 Split-K 与 FP32 partial 累加 |

**核心约束**：`N0` 同时是 LayerNorm 的归一化维和 Matmul1 的归约维。若沿 `N0` 跨核分核，则行统计需要跨核归约，且 Matmul1 需要 Split-K + 原子累加（`M0*M1*4 B` 量级的额外 partial 流量）。因此本设计**沿 `M0` 方向切分行块，一个任务独占 `BM` 行的全部 `N0` 列**。

### 3.1.2 数值稳定性：两遍中心化方差

`var` 有两种算法：

| 算法 | 遍数 | 风险 |
| --- | --- | --- |
| `E[x^2] - (E[x])^2` | 1 遍 | 当 `|mean|` 远大于 `sigma` 时发生灾难性抵消。FP16 输入经 FP32 累加后仍可能触发 |
| 先求 `mean`，再求 `E[(x-mean)^2]` | 2 遍 | 无抵消，代价是第二次读取行数据 |

本设计主路径采用**两遍中心化方差**：因为行块在任务生命周期内已片上驻留（L1 或 L2 slab），第二遍不产生额外 HBM 流量，代价仅为一次片上重读，用可忽略开销换掉一类精度风险。

回退路径（重算式，见 3.2.2 方案 C）采用**分块 Welford 合并**，两块合并公式：

```text
delta = mean_b - mean_a
c     = c_a + c_b
mean  = mean_a + delta * (c_b / c)
M2    = M2_a + M2_b + delta^2 * (c_a * c_b / c)
var   = M2 / N0
```

`c` 使用可覆盖 `N0` 的整型计数，只统计**有效元素**（尾块 padding 不计入）；初始空状态直接接收第一个非空块。

其他数值约定：

- 方差除数使用**总体方差**定义 `N0`，不是 `N0 - 1`；
- `rstd = 1 / sqrt(max(var, 0) + 1e-6)`，`eps` 在开方前相加；`max(var, 0)` 仅防护舍入产生的微小负值，**不修改 NaN/Inf** 来伪造有限结果；
- Vector API 的 repeat 次数若超过接口上限，必须分批执行，不能把大循环次数截断为 `uint8_t`。

### 3.1.3 dtype 语义与舍入边界（精度成败关键）

本算子有两个**必须与拼接基线严格对齐**的 FP16 舍入边界。若为了「更高精度」而跳过其中任一，结果反而会偏离 golden 并可能触发 MARE 判据。

| 阶段 | 计算 dtype | 存储 dtype | 舍入边界 | 对齐原因 |
| --- | --- | --- | --- | --- |
| Matmul0 | FP16 乘数，FP32（L0C）累加 | FP16 `C0` | 完整 `K0` 归约后 FP32 → FP16 | 对齐 `torch.mm` 输出 FP16 `c0` |
| LayerNorm 统计 | **从 FP16 `C0` 升到 FP32** | FP32 `mean`/`var`/`rstd` | 无 | 对齐 `F.layer_norm` 对 FP16 输入内部升精度 |
| LayerNorm 仿射 | FP32（含 FP32 `gamma`/`beta`） | FP16 `C0_norm` | 仿射完成后 FP32 → FP16 | 对齐 LayerNorm 输出 FP16，即 Matmul1 的输入 |
| Matmul1 | FP16 乘数，FP32（L0C）累加 | FP16 `C1` | 完整 `N0` 归约后 FP32 → FP16 | 对齐 `torch.mm` 输出 FP16 `c1` |

三点具体约束：

1. **统计必须基于已舍入到 FP16 的 `C0`**。即 Fixpipe 先把 FP32 累加结果转 FP16 落 UB，AIV 再把 FP16 转 FP32 做统计。若直接用 FP32 累加值做统计，会得到「比 golden 更准但不可比」的结果。这与任务书阶段二公式「对 `C0` 的每一行」的语义一致（`C0` 即阶段一的 FP16 输出）。
2. **两次 Matmul 的 FP32 累加不得在 K 分片之间引入 FP16 舍入**。归约必须完整走 FP32 L0C，只在最后 Fixpipe 时转 FP16 一次。
3. **编码前必须先锁定 golden 的实际 dtype 行为**：在 950PR 环境跑一个 smoke case，打印 `c0.dtype`、`c0_norm.dtype`；若与上表不符，**先修订本节与精度路径再实现 kernel**，不得通过放宽阈值掩盖语义差异。

### 3.1.4 与拼接基线的差异点及正确性措施

| 差异 | 原因 | 正确性措施 |
| --- | --- | --- |
| 3 次 launch → 1 次 MIX kernel | 消除 launch 固定开销，使 LayerNorm 可与 Cube 重叠 | 四阶段跨核 flag 严格串接；行块内语义与三段串行逐一等价 |
| `C0`/`C0_norm` 不落 HBM | 去掉 `C0` 规模的多趟往返 | 行块 ownership 保证生产与消费在同一核组内闭环；`C1` 写区间互不重叠，无原子累加 |
| 方差用两遍中心化算法 | 规避大均值下的灾难性抵消 | 行块已片上驻留，第二遍不产生 HBM 流量 |
| `M1` 方向额外分核仅在行任务不足时启用 | 小 `M0` 时行任务数不足以占满核 | 列方向无归约，分核不改变数值结果；各列组只写自己的 `C1` 列区间 |
| `eps` 为编译期常量 | 任务书未开放该属性 | Host 不暴露该参数，接口与任务书一致 |

## 3.2 总体方案

### 3.2.1 分核方式与片上驻留容量推导

行块 ownership 的代价是：`BM × N0` 的中间激活必须在任务生命周期内保活。以 `CACHE` 为留给 `C0` 的片上容量，`BM = CACHE / (N0 * 2)`：

| `N0` | 每行字节（FP16） | `CACHE ≤ 256 KiB` 时的 `BM` |
| ---: | ---: | ---: |
| 2048 | 4 KiB | 64 |
| 3072 | 6 KiB | 42 → 取 32 |
| 4096 | 8 KiB | 32 |
| 8192 | 16 KiB | **16** |

**这是本设计最关键的一处张力**：`BM` 越小，Matmul0 对 `B0` 的复用越差。Matmul0 中 `B0` 被重复读取的遍数约为 `M0 / BM`。以 `K0 = N0 = 8192`（`B0` 本身 128 MiB）为例：

| `M0` | `BM = 16` 时 `B0` 遍数 | `B0` 累计读取量 | 对应 case 标杆时延 | 倍数 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 1 GiB | 129.231 µs | 5.5× |
| 512 | 32 | 4 GiB | 240.268 µs | 11.9× |
| 1024 | 64 | 8 GiB | 481.076 µs | 11.9× |
| 2048 | 128 | 16 GiB | 867.434 µs | 13.2× |

即便 `B0` 能大量命中 L2，L2 带宽下的重复读取也会成为新的关键路径；而 `BM=16` 还同时恶化了 Cube 的 M 方向分形利用率（`16` 行仅够填满一个分形）。**结论：`N0 = 8192` 档必须放弃整行 L1 驻留**，与 2.4.5 的约束 1 一致。

### 3.2.2 三条候选路径与选型

| 方案 | `C0` 保活位置 | 适用档 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| **A：L1 驻留**（主路径） | AIC 的 L1 | `N0 ≤ 4096` | 完全消除中间 HBM 读写；Matmul1 的 A 直接取自 L1 | 挤压 A/B staging 空间，`BM` 受限于 L1 余量 |
| **B：L2 驻留 slab**（`N0 = 8192` 主路径） | GM workspace slab（按 L2 容量规划，非对外暴露） | `N0 = 8192` | 保住 `BM`（可到 128/256），Matmul0 复用度不受损 | 引入 slab 往返；依赖 L2 命中才划算 |
| **C：片上分块重计算**（回退路径） | 只保活 FP32 统计量，不保存 `C0` | 任意（容量兜底 / 非对齐极端 shape） | 容量无关，功能完整 | Matmul0 需重算约 `1 + ceil(M1/BN1)` 遍；仅作功能兜底，不作性能路径 |

选型结论：

- `N0 ≤ 4096` 走方案 A。此时 `BM ∈ {32, 64}`，`B0` 遍数为 `M0/BM`，与常规分块 GEMM 同一量级，Cube 效率损失可控。
- `N0 = 8192` 走方案 B。`BM` 取 128/256，`C0` slab 总量约 `min(aicCoreNum, rowTasks) × BM × N0 × 2 B`；以 `BM=128`、`N0=8192` 估算，单个行任务的 slab 为 2 MiB，全部活跃行任务合计在数十 MiB 量级，处于 L2 可承载范围（确切容量需按目标型号的 L2 实测确认）。配合 `L2_CACHE_HINT` 编译选项与写回策略提示，使 slab 往返尽量在 L2 闭环。该档中间量仍不对外暴露，符合任务书「中间结果由算子内部 workspace 管理」的要求。
- 方案 C 仅作为容量兜底与边界 shape 的功能路径，**不进性能验收口径**。

**Go / No-Go 判据（每一档都必须先测再定）**：

| 判据 | 通过条件 | 不通过时的动作 |
| --- | --- | --- |
| A 档 Cube 效率 | 方案 A 的 Matmul0 等效算力 ≥ 基线同 case 的 95% | 下调 `CACHE`、增大 `BM`，或把该 `N0` 档迁到方案 B |
| B 档 L2 有效性 | slab 的实测读带宽 ≥ HBM 的 3 倍 | 迁到方案 C 并接受 Matmul0 重算，或压缩 `BM` 走方案 A |
| 切换点 | 全 116 例的 `S_i` 无低于 1.0 的 case | 重新划分档位边界 |

> 上表 `BM`、slab 容量均为设计候选，不是已验证的最优值。最终取值以 116 例 profiling 为准，PR 中将附候选对比数据。

### 3.2.3 分层设计与文件清单

遵循 CATLASS「最大化复用、最小化创新」原则，分工如下：

| 层级 | 规划路径（相对 CATLASS 根目录） | 职责 | 开发方式 |
| --- | --- | --- | --- |
| 样例目录 | `experimental/matmul/matmul_layer_norm_matmul/` | 见下方文件清单 | 新增 |
| Kernel | `include/catlass/gemm/kernel/matmul_layer_norm_matmul.hpp` | 四阶段编排、行块任务与生命周期、跨核同步 | 新增（参考 `matmul_mix_fixpipe_opti.hpp`） |
| Block epilogue | `include/catlass/epilogue/block/block_epilogue_layer_norm_ascend950.hpp` | 行统计、归一化、`gamma`/`beta` 仿射、FP16 回写、UB→L1 | 新增 |
| Dispatch policy | `include/catlass/epilogue/dispatch_policy.hpp` 追加 `EpilogueAscend950LayerNorm<...>` | 承载 `UB_STAGES`、`SPLIT_M` 等策略 | 新增（复用文件） |
| Epilogue 注册 | `include/catlass/epilogue/block/block_epilogue.hpp` 追加 include | 按现有 `CATLASS_ARCH == 3510` 守卫注册 | 修改 1 行 |
| 行归约 tile | `include/catlass/epilogue/tile/tile_row_reduce.hpp` | 行方向 `sum` / `sumsq` 归约 | **新增**（仓库当前无通用 reduce tile） |
| Tile 复用 | `tile_cast.hpp`、`tile_broadcast_mul.hpp`、`copy_ub_to_l1_tla.hpp`、`copy_l0c_to_ub.hpp` | FP16↔FP32 转换、`gamma`/`beta` 广播、UB→L1、L0C→UB | 直接复用 |
| 同步复用 | `include/catlass/arch/cross_core_sync.hpp` | `CrossCoreSetFlag(WithReverse)` / `CrossCoreWaitFlag(WithReverse)` | 直接复用 |
| optest 接入 | 见下方文件清单 | JIT ABI、kernel 组装、torch adapter、Python wrapper、pytest | 新增（按 `catlass-example-to-pytest` skill） |
| ATK 交付 | 见 4.2.2 | ATK 四件套 | 新增（按 `catlass-atk-support` skill） |

**样例与 optest 的文件清单严格按任务书《PR 申请合入》的目录树组织**（`${op_name}` 取 `matmul_layer_norm_matmul`）：

```text
experimental/matmul/matmul_layer_norm_matmul/
  ├── CMakeLists.txt
  ├── matmul_layer_norm_matmul.cpp          # 算子样例入口
  ├── README.md
  └── test_matmul_layer_norm_matmul.py      # 接入 optest 的测试件

include/catlass/
  ├── gemm/kernel/matmul_layer_norm_matmul.hpp
  └── epilogue/
      ├── block/block_epilogue_layer_norm_ascend950.hpp
      └── tile/tile_row_reduce.hpp

tests/optest/
  ├── include/catlass_kernel_jit.h                       # 算子函数签名注册
  ├── kernels/
  │   ├── matmul_layer_norm_matmul/
  │   │   ├── matmul_layer_norm_matmul.cpp
  │   │   ├── matmul_layer_norm_matmul_impl.cpp
  │   │   └── CMakeLists.txt
  │   ├── CMakeLists.txt                                 # 补 add_subdirectory
  │   └── common/kernel_runner.h
  ├── README.md
  ├── src/
  │   ├── catlass_matmul_layer_norm_matmul.cpp
  │   └── include/template/matmul_layer_norm_matmul.h
  └── torch_catlass/
      ├── __init__.py
      └── ops/
          ├── __init__.py
          └── matmul_layer_norm_matmul.py
```

**样例注册**（`experimental/matmul/matmul_layer_norm_matmul/CMakeLists.txt`，使用仓库既有宏）：

```cmake
set_source_files_properties(matmul_layer_norm_matmul.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(matmul_layer_norm_matmul mix matmul_layer_norm_matmul.cpp)
```

kernel type 取 `mix`，因为本算子必须同时使用 AIC 与 AIV；optest 侧对应 `JitKernelType::MIX`，`NPU_ARCH_LIST` 取 `3510`。

## 3.3 Host 侧设计

### 3.3.1 分派与校验顺序

分派顺序固定为：**参数检查 → 空输出处理 → 片上资源检查 → 档位与 TileShape 选择 → workspace 规划 → 输出分配 → 当前 stream 上一次 launch**。

参数校验清单：

1. 五个输入均在 NPU 且位于同一 device；
2. `A0`/`B0`/`B1` 为 FP16 二维，`gamma`/`beta` 为 FP32 一维；
3. `A0.shape[1] == B0.shape[0]`（即 `K0` 一致）；
4. `B0.shape[1] == B1.shape[0] == gamma.numel() == beta.numel()`（即 `N0` 一致）；
5. `A0` 为 RowMajor stride，`B0`/`B1` 为 ColumnMajor stride，`gamma`/`beta` 连续；
6. 所有逻辑维度为正；所有维度乘积、字节数、偏移使用**检查溢出的 64 位算术**，超出底层字段范围时报错，不得截断为 `uint32_t`；
7. 输出固定分配为 `(M0, M1)`、FP16、RowMajor。

输入地址使用 `tensor.data_ptr()` 取逻辑首元素，**不能**用 `tensor.storage().data()`，否则带 storage offset 的合法 view 会读错位置。校验失败返回明确错误且**不启动 kernel**；布局不符时明确报错，不在公共接口内部隐式插入一个未计入的转置 kernel。

边界语义：`M0 = 0` 或 `M1 = 0` 返回对应空输出；`N0 = 0` 无有效归一化域，报参数错误；`K0 = 0` 且 `N0 > 0` 时定义 `C0` 全零，kernel 跳过 Matmul0 并执行 `FP16(beta) @ B1`。

### 3.3.2 分核与任务 ownership

```text
rowTasks   = CeilDiv(M0, BM)
colGroups  = CeilDiv(M1, BN1)                     # 仅当行任务不足时启用
aicUsed    = min(GetCoreNumAic(), rowTasks * colGroups_used)
taskId     = blockIdx
rowTaskId  = taskId / colGroups_used
colGroupId = taskId % colGroups_used
```

当 `rowTasks >= aicCoreNum` 时取 `colGroups_used = 1`，每个任务独占 `BM` 行并覆盖全部 `M1` 输出列。

当小 `M0` 导致行任务不足时，把 `M1` 划分为若干不相交列组，同一行块被多个任务复制处理，各任务只写自己的 `C1` 列区间。**复制会增加 Matmul0 与 LayerNorm 的重复计算，不能仅按「满核」选择**。Host 侧在候选中比较以下估算并用实测系数校准：

```text
F0 = 2 * BM * K0 * N0        # Matmul0 计算量
F1 = 2 * BM * N0 * M1        # Matmul1 计算量
estimatedWork(colGroups) = colGroups * F0 + F1
```

原则：`K0` 大或 Matmul0 占主导时优先列组=1；仅在 `M0` 小、`K0` 较小且 `M1` 大时才考虑复制。任何两个任务的 `C1` 写区间必须互斥。

### 3.3.3 片上容量预算

记 `N0a = RoundUp(N0, 16)`（FP16 32 B 对齐），`BMV = BM / 2`（每个 AIV 子块承担的行数，因 `SPLIT_M` 把 L0C 的 M 维一分为二）。

L1 预算（两个 GEMM 阶段分别校验峰值）：

```text
CACHE    = BM * N0a * sizeof(half)                    # 仅方案 A 占用
L1_MM0   = CACHE + A1_STAGES * BM  * BK0 * sizeof(half)
                  + B1_STAGES * BK0 * BN0 * sizeof(half) + GUARD
L1_MM1   = CACHE + A1_STAGES * BM  * N0a * sizeof(half)   # C0_norm 作为 A 操作数
                  + B1_STAGES * BK1 * BN1 * sizeof(half) + GUARD
max(L1_MM0, L1_MM1) <= 512 KiB
```

UB 预算（每个 AIV 独立）：

```text
UB = UB_STAGES * BMV * BN0 * sizeof(half)    # FP32 Fixpipe tile 中转（含 ping-pong）
   + BMV * BN0 * sizeof(float)               # FP32 归一化工作区
   + 2 * BN0 * sizeof(float)                 # gamma / beta 分片
   + 4 * BMV * sizeof(float)                 # sum / mean / var / rstd
   + reduce_scratch + alignment_and_event_reserve
UB <= 248 KiB
```

L0C 预算：`max(BM * BN0, BM * BN1) * sizeof(float) <= 256 KiB`；方案 C 下 Matmul0 与 Matmul1 的累加器同时存活，须分配不重叠区域并满足 `4*BM*BN0 + 4*BM*BN1 + reserve <= L0C 容量`，**不能套用主路径的 max 预算**。

初始候选档位（`BN0 = BN1 = 256`，`A1_STAGES = 1`，`B1_STAGES = 2`，`UB_STAGES = 2`）：

| 档位 | 触发条件 | `BM` | `BK0`/`BK1` | `CACHE` | 保活位置 |
| --- | --- | ---: | ---: | ---: | --- |
| T0 | `N0 <= 2048` | 64 | 128 | 256 KiB | L1 |
| T1 | `N0 == 3072` | 32 | 128 | 192 KiB | L1 |
| T2 | `N0 == 4096` | 32 | 128 | 256 KiB | L1 |
| T3 | `N0 == 8192` | 128 / 256 | 128 | — | L2 slab |
| T4 | `M0 <= 128`（并行度档） | T0–T2 减半 | 128 | <= 128 KiB | L1 |
| T5 | 通用尾块 / 非对齐 | 保守固定值，16 对齐 | 128 | <= 128 KiB | L1 |

**所有静态容量约束必须以 `static_assert` 落到代码里**，运行期 shape 约束由 `CanImplement` 校验；本文档的估算值不作为实现依据，以组件真实静态分配复算为准。

**分层选择如何落到 JIT**：optest 的 `TParams` 已包含 `l1TileShape` / `l0TileShape` / `swizzle`（编译期 JIT 参数）与 `flag`（通用编译期布尔开关）映射，因此「按 shape 选 TileShape 与路径」可表达为 JIT 编译期参数，无需自造运行时 dispatcher。Host 侧成本模型输出 `(l1TileShape, l0TileShape, flag["l2Slab"], flag["colGroups"])` 后交给 JIT 编译对应变体。

### 3.3.4 GM workspace 布局

```text
[0)                     mean    : M0 * sizeof(float)
[1)                     rstd    : M0 * sizeof(float)
[2) 仅方案 B / C        C0 slab : min(aicUsed, rowTasks) * BM * N0a * sizeof(half)
```

方案 A 主路径的用户 workspace 为 `0`（或仅保留 `mean`/`rstd` 两个诊断槽）。**禁止为了复用现有 multistage workspace kernel 而把 `C0` 无条件写入 GM。**

## 3.4 Kernel 侧设计

### 3.4.1 组件构成

| 组件 | 开发方式 | 说明 |
| --- | --- | --- |
| `BlockMmad`（Matmul0 / Matmul1） | 复用 + 适配 | 复用 950 Mmad 组件；**Matmul1 需要让 A 操作数来自 L1 而非 GM**，这是 kernel 层最需要新写/适配的部分 |
| `BlockScheduler` | 参考改写 | 行块 ownership 调度，参考 `GemmIdentityBlockSwizzle` |
| `BlockEpilogueLayerNorm` | **新增** | 行统计 + 归一化 + 仿射 + FP16 回写 |
| `TileRowReduce`（FP32） | **新增** | 仓库无通用 reduce tile，需按微指令 `ReduceSum` 或 `WholeReduceSum` 自建 |
| `CopyL0CToUBTla`（`SPLIT_M`） | 直接复用 | `dualDstCtl = 1`，M 维一分为二给两个 AIV 子块，要求 `m` 为偶数 |
| `CopyUb2L1Tla` | 直接复用 | RowMajor → zN，把 `C0`/`C0_norm` 送入 L1 供 Matmul1 |
| `TileCast` | 直接复用 | FP16 → FP32 升精度、FP32 → FP16 回写 |
| `TileRowBroadcastMulTla` | 直接复用 | `gamma`/`beta` 按列仿射的行广播乘 |
| `Arch::CrossCoreFlagWithReverse` | 直接复用 | AIC/AIV 四阶段同步 |

`BlockEpilogueLayerNorm` 若仅服务本样例，可先放在 kernel 私有实现中，确认接口稳定后再下沉到公共 `epilogue/block`，避免为单一场景过早扩展公共 API。

### 3.4.2 四阶段流水

一个行任务内部四个阶段，阶段之间用跨核 flag 串接；阶段在**行任务维度**上重叠——AIC 处理任务 `t+1` 的 Phase 0 时，AIV 正在做任务 `t` 的 Phase 1/2。

```text
                    ┌─────────────────── AIC ───────────────────┐   ┌──────── AIV ────────┐
Init ───────────────┤ 解析 shape / TileShape / 任务映射           │   │ 载入 gamma/beta 分片 │
                    │ 划分 L1 / UB / L0A / L0B / L0C 资源        │   │ 初始化行统计 scratch  │
                    │ 初始化 AIC↔AIV flag（set/wait 严格配对）    │   │                     │
                    └───────────────────────────────────────────┘   └─────────────────────┘
                                                     │
Phase 0  遍历 N0 的每个 BN0 tile：                     ▼
    AIC: A0/B0 tile → L1 → L0A/L0B → MMAD（FP32 L0C 累加）
    AIC: Fixpipe FP32 → FP16 C0 tile，经 SPLIT_M 落到两个 AIV 的 UB
    AIV: cast FP16 → FP32，沿 N 累加 rowSum（只统计有效列）
    AIV: 把 FP16 C0 tile 写入保活位置（方案 A：L1；方案 B：GM slab）
                                                     │
Phase 1  AIV: mean = rowSum / N0                      ▼
    AIV: 重读保活位置的 C0，累加 sum((x - mean)^2)  → var
    AIV: rstd = 1 / sqrt(max(var,0) + 1e-6)
                                                     │
Phase 2  AIV: y = (x - mean) * rstd * gamma[n] + beta[n]（全程 FP32）
    AIV: y → FP16，就地覆写保活位置（L1 语义由 C0 变为 C0_norm）
                                                     │
Phase 3  AIC: 以 L1 中 C0_norm[BM, N0] 为 A，GM B1 为 B
    AIC: 沿 K = N0 归约，L0C FP32 累加
    AIC: Fixpipe → FP16 C1，只向 GM 写回一次
```

`gamma`/`beta` 为 `N0` 长度的 FP32 向量。`N0 = 8192` 时两者合计 64 KiB，占 UB 的 26%，因此按 `BN0` 分片搬入并在 `BMV` 行间复用，而不在 Init 阶段整段常驻。

### 3.4.3 同步与死锁约束

1. flag 分配：`mm0Done`（AIC→AIV）、`lnDone`（AIV→AIC）、`mm1Done`（AIC→AIV 反向释放）及各 reverse flag，总数需落在 `FFTS_MAX_FLAG = 7` 的可用范围内；配对 AIC/AIV 使用固定 flag 范围，不得与 CATLASS 组件内部 flag 冲突。
2. 遵守 `MAX_REVERSE_DEPTH = 15` 的硬件约束：同一 flag 连续 set 不超过该深度，否则系统 hang；Phase 3 完成后回送反向 flag 释放保活 buffer。
3. 每个 stage 的 ready/idle 的 set 与 wait 次数在两侧**完全一致**；采用 2 stage ping-pong，保证任务 `t` 的 Phase 3 与任务 `t+1` 的 Phase 0 不争用同一块空间。
4. **尾块行任务也必须走完四个阶段**（可能是空计算）。任何 `CrossCoreWaitFlag` 之前不得存在「部分核不参与」的条件分支，否则跨核等待不配对造成死锁。无有效行的 AIV 仍须履行核组事件协议。
5. AIV 的 block index 必须用 `GetBlockIdx() / GetSubBlockNum()`（或 `/ GetTaskRation()`）映射回所属 AIC，**不能把 AIV 的物理 block index 直接当作 AIC 行任务索引**。
6. 不使用普通 GM 标志轮询构造全设备同步，也不使用额外初始化 kernel（会破坏单 launch 要求）。`AscendC::SyncAll<false>()` 虽有先例，但要求 AIC/AIV 两侧严格配对，本设计**仅在需要跨核组栅栏的收尾 drain 处使用**，不作为阶段串接的基线手段。
7. kernel 退出前执行必要的 BlockMmad drain 与 `PipeBarrier<PIPE_ALL>`；调试构建记录 stage/task/flag，生产构建移除 device print。

### 3.4.4 尾块与对齐

- `M0 % BM != 0`：最后一个行任务按实际行数计算，GM 读写使用 `DataCopyPad` 保护；`SPLIT_M` 要求 M 为偶数，尾行按 `RoundUp` 补齐并让空行走完协议。
- `N0 % BN0 != 0`：`N0a = RoundUp(N0, 16)`，padding 区在 Phase 0 写 `0`，且**不计入行统计的元素计数**（统计始终除以真实 `N0`）；Phase 2 对 padding 区写 `0`，使其对 Matmul1 的 K 归约贡献为 0。
- `M1 % BN1 != 0`：最后一个输出列 tile 用 `DataCopyPad` 按实际列数写回，禁止越界。
- FP16 的 GM/L1/UB 搬运按 32 B 对齐；无法确定对齐时使用 CATLASS 的边界 copy 或 `DataCopyPad` 安全路径。
- padding 只用于存储与计算对齐，**不改变 LayerNorm 的 `N0` 分母**。
- 任务测试集全部对齐，尾块路径由自建泛化用例（4.2.2）覆盖。

## 3.5 支持硬件

| 芯片版本 | 支持 |
| --- | --- |
| Ascend 950PR | √（本任务性能验收平台） |
| Ascend 950DT | 同 `CATLASS_ARCH=3510` 架构，代码可编译；本任务未做真机验收，不作交付承诺 |
| Atlas A2 / A3 | ×（本任务不涉及；`Arch::AtlasA2` 的 L0C 仅 128 KiB，且无 `CopyUb2L1Tla` 通路，TileShape 需重新分档） |

## 3.6 算子约束限制

任务书给出的约束为「无」。本实现不额外声明 shape 白名单，但外部契约仍要求：

1. 仅支持 FP16 输入 / FP16 输出，`gamma`/`beta` 为 FP32；
2. 仅支持 2 维 `A0`/`B0`/`B1`（无 batch 维）；
3. `B0`、`B1` 必须为 ColumnMajor，`A0`、`C1` 必须为 RowMajor；
4. `eps` 固定 `1e-6`，不作为属性开放；
5. `gamma`/`beta` 长度严格等于 `N0`；
6. `mean`/`variance` 不对外输出；
7. 功能路径处理尾块与非对齐 shape；**性能只对任务 116 个 case 承诺验收目标**。

---

# 四、可维可测分析（required）

## 4.1 精度标准与性能标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 功能 | 官方 116 例全部通过，无 crash / timeout / NaN 异常 | 9 月任务书 + 测试集 CSV |
| 精度 | 满足生态算子开源精度标准（FP16 档） | <https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md> |
| 泛化精度 | ATK 泛化 ≥ 200 例精度通过 | 9 月任务书 |
| 性能 | 平均标杆时延 / 平均测试时延 > 1.1 | 9 月任务书 + CSV |
| 融合 | 单次 kernel launch，`C0`/`C0_norm` 无 HBM 中转 | 任务书功能要求 |

误差定义（与开源精度标准口径一致，固定 opbase 标准 revision，不自行放宽容差）：

```text
relative_error = abs(actual - golden) / (abs(golden) + 1e-7)
```

对接近零的 golden 同时记录**最大绝对误差**，避免只看相对误差造成误判。调试构建可按阶段检查 `C0`、`mean`、`var`、`C0_norm`，调试导出不进入验收接口与性能数据。

性能采集使用 `msprof op`，在 Ascend 950PR 上、与标杆**同机同批次**采集。逐 case 记录：

```text
idx, M0, K0, N0, M1, baseline_us, kernel_us, speedup, tier, pathKey,
BM, BN0, BK0, BN1, BK1, colGroups, precision_pass
```

同时报告逐例 `S_i`、算术平均 `S_avg`、`Σ T_ref / Σ T_fused`、P50、最差 case 与退化 case，并说明不同实现方案与 TileShape 调整的依据（任务书明确要求备注）。

## 4.2 测试设计

### 4.2.1 optest：任务测试集全覆盖

Golden 必须**保持任务链路与 dtype**，不得用全 FP32 结果替代：

```python
def golden(a0, b0, b1, gamma, beta, eps=1e-6):
    c0    = (a0.float() @ b0.float()).half()          # Matmul0 → FP16 边界
    x     = c0.float()                                # LayerNorm 内部升精度
    mu    = x.mean(dim=-1, keepdim=True)
    var   = ((x - mu) ** 2).mean(dim=-1, keepdim=True) # 总体方差
    y     = (x - mu) / torch.sqrt(var + eps) * gamma + beta
    c0n   = y.half()                                  # LayerNorm → FP16 边界
    return (c0n.float() @ b1.float()).half()          # Matmul1 → FP16 边界
```

测试同时断言 `c0` / `c0_norm` / `golden` 的 dtype，防止框架版本变化导致精度口径漂移。**不得以「最终一次 `to(float16)`」的高精度结果代替拼接基线的两个中间 FP16 转换。**

用例组织：

| 类别 | 内容 |
| --- | --- |
| 官方主集 | CSV 116 例按原始 idx 参数化 |
| 档位边界 | T0–T5 每档至少 1 例 |
| 尾块 | 不能被 `BM`/`BN0`/`BN1` 整除的补充 shape |
| 最小合法 shape | 小矩阵 smoke |
| 零方差 / 常数行 | 构造 `C0` 常量行，验证 `eps` 与无 NaN |
| 数值分布 | 正负随机、近零、较大有限值、近常数且均值大方差小 |
| `gamma`/`beta` | `gamma=1,beta=0`；`gamma=0`；随机 FP32；含负值 |
| 列主布局 | 非方阵 + 非对称数据，检测转置错误 |
| 负向接口 | dtype / rank / shape / stride / device 不匹配 |

随机用例固定 seed，失败日志打印 seed、shape、档位、TileShape、MERE/MARE 与首个错误位置。运行方式：

```bash
cd tests/optest
bash build.sh
python -c "import torch_catlass; print(hasattr(torch_catlass, 'matmul_layer_norm_matmul'))"
pytest tests/test_matmul_layer_norm_matmul.py -v -s
```

无 Ascend 950 时仅允许用 `@only_on_3510` 跳过运行测试；编译、导入与 ABI 问题**不能用 skip 掩盖**。

### 4.2.2 ATK：泛化 ≥ 200 例

按 `catlass-atk-support` skill 从 optest 测试件生成 ATK 四件套，**目录名 == `name` == 文件后缀**，三锚点（`name` / `register` / `generate`）先定后写：

```text
matmul_layer_norm_matmul/
  ├── matmul_layer_norm_matmul.yaml       # 用例规格，api_type 与 register 一致
  ├── generator_matmul_layer_norm_matmul.py
  ├── execute_matmul_layer_norm_matmul.py # cpu 分支算 golden / npu 分支调 torch_catlass
  └── node.yaml
```

泛化矩阵设计（目标 ≥ 200 例有效通过，计划 240 例以保证扣除异常/重复后仍达标）：

| 类别 | shape 个数 | 说明 |
| --- | ---: | --- |
| 对齐中小型 | 20 | `M0`/`K0`/`N0`/`M1` 均为 16/32/64/128 的倍数，不重复 CSV |
| 带尾块 | 20 | 各维在 16/32/64/128 附近取 ±1、±3 |
| 退化小维度 | 10 | `M0=1`、`M1=1`、`N0=1`、`N0=2`、`K0=1` |
| 跨驻留容量边界的长 `N0` | 10 | 至少覆盖 `N0 ∈ {4095, 4097, 8191, 8193}`，限制 `M0`/`K0`/`M1` 保证可执行 |

每个 shape 配 4 类数据分布 → 共 240 例：① 尺度受控的零均值随机输入；② 正负交替（考验消减误差）；③ 零/常数构造（考验 `eps` 与零方差）；④ 近常数且带非均匀 FP32 `gamma`/`beta`。

生成器固定 seed 并输出去重后的 case manifest，记录 shape、dtype、stride、数据分布、档位、TileShape 与结果。**异常、skip、仅重复运行同一 case 的条目不计入 240 例**，正式报告明确实际成功条数。执行命令（按 skill 约定）：

```bash
atk case -f matmul_layer_norm_matmul.yaml -p generator_matmul_layer_norm_matmul.py
atk task -c result/matmul_layer_norm_matmul/json/all_matmul_layer_norm_matmul.json \
         -n node.yaml --task accuracy -p execute_matmul_layer_norm_matmul.py
```

### 4.2.3 融合性与性能证据

「无中间 HBM 中转」需**三项证据并用**，仅看到一个 kernel 名称不能证明：

1. 源码中不存在 `C0`/`C0_norm` 的 GM 地址；
2. 内存规划中不存在对应分配（方案 A 的 workspace 为 0）；
3. profiling / 编译分析的访存路径符合 `L0C → UB → L1`。

性能采集：

```bash
msprof op --application ./matmul_layer_norm_matmul --warm-up 30 --output ./prof/<case_id>
```

小 shape 提高预热次数避免未提频影响；每例先做正确性检查，再采集设备执行数据。性能分析至少查看：AIC/AIV 与 MTE2/MTE3/FixPipe 占比、AIC 等 AIV 与 AIV 等 AIC 的空泡、每核任务数与长尾、L1/UB 占用与 bank conflict、L2 命中率、四阶段的热点与首尾 flush。任何 TileShape、`colGroups`、流水深度或档位边界的调整都必须保留前后对比数据；切换档位后执行 clean rebuild 并核对日志中的目标名与参数，避免旧二进制污染结果。

无 NPU 时的仿真、静态检查或 CPU golden 可用于开发，**不能替代 950PR 验收结果**。

### 4.2.4 自验证报告

含：用例参数表、逐例精度结论与执行日志/截图、整体通过截图、逐例性能数据与平均比值、不同 TileShape / 档位候选的对比数据（说明最终固化值的依据）、ATK 泛化 manifest 与成功条数、复现步骤。按任务书要求，自验证报告内容补充在 **PR 描述**中，不作为独立文件提交进源码 PR。

## 4.3 兼容性分析

新增算子，不修改 CATLASS 已有组件的对外行为，无兼容性影响。可维护性约定：

1. 新增 epilogue 通过 `include/catlass/epilogue/block/block_epilogue.hpp` 注册，沿用现有 `CATLASS_ARCH == 3510` 宏守卫，不改变已接入算子的 schema；
2. `BlockEpilogueLayerNorm` 按 CATLASS Epilogue 约定实现，后续 `RMSNorm`、`Matmul + 单 LayerNorm` 等算子可直接复用；
3. 档位选择集中在 Host 侧一处表驱动，新增 shape 档不改 kernel 代码；
4. 样例、optest 与 ATK **共用同一 kernel 入口与参数语义**，避免测试路径与实际实现分叉；模板选择函数只依赖 shape、布局与硬件信息，不依赖输入数据值；
5. 所有静态资源约束用 `static_assert` 固化，运行期约束用 `CanImplement` 拦截；
6. 新增样例目录名登记进 `examples/CMakeLists.txt` 的 Ascend 950 列表（见风险 R-07）。

## 4.4 风险与待确认项

| 编号 | 事项 | 影响 | 处置 |
| --- | --- | --- | --- |
| T-01 | 性能口径歧义（2.3.2 的 A / B 两种解析） | 优化优先级与最终判定结论可能不同 | 自验同时报告两种口径；向任务维护者确认主口径后再固化 |
| T-02 | `experimental/` 未被根 `CMakeLists.txt` 纳入构建 | 样例可能不参与默认构建，评审与复现体验受影响 | 确认合入方是否接受；如不接受则同时登记进 `examples/CMakeLists.txt` 或补 `experimental` 的构建接入（见 R-07） |
| T-03 | golden 的两个中间 FP16 舍入边界在 950PR 上的实际行为 | 若与 3.1.3 假设不符，精度路径需重做 | 编码前跑 smoke case 打印 dtype 并锁定；不符先改设计 |
| T-04 | Fixpipe 直出 FP16 落 UB 无仓库先例（现有 16 位样例是 bf16） | 若该路径在 950 上不可用，Phase 0 需改为 FP32 落 UB + AIV 显式 cast | 先用最小 kernel 验证；不可用则走 FP32 中转（UB 预算需重算） |
| T-05 | Matmul1 的 A 操作数来自 L1 的通路需新增或适配 | kernel 层主要工作量与主要技术风险 | 复用 `CopyUb2L1Tla` + 参考 FA 的 A-in-L1 用法；先用最小 case 打通 |
| T-06 | 仓内无通用行 reduce tile | 需新建 `TileRowReduce`，属新增公共面 | 先内联在 epilogue 私有实现中，接口稳定后再下沉 |
| R-01 | `N0=8192` 档 Cube 效率损失 | 该档占标杆时长 38.6%，模型比值仅 1.065–1.093，是 binding constraint | 该档不用方案 A；按 3.2.2 的 Go/No-Go 判据先测再定，备选方案 C |
| R-02 | 方案 A 的 L1 cache 挤压 A/B 双缓冲 | MTE2 阻塞、Cube 空泡 | 下调 `BK`/`BN` 或 `BM`；按 phase 复用 L1；以 `static_assert` 为硬门禁 |
| R-03 | AIV 的 LayerNorm 成为新瓶颈 | AIC 长时间等 AIV，融合收益退化为 2.4.3 水平 | 调整 `BN0`、UB stage、归约实现；流水化 Phase 2/3 |
| R-04 | 两遍方差增加 Vector 耗时 | Phase 1 占比过高 | 先验证精度；只有单遍算法同时满足全部精度用例才允许替换 |
| R-05 | 小 `M0` 核利用率不足 | `rowTasks < aicCoreNum` | 用 3.3.2 的成本模型选 `colGroups`，不盲目满核 |
| R-06 | 列主布局解释错误 | 与 golden 大面积不一致 | 固定 stride 用例 + 非对称数据；检查 TLA Layout 与 wrapper 是否误调 contiguous |
| R-07 | 样例注册位置与任务书目录要求不完全一致 | 可能被评审要求调整 | 按任务书 `experimental/matmul/${op_name}` 提交；在 PR 描述中说明 `experimental` 的构建接入方式与复现命令 |
| R-08 | flag 死锁 / 配对不齐 | kernel timeout 或卡死 | 最小 stage 测试，核对两侧 set/wait；逐步增加流水，不带问题提交 |
| R-09 | 19 µs 级小 case 不达标 | launch/flush 占比高 | 精简静态方案、同步与初始化；提高预热并确认测量口径 |
| R-10 | 提交件夹带非必要内容 | 违反任务书黑名单，PR 被打回 | 提交前对照黑名单逐项自查（见下） |

**任务书「不希望出现的内容」黑名单（提交前十项自查）**：

1. PR 描述/提交中不含任何个人敏感信息；
2. 不在 PR 提交中附注精度、性能测试报告（报告写入 PR 描述）；
3. 不包含编译中间文件与二进制；
4. 不包含辅助精度、性能测试用的临时脚本；
5. 不包含非必要内容（如 Tiling 调优使用的模型权重）。

---

# 五、参考资料

1. 9 月任务书：`9月社区任务-MatmulLayerNormMatmul算子开发/MatmulLayerNormMatmul_task_doc.md` 与 `test_case/MatmulLayerNormMatmul_测试集.csv`
2. 7 月同名任务书（用于版本差异比对）：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/MatmulLayerNormMatmul_task_doc.md>
3. 社区设计模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
4. 社区任务流程及注意事项：<https://gitcode.com/org/cann/discussions/39>
5. CATLASS 创新样例开发流程指南：`docs/zh/1_Practice/10_innovative_example_development_guide.md`
6. CATLASS 性能调试：`docs/zh/1_Practice/evaluation/performance_tools.md`
7. CATLASS optest：`tests/optest/README.md`；接入 skill：`.agents/skills/catlass-example-to-pytest/SKILL.md`
8. CATLASS ATK 支持 skill：`.agents/skills/catlass-atk-support/SKILL.md`
9. 任务书指定参考样例：`examples/44_quant_matmul_full_loadA_tla`
10. 参考合入 PR：<https://gitcode.com/cann/catlass/pull/678>
11. ATK 测试框架：<https://gitcode.com/Ascend/ATK/blob/master/README.md>
12. 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
13. `msprof op` 使用说明：<https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/profiling/msprof_cmd/general_collect_commands.md>
