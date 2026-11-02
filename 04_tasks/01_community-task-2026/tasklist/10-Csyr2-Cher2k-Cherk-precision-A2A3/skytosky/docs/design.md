# 【CANN社区任务】Csyr2/Cher2k/Cherk 算子精度修复设计文档（A2/A3）

目标仓库：<https://gitcode.com/cann/ops-blas>
目标实现目录：`blas/syr2/arch22/`、`blas/herk/arch22/`
目标硬件：Atlas A2/A3（DAV_2201，性能验收 910B3）
贡献者：`skytosky`

# 需求背景

## 需求来源

本设计对应 2026 年 10 月社区任务“Csyr2/Cher2k/Cherk 算子精度修复（A2/A3）”。三个算子已在 ops-blas 合入 arch22 实现，本次不是新增接口或重新设计 ABI，而是修复原地更新时非操作区被改写的问题。

当前缺陷是：实现更新 `uplo` 指定三角时，可能覆盖未引用的另一三角及 `lda/ldc` 超出逻辑阶数 `n` 的 padding。任务要求操作区满足混合容差，同时非操作区与调用前输入逐 bit 一致，包括 NaN payload、±0、±Inf 和非规范数。

## 现状与修复原则

- 公开声明、参数顺序、枚举值、返回码和 Host/Device 归属全部冻结，禁止新增平行 API。
- 只修改 `blas/syr2/arch22/` 与 `blas/herk/arch22/`，不修改 950PR 路径；公共 Host 若确需调整，必须证明不影响 950。
- 采用**计算与写回分离、写回按操作区裁剪**的修复方式，禁止“先写满再恢复”“补全另一三角”或 CPU fallback。
- quick return 不启动写回，整个输入缓冲保持不变；不得为保护非操作区引入与问题规模无关的持久 Device 缓存。

# 需求分析

## 功能与接口

三个接口复用 `include/cann_ops_blas.h` 现有声明：

| 算子 | 数学语义 | 特殊约束 |
| --- | --- | --- |
| Csyr2 | `A = alpha·x·y^T + alpha·y·x^T + A` | 普通对称、不共轭；对角虚部按公式更新；支持负 incx/incy |
| Cher2k | N：`C = alpha·A·B^H + conj(alpha)·B·A^H + beta·C`；C：`C = alpha·A^H·B + conj(alpha)·B^H·A + beta·C` | 仅 N/C；alpha 复数、beta 实数；对角虚部强制为 0 |
| Cherk | N：`C = alpha·A·A^H + beta·C`；C：`C = alpha·A^H·A + beta·C` | 仅 N/C；alpha/beta 实数；对角虚部强制为 0 |

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| `handle` | ops-blas 上下文和 stream | handle | 不涉及 | 有效非空 | 标量 |
| `uplo` | 指定引用/更新三角 | 枚举 | UPPER/LOWER | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` | 标量 |
| `trans` | A/B 操作方式 | 枚举 | N/C | Cher2k/Cherk 不支持 T | 标量 |
| `n/k` | 矩阵阶数与秩 | int32 | int32 | 均不小于 0 | 标量 |
| `alpha/beta` | 标量乘数 | 指针 | 按接口为 complex64 或 float | 非空；quick return 按原语义 | 标量 |
| `x/y` | Csyr2 输入向量 | Tensor | complex64 | incx/incy 非零，支持负步长 | 逻辑长度 n |
| `A/B` | Cher2k/Cherk 输入矩阵 | Tensor | complex64 | 列主序、满足对应 lda/ldb | N/C 对应形状 |
| `A/C` | 原地更新矩阵 | Tensor | complex64 | 仅 uplo 三角可写；另一三角与 padding 不变 | `ld×n` |

quick return 保持原接口语义：Csyr2 在 `n=0` 或 `alpha=(0,0)` 时直接成功返回；Cher2k/Cherk 在 `n=0`，或 `(alpha=0 或 k=0) 且 beta=1` 时直接成功返回。`alpha=0 且 beta!=1` 时仅按 beta 缩放操作区。`beta=0` 时旧 C 不必是有效输入，但非操作区仍不得改写。

## 非操作区定义

列主序下 `M(i,j)=M[i+j·ld]`。UPPER 的操作区为 `i<=j`，LOWER 的操作区为 `j<=i`，均含对角。非操作区包括：

1. 另一严格三角；
2. `n<=i<ld` 的前导维 padding。

这些位置必须零写入，并与调用前输入 `memcmp` 逐 bit 一致。不得先读取另一三角再补写，因为 NaN payload 和符号零必须原样保留。

# 详细设计

## 总体修复结构

将现有实现拆成“计算结果生成”和“结果提交写回”两个逻辑阶段。允许在寄存器或 UB 中计算完整 tile，但 GM 写回必须逐元素满足：

```text
inside_n = row >= 0 && col >= 0 && row < n && col < n
in_uplo  = (uplo == UPPER ? row <= col : row >= col)
may_write = inside_n && in_uplo
```

`may_write=false` 的位置不执行任何 GM store、DataCopy/DataCopyPad 或整块回写。 Host 不再通过事后拷贝恢复另一三角。

## Host 与 Tiling 设计

1. 保留原参数校验、枚举错误码、quick return、alpha/beta 处理和 stream 绑定顺序。
2. Tiling 继续按既有 Cube/Vector 路径选择，但增加 `n/ld/uplo` 对应的行列边界，不通过扩大 ld 触发非操作区写入。
3. 对角 tile 单独标记为跨三角 tile；完整内部 tile 可批量提交，跨三角 tile 进入裁剪写回。
4. 采用运行时 AIV/AIC 核数，不因修复引入固定芯片核数或新的持久 workspace。
5. quick return 在 Host 直接返回，不启动 kernel；scaling-only 路径也使用同一写回裁剪。

## Kernel 写回设计

### 完整内部 tile

当 tile 的行列范围全部位于同一操作三角且全部 `row/col<n` 时，保留现有 Vector/Cube 计算与合并写回，减少性能损失。

### 跨三角 tile 与尾 tile

采用两种等价实现方式，按现有代码结构选择改动较小者：

1. **掩码写回**：逐元素以 `may_write` 控制 store；保留原 FP32 运算顺序。
2. **拆 tile**：将跨三角区域拆成上/下两段连续操作区，分别计算和提交；无法安全批量提交的边界退化为标量逐点写回。

如果现有 Cube 输出先落在 UB，可在 UB 保留完整结果，再仅搬运操作区到 GM。禁止把 UB 中完整矩形直接 DataCopy 到原矩阵。

### 各算子附加语义

- Csyr2：使用 `start=(inc>=0?0:-(n-1)·inc)` 处理负步长；只更新普通对称操作区，不处理对角虚部特判。
- Cher2k/Cherk：按 N/C 选择乘积形式；对角完成更新后强制 `imag=0`，该写入属于操作区，不影响另一三角。
- 所有路径均不使用原子、不补全对称/Hermitian、不读取或恢复非操作区。

## 内存与同步

继续在同一 stream 上顺序执行，保持异步 API。多核按不重叠操作区或输出 tile 分工，避免同一元素重复写。UB 仅作临时计算区，不新增与 n/k 无关的持久 Device 缓存，不以整块原矩阵副本换取修复。

# 支持硬件与约束

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 / A3，DAV_2201 | √ |
| Ascend 950PR | 不修改，必要时仅做交叉回归 |

仅 complex64；保持公开 ABI、quick return、负步长和 N/C 语义；无 CPU fallback、无平行 API、无新增持久缓存。

# 可维可测分析

## 精度、bit 与性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 操作区精度 | 实部/虚部分别 `atol=rtol=2^-13`，匹配率≥0.99，最大误差满足动态 32 ULP 上限 | 任务书 §3.2.1 |
| 非操作区 | 另一三角及 padding 与输入逐 bit 一致，匹配率 100% | 任务书 §2.0、§3.2.2 |
| Csyr2 性能 | 512/1024/2048 分别不高于 7.54/9.72/19.05us | 任务书 §3.3 |
| Cher2k 性能 | 三主场景不高于 1458.91/5240.68/1081.90us | 任务书 §3.3 |
| Cherk 性能 | 三主场景不高于 738.09/2616.28/485.94us | 任务书 §3.3 |

Golden：Csyr2 按 Netlib `ssyr2` 语义自实现，Cher2k/Cherk 使用 Netlib BLAS 3.10.0 cblas。A2 与 A3 均执行精度和 bit 检查，910B3 执行性能门禁。

## 测试设计

1. 复用 `test_cases/{csyr2,cher2k,cherk}/` 原 CSV，不删减功能、精度、性能或负向用例。
2. 将 `check_untouched_bitwise.h` 接入三组 arch22 GTest：调用前保存完整 `ld×n`，调用后操作区走混合容差，另一三角和 padding 走 `memcmp`。
3. 追加 `TC_BIT_*`：Csyr2 22 条，Cher2k/Cherk 各 24 条；覆盖 UPPER/LOWER、紧凑与 padding ld、N/C、NaN payload、±Inf、±0、非规范数和随机有限值。
4. 覆盖 quick return、beta scaling、非法枚举、空指针、负步长、边界 n/k 和跨三角 tile，确保成功返回时非操作区始终不变。
5. 保留完整性能采样并记录 CANN、驱动、Netlib、A2/A3 型号；不通过减少用例或改变比较器达标。

## 兼容性与风险

公开接口不变，950 路径不改。主要风险是跨三角 tile 的写回遗漏、负步长寻址回归、Hermitian 对角置零误写和裁剪带来的性能下降。分别通过全量 bit 哨兵、原 CSV、对角专项和同协议性能 A/B 控制。设计阶段不宣称修复、精度或性能已经通过。

# 实施与交付计划

1. 先建立现有 arch22 精度和非操作区复现证据，确定具体越界写回点。
2. 按完整 tile/跨三角 tile 修复写回，保持公开接口和计算语义。
3. 运行原 CSV 与新增 bit 套件，修复后执行 A2/A3 精度、910B3 性能和交叉回归。
4. 同任务交付 Csyr2、Cher2k、Cherk 三者实现与测试，不拆分；提交设计文档、测试代码、自测报告和个人仓代码地址。
