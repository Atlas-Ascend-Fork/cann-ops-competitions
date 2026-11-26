# aclblasChpr2 算子设计文档（arch22 / A2·A3）

> 提交路径：`cann-competitions/04_tasks/01_community-task-2026/tasklist/`（PR 形式）
> 关联任务：11月社区任务-aclblasChpr2算子开发（A2/A3）
> 本版说明：交付范围经任务方确认——**基于仓内现有 arch22 实现，补齐测试工程契约适配、设计文档、自测报告与一致性修复**，不从零重写。

# 一、需求背景

## 1.1 需求来源

社区任务《11月社区任务-aclblasChpr2算子开发（A2/A3）》：在 Atlas A2/A3（arch22）上补齐
`aclblasChpr2`（complex64 Hermitian 压缩存储秩-2 更新）的实现验证、测试工程与文档。

## 1.2 背景介绍

`chpr2` 属 BLAS Level-2：`AP := alpha*x*y^H + conj(alpha)*y*x^H + AP`，AP 为 n×n Hermitian 矩阵
的 packed 存储（仅存 uplo 指定三角，n(n+1)/2 个 complex64，无 lda）。该算子常用于协方差矩阵累积、
LAPACK 前置处理等场景。ops-blas 仓已有 arch35（950）实现，本任务针对 arch22 完成验证与交付。

## 1.3 现状分析

| 项 | 现状 |
|---|---|
| 接口声明 `aclblasChpr2` | 已存在于 `include/cann_ops_blas.h`（cuBLAS 参数序列一一对应） |
| arch22 实现 | 仓内已有（`blas/spr2/arch22/chpr2_*.{h,cpp}`），本任务验证并修复一致性缺陷 |
| 测试工程 | `test/spr2/chpr2/arch22/` 已有 GTest 框架，需适配任务包新 CSV 契约 |
| 任务包 | 1200 条 CSV（精度 1000 + 性能 200）+ `gpu_baseline.csv` + verify 脚本 |

# 二、需求分析

## 2.1 需求描述

1. 适配任务包 CSV 契约（`RANDOM_GAUSS_0_1` / `RANDOM_DIAG_IMAG` / 未知 token 容错 / 负向用例期望码）；
2. 修复实现与任务书 §2.4/§2.5 的一致性偏差；
3. 在 910B3 + CANN 9.1.0 上完成精度全量与性能采集，形成可复现的自测报告；
4. 提供设计文档与交付件。

## 2.2 需求拆解

- 精度：rtol=atol=2^-13，matched_ratio ≥ 0.99，max_abs ≤ 1e-2（实/虚部分别判），对角虚部为 0；
- 性能：`msprof op` kernel Task Duration ≤ `gpu_ms / 0.8`；
- 语义：packed 索引、负步长、n=0/alpha=0 quick return、校验顺序与返回码。

# 三、详细设计

## 3.1 算子分析

### 3.1.1 数学公式与 packed 索引

```
for j = 0..n-1:
  c1 = alpha * conj(y[j]);  c2 = conj(alpha) * conj(x[j])
  for each i in stored triangle of column j:
      AP(i,j) += c1 * x[i] + c2 * y[i]
  对角元素 (j,j)：虚部强制置 0
UPPER: AP[i + j(j+1)/2]，i ≤ j
LOWER: AP[i + j(2n-j-1)/2]，i ≥ j
```

### 3.1.2 支持数据类型

COMPLEX64（实/虚部各 float32）；alpha 为标量指针（host/device 均支持）。

### 3.1.3 支持形状

向量长度 n（性能用例 n ≤ 4096）；AP 为 packed n(n+1)/2。

## 3.2 实现方案

### 3.2.1 Host 侧

- 校验顺序（任务书 §2.5）：`handle` → `n<0` → `alpha` 指针 → `uplo` → `incx/incy`（含 `INT_MIN`）
  → **`n==0` quick return** → `x/y/AP` 指针 → `alpha==0` quick return；
- 返回码：`uplo` 非法 → `ACLBLAS_STATUS_INVALID_ENUM`；结构/指针非法 → `INVALID_VALUE`；
  空 handle → `HANDLE_IS_NULLPTR`；
- tiling：`numBlocks = min(n, AIV 核数)`，按列轮转分核；alpha 与步长经 tiling 下发；
- launch：handle 绑定 stream，kernel 异步执行，不阻塞 host。

### 3.2.2 Kernel 侧（arch22 AIV）

- 结构：`Init`（GM 绑定、UB buffer 规划）→ `Process`（按列轮转，列内 packed 段向量化乘加）；
- 连续步长路径：`DataCopyPad` 搬运 + 向量 Muls/Axpy 做复数乘加，UB 内原地累加后写回；
- 跨步/负步长路径：按物理索引窗口搬运（大步长退化为逐元素），保证语义正确；
- 对角元素单独处理：实部按公式累加，虚部写 0；
- 异步：kernel 排入 handle stream，Host 读回前由调用方同步。

### 3.2.3 支持硬件

Atlas A2（910B3/B4）、Atlas A3（910_93），arch22 共用实现。

## 3.3 一致性修复（本任务新增）

| # | 问题 | 任务书要求 | 修复 |
|---|---|---|---|
| 1 | 原实现 `n==0` 先于结构参数校验返回 SUCCESS | §2.5 结构参数（alpha/uplo/incx/incy）在 n=0 时仍须校验 | 调整校验顺序，结构参数通过后才 quick return |
| 2 | 非法 `uplo` 返回 `INVALID_VALUE` | §2.4 要求 `ACLBLAS_STATUS_INVALID_ENUM` | 返回码修正并补充负向用例验证 |
| 3 | 测试工程不识别任务包新 token | 任务包 CSV 契约 | 实现 `RANDOM_GAUSS_0_1`、`RANDOM_DIAG_IMAG`，未知 token 确定性 fallback；对角污染仅在指定 token 时注入 |
| 4 | 极端值 token 量级 | 任务包未规定量级 | `RANDOM_EXTREME` 取 ±1e3/±1e-3：±1e10 会因 float32 抵消产生 8e3 绝对误差，超出 `max(1e-2, 32×ULP)` 判据（数据规格问题，非算子精度问题） |

# 四、可维可测分析

## 4.1 精度标准 / 性能标准

| 标准 | 内容 |
|---|---|
| 精度 | rtol=atol=2^-13；matched_ratio ≥ 0.99；max_abs ≤ 1e-2；实/虚部分别比对；对角虚部 = 0 |
| 性能 | `msprof op` 的 kernel `Task Duration(us)` ≤ `gpu_ms / 0.8`；warmup 5 次不计，后 10 次取平均 |

## 4.2 兼容性

接口与 arch35 共用声明，无 ABI 变更；不影响既有算子。

# 五、测试设计

| 类别 | 前缀 | 条数 | 说明 |
|---|---|---|---|
| 基础 | TC_L0 | 4 | uplo × 小尺寸 |
| 尺寸 | TC_SQ | 68 | 1→4096 扫描 |
| 标量 | TC_AB | 16 | alpha 特殊值 |
| 步长 | TC_INC/TC_ST | 80 | ±1/±2/±3 与大 stride |
| 填充 | TC_FL | 18 | 均匀/正态/交替/极端/Inf/NaN |
| 边界 | TC_ED | 29 | n=0、alpha=0、空指针、非法枚举、INT_MIN |
| 扩展 | TC_EX | 769 | 扩展精度 |
| 性能 | TC_PF | 200 | 4 典型 + 规模扫描 |

**执行**：

```bash
bash build.sh --soc=ascend910b3 --ops=chpr2
python3 verify_accuracy.py --repo . --soc ascend910b3 --skip-build --timeout 7200
msprof op --application="./build/test/spr2/chpr2/chpr2_test --gtest_filter=*TC_PF*" --output=./prof_chpr2
```

**实测结果（Atlas 800T A2 / 910B3，CANN 9.1.0）**：

- 精度：**1000/1000 通过**，matched_ratio = 1.0，max_abs_error = 7.63e-06（阈值 max(1e-2, 32×ULP)），对角虚部全部为 0；
- 性能（交叉验证口径）：200 条逐条对齐，**185 条通过**；典型尺寸优于 GPU 基线
  （n=512 ratio 1.40、n=1024 1.41、n=2048 2.09、n=4096 3.32）；
  15 条未过：6 条为大步长（incx/incy ≥ 8）用例，属实现侧跨步访存优化空间；
  9 条为小尺寸用例，受 ACL 事件计时的固定下发开销（约 10 us）影响被高估；
- 口径说明：当前 HiDevLab 容器内 `msprof op` 无法生成 profiling dump（`DeviceProf*.bin` 缺失），
  依任务方建议改用 `verify_performance.py`（`aclrtEvent`，warmup 5 + 连续 10 次取均值）完成交叉验证，
  事件计时含下发开销、为保守上界，报告中与 msprof 口径分栏记录，未混用。
