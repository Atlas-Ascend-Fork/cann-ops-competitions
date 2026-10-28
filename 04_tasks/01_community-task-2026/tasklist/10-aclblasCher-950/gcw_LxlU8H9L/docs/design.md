# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务（2026 年 10 月）：在 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）为 Ascend 950PR（arch35）补齐单精度复数 Hermitian 秩 1 更新算子 `aclblasCher`，以 Ascend C 编程语言开发，验收通过后合入仓库（算子目录 `blas/syr/arch35/`，测试目录 `test/syr/cher/arch35/`）。

## 背景介绍

### aclblasCher 算子实现补齐

基于 ops-blas 仓已有 BLAS 工程框架与同族算子的设计范式，使用 Ascend C 编程语言实现 `aclblasCher` 句柄式 BLAS 接口。

参考实现与相关路径：

- 同族参考算子（仓内同目录族，复数对称秩 1 更新）：`blas/syr/arch35/`（csyr 系）及其测试工程 `test/syr/csyr/`；同族实数算子 `aclblasSsyr`
- 对标基线接口：cuBLAS `cublasCher`；golden 参考实现：Netlib BLAS `cher`（https://www.netlib.org/blas/cher.f ，测试工程以 `cblas_cher`（Netlib CBLAS，CblasColMajor）单标杆生成 golden）
- 接口声明：`include/cann_ops_blas.h`（此前无 `aclblasCher` 声明；仓内已有的 `aclblasCherk` 为 L3 Hermitian 秩 k 更新算子，与本算子不同，勿混淆。本任务新增声明，与其他产品线共用，禁止 950PR 私有平行接口）

### 仓内现状分析

通过对 ops-blas 仓的功能分析，SYR 族当前支持能力如下：

| 参数 | 参数含义 | 数据类型 | 仓内支持情况 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| uplo | 三角存储模式 | 枚举 | ssyr（实数）/csyr（复数对称）支持 | UPPER/LOWER | - |
| n | 矩阵阶数 | int | 支持 | n ≥ 0 | A 为 n×n 方阵 |
| alpha | 标量乘数 | 标量 | ssyr 实数 / csyr 复数 | Hermitian 语义要求 alpha 为**实数** float | - |
| x | 向量 | tensor | ssyr/csyr 支持 | incx ≠ 0，支持负步长 | 逻辑长度 n |
| A | Hermitian 矩阵 | tensor | ssyr/csyr 支持 | 列主序，仅 uplo 三角被引用 | lda×n，lda ≥ max(1,n) |

**complex64 Hermitian 秩 1 更新（cher）在仓内完全缺失**。与同族 `aclblasCsyr` 相比，cher 的核心差异有三点：① 更新公式中 x(j) 取**共轭**（`A(i,j) += alpha·x(i)·conj(x(j))`）；② alpha 为**实数** float 标量（Host 内存，非复数）；③ **对角元素虚部在更新后强制置 0**（Hermitian 语义，`A(i,i) += alpha·|x(i)|²` 为纯实数）。

### aclblasCher 算子功能分析

计算公式（列主序，原地更新）：

- `A = alpha * x * x^H + A`，即 uplo 指定三角内 `A(i,j) += alpha * x(i) * conj(x(j))`
- 对角元素：`A(i,i).real += alpha * |x(i)|²`，`A(i,i).imag = 0`（更新后强制清零）

输入：handle、uplo（UPPER/LOWER）、阶数 n、实数标量 alpha（float，Host 指针）、复数向量 x（逻辑长度 n，物理长度 ≥ 1+(n-1)·|incx|）、步长 incx（任意非零，含负步长反向遍历）、Hermitian 矩阵 A（列主序，lda×n）。

输出：A（原地更新；仅 uplo 指定三角（含对角）被引用和更新，另一三角与 lda padding 行不读不写）。

支持数据类型：COMPLEX64（实部/虚部各 float32）+ FLOAT32 实数 alpha。

支持广播：不涉及。

特殊语义：n=0 或 alpha=0.0 为合法 no-op（quick return，返回 SUCCESS 且不更新 A）；参数合法性校验采用仓内 `aclblasCsyr` 口径。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `aclblasCher` 算子：complex64 数据类型（alpha 为 float 实数），uplo 两填充模式全组合，任意 n（0~4096）、紧凑与 padding 前导维、正负步长（±1/±2/±3）、alpha 特殊值（0.0/±1.0/大值）、Inf/NaN 特殊值、边界与负向用例，精度满足生态算子开源精度标准，性能满足任务书判据（平均单次耗时 ≤ GPU 标杆的 2.5 倍，即倍率 ≥ 0.4）。

## 需求拆解

1. 支持 complex64 单一数据类型、uplo 全组合、任意前导维与正负步长、n=0/alpha=0 快速返回及全量非法参数校验（口径对齐任务书 §2.5 与 cuBLAS）
2. Hermitian 语义三点落地：x(j) 共轭、alpha 实数化、对角虚部更新后置 0；未引用三角与 lda padding 行不读不写（以哨兵值验证）
3. 精度满足生态开源精度标准（实部/虚部分别按 FLOAT32 判定；Inf/NaN 按任务书等效规则判定）
4. 性能满足 200 条性能用例判据（msprof 采集 kernel 平均单次耗时 ≤ 标杆 / 0.4）

# 详细设计（required）

## 算子分析

### 数学公式

- 三角内元素：`A(i,j) += alpha * x(i) * conj(x(j))`，按实/虚分解计算
- 对角元素：`A(i,i).real += alpha * |x(i)|²`，`A(i,i).imag = 0`
- 负步长：逻辑 x(i) 映射物理存储 `(n−1−i)·|incx|`（Netlib 语义反向遍历）

### 支持数据类型

COMPLEX64（float32 实部 + float32 虚部）；alpha 为 FLOAT32 实数标量（Host 指针）。

### 支持形状

A 为 n×n 方阵（列主序，存储 lda×n，lda ≥ max(1,n)）；x 逻辑长度 n；n 上限 4096（单用例 host 侧内存预算 ≤ 512MB）。

## 算子实现

### 实现方案

总体：**单 kernel 直调（AIV 向量核）**。cher 为带宽受限算子（rank-1 更新每元素约 1 读 1 写 A，算术强度低），无需 Cube/矩阵核与多 kernel 流水；实现结构对齐同族 `aclblasCsyr`（同目录、同 launch 协议），差异收敛在共轭、实 alpha、对角清零三点。

#### host侧设计：

**参数校验（对齐仓内 csyr 口径，顺序不可变）**：handle 判空 → n<0 → n=0 快速返回 SUCCESS（不再校验其余参数）→ uplo 非法返回 INVALID_ENUM → lda ≥ max(1,n) → incx ≠ 0 → alpha 指针非空 → alpha=0.0 no-op 返回 SUCCESS（x/A 允许为空）→ x/A 判空。

alpha 为 Host 内存实数指针，无 device 标量解析分支（对比 csyr 相应路径删除，host 路径更短）。

**tiling 策略**：

- 分核：按设备实查 AIV 核数与 n 取小值（不硬编码核数）；
- 按 n 规模选择线程档位（小/大两档，对齐仓内同族常量风格）；
- tiling 以 POD 结构体按值传入 kernel（n、lda、uplo、alpha、incx、线程数），kernel 侧按运行时条件分派路径，无需 TilingKey 模板分支。

#### kernel侧设计：

kernel 按（地址对齐性、incx、n/lda 形状、alpha 是否为 1）分派快慢路径：

- **小形状与特殊形状快路径**：满足紧凑存储、单位步长、8 字节对齐等条件的形状走向量化整块读写路径，压低小尺寸单次耗时（小 n 档预算最紧，约 7us）；alpha=1 时省略标量乘；
- **通用路径**：incx≠1（含全部负步长反向索引）、非对齐地址或超界形状走逐元素标量更新，保证全用例正确性覆盖；
- **对角处理**：所有路径的对角元素统一为实部累加 `alpha·|x(i)|²`、虚部赋值置 0；
- **写保护**：所有路径仅计算并写入 uplo 三角内元素（含对角），三角外与 lda padding 行无任何读/写访问，由索引域裁剪结构性保证；
- **数值口径**：乘加逐次独立舍入，与仓内同族算子的 Inf/NaN 传播口径一致。

aclblasCher 的整体流程：

```
host: 校验(csyr口径) → alpha==0 no-op → tiling → 单 kernel 直调(绑定 stream)
kernel(AIV): 按对齐/incx/形状/alpha 分派快慢路径
            → 三角内 A(i,j) += alpha·x(i)·conj(x(j))，对角 imag 置 0
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（Ascend 950PR，arch35） | √ |

## 算子约束限制

- 数据类型仅 COMPLEX64（alpha 为 FLOAT32 实数标量）；A 原地更新，x 与 A 独立，不支持参数重叠；
- 不支持超出 incx/lda 语义的非连续内存访问；不涉及广播；n 为运行时入参，不要求动态 shape；
- 未引用三角与 lda padding 行不读不写（哨兵值验证）；对角虚部更新后强制为 0；
- 异步执行：依赖 aclblasSetStream 绑定 stream，读回 Device 结果前须同步 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 实部/虚部分别按 FLOAT32 判定：atol=rtol=2⁻¹³（1.2207e-4）、matched_ratio ≥ 0.99、max_abs_error ≤ max(1e-2, 32 ULP)；golden 为 `cblas_cher`（Netlib CBLAS，CblasColMajor）单标杆；比对范围仅 uplo 三角（含对角），对角虚部按 \|imag\| ≤ atol 断言为 0；Inf/NaN 按任务书等效规则判定；未引用三角与 padding 行哨兵写保护；另按仓内测试工程惯例配置 MERE/MARE 判定 | 生态算子开源精度标准（https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md ） |
| 性能标准 | 200 条性能用例（TC_PF）：msprof op 采集 kernel 平均单次耗时（warmup 5 + 有效采样 ≥10）≤ GPU 标杆 / 0.4；四个典型 case（n=512/1024/2048/4096）均需达标 | 任务书 §3.3 性能要求与 gpu_baseline.csv（H100 基线，倍率 0.4） |

## 兼容性分析

新算子（仓内原无 `aclblasCher` 声明），仅在 `include/cann_ops_blas.h` 新增声明供各产品线共用，不修改任何既有接口与算子行为，不涉及兼容性分析。
