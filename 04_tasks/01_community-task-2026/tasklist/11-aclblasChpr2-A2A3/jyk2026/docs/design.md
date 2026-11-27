# aclblasChpr2 算子设计文档（Atlas A2/A3）

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：COMPLEX64（实部/虚部各 FLOAT32）
> 合入位置：`ops-blas/blas/spr2/arch22/`；测试：`ops-blas/test/spr2/chpr2/arch22/`
> 任务书：`aclblasChpr2_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasChpr2 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/spr2/arch22/`；测试代码 → `test/spr2/chpr2/arch22/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入（§4）。

## 背景介绍

### 算子功能定位

单精度复数 Hermitian 压缩存储秩-2 更新：`AP = alpha·x·y^H + conj(alpha)·y·x^H + AP`。x、y 为 n 元素复数向量，n×n Hermitian 矩阵以压缩（packed）格式存放于 AP——按列主序逐列无间隙打包、仅存 uplo 指定的一半三角、共 n(n+1)/2 个复数元素、无前导维度。两项贡献互为共轭转置，求和天然 Hermitian；对角元素虚部在计算后强制置 0。

### 业界对标

- 对标接口：cuBLAS `cublasChpr2`（语义参考 Netlib `chpr2`）。
- 有意差异：非法 uplo 枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（cuBLAS 返回 INVALID_VALUE）；其余结构参数非法返回 INVALID_VALUE。
- quick return：`n=0` 合法 no-op（不豁免结构校验）；`alpha=0` 时 AP 不变（不读 x/y 数值，指针仍须非空）。

### 工程形态

- 工程模式：Host C API + Ascend C kernel 直调（handle 绑定 stream，§2.2）。
- 代码落位：`blas/spr2/arch22/`。
- 仓内近亲：`blas/spr2/arch35/` 已有 `aclblasChpr2` 实现，可参考其压缩索引与秩-2 更新流水，本任务补齐 arch22 实现与测试。

### 接口规格

签名与 `include/cann_ops_blas.h`（约 L515）一致，无需新增声明：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* AP);
```

参数规格固定表（§2.4 逐行）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | - | - | 非空，否则 HANDLE_IS_NULLPTR | - |
| uplo | AP 存储三角模式 | attr int | - | {UPPER, LOWER}，非法→INVALID_ENUM | - |
| n | 方阵阶数 | attr int | - | ≥0；n=0 合法 no-op（不豁免结构校验） | - |
| alpha | 复数标量 | scalar | COMPLEX64 | 非空 | - |
| x/y | 输入向量 | tensor | COMPLEX64 | 步长任意非零 int（含负，反向遍历）；0/INT_MIN→INVALID_VALUE | 逻辑 n 元素 |
| AP | 输出，原地覆写 | tensor | COMPLEX64 | 仅 uplo 三角读写，对角虚部置 0 | n(n+1)/2 元素 |

### 数学公式

`AP = alpha·x·y^H + conj(alpha)·y·x^H + AP`。压缩索引（0 基）：UPPER 时列 j 元素 A(i,j)（i≤j）存于 `AP[i + j*(j+1)/2]`；LOWER 时列 j 元素 A(i,j)（i≥j）存于 `AP[i + j*(2*n-j-1)/2]`；未存一半由 Hermitian 共轭对称性隐含引用。

# 需求分析（required）

## 需求描述

以 Ascend C 在 arch22 上实现 packed 秩-2 更新：每列 j 的更新 = alpha·x(j)·y^H 的第 j 列 + conj(alpha)·y(j)·x^H 的第 j 列 + AP 第 j 列，支持任意非零步长（含负）与完整 quick return 语义，精度达 FLOAT32 分量阈值，性能达 GPU kernel 标杆 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | 秩-2 两项互为共轭转置，逐列计算量 = 2 次复数缩放 + 列内 FMA；未存三角免费（对称性）；对角虚部计算后置 0（无需特殊推导） |
| 2 | 接口原型 | `include/cann_ops_blas.h` L515 已有声明，无需改公共头 |
| 3 | 参数规格 | §2.4 固定表；AP 无前导维，x/y 按步长访问 |
| 4 | 支持数据类型 | COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → arch22（ascend910b3），落 `blas/spr2/arch22/` |
| 6 | 精度要求 | FLOAT32 分量 rtol=atol=2^-13、matched_ratio≥0.99、max_abs≤1e-2 或 32ULP（§3.2，可溯源） |
| 7 | 性能预期 | kernel-to-kernel：`msprof op` Task Duration ≤ `gpu_ms` × 1.25（0.8 倍 GPU 标杆，§3.3，可溯源） |
| 8 | 代码架构选型 | 列级并行（每核负责若干列的秩-2 更新，列间无依赖）+ AP 顺序打包读写；可行性依据：arch35 同构实现已在仓。已排除死路——解包成全矩阵再更新（浪费空间且破坏原地语义）、按行（跨步）组织（AP 列主序打包下行访存不连续） |
| 9 | 评测来源 | golden = CPU cblas `cblas_chpr2`（Netlib `chpr2`）随测试工程提供；性能 baseline = `gpu_baseline.csv`（cuBLAS cublasChpr2 kernel-only） |

# 详细设计（required）

## 算子分析

### 数学公式

见"数学公式"节。展开到列 j（0 基）：AP 列 j 的元素 A(i,j) 更新为

`A(i,j) += alpha·x(i)·conj(y(j)) + conj(alpha)·y(i)·conj(x(j))`（i 在 uplo 三角内），i=j 时结果虚部置 0。

### 支持数据类型

COMPLEX64（实部/虚部各 FLOAT32）；标量 alpha 同型。

### 支持形状

| 参数 | 形状约束 | 退化 |
| --- | --- | --- |
| AP | n(n+1)/2 个复数元素，列主序无间隙打包 | n=0 长度 0 |
| x/y | 逻辑 n 元素，物理至少 1+(n-1)·abs(inc) | 步长可负（反向遍历）；0/INT_MIN 非法 |
| quick return | n=0；alpha=0（AP 不变，x/y 指针仍须非空） | — |

## 算子实现

### 实现方案

#### 总体架构与数据流

```
host: 校验(handle→n→uplo→incx/incy→alpha 指针) → n=0 quick return
      → x/y/AP 指针校验 → alpha=0 quick return → tiling → launch
数据流: GM(AP 列块 + x/y 向量) → UB → 逐列计算两秩-1 项并累加
        → 对角虚部清 0 → GM(AP) 原地写回
```

| kernel 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| 主路径 | n>0 且 alpha≠0 | 逐列秩-2 更新 + 对角虚部置 0 |
| quick return | n=0 或 alpha=0 | 不 launch（alpha=0 时 AP 完全不变） |

#### host 侧设计

- **分核策略**：按列号区间满核均分（列 j 的更新长度 = uplo 三角内该列元素数，UPPER 为 j+1、LOWER 为 n−j），负载按三角形斜率单调，采用等和分段（按 n(n+1)/2 均分）而非等列数均分；
- **数据分块和内存优化策略**：x/y 整体驻留 UB（n 元素向量，规模小）；AP 按列块搬入/写回，double buffer；两秩-1 项在寄存器内合并为一次复数乘加（`x(i)·conj(alpha·y(j)) + y(i)·conj(alpha·x(j))` 形态先在 host 端折叠标量，kernel 每元素 2 乘 2 加）；
- **tilingkey 规划策略**：key 由 {uplo, 步长单位性, 尾块标志} 编码：incx=incy=±1 走连续向量化，否则 gather；
- **数据检测**：按 §2.5 校验顺序：handle 判空 → n<0 → uplo → incx/incy（非零且非 INT_MIN）→ alpha 指针 → n=0 quick return → x/y/AP 指针 → alpha=0 quick return；不做 Device 元素级扫描。

#### kernel 侧设计

- Init：解析 tiling（列区间、AP 列偏移）、按 key 选择向量化/gather 分支；
- Process：CopyIn（AP 列块、x/y）→ Compute（两项秩-1 FMA 累加）→ CopyOut（对角虚部强制置 0 后原地写回）；
- workspace：无额外 GM workspace。

#### 性能优化要点（终态）

- 优化项→预期效果：标量折叠到 host 端→kernel 每元素固定 2 次复数乘加，无逐元素标量乘；AP 打包顺序读写→无索引抖动；等和分核→各核负载均衡。实测按 §3.3 口径（warmup 5、有效采样 10、msprof op）回填自测报告。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2（800T A2/910B3） | ✔（本任务目标） |
| Atlas A3 | ✔（本任务目标） |
| Atlas 950PR | arch35 已有实现，不涉及 |

## 算子约束限制

- 仅支持 packed 列主序压缩布局与步长语义访问，不支持广播、任意非连续布局（§2.5）；
- 不使用 AutoContiguous/ViewCopy；AP 原地覆写；
- incx/incy 不支持 0 与 INT_MIN；
- 非确定性：浮点累加顺序随实现变化，验收按阈值判定；异步执行语义同 §2.5。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 分量按 FLOAT32：rtol=atol=2^-13，matched_ratio≥0.99，max_abs≤1e-2 或 32ULP；golden=cblas_chpr2，实部/虚部分别比对 | §3.2 + mixed_tolerance_standard |
| 性能标准 | msprof op kernel Task Duration ≤ gpu_ms×1.25（cuBLAS cublasChpr2 kernel-only 基线） | §3.3 |

## 测试概览

- 用例分类：精度全量（CSV 驱动，ATK 框架）+ 性能子集（分层按规模档）+ 负向（非法枚举、n<0、零步长/INT_MIN、空指针；含 n=0 叠加非法结构参数不豁免用例）；
- 特殊场景：n=0；alpha=0 且 x/y 含 NaN（不读数值）；UPPER/LOWER 全覆盖；正负步长组合；对角虚部置 0 验证；
- 性能口径：msprof op 正式判据（warmup 5 + 采样 10 取平均）；
- 复现入口：`bash build.sh --ops=chpr2 --soc=ascend910b3`，自验包 readme 步骤。

## 兼容性分析

- 新增 arch22 路径不修改公共头（声明已存在），不影响 arch35 及其他产品线；
- `blas/spr2/arch22/` 新增入口与测试目录独立注册，无兼容性风险。