# aclblasChpmv 算子设计文档（Atlas A2/A3）

> 目标芯片：Atlas A2/A3 系列（arch22 / dav-2201，SOC=ascend910b3）｜CANN 9.1.0
> 数据类型：COMPLEX64（实部/虚部各 FLOAT32）
> 合入位置：`ops-blas/blas/spmv/arch22/`；测试：`ops-blas/test/spmv/chpmv/arch22/`
> 任务书：`aclblasChpmv_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）

# 需求背景（required）

## 需求来源

- 任务名称/批次：2026 年 11 月社区任务——aclblasChpmv 算子开发（Atlas A2/A3，arch22）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → `blas/spmv/arch22/`；测试代码 → `test/spmv/chpmv/arch22/`（任务书 §1、§5）。
- 验收流程：社区任务 IT 系统提交四件套交付件；设计文档以 MR 形式提交至 `cann/cann-ops-competitions` tasklist 通过评审后合入（§4）。

## 背景介绍

### 算子功能定位

单精度复数 Hermitian 压缩存储矩阵-向量乘：`y = alpha·A·x + beta·y`。A 为 n×n Hermitian 复数矩阵，以压缩（packed）格式存放于 AP——按列主序逐列无间隙打包、仅存 uplo 指定的一半三角、共 n(n+1)/2 个复数元素、无前导维度参数。

### 业界对标

- 对标接口：cuBLAS `cublasChpmv`（语义参考 Netlib `chpmv`）。
- 有意差异：非法 uplo 枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`（cuBLAS 返回 INVALID_VALUE），其余结构参数非法返回 INVALID_VALUE；校验顺序与 cuBLAS 错误检查一致（§2.5）。
- Hermitian 语义：对角元素虚部不被引用，实现与 golden 均只取对角实部。

### 工程形态

- 工程模式：Host C API + Ascend C kernel 直调（handle 绑定 stream，§2.2）。
- 代码落位：`blas/spmv/arch22/`，实现 arch22 版本。
- 仓内近亲：`blas/spmv/arch35/` 已有 `aclblasChpmv` 实现，可参考其压缩索引计算与流水结构，本任务补齐 arch22 实现与测试。

### 接口规格

签名与 `include/cann_ops_blas.h`（约 L161）一致，无需新增声明：

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

参数规格固定表（§2.4 逐行）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | - | - | 非空，否则 HANDLE_IS_NULLPTR | - |
| uplo | AP 存储三角模式 | attr int | - | {UPPER, LOWER}，非法→INVALID_ENUM | - |
| n | 方阵阶数 | attr int | - | ≥0；n=0 合法 no-op（不豁免结构校验） | - |
| alpha/beta | 复数标量 | scalar | COMPLEX64 | 非空 | - |
| AP | Hermitian 压缩矩阵 | tensor | COMPLEX64 | 对角虚部忽略 | n(n+1)/2 元素 |
| x/y | 输入/输出向量 | tensor | COMPLEX64 | y 原地覆写；y 可写区与 x/AP 不重叠 | 逻辑 n 元素 |
| incx/incy | 元素步长 | attr int | - | 任意非零 int（含负步长，反向遍历）；0→INVALID_VALUE | - |

### 数学公式

`y = alpha·A·x + beta·y`。压缩索引（0 基）：UPPER 时列 j 元素 A(i,j)（i≤j）存于 `AP[i + j*(j+1)/2]`；LOWER 时列 j 元素 A(i,j)（i≥j）存于 `AP[i + j*(2*n-j-1)/2]`；未存另一半由 A = A^H 共轭对称隐含引用。

# 需求分析（required）

## 需求描述

以 Ascend C 在 arch22 上实现 packed Hermitian 矩阵-向量乘：按 uplo 两种压缩索引访存 AP，支持任意非零步长（含负）与完整 quick return 语义，精度达 FLOAT32 分量阈值（rtol/atol=2^-13，matched_ratio≥0.99），性能达 GPU kernel 标杆 0.8 倍以上。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | A·x 中每行只需 uplo 三角元素；未存三角由共轭对称免费获得；对角虚部免读（只取实部） |
| 2 | 接口原型 | `include/cann_ops_blas.h` L161 已有声明，无需改公共头 |
| 3 | 参数规格 | §2.4 固定表；AP 无前导维、x/y 按步长访问 |
| 4 | 支持数据类型 | COMPLEX64 |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → arch22（ascend910b3），落 `blas/spmv/arch22/` |
| 6 | 精度要求 | FLOAT32 分量 rtol=atol=2^-13、matched_ratio≥0.99、max_abs≤1e-2 或 32ULP（§3.2，可溯源） |
| 7 | 性能预期 | kernel-to-kernel：`msprof op` Task Duration ≤ `gpu_ms` × 1.25（0.8 倍 GPU 标杆，§3.3，可溯源） |
| 8 | 代码架构选型 | 行分块并行（每个输出分量 y(i) 的累加独立）+ AP 列式打包顺序读；可行性依据：arch35 同构实现已在仓。已排除死路——把 AP 解包成全矩阵再乘（浪费 n(n+1)/2→n² 空间与访存） |
| 9 | 评测来源 | golden = CPU cblas `cblas_chpmv`（Netlib `chpmv`）随测试工程提供；性能 baseline = `gpu_baseline.csv`（cuBLAS cublasChpmv kernel-only） |

# 详细设计（required）

## 算子分析

### 数学公式

见"数学公式"节。核内计算按行 i 组织：`y(i) = alpha·Σ_j A(i,j)·x(j) + beta·y(i)`，其中 A(i,j) 从 AP 的压缩索引取值，j>i（或 i>j，按 uplo）取共轭对称项 `conj(A(j,i))`。

### 支持数据类型

COMPLEX64（实部/虚部各 FLOAT32）；标量 alpha/beta 同型。

### 支持形状

| 参数 | 形状约束 | 退化 |
| --- | --- | --- |
| AP | n(n+1)/2 个复数元素，列主序无间隙 | n=0 时长度 0 |
| x/y | 逻辑 n 元素，物理至少 1+(n-1)·abs(inc) | 步长可为负（反向遍历）；n=0 不引用 |
| quick return | alpha=(0,0) 且 beta=(1,0) 直接返回 | beta=(0,0) 不读 y 旧值；alpha=(0,0) 且 beta≠(1,0) 只做缩放 |

## 算子实现

### 实现方案

#### 总体架构与数据流

```
host: 校验(handle→n→uplo→incx/incy→alpha/beta 指针) → n=0 quick return
      → AP/x/y 指针校验 → alpha=(0,0)&&beta=(1,0) quick return → tiling → launch
数据流: GM(AP 打包块 + x 向量块) → UB → 每核累加若干行 y(i)（压缩索引顺序读 AP）
        → beta 折算 → GM(y) 原地写回
```

| kernel 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| 主路径 | n>0 且非 quick return | 行分块计算 alpha·A·x 累加 + beta·y |
| beta 缩放路径 | alpha=(0,0) 且 beta≠(1,0) | 仅 y = beta·y，不读 AP/x |

#### host 侧设计

- **分核策略**：按 y 的行区间满核均分（大 n）或向量级并行（小 n 单核多段），压缩 AP 的列前缀索引在 tiling 期算好传入，避免核内三角函数式重复计算；
- **数据分块和内存优化策略**：AP 按列块搬入 UB（打包顺序即访存顺序，天然连续）；x 分块按行区间对应的列范围裁剪；double buffer 覆盖 AP/x 搬运与 FMA；
- **tilingkey 规划策略**：key 由 {uplo, 步长=±1 与否, 尾块标志} 编码：单位步长走连续向量化路径，非单位/负步长走 gather 路径；
- **数据检测**：按 §2.5 校验顺序：handle 判空 → n<0 → uplo → incx/incy 非零 → alpha/beta 指针 → n=0 quick return → AP/x/y 指针 → alpha=(0,0)&&beta=(1,0) quick return；不做 Device 元素级扫描。

#### kernel 侧设计

- Init：解析 tiling（行区间、AP 列偏移表）、按 key 选择向量化/gather 分支；
- Process：CopyIn（AP 列块、x 块）→ Compute（共轭对称项 conj 组装 + FMA 累加，对角元素只取实部）→ CopyOut（beta 折算后原地写 y）；
- workspace：无额外 GM workspace，全部中间量在 UB。

#### 性能优化要点（终态）

- 优化项→预期效果：AP 顺序打包读→访存连续、无索引抖动；压缩存储省一半访存（n(n+1)/2 vs n²）；单位步长向量化 + double buffer→带宽打满。实测按 §3.3 口径（warmup 5、有效采样 10、msprof op）回填自测报告。

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2（800T A2/910B3） | ✔（本任务目标） |
| Atlas A3 | ✔（本任务目标） |
| Atlas 950PR | arch35 已有实现，不涉及 |

## 算子约束限制

- 仅支持 packed 列主序压缩布局与步长语义访问，不支持广播、任意非连续布局（§2.5）；
- 不使用 AutoContiguous/ViewCopy；y 原地覆写且与 x/AP 不重叠；
- 非确定性：浮点累加顺序随实现变化，验收按阈值判定；异步执行语义同 §2.5。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 分量按 FLOAT32：rtol=atol=2^-13，matched_ratio≥0.99，max_abs≤1e-2 或 32ULP；golden=cblas_chpmv（CblasColMajor），实部/虚部分别比对 | §3.2 + mixed_tolerance_standard |
| 性能标准 | msprof op kernel Task Duration ≤ gpu_ms×1.25（cuBLAS cublasChpmv kernel-only 基线）；[PERF] avg_us 仅交叉验证 | §3.3 |

## 测试概览

- 用例分类：精度全量 1000 条；性能子集 200 条（small/medium/large 分层，单位步长为主，少量 ±大步长与负步长）；负向覆盖非法枚举、n<0、零步长、空指针（含 n=0 叠加非法结构参数，不豁免）；
- 特殊场景：对角虚部非零输入（验证只取实部）、alpha=(0,0)&&beta=(1,0) quick return 不读 x、beta=(0,0) 且 y=NaN（不读旧值）、UPPER/LOWER 全覆盖；
- 性能口径：msprof op 正式判据（warmup 5 + 采样 10 取平均）；
- 复现入口：`bash build.sh --ops=chpmv --soc=ascend910b3`，自验包 readme 步骤。

## 兼容性分析

- 新增 arch22 路径不修改公共头（声明已存在），不影响 arch35 及其他产品线；
- `blas/spmv/arch22/` 内新增入口对既有同目录算子无行为改动；测试目录 `test/spmv/chpmv/arch22/` 独立注册构建目标；
- 无兼容性风险。