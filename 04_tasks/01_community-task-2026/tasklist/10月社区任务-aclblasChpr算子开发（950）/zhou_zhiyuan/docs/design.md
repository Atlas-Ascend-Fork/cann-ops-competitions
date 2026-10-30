# aclblasChpr 算子设计文档（Ascend 950PR）

# 需求背景（required）

## 需求来源

- 社区任务：[10月社区任务-aclblasChpr算子开发（950）](https://www.hiascend.com/activities/task-center/details/f783b3800ad744ef8af7f50ba237de04?menu=tasks)。
- 开发者 / 团队目录：`zhou_zhiyuan`。
- 实现仓库：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，目标架构为 `arch35`。
- 需求依据：任务附件 `aclblasChpr_Atlas950PR_task_doc.md`、`test_cases/README.md` 及[任务讨论帖 #53](https://gitcode.com/cann/ops-blas/discussions/53)中的评审澄清。
- 本文为开发前设计，性能优化方案及阈值须经实现、自测验证；不将环境连通性检查或其他架构的结果视为本算子验收结果。

## 背景介绍

### aclblasChpr 算子实现优化

`Chpr` 是 BLAS Level-2 Hermitian Packed Rank-1 Update：使用单精度复数向量，对按列压缩存储的 Hermitian 矩阵进行秩 1 原地更新。它适用于复数协方差等计算中只需保存一个三角区域的场景。

本任务不是将 TBE 图算子迁移至 ACLNN，而是补齐 ops-blas 已有 `aclblasChpr` API 的 Ascend 950PR 实现。采用 Ascend C kernel 直调，通过 BLAS handle 绑定的 stream 异步执行；不新增 ACLNN 接口、算子图注册或 950PR 私有平行 API。

标杆分工如下：

| 标杆 | 用途 |
| --- | --- |
| cuBLAS `cublasChpr` | 核心功能、参数顺序及 packed/步长语义参考 |
| Netlib BLAS 3.10.0 `cblas_chpr` / `chpr` | 精度 golden、Hermitian 对角及特殊值执行语义参考 |
| 附件 `gpu_baseline.csv` | H100 性能比较依据，共 200 条基线 |

### aclblasChpr 算子实现现状分析

以 ops-blas 提交 `456b3487f939920a103a4f3f1a7d7636c7c94c17` 为设计时的代码参考：

- `include/cann_ops_blas.h` 已声明 `aclblasChpr`；`include/cann_ops_blas_common.h` 定义 `aclblasComplex`，实部、虚部各为一个 `float`。
- `blas/hpr/arch22/` 已有 host、kernel 和 tiling 实现；当前没有 `blas/hpr/arch35/` 实现。`blas/hpr/README.md` 中 Ascend 950PR 尚标为不支持。
- `test/hpr/chpr/` 已有参数、golden 和 arch22 GTest 工程，可复用公共测试设施，但不能直接视为覆盖了本批任务附件及 arch35 验收要求。
- `blas/spr/arch35/` 可参考 packed 寻址、SIMT 及 UB 缓存思路；`blas/gerc/arch35/` 可参考 complex64 交错数据的 MicroAPI SIMD 运算及直调框架。复用工程模式，不照搬不同算子的参数检查或特殊值规则。

# 需求分析（required）

## 需求描述

在 Ascend 950PR、CANN 9.1.0 上实现 `aclblasChpr`，支持 FLOAT32 实数 alpha、COMPLEX64 的 x/ap、上下三角 packed 存储、非零正负步长，以及任务规定的异常和 quick-return 行为。完成可复现的功能、精度、性能自测，并将实现和测试分别提交至 ops-blas 指定目录。

## 需求拆解

1. 复用公共 API，新增 `blas/hpr/arch35/` 实现，保持其他产品线接口及实现不变。
2. 正确处理 packed 索引、原地更新、负步长、对角虚部及 Inf/NaN。
3. 严格实现 handle 优先的参数校验与两类 no-op；无效参数不得发射 kernel 或修改 ap。
4. 为连续输入设计 AIV SIMD 快路径，为一般步长及小规模提供 SIMT 通用路径，不因快路径容量限制而收窄 API。
5. 接入附件全部 1000 条精度、200 条性能用例，补充空 handle、混合分布和内存边界测试。
6. 精度实部/虚部分别满足 FLOAT32 混合容差；每个性能 case 的 NPU 平均 kernel 耗时不高于附件 H100 基线的 2.5 倍。
7. 提交设计文档、自测代码、报告和日志；测试通过后按流程申请实现代码合入。

# 详细设计（required）

## 算子分析

### 数学公式

```text
A ← alpha × x × xᴴ + A
```

令 `x_i = r_i + i*s_i`、`x_j = r_j + i*s_j`，对被存储的非对角元素：

```text
real(A_ij) ← real(A_ij) + alpha × (r_i*r_j + s_i*s_j)
imag(A_ij) ← imag(A_ij) + alpha × (s_i*r_j - r_i*s_j)
```

实际计算采用与 Netlib 相近的分组：先计算列常量 `tReal = alpha*r_j`、`tImag = -alpha*s_j`，再完成 `x_i × (tReal + i*tImag)` 加到原值。对角按 `oldReal + (r_j*tReal - s_j*tImag)` 的分组仅更新实部，并将虚部显式写为 `0.0f`；不能依赖虚部相减抵消为零，也不能让输入对角虚部参与实部运算。

为保持特殊值语义，保留 Netlib 中 `x_j == 0+0i` 时跳过该列非对角更新、仅规范对角虚部的行为，避免无条件计算 `0×Inf/NaN` 污染原本有限的结果。`alpha=0` 的 host no-op 优先于所有矩阵计算，此时 ap 完全不写，包括对角虚部。

### 支持数据类型与接口

```cpp
aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha,
    const aclblasComplex* x, const int incx,
    aclblasComplex* ap);
```

| 参数 | 存储位置 | 类型、形状及语义 |
| --- | --- | --- |
| handle | Host | 有效的 ops-blas 句柄，携带 stream |
| uplo | Host | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | Host | int，`n >= 0` |
| alpha | Host | `const float*`，实数标量；本任务只要求 host pointer mode |
| x | Device | 只读 complex64，逻辑长度 n；n>0 时物理长度至少 `1+(n-1)*abs(incx)` |
| incx | Host | 复数元素单位，除 0 和 INT_MIN 外支持正负 int 步长 |
| ap | Device | complex64，`n*(n+1)/2` 个元素，输入旧值并原地更新 |

### 支持形状及 packed 映射

全部采用 0-based 列主索引，无 lda 参数。令 `M = n*(n+1)/2`：

| 模式 | 列 j 的 packed 基址 B(j) | 有效行 i | packed 下标 |
| --- | --- | --- | --- |
| UPPER | `j*(j+1)/2` | `0 <= i <= j` | `B(j)+i` |
| LOWER | `j*(2*n-j+1)/2` | `j <= i < n` | `B(j)+(i-j)` |

另一半矩阵由共轭对称性隐含，不分配、不访问、不更新。n 为运行时入参，不要求图级 dynamic shape 或超出 incx 语义的非连续 Tensor，不涉及广播。

向量逻辑位置 i 对应的物理位置为：

```text
incx > 0: p(i) = i * incx
incx < 0: p(i) = (n - 1 - i) * (-int64_t(incx))
```

先提升到 64 位再进行乘法、取负和 `2*n` 运算。M、列基址、步长乘积、总字节数均采用安全的 64 位计算，检查元素下标至字节偏移的溢出，不截断为 int；不可表示的地址范围返回 `ACLBLAS_STATUS_INVALID_VALUE`，不下发不安全访问。实际分配长度由调用方保证，不额外设定 n<=4096 等 API 限制。

## 算子实现

### 实现方案

总体流程：

```text
公共 aclblasChpr API
  → handle 判空
  → n 检查 / n=0 quick return
  → uplo、incx、alpha/x/ap 检查
  → 读取 host alpha / alpha=0 no-op
  → 安全地址范围检查、生成 tiling
  → 在 handle->stream 下发 AIV kernel
      ├─ 连续输入且满足 UB 容量/规模条件：SIMD packed-tile 路径
      └─ 小规模、一般正负步长或容量不足：SIMT 通用路径
  → 异步返回；调用方同步 stream 后读取结果
```

#### 3.2.1 host 侧设计

**1. 参数校验和返回值**

按任务书并结合讨论帖评审澄清，固定顺序：

1. `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，优先于 n 及其他参数。
2. `n < 0`：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n == 0`：直接返回 `ACLBLAS_STATUS_SUCCESS`，不读取 alpha，不检查 uplo/incx/x/ap，不发射 kernel。
4. n>0：依次检查 uplo、`incx != 0 && incx != INT_MIN`、alpha/x/ap 非空；非法时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. 读取 host alpha；等于零（包括 -0.0f）时直接成功返回，不更新 ap。
6. 其余场景检查地址计算安全性，生成 tiling 并发射 kernel。资源查询或发射错误按仓内约定映射到已有错误码，不定义新状态码。

不在公共调用中隐式做 Device→Host 拷贝，不为计算结果插入 stream 同步。异步执行错误由调用方在同步时按运行时语义获知，返回 SUCCESS 不等于已经完成 Device 计算。

**2. 分核策略**

按 packed 元素工作量而非相等列数分配任务，避免上下三角列长差异导致核间失衡。设每个 tile 容纳 T 个复数元素，`Q=ceil(M/T)`；使用核心数不超过实际 AIV 核数和 Q，小规模减少核心数。

对核心 b，将连续 tile 区间 `[floor(b*Q/C), floor((b+1)*Q/C))` 分配给它，各核心输出互斥。大部分边界按 GM 写回粒度对齐，尾 tile 单独归属最后一个有效核心；相邻核心不能对同一搬运块进行读改写。若调用地址偏移使对齐条件不满足，则使用精确元素写回的通用路径，不新增用户对齐限制。

一个 packed tile 可能跨列。在 tile 起点对严格递增的 B(j) 做整数二分，求 `B(j)<=k<B(j+1)`，使用哨兵 `B(n)=M`；随后沿列递增处理该 tile 与各列的交集。不用浮点 sqrt 直接决定边界，避免大尺寸舍入误差。UPPER 的 `i=k-B(j)`；LOWER 的 `i=j+k-B(j)`。

**3. UB 和 tiling 参数**

连续 x 快路径在容量允许时将 x 缓存在每核 UB 中；x 为 8*n 字节，不分配 n*n 矩阵。ap 使用至多两个 tile 缓冲；向量寄存器内做交错加载/存储，避免在 GM 中转置数据。

T 的选择须满足 `align(8*n) + 2*align(8*T) + scratch + alignment_margin <= available_UB`。实际可用 UB、向量宽度及 AIV 核数使用目标平台能力和仓内约定获取，不照搬 arch22 的 192 KiB 常量。容量不足时缩小 tile 或回退通用路径；快路径不适用不等于 API 失败。

拟定 tiling 字段包括 `n`、64 位 M、64 位 incx、alpha、uplo、核心数、tile 容量/总数、线程数及路径标识。UPPER/LOWER、连续/一般步长用少量模板或路径枚举分派，不为每个尺寸生成独立 kernel。小规模阈值、T 和线程数在自测后确定，不在本文承诺未经实测的最优值。

#### 3.2.2 kernel 侧设计

**1. SIMD 连续快路径**

- 采用 `KERNEL_TYPE_AIV_ONLY`，通过 CANN 9.1.0 支持的 MicroAPI SIMD 实现，不使用 Cube：秩 1 更新主要受 ap 读写及发射开销影响，转换成通用 GEMM 会引入额外开销。
- 将连续 x 复制到 UB，等待搬运完成。逐 tile 精确读入原 ap，利用列交集找到本段连续的 `x_i` 和列常量 `x_j`。
- 在 SIMD 寄存器中分离交错复数实部/虚部，按有效 mask 计算，再交错写回 ap tile。非对角实部为 `oldReal + r_i*tReal - s_i*tImag`，虚部为 `oldImag + r_i*tImag + s_i*tReal`。
- 对角从非对角向量更新中排除，采用实数路径更新 real，再显式置 imag=0。输入 imag 无论为普通值还是 Inf/NaN 都不参与对角计算。
- 每个 tile 只写回其拥有的有效区间，不将 UB padding 写入 GM。最后一列、跨列短段和向量尾部都使用实际长度；寄存器加载采用合法的尾部掩码或在 UB 内预留并初始化完整向量宽度的保护空间，保护空间计入容量预算，不要求用户提供额外 padding。
- 初版先实现单缓冲正确性；双缓冲仅在事件依赖验证正确且实测有收益时启用。MTE2→V、V→MTE3、MTE3→缓冲复用均有明确等待，未完成写回的缓冲不得被下一轮覆盖。

**2. SIMT 通用路径**

- 使用仓内 arch35 的 `simt_api/asc_simt.h` / `asc_vf_call` 模式，在 AIV 内执行。线程按互不重叠的 packed 元素做 grid-stride 遍历，使用整数映射恢复 i/j。
- 从 GM 按 p(i)、p(j) 读取 x，处理任意合法正负 incx，不为大步长读取包含所有空洞的巨大中间缓冲。
- 每个线程只更新自己负责的复数元素；对角和零列语义与快路径相同。没有跨核归约、atomic、全局 barrier 或辅助 kernel，不申请用户 workspace。
- 小规模以减少发射和 UB 初始化开销为主；一般步长以完整语义为主。不能为了优化 incx=1 而删除其他合法步长支持。

**3. 数值与特殊值控制**

以 FLOAT32 完成复数乘加，不降到 FLOAT16/BFLOAT16。以 Netlib 运算分组为参考，显式区分乘法与累加边界；检查编译器融合/重排是否改变溢出与特殊值分类。必要时使用可控的独立乘、加指令，不用 algebraic fast-math 改写掩盖有限值与非有限值差异。

两条路径统一使用零列跳过、对角独立处理和 alpha-zero host no-op。NaN/Inf 的验收等效规则仅用于比较，不作为将有限值变成 NaN 的理由。

### 工程落点

| 路径 | 计划内容 |
| --- | --- |
| `blas/hpr/arch35/chpr_host.cpp` | 公共 API 的 arch35 实现、校验和 tiling/分派 |
| `blas/hpr/arch35/chpr_kernel.cpp` | AIV kernel、SIMD/SIMT 分支及 launcher |
| `blas/hpr/arch35/chpr_kernel.h` | launcher 声明 |
| `blas/hpr/arch35/chpr_tiling_data.h` | host/kernel 共用 tiling 数据 |
| `blas/hpr/README.md` | 接口约束、示例及 Ascend 950PR 支持状态 |
| `test/hpr/chpr/arch35/` | CSV、GTest、NPU wrapper 和测试说明 |

依照现有架构选择及 `ops_blas_add_gtest_tests` 规则接入，必要时补充最小构建配置；一个目标架构只链接一份 `aclblasChpr` 定义。公共测试解析修改应兼容 arch22 的旧列名，不将 arch35 测试复制成新的顶层测试系统。不提交无用文件、依赖副本或生成的构建产物。

## 支持硬件

| 支持的芯片版本 | 本任务涉及 |
| --- | --- |
| Ascend 950PR（arch35） | 是，CANN 9.1.0 |
| Ascend 950DT | 不在本次验收范围，不因共享架构就宣称已验证 |
| Atlas A2/A3（arch22） | 保留原实现，不作为本任务性能结论 |

开发环境为 Ubuntu 22.04 x86_64；驱动、固件和 Ascend C 工具链须与运行环境配套。

## 算子约束限制

- 仅 complex64 的 Chpr，alpha 为 Host FLOAT32 实数；不扩展 Z hpr、batched、device alpha 或其他接口。
- n>=0，incx 不得为 0 或 INT_MIN；调用方保证 x/ap 的实际分配足够且指针位置符合 BLAS 步长语义。
- ap 原地更新，无 lda，不提供另一三角的显式输出或广播功能。
- 不要求 bit-exact 或确定性计算模式；但必须满足所有有限值误差、特殊值和 no-op 行为。
- x 为只读，不写入步长空洞；n=0、alpha=0 以及参数错误时均不修改 ap。正常更新后对角虚部为 0。
- 异步调用依赖 handle stream；测试读回前必须同步该 stream，不以默认 stream 同步替代。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 设计与验证方法 | 标准来源 |
| --- | --- | --- |
| 功能/状态码 | CSV 驱动 GTest + 独立 API 负向测试，校验返回值与 no-op | 任务书 §2、§3.5 及讨论帖 #53 |
| 精度 | Netlib BLAS 3.10.0 `cblas_chpr`，全量 packed 输出，实/虚部分别统计 | 任务书 §3.2 |
| 性能 | 每个 case 的 `NPU_avg_us <= 2.5 * H100_us` | 任务书 §3.3、附件基线 |
| 采样 | msprof op，5 次 warmup 后连续 10 个有效样本算术平均 | 讨论帖 #53 评审澄清 |
| 内存 | 任务无额外内存性能指标；检查越界、泄漏和错误路径释放 | 任务书 §3.4、工程正确性 |

### 精度判定

FLOAT32 的 `rtol=atol=2^-13`。有限值逐元素满足 `abs(actual-golden)<=atol+rtol*abs(golden)`，用例实部、虚部分别要求匹配比例>=0.99，同时满足最大绝对误差 `1e-2 或 32*ULP` 的任务标准。ULP 分支采用任务引用的 opbase 标准实现，不另自定义更宽松阈值。逐元素检查覆盖全部 M 个 packed 元素，而非抽样。

非有限值单独分类计数：同为 NaN、同号 Inf、NaN 与任意符号 Inf 的组合视为等效；有限值与非有限值、相反符号 Inf 不等效。非有限项不计入有限项的 matched_ratio / max_abs_error，但必须全部通过分类检查；没有有限项时单独记录该情况，不做零分母除法。额外严格检查输出对角 imag 为零，不以 99% 比例掩盖对角错误。

alpha=0、n=0 的 no-op 比较使用原始字节不变判据，避免 NaN、signed zero 影响普通相等比较。对错误返回同样检查 ap 未修改。

### 功能和精度测试计划

1. 原样保留附件 `chpr_test.csv` 的 1200 条用例身份，明确其中 1000 条精度、200 条 `TC_PF` 性能；n 的覆盖上限是测试规模，不是 API 上限。
2. 适配附件 `x_fill` / `ap_fill` 与旧解析器 `x` / `ap` 的列名；识别 alpha 的 `null`、tensor 的 `NULLPTR`，确保实际传入空指针而非生成普通数据。
3. 负向用例先处理非法 n/incx，不能在测试准备阶段对 INT_MIN 调用 int abs 或因非法步长分配巨型数组；计数与字节数计算先提升至 64 位。
4. 覆盖 UPPER/LOWER、n=0/1、奇数、非对齐、2 的幂±1、跨 tile 边界、incx=±1/±2/±3 和更大合法步长。
5. 独立测试空 handle 与 n=0/n<0/非法 uplo/空指针组合，确认 handle 错误优先；验证 n=0 忽略其余参数、alpha=0 不绕过参数检查。
6. 覆盖零向量、交替值、极端有限值、Inf/NaN、非零/非有限输入对角虚部、零列和特殊 alpha；检查 x 与步长空洞不变、ap 两端哨兵不变。
7. 随机类用例中 alpha、x、ap 分别按用例数保持均匀/正态各 50%，特殊值和边界用例不计入分母。均匀范围[-5,5]，正态参数按任务书；复数实部/虚部独立采样。
8. 原始附件分布不足时另补用例，记录分布标签、分母、数量及种子；保留原始用例与补充用例的对应关系，不把生成器预留开关当作已经实现混合分布。
9. 至少覆盖非默认 stream 和两条执行路径的相同语义；在支持条件下做内存检查及重复运行，捕获未初始化读取、尾块越界和缓冲事件竞争。

### 性能测试计划

- 附件性能用例共 200 条，当前 incx=1，n 从 1 扫描至 4096；依然保留其他合法步长的功能覆盖。
- 按 `(uplo,n,incx)` 关联 H100 基线，将 `gpu_ms` 乘 1000 转为 us；不混用毫秒与微秒。基线缺失、样本不足、case 缺失均标为未完成，不按通过处理。
- 使用单 case filter 隔离采样，按 kernel 名和调用顺序提取 `OpBasicInfo.csv` 的 `Task Duration(us)`。遵循 msprof op 的 5 次 warmup 及之后连续 10 个有效样本，不在未核实采样逻辑时叠加 warmup。
- 正式样本只统计算子任务时长，不含分配、拷贝、CPU golden、GTest 初始化和比对；每次调用前在采样区间外恢复原始 ap，避免重复秩 1 更新改变输入分布或造成溢出。
- 返回 SUCCESS 才能计为有效调用；异常及重测原因保留日志。一个 case 涉及多条设备任务时核实任务对应关系并完整计时，不能只选最快任务。
- 附件 `verify_performance.py` 读取的 GTest 整体毫秒耗时只能辅助冒烟检查，不能替代上述 msprof 测量；脚本退出码也不能替代逐 case 数量、状态、基线和样本检查。
- 保存 10 个原始样本、均值、对应 GPU 基线及比值。所有 case 分别达到>=0.4 倍要求，不以平均加速比抵消失败 case。
- 正式性能测量记录芯片、驱动/CANN、设备号、代码 commit 和实际共享负载；受干扰样本不作为独占性能结论。不修改共享设备配置或他人进程。

附件前四条基线对应的目标上限如下，仅为验收门槛，不是 NPU 实测结果：

| uplo | n | incx | H100（us） | NPU 平均耗时上限（us） |
| --- | --- | --- | --- | --- |
| UPPER | 512 | 1 | 11.909 | 29.7725 |
| LOWER | 1024 | 1 | 30.999 | 77.4975 |
| UPPER | 2048 | 1 | 96.491 | 241.2275 |
| LOWER | 4096 | 1 | 570.764 | 1426.9100 |

### 可维护性和风险控制

| 风险 | 控制方法 |
| --- | --- |
| 小规模发射/缓存初始化开销 | 小规模 SIMT，核心数和路径阈值实测选择 |
| packed 跨列及不均衡 | 按元素 tile 分核，整数边界定位，写回块单一所有者 |
| SIMD 搬运尾部越界 | 有效长度/mask，UB 内补齐而不扩大 GM 访问 |
| 负步长或整数溢出 | 64 位寻址、安全字节计算，INT_MIN 在取绝对值前拒绝 |
| Inf/NaN 与融合重排差异 | Netlib 分支与计算分组、对角独立路径、分类比对 |
| 原始脚本漏测或假成功 | 核对期望/实际 case 集合、进程状态、原始日志和有效样本数 |

实现代码保持 host 校验、tiling、数据访问和数值运算职责清晰；复用公共设施，日志记录必要参数及路径，不引入重复 API、无效编译文件或硬编码敏感信息。

## 兼容性分析

- 不改变公共 API/ABI、`aclblasComplex` 布局及状态码，只新增 arch35 实现；arch22 保持原路径。
- 构建按目标架构选取实现，避免重复符号。共享测试参数解析若增加附件列名别名，保留旧列名回退并检查原有测试兼容性。
- 公共 README 仅在实际验证后把 Ascend 950PR 改为支持，不连带宣称其他型号通过验证。
- 不增加 workspace 参数、用户 padding 要求或额外同步语义。

## 交付与评审流程

本次 PR 仅提交本设计文档。设计评审通过并合入后，开展实现和自测；验收提交包含代码仓地址/分支/目录/commit、完整测试代码及可复现步骤、精度实虚部结果、性能原始数据/截图及日志，并按要求邀请 `Ascend-CANN` 访问个人代码仓。测试通过后再申请 ops-blas 实现和测试 PR 合入。

## 参考资料

1. [任务书附件](https://www.hiascend.com/p/resource/202610/7c5e3abfe8fa4e80bd5a1a6095999801.zip)。
2. [任务讨论与验收口径澄清](https://gitcode.com/cann/ops-blas/discussions/53)。
3. [设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
4. [cuBLAS hpr 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hpr)。
5. [Netlib chpr 参考实现](https://www.netlib.org/blas/chpr.f)。
6. [任务引用的生态精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
7. [ops-blas 代码仓](https://gitcode.com/cann/ops-blas)。
