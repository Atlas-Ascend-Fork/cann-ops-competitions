# 需求背景（required）

## 需求来源

2026年10月社区任务：aclblasChpr2 算子开发（Atlas 950PR），任务链接：
https://www.hiascend.com/activities/task-center/details/8e6e450ec9a44869a4956675200cc05b

## 背景介绍

### Chpr2 算子能力补齐

ops-blas 开源仓（https://gitcode.com/cann/ops-blas）的 spr2 算子族目录 `blas/spr2/` 已有实数对称版本
aclblasSspr2（arch22/arch35），复数 Hermitian 版本 aclblasChpr2 尚无实现（`include/cann_ops_blas.h`
中也无声明）。本任务在 `blas/spr2/arch35/` 新增 aclblasChpr2，补齐 Blas 仓 packed 秩-2更新能力。

### 对标接口

cuBLAS `cublasChpr2`（参数序列一一对应）；参考实现 Netlib BLAS `chpr2.f`
（https://www.netlib.org/blas/chpr2.f ）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言（SIMT kernel 直调方式）在 Ascend 950PR 上实现 `aclblasChpr2`：
单精度复数（complex64）Hermitian 压缩存储秩-2更新，完成算子设计、开发、测试全流程。

## 需求拆解

1. 功能与 cuBLAS `cublasChpr2` 核心功能、参数语义完全对齐（含负步长、quick return、校验顺序）
2. packed 压缩存储（0-based，n(n+1)/2 个复数元素，无前导维度）；对角虚部更新后置 0
3. 接口声明放入 `include/cann_ops_blas.h`（与其他产品线共用，禁止 950PR 私有平行接口）
4. 精度满足生态算子开源精度标准（FLOAT32 档：rtol=atol=2^-13，matched_ratio>=0.99，
   max_abs_error<=max(1e-2, 32*ULP)，Inf/NaN 等效规则）
5. 性能达到 GPU（H100）标杆 0.4 倍以上（平均单次耗时 <= 标杆 2.5 倍，msprof/事件计时口径：
   5 次 warmup + 10 次有效采样取平均）

# 详细设计（required）

## 算子分析

### 数学公式

```text
A := alpha * x * y^H + conjg(alpha) * y * x^H + A
```

其中 alpha 为复数标量，x/y 为 n 元素复数向量，A 为 n×n Hermitian 复数矩阵（packed 存储）。
逐元素展开（列主序，A(i,j) 表示第 i 行第 j 列）：

```text
A(i,j) += alpha * x(i) * conj(y(j)) + conjg(alpha) * y(i) * conj(x(j))
```

列内复用形式（对齐 Netlib chpr2.f 的 TEMP1/TEMP2 结构，j 为列号）：

```text
TEMP1 = alpha * conj(y(j));  TEMP2 = conjg(alpha * x(j))
A(i,j) += x(i) * TEMP1 + y(i) * TEMP2        （i != j，非对角）
A(j,j)  = REAL(A(j,j)) + REAL(x(j)*TEMP1 + y(j)*TEMP2)   （对角：实部累加，虚部置 0）
```

零列语义（Netlib）：X(j)=Y(j)=0 时跳过整列非对角计算，但对角虚部仍置 0。

### packed 存储格式（0-based）

- UPPER（上三角按列压缩，A(i,j)，i <= j）：`ap[i + j*(j+1)/2]`
- LOWER（下三角按列压缩，A(i,j)，i >= j）：`ap[i + ((2*n-j-1)*j)/2]`

### 支持数据类型

complex64（aclblasComplex，实部/虚部各 float32）；alpha 为复数标量（Host 指针）。

### 支持形状

n >= 0（n=0 合法 no-op）；x/y 逻辑长度 n（物理长度 1+(n-1)*|inc|）；ap 长度 n(n+1)/2。

## 算子实现

### 实现方案总览

使用 Ascend C SIMT kernel 直调（`asc_vf_call` + `__simt_vf__`），代码位于 `blas/spr2/arch35/`：

```
blas/spr2/arch35/
├── chpr2_tiling_data.h   # tiling 结构与路径阈值常量
├── chpr2_host.cpp        # 参数校验（§2.5 固定顺序）+ tiling + kernel 下发
└── chpr2_kernel.cpp      # 主路径（元素级并行）+ 通用路径（任意步长）
```

#### host 侧设计

校验顺序固定（与任务书 §2.5、arch35 同族 aclblasSspr2 一致）：

```
handle 判空(HANDLE_IS_NULLPTR) -> n<0(INVALID_VALUE) -> n==0(SUCCESS，不访问其余参数)
-> alpha 判空 -> uplo 非法 -> incx/incy 为 0 或 INT_MIN -> x/y/ap 判空（均 INVALID_VALUE，
与 alpha 是否为零无关）-> alpha==(0,0)(SUCCESS，quick return，不访问 x/y/ap)
```

tiling 与分核策略：

- tiling 结构 `Chpr2TilingData`：n、uplo、blockNum（实际启动块数，host 传入）、alphaReal/alphaImag、
  incx、incy
- 块数随工作量缩放：主路径（inc=1）按 packed 元素量 totalE=n(n+1)/2 取
  `blocks = clamp(totalE/32, 1, aivCoreNum)`（每块约 32 元素，小 n 也铺满多核，大 n 用满 56 个
  AIV 核）；通用路径按 `totalE/64`（列数封顶）
- 段几何（segStart/segLen）与 chunk 大小在 __global__ 计算一次作参数传入（避免每线程重复
  64 位除法等重标量序言；packed 索引 <= 33.6M 全程 uint32），
  chunk = max(segLen<=64 ? 2 : 4, segLen/(threads*4))
- 线程数随 n 缩放（`__simt_vf__` 每线程栈帧初始化开销与线程数成正比）：n<=128 用 128 线程/块，
  n<=512 用 256，其余 2048

#### kernel 侧设计

**主路径 `Chpr2GmChunked<UPLO>`（incx=incy=1 且 1<=n<=8192）——元素级并行：**

1. 块 b 处理元素均衡列区间 [colStart, colEnd)（`__global__` 二分定位：首个 ColEnd > seg 的列；
   相邻块使用同一边界函数，按列不交不漏），对应 packed 段 [colBase(colStart), colBase(colEnd))
2. 段内按连续元素 chunk 做 grid-stride（chunk = max(segLen<=64 ? 2 : 4, ceil(segLen/(vfThreads*4)))，
   线程数随 n 缩放 128/256/2048，摊薄列定位固定开销）；
   线程并行度为元素数而非列数，避免长列串行成为瓶颈
3. 每 chunk 从块内列起点顺序定位起始列，随后增量推进（colBase += colLen）；列边界处计算
   TEMP1/TEMP2（volatile 分立舍入）与零列标记；对角元素单独处理（实部累加 Re(x_j*TEMP1+y_j*TEMP2)，
   虚部置 0）；x/y 直读 GM（小向量常驻 L2）
4. 溢出安全判定（每 warp 独立全量扫描 + warp 内硬件归约 asc_reduce_max，无跨 warp 通信）：
   偏置指数满足 `2*sliceMaxExp + alphaMaxExp <= 503` 时所有元素乘积必不溢出 FLT_MAX，
   该块走融合乘快速路径（舍入差 <= 几 ulp，远小于 2^-13 容差，且 Inf/NaN 形态与参考一致）；
   否则走 volatile 分立舍入精确路径。各 warp 扫描同一区间后归约，块内全部线程得到一致的
   sliceMaxExp；不使用 `__ubuf__` 共享内存数组（msSanitizer memcheck 对 `__simt_vf__` 内
   `__ubuf__` 数组访问会误报 illegal，同族已合入实现同样命中），冗余扫描读均为 L1/L2 命中，
   成本可忽略（实测性能不降反升）。

**通用路径 `Chpr2GmStrided`（任意步长非 1 / n>8192）——列并行：**

交错列分配（`col = blockIdx + tid*gridDim`，小 n 时列仍能铺满多个块），运行时步长分支，
每线程完整处理一列（TEMP 复用、零列跳过、对角单独处理），单实例小体积。

#### 精度关键设计：与 cblas 参考实现 bit 级一致

ASC 后端会把 mul+add 模式匹配成融合 madd（`-ffp-contract=off` 仅约束前端），融合会改变
Inf/NaN/溢出边界的结果形态（如 FLT_MAX 乘积由 ±Inf 变为融合精确和的有限值），与 cblas golden
（f2c 逐运算舍入）在极端值用例上产生单侧特殊值分歧（finite vs Inf 判定不通过）。对策：

- 复数乘 `Chpr2ComplexMul` 以 volatile 分立乘积阻断融合（手法同仓内 cgbmv
  ReferenceComplexMultiply），保证逐运算舍入
- 非对角累加顺序对齐参考实现的左结合语义：`((ap + x_i*temp1) + y_i*temp2)`（反汇编验证
  Ubuntu libblas 3.10.0 的 gfortran 编译产物；对角为 `REAL(ap) + REAL(x_j*temp1 + y_j*temp2)`
  即先求和后累加，两者顺序不同）。巨量抵消场景下 `ap += (h1+h2)` 与 `((ap+h1)+h2)` 的差异
  可达 |ap| 量级，远超容差——独立边界测试（均匀 1e18/2e19 幅值）曾捕获该差异，修复后
  与 cblas bit 级一致
- 快速路径仅在块级证明乘积不溢出时启用（此时融合与分立在 Inf/NaN 形态上一致，
  舍入差 <= 几 ulp 在 2^-13 容差内）
- 附加 `-ffp-contract=off`（CMakeLists 对 chpr2_kernel.cpp 的 COMPILE_OPTIONS）作为前端防线

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（Ascend 950PR，arch35） | √ |

## 算子约束限制

- 仅支持 complex64；alpha 为 Host 指针（对齐 cuBLAS 默认 pointer mode = HOST）
- n=0 / alpha=(0,0) 为合法 no-op；incx/incy 非 0 且非 INT_MIN（支持负步长）
- packed 存储无前导维度参数，调用方须保证 ap 实际分配长度 >= n(n+1)/2

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 1000 条精度用例全过（rtol=atol=2^-13，matched_ratio>=0.99，max_abs_error<=max(1e-2,32*ULP)，Inf/NaN 等效规则；golden = Netlib BLAS 3.10.0 cblas_chpr2） | 任务书 §3.2 生态算子开源精度标准 |
| 性能标准 | 200 条性能用例平均单次耗时 <= H100 标杆 2.5 倍（5 warmup + 10 采样取平均） | 任务书 §3.3 |

## 自测结果（Ascend 950PR，CANN 9.1.0）

**精度**：1001/1001 全过（999 条 CSV 精度用例 + 空指针单测 + 套件）。

**性能**（ACL 事件计时，avg_us 为 10 次采样均值；ratio = H100 标杆 / NPU）：

| 用例 | n | uplo | NPU avg_us | H100 us | ratio | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| TC_PF_1001 | 512 | UPPER | 20.203 | 13.54 | 0.670 | PASS |
| TC_PF_1002 | 1024 | UPPER | 49.545 | 33.92 | 0.685 | PASS |
| TC_PF_1003 | 2048 | LOWER | 106.448 | 99.12 | 0.931 | PASS |
| TC_PF_1004 | 4096 | LOWER | 1074.137 | 592.30 | 0.551 | PASS |

（ACL 事件计时，5 次 warmup + 51 次有效采样取算术平均，同时覆盖任务书 §3.3 的 10 次、
§7 的 >10 次与随附 README 的 >50 次采样要求；每次 aclblasChpr2 调用恰好下发 1 个 kernel，
统计完整调用。）

任务书 §3.3 的 4 条典型 case 全部达标；**200 条完整性能用例 200/200 全部达标**
（0 FAIL / 0 NO_REF / 0 MISSING，最紧 ratio 0.463 @ n=3783）。

msprof op 模式在本开发容器无法运行（Profiling channel start failed，错误日志见
evidence/msprof_error.log），正式性能采集口径待任务方确认；当前以 ACL 事件计时为
保守上界证据（事件窗口包含 kernel 下发流水，msprof task duration 只会更低）。

## 兼容性分析

新算子（接口声明新增于 include/cann_ops_blas.h），不涉及兼容性分析。
