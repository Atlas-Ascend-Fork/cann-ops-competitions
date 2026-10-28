# aclblasChpr2 算子设计文档

> 社区任务：10月社区任务-aclblasChpr2算子开发（950）
> 适配硬件：Ascend 950PR（arch35 / DAV_3510）
> CANN 版本：CANN 9.1.0
> 对标接口：cuBLAS `cublasChpr2` / Netlib BLAS `chpr2`
> 工程框架：ops-blas 开源仓 `blas/spr2/arch35/`（kernel 直调，句柄式 BLAS 接口）

---

## 一、需求背景

### 1.1 需求来源

通过社区任务完成昇腾算子开源仓（ops-blas）的算子贡献需求。本任务为在 Ascend 950PR
上使用 Ascend C 编程语言开发单精度复数（complex64）Hermitian 压缩存储秩2更新算子
`aclblasChpr2`，完成算子设计、开发、测试全流程工作，验收通过后合入昇腾算子开源仓
ops-blas 的 `blas/spr2/arch35/` 目录，补齐 Blas 仓能力。

### 1.2 背景介绍

#### 1.2.1 aclblasChpr2 算子开发

`aclblasChpr2` 是 BLAS Level-2 复数 Hermitian 压缩存储秩2更新算子，计算：

```
A = alpha * x * y^H + conjg(alpha) * y * x^H + A
```

其中 `alpha` 为单精度复数标量（Host 侧指针传入），`x`、`y` 为 n 元素复数向量，
`A` 为 n×n Hermitian 复数矩阵，以 packed 压缩格式存储于一维向量 `ap`。

- 对标基线：cuBLAS `cublasChpr2`
  （https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-chpr2 ）
  与 Netlib BLAS `chpr2.f`（https://www.netlib.org/blas/chpr2.f ）。
- 本算子无历史 TBE 版本，为全新算子，参考实现为 cuBLAS / Netlib 浮点语义。
- ops-blas 仓内已有同族实数算子 `aclblasSspr2`（`blas/spr2/arch35/`），工程结构
  可直接扩展复用；复数同族 `aclblasCher`（秩1更新）亦已在仓内，可参考其复数处理。

#### 1.2.2 标杆算子现状分析

##### 1.2.2.1 标杆算子支持的数据类型和数据格式

cuBLAS `cublasChpr2` / Netlib `chpr2` 支持能力：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 数据排布 | 维度 |
|------|----------|----------|--------------|----------|------|
| handle | 库上下文句柄 | handle | - | - | - |
| uplo | 三角选择 | enum | `ACLBLAS_UPPER` / `ACLBLAS_LOWER` | - | - |
| n | 矩阵阶数 | scalar | int32 | - | - |
| alpha | 标量系数 | 指针 | `aclblasComplex*`（COMPLEX64，**Host 指针**） | - | 单值 |
| x | 输入向量 | tensor | `aclblasComplex`（实/虚部各 float32） | ND，交错存储 | 逻辑一维 [n]，物理长度 `1+(n-1)*\|incx\|` |
| incx | x 步长 | scalar | int32 | - | - |
| y | 输入向量 | tensor | `aclblasComplex` | ND，交错存储 | 逻辑一维 [n]，物理长度 `1+(n-1)*\|incy\|` |
| incy | y 步长 | scalar | int32 | - | - |
| ap | 输出 packed 矩阵 | tensor | `aclblasComplex` | packed 压缩一维 | `n(n+1)/2` 个复数元素，**无前导维度参数** |

packed 压缩存储格式（0-based 索引）：

- `uplo = ACLBLAS_UPPER`：上三角按列无间隙压缩，A(i,j)（i ≤ j）存于
  `ap[i + j*(j+1)/2]`；
- `uplo = ACLBLAS_LOWER`：下三角按列无间隙压缩，A(i,j)（i ≥ j）存于
  `ap[i + ((2*n-j-1)*j)/2]`；
- 未存储的另一半由 Hermitian 共轭对称性隐含，不引用、不更新；
- A 为 Hermitian 矩阵：对角元素虚部假定为 0，计算完成后 ap 中对角元素虚部须置 0。

quick return 语义（cuBLAS 文档与 Netlib `chpr2.f` 一致）：`n = 0` 或
`alpha = (0, 0)` 时为合法 no-op，不更新 ap、不访问 x/y，直接返回成功。

##### 1.2.2.2 标杆算子实现描述

Netlib `chpr2.f` 核心逻辑（按列处理）：

```fortran
TEMP1 = ALPHA*X(J)
TEMP2 = CONJG(ALPHA)*Y(J)
DO I = 1,J                       ! UPPER：列 j 的行范围 1..j
    AP(KK) = AP(KK) + X(I)*CONJG(TEMP2) + Y(I)*CONJG(TEMP1)
    KK = KK + 1
END DO
```

实现要点：

1. 逐列处理：每列先计算列因子 `TEMP1 = alpha*x(j)`、`TEMP2 = conjg(alpha)*y(j)`，
   列内各元素复用列因子，将复数乘法次数从 O(n²) 降为 O(n) 次因子计算 +
   O(n²) 次乘加。
2. `incx/incy` 支持任意非零步长（含负步长，负步长按 Netlib 语义反向遍历
   x/y，即 x(j) 取 `x[1+(j-1)*incx]`）。
3. 对角元素增量为 `x(j)*conjg(TEMP2) + y(j)*conjg(TEMP1)`，其虚部互为相反数、
   实部相同，纯实数；对角虚部直接置 0。
4. `alpha = (0,0)` 时无需遍历，直接返回；`n = 0` 同理。

##### 1.2.2.3 标杆算子实现流程图

```
┌──────────────────────────────────────────────────────┐
│ 输入：uplo, n, alpha, x, incx, y, incy, ap           │
└───────────────────────────┬──────────────────────────┘
                            ▼
              n = 0 或 alpha = (0,0) ?
              ├── 是 ──► RETURN（no-op）
              ▼ 否
        逐列 j = 0..n-1：
            TEMP1 = alpha * x(j)，TEMP2 = conjg(alpha) * y(j)
            对列内行 i（UPPER: 0..j，LOWER: j..n-1）：
                ap[packed(i,j)] += x(i)*conjg(TEMP2) + y(i)*conjg(TEMP1)
            对角元素：虚部置 0
                            ▼
                         RETURN
```

---

## 二、需求分析

### 2.1 外部组件依赖

- Ascend C 算子开发工具链（CANN 9.1.0，含 ccec_compiler / tikicpulib）。
- ops-blas 开源仓工程框架（kernel 直调、句柄式 BLAS 接口）。
- 精度比对 golden 由 `chpr2_golden.h` 提供（Netlib `chpr2` 复数语义、double
  精度累加以降低舍入干扰），随测试工程提供，无其他三方软件依赖。

### 2.2 内部适配模块

- ops-blas 仓 `blas/spr2/arch35/`：复用同族实数算子 `aclblasSspr2` 的 host/kernel
  工程结构（tiling 数据结构、kernel 直调、SIMT 线程级并行），并参考复数同族
  `aclblasCher` 的复数分量处理方式。
- ops-blas 仓 `include/cann_ops_blas.h`：新增 `aclblasChpr2` 接口声明（master 现
  无该声明，需随本任务新增），与其他产品线共用同一 API，禁止定义 950PR 私有
  平行接口。
- 测试工程：`test/spr2/chpr2/`（任务书要求嵌套目录），复用 `test/frame/` 公共
  测试框架（fill/verify/blas_test）。

### 2.3 需求模块设计

#### 2.3.1 Ascend C 算子原型

接口声明放入 `include/cann_ops_blas.h`，签名与 `cublasChpr2` 参数序列逐参数对齐，
与仓内同族 `aclblasSspr2` 同构：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x,
    int incx,
    const aclblasComplex* y,
    int incy,
    aclblasComplex* ap);
```

| 参数 | 方向 | 语义 | 异常行为 |
|------|------|------|----------|
| handle | 输入 | 库上下文句柄，携带 stream | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 输入 | 三角选择 | 非法值 → `ACLBLAS_STATUS_INVALID_VALUE` |
| n | 输入 | 矩阵阶数（Host） | n<0 → `ACLBLAS_STATUS_INVALID_VALUE`；n=0 → SUCCESS（no-op） |
| alpha | 输入 | 复数标量（**Host 指针**） | nullptr → `ACLBLAS_STATUS_INVALID_VALUE`；\*(0,0) → SUCCESS（no-op，不访问 x/y/ap） |
| x | 输入 | 单精度复数向量（Device，只读） | n>0 且 alpha 有效时 nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | x 复数步长（Host） | 0 或 INT_MIN → `ACLBLAS_STATUS_INVALID_VALUE` |
| y | 输入 | 单精度复数向量（Device，只读） | 同 x |
| incy | 输入 | y 复数步长（Host） | 同 incx |
| ap | 输出 | packed Hermitian 矩阵（Device，读写） | nullptr → `ACLBLAS_STATUS_INVALID_VALUE` |

返回值 `aclblasStatus_t`，状态码语义与 `include/cann_ops_blas_common.h` 一致。

**参数校验顺序**（任务书 §2.5）：

```
handle == nullptr                        → HANDLE_IS_NULLPTR
n < 0                                    → INVALID_VALUE
n == 0                                   → SUCCESS（no-op）
alpha == nullptr                         → INVALID_VALUE
uplo 非 UPPER/LOWER                      → INVALID_VALUE
incx == 0 / incx == INT_MIN              → INVALID_VALUE
incy == 0 / incy == INT_MIN              → INVALID_VALUE
x / y / ap == nullptr（n>0）             → INVALID_VALUE
alpha == (0,0)                           → SUCCESS（no-op，不访问 x/y/ap）
```

#### 2.3.2 Ascend C 算子相关约束

与标杆算子相比，Ascend C 版本能力对齐情况：

| 标杆能力 | Ascend C 版本 | 说明 |
|----------|---------------|------|
| COMPLEX64 输入/输出 | ✅ 支持 | `aclblasComplex` 交错存储，实/虚部 float 分量独立计算 |
| UPPER / LOWER packing | ✅ 支持 | packed 索引 `i + j(j+1)/2` / `i + (2n-j-1)j/2`，按列映射 |
| incx/incy 任意非零（含负） | ✅ 支持 | 负步长按 Netlib 语义反向遍历，host 校验拦截 0 与 INT_MIN |
| 对角虚部置 0 | ✅ 支持 | 对角增量纯实数，kernel 写回时虚部置 0 |
| n=0 / alpha=(0,0) quick return | ✅ 支持 | 不触发 kernel、不访问 x/y/ap |
| 溢出/Inf/NaN 传播 | ✅ 支持 | 按任务书 §3.2 Inf/NaN 等效规则判定 |
| 逐元素浮点顺序 | 不要求一致 | 按精度阈值判定，不做 bit-exact |

无功能缺失，参数语义与标杆完全对齐。

---

## 三、需求详细设计

### 3.1 调用方式

采用 **kernel 直调**（ops-blas 句柄式 BLAS 接口）：

```
用户应用 → aclblasChpr2(handle, uplo, n, alpha, x, incx, y, incy, ap)
    → handle 绑定 stream（aclblasSetStream）
    → host 侧参数校验（§2.5 顺序）+ tiling 计算
    → 直调 NPU kernel（<<<grid, nullptr, stream>>>）
    → 读回前同步 stream
```

### 3.2 需求总体设计

#### 核心数学分解

秩2更新按列分解（与 Netlib 列序一致）：

```
对每列 j：
    pCol = alpha * conjg(y(j))        TEMP1 路径因子
    qCol = conjg(alpha) * conjg(x(j)) TEMP2 路径因子
    A(:,j) += x * pCol + y * qCol     （仅 uplo 指定三角的行范围）
    A(j,j) 增量 = x(j)*pCol + y(j)*qCol + conjg(同式) = 2*Re(...)，纯实数
```

每列仅需 2 次复数乘计算列因子，列内 n 个元素各做 2 次复数乘加；列因子在
kernel 内按列计算一次、线程间复用，避免 O(n²) 次重复的 `alpha`/`conjg` 前置
乘法。

#### 并行分解策略

packed 压缩格式下列内元素地址连续（UPPER 列 j 基址 `j(j+1)/2`，LOWER 列 j 基址
`(2n-j-1)j/2`，列内步长 1），天然适合**按列分块**的多核并行：

- 以"列组"为粒度切分：各核处理连续的一段列 `j ∈ [colStart, colStart+colNum)`，
  核间无 packed 区间交叠，无跨核数据竞争，无需核间同步；
- 核内 SIMT 线程按列内行区间继续细分：长列（低索引列，UPPER 时长度可达 n）
  由多线程按行分块协同，短列（高索引列）由少量线程处理，负载随 packed 三角
  形状天然递减；
- incx/incy 的负步长在 host 侧归一化为正步长等价形式（起始指针偏移 +
  正步长），kernel 统一按正步长寻址，消除负步长分支。

工程新增三个文件（避免与 `Sspr2*` 符号冲突）：

- `chpr2_tiling_data.h`：`Chpr2TilingData`（n / uplo / alphaReal / alphaImag /
  incx / incy）；
- `chpr2_host.cpp`：参数校验（§2.5 顺序）、quick return、tiling、launch、
  kernel 派发；
- `chpr2_kernel.cpp`：SIMT 模板 kernel（UPLO×INCX×INCY 特化组合）。

#### 3.2.1 host 侧设计

##### 3.2.1.1 分核策略

- 优先使用满核，按 packed 三角总工作量（约 n(n+1)/2 复数元素）与列分布动态
  决定核数，上限取平台 SIMT 核数；
- 大 n：多核均分列组，前若干核多承担 1 列（大核/小核策略），列间工作量差异
  由"按列加权分配"缓解——按列累计 packed 元素数（前缀和思想）而非简单列数
  均分，使各核 packed 元素量接近；
- 小 n（n ≤ 单核可容纳列数）：单核处理，避免多核启动开销。

##### 3.2.1.2 数据分块和内存优化策略

SIMT 路径：线程直接访问 GM，不经 MTE2/UB 中转（packed 索引不规则，UB 分块
搬运收益有限且引入额外同步）。

- 每核启动 `nthreads` 个线程（`asc_vf_call`，`LAUNCH_BOUND` 对齐 SIMT 线程
  上下限），线程按 `(列, 列内行块)` 二级映射遍历本核 packed 区间；
- alpha 拆分为实部/虚部两个 float 标量随 tiling 下发（Host 指针，launch 前
  读取，避免 Device 侧指针解引用）；
- 复数乘加以实/虚部 float 分量展开（6 乘 4 加/元素对），由编译器 FMA 融合；
- 对角元素单独判定（`i == j`）：增量取实部 2 倍、虚部写 0。

##### 3.2.1.3 tilingkey 规划策略

UPLO(2) × INCX 方向(2) × INCY 方向(2) = 8 个模板特化组合，编译期展开
packing 基址公式与（归一化后的）步长寻址，消除运行时分支；步长幅值
|incx|/|incy| 作为运行时参数传入，不进入 tilingkey。

#### 3.2.2 kernel 侧设计

##### 3.2.2.1 kernel 侧实现描述

```
每 block（负责列组 [colStart, colStart+colNum)）：
  线程 t：遍历列 j ∈ 本核列组（跨步 blockDim）
      计算 pCol = alpha*conjg(y(j)), qCol = conjg(alpha)*conjg(x(j))
      列内行区间按线程继续分块：
          UPPER: i ∈ [0, j]，基址 j(j+1)/2
          LOWER: i ∈ [j, n)，基址 (2n-j-1)j/2
          ap[base+i] += x(i)*pCol + y(i)*qCol（实/虚部分量乘加）
          i == j 时：实部增量保留、虚部置 0
```

- 长列多线程协同：单列过长时（如 UPPER 的 j≈n 列），线程按行区间分块，
  列内无写冲突（各线程行区间不相交）；
- 对角虚部：由行分块边界保证对角元素唯一归属单线程，直接写 0，无需原子
  操作或核间同步。

##### 3.2.2.2 Ascend C 实现流程图

```
      aclblasChpr2(handle, uplo, n, alpha, x, incx, y, incy, ap)
                          │
        ┌─────────────────▼──────────────────┐
        │ handle==nullptr                    │──是──► HANDLE_IS_NULLPTR
        │ n<0                                │──是──► INVALID_VALUE
        │ n==0                               │──是──► SUCCESS（no-op）
        │ alpha==nullptr                     │──是──► INVALID_VALUE
        │ uplo 非法                          │──是──► INVALID_VALUE
        │ incx/incy == 0 或 INT_MIN          │──是──► INVALID_VALUE
        │ x/y/ap==nullptr（n>0）             │──是──► INVALID_VALUE
        │ alpha==(0,0)                       │──是──► SUCCESS（不访问 x/y/ap）
        └─────────────────┬──────────────────┘
                          ▼
          tiling：useCoreNum/colStart/colNum + alpha 分量
                          ▼
        ┌─────────────────▼──────────────────┐
        │ chpr2_simt_kernel（UPLO×INCX×INCY  │
        │ 8 特化模板之一）                    │
        │  逐列：列因子 pCol/qCol             │
        │  列内：线程行分块乘加                │
        │  对角：虚部置 0                     │
        └─────────────────┬──────────────────┘
                          ▼
                 同步 stream，返回 SUCCESS
```

##### 3.2.2.3 Ascend C 实现流程图与标杆算子流程图存在的差异点和原因

| 差异点 | 标杆算子（Netlib chpr2） | Ascend C 实现 | 原因 |
|--------|---------------------------|---------------|------|
| 执行方式 | 单线程标量逐列循环 | 多核 SIMT kernel 直调（列组分核 + 列内行分块） | NPU 并行架构；packed 列内地址连续，天然可并行且核间无冲突 |
| 列因子复用 | 每列 2 次复数乘（TEMP1/TEMP2） | 相同分解（pCol/qCol），但列因子随 tiling 分布到各线程 | 数学分解一致，并行化不增加复数乘次数 |
| 负步长处理 | 负下标遍历（Fortran 1-based 反向） | host 侧归一化为正步长等价寻址 | 消除 kernel 内符号分支，模板特化更精简 |
| 对角虚部 | 显式置 0（Fortran 语义） | 行分块保证对角唯一归属，写回时置 0 | 语义一致；避免多线程对角竞争 |
| alpha 读取 | 栈上复数直接使用 | Host 指针 launch 前读取，拆实/虚部入 tiling | alpha 为 Host 指针，Device 侧不重复搬运 |
| 浮点求和顺序 | 逐元素顺序累加 | 按列独立累加，元素间无跨列依赖 | packed 元素一对一更新（无归约），顺序差异仅来自乘加融合，按精度阈值判定 |

### 3.3 支持硬件

| 芯片版本 | 支持 |
|----------|:---:|
| Atlas A2/A3（ascend910b*） | ✗ |
| Ascend 950PR（arch35 / DAV_3510） | ✅ |

与任务书 §3.1 要求的适配硬件（Ascend 950PR）保持一致。

### 3.4 算子约束限制

- 数据类型仅支持 COMPLEX64（`aclblasComplex`，实/虚部各 float32）；不支持
  半精度/双精度/混合精度。
- `ap` 为 packed 一维存储，长度 `n(n+1)/2` 个复数元素，无前导维度参数；仅
  uplo 指定三角被引用和更新，另一三角不访问。
- `incx/incy` 支持任意非零步长（含负）；0 与 INT_MIN 为非法值。
- 无 broadcast / 动态 shape 需求；n 与步长均为 Host 侧 int32 标量。
- 极端值组合下中间乘积可溢出至 ±Inf，按任务书 §3.2 Inf/NaN 等效规则判定。

---

## 四、特性交叉分析

| 特性 | 分析 |
|------|------|
| 精度与性能 | 列因子分解使每元素仅 2 次复数乘加（6 乘 4 加分量展开，FMA 融合）；packed 列内连续访问保证 GM 访存局部性；性能目标：核心 shape 耗时 ≤ H100 对标耗时 / 0.4（任务书 §3.3） |
| 多核扩展 | 列组分核，核间无 packed 区间交叠、无竞争、无同步；按列加权分配缓解三角负载不均 |
| 单路径统一 | UPLO×INCX×INCY 8 模板特化编译期展开，运行时零分支；负步长 host 归一化 |
| 复用性 | 接口声明入公共头供其他产品线复用；SIMT 列组分块骨架与 spr2 族同构，便于 zher2/her2 等后续精度版本复现 |

---

## 五、可维可测分析

### 5.1 精度标准 / 性能标准

**精度标准**（任务书 §3.2，生态算子开源精度标准 COMPLEX64 档）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|------------------------|---------------------|
| COMPLEX64（实/虚部独立判定） | 2⁻¹³ (1.22e-4) | 2⁻¹³ (1.22e-4) | 0.99 | 1e-2 或 32×ULP |

通过条件：实部/虚部分别满足 `|actual - golden| ≤ atol + rtol × |golden|` 的
元素占比 ≥ 0.99，且 `max_abs_error ≤ 1e-2 或 32×ULP`；golden 采用 double
精度累加（近精确参考，避免 float32 舍入顺序差异干扰 32×ULP 判定），对角虚
部置 0 与标杆一致；Inf/NaN 按任务书等效规则判定（同为 NaN、同为同号 Inf、
NaN 与 ±Inf 等效；有限值与非有限值不等效）。

**性能标准**（任务书 §3.3，warmup 后有效采样取平均，对标 gpu_baseline.csv
200 条用例）：

| 验收项 | 标准 |
|--------|------|
| 单用例性能 | 实测耗时 ≤ H100 对标耗时 / 0.4 |
| 采集方式 | 与 gpu_baseline.csv 用例一一对应，逐条对比取达标率 |

**自测设计**：GTest 参数化 CSV 驱动测试（1200 用例：1000 精度 + 200 性能），
维度覆盖：

| 维度 | 覆盖内容 |
|------|----------|
| UPLO | UPPER / LOWER 全覆盖（packing 索引两条路径） |
| n 扫描 | 1/2/3/4/8/16/32/64/128/256/…（小 n 边界 + 中大 n） |
| 步长 | incx/incy ∈ {±1, ±2, ±3, ±4, 大步长}（正负全组合） |
| alpha | (1,0)/(0,0)（no-op）/非实数复数/极小/极大 |
| 填充 | RANDOM_NORM / RANDOM_EXTREME（溢出邻域）/ NaN / Inf / 全零 |
| 异常 | handle 空 / n<0 / n=0 / alpha 空 / uplo 非法 / 步长 0 或 INT_MIN / x/y/ap 空 |
| 性能 | 与 gpu_baseline.csv 对应的 200 条 TC_PF 用例 |

异常类用例在测试工程 wrapper 层配合空指针守卫（Device 侧空指针跳过上传/
回拷），保证异常路径可稳定回归。

### 5.2 兼容性分析

- 新算子，无历史版本，不涉及接口/行为兼容性问题。
- 接口声明放入公共头 `include/cann_ops_blas.h`（新增声明，master 现无），与其
  他产品线共用同一 `aclblasChpr2` API，禁止 950PR 私有平行接口，保证后续产品
  线扩展兼容。
- 仓内 `blas/spr2/README.md` 产品支持表与 `docs/zh/api_list.md` 新增
  `aclblasChpr2` 条目，标注 Ascend 950PR：支持。
