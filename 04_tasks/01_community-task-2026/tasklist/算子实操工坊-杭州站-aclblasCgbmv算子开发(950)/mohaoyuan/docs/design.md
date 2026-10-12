# 需求背景(required)

## 需求来源

通过【算子实操工坊-杭州站】社区任务,向昇腾算子开源仓 ops-blas 贡献单精度复数带状矩阵-向量乘算子
`aclblasCgbmv`(Ascend 950PR / arch35)。目标合入目录:代码 `blas/gbmv/arch35/`,
测试 `test/gbmv/cgbmv/arch35/`。

## 背景介绍

### aclblasCgbmv 算子实现

ops-blas 仓已有实数版 `aclblasSgbmv`(`blas/gbmv/arch35/` 三件套,**单 kernel AIV/SIMT 架构**:
带状矩阵-向量乘为带宽受限的 Level-2 算子,无立方计算),公共头文件 `include/cann_ops_blas.h`
原先无 `aclblasCgbmv` 声明。本任务在同一目录族新增 complex64 接口,参数序列与
cuBLAS `cublasCgbmv` 一一对应,语义对齐 Netlib `cgbmv`(参考 https://www.netlib.org/blas/cgbmv.f )。

本算子不是 TBE 迁移,而是 ops-blas 句柄式 BLAS + Ascend C kernel 直调(arch35 / DAV_3510)。

### 标杆现状分析

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库句柄 | scalar | aclblasHandle_t | 非空,携带 stream | - |
| trans | op(A) 选择 | attr | N / T / C | C 为共轭转置;非法返回 INVALID_VALUE | - |
| m | A 的行数 | scalar | int | ≥0;0 为 no-op | - |
| n | A 的列数 | scalar | int | ≥0 | - |
| kl | 下带宽 | scalar | int | ≥0 | - |
| ku | 上带宽 | scalar | int | ≥0 | - |
| alpha | 复数标量 | COMPLEX64 | Host 或 Device 指针 | 不可为 nullptr | - |
| A | 带状矩阵 | COMPLEX64 | 列主序 ND | 仅 (kl+ku+1)×n 带内元素被引用 | lda ≥ kl+ku+1 |
| x | 输入向量 | COMPLEX64 | ND | 步长 incx 可为负 | max(1,有效长) |
| beta | 复数标量 | COMPLEX64 | Host 或 Device 指针 | (0,0) 时 y 不读取 | - |
| y | 输入/输出向量 | COMPLEX64 | ND | 原地更新 y = alpha·op(A)x + beta·y | 步长 incy 可为负 |

计算公式:

- `y := alpha · op(A) · x + beta · y`
- `op(A) ∈ {A, Aᵀ, Aᴴ}`,A 为 m×n 复数带状矩阵(kl 下带 + ku 上带)

标杆(cuBLAS/Netlib)语义要点:beta=(0,0) 时 y 不被引用(可为无效输入);m=0 或 n=0
直接返回;带外元素不引用;incx/incy 支持负步长(反向索引)。

# 需求分析(required)

## 需求描述

使用 Ascend C(AIV/SIMT 向量编程)在 Ascend 950PR(arch35/DAV_3510)实现 `aclblasCgbmv`:
COMPLEX64、trans N/T/C 全覆盖、alpha/beta 双位置(Host/Device)支持、负步长支持、
beta=(0,0) 与 m/n=0 快路径,精度满足生态算子开源精度标准,性能不高于任务书标杆
(任务书领取时同步,带宽受限 L2 算子)。

接口签名(新增于公共头文件,紧随 `aclblasSgbmv` 之后,供产品线共用):

```cpp
aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

## 需求拆解

1. 公共头文件新增 `aclblasCgbmv` 声明(禁止 950PR 私有平行接口)
2. host 侧:参数校验(cuBLAS 口径)、alpha/beta 复数双位置解析、快路径、tiling、kernel 编排
3. kernel 侧:复数带状点积 + 复数 beta·y 原地融合,单 kernel(AIV/SIMT)
4. trans=C 共轭语义(op(A)=A^H:转置 + 元素共轭)
5. 负步长(incx/incy<0)索引支持
6. 测试工程:CSV 驱动 GTest,golden 由 cblas_cgbmv 生成,复数 mixed tolerance 判定,
   TC_PF 计时路径(warmup + 采样均值)

# 详细设计(required)

## 算子分析

### 数学公式

```
y := alpha · op(A) · x + beta · y          (op ∈ {A, Aᵀ, Aᴴ})

对每个输出元素(trans=N 时 i ∈ [0,m),其余经转置映射):
  y[i] = (beta_r + i·beta_i) · y[i] + (alpha_r + i·alpha_i) · Σ_j op(A)[i][j] · x[j]
  j ∈ 带内区间:trans=N → [max(0,i-kl), min(n,i+ku+1)),其余按 op 映射
```

复数乘加按 (a+bi)(c+di) = (ac−bd) + (ad+bc)i 展开,SIMT 线程内标量累加。

### 支持数据类型

COMPLEX64(实部/虚部各 float32,`aclblasComplex`)。

### 支持形状

m,n ≥ 0 任意;kl,ku ≥ 0;带宽 w=kl+ku+1;lda ≥ max(1, w);incx/incy ≠ 0(支持负值)。

## 算子实现

### 实现方案

**单 kernel 融合**(继承 sgbmv 单 kernel 架构,复数化):

```
[K1] gbmv kernel (AIV·SIMT,唯一 kernel)
     按 outDim(= trans==N ? m : n)均分到向量核;每线程串行处理若干输出元素
     每个输出元素:带内复数点积(alpha 乘) + beta·y 原地累加 + 步长寻址(含负步长)
```

无 workspace 依赖(带内点积寄存器内完成)。

#### host 侧设计

**(1) 参数校验顺序**(cuBLAS 口径,与 golden 同序):

```
handle 非空(否则 HANDLE_IS_NULLPTR)
→ trans 枚举合法(否则 INVALID_VALUE)
→ m/n/kl/ku ≥ 0(否则 INVALID_VALUE)
→ m==0 或 n==0:直接 SUCCESS(Netlib 快速返回)
→ lda ≥ max(1, kl+ku+1)、incx/incy ≠ 0、alpha/beta 非空(否则 INVALID_VALUE)
→ alpha/beta 位置解析(Device 时 8B D2H + 同步)→ A/x/y 非空(否则 INVALID_VALUE)
→ beta=(0,0) 时 y 仅写不读;alpha=(0,0) 时 y = beta·y 缩放路径
```

**(2) 复数 alpha/beta 双位置支持**:`CheckPtrLocation` 判定;Device 指针各 8 字节
D2H 异步拷回 + stream 同步;解析结果以 (re, im) 标量对写入 tiling。

**(3) tiling 策略**(sgbmv 同款):outDim 按核均分,
`rowsPerBlock = CeilDiv(outDim, numBlocks)`,
`numThreads = clamp(CeilAlign(rowsPerBlock, 128), 128, 2048)`。

#### kernel 侧设计

**模板分发**:`<TRANS_IS_T, CONJ>`(N/T 两向 × C 共轭,有效 3 组路径):

- 带元素寻址:op(A)[i][j] 的源位置 = trans=N 时 `A[(ku + i - j) + j·lda]`
  (带存储约定),T/C 时索引转置映射
- **trans=C 共轭**:源元素虚部取负
- **负步长**:`x[j·|incx|]` vs 起点偏移 `incx<0 ? (len-1)·|incx| : 0` 的 BLAS 约定
- **beta 原地**:先读 y[i](仅 beta≠(0,0)),写回 `alpha·点积 + beta·y`,单次读写
- 线程映射:每线程 strided 认领输出元素(与 sgbmv 相同的 SIMT 网格)

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR(arch35 / DAV_3510) | √ |
| Atlas A2 训练/推理系列(arch22) | 不支持 |
| Atlas A3 训练/推理系列 | 不支持 |

## 算子约束限制

- 仅 COMPLEX64、列主序带存储、arch35
- 不支持超出 lda/incx/incy 语义的非连续访问
- y 原地更新(无独立输出缓冲);alpha/beta 非空;incx/incy ≠ 0
- 无 broadcast、无动态 shape 要求(m/n/kl/ku 为运行时入参)

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden = Netlib cblas_cgbmv;实部/虚部分别按 FLOAT32 判定:逐元素 \|actual−golden\| ≤ 2⁻¹⁶ + 2⁻¹⁰·\|golden\|,matched_ratio ≥ 0.99,max_abs_error ≤ max(1e-2, 32×ULP);alpha=(0,0) / beta=(0,0) 用例 EXACT 或同语义比对 | 生态算子开源精度标准(https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准 | 任务书标杆 case(领取时同步具体数值);本算子为带宽受限 L2 算子,带内点积 O(m·w),评测口径 warmup + >50 次采样均值 | 任务书 §3.3 |

测试工程:`test/gbmv/cgbmv/`(CSV 驱动 GTest,参照 `test/gbmv/sgbmv/` 扩展复数标量
alpha_real/alpha_imag、beta_real/beta_imag 列;复数向量填充用 makeBlasComplexArray);
覆盖 trans N/T/C × kl/ku 组合(全带宽/窄带/单对角)× 负步长 × alpha/beta 特殊值
((0,0)/(1,0)/纯虚/大值)× 零维/空指针/非法枚举/非法 ld/非法步长负向 × Inf/NaN 填充 ×
TC_PF 计时路径;README 说明复现步骤(保证验收人可复现)。

## 兼容性分析

新算子,不涉及兼容性分析;接口声明放入公共头文件 `include/cann_ops_blas.h`,
与其他产品线共用,不定义 950PR 私有平行 API。

## 风险与应对

| 风险 | 分析 | 应对 |
| --- | --- | --- |
| 极宽带(w≈n)退化为稠密 GEMV,长K累加误差 | w 大时点积长度大,fp32 误差统计尾部可能触碰 max_abs 限 | 带内分段累加 + mixed tolerance(参照 ctrmm 的 32×ULP 口径经验);极端 case 按仓规范豁免论证 |
| 负步长/转置组合索引错误 | 语义组合多(N/T/C × incx/incy 正负) | 专项负向/步长用例全覆盖,golden 与实现同语义 |
| 性能标杆未同步 | 任务书待领取 | 设计按带宽受限模型规划(合并访存 + 每元素单次 y 读写);领取后按标杆校准 |
