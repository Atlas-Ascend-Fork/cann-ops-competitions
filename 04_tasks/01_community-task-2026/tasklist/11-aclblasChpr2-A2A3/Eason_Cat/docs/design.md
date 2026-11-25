# 设计文档头信息

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 202611社区任务-aclblasChpr2算子开发（A2/A3） |
| 提交团队 | Eason_Cat |
| 任务讨论 | [ops-blas discussions #87](https://gitcode.com/cann/ops-blas/discussions/87) |
| 适配硬件 | Atlas A2/A3，`arch22` |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas` |
| 代码目标目录 | `blas/spr2/arch22/（新增 chpr2_*，与既有 sspr2_* 并存）` |
| 测试目标目录 | `test/spr2/chpr2/arch22/` |
| 文档状态 | 设计文档提交 |

# 需求背景（required）

## 需求来源

社区任务「aclblasChpr2算子开发（A2/A3）」。补齐 Hermitian packed 秩-2 更新。

语义裁决：cuBLAS `cublasChpr2` → Netlib `chpr2` → 仓内 arch35 `chpr2_*` → arch22 `sspr2_*` 写法。

## 背景介绍

### 数学

```text
AP := alpha * x * y^H + conj(alpha) * y * x^H + AP
```

只更新 `uplo` 指定三角；对角结果虚部清零（Hermitian）。`AP` 长度 `n(n+1)/2`，原地更新。

接口（已声明）：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* ap);
```

### 现状

- arch35：`chpr2_kernel.cpp` 全程 **SIMT**（`__simt_callee__` 按 packed 元素分线程），A2 不可搬。
- arch22：已有 `sspr2`（列 slab + UB pack，无 SIMT），是本任务骨架。
- 同族 `chpr`（秩-1）arch22/arch35 均已有，可对照对角虚部清零与共轭外积口径。

## 口径裁决

1. `n == 0` → SUCCESS。
2. `n > 0` 后校验 alpha/uplo/inc/x/y/ap；`alpha == (0,0)` → SUCCESS 且不写 AP（arch35 在指针检查之后）。
3. `incx/incy` 拒绝 0 与 `INT_MIN`；span 字节长度防 `size_t` 溢出（照抄 arch35）。
4. alpha 仅 Host 指针模式（与任务书/arch35 注释一致）。

# 需求分析（required）

在 arch22 实现 complex64 packed HER2；Ascend C 直调；不引入 SIMT/tensor_api。

# 详细设计（required）

## 实现方案

### 目录

```text
blas/spr2/arch22/chpr2_host.cpp
blas/spr2/arch22/chpr2_kernel.cpp|.h
blas/spr2/arch22/chpr2_tiling_data.h
test/spr2/chpr2/arch22/
```

### Host

复用 arch35 校验顺序；tiling 改为按 AIV 核做 packed 列/元素均衡切分（对齐 `Sspr2CoreSegStart` 思路），字段含 n/uplo/alphaRe/alphaIm/incx/incy/xStart/yStart/packedLength/elementsPerBlock。

### Kernel（基于 sspr2 arch22）

1. **向量 pack**：把 x、y 按逻辑顺序打进 UB（正负步长处理同 sspr2 注释块）。
2. **列 slab**：UPPER/LOWER 按 packed 列区间批处理；GM→UB `DataCopyPad`。
3. **更新**：对每个 packed 元素算  
   `u = alpha * x_i * conj(y_j)`，`v = conj(alpha) * y_i * conj(x_j)`，`AP += u + v`；对角取实部。  
   复数乘法拆成 float 乘加，舍入顺序对齐 Netlib（contract-free）。
4. 禁止 SIMT 逐元素 GM store；尾块仍走向量/标量 UB 路径但保持在 Ascend C 模型内。

### 与 arch35 差异

| arch35 | arch22 本设计 |
| --- | --- |
| SIMT 按 packedIndex 二分找列 | 列 slab + 闭式分段 |
| 线程直接写 GM | UB 聚合后 DataCopy 回写 |

## TilingData（草案）

```cpp
struct Chpr2TilingData {
    uint32_t n;
    uint32_t uplo;
    float alphaReal;
    float alphaImag;
    int64_t incx;
    int64_t incy;
    int64_t xStart;
    int64_t yStart;
    uint64_t packedLength;
    uint64_t elementsPerBlock;
    uint32_t useCoreNum;
    uint32_t packMode; // 0: gather; 1: stride1; 2: stride2 probe（对齐 sspr2）
};
```

## 多核切分

核 `c` 负责 packed 元素半开区间 `[start(c), start(c+1))`，`start` 由与 `sspr2` 相同的加权前缀公式给出，保证 UPPER/LOWER 列长差异下负载相对均衡。核内再按 UB 预算把列打成 slab。

# 测试与验收

| 项 | 口径 |
| --- | --- |
| 精度 | 任务 CSV；mixed_tolerance；对角虚部位级为 0 |
| 边界 | n=0/1、alpha=0、上下三角、正负步长、INT 边界 |
| 性能 | 任务书 TC_PF；采样/预热以任务书为准 |
| 回归 | 既有 `sspr2` arch22 全绿 |

# 风险

1. Hermitian 共轭外积比 sspr2 实数外积易在对角/共轭方向写错。  
2. complex 存储使 slab UB 预算更紧，需下调列 batch。  
3. 与 `chpr` arch22 对角清零口径必须一致，避免同族行为分裂。
