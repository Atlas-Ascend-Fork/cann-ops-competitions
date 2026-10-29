# aclblasChpr2 算子设计文档

> 任务：10 月社区任务 · aclblasChpr2 算子开发（Ascend 950PR）
> 文档结构参照 cann-competitions `design_template.md`（需求背景 / 需求分析 / 详细设计 / 可维可测分析）

## 1 需求背景

### 1.1 需求来源

CANN 社区 10 月社区任务：在 Ascend 950PR 上使用 Ascend C 开发 `aclblasChpr2`，补齐 ops-blas 仓 BLAS Level-2 复数 Hermitian packed 秩-2 更新能力，验收通过后合入 <https://gitcode.com/cann/ops-blas>。

### 1.2 背景介绍

BLAS `chpr2` 计算 `A := alpha·x·y^H + conjg(alpha)·y·x^H + A`，其中 `A` 为 n×n Hermitian 矩阵，以 packed 列优先格式存储上/下三角（无前导维度）。ops-blas 仓当前已有实数同族 `aclblasSspr2`（packed 对称秩-2）、复数秩-2 `aclblasCsyr2`（full 存储）与复数 packed 秩-1 `aclblasChpr`（仅 arch22），**缺少 chpr2**。

### 1.3 仓内实现现状分析

| 算子 | 目录 | 与 chpr2 关系 | 950 支持 |
|------|------|---------------|----------|
| aclblasSspr2 | `blas/spr2/arch35` | 实数 packed 秩-2，索引/分核同构 | 是 |
| aclblasCsyr2 | `blas/syr2/arch35` | 复数秩-2（不共轭），alpha 复数 Host 指针 | 是 |
| aclblasChpr | `blas/hpr/arch22` | 复数 Hermitian packed 秩-1，共轭/对角语义参考 | 否（仅 arch22） |
| aclblasCher2k | `blas/her2k/arch35` | Hermitian 秩-2k（level-3），语义参考 | 是 |

结论：chpr2 需新增；实现形态沿用同目录族 `sspr2` 的 arch35 SIMT 直调工程，复数运算与 Hermitian 语义参考 `csyr2`/`chpr`。

### 1.4 功能分析

- 输入：`uplo`、`n`、复数标量 `alpha`（Host）、复数向量 `x/y`、packed 复数矩阵 `ap`；
- 输出：`ap` 原地更新；仅 `uplo` 指定三角被读/写；对角虚部输出置 0；
- quick return：`n = 0` 或 `alpha = (0,0)` 返回成功且不访问数据；
- 步长：任意非零 `incx/incy`，支持负步长。

## 2 需求分析

### 2.1 需求描述

使用 Ascend C 在 Ascend 950PR（arch35）实现 `aclblasChpr2` 的句柄式接口，经 handle 绑定的 stream 直调 NPU kernel；接口声明加入 `include/cann_ops_blas.h`，供各产品线共用。

### 2.2 需求拆解

1. 接口与 cuBLAS `cublasChpr2` 参数一一对应，`alpha` 为 Host 指针；
2. packed 索引：UPPER `i + j(j+1)/2`（i≤j）、LOWER `i + j(2n-j-1)/2`（i≥j）；
3. 参数校验顺序固定（handle → n<0 → n=0 → alpha 判空 → uplo → incx/incy → x/y/ap → alpha=0）；
4. 精度满足 FLOAT32 档（rtol=atol=2^-13、ratio≥0.99、max_abs≤1e-2）；
5. 性能 ≤2.5× GPU H100 标杆；
6. 代码合入 `blas/spr2/arch35/`、测试合入 `test/spr2/chpr2/`。

## 3 详细设计

### 3.1 算子分析

**数学公式**

```
A(i,j) += alpha · x(i) · conj(y(j)) + conj(alpha) · y(i) · conj(x(j))     (i,j) ∈ uplo 三角
```

**数据类型**：`aclblasComplex`（real/imag 各 FP32），alpha 同类型（Host）。

**存储与形状**：`ap` 长度 `n(n+1)/2`，无 lda；`x/y` 物理长度 `1+(n-1)|inc|`。

### 3.2 算子实现

#### 3.2.1 Host 侧设计（`chpr2_host.cpp`）

校验顺序（与任务书 §2.5 一致）：

| 步 | 条件 | 返回 |
|----|------|------|
| 1 | handle 为空 | `HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `INVALID_VALUE` |
| 3 | `n == 0` | `SUCCESS`（不访问其余参数） |
| 4 | alpha 为空 | `INVALID_VALUE` |
| 5 | uplo 非法 | `INVALID_VALUE` |
| 6/7 | `incx/incy == 0` 或 `INT_MIN` | `INVALID_VALUE` |
| 8 | x/y/ap 为空 | `INVALID_VALUE`（即使 alpha=0） |
| 9 | `alpha == (0,0)` | `SUCCESS`（不访问 x/y/ap） |

切分：`blocks = min(n, AIV)`，`threads = min(align_up(ceil(n/blocks), 128), 2048)`；`alpha` 在 Host 解引用后以 `alphaReal/alphaImag` 写入 tiling；不申请 workspace、不落 tiling GM、Host 不同步。

#### 3.2.2 Kernel 侧设计（`chpr2_kernel.cpp`）

- 形态：`extern "C" __global__ __aicore__` + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；
- 两条 SIMT VF 路径：
  - **K0**：GM 直读，支持任意 n 与任意非零 incx/incy；
  - **K1**：`incx=incy=1` 且 `32≤n≤4096` 时把 x/y 装入 UB（`2×8192` FP32 lane = 64 KiB），减少重复 GM 读；
- **列级 grid-stride**（`col = blockIdx.x; col += gridDim.x`）：块间负载峰值/均值 ≤ `1+2G/n`；
- **负步长指针偏置**：`base = gm + 2·(n-1)·|inc|`、`step = 2·inc`，内层无分支；
- **packed 索引**：编译期模板 `UploIsUpper` 消除分支，索引全程 `uint64_t`；
- **对角**：`alpha·u + conj(alpha·u) = 2Re(alpha·u)`，只写实部、虚部置 0、不读旧虚部；
- **运算顺序与 Netlib 对齐**：列级 `t1 = alpha·conj(y_j)`、`t2 = conj(alpha·x_j)`，行级 `ap += x_i·t1 + y_i·t2`；避免 `alpha` 乘在已溢出中间量上产生 `0×Inf = NaN`；
- 浮点语义：`#pragma clang fp contract(off)`，和序固定 `(old+t1)+t2`；
- UB 容量编译期校验：`static_assert(2×8192×4 ≤ UB_SIZE)`。

#### 3.2.3 分核与边界

| n | blocks | threads | 说明 |
|---|--------|---------|------|
| < 32 | min(n, AIV) | 128 | 走 K0 |
| 32–4096（inc=1） | min(n, AIV) | 128–2048 | 走 K1 |
| > 4096 | 28 | ≤2048 | 走 K0 |

关键边界：n=0/1、31/32/33、4095/4096/4097、8192、INT_MIN/0 步长、Inf/NaN、极端值填充。

### 3.3 支持硬件

| 芯片版本 | 支持 |
|----------|------|
| Ascend 950PR / 950DT | √（arch35，`blas/spr2/arch35/`） |
| Atlas A3 / A2 系列 | ×（本批次仅提供 arch35 实现） |

### 3.4 算子约束限制

- `n ≥ 0`，无显式上限（受 HBM 与 packed 长度约束）；
- `incx/incy ≠ 0` 且 `≠ INT_MIN`；
- `alpha` 仅支持 Host 指针（Device 标量不在验收范围）；
- 仅 `uplo` 指定三角被引用/更新；对角虚部输入假定为 0。

## 4 可维可测分析

### 4.1 精度与性能标准

| 验收标准 | 描述 | 标准来源 |
|----------|------|----------|
| 精度 | FLOAT32 档：rtol=atol=2^-13，matched_ratio≥0.99，max_abs≤1e-2；实部/虚部分别判定；golden=cblas_chpr2 | 任务书 §3.2 / 生态算子开源精度标准 |
| 性能 | 平均单次耗时 ≤ GPU H100 标杆 / 0.4（≤2.5×） | 任务书 §3.3 |

### 4.2 实测结果

| 项 | 结果 |
|----|------|
| 精度 | 1200/1200 通过；1077 条计算类实/虚部 ratio 全为 1.000000，最大绝对误差 4.88e-4 |
| 性能 | 200/200 通过；比值 min 0.764 / median 1.272 / max 2.028；n=4096 时 0.98×GPU |

### 4.3 兼容性分析

新增接口与文件，不修改既有算子实现；`include/cann_ops_blas.h` 仅追加声明，向后兼容。测试工程位于 `test/spr2/chpr2/`，不影响 `test/spr2/` 既有 `sspr2_test`。

## 5 交付与验证入口

```bash
git clone https://gitcode.com/cann/ops-blas.git && cd ops-blas
bash build.sh --soc=ascend950 --ops=chpr2 --run
```

测试代码：`test/spr2/chpr2/`（CSV 驱动 + 6 个固定用例 + 200 条性能扫描）。
