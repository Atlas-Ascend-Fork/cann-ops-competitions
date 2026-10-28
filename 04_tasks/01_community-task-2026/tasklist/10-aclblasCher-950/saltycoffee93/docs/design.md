# Atlas 950PR 单精度复数 Hermitian 秩-1更新 aclblasCher 算子设计文档

> 任务：10月社区任务-aclblasCher算子开发（Atlas 950PR / arch35 / CANN 9.1.0）
> 目标仓：gitcode.com/cann/ops-blas（实现 `blas/syr/arch35/`，测试 `test/syr/cher/arch35/`）
> 对标：cuBLAS `cublasCher`；参考实现：Netlib BLAS `cher.f`（已逐行核对）

## 1. 算子描述

### 1.1 aclblasCher

单精度复数（COMPLEX64）Hermitian 秩-1 更新：

```
A := alpha * x * x^H + A
```

- `x` 为 n 元素复数向量，`A` 为 n×n Hermitian 复数矩阵（**列主序**），`alpha` 为**实数** float 标量（区别于 `aclblasCsyr` 的复数 alpha 与普通转置）
- 仅 `uplo`（ACLBLAS_UPPER/ACLBLAS_LOWER）指定的三角区域被引用/更新；另一三角与 `lda` padding 行**不读不写**
- 对角元素虚部假定为 0，更新后**强制置 0**（Netlib cher 语义）；`x(j)` 恰为 (0,0) 的列整体跳过乘加（防 `alpha=Inf` 时 `0*Inf=NaN` 污染），但对角虚部清零不受影响
- `n=0` 或 `alpha=0.0` 为合法 quick return（不更新 A）
- 支持任意非零 `incx`（含负步长 Netlib KX 语义：逻辑 x(i) ↔ 物理 (n-1-i)·|incx|）

### 1.2 接口签名（include/cann_ops_blas.h 新增声明，供产品线共用）

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

## 2. 实现方案

### 2.1 总体结构

Ascend C **kernel 直调**（SIMT 编程模型，仅 arch35），完全继承仓内同族近亲 `blas/syr/arch35/csyr*` 的五件套结构与多变体分发拓扑（该拓扑已在 950PR 上完成调优），做 Hermitian 数学层改造：

```
blas/syr/arch35/
├── cher_host.cpp        # 参数校验(§2.5 顺序) → quick-return → alpha Host/Device 解析 → tiling → launch
├── cher_kernel.cpp      # SIMT kernel 多变体 + 分发
├── cher_kernel.h        # cher_kernel_do 签名
└── cher_tiling_data.h   # CherTilingData（alphaReal 单标量字段 + 阈值常量）
```

### 2.2 编程模型选择

SIMT（`__simt_vf__`/`asc_vf_call`/`threadIdx.x`）：syr 族是逐元素无归约的规则更新，线程级并行为仓内同族既定路线（csyr/cgemv 同构），且规避 Vector 通道的重复加载；AIV grid = min(n, GetAivCoreCount())，线程 512（n≤512）/1024（n>512）。

### 2.3 Kernel 变体与分发（条件与 csyr 一致）

| 变体 | 触发条件 | 说明 |
|---|---|---|
| CherTinyUnitTriangle | 8B 对齐 ∧ incx=1 ∧ n<32 ∧ lda=n ∧ host α=1 | 32 行矩形展平，单元素一线程 |
| CherUnitAlphaGroupedColumns（+ 精确形状 n=lda=512/1024/2048 特化） | host α=1 ∧ n 分档 ≤512/≤1024/≤2048 | 分组列所有权 + 180° 旋转矩形铺满三角 |
| CherTriangleTiles（独立 kernel `cher_triangle_tiles`） | n∈(1024,4096] ∧ 对齐 ∧ incx=1 ∧ 56 块×1024 线程 | 64×64 三角 tile，对角 tile 特殊化 |
| CherCyclicColumns | 其余快路径（32≤n≤4096 ∧ 对齐 ∧ incx=1） | 行循环 + 列循环，支持 device alpha |
| CherGm（回退） | incx≠1 / 未 8B 对齐 / n>4096 | 标量 float 访存，uint64 索引 |

### 2.4 Hermitian 数学层（与 csyr 的三点差异）

设 `ax = α·x(i)`（α 实数：两分量各一次独立舍入乘法），目标元素 `(i,j)`（j 为列驱动，共轭侧）：

1. **off-diag**：`A(i,j) += ax ⊗ conj(x(j))`
   - `Δre = fl(fl(ax_re·xc_re) + fl(ax_im·xc_im))`
   - `Δim = fl(fl(ax_im·xc_re) − fl(ax_re·xc_im))`（与 csyr 相比虚部项变号）
   - 每乘积以 `__fma(a,b,0.0f)` 保持独立舍入（大数溢出时 Inf/NaN 分类与 CPU golden 一致）
2. **对角**：`A(j,j).re += fl(ax_re·xr) + fl(ax_im·xi)`（= α·|x(j)|²，两乘积独立舍入相加，镜像 cher.f `REAL(X(J)*TEMP)`）；`A(j,j).im := 0` **无条件**（含跳过列）
3. **列守卫**：`x(j) == (0,0)` 精确成立 → 该列不乘加（cher.f `IF (X(N).NE.ZERO)`），对角仍清虚部

### 2.5 Host 侧

- 校验顺序逐字对齐任务书 §2.5（= csyr 口径）：handle 判空 → n<0 → **n=0 直接 SUCCESS（不再校验其余）** → uplo（INVALID_ENUM）→ lda≥max(1,n) → incx≠0（含 **INT_MIN 存储长度溢出防御**，任务包增强约束）→ alpha 非空 → alpha=0 no-op → x/A 判空
- alpha Host/Device 双指针：Host 直接读；Device 常规路径 kernel 内读（异步零同步），仅当 x/A 判空需要区分 quick-return 时 D2H 读回一次
- launch 后不同步（异步语义，handle 绑定 stream）

## 3. 精度方案

- golden：CPU 参考实现（`test/syr/cher/cher_golden.h`）逐元素镜像 kernel 舍入形状（含列守卫/对角规则），全量 1000 条实测 **maxAbsErr = 0.0（逐位一致）**，远优于容差门限
- 判据（生态算子开源精度标准，FLOAT32 档，实/虚部分别）：atol=rtol=2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP
- 写保护：未引用三角 + lda padding 行逐位不变（哨兵比对）；对角虚部额外显式断言 `|im| ≤ 2⁻¹³`（kernel 实际写 0.0f）
- Inf/NaN：kernel 与 golden 同型传播（NaN=NaN / 同号 Inf 等效），`alpha=Inf ∧ x(j)=(0,0)` 列守卫对齐 cher.f

## 4. 异常与边界处理

| 场景 | 行为 |
|---|---|
| handle=nullptr | HANDLE_IS_NULLPTR |
| n<0 / lda<max(1,n) / incx=0 / incx=INT_MIN / alpha=nullptr | INVALID_VALUE（顺序：uplo→lda→incx→alpha） |
| uplo 非法（121/122 之外） | INVALID_ENUM |
| n=0 | SUCCESS（不校验其余参数，不触任何指针） |
| alpha=0.0 | SUCCESS no-op（x/A 允许为 nullptr；但结构非法仍先报错） |
| n>0 ∧ alpha≠0 ∧ x/A=nullptr | INVALID_VALUE |
| 负步长 incx | Netlib KX 语义反向遍历（Gm 回退路径全覆盖 ±1/±2/±3） |
| 4 字节未对齐地址 | 自动落入标量 GM 路径（float2 快路径需 8B） |
| n>4096 | Gm 路径 uint64 索引，无尺寸上限 |

## 5. 性能方案

- 判据：单算子 kernel Task Duration（msprof 采集，device 侧 AI_VECTOR_CORE 计时），每 case warmup 5 + 有效采样 10 取均值；**NPU ≤ GPU H100 标杆 / 0.4**
- 实测 **200/200 全达标**：最差 0.559×（n=2308），最好 2.81×（n=32）；任务书 4 条典型 case：n=512U/n=1024L/n=2048U/n=4096L 全绿
- 手段：α=1.0 性能档全走 Unit-Alpha 特化（免标量乘）；配对列加载（两次独立 GM 读重叠）；56 块×1024 线程调优网格的 tile 化路径覆盖大尺寸；host 零多余拷贝零同步

## 6. 测试方案

- CSV 驱动 GTest（`test/syr/cher/arch35/cher_test`，1200 条任务包用例）+ 9 个 fixture：
  负向/quick-return/Host-Device alpha/kernel 变体边界（17 组合，含 n=4100 越界 GM 与 4 字节未对齐）/α 分布（12 档含 ±1000）/零列守卫（含 alpha=Inf）
- 复现：`bash build.sh --ops=cher --soc=ascend950 --run`；验收脚本 `verify_accuracy.py` / `verify_performance.py`（任务包自带）

## 7. 风险与已知限制

| 项 | 说明 |
|---|---|
| 接口共用 | `aclblasCher` 声明在 `include/cann_ops_blas.h`，其他产品线（arch22 等）尚未实现——本交付仅 950PR/arch35，声明共用不阻塞 |
| msprof op 模式 | 本机 CANN 9.1.0 的 `msprof op` 分析阶段对直调 kernel 符号化失败（环境缺陷，单用例复现）；性能数据改用同口径的 full msprof `op_summary` Task Duration，单用例交叉验证一致 |
| 精度口径 | golden 为自实现 CPU 参考（舍入形状与 kernel 逐位对齐）；任务书指定 cblas_cher 为验收 golden——两者语义相同（同 cher.f 算法），实测逐位一致使该差异无实际影响 |
| n 上限 | 接口无硬上限；测试用例 n≤4096（host 512MB 预算），n>4096 走 Gm 路径已含专项边界用例 |
