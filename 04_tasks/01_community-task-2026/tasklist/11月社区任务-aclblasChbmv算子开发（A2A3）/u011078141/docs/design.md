# aclblasChbmv 算子设计文档（arch22，Atlas A2/A3）

> 提交人：opengood（GitCode: u011078141）。任务：11月社区任务-aclblasChbmv算子开发（A2/A3）

# 需求背景（required）

## 需求来源

《aclblasChbmv 算子开发任务书》（214 行，含 §2.4 参数表 / §2.5 实现约束 / §3 验收 / §5 合入目录）
及配套 test_cases/。目标仓 https://gitcode.com/cann/ops-blas ，实现目录 `blas/sbmv/arch22/`
（与实数同族算子 `aclblasSsbmv` 同目录族，仓内已有 ssbmv 四件套可参照）。
对标 cuBLAS `cublasChbmv`，语义参考 Netlib `chbmv.f`。任务讨论帖：https://gitcode.com/cann/ops-blas/discussions/81 。

## 背景介绍

`aclblasChbmv` 计算 `y = alpha * A * x + beta * y`，其中 A 为 n×n **Hermitian 带状**矩阵
（k 条次对角 + k 条超对角），x/y 为 n 元素向量，全部为 complex64。A 按列主序带状格式存储于
(k+1)×n 数组，仅 `uplo` 指定的带形三角被引用，另一侧由 Hermitian 共轭对称性隐含补全，
对角元素虚部假定为 0（只取实部参与计算）。本算子在 ops-blas 仓为**新增接口**
（`include/cann_ops_blas.h` 尚无声明，需按 §2.3 新增，与 950PR 等其他产品线共用）。

# 需求分析（required）

## 接口定义（新增，按任务书 §2.3）

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

`aclblasComplex` 定义复用 `include/cann_ops_blas_common.h`（实部/虚部各 float32）。

## 需求拆解

1. **带状索引语义**（核心正确性风险）：
   - UPPER：A(i,j) 存于 `A(1+k+i-j, j)`，i∈[max(1,j-k), j]，主对角在第 k+1 行，左上 k×k 不引用；
   - LOWER：A(i,j) 存于 `A(1+i-j, j)`，i∈[j, min(n,j+k)]，主对角在第 1 行，右下 k×k 不引用；
   - 未存储侧按 A(j,i) = conj(A(i,j)) 补全；对角只取实部。
2. **校验顺序（任务书 §2.5 明确，含维护者拍板口径 #765/#861）**：
   handle 非空 → n ≥ 0 → **n=0 直接返回 SUCCESS** → uplo/k/lda/incx/incy/alpha/beta/A/x/y。
   注意点：`handle` 判空**最优先**，quick return 不得先于判空（与 aclblasGemmBatchedEx 的
   "校验全部通过后才 no-op"口径相反，本任务书以本条为准）。
3. **整数溢出防护**（issue #780/#866 口径）：`lda ≥ k+1` 的 k+1 在 int64 域计算（防 k=INT32_MAX
   回绕穿透）；`(n-1)×|incx|`、`(n-1)×|incy|` 也在 int64 域计算，超 int32 表示范围返回
   INVALID_VALUE；incx/incy 为 0 或 INT32_MIN 均非法。
4. **非法 uplo 返回 `ACLBLAS_STATUS_INVALID_VALUE`**（§7-5 研发确认口径；注意仓内复数同族
   `aclblasCsymv` 返回 INVALID_ENUM，本算子不跟随，差异由研发后续统一）。
5. **边界语义**：n=0 quick return（不执行计算）；beta=(0,0) 不读 y 旧值；alpha=(0,0) 时
   y=beta×y，实现不保证跳过 A/x 读取（故含 Inf/NaN 的 x 用例不得与 alpha=(0,0) 组合）；
   k > n-1 时按有效带宽 min(k,n-1) 计算。
6. **精度**：golden 由 `cblas_chbmv`（Netlib CBLAS 复数，CblasColMajor，fp64 真值计算后按输出
   FLOAT32 收窄）逐元素比对实部与虚部，FLOAT32 容差 rtol=atol=2^-13、matched_ratio≥0.99、
   max_abs_error≤1e-2 或 32×ULP（ULP 按最大误差点 golden 动态计算，2026-09-24 修订口径）；
   Inf/NaN 按任务书等效判定规则单独统计。golden 生成须按 Hermitian 约束解释 A。
7. **性能**：910B3，msprof op 采 task duration，5 warmup + ≥10 有效样本均值，单 case ≥0.8×GPU
   （标杆数据见 test_cases/gpu_baseline.csv）。

# 详细设计（required）

## 算子分析

y_i = alpha·Σ_{j:|i-j|≤k} Ã(i,j)·x_j + beta·y_i，其中 Ã 为按 Hermitian 规则补全后的矩阵
（对角取实部、未存储侧共轭对称）。全部运算在 complex64：乘加按 (ar·br − ai·bi, ar·bi + ai·br)
展开，累加精度 fp32。

## 实现方案

### 工程结构（与实数同族 ssbmv/arch22 同构）

| 文件 | 内容 |
|---|---|
| `blas/sbmv/arch22/chbmv_host.cpp` | 校验 + tiling + launch、handle/stream 接管 |
| `blas/sbmv/arch22/chbmv_kernel.cpp` | 设备侧 kernel：带状元素遍历 + 复数乘加 |
| `blas/sbmv/arch22/chbmv_kernel.h` | kernel 入口与 tiling 结构声明 |
| `blas/sbmv/arch22/chbmv_tiling_data.h` | host→kernel tiling 结构（`static_assert` 锁偏移） |
| `include/cann_ops_blas.h` | 新增 `aclblasChbmv` 声明（与其他产品线共用） |
| `test/sbmv/chbmv/arch22/` | CSV 驱动 GTest + golden + README 复现步骤 |

### Host 侧设计

1. **校验**：按 §2.5 顺序实现（handle → n → quick return → uplo → k → lda → incx → incy →
   alpha/beta/A/x/y 非空）。`lda ≥ k+1` 与跨度算式一律 int64（对齐仓内 ssbmv 的溢出防护手法）。
2. **标量读取**：alpha/beta 为 Host 或 Device 复数标量，host 侧解析为 2×float 随 tiling 下发
   （复用仓内标量读取工具，沿用 ssbmv 的做法）。
3. **带状参数归一化**：uplo → 统一成 (bandOffset, upper) 二元组下发，kernel 侧只有一套索引
   公式，避免两套分支走样；k 先取 `min(k, n-1)`。
4. **tiling 与并行**：按输出向量 y 的行分块（每个核负责若干行），行块内沿有效带 j∈[i-k, i+k]
   压缩成连续区间；带状 A 的列内布局使同一列相邻行地址连续，便于批量搬运。tiling 结构含
   n/k/lda/incx/incy/alpha/beta/uplo/行块划分，编译期 `static_assert` 锁字段偏移。
5. **原地与别名**：x 与 y 可能同址（in-place 语义），按仓内 ssbmv 的 `CopyInPlaceX` 手法处理
   （需要时经 workspace 暂存 x），避免边写 y 边读 x 的自污染。
6. **stream**：经 handle 绑定 stream 异步下发，不创建隐式全局状态。

### Kernel 侧设计

1. **主路径**：每个核处理一段 y 行；对每行 i，沿 j∈[max(0,i-k), min(n-1,i+k)] 累加：
   - j ≤ i（带内下侧/对角）读存储元素；j > i 时按 uplo 决定是读存储还是取共轭（UPPER 存上侧，
     LOWER 存下侧），对角元素只取实部（虚部按 0 处理）；
   - 复数乘加展开为 4 次实数乘 + 2 次加，累加器为 fp32 实部/虚部各一份。
2. **访存**：带状数组列主序，同一列内带内元素连续 → 用 DataCopyPad 按列搬运到 UB 后向量化；
   非对齐/跨界（带边界处）用小块或标量兜底，避免越界读（UPPER 左上 k×k、LOWER 右下 k×k 明确
   不引用，边界判定逐行按 i/j 范围裁剪）。
3. **步长**：incx/incy ≠ ±1 时按步长 gather（x 侧按 incx 索引，y 侧写回按 incy 索引）；
   负步长按起点偏移计算绝对地址，不做数据重排。
4. **边界分支**：n=0 不下发 kernel（host quick return）；k=0 退化为逐元素 y_i = alpha·Re(A_ii)·x_i
   + beta·y_i；beta=(0,0) 不读 y；alpha=(0,0) 只做 beta 缩放（不保证跳过 A/x 读取，与任务书一致）。
5. **复数约定**：complex64 按 (re, im) 交织布局；Hermitian 补全在读取时按需取共轭
   (re, −im)，不做整矩阵展开（省带宽）。

### 精度设计

- 累加为 fp32 实数域双通道（实部/虚部各累加），与 golden 的 fp64 真值比对按 FLOAT32 容差；
- golden 严格按 Hermitian 约束生成（对角虚部 0、带内共轭、带外 0）；
- 长 n 累加顺序差异按 32×ULP 硬界评估；若出现边缘超界，按仓内既有手法引入补偿/分段累加
  （与 aclblasGemmBatchedEx 的精度分级经验同源，但本算子 k 通常远小于 n，先在实测确认真实需要）。

### 性能设计

- 目标 910B3 单 case ≥0.8×GPU；核心手段：列内连续搬运 + 行块并行 + 复数标量最少化；
- k=0/k 小与满带退化分别建 tiling 分支，避免"小带按满带调度"的空转；
- 实测数据上机后回填自测报告，本设计不预填数字。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（800T A2 / 910B3） | √ |
| Atlas A3 系列（arch22 同源） | √ |

## 算子约束限制

- 不支持超出 lda/incx/incy 语义的非连续访问（任务书 §2.5）；
- 依赖 `aclblasSetStream` 绑定 stream，读回前须同步；
- y 原地覆写，不返回视图；不涉及 broadcast / 动态 shape / 确定性要求。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32：rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP；Inf/NaN 按等效规则 | 任务书 §3.2 / opbase 精度标准（commit 28a1ce5） |
| 性能标准 | 910B3 单 case 平均耗时 ≤1.25×GPU（≥0.8×标杆），msprof op 5 warmup + ≥10 样本 | 任务书 §3.3 |

## 测试设计

- CSV 驱动 GTest（仓内 test 框架），逐 case 覆盖：uplo×尺寸×带宽正交、k=0/k=n-1、
  lda padding、incx/incy ∈ {±1,±2,±3}、标量特殊值 (0,0)/(1,0)/纯虚/大值、Inf/NaN，
  负向：非法 uplo/负 n/负 k/lda<k+1/incx=0/incy=0/INT32_MIN 步长/各指针空（n>0）→ INVALID_VALUE，
  handle 空 → HANDLE_IS_NULLPTR，n=0 → SUCCESS；
- **补充用例**：任务包 gen_csv.py 只生成均匀分布，需自行补 `GAUSS_NORM_5_2` 正态分布用例
  （§3.5 明确要求），并在自测报告说明补齐方式与数量；
- 测试 README 给出 build → 精度 → msprof 性能 的完整复现步骤。

## 兼容性分析

- 新增接口，不改动仓内既有算子；`include/cann_ops_blas.h` 声明与 950PR 共用；
- 实现放在实数同族 `blas/sbmv/arch22/`，命名与 ssbmv 四件套对齐（chbmv_*），便于维护与对照；
- 与仓内 `aclblasCsymv` 的非法 uplo 返回码差异，按任务书 §7-5 执行 INVALID_VALUE 并在此声明。
