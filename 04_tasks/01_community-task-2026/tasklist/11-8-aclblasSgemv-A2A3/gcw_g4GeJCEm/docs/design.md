# aclblasSgemv 算子开发（Atlas A2/A3）设计文档

> 作者：gcw_g4GeJCEm（Z3824）
> 对标接口：cuBLAS `cublasSgemv`（CUBLAS_Library.pdf §2.6.2）
> 语义锚点：仓内 950PR 实现 `blas/gemv/arch35/sgemv_host.cpp`（ValidateSgemvParams，L31-107）
> 交付目录：`blas/gemv/arch22/`、`test/gemv/sgemv/arch22/`

# 需求背景（required）

## 需求来源

11 月社区任务-aclblasSgemv 算子开发（A2/A3）。对标 cuBLAS `cublasSgemv`，在 Atlas A2/A3（910B3，CANN 9.1.0）上以 Ascend C kernel 直调方式补齐 BLAS-2 矩阵-向量能力缺口。

## 背景介绍

ops-blas 仓 gemv 族已有 950PR 的 arch35 实现（`blas/gemv/arch35/sgemv_host.cpp` / `sgemv_kernel.cpp`）与 A2/A3 的复数 arch22 实现（`blas/gemv/arch22/cgemv_*`），单精度实数 arch22 实现缺位。本任务实现 `aclblasSgemv`（声明已在 `include/cann_ops_blas.h` L461-463，与 950PR 共用，不改签名），补齐 A2/A3 单精度实数 GEMV。

# 需求分析（required）

## 需求描述

`y = alpha·op(A)·x + beta·y`（列主序），op(A) 由 trans ∈ {N, T, C} 决定（C 对实数等价于 T）；alpha/beta 为 Host 标量指针；x/incx、y/incy 支持任意非零 int32 步长（含负步长，尾端起算）；m=0 或 n=0 在参数校验通过后快速返回。

## 需求拆解

1. trans 三态 + 非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`（与 950PR 实数实现一致；不采用复数 README 的 INVALID_ENUM 口径）。
2. 校验层级完全对齐 arch35 锚点：handle 判空（`HANDLE_IS_NULLPTR`）→ trans/m/n/lda/incx/incy/alpha/beta → A/x/y 空指针（按逻辑长度条件判）→ 快速返回（m=0 或 n=0）。
3. 任务书新增项：跨度溢出守卫 `(逻辑长度−1)×|inc| > INT32_MAX` 返回 `INVALID_VALUE`（host 侧 int64 计算，issue #866/#860 口径）；arch35 锚点未含此检查，本实现补齐。
4. 短路语义：alpha=0 时 y = beta·y（不读 A/x）；beta=0 时不读 y 旧值、直接覆写；alpha/beta 值不校验（Inf/NaN 按公式自然传播）。
5. 性能：全部性能 case ≥ A100 标杆 0.8 倍（gpu_baseline.csv，910B3 实测）。
6. 精度：mixed_tolerance FLOAT32 档（rtol=atol=2^-13，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32×ULP）；golden 由仓内 `test/gemv/sgemv/sgemv_golden.h`（cblas_sgemv 包装）生成。

# 详细设计（required）

## 接口定义

沿用既有声明，不改签名：

```cpp
aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, const float* x, int incx,
    const float* beta, float* y, int incy);
```

## 总体结构

kernel 直调，host 侧校验 + tiling 计算 + 按 `<<<numBlocks, nullptr, stream>>>` 值传 tiling 下发（与 arch22 scasum/cgemv 同族）。三个 AIV kernel：

| kernel | 路径 | 说明 |
|---|---|---|
| `sgemv_no_trans_aiv_kernel` | trans=N 主计算 | 行独占 / 列切分两子路径 |
| `sgemv_reduce_aiv_kernel` | trans=N 列切分第二相 | workspace 部分和按行归约 |
| `sgemv_do_trans_aiv_kernel` | trans=T/C | 列独占分块点积 |

## trans=N：行独占路径（tall / 常规形状）

y 的 m 行切成 64 行一块的 vertBlock，整块独占分配到核（base=V/C，前 V%C 核 +1），核内按 512 列一段流式处理：

- A 瓦片：一次二维 `DataCopyPad` 载入 64 行 × 512 列（blockLen=行段×4B，srcStride=(lda−行段)×4B），列主序无需转置；
- x 段：incx=1 连续拷贝；incx>1 二维跨步拷贝（blockLen=4B，srcStride=(incx−1)×4B）；incx<0 按 BLAS 尾端语义标量 GetValue 组装（段长 ≤512，开销可忽略）；
- 每列：`Muls(64) + Add(64)` 累加进 y 累加器（xv=0 跳过）；
- 收尾：`yTile = alpha·acc + beta·y_in`（beta=0 跳过 y 读），按 incy 连续/标量散写回。

特性：y 行块核间独占，**零 workspace、零跨核同步**；acc 仅 64 float，UB 占用 ~137KB。vertBlocks < 核数时多余核闲置（行块粒度天然均衡，m=4096 时 64 块 > 40 核）。

## trans=N：列切分 + workspace 归约路径（wide 形状，如 m=4、n=1M）

选择条件：`m×4 ≤ 160KB（部分和 UB 容量） && 40×m×4 ≤ 8MB（workspace 预算） && n×8 ≥ m`（宽矩阵）。不满足则回落行独占路径（m 最大 1M 时行独占天然适配）。

- 相位 1（`sgemv_no_trans_aiv_kernel`）：列均分到核（base+remainder）。m ≤ 64 时走行抽取特化：A 第 i 行是跨步（lda 行距）或连续（lda=m）向量，按 ≤16384 float 分块 `DataCopyPad` 载入，`Mul` x 段后 `ReduceSum` 得行点积，直接得该列域的部分和标量；m > 64 时按 2048 行块流式：每列 `Muls+Add` 进块累加器，块完成写入 workspace 槽位 `[coreIdx, m]`。alpha=0 时核内短路，仅写零工作区。
- 相位 2（`sgemv_reduce_aiv_kernel`）：2048 行块按核均分，每块对 40 个部分和向量做 40 次 `DataCopyPad+Add` 折叠，`y = alpha·Σ + beta·y`。两相位同 stream 串行，无需同步原语。

workspace：`numBlocks × m × 4B ≤ 8MB`（host 分配，用后释放）。

## trans=T/C：列独占分块点积

列均分独占（n 最大 1M，40 核各 ~25K 列）。x 段常驻 UB（m ≤ 16384，最大 64KB）；每列按 8192 float 分块（双缓冲 ping-pong，DMA 与向量计算重叠）：`DataCopyPad` 列块 → `Mul` x 块 → `ReduceSum`（Scasum 同款四参形式 + 8 对齐补零）→ 标量累加。y 值缓冲 512 列一批写回（incy=1 连续；否则标量散写）。m > 16384 的病态用例退化为逐块重载 x（仅精度集覆盖，性能集无此形状）。alpha=0 短路为纯 beta·y 原地缩放。

## 负步长与原地语义

负步长一律尾端起算：逻辑 k ↔ 物理 `(len−1−k)×|inc|`（与 950PR golden/cuBLAS 对齐）。y 原地覆写：读旧值（beta≠0 时）→ 计算 → 写同址。span 溢出已在 host 拒绝，kernel 内索引 int64 计算，无回绕风险。

## 确定性

任务书不要求跨次一致（行内点积归约顺序自由）；单次执行内部顺序固定（无跨核竞争路径：行独占/列独占/相位分离均保证同一输出元素只有一个写入者，workspace 相位经 stream 串行隔离）。

## 短路语义核对（§2.5 逐项）

| 约束项 | 实现结论 |
|---|---|
| 校验层级 | handle→trans→m→n→lda→incx(0/INT32_MIN)→incy(0/INT32_MIN)→alpha→beta→span溢出→A→x→y→quick return，完全对齐 arch35 锚点 + 任务书新增守卫 |
| 非连续支持 | 列主序 lda 任意空洞（二维 srcStride）；incx/incy 任意非零（连续/跨步/标量三档 IO） |
| 原地语义 | y 原地覆写；beta=0 不读旧值 |
| 确定性 | 不要求；单次内部无竞争 |
| 空 Tensor | m=0/n=0 校验后快速返回，不访问数据 |
| alpha=0 短路 | 两 mode 均实现（不读 A/x；beta·y 仍执行） |
| 异步执行 | 计算 stream 异步下发；host 内部同步仅在需要释放 workspace 的列切分路径（与 cgemv arch22 同族口径） |

# 测试设计

- 用例：随任务发布 `sgemv_test.csv`（1200 条：精度 1000 + 性能 200）+ 仓内 `sgemv_golden.h`（cblas_sgemv）。
- 工程：`test/gemv/sgemv/arch22/`（param/wrapper/test），`test/gemv/sgemv/CMakeLists.txt` 补 arch22 GTest 注册分支。
- 负向与 quick-return 用例直调接口（单元素/空指针占位），不按 span 预分配（任务书负向缓冲规则）。
- 正态分布补充：`gen_csv.py --dist mixed` 生成 `GAUSS_NORM_5_2` 补充用例纳入精度自验。

# 已知限制与规避

1. T 模式 m > 16384 时 x 不常驻 UB，逐块重载（正确性不受影响；性能集无该形状）。
2. 行独占路径 vertBlocks < 核数时部分核闲置（m 小于 2560 且 n 小的场景）；此时矩阵本身极小，绝对耗时在微秒级，不影响 ≥0.8× 标杆判据。
3. incx/incy 负步长走标量组装路径（段长 ≤512 或行块 ≤64，开销可忽略）；性能用例不含负步长。
