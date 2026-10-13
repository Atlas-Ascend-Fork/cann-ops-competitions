# aclblasStrsmBatched 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasStrsmBatched_A2A3 任务书》编写，按官方设计模板组织，四个必需一级章节齐全。
> 交付目标仓：https://gitcode.com/cann/ops-blas （`blas/trsmbatched/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 9 月社区任务「aclblasStrsmBatched 算子开发(A2A3)」。在 `ops-blas` 仓为 Atlas A2/A3（`arch22`，性能测试设备 Atlas 800T A2 (910B3)）实现单精度实数（FLOAT32）批量三角求解算子 `aclblasStrsmBatched`：对 `batchCount` 个相互独立、同尺寸的三角系统批量求解，结果**离席写入独立输出 C**（B 只读，非原地覆写）。

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasStrsmBatched_A2A3_task_doc.md` | 功能、接口、硬件、精度/性能门槛与交付要求 |
| `test_cases/`（strsmbatched_test.csv、gpu_baseline.csv、gen_csv.py、verify_accuracy.py、README） | 用例运行方式、golden、性能统计口径 |
| ops-blas 仓 master | 工程框架、头文件、同族实现与测试组织 |

## 1.2 背景介绍

### 算子语义：batch 维天然并行的多次三角求解

对每个 batch i 独立求解（16 组枚举组合 side×uplo×trans×diag）：

```text
side=LEFT : op(A[i]) · X[i] = alpha · B[i]，A[i] 为 m×m（lda ≥ max(1,m)）
side=RIGHT: X[i] · op(A[i]) = alpha · B[i]，A[i] 为 n×n（lda ≥ max(1,n)）
op(A) ∈ {A（OP_N），Aᵀ（OP_T）}；实数无共轭，OP_C 判非法（较 cuBLAS 收紧）
B[i]/C[i] 为 m×n（ldb/ldc ≥ max(1,m)）；C[i] = X[i] 离席写出，与 A/B 不重叠
```

与单右端 trsv 族不同，trsm 的右端是矩阵：**列方向无依赖**（X 的各列独立求解），加上 batch 维独立，天然并行度为 `batchCount × n`（LEFT）；串行依赖链仅在 A 的三角方向上。这决定了本算子是带宽受限型（反算见 §3.2.5），多核划分是设计核心。

### ops-blas 现状分析

基于 ops-blas master 实测核对：

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h:567` | `aclblasStrsmBatched` 声明**已存在，但为 cuBLAS 原地语义**：`..., const float* const A[], int lda, float* const B[], int ldb, int batchCount`（B 覆写，**无 Carray/ldc**） | 任务书签名为其**扩展形式**（ldb 后新增 `Carray/ldc`，B→C 离席）→ **公共 API 变更**，需同步适配 arch35（见 §4.3） |
| `blas/trsmbatched/arch35/` | 已有全套实现（SIMT）：`trsmbatched_scale_kernel / trsmbatched_zero_kernel`（alpha 缩放/置零辅助核）、panel 求解（`SIMT_MAX_THREADS=64`、面板 ≤ `MAX_BLOCK_NB=128`、tileKChunk 按 `FP32_C0_LOCAL` 对齐） | 语义与测试对齐参考；**其 host 对 `batchCount < 1` 返回 INVALID_VALUE，与任务书 batchCount=0 合法 no-op 口径不一致**，统一签名时一并修正 |
| `blas/trsmbatched/arch22/` | **空缺** | 本任务落点：新增 arch22 全套实现（标准 Ascend C） |
| `test/trsmbatched/strsmbatched/` | 家族级 `strsmbatched_param.h / strsmbatched_golden.h / CMakeLists.txt` 已存在，仅有 `arch35/` 测试 | 测试落点 `test/trsmbatched/strsmbatched/arch22/` |

### 功能缺口

1. **arch22 实现缺失**：无 host/kernel/tiling/测试；
2. **公共签名升级**：原地 B 覆写 → B 只读 + C 离席输出（参数 1–11 与 cuBLAS 一一对应，ldb 后插入 Carray/ldc）；
3. **行为口径**：batchCount=0 由 INVALID_VALUE 收敛为合法 no-op（任务书口径）；alpha=0 走 C 置零快速路径（A/B 不引用）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/trsmbatched/arch22/` 以标准 Ascend C 实现 `aclblasStrsmBatched`（kernel 直调、handle 绑定 stream），并按任务书签名更新公共声明。能力达到：FLOAT32 × 16 组枚举全组合 × 非方阵（fat/thin）× 前导维 padding（lda/ldb/ldc）× alpha 特殊值（含 0 置零路径）× batchCount ∈ [0, 1024]；m/n 覆盖精度段 [1, 2048] 与性能段 [16, 4096]；精度满足混合容差（FLOAT32：rtol = atol = 2⁻¹³）；5 条锚点性能 case 达标（910B3，Avg time ≤ GPU 实测 / 0.8）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | FLOAT32 批量三角求解，B→C 离席输出 | 任务书 §2.1 | §3.2.2 kernel 设计；C 与 A/B 不重叠由调用方保证 |
| R02 | 16 组枚举全组合（side×uplo×trans×diag），OP_C 非法 | 任务书 §2.1/§2.4 | Host 统一分发 + kernel 四方向模板化（§3.2.2） |
| R03 | 非方阵 fat/thin（A 阶数取较小维） | 任务书 §3.5 | 面板划分按 side 取 A 阶数，B/C 按 m×n 寻址 |
| R04 | 前导维 padding（lda/ldb/ldc） | 任务书 §2.4 | 列基址 `j·ld*`，访问界按 [0, n)/[0, m) 裁剪 |
| R05 | alpha Host 指针；alpha=0 时 C 置零、A/B 不引用 | 任务书 §2.4/§2.5 | Host 解引用读值入 tiling；zero 快速路径独立 kernel |
| R06 | 指针数组元素级 nullptr 校验 | 任务书 §2.4 | 指针数组 D2H 小拷贝同步校验（§3.2.1，≤8KB） |
| R07 | m=0 / n=0 / batchCount=0 合法 no-op | 任务书 §2.4 | Host 校验层直接返回 SUCCESS，不发射 kernel |
| R08 | 精度：混合容差 FLOAT32 | 任务书 §3.2 | §4.1，Netlib strsm 逐 batch golden |
| R09 | 性能：5 锚点 ≤ GPU/0.8 | 任务书 §3.3 | §3.2.5 反算（最高 78GB/s，多核可行） |
| R10 | 公共 API 变更与 arch35 协同 | 任务书 §2.3/§4 | §4.3 兼容性分析 |
| R11 | 异步执行 | 任务书 §2.5 | kernel 直调绑定 stream，无隐式同步（R06 除外，见说明） |
| R12 | 可测可复现 | 任务书 §3.5 | CSV 驱动 GTest + 固定种子 20260823 |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

对 batch i，令 `P = side=LEFT ? m : n`（A 的阶数），`X = C[i]`。以 LEFT + LOWER + OP_N 为例（列面板前代，k = 0..n−1 对 C 的列，j = 0..P−1 对 A 的三角方向）：

```text
C(:,k) 首列缩放:  C(:,k) = alpha · B(:,k)
对 j = 0..P-1（沿 A 三角方向）:
    c_j = C(j,k) / A(j,j)              （UNIT 时跳过除法且不读对角）
    C(j+1:P, k) -= A(j+1:P, j) · c_j   （列 j 连续段 AXPY）
```

其余方向等价变形：LEFT+UPPER 为回代；LEFT+OP_T 为点积型（`C(i,k) = (αB(i,k) − Σ_{j<i} A(j,i)·C(j,k)) / A(i,i)`，求和项为列 i 连续段）；RIGHT 型为对 C 行方向的对应递推（A 作用在右侧，等价于对 Cᵀ 做 LEFT 型，kernel 内以行列索引互换实现）。UNIT 对角跳过除法且不读取 A 对角位置。

### 3.1.2 支持数据类型

仅 FLOAT32。

### 3.1.3 支持形状

`m, n ≥ 0`；`batchCount ≥ 0`；A[i] 为 `lda × P`（仅 uplo 三角被引用）；B[i]/C[i] 为 `ldb/ldc × n`（列主序，m×n 逻辑矩阵）。UB 分块策略下 m/n 无硬上限（任务用例域 m,n ≤ 4096、batchCount ≤ 1024）。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内同族）

```text
include/cann_ops_blas.h                  # 按任务书签名更新 aclblasStrsmBatched 声明
blas/trsmbatched/arch22/
├── strsmbatched_host.cpp                # 校验 + tiling + kernel 启动
├── strsmbatched_kernel.h                # *_kernel_do 声明
├── strsmbatched_kernel.cpp              # Ascend C kernel 群（zero/主求解）
└── strsmbatched_tiling_data.h           # StrsmbatchedTilingData
test/trsmbatched/strsmbatched/arch22/
├── strsmbatched_npu_wrapper.h
├── strsmbatched_test.cpp                # CSV 驱动 GTest
└── strsmbatched_test.csv                # 任务包 1200 条用例
```

### 3.2.1 Host 侧设计

**签名（公共 API 升级，任务书 §2.3）**：

```cpp
aclblasStatus_t aclblasStrsmBatched(aclblasHandle_t handle, aclblasSideMode_t side,
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha, const float* const* Aarray, int lda,
    const float* const* Barray, int ldb, float* const* Carray, int ldc, int batchCount);
```

校验层（顺序即返回优先级）：

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | handle == nullptr | `HANDLE_IS_NULLPTR` |
| 2 | m < 0 / n < 0 / batchCount < 0 | `INVALID_VALUE` |
| 3 | side/uplo/trans/diag ∉ 合法枚举（trans 含 OP_C） | `INVALID_VALUE` |
| 4 | alpha == nullptr（Host 指针） | `INVALID_VALUE` |
| 5 | batchCount > 0 且 Aarray/Barray/Carray == nullptr | `INVALID_VALUE` |
| 6 | m/n/batchCount 任一为 0 | `SUCCESS`（no-op，不发射） |
| 7 | lda/ldb/ldc 不满足前导维约束 | `INVALID_VALUE` |
| 8 | 指针数组元素级校验（D2H 拷回 `batchCount×8B` ≤ 8KB 同步校验，任一元素为空） | `INVALID_VALUE` |

> R06 说明：指针数组位于 Device 内存，Host 无法直接解引用；采用发射前一次 ≤8KB 的阻塞式 `aclrtMemcpy` 拷回校验（微秒级）。该同步只发生在参数校验阶段，不触及输出数据，与"读回结果前须同步 stream"的异步语义（针对 C）不冲突；换取元素级空指针的确定性返回，是任务书 §2.4 的直接要求。

alpha 为 Host 指针：Host 解引用读值放入 tiling（不作为 Device 地址下发）。alpha == 0 走 `trsmbatched_zero_kernel` 置零路径（C 逐 batch 写零，A/B 不引用），沿用 arch35 先例。

TilingData 要点：`Aarray/Barray/Carray` 的 **Device 地址**（kernel 侧再经一次小搬运取各 batch 指针）、m/n/lda/ldb/ldc、batchCount、枚举、alpha 值、面板参数（panelWidth、每核 batch 组数）。

### 3.2.2 Kernel 侧设计（本任务技术主体）

**总体：AIV-only 多核，"batch 分组 × 列面板" 两级划分，单遍流式。**

#### 两级并行划分

```text
L1 batch 维：batchGroups = min(batchCount, 可用AIV核数)
     每核认领 ceil(batchCount / batchGroups) 个 batch（按序分组，批间完全独立）
L2 batch 内（大矩阵少 batch 时启用，如 bc=2、m=n=4096 锚点）：
     C 的列方向无依赖 → 列面板并行：每面板组一个核，
     面板间沿 A 三角方向仍有前/回代依赖 → 面板内顺序推进、面板宽度 W 内
     对角块三角求解 + 尾部更新（更新为 GEMM 型，行方向再切片并行）
调度：grid = batchGroups × panelWorkers；Host 按 m/n/batchCount 分档选择
     （bc ≥ 核数：仅 L1；bc 小且 P 大：L1×L2；P ≤ 128：单核直解）
```

#### 单核内面板递推（§3.1.1 公式的向量化）

```text
对每个列面板 [k, k+W)：
  ① 装载 B/C 列块（DataCopy 双缓冲，ldb/ldc 步进）
  ② alpha 预缩放（融合进首个面板装载后的向量 Mul）
  ③ 沿 A 三角方向 j 步进：对角元除法（标量 Div，UNIT 跳过）→
     尾段更新 AXPY（向量 Mul + Sub，A 列 j 连续段 tile 装载）
  ④ OP_T 方向：点积型（向量 Mul + ReduceSum，段内 UB 归约）
  ⑤ 面板完成 → C 列块写回（ldc 步进）
```

A 的装载按列 tile 流式（列主序下每列连续，lda 跨列）；仅 uplo 三角区被读取，另一三角与 padding 不访问（以列内起止偏移裁剪）。

### 3.2.3 调用与执行流程图

```text
Host                                              Device (AIV × blocks)
────                                              ────────────────────
aclblasStrsmBatched(...)
  ├─ 校验表 §3.2.1（含指针数组 D2H 元素校验）
  ├─ no-op / alpha=0 ─→ zero_kernel 或直接返回
  ├─ TilingData 组装（分档 L1/L2 划分）
  └─ strsmbatched_kernel_do(t, stream) ────────→  strsmbatched_kernel
                                                     ├─ 取 batch 指针表（小搬运）
                                                     ├─ 按核认领 batch 组/面板组
                                                     ├─ 面板递推（缩放/对角/AXPY|DOT）
                                                     └─ C 写回（离席）
```

### 3.2.4 关键技术点与风险

| # | 点 | 处置 |
| --- | --- | --- |
| 1 | 公共 API 变更（B 原地 → B/C 分离） | 同 PR 内适配 arch35 host 至新签名（内部等价迁移：B 读入 + C 写出），保持两架构单一公共 API；交叉回归见 §4.3 |
| 2 | batchCount=0 口径 | 新实现按任务书返回 SUCCESS；arch35 现网 `<1 → INVALID_VALUE` 随签名适配一并统一，并在测试 CSV 中以 L6 用例锁定 |
| 3 | RIGHT 方向的行列角色互换 | kernel 内以索引变换统一到 LEFT 递推，避免维护两套算法 |
| 4 | 元素级空指针校验的同步代价 | ≤8KB 阻塞拷贝，µs 级；仅在发射前执行一次 |
| 5 | 大 batch 小矩阵（bc=1024, 32×32） | 每核多 batch 串行流水，批间无同步；启动开销单次 |
| 6 | 精度（面板递推累加） | 混合容差 2⁻¹³ 拦截约 1ULP；含规约可放宽 2ULP（任务书 §3.2 说明） |

### 3.2.5 性能设计反算（R09）

各锚点有效数据量（A 仅计三角有效读 + B 读 + C 写）与达标线反算：

| case | bc | m×n | 有效数据量 | 达标线 µs | 需有效带宽 |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 256×256 | ~41 MB | 630.7 | 65 GB/s |
| 2 | 128 | 384×512 | ~239 MB | 3062 | 78 GB/s |
| 3 | 32 | 1024×1024 | ~323 MB | 5134 | 63 GB/s |
| 4 | 8 | 2048×2048 | ~323 MB | 8680 | 37 GB/s |
| 5 | 2 | 4096×4096 | ~323 MB | 14361 | 22.5 GB/s |

arch22 多 AIV 聚合带宽（数百 GB/s 量级）对最高需求 78GB/s 留有余量；计算强度低（每元素一次乘加），带宽受限成立。依赖链仅存在于 A 三角方向（面板数 = P/W 步），列/batch 维并行充分。达标结论：结构上无反例，以 msprof 实测为准（§4.2）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3/910B4） | √ |
| Atlas A3 系列（arch22 同目录） | √ |

## 3.4 算子约束限制

- 不做奇异性检测；NON_UNIT 时调用方保证对角非零（测试侧 boost = max(5, P) 填充）；
- C 与 A/B 不允许内存重叠（离席语义）；alpha 仅支持 Host 指针（较 cuBLAS 收紧）；
- trans 不支持 OP_C（实数收紧，判非法）；不支持超出 lda/ldb/ldc 语义的非连续访问；无广播、无 dynamic shape、无确定性计算承诺。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差 FLOAT32：`|actual−golden| ≤ 2⁻¹³ + 2⁻¹³·|golden|`，matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP（含规约可放宽 2ULP） | 任务书 §3.2；opbase mixed_tolerance_standard |
| 性能标准 | 5 锚点 Avg time ≤ GPU/0.8：630.7 / 3062 / 5134 / 8680 / 14361 µs；msprof op 口径，warmup 后有效采样 ≥ 10 次均值 | 任务书 §3.3；gpu_baseline.csv |

## 4.2 测试方案

- **用例**：任务包 1200 条（1000 精度 + 200 TC_PF；16 组枚举全组合 × 尺寸/非方阵/batch 扫描 × alpha 特殊值 × 前导维 padding × 填充/Inf/NaN × 边界负向含指针数组元素级空指针/OP_C 非法/batchCount 负值），固定种子 20260823；
- **golden**：cblas（Netlib `strsm` 逐 batch）单标杆；测试工程对被引用对角加符号保持偏移 boost = max(5, P)，golden 用同一偏移后矩阵求解，两侧输入严格一致；alpha=0 用例 golden 为全零；
- **执行**：`verify_accuracy.py --repo <ops-blas> --soc ascend910b3 --csv strsmbatched_test.csv`；性能借 `msprof op`（`--gtest_filter` 收敛到单条 TC_PF）采集 OpBasicInfo Task Duration 与 gpu_baseline.csv 逐条关联（NPU ≤ gpu_ms/0.8）；
- **覆盖核查**：L0 16 组全枚举、L1 尺寸、L2 alpha、L3 batch/非方阵、L4 前导维、L5 填充（含 UNIT 下 A 全零）、L6 边界负向全覆盖，缺失场景按任务书 §3.5 补充。

## 4.3 兼容性分析

- **公共 API 变更**：`include/cann_ops_blas.h` 中 `aclblasStrsmBatched` 由原地语义升级为 B/C 分离签名（参数前缀与 cuBLAS 一致，ldb 后插入 Carray/ldc）。**arch35 现有实现须在同一变更中适配**：host 入口改签新名（内部 B→C 迁移），kernel 不变或等价包装；两架构共用单一公共 API，禁止并行双接口。适配属于机械迁移，arch35 测试（`test/trsmbatched/strsmbatched/arch35/`）随 CSV 同源回归；
- **行为口径统一**：batchCount=0 → SUCCESS（arch35 现为 INVALID_VALUE，随适配修正并以 L6 用例锁定）；其余校验次序按 §3.2.1；
- **上游影响**：ops-blas 仓内无其他调用方引用该接口（新增族），对外无行为兼容包袱；README 产品支持表标注 A2/A3（含 800I A2/A3）支持。

# 5. 参考链接

1. 任务书与测试包：`aclblasStrsmBatched_A2A3_task_doc.md`、`test_cases/README.md`；
2. ops-blas 仓：https://gitcode.com/cann/ops-blas （`blas/trsmbatched/arch35/strsmbatched_*` 先例、`include/cann_ops_blas.h:567`）；
3. Netlib BLAS strsm：https://www.netlib.org/blas/strsm.f ；
4. cuBLAS cublasStrsmBatched：https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-strsmbatched ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md 。
