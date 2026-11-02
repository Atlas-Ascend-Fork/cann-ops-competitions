# 需求背景

## 需求来源

[10月社区任务-aclblasCgemm3m算子开发（950）](https://www.hiascend.com/activities/task-center) 任务书要求：在昇腾 Atlas 950PR 上使用 Ascend C 开发单精度复数（complex64）3m 算法通用矩阵乘算子 `aclblasCgemm3m`，对标 cuBLAS `cublasCgemm3m`（legacy 3m 算法），基于 ops-blas 开源仓工程框架实现，验收通过后合入昇腾算子开源仓，补齐 BLAS 仓复数矩阵乘能力。

## 背景介绍

复数 GEMM `C = alpha·op(A)·op(B) + beta·C` 若按实虚分量直接展开需要 **4 次**实数 GEMM（ReA·ReB、ReA·ImB、ImA·ReB、ImA·ImB）。**3m 算法**通过恒等式将其降为 **3 次**实数 GEMM：

```
P1 = (ReA + ImA) · (ReB + ImB)
P2 = ReA · ReB
P3 = ImA · ImB
Re(C) = P2 − P3
Im(C) = P1 − P2 − P3
```

复数乘法量从 4 乘 2 加降为 3 乘 5 加，在实数算力为主体的硬件上获得约 25% 理论节省，是 cuBLAS legacy 3m 接口的经典优化。

**仓内现状（基于 ops-blas 官方 master 实测核实）**：

| 项 | 现状 | 对本任务的意义 |
|---|---|---|
| 接口声明 | `include/cann_ops_blas.h` **尚无** `aclblasCgemm3m` 声明（同族实数版 `aclblasSgemm3m` 声明在 h:557） | 声明随实现新增至公共头，与现有产品线共用，禁止 950PR 私有平行接口 |
| GEMM 模板 | `blas/gemm/arch35/` 有完整 GEMM 实现（任务书 §2.2 明示"基于 same 家族 GEMM kernel 直调，复用 Gemm 模板"） | **开发主体 = 复用 Gemm 模板做实虚分离 + 3 次实 GEMM 组合调度**，kernel 侧不重写矩阵乘核心 |
| 3m 同族参考 | `blas/gemm3m/arch35/` 已有实数版 `sgemm3m_{host,kernel,tiling_data}` | 3m 家族的 tiling 与 host 组织方式直接参照 |
| 合入目录 | 任务书 §5：实现 `blas/gemm/arch35`、测试 `test/gemm/arch35` | 目录与文件按仓规范，无冗余 |

# 需求分析

## 算子功能

```
C = alpha * op(A) * op(B) + beta * C
op(X) = X（N）/ X^T（T）/ X^H（C，共轭转置——复数下 C 与 T 不同，需对虚部取反）
A: op=N 时 m×k，op=T/C 时 k×m；B: op=N 时 k×n，op=T/C 时 n×k；C: m×n（均列主序 COMPLEX64）
```

功能要点清单：

| 编号 | 需求 | 说明 |
|---|---|---|
| FR1 | 3m 分解 | 复数乘按 3 次实数乘线性组合实现（P1/P2/P3 恒等式），设计需明确 3m 路径 |
| FR2 | 实虚分离 | op(A)、op(B) 实部/虚部拆分（解交织），Re(C)/Im(C) 分别按 3m 公式组合后交织写回 |
| FR3 | 转置语义 | transa/transb ∈ {N, T, C} 全组合；T=纯转置，C=共轭转置（实虚分离后对虚部矩阵取负实现共轭） |
| FR4 | 标量处理 | alpha/beta 为复数标量指针（Host/Device），作用于整体结果（epilogue 复数乘） |
| FR5 | 维度与前导维校验 | m/n/k ≥ 0；lda ≥ max(1, N?m:k)、ldb ≥ max(1, N?k:n)、ldc ≥ max(1,m)；handle/alpha/beta nullptr 与非法枚举返回对应错误码（INVALID_VALUE / INVALID_ENUM / HANDLE_IS_NULLPTR） |
| FR6 | 边界行为（对齐 cuBLAS/Netlib quick return） | m=0 或 n=0 → SUCCESS 不写 C；k=0 → C = beta×C（beta=0 置零、beta=1 不变），不引用 A/B；beta=0 时 C 原值不被读取 |
| FR7 | **确定性计算** | 同输入两次执行逐位一致（任务书硬性要求） |
| FR8 | 异步语义 | handle 绑定 stream 直调 kernel；读回前须同步 |

## 公开接口与范围

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,   // ACLBLAS_OP_N / OP_T / OP_C
    aclblasOperation_t transb,   // ACLBLAS_OP_N / OP_T / OP_C
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

声明新增至 `include/cann_ops_blas.h`（与其他产品线共用）；参数序列与 cuBLAS cublasCgemm3m 一一对应。实现范围：`blas/gemm/arch35/` 新增 cgemm3m host/kernel/tiling 文件 + README 支持表（Atlas 950PR：支持）；不改既有算子行为。

# 详细设计

## 算子分析

### 数学公式

```
C = alpha · op(A) · op(B) + beta · C
3m 路径：P1=(ReA+ImA)(ReB+ImB)；P2=ReA·ReB；P3=ImA·ImB
Re(op(A)op(B)) = P2 − P3；Im(op(A)op(B)) = P1 − P2 − P3
```

其中 ReA/ImA/ReB/ImB 为 op 之后（含转置与共轭处理）的实数矩阵视图；三次实 GEMM 均为 FLOAT32。

### 支持数据类型

| 项 | 取值 |
|---|---|
| A/B/C | COMPLEX64（实/虚各 FLOAT32，交织存储） |
| alpha/beta | COMPLEX64 标量指针（Host/Device） |
| 计算 | FLOAT32 分量；精度按 FLOAT32 分量判定（实部/虚部分别比对） |

### 支持形状

m/n/k ≥ 0（含 0 quick return）；典型规模覆盖随任务用例（小质数、2^k、2^k±1、非对齐直至大规模，前导维紧凑与 padding）。

## 算子实现

### Host 侧设计

```
aclblasCgemm3m(handle, ...)
 ├─ 参数校验（handle/枚举/维度/前导维/空指针 → 对应错误码）
 ├─ quick return（m=0 ∨ n=0 → SUCCESS；k=0 → C=beta×C 缩放路径）
 ├─ 标量归一化（alpha/beta Device→Host 同步拷回）
 ├─ tiling（复用 Gemm 模板 tiling；三次实 GEMM 共享同一 m/n/k 形状）
 └─ 调度三次实 GEMM + epilogue（组合 + alpha 复乘 + beta·C 累加 + 交织写回）
```

调度顺序固定为 P1 → P2 → P3（确定性要求，见资源与调度）。

### Kernel 侧设计

1. **实虚分离（解交织）**：GM 交织复数 → UB 双路实数视图（偶/奇 offset 拆 Re/Im），转置/共轭在该阶段经坐标映射与 Im 取负归一为标准列主序实矩阵——后续三次 GEMM 全部走实数路径；
2. **三次实 GEMM**：复用 `blas/gemm/arch35` Gemm 模板的 Cube 分块乘累加核心（m×k×n，K 维循环），仅数据源换为分离后的实矩阵；
3. **组合 epilogue**：P2−P3 / P1−P2−P3 线性组合 → alpha 复数乘（4 乘 2 加展开）→ beta·C 读取累加（beta=0 跳过读）→ 实虚交织单次写回 GM；
4. **无原子操作**：各 tile 结果独立写回，不做跨核原子累加——确定性 + 免同步开销。

### 资源与调度

- **workspace**：分离后 ReA/ImA/ReB/ImB（各 ≤ 原复矩阵实数体量）+ P1/P2/P3 中间实矩阵（m×n×3×4B）；按 tiling 分块复用缓冲控制峰值；数据量估算与对照随自测报告输出；
- **确定性保障**：固定的 kernel/tile 调度顺序（无动态抢占写回）、固定归约顺序（K 维顺序累加）、无原子浮点累加——同输入两次执行逐位一致；
- **多核**：m×n tile 级并行切分（复用 Gemm 模板）；三次 GEMM 串行依赖（P 组合需要全部完成），tile 内流水掩盖。

## 支持硬件

| 项 | 值 |
|---|---|
| 适配硬件 | 昇腾 Atlas 950PR（arch35） |
| 软件 | CANN 9.1.0；golden 由 Netlib BLAS 3.10.0 复数实现生成 |
| README | `blas/gemm/README.md`（或 gemm3m 对应章节）产品支持表标注 Atlas 950PR：支持 |

## 算子约束限制

| 约束 | 说明 |
|---|---|
| 维度/前导维 | 按 FR5 校验；列主序前导维连续存放，不支持非连续步长 |
| 转置枚举 | 仅 N/T/C；其他枚举 INVALID_VALUE |
| 广播/动态 shape | 不涉及（m/n/k 编译期已知静态 shape） |
| 原地/视图 | C 为 in/out（beta≠0 读原值累加写回）；不复用 A/B 内存 |
| 确定性 | 强制（逐位一致） |

# 可维可测分析

## 精度标准与验证

- golden：cblas（Netlib BLAS 复数实现）单标杆，输出全矩阵验证，实部/虚部分别按 FLOAT32 分量判定：rtol=atol=2^-13(1.2e-4)、required_matched_ratio=0.99、max_abs_error=1e-2 or 32 ULP（生态算子开源精度标准）；
- 用例维度（任务书 §3.5 全覆盖）：transa×transb 九宫格全组合（另测非法枚举）；m/n/k 覆盖 0、1、小质数、2 的幂±1、非对齐至大规模；alpha/beta/A/B/C 均匀[-5,5] 50% + 正态 50% 独立采样 + 特殊值 (0,0)/(±1,0)/纯虚数/大值；lda/ldb/ldc 紧凑 + padding；边界/负向（m=0/n=0 SUCCESS 且 C 不变、k=0 SUCCESS 且 C=beta×C、空指针/非法枚举/非法前导维/负维度对应错误码）；Inf/NaN 特殊值按精度标准规则执行；
- **确定性验证**：同输入连续执行 2 次（覆盖代表性规模），输出逐位比对一致（bit-exact）纳入自测报告。

## 性能标准与验证

- 测试设备 Atlas 950PR，ops-blas test 目录测试框架（CSV 用例 + C++ GTest）；性能 case 对照随任务提供的 gpu_baseline.csv（warmup 5 次 + 有效采样 >10 次取均值）；
- 自测报告（官方模板）：用例参数、精度对比结果及截图、性能数据及截图。

## 交付物对照（任务书 §4）

设计文档（本 PR）→ 自测用例与测试代码（精度/性能 case 清单 + 复现 readme）→ 自测报告（精度/性能截图）→ 待验收代码地址（个人仓链接/分支/算子目录 + 邀请 Ascend-CANN 为开发者 + README 规范）。

---

*实现参考：ops-blas 官方 master `blas/gemm/arch35/`（Gemm 模板复用）、`blas/gemm3m/arch35/`（3m 同族 tiling/host 组织）；语义参考 cuBLAS cublasCgemm3m 文档与 Netlib BLAS。*
