# aclblasGemmBatchedEx 算子设计文档（arch22，Atlas A2/A3）

> 草稿 v1（2026-11-04）。目标：cann/cann-competitions PR，路径
> `04_tasks/01_community-task-2026/tasklist/10-28-aclblasGemmBatchedEx-A2A3/OpenGood_WeChat/docs/design.md`。
> PR 中仅含本文件。数字纪律：性能数字上机后回填【待填】，不预填。

# 需求背景（required）

## 需求来源

- 任务：10月社区任务-aclblasGemmBatchedEx算子开发（A2/A3），任务 ID `d945b8ea`
- 任务书：`aclblasGemmBatchedEx_A2_task_doc.md`（接口契约、边界语义、精度/性能验收口径均以任务书为准）
- 对标接口：cuBLAS `cublasGemmBatchedEx`；精度 golden：Netlib BLAS 3.10.0 `cblas_sgemm`
- 目标仓：https://gitcode.com/cann/ops-blas ，实现目录 `blas/gemm_batched_ex/arch22/`，测试目录 `test/gemm_batched_ex/`

## 背景介绍

`aclblasGemmBatchedEx` 是句柄式扩展批量 GEMM 接口：对 `batchCount` 个列主序矩阵组，各 batch 使用独立
A_i/B_i/C_i 指针，共享 m/n/k、leading dimension、alpha/beta。算子在 Atlas A2/A3（arch22，CANN 9.1.0）
上以 Ascend C kernel 直调方式补齐 ops-blas 的该能力；接口声明复用 `include/cann_ops_blas.h` 公共头，
不定义产品私有平行接口（与 950PR 等其他产品线共用）。

对第 i 个 batch：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
```

`op` 由 transa/transb 决定（N / T / C，实数路径 C ≡ T）。本实现为独立撰写；ops-blas 仓内已有的其他
GEMM 实现仅作为生态一致性参照，未复制其代码（与现有实现关系见兼容性分析）。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 上以 Ascend C 实现 `aclblasGemmBatchedEx` 主路径 FLOAT32
（A/B/C = ACL_FLOAT，computeType = ACLBLAS_COMPUTE_32F，algo = ACLBLAS_GEMM_DEFAULT），
支持 N/T/C 全转置组合、任意合法 m/n/k/batchCount/leading dimension、cuBLAS 对齐的
quick-return 与标量语义，精度满足 FLOAT32 混合容差标准，性能达到 GPU 标杆 0.8× 以上。

## 需求拆解

1. 参数校验：按任务书 §2.4 参数表顺序校验，先于一切 no-op 判定（§2.5）；
2. 主路径计算：batch×tile 并行的 Cube 矩阵乘 + Vector epilogue（alpha/beta 组合）；
3. 边界与特例：m/n/batchCount=0 no-op；k=0 与 alpha=0 仅 beta 缩放（beta=1 位精确不变，beta=0 清零）；
4. 负向用例：handle 空指针、非法枚举、负维度、非法前导维、标量/数组空指针、不支持 dtype/compute/algo；
5. 精度：rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP；Inf/NaN 等效判定；
   alpha=0 的 beta 缩放路径 EXACT（位精确）；
6. 性能：Atlas 800T A2(910B3)，msprof op Task Duration，5 warmup + 10 有效样本均值，单 case ≥0.8×GPU。

# 详细设计（required）

## 算子分析

### 数学公式

C_i = alpha · op(A_i) · op(B_i) + beta · C_i（列主序，i = 0..batchCount-1）

### 支持数据类型

A/B/C：ACL_FLOAT；computeType：ACLBLAS_COMPUTE_32F；algo：ACLBLAS_GEMM_DEFAULT。
其余组合（含 FP16/BF16/FP8/INT8 声明位）当前返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，
支持集合以 README 产品支持表为准，不把未声明组合默认为合法输入。

### 支持形状

m/n/k/batchCount 运行时入参（≥0），lda/ldb/ldc 满足 cuBLAS 前导维约束
（transa=N 时 lda≥max(1,m)，否则 lda≥max(1,k)；transb 对称；ldc≥max(1,m)）。
host tiling 支持动态 shape，无编译期 shape 假设；不支持超出前导维语义的非连续访问（与任务书一致）。

## 算子实现

### 实现方案

#### Host 侧设计

入口 `blas/gemm_batched_ex/arch22/gemm_batched_ex_host.cpp`：

1. **参数校验（顺序即任务书 §2.4 参数表序）**：handle → transa/transb（INVALID_ENUM）→
   m/n/k（负值 INVALID_VALUE）→ alpha 指针 → Aarray（batchCount>0 时非空）→ Atype
   （NOT_SUPPORTED）→ lda → Barray → Btype → ldb → beta 指针 → Carray → Ctype → ldc →
   batchCount → computeType → algo。校验全部通过后才做 no-op 判定（§2.5：no-op 场景下
   数组/标量指针仍须满足非空约束）。
2. **标量解析**：alpha/beta 可位于 Host 或 Device，host 侧经指针位置探测 + 必要的异步读回解析为
   float 值，随 tiling 下发，kernel 不再感知标量指针。
3. **no-op 与特例判定**：m=0 / n=0 / batchCount=0 直接返回 SUCCESS；k=0 或 alpha=0 进入
   scaleOnly 路径（beta=1 位精确不变直接返回；beta=0 全零填充；其余逐元素 beta 缩放），
   不引用 A/B。
4. **三路分派**（设计不变量：alpha≠1/beta≠0 永不回退到标量参考实现）：
   - **scaleOnly 路径**：`gemm_batched_ex22_scale` 单一 vector kernel 完成 beta 缩放/清零；
   - **小规模向量路径**：k≤256 且 m·n≤16K elements 时 `gemm_batched_ex22_vector`
     单 kernel 直出（含 alpha/beta 原生组合），避免小 shape 的 Cube 吞吐浪费；
   - **大规模路径**：Cube 主路径 + epilogue（下述）。
5. **大规模路径两种模式**：
   - **direct 模式**（alpha=1 且 beta=0）：Cube 直接写 C，无中间缓冲；
   - **组合模式**（其余 alpha/beta）：Cube 把原始乘积 P 写入 workspace 临时缓冲（每 batch
     padded m×n，8 对齐），vector epilogue 做 C = alpha·P + beta·C；workspace 按"一波 batch"
     分波复用（容量 = min(batchCount, workspace/perBatch)），host 循环下发 (Cube, Vector) 成对
     launch。workspace 实测量级与 1.2×GPU 判据数据在自测报告中披露【待填】。
6. **精度路由**（大规模路径内）：k>256 且 alpha/beta≠默认值时走补偿标量路径
   `gemm_batched_ex22_scalar`；其中 k>1024 或 |alpha|>2 的长归约/放大场景启用软件 FMA
   累加器（见 kernel 侧）。性能用例（官方 200 条实测 alpha=1/beta=0）全部留在快速路径。
7. **tiling 结构**：host→kernel 单块 80B 控制结构（shape/ld/转置/batch 波信息/alpha/beta/
   模式标志/exactFma/splitK 字段），`static_assert` 锁定字段偏移，Host/Kernel 两侧布局一致。

#### Kernel 侧设计

`blas/gemm_batched_ex/arch22/gemm_batched_ex_kernel.cpp`，四个直调入口（单文件单 TU）：

1. **Cube 主入口（`__cube__`）**：经指针数组（GM 侧 GetValue 解引）绑定 batch 矩阵，按
   (batch, tile) 二维任务划分；`MatmulImpl` 执行 tile 级矩阵乘（fp32 累加），Mmad 硬件单舍入
   累加与 golden 的 FMA 链舍入特性一致；N/T/C 用同一套地址映射（C 复用 T 分支）。padding
   前导维只影响地址步进，不改逻辑 shape；写回严格按实际字节数处理尾块，不越界写。
2. **Vector 入口（`__vector__`）**：两个职责——小规模整算（直出 alpha/beta 组合）与组合模式的
   epilogue（DataCopyPad 搬入 P 与旧 C，向量 Muls+Add 组合后写回）。
3. **Scale 入口**：beta 缩放 / 清零，覆盖 k=0、alpha=0 与 beta=1 位精确 no-op 之外的全体特例。
4. **补偿标量入口**：精度关键（长 k、alpha/beta 缩放）列的受控路径——
   Neumaier 补偿求和抑制 Muls+Add 双舍位的误差累积（~k·eps·Σ|ab|）；对超出 32 ULP 逐元素
   硬界的场景（k>1024 或 |alpha|>2）升级为软件 FMA 累加器：Veltkamp 两积分解 + 2Sum，
   逐 MAC 单舍入，位级追踪 ARM VFMA 参考链（AIV 向量层无融合乘加、aicore 禁 double 的
   硬件约束下的唯一精确路径；该结论来自 dav-2201 上机实测）。
5. **硬件约束适配（上机实证，dav-2201/910B3）**：4B 跨步 DataCopyPad gather 不受支持
   （MTE2 32B 块粒度），含 T 转置的元素级装载改用标量 GetValue 寻址；UB 内转置优化列为
   后续增强项，不改变语义。

### 精度设计

golden 用 Netlib `cblas_sgemm` 逐 batch 生成，全矩阵逐元素比对；FLOAT32 按 2^-13/2^-13、
0.99、1e-2 或 32 ULP 判定；Inf/NaN 按任务书等效规则单独统计（NaN↔同号 Inf 等效）；
alpha=0 缩放路径 EXACT。误差控制策略按"参考实现累加语义"选路：golden 是 fp32 FMA 顺序链
（OpenBLAS aarch64 实测复现），故主路径用硬件单舍入的 Cube Mmad 对齐，向量/标量路径按
k 长度与 alpha 幅值分级补偿（见上），把逐元素误差压在 32 ULP 硬界内。

### 性能设计

测试设备 Atlas 800T A2(910B3)，msprof op 模式 Task Duration（5 warmup + 10 有效样本均值），
逐 case 对照 GPU baseline（≥0.8×）。性能关键决策：direct 模式消除中间缓冲往返；小规模路径
单 kernel 直出；官方性能用例（alpha=1/beta=0）不落入补偿路径。分 case 实测数据【待填】
（上机 msprof 采集后回填自测报告）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（含 800T A2 / 910B3） | √ |
| Atlas A3 系列（arch22 同源实现） | √ |

## 算子约束限制

- 当前版本仅接入 FLOAT32 × COMPUTE_32F × GEMM_DEFAULT 组合，其余返回 NOT_SUPPORTED；
- 不支持超出 lda/ldb/ldc 语义的非连续 Tensor（任务书不要求）；
- 不涉及 broadcast、视图、随机数；C 原地覆写；异步执行经 handle 绑定 stream，读回前需同步。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32：rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP；Inf/NaN 等效判定；alpha=0 路径 EXACT | 任务书 §3.2 / 生态算子开源精度标准 |
| 性能标准 | 910B3 单 case 平均耗时 ≤1.25×GPU（≥0.8×标杆），msprof op 口径 5+10 采样 | 任务书 §3.3 |

测试工程 `test/gemm_batched_ex/`：接入官方 200 条 CSV（精度 1200 条变体 + 性能 200 条），
golden 由 cblas_sgemm 生成；补充负向矩阵（batchCount>0 且 Aarray/Barray/Carray=null 独立用例、
标量空指针、非法枚举/维度/前导维、不支持 dtype/compute/algo）与 Inf/NaN 特殊值用例；
测试 README 给出可复现步骤（build → 跑精度 → msprof 采性能），验收人可直接复跑。

## 兼容性分析

- 接口声明、状态码、handle/stream 约定全部复用 ops-blas 公共层（`cann_ops_blas.h`、
  `cann_ops_blas_common.h`、公共 handle 内部设施），不影响其他算子路径；
- README 产品支持表标注 Atlas A2/A3：支持；接口与其他产品线（950PR）共用；
- **与现有实现的关系**：本设计为独立实现（方案留痕见个人仓 docs）。ops-blas master 中已有的
  arch22 同名实现与本实现互不派生；两实现差异点（alpha/beta≠默认值的任意 shape 快速路径、
  校验顺序对任务书 §2.4 的逐参数对齐、标量/数组负向矩阵补全、workspace 披露）将在代码 PR
  与自测报告中逐条说明。
