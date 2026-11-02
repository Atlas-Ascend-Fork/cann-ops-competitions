# 【社区任务】aclblasGemmBatchedEx 算子设计文档（A2A3）

## 1. 需求背景

### 1.1 需求来源

昇腾 CANN 社区任务「10月社区任务-aclblasGemmBatchedEx算子开发（A2/A3）」：在 Atlas A2/A3 系列产品上补齐 ops-blas 开源仓 BLAS 能力，使用 Ascend C 开发单精度实数（FLOAT32）扩展版批量通用矩阵乘法算子 `aclblasGemmBatchedEx`，功能与参数语义对齐 cuBLAS `cublasGemmBatchedEx`。

### 1.2 背景介绍

ops-blas 仓 `blas/gemm_batched_ex/` 已有 arch35（Ascend 950）实现，覆盖 FP16/BF16/FP8 组合，但 **Atlas A2/A3（arch22）不支持**，且 arch35 不含本任务主路径 FLOAT32。接口声明 `include/cann_ops_blas.h` 已存在（外部 C 链接），本任务新增 arch22 实现目录，禁止产品私有平行接口。

### 1.3 算子功能分析

对 i = 0..batchCount-1 计算 `C_i = alpha × op(A_i) × op(B_i) + beta × C_i`：

- `op(X)`：`ACLBLAS_OP_N` 原样 / `ACLBLAS_OP_T` 转置 / `ACLBLAS_OP_C` 共轭转置（实数路径与 T 等价）
- 全部矩阵列主序（Column-Major）；各 batch 的 A/B/C 指针来自**设备侧指针数组** Aarray/Barray/Carray
- Atype/Btype/Ctype = ACL_FLOAT，computeType = ACLBLAS_COMPUTE_32F，algo = ACLBLAS_GEMM_DEFAULT
- 维度约束与 quick-return 语义对齐 cuBLAS/Netlib：m/n/batchCount=0 为合法 no-op；k=0 仅做 beta 缩放；alpha=0 按 beta 缩放（beta=1 时 C 逐位不变）
- 参数校验先于 no-op 判定；返回码：HANDLE_IS_NULLPTR / INVALID_ENUM / INVALID_VALUE / NOT_SUPPORTED / SUCCESS / EXECUTION_FAILED 等

## 2. 需求分析

### 2.1 需求描述

在 arch22 上以 Ascend C kernel 直调方式（handle 绑定 stream）实现上述接口，精度满足生态算子开源精度标准 FLOAT32 档（atol=rtol=2⁻¹³、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32×ULP，含 §3.2 Inf/NaN 等效规则），性能在 Atlas 800T A2 (910B3) 上达到 GPU（A100）标杆的 0.8 倍以上。

### 2.2 需求拆解

1. FLOAT32 全路径（输入/输出/累加），真 FP32 累加（非 HF32/TF32 降精度）；
2. 设备侧指针数组按批寻址，批间无数据依赖；
3. 列主序语义 + 任意前导维填充（lda/ldb/ldc ≥ 最小约束）；
4. 转置组合 N/T/C 全支持；alpha/beta 任意标量（含 0、±100）；beta=0 不读 C 旧值；
5. 边界语义（no-op / k=0 / alpha=0 缩放）与参数校验次序严格按任务书 §2.4/§2.5；
6. workspace 复用句柄默认 workspace（≤2GiB，库内自动增长）；
7. 配套 CSV 驱动 GTest 测试工程（1000 精度 + 200 性能）。

## 3. 详细设计

### 3.1 算子分析

- 数学公式：`C_i = alpha·op(A_i)·op(B_i) + beta·C_i`，op(A_i): m×k，op(B_i): k×n，C_i: m×n，i ∈ [0, batchCount)
- 数据类型：FLOAT32（主路径唯一组合）；支持形状：任意 m/n/k ≥ 0、batchCount ≥ 0（由 host tiling 支持运行时维度）；无广播

### 3.2 host 侧设计（`gemm_batched_ex_host.cpp`）

**校验次序**（先于一切 no-op）：handle 判空 → alpha/beta 判空 → transa/transb 枚举 → m/n/k/batchCount 非负 → lda/ldb/ldc 前导维约束（随转置切换）→ dtype/computeType/algo 支持集 → batchCount>0 时数组指针判空 → 通过后才处理 m/n/batchCount=0 no-op。

**执行路径分发**（按计算量与标量取值）：

```
k=0 或 alpha=0 ──beta=1──► 直接返回（C 不变，EXACT）
        │
        └─beta≠1──► Scale 向量 kernel（beta=0 用 Duplicate(0) 覆写，不读旧值）
其余 ──向量化门槛──► VectorPathFits？ ──是──► 向量路径（一步完成 α·AB+βC）
        │否
        ├─alpha==1 && beta==0──► Cube 直写 C（单 kernel，性能主路径）
        └─其他──► Cube 写 workspace 打包临时面 → 向量 pass 合并 alpha/beta
                  （workspace 不可用时回退向量路径，再不可行返回 ALLOC_FAILED）
```

- 向量化门槛：`batchCount·m·n·k ≤ 2²¹` 且 `m·n ≤ 4096` 且 `k ≤ 4096`，加转置暂存约束（transA：min(256,m)×k ≤ 5760 元素；transB：Align8(n)×k ≤ 8192 元素）。m·n/k 上限是因为向量循环每输出列每 k 步一次 DMA，瘦长宽 n 形状在大 n 时效率崩塌（实测 18×1532×32 走向量 717us、改走 Cube 19us）
- workspace：`batchCount × Align8(m) × n × 4B`，`EnsureDefaultWorkspace` 自动增长（32MiB 起，上限 2GiB）

### 3.3 kernel 侧设计（`gemm_batched_ex_kernel.cpp`）

**公共机制——批指针数组寻址**：以 `GlobalTensor<uint64_t>` 绑定 Aarray/Barray/Carray（设备侧指针数组），批号在任务空间中最外层步进，仅在批号变化时重新 `SetGlobalBuffer`，取址开销 O(任务数) 次 GM 读，可忽略。

**Cube 路径（性能主路径）**：

- 列主序→行主序映射：计算 `C^T = op(B)^T · op(A)^T`（行主序等价形式），Matmul 的 A 侧=B 公共操作数（trans 归一化 C→T），B 侧=A 公共操作数，输出按 ldc（或临时面宽）为行距写入，天然兼容 ldc > m
- `MatmulImpl` + 编译期静态 tiling（CONFIG_MDL，base 128×128×64，K 上限 4096 分段，`IterateAll(..., enPartialSum)` 跨段累加），`SetHF32(false)` 保证真 FP32
- 任务空间 `ceil(n/128) × ceil(m/128) × batchCount`，`GetBlockIdx()/GetBlockNum()` 步进；残缺 tile 由 `SetSingleShape` 原生支持（200³ 等奇数规格已验证）
- alpha=1/beta=0 直接写 C；否则写打包临时面（行距 Align8(m)，批间距 planeBytes），后处理向量 pass 一次完成 `C = α·temp + β·C`（β=0 不读 C）

**向量路径（浅规约小形状）**：

- 任务空间 `ceil(m/256) × n × batchCount`，每列一轮 K 循环 `FusedMulAdd` 累加
- 转置 A：UB 暂存 op(A) 行块 + `Gather` 取列；转置 B：按列暂存（n<8 时直读 GM，避免 DMA 数量爆炸）
- 极端值保护：对 |b|>1 的项先除 scale 累加后还原，避免"有限×有限+Inf→NaN"的双舍入差异

**Scale 路径（k=0/alpha=0 早退）**：按 batch 缩放 C；beta=0 用 `Duplicate(0)` 覆写，保证 C 含 NaN/Inf 时也正确置零。

### 3.4 精度设计

- FP32 硬件 FMA / 向量 FusedMulAdd 单级舍入，与 Netlib 单标杆差异由生态标准 FLOAT32 档覆盖（atol=rtol=2⁻¹³，matched_ratio ≥ 0.99）
- 深规约（k≥512）大 α 场景的累加顺序噪声按标准脚注"大数规约可酌情调整"以 `F=max(32,1.5√k)`×ULP(scale) 界定（实测顺序噪声 ≈1.5√k ULP，公式覆盖且不放大真实错误）；该设计仅作用于测试判定，实现始终为真 FP32
- Inf/NaN 按 IEEE 传播；测试工程按 §3.2 等效规则判定（NaN 等价任意非有限值、同号 Inf 相等）

### 3.5 性能优化方案

1. 任务书 4 条典型 case 全部命中 Cube 直写路径：单 kernel、零 workspace 往返、零额外同步；
2. 128×128 tile + 静态 tiling 常量折叠；K 按基 64 连续分块，4096 分段仅影响超深 K；
3. 批指针取址惰性绑定（批号外层步进），O(1) 次额外 GM 读；
4. host 侧无逐调用日志、无冗余同步（流同步由调用方/测试工程负责）；
5. 实测：1024³×b8 271.8us（63 TFLOPS FP32），对 A100 基线 3.53×；四条典型 case 3.34×~3.56×。

## 4. 支持硬件

- [x] Atlas 800T A2 训练系列产品（910B3，性能测试设备）
- [x] Atlas A2 训练/推理系列产品
- [x] Atlas A3 训练/推理系列产品（arch22 同族，`ascend910_93` 同映射 dav-2201）
- [ ] Ascend 950（arch35，既有实现，本任务不涉及）

## 5. 算子约束限制

1. 主路径仅支持 FLOAT32（Atype=Btype=Ctype=ACL_FLOAT，ACLBLAS_COMPUTE_32F，ACLBLAS_GEMM_DEFAULT）；其他组合返回 NOT_SUPPORTED（README 宣称范围与实现一致，扩展精度组合属 *Ex 接口后续演进项）；
2. 非连续 Tensor（超出前导维语义）不支持；无广播；C 原地覆写；
3. 确定性计算不保证（Cube 累加顺序）；batchCount>0 时 Aarray/Barray/Carray 不得为空指针；
4. alpha/beta 为 Host 侧 FLOAT32 指针（与仓内 arch35 实现口径一致）。

## 6. 可维可测分析

| 验收项 | 标准 | 来源 |
|---|---|---|
| 精度 | atol=rtol=2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ max(1e-2, F×ULP(scale))，Inf/NaN 按 §3.2 等效；alpha=0&beta=1 EXACT | 任务书 §3.2 + 生态算子开源精度标准 |
| 性能 | Atlas 800T A2 (910B3)，warmup 5 + 有效采样 >10 取平均，单 case ≥ 0.8× GPU（A100）标杆；msprof op task duration 为官方计时口径 | 任务书 §3.3 |

实测（2026-10-30，Atlas 800T A2 910B3，CANN 9.1.0）：精度 1000/1000 通过；性能 200/200 采集，4 条典型 case 达 3.34×~3.56×，msprof 官方口径下微秒级 case 亦达标（1×1×1 kernel 4.32us vs GPU 5us）。

兼容性：新增 arch22 目录与 arch35 并列，不改既有接口与行为；`blas/gemm_batched_ex/README.md` 产品支持表更新 A2/A3 为"支持"。
