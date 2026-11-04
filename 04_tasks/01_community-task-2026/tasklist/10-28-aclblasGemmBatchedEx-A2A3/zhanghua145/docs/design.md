# aclblasGemmBatchedEx 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasGemmBatchedEx_A2 任务书》编写，按官方设计模板组织，四个必需一级章节齐全。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemm_batched_ex/arch22/`，测试 `test/gemm_batched_ex/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 10 月社区任务「aclblasGemmBatchedEx算子开发(A2/A3)」。在 ops-blas 仓为 Atlas A2/A3（arch22，性能测试设备 Atlas 800T A2 (910B3)）实现扩展版批量矩阵乘 `aclblasGemmBatchedEx`（对标 cuBLAS `cublasGemmBatchedEx`）：

```text
对 i = 0..batchCount-1 独立计算：C[i] = alpha · op(A[i]) · op(B[i]) + beta · C[i]
A/B/C：设备侧指针数组（列主序，uniform 批：共享 m/n/k/ld/alpha/beta/枚举）
主路径：FLOAT32（Atype/Btype/Ctype = ACL_FLOAT，computeType = FLOAT32 计算类型）
扩展档：*Ex 接口声明的 FP16/BF16 等混合精度组合按 README 宣称范围覆盖
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasGemmBatchedEx_A2_task_doc.md` | 接口签名、参数/校验语义（含三种错误码分层）、精度/性能门槛（§2.3–§2.5、§3.2–§3.3）、200 条用例 |
| `test_cases/`（gemm_batched_ex_test.csv、gpu_baseline.csv、gen_csv.py、verify_accuracy.py、README） | 用例、golden 链路、msprof 采样口径 |
| ops-blas 仓 master（456b348） | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析（master 456b348 逐项实测）

| 对象 | 现状（实测） | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | **`aclblasGemmBatchedEx` 声明已存在（L471）**，19 参与 cuBLAS 逐参对齐（含 Atype/Btype/Ctype、computeType、algo） | 签名零改动，禁平行 API |
| `blas/gemm_batched_ex/arch35/` | **950 版已合入**（gemm_batched_ex_host/kernel 全套） | 语义与测试对齐的直接参考；arch22 不照搬（编程模型/分档差异） |
| `blas/gemm_batched_ex/arch22/` | **缺失** | 本任务落点 |
| `blas/gemm/arch22/gemm_*` | A2/A3 FP32 单实例 GEMM（标准 Ascend C：gemm_host/kernel/kernel.h） | arch22 GEMM 计算底座模板（分块/转置装载/epilogue 同构复用） |
| 指针数组先例 | `blas/trsmbatched/arch35/strsmbatched_host.cpp:623`（`CopyPtrArraysD2H` 逐元素判空） | 指针表发射前小拷贝校验模式 |
| `test/gemm_batched_ex/` | 族级测试组织已存在 | 测试落点（含 csv），结构参考主仓 |

### 功能缺口与本任务差异点

**arch22 整体缺失**。相对单实例 gemm（arch22）的差异即全部工作量：

1. **指针数组**：A/B/C 三表（batchCount 个 64 位 Device 地址），kernel 需逐批取址；发射前指针表校验（整体判空为任务书硬要求，逐元素判空为增强项）；
2. **三态错误码分层**：trans 非法 → `INVALID_ENUM`；dtype/computeType/algo 不支持 → `NOT_SUPPORTED`；维度/ld/指针 → `INVALID_VALUE`（任务书 §2.4 逐参数约定）；
3. **alpha/beta Host/Device 双口径**（任务书 §2.4 "Device 或 Host 内存"）：Host 指针直读；Device 指针走 8B 同步小拷贝读值（§3.2.1）；
4. **快速分支族**：m/n/batchCount=0 no-op、k=0 与 alpha=0 的 beta 缩放（alpha=0 时结果位精确 EXACT 校验）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemm_batched_ex/arch22/` 实现 `aclblasGemmBatchedEx`：FLOAT32 主路径 × transa/transb 9 组 × m/n/k 扫描 × batchCount ∈ [0, 2ⁿ] × ld（trans 相关约束）× alpha/beta（0/±1/大值、Host|Device）× dtype/computeType/algo 合法档；精度 FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP）+ Inf/NaN 等效 + alpha=0 EXACT；性能单 case ≥ 0.8× GPU（×1.25 达标线；锚点 gpu_ms：1024³×8→0.958ms、2048³×8→7.214ms、4096³×4→28.672ms、1024³(T,N)×8→0.967ms）；自测报告含 workspace ≤ 1.2× GPU 佐证。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | 逐批 `C = alpha·op(A)·op(B) + beta·C`（uniform 批） | §2.1 | batch × tile 两级并行（§3.2.2） |
| R02 | transa/transb ∈ {N,T,C}（实数 C≡T）；非法 → `INVALID_ENUM` | §2.4 | Host 校验表第 2 行 |
| R03 | Atype/Btype/Ctype/computeType/algo 支持档外 → `NOT_SUPPORTED` | §2.4 | Host 校验表第 3 行（组合合法性表） |
| R04 | alpha/beta 不可空（先于 no-op 判定）；Host/Device 双口径 | §2.4/§2.5 | §3.2.1 读值策略（Host 直读 / 8B D2H） |
| R05 | Aarray/Barray/Carray 整体判空（batchCount>0 时）先于 no-op | §2.5 | 校验次序固化 |
| R06 | m=0/n=0/bc=0 no-op；k=0 仅 beta 缩放；alpha=0 按 beta 缩放（beta=1 不变 / beta=0 置零），不引用 A/B | §2.1/§2.5 | Host 快速分支 + scale kernel（逐批） |
| R07 | alpha=0 时 `C = beta·C` 位精确（EXACT 校验） | §3.2 | scale kernel 纯缩放路径（无累加舍入） |
| R08 | ld 约束（lda：N→max(1,m)、T/C→max(1,k)；ldb 对偶；ldc ≥ max(1,m)） | §2.4 | 校验 + 全程 j·ld 寻址 |
| R09 | 精度 FLOAT32 档 + Inf/NaN 等效规则（非有限值单列统计） | §3.2 | §4.1 对拍器 |
| R10 | 性能 ≥ 0.8× GPU（msprof 5 warmup + 10 样本） | §3.3 | §3.2.5 反算表 |
| R11 | workspace 报告项 ≤ 1.2× GPU | §4 交付件 3 | 设计零动态 workspace（§4.3） |
| R12 | 200 条任务包用例全覆盖 + 缺口自补 | §3.5 | §4.2 |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
对批 i：C(i)(r,s) = alpha · Σ_{p=0}^{k-1} op(A[i])(r,p) · op(B[i])(p,s) + beta · C(i)(r,s)
        r ∈ [0,m)，s ∈ [0,n)；op ∈ {N, T, C(≡T 实数)}
地址：A[i] = Aarray[i]（Device 64 位指针），元素 (r,p) 在 A[i] + p·lda + r（N）
快速分支（逐批统一执行）：
  k=0 或 alpha=0：C[i] = beta·C[i]（beta=0 → 全 0）；beta=1 → C 不变
```

### 3.1.2 支持数据类型

主路径 FLOAT32（输入/输出/FP32 累加）；alpha/beta 指针类型随 computeType/Ctype（主路径 FP32 标量）；扩展档（FP16/BF16 等）按 README 宣称矩阵，装载期归一 FP32、epilogue 按输出 dtype 舍入（本设计主路径先行，扩展档为同构 dtype 分支）。

### 3.1.3 支持形状

m/n/k ≥ 0；batchCount ≥ 0（用例域至 2ⁿ 批）；ld 约束见 R08；各矩阵字节数 = ld × 列数 × dtype 大小。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/gemm_batched_ex/arch22/
├── gemm_batched_ex_host.cpp        # 校验 + alpha/beta 读值 + tiling + kernel 启动
├── gemm_batched_ex_kernel.h        # *_kernel_do 声明
├── gemm_batched_ex_kernel.cpp      # Cube kernel 群（主循环 + scale）
├── gemm_batched_ex_tiling_data.h   # GemmBatchedExTilingData
└── gemm_batched_ex_tiling_helpers.h
test/gemm_batched_ex/
├── gemm_batched_ex_param.h         # CSV 列映射（含 dtype/computeType/algo 列）
├── gemm_batched_ex_golden.h        # cblas_sgemm 逐批 golden + 对拍器
└── arch22/{gemm_batched_ex_npu_wrapper.h, gemm_batched_ex_test.cpp, gemm_batched_ex_test.csv}
```

### 3.2.1 Host 侧设计

**alpha/beta 读值策略**（Host|Device 双口径）：

```text
发射前一次 8B×2 同步 D2H 探读：若指针位于 Device 地址段 → aclrtMemcpy 读值（阻塞，µs 级）；
Host 指针 → 直读。读值升 FP32 入 tiling（与指针表校验同批次执行，不增加额外同步次数）。
```

**校验层**（顺序即返回优先级）：

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | handle == nullptr | `HANDLE_IS_NULLPTR` |
| 2 | transa/transb ∉ {N,T,C} | `INVALID_ENUM` |
| 3 | (Atype,Btype,Ctype,computeType,algo) ∉ 支持组合表 | `NOT_SUPPORTED` |
| 4 | alpha == nullptr 或 beta == nullptr（先于 no-op） | `INVALID_VALUE` |
| 5 | m < 0 / n < 0 / k < 0 / batchCount < 0 | `INVALID_VALUE` |
| 6 | batchCount > 0 且（Aarray 或 Barray 或 Carray == nullptr） | `INVALID_VALUE` |
| 7 | m == 0 或 n == 0 或 batchCount == 0 | `SUCCESS`（no-op） |
| 8 | k == 0 或 alpha == 0 | 发射逐批 scale kernel（C=beta·C；beta=0 置零；beta=1 且 alpha==0 → 完全跳过发射），`SUCCESS` |
| 9 | ld 约束（R08）不满足 | `INVALID_VALUE` |
| 10 | 指针表逐元素判空（增强项：D2H ≤ 3×8×batchCount 字节小拷贝，trsmbatched 先例） | `INVALID_VALUE` |
| 11 | 通过 → tiling + 主 kernel | — |

**TilingData 要点**：三表 Device 地址、m/n/k/lda/ldb/ldc、batchCount、usedCoreNum、batch 分组分档（§3.2.2）、baseM/baseN/baseK/tileKChunk（沿仓内 arch22 gemm 模板常量）、isTransA/B、hasBeta、alpha/beta（FP32 化）、tail 参数。

### 3.2.2 Kernel 侧设计（Cube，本任务技术主体）

**两级并行：batch 分组 × C 平面分块**

```text
grid = batchGroups × tileGroups
  L1 batch 维：batchGroups = min(batchCount, 可用核数/最小 tile 占用)
       每核认领连续 batch 组（批间完全独立，无同步）
  L2 批内：mBlocks × nBlocks 分块（baseM×baseN 粒度）；块多核摊、块少核省
调度分档：
  大批小阵（1024³×8 档）：L1 主导，每批单块/少块，批间满核
  小批大阵（4096³×4 档）：L2 主导，批内 mBlocks×nBlocks 满核，批间循环
每 (批, 块) 任务（与单实例 gemm arch22 同构复用）：
  ① K 循环（tileKChunk 步进）：A/B 子片装载（N 直搬 / T=C 转置搬运），双缓冲
  ② Cube MAC：FP32×FP32→FP32 累加（fp16/bf16 扩展档装载期归一 FP32 后同路径）
  ③ epilogue：C_tile = alpha·acc + beta·C_tile（beta=0 不读 C），按 Ctype 舍入写回
  ④ 尾块掩码（tailM/tailN/tailK）
scale kernel（k=0 / alpha=0）：逐批逐元素 C = beta·C（beta=0 → 写 0），
  纯缩放无累加舍入（满足 EXACT）；beta=1 且 alpha=0 时 Host 直接返回不发射
```

### 3.2.3 调用与执行流程图

```text
Host                                                  Device (Cube × usedCoreNum)
────                                                  ──────────────────────────
aclblasGemmBatchedEx(...)
  ├─ 校验表 §3.2.1（1-6 非法即返；7 no-op；8 scale 分支）
  ├─ alpha/beta 读值（Host 直读 / Device 8B D2H）
  ├─ 指针表逐元素判空（增强，一次 ≤3×8×bc 小拷贝）
  ├─ tiling 组装（L1/L2 分档 + 尾块参数）
  └─ gemm_batched_ex_kernel_do(t, stream) ──────────→  主 kernel
                                                        ├─ 指针表小搬运（每核本地缓存）
                                                        ├─ 按 (批, 块) 认领任务
                                                        ├─ K 循环装载/MAC/双缓冲
                                                        └─ epilogue + 尾块写回
     （k=0/alpha=0 → scale_kernel 逐批缩放）
```

### 3.2.4 关键技术点与风险

| # | 点 | 处置 |
| --- | --- | --- |
| 1 | 三态错误码分层回归 | 负向用例逐参数锁定（枚举→INVALID_ENUM、dtype/compute/algo→NOT_SUPPORTED、其余→INVALID_VALUE），测试工程断言码值 |
| 2 | alpha/beta Device 指针读值 | 8B×2 阻塞 D2H（与指针表校验同批，不新增同步）；Host 指针直读——两路结果一致入 tiling |
| 3 | alpha=0 EXACT | scale kernel 纯缩放路径（乘法单次舍入对 beta·C 位精确性按 FP32 定义域成立；测试以 EXACT 断言锁定） |
| 4 | uniform 批地址越界 | 逐元素判空增强 + ld/维度校验；元素级地址有效性由调用方保证（任务书口径） |
| 5 | 负载不均（trans 组合内块数差异） | L1/L2 分档按 m·n 权重；4096³×4 档块数充足，满核可行 |
| 6 | workspace 报告项 | 主路径零动态 workspace（片上分块 + tiling 内联），报告以 0 或常数声明 + GPU 对照 |
| 7 | Inf/NaN 等效 | kernel 自然传播；等效规则在对拍器实现（§4.1） |

### 3.2.5 性能设计反算（R10）

| case | 规模 | FLOPS | GPU ms | 达标线 ms（×1.25） | 需 Cube 吞吐 |
| --- | --- | --- | --- | --- | --- |
| base-000 | 1024³×8 | 1.72e10 | 0.958 | 1.198 | 14.4 TFLOPS |
| base-001 | 2048³×8 | 1.37e11 | 7.214 | 9.018 | 15.2 TFLOPS |
| base-002 | 4096³×4 | 5.50e11 | 28.672 | 35.84 | 15.3 TFLOPS |
| base-003 | 1024³(T,N)×8 | 1.72e10 | 0.967 | 1.209 | 14.2 TFLOPS |

四锚点需求一致收敛于 **~15 TFLOPS FP32 Cube**——910B3 FP32 Cube 峰值为数十 TFLOPS 量级，利用率要求约为一半，分块/双缓冲达标结构上无反例；转置档（T,N）与（N,N）需求同量级（装载布局差异不改变吞吐量级）。以 msprof（自带 5 warmup + 10 有效样本）实测为准。

### 3.2.6 UB 与调度预算

片上占用（FP32 主档）：A 子片 baseM×tileKChunk×4B + B 子片 tileKChunk×baseN×4B + C 块 baseM×baseN×4B + 装载双缓冲 ×2——按仓内 arch22 gemm 模板常量（baseM=32/baseN=16/tileKChunk=64）计约 32×64×4×2(双缓冲) + 64×16×4×2 + 32×16×4 ≈ 20.7KB，远低于 AIV 可用 UB；大阵档（4096³）上调 baseM/baseN（64/32）后约 82KB，仍在预算内，K 步进同步上调至 128 减少循环开销。核间调度采用静态任务表（tiling 生成，无动态抢核），保证批间无争用、块间均衡由 mBlocks×nBlocks 交叉分配保证。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3/910B4） | √ |
| Atlas A3 系列（arch22 同目录） | √ |

（自验覆盖一款：精度建议 910B3 或 A3、性能 910B3；A2/A3 均需验收。）

## 3.4 算子约束限制

- 签名复用 `include/cann_ops_blas.h:471` 既有声明，禁平行 API；
- 列主序 + ld 语义（不支持任意 stride）；无广播；C 原地覆写；无确定性承诺；
- 校验先于 no-op（任务书 §2.5 硬约定）；空数组指针语义取 ops-blas 口径；
- 扩展 dtype 档以 README 宣称矩阵为准（主路径 FP32 为验收对象）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度（数值） | FLOAT32 档：`|actual−golden| ≤ 2⁻¹³ + 2⁻¹³·|golden|`，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP；golden = Netlib `cblas_sgemm` 逐批（扩展 dtype 按各自档：FP16 2⁻¹⁰、BF16 2⁻⁷） | 任务书 §3.2；opbase mixed_tolerance_standard |
| 精度（特例） | alpha=0 时 `C=beta·C` 位精确（EXACT）；Inf/NaN 等效规则（同 NaN/同号 Inf/NaN↔Inf 判过，有限↔非有限判不过，非有限值单列统计） | 任务书 §3.2 |
| 性能 | 单 case ≥ 0.8× GPU（≤ gpu_ms×1.25）；msprof op 口径（5 warmup + 10 样本均值，SUCCESS 计样本） | 任务书 §3.3 |
| 内存 | 自测报告含 workspace ≤ 1.2× GPU 佐证（本设计零动态 workspace） | 任务书 §4 交付件 3 |

## 4.2 测试方案

- **用例**：任务包 200 条（transa×transb 组合 × m/n/k 扫描（含 2ⁿ±1、非对齐、k=0）× batchCount 扫描（0/1/小批/2ⁿ）× alpha/beta 特殊值（0/±1/大值、nullptr 负向）× dtype 合法档与不支持档负向 × ld 紧凑/padding × Inf/NaN 填充 × 负向全套（枚举→INVALID_ENUM、dtype/compute/algo→NOT_SUPPORTED、维度负/ld 非法/数组空→INVALID_VALUE、handle 空））；
- **golden**：Netlib `cblas_sgemm` FP32 逐批单标杆；对拍器实现 Inf/NaN 等效与 alpha=0 EXACT 断言；
- **执行**：`verify_accuracy.py --repo <ops-blas> --soc ascend910b3 --csv gemm_batched_ex_test.csv`；
- **性能**：`msprof op --application=.../gemm_batched_ex_test --gtest_filter=*TC_PF*`，OpBasicInfo Task Duration 与 gpu_baseline.csv 逐条关联（NPU ≤ gpu_ms×1.25）；
- **覆盖核查**：任务包 200 条逐条对齐任务书 §3.5 生成规则，缺口场景自补并标注。

## 4.3 兼容性分析

- 公共头文件零改动（声明既有）；`blas/gemm_batched_ex/arch22/` 为纯新增目录，arch35 版零改动；
- 与同批（10.28）`blas/gemm_ex/arch22`、`blas/gemm_grouped_batched_ex/arch22` 任务目录隔离、无共享文件冲突；
- 构建按 arch 目录 glob 自动纳入；README 产品支持表标注 Atlas A2/A3 系列：支持。

# 5. 参考链接

1. 任务书与测试包：`aclblasGemmBatchedEx_A2_task_doc.md`、`test_cases/README.md`；
2. ops-blas 仓：https://gitcode.com/cann/ops-blas （`blas/gemm_batched_ex/arch35` 语义先例、`blas/gemm/arch22` arch22 计算底座、`trsmbatched` 指针表先例）；
3. cuBLAS cublasGemmBatchedEx：https://docs.nvidia.com/cuda/cublas/index.html#cublas-gemmBatchedEx ；
4. Netlib BLAS gemm：https://www.netlib.org/blas/dgemm.f ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md 。
