# aclblasHgemm 算子设计文档（Ascend 950PR，arch35）

> 依据《aclblasHgemm_Atlas950PR 任务书》编写，按官方设计模板组织，四个必需一级章节齐全。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemm/arch35/`，测试 `test/gemm/hgemm/arch35/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 10 月社区任务「aclblasHgemm算子开发(950)」。在 ops-blas 仓为 Ascend 950PR（arch35）实现半精度通用矩阵乘 `aclblasHgemm`（对标 cuBLAS `cublasHgemm`）：

```text
C = alpha · op(A) · op(B) + beta · C
A/B/C：FLOAT16（aclblasHalf），列主序；alpha/beta：FLOAT16 标量（Host 指针）
op ∈ {N, T, C}（FP16 实数下 T ≡ C）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasHgemm_Atlas950PR_task_doc.md` | 接口签名、参数/校验语义、精度/性能门槛（§2.3/§2.4/§2.5/§3.2/§3.3） |
| `test_cases/`（hgemm_test.csv、gpu_baseline.csv、gen_csv.py、verify_accuracy.py、README） | 用例运行方式、golden 链路、msprof 采样口径 |
| ops-blas 仓 master（456b348） | 工程框架与同族 GEMM 模板 |

## 1.2 背景介绍

### ops-blas 现状分析（master 456b348 逐项实测）

| 对象 | 现状（实测） | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | **无 `aclblasHgemm` 声明**（GEMM 族已有 Sgemm/Cgemm 系声明） | 头文件需新增声明（§2.3 签名，供其他产品线共用） |
| `blas/gemm/arch35/gemm_*` | **FP32 版全套已合入**：`gemm_host.cpp + gemm_host_common.h + gemm_kernel.cpp/h + gemm_tiling_data.h + gemm_tiling_helpers.h + gemm_tiling_struct.h` | **直接工程模板**：`GemmTilingData`（m/n/k/lda/ldb/ldc、usedCoreNum、mBlocks/nBlocks、baseM/baseN/baseK、tileKChunk、isTransA/B、hasBeta、alphaReal/alphaImag）即本设计的 tiling 骨架 |
| `blas/gemm/arch35/cgemm_*` | 复数版更细分（compute_kernel / epilogue_kernel / onchip_kernel / deinterleave_common 分阶段流水） | 大规模档位的**分段流水参考**（装载→片上乘累加→epilogue 三段式） |
| `blas/gemm/arch22/gemm_*` | A2/A3 FP32 版（标准 Ascend C） | 不涉及（本任务仅 arch35） |
| `test/gemm/` | 族级测试组织已存在 | 测试落点 `test/gemm/hgemm/arch35/`，复用族框架 |

实测模板关键常量（`gemm_tiling_data.h`）：`GEMM_BASE_M=32、GEMM_BASE_N=16、GEMM_BASE_K=8、GEMM_C0_SIZE=8、GEMM_FRACTAL=16、GEMM_TILE_K_CHUNK=64`——分块以 baseM×baseN 为片上粒度、K 方向按 64 步进。

### 功能缺口与本任务差异点

**arch35 FP16 版整体缺失**。相对 FP32 模板的差异即全部工作量：

1. dtype：A/B/C 为 FP16，**累加与 epilogue 保持在 FP32**（精度依据任务书 §3.2：golden 以 FP32 计算后 RNE 转 FP16 比对）；
2. alpha/beta：`const aclblasHalf*` Host 指针，读值后升 FP32 入 tiling（沿用 `alphaReal/alphaImag` 字段，虚部恒 0）；
3. T ≡ C 等价（实数）：装载分支合并，枚举校验仍三分（非法枚举 → `ACLBLAS_STATUS_INVALID_ENUM`）；
4. 校验次序特殊约定（任务书 §2.5）：**alpha/beta 判空先于 m/n 短路**——`m=0/n=0` 的 no-op 场景 alpha/beta 为 nullptr 仍须返回 `INVALID_VALUE`；A/B/C 判空**后于**短路（no-op 时允许为空）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemm/arch35/` 新增 `hgemm_*` 文件族（与 gemm/cgemm 并列），实现：FLOAT16 × transa/transb 9 组枚举 × m/n/k ∈ [0, 16384]（用例规模上限）× lda/ldb/ldc padding（trans 相关约束）× alpha/beta 特殊值（0、±1、|v|→65504、含 beta=0 时 C 为 NaN 的用例）；精度满足生态标准 FLOAT16 档（rtol = atol = 2⁻¹⁰，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-1 或 32 ULP）+ Inf/NaN 等效规则 + k 过大溢出举证流程；性能**单 case ≥ 0.4× GPU 且全 case 平均 ≥ 0.5×**（gpu_baseline.csv 锚点：512³→9.054µs、1024³→15.077µs、2048³(T,N)→28.199µs、4096³(N,T)→182.406µs）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `C = alpha·op(A)·op(B) + beta·C`，FP16 输入输出、FP32 累加 | §2.1 | §3.2.2 Cube MAC（fp16×fp16→fp32）+ fp32 epilogue 后 RNE 写出 |
| R02 | transa/transb ∈ {N, T, C}，T≡C 合并装载分支 | §2.1 | 装载转置分支两套（N / T=C），枚举校验三分 |
| R03 | 非法枚举返回 `INVALID_ENUM`（区别于参数非法的 `INVALID_VALUE`） | §2.4 | Host 校验表（§3.2.1 第 2 行） |
| R04 | 校验次序：alpha/beta 判空先于 m/n 短路；A/B/C 判空后于短路 | §2.5 | Host 校验表顺序固化，负向用例锁定 |
| R05 | m=0/n=0 no-op（不访问 A/B/C）；k=0 仅 `C = beta·C`（beta=0 置零） | §2.1/§2.5 | Host 快速分支 + 独立 scale kernel |
| R06 | alpha/beta FP16 Host 指针不可空；值域含 0/±1/近 65504 | §2.4 | 读值升 FP32 入 tiling；epilogue 按 0 分支（alpha=0 跳过乘累加路径） |
| R07 | ld 约束（lda：N→max(1,m)、T/C→max(1,k)；ldb 对偶；ldc ≥ max(1,m)） | §2.4 | Host 校验 + 全程 j·ld 寻址 |
| R08 | 精度：FLOAT16 档混合容差 + golden 链路（FP32 cblas_sgemm → RNE→FP16） | §3.2 | §4.1；kernel 内不做 fp16 累加 |
| R09 | Inf/NaN 等效规则（同 NaN/同号 Inf/NaN↔Inf 判过；有限↔非有限判不过；非有限值单列统计） | §3.2 | §4.1 测试侧实现（对拍器） |
| R10 | k 过大溢出不满足阈值时走举证流程（CPU 对拍、误差-k 曲线、累加档说明） | §3.2 | §3.2.4 风险 1 + 自测报告模板项 |
| R11 | 性能双判据（单 case 0.4× + 平均 0.5×）；msprof 5 warmup + 10 样本 | §3.3 | §3.2.5 反算表 |
| R12 | 异步（handle 携带 stream，无 Host 回读） | §2.5 | kernel 直调 |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
C(i,j) = alpha · Σ_{p=0}^{k-1} op(A)(i,p) · op(B)(p,j) + beta · C(i,j)
         i ∈ [0, m)，j ∈ [0, n)
op(A) = A（N）；Aᵀ（T）；Aᴴ（C，实数下与 T 同）
维度：A 为 m×k（N）或 k×m（T/C）；B 为 k×n（N）或 n×k（T/C）；C 为 m×n
数值通路：fp16 输入 → fp32 乘累加（Cube）→ fp32 epilogue → RNE → fp16 输出
```

### 3.1.2 支持数据类型

A/B/C：FLOAT16（`aclblasHalf`）；alpha/beta：FLOAT16 标量（Host 指针，读值升 FP32 使用）；累加与 epilogue 中间量 FP32。

### 3.1.3 支持形状

m/n/k ≥ 0；ld 约束见 R07；用例规模上限 16384（自构造用例受单矩阵内存约束，4096³ FP16 单矩阵 32MB、16384³ 单矩阵 512MB，在 950 HBM 内）。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内 gemm 模板）

```text
include/cann_ops_blas.h                      # 新增 aclblasHgemm 声明（紧邻 GEMM 族）
blas/gemm/arch35/
├── hgemm_host.cpp                           # 校验 + tiling 组装 + kernel 启动（复用 gemm_host_common.h 工具）
├── hgemm_kernel.h                           # hgemm_kernel_do 声明
├── hgemm_kernel.cpp                         # Cube kernel（分块乘累加 + epilogue）
├── hgemm_tiling_data.h                      # HgemmTilingData（对齐 GemmTilingData + fp16 档位字段）
├── hgemm_tiling_helpers.h                   # 尾块/对齐辅助（复用 gemm_tiling_helpers.h 同构逻辑）
└── hgemm_tiling_struct.h                    # 分块结构体
test/gemm/hgemm/arch35/
├── hgemm_npu_wrapper.h                      # stream 绑定与同步封装
├── hgemm_test.cpp                           # CSV 驱动 GTest（含 handle_null 直测）
└── hgemm_test.csv                           # 任务包用例
```

### 3.2.1 Host 侧设计

**校验层**（顺序即返回优先级，逐条对齐任务书 §2.4/§2.5）：

| 顺序 | 条件 | 返回 |
| --- | --- | --- |
| 1 | handle == nullptr | `HANDLE_IS_NULLPTR` |
| 2 | transa ∉ {N,T,C} 或 transb ∉ {N,T,C} | `INVALID_ENUM` |
| 3 | alpha == nullptr 或 beta == nullptr（**先于 m/n 短路**） | `INVALID_VALUE` |
| 4 | m < 0 / n < 0 / k < 0 | `INVALID_VALUE` |
| 5 | m == 0 或 n == 0 | `SUCCESS`（no-op，不访问 A/B/C） |
| 6 | k == 0 | 发射 scale kernel：C = beta·C（beta=0 置零），`SUCCESS` |
| 7 | alpha ≠ 0 且（A == nullptr 或 B == nullptr）；或 C == nullptr（beta ≠ 0 时） | `INVALID_VALUE` |
| 8 | lda/ldb/ldc 不满足 trans 相关约束（R07） | `INVALID_VALUE` |
| 9 | 通过 → 组装 tiling、发射主 kernel | — |

> 顺序 3 与 7 的先后差即任务书 §2.5 "alpha/beta 判空先于短路、A/B/C 判空后于短路"的硬约定，负向用例（m=0 且 alpha=null 等）逐条锁定。

**TilingData**（对齐 `GemmTilingData` 实测字段，fp16 适配）：

```cpp
struct HgemmTilingData {
    int32_t m, n, k;            // 维度
    int32_t lda, ldb, ldc;      // 前导维（列主序）
    int32_t usedCoreNum;        // 实际使用核数
    int32_t mBlocks, nBlocks;   // C 平面分块数（baseM=32 × baseN=16 粒度）
    int32_t baseM, baseN, baseK;// 片上粒度（沿用 32/16/8，fp16 下可上调为 64/32/16 分档）
    int32_t tileKChunk;         // K 步进（默认 64）
    int32_t isTransA, isTransB; // N=0 / T=C=1（实数合并）
    int32_t hasBeta;            // beta≠0 标志（epilogue 跳过读 C）
    float alphaReal, alphaImag; // alpha/beta 读值升 FP32（虚部恒 0）
    // + fp16 专属：dtype 档位、尾块掩码参数（tailM/tailN/tailK）
};
```

### 3.2.2 Kernel 侧设计（Cube，本任务技术主体）

沿用 gemm 模板的「C 平面分块分核 + K 向流式推进」结构：

```text
grid：mBlocks × nBlocks 映射到 usedCoreNum（行列块连续交叉分配，满核优先；
     块数 < 核数时按核摊块；块数 > 核数时每核多块循环）
每块（baseM × baseN 的 C 子块）流程：
  ① K 向循环（tileKChunk = 64 步进）：
     装载 A/B 子片（N：列段直搬；T/C：转置搬运，实数无共轭分支）
     FP16 装载后保持 fp16 进入 Cube MAC（fp16×fp16→fp32 累加寄存器/片上缓冲）
     双缓冲：本 K 段计算与下一段搬运重叠
  ② K 结束得到 acc（fp32，baseM×baseN）
  ③ epilogue：C_tile = alpha·acc + beta·C_tile
     —— alpha/beta 为 fp32 化标量；beta=0 跳过 C 读；
     结果 RNE 舍入 FP16，按 ldc 写回
  ④ 尾块：tailM/tailN 非 0 时子块掩码路径（不越界写）
```

**k=0 scale kernel**：独立小核，逐元素 `C = beta·C`（beta=0 → 写 0），不引用 A/B；m×n 全覆盖、ldc 寻址。

**分档策略**（Host 按 m/n/k 选择）：

| 档 | 条件 | 参数 |
| --- | --- | --- |
| 小 | m·n ≤ 16×16 | 单核直算，baseM/baseN 收缩至问题规模 |
| 中 | ≤ 1024² | 常规分块（32/16/8），K=64 步进 |
| 大 | > 1024² | baseM/baseN 上调（64/32/16），K=128 步进，参考 cgemm 的 compute/epilogue 分段流水组织 |

### 3.2.3 调用与执行流程图

```text
Host                                                Device (Cube × usedCoreNum)
────                                                ──────────────────────────
aclblasHgemm(...)
  ├─ 校验表 §3.2.1（1-4 非法即返；5-6 快速分支）
  ├─ 7-8 指针/ld 校验
  ├─ 读 alpha/beta（FP16→FP32）入 TilingData
  ├─ 分档：mBlocks/nBlocks/baseM/baseN/tileKChunk
  └─ hgemm_kernel_do(t, stream) ──────────────────→  hgemm_kernel
     （k=0 走 scale_kernel）                          ├─ 按核认领 (mBlock, nBlock)
                                                      ├─ K 循环：装载(N/T)→双缓冲→
                                                      │   Cube MAC(fp16→fp32 acc)
                                                      ├─ epilogue：alpha·acc+beta·C→RNE→fp16
                                                      └─ 尾块掩码写回（ldc）
```

### 3.2.4 关键技术点与风险

| # | 点 | 处置 |
| --- | --- | --- |
| 1 | FP16 累加溢出（k 大时 acc 超出 fp16 表达） | **累加全程 fp32**（Cube fp32 累加器），仅最终 RNE 回 fp16；若个别用例仍超阈值（alpha≈65504×k 项），按任务书 §3.2.4 举证流程出分析（误差-k 曲线 + CPU 对拍），不放宽阈值 |
| 2 | Inf/NaN 等效比对 | kernel 不做特殊分支（传播自然发生）；等效规则在对拍器实现（§4.1），非有限值单列统计 |
| 3 | alpha=0 路径 | Host 判 alpha==0 时跳过主 kernel，直接走 beta 缩放核（与 k=0 同路径），避免 A/B 无效输入参与计算 |
| 4 | beta=0 时 C 为 NaN 用例 | epilogue 不读 C（hasBeta=0 分支），NaN 旧值不传播——位级要求由测试锁定 |
| 5 | T≡C 合并 | isTransA/B 两态；枚举三分仅存在于校验层 |
| 6 | 尾块非对齐（m/n/k 非 base 倍数） | 掩码写 + K 尾段缩短装载；性能用例均为对齐尺寸，尾块仅精度路径 |
| 7 | 16384 上限内存 | 单矩阵 512MB（FP16）×3 + C 512MB，950 HBM 可容；tiling 无硬上限 |

### 3.2.5 性能设计反算（R11）

各锚点 FLOPS 与达标线（×2.5；均值判据 0.5× 即均值倍率 ≥0.5）：

| case | 规模 | FLOPS | GPU µs | 达标线 µs | 需 Cube 吞吐 |
| --- | --- | --- | --- | --- | --- |
| base-000 | 512³ | 2.68e8 | 9.054 | 22.6 | 12 TFLOPS |
| base-001 | 1024³ | 2.15e9 | 15.077 | 37.7 | 57 TFLOPS |
| base-002 | 2048³ (T,N) | 1.72e10 | 28.199 | 70.5 | 244 TFLOPS |
| base-003 | 4096³ (N,T) | 1.37e11 | 182.406 | 456.0 | 301 TFLOPS |

950 Cube FP16 峰值为数百 TFLOPS 量级，大尺寸段需求（~300 TFLOPS）接近峰值的合理利用区间——**大尺寸档列为首要实测验证项**（依据实机 Cube 利用率收敛 baseM/baseN 与 K 步进；均值判据可由中小尺寸段的高倍率拉平）。小尺寸段（512³）launch/分核开销主导，单发射多块摊薄。全部以 msprof（5 warmup + 10 样本）实测为准。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

## 3.4 算子约束限制

- 接口签名与任务书 §2.3 一致（供其他产品线共用，禁私有平行 API）；
- 列主序 + ld 前导维语义（不支持任意 stride）；无广播；C 原地覆写；无确定性计算承诺；
- alpha/beta 仅 Host 指针（任务书口径，异于 cuBLAS 的 device 指针可选）；
- 校验次序（alpha/beta 先于短路）为实现契约，负向用例锁定。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度（数值） | FLOAT16 档：`|actual−golden| ≤ 2⁻¹⁰ + 2⁻¹⁰·|golden|`，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-1 或 32 ULP；golden = FP32 `cblas_sgemm` → RNE → FP16 | 任务书 §3.2；opbase mixed_tolerance_standard |
| 精度（非有限值） | 同 NaN / 同号 Inf / NaN↔Inf 判过；有限↔非有限判不过；非有限值不计入两项统计、单独记录并通过 | 任务书 §3.2.5 |
| 精度（溢出举证） | k 过大不满足阈值时：CPU 对拍数据 + 最大误差-k 曲线 + 累加档位说明，不得放宽阈值 | 任务书 §3.2.4 |
| 性能 | 单 case ≥ 0.4×（≤ GPU×2.5）**且**全 case 平均倍率 ≥ 0.5；msprof op 口径（自带 5 warmup，连续 10 样本取均值，SUCCESS 才计样本） | 任务书 §3.3 |

## 4.2 测试方案

- **用例**：任务包 CSV（transa×transb 9 组全组合 × m/n/k 扫描至 16384（含 2ⁿ±1、非对齐、k=0）× alpha/beta 特殊值（0/±1/近 65504；beta=0 含 C 为 NaN）× ld 紧凑/padding × Inf/NaN 填充 × 负向全套（枚举非法→INVALID_ENUM、维度负/ld 非法→INVALID_VALUE、alpha/beta 空（含 m=0 场景）→INVALID_VALUE、A/B/C 空的 no-op 场景→SUCCESS））；
- **golden**：Netlib `cblas_sgemm` FP32 单标杆 → RNE → FP16（任务书链路）；对拍器实现 Inf/NaN 等效规则与非有限值单列统计；
- **执行**：`verify_accuracy.py --repo <ops-blas> --soc ascend950 --csv hgemm_test.csv`（编译 + GTest 逐条 PASS/FAIL）；
- **性能**：`msprof op --application=.../hgemm_test --gtest_filter=*TC_PF*` 采集 OpBasicInfo Task Duration，与 gpu_baseline.csv 逐条关联（单 case ≤ gpu_ms×2.5；全 case 倍率均值 ≥ 0.5）；
- **复现**：测试 readme 固化编译命令、msprof 命令与采样口径，验收人可直接复跑。

## 4.3 兼容性分析

- 公共头文件仅新增 `aclblasHgemm` 声明，不改既有任何声明；
- `blas/gemm/arch35/` 内新增 `hgemm_*` 文件族（与 gemm/cgemm 前缀隔离），复用族级 host_common/tiling_helpers 头（只读包含，不修改）；
- 构建按 arch 目录 glob 自动纳入；arch22/arch35 既有 GEMM 零改动；
- README 产品支持表新增「Ascend 950PR：支持」。

# 5. 参考链接

1. 任务书与测试包：`aclblasHgemm_Atlas950PR_task_doc.md`、`test_cases/README.md`；
2. ops-blas 仓：https://gitcode.com/cann/ops-blas （`blas/gemm/arch35/gemm_*` 直接模板、`cgemm_*` 分段流水参考）；
3. cuBLAS cublasHgemm：https://docs.nvidia.com/cuda/cublas/index.html#cublas-gemm ；
4. Netlib BLAS gemm：https://www.netlib.org/blas/dgemm.f ；
5. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md 。
