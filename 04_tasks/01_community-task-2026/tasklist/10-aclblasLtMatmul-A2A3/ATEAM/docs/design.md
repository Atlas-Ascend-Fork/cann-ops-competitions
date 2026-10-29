# 需求背景（required）

## 需求来源

通过 CANN 社区任务，在 Atlas A2/A3（含 Atlas 800T A2 / Atlas 800T A3）上用 Ascend C kernel 直调补齐 ops-blas 的 `aclblasLtMatmul` FLOAT32 主路径，完成设计、开发、测试，验收后合入 https://gitcode.com/cann/ops-blas 。

依据优先级：**任务书 > 公开头文件 `include/cann_ops_blasLt.h` > 现网实现 > 公开 cuBLAS / Netlib 文档**。

- 接口签名以头文件既有声明为准（任务书标注约 `cann_ops_blasLt.h:909-924`；当前基线声明在约 932–936 行，符号与参数表一致）。
- 句柄 / stream 为平台类型映射：`aclblasLtHandle_t` ↔ `cublasLtHandle_t`，`aclrtStream` ↔ `cudaStream_t`。禁止新增产品私有平行 API。
- 任务书合入路径写 `blas/lt_matmul/arch22` 与 `test/lt_matmul/`。现网实现与测试实际在 `blasLt/` 与 `test/blasLtMatmul/`。**本设计沿用现网 `blasLt/` 目录与 CMake 按 `SOC_ARCH_DIRS` 收集源文件的规则**，不另开一套 ABI；与任务书路径用词不一致处待评审确认。

## 背景介绍

### 算子功能

`aclblasLtMatmul` 计算：

```text
D = alpha * op(A) * op(B) + beta * C
```

- `op(X)` 由 matmul 描述符中的 `transA` / `transB` 决定：`ACLBLAS_OP_N` 为原矩阵，`ACLBLAS_OP_T` 为转置；任务书规定实数路径下 `ACLBLAS_OP_C` 与 `T` 等价。
- A、B、C、D 为矩阵；支持 `C == D` 原地与 `C != D` 异地。异地时 C/D 的 dtype、行列、memory order 须一致。
- 本任务仅单 batch：四个 layout 均 `batchCount = 1` 且 `stridedBatchOffset = 0`。
- 四个 layout 的 `order` 固定为 `ACLBLASLT_ORDER_ROW`。前导维按行主序解释为「一行元素数（行宽）」：`transA=N` 时 `lda ≥ max(1,k)`，否则 `lda ≥ max(1,m)`；`transB=N` 时 `ldb ≥ max(1,n)`，否则 `ldb ≥ max(1,k)`；`ldc ≥ max(1,n)`，`ldd ≥ max(1,n)`。`ld == 0` 按行宽归一为紧凑布局，无广播语义。
- 主路径 dtype：A/B/C/D 为 `ACL_FLOAT`，`computeType` / `scaleType` 为对应 FLOAT32。混合精度、Bias/ReLU/GELU 等 epilogue **本提交范围不含**；主路径对齐 `EPILOGUE_DEFAULT`。

### 当前基线与缺口

现网（950PR / `arch35`）已具备 Lt 句柄式接口与 FP32 Cube 路径，README 产品支持表将 A2/A3 标为不支持。

| 维度 | 当前基线（ops-blas master） | 本任务 |
| --- | --- | --- |
| 公开接口 | `include/cann_ops_blasLt.h` 已声明 `aclblasLtMatmul` | 不改 ABI、不重复定义 |
| Host 入口 | `blasLt/api/aclblaslt_matmul.cpp`：校验 → `BuildMatmulProblem` → `MatmulLaunch` | 共用该入口；A2 所需的 FLOAT32 语义差（见下）在共用 Host 用 dtype/arch 可区分的方式处理，禁止平行函数 |
| 引擎 | `blasLt/matmul/engine/matmul_engine.cpp`：FP32 / MXFP8 / MXFP4 MMAD + `epilogue_alpha_beta` | A2 只接通 FLOAT32 MMAD + α/β epilogue；MXFP **本提交范围不含**，A2 上保持 `NOT_SUPPORTED` |
| FP32 Kernel | 仅 `blasLt/matmul/fp32/arch35/` | 拟新增 `blasLt/matmul/fp32/arch22/` |
| Epilogue | 仅 `blasLt/matmul/epilogue/arch35/`（SIMT） | 拟新增 `arch22` Vector 实现，保持 `epilogue_alpha_beta_do` 符号，避免引擎分叉 |
| 测试 | `test/blasLtMatmul/arch35/` | 拟新增 `test/blasLtMatmul/arch22/`（对应任务书 `test/lt_matmul/`） |
| 产品表 | `blasLt/README.md`：A2/A3 不支持 | 交付时改为支持（FLOAT32 主路径） |

现网 Host 与任务书不完全一致，A2 验收以任务书为准，差异单独列出，不把 950 混合精度能力写成 A2 已交付能力：

| 项 | 现网代码 | 任务书 / 本任务 |
| --- | --- | --- |
| `transA/B = OP_C` | `HasSupportedTransposeModes` 仅放行 N/T，否则 `NOT_SUPPORTED`（注释：无共轭路径） | FLOAT32 实数路径 C 与 T 等价，须验收 |
| C 空指针 | `MatmulLaunch` 在 Host pointerMode 且 `beta ≠ 0` 且 `C == nullptr` 时返回 `INVALID_VALUE` | 任务书 §2.4 写「实现当前不校验 C，须补充」；本设计沿用引擎已有校验 |
| algo 合法性 | 非空时只比较 `workspaceSizeInBytes` 与 `algo->max_workspace_bytes`；`DecodeAlgo` 失败则忽略并走自动 tiling | 任务书要求补充算法支持性校验；负向暂不作为验收判定，拟实现、待评审 |
| `ld == 0` | `aclblasLtMatrixLayoutCreate` 把 `ld==0` 写成 `rows`；matmul 里 `ld==0` 视为 layout 合法；kernel 再用 `lda>0?lda:k` | 行主序应按**行宽**归一 |
| 混合精度 / 量化 | 引擎含 MXFP8/MXFP4 | 本任务不验收 |
| 目录名 | `blasLt/matmul/` | 任务书写 `blas/lt_matmul/` |

为何做：补齐 Blas 仓 A2/A3 的 cuBLASLt 风格矩阵乘，使 950 与 A2/A3 共用同一公开符号。

本任务不涉及 TBE、Python/ATen、广播或 OPP 注册。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（`arch22`）上，通过现有 `aclblasLtMatmul` 句柄式入口直调 NPU kernel，交付单精度实数行主序、单 batch、`EPILOGUE_DEFAULT` 的 `D = α·op(A)·op(B) + β·C`。支持 N/T/C（C≡T）、紧凑与 padding 前导维、C 与 D 原地/异地、Host/Device pointerMode 的 FLOAT32 标量、调用方 workspace（非空须 16B 对齐）。不支持列主序、batch>1、混合精度、非 DEFAULT epilogue、超出 ld 的非连续访问。精度与性能按任务书 §3 验收，本设计阶段不填写未实测数据。

## 需求拆解

1. **接口和 ABI**  
   沿用 `aclblasLtMatmul(...)` 16 参数声明；workspace 与 stream 为显式入参，不得省略。不新增符号。

2. **功能、dtype、shape、format**  
   FLOAT32 × FLOAT32 → FLOAT32；`order=ROW`；`m,n,k ≥ 0`；单 batch。`ld==0` 按行宽紧凑处理。

3. **Host 校验和错误行为**  
   校验顺序按任务书 §2.5：handle → 描述符判空 → alpha/beta 判空 → pointerMode → `m==0`/`n==0` no-op → 布局（含 order 与 batch）→ A/B/D 判空 → workspace。  
   状态码：handle 空 → `NOT_INITIALIZED`；非法值 → `INVALID_VALUE`；非法枚举 → `INVALID_ENUM`；order 非 ROW 或不支持组合 → `NOT_SUPPORTED`；下发失败 → `EXECUTION_FAILED`。与 `include/cann_ops_blas_common.h` 一致。

4. **Kernel、Tiling、分核**  
   不使用 TilingKey。Host 填写 `MatmulFp32TilingData`（及 arch22 epilogue tiling），按 Cube 核数切 MN 块；K 在 L1/L0 上分段。分支由 tiling 内 `transA/transB` 与 Host 路径选择表达，不用虚构 key。

5. **边界与 quick return**  
   `m=0` 或 `n=0`：成功且不计算（此时布局非法或 A/B/D 为空仍成功，描述符与 α/β 仍须非空）。  
   `k=0`：只做 β 缩放，不引用 A/B。  
   `α=0`：不引用 A/B；`β=1` 时 C 不变（异地则等价拷到 D），`β=0` 时 D 置零，其余逐元素缩放。α=0 且指针非空时 `D=β·C` 须位精确（EXACT）。

6. **精度和性能验证**  
   golden：`cblas_sgemm` 全矩阵比对 D。FLOAT32：rtol/atol `2^-13`，matched_ratio ≥ 0.99，max_abs_error ≤ `1e-2` 或 32 ULP。Inf/NaN 按任务书特殊规则。性能：Atlas 800T A2，msprof op，5 次 warmup + 10 次平均，单 case 平均耗时 ≤ GPU 的 1.25 倍（≥ 0.8× GPU）。workspace 峰值 ≤ 输入输出数据总量的 1.2 倍。**上述门限来自任务书，达成情况待测。**

7. **兼容与回归**  
   不改公开头文件。arch35 源文件不纳入 A2 编译（现有 CMake 已按 arch 过滤）。A2 不实现 MXFP。对 `OP_C` 的 FLOAT32 放行若改共用 Host，须保证 950 非实数/无共轭路径不被静默算成转置；拟仅在 A/B 为 `ACL_FLOAT` 时将 C 视为 T。

8. **上层入口**  
   调用方创建 handle、matmulDesc、四个 layout，可选 `aclblasLtMatmulAlgoGetHeuristic`；本算子本身无 Python 入口。

# 详细设计（required）

## 算子分析

### 公式与存储

行主序 ND：逻辑 `op(A)` 为 m×k，`op(B)` 为 k×n，C/D 为 m×n。物理行距为 ld（元素计）。无广播、无视图返回。异步：工作下发到入参 `stream`（可为默认流）；读回前由调用方同步该 stream。

### 标量与 pointerMode

`scaleType` 现网仅接受 `ACL_FLOAT`。`ACLBLASLT_POINTER_MODE_HOST`：Host 解引用 α/β。`ACLBLASLT_POINTER_MODE_DEVICE`：kernel 从 Device 读 4 字节标量，Host 不得把 Device 指针当 Host float 解引用。其它 pointerMode → `INVALID_VALUE`。

### workspace 与原地

引擎现网：需要 epilogue 且（C 与 D 重叠，或 raw 宽度与 D dtype 不一致）时，先把原始乘积写到 workspace，再 epilogue 写 D。本任务 D 固定 FP32，raw 亦为 float，故 **C==D 且需要 epilogue**（Device 标量，或 α≠1 或 β≠0）时必须 staging。非空 workspace 16B 对齐；容量不足 → `INVALID_VALUE`。heuristic 须把该 staging 计入 `max_workspace_bytes`，避免 algo 为 NULL 时引擎仍要 workspace。拟实现，待与 `aclblasLtMatmulAlgoGetHeuristic` 对齐评审。

### 不推测闭源 GPU kernel

对标只采用公开公式、ld 约定、quick return 与错误码语义；不描述 cuBLASLt 内部算法。

## 算子实现

### 总体架构

只保留现网真实层次：

```text
调用方
  → aclblasLtMatmul                         [blasLt/api/aclblaslt_matmul.cpp]
      → 参数校验 / quick return
      → BuildMatmulProblem                  [blasLt/internal]
      → MatmulLaunch                        [blasLt/matmul/engine]
            → matmul_fp32_get_tiling / matmul_fp32_do     [拟：fp32/arch22]
            → epilogue_alpha_beta_do                      [拟：epilogue/arch22]
```

无 Python、无 ATen、无 TBE。CMake 已按 `*_kernel.cpp` 与 `archXX` 目录分流；在 `fp32/arch22` 与 `epilogue/arch22` 落文件即可编进 A2，不必改公开 API CMake 结构。

引擎目前无条件 `#include` MXFP host 头文件。A2 构建排除 `arch35` 后会缺符号。拟实现（待评审，二选一，不并行发明第二套 Launch API）：

- 条件编译：非 arch35 或未开 `ENABLE_BLASLT_MXFP8` 时 MXFP 分支编译为 `NOT_SUPPORTED`；或
- 为 arch22 提供同名弱桩，仅返回 `NOT_SUPPORTED`。

### Host：校验、quick return、tiling

沿用 `aclblaslt_matmul.cpp` 现有顺序，并按任务书补齐/对齐：

1. `lightHandle == nullptr` → `NOT_INITIALIZED`
2. computeDesc / Adesc / Bdesc / Cdesc / Ddesc 任一空 → `INVALID_VALUE`
3. alpha 或 beta 空 → `INVALID_VALUE`
4. pointerMode 非 HOST/DEVICE → `INVALID_VALUE`；`scaleType != ACL_FLOAT` → `INVALID_VALUE`（现网已有）
5. FLOAT32 路径：`transA/transB ∈ {N,T,C}`，C 按 T 编码进 `MatmulProblem.transA/transB`；其它枚举 → `INVALID_ENUM` 或 `NOT_SUPPORTED`（与头文件枚举定义一致，非法枚举优先 `INVALID_ENUM`）
6. 由 D layout 取 m、n，由 A 与 transA 取 k；`m==0 || n==0` → `SUCCESS`（不再查 layout/A/B/D/workspace）
7. 布局：uint32 可表示；单 batch 且 stride 0；行主序 ld 约束；`ld==0` 按行宽；order 非 ROW → `NOT_SUPPORTED`；batch 非法 → `INVALID_VALUE`
8. 本任务 dtype 仅 `ACL_FLOAT`；其它组合 A2 → `NOT_SUPPORTED`
9. 需计算时 A/B/D 空 → `INVALID_VALUE`；Host 模式且 β≠0 且 C 空 → `INVALID_VALUE`
10. workspace 非空未 16B 对齐，或 algo 非空且容量小于 `algo->max_workspace_bytes` → `INVALID_VALUE`
11. `QueryCubeCoreNum` 得到 `numBlocks`，组 `MatmulProblem`，`MatmulLaunch`

**不分 TilingKey。** FP32 tiling 沿用现网 POD `MatmulFp32TilingData`：`m/n/k`、`baseM/baseN/baseK`、`kL1`、`usedCoreNum`、`lda/ldb`、`transA/transB`。arch22 的 `matmul_fp32_get_tiling` 按 A2 L0/L1 预算设默认块（现网 arch35 默认为 `baseM=min(m,128)`、`baseN=min(n,256)`、`baseK`/`kL1` 按 128/512 字节量级）。A2 具体默认值以上机 L1/L0 容量为准，**本设计不把 arch35 数字写成已在 A2 验证的规格**。若调用方传入可 `DecodeAlgo` 的 algo，走现有 `ApplyAlgoTilingOverrideFp32` 覆盖 `baseN`/`kL1`。

分核：MN 平面 `tileNum = ceil(m/baseM)*ceil(n/baseN)`，核 `blockIdx` 以 stride=`blockNum` 领取 tile（与现网 `MatmulFp32Kernel` 一致）。核数取 `min(tileNum, cubeCoreNum)`。

`ld==0`：在组 problem / tiling 时写成行宽，kernel 不再猜测。

algo 支持性（拟实现）：magic 不匹配或字段不在 heuristic 枚举域时返回 `INVALID_VALUE`；任务书称负向暂不验收，默认路径仍允许 `algo==NULL` 走自动 tiling。

### Kernel：MMAD 与 epilogue

**FP32 MMAD（AIC）**

结构对齐现网 `blasLt/matmul/fp32/arch35/matmul_fp32_kernel.cpp`，在 arch22 重写设备侧实现，而不是调用 `blas/gemm/arch22` 的列主序 `sgemm`（布局与入口都不同，禁止把 Lt 接到那条 host 适配器上充数）。

阶段（不是通用 Vector 三段式，因为这是 Cube）：

1. **Init**：按 tiling 绑定 GM、切分 L1/L0 缓冲。
2. **CopyIn（GM→L1 Nd2Nz，L1→L0 LoadData）**：`srcDValue` 使用 **lda/ldb**，支持 padding。
3. **Compute**：`Mmad` 沿 K 累加；K 外层 L1 tile、内层 L0 tile。
4. **CopyOut（Fixpipe NZ→ND）**：写出原始乘积。stride 在「直接写 D 且无 epilogue」时必须用 **ldd**；staging 到 workspace 时用紧凑行宽 n。现网 CopyOut 使用 `dstStride = n`，在 `ldd>n` 且跳过 epilogue（α=1,β=0）时与任务书不符。A2 **拟**：凡 `ldd != n` 或 `ldc != n` 或需要 α/β，均 staging + epilogue，避免漏写 padding 行。

现网 arch35 kernel 使用 L1/L0 ping-pong 与 `SetFlag/WaitFlag`。A2 拟采用同等双缓冲；**未上机确认前不宣称已流水或已达带宽上限**。

`k=0`：不启动 MMAD，只走 epilogue/清零/缩放。

**α/β epilogue（AIV）**

arch35 epilogue 为 SIMT，不能当作 A2 已有实现。arch22 拟用 Ascend C Vector：

- `D[i,j] = alpha * D_raw[i,j] + beta * C[i,j]`（β=0 或不使用 C 时不加 C）
- 按行处理 padding：C 用 ldc，D 用 ldd，raw 用 lddRaw
- Host 标量走 tiling 传入；Device 标量由 kernel 读 GM，保持异步、不做 Host D2H
- α=0 路径跳过 MMAD，epilogue 对 C 做缩放或对 D 清零；该路径 EXACT
- 分核按 `m*n` 元素或按行切分，UB 切块；不用 TilingKey

α=1 且 β=0 且 ldd 紧凑且非 Device 标量：可跳过 epilogue，MMAD 直写 D（现网 `needEpilogue` 逻辑）。A2 保持该短路。

### 目录与构建

拟议目录（与现网 CMake `GLOB_RECURSE` + arch 过滤一致）：

```text
blasLt/
  api/aclblaslt_matmul.cpp          # 已有，按需做 FLOAT32 OP_C / 校验对齐
  matmul/engine/matmul_engine.cpp   # 已有，A2 接通 FP32；MXFP 条件化
  matmul/fp32/arch22/               # 拟新增
    matmul_fp32_host.cpp / .h
    matmul_fp32_kernel.cpp / .h
    matmul_fp32_tiling_data.h       # 可与 arch35 共享 POD；若 ABI 字段不变则复用 common 头，待实现时避免双份定义
  matmul/epilogue/arch22/           # 拟新增
    epilogue_alpha_beta_host.cpp / .h
    epilogue_alpha_beta_kernel.cpp / .h
    epilogue_alpha_beta_tiling_data.h
test/blasLtMatmul/arch22/           # 拟新增 CSV + GTest
  blasLtMatmul_test.cpp
  blasLtMatmul_test.csv
  blasLtMatmul_npu_wrapper.h
```

公共冲突策略：不复制 `include/cann_ops_blasLt.h`；不新增 `aclblasLtMatmulA2`。arch22 与 arch35 同名 launch 函数靠 **只编译当前 SOC 的 arch 目录** 共存，与 `blasLt/CMakeLists.txt` 现逻辑一致。

任务书若强制物理路径 `blas/lt_matmul/arch22`，则只允许 thin wrapper 调入 `blasLt/matmul/fp32/arch22`，禁止第二套校验与 kernel。默认不建该平行树，待评审。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列（含 Atlas 800T A2 / 910B3） | √（任务范围；自验建议精度/性能用 910B3） |
| Atlas A3 系列 | √（任务范围；与 A2 同 arch22，自验证覆盖一种款型即可） |
| Ascend 950PR / arch35 | 不在本任务交付范围（现网已有，本文不勾选为本次验证硬件） |

不写开发机型号与卡号。CANN 版本任务书要求 9.1.0。

## 算子约束限制

- 仅 FLOAT32 主路径；FP16/BF16/FP8/INT8/MXFP 本提交范围不含。
- 仅 `ACLBLASLT_ORDER_ROW`；列主序 `NOT_SUPPORTED`。
- 仅 `batchCount=1` 且 `stridedBatchOffset=0`。
- 仅 `EPILOGUE_DEFAULT`；Bias/ReLU/GELU 不含。
- 不支持超出 lda/ldb/ldc/ldd 的非连续访问，无 broadcast，无 dynamic shape 机制（m/n/k 运行时传入）。
- 不要求确定性累加顺序；非 bit-exact（α=0 缩放除外）。
- 调用方保证：有效 handle 与 desc、Device 上的 A/B/C/D、按 heuristic 分配且 16B 对齐的 workspace、读回前同步 stream。
- `m=0`/`n=0` 的 no-op 发生在布局校验之前（描述符与 α/β 仍非空）。
- 不把 arch35 SIMT epilogue、MXFP kernel、列主序 `sgemm` arch22 标为 A2 Lt 已支持能力。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32：rtol=atol=`2^-13`（约 1.2e-4），required_matched_ratio=0.99，max_abs_error_limit=`1e-2` 或 32 ULP；逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|`。Inf/NaN：同 NaN、同号 Inf、或 NaN 与 ±Inf 视为等效；有限 vs 非有限不通过；非有限元素不计入 matched_ratio / max_abs_error，须全部按上述规则通过。α=0 路径 EXACT。 | 任务书 §3.2；opbase 混合容差标准 |
| 性能标准 | 设备 Atlas 800T A2；msprof op 单算子 task duration（us）；5 次 warmup 不计；其后 10 次 SUCCESS 样本算术平均；单 case Avg time ≤ GPU baseline 的 1.25 倍（≥0.8×）。参考 `test_cases/gpu_baseline.csv`。 | 任务书 §3.3 |
| 内存标准 | 各性能 case 报告 workspace 峰值，不超过输入输出数据总量的 1.2 倍；非空 workspace 16B 对齐。 | 任务书 §3.4 |

本设计阶段不填写未实测耗时或倍率。

## 测试设计

框架：ops-blas `test` 目录 CSV + Python 拉起 C++ GTest，调用 `aclblasLtMatmul`；golden 为 cblas `cblas_sgemm`。任务书要求覆盖约 200 条配套用例，缺场景则补测。

| 类别 | 覆盖维度 | 状态 |
| --- | --- | --- |
| 正向 | transA/transB ∈ {N,T,C}；紧凑与 padding ld；C==D / C!=D；pointerMode HOST（任务书自验表）；α/β 特殊值 0/1/大值；均匀与正态输入 | 待测 |
| 形状 | m/n/k：0、1、小质数、2 的幂及 ±1、非对齐、大规模 | 待测 |
| 边界 | k=0 只缩放；α=0 EXACT；m=0/n=0 no-op（含空指针矩阵仍 SUCCESS） | 待测 |
| 负向 | handle 空 → `NOT_INITIALIZED`；desc/α/β 空 → `INVALID_VALUE`；order 非 ROW → `NOT_SUPPORTED`；非法 ld；A/B/D 空（需计算时）；workspace 未对齐 / 容量不足 | 待测 |
| 特殊值 | 规格允许的 Inf/NaN | 待测 |
| 性能 | gpu_baseline.csv 各 case；warmup 与采样按任务书 | 待测 |
| 回归 | 不修改 arch35 源；A2 构建不链入 950 kernel | 待测 |

C 空指针：任务书写现网不作为负向验收；本设计 Host 模式 β≠0 时返回 `INVALID_VALUE`，对应该行为的用例若与配套 CSV 冲突，以任务书 §2.5 校验顺序为准并在自测报告说明。

无 Python UT 要求则不编造 Python UT 通过结论。

## 兼容性分析

- **新增**：arch22 的 FP32 kernel/epilogue 与 arch22 测试目录；README 产品表 A2/A3 改为支持。
- **增量架构**：同一 `aclblasLtMatmul` 符号，按 SOC 编译不同 arch 目录。
- **dtype/shape 扩展**：本任务不扩展混合精度。
- **并行贡献**：不改 950 量化 kernel；共用 Host 仅做任务书要求的 FLOAT32 语义对齐（OP_C、ld 行宽、C 空指针）。冲突点见风险。
- 不适用项：TBE 版本对比、Python API 兼容——不涉及。

## 风险与对策

| 风险 | 对策（不改为另一套未实现方案） |
| --- | --- |
| 大尺寸 FP32 对 cuBLASLt 达不到 0.8× | 先保证正确性与 heuristic workspace；在自动 tiling 上调 A2 的 baseM/N/K 与核间切分。不引入任务范围外的 epilogue 融合充性能。未达标不写已达成。 |
| 共用 Host 放行 OP_C 影响 950 非共轭路径 | 仅 `ACL_FLOAT` 将 C 映射为 T；其它 dtype 保持现网 `NOT_SUPPORTED`。 |
| 引擎硬依赖 arch35/MXFP 头文件导致 A2 无法链接 | 条件编译或 arch22 桩返回 `NOT_SUPPORTED`，保持 `MatmulLaunch` 单一入口。 |
| `ldd>n` 且跳过 epilogue 时现网 CopyOut stride=n | A2 对非紧凑 D 强制 staging+epilogue。 |
| layout `ld==0` 写成 rows 与行宽语义冲突 | matmul 路径按任务书用行宽；是否回改 `MatrixLayoutCreate` 待评审，避免无任务地改布局 API。 |
| workspace 未计入 C==D staging，heuristic 偏小 | 更新 `CalculateWorkspaceForAscend` / 默认 algo 的 `max_workspace_bytes`，并保证峰值 ≤ 1.2× IO。 |
| 任务书目录 `blas/lt_matmul` 与现网 `blasLt` 不一致 | 默认现网目录；评审要求时再加转发目录，不复制 kernel。 |

---

## 待实现 / 待测 / 待确认

- 待实现：`blasLt/matmul/fp32/arch22`、`epilogue/arch22`、A2 测试树、heuristic staging 字节数、FLOAT32 的 OP_C 映射、引擎 MXFP 条件化。
- 待测：200 条用例精度、Inf/NaN、EXACT、msprof 性能与 workspace 峰值。
- 待确认：任务书路径 vs `blasLt/`；`MatrixLayoutCreate(ld==0)` 是否改为行宽；algo 非法是否对调用方报错。
