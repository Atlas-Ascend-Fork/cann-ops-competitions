# aclblasCherk3mEx 算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasCherk3mEx` |
| 任务编号 | 11 月社区任务-aclblasCherk3mEx 算子开发（950） |
| 目标硬件 | 昇腾 950PR（arch35，DAV_3510） |
| 软件版本 | CANN 9.1.0 及以上（asc-devkit >= 9.1） |
| 开发方式 | Ascend C kernel 直调（handle 绑定 stream） |
| 数据类型 | Atype ∈ {ACL_COMPLEX64, ACL_INT8}，Ctype = ACL_COMPLEX64（不支持组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`） |
| 标量 | alpha/beta 为**实数 FLOAT32**（区别于 Csyrk 族复数标量），Host 与 Device 两种指针位置 |
| 对齐基线 | cuBLAS `cublasCherk3mEx`（语义参考 Netlib `cherk`）；精度 golden 由 CBLAS `cblas_cherk`（Netlib BLAS 3.10.0）生成 |
| 合入仓及目录 | `ops-blas`：算子 `blas/herk/arch35/`，测试 `test/herk/cherk3m_ex/arch35/`，公共头 `include/cann_ops_blas.h` **新增声明** |
| 文档状态 | 设计定稿；接口声明、dtype 支持集与错误码口径按本任务书对齐，精度/性能数据按 §3.3 协议在 950PR 实测后回填自测报告 |

# 1. 需求背景（required）

## 1.1 需求来源

CANN 社区任务 2026，11 月社区任务-aclblasCherk3mEx 算子开发（950）。任务书：`aclblasCherk3mEx_Atlas950PR_task_doc.md`。贡献者：`zhanghua145`；实现目标仓库：[cann/ops-blas](https://gitcode.com/cann/ops-blas)。本设计按[官方模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织。

## 1.2 背景介绍

`cherk3m` 是复数 Hermitian 秩-k 更新的 **3M 方法**实现：把复数乘法拆解为 3 次实数矩阵乘（朴素为 4 次），实数乘法量降 25%，是 cuBLAS `cublasCherk3mEx` 家族在昇腾 Cube 阵列上的同思想移植：

$$
C \leftarrow \alpha \cdot op(A) \cdot op(A)^H + \beta \cdot C
$$

- `trans = ACLBLAS_OP_N`：op(A) = A（A 为 `n×k`）；
- `trans = ACLBLAS_OP_C`：A 以 `k×n` 存储，op(A) = A^H（**共轭转置**）；
- `trans = ACLBLAS_OP_T`：**不属于 Hermitian 秩-k 定义域，返回 `ACLBLAS_STATUS_INVALID_ENUM`**（任务书 §2.1 有意收窄，cuBLAS 列入合法枚举但公式未定义）。

关键语义（与任务书 §2.1 逐项对齐）：

1. **Hermitian 非对称**：乘积是 X·X^H（不是 X·X^T）；C(i,i) 虚部在计算后**强制置 0**（区别于 Csyrk 对角虚部为有效数据）。
2. **三角读写**：C 为 `n×n` Hermitian（列主序），仅 uplo 指定三角被引用与更新，另一三角由共轭对称性 C = C^H 隐含、不写入。
3. **dtype（与 cuBLAS 一致）**：`(A=ACL_INT8, C=ACL_COMPLEX64)` 与 `(A=ACL_COMPLEX64, C=ACL_COMPLEX64)`；`ACL_INT8` 为 complex-int8，每个复元素 `(int8 real, int8 imag)` 交错 2 字节存储（对应 cuBLAS `CUDA_C_8I`），计算前精确提升为 float；其他组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
4. **实数标量**：alpha、beta 为单精度实数标量（任意有限 float），支持 Host 与 Device 两种指针位置。
5. **no-op 与短路**：`n=0` 合法 no-op；`k=0` 或 `alpha=0` 时秩-k 贡献为 0、不引用 A——`beta=1` quick return（C 不变），`beta≠1` 仅对 uplo 三角做 `C=beta·C`；`beta=0` 时不读取 C 旧值。
6. **3M 舍入边界**：3M 中间 FP32 舍入与直接复数乘法不同，结果与 `aclblasCherkEx`/`cublasCherk` 不保证逐位一致，按任务书 §3.2 阈值判定（接口语义与数学定义一致）。
7. **独立性**：`aclblasCherkEx` 与本算子均为 1125 新增的独立接口，彼此不得依赖；3M 三次实数乘必须在**同一次 kernel launch** 内完成，不得用三次独立 GEMM 调用拼接。

`include/cann_ops_blas.h` 现无 `aclblasCherk3mEx` 声明，需按任务书 §2.3 新增公共声明，禁止定义 950PR 私有平行 API。仓内参考：非 Ex `cherk`（`blas/herk/arch35/cherk_host.cpp`/`cherk_kernel.cpp`，参数校验、uplo 三角处理、对角置 0 与缩放逻辑）与 `cgemm3m`（`blas/gemm/arch35/cgemm3m_host.cpp`、`cgemm3m_rounding_kernel.cpp`，3M 分解与舍入处理）。

# 2. 需求分析（required）

## 2.1 接口定义

与 cuBLAS `cublasCherk3mEx` 逐参数对应（任务书 §2.3，参数名 Atype/Ctype）：

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,      // ACLBLAS_UPPER / ACLBLAS_LOWER
    aclblasOperation_t trans,    // ACLBLAS_OP_N / OP_C（OP_T 拒绝）
    int n, int k,
    const float* alpha,          // 实数标量，Host 或 Device 指针
    const void* A, aclDataType Atype, int lda,
    const float* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

## 2.2 参数与校验（任务书 §2.4/§2.5 对齐）

| 约束项 | 规则 | 错误码 |
| --- | --- | --- |
| handle | 非空句柄 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | ∈ {ACLBLAS_UPPER, ACLBLAS_LOWER} | `ACLBLAS_STATUS_INVALID_ENUM` |
| trans | ∈ {ACLBLAS_OP_N, ACLBLAS_OP_C}；**OP_T 拒绝** | `ACLBLAS_STATUS_INVALID_ENUM` |
| Atype/Ctype | Atype ∈ {ACL_COMPLEX64, ACL_INT8} 且 Ctype = ACL_COMPLEX64，组合不支持 | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| n, k | ≥ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda | trans=N：≥ max(1,n)；trans=C：≥ max(1,k) | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldc | ≥ max(1,n) | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha/beta | 指针非空（Host/Device 同类别位置，dispatch 时读回真值） | `ACLBLAS_STATUS_INVALID_VALUE` |
| A | n>0 且 k>0 时非空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| C | n>0 时非空 | `ACLBLAS_STATUS_INVALID_VALUE` |

校验顺序（任务书 §2.5）：handle → uplo/trans → n/k → lda/ldc → alpha/beta/Atype/Ctype 与 A/C 指针 → 结构合法后执行 quick return；`n=0` 时允许 A、C 为 NULL 且不启动 kernel；`k=0` 或 `alpha=0` 时 A 可为 NULL，`beta≠1` 仍需 `C=beta·C`。非法枚举返回 `INVALID_ENUM` 为 ops-blas 约定（cuBLAS 返回 INVALID_VALUE，属有意差异）。

## 2.3 需求拆解

1. 公共 API 声明**新增**合入 `include/cann_ops_blas.h`；状态码语义对齐仓内 `cann_ops_blas_common.h`。
2. `C = α·op(A)·op(A)^H + β·C` 的 **Hermitian 3M 拆解**在 950PR Cube 上单次 launch 高效完成：P0/P1/P2 三实积（§3.1.1），Cube 走 FP16 hi/lo 补偿平面保 FP32 精度、取 FP16 Mmad 双倍吞吐。
3. uplo 三角读写：仅 uplo 三角被计算与写回；**对角虚部计算后强制置 0**。
4. INT8 输入精确提升（int8 → float 无损）后复用同一 3M 管线；Ctype 固定 COMPLEX64。
5. 实数 alpha/beta 全套缩放（含 0、1、-1、±1e4 任务书特殊值），β=1 quick return、β=0 不读 C。
6. 特殊值语义：A 含 Inf/NaN、`alpha=0` 且 A 为 NaN（不引用 A）、`beta=0` 且 C 为 NaN（不读旧 C）等（任务书 §3.5）与 CBLAS golden 按阈值一致；非有限标量与值域超限走 Netlib 同序参考兜底。
7. 性能：kernel-to-kernel 口径，单 case ≤ `gpu_ms` 的 2.5 倍且全 case 加速比平均 ≥ 0.5（§3.3）。
8. 测试工程：任务包 `test_cases/`（1000 精度 + 200 性能，含 `TC_3M_*` 4 条专项）CSV 驱动 GTest，精度 golden 由 CBLAS `cblas_cherk` 生成。

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式与 3M 拆解（Hermitian 专用分解）

令 X = op(A) = Xr + i·Xi（Atype=ACL_INT8 时先按 (int8 re, int8 im) 精确提升为 float）。乘积是 **X·X^H**：B = X^H = Xr^T − i·Xi^T，即 Br = Xr^T、Bi = −Xi^T。非融合 3M 形式：

$$
P_0 = X_r X_r^T,\quad P_1 = X_i X_i^T,\quad P_2 = (X_r + X_i)(X_r - X_i)^T
$$

$$
\mathrm{Re}(X X^H) = P_0 + P_1,\qquad \mathrm{Im}(X X^H) = P_2 - P_0 + P_1
$$

- 三次实数矩阵乘在**同一次 kernel launch** 内完成（Cube kernel 单发射三积），不得用三次独立 GEMM 拼接；
- **不得套用对称 X·X^T 分解**（P2=(Xr+Xi)(Xr+Xi)^T、Re=P0−P1）：会把虚部算成 Xi·Xr^T + Xr·Xi^T（正确值为 Xi·Xr^T − Xr·Xi^T），属实现错误；本 Hermitian 分解的 P2=(Xr+Xi)(Xr−Xi)^T 天然携带负号结构；
- **对角精度**：对角元 Re=P0+P1 为大和相加（无抵消），Im=P2−P0+P1 为三项差（有抵消）。Phase 0 顺带产出对角行和表（FP32 升序累加：Σxr²、Σxi²、Σxr·xi，Netlib 同调理），Phase 2 对角带查表恢复；对角虚部按语义**置 0**（非保留）；
- trans=C 读共轭转置视图：按 (k×n) 存储、列主序逐列读 A，pack 阶段等价交换 K/N 语义并翻转虚部符号，与 trans=N 同型处理；
- 融合路径可参考仓内 cgemm3m 等价公式，但必须保证数学结果等于 X·X^H。

### 3.1.2 支持数据类型与形状

- Atype ∈ {ACL_COMPLEX64（8B/元素）, ACL_INT8（2B/元素，(int8 re, int8 im) 交错）}；Ctype = ACL_COMPLEX64；列主序 ND。
- A：trans=N 为 `n×k`，trans=C 为 `k×n`；C：`n×n`（ldc×n 存储）。
- n、k 为运行时入参（无编译期 shape 要求）；workspace 溢出防护（64 位乘法检查 + 2GiB 上限）。

## 3.2 算子实现

### 3.2.1 host 侧设计（路径分流）

```
aclblasCherk3mEx
  ├─ 校验（§2.2 顺序）→ n==0 quick return（不启动 kernel）
  ├─ alpha/beta 真值读回（ReadScalarFromHostOrDevice，Host/Device 双位置）
  ├─ 短路：α==0 或 k==0
  │    ├─ β==1 → quick return（C 不变）
  │    └─ 否则 → 仅 combine 的 betaOnly 阶段（C_uplo = β×C，不打包、不算积、不引用 A）
  ├─ 小形状 SIMT/向量单发射（C64 输入）：
  │    ├─ vec kernel：trans=N && 5≤n≤112 && k≤256（SIMD 列核，行广播 + Add 树归约）
  │    └─ small/点积 kernel：n≤32（N）/ n≤64（C）工作量门控（每元素一 warp、lane 分 K、树归约）
  ├─ mix kernel（单发射小形状）：n≤128 && k≤512（block 私有 band 乒乓 → cube 3M → 自合并）
  ├─ 非有限标量 / 值域超限 → 强制 Netlib 同序参考 kernel 单独产出 C（Inf/NaN 语义与 golden 一致）
  └─ 三阶段管线（其余全部形状，含全部 ACL_INT8 输入）
```

- ACL_INT8 输入统一走三阶段管线：pack 阶段按 `(int8 re, int8 im)` 精确提升（无损），后续与 C64 同路。
- **值域护栏**：C64 输入元素超出按 |alpha| 动态缩放的 FP16 表示域（guard ≈ 65000）时由 pack kernel 置 range flag，combine kernel 读 flag 自动切换为 SIMT Netlib 同序参考——3M 重排与 Netlib 顺序在大 |alpha|·|x|² 下会产生不同 Inf/NaN 分类，必须回退保证 §3.5 特殊值口径。flag 携带随机 epoch（进程内自增、首值随机），杜绝复用 workspace 中陈旧 flag 误判。
- **cube 成本模型**：band tile 位宽 bm ∈ {64,128}，按每 chunk 实测常数估总耗时取小者；三积共享同一 launch 的 band tile 划分。

**workspace 布局**（512B 对齐，全程 64 位乘法溢出检查）：

```
[P0|P1|P2 全平面 + 对角 tile 平面：fp16 hi/lo 各 ×2 → 6 半平面 ×2B]
[temp: T0..T2 三积 n×n FP32 ×4B + P3 对角校验积]
[对角行和表：band(64 行) × pack核 × 64 floats（含 Σxr·xi 行）]
[range flag 64B]
总量 > ACLBLAS_MAX_WORKSPACE_SIZE(2GiB) → 返回 `ACLBLAS_STATUS_ALLOC_FAILED`
```

### 3.2.2 kernel 侧设计（三阶段管线，单 launch 三积）

```
Phase 0  AIV pack     C64/INT8 → 提升后拆为 FP16 hi/lo 补偿平面（Xr、Xi、Xr+Xi 三平面各 hi/lo）
   │                  （trans=C 时按共轭转置视图装载，Xi 平面符号翻转）
   │                  + 对角行和表（FP32 升序，Netlib 同调理）+ 值域检查（超限写 range flag）
   │                  + 按 64 行带分核，带内连续列 DMA
Phase 1  AIC cube     3M 三积 band tile（单 launch）：T0=Xr·Xr^T、T1=Xi·Xi^T、T2=(Xr+Xi)·(Xr−Xi)^T
   │                  hi/lo 双积补偿（xh·yh + xl·yh + xh·yl）保 FP32 级精度、取 FP16 Mmad 2× 吞吐；
   │                  uplo 三角 tile 跳过（非三角不算不写）
Phase 2  AIV combine  C_uplo = α·(T0+T1, T2−T0+T1) + β·C_old（实数 α/β 缩放，无交叉项）
                      对角带查行和表恢复对角 Re 精度、对角 Im 强制置 0；β=0 不读 C；
                      读 range flag，超限整核切换 Netlib 同序 SIMT 参考兜底
```

**单发射 kernel 族**：

| kernel | 硬件 | 触发条件 | 设计要点 |
| --- | --- | --- | --- |
| vec | AIV | N/C64/upper，5≤n≤112，k≤256 | SIMD 列核，行广播 + Add 树归约；3M Hermitian 公式逐列内联完成 |
| small | AIV(SIMT) | N n≤4；点积方案 N n≤32 / C n≤64 | 每列一 warp；点积方案每元素一 warp、lane 分 K、树归约；对角 Im 置 0 |
| mix | AIC+AIV | n≤128，k≤512，band tile ≤ AIC 核数 | 单发射完成 pack→cube 3M→combine；block 私有带槽乒乓 |
| ref | AIV(SIMT) | 非有限标量 / range flag 命中 | Netlib 逐元素同序（与 CBLAS golden 同调理），容差按构造成立 |

**Phase 2 combine kernel 要点**：

- 实数 α/β 缩放：`Cr = α·Pr + β·Cr_old`、`Ci = α·Pi + β·Ci_old`，Interleave 后连续列写回 uplo 三角；非 uplo 三角不读不写。
- k=0/α=0 时只跑 β 缩放；β=0 时不读 C 旧值（任务书：beta=0 且 C 为 NaN 不传染）。
- 对角带用 pack 行和表恢复 `C[i,i]` 实部，避免 P2−P0+P1 三项差抵消失精度；对角虚部置 0（Hermitian 语义，非保留）。

### 3.2.3 确定性设计

- 全部 kernel 的 band/tile/工作项分配为固定规则，无原子、无锁、无调度随机性；归约顺序固定（cube hi/lo 固定积序、combine 固定树序、参考路径 Netlib 序），同输入同 stream 串行多次执行输出逐位一致。
- 任务书对确定性不作硬性要求（§2.5），本设计天然确定，超出任务要求。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 昇腾 950PR（arch35） | √ |

依赖 CANN 9.1.0 及以上（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`，低于该版本编译与运行跳过本算子，经 `cmake/asc_devkit_version.cmake` 列表驱动登记）。

## 3.4 算子约束限制

- 不支持超出前导维度语义的非连续内存布局（仅 lda/ldc 描述的列主序带填充访问），不使用 AutoContiguous/ViewCopy；不支持广播。
- C 为原地覆写（输入旧值参与 β 项后写回同一内存），不返回视图；仅 uplo 三角被读写；调用方保证 C 可写区与 A 可读区不重叠。
- 异步执行：依赖 `aclblasSetStream` 绑定 stream，接口返回不代表 Device 计算完成，Host 读回前须同步 stream。
- workspace 溢出（>2GiB）返回 `ACLBLAS_STATUS_ALLOC_FAILED`。
- 3M 与直接复数乘舍入过程不同，与 `aclblasCherkEx`/`cublasCherk` 不保证逐位一致（接口语义与数学定义一致，按 §3.2 阈值判定）。

# 4. 可维可测分析

## 4.1 精度标准

对齐[生态算子开源精度标准 mixed_tolerance_standard](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)，本算子输入输出 COMPLEX64 按 FLOAT32 分量判定（实部/虚部分别比对、分别统计）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| FLOAT32（复数分量） | 2^-13 (1.2e-4) | 2^-13 (1.2e-4) | 0.99 | 1e-2 或 32·ULP |

- 判定式：逐分量 `|actual − golden| ≤ atol + rtol·|golden|` 计入 matched；`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max(1e-2, 32·ULP)` 同时满足判过。
- golden 由 CPU 侧 CBLAS **非 3M** `cblas_cherk`（Netlib BLAS 3.10.0）生成：Atype=ACL_COMPLEX64 直接调用；Atype=ACL_INT8 先把 int8 实/虚部精确提升为 float 再调用；需要 float64 参考域时用 `cblas_zherk` 后收窄到 complex64。3M 与 CBLAS 直接复数乘的舍入差异在自测报告中说明，不逐位比对。
- 特殊值（任务书 §3.5）：Inf/NaN 用例（A 含 Inf/NaN；`alpha=0` 且 A 为 NaN——验证不引用 A；`beta=0` 且 C 为 NaN——验证不读旧 C）、**对角虚部非零输入**（验证输入虚部被忽略、输出置 0）、`TC_3M_*` 3M 专项 4 条（交替/极值填充下虚部抵消、int8 极值、对角虚部置 0）。
- 非 uplo 三角与调用前逐位一致（bit-exact 不写入）。
- 仓内参考路径（SIMT Netlib 同序 kernel 与 golden 生成器共享同调理）保证兜底路径与 CBLAS 同序一致。

## 4.2 性能标准

- 协议（任务书 §3.3，kernel-to-kernel 正式判据）：NPU 侧 `msprof op` 采集单算子 kernel `Task Duration(us)`（按 `--gtest_filter=*TC_PF*` 逐 case，warmup 5 次不计入 + 后续 10 次有效样本算术平均，非 SUCCESS 样本剔除并说明），除以 1000 转 ms 与 `gpu_baseline.csv` 的 `gpu_ms`（GPU cuBLAS `cublasCherk3mEx` kernel-only，nsys 采集）对比。
- 判据：单 case NPU ≤ `gpu_ms` 的 2.5 倍（加速比 ≥ 0.4×），且全部性能 case 加速比算术平均 ≥ 0.5×，两者同时满足。
- 性能子集 200 条按 dtype × 规模档分层；本包性能用例全部为 ACL_COMPLEX64。
- int8 补充：Atype=ACL_INT8 性能用例沿用 ACL_COMPLEX64 的 `gpu_ms` 基线对比，采集方式相同。
- 快速回归：GTest `[PERF] case=<name> ... avg_us=<value>`（aclrtEvent 夹连续 10 次下发）由 `verify_performance.py` 解析，仅作趋势判断，不作为正式验收依据。
- 典型锚点（官方 gpu_baseline.csv）：`cherk3m_ex-perf-075`（LOWER/C/4096/4096/C64）gpu_ms=9.93ms → 单 case 上限 24.8ms；`cherk3m_ex-perf-074`（UPPER/C/3999/3999）gpu_ms=9.39ms → 23.5ms。950PR Cube 3M 三积按 Csyrk3mEx 同族管线实测后回填。
- 实测：按 §3.3.1 协议在 950PR 逐 case 采集后回填自测报告（含 200 条性能 case 与 int8 补充）。

## 4.3 兼容性分析

新算子（`include/cann_ops_blas.h` 新增声明），无历史版本兼容问题；不修改仓内既有算子（非 Ex `cherk` 仅作参考，共享层只读引用）；与同为新增的 `aclblasCherkEx` 相互独立、不依赖其接口；不影响其他产品线。

## 4.4 测试与工程交付

- 测试工程：`test/herk/cherk3m_ex/arch35/`（GTest CSV 驱动，加载任务包 `test_cases/cherk3m_ex_test.csv`），在 `test/herk/` 下新建 `cherk3m_ex` 子目录并注册 `--ops=cherk3m_ex --soc=ascend950` 构建目标；任务包提供 `verify_accuracy.py` / `verify_performance.py` / `gen_csv.py` / `gpu_baseline.csv`，数量口径 1000 精度 + 200 性能（small ≤64KiB / medium ≤2MiB / large >2MiB 分层，含 `TC_3M_*` 4 条专项）。
- 精度 golden：CBLAS `cblas_cherk`（§4.1 口径）；实部/虚部分别比对。
- 算子 README：`blas/herk/README.md` 新增 `aclblasCherk3mEx` 章节（产品支持表 950PR：支持；参数/约束/边界/精度/调用示例），按 ops-blas 仓规范。
- 编译与测试入口：`bash build.sh --ops=cherk3m_ex --soc=ascend950`。
- 代码分支：`zhanghua145/ops-blas`（个人仓邀请 `Ascend-CANN` 为开发者），验收通过后 PR 合入 `blas/herk/arch35/` 与 `test/herk/cherk3m_ex/arch35/`；私仓 `task_submission/` 按任务书 §4 目录结构提交自验证材料。

# 5. 参考实现与验收风险

| 项 | 结论 |
| --- | --- |
| 实现完成度 | 设计定稿；架构主体（三阶段 3M Hermitian 管线 + 小形状单发射族 + 参考兜底）沿 Csyrk3mEx 同族架构落地开发分支，Hermitian 分解（P2=(Xr+Xi)(Xr−Xi)^T、Im=P2−P0+P1、对角置 0）与 OP_T 拒绝为合入前对齐项 |
| 精度 | 按任务书 §3.2 阈值以 CBLAS golden 判定，实测随自验执行回填（1000 条，含 TC_3M 专项与特殊值） |
| 性能 | 按任务书 §3.3 kernel-to-kernel 协议实测回填（200 条 + int8 补充），0.4×/0.5× 双门槛 |
| 任务书对齐项 | ①接口签名以任务书 §2.3 为准（新增声明，`aclblasComplex* C`、参数名 Atype/Ctype）；②Atype/Ctype 支持集与 `NOT_SUPPORTED` 错误码；③**OP_T 返回 `INVALID_ENUM`**（定义域收窄）；④实数 alpha/beta（非复数）；⑤3M 三积单 launch（禁止三次 GEMM 拼接）；⑥不得依赖未合入的 `aclblasCherkEx`；⑦对角虚部置 0 |
| 风险 | FP16 hi/lo 补偿平面在超大 k 下对角实部精度依赖行和表恢复（已设计兜底）；int8 性能沿用 C64 基线（任务书允许）；3M 与非 3M 舍入差异需在自测报告中按阈值说明 |
