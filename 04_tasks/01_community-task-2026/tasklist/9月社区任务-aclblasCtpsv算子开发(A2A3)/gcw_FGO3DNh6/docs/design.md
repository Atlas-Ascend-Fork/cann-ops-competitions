# 【社区任务】aclblasCtpsv 算子设计文档（Atlas A2/A3）

| 版本 | 日期 | 作者团队 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-10-10 | gcw_FGO3DNh6 | 初版设计文档，提交评审 |

- **任务编号**：9 月社区任务（A2/A3），任务截止 2026/11/01
- **适配硬件**：Atlas 800I/T A2（910B3，性能测试设备）/ Atlas A3 系列（同 arch22）
- **CANN 版本**：9.1.0
- **开发语言**：Ascend C（kernel 直调，基于 ops-blas 工程框架）
- **目标仓库**：https://gitcode.com/cann/ops-blas
- **目标代码目录**：`blas/tpsv/arch22/`（新增 `ctpsv_*` 四个文件）
- **目标测试目录**：`test/tpsv/ctpsv/arch22/`（复用现有 `ctpsv_param.h` / `ctpsv_golden.h`）

> 本 PR 仅提交设计文档；代码与自测在后续阶段提交 `cann/ops-blas`。

---

## 需求背景（required）

### 需求来源

任务来源为 9 月社区任务 `aclblasCtpsv`（Atlas A2/A3）。目标是在昇腾 NPU（Atlas A2 / A3，arch22 架构）上基于 Ascend C 实现单精度复数（COMPLEX64）三角打包求解算子，原位求解 `op(A) * x = b`（入口 `x` 存右端项 `b`，出口原地覆写为解向量 `x`），对齐 cuBLAS `cublasCtpsv` 与 Netlib `ctpsv` 的接口、packed 存储、转置/共轭语义与原地语义。验收通过后合入昇腾算子开源仓 `cann/ops-blas`。

### 背景介绍

#### 算子定位与数学定义

TPSV 是 BLAS Level-2 的三角 packed 求解算子。给定 n×n 三角矩阵 `A` 的 packed 存储 `AP`（列主序压缩、无前导维，长度 `n(n+1)/2`），原位求解：

```
op(A) * x = b,   入口 x = b, 出口 x = 解
op(A) ∈ { A (OP_N), Aᵀ (OP_T), Aᴴ (OP_C) }
```

- `uplo`：`ACLBLAS_UPPER` / `ACLBLAS_LOWER` —— 取 `AP` 打包的上/下三角。
- `trans`：`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` —— `op(A)` 分别为 `A`、`Aᵀ`、`Aᴴ`（共轭转置；`OP_C` 必须对系数取共轭后再转置）。
- `diag`：`ACLBLAS_NON_UNIT` 读取对角元素参与求解；`ACLBLAS_UNIT` 对角视为 `1+0j`，**不读取** `AP` 对角位。
- `x` 长度为 n，`incx` 可为负（负步长从末尾反向取值）。
- 复数乘加按 `(ar*br − ai*bi, ar*bi + ai*br)` 展开；复数除法采用 Smith 算法（与 libgcc `__divsc3`、参考 BLAS golden 一致）。

#### packed 布局与索引（含勘误交叉验证）

packed 存储为**列主序压缩**：同一列的连续元素在 `AP` 中物理连续。这是本设计的关键——无论 `uplo`/`trans` 如何组合，逐步求解所需的一整段系数都恰好是 `AP` 的某一**连续列切片**，可用 `DataCopy` 连续搬入再向量化计算，避免按行 gather 造成的非连续访存。

**索引统一形式（0 基，列主序）**：

| `uplo` | 元素 A(i,j) 位置 |
| --- | --- |
| UPPER（i ≤ j） | `AP[i + j*(j+1)/2]` |
| LOWER（i ≥ j） | `AP[i + (2*n−j−1)*j/2]` |

该式经三处交叉验证，结论一致、无误：

1. **Netlib `ctpsv.f`**：`KK` 指针推进 `KK = KK − J`（UPPER 回代）/ `KK = KK + (N−J+1)`（LOWER 前向），对角位于每列 `colStart + k`，与上式完全等价。
2. **仓内 arch35 参考实现** `blas/tpsv/arch35/ctpsv_kernel_utils.h`：`CtpsvPackedUpperIdx = i + j*(j+1)/2`、`CtpsvPackedLowerIdx = i + (2*n−j−1)*j/2`。
3. **仓内测试 golden** `test/tpsv/ctpsv/ctpsv_golden.h`：`CtpsvPackedUpperIdxCpu` / `CtpsvPackedLowerIdxCpu` 同上。

> **具体数值反例（确认任务书索引正确、无需勘误）**：取 n=3。
> - UPPER 对角 `A(2,2)`：上式 `AP[2 + 2*3/2] = AP[5]`；packed 总长 `3*4/2 = 6`，末元下标恰为 5，合法上界内。
> - LOWER 对角 `A(2,2)`：上式 `AP[2 + (6−2−1)*2/2] = AP[2 + 3] = AP[5]`，与 UPPER 同址（两三角共享对角线），一致。
>
> 注意：历史上 `aclblasStpsv`/`aclblasCtpmv` 任务书曾把 LOWER 写成 `AP[i + (2n−j+1)j/2]` 或末尾多 `+i`（越界或错位），本任务书 README 给出的 `AP[i + (2n−j−1)j/2]` 经上述三处核验**正确**，实现按此式、并以 cblas 单标杆逐元素验证。

#### 对标接口与现状分析

| 维度 | 现状 | 本任务动作 |
| --- | --- | --- |
| 接口声明 | `aclblasCtpsv` **已存在于** `include/cann_ops_blas.h:172`，签名与任务书一致 | **复用，不改签名、不新增声明** |
| arch22 实数路径 | `blas/tpsv/arch22/stpsv_*`（单精度实数 tpsv）已合入，是 arch22 的直接结构模板 | 沿用其单 AIV / x 常驻 UB / 列窗口游走 / PipeBarrier / 对齐掩码范式，改为复数 |
| arch35 复数路径 | `blas/tpsv/arch35/ctpsv_*`（含 SIMT 多核分块）已合入 | 借鉴其复数乘/除与共轭数值逻辑；**但 arch22 无 SIMT，不移植多核分块**，改用单 AIV 向量化游走 |
| 测试工程 | `test/tpsv/ctpsv/{ctpsv_param.h, ctpsv_golden.h, verify.h}` 已存在（arch35 使用） | arch22 测试目录复用上述公共件，仅新增 wrapper / CMakeLists / CSV 放置 |

#### 功能分析

- 支持 COMPLEX64 单精度复数、`uplo×trans×diag` 共 12 组枚举全组合、正负 `incx`（含负步长）、`n=0` 合法 no-op。
- `n>0` 时 `AP` 与 `x` 不允许内存重叠；`AP` 与 `x` 均 Device 内存，`handle` 绑定 stream 直调。
- 求解类对奇异/近奇异矩阵无定义，不做奇异性检测，也不替换为其他求解器（与 Netlib/cuBLAS 一致）。
- 确定性不要求（浮点累加顺序不保证逐位一致）。

### 需求分析（required）

#### 需求描述（接口原型）

```cpp
// include/cann_ops_blas.h:172（已存在，禁止改动签名）
aclblasStatus_t aclblasCtpsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* AP,   // device, 只读, 长度 n(n+1)/2
    aclblasComplex* x,          // device, 原地输入 b / 输出解, 长度 1+(n-1)*|incx|
    int incx);
```

`aclblasComplex` 为两个连续 FP32 分量（实部、虚部），在 GM 中以 2 个 float32 紧邻存储（见 `arch35/ctpsv_kernel.cpp` 的 `apGM.SetGlobalBuffer(..., apCount*2U)`）。

参数说明与异常行为见 §详细设计「参数与异常行为矩阵」。

#### 需求拆解

1. **复用接口声明**：`include/cann_ops_blas.h:172` 的 `aclblasCtpsv` 已存在且签名正确，直接复用，不新增、不改签名。
2. **全集枚举 + 索引语义**：实现 `uplo×trans×diag` 12 组组合，正确完成 packed 下标换算、`trans=C` 共轭、正负 `incx` 与原地写回。
3. **异常与边界**：`n=0` quick return；非法枚举 / `n<0` / `incx=0` / `n>0` 时空指针返回规定状态码且不下发 kernel；`handle` 空返回 `HANDLE_IS_NULLPTR`。
4. **精度**：以 cblas（Netlib `ctpsv`）单标杆 golden 对输出 `x` 全长度、实部/虚部分量做混合容差校验。
5. **性能**：按任务书 §3.3 五个典型坐标验收，msprof 采集平均单次 kernel 耗时，先 warmup 后有效采样 ≥10 次。
6. **自测**：回放随任务的完整 1200 条用例（1000 精度 + 200 性能/内存），覆盖边界、负向与特殊值（Inf/NaN、UNIT 对角哨兵、负步长、共轭）。

---

## 详细设计（required）

### 算子分析

#### 数学公式（统一形式）

令逻辑索引 `q(i)` 为 `x` 中第 i 个逻辑元素的物理下标：

```
incx > 0 : q(i) = i * |incx|
incx < 0 : q(i) = (n-1) * |incx| - i * |incx|
```

求解统一为逐步递推（依赖方向串行）：

```
x[q(k)] = ( b[q(k)] − Σ_j op(A)(k,j)·x[q(j)] ) / d(k)
```

其中求和只累加依赖方向上**已解出**的分量（前向代入 j<k，回代 j>k）；`d(k)` 在 `diag=UNIT` 时为 `1+0j`，否则为 `op(A)(k,k)`。对 `op(A)=Aᴴ`，读写 `AP` 系数时对其虚部取反（共轭）。

#### packed 索引统一形式

见「背景介绍·packed 布局与索引」。核心结论：

- UPPER 第 k 列基址 `colStart(k) = k*(k+1)/2`，列内元素 `[0 .. k]`（对角在 `+k`）。
- LOWER 第 k 列基址 `colStart(k) = k*(2n−k−1)/2`，列内元素 `[k .. n−1]`（对角在 `+0` 即列首）。
- 任一列除对角外的连续切片，正是每步所需的全部系数。

#### 数学与依赖方向

依据 `uplo`/`trans`，可归并为两类等价形式，二者都只触碰 `AP` 的**同一列连续切片**与 `x` 的同一连续区间，故单一循环体可覆盖全部 8 种 `uplo×trans×diag` 组合：

- **`trans == N`（right-looking 秩-1 更新）**：先解出 `x[k]`，再把其影响散布到剩余分量 `x[rem] -= AP(col)[off] * x[k]`。
- **`trans == T/C`（left-looking 点积）**：先把已解分量与 `AP` 列切片点积累加，再除以对角 `x[k] = (x[k] − Σ) / d`。

方向判定（与 `stpsv_kernel.cpp` 的 `forward = (lower == update)` 一致）：

| `uplo` | `trans` | 求解方向 | 形式 | 第 k 步系数（AP 第 k 列 off-diagonal） | x 区间 |
| --- | --- | --- | --- | --- | --- |
| LOWER | N | 前向（k 递增） | right-looking | `AP[colStart+k+1 .. ]` | `[k+1 .. n-1]` |
| UPPER | N | 回代（k 递减） | right-looking | `AP[colStart .. colStart+k-1]` | `[0 .. k-1]` |
| UPPER | T/C | 前向（k 递增） | left-looking | `AP[colStart .. colStart+k-1]` | `[0 .. k-1]` |
| LOWER | T/C | 回代（k 递减） | left-looking | `AP[colStart+k+1 .. ]` | `[k+1 .. n-1]` |

```mermaid
flowchart TD
    A[入口: 校验参数 / n==0 quick return] --> B[Load x -> UB 常驻]
    B --> C{逐步 s=0..n-1}
    C --> D[k = forward ? s : n-1-s]
    D --> E[取 AP 第 k 列连续切片 colStart..]
    E --> F{trans == N ?}
    F -- 是 --> G[解出 x[k]=x[k]/d; x[rem] -= AP*x[k] 秩1更新]
    F -- 否 --> H[点积 Σ AP*x[finished]; x[k]=(x[k]-Σ)/d; C时共轭AP]
    G --> C
    H --> C
    C --> I[Store x UB -> GM]
```

#### 支持数据类型

- 输入/输出：`ACLBLAS_COMPLEX64`（单精度复数，实部/虚部各 FP32）。
- 不支持 FP16/BF16/双精度；不支持 alpha/beta、前导维、broadcast。

#### 支持形状

- `n >= 0`；`n=0` 合法 no-op。逻辑 `x` 长度 n，物理长度 `1+(n-1)*|incx|`。
- `AP` 长度 `n(n+1)/2`（COMPLEX64，即 `2 * n(n+1)/2` 个 float32）。本设计将解向量 `x` 常驻 UB，故对 `n` 设上限 `ACLBLAS_CTPSV_MAX_N`（见下），超出返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（与 arch22 `stpsv` 同策略，且覆盖任务书全部用例——最大 n=4096）。

#### 参数与异常行为矩阵（校验顺序）

| 校验项 | 条件 | 返回码 | 是否下发 kernel |
| --- | --- | --- | --- |
| handle 空 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 否 |
| 维度非法 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` | 否 |
| 枚举非法 | `uplo/trans/diag` 不在枚举集 | `ACLBLAS_STATUS_INVALID_VALUE` | 否 |
| 步长非法 | `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` | 否 |
| 空指针 | `n>0` 且 `AP==nullptr` 或 `x==nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 否 |
| 阶数超限 | `n > ACLBLAS_CTPSV_MAX_N` | `ACLBLAS_STATUS_NOT_SUPPORTED` | 否 |
| 合法 no-op | `n == 0` | `ACLBLAS_STATUS_SUCCESS` | 否 |
| 正常 | 其余 | `ACLBLAS_STATUS_SUCCESS` + 下发 kernel | 是 |

> 返回码口径**跟随任务书 §2.4 与仓内 `stpsv`/`ctpsv_golden.h`**：非法枚举/维度/步长/空指针一律 `INVALID_VALUE`，handle 空为 `HANDLE_IS_NULLPTR`；**不**使用 `INVALID_ENUM`。`n==0` 即便 handle 空也优先返回 `HANDLE_IS_NULLPTR`（与 `stpsv_host.cpp` 一致：先校验 handle 再走 no-op 捷径）。

### 算子实现

工程结构与文件清单（新增于 `blas/tpsv/arch22/`）：

```
blas/tpsv/arch22/
├── ctpsv_host.cpp        # aclblasCtpsv() 入口：参数校验 + 填充 CtpsvTilingData + launch
├── ctpsv_kernel.cpp      # 单 AIV kernel：gather / solve(列窗口游走) / scatter
├── ctpsv_kernel.h        # kernel 启动桩 void ctpsv_kernel_do(...)
└── ctpsv_tiling_data.h   # CtpsvTilingData + ACLBLAS_CTPSV_MAX_N
```

#### 3.2.1 Host 侧设计

**核心难点 → 归一化**：三角求解存在严格数据依赖，第 k 步必须用依赖方向上已解出的分量，单条右端向量的依赖链不能跨核拆分；因此**主路径单 AIV core**（`<<<1, nullptr, stream>>>`），并行度来自行内向量化与 MTE/Vector 双缓冲重叠，而非多核切分。

**执行路径与分块**：

1. 校验 `handle` → `uplo/trans/diag` 枚举 → `n` → `incx` → `n>0` 时 `AP/x` 指针；错误直接返回状态码，不下发 kernel（`stpsv_host.cpp` 同款）。
2. `n==0` 直接返回 `SUCCESS`，不申请 workspace、不下发 kernel。
3. 计算 `x` 物理长度 `(n>0 ? (n-1)*|incx|+1 : 1)`、AP 长度，填充 `CtpsvTilingData{ap, x, n, uplo, trans, diag, incx}`。
4. **launch 调度**：单 AIV，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`（arch22 无 SIMT，不能用 `asc_vf_call`）。
5. **Tiling**：`x` 常驻 UB；`AP` 按列流式搬入 UB（双缓冲队列）。由于单 AIV 且 x 全程驻留，host 侧无需复杂 tiling 计算，仅把枚举与 `n/incx` 透传 kernel。

**workspace**：无额外 device workspace；解向量常驻 UB，操作数直接来自入参 GM 指针。

**TilingData 结构**（`ctpsv_tiling_data.h`，与 `StpsvTilingData` 同构，字段加 `complex` 语义）：

```cpp
constexpr uint32_t ACLBLAS_CTPSV_MAX_N = 8192U;  // 覆盖用例上限 4096 并留 2x 余量
struct CtpsvTilingData {
    uint64_t ap;     // device 指针：packed 三角矩阵（长度 n(n+1)/2 个复数）
    uint64_t x;      // device 指针：右端项/解向量
    uint32_t n;      // 阶数
    uint32_t uplo;   // ACLBLAS_UPPER / LOWER
    uint32_t trans;  // ACLBLAS_OP_N / T / C
    uint32_t diag;   // ACLBLAS_NON_UNIT / UNIT
    int64_t  incx;   // 非零步长（可负）
};
```

> **UB 预算推导（910B3 每 AIV UB = 196608 B）**：
> 足迹 ≈ `AlignUp32((n+8)*8)`（xBuf）+ `2*tile*8`（aQueue 双缓冲）+ `tile*8`（prodBuf）+ `tile*8`（workBuf）。取 `tile=4096`：队列合计 `131072 B`；留给 xBuf `65536 B` → `(n+8)*8 ≤ 65536` → `n ≤ 8184`。
> 故 `ACLBLAS_CTPSV_MAX_N = 8192` 在 `tile=4096` 下恰好满足，任务书最大用例 `n=4096` 余量充足；更大 `n` 由 host 拒绝为 `NOT_SUPPORTED`（风险项 R1）。

**tilingkey**：沿用 `stpsv` arch22 范式的**单 kernel + 运行时分支**（`lower/update/unitDiag/forward` 四布尔在 `Init` 中一次性算出），不引入 12 个编译期 kernel，亦不使用 tilingkey（相比 arch35 `ctpsv` 的 12-kernel dispatch 更简洁，且 arch22 单 AIV 下分支开销可忽略）。

#### 3.2.2 Kernel 侧设计

**索引统一形式**（与 `stpsv_kernel.cpp` 一致，元素类型换为复数）：

```
ColStart(k): LOWER -> k*(2n-k-1)/2 ; UPPER -> k*(k+1)/2
XPhys(i):    incx>=0 ? i*|incx| : (n-1-i)*|incx|
每步几何（forward = (lower == (trans==N))）:
  if lower : start = k+1 ; len = n-1-k ; aOff = colStart + k + 1
  else     : start = 0   ; len = k     ; aOff = colStart
```

**各阶段伪代码**：

1. **Gather（仅 incx≠1 时逐元素，incx==1 时整块 DMA）**：按 `XPhys(i)` 把逻辑 `x[0..n-1]` 搬入常驻 UB（复数，实部/虚部平面存放）。`x` 全程驻留，仅首尾各一次搬运；计算期间不再访问 `x` 的 GM 端。
2. **Solve（列窗口游走，每步 s=0..n-1）**：
   - 取 `k = forward ? s : n-1-s`，算 `colStart` 与本步切片 `[aOff, aOff+len)`。
   - **复数乘法** `Cmuld(a,b)=(ar*br−ai*bi, ar*bi+ai*br)`；**复数除法** `Cdivd` 用 Smith 算法（与 `ctpsv_kernel_utils.h` 的 `CTPSV_DEFINE_CDIV` 逐字一致，确保与 golden `__divsc3` 同结果）。
   - `trans==N`（右向秩-1 更新）：先 `x[k] /= d(k)`（`diag=NON_UNIT` 时读 `AP[colStart+k]` 做 Smith 除；`UNIT` 时 `d=1+0j` 跳过读）；再用广播标量 `x[k]` 对 `x[start..start+len)` 做复数 `Axpy`（每复数 4 个 FP32 乘加）。
   - `trans==T/C`（左向点积）：在区间 `[start, start+len)` 上对 `AP` 切片与已解 `x` 做复数点积（`trans==C` 时先对 `AP` 虚部取反），结果 `s`；`x[k] = (x[k] − s)/d(k)`。
   - `AP` 切片以 `DataCopyPad` 连续搬入 UB 队列（双缓冲），与 Vector 计算流水重叠。
3. **Scatter**：按 `XPhys(i)` 把解写回 GM（incx==1 整块 DMA；否则逐元素 `SetValue`），逻辑长度以外与 stride gap 保持不变。

**精度策略**：

- 运算序列与 Netlib `ctpsv.f` 完全一致（右向为顺序 `x[rem] −= a*x[k]`，左向为顺序 `s += a*x[j]`）；两者均为**顺序左到右累加**，与参考 BLAS 的累加顺序相同 → 累加舍入一致，ULP 差主要来自复数乘/除的单次舍入（≤1 ULP）。
- 复数除法严格复用 Smith 算法（`__divsc3` 等价）→ 对角除法与 golden 一致（≤1 ULP）。
- `trans==C` 共轭为精确取反（虚部 `−=0`），不引入误差。
- 因此 `max_abs_error` 远优于 `32*ULP`（含规约的极端 case 亦在 `2*ULP` 内），可稳定满足任务书 §3.2 阈值。

**边界与退化**：

- **极小 n（n ≤ 3）**：整个 packed 三角 ≤ 1 个 32B 块，走 `ProcessTiny` 纯标量路径（解向量驻标量寄存器、AP 直读 GM、无 DMA/无 UB/无硬事件），消除 launch 固定开销对小 n 的主导（参考 `stpsv_kernel.cpp` 的 `TINY_MAX_N`）。
- **对齐掩码（LOWER 路径关键）**：LOWER 切片起点 `k+1` 仅每 4 个复数（32B）对齐一次。将窗口左扩到上一 32B 边界，越界 `rem` 个复数 lane 用「**置零乘积**」（而非置零 AP 操作数）掩码——避免 `0*Inf=NaN`、保留 `x−0=x` 的位精确恒等（含 Inf/NaN），消除 LOWER 路径的逐元素标量回退（stpsv 实测 LOWER 曾因此慢 ~40%）。
- **PipeBarrier 必要性**：Vector 指令读取由前一条 Vector 指令写入的 UB 缓冲时需 `PipeBarrier<PIPE_V>()`（Mul→Duplicate 的 WAW、Duplicate→ReduceSum 的 RAW）。本工具链向量管道不自动排序，遗漏会产生静默错误结果——在点积窗口的 `Mul/Duplicate/ReduceSum` 间显式插入。
- **负 incx / UNIT 对角 / Inf/NaN**：`XPhys` 处理负步长；`UNIT` 不形成对角存储地址（该位可为任意值甚至 NaN）；Inf/NaN 在良态对角下传播良定义，须与 golden 一致（由完整 CSV 回归，不可静默过滤）。

### 支持硬件

| 支持的芯片版本 | arch | 状态 |
| --- | --- | --- |
| Atlas 800I/T A2（910B3） | arch22 | 任务书目标（性能测试设备） |
| Atlas A3 系列 | arch22 同族路径 | 任务书要求兼容 |

本设计不包含 950 `arch35` 路径（arch35 已另有 `ctpsv` SIMT 实现），亦不把其他架构目录实现当作 A2/A3 结果。

### 算子约束限制

- 不涉及 broadcast、alpha/beta、前导维或非连续 view；packed `AP` 与 `incx` 语义必须保留。
- `n<0`、`incx=0`、非法枚举、`n>0` 时 `AP/x` 为空均为非法参数；`handle` 空返回 `HANDLE_IS_NULLPTR`。
- `diag=NON_UNIT` 的被引用对角必须非零（golden 同矩阵回代，测试工程对被引用对角加 `boost=max(5,n)` 保证良态）；求解类对奇异矩阵无定义，不做奇异性/近奇异检测，也不替换为其他求解器。
- `diag=UNIT` 时**不得读取** `AP` 对角位置（该存储可为任意值，含 Inf/NaN）。
- 调用方须保持 `AP/x` 与 handle stream 生命周期；`AP` 与 `x` 不允许内存重叠；读回设备结果前须同步该流。
- `n > ACLBLAS_CTPSV_MAX_N (=8192)` 视为不支持，返回 `NOT_SUPPORTED`。

---

## 可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | COMPLEX64 实部/虚部分别判定：`\|actual−golden\| ≤ atol + rtol·\|golden\|`，`rtol = atol = 2^-13 (1.22e-4)`，`required_matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2 或 32*ULP`（含规约场景可放宽至 2ULP） | 任务书 §3.2、生态算子开源精度标准 |
| 性能 | 任务书 §3.3 五个典型坐标的「平均单次 kernel 耗时（us）」：252.2 / 573.9 / 1228 / 1211 / 7327；先 warmup 后有效采样 ≥10 次取平均（msprof `OpBasicInfo.csv` 的 `Task Duration`）；达标条件 NPU ≤ 标杆（即 GPU/0.8 = 1.25×GPU） | 任务书 §3.3、随任务 `gpu_baseline.csv` |

**性能标杆表（任务书 §3.3，A2/A3 达标耗时）**

| case | n | uplo | trans | diag | incx | 达标耗时（us） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | UPPER | N | NON_UNIT | 1 | 252.2 |
| 2 | 512 | LOWER | T | NON_UNIT | 1 | 573.9 |
| 3 | 1024 | UPPER | C | NON_UNIT | 1 | 1228 |
| 4 | 2048 | LOWER | N | UNIT | 1 | 1211 |
| 5 | 4096 | UPPER | T | NON_UNIT | 1 | 7327 |

**达标估算表（临界路径 + 访存量 + launch 开销，假设输入、非实测）**

tpsv 为 O(n²) 顺序求解，关键路径受限于 `AP` 的流式读取（x 常驻 UB，无回写流量）。单 AIV 双缓冲下 kernel 时间 ≈ `max(AP_read_time, compute_time)`：

| case | n | AP 字节（复数） | 设有效带宽 50 GB/s 的 AP 读时 (ms) | 本设计估算 kernel 时 (ms, 假设) | 达标 (ms) | 余量倍数 (达标/估算) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 0.26 MB | 0.005 | ~0.03 | 0.252 | ~8× |
| 2 | 512 | 1.05 MB | 0.021 | ~0.08 | 0.574 | ~7× |
| 3 | 1024 | 4.19 MB | 0.084 | ~0.18 | 1.228 | ~7× |
| 4 | 2048 | 16.8 MB | 0.34 | ~0.45 | 1.211 | ~2.7× |
| 5 | 4096 | 67.2 MB | 1.34 | ~1.6 | 7.327 | ~4.6× |

- 估算假设：910B3 HBM 峰值 ~1.6 TB/s；tpsv 列流式访问顺序性弱，保守取 **有效带宽 50 GB/s**（已留 30× 以上余量）；计算量 O(n²) 复数 MAC 远小于访存，compute 不构成瓶颈；launch 固定开销（~几 µs）相对达标值可忽略。
- **结论**：在最保守带宽假设下，五档典型 case 估算耗时均显著低于达标线（余量 2.7×~8×）。即使 arch22 单 AIV 利用率受限，任务书标杆本身即 GPU/0.8（1.25×GPU），而 tpsv 在 GPU 上同样未被深度优化，故本设计可行。**所有带宽/launch 数值均为假设输入、非实测**，实现阶段以 msprof 实测为准。

### 测试设计

**框架扩展点**：复用现有 `test/tpsv/ctpsv/` 的 `ctpsv_param.h`（CSV 行→`CtpsvParam` 解析，已含 `ap_fill/x_fill` 可选列）、`ctpsv_golden.h`（cblas `ctpsv` golden 封装）、`verify.h`（`applyMixedTolerance`）。arch22 仅新增：

```
test/tpsv/ctpsv/arch22/
├── ctpsv_npu_wrapper.h   # 申请 device 内存并调用 aclblasCtpsv（镜像 arch35 wrapper）
├── ctpsv_test.cpp        # 数据构造/调用/比对（复用 param+golden）
├── ctpsv_test.csv        # 复用任务随附 1200 条用例
└── CMakeLists.txt
```

**用例集**（任务随附 `test_cases/tpsv_test.csv`，1200 条，固定种子可复现）：

| 类别 | 前缀 | 条数 | 覆盖 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 24 | 12 组枚举全组合 × 小尺寸 (8,32) |
| L1 尺寸 | TC_SQ | 92 | 23 种尺寸（含奇数/边界/非对齐）× 轮转枚举 |
| L2 步长 | TC_INC | 24 | incx ∈ ±1/±2/±3（含负步长） |
| L5 填充 | TC_FL | 8 | AP/x 均匀/交替/极端/零/Inf/NaN |
| L5b 覆盖 | TC_CV | 96 | 中尺寸 × 12 组枚举全组合 |
| L6 边界 | TC_ED | 12 | 零维/空指针/非法枚举/负维度/零步长；另含独立 `null handle` 用例 |
| EX 扩展 | TC_EX | 744 | 尺寸池 × 枚举 × 步长确定性采样 |
| PF 性能 | TC_PF | 200 | 5 条任务书典型 case + 尺寸/阶数扫描（纯连续 incx=1） |

**负向对照**：`TC_ED` 覆盖任务书 §2.4 全部异常行为（handle 空→`HANDLE_IS_NULLPTR`；非法枚举/负 n/incx=0/null AP/x→`INVALID_VALUE`；n=0→`SUCCESS`）。`null handle` 由独立 `TEST(CtpsvNullHandle)` 覆盖（CSV 固定 handle 有效）。

**特殊值与补充用例**：
- Inf/NaN 输入：`TC_FL` 的 `VALUE_NORM_INF`/`VALUE_NORM_NAN`，传播良定义、须与 golden 一致。
- UNIT 对角哨兵：对角位不引用（可为任意值）；测试工程对非对角按 `1/n` 缩放保证对角=1 主导（见 `test/tpsv/ctpsv/README.md`）。
- `trans=C` 共轭：12 组枚举中 OP_C 分支单独保留日志，防止被随机 case 淹没。
- masked tail / 非对齐长度：`TC_SQ`/`TC_EX` 含非 2 的幂与 ±1 偏移尺寸，覆盖列窗口尾部 mask。

**可复现口径**：`gen_csv.py --seed 20260823` 输出逐字节一致；精度用例 1000 条、性能 200 条；精度段自动排除 `TC_PF_*`；`verify_accuracy.py` 编译 GTest 并执行、解析逐条 PASS/FAIL。

**交付件**：设计/代码/自测三阶段共用同一接口与验收口径；自测报告含精度（逐元素阈值 + matched ratio + max abs error，附截图）、性能（msprof kernel 时间，warmup+≥10 采样，附截图）、内存（常驻 UB 预算表）三类数据。

### 兼容性分析

- 接口声明置于公共 `include/cann_ops_blas.h`（已存在，复用），实现仅置于 `blas/tpsv/arch22/`；不建立产品私有 API。
- 评测路径不得调用官方 whole-op、Torch/Torch-NPU 等价算子、CPU/cblas 参考或其他后端实现兜底；golden 与 GPU 基线仅用于对照验证，非运行时 fallback。
- 不影响同目录 `stpsv`（实数）与 arch35 `ctpsv`（SIMT）既有实现；新增文件命名 `ctpsv_*`，不与现有 `stpsv_*` 冲突。

### 风险与待确认项

| # | 事项 | 影响 | 处置 |
| --- | --- | --- | --- |
| R1 | `ACLBLAS_CTPSV_MAX_N=8192` 上限（UB 预算决定），超出返回 `NOT_SUPPORTED` | 任务书未明示上限，仅要求 n≥0；但用例最大 n=4096 已覆盖 | 沿用 `stpsv` 同策略；如 admin 要求支持任意 n，需改 x 分块驻留（多趟），列为后续优化 |
| R2 | 精度阈值口径：任务书 §3.2 用 `rtol=atol=2^-13`；而 `test/tpsv/ctpsv/README.md` 记录的 ctpsv 测试阈值为 `2^-16/2^-10` | 若 arch22 测试直接继承现有 ctpsv verify 阈值，会严于任务书 | 以任务书 §3.2（`2^-13`）为 A2/A3 绑定验收；需确认 arch22 测试 `verify.h` 阈值按任务书配置，必要时显式覆盖 |
| R3 | 性能达标口径：任务书 §3.3 注明「平均的标杆时延 / 测试时延 > 1.1」有歧义（逐例等权 vs 总和等权） | tpsv 为逐 case 独立达标，两种口径在逐 case 下等价 | 按逐 case 满足 `NPU_avg ≤ 标杆` 执行；若 admin 要求总和口径再补算 |
| R4 | 单 AIV 顺序求解，峰值利用率受限于 AP 流式访存；超大 n 性能可能不及预期 | 任务书最大 n=4096 在最保守带宽假设下余量 4.6×，风险低 | 实现阶段以 msprof 实测；必要时引入 UB 内 x 分块/双缓冲调优 |
| R5 | `AP` 与 `x` 内存重叠未定义；任务书要求不重叠 | 重叠时结果未定义 | host 不检测重叠（与 cuBLAS/Netlib 一致），文档明确约束由调用方保证 |
| R6 | GitCode 账户/团队名 | 设计文档落盘目录 `tasklist/.../{TeamName}/docs/` 与 fork 推送需明确账户 | 已确认 GitCode 账户/团队名为 `gcw_FGO3DNh6`；设计文档落盘于 `tasklist/.../gcw_FGO3DNh6/docs/`，fork 推送目标 `gcw_FGO3DNh6/cann-ops-competitions`，状态已消解 |
