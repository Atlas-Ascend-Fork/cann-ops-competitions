# 【CANN社区任务】aclblasCherk3mEx 算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务 `aclblasCherk3mEx`（Atlas 950PR / arch35）。目标是在 ops-blas 仓新增
句柄式 BLAS 接口 `aclblasCherk3mEx`，对标 cuBLAS `cublasCherk3mEx`，以
Hermitian 专用 Gauss 3M 分解完成 `C = alpha·op(A)·op(A)^H + beta·C`。

### 背景介绍

- 仓内非 Ex 的 `aclblasCherk` 已提供 4M 实数乘 + AIV 合并的 arch35 路径，
  其三次/四次实数乘通过多次 `gemm_kernel_do` 调用完成，不满足本任务
  “三次实数乘必须在同一次 kernel launch 内完成”的约束。
- 仓内 `aclblasCgemm3m` 已有标准 3M 分解、AIV 平面预处理和单 launch
  三乘积 cube kernel（`gemm_cgemm_products_kernel`）。本算子复用其数学分解与
  kernel，但按 Hermitian 定义强制 `B = X^H`，不能套用 `X·X^T` 的对称分解。
- 本任务新增独立接口，不依赖尚未合入的 `aclblasCherkEx`。

## 需求分析（required）

### 需求描述

实现 `aclblasCherk3mEx`，支持：

| 项 | 范围 |
|---|---|
| Atype | `ACL_COMPLEX64`、`ACL_INT8`（complex-int8，交错 `(int8 real, int8 imag)`） |
| Ctype | `ACL_COMPLEX64` |
| uplo | `ACLBLAS_UPPER`、`ACLBLAS_LOWER` |
| trans | `ACLBLAS_OP_N`、`ACLBLAS_OP_C`；`OP_T` 返回 `INVALID_ENUM` |
| n/k | `n ≥ 0`、`k ≥ 0`；`n=0` no-op，`k=0` 只做 beta 缩放 |
| alpha/beta | FLOAT32 实数，Host 或 Device 指针（须同位置） |

### 需求拆解

1. 新增公开声明 `include/cann_ops_blas.h`。
2. arch35 实现放在 `blas/herk/arch35/`，三次实数乘一次 kernel launch 完成。
3. 只读写 uplo 三角，对角虚部强制为 0；beta=0 不读旧 C；alpha=0/k=0 不引用 A。
4. 参数校验顺序和错误码按任务书 §2.4/§2.5 执行。
5. 在 `test/herk/cherk3m_ex/arch35/` 新增 CSV 驱动 GTest，golden 使用 CPU
   侧非 3M `cblas_cherk`（Netlib），int8 输入精确提升。
6. 精度按 FP32 分量 `rtol=atol=2^-13`、`matched_ratio≥0.99`、`max_abs_error≤1e-2` 判定。

## 详细设计（required）

### 算子分析

令 `X = op(A)`。本算子的乘积是 `X·X^H`（不是 `X·X^T`），标准 3M 分解取
`B = X^H`，即 `Br = Xr^T`、`Bi = -Xi^T`：

```
t1 = Re(X)·Re(X)^T = Xr·Xr^T
t2 = (-Xi)·Xi^T    = -Xi·Xi^T
t3 = (Xr-Xi)·(Xr+Xi)^T
Re = t1 - t2 = Xr·Xr^T + Xi·Xi^T
Im = t3 - t1 - t2 = Xi·Xr^T - Xr·Xi^T
```

`cgemm` 的标准 3M 分解使用同一组公式：A 操作数提供 `ar/ai/ar+ai`，B 操作数
提供 `br/bi/br+bi`，`gemm_cgemm_products_do` 在一次 launch 中输出 `t1/t2/t3`。

### 算子实现

#### host 侧设计

1. **校验顺序**：handle → uplo/trans → n/k → lda/ldc → alpha/beta → Atype/Ctype → A/C 指针。
2. **标量解析**：复用 `CheckPtrLocation` / `ReadAlphaBetaFromDevice`，Host 标量直接解引用，Device 标量 D2H 读取；混合位置拒绝。
3. **退化路径**：`n=0` 直接返回；`alpha=0` 或 `k=0` 且 `beta=1` 直接返回；否则只执行一次 beta-only 三角缩放 kernel。
4. **工作区布局**（512B 对齐）：
   - 六个实平面：A 操作数 `ar/ai/ar+ai` 与 B 操作数 `br/bi/br+bi`，各 `n×k` float；
   - 三个乘积平面 `t1/t2/t3`，各 `ceil16(n)×n` float；
   - int8 路径额外 `rows×cols×8B` 的 complex64 提升缓冲。
   总需求约 `24nk + 12·ceil16(n)·n` 字节（int8 另加 `8nk`），上限 2 GiB。
5. **平面生成**：
   - trans=N：逻辑 `A=X`（transa=N）、`B=A`（transb=C），deinterleave 得到右侧 `Xr/Xi/Xr+Xi`、左侧 `Xr/-Xi/Xr-Xi`。
   - trans=C：逻辑 `A=A^H=X`（transa=C）、`B=A`（transb=N），共轭/转置由 cgemm deinterleave 的内置标志完成。
   - 两条路径均复用 `gemm_cgemm_deinterleave_do`，支持 lda padding 和 padding 区不越界。
6. **三乘融合**：`gemm_cgemm_products_do(..., cgemmThreeProducts=1)`，单个 cube kernel launch 内完成 `t1/t2/t3`。tile 从 `{16,32,64,128}` 中按输出 tile 数与 AIC 核数选择。
7. **Hermitian epilogue**：新增 AIV SIMT kernel，逐元素按 uplo 过滤，计算
   `Re = alpha*(t1-t2) + beta*C.real`、`Im = alpha*(t3-t1-t2) + beta*C.imag`，
   对角虚部写 0；`beta=0` 分支不读旧 C；非 uplo 和 padding 完全不写。

#### kernel 侧设计

- `cherk3m_ex_kernel.cpp` 提供两个 AIV-only kernel：
  1. `cherk3m_ex_int8_to_complex_kernel`：complex-int8 → packed complex64，保留列主序 lda 跨度。
  2. `cherk3m_ex_epilogue_kernel`：带 uplo 掩码、对角置零、alpha/beta 的三角写回。
- 三次实数乘由仓内 `gemm_cgemm_products_kernel` 完成；host 只发射一次
  `gemm_cgemm_products_do`，符合“同一次 kernel launch”约束。

### 支持硬件

| 支持的芯片版本 | 是否支持 |
| --- | --- |
| Ascend 950PR（arch35） | √ |
| Atlas A2/A3（arch22） | × |

## 算子约束限制

- 仅支持上述 dtype 组合；不支持广播、非连续视图、AutoContiguous/ViewCopy。
- 调用方保证 C 的 uplo 三角为 Hermitian、A/C 可读区域不重叠。
- 3M 与直接复数乘的 FP32 舍入不同，结果不要求与 `aclblasCherk` 逐位一致。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | FP32 分量 `rtol=atol=2^-13`，matched_ratio≥0.99，max_abs_error≤1e-2 | 任务书 §3.2 |
| 性能 | `msprof op` kernel Task Duration 对 GPU kernel-only `gpu_ms`：单 case 倍率≥0.4，平均≥0.5 | 任务书 §3.3 |
| 内存 | 不涉及额外内存约束；库工作区上限 2 GiB | 任务书 §3.4 |

## 兼容性分析

新增接口，不修改既有接口签名；`gemm_cgemm_*` 复用为只读依赖，不改变非 Ex
`aclblasCherk` 的行为。低版本 asc-devkit 下通过 `TENSOR_API_OPS` 门控跳过本算子。
