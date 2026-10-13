# aclsparseSpSV 算子设计文档（A2 / A3，arch22）

> 交付路径：`04_tasks/01_community-task-2026/.../design.md`
> 算子：`aclsparseSpSV` — 稀疏三角求解（Sparse Triangular Solve）
> 平台：Atlas A2 / A3（arch22，DAV-2201，AIV），CANN 9.1.0
> 对标：cuSPARSE `cusparseSpSV_*`（bufferSize / analysis / solve / updateMatrix）
> 源码：`sparse/spsv/arch22/`（`spsv.h` / `spsv_tiling_data.h` / `spsv_kernel.h` / `spsv_kernel.cpp` / `spsv_host.cpp`）

---

## 1. 目标与范围

参考 cuSPARSE SpSV，在 A2/A3 上实现稀疏三角求解，支持 **FP32 与 complex64**：

- 四格式输入：CSR / CSC / COO / SLICED_ELL
- LOWER / UPPER、UNIT / NON_UNIT
- opA ∈ {N, T, H}（complex64 的 H 为**真共轭转置**）
- 索引 base 0/1；I32 / I64 索引
- alpha 支持 host 标量与 device 指针（pointerMode）
- 完整生命周期：`createDescr → bufferSize → analysis → solve → updateMatrix → destroyDescr`
- Y 允许与 X 同址（原地求解）；A / X 只读（除 updateMatrix 显式更新 A 的 values）

### 1.1 求解语义

计算 `y` 使 **`op(A) · y = α · x`**，即 `y = α · op(A)⁻¹ · x`。
记 `B = op(A)`，则三角回代作用在 **B** 上：

- `fillB = (opA == N) ? fillA : flip(fillA)` —— 转置会翻转三角侧
- `opA == H` 时对复数取共轭
- 三角形外的条目**跳过**（与 cuSPARSE / `torch.linalg.solve_triangular` 一致）
- UNIT：对角按 1 处理，忽略存储的对角值
- NON_UNIT 且该行无对角：除数为 0 → 传播 IEEE Inf/NaN

---

## 2. 对外接口

```c
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, size_t *bufferSize);

aclsparseStatus_t aclsparseSpSV_analysis(... , void *externalBuffer);
aclsparseStatus_t aclsparseSpSV_solve(...);
aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart);
```

`computeType` ∈ {`ACL_FLOAT`, `ACL_COMPLEX64`}；`alg` 仅 `ACL_SPARSE_SPSV_ALG_DEFAULT`。

---

## 3. 总体架构

arch22 上三角求解**行间存在串行依赖**（第 i 行依赖已解出的行），因此不使用
arch35 的多核 level-set 方案，而采用：

```
analysis            host 侧归一化 B = op(A) → CSR(B) + diagPtr，写入 workspace
   ↓
solve               单核 AIV 行窗回代（LOWER 前向 / UPPER 反向）
```

### 3.1 快路径 vs. 通用路径

| 条件 | 路径 | 说明 |
|---|---|---|
| CSR + op=N + base0 + I32 | **快路径** | kernel 直读原始 CSR，零拷贝 |
| 其他（CSC/COO/SELL、base1、I64、op=T/H） | **通用路径** | analysis 构建 CSR(B) 副本 |

快路径下 workspace 仅存 `diagPtr`；通用路径额外存 CSR(B) 的
`rowPtr / colInd / values`。

### 3.2 host 侧归一化（`Normalize`）

按**原始存储顺序**枚举 A 的条目 `(r_p, c_p, v_p)`（存储下标 `p`），
映射到 B：

| opA | B 中条目 |
|---|---|
| N | `(r_p, c_p, v_p)` |
| T | `(c_p, r_p, v_p)` |
| H | `(c_p, r_p, conj(v_p))` |

按 B 的行做计数 → 前缀和 → 散射，得到 CSR(B)，同时记录：

- `srcPos[d]`：CSR(B) 第 d 槽 → **原始 values 数组下标**（SELL 为槽位下标）
- `diagPtr[i]`：B 第 i 行的对角槽位（无则 -1）

`srcPos` 是 updateMatrix 能跨格式正确工作的关键：newValues 按**原始格式布局**
提供，靠 `srcPos` 散射回 CSR(B)。

各格式的枚举方式：

- **CSR**：`ptrs`=行偏移，`idxs`=列索引
- **CSC**：`ptrs`=列偏移，`idxs`=行索引
- **COO**：`ptrs`=行索引数组，`idxs`=列索引数组（逐条目）
- **SLICED_ELL**：`ptrs`=slicePtr，`idxs`=列索引（**含 -1 填充**，跳过）

### 3.3 workspace 布局

首个子分配 512B 对齐，内部子分配 64B 对齐：

```
[CSR(B) rowPtr: int32[m+1]]        仅通用路径
[CSR(B) colInd: int32[nnzB]]
[CSR(B) values: esz[nnzB]]
[diagPtr:        int32[m]]         总有（nnz==0 时全 -1）
```

`nnz == 0` 时仍分配（全零 rowPtr + 全 -1 diagPtr），使 solve 走正常路径并
自然产出 `Y = α·X`（UNIT）或 Inf/NaN（NON_UNIT 奇异），与 cuSPARSE 一致。

### 3.4 solve kernel

单核、按**行窗**推进（BUF_ROWS=512，窗内非零元 ≤ MAX_ENTRIES=8192）：

```
每窗 5 次批量搬运：rowPtr / diagPtr / colInd / values / X  →  计算  →  Y 回写
```

- LOWER：`lo=0` 起前向；UPPER：`hi=m` 起反向
- 窗口边界：`end` 是**排他下界**（末窗为 -1），故 `lo = end+1, hi = start+1`
  —— 写成 `min(start,end)` 会得到 `lo = -1` 导致 GM 越界（已修复的踩坑）
- Y 读取：窗内命中 UB，窗外回退 GM 标量读
- 原地（X ≡ Y）安全：每行先读自身 `X[i]` 再写 `Y[i]`，且只引用**已解出**行

### 3.5 updateMatrix

- **GENERAL**：newValues 全量替换原始 values → `H2D` 回 `matA->values`；
  若为通用路径则再用 `srcPos` 重散射到 CSR(B)
- **DIAGONAL**：先读出原始 values，按 `diagPtr[i]` 打补丁（`newValues[i]` 为第 i 行新对角），再回写 + 重散射

两者均为 host 侧实现（update 不在热路径）。

---

## 4. 精度语义

- 计算全程 FP32 / complex64（complex64 用 `{r, i}` 双 float）
- 复数除法按 `|b|²` 归一化，与 `torch.linalg.solve_triangular` 一致
- 判定口径：`allclose(rtol=1e-5, atol=1e-8)` 逐元素

### 实测（m=64，对角占优三角阵，NumPy float64/complex128 参考解）

**24 / 24 PASS**（FP32 + complex64 × LOWER/UPPER × UNIT/NON_UNIT × N/T/H），
最大绝对误差 `2.8e-07 ~ 1.9e-06`。

### 规模残差

| 场景 | m | nnz | `‖A·y−αx‖∞/‖αx‖∞` |
|---|---|---|---|
| P-01 | 65536 | 524260 | `2.82e-07` |
| P-02 | 131072 | 1572798 | `4.11e-07` |
| P-03 | 262144 | 3932055 | `4.84e-07` |

---

## 5. 性能

### 5.1 实测（solve，3 次均值，含 warmup）

| 场景 | solve (ms) | 目标 0.25×GPU (ms) | 倍数 |
|---|---|---|---|
| P-01 | 274 | 128.5 | 2.1× |
| P-02 | 785 | 307.5 | 2.6× |
| P-03 | 1925 | 770.0 | 2.5× |

### 5.2 瓶颈定位

带宽 / 行数扫描（固定 m 变带宽、固定 nnz/行 变 m）：

| nnz/行 | 每行 (µs) | 每非零元 (ns) |
|---|---|---|
| 2 | 1.367 | 683.6 |
| 4 | 2.287 | 571.7 |
| 8 | 4.120 | 515.0 |
| 16 | 7.783 | 486.4 |
| 32 | 15.011 | 469.1 |

**每行固定开销 ≈ 0，成本 ≈ 500 ns/非零元** —— 瓶颈是 AIV 上的**标量内层循环**，
GM 事务已不是主要矛盾（按行窗批量搬运后已从 344 ms 降到 274 ms）。

### 5.3 已验证的加速路径（暂未启用）

对列连续的行改用向量指令求点积：

```cpp
Mul(tmpBuf_, valBuf_[base], yBuf_[yoff], L);      // 逐元素乘
ReduceSum<float>(sumBuf_, tmpBuf_, workBuf_, L);  // 归约
```

实测 **P-01：274 ms → 21 ms（12.8×）**，远低于 128.5 ms 目标；复杂行仍走精确标量路径。
但该路径在 FP32 上触发 `EZ9999` 设备异常（疑似 `LocalTensor` 切片 /
`ReduceSum` 尾数对齐问题），已用 `SPSV_ARCH22_ENABLE_VECTOR_DOT` 宏**默认关闭**，
代码保留待修。这是把性能拉到达标的唯一已验证方向，列为第一优先级。

---

## 6. arch22 工程约束（与 arch35 差异，重要）

1. `TPipe` **没有** `AllocTensor`；须 `TPipe pipe_` 成员 +
   `pipe_.InitBuffer(queue, N, bytes)` + `queue.AllocTensor<T>()`（同 spmv arch22）
2. `__global__` 必须带 `__aicore__`；头文件**不要**重复声明 kernel
   （会与定义构成重载集，导致 `<<<>>>` 调用歧义），只声明 `_do` 启动器
3. 纯标量、无 AscendC 内存指令的小 kernel 链接期报 `auto derivate failed`
   → 改为 host 侧实现或并入主 kernel（本次因此移除 3 个小 kernel）
4. `DataCopyExtParams` 初始化禁止 `sizeof` 隐式提升
   （`unsigned long → uint32` 窄化报错），须 `static_cast<uint32_t>(sizeof(T))`
5. `DataCopyPad`：GM→UB 用 4 参（含 `DataCopyPadExtParams`），UB→GM 用 3 参
6. `GlobalTensor`：`gm(i)` 标量读，`gm[i]` 切片

---

## 7. 测试

| 脚本 | 内容 |
|---|---|
| `verify_arch22.py` | ctypes 驱动真实公开 API，24 组合精度（NumPy 参考解） |
| `perf_arch22.py` | P-01/02/03 性能 + 相对残差 |
| `sweep_arch22.py` | 带宽/行数扫描，用于瓶颈归因 |

---

## 8. 已知局限与后续

1. **向量点积路径修复**（唯一已验证达标路径，12.8×）
2. C++ ST（`test/spsv/arch22`，四格式 + 异常/alias/stream）
3. ATK 钩子（`spsv_analysis_npu` / `spsv_update_npu` / `spsv_npu`）与 200 用例
4. CSC / COO / SLICED_ELL 与 base-1 目前**代码路径已实现但未被用例覆盖**
5. 多核并行：串行依赖下单核是安全选择；若需进一步提速，可考虑
   分块流水线（block-level 波前推进 + 跨核同步），代价是复杂度显著上升
