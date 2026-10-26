# aclsolverCheevj 算子设计文档

- **算子名**：`aclsolverCheevj`
- **对标接口**：cuSOLVER `cusolverDnCheevj`（Jacobi 特征值分解）
- **适配硬件**：Atlas A2 训练系列（ascend910b / dav-2201 / arch22）
- **开发语言**：Ascend C + C++（host）
- **目标仓**：`gitcode.com/cann/ops-solver`（`src/cheevj/`）
- **状态**：设计中（本地算法原型已 584/584 通过，见 `dev/prototype/cheevj_reference.py`）

---

## 一、需求背景与接口定义

### 1.1 功能

计算复 Hermitian 矩阵 `A`（complex64，列主序）的特征分解 `A·V = V·diag(w)`：

- `w` 为**升序**排列的实特征值（float32）；
- `jobz=V` 时输出特征向量（列主序覆写 `A`）；`jobz=N` 时**输入 A 不得被修改**；
- `uplo` 指定只读上三角（`U`）或下三角（`L`），另一半由共轭对称补齐。

### 1.2 仓库 API 表面（与任务书 cuSOLVER 原型签名的差异）

任务书原型带有 `work / lwork / rwork / syevjInfo_t params`。仓库（ops-solver）既有 solver
算子统一采用 handle + 枚举 + 出参的形态，workspace 由实现内部管理，因此本算子对外：

```c
aclError aclsolverCheevj(aclsolverHandle_t handle,
                         aclsolverEigMode_t  jobz,   // NOVECTOR / VECTOR
                         aclsolverFillMode_t uplo,   // UPPER / LOWER
                         const int64_t n,
                         std::complex<float> *A,     // 入：Hermitian；V 模式出：特征向量
                         const int64_t lda,
                         float *W,                   // 出：升序特征值
                         int32_t *info);             // 出：0=成功，<0=参数错，>0=未收敛
```

> 兼容性：`jobz`/`uplo` 同时接受公开枚举与字符 `'N'/'n'/'V'/'v'`、`'U'/'u'/'L'/'l'`；
> **不接受** PLASMA/MAGMA 的 121/122 数值编码（与 PLASMA 方向相反，误收会静默读错三角）。

### 1.3 约束

- 仅 complex64（complex64 在 arch22 支持集内；arch35 不支持）；
- `n ≥ 0`（`n == 0` 为合法空输入，直接成功返回）、`lda ≥ max(1, n)`；
- 输入必须是 Hermitian，对角线虚部忽略（强制置 0）；
- 工作空间随 `n²` 增长，超大 `n` 需受设备/主机内存约束。

---

## 二、算法选型（含第一性原理可行性论证）

### 2.1 两条候选路线

| 路线 | 复杂度 | 精度特征 | 并行度 | 结论 |
|---|---|---|---|---|
| **A. 两步循环 Jacobi**（任务书点名算法）| O(S·n³)，S≈6~15 扫描 | 特征值**相对精度高**（尤其小特征值、紧簇），向量正交性好 | 扫描内可按 (p,q) 对并行，但每对更新 2 行 2 列 → 同步密集 | **语义基准 / 小矩阵** |
| **B. Householder 三对角化 + Sturm 二分 + 逆迭代 + 回代** | (4/3)n³ + O(n²·iters) + n³ | 标准 LAPACK 级别 | Sturm 各特征值**完全独立**（天然多核）；三对角化按 panel 分块 | **中大矩阵主力** |

### 2.2 性能目标与理论天花板（先算上界再决定投入）

目标：达到 0.8×A100，即 **NPU 时间 ≤ GPU 时间 ÷ 0.8 = 1.25 × GPU**：

| N | jobz | uplo | A100 (ms) | 预算 (ms) | A2 理论下限估算 | 判定 |
|---|---|---|---|---|---|---|
| 512 | V | U | 21.13 | **26.4** | ~0.05 | 余量充足 |
| 1024 | V | U | 87.72 | **109.6** | ~0.4 | 余量充足 |
| 2048 | V | L | 483.21 | **604.0** | ~3 | 余量充足 |
| 1024 | N | U | 8.29 | **10.4** | ~0.05 | 最紧的一条 |

估算依据（A2 / 910B）：FP32 Cube 全片 ~180 TFLOPS，向量单元 FP32 峰值约其 1/4~1/2；
三对角化 ~8·(4/3)n³ flops，回代 ~8n³ flops，Sturm ~48n² flops。
**即便只跑到理论峰值 1% 的量级，四条标杆都成立 ⇒ 瓶颈在软件调度/访存，不在硬件算力。**
（该结论与 `_refs/任务11` 的方法论一致：性能不达标时先算理论最优，判定天花板后再决定投入。）

### 2.3 选型结论

采用 **B 为主力、A 为语义/小矩阵路径** 的混合分派（与已合入 master 的参考实现方向一致，
但本设计在以下点上有独立取舍，见 §3.6）：

1. `n` 较小或矩阵已接近对角/块对角 → 走特判/小核路径（Jacobi 或闭式解）；
2. 基准形状与中大矩阵 → Householder + Sturm 流水线；
3. 超出设备可承受规模 → host 侧多线程回退（保证功能正确优先）。

---

## 三、总体设计

### 3.1 分派策略（host 侧）

```
输入校验（n/lda/jobz/uplo/指针）→ 空输入早退
   ├─ 对角阵特判          → 闭式（对角排序）
   ├─ 2×2 块对角特判      → 闭式（2×2 解析解）
   ├─ n 小（≤ 129）       → host/OpenMP 回退 or 单核 Jacobi
   ├─ 基准形状 512/1024/2048 → 固定形状 NPU 流水线（最优化路径）
   ├─ 通用形状            → 通用 NPU 流水线 / padding 到固定形状
   └─ n > 2048 或内存超预算 → host 回退（带预算守卫，返回错误码而非抛异常）
```

**三条防御纪律**（来自 `_refs`）：
- 空输入三层防御：aclnn 早退 + tiling `n==0 → coreNum=1` + kernel 空转守卫；
- host 回退前做内存预算估算（如 >8 GiB 直接返回错误码，避免 `std::bad_alloc` 穿越 C 边界）；
- 任何 ACL 运行时调用失败都要**释放此前已获取的设备 buffer**（RAII 风格的 cleanup lambda）。

### 3.2 设备流水线（kernel 侧，分阶段多 launch）

参考实现已验证：**各阶段不能合成一次物理 launch**（AIV/AIC 任务类型、block 数、全局同步边界不同）。

| 阶段 | 职责 | 并行形态 |
|---|---|---|
| ① prepare / pack | 按 uplo 展开成满 Hermitian，拆为 planar（re/im 分离）便于向量化 | 按列分核 |
| ② panel（Householder） | 逐 panel 生成反射器，消掉 panel 下方元素 | 单/少核（强依赖） |
| ③ trailing update（Her2K） | 用反射器更新尾块 `A ← A − U V^H − V U^H` | **Cube/Mmad 或多核 AIV** |
| ④ archive | 反射器归档（供回代使用） | 单核 |
| ⑤ Sturm / bisect | 三对角特征值（Gershgorin 定界 + 符号计数二分） | **每特征值独立 → 满核并行** |
| ⑥ tridiag vectors | 逆迭代求三对角特征向量（带簇内重正交化） | 每/每簇一个核 |
| ⑦ back-transform | `V = Q·Z`，按逆序施加 Householder | 按列分核 |
| ⑧ finalize / output | 升序排序、写回 `W` 与 `A`、写 `info` | 单核 |

### 3.3 关键数值细节（原型已验证，kernel 必须照抄）

1. **复 Householder 的相似变换是 `A' = H^H·A·H`**（不是 H·A·H），反射器
   `H = I − tau·v·v^H`，由 `H^H·x = β·e₁` 解得：
   - `β` **恒为实数**：`β = −sign(Re α)·‖x‖`
   - `tau = −v₀/β = (β − α)/β`（`v₀ = α − β`，`v = x/v₀`，`v[0] = 1`）
   - rank-2 更新：`u = tau·(A·v − 0.5·conj(tau)·(v^H A v)·v)`；`A ← A − u·v^H − v·u^H`
     （**内层是 `conj(tau)`**，写成 `tau` 会在 n≥4 时系统性出错）
2. **逆迭代必须用带部分主元的三角 LU**（LAPACK `dgttrf`/`dgttrs`），且
   **行交换会产生第二超对角 `du2`**，漏掉 `du2` 会让"一有行交换就崩"。
3. **主元扰动 `pivmin` 必须随 λ 走**（`~1.2e-7·max(|λ|, max|e|)`），并取
   "到最近邻特征值间距的 1/4" 为上限；**绝不按全局 ‖T‖ 取**
   （wide_scale 场景会得到 0.1 量级扰动，把小特征值间距抹平 → 向量不正交）。
4. 若实现 Jacobi 路径：`J = D·G`，`D = diag(e^{iθ/2}, e^{−iθ/2})`，
   `G = [[c, s], [−s, c]]`，**`tan(2t) = −2α/(x − z)`**（符号搞反时该 2×2 非对角不归零）。
5. 特征值升序由二分天然保证（各 index 独立），排序只做保险；V 模式按同一 `order` 重排列。

### 3.4 Tiling 与 UB 预算

- **单一来源原则**（`_refs/任务9` 最贵的一课）：host 计算 UB 预算与 kernel 实际分配
  必须使用**同一套常量**（放在共享头/tiling 结构里），不一致不会报错，只会静默越界 →
  表现为"结果部分缺失（全 0）+ 非确定"，极易误判为同步问题。
- 复数按 **2×float planar** 搬运（re/im 分离），天然规避 arch22 上
  `DeInterleave`/`Interleave`/`Scatter` 不可用（`Gather` 可用，但 **srcOffset 是字节偏移**）。
- 分核写回：**多核 AIV 标量 GM 写不可靠**（丢写，粒度 ≥64B）→ 一律
  `UB 计算 → DataCopy/MTE3 写回`；每核独占 64B 对齐的写区间。

### 3.5 TilingKey 规划

`TilingKey = jobz(1) | uplo(1) | size_level(2) | path(2)`：
- `jobz`：N/V；`uplo`：U/L
- `size_level`：`n≤512` / `512<n≤2048` / `n>2048`
- `path`：对角特判 / 固定形状 / 通用 / padding

kernel 内对固定形状（512/1024/2048）走文件内分支（不拆独立文件），通用形状走 tiling 化路径。

### 3.6 与参考实现（已合入 master）的取舍差异

| 项 | 参考实现 | 本设计 |
|---|---|---|
| 固定形状特化 | 512/1024/2048 各有专属实现 | 保留，但**常量与 tiling 同源**（避免 8/16 对齐口径不一致） |
| 正交性兜底 | 紧簇时正交性探针失败 → 整体 host 重解 | 改为**簇内重正交化 + 局部重解**，避免整矩阵回退的 10× 代价 |
| Jacobi 语义 | 未实现（走 Householder） | 保留 Jacobi 路径作为 `jobz` 语义基准与小规模路径（原型已验证 584/584） |
| pivmin | — | 明确 λ-relative + 间距上限（原型实测修掉 wide_scale 正交性失败） |

---

## 四、精度设计

- **判据**（沿用仓库 `experimental_standard.md` 混合容差，complex64 无独立条款 →
  实/虚分别按 FLOAT32 判定）：
  `|a − g| ≤ atol + rtol·|g|`，`rtol = 2⁻¹⁰`、`atol = 2⁻¹⁶`，并满足 `matched_ratio ≥ 0.99`。
- **V 模式附加判据**：残差 `‖A·V − V·diag(w)‖_F / ‖A‖_F < 1e-3`、
  正交性 `‖V^H·V − I‖_F < 1e-2`。
- **golden**：`numpy.linalg.eigh`（float64 输入 → float32 结果），与 kernel 算法不同源，
  属交叉验证；本地另有 float32 全算法原型供数值行为对齐（不是 golden）。
- **已知边界**：极端病态/重特征值矩阵收敛变慢，`info > 0` 表示未收敛（不静默返回成功）。

## 五、性能设计

- **口径**：以 `msprof` 的 kernel `Task Duration` 求和为准（端到端含 H2D/D2H 会放大）；
  另给出端到端中位数作参考，两组都记。多轮取中位数抗噪声。
- **手段优先级**：① 三对角化尾块更新走 Cube（Her2K/Mmad）；② Sturm 按特征值满核并行；
  ③ 回代按列分核 + 双/三缓冲流水；④ 减少 GM 往返（反射器常驻 workspace）。
- **构建即核对 `CMAKE_BUILD_TYPE`**：`_refs/任务2` 的最贵一课是仓库 CMake 强制 Debug
  导致模板库慢 175× —— 开工第一批动作就核对实际 `-O` flags。
- **编译器风险预案**：arch22 上出现过 bisheng `≥-O1` 误编译设备内核（错误值恒定、
  形状相关）→ 精度全挂时**先做"优化级别二分"**（逐 TU pin `-O0`），不要先怀疑算法。

## 六、测试设计

沿用仓库 `test/cheevj` 三件套并扩充：

1. **契约门禁**（`cheevj_contract_test`）：非法参数、`n=0`、`lda` strided、U/L 三角隔离、
   N 模式输入不变性、N/V 一致性、`n` 超上限的错误码路径、V 模式残差与正交性。
2. **正确性**（`gen_data.py` → `cheevj_test` → `verify_result.py`）：
   规模 × {N,V} × {U,L} × 矩阵类型（随机 Hermitian / 对角 / 2×2 块对角 / 重特征值 /
   病态 wide_scale / 单位阵 / n=1,2 边界）× 随机种子可复现。
3. **性能**（`cheevj_benchmark`）：预热 2 次、采样 7 次取中位数，跑任务书 4 条标杆。
4. **本地校验器先行**（`_refs/任务6` 纪律）：索引/代数/几何类错误先在纯 Python 校验器里
   跑通再上真机——本任务即 `dev/prototype/cheevj_reference.py`（已 584/584）。

## 七、风险与预案

| 风险 | 预案 |
|---|---|
| 数值不收敛（重特征值/病态）| `info>0` 显式上报；紧簇走簇内重正交化 |
| 大 `n` 内存不足 | host 回退前预算守卫，返回错误码而非 `bad_alloc` |
| 编译器误编译（arch22 ≥-O1）| 优化级别二分定位，必要时单 TU pin `-O0` 并在文档中记录 |
| 性能口径不一致 | 同时给出 msprof kernel 和与端到端两组数据 |
| UB 溢出（静默）| tiling 常量同源 + 本地预算模拟器 |
