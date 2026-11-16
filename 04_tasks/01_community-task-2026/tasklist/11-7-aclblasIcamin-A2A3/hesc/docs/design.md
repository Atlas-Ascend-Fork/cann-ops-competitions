# aclblasIcamin（A2/A3）算子设计文档

## 需求背景（required）

### 需求来源

【11月社区任务】aclblasIcamin 算子开发（A2/A3）：在 Atlas A2/A3（Ascend 910B3 / 910_93，CANN 9.1.0，ops-blas `arch22`）上用 Ascend C 实现 `aclblasIcamin`，完成设计、开发、测试全流程，验收通过后合入 [cann/ops-blas](https://gitcode.com/cann/ops-blas) `blas/iamin/arch22/`。

### 背景介绍

- 算子语义与 cuBLAS `cublasIcamin` 逐参数对齐：查找单精度复数（COMPLEX64）向量 x 中最小模元素的 **1-based 索引**，模按 BLAS icamin 惯例取 `|Re(x[k])| + |Im(x[k])|`（1-范数，非欧氏模），模相同时返回最小索引；数学表达式 `result = argmin_i (|Re(x[k])| + |Im(x[k])|)`，i = 1..n，k = 1+(i-1)*incx。
- 输出为单个 INT32 整数索引，精度判定为与 golden **精确一致（bit-exact）**。
- 接口声明已在 ops-blas `include/cann_ops_blas.h`（L241）提供，与其他产品线共用，禁止私有平行接口。
- 仓内现状：`blas/iamin/` 仅有 arch35（Ascend 950PR）实现，其复数幅度折叠依赖 `DeInterleave` 与 SIMT API——**两者在 910B（dav_m200）上均不可用**（CANN 9.1.0 头文件中 DeInterleaveImpl 仅有 dav_3510/l300 系实现），因此 arch22 需要一套纯向量指令的幅度折叠方案。

## 需求分析（required）

### 需求拆解

| 需求项 | 方案 |
| --- | --- |
| 正确计算 argmin(abs1) 索引（incx≥1） | incx==1 向量路径：Abs → PairReduceSum(|re|+|im|) → NaN 消毒 → ReduceMin(calIndex)；incx>1 标量路径：逐元素 GM 标量读取 + 严格小于累加 |
| NaN 语义对齐 cblas（严格小于、NaN 永不更新、首元素 NaN 结果为 1） | 向量路径 NaN 幅度改写为 +Inf（Compare(x,x,EQ) 仅 NaN 自比为假 + Select）；0 号核检测 x[0] 为 NaN 时强制 (NaN, idx 0)，终归并以 slot 0 播种，NaN 种子阻断后续严格小于更新 |
| tie 取最小索引 | ReduceMin(calIndex) 返回 tile 内首次出现位置；tile 间/核间按索引升序以严格小于（或等值更小索引）归并 |
| n=0 / incx<1 quick return（result 写 0，SUCCESS） | host 侧 `aclrtMemsetAsync` 异步写 0，不触发 kernel |
| n<0 / 空指针 / (n-1)*incx 超 int32 → INVALID_VALUE | host 校验链，int64 计算跨度（issue #745） |
| 大向量性能 ≥ 0.8× A100 | 40 AIV 分核流式读取 + 双缓冲 MTE2/V 重叠；单核小向量只发 1 个 kernel 直写结果 |

## 详细设计（required）

### host 侧设计

校验顺序与任务书 §2.4 对齐（quick return 先于 x 空指针检查，后于 result 空指针检查）：

```
handle == nullptr            → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
n < 0                        → ACLBLAS_STATUS_INVALID_VALUE
result == nullptr            → ACLBLAS_STATUS_INVALID_VALUE   （任意 n，含 quick return 路径）
n == 0 或 incx < 1           → aclrtMemsetAsync(result, 0) → SUCCESS（不检查 x）
n > 0 且 incx ≥ 1 且 x == nullptr → ACLBLAS_STATUS_INVALID_VALUE
int64 (n-1)*incx ∉ [0, INT32_MAX] → ACLBLAS_STATUS_INVALID_VALUE（host int64 计算）
```

workspace 复用 `aclblasHandle_t` 默认区（`GetEffectiveWorkspace` / 不足时 `EnsureDefaultWorkspace`），每核一对 (minVal, minIdx) float 槽，按 64 float 对齐（32B 整块，方便归并 kernel 一次 DataCopy）。tiling 结构体按值随 kernel 参数下发（cgeru 同款），无逐调用 H2D tiling 拷贝、无 op 内同步。

### 分核及分块设计

- `useCoreNum = min(n, GetAivCoreCount())`（910B3 实测 40 AIV）。
- **incx == 1**：`perCoreN = CeilAlign(CeilDiv(n, blocks), 4)`（4 复数 = 32B），每核切片起点与长度均 32B 对齐，整块 DataCopy 恒对齐；对齐只会收缩核数。`useCoreNum = CeilDiv(n, perCoreN)`，末核 `lastCoreN`。
- **n ≤ 8192（2 个 tile）**：单核路径，一次 kernel 直写 result，无 workspace 往返、无第二发 launch。
- **incx > 1**：按逻辑元素均分（`perCoreN = n/blocks`，末核带余数），标量读取任意对齐均可。
- UB tile = 4096 复数：双缓冲输入队列（2×32KB）+ 幅度缓冲（16KB）+ mask（4KB）+ ReduceMin work（≤0.3KB），约 84KB，远小于 248KB 向量 UB。

### kernel 侧设计

两段式（同 arch35 isamin 布局，kernel 边界天然提供全流水排空，避免 isamax 式单 kernel SyncAll 的跨核 MTE3→MTE2 可见性开销链）：

1. `icamin_kernel<<<useCoreNum>>>`——每核扫描：
   - **向量路径（incx==1）**：软件流水 tile 循环——DeQue tile t（等它的 MTE2）→ 发 tile t+1 拷贝 → 折叠 tile t。折叠链全部用 m200 可用指令：
     `Abs(2T, 原地)` → `PairReduceSum(mag, in, T/32 repeats, mask=64, dstRepStride=1, 1, 8)`（**dstRepStride 单位为 128B 半 repeat，取 1 即输出密集**，从 interleaved abs 直接得到 |re_k|+|im_k|）→ `Compare(mag, mag, EQ)` + `Select`（IEEE 有序自比仅 NaN 为假，NaN 幅度改写为 +Inf）→ `ReduceMin(slot, mag, work, T, calIndex=true)` 得 (值, 局部索引)；tile 尾部以 DataCopyPad(+Inf) + Duplicate(+Inf) 补齐到 64 幅度粒度，补位值恒为 +Inf 不可能胜出。
     - 相关向量指令间插 `PipeBarrier<PIPE_V>()`（cgbmv/isamax arch22 同款防重叠）。
     - tile 候选按 tile 升序做标量归并：严格小于或等值更小索引才更新（V_S 事件等 ReduceMin 落地后读 slot）。
   - **标量路径（incx>1）**：逐元素 GM 标量读（int64 地址防回绕），`|re|+|im|` 后按 golden 同款严格小于累加——NaN 天然不更新、tie 天然取先出现者，无需消毒。
   - **首元素 NaN 强制**：0 号核在归并后读 GM x[0]、x[1]，任一为 NaN 则强制 (NaN, idx 0)——golden 以 abs1(x[0]) 播种且 NaN 阻断一切更新，此为唯一需要特判的分歧点。
   - 单核模式直写 `result = bestIdx + 1`；多核模式经 PipeBarrier 后 DataCopyPad 写入 workspace 槽 (val, idx)。
2. `icamin_reduce_kernel<<<1>>>`（仅多核模式）：一次 DataCopy 读回全部槽（64 float 对齐），标量有序归并——**以 slot 0 播种**（可能携带强制 NaN），后续槽仅在严格小于或等值更小索引时更新；IEEE 比较使 NaN 槽自动透明。写出 1-based INT32。

**正确性论证（分治 = 顺序扫描）**：各切片候选 = 切片内首次出现的最小值（NaN 等价 +Inf 被排除；x[0] 为 NaN 的冻结语义由强制步骤恢复）；切片按索引升序、严格小于归并 ⇒ 与 cblas 顺序扫描逐 case 等价，归约顺序不影响整数结果。

### 支持硬件

- Atlas A2 训练/推理系列（Ascend 910B，dav_m200；实测 Atlas 800T A2 / 910B3）
- Atlas A3 训练/推理系列（Ascend 910_93，dav_c220，同代向量指令）

### 算子约束限制

- n ≥ 0；n = 0 或 incx < 1（含 0/负步长/INT32_MIN）为合法 quick return（result=0，SUCCESS）；本族负步长不反向遍历（对齐仓内 `blas/iamin/README.md` 约束）
- n < 0、handle/x/result 空指针、(n-1)×|incx| 超 int32 → `ACLBLAS_STATUS_INVALID_VALUE` / `HANDLE_IS_NULLPTR`
- 输入 COMPLEX64 交错存储（实部/虚部各 float32），输出 INT32 标量，Device 内存
- 异步执行：依赖 `aclblasSetStream`，读回结果前须同步 stream

## 可维可测分析（required）

### 精度标准 / 性能标准

- 精度：INT32 整数索引 bit-exact（EXPECT_EQ），golden 为测试工程内 cblas 风格 CPU 循环（Netlib 无 icamin 例程）
- 性能：Atlas 800T A2 (910B3) 实测，5 次 warmup 后连续 ≥10 次取平均，判据 avg_us ≤ gpu_ms × 1000 / 0.8（A100-SXM4-80GB / cuBLAS 12.8.4 基线）

### 兼容性分析

- 接口签名与其他产品线共用（`include/cann_ops_blas.h` L241，950 已有 arch35 实现，本任务补 A2/A3）
- 测试工程与 arch35 共用 param/golden/wrapper 头；CMakeLists 在 arch22 / arch35 分支注册 GTest；对 CANN 9.1.0 缺失的 `ACL_HIFLOAT4` 枚举按 chemv 先例做 try-compile 兜底（新 CANN 下检测通过、无副作用）

### 自测与证据（Atlas 800T A2 / 910B3 / CANN 9.1.0）

- **精度：1201/1201 全过**——随包 CSV 1000 条（L0 基础/尺寸扫描/步长/填充/边界负向）+ 200 条 GAUSS_NORM_5_2 正态补充（任务 §3.5 要求，测试框架 `fill.h` 原生支持该 token，按 chemv 先例以独立 supplement CSV 补齐）+ NullHandle 契约单测
- **性能：200/200 全过**（TC_PF 全量，warmup=5 / iters=20）
- 典型 case msprof op Task Duration（kernel 纯耗时）vs A100 基线：

| n | NPU TD(us) | A100(us) | 倍率 |
| --- | --- | --- | --- |
| 1048576 | 10.90 | 18.16 | 1.67× |
| 2097152 | 16.22 | 21.62 | 1.33× |
| 4194304 | 27.34 | 36.61 | 1.34× |

- 小向量（n ≤ 8192 单核路径）aclrtEvent 口径 4.5~6.6µs（A100 约 8.2µs），规模扫描（16K~4M）7.5~35µs 全部在 0.8× 门内

## 提交范围

```
blas/iamin/arch22/            icamin_host.cpp / icamin_kernel.cpp / icamin_kernel.h / icamin_tiling_data.h
blas/iamin/README.md          aclblasIcamin 产品支持表标注 Atlas A2/A3 支持
test/iamin/icamin/CMakeLists.txt   arch22 分支注册 GTest（含 ACL_HIFLOAT4 兜底）
test/iamin/icamin/README.md   测试与复现说明
test/iamin/icamin/arch22/     icamin_test.cpp / icamin_test.csv(1200) / icamin_gauss_supplement.csv(200) / gpu_baseline.csv
test/iamin/icamin/icamin_npu_wrapper.h   span 超 int32 负向用例按任务书 §3.5 单元素 dummy 占位
```
