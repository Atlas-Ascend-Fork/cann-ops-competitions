# 【社区任务】aclsparseCgtsv2 算子（Ascend 950 / A5）设计文档

| 项 | 内容 |
|---|---|
| 任务 | `10月社区任务-aclsparseCgtsv2算子开发(950)`，任务书 `aclsparseCgtsv2_A5_task_doc.md` |
| 目标仓 / 落点 | https://gitcode.com/cann/ops-sparse ：`include/cann_ops_sparse.h`、新增 `sparse/cgtsv2/arch35/`、新增 `test/cgtsv2/`、`torch_npu/cgtsv2_npu.cpp` |
| 平台 | Ascend 950PR / 950DT（A5，DAV_3510，arch35），CANN 9.0.0 及后续配套版本 |
| 提交目录 | `04_tasks/01_community-task-2026/tasklist/10月社区任务-aclsparseCgtsv2算子开发(950)/kevin_lee1231/` |
| 阶段 | 设计评审稿（载体：本文件；实现与实机数据在编码阶段补充） |

# 需求背景（required）

## 需求来源

CANN 社区任务 2026 · 10 月批次，`10月社区任务-aclsparseCgtsv2算子开发(950)`。任务书要求面向 Ascend 950（DAV_3510，`arch35`，A5），用 C++、Ascend C 和 ops-sparse 算子工程实现复数三对角线性方程组求解接口 `aclsparseCgtsv2` / `aclsparseCgtsv2_bufferSizeExt`，功能与 C++ 接口语义对标 cuSPARSE `cusparseCgtsv2`（CUDA Toolkit 13.3 Update 1 所含 cuSPARSE 13.3 Update 1）。

## 背景介绍

### 仓内现状：实型有、复型没有

`ops-sparse` 现在有三种与本题直接相关的资产：

| 资产 | 位置 | 与本题的关系 |
|---|---|---|
| 带部分选主元的实型三对角求解 | `sparse/gtsv2/arch35/`（SIMT，每线程一列 RHS，workspace `align128(3·m·4·n)`） | 算法骨架、主元交换与 fill-in 结构可直接对照；只差复数算术 |
| 同一算子的另一套共享因子实现 | `sparse/gtsv2/arch22/`（factor + solve 两段式，4 组因子 + 1 组 pivot 标记，workspace 与 `n` 无关） | 本设计"公共分解 + 回放"结构的现成先例 |
| 复数类型与复数算术写法 | `include/cann_ops_sparse.h:202-212` 的 `aclsparseComplex {x, y}`；`sparse/spmv/arch35/spmv_kernel.cpp` 的 `float2` 乘加；`sparse/spsm/arch35/spsm_plan_kernel.cpp` 的 Smith 复除分流 | 类型与复乘可直接参照；**模长比较、复除用于主元**在仓内无先例 |

全仓检索 `Cgtsv2` 为零命中：既没有 `aclsparseComplex_t`，也没有 `aclsparseCgtsv2*` 的任何声明、实现、用例。本任务不是"给现有接口补一个架构"，而是**新增一个公开复数 ABI 与一整套 Host / Kernel / 测试**。

### 与 Sgtsv2 的差异，也就是本任务的实质工作量

| 维度 | `aclsparseSgtsv2`（已有） | `aclsparseCgtsv2`（本任务） |
|---|---|---|
| 数据类型 | `float*`，实型 | `aclsparseComplex_t*`（8 字节），复型 |
| 类型定义 | 无 | 新增公开类型，需编译期断言 `sizeof == 8`、`alignof == alignof(float)` |
| ABI | 已冻结 | 新增两个函数，**不修改** `Sgtsv2` 声明与实现 |
| 算术 | 实加/减/乘/除 | 复加/减/乘 + Smith 复除 + 模长比较 |
| 主元判据 | `\|d'[i-1]\| >= \|dl[i]\|`（实绝对值） | 同结构，但判据换成**复数模长** |
| 数值风险 | FP32 沿消元链累积 | 在 FP32 上再叠加复除中间量溢出/下溢与模长平方溢出 |
| 测试设施 | `test/frame` 的混合容差比对可直接用 | `Verifier` 只支持 float/整型，**复数容差比对需另建** |

也就是说：算法结构可以照抄，**复数算术、模长判据、复杂 ABI、复数精度比对是四处新增内容**，不能靠"把 float 换成复数"糊过去。

### 算法

对每个 RHS 列 `j` 求解复三对角方程组 `A·X = B`，`A` 由 `dl/d/du` 给出（0 基下标：`A[i,i]=d[i]`、`A[i,i-1]=dl[i]`、`A[i,i+1]=du[i]`，且 `dl[0]=du[m-1]=0`）。采用带部分选主元的 Thomas 消元，与 LAPACK `zgtsv/zgbtrf` 的 2×2 主元块结构一致：

```text
前向消元（i = 1 … m-1，消去 dl[i]）：
    if |pivot_d| >= |dl[i]|:            # 不换行
        mult      = dl[i] / pivot_d
        d'[i-1]   = pivot_d ; du'[i-1] = pivot_du ; du2'[i-1] = 0
        next_d    = d[i] - mult·pivot_du
        next_du   = du[i]
    else:                               # 交换 row i-1 与 row i
        mult      = pivot_d / dl[i]
        d'[i-1]   = dl[i] ; du'[i-1] = d[i] ; du2'[i-1] = du[i]     # du2 即交换产生的填充
        next_d    = pivot_du - mult·d[i]
        next_du   = -mult·du[i]
回代（i = m-2 … 0）：
    x[m-1] = b'[m-1] / d'[m-1]
    x[i]   = (b'[i] - du'[i]·x[i+1] - du2'[i]·x[i+2]) / d'[i]
```

第 `i` 步的换行把原 `du[i]` 推到 row `i-1` 的第二上对角线，回代因此必须带 `du2'` 项。零主元不加保护：按 IEEE-754 自然产生 Inf/NaN，这是 `Sgtsv2` 与 Legacy API 的既有语义，任务书 2.1 第 5 条也明确要求"不崩溃、不越界，并记录实际返回码和输出"。

# 需求分析（required）

## 需求描述

在 Ascend 950（arch35）上以 C++ + Ascend C 实现两个公开接口，满足任务书 2.3/2.4 的 ABI 与参数契约：`complex64`、列主序就地输出、`bufferSizeExt → 128 字节对齐 workspace → 执行` 两阶段调用、异步 stream、`n=0` 空 RHS 提前返回；精度、性能、内存按任务书 3.2/3.3/3.4 验收；相同输入复跑逐位一致。

```c
typedef struct aclsparseComplex_t {
    float real;
    float imag;
} aclsparseComplex_t;

aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B,
    int ldb, size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B,
    int ldb, void *pBuffer);
```

### 类型与 ABI 约束

1. 新增 `aclsparseComplex_t` 按任务书 2.1 第 2 条定义为连续 `float real; float imag;`，并在头文件内做编译期断言：

```c
#if defined(__cplusplus)
static_assert(sizeof(aclsparseComplex_t) == 8, "aclsparseComplex_t must be 8 bytes");
static_assert(alignof(aclsparseComplex_t) == alignof(float), "aclsparseComplex_t alignment must equal float");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(aclsparseComplex_t) == 8, "aclsparseComplex_t must be 8 bytes");
_Static_assert(_Alignof(aclsparseComplex_t) == _Alignof(float), "aclsparseComplex_t alignment must equal float");
#endif
```

2. 它与仓内已有的 `aclsparseComplex {float x; float y;}` **布局相同但类型不同**：新接口只接受 `aclsparseComplex_t*`，`Sgtsv2` 的 `float*` ABI 与既有复数类型均不修改。设计文档与 README 明确二者可按 `memcpy` 互视为同布局，但不提供隐式转换，避免调用方误把两个 ABI 混用。

3. 公开接口只传地址与整数，raw C ABI 不能推断 dtype、分配长度或 stride；dtype/布局/长度由调用方保证，这与 cuSPARSE 同名接口一致。

### 参数与错误码

| 参数 | 输入/输出 | 类型 | 约束 | 违反时返回 |
|---|---|---|---|---|
| `handle` | 输入 | `aclsparseHandle_t` | 非空 | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| `m` | 属性 | int | `m >= 3`（`n=0` 时不校验） | `INVALID_VALUE` |
| `n` | 属性 | int | `n >= 0`；`n=0` 执行跳过、查询返回 0 | `INVALID_VALUE` |
| `dl/d/du` | 输入 | `const aclsparseComplex_t*` | `n>0` 时非空；`[m]` 连续 | `INVALID_VALUE` |
| `B` | 输入输出 | `aclsparseComplex_t*` | `n>0` 时非空；列主序 `[ldb,n]`，`B[i+j*ldb]` | `INVALID_VALUE` |
| `ldb` | 属性 | int | `ldb >= max(1, m)` | `INVALID_VALUE` |
| `pBufferSizeInBytes` | 输出 | `size_t*` | 非空 | `INVALID_VALUE` |
| `pBuffer` | workspace | `void*` | `n>0` 时非空且 128 字节对齐 | `INVALID_VALUE` |

`bufferSizeExt` 允许 `B == NULL`（查询不使用 B 的值），与 `Sgtsv2` 同语义；两个接口的 workspace 公式必须走同一函数，避免查询与执行不一致。

## 需求拆解

| 编号 | 项 | 验收对应 |
|---|---|---|
| T-1 | 公开类型与两个 ABI：`aclsparseComplex_t`、编译期断言、头文件声明 | 任务书 2.1(2)、2.3、7 |
| T-2 | Host：参数校验、错误码、workspace 查询、切核、双 kernel 下发、stream 异步 | 2.2、2.4 |
| T-3 | Kernel A（factor）：复数部分选主元的公共分解，产出 4 组因子 + 1 组换行标记 | 2.1(1)(3) |
| T-4 | Kernel B（solve）：按因子回放前向消元并回代，`B` 原地输出，`padding` 不写 | 2.1(1)(4) |
| T-5 | 复数数值细节：模长比较的缩放保护、Smith 复除、零主元传播 | 2.1(1)、3.2 |
| T-6 | C++ UT/ST：接口、Host、Kernel、类型布局、异常边界、确定性复跑、完整调用流程 | 2.2、3.5 |
| T-7 | 精度自验：随包 200 条 + 额外 800 条（实部/虚部分别判定） | 3.2 |
| T-8 | 性能自验：203 条逐例 Kernel 中位数倍率，Event 独立列示 | 3.3 |
| T-9 | 内存自验：逐例 io/workspace/对齐申请/恢复副本，两项条件择一说明 | 3.4 |
| T-10 | 端到端适配：`torch.ops.ops_sparse_test.cgtsv2_*` 两个注册点与可导入模块 | 3.5、交付件 2 |
| T-11 | 文档：`sparse/cgtsv2/README.md`、`test/cgtsv2/README.md`、`docs/zh/api_list.md` | 5 |

T-1/T-2/T-3 是主干；T-5 是唯一"仓内没有先例可抄"的部分，单独列出；T-10 是精度、性能、内存三项自验的前置依赖，缺它后面三项跑不起来。

# 详细设计（required）

## 算子分析

### 数学公式

`A·X = B`，`A = tridiag(dl, d, du) ∈ C^{m×m}`，`B ∈ C^{ldb×n}` 列主序，解 `X` 原地覆盖 `B` 的有效区 `[m,n]`。分量展开：

```text
d[0]·X[0,j]   + du[0]·X[1,j]                        = B[0,j]
dl[i]·X[i-1,j] + d[i]·X[i,j] + du[i]·X[i+1,j]        = B[i,j]     (1 <= i <= m-2)
dl[m-1]·X[m-2,j] + d[m-1]·X[m-1,j]                   = B[m-1,j]
```

调用方保证 `dl[0] = 0`、`du[m-1] = 0`，本实现不校验（与 `Sgtsv2` 一致，属用户责任）。

### 算法与主元策略

以公共分解 + 回放两步实现，等价于对每列独立做带主元 Thomas：

1. **公共分解（只做一次）**：选主元与消元只依赖 `dl/d/du`，与 `B` 无关。因此对三对角矩阵做一次前向消元，把每行的 `d'`、`du'`、`du2'`、行乘子 `mult`、换行标记落盘；这组序列对所有 RHS 列相同。
2. **逐列回放**：每列按同一次换行/消元序列更新自己的 `B` 列，再做三项回代。

主元判据为**复数模长**：`|pivot_d| >= |dl[i]|` 不换行，否则换行。实现不调用 `cabsf/hypotf`，改用"平方和比较 + 量级缩放"：平方保序，且省一次开方；当四个分量绝对值的最大值超出 `[2^-62, 2^62]` 时先按该最大值缩放再比，避免 `re²+im²` 上溢/下溢。

与 CPU Golden 的差异（需在报告中记录）：SciPy `solve_banded` 走 LAPACK `zgbtrf`，其复数主元判据是 `CABS1(z)=|Re|+|Im|`，与模长判据在近相等情形可能选出不同主元。两者都是稳定主元策略，解在容差内一致，但**不保证逐位相同**；精度判定按 3.2 的容差口径，不按逐位相等。

### 支持数据类型

仅 `complex64`（`float` 实部 + `float` 虚部）。函数名前缀 `C` 即单精度复数；`S/D/Z` 前缀不在本任务范围。`dl/d/du/B` 必须同为该类型；raw ABI 无法检测实际 dtype，契约由调用方保证。

### 支持形状与布局

| 项 | 约束 |
|---|---|
| `m` | `m >= 3`，无人为上限（只受 workspace 分配与 int32 运算溢出约束） |
| `n` | `n >= 0`；`n = 0` 为合法空 RHS，执行不发射 kernel、查询返回 0 |
| `ldb` | `ldb >= max(1, m)`，允许 `ldb > m` 的 padding；padding 区 `m <= i < ldb` 全程不读写 |
| 内存布局 | `dl/d/du` 连续一维 `[m]`；`B` 列主序连续 `[ldb, n]`，元素地址 `B[i + j*ldb]` |
| 其它 | 不广播、不接受转置视图/任意 stride、不引入 `batchCount`（`n` 不是 batch） |

## 算子实现

### 接口与类型设计

头文件在 `aclsparseSgtsv2_bufferSizeExt` 声明之后新增一段 `Legacy API: aclsparseCgtsv2` 区块：先放 `aclsparseComplex_t` 与两条编译期断言，再按仓内既有注释风格给出两个函数的 `@brief/@param/@return`。不触碰 `Sgtsv2` 的声明与实现，不改任何既有复数类型。

### 总体执行流程

```text
aclsparseCgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
  ├─ handle == NULL                          -> HANDLE_IS_NULLPTR
  ├─ n == 0                                  -> SUCCESS（不发射 kernel，不要求 pBuffer）
  ├─ 校验 m / ldb / 空指针 / pBuffer 128B 对齐
  ├─ aclsparseGetStream(handle, &stream)     -> 取当前 stream
  ├─ wsBytes = ComputeWorkspaceBytes(m, n)   （与 bufferSizeExt 同一函数）
  ├─ kernel A: cgtsv2_factor_kernel  <<<1 block>>>       分解 -> workspace
  └─ kernel B: cgtsv2_solve_kernel   <<<numBlocks>>>     回放 + 回代 -> B（原地）
```

两个 kernel 在同一 stream 顺序发射，kernel 边界天然是栅栏，不需要跨核同步、不需要显式 cache 维护；Host 侧不调用 `aclrtSynchronizeStream`，保持异步语义（仓内规范 `sparse-coding-conventions.md` 明确禁止 Host 同步）。

### Host 侧设计

#### 1. 校验顺序与错误码

顺序固定为 `handle → n == 0 短路 → n < 0 → m < 3 → ldb → 空指针 → pBuffer 对齐`，与 `Sgtsv2` 完全一致（`sparse/gtsv2/arch35/gtsv2_host.cpp:240-283`），保证同一组非法输入在两个算子上的返回码一致：

| 场景 | 返回 |
|---|---|
| `handle == NULL` | `HANDLE_IS_NULLPTR` |
| `n == 0`（执行接口） | `SUCCESS`，不算 workspace、不发射 kernel |
| `n < 0`、`m < 3`、`ldb < max(1,m)` | `INVALID_VALUE` |
| `n > 0` 且 `dl/d/du/B == NULL` | `INVALID_VALUE` |
| `n > 0` 且 `pBuffer == NULL` 或地址非 128 字节对齐 | `INVALID_VALUE` |
| `pBufferSizeInBytes == NULL`（查询） | `INVALID_VALUE` |
| workspace 乘法溢出 | `INVALID_VALUE`，且不写入脏值 |
| `GetAivCoreCount() == 0` | `INTERNAL_ERROR` |

分配失败、kernel 下发失败按仓内惯例交由调用方/ACL 返回，Host 不吞错。

#### 2. workspace 公式与布局

按"每行固定长度、与 `n` 无关"组织，共 4 组 complex64 因子 + 1 组单字节标记：

```text
WS(m) = align_up(33 · m, 128) 字节；n == 0 时返回 0

偏移            区域             类型/长度        含义
[0,        8m)  d'               complex64 × m    消元后主对角线（含交换改写的行）
[8m,      16m)  du'              complex64 × m    消去后上对角线（含第一上对角填充）
[16m,     24m)  du2'             complex64 × m    第二上对角线（换行产生的填充）
[24m,     32m)  mult             complex64 × m    第 i 步行乘子（i=0 不用）
[32m,     33m)  swap             uint8     × m    第 i 步是否交换 row i-1 / i
```

每段起点均为 8 字节整数倍（`m` 为整数时 `8m/16m/24m/32m` 天然 8 对齐），满足 complex64 标量访问对齐；整块 128 字节对齐由调用方保证，Host 校验。`m = 8192` 时 `WS = 270,336` 字节。

选这个结构的理由：分解结果对所有 RHS 列共享，因此 workspace 只与 `m` 有关。这一点与 `arch35 Sgtsv2`（每列一份 `3·m` 因子、共 `24mn` 字节）不同，与 `arch22 Sgtsv2`（共享因子、`align128(align128(17m,128)+85m)`）同构；代价与收益见"方案取舍"。

#### 3. 切核与 TilingData

```cpp
struct Cgtsv2TilingData {
    int32_t m;
    int32_t n;
    int32_t ldb;
    int32_t numBlocks;   // kernel B 的 block 数
    int32_t reserved;    // 对齐填充，禁止数组（仓内 R4）
};
```

- kernel A 固定 1 个 block、1 个执行线程：分解链严格串行，多核无收益。
- kernel B：`numBlocks = max(1, min(CeilDiv(n, 32), aivCoreNum))`，`aivCoreNum` 由 `GetAivCoreCount()` 运行时查询（禁止写死，仓内 R2）。每 block 32 个 SIMT 线程（与 `Sgtsv2 arch35` 的 `kGtsv2PivotThreadsPerBlock` 同值），线程按列 grid-stride。
- Tiling 以 `const` 引用传给 `kernel_do`，kernel 侧按值接收；不分配 GM tiling、不做 H2D 拷贝。

#### 4. 端到端适配（T-10）

ATK 精度/性能/内存三项都通过 `torch.ops.ops_sparse_test` 的两个注册点驱动，因此与算子代码一起交付 `torch_npu/cgtsv2_npu.cpp`（按仓内 `torch_npu/spmv_npu.cpp` 的模式）：

```cpp
TORCH_LIBRARY(ops_sparse_test, m) {
    m.def("cgtsv2_buffer_size_ext_npu(int m, int n, Tensor dl, Tensor d, Tensor du, Tensor B, int ldb) -> int");
    m.def("cgtsv2_npu(int m, int n, Tensor dl, Tensor d, Tensor du, Tensor B, int ldb, Tensor workspace) -> Tensor");
}
```

职责：校验 PrivateUse1 设备、`complex64` dtype、`[m]/[ldb,n]` 形状与列主序 stride `(1, ldb)`；用当前 NPU stream 建 handle；调 `bufferSizeExt`/`cgtsv2`；`B` 原地返回；workspace 128 字节对齐校验（**不能照抄 `spmv_npu.cpp` 的 64 字节常量**）；把 `INVALID_VALUE`/`HANDLE_IS_NULLPTR` 映射成带 `CGTSV2_INVALID_PARAMETER:` 前缀的异常，其它状态码原样抛出，不做 CPU 回退。配套一个可导入的 Python 模块（`import` 时 `torch.ops.load_library` 加载 `libops_sparse_torch.so`），供 `CGTSV2_EXTENSION_MODULE` 使用。

### Kernel 侧设计

两个 kernel 都是 arch35 SIMT 形态（`__simt_vf__` + `asc_vf_call` + `__global__` + `kernel_do`），遵守仓内 SIMT 模板的强制项：入口首行 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`、`extern "C"` 禁止 name mangling、tiling by value、不用 TPipe/TQue、TilingData 不含数组。不新增编译选项（根 `CMakeLists.txt` 明确禁止给混合 SIMT 编译单元加 `--enable-simt`）。

#### 1. Kernel A：公共分解（1 block × 1 thread）

```text
pd = d[0] ; pdu = du[0]                       # d'[0] / du'[0]
预读 cur_d = d[1], cur_du = du[1]
for i = 1 .. m-1:
    dl_i = dl[i]                              # 始终读原值
    if |pd| >= |dl_i|:                        # 不换行
        mult = dl_i / pd                      # Smith 复除
        wsD[i-1] = pd ; wsDu[i-1] = pdu ; wsDu2[i-1] = 0
        next_d  = cur_d - mult·pdu
        next_du = cur_du
        wsSwap[i] = 0
    else:                                     # 交换 row i-1 / row i
        mult = pd / dl_i
        wsD[i-1] = dl_i ; wsDu[i-1] = cur_d ; wsDu2[i-1] = cur_du
        next_d  = pdu - mult·cur_d
        next_du = -mult·cur_du
        wsSwap[i] = 1
    wsMult[i] = mult
    pd = next_d ; pdu = next_du
    预读 cur_d = d[i+1], cur_du = du[i+1]      # 预读掩盖 GM 延迟
wsD[m-1] = pd ; wsDu[m-1] = pdu ; wsDu2[m-1] = 0
```

关键不变式（实现注释与 UT 都要覆盖）：

- 第 `i` 步只读写第 `i-1` 行与第 `i` 行的状态，原始 `d[i]`、`du[i]` 在该步之前从未被写过，`dl[i]` 永远读原值；
- 换行分支三处落盘（`d'`、`du'`、`du2'`）与 `next_du` 的负号是复型与实型唯一的语义差别点；`du2'` 项只在 `i+2 < m` 的回代中参与；
- 末行 `i = m-1` 没有下一行可消去，单独落定，是循环唯一的分叉点。

#### 2. Kernel B：逐列回放与回代（SIMT，每线程一列）

```text
for col = blockIdx.x·blockDim.x + threadIdx.x ; col < n ; col += gridDim.x·blockDim.x:
    off = col · ldb
    # 前向：按 kernel A 的换行/乘子序列回放，就地更新 B 列
    b_prev = B[off + 0]
    for i = 1 .. m-1:
        cur = B[off + i]
        if wsSwap[i]:
            B[off + i-1] = cur
            b_prev = b_prev - wsMult[i]·cur
        else:
            B[off + i-1] = b_prev
            b_prev = cur - wsMult[i]·b_prev
    B[off + m-1] = b_prev
    # 回代：三项，du2' 仅 i+2 < m 参与
    x1 = B[off + m-1] / wsD[m-1] ; B[off + m-1] = x1 ; x2 = 0
    for i = m-2 .. 0:
        num = B[off + i] - wsDu[i]·x1 - (i+2 < m ? wsDu2[i]·x2 : 0)
        x0 = num / wsD[i] ; B[off + i] = x0 ; x2 = x1 ; x1 = x0
```

要点：

- **不再重复分解**：kernel B 只做"按记录回放"，省掉每列一次的主元比较与 `m` 次复除；
- 每线程的 B 列在 GM 上连续，逐行扫描是顺序访问，缓存行可整行复用；`ws*` 数组对同一 block 的所有线程是同址读取，属广播访问；
- 前向与回代都预读下一行的 `B`/因子，掩盖 GM 延迟；`m=3` 时循环退化为 2 步前向 + 1 步回代；
- 只写 `[0, m)` 行，`ldb > m` 的 padding 保持原值；`dl/d/du` 全只读。

#### 3. 复数数值细节

| 问题 | 处理 |
|---|---|
| 模长比较 | 比平方和 `re²+im²`（保序、免开方）；四分量绝对最大值 `s > 2^62` 或 `s < 2^-62` 时先按 `s` 缩放再比；任一分量 NaN → 判 false（不换行），±Inf 按正常比较处理 |
| 复数除法 | Smith 形式：`\|c\| >= \|d\|` 时 `t = d/c`、分母 `c + d·t`，否则对称；避免 `c²+d²` 上溢/下溢；分母为 0 时保持 IEEE 语义（`0/0 → NaN`、`x/0 → ±Inf`） |
| 复乘/复加 | `float2` 形式的 4 乘 2 加，全程 FP32，不升精度 |
| 奇异/病态 | 不拦零主元、不做奇异检测；Inf/NaN 按 IEEE 传播，与 `Sgtsv2` 和 cuSPARSE Legacy 语义一致，实际输出在自测报告单独记录 |
| 精度增强 | 首版不做 Kahan / 迭代修正；若 3.2 的实虚分量判定在病态用例上不达标，再评估引入一步残差修正（`arch22 Sgtsv2` 已有同类做法可参照），作为后续版本而不进首版范围 |
| 确定性 | 分解与回放都是固定顺序、无原子操作、无跨线程归约；分支只依赖输入值，相同输入复跑逐位一致 |

#### 4. 并行度与方案取舍

| 方案 | 分解次数 | workspace | 串行墙钟 | 结论 |
|---|---|---|---|---|
| A. 每 RHS 一列独立分解（`arch35 Sgtsv2` 现形态） | `n` 次 | `align128(24·m·n)`，P-03 约 22 MB | `2m` 步 | 单 kernel、最简单，但 workspace 随 `n` 线性增长，`n` 大时覆盖用例会炸 |
| B. 公共分解 + 两段式（本设计，`arch22 Sgtsv2` 同构） | 1 次 | `align128(33·m)`，P-03 约 236 KB | `3m` 步 | 内存与 `n` 无关，复除少 `n` 倍；多一次 kernel 发射 |
| C. SIMD 跨列向量化 | 1 次 | 同 B | `3m` 步 | `B` 列主序，逐行跨列访问 stride=`ldb`，需 gather/reduce；`n=1` 无并行；不选 |

选 B 的判据是 **3.4 的内存条件与 3.3 的性能门槛同时满足时的余量**：

- 内存：条件 2 要求"方案固有 workspace 绝对值 ≤ 目标硬件 L2 容量"。B 在 203 条性能用例上的 workspace 上界是 `m=8192 → 270,336` 字节，与目标硬件的 L2 容量相差两个数量级以上（具体容量以实测平台参数为准），不需要依赖某型号的具体数值就能满足；A 在 P-03 上 22 MB，在 L2 容量未知时存在被判不过的风险，且 `n` 更大的覆盖用例会到 GB 级。
- 性能：门槛是逐例 `倍率 ≥ 0.3`。B 的墙钟比 A 只多一次 `m` 步的串行分解（`3m` vs `2m`），而 203 条里余量最小的是 `n=1` 的大 `m` 用例：按每步 30~50 周期的保守估算，`m=4480` 约 270~450 μs，对标 190.9 μs 的倍率约 0.42~0.71，仍在门槛之上；`n=64/128` 的多 RHS 用例倍率估算在 1 倍量级。实测以自测报告为准，余量不足时的首选优化是把 `n=1` 走单 kernel 融合快路径（见"风险点"）。
- 不选 SIMD 的理由写清楚：这是数据布局与依赖结构决定的，不是"退而求其次"。仓内规范要求优先评估 SIMD，本方案评估过跨列向量化（方案 C），因列主序 `B` 的行访问步长为 `ldb`、每步还要跨列归约，收益为负。
- 不做 SPIKE/PCR：部分选主元下 reduced system 构造复杂、正确性风险明显，且它主要改善"单列内部并行"，而本任务的正式性能场景都是多 RHS。记为后续可选优化。

### 目录结构与文件清单

```text
include/cann_ops_sparse.h              修改：aclsparseComplex_t + 2 个声明（不动 Sgtsv2）
sparse/cgtsv2/arch35/
  cgtsv2_host.cpp                      新增：Validate / ComputeWorkspaceBytes / Launch
  cgtsv2_kernel.cpp                    新增：factor VF + solve VF + 2 个 __global__ + kernel_do
  cgtsv2_kernel.h                      新增：kCgtsv2PivotThreadsPerBlock、两个 kernel_do 声明
  cgtsv2_tiling_data.h                 新增：Cgtsv2TilingData（5 个 int32，无数组）
sparse/cgtsv2/README.md                新增：接口、约束、workspace、产品支持表
test/cgtsv2/CMakeLists.txt             新增：ops_sparse_add_gtest_tests(cgtsv2 ${OPS_SPARSE} WARN_ON_MISSING_SRC EIGEN)
test/cgtsv2/cgtsv2_golden.h            新增：complex128 带主元带状 Golden（独立实现）
test/cgtsv2/cgtsv2_param.h             新增：CSV 参数结构（m/n/ldb/matrixType/seed 等）
test/cgtsv2/README.md                  新增：环境、编译、ATK 生成/审计/运行、采集步骤
test/cgtsv2/arch35/cgtsv2_npu_wrapper.h 新增：128 字节手工对齐的 DeviceBuffer + 两阶段调用
test/cgtsv2/arch35/cgtsv2_test.cpp     新增：GTest 参数化用例 + 异常用例
test/cgtsv2/arch35/cgtsv2_test.csv     新增：分级用例表
test/cgtsv2/atk/、test/cgtsv2/performance/  新增：随包 ATK YAML/插件/清单/采集脚本
torch_npu/cgtsv2_npu.cpp               新增：ops_sparse_test 两个注册点
torch_npu/CMakeLists.txt               修改：把 cgtsv2_npu.cpp 加入 add_library
docs/zh/api_list.md                    修改：产品支持表与 workspace 描述补 Cgtsv2
```

### 构建与接入

- `sparse/` 下**不需要**新增 CMakeLists：源文件由 `sparse/CMakeLists.txt` 的 `file(GLOB_RECURSE)` 收集，再按 `ARCH_SPECIFIC_DIRS` 过滤、按 `SOC_ARCH_DIRS` 回填，`ascend950 → arch35` 自动成立。
- `test/` 下**必须**新增 `test/cgtsv2/CMakeLists.txt`：`test/CMakeLists.txt` 按 `OP_LIST` 找 `test/<op>/CMakeLists.txt`，缺文件直接 `FATAL_ERROR`。
- 构建/自测命令：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash build.sh --ops=cgtsv2 --soc=ascend950 --run
bash build.sh --ops=cgtsv2 --soc=ascend950 --run --device=0
```

端到端适配是独立可选目标：仓内 `torch_npu/` 由根 `CMakeLists.txt` 的 `BUILD_TORCH_ADAPTER` 开关（默认 `OFF`）拉起，构建时须显式打开并保证 `torch`/`torch_npu` 可导入，产物 `libops_sparse_torch.so` 由适配模块在 `import` 时用 `torch.ops.load_library` 加载；不改变 `build.sh` 的默认行为。

- 测试不允许自带 `main()`（框架固定链 `test/frame/test_main.cpp`）；CSV 由构建宏拷到可执行文件目录，代码用相对文件名打开。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR / Ascend 950DT（A5，DAV_3510，arch35） | √ |
| Atlas A2 / A3 系列（DAV_2201，arch22） | ×（本任务不实现；新增代码全部在 `arch35` 目录，与 A2/A3 解耦） |
| Ascend 310P 等其它架构 | × |

## 算子约束限制

1. 仅 `complex64`；`dl/d/du/B` 必须同类型、连续存储，`B` 为列主序 `[ldb,n]`，元素地址 `B[i+j*ldb]`；
2. `m >= 3`、`n >= 0`、`ldb >= max(1,m)`；`n = 0` 合法且不发射 kernel，`bufferSizeExt` 返回 0；
3. `n > 0` 时 `dl/d/du/B` 非空，`pBuffer` 非空且 128 字节对齐；`bufferSizeExt` 允许 `B == NULL`；
4. `dl[0] = 0`、`du[m-1] = 0` 由调用方保证，实现不校验；
5. `B` 的 padding 行 `m <= i < ldb` 必须保持不变；`dl/d/du` 只读；
6. 真正奇异系统不拦零主元，Inf/NaN 按 IEEE 传播；病态系统不因"病态"标签豁免精度判定；
7. 不引入 `batchCount`，`n` 是 RHS 列数；
8. 相同输入、相同设备与版本下复跑输出逐位一致（分解与回放均为固定顺序、无原子操作）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 以 cuSPARSE `cusparseCgtsv2` 为接口标杆；CPU Golden 用同一份 complex64 输入升 complex128 后求解。实部、虚部分别按 FLOAT32 混合容差：`\|actual-golden\| <= atol + rtol·\|golden\|`，`rtol=atol=2^-13`，两部分匹配率均 ≥ 0.99；每分量最大有限绝对误差 ≤ `max(1e-2, 32·ULP(g_low))`；±Inf 溢出按 RNE 收窄 golden 判同号；特殊值另按分类 mask 与 Inf 符号独立判定 | 任务书 3.2；[混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)（核对提交 `a5e8e716743b20bfb2ffa16e0213c4aa73b823b3`） |
| 性能标准 | 逐例性能倍率 = 标杆 `cusparseCgtsv2` Kernel 总耗时中位数 ÷ NPU 同口径 Kernel 总耗时中位数，203 条（P-01~P-03 + 200 条扩展）全部 ≥ 0.3；预热 ≥ 10 次、正式采样 ≥ 30 次，取中位数；Event 口径独立列示，不参与倍率判定 | 任务书 3.3 |
| 内存标准 | 3.4 两条择一：① 输入输出总量 > 500 MB 时额外内存 ≤ GPU 的 5%；② 方案固有 workspace 绝对值 ≤ 目标硬件 L2 容量。本设计按 ② 说明：`WS(m)=align_up(33m,128)`，203 条最大 270,336 字节 | 任务书 3.4 |

### 精度判定规则（实现按此口径，不自创阈值）

- 实部、虚部**分别**比对，任一分量不满足即该元素失败；整体匹配率按实/虚两部分各自统计，均须 ≥ 0.99。
- 双方均有限时用混合容差；非有限值不参与 `max_abs_error` 与 ULP 锚点计算。
- 溢出去特例：NPU 分量为 ±Inf、Golden 有限时，把 Golden 按 RNE 收窄为 FLOAT32，收窄后须为同号 Inf 才通过。
- 特殊值分类（任务补充语义，独立结论，不替代常规数值判定）：NaN/NaN 通过、NaN/有限或 Inf 失败；±Inf 须同位置同号；有限值位置出现 NaN 失败。NaN 不比较 payload 与符号。
- 随包 200 条 + 验收方另行生成的 800 条，两组分别运行、分别留档；`n=0` 按合法空 RHS 验证；非法参数、真正奇异系统、非有限输入用独立语义与错误码检查，不把 NaN Golden 记为常规数值通过。

### 性能判定规则

- 203 条逐例算倍率，全部 ≥ 0.3；主场景 P-01/P-02/P-03 的标杆 Kernel 中位数分别为 640.896 / 628.944 / 1220.048 μs。
- 每例预热 ≥ 10 次、正式采样 ≥ 30 次，报告单次求解 Kernel 总耗时中位数；每次原地求解前恢复原始 `B`，恢复不计入计时；workspace 在预热与采样期间复用。
- NPU 侧用 Profiler 采集"相同求解范围内所有 Kernel 耗时并求和"（本设计是两个 kernel，必须两者都计入，不能只取 factor kernel）。
- Event 口径在同一 stream 上用 Event 包围求解调用，排除 `B` 恢复，单独列示，不与 Kernel 耗时混用。
- 同步输出内存占用：`io_bytes = (3m + ldb·n)·8`、`workspace_bytes`、对齐后实际申请、`B` 恢复副本，分别统计，不重复相加。

## 自测方案

### 1. C++ UT/ST（`test/cgtsv2/arch35/`）

沿用仓内 GTest + CSV 参数化框架（`test/gtsv2/arch35/gtsv2_test.cpp` 的同构写法），自己补 `complex128` Golden（`std::complex<double>` 独立实现带主元带状消元，不复用 NPU 侧代码）与复数容差比对。`test/frame/verify.h` 的 `Verifier` 只接受 `float`/整型向量、`getMixedToleranceDefaults` 也没有 `COMPLEX64` 分支，因此**在 `test/cgtsv2` 内把实部/虚部拆成两条 float 向量**，分别套用现有 FLOAT32 混合容差与原子的 `max_abs_error/ULP` 判据；不修改 `test/frame`（避免本任务扩大公共测试框架变更面，后续有更多复数算子时再抽取通用能力）。

| 组 | 内容 | 目标 |
|---|---|---|
| L0 | `m=3/4/7/8/16/32`，单/多 RHS，identity/constant/对角占优/随机（可触发换行） | 基本功能与最小规模 |
| L1 | `m` 边界 3、`n=0/1`、`ldb=m+padding`、非 2 幂 `m=1000/1001`、多 RHS `n=32/64/128`；**全实、全虚、实虚同量级、大相位差**四类复数数据 | 形状/布局/复数覆盖 |
| L2 | 大 `m`（4096/7168/8192）、`n=1/64/128`、grid-stride 尾块（`n % 32 != 0`）、多 block | 规模与分核 |
| WB | 白盒构造主元路径：每步必换、仅首步、仅末步、极端主元比（`1e-6:1`、`1e6:1`）、零主对角但非奇异（2×2 主元块）、`du2` 填充链累积 | 主元/fill-in 分支 |
| E | 异常与契约：`aclsparseComplex_t` 的 `sizeof/alignof/字段偏移` 校验（编译期断言 + UT）、`handle=NULL`、`n<0`、`m<3`、`ldb<max(1,m)`、各空指针、`pBuffer` 未对齐、`n=0` 时 `pBuffer=NULL` 仍成功、`bufferSizeExt` 的 `B=NULL` 与返回值一致性、workspace 乘法溢出边界、`dl/d/du` 只读性逐位比较、`B` 的 padding 不变、`B` 原地输出 | 类型布局、错误码与内存契约 |

另需单独覆盖：`bufferSizeExt → 申请（含 128B 手工对齐）→ 执行 → 释放` 的完整流程；非默认 stream 与 `aclsparseSetStream` 切换后的正确性；同 stream 连续下发；`n=0` 与 `m=3` 的最小路径；畸形/极值量级输入（`1e-30`、`1e30` 系数）观察模长缩放与复除的溢出分支。

### 2. ATK 精度与覆盖自验

随包测试工程整体移入 `test/cgtsv2/`（`atk/`、`performance/`、`common/` 原样保留文件名与目录关系）：

```bash
export PYTHON_BIN=/absolute/path/to/python3
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$(pwd)/..${PYTHONPATH:+:$PYTHONPATH}"
export CGTSV2_EXTENSION_MODULE=<适配模块名>

cd test/cgtsv2/atk
atk case -f case_designs/accuracy.yaml -p atk_plugins.py -s 1028                    # 随包 200 条
atk case -f case_designs/accuracy_acceptance_800.yaml -p atk_plugins.py -s 20261029 # 额外 800 条
atk case -f case_designs/accuracy_full_coverage.yaml -p atk_plugins.py -s 1028      # 完整因子网格
atk case -f case_designs/accuracy_special.yaml -p atk_plugins.py -s 1028            # 8 条非有限
atk case -f case_designs/accuracy_contract.yaml -p atk_plugins.py -s 1028           # 12 条合同
atk task -c result/accuracy/json/all_accuracy.json -n nodes_accuracy.yaml -p . -tk accuracy -sp --disable_id_seed
```

- 200 与 800 分别生成、分别执行、分别留档；800 条不得复用 200 条结果；提交内容包含每组种子（200 条用 `-s 1028`，800 条用验收方选定种子）、原生 ATK 输出（JSON/CSV/XLSX）、覆盖审计报告、数值运行日志与未执行项清单；
- 完整因子网格共 66,432 条（38 种 `m` × 40 种正 `n` × 3 种 padding 等价类 × 16 种分布组合，经 `ldb <= 2^20`、总元素 `<= 2^34` 过滤），自测阶段先做官方审计确认元数据，再在资源允许范围内抽样执行，未执行项列明原因，不按通过计数；
- 特殊值 8 条与合同 12 条按任务补充语义独立判定；合同拒绝前缀 `CGTSV2_INVALID_PARAMETER:` 只证明适配层契约，真实 C++ 返回码与指针/对齐边界仍由 C++ UT/ST 覆盖。

### 3. 性能与内存自验

```bash
cd test/cgtsv2
"$PYTHON_BIN" performance/collect_performance.py --backend npu --metric both \
    --case-file performance/performance_cases.json --device 0 \
    --warmup 10 --samples 30 --output results_npu_all
```

- NPU 侧 Kernel 口径用 torch_npu Profiler 逐次抓取、把求解范围内的所有 Kernel 求和；Event 在独立进程用 Event 包围，两者不混用；Kernel 缺失直接失败，不回退 Event。
- 标杆数值取随包 `gpu_baseline.csv` 的 `kernel_median_us`（203 行、`case_index 0..202`、`status=ok`，与清单逐条同 `m/n/ldb/seed/distribution`），逐例算 `R_i` 并汇总；报告中注明未在本地复采 GPU、标杆文件来源提交与 SHA-256，配对依据是属性一致而非张量字节一致。
- 内存：逐例记录 `io_bytes`、`workspace_bytes`、对齐申请、恢复副本与设备 free/total；本方案按 3.4 条件 ② 说明，`WS` 公式与实测 `bufferSizeExt` 返回值须逐例一致。
- 另附 Profiler 证据：确认 Task Type 落在 AI_VECTOR_CORE、Block Num 与 `numBlocks` 预期一致、两个 kernel 都被采集到（无 CPU 回退、无单个 kernel 漏采）。

### 4. 确定性与奇异系统

- 确定性：同一输入、同一设备与版本，恢复原始 `B` 后连续执行三次，输出逐位一致（C++ UT 与上板脚本各留一份记录）；跨版本/跨设备不承诺逐位一致。
- 奇异/近奇异：单列一组用例（零主元、精确奇异、2×2 奇异块），记录实际返回码、是否崩越界、Inf/NaN 出现位置与符号，作为报告结论而非精度通过依据。

## 兼容性分析

- **接口兼容**：只新增类型与两个函数，`include/cann_ops_sparse.h` 中 `Sgtsv2/Nopivot/StridedBatch` 的声明与实现零改动；新增 `aclsparseComplex_t` 与既有 `aclsparseComplex` 布局相同但类型独立，不产生重定义冲突。
- **A2/A3 解耦与交叉回归**：新增代码全部位于 `arch35` 目录，arch22/arch20 构建不含该算子（与仓内 `spsv` 等"仅 arch35"算子同处理方式）；本地用 `--soc=ascend910b` 构建并跑既有 `gtsv2`、`spmv` 等算子用例，确认头文件追加与 `torch_npu/CMakeLists.txt` 改动不影响 A2/A3；A5 上对 `Sgtsv2` 做交叉回归（同一次构建同时编入 `cgtsv2` 与 `gtsv2`，跑 `gtsv2` 全量用例）。
- **公共 Host 冲突**：`cgtsv2` 使用新目录、不碰 `sparse/gtsv2/` 与公共 host 工具；若 A2/A3 侧后续补 `Cgtsv2`，公共逻辑可按需下沉，本 PR 不预留半成品。
- **数值兼容**：算法与 `Sgtsv2` 同构，差别仅复型算术与模长判据；与 LAPACK Golden 的主元判据差异（`CABS1` vs 模长）已在"算法与主元策略"中说明，属容差内的稳定主元差异。

## 风险与待评审确认点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| R-01 | `n=1` 等小 `n` 场景并行度只有 1 个线程，是 203 条中倍率余量最小的部分 | 性能倍率可能接近门槛 | 保守估算倍率 0.42~0.71，高于 0.3；实测若不足，按 `n==1` 走单 kernel 融合快路径（每线程内部分解 + 回放，该路径的 `bufferSizeExt` 单独返回 `align128(24m)`，仅 `n=1` 启用），或降低每步指令数 |
| R-02 | 每步链式依赖的 GM 延迟暴露 | 两个 kernel 都可能被延迟主导 | 都按 `Sgtsv2` 的先预读下一行写法实现；实测超标时计为性能问题按 R-01 的路径优化 |
| R-03 | 模长比较的平方上溢/下溢与 NaN 判定 | 极端量级输入可能选错主元或在 NaN 下行为不一致 | 量级缩放 + 专项用例（`1e-30/1e30` 主元、NaN/Inf 主元），NaN 判定与实型语义对齐 |
| R-04 | Smith 复除在极端量级下的中间溢出 | 结果非有限或精度塌陷 | 按仓内 `spsm` 的分流写法实现，专项覆盖极端量级；普通量级走快路径 |
| R-05 | CPU Golden（LAPACK `CABS1` 判据）与本实现（模长判据）主元序列可能不同 | 近相等主元用例的数值路径不同、误差分布不同 | 两者都是稳定主元策略，按 3.2 容差判定；构造"近相等主元"专项用例观察差异，报告中记录，不要求逐位一致 |
| R-06 | 病态矩阵在 FP32 下前向误差放大 | 精度判定不达标 | 2^-13 混合容差 + 0.99 匹配率已较宽；不足时评估一步残差修正（`arch22 Sgtsv2` 有同类实现），作为后续版本 |
| R-07 | 适配层 workspace 对齐/`n=0` 空 workspace 处理与 ATK 契约不一致 | 精度/性能/内存三项全部跑不起来 | 适配层显式校验 128 字节对齐、`n=0` 直接返回（C++ 侧 `n=0` 早返回先于 `pBuffer` 校验），与 `common/cgtsv2_operation.py` 的 contract 逐条对齐 |
| R-08 | 完整因子网格的大尺寸组合在单卡上无法全部执行 | 覆盖结论不完整 | 先做官方审计，再在资源允许范围内抽样执行并逐项列明未执行用例与资源需求，不把未执行记为通过 |
| R-09 | 设计阶段无 A5 实机，性能/精度均为结构估算 | 评审时缺实测数据 | 本阶段只评审方案与契约；编码阶段在 950 环境补齐实测，自测报告按模板记录实际 CANN/驱动/固件/硬件型号与提交号 |

待评审确认点：

| # | 事项 | 需要确认的内容 |
|---|---|---|
| D-01 | 任务书 2.3 的 `aclsparseComplex_t`（字段 `real/imag`）与仓内已公开的 `aclsparseComplex`（字段 `x/y`）并存 | 是否按任务书新增第二个同布局复数类型；本设计按任务书执行且不改动既有类型，若评审要求复用既有类型，需同步调整两个接口签名 |
| D-02 | `bufferSizeExt` 的 `B` 是否允许 `NULL` | 本设计沿用 `Sgtsv2` 语义允许；若验收要求 `n>0` 时必须非空，需改动查询接口校验 |
| D-03 | 额外 800 条精度用例的生成种子 | 由验收方选定；自测按示例种子 `20261029` 生成留档，是否与验收口径一致 |
| D-04 | 性能标杆是否接受随包 `gpu_baseline.csv` 的 Kernel 中位数 | 本环境未提供 GPU 复采条件；若必须两端现场采集，需额外 GPU 环境与 `ncu` 权限 |
| D-05 | 完整因子网格的抽样执行范围 | 大 `m`/`n` 组合的资源边界与"未执行项"的接受方式 |

## 交付物清单

| 类别 | 内容 |
|---|---|
| 算子实现 | `include/cann_ops_sparse.h`（类型 + 2 个声明）、`sparse/cgtsv2/arch35/` 四件套、`sparse/cgtsv2/README.md` |
| 测试代码 | `test/cgtsv2/` 公共 Golden/参数/README + `test/cgtsv2/arch35/` C++ UT/ST + 随包 ATK/性能附件；`torch_npu/cgtsv2_npu.cpp` 与可导入适配模块 |
| 设计文档 | 本文件（`04_tasks/01_community-task-2026/tasklist/10月社区任务-aclsparseCgtsv2算子开发(950)/kevin_lee1231/docs/design.md`） |
| 自测报告 | 任务书第 4 节 `task_submission/` 七件套：自验证步骤说明、精度/性能/内存报告与日志，含环境版本、原始 Profiler/内存证据、失败项与未执行项说明 |
| 代码地址 | ops-sparse 个人仓分支与目录，PR 合入 `cann/ops-sparse:master` |

## 参考资料

1. 任务书：`10月社区任务-aclsparseCgtsv2算子开发(950)` → `aclsparseCgtsv2_A5_task_doc.md`
2. NVIDIA cuSPARSE gtsv2 系列接口：https://docs.nvidia.com/cuda/cusparse/index.html
3. ops-sparse 仓与贡献指南：https://gitcode.com/cann/ops-sparse 、https://gitcode.com/cann/ops-sparse/blob/master/CONTRIBUTING.md
4. 精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ATK：https://gitcode.com/Ascend/ATK （26.9.8，源码提交 `9504bee6076820ca74259fb919a790a12ef88cf8`）
6. Ascend C 算子开发与 API 文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
