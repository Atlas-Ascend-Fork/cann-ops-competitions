# aclblasSrot 算子设计文档

> 提交目录：cann-competitions `04_tasks/01_community-task-2026`（按社区任务设计文档模板）

# 需求背景（required）

## 需求来源

11月社区任务：在昇腾 NPU（Atlas A2/A3 系列产品）上基于 Ascend C 编程语言开发单精度实数（FLOAT32）Givens 平面旋转算子 `aclblasSrot`，对齐 cuBLAS `cublasSrot`（CUBLAS_Library.pdf §2.5.8）能力，验收通过后合入昇腾算子开源仓 ops-blas（算子目录 `blas/rot/arch22/`，测试目录 `test/rot/srot/arch22/`）。

## 背景介绍

### 算子定位

`aclblasSrot` 为 BLAS-1 双向量原地旋转：对每个逻辑元素 i（0-based，i = 0..n−1）施加 Givens 平面旋转 G = [[c, s], [−s, c]]：

```
x'[px] = c·x[px] + s·y[py]
y'[py] = c·y[py] − s·x[py]        两式右端均取更新前旧值（先读后写）
px = (incx>0) ? i·incx : (n−1−i)·(−incx)      负步长自物理 span 尾端起算（Netlib 语义）
py 同理
```

n=0 为 no-op（不访问向量、不启动 kernel）；NaN/Inf 不拦截，按 IEEE 规则自然传播。

### 主仓现状

ops-blas 主仓已有 950PR 实现（`blas/rot/arch35/`）与同族复数版 arch22 骨架（`blas/rot/arch22/csrot`）；接口声明已存在于 `include/cann_ops_blas.h`（与 arch35 共用），不新增平台私有接口。`blas/rot/README.md` 的 per-arch 差异表为本任务契约依据：arch22 对 n<0、零步长、INT32_MIN 步长返回 INVALID_VALUE（arch35 放行，已声明的跨架构差异非缺陷）。

### 性能验收基线

任务书 §3.3：`gpu_baseline.csv` 200 行（cublasSrot@A100-SXM4-80GB/CUDA 12.8，单 case 6.93~48.83µs）；判据为单 case 平均耗时 ≤ GPU 标杆 1.25×（≥0.8× 标杆），msprof op 模式 task duration，warmup 5 + 有效采样 ≥10 次取均值。n ∈ [1, 4194304]；实测 200 条性能用例全部 incx=incy=1，**连续路径为唯一性能敏感路径**，strided 路径仅被精度用例触达。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（DAV_2201）上以 kernel 直调（Ascend C）方式实现 `aclblasSrot`，接口签名与主仓 `include/cann_ops_blas.h` 完全一致：

```cpp
aclblasStatus_t aclblasSrot(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy,
    const float* c, const float* s);
```

x/y 为 Device FLOAT32 向量原地更新，不返回视图；c/s 为 FLOAT32 标量指针，Host 或 Device 内存均可（运行时经 `aclrtPointerGetAttributes` 逐指针判定，二者可不同侧）。异步执行经 handle 绑定 stream 下发，读回 Device 结果前须同步。

## 需求拆解

1. **功能**：双向量原地 Givens 旋转；正负非零步长；负步长自物理 span 尾端起算
2. **异常与 no-op 语义**（Host 接口层校验，固定顺序）：handle null → HANDLE_IS_NULLPTR（最高优先）；n=0 → SUCCESS 快速返回（不校验其余指针、不启动 kernel）；n<0 / n>0 时 x/y/c/s nullptr / 步长 0 或 INT32_MIN / (n−1)·|stride| 超 int32（Host int64 预判）→ INVALID_VALUE；c/s 标量值不校验
3. **精度**：golden = cblas_srot float32（=仓内 `test/rot/srot/srot_golden.h`，Netlib srot.f 语义）；混合容差 rtol=atol=2⁻¹³；matched_ratio ≥ 0.99；max_abs_error ≤ max(1e-2, 32×ULP)；x/y 分别逐元素判定
4. **性能**：200 条性能 case 逐条 ≤ 1.25× GPU 标杆
5. **交付**：`blas/rot/arch22/`（算子）+ `test/rot/srot/arch22/`（CSV 驱动 GTest，覆盖任务配套 1200 条用例 + GAUSS 正态补充）

## 任务书 §2.5 约束逐项核对结论

| 约束项 | 核对结论 | 实现落点 |
|---|---|---|
| 参数合法性与校验层级 | ✅ 固定顺序校验链全 Host 侧，被拒调用不启动 kernel | srot_host.cpp ValidateSrotInputs |
| 非连续 Tensor 支持 | ✅ BLAS 步长语义即非连续访问，正负步长均生效 | kernel strided 路径 |
| broadcast 规则 | 不涉及：双向量元素个数同为 n，无操作数间广播 | — |
| dynamic shape 要求 | ✅ n/incx/incy 运行时入参，任意合法规模一次适配，无需图模式动态 shape | host tiling 填充 |
| 原地与视图语义 | ✅ x/y 原地更新（先读后写，第二式用旧 x），不返回视图 | kernel 双路径写回次序 |
| 确定性计算要求 | ✅ 逐元素运算，单次执行内确定；同输入同步长跨次执行 bit 级一致 | — |
| 空 Tensor 处理 | ✅ 等价口径为 n=0 快速返回，无空 Device 指针参与计算的路径 | host quick return |
| 0 维 Tensor 处理 | 不涉及：向量算子无 0 维输入 | — |
| 异步执行 | ✅ 经 handle 绑定 stream 下发，返回成功仅表示已提交；API 内部无 aclrtSynchronizeStream | host 下发后立即返回 |

# 详细设计（required）

## 算子分析

### 数学公式

公式与物理下标见背景介绍。**舍入结构（精度关键设计）**：精度 golden（REFBLAS 解析为 Netlib libblas 3.10.0 `cblas_srot`）的实测舍入语义为「c 腿 FMA 融合、s 腿先舍入」（反汇编 `srot_` 快循环与 strided 循环一致）：

```
newX = fl(exact(c·x) + fl(s·y))
newY = fl(exact(c·y) − fl(s·x))
```

两腿全 FMA、全逐操作舍入、无收缩朴素标量式在 c=s=1e10 对消区（x≈y / x≈−y）均产生 ULP(5e10)=8192 级绝对误差、超出 max(1e-2, 32×ULP) 上限（位级实验 16-19 元素越限）。因此两条腿统一采用「Muls(s 积，先舍入) + Axpy(c 积，FMA 融合)」构造，与 golden 逐位一致（数值实验验证）；`Axpy` 单次舍入语义以 c=s=1e10 用例作为冒烟实证闸门。

### 支持数据类型

FLOAT32（x/y/c/s 全 fp32，中间计算不降精度，无 Cast）。

### 支持形状

一维向量，物理长度 1+(n−1)·|inc|；(n−1)·|stride| 超 int32 在 Host 侧以 int64 预判拒绝；步长非零且非 INT32_MIN。n 支持 0、1 及任意合法规模（Host tiling 运行时适配）。

## 算子实现

### 实现方案

#### host 侧设计

Host 侧完成全部校验、c/s 双源解析、tiling 生成与 kernel 下发，Device 侧只做计算。

**1. 参数校验**：固定顺序校验链（handle null → n==0 quick return → x/y null → n<0 → c/s null → 步长 0/INT32_MIN → 跨度溢出 int64 预判），返回仓内既有状态码。

**2. c/s 双源解析**：`CheckPtrLocation`（aclrtPointerGetAttributes）逐指针判定；host 指针解引用一次填 tiling 标量并转发 nullptr；device 指针原样转发 GM_ADDR、tiling 填 0.0f 占位（kernel 从 GM 读标量，无 D2H 拷贝、无同步）；查询失败返回 INVALID_VALUE。

**3. 动态核数**：`GetAivCoreCount()` 运行时获取，`numBlocks = clamp(核数, 1, n)`（issue #716：禁止硬编码；A2 40 AIV / A3 48 AIV 自适应）。

**4. tiling 下发**：tiling 28 字节（totalN/cosValue/sinValue/cIsDevice/sIsDevice/incx/incy）写入 handle workspace 固定偏移（sdot 先例，免每次调用 malloc/free）；拷贝为同步 `aclrtMemcpy`（host 栈对象生命期安全；仓内 async 拷贝先例均配 pinned host 源与 event 生命周期管理，不适用于本处 28B 栈对象）。

**5. 异步返回**：kernel 经 handle stream 下发后立即返回 SUCCESS。

#### kernel 侧设计

单 kernel 入口（AIV-only），kernel 内按 `incx==1 && incy==1` 分派两条路径。

**contiguous 路径**（性能敏感路径，性能用例全为单位步长）：

| 设计项 | 方案 |
|---|---|
| 多核切分 | 逻辑元素维均分，前 n%numBlocks 核各 +1（kernel 侧 GetBlockIdx/GetBlockNum 计算，无 per-core 数组） |
| UB 切分 | tile = 5120 floats（20KB，32B 整除）；8×20KB = 160KB ≤ 192KB（x/y/newX/newY 各双缓冲，余量 32KB） |
| 流水结构 | 手工 ping-pong 四段流水（MTE2→V→MTE3，互异静态 EVENT_ID0/1，csrot 同款事件结构）：CopyIn → Compute（muls_v×2 → PIPE_BARRIER(V) → Axpy×2）→ CopyOut |
| 原地安全 | 原始 x/y 在 UB 输入缓冲存活至两条旋转腿全部算完，CopyOut 阶段才覆写 GM |
| 尾块处理 | 按实际 count 精确计算与搬运（DataCopyPad，GM 侧仅 1B 对齐要求，per-core 起点任意） |

**strided 路径**（非单位步长，仅精度用例触达，不做性能投入）：每核 1024 元素/pass：标量 GetValue 读入连续 staging → 向量腿构造（与 contiguous 路径同款 Muls+Axpy，保持舍入结构；ccec 在 arch22 aicore 代码拒绝 fmaf/double，故复用向量构造）→ 标量散布到 32B 对齐 slot → 每元素 4B DataCopyPad burst 写回；PIPE_BARRIER(ALL) 排序标量 UB 写与 V pipe / MTE3 读；负步长物理下标 (n−1−i)·(−inc) 尾端起算。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 800I/T A2（DAV_2201，含 Ascend910B3） | √ |
| Atlas A3 系列（同 DAV_2201 架构，含 Ascend910_9382） | √ |

## 算子约束限制

- dtype 仅支持 FLOAT32（任务书指定）
- `x == y` 跨向量别名（且 incx≠incy）不在契约内：任务书未定义，cuBLAS 与仓内 arch35 同口径
- golden 舍入语义依赖 REFBLAS 解析结果：精度验收以 Netlib libblas 3.10.0（c 腿融合/s 腿先舍入）为快照；不同 BLAS 库（如逐步舍入实现）语义可能不同，更换精度验收环境前需重新锁定 golden 快照
- c/s 标量值不校验（含 Inf/NaN，按 IEEE 规则自然传播）
- 性能自验环境为 A3（Ascend910_9382）：hidevlab A2 算力申请未获得；A2 与 A3 同为 DAV_2201 架构（同频 1800MHz、同 L2 192MB、同内存速率模型，AIV 核数 40 vs 48），本算子为带宽受限流式负载，预期平台间差异有限；与任务书 910B3 口径的差异在自测报告中如实分列

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 混合容差 rtol=atol=2⁻¹³；matched_ratio ≥ 0.99；max_abs_error ≤ max(1e-2, 32×ULP)；golden = cblas_srot float32；x/y 分别逐元素判定；1200 条 CSV 用例全 PASS（含负向状态码断言） | 任务书 §3.2（生态算子开源精度标准 mixed_tolerance_standard.md FLOAT32 档） |
| 非有限值 | 同 NaN、同号 Inf 判过；NaN↔±Inf 等效；非有限值不计入 ratio 统计 | 任务书 §3.2 |
| 性能标准 | 200 条性能 case 单 case 平均耗时（msprof op task duration，warmup 5 + ≥10 有效样本均值）≤ GPU 标杆 1.25× | 任务书 §3.3 |

**CSV 契约适配声明**：任务包 `srot_test.csv` 按 arch35 语义生成，含 392 条（7 条 n<0 + 385 条 n>0 零步长）用例期望 SUCCESS，与任务书 §2.4/§2.5 的 arch22 契约（n<0、零步长 → INVALID_VALUE）直接矛盾。arch22 侧对这 392 条做确定性适配（仅改 expect_result 列，1000+200 条数保持），依据为任务书 arch22 契约与 `blas/rot/README.md` per-arch 差异表——按验收契约这些行的期望值是错的，适配是义务而非许可；原 CSV 与变换脚本留档可复现。性能 200 条（全单位步长）不受影响；适配后恰好补齐任务书 §3.5 要求的零步长负向校验覆盖。

**负向缓冲规则**：期望非 SUCCESS 的用例不按 (n−1)·|stride| 预分配输入缓冲（INT32_MIN 与溢出组合在 int 域即溢出），以单元素 dummy 缓冲占位，由接口在读取数据前返回预期错误码。

**正态分布补充**：随包 `gen_csv.py` 默认仅均匀分布；按任务书 §3.5-3 以 `--dist mixed` 织入 GAUSS_NORM_5_2（μ∈[−5,5]、σ∈[0.1,2]）正态用例（固定 seed 可复现），数量与覆盖统计写入自测报告。

## 兼容性分析

接口签名与主仓 `include/cann_ops_blas.h` 既有声明逐参数一致，零改动；不修改 arch35 实现/测试与顶层共享 golden（`srot_golden.h`/`srot_param.h`），arch35 契约原样保留，per-arch 差异不外溢。测试目录按主仓惯例置于 `test/rot/srot/arch22/`（family/算子/arch 三层，csrot 同款），CMake 注册复用 `ops_blas_add_gtest_tests`。共享测试框架 `test/frame/csv_loader.h` 的 `ACL_HIFLOAT4` 枚举自 CANN ≥ 9.2 才存在（9.1.0 的 `acl_base_rt.h` 枚举止于 ACL_HIFLOAT8 = 34），无条件编译导致 CANN 9.1.0 上所有 CSV 驱动测试目标编译失败（经既有算子 chemm 对照构建证实，属仓内预存问题，与本任务新增代码无关）；本次以注释守卫该两条目（仓内无 CSV 使用 HIFLOAT4 token），最低支持 CANN 版本定义该枚举后恢复。
