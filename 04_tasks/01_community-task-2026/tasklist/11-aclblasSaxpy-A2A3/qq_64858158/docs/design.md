# aclblasSaxpy 设计说明

## 需求背景（必选）

### 需求来源

为 Atlas A2/A3（arch22，含 ascend910b 系列、ascend910_93）、CANN 9.1.0 及以上环境补齐
单精度实数向量运算接口 `aclblasSaxpy`，对标 cuBLAS `cublasSaxpy`，覆盖任务给出的
1,200 条 CSV 用例与 200 条 A100 性能基线。

### 背景介绍

AXPY 是 BLAS Level-1 基础算子，按下标步长执行原地点加运算：

```text
y[i] = alpha * x[i] + y[i]    i = 0 .. n-1
```

`incx/incy` 为带符号步长：正值自前向后访问，负值按 Netlib 语义从物理跨度的尾端
反向遍历；`n <= 0` 与 `alpha == 0` 为合法快速返回。实现合入 `blas/axpy/arch22/`，
与同目录 `aclblasCaxpy` 共用工程骨架，接口声明沿用 `include/cann_ops_blas.h`
既有定义，不另立平行接口。

## 需求分析

### 需求描述

- 参数语义、返回码与快速返回口径对齐 cuBLAS 官方文档与 Netlib `saxpy.f`。
- 支持正负步长、任意合法规模；`alpha` 为 Inf/NaN 时按 IEEE 规则自然传播。
- 在 Atlas 800T A2（910B3）真机上满足任务混合容差精度规则，并达到 A100
  标杆 0.8 倍以上性能。
- 不改变公开接口、arch35（950PR）既有实现与其他算子行为。

### 需求拆解

1. Host 侧完成参数校验（含步长跨度溢出的 int64 计算）、Netlib 快速返回
   （`n <= 0`、`alpha == 0`）与按值下发的 tiling 构造。
2. 连续布局（`incx == incy == 1`）使用多核 ping-pong 流水线加原子加存储。
3. 非连续布局使用硬件 2D 间隔拷贝 gather、blocked UB 布局与原子 scatter。
4. 测试工程按仓内 CSV + GTest 框架新建 `test/saxpy/arch22/`，覆盖 1,000 条
   精度/边界/负向用例、200 条性能用例并输出 `[PERF]` 计时行。

## 详细设计

### 算子分析

输入、输出均为 FP32。`alpha` 为 Host 内存标量指针；`x` 只读（头文件中为非
`const` 指针，实现不写入）；`y` 原地覆写。公开接口只有一种 dtype，无组合分派。
合法调用满足：`n` 任意 int32，`n > 0` 时 `alpha/x/y` 非空、`incx/incy` 非零且
非 `INT32_MIN`、`(n-1)*|inc|` 不超出 int32 可表示范围。

### 接口定义

```cpp
aclblasStatus_t aclblasSaxpy(
    aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy);
```

签名与 `include/cann_ops_blas.h` 既有声明一致（与 arch35 实现共用）。

### Host 实现方案

- 校验顺序：`handle` 判空（`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）→ `n <= 0`
  快速返回成功（不访问 `alpha/x/y`）→ `n > 0` 时 `alpha`、`x`、`y` 判空、
  `incx/incy` 为 0 或 `INT32_MIN`、`(n-1)*|incx|` 或 `(n-1)*|incy|` 超出
  int32 范围（host 侧 int64 计算，issue #780/#866 口径）均返回
  `ACLBLAS_STATUS_INVALID_VALUE`。
- `*alpha == 0.0f` 快速返回成功，`y` 不变且不读取 `x`（Netlib
  `IF (SA.EQ.0.0) RETURN`）；NaN 与 0.0 比较为假，不触发该路径。
- 分块数取 `min(GetAivCoreCount(), 40, n)`，40 为同目录 `aclblasCaxpy` 的
  `DEFAULT_VECTOR_NUM` 口径。
- `SaxpyTilingData{totalN, activeBlocks, alpha, incx, incy}` 以 kernel 入参按值
  下发（`sgbmv` arch22 先例），随 launch 参数块进入流序，无需 workspace。
- `incx == 1 && incy == 1` 走连续 kernel，其余走步长 kernel。

### Kernel 实现方案

连续路径（`saxpy`）：

- 每个向量核（AIV）认领一段互不重叠的连续元素区间，按 38 KiB（9,720 float）
  瓦片循环；瓦片槽位 `ubX[2]` 双缓冲，装载下一瓦片与当前瓦片计算重叠。
- 计算仅一次 `vmuls`（`ubOut = alpha * ubX`），存储侧启用
  `set_atomic_add + set_atomic_f32`，以原子读改写把 `y += alpha*x` 折叠进
  MTE3 写路径——完全省去 y 的装载级与第二个 UB 槽位。
- 各瓦片/各核写地址互不重叠，单次执行结果确定，且与 cblas
  `SA*SX + SY` 的两级舍入结构一致（真机逐位一致）。

步长路径（`saxpy_strided`）：

- MTE 2D 间隔拷贝的 UB 侧步进粒度是 32 字节块、GM 侧步进是字节，因此
  gather 采用 blocked UB 布局：每个元素独占一个 32 字节块（元素 j 落在
  `ub[j*8]`），4 字节 burst 沿 GM 侧 `srcGap` 硬件跳步。
- 计算为原位 `vmuls` 分块缩放（repeat 为 8 位字段，按 ≤255 repeat 分块），
  仅 lane-0 通道携带有效数据，其余通道为 scatter 永不读取的垃圾值。
- 存储侧以原子 2D scatter 写回：burst 读取每 32 字节块的 lane-0，按
  `dstGap`（字节）写入步长槽位。
- 同号步长下 gather 与 scatter 的逻辑序一致（++升序、--降序），直接配对；
  混合符号时 x 与 y 的逻辑序相反，以标量 lane-0 反转对齐（`PIPE_S`
  事件与 MTE2/V 管线正确排序，对齐 caxpy 标量路径口径）。
- 瓦片 4,096 元素（128 KiB blocked 缓冲），单次调用内 burst 数不超过
  硬件 nBurst 上限 4,088，超出自动分批。

### 支持硬件

- Atlas A2 训练系列产品（ascend910b 系列，自验覆盖 910B3）
- Atlas A3 训练系列产品（ascend910_93，同 arch22 编译）
- CANN 9.1.0 及以上

### 约束（任务书 §2.5 逐项核对结论）

| 约束项 | 结论 |
|---|---|
| 参数合法性与校验层级 | 已实现：handle/n≤0/判空/步长零与 INT32_MIN/跨度溢出，口径见 Host 实现方案 |
| 非连续 Tensor 支持 | 支持：BLAS 步长语义按 incx/incy 间隔访问，正负步长均生效 |
| broadcast 规则 | 不涉及：单向量族逐元素运算 |
| dynamic shape 要求 | 不要求：n/incx/incy 为运行时入参，tiling 按运行时 n 计算 |
| 原地与视图语义 | y 原地覆写，不返回视图；x 只读 |
| 确定性计算要求 | 不要求：单次执行内确定，同输入跨次执行逐位一致 |
| 空 Tensor 处理 | n=0 走快速返回，不启动 kernel |
| 0 维 Tensor 处理 | 不涉及：向量算子无 0 维输入 |
| 异步执行 | 经 handle 绑定 stream 异步下发；alpha 为 Host 标量，host 侧解引用后随 tiling 下发 |

### 已知限制与规避

- 步长路径的混合符号场景使用标量 lane-0 反转，成本随元素数线性增长；
  性能验收 200 条用例全部为连续访存，不受影响。
- `(n-1)*|inc|` 限制在 int32 范围内以保证 kernel 侧 32 位物理寻址；超限
  调用返回 `ACLBLAS_STATUS_INVALID_VALUE`，测试侧按负向用例核对返回码。
- 与 BLAS 惯例一致，假设 x 与 y 存储不重叠；重叠输入的求值顺序未定义。

## 可维可测分析

### 精度与性能标准

- 精度：任务 CSV 1,200 条（1,000 精度/边界/负向 + 200 性能）全部执行，
  另有 `NullHandle`/`NullAlpha` 两条独立 GTest；连续路径逐位一致，步长
  路径按 FLOAT32 混合容差（rtol=atol=2^-13、matched_ratio≥0.99、
  max_abs_error≤max(1e-2, 32*ULP)）判定。真机（910B3）实测
  **1,202/1,202 通过**。
- 性能：TC_PF 用例按预热 5 次、连续 10~100 次下发的事件夹逼取平均并输出
  `[PERF] case=... avg_us=` 计时行。对标 A100-SXM4-80GB 基线，判据
  NPU 耗时 ≤ gpu_ms / 0.8；实测 **200/200 达标**，最差 1.94 倍（n=128
  小尺寸），n≥1M 稳定在 2.2~3.5 倍（n=4194304：9.8 us vs 34.6 us）。
- 复现：`bash build.sh --ops=saxpy --soc=ascend910b3` 后运行
  `build/test/saxpy/saxpy_test`（GTest CSV 驱动，1,200 条用例；
  TC_PF 用例输出 `[PERF]` 计时行，与 `gpu_baseline.csv` 比对判定）。

### 兼容性

- 变更限定于 `blas/axpy/arch22/` 新增实现、`test/saxpy/arch22/` 新增测试、
  `test/saxpy/CMakeLists.txt` 的 arch22 GTest 注册与 `blas/axpy/README.md`
  支持矩阵更新，不改变 API ABI 与 arch35 行为。
- `test/frame/csv_loader.h` 的 `ACL_HIFLOAT4` 枚举表项增加 `#ifdef` 守卫：
  该枚举仅存在于新版 acl 头文件，CANN 9.1.0 工具链下原先会导致全部
  GTest 目标编译失败；守卫对新旧工具链均向后兼容。
