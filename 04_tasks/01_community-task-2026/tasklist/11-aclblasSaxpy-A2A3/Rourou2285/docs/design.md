# aclblasSaxpy（A2/A3）设计文档

## 1. 任务与范围

本任务在 `ops-blas` 的 arch22 构建路径中新增
`aclblasSaxpy`，为 Atlas A2/A3 提供单精度 BLAS Level-1
向量原地更新能力：

```text
y[i] = alpha * x[i] + y[i]
```

代码 PR：
<https://gitcode.com/cann/ops-blas/pull/936>

实现分支：`task/aclblasSaxpy-arch22`

本次范围包括公共 Host 接口、arch22 Kernel、GTest、CSV 精度与性能用例、
GPU baseline、算子 README 和完整自验证材料。

## 2. 需求背景与标杆

语义对齐 cuBLAS `cublasSaxpy` 和 Netlib BLAS `saxpy.f`。

- `alpha` 是 Host 标量指针。
- `x` 和 `y` 是 Device 向量指针。
- `incx`、`incy` 支持正负非零步长，不支持
  `INT32_MIN`。
- `n <= 0` 为合法 no-op，不访问任何标量或向量参数。
- `alpha == 0` 为合法 no-op，不读取 `x`，`y` 保持不变。
- `y` 原地更新；`x` 不修改。

精度 golden 使用 Netlib BLAS 3.10.0。FLOAT32 判据为
`rtol=atol=2^-13`、`matched_ratio>=0.99`，并应用任务书规定的
逐元素最大误差限制。

性能标杆使用任务提供的 GPU baseline。正式验收设备为
Atlas 800T A2 910B3，验收判据为 NPU 平均耗时不超过 GPU 的
1.25 倍。

## 3. 接口与约束

```cpp
aclblasStatus_t aclblasSaxpy(
    aclblasHandle_t handle,
    int n,
    const float* alpha,
    float* x,
    int incx,
    float* y,
    int incy);
```

Host 校验顺序：

1. `handle == nullptr` 返回 `HANDLE_IS_NULLPTR`。
2. `n <= 0` 返回 `SUCCESS`。
3. `alpha`、`x`、`y` 为空返回 `INVALID_VALUE`。
4. `incx` 或 `incy` 为 0 返回 `INVALID_VALUE`。
5. `incx` 或 `incy` 为 `INT32_MIN` 返回 `INVALID_VALUE`。
6. `(n-1)*abs(inc)` 超出 int32 元素跨度约定时返回
   `INVALID_VALUE`。
7. `alpha == 0` 返回 `SUCCESS`。

接口保持异步语义。成功返回表示任务已提交到 handle stream，
不表示 Kernel 已完成。

## 4. 总体架构

```text
aclblasSaxpy (Host)
  -> 参数与跨度校验
  -> alpha 同步解引用
  -> 计算 tiling 与 active block 数
  -> 选择 dense / strided kernel
  -> 提交到 handle stream 并返回
```

不分配 tiling workspace，不产生 tiling H2D。`n`、block 数和
`alpha` 直接作为 Kernel 标量参数下发。

## 5. Host 设计

### 5.1 分核策略

- AIV 核数首次查询后缓存在 handle 中，stream 切换时失效。
- 最多使用设备可用的 48 个 AIV block。
- 小规模输入保证每个 block 至少 8192 个元素，减少无效启动。
- 各 block 按连续逻辑下标均分，剩余元素分配给前若干 block。

### 5.2 正步长

当 `incx > 0` 且 `incx != 1` 时，使用异步 2D Device copy 将
strided `x` 打包为连续布局。`y` 采用相同策略。

正步长计算后，再使用异步 2D Device copy 将 `y` 散回原地址。
scratch 由 handle 持有并按 stream 顺序复用；只有容量增长时同步
handle stream 后重新分配。

### 5.3 负步长

任一步长为负时，选择 device-side indexed kernel 直接按逻辑下标
计算物理地址。该路径不分配 scratch，避免 host 为每个元素下发
一条异步 device copy 指令。

## 6. Kernel 设计

### 6.1 Dense Kernel

适用条件：`incx == 1 && incy == 1`。

每个 AIV block 使用 6 槽 ping-pong UB 布局：

| 区域 | 容量 |
| --- | ---: |
| x ping/pong | 2 × 30 KiB |
| y ping/pong | 2 × 30 KiB |
| output ping/pong | 2 × 30 KiB |

每个 tile 为 7680 个 FLOAT32 元素。下一 tile 的 MTE2 装载与当前
tile 的向量计算、MTE3 回写重叠。

计算使用：

```text
out = vmuls(x, alpha)
out = vadd(y, out)
```

输出使用独立 UB 区域，避免 C220 上不可靠的输入输出原地别名。

### 6.2 Strided Kernel

对负步长和正负混合步长：

```text
xPhysical = incx > 0
    ? logical * abs(incx)
    : (n - 1 - logical) * abs(incx)

yPhysical = incy > 0
    ? logical * abs(incy)
    : (n - 1 - logical) * abs(incy)
```

逻辑元素之间没有依赖，因此可由多个 AIV block 独立处理。

## 7. 可维可测性

新增代码：

```text
blas/axpy/arch22/
test/saxpy/arch22/
```

新增测试资产：

- 5 条显式 API 契约测试。
- 1000 条 CSV 精度与边界用例。
- 200 条性能用例。
- 任务提供的 GPU baseline。
- arch22 CMake GTest 注册。

CSV 覆盖 n 快速返回、对齐边界、正负和混合步长、Inf/NaN、
`alpha=0`、大尺寸和 span 溢出负例。

## 8. 验证结果

环境：

- Ascend A3
- Ubuntu 22.04.5
- CANN 9.1.0
- 1 NPU

构建：

```bash
bash build.sh --soc=ascend910_93 --ops=saxpy
```

结果：

- 构建：PASS
- 精度与 API 契约：`1005/1005 PASS`
- 性能：`200/200 PASS`
- 全量：`1205/1205 PASS`
- GPU/NPU 性能倍率均值：`2.5751`
- GPU/NPU 性能倍率中位数：`2.5635`
- 最差 GPU/NPU 倍率：`1.6701`
- 1M/2M/4M 元素平均耗时：`4.46/6.39/10.47 us`

## 9. 已知限制与后续

- 最终实现仍需在正式 Atlas 800T A2 910B3 上执行性能和功能复测。
- 先前 A2 虚拟实例 `910B3vir05_1c_16g` 的 1005/1005 精度结果
  不能替代正式 A2 性能结论。
- 上游 PR 仅包含代码、公共 README 和测试资产；
  `task_submission` 报告保存在个人 Fork 的独立分支。
