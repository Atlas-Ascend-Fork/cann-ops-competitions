# aclblasIsamin 算子设计文档（Atlas A2/A3）

> 任务：11 月社区任务 `aclblasIsamin`（A2/A3）
> 目标源码仓：`cann/ops-blas`
> 设计目标路径：`blas/iamin/arch22/`
> 测试目标路径：`test/isamin/arch22/`
> 提交人：`quipiter`

本文冻结接口语义、实现边界和验收方法。本文不把 GPU 参考值或未运行的 NPU
结果当作性能结论；正式性能结论以 Atlas 800T A2（910B3）上的任务书流程为准。

## 一、需求背景

### 1.1 需求来源

任务资料包为 `aclblasIsamin 算子开发（A2A3）`，资料包 SHA256 为
`9bc52fd9abf56e939ba98e0267539f71a1a404b0928ddc5d04bd66d4d51f7a0e`。
目标是在 ops-blas 句柄式 BLAS 框架中补齐 Atlas A2/A3 的 `arch22` 实现，
保留现有 Atlas 950 `arch35` 路径。

### 1.2 接口和结果

```cpp
aclblasStatus_t aclblasIsamin(
    aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

对逻辑元素 `x[i * incx]`（`0 <= i < n`）计算绝对值最小的元素下标，结果为
从 1 开始的索引。最小值相同时返回最先出现的元素。NaN 不参与最小值比较；当
所有元素均为 NaN 时，结果保持为 0。输入和结果均为 `float`/`int`，没有隐式
精度转换或其他后端调用。

### 1.3 资料和公开用例

资料包包含 1000 条精度用例和 200 条性能用例，共 1200 条；`TC_PF` 用例保留
在公开 CSV 中用于性能验收。实现和测试不按 case 名称、行号或完整公开形状集合
分派，CSV 只由测试驱动读取。

## 二、契约与边界

| 条件 | 行为 |
| --- | --- |
| `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` 或 `result == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` | 将 `*result` 置 0，返回 SUCCESS |
| `incx < 1` | 按任务书 quick return，将 `*result` 置 0，返回 SUCCESS |
| `n > 0` 且 `incx >= 1` 且 `x == nullptr` | 返回 INVALID_VALUE |
| `(n - 1) * incx > INT32_MAX` | 返回 INVALID_VALUE |
| 其他合法输入 | 返回最小绝对值的 1-based 索引 |

`result` 在 quick return 前仍必须非空；`x` 只在实际需要读取时检查。物理输入
长度为 `(n - 1) * incx + 1` 个 `float`，步长按元素计数。负步长属于 quick
return，不会被当成反向访问。所有公共参数都在 Host 侧完成检查后才申请工作区或
发射 kernel。

## 三、详细设计

### 3.1 Host 侧

Host 实现位于 `blas/iamin/arch22/isamin_host.cpp`，只使用句柄绑定的 stream：

1. 按上表顺序检查句柄、维度、结果指针、quick-return 和步长乘积溢出。
2. 根据 `n` 和设备 AIV 数量选择 `useCoreNum = min(n, coreCount, 40)`。
3. 将逻辑区间均匀分成连续分片，写入 `startOffset[]` 和 `calNum[]`。
4. 申请一段设备工作区：每个参与核心一个 `float` 局部最小绝对值和一个
   `int32` 局部逻辑下标；另申请设备 tiling 数据并复制到设备。
5. 在 `handle->stream` 发射 AIV-only kernel，同步 stream，释放本次调用的设备
   工作区并返回 ACL 错误映射。

quick return 只对结果做异步清零，不读取 `x`，也不申请 kernel 工作区。所有地址和
   字节数计算使用宽整数；工作区申请或 stream 同步失败会返回对应的分配/内部执行
   错误，不改走 CPU、TBE、vendor 或其他架构路径。

### 3.2 Kernel 侧

`blas/iamin/arch22/isamin_kernel.h` 定义 `IsaminKernel`，由
`isamin_kernel.cpp` 编译为 Ascend C kernel。每个 AIV：

1. 从 tiling 读取 `n`、`incx`、本核的逻辑起点和元素数；
2. 按 `xGM[logicalIndex * incx]` 读取元素，计算绝对值；
3. 跳过 NaN，使用严格小于和最早逻辑下标规则更新局部结果；
4. 将局部值和下标写入 typed workspace，并执行 `SyncAll`；
5. 由 block 0 按核心编号顺序再次归约，写出 1-based 结果。

工作区的值区和下标区分别使用 `float` 与 `int32_t`，避免把索引编码到浮点数。
`n > 0` 时每个参与核心至少有一个元素，因此最终归约不会读取未初始化的槽位。
读取、比较和写回均在 kernel 内完成，不加载任务文件、golden、基线或私有答案。

### 3.3 编译和架构选择

`test/isamin/CMakeLists.txt` 在 `SOC_ARCH_DIRS` 包含 `arch22` 时编译
`test/isamin/arch22/isamin_test.cpp`；`arch35` 既有分支保持不变。A2/A3 的产品
支持说明写入 `blas/iamin/README.md`，公共头文件和既有 ABI 不修改。

| SoC/架构 | 入口 | 说明 |
| --- | --- | --- |
| Atlas A2/A3 / `arch22` | `blas/iamin/arch22/` | 本次 AIV 实现 |
| Atlas 950 / `arch35` | 既有目录 | 保留原实现，不参与本次 A2/A3 结论 |

没有按公开 case、形状或 benchmark 名称选择不同实现；不支持的语义不通过隐藏
fallback 完成，而是在 Host 校验阶段返回错误。

## 四、验证方案

### 4.1 正确性

测试入口 `test/isamin/arch22/isamin_test.cpp` 使用仓内公共 GTest、CSV loader、
golden 和 NPU wrapper。测试覆盖：

- 正常 `n`、连续和带步长访问；
- 负数、零、重复最小值和全 NaN；
- `n=0`、`incx<1`、空 `x`、空 `result`、负 `n`、步长乘积溢出；
- 空句柄状态码；
- 全部资料包 1200 条 CSV 用例，其中 200 条 `TC_PF` 性能用例。

黄金实现与 kernel 使用同一接口语义，但测试只比较返回码和最终整数结果，不把
golden 当作运行时 fallback。当前已在本地 CANN 9.0 的 `ascend910b` 配置下完成
源码、Ascend C kernel、共享库和 `isamin_test` 的编译链接验证；任务书指定的
CANN 9.1 和真实 NPU 全量执行仍需在验收设备上复跑。

### 4.2 性能

正式性能设备为 Atlas 800T A2（910B3），按任务书执行 5 次 warmup、至少 10 次
有效采样，记录设备、CANN、编译器、stream 和 kernel 时间。目标为 NPU 时间不高于
GPU 基线的 0.8 倍。GPU CSV 只作比较坐标，不能替代 NPU 实测；在完整 1200 条
用例和规定采样完成前，不作“性能达标”声明。

## 五、约束与无 fallback 边界

本实现仅覆盖任务书规定的 `float` 输入、`int` 结果、正步长逻辑访问和 A2/A3
`arch22` 路径。不得在运行时读取 `tasks/`、资料包、golden、GPU 基线或其他
工作树；不得调用 CPU/reference/TBE/vendor/950 kernel 作为兜底。任何不满足接口
契约的请求必须以状态码失败或 quick return 结束。新增实现不改变 `arch35`，也不
把 arch35 的测试或性能结果转移为本次 A2/A3 证据。

## 六、交付清单

- `blas/iamin/arch22/isamin_host.cpp`
- `blas/iamin/arch22/isamin_kernel.cpp`
- `blas/iamin/arch22/isamin_kernel.h`
- `blas/iamin/arch22/isamin_tiling_data.h`
- `test/isamin/arch22/isamin_test.cpp`
- `test/isamin/arch22/isamin_test.csv`
- `test/isamin/CMakeLists.txt`、公共 golden/wrapper 边界修正及
  `blas/iamin/README.md` 产品支持更新
