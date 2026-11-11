# aclblasSrot 算子设计文档

> 社区任务：11月社区任务-aclblasSrot算子开发（A2/A3）
>
> 适配硬件：Atlas A2 / Atlas A3（arch22）
>
> CANN 版本：9.1.0
>
> 对标接口：cuBLAS `cublasSrot`；数值参考：Netlib BLAS 3.10.0 `srot`
>
> 工程模式：ops-blas，Ascend C kernel 直调

## 一、需求背景

为 ops-blas 补充 A2/A3 上的单精度实数向量平面旋转能力。复用公共 `aclblasSrot` 接口，参考仓内 `blas/rot/arch35/` 的 Srot 与 `blas/rot/arch22/` 的同族算子工程，实现 Host 参数校验、tiling 和 AIV kernel。

对逻辑下标 `i=0..n-1`，两个输出均使用旋转前的输入：

```text
x_new[ix] = c * x_old[ix] + s * y_old[iy]
y_new[iy] = c * y_old[iy] - s * x_old[ix]
index(i, inc) = inc > 0 ? i * inc : (n - 1 - i) * abs(inc)
```

x、y 原地更新；非单位步长的间隔元素保持不变。输入不涉及广播或归约。

## 二、需求分析

### 2.1 接口与参数

复用 `include/cann_ops_blas.h` 中已有声明，不新增私有平行接口：

```cpp
aclblasStatus_t aclblasSrot(
    aclblasHandle_t handle, int n, float* x, int incx,
    float* y, int incy, const float* c, const float* s);
```

| 参数 | 类型与位置 | 约束 |
| --- | --- | --- |
| handle | Host 句柄，携带 stream | 不得为空 |
| n | Host int32 | n=0 成功返回；n<0 非法 |
| x、y | Device FLOAT32 向量 | n>0 时不得为空，物理长度分别为 `1+(n-1)*abs(inc)` |
| incx、incy | Host int32，元素单位 | 支持非零正负步长；不得为 INT_MIN；末元素偏移不超过 INT32_MAX |
| c、s | FLOAT32 标量指针，Host 或 Device | n>0 时不得为空，两个指针的内存位置分别判定 |

### 2.2 需求拆解

- 按任务书规定校验返回码与 n=0 快速返回，不沿用 arch35 对负 n、零步长的放行行为。
- 连续向量使用 AIV 向量计算；跨步向量按逻辑下标映射物理地址。
- 支持 Host、Device 及混合位置的 c/s，不在 Host 端解引用 Device 指针。
- 提供 arch22 CSV 驱动 GTest、算子 README 与可复现的精度、性能验证方案。

## 三、详细设计

### 3.1 Host 与 tiling

校验顺序为：空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n=0 返回成功；其余负 n、空 x/y/c/s、非法步长与偏移溢出返回 `ACLBLAS_STATUS_INVALID_VALUE`。跨度乘积使用 int64 计算，避免 INT_MIN 取绝对值及 int32 乘法溢出。

分别通过 `aclrtPointerGetAttributes` 查询 c/s：Host 值写入按值传递的 tiling，Device 地址直接传给 kernel。tiling 包含 n、incx、incy、Host 标量值；无需临时 GM 工作区或标量 D2H 回读。

分核采用 `blocks=min(AIV核数, ceil(n/1024))`，小规模减少空闲核。各核处理连续的逻辑索引区间，长度为 `n/blocks`，余数分配给前面的核；起点为 `blockId*(n/blocks)+min(blockId,n%blocks)`。单个元素仅有一个核写入。

### 3.2 Kernel 方案

**双单位步长路径**：每核按 4096 个 FLOAT32 分块，通过 DataCopyPad 搬入旧 x/y。使用两个输入队列、两个输出队列与一个计算临时 buffer，共占用 80 KiB UB。通过独立的 Muls、Add 计算两个输出，避免覆盖旧值和额外乘加融合；使用队列事件维护搬入、计算、搬出的依赖。尾块按实际元素数搬运，不越界读写。

**跨步路径**：正步长按 `i*inc` 寻址，负步长按 `(n-1-i)*abs(inc)` 寻址。将一批旧 x/y 暂存到 32 字节对齐的 UB 槽，复用分步向量乘加计算，再用单 FLOAT32 的 DataCopyPad 写回对应物理位置。批次容量为 512 个逻辑元素，槽内 padding 初始化为零，间隔元素不写入。暂存、计算与回写之间建立流水依赖。

Device c/s 在 kernel 中读取；Host c/s 使用 tiling 中的值。不对 `c=1,s=0` 或其他零标量组合短路，保留 Inf/NaN 的公式传播行为。

```text
Host：校验 handle → n=0 快速返回 → 其他参数校验 → 标量位置判定 → tiling → stream 下发
Kernel：读取标量 → 分配逻辑区间 → 连续或跨步读入旧 x/y → 分步乘加 → 原地写回
```

### 3.3 约束核对

| 任务约束 | 设计处理 |
| --- | --- |
| 非连续 Tensor | 通过非零正负步长访问，仅更新逻辑元素 |
| broadcast | 不涉及，x/y 逻辑长度均为 n |
| dynamic shape | 运行时 n 与步长决定 tiling，无编译期 shape 假设 |
| 原地与视图 | x/y 复用各自输入内存，无返回视图 |
| 确定性 | 每个逻辑元素唯一写者，无原子操作与跨核归约 |
| 空 Tensor | n=0 不访问数据、不启动 kernel |
| 0 维 Tensor | 不涉及，c/s 通过标量指针表达 |
| 异步执行 | 使用 handle 绑定的 stream，调用方同步后读取输出 |

### 3.4 已知限制

仅支持 FLOAT32；跨步路径优先保证完整功能，性能子集按任务包的双单位步长场景验证。x/y 需覆盖各自完整物理跨度，部分重叠缓冲区的链式别名行为不在本任务承诺范围内。

## 四、特性交叉分析

| 特性 | 处理方式 |
| --- | --- |
| 原地更新与并行 | 先保存同一元素对的旧值，两个输出使用独立 buffer |
| 尾块与步长 | 连续尾块精确搬运，跨步仅写目标元素，检查间隔元素未修改 |
| 精度与计算顺序 | 分步乘加，分别检查 x/y 的有限值误差和非有限值等价性 |
| Host/Device 标量与异步 | 两指针独立定位，Device 值沿 handle stream 的执行顺序读取 |

## 五、可维可测分析

### 5.1 验证计划

- 在 `test/rot/srot/arch22/` 增加 CSV 驱动 GTest，使用同仓 GTest 注册机制，复用公共参数和 Netlib `cblas_srot` golden 接口。
- 覆盖任务包 1000 条精度用例与 200 条性能子集；补齐正态分布，与有效均匀随机用例成对覆盖；补充空 handle、空标量指针、混合标量位置、确定性及跨度溢出检查。
- 负向用例在构造完整向量前分流，使用 dummy 缓冲区，不按非法跨度分配内存。保留原任务包，对任务书与辅助说明冲突的期望值记录调整依据。
- x/y 分别采用 `atol=rtol=2^-13`、匹配率不低于 0.99、最大绝对误差不超过 `1e-2 或 32×ULP` 的判据。非有限值按任务书等价规则单独检查，间隔元素按位检查保持不变。
- 性能使用 `msprof op` 的 task duration，预热 5 次，有效采样至少 10 次，取全部成功样本的算术平均；每 case 平均耗时不超过 GPU 标杆的 1.25 倍。任务书的基准设备为 910B3，其他 A2/A3 设备记录具体型号并单独列示。
- 内存验收按任务书“不涉及”处理。

### 5.2 兼容性与维护

实现目标目录为 `blas/rot/arch22/`，不改变 arch35 的既有行为。README 明确两种架构对负 n、零步长的差异。按 Host、Device 与测试分层组织代码，避免引入重复接口和无用编译文件。

参考：[cuBLAS Srot](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rot)、[Netlib Srot](https://netlib.org/blas/srot.f)、[ops-blas rot](https://gitcode.com/cann/ops-blas/tree/master/blas/rot)。
