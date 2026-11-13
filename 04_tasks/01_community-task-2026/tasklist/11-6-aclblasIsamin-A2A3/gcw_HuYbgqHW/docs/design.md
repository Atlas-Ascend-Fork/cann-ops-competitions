# aclblasIsamin 算子设计（Atlas A2/A3）

## 1. 需求背景

11 月 CANN 社区任务要求为 `cann/ops-blas` 补齐 Atlas A2/A3 的 BLAS-1 接口 `aclblasIsamin`。实现基于官方 master 的 `28de164d1becd69b1ca380e70bf2fbbcc907fec8`，沿用公共 API 与 950PR 的参数契约，采用 Ascend C kernel 直调。目标环境为 CANN 9.1.0，验收设备为 Atlas 800T A2（910B3）。

## 2. 需求分析

### 2.1 接口与参数

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

| 参数 | 方向及位置 | 类型与布局 | 约束 |
| --- | --- | --- | --- |
| handle | 输入，Host | aclblasHandle_t | 由 aclblasCreate 创建，携带执行 stream |
| n | 输入，Host | int | 逻辑向量长度，负数非法 |
| x | 输入，Device | FLOAT32，ND | 正常计算时有效，物理长度至少为 1 + (n - 1) × incx |
| incx | 输入，Host | int | 元素步长，小于 1 时快速返回 |
| result | 输出，Device | INT32，ND，[1] | 始终有效，正常计算写入 1-based 索引，快速返回写 0 |

### 2.2 计算语义

逻辑元素 `i` 取自物理位置 `x[(i - 1) * incx]`，其中 `i` 从 1 到 `n`。返回绝对值最小的最早逻辑位置。等值、正负零及相同绝对值均保留较早位置。顺序参考以首元素绝对值和位置 1 为初值，首元素为 NaN 时以 FLT_MAX 代替，随后只在遇到严格更小的非 NaN 绝对值时更新。全 NaN 或首 NaN 后只有 Inf、NaN 和 FLT_MAX 时保留位置 1。

## 3. 详细设计

### 3.1 Host 校验与 tiling

按以下顺序处理参数。

1. 空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n < 0` 或空 `result` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n == 0` 或 `incx < 1` 时，通过 `aclrtMemsetAsync` 在绑定 stream 写 4 字节零。提交失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
4. 正常路径验证 `x` 非空，并以 INT64 计算 `(n - 1) * incx`。超出 INT32_MAX 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `n ≤ 512` 使用单 AIV。更大输入查询 AIV 数，将每 64 个逻辑元素视作一组，按 AIV 数向上取整分配组数，依据实际片数下发 block。最后一核将范围截断到 `n`。Tiling 参数为 `n`、`incx`、每核逻辑元素上限和实际 block 数。

单核直接输出，额外全局 workspace 为 0。多核复用 handle 的有效 workspace，每核保留独立的 32 字节槽。有效 workspace 需容纳全部槽，容量不足时返回执行失败。两个 kernel 按 scan、reduce 的顺序下发到同一 stream。

### 3.2 Kernel 实现

- 连续输入按最多 12,288 个 FP32 元素分块搬入 UB，向量段以 64 个元素为单位。多块输入使用两个缓冲，在计算当前块前发起下一块搬运。依次取绝对值、将 NaN 映射为正无穷，然后使用 Ascend C 的 repeat 归约。读取归约寄存器得到整块最小值及其最早位置，跨块继续按严格小于更新。
- 连续尾部用 `DataCopyPad` 搬运有效字节，再按逻辑顺序标量扫描。正跨步路径按 INT64 宽度的物理偏移读取采样元素。两条路径均按严格小于更新，扫描至当前最小值为零即可结束本核。
- 首核保存任务书定义的首元素初值。其余核以正无穷为初值。后续 NaN 在向量路径映射为正无穷，在标量路径通过有序比较跳过。只有严格改善全局首元素初值的候选才会胜出。
- 多核 scan 写出各核的最小值和零基逻辑位置，reduce 按 block 顺序比较。严格小于更新使较早 block 的等值结果保留。最后仅写入 4 字节的 1-based 索引。
- 最大 scan UB 缓冲预算约 147 KiB，包含双输入缓冲、Inf 张量、比较掩码、repeat 归约临时空间和输出槽。小规模输入按实际向量段申请单输入缓冲。连续尾块及输出通过带长度的搬运处理，避免按 32 字节扩写调用方输出。

### 3.3 支持硬件与任务书 §2.5 约束

| 项目 | 设计结论 |
| --- | --- |
| 硬件 | Atlas A2/A3 使用 arch22 AIV 路径，910B3 为自验目标 |
| 参数合法性 | 按 §3.1 校验，输入元素全值域，含 NaN 和 Inf |
| 非连续输入 | 支持正步长，跳过物理间隔元素 |
| 广播 | 单输入向量，无广播 |
| 动态 shape | 运行时根据 n 和 incx 计算 tiling |
| 原地和视图 | x 只读，result 独立输出 |
| 确定性 | tile 内、tile 间及核间均保留第一个并列索引 |
| 空向量 | 在 stream 异步写 0，计算 kernel 不下发 |
| 0 维 Tensor | 输入为向量，输出为一个 Device INT32 |
| 异步执行 | 成功表示提交成功，读取输出前同步绑定 stream |

### 3.4 已知限制与兼容性

大跨步访问的带宽受物理访存跨度影响，性能子集按任务提供的 200 条 `incx = 1` 基线验收。优先在 910B3 上实测 launch-bound 小向量和大向量两阶段开销。Atlas A2/A3 的新实现沿用 `include/cann_ops_blas.h` 的既有声明，950PR 继续使用 arch35 实现。

## 4. 可维可测分析

| 项目 | 通过标准 |
| --- | --- |
| 精度 | GTest 加载原 1000 条精度 CSV，并补充正态、特殊值、跨块并列、未对齐、步长溢出和空指针用例。输出必须与独立 CPU golden 的 INT32 索引完全一致，负向用例比对状态码 |
| 性能 | 910B3 / CANN 9.1.0 按用例运行 msprof op。预热 5 次，至少 10 次有效调用，使用 Task Duration(us)。双 kernel 路径计入两阶段耗时；每条均值 ≤ 对应 A100 标杆 × 1.25 |
| 内存 | 任务书未设内存验收门槛。记录单核 0 字节及多核 32 字节 × block 数的额外全局 workspace |

用例和复现入口位于[私有验收仓的 arch22 测试目录](https://gitcode.com/gcw_HuYbgqHW/ops-blas-isamin-a2-a3/tree/feat/isamin-a2-a3/test/isamin/arch22)。实机编译、精度和性能按上述步骤在 910B3 自验。
