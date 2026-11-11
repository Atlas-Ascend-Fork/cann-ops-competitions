# aclblasSrot（Atlas A2/A3）算子设计

团队：m0_52603995。任务来源：11月社区任务-aclblasSrot算子开发（A2A3）。

# 需求背景（required）

ops-blas master 已有 arch35 Srot 和 arch22 Csrot，缺少 arch22 实数 Srot。本任务使用 Ascend C kernel 直调，复用 `include/cann_ops_blas.h` 公共接口和 handle stream，补齐 A2/A3 FP32 BLAS-1 平面旋转能力。

# 需求分析（required）

```cpp
aclblasStatus_t aclblasSrot(aclblasHandle_t handle, int n, float* x, int incx,
                          float* y, int incy, const float* c, const float* s);
```

`x'=c*x+s*y; y'=c*y-s*x`，右侧均取旧值。x/y 为 Device 一维 FP32 向量，物理长度 `1+(n-1)*abs(inc)`；负步长从物理尾端起算。c/s 分别独立支持 Host/Device 指针。返回码按任务书 §2.4/§2.5 的 A2/A3 契约：空 handle 优先报 HANDLE_IS_NULLPTR；n<0 报 INVALID_VALUE；n=0 返回 SUCCESS 且不访问其他参数；n>0 时检查 x/y/c/s 非空、步长非0且非INT_MIN，64位计算 `(n-1)*abs(inc)` 不得超过 INT32_MAX。

# 详细设计（required）

## Host 与 tiling

1. 校验完成后通过仓内 `CheckPtrLocation`（aclrtPointerGetAttributes）查询 c/s。Host 值写入按值传递的 tiling；Device 指针直接传给 kernel，在同一 stream 上读取，不做 D2H 或强制同步。
2. 从平台取得 AIV 核数和 UB 大小。连续路径每核最小目标工作量 4096 个元素，核数 `min(AIV,ceil(n/4096))`；跨步路径每核最小目标量256，核数 `min(AIV,ceil(n/256))`。分核长度向上对齐128个逻辑元素，最后有效核裁剪，空核退出。
3. 连续 tile 上限4096个元素，4个双缓冲队列（x/y输入、x/y输出）及1个scratch，合计36字节/元素，UB不足时按64元素对齐缩减。
4. 跨步 tile 上限256个逻辑元素，每个元素在UB占8个float槽位；4个双缓冲队列加scratch及反向Gather索引，合计320字节/逻辑元素，按8个逻辑元素对齐缩减。各GM区间只搬参与旋转的4字节，间隔元素不读改写。
5. 不分配GM workspace、tiling设备内存或每次调用的临时设备缓冲；参数按值下发。当前stream异步kernel直调。

## Kernel 算法

连续路径使用 DataCopyPad 精确搬入/写回有效长度，TQue 双缓冲流水。跨步路径用多burst DataCopyPad，每burst4字节、GM间隔 `(abs(inc)-1)*4` 字节、UB32字节槽位；负步长使用Gather反转元素槽位，输出反转回物理升序。GM stride 字节超过uint32时使用逐元素DataCopyPad回退（合法span守卫使此类n很小）。所有地址偏移用64位计算。

计算拆为独立FP32乘法和加减，避免融合乘加与Netlib每步舍入的抵消差异：

```text
Muls(newX, oldX, c)
Muls(newY, oldX, s)
Muls(workX, oldY, c)
Muls(workY, oldY, s)
Add(newX, newX, workY)
Sub(newY, workX, newY)
```

旧x读完后才允许复用x输入槽位。各Vector依赖点使用PIPE_V barrier，TQue负责MTE2/V/MTE3同步；索引由Scalar填充后执行S_V事件。每核负责不重叠的逻辑区间，无跨核同步。n=0不启动kernel；不对c/s值做单位旋转等短路，以保留Inf/NaN自然传播。

## §2.5约束核对

|约束|方案/结论|
|---|---|
|参数合法性|Host按上述顺序校验，溢出在任何缓冲访问前拒绝|
|非连续|支持全部合法非零正负步长，间隔元素保持不变|
|broadcast|不涉及，两个向量参与元素数均为n|
|dynamic shape|无需图模式，运行时n和步长决定tiling|
|原地与视图|两个旧输入保留至各自乘法完成，写回原指针，不返回视图|
|确定性|无规约/原子操作，同输入运行结果确定|
|空Tensor|n=0只检查handle后成功返回，不访问其他指针|
|0维Tensor|不涉及；标量由c/s指针表达|
|异步|使用handle stream，Device标量无Host同步，调用方读回前同步|

x和y部分重叠的跨向量别名无任务书定义，本方案不新增此项语义；交付用例使用独立向量。保留arch35现有行为，不改共享声明。

## 文件与工程集成

新增 `blas/rot/arch22/srot_host.cpp`、`srot_kernel.cpp`、`srot_tiling_data.h`；更新族README中Srot产品与per-arch说明。新增 `test/rot/srot/arch22/` GTest工程，在既有测试CMake显式注册arch22。复用仓内构建，无私有平行接口。

# 可维可测分析

在Atlas 800T A2 / 910B3，CANN9.1.0上构建运行。原始1000精度、200性能CSV完整保留；补充正态分布、返回码契约、nullptr、异步Device标量、负步长与未选中元素/边界guard检查。golden使用Netlib BLAS3.10.0 cblas_srot，float32落盘；x/y分别按atol=rtol=2^-13、matched_ratio>=.99、max_abs_error<=.01或32ULP、任务规定非有限值等价规则判定。

性能按单case采集msprof op BasicInfo Task Duration(us)，5次warmup不计样本，至少10个成功有效样本算术平均，逐case不超过gpu_baseline的1.25倍。性能基线重复参数按原行顺序对应并校验参数，避免字典覆盖。内存§3.4明确不涉及，仍提交模板对应说明文件。

## 材料冲突及验收影响

1. 任务书§2.4/§2.5明确负n、零步长报INVALID_VALUE；原始CSV有392条期望SUCCESS（7条负n、385条正n零步长），附带README沿用arch35口径。本次按任务书正文A2/A3契约实施：保留任务包原始CSV，在工作CSV中将这392条期望返回码修正为INVALID_VALUE，保留逐行差异说明，并将修正后的完整1000条用例纳入全量验收。该项是已明确的材料适配，不作为开发或验证阻塞，不删减任何用例。
2. README和verify_performance.py以aclrtEvent avg_us判定，正文§3.3明确msprof op Task Duration。本实现按正文并保留原脚本，新增真实msprof采集入口。
3. 正文负向分配段把inc<=0统称quick return，与负步长有效矛盾；合法负步长实际按span分配执行，仅非法参数/n=0采用dummy。

## 兼容性与交付

公共签名与ABI不变，arch35不修改。交付私仓实现/测试/README、设计PR、七项task_submission报告日志与ZIP。设计按社区任务仓README的贡献者建目录方式，放置于 `04_tasks/01_community-task-2026/tasklist/11-aclblasSrot-A2A3/m0_52603995/docs/design.md`；`11`表示任务月份，不虚构任务序号。验收实现使用已有个人私仓 `m0_52603995/ops-blas` 的 `community/srot-a2a3` 分支。性能和精度数据仅写实际运行结果；当前设计描述的是待实施/待验证方案，不表示验证通过。
