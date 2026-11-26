# aclblasSaxpy（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务书](https://www.hiascend.com/activities/task-center/details/6ffdb3f781a044588935d6c7d33c3b96)要求提供 FP32 SAXPY。以 ops-blas `60d9d0b153e2` 中 `blas/axpy/arch22/` 为基线，设计连续与跨步访问的分派和回归。本稿用于历史任务设计练习。

## 需求分析

`aclblasSaxpy(handle,n,alpha,x,incx,y,incy)` 计算 y[k]←α·x[k]+y[k]，x/y 为 Device FP32，α 为 Host FP32。增量作用于元素；负增量按反向遍历。y 原地更新，留空的物理位置须保留。

## 详细设计

先检查 handle，空句柄返回 HANDLE_IS_NULLPTR；n≤0 直接成功，不读取指针或启动 kernel。n>0 时校验 α/x/y、零步长、INT32_MIN 和两向量的 int32 索引跨度；用 int64 计算 `(n−1)·abs(inc)`。通过校验后读取 Host α；α=0 时按任务书直接返回，保留 y 的原有位模式，不读取 x。

对 incx=incy=1，以 8 个 float 为一个 32 B 分区单元，把完整单元均分给 AIV 核，完整段以向量乘加和 GM/UB 双缓冲更新，尾段按有效长度写回。连续路径的最低数据流量是读取 x、读取旧 y、写回新 y，共 12n 字节；用它计算有效带宽，与现有路径在同图形比较。通用路径以逻辑索引分片，起点为 `inc<0 ? (n−1)·abs(inc) : 0`；绝对步长为 1 的负步长可连续加载后在 UB 反序，其他大步长只读取有效元素并精确写出 y。内部 DMA 边界按实际地址对齐；未对齐段使用精确写，避免相邻核争写。单核或单 buffer 用于小时延基线，分派阈值按 A2/A3 分别测量。

## 可维可测分析

`test/saxpy/arch22/` 使用附件 1000 条精度与 200 条性能数据，按 §3.2 校验 FP32 容差、匹配率和最大误差，并覆盖零/负 n、双负步长、正负混合步长、大跨度拒绝和 y 洞填充哨兵。910B3 使用 `msprof op` 预热 5 次，至少 10 次有效采样求平均，逐项与 GPU 标杆比较，目标 NPU≤1.25×GPU。上板数据待验证。
