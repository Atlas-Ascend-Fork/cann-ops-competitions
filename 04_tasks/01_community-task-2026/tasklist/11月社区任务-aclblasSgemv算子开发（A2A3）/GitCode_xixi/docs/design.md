# aclblasSgemv（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务书](https://www.hiascend.com/activities/task-center/details/a8c0b7d0985349c59503ec0c34ca7344)要求实现 FP32 GEMV。以 ops-blas `60d9d0b153e2` 的 `blas/gemv/arch22/sgemv_host.cpp` 为基线，评估现有向量、Cube 特化和非单位输出步长路径。本稿用于历史任务设计练习。

## 需求分析

`aclblasSgemv(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy)` 原地执行 y←α·op(A)·x+βy。列主序 A(i,j) 在 `i+j·lda`，N 的 x/y 长度为 n/m，T/C 为 m/n。C 与 T 对实数等价。α/β 为 Host FP32，向量和矩阵为 Device FP32。任意非零合法正负步长由逻辑长度和增量描述，无需连续化副本。

## 详细设计

Host 先检查 handle，然后校验 trans、m/n、`lda≥max(1,m)`、步长和标量指针；A 对 m、n 同时非零时必填，x/y 对各自逻辑长度非零时必填。步长拒绝 0/INT32_MIN，计算 `max(0,len−1)·abs(int64(inc))≤INT32_MAX`。非法句柄与其他非法值分别返回 HANDLE_IS_NULLPTR 和 INVALID_VALUE。结构合法后空维直接返回。非空 α=0 时跳过 A/x，β=1 保持 y，β=0 直接写零，其余只缩放；普通 β=0 路径也跳过旧 y 读取。所有设备操作按 handle stream 执行。

沿用单位步长路径：N 按连续行分块，在 UB 复用 x 列段；T/C 按输出列分块，各列沿连续行归约，共享同一个 x tile。每个输出元素由单核写回，α/β 只合并一次。现有 Cube 特化仅在满足现有形状与标量门槛时启用，通用路径按 FP32 累加。对非单位输出步长，保留单核基线，试验按逻辑 y 分段铺核：地址为 `(inc<0 ? (len−1)·abs(inc) : 0)+k·inc`；以物理写入为准选择批量或精确写回，保护步长洞和相邻核的边界。用实测决定何时增加核数，记录分块导致的 x 重读量、工作区和 kernel 总耗时。

## 可维可测分析

`test/gemv/sgemv/arch22/` 读取附件 1000 条精度与 200 条性能用例。按 §3.2 检查 FP32 容差、匹配率、最大误差及非有限值；补测 T/C 等价、非方阵、lda 填充、负步长哨兵、β=0 时 y 预填 NaN、空维与非法跨度。910B3 上以 `msprof op` 5 次预热、至少 10 次有效调用比较 GPU 标杆，目标 NPU≤1.25×GPU。设计稿未包含上板结果。
