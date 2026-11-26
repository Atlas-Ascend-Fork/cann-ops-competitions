# aclblasChpmv（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/c596a05d90b9471683e41fcf4d026ae5)要求支持 ops-blas 的 arch22 实现。第二版以 ops-blas 的 `60d9d0b153e2` 为基线：`blas/spmv/arch22/chpmv_host.cpp` 已有双阶段计算，当前在结构校验完成前对 n=0 返回，且直接解引用两项标量。本版细化接口校验、标量分派及双阶段/直算选择。

## 需求分析

`aclblasChpmv(handle,uplo,n,alpha,AP,x,incx,beta,y,incy)` 执行 y ← αAx + βy，输入输出为交错 FP32 复数，AP 含 n(n+1)/2 个元素。列主序零基下，上三角 i≤j 的偏移为 j(j+1)/2+i；下三角 i≥j 的偏移为 j(2n−j+1)/2+i−j。另一半用镜像元素共轭得到，矩阵对角只读取实部。x/y 支持负步长，长度为 n 时以 (1−n)·inc 为逻辑起点；步长洞及非输出区域保持原位。

## 详细设计

### Host 与接口

沿用 `include/cann_ops_blas.h` 公共声明及 handle 绑定的 stream，代码置于 `blas/spmv/arch22/`。按任务书顺序检查 handle、n、uplo、步长、α/β 指针；n=0 直接返回，此后检查 AP/x/y。空句柄、非法枚举、其他非法参数分别按仓库状态码返回。用 64 位计算压缩偏移、负步长起点和字节跨度。Host 标量按值传入；Device 标量由核内读取并执行同样分支。α=0、β=1 不写 y；α=0 时跳过 AP/x，β=0 时跳过旧 y 读取。

参照 `blas/axpy_ex/arch22/axpy_ex_host.cpp` 使用 `aclrtPointerGetAttributes` 分别识别 α 和 β 的位置，覆盖 Host/Host、Host/Device、Device/Host、Device/Device 四种组合。非空算例先完成 AP/x/y 判空，再做标量位置分派。设备标量由同一 stream 上的核读取，不发起同步 D2H；双阶段中累加核在 α=0 时跳过 AP/x，归约核也跳过该路径的 workspace 读取，直接执行 no-op 或缩放。取步长绝对值前扩展到 64 位，用受检乘法换算字节跨度。

### Kernel 与分块

直算路径中每个输出块由一个核独占。对行块和列块划分三类：完全位于存储三角的块按连续列段读 AP；位于镜像三角的块交换索引、沿储存列加载并在 UB 转置取共轭；跨对角块用掩码选择两侧地址并将对角虚部分量清零。片上按实虚分离，FP32 维护两组累加器；列块 x 搬入一次，复用于当前行块。大步长 x/y 按逻辑索引访问，尾段按有效长度写回，防止覆盖相邻元素。行块、列块由可用 UB 预算约束；每个缓冲区及尾部安全搬运空间计入预算。

默认保留双阶段基线：每个存储的非对角元素 Aᵢⱼ 同时贡献给 yᵢ 与 yⱼ，贡献分别为 Aᵢⱼ·xⱼ、conj(Aᵢⱼ)·xᵢ；对角只贡献 Re(Aᵢᵢ)·xᵢ。若 `P=n(n+1)/2`、`W=8·align_up(n,8)`、`C=min(n,AIV 核数,floor(workspace_bytes/W))`，则压缩矩阵有效读量为 8P 字节，局部向量占 C·W 字节，写入后由归约核读取；按完整槽搬运估算新增 2·C·W 字节。两次启动保持同一 stream 内有序，归约核按固定核序累加并只写自己拥有的 y 区间。

单次启动的直算路径不使用 workspace，按前述输出块分工，每个块独立完成整段归约。AP 逻辑引用为 n² 项，实际 GM 流量取决于 tile 复用。workspace 容不下一个槽时选择直算；有 workspace 时先对 n≤32 做单次启动试验，比较两条路径的 kernel 总耗时，再按 A2/A3、uplo 和步长类型确认阈值。计入两阶段的两个 kernel 耗时，分别报告直算、阶段累加和归约时间。

## 可维可测分析

测试置于 `test/spmv/chpmv/arch22/`。按任务书附件以 CBLAS chpmv 为 golden，对 1000 条精度输入分别统计实虚分量，FP32 混合容差 rtol=atol=2^-13、匹配率至少 0.99，并执行最大误差门槛。增加上下三角、对角非零虚部、负步长洞、α/β 特值、空 n 和错误优先级的定向断言。200 条分层性能输入比较 `msprof op` kernel 平均耗时与附件 GPU kernel 基线，目标 NPU≤1.25×GPU；端到端计时单列。当前交付为设计方案，性能与精度待设备验收。
