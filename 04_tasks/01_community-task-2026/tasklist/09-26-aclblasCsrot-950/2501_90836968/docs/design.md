# 需求背景（required）

## 需求来源

CANN 社区任务「CANN训练营南科大-aclblasCsrot算子开发(950)」。接口语义参考 cuBLAS cublasCsrot，目标为 Ascend 950PR / CANN 9.1.0；代码基于 ops-blas 提交 `3c1542897f738857e928bce1996f1512220fafbc`。

作者 GitCode 账号：`2501_90836968`。个人仓库：https://gitcode.com/2501_90836968/project_hw 。验收分支 `feat/csrot-950`，验收提交 [1631a4727721e39eb45cda0c2b48af78e6443a79](https://gitcode.com/2501_90836968/project_hw/commit/1631a4727721e39eb45cda0c2b48af78e6443a79)。本文按社区 design_template.md 的需求背景、需求分析、详细设计、可维可测分析组织。

## 背景介绍

两个 complex64 向量应用由实数 c、s 构成的 Givens 平面旋转。该操作可用于 QR 分解等线性代数计算。已有公共接口与 arch22 实现，本次增加 arch35 实现，不新增私有接口。

# 需求分析（required）

## 需求描述

实现 `aclblasCsrot(handle,n,x,incx,y,incy,c,s)`，原地更新两个复数向量，c/s 支持 Host、Device 及混合位置；支持正、负和零步长。

## 需求拆解

1. 实部、虚部各按 float32 运算，y 更新必须读取原始 x。
2. 先校验 handle，再处理 n≤0；正长度校验空指针和 int32 偏移溢出。
3. 连续访问利用 SIMD 搬运与向量指令，非连续访问使用 SIMT。
4. 零步长有循环依赖，按 Netlib 顺序串行处理。
5. 对照 Netlib BLAS 3.10.0，满足分量混合精度；典型性能目标为 21.69、43.19、119.38 μs。

# 详细设计（required）

## 算子分析

### 数学公式

对逻辑下标 i∈[0,n)，读取原始复数 X、Y：

```
x_new.real = c*X.real + s*Y.real
x_new.imag = c*X.imag + s*Y.imag
y_new.real = c*Y.real - s*X.real
y_new.imag = c*Y.imag - s*X.imag
```

负步长以 `(1-n)*inc` 为起始复数偏移，非负步长起始偏移为 0。物理长度为 `1+(n-1)*abs(inc)`；零步长对应一个元素重复参与旋转。

### 支持数据类型及形状

向量为 complex64，交错存储 real/imag 两个 float32；c/s 为 float32 标量，n/incx/incy 为 int。逻辑形状为 [n]，步长为复数元素单位。

## 算子实现

### 3.2.1 Host 侧设计

实现文件：`blas/rot/arch35/csrot_host.cpp`。

参数校验顺序为 handle、n≤0、x/y/c/s 空指针、偏移范围。`(n-1)*inc` 用 int64 计算并确认落在 int32 范围，避免窄化溢出；n=1 时 INT_MIN/INT_MAX 步长合法。

通过 aclrtPointerGetAttributes 分别判断 c、s 的位置。Host 值放入 tiling 结构，Device 值保留指针交由 kernel 在绑定 stream 上读取，不引入 D2H 同步。

连续路径 `incx=incy=1` 选择 tilingKey=0，n≤131072 时最多启用 32 个核，其他规模使用可用 AIV 核数。初始 block 数为 `min(可用核数,ceil(n/1024))`；每核复数数量向上对齐到 4，再据此重算 block 数，最后一核裁剪到有效范围。这样使核间边界按 32 字节对齐。单核以 float 分量为单位切 tile，UB 分配按寄存器宽度 256 字节对齐。

非连续路径选择 tilingKey=1，正常步长按 SIMT 最小 128、最大 2048 线程配置，线程循环覆盖全部元素；任一步长为零时只启用一个 block 的一个逻辑线程顺序执行。

共享 tiling 结构记录 n、分核、tile、步长、线程数、c/s 值及其位置标记。结构不承载用户数据指针。

### 3.2.2 Kernel 侧设计

实现文件：`blas/rot/arch35/csrot_kernel.cpp`。

连续路径将 n 个复数视为 2n 个 float，使用两路输入队列和两路输出队列，均为单缓冲。CopyIn→Compute→CopyOut 循环处理，每个核写入独立区间。整块搬运使用 DataCopy，尾部使用 DataCopyPad。SIMD VF 在寄存器中读取原始 X/Y，分别执行四次 Muls、一次 Add、一次 Sub 后写入输出缓冲；不使用融合乘加。队列负责搬运与计算的同步，去掉原先 UB 工作缓冲及其重复读写。

SIMT 路径每个 lane 处理一对复数，地址计算使用 int64。先完整读取 X/Y 的四个分量，再写回。零步长只允许 block0/thread0 执行，循环内先取旧值再更新，保持跨迭代依赖。

SIMT 默认 FMA 融合与 SIMD 分步舍入不一致。CANN 9.1 下测试证明 pragma 和 fp-contract=off 仍可能融合；将八个中间乘积存为 volatile float，固定乘法舍入边界。回归用例以 `c=s=1e10,x.real=nextafter(1,2),y.real=-1` 检验结果为 1024，旧实现约为 1192.09。未放宽容差。

### 内存设计

输入输出原地复用，Host 不申请额外 Device workspace。连续 UB tile 上限为 `floor((248*1024)/(5*4)/64)*64=12672` 个 float；实际 tile 为 `min(上限,ceil(2*perCoreN/64)*64)`。四个缓冲最大合计 202752 字节，保留 UB 余量；这是静态缓冲预算，不是设备进程总占用。SIMT volatile 可能产生线程私有暂存，由编译器分配，不宣称其为零内存。

向量申请量为 `8*spanX+8*spanY` 字节，Device c/s 最多额外 8 字节。典型连续三规模向量量为 16、32、64 MiB。报告另附 aclrtGetMemInfo 的前后读数和测试进程 peak RSS，区分逻辑申请量、allocator 保留量和进程总量。

## 支持硬件

| 芯片 | 本次验证 |
|---|---|
| Ascend950PR_9579 | x86_64、openEuler 24.03 LTS-SP3、CANN 9.1.0 真机 |
| Ascend 950DT | 未验证 |
| A2/A3 | 公共 arch22 路径保留；不作为本次 arch35 验收结果 |

## 算子约束限制

x/y 有效存储重叠未定义，不作支持承诺。调用方负责指针可访问性、内存范围和生命周期；n≤0 时不访问向量。同步及异步错误遵循仓库运行时模式；数值分类按题目逐分量语义处理，不保证 NaN payload 或有符号零逐位一致。

# 可维可测分析

## 精度标准/性能标准

分量误差满足 atol=rtol=2^-13、matched_ratio≥0.99，同时逐元素误差不高于 `max(0.01,32*ULP(golden))`；NaN 单独分类，Inf 检查符号，步长间隙逐位保持。

原始 1000 精度 + 200 性能 CSV 保留，新增 256 分布用例：逐 shape/stride 枚举 x/y/c/s 的 16 种均匀/正态组合，每个参数均为 128/128。正态均值在 [-5,5]、标准差在 [0.1,2]，不截断尾部，实虚部分别采样；随机种子和标量分布参数记录在 CSV/日志。

额外四项测试为 ParameterValidation、UnalignedComplexViews、LargeScalarCancellation、DeviceMemoryAccounting。覆盖空指针、handle/no-op 优先级、偏移溢出、n=1 极端步长、四种 c/s 位置及非对齐存储。

Golden 来自官方 Reference-LAPACK v3.10.0 中未修改的 BLAS/SRC/csrot.f。该版本 CBLAS 不公开 cblas_csrot，测试以薄 C ABI 适配直接调用 csrot_。构建使用 `-ffp-contract=off -fno-signed-zeros`；后者避免 gfortran 生成零交叉项污染 Inf。参考库先通过独立分量隔离和相消自检，再用于对照。编译参数、源码哈希、链接路径均保留。

性能测试先检精度，预热 5 次、采样 60 次，输入恢复与 golden 均在事件计时区间外；三典型规模另尝试采集 msprof op BasicInfo 各 60 条，但本轮工具报告缺失 DeviceProf1.bin、解析 0 成功/60 失败，不报告当前版本 kernel 耗时。性能判定使用绑定 stream 的事件计时，原始失败日志保留；不复用旧 kernel 的 profiling 数值。测量结果以自测报告为准。

## 兼容性分析

公共头文件中的签名与参数顺序不变，不修改其他架构实现。新增 Netlib 参考库开关仅作用于 csrot_test，不影响算子库 ABI。测试数据、参考源获取、构建和运行命令由 test_cases/run_acceptance.sh 复现。

## 模板来源

https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
