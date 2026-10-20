# aclblasCsrot算子设计文档（Ascend950PR/arch35）

> 任务：CANN训练营南科大-aclblasCsrot算子开发(950)
> 任务书：aclblasCsrot_Atlas950PR_task_doc.md
> 目标仓库：https://gitcode.com/cann/ops-blas

---

# 需求背景（required）

## 需求来源

- 活动：CANN训练营–南方科技大学专场，活动帖【03任务引导】：https://gitcode.com/org/cann/discussions/318
- 任务中心：CANN训练营南科大-aclblasCsrot算子开发(950)
- 任务书：`aclblasCsrot_Atlas950PR_task_doc.md`
- 上游仓库：https://gitcode.com/cann/ops-blas ，算子上库目录`blas/rot/arch35/`，测试上库目录`test/rot/csrot/arch35/`
- 接口声明：`include/cann_ops_blas.h`中已有`aclblasCsrot`，本任务不新增接口，不定义950PR私有API

## 背景介绍

### aclblasCsrot算子实现优化

`aclblasCsrot`是BLAS Level-1辅助例程，对两个等长复数向量施加Givens平面旋转，x、y原地更新。ops-blas仓的`rot`算子族同时提供实数版本`aclblasSrot`与复数版本`aclblasCsrot`；其中`aclblasSrot`已完成arch35（Ascend950PR）适配，`aclblasCsrot`仅有arch22旧实现，尚无arch35实现。

本任务在arch35目录下新增复数旋转实现，复用`aclblasSrot`已验证的双路径框架：连续访问走SIMD流水，非连续访问走SIMT网格步进循环。复数向量按`float`交错视图处理，实部、虚部由同一条向量指令同步旋转。

### 算子实现现状分析

| 项目 | 现状 | 对arch35实现的影响 |
| --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h`已声明`aclblasCsrot(handle, n, x, incx, y, incy, c, s)` | 无需改头文件，直接实现 |
| arch22实现 | `blas/rot/arch22/csrot_host.cpp`、`csrot_kernel.cpp`，固定8个block，仅按连续`float`视图搬运，未实现incx/incy，入口未校验handle、n≤0与空指针 | 不能直接复用到arch35；需按任务书补齐步长、边界和指针位置语义 |
| arch35实数实现 | `blas/rot/arch35/srot_*`已实现连续SIMD与stride SIMT双路径 | 可作为工程脚手架和同步/切分参考 |
| arch35复数实现 | 不存在 | 本任务新增`csrot_{host,kernel,kernel.h,tiling_data.h}` |
| arch35测试 | `test/rot/csrot/arch22/csrot_test.cpp`仅有A2/A3硬编码冒烟用例 | 需参照`srot`新建arch35 CSV驱动GTest工程 |
| 文档 | `blas/rot/README.md`当前标注Csrot在950PR/950DT不支持 | 合入时更新产品支持表与接口说明 |

### 算子功能分析

设`n`为参与旋转的元素个数，`k = 1+(i-1)*incx`、`j = 1+(i-1)*incy`（i=1..n，1-based索引，兼容Fortran），旋转公式为：

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]        （x[k]取旋转前的值）
```

其中`x`、`y`为complex64向量，`c`、`s`为float实数标量。与`crot`不同，`csrot`的正弦因子是实数，因此实部与虚部按同一组`(c,s)`独立旋转，不存在复数乘法的交叉项。典型用途为QR分解、特征值计算中的平面旋转。

# 需求分析（required）

## 需求描述

使用Ascend C在Ascend950PR（arch35）上实现`aclblasCsrot`：

- 数据类型：x、y为complex64（`aclblasComplex`，实部/虚部各float32），c、s为float32
- 硬件与软件：Ascend950PR，CANN9.1.0
- 功能语义对齐cuBLAS`cublasCsrot`与Netlib`csrot.f`
- 接口为句柄式BLAS接口，通过handle绑定的stream异步下发
- 精度满足任务书§3.2，性能满足任务书§3.3的三条标杆耗时
- 不支持950PR私有平行接口；接口声明沿用`include/cann_ops_blas.h`

## 需求拆解

1. **连续路径**：`incx==1 && incy==1`时走SIMD流水，满块用`DataCopy`、尾块用`DataCopyPad`，Vector阶段完成旋转
2. **stride路径**：其余步长组合（正非1、负、零及混合）走SIMT网格步进循环
3. **原地语义**：x、y均为输入输出；每个元素先读齐原始x、y分量，再写回新值，保证y使用旋转前的x
4. **边界对齐Netlib**：`n≤0`为合法no-op；`incx`、`incy`可正可负可零，均不拦截；`c==1&&s==0`的单位旋转不短路
5. **负步长**：从向量尾端起算，按负方向步进
6. **零步长**：按Netlib语义对同一元素反复旋转，形成串行累加，不能由多线程并发执行
7. **标量指针位置**：c、s可独立位于Host或Device内存，运行时用`aclrtPointerGetAttributes`判定，支持host/device/mixed三种组合
8. **数值舍入**：每个乘积、每个加法分别舍入，避免编译器收缩为FMA，使结果稳定对齐Netlib参考语义
9. **测试与验收**：覆盖随任务提供的1200条CSV用例，补充int32偏移溢出、空指针等负向用例，输出精度、性能、内存自测报告

# 详细设计（required）

## 算子分析

### 数学公式

```
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k]

k = 1 + (i-1) * incx,  j = 1 + (i-1) * incy,  i = 1..n
```

`c`、`s`为实数，实部与虚部分量独立计算：

```
x'[k].real = c * x[k].real + s * y[j].real
x'[k].imag = c * x[k].imag + s * y[j].imag
y'[j].real = c * y[j].real - s * x[k].real
y'[j].imag = c * y[j].imag - s * x[k].imag
```

由上式，n个complex64元素等价于2n个连续float lane，实部在偶数lane、虚部在奇数lane；同一条向量`Muls`/`Add`可同时处理实部与虚部。

### 支持数据类型

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| x、y | complex64（`aclblasComplex`） | Device内存，原地更新 |
| c、s | float32 | Host或Device内存标量指针 |
| 计算精度 | float32 | 实部、虚部分别比对 |

### 支持形状

- 逻辑一维`[n]`，n为运行时入参
- 物理跨度`1 + (n-1) * |inc|`，inc可正、负、零
- 不要求额外leading dimension padding
- 不涉及broadcast与dynamic shape

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

##### 1. 接口与参数校验

入口顺序与`aclblasSrot`对齐：

| 顺序 | 条件 | 返回值 | 说明 |
| --- | --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 优先于no-op判断 |
| 2 | `n <= 0` | `ACLBLAS_STATUS_SUCCESS` | 合法no-op，不校验其余参数，不修改x/y |
| 3 | `n > 0`且x、y任一为nullptr | `ACLBLAS_STATUS_INVALID_VALUE` | x、y必须非空 |
| 4 | `n > 0`且c、s任一为nullptr | `ACLBLAS_STATUS_INVALID_VALUE` | c、s必须非空 |
| 5 | `aclrtPointerGetAttributes`失败 | `ACLBLAS_STATUS_INVALID_VALUE` | 无法确定标量指针位置 |
| 6 | `(n-1)*incx`或`(n-1)*incy`超出int32范围 | `ACLBLAS_STATUS_INVALID_VALUE` | 在int64中计算后判断，避免kernel侧偏移溢出 |

校验通过后，算子只负责下发kernel，不同步stream；调用方按BLAS异步语义在读取结果前调用`aclrtSynchronizeStream`。

##### 2. 复数内存布局

`aclblasComplex`定义为：

```cpp
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;
```

数组在Device上按`real, imag, real, imag, ...`连续排列，sizeof为8字节。kernel以`float`视图访问：

- complex元素索引`e`对应float lane偏移`2*e`
- 连续路径的GM偏移按float lane计算，搬入字节数为`dataCount * 8`
- 向量指令作用于`2 * dataCount`个lane

该布局不需要在Device侧做任何格式转换，也不引入额外workspace。

##### 3. TilingData

新增`CsrotTilingData`，字段与`srot`同构，所有计数以complex元素为单位：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `tilingKey` | uint32 | 0=连续SIMD，1=stride SIMT |
| `totalN` | uint32 | 元素总数 |
| `cosValue` / `sinValue` | float | c/s位于Host时的标量值；位于Device时填0占位 |
| `cIsDevice` / `sIsDevice` | uint32 | c/s各自的位置标志，1表示Device指针 |
| `perCoreN` / `remainder` | uint32 | 连续路径每核元素数，前remainder个核各多1个 |
| `tileSize` | uint32 | 连续路径单次搬运的UB上限，单位complex |
| `incx` / `incy` | int32 | stride路径步长 |
| `nthreads` | uint32 | stride路径每block的SIMT线程数 |

##### 4. 分核与数据切分

**连续路径**：

- `numBlocks = min(totalN, aivCoreNum)`，aivCoreNum由`GetAivCoreCount()`获取
- `perCoreN = totalN / numBlocks`，`remainder = totalN % numBlocks`，前`remainder`个核各多处理1个元素
- 不做block对齐取整，避免小n时部分核空转；满块和32B对齐的尾块用`DataCopy`，只有起点或长度不满足32B对齐的尾块才用`DataCopyPad`兜底

**stride路径**：

- `numBlocks = min(ceilDiv(totalN, SIMT_MIN_THREAD_NUM), aivCoreNum)`
- `nthreads = min(alignUp(ceilDiv(totalN, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)`
- kernel内以`blockIdx.x * blockDim.x + threadIdx.x`为起点，按`block_num * blockDim.x`网格步进遍历n个元素
- 小n不会启动满额2048线程，大n仍可占满每block线程数

##### 5. UB切分

连续路径使用TQue三级流水，共5个UB tile：

- `inQueueX`、`inQueueY`：VECIN，MTE2搬入
- `outNewX`、`outNewY`：VECOUT，Vector结果
- `workBuf`：VECCALC，暂存中间乘积

按仓内`UB_SIZE = 248KB`计算，并将tile长度对齐到512B：

```
候选上限 = floor(UB_SIZE / (5 * 8)) = 6348个complex
tileSize = 6144个complex
每tile字节数 = 6144 * 8 = 49152B = 96 * 512B
laneCount = 2 * tileSize = 12288个float
```

选择6144而非6336的理由：三条性能case的单核元素数为16384/32768/65536，按6144切分后的尾块分别是4096/2048/4096个complex，即32KB/16KB/32KB，全部不低于16KB最佳带宽门槛；按6336切分时case2尾块只有1088个complex（8.7KB），会拖低该核最后一段的带宽。6144同时留出8KB的UB余量，避免5块tile刚好顶满248KB。

`laneCount`对应192个repeat，低于`Muls`/`Add`的255个repeat上限。5个tile合计245760B，低于248KB预算，剩余8192B。单核元素数超过tileSize时循环切块，尾块按实际元素数搬运。

512B对齐的收益：三条性能case满核64，单核起始地址为`blockIdx*131072B`等512B整数倍；按6144个complex切分后，三条性能case的尾块4096/2048/4096均为64的整数倍，因此满块和尾块的GM地址、长度全部512B对齐，可全部使用普通`DataCopy`，完全不触发`DataCopyPad`的padding逻辑。非性能用例中不满足32B对齐的尾块才用`DataCopyPad`兜底。512B对齐同时保证相邻核的GM区间不会落在同一个512B请求范围内，规避多核同址访问被硬件串行化。官方`GM地址尽量512B对齐`条目注明该收益仅在A2/A3实测；950上仍保留512B对齐，作为带宽稳定性和核间边界隔离措施。

##### 6. 搬运API选择

- 满块及32B对齐的尾块：使用`DataCopy`，一次搬完`dataCount*8`字节，单次搬运约48KB，远高于16KB的最佳带宽门槛
- 起点或长度不满足32B对齐的尾块：使用`DataCopyPad`，GM侧只要求1字节对齐，自动处理UB尾块填充
- 不使用for循环逐元素搬运；两个向量各一条搬运指令，避免把连续数据拆成小段

##### 7. tilingKey规划

| tilingKey | 条件 | kernel |
| --- | --- | --- |
| 0 | `incx==1 && incy==1` | 连续SIMD：TPipe/TQue，DataCopy/DataCopyPad + Muls/Add |
| 1 | 其余全部组合（含正非1、负、零及混合） | SIMT：GM上网格步进，逐元素旋转 |

`cIsDevice`、`sIsDevice`不进入tilingKey，kernel内按标志选择标量来源，避免指针位置组合导致tilingKey膨胀。

##### 8. c/s标量来源

对c、s分别调用`aclrtPointerGetAttributes`：

- Host指针：Host侧解引用一次，写入`cosValue`/`sinValue`，kernel不再访问该指针
- Device指针：Host侧不解引用，TilingData中填0占位，kernel通过GM指针读取标量
- c、s可独立为Host或Device，支持host/device/mixed三种组合

该设计避免Device标量指针触发D2H拷贝或stream同步，保持接口的异步性。

##### 9. kernel下发

- 连续路径：`csrot_aiv_kernel<<<numBlocks, nullptr, stream>>>(x, y, cPtr, sPtr, tiling)`
- stride路径：`csrot_simt_kernel<<<numBlocks, nullptr, stream>>>(x, y, cPtr, sPtr, tiling)`
- 两条路径均为原地操作，不使用workspace，不调用`aclrtMalloc`
- host热路径不保留逐调用`OP_LOGD`/`OP_LOGI`，避免日志参数求值和函数调用进入21us级时延；诊断信息用编译期宏控制
- 不设置atomic、不清除mask；纯Vector任务显式声明`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，避免Cube核空跑产生额外头开销
- 首版不追加`--cce-no-dcache-flush`：SIMT路径用`*__gm__`直接写GM，不满足该选项前提；若要评估，按性能设计的拆分方案处理

#### 3.2.2 kernel侧设计

##### 1. 连续路径（tilingKey 0）

`CsrotAIV`类分Init与Process两个阶段，Process含CopyIn、Compute、CopyOut：

```
CopyIn:
  满块/对齐尾块: DataCopy(xLocal, xGM[curOffset*2], dataCount*8字节)
                 DataCopy(yLocal, yGM[curOffset*2], dataCount*8字节)
  非对齐尾块:    DataCopyPad(xLocal, xGM[curOffset*2], dataCount*8字节, padParams)
                 DataCopyPad(yLocal, yGM[curOffset*2], dataCount*8字节, padParams)
  EnQue通知Vector阶段MTE2完成

Compute（laneCount = dataCount*2）:
  Muls(newX, xLocal,  cosValue, laneCount)   // round(c*x)
  Muls(work, yLocal,  sinValue, laneCount)   // round(s*y)
  Add (newX, newX,    work,     laneCount)   // round(c*x) + round(s*y)
  Muls(newY, yLocal,  cosValue, laneCount)   // round(c*y)
  Muls(work, xLocal, -sinValue, laneCount)   // round(-s*x)
  Add (newY, newY,    work,     laneCount)   // round(c*y) + round(-s*x)
  EnQue通知MTE3阶段Vector完成

CopyOut:
  满块/对齐尾块: DataCopy(xGM[curOffset*2], newX, dataCount*8字节)
                 DataCopy(yGM[curOffset*2], newY, dataCount*8字节)
  非对齐尾块:    DataCopyPad(xGM[curOffset*2], newX, dataCount*8字节)
                 DataCopyPad(yGM[curOffset*2], newY, dataCount*8字节)
```

- 原始x、y在整个Compute阶段保留在VECIN队列中，结果写入独立VECOUT队列，四个分量全部计算完成后才释放输入，保证原地语义
- `Muls`与`Add`均为单次舍入指令，天然满足“每个乘积、每个加法分别舍入”的数值要求，不使用`Axpy`等FMA语义接口
- 满块和对齐尾块使用`DataCopy`避免padding逻辑；只有非对齐尾块使用`DataCopyPad`，GM侧只要求1字节对齐，Vector阶段只消费实际`laneCount`个元素

##### 2. stride路径（tilingKey 1）

SIMT计算函数逐元素处理一个complex元素，同时读写`2i`与`2i+1`两个float lane：

```
idxX = (incx > 0) ? i : (incx < 0 ? n-1-i : 0)
idxY = (incy > 0) ? i : (incy < 0 ? n-1-i : 0)
laneX = idxX * |incx| * 2
laneY = idxY * |incy| * 2

xr = xGm[laneX];  xi = xGm[laneX+1]
yr = yGm[laneY];  yi = yGm[laneY+1]

nxr = c*xr + s*yr;  nxi = c*xi + s*yi
nyr = c*yr - s*xr;  nyi = c*yi - s*xi

xGm[laneX] = nxr;  xGm[laneX+1] = nxi
yGm[laneY] = nyr;  yGm[laneY+1] = nyi
```

- 负步长：以`n-1-i`为索引，等价于Netlib从`(-N+1)*INC+1`起算并向负方向步进
- 零步长：`idx=0`，同一元素被反复读写；当`incx==0 || incy==0`时，仅`block0.thread0`按i=0..n-1串行执行，其余线程立即返回，保证串行累加顺序与Netlib一致
- 原地安全：四个分量先读入寄存器，再执行四条写入；x、y同址同stride时不会污染源操作数
- 舍入：SIMT路径的`c*xr + s*yr`在优化编译下可能收缩为FMA，设计上通过禁止FMA收缩的写法（volatile中间量或等价编译约束）保证每个乘积先舍入一次、加法再舍入一次
- 标量来源：`c = cIsDevice ? *cPtr : cosValue`，`s = sIsDevice ? *sPtr : sinValue`；同一block内所有线程读取同一GM地址，硬件广播

##### 3. 同步与原地安全

- 连续路径由TQue的EnQue/DeQue管理MTE2→V、V→MTE3同步，不手写SetFlag/WaitFlag
- SIMT路径为纯GM读写，无UB流水，不需要核内同步
- 两条路径均在kernel入口声明AIV-only，不混入Cube代码
- 算子不做隐式stream同步，异步语义由调用方控制

##### 4. 性能设计

**数据通路预算**

case1为n=1048576个complex，x、y各8MB；读16MB、写16MB，共32MB。若按任务书标杆21.69us完成，等效混合带宽为`32MB / 21.69us ≈ 1.475TB/s`。case2同为1.48TB/s量级；case3为128MB/119.38us≈1.07TB/s，压力反而更小。因此优化重点是case1、case2，目标是让MTE2/MTE3持续接近950的可持续混合带宽。

官方高性能Add样例在950上的实测（8192×8192 half，384MB总流量）可作为量级参照：单缓冲、64核、单次搬运43.5KB时Task Duration为251.684us，按总流量/总耗时折算约1.526TB/s；双缓冲为247.665us（约-1.6%）；L2 bypass为182.04us（约-26.5%，但该样例数据量384MB，远超L2，不能直接外推到本任务的32MB）。需要注意Add的读写比为2:1，而Csrot为1:1，写占比更高，实际可达混合带宽可能低于1.526TB/s。因此该数据只能说明单缓冲有达标可能，不能保证；首版以单缓冲+DataCopy+512B对齐为基线，保留双缓冲与L2 bypass两个A/B项，由实测决定是否启用。

**已纳入设计的优化**

| 优化项 | 设计选择 | 依据 |
| --- | --- | --- |
| 核数 | 连续路径用满AIV核（950为64个AIV） | 带宽受限场景需满核并发；官方950 Add样例同样使用64核 |
| 单次搬运粒度 | 约48KB，远高于16KB最佳带宽门槛 | 官方文档指出单次搬运16KB以上才能较好发挥带宽 |
| 512B对齐 | tileSize=6144个complex=96×512B；三条性能case的满块与尾块均为512B整数倍 | 官方950样例建议搬运和计算粒度对齐512B；同时避免相邻核共享同一512B GM请求范围 |
| 同址访问 | 连续路径每核处理互不重叠的连续区间，性能case的核边界均512B对齐 | 多核访问同一512B GM范围会被串行化，核边界对齐后不会共享请求 |
| UB融合 | 6条vector指令的中间结果全部留在UB，不回写GM | 前一步vector输出直接作为下一步输入，避免中间结果搬进搬出 |
| 搬运API | 满块及对齐尾块`DataCopy`；非对齐尾块`DataCopyPad` | 满块无padding需求，避免`DataCopyPad`额外逻辑；只有不满足32B对齐的尾块才需要兜底 |
| 负载均衡 | 均分+前remainder核各多1个元素 | 避免尾核拖尾；三条性能case均可被64整除，无尾差 |
| Kernel类型 | `KERNEL_TYPE_AIV_ONLY` | 纯Vector任务，避免Cube核空跑头开销 |
| Workspace | 无workspace、无`aclrtMalloc` | 原地运算，避免冗余分配与同步 |
| 标量来源 | Host c/s写入TilingData，Device c/s透传GM指针 | Host标量以标量广播参与向量运算，不占GM带宽；Device路径不触发D2H |
| 双缓冲 | 默认单缓冲，保留可选2 buffer | 950 Add样例双缓冲仅-1.6%；本任务计算占比约3%，且单缓冲可保留48KB大块 |
| 输入/输出队列分离 | inQueue与outQueue独立，输入buffer在Compute结束即释放 | 下一tile的MTE2可与上一tile的MTE3重叠；单缓冲只限制Compute(t+1)等待CopyOut(t)释放输出buffer，而计算占比小，已有部分流水重叠 |

**需实测决定的优化**

1. **DoubleBuffer**：若把`InitBuffer`的块数从1改为2，5个tile各减半为24KB，仍在16KB门槛以上；理论收益来自MTE2/MTE3与Vector重叠。由于本任务计算量小，预计收益不超过官方950 Add样例的1.6%，且循环次数翻倍。实现时用编译宏`CSROT_DOUBLE_BUFFER`切换，在同一二进制下做A/B，不预设默认开启。
2. **L2 Cache bypass**：官方Add样例在384MB流量下`CACHE_MODE_DISABLE`带来-26.5%，但本任务case1只有32MB，且评测会对同一组buffer连续采样50次以上，数据可能命中L2；此时bypass会把可命中的读强制打到GM，存在变慢风险。更关键的是，Add样例只对只读输入x/y设bypass、输出z保持NORMAL；本算子x/y既是输入又是输出，`SetL2CacheHint`会同时作用于x/y的读写，收益不确定性更大。默认保持`CACHE_MODE_NORMAL`，提供`CSROT_L2_BYPASS`宏，在平台实测后再决定。
3. **UB bank冲突**：3510的UB为256KB、8个bank group、每组2个bank、每个bank 16KB，组内2读或1读1写。若`msopprof`的`ResourceConflictRatio.csv`显示读读/写写冲突占比高，再在5块buffer之间加32B/64B/256B偏移做对照；官方950 Add样例明确该场景暂不需要考虑bank冲突，因此不预先增加布局复杂度。
4. **核数**：默认满核。若Profiling显示`aiv_mte2_ratio`/`aiv_mte3_ratio`已接近100%但时延仍不达标，再尝试32核或48核做双向夹逼；不能直接套用2201小shape“减核省启动开销”的结论，本任务数据量是MB级，启动开销占比远小于搬运。
5. **DCI尾开销选项**：950的`--cce-no-dcache-flush`可减少尾部DCCI的Clean过程，但其前提是kernel不使用`*__gm__`直接改写GM。本设计的SIMT路径正是用`xGm[idx]=...`直接写GM，因此不能对整个`csrot_kernel.cpp`直接启用。若要评估该选项，需把SIMD与SIMT拆成两个编译单元，只对SIMD文件追加选项，或在SIMT路径显式调用`DataCacheCleanAndInvalid`。该优化放到基线达标之后再评估，不作为首版必选项。
6. **Device c/s的同址标量读**：若c/s在Device内存，64个核会在Init阶段各读一次同一4B标量，落在同一个512B GM请求范围内，可能被串行化。该开销是每核一次性标量读，远小于32MB搬运，但三条性能标杆未指定c/s位置；默认按Host标量测三条标杆，若验收使用device/mixed，则单独记录该路径耗时。若Profiling显示该开销明显，可在报告中说明并改用Host标量路径复测，不为此引入D2H同步或核间广播。

**明确不采用的优化**

- 静态Tensor编程、零tiling入参：官方口径与2201实测的收益在0.1us量级，对21us级算子占比不足0.5%，却会增加同步手写和可维护性风险，收益/风险不成立
- 减少核数：小shape、192KB流量下的8核最优结论不能外推到MB级带宽受限场景
- FMA/`Axpy`：计算量约3%，不是瓶颈；保留`Muls+Muls+Add`的分步舍入语义，避免为省一条向量指令引入精度争议
- Counter模式、mask复用、UB融合：laneCount=12288是64的整数倍，无尾块，Normal模式一次repeat=192即可，不需要Counter模式；6条vector指令已在UB内连续完成，UB融合已满足
- L2 Cache切分：本算子无多核共享数据复用，不涉及

**性能验证方法**

- 性能用例必须预分配Device buffer，算子调用前warmup5次，再循环50次以上；用stream事件或`aclrtSynchronizeStream`包围循环计时，除以有效次数得到平均耗时
- 不能直接用GTest输出的整用例耗时（毫秒级，且含host准备、golden计算与比对），该值只能作为保守上界
- 同时记录端到端（host调用+同步）和kernel Task Duration：若端到端超出标杆而kernel达标，优化重点转到host路径（减少校验分支、去掉逐调用日志、确认`aclrtPointerGetAttributes`开销）
- 用`msopprof`采集`PipeUtilization.csv`、`Memory.csv`、`L2Cache.csv`、`ResourceConflictRatio.csv`，分别看MTE2/MTE3占比、GM读写带宽、L2命中率、UB冲突率
- 每版只改一个变量，记录Task Duration与各流水；回归时保留上一版可执行文件
- 先测单缓冲+DataCopy+512B对齐的基线，再依次评估double buffer、L2 bypass、bank stagger，任何一项没有稳定收益就回退

## 支持硬件

| 芯片版本 | 支持情况 | 说明 |
| --- | --- | --- |
| Ascend950PR（arch35） | √ | 本任务实现 |
| Ascend950DT（arch35） | √ | 同一arch35实现，按任务书要求以950PR验收 |
| Atlas A2/A3（arch22） | √ | 保留既有arch22实现，本任务不改动 |

## 算子约束限制

- `n`为int，n≤0为合法no-op，返回成功且不修改x、y
- `incx`、`incy`可正、负、零，均不拦截；零步长按Netlib语义串行累加
- `(n-1)*incx`、`(n-1)*incy`须落在int32范围内，否则返回`ACLBLAS_STATUS_INVALID_VALUE`
- c、s为float实数标量；复数正弦属于`crot`，不在本算子范围
- x、y必须位于Device内存并原地更新；算子不返回视图
- 异步执行依赖`aclblasSetStream`绑定的stream，读回Device结果前须同步stream
- 不支持额外leading dimension padding、broadcast和dynamic shape

# 可维可测分析

## 精度标准/性能标准

### 精度标准

按任务书§3.2与生态算子开源精度标准，complex64的实部、虚部分别按FLOAT32判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| --- | --- | --- | --- | --- |
| COMPLEX64（实部/虚部按FLOAT32分量） | 2^-13（1.2e-4） | 2^-13（1.2e-4） | 0.99 | 1e-2或32×ULP |

逐元素判定条件：

```
|actual - golden| ≤ atol + rtol × |golden|
```

用例同时满足`matched_ratio ≥ 0.99`且`max_abs_error ≤ 1e-2或32×ULP`时，判定精度通过。

测试工程同时按CSV列配置MERE/MARE辅判据：`mere_threshold=2^-13`、`mare_multiplier=10.0`，复数按实部、虚部分别统计。

golden由cblas（Netlib BLAS 3.10.0复数实现）的`cblas_csrot`生成；测试入口的handle、n≤0、空指针校验与算子保持一致，再调用cblas。

### 性能标准

测试设备为Ascend950PR，COMPLEX64输入，先warmup 5次，再有效采样50次以上取平均：

| case | n | incx | incy | 标杆耗时（Avg time，us） |
| --- | --- | --- | --- | --- |
| 1 | 1048576 | 1 | 1 | 21.69 |
| 2 | 2097152 | 1 | 1 | 43.19 |
| 3 | 4194304 | 1 | 1 | 119.38 |

三条标杆均对应连续路径，验收时以整算子端到端平均耗时为准；需要单独观察kernel耗时时，按测试指导用msprof采集`OpBasicInfo.csv`的`Task Duration`。

### 内存标准

原地逐元素运算，无workspace，不涉及额外内存要求；自测报告记录测试用例的Device内存占用。

## 测试方案

### 测试工程

参照`test/rot/srot`新建arch35 CSV驱动GTest工程：

```
test/rot/csrot/
  CMakeLists.txt                 # arch35使用GTest，arch22保留原冒烟测试
  csrot_param.h                  # CSV参数结构，继承BlasTestParamBase
  csrot_golden.h                 # cblas_csrot golden与入口校验
  arch35/
    csrot_test.cpp               # GTest用例与CSV驱动
    csrot_npu_wrapper.h          # Device内存管理、host/device/mixed标量指针
    csrot_test.csv               # 任务提供的1200条用例
```

`test/rot/csrot/CMakeLists.txt`按SOC分支：

- arch35：`ops_blas_add_gtest_tests(${OPS_BLAS} csrot_test)`
- arch22：保留原`ops_blas_add_tests(${OPS_BLAS})`，不影响既有A2/A3冒烟测试

### 用例覆盖

随任务提供的CSV共1200条：

| 类别 | 前缀 | 条数 | 覆盖内容 |
| --- | --- | --- | --- |
| L0基础 | TC_L0 | 8 | 小尺寸×基础步长组合（±1/±2） |
| L1尺寸 | TC_SQ | 38 | 1至1048576，含质数、2的幂及±1、非对齐值 |
| L2步长 | TC_INC | 49 | incx×incy全组合（±1/±2/±3/0） |
| L3旋转因子 | TC_CS | 20 | 单位旋转、纯交换、零、负值、大值等特殊(c,s) |
| L5填充 | TC_FL | 12 | 均匀/正态、全零、交替、极端值、Inf、NaN |
| L5b指针位置 | TC_PM | 12 | host/device/mixed×连续/步长/负步长/单位旋转 |
| L6边界 | TC_ED | 10 | n=0/负n no-op、零步长、x/y/c/s空指针 |
| EX扩展 | TC_EX | 850 | 尺寸×步长×旋转因子×指针位置的确定性采样 |
| PF性能 | TC_PF | 200 | 3条任务书典型case+规模扫描+device标量+混合 |

随任务用例已移除int32偏移溢出用例，但任务书§2.4仍要求该负向行为，测试工程需补充：

- `n`与`inc`组合使`(n-1)*|inc|`超过INT32_MAX，期望`ACLBLAS_STATUS_INVALID_VALUE`
- `n>0`时x或y为空、c或s为空，期望`ACLBLAS_STATUS_INVALID_VALUE`
- `handle==nullptr`且n≤0，期望`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（校验优先级）
- `n≤0`时x/y/c/s全为空，期望`ACLBLAS_STATUS_SUCCESS`
- incx或incy为0，验证同一元素串行旋转
- 负步长从尾端起算，与cblas结果比对
- `c==1&&s==0`不短路，验证正常旋转路径
- Inf/NaN输入：NaN按`std::isnan`逐元素检查；单位旋转下的Inf按位精确比对
- x、y同址同stride的原地别名场景

### 自测流程

1. 按任务书环境获取方式申请Ascend950PR算力，确认CANN9.1.0环境
2. 将算子和测试代码放入ops-blas对应目录，执行`bash build.sh --soc=ascend950 --ops=csrot`
3. 精度：`python verify_accuracy.py --repo <ops-blas> --soc ascend950 --csv ./csrot_test.csv`
4. 性能：`python verify_performance.py --repo <ops-blas> --soc ascend950`
5. 需要kernel级耗时时，用`msopprof`（或任务指导中的`msprof op`）采集，重点看`PipeUtilization.csv`、`Memory.csv`、`L2Cache.csv`、`ResourceConflictRatio.csv`
6. 性能A/B顺序：单缓冲+DataCopy+512B对齐基线 → double buffer → L2 bypass → bank stagger；每轮只改一个宏，记录Task Duration与各流水占比
7. 输出自测报告：用例参数、实部/虚部精度比对截图、性能数据截图、内存占用，按任务书模板填写
8. 性能测试确认warmup 5次、有效采样>50次

## 兼容性分析

- `aclblasCsrot`参数序列与`cublasCsrot`一一对应，handle及参数顺序无需额外映射
- 接口声明已存在于`include/cann_ops_blas.h`，本任务不新增、不修改公共头文件
- arch22旧实现保留在`blas/rot/arch22/`，按SOC分派，不影响Atlas A2/A3产品线
- 合入时更新`blas/rot/README.md`：Csrot产品支持表标注Ascend950PR/950DT支持，接口原型与头文件保持一致
- 测试工程仅在arch35启用GTest，不改变arch22测试入口

# 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 编译器将`c*x+s*y`收缩为FMA | 与Netlib分步舍入语义存在ULP级差异，相消场景误差放大 | 连续路径使用`Muls+Muls+Add`；stride路径用volatile中间量或编译约束禁止FMA收缩 |
| Device标量指针被Host解引用 | 非法访问或隐式同步 | 先用`aclrtPointerGetAttributes`判定位置；Device指针只透传给kernel |
| 零步长被多线程并发 | 同一地址竞争，结果与Netlib串行累加不一致 | 零步长时仅block0.thread0串行执行，其余线程返回 |
| 负步长索引错误 | 访问越界或结果反向 | 用`n-1-i`映射索引，host侧用int64检查`(n-1)*inc`范围 |
| 小n启动满核 | 启动开销占比高 | `numBlocks=min(n,aivCoreNum)`；stride路径同样按n收敛block数 |
| UB切分超过预算 | 编译失败或运行时越界 | `tileSize`按`UB_SIZE/(5*8)`计算，取6144个complex（96×512B），5块合计245760B<248KB，留8KB余量 |
| 现有arch22测试与新GTest冲突 | 构建多个main或链接失败 | CMake按SOC分支，arch35才启用GTest；arch22保留原入口 |
| `--cce-no-dcache-flush`在CANN9.1.0不支持或误用 | 编译失败，或SIMT直接写GM导致Cache一致性风险 | 首版不启用；若要启用，拆分SIMD/SIMT编译单元只对SIMD文件追加，或SIMT显式清Cache；先做一次编译验证 |
| L2 bypass误开导致变慢 | case1数据仅32MB，重复采样可能命中L2，bypass会把可命中读打到GM | 默认NORMAL；用`CSROT_L2_BYPASS`做A/B，只有实测稳定收益才开启 |
| DoubleBuffer减半tile后收益不足 | 循环次数翻倍，单次搬运从48KB降到24KB | 默认单缓冲；用`CSROT_DOUBLE_BUFFER`做A/B，收益不稳定则回退 |

# 交付与文件规划

## 代码文件

```
blas/rot/arch35/
  csrot_host.cpp          # 入口校验、tiling、标量位置判定、kernel下发
  csrot_kernel.cpp        # SIMD连续路径 + SIMT stride路径
  csrot_kernel.h          # kernel_do声明
  csrot_tiling_data.h     # CsrotTilingData与对齐常量
```

若后续启用DCI选项，需将SIMD/SIMT拆成两个编译单元，CMake只对SIMD文件追加`--cce-no-dcache-flush`；首版不启用。

性能A/B宏（仅编译期开关，默认关闭）：

- `CSROT_DOUBLE_BUFFER`：TQue块数1/2切换
- `CSROT_L2_BYPASS`：x/y是否`SetL2CacheHint(CACHE_MODE_DISABLE)`
- `CSROT_BANK_STAGGER`：5块UB tile之间追加32B/64B/256B偏移

## 测试文件

```
test/rot/csrot/
  CMakeLists.txt          # arch35 GTest / arch22 原测试分支
  csrot_param.h
  csrot_golden.h
  arch35/
    csrot_test.cpp
    csrot_npu_wrapper.h
    csrot_test.csv
```

## 文档更新

- `blas/rot/README.md`：更新Csrot产品支持表、函数原型、约束说明与调用示例
- 自测报告：按任务书模板提交精度、性能、内存截图与日志

## 验收交付件对应

| 交付件 | 本设计对应内容 |
| --- | --- |
| 算子设计文档 | 本文档 |
| 自测用例及测试代码 | `test/rot/csrot/arch35/`与CSV，README说明复现步骤 |
| 自测报告 | 精度、性能、内存数据与截图 |
| 待验收代码地址 | 个人仓分支、`blas/rot/arch35/`目录，邀请Ascend-CANN为开发者 |

# 参考资料

以下口径来自CANNCommunityEdition 9.2.0-beta.2开发指南与asc-devkit官方样例：

1. 优化建议总览：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/optimization_summary.md
2. 分析性能数据（搬运/计算理论耗时、MTE2+MTE3共享带宽）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/performance_analysis/analyze_performance_data.md
3. NPU架构版本3510（UB 256KB、用户可用248KB、AIV每拍256B、bank结构、指令队列深度）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_3510.md
4. 尽量一次搬运较大的数据块（16KB门槛）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/transfer_larger_data_blocks.md
5. 高效使用搬运API（DataCopyParams一次搬完）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/efficient_transfer_api.md
6. 设置L2 CacheMode（bypass适用前提）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md
7. 避免UB bank冲突（3510规格）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md
8. 开启DoubleBuffer：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md
9. 设置DCI编译选项减少尾开销（仅950）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/overhead_optimization/dci_compile_option.md
10. 设置合适核数与Kernel类型：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/overhead_optimization/core_count_kernel_type.md
11. 限制TilingData大小：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/overhead_optimization/limit_tiling_data_size.md
12. 避免TPipe在对象内创建：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/overhead_optimization/avoid_tpipe_init_in_object.md
13. Add高性能样例（含950实测表）：https://gitcode.com/cann/asc-devkit/blob/9.2.0-beta.2/examples/01_simd_cpp_api/05_best_practices/00_vector_compute/add_high_performance/README.md
14. 避免同地址访问（512B GM请求串行化）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/avoid_same_address_access.md
15. GM地址尽量512B对齐：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/gm_address_512b_alignment.md
16. 通过UB融合实现连续vector计算：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/vector_compute/ub_fusion_vector_compute.md
17. Vector算子Counter模式：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/vector_compute/vector_counter_mode.md
