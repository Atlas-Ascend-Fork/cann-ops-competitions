# 需求背景（required）

## 需求来源

本需求来自 CANN 社区任务《aclblasHgemm 算子开发任务书》，基于 CANN 9.1.0 和 Ascend C，为 Ascend 950PR 补充 FLOAT16 通用矩阵乘接口。设计采用社区算子设计文档的需求背景、需求分析、详细设计及可维可测分析结构。

## 背景介绍

算子计算 `C = alpha × op(A) × op(B) + beta × C`，使用 ops-blas 公共句柄接口，通过 handle 绑定的 stream 直调 kernel。A、B、C 及 Host 标量为 FLOAT16，矩阵按列主序保存，内部采用 FP32 累加与系数运算。

实现位于 `blas/gemm/arch35/`。FP16 Cube 模板从同族 GemmEx 实现派生，形成 Hgemm 独立维护的 `hgemm_cube_template.h` 与 `hgemm_pipeline.h`；复用 `GemmExTilingData`、公共 handle、workspace、构建和测试设施。Hgemm 入口调用自己的 Cube kernel，没有直接调用 GemmEx 共享计算模板。Gemm 参数验证两函数从原 Host 抽取为 `gemm_validation.h`，由原 Gemm Host 和 Hgemm 共用。

# 需求分析（required）

## 需求描述

公共声明位于 `include/cann_ops_blas.h`，维数使用 `int`，标量使用 `const aclblasHalf*`，其他产品线共用同一接口名。

```cpp
aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, const aclblasHalf* B, int ldb,
    const aclblasHalf* beta, aclblasHalf* C, int ldc);
```

| 参数 | 类型与位置 | 语义 |
| --- | --- | --- |
| handle | Host 句柄 | 携带 stream 和有效 workspace |
| transa、transb | Host 枚举 | N、T、C；实数 C 与 T 等价 |
| m、n、k | Host int | 非负维数；C 为 m×n，乘积规约长度为 k |
| alpha、beta | Host FLOAT16 标量指针 | 指针非空，分派时读取为 FP32 |
| A、B | Device FLOAT16 只读矩阵 | A：N 时 m×k，T/C 时 k×m；B：N 时 k×n，T/C 时 n×k |
| C | Device FLOAT16 原地输出 | 逻辑 m×n，保留列间 padding |
| lda、ldb、ldc | Host int | 列主序物理列跨度，以元素计 |

前导维约束为 `lda >= max(1, transa==N ? m : k)`、`ldb >= max(1, transb==N ? k : n)`、`ldc >= max(1,m)`。支持九种转置组合、独立前导维、矩形和尾块；不支持广播、batch 或独立于前导维的任意多维 stride。

## 需求拆解

1. 公共声明、Host 校验、列主序映射和句柄式 stream 调用。
2. FP16 输入、FP32 乘积规约及系数运算、最终 FP16 RNE 输出。
3. 空输出、零规约、零系数、非有限值和非法参数处理。
4. 小矩阵及大矩阵分块、双缓冲流水与通用 Vector 后处理。
5. 保留原始 1,200 条测试记录和 200 条 GPU 基线，增加独立补充用例、stream 与内存检查。
6. 按一次 API 的全部必要计算 task 合计性能耗时，保存每例参数、样本和统计结果。

# 详细设计（required）

## 算子分析

### 数学和寻址

一般路径先计算 FP32 乘积 `S(i,j)=Σ op(A)(i,p)×op(B)(p,j)`，再计算 `D(i,j)=FP32(alpha)×S(i,j)+FP32(beta)×FP32(C_old(i,j))`，最后将 D 按 round-to-nearest-even 转为 FLOAT16。

| 元素 | N 模式偏移 | T/C 模式偏移 |
| --- | --- | --- |
| op(A)(i,p) | i+p×lda | p+i×lda |
| op(B)(p,j) | p+j×ldb | j+p×ldb |
| C(i,j) | i+j×ldc | 不适用 |

`k=0` 或 `alpha=0` 不读取 A/B；`beta=0` 不读取原 C。`k=0` 时只做 beta 缩放，不求值 alpha 与乘积。

### 支持硬件和运行环境

| 产品 | 支持情况 | 软件环境 |
| --- | --- | --- |
| Ascend 950PR（arch35） | 支持 | CANN 9.1.0 |

实测设备为 Ascend950PR_9579，28 Cube / 56 Vector 核，额定频率 1650 MHz；驱动 25.7.rc1.10。

## 算子实现

### 文件职责

| 文件 | 职责 |
| --- | --- |
| hgemm_host.cpp | 参数检查、FP16 标量取值、分块与 stream 下发 |
| gemm_validation.h | 与原 Gemm 共用的参数及指针校验 |
| hgemm_cube_template.h | 从同族实现派生的小矩阵 Cube 模板、搬运与输出辅助函数 |
| hgemm_pipeline.h | Hgemm 独立流水线、面板遍历、L1/L0 事件和规约 |
| hgemm_cube_kernel.cpp | FP16/FP32 输出 Cube 入口及 Host launch 封装 |
| hgemm_kernel.h、hgemm_kernel.cpp | epilogue tiling、Vector 缩放/置零/系数运算与写回 |
| test/gemm/hgemm/ | CSV 驱动 GTest、Netlib golden、补充用例与报告器 |

### Host 校验与退化处理

依次检查 handle、转置枚举、非负维数、前导维、alpha/beta 判空；然后处理 `m=0` 或 `n=0`，返回成功且不读取标量值、不访问矩阵、不启动 kernel。非空输出读取 Host 标量并转换为 FP32。

当 `k>0 && alpha!=0` 时 A/B 必须非空；当 `k>0 || beta!=0` 时 C 必须非空。不满足时返回 INVALID_VALUE；非法枚举返回 INVALID_ENUM，空 handle 返回 HANDLE_IS_NULLPTR。

| 退化条件 | 行为 |
| --- | --- |
| m=0 或 n=0 | 不访问 A/B/C，直接返回 |
| k=0、beta=0、C 为空 | 直接返回 |
| k=0 或 alpha=0，C 有效、beta=0 | 仅将逻辑 C 置零，不读旧值 |
| k=0 或 alpha=0，C 有效、beta=1 | C 不变，不启动计算 |
| k=0 或 alpha=0，C 有效、beta 为其他值 | 只对逻辑 C 执行 FP32 beta 缩放后转 FP16 |

### 列主序映射

Host 交换输入 A/B、m/n 和各自的前导维，保留相应转置标记，将输出 C 的列主序缓冲解释为 C 转置的行主序缓冲。内核坐标为 `M=n、N=m、K=k`，左输入为 B，右输入为 A。C 和 T 均转换为同一个实数转置布尔值，不生成整矩阵转置副本。

### 冻结 v2 分块和分派参数

令 P 为运行时 Cube 核数，`S=ceil(M/256)×ceil(N/256)`。以下参数均使用内核行主序坐标。

| 路径条件 | M×N 输出块 | K 方向分块 |
| --- | --- | --- |
| min(M,N,K)<256 | 128×128 非流水线 | BK=256 |
| min(M,N,K)>=256 且 S<=P/2 | 128×128 流水线 | PANEL=256、INNER_K=128 |
| min(M,N,K)>=256 且 P<S<3P | 128×256 流水线 | PANEL=256、INNER_K=64 |
| min(M,N,K)>=256 的其他情况 | 256×256 流水线 | PANEL=256、INNER_K=64 |

`P/2` 使用整数除法。非流水线路径以 128 输出块为基础寻找不超过核数的二维分区，保证实际核心从矩阵有效区域开始。流水线路径使用核数 `min(P, panels)`，每核按 blockId 加 blockNum 步进遍历面板；`GROUP_M=8` 将相邻 M 面板组织成 band，再遍历 N 面板，最后一个 band 使用实际剩余 M 面板数。

流水线以两个 L1 stage 交替搬入 A/B 的 K 面板；面板内按 INNER_K 搬入 L0 并执行 FP16×FP16、FP32 Mmad。L0 stage 数由 L0A 容量和 `max(BM,BN)×INNER_K×sizeof(half)` 决定。MTE、Cube、Fixpipe 使用明确事件建立缓冲复用依赖；每个输出面板完成全部 K 规约后再写出，不进行跨核 split-K。

A/B 地址按实际前导维计算，Nd2Nz 与 LoadData 处理转置；尾块使用有效 M/N/K 尺寸和对齐后的片上布局。Fixpipe 仅写逻辑输出区域。

### 直接输出、workspace 与后处理

当 `alpha=1、beta=0` 且公共参数 m、ldc 均为 16 的倍数时，Cube 完成 FP32 全 K 规约后直接 RNE 写回 FP16 C，不使用乘积 workspace，也不启动 epilogue。

其他乘积路径先写 FP32 workspace，再在同一 stream 上执行 Vector epilogue。workspace 行跨度为 `ldw=align_up(m,16)`，需求为 `n×ldw×sizeof(float)` 字节。Host 检查跨度和字节数溢出；不足或指针为空时返回 ALLOC_FAILED。调用方通过已有 aclblasSetWorkspace 提供空间，接口不在每次调用中同步分配或释放。

Vector 单块最多处理 12,288 个元素。一般路径先读取 FP32 乘积并乘 alpha；只有 beta 非零时读取旧 C、提升为 FP32 并参与运算；最后 RNE 转回 FP16。alpha=1、beta=0 且 ldw 不大于该块大小时，可合并多列读取和转换，写回每列有效 m 个元素，跳过 padding。退化置零/缩放复用此后处理，不建立 A/B 访问路径。

所有计算使用 handle->stream。Host 在返回前完成标量取值，kernel 不持有 Host 标量指针；调用方保持输入、输出及 workspace 有效直到 stream 完成。不同 stream 使用独立 handle/workspace。测试在读取输出前检查提交状态及同步状态。

## 算子约束限制

输入输出为 FLOAT16 实数，Host 标量模式；A/B 只读，C 原地覆写，列间 padding 保持不变。非空指针的存储长度由调用方保证；C 与 A/B 的存储重叠不在本任务支持范围内。维数为运行时 int，测试覆盖到各轴 16384，实际运行规模还受内存和底层地址范围约束。

# 可维可测分析

## 精度验证

使用 Netlib BLAS 3.10.0 的 cblas_sgemm，以已量化为 FLOAT16 的矩阵和标量提升为 FP32 后计算 golden，最后转 FP16 RNE。逻辑 m×n 全矩阵比较，有限值容差为 `2^-10 + 2^-10×abs(golden)`，匹配率至少 0.99，最大误差不大于 `max(0.1,32×ULP(最大误差点golden))`。

双方 NaN、双方同号 Inf、NaN 与 Inf 配对按任务书单列规则处理；有限/非有限配对或异号 Inf 配对不通过。非有限元素独立统计，不并入有限值匹配率和最大误差。测试同时检查 A/B 内容、C padding 及分配前后哨兵。

原 CSV 共 1,200 条：1,000 条精度/功能记录与 200 条性能 shape，后者也执行全矩阵数值检查。另有 81 条补充记录、1 个双 stream/workspace 连续复用测试及 1 个 Netlib 串并行一致性测试。

实际执行 1,283 项：1,251 项矩阵数值比较、29 项状态检查、1 项无输出比较、stream 及 golden 各 1 项；1,283 项通过。XML 中另登记的 200 个 DISABLED 性能测试不计入这 1,283 项。

独立插桩构建对选定边界/分支执行 52 项测试、94 次 kernel 内存检查；这两个计数用于描述所选内存检测范围。

## 性能采集和统计

使用 installed msopprof 的隔离副本，注入库回移植 device-id 修复，采集 BasicInfo 的 task duration。固定 application replay、HGEMM_PROFILE_SAMPLES=60、工具 warm-up=5。每批 10 个 case，共 20 批、200 个 case。

应用每例执行五次真实 Hgemm API 预热，随后连续执行 60 次有效调用。工具 application 模式自身的五次 warmup 是 1 MB D2D memcpy，与应用的五次真实 API 调用分别记录。每次 API 的全部必要 Cube 和 epilogue task duration 相加后作为该次耗时，不计矩阵生成、C 恢复拷贝、CPU golden 或 GTest 总时间。

对每例取 60 个有效 API 耗时的算术平均，计算 `GPU_us/NPU_mean_us`；全套平均为 200 个 case 倍率的算术平均。GPU 基线的 gpu_ms 乘 1000 转 us，按 shape、转置及相应标量/前导维核对。另从同组每例 iteration 5..14 连续取首 10 个有效样本，独立统计，不排序、不选最快样本。

| 统计范围 | 每例样本数 | case 数 | GPU/NPU>=0.4 | GPU/NPU>=1.0 | 倍率算术平均 |
| --- | --- | --- | --- | --- | --- |
| application 完整采集 | 60 | 200 | 200/200（100%） | 142/200（71%） | 1.3395355386582186 |
| 同组连续首 10 派生 | 10 | 200 | 200/200（100%） | 142/200（71%） | 1.3513431191063046 |

保存 23,205 份 OpBasicInfo CSV 及对应原始记录，23 条 875 MHz 低频记录与其他 1650 MHz 记录全部保留，不做离群值删除、替换或频率校正。原始 CSV/log 折叠为可复原 JSONL，复原后通过严格报告器检查 kernel 名称、ordinal、调用状态、warmup 标记、样本数量和完整 case 集合。

## 兼容性与交付核对

公共 Hgemm 签名、枚举和状态码遵循已有库接口；新增实现限定在 arch35。共享验证从原 Gemm Host 原样抽取，GemmEx 的 Host、kernel 和 tiling 数据结构未作生产修改。

原始实测清单包含 20 个源码/测试/文档文件；正式实现提交更新其中两个 README，并新增离线报告器，共 21 个变更文件、相对基底补丁、源码 SHA256 清单、设计、复现步骤及精度/性能/内存原始记录。基底为 ops-blas `05b96ea22cea926ecfd43fc7ac691a795522bd61`。README 的产品表、分块参数和派生复用说明作了文档更新；实测源码清单和最终交付清单分别记录相应字节身份。

## 提交版本与待评审事项

实现代码仓：https://gitcode.com/gcw_fY1MdJ3B/ops-blas 。
实现分支：`feat/hgemm-950pr-submit-20261028`。
实现 PR：https://gitcode.com/cann/ops-blas/pull/737 。
固定版本：https://gitcode.com/gcw_fY1MdJ3B/ops-blas/commit/e1e19ffe5d9c8f008805c65639ffe4a0e5f7a324?ref=feat%2Fhgemm-950pr-submit-20261028 。
设计 PR：https://gitcode.com/cann/cann-ops-competitions/pull/2497 。

本次同步将前期方案更新为实际交付的 v2 实现及已有实测证据。生产实现、设备测试和 CSV
与实测清单一致；后续补充的是 README 和离线性能复算入口，不将新增提交误称为重新实机测量。

请评审确认以下事项；在获得结论前不声明最终验收通过：

1. 任务书第 3.3 节要求逐例至少 0.4 倍 GPU、倍率平均至少 0.5；第 7.4 条另写达到 GPU。
   当前两种统计口径均为 200/200 达到 0.4，142/200 达到 1.0。请确认最终性能门槛。
2. 本次使用五次应用真实预热加 60 个连续有效样本，并从同组取连续首 10 个；不是两次独立重测。
   工具 application 预热为 D2D memcpy，应用额外执行真实 Hgemm 预热。请确认该采集方式。
3. Cube 计算模板由同族 GemmEx 派生后独立维护，共享 tiling、校验、handle/workspace 和工程设施；
   没有直接调用共享 GemmEx 计算模板。请确认这种派生复用方式满足任务书第 2.2 节。

设计 PR、实现 PR 的创建不等于评审通过或合入，已有自测结果不等于平台最终验收结论。
