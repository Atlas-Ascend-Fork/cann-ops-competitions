# aclblasSgemm / aclblasCgemm / aclblasHgemm 算子设计文档（Atlas A2/A3，arch22）

> 任务：11 月社区任务「gemm 类 3 个算子开发（A2/A3）」
> 接口：`aclblasSgemm`（FP32）/ `aclblasCgemm`（COMPLEX64）/ `aclblasHgemm`（FP16），分别对标 cuBLAS `cublasSgemm` / `cublasCgemm` / `cublasHgemm`
> 实现仓：ops-blas（https://gitcode.com/cann/ops-blas ），`blas/gemm/arch22/`；测试：`test/gemm/arch22/`、`test/gemm/cgemm/arch22/`、`test/gemm/hgemm/arch22/`
> 私有开发仓（代码 + 自验用例包）：https://gitcode.com/zhangfeng1133/aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3

# 需求背景（required）

## 需求来源

社区任务要求参考 cuBLAS `cublasSgemm` / `cublasCgemm` / `cublasHgemm`，在 Atlas A2/A3（910B3，arch22/DAV_2201）上使用 Ascend C 基于 ops-blas 开源仓工程开发通用矩阵乘算子族，完成设计、开发、测试全流程，验收通过后合入 `blas/gemm/arch22/`。

## 背景介绍

ops-blas 以句柄式 BLAS 接口工程模式提供 NPU 直调 kernel：host 侧完成参数校验、tiling 规划与 workspace 申请，kernel 侧走 Ascend C Matmul 高阶 API，通过 handle 绑定的 stream 异步下发。

仓内 gemm 族 arch22 现状（本次任务的设计基线）：

| 算子 | arch22 现状 | 本任务工作 |
| --- | --- | --- |
| aclblasSgemm（FP32） | `blas/gemm/arch22/gemm_host.cpp` + `gemm_kernel.cpp` 已合入，多路径派发 + wholegrid 大方阵快路径 | 结构化设计说明 + 任务书符合性核对 + README 支持表更新 |
| aclblasCgemm（COMPLEX64） | `blas/gemm/arch22/cgemm_host.cpp` + `cgemm_kernel.cpp` 已合入，三阶段实数分解 | 同上 |
| aclblasHgemm（FP16） | **仓内不存在**（仅有 arch35 `HgemmStridedBatched` 与 950PR 单矩阵实现） | **新增**公开声明、host/kernel 实现、测试与文档 |

两个硬件前提决定实现形态：

1. arch22 Cube 不支持 COMPLEX64 原生矩阵乘，复数乘加必须分解为实数矩阵乘（任务书明确不采用 3M 分解）；
2. arch22 **没有 L0C→GM 的 FP16 收窄路径**（950PR 可用 `Fixpipe quantPre=F322F16` 直写 FP16 GM，arch22 上该路径输出错误），FP16 GEMM 必须「Cube 写 FP32 产品平面 + AIV `Cast(CAST_RINT)` 收窄」两段式。

# 需求分析（required）

## 需求描述

三个算子共用同一语义框架 `C = alpha * op(A) * op(B) + beta * C`：

| 项 | aclblasSgemm | aclblasCgemm | aclblasHgemm |
| --- | --- | --- | --- |
| A/B/C 与 alpha/beta | FLOAT32 | COMPLEX64（实/虚各 FP32） | FLOAT16（`aclblasHalf`，Host 标量） |
| 累加精度 | FP32（Cube L0C） | 4 个 FP32 实数乘积 | FP32（L0C），输出 RNE 收窄 FP16 |
| `OP_C` 语义 | 等价 `OP_T` | **共轭转置，与 `OP_T` 结果不同** | 等价 `OP_T` |

共性语义（任务书 §2.1/§2.4/§2.5）：列主序、按 lda/ldb/ldc 描述的带填充布局；`beta=0` 不读 C 旧值；`m=0` 或 `n=0` 为合法 quick return（不引用 A/B/C、不启动 kernel）；`k=0` 或 `alpha=0` 只做 `C = beta*C`；仅写 C 的逻辑 m×n 区域；校验顺序 handle → 枚举 → m/n/k → lda/ldb/ldc → alpha/beta 与 A/B/C 指针；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，阶数/前导维/空指针非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。

## 需求拆解

1. `aclblasHgemm` 新增公开声明与 arch22 实现；S/C 既有实现按任务书逐条核对；
2. 列主序到 Cube 行主序的统一映射（`C^T = op(B)^T * op(A)^T`，A/B 互换角色、m/n 互换轴）；
3. 复数（Cgemm）与 FP16（Hgemm）在 arch22 Cube 上的落地策略；
4. FP32 精度保持：Cube 默认精度模式在某些 CANN/SOC 组合下为 HF32，kernel 入口显式关闭；
5. 规避 arch22 实数入口已知缺陷 ops-blas#1077；
6. 精度/性能/测试按任务书 §3 全量覆盖。

# 详细设计（required）

## 算子分析

### 数学公式

```
C = alpha * op(A) * op(B) + beta * C
```

- op(A) 为 m×k、op(B) 为 k×n、C 为 m×n，均列主序；A 为 lda×k（transa=N）或 lda×m（否则），B 为 ldb×n（transb=N）或 ldb×k（否则），C 为 ldc×n。
- Sgemm/Hgemm 为实数乘加；Cgemm 按 `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` 直接展开（4 个实数乘积，非 3M）：

```
re = Ar*Br − Ai*Bi
im = Ar*Bi + Ai*Br
```

- 内部计算统一为行主序 `C^T = op(B)^T * op(A)^T`；转置操作数以 `MatmulType<..., T, isTrans>` 表达。

### 支持数据类型

| 算子 | A/B/C | alpha/beta | 中间量 |
| --- | --- | --- | --- |
| aclblasSgemm | float | float | FP32（L0C 累加 + GM 产品平面） |
| aclblasCgemm | aclblasComplex | aclblasComplex | FP32 实/虚平面 |
| aclblasHgemm | half | half（Host 指针，入 kernel 前转 FP32） | FP32 产品平面 + AIV 收窄 |

### 支持形状

m/n/k ≥ 0 任意组合（含 1、质数、2 的幂 ±1 等尾尺寸）；前导维度满足任务书 §2.4 下界；不支持广播；不要求 dynamic shape 特殊处理（全部运行时入参，host 侧动态 tiling 与 launch）。

## Host/Tiling 设计

### 共性设计（三个算子一致）

- **校验与快速返回**：顺序为 handle → transa/transb 枚举 → m/n/k 非负与 alpha/beta 非空 → `m==0||n==0` 返回 `ACLBLAS_STATUS_SUCCESS` → lda/ldb/ldc 下界 → A/B/C 指针（`k>0 && alpha!=0` 时 A/B 非空；m>0&&n>0 时 C 非空）。`k==0 || alpha==0` 时 `beta==0` 只清零逻辑 m×n、`beta==1` 直接返回、其余做 `C = beta*C`，全程不引用 A/B、不启动 Cube kernel。
- **workspace**：统一经 `EnsureDefaultWorkspace` 按需增长（上限 `ACLBLAS_MAX_WORKSPACE_SIZE` = 2 GiB），不足返回 `ACLBLAS_STATUS_ALLOC_FAILED`；核数一律运行时查询（`GetAicCoreCount`/`GetAivCoreCount`），禁硬编码。
- **下发序列**：pack（如需要）→ cube → vector（accumulate/epilogue/scale），同一 stream 顺序即依赖；不用 MIX task-type 宏（该宏在 arch22 存在证据冲突）。

### aclblasSgemm：路径判定与 tiling

| 路径 | 门限 | 说明 |
| --- | --- | --- |
| wholegrid square | m,n,k ∈ [2048,8192] 且 m%16=0、n%16=0、k%32=0 | 每 block 直映一个 256×256 输出 tile，必须全部发射；`computeK` 按 transA 对齐 128/32，规避右转置半 baseK 管线惩罚 |
| direct whole-problem | m,n,k ≤ 1024 且 m%16=0、n%16=0、k%8=0 且 lda/ldb 紧凑 | 裸 GM 直喂 4 个 rows Matmul 特化（Nn/Nt/Tn/Tt） |
| whole-problem pack | 非 direct 且 m,n,k ≤ 1024 | pack 一次后走同一组 rows 特化 |
| 2D-tiled NN | direct 且 transA=transB=0 且 n≤128、m≥512、k≥512 | 4×5=20 个 256×208 tile + scatter |
| host tile | 其余 | host 按 1024 切 M/N/K，每 tile pack → rows matmul → accumulate → epilogue |

workspace 布局（float 元素、8 元素对齐）：`packA = pm*pk`、`packB = pn*pk`、`product` 与 `accum` 各 `pm*pn`（K 多 tile 时 product→accumulate 中转）。

### aclblasCgemm：三阶段 host 编排

1. **Phase 0（AIV 拆分）**：对 A、B 各发一次 split kernel，按列 round-robin 分核；`GatherMask` 抽实/虚部写成列主序实数平面（`packedLd = Align(rows,16)`）。共轭不在本阶段取反，由 Phase 1 标量符号承载，`OP_T`/`OP_C` 四种组合共用同一拆分。
2. **Phase 1（AIC，四个实数乘积）**：构造子句柄 `sub{stream, workspace=scratch, workspace_owner=User}`（禁止实数入口回收仍在使用的平面），把 4 个乘积交给 arch22 `aclblasSgemmEx`；符号表 `signA/signB = (op==OP_C ? −1 : +1)`，`aiBi = −signA*signB`、`arBi = signB`、`aiBr = signA`，调用前把 `OP_C` 降级为 `OP_T`。**K 按 `CGEMM_K_CHUNK=1024` 分块**，部分积在同一 FP32 平面精确累加——这是对 ops-blas#1077（实数入口 `K>1024` 且 M/N 小于 wholegrid 门限时算错）的规避；`transB!=N && m%16==0 && n%16==0` 时按 `avoidLeftTransDirect` 把 `kCount%8==0` 的块长减 4，绕开左转置未对齐传参问题。
3. **Phase 2（AIV 组装）**：combine kernel 按列 round-robin、列内 1024 复元素切片；`Cr = ar*re − ai*im`、`Ci = ar*im + ai*re`，叠加 `beta*C_old` 的复数运算；**零分量跳过**（避免 `0*Inf=NaN` 污染精确分量）；`Gather` 交织写回 C。

workspace：`Ar, Ai, Br, Bi, re, im, prodRe, prodIm, scratch`；平面总量 `2*(planeA+planeB) + 4*planeR`，`scratch = 2*(planeA+planeB+4*planeR) + 4MiB` 余量。

### aclblasHgemm：direct/pack 双路径与三段下发

**对齐策略**：half 的 32B = 16 元素（FP32 为 8 元素）。

- **direct**：`m/n/k` 与 `lda/ldb` 均为 16 元素倍数。Matmul 经 `SetOrgShape` 携带行距，非紧凑但行距对齐的 lda/ldb 可直接消费；
- **pack**：任一维度或行距不满足 16 元素对齐时，先用 AIV pack kernel 把 A/B 重排为行对齐内部平面（`Align(m,16)/Align(n,16)/Align(k,16)`），Cube 只见 NN 紧凑操作数。

workspace（32B 对齐分段）：`packA = Align(m,16)*Align(k,16) * 2B`、`packB = Align(n,16)*Align(k,16) * 2B`、`product = n * Align(m,16) * 4B`；direct 路径只申请 product。

## Kernel 设计

### aclblasSgemm（`blas/gemm/arch22/gemm_kernel.cpp`）

- **Cube kernel**：`MatmulImpl<MatmulType<GM,ND,float[,true]>, ..., float>` + 静态 tiling（行所有包络 `singleM=64, singleN=1024, singleK=1024, baseM=64, baseN=256, baseK=64`），`Init(nullptr, &pipe)` 不经 host tiling 库；入口处 `SetHF32Mode(DISABLE)`（6 处）；`SetOrgShape` 携带行距与转置标志，`SetSingleShape` 描述本 block 实际行列（含尾块）；`IterateAll(output, 0)` 覆盖写。
- **分核策略**：`cubeBlocks = GetAicCoreCount()`；`GetRowsWindow` 把内部 C^T 的行（公开 n）按 whole waves 切给各 Cube block；wholegrid 路径一个 block 一个 256×256 tile、无 block 循环。
- **AIV kernel**：pack（raw 取向保留 / generic 地址映射两种模式）、accumulate（首块覆盖、其余相加）、epilogue（三阶软件流水：GM→UB、alpha/beta 向量运算、UB→GM）、zero（只写逻辑区）。

### aclblasCgemm（`blas/gemm/arch22/cgemm_kernel.cpp`）

仅含 AIV kernel：split（L243）、accum（L250）、combine（L302）；复数运算全部在 FP32 域按分量进行；combine 的 tiling 为 `CgemmCombineTilingData{m,n,ldc,ldRe,ldIm,hasTemp,alphaRe,alphaIm,betaRe,betaIm}`；委派实数 GEMM 恒以 `beta==0` 调用（避免进入实数入口的 `beta!=0` epilogue 路径）。

### aclblasHgemm（`blas/gemm/arch22/hgemm_kernel.cpp`，本次新增）

- **`hgemm_arch22_pack_kernel`（AIV，仅 pack 路径）**：按输出行分核、经 `TQueBind` UB 队列做连续段拷贝（GM→UB→GM，不做转置）；每行只拷逻辑长度，行距改为对齐后的打包行距；行尾对齐区永不读取。
- **`hgemm_arch22_cube_kernel`（AIC）**：`MatmulImpl<MatmulType<GM,ND,half,TransB>, MatmulType<GM,ND,half,TransA>, MatmulType<GM,ND,float>, float, StaticTiling>`，四组转置特化；静态 tiling 复用仓内 arch22 strided-batched 经验包络 `singleM=128, singleN=128, singleK=4096, baseM=128, baseN=128, baseK=64`（half 使 L1/L0 压力较 FP32 减半）；内部 128×128 输出 tile，`for task = GetBlockIdx(); task < rows*cols; task += GetBlockNum()`；K 方向 `for startK ∈ [0,k) step 4096`，`IterateAll(product[...], startK != 0)` 在 FP32 产品平面累加——归约在 kernel 内完成，不经 host 侧 K 切分，因此不触碰 #1077（该 issue 位于实数 **host 入口** 的 K 切分逻辑）；HF32 不适用于 FP16 操作数。
- **`hgemm_arch22_epilogue_kernel`（AIV）**：按「列 × 行块（256 元素）」分核；读 FP32 产品平面 → `alpha` 缩放 → 读旧 C（half→float、`beta` 缩放）→ `Add` → `Cast(RoundMode::CAST_RINT)` 收窄 FP16 → 只写逻辑 `row<m, col<n`。`alpha=1,beta=0` 也不能跳过 epilogue（仍需 FP16 舍入）——这是与 arch35 直写路径的本质差异；`scaleOnly` 模式承载 host 快速返回（`C=beta*C` 与写零）。
- **同步**：三个 kernel 独立 launch；UB↔GM 拷贝前后 `PipeBarrier<PIPE_ALL>`。

## 硬件与约束

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B3） | √（自验设备） |
| Atlas A3 系列（ascend910_93*） | √（与 A2 同一 arch22 代码路径，核数运行时查询；A3 尚无实机结果） |

### 算子约束限制

1. 仅列主序带填充布局，不使用 AutoContiguous/ViewCopy，不产生相关临时空间；
2. 不支持广播；C 原地覆写，调用方保证 C 与 A/B 存储不重叠；
3. 不要求确定性计算：多核切分与 K 归约顺序由实现按形状与片上容量选择；
4. `aclblasCgemm` 的 `OP_C` 与 `OP_T` 结果必须不同（标量符号区分，测试以非对称复数矩阵验证）；不采用 3M 分解，禁止混入 `cgemm3m` 文件；
5. `aclblasHgemm`：arch22 无 L0C→GM FP16 路径，禁止 cube 直写 FP16 GM；FP16 溢出/次正规按 IEEE RNE 收窄，若硬件 `Cast` 走饱和模式，超上限用例 golden 口径在自测报告说明；
6. 已知问题 ops-blas#1077：Cgemm 以 K 1024 分块规避（修复后可移除）；Sgemm 的 host-tile 路径为该问题影响区；Hgemm 不经过该路径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度（S） | atol=rtol=2^-13、max_abs ≤ 1e-2 或 32 ULP、matched_ratio ≥ 0.99；golden 由 cblas_sgemm 生成 | 任务书 §3.2 + 生态精度标准 |
| 精度（C） | 实部/虚部分别按 FLOAT32 档（2^-13），golden 由 cblas_cgemm 生成 | 任务书 §3.2 |
| 精度（H） | FP16 档：atol=rtol=2^-10、max_abs ≤ 1e-1 或 32 ULP、matched_ratio ≥ 0.99；golden 由测试工程 FP64 累加（cblas_dgemm）后收窄 FP16；Netlib 无 hgemm 参考，不使用 cblas_hgemm | 任务书 §3.2 |
| 性能 | `msprof op` kernel Task Duration ≤ GPU 基线（同接口同参数采集）1.25×，warmup 5 + 10 样本均值；GTest `[PERF] avg_us` 仅交叉验证 | 任务书 §3.3 |

## 测试设计

- 仓内测试：`test/gemm/arch22/`（Sgemm）、`test/gemm/cgemm/arch22/`（Cgemm）、`test/gemm/hgemm/arch22/`（Hgemm，新增，含 `hgemm_npu_wrapper.h` 与共享 runner）；
- 任务包用例：`test_cases/{sgemm,cgemm,hgemm}/*_test.csv`（401/400/399 条）可直接由同一框架加载；Hgemm 的 399 条 = 311 精度 + 22 负向 + 66 性能；
- 必测语义：trans 全组合（9 种）；`alpha=0` 且 A/B 为 NaN 不引用；`beta=0` 且 C 为 NaN 不读旧值；`k=0 && beta≠1` 只做 `C=beta*C`；ldc padding 哨兵；Cgemm 的 `OP_C` 与 `OP_T` 结果不同；Hgemm 的 FP16 上限/次正规/Inf/NaN 与长归约（512×512×1536、300×300×5000）；
- 自验入口：`bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`；任务包 `verify_accuracy.py` / `verify_performance.py`。

## 兼容性分析

`aclblasSgemm`/`aclblasCgemm` 为既有接口，不修改签名与行为；`aclblasHgemm` 为新增公开 API（声明紧随 `aclblasCgemm`，参数序列与 `cublasHgemm` 逐参数对应），与 950PR 版为同一接口的不同 arch 实现；`gemm_param.h` 对任务包 CSV 格式的扩展（alpha/beta 拼写、func 推断、999→0xFF）向后兼容。

## 待上板核实项

1. Hgemm 的 FP16 `MatmulImpl` 静态 tiling 包络在 910B3/CANN 9.1.0 的编译与精度表现；
2. `Cast(CAST_RINT)` 对超 FP16 上限值的饱和语义（inf vs ±65504），决定极值用例 golden 口径；
3. A3 实机精度/性能复跑；`msprof op` 基线采集与 0.8× 判据达标性。
