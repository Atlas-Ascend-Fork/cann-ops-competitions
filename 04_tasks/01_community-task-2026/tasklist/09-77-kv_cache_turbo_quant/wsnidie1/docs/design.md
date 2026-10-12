# KvCacheTurboQuant 算子设计文档

更新：2026-10-10。TeamName：`wsnidie1`。本文记录当前14实现/构建源、标准包 `a2e3195ad9b08ce295059a516313bd812a7b44078c3d06b81db7d3be5875312f` 的设计与正式实测，并补充个人fork76d基线新包 `3c4c2d55a32690a005fa8b894be9de221080cf04bd09052a8ed15af2c15aaee4` 的标准构建和五案原生复现范围。源码与自测材料准备于[个人代码仓](https://gitcode.com/wsnidie1/ops-transformer)，设计文档目标为[个人设计仓](https://gitcode.com/wsnidie1/cann-ops-competitions)中的 `04_tasks/01_community-task-2026/tasklist/09-77-kv_cache_turbo_quant/wsnidie1/docs/design.md`。

当前代码材料尚未推送，本文列出准确仓内路径，不声称这些文件已有在线commit/分支回执。个人仓76d0d013155c04a8c027bcf4f87e54b3e95b2dc0基线已在347007/910B3完成新标准构建和原五案原生调用，详见下文独立复现记录。性能表仍只绑定原a2e包；新3c4c包没有重新执行正式100次性能管线。

# 需求背景

## 需求来源

依据2026-09-29更新的KV Cache在线编码任务书，以及任务方关于正式五案、完整重构NMSE、相关query、边界、字节比较和计时方法的统一回复。本文按官方 `04_tasks/01_community-task-2026/resources/design_template.md` 的需求背景、需求分析、详细设计、可维可测分析四栏目填写。

## 背景介绍

标准 MHA/GQA Prefill 或 Decode 会生成新 Key/Value。在线编码把每个 128 维 BF16 向量从 256 bytes 缩为量化主码、QJL 残差符号和两个 16-bit 尺度。首版正式范围是 A2/Ascend 910B；`D=J=128`、3-bit 主码和 1-bit QJL 时为 `48+16+2+2=68` bytes/向量，即 4.25 bit/channel。2/4-bit 还作为正式五案的精度和性能档位。本算子执行新 KV 的编码，不负责 Attention Score、缓存读取、Value 聚合或完整模型接入。

# 需求分析

## 需求描述与需求拆解

输入是连续 ND `x∈BF16[T,H,128]`、固定 `H_matrix∈FP32/BF16[128,128]`、`S_matrix∈FP32/BF16[J,128]`；`4≤H≤32`，`J` 是 `8…128` 的 8 的倍数。默认 `mse_bits=3`，支持 2/3/4。正式五案的矩阵为 FP32、metadata 为 BF16、`H=8,J=128`、`T=1/64/2048`。现读官方下载 `op.json` 仅列FP32矩阵/BF16 metadata；任务方Q7回复另外要求的BF16矩阵和FP16 metadata作为功能扩展，不计入五案性能门槛。正式输入域是有限连续值；NaN/Inf 和非连续直接 ACLNN 输入不在本次验收范围。


### 输入、属性与输出

| 参数 | 类型 | dtype / shape | 约束 |
|---|---|---|---|
| kvVectors | 输入Tensor | BF16 `[T,H,128]` | 连续ND、有限值 |
| rotationMatrix | 输入Tensor | FP32或BF16 `[128,128]` | 固定旋转矩阵 |
| qjlMatrix | 输入Tensor | FP32或BF16 `[J,128]` | `J=qjlMatrix.shape[0]`，8…128的8的倍数 |
| mse_bits（C API mseBits） | int64属性 | 2、3或4，默认3 | 主码位宽 |
| quantIdx | 调用者预分配输出 | UINT8 `[T,H,128*mseBits/8]` | 主码低位优先packing |
| quantQjl | 调用者预分配输出 | UINT8 `[T,H,J/8]` | QJL符号低位优先packing |
| quantNorm / quantGamma | 调用者预分配输出 | BF16或FP16 `[T,H]` | 两者dtype必须配对一致 |

接口含三Tensor输入和一个`int64_t mseBits`属性；J由qjlMatrix形状导出。用于内积质量评估的ip_query是独立验证资产，不是算子第四输入。

对每个原始向量 `x`，内部用 FP32 中间值计算：

```text
n = ||x||₂
u = x / max(n, zero-safe divisor)
y = u @ H_matrix.T
idx[j] = sum_t [y[j] > threshold_t]               # 严格 >
primary[j] = centroid[idx[j]]
r = y - primary
gamma = n * ||r||₂
r_unit = r / max(||r||₂, zero-safe divisor)
qjl[k] = [ (r_unit @ S_matrix.T)[k] >= 0 ]         # >=0 为 1
```

`zero-safe divisor` 的具体 clamp/where 与随任务发布的 `golden.py` 对齐：全零向量 `n=0, gamma=0`，主码索引分别为 2/3/4-bit 的 `1/3/7`；零投影 QJL 位为 1。两个 UINT8 payload 都低位优先打包。输出 `quant_idx[T,H,128*b/8]`、`quant_qjl[T,H,J/8]`、`quant_norm[T,H]`、`quant_gamma[T,H]`；最后两个输出可同为 BF16 或同为 FP16。

质量重构按任务方澄清使用**主码与 QJL 双分支**：

```text
sign = 2 * unpack(qjl) - 1
r_hat = sqrt(pi/2) / J * (sign @ S_matrix)
x_hat = (quant_norm * centroid[idx] + quant_gamma * r_hat) @ H_matrix
aggregate_NMSE = sum_i ||x_i - x_hat_i||² / sum_i ||x_i||²
vector_ratio_i = ||x_i - x_hat_i||² / max(||x_i||², 1e-12)
q = BF16(x + 0.5 * N(0,1))
dot_error_i = |<q_i,x_hat_i> - <q_i,x_i>| / max(|<q_i,x_i>|,1e-6)
```

五案 aggregate NMSE 都须 `<0.05`，内积误差在 token×head 上的 P95 都须 `<0.1`；3/4-bit 逐向量 max `<0.05`；2-bit P95 `<0.06` 且 max `<0.10`。旧版独立高斯 query 裸相对误差不属于该门槛。

# 详细设计

## 算子分析与实现方案

ACLNN L2 facade 接收三Tensor输入、一个 `int64_t mseBits` 属性和四个调用者预分配输出；J由qjlMatrix形状导出。 `GetWorkspaceSize` 使用执行器内部 24 MiB scratch Tensor 并返回真实 workspace 大小，`SetRepeatable()` 支持预热后同一执行器多次调用。Host tiling 校验 rank、矩阵形状、位宽、head 数、QJL 维数和 metadata 配对 dtype。`T*H<16` 或 `J<128` 走 AIV 路径；其余使用 32 行 tile 的 MIX_AIC_1_2，正式 T64/T2048 属于后者。tiling key 位 0 选路径，位 1/2 选 H/S BF16，位 3 选 metadata FP16；QJL、位宽和总向量数另存 tiling data。workspace 中 A/Y FP32 scratch、BF16 矩阵转 FP32 的私有副本按 worker 隔离，避免写冲突。

MIX 路径中，每个 AIV worker 载入 32×128 BF16，转 FP32，求原始 norm，写归一化 A scratch；AIC 以 FP32 Cube 做 `u@H.T` 并写 Y scratch，AIV 取回后按严格阈值树选主码、重构 centroid 与残差。残差 norm、gamma 与残差归一化完成后，AIV 写第二次 A scratch；AIC 做 `r_unit@S.T`，AIV 将 `>=0` 的比较结果直接作为 QJL 位 payload 写回。metadata 转 BF16/FP16。`V_MTE3`、`MTE3_S`、`MTE2_V` flag 确保 scratch 生产、Cube 读取和 Y 回取的顺序。AIV 小路径保持逐向量计算，用于 T1 和 `J<128`。

v8i 在 `mse_bits=3 && totalVectors≥2048` 时，将 bitplane 重建为 code，按 8 个 code 折叠为 24-bit chunk。所有整数小于 2²⁴，FP32 折叠/Cast 精确；再 Gather successor、移位、Or 拼成连续字节。实际 CANN 9.1 `dav_c220 OrImpl(int32,count)` 将输入 reinterpret 为 int16，却未扩大 count；因此显式使用 uint16 view 和 512 个 halfword 完整覆盖 256 个 word，并在最后 masked ShiftLeft 到 Or 间加入 V barrier，写出前使用 V→MTE3。短 b3 与所有 b2/b4 MIX 案仍走 64-bit scalar SWAR。该修改保持投影、阈值、残差和尺度的运算顺序。

Add-repeat v2 改动局限于 MIX 路径中两次求平方范数的 paired Add：原来每行执行 `product[i]+=product[i+64]`，现在用 `BinaryRepeatParams{1,1,1,16,16,16}` 一次覆盖 32 行，各行内部 64 对的加法顺序不变，再交给 `WholeReduceSum`。这是保留原 TorchNPU Golden 归约树的关键，避免 BF16 norm/gamma 中点舍入变动。

## 支持硬件

| 硬件 | 范围 |
|---|---|
| Atlas 800T A2 / Ascend 910B | 任务范围；a2e包在733540/910B3构建并完成48案，972692/910B4完成正式五档；当前14源独立冷恢复在733540/910B3完成；个人fork76d新3c4c包在347007/910B3完成标准构建及原五案原生复现 |

实测软件环境为CANN9.1.0、aarch64 Linux和Ascend C ACLNN。标准构建Python与TorchNPU运行Python分别管理；更换框架或SDK后需绑定新包的构建及功能回执。

## 算子约束限制

输入为连续ND、有限数。正式五案D=J=128、heads=8、tokens=1/64/2048、矩阵FP32、metadata BF16；2/3/4-bit分别采用任务方分级门槛。接口扩展矩阵BF16、metadata FP16须功能回归。四个输出由调用者预分配且dtype/shape与接口约束匹配。正式资产的query seed和SHA必须与发布件对应，开发生成件不能替代其来源身份。

# 可维可测分析

## 精度标准/性能标准

| 要求 | 判据 | 来源 |
|---|---|---|
| 四输出 | UINT8按payload byte，metadata按Golden和正式case.json；BF16[0,0]要求一致 | 任务书及官方回复 |
| 完整重构质量 | 含主码和QJL；全部aggregate NMSE<0.05；3/4-bit逐向量max<0.05；2-bit P95<0.06、max<0.10 | 官方回复 |
| 内积质量 | q=BF16(x+0.5noise)，分母max(abs(q·x),1e-6)，token×head全部向量P95<0.1 | 官方回复 |
| 五案性能 | 同设备同输入等价PyTorch eager逐案≥10×；20预热、100设备计时、3轮中位 | 官方回复 |
| 额外性能目标 | 允许耗时×0.94，等价加速≥10.638297872× | 用户当前6%目标 |
| Prefill生成率 | Encode不能成为KV生成瓶颈；保持独立同流阶段诊断 | 任务书3.3 |

计时不计输入准备、数据搬运、首次编译和workspace申请。实际workspace数值保留在七件内存日志；任务书3.4不涉及内存硬门，不能把输出编码压缩率当作整体峰值比。

## 当前14源/a2e实际构建、功能与性能

当前本地实现已选定实际测试的14个实现/构建源，集合SHA `930f56627f0aa12ffba3c4c8dcddd5071a36d57a76c55c87278e3110dda0b99b`。kernel仍为 `4312dc2e148eca689320e58c373c0f6efba2038eeac66774b536a418cd0a7219`；标准安装包为 `a2e3195ad9b08ce295059a516313bd812a7b44078c3d06b81db7d3be5875312f`（4,686,868B）。本次只更新Host tiling，按设备实际AIC/AIV核数分配，去掉未被kernel使用的AIV投影GetTiling；保留456B tiling布局，SDK数据缓冲在tiling对象生命周期内有效。未引用的 `op_kernel/kv_cache_turbo_quant_batch_aic.h` 已从当前14源移除。

733540/910B3标准build PID81896/RC0完成。正式5案、16个dtype扩展和27个边界共48案/192完整输出，均与原固定Golden逐字节一致，有限值与原public比较通过；workspace均为25,166,336B。733540/910B3实测五案LaunchKernelV2 blockDim分别为8/16/20/20/20。8 ELF、8 metadata和API共17件与原414完全相同；Host库的新SHA为 `a206fd97d4ed4792f84aff79dd15893694012a834921a9d18c877367127d6b0c`。这些是本次真实NPU结果，不将旧27案证据改标。

972692/910B4对这个a2e包完成自身正式20warm/100设备调用×3轮同机eager计时，controller111226/RC0，eager112168/RC0，ACLNN driver112719/RC0；全部五案达到10倍及用户6%目标，最低加速150.316993041倍。原计时器每轮给100次聚合均值，下面取3轮中位；未产逐次100采样CSV。

| 五案 | ACLNN中位µs | 同机eager中位µs | 加速 |
|---|---:|---:|---:|
| gqa_decode_b1 | 11.112 | 4262.274475 | 383.574016837× |
| gqa_decode_b64 | 28.030 | 4213.385315 | 150.316993041× |
| gqa_prefill_t2048 | 162.279 | 45082.109375 | 277.806181792× |
| gqa_mse2_t2048 | 212.026 | 44885.502930 | 211.698107448× |
| gqa_mse4_t2048 | 291.072 | 47480.327148 | 163.122276098× |

### 完整重构与内积质量

以下为原固定输入上的完整主码+QJL质量分布，统计公式与原query资产保持不变。当前a2e五案20个完整输出与归档统计所用输出逐SHA相同，故按字节身份绑定复用；本次设计文档未重新生成数据或重算数值。

| 固定用例 | aggregate NMSE | 向量 P95 / max | 内积误差 P95 |
|---|---:|---:|---:|
| gqa_decode_b1 | 0.014143 | 0.017501 / 0.017826 | 0.032324 |
| gqa_decode_b64 | 0.012241 | 0.017185 / 0.023072 | 0.033372 |
| gqa_prefill_t2048 | 0.012274 | 0.016911 / 0.035049 | 0.032896 |
| gqa_mse2_t2048 | 0.041852 | 0.053485 / 0.076124 | 0.075565 |
| gqa_mse4_t2048 | 0.003365 | 0.004746 / 0.013737 | 0.015293 |

全部五案满足前述aggregate NMSE、分级逐向量和相关query P95门槛。完整精度分布、输入/H/S/query/output SHA与原seed记录保存在代码仓准备材料 `evidence/quality_statistics_reused/<case>/complete_quality.json`；展示值只作六位小数排版，完整值以原JSON为准。

48功能案及正式计时的输入、Golden、H/S、相关query和完整输出均已核对。a2e正式性能原流程保留前后40个完整eager输出及15round的60个完整native输出，均公开比较、finite和原始字节一致。参考标签是功能等价的独立PyTorch eager；任务方未提供指定的eager计时脚本，TBE性能为不适用。

### 冷恢复与正式资产范围

当前14源另在733540/910B3完成独立冷恢复：标准build PID125863/RC0，包 `c1535685cde11940823ca9aae0a81957a46ece4b26bd56c9b098613e37a00642`，native controller140057/RC0，原五案20完整输出通过。这项证据只绑定冷恢复与原生调用；本设计上表性能仍属于a2e。

### 个人fork76d基线标准构建与五案原生复现

个人代码fork的真实HEAD为 `76d0d013155c04a8c027bcf4f87e54b3e95b2dc0`。该基线的Linux LF归档包含13,888件原框架文件，逐件字节与模式身份已核；在新隔离工作区仅应用当前26件源码/文档，14件实现/构建源仍为前述930f566集合，标准构建后原框架tracked文件没有变化。CANN9.1.0、实际SDK及第三方依赖在构建前后SHA一致，build Python与runtime Python分别使用其真实环境。

347007/910B3的实际controller275622/RC0、标准build275772/RC0、便携native5 controller282455/RC0完成。新标准包为 `3c4c2d55a32690a005fa8b894be9de221080cf04bd09052a8ed15af2c15aaee4`（4,662,527B）。原五案按顺序gqa_decode_b1、gqa_decode_b64、gqa_prefill_t2048、gqa_mse2_t2048、gqa_mse4_t2048的实际调用PID分别为283474、283790、284107、284423、284739，均RC0；五案真实GetWorkspaceSize均为25,166,336B。20个完整输出逐字节等于原固定Golden及a2e对应输出，原public比较和有限值检查通过，未重新生成input/H/S/query或Golden。

新包的实际安装8 ELF、8 metadata、API头/库和Host库逐字节等于已测a2e对应原件，Host仍为a206fd97…；源码、SDK、包、输入和Golden首尾身份均闭合。这是新个人仓框架上的标准构建与五案功能证明；没有对3c4c重新运行20warm/100×3性能，亦没有以短native调用耗时替代上表a2e正式性能。此次原始构建回执、日志和完整输出另作为独立验证附件保留，当前代码payload不包含这批新回收原件，在线附件路径以实际发布回执为准。


最新官方ZIP中的case、generator、Golden、op和taskbook五件来源已核，且与冻结requirements逐SHA一致。当前fixed input/H/S/query是原已测开发固定资产，具有完整SHA、seed及生成器来源记录；尚未取得其等同于正式发布固定bin/矩阵的发布身份回执。已有开发质量/性能通过须与正式固定资产验收分别记录。

自测七件含逐案完整参数、精度/性能和真实日志结果图；图片注明来源，三报告各有5条原生相对PNG链接与5张原字节嵌图。步骤明确三Tensor输入与mseBits属性、J由shape导出。指定Prefill生成率、正式固定资产身份、实际设计评审及验收提交继续按任务流程确认。

## 兼容性分析

新增算子；不替换既有算子接口。原Float32投影、归约树、严格边界、LSB packing及BF16/FP16 metadata保持Golden语义。MIX和小AIV路径均由Host校验后选择；功能扩展与正式性能范围分别记录。工作区按worker隔离，调用者须遵循执行器生命周期及同流串行复用要求。

## 复现和交付入口

入口为[个人代码仓首页](https://gitcode.com/wsnidie1/ops-transformer)。下列均是从仓库根解析的路径；当前文件已在本地准备，代码分支推送前不提供虚构的在线文件链接或commit SHA。

| 内容 | 仓内路径 |
|---|---|
| 当前26件源码/文档 | `experimental/attention/kv_cache_turbo_quant/` |
| 算子/测试说明 | `experimental/attention/kv_cache_turbo_quant/README.md`、`experimental/attention/kv_cache_turbo_quant/tests/README.md` |
| 标准构建入口 | `reproduction/standard_build_current.sh` |
| 固定五案可携入口 | `reproduction/replay_current_package.py`，默认只执行CPU plan |
| 固定input/H/S/query/Golden及原输出 | `evidence/formal5_performance/five_cases/<case>/` |
| 原timer/eager/Golden与公开比较器 | `evidence/formal5_performance/source/`、`evidence/formal5_performance/public_compare/` |
| 构建与48功能原件 | `evidence/build_native48/control/BUILD_RESULT.json`、`evidence/build_native48/control/RUNTIME_RESULT.json` |
| a2e五案计时原件 | `evidence/formal5_performance/control/PERFORMANCE_ACTUAL_RESULT.json`、`evidence/formal5_performance/control/ACTUAL_ROUND_MEANS_DERIVED.csv` |
| c153冷恢复原件 | `evidence/fresh_source14_cold_native5/control/cold_standard_build.receipt.json`、`evidence/fresh_source14_cold_native5/control/RUNTIME_RESULT.json` |
| 输入/源码/包pin | `reproduction/ASSET_MANIFEST.json`、`evidence/SOURCE_SELECTED26_MANIFEST.json`、`evidence/CURRENT_SOURCE14_MANIFEST.json` |
| 七件自测材料 | 根 `task_submission/`：步骤、精度/性能/内存报告及三日志 |
| 原日志结果图 | 根 `qa/`；三报告中的`../qa/`相对链接从task_submission正确解析 |
| 官方来源记录 | `official_asset_provenance/current_public_generator_oct10/RECEIPT.json`、`requirements/` |

标准构建要求真实框架、已安装CANN、cann_3rd依赖和满足框架requirements的build Python；入口核26源SHA，SDK初始化后返回框架根，调用正常build CLI。固定五案native/perf要求新独立work和相应numpy/ml_dtypes或TorchNPU运行Python，读取已有input/Golden；新包必须给出实际package SHA。显式perf保留原initial完整20输出→20warm/100×3 eager→20warm/100×3 ACLNN→final完整20输出→6%评估顺序，计时源、算术和门槛不改，首尾记录实际host/API/8ELF/8metadata身份。

小仓库材料未携带完整离线框架/SDK及全部16dtype/27boundary输入；原48/192结果保留为真实历史证据，可携入口只复现原formal5。完整离线冷恢复归档作为另行提供的冻结附件，不能仅凭所附restore参考代码宣称完成冷恢复。个人仓76d基线的新标准build/native5已按上文真实回执完成；新包功能证据与a2e正式性能分别记录。代码分支仍未push，后续在线提交身份以实际平台回执为准。

本设计提交到[个人设计仓](https://gitcode.com/wsnidie1/cann-ops-competitions)的 `04_tasks/01_community-task-2026/tasklist/09-77-kv_cache_turbo_quant/wsnidie1/docs/design.md`。TeamName为用户确认的`wsnidie1`；设计PR、代码分支提交和正式验收均以实际平台回执为准。
