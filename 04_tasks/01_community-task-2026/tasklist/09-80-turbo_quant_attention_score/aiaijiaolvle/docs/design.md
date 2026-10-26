# TurboQuant Attention Score 算子设计文档

更新日期：2026-10-16。平台：Atlas 800T A2 / Ascend 910B；CANN 9.1.0；Ascend C / ACLNN。

本文对应 pairB 实现：kernel SHA256 `0ed0421318e4b54fcbda54686c96870756469336f738026cc6cf4f5f6b354b7f`，标准包 SHA256 `83077dd8400e1576081369900f8ed613ca3403d17dd01d3c7b0f0b3dff8d8fac`（3,603,539 字节）。五个正式形状、22 项接口及便携验收入口的技术自测已闭合；三档 Decode 每组两种计时均通过 1.2 任务门槛和 1.128 开发目标。设计评审、源码访问安排与正式平台验收分别按实际流程完成，本文不声明已获得正式验收回执。

# 需求背景（required）

## 需求来源

2026 年 9 月社区任务编号 80：`turbo_quant_attention_score`。依据更新任务书及其 `case.json`、`op.json`、`golden.py`、`generate_data.py`，官方附件为 [任务资源](https://www.hiascend.com/p/resource/202609/aee4cfec8ac34dfbbd5667290c471853.zip)。章节沿用社区设计模板的需求背景、需求分析、详细设计、可维可测分析。

## 背景介绍

TurboQuant Key cache 保存旋转空间主量化索引、QJL 残差符号、norm 和 gamma。算子在线解包、查码本，在旋转空间计算主分支内积和 QJL 修正，不把 Key 完整反量化回原空间。Query 旋转及 QJL 投影由调用方预先完成。算子只输出 FP32 attention score；softmax 与 Value 聚合由后续算子负责。范围为标准 MHA/GQA，不覆盖 MLA 和 quantized Value cache 解码。

# 需求分析（required）

## 需求描述

公共接口为 `aclnnTurboQuantAttentionScoreGetWorkspaceSize`、`aclnnTurboQuantAttentionScore`。输入连续 ND，block_size=128；调用方预分配输出。

| 参数 | dtype | 形状和约束 |
|---|---|---|
| query_rotated | BF16 | `[T,Hq,128]` |
| query_qjl | BF16 | `[T,Hq,J]` |
| key_cache_idx | UINT8 | `[P,128,Hkv,16*b]`，主码低位优先 |
| key_cache_qjl | UINT8 | `[P,128,Hkv,J/8]`，符号位低位优先 |
| key_cache_norm | BF16 / FP16 | `[P,128,Hkv]` |
| key_cache_gamma | BF16 / FP16 | `[P,128,Hkv]`，与 norm 独立选择 dtype |
| block_table | INT32 | `[B,C]`，逻辑页到物理页映射 |
| seq_lens | INT32 | `[B]`，请求实际有效长度 |
| attn_scores | FP32 | `[T,Hq,K]`，`K=max(seq_lens)` |
| mse_bits | INT 属性 | 2 / 3 / 4，默认 3 |

`B=1` 或 `B=T`；Hq 为 Hkv 的整数倍；D=128；J 为 8～128 内的 8 的倍数。页表和长度的公共 dtype 固定 INT32。metadata 支持四种独立 BF16/FP16 组合；内部注册的其他 dtype 不扩大此公共接口声明。空输出指有效长度 K=0，不表示支持 T=0。非连续输入不纳入直接 ACLNN 验证范围。

## 需求拆解

1. 主码按位宽解包并查官方 Golden 的 centroid；QJL 是实数 query_qjl 与 ±1 符号的内积。
2. 输出使用真实 `max(seq_lens)`，容量 `128*C` 只作上界；无效物理页、超过单请求长度的位置输出负 Inf。
3. 覆盖 MHA/GQA、2/3/4-bit、合法 J、四种 metadata、B1/BT、非对齐尾部及 K0。
4. 三档 Decode 与同设备 BF16 Paged Full-QK 的 score 路径比较，延迟比值≤1.2；默认 Key cache 字节压缩>3 倍。
5. 用相同原始 Q/K 计算完整 softmax 分布 KL，保留方向、mean、P95、max 和逐行结果。
6. 提供可复现源码/测试说明及 task_submission 七件材料。

任务书 3.3 的 1.2 门槛明确针对 Decode；Prefill 表中数值作为对照保留，不增设任务书未规定的 Prefill 硬门。开发追加 6% 余量，对应 1.128，是开发目标。任务书 3.4 内存要求写“不涉及”；缓存字节与工作区仍分别报告，不把理论缓存字节称为总 HBM 峰值。

# 详细设计（required）

## 算子分析

对有效逻辑 Key k，以页表取得物理页和页内位置；`kv_head=floor(query_head/(Hq/Hkv))`：

```text
primary = sum_d query_rotated[d] * centroid[unpack(idx)[d]]
residual = sum_j query_qjl[j] * (2*unpack(qjl)[j]-1)
score = norm*primary + gamma*sqrt(pi/2)/J*residual
```

码本、系数和位顺序按原 Golden。QJL 位 0/1 对应 −1/+1；query_qjl 保持实数，不把该内积改成两个符号向量的 popcount。算子输出不额外乘 `1/sqrt(D)`；质量检测仅在 softmax 前按约定缩放一次。数学公式、输入、比较阈值和 softmax 温度均未为达标而修改。

## 算子实现

### Host、形状与派发

Host 检查八输入 rank/dtype/维度关系、位宽、GQA 整除、页表容量、长度范围和 workspace 溢出。GetWorkspaceSize 每次读取本次 INT32 seq_lens 的真实值，检查输出 `[T,Hq,max(seq_lens)]`，同一长度指针更新后也重新读取；长度 value dependency 保留。DeviceHbm 的长度经正常 ACL D2H 读取，Host Tensor 才直接读取。物理页号在内核判断有效性。

通用路径按实际长度拆任务，尾部用精确字节搬运。K=0 仍派发一个 AIV no-op block，在 GM 读写前返回；零字节输出与物理 guard 分别检查。

| 路径 | 路由和组织 |
|---|---|
| 通用 AIV，key0～3 | 未命中专用条件的合法输入，覆盖其他 GQA/J/bits/batch 与实际长度 |
| GQA4 Decode，key4～7 | 容量输出、3-bit、J128、Hq=4Hkv、Hkv≤8；低两位区分 metadata dtype |
| 正式 Decode4K | T=B=1、C32、Hkv8；相邻两个 KV head、八个真实 query head 同批计算，832 条平铺记录，共用解码/查表准备 |
| 长 Decode | 单 query、Hkv8、C≥256，BlockLut 路径复用 query 相关主码表与 QJL 表 |
| MIX Prefill，key8～11 | 专用条件且 T>4、B1；按 query group、KV head、页组组织 Cube/AIV 协作 |

Decode4K 的 BF16/BF16 特化使用 48B 主码记录及独立 second Gather 缓冲，其余 metadata 使用对应模板；计算次序和 QJL 残差语义保持。数据搬入、向量读取、输出退休之间使用明确的事件同步，避免覆写仍在使用的缓冲。8 个 ELF 的 key0～7 共64个实际函数跨度与本次 Prefill 优化父版本逐字节相同；本稿性能全部来自当前完整包实际测试，不借用父包成绩。

### Prefill 的矩阵组织与 pairB

Prefill 在旋转空间构造 A/B 中间矩阵，主分支和 QJL 的 FP32 系数在 B 侧完成 norm/gamma 缩放，再拆为 BF16 high/low，使用 K512 的 BF16 Cube 乘法、FP32 结果和 mask。保留高低部分是为了控制数值误差；公共输出仍是 FP32。每个 worker 的 A/B/score 暂存独立。

短 Prefill 仅在 `!cacheQueryA && totalTasks<=usedCores` 且整组页完整有效时使用 pairB。AIV0 准备偶数页，AIV1 准备奇数页；二者写各自不相交、满足对齐要求的 B 工作区。ready flag8 在 B 数据发布后同步，end flag9 约束任务结束；AIV0 保留 A 准备、Matmul 与最终输出，AIV1 不执行这些阶段。无效页、部分尾组和不满足条件的输入继续走原 fallback。B 阶段 Vector reader 在 A 的 MTE2 复用共享缓冲前显式退休。

长 Prefill 保留原 query A 复用和独立 GM 槽。Query 分组最多512 token；4K 每组16页，长序列每组8页。当前实际 workspace：Decode 三档均0B，Prefill4K为125829120B，Prefill32K为121634816B。pairB 没有额外增加这两档 workspace，也没有把 Key 变回原空间。

### 构建与公共 ABI

在完整 ops-transformer 框架及真实依赖中，使用标准 `build.sh --pkg --experimental --soc=ascend910b --ops=turbo_quant_attention_score -j8 -O3` 构建。13 个实际编译源码由完整 SHA 清单和源码 archive 固定；标准包、安装 vendor、公共 API 库、正常调用程序、输入和输出在运行前后绑定 SHA。

公开 ABI 保持8个 Tensor、一个 mse_bits、INT32 页表/长度和实际 K 输出。正常调用 helper 使用 RUNPATH、无 RPATH，ldd-r 与 vendor 解析通过；不是依赖临时插桩或绕过正常加载的结果。

# 可维可测分析

## 精度与功能标准

正式五案的形状见下表。完整输出按原公开 `compare_default([0.001,0])` 比较，mask、有限值、输出字节数和 guards 分开检查。三档 Decode 有当前包完整 Golden 比对；两档 Prefill 的完整输出 SHA 与已全量通过 Golden 的对应输出一致，按内容身份继承，不把它称为本轮重新生成全部 Golden。

| case | T | KV长度 | Hq/Hkv/D | 完整FP32值数 | 结论 |
|---|---:|---:|---|---:|---|
| gqa_decode_kv4k | 1 | 4096 | 32/8/128 | 131072 | 完整通过 |
| gqa_decode_kv64k | 1 | 65536 | 32/8/128 | 2097152 | 完整通过 |
| gqa_decode_kv128k | 1 | 131072 | 32/8/128 | 4194304 | 完整通过 |
| gqa_prefill_kv4k | 512 | 4096 | 32/8/128 | 67108864 | 整输出SHA同值继承通过 |
| gqa_prefill_kv32k | 512 | 32768 | 32/8/128 | 536870912 | 整输出SHA同值继承通过 |

当前包另执行22个接口 caller进程：21项原接口与一项补充无效页组；5项为空输出、17项非空。完整5,105,120字节输出已保存，非空公开比对、空输出shape/bytes和全部前后128 guards通过。覆盖MHA/GQA、2/3/4-bit、J8/64/120/128、B1/BT、四种metadata、序列尾部、零长度与无效页。空输出的数值compare为N/A，不填true；caller数量不等于底层kernel数量。扩展用例使用未修改的独立CPU数学规范和公开比较，不冒称全部通过官方AscendOpTest框架执行。

另有11项当前定向功能，包含两个正式Prefill完整输出和9项T17全有效/混合无效/全无效边界，四种metadata均覆盖。边界expected为独立FP64固定公式后转FP32，来源与官方Golden执行分列。

## 性能标准与当前结果

单位μs。Decode每档3组S-R-R-S、每进程20次预热和100次采样，按原协议分别汇集每组200个Score和200个Reference样本后取中位数。device与host独立报告。Reference为同设备、同固定输入的BF16 Paged Full-QK Gather+QK；不把e9优化父版本称作Full-QK。

三档Decode共36个fresh进程，另有Prefill32K三个fresh进程，合计每种时钟3900个样本和78份完整初末输出收据。所有Decode组的device/host比值均≤1.128，最差为1.075454160563339，满足任务1.2及6%开发余量。

| case | 时钟 | 组 | Score | 同机Full-QK | 比值 |
|---|---|---:|---:|---:|---:|
| gqa_decode_kv4k | device | 1 | 45.440000 | 44.890000 | 1.012252172 |
| gqa_decode_kv4k | device | 2 | 47.200002 | 44.220001 | 1.067390349 |
| gqa_decode_kv4k | device | 3 | 46.900000 | 46.650000 | 1.005359068 |
| gqa_decode_kv4k | host | 1 | 73.320000 | 74.145000 | 0.988873154 |
| gqa_decode_kv4k | host | 2 | 68.675000 | 74.555000 | 0.921132050 |
| gqa_decode_kv4k | host | 3 | 73.920000 | 69.735000 | 1.060012906 |
| gqa_decode_kv64k | device | 1 | 1051.090002 | 996.829987 | 1.054432567 |
| gqa_decode_kv64k | device | 2 | 1052.600026 | 996.450007 | 1.056350061 |
| gqa_decode_kv64k | device | 3 | 1051.080048 | 986.209989 | 1.065777127 |
| gqa_decode_kv64k | host | 1 | 1083.880000 | 1025.655000 | 1.056768602 |
| gqa_decode_kv64k | host | 2 | 1104.970000 | 1027.445000 | 1.075454161 |
| gqa_decode_kv64k | host | 3 | 1082.910000 | 1017.025000 | 1.064782085 |
| gqa_decode_kv128k | device | 1 | 2033.900023 | 1982.429981 | 1.025963107 |
| gqa_decode_kv128k | device | 2 | 2034.009933 | 1985.289991 | 1.024540467 |
| gqa_decode_kv128k | device | 3 | 2033.740044 | 1986.319959 | 1.023873336 |
| gqa_decode_kv128k | host | 1 | 2061.680000 | 2010.205000 | 1.025606841 |
| gqa_decode_kv128k | host | 2 | 2072.785000 | 2015.915000 | 1.028210515 |
| gqa_decode_kv128k | host | 3 | 2064.600000 | 2018.330000 | 1.022924893 |

计时不包含首次编译、输入准备/搬运、workspace申请和GetWorkspaceSize中的长度读取。host窗口是原Run调用/调度及end-record的窗口，不能称为含setup的完整API端到端延迟；device event也不冒称纯kernel profiler时间。普通性能与profile诊断分开。

### Prefill

Prefill4K使用独立12个fresh进程的三组E-C-C-E/C-E-E-C/E-C-C-E优化对照（E=e9父版本，C=pairB），每进程20warm/100，1200样本/时钟。每组Score使用该组两个进程中位数的中位数；不与Decode的pool200协议混淆。当前device三组均低于额外开发目标707.697048μs，host三组均比同组e9改善。

| case | 组 | Score device | Score host | 任务固定表 | device/表 | host/表 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill4K | 1 | 674.524993 | 742.132500 | 627.391 | 1.075126983 | 1.182886748 |
| Prefill4K | 2 | 677.285001 | 770.850000 | 627.391 | 1.079526166 | 1.228659640 |
| Prefill4K | 3 | 675.939992 | 736.515000 | 627.391 | 1.077382353 | 1.173933002 |
| Prefill32K | 1 | 5400.779963 | 5429.785000 | 5910.060 | 0.913828280 | 0.918736020 |
| Prefill32K | 2 | 5410.360098 | 5438.860000 | 5910.060 | 0.915449268 | 0.920271537 |
| Prefill32K | 3 | 5401.370049 | 5431.210000 | 5910.060 | 0.913928124 | 0.918977134 |

固定表不是本次同机Prefill Reference。特别保留Prefill4K host第二组/固定表=1.228659639682431；不删该组，也不把任务书没有规定的Prefill1.2门槛加为验收门。Prefill32K device三组中位数5401.370049μs，host5431.210μs。

## Softmax KL自测

任务书3.2原文为“经softmax后的attention weight分布与全精度版本的KL散度<0.01”。本自测按每case全部token×head行的双向mean报告，五案均通过。该聚合方式在此明确，不声称任务书原文显式规定mean；不追加逐行max硬门。

使用相同固定原始BF16 Q/K，logits仅除一次sqrt(128)，用FP64计算完整softmax和KL，不加温度、不裁剪query行，也不以独立高斯query替换。完整输出与pairedQ/K的SHA同值关系用于已有统计的内容继承；原统计来源与新执行分列。

| case | 行数 | KL(full||score) mean | KL(score||full) mean | 前向P95 | 前向max | 前向行≥.01 |
|---|---:|---:|---:|---:|---:|---:|
| gqa_decode_kv4k | 32 | 0.006353507932614 | 0.006347971250369 | 0.007892302644182 | 0.008723538369809 | 0 |
| gqa_decode_kv64k | 32 | 0.006308033224858 | 0.006302807299514 | 0.007482048159516 | 0.007758545058844 | 0 |
| gqa_decode_kv128k | 32 | 0.006172376950113 | 0.006167801903008 | 0.007349209636541 | 0.007870254224190 | 0 |
| gqa_prefill_kv4k | 16384 | 0.006193061296654 | 0.006188099651896 | 0.007619571031445 | 0.010365379636402 | 2 |
| gqa_prefill_kv32k | 16384 | 0.006198337199210 | 0.006193363194189 | 0.007544370534875 | 0.009580046513217 | 0 |

Prefill4K的2/16384尾行及原Golden同源结果保留为诊断。最差前向KL约0.01036538；这些尾点不隐藏，也不作为额外逐行硬门阻塞case级自测。Score任务没有给相对误差的具体数值门槛，也未要求PPL，本文不引用其他算子的质量规则。

## 便携测试入口与本次新执行范围

参数化`score_validate.py`接受实际bundle、输入根、case、vendor和输出目录，不依赖历史机器名/boot/CONFIG。`preflight`、`compare-kl`仅Python/NumPy，`golden`、`all`使用配套CANN/PyTorch/torch_npu。驱动固定SHA `3c769de8e86bae23b5fec816e3aaca44fc406c6ce7e7941abcf9074f106cf7c8`，官方Golden和数学/比较/KL函数未改。

最终4K入口controller353881/0、driver/Golden353904/0：新执行原Golden一次；全131072值compare[.001,0]通过，maxabs=1.71661376953125e-5；Score与Golden各32行双向KL完整通过。旧native330924/0产生的完整Score复用，累计native1、Golden1；缺torch的原driver330503/1失败及后续环境修正分别保留，未伪装同一成功进程。此处新Golden/KL只覆盖4K；其他四案仍为前述既有全量/整SHA同值证据，不称本轮新执行五案。

实际独立环境为Python3.12.13、torch2.7.1+cpu、torch_npu2.7.1.post10、NumPy1.26.4、PyYAML6.0.3。13项官方wheel有固定URL/字节/SHA和requirements.lock；私有pip check使用-I，CPU import加载CANN。换机创建新venv、离线按lock安装并重新绑定实际环境；不依赖原机器venv可访问。该smoke不产生新性能资格。最后设备健康且空闲。

## 缓存、报告与可复现性

默认D=J128、3-bit：48B主码+16B QJL+2B norm+2B gamma=68B，BF16 Key256B，压缩3.764705882倍。它是Key tensor payload字节比。Decode实际workspace0，两档Prefillworkspace按前文报告。总设备/模型HBM峰值未测，不填0，不增设任务书3.4之外的峰值硬门。

七件材料包括自验证步骤、精度/性能/内存三个工作簿及三份日志；参数、完整数值、所有组和真实来源可追溯。工作簿内嵌的是实际结果表渲染图，不冒充终端截图；TBE截图没有则明确不适用。历史失败、父版本结果和当前包分列。完整大输出的外部内容索引保留字节/SHA，未声称所有GB输出都内嵌在小包内。

当前包22项完整raw已保存；便携入口123个原始归档成员均核完整SHA，本地审计新增bytecode另列，未冒称原归档成员。源码13件、标准包、计时器/Reference、固定输入、完整输出和pairedQ/K前后绑定；保存后回读报告与独立核验均完成。

### 主要证据身份

下表是随验收材料提供的证据文件及SHA，非虚构在线下载链接。技术自测完成与社区设计评审/平台验收回执分别记录。

| 证据 | SHA256 |
|---|---|
| 当前包11项功能 | `f844075d5689fa1f86dc2e8ec51969c80220f050ac039602289455b47cab9292` |
| 当前包22项接口 | `882b139313e62b2691a89a9ec2137576bfdb73f71fa4b7e016ed75c520f95ba9` |
| PF4K独立ordinary12 | `22fc6f1d1c0c0fcb3ce511598a67b0f48531ecdb019ac250a9d3b4f5522317b5` |
| 三Decode及PF32K39进程闭合 | `a1bbced408fb6466a229b51f88640a8ec40f4c8258be674418c50aced253960c` |
| 最终4K便携入口原始闭合 | `91d4bc9472cbf4e08701e58b373e9a43e19c6bf2690023db40157e92e10c2aca` |
| 便携原件独审 | `240d1e3ea143e919d82d4b104acb4241a3e3b3512620dc186e0198111bf41da4` |
| 七件材料索引 | `9c0143bce79a482b18ab132e1f48afa9ae49c37b62399aee2c341d832da67fac` |

## 兼容性分析

新增ACLNN算子，公共八输入/一个属性与任务定义对齐；通用fallback保留接口扩展范围，专用优化不缩减合法bits/J/metadata/batch支持。适配A2/910B和CANN9.1.0；不声明A3双平台验收，不声明非连续输入或MLA支持。当前所有要求的技术自测已完成；后续完成设计PR评审、源码可访问分支与开发者邀请、正式验收提交和回执。验收后的上游源码PR另按流程处理。
