# SquareSumV1 v1（正式版）

已交付 `submissions/SquareSumV1-v1/` 的同源.run与现成ZIP。009完成fresh build、5+3正确性
及完整测量：1449.6095 / 1447.6395 / 1448.1185 µs，均低于1450；最终STOPPED。
源码和二进制未为发布重编译或改包。以后发布只新建目录、复制已测.run，不要求A/B或额外验收。

009保留自有case2/3算法，为mode7动态申请system+user workspace，并通过运行时
`GetUserWorkspace(workspace)`取得用户区；profiling不再出现008的DDR越界。同时汇总器
拒绝空cases和中断收据。7个实际CPU用例通过。以下为历史研发记录，不是发布清单。

008已fresh build并通过非计时5+3正确性，但profiler下case2报DDR越界，没有完整五点总和；
整批失败归档，不发布。case3约69us仅为失败批次单点。006 v1-beta1仍是发布版，最好
1502.1405us。全部原始错误、CSV、源码/ZIP证据见 `results/square_sum_v1/`。

以下保留008设计记录；其中“入口已是user pointer/不使用GetUserWorkspace”的旧解释
已被009替换，不再作为当前ABI要求。

## 008补充：BF16 full-row suffix / FP32完整行微批

- 仅BF16现mode0、R>64且完整预算满足时分流到 `square_sum_fullrow.h`；否则仍用006
  suffix fallback。不改mode7、FP32、half或全局L2策略；只有新full-row路径input单独bypass。
- `P=Align(R,16)`、固定FP32 microRows=16，从B32向下至B16选最大容量可行值，非性能扫描。
  64位计算pitch、stride和物理预算，`P/8<=255`，过门禁后才窄化；R0、极大R和超预算拒绝。
  两BF16整行输入槽 `4*B*P`，FP32 calc `4*16*P`，float sums与双BF16输出合计
  `8*Align(B,16)` 字节，额外保留512B。官方R1025为 **P1040/B31/195776B**，含保留
  **196288B<=196608B**；B32物理199936B拒绝。R1153为P1168/B25/191808B。
- 每批一个2D DataCopyPad完整读取R列、每行右补零到P，双槽预取下一完整行批。直接从
  input[localRow*P] Cast到FP32 calc，subRows=min(16,actualRows-localRow)，随后原位Mul，
  对完整R调用从我们006原样提取的 `FoldAlignedSuffixRows`；不是GM列条带，不新增BF16
  staging、不跨stripe累加。fold仍是half=ceil(active/128)*64、64列对齐Add、最后WRS。
- sums清零覆盖Align(actualRows,16)，微批WRS直写sums[localRow]，最后RINT Cast完整padding。
  全部微批读完才释放input；Cast后PIPE_V保护sums复用。保留双输出pending退休，始终按真实
  count字节DataCopyPad写出（B非16倍数时，即使尾count=16，GM起点也可能不对齐）。输出
  ownership仍用旧CalcOutputRange，尾行/尾列均参与，没有额外kernel或计时改动。

补充恰两个实际CPU输入，无隐藏shape/subtest、未重跑旧套件：

```bash
PYTHONPATH=tests python3 -m unittest -v \
  test_square_sum_fullrow.SquareSumFullRowTests.test_square_sum_fullrow_17x1153_model \
  test_square_sum_fullrow.SquareSumFullRowTests.test_square_sum_fullrow_official_case3_host_coverage
```

首次setUpClass的probe常量与host同名冲突，编译失败 **Ran 0 tests**（0实际输入）；常量移入
probe类后同命令 **Ran 2 tests in 2.176s, OK**。本补充2/2用完，008此前4+本次2=**6/10**。
真实host/policy，以及从源码提取的buffer分配、DMA包装、调度、range与fold用g++ C++17、
`-Wall -Wextra -Werror`、undefined/bounds sanitizer编译；两次host各只请求一个输入。

- `[17,1153]` BF16 axes1 keep=true：16/1 output ownership、完整R1153数值和逐元素lineage、
  输入/微批/输出无遗漏、UB源32B对齐、尾1行和最终Cast padding有界且零初始化。
- 官方case3只做host/规划/完整coverage，不建大张量：mode0/40核/P1040/B31/micro16，
  实际InitBuffer字节与policy相等、B32预算拒绝，双槽生命周期及全部行/微批覆盖。

CPU trace不模拟真实PIPE/DMA/CANN重载、UB分配和设备浮点舍入，也没有70us或1450总和承诺。
本次最终mode7修订更新workspace测试的三extra文本期望；历史wide测试改为提取已有
`FoldAlignedSuffixRows`方法接口，但未运行wide测试，不能视为其回归结果。
`benchmarks/square_sum_workspace_checks.json`前两项原样，追加extra-3 `[17,1153]` BF16 axes1
keep=true。父审后008整体唯一fresh build，同一.run先预ZIP、再私有安装；官方五点+3extra
正确性通过后3轮完整五点性能。此补充未做NPU、准备包、提交或暂存。

## 008唯一新模式及workspace合同

- 仅在原BF16 inner分支且真实连续 `[O,R,I]`、`64<=I<=256`、`R>512`、`O*R>=3200`、
  完整UB合格、workspace不超过1MiB时选 **mode7**。`taskNum=O*R`、固定40核，
  `P=Align(I,16)`、`N=O*R`、`L=ceil(N/C)`、`slots=min(O,ceil(L/R)+1)`；
  user workspace为 `[C,slots,P]` float，共 `uint64(C)*slots*P*4` 字节。共享policy用64位
  推导slots，host全宽判界后device才窄化。失败保留原mode1及32B；其他mode仍32B，
  旧balanced/tight/helper资格不改。
- 保留007的 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)`，所有模式核数不超过40。
  CANN8.5工程入口workspace已是user pointer，直接转为float GM，不调用GetUserWorkspace/
  SetSysWorkspace，不凭空加system区或软件sync flags。这里只使用基础DMA/Gather/硬件SyncAll。
  依据：[工程workspace](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0092.html)、
  [SyncAll](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0204.html)。
- 新 `square_sum_workspace.h` 复用合法BF16 full-row Copy/Index/组树/输出helpers，不重构
  half、FP32、suffix、L2或旧BF16路径。UB为 `TightBf16Bytes(I)+2*P`，唯一变化是单VECOUT
  槽从P个BF16改为P个float；本次private-slot修订不改变任何UB分配。官方case2为
  **188800 B**，N204732/L5119/slots5，workspace **204800 B**。
- 仅借鉴生产者独占暂存这一公开并行算法思想；布局和逆映射独立推导，未读取/复制
  `SSV1-others.zip`或对方源码，不使用每O固定两槽或其owner公式。

## 单kernel生产/消费及地址/生命周期

1. 全部真实输入行沿用自己的CalcRange商余均分（每核行数差<=1），核内只按O边界拆段。
   本核firstO=rowStart/R，当前o的slot=`core*slots+(o-firstO)`；不同生产者写区独占。
   固定B80、G=8/gcd(I,8)，真实地址row*I；复用双输入槽，FP32 Cast/平方/组树累加至
   独立acc[G*I+8]，每段重置acc含sentinel。尾批v=r*I、e=Align(v,16)、
   active=ceil(r/G)*G*I，DMA只读v、补到e，Cast后只在active>e时零填[e,active)，不丢尾。
   full-I byte index每核只构造一次，Gather至[G,P]后沿G树归约。首P个float（含零padding）
   复制到单VECOUT槽，经EnQue/DeQue后normal DataCopy完整P floats到ws[slot*P]。
   FreeTensor保护跨段MTE3写源复用，不让它读取随后覆盖的sumBuf。无atomic、无全区clear。
2. 全部段结束保守PIPE_ALL完成写入，再全核执行唯一SyncAll；屏障前无空核提前返回。
   每个被读取slot都由本次调用完整覆盖；unused槽可保留毒值但永不读取，不是跨调用缓存。
3. 按旧32B BF16 output ownership划分，按O边界裁剪count/innerOffset。独立逆映射：
   `base=N/C, rem=N%C, cut=rem*(base+1)`；
   `owner(row)=row<cut ? row/(base+1) : rem+(row-cut)/base`。mode7门禁保证base>=80。
   输出o的贡献核为owner(o*R)..owner((o+1)*R-1)，连续遍历任意多个核，不限制两贡献者。
   每核`coreStart=core*base+min(core,rem)`，读取slot=`core*slots+(o-coreStart/R)`。
   每个slot仅精确读count个float，DMA右补零至e8=Align(count,8)。calcBuf首
   e16=Align(count,16)先完整零填，然后只Add真实count；input的[e8,e16)不读也不清。
   最终Cast从calc取e16个float，RINT转BF16仅写真实count*2 bytes。标准inQueue/outQueue
   生命周期保护复用，不为Cast多读workspace或其他列/O。

host保证input总元素可由uint32表示且I>=64，故row*I和(outer+1)*R都在已验证输入界内；
workspace至多1MiB，P*4按64B对齐。G*I及P都是8float倍数，连续count组树不窄化repeat/
stride，sentinel、index及全部scratch均已计入UB。实际CANN DataCopy/DataCopyPad重载、
PIPE_ALL/硬件SyncAll的跨核可见性、队列跨dtype复用、FP32合并舍入误差和性能仍须设备验证，
CPU不能证明这些合同。

## 008本地验证及预定设备门禁

```bash
PYTHONPATH=tests python3 -m unittest -v \
  test_square_sum_workspace.SquareSumWorkspaceTests.test_square_sum_workspace_4097x65_model \
  test_square_sum_workspace.SquareSumWorkspaceTests.test_square_sum_workspace_official_case2_host \
  test_square_sum_workspace.SquareSumWorkspaceTests.test_square_sum_workspace_5x769x129_model \
  test_square_sum_workspace.SquareSumWorkspaceTests.test_square_sum_workspace_1025x1024x256_cap_fallback_host
```

mode7初次实现命令 **Ran 4 tests in 2.812s, OK**；当时writer预算4/4已用完，CPU共4/10；
加上上述full-row补充后共6/10。本次最终private-slot修订仅再执行上列四个输入一次：
**Ran 4 tests in 4.890s, OK**，累计 **10/10**，预算耗尽，不再运行任何CPU输入。
真实host/policy及从新头抽取的生产调度、消费方法、owner/slot、既有CalcRange与
CalcPlaneOutputRange以g++ C++17、
`-Wall -Wextra -Werror`、undefined/bounds sanitizer编译。恰四个实际输入，无隐藏subtests：

- BF16 `[1,4097,65]` axes1：slots1/12800B，40贡献核、输出空核同步、输入恰一次覆盖、
  B80/组尾、完整私有P-float写、无clear/atomic/仅1屏障、末列1、calc e16零填及BF16输出。
- 官方case2仅host/UB/完整行调度，不分配大张量：mode7、40核、204732行、上述workspace/UB。
- BF16 `[5,769,129]` axes1：slots2/46080B，行范围跨O、多生产者、未写slot保持NaN毒值
  且绝不读取、input e8..e16毒值不参与Add/Cast、裁剪转换与完整数值/输入覆盖。
- BF16 `[1025,1024,256]`仅host：slots27/1105920B>1MiB拒绝，保持mode1/32B、40核。

小模型使用可精确表示的FP32平方/累加，验证索引和完整贡献，不冒充设备数值容差或同步
验证。未重跑旧测试、全套、NPU或修改计时/transport，未提交/暂存。额外设备JSON
`benchmarks/square_sum_workspace_checks.json` 最初为上述两个小模型shape，均BF16 axes1 keep=true；
此前已追加full-row extra-3；本修订未改JSON，仍3项（前两项purpose里的atomic/clear为旧描述）。
父审后唯一fresh build，同一.run先预ZIP再私有安装；
官方五点及三个extra正确性通过后预定
**3轮完整五点、每点独立进程**性能。失败即留证结束，不追加诊断/编译/重包，最终STOPPED。

以下006及更早实现说明是历史记录，不作为008设备证据。

## 006唯一两项BF16变化

- `square_sum_bf16.h` 在既有 balanced/tight 门禁内保持32B output ownership。仅当
  `2*Align(outputCount,16) <= innerSize` 时选择窄owned-slice；共享policy按已分配的
  `80*innerSize` 元素容量求 `batchRows=min(255, capacity/Align(outputCount,16))`。
  `CopyBalancedInnerBatch(fullRowBase+innerOffset, ...)` 二维DMA只读owned列，源行间隔仍为
  真实innerSize，每行右补零到pitch；没有Gather或索引生成。完整R以双输入槽预取，Cast到
  FP32、原位Mul、原 `AccumulateRows` 行树，只清零/累加acc首pitch个float。Cast/PIPE_V后
  释放输入，树完成/PIPE_V后复用calc；R尾部不丢。较宽片段及完整行仍是005的full-row/Gather。
  两分支共用精确输出helper，保持BF16 RINT和单VECOUT队列跨O的MTE3复用保护。
- `square_sum_suffix.h` 的BF16 fused大suffix固定1024列，不按输入shape选择；每个输入槽
  额外分配32B物理padding，但DMA/Cast从不访问padding。完整UB公式：
  `8*rows*1024 + 64 + 12*Align(rows,16) <= 192KiB`，最大23行、
  **188864 B**；24行超预算。FP32 sum/acc覆盖BF16转换上取整长度并将padding置零；
  非BF16/nonfused的实际分配不变。
- suffix每轮令 `half=ceil(active/128)*64`，只把后 `active-half` 列加到前部对应列，
  其余前half原值保留，接着 `active=half`，最后对真实active（<=64）做WholeReduceSum。
  所有Add UB起点按64float对齐；最大rowStride=128个32B块、repeat=23，不溢出uint8。
  129尾先把第128列加至第0列，再合并64列，完整保留尾贡献，不跨行。主DMA块长2048B，
  源stride始终由真实reduceSize求得；尾部按真实count搬运并右补零，最终只写真实输出。
  Cast/Mul/fold/WRS/acc均为FP32，明确PIPE_V保护依赖和scratch复用；不缩放、不补偿。

## 006本地验证及预定设备门禁

实际CPU输入恰好 **4/10**，writer预算4/4已用完，无隐藏subtest、未运行005四点或binding套件：

```bash
PYTHONPATH=tests python3 -m unittest -v \
  test_square_sum_bf16_wide.SquareSumBf16WideTests.test_square_sum_bf16_narrow_slice_model \
  test_square_sum_bf16_wide.SquareSumBf16WideTests.test_square_sum_bf16_official_case2_slice_policy \
  test_square_sum_bf16_wide.SquareSumBf16WideTests.test_square_sum_suffix_1024_plus_129_model \
  test_square_sum_bf16_wide.SquareSumBf16WideTests.test_square_sum_suffix_official_case3_host_coverage
```

首次命令在setUpClass编译测试probe时遇到 `ops`/`first` 名称冲突，**Ran 0 tests**、没有实际
host输入；改名后同命令 **Ran 4 tests in 2.204s, OK**。真实host/policy/range及从源码抽取的
`AccumulateRows`/suffix fold语句以g++ C++17、`-Wall -Wextra -Werror`、undefined/bounds
sanitizer编译。四个输入依次为 `[3,517,129]` axes1、官方case2 host/policy（不建大张量）、
`[17,1153]` axes1、官方case3 host/policy（不建大张量）。模型检查255+255+7、输入槽回绕、
跨O、acc污染/精确ownership，及完整1024+129数值/lineage、32B operand起点和完整输出coverage。
父审将FP32 sum/acc补齐到BF16转换所需的32个元素，预算修正为23行/188864B，
并仅重跑官方case3 host/policy门禁1个实际输入；本候选CPU共5/10。旧refactor期望同步更新，
未运行其旧套件。

新增 `benchmarks/square_sum_bf16_wide_checks.json` **恰两个** BF16、keep=true额外设备输入，
即上述两个模型shape。CPU整数/lineage模型不是CANN执行，不能证明实际DMA、队列跨PIPE事件、
UB分配/指令重载或设备浮点误差；也没有性能保证。父审后统一checkpoint、上传、唯一fresh build，
同一`.run`先预ZIP再私有安装；官方五点及两个extra正确性通过后，预定 **3轮完整五点**、每点
独立进程性能。失败即保存证据结束，不追加诊断/编译/重包，最终notebook STOPPED。

## 005实现时记录（历史快照，状态以上文为准）

当时005仅完成本地实现/CPU门禁，未做 CANN fresh build、NPU正确性或性能测量，未发布。
004编译、官方五点及extra正确性已过，首完整轮总和1624.202us；但SSH监控中断导致三轮不完整，
已失败归档，不能用作发布门禁。005是新候选，不补测004，不改变官方计时或transport。

## 当前唯一改动：FP32 tight atomic inner

- 在既有 FP32 atomic inner 门禁内替换 `op_kernel/square_sum_tight.h`，不另加第三套tight算法。
  host、核数、清零、全核 `SyncAll` 和精确 atomic 输出保持不变；超预算仍走旧 main/tail atomic。
  004的BF16 tight、half、suffix、平台L2策略及其他模式均不改。
- 固定 `B=40`、`G=8/gcd(I,8)`、`W=G*I`，双输入槽仍每批一次单 block `DataCopyPad` 连续读入
  全部真实数据。每核从 `core*B` 开始，以 `GetBlockNum()*B` 交错完整row tile；起点、步长和
  next候选使用64位，判界后才窄化。所有full tile GM起点32B对齐，只有全局最后tile有尾部。
  空核不提前返回，仍先参与初始化/同步，最后atomic合并。
- 尾批 `v=r*I, e=Align(v,8), N=ceil(r/G)*W`：DMA精确读v并补零 `[v,e)`；DeQue后仅
  `Duplicate[e,N)`，原地平方N个元素。将 `[ceil(r/G),W]` 沿组轴树形Add后累加到独立
  `acc[W+8]`；B40/G8的full批5组和尾批奇数组都完整参与。使用连续count重载，不窄化为uint8。
  预取只占另一个空槽；最终acc Add和PIPE_V后才释放当前槽，不在FreeTensor后访问它。
- 最终固定 `C=64` 列分块：float scratch `[G,64]`、uint32 index `[G,64]`，另用现有未占用的
  `calcBuf` 分配float output `[Align(I,8)]`。CreateVecIndex先生成int32，再转byte并修正row基址，
  首块索引为 `4*(r*I+c)`；后续每块全index加256。最后不足64列时先V_S，将所有row的padding
  索引修复为 `4*W` 的独立零sentinel，再S_V；没有acc越界访问。
- Gather后仅沿G树加，将首行 `Align(count,8)` 个float用Adds零复制到output的32B对齐块起点，
  padding为零。PIPE_V保证scratch读取完毕才进入下一Gather；全部列完成后以V_MTE3保护
  `SetAtomicAdd<float> → DataCopyPad(I*4 bytes) → SetAtomicNone`，只输出一次，output不再复用。
- `square_sum_policy.h` 用完整64位算式判UB：
  `2*B*I*4 + (G*I+8)*4 + 2*G*64*4 + Align(I,8)*4 <= 192KiB`。
  I521：双输入166720 + acc/sentinel16704 + scratch/index4096 + output2112 = **189632 B**。
  I539为 **196032 B**、可用；I541为 **196736 B**、fallback。预算未扩展既有
  `AtomicRows` / `ContiguousInner` 资格，也未修改BF16预算。

## 005本地验证及预定设备检查

writer恰好执行 **4个实际CPU输入**，全部通过（2.192秒）：

```bash
PYTHONPATH=tests python3 -m unittest -v \
  test_square_sum_tight.SquareSumTightTests.test_tiled_tail_slots_tree_gather_model \
  test_square_sum_tight.SquareSumTightTests.test_tiled_official_case5_host_coverage \
  test_square_sum_tight.SquareSumTightTests.test_tiled_odd_width_ub_fits \
  test_square_sum_tight.SquareSumTightTests.test_tiled_odd_width_ub_fallback
```

- `[137,129]` 同一输入的实际host和整数/贡献集合模型：40+40+40+17、slot回绕、Q5/Q3、
  v/e/N=2193/2200/3096的双零区间、64+64+1列、sentinel、独立output和逐列每个输入恰贡献一次。
  编译执行实际device调度语句，用真实host给出的40核推导全局extra的row调度（不再生成CPU张量）。
- 官方Case5实际host/policy和完整row覆盖：271670行、6792个tile，唯一尾tile为core31的30行；
  所有起点对齐，无重叠或漏行，完整预算189632 B。
- `[137,539]`、`[137,541]` 各一次实际host请求，分别验证UB fit/fallback和旧atomic仍有资格。

父本候选已用2个transport mock，writer4个，合计 **6/10**；未重跑旧B32测试、全量套件或额外
输入。CPU host/调度probe使用现有stubs、g++ C++17、`-Wall -Wextra -Werror`及undefined/bounds
sanitizer；模型不是device执行，不能证明CANN重载/UB实际分配、VECIN队列复用、V/S和V_MTE3
同步、FP32浮点误差或性能。未做NPU操作、提交或暂存。

新 `benchmarks/square_sum_tiled_checks.json` 仅预定FP32 `[4817,129]`、axes `[0]`、keep=true，
**621393元素**：40核下core0为137行（40+40+40+17），其他核120行；完整输入及所有双尾保留。
旧extra配置不改，但005批次应选择新配置。父负责审阅/checkpoint、唯一fresh build、同一`.run`
先预生成唯一ZIP再私有安装；官方五点及该extra正确性通过后才运行预定 **3轮完整五点、每点
独立进程性能**，失败即保存证据退出。所有证据对应同一源码/二进制/ZIP，最终notebook STOPPED。

## 首个结构化重构候选（历史说明）

下列布局、算法与本地验证记录属于前一重构候选；其中 FP32 atomic 路径现在仅为超预算
fallback，当前 tight 路径及其数值顺序以本页上半部分为准。通用批次入口见
`benchmarks/SQUARE_SUM_BATCH.md`，前候选预定一轮筛选未执行至性能阶段。

## 源码布局

- `op_host/square_sum_v1.cpp`：tiling 入口、shape/type 推导、算子注册。
- `op_host/square_sum_shape.h`：轴规范化、维度/stride 及 reduction 结构识别。
- `op_host/square_sum_tiling_policy.h`：任务划分和通用模式选择。
- `op_kernel/square_sum_v1.cpp`：仅三个 dtype 的入口。
- `op_kernel/square_sum_kernel.h`：唯一模板 state 和调度。
- `op_kernel/square_sum_{suffix,inner,plane,fallback}.h`：类内算法实现；suffix 包含完整行
  contiguous-block reduction，inner 包含 rank4 reduce02。
- `op_kernel/square_sum_{numeric,index}.h`：数值转换、归约与索引私有方法。
- `op_kernel/square_sum_policy.h`：host/device 共用的结构/UB 参数，放在打包目录内。

不另造接口层；`zip_op.sh` 递归打包 host/kernel 及全部所需头。移除大型 float 特化和
BF16 重复调度，通过 `if constexpr` 仅保留数值转换、scratch/队列选择的必要差别。

## 算法约束（无精确输入 shape 匹配）

| 路径 | 选择/参数依据 | 五点中保留参数及 UB |
| --- | --- | --- |
| BF16 balanced inner | 连续中间归约，尾列<=256；batch=min(255, floor((190KiB−6*pitch)/(8*pitch)))；32B 输出 ownership，每段跨 outer 行裁剪 | Case2：pitch256、94行、194048 B |
| FP32 atomic inner | 连续归约且无 outer 输出；main=floor(cols/64)*64、tail=cols-main；main repeat stride<=255；双slot+acc 按192KiB预算，零尾不分配/访问第二队列 | Case5：main512、tail9/pitch16、46行、196416 B，余192 B |
| half tight plane gather | reduction prefix + 单outer连续output plane + reduction suffix；64<suffix<=128；Gather包含独立zero元素；按精确192KiB预算求<=255行并向下8行对齐 | Case4：source69、padded72、168行、190592 B |
| BF16 fused large suffix | mode0大suffix，DMA源stride可表示；176列为11个BF16 DMA块、至多三条64-lane vector stripe；两输入、两输出、一次FP32展开，按192KiB求batch | Case3：176列、138行、196032 B |

Case3 **仍走 mode0 fused suffix**：完整行 block-reduce 的176KiB预算仅容14行，16行对齐后
为0，因此不能删掉这个实际热路径。泛化后的其它形状由结构/资源限制而非窄shape区间选择。

普通 half wide-plane 优化使用 suffix 为128整倍数的完整双stripe条件；已经被 tight gather
调度提前处理的重复 plane 分支删除。32B ownership、完整尾部、vector repeat/stride 上限
保留。所有 atomic 核先参与初始化/SyncAll，空工作核不提前返回。

五点主要数值顺序不变：BF16 inner 的FP32平方+原树累加、fused suffix 的原stripe折叠、
half tight plane 的原prefix累加后gather，以及FP32 atomic的逐行 MulAddDst 均保留。
**未同时启用下一候选的FP32树归约**，没有缩放/补偿或固定结果。

## 有预算的本地验证

已执行以下 **6 个实际输入**，每个方法只请求一个真实host输入；只编译本算子的实际host
及共享policy，并将实际device `CalcRange` 方法用于单个泛化输入的CPU分区检查：

```bash
PYTHONPATH=tests python3 -m unittest \
  test_square_sum_refactor.SquareSumRefactorTests.test_case1_small_suffix \
  test_square_sum_refactor.SquareSumRefactorTests.test_case2_balanced_inner \
  test_square_sum_refactor.SquareSumRefactorTests.test_case3_live_fused_suffix \
  test_square_sum_refactor.SquareSumRefactorTests.test_case4_tight_plane \
  test_square_sum_refactor.SquareSumRefactorTests.test_case5_atomic_inner \
  test_square_sum_refactor.SquareSumRefactorTests.test_general_atomic_zero_tail_empty_cores
```

五点均 keep=true，第六点为 FP32 `[17,64]` reduce axis0：核数40、23个空核、零tail，
精确覆盖所有17行且没有重复/漏行。与脚本4例相加，本候选已用满10例，不再追加/重复运行。
CPU stubs 不能证明 CANN kernel 编译、DMA/同步或设备数值正确性。

父审末轮另修复原有 rank4 普通路径的短 half tile 缓冲边界：acc 改按
`Align<T>(accTileElems)` 个 float 分配，覆盖 AccumulateSquaredRows 实际访问的输入行
padding。tile=1 时由8增至16个float；half 最大tile192、64行双输入槽+输出+acc总计
50304 B，足够容纳。仅静态核算，预算已尽，未追加CPU/NPU用例；算法/调度不变。

计划内 NPU 检查只有五点及显式配置的三小输入，详见
`benchmarks/square_sum_refactor_checks.json`；先正确性、失败即退，再独立进程五点性能。
三小输入没有在 CPU 进行额外张量测试。编译耗时、真实 UB/指令合法性、性能回退风险须由
父会话授权的同一 fresh build 批次确认，不能用这次纯重构宣称进步。
