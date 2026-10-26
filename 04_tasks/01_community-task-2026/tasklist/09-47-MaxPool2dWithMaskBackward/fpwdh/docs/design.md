# MaxPool2dWithMaskBackward 算子设计文档

# 1 需求背景（required）

## 1.1 需求来源

CANN 社区任务 2026 · 9 月社区任务：实现 `aclnnMaxPool2dWithMask` 正向算子的反向传播算子
`MaxPool2dWithMaskBackward`，基于 Ascend C 编程语言，验收通过后合入
`cann/ops-nn` 仓 `experimental/pooling`。

## 1.2 背景介绍

### 1.2.1 正向算子与反向语义

正向 `aclnnMaxPool2dWithMask`（2D 最大池化）输出 `out` 与 `indices`。`indices` 为
**mask 语义容器**：shape `[N, C, kH*kW, ((Ho*Wo+15)//16 + 1) * 2 * 16]`，dtype `INT8`；
其**前 `4*N*C*Ho*Wo` 个字节**按小端解释为 `INT32` argmax 数组，取值为该输出点最大值在
`(n,c)` 平面内的线性索引 `ih*W + iw`（每通道独立），容器尾部补 0。
该容器与 `INT32` argmax 数组**字节兼容**，因此反向算子无需任何 dtype 转换即可消费。

反向传播语义（与任务书公式一致）：

```
gradInput[n, c, ih, iw] += gradOutput[n, c, oh, ow]    # (ih,iw) 为该窗口的 argmax
gradInput 其余位置 = 0
```

由于 argmax 唯一，对固定输入位置 `q` 与固定窗口偏移 `(kh,kw)`，最多存在**一个**输出位置
`p` 满足 `argmax[p] == q`；因此 `gradInput[q]` 是至多 `kH*kW` 项的和，且各项来自互不相同的
`(kh,kw)`。

### 1.2.2 现状分析

| 项 | 现状 |
|---|---|
| 基线实现 | 同族 3D 算子 `MaxPool3DGradWithArgmax`（ops-nn `pooling/`），2D-with-mask 反向由 TBE 承载 |
| 性能要求 | Atlas 800T A2 上整体性能与现有实现持平，不低于原 TBE 的 95% |
| 数据类型 | `FLOAT32 / FLOAT16 / BFLOAT16`（BF16 仅 Atlas 800T A2） |
| 硬件 | Atlas 800T A2（ascend910b / ascend910_93）、Atlas 300V Pro（ascend310p，需真机回归） |
| 确定性 | 默认非确定性，支持 `aclrtCtxSetSysParamOpt` 开启确定性 |

# 2 需求分析（required）

## 2.1 需求描述

Ascend C 实现 `MaxPool2dWithMaskBackward`：支持 3 种 dtype、NCHW 4-D 输入、
`kernelSize/stride/padding/dilation/ceilMode` 属性（`dilation==1`，`padding ∈ [0, kernelSize]`，
`stride` 长度 0 时取 `kernelSize`）、泛化 shape（含 1×1、超 UB 大图、窗口重叠），
精度满足生态算子开源精度标准，性能不低于 TBE 基线的 95%。

## 2.2 需求拆解

1. 输入输出契约与 shape/attr 校验（op_host）。
2. 泛化 Tiling：lane 宽度、核间切分、核内 UB 切分、重叠/确定性分支。
3. 核函数：向量化 scatter-add，解决"逐元素散射无法向量化"的核心矛盾。
4. 精度：FP32 累加 + 单次 Cast，对齐 golden 语义。
5. 140 个官方用例全覆盖自测 + 内核算法仿真验证。

# 3 详细设计（required）

## 3.1 核心矛盾与方案选型

反向本质是**散射累加**（scatter-add）：第 `p` 个输出写到 `argmax[p]` 指定的任意位置。
向量单元无法做任意步长的**块内散射写**；而块跨度寻址的粒度是 32B（8 个 fp32 / 16 个 fp16），
因此步长 `sw`（1~8）无法直接表达。

**方案：把向量 lane 从"空间位置"换成"NC 通道"。** lane 沿 NC 排布后，"输出位置前进一格"
恰好等于 `sw * baseNc` 个元素 = `sw` 个 32B 块，块跨度寻址即可表达。代价是 argmax/grad/y
需在 UB 内做 `(空间主序, nc 次序)` ↔ `(nc 主序, 空间次序)` 的转置，用 `TransDataTo5HD`
（16×16 tile，arch22 原生支持）完成。该架构与已量产的 `MaxPool3DGradWithArgmax` 一致，
是本实现复用与对齐的基准。

四步计算（对每个输出 tile）：

```
for kh, kw:
    Img2Col : tmpIndex[pos, lane] = 候选输入线性索引 (ih*Wi+iw)      # Adds + 块跨度
    Compare : mask = (tmpIndex == argmaxTran[pos, lane])             # CMPMODE::EQ
    Select  : maskedGrad = mask ? gradTran[pos, lane] : 0            # 16bit 再 Cast 成 fp32
    Col2Img : yTran[ih, iw, lane] += maskedGrad                       # Add + 块跨度
```

`yTran` 是该 tile 覆盖的输入区域 `baseHi × baseWi` 的 FP32 累加器，`baseHi=(baseHo-1)*sh+kh`、
`baseWi=(baseWo-1)*sw+kw`。

## 3.2 精度设计

- 全程 **FP32 累加**（`yTran`、`tmpGrad` 均为 fp32），仅在写回前 Cast 一次
  （fp16 用 `CAST_NONE`，bf16 用 `CAST_RINT`），与 golden 的 `np.add.at` + 单次 cast 语义对齐。
- 窗口重叠时 16-bit 输出**不能**对 fp16/bf16 做 GM 原子累加（精度与指令支持双重限制），
  改为：FP32 workspace 原子累加 → 第二趟统一 Cast 写回（tiling key=100，`isNeedWorkspace`）。
- 本机仿真结果：140 用例 × 6 组 tiling 配置全部通过，非 bit-exact 用例最大绝对误差 ≤ 1.9e-6，
  来源为 FP32 求和顺序与 golden（按输出升序）不同，远小于 1e-4 容差。

## 3.3 host 侧设计（tiling）

### 3.3.1 参数与校验

`self` 的 shape 给出 `gradInput` 形状；`gradOutput` 给出 `Ho/Wo`；host 侧按 golden 同款公式
重算 `Ho/Wo` 并**强校验一致**（ceilMode 含 `-1` 修正项），不一致直接返回
`GRAPH_FAILED`。dtype 三处（gradOutput/self/gradInput）必须一致，`indices` 必须 `INT8`。

### 3.3.2 UB 预算（`CalUBTotalSize`）

单 tile 的 UB 布局（与 kernel 侧 `InitUbBuffer` 严格一一对应）：

```
| xIndex | argmax | grad | yTran | argmaxTran | gradTran | mask | y |
```

- `xIndex`：`baseHi*baseWi*vl` 个 int32（候选索引，按公式生成，不从 GM 搬入）
- `argmax`/`argmaxTran`、`grad`/`gradTran`：`baseHoWo` 对齐到 8（int32/fp32）或 16（fp16）× vl
- `yTran`：恒为 fp32，`baseHi*baseWiAlign*vl`
- `y`：转置回 nc 主序后的写回中转
- `mask`：Compare 的 1B/元素谓词，按 64B 取整（Compare 按 64 元素 repeat 写满）
- 另预留 `SELECT_RESERVED_UB_SIZE = 8192` 给 Select/Cast 内部

### 3.3.3 分层切分策略

1. **lane 宽度 `vl`**（∈ {16,32,48,64}，须为 16 的倍数以满足转置约束、≤ 64 以匹配
   `MAX_LIST_NUM=4`）：`ncDim ≥ 64` 默认 64；`ncDim < 64` 时取 `AlignUp(ncDim,16)`；
   若整平面仍放不下则降为 16。
2. **核间切分**：优先级 `nc → ho → wo`。关键规则：若"单核独占整平面"（`base=平面全尺寸`）
   成立，则**清掉重叠标记**（`isOverLap=false`）——每核只写自己的 nc 通道，天然无跨核冲突，
   可走免原子直写快路。
3. **确定性模式**（`GetDeterministic()==1` 且存在重叠）：只切 `nc`（每核独占完整空间域），
   同核内的原子加按程序序执行 ⇒ 结果确定；不做跨核重叠切分。
4. **核内切分**：`baseHo`/`baseWo` 对 UB 预算做二分搜索（先切 Ho，不够再切 Ho&Wo），
   与参考实现一致。
5. 产出 `tilingKey`：`0` = 无重叠（直写）、`100` = 重叠（原子累加，16-bit 加 workspace 第二趟）。

### 3.3.4 零初始化（`needInitOutput`）

`gradInput` 中未被任何窗口覆盖的位置必须为 0。host 侧判定：

```
(hoDim*kh < hiDim+padHTop+padHBottom) || (woDim*kw < wiDim+padWTop+padWBottom)
|| ((hoDim-1)*sh+kh < hiDim+padHTop) || ((woDim-1)*sw+kw < wiDim+padWTop) || isOverLap
```

命中则先由各核分片把输出（或 workspace）清零，`SyncAll()` 后再进入累加。
非重叠且全覆盖时跳过清零：每个 tile 会把自己有效区域**整块写出**（argmax 命中处为梯度，
其余为 0），等价于覆盖写。

## 3.4 kernel 侧设计

- 入口 `max_pool2d_with_mask_backward(gradOutput, self, indices, gradInput, workspace, tiling)`，
  `self` 仅参与 host 侧 shape 推导，kernel 不读。
- 模板参数 `<TY, IsOverlap>`，由 tiling key 选择实例化；`self`/`indices` 的 dtype 差异
  （int8 容器 → int32 argmax）在 kernel 内以 `GlobalTensor<int32_t>` 直接 reinterpret。
- 关键实现点：
  1. `GenXInitIndex`：用倍增 + `Adds` 在 UB 内生成整块输入区域的候选索引（含 padding 的
     halo 区），避免从 GM 搬运；padding 区的候选值虽落在窗口外，但要么为负/越界、要么其
     线性值对应的真实位置不在窗口内，且 `CopyOut` 只读 `[0, hiValid) × [0, wiValid)`，
     故不会产生假命中。
  2. `DataCopyPad` 进/出的 stride 全部按"块数取整"对齐计算
     （如 `(baseHiWiAlign - AlignUp(wiValid, 8)) / 8`），规避 32B 粒度截断这一隐性偏移。
  3. 空输入三层防御：host 早退 + tiling 1 核空计划 + kernel 循环上界守卫。
  4. `SyncAll()` 前后用显式 event（`V_MTE3`/`MTE2_V`/`V_MTE2`/`MTE3_V`）同步，避免 UB 复用竞争。

## 3.5 目录与接入

```
experimental/pooling/max_pool2d_with_mask_backward/
├── CMakeLists.txt     # ACLNNTYPE aclnn：op_api 由框架 autogen 生成
├── op_host/           # def / infershape / tiling(.h/.cpp)
├── op_kernel/         # 入口 .cpp + common.h + normal.h
├── examples/          # aclnn 调用示例
└── tests/             # 140 用例数据准备 / ST 驱动 / golden 对拍
```

不手写 `op_api` 的原因：int8 容器与 int32 argmax **字节兼容**，autogen 生成的 aclnn 接口
直接透传指针即可，无需转换层；手写反而需要 `ACLNNTYPE aclnn_exclude` 并承担与 autogen
符号冲突的维护成本。

# 4 特性交叉分析

无（不涉及广播、不依赖其他算子的 op_host/op_kernel）。

# 5 可维可测分析（required）

## 5.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|:---:|:---:|:---:|
| 精度 | FP32/FP16/BF16 全部 140 用例通过生态算子开源精度标准（atol=1e-4, rtol=1e-3） | 任务书 + `opbase/docs/ops_precision_standard` |
| 性能 | Atlas 800T A2 整体不低于原 TBE 的 95%；小 shape（<100us）超出 TBE 30% 以上时提供性能仿真图与分析结论 | 任务书 |

**性能口径**：msprof 导出 op_summary，取 **逐 kernel `Task Duration(us)` 求和** 与 TBE 基线
同口径对比；对比 shape 必须超出 L2（避免 cache 命中高估）；多轮取中位数抗噪声。
本算子为访存密集型（读 gradOutput+indices、写 gradInput 各一遍，计算量为 `kH*kW` 次
向量三操作），理论上界由 HBM 带宽决定，优化抓手是 lane 宽度（满 64 lane）与减少
`DePad`/转置的无效带宽。

## 5.2 兼容性分析

新算子，不涉及兼容性分析。

## 5.3 验证状态（截至本设计文档）

| 验证项 | 状态 |
|---|---|
| 内核算法仿真（tiling + 数据流，140 用例 × 6 组配置） | **已通过**（`develop/tools/sim/`，报告见 `tools/sim/report_ub196608_cores40.md`） |
| 编译（bisheng / CI `Compile_*_A2`） | **已通过**（真机 CANN 9.1.0 `build.sh --pkg --experimental --soc=ascend910b`，0 error） |
| 真机精度 ST（140 用例） | **已通过**（140/140，见 5.4） |
| 真机性能 vs TBE | **未验证**：验收机上无 TBE 基线可对拍，仅交付本实现实测数据（见 5.4.2） |
| Atlas 300V Pro（ascend310p） | 已注册配置，向量核数与 `TransDataTo5HD` 支持度**未验证** |

## 5.4 真机实测记录（CANN 9.1.0 / Atlas 800T A2 / 910B / 40 核）

### 5.4.1 精度

`tests/` 全量 ST：`./st_driver data` → `python3.11 compare.py`

```
==== 140/140 cases executed
==== 140/140 PASS; worst 5:
```

对 `Test_070_fp16_medium_k4x6s3x5p2x3c`（16-bit + 重叠，走 workspace 第二趟 Cast）
单独复算：`max_abs = 0.000e+00`，即**与官方 golden 逐位一致**。

### 5.4.2 性能实测与"workspace 访问代价"分析

全量 140 用例逐 kernel 时长（`PERF=2` 取事件均值）：

| 区间 | 用例数 |
|---|---|
| < 100 us | 48 |
| 100 us – 1 ms | 34 |
| 1 ms – 10 ms | 4 |
| > 100 ms | **39**（全部是宽平面 + 大核：二维切分后 FP32 累加器仍放不下 UB，回退 `tilingKey=100`） |

- FP32（59 例，无 workspace）：中位 117 us，最大 6.35 ms（`Test_035` k3x3 s1x1，
  每元素 9 次原子累加，属原子带宽受限）。
- 16-bit（81 例）：中位 522 us，其中 42 例 < 1 ms。
- 本表为**最终状态**（含已启用的 key 200 二维路径）。改造前为 <100 us 48 例、
  100 us–1 ms 34 例、>100 ms 54 例，FP16/BF16 中位 1.30 / 1.51 s —— 原中位与
  "是否走 workspace"完全相关，正是下面 key 200 改造要消除的部分。

**结论：慢的根因是"AI Core 访问 aclnn workspace 显存"在该环境被极度降速，不是算法问题。**
定位过程与证据（复现脚本见 `scratch/r_*.sh`，均在真机执行）：

分段测量（临时 `STAGE_MASK` / `CAST_SKIP_READ` 旋钮，`Test_070`，workspace 偏移 0）：

| 只执行的段 | 耗时 | 说明 |
|---|---|---|
| 什么都不做（`STAGE_MASK=0`） | 25.4 us | launch/executor 正常 |
| 只做 init（写 34 MiB 到 workspace） | **1989 us→1.989 s** | 写方向慢 |
| 只做累加（对 workspace 原子加，**只触碰约 1.8 MB**） | **1.917 s** | 数据量与耗时无关 |
| 只做 cast（读 workspace） | **1.944 s** | 读方向同样慢 |
| cast 但不读 workspace（只写 17.9 MB 到 yGm） | **26.8 us** | 写 yGm 正常（≈670 GB/s） |

关键推论（数据直接给出，不是推测）：

1. **代价与数据量无关**：1.8 MB 原子加 ≈ 34 MiB 顺序写 ≈ 1.9~2.0 s。
2. **代价按"被触碰的显存页"计**：`Test_059`（暂存 110 KB，约 1 个 2 MiB 页）= 55 ms；
   `Test_070`（34 MiB，约 17 页）= 2.0 s ⇒ 约 **55~110 ms / 2 MiB 页**。
3. **只要不碰 workspace，就回到几十微秒量级**（26.8 us）——这是改造的目标上界。

| 实验 | 结果 |
|---|---|
| 同一块 52.5 MB workspace 用 runtime DMA：`aclrtMemsetAsync` / D2D | 6.4 ms / 114 us（≈8 GB/s，正常） |
| 同一缓冲区被 AI Core 读/写（kernel 内 `DataCopy`） | 34 MiB 往返 ≈ 1.9 s（≈36 MB/s，慢约 2 个数量级） |
| 张量缓冲（同 `aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST)`，同 flag）被 AI Core 写 | 25 MiB / 0.45 ms（≈55 GB/s，正常） |
| 换分配策略 `HUGE_FIRST/HUGE_ONLY/NORMAL_ONLY` | 均为 ~2.0 s（与分配策略无关） |
| kernel 只走"不碰 workspace"的分支（`STAGE_MASK=0`） | 21.9 us（说明 launch/executor 本身无异常） |
| 把暂存区整体搬到 workspace 的其它偏移（+2/16 MiB） | 只要**完整落在 buffer 内**就仍然 ~2.0 s；越界（写被丢弃）时才"快"，但结果错误 |
| 放大分配余量使偏移后的暂存区完整落界内 | 逐位正确 + 仍 ~2.0 s |

即：本环境下"op 触达 workspace 的大块数据"本身就是 ~36 MB/s 的量级，与地址、对齐、
分配策略、原子/普通写都无关；参考实现 `MaxPool3DGradWithArgmax`（本实现对齐的基线）
使用**完全相同的 workspace 机制**（`GetUserWorkspace()` + FP32 暂存 + 第二趟 Cast），
因此在同一环境下有同样的量级。

**已排除的假设（逐个实测否定）**：

1. *第二次 `SyncAll` 触发固定等待* —— 拆段测量显示"任一触碰 workspace 的阶段"都是同一量级，
   三段耗时**不叠加**；`STAGE_MASK=0`（内核完全不碰 workspace）仅 21.9 us。
2. *`SetScheduleMode(1)`（batch 调度）开销* —— `SCHED=0` 时仍 2.0 s。
3. *暂存区与 16 MB 系统 workspace 重叠* —— 把暂存区挪到系统段之后（或去掉系统段），
   耗时依旧（此前 62 ms 的"快"实为越界写被丢弃造成的假象，对拍显示结果错误）。
4. *MTE3 未落盘导致的跨核可见性竞态* —— 在第二次 `SyncAll()` 前显式加 `PipeBarrier<PIPE_ALL>()`
   未改变结果；误判来源同上（越界配置）。
5. *16-bit 直接原子累加可省掉 workspace* —— `ATOMIC16=1` 实验（fp16/bf16 直接 `SetAtomicAdd`
   到输出）对拍失败：`Test_100` bf16 最大绝对误差 3.675e+01、`Test_075` fp16 8.86 M/9.83 M
   元素超差，证实 3.2 节"16-bit 不能做 GM 原子累加"的判断。
6. *workspace 不是设备内存 / 分配路径问题* —— `aclrtPointerGetAttributes` 显示 workspace 与
   张量的 `location.type` 均为 1（设备内存）；把 workspace 改为"256 MB 设备大块的切片"
   （`WS_CARVE=1`）耗时仍为 2.008 s。排除内存类型与分配来源。

**免 workspace 改造：tiling key 200（已实现并真机验证）**

16-bit + 窗口重叠路径新增 **key 200**，不再"每核散写自己窗口的 footprint + 跨核原子"，
而是**按输入域把平面按行切成互不重叠的"所有权行带"，每个块只写自己那一段，并把前面
`haloH = floor((kh-1)/sh)` 个窗口行的贡献一起收进来，在 UB 内做 FP32 累加后 Cast 直接写回
`gradInput`**。因此该路径：**无 workspace、无原子、无第二趟 Cast、无跨核 `SyncAll`**，
全部流量只落在张量上。

实现要点（host `SetTilingParams/CalUBTotalSize/SetTilingData` + kernel `SubProcessNoWs`）：

- **二维所有权规则**：块拥有平面矩形
  行 `[ownRow0*sh - padHTop, (ownRow0+ownRowCnt)*sh - padHTop)`、
  列 `[ownCol0*sw - padWTop, (ownCol0+ownColCnt)*sw - padWTop)`（裁剪到平面内）。
  最后一个块没有后继，其所有权延伸到平面末尾（否则尾部 `kh-sh` 行 / `kw-sw` 列无人写）。
- **二维 halo**：窗口 `(oh, ow)` 会写到 `oh*sh + kh - 1 - padHTop` / `ow*sw + kw - 1 - padWTop`，
  因此块必须向前多看 `haloH = floor((kh-1)/sh)` 个窗口行、`haloW = floor((kw-1)/sw)` 个窗口列。
  列方向不切分时没有列边界，故此时 `haloW = 0`（host 只在真正切列时才把它打开）。
- **UB 预算**：`CalUBTotalSize` 在 key 200 下按"halo 扩展后的 footprint"计算，而块的处理步长
  仍是"所有权部分"（`ownStepH = baseHo - haloH`）；tiling 里的 `baseHo` 是扩展值，kernel 据此
  分配 UB 并按所有权步长推进。
- **tile 行距必须等于块窗口宽度**：argmax/grad tile 在 UB 里是"空间紧凑"排列（转置后空间步长
  为 `baseNc`），所以 `Img2ColPart/Col2ImgPart` 的候选索引行距必须等于块窗口宽度
  （这也解释了旧路径为何只在"列切分时 `baseHo` 恒为 1"的前提下成立）。令
  `woSpan = tiling baseWo`，并要求 host 保证它是 **16 的倍数**（`WO_TILE_ALIGN`），则：
  - tile 一行恰好是整数个 32B 块 → 切列后可用"每个窗口行一次 `DataCopyPad`"搬进来
    （`srcStride = woDim - woSpan`、`dstStride = 0`），UB 行距天然等于 `woSpan`；
  - 最后一个列块保持整块宽度并把窗口范围整体左移（`wc0 = woDim - woSpan`），
    因此**每个 tile 列都是真实窗口列**：既不越界读，也不需要额外的占位列清零。
- **UB 预算口径必须与 kernel 一致**（本轮的关键 bug，见 5.4.3）：key 200 的 copy-out 用 16-bit
  行布局，累加器行宽是 `AlignUp(baseWi, 16)` 而非 8；host 的 `CalUBTotalSize` 原先按 8 对齐
  估算，低估约 `baseHi*vl*8*6` 字节，导致 2D 配置**实际超出 UB**（`Test_075` 实需 197,568 B >
  196,608 B）。
- 放不下 UB 时自动回退 key 100（env 无关、纯 host 决策）。

实测（CANN 9.1.0 / 910B / 40 核，`PERF=2`）：

| 项 | 结果 |
|---|---|
| 精度 | **140/140 PASS** |
| key 分布 | key 0：44；key 100（workspace）：52；**key 200（免 workspace）：44** |
| 16-bit 重叠用例 | 81 例中 42 例 < 1 ms（改造前仅 16 例） |
| 典型提升 | `Test_059` 55 ms → **20.9 us**；`Test_068` 1.9 s → **521 us**；`Test_111`/`Test_097`/`Test_127` 等 ~2 s → **24~26 us** |
| 中位耗时 | FP32 **117 us**；16-bit **522 us**（改造前 1.3~1.5 s） |
| 整体分布 | <100 us 61 例、100 us–1 ms 36 例、1–10 ms 4 例、>100 ms 39 例（改造前 48 / 34 / 4 / 54） |

**剩余 39 例**：都是"宽平面 + 大核"（如 `Test_070` w=321、`Test_117` w=1024）。即便按二维切分，
单个 tile 的 FP32 累加器（`baseNc*baseHi*AlignUp(baseWi,16)*4 B`）仍放不进 192 KB UB，而
`baseNc` 已是转置要求的下限 16、无法再降。要覆盖它们需要改数据布局（例如累加器也做 16-bit
分块累加、或换一种分解），属另一次结构性改动，未在本轮实施。

### 5.4.3 修复记录（本轮真机迭代）

| 问题 | 处理 |
|---|---|
| `GetUserWorkspace()` 未调用，内核直接把 `workspace` 形参当暂存区（真实 base）+ 自己再加偏移 → 越界、结果错误 | 明确约定：内核收到的 `workspace` 已是**用户 workspace 基址**（框架语义同 `GetUserWorkspace`），暂存区偏移恒为 0；`wsOffset` 保留在 tiling 中并固定为 0，供后续扩展 |
| 第二趟 Cast 前缺显式流水排空 | 在 `SyncAll()` 前补 `PipeBarrier<PIPE_ALL>()` |
| **host `CalUBTotalSize` 的 `wiAlign` 与 kernel 不一致**（key 0/key 200 的 copy-out 行宽是 `AlignUp(baseWi,16)`，host 按 8 估算） | 低估约 `baseHi*vl*8*6` B → 二维配置实际超出 UB（`Test_075` 197,568 B > 196,608 B）。症状极具迷惑性：**坏点值全为 0（完全没写入）且两次运行结果不一致**。改为与 kernel 同一口径后 140/140 通过 |
| `ATOMIC16` 实验分支（已证伪） | 移除，代码回归单一实现路径 |
| key 200 首版（整宽行带）曾踩的三个坑：host 在 UB 不够时仍做核内列切分（kernel 却假设"块=整宽"）；`offsetArgmax` 误用全局窗口行号；给 `baseWo` 加 `haloW` 使"候选索引行距 ≠ argmax tile 紧凑行距" | 已被**二维版**统一取代：把"整宽"约束升级为"tile 行距 == 块窗口宽度"——host 令 `woSpan = baseWo` 且为 16 的倍数，kernel 用 `woSpan` 作 tile 行距并按窗口行搬运；`offsetArgmax/offsetYW` 仍保持相对本核 |
| key 200：最后一个块没有后继，尾部 `kh-sh` 行无人写（表现为"只有最后一行错"） | 最后一块的所有权延伸到平面末尾（`nextWinRow >= hoDim ⇒ ownR1 = hiDim`），下界由 `padHBottom` 收敛 |

> `PipeBarrier<PIPE_ALL>()` 不改变本环境结果，但它消除了"原子累加（MTE3 异步）→ 跨核
> `SyncAll()` → 第二趟 Cast 读取"这一序列上的可见性隐患（`SyncAll` 只保证各核到达屏障，
> 不保证在途 GM 写已落盘），属更严格的写法，保留。
