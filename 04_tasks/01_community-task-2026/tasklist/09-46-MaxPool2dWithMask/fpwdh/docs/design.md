# MaxPool2dWithMask 算子设计文档

> 目标：在 `ops-nn/experimental/pooling/max_pool2d_with_mask` 实现自研 2D 原生算子，
> 对外暴露 `aclnnMaxPool2dWithMask`，功能与内置实现一致、`out` 与 `indices` 逐字节可对齐。
> 适配：Atlas 800T A2（910B3，arch22/DAV_2201）+ CANN 9.1.0；Atlas 300V Pro 仅功能/精度。
> 关联笔记：`work/notes/semantics.md`（语义规格）、`refs_report.md`（参考实现研究）、`env_facts.md`（环境事实）。

---

## 1. 需求与语义规格

### 1.1 接口

`aclnnMaxPool2dWithMask(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, workspaceSize, executor)`

- 输入 `x`：`BFLOAT16 / FLOAT16 / FLOAT`，格式 `ND`，维度 **3 或 4**
  （4D=NCHW；3D 按 (C,H,W) 处理，内部视作 N=1，输出去掉 N 维）。自带 153 例中 3D 占 **30 条**。
- 输出 `out`：与 `x` 同 dtype；输出 `indices`：**INT8**（mask 容器的字节视图）。
- 属性：`kernelSize`(1~2,>0)、`stride`(长度 0 时等于 kernelSize)、`padding`(1~2, ≥0 或 ≤k/2)、
  `dilation`(**仅支持 1**)、`ceilMode`(bool, 默认 false)。

### 1.2 输出尺寸（★ 三处必须照抄）

```
num = in + 2*p - d*(k-1) - 1
floor: out = floor_div(num, s) + 1
ceil : out = ceil_div(num, s) + 1;  if ((out - 1) * s >= in + p) out -= 1;
```

- **整数除法必须用向下取整语义**（与 numpy `//` 一致）；C++ `/` 是向零截断，`num<0` 的退化配置会算错。
- ceilMode 的最后一条修正**不可漏**，漏了会多出一个输出点。

### 1.3 计算语义（★ 逐字节对齐的关键）

1. 候选位置**仅落在输入 `[0,H) × [0,W)`**；padding 位置等价于 `-inf`，**不参与**最大值。
2. **并列最大值取「kh 主序、kw 次序」中最先出现者**，判据是**严格大于**（`v > best` 才更新）。
3. `argmax` 线性索引 = `ih * W + iw`，**在各自 (n,c) 的 H×W 平面内**（不含 N/C 偏移）。
4. **一像素至少有一个有效候选**：由 ceilMode 的修正可推出 `oh*s - p < H`，故 `-inf` 初值必然被覆盖，
   无需为「整窗皆 padding」做特判。
5. `out` 的值是**输入元素的精确复制**（max 结果），不存在舍入差 ⇒ 应做到与 golden **逐字节相同**。

### 1.4 indices 容器布局（★ 最易错，已实测证实）

```
indices: int8, shape [N, C, kH*kW, maskW]        # 3D 输入时为 [C, kH*kW, maskW]
maskW   = ((Ho*Wo + 15) / 16 + 1) * 2 * 16
整张量字节数 = N*C * kH*kW * maskW               （注意：这与"每 slice 独占一块"不是一回事）
```

- **载荷是跨 slice 连续铺在整个张量最前面的**（实测确认，0 字节差异）：
  `(n, c, oh, ow)` 的 argmax 小端 int32 位于字节偏移
  `((n*C + c) * Ho*Wo + oh*Wo + ow) * 4`；
  即 [0, `N*C*Ho*Wo*4`) 是**一整段连续载荷**，**其余全部字节必须为 0**。
- 因此**不能**按 `slice * indiceBlockSize` 分块写（我曾这么实现，导致 143/153 例 indices 失败）；
  需按 `slice * Ho*Wo * 4` 定位，并把补零当成**一个全局区域** `[N*C*Ho*Wo*4, N*C*kH*kW*maskW)`
  按核切分一次写完（不要每个 slice 各补一次，否则区域重叠冲突）。
- `kH*kW` 与 `maskW` 只是容器外形，**不代表"每行一个 (kh,kw)"**。
- 实测：`kH*kW=4` 时 `maskW ≥ 2*hw+32 > hw`，故容器总够用；但 `kH*kW` 过小（如 =1）会装不下载荷
  ⇒ tiling 里做了硬校验（`payloadBytes > indiceBlockSize` 直接报错），避免越界写坏相邻 slice。
  （注：官方 golden 在 `kernelSize=[1,1]` 上自身也会溢出，该配置不受支持。）
- 比较判据：`out` 容差 `1e-3`，**`indices` 容差 `[0,0]`（必须 bit-exact）**。

---

## 2. 关键 API 约束（已从 CANN 9.1.0 头文件核实，勿凭记忆）

| 事实 | 值 / 结论 | 影响 |
|---|---|---|
| `Compare` | `Compare(dst<U>, src0<T>, src1<T>, CMPMODE, count)`；T 仅 **half/float**（int32 仅 EQ）；mask 为 **1 bit/元素、低位在前** | 值与索引都在 fp32 上比较 |
| `Compare` Level-2 | 需 `count * sizeof(T) % 256 == 0`（fp32 即 **64 元素对齐**） | 向量工作长度按 64 对齐 |
| `SELMODE` | `VSEL_CMPMASK_SPR=0`、`VSEL_TENSOR_SCALAR_MODE=1`、**`VSEL_TENSOR_TENSOR_MODE=2`** | — |
| `CMPMODE` | `LT=0, GT=1, EQ=2, LE=3, GE=4, NE=5` | 用 `GT` |
| `Select` | `dst = selMask ? src0 : src1`；**T 仅 half/float**，int32 无原生写法 | **索引必须以 fp32 承载**（fp16 尾数 11 位无法精确表示 >2048 的线性索引） |
| `Gather` | `Gather(dst<T>, src<T>, srcOffset<uint32_t>, srcBaseOffset, count)`；fp32 支持 | 窗口取数的核心 |
| `Gather.srcBaseOffset` | **单位是字节**，且需满足 `% sizeof(T) == 0` | 每 tap 只换一个标量，**零向量运算** |
| `DataCopyPad` | `blockLen`=字节；**GM 侧 stride=字节，UB 侧 stride=32B 块**；GM 侧地址**无 32B 强制**（现网用法反推） | 可逐行写到任意字节偏移 |
| `DataCopyPad` 尾部 | GM→UB 时**尾部会补写到 32B 边界**，补的内容取自 `mov_pad_val`（`isPad=false` 时为陈旧值） | 必须显式给 `rightPadding`/`paddingValue`，否则 padding 列被写脏 |
| `Add`/`Adds`/`Max` | half/float/int16/**int32**（**uint32 ✗**） | `ow*sW` 偏移向量用标量循环预生成 |
| `Duplicate` | 含 half/bf16/int16/uint16/int32/uint32/float | 填充 -inf / 0 |

**统一计算精度**：kernel 内部一律升 fp32。bf16 无法向量直算（必须升 fp32）；fp16 的索引需要 fp32
才能精确承载。`out` 由 fp32 回写原 dtype 时**无损**（输入值原本就来自该 dtype），故仍可逐字节对齐。

---

## 3. 工程契约（ops-nn experimental 约定）

```
experimental/pooling/max_pool2d_with_mask/
  op_host/CMakeLists.txt                         # add_modules_sources(... OPTYPE max_pool2d_with_mask ACLNNTYPE aclnn)
  op_host/max_pool2d_with_mask_def.cpp           # OpDef：x / out / indices + 5 个 attr（顺序即 aclnn 形参顺序）
  op_host/max_pool2d_with_mask_infershape.cpp    # InferShape + InferDataType
  op_host/max_pool2d_with_mask_tiling.cpp        # Tiling（必须在 op_host 且文件名含 _tiling）
  op_kernel/max_pool2d_with_mask.cpp             # kernel 入口（核函数名 = 目录名）
  op_kernel/max_pool2d_with_mask.h               # kernel 类
  op_kernel/max_pool2d_with_mask_tiling_data.h   # TilingData（host/kernel 共享）
  op_kernel/max_pool2d_with_mask_tiling_key.h    # TilingKey（ASCENDC_TPL_* 模板声明）
```

- `ACLNNTYPE aclnn` = **aclnn 接口由 build 从 OpDef 自动生成**，无需手写 op_api。
  实测生成的签名与内置**完全一致**（含 `bool ceilMode`），前提是 **attr 声明顺序 = kernelSize/stride/padding/dilation/ceilMode**。
- `experimental/pooling/CMakeLists.txt` 会自动发现子目录（只需 `op_host/CMakeLists.txt`），**无需注册**。
- `op_kernel` 无 `CMakeLists.txt`；kernel 由 `add_modules_sources` 写入的 `ASCEND_SUB_CONFIG_PATH` 登记后按类别编译。
- 目录名（snake_case）必须与 kernel 入口名、`opFile.value` 一致；OpDef 名（CamelCase）决定 aclnn 接口名。
- 仅注册 `ascend910b`（A2/A3）；300V Pro 走另一份工程。

## 4. Tiling 设计

- **核间**：按 slice（N*C）切，`slicesPerCore = ceil(totalSlices / coreNum)`，满核优先。
- **核内**：按输出行分块 `rowsPerTile`，本 tile 需输入行 `inRowsPerTile = (rowsPerTile-1)*sH + dH*(kH-1) + 1`。
- **UB 预算**（解 `rowsPerTile`）：`inTile` 每行 `tileWAligned` 个 fp32，另有若干单行 scratch
  （maxVal / idx / tap / tapConst / baseOffF32 / baseOffU32）+ 1 bit/元素的比较掩码 + indices 尾零块
  （8KB）+ 系统预留（8KB+1KB）。逐行递增试探直到放不下。
- ★ **UB 预算口径必须与 kernel 的 `InitBuffer` 逐项一致**，否则 UB 超配 → 运行期 `Synchronize failed`。
  踩过的实例：输出/索引缓冲最初按 `rowsPerTile × outWAligned` 分配（224×224 时各约 165KB），
  而 tiling 只算了输入 tile + 单行 scratch ⇒ 只有大 shape 暴露，小 shape 全过、极难定位。
  现改为**输出/索引按「一个向量块最多打包的 strip 数（J）」留 scratch、逐块算出即搬**，
  预算口径与 `InitBuffer` 一一对应（含 2 字节 dtype 的 raw buffer、J 车道的 idxRow/tap/cand 等）。
- **tile 列布局**：`colOff = AlignUp(pW, alignElems)`，tile 列 = 输入列 + colOff
  ⇒ 真实数据行起点（`row*tileWAligned + colOff`）× elemSize 保持 **32B 对齐**；
  `tileW ≥ max(colOff + W, (Wo-1)*sW + kW + (colOff - pW))`，且 `tileWAligned = AlignUp(tileW, alignElems)`。
- ★ **所有行距/列起点对齐必须按「实际元素字节数」折算到 32B**：`alignElems = 32 / dtypeSize`
  （fp32 → 8、fp16/bf16 → 16）。若一律按 fp32 折算，fp16/bf16 下行起点只有 16B 对齐、
  行距也不是 32B 的整数倍 ⇒ `DataCopyPad` 运行期失败（UB 目的地址有 32B 对齐硬要求）。
- **多行 DMA 的 UB 行距**：`DataCopyPad` 的 UB 侧 stride 以 32B 为单位
  ⇒ `R>1` 时 `colsPerChunk*elemSize` 必须是 32B 整数倍（`R==1` 时行距即整块 64 车道，恒满足）。
- **维度退化**：`N*C==0 || Ho<=0 || Wo<=0` → `BlockDim=1`、kernel 空转，避免 0 长度 buffer。

## 5. Kernel 设计

### 5.1 阶段一（已落地）：正确性优先

- CopyIn：逐输入行 `DataCopyPad` 到 UB（**原始 dtype**，越界行整行 `Duplicate(0)`）；
  计算时按真实坐标 `ih∈[0,H) && iw∈[0,W)` 判界，不做 tile 内补边
  —— 好处是 **tile 只被 MTE2 写、只被标量读**，回避 V→S 同步问题。
- Compute：标量 `GetValue/SetValue`，严格大于更新，记录 `bestIdx = ih*W+iw`。
- CopyOut：**逐行** `DataCopyPad`（UB 行距 `outWAligned` ≠ GM 行距 `Wo`，不能整块搬）；
  索引张量 `ReinterpretCast<int8_t>` 后按字节写，行起始偏移 = `oh*Wo*4`。
- 尾零：容器 `[payloadBytes, indiceBlockSize)` 用 `Duplicate` 出的 0 缓冲分块 `DataCopyPad` 写入
  （必须显式补 0，golden 用 `np.zeros` 建容器并逐字节比对）。
- **arch22 上标量写 GM 不可靠**（两任务独立实测），故所有 GM 写出都走 DMA/MTE3。

### 5.2 阶段二（性能）：向量化 + 车道打包

**每个 64 车道向量块覆盖 (slice q, 输出行 r, 输出列 owc) 三元组**（`lane i -> q = i/(R*C)`、`r = (i/C)%R`、
`owc = i%C`）。host 据此决定 (slicesPerTile, rowsPerChunk=R, colsPerChunk=C)：

| 条件 | slicesPerTile | R | C | 效果 |
|---|---|---|---|---|
| 每 slice 输出 ≤64 且 wOut<64 | 尽量大（`64/(Ho*Wo)`，受 UB 限） | Ho | Wo | **整 slice 进一个块**，多个 slice 一起打包 |
| 其他 | 1 | `64/C`（受 Ho/UB 限） | min(Wo,64) | 行方向打包；wOut≥64 时退化为「一行多列块」 |

于是 **Gather 的字节偏移向量 `offU`** 与**每车道索引常数 `cand`** 都只与车道有关、可一次性预生成：

```
offU[i] = (q*tileSliceStride*rowStride + r*sH*rowStride + owc*sW) * 4      # 字节
cand[i] = r*sH*wIn + owc*sW
```

每块的标量基址只承担 r0/ow0 的变化；每 tap 的内层是**四条 Level-0 指令、零临时缓冲、零屏障**：

```
Gather(tap, tile, offU, baseBytes, 64)                      # L2（内部只设一次满掩码）
Adds(candTmp, cand, kh*wIn + kw, 64)                        # 该 tap 的候选索引（车道基址 + 窗口偏移）
Compare<float,false>(tap, maxRow, CMPMODE::LE, MASK_PLACEHOLDER, rp)   # 无 dst ⇒ 结果写 CMPMASK SPR
Max<float,false>(maxRow, maxRow, tap, MASK_PLACEHOLDER, 1, rp)         # 不碰掩码 SPR
Select<float, VSEL_CMPMASK_SPR>(idxRow, idxRow, candTmp, 1, rp)        # 直接读 CMPMASK SPR
...
Adds(idxRow, idxRow, (ohBase*sH - pH)*W - pW + ow0*sW, 64)              # 每块一次，ohBase=oh0+r0
```

- `Compare` 的无 dst 重载（`vcmp_le`）把 64 位比较结果写进 **CMPMASK SPR**；`Select` 的
  `VSEL_CMPMASK_SPR` 模式（`SelectCal` 路径）**直接读该 SPR**，不建临时缓冲、不打
  `PipeBarrier`（对比 L2 的 `Select` 要 temp buffer + `set_cmpmask` + 2 次 `PipeBarrier<PIPE_V>`）。
  二者紧邻配对，SPR 生命周期不超过一个 tap。
- `isSetMask=false` + Init 里一次性 `ResetMask(); SetMaskNorm();`：工作宽度恰好一个 repeat（64 元素），
  掩码恒为满掩码，内层不再有任何掩码模式切换（L2 的每条算子都会 `set_mask_count/set_mask_norm`）。
- **Init 只预生成 64 个值**（`offU` + `cand`，各 64 次标量写）。候选索引的每 tap 偏移由内层一条
  `Adds` 现场加。★ 这一点很关键：曾改成「预生成 kH·kW 张 64 车道常量表」以省掉每 tap 的那条
  `Adds`，结果初始化的 `kH·kW×64` 次标量写（k=7 时 3136 次）**成了小 shape 的固定开销**，
  使中位 ratio 从 0.54 恶化到 1.28；改回「64 个标量 + 每 tap 一条 Adds」后，
  `2 8 32 32 k7s2p3 f32` 从 0.081→0.022ms（3.7×），`1 1 4 4` 从 0.0196→0.0101ms。
- ★ **不要在 Init 里用「标量写 + 向量 `Adds` 生成表」**：实测该组合会让 kernel 挂死
  （`bench` 超时无输出）。
- 块长**固定 64**（fp32 恰为一个 repeat）。
- 越界车道（不满的 row group / slice group）把 q/r/owc **夹到合法上界**，保证取数落在
  「已分配且恒为 -inf」的区域；这些车道不会被搬出。为此 tile 尾部多留 `(R-1)*sH` 行。
- **DMA 按块合并**：打包后一个块覆盖的输出在 GM 上连续，每块只需 out + indices 各一次 `DataCopyPad`。

### 5.3 阶段二落地过程中修掉/核实的坑（均已在真机复现→定位→验证）

1. **★ `DataCopyPad`（GM→UB）会把尾部写到 32B 边界**，多余字节用 `mov_pad_val` 寄存器填充。
   若 `isPad=false`，该寄存器是**陈旧值** ⇒ 右侧 padding 列被写脏 ⇒ `ow=Wo-1` 的越界 tap
   读到脏值、最大值偏大。真机表现极具误导性：只有「窗口最右列恰好会读 tile 列 colOff+wIn」
   的形状才出错（W=5/7/9 错，W=4/6 对；fp16 走 Cast 路径不受影响）。
   **修法**：显式 `DataCopyPadParams{true, 0, 补到 32B 边界的元素数, 0xFF7FFFFF}`。
2. **`Gather` 的 `srcOffset` 与 `srcBaseOffset` 单位都是字节**（从工程内用法反推确认：
   `ArithProgression(..., sizeof(float)*2, ...)`、`Gather(..., srcBaseOffset=4, ...)`）。
3. **`Compare` Level-2 的 count 必须让 `count*sizeof(T) % 256 == 0`**（fp32 即 64 元素对齐）；
   非对齐的尾数会被 `sumRepeat = count*sizeof(T)/256` 整除**静默丢弃**。
   `Max/Add/Cast/Duplicate/Select` L2 走 counter 模式，count 任意。
4. 内置 aclnn 的 executor 是**一次性**的（`CommonOpExecutorRun` 在 Run 末尾销毁它），
   基准 harness 每次迭代都必须重走两段式接口，否则第二次 Run 会 double free。
5. **★ 车道打包的「标量基址」必须含组偏移 r0**。`woffAll` 的车道部分只含**组内**行号
   `r ∈ [0,R)`，组偏移 `r0 = rg*R` 只能由块级标量补上。曾误写成 `oh0`（漏了 r0），
   真机表现为极具误导性的组合：**`out` 逐字节全对、`indices` 从「第二个 row group 起」全错**
   （224 用例错在第 2 行、56x56 错在第 2 行、32x32 错在第 4 行 —— 恰好都是一个 R 组的长度）。
   因为最大值走 `maxRow`，与索引路径无关，所以只有索引错。
   ★ 教训：`out_diff==0 && idx_diff>0` 且错误边界恰好等于一个 group 长度时，先查标量基址，
   不要怀疑向量算子 —— 当时我误判成 CMPMASK SPR 的时序竞态，白跑了两轮编译。
6. **内层掩码必须显式管理**：工作宽度固定 64（fp32 一个 repeat），Init 里 `ResetMask(); SetMaskNorm();`
   之后，内层所有 L0 调用都用 `isSetMask=false` 并传 `MASK_PLACEHOLDER`，掩码 SPR 全程不动。

### 5.4 实测结论与后续方向

1. **L0 + CMPMASK SPR 路径是正确且最快的**（`vcmp` 无 dst 写 SPR，`vsel` 的
   `VSEL_CMPMASK_SPR` 直接读 SPR）：快速回归 27/27 全对，且相对「L2 掩码张量」版本
   平均快 **1.25–1.3×**（`1 64 112 112 k3s2p1 f32` 0.119→0.082ms、
   `1 3 224 224 k3s1p1 f16` 0.139→0.096ms、`8 64 56 56 k3s2p1 f32` 0.191→0.134ms）。
   省下的是 L2 `Select` 内部的 temp buffer + `set_cmpmask` + 2 次 `PipeBarrier<PIPE_V>`，
   以及每 tap 的掩码张量读写。
2. **每算子成本实测**：把 `Select`/`Gather` 分别替成 `Maxs` 只省 18% / 6%，说明成本是
   摊在所有 L2 包装上的（L2 每条都会 `set_mask_count/set_mask_norm` + 2 次 `set_vector_mask`）。
   改成 L0 后这条开销被消掉，这是本阶段主要收益来源。
3. **剩余可做（按预期收益排序）**：
   - **提高 ILP**：内层仍是串行依赖链（每 tap 的 Compare/Max 依赖上一 tap 的 maxRow），
     可拆成「先 Max 树求最大值、再独立 Compare(EQ) + 反向 Select 树求索引」，把链长从 3×kH·kW 降到 ~2×log。
   - 合并 `Gather`+`Max` 之外的冗余（大 kH·kW 时可用「滑动窗口 max」结构把 kH·kW 次 Max 降到 kH+kW−2 次）。
   - 大 spatial 且 stride==1 时可退化为连续切片取数，省掉 `Gather`。

### 5.5 阶段三（2026-09-19）：真机减项定位 → 三条结构性修复

阶段二结束后仍有 **64/153 例 >1.5×**（最差 16.4×），且共同特征是「小 spatial × 大 batch ×
少 tap」。先做一个**编译期探针**（`MPWM_PROF`，逐项把某一段代码编译掉）在真机上定位，
再动手改 —— 结论与阶段二的猜测**不同**：

| 减项 | `16 64 112 112 k2s2p0 f16` | `8 64 56 56 k3s2p1 f16` | `1 3 224 224 k3s2p1 f32` |
|---|---|---|---|
| P0 基线 | 1.157 ms | 0.247 ms | 0.0313 ms |
| P1 去掉「逐行输入 DMA + 逐行 Cast」 | 0.655 (**−43%**) | 0.132 (**−46%**) | 0.0288 (−8%) |
| P2 去掉 out/idx 的 DMA **传输** | 1.139 (−1.5%) | 0.247 (0%) | 0.0303 (−3%) |
| P3 去掉内层 tap 循环 | 0.943 (−18%) | 0.173 (−30%) | 0.0157 (−50%) |
| P4 去掉 indices 尾零 | 1.167 (+1%) | 0.246 (0%) | 0.0292 (−7%) |

`msprof` 的 AIV 分管道 busy 占比（`16 64 112 112 k2s2p0 f16`，Task 1290us）：
`scalar 0.731 > mte2 0.43 > vec 0.368 > mte3 0.243`，`aiv_icache_miss_rate = 0`。

**结论（与阶段二的猜测不同）**：主导成本是**每个 64 车道块的固定开销**（TQue 的 set/wait 旗标 +
`Duplicate`/`Cast` 的 L2 包装（每次都会设掩码）+ 标量基址），而**不是**向量算力、也不是 DMA 传输量
（P2 直接证明 DMA 传输 ≈ 0）。因此「减少块数 / 减少每块的固定动作」才是有收益的方向。

三条修复（全部在真机 27 例快速回归上逐字节验证）：

1. **输入路径「一次搬完」**（取代逐行 DMA + 逐行 Cast + 每行 4 次队列往返）：
   行距 32B 对齐后，一条 `DataCopyPad(blockCount=行数, blockLen=wIn*elemSize, dstStride=…)`
   覆盖全部可行行；2 字节 dtype 再补**一条整块 `Cast`** 升 fp32。
   为此把 raw buffer 的行距与 fp32 tile 的**元素行距完全对齐**，于是整块 Cast 就是逐元素一一对应。
   - 越界行改为**每 tile 重置极小值**：tile/raw 缓冲跨 tile 复用，上一个 tile 的 DMA 可能写过这些行
     （它只出现在边界 tile，内部 tile 退化 0 次 Duplicate）。
   - `DataCopyPad` 的尾部 32B 补写用 `paddingValue` 显式填极小值：fp32 用 `0xFF7FFFFF`，
     half 用 `-inf(0xFC00)`，bf16 用 `-FLT_MAX` 的 bf16 位型 `0xFF7F`（与 fp32 路径同语义）。
   - 输入不再走 TQue，**必须显式打 `MTE2→V` 旗标**（原来靠 `inQueue.DeQue` 提供这个同步）。
2. **J 个 row strip 打包成「超级块」**（`stripsPerBlock`，非整 slice 路径）：
   一个块覆盖 J 个 strip（车道 = `(j, r, owc)`），J 个 chunk 共享 1 次队列往返、2 次多行 DMA、
   1 条共享候选索引 `Adds`，并把 J 条独立依赖链交错（`Gather` 全部先发）以提高 ILP。
   多行 DMA 要求 UB 侧**行距是 32B 的整数倍** ⇒ `colsPerChunk` 只能取 `{8,16,32,64}` 中
   elemSize 允许的最大者（fp32 `{8,16,32,64}`、fp16/bf16 `{16,32,64}`）；
   直接退化成 `R=1` 会让车道利用率腰斩（实测 `4 16 56 56 k3s2p1 f32` 变慢 35%）。
3. **整 slice 进块且 hw>32 时同样打包**（`ComputeSliceStrips`）：一个 chunk 一个 slice（有效车道 = `hw`），
   J 个 chunk 一块，相邻 slice 的输出在 GM 上连续 ⇒ `dstStride=0` 的多行 DMA。
   同时 `offU` 的 `lanesPerSlice` 在这种布局下必须取满一个 chunk（`q` 恒为 0），
   否则末尾 chunk 的越界车道会再加一份 slice 偏移而越出 tile。

4. **核间负载均衡（带约束）**：旧规则「tile 数 ≈ 核数」在 slice 少时大量空核
   （64 slice × 1 行块 = 64 块 → 只能在 32 核上跑 2 轮）。
   改为在 `rowsPerTile ≥ stripsPerBlock*R`（**保证超级块装满**）的前提下，搜「行块数」使
   `totalSlices × numRowBlocks` 尽量整除 `coreNum`。
   ★ 这个约束是实测逼出来的：不加约束地切小 tile 会把超级块的摊薄效果弄丢，
   `8 64 56 56 k3s2p1 f32` 会从 0.120 恶化到 0.169ms、`8 128 56 56 k3s2p1 f16` 从 0.228 到 0.343ms；
   加上约束后这两例回到原值，同时 `1 64 112 112 k3s2p1 f32` 0.0722→0.0630ms、
   `1 3 224 224 k3s1p1 f16` 0.0774→0.0509ms。

### 5.7 阶段四（2026-09-19）：平面布局 + 直取向量操作数（免 Gather）

阶段三验收后 msprof 复测最慢 4 例：**瓶颈已从标量翻转为向量发射**（vec busy 68–80%，
scalar 40–55%，mte 合计 <35%）—— 阶段三把固定开销砍完后，剩下的就是每 tap 的指令条数。
当前内层每 tap 每 strip 4 条（Gather/Compare/Max/Select），9-tap 形状每 64 输出 36+ 条。

**关键真机发现（判定性实验）**：直接把 tile 切片作为向量操作数（免 Gather）时，
`kW=1&&pW=0` 正常、`kW≥2 或 pW≥1` 一律挂死 ⇒ **v220 向量指令操作数起点必须 32B 对齐**
（单平面布局里 tap 切片起点带 (kw−pW) 偏移）。注：k=1 的「挂」是 tiling 合法性拒绝
（容器装不下 payload，该配置官方 golden 也不支持），与对齐无关。

**修法 —— kw 平面化 tile**：`sW==1` 且 `perSliceOut>64` 且 `wOut≥48` 时，tile 改为
kW 个平面，`plane_kw[c] = input[c − pW + kw]`。平面间移位用 `DataCopyPad leftPadding`
实现（语义从内置 max_pool3d_with_argmax_v2 用法核实：数据落 dst+leftPadding，
块足迹 = CeilDiv(leftPad+blockLen+rightPad, 8)×32B，dst 基址保持 32B 对齐；
kw<pW 时 src=0 + leftPadding=pW−kw，kw≥pW 时 src 右移 kw−pW、leftPad=0）。
于是每个 tap 的取数切片 = plane_kw 内一段**连续且 32B 对齐**的 64 元素：

```
每 tap（覆盖 J 条 strip，R=1、C=64）：
  Adds(candTmpJ, candJ, kh*W+kw, J*64)                  # 1 条共享候选索引
  for j: Compare(plane_kw 切片_j, maxRow_j, LE)→SPR     # J 条（无 Gather！）
         Select(idxRow_j, idxRow_j, candTmpJ_j, SPR)
  Max(maxRow, maxRow, plane_kw 切片_0, MASK, J, rpJ)    # 1 条 J-repeat（src1RepStride=sH*rowStride）
= 2J+1 条（原 4J+1），且消掉了最贵的随机访问 Gather 指令
```

要点与坑：
- **wOut<64 时 C 强制 64**：越界车道读 −inf、结果不搬出（列利用率 wOut/64），
  省 Gather + 索引对减半仍净赚；wOut<48 浪费过大，退回 gather 路径。
- 越界列读的是「Init −inf 打底 + 每tile边界行重置」后的安全区域 ⇒ tileW = AlignUp(wOut,64)+kW−1。
- **Compare/Select 必须用「旧 maxRow」**：先跑完全部 strip 的 Compare→Select 对，
  最后一条 J-repeat Max 统一更新值（顺序错了索引全错）。
- **ccec 不支持 lambda**（device 侧编译报 no matching function for call to object of type lambda）。
- 4b：candAll（kH·kW 张 64 车道候选索引表）在 Process 开头一次性向量 Adds 生成
  （阶段二「Init 标量写+向量 Adds 生成表挂死」的根因是缺 S→V 同步；放在 WaitFlag(S_V) 后安全），
  整 slice / A2 路径内层每 tap 省一条 Adds（5→4 条）。
- 快速回归扩到 35 例（新增 8 个 sW==1 大 spatial + 2 个非对齐尺寸 61×63/70×71）。

### 5.8 阶段四实测（2026-09-19，A2/910B3）

快速回归 35/35 载荷逐字节一致、0 挂死。s1 大 spatial 案例：
`1 64 224 224 k3s1p1 f32` 6.0→2.56×、`4 64 112 112 k3s1p1 f16` 5.8→2.99×、
`1 3 224 224 k3s1p1 f16` 0.21→0.093×、`8 64 56 56 k3s1p1 f32` 3.9→2.62×。
（153 例全量数据见「性能现状」表阶段四行。）

## 6. 验证链路

| 环节 | 方式 | 结果 |
|---|---|---|
| 语义 | `work/proto/verify_golden.py`：两套独立 numpy 实现 × 自带 golden | **4342 条断言逐字节一致** |
| 精度（真机） | 官方 AscendOpTest，自带 153 例（`--op-type custom --op-path <vendor>/op_api`） | **153/153 全过**（306/306 输出张量比较 pass），2026-09-18 于 Atlas 800T A2 / 910B3 / CANN 9.1.0。**阶段一（标量）与阶段二（向量化+车道打包）各自独立跑通一次** |
| 精度（对拍内置） | C++ harness（`work/harness/bench.cpp`）同进程 dlopen 内置 `libopapi_nn.so` 与本算子 `libcust_opapi.so`，同输入逐字节比较 | 153 例全部 **PAYLOAD-BIT-EXACT**（`out` 与 indices 载荷逐字节相同；仅容器尾零区不同——内置不写、本算子显式补 0，golden 要求 0） |
| 性能（对拍内置） | 同上 harness，`aclrtEvent` 量 device 时间，153 例逐例 A/B | 见下表 |
| 性能（交叉校验） | `msprof --ai-core=on` 读 `op_summary` 的真实 kernel 时长 | 与 harness 的相对结论一致（harness 绝对值可信） |

### 性能现状（2026-09-18，A2/910B3，40 AIV）

内置 `aclnnMaxPool2dWithMask` 在 910B 上落到 **`MaxPool3DWithArgmaxV2NcdhwAiC…`**（msprof 实测确认）——
即内置走的是 3D argmax 池化的 TBE/自研 kernel，不是在跑我们的 2D 实现（无 op type 冲突）。

| 版本 | 总 B（内置） | 总 C（本算子） | ratio 中位数 | ratio ≤1 的例数 | 最差 ratio |
|---|---|---|---|---|---|
| 阶段一（标量） | 12.75 ms | 83.76 ms | 0.729 | 80/153 | 57.5× |
| 阶段二（向量化，未打包） | 12.75 ms | 47.91 ms | 0.537 | 83/153 | 17.6× |
| 阶段二（车道打包 + L0/SPR） | 12.72 ms | 42.41 ms | 0.497 | 84/153 | 16.4× |
| 阶段三（输入一次搬完 + 超级块 + 带约束负载均衡） | 12.71 ms | 26.95 ms | 0.356 | 91/153 | 6.0× |
| **阶段四（kw 平面化 + 直取向量操作数免 Gather）** | **12.75 ms** | **23.08 ms** | **0.356** | **89/153** | **6.0×** |

阶段四后最慢的一批集中在 **sW==2 的 f16/bf16 形状**（`16 64 112 112 k2s2` 6.0×、
`16 64 112 112 k3s2` 5.0×、`6 128 56 56 k7s2` 4.9×、`2 64 224 224 k2s2` 4.7×）：
sW=2 时窗口取数跨步访问，平面化/直取无法应用（向量块内元素步长固定连续），
仍需逐 tap Gather，且 f16/bf16 的 Cast 开销照付。这些案例的进一步压缩需要
「sW=2 专用平面布局（每平面存隔列元素，DMA blockLen=1 elem）」或「fp16 原生 128 宽计算」，
均已在纸面分析过但收益/复杂度比不高，留作后续。

⚠ 仍有 **62/153 例 >1.5×**。总时长四阶段累计 **83.76 → 23.08 ms（−72%）**，
最差 ratio **57.5× → 6.0×**，且全程保持与内置算子载荷逐字节一致、官方 153/153 全过。

`msprof` 单例核对（kernel 平均时长）：

| shape | 内置 kernel | 本算子 kernel | 比值 |
|---|---|---|---|
| `1 3 224 224 3 3 2 2 1 1 f32` | 67.7 us | **36.8 us** | **0.54 ✓** |
| `8 64 56 56 3 3 2 2 1 1 f16` | 35.5 us | 271.3 us | 7.65 ✗ |
| `16 64 112 112 2 2 2 2 f16` | 71.0 us | 1274.9 us | 17.96 ✗ |

**结论：语义/精度已完全达标（153/153、载荷逐字节），性能在「大 spatial + 多 tap」的形状上已经反超内置
（k7/k3-on-224 系列 ratio 0.27–0.58），但在「小 spatial、大 batch」的形状上仍差 5–18×。**

### 阶段一落地过程中修掉的 4 个真机 bug（均已在真机复现→定位→验证）

1. **indices 载荷偏移**：载荷是跨 slice 连续铺开，不是每 slice 从块首写（曾致 143/153 例 indices 失败）。
2. **行距对齐按错元素字节数**：一律按 fp32 折算 ⇒ fp16/bf16 行起点仅 16B 对齐 ⇒ 运行期
   `Synchronize failed`（小 W 的 fp16/bf16 全挂、W=16 的却过）。改为按实际 elemSize 折算。
3. **UB 超配**：输出/索引缓冲按 `rowsPerTile` 行分配但预算未计入 ⇒ 只有 224×224 暴露。
   改为输出/索引各只留一行、逐行算出即搬。
4. **同版本号算子包重装被安装器跳过** ⇒ 改了代码/重建/重装后仍在跑旧 kernel。
   每次安装前 `rm -rf $ASCEND_HOME_PATH/opp/vendors/customize_nn`，并核对 kernel 二进制时间戳。

**AscendOpTest 使用坑（已实测）**：`aclnn-gen/parsing/op_parser.py` 读取的属性键是 **`attr`**，
而任务书给的 op json 只有 `attr_desc` ⇒ 属性被静默丢弃、生成的调用缺参、编译报
`cannot convert aclTensor* to const aclIntArray*`。**需补 `attr` 数组**（见
`work/optest/max_pool2d_with_mask_op_with_attr.json`）。`ceilMode` 类型写 `bool`（非 `scalar_bool`）
才能与自动生成的 `bool ceilMode` 形参匹配。