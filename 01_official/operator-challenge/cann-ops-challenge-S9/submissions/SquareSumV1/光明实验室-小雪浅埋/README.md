# SquareSumV1

沿指定轴计算元素平方和，等价于 `torch.sum(x * x, dim=axis, keepdim=keep_dims)`。

- 团队：光明实验室-小雪浅埋
- 赛题：算子挑战赛（S9赛季）· SquareSumV1

## 接口契约

| 项 | 说明 |
| --- | --- |
| 输入 `input` | ND 张量，rank 1–8，各维长度必须为正 |
| 输出 `output` | 与输入同 dtype |
| 属性 `axis` | `listInt`，必填。待归约轴的列表，元素互不重复；负数轴按从后往前归一化（`-1` 为最后一维）。空列表表示归约全部轴，Host 侧形状推导与 Tiling 使用同一套解析逻辑 |
| 属性 `keep_dims` | `bool`，可选，默认 `false`。为 `false` 时移除被归约维，为 `true` 时用长度为 1 的维替换 |
| 支持 dtype | `float16`、`bfloat16`、`float32` |

FP32 直接累加。FP16/BF16 路径先转 FP32 做平方与归约，再转回输出 dtype。

非法 rank、非正维度、越界或重复的 axis、Host 侧不支持的 dtype 均导致形状推导 / Tiling 失败。

## Host 端

`op_host/square_sum_v1.cpp` 提供 GE 接口与 Tiling：

- **InferShape**（`DeriveShape`）：按归约轴掩码重算输出形状，逐维保留非归约维、丢弃或替换归约维。
- **InferDataType**（`DeriveType`）：输出类型与输入一致。
- **Tiling**（`BuildTiling`）：把连续的归约/非归约维合并成连续段，依据段的排列选出拓扑，再据此决定分片规模、tile 行数、多核数、workspace 大小与 Tiling Key。

拓扑共 5 类：

| 拓扑 | 形状模式 | 说明 |
| --- | --- | --- |
| AR | 归约维在最内 | 每行独立归约，行间互不干扰 |
| ARA | 归约-非归约-归约 | 两侧归约共享中间维，需要分相累加 |
| RAR | 归约-非归约-归约（外侧成对） | 外侧两次归约，中间维流式穿过 |
| RA | 归约维在最外 | 归约量巨大且输入维很小，跨核归并 |
| 通用 | 其它任意轴组合 | 正确性优先的标量索引回退路径 |

多核调度方面，Host 按拓扑的并行维度估算核数，上限受 AIV 核数与「每核至少产出一个 32B 对齐块」的约束；归约量小于单个可摊销 DMA tile 时退化为单核，避免多核启动与尾波开销。RA 拓扑需要跨核归并，Host 额外申请 `blockDim × 对齐后中间维` 的 FP32 workspace。

## Kernel 端

`op_kernel/square_sum_v1.cpp` 通过模板参数 `schMode` 派发三种 Tiling Key：

| Tiling Key | 核类型 | 用途 |
| --- | --- | --- |
| `SQUARE_SUM_KEY_NORMAL` (0) | AIV_ONLY | AR / ARA / RAR / 通用路径 |
| `SQUARE_SUM_KEY_SYNC` (1) | MIX_AIV_1_0 | RA 路径：多核部分和写入 workspace，再由 0 号核归并 |
| `SQUARE_SUM_KEY_COMPACT_AR_DIRECT` (2) | AIV_ONLY | BF16 紧凑 AR 直通路径 |

### 优化策略

1. **紧凑行布局**：对 `inner` 不是 8 整数倍的场景，按 32B 对齐粒度重新打包行，消除 padding 带来的向量算力浪费。紧凑 AR / RAR / ARA / RA 各有独立的打包与还原路径。
2. **紧凑 RA 的分相累加**：把归约行分成固定行数的相位组，用多路累加器并行推进，末段再归并成一条。相位组数受 FP32 舍入预算约束（一次乘法舍入加 n 层 FP32 加法，按相对误差预算的 2/3 留余量），超限时启用分段 + checkpoint 落盘，避免累加深度超限。
3. **双缓冲流水**：`inputBufferNum = 2` 时，下一块的 MTE2 搬运与当前块的计算重叠；短任务保留单槽，避免第二个队列的启动成本。
4. **L2 绕过**：输入工作集超过 L2 且为流式访问时，对输入 GlobalTensor 下发 `CACHE_MODE_DISABLE`，避免无收益的缓存置换。
5. **Wave / Stagger 调度**：RA 路径在核数充足时按 tile 轮转分核；AR/ARA 路径对各核的起始 tile 做错峰偏移，让多核的搬运与归约不在同一时刻争抢带宽。
6. **跨核归并**：RA 路径各核把 FP32 部分和写进 workspace 的独立槽位，`SyncAll` 后由 0 号核合并输出。

所有分片规模都受 UB 容量、DMA 对齐、向量 repeat 上限和累加深度四重门禁约束；任一门禁不满足时回退到对应的通用路径，保证正确性优先。

## 构建

### 方式一：随目录直接构建（推荐）

准备 CANN 8.5.0 社区版与 Ascend 910B 工具链：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
bash build.sh
```

`build.sh` 依次执行 `cmake --preset=default` 配置、`binary` 目标编译与 `package` 目标打包，产物输出到 `build_out/`。`CMakePresets.json` 中的 `ASCEND_CANN_PACKAGE_PATH` 指向 CANN 安装目录，`ASCEND_COMPUTE_UNIT` 为 `ascend910b`，与 Host 侧 `AddConfig("ascend910b")` 一致。

### 方式二：msopgen 生成构建工程

`op.json` 描述了算子接口，可用于生成 CANN 算子开发工程：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export SOC_VERSION=Ascend910B1
project_dir="$(mktemp -d)/custom_op"
msopgen gen -i op.json -f aclnn -c "ai_core-${SOC_VERSION}" -lan cpp -out "$project_dir"
rm -rf "$project_dir/op_host" "$project_dir/op_kernel"
cp -R op_host op_kernel "$project_dir/"
(cd "$project_dir" && bash build.sh)
```

两种方式下 `op_host/CMakeLists.txt` 与 `op_kernel/CMakeLists.txt` 都保持原样：Host 侧依赖 `opbuild` 生成算子注册代码，Kernel 侧依赖 `add_kernels_compile` 编译 Ascend C 源码。

## 目录结构

```
.
├── README.md
├── build.sh                      # 编译脚本
├── CMakeLists.txt                # 顶层 CMake
├── CMakePresets.json             # CMake 预设（CANN 8.5.0 / ascend910b）
├── op.json                       # 算子接口描述，供 msopgen 使用
├── op_host/
│   ├── CMakeLists.txt
│   ├── square_sum_v1.cpp         # InferShape / InferDataType / Tiling / 算子注册
│   └── square_sum_v1_tiling.h    # TilingData 结构
└── op_kernel/
    ├── CMakeLists.txt
    ├── square_sum_v1.cpp         # Kernel 入口与各拓扑执行路径
    └── square_sum_v1_tiling_key.h # Tiling Key 定义
```

## 验证状态

- **编译**：已在 CANN 环境完成编译并生成自定义算子安装包（产物 `custom_opp_euleros_aarch64.run`，2026-09-12）。按合仓规范，本次提交只含源码，不附二进制产物。
- **精度 / 性能**：本次提交未随附设备实测数据，仓库 CI 的 `compile` 流水线负责构建校验。
