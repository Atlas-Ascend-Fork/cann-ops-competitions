# Transpose — 6666

本目录是队伍 **6666** 的 S9 `Transpose` 提交，使用 Ascend C 实现按 `dims`
重排张量轴的操作，语义对应 `torch.permute`。公开 OpType 为 `Transpose`，Kernel
入口为 `transpose`，不需要额外 workspace。

## 接口与支持范围

| 类别 | 名称 | 说明 |
| --- | --- | --- |
| 输入 | `inputs` | 连续 ND Tensor，支持 float32、float16、int32、int8 |
| 属性 | `dims` | 必需的轴排列；长度等于输入 rank，轴值互不重复且位于 `[0, rank)` |
| 输出 | `output` | dtype 与输入一致；`output.shape[i] = inputs.shape[dims[i]]` |

支持 rank 1–8 和正长度维度，包含非 32 字节对齐的尾块。不支持空 Tensor、负轴、
rank-0 或 rank 大于 8 的输入。

## 构建

环境为 Linux aarch64、Ascend 910B、CANN 社区版 8.5.0，需安装配套 Toolkit 和
Ascend 910B ops。先加载实际安装路径下的 `set_env.sh`，再在本目录运行：

```bash
bash build.sh
```

脚本从 `ASCEND_HOME_PATH` 获取 CANN 路径，可通过 `ASCEND_PYTHON_EXECUTABLE`
和 `BUILD_JOBS` 指定 Python 解释器与编译并行度。构建结果位于 `build_out/`，生成
同时包含源码和二进制的 `custom_opp_*.run` 安装包。`CMakePresets.json` 提供同一套
默认设置，支持 preset 的 CMake 也可直接使用。

## 实现与优化

- Host 校验轴排列和输入维度，推导输出 Shape/dtype，并根据实际 AIV 核数、UB 容量
  和张量 stride 选择 TilingKey、tile 大小与 BlockDim；BlockDim 不超过任务数。
- 对可合并的连续轴折叠为 run，以 DMA 搬运代替逐元素寻址；矩阵和批量矩阵采用
  UB 分块转置、整行流式读取及 `TransDataTo5HD` 等路径。
- FP16 对齐矩阵采用 16 行 slab 转置；4 字节类型依据 UB 容量选择整行转置、
  packed gather 或分块路径。一般轴排列使用按元素宽度区分的 gather 路径。
- 使用队列双缓冲或显式流水同步组织搬入、重排和写出，使用 `DataCopyPad` 处理
  非对齐尾块；只写回有效行列，保持元素位模式不变。

`op_host/transpose_tiling_rules.h` 是策略选择和多核划分的实现入口；
`op_host/transpose_tiling.h` 定义 Host TilingData，
`op_kernel/transpose_tiling.h` 定义对应的 Device 数据布局；
`op_kernel/transpose.cpp` 实现各 Kernel 路径。
