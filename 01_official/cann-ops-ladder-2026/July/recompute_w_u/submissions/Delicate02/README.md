# RecomputeWu Ascend C 高性能实现

## 提交信息

- 团队名称：Delicate02
- 联系方式：[GitCode / Delicate02](https://gitcode.com/Delicate02)
- 目标硬件：Ascend 910B3
- 验证版本：CANN 8.5.0
- 性能核心来源归档：`RecomputeWu_submission_119991.zip`
- 归档 SHA256：`6f490580e6747309b4c8df7b6631b17dab6621a479ff5dd4fffe47ba159a64a3`

## 算子功能

本实现完成分块 GDN 注意力中的 `recompute_w_u` 计算：

```text
u = (A * beta) @ v
w = (A * beta * exp(g)) @ k
```

其中 `chunk_offsets` 在运行时给出各分块的起止位置；实现支持题目规定的动态 `T`、`H`、`Hg`、`D` 和非满尾块，并按照 GQA 的 `h / (H / Hg)` 关系复用 Key 头。

## 实现方案与优化

- 使用 MIX Kernel：AIV 负责 `A/beta/g` 的向量预处理、BF16 转换和右矩阵准备，AIC 负责矩阵乘，形成 Vector/Cube 流水。
- 根据运行时 shape 选择通用 tiling 路径，在 Matmul 高阶 API 与手写 `Mmad + Fixpipe` 路径之间分派；所有条件均基于维度、任务波次和片上存储容量，不依赖测试点编号或固定完整 shape。
- 按 head/chunk 合并任务，复用 GQA 共享的 K 数据，并通过多槽 workspace 缓冲减少串行等待。
- 对适合的维度采用 K/V 直读、D128 head batch、group pipeline、packed wide 和 staged output 等通用路径，降低 GM 往返和控制开销。
- 使用运行时 `chunk_offsets` 处理跨序列分块和非满尾块，避免假设所有 chunk 均为 64 个 token。

## 目录结构

```text
.
├── README.md
└── code
    ├── op_host
    ├── op_kernel
    ├── examples
    ├── CMakeLists.txt
    └── build.sh
```

## 构建与运行

先加载已安装 CANN 的环境变量，然后执行：

```bash
cd code
bash build.sh -j8
```

在有 Ascend NPU 的环境运行 ACLNN 示例：

```bash
cd code
bash build.sh -e
```

## 正确性与性能

本提交的设备侧矩阵乘、调度、同步和数据搬运路径来自历史最佳性能归档；为满足开源仓 CI，仅对 host 注册代码做了等价重构、增加了 host 参数防御检查、为设备侧 `CeilDivU32` 增加了非法零分母保护，并移除了非必需的通用 UT 脚手架。合法 tiling 下所有分母原本均严格非零，因此这些整理不改变算子行为。CANNJudge 提交 `119991` 为 `22 / 22 Pass`；相同归档源码重投为提交 `121028` 后仍为 `22 / 22 Pass`，并在 2026-07-31 的榜单快照中排名第 1。

当前提交已在 Ascend 910B3、CANN 8.5.0 环境完成 `binary`、`package` 和 `install` 构建。

提交 `119991` 的设备侧 Task Duration 如下（单位：μs）：

| 测试点 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 耗时 | 76.12 | 65.72 | 153.31 | 97.33 | 137.90 | 99.12 | 285.29 | 203.76 | 43.28 | 34.68 | 56.52 |

| 测试点 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 耗时 | 43.98 | 106.13 | 78.94 | 214.55 | 141.28 | 18.00 | 20.91 | 762.40 | 1407.84 | 1596.83 | 2884.56 |

性能数据来自 CANNJudge 对 kernel 的设备侧计时；动态榜单名次和分数可能随其他提交变化。
