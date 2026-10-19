# hccl4py 设计文档

# 需求背景（required）

## 需求来源

任务广场：[10月社区任务-实现hccl4py](https://www.hiascend.com/activities/task-center/details/63abb60433db405e932f374f31d81677)

本设计遵循任务书 §3.1（功能范围，14 个通信算子 + 5 个通信域管理接口），命名与参数顺序遵循任务书 3.1.3 示例风格。

## 背景介绍

HCCL/HCOMM 当前仅提供 C 接口，Python 用户只能各自手写 `ctypes` 声明：结构体字段需手工复刻，偏移错一位即引发底层崩溃，易错、难维护、各项目重复劳动。

为 HCCL/HCOMM 的 C 接口提供 Python 绑定：14 个通信算子（`hccl.h`）+ 5 个通信域管理接口（`hccl_comm.h`），以 `experimental/eco_system/python` 目录下的纯 Python 包 `hccl4py` 交付。绑定采用 Python 标准库 `ctypes` 直接加载 `libhccl.so`（算子与通信域接口位于同一动态库），Python 接口与 C 接口一一对应、参数语义一致，无中间封装层、无第三方 Python 依赖。

- 让 Python 应用以 `import` 方式调用 14 个通信算子，并完成"取 rootInfo → 初始化通信域 → 销毁"全流程，功能与 C 接口一致；
- 声明随仓库头文件同步维护，消除用户手工复刻头文件的负担；
- 作为 `experimental/eco_system/python` 生态组件，不改动 HCCL/HCOMM 本体任何代码。

### 现状分析

- 无官方 Python 绑定，用户各写各的 `ctypes` 声明，重复且易错。
- 验收软硬件环境：上板 2 卡；性能口径：单次调用额外开销 <10μs、抖动 ≤10%（提交口径，不含 stream 同步等待）。

### 功能范围（任务书 §3.1）

- 通信算子（14 个，`hccl.h`）：10 个集合通信算子 + 4 个点对点通信算子。
- 通信域管理接口（5 个，`hccl_comm.h`）。

# 需求分析（required）

## 需求描述

以纯 Python 包 `hccl4py` 通过 `ctypes` 绑定 HCCL/HCOMM 的 C 接口（14 个通信算子 + 5 个通信域管理接口），接口一一对应、参数语义一致、功能与 C 接口一致，单次调用额外开销 <10μs。

## 需求拆解

- 通信域：绑定 5 个接口，支持 rootInfo 与 clusterInfo 两种方式完成初始化与销毁全流程。
- 算子：绑定 14 个通信算子，接口与 C 接口一一对应。
- 工程：纯 Python 交付，无构建、无 wheel、无第三方依赖；不改动 HCCL/HCOMM 本体。
- 性能：单次调用额外开销 <10μs，抖动 ≤10%（通信域接口为低频调用，不涉及该口径）。
- 交付：随仓库源码分发，加入 `PYTHONPATH` 即可用。

# 详细设计（required）

## 算子分析

本任务为 C 接口的 Python 绑定，不实现算子计算；以下按模板三项给出绑定侧等效分析。

- **数学公式**：绑定层无计算公式，计算逻辑由底层 C 算子完成，Python 侧仅转发调用。
- **支持数据类型**：完整覆盖 `hccl_types.h` 的 `HcclDataType`（17 项）与 `HcclReduceOp`（4 项：`SUM/PROD/MAX/MIN`），取值逐项对齐，按调用方传入原样转发。
- **支持形状**：接口不传 shape，仅传 `count` 与数据地址（`all_*_v` 传 `counts`/`displs` 数组）；形状语义由 C 接口约定，绑定不做推导与校验。

## 算子实现

### 实现方案

技术选型：Python 标准库 `ctypes` 直接加载 `libhccl.so`，接口一一对应，无中间封装层、无第三方依赖。

整体架构：

```
                用户 Python 应用
   buffer   : device 内存指针 (c_void_p，acl.rt.malloc 产出)
   stream   : pyACL acl.rt.create_stream() 产出的 void*
   hccl_comm: HcclComm 句柄 (由 hccl4py.comm 创建)
   dtype/op : types.py 的枚举常量
                        │  import hccl4py
                        ▼
   hccl4py  (experimental/eco_system/python/hccl4py)
   ├─ ops/         14 个通信算子   ← hccl 仓范围（每算子一模块，__init__ 扁平再导出）
   ├─ comm/        通信域 5 接口   ← hcomm 仓范围
   ├─ types.py     公开常量层：HcclDataType / HcclReduceOp / HcclResult
   └─ _binding.py  私有声明层：CDLL + argtypes/restype + 结构体（ops/ 与 comm/ 共用）
                        │  ctypes 加载
                        ▼
                    libhccl.so
```

模块职责：

- `__init__.py`：包入口，暴露 `ops`/`comm`/`types` 并转导常用常量。
- `types.py`：公开常量层，枚举取值与 `hccl_types.h` 逐项一致。
- `_binding.py`：私有声明层，动态库单例 + 函数声明 + 结构体定义 + `HcclError`，`ops/` 与 `comm/` 共用。
- `ops/`：算子接口层（hccl 仓），每算子一模块，仅做参数转发。
- `comm/`：通信域接口层（hcomm 仓），5 个接口。

双仓分工：`ops/` 属 hccl 仓、`comm/` 属 hcomm 仓；两仓各自提交完整包目录，声明层内容保持一致，各自独立可运行。

目录结构：

```
experimental/eco_system/python/
├── hccl4py/
│   ├── __init__.py
│   ├── _binding.py            # 声明层（私有，ops/ 与 comm/ 共用）
│   ├── types.py               # 常量层（公开）
│   ├── ops/                   # 通信算子（hccl 仓，每算子一模块）
│   │   ├── __init__.py        # 扁平再导出
│   │   └── broadcast.py
│   └── comm/                  # 通信域（hcomm 仓）
│       └── __init__.py
├── examples/
│   └── 01_basic/
│       └── 01_broadcast.py
└── README.md
```

#### 接口设计

命名与参数顺序遵循任务书 3.1.3 示例风格。成功返回 C 接口的 `HcclResult`（int，0），失败抛 `HcclError`（原样携带错误码）。数组类参数以 `uint64` 连续内存指针传入，绑定原样转发。

- 集合通信算子（10，hccl 仓）：`all_reduce`、`reduce`、`broadcast`、`all_gather`、`all_gather_v`、`scatter`、`reduce_scatter`、`reduce_scatter_v`、`all_to_all`、`all_to_all_v`。
- 点对点通信算子（4，hccl 仓）：`send`、`recv`、`batch_send_recv`、`all_to_all_vc`。
- 通信域管理接口（5，hcomm 仓）：`get_root_info`、`init_rank_config`、`init_cluster_config`、`config_init`、`destroy_comm`。

各分类签名示例：

```python
all_reduce(hccl_comm, send_data, recv_data, count, data_type, reduce_op, stream)
broadcast(hccl_comm, buf, count, data_type, root, stream)
send(hccl_comm, send_data, count, data_type, dest_rank, stream)
init_rank_config(nranks, root_info, rank, config=None)
```

返回值约定：C 侧通信域接口以出参（`rootInfo`/`comm`）带回结果，Python 侧按任务书 3.1.3 风格将出参改为返回值（`init_*` 返回 `hccl_comm`，`get_root_info`/`config_init` 返回结构体），错误以异常承载——失败抛 `HcclError(result_code)`。理由：出参改返回值后无位置承载错误码；通信域创建为低频操作，异常开销可忽略。

数据类型常量：`HcclDataType`（17 项）、`HcclReduceOp`（4 项：`SUM/PROD/MAX/MIN`）、`HcclResult` 错误码（25 项），取值与 `hccl_types.h` 逐项一致。

#### 关键设计要点

- **声明方式**：`ctypes.CDLL("libhccl.so")` 模块级单例，路径支持环境变量覆盖（`HCCL4PY_LIB_PATH`），默认依赖运行环境 `LD_LIBRARY_PATH`；模块加载时即声明 `argtypes`/`restype` 以避开默认慢路径，指针以 `ctypes.c_void_p` 直接传入。
- **结构体对齐**：`HcclRootInfo`、`HcclCommConfig`、`HcclSendRecvItem` 按 `hccl_types.h` 字段顺序与默认对齐定义，布局一致性在 UT 中以 `ctypes.sizeof` 对照预期值校验。
- **inline 函数**：`HcclCommConfigInit` 为 `static inline`、非导出符号，`ctypes` 无法加载，Python 侧在 `config_init()` 内按 C 侧逻辑等价实现，随头文件同步维护。
- **内存/生命周期**：算子为异步提交，调用方需在 stream 同步完成前保持 device 内存与通信域句柄有效；`hccl_comm` 用毕须 `destroy_comm` 且不可重复销毁（C 侧行为原样透传）。
- **错误处理**：成功返回 `HcclResult`（0），失败抛 `HcclError` 并保留原始错误码（1~24 与 1041），不吞码。
- **依赖**：`libhccl.so`（14 算子 + 4 导出通信域符号）；`libascendcl.so`（`aclrtStream` 按 `void*` 透传）；Python 标准库 `ctypes`，无第三方依赖。
- **性能影响**：单次参数转发，无内存拷贝、无数据组装，开销来源为 ctypes 参数编组，预期单次额外开销 <10μs，实测数据在测试报告中给出。

#### Host/Tiling 设计

绑定层在 Python 进程（Host）侧负责：`libhccl.so` 加载、`argtypes`/`restype` 声明、结构体布局、参数编组、出参改返回值与错误码到 `HcclError` 的映射、`config_init()` 的 inline 等价实现。Tiling 与任务切分由底层 C 算子承担，绑定层不参与 tiling 计算与下发。

#### Kernel 设计

绑定层不实现 Kernel；14 个算子的核函数由 `libhccl.so` 提供，Python 侧以 `ctypes` 直接下沉调用，无 Ascend C 开发与编译。

## 支持硬件

- 验收环境（最新任务书要求）：A5 系列产品，CANN 9.1.0 及以上版本，上板 2 卡。
- 绑定为纯 Python，不含芯片相关实现，可运行范围与底层 HCCL/HCOMM 支持的硬件一致。

## 算子约束限制

- 无算子级约束：与 C 接口 1:1 对应，不新增数据类型/形状限制。
- 绑定侧调用约束：算子异步提交，调用方须在 stream 同步完成前保持 device 内存与通信域句柄有效；`hccl_comm` 用毕须 `destroy_comm`，不可重复销毁。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | 绑定不做数值计算，精度与底层 C 接口一致（转发无损） | 任务书 |
| 性能标准 | 单次调用额外开销 <10μs，抖动 ≤10%（提交口径，不含 stream 同步等待） | 任务书 |

## 测试方案

- **功能测试（ST，上板 2 卡）**：算子覆盖 14 个，每个遍历其支持的全部 `HcclDataType`，构造可预判输入并在同步后比对输出；通信域覆盖 rootInfo 与 clusterInfo 两种初始化方式、`config_init()` 配置生效路径及异常路径（抛 `HcclError` 且码值与 C 一致）。
- **性能测试（ST，上板 2 卡）**：同一算子同一规模，C 样例与 Python 样例循环调用对比，统计提交口径单次耗时（不含同步等待），验证增量 ≤10μs、抖动 ≤10%。

## 兼容性分析

- 绑定与 C API 1:1 对应，无自有演进路径，无向后兼容负担；C API 变更时同步更新声明层即可，属源码级同步。
- 位于 `experimental/`，按实验性生态组件定位提供，不承诺跨版本 API 稳定性。
- 仅新增 `experimental/eco_system/python` 目录，不改动 HCCL/HCOMM 本体；纯 Python、无构建、无 wheel，随源码分发。

## 风险评估

| 风险点 | 缓解措施 |
| --- | --- |
| ctypes 参数编组耗时超出 10μs 预算 | 性能测试前置：先做单接口耗时摸底，超标则改为预构造参数对象、减少入口处 Python 层判断 |
| 用户在 stream 未同步前释放缓冲区导致底层访问失效 | 包 README 显式约定持有 device 内存至同步完成；examples 中所有样例按该约定书写 |

## 开放问题

- 2 卡测试环境是物理 2 卡还是也允许逻辑 2 卡（单卡双芯）？
- 补充（最新任务书要求）：验证环境为 A5 系列产品、CANN 9.1.0 以上版本，需确认 A5 环境的获取与算力申请路径。

## 交付说明

- 代码：纯 Python 包 `hccl4py`，位于 `experimental/eco_system/python`；无构建、无 wheel，随源码分发，加入 `PYTHONPATH` 即可用。两仓各自提交完整包目录，声明层保持一致，各自独立可运行。
- 测试：功能用例与 README 提交至 hccl 仓 `test` 目录，通信域用例提交至 hcomm 仓 `test` 目录；性能脚本与对比数据随测试报告交付。
