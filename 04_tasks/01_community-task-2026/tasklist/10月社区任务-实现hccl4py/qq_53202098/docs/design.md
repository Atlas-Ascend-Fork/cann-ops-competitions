# hccl4py 设计文档

## 需求背景（required）

### 需求来源

社区任务《实现 hccl4py》（10 月社区任务）。任务书要求将 HCCL/HCOMM 对接 AI 框架层的 **19 个 C API** 封装为 Python 接口，功能与 C 保持一致，并在规定软硬件环境下完成功能与性能验收。

讨论帖：https://gitcode.com/cann/hccl/discussions/19

### 背景介绍

HCCL（集合通信）与 HCOMM（通信域管理）当前仅提供 C 头文件与动态库。Python 侧用户若直接手写 ctypes，易出现结构体对齐错误、符号绑定遗漏、错误处理不一致等问题。

本任务交付 **hccl4py**：在两仓 `experimental/eco_system/python` 下落地轻量 Python 绑定，使脚本与自研框架可直接调用通信域创建/销毁与集合/点对点算子。

### 现状分析

| 项 | 现状 |
| --- | --- |
| C API | hcomm 5 个通信域接口 + hccl 14 个通信算子接口已就绪 |
| Python | 无官方统一绑定包；社区需自行 ctypes |
| 验收环境 | 任务书要求 A3 或 A5、CANN ≥ 9.2.0、单机 2 卡 |
| 性能口径 | Python 相对原 C 接口额外耗时 ≤ 10µs（抖动 ±10%） |

### 功能范围（任务书 §3.1）

**hcomm（5）**：`HcclGetRootInfo` / `HcclCommInitRootInfoConfig` / `HcclCommInitClusterInfoConfig` / `HcclCommConfigInit` / `HcclCommDestroy`

**hccl（14）**：`HcclAllReduce` / `HcclReduce` / `HcclBroadcast` / `HcclAllGather` / `HcclAllGatherV` / `HcclScatter` / `HcclReduceScatter` / `HcclReduceScatterV` / `HcclAlltoAll` / `HcclAlltoAllV` / `HcclAlltoAllVC` / `HcclSend` / `HcclRecv` / `HcclBatchSendRecv`

## 需求分析（required）

### 需求描述

实现与 C 语义一致的 Python API：通信域可创建/销毁；14 个算子在 2 卡场景结果正确；绑定层相对 C 的额外开销满足 ≤ 10µs。

### 需求拆解

1. 通信域：RootInfo / ClusterInfo 两种建域方式 + Config 初始化 + Destroy。
2. 算子：14 个集合/点对点接口全覆盖，参数类型与枚举取值对齐头文件。
3. 工程：两仓目录均为 `experimental/eco_system/python`，提供 README、examples、可复现测试说明。
4. 性能：同逻辑对比 C 与 Python 耗时并给出数据。
5. 交付：本仓提交设计文档；代码与自测按任务书合入 hccl/hcomm。

## 详细设计（required）

### 总体方案

采用 **ctypes 纯 Python 绑定**（不引入 pybind11/C 扩展编译链）：

```text
Python 应用
    └─ hccl4py
         ├─ _lib.py     加载 libhcomm.so / libhccl.so / libascendcl.so
         ├─ types.py    枚举与结构体布局
         ├─ comm.py     通信域 5 API（hcomm 仓权威）
         └─ ops.py      通信算子 14 API（hccl 仓权威）
              └─ ctypes 调用原 C 导出符号
```

双仓分工：

| 仓 | 路径 | 内容 |
| --- | --- | --- |
| hcomm | `experimental/eco_system/python` | 通信域绑定 |
| hccl | `experimental/eco_system/python` | 通信算子绑定 + examples/tests |

### 接口设计

遵循任务书 §3.1.3 形态示例：

| Python | C | 返回 |
| --- | --- | --- |
| `comm_init_root_info_config(nranks, root_info, rank, config=None)` | `HcclCommInitRootInfoConfig` | `hccl_comm` 句柄 |
| `all_reduce(comm, send, recv, count, data_type, reduce_op, stream)` | `HcclAllReduce` | `status` |

约定：

- `comm` / `stream` / device buffer 使用不透明句柄或设备地址（`int` / `c_void_p`）。
- 非 0 的 `HcclResult` 转为带 API 名的异常，便于脚本排查。
- `HcclCommConfigInit` 为头文件 inline，Python 侧按同等语义填充 magic/version/默认字段。

### 关键设计要点

1. **一次性绑定**：`argtypes`/`restype` 仅设置一次，避免每次调用重置带来额外开销。
2. **结构体对齐**：以运行 CANN 实测 `sizeof`/偏移为准（例如 `HcclSendRecvItem`、`HcclRootInfo`、`HcclCommConfig`），防止 BatchSendRecv 等接口因布局错误失败。
3. **零额外拷贝**：绑定层不做 device 数据拷贝；buffer 生命周期归调用方；异步入队语义与 C 一致，同步由调用方 `aclrtSynchronizeStream` 完成。
4. **库加载顺序**：先 `libhcomm` 再 `libhccl`，并兼容 `LD_LIBRARY_PATH` / toolkit `lib64`。

### 测试方案

| 类型 | 内容 |
| --- | --- |
| 功能 | 单机 ≥2 device：建域 → 14 算子结果校验 → 销毁 |
| 冒烟 | RootInfo + AllReduce |
| 性能 | 同逻辑 C（ctypes 直调或原生二进制）vs Python；额外开销 ≤ 10µs |
| 文档 | tests README 写清复现步骤 |

验收环境以任务书为准：**A3 或 A5 + CANN ≥ 9.2.0 + 单机多卡**。

### 兼容性

- 仅新增 `experimental/` 目录，不影响既有 C API。
- Config/结构体随 toolkit 演进时，以运行环境实测布局校准。

## 风险评估

| 风险 | 应对 |
| --- | --- |
| 结构体布局漂移 | sizeof 断言 + 联调校准 |
| 缺少单机多卡 / CANN≥9.2 官方环境 | 讨论帖申请算力；联调与正式验收环境分开标注 |
| 双仓包名协同 | README 说明 PYTHONPATH / 安装方式 |

## 替代方案

| 方案 | 结论 |
| --- | --- |
| ctypes（采用） | 无编译依赖，易随 so 升级，开销可控 |
| pybind11 | 类型更安全，但增加编译与发布复杂度 |

## 开放问题

1. 正式验收所需单机多卡（A3/A5）+ CANN≥9.2 环境申请路径（已在讨论帖提问）。
2. ClusterInfo（ranktable）样例是否作为必验收用例。

## 交付说明

- **本文档**：提交至 `cann-ops-competitions` 本任务目录 `docs/design.md`，供设计评审。
- **代码与自测**：后续按任务书以 PR 形式分别提交到 `cann/hcomm`、`cann/hccl` 的 `experimental/eco_system/python`（不含在本 PR）。
