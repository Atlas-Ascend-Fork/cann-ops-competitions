# RFC: hccl4py —— HCOMM 通信域管理与类型层的 Python 绑定

- 起始日期：2026-10-18
- RFC PR编号：[cann/hcomm !6096](https://gitcode.com/cann/hcomm/merge_requests/6096)
- 相关Issue：[cann/hcomm #1044](https://gitcode.com/cann/hcomm/issues/1044)
- 模板依据：以[社区任务设计模板](../../../resources/design_template.md)的必选章节为评审基准，RFC 章节作为补充

---

## 概要

为 HCOMM 的 5 个通信域管理 C API（`include/hccl/hccl_comm.h`）提供 Python 绑定，
并交付被两侧共用的类型层（枚举/结构体/动态库加载/错误类型），形成基础包 **hccl4py**。
算子侧绑定（14 个）在 hccl 仓，
两包同时安装时 `import hccl4py` 即可同时使用通信域与算子接口。

## 需求分析（required）

### 需求描述

在 hcomm 仓 `experimental/eco_system/python/` 下交付 `hccl4py` 基础包，覆盖 5 个通信域管理
C API，并集中维护两侧共用的 enum、struct、动态库加载、公开协议和错误类型。`init_*` 成功返回
Python `Comm`，失败抛出 `HcclError`；通信域能力要通过 rootinfo/ranktable 两条真实双卡路径验证。

### 需求拆解

1. 5 个通信域接口完整封装，返回值和失败语义与任务书 §3.1.3 对齐；
2. `HcclRootInfo` 支持无损 `to_bytes()/from_bytes()`，`HcclCommConfig` 按头文件 inline 语义初始化；
3. 运行库、头文件、配置 ABI 来自同一安装；version10/11 选择可由 so 能力和 magic/version 校验；
4. 交叉仓依赖使用公开 `hccl4py.api`，私有模块不得跨仓引用；
5. 源码随仓分发、examples、测试步骤、测试报告和双卡结果可复现；
6. 配套测试指导的 10μs 性能用例仅覆盖算子；通信域控制面记录 median/P95 与 C 直调对比作为参考，
   该验收口径需任务方确认。

## 运行与验收环境

| 项目 | 统一口径 |
|---|---|
| 任务兼容下限 | A5 系列 + CANN 9.1.0 + pyACL；执行初始化、rootinfo/ranktable 建域与销毁冒烟；若任务方指定 9.1.0 为最终验收环境，则运行同一套全量测试 |
| 默认正式验收基线 | A5 系列 + CANN 9.2.0 + pyACL + 单机 2 卡 |
| 配置 ABI | 9.1.0 为 version10/2240B；9.2.0 为 version11/2248B，新增 `hcclChannelSqDepth` |
| 安装一致性 | 头文件、`libhcomm.so`、`libhccl.so`、`libascendcl.so` 与 pyACL 必须来自同一安装根目录 |
| 测试材料 | 当前材料中的 A3/950 不是本设计验收依据；交付前须由任务方/测试负责人同步为 A5 + CANN 9.2.0 |

公开 `v9.1.0`/`v9.2.0` 头文件对比表明，5 个目标接口与 `HcclSendRecvItem`/`HcclDataType`
没有实质 ABI 变化，配置结构从 version10 增至 version11。9.1.0 运行支持通过头文件/so 的同安装
检查和原型指纹保障。最终验收 CANN 版本须由任务方确认；若不能锁定 9.2.0，9.1.0 下的功能、双卡和
性能执行范围不得缩减。

## 背景与动机

* Python 用户建通信域目前只能手写 ctypes（如 `HcclGetRootInfo` 的 4108 字节结构体、
  `HcclCommConfig` 的2240/2248字节结构体（随配置ABI版本变化）），极易出错；
* 结构体布局与 `static inline` 接口（`HcclCommConfigInit`）在动态库中没有符号，
  必须按头文件语义在 Python 侧等价实现，需要集中、可验证地维护；
* 两侧（通信域/算子）共用同一套类型与库加载逻辑，放在下层仓（hcomm）更合理。

## 术语表

| 术语 | 含义 |
|------|------|
| hccl4py | 本 RFC 交付的基础包：类型层 + 通信域接口 |
| configInfo_t | `HcclCommConfig` 前 24 字节的 {size, magicWord, version, reserved} |
| rootinfo | `HcclRootInfo`，4108 字节不透明标识，用于 `HcclCommInitRootInfoConfig` 建域 |
| 公开 API | `hccl4py.api` 及其公开协议/类型；不以下划线模块作为跨仓接口 |

## 架构与接口契约

### 整体架构

```text
   hccl4py_ops（hccl 仓，算子）      hccl4py（本仓，基础 + 通信域）
              │                               │
              └──────── hccl4py.api ──────────┘
                              │
                        ctypes FFI 层
        libhccl.so（算子）   libhcomm.so（通信域）   libascendcl.so（运行时）
```

### 包与分发

第一阶段使用“源码随仓 + `PYTHONPATH`”分发，不依赖 wheel：

```text
<hcomm>/experimental/eco_system/python/  # hccl4py
<hccl>/experimental/eco_system/python/   # hccl4py_ops
```

README 固定安装顺序：

```bash
export PYTHONPATH=<hcomm>/experimental/eco_system/python:<hccl>/experimental/eco_system/python
python -c "import hccl4py, hccl4py_ops"
```

`hccl4py.__init__` 在 `api` 初始化后通过模块级 `__getattr__` 惰性导出 `hccl4py_ops`，避免两包
循环导入；未安装算子包时仍可独立使用通信域。`hccl4py_ops` 只能依赖公开 `hccl4py.api`，
不得导入 `_ffi`/`_config`/`_lifetime` 等私有实现。hcomm 基础包
升级内部模块时，公开 API/协议的回归测试负责守住跨仓契约。正式 wheel 与版本策略为后续问题。

### 对外接口与返回值

| C API | Python 接口 | 返回值 | 验收用例 |
|---|---|---|---|
| `HcclGetRootInfo` | `get_root_info() -> RootInfo` | 成功返回 4108 字节 `RootInfo`；失败抛 `HcclError` | `test_comm_smoke.py::test_root_info_roundtrip` |
| `HcclCommInitRootInfoConfig` | `init_root_info_config(nranks, root_info, rank, config=None) -> Comm` | 成功返回 `Comm`；失败抛 `HcclError` | `test_two_rank.py::test_root_info_init` |
| `HcclCommInitClusterInfoConfig` | `init_cluster_info_config(rank_table, rank, config=None) -> Comm` | 成功返回 `Comm`；失败抛 `HcclError` | `test_two_rank.py::test_ranktable_init` |
| `HcclCommConfigInit`（内联） | `comm_config_init(**fields) -> CommConfig` | 成功返回已写 magic/version 的配置；非法字段抛 `ValueError` | `test_config.py::test_defaults_and_profile` |
| `HcclCommDestroy` | `comm_destroy(comm) -> None` | 成功无返回；失败抛 `HcclError` | `test_two_rank.py::test_destroy_and_cleanup` |

辅助接口：`get_rank_size` / `get_rank_id` / `get_comm_name` / `get_error_string` /
`init_root_info` / `init_cluster_info`。辅助查询成功返回 Python `int`/`str`，失败抛
`HcclError` 并保留 C 错误码。

`root_info` 参数接受 `RootInfo` 或 `to_bytes()` 产生的 4108 字节 `bytes`。两个 `init_*` 在
`config=None` 时先调用 `comm_config_init()` 生成完整默认配置；非空 config 必须由本包初始化，
禁止把全零或字段不完整的 ctypes 结构直接传入。

命名规则：去掉 `Hccl` 前缀，将 lowerCamelCase 转 snake_case；位于 `hccl4py.comm` 命名空间下时，
只为避免重复而省略作为上下文的 `Comm`。例如 `HcclCommInitRootInfoConfig` 对应
`init_root_info_config`，`HcclCommConfigInit` 保留为 `comm_config_init`。任务书中的
`init_rank_config` 不是当前 C 符号名，本设计不提供同名别名，以免把 RootInfo 与 ClusterInfo
两条路径混淆；README 给出逐项映射。

错误语义：通信域接口调用失败（返回值非 0）直接抛出 `HcclError`；算子侧接口（hccl 仓）
默认返回状态码、可切换为抛异常模式，两层语义不一致是有意为之——建域是低频关键路径，
失败即抛便于尽早暴露；算子是高频路径，保留与 C 一致的“返回状态码”语义。

### 依赖接口

* 运行期：`libhcomm.so` 的通信域符号；`libascendcl.so` 同时服务于示例/自测和算子侧运行期，
  用于 current stream 查询与 event 管理；
* 类型定义以实际安装的 `include/hccl/hccl_types.h` 为准，布局由 `tools/c_layout_ref.c` 交叉校验；
* 跨仓只暴露 `hccl4py.api` 中的 `Comm`/`BufferLike`/`HcclDataType`/`HcclResult`/`HcclError`/
  `LibInfo`、ACL 最小封装和通信域清理 callback 协议。

## 影响分析

* 对性能：通信域接口调用频率低（建域级），ctypes 开销可忽略；类型层零拷贝传递指针；
* 对现有功能：纯新增实验目录，不修改 `src/`、`include/`、`pkg_inc/`；
* 对构建：Python 包无运行期编译产物；C 布局校验工具仅测试期使用，不随包发布；
* 对跨仓维护：通过稳定 `hccl4py.api` 解耦，hcomm 内部重构不得直接破坏 hccl 算子包。

## 兼容性考虑

* 不改变 C 接口与头文件；
* 9.1.0（version10/2240B）与 9.2.0（version11/2248B）均支持；9.2.0 为正式验收基线；
* `HcclCommConfig` 先由同一安装根目录头文件预判，再由已加载 so 的能力、magic/version/首建域
  结果兜底；冲突必须在错误中同时报告头文件与 so 路径；
* 动态库覆盖变量只改变安装根目录选择，不允许默认组合“A 安装的头文件 + B 安装的 so”；
* 本 RFC 新增独立实验目录和 Python 包，不涉及在线服务变更，不需要特性开关或灰度发布策略。

## 详细设计

### `_config.py`

- **模块职责**：先解析实际运行时安装，再从同一安装根目录确定 `HcclCommConfig` profile。
- **核心数据**：`RuntimeProfile(header_path, libhcomm_path, version, struct_size, capability)`。
- **关键逻辑**：
  1. 按显式 `HCCL4PY_INSTALL_ROOT`、`HCCL4PY_HCOMM_LIB` 所在 `lib64/` 的父目录、
     `HCCL4PY_LIB_DIR`、`ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME` 顺序选出 **so 安装根目录**；
  2. 只在该根目录搜索 `include/hccl/hccl_types.h`，读取 `HCCL_COMM_CONFIG_VERSION`、
     `HCCL_COMM_CONFIG_MAGIC_WORD` 和字段清单；源码仓头文件不得作为运行期 ABI 依据；
  3. version10 要求 2240B，version11 要求 2248B；用入仓 profile 校验 `sizeof/offsetof`；
  4. 加载实际 `libhcomm.so` 后解析 `HcclGetCommConfigCapability`。
     [官方文档](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/HcclGetCommConfigCapability.md)
     明确返回“标量上限”，对已有枚举字段使用 `capability > HCCL_COMM_CONFIG_<FIELD>`；Qos/SqDepth 没有独立能力枚举，
     由 version/size/profile 与首次建域校验覆盖，不宣称它们经过 capability 判断；
  5. 显式 `HCCL4PY_CONFIG_VERSION` 只用于源码/测试环境，且必须同时提供匹配 profile；验收环境
     禁止用它绕过同安装检查；
  6. 首次建域时若库拒绝 magic/size/version，`HcclError` 同时携带头文件、so、version、size 与
     首个建域错误码，避免只报“初始化失败”。
- **内部接口**：`resolve_runtime()` 向 `_ffi.py` 提供唯一 profile；`libs_info()` 输出最终路径。

### `_ffi.py`

- **模块职责**：维护 ctypes 常量、枚举、结构体、动态库加载、符号缓存和统一错误类型。
- **核心数据结构**：`HcclRootInfo`（4108B）、`HcclCommConfig`、`HcclSendRecvItem` 以及与 C 头文件对应的枚举。
- **关键逻辑**：`HcclCommConfigInit` 由 Python 按头文件语义实现，前 24 字节写入
  `{size, magicWord, version, reserved}`，其余字段填写“未配置”缺省值；
  rootinfo 使用 `ctypes.string_at`/`memmove` 读写，避免 NUL 截断；
  每个函数原型缓存为独立 `CFUNCTYPE`，避免修改共享 `CDLL` 函数属性。
- **关键校验**：5 个目标符号、`HcclGetCommConfigCapability`、类型尺寸和配置首 24 字节均在
  公开 API 返回前校验；未知 profile 拒绝导入。
- **内部接口**：提供 `bind()`、`hccl_lib()`、`hcomm_lib()`、`acl_lib()` 和 `HcclError`；
  跨包可见类型通过 `hccl4py.api` 再导出。

### `_prototypes.py`

- **模块职责**：为 hcomm 5 接口和 hccl 14 接口提供统一的 C 原型解析与指纹归一化。
- **核心数据**：`("函数名", "返回类型", ("参数类型", ...))` 不可变 tuple，以及解析诊断。
- **关键逻辑**：仅使用 Python 标准库；去除注释和属性宏后，以受限词法/括号扫描提取目标声明，
  丢弃参数名并归一化 typedef、固定宽度整数和枚举别名，保留参数顺序与指针层级。任一头文件缺失、
  宏无法求值、声明缺失或解析歧义均抛出 `AbiCompatibilityError` 并拒绝导入。
- **内部接口**：通过 `hccl4py.api.abi` 暴露 `parse_prototypes()`/`prototype_fingerprint()`；
  hcomm 校验 5 接口，hccl `_abi.py` 复用同一实现校验 14 接口。

### `_buffer.py`

- **模块职责**：把 Python device 缓冲区转换为 `(ptr, count, data_type)`。
- **核心数据**：device 指针、元素数量和 `HcclDataType`。
- **关键逻辑**：支持裸 device 指针、`ctypes.c_void_p`、实现 `data_ptr()` 的 device 对象和
  NPU `torch.Tensor`；Tensor 必须 `device.type == "npu"` 且连续。显式 `count`/`data_type`
  优先，仅在可唯一推导时从 Tensor 补齐；裸指针缺少元数据时明确报错。CPU Tensor 和
  `numpy.ndarray` 均不作为数据 buffer，明确拒绝且不触发 H2D 拷贝。
- **内部接口**：`ptr_of()` 和 `describe_buffer()` 供算子绑定复用。

### `comm.py`

- **模块职责**：实现 5 个通信域管理接口及头文件辅助接口。
- **核心数据**：通信域句柄、只读 `rank_size`、`HcclRootInfo`、`HcclCommConfig` 和 `HcclResult`。
- **关键逻辑**：`comm_config_init()` 按 C 内联实现生成默认配置；建域支持 rootinfo 和 ranktable
  两条路径；rootinfo 直接缓存 nranks，ranktable 建域后只查询一次 rank size，并在通信域生命周期内
  作为只读 `Comm.rank_size` 公开。接口非 0 返回值转换为含 API 名与错误字符串的 `HcclError`。
  `Comm` 实现公开 `data_ptr()` 协议。`comm_destroy()` 先调用注册的通信域清理 hook；若 V/VC/Batch 参数 event
  仍 pending，则要求调用方先同步 stream，不静默销毁。
- **内部接口**：对外提供 `get_root_info`、`init_root_info_config`、
  `init_cluster_info_config`、`comm_config_init`、`comm_destroy` 及辅助查询接口。

### `api.py`、包入口与测试辅助

- **模块职责**：`api.py` 是跨仓稳定接口；`__init__.py` 组织公开导出；`_acl.py` 为示例和
  测试提供最小 ACL 封装。
- **核心数据**：公开接口列表 `__all__`，以及按需导入的 `hccl4py_ops` 算子接口。
- **关键逻辑**：仅安装 hcomm 包时只导出通信域与类型接口；两个包都安装时额外导出 14 个算子接口。
  跨仓类型、`Comm.rank_size` 和 cleanup 注册协议只从 `api.py` 暴露，下划线实现不进入 stable contract。
- **内部接口**：`_acl.py` 提供 device、stream、`DeviceBuffer` 和内存申请/释放的基础封装，
  不进入通信域绑定主路径；其公开别名经 `hccl4py.api.acl` 暴露。

## 可维可测分析（required）

### 精度与功能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 返回值契约 | init 类成功返回有效 `Comm`；失败抛 `HcclError`；destroy 失败不得伪装成功 | 任务书 §3.1.3 |
| rootinfo 无损 | `to_bytes()/from_bytes()` 后 4108 字节逐字节一致，含 NUL 与末尾字节 | HcclRootInfo C 定义 |
| 配置语义 | 前 24 字节 size/magic/version 正确，所有未配置字段与 C inline 缺省值逐字段一致 | `HcclCommConfigInit` |
| 建域 | rootinfo 与 ranktable 两条真实 2 卡路径均成功，返回 rank size/id/name 与 C 一致 | 任务书功能范围 |
| 生命周期 | 成功建域后 `comm_destroy` 只执行一次；pending 参数未完成时不得静默销毁 | C API 与 HCCL 异步约束 |

测试文件与覆盖：

| 用例 | 无/有 device | 覆盖 |
|---|---|---|
| `test_config.py` | 无 | 5 个符号、version10/11 profile、2240/2248B、magic/version、默认字段、capability 字段 |
| `test_comm_smoke.py` | 无 | rootinfo 序列化、错误码映射、`libs_info()`、头文件/so 冲突报错 |
| `test_public_api_contract.py` | 无 | 两种包导入顺序、惰性导出、跨仓公开协议、`Comm.rank_size`、无私有模块访问 |
| `test_single_rank.py` | 有 | 单 rank 建域、辅助查询、重复销毁错误 |
| `test_two_rank.py` | 有 | rootinfo/ranktable 两条建域路径、rank 一致性、销毁 |
| `test_cleanup_hook.py` | 有 | V/VC/Batch host 参数 event 未完成时拒绝 comm 销毁，清理后可销毁 |

`test_config.py` 还包含原型解析扰动样例（多行声明、注释、属性宏、无法解析宏）、capability 返回值
和 Qos/SqDepth profile 校验，以及 rootinfo/ranktable 的 rank size 缓存一致性检查。

默认先运行 9.2.0/A5 双卡全量；若任务方确认 9.1.0 为最终验收环境，则在 9.1.0 执行同一套全量任务，
而不是只运行冒烟。
通信域控制面用例的规划预算为 10–15 分钟，报告记录每个用例的环境、参数、日志和截图。

### 性能标准

配套测试指导的性能验证用例仅覆盖算子，故 10μs 指标按算子下发路径执行。通信域接口不在该
性能用例范围内，本设计记录 median/P95 与 C 直调对比作为参考；该口径仍需任务方确认。

| 指标 | 标准 |
|---|---|
| 建域/销毁 | 记录 median/P95 和 C 直调对比，不允许 Python 额外引入同步、重试或隐藏建域 |
| `comm_config_init` | 纯 Python 本地初始化，记录固定耗时和内存大小，不影响通信下发热路径 |
| 数据面性能 | 由 hccl `hccl4py_ops` 的 C/Python 同口径 Batch 验收负责 |

所有测量同时记录 CANN/pyACL、驱动、A5 型号、2 卡拓扑、git commit 和实际 so 路径。含同步的
建域耗时仅作参考，不与算子 host 下发指标混用。

### 兼容性分析

* 不修改 C API、头文件和库符号；
* 9.1.0（version10/2240B）与 9.2.0（version11/2248B）分别维护 profile，未知 profile 明确拒绝；
* `HcclGetCommConfigCapability()` 按官方文档的标量上限语义，对已有枚举字段执行 `>` 比较；
  Qos/SqDepth 无 capability 枚举，由 version/size/profile 与首次建域校验覆盖；
* 避免“头文件来自 A、so 来自 B”：默认从实际 so 的安装根目录定位头文件，源码仓头文件不参与；
* 公开 `hccl4py.api` 保持跨仓兼容；内部 `_config/_ffi/_buffer` 可重构但必须通过跨包回归测试；
* 仅依赖 Python 标准库，NPU Tensor 互操作按可选功能探测。

## 风险评估

1. **头文件与 so 版本错配**：可能把 2248B 配置交给 version10 库。缓解：先确定实际 so 根目录、
   再在同根目录选头文件；profile 必须匹配；加载后校验 capability；首建域错误附带两边路径；
2. **`HcclCommConfigInit` 默认值漂移**：Python 等价实现可能与头文件不一致。缓解：
   `c_layout_ref.c` 输出 size/offset/default bytes，单测逐字段和原始字节断言；
3. **capability 被误用为版本号**：官方接口返回标量上限，不是位掩码。缓解：版本来自同安装头文件
   和入仓 profile；capability 仅对已有枚举字段做上限比较，Qos/SqDepth 不用 capability 枚举，
   magic/version/size 与首次建域再独立校验；
4. **多版本共存**：环境变量可能误选库。缓解：`libs_info()` 打印最终根目录/头文件/so/符号指纹；
   冲突时 fail fast；
5. **跨仓私有耦合**：内部模块变化破坏算子包。缓解：只允许 `hccl4py.api`；两仓 CI 共同运行公开
   API 契约测试；
6. **组网环境差异**：受限容器中 ranktable/真实 EID 不可用。缓解：rootinfo 默认冒烟，ranktable
   使用真实 2 卡端口/EID，建域成功不能替代数据传输验证。

## 交付件与目录映射

| 任务交付项 | 落位目录 | 职责 |
|---|---|---|
| 基础包代码 | `hcomm/experimental/eco_system/python/hccl4py/` | 类型、配置、通信域、公开 `api.py` |
| 代码目录 README | `hcomm/experimental/eco_system/python/README.md` | `PYTHONPATH`、依赖方向、双仓安装顺序 |
| examples | `hcomm/experimental/eco_system/python/examples/01_basic/01_init_comm.py` | rootinfo、config、建立/销毁通信域 |
| 可复现测试说明 | `hcomm/test/ecosystem/hccl4py/README.md` | 2 rank 启动、ranktable、预期结果、清理步骤 |
| 功能测试与报告 | `hcomm/test/ecosystem/hccl4py/`、`.../report/` | 5 接口、两条建域、version 矩阵、环境和截图 |
| 布局交叉校验 | `hcomm/tools/c_layout_ref.c` | 输出 size/offset/default bytes，不随包发布 |
| 算子侧交付 | `hccl/experimental/eco_system/python/hccl4py_ops/` 与 hccl 测试目录 | 14 个接口、生命周期、全 dtype 与 10μs 性能，见 hccl RFC |

`c_layout_ref.c` 是一次性验证工具，需要 C 编译器和匹配的 CANN 头文件；Python 交付包无构建
步骤。工具的命令、include 路径和二进制忽略规则必须写入目录 README。

## 替代方案

1. **C 扩展（Cython / pybind11）**：调用开销更小，但引入编译产物与 ABI 维护成本，且
   `HcclCommConfigInit` 内联实现仍需 C 侧胶水，未采用；
2. **依赖 torch_npu 等框架的绑定**：引入重依赖，无法覆盖自研框架/HPC 脚本场景，未采用；
3. **把类型层放在 hccl 仓**：会形成 hcomm 反向依赖 hccl，违反仓间方向，未采用；
4. **两仓各自复制私有类型层**：容易出现枚举/配置漂移，故改用公开 `hccl4py.api`。

## 开放问题

1. 正式 wheel、独立版本号和跨仓 pip 发布顺序；
2. 是否在本阶段额外暴露 `HcclCreateSubCommConfig` 等非任务范围辅助接口；
3. ranktable 路径在受限容器中是否需要额外的网络诊断工具。
4. 任务方需确认最终验收 CANN 版本；若不能锁定 9.2.0，9.1.0 必须执行同一套全量功能与性能验收。
5. 任务方需确认 10μs 指标仅适用于算子性能用例，并确认通信域 median/P95 仅作参考数据。

## 评审记录

评审意见记录于[设计PR !6096](https://gitcode.com/cann/hcomm/merge_requests/6096)。
