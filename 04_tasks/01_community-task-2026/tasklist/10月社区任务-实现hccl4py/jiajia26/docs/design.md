# 需求背景（required）

## 需求来源

10 月社区任务：实现 **hccl4py** —— 将 HCCL/HCOMM 对接 AI 框架层的 C API 通信能力封装为 Python 接口，使 Python 应用（自研训练框架、HPC 脚本、模型调优工具等）能直接使用 HCCL&HCOMM 的通信能力，Python 接口功能与 C 接口保持一致。

- 任务书：`hccl4py_社区任务书-阶段一.md`（本文唯一范围真值源；配套 `test_cases/README.md` 作为测试用例分类参考）
- 设计文档提交仓：`cann-ops-competitions`（https://gitcode.com/cann/cann-ops-competitions）
- 提交路径：`04_tasks/01_community-task-2026/tasklist/10月社区任务-实现hccl4py/jiajia26/docs/design.md`
- 代码合入仓与合入目录（任务书 §6）：
  - hcomm 仓 https://gitcode.com/cann/hcomm → `experimental/eco_system/python/`（通信域管理 5 接口的 Python 绑定）
  - hccl 仓 https://gitcode.com/cann/hccl → `experimental/eco_system/python/`（通信算子 14 接口的 Python 绑定）
- C API 权威来源（绑定对象与语义基线）：
  - 通信域管理：hcomm 仓 `docs/zh/api_ref/comm_mgr_c/`（含数据类型定义、通信域管理）
  - 通信算子：hccl 仓 `docs/zh/api_ref/comm_op_interface/`（集合通信 + 点对点通信）
- 适配硬件：Ascend 950（A5）系列产品（任务书 2026-10 变更：验收硬件锁定 A5）；CANN 9.1.0 及以上
- 交付节奏：阶段一先交付设计文档（本文），绑定代码、examples、test 后续按仓以 PR 形式提交

## 背景介绍

### hccl4py 实现

HCCL（Huawei Collective Communication Library）与 HCOMM 是基于昇腾 AI 处理器的高性能集合通信库，当前仅提供 C 语言接口（`.h` 头文件 + `libhccl.so`/`libhcomm.so`）。Python 用户直接调用 C API 需手写 ctypes 绑定：结构体布局、枚举取值、指针传递、错误码解释均需自行维护，易错且难维护。hccl4py 对上述 19 个 C API 提供官方 Python 绑定：

```
通信域管理（hcomm 仓，5 个）：
  HcclGetRootInfo / HcclCommInitRootInfoConfig / HcclCommInitClusterInfoConfig /
  HcclCommConfigInit / HcclCommDestroy

通信算子（hccl 仓，14 个）：
  集合通信 11：HcclAllReduce / HcclReduce / HcclBroadcast / HcclAllGather / HcclAllGatherV /
               HcclScatter / HcclReduceScatter / HcclReduceScatterV / HcclAlltoAll /
               HcclAlltoAllV / HcclAlltoAllVC
  点对点 3：  HcclSend / HcclRecv / HcclBatchSendRecv
```

### 接口与仓内现状分析

| 项 | 仓内现状 | 与本任务关系 |
| --- | --- | --- |
| hcomm 仓 `python/hccl/` | 已有 ctypes 形态的旧接口绑定（`HcomCreateGroup`/`HcomDestroyGroup`/`HcomGetRankSize` 等旧 Hcom* 管理接口，whl 形态 `hccl-0.1.0`），`ctypes.CDLL('libhcomm.so')` 的加载方式经仓内验证 | 旧接口族，未覆盖任务书要求的 19 个 C API；本任务不改动它，新代码物理隔离到 `experimental/eco_system/python/` |
| hccl 仓 | 无任何 Python 绑定 | 全新增量 |
| `experimental/` 目录 | 两仓均已存在（hccl 仓已有 `experimental/eco_system/`、`experimental/ops/`） | 合入目标目录按任务书 §6 落位 |
| C 接口头文件 | `hccl/include/hccl.h`（14 算子原型）、hcomm 仓 `include/hccl/hccl_comm.h`（通信域管理原型）、`include/hccl/hccl_types.h`（全部类型定义） | 原型齐备，ctypes 声明可逐一对齐，无需改动任何 C 侧代码 |

结论：

1. 绑定技术沿用仓内已验证的 **ctypes 纯 Python 路线**（hcomm 仓 `python/hccl/manage/api.py` 即 `ctypes.CDLL` 直载 `libhcomm.so`），不引入 C 扩展编译链，交付物为纯 Python 目录/wheel，随 CANN 环境即装即用。
2. 新代码与既有 `python/hccl/` 物理隔离（不同目录、不同接口族），对既有 whl 构建目标零影响。
3. 任务书 19 个接口全部落在上述三个头文件内，类型系统（HcclResult/HcclDataType/HcclReduceOp/HcclRootInfo/HcclCommConfig/HcclSendRecvItem）可完整静态映射，绑定层无动态行为需要探测。

### hccl4py 功能分析

**功能**：为 19 个 C API 提供一一对应的 Python 接口，覆盖通信域生命周期（创建/配置/销毁）与全部通信算子（集合通信 + 点对点）。

**关键 C 语义（绑定契约，语义以官方接口文档为准）**：

- `HcclComm` 为 `void*` 句柄；`HcclRootInfo` 为 4108 字节不透明结构（`char internal[4108]`），root 生成后需广播给非 root rank。
- `HcclCommConfigInit` 是头文件 **static inline** 函数（非导出符号），向 `HcclCommConfig.reserved[24]` 写入 `{size, magicWord=0xf0f0f0f0, version=11, reserved=0}` 头并填充各字段默认值（未配置项为 `0xffffffff`/`-1` 等 NOT_SET 哨兵）。
- 通信算子的 `sendBuf/recvBuf` 为 **Device 内存地址**；`stream` 为 `aclrtStream`（`void*`）；接口异步入队、返回 `HcclResult`，同步由调用方完成。
- V 类接口（AllGatherV/ReduceScatterV/AlltoAllV）的 counts/displs 为 `uint64` 数组（`const void*` 传入），数组长度等于 rank size；AlltoAllVC 的 sendCountMatrix 为 nRanks×nRanks 的二维 uint64 数组；BatchSendRecv 接受 `HcclSendRecvItem` 数组（`{sendRecvType, buf, count, dataType, remoteRank}` 五元组）。
- `HcclResult`：`HCCL_SUCCESS=0`，失败码 1–24（参数/资源/网络/超时/OOM 等），`HCCL_E_IN_STATUS=1041`。

**支持数据类型**（`HcclDataType`）：int8/int16/int32/int64/uint8/uint16/uint32/uint64/fp16/fp32/fp64/bfp16/int128/hif8/fp8e4m3/fp8e5m2/fp8e8m0（各型号支持集以接口文档 dataType 说明为准，绑定层完整枚举、不收紧）。

**归约类型**（`HcclReduceOp`）：SUM=0 / PROD=1 / MAX=2 / MIN=3。

**验收要求**（任务书 §5）：

- 功能：2 卡范围内正常完成通信域创建和销毁、14 个通信算子运行结果正确（含各支持数据类型）。
- 性能：Python 接口相对原 C 接口增加耗时 **<10us**（抖动范围 10% 之内），需同时给出 C 接口与 Python 接口耗时对比。

# 需求分析（required）

## 需求描述

Python 接口命名采用 snake_case，与 C 接口一一对应；接口形态与任务书 §3.1.3 给定的映射风格保持一致并作为契约：

| Python 接口 | C 接口 | Python 返回值 |
| --- | --- | --- |
| `init_rank_config(nranks, root_info, rank, config=None)` | `HcclCommInitRootInfoConfig(nRanks, rootInfo, rank, config, comm)` | `hccl_comm` |
| `all_reduce(hccl_comm, send_data, recv_data, reduce_op, stream=None)` | `HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream)` | `status` |

其余接口同构：参数顺序与 C 原型一致，`dataType` 可由 `data` 实参自动推断（显式传入者优先），出参指针（`comm`）转为 Python 返回值，status 类返回值保持 int（`HcclResult` 值）。

## 需求拆解

1. **hcomm 仓** `experimental/eco_system/python/hccl4py/`：通信域管理 5 接口绑定 + `RootInfo`/`CommConfig` 封装 + 类型/异常定义。
2. **hccl 仓** `experimental/eco_system/python/hccl4py/`：通信算子 14 接口绑定（集合通信 11 + 点对点 3，含 `HcclSendRecvItem` 封装）+ 类型/异常/设备指针工具。
3. **双仓包共存**：顶层包名统一为 `hccl4py`，采用 PEP 420 namespace package，两仓子模块文件物理不重叠，pip 双发行版安装后自动聚合。
4. **绑定层**：ctypes 原型预声明（argtypes/restype）、枚举 IntEnum 化、buffer 指针提取约定、错误码与异常双模式。
5. **examples**：hccl 仓 14 个算子样例（对齐任务书举例 `examples/01_basic/01_allreduce.py` 编号风格）+ hcomm 仓通信域创建/销毁样例，均可在 2 卡复现。
6. **test**：功能用例（对齐任务配套 `test_cases/README.md` 的 14 算子分类）+ C/Python 性能对比脚本，含 README 测试步骤。
7. 各仓代码目录提供 README（安装、环境要求、快速上手）。

# 详细设计（required）

## 总体架构

```
用户代码 / examples（hccl4py API 消费方）
        │
        ▼
hccl4py Python API 层     comm_mgmt.py / comm_config.py（hcomm 仓）
                          collectives.py / p2p.py（hccl 仓）
        │  参数校验、指针提取、dtype 推断、异常翻译
        ▼
hccl4py ctypes 原型层     _hcomm_proto.py / _hccl_proto.py
        │  CDLL 模块级缓存 + argtypes/restype 预绑定
        ▼
C 动态库                  libhcomm.so（通信域管理） / libhccl.so（通信算子）
```

分层职责单一：API 层面向用户（易用性、校验、对象封装），原型层面向 C ABI（零业务逻辑、开销最小化）。

## 包结构与双仓分发

hcomm 仓（通信域管理，5 接口）：

```
experimental/eco_system/python/
├── hccl4py/                        # namespace package（无顶层 __init__.py，PEP 420）
│   ├── _hcomm_proto.py             # libhcomm.so 加载 + 5 接口 ctypes 原型
│   ├── types.py                    # HcclResult / HcclDataType / HcclReduceOp（IntEnum）
│   ├── root_info.py                # RootInfo 封装（4108 字节透明结构）
│   ├── comm_config.py              # CommConfig 封装 + config_init 复刻
│   ├── comm_mgmt.py                # get_root_info / init_rank_config / init_cluster_config / destroy + HcclComm 类
│   └── exceptions.py               # HcclError
├── setup.py                        # 发行名 hccl4py-hcomm
├── README.md
├── examples/                       # 2 卡通信域创建/销毁样例（rootInfo 与 ranktable 两种方式）
└── test/                           # 通信域用例 + 测试步骤 README
```

hccl 仓（通信算子，14 接口）：

```
experimental/eco_system/python/
├── hccl4py/
│   ├── _hccl_proto.py              # libhccl.so 加载 + 14 算子 ctypes 原型
│   ├── types.py / exceptions.py    # 与 hcomm 仓同名同内容（双仓各自携带、字节级一致）
│   ├── buffer.py                   # 设备指针提取 helper
│   ├── send_recv_item.py           # HcclSendRecvItem / SendRecvItem 封装
│   ├── collectives.py              # 11 个集合通信算子
│   └── p2p.py                      # send / recv / batch_send_recv
├── setup.py                        # 发行名 hccl4py-hccl
├── README.md
├── examples/01_basic/              # 01_allreduce.py … 14 个算子样例
└── test/                           # 14 算子功能用例 + C/Python 性能对比
```

**双仓共存机制**：两目录均不含顶层 `__init__.py`，利用 Python 3.3+ 原生 namespace package 合并；`pip install hccl4py-hccl hccl4py-hcomm` 后 `import hccl4py` 自动聚合全部子模块。单仓使用场景（仅算子或仅通信域）直接设置 `PYTHONPATH` 指向对应目录即可，无需安装。`types.py`/`exceptions.py` 双仓各持一份且内容一致，安装顺序无关、互不覆盖。

## 模块设计

### types：枚举与数据类型映射

- `HcclResult(IntEnum)`、`HcclDataType(IntEnum)`、`HcclReduceOp(IntEnum)`：取值与 `hccl_types.h` 逐项一致（含 `HCCL_E_IN_STATUS=1041`、`HCCL_DATA_TYPE_RESERVED=255` 不对外）。
- dtype 推断表：`numpy.dtype`（fp16→HCCL_DATA_TYPE_FP16 等）与 torch dtype → `HcclDataType`；仅当调用方传 `data=` 而未显式传 dataType 时使用，bfp16/hif8/fp8 系无标准 numpy 对应，须显式传枚举。

### comm_mgmt：通信域管理（hcomm 仓）

`HcclComm` 句柄封装：

```python
class HcclComm:
    handle: int        # void* 句柄
    rank: int          # init 时传入并缓存
    rank_size: int     # init 时传入并缓存（阶段一 19 接口外不再调用查询接口）
    def __enter__ / __exit__   # with 语法，退出自动 destroy
    def __del__                # 兜底 destroy（best-effort，仅提示不抛异常）
```

`RootInfo` 封装：ctypes 结构体镜像 `char internal[4108]`，提供 `to_bytes()/from_bytes()`，root rank 将 4108 字节经任意进程间通道（pickle/socket/文件）分发给非 root。

`CommConfig` 封装与 **config_init 复刻**（本设计关键点）：

- C 侧 `HcclCommConfigInit` 为头文件 static inline，非动态库导出符号，ctypes 无法 dlsym —— Python 侧按头文件实现逐字段复刻：
  - `reserved[24]` 头：`size_t size=sizeof(HcclCommConfig)`、`uint32 magicWord=0xf0f0f0f0`、`uint32 version=11`、`uint64 reserved=0`；
  - 字段默认值：`hcclBufferSize/hcclDeterministic/hcclRdmaTrafficClass/hcclRdmaServiceLevel/hcclQos/hcclChannelSqDepth = 0xffffffff`、`hcclExecTimeOut=-1`、`hcclSymWinMaxMemSizePerRank=16(GB)`、字符串字段置空、其余置 0。
- `CommConfig` 以 `@dataclass` 暴露文档字段（buffer_size、deterministic、comm_name、op_expansion_mode、exec_timeout 等），构造时自动完成上述初始化，用户只需覆盖关注项。
- 漂移防护：单测锁定 `sizeof(HcclCommConfig)` 与各字段偏移（以 CANN 9.2.0 头文件为基线），上游 version 字段变更时同步更新并留存对照。

### buffer：指针提取约定

`send_data/recv_data/buf` 等按以下优先级提取地址（一次转换、零拷贝）：

1. `int` → 直接视为设备地址（零开销路径，供性能基线与高级用户）；
2. 具备 `data_ptr()` 方法的对象（torch_npu Tensor 等）→ 调用取地址；
3. `numpy.ndarray` → `.ctypes.data`（Host 地址，仅适用于 Host buffer 场景，如样例演示）。

counts/displs/sendCountMatrix 类数组参数：接受 `list[int]` 或 numpy 数组，内部一次性构造 `ctypes.c_uint64 * n` 并传首地址（C 原型为 `const void*`，与头文件 uint64 数组语义对齐）；批量场景支持调用方直传预构造 ctypes 数组以消除转换开销。Device 内存分配/释放归调用方（`aclrtMalloc`/torch_npu），hccl4py 不引入自有内存池。

### _proto：ctypes 原型层（性能关键路径）

- 模块加载时一次性 `ctypes.CDLL('libhccl.so' / 'libhcomm.so')` 并缓存全部函数对象；19 个接口全部显式声明 `argtypes`/`restype`，消除 ctypes 运行时类型推断开销。
- `HcclComm`/`aclrtStream`/`HcclRootInfo*`/`HcclCommConfig*` 一律 `c_void_p`/`POINTER`；返回 `HcclResult` 映射 `c_int`。
- ctypes 调用期间自动释放 GIL，多线程 Python 程序不被单次调用阻塞；线程安全语义与 C 层一致。

## 接口设计（19 个对照表）

通信域管理（hcomm 仓 `hccl4py.comm_mgmt` / `comm_config`）：

| Python 接口 | C 接口 | Python 返回值 |
| --- | --- | --- |
| `get_root_info()` | `HcclGetRootInfo(rootInfo)` | `RootInfo` |
| `init_rank_config(nranks, root_info, rank, config=None)` | `HcclCommInitRootInfoConfig(nRanks, rootInfo, rank, config, comm)` | `HcclComm` |
| `init_cluster_config(cluster_info, rank, config=None)` | `HcclCommInitClusterInfoConfig(clusterInfo, rank, config, comm)` | `HcclComm` |
| `CommConfig()`（构造即完成 config_init 语义） | `HcclCommConfigInit(config)` | — |
| `destroy(hccl_comm)` | `HcclCommDestroy(comm)` | `status` |

说明：任务书 19 接口清单中的 init 均为 Config 变体；`config=None` 时绑定层自动构造默认 `CommConfig` 传入——C 侧实现（`op_base.cc`/`op_base_v2.cc`）对 nullptr config 直接报错，且要求 magic word 必须经 `HcclCommConfigInit` 初始化，故不接受 `nullptr`；`cluster_info` 为 rank table 文件路径字符串。

通信算子（hccl 仓 `hccl4py.collectives` / `hccl4py.p2p`，`stream` 均缺省 `None`→`nullptr`）：

| Python 接口 | C 接口 | Python 返回值 |
| --- | --- | --- |
| `all_reduce(comm, send_data, recv_data, count, data_type, reduce_op)` | `HcclAllReduce` | `status` |
| `reduce(comm, send_data, recv_data, count, data_type, reduce_op, root)` | `HcclReduce` | `status` |
| `broadcast(comm, buf, count, data_type, root)` | `HcclBroadcast` | `status` |
| `all_gather(comm, send_data, recv_data, send_count, data_type)` | `HcclAllGather` | `status` |
| `all_gather_v(comm, send_data, send_count, recv_data, recv_counts, recv_displs, data_type)` | `HcclAllGatherV` | `status` |
| `scatter(comm, send_data, recv_data, recv_count, data_type, root)` | `HcclScatter` | `status` |
| `reduce_scatter(comm, send_data, recv_data, recv_count, data_type, reduce_op)` | `HcclReduceScatter` | `status` |
| `reduce_scatter_v(comm, send_data, send_counts, send_displs, recv_data, recv_count, data_type, reduce_op)` | `HcclReduceScatterV` | `status` |
| `all_to_all(comm, send_data, send_count, send_type, recv_data, recv_count, recv_type)` | `HcclAlltoAll` | `status` |
| `all_to_all_v(comm, send_data, send_counts, sdispls, send_type, recv_data, recv_counts, rdispls, recv_type)` | `HcclAlltoAllV` | `status` |
| `all_to_all_vc(comm, send_data, send_count_matrix, send_type, recv_data, recv_type)` | `HcclAlltoAllVC` | `status` |
| `send(comm, send_data, count, data_type, dest_rank)` | `HcclSend` | `status` |
| `recv(comm, recv_data, count, data_type, src_rank)` | `HcclRecv` | `status` |
| `batch_send_recv(comm, items)` | `HcclBatchSendRecv(sendRecvInfo, itemNum, comm, stream)` | `status` |

`items` 为 `SendRecvItem(type, buf, count, data_type, remote_rank)` 列表（封装 `HcclSendRecvItem` 五元组），绑定层一次展开为连续 ctypes 数组。

## 关键流程

2 卡通信域创建（rootInfo 方式，单机双卡双进程）：

```
rank0: root = get_root_info();  broadcast root.to_bytes()   # pickle/socket/文件
       comm0 = init_rank_config(2, root, 0)
rank1: b    = <接收 4108 字节>
       comm1 = init_rank_config(2, RootInfo.from_bytes(b), 1)
```

算子调用路径：Python 侧轻量校验（comm 类型、枚举范围、数组长度=rank size）→ 指针/数组提取 → 预绑定原型调用 → 返回 status。异步语义与 C 完全一致：接口仅向 stream 入队，同步由调用方 `aclrtSynchronizeStream`（或 torch_npu 等价 API）完成，样例中显式演示同步时机。

## 错误处理

- 通信算子与 `destroy` 返回 `status`（int，`HcclResult` 值），与任务书 §3.1.3 举例一致；init 类接口 C 出参为 comm 指针，Python 侧成功返回 `HcclComm`、失败抛 `HcclError`（携带 HcclResult 与 `hccl4py.strerror(status)` 可读信息）。
- 提供模块级 `hccl4py.set_raise_on_error(bool)` 全局开关与 `hccl4py.check(status)` 显式检查 helper：脚本/框架默认异常模式，性能基准测试切换 status 模式避免异常机制干扰测量。
- Python 侧参数错误（类型、长度、范围）抛 `TypeError`/`ValueError`，不进入 C 层；C 层返回的非零码不二次翻译，原值透传。

## 性能设计（<10us 预算）

| 开销项 | 典型量级 | 对策 |
| --- | --- | --- |
| ctypes FFI 调用 | ~0.5–2us | argtypes/restype 预绑定；句柄 c_void_p 直传 |
| 指针提取 | ~0.1us（int 路径） | int 直传零转换；tensor `data_ptr()` 单次调用 |
| 数组转换（V 类接口） | ~1–3us/数组 | 一次性 ctypes 数组构造；支持调用方预构造直传 |
| Python 参数校验 | <1us | 仅 O(1) 必要校验，无逐元素扫描 |
| 合计目标 | <5us 典型路径，10us 上限 | |

基准方法：同逻辑用例的 C 版（参照 hccl 仓 `examples/02_collectives` C 样例）与 Python 版各执行 N>1000 次取均值与分布，测量点为接口调用前后 host 单调时钟差（C/Python 同口径，不含 stream 同步），输出 C/Python 双列耗时对比表；抖动以标准差/均值计。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A3 训练/推理系列 | √（同接口族，绑定层不区分型号） |
| Ascend 950PR/DT 系列 | √（任务书验收款型，自测以此为准） |

任务书 2026-10 变更：验收硬件锁定 A5 系列，CANN 9.1.0 及以上；经三版本头文件比对
（9.1.0 / 9.2.0-beta.1 / 9.2.0），本任务 19 个接口的类型布局、枚举与签名完全一致，
绑定层与测试矩阵按 Ascend 950 列官方 dataType 矩阵覆盖（归约类 9 种、ReduceScatterV
7 种、非归约类 16 种含 hif8/fp8 系）。

## 约束限制

- Device 内存生命周期归调用方；hccl4py 不做分配、不持有、不缓存 buffer。
- 不改变 C 接口语义：count 按 data_type 元素个数；V 类接口数组长度等于 rank size；AlltoAllVC 矩阵为 nRanks×nRanks。
- 阶段一范围限定任务书 19 接口；hcomm 其余接口族（零拷贝、对称内存、组调用、suspend/resume 等）不在本期。
- Python ≥3.7，仅依赖标准库 ctypes；numpy 为可选增强（缺失时 int 地址与预构造 ctypes 数组路径仍完整可用）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 2 卡内通信域创建/销毁（rootInfo 与 ranktable 两种方式）；14 算子遍历支持数据类型运行，输出与 numpy/torch golden 按算子语义逐元素比对；V 类接口覆盖非均匀 counts/displs；Send/Recv/BatchSendRecv 覆盖双向与多批次 | 任务书 §5.2；配套 `test_cases/README.md` 用例分类 |
| 性能标准 | Python 相对 C 接口增加耗时 <10us，抖动 <10%；交付 C/Python 双列耗时对比表 | 任务书 §5.2 |

测试工程（后续代码 PR 交付）：

- hccl 仓 `experimental/eco_system/python/test/`：pytest 功能用例（按 14 算子分文件）+ 性能对比脚本（`bench_c_vs_python.py`）+ `run_all.sh` + README（环境变量、2 卡双进程拉起方式、复现步骤）。
- hcomm 仓 `experimental/eco_system/python/test/`：通信域创建/销毁/配置用例，结构同构。
- 拉起方式：单机 2 卡双进程脚本（每进程绑定 `ASCEND_DEVICE_ID`），样例与用例统一经同一 runner 拉起；README 给出逐条命令保证验收人可复现。

## 兼容性分析

- 纯新增目录，不改动两仓任何既有文件、构建目标与导出符号；hcomm 仓既有 `python/hccl/`（Hcom* legacy whl）构建路径不受影响。
- 绑定层仅依赖 Python 标准库 ctypes，无编译环节；`hccl_types.h` 常量（magicWord/version/字段偏移）由单测守护，CANN 版本升级时以头文件 diff 为准同步。
- 接口命名与任务书 §3.1.3 契约一致，后续阶段（更多接口族、性能增强）在同名包内增量演进，不破坏本期 API。

## 风险评估

| 风险 | 应对 |
| --- | --- |
| ctypes 开销实测超 10us | int 直传零开销路径 + 预绑定已将典型路径控制在预算内；若特定接口（AlltoAllV 多数组）仍超，提供预构造 ctypes 数组的绕行用法并写入 README；极端情况备选薄 C 扩展（仅替换原型层，API 层不变） |
| HcclCommConfigInit 复刻漂移（上游 version/布局变更） | 单测锁定 sizeof/偏移/magic 基线；升级流程以头文件 diff 驱动 |
| 双仓 namespace package 误用（用户只装一仓后 import 全量子模块） | README 明确单仓可用范围与 import 验证语句；`__getattr__` 惰性导入给出缺失子模块的安装指引 |
