# RFC: hccl4py —— HCCL 通信算子的 Python 绑定

- 起始日期：2026-10-18
- RFC PR编号：[cann/hccl !3657](https://gitcode.com/cann/hccl/merge_requests/3657)
- 相关Issue：[cann/hccl #824](https://gitcode.com/cann/hccl/issues/824)
- 模板依据：以[社区任务设计模板](../../../resources/design_template.md)的必选章节为评审基准，RFC 章节作为补充

---

## 概要

为 HCCL 的 14 个通信算子 C API（`include/hccl.h`）提供 Python 绑定包 **hccl4py_ops**，
使 Python 用户无需手写 ctypes 即可使用集合通信与点对点通信能力；接口语义与 C 保持一致，
Python 封装引入的额外耗时要求 ≤10us。配套提交自测用例与使用样例。

通信域管理与基础类型层由 hcomm 仓提供，
本 RFC 只覆盖算子侧。

## 需求分析（required）

### 需求描述

在 hccl 仓 `experimental/eco_system/python/` 下交付纯 Python 包 `hccl4py_ops`，覆盖 14 个
通信算子 C API，并提供可复现的 examples、测试、测试报告与性能数据。绑定必须保持 C 接口的
缓冲区、stream、通信域和返回码语义；Python 相对直接 C 调用的单次 host 下发额外耗时不超过
10μs，抖动不超过 10%。

### 需求拆解

1. 接口：14 个算子全部可用，Python 参数名、顺序、计数单位、dtype 和返回码有明确契约；
2. 内存：只接收 device 内存，不隐式执行 H2D/D2D 拷贝；`(device 指针, count, dtype)` 语义可校验；
3. 异步安全：V/VC/Batch 接口构造的 host 参数在 stream 消费完成前保持有效；
4. 兼容：9.1.0 与 9.2.0 的 14 个算子原型兼容，通信域配置按实际加载的 so 做校验；
5. 工程：两仓源码随仓分发，通过 `PYTHONPATH` 组合；跨包只依赖公开 API；
6. 验收：默认覆盖目标环境官方 dtype 支持矩阵，提供双卡正确性、生命周期和性能报告。

## 运行与验收环境

本文统一采用以下口径，并与 hcomm 侧文档保持一致：

| 项目 | 统一口径 | 说明 |
|---|---|---|
| 任务兼容下限 | A5 系列 + CANN 9.1.0 + pyACL | 9.1.0 上执行导入、零拷贝绑定和单卡冒烟；若任务方指定 9.1.0 为最终验收环境，则必须执行同一套全量测试 |
| 默认正式验收基线 | A5 系列 + CANN 9.2.0 + pyACL + 单机 2 卡 | 功能、全量 dtype、rankSize=2 双卡与性能指标默认在此环境验收 |
| 版本一致性 | 头文件、`libhccl.so`、`libhcomm.so`、`libascendcl.so`、pyACL 来自同一个 CANN/驱动安装根目录 | 不允许默认混用两个安装根目录 |
| 测试材料关系 | 当前测试材料中的 A3/950 描述不作为本设计验收依据 | 交付前由任务方/测试负责人同步为 A5 + CANN 9.2.0；若改变验收硬件，设计和测试须同时变更 |

公开标签 `v9.1.0` 与 `v9.2.0` 的核对结果是：`include/hccl.h` 中 14 个算子原型未发生
实质性变化，差异为格式与注释；`HcclDataType` 的有效枚举值也一致。差异集中在
`HcclCommConfig`：9.1.0 为 version10/2240B，9.2.0 为 version11/2248B，并新增
`hcclChannelSqDepth`。因此算子侧可以支持 9.1.0 下限，但版本兼容不能只比较版本号字符串。

运行期执行两级检查：

1. 解析实际加载的 `libhccl.so`，要求 14 个符号全部存在；
2. 将同一安装根目录下 `hccl.h` 的 14 个函数原型归一化为 `_abi.py` 中检查入仓的签名指纹，
   与 9.1.0/9.2.0 公共 profile 精确比较。任一参数类型或顺序不一致即拒绝导入，并报告
   `libhccl.so` 和头文件路径。

头文件原型归一化由 hcomm 基础包的公开 `hccl4py.api.abi` 提供，hccl 侧不另写解析器：

* 解析器仅使用 Python 标准库，先去除注释和预处理干扰，再用受限词法/括号扫描提取目标函数声明；
* 参数名、`const/volatile/restrict`、可见性/属性宏不进入指纹；固定宽度整数、typedef 和枚举别名
  归一为稳定类型名，保留返回类型、参数顺序和指针层级；
* 原始声明 `HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, ...)`
  归一为 `("HcclAllReduce","HcclResult",("void*","void*","uint64_t",...))`；
* 任一头文件缺失、宏无法求值、目标声明找不到、参数解析不唯一或 profile 不匹配，均抛出
  `AbiCompatibilityError` 并拒绝导入，禁止按版本号或符号存在性静默降级；
* 无 device 自测包含多行原型、块注释、`extern "C"`、属性宏和无法解析宏的扰动样例。

`HcclGetCommConfigCapability()` 用于校验配置字段能力，不作为结构体版本号使用；通信域配置
的头文件预判与 so 侧校验见 HCOMM 设计文档。

## 背景与动机

* HCCL 目前只提供 C 接口，Python 用户需自行用 ctypes 手写绑定，易错且难维护；
* 自研训练框架、HPC 脚本、模型调优工具需要直接调用 HCCL 通信能力；
* 需要一个薄、稳定、与 C 语义一致的 Python 绑定，作为生态接入的公共底座。

## 术语表

| 术语 | 含义 |
|------|------|
| hccl4py 项目 | 任务书要求的 Python 绑定库总称 |
| hccl4py_ops | 本 RFC 交付的算子绑定包（本仓） |
| hccl4py 基础包 | hcomm 仓交付的类型层和通信域接口 |
| 缓冲区描述 | `(device 指针, count, data_type)` 三元组；指针必须指向 device 内存 |
| host 辅助参数 | V/VC/Batch 接口使用、由 Python 构造并作为数组指针传入的 counts/displs/matrix/item 数据 |

## 架构与接口契约

### 整体架构

```text
Python 应用 / 框架 / 脚本
        │
   hccl4py_ops（本仓 experimental/eco_system/python）
        │  只依赖公开接口 hccl4py.api
   hccl4py（hcomm 仓 experimental/eco_system/python）
        │
   libhccl.so（14 个算子）        libhcomm.so（通信域）
```

### 包与分发

第一阶段采用“源码随仓 + `PYTHONPATH`”作为唯一验收分发形态：

```text
<hcomm>/experimental/eco_system/python/  # hccl4py 包
<hccl>/experimental/eco_system/python/   # hccl4py_ops 包
```

README 规定设置顺序：

```bash
export PYTHONPATH=<hcomm>/experimental/eco_system/python:<hccl>/experimental/eco_system/python
python -c "import hccl4py, hccl4py_ops"
```

`hccl4py` 在 `api` 初始化完成后通过模块级 `__getattr__` 惰性导出 `hccl4py_ops`，避免两个包
以相反顺序导入时形成循环；只安装 hcomm 侧时通信域接口仍可用。`hccl4py_ops` 不访问
`_ffi`、`_buffer`、`comm` 等下划线私有模块。
正式 wheel、独立版本号和 pip 发布顺序是后续工作，不影响当前双仓验收复现。

### 对外接口与返回值

14 个算子接口与 C API 一一对应。`count`/`data_type` 保留在对应位置；当 buffer 是 NPU Tensor
且语义可唯一推导时，可显式传入 `None` 由绑定层补齐。裸 device 指针不得省略这两项。

| C API | Python 签名 | 落位包 | 验收用例 | 计数/类型推导 |
|---|---|---|---|---|
| `HcclAllReduce` | `all_reduce(send_buf, recv_buf, count, data_type, reduce_op, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_reduce` | Tensor 可推导 |
| `HcclReduce` | `reduce(send_buf, recv_buf, count, data_type, reduce_op, root, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_reduce` | Tensor 可推导 |
| `HcclBroadcast` | `broadcast(buf, count, data_type, root, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_broadcast` | Tensor 可推导 |
| `HcclAllGather` | `all_gather(send_buf, recv_buf, send_count, data_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_gather` | Tensor 可推导 |
| `HcclAllGatherV` | `all_gather_v(send_buf, send_count, recv_buf, recv_counts, recv_displs, data_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_gather_v` | dtype 可推导；计数必须显式 |
| `HcclScatter` | `scatter(send_buf, recv_buf, recv_count, data_type, root, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_scatter` | Tensor 可推导 |
| `HcclReduceScatter` | `reduce_scatter(send_buf, recv_buf, recv_count, data_type, reduce_op, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_reduce_scatter` | Tensor 可推导 |
| `HcclReduceScatterV` | `reduce_scatter_v(send_buf, send_counts, send_displs, recv_buf, recv_count, data_type, reduce_op, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_reduce_scatter_v` | dtype 可推导；计数必须显式 |
| `HcclAlltoAll` | `all_to_all(send_buf, send_count, send_type, recv_buf, recv_count, recv_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_to_all` | 计数必须显式；dtype 可推导 |
| `HcclAlltoAllV` | `all_to_all_v(send_buf, send_counts, send_displs, send_type, recv_buf, recv_counts, recv_displs, recv_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_to_all_v` | dtype 可推导；计数必须显式 |
| `HcclAlltoAllVC` | `all_to_all_vc(send_buf, send_count_matrix, send_type, recv_buf, recv_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_to_all_vc` | dtype 可推导 |
| `HcclSend` | `send(send_buf, count, data_type, dest_rank, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_send_recv` | Tensor 可推导 |
| `HcclRecv` | `recv(recv_buf, count, data_type, src_rank, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_send_recv` | Tensor 可推导 |
| `HcclBatchSendRecv` | `batch_send_recv(items, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_batch_send_recv` | 每项独立指定 |

`send_recv_item` 的五个字段与 `HcclSendRecvItem` 一一对应：

```python
send_recv_item(send_recv_type, buf, count, data_type, remote_rank)
```

其中 `send_recv_type` 为 `HCCL_SEND` 或 `HCCL_RECV`，`remote_rank` 是通信域内对端 rank。
`batch_send_recv` 将 Python item 列表一次性转换为 `ctypes` 连续结构体数组，并登记到保活队列。

默认返回 `HcclResult`。调用 `set_raise_on_error(True)` 后，非 0 结果抛出 `HcclError`，异常包含
错误码、C API 名、`HcclGetErrorString()` 文案以及通信域/stream 的调试信息。`all_reduce` 等
算子不会把出参改成隐式 Python 返回值，避免与 C 的异步语义混淆。

### Buffer 支持矩阵

| 输入 | 是否接受为数据 buffer | 指针/元数据规则 |
|---|---|---|
| `int` / `ctypes.c_void_p` | 是，基准路径 | 视为 device 地址；`count`、`data_type` 必填且非空 |
| 实现 `data_ptr() -> int` 的 device 对象 | 是 | `data_ptr()` 必须非 0；元数据必填 |
| `torch.Tensor` | 是，仅 `device.type == "npu"` 且连续 | `data_ptr()` 用于透传；缺省时从 tensor 推导 dtype/count |
| `torch.Tensor`（CPU） | 否 | 立即抛出 `TypeError`，错误信息提示先迁到 NPU |
| `numpy.ndarray` | 否 | 不受理为数据 buffer；NumPy 内存属于 host，禁止把其指针传给通信算子 |
| 任意 host 内存 | 否 | 绑定层不执行 H2D/D2D 拷贝，也不猜测 residency |

裸指针无法在 Python 侧证明 residency，仍由调用方按 C ABI 保证；该限制属于显式契约，而不是
静默接受。所有 buffer 的所有权归调用方，绑定层不申请、不释放、不迁移 device 数据。

非连续 NPU Tensor 默认拒绝，并提示 `tensor.contiguous()`。绑定层不做隐式连续化，因为隐式拷贝
既改变值生命周期，也可能突破 ≤10μs 预算。空 Tensor、`data_ptr()==0`、负 count 或超出
`uint64` 的 count 均拒绝。

### dtype 映射

`_require_dtype()` 在导入期构建不可变映射表，调用期只做字典查找。无法无损表达的 dtype 不从
Tensor 猜测，调用方改用裸 device 指针并显式传入 `HcclDataType`。

| `HcclDataType` | 值 | PyTorch 自动推导 | NumPy 等价 dtype（仅供元数据说明） |
|---|---:|---|---|
| `HCCL_DATA_TYPE_INT8` | 0 | `torch.int8` | `np.int8` |
| `HCCL_DATA_TYPE_INT16` | 1 | `torch.int16` | `np.int16` |
| `HCCL_DATA_TYPE_INT32` | 2 | `torch.int32` | `np.int32` |
| `HCCL_DATA_TYPE_FP16` | 3 | `torch.float16` | `np.float16` |
| `HCCL_DATA_TYPE_FP32` | 4 | `torch.float32` | `np.float32` |
| `HCCL_DATA_TYPE_INT64` | 5 | `torch.int64` | `np.int64` |
| `HCCL_DATA_TYPE_UINT64` | 6 | `torch.uint64`（存在时） | `np.uint64` |
| `HCCL_DATA_TYPE_UINT8` | 7 | `torch.uint8` | `np.uint8` |
| `HCCL_DATA_TYPE_UINT16` | 8 | `torch.uint16`（存在时） | `np.uint16` |
| `HCCL_DATA_TYPE_UINT32` | 9 | `torch.uint32`（存在时） | `np.uint32` |
| `HCCL_DATA_TYPE_FP64` | 10 | `torch.float64` | `np.float64` |
| `HCCL_DATA_TYPE_BFP16` | 11 | `torch.bfloat16` | 无标准 dtype |
| `HCCL_DATA_TYPE_INT128` | 12 | 无标准映射 | 无标准 dtype |
| `HCCL_DATA_TYPE_HIF8` | 14 | 无标准映射 | 无标准 dtype |
| `HCCL_DATA_TYPE_FP8E4M3` | 15 | `torch.float8_e4m3fn`（存在时） | 无标准 dtype |
| `HCCL_DATA_TYPE_FP8E5M2` | 16 | `torch.float8_e5m2`（存在时） | 无标准 dtype |
| `HCCL_DATA_TYPE_FP8E8M0` | 17 | 无标准映射 | 无标准 dtype |
| `HCCL_DATA_TYPE_RESERVED` | 255 | 非法 | 非法 |

表中的 Torch FP8 名称是候选映射，只有在目标 torch_npu 明确登记与 HCCL 位语义等价时才启用；
否则按“无标准映射”处理。

若 Tensor dtype 不在映射中且 `data_type=None`，抛出 `TypeError` 并列出全部可推导 dtype；
若调用方显式传入 enum，绑定层仍拒绝无法证明存储语义的 Tensor，只允许裸 device 指针。
NumPy 列只用于解释类型对应关系，不代表 NumPy 数组可作为 device buffer。

### count、shape 与校验

* `count`/`counts` 的单位是元素个数，不是字节；displacement 也是元素偏移；
* counts/displs 长度必须等于公开属性 `comm.rank_size`；AlltoAllVC 矩阵长度必须为
  `rank_size * rank_size`；
* `send_count_matrix[i * rank_size + j]` 表示 rank i 发给 rank j 的元素数。二维 list/tuple
  按行优先展平，底层只创建一块连续的 `uint64[rank_size * rank_size]`；
* Tensor 标量 count 缺省时取参与计数缓冲区的 `numel()`；AllGatherV、ReduceScatterV 和
  AlltoAll/AlltoAllV 的标量/V 参数不从总长度猜测，必须显式给出；
* 对 Tensor，校验 `count <= tensor.numel()` 及 `displ + count <= tensor.numel()`；裸指针只有
  `count` 元数据，容量无法检查，由调用方保证；
* shape 不进入 C API。Python 层只校验元素总量和连续性，不改变业务张量形状。

### stream 语义

`stream` 接受 `aclrtStream` 兼容的 `int`、`ctypes.c_void_p` 或实现 `data_ptr()` 的对象。
`stream=None` 时调用 `aclrtGetCurrentStream()` 获取当前 device 的 current stream；取不到或为空
则抛出 `HcclError`，不静默使用默认 0。调用方必须先 `aclrtSetDevice()`，同一通信域的下发须在
同一 device/context 内串行执行，符合 HCCL C 接口约束。current stream 随 thread/device context
变化，不做跨线程缓存；该查询的固定成本计入性能预算。`libascendcl.so` 因此是算子绑定的运行期
依赖，而不是仅测试依赖。

### 异步参数生命周期

所有算子均为异步下发。device buffer 和 comm 至少保持到对应 stream 完成，这一责任与 C API
一致。V/VC/Batch 的 host 辅助参数采用额外的显式保活机制：

1. 每次调用从 Python 序列构造成新的 `ctypes` 数组/结构体数组，不复用旧参数数组；
2. `_lifetime.py` 从按 device 维护的 event 池获取已创建 event；仅池空时调用 `aclrtCreateEvent`；
3. C API 成功返回后，在同一 stream 上记录 `aclrtEvent`；
4. `(event, arrays, metadata)` 登记到与 `(comm_id, stream)` 关联的 pending 队列；
5. 后续调用、`synchronize(stream)` 或 `release(comm, stream)` 时查询 event；完成后回收数组并把
   event 放回池中复用；
6. event 创建/记录失败时进入保守模式：数组保留到显式 `synchronize(stream)`/`release()`，并抛出
   带原始错误码的 `HcclLifetimeError`，绝不提前释放；
7. `comm_destroy` 不代替 stream 同步；销毁通信域前调用方必须同步所有相关 stream，或调用包提供
   的 `release(comm)` 完成清理。

非 V/VC/Batch 的裸指针快速路径不创建 host 数组，也不记录额外 event，避免为常规集合通信增加
固定生命周期开销。AlltoAllV 不得再“复用相同 counts/displs 的旧对象”；每次调用都生成独立数组，
从设计上消除调用返回后参数被覆盖的悬垂风险。

### 依赖接口

* 只依赖公开入口 `hccl4py.api` 提供的 `Comm`（含只读 `rank_size`）、`BufferLike`、`HcclDataType`、`HcclResult`、
  `HcclError`、`LibInfo` 和库加载/能力查询函数；
* 运行期依赖 `libhccl.so` 的 14 个导出符号；加载后执行符号与原型指纹检查；
* `libascendcl.so` 是运行期依赖，用于 `aclrtGetCurrentStream()` 及 V/VC/Batch event 管理；
  ACL 原型统一通过 `hccl4py.api.acl` 获取，不在算子模块重复定义。

## 影响分析

* 对性能：绑定主路径只做一次元数据查找、指针规整和 ctypes 调用；最终以双卡 Batch 实测判定；
* 对现有功能：新增实验目录，不修改 `src/`、`include/`，不影响存量编译与发布；
* 对构建：交付包为纯 Python，无运行期编译产物；`tools/` 下 C/C++ 仅用于一次性布局和性能对照，
  不进入 Python 包、不随 wheel 发布。

## 兼容性考虑

* 不改变任何 C 接口与头文件，向后兼容；
* 14 个算子原型在 CANN 9.1.0 与 9.2.0 间保持兼容；9.2.0 是正式验收基线，9.1.0 是运行下限；
* 运行期以实际 `libhccl.so` 的符号集合和同安装根目录头文件的签名指纹为准，不靠版本号字符串猜测；
* 后续 CANN 版本新增参数/字段时，必须先更新 ABI profile 与测试；未登记 profile 明确拒绝导入；
* 实验目录代码不承诺 API 稳定，后续可平滑迁入正式目录。
* 本 RFC 新增独立 Python 包和实验目录，不涉及在线服务变更，不需要特性开关或灰度发布策略。

## 详细设计

### `hccl4py_ops/__init__.py`

- **模块职责**：定义算子包的公开导出。
- **核心数据**：`__all__` 中的 14 个算子函数和 `send_recv_item` 构造器。
- **关键逻辑**：从 `ops.py` 导入并统一导出公共接口，不包含 ctypes 调用逻辑。
- **内部接口**：只依赖 `hccl4py.api`；不导入 hcomm 仓任何下划线模块。

### `hccl4py_ops/_abi.py`

- **模块职责**：维护 9.1.0/9.2.0 的 14 个函数原型指纹、dtype 枚举和必需符号清单。
- **核心数据**：类型名归一化后的签名 tuple、允许的 CANN profile、loaded lib/header 路径。
- **关键逻辑**：加载 `libhccl.so` 后解析同一安装根目录的 `hccl.h`，与入仓 profile 比较；
  同时检查 14 个符号可解析。失败信息必须包含 so 路径、头文件路径和首个不一致参数。
- **解析与降级**：调用 hcomm 暴露的公开原型解析/指纹 helper，不复制私有解析代码；任何头文件
  缺失、宏无法求值、声明缺失或解析歧义均 fail-closed，禁止退化为仅检查版本号。
- **内部接口**：向 `ops.py` 提供已验证的 `FunctionPrototype`，调用期不再解析头文件。

### `hccl4py_ops/ops.py`

- **模块职责**：实现 14 个 HCCL 通信算子的 ctypes 绑定和参数规整。
- **核心数据结构**：`BufferDesc(ptr, count, data_type)`、`HcclSendRecvItem`、host
  `uint64[]`/item 数组、通信域句柄和 ACL stream。
- **关键逻辑**：导入期创建并缓存独立 `CFUNCTYPE` 原型；快速路径先判断裸指针 + 显式元数据，
  命中后不读取 Tensor；统一执行 residency、连续性、rank、count、dtype 校验；V/VC/Batch
  构造新数组并交给 `_lifetime.py` 保活；返回值保持 `HcclResult`，可选转换异常。
- **内部接口**：公开 `describe_buffer()`/`ptr_of()` 语义由 `hccl4py.api` 提供；
  `_f()`、`_finish()`、`_comm()`、`_stream()`、`_require_dtype()` 保持模块私有。

### `hccl4py_ops/_lifetime.py`

- **模块职责**：管理 V/VC/Batch host 辅助参数从入队到 stream 完成的引用生命周期。
- **核心数据**：`pending[(comm_id, stream)]` 列表，每项为 `(event, arrays, debug_call_site)`；
  `event_pool[device_id]` 保存已完成、可复用 event。
- **关键逻辑**：从 event 池取 event；仅池空时创建。每次提交后在同一 stream 记录 event；
  调用入口按 O(已完成条目) 回收并把 event 放回池；`synchronize()`/`release()` 强制清空对应队列；
  event 创建/记录失败进入保守保留模式。
- **内部接口**：`retain()`、`reap()`、`synchronize()`、`release()`、`pending_count()`；
  以及测试专用 `event_pool_size()`；诊断接口不作为业务 API。

### 类型与错误策略

不使用 `HcclResult` 的隐式布尔判断；错误码、原始字符串和 Python 调用名分开保存。参数校验
错误使用 `TypeError`/`ValueError`；已经调用 C API 后得到的非 0 返回保持 `HcclResult` 或转换
`HcclError`。这种区分便于判断“未下发”和“下发失败”，也避免把 Python 校验错误伪装成 HCCL 错误。

## 可维可测分析（required）

### 精度与功能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 接口一致性 | 14 个 Python 接口均可调用，参数顺序、元素计数、dtype、返回码与 C API 一致 | 任务书 §3.1 |
| 数值正确性 | 整数结果与直接 C 调用逐 bit 一致；浮点结果优先要求同参数、同环境下的 C/Python 输出逐 bit 一致，数学 oracle 另按 dtype 容差检查 | 任务书“功能与 C 一致” |
| 非连续输入 | 明确拒绝，不影响真实 device 数据；提示调用方显式 `contiguous()` | 本设计 |
| 异步生命周期 | stream 未完成时，V/VC/Batch host 参数不被回收；完成后可回收 | 本设计，覆盖 C 异步语义 |

默认全量功能测试位于 `test/ecosystem/hccl4py/`，通信域只用 rootinfo 建一次并在所有算子间复用：

| 层次 | 用例 | 说明 |
|---|---|---|
| 绑定自检 | `test_binding_selfcheck.py` | 无 device：14 个符号、签名指纹、布局、17 个有效 dtype、raw/Tensor 规则、拒绝 NumPy/CPU Tensor |
| 跨仓导入顺序 | `test_public_api_contract.py` | 分别先导入 `hccl4py`/`hccl4py_ops`，验证 `__getattr__` 惰性导出、无私有模块访问 |
| 单卡冒烟 | `test_single_rank.py` | rankSize=1：通信域、stream 获取、14 个接口调用通路、错误与空指针路径 |
| 双卡功能 | `run_two_rank.py` | rankSize=2：在 A5/CANN 9.2.0 官方支持矩阵下遍历全部合法 dtype/reduceOp；覆盖不等长 V、VC 矩阵、Send/Recv 和 Batch |
| 生命周期专项 | `test_param_lifetime.py` | 调用 V/VC/Batch 后立即删除 Python list 并 `gc.collect()`，在 stream 未同步时继续下发多组，验证 event 池复用与 pending 回收，最后同步并校验结果 |
| 性能对比 | `perf_compare.py` | C/Python 同参数、同迭代顺序的 host 下发 Batch 对比 |

按 CANN 9.2.0 当前 A5（Ascend 950）类产品文档，默认矩阵至少有 292 个合法
`(operator, dtype, reduce_op)` 组合：4 个归约接口贡献 132 个，其余 10 个接口各覆盖 16 个 dtype，
共 160 个；另加不等长/矩阵布局/异常路径 8 个。测试一次性建域、逐用例同步，规划预算为
20–30 分钟，实际用例 ID、参数、耗时和截图写入测试报告。

`HCCL4PY_TEST_DTYPES` 只允许开发机缩小调试集合；变量存在时报告醒目标记 “DEVELOPMENT
SUBSET”，验收脚本在 CI/验收模式强制要求变量不存在。任何合法 dtype/reduceOp 缺失均判失败。

### 性能标准

| 指标 | 验收值 | 口径 |
|---|---:|---|
| Python 相对 C 的单次 host 下发增量 | 均值 ≤10μs | 同一 process pattern、同一参数、同一 C API；同步在计时区外 |
| Batch 抖动 | CV ≤10% | 保留全部 Batch 样本；任务书只写“抖动范围10%”，该统计量需与任务方确认 |
| 轮均值稳定性 | 最大偏差 ≤10% | 每轮样本均值相对总均值；备选口径为逐批 ΔT 相对其均值的最大偏差 |
| 含同步的端到端耗时 | 仅报告，不作为 10μs gate | 证明异步语义和用户可见时延 |

任务书原文要求“Python接口相对原C接口增加的耗时需在10us之内（抖动范围10%之内）”。上述 host
下发口径和 CV/轮偏差定义是设计对“抖动10%”的操作化解释，须在验收前由任务方确认唯一口径。
C 基准
`perf/hccl_ops_c.cpp` 直接调用全部 14 个原生 C API；Python 基准使用公开接口。两者使用相同
count、dtype、rank、stream、预热次数、Batch 大小和同步位置。每组连续调用 N 次，停止计时后
同步；组总耗时除以 N 为一个样本。只在 Python计时期间关闭循环 GC，结束后恢复。

性能路径按固定成本拆分，而不是只给最终结果：

| 路径 | 固定成本来源 | 设计措施 |
|---|---|---|
| 裸指针快速路径 | Python 参数检查、`aclrtGetCurrentStream`、指针转换、`CFUNCTYPE` 调用 | 原型导入期缓存；dtype 查表；current stream 不跨线程缓存；不读 Tensor、不分配 host 数组 |
| NPU Tensor 路径 | `device`/`is_contiguous`/`numel`/`data_ptr` 检查 | tensor 方法仅调用一次；结果直接组成 C 参数 |
| AlltoAllV 最差路径 | 10 参数编组、4 个 `uint64[]`、event 获取/记录/查询 | 每参数只复制一次；连续 ctypes 数组；event 从池复用，仅池空时创建 |

2026-10-20 在 Python 3.12.13 / GCC 12.3.1 构建主机上的无 device 指示性微基准（不是板端验收值）：

| 项目 | 中位耗时 |
|---|---:|
| `CFUNCTYPE(c_int64, 3×c_uint64)` 空函数调用 | 0.41μs |
| 7 参数空函数调用 | 0.80μs |
| 10 参数空函数调用 | 1.10μs |
| 构造 4 个 rank=2 的 `c_uint64[2]` | 1.04μs |

最差参数编组示例已占约 2.14μs，10μs 预算剩余约 7.86μs；Tensor 检查、
`aclrtGetCurrentStream`、event 获取/记录/查询与 pending 队列回收必须在目标 A5 环境分组实测，
不能用构建机数据替代。若上述任一组固定成本使路径超预算：

1. 先保留 API，启用已测量最快路径并对热点做 `py-spy`/`perf` 分析；
2. 去除不必要的属性访问、诊断字符串和每调用 closure；
3. 若仍超标，按任务书允许的“实测超预算”条件提供同等语义的可选 C 扩展 fast path；
4. C 扩展必须通过同一回归与 ABI 测试，且保留纯 Python fallback，不能无数据直接替换。

所有性能报告必须记录机器型号、CANN/pyACL 版本、实际 so 路径、git commit、CPU governor（如可配置）、
N、预热次数、轮数、完整原始样本和统计脚本版本。

### 兼容性分析

* 不修改 HCCL C API、头文件和 `.so` 符号；
* 9.1.0/9.2.0 的 14 个算子原型与 dtype 枚举一致；默认在 9.2.0 全量验收。若任务方确认 9.1.0 为最终
  验收环境，则 9.1.0 执行同一功能、双卡和性能全量套件，而不是只做冒烟；
* `HcclCommConfig` version10/11 由 hcomm 基础包按同一安装根目录的头文件和 so 能力选择；
  `HcclGetCommConfigCapability` 的官方文档明确为“标量上限 + 枚举值比较”，故对已有枚举的字段使用
  `capability > HCCL_COMM_CONFIG_<FIELD>`；Qos/SqDepth 没有独立枚举，改由 version/size/profile
  和首次建域校验覆盖，不宣称每个字段都经过 capability 判断；
* 源码分发只依赖 Python 标准库；`torch`/`torch_npu` 是可选互操作依赖，裸指针路径不导入它们；
* 实验目录不承诺 Python API 长期稳定，正式 wheel/API 稳定化不在本阶段范围。

## 风险评估

1. **host 辅助参数提前释放**：event 记录失败或调用方绕过 `release()` 可能导致风险。缓解：
   默认队列保活、失败转保守模式、专项强制 GC 测试；comm 销毁前强制提醒同步所有 stream；
2. **raw pointer residency 无法自动验证**：`int`/`c_void_p` 可能被误传 host 地址。缓解：契约明确
   为 device 地址；README/examples 只用 `_acl.DeviceBuffer` 或 NPU Tensor；不提供 NumPy 数据路径；
3. **dtype/shape 误用**：Tensor dtype 或 count 与 C 支持矩阵不相容。缓解：完整映射、逐项校验、
   对不支持映射的 Tensor 拒绝、默认全矩阵测试；
4. **ABI 漂移**：CANN 升级可能新增参数或结构字段。缓解：符号检查 + fail-closed 原型指纹 +
   未知 profile 拒绝导入；9.2.0 起任何变化同步更新 profile 与测试；
5. **性能抖动**：主机负载、current stream 查询和 event/数组构造影响指标。缓解：固定 Batch、
   GC 控制、event 池、CV/轮偏差、原始样本保留，并分别测快速和最差路径；事件管理成本超预算
   属于 fast path 回退触发条件；
6. **组网差异**：受限容器中 ranktable/真实 EID 不可用。缓解：默认 rootinfo，ranktable 验收只接受
   真实端口/EID，建域成功不能替代跨 rank 数据传输。

## 交付件与目录映射

| 任务交付项 | 落位目录 | 职责 |
|---|---|---|
| 算子绑定代码 | `hccl/experimental/eco_system/python/hccl4py_ops/` | Python 包、公开导出、ABI/lifetime 实现 |
| 代码目录 README | `hccl/experimental/eco_system/python/README.md` | 两仓 PYTHONPATH、依赖方向、最小调用示例 |
| examples | `hccl/experimental/eco_system/python/examples/01_basic/01_allreduce.py` 等 | 只使用 `_acl.DeviceBuffer`/NPU Tensor 的 14 接口示例 |
| 可复现测试说明 | `hccl/test/ecosystem/hccl4py/README.md` | 环境、2 rank 启动命令、参数、清理步骤、预期结果 |
| 功能测试与报告 | `hccl/test/ecosystem/hccl4py/`、`.../report/` | 用例、日志、截图、环境与 commit、逐项 PASS/FAIL |
| 性能对比 | `hccl/test/ecosystem/hccl4py/perf/hccl_ops_c.cpp`、`perf_compare.py` 与报告 | C/Python 同口径、原始 Batch、统计结果 |
| 布局交叉校验 | `hccl/tools/c_layout_ref.c` | 仅验证 enum/struct 布局，不随包发布 |
| hcomm 配套代码 | `hcomm/experimental/eco_system/python/hccl4py/` | 类型层、通信域、公开 `hccl4py.api` |

`hccl_ops_c.cpp` 与 `c_layout_ref.c` 是测试/验证期一次性工具，需要编译器和 CANN 头文件；
交付包本身没有构建步骤。工具编译命令、所需 include/lib 路径和二进制忽略规则写入对应 README，
避免与“运行期纯 Python、零编译”的表述冲突。

## 替代方案

1. **默认使用 C 扩展（Cython / pybind11 / 原生扩展）**：可降低调用开销，但增加编译产物、
   ABI 与发布链；保留为实测超 10μs 后的可选 fast path 回退；
2. **头文件解析自动生成全部绑定**：生成器复杂、输出难评审；当前只生成并校验签名指纹；
3. **复用 hcomm 仓已有 python/hccl 绑定**：面向 group 系内部场景，不覆盖本次 19 个 Config 系接口；
4. **`ctypes` 透传 NumPy 指针**：NumPy 是 host 内存，无法满足 device 地址语义，明确否决。

## 开放问题

1. 正式版本是否提供 wheel、独立版本号以及跨仓 pip 发布顺序；
2. 若任务方将正式验收硬件从 A5 调整到 A3/950，需同步修改设计环境口径、官方 dtype 矩阵和测试材料。
3. 任务方需确认最终验收 CANN 版本；若不能锁定 9.2.0，则 9.1.0 必须纳入同一套全量功能与性能验收。
4. 任务方需确认“10% 抖动”的唯一统计口径：Batch CV 与轮均值偏差，或逐批增量相对均值的最大偏差。
