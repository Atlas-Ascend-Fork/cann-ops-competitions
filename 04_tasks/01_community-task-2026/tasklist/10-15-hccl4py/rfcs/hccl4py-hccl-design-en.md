# RFC: hccl4py — Python Bindings for HCCL Communication Operators

- Start date: 2026-10-18
- RFC PR: [cann/hccl !3657](https://gitcode.com/cann/hccl/merge_requests/3657)
- Related issue: [cann/hccl #824](https://gitcode.com/cann/hccl/issues/824)
- Status: Draft
- Template basis: the required chapters in the [community-task design template](../../../resources/design_template.md), supplemented by the RFC structure

---

## Summary

Provide **hccl4py_ops**, a Python binding package for the 14 HCCL communication operator C APIs in `include/hccl.h`.
The package exposes collective and point-to-point communication without requiring applications to write ctypes bindings. It preserves C API semantics and targets additional Python overhead of ≤10μs. Tests and usage examples accompany the implementation.

The hcomm repository provides communicator management and shared types; see the HCOMM design. This RFC covers the operator bindings.

## Requirements Analysis (required)

### Requirement

Deliver the pure-Python package `hccl4py_ops` under `experimental/eco_system/python/` in the hccl
repository for all 14 communication operator C APIs. The package must preserve C buffer, stream,
communicator, and return-code semantics, and provide reproducible examples, tests, reports, and
performance evidence. Additional host submission overhead must be at most 10μs with no more than 10%
jitter.

### Requirement Breakdown

1. Expose all 14 operators with explicit parameter order, count units, dtype rules, and return semantics.
2. Accept device memory only; never perform implicit H2D/D2D copies or accept NumPy/CPU pointers.
3. Keep `counts`, `displacements`, the AlltoAllVC matrix, and Batch items alive until their stream consumes them.
4. Support the 9.1.0 runtime floor and validate the actual 9.2.0 acceptance ABI using symbols and prototype fingerprints.
5. Distribute both repositories through source layout and `PYTHONPATH`; cross-repository imports use public `hccl4py.api` only.
6. Test the full official dtype matrix, two-device correctness, parameter lifetime, and the ≤10μs host-submission budget.

## Runtime and Acceptance Environment

| Item | Unified definition |
|---|---|
| Task compatibility floor | A5 series, CANN 9.1.0, and pyACL; run import/binding and single-device smoke. If the task owner selects 9.1.0 for final acceptance, run the full suite there as well |
| Default formal acceptance baseline | A5 series, CANN 9.2.0, pyACL, and two devices on one host |
| Version consistency | Headers, `libhccl.so`, `libhcomm.so`, `libascendcl.so`, and pyACL come from one installation root |
| Test-material relationship | A3/950 references in current material are not an acceptance basis; the task/test owner must align them to A5 + CANN 9.2.0 before delivery |

The public `v9.1.0` and `v9.2.0` headers show no substantive change to the 14 operator prototypes;
differences are formatting/comments. `HcclDataType` is also unchanged. `HcclCommConfig` changes from
version10/2240B to version11/2248B and adds `hcclChannelSqDepth`. Therefore the HCCL bindings can
support a 9.1.0 floor, but runtime compatibility is established from the loaded library and its
matching header, not from a version string alone.

At runtime, `_abi.py` resolves all 14 symbols in the loaded `libhccl.so`, normalizes the prototypes
from the header in the same installation root, and compares them with a checked-in 9.1.0/9.2.0
profile. A mismatch rejects import and reports both paths and the first differing parameter.

Prototype normalization is provided by the public hcomm helper `hccl4py.api.abi`; the hccl package
does not implement a second parser:

* The parser uses only the Python standard library. It removes comments/preprocessor noise and scans
  target declarations with a constrained lexer and parenthesis scanner.
* Parameter names, `const/volatile/restrict`, visibility macros, and attribute macros do not enter the
  fingerprint. Fixed-width integers, typedefs, and enum aliases are canonicalized while return type,
  order, and pointer depth remain exact.
* For example, `HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, ...)` becomes
  `("HcclAllReduce","HcclResult",("void*","void*","uint64_t",...))`.
* Missing headers, unresolved macros, absent target declarations, ambiguous parsing, or profile
  mismatch raise `AbiCompatibilityError` and reject import. The parser never falls back to a version
  string or symbol-existence check.
* No-device tests include multiline declarations, block comments, `extern "C"`, attribute macros, and
  an intentionally unresolved macro fixture.

`HcclGetCommConfigCapability()` validates configuration-field capability, not the structure ABI
version; communicator configuration validation is defined in the HCOMM design.

## Background and Motivation

- HCCL exposes C interfaces; using them directly from Python requires manual ctypes bindings that are difficult to maintain.
- Custom training frameworks, HPC scripts, and model tuning tools need direct access to HCCL communication.
- A thin, stable Python binding with C-compatible semantics provides a shared integration layer.

## Terminology

| Term | Meaning |
|---|---|
| hccl4py project | The complete Python binding library required by the community task |
| hccl4py_ops | The operator binding package delivered by this RFC in the hccl repository |
| hccl4py base package | The shared types and communicator interfaces delivered by hcomm |
| Buffer description | `(device pointer, count, data_type)`; the pointer must reference device memory |
| Host-side auxiliary arguments | Python-created counts/displacements, VC matrix, or Batch items passed as host array pointers |

## Architecture and Interface Contract

### Overall Architecture

```text
Python applications / frameworks / scripts
        │
   hccl4py_ops (experimental/eco_system/python in this repository)
        │  depends only on the public hccl4py.api contract
   hccl4py (experimental/eco_system/python in hcomm)
        │
   libhccl.so (14 operators)        libhcomm.so (communicators)
```

### Distribution

Stage one uses source delivery plus `PYTHONPATH` as the only acceptance distribution:

```text
<hcomm>/experimental/eco_system/python/  # hccl4py
<hccl>/experimental/eco_system/python/   # hccl4py_ops
```

```bash
export PYTHONPATH=<hcomm>/experimental/eco_system/python:<hccl>/experimental/eco_system/python
python -c "import hccl4py, hccl4py_ops"
```

`hccl4py` re-exports `hccl4py_ops` lazily through module-level `__getattr__` after `api` is ready,
so either package may be imported first. The communicator package remains usable by itself.
`hccl4py_ops` does not import private modules such as `_ffi`, `_buffer`, or `comm`. Wheels,
independent versions, and pip ordering remain follow-up work and do not block current acceptance.

### Public Interfaces and Returns

The 14 interfaces map directly to C APIs. `count` and `data_type` remain in their C positions.
They may be explicitly set to `None` only when an NPU Tensor provides an unambiguous inference;
raw device pointers require both values.

| C API | Python signature | Package | Acceptance case | Metadata inference |
|---|---|---|---|---|
| `HcclAllReduce` | `all_reduce(send_buf, recv_buf, count, data_type, reduce_op, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_reduce` | Tensor may infer |
| `HcclReduce` | `reduce(send_buf, recv_buf, count, data_type, reduce_op, root, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_reduce` | Tensor may infer |
| `HcclBroadcast` | `broadcast(buf, count, data_type, root, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_broadcast` | Tensor may infer |
| `HcclAllGather` | `all_gather(send_buf, recv_buf, send_count, data_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_gather` | Tensor may infer |
| `HcclAllGatherV` | `all_gather_v(send_buf, send_count, recv_buf, recv_counts, recv_displs, data_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_gather_v` | dtype may infer; counts explicit |
| `HcclScatter` | `scatter(send_buf, recv_buf, recv_count, data_type, root, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_scatter` | Tensor may infer |
| `HcclReduceScatter` | `reduce_scatter(send_buf, recv_buf, recv_count, data_type, reduce_op, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_reduce_scatter` | Tensor may infer |
| `HcclReduceScatterV` | `reduce_scatter_v(send_buf, send_counts, send_displs, recv_buf, recv_count, data_type, reduce_op, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_reduce_scatter_v` | dtype may infer; counts explicit |
| `HcclAlltoAll` | `all_to_all(send_buf, send_count, send_type, recv_buf, recv_count, recv_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_to_all` | counts explicit; dtype may infer |
| `HcclAlltoAllV` | `all_to_all_v(send_buf, send_counts, send_displs, send_type, recv_buf, recv_counts, recv_displs, recv_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_to_all_v` | dtype may infer; counts explicit |
| `HcclAlltoAllVC` | `all_to_all_vc(send_buf, send_count_matrix, send_type, recv_buf, recv_type, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_all_to_all_vc` | dtype may infer |
| `HcclSend` | `send(send_buf, count, data_type, dest_rank, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_send_recv` | Tensor may infer |
| `HcclRecv` | `recv(recv_buf, count, data_type, src_rank, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_send_recv` | Tensor may infer |
| `HcclBatchSendRecv` | `batch_send_recv(items, comm, stream=None)` | `hccl4py_ops` | `run_two_rank.py::test_batch_send_recv` | Each item explicit |

`send_recv_item(send_recv_type, buf, count, data_type, remote_rank)` maps to the five fields of
`HcclSendRecvItem`. `batch_send_recv` copies the item list once into a contiguous `ctypes` structure
array and retains it until stream completion.

Operators return `HcclResult` by default. `set_raise_on_error(True)` may instead raise `HcclError`
with the result code, C API name, string description, communicator, and stream diagnostics.
The wrapper does not turn asynchronous C output into a hidden blocking Python return value.

### Buffer Support Matrix

| Input | Accepted as a data buffer? | Rule |
|---|---|---|
| `int` / `ctypes.c_void_p` | Yes, baseline path | Device address; `count` and `data_type` required |
| Object with `data_ptr() -> int` | Yes | Nonzero device pointer; metadata required |
| `torch.Tensor` | Yes only for `device.type == "npu"` and contiguous storage | Passes `data_ptr()`; dtype/count may be inferred |
| CPU `torch.Tensor` | No | Raises `TypeError`; never performs H2D |
| `numpy.ndarray` | No | NumPy storage is host memory and must not reach a device operator |
| Arbitrary host memory | No | No H2D/D2D copy and no residency guessing |

Raw pointers cannot be proven resident in device memory from Python; that remains an explicit caller
contract. All buffer ownership stays with the caller. Non-contiguous NPU Tensors are rejected with a
message to call `.contiguous()`; implicit copying is prohibited. Null pointers, empty Tensors,
negative counts, and counts outside `uint64` are rejected.

### Dtype Mapping

The immutable mapping is built once at import. Dtypes with no lossless portable Tensor mapping must
use a raw device pointer plus an explicit `HcclDataType`.

| `HcclDataType` | Value | PyTorch inference | NumPy dtype (metadata description only) |
|---|---:|---|---|
| `HCCL_DATA_TYPE_INT8` | 0 | `torch.int8` | `np.int8` |
| `HCCL_DATA_TYPE_INT16` | 1 | `torch.int16` | `np.int16` |
| `HCCL_DATA_TYPE_INT32` | 2 | `torch.int32` | `np.int32` |
| `HCCL_DATA_TYPE_FP16` | 3 | `torch.float16` | `np.float16` |
| `HCCL_DATA_TYPE_FP32` | 4 | `torch.float32` | `np.float32` |
| `HCCL_DATA_TYPE_INT64` | 5 | `torch.int64` | `np.int64` |
| `HCCL_DATA_TYPE_UINT64` | 6 | `torch.uint64` if available | `np.uint64` |
| `HCCL_DATA_TYPE_UINT8` | 7 | `torch.uint8` | `np.uint8` |
| `HCCL_DATA_TYPE_UINT16` | 8 | `torch.uint16` if available | `np.uint16` |
| `HCCL_DATA_TYPE_UINT32` | 9 | `torch.uint32` if available | `np.uint32` |
| `HCCL_DATA_TYPE_FP64` | 10 | `torch.float64` | `np.float64` |
| `HCCL_DATA_TYPE_BFP16` | 11 | `torch.bfloat16` | No standard dtype |
| `HCCL_DATA_TYPE_INT128` | 12 | No standard mapping | No standard dtype |
| `HCCL_DATA_TYPE_HIF8` | 14 | No standard mapping | No standard dtype |
| `HCCL_DATA_TYPE_FP8E4M3` | 15 | `torch.float8_e4m3fn` if available | No standard dtype |
| `HCCL_DATA_TYPE_FP8E5M2` | 16 | `torch.float8_e5m2` if available | No standard dtype |
| `HCCL_DATA_TYPE_FP8E8M0` | 17 | No standard mapping | No standard dtype |
| `HCCL_DATA_TYPE_RESERVED` | 255 | Illegal | Illegal |

Torch FP8 names above are candidates and are enabled only when the target torch_npu explicitly
registers equivalent bit semantics with HCCL; otherwise they are treated as unmapped.

An unmapped Tensor dtype with `data_type=None` raises `TypeError` listing all inferable dtypes.
An explicit enum still does not authorize an ambiguous Tensor storage type; such cases use raw
device pointers. The NumPy column documents numeric equivalence only and never makes NumPy data a
valid device buffer.

### Count, Shape, and Validation

- Counts and displacements are element counts/offsets, never bytes.
- Count/displacement arrays have length `comm.rank_size`; the VC matrix has
  `comm.rank_size * comm.rank_size` entries.
- `send_count_matrix[i * rank_size + j]` is the element count from rank i to rank j. Nested lists are
  flattened row-major into one contiguous `uint64` allocation.
- A Tensor may infer a scalar count as its `numel()` only where that uniquely identifies the counted
  segment. All V counts, AlltoAll counts, and AlltoAllV counts are explicit.
- Tensor capacity checks enforce `count <= numel()` and `displ + count <= numel()`. Raw pointer capacity
  is the caller's responsibility.
- Shape is not passed to C; Python validates total elements and contiguity only.

### Stream Semantics

`stream` accepts an `aclrtStream`-compatible `int`, `ctypes.c_void_p`, or `data_ptr()` provider.
`stream=None` obtains the current stream with `aclrtGetCurrentStream()` and raises `HcclError` if no
stream exists; it never silently substitutes a null/default stream. Callers must select the device
with `aclrtSetDevice()` and issue communicator operations serially on the same device/context. The
current stream is thread/device-context scoped and is not cached across threads; this query is part of
the measured fixed cost. Consequently, `libascendcl.so` is a runtime dependency of the operator
bindings, not a test-only dependency.

### Asynchronous Argument Lifetime

Device buffers and communicators must remain valid until the stream completes, matching C semantics.
V/VC/Batch host arrays have additional protection:

1. Every call creates new `ctypes` argument arrays; previous argument arrays are never reused.
2. `_lifetime.py` acquires an event from a per-device event pool and calls `aclrtCreateEvent` only when
   the pool is empty.
3. After the C call succeeds, record the `aclrtEvent` on the same stream.
4. Store `(event, arrays, metadata)` in a pending queue keyed by `(comm_id, stream)`.
5. Subsequent calls, `synchronize(stream)`, or `release(comm, stream)` query events; completed arrays
   are released and events return to the pool.
6. If event creation/recording fails, retain arrays until explicit synchronization/release and raise
   `HcclLifetimeError`; never free early in degraded mode.
7. `comm_destroy` does not replace stream synchronization. Callers synchronize all streams, or use the
   package cleanup hook, before destruction.

The raw-pointer fast path for non-V/VC/Batch calls allocates no host array and records no additional
event. The previous “reuse identical AlltoAllV arrays” optimization is removed because it amplifies
use-after-free risk.

### Dependencies

- Depend only on public `hccl4py.api`: `Comm` with read-only `rank_size`, buffer protocols,
  dtype/result/error types, library information, capability helpers, and the ACL wrapper.
- Resolve all 14 exported symbols from `libhccl.so` and verify the matching header prototype fingerprint.
- `libascendcl.so` is a runtime dependency for `aclrtGetCurrentStream()` and V/VC/Batch event
  management. ACL prototypes come from `hccl4py.api.acl`; do not duplicate them in the operator package.

## Impact

- Performance: the main path performs one metadata lookup, pointer normalization, and the ctypes call.
  Two-device Batch measurements remain the acceptance gate.
- Existing functionality: the change adds an experimental directory without modifying `src/` or `include/`, leaving existing builds and releases unaffected.
- Build: the delivered package is pure Python. C/C++ files under `tools/` and performance tests are
  one-shot validation utilities, never packaged or shipped as runtime artifacts.

## Compatibility

- Existing C APIs and headers remain unchanged.
- The 14 operator prototypes and dtype enum remain compatible between CANN 9.1.0 and 9.2.0.
  9.2.0 is the formal acceptance baseline and 9.1.0 is the runtime floor.
- Runtime compatibility uses the actual `libhccl.so` symbol set and matching-header prototype
  fingerprint, never a version string alone. Unknown profiles are rejected before import completes.
- APIs in the experimental directory may evolve before migration to a stable location.
- This RFC adds an independent Python package and experimental directory. No online service changes, feature switch, or gradual rollout strategy are required.

## Detailed Design

### `hccl4py_ops/__init__.py`

- **Responsibility**: define the public exports of the operator package.
- **Core data**: the 14 operator functions in `__all__` and the `send_recv_item` constructor.
- **Key logic**: import and consistently export the public interfaces from `ops.py`; no ctypes call logic belongs here.
- **Internal interface**: depend only on `hccl4py.api`; never import a private hcomm module.

### `hccl4py_ops/_abi.py`

- **Responsibility**: hold 9.1.0/9.2.0 prototype fingerprints, dtype values, and required symbols.
- **Core data**: normalized type tuples, accepted profiles, loaded library/header paths.
- **Key logic**: resolve all 14 symbols and compare the matching `hccl.h` prototypes to a checked-in
  profile. Errors identify both paths and the first differing parameter.
- **Parsing and fallback**: use the public hcomm prototype/fingerprint helper. Missing headers,
  unresolved macros, absent declarations, or ambiguous parsing fail closed; never degrade to a
  version-string-only check.
- **Internal interface**: provide validated `FunctionPrototype` objects; no per-call header parsing.

### `hccl4py_ops/ops.py`

- **Responsibility**: implement ctypes bindings and argument normalization for the 14 HCCL communication operators.
- **Core data structures**: `BufferDesc(ptr, count, data_type)`, `HcclSendRecvItem`, host
  `uint64[]`/item arrays, communicator handles, and ACL streams.
- **Key logic**: cache independent `CFUNCTYPE` prototypes at import; short-circuit the raw-pointer +
  explicit-metadata path without touching Tensor attributes; validate residency/contiguity/rank/count/dtype;
  create fresh V/VC/Batch arrays and hand them to `_lifetime.py`; preserve `HcclResult` by default.
- **Internal interfaces**: public buffer semantics come from `hccl4py.api`; `_f()`, `_finish()`,
  `_comm()`, `_stream()`, and `_require_dtype()` remain private.

### `hccl4py_ops/_lifetime.py`

- **Responsibility**: retain host auxiliary arguments from submission until stream completion.
- **Core data**: `pending[(comm_id, stream)]` entries containing `(event, arrays, call_site)` plus
  `event_pool[device_id]` for completed reusable events.
- **Key logic**: acquire an event from the pool and create only when empty; record it on the same stream;
  reap completed events on later calls and return them to the pool; drain on explicit synchronize/release;
  switch to conservative retention if event creation/recording fails.
- **Internal interfaces**: `retain()`, `reap()`, `synchronize()`, `release()`, `pending_count()`, and
  test-only `event_pool_size()`.

### Type and Error Policy

Do not use `HcclResult` as an implicit Boolean. Preserve the code, C string, and Python call name
separately. Pre-submission validation raises `TypeError`/`ValueError`; a nonzero C result remains an
`HcclResult` or becomes `HcclError` according to policy. This distinguishes “not submitted” from
“submission failed.”

## Maintainability and Testability Analysis (required)

### Accuracy and Functional Criteria

| Criterion | Definition | Source |
|---|---|---|
| Interface parity | All 14 Python calls preserve parameter order, element units, dtype, and return code | Task §3.1 |
| Numeric correctness | Integer outputs match direct C calls bit-for-bit; floating outputs first require bitwise C/Python equality in the same environment, then a dtype-specific oracle tolerance | Task “same functionality as C” |
| Non-contiguous input | Reject and ask the caller to call `contiguous()`; never copy implicitly | This design |
| Async lifetime | V/VC/Batch host arrays survive until stream completion and become collectible afterward | C async semantics |

Full functional tests live in `test/ecosystem/hccl4py/`. One rootinfo communicator is reused:

| Level | Test | Coverage |
|---|---|---|
| Binding self-check | `test_binding_selfcheck.py` | No device: 14 symbols, fingerprints, layouts, 17 valid dtype values, raw/Tensor rules, NumPy/CPU rejection |
| Cross-package import order | `test_public_api_contract.py` | Import `hccl4py` first and `hccl4py_ops` first; verify lazy export, public contract, and no private-module access |
| Single-device smoke | `test_single_rank.py` | rankSize=1: communicator, stream lookup, all 14 call paths, errors and null pointers |
| Two-device correctness | `run_two_rank.py` | rankSize=2: every legal dtype/reduceOp in the A5/CANN 9.2.0 matrix, unequal V, VC matrix, Send/Recv, Batch |
| Lifetime test | `test_param_lifetime.py` | Drop lists and `gc.collect()` before synchronization, submit further groups while pending, verify event-pool reuse and queue reaping, then synchronize and validate |
| Performance comparison | `perf_compare.py` | C/Python host-submission Batch measurements with identical parameters/order |

Current CANN 9.2.0 A5 (Ascend 950 class) documentation yields at least 292 legal
`(operator, dtype, reduce_op)` cases: 132 reduction cases and 16 dtypes for each of the remaining
10 interfaces. Add eight unequal-layout/matrix/negative cases. Reuse one communicator and synchronize
per case; planning budget is 20–30 minutes. Report IDs, parameters, durations, and screenshots.

`HCCL4PY_TEST_DTYPES` is for development debugging only. When set, the report is marked
“DEVELOPMENT SUBSET”; CI/acceptance mode rejects the variable being set. Missing legal dtype/op cases
fail acceptance.

### Performance Criteria

| Metric | Requirement | Scope |
|---|---:|---|
| Python minus C host submission | mean ≤10μs | Same process pattern, parameters, and C API; synchronization outside timing |
| Batch jitter | CV ≤10% | Retain every Batch sample. The task only says “jitter within 10%”; task owner confirmation is required |
| Round-mean stability | maximum deviation ≤10% | Compare each round mean with the overall mean; alternative scope is per-Batch ΔT deviation from its mean |
| Synchronization-inclusive end-to-end time | Report only, not the 10μs gate | Documents user-visible asynchronous behavior |

The task requires “additional overhead of Python versus C within 10μs (jitter within 10%).” The host
submission boundary and CV/round-deviation definitions are the design's operationalization of
“10% jitter” and require task-owner confirmation before acceptance. The C baseline
`perf/hccl_ops_c.cpp` calls all 14 native APIs; Python uses the public package. Both use identical
count, dtype, rank, stream, warm-up, Batch size, and synchronization points. Each sample contains N
consecutive calls; timing stops before synchronization, and group time/N is one sample. Disable cyclic
GC only during Python timing and restore it afterward.

The performance path is decomposed rather than deferred entirely to final measurement:

| Path | Fixed cost | Design measure |
|---|---|---|
| Raw-pointer fast path | Python checks, `aclrtGetCurrentStream`, pointer conversion, `CFUNCTYPE` call | Cache prototypes and dtype lookups; no cross-thread current-stream cache; no Tensor access or host array |
| NPU Tensor path | `device`, `is_contiguous`, `numel`, `data_ptr` | Call each property once and construct C arguments directly |
| AlltoAllV worst path | 10 arguments, four `uint64[]`, event acquire/record/query | Copy once, use contiguous arrays, and reuse pooled events; create only when the pool is empty |

Design-time no-device indication on Python 3.12.13 / GCC 12.3.1 (not an A5 acceptance value):

| Measurement | Median |
|---|---:|
| Empty 3×`c_uint64` `CFUNCTYPE` call | 0.41μs |
| Empty 7-argument call | 0.80μs |
| Empty 10-argument call | 1.10μs |
| Construct four rank=2 `c_uint64[2]` arrays | 1.04μs |

The worst-case marshalling example consumes about 2.14μs, leaving about 7.86μs. Tensor checks,
`aclrtGetCurrentStream`, event acquire/record/query, and pending-queue reaping must be measured
separately on A5 and cannot be replaced by build-host numbers. If any group causes the path to exceed
10μs, profile it, remove redundant attribute access/diagnostic formatting/closures, then provide an
optional C-extension fast path only if still over budget. Event-management cost is an explicit
fallback trigger.

Reports record the device model, CANN/pyACL/driver versions, actual `.so` paths, git commit, CPU
governor where configurable, N, warm-up, rounds, every raw sample, and statistics-tool version.

### Compatibility Analysis

- HCCL C APIs, headers, and symbols remain unchanged.
- The 14 prototypes and dtype enum match between 9.1.0 and 9.2.0. The default is full 9.2.0 acceptance;
  if the task owner confirms 9.1.0 as the final environment, the same full functional, two-device, and
  performance suites run on 9.1.0.
- The hcomm base selects `HcclCommConfig` version10/11 from headers and library capability in one install.
- The official documentation defines `HcclGetCommConfigCapability` as a scalar upper bound compared to
  enum values, so enumerated fields use `capability > HCCL_COMM_CONFIG_<FIELD>`. Qos and SqDepth have
  no capability enum entry and are covered by version/size/profile plus first-initialization checks,
  rather than claiming every field is capability-checked.
- Source delivery uses only the Python standard library. `torch`/`torch_npu` are optional interop
  dependencies; the raw-pointer path does not import them.
- Experimental Python API stability is not guaranteed before migration to a stable location.

## Risks

1. **Early release of host auxiliary arguments**: event failure or bypassed cleanup can create dangling
   data. The pending queue, conservative fallback, forced-GC test, and pre-destroy cleanup hook mitigate
   this risk.
2. **Raw pointer residency cannot be verified**: `int`/`c_void_p` may accidentally reference host memory.
   The contract says device address; examples use only `_acl.DeviceBuffer`/NPU Tensors; NumPy is rejected.
3. **Incorrect dtype/count**: unsupported storage or counts may mismatch C. Full mapping, bounded
   validation, rejection of ambiguous Tensor dtypes, and full-matrix tests mitigate this risk.
4. **ABI drift**: symbols/prototypes/structures may change. Symbol checks plus fail-closed prototype
   fingerprints reject unknown profiles; every change updates profiles and tests.
5. **Timing noise**: host load, current-stream lookup, and event/array construction affect results.
   Fixed Batches, GC control, event pooling, CV/round checks, raw samples, and separate fast/worst-path
   tests mitigate this risk.
6. **Network differences**: restricted containers lack ranktable/EIDs. Use rootinfo by default; ranktable
   acceptance requires real ports/EIDs, and communicator creation is not a substitute for transfer.

## Deliverables and Directory Mapping

| Deliverable | Location | Responsibility |
|---|---|---|
| Operator package | `hccl/experimental/eco_system/python/hccl4py_ops/` | Public exports, ABI/lifetime implementation |
| Code-directory README | `hccl/experimental/eco_system/python/README.md` | Two-repository `PYTHONPATH`, dependency direction, minimal usage |
| Examples | `hccl/experimental/eco_system/python/examples/01_basic/01_allreduce.py` and peers | DeviceBuffer/NPU Tensor examples for all interfaces |
| Reproducible test steps | `hccl/test/ecosystem/hccl4py/README.md` | Environment, two-rank launch, parameters, cleanup, expected output |
| Tests and report | `hccl/test/ecosystem/hccl4py/` and `.../report/` | Cases, logs, screenshots, environment/commit, PASS/FAIL |
| Performance comparison | `hccl/test/ecosystem/hccl4py/perf/hccl_ops_c.cpp`, `perf_compare.py`, report | Identical C/Python scope, raw samples, statistics |
| Layout cross-check | `hccl/tools/c_layout_ref.c` | Development-only enum/structure verification |
| hcomm companion | `hcomm/experimental/eco_system/python/hccl4py/` | Types, communicators, public `hccl4py.api` |

The two C tools require a compiler and CANN headers only during validation. The delivered package has no
build step. Their README lists compile commands, include/library paths, and ignore rules for binaries.

## Alternatives

1. **Use a C extension by default** reduces call overhead but adds artifacts, ABI, and release costs;
   it remains an optional fallback only after measured budget failure.
2. **Generate all bindings from headers** adds generator complexity and hard-to-review output; only
   prototype fingerprints are generated/validated here.
3. **Reuse hcomm’s existing python/hccl bindings** targets internal group APIs and not the 19 Config APIs.
4. **Pass NumPy pointers through ctypes** violates device residency and is explicitly rejected.

## Open Questions

1. Whether the stable release provides wheels, independent versions, and cross-repository pip ordering.
2. If the task owner changes formal acceptance to A3/950, the environment, official dtype matrix, and
   test material must change together.
3. The task owner must confirm the final acceptance CANN version. If 9.2.0 cannot be fixed, 9.1.0 must
   run the same full functional and performance acceptance suite.
4. The task owner must confirm the single “10% jitter” definition: Batch CV plus round-mean deviation,
   or maximum per-Batch ΔT deviation from its mean.
