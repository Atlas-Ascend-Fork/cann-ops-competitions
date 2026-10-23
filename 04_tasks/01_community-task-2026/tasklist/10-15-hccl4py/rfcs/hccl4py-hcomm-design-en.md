# RFC: hccl4py — Python Bindings for HCOMM Communicator Management and Shared Types

- Start date: 2026-10-18
- RFC PR: [cann/hcomm !6096](https://gitcode.com/cann/hcomm/merge_requests/6096)
- Related issue: [cann/hcomm #1044](https://gitcode.com/cann/hcomm/issues/1044)
- Status: Draft
- Template basis: the required chapters in the [community-task design template](../../../resources/design_template.md), supplemented by the RFC structure

---

## Summary

Provide Python bindings for the 5 HCOMM communicator management C APIs in `include/hccl/hccl_comm.h`, together with shared enums, structures, dynamic library loading, and error types, as the **hccl4py** base package.
The hccl repository provides the 14 operator bindings; see the HCCL design.
When both packages are installed, `import hccl4py` exposes communicator and operator interfaces.

## Requirements Analysis (required)

### Requirement

Deliver the `hccl4py` base package under `experimental/eco_system/python/` in hcomm for five
communicator-management C APIs, shared enums/structures, library loading, public protocols, and error
types. Successful `init_*` calls return a Python `Comm`; failures raise `HcclError`. Rootinfo and
ranktable creation must both work on a real two-device acceptance environment.

### Requirement Breakdown

1. Bind all five communicator APIs and align their returns with the task’s §3.1.3 contract.
2. Support lossless `HcclRootInfo` serialization and faithful `HcclCommConfigInit` inline semantics.
3. Select configuration version10/11 from one matching installation and validate with capability,
   magic, version, and size checks.
4. Expose stable cross-repository types only through `hccl4py.api`.
5. Deliver source, README, examples, reproducible tests, reports, and two-device results.
6. Performance tests in the companion guide cover operators only. Record communicator median/P95 and
   direct-C comparisons as reference data; task-owner confirmation is required for this scope.

## Runtime and Acceptance Environment

| Item | Unified definition |
|---|---|
| Task compatibility floor | A5 series, CANN 9.1.0, and pyACL; run import, communicator, and destroy smoke. If the task owner selects 9.1.0 for final acceptance, run the full suite there |
| Default formal acceptance baseline | A5 series, CANN 9.2.0, pyACL, and two devices on one host |
| Configuration ABI | 9.1.0 uses version10/2240B; 9.2.0 uses version11/2248B with `hcclChannelSqDepth` |
| Installation consistency | Headers, `libhcomm.so`, `libhccl.so`, `libascendcl.so`, and pyACL share one installation root |
| Test material | A3/950 references in current material are not an acceptance basis; task/test owner must align them to A5 + CANN 9.2.0 |

Public `v9.1.0`/`v9.2.0` headers show no substantive ABI change to the five target APIs,
`HcclSendRecvItem`, or `HcclDataType`; only the configuration structure changes. 9.1.0 support is
validated by same-install header/library checks and prototype fingerprints. The final acceptance CANN
version must be confirmed by the task owner; if 9.2.0 cannot be fixed, 9.1.0 must run the same full
functional, two-device, and performance scope.

## Background and Motivation

- Creating communicators from Python requires manual ctypes bindings, including the 4108-byte `HcclGetRootInfo` structure and the 2240/2248-byte `HcclCommConfig` structure, depending on its ABI version.
- Structure layouts and the `static inline` implementation of `HcclCommConfigInit` must be represented in Python according to the headers rather than resolved as dynamic library symbols. Centralized definitions make this behavior testable and maintainable.
- Communicator and operator bindings share types and library loading logic. The lower-layer hcomm repository is the appropriate location for these shared components.

## Terminology

| Term | Meaning |
|---|---|
| hccl4py | The base package delivered by this RFC, containing shared types and communicator interfaces |
| configInfo_t | The first 24 bytes of `HcclCommConfig`: {size, magicWord, version, reserved} |
| rootinfo | `HcclRootInfo`, a 4108-byte opaque identifier used by `HcclCommInitRootInfoConfig` |
| Public API | `hccl4py.api` and its public protocols/types; underscore modules are not cross-repository contracts |

## Architecture and Interface Contract

### Overall Architecture

```text
   hccl4py_ops (operators in hccl)      hccl4py (base and communicators in hcomm)
              │                               │
              └──────── hccl4py.api ──────────┘
                              │
                        ctypes FFI layer
        libhccl.so (operators)   libhcomm.so (communicators)   libascendcl.so (runtime)
```

### Distribution

Stage one uses source delivery plus `PYTHONPATH`, without wheels:

```text
<hcomm>/experimental/eco_system/python/  # hccl4py
<hccl>/experimental/eco_system/python/   # hccl4py_ops
```

```bash
export PYTHONPATH=<hcomm>/experimental/eco_system/python:<hccl>/experimental/eco_system/python
python -c "import hccl4py, hccl4py_ops"
```

`hccl4py.__init__` exports `hccl4py_ops` lazily through module-level `__getattr__` after `api` is ready,
so either package may be imported first. Communicator interfaces remain usable without the operator
package. Operators may depend only on public `hccl4py.api`, never `_ffi`, `_config`, or
`_lifetime`. A public-API contract test protects cross-repository evolution. Wheels and independent
versions remain follow-up work and do not block acceptance.

### Public Interfaces and Returns

| C API | Python interface | Return | Acceptance case |
|---|---|---|---|
| `HcclGetRootInfo` | `get_root_info() -> RootInfo` | Returns a 4108-byte `RootInfo`; failure raises `HcclError` | `test_comm_smoke.py::test_root_info_roundtrip` |
| `HcclCommInitRootInfoConfig` | `init_root_info_config(nranks, root_info, rank, config=None) -> Comm` | Returns `Comm`; failure raises `HcclError` | `test_two_rank.py::test_root_info_init` |
| `HcclCommInitClusterInfoConfig` | `init_cluster_info_config(rank_table, rank, config=None) -> Comm` | Returns `Comm`; failure raises `HcclError` | `test_two_rank.py::test_ranktable_init` |
| `HcclCommConfigInit` (inline) | `comm_config_init(**fields) -> CommConfig` | Returns initialized config; illegal field raises `ValueError` | `test_config.py::test_defaults_and_profile` |
| `HcclCommDestroy` | `comm_destroy(comm) -> None` | Success has no value; failure raises `HcclError` | `test_two_rank.py::test_destroy_and_cleanup` |

Helpers include `get_rank_size`, `get_rank_id`, `get_comm_name`, `get_error_string`, `init_root_info`, and `init_cluster_info`.
They return Python `int`/`str` on success and raise `HcclError` with the original code on failure.

`root_info` accepts a `RootInfo` or its 4108-byte `to_bytes()` representation. With `config=None`,
both `init_*` functions first call `comm_config_init()` to build complete defaults. A supplied config
must come from this package; an all-zero or partially initialized ctypes structure is rejected.

Naming removes the `Hccl` prefix and converts lowerCamelCase to snake_case. Inside the
`hccl4py.comm` namespace, the contextual `Comm` may be omitted once: `HcclCommInitRootInfoConfig`
becomes `init_root_info_config`; `HcclCommConfigInit` remains `comm_config_init`. The task’s
`init_rank_config` is not the current C symbol, so this design adds no alias that could confuse
RootInfo and ClusterInfo paths. README provides the full mapping.

Communicator interfaces raise `HcclError` for a nonzero return code. Operator interfaces in hccl return status codes by default and provide an exception mode.
Exceptions expose errors on the infrequent communicator initialization path; high-frequency operator calls preserve the C return-code convention.

### Dependencies

- Runtime communicator symbols come from `libhcomm.so`. `libascendcl.so` is used by examples/tests and
  by the operator runtime for current-stream lookup and event management.
- Types follow `include/hccl/hccl_types.h`; `tools/c_layout_ref.c` verifies the layouts.
- The public contract in `hccl4py.api` exposes communicator/buffer protocols, dtype/result/error
  types, library information, the ACL wrapper, and the communicator cleanup registration protocol.

## Impact

- Performance: communicator operations are infrequent, at communicator initialization frequency, so ctypes overhead is negligible. The type layer passes pointers without copying their buffers.
- Existing functionality: the implementation adds an experimental directory without modifying `src/`, `include/`, or `pkg_inc/`.
- Build: the package has no runtime compilation. The C layout utility is validation-only and is not shipped.
- Cross-repository maintenance: public `hccl4py.api` isolates internals from the hccl operator package.

## Compatibility

- Existing C APIs and headers remain unchanged.
- CANN 9.1.0 (version10/2240B) and 9.2.0 (version11/2248B) are supported; 9.2.0 is the acceptance baseline.
- Configuration layout is predicted from the header in the actual library install root, then guarded by
  loaded-library capability, magic/version/size, and first-initialization diagnostics.
- Library overrides change the selected install root. By default they may not combine headers from
  installation A with a library from installation B.
- This RFC adds an independent experimental directory and Python package. No online service changes, feature switch, or gradual rollout strategy are required.

## Detailed Design

### `_config.py`

- **Responsibility**: resolve the actual runtime install first, then select the configuration profile
  from the header in that same install root.
- **Core data**: `RuntimeProfile(header_path, libhcomm_path, version, struct_size, capability)`.
- **Key logic**:
  1. Select the library root from `HCCL4PY_INSTALL_ROOT`, the parent of the `lib64/` directory that
     contains `HCCL4PY_HCOMM_LIB`, `HCCL4PY_LIB_DIR`, `ASCEND_HOME_PATH`, or `ASCEND_TOOLKIT_HOME`.
  2. Search only that root for `include/hccl/hccl_types.h` and read version, magic, and fields.
     A source-checkout header is never runtime ABI evidence.
  3. Require 2240B for version10 and 2248B for version11; validate all offsets against a checked-in profile.
  4. Resolve `HcclGetCommConfigCapability` from the actual library. The
     [official documentation](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/HcclGetCommConfigCapability.md)
     defines a scalar upper-bound value, so enumerated fields use `capability > HCCL_COMM_CONFIG_<FIELD>`.
     Qos and SqDepth have no capability enum and are covered by version/size/profile plus
     first-initialization checks, not by a capability claim.
  5. `HCCL4PY_CONFIG_VERSION` is allowed only for source/test setups with a matching profile; acceptance
     cannot use it to bypass same-install checks.
  6. If the library rejects magic/size/version at first initialization, raise `HcclError` containing
     both paths, selected version/size, and the original code.
- **Internal interface**: `resolve_runtime()` supplies one profile to `_ffi.py`; `libs_info()` reports it.

### `_ffi.py`

- **Responsibility**: define ctypes constants, enums, structures, library loading, symbol caching, and the common error type.
- **Core data structures**: `HcclRootInfo` (4108B), `HcclCommConfig`, `HcclSendRecvItem`, and enums corresponding to the C headers.
- **Key logic**: implement `HcclCommConfigInit` in Python according to the header, writing `{size, magicWord, version, reserved}` to the first 24 bytes and unconfigured defaults to the remaining fields. Serialize rootinfo with `ctypes.string_at`/`memmove` to avoid NUL truncation. Cache each function prototype as an independent `CFUNCTYPE` without mutating shared `CDLL` attributes.
- **Validation**: all five target symbols, `HcclGetCommConfigCapability`, type sizes, and the 24-byte
  configuration header are checked before public APIs are exposed. Unknown profiles reject import.
- **Internal interfaces**: provide `bind()`, `hccl_lib()`, `hcomm_lib()`, `acl_lib()`, and `HcclError` to the communicator, operator, and test layers.

### `_prototypes.py`

- **Responsibility**: provide one C-prototype parser and fingerprint normalizer for the 5 hcomm APIs
  and the 14 hccl APIs.
- **Core data**: immutable `("name", "return type", ("parameter types", ...))` tuples plus parser
  diagnostics.
- **Key logic**: use only the Python standard library. Remove comments/attribute macros, scan target
  declarations with a constrained lexer and parenthesis scanner, discard parameter names, and
  canonicalize typedefs, fixed-width integers, and enum aliases while preserving order and pointer
  depth. Missing headers, unresolved macros, absent declarations, or ambiguous parsing raise
  `AbiCompatibilityError` and reject import.
- **Internal interface**: expose `parse_prototypes()`/`prototype_fingerprint()` through
  `hccl4py.api.abi`; hcomm validates its 5 APIs and hccl `_abi.py` reuses the helper for its 14 APIs.

### `_buffer.py`

- **Responsibility**: convert Python buffers to `(ptr, count, data_type)`.
- **Core data**: device pointer, element count, and `HcclDataType`.
- **Key logic**: support raw device pointers, `ctypes.c_void_p`, objects implementing `data_ptr()`,
  and NPU `torch.Tensor`. A Tensor must have `device.type == "npu"` and be contiguous. Explicit
  `count`/`data_type` takes precedence; derive only unambiguous values. A raw pointer without metadata
  is rejected. CPU Tensors and `numpy.ndarray` are rejected without H2D conversion.
- **Internal interfaces**: `ptr_of()` and `describe_buffer()` are reused by the operator bindings.

### `comm.py`

- **Responsibility**: implement the 5 communicator management APIs and header helper APIs.
- **Core data**: communicator handle, read-only `rank_size`, `HcclRootInfo`, `HcclCommConfig`, and `HcclResult`.
- **Key logic**: `comm_config_init()` follows the C inline defaults. Both rootinfo and ranktable paths
  are provided. Rootinfo caches `nranks`; ranktable queries rank size once after initialization. It is
  exposed as read-only `Comm.rank_size` for the communicator lifetime. Nonzero returns become
  `HcclError` with API and string context. `Comm` implements the public `data_ptr()` protocol.
  `comm_destroy()` invokes registered cleanup hooks; pending V/VC/Batch
  events require the caller to synchronize first rather than silently destroying the communicator.
- **Internal interfaces**: expose `get_root_info`, `init_root_info_config`, `init_cluster_info_config`, `comm_config_init`, `comm_destroy`, and helper queries.

### `api.py`, Package Entry, and Test Helpers

- **Responsibility**: `api.py` is the stable cross-repository contract; `__init__.py` organizes public
  exports; `_acl.py` provides the minimum ACL wrapper for examples/tests.
- **Core data**: public interface list `__all__` and optional `hccl4py_ops` operator exports.
- **Key logic**: with only the hcomm package installed, export communicator and type interfaces. When both packages are installed, also export the 14 operator interfaces.
  Cross-repository types, `Comm.rank_size`, and cleanup registration are public only through `api.py`.
- **Internal interfaces**: `_acl.py` provides devices, streams, `DeviceBuffer`, and allocation helpers;
  public aliases are exposed as `hccl4py.api.acl`.

## Maintainability and Testability Analysis (required)

### Accuracy and Functional Criteria

| Criterion | Definition | Source |
|---|---|---|
| Return contract | `init_*` returns a valid `Comm`; failure raises `HcclError`; destroy failure is never hidden | Task §3.1.3 |
| Rootinfo fidelity | `to_bytes()`/`from_bytes()` preserve all 4108 bytes, including NULs and the final byte | C `HcclRootInfo` |
| Configuration semantics | size/magic/version and every unconfigured field match the C inline defaults | `HcclCommConfigInit` |
| Initialization | Real two-device rootinfo and ranktable paths work; rank size/id/name match C | Task scope |
| Lifetime | A communicator is destroyed once; pending auxiliary parameters prevent silent destruction | C API and async rules |

Test coverage:

| Test | Device required? | Coverage |
|---|---|---|
| `test_config.py` | No | Five symbols, version10/11 profiles, 2240/2248B, magic/version, defaults, capability |
| `test_comm_smoke.py` | No | Rootinfo roundtrip, errors, `libs_info()`, header/library conflicts |
| `test_public_api_contract.py` | No | Both import orders, lazy export, cross-package public protocols, `Comm.rank_size`, no private imports |
| `test_single_rank.py` | Yes | Single-rank creation, helper queries, duplicate-destroy error |
| `test_two_rank.py` | Yes | Rootinfo and ranktable creation, rank consistency, destruction |
| `test_cleanup_hook.py` | Yes | Pending V/VC/Batch events block communicator destruction until cleanup |

`test_config.py` also covers parser perturbations (multiline declarations, comments, attribute macros,
an unresolved macro), capability return shape and Qos/SqDepth profile checks, and rank-size cache
consistency for rootinfo/ranktable.

The default is full CANN 9.2.0/A5 two-device acceptance. If the task owner confirms 9.1.0 as the final
environment, run the same full suite on 9.1.0 rather than smoke tests only. Planning budget for
control-plane cases is 10–15 minutes; reports include environment, parameters, logs, and screenshots.

### Performance Criteria

Performance tests in the companion guide cover operators only, so the 10μs metric applies to operator
submission. Communicator interfaces are outside that test scope; median/P95 and direct-C comparisons
are recorded as reference data and the scope still requires task-owner confirmation.

| Metric | Requirement |
|---|---|
| Create/destroy | Record median/P95 against direct C; Python must not add synchronization, retries, or hidden creation |
| `comm_config_init` | Record local initialization cost and structure size; it is not on the submission hot path |
| Data-plane | Covered by hccl `hccl4py_ops` C/Python identical-scope Batch tests |

Measurements record CANN/pyACL/driver versions, A5 model, two-device topology, git commit, and resolved
library paths. Synchronization-inclusive setup latency is reported separately from operator submission.

### Compatibility Analysis

- C APIs, headers, and symbols remain unchanged.
- version10/2240B and version11/2248B profiles are maintained separately; unknown profiles are rejected.
- `HcclGetCommConfigCapability()` uses the documented scalar upper-bound semantics and `>` for fields
  with an enum. Qos/SqDepth have no capability enum and rely on version/size/profile plus first-init
  validation.
- Headers are located from the actual library install root; source-checkout headers are ignored.
- `hccl4py.api` is the public contract. Internal modules may refactor only with cross-package tests.
- Source delivery uses the standard library; NPU Tensor interoperability is optional and detected.

## Risks

1. **Header/library mismatch** could pass 2248B to a version10 library. Resolve the library root first,
   locate the matching header there, validate the checked-in profile/capability, and include both paths
   in initialization errors.
2. **Inline default drift** could desynchronize Python. `c_layout_ref.c` emits size/offset/default bytes,
   and tests assert each field and raw header bytes.
3. **Misusing capability as a version** would be unsafe. The official API returns a scalar upper bound,
   not a bitmask. Version comes from the matching header/profile; capability compares only enumerated
   fields, while Qos/SqDepth use version/size/profile and first-initialization checks.
4. **Multiple installations** may select the wrong library. `libs_info()` reports the root, header,
   library, and fingerprint; conflicts fail fast.
5. **Private cross-repository coupling** may break the operator package. Only `hccl4py.api` is allowed,
   with a shared contract test across both repositories.
6. **Network differences** may prevent ranktable setup. Use rootinfo smoke by default; ranktable requires
   real ports/EIDs, and creation is not a substitute for transfer validation.

## Deliverables and Directory Mapping

| Deliverable | Location | Responsibility |
|---|---|---|
| Base package | `hcomm/experimental/eco_system/python/hccl4py/` | Types, configuration, communicator, public `api.py` |
| Code-directory README | `hcomm/experimental/eco_system/python/README.md` | `PYTHONPATH`, dependency direction, install order |
| Examples | `hcomm/experimental/eco_system/python/examples/01_basic/01_init_comm.py` | rootinfo, config, create/destroy |
| Reproducible test steps | `hcomm/test/ecosystem/hccl4py/README.md` | Two-rank and ranktable setup, expected output, cleanup |
| Tests and report | `hcomm/test/ecosystem/hccl4py/` and `.../report/` | Five APIs, both init paths, version matrix, screenshots |
| Layout cross-check | `hcomm/tools/c_layout_ref.c` | Size/offset/default bytes; not packaged |
| Operator companion | HCCL `experimental/eco_system/python/hccl4py_ops/` and test directory | 14 APIs, lifetime, dtype matrix, 10μs performance |

`c_layout_ref.c` is a validation-only tool requiring a compiler and matching CANN headers. The Python
package has no build step. Its README lists commands, include paths, and binary ignore rules.

## Alternatives

1. **C extensions using Cython/pybind11** reduce call overhead but add artifacts/ABI maintenance and still
   need glue for the inline configuration function; not selected.
2. **Framework-only bindings such as torch_npu** introduce heavy dependencies and miss custom/HPC cases.
3. **Place shared types in hccl** would make hcomm depend on the upper layer, reversing the repository direction.
4. **Duplicate private type layers in both repositories** would drift; public `hccl4py.api` is the chosen contract.

## Open Questions

1. Wheels, independent versions, and cross-repository pip ordering for a stable release.
2. Whether to expose non-task helpers such as `HcclCreateSubCommConfig` in this phase.
3. Whether restricted-container ranktable setup needs additional network diagnostics.
4. The task owner must confirm the final acceptance CANN version. If 9.2.0 cannot be fixed, 9.1.0
   must run the same full functional and performance acceptance suite.
5. The task owner must confirm that the 10μs metric applies only to operator performance cases and
   that communicator median/P95 values are reference data.

## Review Record

Review comments are tracked in [design PR !6096](https://gitcode.com/cann/hcomm/merge_requests/6096).
