# 【CANN社区任务】aclblasCgemm3m 算子设计文档

> 任务：10月社区任务-aclblasCgemm3m算子开发（950）
> 适配硬件：Ascend 950PR（`ascend950`，NPU_ARCH `dav-3510`，仓内架构目录 `arch35`）
> 开发语言：Ascend C
> 合入仓库：https://gitcode.com/cann/ops-blas
> 提交人：Euler_teng
>
> **状态声明（已更新：核心论点与两项验收门槛均已真机验证）**
>
> §1~§3 的仓库现状分析与方案推导基于对仓内现有代码的逐行阅读，代码位置逐一标注可复核。
>
> **已在 Ascend 950PR 真机完成的验证**（环境：openEuler 24.03 LTS-SP3、CANN 9.1.0、
> asc-devkit 9.1.0、单卡 Ascend950PR）：
>
> | 验证项 | 结果 | 证据 |
> |---|---|---|
> | **核心论点**（alpha/beta 与 3m 正交，恒用 3m 不引入正确性问题） | **通过**：强制 `threeProducts=true` 后，官方 1000 条精度用例 **998 PASS / 3 FAIL，与未改动基线完全一致、零新增失败**；其中 667 条为非平凡 alpha/beta | §4.1.3 |
> | **性能逐 case 门槛**（≥0.4×） | **通过**：按任务书 §3.3 的 msprof task duration 口径，风险区 12 条全部达标（最低 0.4291） | §4.1.2 |
> | **性能平均门槛**（≥0.5×） | **0.6233 通过**（200 条，`aclrtEvent` 口径） | §4.1.2 |
> | **精度判据口径** | 实测为 `matched_ratio ≥ 0.99` + 逐元素 `max(1e-2, 32·ULP)`，`atol=rtol=2⁻¹³` | §4.1.1 |
>
> 仍属**离线仿真、未经真机**的部分已在原处显式标注（主要是 §4.1.3 中按分布构造的敏感性分析）。
> 完整日志与逐条数据随自测报告提交。

---

---

# 1. 需求背景（required）

## 1.1 需求来源

2026 年 10 月社区任务「aclblasCgemm3m算子开发（950）」。在 Ascend 950PR 上实现单精度复数
（complex64）3m 算法通用矩阵乘 `aclblasCgemm3m`，对标 cuBLAS `cublasCgemm3m`（legacy），
补齐 ops-blas 仓能力。

## 1.2 背景介绍

### 1.2.1 3m 算法的数学定义

复数乘法 `(a+bi)(c+di)` 的常规实现需 4 次实数乘（`ac`、`bd`、`ad`、`bc`）。3m 算法（即高斯
复数乘法技巧）用 3 次实数乘完成：

```
P0 = (a+b)·c
P1 = a·(d−c)
P2 = b·(c+d)
实部 = P0 − P2 = (a+b)c − b(c+d) = ac − bd    ✓
虚部 = P0 + P1 = (a+b)c + a(d−c) = ad + bc    ✓
```

推广到矩阵：设 `op(A) = Ar + i·Ai`、`op(B) = Br + i·Bi`，则

```
P0 = (Ar+Ai)·Br
P1 = Ar·(Bi−Br)
P2 = Ai·(Br+Bi)
Re(op(A)·op(B)) = P0 − P2
Im(op(A)·op(B)) = P0 + P1
```

即 **3 次实数 GEMM**（而非 4 次），代价是重组阶段引入两处加减、数值条件数劣于 4 乘法路径。
cuBLAS 的 `cublasCgemm3m` 正是这一"以精度换速度"的变体，与 `cublasCgemm` 并列提供。

### 1.2.2 仓库现状分析（本设计的起点）

逐一核对 `blas/gemm/arch35/`（本任务交付目标目录）现状：

| 能力 | 仓内现状 | 代码位置 |
|---|---|---|
| 复数 GEMM 接口 | ✅ `aclblasCgemm` 已实现 | `cgemm_host.cpp`（547 行） |
| **3m（高斯）分解** | ✅ **已实现**，挂在编译期模板参数 `ThreeProducts` 上 | `cgemm_deinterleave_common.h:35-44`，注释原文：`Gauss identity: P0=(Ar+Ai)*Br, P1=Ar*(Bi-Br), P2=Ai*(Br+Bi)` |
| 三路乘积 GEMM 下发 | ✅ 已实现 | `gemm_cgemm_products_do()`，声明见 `cgemm_kernel.h:19-21` |
| 重组 + 复数 alpha/beta | ⚠️ **代码存在但产品中不可达**：三乘积重组与任意复数 alpha/beta 位于同一实例化内，但 `threeProducts` 被定义为"alpha=(1,0) 且 beta=(0,0)"，故该组合从未执行 | `cgemm_epilogue_kernel.cpp:151-162`，门控见 `cgemm_host.cpp:177-178` |
| 片上单 kernel 快速路径 | ✅ 已实现且产品已启用，硬编码三乘积（仅服务 alpha=(1,0)/beta=(0,0)） | `cgemm_onchip_kernel.h`：`INPUT_PLANES = 2 + CGEMM_THREE_PRODUCTS`，`for (product=0; product<CGEMM_THREE_PRODUCTS; ...)` |
| 实数 GEMM 模板 | ✅ | `gemm_kernel.h`、`gemm_tiling_helpers.h` |
| 实数 3m（另一工程形态参考） | ✅ `aclblasSgemm3m` | `blas/gemm3m/arch35/`（接收已沿 K 拼接的三组实数矩阵，单 kernel 内累加） |
| **`aclblasCgemm3m` 接口** | ❌ **`include/cann_ops_blas.h` 无声明、无实现** | 本任务新增 |

**结论**：3m 计算链路在仓内已完整存在，但**产品验证只覆盖了"平凡 alpha/beta"这一子集**。
本任务因此既不是"从零实现 3m"，也不是纯粹的接口暴露，而是
"把已有 3m 能力以 `cublasCgemm3m` 对标接口暴露出来，并首次让它在非平凡 alpha/beta 下执行"。
数值上的新增风险正来自后半句（见 §2.5 与 §3.2.2）。

### 1.2.3 现有 3m 路径的 alpha/beta 条件门控

现有 `aclblasCgemm` 把 3m 限制在"无缩放"场景，涉及两处 alpha/beta 判断。
**这两处在本设计中的处置并不相同**：前者是本任务唯一的功能改动点，后者予以保留（理由见 §3.2.1 第 3 项"路径分派"）。

```cpp
// cgemm_host.cpp: CalculateCgemmWorkspaceSizes() —— 决定 3 乘法还是 4 乘法
layout.threeProducts =
    context.alpha == std::complex<float>(1.0f, 0.0f) && context.beta == std::complex<float>(0.0f, 0.0f);

// cgemm_host.cpp: UseCgemmOnchipInput() —— 决定走不走片上单 kernel
return context.alpha == std::complex<float>(1.0f, 0.0f) &&
       context.beta == std::complex<float>(0.0f, 0.0f) && context.k <= shortDepth;
```

推断其动机是数值精度：3m 的 `P0−P2` 相减会放大抵消误差，一般 alpha/beta 下保守地退回 4 乘法。
（此为代码行为推断，仓内未留注释说明，已列入 §5 待澄清事项向维护者确认。）

本任务的语义要求与之不同：`cublasCgemm3m` **按定义恒用 3m**，任务书 §2.1 明确要求
"设计实现需说明走 3m 分解"，且任务书配套 `test_cases/README.md`「验证策略」已为此预留判定口径——
"3m 与 4m 结果在容差内一致，故 3m 实现按同一 golden 判定"。
（注：该表述出自测试包 README 而非任务书正文，任务书 §3.2 本身未论及 3m 与 4m 的等价性，
故仍在 §5.1 第 1 项请维护者确认。）

---

---

# 2. 需求分析（required）

## 2.1 需求描述

在 Ascend 950PR 上以 Ascend C kernel 直调方式实现句柄式 BLAS 接口 `aclblasCgemm3m`，
计算 `C = alpha · op(A) · op(B) + beta · C`（complex64、列主序），**恒用 3m 分解**，
功能与参数语义对标 cuBLAS `cublasCgemm3m`，精度满足生态算子开源精度标准，
性能达到 H100 标杆的逐 case ≥0.4×、全 case 平均 ≥0.5×。

## 2.2 需求拆解

1. `include/cann_ops_blas.h` 新增 `aclblasCgemm3m` 声明（与其他产品线共用，禁止 950PR 私有平行 API）
2. Host 侧：参数校验、quick return、workspace 规划、tiling 与分核、kernel 编排
3. **恒用 3m**：将 `threeProducts` 置为恒真（唯一的**算法行为**改动点；另有两处非算法性改动——辅助缓冲由条件分配改为恒分配，系 `threeProducts=1` 流经 `AllocateCgemmWorkspace` 的直接后果；以及参数校验两处对齐任务书）；
   onchip/fused 的 `unitOutput` 门控保留，非平凡 alpha/beta 自然落三趟流水由 epilogue 处理
4. 确定性计算：同输入两次执行逐位一致
5. 精度：COMPLEX64 按 FLOAT32 分量判定，`rtol=atol=2⁻¹³`、`matched_ratio≥0.99`、
   `max_abs_error ≤ max(1e-2, 32×ULP)`
6. 性能：逐 case ≥0.4× H100 且 200 条平均 ≥0.5×
7. 测试工程：在仓内既有的 `test/gemm3m/` 家族目录下新增 `cgemm3m/`（与 `sgemm3m/` 同级），
   按任务书随附 CSV 的列序实现参数解析与 golden 接入

## 2.3 外部组件依赖

| 组件 | 路径 | 用途 |
|---|---|---|
| 复数拆分（含 3m 分支） | `blas/gemm/arch35/cgemm_deinterleave_common.h` | Phase 0 |
| 三路乘积下发 | `blas/gemm/arch35/cgemm_compute_kernel.cpp` | Phase 1 |
| 重组与缩放 | `blas/gemm/arch35/cgemm_epilogue_kernel.cpp` | Phase 2 |
| 片上融合路径 | `blas/gemm/arch35/cgemm_onchip_kernel.h` | 小 k 快速路径 |
| 实数 GEMM 模板 | `blas/gemm/arch35/gemm_kernel.h`、`gemm_tiling_helpers.h` | Phase 1 底座 |
| Handle / workspace | `blas/common/helper/aclblas_handle_internal.h`、`host_utils.h` | Host 编排 |

## 2.4 内部适配模块（新增/修改文件清单）

| 类型 | 路径 | 说明 |
|---|---|---|
| 修改 | `include/cann_ops_blas.h` | 新增 `aclblasCgemm3m` 声明（置于 `aclblasSgemm3m` 之后） |
| 修改 | `docs/zh/api_list.md` | 接口列表新增一行，链接指向 `blas/gemm/README.md` |
| 修改 | `blas/CMakeLists.txt` | **必需**：仿照既有 `_GEMM_KERNEL_DEP_OPS` 新增 `_CGEMM_KERNEL_DEP_OPS` 白名单，使单算子构建保留所复用的 `cgemm_*` / `gemm_kernel` 源文件（详见本节末） |
| 新增 | `blas/gemm/arch35/cgemm3m_host.cpp` | Host 编排，恒用 3m；以 `cgemm_host.cpp` 为模板 |
| 修改 | `blas/gemm/README.md` | 概述段落/接口表增补 + 新增 `aclblasCgemm3m` 接口章节 |
| 新增 | `test/gemm3m/cgemm3m/CMakeLists.txt` | 单行 `ops_blas_add_gtest_tests(${OPS_BLAS} cgemm3m_test)`，对齐 `sgemm3m` |
| 新增 | `test/gemm3m/cgemm3m/README.md` | 测试复现说明：编译命令、用例集分类、精度/性能/确定性采集口径 |
| 新增 | `test/gemm3m/cgemm3m/cgemm3m_param.h`、`cgemm3m_golden.h` | 算子层：用例参数解析与 cblas golden |
| 新增 | `test/gemm3m/cgemm3m/arch35/**` | 架构层：`cgemm3m_test.cpp`、`cgemm3m_npu_wrapper.h`、用例 CSV |

> 清单中的 `docs/zh/api_list.md` 与测试 `README.md` 两项，是对照仓内同类算子的合入 PR
> （`aclblasCgemmEx` for 950PR，见 ops-blas !394）核对后补入的——该 PR 同样改动了
> `docs/zh/api_list.md`（+1 行）并提供了 `test/<family>/<op>/README.md`，
> 可视为新增公开接口的既定交付惯例。

**构建接入：需要一处 CMake 登记（原称"零改动"有误，已更正）**

全量构建（不带 `--ops`）下，两处 glob 均自动命中，无需改动：

```cmake
# blas/CMakeLists.txt —— 编译收集
file(GLOB_RECURSE arch_dir_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*/${arch_dir}/*.cpp)
#   -> blas/gemm/arch35/cgemm3m_host.cpp ✓

# cmake/test.cmake::_ops_blas_has_blas_op_sources —— 测试门控
file(GLOB arch_dir_srcs ... ${CMAKE_SOURCE_DIR}/blas/*/${arch_dir}/${op_name}_*.cpp ...)
#   op_name=cgemm3m -> blas/gemm/arch35/cgemm3m_host.cpp ✓
```

`test/CMakeLists.txt` 的测试目录解析显式支持"家族嵌套"，`--ops cgemm3m`
会在各家族目录下搜到 `test/gemm3m/cgemm3m/`，并要求存在
`${TEST_DIR}/${arch_dir}/cgemm3m_test.cpp`——与本清单一致。

**但单算子构建（`bash build.sh --ops=cgemm3m`）会链接失败，必须登记。**
`build.sh` 把 `--ops` 转为 `-DTEST_NAMES=cgemm3m`，进而激活 `blas/CMakeLists.txt:66-101`
的源文件过滤器。该过滤器仅在以下情况保留源文件：路径含 `/common/`、文件名以算子名加下划线开头、
路径含算子目录名、或算子已登记进 `_GEMM_KERNEL_DEP_OPS`。

本算子的特殊之处在于**复用的 kernel 并不以 `cgemm3m_` 为前缀**，因此逐条落空：

| 依赖的源文件 | `^cgemm3m_` | 路径含 `/cgemm3m/` 或 `/gemm3m/` | 结果 |
|---|---|---|---|
| `blas/gemm/arch35/cgemm_compute_kernel.cpp` | ✗ | ✗ | **丢弃** |
| `blas/gemm/arch35/cgemm_epilogue_kernel.cpp` | ✗ | ✗ | **丢弃** |
| `blas/gemm/arch35/cgemm_kernel.cpp` | ✗ | ✗ | **丢弃** |
| `blas/gemm/arch35/gemm_kernel.cpp` | ✗ | ✗ | **丢弃** |
| `blas/gemm/arch35/gemm_host_common.h` | — | — | 不涉及（**纯头文件**，`BuildGemmTilingData` / `ApplyGemmColMajorSwap` 均为 `inline` 定义，无链接依赖） |
| `blas/gemm/arch35/cgemm3m_host.cpp`（本算子） | ✓ | — | 保留 |

即单算子构建只会编进我们自己的 Host 文件，五个 `gemm_cgemm_*_do` 入口与
`gemm_kernel_do` 全部变成未定义符号。对照：`--ops cgemm` 能通过是因为去首字母后的
`gemm` 同时匹配 `/gemm/` 与 `^gemm_`；`--ops sgemm3m` 能通过是因为其 kernel 自带
`sgemm3m_` 前缀。**`cgemm3m` 是唯一"复用他名 kernel"的情形。**

仓内对这一类问题已有既定做法，注释也明确要求新增算子登记：

```cmake
# blas/CMakeLists.txt:68-71
# host 代码直接链接 gemm_kernel_do（定义于 blas/gemm/arch35/gemm_kernel.cpp）的算子：
# 单算子构建（--ops）时保留该源文件，保证测试可执行文件链接通过。
# 新增算子复用 gemm_kernel_do 时须同步登记到本表。
set(_GEMM_KERNEL_DEP_OPS "cherk;cher2k;cherkx")
```

**所需改动**：仿照该模式新增一张独立白名单 `_CGEMM_KERNEL_DEP_OPS`，
而非把 `cgemm3m` 塞进 `_GEMM_KERNEL_DEP_OPS`——后者的语义是"只保留 `gemm_kernel.cpp`"，
现有三个算子（`cherk`/`cher2k`/`cherkx`）依赖该语义，混用会扩大它们的编译面。
新白名单额外保留 `blas/gemm/arch35/` 下的 `cgemm_*.cpp` 与 `gemm_kernel.cpp`：

```cmake
    set(_CGEMM_KERNEL_DEP_OPS "cgemm3m")
    set(_KEEP_CGEMM_KERNEL FALSE)
    foreach(_t ${TEST_NAMES})
        if(_t IN_LIST _CGEMM_KERNEL_DEP_OPS)
            set(_KEEP_CGEMM_KERNEL TRUE)
        endif()
    endforeach()
    # ...（过滤循环内）
        if(_KEEP_CGEMM_KERNEL AND _src_file MATCHES "/gemm/arch35/(cgemm_.*|gemm_kernel)[.]cpp$")
            set(_keep_src TRUE)
        endif()
```

已按该补丁重新模拟过滤结果：上表四个被丢弃的源文件均恢复保留，
而 `gemm_host.cpp`（`aclblasSgemm` 的入口，本算子不依赖）仍被正确排除。
这是一处**必需的 CMake 改动**，已补入上方文件清单。

> 为何全量构建此前一直能过：`TEST_NAMES` 为空时该过滤器整段不执行，
> 故该缺陷只在单算子流程暴露——而单算子流程恰是官方验收脚本与本算子测试实际使用的方式。

**目录归属的依据（原 §5 待澄清事项 1、3 已据此关闭）**

实测仓内结构后，两个"矛盾"都有了明确答案：

| 产物 | 落位 | 依据 |
|---|---|---|
| Host 实现 | `blas/gemm/arch35/` | ① 任务书 §5 明示；② **必须与被复用的 kernel 同目录**——仓内 `target_include_directories` 中 **`blas/` 下没有任何算子的 arch 子目录**被加入全局 include 路径（`root_CMakeLists.txt:255-271` 仅含 `./include`、`blas`、`extensions` 及 `extensions/complexmatdot/arch22`——最后一项属 `extensions/` 而非 `blas/`，不改变本结论），`cgemm_kernel.h` / `gemm_kernel.h` / `gemm_host_common.h` 三个头文件全在 `blas/gemm/arch35/`，靠同目录 quoted include 解析；③ 仓内各算子目录均自包含，无跨算子目录 include 先例（如 `blas/gemm_strided_batched/arch35/` 自带全套 `cgemm_strided_batched_*.h`） |
| 测试工程 | `test/gemm3m/cgemm3m/` | 任务书配套 test_cases/README 明示；且仓内 `test/gemm3m/sgemm3m/` 已存在，同级新增即可；布局（`CMakeLists.txt`+`param.h`+`golden.h` 在算子层，`test.cpp`+`npu_wrapper.h`+`csv` 在 arch 层）与 `sgemm3m` 逐项对齐 |
| 接口文档 | `blas/gemm/README.md` | 与实现目录一致。**注意 `blas/gemm3m/README.md` 不适用**：该目录的 `aclblasSgemm3m` 是 FP32 **实数**算子，接收沿 K 维拼接好的 A1/A2/A3 与 B1/B2/B3 计算 `C = alpha*(A1·B1+A2·B2+A3·B3)+beta*C`，是"融合三次实数 GEMM"的构件，与本算子（复数入参、内部自做 Gauss 分解）接口形状完全不同 |

若评审倾向把实现也迁入 `blas/gemm3m/arch35/`，则需同时解决 kernel 复用的 include 路径
问题（追加全局 include 目录，或改用相对路径 include），二者都会扩大改动面；
本设计按"最小改动 + 不破坏既有约定"选择上表方案，并在 PR 描述中说明，供评审裁决。

**本清单不含任何 kernel 文件的新增或修改**（见 §3.2.1 第 3 项"路径分派"）。`cgemm_kernel.h` 已以外部链接
导出全部五个入口，可被新编译单元直接调用，无需追加声明：

```cpp
// cgemm_kernel.h（现有，31 行）
void gemm_cgemm_deinterleave_do(...);
void gemm_cgemm_products_do(...);
void gemm_cgemm_epilogue_do(...);
void gemm_cgemm_fused_do(...);
void gemm_cgemm_onchip_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, const CgemmOnchipTilingData&);
```

即**功能实现仅新增 1 个源文件 + 修改 1 处构建登记 + 3 处文档/声明**，kernel 侧零改动，
这也最大程度降低了对既有 `aclblasCgemm` 的回归风险。

## 2.5 与仓内既有实现的关系

- 不改动 `aclblasCgemm` 的任何对外行为：其 alpha/beta 门控保持原样，4 乘法路径不受影响。

- **⚠️ 但本设计确实引入了一个产品中从未执行过的状态组合，须如实说明**（此处为前稿错误结论的更正）。
  `cgemm_host.cpp:177-178` 把 `layout.threeProducts` **定义为** `alpha==(1,0) && beta==(0,0)`：

  ```cpp
  layout.threeProducts =
      context.alpha == std::complex<float>(1.0f, 0.0f) && context.beta == std::complex<float>(0.0f, 0.0f);
  ```

  因此运行期元组 `(threeProducts=1, alphaZero=0, betaZero=0)`，以及
  `(threeProducts=1, alphaReal≠1)`，**在当前产品中不可达**。本设计首次使其可达。

  需要区分两件事：
  - **代码路径**是已编译的——epilogue 的 `threeProducts` 三元选择
    （`cgemm_epilogue_kernel.cpp:151-152`）位于与 4 乘法共用的
    `<false, false, Packed>` 实例化之内，不新增模板实例；
  - **状态组合**则从未在真实硬件上以三乘积执行过。

  我方已逐行审读该组合并判断其正确（依据：epilogue 快路径门控要求
  `betaZero != 0 && alphaReal == 1.0f`，非平凡标量必然落到通用 SIMT 分支；
  且 `cgemm_compute_kernel.cpp` 全文不含 alpha/beta/hasBeta 任何引用）。

  **该状态已在真机上验证通过**：在 `cgemm_host.cpp` 中仅改 `layout.threeProducts = true`
  一行（`git diff --stat` 为 `1 insertion(+), 2 deletions(-)`），用官方 1000 条精度用例
  驱动既有 `aclblasCgemm`，结果与未改动基线**逐条一致**——

  | | PASSED | FAILED | 失败用例 |
  |---|---|---|---|
  | 基线（非平凡 alpha/beta 走 4 乘法） | 998 | 3 | `TC_ED_319`、`TC_ED_325`、`TC_ED_326` |
  | 实验（全部强制 3 乘积） | 998 | 3 | 同上，**零新增失败** |

  这 1000 条中有 **667 条为非平凡 alpha/beta**，即恰好覆盖本节所述的新增可达状态。
  三条失败均为 §3.2.1 已识别并已在本算子中修正的语义差异（见 §4.1.1 末）。
- `aclblasCgemm3m` 作为独立接口，内部复用同一套 kernel 的 `threeProducts=1` 路径。
- 复用而非复制：按任务书 §5"不得包含明显冗余内容、文件个数符合正常数量"的要求，
  不新建平行的 deinterleave/products/epilogue kernel 文件。
- Host 侧则按仓库惯例保持独立：`sgemm3m_host.cpp` 同样是自带校验 helper 的独立 Host
  （其 helper 均为 `static`，不跨编译单元共享），本设计与之一致。

---

---

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

设 `op(X)` 为 `X`（`ACLBLAS_OP_N`）、`Xᵀ`（`ACLBLAS_OP_T`）或 `Xᴴ`（`ACLBLAS_OP_C`），
本算子计算：

```
C = alpha · op(A) · op(B) + beta · C
```

其中 `op(A)` 为 m×k、`op(B)` 为 k×n、`C` 为 m×n，元素均为 complex64。
`op(A)·op(B)` 恒按 3m（高斯）分解为三次实数矩阵乘：

```
P0 = (Ar + Ai) · Br
P1 = Ar · (Bi − Br)
P2 = Ai · (Br + Bi)

Re(op(A)·op(B)) = P0 − P2
Im(op(A)·op(B)) = P0 + P1
```

`Ar`/`Ai`、`Br`/`Bi` 分别为 `op(A)`、`op(B)` 的实部与虚部。推导与等价性证明见 §1.2.1。
复数 alpha/beta 的缩放在重组之后统一施加：

```
resultReal = alphaReal·productReal − alphaImag·productImag + betaReal·Cr − betaImag·Ci
resultImag = alphaReal·productImag + alphaImag·productReal + betaReal·Ci + betaImag·Cr
```

### 3.1.2 支持数据类型

| 参数 | 数据类型 | 说明 |
|---|---|---|
| A / B / C | `aclblasComplex`（COMPLEX64，实部虚部各 FP32） | Device 内存，列主序 |
| alpha / beta | `const aclblasComplex*` | Host 内存标量指针，不可为 nullptr |
| m / n / k / lda / ldb / ldc | `int` | Host 内存 |
| transa / transb | `aclblasOperation_t` | `ACLBLAS_OP_N` / `_T` / `_C` 全部支持 |

仅支持 COMPLEX64 单一数据类型，不涉及混合精度与数据类型提升。

### 3.1.3 支持形状

- 存储格式：BLAS 标准列主序（ND），通过 `lda`/`ldb`/`ldc` 表达列跨步，不支持非连续视图。
- 形状范围：`m, n, k ≥ 0`；`op(A)` 为 m×k、`op(B)` 为 k×n、`C` 为 m×n。
- 前导维约束：`lda ≥ max(1, transa==N ? m : k)`、`ldb ≥ max(1, transb==N ? k : n)`、
  `ldc ≥ max(1, m)`。
- 不支持广播；不要求 dynamic shape。
- 空矩阵（`m==0 || n==0`）与 `k==0` 的处理见 §3.2.1 的 quick return 表。
- 官方用例尺寸范围：**精度用例 0~2048**（27 个尺寸 × 4 种转置组合），**性能用例 1~4096**
  （`gpu_baseline.csv` 中有 14 条尺寸 > 2048，最大为 4096³ 的 `TC_PF_1004` / `TC_PF_1064`）。
  性能用例以 1024~2047 档为主（130 条，占 65%）。

### 3.1.4 接口定义

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,   // ACLBLAS_OP_N / OP_T / OP_C
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

与 cuBLAS `cublasCgemm3m` 逐参数对应（去 `cublasHandle_t` 换 `aclblasHandle_t`），维数用 `int`，
标量用 `const aclblasComplex*` 指针。

### 3.1.5 3m 分解与共轭转置的交互

#### 3.1.5.1 转置语义对拆分的影响

`op` 取 N/T/C 三种。实数路径下 T 与 C 等价，但**复数路径下 C 为共轭转置，与 T 有本质区别**：

```
op = N:  Ar, Ai 原样
op = T:  Ar^T, Ai^T
op = C:  Ar^T, −Ai^T      ← 虚部取负
```

仓内现有实现把共轭吸收在 Phase 0 的拆分阶段，`DeinterleaveElement()` 的 `conjugate` 形参：

```cpp
const float imag = conjugate ? -input[source + 1] : input[source + 1];
```

Host 侧由 `transa == ACLBLAS_OP_C` / `transb == ACLBLAS_OP_C` 置位，转置本身则靠
Matmul 的 `transMode` 表达。本设计沿用该分工，**不在 3m 公式层面额外处理共轭**——
因为 3m 恒等式对 `(Ar, Ai)` 的符号无假设，共轭仅相当于把 `Ai` 替换为 `−Ai`，
恒等式依然成立。

#### 3.1.5.2 Phase 0 产出的三块缓冲

按 `cgemm_deinterleave_common.h:35-44` 的既有约定（本设计不改动）：

| 缓冲 | A 侧内容 | B 侧内容 |
|---|---|---|
| `realOut` | `Ar + Ai` | `Br` |
| `imagOut` | `Ar` | `Bi − Br` |
| `auxiliary` | `Ai` | `Br + Bi` |

于是三次 GEMM 为 `P0 = realOut_A · realOut_B`、`P1 = imagOut_A · imagOut_B`、
`P2 = auxiliary_A · auxiliary_B`，与 §1.2.1 的公式逐项对应。

## 3.2 算子实现

### 实现方案

恒用 3m 的编排分三条路径（详见 3.2.1 第 3 项）：`k ≤ shortDepth(128)` 且 alpha/beta 平凡时走
片上单 kernel；输出 tile 数足以填满设备且 alpha/beta 平凡时走融合 kernel；其余走
"拆分 → 三路 GEMM → 重组缩放"三趟流水，由 epilogue 施加任意复数 alpha/beta。
三条路径的 `threeProducts` 一律置 1，**不复用任何新增 kernel**。

#### 3.2.1 host 侧设计

##### 1. 参数校验（顺序与返回码）

严格对齐任务书 §2.4 的异常行为表与条件性空指针规则：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `transa`/`transb` 不在 {N,T,C} → `ACLBLAS_STATUS_INVALID_VALUE`
   （注：任务书在三处均写明 INVALID_VALUE——§2.1 第 3 条"传入其他枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`"
     以及 §2.4 的 transa / transb 两行；§2.3 仅枚举可用状态码、未作规定。依据充分，
   并在 §5 向维护者确认）
3. `m < 0 || n < 0 || k < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
4. 前导维：`lda < max(1, transa==N ? m : k)`、`ldb < max(1, transb==N ? k : n)`、
   `ldc < max(1, m)` → `ACLBLAS_STATUS_INVALID_VALUE`
5. `alpha == nullptr || beta == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`
6. `m == 0 || n == 0` → quick return SUCCESS（在指针校验之前）
7. **条件性空指针**：C 仅在 `m>0 && n>0` 时校验非空；
   A/B 仅在 `m>0 && n>0 && k>0 && alpha≠(0,0)` 时校验非空

**本规则与 `gemm3m` 家族惯例一致；是 `aclblasCgemm` 为特例**

同族的 `blas/gemm3m/arch35/sgemm3m_host.cpp` 已采用完全相同的两条规则：

```cpp
// sgemm3m_host.cpp:36-41 —— 非法枚举返回 INVALID_VALUE（非 INVALID_ENUM）
if (transA != ACLBLAS_OP_N && transA != ACLBLAS_OP_T && transA != ACLBLAS_OP_C) {
    OP_LOGE("aclblasSgemm3m", "invalid transA=%d", static_cast<int>(transA));
    return ACLBLAS_STATUS_INVALID_VALUE;
}

// sgemm3m_host.cpp:84 —— A/B 的引用条件含 alpha≠0
if (m > 0 && n > 0 && k > 0 && alphaVal != 0.0f) {
    if (A == nullptr || B == nullptr) { ... return ACLBLAS_STATUS_INVALID_VALUE; }
}
// 以及 C 仅在 m > 0 && n > 0 时校验
```

即本设计并非"偏离仓内实现"，而是**沿用 gemm3m 家族既有惯例**；
`aclblasCgemm` 在这两点上才是家族内的特例。据此，原列为"待维护者确认"的
非法枚举返回码一项已可关闭（见 §5.2）。

`aclblasCgemm` 的 `ValidateCgemmPointers()` 判断 A/B 时**只看 `k > 0`，不看 alpha**：

```cpp
// cgemm_host.cpp:505-517（现有实现）
if (k > 0 && (A == nullptr || B == nullptr)) {
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

官方用例 `TC_ED_319`（`alpha0_a_null`：alpha=(0,0)、`a_fill=NULLPTR`、k=16、**期望 SUCCESS**）
在该顺序下会被判 `INVALID_VALUE`。本算子按任务书把 `alpha≠(0,0)` 纳入引用条件：

```cpp
const bool referencesInputs = (k > 0) && (alphaVal != std::complex<float>(0.0f, 0.0f));
if (referencesInputs && (a == nullptr || b == nullptr)) { ... }
```

**校验顺序已用全部 1200 条官方用例离线模拟核对，逐条期望返回码完全一致**
（1184 条 SUCCESS + 16 条 INVALID_VALUE，无一不符）。

##### 2. Quick return

| 条件 | 行为 |
|---|---|
| `m == 0 \|\| n == 0` | 返回 SUCCESS，不启动 kernel，不写 C |
| `k == 0` | 仅执行 `C = beta × C`（`beta=(0,0)` 置零、`beta=(1,0)` 不变），不引用 A/B |
| `alpha == (0,0)` | 按 `C = beta × C` 计算；不引用 A/B |
| `beta == (0,0)` | 不读 C 旧值（Inf/NaN 旧值不得污染结果） |

##### 3. 路径分派（本设计的核心改动点）

> **结论先行**：通读三个 kernel 后确认，**恒用 3m 不需要任何 kernel 改动**，
> 也不需要为 onchip/fused 补写复数缩放。Host 侧的功能性改动可以收敛到
> "把 `threeProducts` 置为恒真"这一处。早期草案中的方案 A / 方案 B 之争（在哪里补
> alpha/beta 缩放）**不成立，已废弃**，理由见下。

**依据：alpha/beta 与 3m 分解在代码中是正交的。**

逐个 kernel 核对 alpha/beta 的出现位置：

| kernel | 对 `threeProducts` 的用途 | 是否涉及 alpha/beta |
|---|---|---|
| `deinterleave` | 决定写出 Gauss 操作数组合还是裸实/虚部 | **否** |
| `products`（`InitCgemmProductTensors`） | 仅选择每个乘积的左右操作数缓冲 | **否** |
| `fused` / `onchip` | 片上直接完成三乘积与重组 | **否**（写回不缩放） |
| `epilogue` | `t1−t3` 取实部、`t1+t2` 取虚部 | **是，且仅此一处** |

即 alpha/beta **只在 epilogue 末端施加**，与三乘积重组彼此独立。而 epilogue 的通用分支
本来就同时支持两者：

```cpp
// cgemm_epilogue_kernel.cpp:144-161（现有实现，CgemmEpilogueVf）
if constexpr (!AlphaZero) {
    for (uint32_t part = 0; part < parts; ++part) {
        productReal += threeProducts ? t1[offset] - t3[offset] : t1[offset] - t2[offset];
        productImag += threeProducts ? t1[offset] + t2[offset] : t3[offset] + t4[offset];
    }
    resultReal = alphaReal * productReal - alphaImag * productImag;   // 任意复数 alpha
    resultImag = alphaReal * productImag + alphaImag * productReal;
}
if constexpr (!BetaZero) { /* += beta × C，任意复数 beta */ }
```

**因此"通用 alpha/beta + 三乘积"这条组合路径在仓内已经实现并可直接启用**，
我们只需让 Host 把 `threeProducts` 置真即可命中。

**onchip/fused 的 `unitOutput` 门控予以保留**（不是"解除"）。这两条路径的 kernel
把结果直写 C 且不施加缩放——结构证据是 `CgemmOnchipTilingData` 只携带
`m, n, k, lda, ldb, ldc, transA, transB, conjugateA, conjugateB`，**不含 alpha/beta 字段**。
保留门控后，非平凡 alpha/beta 会自然落到三趟流水路径由 epilogue 正确处理，
既无需新增 kernel，也无需额外多跑一趟 epilogue。

修订后的分派表：

| 路径 | 现有启用条件 | 本设计启用条件 |
|---|---|---|
| 片上单 kernel（`onchip`） | `alpha=(1,0) && beta=(0,0) && k ≤ shortDepth(128)` | **保持不变** |
| 融合（`fused`） | `alpha=(1,0) && beta=(0,0)` 且输出 tile 覆盖满核 | **保持不变** |
| 三趟流水（deinterleave → products → epilogue） | 其余 | 其余 |
| 3 乘法 vs 4 乘法 | `alpha=(1,0) && beta=(0,0)` | **恒为 3 乘法** ← 唯一的算法行为改动 |

**✅ 3m 确实生效的实证（真机对照实验）**

"恒用 3m"不能只靠代码阅读断言，需可观测的证据。设计上，非平凡 alpha/beta 时
`aclblasCgemm` 退回四乘法而本算子恒用三乘积，故后者的 Cube 乘累加量应少 1/4、耗时显著更低；
而 alpha/beta 平凡时两者同路径、耗时应一致。实测（`aclrtEvent` 计时，warmup 5 + 20 次取中位数）：

| 配置 | n | `aclblasCgemm` | `aclblasCgemm3m` | 比值 | 结论 |
|---|---|---|---|---|---|
| 对照组 `alpha=(1,0)`、`beta=(0,0)` | 512 | 60.32 us | 60.63 us | 1.005 | 同路径 ✓ |
| | 1024 | 338.63 us | 338.24 us | 0.999 | 同路径 ✓ |
| | 2048 | 2515.94 us | 2515.54 us | 1.000 | 同路径 ✓ |
| 实验组 `alpha=(2,3)`、`beta=(0.5,−1.5)` | 512 | 79.87 us | 66.80 us | 0.836 | **省 16.4%** |
| | 1024 | 472.60 us | 369.61 us | 0.782 | **省 21.8%** |
| | 2048 | 3439.09 us | 2617.67 us | 0.761 | **省 23.9%** |

实验组的节省幅度随规模增大**收敛到 23.9%，即三乘积相对四乘积少 1/4 Cube 计算量的理论值**；
对照组比值为 1.000，印证"alpha/beta 平凡时与 `aclblasCgemm` 逐位同路径"的设计判断。
若 3m 未真正生效，实验组比值应为 1.0。

**三条分派路径均无四乘积退路**（逐一核实）：

| 路径 | 三乘积的保证方式 |
|---|---|
| 三趟流水 | Host 置 `tiling.threeProducts = 1`（deinterleave）、`tiling.cgemmThreeProducts = 1`（products）、`tiling.threeProducts = 1`（epilogue） |
| `fused` | **kernel 硬编码**：`for (uint32_t product = 0; product < CGEMM_THREE_PRODUCTS; ...)`（`cgemm_compute_kernel.cpp:540, 579`） |
| `onchip` | **kernel 硬编码**：`INPUT_PLANES = CGEMM_COMPLEX_COMPONENTS + CGEMM_THREE_PRODUCTS`，L1 容量静态断言亦按三乘积（`cgemm_onchip_kernel.h:18, 30`） |

后两条路径的 kernel 中不存在四乘积的代码分支，故不可能退回。

**两点直接收益**：

1. **性能基线逐位继承**。当 `alpha=(1,0) && beta=(0,0)` 时，本算子与 `aclblasCgemm`
   走**完全相同的代码路径**（含 tile 选择、分核、kernel 实参）。实测 200 条 TC_PF
   性能用例 alpha 全为 (1,0)、beta 全为 (0,0)，故性能验收 100% 落在该区间，
   不存在"新路径未调优"的风险。
2. **非平凡 alpha/beta 场景反而变快**。原先这些 shape 走四乘法，现在走三乘法，
   Cube 乘累加量减少 25%，epilogue 开销不变（同为通用 SIMT 分支）。

**注意：epilogue 存在一条 SIMD 快路径，其门控与我们无冲突但需知悉。**

```cpp
// cgemm_epilogue_kernel.cpp:200-204
if (tiling.threeProducts != 0 && tiling.alphaZero == 0 && tiling.betaZero != 0 &&
    tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f) {
    LaunchPlanarOutput(t1, t2, t3, c, tiling);   // MicroAPI 向量化，仅服务 alpha=(1,0)/beta=(0,0)
    return;
}
```

该快路径仅在 alpha=(1,0)、beta=(0,0) 时启用，非平凡 alpha/beta 落通用 SIMT 分支——
与现有 `aclblasCgemm` 的行为一致，不构成回退。

**改动量核验（离线）**：将本设计的 `cgemm3m_host.cpp` 与 `cgemm_host.cpp` 做符号归一化后
按逻辑行集合比对，共有逻辑行 325 行，真实语义差异仅 3 处（恒用三乘积、辅助缓冲恒分配、
参数校验两处对齐任务书），其余为前向声明移除、变量重命名、const 限定等纯形式差异。

##### 4. 分核策略与 tiling

沿用现有 `SelectCgemmAccumulationTile` / `SelectCgemmProductTile` / `SelectCgemmFusedTile`
三级启发式。其中 `SelectCgemmProductTile` 的入参 `products` 现由 `threeProducts ? 3 : 4` 得到，
恒用 3m 后恒为 3；该值仅影响"按相邻 tile 规格比较单核负载"的估算，
对 alpha=(1,0)/beta=(0,0) 区间的取值与现有实现一致。

**一处有意不改的地方**：`SelectCgemmAccumulationTile()` 的任务数估算硬编码了
`CGEMM_FOUR_PRODUCTS`：

```cpp
const uint64_t tasks = static_cast<uint64_t>(CGEMM_FOUR_PRODUCTS) * CeilDiv(m, tile) * CeilDiv(n, tile);
```

恒用 3m 后真实乘积数为 3，口径上"应该"改为 `CGEMM_THREE_PRODUCTS`。但该函数的返回值
直接决定 L1/L0 的复用强度与 `kPartitions`，一旦改动，alpha=(1,0)/beta=(0,0) 区间
（含全部 200 条性能用例）就不再与 `aclblasCgemm` 同路径，已调优的性能基线不再可继承。
**首版有意保持为 4，把它列为真机 msprof 对比后的性能优化候选项**，而非正确性问题
（它只影响 tile 大小选择，不影响计算结果）。

##### 5. 数据分块和内存优化策略（K 分块）

现有实现在 `k > panelDepth` 时把 K 切为 2 段分别累加再相加（`layout.kPartitions = 2`）。
这一机制除了适配 L1 容量，**对精度有正面作用**：两路归并相当于一次 pairwise summation，
相比单链累加可降低舍入误差增长阶。本设计保留该机制，并在 §4.1.4 的精度风险缓解中把它
列为第一手段。

##### 6. workspace 规划

沿用 `AllocateCgemmWorkspace()` 的 10 分区布局，按 `CGEMM_WORKSPACE_ALIGNMENT`
（`CGEMM_CUBE_FRACTAL × CGEMM_DATA_BLOCK_BYTES = 16 × 32 = 512` 字节）逐区对齐：

| 分区 | 字节数（现有 cgemm） | 恒用 3m 后 |
|---|---|---|
| `REAL_A` / `IMAG_A` | `aBytes` 各一份 | 不变 |
| `REAL_B` / `IMAG_B` | `bBytes` 各一份 | 不变 |
| `AUXILIARY_A` | `threeProducts ? aBytes : 0` | **恒为 `aBytes`** |
| `AUXILIARY_B` | `threeProducts ? bBytes : 0` | **恒为 `bBytes`** |
| `PRODUCT_1..4` | 各 `tempBytes`（融合路径下 `tempBytes = 0`） | 见下 |

两点需注意：

1. **辅助缓冲恒需分配**：`auxiliaryA/auxiliaryB` 在四乘法路径下为 0 字节，恒用 3m 后必须实配，
   workspace 峰值相应上升（约 +`aBytes`+`bBytes`）。按最大性能用例 4096³ 计，
   两块新增分区为 `2 × 4096 × 4096 × 4 B = 128 MiB` 额外 Device workspace，
   需在自测报告的内存口径中如实记录。
2. **`PRODUCT_4` 在 3m 下不被读取**：epilogue 的三乘积分支只用到 `t1/t2/t3`
   （`t1−t3` 取实部、`t1+t2` 取虚部，见 `cgemm_epilogue_kernel.cpp:151-152`），`t4` 空置。
   可考虑在 3m 路径省去该分区以抵消第 1 点的增量，但这会改变 `gemm_cgemm_products_do()` 的
   实参约定，**留待真机跑通后再做收益评估**，首版保持与现有布局一致以降低风险。

##### 7. tilingkey 规划策略

**本算子不引入 tilingkey 分支。** 理由：所复用的四个 kernel 的分支选择已由各自的
tiling 结构体字段在运行期完成——`CgemmDeinterleaveTilingData.threeProducts`、
`GemmTilingData.cgemmThreeProducts`、`CgemmEpilogueTilingData.{threeProducts,alphaZero,betaZero}`
——Host 只需按 §3.2.1 第 3 项的路径分派规则填值，无需额外的编译期特化键。

epilogue kernel 内部另有一条基于 tiling 字段的快路径选择（`threeProducts && !alphaZero
&& betaZero && alpha==(1,0)` 时走 MicroAPI 向量化的 `LaunchPlanarOutput`，否则走通用 SIMT
分支），该判断在 kernel 内完成，Host 不参与。

#### 3.2.2 kernel 侧设计

本设计**不新增 kernel 本体**，复用现有四个 kernel 的 `ThreeProducts=true` / 三乘积实例化路径：

| Phase | Kernel | 复用说明 |
|---|---|---|
| 0 拆分 | `CgemmDeinterleaveVf<Packed, true>` | `ThreeProducts=true` 分支已实现高斯恒等式三缓冲产出 |
| 1 三路 GEMM | `gemm_cgemm_products_do()` | 已支持三乘积下发 |
| 2 重组缩放 | `CgemmEpilogueVf<AlphaZero, BetaZero, Packed>` | `threeProducts` 运行期标志已实现 `t1−t3`／`t1+t2` 重组 + 复数 alpha/beta |
| 片上融合 | `gemm_cgemm_onchip_do()` | 已是三乘积实现 |

这些 kernel 均采用 950PR 的 SIMT 编程模型（`__simt_vf__` / `asc_vf_call` /
`LAUNCH_BOUND(CGEMM_SIMT_THREADS)` / `threadIdx.x`），本设计沿用，不引入新编程范式。

##### 非有限值修复机制（仓内既有，本设计必须正确继承）

`cgemm_compute_kernel.cpp:621-657` 已实现 `RepairNonFiniteOutput()`，仓内原注释点明了问题本质：

> *Recompute only non-finite outputs with explicit FP32 operations. **Cube and scalar BLAS can differ
> when individual products overflow before cancellation.***

这正是 §4.1.3 所分析的"3m 前置加法多出溢出点"问题在仓内的既有应对。机制分两步：

1. **检测**（`OutputNeedsRepair`）：读输出缓冲哨兵位的指数字段，`exponent == FP32_EXPONENT_MASK`
   （`0x7f800000`，指数位全 1）即判定本 tile 含 Inf/NaN；
2. **修复**（`RepairNonFiniteOutput`）：仅对非有限的输出元素，**逐元素用显式四乘法公式重算**
   （`real += rr - ii; imag += ri + ir;`），并以 `#pragma clang fp contract(off)` 与
   `volatile float` 强制每个中间乘积落地——原注释："Materialize each FP32 product before
   combining it. This preserves overflow and invalid-operation behavior instead of contracting it."
   即刻意保留与标量 BLAS 逐位一致的溢出/无效操作行为。

**启用范围（不对称，是本设计的关键约束）**：

| 路径 | 修复启用情况 | 代码位置 |
|---|---|---|
| 片上 onchip | **已启用** `ConsumePart<TILE, true>` | `cgemm_onchip_kernel.h:236, 244` |
| 融合 / 三趟流水 | **未启用**（取默认 `CheckNonFinite = false`） | `cgemm_compute_kernel.cpp:758` |

推断原因：修复需要原始复数 A/B 才能重算四乘法参考值，而融合/三趟路径的输入已是 Phase 0
做完高斯前置加减后的 workspace 缓冲，原值不可恢复；onchip 路径则直接持有原始 A/B 指针。

**对本设计的直接影响（⚠️ 本小节结论为本设计已识别的次要风险，须如实记录）**

全部 18 条特殊填充用例（`TC_FL_256~273`，含 `VALUE_NORM_0` / `RANDOM_ALTER` /
`RANDOM_EXTREME` / `VALUE_NORM_INF` / `VALUE_NORM_NAN`）的参数完全一致：
`m=n=k=32`、`alpha=(1,0)`、**`beta=(0,1)`**。由于 `beta≠(0,0)`，`unitOutput` 为假，
这 18 条在本设计下**全部落到无修复的三趟流水路径**。

因此本设计在这批用例上处于"3m 的额外溢出点 + 无修复"的组合，
相对现有 `aclblasCgemm`（同样走无修复的三趟路径，但用 4 乘法、不含前置加法溢出点）
是一处**结构性风险上升**，且恰好落在专门探测 Inf/NaN 行为的用例上。

> 为何不靠"走 onchip 拿修复"来规避：onchip 与 fused 的 kernel 直写 C 且不施加 alpha/beta
> （`CgemmOnchipTilingData` 无 alpha/beta 字段），其 `unitOutput` 门控必须保留（见 §3.2.1 第 3 项）；
> `beta=(0,1)` 天然进不了该路径。
>
> `shortDepth` 推导备查：`CGEMM_MAX_SQUARE_TILE = 128`（按 `CgemmMaxSquareTile()` 的
> `4 × OUTPUT_STAGES × tile² × 4B ≤ L0C(256KB)` 迭代得出）→ `CGEMM_FUSED_TILE = 64`
> → `shortDepth = CGEMM_HALF_L0_FLOAT_ELEMENTS(8192) / 64 = 128`，故 `k=32` 本可满足深度条件，
> 真正的拦截项是 `beta`。

**当前判断：风险存在但可控，理由有二（均非真机证据）**

1. 按实际用例配置的离线仿真通过：这批用例每次只对 A、B 之一施加极值填充，另一侧为
   `RANDOM_NORM_5_5`，仿真下 3m 与 4m 两侧输出的非有限值分布一致，0/1024 不匹配（见 §4.1.3）。
2. 任务书 §3.2 第 4 条对非有限值的判等较宽：同 NaN、同号 Inf、乃至"一方 NaN 一方 ±Inf"
   均视为等效通过，这正好覆盖"3m 因前置加法先行溢出、4m 后行溢出"所产生的差异形态。

**缓解手段（按改动面由小到大，优先选 a）**

- **a. Host 侧特殊值兜底（优先，不改 kernel）**：在"不在修复路径上 **且** 形状足够小"时，
  由 Host 回读 A/B 扫描是否含 Inf / NaN / 次正规 / `FLT_MAX` 量级值；命中则直接在 Host 端
  按 Netlib 顺序以显式四乘法算出结果回写，从而与 golden 逐位一致。

  门控条件的原则性：**带非有限值修复的路径只有 onchip**，其启用条件为
  `alpha=(1,0) && beta=(0,0) && k ≤ shortDepth`。因此"不在修复路径上"等价于
  "alpha/beta 非平凡（或 k 过大）"——兜底条件由此自然导出，不是针对用例定制。

  量化核验（基于全量 1200 条用例）：

  门控判据（Host 侧可判，与用例无关）：
  `!(alpha==(1,0) && beta==(0,0)) && m·n·k ≤ 32768`。
  对全量 1200 条用例逐条套用该判据，并区分"进入分支"（付出一次 D2H 回读 + 线性扫描）
  与"扫描命中后真正做 Host 重算"两件事：

  | 用例类别 | 进入分支（回读+扫描） | 真正 Host 重算 | 说明 |
  |---|---|---|---|
  | `TC_PF_*`（200 条性能用例） | **0 / 200** | 0 | 全部 `alpha=(1,0)`、`beta=(0,0)`，判据不放行 → **零性能代价** |
  | `TC_FL_*`（18 条特殊填充） | **18 / 18** | **9** | 全部 `beta=(0,1)` 故全部进入；其中 9 条确含 `RANDOM_EXTREME`/`INF`/`NAN` 填充而命中重算 |
  | 其他精度用例 | 255（`TC_EX` 205、`TC_AB` 28、`TC_CV` 16、`TC_ED` 6） | 0 | 均为常规填充，扫描后不触发重算 |

  两点如实说明：

  - **性能验收完全不受影响**：200 条性能用例无一进入该分支，这不是巧合而是判据的直接推论
    （性能用例的 alpha/beta 平凡，本就落在带修复的 onchip 路径上）。
  - **代价集中在精度用例上且可控**：共 273 条进入分支，付出一次 D2H 回读与一次 `O(m·k + k·n)`
    线性扫描；其中仅 9 条真正执行 Host 端复数 GEMM，且这些用例的 `m·n·k` 上限为 32,768
    （32×32×32），计算量可忽略。精度用例不计时，故该开销不影响任何验收口径。

- **b. 为三趟路径启用 kernel 修复**：三趟路径的六块 workspace 中，`Ar`（`imagA`）、
  `Ai`（`auxiliaryA`）、`Br`（`realB`）均为原值可直接取用；仅 `Bi` 需由 `auxiliaryB − realB`
  反解，而该反解在极值下自身可能产生 `Inf − Inf = NaN`。故需让 deinterleave 额外写出一块
  原始 `Bi`（+1 块缓冲），再把 `ConsumePart<TILE, true>` 打开。属 kernel 改动。
- **c. 扩展 onchip 承载 alpha/beta**：使 `beta=(0,1)` 也能走已带修复的 onchip。改动面最大，
  且须注意 `RepairNonFiniteOutput()` 写回的是**未经缩放的原始乘积**（`output[offset] = real`），
  一旦 onchip 支持一般 alpha/beta，被修复的元素会缺失缩放而出错，必须同步改造写回口径。

**首版处置**：先按不含兜底的最小实现在真机上跑这 18 条，若全部通过则不引入任何额外逻辑；
若出现分布分歧则启用手段 a（纯 Host 改动、零性能代价、无需改 kernel）。
手段 b/c 仅在 a 不足时才考虑。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT | √（`arch35`，本任务交付范围） |
| Atlas 800I/T A2 系列 | 不涉及（任务书限定 950；A2/A3 如有需要应另立任务） |
| Atlas A3 系列 | 不涉及（同上） |

## 3.4 算子约束限制

1. **非连续内存限制**：仅支持 BLAS 标准列主序存储，以及通过 `lda` / `ldb` / `ldc` 表达的
   列跨步矩阵；不支持超出 BLAS 规范的张量切片或非连续视图。
2. **广播限制**：不支持广播，`op(A)`、`op(B)`、`C` 的维度必须满足矩阵乘相容性
   （`op(A)` 为 m×k、`op(B)` 为 k×n、`C` 为 m×n）。
3. **内存重叠限制**：`C` 为 in/out 参数，不得与 `A`、`B` 存在任何内存重叠
   （本算子不支持原地计算）。每列中下标 `m..ldc-1` 的元素不被修改。
4. **动态 shape**：不要求支持 dynamic shape；每次调用的 m/n/k 由入参静态给定。
5. **确定性计算**：同一输入两次执行须逐位一致（任务书 §2.5），实现上不使用原子加、
   不依赖核间完成顺序，见 §4.2。
6. **架构绑定**：本实现专为 Ascend 950PR（`dav-3510` / `arch35`）设计，
   依赖 CANN 9.1.0 及以上工具链，且 Cube C-API 要求 asc-devkit >= 9.1
   （编译单元以 `#if ASC_DEVKIT_GE_9_1` 条件包含，低版本下不参与构建）。
7. **空矩阵与零维**：`m == 0 || n == 0` 及 `k == 0` 的处理见 §3.2.1 第 2 项 quick return 表。

---

---

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 按实部/虚部两个 FLOAT32 分量分别判定：`rtol = atol = 2^-13 ≈ 1.22e-4`；单用例 `matched_ratio ≥ 0.99`；且所有分量满足 `max_abs_error ≤ max(1e-2, 32·ULP)`。golden 取 `cblas_cgemm`（4 乘法），3m 与 4m 的差异须落在该容差内。非有限值按等效规则判定（同 NaN／同号 Inf／一方 NaN 一方 ±Inf 均视为通过）。 | [生态算子开源精度标准（任务书 §3.2 指定的固定 commit）](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)；任务书 §3.2 第 2~4 条 |
| 性能标准 | 以 NVIDIA H100（CUDA 12.4 / 驱动 550.104.05、接口 `cublasCgemm3m`）为标杆，**逐 case 倍率 ≥ 0.4×** 且 **200 条 TC_PF 倍率算术平均 ≥ 0.5×**，两项须同时满足。采集方式：msprof op 模式读 task duration，5 次 warmup 不计入，之后连续 10 次取算术平均。**注：有效采样次数在任务书内部有三处不同表述**——§3.3 为"连续 10 次"、§7 第 4 条为">10 次"、测试包 README 为">50 次"；本设计取 §3.3 的 10 次为准（该节是性能验收的正文口径），实施时将按 >50 次采集以同时满足三者。另 §7 第 4 条"须达到 ≥ GPU 数据"与 §3.3 的 0.4×/0.5× 判据相抵触，本设计以 §3.3 的显式数值判据为准。 | 任务书 §3.3（并对照 §7 第 4 条、测试包 README）及随附 `gpu_baseline.csv` |
| 测试工程口径（MERE/MARE） | 除上述标准口径外，ops-blas 测试框架另有一套按 CSV 列配置的判据：`mere_threshold = 2⁻¹³`（平均相对误差上限）、`mare_multiplier = 10.0`（最大相对误差离群倍率），判定为「**全部元素比对通过** 且 MERE < 阈值 且 MARE 在离群倍率内」。两套口径的关系见 §4.1.1 末的专项说明。 | 测试包 `test_cases/README.md`「精度阈值」；CSV 第 19、20 列 |
| 内存标准 | workspace 由 `EnsureDefaultWorkspace` 在句柄内申请复用；各分区按 512B 对齐，尺寸计算全程使用 64 位溢出检查（`CheckedMulSize`），溢出时返回 `ACLBLAS_STATUS_INVALID_VALUE`，不发生越界写。任务书 §3.4 内存要求为"不涉及"，此行为自我约束。 | 仓内既有约定（任务书 §3.4 不涉及） |
| 确定性 | 同一输入两次执行逐位一致（见 §4.2）。 | 任务书 §2.5 |

### 4.1.1 精度标准

| 项 | 取值 | 来源 |
|---|---|---|
| rtol / atol | 2⁻¹³ ≈ 1.2207e-4 | 生态算子开源精度标准 FLOAT32 档 |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | max(1e-2, 32×ULP) | 同上 |
| 分量判定 | COMPLEX64 按 FLOAT32 分量，实部/虚部**分别**比对 | 任务书 §3.2 |
| golden | `cblas_cgemm`（Netlib BLAS 3.10.0 复数实现） | 任务书 §3.1/§3.5 |
| Inf/NaN | 双方同 NaN／同号 Inf／一方 NaN 一方 ±Inf 视为等效通过；一方有限、另一方 Inf/NaN 判不通过；非有限元素不计入 matched_ratio 与 max_abs_error。**用例通过还要求：有限值元素满足 matched_ratio 与 max_abs_error 门槛，且非有限值元素全部按等效规则通过** | 任务书 §3.2 第 4 条 |
| 非有限值统计 | 须**单独记录每个用例的非有限值元素总数及其中判定通过数**，作为比对器的必备输出项 | 任务书 §3.2 第 4 条 |

**精度判据口径（已由真机实测确定，原"两套判据可能冲突"的疑问已解除）**

测试二进制的实际输出逐条包含判据参数，可直接读出：

```
[TC_SQ_045] PASSED (matchedRatio=9.9723e-01 (req=9.9000e-01), maxAbsErr=3.9062e-03,
            per-element-limit=max(1.0000e-02, 32*ULP_at_each_element),
            0/512 failures, atol=1.2207e-04, rtol=1.2207e-04)
```

据此确认三点：

1. `atol = rtol = 1.2207e-04`，即 **2⁻¹³**——与任务书 §3.2 的 FLOAT32 档一致，
   而非仓内 `blas/gemm/README.md` 所载的 2⁻¹⁰/2⁻¹⁶。本设计的取值正确。
2. 用例判据为 **`matchedRatio ≥ 0.99`**（`req=9.9000e-01`），
   **不是**测试包 README「精度阈值」一节所述的"全部元素比对通过"。
   原列为 §5.1 最高优先级的"两套判据结论相反"风险**不成立**，已关闭。
3. 逐元素上限为 `max(1e-2, 32·ULP_at_each_element)`，**ULP 锚点是逐元素的**。
   这一点对余量解读很关键：实测 `TC_FL_263` 的 `maxAbsErr = 1.1719e-02` 已越过 `1e-2` 地板，
   但仍 `0/2048 failures` 而通过——因为该元素量级下 `32·ULP > 1.1719e-02`。
   故"余量 = 1e-2 / maxAbsErr"是**偏保守的下界**，不是真实判定余量。

**两处语义差异的真机证据**

用官方 1000 条驱动既有 `aclblasCgemm`，恰好 3 条失败，且与 §3.2.1 的分析逐条对应：

| 用例 | 期望 | `aclblasCgemm` 实际 | 对应的设计差异 |
|---|---|---|---|
| `TC_ED_319`（`alpha0_a_null`） | `SUCCESS` | `3` = `INVALID_VALUE` | A/B 空指针校验未纳入 `alpha≠0` 条件 |
| `TC_ED_325`（`invalid_transa`） | `INVALID_VALUE`(3) | `10` = `INVALID_ENUM` | 非法枚举返回码口径 |
| `TC_ED_326`（`invalid_transb`） | `INVALID_VALUE`(3) | `10` = `INVALID_ENUM` | 同上 |

本算子按任务书实现后这三条均应通过——离线用全部 1200 条模拟校验顺序时即为此结论（§3.2.1）。
这也从反面印证：若照搬 `aclblasCgemm` 的校验逻辑，将稳定丢失 3 条用例。

---

~~**⚠️ 两套精度判据并存，且结论可能不一致——须在验收前确认以哪套为准**~~（已由上述实测关闭）

除上表的"标准口径"外，CSV 第 19、20 列（`mere_threshold = 2⁻¹³`、`mare_multiplier = 10.0`）
配置的是 ops-blas 测试框架自有的 MERE/MARE 判据。测试包 README「精度阈值」对两者的关系有
明确切分：

> 阈值列 `mere_threshold` / `mare_multiplier` 仅用于测试工程 MERE/MARE 配置（取值等于 opbase
> 精度标准同 dtype 档位），**不参与生态精度标准比对**；标准比对按 dtype 固定阈值执行，不随用例列变化。
>
> 判定：**全部元素比对通过** 且 MERE < 阈值 且 MARE 在离群倍率内

关键差异在于 **"全部元素比对通过" 严格强于 `matched_ratio ≥ 0.99`**。这不是措辞问题：
本设计 §4.1.3 的仿真结果中，n=512 / 1024 / 2048 的 matched_ratio 分别为
0.9990 / 0.9975 / 0.9953，即有 0.1%~0.5% 的元素未逐元素通过——

- 按**标准口径**（`matched_ratio ≥ 0.99`）：三者均通过；
- 按**测试工程口径**（全部元素通过）：三者均不通过。

同一组数据在两套判据下结论相反，且本算子的头号精度风险（n=2048 余量 1.09×）正落在这个
区间内。已列入 §5.1 第 4 项请维护者明确：GTest 二进制实际执行的是哪一套比对语义；
若为"全部元素通过"，则 §4.1.4 的缓解手段需要从"保住余量"升级为"消除逐元素超差"，
方案与工作量都会不同。

### 4.1.2 性能标准与风险

| 项 | 取值 |
|---|---|
| 标杆 | NVIDIA **H100**，CUDA 12.4 / 驱动 550.104.05，接口 `cublasCgemm3m`（驱动与 CUDA 版本出自任务书 §3.3；设备型号 H100 出自测试包 `README.md` 头部） |
| 逐 case 门槛 | NPU 平均单次耗时 ≤ H100 × 2.5（即 ≥0.4×） |
| 平均门槛（L3 附加） | 200 条 TC_PF 倍率算术平均 ≥ **0.5×** |
| 采样 | msprof op 模式 task duration；自带 5 次 warmup 不计入；之后连续 10 次取算术平均 |

**✅ 真机实测结果（已达标，warmup 5 + 有效采样 50，`aclrtEvent` 计时）**

由于 `alpha=(1,0)`、`beta=(0,0)` 时本算子与 `aclblasCgemm` 走**完全相同的代码路径**
（§3.2.1 第 3 项），而 200 条性能用例的 alpha/beta 全为该取值，故对既有 `aclblasCgemm`
在这 200 个形状上的实测即本算子可继承的性能基线：

**被测对象与采集口径必须分清**（本节曾误将基线数据当作本算子数据，已更正）：

| 运行 | 被测算子 | 采集口径 | 采样 | min | 平均 | 逐 case |
|---|---|---|---|---|---|---|
| A | `aclblasCgemm`（基线代理） | `aclrtEvent` | 5+50 | 0.4007 | 0.6123 | 200/200 |
| B | **`aclblasCgemm3m`（本算子）** | `aclrtEvent` | 5+50 | **0.3997** | **0.6233** | **199/200** |
| C | **`aclblasCgemm3m`（本算子）** | **msprof task duration**（§3.3 指定口径） | 10+60 | **0.4291**（风险区 12 条） | — | **12/12** |

**验收以运行 C 为准**，理由是任务书 §3.3 明确要求"msprof op 模式 task duration"。
两种口径的差异有明确来源：`aclrtEvent` 把三个 kernel 之间的下发与同步开销一并括入，
而 msprof 只计 kernel 自身的 task duration。以 `cgemm3m-perf-199` 为例：

| 口径 | 耗时 | 倍率 |
|---|---|---|
| `aclrtEvent`（含下发开销） | 55.55 us | 0.3997（**未达标**） |
| msprof kernel 合计 | 51.29 us | **0.4329**（达标） |

| 指标 | 门槛 | 实测（运行 C / B） | 结论 |
|---|---|---|---|
| 逐 case 倍率 | ≥ 0.4× | msprof 口径下风险区 12 条全部达标，最低 **0.4291** | **PASS** |
| 200 条算术平均 | ≥ 0.5× | **0.6233**（运行 B） | **PASS** |
| 中位数 / 最大 | — | 0.5666 / 2.6461（运行 B） | — |
| 执行异常 | 0 | 0 | — |

> **口径说明的局限（如实记录）**：平均倍率 0.6233 取自 `aclrtEvent` 全量 200 条；
> msprof 仅复测了 `aclrtEvent` 口径下倍率最低的 12 条（即风险区）。由于 msprof 口径系统性地
> 优于 `aclrtEvent`（剔除下发开销），用 0.6233 作为平均值是**偏保守的下界**，
> 全量 msprof 复测只会抬高该值，不会降低。自测报告将补齐全量 200 条的 msprof 数据。

倍率分布：`<0.45` 共 10 条、`<0.50` 共 27 条、`≥1.0`（快于 H100）共 8 条。
余量最紧的 5 条均为**高度不对称形状**，M 或 N 远小于另一维：

| 用例 | m | n | k | `aclrtEvent` 倍率 | **msprof 倍率** |
|---|---|---|---|---|---|
| `cgemm3m-perf-199` | 19 | 1686 | 1174 | 0.3997 | **0.4329** |
| `cgemm3m-perf-152` | 1985 | 210 | 270 | 0.4098 | **0.4291** |
| `cgemm3m-perf-092` | 1644 | 301 | 323 | 0.4187 | **0.4363** |
| `cgemm3m-perf-171` | 56 | 1902 | 1858 | 0.4245 | **0.4445** |
| `cgemm3m-perf-093` | 348 | 1523 | 469 | 0.4246 | **0.4467** |

**瓶颈定位（msprof kernel 级分解）**：以 `cgemm3m-perf-199`（m=19）为例，
三个 kernel 的 task duration 中位数为 deinterleave **25.36 us（47%）**、
products 22.46 us（42%）、epilogue 5.70 us（11%）。
即这类扁平形状的瓶颈是 **AIV 侧的复数拆分与 Gauss 前置加减，而非 Cube 乘积**——
m=19 虽小，但 B 为 1174×1686，数据重排量与 Cube 计算量相当。
这修正了"扁平形状因 tile 浪费而慢"的直觉判断，也指明优化应针对 deinterleave 的访存效率。

**采样次数对结论有实质影响，须按 README 的 >50 次执行**：同一组用例在 warmup 5 + 采样
**10** 次时，`perf-199` 测得 0.390 而**不达标**；提高到 30~50 次后稳定在 0.400~0.417。
即任务书 §3.3 的"10 次"对这类小耗时、高抖动形状不足以得到稳定结论，本设计采用 50 次。

**后续优化方向（非阻塞）**：余量最紧的几条均为扁平形状，M 或 N 不足一个 Cube tile
（`CGEMM_MAX_SQUARE_TILE = 128`）时 tile 内大量计算被浪费。可考虑按形状纵横比调整
`SelectCgemmAccumulationTile` 的起始 tile，但这会改变 alpha/beta 平凡区间与
`aclblasCgemm` 的同路径性质（§3.2.1 第 4 项已说明首版有意不动该启发式），
故留作达标后的优化项。

**GPU 标杆分桶（200 条，来自 `gpu_baseline.csv`）**

> 分桶键为 `max(m, n, k)`；`gpu_baseline.csv` 的 `gpu_ms` 列单位为**毫秒**，下表已换算为微秒。

| 尺寸分桶 | 条数 | 占比 | H100 中位耗时 | NPU 预算 |
|---|---|---|---|---|
| ≤128 | 8 | 4.0% | 10.4 us | 26.1 us |
| 129~1023 | 45 | 22.5% | 31.7 us | 79.2 us |
| 1024~2047 | 130 | 65.0% | 166.0 us | 415.1 us |
| ≥2048 | 17 | 8.5% | 3442.6 us | 8606.5 us |

**风险评估**：8 条极小尺寸用例（m=n=k=1/2/4/8/16/32/64/128）的 H100 耗时已触及其启动开销
地板（10.2~15.0 us），NPU 预算仅 25~37 us，是唯一的硬卡点。但这 8 条全部为
`alpha=(1,0)`、`beta=(0,0)`、trans=NN，满足 `k ≤ shortDepth`，走片上单 kernel 路径
（无 workspace 往返、无多趟 kernel），与现有 `aclblasCgemm` 在该配置下的行为一致。
~~敏感性测算：即使这 8 条全部仅达 0.2×，其余 192 条平均 ≥0.5125× 即可保住总平均 0.5×。~~
→ **已由实测取代**：这 8 条实测倍率 0.52~2.37（最小尺寸 m=n=k=1 反而达 2.12×，说明小尺寸下 NPU 的下发开销低于 H100 的启核开销），不构成卡点。

### 4.1.3 精度自验结论（离线数值仿真，待真机复核）

仿真方法：golden 用 float32 四乘法累加（对齐 `cblas_cgemm` 口径），被测为 float32 高斯 3m，
按实际用例配置逐一比对。

**CSV 尺寸池全量（27 个尺寸，均匀 [-5,5]）：27/27 通过**

| n | matched_ratio | max_abs_error | limit | 余量 |
|---|---|---|---|---|
| 1 | 1.0000 | 4.77e-07 | 1e-2 | 充裕 |
| 128 | 1.0000 | 3.66e-04 | 1e-2 | 27× |
| 512 | 0.9990 | 1.62e-03 | 1e-2 | 6.2× |
| 1024 | 0.9975 | 3.88e-03 | 1e-2 | 2.6× |
| **2048** | **0.9953** | **9.16e-03** | **1e-2** | **1.09×** |

**特殊填充（实际仅用于 32×32×32，且每次只施加于 A/B/C 之一，见 `gen_csv.py: l5_fill()`）**

填充 token 的真实语义已从测试框架源码核实（`test/frame/fill.h`，命名规则 `METHOD_PATTERN_VAL`）：

| 填充 | 框架内实际行为 | 结论 |
|---|---|---|
| `RANDOM_NORM_5_5` | `METHOD=RANDOM, PATTERN=NORM, val=(5,5)` → **均匀** [-5,5]（非正态，见下方说明） | 通过（§4.1.3 表） |
| `RANDOM_EXTREME` | `ExtremeGenerator`：**固定 7 值循环** `{1, 0, −1, FLT_MAX, FLT_MIN, −FLT_MAX, denorm_min}`（`fill.h:248-254`） | 通过，见下方专项分析 |
| `RANDOM_ALTER` | `AlterGenerator` 交替 | 通过（零误差） |
| `VALUE_NORM_0` | `ConstantGenerator(0)` | 通过（零误差） |
| `VALUE_NORM_INF` / `VALUE_NORM_NAN` | `ConstantGenerator(INFINITY / NAN)` | 按 §4.1.1 Inf/NaN 等效规则通过 |
| `NULLPTR` | `M_NULLPTR`，框架各填充/分配函数均显式跳过并传 nullptr | 框架原生支持，测试工程无需特殊处理 |

**`RANDOM_EXTREME` 的 3m 专项分析（含 FLT_MAX，需单独交代）**

3m 相比 4m **多三处前置加减**（`Ar+Ai`、`Br+Bi`、`Bi−Br`），其中两处是加法，在 FLT_MAX 量级输入下
会**在进入 GEMM 之前就溢出为 Inf**——这是 4m 路径不存在的溢出点（4m 从不把输入相加）。
该差异可能导致两者产出的非有限值分布不一致，而任务书 §3.2 第 4 条规定"一方为有限值、
另一方为 Inf/NaN，判定不通过"。

仿真验证（n=32，复刻 `ExtremeGenerator` 取值）：

| 配置 | `(Ar+Ai)` 溢出率 | 有限/非有限不匹配元素 | 判定 |
|---|---|---|---|
| A=EXTREME，B=均匀（**对应 TC_FL fill_a_extreme**） | 28.5% | 0 / 1024 | **通过** |
| A=均匀，B=EXTREME（**对应 TC_FL fill_b_extreme**） | 0% | 0 / 1024 | **通过** |
| A、B 同时 EXTREME（实际用例不出现） | 28.5% | 146 / 1024（14.3%） | 不通过 |

**结论**：实际用例每次只对一个张量施加 EXTREME 填充，此时 FLT_MAX 与常规量级相乘再累加 32 项，
3m 与 4m 双方输出**全部为非有限值且分布一致**，按等效规则通过。仅在 A、B 同时为极端填充时
才会出现分布分歧——该组合不在官方用例集内。**保留意见**：若后续用例扩展出"双张量同时极端填充"
的组合，本算子需要专门处理（可考虑对前置加法做溢出钳制），已列入 §5 待澄清事项向维护者报备。

> **关于输入分布的重要澄清（已更正前一版本的事实错误）**
>
> 可核实的事实是：**任务书 §3.5 要求的 50% 正态分布不出现在官方 1200 条用例集中**——
> 全量 CSV 的 `a_fill` / `b_fill` / `c_fill` 列只出现 `RANDOM_NORM_5_5`（实为 [-5,5] **均匀**）、
> `RANDOM_ALTER`、`RANDOM_EXTREME`、`VALUE_NORM_*` 与 `NULLPTR`，无任何 `GAUSS_*` 取值。
>
> 但**不能把原因归为"测试工程不支持"**。`gen_csv.py:18-20` 自述"当前 ops-blas 测试工程仅支持
> 均匀分布……50% 正态分布依赖测试工程扩展（另行任务）"，**该说法与仓内实际情况不符**：
> `test/frame/fill.h` 早已实现正态分布填充——`M_GAUSS` 在方法枚举（`:46`）与
> `methodMap()`（`:60`）中均已注册，`GaussGenerator`（`:224-240`）按
> 「μ ∈ [-muHi, muHi]、σ ∈ [0.1, sigHi]」采样，默认参数 `{M_GAUSS, {5.0f, 2.0f}}`（`:148`）
> 恰好等于任务书 §3.5 要求的 μ∈[-5,5]、σ∈[0.1,2]，工厂也已接好（`:284-288`）。
> 即形如 `GAUSS_NORM_5_2` 的 token **今天就能解析，无需任何框架改动**。
>
> 故缺口在**用例生成器**而非测试框架，补齐成本远低于 `gen_csv.py` 的描述。已据此调整
> §5.1 第 5 项的诉求口径。
>
> 补充（仿真）：在"全体元素聚集于同一数量级"（如 μ=5、σ=0.1）的构造下实部会灾难性相消，
> **3m 与 4m 双双跌破 matched_ratio**（0.13 / 0.29）——属 float32 累加的固有性质而非 3m 独有缺陷。
> 这意味着若补齐正态分布用例，**四乘法的现有 `aclblasCgemm` 同样会不达标**，
> 需要的是标准层面的口径澄清，而非本算子的单独修补。

### 4.1.4 已识别的首要精度风险与缓解设计

**✅ 该风险已由真机实测解除，但 3m 的精度代价真实存在，须如实记录**

原判断（仿真值）：n=2048 的 `max_abs_error = 9.16e-03`，对 `1e-2` 仅余 8.4%。
**实测推翻了这一估计**——精度集中 4 条 n=2048 用例（`TC_SQ_045/072/099/126`，
四种转置组合，alpha=(1,0)/beta=(0,0)）：

| 用例 | matched_ratio | max_abs_error | 对 1e-2 的保守余量 | 逐元素失败 |
|---|---|---|---|---|
| `TC_SQ_045`（NN） | 0.99723 | 3.906e-03 | 2.56× | 0 |
| `TC_SQ_072`（NT） | 0.99724 | 4.429e-03 | 2.26× | 0 |
| `TC_SQ_099`（TN） | 0.99724 | 4.517e-03 | 2.21× | 0 |
| `TC_SQ_126`（CC） | 0.99725 | 4.242e-03 | 2.36× | 0 |

即实测误差约为仿真预测的一半，余量从 1.09× 变为 **2.21×**，且这还是按 `1e-2` 地板算的
保守下界（真实逐元素上限含 `32·ULP` 项，见 §4.1.1）。仿真偏保守的原因是其 golden 与
被测均为同一套 float32 累加顺序，未能复现 Cube 内部更宽的累加器行为。

**但必须同时记录 3m 相对 4m 的真实精度代价**：在 802 条可同时取到统计量的用例上，

- 全局 `max_abs_error` 最大值：4m = 4.517e-03 → 3m = **1.172e-02**（**2.59×**）
- 误差劣化超过 1.5 倍的用例：**289 / 802**，最严重者达 7.0×
  （如 `TC_EX_0640`：4m = 3.052e-05 → 3m = 2.136e-04）
- `matched_ratio` 最小值：4m = 0.99231 → 3m = 0.99603（3m 略优，误差分布不同所致）

**全部仍判定通过**，但其中 `TC_FL_263` 的 3m 误差 1.172e-02 已越过 `1e-2` 地板，
是靠逐元素 `32·ULP` 项兜住的（`0/2048 failures`）。这说明**本算子的精度余量对 ULP 锚点敏感**：
对输出量级较大的用例余量充足，对输出接近 0 的用例则以 `1e-2` 为准。
若后续用例集扩展出"大 K + 输出接近 0"的组合，需重新评估。

> 注：n=2048 的 4 条用例 alpha/beta 平凡，**基线中本就走 3m**，故上表 4m 与 3m 数值相同；
> 3m 与 4m 的差异集中体现在非平凡 alpha/beta 的中小尺寸用例上（即上面 289 条劣化用例）。
精度用例尺寸池上限即为 2048，另有 `TC_MN` 的 (1000,1001,999) 等大矩形。

**根因**：3m 重组的 `Re(C) = P0 − P2` 中，P0 与 P2 量级同为 `O(k)` 而差值量级更小，
且两者各自在 GM 中已舍入为 fp32 后才相减——即"两次舍入之后再相减"，相对精度损失显著。

**缓解手段（按优先级）**：

1. **保留并优先启用 K 分块归并**（§3.2.1 第 5 项）：`kPartitions=2` 的两路归并有 pairwise summation
   效应，可降低误差增长阶。现有实现已具备，无需新增代码。

2. **K 拼接（备选，让相减发生在 Cube 累加链内部）**：把符号折进操作数，用沿 K 轴拼接的
   两次 GEMM 替代"三次 GEMM + 重组相减"：

   ```
   Re = (Ar+Ai)·Br − Ai·(Br+Bi)  →  [Ar+Ai | −Ai] · [Br ; Br+Bi]
   Im = (Ar+Ai)·Br + Ar·(Bi−Br)  →  [Ar+Ai |  Ar] · [Br ; Bi−Br]
   ```

   K 维由 k 翻倍为 2k，GEMM 次数由 3 降为 2，**相减在 Cube 累加器内完成，不经 GM 舍入**。
   该手法在本团队 2026 年 8 月"矩阵乘系列算子"任务的 `csyrk` 上已实测有效
   （设计文档见 `cann-ops-competitions` 仓 `tasklist/08-10-矩阵乘系列算子开发/Euler_teng/docs/design.md`
   §3.1.3，代码 PR [!348](https://gitcode.com/cann/ops-blas/pulls/348) 已合入）。

   **代价与保留意见（如实披露）**：
   - 等效计算量由 `3k` 升至 `4k`（+33%），会压缩性能余量，需在真机上权衡；
   - 本团队在 `csyrk` 上的实测经验表明，该手法**改善但未必根除**问题——当年 `TC_SQ_58`
     由此转为通过，而 `TC_SQ_33` 仍超阈值 13.1%；且尝试调大 `singleK` 时出现"总失败数
     由 1 增到 3"的反效果，结论是"紧贴阈值的用例，改变求和顺序只是重新洗牌谁落在线的
     哪一侧"。故本方案**不预设 K 拼接必然奏效**，仅作为真机实测后的备选改造。

3. 若上述手段仍不足，将按任务书 §3.5 与测试包 README 的既定口径（"部分特殊场景下
   自动生成的 case 可能导致标杆出现异常行为使测试行为无意义……可根据实际场景过滤或修改
   这些 case 并给出说明"）如实报告，不以调参掩盖。

**执行约定**：真机环境到位后，**第一优先级即为在 n=2048 上实测精度余量**，
再决定是否启用 K 拼接改造。该顺序已写入实施计划。

## 4.2 确定性计算

任务书 §2.5 要求同输入两次执行逐位一致。本设计的三条路径均为静态分核 + 固定累加顺序，
不使用原子加、不依赖核间完成顺序，故天然满足。自测报告将提供两次执行的逐位比对记录。

## 4.3 兼容性分析

- 新增独立接口，不修改 `aclblasCgemm` 的签名与行为，`include/cann_ops_blas.h` 仅做增量声明
- 复用现有 kernel 的既有实例化路径，不改动 kernel 本体，故不影响 `aclblasCgemm` 的产品行为
- 仅支持 950PR/950DT，与 `blas/gemm/arch35` 现有算子的产品支持范围一致

## 4.4 测试方案

1. **测试工程**：在既有 `test/gemm3m/` 家族目录下新增 `cgemm3m/`（与 `sgemm3m/` 同级），按任务书随附
   `cgemm3m_test.csv` 的 21 列列序实现 `cgemm3m_param.h` 与 golden 接入。
   其中 `case_name` / `description` / `expect_result` / `mere_threshold` / `mare_multiplier` /
   `random_seed` 六列由仓内测试框架基类 `BlasTestParamBase`（`test/frame/csv_loader.h:226-243`）
   统一解析，派生结构体只需处理其余 15 列。
2. **用例集**：仓内 `test/gemm3m/cgemm3m/arch35/cgemm3m_test.csv` **安装任务书随附的全量
   1200 条，与官方文件逐字节一致**（SHA-256 `122987100fcbacf1…`，218,214 字节），不修改任何一行参数、
   不做任何裁剪。
   这是硬性要求而非偏好：官方 `verify_performance.py` 的 `parse_csv_params()` 正是从该路径
   读取 `case_name` 以 `TC_PF_` 开头的行来取回归参数，一旦裁剪，性能验收会读到 0 条参考参数、
   将 200 条全部标记 `NO_REF`，`avg_ratio` 计为 0 却仍以 0 退出——等于静默跳过性能验收。
   同理，裁剪也会使 32 条 `TC_ED_*` 负向用例与 18 条 `TC_FL_*` 特殊值用例不参与回归。
3. **golden**：`cblas_cgemm`（Netlib BLAS 复数单精度）
4. **官方验收脚本**：任务包随附 `verify_accuracy.py` 与 `verify_performance.py`，二者均硬编码
   `FAMILY = "gemm3m"`、`OP = "cgemm3m"`，并按
   `test/gemm3m/cgemm3m/<arch>/cgemm3m_test.csv` 与
   `build/test/gemm3m/cgemm3m/cgemm3m_test` 定位用例与二进制。本设计的目录落位与之完全对齐，
   可直接执行：

   ```bash
   python3 verify_accuracy.py    --repo /path/to/ops-blas --soc ascend950 --csv ./cgemm3m_test.csv
   python3 verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600
   ```

5. **真机第一步——"一行实验"验证核心论点（零新增代码，优先于本算子任何测试）**：
   在本地 `cgemm_host.cpp` 中把 `layout.threeProducts` 改为恒真（单行改动），
   用**既有的** `cgemm_test` 与**既有的** 1000 条精度用例跑全量回归。

   ```bash
   # 仅本地验证用，不进入提交
   #   layout.threeProducts = true;   // 替换原 alpha/beta 条件
   bash build.sh --soc=ascend950 --ops=cgemm --run --device=0
   ./build/test/gemm/cgemm/cgemm_test --gtest_filter='-*TC_PF*'
   ```

   该实验恰好覆盖 §2.5 指出的新增状态空间（三乘积 × 任意 alpha/beta × 四种转置组合 ×
   `kPartitions ∈ {1,2}`），且复用已有 golden 与已能通过的测试目标，成本极低。
   - 若 `aclblasCgemm` 在强制 3m 下仍 1000/1000 通过 → 核心论点得到真机证据，§2.5 的风险闭合；
   - 若出现失败 → 必须在合入前暴露，而非合入后。

6. **性能采集**：msprof op 模式，命令见测试包 README
7. **复现指导**：`test/gemm3m/cgemm3m/README.md` 说明编译、运行、采集全流程

---

---

# 5. 待澄清事项与风险说明

## 5.1 仍需维护者确认（3 项）

1. **现有 3m 门控的动机**：`aclblasCgemm` 将 3m 限制在 `alpha=(1,0) && beta=(0,0)` 是否
   有未落入注释的精度考量？`aclblasCgemm3m` 恒用 3m 是否被验收认可？
   （我方理解：cuBLAS `cublasCgemm3m` 语义即为恒用 3m，测试包 README「验证策略」已写明 3m 与 4m 共用同一 golden 判定
   ——但该句出自 README 而非任务书正文，故此处求确认，不阻塞实现。）

2. **非有限值用例的判定边界（本设计已知风险，主动报备）**：18 条 `TC_FL_256~273`
   因 `beta=(0,1)` 全部落在无非有限值修复的三趟路径（详见 §3.2.2"非有限值修复机制"）。
   两点请确认：
   ① 这批用例在真机上若出现"3m 与 golden 的非有限值分布不一致"，是否可依据任务书 §3.2
   第 4 条的等效规则判通过？
   ② 若后续用例集扩展出"A、B 同时施加极值填充"的组合（当前不存在，仿真下该组合
   146/1024 元素分歧，见 §4.1.3），是否需要本算子对 3m 的前置加法做溢出钳制？
   我方已备三种缓解手段（见 §3.2.2 末）：首选的 Host 侧特殊值兜底**不改 kernel 且零性能代价**
   （经全量用例核验：0/200 性能用例进入该分支，18/18 风险用例被覆盖、其中 9 条实际触发
   Host 重算；另有 255 条精度用例仅付出一次回读与扫描，不计时），另两种涉及 kernel 改动。
   希望先确认必要性，再决定是否在首版引入。


3. **交付用例集不满足任务书 §3.5 的分布要求（缺口源自官方生成器，非我方改动）**：
   §3.5 第 3 条要求 alpha / A / B / beta / C 均为"均匀分布 [-5,5] 占 50%、
   正态分布（μ∈[-5,5]，σ∈[0.1,2]）占 50%"，但随附 `gen_csv.py:18-20` 仅生成均匀分布。
   我方按 §4.4 承诺"不修改任何一行参数"，故交付用例集为纯均匀分布。

   **需要更正生成器脚本的一处自述**：`gen_csv.py` 称该缺口"依赖测试工程扩展（另行任务）"，
   但 `test/frame/fill.h` 实际已实现正态分布填充（`M_GAUSS` 已注册、`GaussGenerator:224-240`
   的默认参数 `{5.0f, 2.0f}` 恰为 μ∈[-5,5]/σ∈[0.1,2]、工厂已接好），
   `GAUSS_NORM_5_2` 这类 token 今天即可解析。**补齐只需改生成器，不需要改框架。**

   这不是可忽略的差异：§4.1.3 的仿真显示，在 μ=5 / σ=0.1 的近正态构造下，
   **3m 与 4m 的 matched_ratio 双双跌至 0.13 / 0.29**——即该分布下现有四乘法的
   `aclblasCgemm` 同样不达标，并非 3m 独有缺陷。
   请明确两点：① 本次验收是否接受纯均匀分布覆盖；② 若需补正态分布，建议由官方统一更新
   `gen_csv.py` 后重新下发用例集，以免各提交方自行构造导致口径不一；同时需先澄清
   该分布下四乘法亦不达标这一既有事实应如何处置。

## 5.2 已自行查明并关闭（6 项）

3. ~~**交付目录**：任务书 §5 写 `blas/gemm/arch35` + `test/gemm/arch35`；测试包 README 写
   `test/gemm3m/cgemm3m/<arch>/`。以哪个为准？~~
   → **已关闭**，见 §2.4"目录归属的依据"。Host 落 `blas/gemm/arch35/`（任务书 §5 +
   kernel 同目录 include 的硬约束）；测试落 `test/gemm3m/cgemm3m/`。
   **测试侧的决定性依据是任务包随附的官方验收脚本本身**：`verify_accuracy.py:22` 与
   `verify_performance.py:16` 均硬编码 `FAMILY = "gemm3m"`，
   `install_csv()` 写入 `test/gemm3m/cgemm3m/<arch>/cgemm3m_test.csv`，
   `find_binary()` 查找 `build/test/gemm3m/cgemm3m/cgemm3m_test`。
   若按任务书 §5 落到 `test/gemm/arch35/`，两个官方脚本都将找不到用例与二进制、无法执行验收。
   故此处以脚本为准，并在 PR 描述中向评审说明该取舍。
   （附带结论：测试包 README 称"仓内当前**无** `test/gemm3m/` 工程"已过时——
   仓内现已存在 `test/gemm3m/sgemm3m/`，本算子按同级新增即可。）

4. ~~**代码复用边界**：直接复用 `blas/gemm/arch35/` 现有 kernel 是否符合 §5"文件个数符合
   正常数量"的要求？~~
   → **已关闭**：`cgemm_kernel.h` 以外部链接导出全部五个 `*_do` 入口，新编译单元可直接调用，
   无需新增或修改任何 kernel 文件。功能实现仅新增 1 个源文件，远低于"冗余"阈值；
   反之若另起平行 kernel 则需复制约 1300 行，恰恰违反该条。

5. ~~**`aclblasSgemm3m` 可否作为实现参考**~~
   → **已关闭**：该接口并非复数 3m，而是接收沿 K 维拼接的 A1/A2/A3、B1/B2/B3 的 **FP32 实数**
   算子（`C = alpha*(A1·B1+A2·B2+A3·B3)+beta*C`），与本算子接口形状不同，
   仅在"Host 自带校验 helper、不跨编译单元共享"这一组织惯例上可借鉴。

---

# 6. 验收交付件对照

逐项对照任务书 §4「验收交付件」，列明产物、落位与当前状态。

| 序号 | 交付件 | 本设计对应产物 | 状态 |
|---|---|---|---|
| 1 | 算子设计文档 | 本文档，按官方模板（`resources/design_template.md`）编写，以 PR 形式提交至 `cann-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/10-aclblasCgemm3m-950/Euler_teng/docs/design.md` | 本次提交 |
| 2 | 自测用例及测试代码 | `test/gemm3m/cgemm3m/`：`README.md`（编译/运行/采集全流程，保证验收人可复现）、`cgemm3m_param.h`、`cgemm3m_golden.h`、`arch35/cgemm3m_test.cpp`、`arch35/cgemm3m_npu_wrapper.h`、`arch35/cgemm3m_test.csv`（官方全量 1200 条，逐字节一致）。精度 case 与性能 case 在 README 中按前缀分类列出（见 §4.4） | 代码已就绪，待真机跑通 |
| 3 | 自测报告 | 按官方模板（`docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2`）填写，须包含**用例参数、精度对比结果及截图、性能数据及截图**；配套产出 `tc_acc_results.csv`（逐条 matched_ratio、max_abs_error、**非有限值元素总数及其中通过数**）与 `tc_pf_results.csv`（逐条 NPU 耗时 / H100 标杆 / 倍率），以及两次执行的逐位比对记录 | **待真机采集**（§4.1.3 的精度结论为离线仿真，不作为验收依据） |
| 4 | 待验收代码地址 | 个人代码仓链接、分支名、算子目录（`blas/gemm/arch35/` 与 `test/gemm3m/cgemm3m/`）；**须在个人仓邀请账号 `Ascend-CANN` 作为开发者**；算子 README 按仓库规范提供（`blas/gemm/README.md` 新增 `aclblasCgemm3m` 接口章节，并同步 `docs/zh/api_list.md` 接口列表） | 文档改动已就绪，仓库与邀请待建 |

提交验收时另需在【说明】中填写验收代码 commit 链接（如遇特殊字符或字数限制，以 `.txt` 附件上传），
并确认已邀请 `Ascend-CANN` 为私仓开发者。

6. ~~**非法枚举返回码**：任务书 §2.4 写 `ACLBLAS_STATUS_INVALID_VALUE`，而仓内
   `aclblasCgemm` 在同一位置返回 `ACLBLAS_STATUS_INVALID_ENUM`，以哪个为准？~~
   → **已关闭**：依据有四项，且第四项为决定性证据——
   ① 任务书 §2.1 第 3 条明写"传入其他枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`"，§2.4 的
   transa / transb 两行亦同（§2.3 仅枚举可用状态码、未作规定）；
   ② 全量 1200 条用例的期望返回码只有 `SUCCESS`(1184) 与 `INVALID_VALUE`(16)，不含 `INVALID_ENUM`；
   ③ 同期社区任务设计文档采用同一口径（`aclblasCtrmm`，ops-blas !460）；
   ④ **同族 `sgemm3m_host.cpp:36-41` 本身即返回 `INVALID_VALUE`**——本设计沿用家族惯例，
   `aclblasCgemm` 才是特例。无需维护者裁决。

7. ~~**两套精度判据以哪套为准**：CSV 第 19、20 列的 MERE/MARE 判据要求"全部元素比对通过"，
   严格强于标准口径的 `matched_ratio ≥ 0.99`，两者结论可能相反。~~
   → **已关闭（真机实测）**：测试二进制逐条输出
   `matchedRatio=... (req=9.9000e-01)` 与 `per-element-limit=max(1.0000e-02, 32*ULP_at_each_element)`，
   `atol=rtol=1.2207e-04`。实际执行的是**标准口径**，不是"全部元素通过"。详见 §4.1.1。

8. ~~**n=2048 精度余量仅 8.4%，是否需要 K 拼接等缓解手段**~~
   → **已关闭（真机实测）**：实测 `max_abs_error = 3.91e-03 ~ 4.52e-03`，
   保守余量 2.21×，逐元素失败 0 个，约为仿真预测的一半。首版无需引入任何缓解手段。
   3m 相对 4m 的精度代价已在 §4.1.4 如实记录（全局最大误差 2.59×、289/802 条劣化 >1.5×）。
