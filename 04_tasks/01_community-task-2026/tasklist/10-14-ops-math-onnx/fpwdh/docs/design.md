# ops-math 算子 ONNX 插件 protobuf 解耦迁移设计

作者：fpwdh
状态：设计定稿；实现与实机验证进行中（UT 与 om 两轮比对结果将在完成后更新至本文档）
版本：2026-10-24
实现基线：cann/ops-math master（迁移范围静态扫描与任务书清单核对一致：57 个插件文件、79 处 `.ParseParamsFn(` 注册调用、124 个文件×算子条目）

## 1. 需求背景（required）

### 1.1 需求来源

- [昇腾社区任务：ops-math 算子 onnx 插件 protobuf 解耦](https://www.hiascend.com/activities/task-center/details/18e1d3beae784812b2562460542ad6b3?menu=tasks)
- [任务书与官方用例包](https://www.hiascend.com/p/resource/202610/21e704e530834ebdb12a5c7a5d2baa51.zip)
- [迁移参考 Issue #2892](https://gitcode.com/cann/ops-math/issues/2892)（关联已合入 PR：ReverseSequence、npu_indexing 等插件的同类迁移）
- [社区任务流程与注意事项](https://gitcode.com/org/cann/discussions/39)

### 1.2 背景介绍

ops-math 仓现有 124 个 ONNX 算子插件条目（57 个插件源文件）通过 `ParseParamsFn` 注册参数解析函数，签名依赖 `google::protobuf::Message`（实际传入 `ge::onnx::NodeProto`），插件内部遍历 protobuf attribute 并按 `i/f/s` 等字段取值，存在三类问题：

1. 与 protobuf 生成代码强耦合（依赖 ge_onnx.pb.h，proto 变动波及全部插件）；
2. 编译依赖重（拉入 protobuf 头文件与生成代码）；
3. UT 成本高（单测必须构造 protobuf 对象）。

本任务将上述条目全部迁移为 `ParseParamsByOperatorFn`：以 `ge::Operator` 为入参，属性经 `op_src.GetAttr("attribute")` 获取 JSON 字符串后用仓内 nlohmann/json 解析，彻底摆脱 protobuf 依赖，同时保持解析语义与功能不变。

### 1.3 迁移范围边界

- 仅修改各算子目录下 `framework/*_onnx_plugin.cpp` 与对应 framework UT；
- 不涉及算子 kernel、tiling、算子原型；
- `ParseOpToGraphFn` 子图拆图函数（如 Multinomial）不迁移、保持源码等价；
- 涉及 `DynamicInputRegister` 的插件保持动态输入登记逻辑不变。

## 2. 需求分析（required）

### 2.1 数据流对比

| | 迁移前 | 迁移后 |
| --- | --- | --- |
| 解析入口 | `ParseParamsFn(const Message*, ge::Operator&)`，reinterpret/dynamic_cast 到 `NodeProto` | `ParseParamsByOperatorFn(const ge::Operator&, ge::Operator&)` |
| 属性来源 | 遍历 `node->attribute()`，`attr.name()/i()/f()/s()/ints()` | `op_src.GetAttr("attribute", ge::AscendString&)` → nlohmann/json 解析 `{"attribute":[...]}` 数组 |
| 节点名 | `node->name()` | `GetOpName(op_src)`，`op_dest.SetAttr("name", ...)` |
| 输入输出数量 | `node->input_size()/output_size()` | 优先 `op_src.GetAttr("input"/"output", std::vector<ge::AscendString>&)` 名称列表（GE `Message2Operator` 将 NodeProto 重复字符串字段保存为 Operator 列表属性）；直接构造 Operator 的 UT 路径无该属性时回退 `GetInputsSize()/GetOutputsSize()`（与迁移前空 NodeProto `input_size()=0` 行为一致） |
| 依赖头 | onnx_common.h（拉入 ge_onnx.pb.h）+ proto 头 | graph/operator.h、math_onnx_plugin_util.h、log/log.h、register/register.h、nlohmann/json.hpp |

### 2.2 attribute JSON 契约（任务书 §2.4）

| 字段 | 形态 | 约定 |
| --- | --- | --- |
| `attribute` | 顶层键，JSON array | 非法 JSON → 解析函数记录日志并返回 FAILED（仅限解析 attribute JSON 的注册路径） |
| `name` | string | 与 ONNX 标准属性名一致；未匹配到算子属性时忽略（保持迁移前行为） |
| `type` | int | FLOAT=1、INT=2、STRING=3、FLOATS=6、INTS=7；属性名匹配但 type 不符时按缺失处理（UT `Ignore*WithMismatchedType` 系列覆盖） |
| `i` | int64（数字） | 属性存在但键缺省（proto3 零值省略）取 0，不回落默认值 |
| `f` | **字符串承载**（如 `"f":"0.25"`），兼容 GE Pb2Json 对次正规浮点格式化失败时回退的 JSON number | 读 string 后 `atof` 等价转 float，解析失败返回 FAILED；键缺省取 0.0，不回落默认值 |
| `s` | string | `s` 键缺省取空串 |
| `ints`/`floats` | JSON array（数字） | 缺省/空数组按空列表透传 |

### 2.3 语义保持规则（迁移必须逐项保持，官方 624 UT 为行为契约）

1. 属性缺失：保持原有默认值行为并返回 SUCCESS（如 Flatten `axis` 默认 1、Squeeze `axes` 默认空列表）；原实现缺属性即失败的仍返回 FAILED（如 Cast 缺 `to`）。
2. 属性非法（JSON 解析异常/未知异常）：记录日志并返回 FAILED；各分支返回值（SUCCESS/FAILED/PARAM_INVALID 等）与迁移前一致。
3. **不解析 attribute JSON 的注册路径豁免**（75 条目/8 文件：similar_opp 63、sequence 6、Mean 1、Sum 1、Squeeze opset13+ 1、Unsqueeze opset13+ 1、Slice opset10~18 1、Clip opset11 1）：迁移后同样不解析 JSON，非法 JSON 保持原 SUCCESS 语义且不写属性，实现与 UT 都不改。
4. **"仅做合法性校验"的注册路径**（Max experimental/Max/Min/Xor，4 文件 7 条目）：按原语义校验，非法输入仍返回 FAILED。
5. Cast 专项：迁移前 `if (om == -1)` 判据永不成立（`GetOmDtypeFromOnnxDtype` 对未命中返回 `ge::DT_UNDEFINED` 而非 -1），迁移后按工具函数真实返回约定修正为 `DT_UNDEFINED` 判断，使"不支持 dtype"在解析阶段即返回 FAILED（任务书 §2.1 第 7 条指定的唯一例外）。
6. 保留各插件原有属性遍历顺序、首个匹配后 break、重复属性覆盖顺序等细节差异；除解耦外不改任何其他逻辑。

### 2.4 迁移范围

与任务书清单一致（57 文件/124 条目/79 处注册调用），按文件聚合的完整对照表见附录；同一文件多算子经宏注册（如 similar_opp 63 算子、sequence 6 算子）按文件迁移、按条目对账。

## 3. 详细设计（required）

### 3.1 解析函数统一骨架（照仓内已迁移插件 reverse_sequence 范式）

```cpp
#include "graph/operator.h"
#include "math_onnx_plugin_util.h"
#include "log/log.h"
#include "register/register.h"
#include "nlohmann/json.hpp"

namespace domi {
using json = nlohmann::json;

static Status ParseParamsXxx(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // 1. 原有默认值/动态端口登记逻辑原位保持（顺序不变）
    // 2. op_src.GetAttr("attribute", attrs_string)：
    //    成功 → json::parse → 遍历 "attribute" 数组，按 name 匹配、按 2.2 契约取值
    //    属性不存在 → 保持原默认值行为，返回 SUCCESS
    // 3. 捕获 nlohmann::json::exception 与未知异常 → OP_LOGE + FAILED
}
REGISTER_CUSTOM_OP("Xxx")
    .FrameworkType(ONNX)
    .OriginOpType({...})               // 原样保留
    .ParseParamsByOperatorFn(ParseParamsXxx)
    .ImplyType(ImplyType::TVM);
}
```

异常只在解析边界捕获，不穿越注册回调；原有业务校验返回的 `PARAM_INVALID` 等状态原样保留。

### 3.2 公共工具复用

优先复用仓内已有无 protobuf 依赖的 `common/inc/framework/math_onnx_plugin_util.h`：`GetOpName`、`GetOmDtypeFromOnnxDtype`、`Vec2Tensor`、`CreateScalar`、`ChangeFormatFromOnnx`、`DataTypeOnnx`。不为统一而新建平行公共框架；确需新增公共头文件时使用仓前缀命名（任务书 §2.1 第 6 条）。JSON 使用仓构建体系自带的 nlohmann/json，不另引版本。

字段读取统一约定：`i`/`ints`/`floats` 按数字读取（`contains` 判键、缺省取零值/空列表）；`f` 按 string 读取后转 float（兼容 number 回退形态）；`s` 按 string 读取（缺省空串）；`type` 用本地枚举常量比对，替代 protobuf 枚举。

### 3.3 分类迁移策略

| 类别 | 文件/条目 | 迁移动作 |
| --- | --- | --- |
| A：遍历 attribute 解析 | 其余全部（约 42 文件） | 签名/注册/属性读取/name/输入输出数量全量迁移，逐属性保语义 |
| B：不解析 JSON 的注册路径 | similar_opp(63)、sequence(6)、Mean、Sum、Squeeze opset13+、Unsqueeze opset13+、Slice opset10~18、Clip opset11（共 75 条目/8 文件） | 只换签名与输入输出来源（名称列表优先），不引入 JSON 解析；宏注册文件保持宏结构 |
| C：仅合法性校验路径 | Max(experimental)、Max、Min、Xor（4 文件 7 条目） | 保持校验语义与 FAILED 返回 |
| 专项 | Cast | dtype 判据修正为 `DT_UNDEFINED`（§2.3 第 5 条） |
| 不动 | 各 ParseOpToGraphFn / DynamicInputRegister / OriginOpType / ImplyType | 源码 token 等价 |

### 3.4 UT 设计

- 57 个官方 UT 文件（624 用例）按官方建议目录落位（`<算子目录>/tests/ut/framework/`；common 下插件统一 `common/tests/ut/framework/`），include 用仓内原生相对路径，直接 include 插件源文件（与仓内已迁移插件 UT 写法一致）；
- 仓内 `cmake/ut.cmake` 以 `test_*_onnx_plugin.cpp` 自动收集 framework 用例；对缺少 UT 构建入口的 5 个目录（conversion/flatten、conversion/squeeze、conversion/unsqueeze、math/is_inf、random/dsa_random_uniform）按 `math/cast/tests/ut/framework/CMakeLists.txt` 模板补 `add_modules_ut_sources` 入口；
- 双配置构建：默认配置运行 615 用例（experimental 关闭时 `test_max_experimental_onnx_plugin.cpp` 9 例不编入），`--experimental/-DENABLE_EXPERIMENTAL=ON` 配置运行 experimental Max 9 例；两配置合计 624 全绿为通过标准；
- UT 用 ge::Operator 直接构造输入（`SetAttr("attribute", JSON字符串)`），不构造任何 protobuf 对象。

## 4. 可维可测分析（required）

### 4.1 UT 验证

`bash build.sh -u --ophost`（OP_HOST_UT 开关）构建后 `--gtest_filter='Onnx*PluginTest.*'` 全量运行，正常+experimental 双配置逐条对账 624/624，保留完整构建与执行日志。

### 4.2 om 转换精度验证（两轮逐位比对）

官方判据（任务书 §3.2）：同一 onnx 模型、同一输入、同一 ATC 参数，迁移前/后插件各转 om 并推理，输出**逐位（字节）一致**（±0 符号、NaN 载荷计入；固定 seed 的 4 个 Random 算子同样逐位；Multinomial 按 meta 声明的 behavior 档；实测不可复现用例仅限 meta 显式声明档位）。

执行方式：官方 `two_round_compare.sh` 全量 56 个生成脚本；fp32 用例 `--precision_mode=must_keep_origin_dtype` 两轮同参透传；每例留存 `plugin_evidence_*.txt`（`[Register][Check] Plugin of X already registered, skip` 证明自定义插件生效）；迁移前旧插件如报 `undefined symbol: ...ge::onnx::AttributeProto...` 按 `verify_om/compat/README.md` 处理（迁移后插件不需要）。

**用例包修订原则：默认零修订**。确因环境限制（如环境缺自定义算子可执行实现导致的转换/推理失败）需申报 `expect_convert_failure`/`expect_infer_failure` 时，逐例记录：算子/变体、平台、CANN 版本、报错原文、迁移前同样失败的证据，在验收说明中说明并提供代码（任务书 §3.2"用例包修订许可"）。

### 4.3 自证与变异测试

- 每类断言两个独立来源：UT（解析契约）+ om 两轮（端到端行为）+ 静态检查（grep 无 protobuf 残留）；
- 交付前变异测试 ≥5 处（改默认值/删缺省键分支/type 白名单破坏/删异常捕获/golden 错位），每次官方 UT 必须报警，证明尺子有判别力；
- `git diff --check` 与 `bash build.sh --onnxplugin -j8`（正常+experimental）通过后按文件分组提交代码 PR。

### 4.4 交付物（task_submission，按任务书 §4 模板）

1 自验证步骤说明.md（含单算子迁移自查表、用例包修订说明——如无修订则明确声明零修订）；2.1 精度自验证报告.xlsx（UT 624/624 + om 两轮比对逐例结果）；2.2 精度自验证日志.log（原始日志逐文件拼接）；3.1/3.2/4.1/4.2 标注不涉及并引用任务书 §3.3/§3.4。

## 5. 兼容性与风险

| 风险 | 应对 |
| --- | --- |
| attribute JSON 实际形态与文档不符 | 以真机 `std::cout` 打印的真实 JSON 为准校准 UT 输入表示（期望值不变）；设计即按"名称列表优先+描述符回退"双通道规避 |
| `f` 浮点次正规数被 GE 序列化为 number 而非 string | 字段读取同时接受 string 与 number 形态，转 float 失败返回 FAILED；UT 覆盖 |
| 实验目录（experimental）漏测 | 双配置构建分别跑 615+9 并逐条对账 |
| 误测内置插件 | 自定义 OPP 目录注入 + `plugin_evidence` 证据行归档 |
| 基线与开发期间上游漂移 | 固定基线 commit；PR 前重跑受影响验证 |
| 旧接口插件在 CANN 9.1.0 加载失败 | compat 兼容库仅用于迁移前一轮，留痕；迁移后插件无此依赖 |
| 迁移前即转换失败的用例（平台侧限制） | 逐例反馈验收说明 + meta 申报两轮一致失败，不虚计通过 |

## 附录：迁移范围（57 文件 × 124 条目，与任务书 §1 表一致）

| 源文件（相对 ops-math） | 算子条目 |
| --- | --- |
| common/src/framework/concat_from_sequence_onnx_plugin.cpp | ConcatFromSequence |
| common/src/framework/deformable_offsets_onnx_plugin.cpp | DeformableOffsets |
| common/src/framework/einsum_onnx_plugin.cpp | Einsum |
| common/src/framework/eyelike_onnx_plugin.cpp | EyeLike |
| common/src/framework/phony_concat_onnx_plugin.cpp | PhonyConcat |
| common/src/framework/phony_split_onnx_plugin.cpp | PhonySplit |
| common/src/framework/random_normal_like_onnx_plugin.cpp | RandomNormalLike |
| common/src/framework/random_uniform_like_onnx_plugin.cpp | RandomUniformLike |
| common/src/framework/reduceL1_onnx_plugin.cpp | ReduceL1 |
| common/src/framework/reduceL2_onnx_plugin.cpp | ReduceL2 |
| common/src/framework/repeat_with_pad_onnx_plugin.cpp | RepeatWithPad |
| common/src/framework/sequence_onnx_plugin.cpp | SequenceAt/Construct/Empty/Erase/Insert/Length（6） |
| common/src/framework/similar_opp_onnx_plugin.cpp | Abs…Where（63 算子，宏注册） |
| common/src/framework/split_to_sequence_onnx_plugin.cpp | SplitToSequence |
| conversion/clip_by_value/framework/clip_onnx_plugin.cpp | Clip |
| conversion/concat/framework/concat_onnx_plugin.cpp | Concat |
| conversion/depth_to_space/framework/depth_to_space_onnx_plugin.cpp | DepthToSpace |
| conversion/flatten/framework/flatten_onnx_plugin.cpp | Flatten |
| conversion/pad_v3/framework/pad_onnx_plugin.cpp | Pad |
| conversion/reshape/framework/reshape_onnx_plugin.cpp | Reshape |
| conversion/slice/framework/npu_slice_onnx_plugin.cpp | NPUSlice |
| conversion/space_to_depth/framework/space_to_space_onnx_plugin.cpp | SpaceToDepth |
| conversion/split_d/framework/split_onnx_plugin.cpp | Split |
| conversion/squeeze/framework/squeeze_onnx_plugin.cpp | Squeeze |
| conversion/strided_slice_v2/framework/slice_onnx_plugin.cpp | Slice |
| conversion/transpose/framework/transpose_onnx_plugin.cpp | Transpose |
| conversion/trilu/framework/trilu_onnx_plugin.cpp | Trilu |
| conversion/unsqueeze/framework/unsequeeze_onnx_plugin.cpp | Unsqueeze |
| experimental/math/asinh/framework/max_onnx_plugin.cpp | Max |
| math/accumulate_nv2/framework/mean_onnx_plugin.cpp | Mean |
| math/accumulate_nv2/framework/sum_onnx_plugin.cpp | Sum |
| math/addcmul/framework/addcmul_onnx_plugin.cpp | Addcmul |
| math/arg_max_v2/framework/argmax_onnx_plugin.cpp | ArgMax |
| math/arg_min/framework/argmin_onnx_plugin.cpp | ArgMin |
| math/bitwise_xor/framework/xor_onnx_plugin.cpp | Xor |
| math/cast/framework/cast_onnx_plugin.cpp | Cast |
| math/cumsum/framework/cumsum_onnx_plugin.cpp | CumSum |
| math/is_inf/framework/isinf_onnx_plugin.cpp | IsInf |
| math/maximum/framework/max_onnx_plugin.cpp | Max |
| math/minimum/framework/min_onnx_plugin.cpp | Min |
| math/mod/framework/mod_onnx_plugin.cpp | Mod |
| math/one_hot/framework/onehot_onnx_plugin.cpp | OneHot |
| math/reduce_log_sum_exp/framework/reduce_log_sum_exp_onnx_plugin.cpp | ReduceLogSumExp |
| math/reduce_max/framework/reduce_max_onnx_plugin.cpp | ReduceMax |
| math/reduce_mean/framework/reduce_mean_onnx_plugin.cpp | ReduceMean |
| math/reduce_min/framework/reduce_min_onnx_plugin.cpp | ReduceMin |
| math/reduce_prod/framework/reduce_prod_onnx_plugin.cpp | ReduceProd |
| math/reduce_sum/framework/reduce_log_sum_onnx_plugin.cpp | ReduceLogSum |
| math/reduce_sum/framework/reduce_sum_onnx_plugin.cpp | ReduceSum |
| math/right_shift/framework/bitshift_onnx_plugin.cpp | BitShift |
| math/sign_bits_pack/framework/npu_sign_bits_pack_onnx_plugin.cpp | NPUSignBitsPack |
| math/sign_bits_unpack/framework/npu_sign_bits_unpack_onnx_plugin.cpp | NPUSignBitsUnpack |
| math/square_sum_v1/framework/reduce_sum_square_onnx_plugin.cpp | ReduceSumSquare |
| math/topk/framework/topk_onnx_plugin.cpp | TopK |
| random/dsa_random_uniform/framework/random_uniform_onnx_plugin.cpp | RandomUniform |
| random/stateless_random_normal_v2/framework/random_normal_onnx_plugin.cpp | RandomNormal |
| random/stateless_sample_multinomial/framework/multinomial_onnx_plugin.cpp | Multinomial |

> 注：附录中 `space_to_depth` 行文件名以任务书为准（`space_to_depth_onnx_plugin.cpp`）；124 条目 = 上表逐文件条目数之和，与任务书 §1 迁移范围表逐行核对无缺漏。
