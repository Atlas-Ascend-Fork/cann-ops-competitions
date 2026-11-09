// Greater 算子的 TilingData 定义。
//
// 这里的每个字段都是 Host 侧 TilingFunc 写入、Kernel 侧 GET_TILING_DATA 读出的
// Host/Kernel ABI。字段顺序、类型和数量一经使用就不能随意调整：Kernel 按同一份
// 结构体布局解析这块内存，任何错位都会静默读到错误的形状或步长。
//
// 字段按用途分为五组：
//   1. 全局工作量：总元素数、两个输入各自的元素数、每核负责的元素数、单元素字节数。
//   2. 路径选择：mode（广播关系）、标量输入下标、被广播输入下标。
//   3. workspace 兜底路径：三个 workspace 偏移量，-1 表示该输入无需物化。
//   4. DirectBroadcast 规则广播计划：连续低维长度、段数、外层维度与步长。
//   5. BatchBroadcast 与通用 fallback：批处理参数，以及完整的 dims/strides 视图。
#ifndef GREATER_TILING_H
#define GREATER_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GreaterTilingData)
  // 第 1 组：全局工作量。
  // total_elems 是输出元素数，也是广播后的逻辑规模；
  // x1/x2_total_elems 是各输入的真实元素数，标量广播时其中一个为 1；
  // elems_per_core 已由 Host 按 512B 取整，Kernel 用它切分核间归属并裁剪尾块。
  TILING_DATA_FIELD_DEF(int64_t, total_elems);
  TILING_DATA_FIELD_DEF(int64_t, x1_total_elems);
  TILING_DATA_FIELD_DEF(int64_t, x2_total_elems);
  TILING_DATA_FIELD_DEF(int64_t, elems_per_core);
  TILING_DATA_FIELD_DEF(int64_t, rank);
  // 第 2 组：路径选择。
  // mode 取值见 op_host/greater.cpp 与 op_kernel/greater.cpp 中同名常量，必须保持一致；
  // scalar_input_index 为 1 或 2，指出哪个输入是单元素标量；
  // direct_broadcast_input_index 指出规则广播里哪个输入是被重复读取的那一路。
  TILING_DATA_FIELD_DEF(int64_t, mode);
  TILING_DATA_FIELD_DEF(int64_t, scalar_input_index);
  TILING_DATA_FIELD_DEF(int64_t, direct_broadcast_input_index);
  TILING_DATA_FIELD_DEF(int64_t, elem_size);
  // 第 3 组：workspace 兜底路径的偏移量，-1 表示该输入本身已是满形状、无需物化。
  TILING_DATA_FIELD_DEF(int64_t, x1_workspace_offset);
  TILING_DATA_FIELD_DEF(int64_t, x2_workspace_offset);
  TILING_DATA_FIELD_DEF(int64_t, scratch_workspace_offset);
  // 第 4 组：DirectBroadcast 计划。
  // direct_low_size 是可连续搬运的最低维长度，direct_total_segments 是这样的段总数；
  // direct_out_strideN 为输出侧步长，direct_src_strideN 为被广播输入的步长，
  // 其中步长为 0 的轴就是该输入被重复读取的广播轴。
  TILING_DATA_FIELD_DEF(int64_t, direct_low_size);
  TILING_DATA_FIELD_DEF(int64_t, direct_total_segments);
  TILING_DATA_FIELD_DEF(int64_t, direct_outer_rank);
  TILING_DATA_FIELD_DEF(int64_t, direct_outer_dim0);
  TILING_DATA_FIELD_DEF(int64_t, direct_outer_dim1);
  TILING_DATA_FIELD_DEF(int64_t, direct_outer_dim2);
  TILING_DATA_FIELD_DEF(int64_t, direct_outer_dim3);
  TILING_DATA_FIELD_DEF(int64_t, direct_out_stride0);
  TILING_DATA_FIELD_DEF(int64_t, direct_out_stride1);
  TILING_DATA_FIELD_DEF(int64_t, direct_out_stride2);
  TILING_DATA_FIELD_DEF(int64_t, direct_out_stride3);
  TILING_DATA_FIELD_DEF(int64_t, direct_src_stride0);
  TILING_DATA_FIELD_DEF(int64_t, direct_src_stride1);
  TILING_DATA_FIELD_DEF(int64_t, direct_src_stride2);
  TILING_DATA_FIELD_DEF(int64_t, direct_src_stride3);
  // 第 5 组：BatchBroadcast 参数。
  // batch_split_axis 是被拆分的轴，batch_mid_former 是该轴上共享同一 pattern 的行数，
  // batch_pattern_reciprocal 是 pattern 长度的倒数，供 Kernel 用乘法代替整除。
  TILING_DATA_FIELD_DEF(int64_t, batch_split_axis);
  TILING_DATA_FIELD_DEF(int64_t, batch_mid_former);
  TILING_DATA_FIELD_DEF(int64_t, batch_total_tiles);
  TILING_DATA_FIELD_DEF(float, batch_pattern_reciprocal);
  // 通用 fallback 视图：最高 8 维的输出形状 dims、输出步长 out_stride，
  // 以及两个输入各自的步长 x1_stride / x2_stride（广播轴上步长为 0）。
  TILING_DATA_FIELD_DEF(int64_t, dim0);
  TILING_DATA_FIELD_DEF(int64_t, dim1);
  TILING_DATA_FIELD_DEF(int64_t, dim2);
  TILING_DATA_FIELD_DEF(int64_t, dim3);
  TILING_DATA_FIELD_DEF(int64_t, dim4);
  TILING_DATA_FIELD_DEF(int64_t, dim5);
  TILING_DATA_FIELD_DEF(int64_t, dim6);
  TILING_DATA_FIELD_DEF(int64_t, dim7);
  TILING_DATA_FIELD_DEF(int64_t, out_stride0);
  TILING_DATA_FIELD_DEF(int64_t, out_stride1);
  TILING_DATA_FIELD_DEF(int64_t, out_stride2);
  TILING_DATA_FIELD_DEF(int64_t, out_stride3);
  TILING_DATA_FIELD_DEF(int64_t, out_stride4);
  TILING_DATA_FIELD_DEF(int64_t, out_stride5);
  TILING_DATA_FIELD_DEF(int64_t, out_stride6);
  TILING_DATA_FIELD_DEF(int64_t, out_stride7);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride0);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride1);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride2);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride3);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride4);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride5);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride6);
  TILING_DATA_FIELD_DEF(int64_t, x1_stride7);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride0);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride1);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride2);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride3);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride4);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride5);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride6);
  TILING_DATA_FIELD_DEF(int64_t, x2_stride7);
  // workspace 广播用的规范化 4 维视图：canon_x1 / canon_x2 是两个输入压缩后的形状，
  // canon_out 是输出形状。某一维上 canon_x1 != canon_out 就表示该维需要展开。
  TILING_DATA_FIELD_DEF(int64_t, canon_x10);
  TILING_DATA_FIELD_DEF(int64_t, canon_x11);
  TILING_DATA_FIELD_DEF(int64_t, canon_x12);
  TILING_DATA_FIELD_DEF(int64_t, canon_x13);
  TILING_DATA_FIELD_DEF(int64_t, canon_x20);
  TILING_DATA_FIELD_DEF(int64_t, canon_x21);
  TILING_DATA_FIELD_DEF(int64_t, canon_x22);
  TILING_DATA_FIELD_DEF(int64_t, canon_x23);
  TILING_DATA_FIELD_DEF(int64_t, canon_out0);
  TILING_DATA_FIELD_DEF(int64_t, canon_out1);
  TILING_DATA_FIELD_DEF(int64_t, canon_out2);
  TILING_DATA_FIELD_DEF(int64_t, canon_out3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)

}  // namespace optiling

#endif
