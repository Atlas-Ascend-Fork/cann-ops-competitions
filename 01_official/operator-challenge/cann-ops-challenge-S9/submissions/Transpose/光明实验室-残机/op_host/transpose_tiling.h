#ifndef TRANSPOSE_TILING_H
#define TRANSPOSE_TILING_H

// Transpose 算子的 tiling 数据结构定义。
// 职责划分：op_host/transpose.cpp 的 TilingFunc 根据 shape/perm/dtype 选定执行
// 路径(mode)并计算下列字段，SaveToBuffer 序列化后由框架传给 device 侧；
// op_kernel/transpose.cpp 反序列化本结构后按 mode 分发到对应的转置实现。
// 各字段的计算逻辑与取值范围见 op_host/transpose.cpp 中的同名变量。

#include "register/tilingdata_base.h"

namespace optiling {

// 支持的最大张量维度数；超出直接判 tiling 失败。
constexpr uint32_t kTransposeMaxRank = 8;

BEGIN_TILING_DATA_DEF(TransposeTilingData)
  // 有效维度数：host 侧先剔除 size=1 的轴(CollapseSingletonAxes)再记录，
  // kernel 据此解读下面各数组的前 rank 项(尾部冗余项填 1)。
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  // 执行路径编号，与 host 侧 kModeXxx 常量(0~9)一一对应，kernel 据此分发。
  TILING_DATA_FIELD_DEF(uint32_t, mode);
  // 输入元素总数(等于输出元素总数)，generic 等路径用它划分任务。
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  // 输出元素数按 32B(DMA 对齐)向上取整后的元素数，kernel 据此确定
  // 输出 GM buffer 的可写边界，避免补齐写越界。
  TILING_DATA_FIELD_DEF(uint32_t, outputBufferLength);
  // 平均每个 core 分到的任务数 = ceil(taskTotal / blockDim)，
  // kernel 用 blockIdx 推算自己负责的任务区间。
  TILING_DATA_FIELD_DEF(uint32_t, taskLength);
  // 最后一个 core 实际处理的任务数(承担除模余量)；前面的 core 各处理 taskLength 个。
  TILING_DATA_FIELD_DEF(uint32_t, lastTaskLength);
  // 二维直接拷贝网格的规模：kModeBlockCopy 下为 (swapDimB, swapOuter)，
  // last-axis 路径下为行列 tile 网格；为 0 表示该路径不使用二维网格切核。
  TILING_DATA_FIELD_DEF(uint32_t, directDimA);
  TILING_DATA_FIELD_DEF(uint32_t, directDimB);
  // directDimA x directDimB 网格在 core 维上的二维切分因子，
  // blockDim = coreSplitA * coreSplitB，尽量让每核负载均衡。
  TILING_DATA_FIELD_DEF(uint32_t, coreSplitA);
  TILING_DATA_FIELD_DEF(uint32_t, coreSplitB);
  // 单个任务内的搬运单元元素数，含义随 mode 变化：macro tile 面积、
  // 021-batch 的 UB slot 容量、block copy 的块长等，kernel 据此组织循环。
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  // 输出最内层按 perm 天然连续的轴的元素乘积(GetContiguousOutputSuffix)；
  // kModeBlockCopy 以它为块单位直接拷贝，无需真正转置。
  TILING_DATA_FIELD_DEF(uint32_t, innerBlock);
  // 相邻交换轴位置：generic 路径为 perm 中唯一相邻对换的输出轴；
  // last-axis 路径为被换到末位的输入轴在输出中的轴号。
  TILING_DATA_FIELD_DEF(uint32_t, swapAxis);
  // 外层循环规模，含义随 mode 变化：block swap 前缀维乘积、reverse3d 的
  // batch 维、last-axis rank5 的分组大小、021-resident 的常驻输出列数。
  TILING_DATA_FIELD_DEF(uint32_t, swapOuter);
  // 参与转置的二维矩阵行/列规模；block swap 等路径会把若干相邻轴折叠成 A/B 两维。
  TILING_DATA_FIELD_DEF(uint32_t, swapDimA);
  TILING_DATA_FIELD_DEF(uint32_t, swapDimB);
  // 最内层连续维度的元素数(block swap 的后缀维乘积、reverse3d 的 D 维)，
  // 决定 GM 访问的连续段长度。
  TILING_DATA_FIELD_DEF(uint32_t, swapInner);
  // host 选定的 UB tile 尺寸(行 x 列)，决定单次转置规模、DMA 指令参数与
  // 分块数量，是各 tile 化路径的核心参数。
  TILING_DATA_FIELD_DEF(uint32_t, swapTileA);
  TILING_DATA_FIELD_DEF(uint32_t, swapTileB);
  // collapse 后输入各维大小(尾部不足处填 1)。
  TILING_DATA_FIELD_DEF_ARR(uint32_t, kTransposeMaxRank, inputDims);
  // 输入各维元素步长(最内层为 1)，generic/gather 路径用于计算源 GM 地址。
  TILING_DATA_FIELD_DEF_ARR(uint32_t, kTransposeMaxRank, inputStrides);
  // 输出各维大小，outputDims[i] = inputDims[perm[i]]。
  TILING_DATA_FIELD_DEF_ARR(uint32_t, kTransposeMaxRank, outputDims);
  // 输出各维元素步长，与 inputStrides 配合完成逐元素地址映射。
  TILING_DATA_FIELD_DEF_ARR(uint32_t, kTransposeMaxRank, outputStrides);
  // collapse 后的置换：输出第 i 维来自输入第 perm[i] 维。
  TILING_DATA_FIELD_DEF_ARR(uint32_t, kTransposeMaxRank, perm);
END_TILING_DATA_DEF;

// 将上述结构注册为 Transpose 算子的 tiling 结构，框架据此完成 host 到 kernel 的序列化传输。
REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)

}  // namespace optiling

#endif
