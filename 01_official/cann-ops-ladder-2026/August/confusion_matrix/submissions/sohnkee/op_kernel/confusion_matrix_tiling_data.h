/*!
 * \file confusion_matrix_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CONFUSIONMATRIX_TILING_DATA_H_
#define _CONFUSIONMATRIX_TILING_DATA_H_

#include <cstdint>

constexpr int32_t CONFUSION_MATRIX_DTYPE_INT32 = 0;
constexpr int32_t CONFUSION_MATRIX_DTYPE_INT64 = 1;
constexpr int32_t CONFUSION_MATRIX_DTYPE_FLOAT32 = 2;
constexpr int32_t CONFUSION_MATRIX_RESERVED_SPLIT_OUTPUT = 1;
constexpr int32_t CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE = 2;

struct ConfusionMatrixTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 1;
    int64_t numClasses = 0;
    int32_t labelsDtype = 0;
    int32_t predictionsDtype = 0;
    int32_t weightsDtype = 0;
    int32_t outputDtype = 0;
    int32_t hasWeights = 0;
    int32_t reserved = 0;
};

#endif
