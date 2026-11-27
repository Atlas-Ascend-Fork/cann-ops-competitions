/*!
 * \file ctc_loss_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CTCLOSS_TILING_DATA_H_
#define _CTCLOSS_TILING_DATA_H_

struct CtcLossTilingData {
    int64_t timeStep = 0;
    int64_t batchSize = 0;
    int64_t numClasses = 0;
    int64_t maxTargetLength = 0;
    int64_t targetStride = 0;
    int64_t alphaAlign = 0;
    int64_t targetAlign = 0;
    int64_t sliceLength = 0;
    int64_t sliceAlign = 0;
    int64_t sliceCount = 0;
    int64_t sliceTail = 0;
    int64_t tasksPerCore = 0;
    int64_t tailCoreCount = 0;
    int64_t reduction = 0; // 0: none, 1: mean, 2: sum
    int64_t blank = 0;
    int64_t zeroInfinity = 0;
    int64_t probPingPong = 0;
    int64_t probRowAlign = 0;
    int64_t probBandRows = 0;
    int64_t pairMode = 0;
};
#endif
