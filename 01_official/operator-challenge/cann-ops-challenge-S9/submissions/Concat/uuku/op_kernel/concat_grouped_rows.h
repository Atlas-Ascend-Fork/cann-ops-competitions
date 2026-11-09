// Included inside ConcatKernel; not a standalone compilation unit.
// Aligned B32 groups and their existing copy pipeline. Admission lives in the
// generic dispatcher; helpers below retain the original write order and queue depth.

    // Keep every unit on whole 32-byte output blocks. Adjacent atomic groups
    // that are individually aligned can share one UB->GM 2D write, while an
    // atomic group containing a partial block stays on the proven row path.
    template <typename T>
    __aicore__ inline bool TryProcessCoalescedAlignedGroups(uint32_t localElems,
                                                                  GlobalTensor<T>& yGm)
    {
        constexpr uint32_t kAlignElems = 32U / sizeof(T);
        if (sizeof(T) != 4U || alignedGroupCount == 0U || localElems == 0U ||
            outerSize == 0U) {
            return false;
        }

        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            if (AxisLen(inputIdx) == 0U) {
                return false;
            }
        }

        const uint32_t blockNum = GetBlockNum();
        if (blockNum == 0U || alignedGroupCount < blockNum) {
            return false;
        }

        // Bound one packed group by UB capacity and by a per-core output
        // window. The latter keeps enough independent units to occupy all
        // cores even when every input is already 32-byte aligned.
        uint32_t groupLimit = localElems / outerSize;
        const uint32_t coreWindow = outputAxisSize / blockNum;
        groupLimit = MinU32(groupLimit, coreWindow);
        groupLimit = AlignDownU32(groupLimit, kAlignElems);
        if (groupLimit < kAlignElems) {
            return false;
        }

        const uint32_t unitCount = CountCoalescedUnits<T>(groupLimit);
        if (unitCount < blockNum || unitCount >= alignedGroupCount) {
            return false;
        }

        const uint32_t blockIdx = GetBlockIdx();
        AlignedGroupPipelineState pendingGroup{0U, 0U, 0U, 0U, false};
        uint32_t unitIdx = 0U;
        uint32_t atomicInputBegin = 0U;
        uint32_t atomicAxisBegin = 0U;
        uint32_t axisEnd = 0U;
        uint32_t pendingInputBegin = 0U;
        uint32_t pendingAxisBegin = 0U;
        uint32_t pendingElems = 0U;

        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            axisEnd += AxisLen(inputIdx);
            if (axisEnd == atomicAxisBegin || (axisEnd % kAlignElems) != 0U) {
                continue;
            }

            const uint32_t atomicElems = axisEnd - atomicAxisBegin;
            const bool canPackAtomic =
                IsAtomicGroupAligned<T>(atomicInputBegin, inputIdx + 1U) &&
                atomicElems <= groupLimit;
            if (canPackAtomic) {
                if (pendingElems != 0U && pendingElems + atomicElems > groupLimit) {
                    ProcessCoalescedUnit<T>(unitIdx, blockNum, blockIdx,
                                                 pendingInputBegin, atomicInputBegin, yGm,
                                                 pendingAxisBegin, pendingElems, localElems,
                                                 pendingGroup);
                    ++unitIdx;
                    pendingElems = 0U;
                }
                if (pendingElems == 0U) {
                    pendingInputBegin = atomicInputBegin;
                    pendingAxisBegin = atomicAxisBegin;
                }
                pendingElems += atomicElems;
            } else {
                if (pendingElems != 0U) {
                    ProcessCoalescedUnit<T>(unitIdx, blockNum, blockIdx,
                                                 pendingInputBegin, atomicInputBegin, yGm,
                                                 pendingAxisBegin, pendingElems, localElems,
                                                 pendingGroup);
                    ++unitIdx;
                    pendingElems = 0U;
                }
                ProcessCoalescedUnit<T>(unitIdx, blockNum, blockIdx,
                                             atomicInputBegin, inputIdx + 1U, yGm,
                                             atomicAxisBegin, atomicElems, localElems,
                                             pendingGroup);
                ++unitIdx;
            }

            atomicInputBegin = inputIdx + 1U;
            atomicAxisBegin = axisEnd;
        }

        if (pendingElems != 0U) {
            ProcessCoalescedUnit<T>(unitIdx, blockNum, blockIdx,
                                         pendingInputBegin, inputCount, yGm,
                                         pendingAxisBegin, pendingElems, localElems,
                                         pendingGroup);
        }
        FlushAlignedGroupPipeline<T>(pendingGroup, yGm);
        return true;
    }

    // Directly packed groups use only 32-byte-aligned LocalTensor destinations.
    // Keep one ready group in the bound queue so the next group's MTE2 reads can
    // overlap the prior group's MTE3 write. Mixed-alignment groups keep v56's
    // synchronous copy route.
    template <typename T>
    __aicore__ inline void ProcessCoalescedUnit(uint32_t unitIdx, uint32_t blockNum,
                                                     uint32_t blockIdx, uint32_t groupStart,
                                                     uint32_t groupEnd, GlobalTensor<T>& yGm,
                                                     uint32_t groupAxisOffset,
                                                     uint32_t groupElems,
                                                     uint32_t localElems,
                                                     AlignedGroupPipelineState& pendingGroup)
    {
        if ((unitIdx % blockNum) != blockIdx) {
            return;
        }

        if (CanCopyAlignedGroupInOneOutput<T>(groupStart, groupEnd, groupAxisOffset,
                                              groupElems, localElems)) {
            QueueMteGroupedRows<T>(groupStart, groupEnd, 0U, outerSize, groupElems,
                                   !pendingGroup.valid);
            FlushAlignedGroupPipeline<T>(pendingGroup, yGm);
            pendingGroup.groupStart = groupStart;
            pendingGroup.groupEnd = groupEnd;
            pendingGroup.groupAxisOffset = groupAxisOffset;
            pendingGroup.groupElems = groupElems;
            pendingGroup.valid = true;
            return;
        }

        FlushAlignedGroupPipeline<T>(pendingGroup, yGm);
        CopyGroupOrRows<T>(groupStart, groupEnd, yGm, groupAxisOffset,
                                groupElems, localElems);
    }

    template <typename T>
    __aicore__ inline void FlushAlignedGroupPipeline(AlignedGroupPipelineState& pendingGroup,
                                                    GlobalTensor<T>& yGm)
    {
        if (!pendingGroup.valid) {
            return;
        }
        StoreMteGroupedRows<T>(yGm, 0U, outerSize, pendingGroup.groupAxisOffset,
                               pendingGroup.groupElems, pendingGroup.groupElems);
        pendingGroup.valid = false;
    }

    template <typename T>
    __aicore__ inline uint32_t CountCoalescedUnits(uint32_t groupLimit) const
    {
        constexpr uint32_t kAlignElems = 32U / sizeof(T);
        uint32_t unitCount = 0U;
        uint32_t atomicInputBegin = 0U;
        uint32_t atomicAxisBegin = 0U;
        uint32_t axisEnd = 0U;
        uint32_t pendingElems = 0U;

        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            axisEnd += AxisLen(inputIdx);
            if (axisEnd == atomicAxisBegin || (axisEnd % kAlignElems) != 0U) {
                continue;
            }

            const uint32_t atomicElems = axisEnd - atomicAxisBegin;
            const bool canPackAtomic =
                IsAtomicGroupAligned<T>(atomicInputBegin, inputIdx + 1U) &&
                atomicElems <= groupLimit;
            if (canPackAtomic) {
                if (pendingElems != 0U && pendingElems + atomicElems > groupLimit) {
                    ++unitCount;
                    pendingElems = 0U;
                }
                pendingElems += atomicElems;
            } else {
                if (pendingElems != 0U) {
                    ++unitCount;
                    pendingElems = 0U;
                }
                ++unitCount;
            }

            atomicInputBegin = inputIdx + 1U;
            atomicAxisBegin = axisEnd;
        }
        if (pendingElems != 0U) {
            ++unitCount;
        }
        return unitCount;
    }

    template <typename T>
    __aicore__ inline bool IsAtomicGroupAligned(uint32_t groupStart,
                                                      uint32_t groupEnd) const
    {
        constexpr uint32_t kAlignElems = 32U / sizeof(T);
        if (groupStart >= groupEnd) {
            return false;
        }
        for (uint32_t member = groupStart; member < groupEnd; ++member) {
            const uint32_t axisLen = AxisLen(member);
            if (axisLen == 0U || (AxisOffset(member) % kAlignElems) != 0U ||
                (axisLen % kAlignElems) != 0U) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    __aicore__ inline void CopyGroupOrRows(uint32_t groupStart, uint32_t groupEnd,
                                                GlobalTensor<T>& yGm,
                                                uint32_t groupAxisOffset,
                                                uint32_t groupElems,
                                                uint32_t localElems)
    {
        if (CanCopyAlignedGroupInOneOutput<T>(groupStart, groupEnd, groupAxisOffset,
                                              groupElems, localElems)) {
            CopyMteGroupedRows<T>(groupStart, groupEnd, yGm, 0U, outerSize,
                                  groupAxisOffset, groupElems, groupElems);
            return;
        }

        uint32_t memberAxisOffset = groupAxisOffset;
        for (uint32_t member = groupStart; member < groupEnd; ++member) {
            const uint32_t axisLen = AxisLen(member);
            if (axisLen == 0U) {
                continue;
            }
            GlobalTensor<T> xGm;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(member)),
                                static_cast<uint64_t>(outerSize) * axisLen);
            xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            CopyRowsSegment<T>(xGm, yGm, 0U, outerSize, memberAxisOffset,
                               axisLen, outputAxisSize);
            memberAxisOffset += axisLen;
        }
    }

    template <typename T>
    __aicore__ inline bool TryProcessAlignedInputGroups(uint32_t localElems,
                                                             GlobalTensor<T>& yGm)
    {
        constexpr uint32_t kAlignElems = 32U / sizeof(T);
        if (sizeof(T) != 4U || alignedGroupCount == 0U || localElems == 0U) {
            return false;
        }

        // Zero-width inputs have no physical segment to own. Keep them on the
        // generic row path instead of assigning their surrounding aligned group
        // to a single core.
        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            if (AxisLen(inputIdx) == 0U) {
                return false;
            }
        }

        const uint32_t blockNum = GetBlockNum();
        if (blockNum == 0U || alignedGroupCount < blockNum) {
            return false;
        }

        const uint32_t blockIdx = GetBlockIdx();
        uint32_t groupIdx = 0U;
        uint32_t groupInputBegin = 0U;
        uint32_t groupAxisBegin = 0U;
        uint32_t axisEnd = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            axisEnd += AxisLen(inputIdx);
            if (axisEnd == groupAxisBegin || (axisEnd % kAlignElems) != 0U) {
                continue;
            }

            if ((groupIdx % blockNum) == blockIdx) {
                const uint32_t groupElems = axisEnd - groupAxisBegin;
                if (CanCopyAlignedGroupInOneOutput<T>(groupInputBegin, inputIdx + 1U,
                                                      groupAxisBegin, groupElems, localElems)) {
                    CopyMteGroupedRows<T>(groupInputBegin, inputIdx + 1U, yGm, 0U,
                                          outerSize, groupAxisBegin, groupElems, groupElems);
                } else {
                    uint32_t memberAxisOffset = groupAxisBegin;
                    for (uint32_t member = groupInputBegin; member <= inputIdx; ++member) {
                        const uint32_t axisLen = AxisLen(member);
                        if (axisLen == 0U) {
                            continue;
                        }
                        GlobalTensor<T> xGm;
                        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(member)),
                                            static_cast<uint64_t>(outerSize) * axisLen);
                        xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                        CopyRowsSegment<T>(xGm, yGm, 0U, outerSize, memberAxisOffset,
                                           axisLen, outputAxisSize);
                        memberAxisOffset += axisLen;
                    }
                }
            }

            ++groupIdx;
            groupInputBegin = inputIdx + 1U;
            groupAxisBegin = axisEnd;
        }
        return true;
    }

    template <typename T>
    __aicore__ inline bool CanCopyAlignedGroupInOneOutput(uint32_t groupStart,
                                                           uint32_t groupEnd,
                                                           uint32_t groupAxisOffset,
                                                           uint32_t groupElems,
                                                           uint32_t localElems) const
    {
        constexpr uint32_t kAlignElems = 32U / sizeof(T);
        if (groupStart >= groupEnd || groupElems < 2U * kAlignElems ||
            (groupAxisOffset % kAlignElems) != 0U ||
            (groupElems % kAlignElems) != 0U ||
            static_cast<uint64_t>(groupElems) * outerSize > localElems) {
            return false;
        }

        uint32_t nonZeroMembers = 0U;
        for (uint32_t member = groupStart; member < groupEnd; ++member) {
            const uint32_t axisLen = AxisLen(member);
            if (axisLen == 0U) {
                continue;
            }
            if ((AxisOffset(member) % kAlignElems) != 0U ||
                (axisLen % kAlignElems) != 0U) {
                return false;
            }
            ++nonZeroMembers;
        }
        return nonZeroMembers >= 2U;
    }
