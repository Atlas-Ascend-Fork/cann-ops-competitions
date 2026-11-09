if("$ENV{ASCEND_OPERATOR_SIMULATOR_BUILD}" STREQUAL "1")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -DASCENDC_TRACE_ON)
elseif("$ENV{ASCEND_OPERATOR_PROFILE_BUILD}" STREQUAL "1")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g)
endif()

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Transpose
    KERNEL_DIR ./
    COMPUTE_UNIT Ascend910B1
    KERNEL_FILE transpose.cpp
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_SOURCE_DIR}/op_kernel/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
