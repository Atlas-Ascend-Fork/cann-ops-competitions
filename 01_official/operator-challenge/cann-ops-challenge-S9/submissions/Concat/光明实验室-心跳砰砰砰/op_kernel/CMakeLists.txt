npu_op_kernel_sources(ascendc_kernels
    OP_TYPE Concat
    KERNEL_DIR ./
    KERNEL_FILE concat.cpp)
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling)
npu_op_kernel_options(ascendc_kernels ALL OPTIONS
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../op_host")
