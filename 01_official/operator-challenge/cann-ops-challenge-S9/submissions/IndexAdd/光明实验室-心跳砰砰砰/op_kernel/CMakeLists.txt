npu_op_kernel_sources(ascendc_kernels OP_TYPE IndexAdd KERNEL_FILE index_add.cpp)
npu_op_kernel_library(ascendc_kernels
    SRC_BASE "${CMAKE_CURRENT_SOURCE_DIR}/"
    TILING_LIBRARY cust_optiling
)
npu_op_package_add(${vendor_name} LIBRARY ascendc_kernels)
