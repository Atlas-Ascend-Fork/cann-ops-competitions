cmake_minimum_required(VERSION 3.19)
project(concat_op LANGUAGES C CXX)

if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "")
set(vendor_name "customize" CACHE STRING "")
find_package(ASC REQUIRED HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")

npu_op_package(${vendor_name} TYPE RUN)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
npu_op_package_add(${vendor_name}
    LIBRARY cust_optiling cust_opapi cust_op_proto ascendc_kernels)
