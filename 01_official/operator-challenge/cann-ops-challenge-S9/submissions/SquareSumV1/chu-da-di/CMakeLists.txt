cmake_minimum_required(VERSION 3.16)
project(s9_square_sum_v1 LANGUAGES C CXX)

set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "Ascend compute unit")
set(vendor_name "customize" CACHE STRING "Custom operator vendor")
find_package(ASC REQUIRED)

npu_op_package(${vendor_name} TYPE RUN CONFIG
    ENABLE_SOURCE_PACKAGE True
    ENABLE_BINARY_PACKAGE True)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
npu_op_package_add(${vendor_name}
    LIBRARY cust_opapi cust_optiling cust_opproto ascendc_kernels)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/meta_shape_test.cpp")
    add_executable(meta_shape_test tests/meta_shape_test.cpp)
    target_compile_features(meta_shape_test PRIVATE cxx_std_17)
endif()
