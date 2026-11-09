cmake_minimum_required(VERSION 3.19)
project(index_add_operator LANGUAGES C CXX)

if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
set(ASCEND_COMPUTE_UNIT ascend910b CACHE STRING "Ascend compute unit")
set(vendor_name customize CACHE STRING "Custom operator vendor")
find_package(ASC REQUIRED HINTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")

npu_op_package(${vendor_name}
    TYPE RUN
    CONFIG INSTALL_PATH "${CMAKE_BINARY_DIR}"
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
