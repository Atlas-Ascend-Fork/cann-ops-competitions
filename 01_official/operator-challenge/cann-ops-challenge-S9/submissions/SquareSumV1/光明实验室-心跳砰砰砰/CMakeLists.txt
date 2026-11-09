cmake_minimum_required(VERSION 3.19)
project(opp LANGUAGES C CXX)

set(ASCEND_COMPUTE_UNIT ascend910b CACHE STRING "")
set(vendor_name customize CACHE STRING "")
if(NOT ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "" FORCE)
    else()
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "" FORCE)
    endif()
endif()

find_package(ASC REQUIRED HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")

npu_op_package(${vendor_name} TYPE RUN CONFIG
    ENABLE_SOURCE_PACKAGE TRUE
    ENABLE_BINARY_PACKAGE TRUE
    INSTALL_PATH "${CMAKE_BINARY_DIR}")

add_subdirectory(op_host)
add_subdirectory(op_kernel)
