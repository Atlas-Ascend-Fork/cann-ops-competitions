cmake_minimum_required(VERSION 3.19.0)
project(opp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT ASCEND_CANN_PACKAGE_PATH AND DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "Set ASCEND_CANN_PACKAGE_PATH or source the CANN set_env.sh first")
endif()
find_package(ASC REQUIRED HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
if(NOT vendor_name)
    set(vendor_name "customize")
endif()
set(package_name ${vendor_name})
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        ENABLE_BINARY_PACKAGE True
        INSTALL_PATH ${CMAKE_BINARY_DIR}/
)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
