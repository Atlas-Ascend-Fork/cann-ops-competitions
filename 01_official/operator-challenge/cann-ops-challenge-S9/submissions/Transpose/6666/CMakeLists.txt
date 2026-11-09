cmake_minimum_required(VERSION 3.16.0)
project(opp)

find_package(ASC REQUIRED
    HINTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

set(package_name ${vendor_name})
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        ENABLE_SOURCE_PACKAGE True
        ENABLE_BINARY_PACKAGE True
        INSTALL_PATH ${CMAKE_BINARY_DIR}/)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
