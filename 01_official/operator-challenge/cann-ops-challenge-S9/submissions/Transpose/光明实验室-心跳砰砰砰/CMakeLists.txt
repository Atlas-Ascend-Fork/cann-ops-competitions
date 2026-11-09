cmake_minimum_required(VERSION 3.19)
project(transpose_operator LANGUAGES C CXX)

if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
find_package(ASC REQUIRED HINTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")

set(ASCEND_COMPUTE_UNIT ascend910b)
set(package_name customize)
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        ENABLE_SOURCE_PACKAGE TRUE
        ENABLE_BINARY_PACKAGE TRUE
        INSTALL_PATH "${CMAKE_BINARY_DIR}"
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
