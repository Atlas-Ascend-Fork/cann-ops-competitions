cmake_minimum_required(VERSION 3.19.0)
project(opp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()
if(NOT ASCEND_COMPUTE_UNIT)
    set(ASCEND_COMPUTE_UNIT "ascend910b")
endif()
if(NOT vendor_name)
    set(vendor_name "customize")
endif()

find_package(ASC REQUIRED HINTS
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake"
)

set(package_name ${vendor_name})
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        ENABLE_SOURCE_PACKAGE True
        ENABLE_BINARY_PACKAGE True
        INSTALL_PATH ${CMAKE_BINARY_DIR}/
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
