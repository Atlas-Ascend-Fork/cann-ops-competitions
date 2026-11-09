file(GLOB host_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
npu_op_code_gen(SRC ${host_sources} PACKAGE ${vendor_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    OPTIONS OPS_PRODUCT_NAME ${ASCEND_COMPUTE_UNIT})
file(GLOB api_sources "${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp")
set_source_files_properties(${api_sources} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN ${api_sources})
npu_op_library(cust_optiling TILING ${host_sources})
npu_op_library(cust_opproto GRAPH ${host_sources})
target_include_directories(cust_optiling PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel")
target_link_libraries(cust_optiling PRIVATE ${CMAKE_DL_LIBS})
