set(host_source "${CMAKE_CURRENT_SOURCE_DIR}/square_sum_v1.cpp")
npu_op_code_gen(SRC ${host_source}
    PACKAGE ${vendor_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH})

file(GLOB api_sources "${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp")
file(GLOB proto_sources "${ASCEND_AUTOGEN_PATH}/op_proto.cc"
    "${ASCEND_AUTOGEN_PATH}/group_proto/*.cc")
file(GLOB fallback_sources "${ASCEND_AUTOGEN_PATH}/fallback_*.cpp")
set_source_files_properties(${api_sources} ${proto_sources} ${fallback_sources}
    PROPERTIES GENERATED TRUE)

npu_op_library(cust_opapi ACLNN ${api_sources})
npu_op_library(cust_op_proto GRAPH ${host_source} ${proto_sources})
npu_op_library(cust_optiling TILING ${host_source} ${fallback_sources})
foreach(target cust_opapi cust_op_proto cust_optiling)
    target_compile_options(${target} PRIVATE -fvisibility=hidden)
endforeach()

npu_op_package_add(${vendor_name}
    LIBRARY cust_opapi cust_op_proto cust_optiling)
