set(host_sources "${CMAKE_CURRENT_SOURCE_DIR}/concat.cpp")
npu_op_code_gen(SRC ${host_sources}
    PACKAGE ${vendor_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH})

npu_op_library(cust_optiling TILING ${host_sources})
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)

file(GLOB api_sources "${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp")
set_source_files_properties(${api_sources} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN ${api_sources})

file(GLOB proto_sources "${ASCEND_AUTOGEN_PATH}/op_proto.cc")
set_source_files_properties(${proto_sources} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH ${host_sources} ${proto_sources})
