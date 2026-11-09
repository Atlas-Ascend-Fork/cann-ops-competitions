set(ops_srcs "${CMAKE_CURRENT_SOURCE_DIR}/index_add.cpp")
npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${vendor_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB aclnn_src "${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp")
file(GLOB proto_src "${ASCEND_AUTOGEN_PATH}/op_proto.cc")
set_source_files_properties(${aclnn_src} ${proto_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_opapi ACLNN ${aclnn_src})
npu_op_library(cust_op_proto GRAPH ${ops_srcs} ${proto_src})
npu_op_library(cust_optiling TILING ${ops_srcs})

foreach(target cust_opapi cust_op_proto cust_optiling)
    target_compile_options(${target} PRIVATE -fvisibility=hidden)
endforeach()

npu_op_package_add(${vendor_name}
    LIBRARY cust_opapi cust_op_proto cust_optiling
)
