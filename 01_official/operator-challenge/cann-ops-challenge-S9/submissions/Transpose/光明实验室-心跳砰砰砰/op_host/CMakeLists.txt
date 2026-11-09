set(ops_srcs "${CMAKE_CURRENT_SOURCE_DIR}/transpose.cpp")
npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB aclnn_src "${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp")
file(GLOB proto_src "${ASCEND_AUTOGEN_PATH}/op_proto.cc")
file(GLOB fallback_src "${ASCEND_AUTOGEN_PATH}/fallback_*.cpp")
set_source_files_properties(${aclnn_src} ${proto_src} ${fallback_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_opapi ACLNN ${aclnn_src})
npu_op_library(cust_op_proto GRAPH ${ops_srcs} ${proto_src})
npu_op_library(cust_optiling TILING ${ops_srcs} ${fallback_src})

foreach(target cust_opapi cust_op_proto cust_optiling)
    target_compile_options(${target} PRIVATE -fvisibility=hidden)
endforeach()

npu_op_package_add(${package_name}
    LIBRARY cust_opapi cust_op_proto cust_optiling
)
