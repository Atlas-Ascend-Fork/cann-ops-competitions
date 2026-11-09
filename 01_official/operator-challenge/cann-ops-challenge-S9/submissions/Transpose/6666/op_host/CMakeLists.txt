aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${package_name}
    COMPILE_OPTIONS -g
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_src}
)

npu_op_library(cust_optiling TILING
    ${ops_srcs}
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_opapi
        cust_optiling
)
