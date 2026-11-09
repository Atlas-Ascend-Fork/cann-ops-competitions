if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

add_ops_compile_options(Transpose OPTIONS --tiling_key=0,1,2)
add_kernels_compile()
