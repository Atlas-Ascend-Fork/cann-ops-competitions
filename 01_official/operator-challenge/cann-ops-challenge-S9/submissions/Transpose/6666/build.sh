#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${ASCEND_HOME_PATH:?Source the CANN 8.5.0 set_env.sh before building}"
build_dir="${project_dir}/build_out"
python_executable=${ASCEND_PYTHON_EXECUTABLE:-$(command -v python3)}
jobs=${BUILD_JOBS:-$(nproc)}

# Use explicit cache settings so CMake 3.16 can build without preset support.
cmake -S "${project_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_PYTHON_EXECUTABLE="${python_executable}" \
    -DASCEND_COMPUTE_UNIT="${ASCEND_COMPUTE_UNIT:-ascend910b}" \
    -Dvendor_name=customize \
    -DENABLE_SOURCE_PACKAGE=ON \
    -DENABLE_BINARY_PACKAGE=ON \
    -DCMAKE_INSTALL_PREFIX="${build_dir}" \
    "$@"
cmake --build "${build_dir}" --target binary -j"${jobs}"
cmake --build "${build_dir}" --target package -j"${jobs}"
