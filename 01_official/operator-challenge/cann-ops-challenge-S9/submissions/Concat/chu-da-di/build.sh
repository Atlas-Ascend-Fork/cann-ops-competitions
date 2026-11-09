#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cann_root="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}}"
build_dir="${BUILD_DIR:-${script_dir}/build_out}"
jobs="${BUILD_JOBS:-4}"
target="${1:-package}"
if [[ ! -f "${cann_root}/set_env.sh" ]]; then
    printf 'CANN environment not found at %s\n' "${cann_root}" >&2
    printf 'Set ASCEND_CANN_PACKAGE_PATH to the CANN 8.5.0 installation.\n' >&2
    exit 1
fi
# CANN environment scripts can read unset variables.
set +u
source "${cann_root}/set_env.sh"
set -u
export ASCEND_CANN_PACKAGE_PATH="${cann_root}"
cmake -S "${script_dir}" -B "${build_dir}" \
    -DASCEND_CANN_PACKAGE_PATH="${cann_root}" \
    -DCMAKE_PREFIX_PATH="${cann_root}/lib64/cmake;${cann_root}/compiler/tikcpp/ascendc_kernel_cmake" \
    -DASCEND_COMPUTE_UNIT="${ASCEND_COMPUTE_UNIT:-ascend910b}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_SKIP_RPATH=TRUE
cmake --build "${build_dir}" --target binary --parallel "${jobs}"
cmake --build "${build_dir}" --target "${target}" --parallel "${jobs}"

if [[ -f "${script_dir}/tests/meta_shape_test.cpp" ]]; then
    cmake --build "${build_dir}" --target meta_shape_test --parallel "${jobs}"
    "${build_dir}/meta_shape_test"
fi
