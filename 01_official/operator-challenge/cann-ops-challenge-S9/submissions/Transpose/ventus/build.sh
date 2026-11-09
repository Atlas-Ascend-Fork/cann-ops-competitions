#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
cann_path="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-}}}"
if [[ -z "$cann_path" ]]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann; do
        if [[ -d "$candidate/compiler/tikcpp/ascendc_kernel_cmake" ]]; then
            cann_path="$candidate"
            break
        fi
    done
fi
if [[ -z "$cann_path" || ! -d "$cann_path/compiler/tikcpp/ascendc_kernel_cmake" ]]; then
    printf '%s\n' 'CANN ASC CMake modules not found. Source CANN 8.5.0 set_env.sh and set ASCEND_CANN_PACKAGE_PATH.' >&2
    exit 1
fi
command -v cmake >/dev/null
cmake -S "$script_dir" -B "$script_dir/build_out" --preset=default \
    -DASCEND_CANN_PACKAGE_PATH="$cann_path" "$@"
jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 4)}"
cmake --build "$script_dir/build_out" --target binary --parallel "$jobs"
cmake --build "$script_dir/build_out" --target package --parallel "$jobs"
