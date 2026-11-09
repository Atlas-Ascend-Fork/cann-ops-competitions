#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_path="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-${BASE_LIBS_PATH:-}}}}"
if [[ -z "$cann_path" ]]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-8.5.0 /usr/local/Ascend/latest; do
        if [[ -d "$candidate" ]]; then
            cann_path="$candidate"
            break
        fi
    done
fi
if [[ -z "$cann_path" || ! -d "$cann_path/include" ]]; then
    printf '%s\n' 'Set ASCEND_HOME_PATH to the CANN 8.5.0 toolkit directory.' >&2
    exit 1
fi
export ASCEND_HOME_PATH="$cann_path"
target="${1:-package}"
if (( $# > 0 )); then
    shift
fi
cmake -S "$project_dir" --preset default -DASCEND_CANN_PACKAGE_PATH="$cann_path" "$@"
cmake --build "$project_dir/build_out" --target binary --parallel "${BUILD_JOBS:-1}"
if [[ "$target" != binary ]]; then
    cmake --build "$project_dir/build_out" --target "$target" --parallel "${BUILD_JOBS:-1}"
fi
