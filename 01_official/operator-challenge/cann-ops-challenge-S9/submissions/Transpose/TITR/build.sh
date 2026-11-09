#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

cann_path=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-}}}
if [[ -z "$cann_path" || ! -d "$cann_path" ]]; then
    printf '%s\n' 'Source the CANN 8.5.0 set_env.sh first, or set ASCEND_CANN_PACKAGE_PATH.' >&2
    exit 1
fi
export ASCEND_HOME_PATH="$cann_path"

# Keep local packaging within four build workers by default.
build_jobs=${S9_BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-4}}
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'S9_BUILD_JOBS must be a positive integer.' >&2
    exit 1
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs"
export MAKEFLAGS="-j$build_jobs"

target=${1:-package}
if (( $# > 0 )); then shift; fi
cmake --preset default -DASCEND_CANN_PACKAGE_PATH="$cann_path" "$@"
cmake --build build_out --target binary --parallel "$build_jobs"
cmake --build build_out --target "$target" --parallel "$build_jobs"
