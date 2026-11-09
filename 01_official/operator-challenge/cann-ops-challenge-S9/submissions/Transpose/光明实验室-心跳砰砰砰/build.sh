#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${1:-package}"
if (($#)); then
    shift
fi
cann_path="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-}}}"
if [[ -z "$cann_path" ]]; then
    printf '%s\n' 'Source the CANN 8.5.0 set_env.sh before building.' >&2
    exit 1
fi

cmake -S "$project_dir" --preset default -DASCEND_CANN_PACKAGE_PATH="$cann_path" "$@"
cmake --build "$project_dir/build_out" --target binary --parallel "${JOBS:-1}"
if [[ "$target" != binary ]]; then
    cmake --build "$project_dir/build_out" --target "$target" --parallel "${JOBS:-1}"
fi
