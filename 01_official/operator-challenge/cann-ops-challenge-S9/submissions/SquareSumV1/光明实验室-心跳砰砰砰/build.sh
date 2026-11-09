#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cann_root="${ASCEND_CANN_PACKAGE_PATH:-${BASE_LIBS_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}}"
build_jobs="${BUILD_JOBS:-8}"

cd "$script_dir"
cmake --preset default -DASCEND_CANN_PACKAGE_PATH="$cann_root"
cmake --build build_out --target binary --parallel "$build_jobs"
cmake --build build_out --target "${1:-package}" --parallel "$build_jobs"
