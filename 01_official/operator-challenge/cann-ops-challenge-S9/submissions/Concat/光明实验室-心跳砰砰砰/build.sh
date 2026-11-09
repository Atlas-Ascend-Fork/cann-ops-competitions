#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cann_root="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
build_dir="${BUILD_DIR:-${project_dir}/build_out}"
jobs="${BUILD_JOBS:-8}"
target="${1:-package}"
if [[ ! -d "${cann_root}" ]]; then
    printf 'CANN installation not found: %s\n' "${cann_root}" >&2
    exit 1
fi
if [[ -f "${cann_root}/set_env.sh" ]]; then
    set +u
    source "${cann_root}/set_env.sh"
    set -u
fi
cmake -S "${project_dir}" -B "${build_dir}" \
    -DASCEND_CANN_PACKAGE_PATH="${cann_root}" \
    -DASCEND_COMPUTE_UNIT=ascend910b \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target binary --parallel "${jobs}"
cmake --build "${build_dir}" --target "${target}" --parallel "${jobs}"
