#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

# Use the toolkit selected by the caller or the CANN environment setup script.
toolkit_path="${ASCEND_CANN_PACKAGE_PATH:-${BASE_LIBS_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-}}}}"
if [[ -z "$toolkit_path" ]]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-8.5.0; do
        if [[ -d "$candidate" ]]; then
            toolkit_path="$candidate"
            break
        fi
    done
fi
if [[ -z "$toolkit_path" || ! -d "$toolkit_path/include" || ! -x "$toolkit_path/toolkit/tools/opbuild/op_build" ]]; then
    echo "CANN toolkit not found. Source its set_env.sh or set ASCEND_CANN_PACKAGE_PATH." >&2
    exit 1
fi
export ASCEND_HOME_PATH="$toolkit_path"

target="${1:-package}"
if [[ $# -gt 0 ]]; then
    shift
fi
jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"

cmake --preset default -DASCEND_CANN_PACKAGE_PATH="$toolkit_path" "$@"
cmake --build "$project_dir/build_out" --target binary --parallel "$jobs"
cmake --build "$project_dir/build_out" --target "$target" --parallel "$jobs"
