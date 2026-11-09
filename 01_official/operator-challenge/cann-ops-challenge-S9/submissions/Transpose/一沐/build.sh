#!/usr/bin/env bash
set -euo pipefail

# Native CANN 8.5 msopgen build. Never delete files in the caller's directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"
cann_path="${ASCEND_CANN_PACKAGE_PATH:-${BASE_LIBS_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-/usr/local/Ascend/cann-8.5.0}}}}"
if [[ ! -d "$cann_path/include" || ! -d "$cann_path/lib64" ]]; then
    echo "CANN SDK not found: $cann_path" >&2
    echo "Source CANN set_env.sh or set ASCEND_CANN_PACKAGE_PATH before building." >&2
    exit 1
fi
export ASCEND_HOME_PATH="$cann_path"
target="${1:-package}"
jobs="${BUILD_JOBS:-8}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "BUILD_JOBS must be a positive integer." >&2
    exit 1
fi
# The msopgen kernel generator requires an empty build directory. Preserve
# previous outputs instead of deleting them, including after a failed build.
if [[ -L "$script_dir/build_out" ]]; then
    echo "Refusing to replace a symlink at $script_dir/build_out" >&2
    exit 1
fi
if [[ -e "$script_dir/build_out" ]]; then
    backup_dir="$(mktemp -d "$script_dir/build_out.previous.XXXXXX")"
    mv -- "$script_dir/build_out" "$backup_dir/build_out"
    echo "Previous build preserved at $backup_dir/build_out"
fi
cmake --preset=default -S "$script_dir" -B "$script_dir/build_out" \
    "-DASCEND_CANN_PACKAGE_PATH=$cann_path"
if [[ "$target" == package || "$target" == install ]]; then
    cmake --build "$script_dir/build_out" --target binary --parallel "$jobs"
fi
cmake --build "$script_dir/build_out" --target "$target" --parallel "$jobs"
