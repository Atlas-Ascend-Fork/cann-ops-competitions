#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' 'Usage: bash build.sh [package|binary|all]' \
        'Source the CANN 8.5.0 set_env.sh first (Ascend 910B, native Linux).' \
        'ASCEND_HOME_PATH selects the toolkit; JOBS controls build parallelism.' \
        'The local build_out directory is recreated for each build.'
    exit 0
fi

target="${1:-package}"
case "$target" in
    package|binary|all) ;;
    *) printf 'Unsupported target: %s\n' "$target" >&2; exit 2 ;;
esac

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_AICPU_PATH:-${BASE_LIBS_PATH:-}}}}"
if [[ -z "$ASCEND_HOME_PATH" || ! -x "$ASCEND_HOME_PATH/toolkit/tools/opbuild/op_build" ]]; then
    printf '%s\n' 'Source the CANN 8.5.0 set_env.sh and set ASCEND_HOME_PATH to its toolkit root.' >&2
    exit 1
fi

for tool in cmake python3 make g++; do
    command -v "$tool" >/dev/null || { printf 'Missing tool: %s\n' "$tool" >&2; exit 1; }
done
jobs="${JOBS:-$(nproc)}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' 'JOBS must be a positive integer.' >&2; exit 2; }

# The kernel packaging helper expects a fresh, submission-local build tree.
build_dir="$script_dir/build_out"
if [[ -L "$build_dir" ]]; then
    printf '%s\n' 'Refusing to recreate a symlinked build_out directory.' >&2
    exit 1
fi
cmake -E rm -rf "$build_dir"
cmake --preset default -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build "$build_dir" --target binary --parallel "$jobs"
if [[ "$target" != binary ]]; then
    cmake --build "$build_dir" --target "$target" --parallel "$jobs"
fi
