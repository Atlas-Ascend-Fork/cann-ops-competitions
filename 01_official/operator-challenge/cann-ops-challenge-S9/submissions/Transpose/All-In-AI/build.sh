#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash build.sh [package|binary] [extra cmake -D options...]"
    echo "Source the CANN 8.5.0 set_env.sh first. BUILD_JOBS defaults to 8."
    exit 0
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "Build requires Linux and CANN 8.5.0; this host is not supported." >&2
    exit 1
fi
script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_path"
# BASE_LIBS_PATH is used by competition CI; the other names support SDK shells.
cann_path="${BASE_LIBS_PATH:-${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-}}}}"
if [[ -z "$cann_path" || ! -d "$cann_path/include" || ! -d "$cann_path/lib64" ]]; then
    echo "Set ASCEND_HOME_PATH (or BASE_LIBS_PATH) to the initialized CANN 8.5.0 toolkit." >&2
    exit 1
fi
export ASCEND_HOME_PATH="$cann_path"
export ASCEND_CANN_PACKAGE_PATH="$cann_path"
target="${1:-package}"
if [[ $# -gt 0 ]]; then shift; fi
case "$target" in package|binary) ;; *) echo "Unsupported target: $target" >&2; exit 2;; esac
jobs="${BUILD_JOBS:-8}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "BUILD_JOBS must be a positive integer." >&2
    exit 2
fi
cmake --preset=default -DASCEND_CANN_PACKAGE_PATH="$cann_path" "$@"
# CANN's package target alone does not build the separate binary target.
cmake --build build_out --target binary --parallel "$jobs"
if [[ "$target" == package ]]; then
    cmake --build build_out --target package --parallel "$jobs"
fi
