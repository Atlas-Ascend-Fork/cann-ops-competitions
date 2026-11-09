#!/usr/bin/env bash
set -euo pipefail

s9_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash build.sh [package|binary]

Build a native Ascend 910B operator with CANN 8.5.0.
Source the CANN set_env.sh file before running this script.

Environment:
  ASCEND_CANN_PACKAGE_PATH  Optional explicit CANN toolkit directory.
  ASCEND_HOME_PATH          Toolkit directory supplied by set_env.sh.
  S9_BUILD_JOBS             Parallel build jobs (default: 8).

Each build recreates build_out/. The default target is package.
EOF
}

case "${1:-package}" in
    -h|--help)
        usage
        exit 0
        ;;
    package|binary)
        s9_target="${1:-package}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

s9_toolkit="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-${BASE_LIBS_PATH:-}}}}"
if [[ -z "$s9_toolkit" || ! -d "$s9_toolkit/include" || ! -d "$s9_toolkit/compiler" ]]; then
    printf '%s\n' 'CANN toolkit not found. Source the CANN 8.5.0 set_env.sh file first.' >&2
    exit 1
fi

for s9_tool in cmake python3 make; do
    if ! command -v "$s9_tool" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$s9_tool" >&2
        exit 1
    fi
done

s9_jobs="${S9_BUILD_JOBS:-8}"
if [[ ! "$s9_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'S9_BUILD_JOBS must be a positive integer.' >&2
    exit 2
fi

s9_build_dir="$s9_project_dir/build_out"
cd -- "$s9_project_dir"

if [[ -L "$s9_build_dir" ]]; then
    printf '%s\n' 'build_out must be a regular project directory, not a symbolic link.' >&2
    exit 1
fi
rm -rf -- "$s9_build_dir"

cmake -S "$s9_project_dir" -B "$s9_build_dir" --preset=default \
    -DASCEND_CANN_PACKAGE_PATH="$s9_toolkit" \
    -DENABLE_CROSS_COMPILE=OFF \
    -DASCEND_PACK_SHARED_LIBRARY=OFF

cmake --build "$s9_build_dir" --target binary --parallel "$s9_jobs"

if [[ "$s9_target" == package ]]; then
    cmake --build "$s9_build_dir" --target package --parallel "$s9_jobs"
    shopt -s nullglob
    s9_packages=("$s9_build_dir"/custom_opp_*.run)
    if (( ${#s9_packages[@]} == 0 )); then
        printf 'Build finished without an operator package in %s\n' "$s9_build_dir" >&2
        exit 1
    fi
    printf 'Operator package: %s\n' "${s9_packages[@]}"
fi
