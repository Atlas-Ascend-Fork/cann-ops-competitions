#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "用法: bash build.sh [-DCMAKE_VARIABLE=value ...]"
    echo "环境变量: ASCEND_CANN_PACKAGE_PATH 或 ASCEND_HOME_PATH 指定 CANN 8.5.0 路径，JOBS 指定并行数。"
    echo "生成 build_out 下的算子安装包，不自动安装。"
    exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cann_path="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-}}"
if [[ -z "$cann_path" ]]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann; do
        if [[ -d "$candidate" ]]; then
            cann_path="$candidate"
            break
        fi
    done
fi
if [[ -z "$cann_path" || ! -d "$cann_path" ]]; then
    echo "找不到 CANN 开发套件，请先加载其环境并设置 ASCEND_CANN_PACKAGE_PATH。" >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "请安装 CMake 3.19 或更新版本。" >&2
    exit 1
fi

jobs="${JOBS:-8}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS 必须为正整数。" >&2
    exit 1
fi

cmake -S "$script_dir" -B "$script_dir/build_out" --preset=default \
    -DASCEND_CANN_PACKAGE_PATH="$cann_path" "$@"
cmake --build "$script_dir/build_out" --target binary --parallel "$jobs"
cmake --build "$script_dir/build_out" --target package --parallel "$jobs"
