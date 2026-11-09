#!/usr/bin/env bash
# 从工程目录构建 Concat；运行前先加载 CANN 8.5.0 的 set_env.sh。
set -euo pipefail

usage() {
    printf 'Usage: bash build.sh [target]\n'
    printf 'Default target: package (install for shared-library presets).\n'
    printf 'Set JOBS to override parallel build jobs.\n'
}

if [ "$#" -gt 1 ]; then usage >&2; exit 2; fi
if [ "$#" -eq 1 ] && { [ "$1" = '--help' ] || [ "$1" = '-h' ]; }; then usage; exit 0; fi
target=${1:-package}
if [[ ! "$target" =~ ^[A-Za-z0-9_][A-Za-z0-9_.+-]*$ ]]; then
    printf 'Invalid build target: %s\n' "$target" >&2
    usage >&2
    exit 2
fi

script_path=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_path"

# 显式配置优先；仅在没有配置时搜索常见安装位置。
sdk_path=${ASCEND_CANN_PACKAGE_PATH:-${BASE_LIBS_PATH:-${ASCEND_HOME_PATH:-${ASCEND_AICPU_PATH:-}}}}
if [ -z "$sdk_path" ]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-8.5.0 /usr/local/Ascend/latest; do
        if [ -d "$candidate" ]; then sdk_path=$candidate; break; fi
    done
fi
if [ -z "$sdk_path" ] || [ ! -d "$sdk_path/include" ] || [ ! -x "$sdk_path/toolkit/tools/opbuild/op_build" ]; then
    printf 'CANN toolkit not found: %s\n' "${sdk_path:-not configured}" >&2
    printf 'Source the CANN 8.5.0 set_env.sh or set ASCEND_CANN_PACKAGE_PATH to the toolkit root.\n' >&2
    exit 1
fi
sdk_path=$(cd -- "$sdk_path" && pwd)
export ASCEND_HOME_PATH="$sdk_path"
python_bin=${ASCEND_PYTHON_EXECUTABLE:-python3}
command -v "$python_bin" >/dev/null || { printf 'Python executable not found: %s\n' "$python_bin" >&2; exit 1; }
command -v cmake >/dev/null || { printf 'cmake is required.\n' >&2; exit 1; }

if [ -n "${JOBS:-}" ]; then
    jobs=$JOBS
elif command -v nproc >/dev/null; then
    jobs=$(nproc)
elif command -v getconf >/dev/null; then
    jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null) || jobs=1
else
    jobs=1
fi
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'JOBS must be a positive integer: %s\n' "$jobs" >&2
    exit 2
fi

# 直接展开缓存项，兼容没有 --preset 的 CMake 3.16，并保留路径中的空格。
preset_text=$("$python_bin" - "$script_path/CMakePresets.json" "$script_path" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    document = json.load(stream)
presets = [item for item in document["configurePresets"] if item["name"] == "default"]
if len(presets) != 1:
    raise SystemExit("Exactly one default configure preset is required")
for key, entry in presets[0]["cacheVariables"].items():
    value = entry["value"] if isinstance(entry, dict) else entry
    value = str(value).replace("${sourceDir}", sys.argv[2])
    if "\n" in key or "\n" in value or "\r" in key or "\r" in value:
        raise SystemExit("Preset values must not contain newlines")
    print("-D{}={}".format(key, value))
PY
)
preset_opts=()
while IFS= read -r option; do
    if [ -n "$option" ]; then preset_opts+=("$option"); fi
done <<< "$preset_text"

is_on() {
    case "$1" in 1|ON|on|TRUE|true|True|YES|yes) return 0 ;; *) return 1 ;; esac
}
cross_compile=False
binary_package=False
shared_library=False
for option in "${preset_opts[@]}"; do
    case "$option" in
        -DENABLE_CROSS_COMPILE=*) cross_compile=${option#*=} ;;
        -DENABLE_BINARY_PACKAGE=*) binary_package=${option#*=} ;;
        -DASCEND_PACK_SHARED_LIBRARY=*) shared_library=${option#*=} ;;
    esac
done
if [ "$#" -eq 0 ] && is_on "$shared_library"; then target=install; fi

build_dir="$script_path/build_out"
configure() {
    cmake -S "$script_path" -B "$build_dir" "${preset_opts[@]}" \
        "-DASCEND_CANN_PACKAGE_PATH=$sdk_path" \
        "-DASCEND_PYTHON_EXECUTABLE=$python_bin" "$@"
}

printf 'Using CANN toolkit: %s\n' "$sdk_path"
printf 'Build target: %s; jobs: %s\n' "$target" "$jobs"
rm -rf -- "$build_dir"
mkdir -p "$build_dir"

if is_on "$cross_compile" && is_on "$binary_package"; then
    # 先使用本机编译器生成可执行的 tiling 库，再构建目标架构安装包。
    configure -DENABLE_CROSS_COMPILE=False
    cmake --build "$build_dir" --target cust_optiling -j "$jobs"
    native_library=$("$python_bin" - "$build_dir" <<'PY'
from pathlib import Path
import sys
paths = list(Path(sys.argv[1]).rglob("libcust_opmaster_rt2.0.so"))
if len(paths) != 1:
    raise SystemExit("Expected one native tiling library, found {}".format(len(paths)))
print(paths[0].resolve())
PY
)
    native_backup=$(mktemp -d "${TMPDIR:-/tmp}/concat-native-tiling.XXXXXX")
    trap 'rm -rf -- "$native_backup"' EXIT
    cp -- "$native_library" "$native_backup/"
    rm -rf -- "$build_dir"
    mkdir -p "$build_dir/host_native_tiling"
    cp -- "$native_backup/libcust_opmaster_rt2.0.so" "$build_dir/host_native_tiling/"
    configure "-DHOST_NATIVE_TILING_LIB=$build_dir/host_native_tiling/libcust_opmaster_rt2.0.so"
else
    configure
fi

cmake --build "$build_dir" --target binary -j "$jobs"
if [ "$target" != binary ]; then
    cmake --build "$build_dir" --target "$target" -j "$jobs"
fi
