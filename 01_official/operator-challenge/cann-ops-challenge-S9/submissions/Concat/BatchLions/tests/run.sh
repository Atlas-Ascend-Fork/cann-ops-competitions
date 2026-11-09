#!/usr/bin/env bash
# Copyright (c) 2026 ascend_simula contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/s9-concat-tests.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT
cxx=${CXX:-c++}
for source in twin_concat test_concat_layout test_single_band_budget; do
    "$cxx" -std=c++17 -O2 -Wall -Wextra "$script_dir/$source.cpp" -o "$test_dir/$source"
    "$test_dir/$source"
done
