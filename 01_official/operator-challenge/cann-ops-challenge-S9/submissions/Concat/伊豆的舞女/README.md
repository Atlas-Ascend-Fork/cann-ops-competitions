# Concat (Ascend S9)

## Team and build

Team: 伊豆的舞女. Contributor: KaranocaVe.

Build on Linux for Ascend 910B using a compatible CANN SDK and its `msopgen`
tool. The submitted `op_host/` and `op_kernel/` contain the operator sources
and CMake targets. Generate the SDK build framework outside this source tree,
then replace its operator directories with these files:

```bash
source /path/to/ascend-toolkit/set_env.sh
# Set SOC_VERSION to the installed chip model, for example Ascend910B1.
export SOC_VERSION=Ascend910B1
project_dir="$(mktemp -d)/custom_op"
msopgen gen -i op.json -f aclnn -c "ai_core-${SOC_VERSION}" -lan cpp -out "$project_dir"
rm -rf "$project_dir/op_host" "$project_dir/op_kernel"
cp -R op_host op_kernel "$project_dir/"
(cd "$project_dir" && bash build.sh)
```

Run these commands from this submission directory. `op.json` describes the
registered interface for project generation. The generated framework supplies
`opbuild` and `add_kernels_compile`, which the two CMake files require. Use
the generator and build instructions that accompany the selected SDK version.
The generated package is written under `$project_dir/build_out/`.

This operator implements `torch.cat(inputs, dim)` for a dynamic input list.
Inputs must have the same rank, data type, and shape on every axis except
`dim`. `dim` accepts negative indexing. The output keeps the common shape and
has the sum of input sizes on the concatenation axis.

## Interface

- Input: dynamic list `inputs` of ND tensors.
- Attribute: optional `dim` (`int`, default `0`), normalized to `[0, rank)`.
- Output: one ND tensor with the input data type.
- Data types declared by the op: `float32`, `float16`, `int32`, and `int8`.
- The host rejects missing inputs, rank mismatches, incompatible non-concat
  dimensions, negative shape dimensions, out-of-range `dim`, unsupported
  element sizes, and arithmetic overflow while computing logical row sizes.

## Tiling and kernel paths

The host turns the logical tensor into `outer x concat_width x inner` rows and
selects a path from the runtime shape, UB size, core count, and estimated DMA
work. The always-available paths are:

- **ROW (key 2)**: each core owns a balanced range of rows and copies every
  input segment into its output column range.
- **FLAT (key 1/6/9/10)**: for a single logical row, contiguous output ranges
  are split across cores; prepared variants store the input/offset commands in
  tiling data.
- **GRID (key 3/5/8)**: large rows are split across row groups and aligned
  column chunks; key 8 uses the single-buffer prepared-rectangle form.
- **INPUT_LPT (key 4)**: imbalanced input widths are split into weighted row
  tasks and assigned with a longest-processing-time schedule.
- **ROW_GATHER (key 7)**: narrow, high fan-in rows are staged in UB, gathered
  into complete rows, and written back once.

Several additional keys remain available for controlled experiments. Keys 48,
49, 55, and 65 are emitted only when their matching compile-time dispatch
macros are enabled; otherwise the host deliberately falls back to an
always-dispatched key so a disabled kernel cannot receive work.

## Validation

The host code performs checked arithmetic, validates all dynamic input
descriptors, bounds fixed-size tiling tables, and verifies UB/DMA encoding
limits before selecting an optimized plan. Kernel paths repeat key bounds
checks before issuing DMA or gather operations.

Dispatch guards were reviewed locally. CANN device compilation, accuracy
and performance have not been measured.
