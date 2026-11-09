# SquareSumV1

`SquareSumV1` computes the sum of squares along the axes in `axis`.

## Contract

- Input: an ND tensor of rank 1--8 with positive dimensions and dtype
  `float16`, `bfloat16`, or `float32`.
- `axis`: a list of distinct axes. Negative axes are normalized from the
  end (`-1` is the last axis). In this implementation an empty list means
  all axes, matching the host-side parser used by both shape inference and
  tiling.
- `keep_dims`: optional boolean, default `false`. Reduced dimensions are
  removed when false and replaced by length-one dimensions when true.
- Output: the same dtype as the input. General half/BF16 paths convert to
  FP32 for the square and reduction, then cast back for the output; FP32 is
  accumulated directly. A bounded half AR fast path keeps the vector
  reduction in half precision for small, single-tile rows.

Invalid rank, non-positive dimensions, out-of-range or duplicate axes, and
unsupported dtypes cause host shape/tiling failure.

## Tiling and kernel paths

The host flattens contiguous reduced/non-reduced runs and selects one of the
following layouts:

- **AR**: normal dimensions followed by a reduction run.
- **ARA**: normal, reduction, normal runs.
- **RAR**: reduction, normal, reduction runs.
- **RA**: reduction followed by normal dimensions. This path uses per-core
  FP32 partials in workspace and a cross-core merge.
- **Generic**: arbitrary axis patterns, processed by a correctness-first
  scalar indexing fallback.

For suitable large or unaligned workloads, the host may select compact
packing, phase accumulation, double buffering, L2 bypass, and wave/staggered
schedules. These are guarded by UB capacity, DMA alignment, repeat-count, and
accumulation-depth checks; otherwise the corresponding general path is used.

## Validation status

Source and arithmetic boundary checks cover the half AR output slot and vector
repeat limit. No CANN compilation, device accuracy test, or performance run
has been performed in this environment.

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
