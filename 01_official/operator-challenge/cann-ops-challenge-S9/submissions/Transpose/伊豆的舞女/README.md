# Transpose (Ascend C, S9)

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

This directory implements the custom `Transpose` operator used by the S9
operator challenge.  The public interface is the ND tensor form

```text
Transpose(inputs, dims) -> output
```

`inputs` and `output` have the same dtype and rank.  The supported dtypes are
`float32`, `float16`, `int32`, and `int8`; `dims` is a permutation of the
input axes and accepts negative axis indices.  The host validates that `dims`
has exactly the input rank and contains each axis once, then infers the output
shape as `input_shape[dims[i]]`.

## Tiling and device plans

The host computes 64-bit element counts and strides, removes size-one axes
when recognizing layouts, and chooses the cheapest valid plan for the current
UB size and AIV core count.  All plans use `taskCount` independent logical
tasks and set `blockDim = min(core_count, taskCount)` (with the dedicated
16-bit single-TBuf path optionally grouping two stripes per AIV).  Hardware
parameter widths and 32-byte alignment are checked before a specialized plan
is selected; shapes that do not fit a specialized transfer fall back to the
generic affine copier.

The device implementation contains these plan families:

- contiguous copy for identity permutations and empty tensors;
- dense matrix transpose for a single fixed axis (`TransDataTo5HD`);
- strided matrix transpose for fixed and non-fixed axes, including panelized
  MTE2/MTE3 transfers and optional phase-compaction paths for FP32;
- record transpose using `TransDataTo5HD` or Gather for non-block-aligned
  records;
- a generic affine path that copies contiguous output suffixes in UB-sized
  chunks.

Specialized kernels only read and write the valid tail extent.  For record
transfers, a record larger than the device `DataCopyExtParams.blockLen`
limit is rejected during tiling so the generic path handles it safely.

## Validation status

Record DMA length bounds were reviewed locally.
CANN device compilation, accuracy and performance have not been measured.
