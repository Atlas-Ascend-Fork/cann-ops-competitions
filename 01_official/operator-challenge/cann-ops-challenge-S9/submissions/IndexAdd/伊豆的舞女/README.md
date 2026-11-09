# IndexAdd

`IndexAdd` computes an in-place style indexed accumulation into a separate output:

```text
output = self
for m in range(index.shape[0]):
    output[..., index[m], ...] += source[..., m, ...]
```

The `dim` attribute selects the indexed axis and accepts negative values in the
usual Python sense (`-1` means the last axis). `self` and `source` must have the
same rank and element type; `index` is a one-dimensional `int32` tensor; the
source dimension at `dim` must equal `index.shape[0]`; all other source
dimensions must match `self`. Supported element types are `float`, `float16`,
`bfloat16`, `int32`, and `int8` in ND format. The output has the shape and type
of `self`.

The device code skips an index outside `[0, self.shape[dim])`, including a
negative value, instead of issuing an out-of-bounds access. Callers should pass
in-range indices when they require strict `torch.index_add` error semantics.

## Tiling and scheduling

The host computes flattened `outer`, `axis`, and `inner` sizes and chooses a
strategy from the available UB and core count:

- atomic tiled copy/add for general shapes;
- resident UB accumulation for `float`, `float16`, and `int32`;
- narrow masked vector updates and Gather based paths for scalar-inner layouts;
- BF16 accumulation through FP32 temporaries;
- INT8 accumulation through FP16/INT32 conversion, including an axis-oriented
  compare path when it is cheaper;
- cooperative multi-core metadata construction for large Gather tables, with
  optional double buffering and transpose-block Gather variants.

Specialized paths check UB, DMA and workspace limits before selection and
fall back to a general strategy when a specialized layout does not fit.

## Validation

Index consumption paths were reviewed locally.
CANN device compilation, accuracy and performance have not been measured.

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
