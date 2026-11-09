# Greater (Ascend S9)

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

`Greater` implements elementwise `torch.gt(self, other) -> bool` for ND
tensors.  Inputs use trailing-axis broadcasting, so each axis is compatible
when the two dimensions are equal or one of them is `1`; scalar inputs are
valid.  Both inputs must have the same dtype.  The declared input dtypes are
`float32`, `float16`, `bfloat16`, `int32`, and `int8`; the output is an ND
`bool` tensor with the broadcast shape.

## Tiling and kernel paths

The host folds adjacent axes with the same broadcast behavior, computes
checked 64-bit element counts and affine strides, and chooses a plan from the
runtime UB/L2 sizes and AIV core count:

- **Linear span** (key `6`, or key `7` for one-core whole-tensor work) copies
  contiguous input spans and uses aligned UB tiles.
- **Resident row** keeps the broadcast row in UB while batches of full rows
  stream through it.  Group-major and row-major ownership are selected from
  estimated DMA and fragmentation costs.
- **Affine row** handles general broadcast layouts with per-row strides.
- **INT32 resident variants** use a strict-`GT` transform based on Min/Max,
  equality, and Select, avoiding an unverified INT32 floating conversion.  The
  pitched and packed variants are enabled only when repeat, phase, repeat
  stride, and UB ledger limits are satisfied.

DMA tails are padded in UB and written with valid byte counts, so logical output
bytes outside the requested shape are not modified.  The host rejects dtype
mismatches, incompatible dimensions, rank overflow beyond the fixed tiling
table, and checked-arithmetic/UB planning failures.

## Validation

Fixed INT32 buffer capacity and host arithmetic were checked locally.
CANN device compilation, accuracy and performance have not been measured.
