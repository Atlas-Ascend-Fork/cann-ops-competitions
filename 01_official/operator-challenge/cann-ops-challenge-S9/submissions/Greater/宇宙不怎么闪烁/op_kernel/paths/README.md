# Greater device paths

These headers are private class-body/source fragments included by
`../greater.cpp`. They keep the generated Ascend C translation unit identical in
logic while making each data path independently readable:

- `config.h`: named hardware/layout constants and alignment helpers.
- `dispatch.h`: kernel initialization and fast-path selection.
- `buffers.h`: dtype-dependent temporary-buffer setup.
- `dense.h`: same-shape FP32/FP16/INT8 pipeline.
- `bf16_broadcast.h`: BF16 row-broadcast and generic broadcast pipelines.
- `channel_scalar.h`: `[N,C,K] > [C,1]`-style cached scalar broadcast.
- `channel_row.h`: `[N,C,H,W] > [C,1,W]`-style row broadcast.
- `int32_row.h`: INT32 `[N,C,1,W] > [N,C,H,W]` path, including compact Gather.
- `generic.h`: shape-general broadcast/copy fallback and row cache.
- `type_compute.h`: generic-path dtype-specific vector materialization.

Keep declarations in `greater.cpp`; add a specialized path only when it has a
shape/dtype predicate and the generic fallback remains correct. Performance
changes must be measured against an accepted build and must not alter the
framework timer or skip inputs, output elements, or tail handling.
