# FPS Timing Scope Audit

This audit records the timing scope of `scripts/benchmark_fps.py`.

## Relevant CLI Options

`scripts/benchmark_fps.py` exposes:

- `--imgsz`
- `--batch-size`
- `--warmup`
- `--iters`
- `--precision`
- `--device`
- `--no-postprocess`

The `--no-postprocess` flag disables the DRQ/DEIM postprocessor when timing
DRQ-DETR-style models.

## DRQ/DEIM Timing Path

In `benchmark_deim`, the synthetic input tensor is created before timing:

```python
images = torch.rand(args.batch_size, 3, args.imgsz, args.imgsz, device=device)
```

The warmup loop runs `model(images)` and, if enabled, the postprocessor before
measured timing begins.

For CUDA timing, the measured model interval starts immediately before
`model(images)` and ends immediately after model forward:

```python
model_start.record()
outputs = model(images)
model_end.record()
```

Post-processing is measured in a separate interval only when
`include_postprocess` is true:

```python
if include_postprocess:
    post_start.record()
    _ = postprocessor(outputs, orig_sizes)
    post_end.record()
```

For CPU timing, the same scope is represented by:

```python
start = time.perf_counter()
outputs = model(images)
mid = time.perf_counter()
if include_postprocess:
    _ = postprocessor(outputs, orig_sizes)
end = time.perf_counter()
```

## Conclusion

With `--no-postprocess`, the DRQ/DEIM timing result records model forward only.
The synthetic tensor is allocated before the measured loop, so dataset loading
and image preprocessing are outside the timer. NMS and other post-processing
are also outside the reported forward-only paper timing when `--no-postprocess`
is used.

The paper Table 8 values should therefore be described as:

```text
model forward only; preprocessing, NMS, and other post-processing excluded
```
