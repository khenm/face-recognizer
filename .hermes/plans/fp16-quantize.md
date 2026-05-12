# General-Purpose FP16 Quantization Tool

## Goal

A `scripts/quantize.py` CLI that can quantize **any** model in the zoo to FP16:

```bash
uv run python scripts/quantize.py --model minifasnet_v2
uv run python scripts/quantize.py --model codeformer
uv run python scripts/quantize.py --model resnet100_adaface
uv run python scripts/quantize.py --model minifasnet_v2 --output models/minifasv2_fp16.pth
```

## Architecture

```
scripts/quantize.py          ← CLI entrypoint (argument parsing, orchestrates)
src/quantize/__init__.py     ← library: quantize_pytorch(), quantize_onnx()
src/pipeline/liveness.py     ← add fp16 flag (consume quantized model at runtime)
configs/pipeline/liveness.yaml ← add fp16: false
```

## Task list

### Task 1: Quantize library (`src/quantize/__init__.py`)

Two functions:

- **`quantize_pytorch(src_path, dst_path)`** — loads FP32 PyTorch checkpoint,
  converts to `model.half()`, saves FP16 `.pth`. Handles `state_dict` wrapper.
  Works for any PyTorch model that the zoo knows how to load.

- **`quantize_onnx(src_path, dst_path)`** — stub for ONNX models (CodeFormer, etc.),
  converts via ONNX Runtime FP16 optimization pass if needed. Or skips with a
  message if not applicable.

- Helper: **`_load_model_by_name(name, zoo)`** — given a model name, returns the
  loaded PyTorch `nn.Module` in FP32. Uses the same loading logic as the pipeline
  stages (avoids duplication).

### Task 2: CLI (`scripts/quantize.py`)

```
usage: quantize.py [-h] --model MODEL [--source PATH] [--output PATH] [--device DEVICE]

Quantize a model to FP16.

options:
  --model MODEL    Model name from zoo (minifasnet_v2, codeformer, resnet100_adaface)
  --source PATH    Path to FP32 weights (default: auto-download from zoo)
  --output PATH    Output path for FP16 weights (default: <source>_fp16.pth)
  --device DEVICE  Device for loading/quantizing (default: cpu)
```

Flow:
1. Resolve model name → load FP32 model using zoo + model-specific loader
2. `model.half()` → FP16
3. Save to output path
4. Print summary: input size, output size, time taken

### Task 3: Runtime FP16 flag

**`configs/pipeline/liveness.yaml`** — add `fp16: false`:
```yaml
fp16: false    # set true to load model in half precision
```

**`src/pipeline/liveness.py`** — in `_load_model()`:
```python
if self.fp16:
    model = model.half()
```

Same pattern can be added to embedder/restorer later, but start with liveness
since MiniFASNet is the primary target.

### Task 4: Tests

**`tests/test_quantize.py`** (5 tests):
- `test_quantize_pytorch_roundtrip` — MiniFASNet FP32 → FP16 → load → verify dtype `float16`
- `test_quantize_pytorch_output_consistent` — FP32 vs FP16 forward pass outputs within `atol=1e-2`
- `test_quantize_cli_help` — `--help` prints usage
- `test_quantize_cli_invalid_model` — unknown model name → error
- `test_quantize_creates_output_file` — output file exists after quantization

**`tests/test_liveness.py`** — add 1 test:
- `test_check_fp16_mode` — with `fp16=True`, mock model returns half-precision tensor → result still correct

## Files

| File | Action |
|------|--------|
| `src/quantize/__init__.py` | New — `quantize_pytorch`, `quantize_onnx` |
| `scripts/quantize.py` | New — CLI entrypoint |
| `src/pipeline/liveness.py` | Add `fp16` param, `model.half()` |
| `configs/pipeline/liveness.yaml` | Add `fp16: false` |
| `tests/test_quantize.py` | New — 5 tests |
| `tests/test_liveness.py` | Add 1 test |

## Estimated scope

~150 lines implementation, ~6 new tests. Medium (~45 min).
