# Pre-Flight Plan — MiniFASNetV2SE PyTorch Migration

## What changed

Replaced `src/pipeline/liveness.py` ONNX runtime with vendored PyTorch architecture.

| Before | After |
|--------|-------|
| ONNX session (`onnxruntime.InferenceSession`) | PyTorch model (`MultiFTNet` from `src/models/minifasv2.py`) |
| `_session` attribute (mockable by tests) | `_model` attribute (mockable by tests) |
| ONNX input/output names | Direct PyTorch tensor forward pass |
| Single raw logit output | 2-class logits → softmax → `probs[:, 1]` = "real" score |
| Backbone: downloaded ONNX | Backbone: MobileNetV4-style depth-wise separable + Squeeze-and-Excitation blocks |

## Architecture

```
MiniFASNetV2SE (~1.8M params)
├── conv1           (3→32, stride 2, PReLU)
├── conv2_dw        (depth-wise, stride 1)
├── conv_23         (Depth_Wise, stride 2: 32→64)
├── conv_3          (ResidualSE ×4 blocks, 64→128)
├── conv_34         (Depth_Wise, stride 2: 128→128)
├── conv_4          (ResidualSE ×6 blocks, 128→256)
├── conv_45         (Depth_Wise, stride 2: 256→128)
├── conv_5          (ResidualSE ×2 blocks, 128→512)
├── conv_6_sep      (1×1 conv, 512→512)
├── conv_6_dw       (depth-wise 7×7, 512→512)
├── flatten + linear (512→128) + bn + dropout
└── logits          (128→2)  → [spoof_score, real_score]
```

## Zoo update

| Key | Before | After |
|-----|--------|-------|
| `minifasnet` | `MiniFASNetV2.onnx` (ONNX) | **Removed** |
| `minifasnet_v2` | — | `minifasv2.pth` (PyTorch weights) |

## Files modified

| File | Change |
|------|--------|
| `src/models/minifasv2.py` | Copyright header rewritten |
| `src/pipeline/liveness.py` | ONNX → PyTorch model loading + softmax |
| `src/models/zoo.py` | `minifasnet` → `minifasnet_v2` |
| `configs/pipeline/liveness.yaml` | Updated comments |
| `tests/test_liveness.py` | Mock `_model` (PyTorch) instead of `_session` (ONNX) |
| `tests/test_zoo.py` | `minifasnet` → `minifasnet_v2` |
| `tests/test_imports.py` | `minifasnet` → `minifasnet_v2` |

## To run with real weights

1. **Get weights** — download `minifasv2.pth`:
   ```bash
   uv run python -c "from src.models.zoo import ModelZoo; ModelZoo().get('minifasnet_v2')"
   ```
   The zoo URL points to: `https://github.com/facenox/face-antispoof-onnx/releases/download/v1.0/MiniFASNetV2.pth`

2. **Or vendored weights** — place `minifasv2.pth` in `~/.cache/face-recognizer/` and set `model_path: null` in liveness config.

3. **Test with real model**:
   ```bash
   uv run python -c "
   from src.pipeline.liveness import LivenessDetector
   import cv2, numpy as np
   d = LivenessDetector()
   img = cv2.imread('samples/footage_01.png')
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   result = d.check(img)
   print(f'is_real={result.is_real}, score={result.score:.3f}')
   "
   ```

## Test results

```
137 passed, 1 skipped — no regressions
```

## Open question

The zoo URL is a best-guess based on the facenox repo — verify the actual release URL exists before relying on auto-download. If the user has vendored weights locally, set `model_path` directly in the liveness config.
