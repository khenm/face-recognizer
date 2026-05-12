# Pre-Flight Checklist — Real Run

Before starting a real run, work through each section top-to-bottom.

---

## 1. Model Weights (3 of 4 missing)

| Model | File | Status | Action |
|-------|------|--------|--------|
| YOLO26-Nano | `yolo26n.pt` | ✅ 5.5 MB | Ready |
| MiniFASNetV2 | `~/.cache/face-recognizer/MiniFASNetV2.onnx` | ❌ | Download from zoo |
| CodeFormer | `~/.cache/face-recognizer/codeformer.pth` | ❌ | Download from zoo |
| ResNet-100 AdaFace | `~/.cache/face-recognizer/resnet100_adaface.pth` | ❌ | Download or provide custom weights |

### To download:

```bash
uv run python -c "
from src.models.zoo import ModelZoo
zoo = ModelZoo()
zoo.get('minifasnet')       # ~1.5 MB ONNX model
zoo.get('codeformer')       # ~400 MB (large!)
zoo.get('resnet100_adaface') # auto-download from insightface
"
```

> **⚠️ CodeFormer is ~400 MB.** The current restorer is a stub — it will pass through until you integrate the actual sczhou/CodeFormer code. The stub is functional (returns input unchanged) so you can still run the pipeline, just without denoising.

> **⚠️ ResNet-100 AdaFace** — the URL in the zoo is a fallback to insightface buffalo_l. Replace with actual AdaFace-margin weights for best accuracy.

### Current stub behavior:

| Stage | Without weights | Effect |
|-------|----------------|--------|
| Restorer | Stub pass-through | No denoising, face crop passes unchanged to embedder |
| Embedder | Random init | Embeddings will be random → useless for recognition |

> **🔴 Embedder MUST have real weights** for the system to work. Without them, every embedding is random and FAISS will never match.

---

## 2. FAISS Index

```bash
# Create data directory
mkdir -p data

# Index is auto-created by the server on first run,
# or manually via:
uv run python -c "
from src.pipeline.matcher import FaceMatcher
m = FaceMatcher(index_path='data/faiss.index')
m.save()  # creates empty index
"
```

| Item | Status | Action |
|------|--------|--------|
| `data/` directory | ❌ | `mkdir -p data` |
| `data/faiss.index` | ❌ | Auto-created on first server start |

---

## 3. Enrollment Data

| Item | Status | Action |
|------|--------|--------|
| Sample frames | ✅ 3 frames (`footage_01/02/03.png`) | Need 15 per person for K-Means |
| Per-person frames | ❌ | Capture or provide 15 head-rotation frames per person |

### Enrollment command:

```bash
uv run python scripts/enroll.py --name "Alice" samples/footage_01.png samples/footage_02.png samples/footage_03.png
```

> **⚠️ 3 frames is below the minimum 15.** K-Means will still run (k=3 on 3 points = each point is its own cluster), but pose anchors won't be representative. Capture 15 frames with head rotation for best results.

---

## 4. Configuration Review

Edit `configs/pipeline/pipeline.yaml` and `configs/server.yaml`:

| Setting | Default | Check |
|---------|---------|-------|
| `detector.device` | `cpu` | Set to `cuda` if GPU available |
| `liveness.device` | `cpu` | Set to `cuda` if GPU available |
| `restorer.device` | `cpu` | Set to `cuda` if GPU available |
| `embedder.device` | `cpu` | Set to `cuda` if GPU available |
| `matcher.threshold` | `0.65` | Tune after testing — higher = stricter |
| `skip_liveness` | `false` | Set `true` if MiniFASNet unavailable |
| `skip_restorer` | `false` | Set `true` if CodeFormer unavailable |
| `door_relay_url` | `null` | Set to ESP32 relay IP (e.g. `http://192.168.1.100/relay`) |
| `server.port` | `8000` | Confirm not in use |

---

## 5. ESP32-CAM Setup (if using)

| Item | Status | Action |
|------|--------|--------|
| ESP32-CAM flashed | ❌ | Flash with CameraWebServer example or custom firmware |
| Wi-Fi configured | ❌ | Set SSID/password in ESP32 firmware |
| Static IP | ❌ | Reserve in router DHCP or set static |
| Stream URL known | ❌ | Usually `http://<esp32-ip>/capture` |
| Server reachable | ❌ | ESP32 must be on same LAN as server |

### Capture script (not implemented, manual for now):

```bash
# Manual test: capture from ESP32 and POST to server
curl -o capture.jpg http://192.168.1.50/capture
curl -X POST -F "file=@capture.jpg" http://localhost:8000/recognize
```

---

## 6. Door Relay (if using)

| Item | Status | Action |
|------|--------|--------|
| Relay hardware | ❌ | Shelly, Sonoff, or ESP8266 relay |
| Relay URL | ❌ | HTTP endpoint to trigger (e.g. `http://192.168.1.101/relay/0?turn=on`) |
| Configured in pipeline | ❌ | Set `door_relay_url` in `pipeline.yaml` |

---

## 7. Start Server

```bash
uv run python scripts/serve.py
# With overrides:
uv run python scripts/serve.py server.port=9000 matcher.threshold=0.7
```

### Health check:

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

## 8. Verify End-to-End

```bash
# 1. Check health
curl http://localhost:8000/health

# 2. Recognize a sample image
curl -X POST -F "file=@samples/footage_01.png" http://localhost:8000/recognize

# 3. Enroll someone (after collecting 15 frames)
curl -X POST -F "name=Alice" \
  -F "files=@frame_01.png" \
  -F "files=@frame_02.png" \
  ... \
  http://localhost:8000/enroll

# 4. List enrolled people
curl http://localhost:8000/people

# 5. Recognize Alice
curl -X POST -F "file=@alice_test.png" http://localhost:8000/recognize
```

---

## Summary: Critical Blockers

| Priority | Item | Why it blocks |
|----------|------|---------------|
| 🔴 **P0** | ResNet-100 AdaFace weights | Embedder returns random vectors without real weights. FAISS will never match. |
| 🔴 **P0** | Enrollment data (15 frames/person) | Nothing to match against. |
| 🟡 **P1** | MiniFASNet ONNX model | Without it, anti-spoofing is non-functional (set `skip_liveness=true` as workaround). |
| 🟢 **P2** | CodeFormer weights + integration | Restorer is a stub. Non-blocking — pipeline works without denoising. |
| 🟢 **P2** | ESP32-CAM hardware | Manual curl testing works without it. |
| 🟢 **P2** | Door relay hardware | Pipeline returns match results regardless. |
