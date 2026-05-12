# Face Recognizer

Embeddable face recognition access-control server.  Detect → liveness
(anti-spoofing) → align → embed → match against a SQLite + FAISS identity
database.  Ready to embed into door locks, kiosks, or any system that
needs to know *who* is at the door.

```text
Camera → YOLO Detection → Liveness (MiniFASNet) → DFA Alignment → ArcFace Embedding → FAISS Match → Unlock
```

## Quickstart (5 minutes)

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Clone and install
git clone <repo-url>
cd face-recognizer
uv sync

# 2. Download model weights (~300 MB total)
uv run python scripts/download_models.py

# 3. Start the server
uv run python scripts/serve.py

# 4. Enroll a person (liveness ON by default — use a real camera)
curl -F "name=Alice" -F "skip_liveness=true" -F "file=@alice.jpg" http://localhost:8000/enroll

# 5. Recognize
curl -F "file=@doorbell.jpg" http://localhost:8000/recognize
# → {"matched":true,"name":"Alice","confidence":0.83,...}
```

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/people` | List enrolled identities |
| `DELETE` | `/people/{name}` | Remove a person |
| `POST` | `/enroll` | Enroll a face (image + name, liveness on) |
| `POST` | `/recognize` | Recognize a face (image → match) |
| `POST` | `/calibrate` | Recalibrate match threshold |

Full API reference: [`docs/api.md`](docs/api.md)

## Architecture

| Stage | Model | Size | Purpose |
|---|---|---|---|
| Detect | YOLO26-Nano (WIDER Face) | 6 MB | Face bounding boxes |
| Liveness | MiniFASNetV2SE | 7 MB | Anti-spoofing (photos/screens) |
| Align | DFA mobilenet (HuggingFace) | 5 MB | Canonical 112×112 alignment |
| Embed | IResNet-100 ArcFace (Glint360K) | 249 MB | 512-dim face vector |
| Match | FAISS IndexFlatIP + SQLite | — | Cosine similarity search |

Latency: ~120 ms per frame (CPU, macOS aarch64).  Models are pre-loaded
on startup — first request is as fast as the hundredth.

## Configuration

Everything lives in one file: [`configs/config.yaml`](configs/config.yaml)

```bash
# Override at startup
uv run python scripts/serve.py server.port=9000 detector.device=mps
```
