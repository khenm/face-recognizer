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

Key settings:

| Section | Field | Default | Notes |
|---|---|---|---|
| `server` | `port` | `8000` | Bind port |
| `detector` | `device` | `cpu` | `cpu`, `cuda`, `mps` |
| `embedder` | `device` | `cpu` | `cpu`, `cuda`, `mps` |
| `store` | `threshold` | `0.65` | Initial threshold (auto-calibrated) |
| `pipeline` | `skip_liveness` | `false` | Global liveness override |

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Model weights downloaded via `scripts/download_models.py`
- macOS: `KMP_DUPLICATE_LIB_OK=TRUE` (handled automatically by scripts)

## Troubleshooting

### Models not found

Run `uv run python scripts/download_models.py`.  This downloads all
required weights to `models/` and `weights/`.

### macOS: OpenMP symbol conflicts

If you see errors about duplicate OpenMP symbols:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

The server and CLI scripts set this automatically, but custom integrations
may need it.  This is harmless and only needed on macOS.

### No face detected on enrollment

Enrollment runs liveness (anti-spoofing) by default.  Still images and
screenshots will be rejected as spoofs.  Use `skip_liveness=true` for
testing with photos, or use a real camera in production.

### Recalibrating the match threshold

The threshold adapts to your enrolled population after each enrollment.
To force recalibration:

```bash
curl -X POST -F "margin=0.05" http://localhost:8000/calibrate
```

Lower the margin if matches are too strict.

## Database portability

The identity database is two files: `data/identity.db` (SQLite) and
`data/faiss.index` (FAISS).  Both are architecture-independent and can
be copied between servers:

```bash
scp data/identity.db data/faiss.index other-server:face-recognizer/data/
```

For multi-server setups (multiple door locks), either sync the files
after each enrollment or run a single enrollment server that workers
poll.
