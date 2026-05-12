# Face Recognizer API

Embeddable face recognition access-control server.  Detects faces,
verifies liveness (anti-spoofing), aligns, embeds, and matches against
a stored identity database backed by SQLite + FAISS.

## Quickstart

```bash
# Start server (all models loaded on boot — first request is fast)
uv run python scripts/serve.py

# Override port
uv run python scripts/serve.py server.port=9000

# Enroll a person (liveness ON by default)
curl -F "name=Alice" -F "file=@alice.jpg" http://localhost:8000/enroll

# Recognize a face
curl -F "file=@doorbell.jpg" http://localhost:8000/recognize
```

---

## Endpoints

### `GET /health`

Liveness probe.  Returns 200 if the server is running.

**Response** `200`
```json
{"status": "ok"}
```

---

### `GET /people`

List all enrolled identities.

**Response** `200`
```json
{
  "names": ["Alice", "Bob"],
  "threshold": 0.56,
  "vector_count": 2
}
```

| Field | Type | Description |
|---|---|---|
| `names` | `list[str]` | Enrolled person names, sorted alphabetically |
| `threshold` | `float` | Current cosine-similarity match threshold (adaptive) |
| `vector_count` | `int` | Total embedding vectors in the index |

---

### `DELETE /people/{name}`

Remove all embedding rows for a person and rebuild the FAISS index.

**Response** `200`
```json
{"removed": "Alice"}
```

---

### `POST /enroll`

Enroll a new person from a single face image.  Runs the full pipeline:
detect → liveness → align → embed → store.

After enrollment the match threshold is automatically recalibrated
from the updated population.

**Request** `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *required* | Person identifier |
| `file` | `file` | *required* | JPEG or PNG image containing a face |
| `skip_liveness` | `bool` | `false` | Bypass anti-spoofing check |

**Response** `200`
```json
{
  "name": "Alice",
  "row_id": 1,
  "liveness_score": null,
  "quality": 0.85,
  "threshold": 0.56
}
```

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Enrolled name |
| `row_id` | `int` | SQLite row ID |
| `liveness_score` | `float or null` | Liveness score (0–1), `null` if skipped |
| `quality` | `float` | Detection confidence |
| `threshold` | `float` | Match threshold after recalibration |

**Errors**

| Status | Body | When |
|---|---|---|
| `400` | `{"detail":"Empty file"}` | No file uploaded |
| `400` | `{"detail":"Could not decode image"}` | Corrupt / unsupported format |
| `400` | `{"detail":"No face detected"}` | No face found in image |
| `403` | `{"detail":"Spoof detected …"}` | Liveness check failed |
| `422` | *FastAPI validation* | Missing `name` or `file` |

---

### `POST /recognize`

Recognize a face in an uploaded image.  Runs the full pipeline and
returns the best match (if any) from the enrolled identity database.

**Request** `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | `file` | *required* | JPEG or PNG image |
| `skip_liveness` | `bool` | `false` | Bypass anti-spoofing |

**Response** `200`
```json
{
  "matched": true,
  "name": "Alice",
  "confidence": 0.83,
  "is_real": true,
  "face_bbox": [100, 50, 300, 350],
  "per_stage_ms": {"detect": 40.1, "liveness": 3.5, "align": 23.2, "embed": 53.0, "search": 0.02},
  "total_ms": 119.8
}
```

| Field | Type | Description |
|---|---|---|
| `matched` | `bool` | `true` if an identity met the threshold |
| `name` | `str or null` | Matched name, `null` if no match |
| `confidence` | `float or null` | Cosine similarity score (0–1) |
| `is_real` | `bool or null` | Liveness verdict, `null` if skipped |
| `face_bbox` | `[x1,y1,x2,y2] or null` | Face bounding box in pixels |
| `per_stage_ms` | `dict[str,float]` | Per-stage latency |
| `total_ms` | `float` | Total pipeline latency |

**Errors**

| Status | Body | When |
|---|---|---|
| `400` | `{"detail":"Empty file"}` | No file uploaded |
| `400` | `{"detail":"Could not decode image"}` | Corrupt / unsupported format |

---

### `POST /calibrate`

Recalibrate the match threshold from the currently enrolled population.
Finds the maximum cross-identity cosine similarity and adds a safety
margin.  Called automatically after every enrollment — use this to
recalibrate manually (e.g., after batch deletions).

**Request** `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `margin` | `float` | `0.05` | Safety margin added to max cross-identity similarity |

**Response** `200`
```json
{
  "threshold": 0.5632,
  "max_cross_id": 0.5132,
  "margin": 0.05,
  "identities": 20,
  "vectors": 20
}
```

| Field | Type | Description |
|---|---|---|
| `threshold` | `float` | New match threshold |
| `max_cross_id` | `float` | Highest similarity between different identities |
| `margin` | `float` | Safety margin used |
| `identities` | `int` | Number of distinct enrolled people |
| `vectors` | `int` | Total embedding vectors in the index |

---

## Liveness

Liveness (anti-spoofing) is **on by default** for enrollment.
Static photos, screenshots, and video replays are rejected.

Use `skip_liveness=true` when:
- Enrolling from pre-cropped face datasets (LFW, etc.)
- Testing with still images
- The deployment environment guarantees physical presence

---

## Adaptive Thresholding

The match threshold is not a fixed number — it adapts to the enrolled
population.  After each enrollment, `calibrate_threshold()` scans all
cross-identity similarities and sets the threshold just above the most
confusable pair, guaranteeing zero false positives against the
enrolled set.

If no faces match at the current threshold, lower it via recalibration:
```bash
curl -X POST -F "margin=0.02" http://localhost:8000/calibrate
```

---

## Configuration

All settings are in `configs/config.yaml`.  Override at startup:

```bash
uv run python scripts/serve.py detector.device=mps embedder.device=mps
```

| Section | Key | Default | Description |
|---|---|---|---|
| `server` | `host` | `0.0.0.0` | Bind address |
| `server` | `port` | `8000` | Listen port |
| `server` | `log_level` | `info` | uvicorn log level |
| `pipeline` | `skip_liveness` | `false` | Server-wide liveness default |
| `pipeline` | `skip_aligner` | `false` | Server-wide alignment bypass |
| `detector` | `model_name` | `models/yolo26n_tuned.pt` | YOLO model path |
| `detector` | `confidence_threshold` | `0.5` | Min detection confidence |
| `detector` | `device` | `cpu` | `cpu`, `cuda`, `mps` |
| `liveness` | `threshold` | `0.5` | Spoof detection threshold |
| `aligner` | `model_id` | `minchul/cvlface_DFA_mobilenet` | HuggingFace model |
| `embedder` | `weights_path` | `weights/arcface-r100-glint360k.pth` | ArcFace checkpoint |
| `store` | `threshold` | `0.65` | Initial match threshold (overwritten by calibration) |
| `store` | `db_path` | `data/identity.db` | SQLite database file |
| `store` | `index_path` | `data/faiss.index` | FAISS index file |

---

## Pipeline Architecture

```
Enroll:   Image → Detect(YOLO) → Liveness(MiniFAS) → Align(DFA) → Embed(ArcFace) → SQLite + FAISS
Recognize: Image → Detect(YOLO) → Liveness(MiniFAS) → Align(DFA) → Embed(ArcFace) → FAISS search → SQLite lookup
```

All models are loaded on server startup (warmup).  Typical inference
latency: ~120 ms on CPU (macOS aarch64).
