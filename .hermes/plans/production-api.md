# Production Face Recognition API — Implementation Plan

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task.

**Goal:** Rewrite the FastAPI server to use the production pipeline (detect → liveness → align → embed → SQLite/FAISS), expose `/enroll` and `/recognize` endpoints with JSON responses, ready for embedding into door-lock systems.

**Architecture:** FastAPI + uvicorn, factory pattern. Pipeline stages built from Hydra config and injected into the app factory. The existing `FacePipeline` orchestrator (detect → liveness → align → embed → store) and `IdentityStore` (SQLite + FAISS) are the only dependencies — old `FaceMatcher`, `FaceEnroller`, `FaceRestorer` are removed.

**Tech Stack:** FastAPI 0.104+, uvicorn 0.24+, python-multipart 0.0.6+, Pydantic (built into FastAPI), httpx (test client).

---

### Task 1: Clean up serve.py — use new pipeline components

**Objective:** Replace old imports (restorer, matcher, clusterer) with the current pipeline (aligner, IdentityStore).

**File:** `scripts/serve.py`

**What changes:**
- Remove imports: `FaceRestorer`, `FaceMatcher`, `FaceEnroller`
- Add imports: `FaceAligner`, `IdentityStore`
- Build pipeline with current components
- Call `store.load()` and `store.calibrate_threshold()` on startup
- Pass only `pipeline` and `store` to `create_app()`

---

### Task 2: Rewrite server/app.py — new endpoint handlers

**Objective:** Replace the app factory with handlers that use `FacePipeline.enroll()` and `FacePipeline.process()`.

**File:** `src/server/app.py`

**New factory signature:**
```python
def create_app(pipeline: FacePipeline, store: IdentityStore) -> FastAPI:
```

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness probe (keep) |
| GET | `/people` | List enrolled names → `store.list_names()` |
| DELETE | `/people/{name}` | Remove a person → `store.delete(name)` |
| POST | `/recognize` | Upload image → `pipeline.process(image)` → JSON |
| POST | `/enroll` | Upload image + name → `pipeline.enroll(image, name)` → JSON |
| POST | `/calibrate` | Re-calibrate threshold → `store.calibrate_threshold()` |

**Key changes from current:**
- `/enroll` takes 1 image (not 3-15 frames) with liveness on by default
- `/enroll` returns `{name, row_id, liveness_score, quality, threshold}`
- `/recognize` returns `{matched, name, confidence, is_real, face_bbox, per_stage_ms, total_ms}`
- Remove old `matcher.list_people()`, `matcher.remove()`, `enroller.enroll()` calls

---

### Task 3: Add Pydantic response schemas

**Objective:** Define typed response models for API documentation and validation.

**File:** `src/server/schemas.py` (new)

```python
from pydantic import BaseModel

class RecognizeResponse(BaseModel):
    matched: bool
    name: str | None
    confidence: float | None
    is_real: bool | None
    face_bbox: list[float] | None

class EnrollResponse(BaseModel):
    name: str
    row_id: int
    liveness_score: float | None

class PeopleResponse(BaseModel):
    names: list[str]
    threshold: float
    vector_count: int
```

---

### Task 4: Update server.yaml config

**Objective:** Remove old enrollment settings (num_clusters, min_cluster_spread), add door-lock config.

**File:** `configs/server.yaml`

**New config:**
```yaml
host: 0.0.0.0
port: 8000
log_level: info

# Liveness defaults (can be overridden per-request)
enrollment:
  skip_liveness: false

# Door lock relay (optional, for future hardware integration)
door:
  relay_url: null
  open_timeout_sec: 5
```

---

### Task 5: Write API tests

**Objective:** Test both endpoints end-to-end with real pipeline.

**File:** `tests/test_api.py` (new or update)

**Tests:**
1. `test_health` — GET /health → 200
2. `test_enroll_and_recognize` — POST /enroll with image → 200 → POST /recognize with image → matched=True
3. `test_recognize_unknown` — POST /recognize with image → matched=False
4. `test_list_people` — GET /people → includes enrolled name
5. `test_delete_person` — DELETE /people/{name} → 200 → GET /people → name removed
6. `test_empty_file` — POST /recognize with empty → 400
7. `test_missing_name` — POST /enroll without name → 422

**Test setup:**
```python
import pytest
from httpx import ASGITransport, AsyncClient
from src.server.app import create_app
from src.pipeline.orchestrator import FacePipeline

# Use tmp_path fixture for isolated DB
@pytest.fixture
async def client(tmp_path):
    # Build pipeline with test config pointing to tmp_path
    # store.load(), then create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

---

### Task 6: Integration smoke-test

**Objective:** Start server, curl both endpoints, verify JSON output.

```bash
# Terminal 1: start server
uv run python scripts/serve.py server.port=8765

# Terminal 2: smoke test
curl -s http://localhost:8765/health
curl -s http://localhost:8765/people
curl -s -F "name=TestUser" -F "file=@samples/footage_01.png" http://localhost:8765/enroll
curl -s -F "file=@samples/footage_01.png" http://localhost:8765/recognize | python -m json.tool
curl -s -X DELETE http://localhost:8765/people/TestUser
```

Expected: `/health` → `{"status":"ok"}`, `/enroll` → `{"name":"TestUser",...}`, `/recognize` → `{"matched":true,"name":"TestUser",...}`.

---

### Dependencies Check

FastAPI, uvicorn[standard], python-multipart already in `pyproject.toml`. httpx for tests is in dev deps. No new dependencies needed.

### Files Summary

| Action | File |
|--------|------|
| Modify | `scripts/serve.py` |
| Rewrite | `src/server/app.py` |
| Create | `src/server/schemas.py` |
| Modify | `configs/server.yaml` |
| Create | `tests/test_api.py` |

### Order: 1 → 2 → 3 → 4 → 5 → 6
