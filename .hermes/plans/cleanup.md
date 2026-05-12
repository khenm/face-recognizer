# Codebase Cleanup & API Docs — Implementation Plan

> **For Hermes:** Execute carefully — none of these changes should break the 9 API tests.

**Goal:** Remove dead code, unify configs, clean comments, and write API documentation.

**Constraint:** 9/9 tests in `tests/test_api.py` must pass after every change. Run `uv run pytest tests/test_api.py -v` after each task.

---

## Phase 1: Delete Dead Code

### Task 1.1 — Delete unused training utilities

**Files to delete:**
```
src/utils/                   # entire directory (10 files)
  __init__.py
  checkpoint.py
  dist.py
  env.py
  freeze.py
  fsdp.py
  general.py
  gradient_clip.py
  logging.py
  optimizer.py
  tensorboard_writer.py
```

These are PyTorch training utilities (gradient clipping, FSDP, distributed training, checkpointing). The face-recognizer is inference-only. Nothing imports from `src.utils`.

**Verification:** `grep -r "src.utils" src/ scripts/` must return zero results.

---

### Task 1.2 — Delete dead tests

**Files to delete (tests for deleted modules):**
```
tests/test_zoo.py              # ModelZoo was deleted
tests/test_restorer.py         # FaceRestorer was deleted
tests/test_matcher.py          # FaceMatcher was deleted
tests/test_quantize.py         # quantize.py not used
tests/test_enrollment.py       # FaceEnroller/clusterer deleted
tests/test_server.py           # Old server (pre-rewrite)
tests/test_e2e.py              # References old pipeline API
```

**Files to KEEP:**
- `tests/test_api.py` — our new API tests (9 passing)
- `tests/test_detector.py` — YOLO detector tests
- `tests/test_embedder.py` — may need review (tests old ONNX embedder?)
- `tests/test_liveness.py` — liveness tests
- `tests/test_orchestrator.py` — may need review (tests old orchestrator?)
- `tests/test_configs.py` — config tests
- `tests/test_imports.py` — import smoke test
- `tests/__init__.py` — package marker

**Verification:** `uv run pytest tests/ -v --ignore=tests/test_api.py -k "not (test_e2e or test_server or test_matcher or test_restorer or test_zoo)"` — remaining tests should pass or be explicitly broken by stale imports (fix in next task).

---

### Task 1.3 — Fix stale imports in surviving tests

**Files to check and fix:**
- `tests/test_imports.py` — may import deleted modules (ModelZoo, FaceRestorer, etc.)
- `tests/test_orchestrator.py` — may reference old FacePipeline API (matcher, restorer)
- `tests/test_embedder.py` — may reference old ONNX embedder (`model_name: resnet100`)
- `tests/test_configs.py` — may reference deleted config files

**Action:** Read each file. If it imports deleted modules, remove those imports/tests. If the entire file is for deleted modules, delete it. If it tests current code, update imports.

---

### Task 1.4 — Delete unused scripts and empty dirs

**Files to delete:**
```
scripts/finetune.py            # Training — not used
scripts/quantize.py            # Quantization — not used
scripts/eval_quick.py          # Our throwaway eval script
scripts/eval_pipeline.py       # Demo eval with dummy scorer
scripts/setup_eval.py          # LFW setup script (keep? it's used for eval)
```

**Decision on setup_eval.py:** Keep it — it's the canonical way to set up LFW evaluation data.

**Empty directory to delete:**
```
src/enrollment/                # Only __init__.py left (clusterer deleted)
```

**Update:** Delete `[project.scripts]` entries in `pyproject.toml` for deleted scripts.

---

## Phase 2: Unify Configs

### Task 2.1 — Merge config files into single `configs/config.yaml`

**Current state:** 5 config files scattered in 2 directories:
```
configs/
  server.yaml                  # host, port, log_level, door
  pipeline/
    pipeline.yaml              # all stage configs (inline)
    detector.yaml              # DUPLICATE — same values in pipeline.yaml
    liveness.yaml              # DUPLICATE — same values in pipeline.yaml  
    embedder.yaml              # STALE — references old ONNX model
```

**Target:** Single file:
```
configs/config.yaml
```

**Content:** Merge server.yaml into pipeline.yaml (which already has all stage configs):
```yaml
# ── Server ──
server:
  host: 0.0.0.0
  port: 8000
  log_level: info

# ── Pipeline ──
pipeline:
  skip_liveness: false
  skip_aligner: false

# ── Stages ──
detector:
  model_name: models/yolo26n_tuned.pt
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: cpu
  max_faces: 5

liveness:
  device: cpu
  threshold: 0.5
  input_size: [128, 128]
  fp16: false

aligner:
  model_id: minchul/cvlface_DFA_mobilenet
  device: cpu

embedder:
  weights_path: weights/arcface-r100-glint360k.pth
  device: cpu

store:
  db_path: data/identity.db
  index_path: data/faiss.index
  threshold: 0.65

# ── Door lock relay (optional) ──
door:
  relay_url: null
  open_timeout_sec: 5
```

**Files to update:**
- `scripts/serve.py` — change Hydra loading to use single config
- `scripts/enroll.py` — change Hydra config path
- `scripts/recognize.py` — change Hydra config path

**Key change in serve.py:** Instead of two `initialize_config_dir` calls:
```python
# Before:
config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
with initialize_config_dir(version_base=None, config_dir=config_dir):
    server_cfg = compose(config_name="server", overrides=args.overrides)
pipeline_dir = os.path.join(config_dir, "pipeline")
with initialize_config_dir(version_base=None, config_dir=pipeline_dir):
    pipeline_cfg = compose(config_name="pipeline")
# Access: server_cfg.host, pipeline_cfg.detector...
```

```python
# After:
config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name="config", overrides=args.overrides)
# Access: cfg.server.host, cfg.detector...
```

**Files to delete:**
```
configs/server.yaml
configs/pipeline/             # entire directory (pipeline.yaml, detector.yaml, liveness.yaml, embedder.yaml)
```

---

## Phase 3: Clean Comments

### Task 3.1 — Clean messy/deprecated comments

**Files to clean (read each, remove stale comments, keep structural ones):**

- `src/pipeline/detector.py` — Remove "Default model is yolo26n.pt (YOLO26-Nano, 80 COCO classes)" comment (we use face-tuned model). Remove COCO reference.
- `src/pipeline/embedder.py` — Remove "ResNet-100 AdaFace" references (it's ArcFace). Remove "model_name: resnet100" mentions in config comments.
- `src/pipeline/liveness.py` — Remove references to ModelZoo in comments. Remove "MobileNetV4-style" etc.
- `src/pipeline/aligner.py` — Clean up the chdir hack comments. Remove "mobile0.25" references.
- `src/pipeline/orchestrator.py` — Remove "restorer" references if any remain.
- `src/server/app.py` — Subagent wrote it clean. Just verify.

**Principle:** Remove comments that merely restate the code. Keep comments that explain WHY (design decisions, workarounds, non-obvious behavior).

---

## Phase 4: API Documentation

### Task 4.1 — Write `docs/api.md`

**Content:** Structured API reference covering:

1. **Overview** — What the API does, architecture diagram (text-based)
2. **Quickstart** — `uv run python scripts/serve.py` → `curl` example
3. **Endpoints** — Full reference for each:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/people` | GET | List enrolled identities |
| `/people/{name}` | DELETE | Remove a person |
| `/enroll` | POST | Enroll a face (single image, liveness on) |
| `/recognize` | POST | Recognize a face |
| `/calibrate` | POST | Recalibrate match threshold |

Each endpoint documented with:
- Request format (form fields, types, defaults)
- Response schema (JSON fields, types, meanings)
- Error responses (400, 403, 422) with conditions
- Example curl command + response

4. **Configuration** — config.yaml reference, all fields
5. **Liveness** — Explanation of skip_liveness param, when to use it
6. **Threshold calibration** — How adaptive thresholding works

---

## Execution Order

```
Phase 1 (safe deletions):  1.1 → 1.2 → 1.3 → 1.4
  (run tests after each)

Phase 2 (config merge):    2.1
  (run full test suite — this is the riskiest change)

Phase 3 (comments):        3.1
  (no tests needed, just lint)

Phase 4 (docs):            4.1
  (no code changes)
```

## Risk Assessment

| Change | Risk | Why |
|---|---|---|
| Delete src/utils/ | **Low** | Nothing imports them |
| Delete dead tests | **Low** | Tests for deleted modules |
| Fix test imports | **Medium** | May need to rewrite surviving tests |
| Merge configs | **Medium** | Changes Hydra loading paths in 3 scripts |
| Clean comments | **Zero** | Cosmetic only |
| Write docs | **Zero** | New file, no code change |
