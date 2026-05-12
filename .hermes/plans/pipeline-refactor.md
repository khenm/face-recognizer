# Implementation Plan: Pipeline Refactor — ArcFace + DFA Aligner + SQLite

## Overview

Refactor the face recognition pipeline to remove restoration, add ArcFace-style 5-point
landmark alignment (DFA mobilenet from HF), replace the embedder with ArcFace ResNet-100,
and back identity storage with SQLite + FAISS (row-ID-linked).

## New Pipeline Flow

```
              ┌──────────┐
   Enroll:    │ 1 photo  │
              └────┬─────┘
                   ▼
        Detect → Liveness → Align → Embed → SQLite INSERT + FAISS add
               (YOLO)  (MiniFAS)  (DFA)  (ArcFace)     ↑ same row_id

              ┌──────────┐
Recognize:    │ 1 frame  │
              └────┬─────┘
                   ▼
        Detect → Liveness → Align → Embed → FAISS search → SQLite lookup → name
```

## Architecture Decisions

- **Aligner**: `minchul/cvlface_DFA_mobilenet` via HF `trust_remote_code`. Input: face crop → 5-point landmarks → affine warp to 112×112. Output size matches the ArcFace embedder input.
- **Embedder**: ArcFace IResNet-100 backbone from `weights/arcface-r100-glint360k.pth`. 112×112 input, 512-d output, L2-normalized.
- **Storage**: SQLite for metadata + FAISS `IndexFlatIP` for cosine search. FAISS index positions = SQLite row IDs. Single-vector enrollment (no clustering).
- **Delete**: FAISS flat index can't delete single rows — rebuild from SQLite on delete.

## Pre-Flight Checklist

### Model Weights
| Item | Status | Action |
|------|--------|--------|
| `weights/arcface-r100-glint360k.pth` | ❌ | User provides |
| `models/yolo26n_tuned.pt` | ✅ | Already in repo |
| `models/minifasv2.pth` | ✅ | Already in repo |
| `minchul/cvlface_DFA_mobilenet` | ✅ | HF download on first use |

### Deprecated Files to Remove
| File | Reason |
|------|--------|
| `src/pipeline/restorer.py` | Restoration removed |
| `src/models/codeformer_arch.py` | CodeFormer arch, no longer used |
| `src/models/vqgan_arch.py` | VQGAN arch, no longer used |
| `src/models/zoo.py` | No longer needed (HF + local weights) |
| `src/enrollment/clusterer.py` | Single-vector enrollment, no K-Means |
| `models/codeformer.pth` | CodeFormer weights |
| `models/w600k_r50.onnx` | Old embedder, replaced by ArcFace |
| `configs/pipeline/restorer.yaml` | Restorer config |

---

## Task List

### Phase 1: Foundation — New Embedder + Aligner (build from bottom up)

- [ ] **Task 1**: Write ArcFace embedder (`src/pipeline/embedder.py`)
  - IResNet-100 backbone + ArcFace head (head stripped for inference)
  - Loads from `weights/arcface-r100-glint360k.pth`
  - Input: 112×112 RGB, normalize (mean=0.5, std=0.5)
  - Output: 512-d float32, L2-normalized
  - **Verification**: `uv run python -c "from src.pipeline.embedder import ..."` loads and produces (512,) output from dummy 112×112 tensor

- [ ] **Task 2**: Write DFA aligner (`src/pipeline/aligner.py`)
  - Loads `minchul/cvlface_DFA_mobilenet` via HF `AutoModel.from_pretrained(trust_remote_code=True)`
  - Takes face crop (H×W×3 uint8) → square-pad → 160×160 → model → landmarks → affine warp to 112×112
  - Reference landmarks from `reference_landmark()` in aligner_helper
  - Output: aligned 112×112 uint8 RGB face
  - **Verification**: Load model, align a YOLO face crop from `samples/enroll/pic_1.jpg`, verify output is (112,112,3)

### Checkpoint: Foundation
- [ ] Embedder loads and produces valid embeddings
- [ ] Aligner loads from HF and produces 112×112 aligned crops
- [ ] Both modules importable from `src.pipeline`

### Phase 2: Storage Layer

- [ ] **Task 3**: Write database layer (`src/db/store.py`)
  - SQLite schema: `identities(id, name, embedding BLOB, photo_hash, quality_score, enrolled_at)`
  - FAISS `IndexFlatIP(512)` linked by SQLite row ID
  - Methods: `enroll(name, embedding)`, `search(embedding, threshold) → name|None`, `delete(name)`, `list_names()`, `save()`, `load()`
  - Delete rebuilds FAISS index from SQLite
  - **Verification**: Unit test: enroll 2 people, search one, delete, verify removed

- [ ] **Task 4**: Wire storage into orchestrator
  - Replace `FaceMatcher` with `IdentityStore` in orchestrator
  - Update `FacePipeline` constructor and `process()` to use store
  - **Verification**: Existing orchestrator tests pass (or adapt them)

### Checkpoint: Storage
- [ ] SQLite + FAISS store works: enroll, search, delete
- [ ] Orchestrator uses store instead of matcher

### Phase 3: Config & Pipeline Wiring

- [ ] **Task 5**: Update configs
  - Remove `restorer` from `pipeline.yaml` and delete `configs/pipeline/restorer.yaml`
  - Add `aligner:` section to `pipeline.yaml` (model_id, device)
  - Update `embedder:` → `weights_path: weights/arcface-r100-glint360k.pth`, `model_name: iresnet100`
  - Add `store:` section (`db_path`, `index_path`, `threshold`)
  - Update `detector.yaml`, `liveness.yaml` if needed
  - **Verification**: `hydra compose` loads without errors

- [ ] **Task 6**: Update orchestrator
  - Remove `restorer` parameter, add `aligner` parameter
  - New flow: detect → liveness → align → embed → store.search/store.enroll
  - Remove `_fire_door_relay` (or move to store)
  - Update `PipelineResult`
  - **Verification**: `uv run python scripts/recognize.py samples/footage_04.jpg` runs new pipeline

### Checkpoint: Pipeline
- [ ] `recognize.py` runs with new flow (detect → liveness → align → embed → match)
- [ ] `enroll.py` works: single-image enrollment into SQLite

### Phase 4: Scripts & Cleanup

- [ ] **Task 7**: Rewrite `scripts/enroll.py`
  - Single-image mode (no clustering)
  - Flow: detect → liveness → align → embed → store.enroll
  - Remove `--name` dependency on 3+ images (1 image is fine)
  - **Verification**: `uv run python scripts/enroll.py --name me samples/enroll/pic_1.jpg` stores one vector

- [ ] **Task 8**: Update `scripts/recognize.py`
  - Remove `skip_restorer` flag, add `skip_aligner` flag
  - Update checkpoint saving: add `3_align/` stage output
  - **Verification**: Run on footage_04, see match with `me`

- [ ] **Task 9**: Delete deprecated files
  - Remove all files listed in Pre-Flight Checklist above
  - Remove `configs/pipeline/pipeline.yaml` `skip_restorer` field
  - Update `__init__.py` exports
  - **Verification**: `uv run python -c "import src.pipeline"` succeeds, no import errors

- [ ] **Task 10**: Run existing tests, fix breakage
  - Update/remove tests referencing restorer, matcher, clusterer
  - Add basic tests for aligner and new store
  - **Verification**: `uv run pytest` passes

### Checkpoint: Complete
- [ ] All deprecated files removed
- [ ] `uv run pytest` passes
- [ ] Enroll → Recognize round-trip works end-to-end
- [ ] `uv run python scripts/recognize.py samples/footage_04.jpg` matches "me"

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `weights/arcface-r100-glint360k.pth` missing | 🔴 Blocker | User must provide; plan can't proceed until confirmed |
| HF `cvlface_DFA_mobilenet` trust_remote_code fails | 🟡 Medium | Vendor the wrapper code as fallback |
| IResNet-100 architecture not in timm | 🟡 Medium | Vendor IResNet from InsightFace (common pattern) |
| OpenMP conflict (torch + FAISS) | 🟡 Medium | Already handled via `KMP_DUPLICATE_LIB_OK` |
| DFA model expects BCHW tensor | 🟢 Low | Handled in aligner wrapper |

## Open Questions

- Does `weights/arcface-r100-glint360k.pth` exist yet? (Plan blocks without it)
- What quality_score heuristic? (face detection confidence? liveness score? alignment landmark confidence?)
