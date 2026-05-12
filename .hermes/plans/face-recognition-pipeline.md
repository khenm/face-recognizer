# Implementation Plan: Face Recognition Access Control Pipeline

## Overview

Transform the existing PyTorch template into a production face recognition system. The pipeline: ESP32-CAM capture → YOLO face detection → MiniFASNet liveness check → CodeFormer restoration → ResNet-100 AdaFace embedding → FAISS match → door unlock signal. Plus an enrollment flow using K-Means clustering of 15 frames into 3 pose anchors.

## Architecture Decisions

- **Pre-trained weights first**: YOLO26-Nano, MiniFASNet, CodeFormer, and ResNet-100 all start from published checkpoints. Training scripts added later for fine-tuning on custom datasets (ChokePoint for YOLO, CASIA-SURF for anti-spoofing).
- **FAISS as pure Python wrapper**: Use `faiss-cpu` (or `faiss-gpu`) library directly — no training needed. IndexFile saves to disk for persistence.
- **Pipeline class pattern**: Each stage is a callable `nn.Module` + wrapper. A `FacePipeline` orchestrator chains them with early-exit (spoof → stop).
- **FastAPI server**: Lightweight HTTP API for ESP32-CAM to POST images, plus `/enroll` for adding new faces.
- **Keep existing training infra**: Retain Hydra configs, DDP/FSDP trainer for future model fine-tuning — just add new model/dataset/loss configs.

## Task List

### Phase 1: Project Foundation

- [ ] **Task 1: Rename project and update dependencies**
  - Update `pyproject.toml`: name → `face-recognizer`, add deps (faiss-cpu, opencv-python-headless, onnxruntime, fastapi, uvicorn, scikit-learn, pillow)
  - Move to `uv` workspace if not already (check pyproject.toml `[tool.uv.workspace]`)
  - Update `README.md` with project description
  - **AC**: `uv sync` succeeds, imports work
  - **Files**: `pyproject.toml`, `README.md`

- [ ] **Task 2: Create directory structure and config schema**
  - `src/pipeline/` — all pipeline stages live here
  - `src/server/` — FastAPI app
  - `src/enrollment/` — K-Means enrollment logic
  - `configs/pipeline/` — Hydra configs for each stage
  - `configs/server.yaml` — server settings (threshold, host, port)
  - `scripts/serve.py` — server entrypoint
  - `scripts/enroll.py` — enrollment CLI
  - `scripts/recognize.py` — single-image recognition CLI
  - **AC**: Directory structure exists, configs parse with Hydra
  - **Files**: New directories + config files

- [ ] **Task 3: Model weight downloader / zoo**
  - Create `src/models/zoo.py` with a `ModelZoo` class that downloads weights from known URLs
  - Support: YOLO26-Nano (ultralytics), MiniFASNet, CodeFormer, ResNet-100 AdaFace
  - Cache to `~/.cache/face-recognizer/` by default
  - **AC**: `python -c "from src.models.zoo import ModelZoo; zoo = ModelZoo(); zoo.download('yolo_nano')"` works
  - **Files**: `src/models/zoo.py`

### Checkpoint: Foundation
- [ ] `uv run python -c "import src"` succeeds
- [ ] Configs parse without errors
- [ ] Model zoo downloads at least 1 model

### Phase 2: Detection & Anti-Spoofing

- [ ] **Task 4: YOLO face detection module**
  - Create `src/pipeline/detector.py` with `YOLOFaceDetector` class
  - Wraps ultralytics YOLO (yolov8n-face or custom YOLO26-Nano weights)
  - `detect(image: np.ndarray) -> List[BBox]` returns face crops + coordinates
  - Handle no-face-detected gracefully (return empty list)
  - **AC**: Detects faces in test image, returns cropped face tensors
  - **Files**: `src/pipeline/detector.py`, `configs/pipeline/detector.yaml`

- [ ] **Task 5: MiniFASNet anti-spoofing module**
  - Create `src/pipeline/liveness.py` with `LivenessDetector` class
  - Load MiniFASNet (from Silent-Face-Anti-Spoofing repo) — ONNX runtime or PyTorch
  - `check(image: np.ndarray) -> Tuple[bool, float]` returns (is_real, confidence)
  - Handle spoof detection → pipeline early-exit signal
  - **AC**: Classifies real vs spoof faces on test pairs
  - **Files**: `src/pipeline/liveness.py`, `configs/pipeline/liveness.yaml`

### Checkpoint: Detection + Liveness
- [ ] YOLO detects and crops faces
- [ ] Liveness returns real/spoof classification
- [ ] Both work end-to-end on a single image

### Phase 3: Restoration & Recognition

- [ ] **Task 6: CodeFormer face restoration module**
  - Create `src/pipeline/restorer.py` with `FaceRestorer` class
  - Wraps CodeFormer inference (pre-trained weights from sczhou/CodeFormer)
  - `restore(image: np.ndarray) -> np.ndarray` returns cleaned face
  - Pass-through mode when restoration is disabled (config flag)
  - **AC**: Noisy face → clean face, identity preserved
  - **Files**: `src/pipeline/restorer.py`, `configs/pipeline/restorer.yaml`

- [ ] **Task 7: ResNet-100 AdaFace embedding module**
  - Create `src/pipeline/embedder.py` with `FaceEmbedder` class
  - ResNet-100 backbone (timm or torchvision) + AdaFace margin loss head stripped
  - `embed(image: np.ndarray) -> np.ndarray` returns 512-dim L2-normalized vector
  - Batch inference support for enrollment
  - **AC**: Two images of same person → cosine similarity > 0.65; different people → < 0.4
  - **Files**: `src/pipeline/embedder.py`, `configs/pipeline/embedder.yaml`

- [ ] **Task 8: FAISS vector database / matcher**
  - Create `src/pipeline/matcher.py` with `FaceMatcher` class
  - Wraps `faiss.IndexFlatIP` (inner product = cosine similarity for normalized vectors)
  - `add(name: str, vectors: List[np.ndarray])` — add person with pose anchors
  - `search(vector: np.ndarray, k: int = 1) -> List[Tuple[str, float]]`
  - `save(path)` / `load(path)` for index persistence
  - Threshold-based decision: similarity > config threshold → match
  - **AC**: Save/load roundtrip, correct match on known face, unknown below threshold
  - **Files**: `src/pipeline/matcher.py`, `configs/pipeline/matcher.yaml`

### Checkpoint: Full Pipeline (no server)
- [ ] `scripts/recognize.py image.jpg` outputs name + confidence or "unknown"
- [ ] Anti-spoofing correctly rejects spoof images early

### Phase 4: Pipeline Orchestrator

- [ ] **Task 9: Pipeline orchestrator**
  - Create `src/pipeline/orchestrator.py` with `FacePipeline` class
  - Chains: Detector → Liveness (early exit if spoof) → Restorer → Embedder → Matcher
  - `process(image: np.ndarray) -> PipelineResult` with fields: `name`, `confidence`, `is_real`, `face_bbox`, `processing_time_ms`
  - Per-stage timing for debugging
  - Hydra-instantiable with `_target_: src.pipeline.orchestrator.FacePipeline`
  - **AC**: Single call processes image end-to-end, returns structured result
  - **Files**: `src/pipeline/orchestrator.py`, `configs/pipeline/pipeline.yaml`

### Checkpoint: Pipeline Complete
- [ ] `scripts/recognize.py image.jpg` full pipeline works
- [ ] Early exit on spoof works (liveness says fake → stops before restorer)

### Phase 5: Server & Enrollment

- [ ] **Task 10: FastAPI server**
  - Create `src/server/app.py` with FastAPI endpoints:
    - `POST /recognize` — multipart image upload → pipeline result JSON
    - `POST /enroll` — multipart (15 images) + name → enrollment result
    - `GET /health` — liveness probe
    - `GET /people` — list enrolled people
  - CORS for ESP32-CAM
  - Door relay webhook config (POST to relay IP on match)
  - **AC**: `uv run uvicorn src.server.app:app` serves all endpoints
  - **Files**: `src/server/app.py`, `src/server/__init__.py`, `scripts/serve.py`, `configs/server.yaml`

- [ ] **Task 11: Enrollment with K-Means clustering**
  - Create `src/enrollment/clusterer.py` with `FaceEnroller` class
  - Takes 15 face embeddings from different angles
  - K-Means (k=3) → 3 pose anchors (Left, Front, Right)
  - Stores anchors in FAISS index with person name
  - Validation: anchors should have good coverage (inter-cluster distance check)
  - **AC**: `scripts/enroll.py --name "Alice" images/` stores 3 vectors, subsequent recognition matches Alice
  - **Files**: `src/enrollment/clusterer.py`, `src/enrollment/__init__.py`, `scripts/enroll.py`

### Checkpoint: Server + Enrollment
- [ ] Enroll a person via API → recognized correctly afterward
- [ ] Door relay fires on match

### Phase 6: ESP32 Integration & Polish

- [ ] **Task 12: ESP32-CAM capture script**
  - Create `scripts/esp32_capture.py` — polls ESP32-CAM HTTP stream, captures on motion
  - POSTs to `/recognize` endpoint
  - Configurable: ESP32 IP, motion threshold, capture interval
  - **AC**: Script runs, captures from ESP32, POSTs to server
  - **Files**: `scripts/esp32_capture.py`

- [ ] **Task 13: End-to-end tests & documentation**
  - Test fixtures: sample real/spoof face images, known/unknown identities
  - `tests/test_pipeline.py` — full pipeline integration test
  - `tests/test_server.py` — FastAPI endpoint tests
  - `tests/test_enrollment.py` — enrollment + recognition roundtrip
  - Update `README.md` with setup instructions, architecture diagram, usage
  - **AC**: `uv run pytest` passes all tests
  - **Files**: `tests/*`, `README.md`

### Checkpoint: Complete
- [ ] All tests pass
- [ ] README is comprehensive
- [ ] ESP32 → Server → Door flow documented

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CodeFormer weights are large (~400MB) and slow on CPU | High | Make restoration optional/configurable; use ONNX runtime for speed |
| MiniFASNet ONNX conversion may fail | Medium | Fall back to PyTorch inference; pre-convert and cache ONNX model |
| YOLO26-Nano needs ChokePoint fine-tuning for overhead angle | Medium | Start with YOLOv8n-face (general faces); add fine-tuning script as Phase 2b |
| FAISS index persistence across server restarts | Low | Auto-load index on startup; auto-save on enrollment |
| ESP32-CAM Wi-Fi latency | Low | Set generous timeouts; add retry logic |

## Open Questions

- Threshold value for cosine similarity? (Default: 0.65, configurable)
- Door relay protocol? (HTTP webhook assumed; adjust if hardware-specific)
- How many people in the database? (FAISS Flat index is fine for <10K; IVF for larger)
