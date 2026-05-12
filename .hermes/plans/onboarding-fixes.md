# Onboarding Fixes — Implementation Plan

> **Goal:** Make the repo usable within 5 minutes of cloning. Zero tribal knowledge required.

**Context:** The codebase is production-ready (9/9 tests, unified config, clean structure) but the README is 3 generations stale and critical setup steps are undocumented.

---

## Task 1: Rewrite README.md

**File:** `README.md` (complete rewrite)

Replace the stale README with a 5-minute quickstart. Structure:

```
# Face Recognizer
One-liner description

## Quickstart (5 minutes)
1. Prerequisites (uv, Python 3.10+)
2. Clone + install
3. Download models (one command)
4. Start server
5. Enroll + recognize (curl examples)

## API (brief — link to docs/api.md for full ref)
- /health, /enroll, /recognize, /people, /calibrate
- One curl example each

## Architecture
Current pipeline diagram (no deleted components)

## Configuration
- One config file: configs/config.yaml
- Key overrides

## Requirements
- Python 3.10+
- uv
- Models (downloaded via scripts/download_models.py)
```

---

## Task 2: Add model download script

**File:** `scripts/download_models.py` (new)

Downloads all required model weights to their expected locations:

| Model | Source | Destination | Size |
|---|---|---|---|
| YOLO face detector | Kaggle / local | `models/yolo26n_tuned.pt` | ~6 MB |
| MiniFASNet liveness | Local / download | `models/minifasv2.pth` | ~7 MB |
| ArcFace IResNet-100 | Vec2Face HF repo | `weights/arcface-r100-glint360k.pth` | 249 MB |
| DFA aligner | HuggingFace Hub | `~/.cache/huggingface/` (auto) | ~5 MB |

Logic:
```python
# 1. YOLO: if not at models/yolo26n_tuned.pt, print warning (user must provide)
# 2. MiniFASNet: if not at models/minifasv2.pth, print warning
# 3. ArcFace: if not at weights/arcface-r100-glint360k.pth, download from HF
# 4. DFA: pre-download via snapshot_download("minchul/cvlface_DFA_mobilenet")
```

Usage: `uv run python scripts/download_models.py`

Design decision: YOLO and MiniFASNet weights can't be auto-downloaded (Kaggle login wall, custom training). Print clear download URLs and expected paths. ArcFace and DFA are auto-downloadable from HF.

---

## Task 3: Add `.gitignore`

**File:** `.gitignore` (new or update if exists)

```
# Model weights (large binaries)
models/*.pt
models/*.pth
models/*.onnx
weights/

# Database files
data/*.db
data/*.index

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Results (eval output)
results/

# macOS
.DS_Store

# IDE
.idea/
.vscode/
```

---

## Task 4: Add `data/.gitkeep`

**File:** `data/.gitkeep` (new, empty)

So the `data/` directory exists in the repo. `IdentityStore.load()` creates DB files on first run, but having the directory pre-existing avoids confusion.

---

## Task 5: Update `pyproject.toml` classifiers

**File:** `pyproject.toml`

Add Python version and platform classifiers so the error is friendlier if someone uses the wrong setup:

```toml
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
]
```

Also add a `[project.urls]` section pointing to the docs.

---

## Task 6: Add macOS troubleshooting to README

Add a section:

```markdown
## Troubleshooting

### macOS: `KMP_DUPLICATE_LIB_OK` errors

If you see OpenMP symbol conflicts on macOS:

    export KMP_DUPLICATE_LIB_OK=TRUE

This is handled automatically by the scripts but may be needed for custom
integrations.
```

---

## Verification

After all changes:
1. Clone a fresh copy of the repo
2. `uv sync`
3. `uv run python scripts/download_models.py` — succeeds, all models present
4. `uv run python scripts/serve.py &` — starts without errors
5. `curl http://localhost:8000/health` — returns `{"status":"ok"}`
6. `uv run pytest tests/ -q` — 30 passed
7. Read README top-to-bottom — every command works

---

## Files Summary

| Action | File |
|---|---|
| Rewrite | `README.md` |
| Create | `scripts/download_models.py` |
| Create/Update | `.gitignore` |
| Create | `data/.gitkeep` |
| Update | `pyproject.toml` |

## Risk

None — all changes are additive (new files or doc rewrites). Zero code changes.
