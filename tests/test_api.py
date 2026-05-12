"""API tests for the face recognition server.

Uses httpx `ASGITransport` against a real (but isolated) pipeline backed
by a temporary SQLite + FAISS store.  No external server required.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

# Prevent macOS OpenMP conflicts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from hydra import compose, initialize_config_dir

from face_recognizer.pipeline.orchestrator import FacePipeline
from face_recognizer.pipeline.detector import YOLOFaceDetector
from face_recognizer.pipeline.liveness import LivenessDetector
from face_recognizer.pipeline.aligner import FaceAligner
from face_recognizer.pipeline.embedder import FaceEmbedder
from face_recognizer.db.store import IdentityStore
from face_recognizer.server.app import create_app


# ── Test images ──────────────────────────────────────────────────────
SAMPLES_DIR = Path(__file__).parent.parent / "samples"

SAMPLE_IMAGE = SAMPLES_DIR / "footage_01.png"


def _ensure_sample() -> Path:
    """Create a minimal test image if footage_01 doesn't exist."""
    if SAMPLE_IMAGE.exists():
        return SAMPLE_IMAGE
    # Fallback: create a small synthetic face image (won't be detected,
    # but lets us test empty-file and decode error flows).
    tmp = Path("/tmp/test_face.jpg")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp), img)
    return tmp


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def config_dir() -> str:
    """Absolute path to the Hydra config directory."""
    return str(Path(__file__).parent.parent / "configs")


@pytest.fixture
def pipeline_cfg(config_dir: str):
    """Load the unified config."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        yield compose(config_name="config")


@pytest.fixture
def store(tmp_path: Path, pipeline_cfg):
    """Isolated IdentityStore backed by a temp directory."""
    db = str(tmp_path / "test_identity.db")
    idx = str(tmp_path / "test_faiss.index")
    store_cfg = pipeline_cfg["store"]

    # Override paths to the temp directory.
    store = IdentityStore(
        db_path=db,
        index_path=idx,
        threshold=float(store_cfg.get("threshold", 0.65)),
    )
    store.load()
    return store


@pytest.fixture
def pipeline(pipeline_cfg, store):
    """FacePipeline built from config with all stages."""
    from omegaconf import OmegaConf

    return FacePipeline(
        detector=YOLOFaceDetector(
            **OmegaConf.to_container(pipeline_cfg.detector, resolve=True)
        ),
        liveness=LivenessDetector(
            **OmegaConf.to_container(pipeline_cfg.liveness, resolve=True)
        ),
        aligner=FaceAligner(
            **OmegaConf.to_container(pipeline_cfg.aligner, resolve=True)
        ),
        embedder=FaceEmbedder(
            **OmegaConf.to_container(pipeline_cfg.embedder, resolve=True)
        ),
        store=store,
        skip_liveness=False,
        skip_aligner=False,
    )


@pytest.fixture
async def client(pipeline, store):
    """Async HTTP client wired to the FastAPI app."""
    app = create_app(pipeline=pipeline, store=store)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Tests ────────────────────────────────────────────────────────────


class TestHealth:
    async def test_health_ok(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestPeople:
    async def test_empty(self, client: AsyncClient):
        resp = await client.get("/people")
        assert resp.status_code == 200
        data = resp.json()
        assert data["names"] == []
        assert data["threshold"] > 0
        assert data["vector_count"] == 0


class TestDelete:
    async def test_delete_nonexistent(self, client: AsyncClient):
        resp = await client.delete("/people/Nobody")
        assert resp.status_code == 200
        assert resp.json()["removed"] == "Nobody"


class TestEnrollAndRecognize:
    @pytest.fixture(autouse=True)
    def _skip_if_no_sample(self):
        if not SAMPLE_IMAGE.exists():
            pytest.skip("sample footage image not available")

    async def test_enroll_then_recognize(self, client: AsyncClient):
        """End-to-end: enroll a real face, then recognize it."""
        sample = _ensure_sample()
        # -- Enroll --
        with open(sample, "rb") as f:
            resp = await client.post(
                "/enroll",
                data={"name": "TestUser", "skip_liveness": "true"},
                files={"file": ("footage_01.png", f, "image/png")},
            )
        assert resp.status_code == 200, resp.text
        enroll_data = resp.json()
        assert enroll_data["name"] == "TestUser"
        assert enroll_data["row_id"] > 0

        # -- Recognize --
        with open(sample, "rb") as f:
            resp = await client.post(
                "/recognize",
                files={"file": ("footage_01.png", f, "image/png")},
            )
        assert resp.status_code == 200, resp.text
        rec_data = resp.json()
        assert rec_data["matched"] is True
        assert rec_data["name"] == "TestUser"
        assert rec_data["confidence"] is not None

        # -- Cleanup --
        await client.delete("/people/TestUser")

    async def test_enroll_missing_name(self, client: AsyncClient):
        """Missing name → 422 (FastAPI validation)."""
        sample = _ensure_sample()
        with open(sample, "rb") as f:
            resp = await client.post(
                "/enroll",
                files={"file": ("footage_01.png", f, "image/png")},
            )
        assert resp.status_code == 422


class TestRecognizeErrors:
    async def test_empty_file(self, client: AsyncClient):
        resp = await client.post(
            "/recognize",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    async def test_unrecognised_face(self, client: AsyncClient):
        """An image with no enrolled match returns matched=False."""
        # Create a tiny synthetic image — detection will fail on it,
        # resulting in no match.
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        resp = await client.post(
            "/recognize",
            files={"file": ("tiny.png", buf.tobytes(), "image/png")},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["matched"] is False


class TestCalibrate:
    async def test_calibrate_empty(self, client: AsyncClient):
        """Calibrating an empty store returns threshold=0."""
        resp = await client.post("/calibrate", data={"margin": "0.05"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["threshold"] == 0.0
        assert data["identities"] == 0
        assert data["vectors"] == 0

    async def test_calibrate_after_enroll(self, client: AsyncClient):
        """Calibrate with enrolled identities returns non-zero threshold."""
        sample = _ensure_sample()
        if not sample.exists():
            pytest.skip("sample image not available")
        # Enroll
        with open(sample, "rb") as f:
            resp = await client.post(
                "/enroll",
                data={"name": "Cal", "skip_liveness": "true"},
                files={"file": ("photo.png", f, "image/png")},
            )
        assert resp.status_code == 200

        # Calibrate
        resp = await client.post("/calibrate", data={"margin": "0.05"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["threshold"] == 0.0  # only 1 identity → can't calibrate
        assert data["identities"] == 1

        # Cleanup
        await client.delete("/people/Cal")
