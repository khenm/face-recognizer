"""Face recognition server entrypoint.

Usage::

    uv run python scripts/serve.py
    uv run python scripts/serve.py server.port=9000
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from face_recognizer.pipeline.orchestrator import FacePipeline
from face_recognizer.pipeline.detector import YOLOFaceDetector
from face_recognizer.pipeline.liveness import LivenessDetector
from face_recognizer.pipeline.aligner import FaceAligner
from face_recognizer.pipeline.embedder import FaceEmbedder
from face_recognizer.db.store import IdentityStore
from face_recognizer.server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the face recognition server.")
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Hydra overrides, e.g. 'server.port=9000'.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=args.overrides)

    # ------------------------------------------------------------------
    # Build pipeline stages
    # ------------------------------------------------------------------
    detector = YOLOFaceDetector(
        **OmegaConf.to_container(cfg.detector, resolve=True),
    )
    liveness = LivenessDetector(
        **OmegaConf.to_container(cfg.liveness, resolve=True),
    )
    aligner = FaceAligner(
        **OmegaConf.to_container(cfg.aligner, resolve=True),
    )
    embedder = FaceEmbedder(
        **OmegaConf.to_container(cfg.embedder, resolve=True),
    )
    store = IdentityStore(
        **OmegaConf.to_container(cfg.store, resolve=True),
    )
    store.load()
    store.calibrate_threshold(margin=0.05)

    # ------------------------------------------------------------------
    # Create pipeline orchestrator
    # ------------------------------------------------------------------
    pipeline = FacePipeline(
        detector=detector,
        liveness=liveness,
        aligner=aligner,
        embedder=embedder,
        store=store,
        skip_liveness=cfg.pipeline.skip_liveness,
        skip_aligner=cfg.pipeline.skip_aligner,
    )

    # ------------------------------------------------------------------
    # Warmup — force lazy model loads so first real request is fast
    # ------------------------------------------------------------------
    import logging as _logging
    import numpy as np

    _logger = _logging.getLogger(__name__)
    _logger.info("Warming up pipeline (loading YOLO + DFA + ArcFace) …")
    _warmup = np.zeros((112, 112, 3), dtype=np.uint8)
    pipeline._detector.detect(_warmup)
    _logger.info("Pipeline warmed up — ready for requests")

    # ------------------------------------------------------------------
    # Create FastAPI app
    # ------------------------------------------------------------------
    app = create_app(pipeline=pipeline, store=store)

    # ------------------------------------------------------------------
    # Start uvicorn
    # ------------------------------------------------------------------
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level,
    )


if __name__ == "__main__":
    main()
