"""Face recognition pipeline orchestrator.

Chains the four pipeline stages — detection, liveness, alignment,
embedding — and delegates identity storage to :class:`IdentityStore`.

Usage::

    pipeline = FacePipeline(detector=..., liveness=..., aligner=..., embedder=..., store=...)
    result = pipeline.process(image)  # PipelineResult (recognition)
    row_id  = pipeline.enroll(image, name="Alice")  # enrollment
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full pipeline ``process()`` call.

    Attributes
    ----------
    name : str or None
        Matched person name, or ``None`` if no match.
    confidence : float or None
        Cosine similarity score of the match, or ``None``.
    is_real : bool or None
        Liveness verdict (``None`` if liveness was skipped).
    matched : bool
        ``True`` if a person was identified above the store threshold.
    face_bbox : tuple or None
        ``(x1, y1, x2, y2)`` of the matched face, or ``None``.
    per_stage_ms : dict[str, float]
        Wall-clock time in milliseconds for each executed stage.
    total_ms : float
        Total wall-clock time for the full pipeline.
    """

    name: Optional[str] = None
    confidence: Optional[float] = None
    is_real: Optional[bool] = None
    matched: bool = False
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    per_stage_ms: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0


class FacePipeline:
    """Orchestrates the face recognition pipeline.

    Parameters
    ----------
    detector : YOLOFaceDetector
    liveness : LivenessDetector
    aligner : FaceAligner
    embedder : FaceEmbedder
    store : IdentityStore
    skip_liveness : bool
        If ``True``, bypass the liveness stage entirely.
    skip_aligner : bool
        If ``True``, bypass the alignment stage.
    """

    def __init__(
        self,
        *,
        detector,
        liveness,
        aligner,
        embedder,
        store,
        skip_liveness: bool = False,
        skip_aligner: bool = False,
    ) -> None:
        self._detector = detector
        self._liveness = liveness
        self._aligner = aligner
        self._embedder = embedder
        self._store = store
        self.skip_liveness = skip_liveness
        self.skip_aligner = skip_aligner

    # ------------------------------------------------------------------
    # Public API — Recognition
    # ------------------------------------------------------------------

    def process(
        self,
        image: np.ndarray,
        *,
        skip_liveness: Optional[bool] = None,
    ) -> PipelineResult:
        """Run recognition on *image*.

        Parameters
        ----------
        image : np.ndarray
            Input image ``(H, W, 3)`` uint8 RGB.
        skip_liveness : bool or None
            Per-call override for ``self.skip_liveness``.  ``None`` means
            use the instance default.

        Returns
        -------
        PipelineResult
        """
        t_start = time.perf_counter()
        per_stage: Dict[str, float] = {}
        _skip_liveness = self.skip_liveness if skip_liveness is None else skip_liveness

        # ---- 1. Detection ---------------------------------------------------
        t0 = time.perf_counter()
        detections = self._detector.detect(image)
        per_stage["detect"] = (time.perf_counter() - t0) * 1000

        if not detections:
            return PipelineResult(
                matched=False,
                per_stage_ms=per_stage,
                total_ms=(time.perf_counter() - t_start) * 1000,
            )

        # Process faces in confidence order until we find a match
        last_is_real: Optional[bool] = None
        last_bbox: Optional[Tuple[float, float, float, float]] = None
        for det in detections:
            face_crop = det.crop
            last_bbox = det.bbox

            # ---- 2. Liveness ------------------------------------------------
            is_real: Optional[bool] = None
            if not _skip_liveness:
                t0 = time.perf_counter()
                liveness_result = self._liveness.check(face_crop)
                per_stage["liveness"] = (time.perf_counter() - t0) * 1000

                is_real = liveness_result.is_real
                last_is_real = is_real
                if not is_real:
                    continue
            else:
                is_real = None

            # ---- 3. Alignment -----------------------------------------------
            if not self.skip_aligner:
                t0 = time.perf_counter()
                try:
                    face_crop = self._aligner.align(face_crop)
                except RuntimeError:
                    logger.warning("Alignment failed for face — skipping")
                    continue
                per_stage["align"] = (time.perf_counter() - t0) * 1000

            # ---- 4. Embedding -----------------------------------------------
            t0 = time.perf_counter()
            embedding = self._embedder.embed(face_crop)
            per_stage["embed"] = (time.perf_counter() - t0) * 1000

            # ---- 5. Search --------------------------------------------------
            t0 = time.perf_counter()
            matches = self._store.search(embedding, k=1)
            per_stage["search"] = (time.perf_counter() - t0) * 1000

            if matches:
                name, score = matches[0]
                return PipelineResult(
                    name=name,
                    confidence=score,
                    is_real=is_real,
                    matched=True,
                    face_bbox=det.bbox,
                    per_stage_ms=dict(per_stage),
                    total_ms=(time.perf_counter() - t_start) * 1000,
                )

        # No match found across all faces
        return PipelineResult(
            matched=False,
            is_real=last_is_real,
            face_bbox=last_bbox,
            per_stage_ms=per_stage,
            total_ms=(time.perf_counter() - t_start) * 1000,
        )

    # ------------------------------------------------------------------
    # Public API — Enrollment
    # ------------------------------------------------------------------

    def enroll(
        self,
        image: np.ndarray,
        name: str,
        *,
        photo_hash: Optional[str] = None,
        skip_liveness: Optional[bool] = None,
    ) -> Optional[int]:
        """Enroll a person from a single image.

        Detects the best face, runs liveness + alignment + embedding,
        then stores the embedding in the identity database.

        Parameters
        ----------
        image : np.ndarray
            Input image ``(H, W, 3)`` uint8 RGB.
        name : str
            Person identifier.
        photo_hash : str or None
            Optional SHA256 hash of the source image.
        skip_liveness : bool or None
            Per-call override for ``self.skip_liveness``.  ``None`` means
            use the instance default.

        Returns
        -------
        int or None
            SQLite row ID of the enrolled identity, or ``None`` if no
            face was detected.
        """
        _skip_liveness = self.skip_liveness if skip_liveness is None else skip_liveness

        # ---- Detection -------------------------------------------------------
        detections = self._detector.detect(image)
        if not detections:
            logger.warning("Enrollment failed: no face detected for %r", name)
            return None

        # Use highest-confidence face
        det = detections[0]
        face_crop = det.crop
        quality = float(det.confidence)

        # ---- Liveness --------------------------------------------------------
        if not _skip_liveness:
            liveness_result = self._liveness.check(face_crop)
            if not liveness_result.is_real:
                logger.warning(
                    "Enrollment failed: spoof detected for %r (score=%.3f)",
                    name, liveness_result.score,
                )
                return None
            quality = min(quality, float(liveness_result.score))

        # ---- Alignment -------------------------------------------------------
        if not self.skip_aligner:
            try:
                face_crop, align_quality = self._aligner.align(
                    face_crop, return_quality=True,
                )
                quality = min(quality, align_quality)
            except RuntimeError:
                logger.warning("Enrollment failed: alignment failed for %r", name)
                return None

        # ---- Embedding -------------------------------------------------------
        embedding = self._embedder.embed(face_crop)

        # ---- Store -----------------------------------------------------------
        row_id = self._store.enroll(
            name=name,
            embedding=embedding,
            photo_hash=photo_hash,
            quality_score=quality,
        )

        logger.info("Enrolled %r (row_id=%d, quality=%.3f)", name, row_id, quality)
        return row_id
