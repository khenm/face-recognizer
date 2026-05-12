"""FastAPI server for face recognition access control.

Endpoints
---------
GET  /health          — Liveness probe
GET  /people          — List enrolled people
DELETE /people/{name} — Remove a person
POST /recognize       — Recognize a face in an uploaded image
POST /enroll          — Enroll a new person from a single image
POST /calibrate       — Recompute match threshold from enrolled population
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from face_recognizer.pipeline.orchestrator import FacePipeline
from face_recognizer.db.store import IdentityStore
from face_recognizer.server.schemas import (
    CalibrateResponse,
    DeleteResponse,
    EnrollResponse,
    HealthResponse,
    PeopleResponse,
    RecognizeResponse,
)

logger = logging.getLogger(__name__)


def create_app(pipeline: FacePipeline, store: IdentityStore) -> FastAPI:
    """Factory: create the FastAPI app with injected dependencies.

    Parameters
    ----------
    pipeline : FacePipeline
        Initialised pipeline orchestrator (detector, liveness, aligner,
        embedder, and store already wired in).
    store : IdentityStore
        SQLite + FAISS identity store (the same instance wired into
        *pipeline*).
    """
    app = FastAPI(title="Face Recognizer", version="0.1.0")

    # ------------------------------------------------------------------
    # CORS – required for ESP32-CAM and browser-based clients
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Store instances on app.state for endpoint access
    # ------------------------------------------------------------------
    app.state.pipeline = pipeline
    app.state.store = store

    # ==================================================================
    # Endpoints
    # ==================================================================

    # ---- Health -------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Liveness probe — always returns ``{"status": "ok"}``."""
        return HealthResponse(status="ok")

    # ---- People -------------------------------------------------------

    @app.get("/people", response_model=PeopleResponse)
    async def list_people():
        """List every enrolled person name, the current match threshold,
        and the total number of stored embedding vectors."""
        names = app.state.store.list_names()
        # _faiss_to_row length mirrors the FAISS index total — no public
        # accessor yet, so we reach into the store internals.  Falls back
        # to 0 if the index hasn't been built.
        vector_count = (
            len(app.state.store._faiss_to_row)
            if app.state.store._faiss_to_row
            else 0
        )
        return PeopleResponse(
            names=names,
            threshold=app.state.store.threshold,
            vector_count=vector_count,
        )

    @app.delete("/people/{name}", response_model=DeleteResponse)
    async def delete_person(name: str):
        """Remove every embedding row belonging to *name* and rebuild the
        FAISS index."""
        app.state.store.delete(name)
        return DeleteResponse(removed=name)

    # ---- Recognize ----------------------------------------------------

    @app.post("/recognize", response_model=RecognizeResponse)
    async def recognize(
        file: UploadFile = File(...),
        skip_liveness: bool = Form(False),
    ):
        """Recognize a face in an uploaded image.

        Parameters
        ----------
        file : UploadFile
            JPEG or PNG image containing a face.
        skip_liveness : bool
            If ``True``, bypass the anti-spoofing stage (default ``False``).
        """
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # Decode image bytes → BGR → RGB
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = app.state.pipeline.process(image, skip_liveness=skip_liveness)

        # Convert face_bbox from tuple to list for JSON serialisation
        bbox = list(result.face_bbox) if result.face_bbox is not None else None

        return RecognizeResponse(
            matched=result.matched,
            name=result.name,
            confidence=result.confidence,
            is_real=result.is_real,
            face_bbox=bbox,
            per_stage_ms={k: round(v, 2) for k, v in result.per_stage_ms.items()},
            total_ms=round(result.total_ms, 2),
        )

    # ---- Enroll -------------------------------------------------------

    @app.post("/enroll", response_model=EnrollResponse)
    async def enroll(
        name: str = Form(...),
        file: UploadFile = File(...),
        skip_liveness: bool = Form(False),
    ):
        """Enroll a new person from a single face image.

        Parameters
        ----------
        name : str
            Person identifier (e.g. ``"Alice"``).
        file : UploadFile
            JPEG or PNG image containing a face.
        skip_liveness : bool
            If ``True``, bypass the anti-spoofing stage (default ``False``).
        """
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # Decode image bytes → BGR → RGB
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # Pre-flight check — detect face + liveness before enrolling.
        # FacePipeline.enroll() returns ``None`` for both no-face and spoof
        # cases, so we use process() first to produce distinct HTTP errors.
        # ------------------------------------------------------------------
        pre_flight = app.state.pipeline.process(
            image, skip_liveness=skip_liveness
        )

        if pre_flight.face_bbox is None:
            raise HTTPException(status_code=400, detail="No face detected")

        if pre_flight.is_real is False:
            raise HTTPException(
                status_code=403,
                detail="Spoof detected — enrollment rejected",
            )

        # ---- Enroll -------------------------------------------------------
        row_id = app.state.pipeline.enroll(
            image, name, skip_liveness=skip_liveness,
        )

        if row_id is None:
            # Should not be reachable after a clean pre-flight, but guard
            # against race conditions or transient failures.
            raise HTTPException(
                status_code=500, detail="Enrollment failed unexpectedly"
            )

        # ---- Recalibrate match threshold ----------------------------------
        threshold = app.state.store.calibrate_threshold()

        # Detection quality: the best detection confidence from pre_flight
        # serves as a proxy for image quality.
        quality: float = 0.0
        if pre_flight.face_bbox is not None:
            # Re-run detection to get the raw confidence score (not exposed
            # on PipelineResult).  This is a cheap operation.
            detections = app.state.pipeline._detector.detect(image)
            if detections:
                quality = float(detections[0].confidence)

        # NOTE: liveness_score is ``None`` because FacePipeline.enroll()
        # does not expose LivenessResult.score in its return value.  When
        # liveness is *not* skipped and enrollment succeeds, we know the
        # face passed liveness, but the exact score is unavailable via
        # the public API.
        liveness_score: float | None = None

        return EnrollResponse(
            name=name,
            row_id=row_id,
            liveness_score=liveness_score,
            quality=quality,
            threshold=threshold,
        )

    # ---- Calibrate -----------------------------------------------------

    @app.post("/calibrate", response_model=CalibrateResponse)
    async def calibrate(margin: float = Form(0.05)):
        """Recalibrate the match threshold from the currently enrolled
        population.

        Parameters
        ----------
        margin : float
            Safety margin added to the maximum cross-identity similarity
            (default ``0.05``).
        """
        threshold = app.state.store.calibrate_threshold(margin=margin)

        # Best-effort supplementary fields.  calibrate_threshold() only
        # returns the threshold float; we reconstruct the rest.
        max_cross_id = max(threshold - margin, 0.0) if threshold > 0.0 else 0.0
        identities = len(app.state.store.list_names())

        vector_count = (
            len(app.state.store._faiss_to_row)
            if app.state.store._faiss_to_row
            else 0
        )

        return CalibrateResponse(
            threshold=round(threshold, 6),
            max_cross_id=round(max_cross_id, 6),
            margin=margin,
            identities=identities,
            vectors=vector_count,
        )

    return app
