"""YOLO face detection module.

Wraps ultralytics YOLO for face detection and cropping.  Detects faces,
filters by confidence, crops regions, and returns :class:`FaceDetection`
objects.  Uses a WIDER Face-tuned model by default.

Usage::

    detector = YOLOFaceDetector(model_name="models/yolo26n_tuned.pt", device="cpu")
    detections = detector.detect(image)  # List[FaceDetection]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetection:
    """A single detected face with bounding box and cropped pixels.

    Attributes
    ----------
    bbox : tuple of (x1, y1, x2, y2)
        Pixel coordinates (float), top-left inclusive, bottom-right exclusive.
    confidence : float
        Detection confidence in [0, 1].
    crop : np.ndarray
        Cropped face region (H, W, 3) uint8, a **copy** of the source pixels.
    """

    bbox: Tuple[float, float, float, float]
    confidence: float
    crop: np.ndarray


class YOLOFaceDetector:
    """Face detector powered by an ultralytics YOLO model.

    Parameters
    ----------
    model_name : str
        Model path passed to ``ultralytics.YOLO``.
        Default ``"models/yolo26n_tuned.pt"``.
    confidence_threshold : float
        Minimum confidence for a detection to be kept.
    iou_threshold : float
        NMS IoU threshold (passed to YOLO at predict time).
    device : str
        Torch device string (``"cpu"``, ``"cuda:0"``, …).
    max_faces : int
        Maximum number of face detections to return.  Detections are sorted
        by confidence (descending) before truncation.
    class_ids : set of int or None
        Class IDs to keep.  ``None`` means keep all.  Default ``{0}``
        (face class for a ``nc=1`` face-tuned model).

    Attributes
    ----------
    _model : ultralytics.YOLO
        The underlying YOLO model instance.  Public so tests can mock it.
    """

    def __init__(
        self,
        *,
        model_name: str = "models/yolo26n_tuned.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        max_faces: int = 5,
        class_ids: set[int] | None = None,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.max_faces = max_faces
        self.class_ids = class_ids if class_ids is not None else {0}

        from ultralytics import YOLO

        self._model = YOLO(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Run face detection on *image* and return cropped detections.

        Parameters
        ----------
        image : np.ndarray
            Input image as ``(H, W, 3)`` uint8 RGB or ``(H, W)`` uint8
            grayscale.  Grayscale images are auto-converted to RGB.

        Returns
        -------
        list[FaceDetection]
            Detected faces, sorted by descending confidence, capped at
            :attr:`max_faces`.  Empty list if no faces found.
        """
        rgb = _ensure_rgb(image)

        results = self._model(
            rgb,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[FaceDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                cls_id = int(cls)
                if self.class_ids is not None and cls_id not in self.class_ids:
                    continue
                x1, y1, x2, y2 = map(float, xyxy[:4])
                score = float(conf)

                if score < self.confidence_threshold:
                    continue

                crop = rgb[
                    max(0, int(y1)) : min(rgb.shape[0], int(y2)),
                    max(0, int(x1)) : min(rgb.shape[1], int(x2)),
                ].copy()

                detections.append(
                    FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=score,
                        crop=crop,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_faces]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert *image* to 3-channel uint8 RGB if it isn't already."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image
