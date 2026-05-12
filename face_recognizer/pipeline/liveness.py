"""MiniFASNet anti-spoofing / liveness detection module.

Wraps MiniFASNetV2SE (PyTorch, vendored from Silent-Face-Anti-Spoofing)
to classify face crops as **real** (live human) or **spoof** (photo, screen).

Usage::

    detector = LivenessDetector(threshold=0.5)
    result = detector.check(face_crop)        # LivenessResult
    results = detector.check(batch_of_crops)  # List[LivenessResult]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet mean / std used by MiniFASNet preprocessing.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class LivenessResult:
    """Outcome of a liveness check.

    Attributes
    ----------
    is_real : bool
        ``True`` if the face is classified as a live human.
    score : float
        Probability of the *real* class in [0, 1].  Higher = more likely real.
    """

    is_real: bool
    score: float


class LivenessDetector:
    """MiniFASNetV2SE anti-spoofing detector (PyTorch).

    Parameters
    ----------
    model_path : str, Path, or None
        Path to ``.pth`` weights.  If ``None`` the model is downloaded from
        the model zoo on first use.
    device : str
        Torch device (``"cpu"``, ``"cuda"``).
    threshold : float
        Scores ≥ *threshold* are classified as **real**.
    input_size : tuple of (int, int)
        Expected input dimensions ``(H, W)``.  Default ``(80, 80)`` for
        MiniFASNetV2SE.

    Attributes
    ----------
    _model : torch.nn.Module or None
        The underlying PyTorch model.  Lazily created on first ``check()`` call.
        May be set directly in tests to mock inference.
    """

    def __init__(
        self,
        *,
        model_path: Optional[str | Path] = None,
        device: str = "cpu",
        threshold: float = 0.5,
        input_size: Tuple[int, int] = (128, 128),
        fp16: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.input_size = tuple(input_size)
        self.fp16 = fp16

        # Lazily loaded; tests may override.
        self._model: Optional[object] = None  # torch.nn.Module

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        face_crop: np.ndarray,
    ) -> Union[LivenessResult, List[LivenessResult]]:
        """Classify one or more face crops as real or spoof.

        Parameters
        ----------
        face_crop : np.ndarray
            Single crop ``(H, W, 3)`` uint8 RGB,
            ``(H, W)`` uint8 grayscale,
            or batch ``(B, H, W, 3)`` uint8 RGB.

        Returns
        -------
        LivenessResult or list[LivenessResult]
            Single result for a single image; list for a batch.
        """
        import torch

        was_single = face_crop.ndim == 2 or (face_crop.ndim == 3 and face_crop.shape[2] == 3)

        # Normalise to batch (B, H, W, C)
        batch = _ensure_batch(face_crop)

        # Preprocess (B, C, H, W) float32 normalised
        tensor = self._preprocess(batch)
        tensor = torch.from_numpy(tensor).to(self.device)

        # Run inference
        model = self._get_model()
        with torch.no_grad():
            logits = model(tensor)  # (B, 2)

        # Class 0 = real, class 1 = spoof (model training convention)
        probs = torch.softmax(logits, dim=-1)
        real_scores = probs[:, 0].cpu().numpy()

        results = [
            LivenessResult(
                is_real=float(score) >= self.threshold,
                score=float(score),
            )
            for score in real_scores
        ]

        return results[0] if was_single else results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the PyTorch model, loading if necessary."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load MiniFASNetV2SE architecture and weights."""
        import torch

        from face_recognizer.models.minifasv2 import MultiFTNet
        model = MultiFTNet(num_classes=2, embedding_size=128, conv6_kernel=(8, 8))
        model.eval()

        # Load weights
        path = self.model_path
        if path is None:
            # Look for vendored weights
            candidates = [
                "models/minifasv2.pth",
                str(Path(__file__).resolve().parent.parent.parent / "models" / "minifasv2.pth"),
            ]
            for cand in candidates:
                if Path(cand).exists():
                    path = cand
                    break
            if path is None:
                raise FileNotFoundError(
                    "MiniFASNetV2SE weights not found. Set model_path or place "
                    "minifasv2.pth in the models/ directory."
                )

        logger.info("Loading MiniFASNetV2SE weights from %s", path)
        state = torch.load(path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)

        if self.fp16:
            model = model.half()

        model.to(self.device)
        return model

    def _preprocess(self, batch: np.ndarray) -> np.ndarray:
        """Convert (B, H, W, 3) uint8 → (B, 3, H_tgt, W_tgt) float32 normalised."""
        import cv2

        B, H, W, C = batch.shape
        target_h, target_w = self.input_size

        resized = np.empty((B, target_h, target_w, C), dtype=np.float32)
        for i in range(B):
            resized[i] = cv2.resize(batch[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Scale to [0, 1]
        resized /= 255.0

        # Normalise with ImageNet stats
        resized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC → CHW
        return np.transpose(resized, (0, 3, 1, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_batch(image: np.ndarray) -> np.ndarray:
    """Convert *image* to 4-dim batch (B, H, W, C)."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
        image = image[np.newaxis, ...]
    elif image.ndim == 3:
        image = image[np.newaxis, ...]

    if image.shape[-1] == 4:
        image = image[..., :3]

    return image.astype(np.uint8, copy=False)
