"""DFA (Differentiable Face Aligner).

Wraps ``minchul/cvlface_DFA_mobilenet`` from HuggingFace.  The model
outputs a pre-aligned 112×112 canonical face — no manual landmark
extraction or affine warping needed.

Usage::

    aligner = FaceAligner(device="cpu")
    aligned = aligner.align(face_crop)  # (112, 112, 3) uint8
    aligned, quality = aligner.align(face_crop, return_quality=True)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceAligner:
    """DFA mobile0.25 face alignment via HuggingFace.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    device : str
        Torch device (``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        *,
        model_id: str = "minchul/cvlface_DFA_mobilenet",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        face_crop: np.ndarray,
        *,
        return_quality: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, float]:
        """Align a face crop to canonical 112×112.

        Parameters
        ----------
        face_crop : np.ndarray
            ``(H, W, 3)`` uint8 RGB or ``(H, W)`` uint8 grayscale.
        return_quality : bool
            If ``True``, return ``(aligned, quality_score)``.

        Returns
        -------
        np.ndarray or tuple
            Aligned ``(112, 112, 3)`` uint8 RGB image, or tuple with
            quality score.
        """
        import torch

        rgb = _ensure_rgb(face_crop)

        # Convert to tensor: (1, 3, H, W) float [0, 1]
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(self.device)
        )

        # Run model — output[0] is the aligned 112×112 image
        model = self._get_model()
        with torch.no_grad():
            output = model(tensor)

        # output = (aligned_img, _, landmarks, _, theta, _)
        # aligned_img: (1, 3, 112, 112) float [-1, 1] or similar range
        aligned_tensor = output[0]

        # Convert to uint8 numpy
        aligned_np = _tensor_to_uint8(aligned_tensor)

        if return_quality:
            # Use landmark confidence if available (index 3)
            quality = 0.0
            if len(output) > 3 and output[3] is not None:
                conf = output[3]
                if conf.numel() > 0:
                    quality = float(conf.max())
            return aligned_np, quality
        return aligned_np

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the DFA model, downloading on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load DFA mobilenet directly from the HF cache snapshot.

        Bypasses the transformers ``CVLFaceAlignmentModel`` wrapper
        (which has API compatibility issues with newer transformers)
        and loads the aligner architecture + weights directly.
        """
        import os
        import sys
        from pathlib import Path

        import torch
        import yaml
        from omegaconf import OmegaConf

        logger.info("Loading DFA aligner from %s …", self.model_id)

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--minchul--cvlface_DFA_mobilenet" / "snapshots"

        if not model_dir.exists():
            raise RuntimeError(
                f"Model {self.model_id} not cached. Run: "
                f"python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{self.model_id}')\""
            )

        snapshots = sorted(model_dir.iterdir())
        if not snapshots:
            raise RuntimeError(f"No snapshots found for {self.model_id}")
        snapshot = str(snapshots[-1])

        # Add snapshot to sys.path so `from aligners import ...` works
        if snapshot not in sys.path:
            sys.path.insert(0, snapshot)

        # Load config and build aligner
        prev_cwd = os.getcwd()
        os.chdir(snapshot)
        try:
            with open("pretrained_model/model.yaml") as f:
                model_conf = OmegaConf.create(yaml.safe_load(f))

            from aligners import get_aligner

            model = get_aligner(model_conf)
            model.load_state_dict_from_path("pretrained_model/model.pt")
        finally:
            os.chdir(prev_cwd)

        model.eval()
        model.to(self.device)
        logger.info("DFA aligner loaded on %s", self.device)
        return model


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert *image* to 3-channel uint8 RGB."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _tensor_to_uint8(tensor) -> np.ndarray:
    """Convert a (1, 3, H, W) float tensor to (H, W, 3) uint8 numpy."""
    import torch

    img = tensor.detach().cpu()
    if img.ndim == 4:
        img = img.squeeze(0)
    # Handle various input ranges: [-1,1], [0,1], or already scaled
    img_min = img.min()
    img_max = img.max()
    if img_min < 0:
        # [-1, 1] range → rescale to [0, 255]
        img = (img + 1.0) * 127.5
    else:
        # [0, 1] range → rescale to [0, 255]
        img = img * 255.0
    img = img.clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    return img
