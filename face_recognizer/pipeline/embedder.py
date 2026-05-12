"""InsightFace IResNet-100 ArcFace embedding module.

Converts a face crop into a 512-dimensional L2-normalized embedding
using an IResNet-100 backbone trained with ArcFace (Glint360K).

Usage::

    embedder = FaceEmbedder(weights_path="weights/arcface-r100-glint360k.pth")
    vector = embedder.embed(face_crop)        # (512,) float32
    vectors = embedder.embed(batch_of_crops)  # (B, 512) float32
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IResNet model definition (InsightFace)
# ---------------------------------------------------------------------------


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Return a 3×3 convolution with padding=1, no bias."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride,
        padding=1, bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """Return a 1×1 convolution with no bias."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False,
    )


class IBasicBlock(nn.Module):
    """Improved Residual Block (InsightFace IResNet building block).

    Pre-activation design: BN → Conv3×3 → BN → PReLU → Conv3×3 → BN,
    with an optional 1×1 shortcut when dimensions change or stride > 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-05)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-05)

        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels, eps=1e-05),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class IResNet100(nn.Module):
    """IResNet-100 backbone matching the InsightFace arcface-r100-glint360k checkpoint.

    Produces a 512-d embedding from a (B, 3, 112, 112) input.
    """

    def __init__(self) -> None:
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)

        # Residual layers [3, 13, 30, 3] → 49 blocks, 100 conv layers total
        self.layer1 = self._make_layer(64, 64, 3, stride=2)
        self.layer2 = self._make_layer(64, 128, 13, stride=2)
        self.layer3 = self._make_layer(128, 256, 30, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # Embedding head: BN2d → Flatten → FC → BN1d
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-05)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        # First block may have stride > 1 / channel change → needs downsample
        layers.append(IBasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(IBasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a 512-d embedding.

        Args:
            x: (B, 3, 112, 112) float32 tensor in range [-1, 1].

        Returns:
            (B, 512) float32 embedding.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Embedding head
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)

        return x


# ---------------------------------------------------------------------------
# FaceEmbedder public API
# ---------------------------------------------------------------------------


class FaceEmbedder:
    """IResNet-100 ArcFace embedding extractor.

    Parameters
    ----------
    weights_path : str, Path, or None
        Path to the ``arcface-r100-glint360k.pth`` checkpoint.
    device : str
        Torch device (``"cpu"``, ``"cuda"``).
    normalize : bool
        If ``True`` (default), L2-normalize output embeddings.

    Attributes
    ----------
    _model : torch.nn.Module or None
        The underlying embedding model.  Set directly in tests to mock.
    """

    def __init__(
        self,
        *,
        weights_path: Optional[str | Path] = None,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.weights_path = str(weights_path) if weights_path else None
        self.device = device
        self.normalize = normalize

        # Lazily loaded; tests may override directly.
        self._model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from one or more face crops.

        Parameters
        ----------
        image : np.ndarray
            Single crop ``(H, W, 3)`` uint8 RGB,
            ``(H, W)`` uint8 grayscale,
            or batch ``(B, H, W, 3)`` uint8 RGB.

        Returns
        -------
        np.ndarray
            Single image → ``(512,)`` float32, L2-normalized.
            Batch → ``(B, 512)`` float32, L2-normalized.
        """
        was_single = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)

        # Normalise to batch (B, H, W, C)
        batch = _ensure_batch(image)

        # Preprocess (resize, normalize, convert to tensor)
        tensor = self._preprocess(batch)

        # Run model
        model = self._get_model()
        with torch.no_grad():
            embeddings = model(tensor)

        embeddings = embeddings.cpu().numpy().astype(np.float32)

        # L2-normalize
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return embeddings[0] if was_single else embeddings

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self) -> nn.Module:
        """Return the embedding model, loading if necessary."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> nn.Module:
        """Build the IResNet-100 and load pre-trained weights."""
        logger.info("Building IResNet-100 ArcFace backbone")
        model = IResNet100()

        if self.weights_path is not None:
            logger.info("Loading weights from %s", self.weights_path)
            state = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(state, strict=True)
            if missing:
                logger.warning("Missing keys: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys: %s", unexpected)
        else:
            logger.info("No weights provided — using random initialization.")

        model.eval()
        model.to(self.device)
        return model

    def _preprocess(self, batch: np.ndarray) -> torch.Tensor:
        """Convert (B, H, W, 3) uint8 → (B, 3, 112, 112) float32 in [-1, 1].

        InsightFace convention: ``(pixel - 127.5) / 127.5``.
        """
        import cv2

        B, H, W, C = batch.shape
        target_h = target_w = 112

        # Resize to 112×112 if needed
        if H != target_h or W != target_w:
            resized = np.empty((B, target_h, target_w, C), dtype=np.float32)
            for i in range(B):
                resized[i] = cv2.resize(
                    batch[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR,
                )
        else:
            resized = batch.astype(np.float32)

        # Normalize to [-1, 1] (InsightFace convention)
        resized = (resized - 127.5) / 127.5

        # HWC → CHW and to tensor
        tensor = torch.from_numpy(resized).permute(0, 3, 1, 2).to(self.device)
        return tensor


# ---------------------------------------------------------------------------
# Helpers
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
