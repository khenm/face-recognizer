"""Enrollment augmentation — generate multiple embeddings from one image.

Even with a single input image, synthetic augmentations (flip, slight
crops) produce multiple embedding variants for the same identity.  Adding
all variants to FAISS drastically improves Rank-1 matching without
requiring the user to take multiple photos.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


def augment_enrollment(
    image: np.ndarray,
    aligner: object,
    embedder: object,
    n_augments: int = 5,
) -> np.ndarray:
    """Produce multiple (512,) embeddings from a single face image.

    Parameters
    ----------
    image : np.ndarray
        Face crop ``(H, W, 3)`` uint8 RGB or ``(H, W)`` uint8 grayscale.
    aligner : FaceAligner
        Aligner instance with ``.align(image) -> (112,112,3) uint8``.
    embedder : FaceEmbedder
        Embedder instance with ``.embed(image) -> (512,) float32``.
    n_augments : int
        Target number of embeddings (clamped to 1–9).  The returned array
        always includes the original; extra slots are filled by horizontal
        flip and shifted crops.

    Returns
    -------
    np.ndarray
        ``(N, 512)`` float32 embeddings, always L2-normalised.
    """
    n_augments = max(1, min(n_augments, 9))
    embeddings: list[np.ndarray] = []

    # ── 1. Original ──
    aligned = aligner.align(image)
    embeddings.append(embedder.embed(aligned))

    if n_augments == 1:
        return np.stack(embeddings, axis=0)

    # ── 2. Horizontal flip ──
    flipped = cv2.flip(image, 1)
    aligned_flip = aligner.align(flipped)
    embeddings.append(embedder.embed(aligned_flip))

    # ── 3+. Shifted crops (simulates different detection boxes) ──
    h, w = image.shape[:2]
    shifts: list[tuple[int, int]] = [
        (-5, -5), (5, 5), (-5, 5), (5, -5),
        (-10, 0), (10, 0), (0, -10),
    ]
    for dx, dy in shifts:
        if len(embeddings) >= n_augments:
            break
        x1, y1 = max(0, -dx), max(0, -dy)
        x2, y2 = min(w, w - dx), min(h, h - dy)
        if x2 <= x1 or y2 <= y1:
            continue
        shifted = image[y1:y2, x1:x2]
        aligned_aug = aligner.align(shifted)
        embeddings.append(embedder.embed(aligned_aug))

    return np.stack(embeddings, axis=0)
