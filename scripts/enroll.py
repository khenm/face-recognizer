"""Face enrollment CLI — detect, liveness, align, embed, and store.

Usage::

    uv run python scripts/enroll.py --name Alice path/to/photo.jpg
    uv run python scripts/enroll.py --name Bob path/to/photo.jpg skip_liveness=true
    uv run python scripts/enroll.py --name Eve -- path/to/frames/
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import cv2
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from face_recognizer.pipeline.detector import YOLOFaceDetector
from face_recognizer.pipeline.liveness import LivenessDetector
from face_recognizer.pipeline.aligner import FaceAligner
from face_recognizer.pipeline.embedder import FaceEmbedder
from face_recognizer.pipeline.augment import augment_enrollment
from face_recognizer.db.store import IdentityStore


def build_pipeline(cfg):
    """Build pipeline stages from Hydra config."""
    detector = YOLOFaceDetector(**OmegaConf.to_container(cfg.detector, resolve=True))
    liveness = LivenessDetector(**OmegaConf.to_container(cfg.liveness, resolve=True))
    aligner = FaceAligner(**OmegaConf.to_container(cfg.aligner, resolve=True))
    embedder = FaceEmbedder(**OmegaConf.to_container(cfg.embedder, resolve=True))
    store = IdentityStore(**OmegaConf.to_container(cfg.store, resolve=True))
    store.load()
    return detector, liveness, aligner, embedder, store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll a person into the face database.",
    )
    parser.add_argument("--name", type=str, required=True, help="Person name.")
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="Image files or directories.",
    )
    parser.add_argument(
        "--skip-liveness", action="store_true",
        help="Skip liveness check.",
    )
    parser.add_argument(
        "--skip-aligner", action="store_true",
        help="Skip face alignment.",
    )
    parser.add_argument(
        "--augment", type=int, default=0, metavar="N",
        help="Use enrollment augmentation with N variants (1-9). "
             "Generates flip, crop, and shift variants from each image.",
    )
    args = parser.parse_args()

    # -- Collect images --------------------------------------------------------
    image_paths: list[Path] = []
    for p in args.images:
        path = Path(p)
        if path.is_dir():
            image_paths.extend(sorted(path.glob("*")))
        elif path.is_file():
            image_paths.append(path)

    if not image_paths:
        print("Error: no images found.", file=sys.stderr)
        sys.exit(1)

    # -- Build stages ----------------------------------------------------------
    with initialize_config_dir(version_base=None, config_dir="../configs"):
        cfg = compose(config_name="config")

    detector, liveness, aligner, embedder, store = build_pipeline(cfg)
    skip_liveness = args.skip_liveness or cfg.pipeline.skip_liveness
    skip_aligner = args.skip_aligner or cfg.pipeline.skip_aligner
    do_augment = args.augment > 0

    print(f"Enrolling '{args.name}' from {len(image_paths)} image(s) …")
    if do_augment:
        print(f"  Augmentation: {args.augment} variants per image")
        aug_n = max(1, min(args.augment, 9))

    # -- Process each image ----------------------------------------------------
    enrolled = 0
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠ {img_path.name}: could not read — skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect — fall back to whole image if YOLO finds nothing
        # (handles pre-cropped face images like LFW)
        detections = detector.detect(image)
        if detections:
            face_crop = detections[0].crop
            quality = float(detections[0].confidence)
            is_fallback = False
        else:
            # Use whole image as face crop (already a face crop)
            face_crop = image
            quality = 1.0
            is_fallback = True

        # Liveness (skip for pre-cropped images — resizing artifacts)
        if not skip_liveness and not is_fallback:
            print(f"  DEBUG: running liveness (skip_liveness={skip_liveness}, is_fallback={is_fallback})")
            lr = liveness.check(face_crop)
            if not lr.is_real:
                print(f"  ⚠ {img_path.name}: spoof (score={lr.score:.3f}) — skipping")
                continue
            quality = min(quality, float(lr.score))

        # Align (skip for pre-cropped fallback images)
        if not skip_aligner and not is_fallback:
            try:
                face_crop, align_quality = aligner.align(
                    face_crop, return_quality=True,
                )
                quality = min(quality, align_quality)
            except RuntimeError:
                print(f"  ⚠ {img_path.name}: alignment failed — skipping")
                continue
        elif is_fallback:
            # Pre-cropped image — resize to 112×112 directly
            face_crop = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Embed
        embedding = embedder.embed(face_crop)

        # Store
        if do_augment:
            # Generate augmented variants from the aligned face crop
            aug_embs = augment_enrollment(
                face_crop, aligner, embedder, n_augments=aug_n,
            )
            row_ids = store.enroll_many(
                name=args.name,
                embeddings=aug_embs,
                quality_scores=[quality] * aug_embs.shape[0],
            )
            enrolled += aug_embs.shape[0]
            print(
                f"  ✓ {img_path.name}: enrolled {aug_embs.shape[0]} vectors "
                f"(rows {row_ids[0]}–{row_ids[-1]}, quality={quality:.3f})"
            )
        else:
            row_id = store.enroll(
                name=args.name,
                embedding=embedding,
                quality_score=quality,
            )
            enrolled += 1
            print(
                f"  ✓ {img_path.name}: enrolled (row={row_id}, "
                f"quality={quality:.3f})"
            )

    print(f"\n✓ Enrolled {enrolled} image(s) for '{args.name}'.")
    print(f"  People in index: {store.list_names()}")


if __name__ == "__main__":
    main()
