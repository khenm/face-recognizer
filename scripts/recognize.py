"""Single-image face recognition CLI with per-stage checkpoints.

Usage::

    uv run python scripts/recognize.py path/to/image.jpg
    uv run python scripts/recognize.py image.jpg --config pipeline
    uv run python scripts/recognize.py image.jpg store.threshold=0.7

All intermediate outputs are saved to ``results/{image_stem}/``::

    results/{image_name}/
    ├── 1_detect/
    │   ├── annotated.png
    │   ├── face_01.png
    │   └── detections.json
    ├── 2_liveness/
    │   └── liveness.json
    ├── 3_align/
    │   ├── face_01_aligned.png
    │   └── ...
    ├── 4_embed/
    │   ├── face_01.npy
    │   └── ...
    ├── 5_search/
    │   └── search.json
    └── summary.json
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from face_recognizer.pipeline.detector import YOLOFaceDetector
from face_recognizer.pipeline.liveness import LivenessDetector
from face_recognizer.pipeline.aligner import FaceAligner
from face_recognizer.pipeline.embedder import FaceEmbedder
from face_recognizer.db.store import IdentityStore


def build_stages(cfg):
    """Instantiate pipeline stages from Hydra config."""
    detector = YOLOFaceDetector(**OmegaConf.to_container(cfg.detector, resolve=True))
    liveness = LivenessDetector(**OmegaConf.to_container(cfg.liveness, resolve=True))
    aligner = FaceAligner(**OmegaConf.to_container(cfg.aligner, resolve=True))
    embedder = FaceEmbedder(**OmegaConf.to_container(cfg.embedder, resolve=True))
    store = IdentityStore(**OmegaConf.to_container(cfg.store, resolve=True))
    store.load()
    return detector, liveness, aligner, embedder, store, cfg


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def draw_boxes(image: np.ndarray, detections, color=(0, 255, 0), thickness=3) -> np.ndarray:
    """Draw bounding boxes and confidence labels on an RGB image."""
    annotated = image.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"face_{i+1:02d} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
        )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize a face in an image.")
    parser.add_argument("image", type=str, help="Path to the input image.")
    parser.add_argument(
        "--config", type=str, default="config",
        help="Hydra config name (default: config).",
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
        "overrides", nargs="*", default=[],
        help="Hydra overrides, e.g. 'store.threshold=0.8'.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -- Build stages ----------------------------------------------------------
    with initialize_config_dir(version_base=None, config_dir="../configs"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    detector, liveness, aligner, embedder, store, cfg = build_stages(cfg)
    skip_liveness = args.skip_liveness or cfg.pipeline.skip_liveness
    skip_aligner = args.skip_aligner or cfg.pipeline.skip_aligner

    # -- Results directory -----------------------------------------------------
    results_root = Path("results") / image_path.stem
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to {results_root}/")

    t_total_start = time.perf_counter()
    per_stage_ms = {}

    # =========================================================================
    # Stage 1 — Detection
    # =========================================================================
    t0 = time.perf_counter()
    detections = detector.detect(image)
    per_stage_ms["detect"] = (time.perf_counter() - t0) * 1000

    detect_dir = results_root / "1_detect"
    detections_data = []
    for i, det in enumerate(detections):
        face_name = f"face_{i + 1:02d}"
        crop_path = detect_dir / f"{face_name}.png"
        save_image(crop_path, det.crop)
        detections_data.append({
            "face_id": i,
            "bbox": list(det.bbox),
            "confidence": round(det.confidence, 4),
            "crop_path": str(crop_path),
        })
    save_json(detect_dir / "detections.json", {
        "num_faces": len(detections),
        "faces": detections_data,
        "elapsed_ms": round(per_stage_ms["detect"], 2),
    })

    annotated = draw_boxes(image, detections)
    save_image(detect_dir / "annotated.png", annotated)
    print(f"  [1/5] detect → {len(detections)} face(s) in {per_stage_ms['detect']:.1f}ms")

    if not detections:
        # Fallback: use whole image as face crop (pre-cropped images like LFW)
        detections_data.append({
            "face_id": 0,
            "bbox": [0, 0, image.shape[1], image.shape[0]],
            "confidence": 1.0,
            "crop_path": str(detect_dir / "face_01.png"),
        })
        # Use whole image directly — skip YOLO crop
        import copy
        from face_recognizer.pipeline.detector import FaceDetection
        detections = [
            FaceDetection(
                bbox=(0.0, 0.0, float(image.shape[1]), float(image.shape[0])),
                confidence=1.0,
                crop=image.copy(),
            )
        ]

    # =========================================================================
    # Process each detected face through remaining stages
    # =========================================================================
    liveness_dir = results_root / "2_liveness"
    align_dir = results_root / "3_align"
    embed_dir = results_root / "4_embed"
    search_dir = results_root / "5_search"

    all_liveness = []
    all_aligned = []
    all_embeddings = []
    all_searches = []

    t_liveness_total = 0.0
    t_align_total = 0.0
    t_embed_total = 0.0
    t_search_total = 0.0

    for i, det in enumerate(detections):
        face_crop = det.crop.copy()
        face_id_str = f"face_{i + 1:02d}"
        is_fallback = det.confidence == 1.0  # whole-image fallback

        # ---- Stage 2 — Liveness ----------------------------------------------
        t0 = time.perf_counter()
        is_real = None
        liveness_score = None
        if not skip_liveness and not is_fallback:
            liveness_result = liveness.check(face_crop)
            is_real = liveness_result.is_real
            liveness_score = round(liveness_result.score, 4)
        elif is_fallback:
            is_real = None  # skip liveness for pre-cropped images
        elapsed = (time.perf_counter() - t0) * 1000
        t_liveness_total += elapsed

        all_liveness.append({
            "face_id": i,
            "is_real": is_real,
            "score": liveness_score,
            "elapsed_ms": round(elapsed, 2),
        })

        if not skip_liveness and not is_fallback and not is_real:
            print(f"  [2/5] liveness → {face_id_str} SPOOF (score={liveness_score:.3f}) — skipping")
            continue

        # ---- Stage 3 — Alignment ---------------------------------------------
        t0 = time.perf_counter()
        if not skip_aligner and not is_fallback:
            try:
                face_crop = aligner.align(face_crop)
            except RuntimeError:
                print(f"  [3/5] align → {face_id_str} FAILED — skipping")
                continue
        elif is_fallback:
            # Pre-cropped image — resize to 112×112 directly
            face_crop = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        elapsed = (time.perf_counter() - t0) * 1000
        t_align_total += elapsed

        align_path = align_dir / f"{face_id_str}_aligned.png"
        save_image(align_path, face_crop)
        all_aligned.append({
            "face_id": i,
            "aligned_path": str(align_path),
            "size": list(face_crop.shape),
            "elapsed_ms": round(elapsed, 2),
        })

        # ---- Stage 4 — Embedding ---------------------------------------------
        t0 = time.perf_counter()
        embedding = embedder.embed(face_crop)
        elapsed = (time.perf_counter() - t0) * 1000
        t_embed_total += elapsed

        embed_path = embed_dir / f"{face_id_str}.npy"
        embed_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(embed_path), embedding)
        all_embeddings.append({
            "face_id": i,
            "embedding_path": str(embed_path),
            "shape": list(embedding.shape),
            "l2_norm": round(float(np.linalg.norm(embedding)), 6),
            "elapsed_ms": round(elapsed, 2),
        })

        # ---- Stage 5 — Search ------------------------------------------------
        t0 = time.perf_counter()
        matches = store.search(embedding, k=store.threshold if hasattr(store, 'top_k') else 3)
        elapsed = (time.perf_counter() - t0) * 1000
        t_search_total += elapsed

        search_data = {
            "face_id": i,
            "threshold": store.threshold,
            "matches": [
                {"name": name, "score": round(score, 4)}
                for name, score in matches
            ],
            "matched": any(score >= store.threshold for _, score in matches),
            "elapsed_ms": round(elapsed, 2),
        }
        all_searches.append(search_data)

        status = (
            f"liveness={'✅' if is_real else '⚠️'} "
            f"match={'✅' if search_data['matched'] else '❌'}"
        )
        print(f"  [{face_id_str}] {status}")

    # =========================================================================
    # Write per-stage summary files
    # =========================================================================
    per_stage_ms["liveness"] = round(t_liveness_total, 2)
    per_stage_ms["align"] = round(t_align_total, 2)
    per_stage_ms["embed"] = round(t_embed_total, 2)
    per_stage_ms["search"] = round(t_search_total, 2)

    save_json(liveness_dir / "liveness.json", {
        "faces": all_liveness,
        "elapsed_ms": per_stage_ms["liveness"],
    })
    save_json(align_dir / "align.json", {
        "faces": all_aligned,
        "elapsed_ms": per_stage_ms["align"],
    })
    save_json(embed_dir / "embed.json", {
        "faces": all_embeddings,
        "elapsed_ms": per_stage_ms["embed"],
    })
    save_json(search_dir / "search.json", {
        "faces": all_searches,
        "elapsed_ms": per_stage_ms["search"],
    })

    # =========================================================================
    # Summary
    # =========================================================================
    total_ms = (time.perf_counter() - t_total_start) * 1000

    best_match = None
    for m in all_searches:
        if m["matched"] and m["matches"]:
            best_match = m["matches"][0]
            break

    summary = {
        "image": str(image_path),
        "matched": best_match is not None,
        "name": best_match["name"] if best_match else None,
        "confidence": best_match["score"] if best_match else None,
        "total_ms": round(total_ms, 2),
        "per_stage_ms": per_stage_ms,
        "checkpoint_dir": str(results_root),
    }
    save_json(results_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
