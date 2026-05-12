"""LFW evaluation — 20 identities, enrollment + recognition benchmark.

Generates ``results/eval_results.json`` with:
- Overall accuracy / FAR / FRR
- Per-identity breakdown
- Failure-mode classification (near‑miss, low-genuine, high-impostor)
- Per‑stage timing percentiles

Usage::

    uv run python scripts/eval.py   # ~3-5 minutes for ~1,800 queries on CPU
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"
EVAL_MANIFEST = ROOT / "samples" / "lfw_eval" / "manifest.json"


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_pipeline():
    """Build a ``FacePipeline`` from the unified Hydra config."""
    from hydra import compose, initialize_config_dir

    config_dir = os.fspath(ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    from face_recognizer.pipeline.detector import YOLOFaceDetector
    from face_recognizer.pipeline.liveness import LivenessDetector
    from face_recognizer.pipeline.aligner import FaceAligner
    from face_recognizer.pipeline.embedder import FaceEmbedder
    from face_recognizer.db.store import IdentityStore
    from face_recognizer.pipeline.orchestrator import FacePipeline

    detector = YOLOFaceDetector(
        model_name=cfg.detector.model_name,
        confidence_threshold=cfg.detector.confidence_threshold,
        iou_threshold=cfg.detector.iou_threshold,
        device=cfg.detector.device,
        max_faces=cfg.detector.max_faces,
    )
    liveness = LivenessDetector(
        model_path="models/minifasv2.pth",
        device=cfg.liveness.device,
        threshold=cfg.liveness.threshold,
    )
    aligner = FaceAligner(
        model_id=cfg.aligner.model_id,
        device=cfg.aligner.device,
    )
    embedder = FaceEmbedder(
        weights_path=cfg.embedder.weights_path,
        device=cfg.embedder.device,
    )
    store = IdentityStore(
        db_path=cfg.store.db_path,
        index_path=cfg.store.index_path,
        threshold=cfg.store.threshold,
    )
    store.load()

    pipeline = FacePipeline(
        detector=detector,
        liveness=liveness,
        aligner=aligner,
        embedder=embedder,
        store=store,
        skip_liveness=True,   # static photos
        skip_aligner=False,   # 250×250 images, DFA works
    )

    return pipeline, store


def _load_image(path: str) -> np.ndarray:
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _search_all(store, embedding: np.ndarray) -> List[tuple[str, float]]:
    """Like ``store.search()`` but returns ALL results (no threshold filter)."""
    idx = store._ensure_index()
    if idx.ntotal == 0:
        return []

    vec = np.asarray(embedding, dtype=np.float32).ravel()
    vec = vec / (np.linalg.norm(vec) + 1e-12)

    scores, indices = idx.search(vec.reshape(1, -1), idx.ntotal)

    conn = store._ensure_conn()
    results = []
    seen = set()
    for faiss_pos, score in zip(indices[0], scores[0]):
        if faiss_pos < 0:
            continue
        row_id = store._faiss_to_row[faiss_pos]
        row = conn.execute("SELECT name FROM identities WHERE id=?", (row_id,)).fetchone()
        if row is None:
            continue
        name = row[0]
        if name in seen:
            continue
        seen.add(name)
        results.append((name, float(score)))

    return results


# ── Main evaluation ──────────────────────────────────────────────────────

def run_eval() -> Dict[str, Any]:
    t_start = time.perf_counter()

    print("=" * 64, flush=True)
    print("  Face Recognizer — LFW Evaluation (20 identities)", flush=True)
    print("=" * 64, flush=True)
    print(flush=True)

    # ── Load manifest ────────────────────────────────────────────────────
    with open(EVAL_MANIFEST) as fh:
        manifest = json.load(fh)
    identities = manifest["identities"]
    print(f"  Manifest: {len(identities)} identities", flush=True)

    # ── Load pipeline ────────────────────────────────────────────────────
    print("  Loading pipeline …", flush=True)
    pipeline, store = _load_pipeline()

    # Clear any existing data
    for name in store.list_names():
        store.delete(name)

    # ── Enroll ───────────────────────────────────────────────────────────
    print("  Enrolling identities …", flush=True)
    t_e = time.perf_counter()
    enrolled_names: set[str] = set()

    for ident in identities:
        name = ident["name"]
        path = ident["enrollment_image"]
        if not os.path.exists(path):
            print(f"    ✗ {name}: missing", flush=True)
            continue
        image = _load_image(path)
        try:
            row_id = pipeline.enroll(image, name, skip_liveness=True)
        except Exception as exc:
            print(f"    ✗ {name}: {exc}", flush=True)
            continue
        if row_id is not None:
            enrolled_names.add(name)
        else:
            print(f"    ✗ {name}: no face", flush=True)

    t_enroll = (time.perf_counter() - t_e) * 1000
    print(f"    {len(enrolled_names)}/{len(identities)} enrolled ({t_enroll:.0f} ms)", flush=True)

    # Adaptive threshold
    threshold = store.calibrate_threshold(margin=0.05)
    print(f"    Threshold: {threshold:.4f}", flush=True)

    # ── Recognition queries ──────────────────────────────────────────────
    print("  Running queries …", flush=True)

    # Aggregates
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    stage_times: dict[str, list[float]] = defaultdict(list)
    total_times: list[float] = []

    # Counters
    tp = 0   # correctly matched genuine
    tn = 0   # correctly rejected impostor
    fp = 0   # impostor falsely accepted
    fn = 0   # genuine falsely rejected (below threshold or matched wrong person)
    no_face = 0
    # Failure modes (scored after the fact)
    near_miss = 0
    low_genuine = 0
    high_impostor = 0

    per_id: dict[str, dict] = {}
    query_count = sum(len(i["query_images"]) for i in identities)
    total = 0
    progress_every = max(1, query_count // 20)

    for ident in identities:
        true_name = ident["name"]
        safe_name = ident["safe_name"]
        if true_name not in enrolled_names:
            continue

        id_s = {
            "enrolled": True,
            "total": len(ident["query_images"]),
            "tp": 0, "fn": 0, "tn": 0, "fp": 0,
            "no_face": 0,
            "genuine_scores": [],
            "impostor_scores": [],
        }

        for img_rel in ident["query_images"]:
            total += 1
            if total % progress_every == 0:
                pct = 100 * total / query_count
                print(f"    … {total}/{query_count} ({pct:.0f}%)", flush=True)

            img_path = str(ROOT / img_rel)
            if not os.path.exists(img_path):
                id_s["no_face"] += 1
                no_face += 1
                continue

            image = _load_image(img_path)
            result = pipeline.process(image, skip_liveness=True)

            for stage, ms in result.per_stage_ms.items():
                stage_times[stage].append(ms)
            total_times.append(result.total_ms)

            # No face detected
            if result.face_bbox is None:
                id_s["no_face"] += 1
                no_face += 1
                continue

            # Determine ground-truth: what's the best score for the true name?
            # We need the embedding. Pipeline has already computed it, but
            # pipeline.process() doesn't expose it directly. Re-derive via
            # store.search_all for classification purposes.
            embedding = _extract_embedding(pipeline, image)
            if embedding is None:
                id_s["no_face"] += 1
                no_face += 1
                continue

            all_results = _search_all(store, embedding)
            best_name = all_results[0][0] if all_results else None
            best_score = all_results[0][1] if all_results else 0.0

            # Get true-name score
            true_score = 0.0
            for n, s in all_results:
                if n == true_name:
                    true_score = s
                    break

            is_genuine = best_name == true_name
            matched = best_score >= threshold

            if is_genuine:
                genuine_scores.append(true_score)
                if matched:
                    tp += 1
                    id_s["tp"] += 1
                    id_s["genuine_scores"].append(round(true_score, 4))
                else:
                    fn += 1
                    id_s["fn"] += 1
                    id_s["genuine_scores"].append(round(true_score, 4))
                    if true_score > threshold - 0.05:
                        near_miss += 1
                    if true_score < 0.40:
                        low_genuine += 1
            else:
                impostor_scores.append(true_score)
                if matched:
                    fp += 1
                    id_s["fp"] += 1
                    id_s["impostor_scores"].append(round(true_score, 4))
                else:
                    tn += 1
                    id_s["tn"] += 1
                    id_s["impostor_scores"].append(round(true_score, 4))
                    if true_score > 0.50:
                        high_impostor += 1

        per_id[safe_name] = id_s

    t_total = (time.perf_counter() - t_start) * 1000

    # ── Aggregate stats ──────────────────────────────────────────────────
    n_genuine = tp + fn
    n_impostor = tn + fp
    n_classified = n_genuine + n_impostor  # excluding no-face

    gen_arr = np.array(genuine_scores) if genuine_scores else np.zeros(1)
    imp_arr = np.array(impostor_scores) if impostor_scores else np.zeros(1)

    accuracy = (tp + tn) / n_classified * 100 if n_classified else 0.0
    far = fp / n_impostor * 100 if n_impostor else 0.0
    frr = fn / n_genuine * 100 if n_genuine else 0.0

    # Timing
    def _pcts(arr) -> dict:
        a = np.array(arr)
        return {"mean": round(float(a.mean()), 1),
                "p50": round(float(np.percentile(a, 50)), 1),
                "p95": round(float(np.percentile(a, 95)), 1),
                "p99": round(float(np.percentile(a, 99)), 1)}

    timing = {s: _pcts(v) for s, v in stage_times.items()}
    timing["total"] = _pcts(total_times)

    # ── Report ───────────────────────────────────────────────────────────
    return {
        "metadata": {
            "dataset": "LFW subset (Labelled Faces in the Wild)",
            "identities": len(identities),
            "enrolled": len(enrolled_names),
            "total_queries": total,
            "classified": n_classified,
            "no_face": no_face,
            "threshold": round(threshold, 4),
            "threshold_type": "adaptive (max cross-identity + 0.05 margin)",
        },
        "accuracy": {
            "overall_pct": round(accuracy, 2),
            "far_pct": round(far, 2),
            "frr_pct": round(frr, 2),
            "genuine_pairs": n_genuine,
            "impostor_pairs": n_impostor,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
        "failure_breakdown": {
            "near_miss_pct": round(near_miss / n_genuine * 100, 1) if n_genuine else 0,
            "low_genuine_pct": round(low_genuine / n_genuine * 100, 1) if n_genuine else 0,
            "high_impostor_pct": round(high_impostor / n_impostor * 100, 1) if n_impostor else 0,
        },
        "score_distribution": {
            "genuine": {
                "mean": round(float(gen_arr.mean()), 4),
                "std": round(float(gen_arr.std()), 4),
                "p25": round(float(np.percentile(gen_arr, 25)), 4),
                "p50": round(float(np.percentile(gen_arr, 50)), 4),
                "p75": round(float(np.percentile(gen_arr, 75)), 4),
                "min": round(float(gen_arr.min()), 4),
            },
            "impostor": {
                "mean": round(float(imp_arr.mean()), 4),
                "std": round(float(imp_arr.std()), 4),
                "p25": round(float(np.percentile(imp_arr, 25)), 4),
                "p50": round(float(np.percentile(imp_arr, 50)), 4),
                "p75": round(float(np.percentile(imp_arr, 75)), 4),
                "max": round(float(imp_arr.max()), 4),
            },
        },
        "timing": timing,
        "per_identity": per_id,
        "total_wall_time_s": round(t_total / 1000, 1),
    }


def _extract_embedding(pipeline, image: np.ndarray) -> np.ndarray | None:
    """Get the embedding for the first detected face, or None."""
    detections = pipeline._detector.detect(image)
    if not detections:
        return None
    face_crop = detections[0].crop
    if not pipeline.skip_aligner:
        try:
            face_crop = pipeline._aligner.align(face_crop)
        except RuntimeError:
            return None
    return pipeline._embedder.embed(face_crop)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = run_eval()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "eval_results.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    # Print summary
    acc = report["accuracy"]
    fb = report["failure_breakdown"]
    sd = report["score_distribution"]
    tm = report["timing"]
    md = report["metadata"]

    print()
    print("=" * 64)
    print("  EVALUATION RESULTS")
    print("=" * 64)
    print(f"  Accuracy:     {acc['overall_pct']:.2f}%")
    print(f"  FAR:          {acc['far_pct']:.2f}%  ({acc['fp']}/{md['impostor_pairs']})")
    print(f"  FRR:          {acc['frr_pct']:.2f}%  ({acc['fn']}/{md['genuine_pairs']})")
    print(f"  No face:      {md['no_face']}")
    print(f"  Threshold:    {md['threshold']:.4f}")
    print()
    print(f"  Genuine μ±σ:  {sd['genuine']['mean']:.4f} ± {sd['genuine']['std']:.4f}")
    print(f"  Impostor μ±σ: {sd['impostor']['mean']:.4f} ± {sd['impostor']['std']:.4f}")
    print(f"  Separation:   {sd['genuine']['mean'] - sd['impostor']['mean']:.4f}")
    print()
    print(f"  Near-miss:    {fb['near_miss_pct']:.1f}%  of genuine queries")
    print(f"  Low genuine:  {fb['low_genuine_pct']:.1f}%  of genuine queries")
    print(f"  High impostor:{fb['high_impostor_pct']:.1f}%  of impostor queries")
    print()
    print(f"  Pipeline (p50): {tm['total']['p50']:.0f} ms")
    for stage in ["detect", "liveness", "align", "embed", "search"]:
        if stage in tm:
            print(f"    {stage:12s} {tm[stage]['p50']:.0f} ms")
    print()
    print(f"  Wall time: {report['total_wall_time_s']:.0f} s")
    print(f"  Saved:     {out_path}")
