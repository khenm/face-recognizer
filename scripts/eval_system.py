#!/usr/bin/env python3
"""System-level evaluation of the face-recognizer HTTP API.

Hits real HTTP endpoints (/health, /enroll, /recognize, /people, /calibrate,
/people/{name}) for end-to-end system testing of the full stack — from HTTP
request through YOLO → MiniFASNet → DFA → ArcFace → SQLite+FAISS.

Usage::

    # Terminal 1: start server
    uv run python scripts/serve.py

    # Terminal 2: run system test
    uv run python scripts/eval_system.py \\
        --server-url http://localhost:8000 \\
        --enrollment-dir samples/lfw_eval/enrollment \\
        --query-dir samples/lfw_eval/query \\
        --holdout 6 \\
        --seed 42 \\
        --output-dir results/system_test

Outputs (under ``results/system_test/``)::

    raw_results.csv          # every query — genuine + impostor
    enrollment_log.csv       # enrollment outcomes
    summary.json             # TAR, FAR, FRR, latency, threshold
    per_person_tar.csv       # per-identity breakdown
    threshold_sweep.csv      # τ vs FAR/FRR for DET curve
    holdout_split.json       # reproducibility (seed + split)
    edge_cases.json          # edge case test pass/fail
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from urllib.parse import quote as _urlquote

# ── Optional tqdm support ───────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    _tqdm = None  # type: ignore[assignment]


def _plain_progress(iterable, desc: str = ""):
    """Plain periodic-print fallback when tqdm is unavailable."""
    items = list(iterable)
    n = len(items)
    step = max(1, n // 10) if n > 10 else 1
    print(f"  {desc} ({n} items) …", flush=True)
    for i, item in enumerate(items):
        if i % step == 0 or i == n - 1:
            print(f"    {i + 1}/{n}", flush=True)
        yield item


def _progress(iterable, desc: str = "", total: int | None = None, **kwargs):
    """tqdm if available, otherwise plain iteration with periodic print."""
    if HAS_TQDM and _tqdm is not None:
        return _tqdm(iterable, desc=desc, total=total, unit="img", **kwargs)
    return _plain_progress(iterable, desc)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _scan_names(enrollment_dir: Path) -> List[str]:
    """Collect identity names from .jpg stems in the enrollment directory."""
    names = []
    for fp in sorted(enrollment_dir.glob("*.jpg")):
        names.append(fp.stem)
    if not names:
        sys.exit(f"No .jpg files found in {enrollment_dir}")
    return names


def _scan_query_names(query_dir: Path) -> set[str]:
    """Collect identity names from sub-directory names in the query directory."""
    names = set()
    for sub in sorted(query_dir.iterdir()):
        if sub.is_dir():
            names.add(sub.name)
    return names


def _query_images(query_dir: Path, name: str) -> List[Path]:
    """Return all .jpg query images for *name*, sorted."""
    sub = query_dir / name
    if not sub.is_dir():
        return []
    return sorted(sub.glob("*.jpg"))


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _pct_stats(values: List[float]) -> dict:
    """Basic percentiles for a list of floats."""
    if not values:
        return {"mean": 0, "std": 0, "p50": 0, "p95": 0}
    s = sorted(values)
    n = len(s)
    return {
        "mean": round(sum(s) / n, 2),
        "std": round(
            (sum((x - sum(s) / n) ** 2 for x in s) / n) ** 0.5, 2
        ),
        "p50": round(s[int(n * 0.50)], 2),
        "p95": round(s[int(n * 0.95)], 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0 — Preflight
# ═══════════════════════════════════════════════════════════════════════════

def phase0_preflight(
    client: httpx.Client,
    server_url: str,
    enrollment_dir: Path,
    query_dir: Path,
    holdout_count: int,
    seed: int,
    output_dir: Path,
) -> Tuple[List[str], List[str], float]:
    """Health check, clean slate, scan samples, random split.

    Returns (enrolled_names, holdout_names, elapsed_ms).
    """
    t0 = _now_ms()

    # ── Health check ──────────────────────────────────────────────────
    print("Phase 0 — Preflight", flush=True)
    r = client.get(f"{server_url}/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health = r.json()
    assert health.get("status") == "ok", f"Health not ok: {health}"
    print(f"  ✓ Server healthy — {server_url}", flush=True)

    # ── Clean slate ───────────────────────────────────────────────────
    r = client.get(f"{server_url}/people", timeout=10)
    people = r.json()
    existing = people.get("names", [])
    if existing:
        print(f"  Clearing {len(existing)} existing identities …", flush=True)
        for name in existing:
            client.delete(f"{server_url}/people/{_urlquote(name)}", timeout=10)
        # Verify empty
        r = client.get(f"{server_url}/people", timeout=10)
        assert len(r.json().get("names", [])) == 0, "Failed to clear all identities"
    print("  ✓ Clean slate — 0 enrolled", flush=True)

    # ── Scan samples ──────────────────────────────────────────────────
    enroll_names = _scan_names(enrollment_dir)
    query_names = _scan_query_names(query_dir)
    valid = sorted(set(enroll_names) & query_names)
    missing = set(enroll_names) ^ query_names
    if missing:
        print(f"  ⚠ {len(missing)} names mismatch (enrollment ↔ query): {missing}", flush=True)
    print(f"  ✓ {len(valid)} valid identities (enrollment + query match)", flush=True)

    if len(valid) < holdout_count + 1:
        print(f"  ⚠ Only {len(valid)} identities, reducing holdout to max {len(valid) - 1}", flush=True)
        holdout_count = max(1, len(valid) - 1)

    # ── Random split (seeded) ─────────────────────────────────────────
    rng = random.Random(seed)
    shuffled = list(valid)
    rng.shuffle(shuffled)
    enrolled = sorted(shuffled[:-holdout_count])
    holdout = sorted(shuffled[-holdout_count:])
    print(f"  Enrolled: {len(enrolled)}  |  Holdout: {len(holdout)}  (seed={seed})", flush=True)

    # Save split for reproducibility
    split_data = {
        "seed": seed,
        "total": len(valid),
        "enrolled_count": len(enrolled),
        "holdout_count": len(holdout),
        "enrolled": enrolled,
        "holdout": holdout,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "holdout_split.json", "w") as fh:
        json.dump(split_data, fh, indent=2)
    print(f"  ✓ Saved holdout_split.json", flush=True)

    elapsed = _now_ms() - t0
    return enrolled, holdout, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Enrollment
# ═══════════════════════════════════════════════════════════════════════════

def phase1_enrollment(
    client: httpx.Client,
    server_url: str,
    enrolled_names: List[str],
    enrollment_dir: Path,
    output_dir: Path,
) -> Tuple[float, float]:
    """Enroll each identity via POST /enroll, then calibrate.

    Returns (threshold, elapsed_ms).
    """
    t0 = _now_ms()
    enrollment_log: List[Dict[str, Any]] = []

    print("\nPhase 1 — Enrollment", flush=True)

    for name in _progress(enrolled_names, desc="Enrolling"):
        img_path = enrollment_dir / f"{name}.jpg"
        t_start = _now_ms()
        try:
            with open(img_path, "rb") as fh:
                r = client.post(
                    f"{server_url}/enroll",
                    data={"name": name, "skip_liveness": "true"},
                    files={"file": ("frame.jpg", fh, "image/jpeg")},
                    timeout=30,
                )
            elapsed = round(_now_ms() - t_start, 1)
            data = r.json() if r.status_code == 200 else None
            entry = {
                "name": name,
                "status_code": r.status_code,
                "row_id": data["row_id"] if data else None,
                "quality": data.get("quality") if data else None,
                "latency_ms": elapsed,
                "error": None,
            }
            if r.status_code != 200:
                entry["error"] = data.get("detail", f"HTTP {r.status_code}") if data else f"HTTP {r.status_code}"
                print(f"    ✗ {name}: {entry['error']}", flush=True)
            else:
                print(f"    ✓ {name} (id={entry['row_id']}, {elapsed:.0f}ms)", flush=True)
            enrollment_log.append(entry)
        except Exception as e:
            elapsed = round(_now_ms() - t_start, 1)
            enrollment_log.append({
                "name": name,
                "status_code": None,
                "row_id": None,
                "quality": None,
                "latency_ms": elapsed,
                "error": str(e),
            })
            print(f"    ✗ {name}: {e}", flush=True)

    # ── Verify count ──────────────────────────────────────────────────
    r = client.get(f"{server_url}/people", timeout=10)
    people = r.json()
    enrolled_count = len(people.get("names", []))
    print(f"  ✓ {enrolled_count}/{len(enrolled_names)} enrolled", flush=True)

    # ── Calibrate ─────────────────────────────────────────────────────
    r = client.post(f"{server_url}/calibrate", data={"margin": "0.05"}, timeout=10)
    cal = r.json()
    threshold = cal["threshold"]
    print(f"  ✓ Calibrated threshold = {threshold:.4f}", flush=True)

    # ── Save enrollment log ───────────────────────────────────────────
    with open(output_dir / "enrollment_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "status_code", "row_id", "quality", "latency_ms", "error"])
        w.writeheader()
        w.writerows(enrollment_log)
    print(f"  ✓ Saved enrollment_log.csv ({len(enrollment_log)} rows)", flush=True)

    elapsed = _now_ms() - t0
    return threshold, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2+3 — Queries (genuine + impostor)
# ═══════════════════════════════════════════════════════════════════════════

def _run_queries(
    client: httpx.Client,
    server_url: str,
    names: List[str],
    query_dir: Path,
    query_type: str,  # "genuine" or "impostor"
) -> List[Dict[str, Any]]:
    """POST /recognize for every query image of every name."""
    results: List[Dict[str, Any]] = []
    total_images = sum(len(_query_images(query_dir, n)) for n in names)

    desc = f"{query_type.capitalize()} queries"
    printed = 0
    for name in _progress(names, desc=desc, total=total_images):
        for img_path in _query_images(query_dir, name):
            printed += 1
            t_start = _now_ms()
            try:
                with open(img_path, "rb") as fh:
                    r = client.post(
                        f"{server_url}/recognize",
                        data={"skip_liveness": "true"},
                        files={"file": ("frame.jpg", fh, "image/jpeg")},
                        timeout=30,
                    )
                elapsed = round(_now_ms() - t_start, 1)
                data = r.json() if r.status_code == 200 else None
                row = {
                    "image_path": str(img_path),
                    "query_type": query_type,
                    "expected_name": name if query_type == "genuine" else "NONE",
                    "returned_name": data.get("name") if data else None,
                    "confidence": data.get("confidence") if data else None,
                    "matched": data.get("matched") if data else False,
                    "is_real": data.get("is_real") if data else None,
                    "total_ms": data.get("total_ms") if data else elapsed,
                    "detect_ms": data.get("per_stage_ms", {}).get("detect") if data else None,
                    "liveness_ms": data.get("per_stage_ms", {}).get("liveness") if data else None,
                    "align_ms": data.get("per_stage_ms", {}).get("align") if data else None,
                    "embed_ms": data.get("per_stage_ms", {}).get("embed") if data else None,
                    "search_ms": data.get("per_stage_ms", {}).get("search") if data else None,
                    "status_code": r.status_code,
                    "error": None if r.status_code == 200 else (data.get("detail") if data else f"HTTP {r.status_code}"),
                }
            except Exception as e:
                elapsed = round(_now_ms() - t_start, 1)
                row = {
                    "image_path": str(img_path),
                    "query_type": query_type,
                    "expected_name": name if query_type == "genuine" else "NONE",
                    "returned_name": None,
                    "confidence": None,
                    "matched": False,
                    "is_real": None,
                    "total_ms": elapsed,
                    "detect_ms": None,
                    "liveness_ms": None,
                    "align_ms": None,
                    "embed_ms": None,
                    "search_ms": None,
                    "status_code": None,
                    "error": str(e),
                }
            results.append(row)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Edge cases
# ═══════════════════════════════════════════════════════════════════════════

def phase4_edge_cases(
    client: httpx.Client,
    server_url: str,
    enrolled_names: List[str],
    enrollment_dir: Path,
) -> List[Dict[str, Any]]:
    """Test edge cases: empty file, corrupt bytes, duplicate enroll,
    delete + recognize, re-enroll."""
    results: List[Dict[str, Any]] = []

    print("\nPhase 4 — Edge cases", flush=True)

    # ── Empty file ────────────────────────────────────────────────────
    test = "Empty file → expect 400"
    r = client.post(
        f"{server_url}/recognize",
        data={"skip_liveness": "true"},
        files={"file": ("frame.jpg", b"", "image/jpeg")},
        timeout=10,
    )
    passed = r.status_code == 400
    results.append({"test": test, "passed": passed, "status_code": r.status_code, "detail": r.json().get("detail") if r.content else None})
    print(f"  {'✓' if passed else '✗'} {test}  (HTTP {r.status_code})", flush=True)

    # ── Corrupted bytes ───────────────────────────────────────────────
    test = "Corrupted bytes → expect 400"
    r = client.post(
        f"{server_url}/recognize",
        data={"skip_liveness": "true"},
        files={"file": ("frame.jpg", b"\xff\x00NOT_JPEG", "image/jpeg")},
        timeout=10,
    )
    passed = r.status_code == 400
    results.append({"test": test, "passed": passed, "status_code": r.status_code, "detail": r.json().get("detail") if r.content else None})
    print(f"  {'✓' if passed else '✗'} {test}  (HTTP {r.status_code})", flush=True)

    # ── Duplicate enroll ──────────────────────────────────────────────
    if enrolled_names:
        dup_name = enrolled_names[0]
        img_path = enrollment_dir / f"{dup_name}.jpg"
        test = f"Duplicate enroll '{dup_name}' → expect 200 (multiple vectors OK)"

        with open(img_path, "rb") as fh:
            r = client.post(
                f"{server_url}/enroll",
                data={"name": dup_name, "skip_liveness": "true"},
                files={"file": ("frame.jpg", fh, "image/jpeg")},
                timeout=30,
            )
        passed = r.status_code == 200
        results.append({"test": test, "passed": passed, "status_code": r.status_code, "detail": r.json() if r.status_code == 200 else r.json().get("detail")})
        print(f"  {'✓' if passed else '✗'} {test}  (HTTP {r.status_code})", flush=True)

    # ── Delete + recognize ────────────────────────────────────────────
    if len(enrolled_names) >= 2:
        victim = enrolled_names[-1]  # delete last enrolled
        img_path = query_dir = Path("samples/lfw_eval/query")
        victim_imgs = _query_images(query_dir, victim)

        # Delete
        test = f"DELETE /people/{victim} → expect 200"
        r = client.delete(f"{server_url}/people/{_urlquote(victim)}", timeout=10)
        passed = r.status_code == 200
        results.append({"test": test, "passed": passed, "status_code": r.status_code, "detail": r.json() if r.content else None})
        print(f"  {'✓' if passed else '✗'} {test}  (HTTP {r.status_code})", flush=True)

        # Recognize — should not match
        if victim_imgs:
            test = f"Recognize deleted identity '{victim}' → expect no match"
            with open(victim_imgs[0], "rb") as fh:
                r = client.post(
                    f"{server_url}/recognize",
                    data={"skip_liveness": "true"},
                    files={"file": ("frame.jpg", fh, "image/jpeg")},
                    timeout=30,
                )
            data = r.json() if r.status_code == 200 else {}
            passed = r.status_code == 200 and not data.get("matched", True)
            results.append({"test": test, "passed": passed, "status_code": r.status_code, "matched": data.get("matched"), "name": data.get("name")})
            print(f"  {'✓' if passed else '✗'} {test}  (matched={data.get('matched')}, name={data.get('name')})", flush=True)

        # Re-enroll to restore state
        re_enroll_path = enrollment_dir / f"{victim}.jpg"
        test = f"Re-enroll '{victim}' after delete → expect 200"
        with open(re_enroll_path, "rb") as fh:
            r = client.post(
                f"{server_url}/enroll",
                data={"name": victim, "skip_liveness": "true"},
                files={"file": ("frame.jpg", fh, "image/jpeg")},
                timeout=30,
            )
        passed = r.status_code == 200
        results.append({"test": test, "passed": passed, "status_code": r.status_code, "row_id": r.json().get("row_id") if r.status_code == 200 else None})
        print(f"  {'✓' if passed else '✗'} {test}  (HTTP {r.status_code})", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Compute metrics
# ═══════════════════════════════════════════════════════════════════════════

def phase5_metrics(
    all_results: List[Dict[str, Any]],
    threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, dict]]:
    """Compute TAR, FAR, FRR, per-person breakdown, latency stats."""
    print("\nPhase 5 — Metrics", flush=True)

    genuine_rows = [r for r in all_results if r["query_type"] == "genuine"]
    impostor_rows = [r for r in all_results if r["query_type"] == "impostor"]

    n_genuine = len(genuine_rows)
    n_impostor = len(impostor_rows)

    # TAR: matched AND returned_name == expected_name
    tp = sum(
        1 for r in genuine_rows
        if r["matched"] and r["returned_name"] == r["expected_name"]
    )
    # FAR: impostor matched (at server threshold)
    fp = sum(1 for r in impostor_rows if r["matched"])

    tar = tp / n_genuine if n_genuine else 0.0
    far = fp / n_impostor if n_impostor else 0.0
    frr = 1.0 - tar

    # ── Per-person TAR ───────────────────────────────────────────────
    per_person: Dict[str, dict] = {}
    by_name: Dict[str, List[dict]] = {}
    for r in genuine_rows:
        by_name.setdefault(r["expected_name"], []).append(r)

    for name, rows in sorted(by_name.items()):
        correct = sum(1 for r in rows if r["matched"] and r["returned_name"] == name)
        per_person[name] = {
            "total": len(rows),
            "correct": correct,
            "tar": round(correct / len(rows), 4) if rows else 0.0,
        }

    # ── Latency stats ─────────────────────────────────────────────────
    all_total_ms = [r["total_ms"] for r in all_results if r["total_ms"] is not None]

    # Per-stage timing from per_stage_ms in raw results
    stage_names = ["detect", "liveness", "align", "embed", "search"]
    stage_cols = {"detect": "detect_ms", "liveness": "liveness_ms",
                  "align": "align_ms", "embed": "embed_ms", "search": "search_ms"}
    per_stage: Dict[str, dict] = {}
    for stage in stage_names:
        col = stage_cols[stage]
        values = [r[col] for r in all_results if r.get(col) is not None]
        per_stage[stage] = _pct_stats(values)

    latency = _pct_stats(all_total_ms)

    # Score distributions
    genuine_conf = [r["confidence"] for r in genuine_rows if r["confidence"] is not None]
    impostor_conf = [r["confidence"] for r in impostor_rows if r["confidence"] is not None]

    summary = {
        "threshold": round(threshold, 4),
        "genuine_total": n_genuine,
        "impostor_total": n_impostor,
        "tp": tp,
        "fp": fp,
        "TAR": round(tar, 4),
        "FAR": round(far, 4),
        "FRR": round(frr, 4),
        "latency_ms": latency,
        "per_stage_ms": per_stage,
        "genuine_confidence": _pct_stats(genuine_conf),
        "impostor_confidence": _pct_stats(impostor_conf),
    }

    print(f"  TAR = {tar:.4f}  ({tp}/{n_genuine})", flush=True)
    print(f"  FAR = {far:.4f}  ({fp}/{n_impostor})", flush=True)
    print(f"  FRR = {frr:.4f}", flush=True)
    print(f"  Latency p50 = {latency['p50']} ms", flush=True)

    return summary, per_person


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — Threshold sweep
# ═══════════════════════════════════════════════════════════════════════════

def phase6_threshold_sweep(
    all_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sweep τ from 0.20 to 0.80, recompute FAR/FRR from stored confidence scores."""
    print("\nPhase 6 — Threshold sweep (offline)", flush=True)

    genuine_rows = [r for r in all_results if r["query_type"] == "genuine"]
    impostor_rows = [r for r in all_results if r["query_type"] == "impostor"]

    n_genuine = len(genuine_rows)
    n_impostor = len(impostor_rows)

    sweep_rows = []
    for tau_100 in range(20, 81, 2):  # 0.20 to 0.80 step 0.02
        tau = tau_100 / 100.0

        # At threshold τ: matched if confidence >= τ AND returned_name == expected (for genuine)
        tp = sum(
            1 for r in genuine_rows
            if (r["confidence"] or 0) >= tau and r["returned_name"] == r["expected_name"]
        )
        fp = sum(1 for r in impostor_rows if (r["confidence"] or 0) >= tau)

        tar = tp / n_genuine if n_genuine else 0.0
        far = fp / n_impostor if n_impostor else 0.0
        frr = 1.0 - tar

        sweep_rows.append({
            "tau": round(tau, 2),
            "TAR": round(tar, 6),
            "FAR": round(far, 6),
            "FRR": round(frr, 6),
        })

    return sweep_rows


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="System-level face-recognizer evaluation (HTTP API)",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Base URL of the face-recognizer server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--enrollment-dir",
        default="samples/lfw_eval/enrollment",
        help="Directory with one .jpg per identity (default: samples/lfw_eval/enrollment)",
    )
    parser.add_argument(
        "--query-dir",
        default="samples/lfw_eval/query",
        help="Directory with query images in sub-folders per identity (default: samples/lfw_eval/query)",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=6,
        help="Number of identities to hold out as impostors (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/system_test",
        help="Output directory (default: results/system_test)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root
    ROOT = Path(__file__).resolve().parent.parent
    enrollment_dir = ROOT / args.enrollment_dir
    query_dir = ROOT / args.query_dir
    output_dir = ROOT / args.output_dir
    server_url = args.server_url.rstrip("/")

    if not enrollment_dir.is_dir():
        sys.exit(f"Enrollment directory not found: {enrollment_dir}")
    if not query_dir.is_dir():
        sys.exit(f"Query directory not found: {query_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Shared HTTP client (connection pooling) ───────────────────────
    client = httpx.Client(timeout=30.0)
    overall_start = _now_ms()

    try:
        # Phase 0
        enrolled_names, holdout_names, p0_ms = phase0_preflight(
            client, server_url, enrollment_dir, query_dir,
            args.holdout, args.seed, output_dir,
        )

        # Phase 1
        threshold, p1_ms = phase1_enrollment(
            client, server_url, enrolled_names, enrollment_dir, output_dir,
        )

        # Phase 2 — Genuine queries
        print("\nPhase 2 — Genuine queries", flush=True)
        genuine_results = _run_queries(
            client, server_url, enrolled_names, query_dir, "genuine",
        )

        # Phase 3 — Impostor queries
        print("\nPhase 3 — Impostor queries", flush=True)
        impostor_results = _run_queries(
            client, server_url, holdout_names, query_dir, "impostor",
        )

        all_results = genuine_results + impostor_results

        # ── Save raw_results.csv ──────────────────────────────────────
        csv_fields = [
            "image_path", "query_type", "expected_name", "returned_name",
            "confidence", "matched", "is_real", "total_ms",
            "detect_ms", "liveness_ms", "align_ms", "embed_ms", "search_ms",
            "status_code", "error",
        ]
        with open(output_dir / "raw_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\n  ✓ Saved raw_results.csv ({len(all_results)} rows)", flush=True)

        # Phase 4 — Edge cases
        edge_cases = phase4_edge_cases(
            client, server_url, enrolled_names, enrollment_dir,
        )
        with open(output_dir / "edge_cases.json", "w") as fh:
            json.dump(edge_cases, fh, indent=2)
        print(f"  ✓ Saved edge_cases.json ({len(edge_cases)} tests)", flush=True)

        # Phase 5 — Metrics
        summary, per_person = phase5_metrics(all_results, threshold)

        with open(output_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  ✓ Saved summary.json", flush=True)

        with open(output_dir / "per_person_tar.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "total", "correct", "tar"])
            w.writeheader()
            for name, stats in sorted(per_person.items()):
                w.writerow({"name": name, **stats})
        print(f"  ✓ Saved per_person_tar.csv ({len(per_person)} identities)", flush=True)

        # Phase 6 — Threshold sweep
        sweep = phase6_threshold_sweep(all_results)

        with open(output_dir / "threshold_sweep.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["tau", "TAR", "FAR", "FRR"])
            w.writeheader()
            w.writerows(sweep)
        print(f"  ✓ Saved threshold_sweep.csv ({len(sweep)} rows)", flush=True)

        # ── Final summary ─────────────────────────────────────────────
        overall_ms = _now_ms() - overall_start
        print()
        print("=" * 64)
        print("  SYSTEM EVALUATION COMPLETE")
        print("=" * 64)
        print(f"  Server:        {server_url}")
        print(f"  Enrolled:      {len(enrolled_names)}")
        print(f"  Holdout:       {len(holdout_names)}")
        print(f"  Genuine q:     {len(genuine_results)}")
        print(f"  Impostor q:    {len(impostor_results)}")
        print(f"  Threshold:     {threshold:.4f}")
        print(f"  TAR:           {summary['TAR']:.4f}")
        print(f"  FAR:           {summary['FAR']:.4f}")
        print(f"  FRR:           {summary['FRR']:.4f}")
        print(f"  Latency p50:   {summary['latency_ms']['p50']} ms")
        print(f"  Per-stage p50:")
        stage_labels = {"detect": "Detect", "liveness": "Liveness", "align": "Align", "embed": "Embed", "search": "Search"}
        for stage in ["detect", "liveness", "align", "embed", "search"]:
            if stage in summary.get("per_stage_ms", {}):
                ps = summary["per_stage_ms"][stage]
                print(f"    {stage_labels[stage]:12s} p50={ps['p50']} ms  mean={ps['mean']} ms")
        print(f"  Wall time:     {overall_ms / 1000:.0f} s")
        print(f"  Output:        {output_dir}/")
        print()

    finally:
        client.close()


if __name__ == "__main__":
    main()
