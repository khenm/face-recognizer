"""Download model weights required by the face recognition pipeline.

Downloads to ``models/``, ``weights/``, and HuggingFace cache.

Usage::

    uv run python scripts/download_models.py

Total download: ~300 MB.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    # ── YOLO face detector (WIDER Face tuned, ~6 MB) ───────────────────
    "models/yolo26n_tuned.pt": (
        "https://huggingface.co/khenm/yolo26n_face/resolve/main/yolo26n_tuned.pt",
        "YOLO26-Nano face detector (WIDER Face)",
    ),
    # ── MiniFASNet liveness (~2.8 MB) ──────────────────────────────────
    "models/minifasv2.pth": (
        "https://github.com/facenox/face-antispoof-onnx/releases/download/v1.0.0/best_model.pth",
        "MiniFASNetV2SE anti-spoofing",
    ),
    # ── ArcFace IResNet-100 embedding (249 MB) ─────────────────────────
    "weights/arcface-r100-glint360k.pth": (
        "https://huggingface.co/BooBooWu/Vec2Face/resolve/main/weights/arcface-r100-glint360k.pth",
        "IResNet-100 ArcFace (Glint360K)",
    ),
}


def _download(url: str, dest: Path, label: str) -> None:
    """Stream *url* to *dest* with a progress bar."""
    import urllib.request

    print(f"  {label}")
    print(f"    {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # ── progress callback ──────────────────────────────────────────────
    try:
        from tqdm import tqdm

        class _Tqdm:
            def __init__(self):
                self._bar = None

            def __call__(self, block_num, block_size, total_size):
                if self._bar is None and total_size > 0:
                    self._bar = tqdm(
                        total=total_size, unit="B", unit_scale=True,
                        desc=f"      ", leave=False,
                    )
                if self._bar is not None:
                    self._bar.update(block_size)
                    if block_num * block_size >= total_size:
                        self._bar.close()

        reporthook = _Tqdm()
    except ImportError:
        reporthook = None
        print("      (install tqdm for a progress bar)")

    urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"      ✓ {size_mb:.1f} MB → {dest}")


def _ensure_dfa() -> None:
    """Pre-cache the DFA aligner from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    model_id = "minchul/cvlface_DFA_mobilenet"
    print(f"  DFA face aligner")
    print(f"    huggingface: {model_id}")
    path = snapshot_download(model_id)
    print(f"      ✓ cached → {path}")


def main() -> None:
    print("Face Recognizer — model download")
    print(f"  Target directory: {ROOT}")
    print()

    # ── 1. Download weight files ───────────────────────────────────────
    for rel_path, (url, label) in MODELS.items():
        dest = ROOT / rel_path
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ {label} — already exists ({size_mb:.1f} MB)")
            continue
        _download(url, dest, label)

    # ── 2. Pre-cache DFA aligner ───────────────────────────────────────
    print()
    _ensure_dfa()

    print()
    print("All models ready.")
    print("Start the server:  uv run python scripts/serve.py")


if __name__ == "__main__":
    main()
