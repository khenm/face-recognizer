"""Face recognition access control pipeline.

Pipeline stages:
    detector     — YOLO26-Nano face detection and cropping
    liveness     — MiniFASNet anti-spoofing
    aligner      — DFA 5-point landmark alignment (112×112 canonical)
    embedder     — IResNet-100 ArcFace 512-dim embedding
    orchestrator — End-to-end pipeline chaining all stages
"""

from .detector import YOLOFaceDetector, FaceDetection
from .liveness import LivenessDetector, LivenessResult
from .aligner import FaceAligner
from .embedder import FaceEmbedder
from .augment import augment_enrollment
from .orchestrator import FacePipeline, PipelineResult

__all__ = [
    "YOLOFaceDetector",
    "FaceDetection",
    "LivenessDetector",
    "LivenessResult",
    "FaceAligner",
    "FaceEmbedder",
    "augment_enrollment",
    "FacePipeline",
    "PipelineResult",
]
