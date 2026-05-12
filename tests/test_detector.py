"""Tests for YOLOFaceDetector — face detection and cropping."""

from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Dataclass: FaceDetection
# ---------------------------------------------------------------------------

def test_face_detection_dataclass_importable() -> None:
    """Importing FaceDetection must succeed (module may not exist yet)."""
    from face_recognizer.pipeline.detector import FaceDetection

    det = FaceDetection(
        bbox=(10.0, 20.0, 100.0, 120.0),
        confidence=0.95,
        crop=np.zeros((100, 110, 3), dtype=np.uint8),
    )
    assert det.bbox == (10.0, 20.0, 100.0, 120.0)
    assert det.confidence == 0.95
    assert det.crop.shape == (100, 110, 3)


def test_face_detection_fields_are_required() -> None:
    """FaceDetection must require bbox, confidence, crop."""
    from face_recognizer.pipeline.detector import FaceDetection

    with pytest.raises(TypeError):
        FaceDetection()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_detector_constructor_defaults() -> None:
    """YOLOFaceDetector can be constructed with default config values."""
    from face_recognizer.pipeline.detector import YOLOFaceDetector

    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.5,
        iou_threshold=0.45,
        device="cpu",
        max_faces=5,
    )
    assert detector.model_name == "yolo26n.pt"
    assert detector.confidence_threshold == 0.5


def test_detector_constructor_from_config_style() -> None:
    """YOLOFaceDetector accepts config-dict-style arguments (all kwargs)."""
    from face_recognizer.pipeline.detector import YOLOFaceDetector

    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.3,
        iou_threshold=0.5,
        device="cpu",
        max_faces=3,
    )
    assert detector.max_faces == 3
    assert detector.confidence_threshold == 0.3


# ---------------------------------------------------------------------------
# detect() — synthetic / mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_image() -> np.ndarray:
    """A 640×480 RGB image (all black)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """A 640×480 single-channel image."""
    return np.zeros((480, 640), dtype=np.uint8)


def test_detect_empty_image_returns_empty(blank_image: np.ndarray) -> None:
    """An image with no detectable faces returns an empty list."""
    from face_recognizer.pipeline.detector import YOLOFaceDetector

    # Use a mock to avoid downloading the real model
    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.5,
        device="cpu",
    )

    with mock.patch.object(detector, "_model") as mock_model:
        mock_model.return_value = []  # no detections
        results = detector.detect(blank_image)

    assert isinstance(results, list)
    assert len(results) == 0


def test_detect_with_mocked_boxes() -> None:
    """When YOLO returns boxes, detect() produces FaceDetection objects."""
    from unittest.mock import MagicMock

    import numpy as np

    from face_recognizer.pipeline.detector import YOLOFaceDetector

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Simulate a YOLO result with one detection
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = [[100.0, 150.0, 200.0, 250.0]]
    mock_box.conf = [0.92]
    mock_box.cls = [0]  # person class
    mock_result.boxes = mock_box
    mock_result.__len__ = lambda self: 1

    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.5,
        device="cpu",
    )

    # Patch the underlying YOLO model's __call__
    with mock.patch.object(detector, "_model") as mock_model:
        mock_model.return_value = [mock_result]

        results = detector.detect(image)

    assert len(results) == 1
    det = results[0]
    assert det.confidence == 0.92
    assert len(det.bbox) == 4
    assert det.crop.ndim == 3  # RGB crop


def test_detect_grayscale_auto_converts(grayscale_image: np.ndarray) -> None:
    """Grayscale images are auto-converted to 3-channel RGB."""
    from face_recognizer.pipeline.detector import YOLOFaceDetector

    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.5,
        device="cpu",
    )

    with mock.patch.object(detector, "_model") as mock_model:
        mock_model.return_value = []

        # Should not raise — grayscale is handled internally
        results = detector.detect(grayscale_image)

    assert isinstance(results, list)
    assert len(results) == 0


def test_detect_filters_low_confidence() -> None:
    """Detections below confidence_threshold are discarded."""
    from unittest.mock import MagicMock

    import numpy as np

    from face_recognizer.pipeline.detector import YOLOFaceDetector

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = [[100.0, 150.0, 200.0, 250.0]]
    mock_box.conf = [0.3]  # below default 0.5 threshold
    mock_box.cls = [0]
    mock_result.boxes = mock_box
    mock_result.__len__ = lambda self: 1

    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.5,
        device="cpu",
    )

    with mock.patch.object(detector, "_model") as mock_model:
        mock_model.return_value = [mock_result]
        results = detector.detect(image)

    assert len(results) == 0  # filtered out


def test_detect_respects_max_faces() -> None:
    """max_faces caps the number of returned detections."""
    from unittest.mock import MagicMock

    import numpy as np

    from face_recognizer.pipeline.detector import YOLOFaceDetector

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Simulate 5 detections
    mock_results = []
    for i in range(5):
        mr = MagicMock()
        mb = MagicMock()
        mb.xyxy = [[float(i * 50), 100.0, float(i * 50 + 60), 200.0]]
        mb.conf = [0.9]
        mb.cls = [0]
        mr.boxes = mb
        mr.__len__ = lambda self: 1
        mock_results.append(mr)

    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.5,
        max_faces=2,
        device="cpu",
    )

    with mock.patch.object(detector, "_model") as mock_model:
        mock_model.return_value = mock_results
        results = detector.detect(image)

    assert len(results) == 2  # capped at max_faces


def test_detect_crop_is_valid_subarray() -> None:
    """The crop in FaceDetection must be a view/subarray of the original image."""
    from unittest.mock import MagicMock

    import numpy as np

    from face_recognizer.pipeline.detector import YOLOFaceDetector

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    mock_result = MagicMock()
    mock_box = MagicMock()
    # Face in top-left quadrant
    mock_box.xyxy = [[10.0, 10.0, 100.0, 100.0]]
    mock_box.conf = [0.99]
    mock_box.cls = [0]
    mock_result.boxes = mock_box
    mock_result.__len__ = lambda self: 1

    detector = YOLOFaceDetector(
        model_name="yolo26n.pt",
        confidence_threshold=0.5,
        device="cpu",
    )

    with mock.patch.object(detector, "_model") as mock_model:
        mock_model.return_value = [mock_result]
        results = detector.detect(image)

    crop = results[0].crop
    assert crop.shape[0] == 90  # 100 - 10
    assert crop.shape[1] == 90
    assert crop.shape[2] == 3
    # Pixels should match
    np.testing.assert_array_equal(crop[0, 0], image[10, 10])
