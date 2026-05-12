"""Tests for LivenessDetector — MiniFASNetV2SE anti-spoofing (PyTorch)."""

from unittest import mock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Dataclass: LivenessResult
# ---------------------------------------------------------------------------

def test_liveness_result_dataclass_importable() -> None:
    """LivenessResult dataclass must be importable and constructable."""
    from face_recognizer.pipeline.liveness import LivenessResult

    result = LivenessResult(is_real=True, score=0.92)
    assert result.is_real is True
    assert result.score == 0.92


def test_liveness_result_fields_required() -> None:
    """LivenessResult requires is_real and score."""
    from face_recognizer.pipeline.liveness import LivenessResult

    with pytest.raises(TypeError):
        LivenessResult()  # type: ignore[call-arg]


def test_liveness_result_spoof() -> None:
    """Spoof result: is_real=False."""
    from face_recognizer.pipeline.liveness import LivenessResult

    result = LivenessResult(is_real=False, score=0.12)
    assert result.is_real is False
    assert result.score == 0.12


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_liveness_constructor_defaults() -> None:
    """LivenessDetector can be constructed with config values."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.5,
        input_size=(128, 128),
    )
    assert detector.threshold == 0.5
    assert detector.input_size == (128, 128)


def test_liveness_constructor_custom_threshold() -> None:
    """Custom threshold is stored."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.7,
        input_size=(128, 128),
    )
    assert detector.threshold == 0.7


# ---------------------------------------------------------------------------
# check() — mocked PyTorch model
# ---------------------------------------------------------------------------

@pytest.fixture
def face_crop_80x80() -> np.ndarray:
    """A 128×128 RGB face crop (rand ints)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def face_crop_large() -> np.ndarray:
    """A 224×224 RGB face crop (will be resized)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


def test_check_real_face(face_crop_80x80: np.ndarray) -> None:
    """When the model outputs high logit for class 1 (real), result is real."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.5,
        input_size=(128, 128),
    )

    # Mock model: logits favour class 0 (real)
    # softmax([1.73, -1.0]) ≈ [0.939, 0.061]
    mock_model = mock.MagicMock()
    mock_model.return_value = torch.tensor([[1.73, -1.0]])
    detector._model = mock_model

    result = detector.check(face_crop_80x80)
    assert result.is_real is True
    assert result.score == pytest.approx(0.939, abs=0.01)


def test_check_spoof_face(face_crop_80x80: np.ndarray) -> None:
    """When the model outputs low logit for class 0 (real), result is spoof."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.5,
        input_size=(128, 128),
    )

    # Mock model: logits favour class 1 (spoof)
    # softmax([-1.0, 1.73]) ≈ [0.061, 0.939]  → real score = 0.061
    mock_model = mock.MagicMock()
    mock_model.return_value = torch.tensor([[-1.0, 1.73]])
    detector._model = mock_model

    result = detector.check(face_crop_80x80)
    assert result.is_real is False
    assert result.score == pytest.approx(0.06, abs=0.02)


def test_check_resizes_large_crop(face_crop_large: np.ndarray) -> None:
    """A large crop (224×224) is resized to 80×80 before inference."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.5,
        input_size=(128, 128),
    )

    mock_model = mock.MagicMock()
    mock_model.return_value = torch.tensor([[1.73, -1.0]])
    detector._model = mock_model

    result = detector.check(face_crop_large)

    # Verify the input to the model was 80×80
    call_args = mock_model.call_args
    input_tensor = call_args[0][0]  # first positional arg to model()
    assert input_tensor.shape[2] == 128
    assert input_tensor.shape[3] == 128

    assert result.is_real is True


def test_check_grayscale_face() -> None:
    """A grayscale face crop is converted to 3-channel RGB."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.5,
        input_size=(128, 128),
    )

    gray_crop = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    mock_model = mock.MagicMock()
    mock_model.return_value = torch.tensor([[1.73, -1.0]])
    detector._model = mock_model

    # Should not raise
    result = detector.check(gray_crop)
    assert result.is_real is True


def test_check_batch_faces() -> None:
    """Multiple face crops can be checked in one call."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.5,
        input_size=(128, 128),
    )

    batch = np.random.randint(0, 256, (3, 128, 128, 3), dtype=np.uint8)

    mock_model = mock.MagicMock()
    # Batch output: 3 faces. Face 0=real, face 1=spoof, face 2=real
    mock_model.return_value = torch.tensor([
        [1.73, -1.0],   # real (class 0 high)
        [-1.0, 1.73],   # spoof (class 1 high)
        [1.73, -1.0],   # real (class 0 high)
    ])
    detector._model = mock_model

    results = detector.check(batch)

    assert len(results) == 3
    assert results[0].is_real is True
    assert results[1].is_real is False
    assert results[2].is_real is True


# ---------------------------------------------------------------------------
# fp16 mode
# ---------------------------------------------------------------------------

def test_check_fp16_mode() -> None:
    """When fp16=True, the detector works with half-precision model output."""
    from face_recognizer.pipeline.liveness import LivenessDetector

    detector = LivenessDetector(
        model_path=None,
        device="cpu",
        threshold=0.5,
        input_size=(128, 128),
        fp16=True,
    )

    face = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    # Mock model returns half-precision logits
    mock_model = mock.MagicMock()
    mock_model.return_value = torch.tensor([[1.73, -1.0]], dtype=torch.float16)
    detector._model = mock_model

    result = detector.check(face)
    assert result.is_real is True
    assert result.score == pytest.approx(0.939, abs=0.01)
