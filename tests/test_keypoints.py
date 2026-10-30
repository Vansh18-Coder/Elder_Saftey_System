import numpy as np

from src.preprocessing.keypoints import (
    build_temporal_sequence,
    feature_vector_from_landmarks,
    flatten_landmarks,
    normalize_keypoints,
)


def test_flatten_landmarks_returns_expected_length():
    landmarks = [
        {"x": 0.1, "y": 0.2, "z": 0.3, "visibility": 0.8},
        {"x": 0.5, "y": 0.6, "z": 0.7, "visibility": 0.9},
    ]
    feature_vector = flatten_landmarks(landmarks)
    assert feature_vector.shape == (8,)
    assert np.isclose(feature_vector[0], 0.1)
    assert np.isclose(feature_vector[7], 0.9)


def test_feature_vector_from_landmarks_pads_and_trims_to_model_dimension():
    landmarks = [{"x": 0.1, "y": 0.2, "z": 0.3, "visibility": 0.8}]

    padded = feature_vector_from_landmarks(landmarks, feature_dim=6)
    trimmed = feature_vector_from_landmarks(landmarks * 2, feature_dim=6)

    assert padded is not None
    assert padded.shape == (6,)
    assert np.array_equal(padded[4:], np.zeros(2, dtype=np.float32))
    assert trimmed is not None
    assert trimmed.shape == (6,)
    assert feature_vector_from_landmarks(None) is None


def test_normalize_keypoints_scales_coordinates():
    landmarks = [{"x": 320, "y": 240, "z": 0.1, "visibility": 0.7}]
    normalized = normalize_keypoints(landmarks, image_width=640, image_height=480)
    assert normalized.shape == (4,)
    assert normalized[0] == 0.5
    assert normalized[1] == 0.5


def test_build_temporal_sequence_uses_target_length():
    frames = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
    ]
    sequence = build_temporal_sequence(frames, sequence_length=4)
    assert sequence.shape == (4, 2)
    assert np.array_equal(sequence[-3:], np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
