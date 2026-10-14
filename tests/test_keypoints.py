import numpy as np

from src.preprocessing.keypoints import build_temporal_sequence, flatten_landmarks, normalize_keypoints


def test_flatten_landmarks_returns_expected_length():
    landmarks = [
        {"x": 0.1, "y": 0.2, "z": 0.3, "visibility": 0.8},
        {"x": 0.5, "y": 0.6, "z": 0.7, "visibility": 0.9},
    ]
    feature_vector = flatten_landmarks(landmarks)
    assert feature_vector.shape == (8,)
    assert np.isclose(feature_vector[0], 0.1)
    assert np.isclose(feature_vector[7], 0.9)


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
