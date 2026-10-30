from __future__ import annotations

from typing import Iterable, List

import numpy as np


def flatten_landmarks(landmarks: Iterable[dict]) -> np.ndarray:
    """Flatten MediaPipe-style landmark dictionaries into a single feature vector."""
    flat = []
    for landmark in landmarks:
        flat.extend(
            [
                float(landmark.get("x", 0.0)),
                float(landmark.get("y", 0.0)),
                float(landmark.get("z", 0.0)),
                float(landmark.get("visibility", 0.0)),
            ]
        )
    return np.asarray(flat, dtype=np.float32)


def feature_vector_from_landmarks(
    landmarks: Iterable[dict] | None,
    feature_dim: int = 132,
) -> np.ndarray | None:
    """Convert pose landmarks to the fixed-size vector expected by the fall model."""
    if landmarks is None:
        return None
    if feature_dim <= 0:
        raise ValueError("Feature dimension must be greater than zero.")

    features = flatten_landmarks(landmarks)
    if features.size < feature_dim:
        features = np.pad(features, (0, feature_dim - features.size))
    return features[:feature_dim]


def normalize_keypoints(landmarks: Iterable[dict], image_width: float = 1.0, image_height: float = 1.0) -> np.ndarray:
    """Normalize pixel-space keypoints to [0, 1] using image dimensions.

    This is useful when collecting coordinates from a frame that is represented in pixels.
    """
    normalized = []
    for landmark in landmarks:
        x = float(landmark.get("x", 0.0))
        y = float(landmark.get("y", 0.0))
        z = float(landmark.get("z", 0.0))
        visibility = float(landmark.get("visibility", 0.0))

        if image_width > 0:
            x = x / image_width
        if image_height > 0:
            y = y / image_height

        normalized.append({
            "x": x,
            "y": y,
            "z": z,
            "visibility": visibility,
        })

    return flatten_landmarks(normalized)


def build_temporal_sequence(keypoint_frames: List[np.ndarray], sequence_length: int) -> np.ndarray:
    """Simple sequence builder for later GRU model training.

    The current version drops the oldest values if the buffer is longer than the target,
    or pads with zeros if shorter.
    """
    if not keypoint_frames:
        return np.zeros((sequence_length, 0), dtype=np.float32)

    if len(keypoint_frames) >= sequence_length:
        selected = keypoint_frames[-sequence_length:]
    else:
        selected = list(keypoint_frames)
        while len(selected) < sequence_length:
            selected.insert(0, np.zeros_like(selected[0]))

    return np.asarray(selected, dtype=np.float32)
