from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequence_windows(csv_path, sequence_length=30, label=None, stride=1):
    """Load a CSV of keypoints and convert it into fixed-length windows."""
    dataframe = pd.read_csv(csv_path)
    feature_columns = [column for column in dataframe.columns if column.startswith("feature_")]
    values = dataframe[feature_columns].to_numpy(dtype=np.float32)

    if values.shape[0] < sequence_length:
        raise ValueError(
            f"Not enough frames in {csv_path} to build a sequence of length {sequence_length}. "
            f"Found {values.shape[0]} frames."
        )

    windows = []
    for start in range(0, values.shape[0] - sequence_length + 1, stride):
        window = values[start:start + sequence_length]
        windows.append(window)

    labels = np.full(len(windows), label if label is not None else 0, dtype=np.int64)
    return np.asarray(windows, dtype=np.float32), labels
