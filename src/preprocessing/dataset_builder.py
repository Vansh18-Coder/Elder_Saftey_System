from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

EXPECTED_FEATURE_DIM = 132


def load_pose_csv(csv_path: str | Path) -> np.ndarray:
    dataframe = pd.read_csv(csv_path)
    feature_columns = [column for column in dataframe.columns if column.startswith("feature_")]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {csv_path}.")

    feature_matrix = dataframe[feature_columns].to_numpy(dtype=np.float32)
    if feature_matrix.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_DIM} keypoint features in {csv_path}, but found {feature_matrix.shape[1]}."
        )

    return feature_matrix


def pad_or_trim_sequence(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    if sequence.shape[0] >= sequence_length:
        return sequence[:sequence_length]

    pad_width = ((0, sequence_length - sequence.shape[0]), (0, 0))
    return np.pad(sequence, pad_width, mode="constant", constant_values=0.0)


def build_windows_from_csv(csv_path: str | Path, sequence_length: int, stride: int = 1, label: int = 0) -> List[Tuple[np.ndarray, int, str]]:
    feature_matrix = load_pose_csv(csv_path)
    if feature_matrix.shape[0] == 0:
        return []

    windows = []
    if feature_matrix.shape[0] < sequence_length:
        windows.append((pad_or_trim_sequence(feature_matrix, sequence_length), label, str(Path(csv_path).stem)))
        return windows

    for start in range(0, feature_matrix.shape[0] - sequence_length + 1, stride):
        window = feature_matrix[start:start + sequence_length]
        windows.append((pad_or_trim_sequence(window, sequence_length), label, str(Path(csv_path).stem)))

    return windows


def gather_labelled_windows(data_root: str | Path, sequence_length: int, stride: int = 1) -> List[Tuple[np.ndarray, int, str]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    class_names = ["normal", "fall"]
    all_windows: List[Tuple[np.ndarray, int, str]] = []

    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        label_value = 1 if class_name == "fall" else 0
        csv_files = sorted(class_dir.glob("*.csv"))
        for csv_file in csv_files:
            all_windows.extend(build_windows_from_csv(csv_file, sequence_length, stride=stride, label=label_value))

    return all_windows


def split_windows_by_source(windows: Iterable[Tuple[np.ndarray, int, str]], validation_fraction: float = 0.15, test_fraction: float = 0.15):
    grouped: Dict[str, List[Tuple[np.ndarray, int, str]]] = {}
    for window in windows:
        _, _, source_name = window
        grouped.setdefault(source_name, []).append(window)

    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    strata: Dict[Tuple[int, ...], List[str]] = {}
    for source_name, source_windows in grouped.items():
        labels = tuple(sorted({int(window[1]) for window in source_windows}))
        strata.setdefault(labels, []).append(source_name)

    source_splits: Dict[str, str] = {}
    randomizer = random.Random(42)
    for source_names in strata.values():
        shuffled_sources = sorted(source_names)
        randomizer.shuffle(shuffled_sources)
        source_count = len(shuffled_sources)

        if source_count == 1:
            train_count, val_count = 1, 0
        elif source_count == 2:
            train_count, val_count = 1, 0
        else:
            val_count = max(1, int(source_count * validation_fraction))
            test_count = max(1, int(source_count * test_fraction))
            train_count = source_count - val_count - test_count
            while train_count < 1:
                if val_count >= test_count and val_count > 1:
                    val_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    break

        train_end = train_count
        val_end = train_count + val_count
        for source_name in shuffled_sources[:train_end]:
            source_splits[source_name] = "train"
        for source_name in shuffled_sources[train_end:val_end]:
            source_splits[source_name] = "val"
        for source_name in shuffled_sources[val_end:]:
            source_splits[source_name] = "test"

    partitions = {"train": [], "val": [], "test": []}
    for source_name, source_windows in grouped.items():
        partitions[source_splits[source_name]].extend(source_windows)

    return partitions["train"], partitions["val"], partitions["test"]


def save_split_dataset(train_windows, val_windows, test_windows, output_dir: str | Path):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def save_partition(name: str, windows):
        if not windows:
            np.savez_compressed(output_path / f"{name}_x.npz", x=np.zeros((0, 0), dtype=np.float32))
            np.savez_compressed(output_path / f"{name}_y.npz", y=np.zeros((0,), dtype=np.int64))
            return

        x = np.stack([window[0] for window in windows], axis=0).astype(np.float32)
        y = np.asarray([window[1] for window in windows], dtype=np.int64)

        np.savez_compressed(output_path / f"{name}_x.npz", x=x)
        np.savez_compressed(output_path / f"{name}_y.npz", y=y)

        metadata = [
            {"source": source_name, "label": int(label)}
            for _, label, source_name in windows
        ]
        with open(output_path / f"{name}_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    save_partition("train", train_windows)
    save_partition("val", val_windows)
    save_partition("test", test_windows)

    summary = {
        "train_count": len(train_windows),
        "val_count": len(val_windows),
        "test_count": len(test_windows),
        "sequence_length": int(train_windows[0][0].shape[0]) if train_windows else 0,
        "feature_dim": int(train_windows[0][0].shape[1]) if train_windows else EXPECTED_FEATURE_DIM,
    }

    with open(output_path / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
