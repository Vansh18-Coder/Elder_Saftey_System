import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset_builder import build_windows_from_csv, save_split_dataset, split_windows_by_source


def discover_labelled_csvs(dataset_root: str | Path):
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    candidates = []
    for class_name in ["normal", "fall"]:
        class_dir = root / class_name
        if class_dir.exists():
            candidates.append((class_dir, class_name))

    if not candidates:
        for child in sorted(root.rglob("*.csv")):
            parent_name = child.parent.name.lower()
            if parent_name in {"normal", "fall"}:
                candidates.append((child.parent, parent_name))

    if not candidates:
        raise FileNotFoundError(
            "No labelled CSV files found. Expected a dataset root with normal/ and fall/ folders, "
            "or CSV files inside directories named normal/fall."
        )

    labelled_files = []
    seen = set()
    for folder, class_name in candidates:
        for csv_path in sorted(folder.glob("*.csv")):
            key = str(csv_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            labelled_files.append((csv_path, class_name))

    return labelled_files


def main():
    parser = argparse.ArgumentParser(description="Prepare a real benchmark dataset for pose-based fall detection.")
    parser.add_argument("--dataset-root", type=str, default="data/raw/demo_samples", help="Dataset root containing normal/ and fall/ folders or labelled CSVs.")
    parser.add_argument("--sequence-length", type=int, default=30, help="Number of frames per sequence window.")
    parser.add_argument("--stride", type=int, default=1, help="Window stride across frames.")
    parser.add_argument("--output-dir", type=str, default="data/sequences/benchmark_dataset", help="Where to save the prepared sequence split.")
    args = parser.parse_args()

    labelled_files = discover_labelled_csvs(args.dataset_root)
    windows = []
    for csv_path, label_name in labelled_files:
        label_value = 1 if label_name.lower() == "fall" else 0
        windows.extend(build_windows_from_csv(csv_path, sequence_length=args.sequence_length, stride=args.stride, label=label_value))

    if not windows:
        raise RuntimeError(f"No usable pose sequences were assembled from {args.dataset_root}.")

    train_windows, val_windows, test_windows = split_windows_by_source(windows)
    summary = save_split_dataset(train_windows, val_windows, test_windows, args.output_dir)
    print(f"Prepared benchmark dataset for {len(windows)} extracted windows.")
    print(f"Train={summary['train_count']} Val={summary['val_count']} Test={summary['test_count']}")
    print(f"Saved to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
