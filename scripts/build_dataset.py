import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset_builder import gather_labelled_windows, save_split_dataset, split_windows_by_source


def main():
    parser = argparse.ArgumentParser(description="Build a fixed-length sequence dataset from labelled keypoint CSV files.")
    parser.add_argument("--data-root", type=str, default="data/raw/demo_samples", help="Root directory containing normal/ and fall/ CSV files.")
    parser.add_argument("--sequence-length", type=int, default=30, help="Number of frames in each temporal sequence.")
    parser.add_argument("--stride", type=int, default=1, help="Window stride across frames.")
    parser.add_argument("--output-dir", type=str, default="data/sequences/demo_dataset", help="Directory to save train/val/test arrays.")
    args = parser.parse_args()

    windows = gather_labelled_windows(args.data_root, sequence_length=args.sequence_length, stride=args.stride)
    if not windows:
        raise RuntimeError(f"No labelled CSV files were found under {args.data_root}.")

    train_windows, val_windows, test_windows = split_windows_by_source(windows)
    summary = save_split_dataset(train_windows, val_windows, test_windows, args.output_dir)
    print(f"Built dataset with {summary['train_count']} train, {summary['val_count']} validation, and {summary['test_count']} test samples.")
    print(f"Saved outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
