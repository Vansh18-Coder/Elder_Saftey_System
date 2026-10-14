import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_demo_benchmark(source_root: Path, target_root: Path):
    target_root.mkdir(parents=True, exist_ok=True)
    normal_dir = target_root / "normal"
    fall_dir = target_root / "fall"
    normal_dir.mkdir(exist_ok=True)
    fall_dir.mkdir(exist_ok=True)

    normal_files = sorted(normal_dir.glob("*.csv"))
    fall_files = sorted(fall_dir.glob("*.csv"))
    if normal_files and fall_files:
        return

    demo_root = source_root / "demo_benchmark"
    if not demo_root.exists():
        demo_root.mkdir(parents=True, exist_ok=True)
        (demo_root / "normal").mkdir(exist_ok=True)
        (demo_root / "fall").mkdir(exist_ok=True)

    for label, count in [("normal", 5), ("fall", 5)]:
        label_dir = demo_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            row = {f"feature_{j}": float((i + 1) * (j + 1) % 17) / 17.0 for j in range(132)}
            df = pd.DataFrame([row])
            df.to_csv(label_dir / f"{label}_{i}.csv", index=False)

    for label in ["normal", "fall"]:
        source_label_dir = demo_root / label
        dest_label_dir = target_root / label
        dest_label_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(source_label_dir.glob("*.csv")):
            shutil.copy2(csv_path, dest_label_dir / csv_path.name)


def run_command(args):
    completed = subprocess.run(args, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}")


def main():
    parser = argparse.ArgumentParser(description="Run the full benchmark-style pipeline end to end and produce a trained model.")
    parser.add_argument("--source-root", type=str, default="data/raw", help="Root directory containing CSVs or a demo benchmark.")
    parser.add_argument("--dataset-root", type=str, default="data/raw/benchmark", help="Folder with normal/ and fall/ CSVs to prepare.")
    parser.add_argument("--sequence-length", type=int, default=10, help="Length of each temporal sequence.")
    parser.add_argument("--stride", type=int, default=1, help="Stride used when building windows.")
    parser.add_argument("--output-dir", type=str, default="data/sequences/benchmark_dataset", help="Directory to store prepared sequence arrays.")
    parser.add_argument("--model-dir", type=str, default="models/final_model", help="Directory to save the trained model.")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GRU hidden size.")
    parser.add_argument("--num-layers", type=int, default=2, help="GRU layer count.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout.")
    args = parser.parse_args()

    source_root = PROJECT_ROOT / args.source_root
    dataset_root = PROJECT_ROOT / args.dataset_root
    ensure_demo_benchmark(source_root, dataset_root)

    run_command([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_real_dataset.py"),
        "--dataset-root",
        str(dataset_root),
        "--sequence-length",
        str(args.sequence_length),
        "--stride",
        str(args.stride),
        "--output-dir",
        str(PROJECT_ROOT / args.output_dir),
    ])

    run_command([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_gru.py"),
        "--data-dir",
        str(PROJECT_ROOT / args.output_dir),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--output-dir",
        str(PROJECT_ROOT / args.model_dir),
    ])

    print(f"Pipeline completed. Model saved to: {str((PROJECT_ROOT / args.model_dir).resolve())}")


if __name__ == "__main__":
    main()
