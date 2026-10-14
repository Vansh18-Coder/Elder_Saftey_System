import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def discover_csv_files(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")
    return sorted(root.rglob("*.csv"))


def infer_label_from_path(csv_path: Path):
    path_text = csv_path.as_posix().lower()
    if "fall" in path_text and "normal" not in path_text:
        return "fall"
    if "normal" in path_text:
        return "normal"
    return None


def infer_label_from_csv(csv_path: Path):
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return None

    label_column = None
    for candidate in ["label", "class", "activity", "target", "category"]:
        if candidate in frame.columns:
            label_column = candidate
            break

    if label_column is None:
        return None

    values = frame[label_column].astype(str).str.lower().tolist()
    if any(value in {"fall", "falls", "1", "true", "active", "f"} for value in values):
        return "fall"
    if any(value in {"normal", "0", "false", "idle", "n", "nonfall", "non-fall"} for value in values):
        return "normal"
    return None


def organize_benchmark_dataset(source_root: str | Path, target_root: str | Path, dry_run: bool = False):
    source_root = Path(source_root)
    target_root = Path(target_root)

    csv_files = discover_csv_files(source_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {source_root}.")

    normal_dir = target_root / "normal"
    fall_dir = target_root / "fall"
    if not dry_run:
        normal_dir.mkdir(parents=True, exist_ok=True)
        fall_dir.mkdir(parents=True, exist_ok=True)

    assigned = {"normal": [], "fall": []}
    skipped = []

    for csv_path in csv_files:
        if target_root in csv_path.parents:
            continue

        label = infer_label_from_path(csv_path)
        if label is None:
            label = infer_label_from_csv(csv_path)

        if label is None:
            skipped.append(str(csv_path))
            continue

        destination_dir = normal_dir if label == "normal" else fall_dir
        destination = destination_dir / csv_path.name
        if not dry_run:
            if destination.exists():
                destination = destination_dir / f"{csv_path.stem}_{abs(hash(csv_path.resolve()))}.csv"
            shutil.copy2(csv_path, destination)
        assigned[label].append(str(csv_path))

    summary = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "normal_files": len(assigned["normal"]),
        "fall_files": len(assigned["fall"]),
        "skipped_files": len(skipped),
        "skipped_details": skipped,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Organize a benchmark dataset into normal/ and fall/ folders expected by the project.")
    parser.add_argument("--source-root", type=str, default="data/raw", help="Root folder containing CSV files or subfolders to organize.")
    parser.add_argument("--target-root", type=str, default="data/raw/benchmark", help="Destination folder to write the organized benchmark dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Preview the classification without copying files.")
    args = parser.parse_args()

    summary = organize_benchmark_dataset(args.source_root, args.target_root, dry_run=args.dry_run)
    print(f"Organized benchmark dataset summary:")
    print(f"  normal_csvs: {summary['normal_files']}")
    print(f"  fall_csvs: {summary['fall_files']}")
    print(f"  skipped_files: {summary['skipped_files']}")
    if summary["skipped_files"]:
        print("Skipped files:")
        for item in summary["skipped_details"]:
            print(f" - {item}")

    if args.dry_run:
        print(f"Dry run only. No files were copied to {args.target_root}.")
    else:
        print(f"Files organized into {Path(args.target_root).resolve()}")


if __name__ == "__main__":
    main()
