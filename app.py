import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.mediapipe_pose import run_pose_demo


def run_script(script_name: str, extra_args=None):
    extra_args = extra_args or []
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    command = [sys.executable, str(script_path), *[str(arg) for arg in extra_args]]
    return subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)


def main():
    parser = argparse.ArgumentParser(description="Privacy-preserving elderly safety system")
    parser.add_argument(
        "--demo",
        choices=[
            "pose",
            "extract",
            "extract-keypoints",
            "build",
            "build-dataset",
            "prepare",
            "prepare-real-data",
            "convert-ur-benchmark",
            "download-ur-benchmark",
            "download-benchmark",
            "organize-benchmark",
            "organize",
            "train",
            "train-gru",
            "full",
            "evaluate",
            "eval",
            "inference",
            "ui",
        ],
        default="pose",
        help="Which phase of the pipeline to run",
    )
    parser.add_argument("--source", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--limit", type=int, default=30, help="Frames to capture in extraction mode")
    parser.add_argument("--output", type=str, default="data/processed/sample_pose_keypoints.csv", help="CSV path for extracted features")
    parser.add_argument("--data-root", type=str, default="data/raw/demo_samples", help="Directory with labelled normal/fall CSVs")
    parser.add_argument("--source-root", type=str, default="data/raw", help="Source directory for benchmark downloads or raw CSV files")
    parser.add_argument("--dataset", choices=["ur-fall", "up-fall"], default="ur-fall", help="Public benchmark dataset to download or organize")
    parser.add_argument("--download-url", type=str, default="", help="Optional direct archive URL for a public benchmark dataset")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame when converting benchmark videos")
    parser.add_argument("--sequence-length", type=int, default=30, help="Temporal sequence length")
    parser.add_argument("--stride", type=int, default=1, help="Sequence stride")
    parser.add_argument("--output-dir", type=str, default="data/sequences/demo_dataset", help="Directory for generated dataset arrays")
    parser.add_argument("--target-root", type=str, default="data/raw/benchmark", help="Destination folder for benchmark organization")
    parser.add_argument("--data-dir", type=str, default="data/sequences/demo_dataset", help="Dataset directory for GRU training")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size for training")
    parser.add_argument("--dry-run", action="store_true", help="Preview benchmark organization without copying files")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GRU hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="GRU layer count")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout factor")
    parser.add_argument("--model-path", type=str, default="models/gru_demo/gru_fall_classifier.pt", help="Path to trained GRU checkpoint")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold for fall detection")
    args = parser.parse_args()

    demo = args.demo
    if demo in {"extract", "extract-keypoints"}:
        result = run_script(
            "extract_keypoints.py",
            [
                "--source",
                args.source,
                "--width",
                args.width,
                "--height",
                args.height,
                "--limit",
                args.limit,
                "--output",
                args.output,
            ],
        )
        raise SystemExit(result.returncode)

    if demo in {"build", "build-dataset"}:
        result = run_script(
            "build_dataset.py",
            [
                "--data-root",
                args.data_root,
                "--sequence-length",
                args.sequence_length,
                "--stride",
                args.stride,
                "--output-dir",
                args.output_dir,
            ],
        )
        raise SystemExit(result.returncode)

    if demo in {"prepare", "prepare-real-data"}:
        result = run_script(
            "prepare_real_dataset.py",
            [
                "--dataset-root",
                args.data_root,
                "--sequence-length",
                args.sequence_length,
                "--stride",
                args.stride,
                "--output-dir",
                args.output_dir,
            ],
        )
        raise SystemExit(result.returncode)

    if demo == "convert-ur-benchmark":
        result = run_script(
            "convert_ur_fall_dataset.py",
            [
                "--dataset-root",
                args.data_root,
                "--output-root",
                args.target_root,
                "--frame-step",
                args.frame_step,
            ],
        )
        raise SystemExit(result.returncode)

    if demo == "download-ur-benchmark":
        result = run_script(
            "download_ur_fall_dataset.py",
            [
                "--output-root",
                args.data_root,
            ],
        )
        raise SystemExit(result.returncode)

    if demo == "download-benchmark":
        result = run_script(
            "download_real_benchmark.py",
            [
                "--dataset",
                args.dataset,
                "--download-url",
                args.download_url,
                "--source-root",
                args.source_root,
                "--output-root",
                args.target_root,
            ] + (["--dry-run"] if args.dry_run else []),
        )
        raise SystemExit(result.returncode)

    if demo in {"organize-benchmark", "organize"}:
        result = run_script(
            "organize_benchmark_dataset.py",
            [
                "--source-root",
                args.data_root,
                "--target-root",
                args.target_root,
            ] + (["--dry-run"] if args.dry_run else []),
        )
        raise SystemExit(result.returncode)

    if demo in {"train", "train-gru"}:
        result = run_script(
            "train_gru.py",
            [
                "--data-dir",
                args.data_dir,
                "--batch-size",
                args.batch_size,
                "--epochs",
                args.epochs,
                "--lr",
                args.lr,
                "--hidden-dim",
                args.hidden_dim,
                "--num-layers",
                args.num_layers,
                "--dropout",
                args.dropout,
                "--output-dir",
                "models/gru_demo",
            ],
        )
        raise SystemExit(result.returncode)

    if demo == "full":
        result = run_script(
            "run_full_pipeline.py",
            [
                "--source-root",
                args.data_root,
                "--dataset-root",
                args.target_root,
                "--sequence-length",
                args.sequence_length,
                "--stride",
                args.stride,
                "--output-dir",
                args.output_dir,
                "--model-dir",
                "models/final_model",
                "--batch-size",
                args.batch_size,
                "--epochs",
                args.epochs,
                "--lr",
                args.lr,
                "--hidden-dim",
                args.hidden_dim,
                "--num-layers",
                args.num_layers,
                "--dropout",
                args.dropout,
            ],
        )
        raise SystemExit(result.returncode)

    if demo in {"evaluate", "eval"}:
        result = run_script(
            "evaluate_model.py",
            [
                "--model-path",
                args.model_path,
                "--data-dir",
                args.data_dir,
                "--split",
                "test",
            ],
        )
        raise SystemExit(result.returncode)

    if demo == "inference":
        result = run_script(
            "run_inference_demo.py",
            [
                "--model-path",
                args.model_path,
                "--source",
                args.source,
                "--sequence-length",
                args.sequence_length,
                "--threshold",
                args.threshold,
            ],
        )
        raise SystemExit(result.returncode)

    if demo == "ui":
        try:
            from src.interface.simple_ui import SimpleSafetyUI
        except Exception as exc:
            print(f"Unable to start the UI: {exc}")
            raise SystemExit(1)

        app = SimpleSafetyUI()
        app.run()
        return

    run_pose_demo(source=args.source, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
