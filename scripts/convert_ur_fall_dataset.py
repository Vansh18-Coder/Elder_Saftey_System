import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.mediapipe_pose import PoseDetector

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
UP_FALL_ACTIVITY_LABELS = {
    **{activity_id: "normal" for activity_id in range(1, 7)},
    **{activity_id: "fall" for activity_id in range(7, 12)},
}


def infer_label_from_path(path: Path, dataset_root: Path, dataset: str = "ur-fall"):
    relative_parts = path.relative_to(dataset_root).parts

    if dataset == "up-fall":
        activity_id = None
        for index, part in enumerate(relative_parts):
            name = Path(part).stem.lower()
            activity_match = re.match(r"^(?:activity|act)[-_ ]*0*(\d+)$", name)
            if activity_match:
                activity_id = int(activity_match.group(1))
                break

            if name.isdigit() and 1 <= int(name) <= 11 and index > 0:
                parent_name = Path(relative_parts[index - 1]).stem.lower()
                if re.match(r"^(?:subject|sub)[-_ ]*\d+$", parent_name):
                    activity_id = int(name)
                    break

        if activity_id in UP_FALL_ACTIVITY_LABELS:
            return UP_FALL_ACTIVITY_LABELS[activity_id]

        raise ValueError(
            f"Could not infer UP-Fall activity label from {path}. "
            "Expected an activity folder named activity-01 through activity-11."
        )

    if dataset != "ur-fall":
        raise ValueError(f"Unsupported dataset: {dataset}")

    for part in reversed(relative_parts):
        name = Path(part).stem.lower()
        if re.match(r"^fall(?:[-_ ]|$)", name):
            return "fall"
        if re.match(r"^(?:adl|activity|normal)(?:[-_ ]|$)", name):
            return "normal"

    raise ValueError(
        f"Could not infer fall/normal label from {path}. "
        "UR Fall inputs should be named fall-XX-cam0.mp4 or adl-XX-cam0.mp4."
    )


def find_image_sequence_dirs(dataset_root: Path):
    sequence_dirs = []
    for candidate in dataset_root.rglob("*"):
        if candidate.is_dir() and any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in candidate.iterdir()
        ):
            sequence_dirs.append(candidate)
    return sorted(sequence_dirs)


def find_video_files(dataset_root: Path):
    return sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def flatten_landmarks(landmarks):
    if landmarks is None:
        return [0.0] * 132

    values = []
    for landmark in landmarks[:33]:
        values.extend([
            float(landmark.get("x", 0.0)),
            float(landmark.get("y", 0.0)),
            float(landmark.get("z", 0.0)),
            float(landmark.get("visibility", 0.0)),
        ])
    return (values + [0.0] * (132 - len(values)))[:132]


def image_sequence_frames(sequence_dir: Path, frame_step: int = 1):
    return sorted(
        path for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[::frame_step]


def convert_frames(frame_reader, source_path: Path, source_name: str, label: str, output_root: Path, detector: PoseDetector):
    rows = []
    frame_index = 0
    while True:
        frame = frame_reader()
        if frame is None:
            break

        landmarks, _ = detector.process_frame(frame)
        row = {
            "frame_index": frame_index,
            "person_detected": int(landmarks is not None),
        }
        row.update({
            f"feature_{index}": value
            for index, value in enumerate(flatten_landmarks(landmarks))
        })
        rows.append(row)
        frame_index += 1

    if not rows:
        raise RuntimeError(f"No readable frames were found in {source_path}")

    label_dir = output_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    output_path = label_dir / f"{source_name}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} frames to {output_path}")


def convert_video(
    video_path: Path,
    dataset_root: Path,
    output_root: Path,
    detector: PoseDetector,
    frame_step: int,
    dataset: str = "ur-fall",
):
    label = infer_label_from_path(video_path, dataset_root, dataset)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frame_index = 0

    def read_frame():
        nonlocal frame_index
        while True:
            success, frame = capture.read()
            if not success:
                return None
            current_index = frame_index
            frame_index += 1
            if current_index % frame_step == 0:
                return frame

    try:
        source_name = video_path.relative_to(dataset_root).with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
        convert_frames(read_frame, video_path, source_name, label, output_root, detector)
    finally:
        capture.release()


def convert_image_sequence(
    sequence_dir: Path,
    dataset_root: Path,
    output_root: Path,
    detector: PoseDetector,
    frame_step: int = 1,
    dataset: str = "ur-fall",
):
    label = infer_label_from_path(sequence_dir, dataset_root, dataset)
    frame_paths = image_sequence_frames(sequence_dir, frame_step)
    frame_iterator = iter(frame_paths)

    def read_frame():
        for image_path in frame_iterator:
            frame = cv2.imread(str(image_path))
            if frame is not None:
                return frame
        return None

    source_name = sequence_dir.relative_to(dataset_root).as_posix().replace("/", "_").replace("\\", "_")
    convert_frames(read_frame, sequence_dir, source_name, label, output_root, detector)


def convert_dataset(
    dataset_root: Path,
    output_root: Path,
    frame_step: int = 1,
    dataset: str = "ur-fall",
):
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {dataset_root}\n"
            "Download and extract the dataset's camera files first. For UR Fall, run:\n"
            "  python app.py --demo download-ur-benchmark --data-root data/raw/ur_fall\n"
            "Then run this converter again."
        )
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_root}")
    if frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")

    video_files = find_video_files(dataset_root)
    image_dirs = find_image_sequence_dirs(dataset_root)
    if not video_files and not image_dirs:
        raise FileNotFoundError(
            f"No videos or extracted image sequences found under {dataset_root}.\n"
            "Expected video files named for their fall/adl label, or extracted image sequences "
            "inside labelled folders. UP-Fall camera folders must include activity-01 through "
            "activity-11 in their path."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    detector = PoseDetector()
    converted = 0
    labels = {"normal": 0, "fall": 0}
    try:
        if video_files:
            for video_path in video_files:
                label = infer_label_from_path(video_path, dataset_root, dataset)
                convert_video(
                    video_path,
                    dataset_root,
                    output_root,
                    detector,
                    frame_step,
                    dataset,
                )
                labels[label] += 1
                converted += 1
        else:
            for sequence_dir in image_dirs:
                label = infer_label_from_path(sequence_dir, dataset_root, dataset)
                convert_image_sequence(
                    sequence_dir,
                    dataset_root,
                    output_root,
                    detector,
                    frame_step=frame_step,
                    dataset=dataset,
                )
                labels[label] += 1
                converted += 1
    finally:
        detector.close()

    print(f"Converted {converted} sequences: normal={labels['normal']} fall={labels['fall']}")
    print(f"Pose CSVs saved to {output_root}")
    return output_root


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark videos or extracted RGB frames into labelled pose CSVs."
    )
    parser.add_argument("--dataset-root", type=str, default="data/raw/ur_fall", help="Root of the downloaded or extracted dataset.")
    parser.add_argument("--output-root", type=str, default="data/raw/benchmark", help="Destination for normal/ and fall/ pose CSVs.")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth video frame or image.")
    parser.add_argument("--dataset", choices=["ur-fall", "up-fall"], default="ur-fall", help="Dataset naming and label convention.")
    args = parser.parse_args()

    try:
        convert_dataset(
            Path(args.dataset_root),
            Path(args.output_root),
            frame_step=args.frame_step,
            dataset=args.dataset,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
