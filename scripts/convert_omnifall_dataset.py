import argparse
import csv
import math
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.convert_ur_fall_dataset import flatten_landmarks
from scripts.download_omnifall import LABELS as OMNIFALL_LABELS
from src.pose.mediapipe_pose import PoseDetector

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm")
FALL_LABELS = {"fall", "fallen"}
VALID_LABELS = frozenset(OMNIFALL_LABELS)
REQUIRED_COLUMNS = {"path", "label", "start", "end"}


@dataclass(frozen=True)
class Annotation:
    start: float
    end: float
    label: str


def load_annotations(annotation_path: Path):
    grouped = defaultdict(list)
    zero_duration_count = 0
    with annotation_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Annotation file is empty: {annotation_path}")
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Annotation file is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        for row_number, row in enumerate(reader, start=2):
            relative_path = row["path"].strip()
            label = row["label"].strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(f"Unknown OmniFall action label {label!r} on row {row_number}.")
            try:
                start = float(row["start"])
                end = float(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid annotation times on row {row_number}.") from exc
            if not relative_path or not label or not math.isfinite(start) or not math.isfinite(end) or start >= end:
                if relative_path and label and math.isfinite(start) and math.isfinite(end) and start == end:
                    zero_duration_count += 1
                    continue
                raise ValueError(f"Invalid annotation on row {row_number}: expected a path and start < end.")
            grouped[relative_path].append(Annotation(start, end, label))

    overlap_count = 0
    for relative_path, intervals in grouped.items():
        intervals.sort(key=lambda interval: (interval.start, interval.end))
        normalized = []
        for interval in intervals:
            if normalized and interval.start < normalized[-1].end:
                overlap_count += 1
                previous = normalized.pop()
                if previous.start < interval.start:
                    normalized.append(Annotation(previous.start, interval.start, previous.label))
            normalized.append(interval)
        grouped[relative_path] = normalized

    if zero_duration_count:
        warnings.warn(
            f"Ignored {zero_duration_count} zero-duration OmniFall annotation segments.",
            RuntimeWarning,
            stacklevel=2,
        )
    if overlap_count:
        warnings.warn(
            f"Resolved {overlap_count} overlapping OmniFall boundaries by giving precedence "
            "to the later-starting segment.",
            RuntimeWarning,
            stacklevel=2,
        )
    return dict(grouped)


def find_video(video_root: Path, relative_path: str):
    relative = Path(relative_path)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(f"OmniFall path must stay under the video root: {relative_path}")

    candidate = video_root / relative
    if candidate.is_file():
        return candidate

    candidates = [candidate.with_suffix(extension) for extension in VIDEO_EXTENSIONS]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No local video found for OmniFall annotation path '{relative_path}' under {video_root}. "
        "Extract the OmniFall synthetic archive first."
    )


def _output_source_name(relative_path: str):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(relative_path).as_posix()).strip("_")


def _write_segment(rows, label, source_name, segment_index, output_root):
    if not rows:
        return
    label_dir = output_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    output_path = label_dir / f"{source_name}_segment-{segment_index:04d}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} frames to {output_path}")


def convert_video(
    video_path: Path,
    relative_path: str,
    intervals: list[Annotation],
    output_root: Path,
    detector: PoseDetector,
    frame_step: int = 1,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open OmniFall video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has an invalid frame rate, needed to apply time labels: {video_path}")

    source_name = _output_source_name(relative_path)
    frame_index = 0
    interval_index = 0
    segment_index = 0
    active_label = None
    segment_rows = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            current_index = frame_index
            frame_index += 1
            if current_index % frame_step:
                continue

            timestamp = current_index / fps
            while interval_index < len(intervals) and intervals[interval_index].end <= timestamp:
                interval_index += 1

            label = None
            if interval_index < len(intervals):
                interval = intervals[interval_index]
                if interval.start <= timestamp < interval.end:
                    label = "fall" if interval.label in FALL_LABELS else "normal"

            if label != active_label:
                _write_segment(segment_rows, active_label, source_name, segment_index, output_root)
                if segment_rows:
                    segment_index += 1
                segment_rows = []
                active_label = label

            if label is not None:
                landmarks, _ = detector.process_frame(frame)
                row = {
                    "frame_index": current_index,
                    "person_detected": int(landmarks is not None),
                }
                row.update({
                    f"feature_{index}": value
                    for index, value in enumerate(flatten_landmarks(landmarks))
                })
                segment_rows.append(row)

        _write_segment(segment_rows, active_label, source_name, segment_index, output_root)
    finally:
        capture.release()
    return segment_index + int(bool(segment_rows))


def convert_dataset(
    annotation_path: Path,
    video_root: Path,
    output_root: Path,
    frame_step: int = 1,
    limit: int | None = None,
):
    annotation_path = annotation_path.resolve()
    video_root = video_root.resolve()
    output_root = output_root.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"OmniFall annotation CSV not found: {annotation_path}")
    if not video_root.is_dir():
        raise NotADirectoryError(f"OmniFall video folder not found: {video_root}")
    if frame_step < 1:
        raise ValueError("--frame-step must be at least 1.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be greater than zero.")

    annotations = load_annotations(annotation_path)
    if limit is not None:
        annotations = dict(list(sorted(annotations.items()))[:limit])
    if not annotations:
        raise RuntimeError(f"No OmniFall annotations found in {annotation_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    detector = PoseDetector()
    converted = 0
    try:
        for relative_path, intervals in sorted(annotations.items()):
            video_path = find_video(video_root, relative_path)
            convert_video(
                video_path,
                relative_path,
                intervals,
                output_root,
                detector,
                frame_step=frame_step,
            )
            converted += 1
    finally:
        detector.close()

    print(f"Converted {converted} OmniFall videos to normal/fall pose CSV segments.")
    print(f"Pose CSVs saved to {output_root}")
    return output_root


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotated OmniFall videos into the system's normal/fall pose CSV format."
    )
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/omnifall/labels-syn.csv"))
    parser.add_argument("--video-root", type=Path, default=Path("data/raw/omnifall/videos"))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/omnifall_pose"))
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N videos.")
    args = parser.parse_args()

    convert_dataset(
        args.annotations,
        args.video_root,
        args.output_root,
        frame_step=args.frame_step,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
