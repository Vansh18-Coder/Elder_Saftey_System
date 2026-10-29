from pathlib import Path

import numpy as np
import pytest

from scripts import convert_ur_fall_dataset


def test_infer_label_from_ur_video_name():
    root = Path("dataset")

    assert convert_ur_fall_dataset.infer_label_from_path(
        root / "fall-01-cam0.mp4", root
    ) == "fall"
    assert convert_ur_fall_dataset.infer_label_from_path(
        root / "adl-01-cam0.mp4", root
    ) == "normal"


@pytest.mark.parametrize(
    ("activity", "expected_label"),
    [
        ("activity-01", "normal"),
        ("activity-06", "normal"),
        ("activity-07", "fall"),
        ("activity-11", "fall"),
    ],
)
def test_infer_label_from_up_fall_activity_folder(activity, expected_label):
    root = Path("dataset")
    image_sequence = root / "subject-01" / activity / "trial-01" / "Camera1"

    assert convert_ur_fall_dataset.infer_label_from_path(
        image_sequence, root, dataset="up-fall"
    ) == expected_label


def test_infer_label_from_up_fall_numeric_activity_folder():
    root = Path("dataset")
    image_sequence = root / "subject-01" / "7" / "trial-01" / "Camera1"

    assert convert_ur_fall_dataset.infer_label_from_path(
        image_sequence, root, dataset="up-fall"
    ) == "fall"


def test_infer_up_fall_label_rejects_unknown_activity():
    root = Path("dataset")

    with pytest.raises(ValueError, match="Could not infer UP-Fall activity label"):
        convert_ur_fall_dataset.infer_label_from_path(
            root / "subject-01" / "activity-12" / "trial-01" / "Camera1",
            root,
            dataset="up-fall",
        )


def test_infer_label_rejects_unknown_source():
    root = Path("dataset")

    with pytest.raises(ValueError, match="Could not infer fall/normal label"):
        convert_ur_fall_dataset.infer_label_from_path(
            root / "recording-01.mp4", root
        )


def test_missing_dataset_fails_before_pose_detector_initializes(tmp_path, monkeypatch):
    def unexpected_detector():
        raise AssertionError("PoseDetector should not initialize without input data")

    monkeypatch.setattr(convert_ur_fall_dataset, "PoseDetector", unexpected_detector)

    with pytest.raises(FileNotFoundError, match="download-ur-benchmark"):
        convert_ur_fall_dataset.convert_dataset(
            tmp_path / "missing_ur_data",
            tmp_path / "benchmark",
        )


def test_convert_up_fall_camera_sequences_to_pose_csv(tmp_path, monkeypatch):
    class FakePoseDetector:
        def process_frame(self, frame):
            return None, frame

        def close(self):
            pass

    dataset_root = tmp_path / "up_fall_camera"
    output_root = tmp_path / "pose_csvs"
    for activity in ("activity-01", "activity-07"):
        image_dir = dataset_root / "subject-01" / activity / "trial-01" / "Camera1"
        image_dir.mkdir(parents=True)
        for frame_index in range(4):
            image_path = image_dir / f"frame-{frame_index:03d}.png"
            assert convert_ur_fall_dataset.cv2.imwrite(
                str(image_path),
                np.zeros((8, 8, 3), dtype=np.uint8),
            )

    monkeypatch.setattr(convert_ur_fall_dataset, "PoseDetector", FakePoseDetector)

    convert_ur_fall_dataset.convert_dataset(
        dataset_root,
        output_root,
        frame_step=2,
        dataset="up-fall",
    )

    normal_csv = next((output_root / "normal").glob("*.csv"))
    fall_csv = next((output_root / "fall").glob("*.csv"))
    assert len(convert_ur_fall_dataset.pd.read_csv(normal_csv)) == 2
    assert len(convert_ur_fall_dataset.pd.read_csv(fall_csv)) == 2
