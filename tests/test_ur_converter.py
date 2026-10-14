from pathlib import Path

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
