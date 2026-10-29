import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from scripts import convert_omnifall_dataset, download_omnifall


def test_download_annotations_converts_class_ids_to_names(tmp_path, monkeypatch):
    page = {
        "dataset": download_omnifall.DATASET,
        "config": download_omnifall.CONFIG,
        "num_rows_total": 1,
        "features": [
            {"name": "path"},
            {"name": "label"},
            {"name": "start"},
            {"name": "end"},
            {"name": "dataset"},
        ],
        "rows": [{
            "row_idx": 0,
            "row": {
                "path": "fall/fall_ch_001",
                "label": 2,
                "start": 0.0,
                "end": 1.25,
                "dataset": "of-syn",
            },
        }],
    }
    monkeypatch.setattr(
        download_omnifall,
        "urlopen",
        lambda request, timeout: BytesIO(json.dumps(page).encode("utf-8")),
    )

    output_path = download_omnifall.download_annotations(tmp_path / "labels.csv")

    with output_path.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert row["label"] == "fallen"
    assert row["label_id"] == "2"
    assert row["path"] == "fall/fall_ch_001"


def test_download_annotations_resumes_partial_csv(tmp_path, monkeypatch):
    page = {
        "dataset": download_omnifall.DATASET,
        "config": download_omnifall.CONFIG,
        "num_rows_total": 2,
        "features": [
            {"name": "path"},
            {"name": "label"},
            {"name": "start"},
            {"name": "end"},
            {"name": "dataset"},
        ],
        "rows": [{
            "row_idx": 1,
            "row": {
                "path": "fall/fall_ch_002",
                "label": 0,
                "start": 0.0,
                "end": 1.0,
                "dataset": "of-syn",
            },
        }],
    }
    request_urls = []

    def fake_urlopen(request, timeout):
        request_urls.append(request.full_url)
        return BytesIO(json.dumps(page).encode("utf-8"))

    monkeypatch.setattr(download_omnifall, "urlopen", fake_urlopen)
    partial_path = tmp_path / "labels.csv.part"
    partial_path.write_text(
        "path,label_id,start,end,dataset,label\n"
        "fall/fall_ch_001,1,0.0,1.0,of-syn,fall\n",
        encoding="utf-8",
    )

    output_path = download_omnifall.download_annotations(tmp_path / "labels.csv")

    with output_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[-1]["label"] == "walk"
    assert "offset=1" in request_urls[0]


def test_missing_destination_drive_has_actionable_error(tmp_path, monkeypatch):
    original_is_dir = Path.is_dir

    def is_dir(path):
        if str(path) == "Z:\\":
            return False
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)

    with pytest.raises(FileNotFoundError, match=r"Destination drive Z:\\ is not available"):
        download_omnifall.ensure_output_directory(Path("Z:/data/omnifall"))


def test_video_archive_download_resumes_partial_file_and_checks_hash(tmp_path, monkeypatch):
    complete_archive = b"omnifall-video-archive"
    partial_archive = complete_archive[:8]
    archive_path = tmp_path / "omnifall-synthetic_av1.tar"
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    partial_path.write_bytes(partial_archive)
    monkeypatch.setattr(download_omnifall, "VIDEO_ARCHIVE_SIZE", len(complete_archive))
    monkeypatch.setattr(
        download_omnifall,
        "VIDEO_ARCHIVE_SHA256",
        hashlib.sha256(complete_archive).hexdigest(),
    )
    requests = []

    class RangeResponse(BytesIO):
        status = 206

        def __init__(self, content):
            super().__init__(content)
            self.headers = {
                "Content-Range": f"bytes 8-{len(complete_archive) - 1}/{len(complete_archive)}"
            }

    def fake_urlopen(request, timeout):
        requests.append(request)
        return RangeResponse(complete_archive[len(partial_archive):])

    monkeypatch.setattr(download_omnifall, "urlopen", fake_urlopen)

    result = download_omnifall.download_video_archive(tmp_path)

    assert result == archive_path
    assert archive_path.read_bytes() == complete_archive
    assert not partial_path.exists()
    assert requests[0].get_header("Range") == f"bytes={len(partial_archive)}-"


def test_convert_video_splits_normal_and_fall_segments(tmp_path, monkeypatch):
    video_path = tmp_path / "videos" / "fall" / "fall_ch_001.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    annotations_path = tmp_path / "labels.csv"
    annotations_path.write_text(
        "path,label,start,end\n"
        "fall/fall_ch_001,walk,0.0,0.2\n"
        "fall/fall_ch_001,fall,0.2,0.4\n"
        "fall/fall_ch_001,fallen,0.4,0.5\n",
        encoding="utf-8",
    )

    class FakeCapture:
        def __init__(self, _):
            self.frames = iter([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)])

        def isOpened(self):
            return True

        def get(self, _):
            return 10.0

        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None

        def release(self):
            pass

    class FakePoseDetector:
        def process_frame(self, frame):
            return None, frame

        def close(self):
            pass

    monkeypatch.setattr(convert_omnifall_dataset.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(convert_omnifall_dataset, "PoseDetector", FakePoseDetector)

    output_root = tmp_path / "pose"
    convert_omnifall_dataset.convert_dataset(
        annotations_path,
        video_path.parents[1],
        output_root,
    )

    normal_csv = next((output_root / "normal").glob("*.csv"))
    fall_csv = next((output_root / "fall").glob("*.csv"))
    normal = convert_omnifall_dataset.pd.read_csv(normal_csv)
    fall = convert_omnifall_dataset.pd.read_csv(fall_csv)
    assert normal["frame_index"].tolist() == [0, 1]
    assert fall["frame_index"].tolist() == [2, 3, 4]
    assert len([column for column in normal.columns if column.startswith("feature_")]) == 132


def test_load_annotations_resolves_overlapping_segments(tmp_path):
    annotations_path = tmp_path / "overlap.csv"
    annotations_path.write_text(
        "path,label,start,end\n"
        "fall/example,fall,0.0,1.0\n"
        "fall/example,fallen,0.5,1.5\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Resolved 1 overlapping"):
        intervals = convert_omnifall_dataset.load_annotations(annotations_path)["fall/example"]

    assert intervals == [
        convert_omnifall_dataset.Annotation(0.0, 0.5, "fall"),
        convert_omnifall_dataset.Annotation(0.5, 1.5, "fallen"),
    ]
