import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET = "simplexsigil2/omnifall"
CONFIG = "labels-syn"
SPLIT = "train"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
VIDEO_ARCHIVE_URL = (
    "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/"
    "data_files/omnifall-synthetic_av1.tar"
)
VIDEO_ARCHIVE_SIZE = 9_716_480_000
VIDEO_ARCHIVE_SHA256 = "c1f7b3beeedf2ffbdb3a3c131da512ceef0d788a93aac93c656b3978a945f13a"
PAGE_SIZE = 100
EXPECTED_ANNOTATION_ROWS = 19_228
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_ARCHIVE_RETRIES = 8
LABELS = (
    "walk",
    "fall",
    "fallen",
    "sit_down",
    "sitting",
    "lie_down",
    "lying",
    "stand_up",
    "standing",
    "other",
    "kneel_down",
    "kneeling",
    "squat_down",
    "squatting",
    "crawl",
    "jump",
)


def fetch_page(request, offset: int):
    for attempt in range(8):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == 7:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = min(float(retry_after), 60) if retry_after and retry_after.isdigit() else min(2 ** attempt, 60)
            print(f"Hugging Face rate-limited the request at row {offset}; retrying in {delay:g}s.")
            time.sleep(delay)

    raise RuntimeError(f"Unable to download OmniFall annotations at offset {offset}.")


def ensure_output_directory(path: Path):
    directory = Path(path)
    if directory.anchor and not Path(directory.anchor).is_dir():
        raise FileNotFoundError(
            f"Destination drive {directory.anchor} is not available. "
            "Choose a directory on an existing drive."
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output directory {directory}: {exc}") from exc


def download_annotations(output_path: Path, limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be greater than zero.")

    output_path = output_path.resolve()
    ensure_output_directory(output_path.parent)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    if output_path.is_file() and not temporary_path.exists() and limit is None:
        with output_path.open("r", newline="", encoding="utf-8-sig") as existing:
            reader = csv.DictReader(existing)
            valid_header = reader.fieldnames and {"path", "label_id", "label", "start", "end"}.issubset(reader.fieldnames)
            row_count = sum(1 for _ in reader)
        if valid_header and row_count == EXPECTED_ANNOTATION_ROWS:
            print(f"OmniFall annotations already present: {output_path}")
            return output_path

    print(f"Downloading OmniFall {CONFIG} annotations (CC BY-NC 4.0).")
    headers = None
    offset = 0
    if temporary_path.exists() and temporary_path.stat().st_size:
        with temporary_path.open("r", newline="", encoding="utf-8") as partial:
            reader = csv.DictReader(partial)
            headers = reader.fieldnames
            if not headers or not {"path", "label_id", "label", "start", "end"}.issubset(headers):
                raise RuntimeError(f"Cannot resume incomplete annotation file with invalid header: {temporary_path}")
            offset = sum(1 for _ in reader)
        print(f"Resuming OmniFall annotation download at row {offset}.")
        if limit is not None and offset > limit:
            raise ValueError(f"Incomplete annotation file already has {offset} rows, exceeding --limit {limit}.")
        if limit is not None and offset == limit:
            temporary_path.replace(output_path)
            print(f"Saved OmniFall annotations to {output_path}")
            return output_path

    total_rows = None
    written = offset

    with temporary_path.open("a" if headers else "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=headers, extrasaction="raise") if headers else None
        while total_rows is None or offset < total_rows:
            query = urlencode({
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": PAGE_SIZE,
            })
            request = Request(
                f"{ROWS_URL}?{query}",
                headers={"User-Agent": "ElderSafetySystem/1.0"},
            )
            page = fetch_page(request, offset)

            if page.get("dataset", DATASET) != DATASET or page.get("config", CONFIG) != CONFIG:
                raise RuntimeError("Hugging Face returned annotations for an unexpected dataset/config.")
            if total_rows is None:
                total_rows = int(page["num_rows_total"])
                if total_rows == 0:
                    raise RuntimeError("The OmniFall annotation table is empty.")

            rows = page.get("rows", [])
            if not rows:
                raise RuntimeError(f"Hugging Face returned no rows at offset {offset} of {total_rows}.")

            if headers is None:
                feature_headers = [feature["name"] for feature in page["features"]]
                if "label" not in feature_headers:
                    raise RuntimeError("The annotation table does not contain a label column.")
                headers = [
                    ("label_id" if name == "label" else name)
                    for name in feature_headers
                ] + ["label"]
                writer = csv.DictWriter(target, fieldnames=headers, extrasaction="raise")
                writer.writeheader()
            else:
                feature_headers = [feature["name"] for feature in page["features"]]
                expected_headers = [
                    ("label_id" if name == "label" else name)
                    for name in feature_headers
                ] + ["label"]
                if headers != expected_headers:
                    raise RuntimeError("The resumed annotation file schema does not match the Hugging Face dataset.")

            for entry in rows:
                row = entry["row"]
                label_id = int(row["label"])
                if not 0 <= label_id < len(LABELS):
                    raise RuntimeError(f"Unknown OmniFall label id {label_id} at row {entry.get('row_idx')}.")
                output_row = dict(row)
                output_row["label_id"] = label_id
                output_row["label"] = LABELS[label_id]
                writer.writerow(output_row)
                written += 1
                if limit is not None and written >= limit:
                    break

            offset += len(rows)
            print(f"Downloaded {written} of {min(total_rows, limit or total_rows)} annotations")
            if limit is not None and written >= limit:
                break

    temporary_path.replace(output_path)
    print(f"Saved OmniFall annotations to {output_path}")
    return output_path


def download_video_archive(output_dir: Path):
    output_dir = output_dir.resolve()
    ensure_output_directory(output_dir)
    archive_path = output_dir / "omnifall-synthetic_av1.tar"
    if archive_path.exists() and archive_path.stat().st_size == VIDEO_ARCHIVE_SIZE:
        print(f"Video archive already present: {archive_path}")
        return archive_path

    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < VIDEO_ARCHIVE_SIZE:
        raise OSError(
            f"Downloading the OmniFall synthetic videos requires at least "
            f"{VIDEO_ARCHIVE_SIZE / 1_000_000_000:.1f} GB of free space; "
            f"only {free_bytes / 1_000_000_000:.1f} GB is available."
        )

    temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
    digest = hashlib.sha256()
    downloaded = temporary_path.stat().st_size if temporary_path.exists() else 0
    if downloaded > VIDEO_ARCHIVE_SIZE:
        raise RuntimeError(
            f"Partial OmniFall archive is larger than expected ({downloaded} bytes): {temporary_path}"
        )
    if downloaded:
        print(f"Resuming OmniFall video archive at {downloaded / 1_000_000_000:.2f} GB.")
        with temporary_path.open("rb") as partial:
            while True:
                chunk = partial.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)

    retry_count = 0
    last_progress = downloaded
    while downloaded < VIDEO_ARCHIVE_SIZE:
        headers = {"User-Agent": "ElderSafetySystem/1.0"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        request = Request(VIDEO_ARCHIVE_URL, headers=headers)
        try:
            with urlopen(request, timeout=120) as response:
                if downloaded:
                    content_range = response.headers.get("Content-Range", "")
                    range_match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if (
                        response.status != 206
                        or range_match is None
                        or int(range_match.group(1)) != downloaded
                        or int(range_match.group(3)) != VIDEO_ARCHIVE_SIZE
                    ):
                        raise RuntimeError(
                            f"Server did not resume the archive at byte {downloaded}; "
                            f"received Content-Range {content_range!r}. Partial file remains at {temporary_path}."
                        )

                with temporary_path.open("ab" if downloaded else "wb") as target:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_progress >= 100 * 1024 * 1024:
                            print(
                                f"Downloaded {downloaded / 1_000_000_000:.2f} / "
                                f"{VIDEO_ARCHIVE_SIZE / 1_000_000_000:.2f} GB"
                            )
                            last_progress = downloaded
        except HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS_CODES:
                raise
            retry_count += 1
            if retry_count > MAX_ARCHIVE_RETRIES:
                raise RuntimeError(
                    f"OmniFall archive download was rate-limited too many times. "
                    f"Rerun the command to resume from {temporary_path}."
                ) from exc
            delay = min(2 ** (retry_count - 1), 60)
            print(f"Hugging Face returned HTTP {exc.code}; retrying in {delay}s.")
            time.sleep(delay)
            continue
        except (URLError, TimeoutError, IncompleteRead) as exc:
            retry_count += 1
            if retry_count > MAX_ARCHIVE_RETRIES:
                raise RuntimeError(
                    f"OmniFall archive connection failed. Rerun the command to resume "
                    f"from {temporary_path}: {exc}"
                ) from exc
            delay = min(2 ** (retry_count - 1), 60)
            print(f"OmniFall archive connection interrupted; retrying in {delay}s.")
            time.sleep(delay)
            continue

        if downloaded < VIDEO_ARCHIVE_SIZE:
            retry_count += 1
            if retry_count > MAX_ARCHIVE_RETRIES:
                raise RuntimeError(
                    f"Expected a {VIDEO_ARCHIVE_SIZE}-byte OmniFall archive, received {downloaded} bytes. "
                    f"Rerun the command to resume from {temporary_path}."
                )
            delay = min(2 ** (retry_count - 1), 60)
            print(
                f"Archive connection closed at {downloaded / 1_000_000_000:.2f} GB; "
                f"resuming in {delay}s."
            )
            time.sleep(delay)

    print(f"Downloaded {downloaded / 1_000_000_000:.2f} / {VIDEO_ARCHIVE_SIZE / 1_000_000_000:.2f} GB")
    if digest.hexdigest() != VIDEO_ARCHIVE_SHA256:
        raise RuntimeError(
            f"The downloaded OmniFall archive failed its SHA-256 check. "
            f"Incomplete/untrusted download is at {temporary_path}."
        )
    temporary_path.replace(archive_path)
    print(f"Saved OmniFall video archive to {archive_path}")
    return archive_path


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniFall segment annotations and, optionally, the 9.7 GB synthetic-video archive."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/omnifall"))
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N annotation rows.")
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Also download the 9.7 GB synthetic-video archive (requires matching free disk space).",
    )
    args = parser.parse_args()

    download_annotations(args.output_dir / "labels-syn.csv", limit=args.limit)
    if args.download_videos:
        download_video_archive(args.output_dir)


if __name__ == "__main__":
    main()
