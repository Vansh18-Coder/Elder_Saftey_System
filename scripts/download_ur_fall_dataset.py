import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PAGE = "https://fenix.ur.edu.pl/mkepski/ds/uf.html"
VIDEO_BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data/"


def video_filenames():
    return [
        f"{label}-{index:02d}-cam0.mp4"
        for label, count in (("fall", 30), ("adl", 40))
        for index in range(1, count + 1)
    ]


def download_ur_fall_dataset(output_root: Path, limit: int | None = None):
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    filenames = video_filenames()
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than zero.")
        filenames = filenames[:limit]

    print(f"Downloading official UR Fall camera-0 RGB videos from {DATASET_PAGE}")
    print("Dataset license: CC BY-NC-SA 4.0; use only in accordance with its terms.")

    for position, filename in enumerate(filenames, start=1):
        destination = output_root / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[{position}/{len(filenames)}] Already present: {filename}")
            continue

        temporary_path = destination.with_suffix(destination.suffix + ".part")
        url = VIDEO_BASE_URL + filename
        print(f"[{position}/{len(filenames)}] Downloading {filename}")
        try:
            with urlopen(url, timeout=60) as response, temporary_path.open("wb") as target:
                shutil.copyfileobj(response, target)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

        if temporary_path.stat().st_size == 0:
            temporary_path.unlink()
            raise RuntimeError(f"Downloaded an empty file from {url}")
        temporary_path.replace(destination)

    print(f"Downloaded/verified {len(filenames)} camera-0 videos in {output_root}")
    return output_root


def main():
    parser = argparse.ArgumentParser(
        description="Download the official UR Fall camera-0 RGB videos for local research use."
    )
    parser.add_argument("--output-root", type=str, default="data/raw/ur_fall", help="Directory to store downloaded MP4 files.")
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N videos (useful for a smoke test).")
    args = parser.parse_args()

    download_ur_fall_dataset(Path(args.output_root), limit=args.limit)


if __name__ == "__main__":
    main()
