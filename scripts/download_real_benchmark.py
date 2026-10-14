import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.organize_benchmark_dataset import organize_benchmark_dataset

DATASET_GUIDES = {
    "ur-fall": {
        "label": "UR Fall Detection",
        "url": "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
        "notes": "Official dataset page with the public benchmark description and citation details.",
    },
    "up-fall": {
        "label": "UP-Fall Detection",
        "url": "https://sites.google.com/up.edu.mx/har-up/",
        "notes": "Official project page with the dataset files and complete-download links.",
    },
}


def ensure_dataset_structure(root: Path):
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"The benchmark root does not exist yet: {root}")

    has_label_dirs = (root / "normal").exists() and (root / "fall").exists()
    if has_label_dirs:
        return root

    csv_files = sorted(root.rglob("*.csv"))
    if csv_files:
        return root

    raise FileNotFoundError(
        "No benchmark data found. Expected either a normal/ and fall/ folder layout or CSV files under the target dataset root."
    )


def download_archive(url: str, destination_root: Path):
    destination_root.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    archive_name = Path(parsed.path).name or "benchmark_dataset.zip"
    archive_path = destination_root / archive_name
    print(f"Downloading benchmark archive from: {url}")
    urlretrieve(url, str(archive_path))
    return archive_path


def extract_archive(archive_path: Path, destination_root: Path):
    print(f"Extracting: {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(destination_root)


def main():
    parser = argparse.ArgumentParser(description="Download or prepare a public fall-detection benchmark dataset for this project.")
    parser.add_argument("--dataset", choices=["ur-fall", "up-fall"], default="ur-fall", help="Public benchmark to use.")
    parser.add_argument("--download-url", type=str, default="", help="Optional direct archive URL to download.")
    parser.add_argument("--source-root", type=str, default="data/raw", help="Local root to inspect or extract downloaded benchmark data into.")
    parser.add_argument("--output-root", type=str, default="data/raw/benchmark", help="Folder that should end up containing normal/ and fall/ data.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expected benchmark steps without downloading or extracting.")
    args = parser.parse_args()

    dataset = DATASET_GUIDES[args.dataset]
    output_root = PROJECT_ROOT / args.output_root
    source_root = PROJECT_ROOT / args.source_root

    print(f"Benchmark target: {dataset['label']}")
    print(f"Official page: {dataset['url']}")
    print(dataset["notes"])

    if args.dry_run:
        print("Dry run only. No network access or file extraction was performed.")
        return

    if args.download_url:
        archive_path = download_archive(args.download_url, source_root)
        extract_archive(archive_path, source_root)
        extracted_root = source_root
        if not (extracted_root / "normal").exists() and not (extracted_root / "fall").exists():
            print("Downloaded archive extracted. Organizing the benchmark layout into normal/ and fall/ folders...")
            summary = organize_benchmark_dataset(extracted_root, output_root, dry_run=False)
            print(f"normal_csvs={summary['normal_files']} fall_csvs={summary['fall_files']} skipped={summary['skipped_files']}")
            return

    if (output_root / "normal").exists() and (output_root / "fall").exists():
        print(f"Benchmark data already available at: {output_root.resolve()}")
        return

    source_dir = source_root
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)

    print("No local benchmark data found. To continue, place your downloaded dataset under the expected structure:")
    print(f"  {output_root / 'normal'}")
    print(f"  {output_root / 'fall'}")
    print("or provide a direct --download-url to a zipped public benchmark and this script will extract it for you.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
