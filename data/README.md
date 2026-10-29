# Dataset setup

Datasets are not bundled in this repository. Download them from their official
sources and follow the applicable terms of use.

## UP-Fall camera images

The [official UP-Fall dataset page](https://sites.google.com/up.edu.mx/har-up/)
provides camera archives per subject, activity, and trial. The camera images can
be converted to the pose-keypoint CSV format used by this project:

1. Download the camera archives from the official page and extract them locally.
2. Keep the subject, activity, and trial directory names in the extracted paths.
   The activity directory must identify its number, for example
   `subject-01/activity-07/trial-01/Camera1/`.
3. Convert the extracted image sequences:

   ```powershell
   python app.py --demo convert-up-fall-camera --data-root data/raw/up_fall_camera --target-root data/raw/up_fall_pose --frame-step 1
   ```

   Use a larger `--frame-step` to process fewer frames.
4. Build train/validation/test sequences from the resulting pose CSVs:

   ```powershell
   python app.py --demo prepare-real-data --data-root data/raw/up_fall_pose --output-dir data/sequences/up_fall_camera
   ```

UP-Fall activities 1-6 are treated as non-fall and activities 7-11 as fall.
The converter writes the compatible 132-value-per-frame pose features under
`normal/` and `fall/`. This camera path does not use UP-Fall's separate raw
sensor CSV data. When both camera views from a trial are included, they are
kept in the same train/validation/test split.

If you use UP-Fall, cite:

> Lourdes Martínez-Villaseñor, Hiram Ponce, Jorge Brieva, Ernesto Moya-Albor,
> José Núñez-Martínez, and Carlos Peñafort-Asturiano, “UP-Fall Detection
> Dataset: A Multimodal Approach,” *Sensors* 19(9), 1988 (2019),
> https://doi.org/10.3390/s19091988.

## OmniFall

[OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) provides
time-stamped, 16-class action annotations and video resources. This project's
binary fall classifier treats the `fall` and `fallen` action labels as falls;
all other annotated actions are treated as normal. Its annotations are
available under CC BY-NC 4.0. Review the license and the dataset card before
using or redistributing the data. This integration uses OmniFall's synthetic
OF-Synthetic subset; the other configurations annotate external datasets whose
video files are not included in the synthetic archive.

Download the compact synthetic segment annotations first:

```powershell
python scripts/download_omnifall.py --output-dir data/raw/omnifall
```

The synthetic video archive is about 9.7 GB and is not downloaded by default.
Downloading requires at least that much free space; extracting it requires
additional space for the video files, so plan for about 20 GB free on the
destination drive before downloading and extracting. The destination must be
on a drive that exists. For example, on this machine the `D:` drive is
available; use a different existing drive letter if needed. To explicitly
download:

```powershell
python scripts/download_omnifall.py --output-dir D:\data\omnifall --download-videos
```

Extract `omnifall-synthetic_av1.tar` into `D:\data\omnifall\videos`, retaining
its internal folder names. Convert
annotated videos to the system's pose CSV format (use `--limit` for a small
smoke test or `--frame-step` to reduce processing time):

```powershell
python scripts/convert_omnifall_dataset.py --annotations D:\data\omnifall\labels-syn.csv --video-root D:\data\omnifall\videos --output-root data/raw/omnifall_pose --frame-step 2
```

Then prepare sequences and train using the existing pipeline:

```powershell
python app.py --demo prepare-real-data --data-root data/raw/omnifall_pose --sequence-length 30 --output-dir data/sequences/omnifall
python app.py --demo train-gru --data-dir data/sequences/omnifall --epochs 25 --batch-size 8
```

The converter uses annotation timestamps to split mixed-action videos into
normal/fall pose CSV segments. The existing source-based split groups all
segments from the same video together to prevent train/test leakage.

## Other datasets

See the project [README](../README.md) for UR Fall download, conversion, and
training commands.
