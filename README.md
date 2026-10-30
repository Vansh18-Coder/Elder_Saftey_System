# Elderly Safety System

This project is a privacy-conscious elderly fall-detection and emergency-assistance prototype. The implementation starts on a Windows laptop with a webcam and a Python virtual environment, then expands toward a GRU-based temporal classifier, evaluation workflow, and real-time assistance logic.

The repository currently covers environment setup, webcam verification, pose estimation, keypoint extraction, sequencing, model training, evaluation, and benchmark-dataset preparation.

## Project objective

Build a working laptop prototype that:

- opens the webcam,
- detects a visible human body,
- draws skeleton landmarks,
- extracts keypoints in a repeatable format,
- prepares the data pipeline for later GRU training,
- keeps privacy-friendly skeleton-only processing by default.

## Working directory

Use the following directory for all commands in this phase:

```powershell
cd C:\Users\vansh\Desktop\ElderSafteySystem
```

## 1. Check Python and pip

Run these commands in PowerShell:

```powershell
cd C:\Users\vansh\Desktop\ElderSafteySystem
py --version
py -m pip --version
```

Expected output example:

```text
Python 3.11.x
pip 24.x
```

Important compatibility note: MediaPipe is reliable on Python 3.11 for this project. If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, the active environment is using an unsupported Python build or a partially installed MediaPipe package. In that case, recreate the virtual environment with Python 3.11 and reinstall the project dependencies.

If `py` is not recognized, open a new VS Code terminal and confirm Python is installed in PATH, or use the Python launcher from VS Code.

## 2. Create and activate a virtual environment

```powershell
cd C:\Users\vansh\Desktop\ElderSafteySystem
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then verify the interpreter:

```powershell
python --version
python -m pip --version
```

## 3. Install initial dependencies

Create the project requirements file before installing:

```powershell
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe numpy pandas pytest
```

The project also includes a file with the exact dependency list:

```powershell
python -m pip install -r requirements.txt
```

## 4. Verify package imports

```powershell
python -c "import cv2, mediapipe, numpy, pandas; print('OpenCV, MediaPipe, NumPy, Pandas import OK')"
```

Expected output:

```text
OpenCV, MediaPipe, NumPy, Pandas import OK
```

## 5. Test the webcam

Use a short Python command:

```powershell
python -c "import cv2; cap = cv2.VideoCapture(0); print('camera opened:', cap.isOpened()); cap.release()"
```

Expected output:

```text
camera opened: True
```

If the camera is not detected, check:

- the webcam is not being used by another application,
- the system has permission to access the webcam,
- the camera index is not `0` if another device is active,
- the camera driver is installed correctly.

## 6. First pose-estimation program

The first working script is:

- `src/pose/mediapipe_pose.py`

This script opens the webcam, runs MediaPipe pose estimation, draws landmarks, and displays a live skeleton overlay. It also prints a useful status message when no person is detected.

## 7. Run the pose-estimation demo

```powershell
cd C:\Users\vansh\Desktop\ElderSafteySystem
python app.py --demo pose --source 0
```

### Controls

- Press `q` to quit.
- The window will show the live webcam video.
- If a person is detected, a skeleton overlay will be drawn.
- If no person is detected, the status text will tell you so.

## 8. Additional pipeline stages

The application entry point now supports the later project phases as well:

```powershell
# Extract skeleton features to CSV
python app.py --demo extract --source 0 --limit 30 --output data/processed/sample_pose_keypoints.csv

# Build a fixed-length sequence dataset from labelled CSVs
python app.py --demo build-dataset --data-root data/raw/demo_samples --sequence-length 30 --output-dir data/sequences/demo_dataset

# Download or prepare a public benchmark dataset from a real fall repository
python app.py --demo download-benchmark --source-root data/raw --target-root data/raw/benchmark --dataset ur-fall --dry-run

# If you already have a zip or local dataset copy, point it at the archive URL
python app.py --demo download-benchmark --source-root data/raw --target-root data/raw/benchmark --dataset ur-fall --download-url "https://example.com/fall_dataset.zip"

# Download the official UR Fall camera-0 RGB videos into data/raw/ur_fall
python app.py --demo download-ur-benchmark --data-root data/raw/ur_fall

# Convert the downloaded MP4s into the project's benchmark normal/fall pose CSV layout
python app.py --demo convert-ur-benchmark --data-root data/raw/ur_fall --target-root data/raw/benchmark

# Convert extracted UP-Fall camera image sequences (see data/README.md for download and folder-layout details)
python app.py --demo convert-up-fall-camera --data-root data/raw/up_fall_camera --target-root data/raw/up_fall_pose --frame-step 1

# Download OmniFall segment annotations (optional 9.7 GB video download requires --download-videos)
python app.py --demo download-omnifall --data-root data/raw/omnifall

# Convert extracted OmniFall videos to normal/fall pose CSVs
python app.py --demo convert-omnifall --data-root data/raw/omnifall/videos --annotations data/raw/omnifall/labels-syn.csv --target-root data/raw/omnifall_pose --frame-step 2

# Organize raw CSVs into benchmark/normal and benchmark/fall folders for real-data preparation
python app.py --demo organize-benchmark --data-root data/raw --target-root data/raw/benchmark

# Run the full benchmark pipeline automatically from raw data to trained model
python app.py --demo full --data-root data/raw --target-root data/raw/benchmark --sequence-length 10 --output-dir data/sequences/benchmark_dataset --batch-size 2 --epochs 3

# Prepare a real benchmark dataset from a normal/ and fall/ directory layout
python app.py --demo prepare-real-data --data-root data/raw/benchmark --sequence-length 30 --output-dir data/sequences/benchmark_dataset

# Train the GRU classifier
python app.py --demo train-gru --data-dir data/sequences/demo_dataset --epochs 25 --batch-size 8

# Train on the prepared UR Fall sequences and calibrate a fall threshold for at least 95% validation precision
python scripts/train_gru.py --data-dir data/sequences/ur_fall --epochs 25 --batch-size 64 --target-precision 0.95 --output-dir models/ur_fall_precision

# Evaluate the trained classifier on the test split
python scripts/evaluate_model.py --data-dir data/sequences/ur_fall --model-path models/ur_fall_precision/gru_fall_classifier.pt --split test

# Run the real-time inference demo (uses the precision-calibrated threshold saved with the model)
python scripts/run_inference_demo.py --model-path models/ur_fall_precision/gru_fall_classifier.pt --source 0 --sequence-length 30

# Launch a simple alert/status UI
python app.py --demo ui
```

The UR downloader retrieves the camera-0 RGB videos listed on the official dataset page (30 falls and 40 activities of daily living). The dataset page states a CC BY-NC-SA 4.0 license; review and follow its terms before using or redistributing the data. Downloading the dataset is a separate step because the files are research data and are not bundled with this repository.

If the dataset has already been downloaded manually, put `fall-XX-cam0.mp4` and `adl-XX-cam0.mp4` files directly under `data/raw/ur_fall`, or extract RGB frame archives into folders whose paths contain `fall` or `adl`. Then run the converter. The converter rejects unknown labels rather than silently treating them as normal.

## 9. Evaluation workflow

The project includes a benchmark-oriented evaluation script that loads the saved GRU checkpoint and prints standard metrics for the selected split:

- accuracy
- precision
- recall
- F1 score
- specificity
- confusion matrix

This supports an honest model-review step before discussing real-world deployment readiness.

## 10. What this phase proves

By the end of this stage, you should have verified that:

- Python is installed and working,
- the virtual environment works,
- required packages import cleanly,
- the webcam opens successfully,
- MediaPipe detects a human body and draws skeleton landmarks,
- pose keypoints can be extracted for later GRU data preparation.

## 9. Keypoint schema used for the next stage

MediaPipe Pose outputs a 33-landmark skeleton. Each landmark typically contains:

- `x`: normalized horizontal coordinate,
- `y`: normalized vertical coordinate,
- `z`: depth relative to the hip center,
- `visibility`: confidence value, usually in `[0, 1]`.

A compact flattened feature vector is created as:

```text
[x1, y1, z1, visibility1, x2, y2, z2, visibility2, ..., x33, y33, z33, visibility33]
```

This gives a feature dimension of `33 * 4 = 132` values when all visibility values are included.

The exact number of features depends on the chosen landmark schema. Do not assume a fixed dimension without checking which joints are being used for the model.

## 10. Privacy note

This prototype keeps processing focused on skeletal landmarks rather than storing raw video by default. That is a privacy-friendly approach because it reduces raw visual data retention while still enabling movement analysis.

## 11. Project structure

```text
ElderSafteySystem/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── webcam.py
│   ├── pose/
│   │   ├── __init__.py
│   │   └── mediapipe_pose.py
│   └── preprocessing/
│       ├── __init__.py
│       └── keypoints.py
├── tests/
│   └── test_keypoints.py
└── .venv/
```

## 12. Common issues and fixes

### Problem: `cv2.VideoCapture(0)` fails

Check:

- the webcam is not already open in another app,
- VS Code has camera access,
- Windows camera privacy settings are enabled,
- the camera index is not wrong.

### Problem: `ImportError: No module named mediapipe`

Run:

```powershell
python -m pip install -r requirements.txt
```

### Problem: MediaPipe has no person detected

Possible reasons:

- not enough light,
- subject is too far away,
- subject is partly off-camera,
- body is occluded,
- camera view is too cluttered.

### Problem: script exits immediately

Check the console for the exact error message and re-run the command in the same virtual environment.

## 13. Checkpoint before moving to the next phase

Do not continue until all of the following are completed:

- [ ] Python version confirmed
- [ ] Virtual environment created
- [ ] Dependencies installed successfully
- [ ] `cv2` imports correctly
- [ ] `mediapipe` imports correctly
- [ ] webcam opens successfully
- [ ] live pose overlay works
- [ ] script exits cleanly when `q` is pressed

If any item fails, fix the exact root cause before moving to dataset preparation or GRU training.

## 14. Next phase

The next phase will cover:

- keypoint extraction and normalization,
- sequence generation,
- dataset preparation,
- GRU model architecture,
- training and evaluation.

This is intentionally delayed until the camera and pose pipeline are confirmed to work.

## Run the browser-based local fall-detection app

The browser page can run the trained model by connecting to a Python service on the same computer. From the project root, activate the environment and start the service:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_web_app.py
```

Open `http://127.0.0.1:8000` in a browser, press **Start fall detection**, and allow camera access. The page sends individual camera frames over localhost to MediaPipe and the GRU model. Frames are processed in memory and are not saved or uploaded. The detector needs 30 visible-person frames before its first prediction. The alert-threshold slider changes the active model threshold; lowering it can increase false alerts and does not guarantee that falls will be detected.

The server binds to `127.0.0.1` only. Opening `web-demo\index.html` directly or visiting its static Vercel deployment does not connect to the local model; run this Python service to use real inference in the browser. A public deployment with inference would need a separately hosted backend and should not send sensitive camera data to a remote service without an explicit privacy and security design.

This is still an experimental prototype, not an emergency-monitoring product. At the checkpoint's saved 0.75 threshold, a test-split evaluation reported 41.6% recall and 70.6% precision, so it missed more than half of the labeled falls. Do not rely on it for real-world safety or emergency response.
