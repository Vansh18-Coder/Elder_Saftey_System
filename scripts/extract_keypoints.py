import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.mediapipe_pose import PoseDetector


def extract_keypoints_from_capture(source=0, width=640, height=480, limit=30, output_csv=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {source}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = PoseDetector()
    rows = []
    frame_index = 0

    try:
        while frame_index < limit:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, results = detector.process_frame(frame)
            if landmarks is None:
                rows.append({"frame_index": frame_index, "person_detected": 0, **{f"feature_{i}": 0.0 for i in range(132)}})
            else:
                feature_vector = np.asarray([
                    value for landmark in landmarks for value in [
                        landmark["x"],
                        landmark["y"],
                        landmark["z"],
                        landmark["visibility"],
                    ]
                ], dtype=np.float32)
                if feature_vector.shape[0] != 132:
                    feature_vector = np.zeros(132, dtype=np.float32)
                row = {"frame_index": frame_index, "person_detected": 1}
                for idx, value in enumerate(feature_vector):
                    row[f"feature_{idx}"] = float(value)
                rows.append(row)

            frame_index += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    dataframe = pd.DataFrame(rows)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        print(f"Saved {len(dataframe)} frames to {output_path}")
    else:
        print(dataframe.head())

    return dataframe


def main():
    parser = argparse.ArgumentParser(description="Extract pose keypoints from webcam frames")
    parser.add_argument("--source", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--limit", type=int, default=30, help="Number of frames to capture")
    parser.add_argument("--output", type=str, default="data/processed/sample_pose_keypoints.csv", help="CSV output path")
    args = parser.parse_args()

    extract_keypoints_from_capture(
        source=args.source,
        width=args.width,
        height=args.height,
        limit=args.limit,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
