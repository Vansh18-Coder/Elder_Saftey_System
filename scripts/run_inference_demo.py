import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.realtime_predictor import RealTimePredictor
from src.pose.mediapipe_pose import PoseDetector
from src.preprocessing.keypoints import feature_vector_from_landmarks


def main():
    parser = argparse.ArgumentParser(description="Run a GRU-based real-time fall detection demo.")
    parser.add_argument("--model-path", type=str, default="models/ur_fall_precision/gru_fall_classifier.pt", help="Saved model checkpoint.")
    parser.add_argument("--source", type=int, default=0, help="Camera index.")
    parser.add_argument("--sequence-length", type=int, default=30, help="Number of frames in the temporal buffer.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the calibrated fall threshold saved with the model (default: saved threshold, or 0.75 if unavailable).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.source}.")

    predictor = RealTimePredictor(
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        threshold=args.threshold,
    )
    print(f"Using fall confidence threshold: {predictor.threshold:.4f}")
    detector = PoseDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        landmarks, results = detector.process_frame(frame)
        if landmarks is None:
            status = "No person detected"
            cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            frame = detector.draw_landmarks(frame, results)
            vector = feature_vector_from_landmarks(landmarks)
            result = predictor.update(vector)
            if result["status"] == "READY":
                pred_label = "FALL" if result["prediction"] == 1 else "NORMAL"
                cv2.putText(frame, f"Prediction: {pred_label} ({result['confidence']:.2f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Collecting frames: {result['status']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Real-time Fall Detection Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
