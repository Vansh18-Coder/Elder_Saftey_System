import time

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - ImportError is handled at runtime.
    mp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from src.camera.webcam import open_camera


class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        if mp is None:
            raise RuntimeError(
                "mediapipe is not installed. Run: python -m pip install -r requirements.txt"
            ) from _IMPORT_ERROR

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks is None:
            return None, None

        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(
                {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "visibility": float(getattr(landmark, "visibility", 0.0)),
                }
            )
        return landmarks, results

    def draw_landmarks(self, frame, results):
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )
        return frame

    def close(self):
        self.pose.close()


def run_pose_demo(source=0, width=640, height=480, max_frames=None):
    """Open the webcam and display a live skeleton overlay."""
    cap = open_camera(source=source, width=width, height=height)
    detector = PoseDetector()

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read a frame from the webcam. Check the camera connection.")
                break

            frame_count += 1
            landmarks, results = detector.process_frame(frame)

            if landmarks is None:
                status_text = "No person detected"
            else:
                status_text = "Person detected"
                frame = detector.draw_landmarks(frame, results)

            fps = 1.0 / max((time.perf_counter() - start_time) / max(frame_count, 1), 1e-6)
            cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Elderly Safety - Pose Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    print(f"Session closed after {frame_count} frames.")


def main():
    run_pose_demo(source=0, width=640, height=480)


if __name__ == "__main__":
    main()
