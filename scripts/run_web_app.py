from __future__ import annotations

import argparse
import json
import math
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web-demo"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ur_fall_precision" / "gru_fall_classifier.pt"
MAX_FRAME_BYTES = 2 * 1024 * 1024

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.realtime_predictor import RealTimePredictor
from src.pose.mediapipe_pose import PoseDetector
from src.preprocessing.keypoints import feature_vector_from_landmarks


class LocalInferenceServer(HTTPServer):
    predictor: RealTimePredictor
    detector: PoseDetector
    frames_collected: int


class LocalRequestHandler(SimpleHTTPRequestHandler):
    server: LocalInferenceServer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        request_path = urlsplit(self.path).path
        if request_path == "/api/health":
            self._send_json(
                200,
                {
                    "status": "ready",
                    "sequence_length": self.server.predictor.sequence_length,
                    "threshold": self.server.predictor.threshold,
                },
            )
            return
        super().do_GET()

    def do_POST(self):
        request_path = urlsplit(self.path).path
        if request_path == "/api/reset":
            self.server.predictor.reset()
            self.server.frames_collected = 0
            self._send_json(200, {"status": "reset"})
            return
        if request_path == "/api/settings":
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self._send_json(415, {"error": "Send the detector settings as application/json."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "The settings request has an invalid Content-Length."})
                return
            if content_length <= 0 or content_length > 1024:
                self._send_json(400, {"error": "The settings request must contain a small JSON body."})
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "The settings request must contain valid JSON."})
                return
            threshold = payload.get("threshold") if isinstance(payload, dict) else None
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)
                or not 0.0 <= threshold <= 1.0
            ):
                self._send_json(400, {"error": "Threshold must be a finite number between 0 and 1."})
                return
            self.server.predictor.threshold = float(threshold)
            self._send_json(200, {"status": "updated", "threshold": self.server.predictor.threshold})
            return
        if request_path != "/api/predict":
            self._send_json(404, {"error": "Unknown API endpoint."})
            return

        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "image/jpeg":
            self._send_json(415, {"error": "Send each camera frame as an image/jpeg body."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "The frame request has an invalid Content-Length."})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "The frame request is empty."})
            return
        if content_length > MAX_FRAME_BYTES:
            self._send_json(413, {"error": "The camera frame is too large; lower the camera resolution."})
            return

        encoded_frame = self.rfile.read(content_length)
        if len(encoded_frame) != content_length:
            self._send_json(400, {"error": "The camera frame upload was incomplete."})
            return

        frame = cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._send_json(400, {"error": "The request did not contain a valid JPEG frame."})
            return

        landmarks, _ = self.server.detector.process_frame(frame)
        feature_vector = feature_vector_from_landmarks(landmarks)
        if feature_vector is None:
            self.server.predictor.reset()
            self.server.frames_collected = 0
            self._send_json(
                200,
                {
                    "status": "NO_PERSON",
                    "confidence": 0.0,
                    "prediction": 0,
                    "frames_collected": 0,
                    "sequence_length": self.server.predictor.sequence_length,
                    "threshold": self.server.predictor.threshold,
                },
            )
            return

        result = self.server.predictor.update(feature_vector)
        self.server.frames_collected = min(
            self.server.frames_collected + 1,
            self.server.predictor.sequence_length,
        )
        self._send_json(
            200,
            {
                **result,
                "frames_collected": self.server.frames_collected,
                "sequence_length": self.server.predictor.sequence_length,
                "threshold": self.server.predictor.threshold,
            },
        )

    def _send_json(self, status_code: int, payload: dict):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def main():
    parser = argparse.ArgumentParser(
        description="Serve the Elder Safety web page with local MediaPipe and GRU fall inference."
    )
    parser.add_argument("--port", type=int, default=8000, help="Local web server port.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help="Trained GRU model checkpoint path.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Number of visible-person frames required for each prediction.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the model's saved fall threshold.",
    )
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be greater than zero.")

    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        parser.error(f"Model checkpoint not found: {model_path}")

    predictor = RealTimePredictor(
        model_path=model_path,
        sequence_length=args.sequence_length,
        threshold=args.threshold,
    )
    detector = PoseDetector()

    server = LocalInferenceServer(("127.0.0.1", args.port), LocalRequestHandler)
    server.predictor = predictor
    server.detector = detector
    server.frames_collected = 0
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Local fall-detection page: {url}")
    print(f"Model threshold: {predictor.threshold:.4f}")
    print("Camera frames are processed on this computer only. Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the local fall-detection server.")
    finally:
        server.server_close()
        detector.close()


if __name__ == "__main__":
    main()
