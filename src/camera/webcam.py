import cv2


def open_camera(source=0, width=640, height=480):
    """Open a webcam and configure a default resolution."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {source}. Check camera permissions and device availability.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_camera_devices(max_devices=5):
    """Return list of camera indexes that can be opened."""
    available = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available
