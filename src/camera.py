import cv2
import time
import threading
from src.quality import QualityGate


class CameraThread(threading.Thread):
    def __init__(self, name, path, slot, metrics, shutdown_event):
        super().__init__(name=name, daemon=True)
        self.path = path
        self.slot = slot
        self.metrics = metrics
        self.shutdown_event = shutdown_event
        self.quality = QualityGate()

    def run(self):
        cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[ERROR] {self.name}: Cannot open {self.path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration = 1.0 / fps
        next_frame_time = time.monotonic()
        # For 4K video, decode at half resolution natively
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.shutdown_event.is_set():
            ret, frame = cap.read()

            if not ret:
                # EOF — loop back
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.metrics.record_recovery()
                continue

            capture_ts = time.monotonic()

            # Real-time pacing using frame timestamps
            now = time.monotonic()
            wait = next_frame_time - now
            if wait > 0:
                # Use event wait instead of sleep for clean shutdown
                self.shutdown_event.wait(timeout=wait)
            next_frame_time += frame_duration

            # Resize 4K to 1280x720 to reduce memory pressure
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, 720))

            # Quality gate
            result = self.quality.check(frame)
            self.metrics.record_quality(result)

            if result["accepted"]:
                self.slot.put((frame, capture_ts))
                self.metrics.record_capture()
            else:
                self.metrics.record_drop()

        cap.release()
        print(f"[INFO] {self.name}: Camera thread stopped.")
