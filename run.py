import cv2
import numpy as np
import threading
import time
import signal
import argparse
import json
import sys

from src.camera import CameraThread
from src.slot import FrameSlot
from src.preprocess import preprocess_blob, preprocess_numpy
from src.metrics import MetricsCollector
from src.soak import SoakMonitor
from src.benchmark import run_benchmark

shutdown_event = threading.Event()

def sigint_handler(sig, frame):
    print("\n[INFO] SIGINT received. Shutting down...")
    shutdown_event.set()

signal.signal(signal.SIGINT, sigint_handler)


def consumer(slot_a, slot_b, metrics_a, metrics_b, shutdown_event):
    while not shutdown_event.is_set():
        for slot, metrics in [(slot_a, metrics_a), (slot_b, metrics_b)]:
            item = slot.get()
            if item is None:
                continue
            frame, capture_ts = item
            _ = preprocess_numpy(frame)
            metrics.record_preprocess(capture_ts)
        time.sleep(0.001)


def metrics_reporter(metrics_a, metrics_b, shutdown_event):
    while not shutdown_event.is_set():
        time.sleep(5)
        for name, m in [("cam_a", metrics_a), ("cam_b", metrics_b)]:
            report = m.report()
            print(f"\n--- {name} [t={int(time.time() - m.start_time)}s] ---")
            print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video file")
    args = parser.parse_args()

    # Run benchmark once at startup
    run_benchmark(args.input)

    slot_a = FrameSlot()
    slot_b = FrameSlot()
    metrics_a = MetricsCollector()
    metrics_b = MetricsCollector()

    cam_a = CameraThread("cam_a", args.input, slot_a, metrics_a, shutdown_event)
    cam_b = CameraThread("cam_b", args.input, slot_b, metrics_b, shutdown_event)

    consumer_thread = threading.Thread(
        target=consumer,
        args=(slot_a, slot_b, metrics_a, metrics_b, shutdown_event),
        daemon=True
    )
    reporter_thread = threading.Thread(
        target=metrics_reporter,
        args=(metrics_a, metrics_b, shutdown_event),
        daemon=True
    )

    soak = SoakMonitor(shutdown_event)
    soak.start()

    cam_a.start()
    cam_b.start()
    consumer_thread.start()
    reporter_thread.start()

    print(f"[INFO] Pipeline started. Press Ctrl+C to stop.")

    cam_a.join()
    cam_b.join()
    consumer_thread.join(timeout=2)
    reporter_thread.join(timeout=2)

    print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    main()
