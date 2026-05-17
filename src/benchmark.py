import cv2
import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocess import preprocess_blob, preprocess_numpy


def run_benchmark(video_path, n_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720))
        frames.append(frame)
    cap.release()

    if not frames:
        print("[WARN] Benchmark: no frames read")
        return

    # Benchmark blob
    start = time.perf_counter()
    for f in frames:
        preprocess_blob(f)
    blob_avg = (time.perf_counter() - start) / len(frames) * 1000

    # Benchmark numpy
    start = time.perf_counter()
    for f in frames:
        preprocess_numpy(f)
    numpy_avg = (time.perf_counter() - start) / len(frames) * 1000

    ratio = numpy_avg / blob_avg if blob_avg > 0 else 0

    print("\n=== Preprocessing Benchmark ===")
    print(f"frames tested    : {len(frames)}")
    print(f"blob  avg        : {blob_avg:.2f} ms/frame")
    print(f"numpy avg        : {numpy_avg:.2f} ms/frame")
    print(f"numpy is {ratio:.1f}x slower than blob")
    print("================================\n")

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(["cv2.dnn.blob", "numpy"], [blob_avg, numpy_avg], color=["steelblue", "orange"])
    plt.title("Preprocessing Benchmark (ms/frame)")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.savefig("plots/benchmark.png")
    plt.close()
    print("[INFO] Benchmark plot saved to plots/benchmark.png")
