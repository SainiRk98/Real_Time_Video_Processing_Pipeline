import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imagehash
from PIL import Image
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "city_highway.mp4")
SAMPLE_FRAMES = 300


def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n)
    frames = []
    for i in range(0, min(total, n * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Resize to 640x360 to save memory (4K → small)
            frame = cv2.resize(frame, (640, 360))
            frames.append(frame)
    cap.release()
    return frames


def main():
    print("Sampling frames from video...")
    frames = sample_frames(VIDEO_PATH, SAMPLE_FRAMES)
    print(f"Sampled {len(frames)} frames")

    blur_scores = []
    luma_scores = []
    clip_ratios = []
    phash_diffs = []
    prev_hash = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        luma_scores.append(gray.mean())
        clip_ratios.append(((gray < 10) | (gray > 245)).mean())

        pil = Image.fromarray(cv2.resize(gray, (64, 64)))
        h = imagehash.phash(pil)
        if prev_hash is not None:
            phash_diffs.append(abs(h - prev_hash))
        prev_hash = h

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(blur_scores, label="Blur (Laplacian Variance)")
    plt.axhline(80, color='red', linestyle='--', label="Threshold=80")
    plt.title("Blur Score per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/blur_scores.png")
    plt.close()
    print("Saved plots/blur_scores.png")

    plt.figure(figsize=(10, 4))
    plt.plot(luma_scores, label="Mean Luma", color='orange')
    plt.axhline(30, color='red', linestyle='--', label="Min=30")
    plt.axhline(220, color='blue', linestyle='--', label="Max=220")
    plt.title("Mean Luma per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Luma")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/luma_scores.png")
    plt.close()
    print("Saved plots/luma_scores.png")

    plt.figure(figsize=(10, 4))
    plt.plot(phash_diffs, label="Perceptual Hash Diff", color='green')
    plt.axhline(5, color='red', linestyle='--', label="Stuck Threshold=5")
    plt.title("Perceptual Hash Diff (Stuck Frame Detection)")
    plt.xlabel("Frame Index")
    plt.ylabel("Hash Diff")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/phash_diffs.png")
    plt.close()
    print("Saved plots/phash_diffs.png")

    print("\n--- Threshold Summary ---")
    print(f"Blur   : min={min(blur_scores):.1f}, mean={np.mean(blur_scores):.1f}, recommended threshold=80")
    print(f"Luma   : min={min(luma_scores):.1f}, max={max(luma_scores):.1f}, recommended range=[30, 220]")
    print(f"Clip   : max={max(clip_ratios):.3f}, recommended max=0.15")
    print(f"Phash  : min_diff={min(phash_diffs)}, recommended stuck threshold=5")


if __name__ == "__main__":
    main()
