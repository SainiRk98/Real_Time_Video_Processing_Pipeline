import cv2
import numpy as np
import imagehash
from PIL import Image

# Thresholds justified in notebooks/thresholds.ipynb
BLUR_THRESHOLD = 80.0        # variance of Laplacian
LUMA_MIN = 30.0              # minimum mean luma
LUMA_MAX = 220.0             # maximum mean luma
CLIP_RATIO_MAX = 0.15        # max ratio of clipped pixels
PHASH_DIFF_MIN = 2           # min perceptual hash diff to detect stuck frame (city highway slow scenes)


class QualityGate:
    def __init__(self):
        self._prev_hash = None

    def check(self, frame):
        result = {
            "accepted": True,
            "blur": False,
            "exposure": False,
            "stuck": False,
        }

        # --- Blur check ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < BLUR_THRESHOLD:
            result["blur"] = True
            result["accepted"] = False

        # --- Exposure check ---
        luma = gray.mean()
        clipped = ((gray < 10) | (gray > 245)).mean()
        if luma < LUMA_MIN or luma > LUMA_MAX or clipped > CLIP_RATIO_MAX:
            result["exposure"] = True
            result["accepted"] = False

        # --- Stuck frame check ---
        pil_img = Image.fromarray(cv2.resize(gray, (64, 64)))
        current_hash = imagehash.phash(pil_img)
        if self._prev_hash is not None:
            diff = abs(current_hash - self._prev_hash)
            if diff < PHASH_DIFF_MIN:
                result["stuck"] = True
                result["accepted"] = False

        if result["accepted"]:
            self._prev_hash = current_hash

        return result
