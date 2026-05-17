import cv2
import numpy as np

TARGET_SIZE = 640


def letterbox(frame):
    """Resize frame to 640x640 with letterboxing (black padding)."""
    h, w = frame.shape[:2]
    scale = TARGET_SIZE / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_top = (TARGET_SIZE - new_h) // 2
    pad_bottom = TARGET_SIZE - new_h - pad_top
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_right = TARGET_SIZE - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded, scale, pad_top, pad_left


def preprocess_blob(frame):
    """Preprocess using cv2.dnn.blobFromImage — NCHW float32."""
    lb, scale, pad_top, pad_left = letterbox(frame)
    blob = cv2.dnn.blobFromImage(lb, scalefactor=1/255.0, size=(TARGET_SIZE, TARGET_SIZE), swapRB=True)
    return blob, scale, pad_top, pad_left


def preprocess_numpy(frame):
    """Preprocess using hand-written numpy — NCHW float32."""
    lb, scale, pad_top, pad_left = letterbox(frame)
    rgb = lb[:, :, ::-1]                          # BGR → RGB
    arr = rgb.astype(np.float32) / 255.0          # normalize
    nchw = np.transpose(arr, (2, 0, 1))[np.newaxis]  # HWC → NCHW
    return nchw, scale, pad_top, pad_left


def inverse_transform(bbox_lb, scale, pad_top, pad_left):
    """
    Convert bbox from letterbox coords to original source coords.
    bbox_lb: [x1, y1, x2, y2] in letterbox space
    Returns: [x1, y1, x2, y2] in original frame space
    """
    x1, y1, x2, y2 = bbox_lb
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale
    return [x1, y1, x2, y2]
