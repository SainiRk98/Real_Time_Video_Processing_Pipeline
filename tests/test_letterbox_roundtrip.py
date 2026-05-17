import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import preprocess_numpy, inverse_transform, TARGET_SIZE


def make_test_frame(h, w):
    """Create a dummy BGR frame of given size."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def roundtrip(h, w, bbox_orig):
    frame = make_test_frame(h, w)
    _, scale, pad_top, pad_left = preprocess_numpy(frame)

    x1, y1, x2, y2 = bbox_orig
    # Forward: original → letterbox
    lb_x1 = x1 * scale + pad_left
    lb_y1 = y1 * scale + pad_top
    lb_x2 = x2 * scale + pad_left
    lb_y2 = y2 * scale + pad_top

    # Inverse: letterbox → original
    recovered = inverse_transform([lb_x1, lb_y1, lb_x2, lb_y2], scale, pad_top, pad_left)

    error = max(
        abs(recovered[0] - x1),
        abs(recovered[1] - y1),
        abs(recovered[2] - x2),
        abs(recovered[3] - y2),
    )
    return error


def test_roundtrip_landscape():
    error = roundtrip(720, 1280, [100, 50, 400, 300])
    assert error < 1.0, f"Round-trip error too large: {error:.4f}px"


def test_roundtrip_portrait():
    error = roundtrip(1080, 720, [50, 100, 300, 800])
    assert error < 1.0, f"Round-trip error too large: {error:.4f}px"


def test_roundtrip_square():
    error = roundtrip(640, 640, [10, 10, 300, 300])
    assert error < 1.0, f"Round-trip error too large: {error:.4f}px"


def test_roundtrip_4k():
    error = roundtrip(2160, 3840, [200, 100, 1000, 800])
    assert error < 1.0, f"Round-trip error too large: {error:.4f}px"
