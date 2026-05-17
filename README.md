# Video Processing Pipeline

## What is this Project?

This is a production-grade real-time video processing pipeline that simulates two independent camera streams from a single video file. It reads frames from both cameras simultaneously, applies quality filtering, preprocesses frames for AI model consumption, and emits live health metrics every 5 seconds. The system is designed to run continuously for extended periods (10+ minutes) with stable memory usage and clean shutdown behavior.

---

## Project Structure

```
capsitech_task/
├── run.py                          # Entry point
├── src/
│   ├── camera.py                   # Camera thread (reads frames)
│   ├── slot.py                     # Latest-frame-wins mechanism
│   ├── quality.py                  # Quality gate (blur/exposure/stuck)
│   ├── preprocess.py               # Letterbox + NCHW preprocessing
│   ├── metrics.py                  # JSON metrics collector
│   ├── benchmark.py                # blob vs numpy benchmark
│   └── soak.py                     # RSS memory monitor
├── tests/
│   └── test_letterbox_roundtrip.py # Unit tests
├── scripts/
│   └── pick_thresholds.py          # Threshold experiment + plots
├── plots/
│   ├── blur_scores.png             # Blur threshold experiment
│   ├── luma_scores.png             # Exposure threshold experiment
│   ├── phash_diffs.png             # Stuck frame threshold experiment
│   ├── benchmark.png               # blob vs numpy speed comparison
│   └── rss_soak.png                # Memory usage over soak test
├── REPORT.md                       # Written answers
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## Process Flow

```
INPUT: video file (e.g. city_highway.mp4)
         |
         ├──────────────────────────────────┐
         ↓                                  ↓
   Thread: cam_a                      Thread: cam_b
   cv2.VideoCapture()                 cv2.VideoCapture()
   Real-time pacing (FPS)             Real-time pacing (FPS)
   EOF → loop back to start           EOF → loop back to start
         |                                  |
         ↓                                  ↓
   Quality Gate                       Quality Gate
   - Blur check (Laplacian)           - Blur check (Laplacian)
   - Exposure check (luma)            - Exposure check (luma)
   - Stuck frame (phash)              - Stuck frame (phash)
   Rejected → dropped counter         Rejected → dropped counter
         |                                  |
         ↓                                  ↓
   FrameSlot (latest-frame-wins)      FrameSlot (latest-frame-wins)
         |                                  |
         └──────────────┬───────────────────┘
                        ↓
                 Consumer Thread
                 preprocess_numpy()
                 Letterbox 640x640
                 NCHW float32
                        |
                        ↓
OUTPUT: JSON Metrics every 5s (terminal)
        RSS Memory Plot (plots/rss_soak.png)
```

---

## Input

| Field | Detail |
|---|---|
| Type | Video file |
| Format | `.mp4` |
| Command | `--input <filename>` |
| Example | `city_highway.mp4`, `test_1080p.mp4` |
| Resolution | Any (4K will be resized to 1280x720 automatically) |

---

## Output

### 1. Terminal — Benchmark (once at startup)
```
=== Preprocessing Benchmark ===
frames tested    : 100
blob  avg        : 6.95 ms/frame
numpy avg        : 9.31 ms/frame
numpy is 1.3x slower than blob
================================
```

### 2. Terminal — JSON Metrics (every 5 seconds per stream)
```json
--- cam_a [t=5s] ---
{
  "fps_capture": 13.2,
  "fps_preprocess": 13.2,
  "frame_age_ms_p50": 47.8,
  "frame_age_ms_p99": 66.2,
  "frames_dropped": 182,
  "recovery_count": 0,
  "quality_counters": {
    "blur": 0,
    "exposure": 0,
    "stuck": 182
  },
  "rss_mb": 326.17
}
```

### 3. Plots saved to `plots/` folder
| File | What it shows |
|---|---|
| `benchmark.png` | blob vs numpy speed comparison |
| `rss_soak.png` | Memory usage over entire run |
| `blur_scores.png` | Blur threshold experiment |
| `luma_scores.png` | Exposure threshold experiment |
| `phash_diffs.png` | Stuck frame threshold experiment |

### 4. Unit Test Result
```
4/4 PASSED — round-trip error < 1px
```

---

## How to Run

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run Pipeline
```bash
python run.py --input city_highway.mp4
```

### Step 3 — Stop (after 10 minutes)
```
Ctrl+C
```
Expected output:
```
[INFO] SIGINT received. Shutting down...
[INFO] cam_a: Camera thread stopped.
[INFO] cam_b: Camera thread stopped.
[INFO] Clean shutdown complete.
[INFO] RSS plot saved to plots/rss_soak.png
```

### Step 4 — Run Unit Tests
```bash
python -m pytest tests/test_letterbox_roundtrip.py -v
```

### Step 5 — Run Threshold Experiment
```bash
python scripts/pick_thresholds.py
```

---

## Hard Constraints Met

| Constraint | How |
|---|---|
| No GPU / No Cloud / No LLM | Only cv2, numpy, psutil used |
| No unbounded queue.Queue | FrameSlot holds max 1 frame |
| No time.sleep(1/fps) | Frame timestamps used for pacing |
| No while True: except: pass | shutdown_event used for control |
| Less than 600 LOC | ~300 LOC total |

---

## Dependencies

```
opencv-python
numpy
imagehash
Pillow
psutil
pytest
matplotlib
```
