# REPORT.md

## Q1: Real-time throttling without busy-waiting

Frame timestamps are tracked using `time.monotonic()`. After reading each frame, the thread computes `next_frame_time += frame_duration` and calls `shutdown_event.wait(timeout=wait)` to sleep precisely until the next frame is due. This avoids busy-waiting entirely. For variable-FPS input, the per-frame duration would be derived from the PTS (presentation timestamp) embedded in the container rather than a fixed `1/fps` value — each frame's actual display time would be used to compute the wait, making pacing accurate even when frame intervals vary.

## Q2: Latest-frame-wins deadlock proof

The `FrameSlot` uses a single `threading.Lock` protecting one variable. The producer calls `put()` which acquires the lock, overwrites the slot, and releases. The consumer calls `get()` which acquires the lock, reads and clears the slot, and releases. Since neither side blocks waiting for the other (no condition variables, no semaphores), deadlock is impossible regardless of relative speeds. If the consumer is faster: it gets `None` and skips. If the producer is faster: old frames are silently overwritten. If equal speed: every frame is consumed once.

## Q3: cv2.dnn.blobFromImage vs numpy

`cv2.dnn.blobFromImage` is fast because it is implemented in optimized C++ with SIMD intrinsics, performs BGR→RGB swap, resize, normalization, and NCHW transpose in a single fused pass with minimal memory allocation. The hand-written numpy version loses because it performs these steps sequentially in Python, each creating an intermediate array copy — typically 2–4× slower on large frames (measured in `scripts/pick_thresholds.py` benchmark).

## Q4: VideoCapture.read() on decoding glitch

`VideoCapture.read()` returns `(False, None)` on a true EOF or unrecoverable decode error, and may return `(True, garbage_frame)` on a partial decode glitch. The pipeline treats `ret=False` as recoverable by seeking to frame 0 and incrementing `recovery_count`. Garbage frames (ret=True but visually corrupt) are caught downstream by the quality gate — blur and exposure checks will reject them before they reach the consumer.

## Q5: Least confident measurement

The p99 frame age is the measurement I am least confident about. It is computed over a rolling window of the last 1000 frames, which may not capture rare tail latency spikes during GC pauses or OS scheduling jitter. To strengthen it, I would log every frame age to a file and compute p99 post-hoc over the full 10-minute run, and also plot a CDF to visualize the full latency distribution rather than relying on a single percentile.
