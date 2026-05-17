import time
import threading
import psutil
import os


class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._capture_times = []
        self._preprocess_times = []
        self._frame_ages = []
        self._dropped = 0
        self._recovery_count = 0
        self._quality_counters = {"blur": 0, "exposure": 0, "stuck": 0}

    def record_capture(self):
        with self._lock:
            self._capture_times.append(time.monotonic())
            # Keep only last 500 to prevent unbounded growth
            if len(self._capture_times) > 500:
                self._capture_times = self._capture_times[-500:]

    def record_preprocess(self, capture_ts):
        now = time.monotonic()
        age_ms = (now - capture_ts) * 1000
        with self._lock:
            self._preprocess_times.append(now)
            self._frame_ages.append(age_ms)
            # Keep only last 500 to prevent unbounded growth
            if len(self._preprocess_times) > 500:
                self._preprocess_times = self._preprocess_times[-500:]
            if len(self._frame_ages) > 500:
                self._frame_ages = self._frame_ages[-500:]

    def record_drop(self):
        with self._lock:
            self._dropped += 1

    def record_recovery(self):
        with self._lock:
            self._recovery_count += 1

    def record_quality(self, result):
        with self._lock:
            if result["blur"]:
                self._quality_counters["blur"] += 1
            if result["exposure"]:
                self._quality_counters["exposure"] += 1
            if result["stuck"]:
                self._quality_counters["stuck"] += 1

    def _fps(self, times, window=5.0):
        now = time.monotonic()
        recent = [t for t in times if now - t <= window]
        return round(len(recent) / window, 2)

    def _percentile(self, data, p):
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return round(sorted_data[min(idx, len(sorted_data) - 1)], 2)

    def report(self):
        with self._lock:
            rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            ages = list(self._frame_ages[-1000:])
            report = {
                "fps_capture": self._fps(self._capture_times),
                "fps_preprocess": self._fps(self._preprocess_times),
                "frame_age_ms_p50": self._percentile(ages, 50),
                "frame_age_ms_p99": self._percentile(ages, 99),
                "frames_dropped": self._dropped,
                "recovery_count": self._recovery_count,
                "quality_counters": dict(self._quality_counters),
                "rss_mb": round(rss, 2),
            }
        return report
