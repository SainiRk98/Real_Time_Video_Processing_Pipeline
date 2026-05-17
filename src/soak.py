"""
Soak test monitor — tracks RSS memory over time and saves a plot.
Run alongside run.py or import into it.
"""
import psutil
import os
import time
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SoakMonitor:
    def __init__(self, shutdown_event, interval=10):
        self._shutdown = shutdown_event
        self._interval = interval
        self._times = []
        self._rss = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="soak-monitor")

    def start(self):
        self._start = time.time()
        self._thread.start()

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._shutdown.is_set():
            rss = proc.memory_info().rss / (1024 * 1024)
            elapsed = time.time() - self._start
            with self._lock:
                self._times.append(elapsed)
                self._rss.append(rss)
            self._shutdown.wait(timeout=self._interval)
        self._save_plot()

    def _save_plot(self):
        with self._lock:
            times = list(self._times)
            rss = list(self._rss)
        if len(times) < 2:
            return
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(times, rss, color='blue', label='RSS MB')
        plt.title("Memory (RSS) over Soak Test")
        plt.xlabel("Time (seconds)")
        plt.ylabel("RSS (MB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/rss_soak.png")
        plt.close()
        print("[INFO] RSS plot saved to plots/rss_soak.png")
