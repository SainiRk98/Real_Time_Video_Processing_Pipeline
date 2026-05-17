import threading


class FrameSlot:
    """
    Latest-frame-wins slot using a lock-protected variable.
    Only the most recent frame is kept — old frames are overwritten.
    Cannot deadlock: producer always writes, consumer always reads latest.
    No unbounded queue used.
    """

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            self._frame = item

    def get(self):
        with self._lock:
            item = self._frame
            self._frame = None
            return item
