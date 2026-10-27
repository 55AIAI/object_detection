"""Single-owner inference with a bounded, latest-frame mailbox."""
from dataclasses import dataclass
import logging
import threading
import cv2

LOG = logging.getLogger(__name__)
CLIP_INTERVAL = 60


@dataclass(frozen=True)
class Request:
    sequence: int
    generation: int
    frame: object
    roi: object
    confidence: float
    clip_enabled: bool


@dataclass(frozen=True)
class Result:
    sequence: int
    generation: int
    frame: object
    detections: tuple
    labels: dict


def clamp_roi(roi, width, height):
    if roi is None:
        return (0, 0, width, height)
    x1, y1, x2, y2 = roi
    x1, x2 = max(0, min(width, int(x1))), max(0, min(width, int(x2)))
    y1, y2 = max(0, min(height, int(y1))), max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must have positive area inside the frame")
    return x1, y1, x2, y2


class YoloBackend:
    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device

    def reset(self):
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def detect(self, frame, confidence):
        # Let Ultralytics letterbox the original aspect ratio; no stretched 320x180 input.
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  conf=confidence, imgsz=640, device=self.device,
                                  verbose=False)[0]
        if result.boxes is None:
            return []
        out = []
        for index, box in enumerate(result.boxes):
            coordinates = box.xyxy[0].cpu().tolist()
            tid = int(box.id[0]) if box.id is not None else -index - 1
            out.append((*coordinates, tid, self.model.names[int(box.cls[0])],
                        float(box.conf[0])))
        return out


class DetectionModule:
    def __init__(self, backend, clip_module=None, clip_threshold=0.4):
        self.backend, self.clip = backend, clip_module
        self.clip_threshold = clip_threshold
        self._generation = None
        self._cache = {}
        self._cycles = 0
        self._condition = threading.Condition()
        self._pending = self._result = self._error = None
        self._stopping = False
        self._last_submitted = -1
        self._thread = None

    def process(self, request):
        """Synchronous mode for files/tests. Never call alongside the worker."""
        if request.generation != self._generation:
            self.backend.reset()
            self._cache.clear()
            self._cycles = 0
            self._generation = request.generation
        self._cycles += 1
        frame = request.frame
        height, width = frame.shape[:2]
        ox, oy, rx2, ry2 = clamp_roi(request.roi, width, height)
        work = frame[oy:ry2, ox:rx2]
        detections, crops = [], []
        for x1, y1, x2, y2, tid, label, confidence in self.backend.detect(work, request.confidence):
            x1, y1 = max(0, min(work.shape[1], int(x1))), max(0, min(work.shape[0], int(y1)))
            x2, y2 = max(0, min(work.shape[1], int(x2))), max(0, min(work.shape[0], int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, tid, label, confidence))
            crops.append(work[y1:y2, x1:x2])
        active = {d[4] for d in detections if d[4] >= 0}
        self._cache = {tid: value for tid, value in self._cache.items() if tid in active}
        labels = {d[4]: (d[5], d[6]) for d in detections}
        if request.clip_enabled and self.clip is not None:
            selected = [i for i, d in enumerate(detections)
                        if d[4] >= 0 and (d[4] not in self._cache or self._cycles % CLIP_INTERVAL == 0)]
            if selected:
                try:
                    classified = self.clip.classify_batch([crops[i] for i in selected])
                    if len(classified) != len(selected):
                        raise ValueError("CLIP returned an unexpected number of labels")
                    for index, (label, score) in zip(selected, classified):
                        d = detections[index]
                        self._cache[d[4]] = ((label, score) if score > self.clip_threshold
                                             else None)
                except Exception:
                    LOG.warning("CLIP inference failed; using YOLO labels for this frame", exc_info=True)
                    self._cache.clear()
            labels.update({tid: value for tid, value in self._cache.items() if value is not None})
        else:
            self._cache.clear()
        return Result(request.sequence, request.generation, frame, tuple(detections), labels)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Worker already started")
        self._thread = threading.Thread(target=self._run, name="vision-inference", daemon=True)
        self._thread.start()

    def submit(self, request):
        # The main loop transfers its own frame copy. No unbounded queue.
        with self._condition:
            if request.sequence <= self._last_submitted:
                raise ValueError("Frame sequence must increase")
            if self._stopping:
                raise RuntimeError("Worker stopped")
            self._last_submitted = request.sequence
            self._pending = request
            self._condition.notify()

    def latest(self, after=-1):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Detection worker failed") from self._error
            result = self._result
            return result if result is not None and result.sequence > after else None

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    request, self._pending = self._pending, None
                result = self.process(request)
                with self._condition:
                    self._result = result
        except Exception as error:
            with self._condition:
                self._error = error

    def stop(self, timeout=5):
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("Inference did not stop within the shutdown timeout")
