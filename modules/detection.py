import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO

YOLO_W, YOLO_H = 320, 180
TARGET_FPS = 25
CLIP_INTERVAL = 60


class DetectionModule:
    def __init__(self, clip_module, state):
        self.clip = clip_module
        self.state = state  # shared AppState

        print("[YOLO] Loading yolov8s...")
        self.model = YOLO("yolov8s.pt")
        self.model.fuse()
        print("[YOLO] Ready.")

        self._yolo_frame_idx = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        s = self.state
        while s.running:
            with s.frame_lock:
                frame = s.latest_frame

            if frame is None or s.paused:
                time.sleep(0.005)
                continue

            self._yolo_frame_idx += 1
            if self._yolo_frame_idx % s.yolo_skip != 0:
                time.sleep(0.001)
                continue

            orig_h, orig_w = frame.shape[:2]
            sx = orig_w / YOLO_W
            sy = orig_h / YOLO_H

            # Crop to ROI before detection if set
            roi = s.roi
            if roi:
                rx1, ry1, rx2, ry2 = roi
                work = frame[ry1:ry2, rx1:rx2]
                ox, oy = rx1, ry1
            else:
                work = frame
                ox, oy = 0, 0

            small = cv2.resize(work, (YOLO_W, YOLO_H))
            rsx = work.shape[1] / YOLO_W
            rsy = work.shape[0] / YOLO_H

            results = self.model.track(
                small, persist=True, conf=s.conf_threshold, verbose=False
            )[0]

            boxes_data = []
            if results.boxes is not None:
                for box in results.boxes:
                    sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                    x1 = int(sx1 * rsx) + ox
                    y1 = int(sy1 * rsy) + oy
                    x2 = int(sx2 * rsx) + ox
                    y2 = int(sy2 * rsy) + oy
                    tid = int(box.id[0]) if box.id is not None else -1
                    cls_id = int(box.cls[0])
                    yolo_label = self.model.names[cls_id]
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    boxes_data.append((x1, y1, x2, y2, tid, yolo_label, crop))

            active_ids = {bd[4] for bd in boxes_data}

            # Evict stale memory entries
            with s.memory_lock:
                for k in list(s.object_memory):
                    if k not in active_ids:
                        del s.object_memory[k]

            # CLIP classification
            if self.clip.enabled:
                with s.memory_lock:
                    to_classify = [
                        (bd[4], bd[6], bd[5])
                        for bd in boxes_data
                        if bd[4] not in s.object_memory
                        or self._yolo_frame_idx % CLIP_INTERVAL == 0
                    ]
                if to_classify:
                    try:
                        results_clip = self.clip.classify_batch(
                            [x[1] for x in to_classify]
                        )
                        with s.memory_lock:
                            for (tid, _, ylbl), (clbl, conf) in zip(to_classify, results_clip):
                                s.object_memory[tid] = (
                                    (clbl, conf) if conf > 0.4 else (ylbl, 1.0)
                                )
                    except Exception:
                        pass
            else:
                with s.memory_lock:
                    for bd in boxes_data:
                        tid = bd[4]
                        if tid not in s.object_memory:
                            s.object_memory[tid] = (bd[5], 1.0)

            # Update tracking state (trajectories, direction, line crossing)
            s.tracking.update(boxes_data, s)

            # Publish detections (drop crop)
            with s.detection_lock:
                s.latest_detections = [bd[:6] for bd in boxes_data]

            # Dynamic skip adjustment
            if s.fps_display > 0:
                if s.fps_display < TARGET_FPS - 5 and s.yolo_skip < 6:
                    s.yolo_skip += 1
                elif s.fps_display > TARGET_FPS + 5 and s.yolo_skip > 1:
                    s.yolo_skip -= 1
