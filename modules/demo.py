"""Entirely synthetic frames and known boxes; this is not a model benchmark."""
import cv2
import numpy as np


class SyntheticCapture:
    def __init__(self, count=90):
        self.index, self.count = 0, count

    def read(self):
        if self.index >= self.count:
            return False, None
        frame = np.full((480, 800, 3), 35, dtype=np.uint8)
        y = 50 + self.index * 4
        cv2.rectangle(frame, (160, y), (220, y + 40), (50, 200, 90), -1)
        cv2.putText(frame, "SYNTHETIC PIPELINE DEMO", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        self.index += 1
        return True, frame

    def release(self):
        pass


class SyntheticBackend:
    def reset(self):
        pass

    def detect(self, frame, confidence):
        mask = (frame[:, :, 1] == 200) & (frame[:, :, 0] == 50)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                 int(ys.max()) + 1, 1, "synthetic-box", 1.0)]
