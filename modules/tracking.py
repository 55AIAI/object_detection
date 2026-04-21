import cv2
import numpy as np
from collections import defaultdict, deque

MAX_TRAIL = 40       # max trajectory points per object
MAX_MEMORY = 500     # max IDs in any memory dict


class TrackingModule:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.trajectories: dict[int, deque] = {}    # tid -> deque of (cx, cy)
        self.directions: dict[int, str] = {}         # tid -> direction string
        self.line_counts: dict[str, int] = defaultdict(int)  # label -> count
        self.crossed_ids: set = set()

        # Line crossing: (axis, position) — axis='h' or 'v', pos in pixels
        self.crossing_line = ('h', frame_h // 2)

        # Heatmap accumulator
        self.heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
        self.heatmap_alpha = 0.0004   # decay per frame

    def update(self, boxes_data: list, state):
        active_ids = set()

        for (x1, y1, x2, y2, tid, label, *_) in boxes_data:
            if tid < 0:
                continue
            active_ids.add(tid)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Trajectory
            if tid not in self.trajectories:
                self.trajectories[tid] = deque(maxlen=MAX_TRAIL)
            trail = self.trajectories[tid]
            trail.append((cx, cy))

            # Direction from last two points
            if len(trail) >= 2:
                dx = trail[-1][0] - trail[-2][0]
                dy = trail[-1][1] - trail[-2][1]
                if abs(dx) > abs(dy):
                    self.directions[tid] = "RIGHT" if dx > 0 else "LEFT"
                else:
                    self.directions[tid] = "DOWN" if dy > 0 else "UP"

            # Line crossing
            axis, pos = self.crossing_line
            if tid not in self.crossed_ids and len(trail) >= 2:
                prev = trail[-2]
                if axis == 'h':
                    if (prev[1] < pos <= cy) or (prev[1] > pos >= cy):
                        self.crossed_ids.add(tid)
                        with state.memory_lock:
                            lbl = state.object_memory.get(tid, (label, 1.0))[0]
                        self.line_counts[lbl] += 1
                else:
                    if (prev[0] < pos <= cx) or (prev[0] > pos >= cx):
                        self.crossed_ids.add(tid)
                        with state.memory_lock:
                            lbl = state.object_memory.get(tid, (label, 1.0))[0]
                        self.line_counts[lbl] += 1

            # Heatmap
            cv2.circle(self.heatmap, (cx, cy), 18, 1.0, -1)

        # Evict inactive trajectories (keep memory bounded)
        for tid in list(self.trajectories):
            if tid not in active_ids:
                del self.trajectories[tid]
                self.directions.pop(tid, None)

        if len(self.crossed_ids) > MAX_MEMORY:
            self.crossed_ids = set(list(self.crossed_ids)[-MAX_MEMORY:])

        # Decay heatmap
        self.heatmap *= (1.0 - self.heatmap_alpha)
        self.heatmap = np.clip(self.heatmap, 0, 1)

    def get_heatmap_overlay(self, alpha=0.4) -> np.ndarray:
        norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        mask = norm > 5
        overlay = np.zeros_like(colored)
        overlay[mask] = colored[mask]
        return overlay, alpha

    def reset(self):
        self.trajectories.clear()
        self.directions.clear()
        self.line_counts.clear()
        self.crossed_ids.clear()
        self.heatmap[:] = 0
