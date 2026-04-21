import cv2
import time
import os

WINDOW = "YOLO_CLIP"


class ControlsModule:
    def __init__(self, state, tracking):
        self.state = state
        self.tracking = tracking
        self.drawing_roi = False
        self.roi_start = None
        cv2.setMouseCallback(WINDOW, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        s = self.state
        w = s.frame_w

        # Ignore clicks inside the side panel
        if x > w - 230:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Ctrl+click: select target object
                self._try_select_target(x, y)
            else:
                # Start ROI draw
                self.drawing_roi = True
                self.roi_start = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_roi:
            s.roi = self._make_roi(self.roi_start, (x, y), s.frame_h, w)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing_roi:
            self.drawing_roi = False
            roi = self._make_roi(self.roi_start, (x, y), s.frame_h, w)
            if roi and (roi[2] - roi[0]) > 20 and (roi[3] - roi[1]) > 20:
                s.roi = roi
            else:
                s.roi = None
            self.roi_start = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            s.roi = None
            s.target_id = None

    def _make_roi(self, p1, p2, h, w):
        if not p1 or not p2:
            return None
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        return (
            max(0, x1), max(0, y1),
            min(w - 230, x2), min(h, y2)
        )

    def _try_select_target(self, mx, my):
        s = self.state
        with s.detection_lock:
            dets = list(s.latest_detections)
        for (x1, y1, x2, y2, tid, _) in dets:
            if x1 <= mx <= x2 and y1 <= my <= y2:
                s.target_id = tid if s.target_id != tid else None
                return
        s.target_id = None

    def handle_key(self, key: int) -> bool:
        s = self.state
        if key in (ord('q'), 27):
            return False  # signal exit

        elif key == ord('p'):
            s.paused = not s.paused

        elif key == ord('f'):
            s.fullscreen = not s.fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if s.fullscreen else cv2.WINDOW_NORMAL
            )

        elif key == ord('r'):
            with s.memory_lock:
                s.object_memory.clear()
            self.tracking.reset()

        elif key == ord('c'):
            enabled = s.clip_module.toggle()
            print(f"[CLIP] {'ON' if enabled else 'OFF'}")

        elif key == ord('h'):
            s.show_heatmap = not s.show_heatmap

        elif key == ord('v'):
            self._toggle_recording()

        elif key == ord('l'):
            self._cycle_line()

        elif key == ord('x'):
            s.target_id = None
            s.roi = None

        elif key == ord('+') or key == ord('='):
            s.conf_threshold = min(0.95, round(s.conf_threshold + 0.05, 2))

        elif key == ord('-'):
            s.conf_threshold = max(0.05, round(s.conf_threshold - 0.05, 2))

        return True  # continue

    def _toggle_recording(self):
        s = self.state
        if not s.recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(os.path.dirname(__file__), '..', f"recording_{ts}.mp4")
            path = os.path.abspath(path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            s.video_writer = cv2.VideoWriter(path, fourcc, 20, (s.frame_w, s.frame_h))
            s.recording = True
            print(f"[REC] Started -> {path}")
        else:
            if s.video_writer:
                s.video_writer.release()
                s.video_writer = None
            s.recording = False
            print("[REC] Stopped.")

    def _cycle_line(self):
        axis, pos = self.tracking.crossing_line
        h, w = self.state.frame_h, self.state.frame_w - 230
        if axis == 'h' and pos == h // 2:
            self.tracking.crossing_line = ('v', w // 2)
        elif axis == 'v':
            self.tracking.crossing_line = ('h', h // 4)
        elif pos == h // 4:
            self.tracking.crossing_line = ('h', 3 * h // 4)
        else:
            self.tracking.crossing_line = ('h', h // 2)
        self.tracking.crossed_ids.clear()
        self.tracking.line_counts.clear()
