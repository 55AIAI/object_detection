import cv2
import time
from pathlib import Path

WINDOW = "YOLO_CLIP"


class ControlsModule:
    def __init__(self, state, tracking):
        self.state, self.tracking = state, tracking
        self.drawing_roi = False
        self.roi_start = None
        cv2.setMouseCallback(WINDOW, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        s = self.state
        if x > s.frame_w - 230 and not self.drawing_roi:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self._try_select_target(x, y)
            else:
                self.drawing_roi, self.roi_start = True, (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing_roi:
            roi = self._make_roi(self.roi_start, (x, y), s.frame_h, s.frame_w)
            s.roi = roi if roi and roi[2] - roi[0] > 20 and roi[3] - roi[1] > 20 else None
            self.drawing_roi, self.roi_start = False, None
            s.invalidate()
        elif event == cv2.EVENT_RBUTTONDOWN:
            s.roi = None
            self.drawing_roi, self.roi_start = False, None
            s.invalidate()

    @staticmethod
    def _make_roi(p1, p2, h, w):
        if p1 is None or p2 is None:
            return None
        x1, x2 = sorted((p1[0], p2[0]))
        y1, y2 = sorted((p1[1], p2[1]))
        return max(0, x1), max(0, y1), min(w - 230, x2), min(h, y2)

    def _try_select_target(self, mx, my):
        s = self.state
        for detection in s.latest_detections:
            x1, y1, x2, y2, tid, _ = detection[:6]
            if tid >= 0 and x1 <= mx <= x2 and y1 <= my <= y2:
                s.target_id = None if s.target_id == tid else tid
                return
        s.target_id = None

    def handle_key(self, key):
        s = self.state
        if key in (ord('q'), 27):
            return False
        if key == ord('p'):
            s.paused = not s.paused
        elif key == ord('f'):
            s.fullscreen = not s.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                 cv2.WINDOW_FULLSCREEN if s.fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('r'):
            s.invalidate()
        elif key == ord('c'):
            if s.clip_module is None:
                print("[CLIP] Start with --clip to enable classification")
            else:
                s.clip_module.toggle()
                s.invalidate()
        elif key == ord('h'):
            s.show_heatmap = not s.show_heatmap
        elif key == ord('v'):
            self._toggle_recording()
        elif key == ord('l'):
            self._cycle_line()
        elif key == ord('x'):
            s.roi = None
            s.invalidate()
        elif key in (ord('+'), ord('='), ord('-')):
            delta = -0.05 if key == ord('-') else 0.05
            s.conf_threshold = min(0.95, max(0.05, round(s.conf_threshold + delta, 2)))
            s.invalidate()
        return True

    def _toggle_recording(self):
        s = self.state
        if s.recording:
            s.video_writer.release()
            s.video_writer = None
            s.recording = False
            return
        path = Path("output") / f"recording_{time.time_ns()}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                 (s.frame_w, s.frame_h))
        if not writer.isOpened():
            writer.release()
            print("[REC] Codec unavailable; recording was not started")
            return
        s.video_writer, s.recording = writer, True
        print(f"[REC] {path}")

    def _cycle_line(self):
        axis, pos = self.tracking.crossing_line
        h, w = self.state.frame_h, self.state.frame_w - 230
        if axis == 'h' and pos == h // 2:
            line = ('v', w // 2)
        elif axis == 'v':
            line = ('h', h // 4)
        elif pos == h // 4:
            line = ('h', 3 * h // 4)
        else:
            line = ('h', h // 2)
        self.tracking.crossing_line = line
        self.state.invalidate()
