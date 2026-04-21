import cv2
import time
import torch
from collections import defaultdict

from modules.state import AppState
from modules.clip_module import CLIPModule
from modules.tracking import TrackingModule
from modules.detection import DetectionModule
from modules.visualization import (
    draw_detections, draw_roi, draw_panel, draw_alert_banner
)
from modules.controls import ControlsModule, WINDOW

# -------- INIT --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SYSTEM] Device: {device.upper()}")

FRAME_W, FRAME_H = 1280, 720

state = AppState(device, FRAME_W, FRAME_H)
clip_mod = CLIPModule(device)
tracking = TrackingModule(FRAME_W, FRAME_H)

state.clip_module = clip_mod
state.tracking = tracking

detector = DetectionModule(clip_mod, state)
detector.start()

# -------- CAMERA --------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

controls = ControlsModule(state, tracking)

prev_time = time.time()
frozen_frame = None

# -------- MAIN LOOP --------
while True:
    if not state.paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Push latest frame to detector thread
        with state.frame_lock:
            state.latest_frame = frame.copy()

        # Snapshot shared state for this render pass
        with state.detection_lock:
            detections = list(state.latest_detections)
        with state.memory_lock:
            memory_snap = dict(state.object_memory)

        counts_snap = defaultdict(int)
        for (_, _, _, _, tid, ylbl) in detections:
            lbl = memory_snap.get(tid, (ylbl, 1.0))[0]
            counts_snap[lbl] += 1

        # Draw all layers
        draw_detections(frame, detections, memory_snap, tracking,
                        state.target_id, state.show_heatmap)
        draw_roi(frame, state.roi, controls.drawing_roi, controls.roi_start)
        draw_alert_banner(frame, counts_snap)
        draw_panel(frame, state, counts_snap, tracking)

        # FPS
        now = time.time()
        state.fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Recording
        if state.recording and state.video_writer:
            state.video_writer.write(frame)

        frozen_frame = frame.copy()
        cv2.imshow(WINDOW, frame)

    else:
        if frozen_frame is not None:
            disp = frozen_frame.copy()
            h, w = disp.shape[:2]
            cv2.rectangle(disp, (0, h // 2 - 30), (w, h // 2 + 30), (20, 20, 20), -1)
            cv2.putText(disp, "PAUSED", (w // 2 - 80, h // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 80, 255), 3, cv2.LINE_AA)
            cv2.imshow(WINDOW, disp)

    key = cv2.waitKey(1) & 0xFF
    if not controls.handle_key(key):
        break

# -------- CLEANUP --------
state.running = False
if state.video_writer:
    state.video_writer.release()
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Exited cleanly.")
