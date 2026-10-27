import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 230
PANEL_ALPHA = 0.55


def _text(frame, text, pos, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, detections, memory_snap, tracking, target_id, show_heatmap):
    # Heatmap overlay
    if show_heatmap:
        overlay, alpha = tracking.get_heatmap_overlay()
        if overlay is not None:
            mask = tracking.heatmap > 0
            blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            frame[mask] = blended[mask]

    # Trajectories
    for tid, trail in tracking.trajectories.items():
        pts = list(trail)
        for i in range(1, len(pts)):
            t = i / len(pts)
            color = (int(255 * t), int(180 * (1 - t)), 80)
            cv2.line(frame, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    # Crossing line
    axis, pos = tracking.crossing_line
    h, w = frame.shape[:2]
    if axis == 'h':
        cv2.line(frame, (0, pos), (w, pos), (0, 200, 255), 2)
        _text(frame, "LINE", (10, pos - 6), 0.45, (0, 200, 255))
    else:
        cv2.line(frame, (pos, 0), (pos, h), (0, 200, 255), 2)
        _text(frame, "LINE", (pos + 4, 20), 0.45, (0, 200, 255))

    # Bounding boxes
    for (x1, y1, x2, y2, tid, yolo_label, yolo_conf) in detections:
        label, conf = memory_snap.get(tid, (yolo_label, yolo_conf))
        is_target = (target_id is not None and tid == target_id)

        if is_target:
            color = (0, 215, 255)
            thickness = 3
        elif label == "person":
            color = (255, 60, 60)
            thickness = 2
        else:
            color = (60, 220, 60)
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        direction = tracking.directions.get(tid, "")
        tag = f"{label} {conf:.2f} [{direction}] ID:{tid}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.48, 1)
        ty = max(y1 - 6, th + 2)
        cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(frame, tag, (x1 + 1, ty), FONT, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

        # Alert highlight
        if label == "person":
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 1)


def draw_roi(frame, roi, drawing_roi, roi_start):
    if roi:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
        _text(frame, "ROI", (rx1 + 4, ry1 + 18), 0.5, (255, 165, 0))
    if drawing_roi and roi_start:
        _text(frame, "Draw ROI...", (10, frame.shape[0] - 40), 0.5, (255, 165, 0))


def draw_panel(frame, state, counts_snap, tracking):
    h, w = frame.shape[:2]
    panel = frame[0:h, w - PANEL_W:w].copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, panel, 1 - PANEL_ALPHA, 0, panel)
    frame[0:h, w - PANEL_W:w] = panel

    x, y = w - PANEL_W + 8, 22

    def line(text, color=(220, 220, 220), bold=False):
        nonlocal y
        _text(frame, text, (x, y), 0.48, color, 2 if bold else 1)
        y += 20

    line("== YOLO + CLIP ==", (100, 220, 255), bold=True)
    y += 4
    line(f"FPS:    {int(state.fps_display)}", (0, 255, 200))
    line(f"Device: {state.device.upper()}", (180, 180, 255))
    line("Input: latest frame", (180, 180, 180))
    line(f"Conf:   {state.conf_threshold:.2f}", (180, 180, 180))
    y += 6

    # Status flags
    statuses = []
    if state.paused:        statuses.append(("PAUSED",   (0, 80, 255)))
    if state.recording:     statuses.append(("REC",      (0, 0, 255)))
    if state.clip_module is None or not state.clip_module.enabled: statuses.append(("CLIP OFF", (80, 80, 80)))
    if state.show_heatmap:  statuses.append(("HEATMAP",  (0, 200, 180)))
    if state.target_id is not None: statuses.append((f"TGT:{state.target_id}", (0, 215, 255)))

    for txt, col in statuses:
        line(txt, col, bold=True)

    y += 6
    line("-- Counts --", (200, 200, 100))
    for cls, cnt in list(counts_snap.items())[:10]:
        line(f"  {cls}: {cnt}")

    y += 6
    line("-- Line Cross --", (200, 200, 100))
    for cls, cnt in list(tracking.line_counts.items())[:8]:
        line(f"  {cls}: {cnt}")

    y += 6
    line("-- Alerts --", (200, 100, 100))
    if counts_snap.get("person", 0) > 0:
        line("! PERSON DETECTED", (0, 0, 255), bold=True)

    # Key hints at bottom
    hints = [
        "Q:Quit  P:Pause  F:Full",
        "R:Reset  C:CLIP  H:Heat",
        "V:Record  L:Line",
        "+/-:Conf  ROI:drag mouse",
        "Ctrl+click:Target X:Clear",
    ]
    yh = h - len(hints) * 18 - 4
    for hint in hints:
        _text(frame, hint, (x, yh), 0.38, (150, 150, 150))
        yh += 17


def draw_alert_banner(frame, counts_snap):
    if counts_snap.get("person", 0) > 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 180), -1)
        _text(frame, "! ALERT: PERSON DETECTED !", (w // 2 - 160, 20),
              0.65, (255, 255, 255), 2)
