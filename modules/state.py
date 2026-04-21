import threading
from collections import defaultdict


class AppState:
    def __init__(self, device: str, frame_w: int, frame_h: int):
        self.device = device
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.running = True

        # Locks
        self.frame_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.detection_lock = threading.Lock()

        # Shared data
        self.latest_frame = None
        self.latest_detections = []
        self.object_memory = {}

        # Modules (set after construction)
        self.clip_module = None
        self.tracking = None

        # Controls
        self.paused = False
        self.fullscreen = False
        self.recording = False
        self.show_heatmap = False
        self.clip_enabled = True

        # Detection params
        self.conf_threshold = 0.35
        self.yolo_skip = 2

        # FPS
        self.fps_display = 0.0

        # ROI: (x1, y1, x2, y2) or None
        self.roi = None

        # Target lock
        self.target_id = None

        # Video writer
        self.video_writer = None
