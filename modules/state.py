"""State owned by the UI thread; inference workers only publish result packets."""
import threading


class AppState:
    def __init__(self, device, frame_w, frame_h):
        self.device, self.frame_w, self.frame_h = device, frame_w, frame_h
        self.memory_lock = threading.RLock()
        self.detection_lock = threading.RLock()
        self.latest_detections = []
        self.object_memory = {}
        self.clip_module = None
        self.tracking = None
        self.paused = self.fullscreen = self.recording = self.show_heatmap = False
        self.conf_threshold = 0.35
        self.fps_display = 0.0
        self.roi = self.target_id = self.video_writer = None
        self.generation = 0

    def invalidate(self):
        """Discard pending inference after ROI, classifier, or reset changes."""
        self.generation += 1
        self.latest_detections = []
        self.object_memory.clear()
        self.target_id = None
        if self.tracking:
            self.tracking.reset()
