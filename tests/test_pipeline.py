import contextlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from modules.controls import ControlsModule
from modules.detection import DetectionModule, Request, clamp_roi
from modules.state import AppState
from modules.tracking import TrackingModule
from modules.visualization import draw_detections
import yolo


def request(sequence=1, generation=0, roi=None, enabled=False):
    return Request(sequence, generation, np.zeros((480, 800, 3), dtype=np.uint8),
                   roi, 0.35, enabled)


def backend(boxes=None):
    return Mock(detect=Mock(return_value=boxes or []))


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Worker did not complete in time")


BOX = (10, 20, 40, 60, 1, "car", 0.77)


class InferenceTests(unittest.TestCase):
    def test_real_confidence_retained(self):
        result = DetectionModule(backend([BOX])).process(request())
        self.assertEqual(result.labels, {1: ("car", 0.77)})

    def test_roi_coordinates_and_clipping(self):
        model = backend([(-20, -5, 500, 500, 1, "car", 0.7)])
        result = DetectionModule(model).process(request(roi=(100, 80, 200, 160)))
        self.assertEqual(result.detections[0][:4], (100, 80, 200, 160))
        self.assertEqual(model.detect.call_args.args[0].shape[:2], (80, 100))

    def test_empty_box_dropped(self):
        result = DetectionModule(backend([(50, 0, 40, 10, 1, "car", .7)])).process(request())
        self.assertFalse(result.detections)

    def test_invalid_roi_rejected(self):
        for roi in ((2, 2, 2, 5), (900, 0, 950, 20), (10, 20, 5, 3)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                clamp_roi(roi, 800, 480)

    def test_clip_cache_and_disable(self):
        clip = Mock(classify_batch=Mock(return_value=[("truck", .9)]))
        detector = DetectionModule(backend([BOX]), clip)
        self.assertEqual(detector.process(request(enabled=True)).labels[1], ("truck", .9))
        detector.process(request(2, enabled=True))
        self.assertEqual(clip.classify_batch.call_count, 1)
        self.assertEqual(detector.process(request(3)).labels[1], ("car", .77))

    def test_low_clip_score_keeps_yolo_confidence(self):
        clip = Mock(classify_batch=Mock(return_value=[("truck", .1)]))
        result = DetectionModule(backend([BOX]), clip).process(request(enabled=True))
        self.assertEqual(result.labels[1], ("car", .77))

    def test_clip_failure_logged_and_falls_back(self):
        clip = Mock(classify_batch=Mock(side_effect=RuntimeError("synthetic failure")))
        with self.assertLogs("modules.detection", logging.WARNING):
            result = DetectionModule(backend([BOX]), clip).process(request(enabled=True))
        self.assertEqual(result.labels[1], ("car", .77))

    def test_generation_resets_backend_and_cache(self):
        clip = Mock(classify_batch=Mock(return_value=[("truck", .9)]))
        model = backend([BOX])
        detector = DetectionModule(model, clip)
        detector.process(request(enabled=True))
        detector.process(request(2, generation=1, enabled=True))
        self.assertEqual(model.reset.call_count, 2)
        self.assertEqual(clip.classify_batch.call_count, 2)

    def test_untracked_ids_never_share_clip_memory(self):
        clip = Mock()
        boxes = [(*BOX[:4], -1, "car", .77), (70, 20, 100, 60, -2, "person", .8)]
        result = DetectionModule(backend(boxes), clip).process(request(enabled=True))
        self.assertEqual(result.labels, {-1: ("car", .77), -2: ("person", .8)})
        clip.classify_batch.assert_not_called()

    def test_missing_track_evicts_cache(self):
        model = backend([BOX])
        clip = Mock(classify_batch=Mock(return_value=[("truck", .9)]))
        detector = DetectionModule(model, clip)
        detector.process(request(enabled=True))
        model.detect.return_value = []
        detector.process(request(2, enabled=True))
        model.detect.return_value = [BOX]
        detector.process(request(3, enabled=True))
        self.assertEqual(clip.classify_batch.call_count, 2)


    def test_low_score_cache_uses_current_yolo_confidence(self):
        model = backend([BOX])
        clip = Mock(classify_batch=Mock(return_value=[("truck", .1)]))
        detector = DetectionModule(model, clip)
        detector.process(request(enabled=True))
        model.detect.return_value = [(*BOX[:6], .62)]
        result = detector.process(request(2, enabled=True))
        self.assertEqual(result.labels[1], ("car", .62))
        self.assertEqual(clip.classify_batch.call_count, 1)


class WorkerTests(unittest.TestCase):
    def test_unique_frames_latest_pending_and_stop(self):
        entered, release = threading.Event(), threading.Event()
        seen = []
        model = backend()

        def detect(frame, confidence):
            seen.append(int(frame[0, 0, 0]))
            if len(seen) == 1:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Test release timed out")
            return []

        model.detect.side_effect = detect
        detector = DetectionModule(model)
        detector.start()
        try:
            first = request(1)
            first.frame[0, 0, 0] = 1
            detector.submit(first)
            self.assertTrue(entered.wait(3))
            for number in (2, 3):
                item = request(number)
                item.frame[0, 0, 0] = number
                detector.submit(item)
            with self.assertRaises(ValueError):
                detector.submit(request(3))
            release.set()
            wait_for(lambda: detector.latest() is not None and detector.latest().sequence == 3)
            self.assertEqual(seen, [1, 3])
            self.assertIsNone(detector.latest(after=3))
        finally:
            release.set()
            detector.stop()
        self.assertFalse(detector._thread.is_alive())
        with self.assertRaises(RuntimeError):
            detector.submit(request(4))

    def test_worker_error_reaches_caller(self):
        model = backend()
        model.detect.side_effect = ValueError("synthetic backend error")
        detector = DetectionModule(model)
        detector.start()
        try:
            detector.submit(request())
            wait_for(lambda: detector._error is not None)
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                detector.latest()
        finally:
            detector.stop()

    def test_idle_stop(self):
        detector = DetectionModule(backend())
        detector.start()
        detector.stop()
        self.assertFalse(detector._thread.is_alive())


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.state = AppState("cpu", 800, 480)
        self.tracking = self.state.tracking = TrackingModule(800, 480)

    def update(self, y, tid=1):
        self.tracking.update([(10, y - 10, 30, y + 10, tid, "car", .8)], self.state)

    def test_crossing_only_once_and_stationary_direction(self):
        for y in (220, 250, 220, 250, 250):
            self.update(y)
        self.assertEqual(self.tracking.line_counts["car"], 1)
        self.assertEqual(self.tracking.directions[1], "STILL")

    def test_gap_starts_new_track_lifetime(self):
        self.update(220)
        self.update(250)
        self.tracking.update([], self.state)
        self.assertFalse(self.tracking.trajectories)
        self.assertFalse(self.tracking.crossed_ids)
        self.update(220)
        self.update(250)
        self.assertEqual(self.tracking.line_counts["car"], 2)

    def test_untracked_boxes_not_counted(self):
        self.update(220, -1)
        self.update(250, -1)
        self.assertFalse(self.tracking.line_counts)
        self.assertFalse(self.tracking.trajectories)

    def test_reset_clears_render_state(self):
        self.update(220)
        self.state.latest_detections = [BOX]
        self.state.object_memory = {1: ("truck", .9)}
        self.state.invalidate()
        self.assertEqual(self.state.generation, 1)
        self.assertFalse(self.state.latest_detections)
        self.assertFalse(self.state.object_memory)
        self.assertFalse(self.tracking.trajectories)

    def test_heatmap_does_not_darken_unoccupied_pixels(self):
        frame = np.full((480, 800, 3), 100, np.uint8)
        draw_detections(frame, [], {}, self.tracking, None, True)
        self.assertTrue(np.all(frame[400, 400] == 100))


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.state = AppState("cpu", 800, 480)
        self.state.tracking = TrackingModule(800, 480)
        with patch("cv2.setMouseCallback"):
            self.controls = ControlsModule(self.state, self.state.tracking)

    def test_roi_committed_on_release_even_over_panel(self):
        self.controls._mouse(cv2.EVENT_LBUTTONDOWN, 10, 20, 0, None)
        self.assertIsNone(self.state.roi)
        self.controls._mouse(cv2.EVENT_LBUTTONUP, 790, 400, 0, None)
        self.assertFalse(self.controls.drawing_roi)
        self.assertEqual(self.state.roi, (10, 20, 570, 400))
        self.assertEqual(self.state.generation, 1)

    def test_line_change_resets_trails(self):
        self.state.tracking.update([BOX], self.state)
        self.controls.handle_key(ord("l"))
        self.assertFalse(self.state.tracking.trajectories)

    def test_recording_open_failure_does_not_claim_recording(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with patch("cv2.VideoWriter", return_value=writer), contextlib.redirect_stdout(io.StringIO()):
            self.controls.handle_key(ord("v"))
        self.assertFalse(self.state.recording)
        self.assertIsNone(self.state.video_writer)
        writer.release.assert_called_once()


class CliTests(unittest.TestCase):
    def test_import_has_no_camera_model_or_torch_side_effects(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys,yolo; assert 'torch' not in sys.modules; "
                                 "assert 'ultralytics' not in sys.modules"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_demo_video_and_report_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = yolo.parser().parse_args(["--demo", "--headless", "--output", str(root / "demo.mp4"),
                                             "--report", str(root / "demo.json")])
            with contextlib.redirect_stdout(io.StringIO()):
                result = yolo.run(args)
            self.assertEqual(result["processed_frames"], 90)
            self.assertEqual(result["crossings"], {"synthetic-box": 1})
            self.assertEqual(json.loads((root / "demo.json").read_text()), result)
            cap = cv2.VideoCapture(str(root / "demo.mp4"))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 90)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (480, 800, 3))
            finally:
                cap.release()
            with self.assertRaisesRegex(ValueError, "already exists"):
                yolo.run(args)

    def test_empty_source_releases_capture_before_model_load(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        cap.get.return_value = 20
        with patch("cv2.VideoCapture", return_value=cap), self.assertRaisesRegex(RuntimeError, "readable"):
            yolo.run(yolo.parser().parse_args(["--headless"]))
        cap.release.assert_called_once()


    def test_gui_discards_result_from_old_settings(self):
        from modules.detection import Result
        from modules.demo import SyntheticCapture
        detector = Mock()
        detector.latest.side_effect = [
            Result(1, 0, request().frame, (), {}),
            Result(2, 0, request().frame, (BOX,), {1: ("car", .77)})
        ]
        control = Mock(drawing_roi=False, roi_start=None)
        def make_control(state, tracking):
            def handle(key):
                if state.generation == 0:
                    state.invalidate()
                return True
            control.handle_key.side_effect = handle
            return control
        with patch("modules.demo.SyntheticCapture", return_value=SyntheticCapture(2)), \
             patch("yolo.DetectionModule", return_value=detector), \
             patch("yolo.ControlsModule", side_effect=make_control), \
             patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.waitKey", return_value=-1), \
             patch("cv2.getWindowProperty", return_value=1), patch("cv2.destroyAllWindows"), \
             contextlib.redirect_stdout(io.StringIO()):
            result = yolo.run(yolo.parser().parse_args(["--demo"]))
        self.assertEqual(result["captured_frames"], 2)
        self.assertEqual(result["processed_frames"], 1)
        detector.stop.assert_called_once()

    def test_invalid_options(self):
        for options in (["--confidence", "0"], ["--max-frames", "-1"],
                        ["--demo", "--clip"], ["--output", "unused.mp4"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                yolo.run(yolo.parser().parse_args(options))


if __name__ == "__main__":
    unittest.main()
