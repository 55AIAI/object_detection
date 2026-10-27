"""CLI entry point. Importing this module never opens a camera or loads weights."""
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import time

import cv2

from modules.controls import ControlsModule, WINDOW
from modules.detection import DetectionModule, Request, YoloBackend
from modules.state import AppState
from modules.tracking import TrackingModule
from modules.visualization import draw_detections, draw_roi, draw_panel, draw_alert_banner


def parser():
    p = argparse.ArgumentParser(description="YOLO tracking with optional CLIP labels")
    p.add_argument("--source", default="0", help="Camera index or local video path")
    p.add_argument("--weights", default="yolov8n.pt", help="Local trusted YOLO weights")
    p.add_argument("--download-weights", action="store_true", help="Allow Ultralytics weight download")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--clip", action="store_true", help="Load OpenAI CLIP (may download weights)")
    p.add_argument("--clip-threshold", type=float, default=0.4)
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--headless", action="store_true", help="Process sequentially without GUI")
    p.add_argument("--output", type=Path, help="Annotated MP4; headless mode only")
    p.add_argument("--report", type=Path, help="Final JSON summary")
    p.add_argument("--max-frames", type=int, default=0, help="Captured-frame limit; 0 means all")
    p.add_argument("--demo", action="store_true", help="90 synthetic frames; no camera/model/network")
    return p


def open_writer(path, width, height, fps):
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Cannot open MP4 writer; check your OpenCV codecs")
    return writer


def run(args):
    if not 0 < args.confidence <= 1 or not 0 < args.clip_threshold <= 1:
        raise ValueError("Confidence and CLIP threshold must be in (0, 1]")
    if args.max_frames < 0:
        raise ValueError("max-frames must be nonnegative")
    if args.output and not args.headless:
        raise ValueError("--output requires --headless; GUI recording uses V")
    if args.demo and args.clip:
        raise ValueError("--demo uses synthetic boxes and cannot enable CLIP")
    for path in (args.report, args.output):
        if path and path.exists():
            raise ValueError(f"Output already exists: {path}")
    if args.report and args.output and args.report.resolve() == args.output.resolve():
        raise ValueError("Video and report paths must differ")
    capture = detector = state = writer = None
    gui = False
    captured = processed = 0
    fps = 20.0
    try:
        if args.demo:
            from modules.demo import SyntheticCapture, SyntheticBackend
            capture, backend, device = SyntheticCapture(), SyntheticBackend(), "cpu"
        else:
            source = int(args.source) if args.source.isdecimal() else args.source
            if isinstance(source, str) and not Path(source).is_file():
                raise ValueError("source must be a camera index or an existing local video")
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open source: {args.source}")
            fps_value = capture.get(cv2.CAP_PROP_FPS)
            if 0 < fps_value <= 240:
                fps = fps_value
            # Read the source before model loading so camera failures do not download weights.
            backend = None
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Source has no readable frames")
        height, width = frame.shape[:2]
        if width <= 230 or height < 240:
            raise ValueError("Frames must be wider than 230 and at least 240 pixels tall")
        classifier = None
        if not args.demo:
            if not args.download_weights and not Path(args.weights).is_file():
                raise ValueError("Weights not found. Supply --weights or explicitly use --download-weights")
            import torch
            device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable")
            backend = YoloBackend(args.weights, device)
            if args.clip:
                from modules.clip_module import CLIPModule
                classifier = CLIPModule(device)
        state = AppState(device, width, height)
        state.conf_threshold, state.clip_module = args.confidence, classifier
        tracking = state.tracking = TrackingModule(width, height)
        detector = DetectionModule(backend, classifier, args.clip_threshold)
        controls = None
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            gui = True
            controls = ControlsModule(state, tracking)
            detector.start()
        if args.output:
            writer = open_writer(args.output, width, height, fps)
        previous = time.perf_counter()
        last_result = -1
        display_packet = None
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError("Source frame dimensions changed")
            captured += 1
            request = Request(captured, state.generation, frame.copy(), state.roi,
                              state.conf_threshold, classifier is not None and classifier.enabled)
            if args.headless:
                packet = detector.process(request)
            else:
                detector.submit(request)
                packet = detector.latest(last_result)
            if packet is not None and packet.generation == state.generation:
                last_result = packet.sequence
                state.latest_detections, state.object_memory = list(packet.detections), packet.labels
                tracking.update(packet.detections, state)
                processed += 1
                display_packet = packet
            # Boxes and trajectories are always drawn on the frame that produced them.
            if display_packet is not None and display_packet.generation == state.generation:
                display = display_packet.frame.copy()
                counts = Counter(state.object_memory[d[4]][0] for d in state.latest_detections)
                draw_detections(display, state.latest_detections, state.object_memory, tracking,
                                state.target_id, state.show_heatmap)
            else:
                display, counts = frame.copy(), Counter()
            now = time.perf_counter()
            state.fps_display = 1 / max(now - previous, 1e-6)
            previous = now
            draw_roi(display, state.roi, controls.drawing_roi if controls else False,
                     controls.roi_start if controls else None)
            draw_alert_banner(display, counts)
            draw_panel(display, state, counts, tracking)
            if writer is not None:
                writer.write(display)
            if state.recording and state.video_writer is not None:
                state.video_writer.write(display)
            if not args.headless:
                cv2.imshow(WINDOW, display)
                if not controls.handle_key(cv2.waitKey(1) & 0xFF):
                    break
                while state.paused:
                    if not controls.handle_key(cv2.waitKey(30) & 0xFF):
                        return None
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and captured >= args.max_frames:
                break
            ok, frame = capture.read()
        summary = {"mode": "synthetic" if args.demo else "model",
                   "captured_frames": captured, "processed_frames": processed,
                   "crossings": dict(tracking.line_counts),
                   "crossing_policy": "once per continuously visible track; reset on ROI/settings changes"}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary))
        return summary
    finally:
        # Release resources even if stopping a native inference call times out.
        try:
            if detector is not None:
                detector.stop()
        finally:
            if writer is not None:
                writer.release()
            if state is not None and state.video_writer is not None:
                state.video_writer.release()
            if capture is not None:
                capture.release()
            if gui:
                cv2.destroyAllWindows()


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser().exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
