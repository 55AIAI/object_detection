# YOLO + CLIP Vision Pipeline

[![Vision checks](https://github.com/55AIAI/object_detection/actions/workflows/tests.yml/badge.svg)](https://github.com/55AIAI/object_detection/actions/workflows/tests.yml)

A Python desktop vision project combining **YOLOv8 detection, ByteTrack IDs, trajectories,
line crossing, an ROI, and optional OpenAI CLIP classification**. Works with a webcam or
a local video. A synthetic demo exercises the full rendering and counting pipeline
without a camera, neural-network weights, or network access.

## Try the synthetic demo

Python 3.11–3.13. Run from the repository root:

```bash
git clone https://github.com/55AIAI/object_detection.git
cd object_detection
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-core.txt
python yolo.py --demo --headless --output output/demo.mp4 --report output/demo.json
```

The demo generates 90 frames with a moving green rectangle. Expected summary:
90 processed frames and one `synthetic-box` crossing. Boxes are produced from the
known synthetic color, **not YOLO**. The MP4 shows boxes, ID, trail, direction and
counter. CI publishes the generated video/report in the `synthetic-demo` artifact.

## Real models

```bash
python -m pip install -r requirements.txt
python yolo.py --source 0 --weights yolov8n.pt --download-weights
python yolo.py --source 0 --weights yolov8n.pt --clip
python yolo.py --source video.mp4 --weights yolov8n.pt --headless --output output/video.mp4 --report output/video.json
```

The first command explicitly permits downloading YOLO weights; omit
`--download-weights` to require an existing local checkpoint. Use trusted checkpoints.
`--clip` explicitly loads ViT-B/16 and may download its substantially larger weights.
Models are loaded only after the source opens and yields a frame. Nothing opens a
camera or downloads weights when importing `yolo` or asking for `--help`.

CPU is supported; `--device auto` selects CUDA when available. Install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/) for your hardware.
For a CPU-only install, install torch/torchvision from the official CPU index before
installing requirements, as the CI workflow does.

Sources are camera indices or existing local video files. Frame dimensions must stay
constant, exceed 230 pixels wide, and be at least 240 pixels tall for the side panel.
`--max-frames N` limits captured frames. Output files are never intentionally overwritten.
MP4 writing requires an OpenCV build with a working `mp4v` encoder.

## What was improved

- A bounded latest-frame mailbox replaces the polling loop that repeatedly inferred
  the same frame. Each submitted sequence is processed at most once.
- Inference owns its models; the UI owns trajectories, counters, and rendering.
  Published results carry their source frame, labels, sequence and settings generation.
  Boxes are drawn on that exact source frame.
- Reset, committed ROI changes, confidence changes, CLIP toggles and line changes
  invalidate pending results. Tracker/cache state starts fresh on the next inference.
- YOLO receives an aspect-preserving input through Ultralytics letterboxing.
  ROI offsets and boxes are clipped to valid pixels.
- Actual YOLO confidence is retained instead of displaying a fabricated 1.00.
  Missing track IDs remain untracked and cannot share CLIP memory or crossing counts.
- CLIP uses prompted text, normalized features and the learned logit scale.
  Failures produce a warning and fall back to YOLO for that frame.
- Camera, writer and worker cleanup happens even after errors. Recording starts only
  if the codec opens successfully. Importing the entry point has no runtime effects.

## Controls

| Key/action | Behavior |
|---|---|
| Q / Esc | Exit |
| P | Pause/resume |
| F | Fullscreen |
| R | Reset tracking, labels and counts |
| C | Toggle an already loaded CLIP model |
| H | Heatmap |
| V | Record/stop annotated MP4 under `output/` |
| L | Cycle crossing line and reset tracking/counts |
| + / - | Confidence threshold; resets tracking/counts |
| Mouse drag | Commit ROI on release |
| Ctrl + click | Select a tracked object |
| Right click / X | Clear ROI/target and reset tracking/counts |

The rightmost 230 pixels are an overlaid control panel. ROI selection excludes it;
full-frame detection still includes that image region. GUI recordings use a fixed
20 FPS playback rate; headless output uses the source FPS when valid (20 otherwise).
Processing time and output playback time are distinct.

## Semantics and limits

The counter counts each ID **once per uninterrupted visible track lifetime**. A
detection gap removes its trail/crossing memory; a reappearing ID starts a new lifetime
and can be counted again. IDs are short-lived tracker identities, not identified people.

The GUI prioritizes responsiveness and may drop pending frames. Crossing events can
be missed with sparse/incorrect detections. Headless mode processes every readable
frame sequentially and is preferable for reproducible file processing. A decoder
failure after a valid frame is treated as end of input; no claim of complete file
integrity is made.

CLIP scores are relative softmax scores over the 80 COCO prompts, **not calibrated
probabilities or verified corrections**. When enabled, a score above
`--clip-threshold` (default 0.4) replaces the YOLO label. Otherwise YOLO stays.
Reclassification happens on a new tracked ID and every 60 inference cycles.
No accuracy gain, production readiness, hardware FPS, or safety-critical suitability
is claimed. Real-camera behavior and GPU performance still need hardware evaluation.

All video processing is local. Model downloads contact their upstream hosts.
No input footage, model weights or real-person images are committed to this repository.

## Validation

```bash
python -m unittest discover -s tests -v
python -m tests.model_smoke  # full requirements + downloads; CPU compatibility check
```

28 regression tests cover latest-frame behavior, failures and cleanup, ROI bounds,
CLIP cache/fallback, real confidence, untracked IDs, count lifetimes, controls,
import safety, and decoding the generated MP4. CI runs them on Python 3.11/3.12/3.13.
A separate CI job loads real YOLOv8n and CLIP ViT-B/16 weights and runs synthetic pixels
through both models. This tests integration, not recognition quality.

Core and model dependencies have bounded version ranges; CLIP is fixed to a commit.
The model CI artifact records the resolved environment. This is not a complete
cross-platform lock file.

## Structure

- `yolo.py`: CLI, source lifecycle, rendering and sequential/interactive modes
- `modules/detection.py`: YOLO adapter, CLIP policy, inference mailbox
- `modules/clip_module.py`: OpenAI CLIP adapter
- `modules/tracking.py`: trails, direction, heatmap, crossing counts
- `modules/controls.py`, `state.py`, `visualization.py`: desktop controls and display
- `modules/demo.py`: synthetic source and color-based boxes
- `tests/`: regression and real-model smoke checks
- [Design and validation notes](docs/design.md)

## Upstream projects

[Ultralytics tracking](https://docs.ultralytics.com/modes/track/) and
[OpenAI CLIP](https://github.com/openai/CLIP) provide the neural models and their
own licenses. Review their license terms when redistributing or deploying models.
