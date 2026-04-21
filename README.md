# YOLO + CLIP Real-Time Vision Pipeline

A production-quality real-time computer vision system that combines **YOLOv8 object detection** with **OpenAI CLIP zero-shot classification** for accurate, low-latency object recognition from a live webcam feed.

---

## Demo Features

- Real-time bounding boxes with class labels, confidence scores, and track IDs
- Object trajectory trails and movement direction (LEFT / RIGHT / UP / DOWN)
- Line crossing counter per object class
- Movement heatmap overlay
- Region of Interest (ROI) selection via mouse
- Video recording to MP4
- Target object lock (Ctrl+Click)
- Person alert system
- Dynamic FPS optimization
- Runtime confidence threshold control
- CLIP on/off toggle
- Clean side panel UI

---

## How It Works — Theory

### Stage 1: Object Detection with YOLOv8

**YOLO** (You Only Look Once) is a single-stage object detection neural network. Unlike older two-stage detectors (like Faster R-CNN) that first propose regions and then classify them, YOLO solves detection in one forward pass through the network — which is what makes it fast enough for real-time use.

#### Architecture

YOLOv8 is built from three parts:

```
Input Frame
    │
    ▼
┌─────────────┐
│  Backbone   │  CSPDarknet — extracts feature maps at multiple scales
│  (CSP)      │  Captures edges, textures, shapes, object parts
└──────┬──────┘
       │
    ▼
┌─────────────┐
│    Neck     │  PANet (Path Aggregation Network)
│  (PANet)    │  Fuses features from different depths — small and large objects
└──────┬──────┘
       │
    ▼
┌─────────────┐
│    Head     │  Decoupled detection head
│             │  Outputs: bounding box coords + objectness score + class probs
└─────────────┘
```

#### How detection works

The input frame is divided into a grid. Each grid cell predicts multiple bounding boxes (anchor-free in YOLOv8). For each box the network outputs:

- `(cx, cy, w, h)` — center coordinates and dimensions
- `objectness` — probability that any object exists here
- `class vector` — probability distribution over 80 COCO classes

Boxes with low objectness are discarded. The remaining overlapping boxes go through **Non-Maximum Suppression (NMS)** — only the highest-confidence box per object survives.

#### Tracking with ByteTrack

After detection, **ByteTrack** assigns persistent `track_id` integers to each detected object across frames. It works by solving an assignment problem: match detected boxes in the current frame to known tracks from the previous frame using **IoU** (Intersection over Union) as the matching cost. This gives each object a consistent ID across time — enabling trajectories, direction, and line crossing.

#### In this project

- Model: `yolov8s.pt` (small variant — good accuracy/speed balance on CPU)
- Input resolution: **320×180** (downscaled from 1280×720 for speed)
- Bounding boxes are scaled back to original resolution for display
- YOLO runs in a **background thread** — the display loop never waits for it
- Frame skip is adjusted **dynamically** to maintain target FPS (~25)

---

### Stage 2: Zero-Shot Classification with CLIP

**CLIP** (Contrastive Language–Image Pretraining) is a vision-language model trained by OpenAI on 400 million (image, text) pairs from the internet. It learns a shared embedding space where images and their text descriptions land near each other.

#### Architecture

CLIP has two encoders that are trained jointly:

```
Image ──► Image Encoder (ViT-B/16) ──► 512-dim vector
                                              │
                                        cosine similarity
                                              │
Text  ──► Text Encoder  (Transformer) ──► 512-dim vector
```

**ViT-B/16** (Vision Transformer, Base, 16×16 patch size) splits the input image into a grid of 16×16 pixel patches, treats each patch as a token, and runs a Transformer over all of them. This gives it global context that CNNs lack — it can reason about relationships between parts of the image.

#### Training objective

During training, CLIP is given a batch of N (image, text) pairs. It computes all N² cosine similarities between image and text embeddings. The loss pushes the N correct (image, text) pairs to have high similarity and all N²-N incorrect pairs to have low similarity. This is called **contrastive learning**.

The result: CLIP learns rich visual representations aligned with natural language — without ever being trained on a fixed set of classes.

#### Zero-shot inference

At inference time, you can classify an image against **any set of text labels** — no fine-tuning needed:

```
1. Encode all candidate labels as text:
   text_features = encode_text(["a photo of a person", "a photo of a car", ...])

2. Encode the image:
   image_features = encode_image(crop)

3. Compute cosine similarities and softmax:
   scores = softmax(image_features @ text_features.T)

4. Highest score = predicted class
```

This is "zero-shot" because CLIP was never explicitly trained to classify these labels — it generalizes from its pretraining.

#### In this project

- Model: `ViT-B/16` — finer 16×16 patches for better spatial detail than ViT-B/32
- Labels: all **80 COCO class names** (matches YOLO's vocabulary exactly)
- Text features are **pre-encoded once at startup** and reused — zero runtime cost for text
- Classification runs only when a new `track_id` appears, then every 60 frames
- Multiple crops are **batched into one forward pass** per detection cycle
- CLIP overrides the YOLO label only when confidence > 0.4; otherwise YOLO label is kept

---

### Why Use Both?

| | YOLO | CLIP |
|---|---|---|
| Speed | Very fast (single pass) | Slower (ViT forward pass) |
| Training | Supervised on COCO (fixed 80 classes) | Self-supervised on 400M pairs |
| Strengths | Precise boxes, fast localization | Rich semantics, handles ambiguity |
| Weaknesses | Misclassifies visually similar objects | No localization ability |

YOLO finds **where** objects are. CLIP decides **what** they really are. Together, YOLO provides bounding boxes and initial labels; CLIP re-examines each cropped region and can correct misclassifications that YOLO makes due to visual similarity or partial occlusion.

---

## Project Structure

```
CLIP/
├── yolo.py                  # Entry point — main loop
├── requirements.txt
└── modules/
    ├── state.py             # Shared application state and locks
    ├── clip_module.py       # CLIP model loading and batch inference
    ├── detection.py         # YOLO background thread + dynamic frame skip
    ├── tracking.py          # Trajectories, direction, heatmap, line crossing
    ├── visualization.py     # All OpenCV drawing functions
    └── controls.py          # Keyboard and mouse event handlers
```

---

## Installation

**Requirements:** Python 3.9+, webcam

```bash
git clone https://github.com/55AIAI/object_detection.git
cd object_detection

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> On first run, `yolov8s.pt` and `ViT-B/16` weights download automatically (~100 MB total).

---

## Usage

```bash
python yolo.py
```

---

## Controls

| Key / Action | Function |
|---|---|
| `Q` or `Esc` | Quit |
| `P` | Pause / Resume |
| `F` | Toggle fullscreen |
| `R` | Reset all memory and counts |
| `C` | Toggle CLIP classification on/off |
| `H` | Toggle heatmap overlay |
| `V` | Start / Stop video recording |
| `L` | Cycle crossing line position |
| `+` / `-` | Increase / decrease confidence threshold |
| `X` | Clear ROI and target lock |
| **Mouse drag** | Draw Region of Interest (ROI) |
| **Right-click** | Clear ROI and target |
| **Ctrl + Click** | Lock onto a specific object (target mode) |

---

## Requirements

```
ultralytics
opencv-python
torch
torchvision
pillow
git+https://github.com/openai/CLIP.git
```

GPU (CUDA) is supported automatically if available. The pipeline is fully functional on CPU.

---

## Performance Notes

| Hardware | Expected FPS |
|---|---|
| CPU only | 15–25 FPS display / 5–10 FPS detection |
| NVIDIA GPU (CUDA) | 30+ FPS display / 25+ FPS detection |

- YOLO runs on a **320×180** downscaled frame for CPU efficiency
- Frame skip auto-adjusts to maintain target FPS
- CLIP batches all crops into a single forward pass per cycle
- Background threading keeps the display loop decoupled from inference

---

## License

MIT
