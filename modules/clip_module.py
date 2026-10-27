import cv2
import torch
import threading
from PIL import Image

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class CLIPModule:
    def __init__(self, device: str):
        import clip
        self.device = device
        self.enabled = True
        self._lock = threading.Lock()

        print("[CLIP] Loading ViT-B/16...")
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.model.eval()
        self.labels = COCO_LABELS

        tokens = clip.tokenize([f"a photo of a {label}" for label in self.labels]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        print("[CLIP] Ready.")

    def classify_batch(self, crops: list) -> list:
        """BGR numpy crops -> list of (label, confidence)."""
        if not crops:
            return []
        with self._lock:
            imgs = torch.cat([
                self.preprocess(
                    Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                ).unsqueeze(0)
                for c in crops
            ]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(imgs)
                feats /= feats.norm(dim=-1, keepdim=True)
                sims = (self.model.logit_scale.exp() * feats @ self.text_features.T).softmax(dim=-1)
            out = []
            for i in range(len(crops)):
                idx = sims[i].argmax().item()
                out.append((self.labels[idx], sims[i][idx].item()))
        return out

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
