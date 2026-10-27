"""Real CPU inference on synthetic pixels. No accuracy/performance claims."""
import math
import threading
import numpy as np
import torch
import clip
from modules.clip_module import CLIPModule
from modules.detection import YoloBackend


def main():
    # The learned logit scale must be applied; raw cosine softmax would be ~0.73.
    classifier = CLIPModule.__new__(CLIPModule)
    classifier.device = "cpu"
    classifier._lock = threading.Lock()
    classifier.labels = ["first", "second"]
    classifier.text_features = torch.eye(2)
    classifier.preprocess = lambda image: torch.tensor([1.0, 0.0])

    class FakeEncoder:
        logit_scale = torch.tensor(math.log(10))
        def encode_image(self, images):
            return images

    classifier.model = FakeEncoder()
    result = classifier.classify_batch([np.zeros((32, 32, 3), np.uint8)])
    assert result[0][0] == "first" and result[0][1] > .99, result

    pixels = np.full((480, 800, 3), 127, np.uint8)
    backend = YoloBackend("yolov8n.pt", "cpu")
    boxes = backend.detect(pixels, .35)
    assert isinstance(boxes, list)
    backend.reset()
    boxes_again = backend.detect(pixels, .35)
    assert isinstance(boxes_again, list)
    # Actual CLIP loading, prompt encoding, batch preprocessing, and image inference.
    actual = CLIPModule("cpu")
    classified = actual.classify_batch([pixels, pixels[:, :400]])
    assert len(classified) == 2
    assert all(label in actual.labels and math.isfinite(score) and 0 <= score <= 1
               for label, score in classified)
    print("Real YOLOv8n + CLIP ViT-B/16 CPU inference on synthetic pixels passed")
    print("This verifies runtime compatibility, not recognition accuracy")


if __name__ == "__main__":
    main()
