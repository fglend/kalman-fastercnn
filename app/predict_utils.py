import torch
import io
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def filter_predictions(output, score_thresh=0.25):
    boxes, labels, scores = output['boxes'], output['labels'], output['scores']
    keep = scores >= score_thresh
    return boxes[keep], labels[keep], scores[keep]