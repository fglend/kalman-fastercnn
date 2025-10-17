import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from app.config import settings

def build_model(num_classes):
    backbone = resnet_fpn_backbone('resnet101', weights=ResNet101_Weights.DEFAULT)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def load_model():
    device = torch.device(settings.DEVICE)
    ckpt = torch.load(settings.MODEL_PATH, map_location=device)

    model = build_model(settings.NUM_CLASSES)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model