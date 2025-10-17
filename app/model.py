import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# NOTE: we do NOT need ResNet101_Weights here; checkpoint already contains all weights
from app.config import settings

def build_model(num_classes: int) -> FasterRCNN:
    # Avoid downloading pretrained backbone at runtime; your checkpoint has full weights
    backbone = resnet_fpn_backbone(backbone_name="resnet101", weights=None)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def load_model() -> FasterRCNN:
    device = torch.device(settings.DEVICE)
    ckpt = torch.load(settings.MODEL_PATH, map_location=device)  # checkpoint dict

    model = build_model(settings.NUM_CLASSES)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Clean possible DDP prefixes
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_model] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:   print("  missing:", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:print("  unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")

    model.to(device).eval()
    torch.set_grad_enabled(False)
    print(f"✅ Model loaded from {settings.MODEL_PATH} on {settings.DEVICE}")
    return model