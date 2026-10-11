from torchvision.models import resnet50, ResNet50_Weights
from torch import nn


def build_model(pretrained=True):
    # torchvision's ResNet V1.5, ImageNet-1K V2; never fall back silently.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model
