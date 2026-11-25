import torch.nn as nn
from torchvision import models


def get_model(model_name, pretrained=True):

    if model_name == "resnet18":

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        model = models.resnet18(weights=weights)

        model.fc = nn.Linear(
            model.fc.in_features, 37
        )

    elif model_name == "mobilenet_v3":

        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )

        model = models.mobilenet_v3_small(
            weights=weights
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, 37
        )

    elif model_name == "efficientnet_b0":

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )

        model = models.efficientnet_b0(
            weights=weights
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 37
        )

    else:
        raise ValueError("Unknown model name")

    return model
