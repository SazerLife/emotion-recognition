import torch
import torch.nn as nn
from utils.parameters import LABELS_NAMES


class ResNet(nn.Module):
    def __init__(self, resnet_name="resnet18", pretrained=False) -> None:
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            resnet_name,
            pretrained=pretrained,
        )
        self.model.conv1 = nn.Conv2d(
            2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(
            in_features=512, out_features=len(LABELS_NAMES), bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        y = self.sigmoid(x)
        return y
