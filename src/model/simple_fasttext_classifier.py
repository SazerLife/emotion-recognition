import torch
import torch.nn as nn

from utils.parameters import LABELS_NAMES


class SimpleFastTextClassifier(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, in_features)
        # self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features, len(LABELS_NAMES))
        self.sigmoid = nn.Sigmoid()

    def forward(self, featrues: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(featrues)
        x = self.linear1(x)
        # x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
