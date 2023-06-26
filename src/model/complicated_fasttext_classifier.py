from typing import Type
import torch
import torch.nn as nn

from utils.parameters import LABELS_NAMES


class ComplicatedFastTextClassifier(nn.Module):
    def __init__(self, in_features, dropout_p=0.1) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.block1 = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU(),
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Dropout1d(dropout_p),
        )
        self.block2 = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU(),
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Dropout1d(dropout_p),
        )
        self.block3 = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU(),
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Dropout1d(dropout_p),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, len(LABELS_NAMES)),
            nn.Sigmoid(),
        )

        self._init_weight(nn.Linear, torch.nn.init.xavier_uniform_)

    def _init_weight(
        self, ModuleClass: Type[nn.Module], weights_initializator: Type[callable]
    ):
        for module in self.modules():
            if isinstance(module, ModuleClass):
                weights_initializator(module.weight)

    def forward(self, featrues: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(featrues)

        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x) + x

        y = self.classifier(x)
        return y
