import torch
import torch.nn as nn
from utils.parameters import LABELS_NAMES


class SimpleAudioClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        input_lenght: int,
        conv_layers_count: int,
        conv_kernel_size: int,
        conv_stride: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.layernorm = nn.LayerNorm((2, 95, input_lenght))
        self.upsampling = nn.Sequential(
            nn.Conv2d(2, embedding_dim, 3, stride=1),
            nn.ReLU(),
        )

        conv = list()
        for _ in range(conv_layers_count):
            conv.append(
                nn.Sequential(
                    nn.Dropout2d(dropout_p),
                    nn.Conv2d(
                        embedding_dim, embedding_dim * 2, conv_kernel_size, conv_stride
                    ),
                    nn.ReLU(),
                )
            )
            embedding_dim = embedding_dim * 2
        self.conv = nn.ModuleList(conv)

        self.classifier = nn.Sequential(
            nn.Linear(249704, len(LABELS_NAMES), bias=True),  # 949344 487008
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, n_channels, n_mels, length = x.shape
        x = x.reshape(bs, n_channels, n_mels, length)
        x = self.layernorm(x)
        x = self.upsampling(x)
        for conv in self.conv:
            x = conv(x)
        y = self.classifier(x.reshape(bs, -1))
        return y
