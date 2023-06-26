from typing import List
import torch
import torch.nn as nn
from utils.parameters import LABELS_NAMES


class ComplicatedAudioClassifier(nn.Module):
    def __init__(
        self,
        input_lenght: int,
        embedding_dim: int,
        conv_kernel_sizes: List[int],
        conv_strides: List[int],
        lstm_layers_count: int,
        is_bidirectional_lstm: bool,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(conv_kernel_sizes) == len(conv_strides)

        self.layernorm = nn.LayerNorm((2, 95, input_lenght))
        self.upsampling = nn.Sequential(
            nn.Conv2d(2, embedding_dim, 3, stride=1),
            nn.ReLU(),
        )

        conv = list()
        for idx in range(len(conv_strides)):
            conv.append(
                nn.Sequential(
                    nn.Dropout2d(dropout_p),
                    nn.Conv2d(
                        embedding_dim,
                        embedding_dim * 2,
                        conv_kernel_sizes[idx],
                        conv_strides[idx],
                    ),
                    nn.ReLU(),
                )
            )
            embedding_dim = embedding_dim * 2
        self.conv = nn.ModuleList(conv)
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=50,
            hidden_size=embedding_dim,
            num_layers=lstm_layers_count,
            bidirectional=is_bidirectional_lstm,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, len(LABELS_NAMES), bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, n_channels, n_mels, length = x.shape
        x = self.layernorm(x)
        x = self.upsampling(x)
        for conv in self.conv:
            x = conv(x)
        x = x.reshape(bs, self.embedding_dim, -1)
        output, _ = self.lstm(x)
        y = self.classifier(output[:, -1])
        return y
