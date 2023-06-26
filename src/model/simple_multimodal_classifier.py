import torch
import torch.nn as nn

from utils.parameters import LABELS_NAMES


class SimpleMultimodalClassifier(nn.Module):
    def __init__(self, dropout_p=0.1) -> None:
        super().__init__()

        # == Text feature extraction ==
        self.text_layernorm = nn.LayerNorm(600)

        # == Audio feature extraction ==
        self.audio_layernorm = nn.LayerNorm((2, 95, 690))
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 2, (3, 5), (1, 2)),
            nn.Conv2d(2, 2, (3, 5), (1, 2)),
            nn.Conv2d(2, 2, (3, 5), (1, 2)),
            nn.Conv2d(2, 2, (3, 3), (2, 2)),
            nn.Conv2d(2, 2, (3, 3), (1, 2)),
        )
        # output: torch.Size([BS, CHANNELS, 42, 20])

        # == Combining modalities ==
        combinated_dim = 600 + 2 * 840
        self.layernorm = nn.LayerNorm(combinated_dim)
        self.hidden_block = nn.Sequential(
            nn.Linear(combinated_dim, combinated_dim * 2),
            nn.ReLU(),
            nn.Linear(combinated_dim * 2, combinated_dim),
            nn.ReLU(),
            nn.Dropout1d(dropout_p),
        )
        self.classifier = nn.Sequential(
            nn.Linear(combinated_dim, len(LABELS_NAMES)),
            nn.Sigmoid(),
        )

    def forward(self, text_featrues: torch.Tensor, audio_featrues: torch.Tensor):
        bs, text_lenght = text_featrues.shape
        x1 = self.text_layernorm(text_featrues)

        # Converting audio specs to sequences
        bs, n_channels, n_mels, spec_lenght = audio_featrues.shape
        x2 = self.audio_layernorm(audio_featrues)
        x2 = self.feature_extractor(audio_featrues)
        x2 = x2.reshape(bs, n_channels, -1)
        x2 = x2.reshape(bs, -1)

        x = torch.concat((x1, x2), dim=-1)
        x = self.layernorm(x)
        x = self.hidden_block(x)
        y = self.classifier(x)
        return y
