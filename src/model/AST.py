import transformers
import torch
import torch.nn as nn
from utils.parameters import LABELS_NAMES


class ASTPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` into the initial `hidden_states` (patch embeddings) of shape `(batch_size,
    seq_length, hidden_size)` to be consumed by a Transformer.
    """

    def __init__(self, config):
        super().__init__()

        patch_size = config.patch_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = nn.Conv2d(
            2,
            config.hidden_size,
            kernel_size=(patch_size, patch_size),
            stride=(frequency_stride, time_stride),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = input_values.transpose(2, 3)
        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        return embeddings


class AST(nn.Module):
    def __init__(
        self,
        max_length: int,
        num_hidden_layers: int,
        num_mel_bins: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        config = transformers.ASTConfig(
            max_length=max_length,
            num_hidden_layers=num_hidden_layers,
            num_mel_bins=num_mel_bins,
            patch_size=patch_size,
        )
        self.model = transformers.ASTForAudioClassification(config)
        self.model.audio_spectrogram_transformer.embeddings.patch_embeddings = (
            ASTPatchEmbeddings(config)
        )
        self.model.classifier.dense = nn.Linear(
            in_features=768, out_features=len(LABELS_NAMES), bias=True
        )
        self.sigmoid = nn.Sigmoid()
        # print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, channels, mels, lenght = x.shape
        x = x.reshape(bs, channels, lenght, mels)
        x = self.model(input_values=x)
        y = self.sigmoid(x.logits)
        return y
