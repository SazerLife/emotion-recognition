from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class StochasticModel(nn.Module):
    def __init__(self, train_csv: pd.DataFrame) -> None:
        super().__init__()
        self.labels_names = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
        assert set(self.labels_names) - set(train_csv.columns) == set()

        self.params = list()
        for label_name in self.labels_names:
            trainset = train_csv[label_name].to_numpy()
            self.params.append((trainset.mean(), trainset.std()))

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        target = [np.random.normal(*param) for param in self.params]
        target = torch.tensor(
            [label if label > 0 else 0 for label in target], dtype=torch.float32
        )
        return target
