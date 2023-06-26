import argparse
import json
import random
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.model.simple_fasttext_classifier import SimpleFastTextClassifier
from utils.parameters import LABELS_NAMES, SEED
from utils.train_val_funcitons import train_classifier, validate

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():
    parser = get_parser()
    args = parser.parse_args()

    config_path: Path = args.config
    assert config_path.exists()
    config: Dict[str, Dict[str, Any]] = json.loads(config_path.read_text())

    train_csv_path = Path(config["data"]["train"])
    val_csv_path = Path(config["data"]["val"])
    assert train_csv_path.exists() and val_csv_path.exists()
    train_text_features, train_audio_features, train_targets = get_features(
        train_csv_path, "train_modified"
    )
    val_text_features, val_audio_targets, val_targets = get_features(
        val_csv_path, "val_modified"
    )

    shuffle: bool = config["train"]["shuffle"]
    batch_size: int = config["train"]["batch_size"]
    trainloader = get_dataloader(
        train_text_features, train_audio_features, train_targets, shuffle, batch_size
    )
    valloader = get_dataloader(
        val_text_features, val_audio_targets, val_targets, shuffle, batch_size
    )

    device = "cuda:0"
    model, criterion, optimizer = get_model(config, device)

    epochs: int = config["train"]["epochs"]
    eval_epoch: int = config["train"]["eval_epoch"]
    save_epoch: int = config["train"]["save_epoch"]
    save_dir = config_path.parent
    _, _, (best_model, best_epoch) = train_classifier(
        trainloader,
        valloader,
        model,
        criterion,
        # torch.from_numpy(sample_weights),
        optimizer,
        device,
        epochs,
        eval_epoch,
        save_epoch,
        True,
        save_dir,
    )
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_model.state_dict()},
        save_dir / "checkpoints" / f"best_model-{best_epoch}.pt",
    )


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--config", type=Path)
    return parser


def get_features(csv_path: Path, dir_name: str):
    csv = pd.read_csv(csv_path)[["video", "start_time", "end_time"] + LABELS_NAMES]
    markuped_features_dir = Path("data/CMU-MOSEI/fasttext_featrues_markuped_text")
    recognised_features_dir = Path("data/CMU-MOSEI/fasttext_featrues_recognised_text")
    audio_features_dir = Path("data/CMU-MOSEI/audio_featrues")

    text_features: List[torch.Tensor] = list()
    audio_features: List[torch.Tensor] = list()
    targets: List[np.ndarray] = list()
    # sample_weights: List[np.ndarray] = list()

    progress_bar = tqdm(csv.values, desc="Features loading")
    for ytid, start_time, end_time, *labels in progress_bar:
        stem = f"{ytid}_{float(start_time):.4f}_{float(end_time):.4f}"
        # file_name = f"{ytid}_{float(start_time):.4f}_{float(end_time):.4f}.pt"

        try:
            markuped_feature = np.load(
                markuped_features_dir / dir_name / f"{stem}.npy",
            )
            recognised_feature = np.load(
                recognised_features_dir / dir_name / f"{stem}.npy"
            )
            text_feature = torch.from_numpy(
                np.concatenate((markuped_feature, recognised_feature), axis=0)
            )
            audio_feature = torch.load(audio_features_dir / dir_name / f"{stem}.pt")

            labels = np.asarray(labels)
            labels[labels > 0.0] = 1
            labels = labels * np.array([3.34, 4.78, 15.09, 1.0, 3.12, 14.58])
            # sample_weights.append(np.array([3.34, 4.78, 15.09, 1.0, 3.12, 14.58]))

            text_features.append(text_feature)
            audio_features.append(audio_feature)
            targets.append(labels)

        except FileNotFoundError as exception:
            print(exception)

    return text_features, audio_features, targets


def get_dataloader(
    text_features: List[torch.Tensor],
    audio_features: List[torch.Tensor],
    targets: List[int],
    shuffle=True,
    batch_size=128,
) -> DataLoader:
    text_features = torch.stack(text_features).to(torch.float32)
    audio_features = torch.stack(audio_features).to(torch.float32)

    targets = torch.tensor(np.asarray(targets)).to(torch.float32)
    dataset = TensorDataset(text_features, audio_features, targets)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


def get_model(config: Dict[str, Dict[str, Any]], device: str):
    module = import_module(config["model"]["source"])
    ModelClass = getattr(module, config["model"]["name"])
    model: nn.Module = ModelClass(**config["model"]["prams"]).to(device)
    criterion = nn.L1Loss()

    lr: float = config["train"]["lr"]
    weight_decay: float = config["train"]["weight_decay"]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if config["model"]["checkpoint_path"]:
        checkpoint_dict = torch.load(config["model"]["checkpoint_path"])
        model.load_state_dict(checkpoint_dict["state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    return model, criterion, optimizer


if __name__ == "__main__":
    main()
