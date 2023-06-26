import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score

from utils.metrics import binary_weighted_accuracy
from utils.parameters import LABELS_NAMES

# def compute_loss(
#     criterion: nn.Module,
#     sample_weights: torch.FloatTensor,
#     train_preditions: torch.FloatTensor,
#     train_targets: torch.FloatTensor,
# ) -> torch.FloatTensor:
#     loss = criterion(train_preditions, train_targets * )


def train_classifier(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    # sample_weights: torch.FloatTensor,
    optimizer: optim.Optimizer,
    device: str,
    num_epoches: int,
    eval_epoch: int,
    save_epoch: int,
    verbose: bool,
    save_dir: Path,
) -> Tuple[
    Tuple[List[torch.FloatTensor], List[torch.FloatTensor], Dict[str, float]],
    Tuple[List[torch.FloatTensor], List[torch.FloatTensor], Dict[str, float]],
]:
    best_model: nn.Module = None
    best_epoch, best_WA = 0, 0

    train_targets_per_epoch = list()
    train_predits_per_epoch = list()
    train_metrics = defaultdict(list)
    train_metrics["loss"] = list()

    val_targets_per_epoch = list()
    val_predits_per_epoch = list()
    val_metrics = defaultdict(list)
    val_metrics["loss"] = list()

    start = time.time()

    for epoch_idx in range(1, num_epoches + 1):
        train_targets, train_preditions = train_epoch(
            train_dataloader,
            model,
            criterion,
            optimizer,
            device,
        )

        if epoch_idx % eval_epoch == 0:
            # == Count training metric ==
            with torch.no_grad():
                train_targets_per_epoch.append(train_targets)
                train_predits_per_epoch.append(train_preditions)

                loss = criterion(train_preditions, train_targets)

                train_preditions[train_preditions > 0.5] = 1
                train_preditions[train_preditions <= 0.5] = 0
                train_targets[train_targets > 0.0] = 1

                train_metrics["loss"].append(loss.item())
                average_bWA = 0
                for label_index in range(len(LABELS_NAMES)):
                    bWA = binary_weighted_accuracy(
                        train_preditions[:, label_index], train_targets[:, label_index]
                    )
                    train_metrics[f"WA_{LABELS_NAMES[label_index]}"].append(bWA)
                    average_bWA += bWA
                average_bWA = average_bWA / len(LABELS_NAMES)
                train_metrics["Average_bWA"].append(average_bWA)
            # ===========================

            # == Count validation metric ==
            val_targets, val_preditions = validate(val_dataloader, model, device)
            val_targets_per_epoch.append(val_targets)
            val_predits_per_epoch.append(val_preditions)

            loss = criterion(val_preditions, val_targets)

            val_preditions[val_preditions > 0.5] = 1
            val_preditions[val_preditions <= 0.5] = 0
            val_targets[val_targets > 0.0] = 1

            val_metrics["loss"].append(loss.item())
            average_bWA = 0
            for label_index in range(len(LABELS_NAMES)):
                bWA = binary_weighted_accuracy(
                    val_preditions[:, label_index], val_targets[:, label_index]
                )
                val_metrics[f"WA_{LABELS_NAMES[label_index]}"].append(bWA)
                average_bWA += bWA
            average_bWA = average_bWA / len(LABELS_NAMES)
            val_metrics["Average_bWA"].append(average_bWA)
            # =============================

            if val_metrics["Average_bWA"][-1] > best_WA:
                best_model = model
                best_WA = val_metrics["Average_bWA"][-1]
                best_epoch = epoch_idx

        if verbose and (epoch_idx % eval_epoch == 0):
            print(f"Time consuming: {(time.time() - start):.4f}")

            train_log = f"Train: {epoch_idx:4d}/{num_epoches} | Loss: {train_metrics['loss'][-1]:.4f} |"
            for label_name in LABELS_NAMES:
                metric_name = f"WA_{label_name}"
                train_log += f" {metric_name}: {train_metrics[metric_name][-1]:.2f} |"
            train_log += f" Average_bWA: {train_metrics['Average_bWA'][-1]:.2f} |"
            print(train_log)

            val_log = f"Val:   {epoch_idx:4d}/{num_epoches} | Loss: {val_metrics['loss'][-1]:.4f} |"
            for label_name in LABELS_NAMES:
                metric_name = f"WA_{label_name}"
                val_log += f" {metric_name}: {val_metrics[metric_name][-1]:.2f} |"
            val_log += f" Average_bWA: {val_metrics['Average_bWA'][-1]:.2f} |"
            print(val_log)
            print()

            (save_dir / "plots").mkdir(parents=True, exist_ok=True)
            plt.plot(train_metrics["Average_bWA"], label="Train")
            plt.plot(val_metrics["Average_bWA"], label="Val")
            plt.legend()
            plt.savefig(save_dir / "plots" / f"Average_bWA-{epoch_idx}")
            plt.close()

            plt.plot(train_metrics["loss"], label="Train")
            plt.plot(val_metrics["loss"], label="Val")
            plt.legend()
            plt.savefig(save_dir / "plots" / f"Loss-{epoch_idx}")
            plt.close()

            start = time.time()

        if verbose and (epoch_idx % save_epoch == 0):
            (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                save_dir / "checkpoints" / f"{epoch_idx}.pt",
            )

    return (
        (train_targets_per_epoch, train_predits_per_epoch, train_metrics),
        (val_targets_per_epoch, val_predits_per_epoch, val_metrics),
        (best_model, best_epoch),
    )


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    model.train()

    targets = list()
    predictions = list()
    for batch in dataloader:
        X_text, X_audio, y = batch

        X_text: torch.FloatTensor = X_text.to(device)
        X_audio: torch.FloatTensor = X_audio.to(device)
        y: torch.FloatTensor = y.to(device)

        y_pred: torch.FloatTensor = model(X_text, X_audio)
        loss: torch.FloatTensor = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)

        targets.append(y.detach().cpu())
        predictions.append(y_pred.detach().cpu())

    targets = torch.cat(targets)
    predictions = torch.cat(predictions)

    return targets, predictions


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    model.eval()

    targets = list()
    predictions = list()
    for batch in dataloader:
        X_text, X_audio, y = batch

        X_text: torch.FloatTensor = X_text.to(device)
        X_audio: torch.FloatTensor = X_audio.to(device)
        y: torch.FloatTensor = y.to(device)

        y_pred: torch.FloatTensor = model(X_text, X_audio)
        targets.append(y.cpu())
        predictions.append(y_pred.cpu())

    targets = torch.cat(targets)
    predictions = torch.cat(predictions)

    return targets, predictions
