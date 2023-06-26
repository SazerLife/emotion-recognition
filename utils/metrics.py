import torch


def perf_measure(predicts, targets):
    TP, FP = 0, 0
    TN, FN = 0, 0

    for i in range(len(predicts)):
        if targets[i] == predicts[i] == 1:
            TP += 1
        if predicts[i] == 1 and targets[i] != predicts[i]:
            FP += 1
        if targets[i] == predicts[i] == 0:
            TN += 1
        if predicts[i] == 0 and targets[i] != predicts[i]:
            FN += 1

    return (TP, FP, TN, FN)


def binary_weighted_accuracy(predicts: torch.Tensor, targets: torch.Tensor):
    assert len(predicts.shape) == 1 and len(targets.shape) == 1
    TP, _, TN, _ = perf_measure(predicts.tolist(), targets.tolist())
    P, N = len(predicts == 1), len(predicts == 0)

    weighted_accuracy = (TN + TP * N / P) / (2 * N)
    return weighted_accuracy
