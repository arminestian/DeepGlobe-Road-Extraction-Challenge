import torch.nn as nn
import torch

bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()

    dice = (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

    return 1 - dice


def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)