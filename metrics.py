import torch

def dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()

    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2 * intersection + smooth) / (union + smooth)

def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()

    return intersection / (union + 1e-6)