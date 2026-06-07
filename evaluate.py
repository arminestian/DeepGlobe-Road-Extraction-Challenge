from tqdm import tqdm
import torch

def evaluate(model, loader):
    model.eval()

    total_loss = 0
    total_iou = 0
    total_dice = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.cuda()
            masks = masks.cuda()

            preds = model(imgs)

            loss = loss_fn(preds, masks)

            total_loss += loss.item()
            total_iou += iou_score(preds, masks).item()
            total_dice += dice_score(preds, masks).item()

    n = len(loader)

    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n
    }