# training loop unet
import torch
import tqdm
from Dataset import train_loader


num_epochs = 30

scaler = torch.amp.GradScaler('cuda')

for epoch in range(num_epochs):

    model.train()

    train_loss = 0.0
    train_iou = 0.0
    train_dice = 0.0
    n_batches = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for imgs, masks in loop:

        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # with torch.amp.autocast('cuda'):

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        # ---- safety check (IMPORTANT) ----
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {n_batches}")
            break

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # ---- metrics (detach + float-safe) ----
        with torch.no_grad():
            iou = iou_score(preds, masks)
            dice = dice_score(preds, masks)

        train_loss += loss.item()
        train_iou += iou.item()
        train_dice += dice.item()

        n_batches += 1

        loop.set_postfix(loss=loss.item())

    # ---- avoid division by zero ----
    n_batches = max(n_batches, 1)

    train_metrics = {
        "loss": train_loss / n_batches,
        "iou": train_iou / n_batches,
        "dice": train_dice / n_batches,
    }

    val_metrics = evaluate(model, val_loader)

    print(f"""
Epoch {epoch+1}
Train Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} | Dice: {train_metrics['dice']:.4f}
Val   Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f}
""")