from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2



class RoadDataset(Dataset):
    def __init__(self, pairs, patch_size=256, transform=None):
        self.pairs = pairs
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask
            )
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image.transpose(2,0,1))
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
    


train_ds = RoadDataset(train_pairs)
val_ds = RoadDataset(val_pairs)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=0
)