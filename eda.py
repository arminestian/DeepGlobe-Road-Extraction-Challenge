import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data_path  = "C:/Users/armin/Downloads/Documents/Datasets/Deep Globe/archive"
train_path = data_path + "/train"
valid_path = data_path + "/valid"
test_path  = data_path + "/test"


test_mask  = cv2.imread(os.path.join(train_path, os.listdir(train_path)[0]))
test_image = cv2.imread(os.path.join(train_path, os.listdir(train_path)[1]))

# Convert BGR (OpenCV) to RGB (Matplotlib)
test_mask  = cv2.cvtColor(test_mask, cv2.COLOR_BGR2RGB)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))

axes[0].imshow(test_mask)
axes[0].set_title("Mask")
axes[0].axis("off")

axes[1].imshow(test_image)
axes[1].set_title("Image")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# Split train to 80/20

files = os.listdir(train_path)

# Extract unique sample IDs
sample_ids = sorted({
    filename.split('_')[0]
    for filename in files
})

print(f"Number of samples: {len(sample_ids)}")

# 80/20 split
train_ids, val_ids = train_test_split(
    sample_ids,
    test_size=0.2,
    random_state=42
)

print(f"Train samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")

train_pairs = []
val_pairs = []

for sample_id in train_ids:
    image_path = os.path.join(train_path, f"{sample_id}_sat.jpg")
    mask_path = os.path.join(train_path, f"{sample_id}_mask.png")
    train_pairs.append((image_path, mask_path))

for sample_id in val_ids:
    image_path = os.path.join(train_path, f"{sample_id}_sat.jpg")
    mask_path = os.path.join(train_path, f"{sample_id}_mask.png")
    val_pairs.append((image_path, mask_path))