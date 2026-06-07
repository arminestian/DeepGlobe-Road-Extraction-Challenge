import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def predict_image(model, image_path):
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (1024, 1024))

    x = torch.from_numpy(img_resized).float() / 255.0
    x = x.permute(2,0,1).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0,0].cpu().numpy()

    mask = (pred > 0.3).astype(np.float16)

    return img_resized, mask


img, mask = predict_image(model, os.path.join(valid_path, os.listdir(valid_path)[500]))

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Input Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()