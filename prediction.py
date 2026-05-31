import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.signal import istft
import segmentation_models_pytorch as smp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=7
).to(device)

model.load_state_dict(torch.load("/home/dante/code/Unet24_jamming/training_logs/best_unet.pth", map_location=device))
model.eval()


data = sio.loadmat("/home/dante/code/Unet24_jamming/Dataset/data_train_unet/LWM+TRI_-100_00001.mat")

X = data["s_mix"]

X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(X)

mask1 = pred[0,0]
mask2 = pred[0,1]

H, W = X.shape[2], X.shape[3]

mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear').squeeze().cpu().numpy()
mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear').squeeze().cpu().numpy()

img = X[0,0].detach().cpu().numpy()
img = np.log1p(img)

# normalization
mask1_vis = mask1
mask2_vis = mask2

mask1_vis = (mask1_vis - mask1_vis.min()) / (mask1_vis.max() - mask1_vis.min() + 1e-8)
mask2_vis = (mask2_vis - mask2_vis.min()) / (mask2_vis.max() - mask2_vis.min() + 1e-8)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Input |STFT|^2 (log)")
plt.imshow(img, aspect='auto', cmap='jet')
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Mask 1 (normalized)")
plt.imshow(mask1_vis, aspect='auto', cmap='jet')
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Mask 2 (normalized)")
plt.imshow(mask2_vis, aspect='auto', cmap='jet')
plt.colorbar()

plt.tight_layout()
plt.show()