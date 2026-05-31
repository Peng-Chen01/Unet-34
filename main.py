import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

# =========================================================
# Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# =========================================================
# Config
# =========================================================

X_DIR = "/home/dante/code/Unet24_jamming/Dataset/data_train_unet/X"

Y_DIR = "/home/dante/code/Unet24_jamming/Dataset/data_train_unet/Y"

SAVE_DIR = "./training_logs"

NUM_CLASSES = 7

BATCH_SIZE = 8

EPOCHS = 100

LR = 3e-4

PAD_MULTIPLE = 32

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# Padding
# =========================================================

def pad_to_multiple(x, y, multiple=16):

    _, H, W = x.shape

    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple

    x = F.pad(x, (0, pad_w, 0, pad_h))

    y = F.pad(y, (0, pad_w, 0, pad_h))

    return x, y

# =========================================================
# Dataset
# =========================================================

class STFTDataset(Dataset):

    def __init__(self, x_dir, y_dir):

        self.x_dir = x_dir
        self.y_dir = y_dir

        self.files = sorted(os.listdir(x_dir))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        x_name = self.files[idx]

        # =================================================
        # Generate label filename
        # =================================================

        name = x_name.replace(".mat", "")

        parts = name.rsplit("_", 2)

        class_name = parts[0]
        power = parts[1]
        idx_num = parts[2]

        y_name = f"{class_name}_mask_{power}_{idx_num}.mat"

        # =================================================
        # Paths
        # =================================================

        x_path = os.path.join(self.x_dir, x_name)

        y_path = os.path.join(self.y_dir, y_name)

        # =================================================
        # Load
        # =================================================

        x = sio.loadmat(x_path)["s_mix"]

        y = sio.loadmat(y_path)["s_mask"]

        # =================================================
        # preprocess
        # =================================================

        x = np.abs(x).astype(np.float32)

        x = np.log1p(x)

        # =================================================
        # Tensor
        # =================================================

        x = torch.tensor(x).unsqueeze(0)

        # (H,W,C) -> (C,H,W)

        y = torch.tensor(
            y,
            dtype=torch.float32
        ).permute(2,0,1)

        # =================================================
        # Padding
        # =================================================

        x, y = pad_to_multiple(
            x,
            y,
            PAD_MULTIPLE
        )

        return x, y

# =========================================================
# Loader
# =========================================================

dataset = STFTDataset(
    X_DIR,
    Y_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print("Dataset size:", len(dataset))

# =========================================================
# Model
# =========================================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=NUM_CLASSES
).to(device)

# =========================================================
# Loss
# =========================================================

bce = nn.BCEWithLogitsLoss()

l1 = nn.L1Loss()

def criterion(pred, target):

    prob = torch.sigmoid(pred)

    return (
        bce(pred, target)
        + 0.5 * l1(prob, target)
    )

# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# =========================================================
# Fixed visualization sample
# =========================================================

x_vis, y_vis = dataset[0]

x_vis = x_vis.unsqueeze(0)

y_vis = y_vis.numpy()

# =========================================================
# Training
# =========================================================

best_loss = 1e9

loss_history = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    print(f"\n================ Epoch {epoch} ================")

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device).float()

        y = y.to(device).float()

        pred = model(x)

        loss = criterion(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    loss_history.append(avg_loss)

    print(f"Loss: {avg_loss:.6f}")

    # =====================================================
    # Save checkpoint
    # =====================================================

    torch.save(
        model.state_dict(),
        os.path.join(
            SAVE_DIR,
            f"checkpoint_epoch_{epoch}.pth"
        )
    )

    # =====================================================
    # Save best
    # =====================================================

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            os.path.join(
                SAVE_DIR,
                "best_unet.pth"
            )
        )

        print("Saved best model.")

    # =====================================================
    # Visualization
    # =====================================================

    model.eval()

    with torch.no_grad():

        pred_vis = model(
            x_vis.to(device)
        )

        pred_vis = torch.sigmoid(pred_vis)

        pred_vis = pred_vis[0].cpu().numpy()

    # =====================================================
    # Save figure
    # =====================================================

    fig = plt.figure(figsize=(20,20))

    # =====================================================
    # Input
    # =====================================================

    plt.subplot(5,3,1)

    plt.title("Input STFT")

    plt.imshow(
        x_vis[0,0].cpu().numpy(),
        aspect='auto',
        cmap='jet'
    )

    plt.colorbar()

    # =====================================================
    # GT + Prediction
    # =====================================================

    for c in range(NUM_CLASSES):

        # GT
        plt.subplot(5,3,c+2)

        plt.title(f"GT Class {c}")

        plt.imshow(
            y_vis[c],
            aspect='auto',
            cmap='jet'
        )

        plt.colorbar()

        # Prediction
        plt.subplot(5,3,c+2+NUM_CLASSES)

        plt.title(f"Pred Class {c}")

        plt.imshow(
            pred_vis[c],
            aspect='auto',
            cmap='jet'
        )

        plt.colorbar()

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            SAVE_DIR,
            f"epoch_{epoch}.png"
        )
    )

    plt.close(fig)

    torch.cuda.empty_cache()

# =========================================================
# Save Loss Curve
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(
    loss_history,
    marker='o'
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "loss_curve.png"
    )
)

plt.close()

print("\nTraining Finished.")
