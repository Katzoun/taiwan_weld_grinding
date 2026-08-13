#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train_pointcnn.py
import torch
import os
from glob import glob
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset.wireharness_dataset import WireHarenessDataset

# Training parameters
data_dir = "D:/PartAnnotation/00000007/points"   # dataset path
batch_size = 8
num_epochs = 200    # maximum number of training epochs
num_classes = 2     # number of segmentation classes

# Collect all .txt files
all_files = sorted(glob(os.path.join(data_dir, "*.txt")))

# Train/val split
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)   # 80% train / 20% val

# Build datasets
train_set = WireHarenessDataset(train_files)
val_set = WireHarenessDataset(val_files)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)  # training loader
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)     # validation loader


# In[5]:


# Check the point count per sample in train_loader
print("Train dataset point count per sample:")
for batch in train_loader:
    pcs = batch["pcd"]  # shape: [B, N, 3]
    print(f"Point cloud shape: {pcs.shape}")  # print the whole batch's shape
    for i in range(pcs.shape[0]):
        print(f" Sample {i}: {pcs[i].shape[0]} points")  # print each sample's point count
    break  # only print the first batch


# In[9]:


import importlib
# import pointcnn_seg

# importlib.reload(pointcnn_seg)  # reload the whole module

# from pointcnn_seg import PointCNNSeg  # re-import the class

import pointcnn_seg_1024
importlib.reload(pointcnn_seg_1024)  # reload the whole module
from pointcnn_seg_1024 import PointCNNSeg  # re-import the class

learning_rate = 1e-4  # learning rate

# Model
model = PointCNNSeg(num_classes=num_classes)

# Save the checkpoint with the best validation accuracy (based on val_acc_epoch)
checkpoint_callback = ModelCheckpoint(
    dirpath="C:/Users/mmdl/Desktop/Jiayuan/",    # folder to save model checkpoints in
    filename="pointcnn-best_8_1024",             # checkpoint filename
    save_top_k=1,
    monitor="val_acc_epoch",
    mode="max"
)

# Early stopping: stop training if accuracy doesn't improve for a number of epochs
early_stop_callback = EarlyStopping(
    monitor="val_acc_epoch",         # early-stopping metric: validation accuracy
    patience=10,                     # criterion 1: stop if accuracy hasn't improved in 10 epochs
    min_delta=0.0005,                # criterion 2: an improvement only counts if it's > 0.0005
    mode="max",
    verbose=True
)

# Trainer
trainer = Trainer(
    max_epochs=num_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint_callback, early_stop_callback],
    log_every_n_steps=10
)

# Start training
trainer.fit(model, train_loader, val_loader)


# In[10]:


print("Final train accuracy:", trainer.callback_metrics.get("train_acc_epoch"))
print("Final val accuracy:", trainer.callback_metrics.get("val_acc_epoch"))


# In[11]:


ckpt_path = "C:/Users/mmdl/Desktop/Jiayuan/pointcnn-best_8_1024.ckpt"

model = PointCNNSeg.load_from_checkpoint(
    ckpt_path,
    num_classes=2,              # number of segmentation classes
    weight_balance=None
)

model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

correct = 0
total = 0

with torch.no_grad():
    for batch in train_loader:
        x, y = batch['pcd'], batch['label']
        x, y = x.to(model.device), y.to(model.device)

        y_hat = model(x)  # (B, C, N)
        y_hat = y_hat.permute(0, 2, 1)  # (B, N, C)
        pred = torch.argmax(y_hat, dim=-1)  # (B, N)

        correct += (pred.view(-1) == y.view(-1)).sum().item()
        total += y.numel()

train_acc = correct / total
print(f"Train Accuracy (post-hoc): {train_acc:.4f}")


# In[12]:


import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Pick a random validation sample
sample = random.choice(val_set)
pcd = sample['pcd'].unsqueeze(0).to(model.device)  # add batch dimension
gt_label = sample['label'].numpy()

# Model prediction
model.eval()
with torch.no_grad():
    pred_logits = model(pcd)
    pred_classes = torch.argmax(pred_logits.squeeze(0).permute(1, 0), dim=1).cpu().numpy()

# Original point cloud coordinates
coords = pcd.squeeze(0).cpu().numpy()

# Plot
fig = plt.figure(figsize=(12, 6))

# Ground Truth
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title("Ground Truth")
colors_gt = ['blue' if l == 0 else 'red' for l in gt_label]
ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors_gt, s=2)

# Prediction
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title("Model Prediction")
colors_pred = ['blue' if l == 0 else 'red' for l in pred_classes]
ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors_pred, s=2)

plt.tight_layout()
plt.show()


# In[ ]:




