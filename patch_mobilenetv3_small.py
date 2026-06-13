#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================== Imports (kept in your sequence) ===================
import os
import sys
import random
import shutil
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
import torchvision.transforms.functional as TF

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
# ======================================================================

# ---------------- USER CONFIG ----------------
DATA_ROOT = r'D:/claude/SemanticSegmentationUsingFCN-AlexNet1/Dataset4/'
IMAGE_FOLDER = os.path.join(DATA_ROOT, 'ImagesReszed')
LABEL_FOLDER = os.path.join(DATA_ROOT, 'LabelsReszed')  # unused here
RESULTS_FOLDER = os.path.join(DATA_ROOT, 'forgery_results_5models_python_latest')
FIG_FOLDER = os.path.join(RESULTS_FOLDER, 'figures')  # unused here
PATCH_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'patch_overlays')
SEG_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'segmentation_overlays')  # unused here
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)
os.makedirs(PATCH_OVERLAY_FOLDER, exist_ok=True)
os.makedirs(SEG_OVERLAY_FOLDER, exist_ok=True)

# ---------------- Settings ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (64, 64)
PATCH_STRIDE = 32
NET_INPUT = (224, 224)
BATCH_SIZE = 16
EPOCHS = 4
LR = 1e-4
HEAT_THRESHOLD = 0.35
RESULTS_SUB = os.path.join(RESULTS_FOLDER, 'patch_mobilenetv3_small')
TMP_TRAIN = os.path.join(RESULTS_SUB, 'patch_train')
TMP_VAL   = os.path.join(RESULTS_SUB, 'patch_val')
os.makedirs(RESULTS_SUB, exist_ok=True)

def generate_patches(paths, labels, outdir):
    if os.path.exists(outdir): shutil.rmtree(outdir)
    os.makedirs(outdir)
    for p, lab in zip(paths, labels):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        cls_dir = os.path.join(outdir, str(lab))
        os.makedirs(cls_dir, exist_ok=True)
        idx = 0
        for y in range(0, max(1, H - PATCH_SIZE[1] + 1), PATCH_STRIDE):
            for x in range(0, max(1, W - PATCH_SIZE[0] + 1), PATCH_STRIDE):
                pr = img.crop((x, y, x + PATCH_SIZE[0], y + PATCH_SIZE[1])).resize(NET_INPUT)
                pr.save(os.path.join(cls_dir, f'{idx:05d}.png')); idx += 1

def main():
    # Base dataset & splits
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    tf_train = weights.transforms()
    tf_val   = weights.transforms()

    full = datasets.ImageFolder(IMAGE_FOLDER, transform=tf_train)
    class_names = full.classes
    n_train = int(0.8 * len(full))
    n_val   = len(full) - n_train
    train_idx = list(range(len(full))); random.shuffle(train_idx)
    train_idx, val_idx = train_idx[:n_train], train_idx[n_train:]
    train_paths = [full.samples[i][0] for i in train_idx]
    train_labels= [full.samples[i][1] for i in train_idx]
    val_paths   = [full.samples[i][0] for i in val_idx]
    val_labels  = [full.samples[i][1] for i in val_idx]

    # Patch extraction
    print("Generating patches...")
    generate_patches(train_paths, train_labels, TMP_TRAIN)
    generate_patches(val_paths,   val_labels,   TMP_VAL)

    train_ds = datasets.ImageFolder(TMP_TRAIN, transform=tf_train)
    val_ds   = datasets.ImageFolder(TMP_VAL,   transform=tf_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model: MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
        print("Epoch finished.")

    # Heatmaps (assume class index 0 == 'Forged' if present, else first class found containing 'forg')
    forged_idx = 0
    for i, c in enumerate(class_names):
        if 'forg' in c.lower():
            forged_idx = i; break

    os.makedirs(PATCH_OVERLAY_FOLDER, exist_ok=True)
    for i, p in enumerate(val_paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        rows = max(1, (H - PATCH_SIZE[1]) // PATCH_STRIDE + 1)
        cols = max(1, (W - PATCH_SIZE[0]) // PATCH_STRIDE + 1)
        heat = np.zeros((rows, cols), dtype=np.float32)
        r = 0
        with torch.no_grad():
            for y in range(0, max(1, H - PATCH_SIZE[1] + 1), PATCH_STRIDE):
                c = 0
                for x in range(0, max(1, W - PATCH_SIZE[0] + 1), PATCH_STRIDE):
                    pr = img.crop((x, y, x + PATCH_SIZE[0], y + PATCH_SIZE[1])).resize(NET_INPUT)
                    pr_t = tf_val(pr).unsqueeze(0).to(DEVICE)
                    prob = torch.softmax(model(pr_t), dim=1)[0, forged_idx].item()
                    heat[r, c] = prob
                    c += 1
                r += 1
        # Smooth & normalize
        try:
            from scipy.ndimage import gaussian_filter
            heat = gaussian_filter(heat, sigma=1.0)
        except Exception:
            pass
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        heat_np = np.array(heat_img) / 255.0
        mask = heat_np > HEAT_THRESHOLD
        overlay = np.array(img).astype(np.float32)
        overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
        outpath = os.path.join(PATCH_OVERLAY_FOLDER, f'patch_overlay_{i:04d}.png')
        Image.fromarray(overlay.astype(np.uint8)).save(outpath)

    print("Patch overlays saved to", PATCH_OVERLAY_FOLDER)

if __name__ == "__main__":
    main()
