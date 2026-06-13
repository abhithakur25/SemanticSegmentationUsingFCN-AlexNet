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
LABEL_FOLDER = os.path.join(DATA_ROOT, 'LabelsReszed')  # unused in this script
RESULTS_FOLDER = os.path.join(DATA_ROOT, 'forgery_results_5models_python_latest')
FIG_FOLDER = os.path.join(RESULTS_FOLDER, 'figures')
PATCH_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'patch_overlays')  # unused here
SEG_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'segmentation_overlays')  # unused here
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)
os.makedirs(PATCH_OVERLAY_FOLDER, exist_ok=True)
os.makedirs(SEG_OVERLAY_FOLDER, exist_ok=True)

# ---------------- Settings ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-4
RESULTS_SUB = os.path.join(RESULTS_FOLDER, 'mobilenetv3_large')
os.makedirs(RESULTS_SUB, exist_ok=True)

def main():
    # Data (latest weights & transforms)
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    train_tf = weights.transforms()
    val_tf   = weights.transforms()

    ds = datasets.ImageFolder(IMAGE_FOLDER, transform=train_tf)
    class_names = ds.classes
    n_train = int(0.8 * len(ds))
    n_val   = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform   = val_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model: MobileNetV3-Large
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total, correct, running = 0, 0, 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            running += float(loss.item()) * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1} - Train loss {running/total:.4f} | acc {correct/total:.4f}")

        model.eval()
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = out.argmax(1).cpu().numpy()
                y_scores.extend(probs.tolist())
                y_pred.extend(preds.tolist())
                y_true.extend(labels.cpu().numpy().tolist())

        acc = (np.array(y_true) == np.array(y_pred)).mean()
        print(f"Epoch {epoch+1} - Val acc {acc:.4f}")
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(RESULTS_SUB, "best_mobilenetv3_large.pth"))
            best_acc = acc

        # Metrics & plots
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
        json.dump({"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()},
                  open(os.path.join(RESULTS_SUB, "metrics.json"), "w"), indent=2)

        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion")
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_SUB, "confusion.png"))
        plt.close()

        try:
            y_true_np = np.array(y_true)
            y_scores_np = np.array(y_scores)
            plt.figure()
            for i, c in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_true_np == i, y_scores_np[:, i])
                plt.plot(fpr, tpr, label=f"{c} AUC={auc(fpr,tpr):.3f}")
            plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (OvR)")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_SUB, "roc.png"))
            plt.close()
        except Exception:
            pass

    print("Done. Results:", RESULTS_SUB)

if __name__ == "__main__":
    main()
