#!/usr/bin/env python3
"""
forgery_models_5version_compare_latest_gpu_only.py

GPU-only: trains/evaluates multiple models with robust dataset handling.

Models (updated to modern backbones):
- A: EfficientNet-V2-S (was TinyCNN)
- B: ConvNeXt-Tiny (was ResNet-50)
- C: MobileNetV3-Large (was MobileNetV2)
- Patch classifier: MobileNetV3-Small
- Segmentation: DeepLabV3-MobileNetV3-Large -> fallback DeepLabV3-ResNet50 -> fallback UNet

Author: ChatGPT
"""
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

# ---------------- USER CONFIG ----------------
DATA_ROOT = r'D:/claude/SemanticSegmentationUsingFCN-AlexNet1/Dataset4'
IMAGE_FOLDER = os.path.join(DATA_ROOT, 'ImagesReszed')
LABEL_FOLDER = os.path.join(DATA_ROOT, 'LabelsReszed')
RESULTS_FOLDER = os.path.join(DATA_ROOT, 'forgery_results_5models_python_latest')
FIG_FOLDER = os.path.join(RESULTS_FOLDER, 'figures')
PATCH_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'patch_overlays')
SEG_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'segmentation_overlays')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)
os.makedirs(PATCH_OVERLAY_FOLDER, exist_ok=True)
os.makedirs(SEG_OVERLAY_FOLDER, exist_ok=True)

# Pipelines to run
RUN_TINY = False                 # EfficientNetV2-S
RUN_CONVNEXT_TINY = False        # ConvNeXt-Tiny
RUN_MOBILENETV3_L = False        # MobileNetV3-Large
RUN_PATCH_ENSEMBLE = False       # MobileNetV3-Small patch classifier
RUN_SEGMENTATION = False         # DeepLabV3 MobileNetV3-L

# --- GPU-only enforcement ---
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. This script must run on a CUDA-enabled NVIDIA GPU.\n"
        "Install a CUDA-enabled PyTorch build and verify NVIDIA drivers."
    )
DEVICE = torch.device('cuda')
print('Device:', DEVICE, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())
try:
    print('Current GPU:', torch.cuda.get_device_name(0))
except Exception:
    pass

# Hyperparameters
SEED = 0
BATCH_SIZE = 16
MAX_EPOCHS_CLASS = 1
MAX_EPOCHS_SEG = 1
INITIAL_LR = 1e-4
L2_REG = 1e-4
INPUT_SIZE = (224, 224)        # classifier input (H,W)
SEGMENTATION_SIZE = (360, 480) # segmentation (H,W)
PATCH_SIZE = (64, 64)
PATCH_STRIDE = 32
PATCH_NET_INPUT = (224, 224)
PATCH_HEATMAP_THRESHOLD = 0.35

# ImageNet normalization (works across all torchvision pretrains)
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

# Augmentation / preprocessing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    transforms.RandomAffine(degrees=0, translate=(6/INPUT_SIZE[1], 6/INPUT_SIZE[0])),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])

seg_img_transform = transforms.Compose([
    transforms.Resize(SEGMENTATION_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])

# seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- Helper utilities ----------------
def list_image_files(folder, exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
    files = []
    for ext in exts:
        files.extend(sorted(Path(folder).rglob(f'*{ext}')))
    return [str(p) for p in files]

def fuzzy_pair_images_labels(img_folder, label_folder):
    """Match images and label masks by basename using fuzzy logic."""
    img_files = list_image_files(img_folder)
    lbl_files = list_image_files(label_folder)
    img_basenames = [Path(p).stem for p in img_files]
    lbl_basenames = [Path(p).stem for p in lbl_files]
    lbl_map = {bn.lower(): p for bn, p in zip(lbl_basenames, lbl_files)}
    matches_img, matches_lbl = [], []
    # exact
    for i, bn in enumerate(img_basenames):
        key = bn.lower()
        if key in lbl_map:
            matches_img.append(img_files[i]); matches_lbl.append(lbl_map[key])
    if matches_img: return matches_img, matches_lbl
    # contains
    for i, bn in enumerate(img_basenames):
        low = bn.lower()
        for j, lb in enumerate(lbl_basenames):
            if low in lb.lower() or lb.lower() in low:
                matches_img.append(img_files[i]); matches_lbl.append(lbl_files[j]); break
    if matches_img: return matches_img, matches_lbl
    # numeric prefix
    import re
    for i, bn in enumerate(img_basenames):
        m = re.search(r'(\d+)$', bn)
        if m:
            prefix = re.sub(r'\d+$', '', bn)
            for j, lb in enumerate(lbl_basenames):
                if Path(lbl_files[j]).stem.startswith(prefix):
                    matches_img.append(img_files[i]); matches_lbl.append(lbl_files[j]); break
    # fallback same-length pairing
    if not matches_img and len(img_files)==len(lbl_files) and len(img_files)>0:
        print('Fuzzy pairing fallback: pairing by order (verify!).')
        return img_files, lbl_files
    return matches_img, matches_lbl

# ---------------- Datasets ----------------
class ClassificationImageFolder(Dataset):
    """Simple dataset around lists of image paths + labels."""
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        p = self.filepaths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform: img = self.transform(img)
        label = self.labels[idx]
        return img, label, p

def build_classification_datasets(image_root, input_transform, val_transform):
    """Supports standard ImageFolder or infers labels from filenames if no subfolders."""
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f'image folder not found: {image_root}')
    # Try ImageFolder
    try:
        imfolder = datasets.ImageFolder(image_root)
        if imfolder.classes:
            class_names = imfolder.classes
            filepaths = [s[0] for s in imfolder.imgs]
            labels = [s[1] for s in imfolder.imgs]
        else:
            imfolder = None
    except Exception:
        imfolder = None

    if imfolder is None or not getattr(imfolder, 'classes', []):
        print(f'No class subfolders found in {image_root} — inferring labels from filenames...')
        rootp = Path(image_root)
        exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
        files = sorted([p for p in rootp.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if not files:
            raise FileNotFoundError(f'No image files found in {image_root}')
        keyword_map = {
            'forged':'forged','fake':'forged','tamper':'forged','tampered':'forged',
            'original':'original','auth':'original','real':'original','genuine':'original','pristine':'original'
        }
        inferred = []
        for p in files:
            nm = p.stem.lower()
            lab = None
            for kw,canon in keyword_map.items():
                if kw in nm: lab=canon; break
            if lab is None:
                toks = [t for sep in ['_','-','.'] for t in nm.split(sep) if t]
                if toks and not toks[0].isdigit(): lab = toks[0]
                if lab is None and toks and not toks[-1].isdigit(): lab = toks[-1]
            if lab is None: lab='unknown'
            inferred.append((str(p), lab))
        classes = sorted(list(set(l for _,l in inferred)))
        if 'unknown' in classes and len(classes)==1:
            sample = [f.name for f in files[:10]]
            raise RuntimeError(f"Could not infer labels from filenames. Examples: {sample}. "
                               "Use subfolders per class or include tokens like 'forged'/'original'.")
        class_to_idx = {c:i for i,c in enumerate(classes)}
        filepaths = [p for p,_ in inferred]
        labels = [class_to_idx[l] for _,l in inferred]
        class_names = classes

    # filter unreadable
    good_paths, good_labels = [], []
    for p,l in zip(filepaths, labels):
        try:
            with Image.open(p) as _:
                good_paths.append(p); good_labels.append(l)
        except Exception:
            print(f'Warning: could not read image {p}; skipped.')
    filepaths, labels = good_paths, good_labels
    if not filepaths:
        raise RuntimeError('No readable images after filtering.')

    # stratified split 80/20
    idx_per_label = defaultdict(list)
    for i,lab in enumerate(labels): idx_per_label[lab].append(i)
    tr_idx, va_idx = [], []
    for lab, idxs in idx_per_label.items():
        random.shuffle(idxs)
        ntr = max(1, int(0.8*len(idxs))) if len(idxs)>=2 else len(idxs)
        tr_idx += idxs[:ntr]; va_idx += idxs[ntr:]
    random.shuffle(tr_idx); random.shuffle(va_idx)

    tr_paths = [filepaths[i] for i in tr_idx]; tr_labels = [labels[i] for i in tr_idx]
    va_paths = [filepaths[i] for i in va_idx]; va_labels = [labels[i] for i in va_idx]

    train_ds = ClassificationImageFolder(tr_paths, tr_labels, transform=input_transform)
    val_ds   = ClassificationImageFolder(va_paths, va_labels, transform=val_transform)
    return train_ds, val_ds, class_names

class SegmentationDataset(Dataset):
    """Image/mask dataset; masks mapped to {0,1} via heuristic or class_map."""
    def __init__(self, image_paths, mask_paths, img_transform=None, class_map=None):
        assert len(image_paths)==len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transform
        self.class_map = class_map
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        im = Image.open(self.image_paths[idx]).convert('RGB')
        m  = Image.open(self.mask_paths[idx])
        if m.mode!='L': m = m.convert('L')
        m_np = np.array(m, dtype=np.int64)
        if self.class_map is not None:
            mapped = np.zeros_like(m_np, dtype=np.int64)
            for k,v in self.class_map.items():
                mapped[m_np==int(k)] = int(v)
            m_np = mapped
        else:
            uniq = set(np.unique(m_np).tolist())
            if uniq <= {0,1}:
                pass
            elif uniq <= {1,2}:
                m_np = m_np-1
            else:
                m_np = (m_np>127).astype(np.int64)
        if self.img_transform: im_t = self.img_transform(im)
        else: im_t = TF.normalize(TF.to_tensor(im), IMNET_MEAN, IMNET_STD)
        m_t = torch.from_numpy(m_np).long()
        return im_t, m_t, self.image_paths[idx]

# ---------------- Latest model builders ----------------
def build_effnetv2_s(num_classes):
    try:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    except Exception:
        # fallback
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def build_convnext_tiny(num_classes):
    try:
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    except Exception:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_mobilenetv3_large(num_classes):
    try:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    except Exception:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def build_mobilenetv3_small(num_classes):
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    except Exception:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def build_deeplabv3_mnv3_large(num_classes):
    # Try newer MobileNetV3-Large backbone first, then ResNet50
    try:
        weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return model
    except Exception:
        try:
            model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
            return model
        except Exception:
            return None  # will fallback to UNet

class UNetSmall(nn.Module):
    def __init__(self, num_classes, in_ch=3, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base,3,padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base, base*2,3,padding=1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base*2, base*4,3,padding=1), nn.ReLU(True))
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base*4, base*2,3,padding=1), nn.ReLU(True))
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base*2, base,3,padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(base, num_classes, 1)
    def forward(self,x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        u1 = self.up1(e3); d1 = self.dec1(torch.cat([u1,e2], dim=1))
        u2 = self.up2(d1); d2 = self.dec2(torch.cat([u2,e1], dim=1))
        return self.out(d2)

# ---------------- Train / Eval helpers ----------------
def train_classifier(model, train_loader, val_loader, num_epochs, lr, save_path=None):
    """Training loop (supports (img,label,path) or (img,label))."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=L2_REG)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{num_epochs}"):
            if len(batch)==3: imgs, labels, _ = batch
            elif len(batch)==2: imgs, labels = batch
            else: raise ValueError(f"Unexpected batch length: {len(batch)}")
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += float(loss.item()) * imgs.size(0)
            total_train += imgs.size(0)
        scheduler.step()
        train_loss = running_loss / max(1,total_train)

        # val
        model.eval()
        running_loss = 0.0; correct=0; total=0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch)==3: imgs, labels, _ = batch
                else: imgs, labels = batch
                imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += float(loss.item()) * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
        val_loss = running_loss / max(1,total)
        val_acc = correct / max(1,total)
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if save_path and val_acc>best_val_acc:
            torch.save(model.state_dict(), save_path)
            best_val_acc = val_acc
    return model

def evaluate_classifier(model, data_loader, class_names):
    """Return confusion, per-class P/R/F1, y_true, y_pred, scores."""
    model.eval()
    y_true=[]; y_pred=[]; y_scores=[]
    with torch.no_grad():
        for batch in data_loader:
            if len(batch)==3: imgs, labels, _ = batch
            else: imgs, labels = batch
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_scores.extend(probs.tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    return {
        'confusion': cm,
        'precision': np.array(precision),
        'recall': np.array(recall),
        'f1': np.array(f1),
        'y_true': np.array(y_true),
        'y_pred': np.array(y_pred),
        'scores': np.array(y_scores),
    }

def plot_and_save_confusion(cm, class_names, out_path, title='Confusion matrix'):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right'); ax.set_yticklabels(class_names)
    ax.set_ylabel('True label'); ax.set_xlabel('Predicted label'); ax.set_title(title)
    thresh = cm.max()/2.0 if cm.size>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j]>thresh else 'black')
    fig.tight_layout(); plt.savefig(out_path); plt.close(fig)

def plot_rocs(scores, y_true, class_names, out_path_prefix):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plotted=False
    for c in range(len(class_names)):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:,c], scores[:,c])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[c]} (AUC={roc_auc:.3f})')
            plotted=True
        except Exception:
            pass
    if plotted:
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (one-vs-rest)'); plt.legend(); plt.grid(True)
        plt.savefig(out_path_prefix + '_roc.png')
        plt.close()
    else:
        print('ROC not plotted (labels likely constant or single class).')

# ---------------- Patch utilities ----------------
def generate_patches_from_image(img_path, label, patch_size, stride, out_folder, net_input_size):
    os.makedirs(out_folder, exist_ok=True)
    lbl_folder = os.path.join(out_folder, str(label)); os.makedirs(lbl_folder, exist_ok=True)
    I = Image.open(img_path).convert('RGB'); W,H = I.size; cnt=0
    for y in range(0, H - patch_size[1] + 1, stride):
        for x in range(0, W - patch_size[0] + 1, stride):
            pr = I.crop((x, y, x+patch_size[0], y+patch_size[1])).resize(net_input_size)
            pr.save(os.path.join(lbl_folder, f'patch_{cnt:05d}.png')); cnt+=1
    return cnt

def create_patch_datasets(train_paths, train_labels, val_paths, val_labels, patch_size, stride, tmp_train, tmp_val):
    if os.path.exists(tmp_train): shutil.rmtree(tmp_train)
    if os.path.exists(tmp_val): shutil.rmtree(tmp_val)
    os.makedirs(tmp_train, exist_ok=True); os.makedirs(tmp_val, exist_ok=True)
    print('Generating patches for training...')
    for p,lab in zip(train_paths, train_labels):
        generate_patches_from_image(p, lab, patch_size, stride, tmp_train, PATCH_NET_INPUT)
    print('Generating patches for validation...')
    for p,lab in zip(val_paths, val_labels):
        generate_patches_from_image(p, lab, patch_size, stride, tmp_val, PATCH_NET_INPUT)
    ds_train = datasets.ImageFolder(tmp_train, transform=train_transform)
    ds_val   = datasets.ImageFolder(tmp_val,   transform=val_transform)
    return ds_train, ds_val

def create_patch_heatmap_for_image(image_path, net_patch, patch_size, stride, target_class_idx):
    I = Image.open(image_path).convert('RGB'); W,H = I.size
    cols = max(1, (W - patch_size[0]) // stride + 1)
    rows = max(1, (H - patch_size[1]) // stride + 1)
    heat = np.zeros((rows, cols), dtype=np.float32)
    net_patch.eval()
    with torch.no_grad():
        idxr=0
        for y in range(0, H - patch_size[1] + 1, stride):
            idxc=0
            for x in range(0, W - patch_size[0] + 1, stride):
                pr = I.crop((x, y, x+patch_size[0], y+patch_size[1])).resize(PATCH_NET_INPUT)
                pr_t = val_transform(pr).unsqueeze(0).to(DEVICE)
                probs = torch.softmax(net_patch(pr_t), dim=1).cpu().numpy()[0]
                heat[idxr, idxc] = float(probs[target_class_idx]); idxc+=1
            idxr+=1
    # gaussian smoothing if available
    try:
        from scipy.ndimage import gaussian_filter
        heat = gaussian_filter(heat, sigma=1.0)
    except Exception:
        pass
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_up = Image.fromarray((heat*255).astype(np.uint8)).resize((W,H), Image.BILINEAR)
    return np.array(heat_up)/255.0

def red_overlay(image_rgb, mask01, alpha=0.5):
    """Return RGB overlay where mask01==1 is tinted red."""
    img = np.asarray(image_rgb).astype(np.float32)
    mask = mask01.astype(np.float32)[...,None]
    red = np.zeros_like(img); red[...,0] = 255.0
    out = img*(1-alpha*mask) + red*(alpha*mask)
    return out.clip(0,255).astype(np.uint8)

# ---------------- Segmentation helpers ----------------
def seg_forward_out(net, imgs):
    out = net(imgs)
    if isinstance(out, dict) and 'out' in out: return out['out']
    return out

# ---------------- Main ----------------
def main():
    # Classification datasets
    train_ds, val_ds, class_names = build_classification_datasets(IMAGE_FOLDER, train_transform, val_transform)
    print("Classes:", class_names)
    num_workers = 4
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    results = {}

    # A: EfficientNet-V2-S
    if RUN_TINY:
        print('\n--- Model A: EfficientNet-V2-S ---')
        model = build_effnetv2_s(num_classes=len(class_names)).to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER, 'EfficientNetV2_S.pth')
        model = train_classifier(model, train_loader, val_loader, MAX_EPOCHS_CLASS, INITIAL_LR, save_path=savepath)
        res = evaluate_classifier(model, val_loader, class_names)
        results['EffNetV2_S'] = res
        plot_and_save_confusion(res['confusion'], class_names, os.path.join(FIG_FOLDER,'confusion_effnetv2s.png'), 'EffNetV2-S Confusion')
        plot_rocs(res['scores'], res['y_true'], class_names, os.path.join(FIG_FOLDER,'EffNetV2_S'))

    # B: ConvNeXt-Tiny
    if RUN_CONVNEXT_TINY:
        print('\n--- Model B: ConvNeXt-Tiny ---')
        model = build_convnext_tiny(num_classes=len(class_names)).to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER, 'ConvNeXt_Tiny.pth')
        model = train_classifier(model, train_loader, val_loader, MAX_EPOCHS_CLASS, INITIAL_LR/5, save_path=savepath)
        res = evaluate_classifier(model, val_loader, class_names)
        results['ConvNeXt_Tiny'] = res
        plot_and_save_confusion(res['confusion'], class_names, os.path.join(FIG_FOLDER,'confusion_convnext_tiny.png'), 'ConvNeXt-Tiny Confusion')
        plot_rocs(res['scores'], res['y_true'], class_names, os.path.join(FIG_FOLDER,'ConvNeXt_Tiny'))

    # C: MobileNetV3-Large
    if RUN_MOBILENETV3_L:
        print('\n--- Model C: MobileNetV3-Large ---')
        model = build_mobilenetv3_large(num_classes=len(class_names)).to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER, 'MobileNetV3_Large.pth')
        model = train_classifier(model, train_loader, val_loader, MAX_EPOCHS_CLASS, INITIAL_LR/5, save_path=savepath)
        res = evaluate_classifier(model, val_loader, class_names)
        results['MobileNetV3_Large'] = res
        plot_and_save_confusion(res['confusion'], class_names, os.path.join(FIG_FOLDER,'confusion_mnv3_large.png'), 'MobileNetV3-Large Confusion')
        plot_rocs(res['scores'], res['y_true'], class_names, os.path.join(FIG_FOLDER,'MobileNetV3_Large'))

    # D: Patch-based (MobileNetV3-Small)
    if RUN_PATCH_ENSEMBLE:
        print('\n--- Model D: Patch-based (MobileNetV3-Small) ---')
        tr_paths, tr_labels = train_ds.filepaths, train_ds.labels
        va_paths, va_labels = val_ds.filepaths,   val_ds.labels
        tmp_patch_train = os.path.join(RESULTS_FOLDER,'patch_tmp_train')
        tmp_patch_val   = os.path.join(RESULTS_FOLDER,'patch_tmp_val')
        patch_train_ds, patch_val_ds = create_patch_datasets(tr_paths, tr_labels, va_paths, va_labels, PATCH_SIZE, PATCH_STRIDE, tmp_patch_train, tmp_patch_val)
        patch_train_loader = DataLoader(patch_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
        patch_val_loader   = DataLoader(patch_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
        patch_net = build_mobilenetv3_small(num_classes=len(class_names)).to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER,'Patch_MobileNetV3_Small.pth')
        patch_net = train_classifier(patch_net, patch_train_loader, patch_val_loader, max(1, MAX_EPOCHS_CLASS//2), INITIAL_LR, save_path=savepath)
        # Heatmaps
        forged_idx = next((i for i,c in enumerate(class_names) if c.lower()=='forged'), 0)
        for i, p in enumerate(va_paths):
            heat = create_patch_heatmap_for_image(p, patch_net, PATCH_SIZE, PATCH_STRIDE, forged_idx)
            mask = (heat > PATCH_HEATMAP_THRESHOLD).astype(np.uint8)
            img = Image.open(p).convert('RGB')
            overlay = Image.fromarray(red_overlay(img, mask, alpha=0.5))
            overlay.save(os.path.join(PATCH_OVERLAY_FOLDER, f'patch_overlay_{i:04d}.png'))
        print('Patch overlays saved to', PATCH_OVERLAY_FOLDER)

    # E: Segmentation (DeepLabV3 MobileNetV3-Large -> ResNet50 -> UNet)
    if RUN_SEGMENTATION and os.path.isdir(LABEL_FOLDER):
        print('\n--- Model E: Segmentation ---')
        img_files_p, lbl_files_p = fuzzy_pair_images_labels(IMAGE_FOLDER, LABEL_FOLDER)
        if len(img_files_p)==0:
            print('No paired segmentation data found; skipping segmentation.')
        else:
            combo = list(zip(img_files_p, lbl_files_p))
            random.shuffle(combo)
            n = len(combo); ntr = int(0.8*n)
            tr = combo[:ntr]; va = combo[ntr:]
            tr_imgs = [a for a,b in tr]; tr_lbls = [b for a,b in tr]
            va_imgs = [a for a,b in va]; va_lbls = [b for a,b in va]
            seg_train_ds = SegmentationDataset(tr_imgs, tr_lbls, img_transform=seg_img_transform, class_map=None)
            seg_val_ds   = SegmentationDataset(va_imgs, va_lbls, img_transform=seg_img_transform, class_map=None)
            seg_train_loader = DataLoader(seg_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
            seg_val_loader   = DataLoader(seg_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
            num_seg_classes = 2
            seg_net = build_deeplabv3_mnv3_large(num_seg_classes)
            if seg_net is None:
                print('Falling back to small UNet.')
                seg_net = UNetSmall(num_seg_classes)
            seg_net = seg_net.to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(seg_net.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=L2_REG)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
            best_val = 1e9
            for epoch in range(MAX_EPOCHS_SEG):
                seg_net.train(); tr_loss=0.0; ntr=0
                for imgs, masks, _ in tqdm(seg_train_loader, desc=f'Seg train {epoch+1}/{MAX_EPOCHS_SEG}'):
                    imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
                    optimizer.zero_grad()
                    out = seg_forward_out(seg_net, imgs)
                    loss = criterion(out, masks)
                    loss.backward(); optimizer.step()
                    tr_loss += float(loss.item()) * imgs.size(0); ntr += imgs.size(0)
                scheduler.step()
                tr_loss /= max(1,ntr)
                # val
                seg_net.eval(); va_loss=0.0; nva=0
                with torch.no_grad():
                    for imgs, masks, _ in seg_val_loader:
                        imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
                        out = seg_forward_out(seg_net, imgs)
                        loss = criterion(out, masks)
                        va_loss += float(loss.item()) * imgs.size(0); nva += imgs.size(0)
                va_loss /= max(1,nva)
                print(f'Epoch {epoch+1}/{MAX_EPOCHS_SEG}: seg_train_loss={tr_loss:.4f}  seg_val_loss={va_loss:.4f}')
                if va_loss < best_val:
                    torch.save(seg_net.state_dict(), os.path.join(RESULTS_FOLDER,'segmentation_net.pth'))
                    best_val = va_loss
            # overlays on val set
            seg_net.eval()
            for i,(ip,lp) in enumerate(zip(va_imgs, va_lbls)):
                img = Image.open(ip).convert('RGB')
                inp = seg_img_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = seg_forward_out(seg_net, inp)
                pred = out.argmax(1).cpu().numpy()[0].astype(np.uint8)
                pred_img = Image.fromarray(pred*255).resize(img.size, Image.NEAREST)
                mask = (np.array(pred_img)>127).astype(np.uint8)
                overlay = Image.fromarray(red_overlay(img, mask, alpha=0.5))
                overlay.save(os.path.join(SEG_OVERLAY_FOLDER, f'seg_overlay_{i:04d}.png'))
            print('Seg overlays saved to', SEG_OVERLAY_FOLDER)
            # simple pixel metrics
            all_ytrue=[]; all_ypred=[]
            with torch.no_grad():
                for imgs, masks, _ in seg_val_loader:
                    imgs = imgs.to(DEVICE)
                    out = seg_forward_out(seg_net, imgs).argmax(1).cpu().numpy().reshape(-1)
                    all_ypred.extend(out.tolist())
                    all_ytrue.extend(masks.numpy().reshape(-1).tolist())
            p,r,f,_ = precision_recall_fscore_support(all_ytrue, all_ypred, average=None, zero_division=0)
            with open(os.path.join(RESULTS_FOLDER,'segmentation_metrics.json'),'w') as f:
                json.dump({'precision':p.tolist(),'recall':r.tolist(),'f1':f.tolist()}, f, indent=2)

    # Plot comparison for classification
    print('\nPlotting summary metrics...')
    model_keys = [k for k in results.keys() if k in ['EffNetV2_S','ConvNeXt_Tiny','MobileNetV3_Large']]
    if model_keys:
        for metric in ['precision','recall','f1']:
            plt.figure(figsize=(10,4))
            for m in model_keys:
                arr = results[m][metric]
                plt.plot(range(len(arr)), arr, '-o', label=m)
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.xlabel('Class'); plt.ylabel(metric.capitalize()); plt.title(f'{metric.capitalize()} per class')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(FIG_FOLDER, f'{metric}_per_class.png')); plt.close()
        P = [np.mean(results[m]['precision']) for m in model_keys]
        R = [np.mean(results[m]['recall']) for m in model_keys]
        F = [np.mean(results[m]['f1']) for m in model_keys]
        x = np.arange(len(model_keys)); width=0.25
        fig,ax = plt.subplots()
        ax.bar(x-width, P, width, label='Precision')
        ax.bar(x,        R, width, label='Recall')
        ax.bar(x+width,  F, width, label='F1')
        ax.set_xticks(x); ax.set_xticklabels(model_keys)
        ax.legend(); ax.set_ylabel('Macro-average'); ax.set_title('Classification Macro metrics')
        plt.tight_layout(); plt.savefig(os.path.join(FIG_FOLDER,'classification_macro_bar.png')); plt.close()

    # Save concise results
    with open(os.path.join(RESULTS_FOLDER,'results_summary.json'),'w') as f:
        json.dump({k:{kk:(v.tolist() if isinstance(v,np.ndarray) else v)
                      for kk,v in results[k].items() if kk in ['precision','recall','f1']}
                   for k in results}, f, indent=2)
    print('\nAll done. Results saved in:', RESULTS_FOLDER)

if __name__ == '__main__':
    main()
