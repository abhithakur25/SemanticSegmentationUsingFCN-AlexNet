#!/usr/bin/env python3
"""
forgery_models_5version_compare_fixed_complete_gpu_only.py

Complete script (GPU-only): trains/evaluates multiple models and includes robust dataset handling.
This version *requires* a CUDA-enabled NVIDIA GPU and will raise a RuntimeError if none is available.

- TinyCNN
- ResNet-50 (finetune)
- MobileNetV2 (finetune)
- Patch-based classifier + heatmaps
- Segmentation (DeepLabV3 or UNet fallback)

Author: adapted & corrected by ChatGPT
"""
import os
import sys
import time
import random
import shutil
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models, datasets
import torchvision.transforms.functional as TF

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ---------------- USER CONFIG ----------------
DATA_ROOT = r'D:/claude/SemanticSegmentationUsingFCN-AlexNet1/Dataset4'
IMAGE_FOLDER = os.path.join(DATA_ROOT, 'ImagesReszed')
LABEL_FOLDER = os.path.join(DATA_ROOT, 'LabelsReszed')
RESULTS_FOLDER = os.path.join(DATA_ROOT, 'forgery_results_5models_python')
FIG_FOLDER = os.path.join(RESULTS_FOLDER, 'figures')
PATCH_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'patch_overlays')
SEG_OVERLAY_FOLDER = os.path.join(RESULTS_FOLDER, 'segmentation_overlays')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)
os.makedirs(PATCH_OVERLAY_FOLDER, exist_ok=True)
os.makedirs(SEG_OVERLAY_FOLDER, exist_ok=True)

# Pipelines to run
RUN_TINYCNN = True
RUN_RESNET50 = True
RUN_MOBILENETV2 = True
RUN_PATCH_ENSEMBLE = True
RUN_SEGMENTATION = True  # will skip if no label folder exists or mismatch

# --- GPU-only enforcement ---
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. This script must be run on a machine with a CUDA-enabled NVIDIA GPU.\n"
        "If you intended to run on CPU, either edit the script to allow CPU or set RUN_* flags to skip heavy models.\n"
        "Make sure you have installed a CUDA-enabled PyTorch build and that your NVIDIA drivers are up-to-date."
    )
DEVICE = torch.device('cuda')
print('Device:', DEVICE, '| CUDA version:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())
try:
    print('Current GPU name:', torch.cuda.get_device_name(0))
except Exception:
    pass

# Hyperparameters (you can tune)
SEED = 0
RANDOM_STATE = SEED
BATCH_SIZE = 16
MAX_EPOCHS_CLASS = 8
MAX_EPOCHS_SEG = 12
INITIAL_LR = 1e-4
L2_REG = 1e-4
INPUT_SIZE = (224, 224)        # classifier input (H,W)
SEGMENTATION_SIZE = (360, 480) # segmentation H,W
PATCH_SIZE = (64, 64)
PATCH_STRIDE = 32
PATCH_NET_INPUT = (224, 224)   # patch net expects classifier input size
PATCH_HEATMAP_THRESHOLD = 0.35

# Data augmentation transforms for training classification
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    transforms.RandomAffine(degrees=0, translate=(6/INPUT_SIZE[1], 6/INPUT_SIZE[0])),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),  # outputs float in [0,1]
])

val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
])

# For segmentation images (resize but keep mask integer)
seg_img_transform = transforms.Compose([
    transforms.Resize(SEGMENTATION_SIZE),
    transforms.ToTensor()
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
    """
    Match images and label masks by basename fuzzy logic.
    Returns lists of matched (image_file, label_file).
    """
    img_files = list_image_files(img_folder)
    lbl_files = list_image_files(label_folder)
    img_basenames = [Path(p).stem for p in img_files]
    lbl_basenames = [Path(p).stem for p in lbl_files]
    lbl_map = {bn.lower(): p for bn, p in zip(lbl_basenames, lbl_files)}
    matches_img = []
    matches_lbl = []
    # exact basename match
    for i, bn in enumerate(img_basenames):
        key = bn.lower()
        if key in lbl_map:
            matches_img.append(img_files[i])
            matches_lbl.append(lbl_map[key])
    if matches_img:
        return matches_img, matches_lbl
    # fuzzy contains
    for i, bn in enumerate(img_basenames):
        low = bn.lower()
        for j, lb in enumerate(lbl_basenames):
            if low in lb.lower() or lb.lower() in low:
                matches_img.append(img_files[i])
                matches_lbl.append(lbl_files[j])
                break
    if matches_img:
        return matches_img, matches_lbl
    # numeric suffix heuristic (fixed regex)
    import re
    for i, bn in enumerate(img_basenames):
        m = re.search(r'(\d+)$', bn)
        if m:
            prefix = re.sub(r'\d+$', '', bn)
            for j, lb in enumerate(lbl_basenames):
                if Path(lbl_files[j]).stem.startswith(prefix):
                    matches_img.append(img_files[i])
                    matches_lbl.append(lbl_files[j])
                    break
    # fallback pairing by index if counts equal
    if not matches_img and len(img_files) == len(lbl_files) and len(img_files) > 0:
        print('Fuzzy pairing fallback: pairing by order (verify!).')
        return img_files, lbl_files
    return matches_img, matches_lbl

# ---------------- Datasets ----------------

class ClassificationImageFolder(Dataset):
    """Simple dataset wrapper around a list of image paths + labels (derived from folder names)"""
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        p = self.filepaths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label, p

def build_classification_datasets(image_root, input_transform, val_transform, subset_fraction=1.0):
    """
    Robust loader:
    - If `image_root` contains subfolders, use torchvision.datasets.ImageFolder (standard layout).
    - Otherwise, scan files in `image_root` and infer class labels from filenames using heuristics.
    Returns (train_ds, val_ds, class_names).
    """
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f'image folder not found: {image_root}')

    # First try standard ImageFolder (subfolders per class)
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
        # No subfolders: attempt to infer labels from filenames
        print(f'No class subfolders found in {image_root} — attempting to infer labels from filenames...')
        rootp = Path(image_root)
        exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
        files = sorted([p for p in rootp.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if len(files) == 0:
            raise FileNotFoundError(f'No image files found in {image_root} (checked extensions {exts}).')

        # common keywords mapping -> canonical label name
        keyword_map = {
            'forged': 'forged', 'fake': 'forged', 'tamper': 'forged', 'tampered': 'forged',
            'original': 'original', 'auth': 'original', 'real': 'original', 'genuine': 'original',
            'pristine': 'original'
        }

        inferred = []
        for p in files:
            nm = p.stem.lower()
            lab = None
            # 1) keyword match
            for kw, canon in keyword_map.items():
                if kw in nm:
                    lab = canon
                    break
            if lab is None:
                # 2) filename token heuristic: take first token if it looks like a label (not purely numeric)
                sep_tokens = []
                for sep in ['_', '-', '.']:
                    sep_tokens.extend([t for t in nm.split(sep) if t])
                if sep_tokens:
                    first = sep_tokens[0]
                    if first and not first.isdigit():
                        lab = first
                # 3) fallback: try last token (e.g., img_01_forged)
                if lab is None and sep_tokens:
                    last = sep_tokens[-1]
                    if last and not last.isdigit():
                        lab = last
            if lab is None:
                lab = 'unknown'
            inferred.append((str(p), lab))

        # Build label index mapping
        labels_list = [l for (_, l) in inferred]
        classes = sorted(list(set(labels_list)))
        if 'unknown' in classes and len(classes) == 1:
            # all unknown: we can't map—give an informative error with samples
            sample_files = [str(f.name) for f in files[:10]]
            raise FileNotFoundError(
                f"Couldn't infer class labels from filenames in {image_root}.\n"
                f"Sample filenames: {sample_files}\n\n"
                f"Please either:\n"
                f"  - Organize images into subfolders per class (recommended), e.g. {image_root}/forged/*.jpg\n"
                f"  - Or ensure filenames contain class tokens like 'forged' or 'original'."
            )

        class_to_idx = {c:i for i,c in enumerate(classes)}
        filepaths = [p for (p,_) in inferred]
        labels = [class_to_idx[l] for (_,l) in inferred]
        class_names = classes
        print(f'Inferred classes from filenames: {class_names}')

    # optionally filter out unreadable files
    good_paths = []
    good_labels = []
    for p, l in zip(filepaths, labels):
        try:
            with Image.open(p) as _:
                good_paths.append(p)
                good_labels.append(l)
        except Exception:
            print(f'Warning: could not read image {p}; skipping.')

    filepaths = good_paths
    labels = good_labels

    if len(filepaths) == 0:
        raise FileNotFoundError("No readable image files found after filtering.")

    # split per-class
    from collections import defaultdict
    idx_per_label = defaultdict(list)
    for idx, lab in enumerate(labels):
        idx_per_label[lab].append(idx)

    train_idxs = []
    val_idxs = []
    for lab, idxs in idx_per_label.items():
        random.shuffle(idxs)
        ntrain = int(0.8 * len(idxs))
        if ntrain < 1 and len(idxs) >= 2:
            ntrain = 1
        train_idxs.extend(idxs[:ntrain])
        val_idxs.extend(idxs[ntrain:])

    random.shuffle(train_idxs)
    random.shuffle(val_idxs)

    train_paths = [filepaths[i] for i in train_idxs]
    train_labels = [labels[i] for i in train_idxs]
    val_paths = [filepaths[i] for i in val_idxs]
    val_labels = [labels[i] for i in val_idxs]

    train_ds = ClassificationImageFolder(train_paths, train_labels, transform=input_transform)
    val_ds = ClassificationImageFolder(val_paths, val_labels, transform=val_transform)

    return train_ds, val_ds, class_names

class SegmentationDataset(Dataset):
    """
    Expects matching lists: image_paths and mask_paths.
    Mask images must be single-channel where pixel values are label indices (1-based or 0-based)
    We'll map them to 0..(C-1).
    """
    def __init__(self, image_paths, mask_paths, img_transform=None, mask_transform=None, class_map=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.class_map = class_map

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        im = Image.open(self.image_paths[idx]).convert('RGB')
        m = Image.open(self.mask_paths[idx])
        # convert mask to single-channel grayscale
        if m.mode != 'L':
            m = m.convert('L')
        if self.img_transform:
            im_t = self.img_transform(im)
        else:
            im_t = TF.to_tensor(im)
        # mask transform: keep ints
        m_np = np.array(m, dtype=np.int64)
        # Map labels: if class_map provided, apply mapping; otherwise assume values 1/2 -> 0/1
        if self.class_map is not None:
            # class_map is dict mapping from pixel value -> class_idx
            mapped = np.zeros_like(m_np, dtype=np.int64)
            for k, v in self.class_map.items():
                mapped[m_np == int(k)] = int(v)
            m_np = mapped
        else:
            # try to map binary: 0 -> class0, 255/1 -> class1 etc.
            unique = np.unique(m_np)
            # if mask is 0/1 valid -> ok
            if set(unique.tolist()) <= {0,1}:
                pass
            else:
                # if 1/2 as in matlab? map 1->0,2->1
                if set(unique.tolist()) <= {1,2}:
                    m_np = m_np - 1
                else:
                    # if 0..255: threshold
                    m_np = (m_np > 127).astype(np.int64)
        # convert to tensor (long)
        m_t = torch.from_numpy(m_np).long()
        if self.mask_transform:
            m_t = self.mask_transform(m_t)
        return im_t, m_t, self.image_paths[idx]

# ---------------- Model definitions ----------------

class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class UNetSmall(nn.Module):
    # Minimal U-Net for segmentation. Input channels=3, output channels=num_classes
    def __init__(self, num_classes, in_ch=3, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base,3,padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base, base*2,3,padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base*2, base*4,3,padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base*4, base*2,3,padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base*2, base,3,padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base, num_classes, 1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        u1 = self.up1(e3)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        out = self.out(d2)
        return out

# ---------------- Training & eval helpers ----------------

def train_classifier(model, train_loader, val_loader, num_epochs, lr, save_path=None):
    """
    Training loop that accepts dataloaders which yield either:
      - (imgs, labels, paths)  OR
      - (imgs, labels)
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=L2_REG)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    best_val_acc = 0.0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
    warned_two_tuple = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{num_epochs}"):
            # support both (imgs, labels, paths) and (imgs, labels)
            if len(batch) == 3:
                imgs, labels, _ = batch
            elif len(batch) == 2:
                imgs, labels = batch
                if not warned_two_tuple:
                    print("Note: train_loader yields (imgs, labels) tuples — proceeding without paths.")
                    warned_two_tuple = True
            else:
                raise ValueError(f"Unexpected batch tuple length from train_loader: {len(batch)}")
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * imgs.size(0)
            total_train += imgs.size(0)
        scheduler.step()
        train_loss = running_loss / total_train if total_train > 0 else 0.0

        # validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    imgs, labels, _ = batch
                elif len(batch) == 2:
                    imgs, labels = batch
                else:
                    raise ValueError(f"Unexpected batch tuple length from val_loader: {len(batch)}")
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += float(loss.item()) * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc and save_path:
            torch.save(model.state_dict(), save_path)
            best_val_acc = val_acc
    return model, history

# ---------------- Patch utilities ----------------

def generate_patches_from_image(img_path, label, patch_size, stride, out_folder, net_input_size):
    """Extract patches and save under out_folder/<label>/imagepatch_XXXX.png"""
    os.makedirs(out_folder, exist_ok=True)
    lbl_folder = os.path.join(out_folder, str(label))
    os.makedirs(lbl_folder, exist_ok=True)
    I = Image.open(img_path).convert('RGB')
    W, H = I.size
    cnt = 0
    for y in range(0, H - patch_size[1] + 1, stride):
        for x in range(0, W - patch_size[0] + 1, stride):
            pr = I.crop((x, y, x + patch_size[0], y + patch_size[1]))
            pr = pr.resize(net_input_size)
            fname = os.path.join(lbl_folder, f'patch_{cnt:05d}.png')
            pr.save(fname)
            cnt += 1
    return cnt

def create_patch_datasets(imds_train_paths, imds_train_labels, imds_val_paths, imds_val_labels, patch_size, stride, tmp_train, tmp_val):
    if os.path.exists(tmp_train): shutil.rmtree(tmp_train)
    if os.path.exists(tmp_val): shutil.rmtree(tmp_val)
    os.makedirs(tmp_train, exist_ok=True)
    os.makedirs(tmp_val, exist_ok=True)
    print('Generating patches for training...')
    for p, lab in zip(imds_train_paths, imds_train_labels):
        generate_patches_from_image(p, lab, patch_size, stride, tmp_train, PATCH_NET_INPUT)
    print('Generating patches for validation...')
    for p, lab in zip(imds_val_paths, imds_val_labels):
        generate_patches_from_image(p, lab, patch_size, stride, tmp_val, PATCH_NET_INPUT)
    # Now use ImageFolder
    ds_train = datasets.ImageFolder(tmp_train, transform=train_transform)
    ds_val = datasets.ImageFolder(tmp_val, transform=val_transform)
    return ds_train, ds_val

def create_patch_heatmap_for_image(image_path, net_patch, patch_size, stride, target_class_idx):
    I = Image.open(image_path).convert('RGB')
    W, H = I.size
    cols = max(1, (W - patch_size[0]) // stride + 1)
    rows = max(1, (H - patch_size[1]) // stride + 1)
    heat = np.zeros((rows, cols), dtype=np.float32)
    net_patch.eval()
    idxr = 0
    with torch.no_grad():
        for y in range(0, H - patch_size[1] + 1, stride):
            idxc = 0
            for x in range(0, W - patch_size[0] + 1, stride):
                pr = I.crop((x, y, x + patch_size[0], y + patch_size[1]))
                pr = pr.resize(PATCH_NET_INPUT)
                pr_t = transforms.ToTensor()(pr).unsqueeze(0).to(DEVICE)
                scores = net_patch(pr_t)
                probs = nn.functional.softmax(scores, dim=1).cpu().numpy()[0]
                p = probs[target_class_idx]
                heat[idxr, idxc] = p
                idxc += 1
            idxr += 1
    # gaussian smoothing (requires scipy)
    from scipy.ndimage import gaussian_filter
    heat = gaussian_filter(heat, sigma=1.0)
    # normalize to [0,1]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    # scale back up to image size
    heat_up = Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
    return np.array(heat_up) / 255.0

def _seg_forward_get_out(net, imgs):
    """
    Call segmentation network and always return tensor with shape (N, C, H, W).
    Supports models returning dicts (like torchvision deeplab).
    """
    out = net(imgs)
    if isinstance(out, dict) and 'out' in out:
        return out['out']
    return out
def evaluate_classifier(model, data_loader, class_names):
    """
    Evaluation that supports dataloaders yielding either (img, label, path) or (img, label).
    Returns dictionary of metrics + predicted scores.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    paths = []
    warned_two_tuple = False

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                imgs, labels, pths = batch
            elif len(batch) == 2:
                imgs, labels = batch
                pths = [None] * imgs.size(0)
                if not warned_two_tuple:
                    print("Note: data_loader yields (imgs, labels) with no paths; 'paths' in results will be None for those samples.")
                    warned_two_tuple = True
            else:
                raise ValueError(f"Unexpected batch tuple length from data_loader: {len(batch)}")

            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_scores.extend(probs.tolist())
            # extend paths list; if pths is tensor/list of strings in ImageFolder it's fine,
            # if pths contains Nones we still extend with Nones
            if isinstance(pths, (list, tuple)):
                paths.extend(pths)
            else:
                # in case pths is a tensor or other type, convert to python list of strings where possible
                try:
                    paths.extend([str(x) for x in pths])
                except Exception:
                    paths.extend([None] * len(preds))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    return {
        'confusion': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': np.array(y_true),
        'y_pred': np.array(y_pred),
        'scores': np.array(y_scores),
        'paths': paths
    }

# ---------------- Main script ----------------

def main():
    # Prepare classification datasets
    train_ds, val_ds, class_names = build_classification_datasets(IMAGE_FOLDER, train_transform, val_transform)
    print("Classes:", class_names)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    # Model A: TinyCNN
    if RUN_TINYCNN:
        print('\n--- TinyCNN ---')
        num_classes = len(class_names)
        model_tiny = TinyCNN(num_classes).to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER, 'TinyCNN.pth')
        model_tiny, hist = train_classifier(model_tiny, train_loader, val_loader, MAX_EPOCHS_CLASS, INITIAL_LR, save_path=savepath)
        # Evaluate
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        res_tiny = evaluate_classifier(model_tiny, val_dl, class_names)
        results['TinyCNN'] = res_tiny
        plot_and_save_confusion(res_tiny['confusion'], class_names, os.path.join(FIG_FOLDER,'confusion_tinycnn.png'), title='TinyCNN Confusion')
        plot_rocs(res_tiny['scores'], res_tiny['y_true'], class_names, os.path.join(FIG_FOLDER,'TinyCNN'))
        print('TinyCNN done.')

    # Model B: ResNet50 finetune
    if RUN_RESNET50:
        print('\n--- ResNet50 finetune ---')
        num_classes = len(class_names)
        try:
            net = models.resnet50(pretrained=True)
            # replace final FC
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        except Exception as e:
            print('ResNet50 not available, using ResNet18 fallback:', e)
            net = models.resnet18(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        net = net.to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER,'ResNet50_finetune.pth')
        train_loader_t = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_t = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        net, hist = train_classifier(net, train_loader_t, val_loader_t, MAX_EPOCHS_CLASS, INITIAL_LR/5, save_path=savepath)
        res_r50 = evaluate_classifier(net, val_loader_t, class_names)
        results['ResNet50'] = res_r50
        plot_and_save_confusion(res_r50['confusion'], class_names, os.path.join(FIG_FOLDER,'confusion_resnet50.png'), title='ResNet50 Confusion')
        plot_rocs(res_r50['scores'], res_r50['y_true'], class_names, os.path.join(FIG_FOLDER,'ResNet50'))
        print('ResNet50 done.')

    # Model C: MobileNetV2 finetune
    if RUN_MOBILENETV2:
        print('\n--- MobileNetV2 finetune ---')
        num_classes = len(class_names)
        try:
            netm = models.mobilenet_v2(pretrained=True)
            netm.classifier[1] = nn.Linear(netm.classifier[1].in_features, num_classes)
        except Exception as e:
            print('MobileNetv2 not available, using ResNet18 fallback:', e)
            netm = models.resnet18(pretrained=True)
            netm.fc = nn.Linear(netm.fc.in_features, num_classes)
        netm = netm.to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER,'MobileNetv2_finetune.pth')
        train_loader_m = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_m = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        netm, hist = train_classifier(netm, train_loader_m, val_loader_m, MAX_EPOCHS_CLASS, INITIAL_LR/5, save_path=savepath)
        res_mob = evaluate_classifier(netm, val_loader_m, class_names)
        results['MobileNetv2'] = res_mob
        plot_and_save_confusion(res_mob['confusion'], class_names, os.path.join(FIG_FOLDER,'confusion_mobilenetv2.png'), title='MobileNetv2 Confusion')
        plot_rocs(res_mob['scores'], res_mob['y_true'], class_names, os.path.join(FIG_FOLDER,'MobileNetv2'))
        print('MobileNetv2 done.')

    # Model D: Patch ensemble
    if RUN_PATCH_ENSEMBLE:
        print('\n--- Patch-based ensemble ---')
        train_paths = train_ds.filepaths
        train_labels = train_ds.labels
        val_paths = val_ds.filepaths
        val_labels = val_ds.labels
        tmp_patch_train = os.path.join(RESULTS_FOLDER,'patch_tmp_train')
        tmp_patch_val = os.path.join(RESULTS_FOLDER,'patch_tmp_val')
        patch_train_ds, patch_val_ds = create_patch_datasets(train_paths, train_labels, val_paths, val_labels, PATCH_SIZE, PATCH_STRIDE, tmp_patch_train, tmp_patch_val)
        patch_train_loader = DataLoader(patch_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        patch_val_loader = DataLoader(patch_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        num_classes = len(class_names)
        net_patch = TinyCNN(num_classes).to(DEVICE)
        savepath = os.path.join(RESULTS_FOLDER,'PatchTinyCNN.pth')
        net_patch, hist = train_classifier(net_patch, patch_train_loader, patch_val_loader, max(1, MAX_EPOCHS_CLASS//2), INITIAL_LR, save_path=savepath)
        # generate heatmaps for val images and save overlays
        forged_class_idx = None
        for idx, cname in enumerate(class_names):
            if cname.lower() == 'forged':
                forged_class_idx = idx
        if forged_class_idx is None:
            forged_class_idx = 0
        for i, p in enumerate(val_paths):
            heat = create_patch_heatmap_for_image(p, net_patch, PATCH_SIZE, PATCH_STRIDE, forged_class_idx)
            maskBW = heat > PATCH_HEATMAP_THRESHOLD
            img = Image.open(p).convert('RGB')
            overlay = Image.fromarray((np.array(img) * (1 - 0.5) + np.dstack([maskBW*255, np.zeros_like(maskBW)*0, np.zeros_like(maskBW)*0])*0.5).astype(np.uint8))
            outname = os.path.join(PATCH_OVERLAY_FOLDER, f'patch_overlay_{i:04d}.png')
            overlay.save(outname)
        results['PatchEnsemble'] = {'model': 'PatchTinyCNN'}
        print('Patch ensemble done. Overlays saved to', PATCH_OVERLAY_FOLDER)

    # Model E: Segmentation
    if RUN_SEGMENTATION and os.path.isdir(LABEL_FOLDER):
        print('\n--- Segmentation pipeline ---')
        img_files_p, lbl_files_p = fuzzy_pair_images_labels(IMAGE_FOLDER, LABEL_FOLDER)
        if len(img_files_p) == 0:
            print('No paired segmentation data found; skipping segmentation.')
        else:
            combined = list(zip(img_files_p, lbl_files_p))
            random.shuffle(combined)
            n = len(combined)
            ntrain = int(0.8*n)
            train_pairs = combined[:ntrain]
            val_pairs = combined[ntrain:]
            train_imgs = [a for a,b in train_pairs]; train_lbls = [b for a,b in train_pairs]
            val_imgs = [a for a,b in val_pairs]; val_lbls = [b for a,b in val_pairs]
            seg_train_ds = SegmentationDataset(train_imgs, train_lbls, img_transform=seg_img_transform, class_map=None)
            seg_val_ds = SegmentationDataset(val_imgs, val_lbls, img_transform=seg_img_transform, class_map=None)
            seg_train_loader = DataLoader(seg_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
            seg_val_loader = DataLoader(seg_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            num_seg_classes = 2
            try:
                seg_net = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_seg_classes)
                print('Using deeplabv3_resnet50 for segmentation.')
            except Exception as e:
                print('deeplab not available; using small UNet fallback.', e)
                seg_net = UNetSmall(num_seg_classes)
            seg_net = seg_net.to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(seg_net.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=L2_REG)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
            best_loss = 1e9
            for epoch in range(MAX_EPOCHS_SEG):
                seg_net.train()
                train_loss = 0.0
                for imgs, masks, _ in tqdm(seg_train_loader, desc=f'Seg train epoch {epoch+1}/{MAX_EPOCHS_SEG}'):
                    imgs = imgs.to(DEVICE)
                    masks = masks.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = _seg_forward_get_out(seg_net, imgs)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item()) * imgs.size(0)
                scheduler.step()
                train_loss = train_loss / len(seg_train_loader.dataset)
                seg_net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for imgs, masks, _ in seg_val_loader:
                        imgs = imgs.to(DEVICE)
                        masks = masks.to(DEVICE)
                        outputs = _seg_forward_get_out(seg_net, imgs)
                        loss = criterion(outputs, masks)
                        val_loss += float(loss.item()) * imgs.size(0)
                val_loss = val_loss / len(seg_val_loader.dataset)
                print(f'Epoch {epoch+1}/{MAX_EPOCHS_SEG}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
                if val_loss < best_loss:
                    torch.save(seg_net.state_dict(), os.path.join(RESULTS_FOLDER, 'segmentation_net.pth'))
                    best_loss = val_loss
            # produce predicted overlays on validation set
            seg_net.eval()
            for i, (img_path, lbl_path) in enumerate(zip(val_imgs, val_lbls)):
                img = Image.open(img_path).convert('RGB')
                img_resized = seg_img_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = _seg_forward_get_out(seg_net, img_resized)
                probs = nn.functional.softmax(out, dim=1).cpu().numpy()[0]
                pred = probs.argmax(axis=0).astype(np.uint8)
                pred_img = Image.fromarray(pred*255)
                pred_up = pred_img.resize(img.size, resample=Image.NEAREST)
                forged_mask = np.array(pred_up) > 127
                overlay = Image.fromarray((np.array(img) * (1 - 0.5) + np.dstack([forged_mask*255, np.zeros_like(forged_mask)*0, np.zeros_like(forged_mask)*0])*0.5).astype(np.uint8))
                outname = os.path.join(SEG_OVERLAY_FOLDER, f'seg_overlay_{i:04d}.png')
                overlay.save(outname)
            print('Segmentation overlays saved to', SEG_OVERLAY_FOLDER)
            # Evaluate segmentation metrics roughly (pixel-level)
            all_ytrue = []
            all_ypred = []
            for imgs, masks, pths in seg_val_loader:
                imgs = imgs.to(DEVICE)
                masks_flat = masks.numpy().reshape(-1)
                with torch.no_grad():
                    out = _seg_forward_get_out(seg_net, imgs)
                preds = out.argmax(dim=1).cpu().numpy().reshape(-1)
                all_ytrue.extend(masks_flat.tolist())
                all_ypred.extend(preds.tolist())
            p, r, f, _ = precision_recall_fscore_support(all_ytrue, all_ypred, average=None, zero_division=0)
            results['Segmentation'] = {'precision': p.tolist(), 'recall': r.tolist(), 'f1': f.tolist()}
            with open(os.path.join(RESULTS_FOLDER,'segmentation_metrics.json'), 'w') as f:
                json.dump(results['Segmentation'], f, indent=2)
    else:
        print('Segmentation disabled or label folder missing; skipping.')

    # Plot summary metrics (classification)
    print('Plotting summary metrics...')
    model_keys = [k for k in results.keys() if k in ['TinyCNN','ResNet50','MobileNetv2']]
    if model_keys:
        for metric in ['precision','recall','f1']:
            plt.figure(figsize=(10,4))
            for m in model_keys:
                arr = results[m][metric]
                plt.plot(range(len(arr)), arr, '-o', label=m)
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.xlabel('Class'); plt.ylabel(metric.capitalize()); plt.title(f'{metric.capitalize()} per class')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_FOLDER, f'{metric}_per_class.png'))
            plt.close()
        P = [np.mean(results[m]['precision']) for m in model_keys]
        R = [np.mean(results[m]['recall']) for m in model_keys]
        F = [np.mean(results[m]['f1']) for m in model_keys]
        x = np.arange(len(model_keys))
        width = 0.25
        fig, ax = plt.subplots()
        ax.bar(x - width, P, width, label='Precision')
        ax.bar(x, R, width, label='Recall')
        ax.bar(x + width, F, width, label='F1')
        ax.set_xticks(x); ax.set_xticklabels(model_keys)
        ax.legend(); ax.set_ylabel('Macro-average'); ax.set_title('Classification Macro metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_FOLDER,'classification_macro_bar.png'))
        plt.close()
    # Save full results
    with open(os.path.join(RESULTS_FOLDER, 'results_summary.json'), 'w') as f:
        json.dump({k: {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk,v in results[k].items() if kk in ['precision','recall','f1']} for k in results}, f, indent=2)
    print('All done. Results saved in', RESULTS_FOLDER)

if __name__ == '__main__':
    main()
