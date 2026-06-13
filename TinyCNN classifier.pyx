#!/usr/bin/env python3
"""
TinyCNN classifier (GPU-only)
- Trains a small CNN on a 2+ class image dataset
- Saves best weights and basic figures/metrics

Author: adapted by ChatGPT
"""
import os, sys, random, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ---------------- USER CONFIG ----------------
DATA_ROOT = r'D:/claude/SemanticSegmentationUsingFCN-AlexNet1/Dataset4'
IMAGE_FOLDER = os.path.join(DATA_ROOT, 'ImagesReszed')
RESULTS_FOLDER = os.path.join(DATA_ROOT, 'tinycnn_results')
FIG_FOLDER = os.path.join(RESULTS_FOLDER, 'figures')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)

SEED = 0
BATCH_SIZE = 16
MAX_EPOCHS = 8
INITIAL_LR = 1e-4
L2_REG = 1e-4
INPUT_SIZE = (224, 224)

# ------------- GPU ONLY ENFORCEMENT -------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available; TinyCNN script requires CUDA.")
DEVICE = torch.device("cuda")
print('Device:', DEVICE, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))

# ---------------- Transforms ----------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    transforms.RandomAffine(degrees=0, translate=(6/INPUT_SIZE[1], 6/INPUT_SIZE[0])),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
])

# ---------------- Data utils ----------------
class FlatDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, i):
        p = self.filepaths[i]
        img = Image.open(p).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

def list_image_files(folder, exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
    files = []
    for ext in exts:
        files.extend(sorted(Path(folder).rglob(f'*{ext}')))
    return [str(p) for p in files]

def build_classification_datasets(image_root, tr_tf, val_tf):
    # Try ImageFolder first
    try:
        imf = datasets.ImageFolder(image_root)
        if imf.classes:
            paths = [p for p,_ in imf.samples]
            labels = [y for _,y in imf.samples]
            class_names = imf.classes
        else:
            imf = None
    except Exception:
        imf = None

    if imf is None:
        print(f'No class subfolders found in {image_root}; inferring labels from filenames...')
        files = list_image_files(image_root)
        if not files:
            raise FileNotFoundError('No images found.')
        keyword_map = {
            'forged':'forged','fake':'forged','tamper':'forged','tampered':'forged',
            'original':'original','auth':'original','real':'original','genuine':'original','pristine':'original'
        }
        inferred = []
        for p in files:
            name = Path(p).stem.lower()
            lab = None
            for kw, canon in keyword_map.items():
                if kw in name:
                    lab = canon; break
            if lab is None:
                toks = []
                for sep in ['_','-','.']: toks += [t for t in name.split(sep) if t]
                if toks and not toks[0].isdigit(): lab = toks[0]
                if lab is None and toks and not toks[-1].isdigit(): lab = toks[-1]
            if lab is None: lab = 'unknown'
            inferred.append((p, lab))
        classes = sorted(set([lab for _,lab in inferred]))
        if classes == ['unknown']:
            raise RuntimeError('Could not infer classes from filenames. Use subfolders or put tokens in filenames.')
        class_to_idx = {c:i for i,c in enumerate(classes)}
        paths = [p for p,_ in inferred]
        labels = [class_to_idx[l] for _,l in inferred]
        class_names = classes

    # filter unreadable
    good_p, good_y = [], []
    for p,y in zip(paths, labels):
        try:
            Image.open(p).close()
            good_p.append(p); good_y.append(y)
        except Exception:
            print('Skipping unreadable:', p)
    paths, labels = good_p, good_y

    # split per-class 80/20
    from collections import defaultdict
    idxs = defaultdict(list)
    for i,y in enumerate(labels): idxs[y].append(i)
    train_idx, val_idx = [], []
    rng = random.Random(SEED)
    for y, arr in idxs.items():
        rng.shuffle(arr)
        ntr = max(1, int(0.8*len(arr))) if len(arr)>1 else 1
        train_idx += arr[:ntr]
        val_idx += arr[ntr:] if len(arr)>1 else arr[:0]
    tr_paths = [paths[i] for i in train_idx]; tr_y = [labels[i] for i in train_idx]
    va_paths = [paths[i] for i in val_idx];   va_y = [labels[i] for i in val_idx]

    train_ds = FlatDataset(tr_paths, tr_y, tr_tf)
    val_ds   = FlatDataset(va_paths, va_y, val_tf)
    return train_ds, val_ds, class_names

# ---------------- Model ----------------
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
            nn.Linear(128,256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

# ---------------- Train/Eval ----------------
def train(model, tr_loader, va_loader, epochs, lr, savepath):
    model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=L2_REG)
    sch = optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)
    best_acc = 0.0
    for ep in range(epochs):
        model.train(); run=0.0; n=0
        for x,y in tr_loader:
            x=x.to(DEVICE); y=y.to(DEVICE)
            opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step()
            run += float(loss.item())*x.size(0); n+=x.size(0)
        sch.step()
        tr_loss = run/max(1,n)
        va_loss, correct, tot = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for x,y in va_loader:
                x=x.to(DEVICE); y=y.to(DEVICE)
                out=model(x); loss=crit(out,y)
                va_loss += float(loss.item())*x.size(0)
                pred = out.argmax(1)
                correct += (pred==y).sum().item(); tot+=y.size(0)
        va_loss /= max(1,tot); acc = correct/max(1,tot)
        print(f'Epoch {ep+1}/{epochs}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}, val_acc={acc:.4f}')
        if acc>best_acc:
            torch.save(model.state_dict(), savepath)
            best_acc=acc

def evaluate(model, loader, num_classes):
    model.eval()
    ys, yhat = [], []
    with torch.no_grad():
        for x,y in loader:
            x=x.to(DEVICE)
            out=model(x)
            yhat += out.argmax(1).cpu().tolist()
            ys += y.tolist()
    cm = confusion_matrix(ys, yhat, labels=list(range(num_classes)))
    P,R,F,_ = precision_recall_fscore_support(ys, yhat, labels=list(range(num_classes)), zero_division=0)
    return cm, P,R,F

def plot_confusion(cm, classes, savepath, title='Confusion Matrix'):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    tick = np.arange(len(classes))
    plt.xticks(tick, classes, rotation=45); plt.yticks(tick, classes)
    plt.tight_layout(); plt.xlabel('Pred'); plt.ylabel('True')
    plt.savefig(savepath, dpi=150); plt.close()

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train_ds, val_ds, class_names = build_classification_datasets(IMAGE_FOLDER, train_transform, val_transform)
    print('Classes:', class_names)
    tr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    va = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = TinyCNN(num_classes=len(class_names))
    savepath = os.path.join(RESULTS_FOLDER, 'tinycnn_best.pth')
    train(model, tr, va, MAX_EPOCHS, INITIAL_LR, savepath)

    # reload best for eval
    model.load_state_dict(torch.load(savepath, map_location=DEVICE))
    cm,P,R,F = evaluate(model, va, len(class_names))
    plot_confusion(cm, class_names, os.path.join(FIG_FOLDER,'confusion_tinycnn.png'))

    summary = {
        'classes': class_names,
        'precision': [float(x) for x in P],
        'recall':    [float(x) for x in R],
        'f1':        [float(x) for x in F]
    }
    with open(os.path.join(RESULTS_FOLDER,'metrics_tinycnn.json'),'w') as f:
        json.dump(summary, f, indent=2)
    print('Done. Results in', RESULTS_FOLDER)

if __name__=='__main__':
    main()
