#!/usr/bin/env python3
"""
ResNet-50 finetune (GPU-only)
- Uses torchvision weights API when available; falls back gracefully
"""
import os, random, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

DATA_ROOT = r'D:/claude/SemanticSegmentationUsingFCN-AlexNet1/Dataset4'
IMAGE_FOLDER = os.path.join(DATA_ROOT, 'ImagesReszed')
RESULTS_FOLDER = os.path.join(DATA_ROOT, 'resnet50_results')
FIG_FOLDER = os.path.join(RESULTS_FOLDER, 'figures')
os.makedirs(RESULTS_FOLDER, exist_ok=True); os.makedirs(FIG_FOLDER, exist_ok=True)

SEED=0; BATCH_SIZE=16; EPOCHS=8; LR=2e-5; L2=1e-4; INPUT_SIZE=(224,224)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    transforms.RandomAffine(degrees=0, translate=(6/INPUT_SIZE[1], 6/INPUT_SIZE[0])),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
])
val_tf = transforms.Compose([transforms.Resize(INPUT_SIZE), transforms.ToTensor()])

class Flat(Dataset):
    def __init__(self, ps, ys, tf=None): self.ps=ps; self.ys=ys; self.tf=tf
    def __len__(self): return len(self.ps)
    def __getitem__(self,i):
        im=Image.open(self.ps[i]).convert('RGB'); 
        return (self.tf(im) if self.tf else transforms.ToTensor()(im)), self.ys[i]

def list_imgs(folder):
    exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff'); out=[]
    for e in exts: out+=sorted(Path(folder).rglob(f'*{e}'))
    return [str(p) for p in out]

def build_ds(root):
    try:
        imf=datasets.ImageFolder(root)
        if imf.classes:
            paths=[p for p,_ in imf.samples]; labels=[y for _,y in imf.samples]; classes=imf.classes
        else: imf=None
    except: imf=None
    if imf is None:
        print('Inferring labels from filenames...')
        files=list_imgs(root); 
        if not files: raise FileNotFoundError('No images.')
        kw={'forged':'forged','fake':'forged','tamper':'forged','tampered':'forged',
            'original':'original','auth':'original','real':'original','genuine':'original','pristine':'original'}
        inf=[]
        for p in files:
            nm=Path(p).stem.lower(); lab=None
            for k,v in kw.items():
                if k in nm: lab=v; break
            if lab is None:
                toks=[]; 
                for s in ['_','-','.']: toks+=[t for t in nm.split(s) if t]
                if toks and not toks[0].isdigit(): lab=toks[0]
                if lab is None and toks and not toks[-1].isdigit(): lab=toks[-1]
            inf.append((p, lab if lab else 'unknown'))
        classes=sorted(set(l for _,l in inf))
        if classes==['unknown']: raise RuntimeError('Failed to infer classes.')
        map_={c:i for i,c in enumerate(classes)}
        paths=[p for p,_ in inf]; labels=[map_[l] for _,l in inf]
    # filter unreadable
    p2,y2=[],[]
    for p,y in zip(paths,labels):
        try: Image.open(p).close(); p2.append(p); y2.append(y)
        except: print('Bad:', p)
    paths,labels=p2,y2
    # split 80/20 per-class
    from collections import defaultdict
    dd=defaultdict(list); [dd[y].append(i) for i in range(len(labels))]
    rng=random.Random(SEED); tr_idx=[]; va_idx=[]
    for y,arr in dd.items():
        rng.shuffle(arr); ntr=max(1,int(0.8*len(arr))) if len(arr)>1 else 1
        tr_idx+=arr[:ntr]; va_idx+=arr[ntr:] if len(arr)>1 else arr[:0]
    trp=[paths[i] for i in tr_idx]; try_=[labels[i] for i in tr_idx]
    vap=[paths[i] for i in va_idx]; vay=[labels[i] for i in va_idx]
    return Flat(trp,try_,train_tf), Flat(vap,vay,val_tf), classes

def make_model(nc):
    # Try new weights API, fall back to old
    try:
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, nc)
    return net

def train(model, tr, va):
    model.to(DEVICE)
    crit=nn.CrossEntropyLoss()
    opt=optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=L2)
    sch=optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)
    best=0.0; savepath=os.path.join(RESULTS_FOLDER,'resnet50_best.pth')
    for ep in range(EPOCHS):
        model.train(); run=0.0;n=0
        for x,y in tr:
            x=x.to(DEVICE); y=y.to(DEVICE)
            opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step()
            run+=float(loss.item())*x.size(0); n+=x.size(0)
        sch.step()
        model.eval(); corr=0; tot=0; vloss=0.0
        with torch.no_grad():
            for x,y in va:
                x=x.to(DEVICE); y=y.to(DEVICE)
                o=model(x); l=crit(o,y)
                vloss+=float(l.item())*x.size(0)
                corr+=(o.argmax(1)==y).sum().item(); tot+=y.size(0)
        acc=corr/max(1,tot); vloss/=max(1,tot)
        print(f'Epoch {ep+1}/{EPOCHS}: tr_loss={run/max(1,n):.4f}, val_loss={vloss:.4f}, val_acc={acc:.4f}')
        if acc>best: best=acc; torch.save(model.state_dict(), savepath)
    return savepath

def eval_model(model, va, ncls):
    model.eval(); ys=[]; yh=[]
    with torch.no_grad():
        for x,y in va:
            x=x.to(DEVICE); o=model(x); yh+=o.argmax(1).cpu().tolist(); ys+=y.tolist()
    cm=confusion_matrix(ys,yh,labels=list(range(ncls)))
    P,R,F,_=precision_recall_fscore_support(ys,yh,labels=list(range(ncls)),zero_division=0)
    return cm,P,R,F

def plot_cm(cm, classes, path):
    plt.figure(figsize=(5,4)); plt.imshow(cm); plt.title('ResNet50 Confusion'); plt.colorbar()
    t=np.arange(len(classes)); plt.xticks(t,classes,rotation=45); plt.yticks(t,classes)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tr_ds, va_ds, classes = build_ds(IMAGE_FOLDER)
    tr=DataLoader(tr_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    va=DataLoader(va_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)
    model=make_model(len(classes))
    best=train(model,tr,va)
    model.load_state_dict(torch.load(best,map_location=DEVICE))
    cm,P,R,F=eval_model(model,va,len(classes))
    plot_cm(cm,classes,os.path.join(FIG_FOLDER,'confusion_resnet50.png'))
    with open(os.path.join(RESULTS_FOLDER,'metrics_resnet50.json'),'w') as f:
        json.dump({'classes':classes,'precision':[float(x) for x in P],'recall':[float(x) for x in R],'f1':[float(x) for x in F]},f,indent=2)
    print('Done ->', RESULTS_FOLDER)

if __name__=='__main__': main()
