#!/usr/bin/env python3
import argparse, os, json, random, yaml
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()

def load_cfg(path):
    with open(path,"r") as f: cfg=yaml.safe_load(f)
    cfg.setdefault("data",{}); cfg.setdefault("system",{}); cfg.setdefault("train",{})
    d=cfg["data"]; d.setdefault("root","./pests_data"); d.setdefault("img_size",224); d.setdefault("resize",256); d.setdefault("train_frac",0.7); d.setdefault("val_frac",0.15); d.setdefault("test_frac",0.15); d.setdefault("batch_size",64); d.setdefault("num_workers",4); d.setdefault("balance_seed",42)
    s=cfg["system"]; s.setdefault("device","cuda" if torch.cuda.is_available() else "cpu"); s.setdefault("seed",42); s.setdefault("fp16",False)
    t=cfg["train"]; t.setdefault("epochs",10); t.setdefault("lr",5e-4); t.setdefault("weight_decay",1e-4); t.setdefault("freeze_backbone",True); t.setdefault("label_smoothing",0.0); t.setdefault("ckpt_dir","./checkpoints"); t.setdefault("ckpt_name","mobilenetv3_best.pt"); t.setdefault("warmup_epochs",1)
    return cfg

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_transforms(img_size, resize):
    tr = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(img_size), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2,0.2,0.2,0.1), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ev = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return tr, ev

def stratified_split(ds, a,b,c, seed=42):
    assert abs(a+b+c-1.0)<1e-6
    rng=random.Random(seed); cls_idxs={}
    for i,(_,y) in enumerate(ds.samples): cls_idxs.setdefault(y,[]).append(i)
    tr,va,te=[],[],[]
    for y,idxs in cls_idxs.items():
        rng.shuffle(idxs); n=len(idxs); ntr=int(round(a*n)); nva=int(round(b*n)); nte=n-ntr-nva
        tr+=idxs[:ntr]; va+=idxs[ntr:ntr+nva]; te+=idxs[ntr+nva:]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te); return tr,va,te

def build_model(C, freeze_backbone=True):
    w=MobileNet_V3_Large_Weights.IMAGENET1K_V1; model=mobilenet_v3_large(weights=w)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, C)
    if freeze_backbone:
        for p in model.features.parameters(): p.requires_grad=False
    return model

def accuracy(logits,y):
    with torch.no_grad():
        pred=logits.argmax(1); return (pred==y).float().mean().item()*100.0

def main():
    args=parse_args(); cfg=load_cfg(args.config); set_seed(cfg["system"]["seed"]); device=torch.device(cfg["system"]["device"])
    tr_tf, ev_tf = build_transforms(cfg["data"]["img_size"], cfg["data"]["resize"])
    full = datasets.ImageFolder(cfg["data"]["root"], transform=tr_tf)
    tr_idx, va_idx, te_idx = stratified_split(full, cfg["data"]["train_frac"], cfg["data"]["val_frac"], cfg["data"]["test_frac"], cfg["data"]["balance_seed"])
    train_ds = Subset(datasets.ImageFolder(cfg["data"]["root"], transform=tr_tf), tr_idx)
    val_ds   = Subset(datasets.ImageFolder(cfg["data"]["root"], transform=ev_tf), va_idx)
    test_ds  = Subset(datasets.ImageFolder(cfg["data"]["root"], transform=ev_tf), te_idx)
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    C = len(full.class_to_idx); model=build_model(C, cfg["train"]["freeze_backbone"]).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True); best_path=os.path.join(cfg["train"]["ckpt_dir"], cfg["train"]["ckpt_name"]); best_top1=-1.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train(); tot_loss=0; tot_acc=0; n=0
        for x,y in train_dl:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            logits=model(x); loss=ce(logits,y); loss.backward(); opt.step()
            b=x.size(0); tot_loss+=loss.item()*b; tot_acc+=accuracy(logits,y)*b; n+=b
        va_loss=0; va_acc=0; nva=0; model.eval()
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device); logits=model(x); loss=ce(logits,y); b=x.size(0)
                va_loss+=loss.item()*b; va_acc+=accuracy(logits,y)*b; nva+=b
        tr_loss, tr_acc = tot_loss/max(1,n), tot_acc/max(1,n)
        va_loss, va_acc = va_loss/max(1,nva), va_acc/max(1,nva)
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train {tr_loss:.4f}/{tr_acc:.2f} | val {va_loss:.4f}/{va_acc:.2f}")
        if va_acc>best_top1:
            best_top1=va_acc
            torch.save({"model_state":model.state_dict(),"class_to_idx":full.class_to_idx,"cfg":cfg}, best_path)
            print(f"  -> saved best to {best_path} (val top1 {best_top1:.2f})")
    manifest={"class_to_idx": full.class_to_idx, "splits":{"train":[full.samples[i][0] for i in tr_idx], "val":[full.samples[i][0] for i in va_idx], "test":[full.samples[i][0] for i in te_idx]}, "best_checkpoint": best_path, "best_val_top1": best_top1}
    with open(os.path.join(cfg["train"]["ckpt_dir"], "split_manifest.json"), "w") as f: json.dump(manifest, f, indent=2)
    print("Saved split_manifest.json")
if __name__=="__main__":
    main()

