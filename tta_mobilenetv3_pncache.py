#!/usr/bin/env python3
"""
Test-Time Adaptation for MobileNetV3 with Positive/Negative Cache (No Backprop)
Reads all settings from a YAML config.
"""
import argparse, csv, os, yaml
from collections import deque
from typing import List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torchvision

def parse_args():
    p = argparse.ArgumentParser(description="PN-cache TTA for MobileNetV3 (YAML-configured)")
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

def load_cfg(path):
    with open(path, "r") as f: cfg = yaml.safe_load(f)
    cfg.setdefault("data", {}); cfg.setdefault("system", {}); cfg.setdefault("model", {}); cfg.setdefault("tta", {}); cfg.setdefault("output", {})
    d=cfg["data"]; d.setdefault("data_dir","./data"); d.setdefault("pattern",None); d.setdefault("batch_size",64); d.setdefault("num_workers",4); d.setdefault("resize",256); d.setdefault("crop",224); d.setdefault("eval_if_labels",False)
    s=cfg["system"]; s.setdefault("device","cuda" if torch.cuda.is_available() else "cpu"); s.setdefault("fp16",False); s.setdefault("seed",42)
    m=cfg["model"]; m.setdefault("arch","mobilenet_v3_large")
    t=cfg["tta"]; t.setdefault("alpha",1.0); t.setdefault("beta",8.0); t.setdefault("gamma",6.0); t.setdefault("temperature",1.0); t.setdefault("tau_pos",0.80); t.setdefault("tau_ref",0.75); t.setdefault("tau_neg",0.20); t.setdefault("top_neg",3); t.setdefault("pos_cache_size",128); t.setdefault("neg_cache_size",64); t.setdefault("sim_aggregate","topk_mean"); t.setdefault("topk",5); t.setdefault("second_pass",True)
    o=cfg["output"]; o.setdefault("save_csv","predictions.csv")
    return cfg

def set_seed(seed:int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class UnlabeledImageDataset(Dataset):
    def __init__(self, root:str, transform=None, pattern:str=None, eval_if_labels:bool=False):
        self.transform=transform; self.eval_if_labels=eval_if_labels; self.samples=[]
        if eval_if_labels:
            classes=[]; class_to_idx={}
            for entry in sorted(os.listdir(root)):
                full=os.path.join(root,entry)
                if os.path.isdir(full): class_to_idx[entry]=len(classes); classes.append(entry)
            if len(classes)>0:
                for cname,cidx in class_to_idx.items():
                    cdir=os.path.join(root,cname)
                    for dp,_,fns in os.walk(cdir):
                        for f in sorted(fns):
                            if not _is_image_file(f): continue
                            if pattern and (pattern not in f): continue
                            self.samples.append((os.path.join(dp,f), cidx))
            else:
                for dp,_,fns in os.walk(root):
                    for f in sorted(fns):
                        if not _is_image_file(f): continue
                        if pattern and (pattern not in f): continue
                        self.samples.append((os.path.join(dp,f), -1))
        else:
            for dp,_,fns in os.walk(root):
                for f in sorted(fns):
                    if not _is_image_file(f): continue
                    if pattern and (pattern not in f): continue
                    self.samples.append((os.path.join(dp,f), -1))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label, path

def _is_image_file(fname:str)->bool:
    exts=[".jpg",".jpeg",".png",".bmp",".tiff",".webp"]; fl=fname.lower(); return any(fl.endswith(e) for e in exts)

class FeatureHook:
    def __init__(self, layer: nn.Module):
        self.z=None; self.hook=layer.register_forward_pre_hook(self._hook_fn)
    def _hook_fn(self, module, inputs):
        with torch.no_grad(): self.z = inputs[0].detach()
    def close(self): self.hook.remove()

class PosNegCache:
    def __init__(self, num_classes:int, pos_size:int, neg_size:int):
        self.C=num_classes; self.positives=[deque(maxlen=pos_size) for _ in range(num_classes)]; self.negatives=[deque(maxlen=neg_size) for _ in range(num_classes)]
    @torch.no_grad()
    def add_positive(self, cls:int, z:torch.Tensor): self.positives[cls].append(z.cpu())
    @torch.no_grad()
    def add_negative(self, cls:int, z:torch.Tensor): self.negatives[cls].append(z.cpu())
    @torch.no_grad()
    def aggregate_similarity(self, z:torch.Tensor, agg:str="topk_mean", topk:int=5):
        device=z.device; N,D=z.shape; C=self.C
        S_pos=torch.zeros(N,C,device=device); S_neg=torch.zeros(N,C,device=device)
        for c in range(C):
            if len(self.positives[c])>0:
                P=torch.stack(list(self.positives[c]),dim=0).to(device); sims=z@P.t(); S_pos[:,c]=_aggregate(sims,agg,topk)
            if len(self.negatives[c])>0:
                Q=torch.stack(list(self.negatives[c]),dim=0).to(device); sims=z@Q.t(); S_neg[:,c]=_aggregate(sims,agg,topk)
        return S_pos,S_neg

def _aggregate(sims:torch.Tensor, agg:str, topk:int)->torch.Tensor:
    if sims.numel()==0: return torch.zeros(sims.shape[0], device=sims.device)
    if agg=="max": out,_=sims.max(dim=1); return out
    elif agg=="mean": return sims.mean(dim=1)
    elif agg=="topk_mean":
        k=min(topk, sims.shape[1]); topk_vals,_=torch.topk(sims, k=k, dim=1); return topk_vals.mean(dim=1)
    else: raise ValueError(f"Unknown agg {agg}")

def build_model(device:str="cpu"):
    try:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1; model=mobilenet_v3_large(weights=weights)
    except Exception:
        model=torchvision.models.mobilenet_v3_large(pretrained=True)
    model.eval().to(device)
    if not isinstance(model.classifier, nn.Sequential): raise RuntimeError("Unexpected MobileNetV3 structure.")
    final_linear=None
    for m in reversed(model.classifier):
        if isinstance(m, nn.Linear): final_linear=m; break
    if final_linear is None: raise RuntimeError("No final Linear layer found.")
    hook=FeatureHook(final_linear); num_classes=final_linear.out_features
    return model, hook, num_classes

def build_transform(resize:int, crop:int):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

@torch.no_grad()
def refine_logits(base_logits, z, cache:PosNegCache, alpha, beta, gamma, sim_aggregate, topk):
    z_norm = torch.nn.functional.normalize(z, dim=1)
    S_pos, S_neg = cache.aggregate_similarity(z_norm, agg=sim_aggregate, topk=topk)
    refined = alpha*base_logits + beta*S_pos - gamma*S_neg
    return refined

class _nullcontext:
    def __enter__(self): return self
    def __exit__(self,*args): return False

def main():
    args=parse_args(); cfg=load_cfg(args.config); set_seed(cfg["system"]["seed"])
    device=torch.device(cfg["system"]["device"])
    tfm=build_transform(cfg["data"]["resize"], cfg["data"]["crop"])
    ds=UnlabeledImageDataset(cfg["data"]["data_dir"], transform=tfm, pattern=cfg["data"]["pattern"], eval_if_labels=cfg["data"]["eval_if_labels"])
    dl=DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    model, hook, num_classes = build_model(device=str(device))
    cache=PosNegCache(num_classes, pos_size=cfg["tta"]["pos_cache_size"], neg_size=cfg["tta"]["neg_cache_size"])
    autocast_ctx = torch.cuda.amp.autocast if (cfg['system']['fp16'] and device.type=='cuda') else _nullcontext
    rows=[("path","pred_base","conf_base","pred_refined","conf_refined")]
    tot=0; correct_base=0; correct_ref=0
    for imgs, labels, paths in dl:
        imgs=imgs.to(device, non_blocking=True); labels=labels.to(device, non_blocking=True)
        with autocast_ctx():
            logits=model(imgs); z=hook.z
            base_probs=torch.softmax(logits/cfg["tta"]["temperature"], dim=1)
            conf_base, pred_base = base_probs.max(dim=1)
            refined_logits = refine_logits(logits, z, cache, cfg["tta"]["alpha"], cfg["tta"]["beta"], cfg["tta"]["gamma"], cfg["tta"]["sim_aggregate"], cfg["tta"]["topk"])
            refined_probs = torch.softmax(refined_logits/cfg["tta"]["temperature"], dim=1)
            conf_ref, pred_ref = refined_probs.max(dim=1)
            # update caches
            pos_mask = conf_ref >= cfg["tta"]["tau_pos"]
            for i in range(imgs.size(0)):
                if pos_mask[i]:
                    c=int(pred_ref[i].item()); cache.add_positive(c, torch.nn.functional.normalize(z[i], dim=0))
            for i in range(imgs.size(0)):
                probs_i = refined_probs[i]; pred_c=int(pred_ref[i].item())
                vals, idxs = torch.topk(probs_i, k=min(cfg["tta"]["top_neg"]+1, num_classes))
                added=0
                for val, idx in zip(vals, idxs):
                    cls=int(idx.item())
                    if cls==pred_c: continue
                    if float(val.item())>=cfg["tta"]["tau_neg"]:
                        cache.add_negative(cls, torch.nn.functional.normalize(z[i], dim=0)); added+=1
                    if added>=cfg["tta"]["top_neg"]: break
            # optional second pass
            if cfg["tta"]["second_pass"]:
                refined_logits2 = refine_logits(logits, z, cache, cfg["tta"]["alpha"], cfg["tta"]["beta"], cfg["tta"]["gamma"], cfg["tta"]["sim_aggregate"], cfg["tta"]["topk"])
                refined_probs2 = torch.softmax(refined_logits2/cfg["tta"]["temperature"], dim=1)
                conf_ref2, pred_ref2 = refined_probs2.max(dim=1)
                final_pred = pred_ref2.clone(); final_conf = conf_ref2.clone()
                fallback = conf_ref2 < cfg["tta"]["tau_ref"]
                final_pred[fallback] = pred_ref[fallback]; final_conf[fallback] = conf_ref[fallback]
                fb2 = final_conf < cfg["tta"]["tau_ref"]
                final_pred[fb2] = pred_base[fb2]; final_conf[fb2] = conf_base[fb2]
            else:
                final_pred, final_conf = pred_ref, conf_ref
        for pth, pb, cb, pr, cr in zip(paths, pred_base.tolist(), conf_base.tolist(), final_pred.tolist(), final_conf.tolist()):
            rows.append((pth, int(pb), float(cb), int(pr), float(cr)))
    with open(cfg["output"]["save_csv"], "w", newline="") as f:
        import csv; writer=csv.writer(f); writer.writerows(rows)
    hook.close()
    print(f"[DONE] Saved predictions to: {cfg['output']['save_csv']}")

if __name__=="__main__":
    main()
