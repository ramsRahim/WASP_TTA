#!/usr/bin/env python3
# Baseline vs PN-cache TTA evaluation on test split

import argparse, os, json, yaml, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--config", required=True); return p.parse_args()

def load_cfg(path):
    with open(path,"r") as f: cfg=yaml.safe_load(f)
    cfg.setdefault("system",{}); cfg.setdefault("data",{}); cfg.setdefault("tta",{})
    s=cfg["system"]; s.setdefault("device","cuda" if torch.cuda.is_available() else "cpu")
    # Handle empty device string
    if not s["device"]:
        s["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    d=cfg["data"]; d.setdefault("img_size",224); d.setdefault("resize",256); d.setdefault("batch_size",64); d.setdefault("num_workers",4); d.setdefault("manifest_path","./checkpoints/split_manifest.json"); d.setdefault("ckpt_path","./checkpoints/mobilenetv3_best.pt")
    t=cfg["tta"]; t.setdefault("alpha",1.0); t.setdefault("beta",8.0); t.setdefault("gamma",6.0); t.setdefault("temperature",1.0); t.setdefault("tau_pos",0.80); t.setdefault("tau_ref",0.75); t.setdefault("tau_neg",0.20); t.setdefault("top_neg",3); t.setdefault("pos_cache_size",128); t.setdefault("neg_cache_size",64); t.setdefault("sim_aggregate","topk_mean"); t.setdefault("topk",5); t.setdefault("second_pass",True)
    # Add warmup settings
    t.setdefault("warmup_enabled", True); t.setdefault("warmup_samples_per_class", 10); t.setdefault("warmup_tau_pos", 0.90); t.setdefault("warmup_tau_neg", 0.15)
    return cfg

class SimpleImageDataset(Dataset):
    def __init__(self, paths, class_to_idx, img_size=224, resize=256):
        self.paths=paths; self.class_to_idx=class_to_idx
        self.tf=transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p=self.paths[idx]; img=Image.open(p).convert("RGB")
        cls_name=os.path.basename(os.path.dirname(p)); y=self.class_to_idx[cls_name]
        return self.tf(img), y, p

class FeatureHook:
    def __init__(self, layer: nn.Module): self.z=None; self.hook=layer.register_forward_pre_hook(self._hook)
    def _hook(self, module, inputs): 
        with torch.no_grad(): self.z=inputs[0].detach()
    def close(self): self.hook.remove()

class PosNegCache:
    def __init__(self, C, P, N):
        from collections import deque
        self.C=C; self.positives=[deque(maxlen=P) for _ in range(C)]; self.negatives=[deque(maxlen=N) for _ in range(C)]
    @torch.no_grad()
    def add_positive(self,c,z): self.positives[c].append(z.cpu())
    @torch.no_grad()
    def add_negative(self,c,z): self.negatives[c].append(z.cpu())
    @torch.no_grad()
    def agg(self, z, agg="topk_mean", topk=5):
        device=z.device; N=z.shape[0]; Spos=torch.zeros(N,self.C,device=device); Sneg=torch.zeros(N,self.C,device=device)
        for c in range(self.C):
            if len(self.positives[c])>0:
                P=torch.stack(list(self.positives[c]),0).to(device); sims=z@P.t(); Spos[:,c]=_agg(sims, agg, topk)
            if len(self.negatives[c])>0:
                Q=torch.stack(list(self.negatives[c]),0).to(device); sims=z@Q.t(); Sneg[:,c]=_agg(sims, agg, topk)
        return Spos,Sneg
    
    def cache_stats(self):
        """Return cache statistics"""
        pos_counts = [len(self.positives[c]) for c in range(self.C)]
        neg_counts = [len(self.negatives[c]) for c in range(self.C)]
        return {"pos_total": sum(pos_counts), "neg_total": sum(neg_counts), 
                "pos_per_class": pos_counts, "neg_per_class": neg_counts}

def _agg(sims, agg, topk):
    if sims.numel()==0: return torch.zeros(sims.shape[0], device=sims.device)
    if agg=="max": out,_=sims.max(1); return out
    if agg=="mean": return sims.mean(1)
    if agg=="topk_mean":
        k=min(topk, sims.shape[1]); v,_=torch.topk(sims, k=k, dim=1); return v.mean(1)
    raise ValueError("bad agg")

@torch.no_grad()
def warmup_cache(model, hook, val_dl, device, tta_cfg, C):
    """Warmup cache using validation data with ground truth labels"""
    print("Warming up cache with validation data...")
    cache = PosNegCache(C, tta_cfg["pos_cache_size"], tta_cfg["neg_cache_size"])
    
    # Track samples per class for balanced warmup
    class_counts = [0] * C
    max_per_class = tta_cfg["warmup_samples_per_class"]
    
    for x, y, _ in val_dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        z = hook.z
        probs = F.softmax(logits / tta_cfg["temperature"], dim=1)
        conf, pred = probs.max(1)
        
        for i in range(x.size(0)):
            true_class = int(y[i].item())
            pred_class = int(pred[i].item())
            confidence = float(conf[i].item())
            feature = F.normalize(z[i], dim=0)
            
            # Add to positive cache if prediction is correct and confident
            if pred_class == true_class and confidence >= tta_cfg["warmup_tau_pos"]:
                if class_counts[true_class] < max_per_class:
                    cache.add_positive(true_class, feature)
                    class_counts[true_class] += 1
            
            # Add to negative cache for incorrect classes with sufficient confidence
            if confidence >= tta_cfg["warmup_tau_neg"]:
                # Add current feature as negative for all other classes
                for c in range(C):
                    if c != true_class and class_counts[c] < max_per_class:
                        cache.add_negative(c, feature)
        
        # Check if we have enough samples for all classes
        if all(count >= max_per_class for count in class_counts):
            break
    
    stats = cache.cache_stats()
    print(f"Cache warmed up: {stats['pos_total']} positive, {stats['neg_total']} negative features")
    print(f"Avg per class: {stats['pos_total']/C:.1f} pos, {stats['neg_total']/C:.1f} neg")
    
    return cache

@torch.no_grad()
def eval_baseline(model, dl, device):
    model.eval(); corr=0; tot=0
    for x,y,_ in dl:
        x,y=x.to(device),y.to(device); pred=model(x).argmax(1); corr+=(pred==y).sum().item(); tot+=x.size(0)
    return 100.0*corr/tot

@torch.no_grad()
def refine_logits(logits, z, cache, alpha, beta, gamma, sim_aggregate, topk):
    zN=F.normalize(z, dim=1); Spos,Sneg=cache.agg(zN, agg=sim_aggregate, topk=topk); return alpha*logits + beta*Spos - gamma*Sneg

@torch.no_grad()
def eval_tta(model, hook, dl, device, tta_cfg, C, pre_warmed_cache=None):
    if pre_warmed_cache is not None:
        cache = pre_warmed_cache
        print("Using pre-warmed cache for TTA evaluation")
    else:
        cache = PosNegCache(C, tta_cfg["pos_cache_size"], tta_cfg["neg_cache_size"])
        print("Starting TTA evaluation with empty cache")
    
    corr=0; tot=0
    for x,y,_ in dl:
        x,y=x.to(device),y.to(device)
        logits=model(x); z=hook.z
        ref1=refine_logits(logits, z, cache, tta_cfg["alpha"], tta_cfg["beta"], tta_cfg["gamma"], tta_cfg["sim_aggregate"], tta_cfg["topk"])
        probs1=F.softmax(ref1/tta_cfg["temperature"], dim=1); conf1, pred1 = probs1.max(1)
        # update caches
        pos_mask=conf1>=tta_cfg["tau_pos"]
        for i in range(x.size(0)):
            if pos_mask[i]: cache.add_positive(int(pred1[i].item()), F.normalize(z[i], dim=0))
        for i in range(x.size(0)):
            pi=probs1[i]; pc=int(pred1[i].item())
            vals,idxs=torch.topk(pi, k=min(tta_cfg["top_neg"]+1, C))
            added=0
            for val,idx in zip(vals,idxs):
                c=int(idx.item())
                if c==pc: continue
                if float(val.item())>=tta_cfg["tau_neg"]:
                    cache.add_negative(c, F.normalize(z[i], dim=0)); added+=1
                if added>=tta_cfg["top_neg"]: break
        if tta_cfg["second_pass"]:
            ref2=refine_logits(logits, z, cache, tta_cfg["alpha"], tta_cfg["beta"], tta_cfg["gamma"], tta_cfg["sim_aggregate"], tta_cfg["topk"])
            probs2=F.softmax(ref2/tta_cfg["temperature"], dim=1); conf2, pred2 = probs2.max(1)
            final_pred=pred2.clone(); fallback=conf2<tta_cfg["tau_ref"]; final_pred[fallback]=pred1[fallback]
        else:
            final_pred=pred1
        corr+=(final_pred==y).sum().item(); tot+=x.size(0)
    return 100.0*corr/tot

def main():
    args=parse_args(); cfg=load_cfg(args.config); device=torch.device(cfg["system"]["device"])
    with open(cfg["data"]["manifest_path"],"r") as f: man=json.load(f)
    class_to_idx=man["class_to_idx"]; test_paths=man["splits"]["test"]; C=len(class_to_idx)
    
    # Create datasets
    test_ds=SimpleImageDataset(test_paths, class_to_idx, img_size=cfg["data"]["img_size"], resize=cfg["data"]["resize"])
    test_dl=DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    
    if cfg["tta"]["warmup_enabled"]:
        val_paths = man["splits"]["val"]
        val_ds = SimpleImageDataset(val_paths, class_to_idx, img_size=cfg["data"]["img_size"], resize=cfg["data"]["resize"])
        val_dl = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    
    # Load model
    w=MobileNet_V3_Large_Weights.IMAGENET1K_V1; model=mobilenet_v3_large(weights=w)
    inF=model.classifier[-1].in_features; model.classifier[-1]=nn.Linear(inF, C)
    ckpt=torch.load(cfg["data"]["ckpt_path"], map_location="cpu"); model.load_state_dict(ckpt["model_state"]); model.eval().to(device)
    
    # Setup feature hook
    final_linear=None
    for m in reversed(model.classifier):
        if isinstance(m, nn.Linear): final_linear=m; break
    hook=FeatureHook(final_linear)
    
    # Evaluate baseline
    base=eval_baseline(model, test_dl, device)
    
    # Evaluate TTA with optional warmup
    if cfg["tta"]["warmup_enabled"]:
        warmed_cache = warmup_cache(model, hook, val_dl, device, cfg["tta"], C)
        tta = eval_tta(model, hook, test_dl, device, cfg["tta"], C, warmed_cache)
        print(f"TTA (with warmup) Top-1: {tta:.2f}%")
    else:
        tta = eval_tta(model, hook, test_dl, device, cfg["tta"], C, None)
        print(f"TTA (no warmup) Top-1: {tta:.2f}%")
    
    print(f"Baseline Top-1: {base:.2f}%")
    print(f"Delta: {tta-base:+.2f}%")
    hook.close()

if __name__=="__main__":
    main()


#  energy based caching
# augment the dataset with artificial adversarial samples