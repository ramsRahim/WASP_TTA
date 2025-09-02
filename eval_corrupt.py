#!/usr/bin/env python3
import argparse, os, json, yaml, torch
from eval_tta import load_cfg, SimpleImageDataset, FeatureHook, eval_baseline, eval_tta, warmup_cache
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn
import io 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--corruption", choices=['gaussian_noise', 'motion_blur', 'brightness', 'contrast', 'jpeg_compression', 'gaussian_blur'], required=True)
    p.add_argument("--severity", type=int, choices=[1,2,3,4,5], default=3)
    return p.parse_args()

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
    def __init__(self, paths, class_to_idx, img_size=224, resize=256, corruption_type=None, corruption_severity=1):
        self.paths=paths; self.class_to_idx=class_to_idx; self.corruption_type=corruption_type; self.corruption_severity=corruption_severity
        
        # Base transform without corruption
        self.base_tf = transforms.Compose([
            transforms.Resize(resize), 
            transforms.CenterCrop(img_size), 
            transforms.ToTensor()
        ])
        
        # Normalization (applied after corruption)
        self.normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        
        # Define corruption transforms
        self.corruption_transforms = {
            'gaussian_noise': lambda x, s: self._add_gaussian_noise(x, 0.1 * s),
            'motion_blur': lambda x, s: self._motion_blur(x, int(3 + 2*s)),
            'brightness': lambda x, s: transforms.ColorJitter(brightness=0.2*s)(x),
            'contrast': lambda x, s: transforms.ColorJitter(contrast=0.2*s)(x),
            'jpeg_compression': lambda x, s: self._jpeg_compression(x, max(10, 100-10*s)),
            'gaussian_blur': lambda x, s: transforms.GaussianBlur(kernel_size=3, sigma=0.5*s)(x),
        }
    
    def _add_gaussian_noise(self, tensor, std):
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0, 1)
    
    def _motion_blur(self, tensor, kernel_size):
        # Simple motion blur approximation using Gaussian blur
        return transforms.GaussianBlur(kernel_size, sigma=kernel_size/3)(tensor)
    
    def _jpeg_compression(self, tensor, quality):
        # Convert to PIL, apply JPEG compression, convert back
        pil_img = transforms.ToPILImage()(tensor)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        return transforms.ToTensor()(compressed)
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        cls_name = os.path.basename(os.path.dirname(p))
        y = self.class_to_idx[cls_name]
        
        # Apply base transform
        tensor = self.base_tf(img)
        
        # Apply corruption if specified
        if self.corruption_type and self.corruption_type in self.corruption_transforms:
            tensor = self.corruption_transforms[self.corruption_type](tensor, self.corruption_severity)
        
        # Apply normalization
        tensor = self.normalize(tensor)
        
        return tensor, y, p
    
def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(cfg["system"]["device"])
    
    with open(cfg["data"]["manifest_path"],"r") as f: 
        man = json.load(f)
    
    class_to_idx = man["class_to_idx"]
    test_paths = man["splits"]["test"]
    C = len(class_to_idx)
    
    print(f"Testing with {args.corruption} corruption at severity {args.severity}")
    
    # Create corrupted test dataset
    test_ds = SimpleImageDataset(
        test_paths, class_to_idx, 
        img_size=cfg["data"]["img_size"], 
        resize=cfg["data"]["resize"],
        corruption_type=args.corruption,
        corruption_severity=args.severity
    )
    test_dl = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, 
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)
    
    # Clean validation data for warmup
    if cfg["tta"]["warmup_enabled"]:
        val_paths = man["splits"]["val"]
        val_ds = SimpleImageDataset(val_paths, class_to_idx, 
                                  img_size=cfg["data"]["img_size"], 
                                  resize=cfg["data"]["resize"])
        val_dl = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, 
                           num_workers=cfg["data"]["num_workers"], pin_memory=True)
    
    # Load model
    w = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = mobilenet_v3_large(weights=w)
    inF = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(inF, C)
    ckpt = torch.load(cfg["data"]["ckpt_path"], map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    
    # Setup feature hook
    final_linear = None
    for m in reversed(model.classifier):
        if isinstance(m, nn.Linear): 
            final_linear = m
            break
    hook = FeatureHook(final_linear)
    
    # Evaluate baseline on corrupted data
    base = eval_baseline(model, test_dl, device)
    
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
    print(f"Corruption: {args.corruption}, Severity: {args.severity}")
    
    hook.close()

if __name__ == "__main__":
    main()