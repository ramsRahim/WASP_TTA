#!/usr/bin/env python3
# filepath: /home/rahim/exp/tta-pncache-tta/eval_corrupt.py
import argparse, os, json, yaml, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import functional as TF
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate corruption robustness: Vanilla vs Fine-tuned vs TTA")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--manifest_path", default="./checkpoints/split_manifest.json", help="Path to dataset split manifest")
    parser.add_argument("--vanilla_checkpoint", help="Path to vanilla model checkpoint") 
    parser.add_argument("--finetuned_checkpoint", required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--output_csv", default="corruption_robustness_results.csv", help="Output CSV file")
    parser.add_argument("--include_tta", action="store_true", help="Include TTA evaluation")
    return parser.parse_args()

def load_cfg(cfg_path):
    """Load YAML configuration"""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg.setdefault("data", {})
    cfg.setdefault("system", {})
    cfg.setdefault("tta", {})
    
    d = cfg["data"]
    d.setdefault("img_size", 224)
    d.setdefault("resize", 256)
    d.setdefault("batch_size", 64)
    d.setdefault("num_workers", 4)
    
    s = cfg["system"]
    s.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    if not s["device"]:
        s["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TTA defaults
    t = cfg["tta"]
    t.setdefault("alpha", 1.0)
    t.setdefault("beta", 8.0)
    t.setdefault("gamma", 6.0)
    t.setdefault("temperature", 1.0)
    t.setdefault("tau_pos", 0.80)
    t.setdefault("tau_ref", 0.75)
    t.setdefault("tau_neg", 0.20)
    t.setdefault("pos_cache_size", 128)
    t.setdefault("neg_cache_size", 64)
    t.setdefault("sim_aggregate", "topk_mean")
    t.setdefault("topk", 5)
    t.setdefault("warmup_enabled", True)
    t.setdefault("warmup_samples_per_class", 10)
    
    return cfg

def load_manifest(manifest_path):
    """Load dataset split manifest"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest

def build_transforms(img_size, resize):
    """Build data transforms"""
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MobileNetV3WithContrastive(nn.Module):
    """MobileNetV3 with contrastive learning components"""
    def __init__(self, num_classes, feature_dim=128, freeze_backbone=True, dropout_rate=0.3):
        super().__init__()
        
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_large(weights=weights)
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(960, num_classes)
        )
        
        self.contrastive_head = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
    
    def get_features(self, x):
        features = self.backbone.features(x)
        return self.backbone.avgpool(features).flatten(1)
    
    def get_contrastive_features(self, x):
        backbone_features = self.get_features(x)
        return F.normalize(self.contrastive_head(backbone_features), dim=1)
    
    def forward(self, x):
        features = self.get_features(x)
        return self.backbone.classifier(features)

class CorruptionDataset(Dataset):
    """Dataset that applies specific corruption to images from manifest paths"""
    def __init__(self, image_paths, class_to_idx, corruption_type, severity, transform=None):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Extract class name from path
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]
        
        img = Image.open(img_path).convert('RGB')
        
        # Apply corruption
        if self.corruption_type != 'clean':
            img = self._apply_corruption(img, self.corruption_type, self.severity)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, img_path
    
    def _apply_corruption(self, img, corruption_type, severity):
        """Apply corruption - matches train_finetune.py implementation"""
        if corruption_type == 'gaussian_noise':
            return self._add_gaussian_noise(img, severity)
        elif corruption_type == 'motion_blur':
            return self._add_motion_blur(img, severity)
        elif corruption_type == 'brightness':
            return self._adjust_brightness(img, severity)
        elif corruption_type == 'contrast':
            return self._adjust_contrast(img, severity)
        elif corruption_type == 'rain':
            return self._add_rain(img, severity)
        elif corruption_type == 'snow':
            return self._add_snow(img, severity)
        elif corruption_type == 'frost':
            return self._add_frost(img, severity)
        elif corruption_type == 'fog':
            return self._add_fog(img, severity)
        elif corruption_type == 'low_light':
            return self._adjust_low_light(img, severity)
        elif corruption_type == 'high_light':
            return self._adjust_high_light(img, severity)
        elif corruption_type == 'shadow':
            return self._add_shadow(img, severity)
        elif corruption_type == 'glare':
            return self._add_glare(img, severity)
        elif corruption_type == 'haze':
            return self._add_haze(img, severity)
        elif corruption_type == 'mist':
            return self._add_mist(img, severity)
        else:
            return img
    
    def _add_gaussian_noise(self, img, severity):
        tensor = TF.to_tensor(img)
        noise_std = 0.02 + (severity / 5.0) * 0.08
        noise = torch.randn_like(tensor) * noise_std
        noisy_tensor = torch.clamp(tensor + noise, 0, 1)
        return TF.to_pil_image(noisy_tensor)
    
    def _add_motion_blur(self, img, severity):
        kernel_size = min(15, 3 + 2 * severity)
        sigma = max(0.5, severity * 0.5)
        return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
    
    def _adjust_brightness(self, img, severity):
        factor = 0.5 + (severity / 5.0)
        return TF.adjust_brightness(img, brightness_factor=factor)
    
    def _adjust_contrast(self, img, severity):
        factor = 0.3 + (severity / 5.0) * 1.4
        return TF.adjust_contrast(img, contrast_factor=factor)
    
    def _add_rain(self, img, severity):
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img.size
        
        num_drops = int(30 * severity)
        for _ in range(num_drops):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            length = random.randint(10, 25)
            x2 = x1 + random.randint(-3, 3)
            y2 = min(height - 1, y1 + length)
            
            try:
                draw.line([(x1, y1), (x2, y2)], fill=(200, 200, 255), width=1)
            except:
                pass
        return img_copy
    
    def _add_snow(self, img, severity):
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img.size
        
        snow_density = int(50 * severity)
        for _ in range(snow_density):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 3)
            
            try:
                draw.ellipse([x-size, y-size, x+size, y+size], fill=(255, 255, 255))
            except:
                pass
        return img_copy
    
    def _add_frost(self, img, severity):
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.3 + 0.1 * severity)
        frost_overlay = Image.new('RGB', img.size, (150, 150, 255))
        frost_alpha = int(30 * severity)
        alpha_mask = Image.new('L', img.size, frost_alpha)
        return Image.composite(frost_overlay, img, alpha_mask)
    
    def _add_fog(self, img, severity):
        fog_intensity = int(50 + 30 * severity)
        fog_overlay = Image.new('RGB', img.size, (200, 200, 200))
        alpha_mask = Image.new('L', img.size, fog_intensity)
        return Image.composite(fog_overlay, img, alpha_mask)
    
    def _adjust_low_light(self, img, severity):
        brightness_factor = 1.0 - (severity / 5.0) * 0.7
        contrast_factor = 0.8 - (severity / 5.0) * 0.3
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        return img
    
    def _adjust_high_light(self, img, severity):
        brightness_factor = 1.0 + (severity / 5.0) * 0.8
        return TF.adjust_brightness(img, brightness_factor)
    
    def _add_shadow(self, img, severity):
        shadow_strength = int(40 + severity * 20)
        shadow_overlay = Image.new('RGB', img.size, (0, 0, 0))
        alpha_mask = Image.new('L', img.size, shadow_strength)
        return Image.composite(shadow_overlay, img, alpha_mask)
    
    def _add_glare(self, img, severity):
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img.size
        
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)
        radius = 15 + severity * 8
        
        try:
            glare_color = (255, 255, min(255, 200 + severity * 10))
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], 
                        fill=glare_color)
        except:
            pass
        return img_copy
    
    def _add_haze(self, img, severity):
        haze_intensity = int(30 + 10 * severity)
        haze_overlay = Image.new('RGB', img.size, (220, 220, 200))
        alpha_mask = Image.new('L', img.size, haze_intensity)
        return Image.composite(haze_overlay, img, alpha_mask)
    
    def _add_mist(self, img, severity):
        mist_intensity = int(40 + 10 * severity)
        mist_overlay = Image.new('RGB', img.size, (200, 210, 220))
        alpha_mask = Image.new('L', img.size, mist_intensity)
        return Image.composite(mist_overlay, img, alpha_mask)

class FeatureHook:
    """Feature extraction hook for TTA"""
    def __init__(self, layer):
        self.layer = layer
        self.z = None
        self.hook_handle = layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        self.z = output
    
    def close(self):
        self.hook_handle.remove()

class PosNegCache:
    """Positive/Negative cache for TTA"""
    def __init__(self, num_classes, pos_size=128, neg_size=64):
        from collections import deque
        self.C = num_classes
        self.positives = [deque(maxlen=pos_size) for _ in range(num_classes)]
        self.negatives = [deque(maxlen=neg_size) for _ in range(num_classes)]
    
    def add_positive(self, class_idx, feature):
        self.positives[class_idx].append(feature.detach().cpu())
    
    def add_negative(self, class_idx, feature):
        self.negatives[class_idx].append(feature.detach().cpu())
    
    @torch.no_grad()
    def aggregate_similarity(self, z, agg="topk_mean", topk=5):
        device = z.device
        N = z.shape[0]
        Spos = torch.zeros(N, self.C, device=device)
        Sneg = torch.zeros(N, self.C, device=device)
        
        for c in range(self.C):
            if len(self.positives[c]) > 0:
                P = torch.stack(list(self.positives[c]), 0).to(device)
                sims = z @ P.t()
                if agg == "topk_mean":
                    Spos[:, c] = sims.topk(min(topk, sims.shape[1]), dim=1)[0].mean(1)
                else:
                    Spos[:, c] = sims.mean(1)
            
            if len(self.negatives[c]) > 0:
                Q = torch.stack(list(self.negatives[c]), 0).to(device)
                sims = z @ Q.t()
                if agg == "topk_mean":
                    Sneg[:, c] = sims.topk(min(topk, sims.shape[1]), dim=1)[0].mean(1)
                else:
                    Sneg[:, c] = sims.mean(1)
        
        return Spos, Sneg

def load_model(checkpoint_path, model_type, num_classes, feature_dim=256):
    """Load model from checkpoint"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        if model_type == 'vanilla':
            # Create vanilla model without loading checkpoint
            print(f"Creating vanilla MobileNet model (no checkpoint provided)")
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            model = mobilenet_v3_large(weights=weights)
            model.classifier = nn.Linear(960, num_classes)
            # Initialize randomly for our classes
            nn.init.xavier_uniform_(model.classifier.weight)
            nn.init.zeros_(model.classifier.bias)
            return model, []
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint first to determine model structure
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to determine model type from checkpoint
    if 'model_type' in checkpoint:
        detected_type = checkpoint['model_type']
        if 'contrastive' in detected_type.lower():
            model_type = 'contrastive'
    
    # Create model based on type
    if model_type == 'contrastive' or 'contrastive_head' in str(checkpoint.get('model_state_dict', {}).keys()):
        model = MobileNetV3WithContrastive(
            num_classes=num_classes, 
            feature_dim=feature_dim, 
            freeze_backbone=False
        )
        print(f"✓ Created contrastive model")
    else:  # vanilla
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights)
        model.classifier = nn.Linear(960, num_classes)
        print(f"✓ Created vanilla model")
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✓ Loaded {model_type} model from {checkpoint_path}")
    return model, checkpoint.get('class_names', [])

@torch.no_grad()
def evaluate_model(model, dataloader, device, model_name="Model"):
    """Evaluate model on dataloader"""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels, _ in tqdm(dataloader, desc=f"Evaluating {model_name}"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

@torch.no_grad()
def warmup_cache(model, hook, val_dl, device, tta_cfg, C):
    """Warmup cache using validation data"""
    cache = PosNegCache(C, tta_cfg["pos_cache_size"], tta_cfg["neg_cache_size"])
    
    for x, y, _ in tqdm(val_dl, desc="Warming up cache"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        z = F.normalize(hook.z, dim=1)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        
        # Add high-confidence correct predictions as positives
        correct_mask = (pred == y) & (conf >= tta_cfg.get("warmup_tau_pos", 0.8))
        for i in range(x.size(0)):
            if correct_mask[i]:
                cache.add_positive(int(y[i].item()), z[i])
    
    return cache

@torch.no_grad()
def evaluate_with_tta(model, hook, dataloader, device, tta_cfg, C, pre_warmed_cache=None):
    """Evaluate model with TTA"""
    if pre_warmed_cache is not None:
        cache = pre_warmed_cache
    else:
        cache = PosNegCache(C, tta_cfg["pos_cache_size"], tta_cfg["neg_cache_size"])
    
    correct = 0
    total = 0
    
    for x, y, _ in tqdm(dataloader, desc="Evaluating with TTA"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        z = F.normalize(hook.z, dim=1)
        
        # Refine logits using cache
        Spos, Sneg = cache.aggregate_similarity(z, tta_cfg["sim_aggregate"], tta_cfg["topk"])
        refined_logits = tta_cfg["alpha"] * logits + tta_cfg["beta"] * Spos - tta_cfg["gamma"] * Sneg
        
        probs = F.softmax(refined_logits / tta_cfg["temperature"], dim=1)
        conf, pred = probs.max(1)
        
        # Update cache
        for i in range(x.size(0)):
            if conf[i] >= tta_cfg["tau_pos"]:
                cache.add_positive(int(pred[i].item()), z[i])
            elif conf[i] <= tta_cfg["tau_neg"]:
                cache.add_negative(int(pred[i].item()), z[i])
        
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy

def get_model_num_classes(model):
    """Get number of classes from model"""
    if hasattr(model, 'backbone'):
        # Contrastive model
        final_layer = model.backbone.classifier[-1]
    else:
        # Vanilla model  
        final_layer = model.classifier
    
    if hasattr(final_layer, 'out_features'):
        return final_layer.out_features
    elif hasattr(final_layer, 'weight'):
        return final_layer.weight.shape[0]
    else:
        return None

def evaluate_single_corruption(models, val_paths, test_paths, class_to_idx, cfg, corruption_type, severity, device, include_tta=False):
    """Evaluate all models on a single corruption"""
    
    # Create transform
    transform = build_transforms(cfg["data"]["img_size"], cfg["data"]["resize"])
    
    # Create corrupted test dataset
    corrupted_ds = CorruptionDataset(
        test_paths, class_to_idx, corruption_type, severity, transform=transform
    )
    
    corrupted_dl = DataLoader(
        corrupted_ds, 
        batch_size=cfg["data"]["batch_size"], 
        shuffle=False, 
        num_workers=cfg["data"]["num_workers"]
    )
    
    results = {
        'corruption': corruption_type,
        'severity': severity,
    }
    
    # Evaluate each model
    for model_name, model in models.items():
        if model is None:
            continue
            
        acc = evaluate_model(model, corrupted_dl, device, f"{model_name} ({corruption_type}_sev{severity})")
        results[f'{model_name.lower()}_baseline'] = acc
        
        # TTA evaluation for fine-tuned model
        if include_tta and model_name == 'Fine-tuned':
            # Setup hook for TTA
            final_linear = None
            if hasattr(model, 'backbone'):
                # Contrastive model
                for m in reversed(model.backbone.classifier):
                    if isinstance(m, nn.Linear):
                        final_linear = m
                        break
            else:
                # Vanilla model
                for m in reversed(model.classifier):
                    if isinstance(m, nn.Linear):
                        final_linear = m
                        break
            
            if final_linear is not None:
                hook = FeatureHook(final_linear)
                
                # Get number of classes
                num_classes = get_model_num_classes(model)
                
                # Create validation dataloader for warmup
                val_ds = CorruptionDataset(val_paths, class_to_idx, 'clean', 0, transform=transform)
                val_dl = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, 
                                   num_workers=cfg["data"]["num_workers"])
                
                # Warmup cache with clean validation data
                cache = warmup_cache(model, hook, val_dl, device, cfg["tta"], num_classes)
                
                # Evaluate with TTA
                tta_acc = evaluate_with_tta(model, hook, corrupted_dl, device, cfg["tta"], num_classes, cache)
                results['fine-tuned_tta'] = tta_acc
                results['tta_improvement'] = tta_acc - acc
                
                hook.close()
    
    return results

def print_comprehensive_table(all_results, include_tta=False):
    """Print comprehensive results table"""
    if not all_results:
        print("No results to display!")
        return
    
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*120}")
    print("COMPREHENSIVE CORRUPTION ROBUSTNESS EVALUATION")
    print(f"{'='*120}")
    
    # Print detailed results
    print(f"\n{'DETAILED RESULTS'}")
    header = f"{'Corruption':<15} {'Sev':<4}"
    if 'vanilla_baseline' in df.columns:
        header += f" {'Vanilla':<8}"
    if 'fine-tuned_baseline' in df.columns:
        header += f" {'Fine-tuned':<11}"
    if include_tta and 'fine-tuned_tta' in df.columns:
        header += f" {'TTA':<8} {'TTA Δ':<8}"
    if 'vanilla_baseline' in df.columns and 'fine-tuned_baseline' in df.columns:
        header += f" {'FT Δ':<8}"
    
    print(header)
    print("-" * 120)
    
    for _, row in df.iterrows():
        corruption = row['corruption'][:12]
        severity = int(row['severity'])
        
        line = f"{corruption:<15} {severity:<4}"
        
        if 'vanilla_baseline' in row:
            vanilla = row['vanilla_baseline']
            line += f" {vanilla:<8.2f}"
        
        if 'fine-tuned_baseline' in row:
            finetuned = row['fine-tuned_baseline']
            line += f" {finetuned:<11.2f}"
        
        if include_tta and 'fine-tuned_tta' in row:
            tta = row['fine-tuned_tta']
            tta_delta = row.get('tta_improvement', 0)
            line += f" {tta:<8.2f} {tta_delta:<8.2f}"
        
        if 'vanilla_baseline' in row and 'fine-tuned_baseline' in row:
            ft_delta = row['fine-tuned_baseline'] - row['vanilla_baseline']
            line += f" {ft_delta:<8.2f}"
        
        print(line)
    
    # Calculate averages by corruption type
    print(f"\n{'AVERAGE BY CORRUPTION TYPE'}")
    print(header)
    print("-" * 120)
    
    corruption_summary = df.groupby('corruption').agg({
        col: 'mean' for col in df.columns if col not in ['corruption', 'severity']
    }).round(2)
    
    for corruption, row in corruption_summary.iterrows():
        corruption_name = corruption[:12]
        
        line = f"{corruption_name:<15} {'Avg':<4}"
        
        if 'vanilla_baseline' in row:
            vanilla = row['vanilla_baseline']
            line += f" {vanilla:<8.2f}"
        
        if 'fine-tuned_baseline' in row:
            finetuned = row['fine-tuned_baseline']
            line += f" {finetuned:<11.2f}"
        
        if include_tta and 'fine-tuned_tta' in row:
            tta = row['fine-tuned_tta']
            tta_delta = row.get('tta_improvement', 0)
            line += f" {tta:<8.2f} {tta_delta:<8.2f}"
        
        if 'vanilla_baseline' in row and 'fine-tuned_baseline' in row:
            ft_delta = row['fine-tuned_baseline'] - row['vanilla_baseline']
            line += f" {ft_delta:<8.2f}"
        
        print(line)
    
    # Overall averages
    print("-" * 120)
    line = f"{'OVERALL AVERAGE':<15} {'All':<4}"
    
    if 'vanilla_baseline' in df.columns:
        overall_vanilla = df['vanilla_baseline'].mean()
        line += f" {overall_vanilla:<8.2f}"
    
    if 'fine-tuned_baseline' in df.columns:
        overall_finetuned = df['fine-tuned_baseline'].mean()
        line += f" {overall_finetuned:<11.2f}"
    
    if include_tta and 'fine-tuned_tta' in df.columns:
        overall_tta = df['fine-tuned_tta'].mean()
        overall_tta_delta = df['tta_improvement'].mean()
        line += f" {overall_tta:<8.2f} {overall_tta_delta:<8.2f}"
    
    if 'vanilla_baseline' in df.columns and 'fine-tuned_baseline' in df.columns:
        overall_ft_delta = overall_finetuned - overall_vanilla
        line += f" {overall_ft_delta:<8.2f}"
    
    print(line)
    print(f"{'='*120}")
    
    # Summary insights
    print(f"\nKEY INSIGHTS:")
    if 'vanilla_baseline' in df.columns and 'fine-tuned_baseline' in df.columns:
        overall_ft_improvement = overall_finetuned - overall_vanilla
        print(f"• Fine-tuning improves robustness by {overall_ft_improvement:+.2f}% on average")
    
    if include_tta and 'fine-tuned_tta' in df.columns:
        overall_tta_improvement = df['tta_improvement'].mean()
        print(f"• TTA provides additional {overall_tta_improvement:+.2f}% improvement over fine-tuned model")
        
        if 'vanilla_baseline' in df.columns:
            total_improvement = overall_tta - overall_vanilla
            print(f"• Combined improvement over vanilla: {total_improvement:+.2f}%")
    
    # Best and worst corruptions
    if len(corruption_summary) > 1 and 'fine-tuned_baseline' in corruption_summary.columns:
        best_corruption = corruption_summary['fine-tuned_baseline'].idxmax()
        worst_corruption = corruption_summary['fine-tuned_baseline'].idxmin()
        
        print(f"• Most robust to: {best_corruption} ({corruption_summary.loc[best_corruption, 'fine-tuned_baseline']:.2f}%)")
        print(f"• Least robust to: {worst_corruption} ({corruption_summary.loc[worst_corruption, 'fine-tuned_baseline']:.2f}%)")
        
        if include_tta and 'tta_improvement' in corruption_summary.columns:
            best_tta_corruption = corruption_summary['tta_improvement'].idxmax()
            print(f"• Best TTA improvement: {best_tta_corruption} ({corruption_summary.loc[best_tta_corruption, 'tta_improvement']:+.2f}%)")

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(cfg["system"]["device"])
    
    # Set random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Load manifest
    manifest = load_manifest(args.manifest_path)
    
    # Extract dataset info from manifest
    class_to_idx = manifest["class_to_idx"]
    test_paths = manifest["splits"]["test"]
    val_paths = manifest["splits"]["val"]
    num_classes = manifest["dataset_info"]["num_classes"]
    class_names = manifest["dataset_info"]["class_names"]
    
    print(f"Using manifest: {args.manifest_path}")
    print(f"Detected {num_classes} classes: {class_names}")
    print(f"Test samples: {len(test_paths)}, Val samples: {len(val_paths)}")
    
    # Load models
    models = {}
    
    # Load fine-tuned model
    finetuned_model, _ = load_model(
        args.finetuned_checkpoint, 'contrastive', num_classes,
        cfg.get("corruption_finetune", {}).get("feature_dim", 256)
    )
    finetuned_model.to(device)
    models['Fine-tuned'] = finetuned_model
    
    # Load vanilla model if provided
    if args.vanilla_checkpoint:
        try:
            vanilla_model, _ = load_model(args.vanilla_checkpoint, 'vanilla', num_classes)
            vanilla_model.to(device)
            models['Vanilla'] = vanilla_model
        except Exception as e:
            print(f"Warning: Could not load vanilla model: {e}")
            print("Continuing with fine-tuned model only...")
    
    # Define all corruptions and severities to test
    all_corruptions = [
        'gaussian_noise', 'motion_blur', 'brightness', 'contrast',
        'rain', 'snow', 'frost', 'fog', 'low_light', 'high_light', 
        'shadow', 'glare', 'haze', 'mist'
    ]
    severities = [1, 2, 3, 4, 5]
    
    print(f"\nTesting {len(all_corruptions)} corruption types at {len(severities)} severity levels")
    print(f"Corruptions: {all_corruptions}")
    print(f"Severities: {severities}")
    print(f"Total tests: {len(all_corruptions) * len(severities)}")
    
    # Run evaluation
    all_results = []
    total_tests = len(all_corruptions) * len(severities)
    current_test = 0
    
    for corruption in all_corruptions:
        for severity in severities:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] Evaluating {corruption} at severity {severity}")
            
            try:
                results = evaluate_single_corruption(
                    models, val_paths, test_paths, class_to_idx, cfg, 
                    corruption, severity, device, args.include_tta
                )
                all_results.append(results)
                
                # Print immediate results
                result_str = ""
                if 'fine-tuned_baseline' in results:
                    result_str += f"  Fine-tuned: {results['fine-tuned_baseline']:.2f}%"
                if 'vanilla_baseline' in results:
                    vanilla_acc = results['vanilla_baseline']
                    ft_delta = results['fine-tuned_baseline'] - vanilla_acc
                    result_str += f" | Vanilla: {vanilla_acc:.2f}% | FT Δ: {ft_delta:+.2f}%"
                if args.include_tta and 'fine-tuned_tta' in results:
                    tta_acc = results['fine-tuned_tta']
                    tta_delta = results['tta_improvement']
                    result_str += f" | TTA: {tta_acc:.2f}% | TTA Δ: {tta_delta:+.2f}%"
                
                print(result_str)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print comprehensive results table
    if all_results:
        print_comprehensive_table(all_results, args.include_tta)
        
        # Save to CSV
        df = pd.DataFrame(all_results)
        df['timestamp'] = datetime.now().isoformat()
        df.to_csv(args.output_csv, index=False)
        print(f"\nDetailed results saved to: {args.output_csv}")
        
        # Print model info
        print(f"\nMODEL INFORMATION:")
        print(f"Manifest: {args.manifest_path}")
        print(f"Fine-tuned model: {args.finetuned_checkpoint}")
        if args.vanilla_checkpoint:
            print(f"Vanilla model: {args.vanilla_checkpoint}")
        print(f"Total corruptions tested: {len(set(r['corruption'] for r in all_results))}")
        print(f"Total combinations tested: {len(all_results)}")
        
    else:
        print("No results generated!")
    
    print(f"\nEvaluation completed!")

if __name__ == "__main__":
    main()