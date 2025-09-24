#!/usr/bin/env python3
"""
Enhanced Fine-tuning Script with Clean Data Regularization
"""

import argparse
import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Fine-tuning with Clean Regularization')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--corruption_finetune', action='store_true', 
                       help='Enable corruption fine-tuning')
    parser.add_argument('--vanilla_checkpoint', type=str, 
                       default='./checkpoints/mobilenetv3_13classes_best_vanilla.pt',
                       help='Path to vanilla model checkpoint')
    parser.add_argument('--save_name', type=str, 
                       default='mobilenetv3_13classes_enhanced_weather_contrastive.pt',
                       help='Save name for fine-tuned model')
    return parser.parse_args()

def load_cfg(cfg_path):
    """Load YAML configuration"""
    print(f"Loading config from: {cfg_path}")
    
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        print("✅ Config loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")
    
    # Set defaults compatible with both scripts
    cfg.setdefault("data", {})
    cfg.setdefault("system", {})
    cfg.setdefault("train", {})
    cfg.setdefault("corruption_finetune", {})
    
    # Data defaults
    d = cfg["data"]
    d.setdefault("root", "./data")
    d.setdefault("img_size", 224)
    d.setdefault("resize", 256)
    d.setdefault("train_frac", 0.7)
    d.setdefault("val_frac", 0.15)
    d.setdefault("test_frac", 0.15)
    d.setdefault("batch_size", 32)
    d.setdefault("num_workers", 4)
    d.setdefault("balance_seed", 42)
    
    # System defaults
    s = cfg["system"]
    s.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    s.setdefault("seed", 42)
    
    # Train defaults
    t = cfg["train"]
    t.setdefault("epochs", 20)
    t.setdefault("lr", 5e-4)
    t.setdefault("weight_decay", 1e-4)
    t.setdefault("freeze_backbone", True)
    t.setdefault("label_smoothing", 0.0)
    t.setdefault("dropout_rate", 0.3)
    t.setdefault("ckpt_dir", "./checkpoints")
    t.setdefault("warmup_epochs", 2)
    
    # Corruption finetune defaults with clean regularization
    cf = cfg["corruption_finetune"]
    cf.setdefault("enabled", True)
    cf.setdefault("lr", 1e-4)
    cf.setdefault("weight_decay", 2e-4)
    cf.setdefault("feature_dim", 256)
    cf.setdefault("clean_reg_weight", 0.4)
    cf.setdefault("contrastive_weight", 0.3)
    cf.setdefault("consistency_weight", 0.2)
    cf.setdefault("stages", 3)
    cf.setdefault("epochs_per_stage", 5)
    
    # Convert types
    for key in ["lr", "weight_decay", "label_smoothing", "dropout_rate"]:
        if key in t:
            t[key] = float(t[key])
    
    for key in ["lr", "weight_decay", "clean_reg_weight", "contrastive_weight", "consistency_weight"]:
        if key in cf:
            cf[key] = float(cf[key])
    
    return cfg

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_num_classes(data_root):
    """Auto-detect number of classes from dataset"""
    try:
        temp_ds = datasets.ImageFolder(data_root)
        num_classes = len(temp_ds.classes)
        class_names = sorted(temp_ds.classes)
        print(f"Detected {num_classes} classes: {class_names}")
        return num_classes, class_names
    except Exception as e:
        print(f"Error detecting classes: {e}")
        return 13, ['unknown']

def stratified_split(dataset, train_frac, val_frac, test_frac, seed=42):
    """Create stratified splits"""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    targets = [dataset.samples[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_frac, stratify=targets, random_state=seed
    )
    
    train_val_targets = [targets[i] for i in train_val_indices]
    val_size = val_frac / (train_frac + val_frac)
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, stratify=train_val_targets, random_state=seed
    )
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

def build_transforms(img_size, resize, is_training=True):
    """Build transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(img_size, padding=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def save_split_manifest(train_ds, val_ds, test_ds, full_ds, cfg):
    """Save split manifest"""
    manifest = {
        "dataset_info": {
            "num_classes": len(full_ds.classes),
            "class_names": sorted(full_ds.classes),
            "total_samples": len(train_ds) + len(val_ds) + len(test_ds)
        },
        "splits": {
            "train": [full_ds.samples[i][0] for i in train_ds.indices],
            "val": [full_ds.samples[i][0] for i in val_ds.indices],
            "test": [full_ds.samples[i][0] for i in test_ds.indices]
        },
        "class_to_idx": full_ds.class_to_idx,
        "split_config": {
            "train_frac": cfg["data"]["train_frac"],
            "val_frac": cfg["data"]["val_frac"], 
            "test_frac": cfg["data"]["test_frac"],
            "seed": cfg["data"]["balance_seed"]
        },
        "created_timestamp": datetime.now().isoformat(),
        "data_root": cfg["data"]["root"]
    }
    
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    manifest_path = os.path.join(cfg["train"]["ckpt_dir"], "split_manifest.json")
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Split manifest saved to: {manifest_path}")
    return manifest_path

class CompatibleMobileNetV3(nn.Module):
    """MobileNetV3 compatible with vanilla checkpoints + contrastive capabilities"""
    
    def __init__(self, num_classes, feature_dim=256, freeze_backbone=True, dropout_rate=0.3):
        super().__init__()
        
        # Load pretrained MobileNetV3 (same as vanilla)
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_large(weights=weights)
        
        # MobileNetV3-Large has 960 features before the final classifier
        backbone_features = 960
        
        # Replace classifier to match vanilla structure
        self.backbone.classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(512, num_classes)
        )
        
        # Additional heads for contrastive learning
        self.contrastive_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feature_dim)
        )
        
        self.consistency_head = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            print("✅ Backbone frozen - only training classifier")
        else:
            print("✅ Full model training - backbone not frozen")
        
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
    
    def forward(self, x):
        """Standard forward pass (compatible with vanilla)"""
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before final classifier"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_contrastive_features(self, x):
        """Get normalized contrastive features"""
        features = self.get_features(x)
        contrastive_features = self.contrastive_head(features)
        return F.normalize(contrastive_features, dim=1)
    
    def get_consistency_features(self, x):
        """Get consistency features"""
        features = self.get_features(x)
        return self.consistency_head(features)

def load_vanilla_checkpoint(model, checkpoint_path, device):
    """Load vanilla model checkpoint into compatible model"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Vanilla checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        print(f"Loading vanilla checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            vanilla_state_dict = checkpoint['model_state_dict']
        else:
            vanilla_state_dict = checkpoint
        
        # Load only the backbone weights (ignore contrastive heads)
        model_state_dict = model.state_dict()
        
        # Create mapping for compatible weights
        compatible_weights = {}
        for key, value in vanilla_state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                compatible_weights[key] = value
        
        # Load compatible weights
        missing_keys, unexpected_keys = model.load_state_dict(compatible_weights, strict=False)
        
        print(f"✅ Loaded vanilla backbone successfully")
        print(f"📝 New heads will be trained from scratch: {len(missing_keys)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading vanilla checkpoint: {e}")
        return False

class WeatherCorruptionDataset(Dataset):
    """Dataset with on-the-fly weather corruption"""
    
    def __init__(self, base_dataset, transform=None, corruption_prob=0.5, 
                 severity_range=(1, 3), enabled_corruptions=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.corruption_prob = corruption_prob
        self.severity_range = severity_range
        
        # All available corruptions
        self.all_corruptions = [
            'gaussian_noise', 'motion_blur', 'brightness', 'contrast',
            'rain', 'snow', 'frost', 'fog', 'low_light', 'high_light', 
            'shadow', 'glare', 'haze', 'mist'
        ]
        
        self.enabled_corruptions = enabled_corruptions or self.all_corruptions
        
        print(f"📝 Corruption dataset: severity {list(severity_range)}, prob {corruption_prob}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if hasattr(self.base_dataset, 'dataset'):
            img_path, label = self.base_dataset.dataset.samples[self.base_dataset.indices[idx]]
        else:
            img_path, label = self.base_dataset.samples[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        # Apply corruption with probability
        if random.random() < self.corruption_prob:
            corruption_type = random.choice(self.enabled_corruptions)
            severity = random.randint(*self.severity_range)
            img = self._apply_corruption(img, corruption_type, severity)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label, img_path
    
    def update_corruption_params(self, severity_range, corruption_prob):
        """Update corruption parameters"""
        self.severity_range = severity_range
        self.corruption_prob = corruption_prob
        print(f"📝 Updated corruption dataset: severity {list(severity_range)}, prob {corruption_prob}")
    
    def _apply_corruption(self, img, corruption_type, severity):
        """Apply specified corruption"""
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

def compute_contrastive_loss(features1, labels1, features2, labels2, temperature=0.1):
    """Compute InfoNCE contrastive loss with numerical stability"""
    # Combine features and labels
    all_features = torch.cat([features1, features2], dim=0)
    all_labels = torch.cat([labels1, labels2], dim=0)
    
    # Normalize features for numerical stability
    all_features = F.normalize(all_features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_features, all_features.T) / temperature
    
    # Create mask for positive pairs (same class)
    label_matrix = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
    
    # Remove self-similarity
    label_matrix.fill_diagonal_(False)
    
    # Check if we have any positive pairs
    pos_pairs = label_matrix.sum(dim=1)
    if pos_pairs.sum() == 0:
        # No positive pairs, return zero loss
        return torch.tensor(0.0, device=all_features.device, requires_grad=True)
    
    # Apply temperature scaling with numerical stability
    max_sim = similarity_matrix.max(dim=1, keepdim=True)[0]
    similarity_matrix = similarity_matrix - max_sim  # Numerical stability
    
    # Compute InfoNCE loss
    exp_sim = torch.exp(similarity_matrix)
    sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
    
    pos_sim = exp_sim * label_matrix.float()
    
    # Add small epsilon to prevent log(0)
    eps = 1e-8
    pos_sum = torch.sum(pos_sim, dim=1) + eps
    
    # Only compute loss for samples that have positive pairs
    valid_mask = pos_pairs > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=all_features.device, requires_grad=True)
    
    loss = -torch.log(pos_sum[valid_mask] / (sum_exp_sim.squeeze()[valid_mask] + eps))
    
    return loss.mean()

def train_epoch_with_clean_regularization(model, corrupted_loader, clean_loader, optimizer, device, cfg):
    """Training epoch with clean data regularization and better debugging"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Create iterator for clean data
    clean_iter = iter(clean_loader)
    cf_cfg = cfg["corruption_finetune"]
    
    for batch_idx, (corrupt_imgs, corrupt_labels, _) in enumerate(corrupted_loader):
        # Get clean batch
        try:
            clean_imgs, clean_labels, _ = next(clean_iter)
        except StopIteration:
            clean_iter = iter(clean_loader)
            clean_imgs, clean_labels, _ = next(clean_iter)
        
        # Move to device
        corrupt_imgs = corrupt_imgs.to(device)
        corrupt_labels = corrupt_labels.to(device)
        clean_imgs = clean_imgs.to(device)
        clean_labels = clean_labels.to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass on corrupted data
            corrupt_logits = model(corrupt_imgs)
            corrupt_contrastive = model.get_contrastive_features(corrupt_imgs)
            corrupt_consistency = model.get_consistency_features(corrupt_imgs)
            
            # Forward pass on clean data
            clean_logits = model(clean_imgs)
            clean_contrastive = model.get_contrastive_features(clean_imgs)
            clean_consistency = model.get_consistency_features(clean_imgs)
            
            # Check for NaN in features
            if torch.isnan(corrupt_logits).any() or torch.isnan(clean_logits).any():
                print(f"NaN detected in logits at batch {batch_idx}")
                continue
            
            # 1. Classification losses
            corrupt_cls_loss = F.cross_entropy(corrupt_logits, corrupt_labels)
            clean_cls_loss = F.cross_entropy(clean_logits, clean_labels)
            
            # 2. Contrastive loss (InfoNCE) - with fallback
            try:
                contrastive_loss = compute_contrastive_loss(
                    corrupt_contrastive, corrupt_labels, 
                    clean_contrastive, clean_labels,
                    temperature=cf_cfg.get("contrastive_temperature", 0.1)
                )
            except Exception as e:
                print(f"Contrastive loss error at batch {batch_idx}: {e}")
                contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 3. Consistency loss
            consistency_loss = F.mse_loss(corrupt_consistency, clean_consistency.detach())
            
            # Check individual losses for NaN
            if torch.isnan(corrupt_cls_loss) or torch.isnan(clean_cls_loss) or \
               torch.isnan(contrastive_loss) or torch.isnan(consistency_loss):
                print(f"NaN in individual losses at batch {batch_idx}")
                print(f"Corrupt cls: {corrupt_cls_loss}, Clean cls: {clean_cls_loss}")
                print(f"Contrastive: {contrastive_loss}, Consistency: {consistency_loss}")
                continue
            
            # 4. Combined loss with clean regularization
            total_loss_batch = (
                corrupt_cls_loss + 
                cf_cfg["clean_regularization_weight"] * clean_cls_loss +
                cf_cfg["contrastive_weight"] * contrastive_loss +
                cf_cfg["consistency_weight"] * consistency_loss
            )
            
            # Check final loss
            if torch.isnan(total_loss_batch):
                print(f"NaN in total loss at batch {batch_idx}")
                continue
            
            total_loss_batch.backward()
            
            # Check gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm):
                print(f"NaN gradients at batch {batch_idx}")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            _, predicted = torch.max(corrupt_logits.data, 1)
            total += corrupt_labels.size(0)
            correct += (predicted == corrupt_labels).sum().item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: Loss {total_loss_batch.item():.4f}, "
                      f"Acc {100.*correct/total:.2f}%, Grad norm: {grad_norm:.4f}")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            optimizer.zero_grad()
            continue
    
    avg_loss = total_loss / len(corrupted_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def corruption_finetune_with_clean_regularization(model, train_ds, val_ds, cfg, device, class_names):
    """Enhanced corruption fine-tuning with clean regularization"""
    
    print("\n" + "="*80)
    print("🚀 Enhanced Corruption Fine-tuning with Clean Data Regularization")
    print("="*80)
    
    cf_cfg = cfg["corruption_finetune"]
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cf_cfg["lr"],
        weight_decay=cf_cfg["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cf_cfg["stages"] * cf_cfg["epochs_per_stage"]
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Progressive training stages
    stages = [
        {"name": "Low Severity", "severity_range": (1, 2), "corruption_prob": 0.3,
         "clean_reg_weight": 0.6, "contrastive_weight": 0.2, "consistency_weight": 0.3},
        {"name": "Medium Severity", "severity_range": (2, 4), "corruption_prob": 0.5,
         "clean_reg_weight": 0.4, "contrastive_weight": 0.3, "consistency_weight": 0.2},
        {"name": "High Severity", "severity_range": (3, 5), "corruption_prob": 0.7,
         "clean_reg_weight": 0.3, "contrastive_weight": 0.4, "consistency_weight": 0.1}
    ]
    
    best_val_acc = 0
    global_epoch = 0
    
    print(f"Fine-tuning Configuration:")
    print(f"  Learning Rate: {cf_cfg['lr']}")
    print(f"  Weight Decay: {cf_cfg['weight_decay']}")
    print(f"  Stages: {cf_cfg['stages']}")
    print(f"  Epochs per Stage: {cf_cfg['epochs_per_stage']}")
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*80}")
        print(f"STAGE {stage_idx + 1}: {stage['name']}")
        print(f"  Severity Range: {list(stage['severity_range'])}")
        print(f"  Corruption Probability: {stage['corruption_prob']}")
        print(f"  Clean Regularization Weight: {stage['clean_reg_weight']}")
        print(f"  Contrastive Weight: {stage['contrastive_weight']}")
        print(f"  Consistency Weight: {stage['consistency_weight']}")
        print("="*80)
        
        # Update config for this stage
        cfg["corruption_finetune"]["clean_reg_weight"] = stage["clean_reg_weight"]
        cfg["corruption_finetune"]["contrastive_weight"] = stage["contrastive_weight"]
        cfg["corruption_finetune"]["consistency_weight"] = stage["consistency_weight"]
        
        # Create corrupted training dataset
        corrupted_transform = build_transforms(
            cfg["data"]["img_size"], 
            cfg["data"]["resize"], 
            is_training=True
        )
        
        corrupted_train_ds = WeatherCorruptionDataset(
            train_ds, 
            transform=corrupted_transform,
            corruption_prob=stage["corruption_prob"], 
            severity_range=stage["severity_range"]
        )
        
        # Clean training dataset (no corruption)
        clean_train_ds = WeatherCorruptionDataset(
            train_ds, 
            transform=corrupted_transform,
            corruption_prob=0.0,  # No corruption for clean data
            severity_range=(1, 1)
        )
        
        # Create data loaders
        corrupted_train_dl = DataLoader(
            corrupted_train_ds, 
            batch_size=cfg["data"]["batch_size"], 
            shuffle=True, 
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True
        )
        
        clean_train_dl = DataLoader(
            clean_train_ds, 
            batch_size=cfg["data"]["batch_size"], 
            shuffle=True, 
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataset (clean)
        val_transform = build_transforms(
            cfg["data"]["img_size"], 
            cfg["data"]["resize"], 
            is_training=False
        )
        val_ds.dataset.transform = val_transform
        val_dl = DataLoader(
            val_ds, 
            batch_size=cfg["data"]["batch_size"], 
            shuffle=False, 
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True
        )
        
        # Training loop for this stage
        for epoch in range(cf_cfg["epochs_per_stage"]):
            global_epoch += 1
            print(f"\nStage {stage_idx + 1}, Epoch {epoch + 1}/{cf_cfg['epochs_per_stage']} (Global: {global_epoch})")
            
            # Train with clean regularization
            train_loss, train_acc = train_epoch_with_clean_regularization(
                model, corrupted_train_dl, clean_train_dl, optimizer, device, cfg
            )
            
            # Validate
            val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
            
            # Update learning rate
            scheduler.step()
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                checkpoint = {
                    'epoch': global_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': model.num_classes,
                    'class_names': class_names,
                    'config': cfg,
                    'model_type': 'enhanced_clean_regularized',
                    'timestamp': datetime.now().isoformat()
                }
                
                save_path = os.path.join(cfg["train"]["ckpt_dir"], args.save_name)
                torch.save(checkpoint, save_path)
                print(f"✅ New best model saved: {val_acc:.2f}%")
    
    print(f"\n🎉 Clean regularization fine-tuning completed!")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
    
    return best_val_acc

def main():
    global args
    print("🚀 Enhanced Corruption Fine-tuning with Clean Data Regularization")
    args = parse_args()
    
    cfg = load_cfg(args.config)
    set_seed(cfg["system"]["seed"])
    device = torch.device(cfg["system"]["device"])
    
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Config: {args.config}")
    
    # Auto-detect dataset
    num_classes, class_names = get_num_classes(cfg["data"]["root"])
    
    # Load dataset
    full_ds = datasets.ImageFolder(cfg["data"]["root"], transform=None)
    train_ds, val_ds, test_ds = stratified_split(
        full_ds, 
        cfg["data"]["train_frac"], 
        cfg["data"]["val_frac"], 
        cfg["data"]["test_frac"], 
        cfg["data"]["balance_seed"]
    )
    
    print(f"Dataset splits - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Save manifest
    manifest_path = save_split_manifest(train_ds, val_ds, test_ds, full_ds, cfg)
    
    # Create compatible model
    model = CompatibleMobileNetV3(
        num_classes=num_classes,
        feature_dim=cfg["corruption_finetune"]["feature_dim"],
        freeze_backbone=cfg["train"]["freeze_backbone"],
        dropout_rate=cfg["train"]["dropout_rate"]
    )
    model.to(device)
    
    print(f"Model: CompatibleMobileNetV3 with Clean Regularization")
    print(f"Classes: {num_classes}")
    print(f"Backbone frozen: {cfg['train']['freeze_backbone']}")
    
    # Load vanilla checkpoint
    if load_vanilla_checkpoint(model, args.vanilla_checkpoint, device):
        print("✅ Successfully loaded vanilla model for fine-tuning")
    else:
        print("⚠️  Starting fine-tuning from pretrained ImageNet weights")
    
    # Run corruption fine-tuning with clean regularization
    if args.corruption_finetune:
        final_acc = corruption_finetune_with_clean_regularization(
            model, train_ds, val_ds, cfg, device, class_names
        )
        print(f"\n✅ Fine-tuning completed! Final accuracy: {final_acc:.2f}%")
        print(f"💾 Enhanced model saved as: {args.save_name}")
    else:
        print("Use --corruption_finetune flag to start fine-tuning")

if __name__ == "__main__":
    main()