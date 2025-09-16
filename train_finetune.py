#!/usr/bin/env python3
# filepath: /home/rahim/exp/tta-pncache-tta/train_finetune.py
"""
Updated training script with automatic class detection and contrastive learning
"""
import argparse, os, json, yaml, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import functional as TF
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import random

def load_cfg(cfg_path):
    """Load YAML configuration"""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set defaults with proper type conversion
    cfg.setdefault("data", {})
    cfg.setdefault("system", {})
    cfg.setdefault("train", {})
    
    # Data defaults
    d = cfg["data"]
    d.setdefault("root", "./pests_data")
    d.setdefault("img_size", 224)
    d.setdefault("resize", 256)
    d.setdefault("train_frac", 0.7)
    d.setdefault("val_frac", 0.15)
    d.setdefault("test_frac", 0.15)
    d.setdefault("batch_size", 64)
    d.setdefault("num_workers", 4)
    d.setdefault("balance_seed", 42)
    
    # System defaults
    s = cfg["system"]
    s.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    s.setdefault("seed", 42)
    s.setdefault("fp16", False)
    # Handle empty device string
    if not s["device"]:
        s["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Train defaults with type conversion
    t = cfg["train"]
    t.setdefault("epochs", 10)
    t.setdefault("lr", 5e-4)
    t.setdefault("weight_decay", 1e-4)
    t.setdefault("freeze_backbone", True)
    t.setdefault("label_smoothing", 0.0)
    t.setdefault("ckpt_dir", "./checkpoints")
    t.setdefault("ckpt_name", "mobilenetv3_best.pt")
    t.setdefault("warmup_epochs", 1)
    
    # Convert string values to proper types
    t["lr"] = float(t["lr"])
    t["weight_decay"] = float(t["weight_decay"])
    t["label_smoothing"] = float(t["label_smoothing"])
    
    # Corruption finetune defaults
    if "corruption_finetune" in cfg:
        cf = cfg["corruption_finetune"]
        if "lr" in cf:
            cf["lr"] = float(cf["lr"])
        if "weight_decay" in cf:
            cf["weight_decay"] = float(cf["weight_decay"])
        if "contrastive_temperature" in cf:
            cf["contrastive_temperature"] = float(cf["contrastive_temperature"])
        if "contrastive_weight" in cf:
            cf["contrastive_weight"] = float(cf["contrastive_weight"])
    
    return cfg

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_num_classes(data_root):
    """Automatically detect number of classes from dataset structure"""
    full_ds = datasets.ImageFolder(data_root, transform=None)
    return len(full_ds.classes), full_ds.classes

def stratified_split(dataset, train_frac, val_frac, test_frac, seed=42):
    """Stratified split maintaining class distribution"""
    targets = [dataset.targets[i] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_frac, stratify=targets, random_state=seed
    )
    
    # Second split: separate train and validation
    train_val_targets = [targets[i] for i in train_val_indices]
    val_size = val_frac / (train_frac + val_frac)
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, stratify=train_val_targets, random_state=seed
    )
    
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)
    
    return train_ds, val_ds, test_ds

def build_transforms(img_size, resize, corruption_mode=False):
    """Build data transforms"""
    if corruption_mode:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(img_size),
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

class MobileNetV3WithContrastive(nn.Module):
    """MobileNetV3 with simple contrastive learning"""
    def __init__(self, num_classes, feature_dim=128, freeze_backbone=True, dropout_rate=0.3):
        super().__init__()
        
        # Load pretrained backbone
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_large(weights=weights)
        
        # Simple classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(960, num_classes)
        )
        
        # Simple contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        print(f"Model initialized with backbone feature dim: 960")
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
    
    def get_features(self, x):
        """Extract backbone features"""
        features = self.backbone.features(x)
        return self.backbone.avgpool(features).flatten(1)
    
    def get_contrastive_features(self, x):
        """Get features for contrastive learning"""
        backbone_features = self.get_features(x)
        return F.normalize(self.contrastive_head(backbone_features), dim=1)
    
    def forward(self, x):
        """Forward pass for classification"""
        features = self.get_features(x)
        return self.backbone.classifier(features)

class ContrastiveCorruptionDataset(Dataset):
    """Dataset that applies corruptions for contrastive learning - NumPy 2.0 compatible"""
    def __init__(self, base_dataset, cfg, transform=None):
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.transform = transform
        
        # Use ALL corruption types from config including weather/lighting
        self.corruption_types = cfg.get("corruption_types", [
            "gaussian_noise", "motion_blur", "brightness", "contrast",
            "rain", "snow", "frost", "fog", "low_light", "high_light", 
            "shadow", "glare", "haze", "mist"
        ])
        
        print(f"ContrastiveCorruptionDataset initialized with {len(self.corruption_types)} corruption types:")
        print(f"  {self.corruption_types}")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original image and label
        if hasattr(self.base_dataset, 'dataset'):
            img_path, label = self.base_dataset.dataset.samples[self.base_dataset.indices[idx]]
        else:
            img_path, label = self.base_dataset.samples[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        # Create corrupted version
        corrupted_img = img.copy()
        
        # Apply random corruption
        if np.random.rand() < self.cfg.get("corruption_prob", 0.7):
            corruption_type = np.random.choice(self.corruption_types)
            severity = np.random.randint(
                self.cfg.get("severity_range", [1, 3])[0],
                self.cfg.get("severity_range", [1, 3])[1] + 1
            )
            corrupted_img = self.apply_corruption(corrupted_img, corruption_type, severity)
        
        # Apply transforms
        if self.transform:
            clean_img = self.transform(img)
            corrupted_img = self.transform(corrupted_img)
        
        return clean_img, corrupted_img, label
    
    def apply_corruption(self, img, corruption_type, severity):
        """Apply specific corruption to PIL image - NumPy 2.0 compatible"""
        if corruption_type == 'gaussian_noise':
            return self._add_gaussian_noise(img, severity)
        elif corruption_type == 'motion_blur':
            return self._add_motion_blur(img, severity)
        elif corruption_type == 'brightness':
            return self._adjust_brightness(img, severity)
        elif corruption_type == 'contrast':
            return self._adjust_contrast(img, severity)
        elif corruption_type == 'rain':
            return self._add_rain_pil(img, severity)
        elif corruption_type == 'snow':
            return self._add_snow_pil(img, severity)
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
            return self._add_glare_pil(img, severity)
        elif corruption_type == 'haze':
            return self._add_haze(img, severity)
        elif corruption_type == 'mist':
            return self._add_mist(img, severity)
        else:
            return img
    
    def _add_gaussian_noise(self, img, severity):
        """Add Gaussian noise to PIL image"""
        tensor = TF.to_tensor(img)
        noise_std = 0.02 + (severity / 5.0) * 0.08
        noise = torch.randn_like(tensor) * noise_std
        noisy_tensor = torch.clamp(tensor + noise, 0, 1)
        return TF.to_pil_image(noisy_tensor)
    
    def _add_motion_blur(self, img, severity):
        """Add motion blur"""
        kernel_size = min(15, 3 + 2 * severity)
        sigma = max(0.5, severity * 0.5)
        return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
    
    def _adjust_brightness(self, img, severity):
        """Adjust brightness"""
        factor = 0.5 + (severity / 5.0)
        return TF.adjust_brightness(img, brightness_factor=factor)
    
    def _adjust_contrast(self, img, severity):
        """Adjust contrast"""
        factor = 0.3 + (severity / 5.0) * 1.4
        return TF.adjust_contrast(img, contrast_factor=factor)
    
    def _add_rain_pil(self, img, severity):
        """Add rain effect using PIL only"""
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img.size
        
        # Number of rain drops based on severity
        num_drops = int(30 * severity)
        
        for _ in range(num_drops):
            # Random rain drop position and length
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            length = random.randint(10, 25)
            
            x2 = x1 + random.randint(-3, 3)
            y2 = min(height - 1, y1 + length)
            
            # Draw rain line
            try:
                draw.line([(x1, y1), (x2, y2)], fill=(200, 200, 255), width=1)
            except:
                pass
        
        return img_copy
    
    def _add_snow_pil(self, img, severity):
        """Add snow effect using PIL only"""
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img.size
        
        # Snow density based on severity
        snow_density = int(50 * severity)
        
        for _ in range(snow_density):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 3)
            
            try:
                # Draw snow as small circles
                draw.ellipse([x-size, y-size, x+size, y+size], fill=(255, 255, 255))
            except:
                pass
        
        return img_copy
    
    def _add_frost(self, img, severity):
        """Add frost effect"""
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.3 + 0.1 * severity)  # Desaturate
        
        # Add slight blue tint by blending
        frost_overlay = Image.new('RGB', img.size, (150, 150, 255))
        frost_alpha = int(30 * severity)
        
        # Create alpha mask
        alpha_mask = Image.new('L', img.size, frost_alpha)
        
        # Blend with frost overlay
        return Image.composite(frost_overlay, img, alpha_mask)
    
    def _add_fog(self, img, severity):
        """Add fog effect"""
        # Create fog overlay
        fog_intensity = int(50 + 30 * severity)
        fog_overlay = Image.new('RGB', img.size, (200, 200, 200))
        alpha_mask = Image.new('L', img.size, fog_intensity)
        
        # Blend with original image
        return Image.composite(fog_overlay, img, alpha_mask)
    
    def _adjust_low_light(self, img, severity):
        """Simulate low light conditions"""
        brightness_factor = 1.0 - (severity / 5.0) * 0.7  # Reduce brightness
        contrast_factor = 0.8 - (severity / 5.0) * 0.3     # Reduce contrast
        
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        return img
    
    def _adjust_high_light(self, img, severity):
        """Simulate high light/overexposure"""
        brightness_factor = 1.0 + (severity / 5.0) * 0.8  # Increase brightness
        img = TF.adjust_brightness(img, brightness_factor)
        
        # Add slight overexposure effect
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + severity * 0.1)
        return img
    
    def _add_shadow(self, img, severity):
        """Add shadow effect using PIL blending"""
        shadow_strength = int(40 + severity * 20)
        
        # Create shadow overlay
        if random.random() > 0.5:
            # Vertical shadow - create gradient
            shadow_overlay = Image.new('RGB', img.size, (0, 0, 0))
        else:
            # Horizontal shadow
            shadow_overlay = Image.new('RGB', img.size, (0, 0, 0))
        
        alpha_mask = Image.new('L', img.size, shadow_strength)
        
        # Blend shadow with image
        return Image.composite(shadow_overlay, img, alpha_mask)
    
    def _add_glare_pil(self, img, severity):
        """Add glare effect using PIL"""
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img.size
        
        # Random glare position
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)
        radius = 15 + severity * 8
        
        # Draw glare as bright circle
        try:
            glare_color = (255, 255, min(255, 200 + severity * 10))
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], 
                        fill=glare_color)
        except:
            pass
        
        return img_copy
    
    def _add_haze(self, img, severity):
        """Add haze effect"""
        # Create haze overlay (slightly warm/yellowish)
        haze_intensity = int(30 + 10 * severity)
        haze_overlay = Image.new('RGB', img.size, (220, 220, 200))
        alpha_mask = Image.new('L', img.size, haze_intensity)
        
        # Blend with original image
        return Image.composite(haze_overlay, img, alpha_mask)
    
    def _add_mist(self, img, severity):
        """Add mist effect"""
        # Create mist overlay (cooler than haze)
        mist_intensity = int(40 + 10 * severity)
        mist_overlay = Image.new('RGB', img.size, (200, 210, 220))
        alpha_mask = Image.new('L', img.size, mist_intensity)
        
        # Blend with original image
        return Image.composite(mist_overlay, img, alpha_mask)

def simple_contrastive_loss(z_clean, z_corrupted, temperature=0.07):
    """Simple contrastive loss without any regularization"""
    batch_size = z_clean.shape[0]
    
    # Normalize features
    z_clean = F.normalize(z_clean, dim=1)
    z_corrupted = F.normalize(z_corrupted, dim=1)
    
    # Create positive pairs: clean[i] should be similar to corrupted[i]
    features = torch.cat([z_clean, z_corrupted], dim=0)
    sim_matrix = torch.mm(features, features.t()) / temperature
    
    # Create labels for positive pairs
    contrastive_labels = torch.cat([
        torch.arange(batch_size, 2*batch_size), 
        torch.arange(0, batch_size)
    ]).to(z_clean.device)
    
    # Remove self-similarities
    mask = torch.eye(2*batch_size, dtype=torch.bool).to(z_clean.device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # Standard contrastive loss
    return F.cross_entropy(sim_matrix, contrastive_labels)

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """Standard training epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f"Train {epoch+1}/{total_epochs}")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

def train_epoch_contrastive_simple(model, dataloader, optimizer, criterion, device, epoch, total_epochs, cfg):
    """Simple contrastive training with weather/lighting corruptions"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    contrastive_weight = cfg.get("contrastive_weight", 0.3)
    contrastive_temperature = cfg.get("contrastive_temperature", 0.07)
    
    progress_bar = tqdm(dataloader, desc=f"Enhanced Contrastive Train {epoch}/{total_epochs}")
    
    for batch_idx, (clean_imgs, corrupted_imgs, labels) in enumerate(progress_bar):
        clean_imgs = clean_imgs.to(device)
        corrupted_imgs = corrupted_imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Classification loss on clean images
        clean_logits = model(clean_imgs)
        classification_loss = criterion(clean_logits, labels)
        total_loss_val = classification_loss
        
        # Simple contrastive learning (only if we have enough samples)
        contrastive_loss_val = torch.tensor(0.0, device=device)
        if clean_imgs.size(0) >= 2:
            try:
                # Get contrastive features for all samples
                clean_features = model.get_contrastive_features(clean_imgs)
                corrupted_features = model.get_contrastive_features(corrupted_imgs)
                
                # Simple contrastive loss
                contrastive_loss_val = simple_contrastive_loss(
                    clean_features, corrupted_features, contrastive_temperature
                )
                total_loss_val = total_loss_val + contrastive_weight * contrastive_loss_val
                
            except Exception as e:
                print(f"Warning: Contrastive learning skipped for batch {batch_idx}: {e}")
                contrastive_loss_val = torch.tensor(0.0, device=device)
        
        total_loss_val.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += total_loss_val.item()
        _, predicted = clean_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{total_loss_val.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Cls': f'{classification_loss.item():.4f}',
            'Contr': f'{contrastive_loss_val.item():.4f}' if isinstance(contrastive_loss_val, torch.Tensor) else '0.0000'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(dataloader, desc="Validation"):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

def build_model(num_classes, freeze_backbone, contrastive_enabled, feature_dim, dropout_rate=0.3):
    """Build model based on configuration"""
    if contrastive_enabled:
        return MobileNetV3WithContrastive(
            num_classes=num_classes,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    else:
        # Standard vanilla MobileNet
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights)
        
        # Simple classifier
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(960, num_classes)
        )
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        
        return model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MobileNetV3 with optional contrastive learning")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--corruption_finetune", action="store_true", help="Enable corruption fine-tuning")
    return parser.parse_args()

def save_split_manifest(train_ds, val_ds, test_ds, full_ds, cfg):
    """Save dataset split manifest for evaluation"""
    # Extract paths and labels
    def get_paths_and_labels(subset):
        paths = []
        labels = []
        for idx in subset.indices:
            path, label = subset.dataset.samples[idx]
            paths.append(path)
            labels.append(label)
        return paths, labels
    
    train_paths, train_labels = get_paths_and_labels(train_ds)
    val_paths, val_labels = get_paths_and_labels(val_ds)
    test_paths, test_labels = get_paths_and_labels(test_ds)
    
    manifest = {
        'dataset_info': {
            'root': cfg['data']['root'],
            'total_samples': len(full_ds),
            'num_classes': len(full_ds.classes),
            'class_names': full_ds.classes,
            'created_at': datetime.now().isoformat()
        },
        'class_to_idx': full_ds.class_to_idx,
        'splits': {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        },
        'split_config': {
            'train_frac': cfg['data']['train_frac'],
            'val_frac': cfg['data']['val_frac'],
            'test_frac': cfg['data']['test_frac'],
            'balance_seed': cfg['data']['balance_seed']
        }
    }
    
    # Save manifest
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    manifest_path = os.path.join(cfg["train"]["ckpt_dir"], "split_manifest.json")
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Split manifest saved to: {manifest_path}")
    return manifest_path

def corruption_finetune_simple(model, train_ds, val_dl, cfg, device, class_to_idx, manifest_path):
    """Simple corruption fine-tuning with weather/lighting corruptions"""
    
    corruption_cfg = cfg["corruption_finetune"]
    
    print(f"Enhanced corruption fine-tuning configuration:")
    print(f"  Epochs: {corruption_cfg['epochs']}")
    print(f"  Learning rate: {corruption_cfg['lr']}")
    print(f"  Contrastive weight: {corruption_cfg.get('contrastive_weight', 0.3)}")
    print(f"  Corruption types: {len(corruption_cfg.get('corruption_types', []))}")
    print(f"  Weather/Lighting corruptions included: {any(c in corruption_cfg.get('corruption_types', []) for c in ['rain', 'snow', 'fog', 'haze'])}")
    
    # Create enhanced corruption dataset with weather/lighting effects
    corruption_ds = ContrastiveCorruptionDataset(
        train_ds, 
        corruption_cfg, 
        transform=train_ds.dataset.transform
    )
    
    # Data loader for corruption fine-tuning - reduce num_workers to avoid multiprocessing issues
    effective_batch_size = max(4, cfg["data"]["batch_size"])
    corruption_dl = DataLoader(
        corruption_ds, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid NumPy multiprocessing issues
        drop_last=True,
        pin_memory=False
    )
    
    print(f"Enhanced corruption fine-tuning with {len(corruption_dl)} batches")
    
    # Setup optimizer and scheduler for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=corruption_cfg["lr"], 
        weight_decay=corruption_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, corruption_cfg["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    best_corruption_val_acc = 0
    
    for epoch in range(corruption_cfg["epochs"]):
        print(f"\nCorruption Fine-tune Epoch {epoch+1}/{corruption_cfg['epochs']}")
        print("-" * 60)
        
        # Enhanced contrastive training with weather/lighting corruptions
        train_loss, train_acc = train_epoch_contrastive_simple(
            model, corruption_dl, optimizer, criterion, device, 
            epoch+1, corruption_cfg["epochs"], corruption_cfg
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        scheduler.step()
        
        print(f"\nCorruption Fine-tune Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best corruption fine-tuned model
        if val_acc > best_corruption_val_acc:
            best_corruption_val_acc = val_acc
            
            # Enhanced checkpoint for corruption fine-tuned model
            corruption_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'best_corruption_val_acc': best_corruption_val_acc,
                'num_classes': len(class_to_idx),
                'class_names': sorted(class_to_idx.keys()),
                'class_to_idx': class_to_idx,
                'model_type': 'enhanced_contrastive',
                'contrastive_enabled': True,
                'corruption_config': corruption_cfg,
                'weather_corruptions_enabled': True,
                'config': cfg,
                'manifest_path': manifest_path
            }
            
            ckpt_name = corruption_cfg.get("ckpt_name", "mobilenetv3_enhanced_weather_contrastive.pt")
            corruption_ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], ckpt_name)
            
            torch.save(corruption_checkpoint, corruption_ckpt_path)
            print(f"  ✓ Best enhanced corruption model saved: {corruption_ckpt_path}")
            print(f"  ✓ Best corruption validation accuracy: {best_corruption_val_acc:.2f}%")
            
            # Update manifest with corruption model info
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                manifest['corruption_checkpoint'] = corruption_ckpt_path
                manifest['corruption_finetuned'] = True
                manifest['weather_corruptions_enabled'] = True
                manifest['best_corruption_acc'] = best_corruption_val_acc
                
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                    
            except Exception as e:
                print(f"Warning: Could not update manifest: {e}")
    
    print(f"\nEnhanced corruption fine-tuning completed!")
    print(f"Best corruption validation accuracy: {best_corruption_val_acc:.2f}%")
    print(f"Weather/Lighting corruptions included: rain, snow, frost, fog, low_light, high_light, shadow, glare, haze, mist")
    
    return best_corruption_val_acc

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["system"]["seed"])
    device = torch.device(cfg["system"]["device"])
    
    # Auto-detect number of classes
    num_classes, class_names = get_num_classes(cfg["data"]["root"])
    print(f"Detected {num_classes} classes: {class_names}")
    print(f"Training with {num_classes} classes: {class_names}")
    
    # Build transforms
    train_transform = build_transforms(
        cfg["data"]["img_size"], 
        cfg["data"]["resize"], 
        corruption_mode=True
    )
    val_transform = build_transforms(
        cfg["data"]["img_size"], 
        cfg["data"]["resize"], 
        corruption_mode=False
    )
    
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
    
    # Save manifest for evaluation scripts
    manifest_path = save_split_manifest(train_ds, val_ds, test_ds, full_ds, cfg)
    
    # Build model
    contrastive_enabled = (args.corruption_finetune or 
                          cfg.get("corruption_finetune", {}).get("contrastive_enabled", False))
    
    dropout_rate = cfg.get("train", {}).get("dropout_rate", 0.3)
    feature_dim = cfg.get("corruption_finetune", {}).get("feature_dim", 128)
    
    model = build_model(
        num_classes, 
        cfg["train"]["freeze_backbone"], 
        contrastive_enabled,
        feature_dim,
        dropout_rate
    )
    model.to(device)
    
    model_type = "enhanced_contrastive" if contrastive_enabled else "vanilla"
    print(f"Model type: {model_type}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Dropout rate: {dropout_rate}")
    
    # Create data loaders with proper batch size handling
    effective_batch_size = max(4, cfg["data"]["batch_size"])
    print(f"Using effective batch size: {effective_batch_size}")
    
    # Apply transforms to datasets
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform
    
    # Create data loaders - reduce num_workers to avoid NumPy issues
    train_dl = DataLoader(
        train_ds, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid NumPy multiprocessing issues
        drop_last=True,
        pin_memory=False
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=effective_batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid NumPy multiprocessing issues
        pin_memory=False
    )
    
    print(f"Training batches: {len(train_dl)}, Validation batches: {len(val_dl)}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["train"]["lr"], 
        weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["train"]["epochs"])
    
    # Training metrics tracking
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Initial training phase
    print(f"\n{'='*50}")
    print(f"Starting initial training for {cfg['train']['epochs']} epochs")
    print(f"{'='*50}")
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        print("-" * 50)
        
        # Training phase
        if contrastive_enabled:
            # Create enhanced contrastive corruption dataset with weather effects
            corruption_cfg = cfg.get("corruption_finetune", {})
            corruption_ds = ContrastiveCorruptionDataset(
                train_ds, 
                corruption_cfg, 
                transform=train_transform
            )
            
            corruption_dl = DataLoader(
                corruption_ds, 
                batch_size=effective_batch_size, 
                shuffle=True, 
                num_workers=0,  # Set to 0 to avoid NumPy multiprocessing issues
                drop_last=True,
                pin_memory=False
            )
            
            print(f"Using enhanced contrastive training with {len(corruption_dl)} batches")
            train_loss, train_acc = train_epoch_contrastive_simple(
                model, corruption_dl, optimizer, criterion, device, 
                epoch+1, cfg["train"]["epochs"], corruption_cfg
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_dl, optimizer, criterion, device, epoch, cfg["train"]["epochs"]
            )
        
        # Validation phase
        print("Running validation...")
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
            
            # Create comprehensive checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'class_names': class_names,
                'class_to_idx': full_ds.class_to_idx,
                'model_type': model_type,
                'contrastive_enabled': contrastive_enabled,
                'weather_corruptions_enabled': contrastive_enabled,
                'feature_dim': feature_dim,
                'dropout_rate': dropout_rate,
                'cfg': cfg,
                'manifest_path': manifest_path,
                'training_metrics': {
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'val_losses': val_losses,
                    'val_accs': val_accs
                }
            }
            
            # Save with descriptive name
            base_name = cfg["train"]["ckpt_name"].replace(".pt", f"_{model_type}.pt")
            checkpoint_path = os.path.join(cfg["train"]["ckpt_dir"], base_name)
            
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ New best model saved: {checkpoint_path}")
            print(f"  ✓ Best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Enhanced corruption fine-tuning phase
    if args.corruption_finetune or cfg.get("corruption_finetune", {}).get("enabled", False):
        print(f"\n{'='*70}")
        print("Starting Enhanced Corruption Fine-tuning Phase")
        print("Including Weather & Lighting Corruptions")
        print(f"{'='*70}")
        
        # Load best model for fine-tuning
        base_name = cfg["train"]["ckpt_name"].replace(".pt", f"_{model_type}.pt")
        checkpoint_path = os.path.join(cfg["train"]["ckpt_dir"], base_name)
        
        if os.path.exists(checkpoint_path):
            print(f"Loading best model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Enhanced corruption fine-tuning with weather/lighting effects
        corruption_finetune_simple(model, train_ds, val_dl, cfg, device, full_ds.class_to_idx, manifest_path)
    
    # Final summary
    print(f"\n{'='*70}")
    print("Enhanced Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Model type: {model_type}")
    print(f"Contrastive enabled: {contrastive_enabled}")
    print(f"Weather/Lighting corruptions: {contrastive_enabled}")
    if contrastive_enabled:
        corruption_types = cfg.get("corruption_finetune", {}).get("corruption_types", [])
        weather_types = [c for c in corruption_types if c in ['rain', 'snow', 'frost', 'fog', 'low_light', 'high_light', 'shadow', 'glare', 'haze', 'mist']]
        print(f"Weather corruption types: {weather_types}")

if __name__ == "__main__":
    main()