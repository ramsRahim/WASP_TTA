#!/usr/bin/env python3
import argparse, os, json, random, yaml
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--corruption_finetune", action="store_true", 
                   help="Enable corruption-aware fine-tuning")
    p.add_argument("--base_model", type=str, default=None,
                   help="Path to base model for fine-tuning")
    return p.parse_args()

def load_cfg(path):
    with open(path, "r") as f: 
        cfg = yaml.safe_load(f)
    
    cfg.setdefault("data", {})
    cfg.setdefault("system", {})
    cfg.setdefault("train", {})
    cfg.setdefault("corruption_finetune", {})
    
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
    
    # Training defaults with proper type conversion
    t = cfg["train"]
    t.setdefault("epochs", 10)
    t.setdefault("lr", 5e-4)
    t.setdefault("weight_decay", 1e-4)
    t.setdefault("freeze_backbone", True)
    t.setdefault("label_smoothing", 0.0)
    t.setdefault("ckpt_dir", "./checkpoints")
    t.setdefault("ckpt_name", "mobilenetv3_best.pt")
    t.setdefault("warmup_epochs", 1)
    
    # Ensure numeric types for training parameters
    t["lr"] = float(t["lr"])
    t["weight_decay"] = float(t["weight_decay"])
    t["label_smoothing"] = float(t["label_smoothing"])
    t["epochs"] = int(t["epochs"])
    t["warmup_epochs"] = int(t["warmup_epochs"])
    
    # Corruption fine-tuning defaults with proper type conversion
    cf = cfg["corruption_finetune"]
    cf.setdefault("enabled", False)
    cf.setdefault("corruption_prob", 0.7)
    cf.setdefault("severity_range", [1, 3])
    cf.setdefault("epochs", 5)
    cf.setdefault("lr", 1e-4)
    cf.setdefault("weight_decay", 1e-4)
    cf.setdefault("corruption_types", ["gaussian_noise", "motion_blur", "brightness", "contrast"])
    cf.setdefault("clean_prob", 0.3)
    cf.setdefault("ckpt_name", "mobilenetv3_corruption_finetuned.pt")
    cf.setdefault("warmup_epochs", 1)
    
    # Contrastive learning parameters
    cf.setdefault("contrastive_enabled", True)
    cf.setdefault("contrastive_temperature", 0.07)
    cf.setdefault("contrastive_weight", 0.5)
    cf.setdefault("feature_dim", 128)
    cf.setdefault("contrastive_pairs_per_batch", 0.8)
    
    # Ensure numeric types for corruption fine-tuning parameters
    cf["corruption_prob"] = float(cf["corruption_prob"])
    cf["lr"] = float(cf["lr"])
    cf["weight_decay"] = float(cf["weight_decay"])
    cf["clean_prob"] = float(cf["clean_prob"])
    cf["epochs"] = int(cf["epochs"])
    cf["warmup_epochs"] = int(cf["warmup_epochs"])
    cf["contrastive_temperature"] = float(cf["contrastive_temperature"])
    cf["contrastive_weight"] = float(cf["contrastive_weight"])
    cf["feature_dim"] = int(cf["feature_dim"])
    cf["contrastive_pairs_per_batch"] = float(cf["contrastive_pairs_per_batch"])
    
    # Ensure severity_range is list of ints
    if isinstance(cf["severity_range"], list):
        cf["severity_range"] = [int(x) for x in cf["severity_range"]]
    
    return cfg

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)

class MobileNetV3WithContrastive(nn.Module):
    """MobileNetV3 with proper feature extraction for contrastive learning"""
    def __init__(self, num_classes, feature_dim=128, freeze_backbone=True):
        super().__init__()
        
        # Load base model
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_large(weights=weights)
        
        # Get the feature dimension before classifier
        self.backbone_feature_dim = self.backbone.classifier[0].in_features  # Should be 960
        
        # Replace classifier
        self.backbone.classifier = nn.Linear(self.backbone_feature_dim, num_classes)
        
        # Add projection head for contrastive learning
        self.projection_head = ContrastiveProjectionHead(
            self.backbone_feature_dim, 
            hidden_dim=512, 
            output_dim=feature_dim
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        print(f"Model initialized with backbone feature dim: {self.backbone_feature_dim}")
    
    def forward(self, x, return_features=False):
        # Extract features using the backbone's feature extractor
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)  # Shape: [batch_size, 960]
        
        # Classification logits
        logits = self.backbone.classifier(features)
        
        if return_features:
            return logits, features
        else:
            return logits
    
    def get_contrastive_features(self, x):
        """Get features for contrastive learning"""
        with torch.no_grad():
            features = self.backbone.features(x)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
        
        # Project to contrastive space
        projected = self.projection_head(features)
        return projected

class ContrastiveCorruptionDataset(Dataset):
    """Dataset that creates clean-corrupted pairs for contrastive learning"""
    
    def __init__(self, base_dataset, corruption_config, transform=None, contrastive_enabled=True):
        self.base_dataset = base_dataset
        self.corruption_config = corruption_config
        self.transform = transform
        self.contrastive_enabled = contrastive_enabled
        
        # Extract corruption parameters
        self.corruption_prob = corruption_config.get('corruption_prob', 0.7)
        self.severity_range = corruption_config.get('severity_range', [1, 3])
        self.corruption_types = corruption_config.get('corruption_types', 
                                                    ["gaussian_noise", "motion_blur", "brightness", "contrast"])
        self.clean_prob = corruption_config.get('clean_prob', 0.3)
        self.contrastive_pairs_per_batch = corruption_config.get('contrastive_pairs_per_batch', 0.8)
        
        print(f"Contrastive Corruption Dataset: {len(self.corruption_types)} types, "
              f"prob={self.corruption_prob}, severity={self.severity_range}")
        print(f"Contrastive learning: {'enabled' if contrastive_enabled else 'disabled'}")
    
    def apply_corruption(self, img, corruption_type, severity):
        """Apply specific corruption to PIL image"""
        if corruption_type == 'gaussian_noise':
            return self._add_gaussian_noise(img, severity)
        elif corruption_type == 'motion_blur':
            kernel_size = min(15, 3 + 2 * severity)
            sigma = max(0.5, severity * 0.5)
            return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        elif corruption_type == 'brightness':
            factor = 0.5 + (severity / 5.0)
            return TF.adjust_brightness(img, brightness_factor=factor)
        elif corruption_type == 'contrast':
            factor = 0.3 + (severity / 5.0) * 1.4
            return TF.adjust_contrast(img, contrast_factor=factor)
        else:
            return img
    
    def _add_gaussian_noise(self, img, severity):
        """Add gaussian noise to PIL image"""
        tensor = TF.to_tensor(img)
        noise_std = 0.02 + (severity / 5.0) * 0.08
        noise = torch.randn_like(tensor) * noise_std
        noisy_tensor = torch.clamp(tensor + noise, 0, 1)
        return TF.to_pil_image(noisy_tensor)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample
        if hasattr(self.base_dataset, 'dataset'):
            real_idx = self.base_dataset.indices[idx]
            img_path, label = self.base_dataset.dataset.samples[real_idx]
        else:
            img_path, label = self.base_dataset.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Create clean version
        clean_img = img.copy()
        if self.transform:
            clean_tensor = self.transform(clean_img)
        else:
            clean_tensor = TF.to_tensor(clean_img)
        
        # Decide whether to create contrastive pair
        create_contrastive_pair = (
            self.contrastive_enabled and 
            random.random() < self.contrastive_pairs_per_batch
        )
        
        if create_contrastive_pair:
            # Create corrupted version of the same image
            corruption_type = random.choice(self.corruption_types)
            severity = random.randint(*self.severity_range)
            corrupted_img = self.apply_corruption(img.copy(), corruption_type, severity)
            
            if self.transform:
                corrupted_tensor = self.transform(corrupted_img)
            else:
                corrupted_tensor = TF.to_tensor(corrupted_img)
            
            return {
                'clean': clean_tensor,
                'corrupted': corrupted_tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'is_contrastive_pair': torch.tensor(1, dtype=torch.bool),
                'corruption_type': corruption_type,
                'severity': severity
            }
        else:
            # Standard single image (might be corrupted based on corruption_prob)
            if random.random() < self.corruption_prob:
                corruption_type = random.choice(self.corruption_types)
                severity = random.randint(*self.severity_range)
                img = self.apply_corruption(img, corruption_type, severity)
            
            if self.transform:
                tensor = self.transform(img)
            else:
                tensor = TF.to_tensor(img)
            
            return {
                'clean': tensor,
                'corrupted': tensor,  # Same as clean for non-contrastive samples
                'label': torch.tensor(label, dtype=torch.long),
                'is_contrastive_pair': torch.tensor(0, dtype=torch.bool),
                'corruption_type': 'none',
                'severity': 0
            }

def contrastive_loss(z_clean, z_corrupted, temperature=0.07):
    """
    Compute contrastive loss between clean and corrupted features
    """
    batch_size = z_clean.shape[0]
    
    # Normalize features
    z_clean = F.normalize(z_clean, dim=1)
    z_corrupted = F.normalize(z_corrupted, dim=1)
    
    # Compute similarity matrix
    # Positive pairs: clean[i] with corrupted[i]
    # Negative pairs: clean[i] with corrupted[j] where j != i, and clean[i] with clean[j] where j != i
    
    # Concatenate clean and corrupted features
    features = torch.cat([z_clean, z_corrupted], dim=0)  # [2*batch_size, feature_dim]
    
    # Compute similarity matrix
    sim_matrix = torch.mm(features, features.t()) / temperature  # [2*batch_size, 2*batch_size]
    
    # Create labels for positive pairs
    # For clean[i], the positive is corrupted[i] (at index batch_size + i)
    # For corrupted[i], the positive is clean[i] (at index i - batch_size)
    labels = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z_clean.device)
    
    # Remove self-similarities (diagonal)
    mask = torch.eye(2*batch_size, dtype=torch.bool).to(z_clean.device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

def build_transforms(img_size, resize, corruption_mode=False):
    if corruption_mode:
        # For corruption training, use more aggressive augmentation
        tr = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Standard training transforms
        tr = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Evaluation transforms (no augmentation)
    ev = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return tr, ev

def stratified_split(ds, a, b, c, seed=42):
    assert abs(a + b + c - 1.0) < 1e-6
    rng = random.Random(seed)
    cls_idxs = {}
    
    for i, (_, y) in enumerate(ds.samples):
        cls_idxs.setdefault(y, []).append(i)
    
    tr, va, te = [], [], []
    for y, idxs in cls_idxs.items():
        rng.shuffle(idxs)
        n = len(idxs)
        ntr = int(round(a * n))
        nva = int(round(b * n))
        nte = n - ntr - nva
        tr += idxs[:ntr]
        va += idxs[ntr:ntr + nva]
        te += idxs[ntr + nva:]
    
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te

def build_model(C, freeze_backbone=True, contrastive_enabled=False, feature_dim=128):
    if contrastive_enabled:
        return MobileNetV3WithContrastive(C, feature_dim, freeze_backbone)
    else:
        # Standard model without contrastive learning
        w = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=w)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, C)
        
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        
        return model

def accuracy(logits, y):
    with torch.no_grad():
        pred = logits.argmax(1)
        return (pred == y).float().mean().item() * 100.0

def train_epoch_contrastive(model, train_dl, optimizer, criterion, device, epoch, total_epochs, contrastive_cfg):
    """Train for one epoch with contrastive learning"""
    model.train()
    tot_loss = 0
    tot_cls_loss = 0
    tot_contrastive_loss = 0
    tot_acc = 0
    n = 0
    contrastive_pairs = 0
    
    contrastive_weight = contrastive_cfg.get('contrastive_weight', 0.5)
    contrastive_temperature = contrastive_cfg.get('contrastive_temperature', 0.07)
    
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{total_epochs}")
    for batch in pbar:
        clean_imgs = batch['clean'].to(device)
        corrupted_imgs = batch['corrupted'].to(device)
        labels = batch['label'].to(device)
        is_contrastive_pair = batch['is_contrastive_pair'].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass for clean images (classification)
        clean_logits = model(clean_imgs)
        
        # Classification loss (always computed)
        cls_loss = criterion(clean_logits, labels)
        total_loss = cls_loss
        
        # Contrastive loss (only for contrastive pairs)
        contrastive_loss_val = 0
        contrastive_mask = is_contrastive_pair
        if contrastive_mask.any():
            # Get contrastive features for pairs
            clean_features = model.get_contrastive_features(clean_imgs[contrastive_mask])
            corrupted_features = model.get_contrastive_features(corrupted_imgs[contrastive_mask])
            
            contrastive_loss_val = contrastive_loss(
                clean_features, corrupted_features, contrastive_temperature
            )
            total_loss = total_loss + contrastive_weight * contrastive_loss_val
            contrastive_pairs += contrastive_mask.sum().item()
        
        total_loss.backward()
        optimizer.step()
        
        b = clean_imgs.size(0)
        tot_loss += total_loss.item() * b
        tot_cls_loss += cls_loss.item() * b
        if isinstance(contrastive_loss_val, torch.Tensor):
            tot_contrastive_loss += contrastive_loss_val.item() * b
        tot_acc += accuracy(clean_logits, labels) * b
        n += b
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{tot_loss/n:.4f}",
            'cls': f"{tot_cls_loss/n:.4f}",
            'contr': f"{tot_contrastive_loss/n:.4f}",
            'acc': f"{tot_acc/n:.2f}%",
            'pairs': contrastive_pairs
        })
    
    return tot_loss / max(1, n), tot_acc / max(1, n), tot_cls_loss / max(1, n), tot_contrastive_loss / max(1, n)

def train_epoch(model, train_dl, optimizer, criterion, device, epoch, total_epochs):
    """Standard train for one epoch (no contrastive learning)"""
    model.train()
    tot_loss = 0
    tot_acc = 0
    n = 0
    
    pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{total_epochs}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        b = x.size(0)
        tot_loss += loss.item() * b
        tot_acc += accuracy(logits, y) * b
        n += b
        
        pbar.set_postfix({
            'loss': f"{tot_loss/n:.4f}",
            'acc': f"{tot_acc/n:.2f}%"
        })
    
    return tot_loss / max(1, n), tot_acc / max(1, n)

def validate_epoch(model, val_dl, criterion, device):
    """Validate for one epoch"""
    model.eval()
    va_loss = 0
    va_acc = 0
    nva = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_dl, desc="Validation"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            b = x.size(0)
            va_loss += loss.item() * b
            va_acc += accuracy(logits, y) * b
            nva += b
    
    return va_loss / max(1, nva), va_acc / max(1, nva)

def corruption_finetune(model, train_ds, val_dl, cfg, device, class_to_idx):
    """Fine-tune model on corruption-augmented data with contrastive learning"""
    print("\n" + "="*60)
    print("STARTING CORRUPTION-AWARE FINE-TUNING WITH CONTRASTIVE LEARNING")
    print("="*60)
    
    cf_cfg = cfg["corruption_finetune"]
    contrastive_enabled = cf_cfg.get("contrastive_enabled", True)
    
    print(f"Corruption fine-tune LR: {cf_cfg['lr']} (type: {type(cf_cfg['lr'])})")
    print(f"Weight decay: {cf_cfg['weight_decay']} (type: {type(cf_cfg['weight_decay'])})")
    print(f"Contrastive learning: {'enabled' if contrastive_enabled else 'disabled'}")
    if contrastive_enabled:
        print(f"Contrastive weight: {cf_cfg['contrastive_weight']}")
        print(f"Contrastive temperature: {cf_cfg['contrastive_temperature']}")
        print(f"Feature dimension: {cf_cfg['feature_dim']}")
    
    # Create corruption-augmented dataset with contrastive pairs
    _, corruption_transform = build_transforms(
        cfg["data"]["img_size"], 
        cfg["data"]["resize"], 
        corruption_mode=True
    )
    
    corruption_train_ds = ContrastiveCorruptionDataset(
        train_ds, cf_cfg, transform=corruption_transform, contrastive_enabled=contrastive_enabled
    )
    
    corruption_train_dl = DataLoader(
        corruption_train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    
    # Setup training for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cf_cfg["lr"],
        weight_decay=cf_cfg["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cf_cfg["epochs"]
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    
    # Training loop
    best_val_acc = -1.0
    best_model_state = None
    
    for epoch in range(cf_cfg["epochs"]):
        # Train with contrastive learning
        if contrastive_enabled:
            train_loss, train_acc, cls_loss, contr_loss = train_epoch_contrastive(
                model, corruption_train_dl, optimizer, criterion, device, epoch, cf_cfg["epochs"], cf_cfg
            )
            print(f"Corruption Fine-tune Epoch {epoch+1}/{cf_cfg['epochs']} | "
                  f"train {train_loss:.4f}/{train_acc:.2f}% | "
                  f"cls {cls_loss:.4f} | contr {contr_loss:.4f}")
        else:
            # For non-contrastive, we need to create a standard dataloader
            standard_samples = []
            for i in range(len(corruption_train_ds)):
                sample = corruption_train_ds[i]
                standard_samples.append((sample['clean'], sample['label']))
            
            standard_train_dl = DataLoader(
                standard_samples,
                batch_size=cfg["data"]["batch_size"],
                shuffle=True,
                num_workers=cfg["data"]["num_workers"],
                pin_memory=True
            )
            
            train_loss, train_acc = train_epoch(
                model, standard_train_dl, optimizer, criterion, device, epoch, cf_cfg["epochs"]
            )
            print(f"Corruption Fine-tune Epoch {epoch+1}/{cf_cfg['epochs']} | "
                  f"train {train_loss:.4f}/{train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        print(f"  val {val_loss:.4f}/{val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  -> New best validation accuracy: {best_val_acc:.2f}%")
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save corruption fine-tuned model
    corruption_ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], cf_cfg["ckpt_name"])
    save_dict = {
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "cfg": cfg,
        "corruption_finetuned": True,
        "contrastive_enabled": contrastive_enabled,
        "best_corruption_val_acc": best_val_acc
    }
    
    torch.save(save_dict, corruption_ckpt_path)
    
    print(f"Corruption fine-tuned model saved to: {corruption_ckpt_path}")
    print(f"Best corruption validation accuracy: {best_val_acc:.2f}%")
    
    return best_val_acc

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["system"]["seed"])
    device = torch.device(cfg["system"]["device"])
    
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Corruption fine-tuning: {args.corruption_finetune}")
    print(f"  Base model: {args.base_model}")
    
    # Build transforms
    tr_tf, ev_tf = build_transforms(cfg["data"]["img_size"], cfg["data"]["resize"])
    
    # Load dataset and create splits
    full = datasets.ImageFolder(cfg["data"]["root"], transform=tr_tf)
    tr_idx, va_idx, te_idx = stratified_split(
        full, 
        cfg["data"]["train_frac"], 
        cfg["data"]["val_frac"], 
        cfg["data"]["test_frac"], 
        cfg["data"]["balance_seed"]
    )
    
    # Create datasets
    train_ds = Subset(datasets.ImageFolder(cfg["data"]["root"], transform=tr_tf), tr_idx)
    val_ds = Subset(datasets.ImageFolder(cfg["data"]["root"], transform=ev_tf), va_idx)
    test_ds = Subset(datasets.ImageFolder(cfg["data"]["root"], transform=ev_tf), te_idx)
    
    # Create data loaders
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, 
                         num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, 
                       num_workers=cfg["data"]["num_workers"], pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, 
                        num_workers=cfg["data"]["num_workers"], pin_memory=True)
    
    C = len(full.class_to_idx)
    print(f"Dataset: {len(full)} images, {C} classes")
    print(f"Splits: {len(tr_idx)} train, {len(va_idx)} val, {len(te_idx)} test")
    
    # Build model (with contrastive head if needed)
    contrastive_enabled = (
        (args.corruption_finetune or cfg["corruption_finetune"]["enabled"]) and 
        cfg["corruption_finetune"].get("contrastive_enabled", True)
    )
    
    model = build_model(
        C, 
        cfg["train"]["freeze_backbone"], 
        contrastive_enabled=contrastive_enabled,
        feature_dim=cfg["corruption_finetune"].get("feature_dim", 128)
    ).to(device)
    
    # Load base model if provided
    if args.base_model and os.path.exists(args.base_model):
        print(f"Loading base model from: {args.base_model}")
        checkpoint = torch.load(args.base_model, map_location=device)
        
        # Handle loading when contrastive head might not be in checkpoint
        model_state = checkpoint['model_state']
        try:
            model.load_state_dict(model_state, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Attempting partial loading (ignoring projection_head)...")
            model.load_state_dict(model_state, strict=False)
    
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    
    # Initialize variables for tracking results
    best_top1 = None
    corruption_val_acc = None
    
    # Regular training (if not starting with corruption fine-tuning)
    if not args.corruption_finetune:
        print("\n" + "="*60)
        print("STARTING REGULAR TRAINING")
        print("="*60)
        
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=cfg["train"]["lr"], 
            weight_decay=cfg["train"]["weight_decay"]
        )
        ce = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
        
        best_path = os.path.join(cfg["train"]["ckpt_dir"], cfg["train"]["ckpt_name"])
        best_top1 = -1.0
        
        for epoch in range(cfg["train"]["epochs"]):
            # Train
            train_loss, train_acc = train_epoch(model, train_dl, opt, ce, device, epoch, cfg["train"]["epochs"])
            
            # Validate
            val_loss, val_acc = validate_epoch(model, val_dl, ce, device)
            
            print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
                  f"train {train_loss:.4f}/{train_acc:.2f}% | "
                  f"val {val_loss:.4f}/{val_acc:.2f}%")
            
            if val_acc > best_top1:
                best_top1 = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "class_to_idx": full.class_to_idx,
                    "cfg": cfg
                }, best_path)
                print(f"  -> saved best to {best_path} (val top1 {best_top1:.2f}%)")
    
    # Corruption fine-tuning with contrastive learning
    if args.corruption_finetune or cfg["corruption_finetune"]["enabled"]:
        corruption_val_acc = corruption_finetune(model, train_ds, val_dl, cfg, device, full.class_to_idx)
    
    # Create manifest with only JSON-serializable data
    manifest = {
        "class_to_idx": full.class_to_idx,
        "splits": {
            "train": [full.samples[i][0] for i in tr_idx],
            "val": [full.samples[i][0] for i in va_idx],
            "test": [full.samples[i][0] for i in te_idx]
        },
        "best_checkpoint": os.path.join(cfg["train"]["ckpt_dir"], cfg["train"]["ckpt_name"]),
        "corruption_checkpoint": os.path.join(cfg["train"]["ckpt_dir"], cfg["corruption_finetune"]["ckpt_name"]) if (args.corruption_finetune or cfg["corruption_finetune"]["enabled"]) else None,
        "corruption_finetuned": args.corruption_finetune or cfg["corruption_finetune"]["enabled"],
        "contrastive_enabled": contrastive_enabled
    }
    
    # Add accuracy values only if they exist and are numeric
    if best_top1 is not None:
        manifest["best_val_top1"] = float(best_top1)
    
    if corruption_val_acc is not None:
        manifest["best_corruption_val_acc"] = float(corruption_val_acc)
    
    manifest_path = os.path.join(cfg["train"]["ckpt_dir"], "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nSaved manifest to: {manifest_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()