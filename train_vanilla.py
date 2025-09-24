#!/usr/bin/env python3
# filepath: /home/rahim/exp/tta-pncache-tta/train_vanilla.py
"""
Vanilla MobileNetV3 Training Script
Simple baseline training without corruption augmentation or contrastive learning
"""

import argparse
import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='Vanilla MobileNetV3 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_name', type=str, default='mobilenetv3_13classes_best_vanilla.pt', 
                       help='Checkpoint save name')
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
    
    # Set defaults
    cfg.setdefault("data", {})
    cfg.setdefault("system", {})
    cfg.setdefault("train", {})
    
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
    
    # Convert to proper types
    t["lr"] = float(t["lr"])
    t["weight_decay"] = float(t["weight_decay"])
    t["label_smoothing"] = float(t["label_smoothing"])
    t["dropout_rate"] = float(t["dropout_rate"])
    
    return cfg

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
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
    """Create stratified splits maintaining class distribution"""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    targets = [dataset.samples[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    
    # Split into train+val and test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_frac, stratify=targets, random_state=seed
    )
    
    # Split train+val into train and val
    train_val_targets = [targets[i] for i in train_val_indices]
    val_size = val_frac / (train_frac + val_frac)
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, stratify=train_val_targets, random_state=seed
    )
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

def build_transforms(img_size, resize, is_training=True):
    """Build data transforms"""
    if is_training:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(img_size, padding=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        # Validation/test transforms
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def save_split_manifest(train_ds, val_ds, test_ds, full_ds, cfg, save_path):
    """Save dataset split manifest"""
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
        "data_root": cfg["data"]["root"],
        "model_type": "vanilla"
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Split manifest saved to: {save_path}")
    return save_path

class VanillaMobileNetV3(nn.Module):
    """Vanilla MobileNetV3 model for classification"""
    
    def __init__(self, num_classes, freeze_backbone=True, dropout_rate=0.3):
        super().__init__()
        
        # Load pretrained MobileNetV3
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_large(weights=weights)
        
        # MobileNetV3-Large has 960 features before the final classifier
        # Let's check the actual structure and fix it
        backbone_features = 960
        
        # Replace the classifier with a simpler one that matches the backbone features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(512, num_classes)
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
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before the final classifier"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """Training epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Train {epoch}/{total_epochs}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        if batch_idx % 10 == 0:  # Update every 10 batches to reduce overhead
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
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
        
        # Update progress bar less frequently
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, train_acc, val_acc, save_path, cfg, class_names):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'num_classes': model.num_classes,
        'class_names': class_names,
        'config': cfg,
        'model_type': 'vanilla',
        'freeze_backbone': model.freeze_backbone,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path} (Val Acc: {val_acc:.2f}%)")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint for resuming training"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 0, 0.0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        val_acc = checkpoint.get('val_acc', 0.0)
        
        print(f"✅ Resumed from epoch {epoch} with val acc {val_acc:.2f}%")
        return epoch, val_acc
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return 0, 0.0

def create_optimizer_and_scheduler(model, cfg):
    """Create optimizer and learning rate scheduler"""
    train_cfg = cfg["train"]
    
    # Different learning rates for frozen vs unfrozen parts
    if model.freeze_backbone:
        # Only classifier parameters
        params = model.backbone.classifier.parameters()
        lr = train_cfg["lr"]
    else:
        # Different learning rates for backbone and classifier
        backbone_params = list(model.backbone.features.parameters())
        classifier_params = list(model.backbone.classifier.parameters())
        
        params = [
            {'params': backbone_params, 'lr': train_cfg["lr"] * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': train_cfg["lr"]}        # Higher LR for classifier
        ]
        lr = train_cfg["lr"]
    
    optimizer = torch.optim.AdamW(
        params, 
        lr=lr,
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    warmup_epochs = train_cfg["warmup_epochs"]
    total_epochs = train_cfg["epochs"]
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def main():
    print("🚀 Vanilla MobileNetV3 Training")
    args = parse_args()
    
    # Load configuration
    cfg = load_cfg(args.config)
    device = torch.device(cfg["system"]["device"])
    set_seed(cfg["system"]["seed"])
    
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Config: {args.config}")
    
    # Auto-detect dataset
    num_classes, class_names = get_num_classes(cfg["data"]["root"])
    
    # Build transforms
    train_transform = build_transforms(
        cfg["data"]["img_size"], 
        cfg["data"]["resize"], 
        is_training=True
    )
    val_transform = build_transforms(
        cfg["data"]["img_size"], 
        cfg["data"]["resize"], 
        is_training=False
    )
    
    # Load dataset and create splits
    print("📁 Loading dataset...")
    full_ds = datasets.ImageFolder(cfg["data"]["root"], transform=None)
    train_ds, val_ds, test_ds = stratified_split(
        full_ds, 
        cfg["data"]["train_frac"], 
        cfg["data"]["val_frac"], 
        cfg["data"]["test_frac"], 
        cfg["data"]["balance_seed"]
    )
    
    print(f"Dataset splits - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Save split manifest
    manifest_path = save_split_manifest(
        train_ds, val_ds, test_ds, full_ds, cfg, 
        "./checkpoints/split_manifest.json"
    )
    
    # Apply transforms
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform
    
    # Create data loaders
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg["data"]["batch_size"], 
        shuffle=True, 
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    
    val_dl = DataLoader(
        val_ds, 
        batch_size=cfg["data"]["batch_size"], 
        shuffle=False, 
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    
    # Create model
    print("🏗️  Building model...")
    model = VanillaMobileNetV3(
        num_classes=num_classes,
        freeze_backbone=cfg["train"]["freeze_backbone"],
        dropout_rate=cfg["train"]["dropout_rate"]
    )
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, cfg)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["train"]["label_smoothing"]
    )
    
    print(f"📊 Training Configuration:")
    print(f"  Model: VanillaMobileNetV3")
    print(f"  Classes: {num_classes}")
    print(f"  Epochs: {cfg['train']['epochs']}")
    print(f"  Learning Rate: {cfg['train']['lr']}")
    print(f"  Batch Size: {cfg['data']['batch_size']}")
    print(f"  Freeze Backbone: {cfg['train']['freeze_backbone']}")
    print(f"  Label Smoothing: {cfg['train']['label_smoothing']}")
    print(f"  Dropout Rate: {cfg['train']['dropout_rate']}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
    
    # Training loop
    print(f"\n🎯 Starting training from epoch {start_epoch + 1}...")
    print("=" * 80)
    
    train_history = []
    val_history = []
    
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        epoch_num = epoch + 1
        
        print(f"\nEpoch {epoch_num}/{cfg['train']['epochs']}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_dl, optimizer, criterion, device, epoch_num, cfg["train"]["epochs"]
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"Results:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save history
        train_history.append({
            'epoch': epoch_num,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(cfg["train"]["ckpt_dir"], args.save_name)
            save_checkpoint(
                model, optimizer, scheduler, epoch_num, train_acc, val_acc, 
                save_path, cfg, class_names
            )
            print(f"🎉 New best model! Validation accuracy: {val_acc:.2f}%")
        
        # Save regular checkpoint every 5 epochs
        if epoch_num % 5 == 0:
            regular_save_path = os.path.join(
                cfg["train"]["ckpt_dir"], 
                f"mobilenetv3_vanilla_epoch_{epoch_num}.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch_num, train_acc, val_acc,
                regular_save_path, cfg, class_names
            )
    
    # Final evaluation on test set
    print(f"\n🧪 Final evaluation on test set...")
    test_dl = DataLoader(
        test_ds, 
        batch_size=cfg["data"]["batch_size"], 
        shuffle=False, 
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    
    # Load best model for final evaluation
    best_model_path = os.path.join(cfg["train"]["ckpt_dir"], args.save_name)
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model for final evaluation")
    
    test_loss, test_acc = validate_epoch(model, test_dl, criterion, device)
    
    # Save training history
    history_path = os.path.join(cfg["train"]["ckpt_dir"], "vanilla_training_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'train_history': train_history,
            'final_test_acc': test_acc,
            'final_test_loss': test_loss,
            'best_val_acc': best_val_acc,
            'config': cfg,
            'class_names': class_names
        }, f, indent=2)
    
    print(f"\n🎊 Training completed!")
    print("=" * 80)
    print(f"📈 Results Summary:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Final Test Accuracy: {test_acc:.2f}%")
    print(f"  Model saved: {best_model_path}")
    print(f"  Training history: {history_path}")
    print(f"  Split manifest: {manifest_path}")

if __name__ == "__main__":
    main()