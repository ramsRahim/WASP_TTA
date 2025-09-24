#!/usr/bin/env python3
"""
Fixed Corruption Evaluation Script
Evaluates both vanilla and fine-tuned models on various weather corruptions
"""

import argparse
import json
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from tqdm import tqdm
import random
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Corruption Robustness Evaluation')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--vanilla_checkpoint', required=True, help='Vanilla model checkpoint')
    parser.add_argument('--finetuned_checkpoint', required=True, help='Fine-tuned model checkpoint')
    parser.add_argument('--manifest_path', required=True, help='Split manifest path')
    parser.add_argument('--output_dir', default='./eval_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    return parser.parse_args()

def load_config(config_path):
    """Load evaluation configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults
    config.setdefault('data', {})
    config['data'].setdefault('img_size', 224)
    config['data'].setdefault('resize', 256)
    
    config.setdefault('corruption', {})
    config['corruption'].setdefault('severities', [1, 2, 3, 4, 5])
    config['corruption'].setdefault('types', [
        'gaussian_noise', 'motion_blur', 'brightness', 'contrast',
        'rain', 'snow', 'frost', 'fog', 'low_light', 'high_light',
        'shadow', 'glare', 'haze', 'mist'
    ])
    
    return config

def load_manifest(manifest_path):
    """Load dataset split manifest"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint to determine model architecture"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
    
    # Find classifier layers
    classifier_layers = {}
    contrastive_layers = {}
    consistency_layers = {}
    
    for key in state_dict.keys():
        if 'backbone.classifier' in key:
            classifier_layers[key] = state_dict[key].shape
        elif 'contrastive_head' in key:
            contrastive_layers[key] = state_dict[key].shape
        elif 'consistency_head' in key:
            consistency_layers[key] = state_dict[key].shape
    
    print(f"  Classifier layers: {classifier_layers}")
    print(f"  Contrastive layers: {contrastive_layers}")
    print(f"  Consistency layers: {consistency_layers}")
    
    return classifier_layers, contrastive_layers, consistency_layers

class VanillaMobileNetV3(nn.Module):
    """Vanilla MobileNetV3 model - matches train_vanilla.py"""
    
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # Load pretrained MobileNetV3
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_large(weights=weights)
        
        # MobileNetV3-Large has 960 features before the final classifier
        backbone_features = 960
        
        # Replace the classifier - matches train_vanilla.py
        self.backbone.classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)

class AdaptiveMobileNetV3(nn.Module):
    """Adaptive MobileNetV3 that builds architecture from checkpoint"""
    
    def __init__(self, num_classes, classifier_layers, contrastive_layers=None, consistency_layers=None, dropout_rate=0.2):
        super().__init__()
        
        # Load pretrained MobileNetV3
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_large(weights=weights)
        
        backbone_feature_dim = 960
        
        # Build classifier from checkpoint architecture
        self._build_classifier(classifier_layers, dropout_rate)
        
        # Build contrastive head if present
        if contrastive_layers:
            self._build_contrastive_head(contrastive_layers, backbone_feature_dim, dropout_rate)
        
        # Build consistency head if present  
        if consistency_layers:
            self._build_consistency_head(consistency_layers, backbone_feature_dim)
        
        self.num_classes = num_classes
    
    def _build_classifier(self, classifier_layers, dropout_rate):
        """Build classifier from checkpoint layer info"""
        layers = []
        
        # Sort layers by index
        sorted_layers = sorted(classifier_layers.items())
        
        for i, (layer_name, shape) in enumerate(sorted_layers):
            if 'weight' in layer_name:
                layer_idx = int(layer_name.split('.')[2])  # backbone.classifier.X.weight
                
                if layer_idx == 0:
                    # First layer
                    in_features, out_features = shape[1], shape[0]
                    layers.append(nn.Linear(in_features, out_features))
                    layers.append(nn.Hardswish(inplace=True))
                    layers.append(nn.Dropout(p=dropout_rate, inplace=True))
                else:
                    # Find corresponding bias to get output features
                    bias_key = layer_name.replace('weight', 'bias')
                    if bias_key in classifier_layers:
                        out_features = classifier_layers[bias_key][0]
                        in_features = shape[1]
                        layers.append(nn.Linear(in_features, out_features))
                        
                        # Add activation except for last layer
                        if layer_idx < max([int(k.split('.')[2]) for k in classifier_layers.keys() if 'weight' in k]):
                            layers.append(nn.Hardswish(inplace=True))
                            layers.append(nn.Dropout(p=dropout_rate, inplace=True))
        
        self.backbone.classifier = nn.Sequential(*layers)
        print(f"✅ Built classifier: {self.backbone.classifier}")
    
    def _build_contrastive_head(self, contrastive_layers, backbone_feature_dim, dropout_rate):
        """Build contrastive head from checkpoint"""
        layers = []
        sorted_layers = sorted(contrastive_layers.items())
        
        for layer_name, shape in sorted_layers:
            if 'weight' in layer_name:
                layer_idx = int(layer_name.split('.')[1])  # contrastive_head.X.weight
                
                if layer_idx == 0:
                    in_features, out_features = shape[1], shape[0]
                    layers.append(nn.Linear(in_features, out_features))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout_rate))
                else:
                    bias_key = layer_name.replace('weight', 'bias')
                    if bias_key in contrastive_layers:
                        out_features = contrastive_layers[bias_key][0]
                        in_features = shape[1]
                        layers.append(nn.Linear(in_features, out_features))
        
        self.contrastive_head = nn.Sequential(*layers)
        print(f"✅ Built contrastive head: {self.contrastive_head}")
    
    def _build_consistency_head(self, consistency_layers, backbone_feature_dim):
        """Build consistency head from checkpoint"""
        layers = []
        sorted_layers = sorted(consistency_layers.items())
        
        for layer_name, shape in sorted_layers:
            if 'weight' in layer_name:
                layer_idx = int(layer_name.split('.')[1])  # consistency_head.X.weight
                
                bias_key = layer_name.replace('weight', 'bias')
                if bias_key in consistency_layers:
                    out_features = consistency_layers[bias_key][0]
                    in_features = shape[1]
                    layers.append(nn.Linear(in_features, out_features))
                    
                    # Add ReLU except for last layer
                    if layer_idx < max([int(k.split('.')[1]) for k in consistency_layers.keys() if 'weight' in k]):
                        layers.append(nn.ReLU(inplace=True))
        
        self.consistency_head = nn.Sequential(*layers)
        print(f"✅ Built consistency head: {self.consistency_head}")
    
    def get_features(self, x):
        """Extract features before the final classifier"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_contrastive_features(self, x):
        """Get normalized contrastive features"""
        if hasattr(self, 'contrastive_head'):
            features = self.get_features(x)
            contrastive_features = self.contrastive_head(features)
            return F.normalize(contrastive_features, dim=1)
        return None
    
    def get_consistency_features(self, x):
        """Get consistency features"""
        if hasattr(self, 'consistency_head'):
            features = self.get_features(x)
            return self.consistency_head(features)
        return None
    
    def forward(self, x):
        """Forward pass through the backbone classifier"""
        features = self.get_features(x)
        return self.backbone.classifier(features)

class WeatherCorruptionDataset(Dataset):
    """Dataset with weather corruption augmentation"""
    
    def __init__(self, image_paths, labels, transform=None, corruption_type=None, severity=1):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.corruption_type = corruption_type
        self.severity = severity
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply corruption if specified
        if self.corruption_type and self.corruption_type != 'clean':
            img = self._apply_corruption(img, self.corruption_type, self.severity)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def _apply_corruption(self, img, corruption_type, severity):
        """Apply the specified corruption type and severity"""
        try:
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
        except Exception as e:
            print(f"Error applying corruption {corruption_type}: {e}")
            return img
    
    def _add_gaussian_noise(self, img, severity):
        tensor = TF.to_tensor(img)
        noise_std = 0.02 + (severity / 5.0) * 0.08
        noise = torch.randn_like(tensor) * noise_std
        noisy_tensor = torch.clamp(tensor + noise, 0, 1)
        return TF.to_pil_image(noisy_tensor)
    
    def _add_motion_blur(self, img, severity):
        kernel_size = min(15, 3 + 2 * severity)
        if kernel_size % 2 == 0:
            kernel_size += 1
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

def load_model(checkpoint_path, num_classes, device='cuda'):
    """Load model from checkpoint with adaptive architecture"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Inspect checkpoint to determine architecture
    classifier_layers, contrastive_layers, consistency_layers = inspect_checkpoint(checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type
    if contrastive_layers or consistency_layers:
        # Enhanced model with contrastive/consistency heads
        model = AdaptiveMobileNetV3(
            num_classes=num_classes,
            classifier_layers=classifier_layers,
            contrastive_layers=contrastive_layers,
            consistency_layers=consistency_layers
        )
        model_type = 'adaptive_enhanced'
    else:
        # Vanilla model
        model = VanillaMobileNetV3(
            num_classes=num_classes,
            dropout_rate=0.3
        )
        model_type = 'vanilla'
    
    model.to(device)
    
    # Load state dict
    try:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Model loaded successfully from {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    
    return model, model_type

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        
        try:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
        except Exception as e:
            print(f"Error in evaluation batch: {e}")
            continue
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, predictions, targets

def build_transforms(img_size, resize):
    """Build evaluation transforms"""
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_corruption_results_table(results, corruption_types, severities):
    """Create a formatted results table"""
    print(f"\n{'='*100}")
    print(f"{'CORRUPTION ROBUSTNESS EVALUATION RESULTS':^100}")
    print(f"{'='*100}")
    
    # Header
    header = f"{'Corruption':<15} {'Model':<10}"
    for sev in severities:
        header += f"{'Sev ' + str(sev):<8}"
    header += f"{'mCE':<8} {'Improvement':<12}"
    print(header)
    print("-" * 100)
    
    for corruption in corruption_types + ['clean']:
        for model_name in ['Vanilla', 'Fine-tuned']:
            if corruption in results and model_name in results[corruption]:
                row = f"{corruption:<15} {model_name:<10}"
                
                # Add severity results
                sev_results = results[corruption][model_name]
                if isinstance(sev_results, dict):
                    for sev in severities:
                        if sev in sev_results:
                            row += f"{sev_results[sev]:<8.1f}"
                        else:
                            row += f"{'N/A':<8}"
                else:
                    # Clean accuracy (single value)
                    row += f"{sev_results:<8.1f}"
                    for _ in severities[1:]:
                        row += f"{'N/A':<8}"
                
                # Add mCE and improvement if available
                if 'mCE' in results[corruption]:
                    if model_name in results[corruption]['mCE']:
                        row += f"{results[corruption]['mCE'][model_name]:<8.1f}"
                    else:
                        row += f"{'N/A':<8}"
                else:
                    row += f"{'N/A':<8}"
                
                if 'improvement' in results[corruption]:
                    row += f"{results[corruption]['improvement']:<12.1f}"
                else:
                    row += f"{'N/A':<12}"
                
                print(row)
        
        if corruption != 'clean':
            print("-" * 100)

def save_results(results, output_dir):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    results_file = os.path.join(output_dir, 'corruption_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")

def main():
    args = parse_args()
    
    # Load configuration and manifest
    config = load_config(args.config)
    manifest = load_manifest(args.manifest_path)
    
    print(f"Using manifest: {args.manifest_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dataset info
    class_names = manifest['dataset_info']['class_names']
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")
    
    # Prepare test data
    test_paths = manifest['splits']['test']
    
    # Create class to index mapping
    class_to_idx = manifest['class_to_idx']
    
    # Extract labels from paths
    def get_label_from_path(path):
        class_name = os.path.basename(os.path.dirname(path))
        return class_to_idx[class_name]
    
    test_labels = [get_label_from_path(path) for path in test_paths]
    
    print(f"Test samples: {len(test_paths)}")
    
    # Build transforms
    transform = build_transforms(config['data']['img_size'], config['data']['resize'])
    
    # Load models
    print("📥 Loading models...")
    
    # Load vanilla model
    vanilla_model, vanilla_type = load_model(
        args.vanilla_checkpoint, 
        num_classes, 
        device=device
    )
    print(f"✓ Loaded vanilla model (type: {vanilla_type})")
    
    # Load fine-tuned model
    finetuned_model, finetuned_type = load_model(
        args.finetuned_checkpoint, 
        num_classes, 
        device=device
    )
    print(f"✓ Loaded fine-tuned model (type: {finetuned_type})")
    
    # Evaluation
    print("\n🧪 Starting corruption robustness evaluation...")
    
    results = {}
    corruption_types = config['corruption']['types']
    severities = config['corruption']['severities']
    
    # Evaluate clean performance first
    print("\n📊 Evaluating clean performance...")
    clean_dataset = WeatherCorruptionDataset(
        test_paths, test_labels, transform=transform, 
        corruption_type='clean', severity=0
    )
    clean_loader = DataLoader(
        clean_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )
    
    vanilla_clean_acc, _, _ = evaluate_model(vanilla_model, clean_loader, device)
    finetuned_clean_acc, _, _ = evaluate_model(finetuned_model, clean_loader, device)
    
    results['clean'] = {
        'Vanilla': vanilla_clean_acc,
        'Fine-tuned': finetuned_clean_acc
    }
    
    print(f"Clean Accuracy - Vanilla: {vanilla_clean_acc:.2f}%, Fine-tuned: {finetuned_clean_acc:.2f}%")
    
    # Evaluate corruptions
    for corruption in corruption_types:
        print(f"\n🌩️  Evaluating {corruption}...")
        
        results[corruption] = {
            'Vanilla': {},
            'Fine-tuned': {}
        }
        
        for severity in severities:
            print(f"  Severity {severity}...")
            
            # Create corrupted dataset
            corrupted_dataset = WeatherCorruptionDataset(
                test_paths, test_labels, transform=transform,
                corruption_type=corruption, severity=severity
            )
            corrupted_loader = DataLoader(
                corrupted_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers
            )
            
            # Evaluate both models
            vanilla_acc, _, _ = evaluate_model(vanilla_model, corrupted_loader, device)
            finetuned_acc, _, _ = evaluate_model(finetuned_model, corrupted_loader, device)
            
            results[corruption]['Vanilla'][severity] = vanilla_acc
            results[corruption]['Fine-tuned'][severity] = finetuned_acc
            
            print(f"    Vanilla: {vanilla_acc:.2f}%, Fine-tuned: {finetuned_acc:.2f}%")
        
        # Calculate improvement
        avg_vanilla = np.mean(list(results[corruption]['Vanilla'].values()))
        avg_finetuned = np.mean(list(results[corruption]['Fine-tuned'].values()))
        improvement = avg_finetuned - avg_vanilla
        results[corruption]['improvement'] = improvement
        
        print(f"  Average improvement: {improvement:.2f}%")
    
    # Display and save results
    create_corruption_results_table(results, corruption_types, severities)
    save_results(results, args.output_dir)
    
    # Summary statistics
    print(f"\n📈 SUMMARY:")
    total_improvements = [results[c]['improvement'] for c in corruption_types if 'improvement' in results[c]]
    if total_improvements:
        avg_improvement = np.mean(total_improvements)
        
        print(f"Average improvement across all corruptions: {avg_improvement:.2f}%")
        print(f"Best improvement: {max(total_improvements):.2f}%")
        print(f"Worst improvement: {min(total_improvements):.2f}%")
    else:
        print("No improvement data available")
    
    print(f"\n✅ Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()