#!/usr/bin/env python3
import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path to import from main package
sys.path.append(str(Path(__file__).parent.parent))

from eval_tta import SimpleImageDataset, FeatureHook, eval_baseline
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Run Enhanced Multi-Scale TTA Evaluation")
    parser.add_argument('--config', type=str, default='multiScaleFeature/config_enhanced_eval.yaml',
                       help='Path to enhanced evaluation config')
    parser.add_argument('--corruption', type=str, default=None,
                       help='Single corruption to test (if not specified, runs all)')
    parser.add_argument('--severity', type=int, default=None,
                       help='Single severity to test (if not specified, runs all)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    return parser.parse_args()

def load_enhanced_cfg(config_path):
    """Load enhanced configuration with validation"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set defaults for missing keys
    cfg.setdefault('system', {})
    cfg.setdefault('data', {})
    cfg.setdefault('enhanced_tta', {})
    cfg.setdefault('evaluation', {})
    cfg.setdefault('output', {})
    
    # System defaults
    s = cfg['system']
    s.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    s.setdefault('seed', 42)
    s.setdefault('fp16', False)
    
    # Data defaults
    d = cfg['data']
    d.setdefault('manifest_path', './checkpoints/split_manifest.json')
    d.setdefault('ckpt_path', './checkpoints/mobilenetv3_best.pt')
    d.setdefault('img_size', 224)
    d.setdefault('resize', 256)
    d.setdefault('batch_size', 64)
    d.setdefault('num_workers', 4)
    
    return cfg

def get_adaptive_config(corruption_type, severity, cfg):
    """Get adaptive configuration based on corruption type and severity"""
    base_cfg = cfg.copy()
    
    # Get adaptive parameters if they exist
    if 'adaptive_params' in cfg['enhanced_tta']:
        if corruption_type in cfg['enhanced_tta']['adaptive_params']:
            severity_key = f'severity_{severity}'
            if severity_key in cfg['enhanced_tta']['adaptive_params'][corruption_type]:
                # Update TTA parameters with adaptive ones
                adaptive_params = cfg['enhanced_tta']['adaptive_params'][corruption_type][severity_key]
                base_cfg['enhanced_tta'].update(adaptive_params)
    
    return base_cfg

def detect_model_layer_names(model):
    """Automatically detect the correct layer names for feature extraction"""
    layer_names = []
    
    # Check if this is a contrastive model (has backbone attribute)
    if hasattr(model, 'backbone'):
        print("Detected contrastive model with backbone")
        # For MobileNetV3WithContrastive, features are in backbone.features
        base_path = 'backbone.features'
        
        # Check available layers in backbone.features
        features_module = model.backbone.features
        print(f"Available feature layers: {len(features_module)}")
        
        # Select representative layers (similar to original selection)
        total_layers = len(features_module)
        if total_layers >= 16:
            # Use indices similar to original: 3, 6, 12, 16
            selected_indices = [3, 6, 12, 16]
        else:
            # Fallback for models with fewer layers
            selected_indices = [
                max(1, total_layers // 4),
                max(2, total_layers // 2), 
                max(3, 3 * total_layers // 4),
                max(4, total_layers - 1)
            ]
        
        layer_names = [f'{base_path}.{idx}' for idx in selected_indices]
        
    else:
        print("Detected standard model")
        # Standard MobileNetV3 model
        base_path = 'features'
        
        # Check available layers
        features_module = model.features
        total_layers = len(features_module)
        
        if total_layers >= 16:
            selected_indices = [3, 6, 12, 16]
        else:
            selected_indices = [
                max(1, total_layers // 4),
                max(2, total_layers // 2),
                max(3, 3 * total_layers // 4), 
                max(4, total_layers - 1)
            ]
        
        layer_names = [f'{base_path}.{idx}' for idx in selected_indices]
    
    print(f"Auto-detected layer names: {layer_names}")
    return layer_names

class MultiScaleFeatureHook:
    """Enhanced multi-scale feature hook with dynamic attention fusion"""
    def __init__(self, model, layer_names=None):
        self.model = model
        self.features = {}
        self.hooks = []
        self.device = next(model.parameters()).device
        
        # Auto-detect layer names if not provided
        if layer_names is None:
            layer_names = detect_model_layer_names(model)
        
        self.layer_names = layer_names
        self.feature_dims = None  # Will be determined dynamically
        self.attention_initialized = False
        
        # Register hooks
        successful_hooks = 0
        for layer_name in layer_names:
            layer = self._get_layer_by_name(model, layer_name)
            if layer is not None:
                hook = layer.register_forward_hook(self._make_hook_fn(layer_name))
                self.hooks.append(hook)
                successful_hooks += 1
            else:
                print(f"Warning: Could not find layer {layer_name}")
        
        if successful_hooks == 0:
            raise ValueError(f"No valid layers found from {layer_names}")
        
        print(f"Successfully registered {successful_hooks} hooks out of {len(layer_names)} requested")
    
    def _get_layer_by_name(self, model, layer_name):
        """Get layer by dotted name"""
        try:
            attrs = layer_name.split('.')
            layer = model
            for attr in attrs:
                if attr.isdigit():
                    layer = layer[int(attr)]
                else:
                    layer = getattr(layer, attr)
            return layer
        except (AttributeError, IndexError, KeyError):
            print(f"Warning: Layer {layer_name} not found")
            return None
    
    def _make_hook_fn(self, layer_name):
        def hook_fn(module, input, output):
            with torch.no_grad():
                self.features[layer_name] = output.detach()
        return hook_fn
    
    def _init_attention_module(self):
        """Initialize attention module with dynamic feature dimensions"""
        if self.attention_initialized or self.feature_dims is None:
            return
        
        # Attention network: projects each layer to common dimension then computes weights
        self.attention_dim = 256
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.attention_dim) for dim in self.feature_dims
        ])
        
        # Attention scoring network
        self.attention_net = nn.Sequential(
            nn.Linear(self.attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Single attention score per layer
        )
        
        # Final projection to output dimension
        self.output_projection = nn.Linear(self.attention_dim, 512)
        
        # Move to device
        self.projections = self.projections.to(self.device)
        self.attention_net = self.attention_net.to(self.device)
        self.output_projection = self.output_projection.to(self.device)
        
        self.attention_initialized = True
        print(f"Attention module initialized with feature dims: {self.feature_dims}")
    
    def _determine_feature_dimensions(self):
        """Determine feature dimensions by examining current features"""
        if not self.features:
            return None
        
        dims = []
        valid_layers = []
        for layer_name in self.layer_names:
            if layer_name in self.features:
                feat = self.features[layer_name]
                # Global average pooling to get feature dimension
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                dims.append(pooled.shape[1])
                valid_layers.append(layer_name)
        
        if dims:
            self.feature_dims = dims
            self.valid_layer_names = valid_layers
            print(f"Determined feature dimensions: {dict(zip(valid_layers, dims))}")
            return dims
        return None
    
    def get_multi_scale_features(self, fusion_strategy='concat', layer_weights=None):
        """Extract and fuse multi-scale features"""
        if not self.features:
            return None
        
        # Determine feature dimensions if not already done
        if self.feature_dims is None:
            self._determine_feature_dimensions()
        
        if fusion_strategy == 'attention':
            self._init_attention_module()
            if self.attention_initialized:
                return self._attention_fusion()
            else:
                # Fallback to concat if attention init fails
                return self._concat_fusion()
        elif fusion_strategy == 'adaptive_attention':
            self._init_attention_module()
            if self.attention_initialized:
                return self._adaptive_attention_fusion()
            else:
                return self._concat_fusion()
        elif fusion_strategy == 'concat':
            return self._concat_fusion()
        elif fusion_strategy == 'weighted_sum':
            return self._weighted_sum_fusion(layer_weights)
        else:
            return self._concat_fusion()  # Default to concat for compatibility
    
    def _attention_fusion(self):
        """Attention-based fusion of multi-scale features"""
        if not self.attention_initialized:
            return self._concat_fusion()
        
        feature_list = []
        projected_features = []
        
        # Use valid layers that we determined during feature dimension detection
        valid_layers = getattr(self, 'valid_layer_names', self.layer_names)
        
        # Extract and project features from each layer
        for i, layer_name in enumerate(valid_layers):
            if layer_name in self.features and i < len(self.projections):
                feat = self.features[layer_name]
                # Global average pooling
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                
                # Project to common dimension
                try:
                    projected = self.projections[i](pooled)
                    projected_features.append(projected)
                    feature_list.append(pooled)
                except RuntimeError as e:
                    print(f"Projection error for layer {layer_name}: {e}")
                    print(f"Feature shape: {pooled.shape}, Expected: {self.feature_dims[i]}")
                    continue
        
        if not projected_features:
            return self._concat_fusion()
        
        # Stack projected features [batch_size, num_layers, attention_dim]
        try:
            stacked_features = torch.stack(projected_features, dim=1)
            
            # Compute attention weights for each layer
            attention_scores = []
            for i in range(len(projected_features)):
                score = self.attention_net(projected_features[i])  # [batch_size, 1]
                attention_scores.append(score)
            
            # Stack and normalize attention scores
            attention_scores = torch.stack(attention_scores, dim=1)  # [batch_size, num_layers, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Apply attention weights
            attended_features = (stacked_features * attention_weights).sum(dim=1)  # [batch_size, attention_dim]
            
            # Final projection
            output_features = self.output_projection(attended_features)
            
            return output_features
            
        except Exception as e:
            print(f"Attention fusion error: {e}")
            return self._concat_fusion()
    
    def _adaptive_attention_fusion(self):
        """Corruption-adaptive attention fusion"""
        if not self.attention_initialized:
            return self._concat_fusion()
        
        feature_list = []
        projected_features = []
        
        # Use valid layers
        valid_layers = getattr(self, 'valid_layer_names', self.layer_names)
        
        # Extract features
        for i, layer_name in enumerate(valid_layers):
            if layer_name in self.features and i < len(self.projections):
                feat = self.features[layer_name]
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                
                try:
                    projected = self.projections[i](pooled)
                    projected_features.append(projected)
                except RuntimeError:
                    continue
        
        if not projected_features:
            return self._concat_fusion()
        
        try:
            # Compute feature quality/importance for attention
            feature_qualities = []
            for projected in projected_features:
                # Use feature variance and norm as quality indicators
                var = torch.var(projected, dim=1, keepdim=True)
                norm = torch.norm(projected, dim=1, keepdim=True)
                quality = var * norm  # Higher variance and norm = more informative
                feature_qualities.append(quality)
            
            # Combine projected features with quality-based attention
            stacked_features = torch.stack(projected_features, dim=1)
            
            # Modified attention network for combined input
            attention_scores = []
            for i in range(len(projected_features)):
                # Use both feature content and quality for attention
                feature_input = projected_features[i]
                quality_input = feature_qualities[i].expand(-1, self.attention_dim)
                combined = feature_input + 0.1 * quality_input  # Small quality bias
                
                score = self.attention_net(combined)
                attention_scores.append(score)
            
            attention_scores = torch.stack(attention_scores, dim=1)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Apply attention
            attended_features = (stacked_features * attention_weights).sum(dim=1)
            output_features = self.output_projection(attended_features)
            
            return output_features
            
        except Exception as e:
            print(f"Adaptive attention fusion error: {e}")
            return self._concat_fusion()
    
    def _concat_fusion(self):
        """Original concatenation fusion"""
        feature_list = []
        
        # Use all available features (not just the originally requested ones)
        for layer_name in self.layer_names:
            if layer_name in self.features:
                feat = self.features[layer_name]
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                feature_list.append(pooled)
        
        if feature_list:
            result = torch.cat(feature_list, dim=1)
            print(f"Concatenated features shape: {result.shape}")
            return result
        return None
    
    def _weighted_sum_fusion(self, layer_weights=None):
        """Weighted sum of features from different scales"""
        if layer_weights is None:
            layer_weights = [1.0] * len(self.layer_names)
        
        feature_list = []
        for i, layer_name in enumerate(self.layer_names):
            if layer_name in self.features and i < len(layer_weights):
                feat = self.features[layer_name]
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                feature_list.append(pooled * layer_weights[i])
        
        if feature_list:
            # For weighted sum, we need features of same dimension
            # Just return concatenation for now
            return torch.cat(feature_list, dim=1)
        return None
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}

class MultiScalePosNegCache:
    """Enhanced cache with quality tracking"""
    def __init__(self, num_classes, pos_cache_size, neg_cache_size, feature_dim):
        self.num_classes = num_classes
        self.pos_cache_size = pos_cache_size
        self.neg_cache_size = neg_cache_size
        
        # Initialize caches - using same structure as original PosNegCache
        self.pos_cache = [[] for _ in range(num_classes)]
        self.neg_cache = [[] for _ in range(num_classes)]
        self.pos_qualities = [[] for _ in range(num_classes)]
        self.neg_qualities = [[] for _ in range(num_classes)]
    
    def add_positive(self, class_id, feature, quality_score=1.0):
        """Add positive feature with quality score"""
        if len(self.pos_cache[class_id]) < self.pos_cache_size:
            self.pos_cache[class_id].append(feature.cpu())
            self.pos_qualities[class_id].append(quality_score)
        else:
            # Only replace if we have existing samples
            if len(self.pos_qualities[class_id]) > 0:
                min_idx = np.argmin(self.pos_qualities[class_id])
                if quality_score > self.pos_qualities[class_id][min_idx]:
                    self.pos_cache[class_id][min_idx] = feature.cpu()
                    self.pos_qualities[class_id][min_idx] = quality_score
    
    def add_negative(self, class_id, feature, quality_score=1.0):
        """Add negative feature with quality score"""
        if len(self.neg_cache[class_id]) < self.neg_cache_size:
            self.neg_cache[class_id].append(feature.cpu())
            self.neg_qualities[class_id].append(quality_score)
        else:
            # Only replace if we have existing samples
            if len(self.neg_qualities[class_id]) > 0:
                min_idx = np.argmin(self.neg_qualities[class_id])
                if quality_score > self.neg_qualities[class_id][min_idx]:
                    self.neg_cache[class_id][min_idx] = feature.cpu()
                    self.neg_qualities[class_id][min_idx] = quality_score
    
    def get_positive_similarities(self, feature, device):
        """Get similarities with positive samples"""
        similarities = torch.zeros(self.num_classes, device=device)
        
        for class_id in range(self.num_classes):
            if self.pos_cache[class_id]:
                pos_features = torch.stack(self.pos_cache[class_id]).to(device)
                sims = F.cosine_similarity(feature.unsqueeze(0), pos_features)
                similarities[class_id] = sims.mean()
        
        return similarities
    
    def get_negative_similarities(self, feature, device):
        """Get similarities with negative samples"""
        similarities = torch.zeros(self.num_classes, device=device)
        
        for class_id in range(self.num_classes):
            if self.neg_cache[class_id]:
                neg_features = torch.stack(self.neg_cache[class_id]).to(device)
                sims = F.cosine_similarity(feature.unsqueeze(0), neg_features)
                similarities[class_id] = sims.mean()
        
        return similarities
    
    def cache_stats(self):
        """Get cache statistics"""
        pos_total = sum(len(cache) for cache in self.pos_cache)
        neg_total = sum(len(cache) for cache in self.neg_cache)
        
        return {
            'pos_total': pos_total,
            'neg_total': neg_total,
            'pos_avg': pos_total / self.num_classes,
            'neg_avg': neg_total / self.num_classes
        }
    
    def get_average_quality(self):
        """Get average quality scores"""
        all_pos_qualities = [q for qualities in self.pos_qualities for q in qualities]
        all_neg_qualities = [q for qualities in self.neg_qualities for q in qualities]
        
        return {
            'pos_avg_quality': np.mean(all_pos_qualities) if all_pos_qualities else 0.0,
            'neg_avg_quality': np.mean(all_neg_qualities) if all_neg_qualities else 0.0
        }

def compute_feature_quality(feature, corruption_type, prediction_confidence, severity=1):
    """Compute quality score for caching decisions"""
    # Base quality from prediction confidence
    base_quality = prediction_confidence
    
    # Feature statistics
    feature_var = torch.var(feature).item()
    feature_norm = torch.norm(feature).item()
    
    # Variance bonus (informativeness)
    variance_bonus = min(0.3, feature_var * 0.1)
    
    # Norm penalty for extreme values
    if feature_norm > 20.0 or feature_norm < 0.01:
        norm_penalty = 0.2
    else:
        norm_penalty = 0.0
    
    # Corruption-specific adjustments
    corruption_factor = 1.0
    if corruption_type == 'gaussian_noise':
        corruption_factor = max(0.5, 1.0 - 0.1 * severity)
    elif corruption_type in ['brightness', 'contrast']:
        corruption_factor = min(1.2, 1.0 + 0.05 * severity)
    elif corruption_type == 'motion_blur':
        corruption_factor = max(0.7, 1.0 - 0.05 * severity)
    
    # Final quality score
    quality_score = (base_quality + variance_bonus - norm_penalty) * corruption_factor
    
    return max(0.0, min(1.0, quality_score))

def load_model_and_data(cfg):
    """Load model and datasets using the manifest"""
    device = torch.device(cfg['system']['device'])
    
    # Load manifest
    manifest_path = cfg['data']['manifest_path']
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    class_to_idx = manifest['class_to_idx']
    C = len(class_to_idx)
    
    # Check if we have a corruption fine-tuned model
    corruption_checkpoint = manifest.get('corruption_checkpoint')
    use_corruption_model = (
        corruption_checkpoint and 
        os.path.exists(corruption_checkpoint) and 
        manifest.get('corruption_finetuned', False)
    )
    
    if use_corruption_model:
        print(f"Loading corruption fine-tuned model: {corruption_checkpoint}")
        ckpt_path = corruption_checkpoint
    else:
        print(f"Loading standard model: {cfg['data']['ckpt_path']}")
        ckpt_path = cfg['data']['ckpt_path']
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint first to check if it has contrastive components
    checkpoint = torch.load(ckpt_path, map_location=device)
    has_contrastive = checkpoint.get('contrastive_enabled', False)
    
    if has_contrastive:
        print("Model has contrastive learning components")
        # Import the contrastive model class
        sys.path.append(str(Path(__file__).parent.parent))
        from train_finetune import MobileNetV3WithContrastive
        
        feature_dim = checkpoint['cfg']['corruption_finetune'].get('feature_dim', 128)
        model = MobileNetV3WithContrastive(C, feature_dim, freeze_backbone=False)
    else:
        print("Using standard MobileNetV3 model")
        # Standard model
        w = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=w)
        inF = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(inF, C)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {C} classes")
    
    # Create datasets
    val_paths = manifest['splits']['val']
    test_paths = manifest['splits']['test']
    
    val_ds = SimpleImageDataset(val_paths, class_to_idx, 
                               img_size=cfg['data']['img_size'], 
                               resize=cfg['data']['resize'])
    test_ds = SimpleImageDataset(test_paths, class_to_idx,
                                img_size=cfg['data']['img_size'],
                                resize=cfg['data']['resize'])
    
    val_dl = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], 
                       shuffle=False, num_workers=cfg['data']['num_workers'])
    test_dl = DataLoader(test_ds, batch_size=cfg['data']['batch_size'],
                        shuffle=False, num_workers=cfg['data']['num_workers'])
    
    return model, val_dl, test_dl, device, C, class_to_idx

class CorruptionDataset(torch.utils.data.Dataset):
    """Dataset for corrupted images with proper corruption application"""
    def __init__(self, paths, class_to_idx, corruption_type, severity, img_size=224, resize=256):
        self.paths = paths
        self.class_to_idx = class_to_idx
        self.corruption_type = corruption_type
        self.severity = severity
        
        # Create samples list like SimpleImageDataset does
        self.samples = []
        for path in paths:
            cls_name = os.path.basename(os.path.dirname(path))
            if cls_name in class_to_idx:
                self.samples.append((path, class_to_idx[cls_name]))
        
        print(f"Created corrupted dataset with {len(self.samples)} samples")
        print(f"Corruption: {corruption_type}, Severity: {severity}")
    
    def _apply_corruption(self, img, corruption_type, severity):
        """Apply corruption to PIL image - matches train_finetune.py implementation"""
        if corruption_type == 'gaussian_noise':
            return self._add_gaussian_noise(img, severity)
        elif corruption_type == 'motion_blur':
            kernel_size = min(15, 3 + 2 * severity)
            sigma = max(0.5, severity * 0.5)
            return transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        elif corruption_type == 'brightness':
            factor = 0.5 + (severity / 5.0)  # Matches train_finetune.py
            return transforms.functional.adjust_brightness(img, brightness_factor=factor)
        elif corruption_type == 'contrast':
            factor = 0.3 + (severity / 5.0) * 1.4  # Matches train_finetune.py
            return transforms.functional.adjust_contrast(img, contrast_factor=factor)
        else:
            return img
    
    def _add_gaussian_noise(self, img, severity):
        """Add gaussian noise to PIL image - matches train_finetune.py"""
        tensor = transforms.functional.to_tensor(img)
        noise_std = 0.02 + (severity / 5.0) * 0.08  # Matches train_finetune.py
        noise = torch.randn_like(tensor) * noise_std
        noisy_tensor = torch.clamp(tensor + noise, 0, 1)
        return transforms.functional.to_pil_image(noisy_tensor)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load image
        img = Image.open(path).convert('RGB')
        
        # Apply corruption BEFORE other transforms
        if self.corruption_type:
            img = self._apply_corruption(img, self.corruption_type, self.severity)
        
        # Apply standard transforms (resize, crop, normalize)
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensor = tf(img)
        
        return tensor, torch.tensor(label, dtype=torch.long), path

def enhanced_warmup_cache(model, hook, val_dl, device, tta_cfg, C, corruption_type="unknown", corruption_severity=1):
    """Enhanced warmup with corrupted validation data"""
    
    # Get feature dimension by running a dummy forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
        dummy_features = hook.get_multi_scale_features(
            tta_cfg.get("fusion_strategy", "concat"),
            tta_cfg.get("layer_weights", None)
        )
        if dummy_features is None:
            raise ValueError("Could not extract features. Check hook setup.")
        feature_dim = dummy_features.shape[1]
    
    print(f"Feature dimension for cache: {feature_dim}")
    
    # Initialize enhanced cache
    cache = MultiScalePosNegCache(
        C, 
        tta_cfg["pos_cache_size"], 
        tta_cfg["neg_cache_size"], 
        feature_dim
    )
    
    quality_threshold = tta_cfg.get("quality_threshold", 0.05)  # Lower threshold for corrupted data
    
    print(f"Warming up cache for {corruption_type} (severity {corruption_severity})...")
    print(f"Quality threshold: {quality_threshold}")
    print("Using CORRUPTED validation data for warmup...")
    
    # Create corrupted validation dataset for warmup
    # Get validation paths from the dataloader
    val_paths = []
    class_to_idx = {}
    
    # Extract paths and class mapping from validation dataloader
    for batch_idx, (x, y, paths) in enumerate(val_dl):
        for i, path in enumerate(paths):
            val_paths.append(path)
            cls_name = os.path.basename(os.path.dirname(path))
            if cls_name not in class_to_idx:
                class_to_idx[cls_name] = y[i].item()
        if batch_idx == 0:  # Just get the class mapping
            break
    
    # Reset and get all paths
    val_paths = []
    for x, y, paths in val_dl:
        val_paths.extend(paths)
    
    # Create corrupted validation dataset
    corrupted_val_ds = CorruptionDataset(
        val_paths, class_to_idx, corruption_type, corruption_severity
    )
    
    corrupted_val_dl = DataLoader(
        corrupted_val_ds, 
        batch_size=tta_cfg.get("batch_size", 32),  # Smaller batch for warmup
        shuffle=True,  # Shuffle for better diversity
        num_workers=2
    )
    
    print(f"Created corrupted validation dataset with {len(corrupted_val_ds)} samples")
    
    high_quality_count = 0
    low_quality_count = 0
    clean_samples_used = 0
    corrupted_samples_used = 0
    
    # Track samples per class for balanced warmup
    class_counts = [0] * C
    max_per_class = tta_cfg.get("warmup_samples_per_class", 10)
    
    print(f"Target samples per class: {max_per_class}")
    
    with torch.no_grad():
        # Process both clean and corrupted validation data
        print("Processing corrupted validation data...")
        for batch_idx, (x, y, _) in enumerate(tqdm(corrupted_val_dl, desc="Corrupted Warmup")):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            # Get multi-scale features
            features = hook.get_multi_scale_features(
                tta_cfg.get("fusion_strategy", "concat"),
                tta_cfg.get("layer_weights", None)
            )
            
            if features is None:
                continue
            
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)
            
            for i in range(x.size(0)):
                true_class = int(y[i].item())
                pred_class = int(pred[i].item())
                confidence = float(conf[i].item())
                
                # Skip if we have enough samples for this class
                if class_counts[true_class] >= max_per_class:
                    continue
                
                # Compute feature quality
                quality = compute_feature_quality(
                    features[i], corruption_type, confidence, corruption_severity
                )
                
                if quality < quality_threshold:
                    low_quality_count += 1
                    continue
                
                high_quality_count += 1
                corrupted_samples_used += 1
                feature_norm = F.normalize(features[i], dim=0)
                
                # Add to cache based on correctness and confidence
                if pred_class == true_class and confidence >= tta_cfg.get("warmup_tau_pos", 0.8):
                    cache.add_positive(true_class, feature_norm, quality)
                    class_counts[true_class] += 1
                elif pred_class != true_class and confidence >= tta_cfg.get("warmup_tau_neg", 0.2):
                    cache.add_negative(pred_class, feature_norm, quality)
                
                # Also add as negative for incorrect classes
                if confidence >= tta_cfg.get("warmup_tau_neg", 0.2):
                    for c in range(C):
                        if c != true_class:
                            cache.add_negative(c, feature_norm, quality * 0.8)  # Lower quality for negatives
            
            # Check if we have enough samples for all classes
            if all(count >= max_per_class for count in class_counts):
                print("Sufficient samples collected for all classes")
                break
        
        # If we still need more samples, use some clean validation data
        remaining_classes = [i for i, count in enumerate(class_counts) if count < max_per_class]
        if remaining_classes:
            print(f"Need more samples for {len(remaining_classes)} classes, using clean validation data...")
            
            for batch_idx, (x, y, _) in enumerate(tqdm(val_dl, desc="Clean Warmup Supplement")):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                
                features = hook.get_multi_scale_features(
                    tta_cfg.get("fusion_strategy", "concat"),
                    tta_cfg.get("layer_weights", None)
                )
                
                if features is None:
                    continue
                
                probs = F.softmax(logits, dim=1)
                conf, pred = probs.max(1)
                
                for i in range(x.size(0)):
                    true_class = int(y[i].item())
                    pred_class = int(pred[i].item())
                    confidence = float(conf[i].item())
                    
                    # Only process classes that need more samples
                    if true_class not in remaining_classes or class_counts[true_class] >= max_per_class:
                        continue
                    
                    quality = compute_feature_quality(
                        features[i], "clean", confidence, 0
                    )
                    
                    if quality < quality_threshold:
                        continue
                    
                    clean_samples_used += 1
                    feature_norm = F.normalize(features[i], dim=0)
                    
                    if pred_class == true_class and confidence >= tta_cfg.get("warmup_tau_pos", 0.9):
                        cache.add_positive(true_class, feature_norm, quality)
                        class_counts[true_class] += 1
                        
                        if class_counts[true_class] >= max_per_class:
                            remaining_classes.remove(true_class)
                
                if not remaining_classes:
                    break
    
    stats = cache.cache_stats()
    quality_stats = cache.get_average_quality()
    
    print(f"\n=== WARMUP CACHE STATISTICS ===")
    print(f"Cache populated: {stats['pos_total']} positive, {stats['neg_total']} negative features")
    print(f"Avg per class: {stats['pos_avg']:.1f} pos, {stats['neg_avg']:.1f} neg")
    print(f"Corrupted samples used: {corrupted_samples_used}")
    print(f"Clean samples used: {clean_samples_used}")
    print(f"High quality samples: {high_quality_count}")
    print(f"Low quality rejected: {low_quality_count}")
    print(f"Average quality - Pos: {quality_stats['pos_avg_quality']:.3f}, Neg: {quality_stats['neg_avg_quality']:.3f}")
    print(f"Class distribution: {class_counts}")
    print(f"================================\n")
    
    return cache

def eval_enhanced_tta(model, hook, dl, device, tta_cfg, C, pre_warmed_cache=None, corruption_type="unknown"):
    """Enhanced TTA evaluation with detailed debugging"""
    
    if pre_warmed_cache is None:
        raise ValueError("Pre-warmed cache is required for enhanced TTA")
    
    cache = pre_warmed_cache
    
    all_y = []
    all_final_pred = []
    
    # Debug counters
    total_samples = 0
    tta_applied = 0
    low_quality_rejected = 0
    high_confidence_skipped = 0
    cache_updates = 0
    feature_extraction_failures = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(tqdm(dl, desc="Enhanced TTA")):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Get multi-scale features
            features = hook.get_multi_scale_features(
                tta_cfg.get("fusion_strategy", "concat"),
                tta_cfg.get("layer_weights", None)
            )
            
            if features is None:
                feature_extraction_failures += x.size(0)
                _, pred = logits.max(1)
                all_y.extend(y.cpu().numpy())
                all_final_pred.extend(pred.cpu().numpy())
                continue
            
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)
            final_pred = pred.clone()
            
            for i in range(x.size(0)):
                total_samples += 1
                
                # Compute feature quality
                quality = compute_feature_quality(
                    features[i], corruption_type, conf[i].item()
                )
                
                if quality < tta_cfg.get("quality_threshold", 0.05):
                    low_quality_rejected += 1
                    continue
                
                # Check confidence for TTA application
                if conf[i] < tta_cfg.get("tau_ref", 0.6):
                    # Apply TTA
                    feature_norm = F.normalize(features[i], dim=0)
                    pos_sims = cache.get_positive_similarities(feature_norm, device)
                    neg_sims = cache.get_negative_similarities(feature_norm, device)
                    
                    # Check if we have any cached features
                    if pos_sims.max() > 0 or neg_sims.max() > 0:
                        pos_adjustment = tta_cfg.get("beta", 3.0) * pos_sims
                        neg_adjustment = tta_cfg.get("gamma", 2.0) * neg_sims
                        
                        adjusted_logits = logits[i] + pos_adjustment - neg_adjustment
                        final_pred[i] = adjusted_logits.argmax()
                        tta_applied += 1
                else:
                    high_confidence_skipped += 1
                
                # Update cache
                if conf[i] >= tta_cfg.get("tau_pos", 0.8):
                    cache.add_positive(int(final_pred[i].item()), F.normalize(features[i], dim=0), quality)
                    cache_updates += 1
                elif conf[i] <= tta_cfg.get("tau_neg", 0.2):
                    cache.add_negative(int(final_pred[i].item()), F.normalize(features[i], dim=0), quality)
                    cache_updates += 1
            
            all_y.extend(y.cpu().numpy())
            all_final_pred.extend(final_pred.cpu().numpy())
    
    # Print debug statistics
    print(f"\n=== DEBUG STATISTICS for {corruption_type} ===")
    print(f"Total samples: {total_samples}")
    print(f"TTA applied: {tta_applied} ({100*tta_applied/total_samples:.1f}%)")
    print(f"High confidence skipped: {high_confidence_skipped} ({100*high_confidence_skipped/total_samples:.1f}%)")
    print(f"Low quality rejected: {low_quality_rejected} ({100*low_quality_rejected/total_samples:.1f}%)")
    print(f"Feature extraction failures: {feature_extraction_failures}")
    print(f"Cache updates: {cache_updates}")
    
    # Final stats
    cache_stats = cache.cache_stats()
    print(f"Final cache: {cache_stats['pos_total']} pos, {cache_stats['neg_total']} neg")
    
    # Compute accuracy
    all_y = np.array(all_y)
    all_final_pred = np.array(all_final_pred)
    accuracy = 100.0 * (all_y == all_final_pred).mean()
    
    return accuracy

def create_corrupted_dataloader(manifest, class_to_idx, corruption_type, severity, cfg):
    """Create dataloader for corrupted test data"""
    test_paths = manifest['splits']['test']
    
    corrupted_ds = CorruptionDataset(
        test_paths, class_to_idx, corruption_type, severity,
        cfg['data']['img_size'], cfg['data']['resize']
    )
    
    corrupted_dl = DataLoader(
        corrupted_ds, 
        batch_size=cfg['data']['batch_size'],
        shuffle=False, 
        num_workers=cfg['data']['num_workers']
    )
    
    return corrupted_dl

def evaluate_single_corruption(model, val_dl, manifest, class_to_idx, device, C, 
                             corruption_type, severity, cfg):
    """Evaluate single corruption with enhanced TTA"""
    
    print(f"\n{'='*60}")
    print(f"Testing {corruption_type} at severity {severity}")
    print(f"{'='*60}")
    
    # Get adaptive configuration
    adaptive_cfg = get_adaptive_config(corruption_type, severity, cfg)
    tta_cfg = adaptive_cfg['enhanced_tta']
    
    # Create corrupted test dataloader
    corrupted_dl = create_corrupted_dataloader(
        manifest, class_to_idx, corruption_type, severity, cfg
    )
    
    # Initialize multi-scale hook (with auto-detection for contrastive models)
    hook = MultiScaleFeatureHook(model)  # Let it auto-detect layer names
    
    try:
        # Warmup cache with CORRUPTED validation data
        cache = enhanced_warmup_cache(
            model, hook, val_dl, device, tta_cfg, C, corruption_type, severity
        )
        
        # Evaluate with enhanced TTA
        print("Running enhanced TTA evaluation...")
        tta_acc = eval_enhanced_tta(
            model, hook, corrupted_dl, device, tta_cfg, C, cache, corruption_type
        )
        
        # Baseline evaluation on corrupted data
        print("Evaluating baseline on corrupted data...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in corrupted_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                _, pred = logits.max(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        baseline_acc = 100.0 * correct / total
        delta = tta_acc - baseline_acc
        
        # Print results
        print(f"\nRESULTS:")
        print(f"Enhanced TTA Top-1: {tta_acc:.2f}%")
        print(f"Baseline Top-1: {baseline_acc:.2f}%")
        print(f"Delta: {delta:+.2f}%")
        print(f"Corruption: {corruption_type}, Severity: {severity}")
        
        return {
            'corruption': corruption_type,
            'severity': severity,
            'tta_acc': tta_acc,
            'baseline_acc': baseline_acc,
            'delta': delta,
            'timestamp': datetime.now().isoformat()
        }
        
    finally:
        # Cleanup hooks
        hook.cleanup()

def run_enhanced_evaluation(cfg, args):
    """Main evaluation loop"""
    
    # Load model and data
    print("Loading model and datasets...")
    model, val_dl, test_dl, device, C, class_to_idx = load_model_and_data(cfg)
    
    # Load manifest for corrupted data creation
    with open(cfg['data']['manifest_path'], 'r') as f:
        manifest = json.load(f)
    
    # Determine which corruptions and severities to test
    if args.corruption and args.severity:
        test_combinations = [(args.corruption, args.severity)]
    elif args.corruption:
        test_combinations = [(args.corruption, s) for s in cfg['evaluation']['severities']]
    elif args.severity:
        test_combinations = [(c, args.severity) for c in cfg['evaluation']['corruptions']]
    else:
        test_combinations = [(c, s) for c in cfg['evaluation']['corruptions'] 
                           for s in cfg['evaluation']['severities']]
    
    # Run evaluations
    all_results = []
    
    for corruption_type, severity in test_combinations:
        try:
            result = evaluate_single_corruption(
                model, val_dl, manifest, class_to_idx, device, C,
                corruption_type, severity, cfg
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"Error evaluating {corruption_type} severity {severity}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if cfg['evaluation']['save_results']:
        os.makedirs(args.output_dir, exist_ok=True)
        
        results_file = os.path.join(args.output_dir, cfg['evaluation']['results_file'])
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("ENHANCED TTA EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"{result['corruption']:<15} Sev{result['severity']}: "
              f"TTA {result['tta_acc']:5.2f}% | "
              f"Base {result['baseline_acc']:5.2f}% | "
              f"Δ {result['delta']:+5.2f}%")
    
    # Calculate average improvement
    avg_delta = sum(r['delta'] for r in all_results) / len(all_results) if all_results else 0
    print(f"\nAverage improvement: {avg_delta:+.2f}%")
    
    return all_results

def main():
    args = parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Please create the config file first.")
        return
    
    cfg = load_enhanced_cfg(args.config)
    
    # Set random seed
    torch.manual_seed(cfg['system']['seed'])
    
    print("Starting Enhanced Multi-Scale TTA Evaluation")
    print(f"Config: {args.config}")
    print(f"Device: {cfg['system']['device']}")
    
    # Run evaluation
    try:
        results = run_enhanced_evaluation(cfg, args)
        print("\nEnhanced evaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()