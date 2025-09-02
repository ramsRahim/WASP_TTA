#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Analyze dataset details and view sample images")
    p.add_argument("--manifest", default="./checkpoints/split_manifest.json", help="Path to split manifest")
    p.add_argument("--show-samples", type=int, default=3, help="Number of sample images per class to show")
    p.add_argument("--max-classes", type=int, default=5, help="Maximum number of classes to show samples for")
    p.add_argument("--figsize", nargs=2, type=int, default=[15, 10], help="Figure size for plots")
    return p.parse_args()

def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)

def analyze_splits(manifest):
    """Analyze the dataset splits and class distribution"""
    splits = manifest["splits"]
    class_to_idx = manifest["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    total_train = len(splits["train"])
    total_val = len(splits["val"])
    total_test = len(splits["test"])
    total_all = total_train + total_val + total_test
    
    print(f"Total samples: {total_all}")
    print(f"Training samples: {total_train} ({100*total_train/total_all:.1f}%)")
    print(f"Validation samples: {total_val} ({100*total_val/total_all:.1f}%)")
    print(f"Test samples: {total_test} ({100*total_test/total_all:.1f}%)")
    print(f"Number of classes: {len(class_to_idx)}")
    print()
    
    # Class distribution analysis
    def get_class_counts(paths):
        counts = Counter()
        for path in paths:
            class_name = os.path.basename(os.path.dirname(path))
            counts[class_name] += 1
        return counts
    
    train_counts = get_class_counts(splits["train"])
    val_counts = get_class_counts(splits["val"])
    test_counts = get_class_counts(splits["test"])
    
    print("CLASS DISTRIBUTION:")
    print("-" * 60)
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)
    
    for class_name in sorted(class_to_idx.keys()):
        tr = train_counts.get(class_name, 0)
        va = val_counts.get(class_name, 0)
        te = test_counts.get(class_name, 0)
        tot = tr + va + te
        print(f"{class_name:<20} {tr:<8} {va:<8} {te:<8} {tot:<8}")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_train:<8} {total_val:<8} {total_test:<8} {total_all:<8}")
    print()
    
    return train_counts, val_counts, test_counts, class_to_idx

def plot_class_distribution(train_counts, val_counts, test_counts, figsize):
    """Plot class distribution across splits"""
    classes = sorted(train_counts.keys())
    train_vals = [train_counts.get(c, 0) for c in classes]
    val_vals = [val_counts.get(c, 0) for c in classes]
    test_vals = [test_counts.get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, train_vals, width, label='Train', alpha=0.8)
    ax.bar(x, val_vals, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_vals, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Distribution Across Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def show_sample_images(manifest, num_samples, max_classes, figsize):
    """Display sample images from each class"""
    splits = manifest["splits"]
    class_to_idx = manifest["class_to_idx"]
    
    # Get sample paths for each class from training set
    class_samples = {}
    for path in splits["train"]:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class_samples:
            class_samples[class_name] = []
        class_samples[class_name].append(path)
    
    # Limit to max_classes
    classes_to_show = sorted(class_samples.keys())[:max_classes]
    
    fig, axes = plt.subplots(len(classes_to_show), num_samples, figsize=figsize)
    if len(classes_to_show) == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, class_name in enumerate(classes_to_show):
        sample_paths = class_samples[class_name][:num_samples]
        
        for j, path in enumerate(sample_paths):
            try:
                img = Image.open(path).convert('RGB')
                if j < len(axes[i]):
                    axes[i][j].imshow(img)
                    axes[i][j].set_title(f"{class_name}\n{os.path.basename(path)}", fontsize=8)
                    axes[i][j].axis('off')
            except Exception as e:
                print(f"Error loading {path}: {e}")
                if j < len(axes[i]):
                    axes[i][j].text(0.5, 0.5, 'Error\nloading\nimage', 
                                   ha='center', va='center', transform=axes[i][j].transAxes)
                    axes[i][j].axis('off')
        
        # Hide empty subplots
        for j in range(len(sample_paths), num_samples):
            if j < len(axes[i]):
                axes[i][j].axis('off')
    
    plt.suptitle(f'Sample Images (showing {len(classes_to_show)} classes)', fontsize=14)
    plt.tight_layout()
    plt.show()

def check_data_integrity(manifest):
    """Check if all files in manifest exist"""
    print("DATA INTEGRITY CHECK:")
    print("-" * 30)
    
    missing_files = []
    for split_name, paths in manifest["splits"].items():
        missing_in_split = []
        for path in paths:
            if not os.path.exists(path):
                missing_in_split.append(path)
        
        if missing_in_split:
            missing_files.extend(missing_in_split)
            print(f"{split_name}: {len(missing_in_split)} missing files")
        else:
            print(f"{split_name}: All {len(paths)} files exist ✓")
    
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} total missing files!")
        print("First few missing files:")
        for path in missing_files[:5]:
            print(f"  {path}")
    else:
        print(f"\n✓ All files exist and are accessible")
    
    print()

def main():
    args = parse_args()
    
    if not os.path.exists(args.manifest):
        print(f"Error: Manifest file not found: {args.manifest}")
        print("Make sure you've run the training script first to generate the manifest.")
        return
    
    manifest = load_manifest(args.manifest)
    
    # Analyze dataset
    train_counts, val_counts, test_counts, class_to_idx = analyze_splits(manifest)
    
    # Check data integrity
    check_data_integrity(manifest)
    
    # Plot distribution
    print("Generating class distribution plot...")
    plot_class_distribution(train_counts, val_counts, test_counts, tuple(args.figsize))
    
    # Show sample images
    if args.show_samples > 0:
        print(f"Showing {args.show_samples} sample images per class...")
        show_sample_images(manifest, args.show_samples, args.max_classes, tuple(args.figsize))

if __name__ == "__main__":
    main()