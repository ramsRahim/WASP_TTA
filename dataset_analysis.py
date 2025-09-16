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
    p.add_argument("--max-classes", type=int, default=13, help="Maximum number of classes to show samples for") # Updated for 13 classes
    p.add_argument("--figsize", nargs=2, type=int, default=[18, 12], help="Figure size for plots") # Larger figure for more classes
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
    print("-" * 70)
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'%':<6}")
    print("-" * 70)
    
    for class_name in sorted(class_to_idx.keys()):
        tr = train_counts.get(class_name, 0)
        va = val_counts.get(class_name, 0)
        te = test_counts.get(class_name, 0)
        tot = tr + va + te
        pct = 100.0 * tot / total_all
        print(f"{class_name:<20} {tr:<8} {va:<8} {te:<8} {tot:<8} {pct:<6.1f}")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_train:<8} {total_val:<8} {total_test:<8} {total_all:<8} {100.0:<6.1f}")
    print()
    
    # Check for class imbalance
    total_counts = [train_counts.get(cls, 0) + val_counts.get(cls, 0) + test_counts.get(cls, 0) 
                   for cls in sorted(class_to_idx.keys())]
    
    min_samples = min(total_counts)
    max_samples = max(total_counts)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    print(f"CLASS IMBALANCE ANALYSIS:")
    print(f"Min samples per class: {min_samples}")
    print(f"Max samples per class: {max_samples}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("⚠️  Warning: Significant class imbalance detected!")
    else:
        print("✅ Classes are reasonably balanced")
    print()
    
    return train_counts, val_counts, test_counts, class_to_idx

def plot_class_distribution(train_counts, val_counts, test_counts, figsize):
    """Plot class distribution across splits - updated for 13 classes"""
    classes = sorted(train_counts.keys() | val_counts.keys() | test_counts.keys())
    
    train_vals = [train_counts.get(cls, 0) for cls in classes]
    val_vals = [val_counts.get(cls, 0) for cls in classes]
    test_vals = [test_counts.get(cls, 0) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Bar plot
    ax1.bar(x - width, train_vals, width, label='Train', alpha=0.8)
    ax1.bar(x, val_vals, width, label='Validation', alpha=0.8)
    ax1.bar(x + width, test_vals, width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Dataset Distribution Across Splits (13 Classes)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart for total distribution
    total_vals = [t + v + te for t, v, te in zip(train_vals, val_vals, test_vals)]
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    ax2.pie(total_vals, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Overall Class Distribution')
    
    plt.tight_layout()
    plt.savefig('./results/class_distribution_13classes.png', dpi=150, bbox_inches='tight')
    plt.show()

def show_sample_images(manifest, num_samples, max_classes, figsize):
    """Display sample images from each class - updated for 13 classes"""
    class_to_idx = manifest["class_to_idx"]
    all_paths = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    
    # Group paths by class
    class_paths = {}
    for path in all_paths:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class_paths:
            class_paths[class_name] = []
        class_paths[class_name].append(path)
    
    # Select classes to display
    classes_to_show = sorted(class_paths.keys())[:max_classes]
    
    if len(classes_to_show) > 9:
        # For 13 classes, use a larger grid
        rows = 4
        cols = 4
    else:
        rows = int(np.ceil(len(classes_to_show) / 3))
        cols = min(3, len(classes_to_show))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for i, class_name in enumerate(classes_to_show):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Select random samples from this class
        import random
        random.seed(42)  # For reproducible results
        sample_paths = random.sample(class_paths[class_name], 
                                   min(num_samples, len(class_paths[class_name])))
        
        # Create a collage of sample images
        images = []
        for path in sample_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((100, 100))  # Smaller size for display
                images.append(np.array(img))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if images:
            if len(images) == 1:
                collage = images[0]
            else:
                # Create horizontal collage
                collage = np.hstack(images)
            
            ax.imshow(collage)
            ax.set_title(f"{class_name}\n({len(class_paths[class_name])} samples)", fontsize=10)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"{class_name}\n(No images)", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(classes_to_show), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Sample Images from Each Class (showing {len(classes_to_show)} classes)', fontsize=14)
    plt.tight_layout()
    plt.savefig('./results/sample_images_13classes.png', dpi=150, bbox_inches='tight')
    plt.show()

def check_data_integrity(manifest):
    """Check if all files in manifest exist"""
    print("CHECKING DATA INTEGRITY:")
    print("-" * 40)
    
    missing_files = []
    total_files = 0
    
    for split_name, paths in manifest["splits"].items():
        split_missing = []
        for path in paths:
            total_files += 1
            if not os.path.exists(path):
                split_missing.append(path)
                missing_files.append(path)
        
        print(f"{split_name.capitalize()} split: {len(paths)} files, {len(split_missing)} missing")
    
    print(f"\nTotal files: {total_files}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print("⚠️  Warning: Missing files detected!")
        print("First few missing files:")
        for path in missing_files[:5]:
            print(f"  - {path}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    else:
        print("✅ All files exist")
    
    print()

def main():
    args = parse_args()
    
    if not os.path.exists(args.manifest):
        print(f"Error: Manifest file not found: {args.manifest}")
        print("Make sure you've run the training script first to generate the manifest.")
        print("Expected classes: ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'spotted_lanternfly', 'wasp', 'weevil']")
        return
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    manifest = load_manifest(args.manifest)
    
    # Verify we have 13 classes including spotted_lanternfly
    expected_classes = {'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 
                       'grasshopper', 'moth', 'slug', 'snail', 'spotted_lanternfly', 'wasp', 'weevil'}
    actual_classes = set(manifest["class_to_idx"].keys())
    
    print(f"Expected 13 classes, found {len(actual_classes)} classes")
    print(f"Classes found: {sorted(actual_classes)}")
    
    missing_classes = expected_classes - actual_classes
    extra_classes = actual_classes - expected_classes
    
    if missing_classes:
        print(f"⚠️  Missing classes: {missing_classes}")
    if extra_classes:
        print(f"ℹ️  Extra classes: {extra_classes}")
    if actual_classes == expected_classes:
        print("✅ All expected classes found including spotted_lanternfly!")
    print()
    
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