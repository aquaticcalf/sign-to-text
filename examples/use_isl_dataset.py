import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.isl_dataloader import get_isl_dataloaders
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_batch(images, labels, classes):
    """Plot a batch of images"""
    # If images is shape [batch, seq_len, channels, height, width]
    if len(images.shape) == 5:
        # Take first frame from each sequence
        images = images[:, 0]
    
    batch_size = images.shape[0]
    fig, axes = plt.subplots(1, min(batch_size, 8), figsize=(15, 3))
    
    for i, ax in enumerate(axes):
        if i >= batch_size:
            break
            
        # Convert tensor to numpy and transpose to HWC
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Display
        ax.imshow(img)
        ax.set_title(f"{classes[labels[i].item()]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_batch.png')
    print(f"Sample batch saved to sample_batch.png")

def main():
    # Get dataloaders
    level = 'word_level'  # or 'sentence_level'
    use_sequences = True  # or False for single frames
    
    dataloaders = get_isl_dataloaders(
        level=level,
        use_sequences=use_sequences,
        batch_size=8,
        max_seq_len=8 if use_sequences else None
    )
    
    # Get a batch from the train dataloader
    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Get class names
    if level == 'word_level':
        mapping_path = Path('data/isl_dataset/word_level/class_mapping.json')
    else:
        mapping_path = Path('data/isl_dataset/sentence_level/class_mapping.json')
    
    import json
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        classes = mapping['classes']
    
    # Plot batch
    plot_batch(images, labels, classes)

if __name__ == "__main__":
    main() 