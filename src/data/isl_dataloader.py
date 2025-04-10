import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISLFrameDataset(Dataset):
    """Dataset for single frame classification from ISL dataset"""
    def __init__(self, data_dir, split='train', level='word_level', transform=None, max_seq_len=20):
        self.data_dir = Path(data_dir) / level
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load class mapping
        with open(self.data_dir / 'class_mapping.json', 'r') as f:
            mapping = json.load(f)
            self.classes = mapping['classes']
            self.class_to_idx = mapping['class_to_idx']
        
        # Load split data
        with open(self.data_dir / 'splits.json', 'r') as f:
            splits = json.load(f)
            self.samples = splits[split]
        
        # Set up default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.RandomResizedCrop(224, 224),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        
        # Construct the full path
        seq_dir = self.data_dir / self.split / path
        
        # Load frames
        frames = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            frames.extend(list(seq_dir.glob(ext)))
        
        # Sort frames
        frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x.name))))
        
        # Select frames (uniformly sample if too many)
        if len(frames) > self.max_seq_len:
            indices = np.linspace(0, len(frames)-1, self.max_seq_len, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Read and process frames
        processed_frames = []
        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]
                
            processed_frames.append(img)
        
        # Make sure we have at least one frame
        if not processed_frames:
            # Create a blank frame if no frames found
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                transformed = self.transform(image=blank)
                blank = transformed["image"]
            processed_frames.append(blank)
        
        # Pad sequence if needed
        while len(processed_frames) < self.max_seq_len:
            processed_frames.append(torch.zeros_like(processed_frames[0]))
        
        # Stack frames into a tensor
        frames_tensor = torch.stack(processed_frames[:self.max_seq_len])
        
        # Return frames tensor and class index
        return frames_tensor, class_idx

class ISLSingleFrameDataset(Dataset):
    """Dataset for single frame classification from ISL dataset"""
    def __init__(self, data_dir, split='train', level='word_level', transform=None):
        self.data_dir = Path(data_dir) / level
        self.split = split
        
        # Load class mapping
        with open(self.data_dir / 'class_mapping.json', 'r') as f:
            mapping = json.load(f)
            self.classes = mapping['classes']
            self.class_to_idx = mapping['class_to_idx']
        
        # Load split data
        with open(self.data_dir / 'splits.json', 'r') as f:
            splits = json.load(f)
            self.sequence_samples = splits[split]
        
        # Expand sequence samples to individual frames
        self.samples = []
        for path, class_idx in self.sequence_samples:
            seq_dir = self.data_dir / self.split / path
            for frame_path in seq_dir.glob('*.jpg'):
                self.samples.append((str(frame_path.relative_to(self.data_dir / self.split)), class_idx))
        
        # Set up default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.RandomResizedCrop(224, 224),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        
        # Construct the full path
        full_path = self.data_dir / self.split / path
        
        # Read image
        img = cv2.imread(str(full_path))
        if img is None:
            raise ValueError(f"Could not read image {full_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        
        return img, class_idx

def get_isl_dataloaders(data_dir='data/isl_dataset', level='word_level', batch_size=32, 
                        num_workers=4, use_sequences=True, max_seq_len=20):
    """Get dataloaders for ISL dataset
    
    Args:
        data_dir: Path to the processed ISL dataset
        level: 'word_level' or 'sentence_level'
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        use_sequences: If True, use sequence dataset, else use single frame dataset
        max_seq_len: Maximum sequence length for sequence dataset
    """
    # Choose dataset class based on whether we want sequences or single frames
    dataset_class = ISLFrameDataset if use_sequences else ISLSingleFrameDataset
    
    # Create datasets
    train_dataset = dataset_class(
        data_dir, 
        split='train', 
        level=level,
        max_seq_len=max_seq_len if use_sequences else None
    )
    
    val_dataset = dataset_class(
        data_dir, 
        split='val', 
        level=level,
        max_seq_len=max_seq_len if use_sequences else None
    )
    
    test_dataset = dataset_class(
        data_dir, 
        split='test', 
        level=level,
        max_seq_len=max_seq_len if use_sequences else None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader} 