import os
import kaggle
import pandas as pd
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# Create necessary directories
DATASET_DIR = Path('data/isl_dataset')
PROCESSED_DIR = Path('data/processed')
RAW_DIR = Path('data/raw')

for dir_path in [DATASET_DIR, PROCESSED_DIR, RAW_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Download the dataset using Kaggle API
def download_dataset():
    print("Downloading ISL dataset from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'drblack00/isl-csltr-indian-sign-language-dataset',
        path=RAW_DIR,
        unzip=True
    )
    print("Download complete.")

# Process the dataset for use in the sign-to-text model
def process_dataset():
    print("Processing ISL dataset...")
    
    # Check for dataset paths
    corpus_dir = RAW_DIR / 'ISL_CSLRT_Corpus'
    if not corpus_dir.exists():
        potential_paths = list(RAW_DIR.glob('*'))
        for path in potential_paths:
            if 'ISL' in path.name and path.is_dir():
                corpus_dir = path
                break
        if not corpus_dir.exists():
            raise FileNotFoundError("Could not find ISL_CSLRT_Corpus directory")
    
    # Process metadata files
    metadata_dir = corpus_dir / 'corpus_csv_files'
    word_details_path = next(metadata_dir.glob('*word_details*'), None)
    sentence_details_path = next(metadata_dir.glob('*frame_details*'), None)
    corpus_details_path = next(metadata_dir.glob('*Corpus_details*'), None)
    
    # Create word-level dataset
    word_frames_dir = corpus_dir / 'Frames_Word_Level'
    if word_frames_dir.exists() and word_details_path:
        process_word_level(word_frames_dir, word_details_path)
    
    # Create sentence-level dataset
    sentence_frames_dir = corpus_dir / 'Frames_Sentence_Level'
    if sentence_frames_dir.exists() and sentence_details_path:
        process_sentence_level(sentence_frames_dir, sentence_details_path)

def process_word_level(frames_dir, metadata_path):
    print("Processing word-level data...")
    
    # Read metadata
    if metadata_path.suffix.lower() == '.xlsx':
        metadata = pd.read_excel(metadata_path)
    else:
        metadata = pd.read_csv(metadata_path)
    
    # Create output directories
    output_dir = DATASET_DIR / 'word_level'
    output_dir.mkdir(exist_ok=True)
    
    # Get unique words/classes
    if 'Word' in metadata.columns:
        classes = metadata['Word'].unique().tolist()
    elif 'sign' in metadata.columns:
        classes = metadata['sign'].unique().tolist()
    else:
        classes = []
        for folder in frames_dir.iterdir():
            if folder.is_dir():
                classes.append(folder.name)
    
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # Save class mapping
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f)
    
    # Create train/val/test splits
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Process each word folder
    for cls in tqdm(classes):
        cls_dir = frames_dir / cls
        if not cls_dir.exists():
            continue
            
        # Get all frame sequences
        sequences = []
        for seq_dir in cls_dir.iterdir():
            if seq_dir.is_dir():
                sequences.append(seq_dir)
        
        # Shuffle sequences
        np.random.shuffle(sequences)
        
        # Split: 80% train, 10% val, 10% test
        n_train = int(0.8 * len(sequences))
        n_val = int(0.1 * len(sequences))
        
        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train:n_train+n_val]
        test_seqs = sequences[n_train+n_val:]
        
        # Process sequences
        for seq_dir in train_seqs:
            seq_id = process_sequence(seq_dir, output_dir / 'train' / cls, class_to_idx[cls])
            train_samples.append((f"{cls}/{seq_id}", class_to_idx[cls]))
            
        for seq_dir in val_seqs:
            seq_id = process_sequence(seq_dir, output_dir / 'val' / cls, class_to_idx[cls])
            val_samples.append((f"{cls}/{seq_id}", class_to_idx[cls]))
            
        for seq_dir in test_seqs:
            seq_id = process_sequence(seq_dir, output_dir / 'test' / cls, class_to_idx[cls])
            test_samples.append((f"{cls}/{seq_id}", class_to_idx[cls]))
    
    # Save splits
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    print(f"Word-level dataset processed. Total classes: {len(classes)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

def process_sentence_level(frames_dir, metadata_path):
    print("Processing sentence-level data...")
    
    # Read metadata
    if metadata_path.suffix.lower() == '.xlsx':
        metadata = pd.read_excel(metadata_path)
    else:
        metadata = pd.read_csv(metadata_path)
    
    # Create output directories
    output_dir = DATASET_DIR / 'sentence_level'
    output_dir.mkdir(exist_ok=True)
    
    # Get unique sentences/classes
    if 'Sentence' in metadata.columns:
        classes = metadata['Sentence'].unique().tolist()
    else:
        classes = []
        for folder in frames_dir.iterdir():
            if folder.is_dir():
                classes.append(folder.name)
    
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # Save class mapping
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f)
    
    # Create train/val/test splits
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Process each sentence folder
    for cls in tqdm(classes):
        cls_dir = frames_dir / cls
        if not cls_dir.exists():
            # Try using a cleaned/simplified name
            for potential_dir in frames_dir.iterdir():
                if potential_dir.is_dir() and cls.lower() in potential_dir.name.lower():
                    cls_dir = potential_dir
                    break
            if not cls_dir.exists():
                continue
            
        # Get all frame sequences
        sequences = []
        for seq_dir in cls_dir.iterdir():
            if seq_dir.is_dir():
                sequences.append(seq_dir)
        
        # Shuffle sequences
        np.random.shuffle(sequences)
        
        # Split: 80% train, 10% val, 10% test
        n_train = int(0.8 * len(sequences))
        n_val = int(0.1 * len(sequences))
        
        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train:n_train+n_val]
        test_seqs = sequences[n_train+n_val:]
        
        # Process sequences
        for seq_dir in train_seqs:
            seq_id = process_sequence(seq_dir, output_dir / 'train' / cls, class_to_idx[cls])
            train_samples.append((f"{cls}/{seq_id}", class_to_idx[cls]))
            
        for seq_dir in val_seqs:
            seq_id = process_sequence(seq_dir, output_dir / 'val' / cls, class_to_idx[cls])
            val_samples.append((f"{cls}/{seq_id}", class_to_idx[cls]))
            
        for seq_dir in test_seqs:
            seq_id = process_sequence(seq_dir, output_dir / 'test' / cls, class_to_idx[cls])
            test_samples.append((f"{cls}/{seq_id}", class_to_idx[cls]))
    
    # Save splits
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    print(f"Sentence-level dataset processed. Total classes: {len(classes)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

def process_sequence(seq_dir, output_dir, class_idx):
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_id = seq_dir.name
    seq_output_dir = output_dir / seq_id
    seq_output_dir.mkdir(exist_ok=True)
    
    # Get all frames in order
    frames = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        frames.extend(list(seq_dir.glob(ext)))
    
    # Sort frames by name (assuming numerical ordering)
    frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x.name))))
    
    # Process each frame
    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        # Resize to standard size
        img = cv2.resize(img, (224, 224))
        
        # Save processed frame
        output_path = seq_output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(output_path), img)
    
    # Create metadata.json with sequence info
    metadata = {
        "num_frames": len(frames),
        "class_idx": class_idx
    }
    
    with open(seq_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return seq_id

def main():
    download_dataset()
    process_dataset()
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main() 