import os
import kaggle
import pandas as pd
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import mediapipe as mp
import argparse

# MediaPipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create necessary directories
DATASET_DIR = Path('data/isl_dataset')
PROCESSED_DIR = Path('data/processed')
RAW_DIR = Path('data/raw')
FEATURES_DIR = Path('data/features/mediapipe')

for dir_path in [DATASET_DIR, PROCESSED_DIR, RAW_DIR, FEATURES_DIR]:
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

# Extract MediaPipe features from video frames
def extract_mediapipe_features(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True
    ) as holistic:
        results = holistic.process(image)
        
        # Extract keypoints
        pose = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            # Fill with zeros if no landmarks detected
            pose = [0.0] * (33 * 4)  # 33 landmarks with x,y,z,visibility
        
        face = []
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                face.extend([landmark.x, landmark.y, landmark.z])
        else:
            face = [0.0] * (478 * 3)  # 478 landmarks with x,y,z
        
        left_hand = []
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                left_hand.extend([landmark.x, landmark.y, landmark.z])
        else:
            left_hand = [0.0] * (21 * 3)  # 21 landmarks with x,y,z
        
        right_hand = []
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                right_hand.extend([landmark.x, landmark.y, landmark.z])
        else:
            right_hand = [0.0] * (21 * 3)  # 21 landmarks with x,y,z
        
        # Combine all features
        keypoints = pose + face + left_hand + right_hand
        return np.array(keypoints, dtype=np.float32)

# Process word-level data
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
    features_dir = FEATURES_DIR / 'word_level'
    features_dir.mkdir(exist_ok=True, parents=True)
    
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
    all_samples = []
    
    # Process each word folder
    for i, cls in enumerate(tqdm(classes)):
        cls_dir = frames_dir / cls
        if not cls_dir.exists():
            continue
            
        # Get all frame sequences
        sequences = []
        for seq_dir in cls_dir.iterdir():
            if seq_dir.is_dir():
                sequences.append(seq_dir)
        
        if not sequences:
            continue
        
        # Process each sequence
        for seq_dir in sequences:
            seq_id = seq_dir.name
            
            # Extract features from each frame in the sequence
            frames = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                frames.extend(list(seq_dir.glob(ext)))
            
            # Sort frames by name
            frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x.name))))
            
            if not frames:
                continue
            
            # Process frames and extract MediaPipe features
            sequence_features = []
            for frame_path in frames:
                features = extract_mediapipe_features(frame_path)
                if features is not None:
                    sequence_features.append(features)
            
            if not sequence_features:
                continue
            
            # Save features
            sequence_features = np.array(sequence_features)
            feature_path = features_dir / f"{cls}_{seq_id}.npy"
            np.save(feature_path, sequence_features)
            
            # Add to samples list with class label
            all_samples.append({
                "id": f"{cls}_{seq_id}",
                "path": str(feature_path.relative_to(FEATURES_DIR)),
                "class": cls,
                "class_idx": class_to_idx[cls],
                "num_frames": len(sequence_features)
            })
    
    # Split samples into train/val/test
    np.random.shuffle(all_samples)
    n_samples = len(all_samples)
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train+n_val]
    test_samples = all_samples[n_train+n_val:]
    
    # Save splits
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    # Generate TSV files for Fairseq
    generate_fairseq_files(train_samples, val_samples, test_samples, output_dir, 'word_level')
    
    print(f"Word-level dataset processed. Total samples: {n_samples}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

# Process sentence-level data
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
    features_dir = FEATURES_DIR / 'sentence_level'
    features_dir.mkdir(exist_ok=True, parents=True)
    
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
    
    # Create samples list
    all_samples = []
    
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
        
        if not sequences:
            continue
        
        # Process each sequence
        for seq_dir in sequences:
            seq_id = seq_dir.name
            
            # Extract features from each frame in the sequence
            frames = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                frames.extend(list(seq_dir.glob(ext)))
            
            # Sort frames by name
            frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x.name))))
            
            if not frames:
                continue
            
            # Process frames and extract MediaPipe features
            sequence_features = []
            for frame_path in frames:
                features = extract_mediapipe_features(frame_path)
                if features is not None:
                    sequence_features.append(features)
            
            if not sequence_features:
                continue
            
            # Save features
            sequence_features = np.array(sequence_features)
            feature_path = features_dir / f"{cls}_{seq_id}.npy"
            np.save(feature_path, sequence_features)
            
            # Add to samples list with text
            all_samples.append({
                "id": f"{cls}_{seq_id}",
                "path": str(feature_path.relative_to(FEATURES_DIR)),
                "text": cls,
                "num_frames": len(sequence_features)
            })
    
    # Split samples into train/val/test
    np.random.shuffle(all_samples)
    n_samples = len(all_samples)
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train+n_val]
    test_samples = all_samples[n_train+n_val:]
    
    # Save splits
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    # Generate TSV files for Fairseq
    generate_fairseq_files(train_samples, val_samples, test_samples, output_dir, 'sentence_level')
    
    print(f"Sentence-level dataset processed. Total samples: {n_samples}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

# Generate Fairseq TSV files and other necessary files
def generate_fairseq_files(train_samples, val_samples, test_samples, output_dir, level):
    # Generate text files with transcriptions for SentencePiece training
    if level == 'sentence_level':
        # For sentence level, extract text from samples
        with open(output_dir / 'train.en', 'w') as f:
            for sample in train_samples:
                f.write(f"{sample['text']}\n")
        
        with open(output_dir / 'val.en', 'w') as f:
            for sample in val_samples:
                f.write(f"{sample['text']}\n")
        
        with open(output_dir / 'test.en', 'w') as f:
            for sample in test_samples:
                f.write(f"{sample['text']}\n")
        
        # Combine for SPM training
        with open(output_dir / 'all.en', 'w') as f:
            for sample in train_samples + val_samples:
                f.write(f"{sample['text']}\n")
    else:
        # For word level, use class names
        with open(output_dir / 'train.en', 'w') as f:
            for sample in train_samples:
                f.write(f"{sample['class']}\n")
        
        with open(output_dir / 'val.en', 'w') as f:
            for sample in val_samples:
                f.write(f"{sample['class']}\n")
        
        with open(output_dir / 'test.en', 'w') as f:
            for sample in test_samples:
                f.write(f"{sample['class']}\n")
        
        # Combine for SPM training
        with open(output_dir / 'all.en', 'w') as f:
            for sample in train_samples + val_samples:
                f.write(f"{sample['class']}\n")
    
    # Generate ID lists
    with open(output_dir / 'train.ids', 'w') as f:
        for sample in train_samples:
            f.write(f"{sample['id']}\n")
    
    with open(output_dir / 'val.ids', 'w') as f:
        for sample in val_samples:
            f.write(f"{sample['id']}\n")
    
    with open(output_dir / 'test.ids', 'w') as f:
        for sample in test_samples:
            f.write(f"{sample['id']}\n")
    
    # Generate TSV files
    generate_tsv(train_samples, output_dir / 'train.tsv')
    generate_tsv(val_samples, output_dir / 'dev.tsv')
    generate_tsv(test_samples, output_dir / 'test.tsv')

# Generate TSV file for Fairseq
def generate_tsv(samples, output_path):
    with open(output_path, 'w') as f:
        for sample in samples:
            if 'text' in sample:
                # For sentence level
                f.write(f"{sample['id']}\t{os.path.join(FEATURES_DIR, sample['path'])}\t{sample['text']}\n")
            else:
                # For word level
                f.write(f"{sample['id']}\t{os.path.join(FEATURES_DIR, sample['path'])}\t{sample['class']}\n")

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
    
    # Create word-level dataset
    word_frames_dir = corpus_dir / 'Frames_Word_Level'
    if word_frames_dir.exists() and word_details_path:
        process_word_level(word_frames_dir, word_details_path)
    
    # Create sentence-level dataset
    sentence_frames_dir = corpus_dir / 'Frames_Sentence_Level'
    if sentence_frames_dir.exists() and sentence_details_path:
        process_sentence_level(sentence_frames_dir, sentence_details_path)

def main():
    parser = argparse.ArgumentParser(description='Prepare ISL dataset for Fairseq sign-to-text model')
    parser.add_argument('--download', action='store_true', help='Download the dataset from Kaggle')
    parser.add_argument('--process', action='store_true', help='Process the dataset after downloading')
    
    args = parser.parse_args()
    
    if args.download:
        download_dataset()
    
    if args.process or not args.download:  # Process by default if no args provided
        process_dataset()
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main() 