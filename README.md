
# Sign Language Translation with ISL-CSLRT Dataset on Google Colab

This guide walks you through setting up and training a sign language translation model using the Indian Sign Language (ISL-CSLRT) dataset on Google Colab.

## Table of Contents

1. [Setting Up Google Colab](#setting-up-google-colab)
2. [Installing Dependencies](#installing-dependencies)
3. [Downloading and Processing the Dataset](#downloading-and-processing-the-dataset)
4. [Training Tokenizers](#training-tokenizers)
5. [Preprocessing Data for Fairseq](#preprocessing-data-for-fairseq)
6. [Training the Model](#training-the-model)
7. [Evaluating the Model](#evaluating-the-model)
8. [Converting to TFLite](#converting-to-tflite)
9. [Troubleshooting](#troubleshooting)

## Setting Up Google Colab

1. Open a new notebook on [Google Colab](https://colab.research.google.com)

2. Enable GPU acceleration:
   ```
   Runtime > Change runtime type > Hardware accelerator > GPU
   ```

3. Mount Google Drive to save your data and models:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   %mkdir -p /content/drive/MyDrive/sign-to-text
   %cd /content/drive/MyDrive/sign-to-text
   ```

## Installing Dependencies

Run the following commands to install all required dependencies:

```python
# Install fairseq
!pip install fairseq

# Install other dependencies
!pip install kaggle sentencepiece mediapipe opencv-python numpy pandas matplotlib tqdm openpyxl

# For handling Kaggle API
!pip install --upgrade kaggle
```

## Downloading and Processing the Dataset

### 1. Set up Kaggle API credentials

```python
# Upload your kaggle.json file to Colab or input your credentials
import os
os.makedirs('/root/.kaggle', exist_ok=True)

# Option 1: Upload from local machine
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json file
!cp kaggle.json /root/.kaggle/

# Option 2: Create from credentials
'''
!echo '{
  "username":"YOUR_KAGGLE_USERNAME",
  "key":"YOUR_KAGGLE_API_KEY"
}' > /root/.kaggle/kaggle.json
'''

# Set permissions
!chmod 600 /root/.kaggle/kaggle.json
```

### 2. Create separated download and process scripts

Since the dataset download might time out in Colab, let's create separate scripts for download and processing:

```python
%%writefile download_isl_dataset.py
import os
import time
import subprocess
import kaggle
from pathlib import Path

# Create raw data directory
RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_with_retry():
    """Download the dataset with retry logic"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt+1}/{max_retries}...")
            
            # Authenticate with Kaggle
            kaggle.api.authenticate()
            
            # Download in smaller chunks if possible
            dataset_slug = 'drblack00/isl-csltr-indian-sign-language-dataset'
            
            # Try to use the kaggle CLI directly instead of the API
            cmd = f"kaggle datasets download {dataset_slug} -p {RAW_DIR} --unzip"
            
            # Run with timeout
            process = subprocess.run(
                cmd, 
                shell=True,
                timeout=600  # 10 minute timeout
            )
            
            if process.returncode == 0:
                print("Download completed successfully!")
                return True
                
        except subprocess.TimeoutExpired:
            print(f"Download timed out on attempt {attempt+1}")
        except Exception as e:
            print(f"Error during download: {e}")
        
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    return False

if __name__ == "__main__":
    success = download_with_retry()
    
    if success:
        print("Dataset downloaded successfully. Now you can run:")
        print("python process_isl_dataset.py")
    else:
        print("Failed to download the dataset after multiple attempts.")
        print("Try manually downloading from: https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset")
```

Now create the processing script:

```python
%%writefile process_isl_dataset.py
# Same as prepare_isl_dataset_fairseq.py but without the download functionality

import os
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

# Extract MediaPipe features from video frames
def extract_mediapipe_features(image_path):
    # Implementation from the original script
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

# Process word-level and sentence-level data functions from original script
# ...

# Process dataset function from original script
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
            raise FileNotFoundError("Could not find ISL_CSLRT_Corpus directory. Please ensure dataset is downloaded.")
    
    # Process metadata files
    metadata_dir = corpus_dir / 'corpus_csv_files'
    word_details_path = next(metadata_dir.glob('*word_details*'), None)
    sentence_details_path = next(metadata_dir.glob('*frame_details*'), None)
    
    # Create word-level dataset
    word_frames_dir = corpus_dir / 'Frames_Word_Level'
    if word_frames_dir.exists() and word_details_path:
        process_word_level(word_frames_dir, word_details_path)
    else:
        print("Word level data not found or metadata missing")
    
    # Create sentence-level dataset
    sentence_frames_dir = corpus_dir / 'Frames_Sentence_Level'
    if sentence_frames_dir.exists() and sentence_details_path:
        process_sentence_level(sentence_frames_dir, sentence_details_path)
    else:
        print("Sentence level data not found or metadata missing")

if __name__ == "__main__":
    process_dataset()
    print("Dataset processing complete!")
```

### 3. Download and process the dataset with alternative methods

Option 1: Use the separated scripts (recommended):
```python
# First try to download with retry logic
!python download_isl_dataset.py

# After download completes, process the data
!python process_isl_dataset.py
```

Option 2: Manual download and upload:
```python
# If download keeps failing, manually download from Kaggle and upload to Drive
from google.colab import drive
drive.mount('/content/drive')

# Create directory
!mkdir -p data/raw

# Copy from Drive (adjust path to your uploaded file location)
!cp /content/drive/MyDrive/path/to/isl-csltr-indian-sign-language-dataset.zip data/raw/

# Extract
!unzip data/raw/isl-csltr-indian-sign-language-dataset.zip -d data/raw/

# Process (without download)
!python process_isl_dataset.py
```

Option 3: For handling large dataset sizes, process in smaller chunks:
```python
# Create a script to process only word level data
%%writefile process_word_only.py
# Modify process_dataset to only handle word level data
# ...

# Run it
!python process_word_only.py
```

## Training Tokenizers

Train SentencePiece models for both word and sentence levels:

```python
# For word-level
!python train_spm.py --input data/isl_dataset/word_level/all.en --model-prefix data/isl_dataset/word_level/spm_bpe1000 --vocab-size 1000

# For sentence-level
!python train_spm.py --input data/isl_dataset/sentence_level/all.en --model-prefix data/isl_dataset/sentence_level/spm_bpe4000 --vocab-size 4000
```

## Preprocessing Data for Fairseq

Create a dummy source dictionary and preprocess the data:

```python
# Create dummy source dictionary
!echo "<UNUSED> 0" > ./dummy_dict.txt
!echo "<PAD> 1" >> ./dummy_dict.txt
!echo "<EOS> 2" >> ./dummy_dict.txt
!echo "<UNK> 3" >> ./dummy_dict.txt

# Clone fairseq repository to get example files
!git clone https://github.com/facebookresearch/fairseq.git
!cp -r fairseq/examples/sign_language ./examples/

# For word-level
!fairseq-preprocess --task sign_to_text \
    --source-lang sign --target-lang en \
    --trainpref data/isl_dataset/word_level/train.tsv \
    --validpref data/isl_dataset/word_level/dev.tsv \
    --testpref data/isl_dataset/word_level/test.tsv \
    --destdir data-bin/isl_word_level \
    --config mediapipe_config.yaml \
    --srcdict ./dummy_dict.txt \
    --tgtdict data/isl_dataset/word_level/spm_bpe1000.vocab

# For sentence-level
!fairseq-preprocess --task sign_to_text \
    --source-lang sign --target-lang en \
    --trainpref data/isl_dataset/sentence_level/train.tsv \
    --validpref data/isl_dataset/sentence_level/dev.tsv \
    --testpref data/isl_dataset/sentence_level/test.tsv \
    --destdir data-bin/isl_sentence_level \
    --config mediapipe_config.yaml \
    --srcdict ./dummy_dict.txt \
    --tgtdict data/isl_dataset/sentence_level/spm_bpe4000.vocab
```

## Training the Model

Train the model using Fairseq's Hydra configuration:

```python
# For word-level (may take several hours)
!fairseq-hydra-train \
    --config-dir examples/sign_language/config/wmt-slt \
    --config-name srf_4k \
    task.data=./data-bin/isl_word_level \
    task.sentencepiece_model=./data/isl_dataset/word_level/spm_bpe1000.model \
    checkpoint.save_dir=./checkpoints/isl_word_level \
    hydra.run.dir=./checkpoints/isl_word_level/hydra_run \
    distributed_training.distributed_world_size=1 \
    dataset.num_workers=2 \
    dataset.batch_size=16 \
    dataset.max_tokens=4096

# For sentence-level (may take several hours)
!fairseq-hydra-train \
    --config-dir examples/sign_language/config/wmt-slt \
    --config-name srf_4k \
    task.data=./data-bin/isl_sentence_level \
    task.sentencepiece_model=./data/isl_dataset/sentence_level/spm_bpe4000.model \
    checkpoint.save_dir=./checkpoints/isl_sentence_level \
    hydra.run.dir=./checkpoints/isl_sentence_level/hydra_run \
    distributed_training.distributed_world_size=1 \
    dataset.num_workers=2 \
    dataset.batch_size=16 \
    dataset.max_tokens=4096
```

If training takes too long on Colab, you can:
1. Reduce the number of epochs (`optimization.max_epoch=15`)
2. Use a smaller model (`model.encoder.encoder_layers=4 model.decoder.decoder_layers=4`)
3. Save checkpoints more frequently to handle Colab disconnections (`checkpoint.save_interval=1`)

## Evaluating the Model

Generate translations for the test set:

```python
# For word-level
!mkdir -p ./results/isl_word_level_test

!fairseq-generate ./data-bin/isl_word_level \
    --task sign_to_text \
    --source-lang sign --target-lang en \
    --gen-subset test \
    --path ./checkpoints/isl_word_level/checkpoint_best.pt \
    --config-dir examples/sign_language/config/wmt-slt \
    --config-name srf_4k \
    --sentencepiece-model ./data/isl_dataset/word_level/spm_bpe1000.model \
    --scoring sacrebleu \
    --beam 5 \
    --max-tokens 30000 \
    --results-path ./results/isl_word_level_test

# For sentence-level
!mkdir -p ./results/isl_sentence_level_test

!fairseq-generate ./data-bin/isl_sentence_level \
    --task sign_to_text \
    --source-lang sign --target-lang en \
    --gen-subset test \
    --path ./checkpoints/isl_sentence_level/checkpoint_best.pt \
    --config-dir examples/sign_language/config/wmt-slt \
    --config-name srf_4k \
    --sentencepiece-model ./data/isl_dataset/sentence_level/spm_bpe4000.model \
    --scoring sacrebleu \
    --beam 5 \
    --max-tokens 30000 \
    --results-path ./results/isl_sentence_level_test
```

## Converting to TFLite

For mobile deployment, convert the model to TensorFlow Lite format:

```python
# Install ONNX and TensorFlow dependencies
!pip install onnx onnxruntime tensorflow

# Convert the model (adjust path and parameters as needed)
!python examples/sign_language/scripts/convert_to_tflite.py \
    --checkpoint ./checkpoints/isl_sentence_level/checkpoint_best.pt \
    --data-bin ./data-bin/isl_sentence_level \
    --feat-dim 1839 \
    --seq-len 256 \
    --output-onnx ./tflite_models/encoder.onnx \
    --output-tflite ./tflite_models/encoder.tflite
```

## Troubleshooting

### Kaggle Download Issues

If the Kaggle dataset download keeps failing:

1. **Direct manual download**:
   - Download from [ISL-CSLRT Dataset on Kaggle](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset)
   - Upload to Google Drive
   - Copy and extract in Colab

2. **Using wget with Kaggle cookies**:
   ```python
   # Get download link using Kaggle API
   import kaggle
   kaggle.api.authenticate()
   
   dataset_info = kaggle.api.dataset_list(search="drblack00/isl-csltr-indian-sign-language-dataset")[0]
   download_url = f"https://www.kaggle.com/datasets/download/{dataset_info.ref}"
   
   # Use wget with cookies
   !wget --load-cookies ~/.kaggle/cookies.txt -O data/raw/isl-dataset.zip "{download_url}"
   ```

### Runtime Disconnections
Google Colab may disconnect during long-running tasks. To mitigate this:

1. Save work frequently
2. Break the process into smaller steps
3. Use `%%capture` for non-interactive outputs
4. Consider using Colab Pro for longer runtimes

### Out of Memory Errors
If you encounter OOM errors:

1. Reduce batch size (`dataset.batch_size=8`)
2. Reduce model size
3. Restart the runtime to clear memory

### MediaPipe Installation Issues
If MediaPipe fails to install properly:

```python
!pip uninstall -y mediapipe
!pip install mediapipe==0.10.0
```

### Handling File Persistence

Colab sessions are temporary. To ensure your work is saved:

```python
# Create a zip of your processed data
!zip -r /content/drive/MyDrive/sign-to-text/processed_data.zip data/

# Create a zip of your trained models
!zip -r /content/drive/MyDrive/sign-to-text/trained_models.zip checkpoints/
```

To restore your work in a new session:

```python
# Unzip your processed data
!unzip /content/drive/MyDrive/sign-to-text/processed_data.zip

# Unzip your trained models
!unzip /content/drive/MyDrive/sign-to-text/trained_models.zip
```

Remember that Google Colab has session time limits, so for extensive training, consider using Google Cloud or other cloud GPU providers for uninterrupted training.
