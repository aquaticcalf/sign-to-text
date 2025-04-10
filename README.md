
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

### 2. Create the preparation scripts

Create `prepare_isl_dataset_fairseq.py`:

```python
%%writefile prepare_isl_dataset_fairseq.py
# Paste the entire content of scripts/prepare_isl_dataset_fairseq.py here
# (The full script contains 442 lines, so it's not included here for brevity)
```

Create `train_spm.py`:

```python
%%writefile train_spm.py
# Paste the entire content of scripts/train_spm.py here
```

Create `mediapipe_config.yaml`:

```python
%%writefile mediapipe_config.yaml
modality: MEDIAPIPE
process_steps:
  - type: instance_normalize
  # Optional: Subsample frames if sequences are too long
  # - type: subsample
  #   rate: 2
```

### 3. Download and process the dataset

```python
# Download and process the dataset (takes time)
!python prepare_isl_dataset_fairseq.py --download --process
```

This will:
- Download the ISL-CSLRT dataset from Kaggle
- Extract MediaPipe features from video frames
- Create train/validation/test splits
- Generate TSV files for Fairseq

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

### Dataset Processing Issues

If dataset processing fails, try processing in smaller chunks:

```python
# Process only word level data
!python prepare_isl_dataset_fairseq.py --process-word-only
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
