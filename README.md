# `bananagrams`

`bananagrams` is a complete pipeline for building a dataset and training a deep learning model for image editing mask prediction. The project processes image pairs from Apple's pico-banana-400k dataset to generate aligned binary masks that indicate which regions of an image were modified, then trains a neural network to predict these masks from images and text instructions.

The project consists of two main components:

1. **Dataset Builder**: Downloads image pairs, computes feature-based alignment using SIFT, generates difference masks, and packages data into WebDataset tar shards for efficient training.

2. **Mask Prediction Model**: A multi-modal deep learning model that combines ResNet-50 image features with CLIP text embeddings through cross-attention to predict binary masks indicating image edit regions.

The tool is designed for researchers and developers working on image editing, inpainting, and instruction-guided image manipulation tasks.

## Installation

### From source

First install the dependencies:

- Python 3.8 or above.

- CUDA-capable GPU (recommended for model training).

- AWS CLI and credentials (for S3 dataset storage, optional for local development).

Install dataset builder dependencies:

```bash
cd dataset
pip install -r requirements.txt
```

Install model training dependencies:

```bash
cd model
pip install -r requirements.txt
```

## Usage

### Dataset Building

Build the dataset from Apple's pico-banana-400k manifest:

```bash
cd dataset
python build.py
```

For development/testing with local extraction:

```bash
python build.py --dev
```

Test with a limited number of samples:

```bash
python build.py --dev --max-samples 10
```

Validate generated masks:

```bash
python validate.py dev_extracted/shard-000000 --output validation_overlays/
```

### Model Training

Train the mask prediction model:

```bash
cd model
python train.py --config config.yaml
```

Resume training from a checkpoint:

```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

## Configuration

### Dataset Configuration

Edit `dataset/config.yaml` to configure:

- Apple CDN prefix and S3 bucket settings
- Train/dev/test split ratios
- Edit types to include
- Mask generation parameters (blur kernel size, threshold, etc.)

### Model Configuration

Edit `model/config.yaml` to configure:

- Model architecture (encoder backbones, fusion layers, decoder channels)
- Training hyperparameters (batch size, learning rate, optimizer, scheduler)
- Data augmentation settings
- Loss function weights (BCE and Dice)
- Logging and checkpointing options

## Architecture

The mask prediction model consists of:

1. **Image Encoder**: ResNet-50 backbone with coordinate channels, extracting multi-scale features for U-Net skip connections.

2. **Text Encoder**: CLIP text tower (ViT-B/32) for encoding edit instructions.

3. **Cross-Attention Fusion**: Multi-layer cross-attention mechanism where image features attend to text features.

4. **Mask Decoder**: U-Net style decoder with skip connections, progressively upsampling to full resolution.

The model is trained with a combined Binary Cross-Entropy and Dice loss for robust segmentation performance.

## Dataset Format

The dataset is packaged as WebDataset tar shards containing:

- `*.input.jpg`: Original input images
- `*.mask.png`: Binary masks indicating edited regions
- `*.text.txt`: Edit instruction text
- `*.summary.txt`: Summarized edit description
- `*.meta.json`: Metadata including edit type and URLs

Shards are organized into train/dev/test splits and can be stored locally or on S3.

## Resources

The following resources are available:

- [Apple pico-banana-400k dataset](https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb)

- [CLIP model repository](https://github.com/openai/CLIP)
