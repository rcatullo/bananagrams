# Bananagrams Dataset Builder

## Overview

A dataset builder for the **pico-banana-400k** image editing dataset. Downloads image pairs from Apple's dataset, computes aligned edit masks, and packages them into WebDataset tar shards for training.

## What It Does

The builder processes image editing data through the following pipeline:

1. **Downloads** input and output image pairs from Apple's CDN
2. **Aligns** images using SIFT feature matching to handle crops, zooms, and shifts
3. **Computes** difference masks showing edited regions
4. **Validates** alignment quality (automatically rejects poor alignments)
5. **Cleans** masks with noise reduction and edge artifact removal
6. **Splits** samples deterministically into train/dev/test sets (90%/5%/5%)
7. **Packages** results into WebDataset tar shards

Each processed sample includes:
- `*.input.jpg` - Original input image
- `*.mask.png` - Binary mask of edited regions (aligned with input)
- `*.text.txt` - Edit instruction text
- `*.summary.txt` - Summarized edit instruction
- `*.meta.json` - Metadata (URLs, edit type, etc.)

## Key Features

- **SIFT Alignment**: Accurately aligns images before computing differences
- **Dynamic Edge Detection**: Automatically detects and removes alignment artifacts at image borders
- **Noise Filtering**: Keeps only the 5 largest mask components
- **Quality Validation**: Rejects samples with poor alignment (tracks success rate)
- **Deterministic Splits**: Automatically creates train/dev/test splits (90%/5%/5%)
- **Resume Support**: Tracks processed IDs to safely resume after interruption

## Requirements

```bash
pip install -r requirements.txt
```

## Configuration

All settings are configured in `config.yaml`. Key settings include:

- **Split ratios**: `train_ratio`, `dev_ratio`, `test_ratio` (must sum to 1.0)
- **Edit types**: List of edit categories to include in the dataset
- **Processing**: `samples_per_shard`, `request_timeout`, `request_retries`
- **S3 settings**: Bucket and prefix paths for uploading

See `config.yaml` for all available options.

## Usage

### Development Mode (Local Testing)

Process samples locally without uploading to S3:

```bash
# Test with 30 samples
python build.py --dev --max-samples 30

# Test with 100 samples
python build.py --dev --max-samples 100
```

**Output**: Extracted shards organized by split:
- `dev_extracted/train/shard-000000/` (90% of samples)
- `dev_extracted/dev/shard-000000/` (5% of samples)
- `dev_extracted/test/shard-000000/` (5% of samples)

### Production Mode (Full Dataset)

Build complete dataset and upload to S3:

```bash
python build.py
```

**Output**: Tar shards uploaded to S3, organized by split:
- `s3://pico-banana-400k/processed/webdataset/train/` (90% of samples)
- `s3://pico-banana-400k/processed/webdataset/dev/` (5% of samples)
- `s3://pico-banana-400k/processed/webdataset/test/` (5% of samples)

### Validation

Verify mask quality and alignment:

```bash
# Validate train split
python validate.py dev_extracted/train/shard-000000 --output validation_overlays

# Validate dev split
python validate.py dev_extracted/dev/shard-000000

# Check only first N samples
python validate.py dev_extracted/test/shard-000000 --max-samples 5
```

**Output**: Statistics on mask coverage, component counts, and edge proximity. Optional overlay images showing masks highlighted on input images.

## Data Splits

Samples are **deterministically assigned** to splits using MD5 hash of sample IDs:
- **Train**: 90% of samples (~360k in full dataset)
- **Dev**: 5% of samples (~20k in full dataset)
- **Test**: 5% of samples (~20k in full dataset)

## Files

- **`build.py`** - Main dataset builder
- **`mask.py`** - Mask computation module (SIFT alignment, edge detection, filtering)
- **`validate.py`** - Mask validation and visualization tool
- **`config.yaml`** - Configuration file (edit to customize behavior)
- **`requirements.txt`** - Python dependencies
- **`processed_ids.txt`** - Tracks successfully processed samples (auto-generated)
- **`sft.jsonl`** - Cached manifest from Apple (auto-downloaded)

## Methods

### Mask Generation (`mask.py`)

1. **Align images**: SIFT feature matching with homography estimation
2. **Compute difference**: Pixel-wise comparison of aligned images
3. **Blur and threshold**: Gaussian blur then threshold
4. **Morphological ops**: Opening and closing to remove noise
5. **Detect edges**: Scan from borders inward to find dark/distorted regions
6. **Remove artifacts**: Zero out detected edge regions
7. **Filter components**: Keep only 5 largest connected regions

### Dataset Building (`build.py`)

1. **Load manifest**: Downloads Apple's JSONL manifest (cached locally)
2. **Filter edits**: Keeps only specified edit types (add, remove, replace, etc.)
3. **Process samples**: For each valid entry:
   - Download input and output images
   - Generate aligned mask
   - Encode as JPEG/PNG
   - **Assign to split** (train/dev/test) using hash of sample ID
4. **Package shards**: Groups samples by split into tar files (2048 samples per shard)
5. **Upload or extract**: Either uploads to S3 or extracts locally (dev mode)
6. **Track progress**: Appends processed IDs to prevent duplicates

### Validation (`validate.py`)

1. **Load samples**: Reads extracted shard directory
2. **Analyze masks**: Computes coverage %, component count, edge proximity
3. **Create overlays** (optional): Generates side-by-side visualizations
4. **Report issues**: Flags samples with potential problems

## Output Statistics

After running, you'll see split breakdown and alignment statistics:

```
============================================================
DATASET BUILDING COMPLETE
============================================================
Total samples written: 30

Split breakdown:
  TRAIN: 28 samples (93.3%)
  DEV: 1 samples (3.3%)
  TEST: 1 samples (3.3%)

Alignment Statistics:
  Samples attempted: 30
  Alignment success: 30 (100.0%)
  Alignment failed:  0 (0.0%)

  Alignment quality:
    Avg inlier ratio: 0.832
    Avg matches: 402
============================================================
```

**Success rate >80%** indicates good dataset quality. Split percentages will be closer to 90/5/5 with larger sample counts.

## Troubleshooting

**Low success rate (<50%)**:
- Input/output images may be too different
- Dataset may have quality issues

**Many edge warnings**:
- Expected behavior - some edits genuinely occur near edges
- Only concerning if masks look obviously misaligned in validation overlays

**Out of memory**:
- Process in smaller batches using `--max-samples`
- Reduce `SAMPLES_PER_SHARD` constant in `build.py`

## Training Recommendations

The generated masks are aligned with input images, so during training:

```python
# Apply same transforms to both
transform = Compose([Resize((512, 512)), ...])
image = transform(image)
mask = transform(mask)
```

Masks are binary (0 or 255) and represent edited regions in input image coordinates.
