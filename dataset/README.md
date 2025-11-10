# README.md

## Overview

This folder contains a one-pass dataset builder for the **pico-banana-400k** image editing dataset.

Given the JSONL manifest (`sft_manifest.txt`), the script:

1. Streams manifest entries from the Apple CDN or downlaods manifest if local cache path is provided.
2. Filters to a specified set of `edit_type` values.
3. For each remaining record:
   - Downloads the input image (`open_image_input_url`).
   - Downloads the edited output image (`${apple_cdn_prefix}/${output_image}`).
   - Verifies both images load; invalid entries are discarded.
   - Computes a per-pixel difference mask (output − input), thresholded to a binary mask.
4. Packs `(input image, mask, text, summarized_text, metadata)` into **WebDataset-style tar shards**.
5. Uploads each shard to `s3://pico-banana-400k/<s3_prefix>/shard-XXXXXX.tar`.
6. Tracks which datapoints have been uploaded in `processed_ids.txt` allowing safe resume after failure.

Each sample inside a shard has files:

- `<id_sanitized>.input.jpg`
- `<id_sanitized>.mask.png`
- `<id_sanitized>.text.txt`
- `<id_sanitized>.summary.txt`
- `<id_sanitized>.meta.json`

where:

- `sample_id` = canonical `output_image` string from the manifest (e.g. `images/positive-edit/1.png`).
- `id_sanitized` = `sample_id` with `/` replaced by `_` (e.g. `images_positive-edit_1.png`).

`processed_ids.txt` always stores the **canonical** `sample_id` (with `/`).

---

## Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install boto3 pillow requests tqdm pytest numpy
