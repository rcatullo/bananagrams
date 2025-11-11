"""
Simplified dataset builder from pico-banana-400k.

Downloads image pairs from Apple's dataset, computes alignment and masks,
and packages into WebDataset tar shards.

Usage:
    python build.py [--dev] [--max-samples N]
"""

import argparse
import dataclasses
import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from typing import Iterable, List, Optional

import boto3
import numpy as np
import requests
import yaml
from PIL import Image
from tqdm import tqdm

from mask import compute_mask


# Load configuration
def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Global config (loaded at module import)
CONFIG = load_config()

# Extract config values
APPLE_CDN_PREFIX = CONFIG["apple_cdn_prefix"]
MANIFEST_URL = f"{APPLE_CDN_PREFIX}/jsonl/sft.jsonl"
S3_BUCKET = CONFIG["s3_bucket"]
S3_PREFIXES = CONFIG["s3_prefixes"]
PROCESSED_IDS_PATH = CONFIG["processed_ids_path"]
MANIFEST_CACHE = CONFIG["manifest_cache"]

TRAIN_RATIO = CONFIG["train_ratio"]
DEV_RATIO = CONFIG["dev_ratio"]
TEST_RATIO = CONFIG["test_ratio"]

EDIT_TYPES = CONFIG["edit_types"]

SAMPLES_PER_SHARD = CONFIG["samples_per_shard"]
REQUEST_TIMEOUT = CONFIG["request_timeout"]
REQUEST_RETRIES = CONFIG["request_retries"]


def get_split(sample_id: str) -> str:
    """
    Deterministically assign sample to train/dev/test split.
    
    Uses MD5 hash of sample_id for reproducible assignment.
    """
    hash_val = int(hashlib.md5(sample_id.encode()).hexdigest(), 16)
    ratio = (hash_val % 100) / 100.0
    
    if ratio < TRAIN_RATIO:
        return "train"
    elif ratio < TRAIN_RATIO + DEV_RATIO:
        return "dev"
    else:
        return "test"


@dataclass
class Sample:
    """A single training sample."""
    sample_id: str
    input_jpeg: bytes
    mask_png: bytes
    text: bytes
    summarized_text: bytes
    meta_json: bytes


@dataclass
class Stats:
    """Processing statistics."""
    total_attempted: int = 0
    alignment_success: int = 0
    alignment_failed: int = 0
    failure_reasons: dict = dataclasses.field(default_factory=dict)
    alignment_metrics: list = dataclasses.field(default_factory=list)


def load_processed_ids(path: str) -> set:
    """Load set of already processed sample IDs."""
    ids = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(line)
    return ids


def append_processed_ids(path: str, ids: List[str]) -> None:
    """Append newly processed IDs to tracking file."""
    if not ids:
        return
    with open(path, "a") as f:
        for pid in ids:
            f.write(pid + "\n")


def download_image(url: str) -> Optional[Image.Image]:
    """Download and load an image from URL."""
    for _ in range(REQUEST_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                continue
            img = Image.open(io.BytesIO(resp.content))
            img.load()
            return img
        except Exception:
            continue
    return None


def iter_manifest_lines(
    url: str,
    max_records: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> Iterable[str]:
    """Iterate manifest lines from URL or cache."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                yield line
        return
    
    print(f"Downloading manifest from {url}...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    content = resp.text
    print(f"Download complete ({len(resp.content)} bytes)")
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(content)
        print(f"Cached to {cache_path}")
    
    for i, line in enumerate(content.splitlines()):
        if max_records and i >= max_records:
            break
        yield line


def process_manifest_line(
    line: str,
    processed_ids: set,
    stats: Stats,
) -> Optional[Sample]:
    """
    Process a single manifest line into a Sample.
    
    Returns None if sample should be skipped.
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    edit_type = rec.get("edit_type")
    if edit_type not in EDIT_TYPES:
        return None
    
    output_image_rel = rec.get("output_image")
    if not output_image_rel:
        return None
    
    sample_id = output_image_rel
    if sample_id in processed_ids:
        return None
    
    input_url = rec.get("open_image_input_url")
    if not input_url:
        return None
    
    text = rec.get("text") or ""
    summarized_text = rec.get("summarized_text") or ""
    output_url = f"{APPLE_CDN_PREFIX}/{output_image_rel}"
    
    input_img = download_image(input_url)
    if input_img is None:
        return None
    
    output_img = download_image(output_url)
    if output_img is None:
        return None
    
    stats.total_attempted += 1
    
    mask_result = compute_mask(input_img, output_img)
    
    if not mask_result.success:
        stats.alignment_failed += 1
        reason = mask_result.failure_reason
        stats.failure_reasons[reason] = stats.failure_reasons.get(reason, 0) + 1
        return None
    
    if mask_result.alignment_result:
        stats.alignment_success += 1
        stats.alignment_metrics.append({
            'sample_id': sample_id,
            'num_matches': mask_result.alignment_result.num_matches,
            'inlier_ratio': mask_result.alignment_result.inlier_ratio,
        })
    
    input_bytes = encode_jpeg(input_img)
    mask_bytes = encode_png(mask_result.mask)
    
    meta = {
        "open_image_input_url": input_url,
        "text": text,
        "output_image": output_image_rel,
        "edit_type": edit_type,
        "summarized_text": summarized_text,
        "output_url": output_url,
    }
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    
    return Sample(
        sample_id=sample_id,
        input_jpeg=input_bytes,
        mask_png=mask_bytes,
        text=text.encode("utf-8"),
        summarized_text=summarized_text.encode("utf-8"),
        meta_json=meta_bytes,
    )


def encode_jpeg(img: Image.Image, quality: int = 95) -> bytes:
    """Encode image as JPEG."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_png(img: Image.Image) -> bytes:
    """Encode image as PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_tar_shard(samples: List[Sample], shard_index: int, tmp_dir: str) -> str:
    """Write samples to tar shard."""
    os.makedirs(tmp_dir, exist_ok=True)
    tar_path = os.path.join(tmp_dir, f"shard-{shard_index:06d}.tar")
    
    with tarfile.open(tar_path, "w") as tf:
        for s in samples:
            prefix = s.sample_id.replace("/", "_")
            
            for name, data in [
                (f"{prefix}.input.jpg", s.input_jpeg),
                (f"{prefix}.mask.png", s.mask_png),
                (f"{prefix}.text.txt", s.text),
                (f"{prefix}.summary.txt", s.summarized_text),
                (f"{prefix}.meta.json", s.meta_json),
            ]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    
    return tar_path


def upload_to_s3(tar_path: str, shard_index: int, split: str, s3_client) -> str:
    """Upload tar shard to S3."""
    key = f"{S3_PREFIXES[split]}/shard-{shard_index:06d}.tar"
    s3_client.upload_file(tar_path, S3_BUCKET, key)
    return key


def extract_shard(tar_path: str, shard_index: int, split: str, extract_dir: str) -> str:
    """Extract tar shard for inspection."""
    split_dir = os.path.join(extract_dir, split)
    shard_dir = os.path.join(split_dir, f"shard-{shard_index:06d}")
    os.makedirs(shard_dir, exist_ok=True)
    
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(shard_dir)
    
    return shard_dir


def flush_shard(
    samples: List[Sample],
    shard_index: int,
    split: str,
    processed_ids: set,
    dev_mode: bool,
    dev_extract_dir: Optional[str] = None,
    s3_client = None,
) -> None:
    """Write shard and either upload to S3 or extract locally."""
    if not samples:
        return
    
    shard_ids = [s.sample_id for s in samples]
    tar_path = write_tar_shard(samples, shard_index, "tmp_shards")
    
    append_processed_ids(PROCESSED_IDS_PATH, shard_ids)
    processed_ids.update(shard_ids)
    
    if dev_mode:
        # Dev mode: extract locally
        extract_dir = extract_shard(tar_path, shard_index, split, dev_extract_dir or "dev_extracted")
        print(f"[{split.upper()}] Created shard {shard_index} at {tar_path} and extracted to {extract_dir} ({len(samples)} samples)")
    else:
        # Production: upload to S3
        key = upload_to_s3(tar_path, shard_index, split, s3_client)
        print(f"[{split.upper()}] Uploaded shard {shard_index} to s3://{S3_BUCKET}/{key} ({len(samples)} samples)")
        
        # Clean up tar file
        try:
            os.remove(tar_path)
        except OSError:
            pass


def build_dataset(dev_mode: bool = False, max_samples: Optional[int] = None):
    """
    Main dataset building pipeline.
    
    Deterministically splits samples into train/dev/test sets.
    
    Args:
        dev_mode: If True, extract shards locally instead of uploading to S3
        max_samples: Maximum number of samples to process (for testing)
    """
    os.makedirs("tmp_shards", exist_ok=True)
    
    processed_ids = load_processed_ids(PROCESSED_IDS_PATH)
    print(f"Loaded {len(processed_ids)} processed IDs from {PROCESSED_IDS_PATH}")
    print(f"Split ratios: Train={TRAIN_RATIO:.0%}, Dev={DEV_RATIO:.0%}, Test={TEST_RATIO:.0%}")
    
    if dev_mode:
        dev_extract_dir = "dev_extracted"
        os.makedirs(dev_extract_dir, exist_ok=True)
        print(f"DEV MODE: Shards will be extracted to {dev_extract_dir}")
        s3_client = None
    else:
        dev_extract_dir = None
        s3_client = boto3.client("s3")
        print("PRODUCTION MODE: Shards will be uploaded to S3")
    
    stats = Stats()
    
    # Separate buffers and shard indices for each split
    split_buffers = {"train": [], "dev": [], "test": []}
    split_indices = {"train": 0, "dev": 0, "test": 0}
    split_counts = {"train": 0, "dev": 0, "test": 0}
    
    line_iter = iter_manifest_lines(MANIFEST_URL, None, MANIFEST_CACHE)
    
    for line in tqdm(line_iter, desc="Processing manifest", unit="line"):
        sample = process_manifest_line(line, processed_ids, stats)
        if sample is None:
            continue
        
        # Determine which split this sample belongs to
        split = get_split(sample.sample_id)
        split_buffers[split].append(sample)
        split_counts[split] += 1
        
        total_samples = sum(split_counts.values())
        
        # Check if we've hit max samples
        if max_samples and total_samples >= max_samples:
            break
        
        # Check if any split's shard is full
        for split_name in ["train", "dev", "test"]:
            if len(split_buffers[split_name]) >= SAMPLES_PER_SHARD:
                flush_shard(
                    split_buffers[split_name],
                    split_indices[split_name],
                    split_name,
                    processed_ids,
                    dev_mode,
                    dev_extract_dir,
                    s3_client
                )
                split_indices[split_name] += 1
                split_buffers[split_name] = []
    
    # Flush remaining samples in all splits
    for split_name in ["train", "dev", "test"]:
        if split_buffers[split_name]:
            flush_shard(
                split_buffers[split_name],
                split_indices[split_name],
                split_name,
                processed_ids,
                dev_mode,
                dev_extract_dir,
                s3_client
            )
    
    # Print statistics
    total_samples = sum(split_counts.values())
    print(f"\n{'='*60}")
    print(f"DATASET BUILDING COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples written: {total_samples}")
    print(f"\nSplit breakdown:")
    for split_name in ["train", "dev", "test"]:
        count = split_counts[split_name]
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {split_name.upper()}: {count} samples ({pct:.1f}%)")
    
    if stats.total_attempted > 0:
        success_rate = (stats.alignment_success / stats.total_attempted) * 100
        print(f"\nAlignment Statistics:")
        print(f"  Samples attempted: {stats.total_attempted}")
        print(f"  Alignment success: {stats.alignment_success} ({success_rate:.1f}%)")
        print(f"  Alignment failed:  {stats.alignment_failed} ({100-success_rate:.1f}%)")
        
        if stats.failure_reasons:
            print(f"\n  Failure reasons:")
            for reason, count in sorted(stats.failure_reasons.items(), key=lambda x: x[1], reverse=True):
                pct = (count / stats.total_attempted) * 100
                print(f"    {reason}: {count} ({pct:.1f}%)")
        
        if stats.alignment_metrics:
            inlier_ratios = [m['inlier_ratio'] for m in stats.alignment_metrics]
            num_matches = [m['num_matches'] for m in stats.alignment_metrics]
            print(f"\n  Alignment quality:")
            print(f"    Avg inlier ratio: {np.mean(inlier_ratios):.3f}")
            print(f"    Avg matches: {np.mean(num_matches):.0f}")
    
    print(f"{'='*60}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build pico-banana-400k dataset with aligned masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build full dataset (uploads to S3)
    python build.py

    # Dev mode (local extraction, no S3)
    python build.py --dev

    # Test with limited samples
    python build.py --dev --max-samples 10
        """
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: extract shards locally instead of uploading to S3"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    build_dataset(dev_mode=args.dev, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

