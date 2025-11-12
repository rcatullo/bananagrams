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
import time
import logging
import os
import tarfile
from dataclasses import dataclass
from typing import Iterable, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import boto3
import numpy as np
import requests
from config import load_config
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from mask import compute_mask
CONFIG = load_config()

REQUEST_RETRIES = CONFIG["request_retries"]


# Global config (loaded at module import)
session = requests.Session()

#Some network settings
retries = Retry(
    total=REQUEST_RETRIES,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 503, 504),
)
adapter = HTTPAdapter(
    max_retries=retries,
    pool_connections=50,
    pool_maxsize=50,
)
session.mount("http://", adapter)
session.mount("https://", adapter)


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

download_pool = ThreadPoolExecutor(max_workers=8)


def get_split_for_shard(shard_index: int) -> str:
    """
    Deterministically assign shard to train/dev/test split.
    
    Uses shard index modulo 100 for reproducible assignment.
    """
    ratio = (shard_index % 100) / 100.0
    
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
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        return img
    except Exception:
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
    # Stream the response to avoid loading entire manifest into memory
    resp = session.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    
    # Open cache file if needed
    cache_file = None
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        cache_file = open(cache_path, "w")
    
    bytes_downloaded = 0
    line_count = 0
    
    try:
        # Stream lines from response
        for line_bytes in resp.iter_lines():
            if line_bytes:  # Skip empty lines
                line = line_bytes.decode('utf-8')
                bytes_downloaded += len(line_bytes)
                
                # Write to cache if enabled
                if cache_file:
                    cache_file.write(line + '\n')
                
                # Yield the line
                line_count += 1
                if max_records and line_count > max_records:
                    break
                yield line
        
        print(f"Download complete (~{bytes_downloaded} bytes, {line_count} lines)")
        if cache_path:
            print(f"Cached to {cache_path}")
    finally:
        if cache_file:
            cache_file.close()

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
    t0 = time.perf_counter()

    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    t_json = time.perf_counter()

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

    t1 = time.perf_counter()

    fut_in  = download_pool.submit(download_image, input_url)
    fut_out = download_pool.submit(download_image, output_url)

    input_img = fut_in.result()
    if input_img is None:
        return None
    
    t2 = time.perf_counter()
    output_img = fut_out.result()
    if output_img is None:
        return None


    t3 = time.perf_counter()
    stats.total_attempted += 1

    t4 = time.perf_counter()
    mask_result = compute_mask(input_img, output_img)
    t5 = time.perf_counter()


    if not mask_result.success:
        stats.alignment_failed += 1
        reason = mask_result.failure_reason
        stats.failure_reasons[reason] = stats.failure_reasons.get(reason, 0) + 1
        logging.info(
            f"TIMING sample={sample_id} json={t_json-t0:.4f}s "
            f"dl_in={t2-t1:.4f}s dl_out={t3-t2:.4f}s mask={t5-t4:.4f}s (FAILED)"
        )
        return None
    
    if mask_result.alignment_result:
        stats.alignment_success += 1
        stats.alignment_metrics.append({
            'sample_id': sample_id,
            'num_matches': mask_result.alignment_result.num_matches,
            'inlier_ratio': mask_result.alignment_result.inlier_ratio,
        })
    t6 = time.perf_counter()

    input_bytes = encode_jpeg(input_img)
    mask_bytes = encode_png(mask_result.mask)
    t7 = time.perf_counter()
    logging.info(
        f"TIMING sample={sample_id} json={t_json-t0:.4f}s "
        f"dl_in={t2-t1:.4f}s dl_out={t3-t2:.4f}s mask={t5-t4:.4f}s "
        f"encode={t7-t6:.4f}s total={t7-t0:.4f}s"
    )


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
    
    # Only append to processed_ids in production mode
    if not dev_mode:
        append_processed_ids(PROCESSED_IDS_PATH, shard_ids)
    processed_ids.update(shard_ids)
    
    if dev_mode:
        # Dev mode: extract locally
        extract_dir = extract_shard(tar_path, shard_index, split, dev_extract_dir or "dev_extracted")
        print(f"[{split.upper()}] Created shard {shard_index} at {tar_path} and extracted to {extract_dir} ({len(samples)} samples)")
        logging.info(f"Flushed shard {shard_index} to {extract_dir} with {len(samples)} samples (dev mode)")
    else:
        # Production: upload to S3
        key = upload_to_s3(tar_path, shard_index, split, s3_client)
        print(f"[{split.upper()}] Uploaded shard {shard_index} to s3://{S3_BUCKET}/{key} ({len(samples)} samples)")
        logging.info(f"Flushed shard {shard_index} to s3://{S3_BUCKET}/{key} with {len(samples)} samples")
        
        # Clean up tar file
        try:
            os.remove(tar_path)
        except OSError:
            pass


def build_dataset(dev_mode: bool = False, max_samples: Optional[int] = None):
    """
    Main dataset building pipeline.
    
    Assigns entire shards to train/dev/test splits (not individual samples).
    
    Args:
        dev_mode: If True, extract shards locally instead of uploading to S3
        max_samples: Maximum number of samples to process (for testing)
    """
    # Set up logging to file
    log_filename = "build_dataset.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Starting dataset build in {'DEV' if dev_mode else 'PRODUCTION'} mode")
    
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
    
    # Calculate starting shard index based on already processed samples
    num_processed = len(processed_ids)
    global_shard_index = num_processed // SAMPLES_PER_SHARD
    
    # Calculate per-split shard indices
    split_indices = {"train": 0, "dev": 0, "test": 0}
    for i in range(global_shard_index):
        split = get_split_for_shard(i)
        split_indices[split] += 1
    
    if global_shard_index > 0:
        print(f"Resuming from global shard index {global_shard_index}")
        print(f"Per-split shard indices: {split_indices}")
    
    # Single buffer for all samples, shard assigned to split when flushed
    sample_buffer = []
    split_counts = {"train": 0, "dev": 0, "test": 0}
    
    line_iter = iter_manifest_lines(MANIFEST_URL, None, MANIFEST_CACHE)
    
    for line in tqdm(line_iter, desc="Processing manifest", unit="line"):
        sample = process_manifest_line(line, processed_ids, stats)
        if sample is None:
            continue
        
        # Log each positive sample
        logging.info(f"Positive sample created: {sample.sample_id} (current shard buffer size: {len(sample_buffer) + 1})")
        
        sample_buffer.append(sample)
        
        # Check if we've hit max samples
        if max_samples and len(sample_buffer) >= max_samples:
            break
        
        # Check if shard is full
        if len(sample_buffer) >= SAMPLES_PER_SHARD:
            # Assign this shard to a split based on global shard index
            split = get_split_for_shard(global_shard_index)
            
            flush_shard(
                sample_buffer,
                split_indices[split],
                split,
                processed_ids,
                dev_mode,
                dev_extract_dir,
                s3_client
            )
            
            # Update counters
            split_counts[split] += len(sample_buffer)
            split_indices[split] += 1
            global_shard_index += 1
            sample_buffer = []
    
    # Flush remaining samples
    if sample_buffer:
        split = get_split_for_shard(global_shard_index)
        flush_shard(
            sample_buffer,
            split_indices[split],
            split,
            processed_ids,
            dev_mode,
            dev_extract_dir,
            s3_client
        )
        split_counts[split] += len(sample_buffer)
    
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

