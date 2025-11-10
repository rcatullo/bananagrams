import argparse
import dataclasses
import io
import json
import os
import tarfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import boto3
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


# -------------------------
# Defaults / config
# -------------------------

DEFAULT_APPLE_CDN_PREFIX = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb"
DEFAULT_SFT_MANIFEST_LINK = (
    f"{DEFAULT_APPLE_CDN_PREFIX}/jsonl/sft.jsonl"
)
DEFAULT_S3_BUCKET = "pico-banana-400k"
DEFAULT_S3_PREFIX = "processed/webdataset/train"
DEFAULT_PROCESSED_IDS_PATH = "processed_ids.txt"

DEFAULT_EDIT_TYPES = [
    "Add a new object to the scene",
    "Remove an existing object",
    "Replace one object category with another",
    "Change an object's attribute (e.g., color/material)",
    "Relocate an object (change its position/spatial relation)",
    "Change the size/shape/orientation of an object",
]


@dataclass
class BuilderConfig:
    apple_cdn_prefix: str = DEFAULT_APPLE_CDN_PREFIX
    manifest_url: str = DEFAULT_SFT_MANIFEST_LINK
    s3_bucket: str = DEFAULT_S3_BUCKET
    s3_prefix: str = DEFAULT_S3_PREFIX

    edit_types: Tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(DEFAULT_EDIT_TYPES)
    )

    samples_per_shard: int = 2048

    request_timeout: float = 10.0
    request_retries: int = 3

    mask_threshold: int = 25

    processed_ids_path: str = DEFAULT_PROCESSED_IDS_PATH
    tmp_dir: str = "tmp_shards"
    manifest_cache_path: Optional[str] = "sft.jsonl"

    max_records: Optional[int] = None
    max_samples: Optional[int] = None

    verbose: bool = True


@dataclass
class Sample:
    sample_id: str
    input_jpeg: bytes
    mask_png: bytes
    text: bytes
    summarized_text: bytes
    meta_json: bytes


# -------------------------
# Helpers
# -------------------------

def load_processed_ids(path: str) -> set:
    ids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(line)
    return ids


def append_processed_ids(path: str, ids: List[str]) -> None:
    if not ids:
        return
    with open(path, "a", encoding="utf-8") as f:
        for pid in ids:
            f.write(pid + "\n")


def download_image(
    url: str,
    timeout: float,
    retries: int,
) -> Optional[Image.Image]:
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                continue
            data = io.BytesIO(resp.content)
            img = Image.open(data)
            img.load()
            return img
        except Exception:
            continue
    return None


def compute_mask(
    input_img: Image.Image,
    output_img: Image.Image,
    threshold: int,
) -> Image.Image:
    input_img = input_img.convert("RGB")
    output_img = output_img.convert("RGB")

    if input_img.size != output_img.size:
        output_img = output_img.resize(input_img.size, resample=Image.BILINEAR)

    in_arr = np.asarray(input_img, dtype=np.int16)
    out_arr = np.asarray(output_img, dtype=np.int16)

    diff = np.abs(out_arr - in_arr)
    diff_mag = diff.max(axis=2)
    mask = (diff_mag > threshold).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask, mode="L")
    return mask_img


def encode_jpeg(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_manifest_line(
    line: str,
    cfg: BuilderConfig,
    processed_ids: set,
) -> Optional[Sample]:
    line = line.strip()
    if not line:
        return None

    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None

    edit_type = rec.get("edit_type")
    if edit_type not in cfg.edit_types:
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

    output_url = f"{cfg.apple_cdn_prefix}/{output_image_rel}"

    input_img = download_image(
        input_url,
        timeout=cfg.request_timeout,
        retries=cfg.request_retries,
    )
    if input_img is None:
        return None

    output_img = download_image(
        output_url,
        timeout=cfg.request_timeout,
        retries=cfg.request_retries,
    )
    if output_img is None:
        return None

    mask_img = compute_mask(input_img, output_img, cfg.mask_threshold)

    input_bytes = encode_jpeg(input_img)
    mask_bytes = encode_png(mask_img)

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


def iter_manifest_lines(
    url: str,
    max_records: Optional[int] = None,
    local_cache_path: Optional[str] = None,
) -> Iterable[str]:
    """
    Iterate over lines from a JSONL manifest.
    
    If local_cache_path is provided and exists, read from there.
    Otherwise, download the entire file first (optionally cache it), then iterate.
    """
    if local_cache_path and os.path.exists(local_cache_path):
        with open(local_cache_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_records is not None and i >= max_records:
                    break
                yield line
        return
    
    print(f"Downloading manifest from {url}...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    content = resp.text
    print(f"Download complete ({len(content)} bytes)")
    
    if local_cache_path:
        os.makedirs(os.path.dirname(local_cache_path) or ".", exist_ok=True)
        with open(local_cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Cached manifest to {local_cache_path}")
    
    for i, line in enumerate(content.splitlines()):
        if max_records is not None and i >= max_records:
            break
        yield line


def _add_bytes_to_tar(
    tar: tarfile.TarFile,
    data: bytes,
    name: str,
) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_tar_shard(
    samples: List[Sample],
    shard_index: int,
    tmp_dir: str,
) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    tar_path = os.path.join(tmp_dir, f"shard-{shard_index:06d}.tar")

    with tarfile.open(tar_path, "w") as tf:
        for s in samples:
            prefix = s.sample_id.replace("/", "_")

            _add_bytes_to_tar(
                tf,
                data=s.input_jpeg,
                name=f"{prefix}.input.jpg",
            )
            _add_bytes_to_tar(
                tf,
                data=s.mask_png,
                name=f"{prefix}.mask.png",
            )
            _add_bytes_to_tar(
                tf,
                data=s.text,
                name=f"{prefix}.text.txt",
            )
            _add_bytes_to_tar(
                tf,
                data=s.summarized_text,
                name=f"{prefix}.summary.txt",
            )
            _add_bytes_to_tar(
                tf,
                data=s.meta_json,
                name=f"{prefix}.meta.json",
            )

    return tar_path


def upload_tar_to_s3(
    tar_path: str,
    shard_index: int,
    cfg: BuilderConfig,
    s3_client=None,
) -> str:
    if s3_client is None:
        s3_client = boto3.client("s3")

    key = f"{cfg.s3_prefix}/shard-{shard_index:06d}.tar"
    s3_client.upload_file(tar_path, cfg.s3_bucket, key)
    return key


# -------------------------
# Main builder
# -------------------------

def build_dataset(cfg: BuilderConfig) -> None:
    os.makedirs(cfg.tmp_dir, exist_ok=True)

    processed_ids = load_processed_ids(cfg.processed_ids_path)
    if cfg.verbose:
        print(f"Loaded {len(processed_ids)} processed ids from {cfg.processed_ids_path}")

    s3_client = boto3.client("s3")

    samples_buffer: List[Sample] = []
    shard_index = 0
    total_samples = 0

    line_iter = iter_manifest_lines(
        cfg.manifest_url, 
        cfg.max_records,
        cfg.manifest_cache_path,
    )

    for line in tqdm(line_iter, desc="Processing manifest", unit="line"):
        sample = process_manifest_line(line, cfg, processed_ids)
        if sample is None:
            continue

        samples_buffer.append(sample)
        total_samples += 1

        if cfg.max_samples is not None and total_samples >= cfg.max_samples:
            if samples_buffer:
                _flush_shard(
                    samples_buffer,
                    shard_index,
                    cfg,
                    s3_client,
                    processed_ids,
                )
                shard_index += 1
            break

        if len(samples_buffer) >= cfg.samples_per_shard:
            _flush_shard(
                samples_buffer,
                shard_index,
                cfg,
                s3_client,
                processed_ids,
            )
            shard_index += 1
            samples_buffer = []

    if samples_buffer:
        _flush_shard(
            samples_buffer,
            shard_index,
            cfg,
            s3_client,
            processed_ids,
        )

    if cfg.verbose:
        print(f"Finished. Total samples written: {total_samples}")


def _flush_shard(
    samples_buffer: List[Sample],
    shard_index: int,
    cfg: BuilderConfig,
    s3_client,
    processed_ids: set,
) -> None:
    if not samples_buffer:
        return

    shard_ids = [s.sample_id for s in samples_buffer]

    tar_path = write_tar_shard(samples_buffer, shard_index, cfg.tmp_dir)
    key = upload_tar_to_s3(tar_path, shard_index, cfg, s3_client)

    append_processed_ids(cfg.processed_ids_path, shard_ids)
    processed_ids.update(shard_ids)

    if cfg.verbose:
        print(
            f"Uploaded shard {shard_index} to s3://{cfg.s3_bucket}/{key} "
            f"({len(samples_buffer)} samples)"
        )

    try:
        os.remove(tar_path)
    except OSError:
        pass


# -------------------------
# CLI
# -------------------------

def parse_args() -> BuilderConfig:
    p = argparse.ArgumentParser(
        description="One-pass dataset builder for pico-banana-400k (sequential)"
    )

    p.add_argument(
        "--apple-cdn-prefix",
        default=DEFAULT_APPLE_CDN_PREFIX,
    )
    p.add_argument(
        "--manifest-url",
        default=DEFAULT_SFT_MANIFEST_LINK,
    )
    p.add_argument(
        "--s3-bucket",
        default=DEFAULT_S3_BUCKET,
    )
    p.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
    )
    p.add_argument(
        "--samples-per-shard",
        type=int,
        default=2048,
    )
    p.add_argument(
        "--mask-threshold",
        type=int,
        default=25,
    )
    p.add_argument(
        "--processed-ids-path",
        default=DEFAULT_PROCESSED_IDS_PATH,
    )
    p.add_argument(
        "--tmp-dir",
        default="tmp_shards",
    )
    p.add_argument(
        "--manifest-cache-path",
        default="sft.jsonl",
        help="Local path to cache the manifest file.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max manifest lines to consider (for debugging).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max successful samples to write (for debugging).",
    )
    p.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose logging.",
    )

    args = p.parse_args()

    return BuilderConfig(
        apple_cdn_prefix=args.apple_cdn_prefix,
        manifest_url=args.manifest_url,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        samples_per_shard=args.samples_per_shard,
        mask_threshold=args.mask_threshold,
        processed_ids_path=args.processed_ids_path,
        tmp_dir=args.tmp_dir,
        manifest_cache_path=args.manifest_cache_path,
        max_records=args.max_records,
        max_samples=args.max_samples,
        verbose=not args.no_verbose,
    )


if __name__ == "__main__":
    cfg = parse_args()
    build_dataset(cfg)
