import io
import json
import os
import tarfile
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from builder import (
    BuilderConfig,
    Sample,
    compute_mask,
    process_manifest_line,
    write_tar_shard,
)


def make_image(color, size=(16, 16)):
    img = Image.new("RGB", size, color=color)
    return img


def test_compute_mask_simple():
    img1 = make_image((0, 0, 0))
    img2 = make_image((255, 255, 255))

    mask = compute_mask(img1, img2, threshold=10)
    arr = np.array(mask)
    assert arr.shape == (16, 16)
    assert arr.min() == 255
    assert arr.max() == 255


def test_process_manifest_line_filters_edit_type():
    cfg = BuilderConfig(edit_types=("Keep only this",))
    processed_ids = set()

    rec = {
        "open_image_input_url": "http://example.com/in.jpg",
        "text": "foo",
        "output_image": "images/x.png",
        "edit_type": "Some other type",
        "summarized_text": "bar",
    }
    line = json.dumps(rec)

    with mock.patch("builder.download_image") as dl:
        sample = process_manifest_line(line, cfg, processed_ids)
        assert sample is None
        dl.assert_not_called()


def test_process_manifest_line_download_and_mask():
    cfg = BuilderConfig(
        edit_types=("Keep",),
        mask_threshold=10,
    )
    processed_ids = set()

    rec = {
        "open_image_input_url": "http://example.com/in.jpg",
        "text": "foo",
        "output_image": "images/x.png",
        "edit_type": "Keep",
        "summarized_text": "bar",
    }
    line = json.dumps(rec)

    img_in = make_image((0, 0, 0))
    img_out = make_image((255, 255, 255))

    def fake_download(url, timeout, retries):
        if "in.jpg" in url:
            return img_in
        if "images/x.png" in url:
            return img_out
        return None

    with mock.patch("builder.download_image", side_effect=fake_download):
        s = process_manifest_line(line, cfg, processed_ids)

    assert isinstance(s, Sample)
    # canonical id is raw output_image (with slash)
    assert s.sample_id == "images/x.png"
    assert s.text == b"foo"
    assert s.summarized_text == b"bar"

    mask_img = Image.open(io.BytesIO(s.mask_png))
    arr = np.array(mask_img)
    assert arr.min() == 255
    assert arr.max() == 255


def test_write_tar_shard_roundtrip(tmp_path):
    samples = [
        Sample(
            sample_id="sample1",
            input_jpeg=b"input1",
            mask_png=b"mask1",
            text=b"text1",
            summarized_text=b"sum1",
            meta_json=b"{}",
        ),
        Sample(
            sample_id="sample2",
            input_jpeg=b"input2",
            mask_png=b"mask2",
            text=b"text2",
            summarized_text=b"sum2",
            meta_json=b"{}",
        ),
    ]
    tar_path = write_tar_shard(samples, shard_index=0, tmp_dir=str(tmp_path))

    assert os.path.exists(tar_path)

    with tarfile.open(tar_path, "r") as tf:
        names = sorted(tf.getnames())
        # 5 files per sample
        assert len(names) == 10
        assert "sample1.input.jpg" in names
        assert "sample1.mask.png" in names
        assert "sample1.text.txt" in names
        assert "sample1.summary.txt" in names
        assert "sample1.meta.json" in names

        member = tf.getmember("sample1.text.txt")
        f = tf.extractfile(member)
        assert f.read() == b"text1"
