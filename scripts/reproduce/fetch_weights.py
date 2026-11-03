#!/usr/bin/env python3
"""Fetch one expert's fused gate_up slice of Qwen3.8-2.4T-A95B (bf16) by HTTP range.

The checkpoint stores all 512 experts of a layer as one 3-D tensor
`model.layers.46.mlp.experts.gate_up_proj` with shape [512, 4096, 8192] (bf16,
34,359,738,368 bytes) alone in one shard.  Expert e is the contiguous slice
[e * 67,108,864, (e + 1) * 67,108,864) of the tensor data: a bf16 matrix of shape
[4096, 8192] (rows 0-2047 and 2048-4095 are the gate and up halves of the fused
projection; which half is which does not matter for this benchmark).
Only that slice is downloaded; the bytes are written unchanged.

Usage: python3 scripts/reproduce/fetch_weights.py --output inputs/qwen38-expert0
Needs network access to huggingface.co and the `requests` package. For expert 0 the
SHA-256 of the result is checked against the file used for the published numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import requests

REPO = "Qwen/Qwen3.8-2.4T-A95B"
REVISION = "207bd685a7e3696cfaff12ded7c6a7ea0f88c996"
SHARD = "model-00106-of-00213.safetensors"
TENSOR = "model.layers.46.mlp.experts.gate_up_proj"
SHAPE = [512, 4096, 8192]
DTYPE = "BF16"
HEADER_BYTES = 152
TENSOR_BYTES = 34_359_738_368
SLICE_BYTES = 4096 * 8192 * 2  # 67,108,864
SHARD_BYTES = 34_359_738_528  # 8 + 152 + TENSOR_BYTES
EXPERT0_SHA256 = "f51d72cada2a1771261882f2f4dc56667c6059547ad385de57220657c72b6587"


def fetch_range(url: str, start: int, end: int, target: Path) -> int:
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    expected = end - start + 1
    with requests.get(f"{url}?download=true&range={start}-{end}", headers=headers,
                      timeout=(30, 300), stream=True) as response:
        if response.status_code != 206:
            raise RuntimeError(f"range request returned HTTP {response.status_code}, expected 206")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"unexpected Content-Range {content_range!r}; refusing a full-shard download")
        count = 0
        with target.open("xb") as stream:
            for chunk in response.iter_content(1 << 20):
                if not chunk:
                    continue
                count += len(chunk)
                if count > expected:
                    raise RuntimeError("range response exceeded the requested byte count")
                stream.write(chunk)
        if count != expected:
            raise RuntimeError(f"truncated range response: received {count}, expected {expected}")
    return count


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new directory; must not exist")
    p.add_argument("--expert", type=int, default=0, help="expert index 0..511")
    a = p.parse_args()
    if not 0 <= a.expert < SHAPE[0]:
        raise ValueError("expert index out of range")
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{SHARD}"
    with tempfile.TemporaryDirectory(prefix="qwen38-fetch-", dir=out.parent) as tmp:
        stage = Path(tmp)
        fetch_range(url, 0, 7, stage / "len.bin")
        header_length = struct.unpack("<Q", (stage / "len.bin").read_bytes())[0]
        if header_length != HEADER_BYTES:
            raise RuntimeError(f"header length {header_length} != expected {HEADER_BYTES}")
        fetch_range(url, 8, 7 + header_length, stage / "header.bin")
        header_raw = (stage / "header.bin").read_bytes()
        header = json.loads(header_raw)
        header.pop("__metadata__", None)
        if list(header.keys()) != [TENSOR]:
            raise RuntimeError(f"unexpected tensors in shard: {list(header.keys())}")
        entry = header[TENSOR]
        if entry["dtype"] != DTYPE or entry["shape"] != SHAPE or entry["data_offsets"] != [0, TENSOR_BYTES]:
            raise RuntimeError(f"unexpected tensor entry: {entry}")
        start = 8 + header_length + a.expert * SLICE_BYTES
        end = start + SLICE_BYTES - 1
        if end >= SHARD_BYTES:
            raise RuntimeError("slice exceeds shard size")
        weights = stage / "weights.bin"
        fetch_range(url, start, end, weights)
        if weights.stat().st_size != SLICE_BYTES:
            raise RuntimeError("slice size mismatch")
        manifest = {
            "schema": "lds-weight-source-v1",
            "model_id": "qwen38-2p4t-a95b-bf16-expert-gateup",
            "hf_repo": REPO,
            "revision": REVISION,
            "tensor_name": f"{TENSOR}[{a.expert}] (fused gate_up of one expert)",
            "shard": SHARD,
            "source": "native",
            "conversion": "none; contiguous byte slice of the 3-D tensor",
            "encoding": "BF16 (little-endian)",
            "shape": [4096, 8192],
            "byte_matrix_shape": [4096, 16384],
            "expert": a.expert,
            "layer": 46,
            "tensor_shape": SHAPE,
            "checkpoint_header_bytes": header_length,
            "checkpoint_header_sha256": hashlib.sha256(header_raw).hexdigest(),
            "slice_absolute_byte_range": [start, end],
            "weights_file": "weights.bin",
            "bytes": SLICE_BYTES,
            "sha256": sha256_file(weights),
        }
        if a.expert == 0 and manifest["sha256"] != EXPERT0_SHA256:
            raise RuntimeError(f"expert 0 SHA-256 {manifest['sha256']} != expected {EXPERT0_SHA256}")
        (stage / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        out.mkdir()
        os.replace(weights, out / "weights.bin")
        os.replace(stage / "source.json", out / "source.json")
    print(json.dumps({"output": str(out), "bytes": SLICE_BYTES, "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
