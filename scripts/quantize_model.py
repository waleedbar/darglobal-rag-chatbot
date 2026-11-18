"""Shrink the cached ONNX embedding model to int8 (dynamic quantization) at image build time.

Keeps the container comfortably under the 512 MB RAM of free hosting tiers: the float model plus
onnxruntime buffers left almost no headroom. The int8 model is used both to build the index and to
embed queries, so vectors stay consistent. If anything fails, the float model is left untouched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    cache = Path(os.getenv("MODEL_CACHE", "models"))
    models = sorted(cache.rglob("*.onnx"))
    if not models:
        print("quantize: no ONNX model found, skipping")
        return 0
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:  # pragma: no cover
        print(f"quantize: onnxruntime.quantization unavailable ({exc}), skipping")
        return 0
    for path in models:
        real = path.resolve()  # HF cache files are symlinks into blobs/
        tmp = real.with_suffix(".int8.tmp")
        before = real.stat().st_size
        try:
            quantize_dynamic(str(real), str(tmp), weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gather"])
            tmp.replace(real)
            print(f"quantize: {path.name} {before / 1e6:.0f} MB -> {real.stat().st_size / 1e6:.0f} MB")
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            print(f"quantize: kept float model for {path.name} ({exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
