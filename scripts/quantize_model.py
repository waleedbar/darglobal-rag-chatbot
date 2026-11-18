"""Shrink the cached ONNX embedding model to int8 at image build time (keeps the app under 512 MB RAM).

The published ONNX file stores float16 weights. On CPU, onnxruntime casts those to float32 at load,
so the resident size more than doubles. Here we:
  1. convert every float16 tensor / type in the graph to float32,
  2. dynamically quantize MatMul + Gather (incl. the 250k-token embedding table) weights to int8,
  3. sanity-check the result against the float32 graph (cosine similarity of sentence embeddings).
The int8 model is used for both the index and queries, so vectors stay consistent. If any step fails
the original file is left untouched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def to_float32(src: Path, dst: Path) -> None:
    import onnx
    from onnx import TensorProto, numpy_helper

    m = onnx.load(str(src))

    def fix_tensor(t):
        if t.data_type == TensorProto.FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))

    def fix_graph(g):
        for init in g.initializer:
            fix_tensor(init)
        for node in g.node:
            for a in node.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT16:
                    a.i = TensorProto.FLOAT
                if a.type == onnx.AttributeProto.TENSOR:
                    fix_tensor(a.t)
                if a.type == onnx.AttributeProto.GRAPH:
                    fix_graph(a.g)
        for vi in list(g.input) + list(g.output):
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                vi.type.tensor_type.elem_type = TensorProto.FLOAT
        g.ClearField("value_info")

    fix_graph(m.graph)
    onnx.save(m, str(dst))


def sentence_vecs(model: Path, tok_file: Path, optimise: bool) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    so = ort.SessionOptions()
    if not optimise:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    tok = Tokenizer.from_file(str(tok_file))
    tok.enable_padding(pad_id=1, pad_token="<pad>")
    enc = tok.encode_batch(["When is the handover of Neptune?", "ما هي أرخص فيلا في جدة؟",
                            "Dar Global luxury villas in Muscat, Oman"])
    ids = np.array([e.ids for e in enc], dtype="int64")
    mask = np.array([e.attention_mask for e in enc], dtype="int64")
    feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    h = sess.run(None, {k: v for k, v in feed.items() if k in names})[0].astype("float32")
    if h.ndim == 3:
        h = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def main() -> int:
    cache = Path(os.getenv("MODEL_CACHE", "models"))
    models = sorted(p for p in cache.rglob("*.onnx") if (p.parent / "tokenizer.json").exists())
    if not models:
        print("quantize: no ONNX model found, skipping")
        return 0
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for path in models:
        real = path.resolve()  # HF cache files are symlinks into blobs/
        f32, q8 = real.with_suffix(".f32.tmp"), real.with_suffix(".int8.tmp")
        tok = path.parent / "tokenizer.json"
        try:
            to_float32(real, f32)
            quantize_dynamic(str(f32), str(q8), weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gather"])
            ref, new = sentence_vecs(f32, tok, True), sentence_vecs(q8, tok, False)
            cos = float((ref * new).sum(1).min())
            print(f"quantize: {path.name} {real.stat().st_size / 1e6:.0f} MB -> {q8.stat().st_size / 1e6:.0f} MB, "
                  f"min cosine vs float32 = {cos:.4f}")
            if cos < 0.95:
                raise ValueError("quantized embeddings drifted too far")
            q8.replace(real)
        except Exception as exc:
            print(f"quantize: kept original model for {path.name} ({type(exc).__name__}: {exc})")
        finally:
            f32.unlink(missing_ok=True)
            q8.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
