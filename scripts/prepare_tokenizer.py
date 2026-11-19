from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLES = [
    "When is the handover of Neptune?",
    "ما هي مشاريع دار جلوبال في عُمان؟",
    "أرخص فيلا للبيع في جدة، حي الشاطئ - 1,250,000 ريال",
    "Trump International Hotel & Tower Dubai: 2-4 bedroom apartments from AED 3.5M (Q4 2027).",
    "Aida, Muscat — Oman's first ultra-luxury hilltop community!!  ",
    "شقة 3 غرف نوم، مساحة 180 م² — الرياض/النرجس. Price: SAR 950,000",
    "Ünïcödé façade naïve café 😀 ١٢٣٤٥ ٦٧٨",
    " ".join(["Dar Global luxury branded residences in London, Marbella, Dubai, Riyadh and Jeddah."] * 20),
]


def main() -> int:
    cache = Path(os.getenv("MODEL_CACHE", "models"))
    toks = sorted(cache.rglob("tokenizer.json"))
    if not toks:
        print("tokenizer: no tokenizer.json found, skipping")
        return 0
    from huggingface_hub import hf_hub_download

    from rag.embeddings import HFTokenizer, SentencePieceXLMR

    src = hf_hub_download("FacebookAI/xlm-roberta-base", "sentencepiece.bpe.model", cache_dir=str(cache / "_spm"))
    for tj in toks:
        dst = tj.parent / "sentencepiece.bpe.model"
        shutil.copyfile(src, dst)
        ref, sp = HFTokenizer(str(tj), 128), SentencePieceXLMR(str(dst), 128)
        bad = [t[:40] for t in SAMPLES if ref.batch([t])[0].tolist() != sp.batch([t])[0].tolist()]
        if bad:
            dst.unlink()
            print(f"tokenizer: SentencePiece mismatch on {bad}; keeping tokenizer.json")
        else:
            print(f"tokenizer: SentencePiece matches tokenizer.json on {len(SAMPLES)} samples -> using it")
    shutil.rmtree(cache / "_spm", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
