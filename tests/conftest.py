import os
import tempfile

os.environ.setdefault("EMBED_BACKEND", "hash")
os.environ.setdefault("LLM_BACKEND", "echo")
if "INDEX_DIR" not in os.environ:
    os.environ["INDEX_DIR"] = tempfile.mkdtemp()
    from rag.embeddings import HashEmbedder
    from rag.store import build

    build(HashEmbedder())
