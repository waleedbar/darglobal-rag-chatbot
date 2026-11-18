"""Push this repo to a Hugging Face Docker Space and set its secrets (run by GitHub Actions).

Env: HF_TOKEN (write token), HF_SPACE (e.g. user/darglobal-rag-chatbot),
     OPENROUTER_API_KEY and DATA_KEY (stored as Space secrets, never in files).
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FRONT_MATTER = """---
title: DarGlobal & Wasalt RAG Chatbot
emoji: 🏙️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Bilingual RAG over public Dar Global & Wasalt data
---

"""
SKIP = {".git", ".github", "tests", "index", "models", "__pycache__", ".pytest_cache"}


def main() -> None:
    api, space = HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_SPACE"]
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    for name in ("OPENROUTER_API_KEY", "DATA_KEY"):
        if os.environ.get(name):
            api.add_space_secret(space, name, os.environ[name])
    api.add_space_variable(space, "PUBLIC_URL", f"https://{space.replace('/', '-').lower()}.hf.space")
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "space"
        shutil.copytree(ROOT, dst, ignore=lambda d, names: [n for n in names if n in SKIP or n.endswith(".pyc")])
        for f in (dst / "data" / "raw" / "wasalt.jsonl",):  # never upload plain Wasalt data
            f.unlink(missing_ok=True)
        readme = dst / "README.md"
        readme.write_text(FRONT_MATTER + readme.read_text(encoding="utf-8"), encoding="utf-8")
        api.upload_folder(folder_path=str(dst), repo_id=space, repo_type="space",
                          commit_message="Deploy from GitHub", delete_patterns=["*"])
    print(f"deployed: https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
