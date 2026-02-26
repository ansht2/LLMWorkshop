import hashlib
import json
import os
import time
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

DEMO_DATA_DIR = Path("demo_data")
DEMO_INDEX_DIR = Path("prof_embeddings_demo")
MANIFEST_PATH = DEMO_INDEX_DIR / "ingested_files.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, str]) -> None:
    DEMO_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def configure_models() -> None:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before running this script.")

    os.environ["GOOGLE_API_KEY"] = api_key
    Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-001",
        embed_batch_size=1,
        retries=8,
        retry_min_seconds=2,
        retry_max_seconds=30,
    )


def load_single_file(path: Path):
    docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    for doc in docs:
        doc.metadata["source_filename"] = path.name
    return docs


def main() -> None:
    if not DEMO_DATA_DIR.exists():
        raise RuntimeError(f"Missing {DEMO_DATA_DIR}")

    configure_models()

    all_files = sorted(DEMO_DATA_DIR.glob("*.txt"))
    if not all_files:
        print("No files in demo_data/")
        return

    manifest = load_manifest()
    pending_files = []
    for p in all_files:
        digest = file_sha256(p)
        if manifest.get(p.name) != digest:
            pending_files.append((p, digest))

    if not pending_files:
        print("No new or changed files. Index is already up to date.")
        return

    index_exists = (DEMO_INDEX_DIR / "index_store.json").exists()
    if index_exists:
        storage_context = StorageContext.from_defaults(persist_dir=str(DEMO_INDEX_DIR))
        index = load_index_from_storage(storage_context)
        print(f"Loaded existing demo index. Appending {len(pending_files)} file(s).")
    else:
        first_file, first_digest = pending_files[0]
        first_docs = load_single_file(first_file)
        index = VectorStoreIndex.from_documents(first_docs, show_progress=True)
        index.storage_context.persist(persist_dir=str(DEMO_INDEX_DIR))
        manifest[first_file.name] = first_digest
        save_manifest(manifest)
        pending_files = pending_files[1:]
        print(f"Created demo index with {first_file.name}.")

    for i, (path, digest) in enumerate(pending_files, start=1):
        docs = load_single_file(path)
        for doc in docs:
            index.insert(doc)
        index.storage_context.persist(persist_dir=str(DEMO_INDEX_DIR))
        manifest[path.name] = digest
        save_manifest(manifest)
        print(f"[{i}/{len(pending_files)}] Appended {path.name}")
        time.sleep(0.6)

    print("Done. Demo index updated incrementally.")


if __name__ == "__main__":
    main()
