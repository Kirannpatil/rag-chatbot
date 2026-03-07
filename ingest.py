"""
ingest.py
─────────
Reads all documents from the data/ folder using
LlamaParse for accurate table extraction,
chunks them, converts to embeddings, and saves
the vector index to storage/.

Run this ONCE (or whenever you add new documents):
    python ingest.py
"""

import os
from dotenv import load_dotenv
load_dotenv()
import shutil
from llama_parse import LlamaParse
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Embedding model (runs fully locally, no API key needed) ──
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── No LLM needed during ingestion ──
Settings.llm = None

# ── LlamaParse API Key ──
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")


def main():
    print("=" * 50)
    print("  RAG Ingest — LlamaParse Edition")
    print("=" * 50)

    if not os.listdir(DATA_DIR):
        print(f"\n❌  No files found in {DATA_DIR}")
        print("    Put your return_policy.pdf there and re-run.\n")
        return

    # ── LlamaParse — table aware PDF parser ──
    print("\n📄  Parsing PDF with LlamaParse...")
    parser = LlamaParse(
        api_key=LLAMA_PARSE_API_KEY,
        result_type="markdown",       # converts table to clean markdown
        verbose=True,
        language="en",
    )

    # Use LlamaParse for PDFs, fallback to default for other files
    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        DATA_DIR,
        file_extractor=file_extractor
    ).load_data()

    print(f"✅  Loaded {len(documents)} document chunk(s)")

    # Preview first chunk so you can verify extraction quality
    if documents:
        print(f"\n--- Preview of first chunk ---")
        print(documents[0].text[:500])
        print("------------------------------\n")

    print("⚙️   Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        chunk_size=512,
        chunk_overlap=64,
        show_progress=True,
    )

    # Clear old storage and save new
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
        print("🗑️   Cleared old index")

    os.makedirs(STORAGE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"\n✅  Index saved to: {STORAGE_DIR}")
    print("\n🚀  Now run:  python app.py\n")


if __name__ == "__main__":
    main()
