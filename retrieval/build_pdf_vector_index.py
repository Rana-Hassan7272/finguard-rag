import hashlib
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")
PDF_CORPUS_PATH = Path("retrieval/artifacts/pdf_corpus.jsonl")
PDF_VECTOR_INDEX_PATH = Path("retrieval/artifacts/pdf_faiss.index")
PDF_VECTOR_META_PATH = Path("retrieval/artifacts/pdf_faiss_meta.pkl")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_corpus(path: Path) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def compute_corpus_version(docs: list[dict]) -> str:
    content = json.dumps([d["doc_id"] for d in docs], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def build_index(docs: list[dict], model: SentenceTransformer) -> tuple[faiss.Index, list[dict]]:
    texts = [d["retrieval_text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, docs


def save_index(index: faiss.Index, docs: list[dict], corpus_version: str, embedder_version: str):
    PDF_VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(PDF_VECTOR_INDEX_PATH))
    meta = {
        "corpus_version": corpus_version,
        "embedder_version": embedder_version,
        "doc_ids": [d["doc_id"] for d in docs],
        "docs": docs,
    }
    with open(PDF_VECTOR_META_PATH, "wb") as f:
        pickle.dump(meta, f)


def main():
    cfg = load_config()
    model_name = cfg["embedding_model"].get("local_path") or cfg["embedding_model"]["hub_id"]
    embedder_version = cfg["embedding_model"]["version_tag"]
    model = SentenceTransformer(model_name)
    docs = load_corpus(PDF_CORPUS_PATH)
    corpus_version = compute_corpus_version(docs)
    index, docs = build_index(docs, model)
    save_index(index, docs, corpus_version, embedder_version)
    print(f"PDF vector index built: {len(docs)} chunks, corpus_version={corpus_version}")


if __name__ == "__main__":
    main()