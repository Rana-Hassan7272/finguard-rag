import json
import pickle
import hashlib
from pathlib import Path
from rank_bm25 import BM25Okapi
import sys

PDF_CORPUS_PATH = Path("retrieval/artifacts/pdf_corpus.jsonl")
PDF_BM25_INDEX_PATH = Path("retrieval/artifacts/pdf_bm25.pkl")

try:
    from retrieval.normalization import tokenize
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from retrieval.normalization import tokenize


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


def build_bm25(docs: list[dict]) -> BM25Okapi:
    corpus = [tokenize(d["retrieval_text"]) for d in docs]
    return BM25Okapi(corpus)


def save_index(bm25: BM25Okapi, docs: list[dict], corpus_version: str):
    PDF_BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": bm25,
        "docs": docs,
        "doc_ids": [d["doc_id"] for d in docs],
        "corpus_version": corpus_version,
    }
    with open(PDF_BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)


def main():
    docs = load_corpus(PDF_CORPUS_PATH)
    corpus_version = compute_corpus_version(docs)
    bm25 = build_bm25(docs)
    save_index(bm25, docs, corpus_version)
    print(f"PDF BM25 index built: {len(docs)} chunks, corpus_version={corpus_version}")


if __name__ == "__main__":
    main()