"""
ingestion/pdf_corpus_builder.py

Converts raw chunks from pdf_chunker.py into the canonical PDF corpus schema,
runs content-hash deduplication, infers categories, and saves:
  - retrieval/artifacts/pdf_corpus.jsonl
  - retrieval/artifacts/pdf_docstore.json
  - retrieval/artifacts/pdf_corpus_stats.json

Run:
  python ingestion/pdf_corpus_builder.py
  python ingestion/pdf_corpus_builder.py --pdf_dir data/documents --dry_run
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

try:
    from ingestion.infer_category import infer_category_batch
    from ingestion.pdf_chunker import load_and_chunk
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ingestion.infer_category import infer_category_batch
    from ingestion.pdf_chunker import load_and_chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pdf_corpus_builder")

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")
DEFAULT_PDF_DIR = "data/documents"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dedup_by_content(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    seen: set[str] = set()
    kept, removed = [], []
    for chunk in chunks:
        h = content_hash(chunk["retrieval_text"])
        if h in seen:
            removed.append(chunk)
        else:
            seen.add(h)
            chunk["content_hash"] = h
            kept.append(chunk)
    log.info(f"Content dedup: kept={len(kept)}, removed={len(removed)}")
    return kept, removed


def build_canonical_doc(chunk: dict, idx: int) -> dict:
    return {
        "doc_id": chunk["doc_id"],
        "doc_type": "pdf_chunk",
        "retrieval_text": chunk["retrieval_text"],
        "chunk_text": chunk["chunk_text"],
        "source_file": chunk["source_file"],
        "page_no": chunk.get("page_no"),
        "chunk_id": chunk.get("chunk_id", idx),
        "parent_section": chunk.get("parent_section"),
        "category": chunk.get("category", "general"),
        "category_confidence": chunk.get("category_confidence", 0.0),
        "difficulty": None,
        "content_hash": chunk.get("content_hash", content_hash(chunk["retrieval_text"])),
    }


def build_corpus(args: argparse.Namespace) -> None:
    cfg = load_config()
    artifacts_root = Path(cfg["artifacts"]["root"])
    artifacts_root.mkdir(parents=True, exist_ok=True)

    out_corpus = artifacts_root / "pdf_corpus.jsonl"
    out_docstore = artifacts_root / "pdf_docstore.json"
    out_stats = artifacts_root / "pdf_corpus_stats.json"

    t0 = time.time()
    raw_chunks = load_and_chunk(
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    log.info(f"Loaded {len(raw_chunks)} raw chunks in {time.time()-t0:.1f}s")

    raw_chunks = infer_category_batch(raw_chunks, text_field="retrieval_text")
    kept, removed = dedup_by_content(raw_chunks)

    docs = [build_canonical_doc(chunk, idx) for idx, chunk in enumerate(kept)]

    cat_dist = Counter(d["category"] for d in docs)
    file_dist = Counter(d["source_file"] for d in docs)
    stats = {
        "total_raw_chunks": len(raw_chunks),
        "total_removed_duplicates": len(removed),
        "total_final_docs": len(docs),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "category_distribution": dict(cat_dist),
        "file_distribution": dict(file_dist),
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    log.info("=== PDF Corpus Build Summary ===")
    log.info(f"  Raw chunks    : {stats['total_raw_chunks']}")
    log.info(f"  Removed (dup) : {stats['total_removed_duplicates']}")
    log.info(f"  Final docs    : {stats['total_final_docs']}")
    log.info(f"  Categories    : {dict(cat_dist)}")
    log.info(f"  Files         : {len(file_dist)} PDFs indexed")

    if args.dry_run:
        log.info("DRY RUN — no files written.")
        return

    with open(out_corpus, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    docstore = {doc["doc_id"]: doc for doc in docs}
    with open(out_docstore, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info(f"Artifacts written to {artifacts_root}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PDF retrieval corpus")
    p.add_argument("--pdf_dir", default=DEFAULT_PDF_DIR)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=100)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    build_corpus(parse_args())