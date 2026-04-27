"""
retrieval/build_bm25_index.py
==============================
Builds BM25 index artifacts from the deduplicated corpus.

Responsibilities:
  - Load corpus.jsonl (built by build_corpus.py)
  - Tokenize question text with light normalization
  - Build BM25 structure using rank_bm25 (BM25Okapi)
  - Serialize tokenized corpus and manifest for runtime loading
  - Ensure doc_id alignment with the vector index side

NOTE: BM25 is stateless at runtime — we serialize the tokenized corpus
and reconstruct BM25Okapi from it on each server startup. This is fast
and avoids pickling BM25 objects across Python versions.

Run:
  python retrieval/build_bm25_index.py
  python retrieval/build_bm25_index.py --force
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

import yaml
try:
    from retrieval.normalization import tokenize
except ModuleNotFoundError:
    # Allow direct script execution: python retrieval/build_bm25_index.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from retrieval.normalization import tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_bm25_index")

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_artifact_path(template: str, version: str) -> Path:
    return Path(template.replace("{version}", version))


# ---------------------------------------------------------------------------
# Minimal stopword lists
# "minimal" mode — only remove the most common function words
# We keep domain terms like "kaise", "kya", "hai" because they carry
# semantic weight in Roman Urdu queries
# ---------------------------------------------------------------------------

ENGLISH_STOPWORDS_MINIMAL = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "that", "this", "these", "those", "it", "its",
}

# Roman Urdu minimal stopwords — keep domain terms, only remove pure filler
ROMAN_URDU_STOPWORDS_MINIMAL = {
    "aur", "ya", "bhi", "sirf", "woh", "yeh",
}

ALL_MINIMAL_STOPWORDS = ENGLISH_STOPWORDS_MINIMAL | ROMAN_URDU_STOPWORDS_MINIMAL


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        log.error(f"Corpus not found: {corpus_path} — run build_corpus.py first")
        sys.exit(1)

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    log.info(f"Loaded {len(docs)} documents from corpus")
    return docs


# ---------------------------------------------------------------------------
# Alignment check with vector index
# ---------------------------------------------------------------------------

def check_alignment_with_vector_index(docs: list[dict], cfg: dict) -> bool:
    """
    Verify that the corpus loaded here has the same doc_ids in the same order
    as the vector index id_map.

    If misaligned, FAISS row indices and BM25 row indices will map to different
    docs — silent correctness failure. Catch it here.
    """
    version_tag = cfg["embedding_model"]["version_tag"]
    id_map_path = resolve_artifact_path(cfg["artifacts"]["faiss_id_map"], version_tag)

    if not id_map_path.exists():
        log.warning(
            f"Vector id_map not found at {id_map_path} — skipping alignment check. "
            f"Build vector index first if running full pipeline."
        )
        return True  # non-fatal; index may not be built yet

    with open(id_map_path, "r") as f:
        id_map = json.load(f)

    # id_map keys are string FAISS row indices
    vector_doc_ids = [id_map[str(i)] for i in range(len(id_map))]
    bm25_doc_ids = [doc["doc_id"] for doc in docs]

    if len(vector_doc_ids) != len(bm25_doc_ids):
        log.error(
            f"ALIGNMENT MISMATCH: vector index has {len(vector_doc_ids)} docs, "
            f"BM25 corpus has {len(bm25_doc_ids)} docs. "
            f"Rebuild both from the same corpus.jsonl."
        )
        return False

    mismatches = [
        (i, v, b)
        for i, (v, b) in enumerate(zip(vector_doc_ids, bm25_doc_ids))
        if v != b
    ]
    if mismatches:
        log.error(
            f"ALIGNMENT MISMATCH at {len(mismatches)} positions. "
            f"First mismatch: row={mismatches[0][0]}, "
            f"vector={mismatches[0][1]}, bm25={mismatches[0][2]}"
        )
        return False

    log.info(f"Alignment check PASSED: BM25 and vector index are in sync ({len(docs)} docs)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_bm25_index(args: argparse.Namespace) -> None:
    cfg = load_config()

    version_tag = cfg["embedding_model"]["version_tag"]
    tokenizer_mode = cfg["bm25"]["tokenizer_mode"]
    stopword_mode = cfg["bm25"]["stopword_mode"]
    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]

    tokens_path = resolve_artifact_path(cfg["artifacts"]["bm25_tokens"], version_tag)
    manifest_path = resolve_artifact_path(cfg["artifacts"]["bm25_manifest"], version_tag)

    # Safety: do not overwrite without --force
    if manifest_path.exists() and not args.force:
        log.error(
            f"BM25 index already exists for version '{version_tag}': {manifest_path}\n"
            f"To overwrite: --force | To build new version: update version_tag in config."
        )
        sys.exit(1)

    tokens_path.parent.mkdir(parents=True, exist_ok=True)

    # Load corpus
    corpus_path = Path(cfg["artifacts"]["corpus_jsonl"])
    docs = load_corpus(corpus_path)

    if len(docs) == 0:
        log.error("Corpus is empty. Run build_corpus.py first.")
        sys.exit(1)

    # Alignment check with vector side
    if not check_alignment_with_vector_index(docs, cfg):
        if not args.force_alignment:
            log.error("Aborting due to alignment mismatch. Use --force_alignment to override.")
            sys.exit(1)
        log.warning("--force_alignment set — continuing despite alignment mismatch (RISKY)")

    # Tokenize all questions
    log.info(f"Tokenizing {len(docs)} questions (mode={tokenizer_mode}, stopwords={stopword_mode})...")
    t0 = time.time()

    tokenized_corpus = []
    empty_token_count = 0
    token_length_samples = []

    for doc in docs:
        tokens = tokenize(doc["question"], tokenizer_mode, stopword_mode)
        tokenized_corpus.append(tokens)

        if not tokens:
            empty_token_count += 1
            log.warning(f"doc_id={doc['doc_id']} produced empty token list: '{doc['question'][:60]}'")

        token_length_samples.append(len(tokens))

    elapsed = time.time() - t0
    avg_len = sum(token_length_samples) / len(token_length_samples) if token_length_samples else 0

    log.info(
        f"Tokenization complete: elapsed={elapsed:.2f}s | "
        f"avg_tokens_per_doc={avg_len:.1f} | empty_docs={empty_token_count}"
    )

    # Validate BM25 works on this tokenized corpus
    log.info("Validating BM25 build...")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.error("Run: pip install rank-bm25")
        sys.exit(1)

    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Quick sanity test: score a known question against corpus
    test_doc = docs[0]
    test_tokens = tokenize(test_doc["question"], tokenizer_mode, stopword_mode)
    test_scores = bm25.get_scores(test_tokens)
    top_idx = int(test_scores.argmax())

    if top_idx == 0:
        log.info(f"BM25 self-retrieval test PASSED: doc_0 retrieved itself (score={test_scores[top_idx]:.4f})")
    else:
        log.warning(
            f"BM25 self-retrieval test: doc_0 did not retrieve itself — "
            f"top was doc_{top_idx} (score={test_scores[top_idx]:.4f}). "
            f"This may be OK if doc_0 has very short question text."
        )

    # Corpus fingerprint for manifest
    combined = "|".join(":".join(tokens) for tokens in tokenized_corpus)
    corpus_token_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # Build artifacts payload
    # Store: list of {"doc_id": ..., "tokens": [...]}
    # Keep doc_id alongside tokens for alignment verification at load time
    corpus_tokens_artifact = [
        {"doc_id": doc["doc_id"], "tokens": tokens}
        for doc, tokens in zip(docs, tokenized_corpus)
    ]

    manifest = {
        "embedder_version_tag": version_tag,
        "tokenizer_mode": tokenizer_mode,
        "stopword_mode": stopword_mode,
        "bm25_k1": k1,
        "bm25_b": b,
        "corpus_doc_count": len(docs),
        "corpus_token_hash": corpus_token_hash,
        "avg_tokens_per_doc": round(avg_len, 2),
        "empty_token_docs": empty_token_count,
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_paths": {
            "bm25_tokens": str(tokens_path),
            "manifest": str(manifest_path),
        },
    }

    # Write artifacts
    log.info(f"Writing BM25 token corpus → {tokens_path}")
    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump(corpus_tokens_artifact, f, ensure_ascii=False, indent=1)

    log.info(f"Writing BM25 manifest → {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("=== BM25 Index Build Complete ===")
    log.info(f"  Version tag          : {version_tag}")
    log.info(f"  Documents indexed    : {len(docs)}")
    log.info(f"  Avg tokens/doc       : {avg_len:.1f}")
    log.info(f"  Empty token docs     : {empty_token_count}")
    log.info(f"  Corpus token hash    : {corpus_token_hash}")
    log.info(f"  Token corpus file    : {tokens_path}")
    log.info(f"  Manifest             : {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BM25 index from deduplicated corpus"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing BM25 index for this version tag",
    )
    parser.add_argument(
        "--force_alignment",
        action="store_true",
        help="Continue even if BM25 and vector corpus doc_ids are misaligned (RISKY)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_bm25_index(args)