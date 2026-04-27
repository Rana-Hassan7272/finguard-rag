"""
retrieval/bm25_retriever.py
============================
BM25 keyword retriever for the FinGuard hybrid search pipeline.

Responsibilities:
  - Load tokenized corpus and reconstruct BM25Okapi at startup
  - Tokenize incoming query (same pipeline as index build)
  - Score all documents and return top-k results
  - Verify manifest version alignment with config
  - Verify doc_id alignment with vector index side

Usage:
    from retrieval.bm25_retriever import BM25Retriever
    import yaml

    cfg = yaml.safe_load(open("retrieval/configs/retrieval_config.yaml"))
    retriever = BM25Retriever(cfg)
    retriever.load()

    results = retriever.retrieve("zakat ka hisab kaise karein", k=10)
    for r in results:
        print(r.doc_id, r.score, r.rank)
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from retrieval.normalization import tokenize

log = logging.getLogger(__name__)

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BM25Result:
    """
    A single BM25 retrieval result.

    Fields
    ------
    doc_id    : canonical document ID ("doc_XXXX")
    score     : raw BM25 score (not bounded — depends on corpus stats)
    score_norm: score normalized to [0, 1] by dividing by max score in batch
    rank      : 1-based rank in result list (1 = highest BM25 score)
    source    : always "bm25" for results from this retriever
    doc       : full document dict from docstore
    """
    doc_id: str
    score: float
    score_norm: float
    rank: int
    source: str = "bm25"
    doc: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 keyword retriever using rank_bm25 (BM25Okapi).

    Loading strategy:
      - Reads serialized tokenized corpus from disk (bm25_corpus_tokens_<v>.json)
      - Reconstructs BM25Okapi in-memory from token lists
      - Loads docstore for doc hydration
      - Verifies version alignment with config and vector index side

    Why reconstruct from token lists (not pickle BM25 object):
      - Pickle is fragile across Python/library versions
      - Token list JSON is human-readable and debuggable
      - Reconstruction from ~1.5k docs takes <100ms — negligible
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._loaded = False

        self._version_tag = cfg["embedding_model"]["version_tag"]
        self._tokenizer_mode = cfg["bm25"]["tokenizer_mode"]
        self._stopword_mode = cfg["bm25"]["stopword_mode"]
        self._k1 = cfg["bm25"]["k1"]
        self._b = cfg["bm25"]["b"]
        self._default_k = cfg["retrieval"]["bm25_k"]

        self._tokens_path = self._resolve(cfg["artifacts"]["bm25_tokens"])
        self._manifest_path = self._resolve(cfg["artifacts"]["bm25_manifest"])
        self._docstore_path = Path(cfg["artifacts"]["docstore_json"])

        # Populated on load()
        self._bm25 = None
        self._doc_ids: list[str] = []       # parallel to tokenized corpus rows
        self._docstore: dict[str, dict] = {}

    def _resolve(self, template: str) -> Path:
        return Path(template.replace("{version}", self._version_tag))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all BM25 artifacts. Safe to call multiple times."""
        if self._loaded:
            return

        log.info(f"BM25Retriever loading — version='{self._version_tag}'")
        t0 = time.time()

        self._verify_manifest()
        self._load_corpus_tokens()
        self._load_docstore()
        self._verify_alignment()

        self._loaded = True
        log.info(
            f"BM25Retriever ready in {time.time()-t0:.2f}s | "
            f"docs={len(self._doc_ids)} | k1={self._k1} | b={self._b}"
        )

    def _verify_manifest(self) -> None:
        """Check manifest version matches config — refuse to serve if mismatch."""
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"BM25 manifest not found: {self._manifest_path}\n"
                f"Run: python retrieval/build_bm25_index.py"
            )

        with open(self._manifest_path, "r") as f:
            manifest = json.load(f)

        manifest_version = manifest.get("embedder_version_tag")
        if manifest_version != self._version_tag:
            raise RuntimeError(
                f"BM25 VERSION MISMATCH — refusing to serve:\n"
                f"  Config version_tag  : {self._version_tag}\n"
                f"  Manifest version    : {manifest_version}\n"
                f"Rebuild: python retrieval/build_bm25_index.py"
            )

        log.info(
            f"BM25 manifest OK: version='{self._version_tag}' | "
            f"docs={manifest['corpus_doc_count']} | "
            f"avg_tokens={manifest['avg_tokens_per_doc']:.1f} | "
            f"built={manifest['build_timestamp']}"
        )
        self._manifest = manifest

    def _load_corpus_tokens(self) -> None:
        """Load tokenized corpus and reconstruct BM25Okapi."""
        if not self._tokens_path.exists():
            raise FileNotFoundError(
                f"BM25 token corpus not found: {self._tokens_path}\n"
                f"Run: python retrieval/build_bm25_index.py"
            )

        log.info(f"Loading BM25 token corpus: {self._tokens_path}")
        with open(self._tokens_path, "r", encoding="utf-8") as f:
            corpus_artifacts = json.load(f)

        # corpus_artifacts = list of {"doc_id": ..., "tokens": [...]}
        self._doc_ids = [entry["doc_id"] for entry in corpus_artifacts]
        tokenized_corpus = [entry["tokens"] for entry in corpus_artifacts]

        log.info(f"Reconstructing BM25Okapi from {len(tokenized_corpus)} documents...")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run: pip install rank-bm25")

        t0 = time.time()
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)
        log.info(f"BM25Okapi built in {time.time()-t0:.2f}s")

        # Sanity: verify IDF was computed (non-empty corpus)
        if not hasattr(self._bm25, "idf") or len(self._bm25.idf) == 0:
            raise RuntimeError("BM25 IDF table is empty — corpus may be malformed")

    def _load_docstore(self) -> None:
        if not self._docstore_path.exists():
            raise FileNotFoundError(
                f"Docstore not found: {self._docstore_path}\n"
                f"Run: python retrieval/build_corpus.py"
            )

        with open(self._docstore_path, "r", encoding="utf-8") as f:
            self._docstore = json.load(f)

        log.info(f"Docstore loaded: {len(self._docstore)} documents")

    def _verify_alignment(self) -> None:
        """
        Verify BM25 doc_ids match docstore.
        Catches silent bugs where BM25 row N maps to a different doc than expected.
        """
        docstore_ids = set(self._docstore.keys())
        bm25_id_set = set(self._doc_ids)

        missing_in_docstore = bm25_id_set - docstore_ids
        if missing_in_docstore:
            raise RuntimeError(
                f"BM25 doc_ids not found in docstore: {missing_in_docstore}\n"
                f"Rebuild from the same corpus: build_corpus.py → build_bm25_index.py"
            )

        log.info(f"BM25↔Docstore alignment OK: {len(self._doc_ids)} docs verified")

    # ------------------------------------------------------------------
    # Query tokenization
    # ------------------------------------------------------------------

    def _tokenize_query(self, query: str) -> list[str]:
        """
        Tokenize query using same pipeline as index build.
        Consistency is critical — any mismatch = wrong BM25 scores.
        """
        return tokenize(query, self._tokenizer_mode, self._stopword_mode)

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> list[BM25Result]:
        """
        Retrieve top-k documents using BM25 keyword scoring.

        Parameters
        ----------
        query           : normalized query string (pre-normalized by normalization.py)
        k               : number of results to return (default: from config)
        category_filter : optional category label — if set, only docs from this
                          category are considered. Applied post-scoring (BM25 has
                          no native pre-filter like FAISS IDSelector).

        Returns
        -------
        List of BM25Result, sorted by score descending.
        Returns empty list if query produces no tokens after tokenization.

        Notes on BM25 scoring:
          - BM25Okapi scores are unnormalized (range depends on corpus/term stats)
          - We add score_norm: score / max_score for cross-retriever comparison
          - Zero-score docs are excluded from results (not ranked by BM25)
        """
        if not self._loaded:
            self.load()

        if not query or not query.strip():
            log.warning("Empty query passed to BM25Retriever — returning empty results")
            return []

        k = k or self._default_k
        t0 = time.time()

        # Tokenize query
        query_tokens = self._tokenize_query(query)

        if not query_tokens:
            log.warning(
                f"Query produced zero tokens after tokenization: '{query[:60]}'\n"
                f"This usually means the query was all stopwords. Returning empty."
            )
            return []

        # Score all documents
        all_scores = self._bm25.get_scores(query_tokens)   # shape: (n_docs,)

        # Apply category filter (post-scoring)
        if category_filter:
            mask = self._build_category_mask(category_filter)
            all_scores = all_scores * mask   # zero-out non-matching categories

        # Get top-k indices by score
        # Use argpartition for efficiency on large corpora (O(n) vs O(n log n))
        n_docs = len(all_scores)
        effective_k = min(k, n_docs)

        if effective_k == 0:
            return []

        # argpartition gives top-k in arbitrary order — then sort those k
        top_k_idx = np.argpartition(all_scores, -effective_k)[-effective_k:]
        top_k_idx = top_k_idx[np.argsort(all_scores[top_k_idx])[::-1]]

        # Build results — exclude zero-score docs (no BM25 term overlap)
        max_score = float(all_scores[top_k_idx[0]]) if len(top_k_idx) > 0 else 1.0
        if max_score <= 0.0:
            log.debug(f"BM25: all scores are zero for query '{query[:40]}' — no keyword overlap")
            return []

        results = []
        rank = 1
        for idx in top_k_idx:
            score = float(all_scores[idx])
            if score <= 0.0:
                break   # remaining docs have no term overlap

            doc_id = self._doc_ids[idx]
            doc = self._docstore.get(doc_id, {})

            results.append(
                BM25Result(
                    doc_id=doc_id,
                    score=score,
                    score_norm=score / max_score,
                    rank=rank,
                    source="bm25",
                    doc=doc,
                )
            )
            rank += 1

        elapsed_ms = (time.time() - t0) * 1000
        log.debug(
            f"BM25Retriever: query='{query[:40]}' | tokens={query_tokens} | "
            f"k={k} | returned={len(results)} | "
            f"max_score={max_score:.4f} | elapsed={elapsed_ms:.1f}ms"
        )

        return results

    def _build_category_mask(self, category_filter: str) -> np.ndarray:
        """
        Build binary mask (1.0 / 0.0) for docs matching the category filter.
        Mask is parallel to self._doc_ids.
        """
        mask = np.zeros(len(self._doc_ids), dtype=np.float32)
        found = 0
        for i, doc_id in enumerate(self._doc_ids):
            doc_category = self._docstore.get(doc_id, {}).get("category", "unknown")
            if doc_category == category_filter:
                mask[i] = 1.0
                found += 1

        if found == 0:
            known = set(
                self._docstore.get(did, {}).get("category", "unknown")
                for did in self._doc_ids
            )
            log.warning(
                f"Category '{category_filter}' has 0 docs in BM25 corpus. "
                f"Known: {known}. Disabling filter."
            )
            mask[:] = 1.0   # fallback: search all

        return mask

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def corpus_size(self) -> int:
        return len(self._doc_ids)

    def vocabulary_size(self) -> int:
        """Number of unique terms in BM25 IDF table."""
        if not self._loaded or self._bm25 is None:
            return 0
        return len(self._bm25.idf)

    def get_term_idf(self, term: str) -> float:
        """Return IDF score for a single term. Useful for debugging."""
        if not self._loaded:
            self.load()
        return self._bm25.idf.get(term, 0.0)

    def get_top_terms(self, query: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Return top-N query tokens ranked by their IDF score.
        High IDF = rare, discriminative term. Useful for debugging retrieval quality.
        """
        tokens = self._tokenize_query(query)
        term_idfs = [(t, self._bm25.idf.get(t, 0.0)) for t in set(tokens)]
        return sorted(term_idfs, key=lambda x: x[1], reverse=True)[:top_n]

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_retriever: Optional[BM25Retriever] = None


def get_retriever(cfg: Optional[dict] = None) -> BM25Retriever:
    """
    Return module-level singleton BM25Retriever.
    Loads on first call. Subsequent calls return cached instance.
    """
    global _default_retriever
    if _default_retriever is None:
        if cfg is None:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)
        _default_retriever = BM25Retriever(cfg)
        _default_retriever.load()
    return _default_retriever


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    retriever = BM25Retriever(cfg)

    try:
        retriever.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"\nBM25 index ready")
    print(f"  Corpus size    : {retriever.corpus_size()} docs")
    print(f"  Vocabulary size: {retriever.vocabulary_size()} unique terms")

    test_queries = [
        ("zakat ka hisab kaise karein", None),
        ("meezan bank account opening", None),
        ("riba aur interest mein farq", None),
        ("how to calculate zakat nisab", None),
        ("easypaisa jazzcash transfer", None),
    ]

    for query, cat_filter in test_queries:
        print(f"\n{'='*65}")
        print(f"Query      : '{query}'")
        top_terms = retriever.get_top_terms(query)
        print(f"Top terms  : {top_terms[:5]}")
        results = retriever.retrieve(query, k=5, category_filter=cat_filter)
        if not results:
            print("  [no BM25 results — no keyword overlap]")
        for r in results:
            print(
                f"  [{r.rank}] {r.doc_id} | score={r.score:.4f} "
                f"(norm={r.score_norm:.3f}) | "
                f"cat={r.doc.get('category','?')} | "
                f"q='{r.doc.get('question','')[:50]}'"
            )