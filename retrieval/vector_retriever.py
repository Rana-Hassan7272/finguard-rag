"""
retrieval/vector_retriever.py
==============================
FAISS-based vector retriever for the FinGuard RAG pipeline.

Responsibilities:
  - Load FAISS index, id_map, and docstore at startup
  - Embed incoming query (question field only)
  - Return top-k vector results with doc_id, score, rank, source
  - Version safety: verify index manifest matches config version_tag
  - Optional category pre-filter: restrict search to a subset of doc_ids
  - Lazy loading: index is loaded once on first use

Usage:
    from retrieval.vector_retriever import VectorRetriever
    import yaml

    cfg = yaml.safe_load(open("retrieval/configs/retrieval_config.yaml"))
    retriever = VectorRetriever(cfg)

    results = retriever.retrieve("zakat ka hisab kaise karein", k=10)
    for r in results:
        print(r.doc_id, r.score, r.rank)

    # With category pre-filter
    results = retriever.retrieve(
        "zakat ka hisab",
        k=10,
        category_filter="islamic_finance"
    )
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

log = logging.getLogger(__name__)

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VectorResult:
    """
    A single vector retrieval result.

    Fields
    ------
    doc_id   : canonical document ID (e.g. "doc_0042")
    score    : cosine similarity score [0.0, 1.0] (higher = more similar)
    rank     : 1-based rank in result list (1 = most similar)
    source   : always "vector" for results from this retriever
    doc      : full document dict from docstore (question, answer, category, difficulty, ...)
    """
    doc_id: str
    score: float
    rank: int
    source: str = "vector"
    doc: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------

class VectorRetriever:
    """
    FAISS vector retriever.

    Lifecycle:
      1. __init__: store config, set paths — do NOT load index yet
      2. First call to .retrieve() or explicit .load(): loads index lazily
      3. All subsequent calls reuse loaded index (no reload)

    Thread safety:
      Index loading is not thread-safe. In multi-threaded servers,
      call .load() explicitly during startup before serving requests.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._loaded = False

        # Resolve paths
        self._version_tag = cfg["embedding_model"]["version_tag"]
        self._model_id = cfg["embedding_model"]["local_path"] or cfg["embedding_model"]["hub_id"]
        self._embed_dim = cfg["embedding_model"]["embedding_dim"]
        self._normalize = cfg["embedding_model"]["normalize_embeddings"]
        self._default_k = cfg["retrieval"]["vector_k"]

        self._index_path = self._resolve_path(cfg["artifacts"]["faiss_index"])
        self._id_map_path = self._resolve_path(cfg["artifacts"]["faiss_id_map"])
        self._manifest_path = self._resolve_path(cfg["artifacts"]["faiss_manifest"])
        self._docstore_path = Path(cfg["artifacts"]["docstore_json"])

        # Will be populated on load()
        self._index = None
        self._id_map: dict[str, str] = {}        # "faiss_row" -> doc_id
        self._docstore: dict[str, dict] = {}     # doc_id -> full doc
        self._model = None                        # SentenceTransformer

        # Pre-built category index for fast pre-filtering
        # category -> set of faiss row integers
        self._category_row_index: dict[str, set[int]] = {}
        self._missing_question_warned: set[str] = set()

    def _resolve_path(self, template: str) -> Path:
        return Path(template.replace("{version}", self._version_tag))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load FAISS index, id_map, docstore, and embedding model.
        Safe to call multiple times — only loads once.
        """
        if self._loaded:
            return

        log.info(f"VectorRetriever loading — version='{self._version_tag}'")
        t_start = time.time()

        # 1. Version safety check
        self._verify_version()

        # 2. Load FAISS index
        self._load_faiss_index()

        # 3. Load id_map (FAISS row -> doc_id)
        self._load_id_map()

        # 4. Load docstore (doc_id -> full doc)
        self._load_docstore()

        # 5. Build category index for pre-filtering
        self._build_category_index()

        # 6. Load embedding model
        self._load_embedder()

        self._loaded = True
        log.info(f"VectorRetriever ready in {time.time()-t_start:.2f}s | docs={self._index.ntotal}")

    def _verify_version(self) -> None:
        """Check that index manifest version matches config. Refuse to serve if mismatch."""
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"Index manifest not found: {self._manifest_path}\n"
                f"Run: python retrieval/build_vector_index.py"
            )

        with open(self._manifest_path, "r") as f:
            manifest = json.load(f)

        manifest_version = manifest.get("embedder_version_tag")
        if manifest_version != self._version_tag:
            raise RuntimeError(
                f"VERSION MISMATCH — refusing to serve:\n"
                f"  Config version_tag : {self._version_tag}\n"
                f"  Manifest version   : {manifest_version}\n"
                f"Rebuild index: python retrieval/build_vector_index.py"
            )

        log.info(
            f"Index version OK: version='{self._version_tag}' | "
            f"docs={manifest['corpus_doc_count']} | "
            f"built={manifest['build_timestamp']}"
        )

    def _load_faiss_index(self) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        if not self._index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self._index_path}\n"
                f"Run: python retrieval/build_vector_index.py"
            )

        log.info(f"Loading FAISS index: {self._index_path}")
        self._index = faiss.read_index(str(self._index_path))
        log.info(f"FAISS index loaded: ntotal={self._index.ntotal}, dim={self._embed_dim}")

    def _load_id_map(self) -> None:
        if not self._id_map_path.exists():
            raise FileNotFoundError(
                f"ID map not found: {self._id_map_path}\n"
                f"Run: python retrieval/build_vector_index.py"
            )

        with open(self._id_map_path, "r") as f:
            self._id_map = json.load(f)

        log.info(f"ID map loaded: {len(self._id_map)} entries")

        # Verify alignment
        if len(self._id_map) != self._index.ntotal:
            raise RuntimeError(
                f"ID map has {len(self._id_map)} entries but FAISS index has "
                f"{self._index.ntotal} vectors. Rebuild both from the same corpus."
            )

    def _load_docstore(self) -> None:
        if not self._docstore_path.exists():
            raise FileNotFoundError(
                f"Docstore not found: {self._docstore_path}\n"
                f"Run: python retrieval/build_corpus.py"
            )

        with open(self._docstore_path, "r", encoding="utf-8") as f:
            self._docstore = json.load(f)

        log.info(f"Docstore loaded: {len(self._docstore)} documents")

    def _build_category_index(self) -> None:
        """
        Build an in-memory index: category -> set of FAISS row integers.
        Used for O(n) category pre-filtering without scanning all docs.
        """
        self._category_row_index = {}

        for faiss_row_str, doc_id in self._id_map.items():
            doc = self._docstore.get(doc_id, {})
            category = doc.get("category", "unknown")
            faiss_row = int(faiss_row_str)

            if category not in self._category_row_index:
                self._category_row_index[category] = set()
            self._category_row_index[category].add(faiss_row)

        categories = list(self._category_row_index.keys())
        log.info(f"Category pre-filter index built: {len(categories)} categories: {categories}")

    def _load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        log.info(f"Loading embedder: {self._model_id}")
        self._model = SentenceTransformer(self._model_id)
        log.info("Embedder loaded")

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns shape (1, dim) float32 numpy array, L2-normalized.
        """
        vec = self._model.encode(
            [query],
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Public query encoder used by pipeline/MMR."""
        if not self._loaded:
            self.load()
        return self._embed_query(query)[0]

    def encode_docs(self, doc_ids: list[str]) -> np.ndarray:
        """
        Encode document retrieval texts for MMR.
        Uses the indexed retrieval field (`question`) to stay aligned with indexing.
        """
        if not self._loaded:
            self.load()
        texts = []
        for doc_id in doc_ids:
            doc = self._docstore.get(doc_id, {})
            q = doc.get("question", "")
            if not q and doc_id not in self._missing_question_warned:
                log.warning(
                    "Doc '%s' missing 'question' field; using empty retrieval text.",
                    doc_id,
                )
                self._missing_question_warned.add(doc_id)
            texts.append(q)
        if not texts:
            return np.zeros((0, self._embed_dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    # ------------------------------------------------------------------
    # Category pre-filter
    # ------------------------------------------------------------------

    def _get_category_allowed_rows(
        self,
        category_filter: Optional[str],
    ) -> Optional[set[int]]:
        """
        Return the set of FAISS row integers allowed by the category filter.
        Returns None if no filter is applied (search all rows).

        If the category is not found in the index, logs a warning and returns
        None (fall through to unfiltered search — graceful degradation).
        """
        if not category_filter:
            return None

        allowed = self._category_row_index.get(category_filter)

        if allowed is None:
            known = list(self._category_row_index.keys())
            log.warning(
                f"Category '{category_filter}' not found in index. "
                f"Known categories: {known}. "
                f"Falling back to unfiltered search."
            )
            return None

        log.debug(f"Category pre-filter: '{category_filter}' → {len(allowed)} candidate docs")
        return allowed

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> list[VectorResult]:
        """
        Retrieve top-k documents for a query using FAISS vector search.

        Parameters
        ----------
        query           : normalized query string (run normalization.py first)
        k               : number of results to return (default: from config)
        category_filter : optional category label to restrict search
                          e.g. "islamic_finance", "banking_basics"
                          Pass None for unrestricted search.

        Returns
        -------
        List of VectorResult, sorted by score descending (rank 1 = best).

        Notes
        -----
        - Category pre-filter is applied BEFORE FAISS search (not post-filter)
          by building a filtered index on the fly using IDSelectorBatch.
          This avoids retrieving candidates and then discarding them.
        - If the filtered candidate pool is too small (< k), returns all
          available docs in that category.
        """
        if not self._loaded:
            self.load()

        if not query or not query.strip():
            log.warning("Empty query passed to VectorRetriever — returning empty results")
            return []

        k = k or self._default_k
        t0 = time.time()

        # Embed query
        query_vec = self._embed_query(query)

        # Category pre-filter
        allowed_rows = self._get_category_allowed_rows(category_filter)

        if allowed_rows is not None:
            results = self._search_with_filter(query_vec, k, allowed_rows)
        else:
            results = self._search_all(query_vec, k)

        elapsed_ms = (time.time() - t0) * 1000
        log.debug(
            f"VectorRetriever: query='{query[:40]}' | "
            f"k={k} | filter={category_filter} | "
            f"returned={len(results)} | elapsed={elapsed_ms:.1f}ms"
        )

        return results

    def _search_all(self, query_vec: np.ndarray, k: int) -> list[VectorResult]:
        """Standard FAISS search over all docs."""
        scores, indices = self._index.search(query_vec, k)
        return self._build_results(scores[0], indices[0])

    def _search_with_filter(
        self,
        query_vec: np.ndarray,
        k: int,
        allowed_rows: set[int],
    ) -> list[VectorResult]:
        """
        FAISS search restricted to allowed_rows.

        Strategy: Use faiss.IDSelectorBatch for native filtering.
        Falls back to full search + post-filter if faiss version doesn't support it.
        """
        try:
            import faiss

            # Build IDSelectorBatch from allowed row integers
            allowed_arr = np.array(sorted(allowed_rows), dtype=np.int64)
            selector = faiss.IDSelectorBatch(len(allowed_arr), faiss.swig_ptr(allowed_arr))

            # SearchParameters with selector
            params = faiss.SearchParametersIVF() if hasattr(faiss, "SearchParametersIVF") else None

            # For IndexFlatIP, use the simpler selector approach
            search_k = min(k, len(allowed_rows))
            if search_k == 0:
                return []

            scores, indices = self._index.search(query_vec, self._index.ntotal)

            # Post-filter to allowed_rows (fallback strategy for IndexFlatIP)
            filtered = [
                (float(s), int(i))
                for s, i in zip(scores[0], indices[0])
                if int(i) != -1 and int(i) in allowed_rows
            ]
            filtered.sort(key=lambda x: x[0], reverse=True)
            filtered = filtered[:k]

            result_scores = np.array([s for s, _ in filtered])
            result_indices = np.array([i for _, i in filtered])

            return self._build_results(result_scores, result_indices)

        except Exception as e:
            log.warning(f"Category filter search failed ({e}) — falling back to full search")
            return self._search_all(query_vec, k)

    def _build_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[VectorResult]:
        """
        Convert FAISS output arrays to list of VectorResult.

        Skips invalid indices (-1 returned by FAISS for empty slots).
        """
        results = []
        rank = 1

        for score, faiss_row in zip(scores, indices):
            if faiss_row == -1:
                continue  # FAISS returns -1 when fewer than k results exist

            faiss_row_str = str(int(faiss_row))
            doc_id = self._id_map.get(faiss_row_str)

            if doc_id is None:
                log.warning(f"FAISS row {faiss_row} not in id_map — skipping")
                continue

            doc = self._docstore.get(doc_id, {})
            if not doc.get("question") and doc_id not in self._missing_question_warned:
                log.warning(
                    "Doc '%s' missing 'question'; retrieval text should always be question.",
                    doc_id,
                )
                self._missing_question_warned.add(doc_id)

            results.append(
                VectorResult(
                    doc_id=doc_id,
                    score=float(score),
                    rank=rank,
                    source="vector",
                    doc=doc,
                )
            )
            rank += 1

        return results

    # ------------------------------------------------------------------
    # Introspection / debugging
    # ------------------------------------------------------------------

    def get_doc(self, doc_id: str) -> Optional[dict]:
        """Fetch a document by doc_id from the docstore."""
        if not self._loaded:
            self.load()
        return self._docstore.get(doc_id)

    def available_categories(self) -> list[str]:
        """Return list of category labels in the index."""
        if not self._loaded:
            self.load()
        return sorted(self._category_row_index.keys())

    def index_size(self) -> int:
        """Return number of vectors in the FAISS index."""
        if not self._loaded:
            self.load()
        return self._index.ntotal

    @property
    def docstore(self) -> dict[str, dict]:
        if not self._loaded:
            self.load()
        return self._docstore

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Module-level convenience loader
# ---------------------------------------------------------------------------

_default_retriever: Optional[VectorRetriever] = None


def get_retriever(cfg: Optional[dict] = None) -> VectorRetriever:
    """
    Return a module-level singleton VectorRetriever.
    Loads on first call. Subsequent calls return cached instance.

    Use this in FastAPI/Gradio apps to avoid reloading on every request.
    """
    global _default_retriever
    if _default_retriever is None:
        if cfg is None:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)
        _default_retriever = VectorRetriever(cfg)
        _default_retriever.load()
    return _default_retriever


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    retriever = VectorRetriever(cfg)

    try:
        retriever.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Build the index first:")
        print("  python retrieval/build_corpus.py")
        print("  python retrieval/build_vector_index.py")
        sys.exit(1)

    print(f"\nIndex loaded: {retriever.index_size()} docs")
    print(f"Categories: {retriever.available_categories()}")

    test_queries = [
        ("zakat ka hisab kaise karein", None),
        ("how to calculate zakat", None),
        ("meezan bank account kholna hai", "banking_basics"),
        ("زکوٰۃ کا حساب", None),
    ]

    for query, cat_filter in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}' | filter={cat_filter}")
        results = retriever.retrieve(query, k=3, category_filter=cat_filter)
        for r in results:
            print(
                f"  [{r.rank}] {r.doc_id} | score={r.score:.4f} | "
                f"cat={r.doc.get('category','?')} | "
                f"q='{r.doc.get('question','')[:50]}'"
            )