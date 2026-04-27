"""
retrieval/fusion.py
====================
Reciprocal Rank Fusion (RRF) for the FinGuard hybrid search pipeline.

Implements language-aware weighted RRF that combines vector and BM25 results
into a single ranked list, with full per-source contribution tracing.

Theory:
  Standard RRF score for a doc:
      rrf(d) = Σ  1 / (k + rank_i(d))
             i ∈ retrievers

  Weighted RRF (our extension):
      rrf(d) = Σ  w_i / (k + rank_i(d))
             i ∈ retrievers

  Where w_i is the language-aware weight for retriever i.
  This lets us smoothly emphasize BM25 for English and vector for Roman Urdu.

Usage:
    from retrieval.fusion import fuse

    fused = fuse(
        vector_results=vector_results,
        bm25_results=bm25_results,
        bm25_weight=0.3,
        vector_weight=0.7,
        rrf_k=60,
    )
    for item in fused:
        print(item.doc_id, item.rrf_score, item.sources)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fused result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusedResult:
    """
    A single document in the fused ranked list.

    Fields
    ------
    doc_id          : canonical doc ID
    rrf_score       : final weighted RRF score (higher = more relevant)
    rank            : 1-based rank in fused list
    doc             : full document dict
    sources         : which retrievers returned this doc ("vector", "bm25", "both")
    vector_rank     : rank in vector results (None if not retrieved by vector)
    bm25_rank       : rank in BM25 results (None if not retrieved by BM25)
    vector_rrf_contrib : weighted RRF contribution from vector side
    bm25_rrf_contrib   : weighted RRF contribution from BM25 side
    vector_score    : raw vector similarity score (None if not in vector results)
    bm25_score      : raw BM25 score (None if not in BM25 results)
    bm25_score_norm : normalized BM25 score (None if not in BM25 results)
    """
    doc_id: str
    rrf_score: float
    rank: int
    doc: dict = field(default_factory=dict)
    sources: str = "unknown"              # "vector" | "bm25" | "both"
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    vector_rrf_contrib: float = 0.0
    bm25_rrf_contrib: float = 0.0
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    bm25_score_norm: Optional[float] = None


# ---------------------------------------------------------------------------
# Core RRF computation
# ---------------------------------------------------------------------------

def _rrf_contribution(rank: int, rrf_k: int, weight: float) -> float:
    """
    Weighted RRF contribution for a single retriever result.

    rrf_k = 60 is the canonical constant from the original Cormack et al. paper.
    Higher rrf_k → smoother rank differences (top ranks matter less).
    Lower rrf_k → more aggressive top-rank boosting.
    """
    return weight / (rrf_k + rank)


def fuse(
    vector_results: list,           # list[VectorResult]
    bm25_results: list,             # list[BM25Result]
    bm25_weight: float,
    vector_weight: float,
    rrf_k: int = 60,
    top_k: Optional[int] = None,
) -> list[FusedResult]:
    """
    Weighted Reciprocal Rank Fusion of vector and BM25 result lists.

    Parameters
    ----------
    vector_results  : ranked list from VectorRetriever (VectorResult objects)
    bm25_results    : ranked list from BM25Retriever (BM25Result objects)
    bm25_weight     : weight for BM25 contribution (from language detection)
    vector_weight   : weight for vector contribution (from language detection)
    rrf_k           : RRF constant (default 60)
    top_k           : truncate output to top_k items (None = return all)

    Returns
    -------
    List of FusedResult sorted by rrf_score descending.

    Design notes:
    - Docs appearing in BOTH retrievers get contributions from both sides
      (the "both" bonus is a key property of RRF that rewards consensus)
    - Docs appearing in only one retriever still get their single contribution
    - We track per-source contributions for diagnostics and debugging
    - Weights should sum to 1.0 but we do NOT enforce this — allows flexibility
      e.g. (0.3, 0.7) or (0.5, 0.5) or any other ratio
    """
    if not vector_results and not bm25_results:
        log.warning("fuse() called with empty vector AND bm25 results — returning empty")
        return []

    # Accumulator: doc_id -> score components
    rrf_scores: dict[str, float] = {}
    vector_contribs: dict[str, float] = {}
    bm25_contribs: dict[str, float] = {}

    # Index results by doc_id for metadata lookup
    vector_by_id: dict[str, object] = {}
    bm25_by_id: dict[str, object] = {}

    # Process vector results
    for result in vector_results:
        doc_id = result.doc_id
        contrib = _rrf_contribution(result.rank, rrf_k, vector_weight)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contrib
        vector_contribs[doc_id] = contrib
        vector_by_id[doc_id] = result

    # Process BM25 results
    for result in bm25_results:
        doc_id = result.doc_id
        contrib = _rrf_contribution(result.rank, rrf_k, bm25_weight)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contrib
        bm25_contribs[doc_id] = contrib
        bm25_by_id[doc_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    # Build FusedResult objects with full trace
    fused = []
    for rank, doc_id in enumerate(sorted_ids, start=1):
        in_vector = doc_id in vector_by_id
        in_bm25 = doc_id in bm25_by_id

        if in_vector and in_bm25:
            sources = "both"
            doc = vector_by_id[doc_id].doc     # prefer vector doc (always hydrated)
        elif in_vector:
            sources = "vector"
            doc = vector_by_id[doc_id].doc
        else:
            sources = "bm25"
            doc = bm25_by_id[doc_id].doc

        v_result = vector_by_id.get(doc_id)
        b_result = bm25_by_id.get(doc_id)

        fused.append(
            FusedResult(
                doc_id=doc_id,
                rrf_score=rrf_scores[doc_id],
                rank=rank,
                doc=doc,
                sources=sources,
                vector_rank=v_result.rank if v_result else None,
                bm25_rank=b_result.rank if b_result else None,
                vector_rrf_contrib=vector_contribs.get(doc_id, 0.0),
                bm25_rrf_contrib=bm25_contribs.get(doc_id, 0.0),
                vector_score=v_result.score if v_result else None,
                bm25_score=b_result.score if b_result else None,
                bm25_score_norm=b_result.score_norm if b_result else None,
            )
        )

    log.debug(
        f"RRF fusion: vector={len(vector_results)} | bm25={len(bm25_results)} | "
        f"merged={len(sorted_ids)} unique | "
        f"weights=(bm25={bm25_weight}, vec={vector_weight}) | rrf_k={rrf_k}"
    )

    return fused


# ---------------------------------------------------------------------------
# Fusion stats (for diagnostics / logging)
# ---------------------------------------------------------------------------

def compute_fusion_stats(fused: list[FusedResult]) -> dict:
    """
    Compute statistics on a fused result list for diagnostics.
    Used by pipeline.py to build the debug payload.
    """
    if not fused:
        return {}

    both_count = sum(1 for r in fused if r.sources == "both")
    vector_only = sum(1 for r in fused if r.sources == "vector")
    bm25_only = sum(1 for r in fused if r.sources == "bm25")
    total = len(fused)

    scores = [r.rrf_score for r in fused]
    v_contribs = [r.vector_rrf_contrib for r in fused]
    b_contribs = [r.bm25_rrf_contrib for r in fused]

    return {
        "total_candidates": total,
        "source_overlap": {
            "both": both_count,
            "vector_only": vector_only,
            "bm25_only": bm25_only,
            "overlap_rate": round(both_count / total, 3) if total > 0 else 0.0,
        },
        "rrf_score_range": {
            "max": round(max(scores), 6),
            "min": round(min(scores), 6),
            "top1": round(scores[0], 6) if scores else 0.0,
        },
        "avg_vector_contrib": round(sum(v_contribs) / total, 6) if total > 0 else 0.0,
        "avg_bm25_contrib": round(sum(b_contribs) / total, 6) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataclasses import dataclass as dc

    # Minimal mock result objects for testing
    @dc
    class MockVec:
        doc_id: str
        score: float
        rank: int
        source: str = "vector"
        doc: dict = field(default_factory=dict)

    @dc
    class MockBM25:
        doc_id: str
        score: float
        score_norm: float
        rank: int
        source: str = "bm25"
        doc: dict = field(default_factory=dict)

    # Simulate: vector and BM25 partly agree, partly disagree
    vector_results = [
        MockVec("doc_0001", 0.92, 1, doc={"question": "zakat ka hisab", "category": "islamic_finance"}),
        MockVec("doc_0002", 0.88, 2, doc={"question": "nisab kya hai", "category": "islamic_finance"}),
        MockVec("doc_0007", 0.76, 3, doc={"question": "riba haram hai", "category": "islamic_finance"}),
        MockVec("doc_0012", 0.71, 4, doc={"question": "meezan account", "category": "banking"}),
        MockVec("doc_0019", 0.65, 5, doc={"question": "interest rate", "category": "loans"}),
    ]

    bm25_results = [
        MockBM25("doc_0002", 14.2, 1.00, 1, doc={"question": "nisab kya hai", "category": "islamic_finance"}),
        MockBM25("doc_0001", 12.8, 0.90, 2, doc={"question": "zakat ka hisab", "category": "islamic_finance"}),
        MockBM25("doc_0031", 9.1,  0.64, 3, doc={"question": "zakat on gold", "category": "islamic_finance"}),
        MockBM25("doc_0007", 6.4,  0.45, 4, doc={"question": "riba haram hai", "category": "islamic_finance"}),
        MockBM25("doc_0044", 3.2,  0.23, 5, doc={"question": "savings account", "category": "banking"}),
    ]

    print("=== RRF Fusion Test ===")
    print(f"\nRoman Urdu weights: bm25=0.30, vector=0.70")
    fused = fuse(vector_results, bm25_results, bm25_weight=0.30, vector_weight=0.70)
    print(f"\n{'Rank':<5} {'doc_id':<10} {'rrf_score':<12} {'sources':<8} "
          f"{'vec_rank':<10} {'bm25_rank':<10} {'v_contrib':<12} {'b_contrib'}")
    print("-" * 80)
    for r in fused:
        print(
            f"{r.rank:<5} {r.doc_id:<10} {r.rrf_score:<12.6f} {r.sources:<8} "
            f"{str(r.vector_rank):<10} {str(r.bm25_rank):<10} "
            f"{r.vector_rrf_contrib:<12.6f} {r.bm25_rrf_contrib:.6f}"
        )

    stats = compute_fusion_stats(fused)
    print(f"\nStats: {stats}")

    print(f"\nEnglish weights: bm25=0.60, vector=0.40")
    fused_en = fuse(vector_results, bm25_results, bm25_weight=0.60, vector_weight=0.40)
    print(f"Top-3: {[(r.doc_id, r.sources, round(r.rrf_score, 5)) for r in fused_en[:3]]}")