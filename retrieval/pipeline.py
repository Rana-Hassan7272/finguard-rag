"""
retrieval/pipeline.py
======================
Retrieval pipeline orchestrator — Phases 2, 3, and 5.

Orchestration order:
  1. Normalize query
  2. Detect language → fusion weights
  3. Detect category (Phase 5)
  4. Apply filter policy → category_filter string or None
  5. Vector retrieval (top-k, with optional category pre-filter)
  6. BM25 retrieval (top-k, with optional category pre-filter)
  7. Weighted RRF fusion
  8. MMR diversity filter → final top-10
  9. Return docs + full diagnostics payload

Usage:
    from retrieval.pipeline import RetrievalPipeline
    import yaml

    cfg = yaml.safe_load(open("retrieval/configs/retrieval_config.yaml"))
    pipeline = RetrievalPipeline(cfg)
    pipeline.load()

    output = pipeline.run("zakat ka hisab kaise karein")
    print(output.docs)           # list of top-10 FusedResult
    print(output.diagnostics)    # full debug payload
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from metadata.category_detector import CategoryDetector
from metadata.filter_policy import FilterDecision, FilterPolicy
from query.expander import expand_query
from query.router import route_query
from retrieval.bm25_retriever import BM25Retriever
from retrieval.fusion import FusedResult, compute_fusion_stats, fuse
from retrieval.language import LanguageResult, detect_and_get_weights
from retrieval.mmr import MMRResult, run_mmr
from retrieval.normalization import normalize_query
from retrieval.vector_retriever import VectorRetriever
from retrieval.dual_retriever import DualRetriever

log = logging.getLogger(__name__)

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

@dataclass
class RetrievalOutput:
    """
    Full output of one pipeline run.

    Fields
    ------
    query_raw        : original query string before normalization
    query_normalized : query after normalization.py
    docs             : final ranked docs after MMR (list of MMRResult)
    diagnostics      : complete debug payload dict
    total_ms         : wall-clock time for full pipeline run
    """
    query_raw: str
    query_normalized: str
    docs: list[MMRResult]
    diagnostics: dict
    total_ms: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """
    End-to-end retrieval pipeline: normalize → detect → retrieve → fuse → MMR.

    All components are loaded once at startup and reused across requests.
    The pipeline is stateless per-request (safe for concurrent use after load()).

    Loading:
        Call pipeline.load() explicitly at server startup.
        Or let it lazy-load on first run() call.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._loaded = False

        # Config values
        self._vector_k = cfg["retrieval"]["vector_k"]
        self._bm25_k = cfg["retrieval"]["bm25_k"]
        self._rrf_k = cfg["fusion"]["rrf_k"]
        self._mmr_lambda = cfg["mmr"]["lambda"]
        self._mmr_output_k = cfg["mmr"]["output_k"]

        # Components — initialized on load()
        self._vector_retriever: Optional[VectorRetriever] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._category_detector: Optional[CategoryDetector] = None
        self._filter_policy: Optional[FilterPolicy] = None
        self._dual_retriever: Optional[DualRetriever] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all pipeline components. Safe to call multiple times."""
        if self._loaded:
            return

        log.info("RetrievalPipeline loading all components...")
        t0 = time.time()

        self._vector_retriever = VectorRetriever(self.cfg)
        self._vector_retriever.load()

        self._bm25_retriever = BM25Retriever(self.cfg)
        self._bm25_retriever.load()

        self._category_detector = CategoryDetector(self.cfg)

        # Register known categories from the loaded vector index
        known_cats = self._vector_retriever.available_categories()
        self._filter_policy = FilterPolicy(self.cfg, known_categories=known_cats)
        try:
            self._dual_retriever = DualRetriever(self.cfg)
        except Exception as e:
            self._dual_retriever = None
            log.warning(f"DualRetriever unavailable, falling back to QA-only pipeline: {e}")

        self._loaded = True
        log.info(
            f"RetrievalPipeline ready in {time.time()-t0:.2f}s | "
            f"vector_docs={self._vector_retriever.index_size()} | "
            f"bm25_docs={self._bm25_retriever.corpus_size()} | "
            f"known_categories={known_cats}"
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        vector_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        mmr_output_k: Optional[int] = None,
        force_category: Optional[str] = None,
    ) -> RetrievalOutput:
        """
        Run the full retrieval pipeline for a single query.

        Parameters
        ----------
        query          : raw user query (any language/script)
        vector_k       : override config vector_k for this request
        bm25_k         : override config bm25_k for this request
        mmr_output_k   : override config mmr_output_k for this request
        force_category : bypass detection and force this category filter
                         (useful for testing and eval ablation)

        Returns
        -------
        RetrievalOutput with docs (MMRResult list) and full diagnostics
        """
        self._ensure_loaded()

        t_total = time.time()
        stage_times: dict[str, float] = {}

        # ── Stage 1: Normalize + route + expand (Phase 7) ──────────────
        t = time.time()
        route = route_query(query)
        query_normalized = route["normalized_query"]
        query_expanded = expand_query(query_normalized, language=route["language"])
        stage_times["normalize_route_expand_ms"] = round((time.time() - t) * 1000, 2)

        if not query_normalized:
            log.warning(f"Query normalized to empty string: '{query}'")
            return self._empty_output(query, query_normalized, stage_times)

        # ── Stage 2: Language fusion weights ────────────────────────────
        t = time.time()
        lang_result, bm25_weight, vector_weight = detect_and_get_weights(
            query_normalized, self.cfg
        )
        stage_times["language_ms"] = round((time.time() - t) * 1000, 2)

        log.debug(
            f"Language: {lang_result.label} (conf={lang_result.confidence}) | "
            f"weights: bm25={bm25_weight}, vec={vector_weight}"
        )

        # ── Stage 3: Category detection (Phase 5) ───────────────────────
        t = time.time()
        if force_category is not None:
            # Bypass detector — used in eval ablation
            from metadata.category_detector import CategoryDetectionResult
            detection = CategoryDetectionResult(
                predicted_category=force_category,
                confidence=1.0,
                matched_keywords=["[forced]"],
                all_scores={force_category: 1.0},
                method="forced",
            )
        else:
            detection = self._category_detector.detect(query_normalized)

        stage_times["category_detect_ms"] = round((time.time() - t) * 1000, 2)

        # ── Stage 4: Filter policy ──────────────────────────────────────
        t = time.time()
        filter_decision: FilterDecision = self._filter_policy.decide(detection)
        stage_times["filter_policy_ms"] = round((time.time() - t) * 1000, 2)

        category_filter = filter_decision.category_filter

        log.debug(
            f"Filter: apply={filter_decision.apply_filter} | "
            f"category={category_filter} | reason={filter_decision.reason}"
        )

        use_dual = bool(self._dual_retriever) and self.cfg.get("phase6_7", {}).get("enable_dual_retriever", True)

        # ── Stage 5+: Dual retrieval path (Phase 6) ─────────────────────
        if use_dual:
            t = time.time()
            ok = mmr_output_k or self._mmr_output_k
            dual_out = self._dual_retriever.retrieve(
                query=query_expanded,
                language=route["language"],
                query_intent=route["query_intent"],
                category_filter=category_filter,
                top_k=ok,
            )
            stage_times["dual_retrieve_ms"] = round((time.time() - t) * 1000, 2)

            mmr_results = [
                MMRResult(
                    doc_id=d.get("doc_id", ""),
                    relevance_score=round(float(d.get("_rrf_score", 0.0)), 4),
                    novelty_score=1.0,
                    mmr_score=round(float(d.get("_rrf_score", 0.0)), 4),
                    rank=i + 1,
                    metadata={"doc": d, "sources": d.get("_source", "unknown")},
                )
                for i, d in enumerate(dual_out.docs)
            ]

            total_ms = round((time.time() - t_total) * 1000, 2)
            diagnostics = {
                "query_raw": query,
                "query_normalized": query_normalized,
                "query_expanded": query_expanded,
                "language_detected": lang_result.label,
                "language_confidence": lang_result.confidence,
                "weights_used": {"bm25": bm25_weight, "vector": vector_weight, "source": lang_result.label},
                "query_intent": route["query_intent"],
                "source_routing_weights": {"qa": dual_out.diagnostics.get("qa_weight"), "pdf": dual_out.diagnostics.get("pdf_weight")},
                "category_detected": detection.predicted_category,
                "category_confidence": detection.confidence,
                "filter_applied": filter_decision.apply_filter,
                "filter_reason": filter_decision.reason,
                "source_mix": {
                    "qa_docs": dual_out.diagnostics.get("qa_docs", 0),
                    "pdf_docs": dual_out.diagnostics.get("pdf_docs", 0),
                },
                "stage_latency_ms": stage_times,
                "total_ms": total_ms,
                "mode": "dual_retriever",
            }
            return RetrievalOutput(
                query_raw=query,
                query_normalized=query_normalized,
                docs=mmr_results,
                diagnostics=diagnostics,
                total_ms=total_ms,
            )

        # ── Stage 5: Vector retrieval (QA-only fallback) ────────────────
        t = time.time()
        vk = vector_k or self._vector_k
        vector_results = self._vector_retriever.retrieve(
            query_expanded, k=vk, category_filter=category_filter
        )
        stage_times["vector_ms"] = round((time.time() - t) * 1000, 2)

        # ── Stage 6: BM25 retrieval ─────────────────────────────────────
        t = time.time()
        bk = bm25_k or self._bm25_k
        bm25_results = self._bm25_retriever.retrieve(
            query_expanded, k=bk, category_filter=category_filter
        )
        stage_times["bm25_ms"] = round((time.time() - t) * 1000, 2)

        # ── Stage 7: Weighted RRF fusion ────────────────────────────────
        t = time.time()
        fused = fuse(
            vector_results=vector_results,
            bm25_results=bm25_results,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rrf_k=self._rrf_k,
        )
        fusion_stats = compute_fusion_stats(fused)
        stage_times["fusion_ms"] = round((time.time() - t) * 1000, 2)

        if not fused:
            log.warning(f"No fused results for query: '{query_normalized}'")
            return self._empty_output(query, query_normalized, stage_times)

        # ── Stage 8: MMR ────────────────────────────────────────────────
        t = time.time()
        ok = mmr_output_k or self._mmr_output_k

        # Build candidate embeddings for MMR using same vector embedder.
        candidate_ids = [r.doc_id for r in fused]
        candidate_scores = [r.rrf_score for r in fused]
        candidate_embeddings = self._vector_retriever.encode_docs(candidate_ids)
        query_vec = self._vector_retriever.encode_query(query_expanded)
        candidate_meta = [
            {
                "doc": r.doc,
                "sources": r.sources,
                "rrf_score": r.rrf_score,
            }
            for r in fused
        ]
        mmr_results = run_mmr(
            query_embedding=query_vec,
            candidate_ids=candidate_ids,
            candidate_embeddings=candidate_embeddings,
            candidate_scores=candidate_scores,
            top_k=ok,
            lambda_param=self._mmr_lambda,
            metadata=candidate_meta,
        )
        stage_times["mmr_ms"] = round((time.time() - t) * 1000, 2)

        total_ms = round((time.time() - t_total) * 1000, 2)

        # ── Diagnostics payload ─────────────────────────────────────────
        diagnostics = self._build_diagnostics(
            query_raw=query,
            query_normalized=query_normalized,
            query_expanded=query_expanded,
            lang_result=lang_result,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            query_intent=route["query_intent"],
            detection=detection,
            filter_decision=filter_decision,
            vector_results=vector_results,
            bm25_results=bm25_results,
            fused=fused,
            fusion_stats=fusion_stats,
            mmr_results=mmr_results,
            stage_times=stage_times,
            total_ms=total_ms,
        )

        return RetrievalOutput(
            query_raw=query,
            query_normalized=query_normalized,
            docs=mmr_results,
            diagnostics=diagnostics,
            total_ms=total_ms,
        )

    # ------------------------------------------------------------------
    # Diagnostics builder
    # ------------------------------------------------------------------

    def _build_diagnostics(
        self,
        query_raw: str,
        query_normalized: str,
        query_expanded: str,
        lang_result: LanguageResult,
        bm25_weight: float,
        vector_weight: float,
        query_intent: str,
        detection,
        filter_decision: FilterDecision,
        vector_results: list,
        bm25_results: list,
        fused: list[FusedResult],
        fusion_stats: dict,
        mmr_results: list[MMRResult],
        stage_times: dict,
        total_ms: float,
    ) -> dict:
        """
        Build complete diagnostics payload.

        Every decision made by the pipeline is captured here:
        language detection, fusion weights, filter policy, per-source
        rankings, fused rankings, and post-MMR rankings.

        This payload is logged to JSONL for observability and can be
        shown in the Gradio UI for explainability.
        """
        return {
            # ── Query ──
            "query_raw": query_raw,
            "query_normalized": query_normalized,
            "query_expanded": query_expanded,

            # ── Language detection ──
            "language_detected": lang_result.label,
            "language_confidence": lang_result.confidence,
            "language_detail": {
                "urdu_ratio": lang_result.urdu_ratio,
                "english_ratio": lang_result.english_ratio,
                "roman_urdu_ratio": lang_result.roman_urdu_ratio,
            },

            # ── Fusion weights ──
            "weights_used": {
                "bm25": bm25_weight,
                "vector": vector_weight,
                "source": lang_result.label,
            },
            "query_intent": query_intent,

            # ── Phase 5: Category / filter ──
            "category_detected": detection.predicted_category,
            "category_confidence": detection.confidence,
            "category_matched_keywords": detection.matched_keywords,
            "category_all_scores": detection.all_scores,
            "filter_applied": filter_decision.apply_filter,
            "filter_reason": filter_decision.reason,

            # ── Per-source top docs ──
            "vector_top_docs": [
                {
                    "rank": r.rank,
                    "doc_id": r.doc_id,
                    "score": round(r.score, 6),
                    "category": r.doc.get("category"),
                    "question": r.doc.get("question", "")[:80],
                }
                for r in vector_results[:5]
            ],
            "bm25_top_docs": [
                {
                    "rank": r.rank,
                    "doc_id": r.doc_id,
                    "score": round(r.score, 4),
                    "score_norm": round(r.score_norm, 4),
                    "category": r.doc.get("category"),
                    "question": r.doc.get("question", "")[:80],
                }
                for r in bm25_results[:5]
            ],

            # ── Fused ranking ──
            "fused_ranking": [
                {
                    "rank": r.rank,
                    "doc_id": r.doc_id,
                    "rrf_score": round(r.rrf_score, 6),
                    "sources": r.sources,
                    "vector_rank": r.vector_rank,
                    "bm25_rank": r.bm25_rank,
                    "v_contrib": round(r.vector_rrf_contrib, 6),
                    "b_contrib": round(r.bm25_rrf_contrib, 6),
                    "question": r.doc.get("question", "")[:80],
                }
                for r in fused[:10]
            ],
            "fusion_stats": fusion_stats,

            # ── Post-MMR ranking ──
            "mmr_final_docs": [
                {
                    "rank": r.rank,
                    "doc_id": r.doc_id,
                    "relevance_score": round(r.relevance_score, 6),
                    "novelty_score": round(r.novelty_score, 6),
                    "mmr_score": round(r.mmr_score, 6),
                    "sources": r.metadata.get("sources"),
                    "category": r.metadata.get("doc", {}).get("category"),
                    "question": r.metadata.get("doc", {}).get("question", "")[:80],
                    "answer_preview": r.metadata.get("doc", {}).get("answer", "")[:120],
                }
                for r in mmr_results
            ],

            # ── Latency per stage ──
            "stage_latency_ms": stage_times,
            "total_ms": total_ms,
        }

    # ------------------------------------------------------------------
    # Empty output helper
    # ------------------------------------------------------------------

    def _empty_output(
        self, query_raw: str, query_normalized: str, stage_times: dict
    ) -> RetrievalOutput:
        return RetrievalOutput(
            query_raw=query_raw,
            query_normalized=query_normalized,
            docs=[],
            diagnostics={
                "query_raw": query_raw,
                "query_normalized": query_normalized,
                "stage_latency_ms": stage_times,
                "total_ms": sum(stage_times.values()),
                "error": "empty query or no results",
            },
            total_ms=sum(stage_times.values()),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def available_categories(self) -> list[str]:
        self._ensure_loaded()
        return self._vector_retriever.available_categories()

    def filter_stats(self) -> dict:
        """Return filter policy session stats."""
        self._ensure_loaded()
        return self._filter_policy.get_stats()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_pipeline: Optional[RetrievalPipeline] = None


def get_pipeline(cfg: Optional[dict] = None) -> RetrievalPipeline:
    """Return module-level singleton RetrievalPipeline."""
    global _pipeline
    if _pipeline is None:
        if cfg is None:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)
        _pipeline = RetrievalPipeline(cfg)
        _pipeline.load()
    return _pipeline


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline = RetrievalPipeline(cfg)

    try:
        pipeline.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Build indexes first:")
        print("  python retrieval/build_corpus.py")
        print("  python retrieval/build_vector_index.py")
        print("  python retrieval/build_bm25_index.py")
        sys.exit(1)

    test_queries = [
        "zakat ka hisab kaise karein",
        "meezan bank account kholna hai",
        "how to calculate zakat nisab",
        "easypaisa se paise bhejein",
        "رباء کیا ہے اسلام میں",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        output = pipeline.run(query)
        d = output.diagnostics
        print(f"Query     : {output.query_raw}")
        print(f"Normalized: {output.query_normalized}")
        print(f"Language  : {d['language_detected']} (conf={d['language_confidence']:.2f})")
        print(f"Weights   : bm25={d['weights_used']['bm25']} | vec={d['weights_used']['vector']}")
        print(f"Category  : {d['category_detected']} (conf={d['category_confidence']:.3f})")
        print(f"Filter    : {d['filter_applied']} — {d['filter_reason']}")
        print(f"Latency   : {output.total_ms:.1f}ms | {d['stage_latency_ms']}")
        print(f"\nTop-{len(output.docs)} docs after MMR:")
        for r in output.docs:
            print(
                f"  [{r.rank}] {r.doc_id} | "
                f"mmr={r.mmr_score:.4f} | rel={r.relevance_score:.4f} | "
                f"nov={r.novelty_score:.4f} | src={r.metadata.get('sources')} | "
                f"cat={r.metadata.get('doc',{}).get('category')} | "
                f"q='{r.metadata.get('doc',{}).get('question','')[:55]}'"
            )