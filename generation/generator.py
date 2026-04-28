"""
generation/generator.py

Orchestrates the full generation flow for one user query:

  1. Check reranker confidence gate — hard block if score below threshold
  2. Check semantic cache L1 (answer cache) — return immediately if hit
  3. Check semantic cache L2 (retrieval cache) — skip retrieval if hit
  4. Build prompt from reranked docs
  5. Call LLM (Groq primary / OpenAI fallback)
  6. Store result in cache (L1 + L2)
  7. Return structured GenerationOutput

The generator is the boundary between retrieval and generation.
It never touches FAISS, BM25, or the reranker directly — it receives
already-ranked docs and a query embedding from the pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from cache.cache_stats import CacheStatsTracker, get_tracker
from cache.semantic_cache import SemanticCache
from generation.llm_client import LLMClient, LLMResponse
from generation.prompt_builder import BuiltPrompt, build_fallback_message, build_prompt

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    passed: bool
    top_score: float
    reason: str


@dataclass
class GenerationOutput:
    answer: str
    gate: GateResult
    cache_level: Optional[int]
    cache_hit: bool
    prompt: Optional[BuiltPrompt]
    llm_response: Optional[LLMResponse]
    language: str
    retrieval_doc_ids: list[str]
    total_ms: float
    stage_ms: dict = field(default_factory=dict)


class Generator:
    """
    Confidence gate + cache + LLM orchestrator.

    Parameters
    ----------
    cfg          : full retrieval_config.yaml dict
    cache        : SemanticCache instance (shared with pipeline)
    llm_client   : LLMClient instance
    stats_tracker: CacheStatsTracker for observability
    """

    def __init__(
        self,
        cfg: dict,
        cache: SemanticCache,
        llm_client: LLMClient,
        stats_tracker: Optional[CacheStatsTracker] = None,
    ):
        self.cfg = cfg
        self.cache = cache
        self.llm = llm_client
        self.tracker = stats_tracker or get_tracker()

        reranker_cfg = cfg.get("reranker", {})
        self.confidence_threshold: float = reranker_cfg.get("confidence_threshold", 0.40)

        log.info(
            f"Generator initialized: confidence_threshold={self.confidence_threshold} | "
            f"primary={cfg.get('generation', {}).get('primary_provider', 'groq')}"
        )

    def generate(
        self,
        query: str,
        query_embedding: np.ndarray,
        reranked_docs: list[dict],
        reranker_scores: list[float],
        language: str = "roman_urdu",
    ) -> GenerationOutput:
        """
        Run the full generation flow for one query.

        Parameters
        ----------
        query            : normalized query string
        query_embedding  : L2-normalized query vector (dim=768)
        reranked_docs    : list of doc dicts from cross-encoder reranker, best-first
                           each doc must have at minimum: doc_id, doc_type,
                           question/answer (QA) or retrieval_text/source_file (PDF)
        reranker_scores  : parallel list of float scores from the reranker
        language         : detected query language for fallback message selection

        Returns
        -------
        GenerationOutput with answer, gate result, cache metadata, and LLM info
        """
        t_total = time.time()
        stage_ms: dict[str, float] = {}
        reranked_docs, reranker_scores = self._normalize_inputs(reranked_docs, reranker_scores)

        # ── Stage 1: Confidence gate ─────────────────────────────────────
        t = time.time()
        gate = self._evaluate_gate(reranked_docs, reranker_scores)
        stage_ms["gate_ms"] = round((time.time() - t) * 1000, 2)

        if not gate.passed:
            fallback = build_fallback_message(language)
            log.info(f"Gate BLOCKED: {gate.reason} | query='{query[:50]}'")
            return GenerationOutput(
                answer=fallback,
                gate=gate,
                cache_level=None,
                cache_hit=False,
                prompt=None,
                llm_response=None,
                language=language,
                retrieval_doc_ids=[],
                total_ms=round((time.time() - t_total) * 1000, 2),
                stage_ms=stage_ms,
            )

        # ── Stage 2: Cache lookup ────────────────────────────────────────
        t = time.time()
        cache_result = self.cache.lookup(query_embedding)
        stage_ms["cache_lookup_ms"] = round((time.time() - t) * 1000, 2)

        self.tracker.record(
            hit=cache_result.hit,
            level=cache_result.level,
            similarity=cache_result.similarity,
            query_text=query,
        )

        if cache_result.hit and cache_result.level == 1 and cache_result.answer:
            log.debug(f"Cache L1 HIT: sim={cache_result.similarity} query='{query[:50]}'")
            return GenerationOutput(
                answer=cache_result.answer,
                gate=gate,
                cache_level=1,
                cache_hit=True,
                prompt=None,
                llm_response=None,
                language=language,
                retrieval_doc_ids=cache_result.doc_ids,
                total_ms=round((time.time() - t_total) * 1000, 2),
                stage_ms=stage_ms,
            )
        if cache_result.hit and cache_result.level == 2 and cache_result.doc_ids:
            return GenerationOutput(
                answer="",
                gate=gate,
                cache_level=2,
                cache_hit=True,
                prompt=None,
                llm_response=None,
                language=language,
                retrieval_doc_ids=cache_result.doc_ids,
                total_ms=round((time.time() - t_total) * 1000, 2),
                stage_ms=stage_ms,
            )

        # ── Stage 3: Build prompt ────────────────────────────────────────
        t = time.time()
        built_prompt = build_prompt(question=query, docs=reranked_docs)
        stage_ms["prompt_build_ms"] = round((time.time() - t) * 1000, 2)

        # ── Stage 4: LLM call ────────────────────────────────────────────
        t = time.time()
        llm_response = self.llm.generate(built_prompt.prompt)
        stage_ms["llm_ms"] = round((time.time() - t) * 1000, 2)

        if not llm_response.success or not llm_response.text:
            fallback = build_fallback_message(language)
            log.warning(
                f"LLM failed: {llm_response.error} | "
                f"returning fallback for query='{query[:50]}'"
            )
            return GenerationOutput(
                answer=fallback,
                gate=gate,
                cache_level=None,
                cache_hit=False,
                prompt=built_prompt,
                llm_response=llm_response,
                language=language,
                retrieval_doc_ids=[],
                total_ms=round((time.time() - t_total) * 1000, 2),
                stage_ms=stage_ms,
            )

        answer = llm_response.text.strip()

        # ── Stage 5: Store in cache ──────────────────────────────────────
        t = time.time()
        doc_ids = [d.get("doc_id", "") for d in reranked_docs]
        self.cache.store(
            query_embedding=query_embedding,
            query_text=query,
            doc_ids=doc_ids,
            answer=answer,
        )
        stage_ms["cache_store_ms"] = round((time.time() - t) * 1000, 2)

        log.debug(
            f"Generation complete: provider={llm_response.provider} | "
            f"latency={llm_response.latency_ms:.0f}ms | "
            f"tokens={llm_response.completion_tokens} | "
            f"query='{query[:50]}'"
        )

        return GenerationOutput(
            answer=answer,
            gate=gate,
            cache_level=None,
            cache_hit=False,
            prompt=built_prompt,
            llm_response=llm_response,
            language=language,
            retrieval_doc_ids=[],
            total_ms=round((time.time() - t_total) * 1000, 2),
            stage_ms=stage_ms,
        )

    @staticmethod
    def _normalize_inputs(
        reranked_docs: list,
        reranker_scores: list[float],
    ) -> tuple[list[dict], list[float]]:
        """
        Accept either:
        - list[dict] docs, or
        - list[MMRResult]-style objects from retrieval.pipeline (with metadata.doc)
        """
        if not reranked_docs:
            return [], reranker_scores or []

        if isinstance(reranked_docs[0], dict):
            scores = reranker_scores or [float(d.get("reranker_score", d.get("mmr_score", d.get("_rrf_score", 0.0)))) for d in reranked_docs]
            return reranked_docs, scores

        docs: list[dict] = []
        scores: list[float] = []
        for item in reranked_docs:
            md = getattr(item, "metadata", {}) or {}
            doc = md.get("doc", {})
            docs.append(doc if isinstance(doc, dict) else {})
            if reranker_scores:
                scores.append(float(reranker_scores[min(len(scores), len(reranker_scores) - 1)]))
            else:
                scores.append(float(getattr(item, "mmr_score", 0.0)))
        return docs, scores

    def _evaluate_gate(
        self,
        reranked_docs: list[dict],
        reranker_scores: list[float],
    ) -> GateResult:
        """
        Evaluate confidence gate.

        Rules (in priority order):
          - No docs at all → block (no_docs)
          - Top doc has empty text → block (empty_doc_text)
          - Top reranker score < threshold → block (score_below_threshold)
          - Otherwise → pass
        """
        if not reranked_docs or not reranker_scores:
            return GateResult(
                passed=False,
                top_score=0.0,
                reason="no_docs",
            )

        top_score = float(reranker_scores[0])
        top_doc = reranked_docs[0]

        doc_text = (
            top_doc.get("answer")
            or top_doc.get("retrieval_text")
            or top_doc.get("chunk_text")
            or ""
        ).strip()

        if not doc_text:
            return GateResult(
                passed=False,
                top_score=top_score,
                reason="empty_doc_text",
            )

        if top_score < self.confidence_threshold:
            return GateResult(
                passed=False,
                top_score=top_score,
                reason=f"score_below_threshold ({top_score:.4f} < {self.confidence_threshold})",
            )

        return GateResult(
            passed=True,
            top_score=top_score,
            reason=f"passed ({top_score:.4f} >= {self.confidence_threshold})",
        )


_generator: Optional[Generator] = None


def get_generator(
    cfg: Optional[dict] = None,
    cache: Optional[SemanticCache] = None,
    llm_client: Optional[LLMClient] = None,
) -> Generator:
    """
    Return module-level singleton Generator.
    Call once at server startup — all components are shared across requests.
    """
    global _generator
    if _generator is None:
        if cfg is None:
            from pathlib import Path
            import yaml
            with open("retrieval/configs/retrieval_config.yaml", "r") as f:
                cfg = yaml.safe_load(f)

        if cache is None:
            from cache.semantic_cache import build_cache_from_config
            cache = build_cache_from_config(cfg)

        if llm_client is None:
            from generation.llm_client import LLMClient
            llm_client = LLMClient(cfg)

        _generator = Generator(cfg=cfg, cache=cache, llm_client=llm_client)

    return _generator