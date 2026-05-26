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
        self.extractive_gate_floor: float = reranker_cfg.get("extractive_gate_floor", 0.22)
        adaptive_cfg = reranker_cfg.get("adaptive_gate", {})
        self.adaptive_gate_enabled: bool = adaptive_cfg.get("enabled", True)
        self.gate_min_threshold: float = adaptive_cfg.get("min_threshold", 0.45)
        self.gate_max_threshold: float = adaptive_cfg.get("max_threshold", 0.62)
        self.gate_strong_gap: float = adaptive_cfg.get("strong_gap", 0.08)
        self.gate_weak_gap: float = adaptive_cfg.get("weak_gap", 0.02)
        self.gate_strong_gap_bonus: float = adaptive_cfg.get("strong_gap_bonus", 0.05)
        self.gate_weak_gap_penalty: float = adaptive_cfg.get("weak_gap_penalty", 0.03)
        self.gate_short_query_bonus: float = adaptive_cfg.get("short_query_bonus", 0.02)
        self.gate_short_query_max_tokens: int = adaptive_cfg.get("short_query_max_tokens", 3)

        log.info(
            f"Generator initialized: confidence_threshold={self.confidence_threshold} | "
            f"extractive_gate_floor={self.extractive_gate_floor} | "
            f"adaptive_gate_enabled={self.adaptive_gate_enabled} | "
            f"primary={cfg.get('generation', {}).get('primary_provider', 'groq')}"
        )

    def generate(
        self,
        query: str,
        query_embedding: np.ndarray,
        reranked_docs: list[dict],
        reranker_scores: list[float],
        language: str = "roman_urdu",
        threshold_override: Optional[float] = None,
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
        gate = self._evaluate_gate(
            query=query,
            reranked_docs=reranked_docs,
            reranker_scores=reranker_scores,
            threshold_override=threshold_override,
        )
        stage_ms["gate_ms"] = round((time.time() - t) * 1000, 2)

        if not gate.passed:
            extractive = self._answer_from_docs(reranked_docs, language, min_len=60)
            if (
                extractive
                and gate.top_score >= self.extractive_gate_floor
                and gate.reason.startswith("score_below_threshold")
            ):
                relaxed = GateResult(
                    passed=True,
                    top_score=gate.top_score,
                    reason=(
                        f"extractive_gate_floor ({gate.top_score:.4f} ≥ {self.extractive_gate_floor}, "
                        f"below llm_threshold {self.confidence_threshold})"
                    ),
                )
                log.info(
                    "Gate strict fail → extractive_floor path | score=%.4f | query='%s'",
                    gate.top_score,
                    query[:50],
                )
                return GenerationOutput(
                    answer=extractive,
                    gate=relaxed,
                    cache_level=None,
                    cache_hit=False,
                    prompt=None,
                    llm_response=None,
                    language=language,
                    retrieval_doc_ids=[d.get("doc_id", "") for d in reranked_docs[:5]],
                    total_ms=round((time.time() - t_total) * 1000, 2),
                    stage_ms=stage_ms,
                )
            fallback = build_fallback_message(language, query=query)
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

        if not llm_response.success or not llm_response.text.strip():
            extractive = self._answer_from_docs(reranked_docs, language, min_len=60)
            if extractive:
                log.warning(
                    "LLM failed (%s); serving top QA answer as extractive fallback | query='%s'",
                    llm_response.error or "empty text",
                    query[:50],
                )
                return GenerationOutput(
                    answer=extractive,
                    gate=gate,
                    cache_level=None,
                    cache_hit=False,
                    prompt=built_prompt,
                    llm_response=llm_response,
                    language=language,
                    retrieval_doc_ids=[d.get("doc_id", "") for d in reranked_docs[:5]],
                    total_ms=round((time.time() - t_total) * 1000, 2),
                    stage_ms=stage_ms,
                )
            fallback = build_fallback_message(language, query=query)
            log.warning(
                f"LLM failed: {llm_response.error} | "
                f"returning generic fallback for query='{query[:50]}'"
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

    # Pakistani finance vocabulary — Roman Urdu + English + Urdu script terms
    _FINANCE_TERMS = frozenset([
        # Islamic finance
        "zakat", "zakaat", "zakkat", "riba", "halal", "haram", "murabaha",
        "ijarah", "musharakah", "mudarabah", "sukuk", "nisab", "takaful",
        "waqf", "hawl", "ushr", "fidya",
        # Banking
        "bank", "account", "savings", "deposit", "loan", "credit", "debit",
        "atm", "iban", "swift", "cheque", "overdraft", "mortgage", "finance",
        "interest", "profit", "markup", "installment", "qist", "emi",
        "hbl", "ubl", "meezan", "nbp", "mcb", "faysal", "askari", "js",
        "habib", "alfalah", "standard", "chartered", "scb",
        # Digital finance
        "easypaisa", "jazzcash", "raast", "nift", "1link", "paypak",
        "mobile", "wallet", "transfer", "transaction", "payment",
        # Investment
        "invest", "mutual", "fund", "stock", "shares", "nsc", "prize",
        "bond", "treasury", "tbill", "psx", "dividend", "portfolio",
        "naya", "roshan", "digital",
        # Tax / regulatory
        "tax", "fbr", "ntn", "iris", "withholding", "income", "return",
        "sbp", "secp", "pmex",
        # General finance
        "paise", "paisa", "rupee", "rupees", "pkr", "money", "cash",
        "salary", "income", "expense", "budget", "saving", "kharch",
        "kamai", "tankhwa", "rozgaar", "insurance", "bima",
        # Urdu script fragments
        "بینک", "قرض", "زکوٰۃ", "سود", "سرمایہ", "پیسہ",
    ])

    @classmethod
    def _has_finance_terms(cls, query: str) -> bool:
        """Return True if query contains at least one Pakistani finance keyword."""
        q_lower = query.lower()
        return any(term in q_lower for term in cls._FINANCE_TERMS)

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

    @staticmethod
    def _urdu_script_ratio(text: str) -> float:
        non_ws = [c for c in text if not c.isspace()]
        if not non_ws:
            return 0.0
        urdu = sum(
            1
            for c in non_ws
            if ("\u0600" <= c <= "\u06ff") or ("\ufb50" <= c <= "\ufdff")
        )
        return urdu / len(non_ws)

    @classmethod
    def _pick_answer_for_language(
        cls,
        docs: list[dict],
        language: str,
        min_len: int = 60,
    ) -> str:
        """
        Prefer an extractive snippet whose script matches the query language.
        If none match heuristics, pick the best-effort candidate (most/least Urdu)
        while preserving rerank order for ties.
        """
        lang = language if language in ("urdu", "english", "roman_urdu") else "roman_urdu"
        candidates: list[tuple[float, str]] = []
        for d in docs:
            a = (d.get("answer") or d.get("retrieval_text") or "").strip()
            if len(a) < min_len:
                continue
            candidates.append((cls._urdu_script_ratio(a), a))
        if not candidates:
            return ""

        if lang == "urdu":
            for ur, txt in candidates:
                if ur >= 0.08:
                    return txt
            return max(candidates, key=lambda x: x[0])[1]

        if lang == "english":
            for ur, txt in candidates:
                if ur < 0.06:
                    return txt
            return min(candidates, key=lambda x: x[0])[1]

        for ur, txt in candidates:
            if ur < 0.25:
                return txt
        return min(candidates, key=lambda x: x[0])[1]

    def _answer_from_docs(self, docs: list[dict], language: str, min_len: int = 60) -> str:
        picked = self._pick_answer_for_language(docs, language, min_len=min_len)
        if picked:
            return picked
        return self._first_long_answer(docs, min_len=min_len)

    @staticmethod
    def _first_long_answer(docs: list[dict], min_len: int = 60) -> str:
        for d in docs:
            a = (d.get("answer") or d.get("retrieval_text") or "").strip()
            if len(a) >= min_len:
                return a
        return ""

    def _compute_effective_threshold(
        self,
        query: str,
        reranker_scores: list[float],
    ) -> tuple[float, str]:
        """
        Compute an adaptive confidence threshold for this query.
        Keeps the score gate bounded to avoid over-relaxing.
        """
        base = self.confidence_threshold
        if not self.adaptive_gate_enabled:
            return base, f"static threshold={base:.4f}"

        adjustment = 0.0
        reasons: list[str] = []

        if len(reranker_scores) >= 2:
            gap = float(reranker_scores[0]) - float(reranker_scores[1])
            if gap >= self.gate_strong_gap:
                adjustment -= self.gate_strong_gap_bonus
                reasons.append(f"strong_gap={gap:.4f}")
            elif gap <= self.gate_weak_gap:
                adjustment += self.gate_weak_gap_penalty
                reasons.append(f"weak_gap={gap:.4f}")

        token_count = len(query.split())
        if token_count <= self.gate_short_query_max_tokens:
            adjustment -= self.gate_short_query_bonus
            reasons.append(f"short_query_tokens={token_count}")

        effective = max(self.gate_min_threshold, min(self.gate_max_threshold, base + adjustment))
        if not reasons:
            reasons.append("no_adjustment")
        return effective, ", ".join(reasons)

    def _evaluate_gate(
        self,
        query: str,
        reranked_docs: list[dict],
        reranker_scores: list[float],
        threshold_override: Optional[float] = None,
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

        # Hard early-exit: if the top score is essentially zero, retrieval truly failed.
        # Don't bother running adaptive threshold logic — block immediately.
        HARD_BLOCK_FLOOR = 0.10
        if top_score < HARD_BLOCK_FLOOR:
            return GateResult(
                passed=False,
                top_score=top_score,
                reason=f"hard_block_low_score ({top_score:.4f} < {HARD_BLOCK_FLOOR})",
            )

        # Out-of-scope domain check: if query has no finance terms AND score is
        # below OOS_THRESHOLD, reject as out-of-domain.
        OOS_THRESHOLD = 0.50
        if top_score < OOS_THRESHOLD and not self._has_finance_terms(query):
            return GateResult(
                passed=False,
                top_score=top_score,
                reason=f"out_of_scope (score={top_score:.4f} < {OOS_THRESHOLD}, no finance terms detected)",
            )

        if threshold_override is not None:
            effective_threshold = float(threshold_override)
            threshold_reason = f"override (pdf_retry_threshold={effective_threshold:.4f})"
        else:
            effective_threshold, threshold_reason = self._compute_effective_threshold(
                query=query,
                reranker_scores=reranker_scores,
            )

        if top_score < effective_threshold:
            return GateResult(
                passed=False,
                top_score=top_score,
                reason=(
                    f"score_below_threshold ({top_score:.4f} < {effective_threshold:.4f}) | "
                    f"base={self.confidence_threshold:.4f} | {threshold_reason}"
                ),
            )

        return GateResult(
            passed=True,
            top_score=top_score,
            reason=(
                f"passed ({top_score:.4f} >= {effective_threshold:.4f}) | "
                f"base={self.confidence_threshold:.4f} | {threshold_reason}"
            ),
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