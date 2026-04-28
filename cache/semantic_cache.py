"""
cache/semantic_cache.py

Two-level in-memory semantic cache for the FinGuard RAG pipeline.

Level 1 — Answer cache  (threshold 0.95):
    Query embedding matches a cached query closely enough that we can
    return the previously generated answer directly. No retrieval, no LLM.
    Latency: ~5ms total.

Level 2 — Retrieval cache (threshold 0.90):
    Query embedding is similar enough that the same set of documents
    would be retrieved. Skip retrieval and go straight to reranker.
    Latency: ~130ms total (reranker + LLM only).

Cache invalidation:
    Every entry stores the corpus_version hash (from corpus manifest).
    When the corpus is rebuilt, the version hash changes and all stale
    entries are silently skipped — no manual clearing required.

TTL:
    Entries older than ttl_hours are treated as misses and evicted lazily
    on the next access attempt.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query_embedding: np.ndarray
    query_text: str
    answer: Optional[str]
    doc_ids: list[str]
    corpus_version: str
    timestamp: float
    ttl_seconds: float
    hit_count: int = 0
    last_hit_ts: float = 0.0

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds

    def is_valid_for_corpus(self, corpus_version: str) -> bool:
        return self.corpus_version == corpus_version

    def touch(self) -> None:
        self.hit_count += 1
        self.last_hit_ts = time.time()


@dataclass
class CacheLookupResult:
    hit: bool
    level: Optional[int]
    answer: Optional[str]
    doc_ids: list[str]
    similarity: float
    entry_age_seconds: float
    reason: str


class SemanticCache:
    """
    Two-level in-memory semantic cache.

    Thread-safe via a single RW lock. For read-heavy workloads at high QPS
    you can split to per-shard locks, but for hundreds of QPS this is fine.

    Parameters
    ----------
    l1_threshold    : cosine sim threshold for answer cache hit (default 0.95)
    l2_threshold    : cosine sim threshold for retrieval cache hit (default 0.90)
    ttl_seconds     : entry lifetime in seconds (default 48h = 172800)
    max_entries     : evict oldest entries when cache grows beyond this size
    corpus_version  : current corpus version hash — entries from other versions are invalid
    """

    def __init__(
        self,
        l1_threshold: float = 0.95,
        l2_threshold: float = 0.90,
        ttl_seconds: float = 172800,
        max_entries: int = 2000,
        corpus_version: str = "default",
    ):
        self.l1_threshold = l1_threshold
        self.l2_threshold = l2_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.corpus_version = corpus_version

        self._entries: list[CacheEntry] = []
        self._lock = Lock()

        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions_ttl": 0,
            "evictions_version": 0,
            "evictions_capacity": 0,
        }

        log.info(
            f"SemanticCache initialized: l1={l1_threshold} | l2={l2_threshold} | "
            f"ttl={ttl_seconds}s | max_entries={max_entries} | "
            f"corpus_version={corpus_version}"
        )

    def lookup(self, query_embedding: np.ndarray) -> CacheLookupResult:
        """
        Search cache for a semantically similar query.

        Returns a CacheLookupResult. If hit=True and level=1, answer is populated.
        If hit=True and level=2, doc_ids is populated (skip retrieval, go to reranker).
        If hit=False, run the full pipeline.
        """
        with self._lock:
            if not self._entries:
                self._stats["misses"] += 1
                return CacheLookupResult(False, None, None, [], 0.0, 0.0, "cache_empty")

            valid_entries = [
                e for e in self._entries
                if not e.is_expired() and e.is_valid_for_corpus(self.corpus_version)
            ]

            if not valid_entries:
                self._stats["misses"] += 1
                return CacheLookupResult(False, None, None, [], 0.0, 0.0, "no_valid_entries")

            query_embedding = np.asarray(query_embedding).reshape(-1)
            # Batch cosine similarity: (n_entries, dim) @ (dim,) → (n_entries,)
            cached_embs = np.stack([e.query_embedding for e in valid_entries])
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            c_norms = cached_embs / (np.linalg.norm(cached_embs, axis=1, keepdims=True) + 1e-9)
            similarities = c_norms @ q_norm

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])
            best_entry = valid_entries[best_idx]
            age = time.time() - best_entry.timestamp

            if best_sim >= self.l1_threshold and best_entry.answer is not None:
                best_entry.touch()
                self._stats["l1_hits"] += 1
                return CacheLookupResult(
                    hit=True,
                    level=1,
                    answer=best_entry.answer,
                    doc_ids=best_entry.doc_ids,
                    similarity=round(best_sim, 4),
                    entry_age_seconds=round(age, 1),
                    reason=f"l1_hit sim={best_sim:.4f}",
                )

            if best_sim >= self.l2_threshold and best_entry.doc_ids:
                best_entry.touch()
                self._stats["l2_hits"] += 1
                return CacheLookupResult(
                    hit=True,
                    level=2,
                    answer=None,
                    doc_ids=best_entry.doc_ids,
                    similarity=round(best_sim, 4),
                    entry_age_seconds=round(age, 1),
                    reason=f"l2_hit sim={best_sim:.4f}",
                )

            self._stats["misses"] += 1
            return CacheLookupResult(
                hit=False,
                level=None,
                answer=None,
                doc_ids=[],
                similarity=round(best_sim, 4),
                entry_age_seconds=0.0,
                reason=f"miss best_sim={best_sim:.4f}",
            )

    def store(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        doc_ids: list[str],
        answer: Optional[str] = None,
    ) -> None:
        """
        Store a pipeline result in the cache.

        Both answer and doc_ids can be set independently.
        A retrieval-only result (no LLM answer yet) sets answer=None.
        When the LLM answer arrives, call store() again with the same
        embedding — it will update the existing entry if found.
        """
        with self._lock:
            query_embedding = np.asarray(query_embedding).reshape(-1)
            emb_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

            # Check if an entry already exists for this query (update it)
            if self._entries:
                cached_embs = np.stack([e.query_embedding for e in self._entries])
                c_norms = cached_embs / (
                    np.linalg.norm(cached_embs, axis=1, keepdims=True) + 1e-9
                )
                sims = c_norms @ emb_normalized
                best_idx = int(np.argmax(sims))
                if float(sims[best_idx]) >= 0.99:
                    existing = self._entries[best_idx]
                    if doc_ids:
                        existing.doc_ids = doc_ids
                    if answer is not None:
                        existing.answer = answer
                    existing.corpus_version = self.corpus_version
                    existing.timestamp = time.time()
                    return

            # Evict if at capacity — remove oldest entries first
            if len(self._entries) >= self.max_entries:
                self._evict_oldest(count=max(1, self.max_entries // 10))

            entry = CacheEntry(
                query_embedding=emb_normalized.copy(),
                query_text=query_text,
                answer=answer,
                doc_ids=doc_ids,
                corpus_version=self.corpus_version,
                timestamp=time.time(),
                ttl_seconds=self.ttl_seconds,
            )
            self._entries.append(entry)
            self._stats["stores"] += 1

    def _evict_oldest(self, count: int) -> None:
        self._entries.sort(key=lambda e: e.timestamp)
        self._entries = self._entries[count:]
        self._stats["evictions_capacity"] += count

    def evict_expired(self) -> int:
        with self._lock:
            before = len(self._entries)
            self._entries = [
                e for e in self._entries
                if not e.is_expired() and e.is_valid_for_corpus(self.corpus_version)
            ]
            removed = before - len(self._entries)
            if removed:
                self._stats["evictions_ttl"] += removed
                log.debug(f"Evicted {removed} expired/stale cache entries")
            return removed

    def invalidate_corpus(self, new_corpus_version: str) -> int:
        """
        Update corpus version — all entries from old version become invalid.
        Call this after rebuilding the corpus/indexes.
        """
        with self._lock:
            old_version = self.corpus_version
            self.corpus_version = new_corpus_version
            stale = sum(1 for e in self._entries if e.corpus_version != new_corpus_version)
            self._stats["evictions_version"] += stale
            log.info(
                f"Corpus version updated: {old_version} → {new_corpus_version} | "
                f"{stale} stale entries will be skipped on next lookup"
            )
            return stale

    def clear(self) -> None:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            log.info(f"Cache cleared: removed {count} entries")

    def stats(self) -> dict:
        with self._lock:
            total = self._stats["l1_hits"] + self._stats["l2_hits"] + self._stats["misses"]
            return {
                **self._stats,
                "total_lookups": total,
                "l1_hit_rate": round(self._stats["l1_hits"] / max(total, 1), 4),
                "l2_hit_rate": round(self._stats["l2_hits"] / max(total, 1), 4),
                "combined_hit_rate": round(
                    (self._stats["l1_hits"] + self._stats["l2_hits"]) / max(total, 1), 4
                ),
                "current_size": len(self._entries),
                "corpus_version": self.corpus_version,
            }

    def __len__(self) -> int:
        return len(self._entries)


def compute_corpus_version(artifacts_root: str = "retrieval/artifacts") -> str:
    """
    Compute a short hash representing the current corpus state.
    Reads qa corpus manifest + pdf corpus stats if present.
    Use this to initialize SemanticCache.corpus_version.
    """
    root = Path(artifacts_root)
    components = []

    for fname in ["corpus_stats.json", "pdf_corpus_stats.json"]:
        p = root / fname
        if p.exists():
            components.append(p.read_text(encoding="utf-8"))

    for pattern in ["index_manifest_*.json"]:
        for p in sorted(root.glob(pattern)):
            components.append(p.read_text(encoding="utf-8"))

    if not components:
        log.warning("No corpus manifests found — using 'default' corpus version")
        return "default"

    combined = "|".join(components)
    version = hashlib.sha256(combined.encode()).hexdigest()[:8]
    log.info(f"Computed corpus_version: {version}")
    return version


def build_cache_from_config(cfg: dict) -> SemanticCache:
    """
    Construct a SemanticCache from retrieval_config.yaml settings.
    The plan specifies two thresholds; existing config has one — we derive both.
    """
    cache_cfg = cfg.get("cache", {})
    base_threshold = cache_cfg.get("similarity_threshold", 0.92)

    l1_threshold = cache_cfg.get("l1_threshold", max(base_threshold, 0.95))
    l2_threshold = cache_cfg.get("l2_threshold", min(base_threshold, 0.90))
    ttl_seconds = cache_cfg.get("ttl_seconds", 172800)
    max_entries = cache_cfg.get("max_entries", 2000)

    artifacts_root = cfg.get("artifacts", {}).get("root", "retrieval/artifacts")
    corpus_version = compute_corpus_version(artifacts_root)

    return SemanticCache(
        l1_threshold=l1_threshold,
        l2_threshold=l2_threshold,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        corpus_version=corpus_version,
    )