"""
cache/cache_stats.py

Cache statistics tracker that wraps SemanticCache and feeds the
observability pipeline. Tracks hit rates, latency savings, eviction
counts, and exposes a summary dict for the JSONL logger.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

log = logging.getLogger(__name__)

ROLLING_WINDOW = 500


@dataclass
class CacheEventRecord:
    timestamp: float
    level: Optional[int]
    hit: bool
    similarity: float
    query_text: str
    latency_saved_ms: Optional[float]


class CacheStatsTracker:
    """
    Records every cache lookup event and provides aggregated stats.

    Maintains a rolling window of the last N events for recent metrics
    alongside all-time counters. Feed the output of snapshot() directly
    into the observability logger.

    Estimated latency savings:
        L1 hit saves full pipeline: ~800ms
        L2 hit saves retrieval:     ~600ms
        Miss: 0ms saved
    """

    L1_LATENCY_SAVED_MS = 800.0
    L2_LATENCY_SAVED_MS = 600.0

    def __init__(self, rolling_window: int = ROLLING_WINDOW):
        self._window = deque(maxlen=rolling_window)
        self._lock = Lock()

        self._all_time = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "total_latency_saved_ms": 0.0,
            "evictions_ttl": 0,
            "evictions_version": 0,
            "evictions_capacity": 0,
        }

    def record(
        self,
        hit: bool,
        level: Optional[int],
        similarity: float,
        query_text: str = "",
    ) -> None:
        with self._lock:
            latency_saved = None
            if hit and level == 1:
                latency_saved = self.L1_LATENCY_SAVED_MS
                self._all_time["l1_hits"] += 1
                self._all_time["total_latency_saved_ms"] += self.L1_LATENCY_SAVED_MS
            elif hit and level == 2:
                latency_saved = self.L2_LATENCY_SAVED_MS
                self._all_time["l2_hits"] += 1
                self._all_time["total_latency_saved_ms"] += self.L2_LATENCY_SAVED_MS
            else:
                self._all_time["misses"] += 1
            self._window.append(
                CacheEventRecord(
                    timestamp=time.time(),
                    level=level,
                    hit=hit,
                    similarity=round(similarity, 4),
                    query_text=query_text[:80],
                    latency_saved_ms=latency_saved,
                )
            )

    def record_eviction(self, reason: str, count: int = 1) -> None:
        key = f"evictions_{reason}"
        if key in self._all_time:
            self._all_time[key] += count

    def snapshot(self) -> dict:
        with self._lock:
            window = list(self._window)

        total_all = self._all_time["l1_hits"] + self._all_time["l2_hits"] + self._all_time["misses"]
        total_saved_s = self._all_time["total_latency_saved_ms"] / 1000.0

        recent_hits = sum(1 for e in window if e.hit)
        recent_total = len(window)
        recent_l1 = sum(1 for e in window if e.hit and e.level == 1)
        recent_l2 = sum(1 for e in window if e.hit and e.level == 2)
        recent_saved = sum(e.latency_saved_ms or 0 for e in window)

        avg_sim_hits = (
            sum(e.similarity for e in window if e.hit) / max(recent_hits, 1)
        )
        avg_sim_misses = (
            sum(e.similarity for e in window if not e.hit)
            / max(recent_total - recent_hits, 1)
        )

        return {
            "all_time": {
                "l1_hits": self._all_time["l1_hits"],
                "l2_hits": self._all_time["l2_hits"],
                "misses": self._all_time["misses"],
                "total_lookups": total_all,
                "l1_hit_rate": round(self._all_time["l1_hits"] / max(total_all, 1), 4),
                "l2_hit_rate": round(self._all_time["l2_hits"] / max(total_all, 1), 4),
                "combined_hit_rate": round(
                    (self._all_time["l1_hits"] + self._all_time["l2_hits"])
                    / max(total_all, 1),
                    4,
                ),
                "total_latency_saved_ms": round(self._all_time["total_latency_saved_ms"], 1),
                "total_latency_saved_hours": round(total_saved_s / 3600, 4),
                "evictions_ttl": self._all_time["evictions_ttl"],
                "evictions_version": self._all_time["evictions_version"],
                "evictions_capacity": self._all_time["evictions_capacity"],
            },
            "recent": {
                "window_size": recent_total,
                "l1_hits": recent_l1,
                "l2_hits": recent_l2,
                "misses": recent_total - recent_hits,
                "hit_rate": round(recent_hits / max(recent_total, 1), 4),
                "latency_saved_ms": round(recent_saved, 1),
                "avg_similarity_on_hits": round(avg_sim_hits, 4),
                "avg_similarity_on_misses": round(avg_sim_misses, 4),
            },
        }

    def log_summary(self) -> None:
        s = self.snapshot()
        at = s["all_time"]
        rc = s["recent"]
        log.info(
            f"Cache stats — all-time: hits={at['l1_hits']}L1/{at['l2_hits']}L2 "
            f"miss={at['misses']} rate={at['combined_hit_rate']:.1%} "
            f"saved={at['total_latency_saved_ms']/1000:.1f}s | "
            f"recent({rc['window_size']}): rate={rc['hit_rate']:.1%}"
        )

    def reset(self) -> None:
        with self._lock:
            self._window.clear()
        for k in self._all_time:
            self._all_time[k] = 0


_tracker: Optional[CacheStatsTracker] = None


def get_tracker() -> CacheStatsTracker:
    global _tracker
    if _tracker is None:
        _tracker = CacheStatsTracker()
    return _tracker