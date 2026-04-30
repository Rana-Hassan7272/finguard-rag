"""
observability/logger.py

Writes one structured JSONL log entry per pipeline request.
Thread-safe, handles all I/O errors gracefully so a logging failure
never crashes the serving path. Supports log rotation by date.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "observability/logs"
DEFAULT_LOG_FILE = "requests.jsonl"


@dataclass
class RequestLogEntry:
    timestamp: str
    query: str
    query_language: str
    query_intent: str
    category_detected: Optional[str]
    category_confidence: float
    filter_applied: bool
    cache_level: str
    corpus_version: str
    retrieval_ms: float
    reranker_ms: float
    llm_ms: float
    total_ms: float
    top_reranker_score: float
    gate_passed: bool
    gate_reason: str
    source_mix: dict
    retrieved_doc_ids: list[str]
    answer_generated: bool
    query_normalized: Optional[str] = None
    language_confidence: Optional[float] = None
    filter_reason: Optional[str] = None
    cache_similarity: Optional[float] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_tokens: Optional[int] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


class RequestLogger:
    """
    JSONL request logger. One line per request, atomically written.

    Features:
    - Thread-safe via per-file lock
    - Graceful: I/O errors are logged to stderr, never raised
    - Optional daily log rotation: requests_2025-01-15.jsonl
    - Flush after every write so tail -f works in real time
    - Configurable log directory and rotation policy
    """

    def __init__(
        self,
        log_dir: str = DEFAULT_LOG_DIR,
        rotate_daily: bool = True,
        enabled: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.rotate_daily = rotate_daily
        self.enabled = enabled
        self._lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._current_date: Optional[str] = None

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"RequestLogger initialized: dir={self.log_dir} rotate_daily={rotate_daily}")

    def _get_log_path(self) -> Path:
        if self.rotate_daily:
            date_str = time.strftime("%Y-%m-%d", time.gmtime())
            return self.log_dir / f"requests_{date_str}.jsonl"
        return self.log_dir / DEFAULT_LOG_FILE

    def _ensure_file_open(self) -> bool:
        target = self._get_log_path()
        date_str = time.strftime("%Y-%m-%d", time.gmtime())

        if self._current_file != target or self._current_date != date_str:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
            try:
                self._file_handle = open(target, "a", encoding="utf-8", buffering=1)
                self._current_file = target
                self._current_date = date_str
            except OSError as exc:
                log.error(f"RequestLogger: cannot open log file {target}: {exc}")
                self._file_handle = None
                return False
        return self._file_handle is not None

    def write(self, entry: RequestLogEntry) -> bool:
        if not self.enabled:
            return True

        with self._lock:
            if not self._ensure_file_open():
                return False
            try:
                record = {k: v for k, v in asdict(entry).items() if v is not None and v != {}}
                line = json.dumps(record, ensure_ascii=False)
                self._file_handle.write(line + "\n")
                self._file_handle.flush()
                return True
            except (OSError, TypeError, ValueError) as exc:
                log.error(f"RequestLogger: write failed: {exc}")
                return False

    def write_dict(self, d: dict) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            if not self._ensure_file_open():
                return False
            try:
                line = json.dumps(d, ensure_ascii=False, default=str)
                self._file_handle.write(line + "\n")
                self._file_handle.flush()
                return True
            except (OSError, TypeError, ValueError) as exc:
                log.error(f"RequestLogger: write_dict failed: {exc}")
                return False

    def close(self) -> None:
        with self._lock:
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None


def build_log_entry(
    query: str,
    total_ms: float,
    query_language: str = "roman_urdu",
    query_intent: str = "practical",
    query_normalized: Optional[str] = None,
    language_confidence: Optional[float] = None,
    category_detected: Optional[str] = None,
    category_confidence: float = 0.0,
    filter_applied: bool = False,
    filter_reason: Optional[str] = None,
    cache_level: Optional[int] = None,
    cache_similarity: Optional[float] = None,
    corpus_version: str = "default",
    retrieval_ms: float = 0.0,
    reranker_ms: float = 0.0,
    llm_ms: float = 0.0,
    top_reranker_score: float = 0.0,
    gate_passed: bool = False,
    gate_reason: str = "",
    retrieved_doc_ids: Optional[list[str]] = None,
    answer_generated: bool = False,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_tokens: Optional[int] = None,
    error: Optional[str] = None,
    source_mix: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> RequestLogEntry:
    doc_ids = retrieved_doc_ids or []

    # Prefer explicit source mix if caller already computed it (pipeline diagnostics).
    if source_mix:
        # Support both schemas:
        # - {"qa": 3, "pdf": 7}
        # - {"qa_docs": 3, "pdf_docs": 7}
        source_mix_norm = {
            "qa": int(source_mix.get("qa", source_mix.get("qa_docs", 0))),
            "pdf": int(source_mix.get("pdf", source_mix.get("pdf_docs", 0))),
        }
    else:
        source_mix_norm: dict[str, int] = {"qa": 0, "pdf": 0}
        for doc_id in doc_ids:
            if doc_id.startswith("pdf_"):
                source_mix_norm["pdf"] += 1
            else:
                source_mix_norm["qa"] += 1

    cache_level_str = "miss"
    if cache_level == 1:
        cache_level_str = "l1_hit"
    elif cache_level == 2:
        cache_level_str = "l2_hit"

    return RequestLogEntry(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        query=query[:500],
        query_normalized=query_normalized,
        query_language=query_language,
        language_confidence=language_confidence,
        query_intent=query_intent,
        category_detected=category_detected,
        category_confidence=round(category_confidence, 4),
        filter_applied=filter_applied,
        filter_reason=filter_reason,
        cache_level=cache_level_str,
        cache_similarity=cache_similarity,
        corpus_version=corpus_version,
        retrieval_ms=round(retrieval_ms, 2),
        reranker_ms=round(reranker_ms, 2),
        llm_ms=round(llm_ms, 2),
        total_ms=round(total_ms, 2),
        top_reranker_score=round(top_reranker_score, 4),
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        source_mix=source_mix_norm,
        retrieved_doc_ids=doc_ids[:20],
        answer_generated=answer_generated,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_tokens=llm_tokens,
        error=error,
        extra=extra or {},
    )


_logger_instance: Optional[RequestLogger] = None


def get_logger(cfg: Optional[dict] = None) -> RequestLogger:
    global _logger_instance
    if _logger_instance is None:
        log_dir = DEFAULT_LOG_DIR
        rotate_daily = True
        enabled = True
        if cfg:
            obs_cfg = cfg.get("observability", {})
            log_dir = obs_cfg.get("log_dir", DEFAULT_LOG_DIR)
            rotate_daily = obs_cfg.get("rotate_daily", True)
            enabled = obs_cfg.get("enabled", True)
        _logger_instance = RequestLogger(
            log_dir=log_dir,
            rotate_daily=rotate_daily,
            enabled=enabled,
        )
    return _logger_instance