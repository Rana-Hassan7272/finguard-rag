"""
metadata/filter_policy.py
==========================
Policy layer: decides whether to apply category pre-filter to retrieval.

Separates the DETECTION concern (category_detector.py) from the
APPLICATION concern (this file). This layered design lets you change
filter aggressiveness without touching the detector logic.

Policy rules (evaluated in order):
  1. If metadata_filter.enabled=false in config → never filter
  2. If detected category is None → no filter (no confident match)
  3. If confidence < threshold → no filter (low confidence → risk of wrong filter)
  4. If strict_mode=true → also check that category exists in corpus → filter
  5. Otherwise → apply filter

Usage:
    from metadata.filter_policy import FilterPolicy
    from metadata.category_detector import CategoryDetector

    detector = CategoryDetector(cfg)
    policy   = FilterPolicy(cfg)

    detection = detector.detect("zakat ka hisab kaise karein")
    decision  = policy.decide(detection)

    print(decision.apply_filter)       # True
    print(decision.category_filter)    # "islamic_finance"
    print(decision.reason)             # "confidence=0.85 >= threshold=0.65"
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from metadata.category_detector import CategoryDetectionResult

log = logging.getLogger(__name__)

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")


# ---------------------------------------------------------------------------
# Filter decision
# ---------------------------------------------------------------------------

@dataclass
class FilterDecision:
    """
    Output of the filter policy layer.

    Fields
    ------
    apply_filter     : True = pass category_filter to retrievers
    category_filter  : category label string to filter on (None if not filtering)
    reason           : human-readable explanation of the decision
    confidence       : detector confidence that drove this decision
    detection_result : full CategoryDetectionResult for diagnostics
    """
    apply_filter: bool
    category_filter: Optional[str]
    reason: str
    confidence: float
    detection_result: CategoryDetectionResult


# ---------------------------------------------------------------------------
# FilterPolicy
# ---------------------------------------------------------------------------

class FilterPolicy:
    """
    Decides when and how to apply metadata pre-filtering.

    Configuration keys (all under metadata_filter in retrieval_config.yaml):
      enabled              : bool   — master switch
      confidence_threshold : float  — minimum detector confidence to filter
      strict_mode          : bool   — also verify category has docs in corpus
      fallback_behavior    : str    — "search_all" (only supported value)
      known_categories     : list   — optional explicit list of valid categories
                                      used in strict_mode to validate predicted label
    """

    def __init__(self, cfg: dict, known_categories: Optional[list[str]] = None):
        self.cfg = cfg
        mf_cfg = cfg.get("metadata_filter", {})

        self.enabled: bool = mf_cfg.get("enabled", True)
        self.confidence_threshold: float = mf_cfg.get("confidence_threshold", 0.65)
        self.strict_mode: bool = mf_cfg.get("strict_mode", False)
        self.fallback_behavior: str = mf_cfg.get("fallback_behavior", "search_all")

        # Optional explicit set of valid categories
        # In strict_mode, predictions outside this set are rejected
        self.known_categories: Optional[set[str]] = (
            set(known_categories) if known_categories else None
        )

        # Stats tracking for observability
        self._stats = {
            "total_decisions": 0,
            "filter_applied": 0,
            "filter_skipped_disabled": 0,
            "filter_skipped_no_detection": 0,
            "filter_skipped_low_confidence": 0,
            "filter_skipped_unknown_category": 0,
        }

        log.info(
            f"FilterPolicy initialized: enabled={self.enabled} | "
            f"threshold={self.confidence_threshold} | "
            f"strict_mode={self.strict_mode}"
        )

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def decide(self, detection: CategoryDetectionResult) -> FilterDecision:
        """
        Evaluate filter policy given a detection result.

        Parameters
        ----------
        detection : CategoryDetectionResult from CategoryDetector.detect()

        Returns
        -------
        FilterDecision with apply_filter, category_filter, reason, confidence
        """
        self._stats["total_decisions"] += 1

        # Rule 1: Master switch
        if not self.enabled:
            self._stats["filter_skipped_disabled"] += 1
            return FilterDecision(
                apply_filter=False,
                category_filter=None,
                reason="metadata_filter.enabled=false in config",
                confidence=detection.confidence,
                detection_result=detection,
            )

        # Rule 2: No category detected
        if detection.predicted_category is None:
            self._stats["filter_skipped_no_detection"] += 1
            return FilterDecision(
                apply_filter=False,
                category_filter=None,
                reason="no category detected (below detection threshold)",
                confidence=0.0,
                detection_result=detection,
            )

        # Rule 3: Confidence too low
        if detection.confidence < self.confidence_threshold:
            self._stats["filter_skipped_low_confidence"] += 1
            return FilterDecision(
                apply_filter=False,
                category_filter=None,
                reason=(
                    f"confidence={detection.confidence:.4f} < "
                    f"threshold={self.confidence_threshold} — "
                    f"too uncertain to pre-filter"
                ),
                confidence=detection.confidence,
                detection_result=detection,
            )

        # Rule 4: Strict mode — verify category is in known_categories
        predicted = detection.predicted_category
        if self.strict_mode and self.known_categories is not None:
            if predicted not in self.known_categories:
                self._stats["filter_skipped_unknown_category"] += 1
                return FilterDecision(
                    apply_filter=False,
                    category_filter=None,
                    reason=(
                        f"strict_mode: predicted category '{predicted}' not in "
                        f"known_categories {sorted(self.known_categories)}"
                    ),
                    confidence=detection.confidence,
                    detection_result=detection,
                )

        # Rule 5: Apply filter
        self._stats["filter_applied"] += 1
        return FilterDecision(
            apply_filter=True,
            category_filter=predicted,
            reason=(
                f"confidence={detection.confidence:.4f} >= "
                f"threshold={self.confidence_threshold} | "
                f"matched_keywords={detection.matched_keywords}"
            ),
            confidence=detection.confidence,
            detection_result=detection,
        )

    # ------------------------------------------------------------------
    # Session stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return aggregated decision stats for this session.
        Useful for logging and monitoring filter effectiveness.
        """
        total = self._stats["total_decisions"]
        applied = self._stats["filter_applied"]
        return {
            **self._stats,
            "filter_rate": round(applied / total, 4) if total > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0

    # ------------------------------------------------------------------
    # Batch evaluation helper
    # ------------------------------------------------------------------

    def decide_batch(
        self, detections: list[CategoryDetectionResult]
    ) -> list[FilterDecision]:
        """Evaluate policy for a list of detections."""
        return [self.decide(d) for d in detections]

    # ------------------------------------------------------------------
    # Runtime category registration (for strict_mode)
    # ------------------------------------------------------------------

    def register_known_categories(self, categories: list[str]) -> None:
        """
        Register the categories that actually exist in the corpus.
        Call this after loading the index/docstore.

        In strict_mode, any prediction outside this set is rejected.
        This prevents filtering on a category that has 0 docs — which
        would return empty results.
        """
        self.known_categories = set(categories)
        log.info(f"FilterPolicy: registered {len(categories)} known categories: {sorted(categories)}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_policy_cache: dict[int, FilterPolicy] = {}


def get_policy(
    cfg: Optional[dict] = None,
    known_categories: Optional[list[str]] = None,
) -> FilterPolicy:
    """Return module-level singleton FilterPolicy."""
    if cfg is None:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
    cfg_id = id(cfg)
    if cfg_id not in _policy_cache:
        _policy_cache[cfg_id] = FilterPolicy(cfg, known_categories)
    return _policy_cache[cfg_id]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from metadata.category_detector import CategoryDetector
    from typing import Optional

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    detector = CategoryDetector(cfg)
    policy = FilterPolicy(cfg)

    known_cats = [
        "islamic_finance", "banking_basics", "loans",
        "digital_payments", "investment", "insurance", "tax", "remittance"
    ]
    policy.register_known_categories(known_cats)

    test_queries = [
        "zakat ka hisab kaise karein",               # high confidence → filter
        "meezan bank account kholna hai",             # high confidence → filter
        "easypaisa se paise bhejein",                 # high confidence → filter
        "what is account",                            # ambiguous → maybe no filter
        "aaj ka mosam kaisa hai",                     # off-domain → no filter
        "",                                           # empty → no filter
        "riba aur interest mein kya farq hai",        # high confidence → filter
    ]

    print(f"\n{'Query':<50} {'Apply':<6} {'Category':<20} {'Conf':<6} Reason")
    print("-" * 130)
    for query in test_queries:
        detection = detector.detect(query)
        decision = policy.decide(detection)
        print(
            f"{query[:49]:<50} {str(decision.apply_filter):<6} "
            f"{str(decision.category_filter):<20} "
            f"{decision.confidence:<6.3f} "
            f"{decision.reason}"
        )

    print(f"\nSession stats: {policy.get_stats()}")