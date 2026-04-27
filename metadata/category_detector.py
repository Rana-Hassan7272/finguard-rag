"""
metadata/category_detector.py
===============================
Pre-retrieval category detector for the FinGuard RAG pipeline.

Detects the likely category of a query BEFORE retrieval so we can
apply a metadata pre-filter to restrict the search space.

Strategy: Rule-based keyword mapping (no ML model required)
Rationale:
  - Fast: microseconds, not milliseconds
  - Controllable: you can inspect and edit the keyword map directly
  - No cold-start: works on day 1 without training data
  - Transparent: confidence is based on match count, not a black box

Confidence scoring:
  - Each keyword match contributes a weight (exact > partial > synonym)
  - Total score is normalized against a max-confidence threshold
  - If no category exceeds the minimum confidence → return None (no filter)

Usage:
    from metadata.category_detector import CategoryDetector

    detector = CategoryDetector(cfg)
    result = detector.detect("zakat ka hisab kaise karein")

    print(result.predicted_category)   # "islamic_finance"
    print(result.confidence)           # 0.85
    print(result.matched_keywords)     # ["zakat", "hisab"]
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class CategoryDetectionResult:
    """
    Result of category detection on a single query.

    Fields
    ------
    predicted_category : category label string, or None if below confidence threshold
    confidence         : float in [0.0, 1.0] — how confident we are in the label
    matched_keywords   : list of keywords that triggered the prediction
    all_scores         : dict of {category: score} for all categories (for debugging)
    method             : always "keyword_map" for this detector
    """
    predicted_category: Optional[str]
    confidence: float
    matched_keywords: list[str]
    all_scores: dict[str, float] = field(default_factory=dict)
    method: str = "keyword_map"


# ---------------------------------------------------------------------------
# Default keyword map
# ---------------------------------------------------------------------------
# Structure: category -> list of (keyword, weight) tuples
#
# Weight guide:
#   1.0 = highly domain-specific term, near-certain signal
#   0.7 = strong signal but could appear in adjacent categories
#   0.5 = moderate signal — useful if combined with other matches
#   0.3 = weak signal — only counts when other matches present
#
# Keywords should be lowercase and work on normalized Roman Urdu / English.
# Urdu script keywords are also supported — the detector lowercases Latin only.

DEFAULT_KEYWORD_MAP: dict[str, list[tuple[str, float]]] = {

    "islamic_finance": [
        # Core Islamic finance terms — very high weight
        ("zakat", 1.0),
        ("zakaat", 1.0),
        ("zakah", 1.0),
        ("riba", 1.0),
        ("sood", 1.0),          # Roman Urdu for riba/interest
        ("halal", 0.9),
        ("haram", 0.9),
        ("murabaha", 1.0),
        ("murabahah", 1.0),
        ("musharakah", 1.0),
        ("musharika", 1.0),     # Roman Urdu variant
        ("ijarah", 1.0),
        ("ijara", 1.0),
        ("diminishing musharakah", 1.0),
        ("sukuk", 1.0),
        ("takaful", 1.0),
        ("waqf", 1.0),
        ("nisab", 1.0),
        ("ushr", 1.0),
        ("sadqah", 0.8),
        ("sadqa", 0.8),
        ("khums", 0.9),
        ("fitrana", 0.9),
        ("fitra", 0.8),
        ("islamic banking", 1.0),
        ("shariah", 0.9),
        ("sharia", 0.9),
        ("fatwa", 0.7),
        ("profit sharing", 0.7),
        ("interest free", 0.9),
        ("سود", 1.0),           # Urdu script: riba/interest
        ("زکوٰۃ", 1.0),         # Urdu: zakat
        ("حلال", 0.9),
        ("حرام", 0.9),
    ],

    "banking_basics": [
        # Account operations and general banking
        ("account", 0.6),
        ("account kholna", 0.9),
        ("account open", 0.9),
        ("current account", 0.9),
        ("savings account", 0.9),
        ("bachat", 0.7),         # Roman Urdu: savings
        ("bank account", 0.9),
        ("cheque", 0.8),
        ("check", 0.5),
        ("passbook", 0.8),
        ("statement", 0.6),
        ("bank statement", 0.9),
        ("kyc", 0.8),
        ("cnic", 0.8),
        ("nadra", 0.7),
        ("branch", 0.5),
        ("atm", 0.7),
        ("debit card", 0.8),
        ("debit", 0.5),
        ("pin", 0.5),
        ("mobile banking", 0.8),
        ("internet banking", 0.8),
        ("online banking", 0.8),
        ("hbl", 0.6),
        ("ubl", 0.6),
        ("nbp", 0.6),
        ("meezan", 0.6),
        ("bank alfalah", 0.6),
        ("mcb", 0.6),
        ("allied bank", 0.6),
        ("بینک", 0.7),           # Urdu: bank
        ("اکاؤنٹ", 0.8),
    ],

    "loans": [
        # Loan and financing products
        ("loan", 0.9),
        ("qarz", 0.9),           # Roman Urdu/Urdu: loan
        ("qarza", 0.9),
        ("financing", 0.7),
        ("finance", 0.5),
        ("credit", 0.6),
        ("borrow", 0.7),
        ("emi", 0.9),
        ("installment", 0.8),
        ("qist", 0.9),           # Roman Urdu: installment
        ("monthly payment", 0.7),
        ("repayment", 0.8),
        ("home loan", 1.0),
        ("car loan", 1.0),
        ("personal loan", 1.0),
        ("business loan", 1.0),
        ("mortgage", 0.9),
        ("collateral", 0.8),
        ("zimanaat", 0.8),       # Roman Urdu: guarantee/collateral
        ("guarantor", 0.7),
        ("credit score", 0.8),
        ("debt", 0.7),
        ("qarz utarna", 0.9),    # Roman Urdu: paying off debt
        ("قرض", 0.9),            # Urdu: loan
        ("قسط", 0.9),
    ],

    "digital_payments": [
        # Mobile wallets and digital transfer
        ("easypaisa", 1.0),
        ("jazzcash", 1.0),
        ("jazz cash", 1.0),
        ("raast", 0.9),
        ("ibft", 0.9),
        ("nift", 0.8),
        ("1link", 0.9),
        ("mobile wallet", 0.9),
        ("digital wallet", 0.9),
        ("transfer", 0.5),
        ("paise transfer", 0.8),   # Roman Urdu
        ("paise bhejein", 0.9),
        ("send money", 0.8),
        ("paise bhejna", 0.9),
        ("receive money", 0.7),
        ("qr code", 0.7),
        ("payment", 0.4),
        ("online payment", 0.7),
        ("bill payment", 0.7),
        ("mobile top up", 0.7),
        ("ایزی پیسہ", 1.0),      # Urdu script
        ("جاز کیش", 1.0),
    ],

    "investment": [
        # Investment and savings products
        ("investment", 0.8),
        ("invest", 0.7),
        ("stock", 0.8),
        ("share", 0.6),
        ("psx", 0.9),
        ("mutual fund", 1.0),
        ("mutual funds", 1.0),
        ("nit", 0.8),              # National Investment Trust
        ("national savings", 0.9),
        ("behbood", 0.9),
        ("pensioner", 0.8),
        ("prize bond", 1.0),
        ("sssc", 0.8),
        ("dssc", 0.8),
        ("treasury bill", 0.9),
        ("t-bill", 0.9),
        ("pib", 0.8),              # Pakistan Investment Bond
        ("term deposit", 0.9),
        ("fixed deposit", 0.9),
        ("return on investment", 0.9),
        ("roi", 0.7),
        ("dividend", 0.8),
        ("capital gain", 0.9),
        ("sarmaya", 0.7),          # Roman Urdu: capital/investment
        ("منافع", 0.7),            # Urdu: profit
    ],

    "insurance": [
        # Conventional and takaful insurance
        ("insurance", 0.9),
        ("bima", 0.9),             # Roman Urdu/Urdu: insurance
        ("takaful", 0.9),
        ("life insurance", 1.0),
        ("health insurance", 1.0),
        ("car insurance", 1.0),
        ("vehicle insurance", 1.0),
        ("claim", 0.7),
        ("premium", 0.6),
        ("policy", 0.5),
        ("coverage", 0.7),
        ("beneficiary", 0.7),
        ("insurer", 0.7),
        ("bima karwana", 0.9),
        ("بیمہ", 0.9),             # Urdu: insurance
    ],

    "tax": [
        # Taxation in Pakistan
        ("tax", 0.8),
        ("income tax", 1.0),
        ("withholding tax", 1.0),
        ("wht", 1.0),
        ("filer", 0.9),
        ("non-filer", 0.9),
        ("nonfiler", 0.9),
        ("ntn", 0.9),              # National Tax Number
        ("fbr", 0.9),              # Federal Board of Revenue
        ("return file", 0.9),
        ("tax return", 1.0),
        ("tax exempt", 0.9),
        ("tax deduction", 0.9),
        ("sales tax", 0.9),
        ("gst", 0.8),
        ("cess", 0.8),
        ("ٹیکس", 0.9),             # Urdu: tax
    ],

    "remittance": [
        # International money transfer
        ("remittance", 1.0),
        ("foreign remittance", 1.0),
        ("send money abroad", 1.0),
        ("overseas", 0.8),
        ("foreign transfer", 0.9),
        ("swift", 0.9),
        ("iban", 0.9),
        ("western union", 1.0),
        ("moneygram", 1.0),
        ("wise", 0.7),
        ("roshan digital account", 1.0),
        ("rda", 0.9),
        ("nrp", 0.8),              # Non-Resident Pakistani
        ("bahar se paise", 0.9),   # Roman Urdu: money from abroad
        ("dollar", 0.5),
        ("exchange rate", 0.8),
        ("forex", 0.8),
        ("ترسیلات", 0.9),          # Urdu: remittances
    ],
}


# ---------------------------------------------------------------------------
# CategoryDetector
# ---------------------------------------------------------------------------

class CategoryDetector:
    """
    Rule-based query-to-category classifier.

    Keyword map is loaded from config (metadata_filter.keyword_map).
    If the config does not define a keyword map, falls back to DEFAULT_KEYWORD_MAP.

    Design:
      - Scans the normalized query for keyword matches (substring + word-boundary)
      - Accumulates weighted scores per category
      - Selects the category with the highest total score
      - Returns None if the winner score < min_confidence (policy handled upstream)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        mf_cfg = cfg.get("metadata_filter", {})

        # Load keyword map — from config if present, else use built-in default
        self._keyword_map = self._load_keyword_map(mf_cfg)

        # Confidence saturation point: a single "perfect" match saturates at this score
        # Queries with multiple matches will exceed 1.0 before clamping → confidence=1.0
        self._saturation_score = 2.0

        # All valid categories (from keyword map keys)
        self.valid_categories: list[str] = sorted(self._keyword_map.keys())

        log.info(
            f"CategoryDetector initialized: {len(self.valid_categories)} categories | "
            f"source={'config' if mf_cfg.get('keyword_map') else 'built-in'}"
        )

    def _load_keyword_map(
        self, mf_cfg: dict
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Load keyword map from config or fall back to DEFAULT_KEYWORD_MAP.

        Config format (retrieval_config.yaml):
          metadata_filter:
            keyword_map:
              islamic_finance:
                - keyword: "zakat"
                  weight: 1.0
              ...
        """
        config_map = mf_cfg.get("keyword_map")
        if not config_map:
            return DEFAULT_KEYWORD_MAP

        # Parse config YAML format into (keyword, weight) tuples
        parsed: dict[str, list[tuple[str, float]]] = {}
        for category, entries in config_map.items():
            if isinstance(entries, list):
                kws = []
                for entry in entries:
                    if isinstance(entry, dict):
                        kws.append((entry["keyword"], float(entry.get("weight", 0.7))))
                    elif isinstance(entry, str):
                        kws.append((entry, 0.7))
                parsed[category] = kws
            else:
                log.warning(f"Skipping malformed keyword_map entry for '{category}'")

        return parsed if parsed else DEFAULT_KEYWORD_MAP

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, query: str) -> CategoryDetectionResult:
        """
        Detect the most likely category for a query.

        Parameters
        ----------
        query : normalized query string (pre-normalized by normalization.py)

        Returns
        -------
        CategoryDetectionResult with predicted_category (may be None),
        confidence [0,1], matched_keywords, and all_scores for diagnostics.

        Matching strategy:
          - Lowercase Latin portion for comparison (Urdu script preserved)
          - Check multi-word keywords before single-word (longest match wins)
          - Word-boundary check prevents "loan" matching "loansome"
        """
        if not query or not query.strip():
            return CategoryDetectionResult(
                predicted_category=None,
                confidence=0.0,
                matched_keywords=[],
                all_scores={},
            )

        query_lower = query.lower().strip()
        category_scores: dict[str, float] = {}
        category_matches: dict[str, list[str]] = {}

        for category, keyword_weights in self._keyword_map.items():
            score = 0.0
            matched: list[str] = []

            # Sort by keyword length descending — match longer phrases first
            sorted_kws = sorted(keyword_weights, key=lambda x: len(x[0]), reverse=True)

            for keyword, weight in sorted_kws:
                kw_lower = keyword.lower()

                # Multi-word keyword: substring match (phrase must appear intact)
                if " " in kw_lower:
                    if kw_lower in query_lower:
                        score += weight
                        matched.append(keyword)
                else:
                    # Single-word keyword: require word boundary
                    # Use \b for Latin, simple containment for Urdu script
                    if self._is_urdu_script(keyword):
                        if keyword in query:
                            score += weight
                            matched.append(keyword)
                    else:
                        pattern = r"(?<![a-zA-Z])" + re.escape(kw_lower) + r"(?![a-zA-Z])"
                        if re.search(pattern, query_lower):
                            score += weight
                            matched.append(keyword)

            category_scores[category] = score
            category_matches[category] = matched

        # Select best category
        if not category_scores or max(category_scores.values()) == 0.0:
            return CategoryDetectionResult(
                predicted_category=None,
                confidence=0.0,
                matched_keywords=[],
                all_scores=category_scores,
            )

        best_category = max(category_scores, key=lambda c: category_scores[c])
        best_score = category_scores[best_category]

        # Normalize to [0, 1] with saturation
        confidence = min(1.0, best_score / self._saturation_score)
        confidence = round(confidence, 4)

        return CategoryDetectionResult(
            predicted_category=best_category,
            confidence=confidence,
            matched_keywords=category_matches[best_category],
            all_scores={k: round(v, 4) for k, v in category_scores.items()},
        )

    @staticmethod
    def _is_urdu_script(text: str) -> bool:
        """Return True if text contains Urdu/Arabic Unicode characters."""
        return any(0x0600 <= ord(c) <= 0x06FF for c in text)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_keywords_for_category(self, category: str) -> list[tuple[str, float]]:
        """Return all keywords registered for a category."""
        return self._keyword_map.get(category, [])

    def explain(self, query: str) -> str:
        """
        Human-readable explanation of detection decision.
        Useful for debugging why a category was or wasn't detected.
        """
        result = self.detect(query)
        lines = [
            f"Query     : '{query}'",
            f"Predicted : {result.predicted_category}",
            f"Confidence: {result.confidence:.4f}",
            f"Matched   : {result.matched_keywords}",
            "",
            "All category scores:",
        ]
        for cat, score in sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 10)
            lines.append(f"  {cat:<25} {score:>6.4f}  {bar}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_detector_cache: dict[int, CategoryDetector] = {}


def get_detector(cfg: Optional[dict] = None) -> CategoryDetector:
    """Return a singleton CategoryDetector for the given config."""
    if cfg is None:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
    cfg_id = id(cfg)
    if cfg_id not in _detector_cache:
        _detector_cache[cfg_id] = CategoryDetector(cfg)
    return _detector_cache[cfg_id]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from typing import Optional

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    detector = CategoryDetector(cfg)

    test_cases = [
        # (query, expected_category)
        ("zakat ka hisab kaise karein",           "islamic_finance"),
        ("meezan bank account kholna hai",         "banking_basics"),
        ("easypaisa se paise bhejein",             "digital_payments"),
        ("home loan ki qist kitni hogi",           "loans"),
        ("how to file income tax return fbr",      "tax"),
        ("prize bond result check karna hai",      "investment"),
        ("roshan digital account overseas",        "remittance"),
        ("life insurance takaful mein farq",       "insurance"),
        ("riba kya hai islam mein",                "islamic_finance"),
        ("مشارکہ اور مرابحہ میں کیا فرق ہے",       "islamic_finance"),
        ("زکوٰۃ کا نصاب کتنا ہے",                  "islamic_finance"),
        ("what is nisab for zakat 2024",            "islamic_finance"),
        ("what time does hbl branch open",          "banking_basics"),
        ("send money to uk from pakistan swift",    "remittance"),
        ("aaj ka mosam",                            None),   # off-domain
        ("",                                        None),   # empty
    ]

    print(f"\n{'Query':<50} {'Expected':<20} {'Got':<20} {'Conf':<6} {'Pass'}")
    print("-" * 110)
    passed = 0
    for query, expected in test_cases:
        result = detector.detect(query)
        ok = result.predicted_category == expected
        if ok:
            passed += 1
        status = "✅" if ok else "❌"
        print(
            f"{query[:49]:<50} {str(expected):<20} "
            f"{str(result.predicted_category):<20} "
            f"{result.confidence:<6.3f} {status}"
        )
        if not ok:
            print(f"   matched: {result.matched_keywords}")

    print(f"\nResult: {passed}/{len(test_cases)} passed")
    print("\n--- Explain example ---")
    print(detector.explain("zakat ka hisab aur nisab kya hai meezan bank mein"))