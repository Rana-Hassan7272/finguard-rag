"""
retrieval/language.py
======================
Runtime language detector for query routing in the hybrid search pipeline.

Detects: "urdu" | "english" | "roman_urdu" | "unknown"
Exposes: get_fusion_weights() — returns (bm25_weight, vector_weight) from config

Design principles:
  - Zero external dependencies (no langdetect, no heavy models)
  - Deterministic — same input always returns same label
  - Fast — runs in microseconds, not milliseconds
  - Transparent — logic is readable and adjustable

Language heuristics:
  - Urdu script Unicode range: U+0600–U+06FF (Arabic script block)
    Urdu uses this range. If >15% of chars are in this range → "urdu"
  - English: if >50% of words are in an English vocabulary set → "english"
  - Otherwise: "roman_urdu" (default for Pakistani financial queries)

Usage:
    from retrieval.language import detect_language, get_fusion_weights

    label, confidence = detect_language("zakat ka hisab kaise karein")
    # → ("roman_urdu", 0.85)

    bm25_w, vec_w = get_fusion_weights(label, cfg)
    # → (0.30, 0.70)
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

CONFIG_PATH = Path("retrieval/configs/retrieval_config.yaml")

# ---------------------------------------------------------------------------
# Arabic/Urdu Unicode ranges
# ---------------------------------------------------------------------------
# U+0600–U+06FF: Arabic block (Urdu script lives here)
# U+FB50–U+FDFF: Arabic Presentation Forms-A (some Urdu extended chars)
# U+FE70–U+FEFF: Arabic Presentation Forms-B
URDU_SCRIPT_RANGES = [
    (0x0600, 0x06FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]


def is_urdu_char(char: str) -> bool:
    """Return True if character is in Urdu/Arabic Unicode range."""
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in URDU_SCRIPT_RANGES)


def urdu_char_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are Urdu script."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    urdu_chars = sum(1 for c in non_space if is_urdu_char(c))
    return urdu_chars / len(non_space)


# ---------------------------------------------------------------------------
# English vocabulary heuristic
# ---------------------------------------------------------------------------
# Core English function + finance words — if enough of these appear,
# it's likely an English query. Intentionally small set.
ENGLISH_WORD_SET = {
    # Finance terms
    "how", "what", "is", "are", "the", "a", "an", "to", "of", "in",
    "for", "can", "my", "me", "calculate", "open", "account", "bank",
    "loan", "zakat", "saving", "investment", "insurance", "tax",
    "income", "return", "profit", "interest", "pay", "payment",
    "card", "credit", "debit", "transfer", "pakistan", "rupee",
    "apply", "get", "need", "want", "much", "many", "when", "where",
    "which", "who", "difference", "between", "and", "or", "with",
    "from", "do", "does", "will", "should", "charge", "rate", "limit",
    "minimum", "maximum", "required", "eligible", "process", "steps",
}


def english_word_ratio(tokens: list[str]) -> float:
    """Fraction of tokens that are common English words."""
    if not tokens:
        return 0.0
    english_count = sum(1 for t in tokens if t.lower() in ENGLISH_WORD_SET)
    return english_count / len(tokens)


# ---------------------------------------------------------------------------
# Roman Urdu signal words
# These appear frequently in Roman Urdu financial queries
# ---------------------------------------------------------------------------
ROMAN_URDU_SIGNALS = {
    "ka", "ki", "ke", "hai", "hain", "kya", "kaise", "kab", "kahan",
    "kitna", "kitni", "mein", "se", "pe", "par", "ko", "ne", "wala",
    "wali", "hota", "hoti", "hote", "karna", "karein", "karo",
    "karein", "chahiye", "milta", "milti", "dena", "lena",
    "zaroorat", "faida", "nuksan", "sood", "halal", "haram",
    "zakat", "riba", "murabaha", "musharakah", "ijarah",
    "easypaisa", "jazzcash", "raast", "meezan", "hbl", "ubl", "nbp",
    "naya", "purana", "pehle", "baad", "abhi", "sirf",
}


def roman_urdu_signal_ratio(tokens: list[str]) -> float:
    """Fraction of tokens that are Roman Urdu signal words."""
    if not tokens:
        return 0.0
    signal_count = sum(1 for t in tokens if t.lower() in ROMAN_URDU_SIGNALS)
    return signal_count / len(tokens)


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
URDU_SCRIPT_THRESHOLD = 0.15      # >15% Urdu chars → "urdu"
ENGLISH_WORD_THRESHOLD = 0.40     # >40% English words → "english"
ROMAN_URDU_SIGNAL_THRESHOLD = 0.10  # >10% Roman Urdu signals → "roman_urdu"


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------
@dataclass
class LanguageResult:
    label: str          # "urdu" | "english" | "roman_urdu" | "unknown"
    confidence: float   # 0.0–1.0 (rough estimate, not calibrated probability)
    urdu_ratio: float
    english_ratio: float
    roman_urdu_ratio: float


def detect_language(text: str) -> LanguageResult:
    """
    Detect language of a query string.

    Returns a LanguageResult with label and confidence.

    Decision logic (priority order):
      1. Urdu script chars > threshold → "urdu"      (script is unambiguous)
      2. English word ratio > threshold → "english"
      3. Roman Urdu signal ratio > threshold → "roman_urdu"
      4. Short query (<3 tokens) → "roman_urdu"      (safe default for this domain)
      5. Fallback → "unknown"
    """
    if not text or not text.strip():
        return LanguageResult("unknown", 0.0, 0.0, 0.0, 0.0)

    text_clean = text.strip()
    tokens = text_clean.lower().split()

    urdu_ratio = urdu_char_ratio(text_clean)
    eng_ratio = english_word_ratio(tokens)
    ru_ratio = roman_urdu_signal_ratio(tokens)

    # Rule 1: Urdu script — highest priority (script is deterministic)
    if urdu_ratio >= URDU_SCRIPT_THRESHOLD:
        confidence = min(1.0, urdu_ratio * 4)   # scale 0.15→0.6 to 0.6→1.0
        return LanguageResult("urdu", round(confidence, 2), urdu_ratio, eng_ratio, ru_ratio)

    # Rule 2: English — enough English vocabulary
    if eng_ratio >= ENGLISH_WORD_THRESHOLD:
        confidence = min(1.0, eng_ratio * 1.5)
        return LanguageResult("english", round(confidence, 2), urdu_ratio, eng_ratio, ru_ratio)

    # Rule 3: Roman Urdu signal words present
    if ru_ratio >= ROMAN_URDU_SIGNAL_THRESHOLD:
        confidence = min(1.0, ru_ratio * 4)
        return LanguageResult("roman_urdu", round(confidence, 2), urdu_ratio, eng_ratio, ru_ratio)

    # Rule 4: Short query in Latin script → default to roman_urdu for this domain
    # Single-word queries like "zakat", "riba" are almost always Roman Urdu domain terms
    if len(tokens) <= 3 and urdu_ratio == 0.0:
        return LanguageResult("roman_urdu", 0.5, urdu_ratio, eng_ratio, ru_ratio)

    # Rule 5: Fallback
    return LanguageResult("unknown", 0.3, urdu_ratio, eng_ratio, ru_ratio)


def detect_language_label(text: str) -> str:
    """
    Simplified interface — return label string only.
    Use this when you only need the label for weight lookup.
    """
    return detect_language(text).label


# ---------------------------------------------------------------------------
# Fusion weight lookup
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_fusion_weights(
    language_label: str,
    cfg: Optional[dict] = None,
) -> tuple[float, float]:
    """
    Return (bm25_weight, vector_weight) for the detected language.

    Pulls from language_weights table in retrieval_config.yaml.
    Falls back to "unknown" weights if label not found in config.

    Parameters
    ----------
    language_label : one of "urdu", "english", "roman_urdu", "unknown"
    cfg            : config dict (loaded if None)

    Returns
    -------
    (bm25_weight, vector_weight) — both floats, sum to 1.0
    """
    if cfg is None:
        cfg = load_config()

    weights_table = cfg["fusion"]["language_weights"]

    # Use label from config, fall back to "unknown"
    weights = weights_table.get(language_label, weights_table.get("unknown", {}))

    bm25_w = weights.get("bm25_weight", 0.40)
    vec_w = weights.get("vector_weight", 0.60)

    return bm25_w, vec_w


# ---------------------------------------------------------------------------
# Convenience: detect + get weights in one call
# ---------------------------------------------------------------------------

def detect_and_get_weights(
    text: str,
    cfg: Optional[dict] = None,
) -> tuple[LanguageResult, float, float]:
    """
    Detect language and immediately return fusion weights.

    Returns (language_result, bm25_weight, vector_weight)

    Typical usage in hybrid retriever:
        lang_result, bm25_w, vec_w = detect_and_get_weights(query, cfg)
        log_event("lang_detected", lang_result.label, lang_result.confidence)
    """
    result = detect_language(text)
    bm25_w, vec_w = get_fusion_weights(result.label, cfg)
    return result, bm25_w, vec_w


# ---------------------------------------------------------------------------
# Quick test / standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_queries = [
        # Roman Urdu
        ("zakat ka hisab kaise karein", "roman_urdu"),
        ("easypaisa se paise kaise bhejein", "roman_urdu"),
        ("meezan bank ka account kholna hai", "roman_urdu"),
        ("loan ki qist kitni hogi", "roman_urdu"),
        # Urdu script
        ("زکوٰۃ کا حساب کیسے کریں", "urdu"),
        ("مشارکہ اور مرابحہ میں کیا فرق ہے", "urdu"),
        # English
        ("how to calculate zakat in pakistan", "english"),
        ("what is the difference between murabaha and ijarah", "english"),
        ("can I open an account at meezan bank online", "english"),
        # Edge cases
        ("zakat", "roman_urdu"),
        ("riba", "roman_urdu"),
        ("", "unknown"),
    ]

    print(f"\n{'Query':<45} {'Expected':<12} {'Got':<12} {'Conf':<6} {'Pass'}")
    print("-" * 90)

    passed = 0
    for query, expected in test_queries:
        result = detect_language(query)
        bm25_w, vec_w = get_fusion_weights(result.label)
        ok = "✅" if result.label == expected else "❌"
        if result.label == expected:
            passed += 1
        print(
            f"{query[:44]:<45} {expected:<12} {result.label:<12} "
            f"{result.confidence:<6.2f} {ok}  "
            f"(bm25={bm25_w}, vec={vec_w})"
        )

    print(f"\nResult: {passed}/{len(test_queries)} passed")